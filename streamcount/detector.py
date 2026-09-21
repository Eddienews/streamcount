"""Local ONNX object detector (zero API cost) with tiled inference.

The bundled default model is a YOLOv8-pose ONNX export (person only). The same
code path also handles YOLOv8 *detection* exports (COCO 80 classes) so vehicles
can be counted with ``--classes car,truck,bus,motorcycle`` and a detection model
pointed at with ``--model``.

Why tiles: at 640x640 a nano model misses distant/small pedestrians at night
(measured: 3 of ~12 people in a 1080p night street scene). Splitting the frame
into an NxN grid with overlap and running the model per tile raises recall to
10-13 of ~12 (conf 0.25, tiles 2-3), at the cost of ~N^2 inference passes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# COCO-80 names (used by detection exports; pose exports are person-only)
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

MODEL_URLS = {
    "yolov8n-pose": (
        "https://huggingface.co/Xenova/yolov8n-pose/resolve/main/onnx/model.onnx",
        13_484_153,
    ),
    "yolov8s-pose": (
        "https://huggingface.co/Xenova/yolov8s-pose/resolve/main/onnx/model.onnx",
        46_787_284,
    ),
}

Detection = tuple[float, float, float, float, float, str]  # x1, y1, x2, y2, score, label


def cache_dir() -> Path:
    import os

    root = os.environ.get("STREAMCOUNT_HOME")
    base = Path(root) if root else Path.home() / ".cache" / "streamcount"
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_model(variant: str = "yolov8n-pose") -> Path:
    """Download (once) and return the path of a bundled variant."""
    if variant not in MODEL_URLS:
        raise SystemExit(f"unknown model variant {variant!r}; options: {', '.join(MODEL_URLS)}")
    url, expected = MODEL_URLS[variant]
    dest = cache_dir() / "models" / f"{variant}.onnx"
    if dest.exists() and dest.stat().st_size == expected:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {variant} ({expected / 1e6:.1f} MB) -> {dest}")
    # trust_env=False avoids stale netrc/proxy credentials that break HF downloads
    session = requests.Session()
    session.trust_env = False
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


class Detector:
    def __init__(
        self,
        model_path: Path | str,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        tiles: int = 1,
        tile_overlap: float = 0.15,
        classes: tuple[str, ...] = ("person",),
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.fixed_input = isinstance(shape[2], int) and isinstance(shape[3], int)
        if self.fixed_input:
            imgsz = int(shape[2])
        self.imgsz = imgsz
        self.conf = conf
        self.iou_thr = iou
        self.tiles = max(1, tiles)
        self.tile_overlap = tile_overlap

        # (1, 4+nc, 8400) detection export vs (1, 56, 8400) pose export
        channels = self.session.get_outputs()[0].shape[1]
        self.is_pose = channels == 56
        self.num_classes = 1 if self.is_pose else int(channels) - 4
        names = ["person"] if self.is_pose else COCO80[: self.num_classes]
        self.wanted = {i for i, n in enumerate(names) if n in classes} or {0}
        self.names = names

    # ------------------------------------------------------------------ tiling
    def detect(self, image: Image.Image) -> list[Detection]:
        if self.tiles <= 1:
            return self._detect_single(image)
        w, h = image.size
        n = self.tiles
        tw, th = w / n, h / n
        pad_x, pad_y = tw * self.tile_overlap, th * self.tile_overlap
        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        labels: list[str] = []
        for iy in range(n):
            for ix in range(n):
                x0, y0 = max(0.0, ix * tw - pad_x), max(0.0, iy * th - pad_y)
                x1, y1 = min(float(w), (ix + 1) * tw + pad_x), min(float(h), (iy + 1) * th + pad_y)
                tile = image.crop((int(x0), int(y0), int(x1), int(y1)))
                for dx1, dy1, dx2, dy2, score, label in self._detect_single(tile):
                    boxes.append((dx1 + x0, dy1 + y0, dx2 + x0, dy2 + y0))
                    scores.append(score)
                    labels.append(label)
        if not boxes:
            return []
        merged = self._merge_stacked(list(zip(boxes, scores, labels, strict=True)))
        return merged

    @staticmethod
    def _compute_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
        x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / np.maximum(area_a + area_b - inter, 1e-9)

    @classmethod
    def _nms(cls, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = cls._compute_iou(boxes[i], boxes[rest])
            order = rest[ious <= iou_thr]
        return keep

    @staticmethod
    def _merge_stacked(items: list[tuple[tuple[float, float, float, float], float, str]],
                       min_x_overlap: float = 0.85, max_gap_frac: float = 0.12,
                       max_h_ratio: float = 1.6
                       ) -> list[tuple[float, float, float, float, float, str]]:
        """Merge torso+legs splits of the SAME person (typical in tiled inference).

        Strict criteria so two different people aligned in the image are not merged:
        x-overlap >= 85%, vertical gap <= 12% of the smaller height, height ratio <= 1.6.
        """
        ordered = sorted(items, key=lambda it: (it[0][3] - it[0][1]) * (it[0][2] - it[0][0]),
                         reverse=True)
        out: list[tuple[tuple[float, float, float, float], float, str]] = []
        for box, score, label in ordered:
            for i, (o_box, o_score, o_label) in enumerate(out):
                if label != o_label:
                    continue
                x_overlap = min(box[2], o_box[2]) - max(box[0], o_box[0])
                x_union = max(box[2], o_box[2]) - min(box[0], o_box[0])
                if x_union <= 0 or x_overlap / x_union < min_x_overlap:
                    continue
                gap = max(box[1], o_box[1]) - min(box[3], o_box[3])
                h_a, h_b = box[3] - box[1], o_box[3] - o_box[1]
                h_min, h_max = min(h_a, h_b), max(h_a, h_b)
                if gap <= max_gap_frac * h_min and h_max / max(h_min, 1e-6) <= max_h_ratio:
                    out[i] = (
                        (min(box[0], o_box[0]), min(box[1], o_box[1]),
                         max(box[2], o_box[2]), max(box[3], o_box[3])),
                        max(score, o_score),
                        label,
                    )
                    break
            else:
                out.append((box, score, label))
        return [(b[0], b[1], b[2], b[3], s, lb) for b, s, lb in out]

    def _detect_single(self, image: Image.Image) -> list[Detection]:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        dw, dh = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[dh: dh + nh, dw: dw + nw] = np.asarray(
            Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
        )
        tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        pred = self.session.run(None, {self.input_name: tensor})[0][0].T  # (8400, 4+nc|56)

        if self.is_pose:
            conf = pred[:, 4]
            sel = pred[conf >= self.conf]
            if sel.size == 0:
                return []
            labels = ["person"] * len(sel)
        else:
            class_scores = pred[:, 4: 4 + self.num_classes]
            class_ids = class_scores.argmax(axis=1)
            conf = class_scores[np.arange(len(pred)), class_ids]
            mask = (conf >= self.conf) & np.isin(class_ids, list(self.wanted))
            sel = pred[mask]
            class_ids = class_ids[mask]
            labels = [self.names[int(c)] for c in class_ids]
            if sel.size == 0:
                return []

        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        scores = sel[:, 4] if self.is_pose else conf
        keep = self._nms(boxes, np.asarray(scores, dtype=np.float32), self.iou_thr)
        return [
            (float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]),
             float(scores[i]), labels[i])
            for i in keep
        ]


class DetectorStats:
    """Small helper to time detection passes in logs."""

    def __init__(self) -> None:
        self.last_ms = 0

    def time(self, fn, *args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        self.last_ms = int((time.time() - t0) * 1000)
        return result
