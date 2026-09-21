# Third-party notices

streamcount itself is MIT (see LICENSE). It **does not bundle** detection model weights.

## Models downloaded on demand

| Variant | Source | License of the weights |
|---|---|---|
| `yolov8n-pose` (default) | Hugging Face `Xenova/yolov8n-pose` (ONNX export) | Ultralytics YOLOv8 — **AGPL-3.0** |
| `yolov8s-pose` | Hugging Face `Xenova/yolov8s-pose` (ONNX export) | Ultralytics YOLOv8 — **AGPL-3.0** |
| `yolo11n` (COCO 80 classes, people + vehicles) | Hugging Face `webnn/yolo11n` (ONNX export) | Ultralytics YOLO11 — **AGPL-3.0** |

The AGPL-3.0 of the *weights* is separate from this repository's MIT license. If you use
streamcount commercially, either comply with AGPL-3.0 or supply your own weights via
`--model path/to/model.onnx` (YOLOv8 detection exports with COCO classes are also supported).

## Runtime dependencies

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — MIT
- [NumPy](https://numpy.org) — BSD-3-Clause
- [Pillow](https://python-pillow.org) — HPND
- [Requests](https://requests.readthedocs.io) — Apache-2.0
- [ffmpeg](https://ffmpeg.org) — LGPL/GPL (invoked as an external binary, not linked)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — Unlicense (optional, only for YouTube sources)
