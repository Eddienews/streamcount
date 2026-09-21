"""Run pipeline: frames in -> counts out (CSV + events + annotated frames + summary)."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from .detector import VEHICLE_CLASSES, Detector, resolve_classes
from .sources import (
    frames_from_ffmpeg,
    frames_from_images,
    parse_headers,
    resolve_stream_url,
)
from .tracking import FlowTracker
from .vlm import VlmCounter, resolve_api_key


@dataclass
class RunConfig:
    # source (exactly one of these)
    url: str | None = None
    video: Path | None = None
    image: Path | None = None
    images_dir: Path | None = None
    yt_dlp_js_runtime: str | None = None
    headers: str = ""
    seek_s: float = 0.0

    # sampling
    interval: float = 2.0
    frames: int = 0            # 0 = auto (10 for streams, all for images)
    timeline: str = "auto"     # auto | wall | video

    # engines
    engine: str = "yolo"       # yolo | vlm | both
    target: str = "people"     # people | cars | people,cars
    model: Path | None = None
    conf: float = 0.25
    iou: float = 0.5
    tiles: int = 1
    tile_overlap: float = 0.15
    vlm_key: str | None = None
    vlm_base_url: str = "https://openrouter.ai/api/v1"
    vlm_model: str = "google/gemini-2.5-flash-lite"

    # flow mode
    flow: bool = False
    flow_min_hits: int = 3
    flow_min_move: float = 40.0
    flow_max_age: float = 5.0
    vlm_check: float = 0.0     # seconds between recall checks (0 = off)

    # output
    out_dir: Path = Path("runs")
    annotate: bool = False
    tag: str = ""


@dataclass
class RunResult:
    exit_code: int = 0
    run_dir: Path | None = None
    csv_path: Path | None = None
    events_path: Path | None = None
    summary_path: Path | None = None
    total_frames: int = 0
    passes: int | None = None
    summary: dict = field(default_factory=dict)


def _log(message: str = "") -> None:
    print(message, flush=True)


def _annotate(image: Image.Image, boxes, count: int, engine: str, ids=None) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, box in enumerate(boxes or []):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
        if ids:
            label = f"id{ids[index]}"
        elif len(box) > 4:
            label = f"{box[5] if len(box) > 5 else 'obj'} {box[4]:.2f}"
        else:
            label = "obj"
        draw.text((x1 + 3, max(2, y1 - 13)), label, fill=(255, 60, 60))
    stamp = f"{engine} | count: {count} | {datetime.now():%H:%M:%S}"
    draw.rectangle([0, 0, 8 + 7 * len(stamp), 20], fill=(0, 0, 0))
    draw.text((4, 5), stamp, fill=(255, 210, 60))
    return canvas


def _default_timeline(source: str) -> str:
    lowered = source.lower()
    if ".m3u8" in lowered or lowered.startswith(("rtsp://", "rtmp://")):
        return "wall"
    return "video"


def run(config: RunConfig) -> RunResult:
    result = RunResult()
    headers = parse_headers(config.headers)

    # ------------------------------------------------------------------ source
    if config.image or config.images_dir:
        path = config.image or config.images_dir
        frames = frames_from_images(path)
        source = str(path)
        total_planned = config.frames or len(frames)
    elif config.video:
        source = str(config.video)
        total_planned = config.frames or 10**9
        frames = None
    elif config.url:
        source = resolve_stream_url(config.url, js_runtime=config.yt_dlp_js_runtime)
        total_planned = config.frames or 10
        frames = None
    else:
        raise SystemExit("no source given: use --url, --youtube, --video, --image or --images")

    timeline_mode = config.timeline
    if timeline_mode == "auto":
        timeline_mode = "video" if (config.image or config.images_dir) else _default_timeline(source)

    # ----------------------------------------------------------------- engines
    detector = None
    vlm = None
    if config.engine in ("yolo", "both") or (config.flow and config.vlm_check > 0):
        from .detector import ensure_model

        model_path = config.model or ensure_model("yolov8n-pose")
        classes = resolve_classes(config.target)
        detector = Detector(
            model_path,
            conf=config.conf,
            iou=config.iou,
            tiles=config.tiles,
            tile_overlap=config.tile_overlap,
            classes=classes,
        )
    vlm_needed = config.engine in ("vlm", "both") or (config.flow and config.vlm_check > 0)
    if vlm_needed:
        key, origin = resolve_api_key(config.vlm_key)
        if not key:
            raise SystemExit(
                "VLM engine needs an API key: --vlm-key, STREAMCOUNT_VLM_KEY or "
                "OPENROUTER_API_KEY (env or .env file)"
            )
        resolved = set(resolve_classes(config.target))
        vlm_target = "cars" if resolved & set(VEHICLE_CLASSES) else "people"
        vlm = VlmCounter(key, base_url=config.vlm_base_url, model=config.vlm_model,
                         target=vlm_target)
        _log(f"[info] VLM: {config.vlm_model} @ {config.vlm_base_url} (key from {origin})")

    if config.flow and detector is None:
        raise SystemExit("--flow needs the local detector (--engine yolo or both)")

    # ------------------------------------------------------------------ outputs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{config.engine}" + (f"_{config.tag}" if config.tag else "")
    run_dir = Path(config.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "frames.csv"
    events_path = run_dir / "events.csv" if config.flow else None
    annotate_dir = run_dir / "frames" if config.annotate else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    tracker = (
        FlowTracker(
            min_hits=config.flow_min_hits,
            min_move_px=config.flow_min_move,
            max_age_s=config.flow_max_age,
        )
        if config.flow
        else None
    )

    _log(f"[run] {run_dir.name} | source: {source[:110]}")
    header = f"{'frame':>5} {'engine':<5} {'count':>6} {'ms':>6}"
    _log(header + ("  tracks/passes" if tracker else "  notes"))

    counts_by_engine: dict[str, list[int]] = {}
    vlm_check_counts: list[int] = []
    t_rel = 0.0
    t0 = time.monotonic()
    frames_done = 0

    source_frames: Iterator[tuple[int, Image.Image]]
    if frames is None:
        source_frames = frames_from_ffmpeg(
            source, config.interval, headers=headers, max_frames=total_planned,
            seek_s=config.seek_s,
        )
    else:
        source_frames = iter(frames)

    events_handle = (
        open(events_path, "w", newline="", encoding="utf-8")  # noqa: SIM115 (closed below)
        if events_path
        else None
    )
    if events_handle:
        csv.writer(events_handle).writerow(
            ["ts_iso", "t_rel_s", "event", "id", "hits", "disp_px", "total"]
        )

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as frames_csv:
            writer = csv.writer(frames_csv)
            writer.writerow(
                ["ts_iso", "frame", "engine", "count", "latency_ms", "source",
                 "active_tracks", "passes_total", "vlm_count", "notes"]
            )
            for index, image in source_frames:
                if index > total_planned:
                    break
                frames_done = index
                t_rel = (time.monotonic() - t0) if timeline_mode == "wall" else (index - 1) * config.interval

                if tracker is not None:
                    t_detect = time.time()
                    detections = detector.detect(image)  # type: ignore[union-attr]
                    ms = int((time.time() - t_detect) * 1000)
                    boxes = [(d[0], d[1], d[2], d[3]) for d in detections]
                    active, events = tracker.update(t_rel, boxes)
                    for event in events:
                        _log(f"   >>> PASS id{event['id']} t={event['t']}s "
                             f"(hits={event['hits']}, disp={event['disp']}px) -> total={event['n']}")
                        csv.writer(events_handle).writerow(
                            [datetime.now().isoformat(timespec="seconds"), event["t"], "pass",
                             event["id"], event["hits"], event["disp"], event["n"]]
                        )
                    events_handle.flush()  # type: ignore[union-attr]
                    counts_by_engine.setdefault("yolo", []).append(len(detections))

                    vlm_count: int | str = ""
                    if vlm is not None and config.vlm_check > 0 and \
                            (t_rel % config.vlm_check) < config.interval:
                        t_vlm = time.time()
                        answer = vlm.count(image)
                        vlm_count = answer["count"]
                        vlm_check_counts.append(answer["count"])
                        _log(f"{index:>5} {'vlm':<5} {answer['count']:>6} "
                             f"{int((time.time() - t_vlm) * 1000):>6}  (recall check)")

                    _log(f"{index:>5} {'yolo':<5} {len(detections):>6} {ms:>6}  "
                         f"active={len(active)} passes={tracker.total}")
                    writer.writerow(
                        [datetime.now().isoformat(timespec="seconds"), index, "yolo",
                         len(detections), ms, source, len(active), tracker.total, vlm_count, ""]
                    )
                    frames_csv.flush()
                    if annotate_dir:
                        flow_boxes = [(tr["box"][0], tr["box"][1], tr["box"][2], tr["box"][3])
                                      for tr in active]
                        _annotate(image, flow_boxes, len(active),
                                  f"flow passes={tracker.total}",
                                  ids=[tr["id"] for tr in active]).save(
                            annotate_dir / f"f{index:04d}_flow.jpg", quality=88)
                    continue

                for name in (["yolo"] if detector else []) + (["vlm"] if vlm else []):
                    t_engine = time.time()
                    notes = ""
                    if name == "yolo":
                        detections = detector.detect(image)  # type: ignore[union-attr]
                        boxes = [(d[0], d[1], d[2], d[3]) for d in detections]
                        count = len(detections)
                    else:
                        answer = vlm.count(image)  # type: ignore[union-attr]
                        boxes, count = [], answer["count"]
                        notes = answer["notes"] + (" [uncertain]" if answer["uncertain"] else "")
                    ms = int((time.time() - t_engine) * 1000)
                    counts_by_engine.setdefault(name, []).append(count)
                    _log(f"{index:>5} {name:<5} {count:>6} {ms:>6}  {notes}")
                    writer.writerow(
                        [datetime.now().isoformat(timespec="seconds"), index, name, count, ms,
                         source, "", "", "", notes]
                    )
                    frames_csv.flush()
                    if annotate_dir:
                        _annotate(image, boxes, count, name).save(
                            annotate_dir / f"f{index:04d}_{name}.jpg", quality=88)
    finally:
        if events_handle:
            events_handle.close()

    # ------------------------------------------------------------------ summary
    _log("\n--- summary ---")
    summary: dict = {
        "source": source,
        "engine": config.engine,
        "target": config.target,
        "interval_s": config.interval,
        "frames": frames_done,
        "run_dir": str(run_dir),
    }
    if tracker is not None:
        tracker.finalize(t_rel)
        duration_s = max(config.interval, t_rel)
        summary.update(
            flow=True,
            passes=tracker.total,
            duration_s=round(duration_s, 1),
            passes_per_min=round(tracker.total / (duration_s / 60), 2) if duration_s else 0,
            discarded_short_tracks=tracker.n_short,
            static_confirmed=tracker.n_static,
        )
        _log(f"FLOW: {tracker.total} passers-by in {duration_s / 60:.1f} min "
             f"({tracker.total / max(duration_s / 60, 1e-9):.1f}/min)")
        _log(f"  discarded: {tracker.n_short} short tracks (noise) | "
             f"{tracker.n_static} confirmed but stationary")
        if vlm_check_counts:
            mean_vlm = sum(vlm_check_counts) / len(vlm_check_counts)
            yolo_counts = counts_by_engine.get("yolo", [0])
            mean_yolo = sum(yolo_counts) / max(len(yolo_counts), 1) or 1
            recall = max(0.05, min(1.0, mean_yolo / mean_vlm))
            corrected = round(tracker.total / recall)
            summary.update(vlm_check_mean=round(mean_vlm, 1),
                           detector_mean_visible=round(mean_yolo, 1),
                           detector_recall=round(recall, 2),
                           passes_recall_corrected=corrected)
            _log(f"  VLM recall check: yolo visible={mean_yolo:.1f} vs vlm={mean_vlm:.1f} "
                 f"-> recall~{recall * 100:.0f}%")
            _log(f"  recall-corrected passes ~ {corrected} (floor: {tracker.total})")

    for name, counts in counts_by_engine.items():
        arr = [c for c in counts if c >= 0]
        if arr:
            _log(f"{name:<5} frames={len(counts)} min={min(arr)} max={max(arr)} "
                 f"avg={sum(arr) / len(arr):.1f} (per-frame count is instantaneous)")
        else:
            _log(f"{name:<5} no valid counts")

    if tracker is not None and events_path:
        from .report import draw_chart, load_events, passes_per_minute

        per_minute = passes_per_minute(load_events(run_dir), duration_s=t_rel)
        chart = draw_chart(per_minute, run_dir / "chart.png")
        summary["chart"] = str(chart)
        summary["passes_per_minute"] = per_minute
        _log(f"chart: {chart}")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"CSV: {csv_path}")
    if events_path:
        _log(f"events: {events_path}")
    _log(f"summary: {summary_path}")

    result.run_dir = run_dir
    result.csv_path = csv_path
    result.events_path = events_path
    result.summary_path = summary_path
    result.total_frames = frames_done
    result.passes = tracker.total if tracker else None
    result.summary = summary
    return result
