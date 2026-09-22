"""Run pipeline: frames in -> counts out (CSV + events + annotated frames + summary).

Optionally writes an annotated timelapse MP4 of the run as it goes (`--timelapse`).
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from .detector import VEHICLE_CLASSES, Detector, resolve_classes
from .router import JevRouter, RouterMetrics, describe_frame, resolve_jev_key, scene_change
from .sources import (
    default_headers,
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
    duration: float = 0.0      # minutes; the run closes itself cleanly at the limit (0 = off)
    jev_router: bool = False   # let Jev gate each recall check (needs a TypeSafe key)
    jev_key: str | None = None
    jev_model: str = "jev-latest"
    jev_threshold: float = 0.5

    # output
    out_dir: Path = Path("runs")
    annotate: bool = False
    keep_frames: int = 0       # keep only the last N annotated frames on disk (0 = keep all)
    timelapse: bool = False    # write an annotated timelapse MP4 of the run (run_dir/timelapse.mp4)
    timelapse_fps: float = 12.0
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


_FRAME_NAME_RE = re.compile(r"^f(\d+)_")


def frame_index(path: Path) -> int | None:
    """Numeric index of an annotated frame (`f0007_flow.jpg` -> 7); None if not ours."""
    match = _FRAME_NAME_RE.match(path.name)
    return int(match.group(1)) if match else None


def prune_frames(directory: Path, keep: int) -> int:
    """Delete all but the last `keep` annotated frames (by frame index); return files deleted.

    Frame numbers are zero-padded, but a plain name sort breaks at f10000 (it sorts *before*
    f9999), so retention compares the parsed integer index. Files we did not name are left
    alone, and a file a reader still holds open is retried on the next frame.
    """
    if keep <= 0:
        return 0
    by_index: dict[int, list[Path]] = {}
    for path in directory.iterdir():
        index = frame_index(path)
        if index is not None:
            by_index.setdefault(index, []).append(path)
    kept = set(sorted(by_index)[-keep:])
    removed = 0
    for index, paths in by_index.items():
        if index in kept:
            continue
        for path in paths:
            try:
                path.unlink()
                removed += 1
            except OSError:  # e.g. the dashboard is reading it right now
                pass
    return removed


class _Recorder:
    """Writes annotated frames into an MP4 while a run is going (ffmpeg image2pipe).

    Lazy: ffmpeg starts on the first frame, so a run that gets none leaves no broken file.
    A recorder that dies mid-run disables itself instead of taking the counting run down.
    """

    def __init__(self, path: Path, fps: float) -> None:
        self.path = Path(path)
        self.fps = fps
        self.frames = 0
        self._proc: subprocess.Popen | None = None
        self._buffer = io.BytesIO()
        self._dead = False

    def _start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{self.fps:g}", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise SystemExit("--timelapse needs ffmpeg on PATH") from exc

    def write(self, image: Image.Image) -> None:
        if self._dead:
            return
        if self._proc is None:
            self._start()
        self._buffer.seek(0)
        self._buffer.truncate()
        image.convert("RGB").save(self._buffer, "JPEG", quality=85)
        try:
            self._proc.stdin.write(self._buffer.getvalue())  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
        except OSError:  # encoder died (bad path, disk): warn once, keep counting
            _log(f"[warn] timelapse encoder died: {self._finish(timeout=10)[1] or 'no stderr'}")
            self._dead = True
            return
        self.frames += 1

    def _finish(self, timeout: float) -> tuple[int | None, str]:
        """Close stdin and reap the encoder; returns (returncode, stderr tail).

        ``proc.stdin`` is detached before ``communicate()``: on Linux, communicate
        flushes stdin and raises ``ValueError: flush of closed file`` once the pipe
        is closed (Windows tolerates it — the first CI run failed only on Linux).
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return None, ""
        stdin, proc.stdin = proc.stdin, None
        if stdin is not None:
            with contextlib.suppress(OSError):
                stdin.close()
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return None, "timed out"
        except OSError as exc:  # the child was already gone; returncode is set
            return proc.returncode, str(exc)
        return proc.returncode, (err or b"").decode("utf-8", "replace").strip()[:200]

    def close(self) -> None:
        if self._proc is None:
            return
        code, err = self._finish(timeout=120)
        if code not in (0, None):  # None = killed after a timeout, err carries the story
            _log(f"[warn] timelapse encoder exited {code}: {err}")
        elif err == "timed out":
            _log("[warn] timelapse encoder did not exit; killed")


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

    # webcam CDNs that gate on headers: add the one they require when the user gave none
    resolved = default_headers(source, headers)
    if resolved != headers:
        _log("[info] EarthCam source: Referer header added automatically (--headers overrides)")
    headers = resolved

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

    router = None
    if config.jev_router:
        if not (config.flow and config.vlm_check > 0):
            raise SystemExit(
                "--jev-router gates the --vlm-check anchor: run with --flow --vlm-check N"
            )
        jev_key, jev_origin = resolve_jev_key(config.jev_key)
        if not jev_key:
            raise SystemExit(
                "--jev-router needs a TypeSafe key: --jev-key, STREAMCOUNT_JEV_KEY or "
                "TYPESAFE_API_KEY (env or .env file)"
            )
        router = JevRouter(jev_key, model=config.jev_model, threshold=config.jev_threshold)
        _log(f"[info] Jev router: {config.jev_model} (key from {jev_origin}); a recall check "
             f"runs when uncertainty >= {config.jev_threshold:g}")

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
    recorder = _Recorder(run_dir / "timelapse.mp4", config.timelapse_fps) \
        if config.timelapse else None
    # in "both" mode the video records the detector's frames (one annotated pass per frame)
    record_engine = "yolo" if detector is not None else "vlm"

    tracker = (
        FlowTracker(
            min_hits=config.flow_min_hits,
            min_move_px=config.flow_min_move,
            max_age_s=config.flow_max_age,
        )
        if config.flow
        else None
    )

    limit_note = f" | limit: {config.duration:g} min" if config.duration else ""
    _log(f"[run] {run_dir.name} | source: {source[:110]}{limit_note}")
    header = f"{'frame':>5} {'engine':<5} {'count':>6} {'ms':>6}"
    _log(header + ("  tracks/passes" if tracker else "  notes"))

    counts_by_engine: dict[str, list[int]] = {}
    vlm_check_counts: list[int] = []
    router_prev_small = None          # the last decision's frame sample (scene change)
    router_last_check_at: float | None = None
    router_last_vlm: int | None = None
    router_last_local: int | None = None
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

    # a STOP file in the run dir is the panel's graceful stop (hard kill only as fallback)
    stop_file = run_dir / "STOP"

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
                if stop_file.exists():
                    _log(f"[stop] STOP file seen — closing {run_dir.name} cleanly")
                    break
                if config.duration and time.monotonic() - t0 >= config.duration * 60:
                    _log(f"[stop] duration limit reached ({config.duration:g} min) — "
                         f"closing {run_dir.name} cleanly")
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
                        escalate = True
                        if router is not None:
                            now = time.monotonic()
                            brightness, small = describe_frame(image)
                            metrics = RouterMetrics(
                                target=config.target,
                                interval_s=config.interval,
                                detections=len(detections),
                                tracks_active=len(active),
                                tracks_moving=sum(
                                    1 for tr in active if tr["disp"] >= config.flow_min_move
                                ),
                                low_conf=sum(
                                    1 for d in detections if d[4] < config.conf + 0.15
                                ),
                                brightness=brightness,
                                scene_change=scene_change(small, router_prev_small),
                                passes_total=tracker.total,
                                seconds_since_last_check=(now - router_last_check_at)
                                if router_last_check_at is not None else 0.0,
                                last_vlm_count=router_last_vlm,
                                last_vlm_local=router_last_local,
                            )
                            router_prev_small = small
                            router_last_check_at = now
                            decision = router.decide(metrics)
                            escalate = decision.escalate
                            unc = "-" if decision.uncertainty is None else f"{decision.uncertainty:.2f}"
                            conf = "-" if decision.confidence is None else f"{decision.confidence:.2f}"
                            note = f" [{decision.error}]" if decision.error else ""
                            _log(f"[jev]       escalate={str(decision.escalate):<5} "
                                 f"uncertainty={unc} conf={conf} reason={decision.reason} "
                                 f"({decision.latency_ms} ms){note}")
                        if escalate:
                            t_vlm = time.time()
                            answer = vlm.count(image)
                            vlm_count = answer["count"]
                            vlm_check_counts.append(answer["count"])
                            router_last_vlm = answer["count"]
                            router_last_local = len(detections)
                            _log(f"{index:>5} {'vlm':<5} {answer['count']:>6} "
                                 f"{int((time.time() - t_vlm) * 1000):>6}  (recall check)")

                    _log(f"{index:>5} {'yolo':<5} {len(detections):>6} {ms:>6}  "
                         f"active={len(active)} passes={tracker.total}")
                    writer.writerow(
                        [datetime.now().isoformat(timespec="seconds"), index, "yolo",
                         len(detections), ms, source, len(active), tracker.total, vlm_count, ""]
                    )
                    frames_csv.flush()
                    if annotate_dir or recorder:
                        flow_boxes = [(tr["box"][0], tr["box"][1], tr["box"][2], tr["box"][3])
                                      for tr in active]
                        canvas = _annotate(image, flow_boxes, len(active),
                                           f"flow passes={tracker.total}",
                                           ids=[tr["id"] for tr in active])
                        if annotate_dir:
                            canvas.save(annotate_dir / f"f{index:04d}_flow.jpg", quality=88)
                            if config.keep_frames:
                                prune_frames(annotate_dir, config.keep_frames)
                        if recorder:
                            recorder.write(canvas)
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
                    if annotate_dir or (recorder and name == record_engine):
                        canvas = _annotate(image, boxes, count, name)
                        if annotate_dir:
                            canvas.save(annotate_dir / f"f{index:04d}_{name}.jpg", quality=88)
                            if config.keep_frames:
                                prune_frames(annotate_dir, config.keep_frames)
                        if recorder and name == record_engine:
                            recorder.write(canvas)
    finally:
        if events_handle:
            events_handle.close()
        if recorder:
            recorder.close()

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
    if config.keep_frames:
        summary["keep_frames"] = config.keep_frames
    if recorder is not None and recorder.frames:
        summary.update(timelapse=str(recorder.path), timelapse_frames=recorder.frames,
                       timelapse_fps=recorder.fps)
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

        if router is not None:
            stats = router.stats()
            summary.update(jev_router=stats)
            _log(f"  Jev router: {stats['escalations']}/{stats['checks']} checks escalated "
                 f"({stats['skipped']} skipped, {stats['errors']} errors, "
                 f"mean {stats.get('mean_latency_ms', 0)} ms)")

        if config.duration:
            summary.update(duration_minutes=config.duration)

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
    if recorder is not None and recorder.frames:
        _log(f"timelapse: {recorder.path} ({recorder.frames} frames @ {recorder.fps:g} fps)")
    _log(f"CSV: {csv_path}")
    if events_path:
        _log(f"events: {events_path}")
    _log(f"summary: {summary_path}")

    if frames_done == 0:
        print(
            "ERROR: 0 frames received — the source could not be read (typo in the URL, "
            "expired token, missing --headers 'Referer=...', or the stream is offline).\n"
            f"       the (empty) run folder is still at {run_dir} for inspection.",
            file=sys.stderr,
            flush=True,
        )
        result.exit_code = 3

    result.run_dir = run_dir
    result.csv_path = csv_path
    result.events_path = events_path
    result.summary_path = summary_path
    result.total_frames = frames_done
    result.passes = tracker.total if tracker else None
    result.summary = summary
    return result
