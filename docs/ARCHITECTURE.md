# Architecture

```
source ──► sampling ──► engine ──► tracker ──► outputs
```

## Modules

| module | responsibility |
|---|---|
| `streamcount/sources.py` | frame acquisition: ffmpeg pipe for HLS/RTSP/direct URLs and local files, `yt-dlp` resolution for YouTube, image folders, m3u8 extraction from webcam pages |
| `streamcount/detector.py` | local ONNX detector: letterboxing, class filtering, NMS, tiled inference, stacked-box merging, model download/cache |
| `streamcount/vlm.py` | vision-LLM counting through any OpenAI-compatible endpoint; tolerant JSON parsing; key resolution (flag → env → `.env`) |
| `streamcount/tracking.py` | `FlowTracker`: association, confirmation, displacement gate, ghost re-identification |
| `streamcount/pipeline.py` | orchestration, CSVs, annotation, recall correction, `summary.json` |
| `streamcount/cli.py` | `run`, `find-stream`, `download-model` |

Nothing is hidden behind a service: the whole pipeline is these six modules plus `ffmpeg`
(and optionally `yt-dlp`).

## Sampling

One frame every `--interval` seconds, decoded from an ffmpeg `image2pipe` MJPEG stream and
split on JPEG markers — no temp files, clean shutdown when `--frames` is reached.
`-rw_timeout 15s` is set so a dead stream fails fast instead of hanging.

**Timeline:** live sources (`*.m3u8`, `rtsp://`) are timestamped with the **wall clock**
(a slow consumer must not compress time); recordings and image sequences use **video time**
(`frame_index * interval + seek`). Override with `--timeline wall|video`.

## Engines

- **Local detector** (`detector.py`): YOLOv8 ONNX export. Supports pose exports
  (person-only, 56 channels) and detection exports (COCO 80 classes, `4+nc` channels), so
  `--target cars` works with a detection model. `--tiles N` splits the frame into an NxN grid
  with 15 % overlap and runs the model on each tile — the cheapest way to recover small and
  distant objects without a bigger network (see MEASUREMENTS for why).
- **VLM** (`vlm.py`): one frame → strict-JSON count. Temperature 0, `max_width` 1280.
  Cannot assign identities across frames; use it per-frame or as a recall anchor.

## Flow tracker

States: `new → confirmed (≥ min_hits) → counted (displacement ≥ min_move_px) → dead/ghost`.

- Association: greedy over `IoU ≥ 0.15` **or** centre distance ≤ 0.9 × box height. Low-frame
  sampling makes IoU alone too strict.
- Confirmation: `--flow-min-hits` detections (default 3) keeps single-frame false positives
  out (measured: they are 30-40 % of raw detections on a noisy night scene).
- Pass gate: a confirmed track counts **once** when it has moved `--flow-min-move` px
  (default 40). Standing people never count — this is the difference between "who passed"
  and "how many are visible".
- Ghosts: dead tracks stay re-identifiable for 15 s; a nearby reappearing detection inherits
  the id, so brief detector drop-outs do not double-count a person.

## Recall anchoring (`--vlm-check`)

Every N seconds the same frame goes to both engines. Mean visible counts (`yolo` vs `vlm`)
give a recall ratio; the summary reports `passes` (floor) and
`passes_recall_corrected = passes / recall`. This is deliberately conservative: if the VLM
over-counts a dense crowd, the correction is capped at 1.0 (never below the raw floor).

## Outputs

| file | schema |
|---|---|
| `frames.csv` | `ts_iso, frame, engine, count, latency_ms, source, active_tracks, passes_total, vlm_count, notes` |
| `events.csv` | `ts_iso, t_rel_s, event, id, hits, disp_px, total` (one row per counted passer-by) |
| `frames/fNNNN_*.jpg` | annotated frames (boxes, ids, running total) |
| `summary.json` | source, engines, frames, flow stats, recall data |

## Extension points

Custom engines implement `detect(image) -> list[(x1, y1, x2, y2, score, label)]` (detector) or
`count(image) -> {"count", "uncertain", "notes"}` (VLM). The tracker consumes plain boxes, so
it is engine-agnostic. GPU execution providers slot into `Detector.__init__`.
