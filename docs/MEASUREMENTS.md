# Measurements

Everything below was measured on live/recorded public streams on 2026-09-21 (00:05–00:25
local time, UTC-4) with the code in this repository. Scene: 1920×1080 night street corner
(Bourbon St, New Orleans, via an EarthCam HLS stream), 8–25 people in frame, mixed
pedestrians, vendors and seated people.

## Per-frame counting

| configuration | result on reference frame | notes |
|---|---|---|
| manual count (ground truth by careful inspection) | **~12** | one "object" turned out to be a table with a candle — not a person |
| YOLOv8n-pose, whole frame, conf 0.35 | 3 | only the nearest people |
| YOLOv8n-pose, whole frame, conf 0.25 | 4 | |
| YOLOv8n-pose, `--tiles 2`, conf 0.25 | 9–10 | captures mid-distance walkers |
| YOLOv8n-pose, `--tiles 3`, conf 0.25 | 13 | 1–2 duplicates survive (upper/lower body splits) |
| YOLOv8s-pose, `--tiles 2`, conf 0.25 | 10 | 2.5× slower than nano, **no recall gain** on this scene |
| **VLM gemini-2.5-flash-lite** | **12 / 13 / 12** (3 runs of the same frame) | temperature 0 — the model is not deterministic; treat as ±1–2 |

False positives: at conf 0.25–0.35 the detector flagged a table with a candle and some
barrier/pavement textures. `--flow-min-hits 3` filters these out of the counts (they never
survive three samples), which is why flow defaults are stricter than per-frame defaults.

## Speed (CPU, this laptop)

| engine | ms/frame |
|---|---|
| YOLOv8n-pose, 640 | 135–220 |
| YOLOv8n-pose, `--tiles 2` | 700–1,200 |
| YOLOv8n-pose, `--tiles 3` | ~1,470 |
| YOLOv8s-pose, `--tiles 2` | ~1,540 |
| YOLOv8s-pose **int8 ONNX export from the same hub** | 6,353 (broken export — don't use) |
| VLM (API) | 1,700–2,400 |

Models: `yolov8n-pose` ONNX 13.5 MB, `yolov8s-pose` ONNX 46.8 MB.

## Flow counting (who passes, each person once)

| run | configuration | result |
|---|---|---|
| live, 2.5 min | min-hits 2, min-move 25 px | **55 passes** (22/min); 23 short tracks discarded; 3 confirmed but stationary |
| fixed replay, 80 s (40 frames) | min-hits 2, min-move 25 px | 37 passes; 21 noise tracks |
| fixed replay, same frames | **min-hits 3, min-move 40 px** (shipped defaults) | **28 passes** (21/min); 34 noise tracks |
| fixed replay + VLM every 20 s | recall anchor | yolo visible avg **8.0** vs VLM **17.5** → recall ≈ **46 %** → corrected ≈ **61** passes |

Interpretation: the raw tracker total is a **floor** on a hard night scene (low detector
recall, dense crowd). The honest answer for that corner is a range — ~28 floor, ~61
recall-corrected over 80 s (≈ 21–46 passers-by per minute at midnight). This is why the
pipeline always reports recall alongside the count.

Deterministic replay is the recommended way to compare parameters: capture frames once with
ffmpeg (`-vf fps=1/2 -frames:v 40`), then run `streamcount run --images capture ...` — same
input every time.

## One-hour live run (the strongest evidence in this repo)

2026-09-21, 00:31–01:30 local time, same night street camera, `--tiles 2 --conf 0.25 --interval 2
--flow --flow-min-hits 3 --flow-min-move 40 --vlm-check 60`, single uninterrupted process.

| metric | value |
|---|---|
| frames processed | **1,800** (1 every 2 s; 59.8 min of wall time, no drift) |
| passers-by counted (floor) | **1,001** (16.7/min) |
| peak minute | **40/min** (minute 19); other busy minutes: 0 (30), 1 (28), 24 (28), 40 (30) |
| minutes with zero passers-by | **0** — the street never emptied at any point in the hour |
| most confirmations in one sample | 6 |
| discarded short tracks (noise) | 872 of ~1,500 spawned ids (≈1 in 3; filtered by `--flow-min-hits 3`) |
| confirmed but stationary | 55 (never counted — vendors/buskers) |
| VLM recall checks | 33 (every 60 s): yolo visible 8.3 vs VLM 11.8 → **recall ≈ 70 %** |
| recall-corrected estimate | **≈ 1,430 passers-by/hour (≈ 24/min)** |

Chart: `assets/hour-live-run-chart.png`. Animated replay of the same data (bars filling minute
by minute): `assets/hour-live-run.gif`, rendered with `scripts/render_hour_gif.py`.
Raw data: `frames.csv` + `events.csv` of that run.

Two things this hour proves beyond the shorter runs: the pipeline holds for an hour without
drift or leaks (1,800/1,800 frames, stable ~0.7 s/frame CPU), and detector recall in this
scene is **density-dependent** — ~46 % during a dense-crowd window vs **70 % averaged over
the full hour** — which is exactly why the tool reports a range (floor → recall-corrected)
instead of a single number.

## Dense-crowd behaviour (live, 00:11–00:12)

| moment | yolo (`--tiles 2`) | VLM | manual |
|---|---|---|---|
| crowd building | — | 23–24 | ~20–24 |
| crowded street | 15 (correct boxes, misses the background) | 30–34 | ~25–30 |

In dense scenes the VLM tends to over-count and the nano detector under-counts; both should
be read as ±10–20 %.

## Non-camera sources (ingestion checks)

| source | method | result |
|---|---|---|
| YouTube recording (walking tour) | `yt-dlp -g` → frame at t=420 s → VLM | 19 people (manual ~13–19) |
| YouTube **live** stream | `yt-dlp -g` with `--yt-js-runtime node:...` → HLS manifest → frame → VLM | 24 people (manual ~20–25) |
| EarthCam page | `streamcount find-stream --cam-id 4280` | fresh tokenised .m3u8 URL |

Note: YouTube extraction requires `yt-dlp` on PATH; when YouTube demands a JS runtime, pass
`--yt-js-runtime node:/path/to/node`. Live and recording both ingest through the same
ffmpeg pipe.
