# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `run --keep-frames N`: cap the annotated frames kept on disk. A live session otherwise writes
  ~320 MB/h of JPEGs while the dashboard only ever shows the newest one; the last N frames are
  kept instead (`0` = keep every frame, the CLI default). Dashboard runs pass `--keep-frames 10`
  (editable under *options*); `summary.json` records `keep_frames` when set.
- `run --timelapse` (`--timelapse-fps`, default 12): the run writes an annotated **timelapse MP4**
  while it is going (ffmpeg image2pipe → `timelapse.mp4` in the run folder, finalized when the run
  stops). Works together with `--keep-frames` — the video does not depend on the JPEGs. The
  dashboard exposes it as a *record mp4* option and links the file on the page.
- `run --jev-router` (`--jev-key`, `--jev-model`, `--jev-threshold`; off by default): with
  `--vlm-check N` the recall anchor stops following the clock. Every N seconds Jev (TypeSafe's
  System One, `api.typesafe.ai/v1/systemone`) answers one typed question about whether the
  local reading is likely wrong (dark or blurred frame, dense scene, detections at the
  confidence floor, a scene change, a disagreeing last anchor) and the paid VLM call runs only
  when the answer says it is worth it. What leaves the machine is a paragraph of numbers, never
  a frame; a failing router escalates anyway (the anchor is the safe side); `summary.json`
  records `jev_router: {checks, escalations, skipped, errors, mean_latency_ms}`.
- A **time limit** for any run: `run --duration MIN` (and `duração (min)` in the dashboard)
  closes the session cleanly when the mark is reached — the same graceful path as the panel's
  stop, so `summary.json` (with `duration_minutes`), `chart.png` and the `timelapse.mp4` are all
  written. `0` keeps the old behaviour: run until stopped. Validated end to end: a run with
  `--duration 0.02` stops short of its frame cap and still writes summary + chart + MP4.
- **Model choice next to the switches**: the dashboard exposes `modelo vlm` + `servidor vlm`
  (`--vlm-model`, `--vlm-base-url` — any OpenAI-compatible provider) and `modelo jev`
  (`--jev-model`), so a run can use whichever model the operator has a key for. Keys stay in the
  environment.

### Fixed
- **The dashboard now shows what a run was asked to do**: while a limit is set, the meta line
  carries `limite N min (restam X)` as it counts down, plus `mp4` and `vlm`/`Jev` tags for the
  hybrid legs. Before this the page echoed only interval/tiles/engine/conf/target — a
  ten-minute limit or an active timelapse was invisible even though both were applied (the run
  log had them; the page did not).
- **EarthCam links now work without a manual `Referer`**: the HLS CDN answers 403 without one
  (token or not), so the pipeline adds `Referer=https://www.earthcam.com/` automatically when the
  user set none — explicit `--headers` always wins.
- The dashboard's `/latest.jpg` picked the newest frame with a plain name sort, which puts
  `f10000` *before* `f9999` — past 9,999 frames (~5.5 h at 2 s) the page could serve a stale
  frame. The numeric frame index now decides.
- **Stopping a panel run is graceful**: the supervisor drops a `STOP` file in the run directory
  and the pipeline leaves the frame loop normally, writing `summary.json` and `chart.png` (+
  closing the timelapse encoder). Before this a manual stop was a hard terminate: the MP4 only
  survived because ffmpeg finalises on EOF, and the summary and the chart were lost.

### Changed
- README assets refreshed from real runs: `assets/dashboard.png` now shows the *target* picker
  and *record mp4* right under the link field (no need to open *options*) plus the
  `▶ timelapse.mp4` link of a stopped run (43 frames, 28 passers-by); `assets/demo.gif` was
  rebuilt from a 3-minute run's frames (90 frames, 55 passers-by) with the new
  `scripts/render_demo_gif.py`. Three unreferenced screenshots dropped from `assets/`.
- The *target* (people/cars) and *record mp4* controls moved out of *options*: they sit right
  under the link field now, so neither needs the advanced panel. The language toggle also
  translates the target options (they were hardcoded Portuguese).
- The dashboard *options* grid gained the hybrid pair: `vlm every (s)` (`--vlm-check`) and the
  **Jev router** switch (`--jev-router`) — the CLI's option-or-not, exposed, with keys still
  coming from the environment (the page never carries them).
- New README asset `assets/cars-budapest.jpg`: a real 10-minute `--target cars` session on a
  live Danube camera — 114 vehicles, 303 frames, self-closed at its `--duration 10` limit.

## [0.1.0] - 2026-09-21

First public version.

### Added
- `streamcount serve`: a **local live dashboard** (stdlib only, bound to 127.0.0.1) — paste a
  link in the page and watch the count running: annotated frame every interval, passers-by,
  in-scene count, per-minute bars, latest passes with ids, log tail, start/stop buttons.
  Dashboard runs are live sessions (no frame cap) and land in `runs/web/`.
- `streamcount run`: per-frame counting from HLS/RTSP/direct URLs, YouTube links (live and
  recordings, via yt-dlp), local videos and image folders. `rtsp://` sources get
  `-rtsp_transport tcp` automatically (validated against a local mediamtx server).
- Two interchangeable engines: local ONNX detector (`yolo`) and vision-LLM API (`vlm`),
  plus `both` for side-by-side calibration on the same frame.
- Tiled inference (`--tiles N`) with stacked-box merging (torso+legs) to recover small and
  distant objects without doubling counts.
- Flow mode (`--flow`): stable ids, confirmation threshold, displacement gate for
  passers-by, and ghost re-identification across detector drop-outs.
- Recall anchoring (`--vlm-check`): periodic VLM counts measure the detector's recall and
  the pipeline reports a corrected pass estimate alongside the raw floor.
- `streamcount find-stream`: extracts a fresh (expiring-token) HLS URL from webcam pages.
- `streamcount download-model`: pre-fetches ONNX weights into the local cache.
- COCO detection model variant (`yolo11n`, 10.7 MB) and `--target cars` — vehicle counting
  validated on a real traffic video (detector 4/4 vehicles boxed; see docs/MEASUREMENTS.md).
- `streamcount report`: per-minute pass rates, peak minute, optional chart PNG (Pillow only),
  tolerant of legacy events files.
- Demo GIF (`assets/demo.gif`) generated from a real live flow run; animated one-hour chart
  (`assets/hour-live-run.gif`, `scripts/render_hour_gif.py`).
- Outputs: `frames.csv`, `events.csv`, annotated frames, `summary.json` (+ `chart.png` in flow runs).
- Offline test suite (54 tests) covering the tracker, reporting, detector decode, parsing,
  dashboard routes, URL helpers and box merging — plus **4 end-to-end tests** with real ffmpeg
  + ONNX inference on a synthetic video (CI job `e2e`), including the zero-frames failure mode.
- README "Known limitations" table: scope, coverage and open risks stated up front.
- Dockerfile + docker-compose (image validated end-to-end: full pipeline ran inside the
  container with a host volume), GitHub Actions CI + tag-driven release workflow.
- PEP 639 license metadata; `sdist`/`wheel` build clean (`twine check` passes).

### Fixed
- A run that received **zero frames** (typo'd URL, expired token, missing `Referer`, offline
  stream) now prints an explicit `ERROR: 0 frames received` and exits **3**, instead of
  looking like a successful empty run. Covered by an e2e test.
- COCO detection path crashed NMS (scores were not masked together with boxes) — caught by
  the vehicle validation on first real detection-model run. Regression tests:
  `tests/test_detector_decode.py`.

### Measured baseline (night street cam, 1080p)
- Full-frame nano detector: 3 of ~12 people; tiles 2×2: 9-10 of ~12 (conf 0.25).
- VLM (gemini-2.5-flash-lite): 12/13/12 on repeated runs of the same frame; ~$0.0002/frame.
- Flow: 55 passes in 2.5 min (live), 28 passes in 80 s (fixed replay) with detector recall
  ~46%, corrected estimate ~61. Full data: docs/MEASUREMENTS.md.
- One-hour uninterrupted live run: 1,800 frames, **1,001 passers-by** (16.7/min, peak 40/min),
  recall ~70% over the hour → corrected ≈ 1,430. Chart: assets/hour-live-run-chart.png.
