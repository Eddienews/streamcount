# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [SemVer](https://semver.org/).

## [0.1.0] - 2026-09-21

First public version.

### Added
- `streamcount run`: per-frame counting from HLS/RTSP/direct URLs, YouTube links (live and
  recordings, via yt-dlp), local videos and image folders.
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
- Offline test suite (35 tests) covering the tracker, reporting, detector decode, parsing,
  URL helpers and box merging — plus **3 end-to-end tests** with real ffmpeg + ONNX inference
  on a synthetic video (CI job `e2e`).
- README "Known limitations" table: scope, coverage and open risks stated up front.
- Dockerfile + docker-compose, GitHub Actions CI + tag-driven release workflow.
- PEP 639 license metadata; `sdist`/`wheel` build clean (`twine check` passes).

### Fixed
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
