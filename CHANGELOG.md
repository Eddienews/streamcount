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
- Outputs: `frames.csv`, `events.csv`, annotated frames, `summary.json`.
- Offline test suite (17 tests) covering the tracker, parsing, URL helpers and box merging.
- Dockerfile + docker-compose, GitHub Actions CI.

### Measured baseline (night street cam, 1080p)
- Full-frame nano detector: 3 of ~12 people; tiles 2×2: 9-10 of ~12 (conf 0.25).
- VLM (gemini-2.5-flash-lite): 12/13/12 on repeated runs of the same frame; ~$0.0002/frame.
- Flow: 55 passes in 2.5 min (live), 28 passes in 80 s (fixed replay) with detector recall
  ~46%, corrected estimate ~61. Full data: docs/MEASUREMENTS.md.
