<div align="center">

# streamcount

**Count people and vehicles in any live stream or recording.**
Paste a link — YouTube live, YouTube video, HLS, RTSP, or a local file — get per-frame counts
and unique passers-by, with a local ONNX detector (free, offline) or a vision LLM (your own API key).

[![CI](https://github.com/Eddienews/streamcount/actions/workflows/ci.yml/badge.svg)](https://github.com/Eddienews/streamcount/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

<img src="https://github.com/Eddienews/streamcount/raw/main/assets/screenshot-flow-live.jpg" width="640" alt="Live flow counting on a night street camera — each tracked person gets an id and the running total of passers-by is shown">

<img src="https://github.com/Eddienews/streamcount/raw/main/assets/demo.gif" width="640" alt="Animated demo: tracked ids on a live night street, running pass total in the corner">

<img src="https://github.com/Eddienews/streamcount/raw/main/assets/hour-live-run.gif" width="640" alt="One-hour live run replayed minute by minute: passers-by per minute (bars) and cumulative total (line)">

</div>

---

## Why it exists

Every "people counter" repo assumes you own the camera and a GPU, or that you will pay a
cloud service per stream. streamcount starts from the opposite end:

- **Any source**: an HLS (.m3u8) link, an RTSP camera, a YouTube live stream *or* a YouTube
  recording, a local video, or a folder of frames.
- **Two engines, one CLI**:
  - `--engine yolo` — local ONNX detector. **Zero API cost**, runs offline. CPU-only is fine.
  - `--engine vlm` — any vision LLM behind an OpenAI-compatible endpoint (OpenRouter by
    default, bring your own key). Better on distant/occluded people, costs per frame.
  - `--engine both` — the same frame through both, side by side, for calibration.
- **"How many are there now" *and* "how many passed"**: per-frame counts plus a tracker that
  assigns ids and counts each person once — people who just stand around never count as
  passers-by ([flow mode](#flow-mode--who-passed-not-how-many-are-on-screen)).

## Install

```bash
# from source (recommended while pre-PyPI)
git clone https://github.com/Eddienews/streamcount
cd streamcount && python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# or with Docker (ffmpeg included; runs and the model cache land in ./data — validated
# end-to-end, see docs/MEASUREMENTS.md)
docker compose run --rm streamcount run --youtube "https://www.youtube.com/watch?v=..." --frames 5
```

Requirements: Python 3.10+, `ffmpeg` on PATH. YouTube links additionally need
`yt-dlp` (and, only when YouTube demands it, a JS runtime: `--yt-js-runtime node:/path/to/node`).

The default model (YOLOv8n-pose ONNX, 13 MB) downloads automatically on first run into
`~/.cache/streamcount`. Pre-download with `streamcount download-model`.

## Quickstart

```bash
# A public night-street webcam, local detector, 30 frames every 2 s, annotated output
streamcount run --url "https://videos-3.earthcam.com/fecnetwork/4280.flv/playlist.m3u8?t=..." \
    --headers "Referer=https://www.earthcam.com/" \
    --engine yolo --tiles 2 --interval 2 --frames 30 --annotate

# YouTube live stream, vision LLM, one frame every 5 s
export OPENROUTER_API_KEY=sk-or-...
streamcount run --youtube "https://www.youtube.com/watch?v=LIVE_ID" \
    --engine vlm --interval 5 --frames 120

# A recording: sample from 2 min in, 60 frames
streamcount run --video my_recording.mp4 --seek 120 --engine both --frames 60

# Count who PASSES (unique ids) for 1 hour, with a VLM recall check every 60 s
streamcount run --url "$STREAM" --engine yolo --tiles 2 --interval 2 --frames 1800 \
    --flow --vlm-check 60 --annotate

# Webcam pages hide expiring tokens: extract a fresh stream URL
streamcount find-stream "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/" --cam-id 4280

# After a run: per-minute rates, peak minute, and a chart PNG
streamcount report runs/20260921_003057_yolo_hora1 --chart chart.png
```

### Counting vehicles

```bash
streamcount download-model --variant yolo11n          # COCO 80-class export (~10 MB)
streamcount run --video traffic.mp4 --target cars \
    --model ~/.cache/streamcount/models/yolo11n.onnx --conf 0.35 --frames 60
```

Vehicles need a COCO detection model. The default pose model is people-only and says so
loudly if you point it at `--target cars`. Validated on a downtown NYC driving video:
the detector boxed 4/4 visible vehicles per frame; the VLM, asked for cars, said 6 — it
plausibly counts partially occluded vehicles that the detector skips. Treat the pair as a range.

Outputs (one folder per run):

| file | content |
|---|---|
| `frames.csv` | one row per sampled frame per engine: count, latency, active tracks, passes total |
| `events.csv` | flow mode: one row per person counted (`t`, `id`, `hits`, `displacement`, running total) |
| `frames/*.jpg` | annotated frames (`--annotate`): boxes, track ids, running total |
| `summary.json` | machine-readable summary, including recall correction when `--vlm-check` is used |

## Flow mode — "who passed", not "how many are on screen"

A per-frame count answers *"how many people are visible right now"*. It cannot answer
*"how many people passed during the last hour"* — the same person appears in dozens of
frames. Flow mode tracks:

1. detections are associated into tracks (IoU or centre distance — sampling gaps of 1-5 s
   make IoU alone too strict);
2. a track is **confirmed** after `--flow-min-hits` detections (default 3, filters noise);
3. it is **counted once** when it has moved `--flow-min-move` px (default 40) from where it
   first appeared — standing people (vendors, buskers) are never counted as passers-by;
4. dead tracks become ghosts for 15 s, so a detection drop-out does not spawn a second count.

```mermaid
flowchart LR
    A[source<br/>HLS · RTSP · YouTube · file] --> B[ffmpeg sampling<br/>1 frame / N seconds]
    B --> C{engine}
    C -->|yolo| D[ONNX detector<br/>+ optional NxN tiles]
    C -->|vlm| E[vision LLM<br/>bring your own key]
    D --> F[tracker<br/>ids · confirmation · ghosts]
    F --> G[frames.csv<br/>events.csv<br/>annotated frames]
    E --> G
    D -.recall anchor.- E
```

## Accuracy: what this does and does not deliver

Measured on a real 1080p night street camera (see [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md)):

- The nano detector on the **full frame** found 3 of ~12 people in the scene. With
  **`--tiles 2`** it found 9-10 of ~12 (one duplicate pair), at ~0.8-1.2 s/frame on CPU.
- The **VLM** counted 12 / 13 / 12 on three runs of the *same* frame (temperature 0) —
  treat its number as ±1-2, not a census. In a dense crowd it over-counts (~30-34 vs a
  manual ~25-30).
- In flow mode, the detector's **recall was ~46%** on that night scene (8.0 visible detected
  vs 17.5 seen by the VLM). Raw tracker passes are therefore a **floor**; `--vlm-check`
  measures the recall and reports a corrected estimate (28 → ~61 passes over 80 s of video).
- **One uninterrupted hour of that camera** (1,800 frames, 1 every 2 s): **1,001 people counted
  passing** (16.7/min, peak 40/min), recall ~70% over the hour → corrected ≈ **1,430/hour**.
  Every minute of that hour had at least one passer-by. Raw run + chart in
  [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md).
- False positives do happen (a table with a candle was once "a person") — validate visually
  before trusting a threshold-critical number.

**Use it for trend, throughput and relative comparison. Do not use it as a legally exact
census.** The tool reports latency, recall and uncertainty on purpose — most counting demos do not.

## Cost of the AI (vision LLM engine)

Measured on `google/gemini-2.5-flash-lite` via OpenRouter, 1080p frame: **1,887 input +
~33 output tokens per frame ≈ $0.0002/frame**.

| Sampling | Frames/hour | Cost/hour |
|---|---|---|
| every 2 s | 1,800 | ~$0.36 |
| every 5 s | 720 | ~$0.15 |
| every 30 s | 120 | ~$0.02 |
| local detector only | — | **$0.00** (your CPU) |

Reducing the frame resolution did **not** reduce token cost in our measurements (same 1,887
tokens at 1280 px and 640 px) — the sampling rate is the real cost lever. Recordings can use
batch endpoints (~50 % cheaper). Full table and provider comparison (qwen, gemma):
[docs/COSTS.md](docs/COSTS.md).

## Sources

| Source | How | Notes |
|---|---|---|
| HLS / .m3u8 | `--url` | webcams often need `--headers "Referer=..."`; tokens expire → `find-stream` |
| RTSP | `--url rtsp://...` | same ffmpeg pipe; TCP transport is added automatically for cameras behind NAT/firewalls |
| YouTube live | `--youtube URL` | yt-dlp; may need `--yt-js-runtime node:...` |
| YouTube recording | `--youtube URL` | sample with `--interval`/`--seek` |
| Local video | `--video file.mp4` | |
| Images | `--image f.jpg` / `--images DIR` | a folder is treated as a frame sequence |

## Limitations & ethics

- **No identification.** This counts people; it does not recognize faces, and it should not
  be extended to do so. Aggregate counting in public spaces is one thing, biometric
  surveillance is another (and illegal without a proper legal basis in many jurisdictions).
- Respect the terms of the streams you process; do not redistribute third-party streams, and
  keep polling rates sane (≥ 1 s).
- The VLM engine sends sampled frames (JPEG, ~1280 px, no audio) to the provider you
  configure. The local detector sends nothing anywhere.
- Bundled model weights are **not** included in this repository. streamcount downloads
  Ultralytics YOLOv8-pose ONNX exports on first use; those weights are **AGPL-3.0** licensed
  by their authors. For commercial use, either comply with AGPL or point `--model` at weights
  you are licensed to use. See `LICENSE` and `NOTICE.md`.

## Known limitations (verified scope — read before trusting a number)

This project publishes both what it does and what it has *not* been tested against.

| area | status |
|---|---|
| **Detector recall** | 46–70 % depending on scene density (night street). Counts are a **range** (floor → recall-corrected), never a census. Dense crowds: ±10–20 %. |
| **RTSP** | ingest validated end-to-end against a local **mediamtx** server (per-frame run + flow run, TCP transport; UDP also decoded). Not yet exercised against a physical camera. |
| **Vehicle flow** | per-frame vehicle counting is validated (4/4 boxed on a traffic video); *flow counting of vehicles* (ids across time) is validated for people only. |
| **VLM coverage** | validated with `gemini-2.5-flash-lite` only; `gemma-3-12b` untested; `qwen3.7-flash` failed our strict-JSON reply and needs prompt tuning. |
| **Run length** | longest validated run: **1 hour** (1,800 frames). Memory/stability beyond that is unverified. |
| **Platforms** | integration (ffmpeg/yt-dlp pipes) exercised on **Windows**; CI runs the offline unit suite plus an ffmpeg end-to-end smoke test on Linux. macOS untested. |
| **Scenes** | two scenes measured (night pedestrian street; daytime traffic). Rain, snow, extreme crowds untested. |
| **Docker** | image **validated end-to-end**: built from `Dockerfile`, ran the full pipeline (ffmpeg + ONNX inside the container) on a mounted volume, model cache persisted to the host. Compose file: config-validated only. |
| **YouTube** | extraction depends on yt-dlp and occasionally on a JS runtime or cookies; expect periodic breakage as YouTube changes. |
| **Re-identification** | a person leaving and returning after > 15 s is counted a second time (technically two visits). |
| **Model licenses** | bundled *downloads* are AGPL-3.0 Ultralytics weights — commercial users should swap in their own model (see NOTICE.md). |

## Roadmap

- [x] COCO detection model variant + validated vehicle counting (`--target cars`)
- [ ] Zones / line-crossing counts (in/out direction)
- [ ] Web UI (paste link → live dashboard)
- [ ] Re-ID across longer occlusions
- [ ] PyPI release + one-file binaries

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The test suite is split into
35 offline tests and 3 end-to-end tests (`pytest -m e2e`, needs ffmpeg) so you can hack on it
without a camera or a key. Want to just *use* it first? [docs/MANUAL-TEST.md](docs/MANUAL-TEST.md)
walks through every source type in ten minutes.

## License

MIT — see [LICENSE](LICENSE). Third-party model weights are **not** MIT; see the note above.
