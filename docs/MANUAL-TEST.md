# Manual test plan

About ten minutes, no API key required. Numbers in brackets are what a healthy run looks
like — they come from the measurements in [MEASUREMENTS.md](MEASUREMENTS.md).

## 0. Install

```bash
git clone <your fork or this repo> && cd streamcount
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
streamcount --version
```

Requirements: Python 3.10+, `ffmpeg` on PATH. Add `yt-dlp` for YouTube sources.

## 1. Automated suite (offline, ~15 s)

```bash
pytest -m "not e2e"     # [35 passed]
pytest -m e2e           # [3 passed] needs ffmpeg; downloads the default model once
```

## 2. Deterministic replay — the reference number

Point it at any folder of frames. With the 40 captured frames that ship the measurement
work (not in the repo — capture your own with `ffmpeg -i <video> -vf fps=1/2 seq%03d.jpg`):

```bash
streamcount run --images ./frames --flow --tiles 2 --conf 0.25 --interval 2
```

[Reference: 28 passers-by on the Bourbon Street night capture — the same number the
prototype produced, twice.]

## 3. Live HLS (a webcam page)

```bash
streamcount find-stream "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/"
streamcount run --url "<url printed above>" --headers "Referer=https://www.earthcam.com/" \
    --engine yolo --tiles 2 --flow --interval 2 --frames 75
```

[Expect: a frame every 2 s, counts fluctuating with street traffic, `passes` climbing.]

## 4. YouTube live or recording

```bash
streamcount run --youtube "https://www.youtube.com/watch?v=<id>" --frames 5 --interval 30
# if yt-dlp complains about JavaScript:
streamcount run --youtube "<url>" --yt-js-runtime node:/path/to/node --frames 5
```

## 5. Vehicles (`--target cars`)

```bash
streamcount download-model --variant yolo11n
streamcount run --video traffic.mp4 --target cars \
    --model ~/.cache/streamcount/models/yolo11n.onnx --conf 0.35 --frames 30 --interval 2
```

[Expect: boxes on cars/trucks/buses, not on pedestrians — the COCO model knows both.]

## 6. Docker (no Python on the host needed)

```bash
docker compose run --rm streamcount run --url "<hls>" --flow --tiles 2 --frames 10
# or with the image directly:
docker build -t streamcount:test .
docker run --rm -v "$PWD/data:/data" streamcount:test run --video /data/clip.mp4 --frames 3
```

[Expect: the same CSVs under `./data/runs/`; the model cache lands in `./data/models/`.]

## 7. The report

```bash
streamcount report runs/<run_dir> --chart chart.png
```

[Expect: `passers-by: N`, per-minute table, peak minute, and a chart PNG. A run that
counted nobody is a valid result (`passers-by: 0`, exit 0).]

## How to judge the result

- **Counts are a floor.** The detector's recall was measured at 46–70 % depending on scene.
  A number lower than your own eyeball count is expected, not a bug — use `--vlm-check` to
  measure recall on your own scene and read the corrected range.
- **The VLM is not deterministic** (12/13/12 on the same frame at temperature 0) — expect a
  range from repeated runs.
- **Failure looks like**: `ERROR: 0 frames received` with **exit code 3** (typo in the URL,
  expired `t=`/`td=` token, missing `Referer`, or the stream is offline), ffmpeg HTTP 403,
  or `yt-dlp failed` (usually a JS runtime or a video that needs sign-in). A run that got
  frames but counted nobody exits 0 with `passers-by: 0` — that one is a valid result.
- When reporting a problem, include: the command, the source type (HLS/YouTube/file), and
  the run folder (it has `frames.csv`, `summary.json` and the log).
