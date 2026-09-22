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
pytest -m "not e2e"     # [81 passed]
pytest -m e2e           # [4 passed] needs ffmpeg; downloads the default model once
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
streamcount run --url "<url printed above>" \
    --engine yolo --tiles 2 --flow --interval 2 --frames 75
# (EarthCam's required Referer header is added automatically; --headers overrides)
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

## 8. Live dashboard (paste the link, watch it count)

```bash
streamcount serve --port 8766 --open
```

Paste any link in the page (HLS, RTSP, YouTube, local file) and press *count*: the annotated
frame refreshes every interval, next to passers-by, in-scene count, per-minute bars and the
log tail. *stop* ends the run. The run is written to `runs/web/<run_dir>` like any other, so
`streamcount report` works on it afterwards.

[Expect: numbers climbing within ~15 s; a frame with red boxes and `id…` labels. After a few
minutes, `runs/web/<run_dir>/frames/` holds 10 JPEGs, not hundreds — dashboard runs keep only
the last 10 annotated frames (`--keep-frames`). Tick *record mp4* before pressing count to get
an annotated `timelapse.mp4` in the run folder; the page shows a ▶ link once it exists. Press
*stop* and the run closes gracefully — the folder gains `summary.json` + `chart.png`.
*options* also carries `vlm every (s)` and the *Jev router* switch for a hybrid run.]

## 9. Optional: the Jev router (TypeSafe)

Off by default; needs a TypeSafe key (`console.typesafe.ai/settings/keys`). To check the gate
without any paid vision model, point the anchor at a stub: any server answering the OpenAI
shape with `{"count": 7}` works.

```bash
export TYPESAFE_API_KEY=...
streamcount run --images <frames-dir> --flow --tiles 2 --interval 2 --frames 24 \
    --vlm-check 6 --vlm-key stub --vlm-base-url http://127.0.0.1:8899/v1 --jev-router
```

[Expect: one `[jev] escalate=… uncertainty=… reason=…` line per check, VLM lines only for the
escalated checks, and `summary.json` with `jev_router: {checks, escalations, skipped, errors,
mean_latency_ms}`. A live 24-frame replay gave 8 decisions / 1 escalation / 0 errors, ~450 ms
each; a direct probe of the real API separated a calm daylight state (uncertainty 0.11, skip)
from a dark dense one (0.87, escalate).]

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
