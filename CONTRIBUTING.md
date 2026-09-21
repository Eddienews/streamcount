# Contributing to streamcount

Thanks for helping. This project optimises for **honest, measurable behaviour** — a smaller
feature set with real measurements beats a big one with vibes.

## Dev setup

```bash
git clone https://github.com/Eddienews/streamcount
cd streamcount
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                            # 17 tests, all offline
ruff check .
```

You do **not** need a camera, a stream or an API key to develop: the test suite is fully
offline (tracker, parsing, URL helpers, box merging).

## What makes a good PR here

1. **One behaviour per PR**, with a test that fails before and passes after.
2. **Numbers, not adjectives.** If you change detection/tracking defaults, include a measured
   before/after (see below) instead of "feels better".
3. **No secrets, ever.** No keys, tokens or stream URLs with credentials in code, tests,
   fixtures or commits. Use environment variables (`.env.example` documents them).
4. **Keep the engines honest.** If the VLM and the detector disagree, that is information —
   expose it, don't hide it.

## How to measure changes (deterministic method)

Live streams are not reproducible, so capture a fixed frame sequence once and replay it:

```bash
# capture 40 frames, 1 every 2 s
ffmpeg -y -loglevel error -headers $'Referer: <referer>\r\n' -i "$STREAM_URL" \
    -vf fps=1/2 -frames:v 40 -q:v 3 capture/seq%03d.jpg

# replay through any parameter combination — same input every time
streamcount run --images capture --tiles 2 --conf 0.25 --flow --tag experiment
```

Report the summary block (`FLOW:` line, discarded tracks, recall) in your PR.

## Good first issues

- COCO detection model variant so `--target cars` works with the local engine.
- Direction in flow mode (did the person enter or leave?).
- `--output-webhook` for threshold alerts.
- Windows/GPU (CUDA / DirectML) execution-provider support in `detector.py`.

## Reporting bugs

Include: the command line, the source type (not the secret URL), `summary.json`, and — if
relevant — one annotated frame. Rough counts (was the scene empty, sparse ~10, dense 50+?)
help a lot.
