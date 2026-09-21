# Cost of the AI (vision-LLM engine)

All prices are USD, taken from the OpenRouter model list on 2026-09-21, and **token usage was
measured**, not estimated: one 1080p frame resized to 1280 px wide, prompt + JSON reply,
`google/gemini-2.5-flash-lite`, temperature 0.

## Measured usage per frame

| | tokens |
|---|---|
| input (image + prompt) | **1,887** |
| output (strict JSON) | ~31-36 |

**≈ $0.0002 per frame** on `gemini-2.5-flash-lite` (in $0.10/M, out $0.40/M).
Latency: 1.8-2.4 s per frame on OpenRouter.

Notable finding: **lowering the frame resolution did not lower the token count** in our
measurements (1,887 tokens at 1280 px *and* at 640 px — the provider bills by image tiles).
The sampling rate is the real cost lever, not the resolution.

## Cost per hour of video

| sampling | frames/hour | gemini-2.5-flash-lite | gemma-3-12b-it ($0.05/$0.15) |
|---|---|---|---|
| every 2 s | 1,800 | ~$0.36 | ~$0.18 |
| every 5 s | 720 | ~$0.15 | ~$0.07 |
| every 10 s | 360 | ~$0.07 | ~$0.04 |
| every 30 s | 120 | ~$0.02 | ~$0.01 |
| local detector | — | **$0.00** (your CPU) | **$0.00** |

Anchors, to keep estimates honest:

- one stream monitored 24/7 at 1 frame / 5 s ≈ 17,280 frames/day ≈ **$104/month** on flash-lite;
- recordings (offline) can use *batch* endpoints (flash-lite batch: $0.05/M in) — roughly half;
- qwen3.7-flash ($0.03/$0.13) is ~4× cheaper per frame but **failed the strict-JSON reply** in
  our test (it answered in prose); treat non-flagship models as needing prompt tuning;
- the **hybrid** pattern used by `--flow --vlm-check 60`: local detector on every frame (free)
  + VLM once a minute as a recall anchor ≈ **$0.012/hour** of AI cost, with flow counting
  still available.

## Cost control checklist for deployments

1. Start at `--interval 5` and `--vlm-check` for anchoring, not per-frame VLM.
2. Prefer the local detector for always-on jobs; reserve the VLM for validation windows.
3. Cap frames per job (`--frames`) — a job's AI ceiling is `frames × $0.0002`.
4. Bring-your-own-key by design: the person running the job owns the API bill.
