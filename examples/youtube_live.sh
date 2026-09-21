#!/usr/bin/env bash
# YouTube live stream -> vision-LLM counting, one frame every 5 seconds.
#
# Needs: yt-dlp on PATH (plus `--yt-js-runtime node:/path/to/node` if YouTube asks
# for a JS runtime) and an API key in OPENROUTER_API_KEY (or STREAMCOUNT_VLM_KEY).
set -euo pipefail

URL="${1:?usage: youtube_live.sh <youtube-live-url> [frames]}"
FRAMES="${2:-120}"

streamcount run \
  --youtube "$URL" \
  --engine vlm --vlm-model google/gemini-2.5-flash-lite \
  --interval 5 --frames "$FRAMES" \
  --annotate --tag youtube-live

# Cost note: ~$0.0002 per frame -> 120 frames ≈ 2.4 cents.
