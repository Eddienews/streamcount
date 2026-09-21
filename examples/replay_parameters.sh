#!/usr/bin/env bash
# Deterministic parameter comparison: capture frames ONCE, replay many times.
#
# Live streams change between runs, so any "before/after" measured on live video is
# anecdote. Capture a fixed sequence, then replay it through different settings.
set -euo pipefail

STREAM="${STREAM:?set STREAM to an HLS/RTSP url}"
REFERER="${REFERER:-}"
FRAMES="${FRAMES:-40}"

mkdir -p capture
ffmpeg -y -loglevel error \
  ${REFERER:+-headers "Referer: $REFERER\r\n"} \
  -i "$STREAM" -vf fps=1/2 -frames:v "$FRAMES" -q:v 3 capture/seq%03d.jpg

for cfg in "sensitive 2 25" "default 3 40" "strict 4 60"; do
  set -- $cfg
  echo "=== $1 (min-hits=$2, min-move=$3) ==="
  streamcount run --images capture --tiles 2 --conf 0.25 --flow \
    --flow-min-hits "$2" --flow-min-move "$3" --tag "replay-$1" \
    | grep -E "FLOW|discarded"
done
