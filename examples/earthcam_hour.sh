#!/usr/bin/env bash
# Count who passes on an EarthCam-style webcam for one hour.
#
# The page hosts tokenised HLS URLs that expire, so fetch a fresh one right before
# starting. Adjust PAGE/CAM_ID to another camera if you like.
set -euo pipefail

PAGE="${PAGE:-https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/}"
CAM_ID="${CAM_ID:-4280}"
REFERER="${REFERER:-https://www.earthcam.com/}"

STREAM=$(streamcount find-stream "$PAGE" --cam-id "$CAM_ID")
echo "stream: ${STREAM:0:90}..."

streamcount run \
  --url "$STREAM" \
  --headers "Referer=$REFERER" \
  --engine yolo --tiles 2 --conf 0.25 \
  --interval 2 --frames 1800 \
  --flow --flow-min-hits 3 --flow-min-move 40 \
  --vlm-check 60 \
  --annotate --tag earthcam-hour
