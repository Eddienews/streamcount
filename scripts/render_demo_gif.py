#!/usr/bin/env python3
"""Render the README demo GIF from a run's annotated frames.

Usage:
    python scripts/render_demo_gif.py <run_dir> <out.gif> [--frames 36] [--width 720]

Samples the run's annotated frames evenly (newest frames kept), scales them down
and writes an optimised looping GIF — the asset shown in the README. All frames
come from a real run; nothing is simulated. Needs Pillow only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def pick_frames(run_dir: Path, want: int) -> list[Path]:
    frames_dir = run_dir / "frames"
    frames = sorted(
        (p for p in frames_dir.iterdir() if p.suffix == ".jpg"),
        key=lambda p: int("".join(ch for ch in p.name.split("_")[0] if ch.isdigit()) or 0),
    )
    if not frames:
        raise SystemExit(f"no annotated frames in {frames_dir}")
    if len(frames) <= want:
        return frames
    step = len(frames) / want
    return [frames[min(int(i * step), len(frames) - 1)] for i in range(want)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--frames", type=int, default=36, help="frames in the GIF (default 36)")
    parser.add_argument("--width", type=int, default=720, help="output width in px (default 720)")
    parser.add_argument("--ms", type=int, default=130, help="per-frame duration in ms (default 130)")
    parser.add_argument("--colors", type=int, default=128,
                        help="palette size per frame (default 128; 0 keeps truecolour quantisation)")
    args = parser.parse_args()

    picked = pick_frames(args.run_dir, args.frames)
    images: list[Image.Image] = []
    for path in picked:
        image = Image.open(path).convert("RGB")
        height = round(image.height * args.width / image.width)
        height -= height % 2  # keep even dimensions for video encoders and viewers
        frame = image.resize((args.width, height), Image.LANCZOS)
        if args.colors:
            frame = frame.quantize(colors=args.colors, method=Image.MEDIANCUT)
        images.append(frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=args.ms,
        loop=0,
        optimize=True,
    )
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"{args.out}: {len(images)} frames @ {args.ms}ms, {args.width}px wide, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
