#!/usr/bin/env python3
"""Render an animated per-minute chart GIF from a run's events CSV.

Usage:
    python scripts/render_hour_gif.py <events.csv> <out.gif>

The animation replays the run minute by minute: bars grow as each minute of
passers-by is counted, the cumulative line advances. All data comes from the
events file — nothing is simulated. Palette kept small so the GIF stays light.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamcount.report import load_events, passes_per_minute  # noqa: E402

WIDTH, HEIGHT = 900, 300
MARGIN_L, MARGIN_B, MARGIN_T, MARGIN_R = 48, 34, 26, 16
BAR_RGB = (255, 176, 46)
LINE_RGB = (120, 200, 255)
GRID_RGB = (45, 45, 50)
TEXT_RGB = (220, 220, 220)


def render(events_file: Path, out_path: Path, frame_ms: int = 120) -> Path:
    events = load_events(events_file)
    per_minute = passes_per_minute(events)
    if not per_minute:
        raise SystemExit("no events found in the given file")
    total_all = sum(per_minute)
    peak = max(per_minute) or 1

    plot_w = WIDTH - MARGIN_L - MARGIN_R
    plot_h = HEIGHT - MARGIN_T - MARGIN_B
    bar_w = max(3, plot_w // len(per_minute))

    frames: list[Image.Image] = []
    for upto in range(1, len(per_minute) + 1):
        visible = per_minute[:upto]
        image = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 20))
        draw = ImageDraw.Draw(image)
        draw.line([(MARGIN_L, MARGIN_T), (MARGIN_L, MARGIN_T + plot_h)], fill=(90, 90, 96))
        draw.line([(MARGIN_L, MARGIN_T + plot_h), (WIDTH - MARGIN_R, MARGIN_T + plot_h)],
                  fill=(90, 90, 96))
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = MARGIN_T + plot_h - int(plot_h * frac)
            draw.line([(MARGIN_L, y), (WIDTH - MARGIN_R, y)], fill=GRID_RGB)
            draw.text((6, y - 6), str(round(peak * frac)), fill=(150, 150, 150))

        running = 0
        points = []
        for i, value in enumerate(visible):
            x = MARGIN_L + i * bar_w
            bar_h = int(plot_h * value / peak)
            draw.rectangle(
                [x + 1, MARGIN_T + plot_h - bar_h, x + bar_w - 2, MARGIN_T + plot_h],
                fill=BAR_RGB,
            )
            running += value
            points.append((x + bar_w // 2,
                           MARGIN_T + plot_h - int(plot_h * running / total_all)))
        if len(points) > 1:
            draw.line(points, fill=LINE_RGB, width=2)

        draw.text((MARGIN_L, 7),
                  f"passers-by per minute — minute {upto}/{len(per_minute)}  "
                  f"(so far {running} of {total_all}, peak {peak}/min)",
                  fill=TEXT_RGB)
        draw.text((MARGIN_L, HEIGHT - 22), "minute", fill=(150, 150, 150))
        frames.append(image.quantize(colors=64, method=Image.MEDIANCUT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=frame_ms, loop=0, optimize=True)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    path = render(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"gif: {path} ({path.stat().st_size // 1024} KB)")
