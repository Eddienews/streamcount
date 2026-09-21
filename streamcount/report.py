"""Post-run reporting: per-minute rates, peaks and a simple chart (Pillow only).

Reads a finished run folder (events.csv + summary.json) and answers the questions a
stakeholder actually asks: how many passed in total, how fast, when were the peaks,
and what does the curve look like minute by minute.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def _find_events_file(run_dir: Path) -> Path | None:
    """Locate the events CSV: `events.csv`, a legacy `*_eventos.csv`, any CSV with a
    `t_rel_s` + `total` header, or a direct path to the file itself."""
    run_dir = Path(run_dir)
    if run_dir.is_file():
        return run_dir
    candidate = run_dir / "events.csv"
    if candidate.exists():
        return candidate
    for path in sorted(run_dir.glob("*.csv")):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                header = handle.readline()
        except OSError:
            continue
        if "t_rel_s" in header and "total" in header:
            return path
    return None


def load_events(run_dir: Path) -> list[dict]:
    """Load a run's events CSV (one row per counted passer-by)."""
    path = _find_events_file(Path(run_dir))
    if path is None:
        return []
    events: list[dict] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                events.append({"t": float(row["t_rel_s"]), "id": int(row["id"])})
            except (KeyError, ValueError):
                continue
    return events


def passes_per_minute(events: list[dict], duration_s: float | None = None) -> list[int]:
    """Bucket pass events into one-minute bins (bin 0 = first minute)."""
    if not events:
        return []
    last = duration_s if duration_s is not None else max(e["t"] for e in events)
    buckets = Counter(int(e["t"] // 60) for e in events)
    total_bins = int(last // 60) + 1
    return [buckets.get(i, 0) for i in range(total_bins)]


def draw_chart(per_minute: list[int], out_path: Path, width: int = 960, height: int = 320) -> Path:
    """Bars per minute + cumulative line. Dependency-light (Pillow)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    margin_l, margin_b, margin_t, margin_r = 48, 34, 22, 16
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(image)

    if not per_minute:
        draw.text((margin_l, margin_t), "no events to plot", fill=(200, 200, 200))
        image.save(out_path)
        return out_path

    peak = max(per_minute) or 1
    n = len(per_minute)
    bar_w = max(2, plot_w // max(n, 1))

    # axes
    draw.line([(margin_l, margin_t), (margin_l, margin_t + plot_h)], fill=(90, 90, 96))
    draw.line([(margin_l, margin_t + plot_h), (width - margin_r, margin_t + plot_h)], fill=(90, 90, 96))
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h - int(plot_h * frac)
        draw.line([(margin_l, y), (width - margin_r, y)], fill=(45, 45, 50))
        draw.text((6, y - 6), str(round(peak * frac)), fill=(150, 150, 150))

    # bars + cumulative line
    total = sum(per_minute)
    cumulative = 0
    points = []
    for i, value in enumerate(per_minute):
        x = margin_l + i * bar_w
        bar_h = int(plot_h * value / peak)
        draw.rectangle(
            [x + 1, margin_t + plot_h - bar_h, x + bar_w - 2, margin_t + plot_h],
            fill=(255, 176, 46),
        )
        cumulative += value
        y_cum = margin_t + plot_h - int(plot_h * cumulative / total)
        points.append((x + bar_w // 2, y_cum))
    if len(points) > 1:
        draw.line(points, fill=(120, 200, 255), width=2)

    draw.text((margin_l, 6), f"passers-by per minute  (peak {peak}/min, total {total})",
              fill=(230, 230, 230))
    draw.text((margin_l, height - 22), "minute", fill=(150, 150, 150))
    image.save(out_path)
    return out_path


def build_report(run_dir: Path, chart_path: Path | None = None) -> dict:
    """Summarise a run: totals, duration, per-minute rates, peak minute, chart."""
    run_dir = Path(run_dir)
    summary_file = run_dir / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    events = load_events(run_dir)
    duration = summary.get("duration_s")
    per_minute = passes_per_minute(events, duration)
    report = {
        "run_dir": str(run_dir),
        "passes": summary.get("passes", len(events)),
        "duration_s": duration,
        "duration_min": round(duration / 60, 1) if duration else None,
        "passes_per_minute": per_minute,
        "peak_per_minute": max(per_minute) if per_minute else 0,
        "mean_per_minute": round(sum(per_minute) / len(per_minute), 1) if per_minute else 0,
        "recall": summary.get("detector_recall"),
        "passes_recall_corrected": summary.get("passes_recall_corrected"),
    }
    if chart_path:
        draw_chart(per_minute, Path(chart_path))
        report["chart"] = str(chart_path)
    return report
