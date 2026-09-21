"""Reporting helpers (offline)."""

import json
from pathlib import Path

from PIL import Image

from streamcount.report import build_report, draw_chart, load_events, passes_per_minute


def _write_run(run_dir: Path, events: list[tuple[float, int]], duration_s: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "events.csv", "w", encoding="utf-8") as handle:
        handle.write("ts_iso,t_rel_s,event,id,hits,disp_px,total\n")
        for i, (t, track_id) in enumerate(events, 1):
            handle.write(f"2026-01-01T00:00:{min(i, 59):02d}, {t}, pass, {track_id}, 3, 50, {i}\n")
    (run_dir / "summary.json").write_text(
        json.dumps({"passes": len(events), "duration_s": duration_s,
                    "detector_recall": 0.46, "passes_recall_corrected": round(len(events) / 0.46)}),
        encoding="utf-8",
    )


def test_passes_per_minute_buckets():
    events = [{"t": 5.0, "id": 1}, {"t": 65.0, "id": 2}, {"t": 70.0, "id": 3}]
    assert passes_per_minute(events, duration_s=185.0) == [1, 2, 0, 0]


def test_passes_per_minute_empty():
    assert passes_per_minute([]) == []


def test_load_events_tolerates_junk(tmp_path):
    (tmp_path / "events.csv").write_text(
        "ts_iso,t_rel_s,event,id,hits,disp_px,total\n"
        "x, 12.0, pass, 7, 3, 60, 1\n"
        "x, not_a_number, pass, 8, 3, 60, 2\n",
        encoding="utf-8",
    )
    events = load_events(tmp_path)
    assert events == [{"t": 12.0, "id": 7}]


def test_load_events_accepts_legacy_eventos_name(tmp_path):
    (tmp_path / "20260101_x_yolo_eventos.csv").write_text(
        "ts_iso,t_rel_s,evento,id,hits,disp_px,total_passaram\n"
        "x, 30.0, passou, 5, 3, 70, 1\n",
        encoding="utf-8",
    )
    assert load_events(tmp_path) == [{"t": 30.0, "id": 5}]


def test_load_events_accepts_direct_file(tmp_path):
    events_file = tmp_path / "whatever.csv"
    events_file.write_text(
        "ts_iso,t_rel_s,event,id,hits,disp_px,total\nx, 5.0, pass, 1, 3, 50, 1\n",
        encoding="utf-8",
    )
    assert load_events(events_file) == [{"t": 5.0, "id": 1}]


def test_build_report_and_chart(tmp_path):
    _write_run(tmp_path, [(10.0, 1), (75.0, 2), (80.0, 3), (130.0, 4)], duration_s=185.0)
    chart = tmp_path / "chart.png"
    report = build_report(tmp_path, chart_path=chart)
    assert report["passes"] == 4
    assert report["passes_per_minute"] == [1, 2, 1, 0]
    assert report["peak_per_minute"] == 2
    assert report["recall"] == 0.46
    assert chart.exists() and chart.stat().st_size > 1000
    with Image.open(chart) as image:
        assert image.size[0] >= 400


def test_draw_chart_without_events(tmp_path):
    out = draw_chart([], tmp_path / "empty.png")
    assert out.exists()
