"""End-to-end CLI smoke test: real ffmpeg, real ONNX model, real files on disk.

Marked ``e2e`` because it needs ffmpeg on PATH and downloads the default model once.
The synthetic video is a moving test pattern — this suite validates the *plumbing*
(sampling through the ffmpeg pipe, detection, tracking, CSV/summary/chart writing,
exit codes, the report command), not detection quality (covered in docs/MEASUREMENTS.md).

Local run:  pytest -m e2e
CI run:     the `e2e` job installs ffmpeg first (see .github/workflows/ci.yml)
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "streamcount", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def synthetic_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    video = tmp_path / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=20:size=640x360:rate=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    return video


def test_cli_flow_run_end_to_end(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    result = _cli(
        ["run", "--video", str(synthetic_video), "--frames", "3", "--interval", "1",
         "--flow", "--annotate", "--keep-frames", "2", "--timelapse", "--out", str(out)],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    run_dirs = list(out.iterdir())
    assert len(run_dirs) == 1, f"expected one run folder, got {run_dirs}"
    run_dir = run_dirs[0]

    rows = list(csv.DictReader((run_dir / "frames.csv").open(encoding="utf-8")))
    assert len(rows) == 3, "one CSV row per sampled frame expected"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 3
    assert summary["flow"] is True
    assert (run_dir / "events.csv").exists()
    assert (run_dir / "chart.png").exists(), "flow runs must generate a chart"
    assert (run_dir / "timelapse.mp4").exists(), "--timelapse must write an MP4"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(run_dir / "timelapse.mp4")],
        capture_output=True, text=True,
    )
    assert float(probe.stdout.strip()) > 0, "the timelapse must have a positive duration"

    kept = sorted(p.name for p in (run_dir / "frames").glob("*.jpg"))
    assert kept == ["f0002_flow.jpg", "f0003_flow.jpg"], \
        "--keep-frames 2 must leave only the newest annotated frames on disk"


def test_cli_report_on_finished_run(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    result = _cli(
        ["run", "--video", str(synthetic_video), "--frames", "2", "--interval", "1",
         "--flow", "--out", str(out)],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    run_dir = next(out.iterdir())

    chart = tmp_path / "report.png"
    report = _cli(["report", str(run_dir), "--chart", str(chart)], cwd=tmp_path)
    # a test pattern has no passers-by: an empty report is a valid answer (exit 0)
    assert report.returncode == 0, report.stderr
    assert "passers-by: 0" in report.stdout
    assert chart.exists()


def test_cli_run_without_source_fails_clearly(tmp_path: Path) -> None:
    result = _cli(["run"], cwd=tmp_path)
    assert result.returncode == 2
    assert "provide a source" in result.stderr


def test_cli_run_with_unreadable_source_exits_3(tmp_path: Path) -> None:
    """A source that yields zero frames must fail loudly, not look like an empty run."""
    result = _cli(
        ["run", "--video", str(tmp_path / "does-not-exist.mp4"), "--frames", "2",
         "--out", str(tmp_path / "runs")],
        cwd=tmp_path,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "0 frames received" in result.stderr
