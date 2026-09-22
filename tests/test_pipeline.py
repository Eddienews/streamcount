"""Annotated-frame housekeeping: index parsing and on-disk retention (offline)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from streamcount.pipeline import _Recorder, frame_index, prune_frames


def _frames(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_bytes(b"jpeg")


def test_frame_index_parses_our_names():
    assert frame_index(Path("f0007_flow.jpg")) == 7
    assert frame_index(Path("f10000_yolo.jpg")) == 10000
    assert frame_index(Path("chart.png")) is None


def test_prune_keeps_only_the_last_n_frames(tmp_path: Path):
    _frames(tmp_path, *[f"f{i:04d}_flow.jpg" for i in range(1, 13)])
    assert prune_frames(tmp_path, 3) == 9
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "f0010_flow.jpg", "f0011_flow.jpg", "f0012_flow.jpg"]


def test_recorder_close_reaps_encoder_without_flush_error(tmp_path: Path):
    """Regression: closing stdin then communicate() raised ValueError on Linux

    ("flush of closed file"). Windows tolerated it, so only the CI e2e caught it.
    """
    stub = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    recorder = _Recorder(tmp_path / "timelapse.mp4", 12.0)
    recorder._proc = stub  # stand-in for a live ffmpeg encoder; no ffmpeg needed here
    stub.stdin.write(b"\x00" * 16)
    stub.stdin.flush()
    recorder.close()
    assert stub.stdin is None, "stdin must be detached before communicate()"
    assert stub.returncode == 0, "the encoder must be reaped cleanly"
    assert recorder._proc is None


def test_prune_keeps_all_engines_of_a_kept_frame(tmp_path: Path):
    _frames(tmp_path, "f0009_yolo.jpg", "f0009_vlm.jpg", "f0010_yolo.jpg", "f0010_vlm.jpg")
    assert prune_frames(tmp_path, 1) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f0010_vlm.jpg", "f0010_yolo.jpg"]


def test_prune_prefers_the_numeric_index(tmp_path: Path):
    _frames(tmp_path, "f9999_flow.jpg", "f10000_flow.jpg")
    prune_frames(tmp_path, 1)
    assert [p.name for p in tmp_path.iterdir()] == ["f10000_flow.jpg"]


def test_prune_noop_for_keep_zero_or_more_than_present(tmp_path: Path):
    _frames(tmp_path, "f0001_flow.jpg", "f0002_flow.jpg")
    assert prune_frames(tmp_path, 0) == 0
    assert prune_frames(tmp_path, 5) == 0
    assert len(list(tmp_path.iterdir())) == 2


def test_prune_leaves_foreign_files_alone(tmp_path: Path):
    _frames(tmp_path, "f0001_flow.jpg", "f0002_flow.jpg", "readme.txt")
    assert prune_frames(tmp_path, 1) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f0002_flow.jpg", "readme.txt"]
