"""Live dashboard: source detection, supervisor plumbing and the HTTP layer.

All offline: the HTTP tests use a stub supervisor, so no ffmpeg, no model, no key.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from streamcount.webapp import PAGE_HTML, Supervisor, detect_source, make_handler


# --------------------------------------------------------------------------- #
# Source detection
# --------------------------------------------------------------------------- #
def test_detect_youtube():
    flag, _ = detect_source("https://www.youtube.com/watch?v=abc123")
    assert flag == "--youtube"


def test_detect_youtube_short_link():
    flag, _ = detect_source("https://youtu.be/abc123")
    assert flag == "--youtube"


def test_detect_rtsp_and_hls():
    assert detect_source("rtsp://192.168.1.50:554/stream1")[0] == "--url"
    assert detect_source("https://videos.example.com/live/playlist.m3u8?t=abc")[0] == "--url"


def test_detect_local_file_and_dir(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    assert detect_source(str(video))[0] == "--video"
    assert detect_source(str(tmp_path))[0] == "--images"


def test_detect_rejects_garbage():
    with pytest.raises(ValueError):
        detect_source("aquilo ali do lado")
    with pytest.raises(ValueError):
        detect_source("   ")


# --------------------------------------------------------------------------- #
# Page + supervisor internals
# --------------------------------------------------------------------------- #
def test_page_has_the_essentials():
    for marker in ("passers-by", "/api/status", "/api/start", "/api/stop", "/latest.jpg"):
        assert marker in PAGE_HTML
    assert "e8a33d" in PAGE_HTML, "amber accent must survive edits"


def test_tail_frames_parses_last_row(tmp_path: Path):
    csv = tmp_path / "frames.csv"
    csv.write_text(
        "ts_iso,frame,engine,count,latency_ms,source,active_tracks,passes_total,vlm_count,notes\n"
        "2026-09-21T21:16:05,1,yolo,8,144,src,7,\n"
        "2026-09-21T21:16:09,40,yolo,20,906,src,31,28,,\n",
        encoding="utf-8",
    )
    status = Supervisor._tail_frames(csv)
    assert status["frames"] == 40 and status["count"] == 20
    assert status["active"] == 31 and status["passes"] == 28 and status["latency_ms"] == 906


def test_passers_per_minute_buckets():
    assert Supervisor.passers_per_minute([]) == []
    assert Supervisor.passers_per_minute([5.0, 10.0]) == [2]
    assert Supervisor.passers_per_minute([10.0, 65.0, 130.0]) == [1, 1, 1]


def test_read_events(tmp_path: Path):
    csv = tmp_path / "events.csv"
    csv.write_text(
        "ts_iso,t_rel_s,event,id,hits,disp_px,total\n"
        "2026-09-21T21:15:29,4.0,pass,5,3,51,1\n"
        "2026-09-21T21:15:41,16.0,pass,9,3,88,2\n",
        encoding="utf-8",
    )
    events, times = Supervisor._read_events(csv)
    assert times == [4.0, 16.0]
    assert events[0]["id"] == "5" and events[1]["disp_px"] == "88"


def test_build_run_command_is_live_and_annotated(tmp_path: Path):
    from streamcount.webapp import build_run_command

    cmd = build_run_command("py", tmp_path, "webabc", "--url", "rtsp://cam/1",
                            {"tiles": 2, "interval": 2})
    assert "--annotate" in cmd and "--flow" in cmd
    assert cmd[cmd.index("--frames") + 1] == "999999", "dashboard runs are live sessions"
    assert cmd[cmd.index("--tiles") + 1] == "2"
    assert cmd[cmd.index("--target") + 1] == "people"


def test_build_run_command_respects_explicit_frames(tmp_path: Path):
    from streamcount.webapp import build_run_command

    cmd = build_run_command("py", tmp_path, "t", "--youtube", "https://youtu.be/x",
                            {"frames": 30, "tiles": 1, "interval": 0.2})
    assert cmd[cmd.index("--frames") + 1] == "30"
    assert "--tiles" not in cmd, "tiles=1 means off"
    assert cmd[cmd.index("--interval") + 1] == "1", "interval is clamped to >= 1 s"
    assert cmd[cmd.index("--engine") + 1] == "yolo"


# --------------------------------------------------------------------------- #
# HTTP layer (stub supervisor — nothing is spawned)
# --------------------------------------------------------------------------- #
class StubSupervisor:
    def __init__(self) -> None:
        self.started: tuple | None = None
        self.stops = 0

    def start(self, source: str, **options) -> dict:
        if source == "boom":
            raise ValueError("nope")
        self.started = (source, options)
        return {"ok": True, "tag": "webabc"}

    def stop(self) -> dict:
        self.stops += 1
        return {"ok": True}

    def status(self) -> dict:
        return {"running": False, "passes": 7, "per_minute": [1, 2]}

    def _find_run_dir(self):
        return None


@pytest.fixture()
def dashboard():
    stub = StubSupervisor()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(stub))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", stub
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - localhost
        return response.status, response.read()


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        return response.status, json.loads(response.read())


def test_page_route(dashboard):
    base, _ = dashboard
    status, body = _get(base + "/")
    assert status == 200 and b"streamcount" in body


def test_status_route(dashboard):
    base, _ = dashboard
    status, body = _get(base + "/api/status")
    assert status == 200
    assert json.loads(body)["passes"] == 7


def test_start_route_passes_source_and_options(dashboard):
    base, stub = dashboard
    status, body = _post(base + "/api/start",
                         {"source": "rtsp://cam/1", "options": {"tiles": 2, "interval": 3}})
    assert status == 200 and body["ok"] is True
    assert stub.started == ("rtsp://cam/1", {"tiles": 2, "interval": 3})


def test_start_route_reports_bad_source(dashboard):
    base, _ = dashboard
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base + "/api/start", {"source": "boom"})
    assert excinfo.value.code == 400


def test_stop_route(dashboard):
    base, stub = dashboard
    status, _ = _post(base + "/api/stop", {})
    assert status == 200 and stub.stops == 1


def test_latest_frame_route_without_run(dashboard):
    base, _ = dashboard
    status, body = _get(base + "/latest.jpg")
    assert status == 204 and body == b""


def test_routes_ignore_query_strings(dashboard):
    """The page appends ?t= cache busters to /latest.jpg — routing must ignore the query."""
    base, _ = dashboard
    status, _ = _get(base + "/latest.jpg?t=1790041041770")
    assert status == 204, "cache-busted frame URL must not 404"
    status, body = _get(base + "/api/status?t=1")
    assert status == 200 and json.loads(body)["passes"] == 7


def test_unknown_route_404(dashboard):
    base, _ = dashboard
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base + "/nope")
    assert excinfo.value.code == 404


def test_servers_bind_both_loopbacks():
    """Browsers may resolve localhost to ::1 — the dashboard must answer on both."""
    from streamcount.webapp import make_servers

    servers, errors = make_servers(0, make_handler(StubSupervisor()))
    try:
        families = {server.address_family for server in servers}
        assert socket.AF_INET in families, f"IPv4 loopback missing: {errors}"
        assert socket.AF_INET6 in families, f"IPv6 loopback missing: {errors}"
    finally:
        for server in servers:
            server.server_close()


def test_port_already_in_use_fails_loudly():
    """allow_reuse_address must stay off: a second instance may not hijack the port."""
    from streamcount.webapp import make_servers

    servers, _ = make_servers(0, make_handler(StubSupervisor()))
    try:
        assert servers, "need a first server for this test"
        port = servers[0].server_address[1]
        second, errors = make_servers(port, make_handler(StubSupervisor()))
        try:
            assert not any(s.address_family == socket.AF_INET for s in second)
            assert any("127.0.0.1" in line for line in errors)
        finally:
            for server in second:
                server.server_close()
    finally:
        for server in servers:
            server.server_close()
