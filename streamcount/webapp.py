"""Local live dashboard: paste a link, watch the count in real time.

A thin supervisor around `streamcount run`: it launches the run as a child process,
tails the run's CSVs and annotated frames, and serves a single-page app on 127.0.0.1.
No extra dependencies (stdlib http.server), no data leaves the machine.

Routes (all bound to localhost):
    GET  /              single-page dashboard
    GET  /api/status    run state as JSON (counts, per-minute series, log tail)
    GET  /latest.jpg    newest annotated frame (no-store)
    POST /api/start     {"source": "<link>", ...options} -> start a run
    POST /api/stop      stop the current run
"""

from __future__ import annotations

import csv
import json
import secrets
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .sources import is_youtube

LATEST_FRAME_SUFFIXES = ("_flow.jpg", "_yolo.jpg", "_vlm.jpg", "_grid.jpg")

# --------------------------------------------------------------------------- #
# Source detection
# --------------------------------------------------------------------------- #
def detect_source(value: str) -> tuple[str, str]:
    """Map what the user pasted to (flag, argument).

    Raises ValueError with a human message (the page shows it verbatim).
    """
    text = (value or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError("paste a link or a file path")
    lowered = text.lower()
    if is_youtube(text):
        return "--youtube", text
    if lowered.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return "--url", text
    path = Path(text)
    if path.is_dir():
        return "--images", str(path)
    if path.is_file():
        return "--video", str(path)
    raise ValueError("not a recognised link (http/rtsp/youtube) or an existing file")


def build_run_command(python: str, runs_root: Path, tag: str, source_flag: str,
                      source_arg: str, options: dict) -> list[str]:
    """Build the `streamcount run` argv for a dashboard run (pure, unit-testable)."""
    cmd = [python, "-m", "streamcount", "run", source_flag, source_arg,
           "--out", str(runs_root), "--tag", tag, "--annotate"]
    # a dashboard run only ever shows the newest frame; cap the rest (a live session would
    # otherwise write ~320 MB/h of annotated JPEGs). 0 = keep every frame.
    keep = options.get("keep_frames", 10)
    cmd += ["--keep-frames", str(max(0, 10 if keep is None else int(keep)))]
    if options.get("timelapse"):
        fps = float(options.get("timelapse_fps") or 12)
        cmd += ["--timelapse", "--timelapse-fps", f"{max(fps, 1):g}"]
    interval = float(options.get("interval") or 2.0)
    cmd += ["--interval", f"{max(interval, 1.0):g}"]
    tiles = int(options.get("tiles") or 2)
    if tiles > 1:
        cmd += ["--tiles", str(tiles)]
    engine = options.get("engine") or "yolo"
    if engine in ("yolo", "vlm", "both"):
        cmd += ["--engine", engine]
    conf = options.get("conf")
    if conf:
        cmd += ["--conf", f"{float(conf):g}"]
    # a dashboard run is a live session: it stops when the user presses stop, not after
    # the CLI default (10 frames on streams)
    cmd += ["--frames", str(int(options.get("frames") or 999_999))]
    target = options.get("target") or "people"
    cmd += ["--target", target]
    if target == "cars":
        from .detector import ensure_model

        cmd += ["--model", str(ensure_model("yolo11n"))]
    if options.get("headers"):
        cmd += ["--headers", str(options["headers"])]
    if options.get("flow", True):
        cmd += ["--flow"]
        vlm_check = float(options.get("vlm_check") or 0)
        if vlm_check > 0:
            cmd += ["--vlm-check", f"{vlm_check:g}"]
            if options.get("jev_router"):
                cmd += ["--jev-router"]
    return cmd


# --------------------------------------------------------------------------- #
# Run supervisor
# --------------------------------------------------------------------------- #
class Supervisor:
    """Owns at most one `streamcount run` child process and reports on it."""

    def __init__(self, runs_root: Path, python: str | None = None) -> None:
        self.runs_root = Path(runs_root)
        self.python = python or sys.executable
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._tag: str | None = None
        self._run_dir: Path | None = None
        self._log_path: Path | None = None
        self._log_handle = None
        self._source = ""
        self._options: dict = {}
        self._started_at = 0.0

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, source: str, **options) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise ValueError("a run is already in progress — stop it first")
            flag, arg = detect_source(source)
            tag = f"web{secrets.token_hex(3)}"
            self.runs_root.mkdir(parents=True, exist_ok=True)
            self._log_path = self.runs_root / f"{tag}.log"
            cmd = build_run_command(self.python, self.runs_root, tag, flag, arg, options)
            interval = float(options.get("interval") or 2.0)
            tiles = int(options.get("tiles") or 2)
            engine = options.get("engine") or "yolo"
            conf = options.get("conf")
            target = options.get("target") or "people"
            self._log_handle = self._log_path.open("w", encoding="utf-8", errors="replace")
            self._proc = subprocess.Popen(
                cmd, stdout=self._log_handle, stderr=subprocess.STDOUT, text=True
            )
            self._tag = tag
            self._run_dir = None
            self._source = arg if flag != "--images" else arg
            self._options = {"interval": interval, "tiles": tiles, "engine": engine,
                             "conf": conf, "target": target}
            self._started_at = time.monotonic()
            return {"ok": True, "tag": tag, "command": " ".join(cmd)}

    def stop(self) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return {"ok": True, "note": "no run in progress"}
            proc = self._proc
            run_dir = self._find_run_dir()
            if run_dir is not None:
                # graceful: the pipeline sees this, leaves the frame loop and still writes
                # summary.json + chart.png (the timelapse encoder closes properly, too)
                (run_dir / "STOP").write_text("stop\n", encoding="utf-8")
        if run_dir is not None:
            try:
                proc.wait(timeout=25)
                return {"ok": True, "note": "run stopped"}
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass
        if proc.poll() is None:
            proc.terminate()  # no run dir yet, or it did not stop in time
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                proc.kill()
        return {"ok": True, "note": "run stopped (forced)"}

    # -- introspection ------------------------------------------------------ #
    def _find_run_dir(self) -> Path | None:
        if self._run_dir and self._run_dir.exists():
            return self._run_dir
        if not self._tag:
            return None
        candidates = sorted(
            self.runs_root.glob(f"*_{self._tag}"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        self._run_dir = candidates[0] if candidates else None
        return self._run_dir

    @staticmethod
    def _tail_frames(csv_path: Path) -> dict:
        out: dict = {}
        if not csv_path.exists():
            return out
        try:
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8", errors="replace")))
        except OSError:
            return out
        if not rows:
            return out
        last = rows[-1]
        out["frames"] = int(last.get("frame") or 0)
        out["count"] = int(last.get("count") or 0)
        out["latency_ms"] = int(float(last.get("latency_ms") or 0))
        out["active"] = int(last.get("active_tracks") or 0)
        out["passes"] = int(last.get("passes_total") or 0)
        out["vlm_count"] = last.get("vlm_count") or None
        return out

    @staticmethod
    def _read_events(csv_path: Path) -> tuple[list[dict], list[float]]:
        events: list[dict] = []
        times: list[float] = []
        if not csv_path.exists():
            return events, times
        try:
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8", errors="replace")))
        except OSError:
            return events, times
        for row in rows:
            try:
                t_rel = float(row.get("t_rel_s") or 0)
            except ValueError:
                continue
            times.append(t_rel)
            events.append({"ts": row.get("ts_iso", ""), "t_rel_s": t_rel,
                           "id": row.get("id", ""), "hits": row.get("hits", ""),
                           "disp_px": row.get("disp_px", "")})
        return events, times

    @staticmethod
    def passers_per_minute(times: list[float]) -> list[int]:
        if not times:
            return []
        total = max(times)
        buckets = [0] * (int(total // 60) + 1)
        for t in times:
            buckets[int(t // 60)] += 1
        return buckets

    def status(self) -> dict:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            exit_code = None if running or self._proc is None else self._proc.returncode
            run_dir = self._find_run_dir()
            payload: dict = {
                "running": running,
                "exit_code": exit_code,
                "source": self._source,
                "options": self._options,
                "elapsed_s": round(time.monotonic() - self._started_at, 1) if self._started_at else 0,
                "frames": 0, "count": 0, "active": 0, "passes": 0, "latency_ms": 0,
                "vlm_count": None, "per_minute": [], "last_events": [], "log_tail": [],
                "run_dir": str(run_dir) if run_dir else None,
                "finished": False, "timelapse": False,
            }
            if run_dir:
                payload.update(self._tail_frames(run_dir / "frames.csv"))
                events, times = self._read_events(run_dir / "events.csv")
                payload["per_minute"] = self.passers_per_minute(times)
                payload["last_events"] = events[-6:][::-1]
                payload["finished"] = (run_dir / "summary.json").exists()
                payload["timelapse"] = (run_dir / "timelapse.mp4").exists()
            if self._log_path and self._log_path.exists():
                try:
                    tail = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    payload["log_tail"] = tail[-8:]
                except OSError:
                    pass
            return payload


def latest_frame(frames_dir: Path) -> Path | None:
    """Newest annotated frame in a run's frames/ dir (numeric index, not name order)."""
    from .pipeline import frame_index  # lazy: keep the web server's import list light

    frames = [p for p in frames_dir.iterdir() if p.suffix == ".jpg"]
    if not frames:
        return None

    def sort_key(path: Path) -> tuple[int, str]:
        index = frame_index(path)
        return (index if index is not None else -1, path.name)

    return max(frames, key=sort_key)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
def make_handler(supervisor: Supervisor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "streamcount"
        protocol_version = "HTTP/1.1"

        # -- helpers ------------------------------------------------------- #
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload: dict, code: int = 200) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _error(self, code: int, message: str) -> None:
            self._json({"ok": False, "error": message}, code=code)

        def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
            pass

        def _route(self) -> str:
            """Path without the query string (the page appends ?t= cache busters)."""
            return urlsplit(self.path).path

        # -- routes -------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802
            route = self._route()
            if route in ("/", "/index.html"):
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route == "/api/status":
                self._json(supervisor.status())
                return
            if route == "/latest.jpg":
                run_dir = supervisor._find_run_dir()  # noqa: SLF001 - same module
                frame = None
                if run_dir and (run_dir / "frames").exists():
                    frame = latest_frame(run_dir / "frames")
                if frame is None:
                    self._send(204, b"", "image/jpeg")
                    return
                self._send(200, frame.read_bytes(), "image/jpeg")
                return
            if route == "/timelapse.mp4":
                run_dir = supervisor._find_run_dir()  # noqa: SLF001 - same module
                video = run_dir / "timelapse.mp4" if run_dir else None
                if video is None or not video.exists():
                    self._send(204, b"", "video/mp4")
                    return
                self._send(200, video.read_bytes(), "video/mp4")
                return
            self._error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._error(400, "invalid JSON body")
                return
            route = self._route()
            if route == "/api/start":
                try:
                    result = supervisor.start(payload.get("source", ""), **payload.get("options", {}))
                except ValueError as exc:
                    self._error(400, str(exc))
                    return
                except Exception as exc:  # pragma: no cover - defensive
                    self._error(500, f"failed to start: {exc}")
                    return
                self._json(result)
                return
            if route == "/api/stop":
                self._json(supervisor.stop())
                return
            self._error(404, "not found")

    return Handler


class _DashboardServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a SECOND process hijack the same port (the "phantom
    # server" that kept serving old code). Refuse to share the port instead.
    allow_reuse_address = False


class _DashboardServerV6(_DashboardServer):
    address_family = socket.AF_INET6


def make_servers(port: int, handler) -> tuple[list[ThreadingHTTPServer], list[str]]:
    """Bind both loopback addresses (127.0.0.1 and ::1). Returns (servers, errors).

    Browsers happily pick IPv6 for `localhost`; serving only IPv4 makes them show
    ERR_CONNECTION_REFUSED depending on how the name resolves.
    """
    servers: list[ThreadingHTTPServer] = []
    errors: list[str] = []
    for cls, address, label in (
        (_DashboardServer, ("127.0.0.1", port), "127.0.0.1"),
        (_DashboardServerV6, ("::1", port), "[::1]"),
    ):
        try:
            servers.append(cls(address, handler))
        except OSError as exc:
            errors.append(f"{label}:{port} -> {exc.strerror or exc}")
    return servers, errors


def serve(port: int = 8766, runs_root: Path = Path("runs"), open_browser: bool = False) -> None:
    supervisor = Supervisor(runs_root)
    servers, errors = make_servers(port, make_handler(supervisor))
    if not servers:
        print("could not bind any loopback address to serve the dashboard:")
        for line in errors:
            print(f"  {line}")
        print("another `streamcount serve` may already be running — pick another --port.")
        return
    url = f"http://127.0.0.1:{port}/"
    print(f"streamcount dashboard: {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    extra = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers[1:]]
    for thread in extra:
        thread.start()
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        supervisor.stop()
        for server in servers:
            server.server_close()


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
PAGE_HTML = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>streamcount</title>
<style>
  :root {
    --bg: #14100c; --ink: #ece5da; --dim: #8c8172; --amber: #e8a33d; --line: #2b241c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.6 Georgia, "Times New Roman", serif;
    -webkit-font-smoothing: antialiased;
  }
  main { max-width: 880px; margin: 0 auto; padding: 48px 24px 96px; }
  header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 48px; }
  h1 { font-size: 22px; font-weight: normal; letter-spacing: .06em; margin: 0; }
  .status { font: 11px/1 -apple-system, Segoe UI, sans-serif; letter-spacing: .18em;
            text-transform: uppercase; color: var(--dim); }
  .status .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
                 background: var(--dim); margin-right: 7px; vertical-align: middle; }
  .status.live .dot { background: var(--amber); }
  .spacer { flex: 1; }
  #lang { font: 11px/1 -apple-system, Segoe UI, sans-serif; letter-spacing: .14em;
          text-transform: uppercase; color: var(--dim); cursor: pointer; }
  #lang:hover { color: var(--amber); }

  form { margin-bottom: 56px; }
  label.hint { display: block; font-size: 15px; color: var(--dim); margin-bottom: 10px; }
  .row { display: flex; align-items: baseline; gap: 18px; }
  .qrow { display: flex; align-items: baseline; gap: 34px; margin-top: 16px; }
  input[type=text] {
    flex: 1; background: transparent; border: 0; border-bottom: 1px solid var(--line);
    color: var(--ink); font: 17px/1.6 Georgia, serif; padding: 6px 2px;
  }
  input[type=text]:focus { outline: none; border-bottom-color: var(--amber); }
  button.link {
    background: none; border: 0; color: var(--amber); font: 16px/1 Georgia, serif;
    cursor: pointer; padding: 6px 0; white-space: nowrap;
  }
  button.link:hover { text-decoration: underline; }
  button.link.muted { color: var(--dim); }
  .meta { margin-top: 12px; font: 11px/1.8 -apple-system, Segoe UI, sans-serif;
          letter-spacing: .16em; text-transform: uppercase; color: var(--dim); }
  .meta b { color: var(--ink); font-weight: normal; }

  details { margin-top: 18px; }
  details summary { font: 11px/1 -apple-system, Segoe UI, sans-serif; letter-spacing: .16em;
                    text-transform: uppercase; color: var(--dim); cursor: pointer; }
  details .opts { display: flex; flex-wrap: wrap; gap: 26px; margin-top: 16px; }
  .opt { font: 12px/1.7 -apple-system, Segoe UI, sans-serif; color: var(--dim); }
  .opt input, .opt select {
    background: transparent; border: 0; border-bottom: 1px solid var(--line);
    color: var(--ink); font: 14px/1.6 -apple-system, Segoe UI, sans-serif; width: 84px;
  }
  .opt input.wide { width: 260px; }
  .opt select { width: auto; }
  .opt input[type=checkbox] { width: auto; border: 0; accent-color: var(--amber); vertical-align: -1px; }

  .numbers { display: flex; gap: 56px; flex-wrap: wrap; margin-bottom: 8px; }
  .num .label { font: 11px/1 -apple-system, Segoe UI, sans-serif; letter-spacing: .18em;
                text-transform: uppercase; color: var(--dim); margin-bottom: 10px; }
  .num .value { font: 46px/1 Georgia, serif; font-variant-numeric: tabular-nums; }
  .num.accent .value { color: var(--amber); }
  .submeta { font: 11px/1.8 -apple-system, Segoe UI, sans-serif; letter-spacing: .16em;
             text-transform: uppercase; color: var(--dim); margin: 18px 0 40px; }

  #frame { width: 100%; margin-bottom: 48px; display: none; }
  #chart { width: 100%; height: 90px; margin-bottom: 8px; }
  .chartlabel { font: 11px/1.8 -apple-system, Segoe UI, sans-serif; letter-spacing: .16em;
                text-transform: uppercase; color: var(--dim); margin-bottom: 40px; }

  h2 { font: 11px/1 -apple-system, Segoe UI, sans-serif; letter-spacing: .18em;
       text-transform: uppercase; color: var(--dim); font-weight: normal; margin: 0 0 14px; }
  table { width: 100%; border-collapse: collapse;
          font: 14px/1.7 -apple-system, Segoe UI, sans-serif; font-variant-numeric: tabular-nums; }
  td { padding: 3px 0; color: var(--ink); }
  td.dim { color: var(--dim); }
  #error { color: #e07a5f; font-size: 14px; margin-top: 14px; display: none; }
  #log { margin-top: 48px; font: 12px/1.8 ui-monospace, Consolas, monospace; color: var(--dim);
         white-space: pre-wrap; }
</style>
</head>
<body>
<main>
  <header>
    <h1>streamcount</h1>
    <span class="status" id="status"><span class="dot"></span><span id="statusText">idle</span></span>
    <span class="spacer"></span>
    <span id="lang">EN</span>
  </header>

  <form id="form" autocomplete="off">
    <label class="hint" id="hint">Cole o link do vídeo ao vivo ou da gravação</label>
    <div class="row">
      <input type="text" id="source" placeholder="https://…  ·  rtsp://…  ·  youtube.com/watch?v=…  ·  C:\\video.mp4">
      <button class="link" type="submit" id="go">contar →</button>
      <button class="link muted" type="button" id="stop" style="display:none">parar</button>
    </div>
    <div id="error"></div>
    <div class="qrow">
      <label class="opt" id="lTarget">alvo
        <select id="target"><option value="people">pessoas</option><option value="cars">carros</option></select>
      </label>
      <label class="opt" id="lMp4">gravar mp4 <input type="checkbox" id="mp4"> <input type="number" id="mp4fps" value="12" min="1" max="60" step="1"></label>
    </div>
    <details>
      <summary id="advSummary">ajustes</summary>
      <div class="opts">
        <label class="opt" id="lInterval">intervalo (s) <input type="number" id="interval" value="2" min="1" max="60" step="1"></label>
        <label class="opt">tiles <input type="number" id="tiles" value="2" min="1" max="3" step="1"></label>
        <label class="opt">motor
          <select id="engine"><option value="yolo">yolo (local)</option><option value="vlm">vlm (API)</option><option value="both">both</option></select>
        </label>
        <label class="opt">confiança <input type="number" id="conf" value="0.25" min="0.05" max="0.9" step="0.05"></label>
        <label class="opt" id="lKeep">guardar frames <input type="number" id="keep" value="10" min="0" max="999" step="1"></label>
        <label class="opt" id="lVlmCheck">vlm a cada (s) <input type="number" id="vlmcheck" value="0" min="0" max="3600" step="5"></label>
        <label class="opt" id="lJev">roteador (Jev) <input type="checkbox" id="jev"></label>
        <label class="opt">headers <input type="text" class="wide" id="headers" placeholder="Referer=https://www.earthcam.com/"></label>
      </div>
    </details>
  </form>

  <div class="numbers">
    <div class="num accent"><div class="label" id="lPasses">passers-by</div><div class="value" id="passes">0</div></div>
    <div class="num"><div class="label" id="lNow">na cena agora</div><div class="value" id="count">0</div></div>
    <div class="num"><div class="label" id="lActive">em movimento</div><div class="value" id="active">0</div></div>
    <div class="num"><div class="label" id="lRate">ritmo</div><div class="value" id="rate">—</div></div>
  </div>
  <div class="submeta" id="submeta">sem contagem ativa</div>

  <img id="frame" alt="">
  <canvas id="chart" width="880" height="90"></canvas>
  <div class="chartlabel" id="lPerMinute">passagens por minuto</div>
  <div class="chartlabel" id="mp4box" style="display:none"><a href="/timelapse.mp4" target="_blank" style="color: var(--amber); text-decoration: none;">&#9654; timelapse.mp4</a></div>

  <h2 id="lLast">últimas passagens</h2>
  <table><tbody id="events"><tr><td class="dim">—</td></tr></tbody></table>

  <div id="log"></div>
</main>
<script>
const T = {
  pt: { stopped:"parado", live:"contando", hint:"Cole o link do vídeo ao vivo ou da gravação",
        go:"contar →", stop:"parar", adv:"ajustes", passes:"passers-by", now:"na cena agora",
        active:"em movimento", rate:"ritmo", perMinute:"passagens por minuto",
        last:"últimas passagens", idle:"sem contagem ativa", frames:"frames",
        interval:"intervalo", engine:"motor", of:"de", keep:"guardar frames", record:"gravar mp4",
        target:"alvo", people:"pessoas", cars:"carros", vlmCheck:"vlm a cada (s)",
        jev:"roteador (Jev)" },
  en: { stopped:"idle", live:"counting", hint:"Paste a live stream or recording link",
        go:"count →", stop:"stop", adv:"options", passes:"passers-by", now:"in scene now",
        active:"moving", rate:"rate", perMinute:"passes per minute",
        last:"latest passes", idle:"no active run", frames:"frames",
        interval:"interval", engine:"engine", of:"of", keep:"keep frames", record:"record mp4",
        target:"target", people:"people", cars:"cars", vlmCheck:"vlm every (s)",
        jev:"Jev router" }
};
let lang = localStorage.getItem("sc-lang") || "en";
function applyLang() {
  const t = T[lang];
  document.getElementById("lang").textContent = lang === "pt" ? "EN" : "PT";
  document.getElementById("statusText").textContent = t.stopped;
  document.getElementById("hint").textContent = t.hint;
  document.getElementById("go").textContent = t.go;
  document.getElementById("stop").textContent = t.stop;
  document.getElementById("advSummary").textContent = t.adv;
  document.getElementById("lPasses").textContent = t.passes;
  document.getElementById("lNow").textContent = t.now;
  document.getElementById("lActive").textContent = t.active;
  document.getElementById("lRate").textContent = t.rate;
  document.getElementById("lPerMinute").textContent = t.perMinute;
  document.getElementById("lLast").textContent = t.last;
  document.getElementById("lInterval").firstChild.textContent = t.interval + " (s) ";
  document.getElementById("lKeep").firstChild.textContent = t.keep + " ";
  document.getElementById("lTarget").firstChild.textContent = t.target + " ";
  document.getElementById("lMp4").firstChild.textContent = t.record + " ";
  document.getElementById("lVlmCheck").firstChild.textContent = t.vlmCheck + " ";
  document.getElementById("lJev").firstChild.textContent = t.jev + " ";
  const tsel = document.getElementById("target");
  tsel.options[0].textContent = t.people;
  tsel.options[1].textContent = t.cars;
}
document.getElementById("lang").onclick = () => {
  lang = lang === "pt" ? "en" : "pt"; localStorage.setItem("sc-lang", lang); applyLang();
};

const $ = (id) => document.getElementById(id);
function showError(msg) {
  const box = $("error");
  if (!msg) { box.style.display = "none"; return; }
  box.textContent = msg; box.style.display = "block";
}

$("form").onsubmit = async (ev) => {
  ev.preventDefault(); showError("");
  const keepVal = parseInt($("keep").value, 10);
  const options = {
    interval: parseFloat($("interval").value || "2"),
    tiles: parseInt($("tiles").value || "2", 10),
    engine: $("engine").value, conf: parseFloat($("conf").value || "0.25"),
    target: $("target").value, headers: $("headers").value.trim() || null,
    keep_frames: Number.isFinite(keepVal) ? keepVal : 10, flow: true,
    timelapse: $("mp4").checked, timelapse_fps: parseInt($("mp4fps").value, 10) || 12,
    vlm_check: parseFloat($("vlmcheck").value) || 0, jev_router: $("jev").checked
  };
  const res = await fetch("/api/start", { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ source: $("source").value, options }) });
  const data = await res.json();
  if (!data.ok) showError(data.error || "falha ao iniciar");
};
$("stop").onclick = async () => {
  const data = await (await fetch("/api/stop", { method:"POST", body:"{}" })).json();
  if (!data.ok) showError(data.error || "");
};

function drawChart(series) {
  const c = $("chart"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (!series.length) return;
  const n = series.length, pad = 4;
  const w = (c.width - pad * 2) / Math.max(n, 1);
  const max = Math.max(...series, 1);
  for (let i = 0; i < n; i++) {
    const h = (series[i] / max) * (c.height - 22);
    ctx.fillStyle = "#e8a33d";
    ctx.globalAlpha = 0.85;
    ctx.fillRect(pad + i * w + 1, c.height - 18 - h, Math.max(w - 3, 2), h);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = "#2b241c"; ctx.beginPath();
  ctx.moveTo(0, c.height - 17.5); ctx.lineTo(c.width, c.height - 17.5); ctx.stroke();
}

async function tick() {
  try {
    const s = await (await fetch("/api/status")).json();
    const t = T[lang];
    const st = $("status");
    st.className = "status" + (s.running ? " live" : "");
    $("statusText").textContent = s.running ? t.live : (s.exit_code === 3 ? "erro" : t.stopped);
    $("passes").textContent = s.passes ?? 0;
    $("count").textContent = s.count ?? 0;
    $("active").textContent = s.active ?? 0;
    const mins = Math.max((s.elapsed_s || 0) / 60, 1e-9);
    $("rate").textContent = s.passes ? (s.passes / mins).toFixed(1) + "/min" : "—";
    $("stop").style.display = s.running ? "inline" : "none";
    const bits = [];
    if (s.run_dir) bits.push((s.frames || 0) + " " + t.frames);
    if (s.options && s.options.interval) bits.push(t.interval + " " + s.options.interval + "s");
    if (s.options && s.options.engine) bits.push(t.engine + " " + s.options.engine);
    if (s.source) bits.push(s.source.length > 64 ? s.source.slice(0, 64) + "…" : s.source);
    $("submeta").textContent = bits.length ? bits.join(" · ") : t.idle;
    drawChart(s.per_minute || []);
    const rows = (s.last_events || []).map(e =>
      `<tr><td class="dim">${(e.ts || "").slice(11)}</td><td>id${e.id}</td>` +
      `<td class="dim">${e.disp_px}px</td></tr>`).join("");
    $("events").innerHTML = rows || `<tr><td class="dim">—</td></tr>`;
    const img = $("frame");
    if (s.frames > 0) {
      img.style.display = "block";
      img.src = "/latest.jpg?t=" + Date.now();
    }
    // a still-growing MP4 has no moov atom yet — only offer it once the run is over
    $("mp4box").style.display = (s.timelapse && !s.running) ? "block" : "none";
    $("log").textContent = (s.log_tail || []).slice(-6).join("\\n");
  } catch (e) { /* servidor fora: tenta de novo no próximo tick */ }
}
applyLang(); tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""
