"""Frame sources: HLS/RTSP/HTTP streams and local files via ffmpeg, YouTube via
yt-dlp, image folders, and an m3u8 finder for webcam pages.

Nothing here needs a cloud service: ffmpeg and yt-dlp are plain local binaries.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import requests
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# URL helpers (pure, unit-testable)
# --------------------------------------------------------------------------- #
def parse_headers(spec: str) -> dict[str, str]:
    """Parse 'Referer=https://x/;User-Agent=Mozilla' into a dict."""
    headers: dict[str, str] = {}
    for part in filter(None, (p.strip() for p in (spec or "").split(";"))):
        key, _, value = part.partition("=")
        if key.strip() and value.strip():
            headers[key.strip()] = value.strip()
    return headers


def default_headers(source: str, headers: dict[str, str] | None = None) -> dict[str, str]:
    """Headers for a source, adding defaults the site requires when the user gave none.

    EarthCam's HLS CDN answers 403 without a Referer from the site (token or not), so a
    pasted EarthCam link works out of the box. An explicit Referer always wins.
    """
    merged = dict(headers or {})
    if "earthcam.com" in (source or "").lower() and \
            not any(key.lower() == "referer" for key in merged):
        merged["Referer"] = "https://www.earthcam.com/"
    return merged


def is_youtube(url: str) -> bool:
    return bool(re.match(r"^https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", url or ""))


def extract_playlist_urls(html: str, cam_id: str | None = None) -> list[str]:
    """Find HLS playlist URLs inside a webcam page (HTML/JS), optionally for one cam id."""
    text = html.replace("\\u0026", "&").replace("\\/", "/")
    pattern = r"https://[\w.-]+/[^\s\"'<>\\]*playlist\.m3u8\?[^\s\"'<>\\]+"
    urls = re.findall(pattern, text)
    if cam_id:
        urls = [u for u in urls if f"/{cam_id}." in u or f"/{cam_id}/" in u]
    # de-duplicate, keep order
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


# --------------------------------------------------------------------------- #
# YouTube / page resolution
# --------------------------------------------------------------------------- #
def resolve_stream_url(url: str, js_runtime: str | None = None) -> str:
    """Return a directly-playable media URL.

    For YouTube links this shells out to yt-dlp (live streams yield an HLS
    manifest URL; recordings yield a video URL). Any other URL is returned as is.
    """
    if not is_youtube(url):
        return url
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        raise SystemExit(
            "YouTube links need yt-dlp on PATH (pip install yt-dlp / brew install yt-dlp)."
        )
    cmd = [yt_dlp, "-f", "bv*[height<=720]+ba/b[height<=720]/b", "-g", url]
    if js_runtime:
        cmd += ["--js-runtimes", js_runtime]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise SystemExit(f"yt-dlp failed:\n{result.stderr.strip()[:600]}")
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip().startswith("http")]
    if not lines:
        raise SystemExit("yt-dlp returned no media URL")
    return lines[0]


def find_stream_in_page(page_url: str, cam_id: str | None = None) -> str:
    """Fetch a webcam page and return the first fresh HLS playlist URL found."""
    response = requests.get(page_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    urls = extract_playlist_urls(response.text, cam_id=cam_id)
    if not urls:
        raise SystemExit(f"no playlist.m3u8 URL found in {page_url}"
                         + (f" for cam {cam_id}" if cam_id else ""))
    return urls[0]


# --------------------------------------------------------------------------- #
# Frame iteration
# --------------------------------------------------------------------------- #
def frames_from_ffmpeg(
    source: str,
    interval: float,
    headers: dict[str, str] | None = None,
    max_frames: int = 10,
    seek_s: float = 0.0,
    ffmpeg_bin: str = "ffmpeg",
) -> Iterator[tuple[int, Image.Image]]:
    """Yield (index, PIL.Image) sampled every `interval` seconds from a stream or file.

    Decodes MJPEG frames out of an ffmpeg pipe (no temp files). Kills ffmpeg
    cleanly when `max_frames` is reached.
    """
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    if seek_s:
        cmd += ["-ss", f"{seek_s:g}"]
    if source.lower().startswith("rtsp://"):
        # TCP for live cameras: portable across NAT/firewalls (UDP/RTP needs open ports on
        # the camera side). Validated against a local mediamtx server over both transports.
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [
        "-i", source,
        "-rw_timeout", "15000000",
        "-vf", f"fps=1/{interval:g}",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = bytearray()
    index = 0
    try:
        while index < max_frames:
            chunk = proc.stdout.read(1 << 16)
            if not chunk:
                break
            buffer.extend(chunk)
            while index < max_frames:
                start = buffer.find(b"\xff\xd8\xff")
                if start < 0:
                    if len(buffer) > 4_000_000:
                        del buffer[:-3]
                    break
                end = buffer.find(b"\xff\xd9", start + 3)
                if end < 0:
                    if start > 0:
                        del buffer[:start]
                    break
                jpeg = bytes(buffer[start: end + 2])
                del buffer[: end + 2]
                try:
                    image = Image.open(io.BytesIO(jpeg))
                    image.load()
                except OSError:
                    continue
                index += 1
                yield index, image
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr = proc.stderr.read().decode("utf-8", "replace").strip()
        if stderr and index == 0:
            print(f"[ffmpeg] {stderr[:400]}")


def frames_from_images(path: Path) -> list[tuple[int, Image.Image]]:
    """Load a single image or a folder of images (sorted) as a frame sequence."""
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    )
    frames: list[tuple[int, Image.Image]] = []
    for index, file in enumerate(files, 1):
        image = Image.open(file)
        image.load()
        frames.append((index, image))
    return frames
