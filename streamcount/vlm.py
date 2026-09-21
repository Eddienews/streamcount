"""Vision-LLM counter (bring your own key, OpenAI-compatible endpoints).

The VLM reads a single frame and returns a strict JSON count. It is the most
accurate engine for sparse scenes (distant/occluded people) but it cannot assign
stable identities across frames -- so it does not replace the tracker for
"how many passed"; use it per-frame, or as a periodic recall anchor
(``--vlm-check``) for the local detector.

Measured on OpenRouter / google-gemini-2.5-flash-lite (1080p night scene frame,
1280 px wide): 1.887 input tokens + ~33 output tokens per call, ~1.8-2.4 s/frame.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

import requests
from PIL import Image

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

TARGET_PROMPTS = {
    "people": (
        "Count every PERSON visible in this image — include people who are small, distant, "
        "seated, partially occluded or cut off at the frame edge. Do not count mannequins, "
        "statues, posters, reflections or the camera operator."
    ),
    "cars": (
        "Count every VEHICLE visible in this image — cars, vans, SUVs, trucks, buses and "
        "motorcycles (moving or parked). Do not count bicycles, reflections or toy vehicles."
    ),
}


def parse_count_response(text: str) -> dict:
    """Extract the strict-JSON counting answer from a model reply (tolerant parsing)."""
    match = re.search(r"\{.*?\}", text or "", re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "count": int(data.get("count", -1)),
                "uncertain": bool(data.get("uncertain", False)),
                "notes": str(data.get("notes", ""))[:200],
            }
        except (ValueError, TypeError):
            pass
    return {"count": -1, "uncertain": True, "notes": f"non-JSON reply: {str(text)[:140]}"}


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_api_key(explicit: str | None = None) -> tuple[str | None, str]:
    """Key resolution order: --vlm-key > env > .env file. Returns (key, source)."""
    if explicit:
        return explicit, "--vlm-key"
    for env_name in ("STREAMCOUNT_VLM_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value.strip(), f"env {env_name}"
    dotenv = _read_dotenv(Path.cwd() / ".env")
    dotenv.update({k: v for k, v in _read_dotenv(Path.home() / ".streamcount.env").items()})
    for env_name in ("STREAMCOUNT_VLM_KEY", "OPENROUTER_API_KEY"):
        value = dotenv.get(env_name)
        if value:
            return value, ".env file"
    return None, "not found"


class VlmCounter:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        target: str = "people",
        timeout: int = 120,
        max_tokens: int = 300,
        max_width: int = 1280,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.target = target
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_width = max_width
        self.last_usage: dict = {}

    def _prompt(self) -> str:
        base = TARGET_PROMPTS.get(self.target, TARGET_PROMPTS["people"])
        return (
            f"{base}\n"
            "Reply with ONLY a JSON object, no markdown: "
            '{"count": <int>, "uncertain": <true|false>, "notes": "<one short sentence>"}'
        )

    def _encode(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        if img.width > self.max_width:
            img = img.resize(
                (self.max_width, int(img.height * self.max_width / img.width)), Image.LANCZOS
            )
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def count(self, image: Image.Image) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt()},
                        {"type": "image_url", "image_url": {"url": self._encode(image)}},
                    ],
                }
            ],
        }
        response = requests.post(
            self.url, headers=self.headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        self.last_usage = data.get("usage", {}) or {}
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return parse_count_response(text or "")
