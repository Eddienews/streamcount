"""Jev escalation router (TypeSafe, bring your own key).

The hybrid loop (``--flow --vlm-check N``) spends one paid vision-model call every
N seconds as a recall anchor. With ``--jev-router`` the loop first asks Jev
(TypeSafe's System One model) one typed question -- *is the local reading likely
inaccurate right now?* -- and pays for the anchor only when the answer says it
might be. Jev classifies; it never generates text. It also never sees the video:
what leaves the machine is a short paragraph of numbers (detections, brightness,
scene change, how the previous anchor compared), so the run log shows exactly
what the decision was based on. Off by default; without a key it stays off.

Wire format (the same one ForgeOS Browser's decider uses in production):

    POST https://api.typesafe.ai/v1/systemone
    {"state": "...", "model": "jev-latest",
     "questions": {"<name>": {"type": "noul"|"choice", "instructions": "...",
                              "criteria": {"<option>": "what it means"}}}}
    -> {"answers": {"<name>": {"noul": 0.82, "confidence": 0.91,
                               "choice": "...", "probabilities": {...}}}}

A call that fails to answer escalates anyway (fail open): the anchor is the safe
side, the skip is the optimisation.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TIMEOUT = 15.0
FRAME_SAMPLE = (32, 18)  # downscale used for the brightness / scene-change metrics


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


def resolve_jev_key(explicit: str | None = None) -> tuple[str | None, str]:
    """Key resolution order: --jev-key > env > .env file. Returns (key, source)."""
    if explicit:
        return explicit, "--jev-key"
    for env_name in ("STREAMCOUNT_JEV_KEY", "TYPESAFE_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value.strip(), f"env {env_name}"
    dotenv = _read_dotenv(Path.cwd() / ".env")
    dotenv.update(_read_dotenv(Path.home() / ".streamcount.env"))
    for env_name in ("STREAMCOUNT_JEV_KEY", "TYPESAFE_API_KEY"):
        value = dotenv.get(env_name)
        if value:
            return value, ".env file"
    return None, "not found"


@dataclass
class RouterMetrics:
    """The cheap numbers one decision is based on (no pixels leave the machine)."""

    target: str
    interval_s: float
    detections: int
    tracks_active: int
    tracks_moving: int
    low_conf: int
    brightness: float           # 0 = black, 1 = white (downscaled mean)
    scene_change: float         # mean abs difference vs the previous decision frame
    passes_total: int
    seconds_since_last_check: float
    last_vlm_count: int | None  # the previous anchor's count, if any
    last_vlm_local: int | None  # what the detector said at that anchor


def describe_frame(image) -> tuple[float, object]:
    """Brightness plus a small grayscale sample for the router -- roughly 1 ms."""
    import numpy as np

    small = image.convert("L").resize(FRAME_SAMPLE)
    arr = np.asarray(small, dtype="float32") / 255.0
    return float(arr.mean()), arr


def scene_change(small, previous) -> float:
    """Mean absolute difference between two samples (0 = still), as a 0..1 float."""
    import numpy as np

    if previous is None:
        return 0.0
    return float(np.abs(np.asarray(small) - np.asarray(previous)).mean())


def build_state(m: RouterMetrics) -> str:
    lines = [
        "TASK: a video stream is being counted by a local detector (people or vehicles "
        "passing). An independent vision-model recount is the accuracy anchor, allowed at "
        "most once every --vlm-check seconds. Decide whether spending one anchor call on "
        "THIS moment is worth its cost.",
        f"SETTING: target={m.target}, one frame every {m.interval_s:g}s",
        f"LOCAL READING NOW: detections={m.detections}, tracks active={m.tracks_active}, "
        f"tracks moving={m.tracks_moving}, detections near the confidence floor={m.low_conf}",
        f"IMAGE CONDITIONS: brightness {m.brightness:.2f} of 1.0 "
        f"({'night' if m.brightness < 0.35 else 'dim' if m.brightness < 0.6 else 'daylight'}), "
        f"change vs the last decision frame {m.scene_change:.2f} of 1.0",
        f"HISTORY: {m.passes_total} passes so far, "
        f"{m.seconds_since_last_check:.0f}s since the last decision",
    ]
    if m.last_vlm_count is not None and m.last_vlm_local is not None:
        base = max(m.last_vlm_local, 1)
        disagree = abs(m.last_vlm_count - m.last_vlm_local) / base
        lines.append(
            f"LAST ANCHOR: the vision model counted {m.last_vlm_count} while the detector "
            f"said {m.last_vlm_local} (disagreement {disagree:.0%})"
        )
    return "\n".join(lines)


QUESTION_INSTRUCTIONS = (
    "The local detector's reading of this moment may be wrong in a way that matters: "
    "people or vehicles too small, distant or occluded to be found reliably; a dark, "
    "blurred or noisy frame; subjects so close together that detections merge; many "
    "detections sitting right at the confidence floor; a scene that just changed (the "
    "camera moved or the view rotated) so the tracker may carry stale identities; or a "
    "previous anchor that disagreed with the detector. Answer how likely it is that an "
    "independent vision-model recount right now would change the picture."
)

REASONS = {
    "dense_crowd": "many subjects close together, detections merge or overlap",
    "small_or_distant": "subjects tiny, far away or heavily occluded in the frame",
    "low_light": "the frame is dark, blurred or noisy",
    "near_threshold": "several detections sit right at the confidence floor",
    "scene_changed": "the scene changed (camera moved or view rotated), the tracker may be stale",
    "anchor_disagreed": "the previous vision-model recount disagreed with the detector",
    "nothing_unusual": "the reading looks reliable; no recount is needed now",
}


def build_questions() -> dict:
    """The typed questions, both answered in one pass (that is the API's design)."""
    return {
        "uncertainty": {"type": "noul", "instructions": QUESTION_INSTRUCTIONS},
        "reason": {
            "type": "choice",
            "instructions": "Which single factor most drives that judgment for this moment?",
            "criteria": dict(REASONS),
        },
    }


@dataclass
class RouterDecision:
    escalate: bool
    uncertainty: float | None
    confidence: float | None
    reason: str
    latency_ms: int
    error: str | None = None


class JevRouter:
    """One decision per scheduled anchor point: spend the paid call, or skip it."""

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float = DEFAULT_TIMEOUT,
        transport=None,
    ) -> None:
        if not api_key:
            raise ValueError("the Jev router needs a TypeSafe API key")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.threshold = threshold
        self.timeout = timeout
        self._transport = transport or _http_post
        self.calls = 0
        self.escalations = 0
        self.errors = 0
        self._latencies: list[int] = []

    def decide(self, metrics: RouterMetrics) -> RouterDecision:
        body = json.dumps(
            {"state": build_state(metrics), "model": self.model, "questions": build_questions()}
        ).encode("utf-8")
        started = time.monotonic()
        self.calls += 1
        try:
            raw = self._transport(
                self.endpoint,
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                body,
                self.timeout,
            )
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            answers = payload.get("answers") or {}
            uncertainty = (answers.get("uncertainty") or {}).get("noul")
            confidence = (answers.get("uncertainty") or {}).get("confidence")
            reason = (answers.get("reason") or {}).get("choice") or "unstated"
            if not isinstance(uncertainty, (int, float)):
                raise ValueError("the response carried no uncertainty answer")
            escalate = float(uncertainty) >= self.threshold
            error = None
        except Exception as exc:  # fail open: the anchor is the safe side
            self.errors += 1
            escalate, uncertainty, confidence, reason = True, None, None, "error"
            error = f"{type(exc).__name__}: {exc}"[:200]
        latency = int((time.monotonic() - started) * 1000)
        self._latencies.append(latency)
        if escalate:
            self.escalations += 1
        return RouterDecision(escalate, uncertainty, confidence, reason, latency, error)

    def stats(self) -> dict:
        payload: dict = {
            "model": self.model,
            "checks": self.calls,
            "escalations": self.escalations,
            "skipped": self.calls - self.escalations,
            "errors": self.errors,
            "threshold": self.threshold,
        }
        if self._latencies:
            payload["mean_latency_ms"] = round(sum(self._latencies) / len(self._latencies))
        return payload


def _http_post(url: str, headers: dict, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
        return response.read()
