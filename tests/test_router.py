"""Jev router — request shape, thresholds, fail-open policy, frame metrics (offline)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from streamcount.router import (
    JevRouter,
    RouterMetrics,
    build_questions,
    build_state,
    describe_frame,
    resolve_jev_key,
    scene_change,
)


def _metrics(**over):
    base = dict(
        target="people",
        interval_s=2.0,
        detections=6,
        tracks_active=4,
        tracks_moving=2,
        low_conf=1,
        brightness=0.18,
        scene_change=0.02,
        passes_total=10,
        seconds_since_last_check=30.0,
        last_vlm_count=5,
        last_vlm_local=6,
    )
    base.update(over)
    return RouterMetrics(**base)


def _reply(payload: dict) -> bytes:
    return json.dumps(payload).encode()


def _router(transport, **over):
    return JevRouter("test-key", transport=transport, **over)


def test_state_carries_the_decision_facts():
    state = build_state(_metrics())
    assert "detections=6" in state
    assert "brightness 0.18" in state and "night" in state
    assert "disagreement" in state and "30s since the last decision" in state


def test_questions_shape():
    questions = build_questions()
    assert questions["uncertainty"]["type"] == "noul"
    assert questions["reason"]["type"] == "choice"
    assert "nothing_unusual" in questions["reason"]["criteria"]


def test_decide_escalates_above_threshold():
    captured = {}

    def transport(url, headers, body, timeout):
        captured["url"] = url
        captured["auth"] = headers["Authorization"]
        captured["body"] = json.loads(body)
        return _reply(
            {
                "answers": {
                    "uncertainty": {"noul": 0.82, "confidence": 0.91},
                    "reason": {"choice": "dense_crowd"},
                }
            }
        )

    router = _router(transport)
    decision = router.decide(_metrics(detections=18, low_conf=5))
    assert decision.escalate is True
    assert decision.uncertainty == 0.82
    assert decision.reason == "dense_crowd"
    assert decision.error is None
    assert captured["url"].endswith("/v1/systemone")
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "jev-latest"
    assert set(captured["body"]["questions"]) == {"uncertainty", "reason"}
    assert router.stats() == {
        "model": "jev-latest",
        "checks": 1,
        "escalations": 1,
        "skipped": 0,
        "errors": 0,
        "threshold": 0.5,
        "mean_latency_ms": router.stats()["mean_latency_ms"],
    }


def test_decide_skips_below_threshold():
    def transport(url, headers, body, timeout):
        return _reply(
            {
                "answers": {
                    "uncertainty": {"noul": 0.12, "confidence": 0.8},
                    "reason": {"choice": "nothing_unusual"},
                }
            }
        )

    router = _router(transport)
    decision = router.decide(_metrics())
    assert decision.escalate is False
    assert decision.reason == "nothing_unusual"
    assert router.stats()["skipped"] == 1


def test_threshold_is_configurable():
    def transport(url, headers, body, timeout):
        return _reply(
            {"answers": {"uncertainty": {"noul": 0.4}, "reason": {"choice": "low_light"}}}
        )

    assert _router(transport).decide(_metrics()).escalate is False
    assert _router(transport, threshold=0.3).decide(_metrics()).escalate is True


def test_fail_open_on_transport_error():
    def transport(url, headers, body, timeout):
        raise OSError("network down")

    router = _router(transport)
    decision = router.decide(_metrics())
    assert decision.escalate is True, "a failing router must keep the anchor"
    assert decision.error and "network down" in decision.error
    assert router.stats()["errors"] == 1


def test_fail_open_on_answerless_response():
    def transport(url, headers, body, timeout):
        return b'{"answers": {}}'

    decision = _router(transport).decide(_metrics())
    assert decision.escalate is True and decision.error


def test_router_needs_a_key():
    with pytest.raises(ValueError):
        JevRouter(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JevRouter("")


def test_frame_metrics():
    black = Image.new("L", (64, 36), 0)
    white = Image.new("L", (64, 36), 255)
    dark_mean, dark_sample = describe_frame(black)
    bright_mean, bright_sample = describe_frame(white)
    assert dark_mean == 0.0 and bright_mean == 1.0
    assert scene_change(dark_sample, bright_sample) == 1.0
    assert scene_change(dark_sample, dark_sample) == 0.0
    assert scene_change(dark_sample, None) == 0.0


def test_resolve_key_order(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STREAMCOUNT_JEV_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("streamcount.router.Path.home", lambda: tmp_path)
    assert resolve_jev_key("direct") == ("direct", "--jev-key")
    assert resolve_jev_key()[0] is None
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    assert resolve_jev_key() == ("from-env", "env TYPESAFE_API_KEY")
    monkeypatch.delenv("TYPESAFE_API_KEY")
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from-dotenv\n", encoding="utf-8")
    assert resolve_jev_key() == ("from-dotenv", ".env file")
