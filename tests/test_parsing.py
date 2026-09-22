"""Pure helpers: VLM reply parsing, header parsing, URL detection, box merging."""

import numpy as np

from streamcount.detector import COCO80, Detector, class_filter, resolve_classes
from streamcount.pipeline import _default_timeline
from streamcount.sources import (
    default_headers,
    extract_playlist_urls,
    is_youtube,
    parse_headers,
)
from streamcount.vlm import parse_count_response


# ------------------------------------------------------------------ VLM parsing
def test_parse_clean_json():
    result = parse_count_response(
        '{"count": 12, "moving": 3, "uncertain": false, "notes": "ok"}')
    assert result == {"count": 12, "moving": 3, "uncertain": False, "notes": "ok"}


def test_parse_moving_is_optional():
    """Replies without the field keep -1 = "no data" (never 0)."""
    result = parse_count_response('{"count": 12, "uncertain": false, "notes": "ok"}')
    assert result["count"] == 12 and result["moving"] == -1


def test_parse_moving_tolerates_sloppy_values():
    assert parse_count_response('{"count": 5, "moving": null}')["moving"] == -1
    assert parse_count_response('{"count": 5, "moving": "about 2"}')["moving"] == -1
    assert parse_count_response('{"count": 5, "moving": 2.0}')["moving"] == 2


def test_parse_json_inside_prose():
    result = parse_count_response('Here you go:\n{"count": 7, "uncertain": true, "notes": "crowd"}\nEnjoy')
    assert result["count"] == 7 and result["uncertain"] is True


def test_parse_markdown_fence():
    result = parse_count_response('```json\n{"count": 3, "uncertain": false, "notes": ""}\n```')
    assert result["count"] == 3


def test_parse_garbage_is_flagged():
    result = parse_count_response("There are about twenty people.")
    assert result["count"] == -1 and result["uncertain"] is True


# ------------------------------------------------------------------- URL helpers
def test_parse_headers():
    headers = parse_headers("Referer=https://example.com/;User-Agent=Mozilla/5.0")
    assert headers == {"Referer": "https://example.com/", "User-Agent": "Mozilla/5.0"}


def test_parse_headers_empty():
    assert parse_headers("") == {}
    assert parse_headers("junk") == {}


def test_default_headers_adds_earthcam_referer():
    url = "https://videos-3.earthcam.com/fecnetwork/32781.flv/playlist.m3u8?t=a&td=1"
    assert default_headers(url, {}) == {"Referer": "https://www.earthcam.com/"}
    assert default_headers(url, {"referer": "https://mine/"}) == {"referer": "https://mine/"}, \
        "an explicit referer always wins"
    assert default_headers("https://other.example/live.m3u8", {}) == {}
    assert default_headers(url, {"User-Agent": "Mozilla/5.0"}) == \
        {"User-Agent": "Mozilla/5.0", "Referer": "https://www.earthcam.com/"}


def test_is_youtube():
    assert is_youtube("https://www.youtube.com/watch?v=abc")
    assert is_youtube("https://youtu.be/abc")
    assert not is_youtube("https://videos-3.earthcam.com/x/playlist.m3u8?t=1")


def test_extract_playlist_urls_filters_cam_id():
    html = (
        'src="https://cdns.example.com/fecnetwork/4280.flv/playlist.m3u8?t=AAA&amp;td=1" '
        'and "https:\\/\\/cdns.example.com\\/fecnetwork\\/9999.flv\\/playlist.m3u8?t=BBB"'
    )
    all_urls = extract_playlist_urls(html)
    assert len(all_urls) == 2
    only_4280 = extract_playlist_urls(html, cam_id="4280")
    assert len(only_4280) == 1 and "4280" in only_4280[0]


def test_default_timeline():
    assert _default_timeline("https://x/y/playlist.m3u8?t=1") == "wall"
    assert _default_timeline("rtsp://cam/stream") == "wall"
    assert _default_timeline("https://rr4---googlevideo.com/videoplayback?id=1") == "video"


# -------------------------------------------------------------------- box merge
def test_merge_stacked_torso_plus_legs():
    torso = ((100, 100, 220, 260), 0.8, "person")
    legs = ((105, 265, 215, 400), 0.7, "person")
    merged = Detector._merge_stacked([torso, legs])
    assert len(merged) == 1


def test_merge_stacked_keeps_side_by_side():
    torso = ((100, 100, 220, 260), 0.8, "person")
    legs = ((105, 265, 215, 400), 0.7, "person")
    other = ((400, 300, 520, 500), 0.9, "person")
    assert len(Detector._merge_stacked([torso, legs, other])) == 2


def test_merge_stacked_respects_labels():
    person = ((100, 100, 220, 260), 0.8, "person")
    car = ((105, 265, 215, 400), 0.7, "car")
    assert len(Detector._merge_stacked([person, car])) == 2


def test_nms_removes_overlap():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = Detector._nms(boxes, scores, 0.5)
    assert keep == [0, 2]


# ------------------------------------------------------------------- targets
def test_resolve_classes_aliases():
    assert resolve_classes("people") == ("person",)
    assert resolve_classes("cars") == ("car", "motorcycle", "bus", "truck")
    assert resolve_classes("people,cars") == ("person", "car", "motorcycle", "bus", "truck")


def test_resolve_classes_direct_coco_names():
    assert resolve_classes("truck,person") == ("truck", "person")
    assert resolve_classes("") == ("person",)
    assert resolve_classes("garbage") == ("person",)


def test_class_filter_matches_coco_indices():
    # COCO indices: person=0, car=2, motorcycle=3, bus=5, truck=7
    assert class_filter(COCO80, resolve_classes("cars")) == {2, 3, 5, 7}
    assert class_filter(COCO80, resolve_classes("people,cars")) == {0, 2, 3, 5, 7}
    # pose exports are person-only: asking for vehicles yields an empty filter,
    # which the Detector turns into a clear SystemExit instead of silent people counting
    assert class_filter(["person"], resolve_classes("cars")) == set()
