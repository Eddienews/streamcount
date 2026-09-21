"""Flow tracker behaviour (offline, no network)."""

from streamcount.tracking import FlowTracker


def box(x, y, w=120, h=200):
    return (x, y, x + w, y + h)


def test_walker_counts_once():
    tracker = FlowTracker()
    for i in range(6):
        tracker.update(i * 2.0, [box(100 + i * 60, 400)])
    assert len(tracker.passes) == 1
    assert tracker.n_short == 0 and tracker.n_static == 0


def test_gap_ghost_reid():
    tracker = FlowTracker()
    seq = [box(100, 400), box(160, 400), None, None, box(280, 400), box(340, 400)]
    for i, det in enumerate(seq):
        tracker.update(i * 2.0, [det] if det else [])
    assert len(tracker.passes) == 1


def test_two_people_two_passes():
    tracker = FlowTracker()
    for i in range(6):
        tracker.update(i * 2.0, [box(100 + i * 60, 300), box(1400 - i * 70, 600)])
    assert len(tracker.passes) == 2


def test_static_person_not_counted():
    tracker = FlowTracker()
    for i in range(10):
        tracker.update(i * 2.0, [box(800, 500)])
    assert len(tracker.passes) == 0
    tracker.finalize(20.0)
    assert tracker.n_static == 1 and tracker.n_short == 0


def test_single_frame_noise_discarded():
    tracker = FlowTracker()
    tracker.update(0.0, [box(500, 500)])
    tracker.update(2.0, [])
    tracker.update(10.0, [])
    assert len(tracker.passes) == 0
    tracker.finalize(20.0)
    assert tracker.n_short == 1


def test_events_carry_increasing_totals():
    tracker = FlowTracker()
    events = []
    for i in range(10):
        _, new = tracker.update(i * 2.0, [box(100 + i * 60, 300), box(1200 - i * 60, 700)])
        events.extend(new)
    assert [e["n"] for e in events] == [1, 2]
    assert tracker.total == 2
