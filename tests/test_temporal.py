import pytest

from ppe.policy import assess
from ppe.schemas import Detection
from ppe.temporal import TemporalConfig, TemporalTracker


def frame(status="missing", x=0):
    detections = [Detection(label="person", confidence=.9, box=(x, 0, x+40, 100))]
    if status != "unknown":
        detections.append(Detection(label="no_helmet" if status == "missing" else "helmet",
                                    confidence=.3 if status == "missing" else .9,
                                    box=(x+10, 2, x+30, 20)))
    return detections


def update(tracker, time, status="missing", x=0):
    detections = frame(status, x)
    return tracker.update(detections, assess(detections), time, (200, 100), "test-only")


def tracker(**kwargs):
    return TemporalTracker("source-1", TemporalConfig(confirm_observations=3, min_duration_ms=200,
                                                      max_gap_ms=500, clear_observations=2, **kwargs))


def test_duration_observations_and_one_event_per_episode():
    t = tracker()
    assert update(t, 0)[1] == []
    assert update(t, 50)[1] == []
    assert update(t, 100)[1] == []  # observation count alone is insufficient
    _, events = update(t, 200)
    assert len(events) == 1
    assert events[0].consecutive_observations == 4
    assert events[0].min_evidence_confidence == .3
    assert update(t, 300)[1] == []
    update(t, 400, "present")
    update(t, 500, "present")
    update(t, 600)
    update(t, 700)
    assert len(update(t, 800)[1]) == 1


def test_unknown_and_absence_interrupt_pending_evidence():
    t = tracker()
    update(t, 0)
    update(t, 100)
    update(t, 200, "unknown")
    assert update(t, 300)[0][0]["missing_streak"] == 1
    t.update([], [], 400, (200, 100), "test-only")
    assert update(t, 500)[0][0]["missing_streak"] == 1


def test_track_expiry_and_different_people_cannot_share_streaks():
    t = tracker()
    first = update(t, 0)[0][0]["track_id"]
    other = update(t, 100, x=100)[0][0]
    assert other["track_id"] != first
    assert other["missing_streak"] == 1
    observations, events = update(t, 700)
    assert observations[0]["track_id"] != first
    assert events == []


def test_ambiguous_association_does_not_transfer_evidence():
    t = tracker()
    detections = frame(x=0) + frame(x=30)
    t.update(detections, assess(detections), 0, (200, 100), "test-only")
    shifted = frame(x=15)
    observations, events = t.update(shifted, assess(shifted), 100, (200, 100), "test-only")
    assert observations == [] and events == []
    assert all(track.missing_streak == 0 for track in t.tracks.values())


def test_work_zone_excludes_person_feet_outside_area():
    t = tracker(work_zone=(.5, 0, 1, 1))
    for time in (0, 100, 200):
        observations, events = update(t, time)
        assert not observations[0]["in_work_zone"]
        assert events == []
    for time in (300, 400):
        assert update(t, time, x=120)[1] == []
    assert len(update(t, 500, x=120)[1]) == 1


def test_invalid_configuration_and_timestamp_rejected():
    with pytest.raises(ValueError):
        TemporalConfig(confirm_observations=1)
    with pytest.raises(ValueError):
        TemporalConfig(work_zone=(1, 0, 0, 1))
    t = tracker()
    update(t, 100)
    with pytest.raises(ValueError):
        update(t, 99)
    with pytest.raises(ValueError):
        update(t, float("nan"))
