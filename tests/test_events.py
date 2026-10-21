from ppe.events import EventStore
from ppe.schemas import ViolationEvent


def event(key, source):
    return ViolationEvent(event_id=key, source_id=source, track_id=1,
                          observed_at_utc="2026-10-20T10:00:00+00:00", first_evidence_ms=0,
                          confirmed_at_ms=500, consecutive_observations=5,
                          min_evidence_confidence=.3, person_box=(0, 0, 100, 200), model="test-only")


def test_persistence_deduplication_and_source_filter(tmp_path):
    path = tmp_path / "events.sqlite3"
    store = EventStore(path)
    store.add(event("a", "source-a"))
    store.add(event("a", "source-a"))
    store.add(event("b", "source-b"))
    reopened = EventStore(path)
    assert len(reopened.list()) == 2
    assert len(reopened.list(limit=1)) == 1
    assert reopened.list(source_id="source-a")[0]["event_id"] == "a"
    assert reopened.list(source_id="' OR 1=1 --") == []
