from ppe.policy import assess
from ppe.schemas import Detection


def d(label, box, confidence=0.9):
    return Detection(label=label, box=box, confidence=confidence)


def test_absence_is_unknown():
    result = assess([d("person", (0, 0, 100, 200))])[0]
    assert result.helmet == result.vest == "unknown"


def test_explicit_missing_helmet_and_present_vest():
    result = assess([d("person", (0, 0, 100, 200)),
                     d("no_helmet", (30, 5, 70, 40)),
                     d("vest", (20, 60, 80, 130))])[0]
    assert result.helmet == "missing"
    assert result.vest == "present"


def test_conflicting_evidence_is_unknown():
    result = assess([d("person", (0, 0, 100, 200)),
                     d("helmet", (30, 5, 70, 40)),
                     d("no_helmet", (30, 5, 70, 40))])[0]
    assert result.helmet == "unknown"


def test_overlapping_people_do_not_share_helmet():
    results = assess([d("person", (0, 0, 100, 200)),
                      d("person", (10, 0, 110, 200)),
                      d("helmet", (30, 5, 70, 40))])
    assert all(p.helmet == "unknown" for p in results)


def test_low_confidence_and_other_person_gear_are_ignored():
    results = assess([d("person", (0, 0, 100, 200)),
                      d("person", (200, 0, 300, 200)),
                      d("helmet", (230, 5, 270, 40)),
                      d("no_helmet", (30, 5, 70, 40), .2)])
    assert results[0].helmet == "unknown"
    assert results[1].helmet == "present"


def test_missing_helmet_uses_separate_candidate_threshold():
    result = assess([d("person", (0, 0, 100, 200)), d("no_helmet", (30, 5, 70, 40), .3)])[0]
    assert result.helmet == "missing"
    assert result.helmet_confidence == .3
