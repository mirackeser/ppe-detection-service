from ppe.schemas import Detection
from scripts.error_analysis import match_errors


def test_matching_is_one_to_one_and_class_aware():
    person = Detection(label="person", confidence=.9, box=(0, 0, 100, 100))
    helmet = Detection(label="helmet", confidence=.9, box=(0, 0, 100, 100))
    missing, extra = match_errors([person], [person, person])
    assert missing == [] and len(extra) == 1
    assert match_errors([person], [helmet]) == ([0], [0])
