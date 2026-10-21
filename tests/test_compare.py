from ppe.schemas import Detection
from scripts.compare import compare_detections


def box(label="person", confidence=.9, coords=(0, 0, 100, 200)):
    return Detection(label=label, confidence=confidence, box=coords)


def test_order_does_not_matter():
    left = [box(), box("helmet", coords=(0, 0, 20, 30))]
    assert compare_detections(left, left[::-1])["passed"]


def test_missing_box_wrong_label_and_confidence_drift_fail():
    assert not compare_detections([box()], [])["passed"]
    assert not compare_detections([box()], [box("vest")])["passed"]
    assert not compare_detections([box()], [box(confidence=.5)])["passed"]
