import pytest

from scripts.select_model import decide


def inputs(candidate_precision=.7):
    baseline = {"model_sha256": "baseline", "split": "val", "imgsz": 640, "mAP50": .56,
                "class_metrics": [{"Class": "no_helmet", "mAP50": .44}]}
    candidate = {"model_sha256": "candidate", "split": "val", "imgsz": 640, "mAP50": .58,
                 "class_metrics": [{"Class": "no_helmet", "mAP50": .50}]}
    def calibration(key, precision, recall):
        return {"model_sha256": key, "split": "val", "imgsz": 640,
                "thresholds": [{"threshold": .25, "precision": precision, "recall": recall}]}
    return baseline, candidate, calibration("baseline", .6, .33), calibration("candidate", candidate_precision, .4)


def test_promote_requires_all_gates():
    assert decide(*inputs())["promote_candidate"]
    assert not decide(*inputs(candidate_precision=.5))["promote_candidate"]


def test_stale_calibration_is_rejected():
    values = inputs()
    values[3]["model_sha256"] = "different-model"
    with pytest.raises(ValueError):
        decide(*values)
