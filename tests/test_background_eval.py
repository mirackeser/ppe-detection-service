import pytest

from scripts.evaluate_background import summarize
from scripts.cohort import check_validation_lock, validation_fingerprint


def row(identity, group, split="val", labels=(), role="background_negative"):
    return {"id":identity, "scene_id":group, "split":split, "sample_role":role,
            "false_detection":role == "background_negative" and bool(labels),
            "target_detections":[{"label":label} for label in labels], "single_frame_missing_candidates":0}


def test_control_denominators_keep_groups_and_positive_samples_separate():
    rows = [row("a","group-a",labels=("helmet",)), row("b","group-a"),
            row("c","group-b"), row("d","group-c",labels=("person",),role="positive")]
    result = summarize(rows)["val"]
    assert result["negative_images"] == 3 and result["source_groups"] == 2
    assert result["image_fraction_with_false_detection"] == pytest.approx(1/3)
    assert result["group_fraction_with_false_detection"] == .5
    assert result["predicted_classes"] == {"helmet":1}


def test_absent_controls_are_unavailable_not_zero_error():
    result = summarize([row("positive","a",role="positive")])["val"]
    assert result["negative_images"] == 0
    assert result["image_fraction_with_false_detection"] is None
    assert result["group_fraction_with_false_detection"] is None


def test_validation_lock_survives_training_expansion_but_rejects_changed_ground_truth():
    records = [{"id":"a","split":"val","scene_id":"s","image_sha256":"image","label_sha256":"labels"},
               {"id":"b","split":"train"}]
    lock = {"samples":[records[0].copy()],"validation_fingerprint":validation_fingerprint(records)}
    expanded = records + [{"id":"new-training-sample","split":"train"}]
    assert check_validation_lock(expanded,lock) == lock["validation_fingerprint"]
    changed = [dict(r) for r in records]
    changed[0]["label_sha256"] = "new-labels"
    with pytest.raises(ValueError,match="frozen lock"):
        check_validation_lock(changed,lock)
