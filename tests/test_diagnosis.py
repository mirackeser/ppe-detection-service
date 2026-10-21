import copy
import json

from PIL import Image
import pytest
import yaml

from scripts.diagnose_data import (diagnose, error_breakdown, near_pairs,
                                   projected_size, read_labels, sha256,
                                   size_bucket, verify_cache)
from scripts.prepare_data import NAMES


def fixture_data(tmp_path):
    """No test directory: diagnostic must not require or inspect held-out data."""
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(NAMES))}))
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (100, 200), "red").save(tmp_path / "images" / split / "a.png")
        (tmp_path / "labels" / split / "a.txt").write_text("7 .5 .25 .2 .1\n")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fixture only; model is never loaded")
    label = tmp_path / "labels/val/a.txt"
    truth, _ = read_labels(label, (100, 200))
    row = {"image": "images/val/a.png", "image_sha256": sha256(tmp_path / "images/val/a.png"),
           "label_sha256": sha256(label), "truth": [json.loads(d.model_dump_json()) for d in truth], "predictions": []}
    index = {"model_sha256": sha256(weights), "imgsz": 640, "split": "val", "confidence_floor": .25,
             "ranked_images": [row]}
    return index, weights


def test_letterbox_size_uses_long_edge_and_exact_boundaries():
    assert projected_size((0, 0, 32, 100), (1280, 720), 640) == (16, 50)
    assert size_bucket((0, 0, 32, 100), (1280, 720), 640) == "min_side_16_32"
    assert size_bucket((0, 0, 31, 100), (1280, 720), 640) == "min_side_lt16"
    assert size_bucket((0, 0, 64, 100), (1280, 720), 640) == "min_side_ge32"


def test_diagnosis_covers_train_val_and_does_not_touch_test(tmp_path):
    index, weights = fixture_data(tmp_path)
    report, records = diagnose(tmp_path, index, weights)
    assert set(report["splits"]) == {"train", "val"}
    assert report["validation_errors"]["totals"] == {"tp": 0, "fp": 0, "fn": 1}
    assert len(report["data_manifest"]) == 2
    assert report["cross_split_similarity_candidates"]  # candidate, not automatic exclusion
    assert len(records) == 2


@pytest.mark.parametrize("mutation", ["labels", "image", "weights", "missing", "duplicate", "truth", "split", "floor"])
def test_cache_fails_closed_on_stale_or_partial_inputs(tmp_path, mutation):
    index, weights = fixture_data(tmp_path)
    _, records = diagnose(tmp_path, index, weights)
    changed = copy.deepcopy(index)
    if mutation in {"labels", "image"}:
        changed["ranked_images"][0]["label_sha256" if mutation == "labels" else "image_sha256"] = "stale"
    elif mutation == "weights":
        weights.write_bytes(b"new weights")
    elif mutation == "missing":
        changed["ranked_images"] = []
    elif mutation == "duplicate":
        changed["ranked_images"] *= 2
    elif mutation == "truth":
        changed["ranked_images"][0]["truth"] = []
    else:
        changed["split" if mutation == "split" else "confidence_floor"] = "test" if mutation == "split" else .5
    with pytest.raises(ValueError):
        verify_cache(changed, records, weights, 640)


def test_localization_error_is_not_reported_as_no_output():
    def box(x, label="no_helmet"):
        return {"box": [x, 0, x+20, 20], "label": label, "confidence": 1}
    index = {"ranked_images": [{"image": "a.jpg", "truth": [box(0), box(100)],
                                "predictions": [box(10), box(100, "helmet")]}]}
    result = error_breakdown(index, {"a.jpg": (640, 640)}, 640)
    assert result["totals"] == {"tp": 0, "fp": 1, "fn": 2}
    assert result["fn_hints"] == {"same_class_overlap_0.1_to_0.5": 1, "helmet_confusion_iou_ge0.5": 1}
    assert result["fp_hints"] == {"localization_overlap_0.1_to_0.5": 1}


def test_outside_box_is_recorded_without_mutating_source(tmp_path):
    path = tmp_path / "label.txt"
    original = "7 0.05 0.5 0.2 0.2\n"
    path.write_text(original)
    truth, outside = read_labels(path, (100, 100))
    assert outside[0]["raw_box"][0] == -5
    assert truth[0].box[0] == 0
    assert path.read_text() == original


def test_similarity_compares_cross_split_only_and_preserves_threshold():
    records = [{"image": name, "split": split, "dhash128": fingerprint}
               for name, split, fingerprint in (("a", "train", "0"), ("b", "train", "0"),
                                                ("c", "val", "ff"), ("d", "val", "1ff"))]
    pairs = near_pairs(records)
    assert [(p["train"], p["val"]) for p in pairs] == [("a", "c"), ("b", "c")]
