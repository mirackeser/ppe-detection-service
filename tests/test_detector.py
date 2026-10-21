import pytest

from ppe.detector import YoloDetector


def test_weights_must_exist_locally(tmp_path):
    with pytest.raises(ValueError, match="local trained"):
        YoloDetector(str(tmp_path / "absent.pt"))


def test_coco_model_rejected_before_warmup(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    weights = tmp_path / "coco.pt"
    weights.write_bytes(b"test stub, not a model")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda *a, **kw: SimpleNamespace(names={0: "person", 1: "car"})))
    with pytest.raises(ValueError, match="COCO weights are insufficient"):
        YoloDetector(str(weights))
