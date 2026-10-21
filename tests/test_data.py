from PIL import Image

from scripts.audit_data import audit


def dataset(tmp_path):
    for i, split in enumerate(("train", "val", "test")):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (10, 10), (i * 50, 0, 0)).save(tmp_path / "images" / split / "a.png")
        (tmp_path / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return tmp_path


def test_clean_dataset(tmp_path):
    assert audit(dataset(tmp_path))["passed"]


def test_pixel_duplicate_cross_split_is_failure(tmp_path):
    root = dataset(tmp_path)
    Image.open(root / "images/train/a.png").save(root / "images/val/a.png")
    report = audit(root)
    assert not report["passed"]
    assert len(report["cross_split_pixel_duplicates"]) == 1


def test_invalid_label_and_missing_label(tmp_path):
    root = dataset(tmp_path)
    (root / "labels/train/a.txt").write_text("0 nan 0.5 0.2 0.2\n11 0.5 0.5 0.2 0.2")
    (root / "images/test/b.png").write_bytes((root / "images/test/a.png").read_bytes())
    report = audit(root)
    assert not report["passed"]
    assert len(report["errors"]) == 3
