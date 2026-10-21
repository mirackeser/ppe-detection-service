"""Audit YOLO labels and exact cross-split duplicate images, without changing data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def audit(root: Path, class_count: int = 11):
    errors, splits = [], {}
    hashes = defaultdict(list)
    for split in ("train", "val", "test"):
        images_root, labels_root = root / "images" / split, root / "labels" / split
        paths = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            errors.append(f"{split}: no images")
        counts = Counter()
        empty = 0
        for path in paths:
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    # Pixel digest also detects differently encoded copies.
                    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    hashes[digest].append({"split": split, "image": relative})
            except OSError:
                errors.append(f"{relative}: invalid image")
            label = labels_root / path.relative_to(images_root).with_suffix(".txt")
            if not label.exists():
                errors.append(f"{relative}: missing annotation")
                continue
            rows = label.read_text().splitlines()
            if not any(row.strip() for row in rows):
                empty += 1
            for line_number, row in enumerate(rows, 1):
                if not row.strip():
                    continue
                try:
                    values = [float(value) for value in row.split()]
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError()
                    cls, x, y, w, h = values
                    if cls != int(cls) or not 0 <= cls < class_count or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError()
                    counts[int(cls)] += 1
                except ValueError:
                    errors.append(f"{label.relative_to(root)}:{line_number}: invalid YOLO row")
        orphaned = [str(p.relative_to(root)) for p in labels_root.rglob("*.txt")
                    if not any((images_root / p.relative_to(labels_root).with_suffix(ext)).exists()
                               for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"))]
        splits[split] = {"images": len(paths), "empty_labels": empty,
                         "class_instances": dict(sorted(counts.items())), "orphaned_labels": orphaned}
    duplicates = [group for group in hashes.values() if len({entry["split"] for entry in group}) > 1]
    return {"splits": splits, "errors": errors, "cross_split_pixel_duplicates": duplicates,
            "passed": not errors and not duplicates,
            "limitations": "Exact decoded-pixel duplicates only. Near-duplicate frames and scene identity require separate review."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/construction-ppe"))
    parser.add_argument("--output", type=Path, default=Path("reports/data-audit.json"))
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit saved to {args.output}: passed={report['passed']}, errors={len(report['errors'])}, cross-split duplicate groups={len(report['cross_split_pixel_duplicates'])}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
