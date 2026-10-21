"""Inventory SHEL5K source without approving labels or choosing test examples."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageOps

from scripts.diagnose_data import difference_hash
from scripts.prepare_shel5k import file_hash

MAPPING = {"helmet": "helmet", "head": "no_helmet",
           "person_with_helmet": "person", "person_no_helmet": "person",
           "head_with_helmet": None, "face": None}
TARGET_NAMES = ["helmet", "no_helmet", "person"]


def parse_voc(path, actual_size):
    raw = path.read_bytes()
    if len(raw) > 2*1024**2 or b"<!doctype" in raw.lower() or b"<!entity" in raw.lower():
        raise ValueError("XML declarations or excessive document size")
    xml = ET.fromstring(raw)
    size = tuple(int(xml.findtext(f"size/{field}", "0")) for field in ("width", "height"))
    if size != actual_size or min(size) <= 0:
        raise ValueError(f"XML/image size mismatch: {size} versus {actual_size}")
    filename = xml.findtext("filename", "")
    if Path(filename).name != filename or "\\" in filename or Path(filename).stem != path.stem:
        raise ValueError("Unexpected XML image filename")
    objects, warnings = [], []
    for number, obj in enumerate(xml.findall("object"), 1):
        name = obj.findtext("name", "").strip()
        if name not in MAPPING:
            raise ValueError(f"Unmapped source class: {name}")
        original = [float(obj.findtext(f"bndbox/{key}", "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        if (not all(math.isfinite(v) and v == int(v) for v in original)
                or not (0 <= original[0] <= original[2] <= size[0] and 0 <= original[1] <= original[3] <= size[1])):
            raise ValueError(f"Invalid VOC box at object {number}: {original}")
        # VOC inclusive 1-based coordinates -> 0-based half-open xyxy.
        # LabelImg sometimes writes zero minima; retain and report that boundary repair.
        box = [max(0, original[0]-1), max(0, original[1]-1), original[2], original[3]]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Empty VOC box at object {number}")
        if min(original[:2]) == 0:
            warnings.append({"object": number, "kind": "zero_minimum_clamped", "original_box": original})
        difficult = obj.findtext("difficult", "0")
        truncated = obj.findtext("truncated", "0")
        if difficult not in {"0", "1"} or truncated not in {"0", "1"}:
            raise ValueError("Invalid difficult/truncated flag")
        objects.append({"source_object": number, "source_label": name, "label": MAPPING[name],
                        "box": box, "source_box": original,
                        "difficult": difficult == "1", "truncated": truncated == "1"})
    return objects, warnings


def inventory(root):
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    annotations = {p.stem: p for p in root.rglob("*.xml")}
    if len(annotations) != len(list(root.rglob("*.xml"))) or len({p.stem for p in images}) != len(images):
        raise ValueError("Duplicate source basenames")
    records, errors = [], []
    counts = Counter()
    for path in images:
        annotation = annotations.get(path.stem)
        if annotation is None:
            errors.append({"image": str(path.relative_to(root)), "error": "missing annotation"})
            continue
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                size = image.size
                pixels = hashlib.sha256(str(size).encode()+image.tobytes()).hexdigest()
                dhash = f"{difference_hash(image):032x}"
            objects, warnings = parse_voc(annotation, size)
        except (ValueError, OSError, ET.ParseError) as exc:
            errors.append({"image": str(path.relative_to(root)), "error": str(exc)})
            continue
        counts.update(obj["source_label"] for obj in objects)
        records.append({"id": path.stem, "image": str(path.relative_to(root)), "annotation": str(annotation.relative_to(root)),
                        "image_sha256": file_hash(path), "annotation_sha256": file_hash(annotation),
                        "pixel_sha256": pixels, "dhash128": dhash, "size": list(size),
                        "objects": objects, "conversion_warnings": warnings})
    image_ids = {p.stem for p in images}
    orphaned = sorted(set(annotations)-image_ids)
    return {"source_root": str(root.resolve()), "source": json.loads((root / "provenance.json").read_text()),
            "target_names": TARGET_NAMES, "mapping": MAPPING, "records": records,
            "errors": errors, "orphaned_annotations": orphaned,
            "summary": {"source_images": len(images), "source_annotations": len(annotations), "valid_images": len(records),
                        "quarantined_images": len(errors), "class_counts_valid_only": dict(counts)},
            "ready_for_training": False, "status": "unreviewed_source_inventory"}


def review_sample(records, count=48, seed=42):
    """Fixed exploratory sample, NOT evaluation/test split selection."""
    rng = random.Random(seed)
    positives = [r for r in records if any(o["label"] == "no_helmet" for o in r["objects"])]
    others = [r for r in records if not any(o["label"] == "no_helmet" for o in r["objects"])]
    return sorted(rng.sample(positives, min(len(positives), count*3//4)) +
                  rng.sample(others, min(len(others), count-count*3//4)), key=lambda r: r["id"])


def review_sheets(records, root, output):
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), 12):
        sheet = Image.new("RGB", (1280, 1080), "#202020")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(records[start:start+12]):
            with Image.open(root / row["image"]) as source:
                image = source.convert("RGB")
            boxes = ImageDraw.Draw(image)
            for obj in row["objects"]:
                if obj["label"] in {"helmet", "no_helmet"}:
                    color = "cyan" if obj["label"] == "helmet" else "lime"
                    boxes.rectangle(obj["box"], outline=color, width=2)
                    boxes.text((obj["box"][0], obj["box"][1]), str(obj["source_object"]), fill=color, stroke_fill="black", stroke_width=1)
            image = ImageOps.contain(image, (314, 326))
            x, y = (i % 4)*320, (i // 4)*360
            sheet.paste(image, (x, y+30))
            draw.text((x+3, y+4), row["id"], fill="white")
        sheet.save(output / f"sample-{start//12+1:02}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/shel5k-v4-source"))
    parser.add_argument("--output", type=Path, default=Path("runs/shel5k-review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = inventory(args.root)
    path = args.output / "inventory.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    sample = review_sample(result["records"])
    review_sheets(sample, args.root, args.output)
    review = {"inventory_sha256": file_hash(path), "purpose": "exploratory pilot review; no held-out evaluation claims",
              "reviewer": "pending", "items": [{"id": r["id"], "image_sha256": r["image_sha256"],
              "annotation_sha256": r["annotation_sha256"], "decision": "pending", "scene_id": None,
              "domain": "unreviewed", "labels_reviewed": False, "notes": "", "corrections": []} for r in sample]}
    template = args.output / "review-template.json"
    if not template.exists():
        template.write_text(json.dumps(review, indent=2), encoding="utf-8")
    summary = {**result["summary"], "inventory_sha256": file_hash(path), "errors": result["errors"],
               "mapping": MAPPING, "orphaned_annotations": result["orphaned_annotations"],
               "sample_ids": [r["id"] for r in sample], "ready_for_training": False}
    Path("reports/shel5k-inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
