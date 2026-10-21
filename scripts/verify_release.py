"""Independently verify released images/YOLO labels and their review provenance."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import yaml

from scripts.curate_dataset import assign_splits, checked_path
from scripts.diagnose_data import difference_hash
from scripts.prepare_shel5k import file_hash
from scripts.shel5k_inventory import TARGET_NAMES


def verify_release(root):
    root = root.resolve()
    metadata = json.loads((root / "manifest.json").read_text())
    config = yaml.safe_load((root / "data.yaml").read_text())
    review_path = root / "review.json"
    errors, counts, scenes = [], defaultdict(Counter), defaultdict(set)
    if config != {"path": str(root), "train": "images/train", "val": "images/val", "names": dict(enumerate(TARGET_NAMES))}:
        errors.append("Unexpected YAML paths/classes (pilot has no test split)")
    if file_hash(review_path) != metadata["review_sha256"]:
        errors.append("Review changed after export")
    review = json.loads(review_path.read_text())
    accepted = {i["id"]: i for i in review["items"] if i["decision"] == "accept"}
    if review["inventory_sha256"] != metadata["inventory_sha256"]:
        errors.append("Inventory provenance mismatch")
    if metadata.get("target_names") != TARGET_NAMES:
        errors.append("Unexpected manifest class mapping")
    expected, ids, image_hashes = set(), set(), set()
    decoded_images, split_rows, source_groups = [], [], {}
    for record in metadata["files"]:
        identity, split = record["id"], record["split"]
        if identity in ids or split not in {"train", "val"}:
            errors.append("Duplicate ID or invalid split")
        ids.add(identity)
        item = accepted.get(identity)
        source_id = record.get("source_id", identity)
        if item and item.get("source_id", identity) != source_id:
            errors.append(f"Sample/source ID mismatch: {identity}")
        if source_id in source_groups and source_groups[source_id] != record["scene_id"]:
            errors.append(f"Source variants split into different scenes: {source_id}")
        source_groups[source_id] = record["scene_id"]
        role = record.get("sample_role", "positive")
        if item and (role != item.get("sample_role", "positive") or role not in {"positive", "background_negative"}):
            errors.append(f"Review/sample role mismatch: {identity}")
        if metadata.get("split_policy") == "reviewed-context-v4":
            if item is None or item.get("lighting") not in {"normal", "low_light"} or item.get("lighting") != record.get("lighting"):
                errors.append(f"Missing reviewed lighting condition: {identity}")
        if item is None or item.get("labels_reviewed") is not True or item.get("scene_id") != record["scene_id"]:
            errors.append(f"Missing accepted annotation/scene review: {identity}")
        if item and (item.get("domain") not in {"construction", "industrial"} or not item.get("notes", "").strip()):
            errors.append(f"Missing domain/context review: {identity}")
        if item and (item.get("image_sha256") != record.get("source_image_sha256", record["image_sha256"]) or item.get("annotation_sha256") != record["source_annotation_sha256"]):
            errors.append(f"Review/source provenance mismatch: {identity}")
        if item and item.get("crop") != record.get("crop"):
            errors.append(f"Review/crop mismatch: {identity}")
        scenes[split].add(record["scene_id"])
        if record["image_sha256"] in image_hashes:
            errors.append("Duplicate image bytes")
        image_hashes.add(record["image_sha256"])
        for field in ("image", "label"):
            expected.add(record[field])
            path = checked_path(root, record[field])
            if not path.is_file() or file_hash(path) != record[field+"_sha256"]:
                errors.append(f"Missing or changed release file: {record[field]}")
        image_path, label_path = checked_path(root, record["image"]), checked_path(root, record["label"])
        if image_path.is_file():
            with Image.open(image_path) as image:
                if "size" in record and list(image.size) != record["size"]:
                    errors.append(f"Image dimensions differ from manifest: {identity}")
                rgb = image.convert("RGB")
                decoded_images.append({"id": identity, "split": split, "scene_id": record["scene_id"],
                                       "pixel_hash": hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),
                                       "dhash": difference_hash(rgb)})
        target_objects = []
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    cls, x, y, w, h = map(float, line.split())
                    if (not all(math.isfinite(v) for v in (cls, x, y, w, h)) or cls != int(cls)
                            or not 0 <= cls < len(TARGET_NAMES) or w <= 0 or h <= 0
                            or x-w/2 < -1e-8 or y-h/2 < -1e-8 or x+w/2 > 1+1e-8 or y+h/2 > 1+1e-8):
                        raise ValueError("invalid row")
                    counts[split][TARGET_NAMES[int(cls)]] += 1
                    target_objects.append({"label": TARGET_NAMES[int(cls)]})
                except ValueError:
                    errors.append(f"Invalid YOLO geometry/class: {record['label']}")
        if not target_objects:
            confirmation = (item or {}).get("negative_review", {})
            if (role != "background_negative" or not record.get("crop") or confirmation.get("confirmed_absent") is not True
                    or set(confirmation.get("reviewed_classes", [])) != set(TARGET_NAMES)
                    or not confirmation.get("reviewer", "").strip() or not confirmation.get("reason", "").strip()):
                errors.append(f"Unreviewed empty-label sample: {identity}")
        elif role == "background_negative":
            errors.append(f"Negative sample contains target annotations: {identity}")
        split_rows.append({"record": {"id": identity}, "review": item or {"scene_id": record["scene_id"]},
                           "objects": target_objects, "recorded_split": split})
    if ids != set(accepted):
        errors.append("Export does not contain exactly the accepted review IDs")
    actual = {str(p.relative_to(root)) for folder in ("images", "labels") for p in (root/folder).rglob("*") if p.is_file()}
    if actual != expected:
        errors.append("Unregistered or missing image/label files")
    if scenes["train"] & scenes["val"]:
        errors.append("Scene leakage between train and validation")
    try:
        assign_splits(split_rows, metadata.get("split_policy", "pilot-group-v2"))
        if any(row["split"] != row["recorded_split"] for row in split_rows):
            errors.append("Split assignment differs from the frozen deterministic pilot policy")
    except ValueError as exc:
        errors.append(str(exc))
    adjudicated = {tuple(sorted(p["ids"])) for p in review.get("different_scene_pairs", []) if p.get("reason", "").strip()}
    for i, a in enumerate(decoded_images):
        for b in decoded_images[i+1:]:
            if a["pixel_hash"] == b["pixel_hash"]:
                errors.append(f"Duplicate decoded release images: {a['id']}, {b['id']}")
            if a["split"] != b["split"] and (a["dhash"] ^ b["dhash"]).bit_count() <= 12:
                if tuple(sorted((a["id"], b["id"]))) not in adjudicated:
                    errors.append(f"Unresolved post-crop scene similarity: {a['id']}, {b['id']}")
    for split in ("train", "val"):
        if dict(counts[split]) != metadata["summary"]["class_instances"][split]:
            errors.append(f"Instance counts differ from manifest: {split}")
        if len(scenes[split]) < 2 or any(counts[split][name] == 0 for name in TARGET_NAMES):
            errors.append(f"Pilot coverage insufficient: {split}")
    coverage = {}
    for split in ("train", "val"):
        rows = [r for r in metadata["files"] if r["split"] == split]
        source_count = len({r.get("source_id", r["id"]) for r in rows})
        roles = dict(Counter(r.get("sample_role", "positive") for r in rows))
        coverage[split] = {"samples": len(rows), "unique_sources": source_count, "scene_groups": len(scenes[split]),
                           "sample_roles": roles, "lighting_groups": {condition: len({r["scene_id"] for r in rows if r.get("lighting") == condition}) for condition in ("normal", "low_light")}}
        if "unique_sources" in metadata["summary"] and source_count != metadata["summary"]["unique_sources"].get(split):
            errors.append(f"Source counts differ from manifest: {split}")
        if "sample_roles" in metadata["summary"] and roles != metadata["summary"]["sample_roles"].get(split):
            errors.append(f"Sample role counts differ from manifest: {split}")
    return {"passed": not errors, "pilot_training_ready": not errors, "deployment_ready": False,
            "substantive_training_ready": False, "max_verified_epochs": 3,
            "manifest_sha256": file_hash(root / "manifest.json"), "data_yaml_sha256": file_hash(root / "data.yaml"),
            "errors": errors, "class_instances": {k: dict(v) for k, v in counts.items()}, "coverage": coverage,
            "scope": "Mechanical and review-provenance checks for a small pilot; not independent human label QA or final model accuracy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_release(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
