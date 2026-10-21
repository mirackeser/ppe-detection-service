"""Build a versioned, reviewed helmet pilot; reject incomplete quality gates.

Only explicitly accepted images enter the release. Source files are never edited.
The pilot has train/val only: it is not a final held-out benchmark or a vest model.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import random
import shutil
import tempfile

import yaml
from PIL import Image

from scripts.prepare_shel5k import file_hash
from scripts.shel5k_inventory import TARGET_NAMES


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes root")
    return path


def corrected_objects(record, item):
    objects = json.loads(json.dumps(record["objects"]))
    by_number = {o["source_object"]: o for o in objects}
    seen = set()
    for patch in item.get("corrections", []):
        number = patch["source_object"]
        if number in seen or number not in by_number or not patch.get("reason", "").strip():
            raise ValueError("Duplicate, unknown or unexplained annotation correction")
        seen.add(number)
        obj = by_number[number]
        if patch["old_box"] != obj["box"]:
            raise ValueError("Correction does not match original annotation")
        box = patch["new_box"]
        width, height = record["size"]
        if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError("Invalid corrected box")
        obj["box"] = box
    return objects


def apply_reviewed_crop(objects, size, item):
    crop = item.get("crop")
    if crop is None:
        return objects, list(size), []
    if (len(crop) != 4 or not all(type(v) is int for v in crop)
            or not 0 <= crop[0] < crop[2] <= size[0] or not 0 <= crop[1] < crop[3] <= size[1]
            or not item.get("crop_reason", "").strip()):
        raise ValueError("Crop requires reviewed integer bounds and a reason")
    converted, log = [], []
    for original in objects:
        obj = dict(original)
        x1, y1, x2, y2 = original["box"]
        clipped = [max(x1, crop[0]), max(y1, crop[1]), min(x2, crop[2]), min(y2, crop[3])]
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            log.append({"source_object": obj["source_object"], "action": "outside_reviewed_crop"})
            continue
        # Partial head evidence cannot silently become a negative target.
        if obj["label"] in {"helmet", "no_helmet"} and clipped != original["box"]:
            raise ValueError("Crop intersects a target head/helmet; review different bounds or exclude image")
        obj["box"] = [clipped[0]-crop[0], clipped[1]-crop[1], clipped[2]-crop[0], clipped[3]-crop[1]]
        if clipped != original["box"]:
            log.append({"source_object": obj["source_object"], "action": "clipped_to_crop", "original_box": original["box"], "box": obj["box"]})
        converted.append(obj)
    return converted, [crop[2]-crop[0], crop[3]-crop[1]], log


def assign_splits(accepted, policy="pilot-group-v2"):
    """Choose groups before training, using coverage only, never model scores.

    Preserve v2 for the frozen initial pilot. V3 uses a seeded, bounded search
    on larger reviewed cohorts and balances class/image proportions. Stable
    scene hashes resolve ties. Neither policy consumes inference results.
    """
    groups = sorted({r["review"]["scene_id"] for r in accepted},
                    key=lambda s: hashlib.sha256(("ppe-pilot-v1:"+s).encode()).hexdigest())
    if policy not in {"pilot-group-v2", "reviewed-group-v3", "reviewed-context-v4"}:
        raise ValueError("Unknown frozen split policy")
    if len(groups) < 4 or (policy == "pilot-group-v2" and len(groups) > 20):
        raise ValueError("Pilot splitter requires 4..20 groups; larger reviewed releases require reviewed-group-v3")
    n_val = max(2, round(len(groups)*.2))
    def covered(rows):
        labels = {o["label"] for r in rows for o in r["objects"]}
        mixed = any({o["label"] for o in r["objects"]} >= {"helmet", "no_helmet"} for r in rows)
        if not (labels >= set(TARGET_NAMES) and mixed):
            return False
        if policy == "reviewed-context-v4":
            for condition in ("normal", "low_light"):
                if len({r["review"]["scene_id"] for r in rows if r["review"].get("lighting") == condition}) < 2:
                    return False
            if len({r["review"]["scene_id"] for r in rows if r["review"].get("sample_role") == "background_negative"}) < 2:
                return False
        return True
    if policy == "pilot-group-v2" or len(groups) <= 20:
        candidates = combinations(groups, n_val)
    else:
        # Bounded search over group coverage/balance only, never model scores.
        rng = random.Random(20261021)
        candidates = {tuple(sorted(groups[:n_val]))}
        for _ in range(4096):
            candidates.add(tuple(sorted(rng.sample(groups, n_val))))
        candidates = sorted(candidates)
    total_counts = Counter(o["label"] for r in accepted for o in r["objects"] if o["label"] in TARGET_NAMES)
    selected, best_score = None, None
    for combination in candidates:
        validation = set(combination)
        val = [r for r in accepted if r["review"]["scene_id"] in validation]
        train = [r for r in accepted if r["review"]["scene_id"] not in validation]
        if covered(val) and covered(train):
            if policy == "pilot-group-v2":
                selected = validation
                break
            val_counts = Counter(o["label"] for r in val for o in r["objects"] if o["label"] in TARGET_NAMES)
            score = (sum(abs(val_counts[name]/total_counts[name]-.2) for name in TARGET_NAMES)
                     + abs(len(val)/len(accepted)-.2), tuple(sorted(validation)))
            if best_score is None or score < best_score:
                selected, best_score = validation, score
    if selected is not None:
        for row in accepted:
            row["split"] = "val" if row["review"]["scene_id"] in selected else "train"
        return
    raise ValueError("Reviewed groups cannot provide required class, mixed-PPE and context coverage in both splits")


def plan_release(inventory, review):
    if not review.get("reviewer", "").strip() or review["reviewer"] == "pending":
        raise ValueError("A recorded reviewer is required")
    records = {r["id"]: r for r in inventory["records"]}
    if len(records) != len(inventory["records"]):
        raise ValueError("Duplicate inventory IDs")
    accepted, seen = [], set()
    for item in review["items"]:
        identity = item["id"]
        source_identity = item.get("source_id", identity)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identity) or identity in seen or source_identity not in records:
            raise ValueError("Duplicate or unknown review ID")
        seen.add(identity)
        if item["decision"] not in {"accept", "exclude", "pending"}:
            raise ValueError("Unknown review decision")
        if item["decision"] != "accept":
            continue
        record = records[source_identity]
        if any(item.get(k) != record[k] for k in ("image_sha256", "annotation_sha256")):
            raise ValueError("Review source hash mismatch")
        if (item.get("labels_reviewed") is not True or item.get("domain") not in {"construction", "industrial"}
                or not item.get("notes", "").strip() or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,100}", item.get("scene_id") or "")):
            raise ValueError("Accepted image lacks annotation/domain/scene review")
        if any(o["difficult"] for o in record["objects"] if o["label"] is not None):
            raise ValueError("Difficult target annotations require separate adjudication; exclude this image")
        objects = corrected_objects(record, item)
        objects, size, crop_log = apply_reviewed_crop(objects, record["size"], item)
        target_objects = [o for o in objects if o["label"] in TARGET_NAMES]
        role = item.get("sample_role", "positive")
        if role not in {"positive", "background_negative"}:
            raise ValueError("Unknown sample role")
        if not target_objects:
            confirmation = item.get("negative_review", {})
            if (role != "background_negative" or objects or not item.get("crop")
                    or confirmation.get("confirmed_absent") is not True
                    or set(confirmation.get("reviewed_classes", [])) != set(TARGET_NAMES)
                    or not confirmation.get("reviewer", "").strip()
                    or not confirmation.get("reason", "").strip()):
                raise ValueError("Empty target labels require explicit, object-free background crop review")
        elif role == "background_negative":
            raise ValueError("Background negative contains annotated target objects")
        if review.get("split_policy") == "reviewed-context-v4" and item.get("lighting") not in {"normal", "low_light"}:
            raise ValueError("Context release requires reviewed lighting condition")
        accepted.append({"record": record, "review": item, "objects": objects,
                         "size": size, "crop_log": crop_log})
    if not accepted:
        raise ValueError("No accepted images")
    exceptions = {}
    for pair in review.get("different_scene_pairs", []):
        key = tuple(sorted(pair["ids"]))
        if len(key) != 2 or key[0] == key[1] or not pair.get("reason", "").strip():
            raise ValueError("Invalid scene-pair adjudication")
        exceptions[key] = pair
    collisions = []
    for i, a in enumerate(accepted):
        for b in accepted[i+1:]:
            left, right = a["record"], b["record"]
            if left["id"] == right["id"]:
                if (a["review"]["scene_id"] != b["review"]["scene_id"]
                        or a["review"].get("crop") == b["review"].get("crop")):
                    raise ValueError("Source variants must share one scene and have distinct reviewed crops")
                continue
            if left["pixel_sha256"] == right["pixel_sha256"]:
                raise ValueError(f"Duplicate pixels must be deduplicated: {left['id']}, {right['id']}")
            distance = (int(left["dhash128"], 16) ^ int(right["dhash128"], 16)).bit_count()
            if distance <= 12 and a["review"]["scene_id"] != b["review"]["scene_id"]:
                key = tuple(sorted((left["id"], right["id"])))
                if key not in exceptions:
                    collisions.append({"ids": key, "distance": distance})
    if collisions:
        raise ValueError(f"Unresolved similar-scene candidates: {collisions}")
    assign_splits(accepted, review.get("split_policy", "pilot-group-v2"))
    counts, groups = {}, {}
    for split in ("train", "val"):
        rows = [r for r in accepted if r["split"] == split]
        counts[split] = dict(Counter(o["label"] for r in rows for o in r["objects"] if o["label"] is not None))
        groups[split] = sorted({r["review"]["scene_id"] for r in rows})
        if len(groups[split]) < 2 or any(counts[split].get(name, 0) == 0 for name in TARGET_NAMES):
            raise ValueError(f"Pilot needs >=2 scene groups and all target classes in {split}")
        if not any({o["label"] for o in r["objects"]} >= {"helmet", "no_helmet"} for r in rows):
            raise ValueError(f"Pilot needs a mixed helmet/no_helmet image in {split}")
    return accepted, {"class_instances": counts, "scene_groups": groups,
                      "images": dict(Counter(r["split"] for r in accepted)),
                      "unique_sources": {s: len({r["record"]["id"] for r in accepted if r["split"] == s}) for s in ("train", "val")},
                      "sample_roles": {s: dict(Counter(r["review"].get("sample_role", "positive") for r in accepted if r["split"] == s)) for s in ("train", "val")},
                      "review_decisions": dict(Counter(i["decision"] for i in review["items"]))}


def build_release(inventory_path, review_path, destination):
    if destination.exists():
        raise FileExistsError("Release exists; choose a new version, never overwrite")
    inventory = json.loads(inventory_path.read_text())
    review = json.loads(review_path.read_text())
    if review.get("inventory_sha256") != file_hash(inventory_path):
        raise ValueError("Review bound to different inventory")
    selected, summary = plan_release(inventory, review)
    root = Path(inventory["source_root"])
    # Fail before creating output on changed/missing source files.
    for row in selected:
        record = row["record"]
        for field, digest in (("image", "image_sha256"), ("annotation", "annotation_sha256")):
            if file_hash(checked_path(root, record[field])) != record[digest]:
                raise ValueError(f"Source changed after review: {record['id']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ppe-release-", dir=destination.parent))
    try:
        manifest = []
        for row in selected:
            record, split = row["record"], row["split"]
            source = checked_path(root, record["image"])
            sample_id = row["review"]["id"]
            image = staging / "images" / split / (sample_id+source.suffix.lower())
            label = staging / "labels" / split / (sample_id+".txt")
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            if row["review"].get("crop") is not None:
                with Image.open(source) as original:
                    original.convert("RGB").crop(row["review"]["crop"]).save(image)
            else:
                shutil.copyfile(source, image)
            width, height = row["size"]
            lines = []
            for obj in row["objects"]:
                if obj["label"] is None:
                    continue
                x1, y1, x2, y2 = obj["box"]
                values = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
                lines.append(str(TARGET_NAMES.index(obj["label"]))+" "+" ".join(f"{v:.9f}" for v in values))
            label.write_text("\n".join(lines)+("\n" if lines else ""))
            manifest.append({"id": sample_id, "source_id": record["id"], "split": split, "scene_id": row["review"]["scene_id"],
                             "sample_role": row["review"].get("sample_role", "positive"),
                             "lighting": row["review"].get("lighting", "unreviewed"),
                             "image": str(image.relative_to(staging)), "label": str(label.relative_to(staging)),
                             "image_sha256": file_hash(image), "label_sha256": file_hash(label),
                             "source_image_sha256": record["image_sha256"], "size": row["size"],
                             "source_annotation_sha256": record["annotation_sha256"],
                             "crop": row["review"].get("crop"), "crop_log": row["crop_log"],
                             "corrections": row["review"].get("corrections", []),
                             "conversion_warnings": record["conversion_warnings"]})
        data = {"path": str(destination.resolve()), "train": "images/train", "val": "images/val",
                "names": dict(enumerate(TARGET_NAMES))}
        (staging / "data.yaml").write_text(yaml.safe_dump(data))
        shutil.copyfile(review_path, staging / "review.json")
        metadata = {"version": destination.name, "scope": "reviewed helmet pilot; not full PPE dataset or final benchmark",
                    "source": inventory["source"], "reviewer": review["reviewer"],
                    "inventory_sha256": file_hash(inventory_path), "review_sha256": file_hash(review_path),
                    "split_policy": review.get("split_policy", "pilot-group-v2"),
                    "split_method": ("reviewed-context-v4: seeded grouped balance search plus >=2 normal-light, >=2 low-light and >=2 reviewed background-negative groups per split; source variants stay in parent scene; no model scores"
                                     if review.get("split_policy") == "reviewed-context-v4" else
                                     "reviewed-group-v3: SHA256 scene order; seed 20261021, up to 4096 sampled group combinations for >20 groups; minimize per-class and image-ratio deviation from 0.2; no model scores"
                                     if review.get("split_policy") == "reviewed-group-v3" else
                                     "pilot-group-v2: stable SHA256 scene order, first coverage-valid group combination; max(2,round(groups*0.2)) val groups; no model scores"),
                    "mapping": inventory["mapping"], "target_names": TARGET_NAMES,
                    "summary": summary, "files": manifest, "ready_for_training": False,
                    "remaining_gate": "Run independent release verification before pilot training; no deployment promotion.",
                    "limitations": ["Assistant visual review is not independent human annotation QA.",
                                    "Perceptual hashing cannot prove scene independence.",
                                    "No vest supervision; never mix into a vest dataset with missing labels.",
                                    "No test split; pilot validation cannot serve as final performance evidence."]}
        (staging / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("runs/shel5k-review/inventory.json"))
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_release(args.inventory, args.review, args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
