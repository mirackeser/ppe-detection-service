"""Read-only train/val diagnosis. Never opens the held-out test split.

Reports label geometry, domain-review material, perceptual duplicate candidates,
and fixed-threshold errors. Similarity and geometric warnings are review hints,
not automatic reasons to delete labels or images.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from PIL import Image, ImageDraw, ImageOps

from ppe.schemas import Detection
from scripts.compare import iou
from scripts.error_analysis import match_errors
from scripts.prepare_data import NAMES


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projected_size(box, image_size, imgsz):
    """Box dimensions after square letterboxing (padding does not change size)."""
    scale = imgsz / max(image_size)
    return ((box[2] - box[0]) * scale, (box[3] - box[1]) * scale)


def size_bucket(box, image_size, imgsz):
    width, height = projected_size(box, image_size, imgsz)
    # Custom diagnostic bins, NOT COCO's area-based small/medium/large metric.
    return "min_side_lt16" if min(width, height) < 16 else "min_side_16_32" if min(width, height) < 32 else "min_side_ge32"


def read_labels(path, size):
    detections, outside = [], []
    width, height = size
    for line, row in enumerate(path.read_text().splitlines(), 1):
        if not row.strip():
            continue
        values = list(map(float, row.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid label: {path}:{line}")
        cls, x, y, w, h = values
        if cls != int(cls) or not 0 <= cls < len(NAMES) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid label: {path}:{line}")
        raw = ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
        if raw[0] < -1 or raw[1] < -1 or raw[2] > width+1 or raw[3] > height+1:
            outside.append({"line": line, "label": NAMES[int(cls)].lower(), "raw_box": raw})
        box = (max(0, raw[0]), max(0, raw[1]), min(width, raw[2]), min(height, raw[3]))
        detections.append(Detection(label=NAMES[int(cls)].lower(), confidence=1, box=box))
    return detections, outside


def difference_hash(image):
    """128-bit horizontal + vertical gradient fingerprint; not identity proof."""
    gray = image.convert("L")
    horizontal = gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    vertical = gray.resize((8, 9), Image.Resampling.LANCZOS).tobytes()
    bits = [horizontal[y*9+x] > horizontal[y*9+x+1] for y in range(8) for x in range(8)]
    bits += [vertical[y*8+x] > vertical[(y+1)*8+x] for y in range(8) for x in range(8)]
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    return result


def near_pairs(records, max_distance=8):
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    pairs = []
    for a in train:
        for b in val:
            distance = (int(a["dhash128"], 16) ^ int(b["dhash128"], 16)).bit_count()
            if distance <= max_distance:
                pairs.append({"train": a["image"], "val": b["image"], "distance": distance})
    return sorted(pairs, key=lambda p: (p["distance"], p["train"], p["val"]))


def verify_cache(index, records, model_path, imgsz):
    if index.get("model_sha256") != sha256(model_path) or index.get("imgsz") != imgsz:
        raise ValueError("Prediction cache model or resolution mismatch")
    if index.get("split") != "val" or index.get("confidence_floor") != .25:
        raise ValueError("Regenerate error-analysis cache with explicit val/confidence provenance")
    expected = {Path(r["image"]).name: r for r in records if r["split"] == "val"}
    seen = set()
    for row in index["ranked_images"]:
        name = Path(row["image"]).name
        if name in seen or name not in expected:
            raise ValueError("Prediction cache has duplicate or unexpected images")
        seen.add(name)
        current = expected[name]
        if any(row.get(key) != current[key] for key in ("image_sha256", "label_sha256", "truth")):
            # JSON uses lists; internal model_dump is normalized by the caller.
            raise ValueError(f"Stale or modified prediction-cache inputs: {name}")
    if seen != set(expected):
        raise ValueError("Prediction cache does not cover the full validation split")


def error_breakdown(index, sizes, imgsz):
    buckets = {key: Counter(gt=0, tp=0, fn=0) for key in ("min_side_lt16", "min_side_16_32", "min_side_ge32")}
    reasons, fp_reasons = Counter(), Counter()
    totals = Counter(tp=0, fp=0, fn=0)
    review = []
    for row in index["ranked_images"]:
        truth = [Detection(**d) for d in row["truth"] if d["label"] == "no_helmet"]
        all_predictions = [Detection(**d) for d in row["predictions"]]
        predictions = [d for d in all_predictions if d.label == "no_helmet"]
        missing, extra = match_errors(truth, predictions)
        totals.update(tp=len(truth)-len(missing), fp=len(extra), fn=len(missing))
        for i, target in enumerate(truth):
            bucket = buckets[size_bucket(target.box, sizes[Path(row["image"]).name], imgsz)]
            bucket["gt"] += 1
            bucket["fn" if i in missing else "tp"] += 1
            if i in missing:
                same = max((iou(target.box, d.box) for d in predictions), default=0)
                wrong_helmet = max((iou(target.box, d.box) for d in all_predictions if d.label == "helmet"), default=0)
                reason = "same_class_overlap_0.1_to_0.5" if .1 <= same < .5 else "helmet_confusion_iou_ge0.5" if wrong_helmet >= .5 else "unmatched_at_conf0.25"
                reasons[reason] += 1
        for i in extra:
            overlap = max((iou(predictions[i].box, t.box) for t in truth), default=0)
            fp_reasons["duplicate_iou_ge0.5" if overlap >= .5 else "localization_overlap_0.1_to_0.5" if overlap >= .1 else "no_matching_gt_overlap"] += 1
        if truth or predictions:
            review.append({"image": Path(row["image"]).name, "gt": len(truth), "tp": len(truth)-len(missing), "fn": len(missing), "fp": len(extra)})
    return {"totals": dict(totals), "by_projected_min_side": {k: dict(v) for k, v in buckets.items()},
            "fn_hints": dict(reasons), "fp_hints": dict(fp_reasons), "images": sorted(review, key=lambda r: r["image"])}


def contact_sheets(records, root, output, prefix):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(records), 12):
        canvas = Image.new("RGB", (1200, 1040), "#202020")
        draw = ImageDraw.Draw(canvas)
        for i, record in enumerate(records[start:start+12]):
            with Image.open(root / record["image"]) as source:
                image = source.convert("RGB")
            boxes = ImageDraw.Draw(image)
            for d in record["truth"]:
                if d["label"] in {"no_helmet", "helmet"}:
                    boxes.rectangle(d["box"], outline="lime" if d["label"] == "no_helmet" else "cyan", width=3)
            image = ImageOps.contain(image, (394, 230))
            x, y = (i % 3)*400, (i // 3)*260
            canvas.paste(image, (x, y+26))
            draw.text((x+3, y+4), record["image"], fill="white")
        path = output / f"{prefix}-{start//12+1:02}.jpg"
        canvas.save(path)
        paths.append(str(path))
    return paths


def similarity_sheets(pairs, root, output):
    paths = []
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), 6):
        canvas = Image.new("RGB", (1000, 1200), "#202020")
        draw = ImageDraw.Draw(canvas)
        for i, pair in enumerate(pairs[start:start+6]):
            for j, split in enumerate(("train", "val")):
                with Image.open(root / pair[split]) as source:
                    image = ImageOps.contain(source.convert("RGB"), (494, 174))
                canvas.paste(image, (j*500, i*200+24))
                draw.text((j*500+3, i*200+3), f"{pair[split]} distance={pair['distance']}", fill="white")
        path = output / f"similarity-{start//6+1:02}.jpg"
        canvas.save(path)
        paths.append(str(path))
    return paths


def diagnose(root, index, model_path, imgsz=640):
    import yaml
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if names != dict(enumerate(NAMES)):
        raise ValueError("Unexpected class mapping; inspect before analysis")
    records, split_reports = [], {}
    for split in ("train", "val"):
        counts, buckets = Counter(), Counter()
        min_sides, positive_images, cooccurrence = [], 0, 0
        for path in sorted((root / "images" / split).iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            label_path = root / "labels" / split / path.with_suffix(".txt").name
            with Image.open(path) as image:
                size, fingerprint = image.size, difference_hash(image)
            truth, outside = read_labels(label_path, size)
            classes = Counter(d.label for d in truth)
            counts.update(classes)
            positive_images += bool(classes["no_helmet"])
            cooccurrence += bool(classes["no_helmet"] and classes["helmet"])
            for d in truth:
                if d.label == "no_helmet":
                    buckets[size_bucket(d.box, size, imgsz)] += 1
                    min_sides.append(min(projected_size(d.box, size, imgsz)))
            records.append({"image": str(path.relative_to(root)), "split": split, "size": list(size),
                            "image_sha256": sha256(path), "label_sha256": sha256(label_path),
                            "dhash128": f"{fingerprint:032x}", "boxes_outside_image_gt1px": outside,
                            "truth": [json.loads(d.model_dump_json()) for d in truth]})
        split_reports[split] = {"images": sum(r["split"] == split for r in records), "class_instances": dict(counts),
                                "no_helmet_images": positive_images, "helmet_and_no_helmet_same_image": cooccurrence,
                                "no_helmet_min_side_at_imgsz": {"median": statistics.median(min_sides) if min_sides else None,
                                                                "bins": dict(buckets)}}
    verify_cache(index, records, model_path, imgsz)
    manifest = [{k: r[k] for k in ("image", "image_sha256", "label_sha256")} for r in records]
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    sizes = {Path(r["image"]).name: r["size"] for r in records if r["split"] == "val"}
    report = {"scope": "train/val diagnosis only; held-out test not opened", "imgsz": imgsz,
              "model_sha256": index["model_sha256"], "data_manifest_sha256": manifest_sha,
              "splits": split_reports, "validation_errors": error_breakdown(index, sizes, imgsz),
              "cross_split_similarity_candidates": near_pairs(records),
              "similarity_method": "128-bit horizontal+vertical dHash, Hamming <=8; manual review required",
              "boxes_outside_image_gt1px": [{"image": r["image"], "boxes": r["boxes_outside_image_gt1px"]} for r in records if r["boxes_outside_image_gt1px"]],
              "limitations": ["Conf >=0.25 cached detections cannot distinguish low-confidence suppression from no detection.",
                              "Box size and overlap hints do not establish occlusion or annotation errors.",
                              "Perceptual hashes can miss scene leakage and can match unrelated simple images.",
                              "Existing benchmark labels remain unchanged; domain review is separate from detection accuracy."],
              "data_manifest": manifest}
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/construction-ppe"))
    parser.add_argument("--index", type=Path, default=Path("runs/error-analysis/index.json"))
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/no-helmet-diagnosis.json"))
    parser.add_argument("--sheets", type=Path, default=Path("runs/data-review"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.imgsz < 32:
        parser.error("imgsz must be >=32")
    report, records = diagnose(args.root, json.loads(args.index.read_text()), args.weights, args.imgsz)
    positives = [r for r in records if any(d["label"] == "no_helmet" for d in r["truth"])]
    train = [r for r in positives if r["split"] == "train"]
    selected = random.Random(42).sample(train, min(24, len(train)))
    validation = [r for r in positives if r["split"] == "val"]
    report["contact_sheets"] = contact_sheets(validation, args.root, args.sheets, "val-labels") + contact_sheets(selected, args.root, args.sheets, "train-seed42")
    report["similarity_sheets"] = similarity_sheets(report["cross_split_similarity_candidates"], args.root, args.sheets)
    report["train_visual_sample"] = [r["image"] for r in selected]
    report["val_visual_review_images"] = [r["image"] for r in validation]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"splits": report["splits"], "errors": report["validation_errors"],
                      "similarity_candidates": len(report["cross_split_similarity_candidates"]),
                      "outside_box_images": len(report["boxes_outside_image_gt1px"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
