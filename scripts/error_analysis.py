"""Rank validation images by unmatched GT/predictions at fixed conf=.25, IoU=.5."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ppe.detector import YoloDetector
from ppe.schemas import Detection
from scripts.compare import iou
from scripts.prepare_data import NAMES


def match_errors(truth, predictions, threshold=.5):
    candidates = sorted(((iou(a.box, b.box), i, j) for i, a in enumerate(truth)
                         for j, b in enumerate(predictions) if a.label == b.label), reverse=True)
    matched_truth, matched_predictions = set(), set()
    for overlap, i, j in candidates:
        if overlap >= threshold and i not in matched_truth and j not in matched_predictions:
            matched_truth.add(i)
            matched_predictions.add(j)
    return ([i for i in range(len(truth)) if i not in matched_truth],
            [i for i in range(len(predictions)) if i not in matched_predictions])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--root", type=Path, default=Path("datasets/construction-ppe"))
    parser.add_argument("--output", type=Path, default=Path("runs/error-analysis"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    detector = YoloDetector(args.weights, imgsz=args.imgsz)
    records = []
    for path in sorted((args.root / "images/val").glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        truth = []
        label = args.root / "labels/val" / path.with_suffix(".txt").name
        for row in label.read_text().splitlines():
            if not row.strip():
                continue
            cls, x, y, w, h = map(float, row.split())
            truth.append(Detection(label=NAMES[int(cls)].lower(), confidence=1,
                                   box=(max(0,(x-w/2)*width), max(0,(y-h/2)*height),
                                        min(width,(x+w/2)*width), min(height,(y+h/2)*height))))
        predictions = detector.predict(image)
        missing, extra = match_errors(truth, predictions)
        records.append({"image": str(path),
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                        "false_negatives": len(missing), "false_positives": len(extra),
                        "missed_classes": [truth[i].label for i in missing],
                        "extra_classes": [predictions[i].label for i in extra],
                        "truth": [d.model_dump() for d in truth],
                        "predictions": [d.model_dump() for d in predictions]})
    records.sort(key=lambda r: r["false_negatives"] + r["false_positives"], reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for rank, record in enumerate(records[:args.limit], 1):
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for group, color, prefix in ((record["truth"], "lime", "GT"), (record["predictions"], "red", "PRED")):
            for d in group:
                draw.rectangle(d["box"], outline=color, width=2)
                draw.text((d["box"][0], d["box"][1]), f"{prefix} {d['label']}", fill=color, stroke_width=1, stroke_fill="black")
        image.save(args.output / f"{rank:02}-{Path(record['image']).stem}.jpg")
    (args.output / "index.json").write_text(json.dumps({"model_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                                                       "imgsz": args.imgsz, "split": "val", "confidence_floor": .25,
                                                       "scope": "validation; conf=.25; IoU=.5; greedy same-class matching; not mAP",
                                                       "ranked_images": records}, indent=2), encoding="utf-8")
    print(f"Saved validation error analysis to {args.output}")


if __name__ == "__main__":
    main()
