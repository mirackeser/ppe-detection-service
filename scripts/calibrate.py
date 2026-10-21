"""Measure fixed-threshold detection tradeoffs from validation predictions only."""
import argparse
import json
from pathlib import Path

from ppe.schemas import Detection
from scripts.error_analysis import match_errors


def calibration(index, label="no_helmet"):
    results = []
    for threshold in (.25, .35, .5, .65):
        tp = fp = fn = 0
        for row in index["ranked_images"]:
            truth = [Detection(**d) for d in row["truth"] if d["label"] == label]
            predictions = [Detection(**d) for d in row["predictions"]
                           if d["label"] == label and d["confidence"] >= threshold]
            missing, extra = match_errors(truth, predictions)
            tp += len(truth)-len(missing)
            fp += len(extra)
            fn += len(missing)
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        results.append({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
                        "precision": precision, "recall": recall,
                        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
    return {"model_sha256": index["model_sha256"], "imgsz": index["imgsz"],
            "class": label, "split": "val", "iou": .5,
            "scope": "Single-frame object detections, not person-level events or held-out test accuracy",
            "thresholds": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("runs/error-analysis/index.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/calibration-baseline.json"))
    args = parser.parse_args()
    result = calibration(json.loads(args.index.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved validation threshold tradeoffs to {args.output}")


if __name__ == "__main__":
    main()
