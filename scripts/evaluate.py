"""Evaluate selected weights; validation by default, held-out test is explicit."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/construction-ppe/data.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final-model", action="store_true",
                        help="Required for held-out test evaluation, after model selection")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--output", default=Path("reports/evaluation.json"), type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.final_model:
        parser.error("Held-out test requires --final-model; tune on validation only")
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("Weights and dataset configuration must exist locally")
    from ultralytics import YOLO
    metrics = YOLO(str(args.weights), task="detect").val(
        data=str(args.data), split=args.split, device=args.device, imgsz=args.imgsz,
        batch=1, workers=0, rect=False, project=str(Path("runs/evaluate").resolve()), plots=True, verbose=False)
    report = {"split": args.split, "weights": str(args.weights), "imgsz": args.imgsz,
              "model_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "rect": False,
              "device": args.device, "mAP50": float(metrics.box.map50),
              "mAP50_95": float(metrics.box.map),
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "class_metrics": metrics.summary()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda value: value.item()), encoding="utf-8")
    print(f"Evaluation saved to {args.output}")


if __name__ == "__main__":
    main()
