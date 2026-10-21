"""Compare final detections on real images; empty outputs cannot establish parity."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from ppe.detector import YoloDetector


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0


def compare_detections(left, right, min_iou=.95, max_conf_delta=.01):
    candidates = sorted(((iou(a.box, b.box), i, j) for i, a in enumerate(left)
                         for j, b in enumerate(right) if a.label == b.label), reverse=True)
    used_left, used_right, matches = set(), set(), []
    for overlap, i, j in candidates:
        if overlap < min_iou or i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append({"iou": overlap, "confidence_delta": abs(left[i].confidence - right[j].confidence)})
    unmatched = len(left) + len(right) - 2 * len(matches)
    return {"matches": matches, "unmatched": unmatched,
            "passed": unmatched == 0 and all(m["confidence_delta"] <= max_conf_delta for m in matches)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--output", default=Path("reports/parity.json"), type=Path)
    args = parser.parse_args()
    if Path(args.pytorch).suffix != ".pt" or Path(args.onnx).suffix != ".onnx":
        parser.error("Expected .pt and .onnx models")
    pt, onnx = YoloDetector(args.pytorch, imgsz=args.imgsz), YoloDetector(args.onnx, imgsz=args.imgsz)
    results = []
    image_hashes = []
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image_hashes.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        results.append({"image": path.name, **compare_detections(pt.predict(image), onnx.predict(image))})
    has_matches = any(row["matches"] for row in results)
    passed = bool(results) and has_matches and all(row["passed"] for row in results)
    report = {"passed": passed, "status": "passed" if passed else "failed_or_inconclusive",
              "pytorch_sha256": hashlib.sha256(Path(args.pytorch).read_bytes()).hexdigest(),
              "onnx_sha256": hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest(),
              "images": image_hashes, "imgsz": args.imgsz,
              "scope": "Post-NMS detection parity only; not accuracy evaluation or raw tensor parity",
              "min_iou": .95, "max_confidence_delta": .01, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Parity: {report['status']}; report={args.output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
