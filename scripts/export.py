"""Export a local PPE checkpoint to fixed-shape FP32 ONNX without auto-install."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.weights.is_file() or args.weights.suffix != ".pt":
        parser.error("Supply a local .pt checkpoint")
    from ultralytics import YOLO
    model = YOLO(str(args.weights), task="detect")
    output = model.export(format="onnx", imgsz=args.imgsz, dynamic=False, batch=1,
                          quantize=32, simplify=False, opset=17, device="cpu", nms=False)
    manifest = {"source_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
                "imgsz": args.imgsz, "batch": 1, "precision": "FP32", "opset": 17,
                "dynamic": False, "embedded_nms": False}
    Path(output).with_suffix(".export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {output}; run scripts.compare and validation before deployment")


if __name__ == "__main__":
    main()
