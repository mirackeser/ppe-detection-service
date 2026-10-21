"""Run an actual trained detector through the HTTP contract and save an annotated image."""
import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageOps

from ppe.api import create_app
from ppe.detector import YoloDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    detector = YoloDetector(args.weights, imgsz=args.imgsz)
    with TestClient(create_app(detector)) as client:
        response = client.post("/detect/image", files={"file": (args.image.name, args.image.read_bytes())})
        response.raise_for_status()
        result = response.json()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "response.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    for d in result["detections"]:
        color = "red" if d["label"] == "no_helmet" else "lime"
        draw.rectangle(d["box"], outline=color, width=3)
        draw.text((d["box"][0], max(0,d["box"][1]-12)), f"{d['label']} {d['confidence']:.2f}",
                  fill=color, stroke_width=1, stroke_fill="black")
    image.save(args.output / "detections.jpg")
    print(f"Real model API demo saved to {args.output}")


if __name__ == "__main__":
    main()
