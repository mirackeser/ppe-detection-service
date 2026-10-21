"""Review one YOLO person crop with a local VLM; no automatic safety decision."""
import argparse
import json
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from ppe.detector import YoloDetector
from ppe.policy import assess
from ppe.vlm import SmolVLMReviewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--person-index", type=int, default=0,
                        help="Index within detected people, not all detections")
    parser.add_argument("--model", choices=["HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct"],
                        default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("reports/vlm-review.json"))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.rounds < 1:
        parser.error("warmup must be non-negative and rounds must be positive")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    detector = YoloDetector(args.weights, imgsz=args.imgsz)
    detections = detector.predict(image)
    assessments = assess(detections)
    if not 0 <= args.person_index < len(assessments):
        parser.error(f"Select a person index in [0, {len(assessments) - 1}]")
    assessment = assessments[args.person_index]
    person = detections[assessment.person_index]
    crop = image.crop(tuple(round(value) for value in person.box))
    # Dedicated second stage: a slow reviewer is never placed in the streaming path.
    reviewer = SmolVLMReviewer(args.model, args.device)
    for _ in range(args.warmup):
        reviewer.review(crop)
    reviews = [reviewer.review(crop).to_dict() for _ in range(args.rounds)]
    result = {"image": str(args.image), "person": person.model_dump(),
              "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "crop_pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
              "crop_size": list(crop.size), "warmup": args.warmup,
              "yolo_assessment": assessment.model_dump(), "review": reviews[0], "reviews": reviews,
              "limitations": "Single person crop repeated, not dataset-level accuracy evaluation. VLM output is unvalidated free text. Timing excludes model load and YOLO."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved advisory VLM review to {args.output}")


if __name__ == "__main__":
    main()
