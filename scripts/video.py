import argparse
import json
from pathlib import Path

from ppe.detector import YoloDetector
from ppe.video import process_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/video/annotated.mp4"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    engine = YoloDetector(args.weights, args.device, args.imgsz)
    print(json.dumps(process_video(args.source, args.output, engine, args.stride, args.max_frames), indent=2))


if __name__ == "__main__":
    main()
