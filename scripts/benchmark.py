"""Serial, decoded-image-to-detections benchmark; not HTTP latency or stream FPS."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from statistics import mean
from time import perf_counter

from PIL import Image, ImageOps

from ppe.detector import YoloDetector


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1:
        parser.error("rounds and warmup must be positive")
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error("No images found")
    # Bound memory; decode outside timed inference and load one image at a time.
    engine = YoloDetector(args.weights, args.device, args.imgsz)
    import torch
    cuda = args.device.isdigit() or args.device.startswith("cuda")
    def load(path):
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    def synchronize():
        import torch
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device.isdigit() or args.device.startswith("cuda"):
            torch.cuda.synchronize()
    warmup_image = load(paths[0])
    for _ in range(args.warmup):
        engine.predict(warmup_image)
    synchronize()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.rounds):
        for path in paths:
            image = load(path)
            synchronize()
            start = perf_counter()
            engine.predict(image)
            synchronize()
            times.append((perf_counter() - start) * 1000)
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "serial predict: preprocess + inference + postprocess + CPU result transfer; excludes decode, HTTP, model load",
        "runtime": engine.runtime, "device": args.device, "imgsz": args.imgsz,
        "batch": 1, "warmup": args.warmup, "samples": len(times),
        "model_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "images": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "p50_ms": percentile(times, .5), "p95_ms": percentile(times, .95),
        "mean_ms": mean(times), "serial_fps": 1000 / mean(times), "raw_ms": times,
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "torch_num_threads": torch.get_num_threads(),
        "versions": {name: importlib.metadata.version(name) for name in ["ultralytics", "torch", "Pillow"]},
    }
    report["precision"] = "FP32 default; verify exported model precision before comparison"
    if engine.runtime == "onnxruntime":
        report["versions"]["onnxruntime"] = importlib.metadata.version("onnxruntime")
        report["execution_providers"] = engine._yolo.predictor.model.session.get_providers()
        report["ort_intra_op_num_threads"] = engine._yolo.predictor.model.session.get_session_options().intra_op_num_threads
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = peak if platform.system() == "Darwin" else peak * 1024
    except ImportError:
        report["process_peak_rss_bytes"] = None
    report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if cuda else None
    report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory() if args.device == "mps" else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(times)} measurements to {args.output}")


if __name__ == "__main__":
    main()
