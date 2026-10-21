"""Validate comparable benchmark runs and emit a concise evidence report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", type=Path, default=Path("reports/pytorch-cpu.json"))
    parser.add_argument("--onnx", type=Path, default=Path("reports/onnx-cpu.json"))
    parser.add_argument("--export", type=Path, default=Path("models/best.export.json"))
    parser.add_argument("--parity", type=Path, default=Path("reports/parity.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.md"))
    args = parser.parse_args()
    pt, ort, export, parity = [json.loads(path.read_text()) for path in [args.pytorch, args.onnx, args.export, args.parity]]
    for key in ["images", "imgsz", "batch", "device", "samples", "warmup", "platform"]:
        if pt[key] != ort[key]:
            parser.error(f"Runs are not comparable: {key} differs")
    if pt["model_sha256"] != export["source_sha256"] or ort["model_sha256"] != export["onnx_sha256"]:
        parser.error("Benchmark weights do not match the export manifest")
    if not parity["passed"]:
        parser.error("Detection parity is failed or inconclusive")
    if (parity["pytorch_sha256"] != pt["model_sha256"] or parity["onnx_sha256"] != ort["model_sha256"]
            or parity["images"] != pt["images"] or parity["imgsz"] != pt["imgsz"]):
        parser.error("Parity report does not belong to these weights/images/settings")
    ratio = pt["mean_ms"] / ort["mean_ms"]
    report = ("# PPE inference benchmark\n\n"
              f"Hardware environment: {pt['platform']}; {pt['machine']}; logical CPU count {pt['cpu_count']}.\n\n"
              f"Both runs: CPU FP32, batch 1, {pt['imgsz']} px square input, {len(pt['images'])} identical image hashes, "
              f"{pt['samples']} samples per runtime after {pt['warmup']} warmup predictions.\n\n"
              f"PyTorch: p50 {pt['p50_ms']:.2f} ms; p95 {pt['p95_ms']:.2f} ms; mean {pt['mean_ms']:.2f} ms; "
              f"serial throughput {pt['serial_fps']:.2f} FPS.\n\n"
              f"ONNX Runtime: p50 {ort['p50_ms']:.2f} ms; p95 {ort['p95_ms']:.2f} ms; mean {ort['mean_ms']:.2f} ms; "
              f"serial throughput {ort['serial_fps']:.2f} FPS.\n\n"
              f"Measured mean-latency ratio (PyTorch / ONNX): {ratio:.2f}x. Values below 1 mean ONNX was slower.\n\n"
              "Scope: preprocessing + inference + postprocessing + CPU result transfer. Image decode, HTTP, "
              "model load and video transport excluded. These are serial prediction timings, not camera FPS.\n\n"
              f"Thread configuration: torch={pt['torch_num_threads']}; ONNX intra-op={ort.get('ort_intra_op_num_threads')} "
              "(0 means runtime default). Defaults were used, not an equal-thread microbenchmark.\n\n"
              "Final-detection parity passed on this sample. This is separate from detection accuracy. "
              "Full raw timings, memory, versions and model/image hashes are preserved in the input JSON files.\n")
    pt_validation = Path("reports/validation-pytorch.json")
    ort_validation = Path("reports/validation-onnx.json")
    if pt_validation.exists() and ort_validation.exists():
        pv, ov = [json.loads(path.read_text()) for path in [pt_validation, ort_validation]]
        if pv["model_sha256"] != pt["model_sha256"] or ov["model_sha256"] != ort["model_sha256"]:
            parser.error("Validation reports do not belong to benchmark weights")
        if pv["split"] != "val" or ov["split"] != "val" or pv["imgsz"] != ov["imgsz"] or pv["rect"] != ov["rect"]:
            parser.error("Validation settings differ")
        report += (f"\nValidation mAP50 across all 11 classes: PyTorch {pv['mAP50']:.6f}; ONNX {ov['mAP50']:.6f}. "
                   f"Absolute difference {abs(pv['mAP50']-ov['mAP50']):.8f}. "
                   "This comparison used the validation split; held-out test accuracy is not reported.\n")
        missing_helmet = next((row for row in pv["class_metrics"] if row["Class"] == "no_helmet"), None)
        if missing_helmet:
            report += (f"\nKey limitation: no_helmet validation recall {missing_helmet['Box-R']:.4f}, "
                       f"AP50 {missing_helmet['mAP50']:.4f}. "
                       "Missing-helmet detection needs improvement before safety-alert deployment.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved evidence report to {args.output}")


if __name__ == "__main__":
    main()
