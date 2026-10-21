# PPE inference benchmark

Hardware environment: macOS-26.3.1-arm64-arm-64bit; arm64; logical CPU count 12.

Both runs: CPU FP32, batch 1, 640 px square input, 20 identical image hashes, 100 samples per runtime after 10 warmup predictions.

PyTorch: p50 22.41 ms; p95 23.12 ms; mean 22.49 ms; serial throughput 44.46 FPS.

ONNX Runtime: p50 12.52 ms; p95 14.12 ms; mean 12.77 ms; serial throughput 78.33 FPS.

Measured mean-latency ratio (PyTorch / ONNX): 1.76x. Values below 1 mean ONNX was slower.

Scope: preprocessing + inference + postprocessing + CPU result transfer. Image decode, HTTP, model load and video transport excluded. These are serial prediction timings, not camera FPS.

Thread configuration: torch=8; ONNX intra-op=0 (0 means runtime default). Defaults were used, not an equal-thread microbenchmark.

Final-detection parity passed on this sample. This is separate from detection accuracy. Full raw timings, memory, versions and model/image hashes are preserved in the input JSON files.

Validation mAP50 across all 11 classes: PyTorch 0.565333; ONNX 0.565340. Absolute difference 0.00000662. This comparison used the validation split; held-out test accuracy is not reported.

Key limitation: no_helmet validation recall 0.4065, AP50 0.4414. Missing-helmet detection needs improvement before safety-alert deployment.
