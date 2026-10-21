# PPE Detection Service

A computer vision service for personal protective equipment analysis, built with YOLO, FastAPI and ONNX Runtime. It combines image and video inference, WebSocket frame processing, conservative equipment assessment and persistent event records with reproducible model and data evaluation tools.

**Status:** Active development · Portfolio prototype · Not validated for operational safety decisions

## What it does

- **Image and video inference:** upload images or video through an API or the included browser interface; export annotated video and frame records.
- **Frame streaming:** process binary JPEG/PNG frames over WebSocket, retaining the latest pending frame when inference falls behind.
- **Equipment assessment:** associate helmet, bare-head and vest detections with people; return `unknown` when evidence is absent or conflicting.
- **Temporal events:** require repeated observations over time before recording a missing-helmet event. Store source, model hash and policy settings in SQLite.
- **Runtime evaluation:** export ONNX models, check detection parity and compare PyTorch/ONNX latency under recorded conditions.
- **Data quality controls:** review annotations, group related scenes, version dataset releases and verify hashes, crops, splits and background controls before training.
- **Optional VLM review:** run a local SmolVLM on a detected person crop as an offline advisory experiment, outside the streaming path.

## Architecture

```text
Image / Video / WebSocket
           |
   Decode and validation
           |
 YOLO detector (.pt / .onnx)
           |
 Person–equipment association
           |
   present / missing / unknown
           |
 Temporal confirmation (video / stream)
           |
 SQLite events + API / browser interface

Separate tools: dataset review → release verification → training → evaluation
                person crop → optional local VLM review
```

The service loads one local model per process. A model must contain `person`, `helmet`, `vest` and `no_helmet` labels; generic COCO weights and the current three-class data pilot are rejected. Concurrent inference requests are bounded; a busy service returns HTTP 429.

## Measured results

A YOLO11n baseline was trained for 15 epochs at 640 px on an Apple M4 Pro using MPS. The following runtime comparison used **CPU FP32, batch size 1, 640 px, 20 identical images × 5 rounds**, after 10 warmup predictions per runtime:

- **PyTorch:** mean 22.49 ms; p50 22.41 ms; p95 23.12 ms.
- **ONNX Runtime:** mean 12.77 ms; p50 12.52 ms; p95 14.12 ms.
- **Mean latency ratio:** 1.76× in this local measurement. Final-detection parity passed on the comparison sample.

Timings include preprocessing, inference, postprocessing and CPU result transfer. They exclude image decoding, HTTP, model loading and transport; they are **not live camera FPS**. Runtime thread defaults differed. See the [benchmark and measurement scope](reports/benchmark.md).

Historical validation mAP50 was approximately 0.565 across 11 classes; `no_helmet` AP50 was 0.441. Subsequent annotation review found incorrect boxes, domain mismatch and related train/validation scenes. These scores are retained as historical baseline evidence, **not independent field accuracy**. A higher classification-loss-weight candidate was rejected; the active model was preserved. See [data diagnosis](reports/data-diagnosis.md) and [model selection](reports/model-selection.json).

## Quick start

Use **Python 3.11–3.13**; the local development environment was tested with Python 3.12.

```bash
git clone https://github.com/mirackeser/ppe-detection-service.git
cd ppe-detection-service
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,vision]'
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

### Provide a compatible model

**Datasets and trained weights are not included in this repository.** Place a locally obtained compatible PPE model at `models/best.onnx` or `models/best.pt`. The application validates the class names before serving inference. It does not download a replacement or generate placeholder detections.

```bash
PPE_MODEL_PATH=models/best.onnx PPE_DEVICE=cpu \
  uvicorn ppe.api:app --host 127.0.0.1 --port 8000 \
  --workers 1 --ws-max-size 8388608 --ws-max-queue 1
```

Open the [browser interface](http://127.0.0.1:8000/) or [interactive API documentation](http://127.0.0.1:8000/docs). The camera starts only through an explicit user action. Stop the server with `Ctrl+C`.

```bash
curl -F 'file=@frame.jpg' http://127.0.0.1:8000/detect/image
```

Configuration examples are in [.env.example](.env.example). The server can load a local file with `uvicorn ppe.api:app --env-file .env`; keep that file outside version control. The [macOS package snapshot](requirements-macos-py312.lock.txt) records the development environment and is not a portable CUDA lockfile.

## API

- `GET /` — browser interface for uploads, camera frames and events.
- `GET /healthz` — model readiness; returns 503 if unavailable.
- `POST /detect/image` — JPEG/PNG, up to 8 MiB and 16 megapixels; boxes use original-image coordinates.
- `POST /detect/video` — video up to 64 MiB; returns annotated MP4 and JSONL artifact links. Audio is not retained.
- `WS /stream` — binary image frames; responses include frame identifiers.
- `GET /events` — paginated SQLite events with optional source filter.
- `GET /metrics` — frame counters, rejected requests, dropped frames and latency histograms.
- `GET /artifacts/{id}/{kind}` — generated video or record download.

`pipeline_ms` measures decoding, prediction and policy evaluation after the upload is read; it is not end-to-end HTTP latency. Video job duration is not part of the frame latency histogram.

## Assessment and event policy

The default policy uses confidence thresholds of 0.50 for people, helmets and vests, and 0.25 for explicit bare-head detections. Missing or conflicting evidence produces `unknown`; absence of a detected vest alone does not establish a violation.

For video and streaming, a missing-helmet event requires at least **five consecutive processed observations and 500 ms** of explicit evidence on a matched track. An unknown or missing observation resets confirmation; tracks expire after 1,000 ms. Each episode emits one event and rearms after three helmet-present observations. An optional normalized rectangular work zone limits eligible positions.

Matching uses geometry and IoU, not person re-identification. Occlusion and crossing people can break tracks or change identities. Persistent false detections can still satisfy temporal rules. Work context determines whether PPE is required; the detector does not establish that requirement.

## Data revision and training

The latest reviewed pilot, `helmet-pilot-v3`, contains **43 samples from 32 source photos across 27 scene groups**. Eleven samples are reviewed background crops from the same sources, not independent new scenes. Train and validation both include normal and low-light groups.

Releases record source hashes, crop decisions, class mappings and scene membership. Empty labels require an explicit absence review. Related source variants cannot cross splits, and training uses a working copy so cache creation does not modify an immutable release.

A one-epoch CPU training-path check passed, including batches containing only background samples. **Long training remains blocked:** nine project coverage targets are unmet. The pilot allows at most three epochs for pipeline checks and lacks the `vest` class required by the full service.

A fixed-model background check found a furnace incorrectly detected as a person in **one of two validation controls**. That case remains in the frozen development validation cohort. This small sample does not estimate a general false-positive rate. See the [current stage report](reports/data-context-v3.md), [readiness audit](reports/data-readiness-v3.json), [annotation policy](docs/annotation-policy.md) and [release workflow](docs/data-revision-workflow.md).

Training tools require local reviewed data, which must be prepared separately:

```bash
python -m scripts.verify_release datasets/helmet-pilot-v3 \
  --output reports/helmet-pilot-v3-verification.json
python -m scripts.data_readiness datasets/helmet-pilot-v3 \
  --policy config/data-readiness-v1.json \
  --output reports/data-readiness-v3.json --require-ready
```

The second command intentionally fails until coverage targets are met. The [Colab notebook](notebooks/train_colab.ipynb) accepts separately uploaded source and reviewed-data ZIPs. Its GPU execution has not yet been verified on Colab.

## Runtime experiments

Use the same data, model provenance and preprocessing when comparing runtimes:

```bash
python -m scripts.export --weights models/best.pt --imgsz 640
python -m scripts.compare --pytorch models/best.pt --onnx models/best.onnx \
  --images datasets/benchmark
python -m scripts.benchmark --weights models/best.onnx \
  --images datasets/benchmark --device cpu --output reports/onnx-cpu.json
```

Optional local VLM dependencies and a single-crop review:

```bash
python -m pip install -e '.[vlm]'
python -m scripts.download_vlm
python -m scripts.review_frame --weights models/best.pt --image frame.jpg
```

The 256M and 500M SmolVLM experiments used one person crop. Their free-text output is unvalidated; no dataset-level VLM accuracy or real-time VLM integration is claimed.

## Testing and reproducibility

**92 local tests passed** at the current data revision. Tests cover API contracts, malformed inputs, model requirements, equipment association, temporal state, event persistence, runtime comparison and dataset release integrity. API contract tests use a test detector; separate local integration reports document real-model runs.

GitHub Actions runs the CPU test suite without model weights or private datasets. Training, dataset-dependent evaluation and GPU performance are separate experiments. Reports preserve model/data hashes, configuration and measurement scope where applicable. Some historical reports contain machine-specific paths; those files are evidence records, not portable configuration.

Docker and Compose files are included; container execution has not yet been verified. The default local service has no authentication and no automatic artifact retention policy; deployment hardening remains unfinished.

## Repository layout

```text
ppe/          FastAPI service, detector, policies, temporal tracking and browser UI
scripts/      Data review, training, export, evaluation and integration tools
tests/        Automated API, policy, runtime and data-pipeline tests
config/       Versioned data coverage targets
reports/      Recorded experiments, audits and verification evidence
docs/         Annotation and dataset release contracts
notebooks/    Colab training workflow
```

Datasets, weights, virtual environments, local secrets and generated runtime outputs are excluded from version control.

## Next milestones

1. Expand independent reviewed scenes, bare-head examples and background controls; obtain independent annotation review.
2. Complete coverage targets, then run controlled training with fixed evaluation cohorts.
3. Establish reviewed vest coverage and a compatible full-PPE model.
4. Evaluate real videos and temporal event precision/recall before extending VLM integration.
5. Validate container execution, deployment controls and target-hardware performance.

The held-out Construction-PPE test split has not been used during this data revision.

## Data sources and attribution

- [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) — historical baseline dataset; source records are described in the data audit.
- [SHEL5K v4](https://data.mendeley.com/datasets/9rcv8mm682/4) — Gochoo, 2022, DOI: `10.17632/9rcv8mm682.4`, published under CC BY 4.0; used for the reviewed helmet pilot. [Recorded source metadata](reports/shel5k-source.json).
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO training and inference tooling.
- [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) — optional local vision-language experiments.

Third-party data, models and dependencies retain their respective licenses. No dataset or model redistribution is included here. This repository does not currently include a standalone license grant for its original source code.

Maintained by [Miraç Keser](https://github.com/mirackeser).
