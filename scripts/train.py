"""Run from repository root: python -m scripts.train --device 0 (Colab)."""
import argparse
import json
import platform
from pathlib import Path
import shutil
import tempfile


def training_data_gate(data, legacy_baseline=False):
    """Reject unreviewed or changed data before loading/downloading a model."""
    if legacy_baseline:
        return {"scope": "legacy baseline reproduction; not approved revised-data training",
                "pilot_training_ready": False, "explicit_legacy_override": True}
    root = Path(data).resolve().parent
    if not (root / "manifest.json").is_file() or Path(data).name != "data.yaml":
        raise ValueError("Training requires a reviewed release with manifest.json. For historical baseline reproduction only, use --legacy-baseline.")
    from scripts.verify_release import verify_release
    verification = verify_release(root)
    if not verification["passed"]:
        raise ValueError(f"Dataset verification failed: {verification['errors']}")
    return verification


def stage_training_data(data, working_parent=Path("runs/train-inputs")):
    """Keep framework-generated caches outside the immutable reviewed release."""
    from scripts.verify_release import verify_release
    import yaml
    source = Path(data).resolve().parent
    metadata = json.loads((source / "manifest.json").read_text())
    working_parent.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="verified-", dir=working_parent)).resolve()
    try:
        files = {"manifest.json", "review.json", "data.yaml"}
        files.update(row[field] for row in metadata["files"] for field in ("image", "label"))
        from scripts.curate_dataset import checked_path
        for relative in files:
            target = checked_path(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked_path(source, relative), target)
        path = destination / "data.yaml"
        config = yaml.safe_load(path.read_text())
        config["path"] = str(destination)
        path.write_text(yaml.safe_dump(config))
        result = verify_release(destination)
        if not result["passed"]:
            raise ValueError(f"Staged data verification failed: {result['errors']}")
    except Exception:
        shutil.rmtree(destination)
        raise
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="construction-ppe.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="ppe-baseline")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of TRAIN only; use <1 solely for pipeline smoke tests")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cls-weight", type=float, default=0.5,
                        help="Classification loss gain; baseline=0.5")
    parser.add_argument("--legacy-baseline", action="store_true",
                        help="Explicitly reproduce historical unreviewed-data baseline; not a data-quality approval")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or not 0 < args.fraction <= 1 or args.cls_weight <= 0:
        parser.error("epochs/batch must be positive, imgsz >=32 and fraction in (0,1]")
    try:
        data_verification = training_data_gate(args.data, args.legacy_baseline)
        if not args.legacy_baseline and args.epochs > data_verification["max_verified_epochs"]:
            raise ValueError("This small reviewed pilot supports at most 3 smoke-test epochs. Expand and approve the dataset before substantive training.")
        source_data = args.data
        if not args.legacy_baseline:
            args.data = str(stage_training_data(args.data))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.device == "auto":
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Selected local training device: {args.device}", flush=True)
    from ultralytics import YOLO, settings
    # Keep downloaded data and outputs inside this project.
    settings.update({"datasets_dir": str(Path("datasets").resolve())})
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, seed=42,
                deterministic=True, project=str(Path("runs/train").resolve()), name=args.name,
                exist_ok=False, fraction=args.fraction, workers=args.workers, cls=args.cls_weight)
    import torch, ultralytics
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                   "requested_device": args.device, "seed": 42,
                   "data_verification": data_verification,
                   "source_data_yaml": source_data, "training_data_yaml": args.data,
                   "deterministic_requested": True,
                   "reproducibility_limit": "Some MPS operations may remain nondeterministic; preserve warning logs."}
    (Path(model.trainer.save_dir) / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print("Training complete. Preserve args.yaml, results.csv and weights/best.pt.")
    print("Use validation for tuning. Evaluate held-out test only after choosing the final model.")


if __name__ == "__main__":
    main()
