"""Check the real pilot release, local smoke run and Colab package round trip."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import yaml

from scripts.prepare_shel5k import file_hash, safe_extract
from scripts.verify_release import verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("datasets/helmet-pilot-v1"))
    parser.add_argument("--package", type=Path, default=Path("output/helmet-pilot-v1.zip"))
    parser.add_argument("--run", type=Path, default=Path("runs/train/helmet-pilot-v1-isolated-smoke"))
    parser.add_argument("--output", type=Path, default=Path("reports/data-revision-integration.json"))
    args = parser.parse_args()
    release_check = verify_release(args.release)
    metadata = json.loads((args.release / "manifest.json").read_text())
    environment = json.loads((args.run / "environment.json").read_text())
    settings = yaml.safe_load((args.run / "args.yaml").read_text())
    with (args.run / "results.csv").open() as stream:
        epochs = list(csv.DictReader(stream))
    training_root = Path(environment["training_data_yaml"]).parent
    checks = {
        "release_verifies_after_training": release_check["passed"],
        "smoke_completed_one_epoch": len(epochs) == 1 and settings["epochs"] == 1 and (args.run / "weights/best.pt").is_file(),
        "smoke_bound_to_reviewed_manifest": environment["data_verification"]["manifest_sha256"] == release_check["manifest_sha256"],
        "training_used_separate_working_copy": training_root.resolve() != args.release.resolve() and training_root.is_dir(),
        "framework_cache_isolated": (training_root / "labels/train.cache").is_file() and not (args.release / "labels/train.cache").exists(),
        "active_pt_unchanged": file_hash(Path("models/best.pt")) == json.loads(Path("reports/pytorch-cpu.json").read_text())["model_sha256"],
        "active_onnx_unchanged": file_hash(Path("models/best.onnx")) == json.loads(Path("reports/onnx-cpu.json").read_text())["model_sha256"],
    }
    inventory_path = Path("runs/shel5k-review/inventory.json")
    inventory = json.loads(inventory_path.read_text())
    checks["review_bound_to_inventory"] = metadata["inventory_sha256"] == file_hash(inventory_path)
    original = {r["id"]: r for r in inventory["records"]}
    source = Path(inventory["source_root"])
    checks["accepted_source_files_unchanged"] = all(
        file_hash(source / original[r.get("source_id", r["id"])][field]) == original[r.get("source_id", r["id"])][digest]
        for r in metadata["files"] for field, digest in (("image", "image_sha256"), ("annotation", "annotation_sha256")))
    with tempfile.TemporaryDirectory(prefix="ppe-colab-roundtrip-") as temporary:
        relocated = (Path(temporary) / "release").resolve()
        safe_extract(args.package, relocated)
        config_path = relocated / "data.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["path"] = str(relocated)
        config_path.write_text(yaml.safe_dump(config))
        relocation = verify_release(relocated)
        checks["colab_package_relocation_passed"] = relocation["passed"]
        checks["package_contains_attribution"] = (relocated / "ATTRIBUTION.txt").is_file()
    from ppe.detector import YoloDetector
    try:
        YoloDetector(str(args.run / "weights/best.pt"))
    except ValueError as exc:
        checks["incompatible_pilot_rejected_by_ppe_service"] = "vest" in str(exc)
    else:
        checks["incompatible_pilot_rejected_by_ppe_service"] = False
    report = {"passed": all(checks.values()), "checks": checks, "release_summary": metadata["summary"],
              "smoke": {"epochs": settings["epochs"], "imgsz": settings["imgsz"], "device": settings["device"],
                        "weights_sha256": file_hash(args.run / "weights/best.pt"), "run": str(args.run)},
              "package_sha256": file_hash(args.package), "manifest_sha256": release_check["manifest_sha256"],
              "relocation_errors": relocation["errors"],
              "scope": "Pipeline integration only; no model improvement, independent human QA, held-out test or deployment accuracy claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
