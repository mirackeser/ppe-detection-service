"""Evaluate a frozen, reviewed background-control cohort at a fixed threshold.

Reports false detections per image/group, not temporal alarm rates or full model
accuracy. Positive samples provide only a nonempty-output sanity check.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

from ppe.detector import YoloDetector
from ppe.policy import assess
from scripts.prepare_shel5k import file_hash
from scripts.shel5k_inventory import TARGET_NAMES
from scripts.verify_release import verify_release
from scripts.cohort import check_validation_lock, validation_fingerprint


def summarize(rows):
    result = {}
    for split in ("train", "val"):
        controls = [r for r in rows if r["split"] == split and r["sample_role"] == "background_negative"]
        groups = {r["scene_id"] for r in controls}
        affected = {r["scene_id"] for r in controls if r["false_detection"]}
        result[split] = {"negative_images": len(controls), "source_groups": len(groups),
                         "images_with_false_detection": sum(r["false_detection"] for r in controls),
                         "image_fraction_with_false_detection": sum(r["false_detection"] for r in controls)/len(controls) if controls else None,
                         "groups_with_false_detection": len(affected),
                         "group_fraction_with_false_detection": len(affected)/len(groups) if groups else None,
                         "false_detection_boxes": sum(len(r["target_detections"]) for r in controls),
                         "predicted_classes": dict(Counter(d["label"] for r in controls for d in r["target_detections"])),
                         "single_frame_missing_candidates": sum(r["single_frame_missing_candidates"] for r in controls)}
    return result


def evaluate(root, detector, confidence=.25, cohort_lock=None):
    if not .25 <= confidence <= 1:
        raise ValueError("Confidence must be >= detector's 0.25 prediction floor and <=1")
    verification = verify_release(root)
    if not verification["passed"]:
        raise ValueError(f"Unverified cohort: {verification['errors']}")
    manifest = json.loads((root / "manifest.json").read_text())
    cohort_hash = validation_fingerprint(manifest["files"])
    if cohort_lock is not None:
        check_validation_lock(manifest["files"],cohort_lock)
    if not any(r.get("sample_role") == "background_negative" for r in manifest["files"]):
        raise ValueError("No reviewed negative controls; no false-detection rate can be measured")
    rows = []
    for record in manifest["files"]:
        with Image.open(root / record["image"]) as source:
            image = source.convert("RGB")
        detections = detector.predict(image)
        target = [d.model_dump(mode="json") for d in detections if d.label in TARGET_NAMES and d.confidence >= confidence]
        negative = record.get("sample_role", "positive") == "background_negative"
        rows.append({"id": record["id"], "source_id": record.get("source_id", record["id"]),
                     "scene_id": record["scene_id"], "split": record["split"],
                     "sample_role": record.get("sample_role", "positive"), "lighting": record.get("lighting"),
                     "image_sha256": record["image_sha256"], "target_detections": target,
                     "false_detection": negative and bool(target),
                     "single_frame_missing_candidates": sum(a.helmet == "missing" for a in assess(detections))})
    positive_rows = [r for r in rows if r["sample_role"] == "positive"]
    sanity_passed = any(r["target_detections"] for r in positive_rows)
    return {"utc_time": datetime.now(timezone.utc).isoformat(), "evaluation_completed": True,
            "protocol_version": "background-controls-v1", "data_yaml_sha256": verification["data_yaml_sha256"],
            "validation_fingerprint":cohort_hash, "cohort_lock_checked":cohort_lock is not None,
            "positive_output_sanity_passed": sanity_passed,
            "positive_images_with_detections": sum(bool(r["target_detections"]) for r in positive_rows),
            "positive_images": len(positive_rows), "manifest_sha256": verification["manifest_sha256"],
            "review_sha256": manifest["review_sha256"], "model": detector.model,
            "model_sha256": detector.model_sha256, "runtime": detector.runtime, "imgsz": detector.imgsz,
            "confidence": confidence, "target_classes": TARGET_NAMES, "by_split": summarize(rows), "rows": rows,
            "single_frame_policy": {"person_helmet_vest_confidence": .5, "no_helmet_confidence": .25,
                                    "policy_source_sha256": file_hash(Path("ppe/policy.py")),
                                    "scope": "Geometric single-frame candidates; no temporal event evaluation"},
            "scope": "Reviewed derived-background controls only; train results are diagnostics, validation is a small development cohort; no temporal alarms or general field accuracy",
            "limitations": ["Cropped controls inherit their original scene; sample count is not independent scene count.",
                            "No confidence interval or production false-alarm rate is inferred from this small cohort.",
                            "Zero control detections would not establish good recall; positive-output check is only a sanity check.",
                            "Some source images may overlap historical pretrained/baseline data; not an independent held-out benchmark."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path)
    args = parser.parse_args()
    detector = YoloDetector(str(args.weights), device="cpu", imgsz=640)
    result = evaluate(args.root, detector, cohort_lock=json.loads(args.cohort_lock.read_text()) if args.cohort_lock else None)
    if args.cohort_lock:
        result["cohort_lock_sha256"] = file_hash(args.cohort_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"evaluation_completed": True, "positive_output_sanity_passed": result["positive_output_sanity_passed"], "by_split": result["by_split"]}, indent=2))
    # Evaluation completion is not a model-quality acceptance decision.
    raise SystemExit(0 if result["positive_output_sanity_passed"] else 1)


if __name__ == "__main__":
    main()
