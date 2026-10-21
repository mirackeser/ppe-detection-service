"""Conservative, explicit validation-only gate; never copies/promotes weights itself."""
import argparse
import json
from pathlib import Path


def decide(baseline, candidate, baseline_calibration, candidate_calibration):
    for evaluation, calibration in [(baseline, baseline_calibration), (candidate, candidate_calibration)]:
        if evaluation["model_sha256"] != calibration["model_sha256"] or evaluation["split"] != "val" or calibration["split"] != "val":
            raise ValueError("Evaluation/calibration provenance mismatch or wrong split")
        if evaluation["imgsz"] != calibration["imgsz"]:
            raise ValueError("Calibration resolution differs from evaluation")
    if baseline["imgsz"] != candidate["imgsz"]:
        raise ValueError("Use identical inference resolution for this experiment")
    def missing_helmet(evaluation):
        return next(row for row in evaluation["class_metrics"] if row["Class"] == "no_helmet")
    def fixed(calibration):
        return next(row for row in calibration["thresholds"] if row["threshold"] == .25)
    b, c = missing_helmet(baseline), missing_helmet(candidate)
    bf, cf = fixed(baseline_calibration), fixed(candidate_calibration)
    checks = {"no_helmet_AP50_gain_at_least_0.02": c["mAP50"]-b["mAP50"] >= .02,
              "overall_mAP50_regression_at_most_0.01": candidate["mAP50"] >= baseline["mAP50"]-.01,
              "fixed_threshold_recall_not_lower": cf["recall"] >= bf["recall"],
              "fixed_threshold_precision_not_lower": cf["precision"] >= bf["precision"]}
    return {"promote_candidate": all(checks.values()), "checks": checks,
            "baseline_sha256": baseline["model_sha256"], "candidate_sha256": candidate["model_sha256"],
            "baseline_no_helmet": b, "candidate_no_helmet": c,
            "baseline_fixed_threshold": bf, "candidate_fixed_threshold": cf,
            "limitations": "Validation-only model selection on a small sample. No claim of statistical significance or test performance."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-eval", type=Path, default=Path("reports/validation-pytorch.json"))
    parser.add_argument("--candidate-eval", type=Path, default=Path("reports/validation-cls1.json"))
    parser.add_argument("--baseline-calibration", type=Path, default=Path("reports/calibration-baseline.json"))
    parser.add_argument("--candidate-calibration", type=Path, default=Path("reports/calibration-cls1.json"))
    args = parser.parse_args()
    result = decide(*[json.loads(path.read_text()) for path in [args.baseline_eval, args.candidate_eval,
                                                               args.baseline_calibration, args.candidate_calibration]])
    Path("reports/model-selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Validation gate: promote_candidate={result['promote_candidate']}")


if __name__ == "__main__":
    main()
