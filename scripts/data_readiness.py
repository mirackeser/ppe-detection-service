"""Audit explicit dataset coverage targets without confusing them with test success."""
import argparse
import json
from pathlib import Path

from scripts.prepare_shel5k import file_hash
from scripts.verify_release import verify_release


def assess_readiness(manifest, verification, policy):
    checks = [{"name":"release_integrity", "passed": verification["passed"]}]
    items = manifest["files"]
    actual = {}
    for split in ("train", "val"):
        rows = [r for r in items if r["split"] == split]
        positive = {r["scene_id"] for r in rows if r.get("sample_role", "positive") == "positive"}
        negative = {r["scene_id"] for r in rows if r.get("sample_role") == "background_negative"}
        counts = verification["class_instances"].get(split, {})
        lighting = {condition: len({r["scene_id"] for r in rows if r.get("lighting") == condition})
                    for condition in policy["minimum_lighting_scene_groups"]}
        actual[split] = {"samples":len(rows), "positive_scene_groups":len(positive), "negative_scene_groups":len(negative),
                         "unique_sources":len({r.get("source_id",r["id"]) for r in rows}),
                         "class_instances":counts, "lighting_scene_groups":lighting}
        requirements = [("positive_scene_groups",len(positive),policy["minimum_positive_scene_groups"][split]),
                        ("negative_scene_groups",len(negative),policy["minimum_negative_scene_groups"][split])]
        requirements += [(f"instances/{name}",counts.get(name,0),minimum) for name,minimum in policy["minimum_class_instances"][split].items()]
        requirements += [(f"lighting/{name}",lighting[name],minimum) for name,minimum in policy["minimum_lighting_scene_groups"].items()]
        checks.extend({"name":f"{split}/{name}", "actual":value, "minimum":minimum, "passed":value>=minimum}
                      for name,value,minimum in requirements)
    return {"policy_id":policy["policy_id"], "audit_completed":True,
            "substantive_training_ready":all(c["passed"] for c in checks), "deployment_ready":False,
            "actual":actual, "checks":checks, "unmet_requirements":[c for c in checks if not c["passed"]],
            "scope":policy["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--policy", type=Path, default=Path("config/data-readiness-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    result = assess_readiness(manifest, verify_release(args.root), json.loads(args.policy.read_text()))
    result.update(manifest_sha256=file_hash(args.root / "manifest.json"), policy_sha256=file_hash(args.policy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:result[k] for k in ("audit_completed","substantive_training_ready","actual","unmet_requirements")},indent=2))
    raise SystemExit(1 if args.require_ready and not result["substantive_training_ready"] else 0)


if __name__ == "__main__":
    main()
