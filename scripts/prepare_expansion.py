"""Select sparse expansion candidates before inference, with reproducible strata."""
import argparse
import json
from pathlib import Path
import random

from scripts.prepare_shel5k import file_hash
from scripts.shel5k_inventory import review_sheets


def select_candidates(records, excluded_ids, seed=20261021,
                      counts=None):
    counts = counts or {"helmet_only": 36, "mixed": 24, "head_only": 12}
    groups = {name: [] for name in counts}
    for row in sorted(records, key=lambda r: r["id"]):
        if row["id"] in excluded_ids:
            continue
        targets = [o for o in row["objects"] if o["label"] is not None]
        heads = [o for o in targets if o["label"] in {"helmet", "no_helmet"}]
        if not heads or len(heads) > 6 or any(o["difficult"] for o in targets):
            continue
        labels = {o["label"] for o in heads}
        name = "mixed" if len(labels) == 2 else "helmet_only" if "helmet" in labels else "head_only"
        groups[name].append(row)
    rng, selected = random.Random(seed), []
    for name, count in counts.items():
        if len(groups[name]) < count:
            raise ValueError(f"Insufficient candidates in stratum {name}")
        selected.extend(rng.sample(groups[name], count))
    return selected, {name: len(rows) for name, rows in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("runs/shel5k-review/inventory.json"))
    parser.add_argument("--previous-review", type=Path, default=Path("reports/shel5k-pilot-review.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; candidate batches are preserved")
    inventory = json.loads(args.inventory.read_text())
    previous = json.loads(args.previous_review.read_text())
    selected, pools = select_candidates(inventory["records"], {r["id"] for r in previous["items"]})
    args.output.mkdir(parents=True)
    metadata = {"seed": 20261021, "inventory_sha256": file_hash(args.inventory),
                "previous_review_sha256": file_hash(args.previous_review),
                "scope": "Pre-inference candidates, <=6 head/helmet objects; not label approval or evaluation data",
                "pool_counts": pools, "ids": [r["id"] for r in selected]}
    (args.output / "candidate-ids.json").write_text(json.dumps(metadata, indent=2))
    review_sheets(selected, Path(inventory["source_root"]), args.output)
    print(f"Prepared {len(selected)} candidates in {args.output}")


if __name__ == "__main__":
    main()
