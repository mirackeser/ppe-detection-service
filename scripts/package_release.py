"""Package only verified release files, including source attribution, for Colab."""
import argparse
import json
from pathlib import Path
import zipfile

from scripts.curate_dataset import checked_path
from scripts.verify_release import verify_release


def package_release(root, output):
    verification = verify_release(root)
    if not verification["passed"]:
        raise ValueError(f"Release verification failed: {verification['errors']}")
    metadata = json.loads((root / "manifest.json").read_text())
    files = {"data.yaml", "manifest.json", "review.json"}
    files.update(r[field] for r in metadata["files"] for field in ("image", "label"))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previously shared packages.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.write(checked_path(root, name), name)
        source = metadata["source"]
        attribution = "\n".join(str(source.get(k, "")) for k in ("attribution", "source", "license_as_published", "license_url", "paper"))
        archive.writestr("ATTRIBUTION.txt", attribution+"\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(package_release(args.root, args.output))


if __name__ == "__main__":
    main()
