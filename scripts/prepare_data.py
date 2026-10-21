"""Download the official dataset without executing embedded download scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
         "no_helmet", "no_goggle", "no_gloves", "no_boots"]


def download(url: str, partial: Path):
    """Resume byte ranges after transient connection failures; never accept a wrong offset."""
    failures = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={offset}-{offset + 4 * 1024 * 1024 - 1}",
            "User-Agent": "PPE-Portfolio-Dataset-Downloader/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status == 206:
                    import re
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if match is None or int(match[1]) != offset:
                        raise ValueError("Server returned an unexpected byte range")
                    body = response.read()
                    if len(body) != int(match[2]) - offset + 1:
                        raise OSError("Incomplete download range")
                    with partial.open("ab") as output:
                        output.write(body)
                    total = int(match[3])
                else:
                    # Some servers ignore Range. Restart rather than appending corrupt data.
                    with partial.open("wb") as output:
                        shutil.copyfileobj(response, output)
                    total = int(response.headers.get("Content-Length", partial.stat().st_size))
            size = partial.stat().st_size
            print(f"Downloaded {size / 1048576:.1f}/{total / 1048576:.1f} MiB", flush=True)
            if size == total:
                return
            if size > total:
                raise ValueError("Downloaded file exceeds expected length")
            failures = 0
        except (OSError, urllib.error.URLError) as error:
            failures += 1
            if failures >= 4:
                raise
            print(f"Download interrupted; retaining partial file. Retry {failures}: {error}", flush=True)
            time.sleep(failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "construction-ppe.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        download(URL, partial)
        partial.replace(archive)
    # Validate the entire archive before writing any member.
    root = destination / "construction-ppe"
    with zipfile.ZipFile(archive) as bundle:
        # Official release currently stores images/labels directly at ZIP root.
        prefixed = any(name.startswith("construction-ppe/images/") for name in bundle.namelist())
        extraction_root = destination if prefixed else root
        extraction_root.mkdir(parents=True, exist_ok=True)
        for entry in bundle.infolist():
            path = (extraction_root / entry.filename).resolve()
            if not path.is_relative_to(extraction_root) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
        bundle.extractall(extraction_root)
    for split in ("train", "val", "test"):
        if not (root / "images" / split).is_dir() or not (root / "labels" / split).is_dir():
            raise ValueError(f"Unexpected dataset layout: {split}")
    import yaml
    config = root / "data.yaml"
    original = yaml.safe_load(config.read_text(encoding="utf-8"))
    original_names = original.get("names")
    if original_names != dict(enumerate(NAMES)):
        raise ValueError("Official archive class mapping changed; inspect before remapping")
    (root / "data.source.yaml").write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    config.write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                      "val": "images/val", "test": "images/test",
                                      "names": dict(enumerate(NAMES))}), encoding="utf-8")
    provenance = {"source": URL, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "documentation": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
                  "license_as_documented": "AGPL-3.0", "class_mapping": NAMES}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared {config}; run scripts.audit_data before training")


if __name__ == "__main__":
    main()
