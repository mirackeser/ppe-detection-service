"""Acquire the published SHEL5K v4 release as an immutable local source.

This only imports source data; it does not approve labels or start training.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from scripts.prepare_data import download

SOURCE = "https://data.mendeley.com/datasets/9rcv8mm682/4"
URL = "https://data.mendeley.com/public-api/zip/9rcv8mm682/download/4"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_extract(archive, destination):
    """Validate all members before extraction; refuse overwrites and links."""
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 30000 or sum(e.file_size for e in entries) > 4 * 1024**3:
            raise ValueError("Unexpected archive size")
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or stat.S_ISLNK(mode) or entry.filename in names):
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
            names.add(entry.filename)
        destination.mkdir(parents=True)
        try:
            # zipfile verifies CRC while reading. No downloaded code is executed.
            bundle.extractall(destination)
        except Exception:
            shutil.rmtree(destination)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/shel5k-v4-source"))
    parser.add_argument("--archive", type=Path, default=Path("datasets/shel5k-v4.zip"))
    args = parser.parse_args()
    if args.root.exists():
        parser.error("Source destination exists; never overwrite an imported source")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.archive.exists():
        partial = args.archive.with_suffix(".partial")
        download(URL, partial)
        partial.replace(args.archive)
    safe_extract(args.archive, args.root)
    provenance = {"dataset": "SHEL5K", "release": 4, "doi": "10.17632/9rcv8mm682.4",
                  "source": SOURCE, "download_url": URL, "archive_sha256": file_hash(args.archive),
                  "archive_bytes": args.archive.stat().st_size,
                  "license_as_published": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                  "attribution": "Gochoo, Munkhjargal (2022), Safety Helmet detection with Extended Labels 5K images (SHEL5K), Mendeley Data V4, doi:10.17632/9rcv8mm682.4",
                  "paper": "https://doi.org/10.3390/s22062315",
                  "status": "source_import_only; annotation and scene review pending",
                  "scope": "Helmet/head/person annotations; no vest annotations"}
    (args.root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    Path("reports/shel5k-source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Imported {args.root}; not approved for training.", flush=True)


if __name__ == "__main__":
    main()
