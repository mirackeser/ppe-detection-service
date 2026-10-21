"""Fingerprint development validation data independently of training expansion."""
import hashlib
import json


def validation_fingerprint(records):
    keys = ("id", "source_id", "scene_id", "image_sha256", "label_sha256", "sample_role", "lighting", "crop")
    rows = [{key:r.get(key) for key in keys} for r in records if r["split"] == "val"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Validation cohort is empty or has duplicate IDs")
    payload = json.dumps(sorted(rows,key=lambda r:r["id"]),sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_validation_lock(records, lock):
    actual = validation_fingerprint(records)
    expected = validation_fingerprint(lock["samples"])
    if lock.get("validation_fingerprint",expected) != expected or actual != expected:
        raise ValueError("Validation cohort differs from frozen lock; use a new explicit cohort version")
    return actual
