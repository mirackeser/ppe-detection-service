import copy
import hashlib
import json
from pathlib import Path
import random
import zipfile

from PIL import Image
import pytest

from scripts.curate_dataset import apply_reviewed_crop, assign_splits, build_release, corrected_objects, plan_release
from scripts.prepare_shel5k import file_hash, safe_extract
from scripts.shel5k_inventory import MAPPING, TARGET_NAMES, parse_voc
from scripts.verify_release import verify_release
from scripts.prepare_expansion import select_candidates
from scripts.train import stage_training_data, training_data_gate
from scripts.train import main as train_main
from scripts.evaluate_background import evaluate as evaluate_background


def xml_file(tmp_path, name="head", box=(1, 1, 20, 30)):
    path = tmp_path / "sample.xml"
    path.write_text(f"""<annotation><filename>sample.png</filename><size><width>40</width><height>40</height></size>
    <object><name>{name}</name><bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox></object></annotation>""")
    return path


def test_voc_conversion_preserves_inclusive_pixels_and_maps_head(tmp_path):
    objects, warnings = parse_voc(xml_file(tmp_path), (40, 40))
    assert objects[0]["box"] == [0, 0, 20, 30]
    assert objects[0]["label"] == "no_helmet"
    assert not warnings
    objects, warnings = parse_voc(xml_file(tmp_path, box=(0, 0, 20, 30)), (40, 40))
    assert objects[0]["box"] == [0, 0, 20, 30]
    assert warnings[0]["kind"] == "zero_minimum_clamped"


@pytest.mark.parametrize("name,box,size", [("unexpected", (1,1,20,30), (40,40)),
                                           ("head", (1,1,20,30), (80,40)),
                                           ("head", (21,1,20,30), (40,40)),
                                           ("head", (1,1,41,30), (40,40)),
                                           ("head", (1,1,float('nan'),30), (40,40))])
def test_invalid_source_is_quarantined_not_silently_remapped(tmp_path, name, box, size):
    with pytest.raises(ValueError):
        parse_voc(xml_file(tmp_path, name, box), size)


def test_xml_entity_declarations_rejected(tmp_path):
    p = xml_file(tmp_path)
    p.write_text('<!DOCTYPE annotation [<!ENTITY a "test">]>'+p.read_text())
    with pytest.raises(ValueError):
        parse_voc(p, (40,40))


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "folder\\escape.txt"])
def test_archive_validation_precedes_any_extraction(tmp_path, member):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("legitimate.txt", "ok")
        out.writestr(member, "bad")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def release_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    scenes = {"train": [], "val": []}
    for i in range(100):
        scene = f"scene-{i}"
        split = "val" if int(hashlib.sha256(("ppe-pilot-v1:"+scene).encode()).hexdigest()[:8], 16)%5 == 0 else "train"
        if len(scenes[split]) < 2:
            scenes[split].append(scene)
    records, items = [], []
    for i, scene in enumerate(scenes["train"]+scenes["val"]):
        identity = f"img{i}"
        image = source / f"{identity}.png"
        Image.frombytes("RGB", (40,40), random.Random(i).randbytes(40*40*3)).save(image)
        annotation = source / f"{identity}.xml"
        annotation.write_text("fixture annotation")
        objects = [{"source_object": n+1, "source_label": label, "label": label,
                    "box": [n*10,0,n*10+8,20], "source_box": [n*10+1,1,n*10+8,20],
                    "difficult": False, "truncated": False} for n, label in enumerate(TARGET_NAMES)]
        record = {"id": identity, "image": image.name, "annotation": annotation.name,
                  "image_sha256": file_hash(image), "annotation_sha256": file_hash(annotation),
                  "pixel_sha256": f"fixture-pixels-{i}", "dhash128": f"{((1<<32)-1) << (i*32):032x}",
                  "size": [40,40], "objects": objects, "conversion_warnings": []}
        records.append(record)
        items.append({"id": identity, "image_sha256": record["image_sha256"], "annotation_sha256": record["annotation_sha256"],
                      "decision": "accept", "scene_id": scene, "domain": "industrial", "labels_reviewed": True,
                      "notes": "Test fixture", "corrections": []})
    inventory = {"source_root": str(source), "source": {"dataset": "fixture"}, "mapping": MAPPING,
                 "target_names": TARGET_NAMES, "records": records}
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    review = {"reviewer": "test only", "inventory_sha256": file_hash(inventory_path), "items": items}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    return inventory_path, review_path, inventory, review


def test_release_roundtrip_and_post_export_tamper_detection(tmp_path):
    inventory_path, review_path, inventory, review = release_inputs(tmp_path)
    destination = tmp_path / "pilot-v1"
    result = build_release(inventory_path, review_path, destination)
    assert result["summary"]["images"] == {"train": 2, "val": 2}
    assert verify_release(destination)["pilot_training_ready"]
    assert not verify_release(destination)["deployment_ready"]
    assert not (destination / "images/test").exists()
    label = destination / result["files"][0]["label"]
    label.write_text("0 .5 .5 2 2\n")
    verified = verify_release(destination)
    assert not verified["passed"]
    assert any("geometry" in error for error in verified["errors"])


@pytest.mark.parametrize("fault", ["unreviewed", "wrong_hash", "domain", "scene", "duplicates", "similarity", "difficult"])
def test_incomplete_quality_gates_prevent_export(tmp_path, fault):
    _, _, inventory, review = release_inputs(tmp_path)
    item = review["items"][0]
    if fault == "unreviewed": item["labels_reviewed"] = False
    elif fault == "wrong_hash": item["image_sha256"] = "stale"
    elif fault == "domain": item["domain"] = "sports"
    elif fault == "scene": item["scene_id"] = None
    elif fault == "duplicates": inventory["records"][1]["pixel_sha256"] = inventory["records"][0]["pixel_sha256"]
    elif fault == "similarity": inventory["records"][1]["dhash128"] = inventory["records"][0]["dhash128"]
    else: inventory["records"][0]["objects"][0]["difficult"] = True
    with pytest.raises(ValueError):
        plan_release(inventory, review)


def test_pending_images_are_excluded_and_source_changes_fail_before_output(tmp_path):
    inventory_path, review_path, inventory, review = release_inputs(tmp_path)
    extra = copy.deepcopy(review["items"][0]); extra["id"] = "pending_img"; extra["decision"] = "pending"
    record = copy.deepcopy(inventory["records"][0]); record["id"] = "pending_img"
    inventory["records"].append(record); review["items"].append(extra)
    selected, _ = plan_release(inventory, review)
    assert len(selected) == 4
    Path(inventory["source_root"], "img0.xml").write_text("changed")
    with pytest.raises(ValueError, match="Source changed"):
        build_release(inventory_path, review_path, tmp_path / "pilot")
    assert not (tmp_path / "pilot").exists()


def test_corrections_are_bound_to_old_box_and_do_not_mutate_inventory(tmp_path):
    _, _, inventory, review = release_inputs(tmp_path)
    record, item = inventory["records"][0], review["items"][0]
    old = copy.deepcopy(record["objects"][0]["box"])
    item["corrections"] = [{"source_object": 1, "old_box": old, "new_box": [0,1,9,21], "reason": "Fixture correction"}]
    assert corrected_objects(record, item)[0]["box"] == [0,1,9,21]
    assert record["objects"][0]["box"] == old
    item["corrections"][0]["old_box"] = [1,1,2,2]
    with pytest.raises(ValueError, match="original annotation"):
        corrected_objects(record, item)


def test_release_never_overwrites_existing_version(tmp_path):
    inventory_path, review_path, _, _ = release_inputs(tmp_path)
    destination = tmp_path / "pilot"
    destination.mkdir(); (destination / "keep.txt").write_text("user work")
    with pytest.raises(FileExistsError):
        build_release(inventory_path, review_path, destination)
    assert (destination / "keep.txt").read_text() == "user work"


def test_training_gate_blocks_unreviewed_and_modified_data(tmp_path):
    with pytest.raises(ValueError, match="reviewed release"):
        training_data_gate(tmp_path / "data.yaml")
    assert training_data_gate(tmp_path / "data.yaml", legacy_baseline=True)["explicit_legacy_override"]
    inventory_path, review_path, _, _ = release_inputs(tmp_path)
    destination = tmp_path / "pilot"
    result = build_release(inventory_path, review_path, destination)
    assert training_data_gate(destination / "data.yaml")["pilot_training_ready"]
    (destination / result["files"][0]["label"]).write_text("0 .5 .5 .5 .5\n")
    with pytest.raises(ValueError, match="verification failed"):
        training_data_gate(destination / "data.yaml")


def test_crop_preserves_head_and_translates_clipped_person():
    objects = [{"source_object": 1, "label": "no_helmet", "box": [10,20,30,40]},
               {"source_object": 2, "label": "person", "box": [0,20,50,100]},
               {"source_object": 3, "label": "helmet", "box": [10,0,30,8]}]
    before = copy.deepcopy(objects)
    result, size, log = apply_reviewed_crop(objects, (100,100), {"crop": [0,10,60,80], "crop_reason": "Remove reflected border"})
    assert size == [60,70]
    assert [o["box"] for o in result] == [[10,10,30,30], [0,10,50,70]]
    assert {r["action"] for r in log} == {"clipped_to_crop", "outside_reviewed_crop"}
    assert objects == before


@pytest.mark.parametrize("crop,reason", [([0,30,60,80], "clips head"), ([0,0,101,80], "outside"),
                                         ([0,0,60,80], ""), ([0.,0,60,80], "float")])
def test_unreviewed_or_head_intersecting_crop_is_rejected(crop, reason):
    objects = [{"source_object": 1, "label": "no_helmet", "box": [10,20,30,40]}]
    with pytest.raises(ValueError):
        apply_reviewed_crop(objects, (100,100), {"crop": crop, "crop_reason": reason})


def test_cropped_release_has_source_and_derived_provenance(tmp_path):
    inventory_path, review_path, inventory, review = release_inputs(tmp_path)
    for item in review["items"]:
        item.update(crop=[0,0,40,30], crop_reason="Test reviewed crop")
    review_path.write_text(json.dumps(review))
    result = build_release(inventory_path, review_path, tmp_path / "cropped")
    assert verify_release(tmp_path / "cropped")["passed"]
    first = result["files"][0]
    assert first["source_image_sha256"] != first["image_sha256"]
    assert first["size"] == [40,30]


def test_group_assignment_is_stable_when_input_order_changes(tmp_path):
    _, _, inventory, review = release_inputs(tmp_path)
    first, _ = plan_release(inventory, review)
    review["items"].reverse()
    second, _ = plan_release(inventory, review)
    assert {r["record"]["id"]:r["split"] for r in first} == {r["record"]["id"]:r["split"] for r in second}


def test_independent_verifier_catches_post_crop_duplicates(tmp_path):
    inventory_path, review_path, _, _ = release_inputs(tmp_path)
    destination = tmp_path / "pilot"
    result = build_release(inventory_path, review_path, destination)
    first, second = result["files"][:2]
    (destination / second["image"]).write_bytes((destination / first["image"]).read_bytes())
    # Even if byte hashes in the manifest are refreshed, decoded duplicate check must fail.
    manifest = json.loads((destination / "manifest.json").read_text())
    manifest["files"][1]["image_sha256"] = file_hash(destination / second["image"])
    (destination / "manifest.json").write_text(json.dumps(manifest))
    assert any("Duplicate decoded" in e for e in verify_release(destination)["errors"])


def test_training_uses_verified_working_copy_for_framework_caches(tmp_path):
    inventory_path, review_path, _, _ = release_inputs(tmp_path)
    release = tmp_path / "release"
    build_release(inventory_path, review_path, release)
    staged = stage_training_data(release / "data.yaml", tmp_path / "training-inputs")
    assert verify_release(staged.parent)["passed"]
    (staged.parent / "labels/train.cache").write_bytes(b"framework cache")
    assert not (release / "labels/train.cache").exists()
    assert verify_release(release)["passed"]


def test_small_pilot_blocks_long_training_before_staging(tmp_path, monkeypatch, capsys):
    inventory_path, review_path, _, _ = release_inputs(tmp_path)
    release = tmp_path / "release"
    build_release(inventory_path, review_path, release)
    monkeypatch.setattr("sys.argv", ["train", "--data", str(release / "data.yaml"), "--epochs", "50"])
    with pytest.raises(SystemExit) as exc:
        train_main()
    assert exc.value.code == 2
    assert "at most 3" in capsys.readouterr().err


def test_expanded_group_split_is_balanced_stable_and_keeps_families_together():
    rows = []
    for i in range(30):
        for j in range(2 if i%4 == 0 else 1):
            labels = ["helmet", "person"] if i%3 == 0 else ["no_helmet", "person"] if i%3 == 1 else TARGET_NAMES
            rows.append({"record": {"id": f"sample-{i}-{j}"}, "review": {"scene_id": f"scene-{i}"},
                         "objects": [{"label": label} for label in labels]})
    with pytest.raises(ValueError, match="larger reviewed releases"):
        assign_splits(copy.deepcopy(rows))
    first, second = copy.deepcopy(rows), list(reversed(copy.deepcopy(rows)))
    assign_splits(first, "reviewed-group-v3")
    assign_splits(second, "reviewed-group-v3")
    assert {r["record"]["id"]:r["split"] for r in first} == {r["record"]["id"]:r["split"] for r in second}
    groups = {s: {r["review"]["scene_id"] for r in first if r["split"] == s} for s in ("train", "val")}
    assert len(groups["val"]) == 6 and not groups["train"] & groups["val"]
    from collections import Counter
    total = Counter(o["label"] for r in first for o in r["objects"])
    val = Counter(o["label"] for r in first if r["split"] == "val" for o in r["objects"])
    assert all(abs(val[name]/total[name]-.2) < .07 for name in TARGET_NAMES)


def test_unknown_split_policy_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown frozen"):
        assign_splits([], "typo-policy")


def test_candidate_sampling_is_stratified_stable_and_excludes_previous_reviews():
    rows = []
    for group, labels in (("helmet", ["helmet"]), ("head", ["no_helmet"]), ("mixed", ["helmet", "no_helmet"])):
        for i in range(5):
            rows.append({"id": f"{group}-{i}", "objects": [{"label": label, "difficult": False} for label in labels]})
    excluded = {"helmet-0", "head-0", "mixed-0"}
    counts = {"helmet_only": 2, "mixed": 2, "head_only": 2}
    selected, pools = select_candidates(rows, excluded, counts=counts)
    reversed_selected, _ = select_candidates(list(reversed(rows)), excluded, counts=counts)
    assert [r["id"] for r in selected] == [r["id"] for r in reversed_selected]
    assert len(selected) == 6 and all(r["id"] not in excluded for r in selected)
    assert pools == {"helmet_only": 4, "mixed": 4, "head_only": 4}
    rows[0]["objects"][0]["difficult"] = True
    with pytest.raises(ValueError, match="Insufficient"):
        select_candidates(rows, set(), counts={"helmet_only": 5, "mixed": 1, "head_only": 1})


def add_background_variant(review):
    item = copy.deepcopy(review["items"][0])
    item.update(id="img0-background", source_id="img0", sample_role="background_negative",
                crop=[0,25,40,40], crop_reason="Reviewed object-free area",
                negative_review={"confirmed_absent": True, "reviewed_classes": TARGET_NAMES,
                                 "reviewer": "test fixture", "reason": "Object-free test crop"})
    review["items"].append(item)
    return item


def test_background_variant_keeps_source_group_and_exports_empty_labels(tmp_path):
    ip, rp, _, review = release_inputs(tmp_path)
    add_background_variant(review); rp.write_text(json.dumps(review))
    destination = tmp_path / "release"
    metadata = build_release(ip, rp, destination)
    positive = next(r for r in metadata["files"] if r["id"] == "img0")
    negative = next(r for r in metadata["files"] if r["id"] == "img0-background")
    assert negative["source_id"] == "img0" and negative["split"] == positive["split"]
    assert (destination / negative["label"]).read_bytes() == b""
    assert verify_release(destination)["passed"]


@pytest.mark.parametrize("fault", ["missing_confirmation", "contradictory_objects", "different_scene", "implicit_empty", "ignored_face"])
def test_invalid_negative_or_cross_scene_variant_is_rejected(tmp_path, fault):
    _, _, inventory, review = release_inputs(tmp_path)
    item = add_background_variant(review)
    if fault == "missing_confirmation": item.pop("negative_review")
    elif fault == "contradictory_objects": item["crop"] = [0,0,40,40]
    elif fault == "different_scene": item["scene_id"] = "fake-new-scene"
    elif fault == "implicit_empty": item.pop("sample_role")
    else: inventory["records"][0]["objects"].append({"source_object": 4, "label": None, "box": [0,25,10,35], "difficult": False})
    with pytest.raises(ValueError):
        plan_release(inventory, review)


def test_context_split_requires_lighting_and_negative_groups_per_split():
    rows = []
    for i in range(30):
        context = {"scene_id":f"scene-{i}", "lighting":"low_light" if i%3 == 0 else "normal", "sample_role":"positive"}
        rows.append({"review":context, "objects":[{"label":n} for n in TARGET_NAMES]})
        if i%2 == 0:
            rows.append({"review":{**context,"sample_role":"background_negative"},"objects":[]})
    assign_splits(rows, "reviewed-context-v4")
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        for lighting in ("low_light", "normal"):
            assert len({r["review"]["scene_id"] for r in selected if r["review"]["lighting"] == lighting}) >= 2
        assert len({r["review"]["scene_id"] for r in selected if r["review"]["sample_role"] == "background_negative"}) >= 2
    with pytest.raises(ValueError, match="context coverage"):
        assign_splits([r for r in rows if r["review"]["sample_role"] == "positive"], "reviewed-context-v4")


def test_background_evaluation_rejects_empty_cohort_and_flags_trivial_detector(tmp_path):
    ip, rp, _, review = release_inputs(tmp_path)
    root = tmp_path / "positive-only"
    build_release(ip, rp, root)
    class EmptyDetector:
        model = "test-null"
        model_sha256 = "fixture"
        runtime = "fixture"
        imgsz = 640
        def predict(self, image): return []
    with pytest.raises(ValueError, match="No reviewed negative"):
        evaluate_background(root, EmptyDetector())
    add_background_variant(review); rp.write_text(json.dumps(review))
    controls = tmp_path / "with-controls"
    build_release(ip, rp, controls)
    result = evaluate_background(controls, EmptyDetector())
    assert result["evaluation_completed"] and not result["positive_output_sanity_passed"]
    assert result["positive_images"] == 4
    with pytest.raises(ValueError, match="prediction floor"):
        evaluate_background(controls, EmptyDetector(), confidence=.1)
