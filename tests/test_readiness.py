import copy

from scripts.data_readiness import assess_readiness


def test_derived_crops_do_not_inflate_readiness_and_integrity_is_mandatory():
    files = []
    for split in ("train","val"):
        files.append({"id":split,"source_id":split,"scene_id":split,"split":split,"sample_role":"positive","lighting":"normal"})
        for i in range(20):
            files.append({"id":f"{split}-crop-{i}","source_id":split,"scene_id":split,"split":split,"sample_role":"background_negative","lighting":"normal"})
    verification = {"passed":True,"class_instances":{s:{"helmet":10} for s in ("train","val")}}
    policy = {"policy_id":"fixture","scope":"test only","minimum_positive_scene_groups":{s:2 for s in ("train","val")},
              "minimum_negative_scene_groups":{s:2 for s in ("train","val")},
              "minimum_class_instances":{s:{"helmet":1} for s in ("train","val")},"minimum_lighting_scene_groups":{"normal":1}}
    result = assess_readiness({"files":files},verification,policy)
    assert result["actual"]["val"]["samples"] == 21
    assert result["actual"]["val"]["negative_scene_groups"] == 1
    assert result["actual"]["val"]["unique_sources"] == 1
    assert not result["substantive_training_ready"]
    relaxed = copy.deepcopy(policy)
    relaxed["minimum_positive_scene_groups"]={s:1 for s in ("train","val")}
    relaxed["minimum_negative_scene_groups"]={s:1 for s in ("train","val")}
    assert assess_readiness({"files":files},verification,relaxed)["substantive_training_ready"]
    verification["passed"] = False
    assert not assess_readiness({"files":files},verification,relaxed)["substantive_training_ready"]
