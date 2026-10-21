"""Single-frame evidence only; no safety certification or temporal event claims."""
from ppe.schemas import Assessment, Detection


def assess(detections: list[Detection], threshold: float = 0.5,
           missing_threshold: float = 0.25) -> list[Assessment]:
    people = [(i, d) for i, d in enumerate(detections)
              if d.label.lower() == "person" and d.confidence >= threshold]
    assigned = {i: {} for i, _ in people}
    for gear in detections:
        label = gear.label.lower().replace("-", "_")
        cutoff = missing_threshold if label == "no_helmet" else threshold
        if label not in {"helmet", "no_helmet", "vest"} or gear.confidence < cutoff:
            continue
        cx = (gear.box[0] + gear.box[2]) / 2
        cy = (gear.box[1] + gear.box[3]) / 2
        candidates = []
        for i, person in people:
            x1, y1, x2, y2 = person.box
            relative_y = (cy - y1) / (y2 - y1)
            in_region = (0 <= relative_y <= 0.4) if label != "vest" else (0.2 <= relative_y <= 0.8)
            if x1 <= cx <= x2 and in_region:
                candidates.append(((x2 - x1) * (y2 - y1), i))
        # Ambiguous overlapping people are left unknown rather than sharing gear.
        if len(candidates) == 1:
            labels = assigned[candidates[0][1]]
            labels[label] = max(labels.get(label, 0), gear.confidence)
    result = []
    for i, _ in people:
        labels = assigned[i]
        helmet = "unknown"
        if "helmet" in labels and "no_helmet" not in labels:
            helmet = "present"
        elif "no_helmet" in labels and "helmet" not in labels:
            helmet = "missing"
        result.append(Assessment(person_index=i, helmet=helmet,
                                 helmet_confidence=labels.get("helmet" if helmet == "present" else "no_helmet") if helmet != "unknown" else None,
                                 vest="present" if "vest" in labels else "unknown"))
    return result
