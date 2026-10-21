"""Conservative IoU association and temporal evidence; no re-identification claim."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from uuid import uuid4

from ppe.schemas import Assessment, Detection, ViolationEvent


@dataclass(frozen=True)
class TemporalConfig:
    confirm_observations: int = 5
    min_duration_ms: float = 500
    max_gap_ms: float = 1000
    clear_observations: int = 3
    match_iou: float = .3
    max_tracks: int = 128
    work_zone: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.confirm_observations < 2 or self.clear_observations < 1 or self.max_tracks < 1:
            raise ValueError("Invalid temporal observation/track limits")
        if not all(math.isfinite(v) for v in (self.min_duration_ms, self.max_gap_ms, self.match_iou)):
            raise ValueError("Temporal parameters must be finite")
        if self.min_duration_ms < 0 or self.max_gap_ms <= 0 or not 0 < self.match_iou <= 1:
            raise ValueError("Invalid temporal time/IoU limits")
        if self.work_zone is not None:
            x1, y1, x2, y2 = self.work_zone
            if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError("work_zone must be a normalized positive-area xyxy rectangle")


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    last_ms: float
    missing_streak: int = 0
    first_missing_ms: float = 0
    minimum_confidence: float = 1
    emitted: bool = False
    clear_streak: int = 0

    def reset_candidate(self):
        self.missing_streak = 0
        self.minimum_confidence = 1
        self.clear_streak = 0


class TemporalTracker:
    def __init__(self, source_id: str, config: TemporalConfig | None = None,
                 source_kind="unspecified", source_label=None):
        self.source_id = source_id
        self.config = config or TemporalConfig()
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.last_update_ms = -1.0
        self.source_kind, self.source_label = source_kind, source_label

    def update(self, detections: list[Detection], assessments: list[Assessment], now_ms: float,
               image_size: tuple[int, int], model: str, model_sha256: str | None = None):
        if not math.isfinite(now_ms) or now_ms < self.last_update_ms:
            raise ValueError("Frame timestamps must be finite and non-decreasing")
        self.last_update_ms = now_ms
        cfg = self.config
        self.tracks = {key: track for key, track in self.tracks.items() if now_ms-track.last_ms <= cfg.max_gap_ms}
        people = {a.person_index: detections[a.person_index] for a in assessments}
        candidates = {i: [key for key, track in self.tracks.items() if overlap(person.box, track.box) >= cfg.match_iou]
                      for i, person in people.items()}
        track_candidates = {key: [i for i, keys in candidates.items() if key in keys] for key in self.tracks}
        assignments = {i: keys[0] for i, keys in candidates.items()
                       if len(keys) == 1 and len(track_candidates[keys[0]]) == 1}
        # A lost or ambiguous observation cannot extend a missing-helmet streak.
        for key, track in self.tracks.items():
            if key not in assignments.values():
                track.reset_candidate()
        for i, person in people.items():
            # Leave ambiguous matches untracked instead of transferring evidence between people.
            if i not in assignments and not candidates[i] and len(self.tracks) < cfg.max_tracks:
                key = self.next_id
                self.next_id += 1
                self.tracks[key] = Track(key, person.box, now_ms)
                assignments[i] = key
        events, observations = [], []
        for assessment in assessments:
            i = assessment.person_index
            if i not in assignments:
                continue
            track = self.tracks[assignments[i]]
            track.box, track.last_ms = people[i].box, now_ms
            in_zone = True
            if cfg.work_zone is not None:
                width, height = image_size
                x = (track.box[0]+track.box[2])/2/width
                y = track.box[3]/height
                x1, y1, x2, y2 = cfg.work_zone
                in_zone = x1 <= x <= x2 and y1 <= y <= y2
            if not in_zone:
                track.reset_candidate()
                track.emitted = False
            elif assessment.helmet == "missing" and assessment.helmet_confidence is not None:
                if track.missing_streak == 0:
                    track.first_missing_ms = now_ms
                track.missing_streak += 1
                track.clear_streak = 0
                track.minimum_confidence = min(track.minimum_confidence, assessment.helmet_confidence)
                if (not track.emitted and track.missing_streak >= cfg.confirm_observations
                        and now_ms-track.first_missing_ms >= cfg.min_duration_ms):
                    events.append(ViolationEvent(event_id=uuid4().hex, source_id=self.source_id,
                                  track_id=track.track_id, observed_at_utc=datetime.now(timezone.utc).isoformat(),
                                  first_evidence_ms=track.first_missing_ms, confirmed_at_ms=now_ms,
                                  consecutive_observations=track.missing_streak,
                                  min_evidence_confidence=track.minimum_confidence, person_box=track.box, model=model,
                                  model_sha256=model_sha256, policy=asdict(cfg),
                                  source_kind=self.source_kind, source_label=self.source_label))
                    track.emitted = True
            elif assessment.helmet == "present":
                track.missing_streak = 0
                track.minimum_confidence = 1
                track.clear_streak += 1
                if track.clear_streak >= cfg.clear_observations:
                    track.emitted = False
            else:
                track.reset_candidate()
            observations.append({"track_id": track.track_id, "person_index": i,
                                 "helmet": assessment.helmet, "in_work_zone": in_zone,
                                 "missing_streak": track.missing_streak, "event_issued": track.emitted})
        return observations, events
