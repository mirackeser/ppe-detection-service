from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    box: tuple[float, float, float, float]  # original-image xyxy pixels

    @model_validator(mode="after")
    def valid_box(self):
        import math
        x1, y1, x2, y2 = self.box
        if not all(math.isfinite(v) for v in self.box) or not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("box must be finite, positive-area xyxy coordinates")
        return self


class Assessment(BaseModel):
    person_index: int
    helmet: Literal["present", "missing", "unknown"]
    vest: Literal["present", "unknown"]
    helmet_confidence: float | None = Field(default=None, ge=0, le=1)


class DetectionResponse(BaseModel):
    detections: list[Detection]
    assessments: list[Assessment]
    pipeline_ms: float
    model: str
    runtime: str
    image_size: tuple[int, int]
    model_sha256: str | None = None


class ViolationEvent(BaseModel):
    event_id: str
    source_id: str
    track_id: int
    kind: Literal["missing_helmet"] = "missing_helmet"
    observed_at_utc: str
    first_evidence_ms: float
    confirmed_at_ms: float
    consecutive_observations: int
    min_evidence_confidence: float = Field(ge=0, le=1)
    person_box: tuple[float, float, float, float]
    model: str
    model_sha256: str | None = None
    policy: dict = Field(default_factory=dict)
    confirmation: Literal["temporal_model_evidence"] = "temporal_model_evidence"
    source_kind: Literal["stream", "video", "unspecified"] = "unspecified"
    source_label: str | None = None
