"""Local video processing with per-frame evidence and conservative temporal events."""
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from PIL import Image

from ppe.detector import Detector
from ppe.policy import assess
from ppe.temporal import TemporalConfig, TemporalTracker


def process_video(source: Path, output: Path, detector: Detector, stride: int = 1,
                  max_frames: int | None = None, *, source_id: str | None = None,
                  temporal_config: TemporalConfig | None = None, event_sink=None, source_label=None):
    import cv2
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("stride and max_frames must be positive")
    if not source.is_file():
        raise ValueError("Video source must be a local file")
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from source")
    capture = cv2.VideoCapture(str(source))
    writer = None
    sidecar = output.with_suffix(".jsonl")
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot open video")
    output.parent.mkdir(parents=True, exist_ok=True)
    decoded, processed, event_count = 0, 0, 0
    source_id = source_id or uuid4().hex
    tracker = TemporalTracker(source_id, temporal_config, source_kind="video", source_label=source_label or source.name)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps <= 1000:
            raise ValueError("Video has invalid frame rate")
        with sidecar.open("w", encoding="utf-8") as records:
            while max_frames is None or decoded < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[0] * frame.shape[1] > 16_000_000:
                    raise ValueError("Video frame exceeds 16 megapixels")
                index = decoded
                decoded += 1
                if index % stride:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / stride, (width, height))
                    if not writer.isOpened():
                        raise ValueError("Cannot create MP4 output")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                start = perf_counter()
                detections = detector.predict(image)
                elapsed = (perf_counter() - start) * 1000
                assessments = assess(detections)
                tracks, events = tracker.update(detections, assessments, index/fps*1000,
                                                image.size, detector.model, getattr(detector, "model_sha256", None))
                for event in events:
                    if event_sink is not None:
                        event_sink(event)
                event_count += len(events)
                records.write(json.dumps({"frame_index": index, "source_time_ms": index / fps * 1000,
                                          "predict_ms": elapsed,
                                          "detections": [d.model_dump() for d in detections],
                                          "assessments": [a.model_dump() for a in assessments],
                                          "source_id": source_id, "tracks": tracks,
                                          "events": [event.model_dump() for event in events]}) + "\n")
                for detection in detections:
                    x1, y1, x2, y2 = (int(v) for v in detection.box)
                    color = (0, 165, 255) if detection.label == "no_helmet" else (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}",
                                (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                if tracker.config.work_zone is not None:
                    x1, y1, x2, y2 = tracker.config.work_zone
                    width, height = image.size
                    cv2.rectangle(frame, (int(x1*width), int(y1*height)),
                                  (int(x2*width), int(y2*height)), (255, 200, 0), 2)
                for track in tracks:
                    person = detections[track["person_index"]]
                    x1, y1, x2, y2 = (int(v) for v in person.box)
                    confirmed = track["event_issued"] and track["helmet"] == "missing" and track["in_work_zone"]
                    color = (0, 0, 255) if confirmed else (0, 165, 255) if track["helmet"] == "missing" else (180, 180, 180)
                    text = f"ID {track['track_id']} | {'TEMPORAL EVENT' if confirmed else track['helmet']} | n={track['missing_streak']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, text, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                writer.write(frame)
                processed += 1
        if processed == 0:
            raise ValueError("No decodable frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return {"decoded_frames": decoded, "processed_frames": processed,
            "source_id": source_id, "event_count": event_count,
            "frame_limit_reached": max_frames is not None and decoded >= max_frames,
            "output": str(output), "records": str(sidecar), "audio_preserved": False,
            "scope": "offline video with conservative IoU tracks and temporal missing-helmet candidates"}
