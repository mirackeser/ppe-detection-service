import json

import pytest

from ppe.video import process_video
from ppe.schemas import Detection


class Detector:
    model = "test-only"
    def predict(self, image):
        return [Detection(label="person", confidence=.9, box=(1, 1, 20, 30))]


def test_offline_video_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    result = process_video(source, tmp_path / "out.mp4", Detector(), stride=2)
    assert result["processed_frames"] == 3
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [r["frame_index"] for r in records] == [0, 2, 4]
    assert [r["source_time_ms"] for r in records] == [0, 200, 400]
    capture = cv2.VideoCapture(result["output"])
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
