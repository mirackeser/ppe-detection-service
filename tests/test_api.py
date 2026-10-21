from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from ppe.api import create_app, MAX_BYTES
from ppe.schemas import Detection


class TestDetector:
    __test__ = False
    model = "test-only"
    runtime = "test-only"

    def predict(self, image):
        return [Detection(label="person", confidence=.9, box=(0, 0, 20, 30))]


def test_unconfigured_service_does_not_fake_predictions(monkeypatch):
    monkeypatch.delenv("PPE_MODEL_PATH", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/healthz").status_code == 503
        assert client.post("/detect/image", files={"file": ("a.jpg", b"x")}).status_code == 503
        assert client.get("/").status_code == 200
        assert client.get("/ui/app.js").status_code == 200
        assert client.get("/ui/style.css").status_code == 200
        assert client.get("/ui/anythingelse").status_code == 422


def test_image_and_bad_uploads():
    buf = BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    with TestClient(create_app(TestDetector())) as client:
        assert client.get("/healthz").status_code == 200
        response = client.post("/detect/image", files={"file": ("frame.png", buf.getvalue())})
        assert response.status_code == 200
        assert response.json()["assessments"][0]["helmet"] == "unknown"
        assert response.json()["pipeline_ms"] >= 0
        assert client.post("/detect/image", files={"file": ("x.png", b"invalid")}).status_code == 422
        assert client.post("/detect/image", files={"file": ("x.png", b"x" * (MAX_BYTES + 1))}).status_code == 413
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "ppe_inference_completed_total 1" in metrics.text
        assert "ppe_invalid_total 2" in metrics.text


def test_busy_worker_rejects_request():
    app = create_app(TestDetector())
    with TestClient(app) as client:
        app.state.busy = True
        assert client.post("/detect/image", files={"file": ("x.png", b"x")}).status_code == 429


def test_stream_processes_binary_frame_and_rejects_parallel_http():
    buf = BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    with TestClient(create_app(TestDetector())) as client:
        with client.websocket_connect("/stream") as socket:
            socket.send_bytes(buf.getvalue())
            result = socket.receive_json()
            assert result["assessments"][0]["helmet"] == "unknown"
            assert client.post("/detect/image", files={"file": ("x.png", b"x")}).status_code == 429


def test_stream_recovers_from_invalid_image():
    with TestClient(create_app(TestDetector())) as client:
        with client.websocket_connect("/stream") as socket:
            socket.send_bytes(b"invalid")
            assert "error" in socket.receive_json()
            buf = BytesIO()
            Image.new("RGB", (32, 32)).save(buf, format="PNG")
            socket.send_bytes(buf.getvalue())
            assert "detections" in socket.receive_json()


def test_stream_drops_stale_frame_under_backpressure():
    from threading import Event
    class BlockingDetector(TestDetector):
        started = Event()
        release = Event()
        calls = 0

        def predict(self, image):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                assert self.release.wait(5), "Test did not release inference"
            return super().predict(image)

    engine = BlockingDetector()
    buf = BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/stream") as socket:
            socket.send_bytes(buf.getvalue())
            assert engine.started.wait(5)
            try:
                socket.send_bytes(buf.getvalue())
                socket.send_bytes(buf.getvalue())
                assert "ppe_dropped_frames_total 1" in client.get("/metrics").text
            finally:
                engine.release.set()
            assert socket.receive_json()["frame_id"] == 0
            assert socket.receive_json()["frame_id"] == 2


def test_video_upload_and_artifacts(tmp_path):
    import pytest
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    with TestClient(create_app(TestDetector(), tmp_path / "artifacts")) as client:
        response = client.post("/detect/video?max_frames=2", files={"file": ("x.mp4", source.read_bytes())})
        assert response.status_code == 200
        result = response.json()
        assert result["processed_frames"] == 2
        assert result["frame_limit_reached"]
        assert client.get(result["video_url"]).status_code == 200
        assert len(client.get(result["records_url"]).text.splitlines()) == 2
        assert client.get("/artifacts/invalid/video").status_code == 404
        assert client.post("/detect/video", files={"file": ("x.mp4", b"invalid")}).status_code == 422


def test_stream_emits_and_persists_confirmed_event(tmp_path):
    from ppe.temporal import TemporalConfig
    class MissingDetector(TestDetector):
        def predict(self, image):
            return [Detection(label="person", confidence=.9, box=(0, 0, 20, 30)),
                    Detection(label="no_helmet", confidence=.3, box=(2, 1, 18, 8))]
    buf = BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    config = TemporalConfig(confirm_observations=2, min_duration_ms=0)
    app = create_app(MissingDetector(), tmp_path, config)
    with TestClient(app) as client:
        # A still-image candidate cannot generate a temporal event.
        client.post("/detect/image", files={"file": ("x.png", buf.getvalue())})
        assert client.get("/events").json() == []
        with client.websocket_connect("/stream") as socket:
            socket.send_bytes(buf.getvalue())
            assert socket.receive_json()["events"] == []
            socket.send_bytes(buf.getvalue())
            result = socket.receive_json()
            assert len(result["events"]) == 1
            socket.send_bytes(buf.getvalue())
            assert socket.receive_json()["events"] == []
        events = client.get("/events", params={"source_id": result["source_id"]}).json()
        assert len(events) == 1
        assert events[0]["min_evidence_confidence"] == .3
        assert client.get("/events?limit=1001").status_code == 422
