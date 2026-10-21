import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal
from uuid import uuid4
import warnings

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from ppe.detector import Detector, YoloDetector
from ppe.events import EventStore
from ppe.metrics import Metrics
from ppe.policy import assess
from ppe.schemas import DetectionResponse
from ppe.video import process_video
from ppe.temporal import TemporalConfig, TemporalTracker

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_VIDEO_BYTES = 64 * 1024 * 1024


def decode_image(data: bytes) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(BytesIO(data)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Image exceeds 16 megapixels")
                if source.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG are supported")
                return ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Invalid or oversized image") from exc


def create_app(detector: Detector | None = None, artifact_dir: Path = Path("runs/api"),
               temporal_config: TemporalConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.detector = detector
        app.state.busy = False
        app.state.metrics = Metrics()
        app.state.events = await asyncio.to_thread(EventStore, artifact_dir / "events.sqlite3")
        app.state.temporal_config = temporal_config or TemporalConfig(
            confirm_observations=int(os.getenv("PPE_CONFIRM_OBSERVATIONS", "5")),
            min_duration_ms=float(os.getenv("PPE_MIN_DURATION_MS", "500")),
            work_zone=tuple(json.loads(os.environ["PPE_WORK_ZONE"])) if os.getenv("PPE_WORK_ZONE") else None)
        weights = os.getenv("PPE_MODEL_PATH")
        if detector is None and weights:
            app.state.detector = await asyncio.to_thread(
                YoloDetector, weights, os.getenv("PPE_DEVICE", "cpu"))
        yield

    app = FastAPI(title="PPE Detection Service", version="0.2.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/ui/{asset}", include_in_schema=False)
    async def ui_asset(asset: Literal["style.css", "app.js"]):
        return FileResponse(Path(__file__).parent / "static" / asset)

    @app.get("/healthz")
    def health():
        ready = app.state.detector is not None
        return JSONResponse({"status": "ready" if ready else "model_not_configured"},
                            status_code=200 if ready else 503)

    def work(data):
        start = perf_counter()
        image = decode_image(data)
        engine = app.state.detector
        detections = engine.predict(image)
        return DetectionResponse(detections=detections, assessments=assess(detections),
                                 pipeline_ms=(perf_counter() - start) * 1000,
                                 model=engine.model, runtime=engine.runtime, image_size=image.size,
                                 model_sha256=getattr(engine, "model_sha256", None))

    def record(task):
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            app.state.metrics.observe(task.result().pipeline_ms / 1000)
        elif isinstance(error, ValueError):
            app.state.metrics.invalid += 1
        else:
            app.state.metrics.failed += 1

    @app.post("/detect/image", response_model=DetectionResponse)
    async def detect(file: UploadFile = File(...)):
        engine = app.state.detector
        if engine is None:
            raise HTTPException(503, "Set PPE_MODEL_PATH to trained PPE weights")
        # Bounded admission: no unbounded queue around a non-thread-safe model.
        if app.state.busy:
            app.state.metrics.rejected += 1
            raise HTTPException(429, "Inference worker busy; retry later")
        app.state.busy = True
        task = None
        try:
            data = await file.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                app.state.metrics.invalid += 1
                raise HTTPException(413, "Maximum upload size is 8 MiB")
            task = asyncio.create_task(asyncio.to_thread(work, data))
            # Keep worker ownership until thread finishes, even on client cancellation.
            def release(_):
                app.state.busy = False
                record(_)
            task.add_done_callback(release)
            return await asyncio.shield(task)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if task is None:
                app.state.busy = False
            await file.close()

    @app.get("/metrics")
    async def metrics():
        return Response(app.state.metrics.render(app.state.busy),
                        media_type="text/plain; version=0.0.4")

    @app.get("/events")
    async def events(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                     source_id: str | None = Query(None, max_length=128)):
        return await asyncio.to_thread(app.state.events.list, limit, offset, source_id)

    @app.post("/detect/video")
    async def detect_video(file: UploadFile = File(...), max_frames: int = Query(600, ge=1, le=3000),
                           stride: int = Query(1, ge=1, le=30)):
        if app.state.detector is None:
            raise HTTPException(503, "Model not configured")
        if app.state.busy:
            app.state.metrics.rejected += 1
            raise HTTPException(429, "Inference worker busy; retry later")
        app.state.busy = True
        task = None
        artifact_id = uuid4().hex
        def process(data):
            output = artifact_dir / f"{artifact_id}.mp4"
            try:
                with TemporaryDirectory(prefix="ppe-video-") as directory:
                    source = Path(directory) / "upload.mp4"
                    source.write_bytes(data)
                    result = process_video(source, output, app.state.detector, stride, max_frames,
                                           source_id=artifact_id, temporal_config=app.state.temporal_config,
                                           event_sink=app.state.events.add,
                                           source_label=Path(file.filename or "Video").name[:128])
                return {"artifact_id": artifact_id, "processed_frames": result["processed_frames"],
                        "decoded_frames": result["decoded_frames"],
                        "frame_limit_reached": result["frame_limit_reached"],
                        "source_id": artifact_id, "event_count": result["event_count"],
                        "audio_preserved": False,
                        "video_url": f"/artifacts/{artifact_id}/video",
                        "records_url": f"/artifacts/{artifact_id}/records"}
            except Exception:
                # These paths are owned exclusively by this request.
                output.unlink(missing_ok=True)
                output.with_suffix(".jsonl").unlink(missing_ok=True)
                raise
        try:
            data = await file.read(MAX_VIDEO_BYTES + 1)
            if len(data) > MAX_VIDEO_BYTES:
                raise HTTPException(413, "Maximum video upload size is 64 MiB")
            task = asyncio.create_task(asyncio.to_thread(process, data))
            def release(future):
                app.state.busy = False
                if not future.cancelled():
                    error = future.exception()
                    if error is None:
                        app.state.metrics.events += future.result()["event_count"]
            task.add_done_callback(release)
            return await asyncio.shield(task)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            if task is None:
                app.state.busy = False
            await file.close()

    @app.get("/artifacts/{artifact_id}/{kind}")
    async def artifact(artifact_id: str, kind: Literal["video", "records"]):
        if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
            raise HTTPException(404, "Artifact not found")
        path = artifact_dir / f"{artifact_id}{'.mp4' if kind == 'video' else '.jsonl'}"
        if not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, media_type="video/mp4" if kind == "video" else "application/x-ndjson")

    @app.websocket("/stream")
    async def stream(socket: WebSocket, source_label: str = Query("Akış", max_length=128)):
        await socket.accept()
        if app.state.detector is None or app.state.busy:
            app.state.metrics.rejected += 1
            await socket.close(code=1013, reason="Model unavailable or worker busy")
            return
        app.state.busy = True
        queue = asyncio.Queue(maxsize=1)
        source_id = uuid4().hex
        tracker = TemporalTracker(source_id, app.state.temporal_config, source_kind="stream", source_label=source_label)

        async def receive():
            frame_id = 0
            try:
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    data = message.get("bytes")
                    if data is None or len(data) > MAX_BYTES:
                        app.state.metrics.invalid += 1
                        await socket.close(code=1003 if data is None else 1009,
                                           reason="Send a JPEG/PNG binary frame of at most 8 MiB")
                        break
                    if queue.full():
                        queue.get_nowait()
                        app.state.metrics.dropped += 1
                    queue.put_nowait((frame_id, data, perf_counter()*1000))
                    frame_id += 1
            finally:
                if queue.full():
                    queue.get_nowait()
                    app.state.metrics.dropped += 1
                queue.put_nowait(None)

        receiver = asyncio.create_task(receive())
        inference = None
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                frame_id, data, received_ms = item
                inference = asyncio.create_task(asyncio.to_thread(work, data))
                inference.add_done_callback(record)
                try:
                    response = await asyncio.shield(inference)
                except ValueError as error:
                    if receiver.done():
                        break
                    await socket.send_json({"frame_id": frame_id, "error": str(error)})
                    continue
                if receiver.done():
                    break
                tracks, events = tracker.update(response.detections, response.assessments,
                                                received_ms, response.image_size, response.model, response.model_sha256)
                for event in events:
                    await asyncio.to_thread(app.state.events.add, event)
                    app.state.metrics.events += 1
                await socket.send_json({"frame_id": frame_id, "source_id": source_id,
                                        "tracks": tracks, "events": [e.model_dump() for e in events],
                                        **response.model_dump(mode="json")})
        except WebSocketDisconnect:
            pass
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            # A disconnected client cannot make the model available while its thread runs.
            if inference is not None and not inference.done():
                inference.add_done_callback(lambda _: setattr(app.state, "busy", False))
            else:
                app.state.busy = False

    return app


app = create_app()
