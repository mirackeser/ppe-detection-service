"""Transport/state-machine proof with a repeated public validation photo, not event accuracy."""
import asyncio
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from ppe.policy import assess
from ppe.schemas import Detection


def prepare():
    import cv2
    import numpy as np
    index = json.loads(Path("runs/error-analysis/index.json").read_text())
    candidates = []
    for row in index["ranked_images"]:
        detections = [Detection(**d) for d in row["predictions"]]
        assessments = assess(detections)
        for a in assessments:
            if a.helmet == "missing" and len(assessments) == 1:
                candidates.append((a.helmet_confidence, row["image"]))
    if not candidates:
        raise ValueError("No single-person validation candidate available for this transport check")
    _, source = max(candidates)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((640, 640))
        width, height = image.size
        image = image.crop((0, 0, width-width%2, height-height%2))
        image.save("runs/temporal-demo.jpg", quality=95)
    path = Path("runs/DEMO_tekrarlanan_fotograf.mp4")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, image.size)
    if not writer.isOpened():
        raise ValueError("Cannot create demo video")
    for _ in range(12):
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()
    return source, path


async def check_stream():
    from websockets.asyncio.client import connect
    payload = Path("runs/temporal-demo.jpg").read_bytes()
    results = []
    async with connect("ws://127.0.0.1:8000/stream?source_label=DEMO_tekrarlanan_fotograf") as socket:
        for _ in range(8):
            await socket.send(payload)
            results.append(json.loads(await asyncio.wait_for(socket.recv(), 30)))
            await asyncio.sleep(.11)
    events = [event for result in results for event in result["events"]]
    if not events or any(count != 1 for count in Counter(e["track_id"] for e in events).values()):
        raise ValueError("Expected one temporal event per stable track in the replay stream")
    return results[-1]["source_id"], len(events)


def main():
    import httpx
    source, path = prepare()
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=120) as client:
        response = client.post("/detect/video?max_frames=12", files={"file": (path.name, path.read_bytes())})
        response.raise_for_status()
        video = response.json()
        assert video["event_count"] >= 1, "Repeated video candidate did not reach temporal confirmation"
        video_events = client.get("/events", params={"source_id": video["source_id"]}).json()
        assert len(video_events) == video["event_count"]
        source_id, count = asyncio.run(check_stream())
        stream_events = client.get("/events", params={"source_id": source_id}).json()
        assert len(stream_events) == count
    report = {"input": source, "synthetic_repeated_still": True,
              "scope": "Real detector + local video/WebSocket + temporal state + SQLite persistence; not camera or event accuracy",
              "video_source_id": video["source_id"], "stream_source_id": source_id,
              "video_events": len(video_events), "stream_events": len(stream_events),
              "event_deduplication": "passed", "persistence": "passed"}
    Path("reports/temporal-integration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
