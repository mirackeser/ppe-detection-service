"""Exercise the running localhost server with real model HTTP/video/WebSocket calls."""
import argparse
import asyncio
import json
from pathlib import Path


async def stream(image: bytes):
    from websockets.asyncio.client import connect
    responses = []
    async with connect("ws://127.0.0.1:8000/stream") as socket:
        for _ in range(5):
            await socket.send(image)
            response = json.loads(await asyncio.wait_for(socket.recv(), 30))
            if "detections" not in response:
                raise ValueError(f"Unexpected WebSocket response: {response}")
            responses.append(response)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    import httpx
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=120) as client:
        health = client.get("/healthz")
        health.raise_for_status()
        image = client.post("/detect/image", files={"file": (args.image.name, args.image.read_bytes())})
        image.raise_for_status()
        video = client.post("/detect/video?max_frames=10", files={"file": (args.video.name, args.video.read_bytes())})
        video.raise_for_status()
        for key in ["video_url", "records_url"]:
            client.get(video.json()[key]).raise_for_status()
        responses = asyncio.run(stream(args.image.read_bytes()))
        metrics = client.get("/metrics")
        metrics.raise_for_status()
    report = {"health_status": health.status_code, "model": image.json()["model"],
              "runtime": image.json()["runtime"], "image_detections": len(image.json()["detections"]),
              "video_processed_frames": video.json()["processed_frames"],
              "artifact_downloads": "passed", "websocket_responses": len(responses),
              "metrics_status": metrics.status_code,
              "scope": "Localhost transport integration with selected input files, not a physical camera/RTSP test",
              "video_source": str(args.video), "image_source": str(args.image)}
    output = Path("reports/service-integration.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
