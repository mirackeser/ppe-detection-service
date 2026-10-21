"""Relay local video/camera/RTSP frames to the binary-frame WebSocket API."""
import argparse
import asyncio
import json
from pathlib import Path
import time


async def run(args):
    import cv2
    from websockets.asyncio.client import connect
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot open video source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        async with connect(args.url, max_size=8 * 1024 * 1024, max_queue=1) as socket:
            with args.output.open("w", encoding="utf-8") as records:
                async def receive():
                    async for message in socket:
                        result = json.loads(message)
                        records.write(json.dumps(result) + "\n")
                        records.flush()
                reader = asyncio.create_task(receive())
                try:
                    while not reader.done() and (args.max_frames == 0 or count < args.max_frames):
                        start = time.perf_counter()
                        ok, frame = await asyncio.to_thread(capture.read)
                        if not ok:
                            break
                        ok, encoded = cv2.imencode(".jpg", frame)
                        if not ok:
                            raise ValueError("Cannot encode source frame")
                        await socket.send(encoded.tobytes())
                        count += 1
                        await asyncio.sleep(max(0, 1 / args.fps - (time.perf_counter() - start)))
                    # Keep the connection open briefly so the final pending response can arrive.
                    try:
                        await asyncio.wait_for(asyncio.shield(reader), timeout=args.drain_seconds)
                    except asyncio.TimeoutError:
                        pass
                finally:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
    finally:
        capture.release()
    print(f"Sent {count} frames; received responses saved to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Local video, camera index, or operator-supplied RTSP URL")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stream")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--drain-seconds", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/stream/responses.jsonl"))
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 0 or args.drain_seconds <= 0:
        parser.error("fps/drain-seconds must be positive; max-frames >=0")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
