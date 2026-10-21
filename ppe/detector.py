from pathlib import Path
import hashlib
from typing import Protocol

from PIL import Image

from ppe.schemas import Detection


class Detector(Protocol):
    model: str
    runtime: str

    def predict(self, image: Image.Image) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640):
        path = Path(weights)
        if not path.is_file() or path.suffix not in {".pt", ".onnx"}:
            raise ValueError("Provide a local trained .pt or .onnx file")
        from ultralytics import YOLO
        self._yolo = YOLO(str(path), task="detect")
        self.model = path.name
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.runtime = "onnxruntime" if path.suffix == ".onnx" else "pytorch"
        self.device, self.imgsz = device, imgsz
        names = {str(n).lower().replace("-", "_") for n in self._yolo.names.values()}
        if not {"person", "helmet", "vest", "no_helmet"}.issubset(names):
            raise ValueError("Model must contain person, helmet, vest and no_helmet labels; COCO weights are insufficient")
        self.predict(Image.new("RGB", (imgsz, imgsz)))

    def predict(self, image: Image.Image) -> list[Detection]:
        result = self._yolo.predict(image, device=self.device, imgsz=self.imgsz,
                                    conf=0.25, iou=0.7, rect=False, verbose=False)[0]
        return [Detection(label=result.names[int(box.cls.item())].lower().replace("-", "_"),
                          confidence=float(box.conf.item()),
                          box=tuple(box.xyxy[0].tolist())) for box in result.boxes]
