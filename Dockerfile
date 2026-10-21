FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# CPU deployment. CUDA/TensorRT require a separate image and real GPU validation.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY ppe ./ppe
RUN pip install --no-cache-dir '.[vision]'
RUN useradd --create-home ppe && mkdir -p /app/runs && chown ppe:ppe /app/runs
USER ppe
EXPOSE 8000
CMD ["uvicorn", "ppe.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-max-size", "8388608", "--ws-max-queue", "1"]
