FROM python:3.12-slim

WORKDIR /app
# Build context is model-server/, not gateway/, so models.yaml is reachable.
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/app ./app
# Catalogue ships inside the image, so the gateway runs standalone.
COPY models.yaml ./app/models.yaml

EXPOSE 8000
# One worker: pure async I/O, no CPU work. Scale with replicas, not workers,
# so upstream connection pools stay predictable.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
