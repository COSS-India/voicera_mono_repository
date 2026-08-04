# Same CUDA/torch base as the AI4Bharat TTS server so MioCodec's torch ops match
# the driver stack already validated in this cluster.
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app
COPY requirements.txt .
# torch ships in the base image; keep it out of the pip resolve so the wheel
# isn't downgraded/rebuilt.
RUN grep -vE '^\s*torch' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir -r /tmp/req.txt

COPY . .
EXPOSE 8003
CMD ["python3", "server.py", "--port", "8003"]
