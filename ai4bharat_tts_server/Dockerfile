FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app
COPY requirements.txt .
RUN grep -vE '^(torch|flashinfer)' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir -r /tmp/req.txt && \
    PIP_CONSTRAINT= pip3 install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129 && \
    pip3 uninstall -y torchvision torchaudio flash_attn apex torch_tensorrt torchao && \
    PIP_CONSTRAINT= pip3 install --no-cache-dir flashinfer-python==0.6.7 flashinfer-cubin==0.6.7

COPY . .
EXPOSE 8002
CMD ["python3", "server.py", "--port", "8002"]
