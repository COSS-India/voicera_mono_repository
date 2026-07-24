FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app
COPY requirements.txt .
RUN grep -vE '^(torch|flashinfer)' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir -r /tmp/req.txt && \
    PIP_CONSTRAINT= pip3 install --no-cache-dir flashinfer-python==0.6.7 flashinfer-cubin==0.6.7

COPY . .
EXPOSE 8002
CMD ["python3", "server.py", "--port", "8002"]
