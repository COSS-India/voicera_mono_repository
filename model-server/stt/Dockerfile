# AI4Bharat Indic Conformer (NeMo). Mirrors the install sequence setup.sh has
# been running on the host, but at image-build time so a deploy stops mutating
# source on the running machine.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG NEMO_REF=nemo-v2
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PORT=8001

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common git wget ca-certificates libsndfile1 ffmpeg \
 && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
 && wget -q https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# The two exp_manager patches setup.sh applies with sed, applied here instead.
# /opt/nemo is inside the image and never bind-mounted, so they always hold.
RUN git clone --branch ${NEMO_REF} --depth 1 https://github.com/AI4Bharat/NeMo.git /opt/nemo \
 && sed -i 's/from pytorch_lightning.loggers import MLFlowLogger, NeptuneLogger, TensorBoardLogger, WandbLogger/from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger/' /opt/nemo/nemo/utils/exp_manager.py \
 && sed -i 's/^    if create_neptune_logger:/    if False and create_neptune_logger:/' /opt/nemo/nemo/utils/exp_manager.py \
 && pip install --no-cache-dir -e "/opt/nemo[asr]"

RUN pip install --no-cache-dir \
      fastapi uvicorn pydantic torchaudio python-dotenv numpy python-multipart \
      "onnxruntime-gpu>=1.24" "ml_dtypes>=0.5" --upgrade protobuf ml_dtypes

WORKDIR /app
COPY . /app

EXPOSE 8001
CMD ["python3", "server.py"]
