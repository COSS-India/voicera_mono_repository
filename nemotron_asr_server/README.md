# AI4Bharat Nemotron Streaming ASR Server

## Prerequisites

- NVIDIA GPU with a driver supporting CUDA 13.0 or later
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Docker with the Compose plugin (`docker compose`)
- ~5 GB free disk for model checkpoints, plus space for the Docker image
- A Hugging Face account with access granted to both gated checkpoints

## Installation

1. Clone this repository and `cd` into it.

2. Both checkpoints are gated on Hugging Face — request access on each model
   page, then log in:

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

   Download them into `models/`:

   ```bash
   huggingface-cli download ai4bharat/indic-asr-nemotron-600m \
       --local-dir models/indic-asr-nemotron-600m

   huggingface-cli download ai4bharat/bhili-asr-nemotron-600m \
       --local-dir models/bhili-asr-nemotron-600m
   ```

   The server expects these exact files to exist after download:

   ```
   models/indic-asr-nemotron-600m/indic_nemotron_v1_1_sft_600k_lr1-averaged-40k.nemo
   models/bhili-asr-nemotron-600m/indic_nemotron_bhili_sft_lr1-averaged.nemo
   ```

3. Build and start the server:

   ```bash
   docker compose up -d --build
   ```

4. Verify it's healthy:

   ```bash
   curl http://localhost:8000/health
   ```

   A `"status": "healthy"` response with both models listed under `models_loaded`
   means the server is ready.

5. Open `http://localhost:8000` in a browser for the demo page, or point an
   OpenAI-compatible client at `http://localhost:8000/v1`.
