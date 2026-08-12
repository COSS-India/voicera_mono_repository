# Model setup

The checkpoint is **not part of this repository** — it is 7.6 GB (7.1 GiB, which is
what `du -h` prints), and its license is AI4Bharat's, not ours. This page covers how
to get it, where to put it, and how to confirm it worked.

## What you need

The AI4Bharat Orpheus Indic checkpoint: a fine-tune of Orpheus 3B covering 22
scheduled Indian languages, 40 speakers, 12 speaking styles. Five files, and they
usually arrive in **two separate folders** — see
[assembling from the raw checkpoint](#assembling-from-the-raw-training-checkpoint):

| File | Size | Usually found in | Purpose |
|---|---|---|---|
| `model.safetensors` | 7.6 GB | `checkpoint-5679/` | The weights |
| `config.json` | 862 B | `checkpoint-5679/` | Architecture |
| `generation_config.json` | 179 B | `checkpoint-5679/` | Generation defaults |
| `tokenizer.json` | 22 MB | `llama-3-audio-tok_trimmed/` | Tokenizer, including the audio and speaker tokens |
| `tokenizer_config.json` | 326 B | `llama-3-audio-tok_trimmed/` | Tokenizer settings |

All five are required, and they all go in **one** directory — the split above is how
the training run saved them, not how the server wants them.

`tokenizer.json` matters more than its size suggests: it carries the `<|speaker>` /
`<|style>` marker tokens and the SNAC audio-code vocabulary that this server
addresses by id. It is **not** inside the checkpoint folder, and the server cannot
start without it.

### Expected checksums

These are the five files this server has been verified against. If your source did
not ship a `SHA256SUMS`, use this to confirm you assembled the right thing:

```
79cdcf947d131f1642aaebd47d6fef44b35e2f4101b9961721b7c0b384d53582  config.json
d8db84d9d12a7879f1e146aaed062a748d74767bc416f680bde52c794e8c972c  generation_config.json
b6fba8bf4aae47f02b2ca7e46b7ce06cd2841fe832de9424729794429af288c7  model.safetensors
7b997dcb61e08dd3c748a5a30b47bdb1ca10cfa2cbd07a73d1ef74098d0d4097  tokenizer.json
fababf790564f9437f8ffa3da373e4791e6f2bd4403b6c834b69f28026b8ea89  tokenizer_config.json
```

A different fine-tune will have different hashes and that is fine — this is a check
that your *copy* is complete and untruncated, not a requirement.

## Assembling from the raw training checkpoint

This is what a shared Drive folder normally contains: the HuggingFace-Trainer output
directory, plus the tokenizer beside it. Roughly 29 GB, of which you need 7.6 GB.

```
<the folder you were given>/
├── checkpoint-5679/
│   ├── config.json                 <- copy
│   ├── generation_config.json      <- copy
│   ├── model.safetensors    7.6 GB <- copy
│   ├── optimizer.bin       13.2 GB  ┐
│   ├── pytorch_model_fsdp.bin 7.6 GB│  training state — skip all of it,
│   ├── rng_state_*.pth  (32 files)  │  ~21 GB you do not need to serve
│   ├── scheduler.pt                 │
│   ├── trainer_state.json           │
│   └── training_args.bin           ┘
└── llama-3-audio-tok_trimmed/
    ├── tokenizer.json      22 MB   <- copy
    └── tokenizer_config.json       <- copy
```

Flatten the five into one directory:

```bash
DRIVE=/path/to/the/folder/you/were/given
mkdir -p models/orpheus-indic-5679
cp "$DRIVE"/checkpoint-5679/{config.json,generation_config.json,model.safetensors} \
   models/orpheus-indic-5679/
cp "$DRIVE"/llama-3-audio-tok_trimmed/{tokenizer.json,tokenizer_config.json} \
   models/orpheus-indic-5679/
```

Then check the five hashes against the table above:

```bash
cd models/orpheus-indic-5679 && sha256sum *
```

Two mistakes to avoid, because neither gives a helpful error:

- **Copying `checkpoint-5679/` as the model directory.** The tokenizer is not in it,
  so the load fails with a tokenizer error rather than anything about the checkpoint.
- **Copying the whole folder.** `pytorch_model_fsdp.bin` and `optimizer.bin` are the
  same weights in training format plus the optimizer state. They cost 21 GB of disk
  and do nothing here.

`config.json` from this checkpoint was written by **transformers v5** (nested
`rope_parameters`, a top-level `dtype` key). The pinned stack in `constraints.txt`
reads it as-is; an older transformers 4.x stack needs the config rewritten to the v4
schema. If you are using this repository's Docker image, that is already handled.

## Where else to get it

The checkpoint is distributed by whoever runs your deployment — this repository
deliberately hard-codes no download URL, since the weights are not ours to
redistribute. Use whichever of these applies to you:

**Google Drive.** Pull the folder down first, then assemble the five files out of it
with the steps in [the section above](#assembling-from-the-raw-training-checkpoint) —
a Drive folder is almost always the raw checkpoint, not a ready-to-serve directory.
From a headless machine, [`gdown`](https://github.com/wkentaro/gdown) handles the
confirmation interstitial that plain `curl` trips over:

```bash
pip install gdown
gdown --folder '<your-google-drive-folder-url>' -O /path/to/drive-download
```

That fetches everything, training state included (~29 GB). To avoid the 21 GB you do
not need, download the five files individually instead — in the Drive web UI, or by
file id:

```bash
gdown '<file-id-of-model.safetensors>' -O models/orpheus-indic-5679/model.safetensors
# …and the same for config.json, generation_config.json,
# tokenizer.json, tokenizer_config.json
```

**An archive someone made for you.** If you were handed a `.tar.gz` or `.zip` of the
assembled directory, it already has the right shape:

```bash
mkdir -p models
tar -xzf orpheus-indic-5679.tar.gz -C models/   # or: unzip orpheus-indic-5679.zip -d models/
```

**Object storage (S3, GCS, Azure).**

```bash
aws s3 sync s3://<your-bucket>/<prefix>/orpheus-indic-5679 models/orpheus-indic-5679
# or
gsutil -m rsync -r gs://<your-bucket>/orpheus-indic-5679 models/orpheus-indic-5679
```

**HuggingFace Hub**, if your copy is hosted there. Either download it:

```bash
pip install 'huggingface_hub[cli]'
hf download <org>/<repo> --local-dir models/orpheus-indic-5679
```

…or skip the download entirely and let the server fetch it on first boot by putting
the repo id in the config:

```bash
ORPHEUS_MODEL_PATH=<org>/<repo>
```

Anything that is not an existing local directory is treated as a HuggingFace repo id.
Note that with Docker this downloads into the mounted `hf-cache/`, so the first boot
needs network access and takes as long as the download.

**Copying from another machine.**

```bash
rsync -avP --progress user@host:/path/to/orpheus-indic-5679/ models/orpheus-indic-5679/
```

## Where to put it

The layout the default config expects:

```
ai4bharat-orpheus-indic-tts/
├── models/
│   └── orpheus-indic-5679/       <-- this directory name is the default
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── hf-cache/                     <-- SNAC codec cache; `mkdir -p hf-cache` before
│                                     the first `up`, or Docker creates it root-owned
│                                     and the container (uid 1000) cannot write it
├── config.yaml
└── docker-compose.yml
```

`models/` and `hf-cache/` are both in `.gitignore` — weights never get committed.

**If you put it somewhere else**, point the config at it. Remember that Compose
bind-mounts the host's `./models` to `/models` inside the container, so
`ORPHEUS_MODEL_PATH` is a *container* path:

```bash
# .env — the directory is named differently
ORPHEUS_MODEL_PATH=/models/my-checkpoint-name
```

To use a directory outside the repository, change the mount instead:

```yaml
# docker-compose.yml
volumes:
  - /data/checkpoints:/models:ro
```

Running without Docker? Then it is a host path, absolute or relative to the repo root:

```bash
ORPHEUS_MODEL_PATH=./models/orpheus-indic-5679
```

## Verify before starting

**1. All five files present, and the big one really is big:**

```bash
ls -la models/orpheus-indic-5679/
du -h models/orpheus-indic-5679/model.safetensors    # expect ~7.1G
```

A `model.safetensors` of a few hundred bytes means you downloaded a Git LFS pointer
file rather than the object. Re-download with `git lfs pull`, or use one of the
methods above.

**2. Checksums, if your source provided them.** A `SHA256SUMS` file is the quickest
way to rule out a truncated transfer:

```bash
cd models/orpheus-indic-5679 && sha256sum -c SHA256SUMS
```

No checksum file? Generate one now so future copies can be verified:

```bash
cd models/orpheus-indic-5679 && sha256sum * > SHA256SUMS
```

**3. The JSON parses:**

```bash
python3 -c "import json; print(json.load(open('models/orpheus-indic-5679/config.json'))['architectures'])"
```

## Start and confirm

```bash
docker compose up -d --build
docker compose logs -f tts
```

Look for these lines in order:

```
loading SNAC codec hubertsiuzdak/snac_24khz on cuda
GPU 0: <your GPU>, compute capability X.Y, NN.N GiB, 1 device(s) visible
roster: 22 languages, 40 speakers, 12 styles (template=indic)
loading model /models/orpheus-indic-5679 (dtype=auto quantization=fp8 max_num_seqs=256 ...)
warmup: concurrency 1
...
ready: serving orpheus-indic
```

Then confirm end to end:

```bash
curl -s localhost:9000/health | jq
# {"status":"ok","ready":true,...}

curl -s localhost:9000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"orpheus","voice":"Amit","input":"नमस्ते, आज मौसम बहुत अच्छा है।","response_format":"wav"}' \
  -o /tmp/test.wav

python3 -c "import wave; w=wave.open('/tmp/test.wav'); print(w.getnframes()/w.getframerate(), 's @', w.getframerate(), 'Hz')"
# expect a couple of seconds at 24000 Hz
```

## Problems

**`does not appear to have a file named config.json`** — the path is wrong, or points
at a parent directory rather than the checkpoint directory itself. Check
`ORPHEUS_MODEL_PATH` against what is actually mounted at `/models`:

```bash
docker compose exec tts ls -la /models
```

**`We couldn't connect to https://huggingface.co`** — the path was not found on disk,
so it was treated as a HuggingFace repo id and the download failed. Almost always a
typo in the path or a mount that did not land.

**Downloads a model you did not expect** — same cause. Anything that is not an
existing directory is taken as a repo id.

**Server starts but audio is silence or noise** — likely a mismatched
`tokenizer.json`. The audio-token ids and speaker markers must match the checkpoint;
a tokenizer from a different Orpheus variant loads without complaint and produces
garbage. Re-copy all five files from one source.

**Permission denied reading `/models`** — the container runs as UID 1000. Make the
directory readable:

```bash
chmod -R a+rX models/
```

## Serving a different Orpheus checkpoint

This server is not tied to the Indic checkpoint. To serve another one, point
`ORPHEUS_MODEL_PATH` at it and replace `voices.json` to describe it:

```json
{
  "prompt_template": "plain",
  "default_style": null,
  "styles": [],
  "languages": [
    {"code": "en", "name": "English",
     "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
     "sample": "Hello, this is a test of the speech system."}
  ]
}
```

`prompt_template` selects the prompt shape: `"indic"` uses the AI4Bharat
speaker/style markers, `"plain"` uses upstream Orpheus's `"{voice}: {text}"` form
(for example `canopylabs/orpheus-3b-0.1-ft`).

Speaker names should be unique across languages. That uniqueness is what lets an
OpenAI client select a language with nothing but the standard `voice` field; if a
name is shared, the server logs a warning at boot and clients must send the
`language` extension for that speaker.
