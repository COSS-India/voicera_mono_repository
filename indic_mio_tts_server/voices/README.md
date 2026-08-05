# Preset voices

Indic-Mio is a zero-shot voice-cloning TTS: a *voice* is a speaker embedding
derived from one reference clip. The embedding is timbre only, so **one voice
works across all 22 Indic languages + English** — you do not need a voice per
language.

## Layout

```
voices/
  manifest.json      # {"default": <id>, "voices": [{name, gender, ref, source}, ...]}
  refs/<ref>.wav     # one clean reference clip per voice (5-15s, single speaker)
```

At startup the server derives each voice's embedding from its ref clip once and
caches it under `MIO_VOICES_CACHE_DIR` (default `/root/.cache/huggingface/
indic_mio_voices`, persisted in the HF volume). Subsequent boots load the cache.

If `manifest.json` is absent or no ref clip is usable, the server falls back to
the single legacy embedding (`MIO_REFERENCE_REPO`/`MIO_REFERENCE_FILE`) — i.e.
old behavior, nothing breaks.

## Populating the reference clips

The clips are **not committed** (they are audio you must audition and accept).
Two ways to get them:

1. **Curate from ai4bharat/Rasa** (recommended; CC-BY-4.0):
   ```bash
   pip install datasets soundfile
   export HF_TOKEN=hf_...            # Rasa is gated: accept terms on the HF page first
   python scripts/fetch_rasa_refs.py   # writes candidate clips into voices/refs/
   ```
   Audition the clips, keep the best ~10s each, name them per `manifest.json`
   (`aditi.wav`, `meera.wav`, ...). Adjust languages/speakers in the script to
   taste.

2. **Bring your own**: drop 5 clean mono wavs into `refs/`, named to match the
   `ref` fields in `manifest.json`.

Then (optionally) pre-bake the embeddings so first boot is instant, and rebuild:
```bash
python scripts/build_voices.py        # encodes refs -> cached .pt (needs GPU + codec)
```

## Choosing / adding voices

Edit `manifest.json`: add an entry `{name, gender, ref}` and a matching wav in
`refs/`. `name` is the voice id shown to users and stored in the agent's
`tts_model.speaker`; keep it stable once shipped. The frontend voice list lives
in `voicera_frontend/tts.json` (`indic-mio.voices`) — keep the two in sync.

## Attribution

Reference clips sourced from **ai4bharat/Rasa** (https://huggingface.co/datasets/ai4bharat/Rasa),
licensed **CC-BY-4.0**. See `NOTICE`.
