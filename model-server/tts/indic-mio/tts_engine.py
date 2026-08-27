"""Two-stage streaming Indic-Mio TTS engine.

Stage 1 (token generation): a vLLM OpenAI-compatible server runs SPRINGLab/Indic-Mio
(a Qwen3-0.6B fine-tune). We open a *streaming* chat completion; the model emits
speech tokens as the literal strings "<|s_1234|>". vLLM owns all concurrency here
(continuous batching + paged KV), so this process stays thin and I/O-bound.

Stage 2 (codec decode): MioCodec turns the content-token indices into a waveform.
To keep time-to-first-byte low we decode *incrementally* as tokens stream in,
rather than waiting for the whole utterance. Each incremental decode runs over the
full token prefix (full left context) but holds back a short look-ahead tail so
emitted audio always had enough right context -> no flush-boundary artifacts.
Decode is bounded by a semaphore and runs off the event loop.

Set MIO_STREAM_DECODE=false to fall back to a single whole-utterance decode.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator

import aiohttp
import numpy as np

from config import DEFAULT_SAMPLE_RATE, SPEECH_TOKEN_PATTERN, Config

logger = logging.getLogger("indic_mio.engine")

_TOKEN_RE = re.compile(SPEECH_TOKEN_PATTERN)


class TTSGenerationError(RuntimeError):
    """Raised when generation or decoding cannot produce audio."""


def parse_speech_tokens(text: str) -> list[int]:
    """Extract all MioCodec content-token indices from raw LLM output text."""
    return [int(m) for m in _TOKEN_RE.findall(text)]


class MioTTSEngine:
    """Owns the vLLM HTTP client and the MioCodec model."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._codec: Any = None
        self._torch: Any = None
        self._global_embedding: Any = None  # legacy single-voice fallback embedding
        # Preset voices: id -> speaker global_embedding. Populated from the voices
        # bundle at load_codec(); empty when no bundle is present (legacy mode).
        self._embeddings: dict[str, Any] = {}
        self._default_voice: str | None = None
        self._sample_rate: int = DEFAULT_SAMPLE_RATE
        # Bound concurrent GPU decodes across all in-flight requests.
        import asyncio  # local: avoid importing at module scope for test tooling

        self._decode_sem = asyncio.Semaphore(max(1, config.decode_concurrency))

    # ---- lifecycle -------------------------------------------------------

    def load_codec(self) -> None:
        """Load MioCodec + the speaker embedding onto the GPU. Blocking; call once."""
        import torch  # local import: only the server process needs CUDA
        from miocodec import MioCodecModel

        logger.info("Loading MioCodec: %s", self._config.codec_model_id)
        codec = MioCodecModel.from_pretrained(self._config.codec_model_id)
        codec = codec.eval().to(self._config.device)
        self._codec = codec
        self._torch = torch

        sr = getattr(getattr(codec, "config", None), "sample_rate", None)
        self._sample_rate = int(sr) if sr else DEFAULT_SAMPLE_RATE
        logger.info("MioCodec ready (sample_rate=%d Hz)", self._sample_rate)

        # Preferred path: a bundle of preset voices. If none is present (no
        # manifest / no usable ref clips) fall back to the single legacy voice so
        # the server still works exactly as before.
        self._embeddings = self._load_voice_embeddings()
        if self._embeddings:
            self._default_voice = self._resolve_default_voice()
            logger.info(
                "Preset voices ready: %s (default=%s)",
                ", ".join(sorted(self._embeddings)),
                self._default_voice,
            )
        else:
            self._global_embedding = self._load_or_build_speaker_embedding()
            logger.info(
                "No preset voices; using legacy single embedding (shape=%s)",
                tuple(self._global_embedding.shape),
            )

    # ---- preset voices ---------------------------------------------------

    def _resolve_default_voice(self) -> str:
        """Pick the default voice id from config, else manifest, else first."""
        want = (self._config.default_voice or "").strip()
        if want and want in self._embeddings:
            return want
        if want:
            logger.warning("MIO_DEFAULT_VOICE=%r not in voices; ignoring", want)
        if self._manifest_default and self._manifest_default in self._embeddings:
            return self._manifest_default
        return sorted(self._embeddings)[0]

    def _load_voice_embeddings(self) -> dict:
        """Build the {voice_id: global_embedding} map from the voices bundle.

        Reads `<voices_dir>/manifest.json`; for each voice, reuses a cached
        `<voices_cache_dir>/<id>.pt` if present, else derives it from the voice's
        reference clip (`<voices_dir>/refs/<ref>`) and caches it. A voice whose
        ref is missing/unreadable is skipped with a warning. Returns {} if the
        manifest is absent or no voice could be built (-> legacy fallback).
        """
        import json
        import os

        self._manifest_default = ""
        manifest_path = os.path.join(self._config.voices_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            logger.info("No voices manifest at %s", manifest_path)
            return {}

        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Cannot read voices manifest %s: %s", manifest_path, e)
            return {}

        self._manifest_default = str(manifest.get("default", "") or "")
        cache_dir = self._config.voices_cache_dir
        refs_dir = os.path.join(self._config.voices_dir, "refs")

        embeddings: dict = {}
        for entry in manifest.get("voices", []):
            name = str(entry.get("name", "") or "").strip()
            if not name:
                continue
            cache_path = os.path.join(cache_dir, f"{name}.pt")
            try:
                if os.path.exists(cache_path):
                    embeddings[name] = self._load_cached_embedding(cache_path)
                    continue
                ref_name = str(entry.get("ref", "") or "").strip()
                ref_path = os.path.join(refs_dir, ref_name) if ref_name else ""
                if not ref_path or not os.path.exists(ref_path):
                    logger.warning("Voice %r: ref clip missing (%s); skipping", name, ref_path)
                    continue
                emb = self.encode_reference(ref_path)
                self._save_cached_embedding(emb, cache_path)
                embeddings[name] = emb
            except Exception as e:  # noqa: BLE001 - one bad voice must not sink the rest
                logger.warning("Voice %r: failed to build embedding: %s", name, e)
        return embeddings

    def _load_cached_embedding(self, path: str):
        torch = self._torch
        logger.info("Loading cached voice embedding: %s", path)
        emb = torch.load(path, map_location=self._config.device)
        return emb.to(device=self._config.device, dtype=torch.float32)

    def _save_cached_embedding(self, emb, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._torch.save(emb.cpu(), path)
        logger.info("Cached voice embedding -> %s", path)

    def encode_reference(self, ref_path: str):
        """Derive a speaker global_embedding from a local reference wav.

        Shared by the preset-voice loader and the offline build script. The clip
        is resampled to the codec sample rate and encoded once.
        """
        torch = self._torch
        from miocodec.util import load_audio

        logger.info("Encoding reference clip: %s", ref_path)
        waveform = load_audio(ref_path, sample_rate=self._sample_rate)
        waveform = waveform.to(device=self._config.device, dtype=torch.float32)
        with torch.inference_mode():
            feats = self._codec.encode(waveform, return_content=False, return_global=True)
        return feats.global_embedding.detach().to(
            device=self._config.device, dtype=torch.float32
        )

    def _resolve_embedding(self, voice: str | None):
        """Return the decode embedding for a requested voice id.

        Preset mode: exact match -> that voice; unknown/None -> default voice.
        Legacy mode (no bundle): always the single global embedding.
        """
        if self._embeddings:
            key = (voice or "").strip()
            if key and key in self._embeddings:
                return self._embeddings[key]
            if key:
                logger.debug("Unknown voice %r; using default %s", key, self._default_voice)
            return self._embeddings[self._default_voice]
        return self._global_embedding

    def _load_or_build_speaker_embedding(self):
        """Return the cached speaker global_embedding, deriving it once if absent.

        MioCodec.decode() requires a global_embedding (speaker). We encode one
        Indic-Mio reference sample a single time and cache the vector to disk, so
        subsequent boots (and every request) reuse it without touching the encoder.
        """
        import os

        torch = self._torch
        path = self._config.speaker_embed_path
        device = self._config.device

        if path and os.path.exists(path):
            logger.info("Loading cached speaker embedding: %s", path)
            emb = torch.load(path, map_location=device)
            return emb.to(device=device, dtype=torch.float32)

        logger.info(
            "No cached speaker embedding; deriving from %s:%s",
            self._config.reference_repo,
            self._config.reference_file,
        )
        from huggingface_hub import hf_hub_download

        ref_path = hf_hub_download(
            repo_id=self._config.reference_repo, filename=self._config.reference_file
        )
        emb = self.encode_reference(ref_path)

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(emb.cpu(), path)
            logger.info("Cached speaker embedding -> %s", path)
        return emb

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=self._config.llm_timeout)
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    # ---- public API ------------------------------------------------------

    async def synthesize_stream(
        self, text: str, voice: str | None = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """Yield float32 mono PCM chunks as they are produced.

        `voice` selects the speaker embedding (preset voice id); unknown/None ->
        default voice, or the legacy single embedding if no voices bundle exists.

        Raises TTSGenerationError before the first chunk if generation fails.
        """
        clean = text.strip()
        if not clean:
            raise TTSGenerationError("empty text")
        if len(clean) > self._config.max_text_length:
            raise TTSGenerationError(f"text too long (max {self._config.max_text_length} chars)")

        cfg = self._config
        embedding = self._resolve_embedding(voice)
        margin = max(0, cfg.lookahead_tokens) * self._samples_per_token()

        tokens: list[int] = []
        decoded_at = 0          # token count at last decode
        emitted_samples = 0     # samples already yielded
        produced_any = False

        async for tokens in self._stream_tokens(clean, tokens):
            if not cfg.stream_decode:
                continue
            if len(tokens) - decoded_at < cfg.flush_tokens:
                continue
            decoded_at = len(tokens)
            wav = await self._decode(tokens, embedding)
            safe = wav.size - margin  # hold back the look-ahead tail
            if safe > emitted_samples:
                produced_any = True
                yield wav[emitted_samples:safe]
                emitted_samples = safe

        # Final decode over the complete sequence; emit everything remaining
        # (no hold-back at the true end of the utterance).
        if not tokens:
            raise TTSGenerationError("No speech tokens found in LLM output.")
        wav = await self._decode(tokens, embedding)
        if wav.size > emitted_samples:
            produced_any = True
            yield wav[emitted_samples:]

        if not produced_any:
            raise TTSGenerationError("decode produced no audio")

    def _samples_per_token(self) -> int:
        rate = max(1, self._config.token_rate_hz)
        return max(1, self._sample_rate // rate)

    # ---- stage 1: streaming token generation -----------------------------

    async def _stream_tokens(
        self, text: str, tokens: list[int]
    ) -> AsyncGenerator[list[int], None]:
        """Stream from vLLM, yielding the growing `tokens` list as new tokens arrive."""
        if self._session is None:
            raise TTSGenerationError("engine session not started")

        cfg = self._config
        payload = {
            "model": cfg.llm_model,
            "messages": [{"role": "user", "content": text}],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "repetition_penalty": cfg.repetition_penalty,
            "stream": True,
            # CRITICAL: "<|s_N|>" are special tokens. Without this vLLM strips
            # them during detokenization and no audio is produced.
            "skip_special_tokens": False,
        }
        url = f"{cfg.llm_base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.llm_api_key}"}

        buffer = ""       # accumulated text not yet fully parsed
        consumed = 0      # index in `buffer` up to which tokens are extracted

        try:
            async with self._session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise TTSGenerationError(f"vLLM returned {resp.status}: {body[:300]}")

                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue
                    if not delta:
                        continue

                    buffer += delta
                    new, consumed = self._extract_new_tokens(buffer, consumed)
                    if new:
                        tokens.extend(new)
                        yield tokens
        except aiohttp.ClientError as e:
            raise TTSGenerationError(f"vLLM request failed: {e}") from e

        # Trailing parse in case the stream ended mid-buffer.
        new, consumed = self._extract_new_tokens(buffer, consumed)
        if new:
            tokens.extend(new)
            yield tokens

    @staticmethod
    def _extract_new_tokens(buffer: str, consumed: int) -> tuple[list[int], int]:
        """Return tokens found after `consumed` and the new consumed index.

        Only fully-closed "<|s_N|>" matches are taken; a token split across stream
        chunks stays buffered until its closing "|>" arrives.
        """
        new: list[int] = []
        last_end = consumed
        for m in _TOKEN_RE.finditer(buffer, consumed):
            new.append(int(m.group(1)))
            last_end = m.end()
        return new, last_end

    # ---- stage 2: codec decode ------------------------------------------

    async def _decode(self, tokens: list[int], embedding) -> np.ndarray:
        import asyncio

        async with self._decode_sem:
            return await asyncio.to_thread(self._decode_blocking, tokens, embedding)

    def _decode_blocking(self, tokens: list[int], embedding) -> np.ndarray:
        torch = self._torch
        # MioCodecModel.decode(global_embedding, content_token_indices):
        #   content_token_indices is 1D (seq_len,); global_embedding is (dim,).
        indices = torch.tensor(tokens, dtype=torch.long, device=self._config.device)
        with torch.inference_mode():
            wav = self._codec.decode(
                embedding, content_token_indices=indices
            )
        arr = wav.detach().to("cpu", dtype=torch.float32).numpy()
        return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
