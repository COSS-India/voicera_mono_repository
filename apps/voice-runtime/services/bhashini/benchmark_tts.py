#!/usr/bin/env python3
"""Benchmark Bhashini NVCF gRPC TTS latency under concurrent load.

Uses the same gRPC endpoint and protobuf API as services/bhashini/tts.py
(grpc.nvcf.nvidia.com, TTSService.Synthesize streaming).

Usage (from voice_2_voice_server/):
    python services/bhashini/benchmark_tts.py
    python services/bhashini/benchmark_tts.py --output-dir ./tts_out
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np

BHASHINI_DIR = Path(__file__).resolve().parent
ROOT = BHASHINI_DIR.parents[1]
sys.path.insert(0, str(BHASHINI_DIR))
sys.path.insert(0, str(ROOT))

import tts_pb2  # noqa: E402
import tts_pb2_grpc  # noqa: E402

# =============================================================================
# CONFIG — edit these variables for your test run
# =============================================================================

# Number of identical requests fired in parallel.
CONCURRENT_REQUESTS = 5

# Marathi sentence sent CONCURRENT_REQUESTS times in parallel.
BENCHMARK_TEXT = "कृपया थांबा, मी माहिती शोधत आहे"

SPEAKER = "Radha"
DESCRIPTION = (
    "speaks with a slightly higher pitch in a close sounding environment. "
    "The voice is clear with subtle emotional depth and a normal pace "
    "all captured in high-quality recording."
)
LANGUAGE = "mr"
DEFAULT_SAMPLE_RATE = 44100
GRPC_TIMEOUT_S = 120.0

# =============================================================================

ENV_PATH = ROOT / ".env"
DEFAULT_GRPC_URL = "grpc.nvcf.nvidia.com:443"


@dataclass
class SynthesisResult:
    request_index: int
    text: str
    ttft_ms: float | None
    total_ms: float
    chunk_count: int
    pcm_bytes: bytes
    sample_rate: int
    error: str | None = None


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        os.environ.setdefault(key.strip(), value)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _full_description(speaker: str, description: str) -> str:
    if speaker:
        return f"{speaker} {description}"
    return description


def _to_pcm16_bytes(audio_chunk: np.ndarray) -> bytes:
    if np.issubdtype(audio_chunk.dtype, np.floating):
        pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()
    if audio_chunk.dtype == np.int16:
        return audio_chunk.tobytes()
    if np.issubdtype(audio_chunk.dtype, np.integer):
        pcm = np.clip(audio_chunk, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        return pcm.tobytes()
    pcm = np.clip(audio_chunk.astype(np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def synthesize(
    text: str,
    *,
    request_index: int,
    speaker: str,
    description: str,
    language: str,
    timeout_s: float,
) -> SynthesisResult:
    auth_token = _require_env("BHASHINI_TTS_AUTH_TOKEN")
    function_id = _require_env("BHASHINI_TTS_FUNCTION_ID")
    grpc_url = os.getenv("BHASHINI_TTS_GRPC_URL", DEFAULT_GRPC_URL).strip()

    metadata = [
        ("authorization", f"Bearer {auth_token}"),
        ("function-id", function_id),
    ]
    request = tts_pb2.SynthesizeRequest(
        prompt=text,
        description=_full_description(speaker, description),
        language=language,
    )

    credentials = grpc.ssl_channel_credentials()
    started_at = time.perf_counter()
    first_chunk_at: float | None = None
    chunk_count = 0
    audio_parts: list[bytes] = []
    sample_rate = DEFAULT_SAMPLE_RATE

    with grpc.secure_channel(grpc_url, credentials) as channel:
        stub = tts_pb2_grpc.TTSServiceStub(channel)
        responses = stub.Synthesize(request, metadata=metadata, timeout=timeout_s)

        for response in responses:
            which = response.WhichOneof("payload")

            if which == "meta":
                sample_rate = response.meta.sample_rate or sample_rate

            elif which == "audio":
                arr = np.frombuffer(response.audio.pcm_data, dtype=np.float32)
                if arr.size > 0:
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    chunk_count += 1
                    audio_parts.append(_to_pcm16_bytes(arr))

            elif which == "done":
                break

    finished_at = time.perf_counter()
    ttft_ms = (first_chunk_at - started_at) * 1000.0 if first_chunk_at is not None else None
    total_ms = (finished_at - started_at) * 1000.0

    return SynthesisResult(
        request_index=request_index,
        text=text,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        chunk_count=chunk_count,
        pcm_bytes=b"".join(audio_parts),
        sample_rate=sample_rate,
    )


def _run_request(
    request_index: int,
    text: str,
    speaker: str,
    description: str,
    language: str,
    timeout_s: float,
) -> SynthesisResult:
    try:
        return synthesize(
            text,
            request_index=request_index,
            speaker=speaker,
            description=description,
            language=language,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return SynthesisResult(
            request_index=request_index,
            text=text,
            ttft_ms=None,
            total_ms=0.0,
            chunk_count=0,
            pcm_bytes=b"",
            sample_rate=DEFAULT_SAMPLE_RATE,
            error=str(exc),
        )


def _print_result(result: SynthesisResult, total_requests: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"Request {result.request_index}/{total_requests}")
    print(f"{'=' * 60}")
    print(f"Text       : {result.text!r}")
    if result.error:
        print(f"Error      : {result.error}")
        return
    if result.ttft_ms is None:
        print("TTFT       : N/A (no audio received)")
    else:
        print(f"TTFT       : {result.ttft_ms:.1f} ms")
    print(f"Total time : {result.total_ms:.1f} ms")
    print(f"Chunks     : {result.chunk_count}")
    print(f"Sample rate: {result.sample_rate} Hz")
    print(f"Audio bytes: {len(result.pcm_bytes)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Benchmark Bhashini NVCF gRPC TTS with {CONCURRENT_REQUESTS} concurrent requests"
    )
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=CONCURRENT_REQUESTS,
        help=f"Number of parallel requests (CONFIG: {CONCURRENT_REQUESTS})",
    )
    parser.add_argument(
        "--text",
        default=BENCHMARK_TEXT,
        help="Input text for every request",
    )
    parser.add_argument("--speaker", default=SPEAKER, help=f"Speaker name (CONFIG: {SPEAKER})")
    parser.add_argument("--description", default=DESCRIPTION, help="Voice/style description")
    parser.add_argument("--language", default=LANGUAGE, help=f"Language code (CONFIG: {LANGUAGE})")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=GRPC_TIMEOUT_S,
        help=f"Per-request gRPC timeout in seconds (CONFIG: {GRPC_TIMEOUT_S})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save output WAV files",
    )
    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be >= 1")

    _load_env(ENV_PATH)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    grpc_url = os.getenv("BHASHINI_TTS_GRPC_URL", DEFAULT_GRPC_URL).strip()
    suite_started = time.perf_counter()
    print("--- active config ---")
    print(f"  grpc_url={grpc_url}")
    print(f"  concurrent_requests={args.requests}")
    print(f"  language={args.language}")
    print(f"  speaker={args.speaker!r}")
    print(f"  text={args.text!r}\n")
    print(f"Sending {args.requests} parallel requests...")

    with ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = {
            executor.submit(
                _run_request,
                i,
                args.text,
                args.speaker,
                args.description,
                args.language,
                args.timeout_s,
            ): i
            for i in range(1, args.requests + 1)
        }
        results: list[SynthesisResult] = []
        for future in as_completed(futures):
            results.append(future.result())

    suite_total_ms = (time.perf_counter() - suite_started) * 1000.0
    results.sort(key=lambda r: r.request_index)

    for result in results:
        _print_result(result, args.requests)
        if args.output_dir is not None and result.pcm_bytes and not result.error:
            wav_path = args.output_dir / f"concurrent_{result.request_index}.wav"
            _write_wav(wav_path, result.pcm_bytes, result.sample_rate)
            print(f"Saved WAV  : {wav_path}")

    ok_results = [r for r in results if not r.error and r.chunk_count > 0]
    ttft_values = [r.ttft_ms for r in ok_results if r.ttft_ms is not None]
    total_values = [r.total_ms for r in ok_results]

    print(f"\n{'=' * 60}")
    print(f"Summary ({args.requests} parallel requests, same sentence)")
    print(f"{'=' * 60}")
    print(f"Requests completed : {len(ok_results)}/{args.requests}")
    print(f"Wall-clock time    : {suite_total_ms:.1f} ms")
    print()
    print(f"{'#':<4} {'TTFT (ms)':>10} {'Total (ms)':>12} {'Chunks':>8} {'Status':>10}")
    print("-" * 50)
    for r in results:
        ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms is not None else "N/A"
        total = f"{r.total_ms:.1f}" if not r.error else "—"
        status = "OK" if not r.error and r.chunk_count > 0 else "FAIL"
        print(f"{r.request_index:<4} {ttft:>10} {total:>12} {r.chunk_count:>8} {status:>10}")

    if ttft_values:
        print()
        print(f"Avg TTFT  : {sum(ttft_values) / len(ttft_values):.1f} ms")
        print(f"Max TTFT  : {max(ttft_values):.1f} ms")
    if total_values:
        print(f"Avg total : {sum(total_values) / len(total_values):.1f} ms")
        print(f"Max total : {max(total_values):.1f} ms")

    raise SystemExit(0 if len(ok_results) == args.requests else 1)


if __name__ == "__main__":
    main()
