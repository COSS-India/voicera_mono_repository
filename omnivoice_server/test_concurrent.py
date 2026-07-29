"""
test_concurrent.py

Sends all N requests as a single batched generate() call to one model on one GPU.
All users submit at t=0 and all receive their audio at the same time.

The key metric: how long does every user wait? (same for all — it's one batch)
  latency = time from submit → audio ready = single generate() call duration

Usage:
    python test_concurrent.py [--num_requests 10] [--model k2-fsa/OmniVoice] \
                              [--device cuda:0] [--res_dir results_concurrent/]
"""

import argparse
import os
import statistics
import time

import soundfile as sf
import torch

from omnivoice import OmniVoice

SAMPLING_RATE = 24_000
REF_AUDIO = "ref.wav"
REF_TEXT = "Hello, this is a reference audio for voice cloning."

TEXTS = [
    "Hello, this is user number {i} speaking. Please generate my audio as fast as possible.",
    "The quick brown fox jumps over the lazy dog, request number {i}.",
    "OmniVoice is a high quality text-to-speech system, user {i} here.",
    "This is a concurrent batch test from user {i}. I hope to get a response quickly.",
    "Speech synthesis at scale, checking latency for user {i}.",
]


def build_requests(num_requests: int):
    reqs = []
    for i in range(num_requests):
        tmpl = TEXTS[i % len(TEXTS)]
        reqs.append({
            "id": i,
            "text": tmpl.format(i=i),
            "ref_audio": REF_AUDIO,
            "ref_text": REF_TEXT,
        })
    return reqs


def load_model(model_id: str, device: str) -> OmniVoice:
    print(f"Loading model on {device} ...")
    model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=torch.float16)
    print("Warming up ...")
    model.generate(text=["warmup"], language=["en"],num_step=16,
                   ref_audio=[REF_AUDIO], ref_text=[REF_TEXT])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print("Ready.\n")
    return model


def get_parser():
    p = argparse.ArgumentParser(description="OmniVoice single-batch latency tester")
    p.add_argument("--num_requests", type=int, default=10)
    p.add_argument("--model",   type=str, default="k2-fsa/OmniVoice")
    p.add_argument("--device",  type=str, default="cuda:0")
    p.add_argument("--res_dir", type=str, default="results_concurrent/")
    return p


def main():
    args = get_parser().parse_args()
    os.makedirs(args.res_dir, exist_ok=True)

    model = load_model(args.model, args.device)
    requests = build_requests(args.num_requests)

    texts      = [r["text"]      for r in requests]
    ref_audios = [r["ref_audio"] for r in requests]
    ref_texts  = [r["ref_text"]  for r in requests]

    print(f"Throwing all {args.num_requests} requests at once as a single batch on {args.device} ...\n")

    # All users submit at exactly the same moment
    t_submit = time.perf_counter()

    audios = model.generate(
        text=texts,
        language=["en"] * args.num_requests,
        ref_audio=ref_audios,
        ref_text=ref_texts,
    )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    # Every user gets their audio at the same time
    latency = time.perf_counter() - t_submit

    # Per-request metrics
    results = []
    col = "  {:>8}  {:>10}  {:>8}"
    print(col.format("ID", "AUDIO_DUR", "RTF"))
    print("  " + "-" * 32)

    for req, audio in zip(requests, audios):
        audio_dur = audio.shape[-1] / SAMPLING_RATE
        rtf = latency / audio_dur if audio_dur > 0 else float("inf")

        save_path = os.path.join(args.res_dir, f"req_{req['id']:03d}.wav")
        sf.write(save_path, audio, SAMPLING_RATE)

        results.append({"id": req["id"], "audio_dur": audio_dur, "rtf": rtf})
        print(col.format(f"req_{req['id']:03d}", f"{audio_dur:.3f}s", f"{rtf:.4f}"))

    total_audio = sum(r["audio_dur"] for r in results)
    rtfs = [r["rtf"] for r in results]

    print()
    print("=" * 56)
    print(f"SUMMARY  (1 GPU · 1 model · {args.num_requests} requests batched)")
    print("=" * 56)
    print(f"  Requests:                  {len(results)}/{args.num_requests}")
    print(f"  Batch generation time:     {latency:.3f}s")
    print(f"  Total audio generated:     {total_audio:.3f}s")
    print()
    print(f"  Every user hears audio after: {latency:.3f}s")
    print()
    print(f"  RTF (gen_time / audio_dur per request)")
    print(f"    Mean:                    {statistics.mean(rtfs):.4f}")
    print(f"    Best:                    {min(rtfs):.4f}")
    print(f"    Worst:                   {max(rtfs):.4f}")
    print(f"\n  Output: {os.path.abspath(args.res_dir)}")
    print("=" * 56)


if __name__ == "__main__":
    main()
