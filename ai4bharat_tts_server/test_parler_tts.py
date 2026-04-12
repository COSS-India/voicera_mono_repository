import torch
import os
import time
from inference.runner import ParlerTTSModelRunner, TTSRequest

here = os.path.dirname(__file__)

HINDI_PROMPTS = [
    "नमस्ते, आप कैसे हैं? आज का दिन कैसा रहा?",
    "मुझे बताइए, क्या आपने खाना खाया?",
    "आज मौसम बहुत अच्छा है, है ना?",
    "क्या आप मुझे रास्ता बता सकते हैं?",
    "मेरा नाम विद्या है और मैं दिल्ली से हूँ।",
    "कृपया थोड़ा धीरे बोलिए, मुझे समझ नहीं आया।",
    "आपकी आवाज़ बहुत मधुर है।",
    "मैं कल आपसे मिलने आऊँगा।",
]

DESCRIPTION  = "Vidya's voice is monotone."
BATCH_SIZE   = 10
DECODE_EVERY = 60   # match your server's --decode-every


@torch.no_grad()
def benchmark_runner():
    print(f"Loading model runner from: {os.path.join(here, 'checkpoints')}")
    model_runner = ParlerTTSModelRunner(os.path.join(here, "checkpoints"))

    requests = [
        TTSRequest(
            prompt=HINDI_PROMPTS[i % len(HINDI_PROMPTS)],
            description=DESCRIPTION,
        )
        for i in range(BATCH_SIZE)
    ]

    print(f"\nSending {BATCH_SIZE} requests...\n")

    # ── Prefill ───────────────────────────────────────────────────────────────
    prefill_start = time.perf_counter()
    for req in requests:
        model_runner.prefill(req)
    prefill_end  = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    # ── Decode loop ───────────────────────────────────────────────────────────
    step_times   = []
    ttff         = None          # time from prefill_start → first real audio bytes
    decode_start = time.perf_counter()
    step_idx     = 0

    while len(model_runner.running_requests) > 0:
        step_start = time.perf_counter()
        model_runner.step()
        model_runner.check_stopping_criteria()
        step_elapsed = time.perf_counter() - step_start
        step_times.append(step_elapsed * 1000)
        step_idx += 1

        print(
            f"Step {step_idx:04d} | "
            f"Running: {len(model_runner.running_requests):>3} | "
            f"Step time: {step_elapsed * 1000:.2f} ms"
        )

        # Periodic audio decode — mirrors server decode_every logic
        if step_idx % DECODE_EVERY == 0:
            t0 = time.perf_counter()
            audio_dict = model_runner.audio_decode()
            audio_decode_ms = (time.perf_counter() - t0) * 1000
            print(f"  └─ audio_decode() at step {step_idx}: {audio_decode_ms:.2f} ms")

            # TTFF = prefill_start → first non-empty audio chunk returned
            if ttff is None:
                has_audio = any(
                    arr is not None and hasattr(arr, '__len__') and len(arr) > 0
                    for arr in audio_dict.values()
                )
                if has_audio:
                    ttff = (time.perf_counter() - prefill_start) * 1000
                    print(f"  └─ *** TTFF recorded: {ttff:.2f} ms ***")

    total_decode_time = time.perf_counter() - decode_start

    # ── Final audio decode (flush remaining tokens) ───────────────────────────
    t0 = time.perf_counter()
    model_runner.audio_decode()
    final_decode_ms = (time.perf_counter() - t0) * 1000

    total_wall_time = prefill_time + total_decode_time

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'BENCHMARK RESULTS':^60}")
    print("=" * 60)
    print(f"  Batch size                   : {BATCH_SIZE}")
    print(f"  Decode every (steps)         : {DECODE_EVERY}")
    print(f"  Total decode steps           : {step_idx}")
    print()
    print(f"  Prefill time                 : {prefill_time * 1000:.2f} ms")
    print(f"  Total decode time            : {total_decode_time * 1000:.2f} ms")
    print(f"  Final audio_decode() time    : {final_decode_ms:.2f} ms")
    print(f"  Total wall time              : {total_wall_time * 1000:.2f} ms")
    print()
    print(f"  TTFF (prefill → first audio) : {f'{ttff:.2f} ms' if ttff else 'not reached (increase steps or lower DECODE_EVERY)'}")
    print(f"  Expected TTFF estimate       : {prefill_time * 1000 + DECODE_EVERY * (sum(step_times[:DECODE_EVERY]) / max(len(step_times[:DECODE_EVERY]), 1)):.2f} ms  (prefill + {DECODE_EVERY} steps)")
    print()
    print(f"  Avg step time                : {sum(step_times) / len(step_times):.2f} ms")
    print(f"  Min step time                : {min(step_times):.2f} ms")
    print(f"  Max step time                : {max(step_times):.2f} ms")
    print(f"  Avg time per request (amort) : {total_wall_time * 1000 / BATCH_SIZE:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_runner()