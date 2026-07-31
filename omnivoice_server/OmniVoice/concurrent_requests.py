"""
concurrent_load_test.py

Loads OmniVoice once, then fires N "concurrent" client requests via a
thread pool and reports per-request + aggregate timing.

NOTE: On a single GPU, CUDA compute for one process still executes on the
default stream, so this does not test true parallel GPU compute — it tests
how the model handles a burst of simultaneous requests (queuing/throughput),
which is the realistic signal for a live server.
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from omnivoice import OmniVoice

MODEL_ID = "k2-fsa/OmniVoice"
REF_AUDIO = "ref.wav"
REF_TEXT = "Transcription of the reference audio."
SAMPLE_RATE = 24000
NUM_REQUESTS = 10
NUM_WORKERS = 10  # drop this if you OOM on the 4070

TEXTS = [
    f"This is concurrent test request number {i}, checking system behavior under load."
    for i in range(NUM_REQUESTS)
]


def load_model():
    print("Loading model...")
    model = OmniVoice.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.float16)
    _ = model.generate(text="warmup", ref_audio=REF_AUDIO, ref_text=REF_TEXT)
    torch.cuda.synchronize()
    print("Model loaded and warmed up.\n")
    return model


def run_single_request(model, req_id, text):
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    audio = model.generate(text=text, ref_audio=REF_AUDIO, ref_text=REF_TEXT)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    gen_time = t_end - t_start
    audio_arr = audio[0]
    audio_duration = audio_arr.shape[-1] / SAMPLE_RATE
    rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")

    return {
        "id": req_id,
        "gen_time": gen_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
    }


def main():
    model = load_model()
    print(f"Dispatching {NUM_REQUESTS} concurrent requests (workers={NUM_WORKERS})...\n")

    batch_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(run_single_request, model, i, TEXTS[i]): i
            for i in range(NUM_REQUESTS)
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print(f"Request {r['id']:2d}: gen_time={r['gen_time']:.3f}s  "
                  f"audio_dur={r['audio_duration']:.3f}s  RTF={r['rtf']:.4f}")

    wall_clock_time = time.perf_counter() - batch_start

    gen_times = [r["gen_time"] for r in results]
    rtfs = [r["rtf"] for r in results]
    total_audio_duration = sum(r["audio_duration"] for r in results)

    print("\n--- Summary ---")
    print(f"Requests completed:          {len(results)}")
    print(f"Total wall-clock time:       {wall_clock_time:.3f}s")
    print(f"Sum of individual gen times: {sum(gen_times):.3f}s")
    print(f"Total audio generated:       {total_audio_duration:.3f}s")
    print(f"Avg per-request RTF:         {statistics.mean(rtfs):.4f}")
    print(f"Median per-request RTF:      {statistics.median(rtfs):.4f}")
    print(f"Min / Max gen_time:          {min(gen_times):.3f}s / {max(gen_times):.3f}s")
    print(f"Effective concurrent RTF:    {wall_clock_time / total_audio_duration:.4f}")
    print(f"Throughput (realtime-x):     {total_audio_duration / wall_clock_time:.2f}x")
    print(f"({NUM_REQUESTS} requests handled in {wall_clock_time:.2f}s wall-clock)")


if __name__ == "__main__":
    main()