import asyncio
import aiohttp
import argparse
import io
import statistics
import time
import wave

URL = "http://localhost:8091/v1/audio/speech"

TEXT = (
    "Hello! This is OmniVoice running on vLLM Omni. "
    "This sentence is being used for latency benchmarking."
)

payload = {
    "model": "k2-fsa/OmniVoice",
    "input": TEXT,
}


async def benchmark_request(session, idx):
    start = time.perf_counter()

    async with session.post(URL, json=payload) as resp:
        first_byte = time.perf_counter()

        audio = await resp.read()

    end = time.perf_counter()

    # Parse WAV duration
    with wave.open(io.BytesIO(audio), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        audio_duration = frames / rate

    return {
        "id": idx,
        "status": resp.status,
        "ttfb": first_byte - start,
        "latency": end - start,
        "audio_duration": audio_duration,
        "rtf": (end - start) / audio_duration,
        "size": len(audio),
    }


def percentile(values, p):
    values = sorted(values)
    k = int((len(values) - 1) * p / 100)
    return values[k]


async def run(concurrency, total_requests):

    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:

        sem = asyncio.Semaphore(concurrency)

        async def worker(i):
            async with sem:
                return await benchmark_request(session, i)

        tasks = [worker(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    return results


def print_summary(results):

    latencies = [r["latency"] for r in results]
    ttfb = [r["ttfb"] for r in results]
    rtf = [r["rtf"] for r in results]

    print("\n================ SUMMARY ================\n")

    print(f"Requests           : {len(results)}")
    print(f"Success            : {sum(r['status']==200 for r in results)}")

    print()

    print(f"Average latency    : {statistics.mean(latencies):.3f}s")
    print(f"P50 latency        : {percentile(latencies,50):.3f}s")
    print(f"P95 latency        : {percentile(latencies,95):.3f}s")
    print(f"P99 latency        : {percentile(latencies,99):.3f}s")

    print()

    print(f"Average TTFB       : {statistics.mean(ttfb):.3f}s")

    print()

    print(f"Average RTF        : {statistics.mean(rtf):.4f}")

    print("\n============= PER REQUEST =============\n")

    print(
        f"{'ID':<6}"
        f"{'LATENCY':>12}"
        f"{'TTFB':>10}"
        f"{'AUDIO':>12}"
        f"{'RTF':>10}"
        f"{'SIZE':>10}"
    )

    print("-"*64)

    for r in results:
        print(
            f"{r['id']:<6}"
            f"{r['latency']:>11.3f}s"
            f"{r['ttfb']:>9.3f}s"
            f"{r['audio_duration']:>11.3f}s"
            f"{r['rtf']:>9.4f}"
            f"{r['size']:>10}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Total requests",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run(
            concurrency=args.concurrency,
            total_requests=args.requests,
        )
    )

    print_summary(results)