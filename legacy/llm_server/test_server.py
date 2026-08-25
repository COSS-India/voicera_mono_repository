import requests
import json
import time
import sys

# ============================================================
# CONFIG — match these to your server.py settings
# ============================================================

HOST = "localhost"
PORT = 8000
MODEL = "Qwen/Qwen3-8B"

BASE_URL = f"http://{HOST}:{PORT}"

# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def print_pass(msg):
    print(f"  ✅ PASS — {msg}")

def print_fail(msg):
    print(f"  ❌ FAIL — {msg}")

def print_info(label, value):
    print(f"  {label:<25} {value}")

# ============================================================
# TEST 1 — Health Check
# ============================================================

def test_health():
    print_header("TEST 1: Health Check")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print_pass(f"Server is healthy (HTTP {r.status_code})")
            return True
        else:
            print_fail(f"Unexpected status code: {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_fail(f"Cannot connect to {BASE_URL} — is server.py running?")
        sys.exit(1)

# ============================================================
# TEST 2 — Model Available
# ============================================================

def test_models():
    print_header("TEST 2: Model Availability")
    r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
    data = r.json()
    models = [m["id"] for m in data.get("data", [])]
    print_info("Available models:", str(models))
    if MODEL in models:
        print_pass(f"'{MODEL}' is loaded and available")
        return True
    else:
        print_fail(f"'{MODEL}' not found. Found: {models}")
        return False

# ============================================================
# TEST 3 — Basic Inference (No Thinking)
# ============================================================

def test_basic_inference():
    print_header("TEST 3: Basic Inference (Thinking OFF)")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful voice assistant. /no_think"},
            {"role": "user", "content": "Say hello in one short sentence."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }

    start = time.time()
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
    elapsed = time.time() - start

    if r.status_code != 200:
        print_fail(f"HTTP {r.status_code}: {r.text}")
        return False

    data = r.json()
    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    reasoning = choice.get("reasoning", "")
    finish_reason = data["choices"][0]["finish_reason"]
    tokens_used = data["usage"]["completion_tokens"]

    print_info("Response:", repr(content.strip()))
    print_info("Reasoning field:", repr(reasoning.strip()) if reasoning else "empty ✅")
    print_info("Finish reason:", finish_reason)
    print_info("Completion tokens:", tokens_used)
    print_info("Round-trip time:", f"{elapsed:.2f}s")

    if content and content.strip():
        print_pass("Got a clean response in 'content' field")
    else:
        print_fail("'content' is empty — thinking mode may still be ON")
        return False

    if reasoning and reasoning.strip():
        print_fail(f"'reasoning' field is not empty — thinking is still ON: {repr(reasoning[:80])}")
        return False
    else:
        print_pass("Thinking is OFF — 'reasoning' field is empty")

    if finish_reason == "stop":
        print_pass("Model stopped naturally (not cut off)")
    else:
        print_fail(f"finish_reason is '{finish_reason}' — model was cut off, increase max_tokens")

    return True

# ============================================================
# TEST 4 — Streaming
# ============================================================

def test_streaming():
    print_header("TEST 4: Streaming Response")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful voice assistant. /no_think"},
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }

    start = time.time()
    first_token_time = None
    chunks = []

    try:
        with requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=30
        ) as r:
            for line in r.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk_data = json.loads(line[6:])
                        delta = chunk_data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time() - start
                            chunks.append(delta)

    except Exception as e:
        print_fail(f"Streaming error: {e}")
        return False

    full_response = "".join(chunks).strip()
    total_time = time.time() - start

    print_info("Full streamed response:", repr(full_response))
    print_info("Time to first token (TTFT):", f"{first_token_time:.2f}s" if first_token_time else "N/A")
    print_info("Total stream time:", f"{total_time:.2f}s")
    print_info("Total chunks received:", len(chunks))

    if full_response:
        print_pass("Streaming works — received chunked tokens")
    else:
        print_fail("No content received via stream")
        return False

    if first_token_time and first_token_time < 3.0:
        print_pass(f"TTFT is {first_token_time:.2f}s — good for telephony (< 3s)")
    elif first_token_time:
        print_fail(f"TTFT is {first_token_time:.2f}s — too slow for telephony (> 3s)")

    return True

# ============================================================
# TEST 5 — Telephony Simulation
# ============================================================

def test_telephony_simulation():
    print_header("TEST 5: Telephony Simulation (Multi-turn)")

    system_prompt = """You are a voice assistant for a telephony agent. 
Be concise, friendly, and conversational. Keep responses under 2 sentences. /no_think"""

    conversation = [
        {"role": "system", "content": system_prompt}
    ]

    turns = [
        "Hi, I'd like to book an appointment.",
        "Tomorrow at 3pm please.",
        "Yes, that works. Thank you!"
    ]

    print()
    all_passed = True
    for i, user_msg in enumerate(turns):
        conversation.append({"role": "user", "content": user_msg})

        start = time.time()
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": conversation,
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": False
            },
            timeout=30
        )
        elapsed = time.time() - start

        data = r.json()
        reply = data["choices"][0]["message"]["content"].strip()
        tokens = data["usage"]["completion_tokens"]

        conversation.append({"role": "assistant", "content": reply})

        print(f"  Turn {i+1}:")
        print(f"    User    : {user_msg}")
        print(f"    Agent   : {reply}")
        print(f"    Tokens  : {tokens} | Time: {elapsed:.2f}s")

        if not reply:
            print_fail("Empty response on turn {i+1}")
            all_passed = False

    if all_passed:
        print_pass("Multi-turn conversation worked end-to-end")
    return all_passed

# ============================================================
# TEST 6 — Latency Benchmark
# ============================================================

def test_latency_benchmark():
    print_header("TEST 6: Latency Benchmark (5 requests)")

    times = []
    for i in range(5):
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a voice assistant. /no_think"},
                {"role": "user", "content": f"What is {i+1} + {i+2}? Answer in one word."}
            ],
            "max_tokens": 20,
            "temperature": 0.7,
            "stream": False
        }
        start = time.time()
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=30)
        elapsed = time.time() - start
        times.append(elapsed)

        reply = r.json()["choices"][0]["message"]["content"].strip()
        print(f"  Request {i+1}: {elapsed:.2f}s → {repr(reply)}")

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)

    print()
    print_info("Average latency:", f"{avg:.2f}s")
    print_info("Min latency:", f"{min_t:.2f}s")
    print_info("Max latency:", f"{max_t:.2f}s")

    if avg < 2.0:
        print_pass(f"Average latency {avg:.2f}s is good for telephony")
    else:
        print_fail(f"Average latency {avg:.2f}s is high — consider reducing max_model_len or max_tokens")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Qwen3-8B vLLM Server — Test Suite")
    print(f"  Target: {BASE_URL}")
    print("="*55)

    results = {}
    results["health"]       = test_health()
    results["models"]       = test_models()
    results["inference"]    = test_basic_inference()
    results["streaming"]    = test_streaming()
    results["telephony"]    = test_telephony_simulation()
    test_latency_benchmark()

    print_header("SUMMARY")
    all_ok = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  🟢 All tests passed — server is ready for Pipecat!")
    else:
        print("  🔴 Some tests failed — check logs above.")
    print()
