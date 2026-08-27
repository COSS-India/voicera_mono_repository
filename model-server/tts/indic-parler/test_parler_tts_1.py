import os
import time

import torch
from inference.runner import ParlerTTSModelRunner, TTSRequest

here = os.path.dirname(__file__)


@torch.no_grad()
def test_runner_obj():
    model_runner = ParlerTTSModelRunner(os.path.join(here, "checkpoints"))
    step_times = []
    bs = 24
    decode_every = 60
    requests = [
        TTSRequest(
            prompt="अरे, तुम आज कैसे हो? कैसे हो? कैसे हो? कैसे हो?",
            description="Vidya's voice is monotone.",
        )
        for _ in range(bs)
    ]
    for req in requests:
        model_runner.prefill(req)

    idx = 0
    total_time = time.time()
    dac_ms_total = 0.0
    while len(model_runner.running_requests) > 0:
        idx += 1
        start = time.time()
        model_runner.step()
        model_runner.check_stopping_criteria()
        step_ms = 1000 * (time.time() - start)
        step_times.append(step_ms)
        if idx <= 5 or idx % 60 == 0:
            print("model runner step", len(model_runner.running_requests), round(step_ms, 2))

        # Spread EOS finals while the batch is still stepping.
        if model_runner._pending_final_tokens and idx % 5 == 0:
            old_live = model_runner._dac_max_live_per_tick
            model_runner._dac_max_live_per_tick = 0
            t0 = time.time()
            model_runner.audio_decode()
            dac_ms_total += 1000 * (time.time() - t0)
            model_runner._dac_max_live_per_tick = old_live

        if idx % decode_every == 0:
            t0 = time.time()
            model_runner.audio_decode()
            dt = 1000 * (time.time() - t0)
            dac_ms_total += dt
            print("------audio decode------", round(dt, 2))

    while model_runner._pending_final_tokens:
        t0 = time.time()
        model_runner.audio_decode()
        dt = 1000 * (time.time() - t0)
        dac_ms_total += dt
        print("------final drain------", round(dt, 2))

    avg = sum(step_times) / len(step_times)
    effective = 1000 * (time.time() - total_time) / len(step_times)
    print("average step time", avg)
    print("effective average step time", round(effective, 2))
    print(
        "dac amortized ms/step",
        round(dac_ms_total / len(step_times), 2),
        "total_dac_ms",
        round(dac_ms_total, 1),
    )


test_runner_obj()
