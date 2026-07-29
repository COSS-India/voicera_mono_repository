import torch
import os
from inference.modeling import ParlerTTS
from inference.config import device
from inference.paging import VirtualMemory
from inference.runner import ParlerTTSModelRunner, TTSRequest

here = os.path.dirname(__file__)


@torch.no_grad()
def test_runner_obj():
    model_runner = ParlerTTSModelRunner(os.path.join(here, "checkpoints"))

    bs = 24
    requests = [
        TTSRequest(
            prompt="अरे, तुम आज कैसे हो? कैसे हो? कैसे हो? कैसे हो?",
            description="Vidya's voice is monotone.",
        )
        for _ in range(bs)
    ]
    for req in requests:
        model_runner.prefill(req)

    import time

    idx = 0
    step_ms = []
    while len(model_runner.running_requests) > 0:
        idx = idx + 1
        torch.cuda.synchronize()
        start = time.perf_counter()
        model_runner.step()
        model_runner.check_stopping_criteria()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        step_ms.append(elapsed)
        print(
            "model runner step",
            len(model_runner.running_requests),
            round(elapsed, 2),
        )
        if idx % 60000 == 0:
            start = time.perf_counter()
            model_runner.audio_decode()
            torch.cuda.synchronize()
            print("----------", round(1000 * (time.perf_counter() - start), 2))

    model_runner.audio_decode()
    # Skip first few steps (warmup / CUDA context)
    steady = step_ms[5:] if len(step_ms) > 5 else step_ms
    print(
        f"steps={len(step_ms)} median_ms={sorted(steady)[len(steady)//2]:.2f} "
        f"mean_ms={sum(steady)/len(steady):.2f}"
    )


test_runner_obj()
