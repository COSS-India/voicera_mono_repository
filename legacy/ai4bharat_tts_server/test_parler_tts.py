import torch
import os
import time
import threading
import subprocess
from inference.runner import ParlerTTSModelRunner, TTSRequest

here = os.path.dirname(__file__)


@torch.no_grad()
def test_runner_obj():
    model_runner = ParlerTTSModelRunner(
        os.path.join(here, "checkpoints"), use_cuda_graph=True
    )

    bs = 32
    requests = [
        TTSRequest(
            prompt="अरे, तुम आज कैसे हो? कैसे हो? कैसे हो? कैसे हो?",
            description="Vidya's voice is monotone.",
        )
        for _ in range(bs)
    ]
    for req in requests:
        model_runner.prefill(req)

    gpu_utils = []
    stop_poll = False

    def _poll_gpu():
        while not stop_poll:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                gpu_utils.append(int(out.split()[0]))
            except Exception:
                pass
            time.sleep(0.02)

    poller = threading.Thread(target=_poll_gpu, daemon=True)
    poller.start()

    idx = 0
    step_events = []
    max_code = 0
    while len(model_runner.running_requests) > 0:
        idx += 1
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        model_runner.step()
        end_ev.record()
        step_events.append((start_ev, end_ev))
        # Evict finished requests every step (continuous batching).
        model_runner.check_stopping_criteria()
        if idx <= 5 or idx % 60 == 0:
            print(
                "model runner step",
                len(model_runner.running_requests),
                f"graphs={len(model_runner._cuda_graphs)}",
            )
        if idx % 240 == 0:
            torch.cuda.synchronize()
            for req in model_runner.running_requests.values():
                t = torch.cat(req.token_cache, -1)
                fixed = model_runner._stacked_audio_codes_from_timeline(t)
                if fixed is not None:
                    max_code = max(max_code, int(fixed.max()))
            print("---------- code spot-check ok, max_code", max_code)
            if max_code > 1023:
                raise RuntimeError(f"invalid audio code {max_code}")

    torch.cuda.synchronize()
    stop_poll = True
    poller.join(timeout=0.5)

    step_ms = [s.elapsed_time(e) for s, e in step_events]
    steady = step_ms[5:] if len(step_ms) > 5 else step_ms
    replay = [m for m in steady if m < 20]
    if not replay:
        replay = steady
    util_msg = ""
    if gpu_utils:
        gs = sorted(gpu_utils)
        util_msg = (
            f" gpu_util_median={gs[len(gs)//2]} "
            f"gpu_util_p90={gs[int(0.9*len(gs))]} "
            f"gpu_util_max={max(gs)} "
            f"gpu_util_mean={sum(gs)/len(gs):.1f}"
        )
    print(
        f"steps={len(step_ms)} median_ms={sorted(replay)[len(replay)//2]:.2f} "
        f"mean_ms={sum(replay)/len(replay):.2f} "
        f"cuda_graph={model_runner.use_cuda_graph} max_audio_code={max_code}"
        f"{util_msg}"
    )


test_runner_obj()
