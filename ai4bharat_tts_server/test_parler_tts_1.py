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
    step_times = []
    bs = 16
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
    while len(model_runner.running_requests) > 0:
        idx = idx + 1
        start = time.time()
        model_runner.step()

        if idx== 100:
            bs = 4
            requests = [
                TTSRequest(
                    prompt="अरे, तुम आज कैसे हो? कैसे हो? कैसे हो? कैसे हो?",
                    description="Vidya's voice is monotone. ",
                )
                for _ in range(bs)
            ]
            for req in requests:
                model_runner.prefill(req)

        model_runner.check_stopping_criteria()
        step_times.append(1000 * (time.time() - start))
        print("model runner step",len(model_runner.running_requests),1000 * (time.time() - start),)
        #if idx%60==0:
        #    model_runner.audio_decode()

    model_runner.audio_decode()
    print('average step time',sum(step_times)/len(step_times))
test_runner_obj()
