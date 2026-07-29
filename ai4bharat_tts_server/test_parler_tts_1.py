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
    bs = 11
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
    total_time = time.time()
    while len(model_runner.running_requests) > 0:
        idx = idx + 1
        start = time.time()
        model_runner.step()

        if idx== 1000000:
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
        print("model runner step",len(model_runner.running_requests),round(1000 * (time.time() - start),2 ),)
        if idx%60==0:
            start = time.time()
            model_runner.audio_decode()
            print("------audio decode------",round(1000 * (time.time() - start),2))
    model_runner.audio_decode()
    print('average step time',sum(step_times)/len(step_times))
    print('effective average step time',round(1000*(time.time()-total_time)/len(step_times),2))   
test_runner_obj()
