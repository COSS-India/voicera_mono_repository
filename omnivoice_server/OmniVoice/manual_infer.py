from omnivoice import OmniVoice
import soundfile as sf
import torch
import time

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

text = "Hello मी वसुधा कृषी विभाग महाराष्ट्र शासन ची आपली डिजिटल शेती सहाय्यक हा कॉल प्रशिक्षण आणि गुणवत्ता तपासणीसाठी नोंदवला जात आहे आपली वैयक्तिक माहिती तृतीय पक्षासोबत शेअर केली जाणार नाही मी आपली कशी मदत करू"
ref_audio = "ref.wav"
ref_text = "Transcription of the reference audio."

# --- Warmup (discard this run — CUDA kernel JIT/warmup skews first-call timing) ---
_ = model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text)
torch.cuda.synchronize()  # ensure warmup fully finishes before timing starts

# --- Timed run(s) ---
n_runs = 30
results = []

for i in range(n_runs):
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    audio = model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text)

    torch.cuda.synchronize()  # wait for all GPU work to actually finish
    t_end = time.perf_counter()

    gen_time = t_end - t_start
    audio_arr = audio[0]
    audio_duration = len(audio_arr) / 24000  # 24kHz sample rate
    rtf = gen_time / audio_duration

    results.append((gen_time, audio_duration, rtf))
    print(f"Run {i+1}: gen_time={gen_time:.3f}s | audio_dur={audio_duration:.3f}s | RTF={rtf:.4f}")

avg_rtf = sum(r[2] for r in results) / len(results)
print(f"\nAvg RTF over {n_runs} runs: {avg_rtf:.4f}")