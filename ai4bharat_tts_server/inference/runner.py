import os
import time
import torch
import uuid
from inference.modeling import ParlerTTS
from inference.config import device
from inference.paging import VirtualMemory
import math
import transformers
import numpy as np

class TTSRequest:
    def __init__(self, prompt, description, pid=None):
        self.pid = uuid.uuid4().hex[:6] if pid is None else pid
        self.prompt = prompt
        self.description = description
        self.decoder_input_ids = []
        self.decoder_position_ids = []
        self.token_cache = []
        self.audio_to_yield = 0

    def __repr__(self):
        return f"""TTSRequest(
        pid={self.pid},
        prompt='{self.prompt}',
        description='{self.description}',
        decoder_input_ids={self.decoder_input_ids},
        decoder_position_ids={self.decoder_position_ids}
    )
    """

class ParlerTTSModelRunner:
    def __init__(self, checkpoint_path, play_steps=60):
        self.model = ParlerTTS(checkpoint_path).eval().to(device)
        self._maybe_quantize()
        num_kv_heads = self.model.config["text_encoder"]["num_heads"]
        head_dim = self.model.config["decoder"]["hidden_size"] // num_kv_heads
        num_layers = self.model.config["decoder"]["num_hidden_layers"]
        self.self_attn_vmem = VirtualMemory(
            max_num_pages=1024,
            num_kv_heads=num_kv_heads,
            page_size=8,
            head_dim=head_dim,
            num_layers=num_layers,
            type="paged",
        )
        self.cross_attn_vmem = VirtualMemory(
            max_num_pages=1024,
            num_kv_heads=num_kv_heads,
            page_size=8,
            head_dim=head_dim,
            num_layers=num_layers,
            type="paged",
        )
        self.topk_processor = transformers.TopKLogitsWarper(top_k=50)
        self.num_codebooks = self.model.config["decoder"]["num_codebooks"]
        self.bos_token_id = self.model.config["decoder"]["bos_token_id"]
        self.eos_token_id = self.model.config["decoder"]["eos_token_id"]
        self.running_requests = {}
        self._pending_audio_decode = {}
        self.debug_nan = bool(int(os.environ.get("TTS_DEBUG_NAN", "0")))
        # Per-step profiler (TTS_PROFILE=1). Timing only, no logic change.
        # Answers launch/host-bound vs compute-bound -> whether CUDA graphs help.
        self.profile = bool(int(os.environ.get("TTS_PROFILE", "0")))
        self._prof_n = 0
        self._prof_wall = 0.0
        self._prof_events = []  # (whole_start, whole_end, dec_end, smp_end, batch)
        self._prof_window = int(os.environ.get("TTS_PROFILE_WINDOW", "100"))
        dac_cfg = self.model.dac.config
        hop = math.floor(dac_cfg.sampling_rate / dac_cfg.frame_rate)
        print(
            f"[tts] DAC sampling_rate={dac_cfg.sampling_rate} "
            f"frame_rate={dac_cfg.frame_rate} (realtime target={dac_cfg.frame_rate} tok/s/stream)",
            flush=True,
        )
        self._audio_stride = max(0, hop * (play_steps - self.num_codebooks) // 6)

    def _maybe_quantize(self):
        """Weight-only quantization of the per-step decode weights (TTS_QUANT).

        Decode is memory-bandwidth-bound (profiler: step time flat vs batch,
        busy~100%), so halving the bytes read per step ~= proportional speedup.
        Scoped to decoder_layers + lm_heads (what every decode step reads);
        the T5 encoder / embeddings (prefill-only) are left in fp16.

        Off by default. Any failure -> log + keep the fp16 model unchanged.
        """
        quant = os.environ.get("TTS_QUANT", "").strip().lower()
        if quant in ("", "0", "none", "off", "fp16"):
            return
        try:
            from torchao.quantization import (
                quantize_,
                int8_weight_only,
                float8_weight_only,
            )

            if quant == "int8":
                q = int8_weight_only()
            elif quant == "fp8":
                q = float8_weight_only()
            else:
                print(f"[tts] TTS_QUANT={quant!r} unknown; running fp16", flush=True)
                return
            quantize_(self.model.decoder_layers, q)
            quantize_(self.model.lm_heads, q)
            print(f"[tts] applied {quant} weight-only quant to decode path", flush=True)
        except Exception as e:  # never break the working fp16 path
            print(f"[tts] TTS_QUANT={quant} failed ({e!r}); running fp16", flush=True)

    def _stacked_audio_codes_from_timeline(self, audio_tokens):
        # Strip delay/boundary framing; need T = L - num_codebooks - 1 >= 1 for DAC.
        if audio_tokens.shape[1] < self.num_codebooks + 2:
            return None
        rows = [
            audio_tokens[cb, cb + 1 : -self.num_codebooks + cb]
            for cb in range(self.num_codebooks)
        ]
        return torch.stack(rows).unsqueeze(0)

    def _audio_numpy_from_token_cache(self, token_cache):
        if len(token_cache) == 0:
            return None
        audio_tokens = torch.cat(token_cache, dim=-1)
        audio_tokens_fixed = self._stacked_audio_codes_from_timeline(audio_tokens)
        if audio_tokens_fixed is None:
            return None
        return self.decode_audio_parts([audio_tokens_fixed])[0]

    def _safe_probs(self, scores):
        """``scores`` (..., vocab) -> ``probs`` (..., vocab) that is guaranteed
        finite, non-negative, and positive-summing per row, so ``torch.multinomial``
        can never hit the "inf, nan or element < 0" CUDA device-side assert.

        1. NaN and +inf -> -inf. NaN is the value named in the crash; ``torch.isinf``
           alone (as in some proposed fixes) does NOT detect NaN and would let it
           reach multinomial.
        2. Any row with no finite candidate left (all -inf) would softmax to NaN, so
           collapse it to an EOS one-hot -- this cleanly terminates just that request
           instead of poisoning the shared batch.
        3. A final no-op-if-clean scrub of the probabilities as insurance.

        Everything is elementwise / masked -> no ``.item()``/``.any()`` branch and
        therefore no per-step GPU->CPU synchronization on the hot path.

        Returns ``(probs, dead)`` where ``dead`` has the leading (row) shape.
        """
        neg_inf = torch.tensor(-math.inf, device=scores.device, dtype=scores.dtype)
        # 1. NaN / +inf -> -inf
        scores = torch.where(
            torch.isnan(scores) | torch.isposinf(scores), neg_inf, scores
        )
        # 2. dead rows (no finite candidate) -> EOS one-hot
        dead = ~torch.isfinite(scores).any(dim=-1, keepdim=True)  # (..., 1)
        is_eos = torch.arange(scores.shape[-1], device=scores.device) == self.eos_token_id
        eos_row = torch.where(is_eos, torch.zeros_like(scores), neg_inf.expand_as(scores))
        scores = torch.where(dead, eos_row, scores)
        probs = torch.softmax(scores, dim=-1)
        # 3. insurance: scrub any residual non-finite / negative (branch-free)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return probs, dead.squeeze(-1)

    def prefill(self, request):

        encoder_hidden_states, prompt_hidden_states = self.model.encode(
            [request.prompt], [request.description]
        )
        decoder_input_ids = torch.full(
            (self.num_codebooks, 1), self.bos_token_id, dtype=torch.int32, device=device
        )
        decoder_position_ids = torch.arange(
            prompt_hidden_states.shape[1] + 1, dtype=torch.int32, device=device
        ).unsqueeze(0)

        request.decoder_input_ids.append(decoder_input_ids)
        request.token_cache.append(decoder_input_ids)

        request.decoder_position_ids.append(decoder_position_ids)

        logits, model_kv_cache, model_encoder_kv_cache = self.model.prefill(
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            prompt_hidden_states=prompt_hidden_states,
        )
        self.self_attn_vmem.prefill(pid=request.pid, model_kv_cache=model_kv_cache)
        self.cross_attn_vmem.prefill(
            pid=request.pid, model_kv_cache=model_encoder_kv_cache
        )
        next_decoder_input_ids = self._sample_prefill(request, logits)
        next_decoder_position_ids = decoder_position_ids[:, -1:] + 1
        request.decoder_input_ids.append(next_decoder_input_ids)
        request.decoder_position_ids.append(next_decoder_position_ids)
        request.token_cache.append(next_decoder_input_ids)

        self.running_requests[request.pid] = request

    def _sample_prefill(self, request, logits, sampling="multinomial"):
        if sampling == "argmax":
            sampled_tokens = logits.argmax(dim=-1)[0, :, -1:]
        else:
            scores = logits[0, :, -1]  # (num_codebooks, vocab)
            scores = self.topk_processor(input_ids=None, scores=scores)
            probs, _ = self._safe_probs(scores)
            sampled_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), 1
            ).view(scores.size(0), 1)

        mask = torch.arange(self.num_codebooks) < len(request.decoder_input_ids)
        next_decoder_input_ids = torch.where(
            mask.to(device), sampled_tokens.squeeze(), self.bos_token_id
        ).unsqueeze(-1)
        return next_decoder_input_ids

    def step(self):
        sorted_pids = sorted(self.running_requests.keys())
        if len(sorted_pids) == 0:
            return

        prof = self.profile
        if prof:
            e_start = torch.cuda.Event(enable_timing=True)
            e_dec = torch.cuda.Event(enable_timing=True)
            e_smp = torch.cuda.Event(enable_timing=True)
            t_wall0 = time.perf_counter()
            e_start.record()

        decoder_input_ids = torch.cat(
            [self.running_requests[pid].decoder_input_ids[-1] for pid in sorted_pids],
            dim=0,
        )
        decoder_position_ids = torch.cat(
            [
                self.running_requests[pid].decoder_position_ids[-1]
                for pid in sorted_pids
            ],
            dim=0,
        )
        logits = self.model.decode(
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            model_kv_cache_vmem=self.self_attn_vmem,
            model_encoder_kv_cache_vmem=self.cross_attn_vmem,
        )
        if prof:
            e_dec.record()

        next_decoder_position_ids = decoder_position_ids[:, -1:] + 1
        next_decoder_input_ids = self._sample_decode(logits=logits)

        if prof:
            e_smp.record()
            self._prof_wall += time.perf_counter() - t_wall0
            self._prof_events.append((e_start, e_dec, e_smp, len(sorted_pids)))
            self._prof_n += 1
            if self._prof_n % self._prof_window == 0:
                self._prof_flush()

        for bid, pid in enumerate(sorted_pids):
            self.running_requests[pid].decoder_input_ids.append(
                next_decoder_input_ids[bid]
            )
            self.running_requests[pid].token_cache.append(
                next_decoder_input_ids[bid]
            )
            self.running_requests[pid].decoder_position_ids.append(
                next_decoder_position_ids[bid].unsqueeze(0)
            )

    def _sample_decode(self, logits, sampling="multinomial"):
        sorted_pids = sorted(self.running_requests.keys())
        if sampling == "argmax":
            sampled_tokens = logits.argmax(dim=-1)
        else:
            scores = logits[:, :, 0]  # (batch, num_codebooks, vocab)
            stacked_decoder_input_ids = torch.stack(
                [
                    self.running_requests[pid].decoder_input_ids[-1][:, 0]
                    for pid in sorted_pids
                ],
                dim=0,
            )
            # find number of eos per batch in input ids
            eos_num = (stacked_decoder_input_ids == self.eos_token_id).sum(dim=1)
            eos_token_mask = torch.arange(self.num_codebooks, device=device).unsqueeze(
                0
            ) > eos_num.unsqueeze(1)
            eos_scores = scores[:, :, self.eos_token_id]
            eos_scores[eos_token_mask] = -math.inf

            scores = self.topk_processor(input_ids=None, scores=scores)

            probs, dead = self._safe_probs(scores)

            if self.debug_nan and bool(dead.any()):
                dead_cpu = dead.detach().cpu()
                for bid, pid in enumerate(sorted_pids):
                    if bool(dead_cpu[bid].any()):
                        print(
                            f"[warn] non-finite/all-masked scores for pid={pid}; "
                            f"forcing EOS on affected codebooks"
                        )

            sampled_tokens = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(scores.shape[:2])

            # set eos token forcibly, but only if eos_num.max() > 0:
            eos_token_mask[eos_num == 0] = True
            sampled_tokens[~eos_token_mask] = self.eos_token_id

        # set bos mask
        current_seq_lens = torch.tensor(
            [len(self.running_requests[pid].decoder_input_ids) for pid in sorted_pids],
            dtype=torch.int32,
            device=device,
        )
        bos_token_mask = torch.arange(self.num_codebooks, device=device).unsqueeze(
            0
        ) >= current_seq_lens.unsqueeze(1)
        sampled_tokens[bos_token_mask] = self.bos_token_id
        return sampled_tokens.unsqueeze(-1)

    def _prof_flush(self):
        # Single sync per window (not per step) so profiling itself stays cheap.
        torch.cuda.synchronize()
        n = len(self._prof_events)
        if n == 0:
            return
        gpu_ms = sum(s.elapsed_time(smp) for s, _, smp, _ in self._prof_events)
        dec_ms = sum(s.elapsed_time(d) for s, d, _, _ in self._prof_events)
        smp_ms = sum(d.elapsed_time(smp) for _, d, smp, _ in self._prof_events)
        wall_ms = self._prof_wall * 1000.0
        avg_b = sum(b for _, _, _, b in self._prof_events) / n
        busy = 100.0 * gpu_ms / wall_ms if wall_ms > 0 else 0.0
        verdict = "COMPUTE-bound (graph ~no help)" if busy > 85 else "LAUNCH/HOST-bound (CUDA graph WILL help)"
        print(
            f"[prof pid={os.getpid()}] steps={n} avg_batch={avg_b:.1f} "
            f"wall={wall_ms/n:.3f}ms/step gpu={gpu_ms/n:.3f}ms/step "
            f"(decode={dec_ms/n:.3f} sample={smp_ms/n:.3f}) busy={busy:.0f}% -> {verdict}",
            flush=True,
        )
        self._prof_events.clear()
        self._prof_wall = 0.0

    def check_stopping_criteria(self):
        sorted_pids = sorted(self.running_requests.keys())
        if not sorted_pids:
            return
        stacked = torch.stack(
            [self.running_requests[pid].decoder_input_ids[-1] for pid in sorted_pids],
            dim=0,
        )  # (batch, num_codebooks, 1)
        stop_flags = (
            (stacked == self.eos_token_id).reshape(len(sorted_pids), -1).all(dim=1).tolist()
        )
        for pid, to_stop in zip(sorted_pids, stop_flags):
            if to_stop:
                self.evict(self.running_requests[pid])

    def free(self, request):
        self.self_attn_vmem.free(request.pid)
        self.cross_attn_vmem.free(request.pid)

    def evict(self, request):
        audio = self._audio_numpy_from_token_cache(request.token_cache)
        if audio is not None:
            tail = audio[request.audio_to_yield :]
            if tail.size:
                self._pending_audio_decode[request.pid] = tail
        del self.running_requests[request.pid]
        self.free(request)

    def decode_audio_parts(self, list_of_audio_ids):
        audio_ids_e = torch.cat(list_of_audio_ids, -1)
        audio = self.model.dac.decode(audio_codes=audio_ids_e)[0]
        audio_arr = audio[0].detach().cpu().numpy().astype("float")
        token_counts = [a.shape[-1] for a in list_of_audio_ids]
        total_tokens = sum(token_counts)
        total_samples = audio_arr.shape[-1]
        cumulative = 0
        split_indices = []
        for count in token_counts[:-1]:
            cumulative += count
            split_indices.append(int(total_samples * cumulative / total_tokens))
        return np.split(audio_arr, split_indices, axis=-1)

    def audio_decode(self):
        audio_dict = dict(self._pending_audio_decode)
        self._pending_audio_decode.clear()
        sorted_pids = sorted(self.running_requests.keys())
        list_of_audio_tokens = []
        decoded_pids = []
        for pid in sorted_pids:
            token_cache = self.running_requests[pid].token_cache
            if len(token_cache) == 0:
                continue

            audio_tokens = torch.cat(token_cache, dim=-1)
            audio_tokens_fixed = self._stacked_audio_codes_from_timeline(audio_tokens)
            if audio_tokens_fixed is None:
                continue
            list_of_audio_tokens.append(audio_tokens_fixed)
            decoded_pids.append(pid)

        if len(list_of_audio_tokens) == 0:
            return audio_dict
        self.list_of_audio_tokens = list_of_audio_tokens
        audio_arrays = self.decode_audio_parts(list_of_audio_tokens)
        S = self._audio_stride
        for pid, audio_arr in zip(decoded_pids, audio_arrays):
            req = self.running_requests[pid]
            t0 = req.audio_to_yield
            if S > 0 and len(audio_arr) > t0 + S:
                req.audio_to_yield = len(audio_arr) - S
                audio_dict[pid] = audio_arr[t0:-S]
            elif S == 0 and len(audio_arr) > t0:
                req.audio_to_yield = len(audio_arr)
                audio_dict[pid] = audio_arr[t0:]
        return audio_dict