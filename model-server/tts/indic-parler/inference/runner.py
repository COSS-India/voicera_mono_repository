import math
import uuid

import numpy as np
import torch
import transformers
from inference.config import device
from inference.modeling import ParlerTTS
from inference.paging import VirtualMemory


class TTSRequest:
    def __init__(self, prompt, description, pid=None):
        self.pid = uuid.uuid4().hex[:6] if pid is None else pid
        self.prompt = prompt
        self.description = description
        self.decoder_input_ids = []
        self.decoder_position_ids = []
        self.token_cache = []
        self.audio_to_yield = 0
        self.finished = False
        # Code-frame cursor for incremental DAC (avoids full-history re-decode).
        self.dac_code_cursor = 0

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
    def __init__(self, checkpoint_path, play_steps=60, use_cuda_graph=True):
        self.model = ParlerTTS(checkpoint_path).eval().to(device)
        num_kv_heads = self.model.config["text_encoder"]["num_heads"]
        head_dim = self.model.config["decoder"]["hidden_size"] // num_kv_heads
        num_layers = self.model.config["decoder"]["num_hidden_layers"]
        # Dense SDPA KV; CUDA graphs enabled once seq-bucket capture is stable.
        self.self_attn_vmem = VirtualMemory(
            max_num_pages=1024,
            num_kv_heads=num_kv_heads,
            page_size=8,
            head_dim=head_dim,
            num_layers=num_layers,
            type="dense",
            max_seq_len=768*3,
            max_batch_size=24,
        )
        self.cross_attn_vmem = VirtualMemory(
            max_num_pages=1024,
            num_kv_heads=num_kv_heads,
            page_size=8,
            head_dim=head_dim,
            num_layers=num_layers,
            type="dense",
            max_seq_len=128*3,
            max_batch_size=24,
        )
        self.topk_processor = transformers.TopKLogitsWarper(top_k=50)
        self.num_codebooks = self.model.config["decoder"]["num_codebooks"]
        self.bos_token_id = self.model.config["decoder"]["bos_token_id"]
        self.eos_token_id = self.model.config["decoder"]["eos_token_id"]
        self.running_requests = {}
        self._pending_audio_decode = {}
        # Final utterances queued on evict; DAC runs in audio_decode() (not on step path).
        self._pending_final_tokens = {}
        dac_cfg = self.model.dac.config
        hop = math.floor(dac_cfg.sampling_rate / dac_cfg.frame_rate)
        print(dac_cfg.sampling_rate, dac_cfg.frame_rate)
        self._dac_hop = hop
        # Short lookback is enough for DAC continuity; keeps periodic windows ~O(play_steps).
        self._dac_context_frames = 8
        # Cap live DAC per tick (RR). Full-24 each tick is ~100ms+ and fights KV VRAM.
        # Prefer advancing yield for a few streams so EOS tails stay short.
        # ~8 live/tick ⇒ each of 24 streams gets DAC ~2× before typical EOS (keeps finals short).
        self._dac_max_live_per_tick = 8
        self._dac_max_finals_per_tick = 2
        self._dac_micro_batch = 1
        self._audio_stride = max(0, hop * (play_steps - self.num_codebooks) // 6)
        self._dac_compiled = False
        self.use_cuda_graph = bool(use_cuda_graph) and device.type == "cuda"
        self._cuda_graphs = {}
        self._cuda_graph = None
        self._cuda_graph_bs = None
        self._cuda_graph_bucket = None
        self._cuda_graph_retired = False
        self._cg_hold_key = None
        self._cg_hold_steps = 0
        self._session_peak_bs = 0
        # Fast capture while batch is stable/growing; avoid recapture storms while draining.
        self._cg_capture_after_grow = 2
        self._cg_capture_after_shrink = 6
        self._cg_input_ids = None
        self._cg_position_ids = None
        self._cg_logits = None
        self._cg_self_updater = None
        self._cg_self_attn = None
        self._cg_cross_attn = None

    def _ordered_pids(self):
        # Match dense KV slot order (insertion / slot index), not lexical pid order.
        return sorted(
            self.running_requests.keys(),
            key=lambda p: self.self_attn_vmem.pid_to_slot[p],
        )

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
        # Drop any post-EOS padding frames (wave-end batch padding).
        trimmed = []
        for t in token_cache:
            trimmed.append(t)
            if bool(torch.all(t == self.eos_token_id).item()):
                break
        audio_tokens = torch.cat(trimmed, dim=-1)
        audio_tokens_fixed = self._stacked_audio_codes_from_timeline(audio_tokens)
        if audio_tokens_fixed is None:
            return None
        return self.decode_audio_parts([audio_tokens_fixed])[0]

    def prefill(self, request):
        if len(self.running_requests) == 0:
            # New generation wave — allow CUDA graph capture again.
            self._cuda_graph_retired = False
            self._session_peak_bs = 0
            self._invalidate_cuda_graph()

        self.running_requests[request.pid] = request

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

    def _sample_prefill(self, request, logits, sampling="multinomial"):
        if sampling == "argmax":
            sampled_tokens = logits.argmax(dim=-1)[0, :, -1:]
        else:
            scores = logits[0, :, -1]
            scores = self.topk_processor(input_ids=None, scores=scores)
            sampled_tokens = torch.multinomial(
                torch.softmax(scores, dim=-1).view(-1, scores.size(-1)), 1
            ).view(scores.size(0), 1)

        mask = torch.arange(self.num_codebooks) < len(request.decoder_input_ids)
        next_decoder_input_ids = torch.where(
            mask.to(device), sampled_tokens.squeeze(), self.bos_token_id
        ).unsqueeze(-1)
        return next_decoder_input_ids

    def step(self):
        sorted_pids = self._ordered_pids()
        if len(sorted_pids) == 0:
            return

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
        if self.use_cuda_graph:
            logits = self._decode_with_cuda_graph(
                decoder_input_ids, decoder_position_ids
            )
        else:
            logits = self.model.decode(
                decoder_input_ids=decoder_input_ids,
                decoder_position_ids=decoder_position_ids,
                model_kv_cache_vmem=self.self_attn_vmem,
                model_encoder_kv_cache_vmem=self.cross_attn_vmem,
            )

        next_decoder_position_ids = decoder_position_ids[:, -1:] + 1
        next_decoder_input_ids = self._sample_decode(
            logits=logits, sorted_pids=sorted_pids
        )

        for bid, pid in enumerate(sorted_pids):
            req = self.running_requests[pid]
            tok = next_decoder_input_ids[bid]
            req.decoder_input_ids.append(tok)
            req.token_cache.append(tok)
            req.decoder_position_ids.append(
                next_decoder_position_ids[bid].unsqueeze(0)
            )
            # Compact token_cache occasionally so DAC prep isn't O(steps) cats.
            if len(req.token_cache) >= 64:
                req.token_cache = [torch.cat(req.token_cache, dim=-1)]

    def _invalidate_cuda_graph(self):
        self._cuda_graphs = {}
        self._cuda_graph = None
        self._cuda_graph_bs = None
        self._cuda_graph_bucket = None
        self._cg_hold_key = None
        self._cg_hold_steps = 0
        self._session_peak_bs = 0
        self._cg_input_ids = None
        self._cg_position_ids = None
        self._cg_logits = None
        self._cg_self_updater = None
        self._cg_self_attn = None
        self._cg_cross_attn = None

    def _retire_cuda_graph(self):
        """Drop captured graphs; continue in eager until a new stable bs is captured."""
        self._invalidate_cuda_graph()
        try:
            self.self_attn_vmem.disable_cuda_graph()
            self.cross_attn_vmem.disable_cuda_graph()
        except Exception:
            pass
        # Do not permanently disable — allow re-capture if bs stabilizes.

    @staticmethod
    def _seq_bucket(seq_len, max_seq_len):
        # Monotonic-ish coarse buckets: fewer CUDA-graph recaptures than 128/256/512/768.
        # Early 128 keeps first ~1s fast; then jump to 512 to avoid a mid-run 256 capture.
        if seq_len <= 128:
            return min(128, max_seq_len)
        if seq_len <= 512:
            return min(512, max_seq_len)
        return max_seq_len

    def _decode_with_cuda_graph(self, decoder_input_ids, decoder_position_ids):
        bs = decoder_input_ids.shape[0] // self.num_codebooks

        # Holes after eviction: run eager (index_select path) until batch is contiguous.
        slots = [
            self.self_attn_vmem.pid_to_slot[p] for p in self._ordered_pids()
        ]
        contiguous = slots == list(range(bs))
        if not contiguous:
            self._cuda_graph = None
            self._cuda_graph_bs = None
            self._cg_hold_key = None
            self._cg_hold_steps = 0
            self.self_attn_vmem.disable_cuda_graph()
            self.cross_attn_vmem.disable_cuda_graph()
            return self.model.decode(
                decoder_input_ids=decoder_input_ids,
                decoder_position_ids=decoder_position_ids,
                model_kv_cache_vmem=self.self_attn_vmem,
                model_encoder_kv_cache_vmem=self.cross_attn_vmem,
            )

        live_before = self.self_attn_vmem.max_host_seq_len(bs) + 1
        bucket = self._seq_bucket(live_before, self.self_attn_vmem.max_seq_len)
        cross_bucket = self._seq_bucket(
            max(self.cross_attn_vmem.max_host_seq_len(bs), 1),
            self.cross_attn_vmem.max_seq_len,
        )
        key = (bs, bucket, cross_bucket)
        self._session_peak_bs = max(self._session_peak_bs, bs)

        if key != self._cg_hold_key:
            self._cg_hold_key = key
            self._cg_hold_steps = 1
        else:
            self._cg_hold_steps += 1

        entry = self._cuda_graphs.get(key)
        # Reuse an already-captured graph immediately.
        if entry is not None:
            self.self_attn_vmem.enable_cuda_graph(bs)
            self.cross_attn_vmem.enable_cuda_graph(bs)
            self._cuda_graph_bs = bs
            # Side-effect grow for captured closures' write-position buffers.
            self.self_attn_vmem.get_decode_closures(grow=True, attn_len=bucket)
            self.cross_attn_vmem.get_decode_closures(grow=False, attn_len=cross_bucket)
            entry["input_ids"].copy_(decoder_input_ids)
            entry["pos_ids"].copy_(decoder_position_ids)
            entry["graph"].replay()
            return entry["logits"]

        shrinking = bs < self._session_peak_bs
        need = (
            self._cg_capture_after_shrink if shrinking else self._cg_capture_after_grow
        )
        if self._cg_hold_steps < need:
            self.self_attn_vmem.disable_cuda_graph()
            self.cross_attn_vmem.disable_cuda_graph()
            return self.model.decode(
                decoder_input_ids=decoder_input_ids,
                decoder_position_ids=decoder_position_ids,
                model_kv_cache_vmem=self.self_attn_vmem,
                model_encoder_kv_cache_vmem=self.cross_attn_vmem,
            )

        # Stable — capture once for this (bs, bucket).
        self._cuda_graphs.clear()
        self.self_attn_vmem.enable_cuda_graph(bs)
        self.cross_attn_vmem.enable_cuda_graph(bs)
        self._cuda_graph_bs = bs

        self_updater, self_attn = self.self_attn_vmem.get_decode_closures(
            grow=True, attn_len=bucket
        )
        _, cross_attn = self.cross_attn_vmem.get_decode_closures(
            grow=False, attn_len=cross_bucket
        )

        static_ids = decoder_input_ids.clone()
        static_pos = decoder_position_ids.clone()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            logits = self.model.decode_forward(
                static_ids, static_pos, self_updater, self_attn, cross_attn
            )
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits = self.model.decode_forward(
                static_ids, static_pos, self_updater, self_attn, cross_attn
            )
        g.replay()
        entry = {
            "graph": g,
            "input_ids": static_ids,
            "pos_ids": static_pos,
            "logits": logits,
        }
        self._cuda_graphs[key] = entry
        self._cuda_graph = g
        self._cuda_graph_bucket = key
        return logits

    def _sample_decode(self, logits, sampling="multinomial", sorted_pids=None):
        if sorted_pids is None:
            sorted_pids = self._ordered_pids()
        if sampling == "argmax":
            # logits: (bs, codebooks, seq, vocab) — take last seq position
            sampled_tokens = logits[:, :, -1, :].argmax(dim=-1)
        else:
            scores = logits[:, :, 0]
            stacked_decoder_input_ids = torch.stack(
                [
                    self.running_requests[pid].decoder_input_ids[-1][:, 0]
                    for pid in sorted_pids
                ],
                dim=0,
            )
            # find number of eos per batch in input ids
            eos_num = (stacked_decoder_input_ids == self.eos_token_id).sum(dim=1)
            # do not allow eos token for eos_num + 1 to rest of codebooks
            eos_token_mask = torch.arange(self.num_codebooks, device=device).unsqueeze(
                0
            ) > eos_num.unsqueeze(1)
            scores[eos_token_mask, self.eos_token_id] = -math.inf

            # get samples from scores now
            scores = self.topk_processor(input_ids=None, scores=scores)
            sampled_tokens = torch.multinomial(
                torch.softmax(scores, dim=-1).view(-1, scores.shape[-1]), num_samples=1
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

    def check_stopping_criteria(self):
        """Evict finished requests immediately (continuous batching / streaming)."""
        to_evict = []
        max_len = self.self_attn_vmem.max_seq_len
        for pid in self._ordered_pids():
            req = self.running_requests[pid]
            decoder_input_ids = req.decoder_input_ids[-1]
            if bool(torch.all(decoder_input_ids == self.eos_token_id).item()):
                to_evict.append(req)
                continue
            slot = self.self_attn_vmem.pid_to_slot[pid]
            if self.self_attn_vmem._host_seq_lens[slot] >= max_len - 2:
                to_evict.append(req)
        if not to_evict:
            return
        # Batch free: compact once after all frees to avoid repeated KV moves.
        # Defer DAC to audio_decode() so check_stopping stays off the step hot path.
        for i, req in enumerate(to_evict):
            is_last = i == len(to_evict) - 1
            token_cache = req.token_cache
            audio_to_yield = req.audio_to_yield
            pid = req.pid
            del self.running_requests[pid]
            self.free(req, compact=is_last)
            self._pending_final_tokens[pid] = (token_cache, audio_to_yield)

    def free(self, request, compact=True):
        self.self_attn_vmem.free(request.pid, compact=compact)
        self.cross_attn_vmem.free(request.pid, compact=compact)
        # Drop active graph pointer; keep hysteresis so we don't recapture every eviction.
        self._cuda_graph = None
        self._cuda_graph_bs = None
        self._cuda_graphs.clear()
        self._cg_hold_key = None
        self._cg_hold_steps = 0
        try:
            self.self_attn_vmem.disable_cuda_graph()
            self.cross_attn_vmem.disable_cuda_graph()
        except Exception:
            pass

    def evict(self, request):
        token_cache = request.token_cache
        audio_to_yield = request.audio_to_yield
        pid = request.pid
        del self.running_requests[pid]
        self.free(request, compact=True)
        self._pending_final_tokens[pid] = (token_cache, audio_to_yield)

    def _flush_pending_final_tokens(self):
        """DAC any utterances queued at evict time."""
        out = {}
        pending = self._pending_final_tokens
        self._pending_final_tokens = {}
        for pid, (token_cache, audio_to_yield) in pending.items():
            try:
                audio = self._audio_numpy_from_token_cache(token_cache)
            except Exception:
                torch.cuda.empty_cache()
                try:
                    audio = self._audio_numpy_from_token_cache(token_cache)
                except Exception:
                    audio = None
            if audio is not None:
                tail = audio[audio_to_yield:]
                if tail.size:
                    out[pid] = tail
        return out

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

    def _prepare_audio_decode_inputs(self):
        """Snapshot codes for DAC without running it (safe while stepping continues).

        Live requests use *incremental* code windows (lookback ``_dac_context_frames``)
        so DAC cost stays roughly O(decode_every) instead of O(total_steps).
        """
        seed = dict(self._pending_audio_decode)
        self._pending_audio_decode.clear()

        hop = self._dac_hop
        ctx = self._dac_context_frames
        S = self._audio_stride

        live_codes = []
        live_meta = []  # (pid, code_start, skip_samples, stride)
        # Round-robin subset of live pids to keep periodic DAC bounded at high BS.
        live_pids = self._ordered_pids()
        if not hasattr(self, "_dac_rr") or self._dac_rr >= len(live_pids):
            self._dac_rr = 0
        max_live = min(len(live_pids), int(self._dac_max_live_per_tick))
        if live_pids:
            ordered = live_pids[self._dac_rr :] + live_pids[: self._dac_rr]
            selected = ordered[:max_live]
            self._dac_rr = (self._dac_rr + max_live) % max(1, len(live_pids))
        else:
            selected = []

        for pid in selected:
            req = self.running_requests[pid]
            if len(req.token_cache) == 0:
                continue
            audio_tokens = torch.cat(req.token_cache, dim=-1)
            fixed = self._stacked_audio_codes_from_timeline(audio_tokens)
            if fixed is None:
                continue
            n_codes = int(fixed.shape[-1])
            # Only need codes covering new audio past audio_to_yield, plus lookback.
            need_from = max(0, int(req.audio_to_yield) // hop - ctx)
            if n_codes <= need_from:
                continue
            # If almost nothing new beyond stride holdback, skip.
            approx_full_samples = n_codes * hop
            if S > 0 and approx_full_samples <= req.audio_to_yield + S:
                continue
            codes_slice = fixed[:, :, need_from:].detach().contiguous().clone()
            skip_samples = max(0, int(req.audio_to_yield) - need_from * hop)
            live_codes.append(codes_slice)
            live_meta.append((pid, need_from, skip_samples, S))

        # Cap finals per tick; leave the rest queued so a wave of EOS doesn't stall steps.
        pending_finals = self._pending_final_tokens
        final_items = list(pending_finals.items())
        take_n = min(len(final_items), int(self._dac_max_finals_per_tick))
        take = final_items[:take_n]
        self._pending_final_tokens = dict(final_items[take_n:])

        final_codes = []
        # (pid, skip_samples) — incremental tail decode, not full history.
        final_meta = []
        for pid, (token_cache, audio_to_yield) in take:
            if len(token_cache) == 0:
                continue
            trimmed = []
            for t in token_cache:
                trimmed.append(t)
                if bool(torch.all(t == self.eos_token_id).item()):
                    break
            audio_tokens = torch.cat(trimmed, dim=-1)
            fixed = self._stacked_audio_codes_from_timeline(audio_tokens)
            if fixed is None:
                continue
            n_codes = int(fixed.shape[-1])
            need_from = max(0, int(audio_to_yield) // hop - ctx)
            if n_codes <= need_from:
                # Nothing left to emit; drop.
                continue
            codes_slice = fixed[:, :, need_from:].detach().contiguous().clone()
            skip_samples = max(0, int(audio_to_yield) - need_from * hop)
            final_codes.append(codes_slice)
            final_meta.append((pid, skip_samples))

        return {
            "seed": seed,
            "live_codes": live_codes,
            "live_meta": live_meta,
            "final_codes": final_codes,
            "final_meta": final_meta,
            "hop": hop,
        }

    def start_audio_decode_async(self, stream):
        """
        Launch DAC on ``stream`` without blocking the default decode stream.
        Returns a handle for ``try_finish_audio_decode_async``, or None if nothing to do.
        """
        snap = self._prepare_audio_decode_inputs()
        if (
            not snap["seed"]
            and not snap["live_codes"]
            and not snap["final_codes"]
        ):
            return None

        all_codes = snap["live_codes"] + snap["final_codes"]
        handle = {
            "event": torch.cuda.Event(),
            "snap": snap,
            "gpu_audios": None,
        }
        if not all_codes:
            handle["event"].record(stream)
            return handle

        # DAC stream must see completed clones from the decode stream.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Micro-batch=1: DAC scales poorly in B and fragments VRAM next to graphs.
            gpu_audios = [None] * len(all_codes)
            micro = max(1, int(self._dac_micro_batch))
            for start in range(0, len(all_codes), micro):
                chunk = all_codes[start : start + micro]
                max_t = max(c.shape[-1] for c in chunk)
                n_cb = chunk[0].shape[1]
                batched = chunk[0].new_zeros((len(chunk), n_cb, max_t))
                lengths = []
                for i, codes in enumerate(chunk):
                    t = codes.shape[-1]
                    batched[i, :, :t] = codes[0]
                    lengths.append(t)
                audio_b = self.model.dac.decode(audio_codes=batched)[0]
                if audio_b.dim() == 3:
                    audio_b = audio_b.squeeze(1)
                total_samples = audio_b.shape[-1]
                for i, t in enumerate(lengths):
                    gpu_audios[start + i] = audio_b[
                        i, : max(1, int(total_samples * t / max_t))
                    ].contiguous()
                del audio_b, batched
            handle["gpu_audios"] = gpu_audios
        handle["event"].record(stream)
        return handle

    def try_finish_audio_decode_async(self, handle, block=False):
        """
        If DAC finished (or block=True), return audio_dict and apply yield cursors.
        Returns None if still running and block=False.
        """
        if handle is None:
            return {}
        ev = handle["event"]
        if not block and not ev.query():
            return None
        ev.synchronize()

        snap = handle["snap"]
        audio_dict = dict(snap["seed"])
        hop = snap["hop"]
        gpu_audios = handle.get("gpu_audios")
        if not gpu_audios:
            return audio_dict

        n_live = len(snap["live_meta"])
        for i, (pid, code_start, skip_samples, S) in enumerate(snap["live_meta"]):
            audio_i = gpu_audios[i].detach().float().cpu().numpy()
            if skip_samples >= len(audio_i):
                continue
            usable = audio_i[skip_samples:]
            if S > 0:
                if len(usable) <= S:
                    continue
                chunk = usable[:-S]
                new_yield = code_start * hop + (len(audio_i) - S)
            else:
                chunk = usable
                new_yield = code_start * hop + len(audio_i)
            if chunk.size == 0:
                continue
            audio_dict[pid] = chunk
            req = self.running_requests.get(pid)
            if req is not None:
                req.audio_to_yield = int(new_yield)

        for j, (pid, skip_samples) in enumerate(snap["final_meta"]):
            audio_i = gpu_audios[n_live + j].detach().float().cpu().numpy()
            if skip_samples >= len(audio_i):
                continue
            tail = audio_i[skip_samples:]
            if tail.size:
                audio_dict[pid] = tail
        return audio_dict

    def audio_decode(self):
        """Synchronous DAC. Drains remaining finals if callers keep invoking after EOS."""
        if device.type != "cuda":
            # Fall back to original sync path on CPU.
            audio_dict = dict(self._pending_audio_decode)
            self._pending_audio_decode.clear()
            audio_dict.update(self._flush_pending_final_tokens())
            sorted_pids = self._ordered_pids()
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
            if not list_of_audio_tokens:
                return audio_dict
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

        if not hasattr(self, "_dac_stream") or self._dac_stream is None:
            self._dac_stream = torch.cuda.Stream()
        out = {}
        # One capped tick while decode is still running. After the batch drains,
        # keep going until pending finals are empty (still incremental / capped).
        drain_all = len(self.running_requests) == 0
        for _ in range(64 if drain_all else 1):
            handle = self.start_audio_decode_async(self._dac_stream)
            if handle is None:
                break
            part = self.try_finish_audio_decode_async(handle, block=True) or {}
            for pid, arr in part.items():
                prev = out.get(pid)
                out[pid] = arr if prev is None else np.concatenate([prev, arr], axis=-1)
            if not self._pending_final_tokens:
                break
            if drain_all and device.type == "cuda":
                torch.cuda.empty_cache()
        return out



