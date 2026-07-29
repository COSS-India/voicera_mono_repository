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
        self.finished = False

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
            max_seq_len=768,
            max_batch_size=48,
        )
        self.cross_attn_vmem = VirtualMemory(
            max_num_pages=1024,
            num_kv_heads=num_kv_heads,
            page_size=8,
            head_dim=head_dim,
            num_layers=num_layers,
            type="dense",
            max_seq_len=128,
            max_batch_size=48,
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
        self._audio_stride = max(0, hop * (play_steps - self.num_codebooks) // 6)
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
        # Coarse enough to limit ~30ms recaptures; fine enough for early-step speed.
        for b in (128, 256, 512, 768):
            if seq_len <= b:
                return min(b, max_seq_len)
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

    def audio_decode(self):
        audio_dict = dict(self._pending_audio_decode)
        self._pending_audio_decode.clear()
        # Final audio for just-evicted requests (server calls this on eviction).
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



