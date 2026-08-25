import torch
from inference.config import device
import math
import flashinfer

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend

    # Short masked decode: MATH keeps SMs busy. Longer buckets: efficient kernels
    # keep step time well under 7ms.
    _SDPA_SHORT = [SDPBackend.MATH]
    _SDPA_LONG = [
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]
    _SDPA_SHORT_LEN = 256
except Exception:  # pragma: no cover
    sdpa_kernel = None
    _SDPA_SHORT = _SDPA_LONG = None
    _SDPA_SHORT_LEN = 256



class PageTable:
    def __init__(self, page_size, max_num_pages=None):
        # dictionary from pid -> pages
        self.page_size = page_size
        self.max_num_pages = max_num_pages

        # pid -> list of pages
        self.pid_page_table = {}
        # pid -> total memory size
        self.pid_mem_sizes = {}

    def allocate(self, pid, mem_size):
        try:
            current_mem_size = self.pid_mem_sizes[pid]
        except KeyError:
            current_mem_size = 0
            self.pid_mem_sizes[pid] = 0
            self.pid_page_table[pid] = []

        new_mem_size = current_mem_size + mem_size
        self.pid_mem_sizes[pid] = new_mem_size

        num_existing_pages = len(self.pid_page_table[pid])
        num_new_pages = math.ceil(new_mem_size / self.page_size)
        num_pages_to_allocate = num_new_pages - num_existing_pages
        if num_pages_to_allocate <= 0:
            return

        all_occupied_pages = {
            p for pages in self.pid_page_table.values() for p in pages
        }
        if len(all_occupied_pages) == 0:
            # no pages used so far
            self.pid_page_table[pid].extend(range(0, num_pages_to_allocate))
            self._check_page_budget()
            return
        else:
            first_occupied_page = min(all_occupied_pages)
            # look for space on the left
            if first_occupied_page > 0:
                if first_occupied_page >= num_pages_to_allocate:
                    self.pid_page_table[pid].extend(range(0, num_pages_to_allocate))
                    self._check_page_budget()
                    return
                else:
                    self.pid_page_table[pid].extend(range(0, first_occupied_page))
                    num_pages_to_allocate = num_pages_to_allocate - first_occupied_page

            # look for gaps
            last_occupied_page = max(all_occupied_pages)
            gaps = sorted(
                set(range(first_occupied_page, last_occupied_page + 1))
                - all_occupied_pages
            )
            if len(gaps) >= num_pages_to_allocate:
                self.pid_page_table[pid].extend(gaps[:num_pages_to_allocate])
                self._check_page_budget()
                return
            else:
                self.pid_page_table[pid].extend(gaps)
                num_pages_to_allocate = num_pages_to_allocate - len(gaps)

            # look for space on the right
            self.pid_page_table[pid].extend(
                range(
                    last_occupied_page + 1,
                    last_occupied_page + 1 + num_pages_to_allocate,
                )
            )
            self._check_page_budget()
            return

    def _check_page_budget(self):
        if self.max_num_pages is None:
            return
        occupied = {p for pages in self.pid_page_table.values() for p in pages}
        if occupied and max(occupied) >= self.max_num_pages:
            raise RuntimeError(
                f"paged KV overflow: need page {max(occupied)} but max_num_pages="
                f"{self.max_num_pages} (occupied={len(occupied)})"
            )

    def free(self, pid):
        del self.pid_page_table[pid]
        del self.pid_mem_sizes[pid]

    def print_state(self):
        print(
            "pid_page_table:", self.pid_page_table, "pid_mem_sizes:", self.pid_mem_sizes
        )

    def convert_to_flashinfer(self):
        sorted_pids = sorted(self.pid_page_table.keys())
        pages = []
        indptr = [0]
        last_page_lens = []
        for pid in sorted_pids:
            pages.extend(self.pid_page_table[pid])
            indptr.append(len(pages))
            mem = self.pid_mem_sizes[pid]
            last = mem % self.page_size
            if mem > 0 and last == 0:
                last = self.page_size
            last_page_lens.append(last)

        # Single H2D copy each; avoid Tensor(list).to(device) chaining.
        page_indices = torch.tensor(pages, dtype=torch.int32, device=device)
        page_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)
        last_page_lens = torch.tensor(last_page_lens, dtype=torch.int32, device=device)
        return page_indices, page_indptr, last_page_lens

    def convert_to_sdpa_ragged_attn_mask(self, max_pages):
        # attn_mask shape -> (num_pids, max_pages * page_size)
        sorted_pids = sorted(self.pid_page_table.keys())
        attn_mask = torch.zeros(
            (len(sorted_pids), max_pages * self.page_size), dtype=torch.bool
        )
        for bid, pid in enumerate(sorted_pids):
            for page in self.pid_page_table[pid]:
                attn_mask[bid, page * self.page_size : (page + 1) * self.page_size] = 1

            # deal with last page carefully
            last_page = self.pid_page_table[pid][-1]
            last_page_len = self.pid_mem_sizes[pid] % self.page_size
            if self.pid_mem_sizes[pid] > 0 and last_page_len == 0:
                last_page_len = self.page_size
            attn_mask[
                bid,
                last_page * self.page_size : last_page * self.page_size + last_page_len,
            ] = 1
            attn_mask[
                bid,
                last_page * self.page_size
                + last_page_len : (last_page + 1) * self.page_size,
            ] = 0

        return attn_mask.to(device)


class VirtualMemoryPaged:
    def __init__(self, max_num_pages, page_size, num_kv_heads, head_dim, num_layers):
        self.max_num_pages = max_num_pages
        self.paged_model_kv_cache = [
            torch.zeros(
                (max_num_pages, 2, page_size, num_kv_heads, head_dim),
                dtype=torch.float16,
                device=device,
            )
            for _ in range(num_layers)
        ]
        self.page_table = PageTable(page_size=page_size, max_num_pages=max_num_pages)
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_kv_heads  # no gqa for now
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_layers = num_layers
        # Invalidate static (cross-attn) decode plans when page table changes.
        self._layout_version = 0
        self._planned_layout_version = -1

        self._workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buffer
        )
        # CUDA-graph mode: fixed-address page metadata buffers (set via enable_cuda_graph).
        self._cg_bs = None

    def enable_cuda_graph(self, batch_size):
        """Switch decode wrapper to FlashInfer CUDA-graph-safe fixed buffers."""
        if self._cg_bs == batch_size:
            return
        self._cg_bs = batch_size
        self._cg_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        self._cg_indices = torch.zeros(
            self.max_num_pages, dtype=torch.int32, device=device
        )
        self._cg_last_page_len = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        self._cg_positions = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self._cg_batch_indices = torch.arange(
            batch_size, dtype=torch.int32, device=device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buffer,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self._cg_indptr,
            paged_kv_indices_buffer=self._cg_indices,
            paged_kv_last_page_len_buffer=self._cg_last_page_len,
        )
        self._planned_layout_version = -1

    def disable_cuda_graph(self):
        """Restore eager FlashInfer wrapper (batch size may change again)."""
        if self._cg_bs is None:
            return
        self._cg_bs = None
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buffer
        )
        self._planned_layout_version = -1

    def _bump_layout(self):
        self._layout_version += 1

    def _plan_decode(self, kv_indptr, kv_indices, kv_last_page_len):
        self.decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            data_type=torch.float16,
        )
        self._planned_layout_version = self._layout_version

    def _copy_into_cg_buffers(self, kv_indptr, kv_indices, kv_last_page_len, positions=None):
        n_idx = kv_indices.numel()
        self._cg_indptr.copy_(kv_indptr)
        self._cg_indices[:n_idx].copy_(kv_indices)
        self._cg_last_page_len.copy_(kv_last_page_len)
        if positions is not None:
            self._cg_positions.copy_(positions)
        return n_idx

    def prefill(self, pid, model_kv_cache):
        n_seq = model_kv_cache[0][0].shape[2]
        assert model_kv_cache[0][0].shape[0] == 1

        self.page_table.allocate(pid=pid, mem_size=n_seq)
        self._bump_layout()
        sorted_pids = sorted(self.page_table.pid_mem_sizes.keys())
        bid = sorted_pids.index(pid)
        batch_indices = torch.full((n_seq,), bid, dtype=torch.int32, device=device)
        positions = torch.arange(n_seq, dtype=torch.int32, device=device)
        kv_indices, kv_indptr, kv_last_page_len = (
            self.page_table.convert_to_flashinfer()
        )

        for layer in range(self.num_layers):
            append_key = model_kv_cache[layer][0][0].transpose(0, 1).contiguous().half()
            append_value = (
                model_kv_cache[layer][1][0].transpose(0, 1).contiguous().half()
            )
            flashinfer.append_paged_kv_cache(
                append_key=append_key,
                append_value=append_value,
                batch_indices=batch_indices,
                positions=positions,
                paged_kv_cache=self.paged_model_kv_cache[layer],
                kv_indices=kv_indices,
                kv_indptr=kv_indptr,
                kv_last_page_len=kv_last_page_len,
            )

    def free(self, pid):
        self.page_table.free(pid)
        self._bump_layout()

    def get_decode_closures(self, grow=True, attn_len=None):
        """
        grow=True  (self-attn): allocate 1 token slot per seq, plan, return updater+attn.
        grow=False (cross-attn): KV is static; plan only when layout changed, attn only.

        In CUDA-graph mode, closures close over fixed ``_cg_*`` buffers so a captured
        graph keeps reading/writing the same addresses after each in-place prepare.
        """
        # attn_len ignored for paged (kept for API parity with dense).
        _ = attn_len
        sorted_pids = sorted(self.page_table.pid_mem_sizes.keys())
        n = len(sorted_pids)
        use_cg = self._cg_bs is not None
        if use_cg and n != self._cg_bs:
            raise RuntimeError(
                f"cuda-graph batch size mismatch: wrapper={self._cg_bs} active={n}"
            )

        if grow:
            for pid in sorted_pids:
                self.page_table.allocate(pid=pid, mem_size=1)
            self._bump_layout()

            positions = torch.tensor(
                [self.page_table.pid_mem_sizes[pid] - 1 for pid in sorted_pids],
                dtype=torch.int32,
                device=device,
            )
            kv_indices, kv_indptr, kv_last_page_len = (
                self.page_table.convert_to_flashinfer()
            )

            if use_cg:
                self._copy_into_cg_buffers(
                    kv_indptr, kv_indices, kv_last_page_len, positions=positions
                )
                n_idx = kv_indices.numel()
                self._plan_decode(
                    self._cg_indptr, self._cg_indices[:n_idx], self._cg_last_page_len
                )
                positions = self._cg_positions
                batch_indices = self._cg_batch_indices
                kv_indices = self._cg_indices
                kv_indptr = self._cg_indptr
                kv_last_page_len = self._cg_last_page_len
            else:
                batch_indices = torch.arange(n, dtype=torch.int32, device=device)
                self._plan_decode(kv_indptr, kv_indices, kv_last_page_len)

            def _cache_updater(layer_id, append_kv):
                num_batches = append_kv[0].shape[0]
                num_seqs = append_kv[0].shape[2]
                assert num_batches == n, "batch size doesn't match active sequences"
                assert num_seqs == 1, "decode step assumes only 1 token decoded per batch"
                append_key = append_kv[0][:, :, 0].contiguous()
                append_value = append_kv[1][:, :, 0].contiguous()
                if append_key.dtype != torch.float16:
                    append_key = append_key.half()
                    append_value = append_value.half()
                flashinfer.append_paged_kv_cache(
                    append_key=append_key,
                    append_value=append_value,
                    batch_indices=batch_indices,
                    positions=positions,
                    paged_kv_cache=self.paged_model_kv_cache[layer_id],
                    kv_indices=kv_indices,
                    kv_indptr=kv_indptr,
                    kv_last_page_len=kv_last_page_len,
                )

        else:
            # Static cross-attn: skip allocate; replan only after prefill/free.
            if self._planned_layout_version != self._layout_version:
                kv_indices, kv_indptr, kv_last_page_len = (
                    self.page_table.convert_to_flashinfer()
                )
                if use_cg:
                    self._copy_into_cg_buffers(
                        kv_indptr, kv_indices, kv_last_page_len
                    )
                    n_idx = kv_indices.numel()
                    self._plan_decode(
                        self._cg_indptr,
                        self._cg_indices[:n_idx],
                        self._cg_last_page_len,
                    )
                else:
                    self._plan_decode(kv_indptr, kv_indices, kv_last_page_len)

            def _cache_updater(layer_id, append_kv):
                raise RuntimeError(
                    "cross-attn KV is static; cache updater must not be called"
                )

        def _attn(layer_id, q):
            num_seqs = q.shape[2]
            assert num_seqs == 1, "decode step assumes only 1 token decoded per batch"
            q = q.squeeze(2)
            return self.decode_wrapper.run(
                q, self.paged_model_kv_cache[layer_id]
            ).unsqueeze(2)

        return _cache_updater, _attn


class VirtualMemorySDPA:
    def __init__(self, max_num_pages, page_size, num_kv_heads, head_dim, num_layers):
        self.pid_kv_cache = {}

    def enable_cuda_graph(self, batch_size):
        raise RuntimeError("CUDA graphs require type='paged' virtual memory")

    def disable_cuda_graph(self):
        pass

    def prefill(self, pid, model_kv_cache):
        assert model_kv_cache[0][0].shape[0] == 1
        self.pid_kv_cache[pid] = model_kv_cache

    def get_decode_closures(self, grow=True, attn_len=None):
        _ = attn_len
        sorted_pids = sorted(self.pid_kv_cache.keys())

        def _cache_updater(layer_id, append_kv):
            if not grow:
                raise RuntimeError(
                    "cross-attn KV is static; cache updater must not be called"
                )
            for bid, pid in enumerate(sorted_pids):
                self.pid_kv_cache[pid][layer_id] = (
                    torch.cat(
                        [
                            self.pid_kv_cache[pid][layer_id][0],
                            append_kv[0][bid].unsqueeze(0),
                        ],
                        dim=2,
                    ),
                    torch.cat(
                        [
                            self.pid_kv_cache[pid][layer_id][1],
                            append_kv[1][bid].unsqueeze(0),
                        ],
                        dim=2,
                    ),
                )

        def _attn(layer_id, q):
            num_seqs = q.shape[2]
            assert num_seqs == 1, "decode step assumes only 1 token decoded per batch"

            keys_list = [
                self.pid_kv_cache[pid][layer_id][0].squeeze(0) for pid in sorted_pids
            ]
            values_list = [
                self.pid_kv_cache[pid][layer_id][1].squeeze(0) for pid in sorted_pids
            ]
            # each: (num_heads, seq_len_i, head_dim)

            seq_lens = [k.shape[1] for k in keys_list]
            max_len = max(seq_lens)
            batch = len(sorted_pids)
            num_heads, _, head_dim = keys_list[0].shape

            # Pad k/v to (batch, num_heads, max_len, head_dim)
            keys_padded = torch.zeros(
                batch, num_heads, max_len, head_dim, device=q.device, dtype=q.dtype
            )
            values_padded = torch.zeros(
                batch, num_heads, max_len, head_dim, device=q.device, dtype=q.dtype
            )
            for i, (k, v, slen) in enumerate(zip(keys_list, values_list, seq_lens)):
                keys_padded[i, :, :slen, :] = k
                values_padded[i, :, :slen, :] = v

            mask = torch.zeros(batch, 1, 1, max_len, device=q.device, dtype=torch.bool)
            for i, slen in enumerate(seq_lens):
                mask[i, :, :, :slen] = True
            additive_mask = torch.zeros(
                batch, 1, 1, max_len, device=q.device, dtype=q.dtype
            )
            additive_mask.masked_fill_(~mask, float("-inf"))

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, keys_padded, values_padded, attn_mask=additive_mask
            )
            return attn_output

        return _cache_updater, _attn

    def free(self, pid):
        del self.pid_kv_cache[pid]


class VirtualMemoryCompare:
    def __init__(self, max_num_pages, page_size, num_kv_heads, head_dim, num_layers):
        self.vm_paged = VirtualMemoryPaged(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
        self.vm_sdpa = VirtualMemorySDPA(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )

    def enable_cuda_graph(self, batch_size):
        self.vm_paged.enable_cuda_graph(batch_size)

    def disable_cuda_graph(self):
        self.vm_paged.disable_cuda_graph()

    def prefill(self, pid, model_kv_cache):
        self.vm_paged.prefill(pid, model_kv_cache)
        self.vm_sdpa.prefill(pid, model_kv_cache)

    def get_decode_closures(self, grow=True, attn_len=None):
        paged_cache_updater, paged_attn = self.vm_paged.get_decode_closures(
            grow=grow, attn_len=attn_len
        )
        sdpa_cache_updater, sdpa_attn = self.vm_sdpa.get_decode_closures(
            grow=grow, attn_len=attn_len
        )

        def _cache_updater(layer_id, append_kv):
            paged_cache_updater(layer_id, append_kv)
            sdpa_cache_updater(layer_id, append_kv)

        def _attn(layer_id, q):
            out_paged = paged_attn(layer_id, q)
            out_sdpa = sdpa_attn(layer_id, q)

            diff = (out_paged - out_sdpa).float()
            max_abs = diff.abs().max()

            if max_abs > 1e-2:
                print(f"max_abs={max_abs:.6g}")
            return out_paged

        return _cache_updater, _attn

    def free(self, pid):
        self.vm_paged.free(pid)
        self.vm_sdpa.free(pid)


class VirtualMemoryDense:
    """
    Contiguous KV cache + SDPA. Fixed shapes so the full decode step can be
    captured in a CUDA graph (unlike FlashInfer on this stack).
    """

    def __init__(
        self,
        max_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        num_layers,
        max_seq_len=1024,
        max_batch_size=32,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.k_cache = torch.zeros(
            num_layers,
            max_batch_size,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        # 0 = attend, -inf = masked
        self.attn_mask = torch.full(
            (max_batch_size, 1, 1, max_seq_len),
            float("-inf"),
            dtype=torch.float16,
            device=device,
        )
        self.pid_to_slot = {}
        self._free_slots = list(range(max_batch_size))
        self._cg_bs = None
        # Persistent buffers closed over by CUDA-graph closures (updated in-place).
        self._write_slots = torch.arange(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._write_positions = torch.zeros(
            max_batch_size, dtype=torch.int64, device=device
        )
        self._active_n = 0
        # Host-side seq lengths — avoid GPU sync (.item) on the hot decode path.
        self._host_seq_lens = [0] * max_batch_size

    def enable_cuda_graph(self, batch_size):
        if batch_size > self.max_batch_size:
            raise RuntimeError(
                f"batch_size {batch_size} > max_batch_size {self.max_batch_size}"
            )
        self._cg_bs = batch_size

    def disable_cuda_graph(self):
        self._cg_bs = None

    def max_host_seq_len(self, n=None):
        if n is None:
            n = len(self.pid_to_slot)
        if n <= 0:
            return 0
        return max(self._host_seq_lens[:n])

    def prefill(self, pid, model_kv_cache):
        assert model_kv_cache[0][0].shape[0] == 1
        if pid in self.pid_to_slot:
            raise RuntimeError(f"pid {pid} already prefilling")
        if not self._free_slots:
            raise RuntimeError("dense KV: no free slots")
        slot = self._free_slots.pop(0)
        self.pid_to_slot[pid] = slot
        n_seq = model_kv_cache[0][0].shape[2]
        if n_seq > self.max_seq_len:
            raise RuntimeError(
                f"prefill len {n_seq} > max_seq_len {self.max_seq_len}"
            )
        for layer in range(self.num_layers):
            self.k_cache[layer, slot, :, :n_seq].copy_(model_kv_cache[layer][0][0])
            self.v_cache[layer, slot, :, :n_seq].copy_(model_kv_cache[layer][1][0])
        self.seq_lens[slot] = n_seq
        self._host_seq_lens[slot] = n_seq
        self.attn_mask[slot].fill_(float("-inf"))
        self.attn_mask[slot, 0, 0, :n_seq] = 0

    def free(self, pid, compact=True):
        slot = self.pid_to_slot.pop(pid)
        self.seq_lens[slot] = 0
        self._host_seq_lens[slot] = 0
        self.attn_mask[slot].fill_(float("-inf"))
        self._free_slots.append(slot)
        self._free_slots.sort()
        if compact:
            self.compact()

    def compact(self):
        """Repack active sequences into slots 0..n-1 (required for CUDA graphs)."""
        if not self.pid_to_slot:
            self._free_slots = list(range(self.max_batch_size))
            self._host_seq_lens = [0] * self.max_batch_size
            return
        ordered = sorted(self.pid_to_slot.items(), key=lambda kv: kv[1])
        if [s for _, s in ordered] == list(range(len(ordered))):
            return  # already dense
        new_map = {}
        for new_slot, (pid, old_slot) in enumerate(ordered):
            if new_slot == old_slot:
                new_map[pid] = new_slot
                continue
            # Move KV + metadata
            self.k_cache[:, new_slot].copy_(self.k_cache[:, old_slot])
            self.v_cache[:, new_slot].copy_(self.v_cache[:, old_slot])
            self.seq_lens[new_slot] = self.seq_lens[old_slot]
            self._host_seq_lens[new_slot] = self._host_seq_lens[old_slot]
            self.attn_mask[new_slot].copy_(self.attn_mask[old_slot])
            self.seq_lens[old_slot] = 0
            self._host_seq_lens[old_slot] = 0
            self.attn_mask[old_slot].fill_(float("-inf"))
            new_map[pid] = new_slot
        self.pid_to_slot = new_map
        n = len(new_map)
        self._free_slots = list(range(n, self.max_batch_size))

    def get_decode_closures(self, grow=True, attn_len=None):
        sorted_pids = sorted(
            self.pid_to_slot.keys(), key=lambda p: self.pid_to_slot[p]
        )
        n = len(sorted_pids)
        if n == 0:
            raise RuntimeError("no active sequences")
        slots = [self.pid_to_slot[pid] for pid in sorted_pids]
        contiguous = slots == list(range(n))
        if self._cg_bs is not None and not contiguous:
            raise RuntimeError(
                "dense CUDA graph requires contiguous slots 0..bs-1"
            )
        if self._cg_bs is not None and n != self._cg_bs:
            raise RuntimeError(
                f"cuda-graph batch size mismatch: wrapper={self._cg_bs} active={n}"
            )

        # Contiguous 0..n-1: slots buffer is already arange; skip host->device copy.
        if not contiguous:
            self._write_slots[:n].copy_(
                torch.tensor(slots, dtype=torch.int64, device=device)
            )
        slot_tensor = self._write_slots[:n]
        self._active_n = n

        if grow:
            live_before = self.max_host_seq_len(n)
            if live_before >= self.max_seq_len:
                raise RuntimeError(
                    f"sequence exceeded max_seq_len={self.max_seq_len}"
                )
            # Write at current length, then bump host + device counters.
            self._write_positions[:n].copy_(self.seq_lens[slot_tensor].to(torch.int64))
            positions = self._write_positions[:n]
            self.seq_lens[slot_tensor] = self.seq_lens[slot_tensor] + 1
            for s in slots:
                self._host_seq_lens[s] += 1
            self.attn_mask[slot_tensor, 0, 0, positions] = 0

            def _cache_updater(layer_id, append_kv):
                assert append_kv[0].shape[0] == self._active_n
                k = append_kv[0].squeeze(2)
                v = append_kv[1].squeeze(2)
                if k.dtype != torch.float16:
                    k = k.half()
                    v = v.half()
                sl = self._write_slots[: self._active_n]
                pos = self._write_positions[: self._active_n]
                self.k_cache[layer_id, sl, :, pos, :] = k
                self.v_cache[layer_id, sl, :, pos, :] = v

        else:

            def _cache_updater(layer_id, append_kv):
                raise RuntimeError(
                    "cross-attn KV is static; cache updater must not be called"
                )

        k_view = self.k_cache[:, :n]
        v_view = self.v_cache[:, :n]
        mask_view = self.attn_mask[:n]
        use_index = not contiguous
        live_len = self.max_host_seq_len(n)
        # Fixed attn_len for CUDA-graph capture/replay (seq-length bucket).
        cur_len = int(attn_len) if attn_len is not None else live_len
        if cur_len < live_len:
            raise RuntimeError(f"attn_len {cur_len} < live seq {live_len}")
        if cur_len > self.max_seq_len:
            cur_len = self.max_seq_len

        def _attn(layer_id, q):
            if use_index:
                sl = self._write_slots[: self._active_n]
                k = self.k_cache[layer_id].index_select(0, sl)[:, :, :cur_len]
                v = self.v_cache[layer_id].index_select(0, sl)[:, :, :cur_len]
                m = self.attn_mask.index_select(0, sl)[:, :, :, :cur_len]
            else:
                k = k_view[layer_id, :, :, :cur_len]
                v = v_view[layer_id, :, :, :cur_len]
                m = mask_view[:, :, :, :cur_len]
            if sdpa_kernel is None:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=m
                )
            backends = _SDPA_SHORT if cur_len <= _SDPA_SHORT_LEN else _SDPA_LONG
            with sdpa_kernel(backends):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=m
                )

        return _cache_updater, _attn


def VirtualMemory(
    max_num_pages, page_size, num_kv_heads, head_dim, num_layers, type="sdpa", **kwargs
):
    if type == "paged":
        return VirtualMemoryPaged(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
    elif type == "sdpa":
        return VirtualMemorySDPA(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
    elif type == "dense":
        return VirtualMemoryDense(
            max_num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            num_layers,
            **kwargs,
        )
    elif type == "compare":
        return VirtualMemoryCompare(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
    raise ValueError(f"unknown VirtualMemory type: {type}")

