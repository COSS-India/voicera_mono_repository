import torch
from inference.config import device
import math
import flashinfer


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

        workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer
        )

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

    def get_decode_closures(self, grow=True):
        """
        grow=True  (self-attn): allocate 1 token slot per seq, plan, return updater+attn.
        grow=False (cross-attn): KV is static; plan only when layout changed, attn only.
        """
        sorted_pids = sorted(self.page_table.pid_mem_sizes.keys())
        n = len(sorted_pids)

        if grow:
            for pid in sorted_pids:
                self.page_table.allocate(pid=pid, mem_size=1)
            self._bump_layout()

            positions = torch.tensor(
                [self.page_table.pid_mem_sizes[pid] - 1 for pid in sorted_pids],
                dtype=torch.int32,
                device=device,
            )
            batch_indices = torch.arange(n, dtype=torch.int32, device=device)
            kv_indices, kv_indptr, kv_last_page_len = (
                self.page_table.convert_to_flashinfer()
            )
            self._plan_decode(kv_indptr, kv_indices, kv_last_page_len)

            def _cache_updater(layer_id, append_kv):
                num_batches = append_kv[0].shape[0]
                num_seqs = append_kv[0].shape[2]
                assert num_batches == len(
                    self.page_table.pid_mem_sizes
                ), "batch size doesn't match active sequences"
                assert num_seqs == 1, "decode step assumes only 1 token decoded per batch"
                # Already fp16 from attention projs; keep contiguous for flashinfer.
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

    def prefill(self, pid, model_kv_cache):
        assert model_kv_cache[0][0].shape[0] == 1
        self.pid_kv_cache[pid] = model_kv_cache

    def get_decode_closures(self, grow=True):
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

    def prefill(self, pid, model_kv_cache):
        self.vm_paged.prefill(pid, model_kv_cache)
        self.vm_sdpa.prefill(pid, model_kv_cache)

    def get_decode_closures(self, grow=True):
        paged_cache_updater, paged_attn = self.vm_paged.get_decode_closures(grow=grow)
        sdpa_cache_updater, sdpa_attn = self.vm_sdpa.get_decode_closures(grow=grow)

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


def VirtualMemory(
    max_num_pages, page_size, num_kv_heads, head_dim, num_layers, type="sdpa"
):
    if type == "paged":
        return VirtualMemoryPaged(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
    elif type == "sdpa":
        return VirtualMemorySDPA(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
    elif type == "compare":
        return VirtualMemoryCompare(
            max_num_pages, page_size, num_kv_heads, head_dim, num_layers
        )
