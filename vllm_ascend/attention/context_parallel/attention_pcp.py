#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""GQA Prefill Context Parallel attention (PCP-only, dcp=1).

Prefill: each rank computes Q/K/V for its sequence shard, all-gathers the
current-chunk KV, and attends with a position/request mask.
Decode: Q is replicated; each rank attends to local paged KV and merges LSE
on ``pcp_group``. Q heads are not gathered (that is the DCP path).
"""

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    PCPImplMixin,
    _update_out_and_lse,
    get_dcp_local_seq_lens,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.memcache_comm_fence import record_attention_compute_start


@dataclass
class AscendAttentionPCPMetadata(AscendMetadata):
    query_positions: torch.Tensor | None = None
    token_req_ids: torch.Tensor | None = None
    local_seq_lens: torch.Tensor | None = None
    local_context_lens: torch.Tensor | None = None
    pcp_size: int = 1
    pcp_rank: int = 0
    max_tokens_across_pcp: int = 0


class AscendAttentionPCPMetadataBuilder(AscendAttentionMetadataBuilder):
    metadata_cls = AscendAttentionPCPMetadata

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parallel_config = self.vllm_config.parallel_config
        self.pcp_size = parallel_config.prefill_context_parallel_size
        self.interleave_size = parallel_config.cp_kv_cache_interleave_size
        from vllm.distributed import get_pcp_group

        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0

    def _build_backend_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, object]:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        positions = common_attn_metadata.positions
        if positions is not None:
            query_positions = positions[:num_actual_tokens]
        else:
            query_positions = None

        query_start_loc = common_attn_metadata.query_start_loc_cpu[: common_attn_metadata.num_reqs + 1]
        token_req_ids = None
        if query_start_loc is not None:
            q_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32)
            req_ids = torch.arange(q_lens.numel(), dtype=torch.int32)
            token_req_ids = torch.repeat_interleave(req_ids, q_lens).to(self.device)

        seq_lens_t = seq_lens if isinstance(seq_lens, torch.Tensor) else torch.as_tensor(seq_lens, dtype=torch.int32)
        local_seq_lens = get_dcp_local_seq_lens(
            seq_lens_t.to(dtype=torch.int32),
            self.pcp_size,
            self.interleave_size,
        )[:, self.pcp_rank]
        computed = common_attn_metadata.num_computed_tokens_cpu
        if computed is not None:
            computed_t = computed[: common_attn_metadata.num_reqs].to(dtype=torch.int32)
            local_context_lens = get_dcp_local_seq_lens(
                computed_t,
                self.pcp_size,
                self.interleave_size,
            )[:, self.pcp_rank]
        else:
            local_context_lens = None

        return {
            "query_positions": query_positions,
            "token_req_ids": token_req_ids,
            "local_seq_lens": local_seq_lens.to(self.device),
            "local_context_lens": None if local_context_lens is None else local_context_lens.to(self.device),
            "pcp_size": self.pcp_size,
            "pcp_rank": self.pcp_rank,
            "max_tokens_across_pcp": num_actual_tokens,
        }


class AscendAttentionPCPImpl(PCPImplMixin, AscendAttentionBackendImpl):
    supports_pcp = True
    can_return_lse_for_decode = True

    def _build_prefill_mask(
        self,
        q_pos: torch.Tensor,
        q_req: torch.Tensor,
        k_pos: torch.Tensor,
        k_req: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        keep = (q_req[:, None] == k_req[None, :]) & (q_pos[:, None] >= k_pos[None, :])
        mask_value = float("-inf") if dtype == torch.float16 else 1
        if dtype == torch.float16:
            attn_mask = torch.zeros(q_pos.shape[0], k_pos.shape[0], dtype=dtype, device=q_pos.device)
            return attn_mask.masked_fill_(~keep, mask_value)
        return (~keep).to(torch.uint8)

    def _fia_tnd(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        atten_mask: torch.Tensor | None,
        actual_seq_q: list[int] | torch.Tensor,
        actual_seq_kv: list[int] | torch.Tensor,
        sparse_mode: int,
        softmax_lse_flag: bool = True,
        block_table: torch.Tensor | None = None,
        block_size: int | None = None,
    ):
        kwargs = dict(
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=atten_mask,
            scale=self.scale,
            sparse_mode=sparse_mode,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=softmax_lse_flag,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
        )
        if block_table is not None:
            kwargs["block_table"] = block_table
            kwargs["block_size"] = block_size
        return torch_npu.npu_fused_infer_attention_score(query, key, value, **kwargs)

    def _forward_prefill_pcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendAttentionPCPMetadata,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if num_tokens == 0:
            empty = query.new_zeros(0, self.num_heads, self.head_size)
            empty_lse = query.new_zeros(0, self.num_heads, 1)
            return empty, empty_lse
        q_pos = attn_metadata.query_positions[:num_tokens]
        q_req = attn_metadata.token_req_ids[:num_tokens] if attn_metadata.token_req_ids is not None else torch.zeros(
            num_tokens, dtype=torch.int32, device=query.device
        )
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        pad_to = max(
            int(getattr(_EXTRA_CTX, "max_tokens_across_pcp", 0) or 0),
            int(attn_metadata.max_tokens_across_pcp),
            num_tokens,
        )
        if key.shape[0] < pad_to:
            pad = pad_to - key.shape[0]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, pad))
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, pad))
            k_pos = torch.nn.functional.pad(q_pos, (0, pad), value=-1)
            k_req = torch.nn.functional.pad(q_req, (0, pad), value=-1)
        else:
            k_pos = q_pos
            k_req = q_req
        key_all = self._pcp_all_gather(key[:pad_to], 0)
        value_all = self._pcp_all_gather(value[:pad_to], 0)
        pos_all = self._pcp_all_gather(k_pos[:pad_to], 0)
        req_all = self._pcp_all_gather(k_req[:pad_to], 0)

        outs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        zeros = torch.zeros(num_tokens, self.num_heads, self.head_size, dtype=query.dtype, device=query.device)
        neg_inf = torch.full(
            (num_tokens, self.num_heads, 1),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=query.device,
        )
        for rank in range(self.pcp_size):
            sl = slice(rank * pad_to, (rank + 1) * pad_to)
            pos_r = pos_all[sl]
            valid = pos_r >= 0
            if not bool(valid.any()):
                outs.append(zeros)
                lses.append(neg_inf)
                continue
            key_r = key_all[sl][valid]
            value_r = value_all[sl][valid]
            pos_r = pos_r[valid]
            req_r = req_all[sl][valid]
            atten_mask = self._build_prefill_mask(q_pos, q_req, pos_r, req_r, query.dtype)
            attn_out, attn_lse = self._fia_tnd(
                query,
                key_r.contiguous(),
                value_r.contiguous(),
                atten_mask=atten_mask,
                actual_seq_q=[num_tokens],
                actual_seq_kv=[int(key_r.shape[0])],
                sparse_mode=0,
                softmax_lse_flag=True,
            )
            if attn_lse.dim() == 2:
                attn_lse = attn_lse.unsqueeze(-1)
            outs.append(attn_out)
            lses.append(attn_lse.to(torch.float32))
        out_stack = torch.stack([out.to(torch.float32) for out in outs], dim=0)
        lse_stack = torch.stack(lses, dim=0)
        merged, merged_lse = _update_out_and_lse(out_stack, lse_stack)
        return merged.to(query.dtype), merged_lse

    def _forward_decode_pcp(
        self,
        query: torch.Tensor,
        attn_metadata: AscendAttentionPCPMetadata,
    ) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None
        num_decodes = attn_metadata.num_decodes
        if num_decodes == 0:
            return query
        decode_query = query[: attn_metadata.num_decode_tokens]
        k_nope = self.key_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.value_cache.shape[0], self.value_cache.shape[1], -1)
        local_kv_lens = attn_metadata.local_seq_lens[:num_decodes]
        actual_seq_q = attn_metadata.actual_seq_lengths_q[:num_decodes]
        attn_out, attn_lse = self._fia_tnd(
            decode_query,
            k_nope,
            value,
            atten_mask=None,
            actual_seq_q=actual_seq_q,
            actual_seq_kv=local_kv_lens,
            sparse_mode=0,
            softmax_lse_flag=True,
            block_table=attn_metadata.block_tables[:num_decodes],
            block_size=self.key_cache.shape[1],
        )
        if attn_lse.dim() == 2:
            attn_lse = attn_lse.unsqueeze(-1)
        merged_out, _ = self._merge_pcp_attention_output(attn_out, attn_lse)
        return merged_out

    def _forward_context_pcp(
        self,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendAttentionPCPMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if attn_metadata.local_context_lens is None:
            return None
        context_lens = attn_metadata.local_context_lens
        if int(context_lens.max().item()) <= 0:
            return None
        cache_key, cache_value = kv_cache[0], kv_cache[1]
        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decodes
        prefill_query = query[attn_metadata.num_decode_tokens : attn_metadata.num_actual_tokens]
        local_ctx = context_lens[num_decodes : num_decodes + num_prefills]
        total_toks = int(local_ctx.sum().item())
        if total_toks <= 0:
            zeros = torch.zeros(
                prefill_query.shape[0],
                self.num_heads,
                self.head_size,
                dtype=prefill_query.dtype,
                device=prefill_query.device,
            )
            lse = torch.full(
                (prefill_query.shape[0], self.num_heads, 1),
                fill_value=-float("inf"),
                dtype=torch.float32,
                device=prefill_query.device,
            )
            return zeros, lse

        key = torch.empty(
            total_toks, cache_key.size(2), cache_key.size(-1), dtype=prefill_query.dtype, device=prefill_query.device
        )
        value = torch.empty_like(key)
        DeviceOperator.kv_cache_load(
            cache_key,
            cache_value,
            attn_metadata.block_tables[num_decodes : num_decodes + num_prefills],
            local_ctx,
            torch.zeros(num_prefills, dtype=torch.int32, device=prefill_query.device),
            key=key,
            value=value,
        )
        actual_q = attn_metadata.actual_seq_lengths_q
        if num_decodes > 0:
            q_base = actual_q[num_decodes - 1]
            actual_q = [v - q_base for v in actual_q[num_decodes:]]
        attn_out, attn_lse = self._fia_tnd(
            prefill_query,
            key.contiguous(),
            value.contiguous(),
            atten_mask=None,
            actual_seq_q=actual_q,
            actual_seq_kv=torch.cumsum(local_ctx, dim=0),
            sparse_mode=0,
            softmax_lse_flag=True,
        )
        if attn_lse.dim() == 2:
            attn_lse = attn_lse.unsqueeze(-1)
        return self._merge_pcp_attention_output(attn_out, attn_lse)

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, AscendAttentionPCPMetadata)
        record_attention_compute_start()
        num_actual = attn_metadata.num_actual_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        if has_decode:
            output[: attn_metadata.num_decode_tokens] = self._forward_decode_pcp(query, attn_metadata)
        if has_prefill:
            prefill_q = query[attn_metadata.num_decode_tokens : num_actual]
            prefill_k = key[attn_metadata.num_decode_tokens : num_actual]
            prefill_v = value[attn_metadata.num_decode_tokens : num_actual]
            current_out, current_lse = self._forward_prefill_pcp(prefill_q, prefill_k, prefill_v, attn_metadata)
            if attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                context = self._forward_context_pcp(query, kv_cache, attn_metadata)
                if context is not None:
                    context_out, context_lse = context
                    current_out, _ = _update_out_and_lse(
                        torch.stack(
                            [current_out.to(torch.float32), context_out.to(torch.float32)],
                            dim=0,
                        ),
                        torch.stack(
                            [current_lse.to(torch.float32), context_lse.to(torch.float32)],
                            dim=0,
                        ),
                    )
                    current_out = current_out.to(prefill_q.dtype)
            output[attn_metadata.num_decode_tokens : num_actual] = current_out
        return output
