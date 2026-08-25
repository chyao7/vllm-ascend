#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from __future__ import annotations

import torch
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, insert_slice


# TODO: UB size differs across chips; consider whether BLOCK_SIZE can
# be dynamically computed with a formula instead of autotuning {1,2,4}.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1}),
        triton.Config({"BLOCK_SIZE": 2}),
        triton.Config({"BLOCK_SIZE": 4}),
    ],
    key=["q_cols", "k_cols"],
)
@triton.jit
def _split_qkv_and_compute_local_qk_var_kernel(
    input_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    qk_var_ptr,
    num_tokens,
    q_cols: tl.constexpr,
    k_cols: tl.constexpr,
    q_cols_pow2: tl.constexpr,
    k_cols_pow2: tl.constexpr,
    qkv_stride: tl.constexpr,
    q_inv_size: tl.constexpr,
    k_inv_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid Stride Loop + batch loading + precomputed reciprocal.
    (BLOCK_SIZE is limited to 1-4 to prevent UB overflow for large hidden_size)
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_range = tl.arange(0, BLOCK_SIZE)

    # Grid Stride Loop: each program processes BLOCK_SIZE tokens at a time
    stride = num_pids * BLOCK_SIZE
    start_token_idx = pid * BLOCK_SIZE

    for block_start in tl.range(start_token_idx, num_tokens, stride):
        token_indices = block_start + block_range
        token_mask = (token_indices < num_tokens)[:, None]

        # === Batch load QKV data ===
        # Q: [BLOCK_SIZE, q_cols]
        q_offset = tl.arange(0, q_cols_pow2)[None, :]
        q_mask = token_mask & (q_offset < q_cols)
        q_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_offset,
            mask=q_mask,
            other=0.0,
        )
        q_batch_f32 = q_batch.to(tl.float32)

        # K: [BLOCK_SIZE, k_cols], K follows immediately after Q
        k_offset = tl.arange(0, k_cols_pow2)[None, :]
        k_mask = token_mask & (k_offset < k_cols)
        k_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_cols + k_offset,
            mask=k_mask,
            other=0.0,
        )
        k_batch_f32 = k_batch.to(tl.float32)

        # V: [BLOCK_SIZE, k_cols], V is at offset Q + 2*K
        v_offset = tl.arange(0, k_cols_pow2)[None, :]
        v_mask = token_mask & (v_offset < k_cols)
        v_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_cols + k_cols + v_offset,
            mask=v_mask,
            other=0.0,
        )

        # === Batch compute sum of squares ===
        q_squaresum = tl.sum(q_batch_f32 * q_batch_f32, axis=-1) * q_inv_size
        k_squaresum = tl.sum(k_batch_f32 * k_batch_f32, axis=-1) * k_inv_size

        # === Batch store QKV output ===
        # Store Q
        q_out_offset = token_indices[:, None] * q_cols + q_offset
        q_out_mask = token_mask & (q_offset < q_cols)
        tl.store(q_out_ptr + q_out_offset, q_batch, mask=q_out_mask)

        # Store K
        k_out_offset = token_indices[:, None] * k_cols + k_offset
        k_out_mask = token_mask & (k_offset < k_cols)
        tl.store(k_out_ptr + k_out_offset, k_batch, mask=k_out_mask)

        # Store V
        v_out_offset = token_indices[:, None] * k_cols + v_offset
        v_out_mask = token_mask & (v_offset < k_cols)
        tl.store(v_out_ptr + v_out_offset, v_batch, mask=v_out_mask)

        # === Store variance ===
        var_offset = token_indices * 2
        var_mask = token_indices < num_tokens
        tl.store(qk_var_ptr + var_offset, q_squaresum, mask=var_mask)
        tl.store(qk_var_ptr + var_offset + 1, k_squaresum, mask=var_mask)


# Token tile size for _apply_global_rmsnorm_kernel. Measured on Ascend 910B
# with MiniMax-M2.5 shapes (TP4: 12q+2kv heads, TP8: 6q+1kv): BLOCK_T=4 gives
# ~1.5x over the single-token version at 16K tokens and ties at small token
# counts; BLOCK_T=8 overflows UB at TP4. The kernel uses a 2D
# (token*head, head_dim) tile layout because the CANN Triton backend
# mis-plans UB for 3D extract_slice/insert_slice with constexpr sizes.
_APPLY_GLOBAL_RMSNORM_BLOCK_T = 4


@triton.jit
def _apply_global_rmsnorm_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    cs_row_stride,
    q_weight_ptr,
    k_weight_ptr,
    qk_global_var_ptr,
    eps: tl.constexpr,
    inv_tp_world: tl.constexpr,
    num_tokens,
    q_cols: tl.constexpr,
    k_cols: tl.constexpr,
    q_num_heads: tl.constexpr,
    k_num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    tokens_per_program = tl.cdiv(num_tokens, num_programs)
    program_token_offset = pid * tokens_per_program
    program_token_end = min(program_token_offset + tokens_per_program, num_tokens)

    # Rows are (token, head) pairs: row = local_token * num_heads + head.
    q_rows: tl.constexpr = BLOCK_T * q_num_heads
    k_rows: tl.constexpr = BLOCK_T * k_num_heads

    q_row_arange = tl.arange(0, q_rows)
    q_head_in_row = q_row_arange % q_num_heads
    q_tok_in_tile = q_row_arange // q_num_heads
    k_row_arange = tl.arange(0, k_rows)
    k_head_in_row = k_row_arange % k_num_heads
    k_tok_in_tile = k_row_arange // k_num_heads

    hd_offsets = tl.arange(0, head_dim)[None, :]
    half_offsets = tl.arange(0, HALF)[None, :]

    # weight broadcast per row: [rows, head_dim]
    q_weight = tl.load(q_weight_ptr + q_head_in_row[:, None] * head_dim + hd_offsets).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + k_head_in_row[:, None] * head_dim + hd_offsets).to(tl.float32)

    num_tiles = tl.cdiv(tokens_per_program, BLOCK_T)
    for tile_iter in tl.range(num_tiles):
        tile_base = program_token_offset + tile_iter * BLOCK_T

        q_token_idx = tile_base + q_tok_in_tile
        k_token_idx = tile_base + k_tok_in_tile
        q_tok_mask = q_token_idx < program_token_end
        k_tok_mask = k_token_idx < program_token_end
        q_mask = q_tok_mask[:, None]
        k_mask = k_tok_mask[:, None]

        q_gv = tl.load(qk_global_var_ptr + q_token_idx * 2, mask=q_tok_mask, other=0.0).to(tl.float32)
        k_gv = tl.load(qk_global_var_ptr + k_token_idx * 2 + 1, mask=k_tok_mask, other=0.0).to(tl.float32)
        q_scale = 1.0 / tl.sqrt(q_gv * inv_tp_world + eps)
        k_scale = 1.0 / tl.sqrt(k_gv * inv_tp_world + eps)

        q_offsets = q_token_idx[:, None] * q_cols + q_head_in_row[:, None] * head_dim + hd_offsets
        q_vals_raw = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        q_vals = q_vals_raw.to(tl.float32) * q_scale[:, None] * q_weight

        k_offsets = k_token_idx[:, None] * k_cols + k_head_in_row[:, None] * head_dim + hd_offsets
        k_vals_raw = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        k_vals = k_vals_raw.to(tl.float32) * k_scale[:, None] * k_weight

        # Neox-style RoPE on the first rotary_dim dimensions of each head;
        # cos/sin broadcast per row: [rows, HALF]
        q_cos = tl.load(cos_ptr + q_token_idx[:, None] * cs_row_stride + half_offsets, mask=q_mask, other=0.0).to(tl.float32)
        q_sin = tl.load(sin_ptr + q_token_idx[:, None] * cs_row_stride + half_offsets, mask=q_mask, other=0.0).to(tl.float32)
        k_cos = tl.load(cos_ptr + k_token_idx[:, None] * cs_row_stride + half_offsets, mask=k_mask, other=0.0).to(tl.float32)
        k_sin = tl.load(sin_ptr + k_token_idx[:, None] * cs_row_stride + half_offsets, mask=k_mask, other=0.0).to(tl.float32)

        q1 = extract_slice(q_vals, offsets=(0, 0), sizes=(q_rows, HALF), strides=(1, 1))
        q2 = extract_slice(q_vals, offsets=(0, HALF), sizes=(q_rows, HALF), strides=(1, 1))
        q_vals = insert_slice(
            q_vals,
            q1 * q_cos - q2 * q_sin,
            offsets=(0, 0),
            sizes=(q_rows, HALF),
            strides=(1, 1),
        )
        q_vals = insert_slice(
            q_vals,
            q2 * q_cos + q1 * q_sin,
            offsets=(0, HALF),
            sizes=(q_rows, HALF),
            strides=(1, 1),
        )
        tl.store(q_ptr + q_offsets, q_vals.to(q_vals_raw.dtype), mask=q_mask)

        k1 = extract_slice(k_vals, offsets=(0, 0), sizes=(k_rows, HALF), strides=(1, 1))
        k2 = extract_slice(k_vals, offsets=(0, HALF), sizes=(k_rows, HALF), strides=(1, 1))
        k_vals = insert_slice(
            k_vals,
            k1 * k_cos - k2 * k_sin,
            offsets=(0, 0),
            sizes=(k_rows, HALF),
            strides=(1, 1),
        )
        k_vals = insert_slice(
            k_vals,
            k2 * k_cos + k1 * k_sin,
            offsets=(0, HALF),
            sizes=(k_rows, HALF),
            strides=(1, 1),
        )
        tl.store(k_ptr + k_offsets, k_vals.to(k_vals_raw.dtype), mask=k_mask)


def split_qkv_tp_rmsnorm_rope_impl(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    tp_world: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = input.shape[0]
    input_2d = input.view(num_tokens, -1)
    q = torch.empty(num_tokens, q_hidden_size, device=input.device, dtype=input.dtype)
    k = torch.empty(num_tokens, kv_hidden_size, device=input.device, dtype=input.dtype)
    v = torch.empty(num_tokens, kv_hidden_size, device=input.device, dtype=input.dtype)
    if num_tokens == 0:
        return q, k, v

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)
    q_cols = q_hidden_size
    k_cols = kv_hidden_size
    q_num_heads = q_hidden_size // head_dim
    k_num_heads = kv_hidden_size // head_dim

    qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=q.device)
    # Precompute reciprocal to avoid division inside kernel
    q_inv_size = 1.0 / q_cols
    k_inv_size = 1.0 / k_cols
    # Pad to power-of-2 for tl.arange (required by Ascend NPU Triton backend)
    q_cols_pow2 = 1 << (q_cols - 1).bit_length()
    k_cols_pow2 = 1 << (k_cols - 1).bit_length()
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input_2d,
        q,
        k,
        v,
        qk_var,
        num_tokens,
        q_cols,
        k_cols,
        q_cols_pow2,
        k_cols_pow2,
        q_cols + 2 * k_cols,
        q_inv_size,
        k_inv_size,
    )
    if tp_world > 1:
        qk_var = tensor_model_parallel_all_reduce(qk_var)

    cos_2d = cos.view(num_tokens, -1)
    sin_2d = sin.view(num_tokens, -1)
    q_2d = q.view(num_tokens, -1)
    k_2d = k.view(num_tokens, -1)
    _apply_global_rmsnorm_kernel[grid](
        q_2d,
        k_2d,
        cos_2d,
        sin_2d,
        cos_2d.stride(0),
        q_weight,
        k_weight,
        qk_var,
        eps,
        1.0 / tp_world,
        num_tokens,
        q_cols,
        k_cols,
        q_num_heads,
        k_num_heads,
        head_dim,
        rotary_dim,
        rotary_dim // 2,
        _APPLY_GLOBAL_RMSNORM_BLOCK_T,
    )

    return q, k, v


def split_qkv_tp_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    tp_world: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = input.shape[0]
    q_out = torch.empty(
        num_tokens,
        q_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    k_out = torch.empty(
        num_tokens,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    v_out = torch.empty(
        num_tokens,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    return q_out, k_out, v_out


direct_register_custom_op(
    op_name="split_qkv_tp_rmsnorm_rope",
    op_func=split_qkv_tp_rmsnorm_rope_impl,
    fake_impl=split_qkv_tp_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
