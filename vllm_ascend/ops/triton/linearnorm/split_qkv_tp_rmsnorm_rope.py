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

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


# The Q/K/V rows are loaded as (token, head, head_dim) 3D segments instead of
# flat (token, cols_pow2) 2D tiles: q_cols=768 (TP8) would otherwise be padded
# to 1024 for tl.arange, wasting 33% UB per tile and forcing BLOCK_SIZE<=4.
# Masks stay token-dim-only (the same form _apply_global_rmsnorm_kernel uses);
# the CANN Triton backend scalarizes loads whose mask mixes in a column term.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1}),
        triton.Config({"BLOCK_SIZE": 2}),
        triton.Config({"BLOCK_SIZE": 4}),
        triton.Config({"BLOCK_SIZE": 8}),
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
    qkv_stride: tl.constexpr,
    q_num_heads: tl.constexpr,
    k_num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    q_inv_size: tl.constexpr,
    k_inv_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid Stride Loop + batch loading + precomputed reciprocal.
    Each Q/K/V row is accessed as [BLOCK_SIZE, num_heads, head_dim] segments,
    so no power-of-2 column padding is needed (head_dim is always a power of 2).
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_range = tl.arange(0, BLOCK_SIZE)

    # Grid Stride Loop: each program processes BLOCK_SIZE tokens at a time
    stride = num_pids * BLOCK_SIZE
    start_token_idx = pid * BLOCK_SIZE

    q_heads = tl.arange(0, q_num_heads)
    k_heads = tl.arange(0, k_num_heads)
    hd_offsets = tl.arange(0, head_dim)

    for block_start in tl.range(start_token_idx, num_tokens, stride):
        token_indices = block_start + block_range
        m3 = (token_indices < num_tokens)[:, None, None]

        # === Batch load QKV data as (token, head, head_dim) segments ===
        row_base = token_indices[:, None, None] * qkv_stride
        q_in_off = row_base + q_heads[None, :, None] * head_dim + hd_offsets[
            None, None, :]
        q_batch = tl.load(input_ptr + q_in_off, mask=m3, other=0.0)
        q_batch_f32 = q_batch.to(tl.float32)

        # K follows immediately after Q in the fused row, V after K
        k_in_off = row_base + q_cols + k_heads[
            None, :, None] * head_dim + hd_offsets[None, None, :]
        k_batch = tl.load(input_ptr + k_in_off, mask=m3, other=0.0)
        k_batch_f32 = k_batch.to(tl.float32)

        v_batch = tl.load(input_ptr + k_in_off + k_cols, mask=m3, other=0.0)

        # === Batch compute sum of squares ===
        q_squaresum = tl.sum(tl.sum(q_batch_f32 * q_batch_f32, axis=-1),
                             axis=-1) * q_inv_size
        k_squaresum = tl.sum(tl.sum(k_batch_f32 * k_batch_f32, axis=-1),
                             axis=-1) * k_inv_size

        # === Batch store QKV output ===
        q_out_off = token_indices[:, None, None] * q_cols + q_heads[
            None, :, None] * head_dim + hd_offsets[None, None, :]
        tl.store(q_out_ptr + q_out_off, q_batch, mask=m3)

        k_out_off = token_indices[:, None, None] * k_cols + k_heads[
            None, :, None] * head_dim + hd_offsets[None, None, :]
        tl.store(k_out_ptr + k_out_off, k_batch, mask=m3)
        tl.store(v_out_ptr + k_out_off, v_batch, mask=m3)

        # === Store variance ===
        var_offset = token_indices * 2
        var_mask = token_indices < num_tokens
        tl.store(qk_var_ptr + var_offset, q_squaresum, mask=var_mask)
        tl.store(qk_var_ptr + var_offset + 1, k_squaresum, mask=var_mask)


# Token tile size for _apply_global_rmsnorm_kernel. Measured on Ascend 910B
# with MiniMax-M2.5 shapes (TP4: 12q+2kv heads, TP8: 6q+1kv): BLOCK_T=8 gives
# 1.26x (TP8) / 1.12x (TP4) over the single-token version at 16K tokens and
# ties at small token counts (host-launch bound there). Each head is processed
# as three plain 3D segments (the two rotary halves and the pass-through tail)
# instead of extract_slice/insert_slice on a flattened (token*head) row
# encoding: the CANN Triton backend mis-plans UB for constexpr-size 3D slices,
# and scalarizes masked loads/stores when the mask is combined with a
# %//-encoded row-validity term.
_APPLY_GLOBAL_RMSNORM_BLOCK_T = 8


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
    program_token_end = min(program_token_offset + tokens_per_program,
                            num_tokens)

    token_tile = tl.arange(0, BLOCK_T)
    q_heads = tl.arange(0, q_num_heads)
    k_heads = tl.arange(0, k_num_heads)
    half_offsets = tl.arange(0, HALF)
    PASS: tl.constexpr = head_dim - rotary_dim

    # per-head weight segments, hoisted out of the token loop
    q_w1 = tl.load(q_weight_ptr + q_heads[:, None] * head_dim +
                   half_offsets[None, :]).to(tl.float32)
    q_w2 = tl.load(q_weight_ptr + q_heads[:, None] * head_dim + HALF +
                   half_offsets[None, :]).to(tl.float32)
    k_w1 = tl.load(k_weight_ptr + k_heads[:, None] * head_dim +
                   half_offsets[None, :]).to(tl.float32)
    k_w2 = tl.load(k_weight_ptr + k_heads[:, None] * head_dim + HALF +
                   half_offsets[None, :]).to(tl.float32)
    if PASS > 0:
        pass_offsets = tl.arange(0, PASS)
        q_wp = tl.load(q_weight_ptr + q_heads[:, None] * head_dim +
                       rotary_dim + pass_offsets[None, :]).to(tl.float32)
        k_wp = tl.load(k_weight_ptr + k_heads[:, None] * head_dim +
                       rotary_dim + pass_offsets[None, :]).to(tl.float32)

    num_tiles = tl.cdiv(tokens_per_program, BLOCK_T)
    for tile_iter in tl.range(num_tiles):
        token_offsets = program_token_offset + tile_iter * BLOCK_T + token_tile
        token_mask = token_offsets < program_token_end
        m2 = token_mask[:, None]
        m3 = token_mask[:, None, None]

        q_gv = tl.load(qk_global_var_ptr + token_offsets * 2,
                       mask=token_mask,
                       other=0.0).to(tl.float32)
        k_gv = tl.load(qk_global_var_ptr + token_offsets * 2 + 1,
                       mask=token_mask,
                       other=0.0).to(tl.float32)
        q_scale = 1.0 / tl.sqrt(q_gv * inv_tp_world + eps)
        k_scale = 1.0 / tl.sqrt(k_gv * inv_tp_world + eps)

        cos_row = tl.load(cos_ptr + token_offsets[:, None] * cs_row_stride +
                          half_offsets[None, :],
                          mask=m2,
                          other=0.0).to(tl.float32)
        sin_row = tl.load(sin_ptr + token_offsets[:, None] * cs_row_stride +
                          half_offsets[None, :],
                          mask=m2,
                          other=0.0).to(tl.float32)
        cos_b = cos_row[:, None, :]
        sin_b = sin_row[:, None, :]

        # Q: neox RoPE on the two rotary half segments, rmsnorm on the tail
        q_base = token_offsets[:, None, None] * q_cols + q_heads[
            None, :, None] * head_dim
        q1_off = q_base + half_offsets[None, None, :]
        q2_off = q1_off + HALF
        q1_raw = tl.load(q_ptr + q1_off, mask=m3, other=0.0)
        q2_raw = tl.load(q_ptr + q2_off, mask=m3, other=0.0)
        q1n = q1_raw.to(tl.float32) * q_scale[:, None, None] * q_w1[None, :, :]
        q2n = q2_raw.to(tl.float32) * q_scale[:, None, None] * q_w2[None, :, :]
        tl.store(q_ptr + q1_off, (q1n * cos_b - q2n * sin_b).to(q1_raw.dtype),
                 mask=m3)
        tl.store(q_ptr + q2_off, (q2n * cos_b + q1n * sin_b).to(q2_raw.dtype),
                 mask=m3)
        if PASS > 0:
            qp_off = q_base + rotary_dim + pass_offsets[None, None, :]
            qp_raw = tl.load(q_ptr + qp_off, mask=m3, other=0.0)
            tl.store(q_ptr + qp_off,
                     (qp_raw.to(tl.float32) * q_scale[:, None, None] *
                      q_wp[None, :, :]).to(qp_raw.dtype),
                     mask=m3)

        # K: same segment structure
        k_base = token_offsets[:, None, None] * k_cols + k_heads[
            None, :, None] * head_dim
        k1_off = k_base + half_offsets[None, None, :]
        k2_off = k1_off + HALF
        k1_raw = tl.load(k_ptr + k1_off, mask=m3, other=0.0)
        k2_raw = tl.load(k_ptr + k2_off, mask=m3, other=0.0)
        k1n = k1_raw.to(tl.float32) * k_scale[:, None, None] * k_w1[None, :, :]
        k2n = k2_raw.to(tl.float32) * k_scale[:, None, None] * k_w2[None, :, :]
        tl.store(k_ptr + k1_off, (k1n * cos_b - k2n * sin_b).to(k1_raw.dtype),
                 mask=m3)
        tl.store(k_ptr + k2_off, (k2n * cos_b + k1n * sin_b).to(k2_raw.dtype),
                 mask=m3)
        if PASS > 0:
            kp_off = k_base + rotary_dim + pass_offsets[None, None, :]
            kp_raw = tl.load(k_ptr + kp_off, mask=m3, other=0.0)
            tl.store(k_ptr + kp_off,
                     (kp_raw.to(tl.float32) * k_scale[:, None, None] *
                      k_wp[None, :, :]).to(kp_raw.dtype),
                     mask=m3)


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
    q = torch.empty(num_tokens,
                    q_hidden_size,
                    device=input.device,
                    dtype=input.dtype)
    k = torch.empty(num_tokens,
                    kv_hidden_size,
                    device=input.device,
                    dtype=input.dtype)
    v = torch.empty(num_tokens,
                    kv_hidden_size,
                    device=input.device,
                    dtype=input.dtype)
    if num_tokens == 0:
        return q, k, v

    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore), )
    q_cols = q_hidden_size
    k_cols = kv_hidden_size
    q_num_heads = q_hidden_size // head_dim
    k_num_heads = kv_hidden_size // head_dim

    qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=q.device)
    # Precompute reciprocal to avoid division inside kernel
    q_inv_size = 1.0 / q_cols
    k_inv_size = 1.0 / k_cols
    _split_qkv_and_compute_local_qk_var_kernel[grid](
        input_2d,
        q,
        k,
        v,
        qk_var,
        num_tokens,
        q_cols,
        k_cols,
        q_cols + 2 * k_cols,
        q_num_heads,
        k_num_heads,
        head_dim,
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
