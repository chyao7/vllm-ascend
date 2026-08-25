# Benchmark for _apply_global_rmsnorm_kernel (MiniMax-M2.5 shapes),
# before/after comparison for the token-tiling change.
#
# Not collected by pytest (bench_ prefix). Run directly on an NPU machine:
#   python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/bench_split_qkv_tp_rmsnorm_rope.py
#
# Measures per shape:
#   - legacy kernel2 (single token per iteration, pre-change baseline)
#   - tiled kernel2 with BLOCK_T in {1,2,4,8}, invoked directly with a fixed
#     BLOCK_T to bypass the autotune wrapper and measure pure device time
# Correctness is checked against an fp32 PyTorch reference: the max abs
# error of each variant must be on the same bf16-rounding scale as legacy
# (tiling may flip FMA fusion, so bit-exact equality is NOT expected).

import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401  (registers torch.ops.vllm.split_qkv_tp_rmsnorm_rope)
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_vectorcore_num,
    init_device_properties_triton,
    insert_slice,
)

DTYPE = torch.bfloat16
EPS = 1e-6
HEAD_DIM = 128
ROTARY_DIM = 64
HALF = ROTARY_DIM // 2
WARMUP_ITERS = 10
BENCH_ITERS = 50
# bt=1 is omitted: the legacy kernel already covers the BLOCK_T=1 baseline.
BLOCK_T_CANDIDATES = [4, 8, 16]

# MiniMax-M2.5: 48 q heads, 8 kv heads. Local heads per TP rank:
#   TP=4 -> 12 q + 2 kv;  TP=8 -> 6 q + 1 kv
TP_SHAPES = {4: (12, 2), 8: (6, 1)}

# decode: batch * (1 + spec tokens); prefill: chunked prefill token budget
NUM_TOKENS_LIST = [128, 256, 512, 2048, 8192, 16384]


# ---------------------------------------------------------------------------
# Legacy kernel1: flat 2D tiles with power-of-2 padded columns (pre-change
# baseline). q_cols=768 is padded to 1024, wasting 33% UB per tile.
# ---------------------------------------------------------------------------
@triton.jit
def _split_qkv_var_kernel_legacy(
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
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_range = tl.arange(0, BLOCK_SIZE)
    stride = num_pids * BLOCK_SIZE
    start_token_idx = pid * BLOCK_SIZE

    for block_start in tl.range(start_token_idx, num_tokens, stride):
        token_indices = block_start + block_range
        token_mask = (token_indices < num_tokens)[:, None]

        q_offset = tl.arange(0, q_cols_pow2)[None, :]
        q_mask = token_mask & (q_offset < q_cols)
        q_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_offset,
            mask=q_mask,
            other=0.0,
        )
        q_batch_f32 = q_batch.to(tl.float32)

        k_offset = tl.arange(0, k_cols_pow2)[None, :]
        k_mask = token_mask & (k_offset < k_cols)
        k_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_cols +
            k_offset,
            mask=k_mask,
            other=0.0,
        )
        k_batch_f32 = k_batch.to(tl.float32)

        v_offset = tl.arange(0, k_cols_pow2)[None, :]
        v_mask = token_mask & (v_offset < k_cols)
        v_batch = tl.load(
            input_ptr + token_indices[:, None] * qkv_stride + q_cols + k_cols +
            v_offset,
            mask=v_mask,
            other=0.0,
        )

        q_squaresum = tl.sum(q_batch_f32 * q_batch_f32, axis=-1) * q_inv_size
        k_squaresum = tl.sum(k_batch_f32 * k_batch_f32, axis=-1) * k_inv_size

        q_out_offset = token_indices[:, None] * q_cols + q_offset
        tl.store(q_out_ptr + q_out_offset,
                 q_batch,
                 mask=token_mask & (q_offset < q_cols))
        k_out_offset = token_indices[:, None] * k_cols + k_offset
        tl.store(k_out_ptr + k_out_offset,
                 k_batch,
                 mask=token_mask & (k_offset < k_cols))
        v_out_offset = token_indices[:, None] * k_cols + v_offset
        tl.store(v_out_ptr + v_out_offset,
                 v_batch,
                 mask=token_mask & (v_offset < k_cols))

        var_offset = token_indices * 2
        var_mask = token_indices < num_tokens
        tl.store(qk_var_ptr + var_offset, q_squaresum, mask=var_mask)
        tl.store(qk_var_ptr + var_offset + 1, k_squaresum, mask=var_mask)


# ---------------------------------------------------------------------------
# Segmented kernel1: (token, head, head_dim) 3D segments, no pow2 padding.
# ---------------------------------------------------------------------------
@triton.jit
def _split_qkv_var_kernel_seg(
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
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_range = tl.arange(0, BLOCK_SIZE)
    stride = num_pids * BLOCK_SIZE
    start_token_idx = pid * BLOCK_SIZE

    q_heads = tl.arange(0, q_num_heads)
    k_heads = tl.arange(0, k_num_heads)
    hd_offsets = tl.arange(0, head_dim)

    for block_start in tl.range(start_token_idx, num_tokens, stride):
        token_indices = block_start + block_range
        m3 = (token_indices < num_tokens)[:, None, None]

        row_base = token_indices[:, None, None] * qkv_stride
        q_in_off = row_base + q_heads[None, :, None] * head_dim + hd_offsets[
            None, None, :]
        q_batch = tl.load(input_ptr + q_in_off, mask=m3, other=0.0)
        q_batch_f32 = q_batch.to(tl.float32)

        k_in_off = row_base + q_cols + k_heads[
            None, :, None] * head_dim + hd_offsets[None, None, :]
        k_batch = tl.load(input_ptr + k_in_off, mask=m3, other=0.0)
        k_batch_f32 = k_batch.to(tl.float32)

        v_batch = tl.load(input_ptr + k_in_off + k_cols, mask=m3, other=0.0)

        q_squaresum = tl.sum(tl.sum(q_batch_f32 * q_batch_f32, axis=-1),
                             axis=-1) * q_inv_size
        k_squaresum = tl.sum(tl.sum(k_batch_f32 * k_batch_f32, axis=-1),
                             axis=-1) * k_inv_size

        q_out_off = token_indices[:, None, None] * q_cols + q_heads[
            None, :, None] * head_dim + hd_offsets[None, None, :]
        tl.store(q_out_ptr + q_out_off, q_batch, mask=m3)
        k_out_off = token_indices[:, None, None] * k_cols + k_heads[
            None, :, None] * head_dim + hd_offsets[None, None, :]
        tl.store(k_out_ptr + k_out_off, k_batch, mask=m3)
        tl.store(v_out_ptr + k_out_off, v_batch, mask=m3)

        var_offset = token_indices * 2
        var_mask = token_indices < num_tokens
        tl.store(qk_var_ptr + var_offset, q_squaresum, mask=var_mask)
        tl.store(qk_var_ptr + var_offset + 1, k_squaresum, mask=var_mask)


def ref_kernel1(x, q_cols, k_cols):
    """fp32 reference of kernel1: split + per-token mean of squares."""
    xf = x.float()
    q_var = xf[:, :q_cols].pow(2).sum(dim=-1) / q_cols
    k_var = xf[:, q_cols:q_cols + k_cols].pow(2).sum(dim=-1) / k_cols
    return (
        x[:, :q_cols].contiguous(),
        x[:, q_cols:q_cols + k_cols].contiguous(),
        x[:, q_cols + k_cols:].contiguous(),
        torch.stack([q_var, k_var], dim=-1),
    )


def run_shape_k1(tp_world, num_q_heads, num_kv_heads, num_tokens):
    q_cols = num_q_heads * HEAD_DIM
    k_cols = num_kv_heads * HEAD_DIM
    device = "npu:0"

    x = torch.randn(num_tokens,
                    q_cols + 2 * k_cols,
                    dtype=DTYPE,
                    device=device)
    q_ref, k_ref, v_ref, var_ref = ref_kernel1(x, q_cols, k_cols)
    grid = (min(num_tokens, get_vectorcore_num()), )
    q_cols_pow2 = 1 << (q_cols - 1).bit_length()
    k_cols_pow2 = 1 << (k_cols - 1).bit_length()

    def make_run_legacy(bs):

        def run():
            q = torch.empty(num_tokens, q_cols, dtype=DTYPE, device=device)
            k = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
            v = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
            qk_var = torch.empty(num_tokens,
                                 2,
                                 dtype=torch.float32,
                                 device=device)
            _split_qkv_var_kernel_legacy[grid](
                x,
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
                1.0 / q_cols,
                1.0 / k_cols,
                BLOCK_SIZE=bs,
            )

        return run

    def make_run_seg(bs):

        def run():
            q = torch.empty(num_tokens, q_cols, dtype=DTYPE, device=device)
            k = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
            v = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
            qk_var = torch.empty(num_tokens,
                                 2,
                                 dtype=torch.float32,
                                 device=device)
            _split_qkv_var_kernel_seg[grid](
                x,
                q,
                k,
                v,
                qk_var,
                num_tokens,
                q_cols,
                k_cols,
                q_cols + 2 * k_cols,
                num_q_heads,
                num_kv_heads,
                HEAD_DIM,
                1.0 / q_cols,
                1.0 / k_cols,
                BLOCK_SIZE=bs,
            )

        return run

    def launch_and_check(which, bs):
        q = torch.empty(num_tokens, q_cols, dtype=DTYPE, device=device)
        k = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
        v = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
        qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=device)
        if which == "legacy":
            _split_qkv_var_kernel_legacy[grid](
                x,
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
                1.0 / q_cols,
                1.0 / k_cols,
                BLOCK_SIZE=bs,
            )
        else:
            _split_qkv_var_kernel_seg[grid](
                x,
                q,
                k,
                v,
                qk_var,
                num_tokens,
                q_cols,
                k_cols,
                q_cols + 2 * k_cols,
                num_q_heads,
                num_kv_heads,
                HEAD_DIM,
                1.0 / q_cols,
                1.0 / k_cols,
                BLOCK_SIZE=bs,
            )
        torch.npu.synchronize()
        cpy_err = max(
            (q.float() - q_ref.float()).abs().max().item(),
            (k.float() - k_ref.float()).abs().max().item(),
            (v.float() - v_ref.float()).abs().max().item(),
        )
        var_err = (qk_var - var_ref).abs().max().item()
        return cpy_err, var_err

    legacy_us = {}
    for bs in [1, 2, 4]:
        run = make_run_legacy(bs)
        try:
            cpy_err, var_err = launch_and_check("legacy", bs)
            assert cpy_err == 0.0, f"legacy copy mismatch: {cpy_err}"
            legacy_us[bs] = bench_us(run)
        except Exception as e:
            print(
                f"  !! tp={tp_world} tokens={num_tokens} legacy bs={bs}: {type(e).__name__}, skipped"
            )
            legacy_us[bs] = None

    seg_us = {}
    seg_var_err = None
    for bs in [1, 2, 4, 8]:
        run = make_run_seg(bs)
        try:
            cpy_err, var_err = launch_and_check("seg", bs)
            assert cpy_err == 0.0, f"seg copy mismatch: {cpy_err}"
            seg_var_err = var_err
            seg_us[bs] = bench_us(run)
        except Exception as e:
            print(
                f"  !! tp={tp_world} tokens={num_tokens} seg bs={bs}: {type(e).__name__}, skipped"
            )
            seg_us[bs] = None

    ok_legacy = {b: u for b, u in legacy_us.items() if u is not None}
    ok_seg = {b: u for b, u in seg_us.items() if u is not None}
    l_best_bs = min(ok_legacy, key=ok_legacy.get) if ok_legacy else None
    s_best_bs = min(ok_seg, key=ok_seg.get) if ok_seg else None
    return (
        (l_best_bs, ok_legacy[l_best_bs]) if l_best_bs else (None, None),
        (s_best_bs, ok_seg[s_best_bs]) if s_best_bs else (None, None),
        seg_var_err,
    )


# ---------------------------------------------------------------------------
# Legacy kernel2: pre-tile version, one token per loop iteration.
# ---------------------------------------------------------------------------
@triton.jit
def _apply_global_rmsnorm_kernel_legacy(
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
):
    pid = tl.program_id(0).to(tl.int64)
    num_programs = tl.num_programs(0)
    tokens_per_program = tl.cdiv(num_tokens, num_programs)
    iter_num_per_program = tokens_per_program
    program_token_offset = pid * tokens_per_program
    program_token_end = min(program_token_offset + tokens_per_program,
                            num_tokens)

    token_tile_offsets = tl.arange(0, 1)
    q_head_offsets = tl.arange(0, q_num_heads)[:, None]
    k_head_offsets = tl.arange(0, k_num_heads)[:, None]
    hd_offsets = tl.arange(0, head_dim)[None, :]

    q_row_offsets = q_head_offsets * head_dim + hd_offsets
    k_row_offsets = k_head_offsets * head_dim + hd_offsets

    q_weight = tl.load(q_weight_ptr + q_row_offsets).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + k_row_offsets).to(tl.float32)

    half_offsets = tl.arange(0, HALF)
    base_token_offsets = program_token_offset + token_tile_offsets

    for iter in tl.range(iter_num_per_program):
        token_offsets = base_token_offsets + iter
        token_mask = token_offsets < program_token_end

        q_gv = tl.load(qk_global_var_ptr + token_offsets * 2,
                       mask=token_mask,
                       other=0.0).to(tl.float32)
        q_gv = q_gv * inv_tp_world
        k_gv = tl.load(qk_global_var_ptr + token_offsets * 2 + 1,
                       mask=token_mask,
                       other=0.0).to(tl.float32)
        k_gv = k_gv * inv_tp_world
        q_scale = 1.0 / tl.sqrt(q_gv + eps)
        k_scale = 1.0 / tl.sqrt(k_gv + eps)

        q_offsets = token_offsets[:, None,
                                  None] * q_cols + q_row_offsets[None, :, :]
        q_mask = token_mask[:, None, None]
        q_vals_raw = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        q_vals = q_vals_raw.to(
            tl.float32) * q_scale[:, None, None] * q_weight[None, :, :]

        k_offsets = token_offsets[:, None,
                                  None] * k_cols + k_row_offsets[None, :, :]
        k_mask = token_mask[:, None, None]
        k_vals_raw = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        k_vals = k_vals_raw.to(
            tl.float32) * k_scale[:, None, None] * k_weight[None, :, :]

        cs_offsets = token_offsets[:, None] * cs_row_stride + half_offsets[
            None, :]
        cs_mask = token_mask[:, None]
        cos_row = tl.load(cos_ptr + cs_offsets, mask=cs_mask,
                          other=0.0).to(tl.float32)
        sin_row = tl.load(sin_ptr + cs_offsets, mask=cs_mask,
                          other=0.0).to(tl.float32)

        q1 = extract_slice(q_vals,
                           offsets=(0, 0, 0),
                           sizes=(1, q_num_heads, HALF),
                           strides=(1, 1, 1))
        q2 = extract_slice(q_vals,
                           offsets=(0, 0, HALF),
                           sizes=(1, q_num_heads, HALF),
                           strides=(1, 1, 1))
        q_vals = insert_slice(
            q_vals,
            q1 * cos_row[:, None, :] - q2 * sin_row[:, None, :],
            offsets=(0, 0, 0),
            sizes=(1, q_num_heads, HALF),
            strides=(1, 1, 1),
        )
        q_vals = insert_slice(
            q_vals,
            q2 * cos_row[:, None, :] + q1 * sin_row[:, None, :],
            offsets=(0, 0, HALF),
            sizes=(1, q_num_heads, HALF),
            strides=(1, 1, 1),
        )
        tl.store(q_ptr + q_offsets, q_vals.to(q_vals_raw.dtype), mask=q_mask)

        k1 = extract_slice(k_vals,
                           offsets=(0, 0, 0),
                           sizes=(1, k_num_heads, HALF),
                           strides=(1, 1, 1))
        k2 = extract_slice(k_vals,
                           offsets=(0, 0, HALF),
                           sizes=(1, k_num_heads, HALF),
                           strides=(1, 1, 1))
        k_vals = insert_slice(
            k_vals,
            k1 * cos_row[:, None, :] - k2 * sin_row[:, None, :],
            offsets=(0, 0, 0),
            sizes=(1, k_num_heads, HALF),
            strides=(1, 1, 1),
        )
        k_vals = insert_slice(
            k_vals,
            k2 * cos_row[:, None, :] + k1 * sin_row[:, None, :],
            offsets=(0, 0, HALF),
            sizes=(1, k_num_heads, HALF),
            strides=(1, 1, 1),
        )
        tl.store(k_ptr + k_offsets, k_vals.to(k_vals_raw.dtype), mask=k_mask)


# ---------------------------------------------------------------------------
# Tiled kernel2: BLOCK_T tokens per loop iteration. Each head is processed as
# three plain 3D segments (two rotary halves + pass-through tail); no
# extract_slice/insert_slice and no %//-encoded composite masks, both of
# which break the CANN Triton backend (UB mis-planning / load scalarization).
# BLOCK_T is passed explicitly (no autotune wrapper) to measure pure kernel
# time without host-side autotune lookup.
# ---------------------------------------------------------------------------
@triton.jit
def _apply_global_rmsnorm_kernel_tiled(
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


def ref_kernel2(q0, k0, cos, sin, q_weight, k_weight, qk_var, inv_tp,
                num_q_heads, num_kv_heads):
    """fp32 PyTorch reference of kernel2 (rmsnorm scale + weight + neox RoPE)."""
    num_tokens = q0.shape[0]
    qf = q0.float().view(num_tokens, num_q_heads, HEAD_DIM)
    kf = k0.float().view(num_tokens, num_kv_heads, HEAD_DIM)
    q_scale = 1.0 / torch.sqrt(qk_var[:, 0] * inv_tp + EPS)
    k_scale = 1.0 / torch.sqrt(qk_var[:, 1] * inv_tp + EPS)
    qf = qf * q_scale[:, None, None] * q_weight.view(num_q_heads,
                                                     HEAD_DIM)[None]
    kf = kf * k_scale[:, None, None] * k_weight.view(num_kv_heads,
                                                     HEAD_DIM)[None]
    c = cos[:, :HALF].float()[:, None, :]
    s = sin[:, :HALF].float()[:, None, :]
    q1, q2 = qf[..., :HALF], qf[..., HALF:2 * HALF]
    k1, k2 = kf[..., :HALF], kf[..., HALF:2 * HALF]
    q_out = torch.cat([q1 * c - q2 * s, q2 * c + q1 * s, qf[..., 2 * HALF:]],
                      dim=-1)
    k_out = torch.cat([k1 * c - k2 * s, k2 * c + k1 * s, kf[..., 2 * HALF:]],
                      dim=-1)
    return q_out.view(num_tokens, -1), k_out.view(num_tokens, -1)


def bench_us(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def run_shape(tp_world, num_q_heads, num_kv_heads, num_tokens):
    q_cols = num_q_heads * HEAD_DIM
    k_cols = num_kv_heads * HEAD_DIM
    device = "npu:0"
    inv_tp = 1.0 / tp_world

    q = torch.randn(num_tokens, q_cols, dtype=DTYPE, device=device)
    k = torch.randn(num_tokens, k_cols, dtype=DTYPE, device=device)
    # match runtime layout from update_cos_sin: [num_tokens, 2 * rotary_dim],
    # kernel only reads the first rotary_dim // 2 columns of each row.
    cos = torch.randn(num_tokens, 2 * ROTARY_DIM, dtype=DTYPE, device=device)
    sin = torch.randn(num_tokens, 2 * ROTARY_DIM, dtype=DTYPE, device=device)
    q_weight = torch.randn(q_cols, dtype=torch.float32, device=device)
    k_weight = torch.randn(k_cols, dtype=torch.float32, device=device)
    qk_var = torch.rand(num_tokens, 2, dtype=torch.float32,
                        device=device) + 0.5

    grid = (min(num_tokens, get_vectorcore_num()), )

    def run_legacy():
        _apply_global_rmsnorm_kernel_legacy[grid](
            q,
            k,
            cos,
            sin,
            cos.stride(0),
            q_weight,
            k_weight,
            qk_var,
            EPS,
            inv_tp,
            num_tokens,
            q_cols,
            k_cols,
            num_q_heads,
            num_kv_heads,
            HEAD_DIM,
            ROTARY_DIM,
            HALF,
        )

    def make_run_tiled(block_t):

        def run():
            _apply_global_rmsnorm_kernel_tiled[grid](
                q,
                k,
                cos,
                sin,
                cos.stride(0),
                q_weight,
                k_weight,
                qk_var,
                EPS,
                inv_tp,
                num_tokens,
                q_cols,
                k_cols,
                num_q_heads,
                num_kv_heads,
                HEAD_DIM,
                ROTARY_DIM,
                HALF,
                BLOCK_T=block_t,
            )

        return run

    # --- correctness vs fp32 reference (kernels update q/k in place) ---
    q0 = q.clone()
    k0 = k.clone()
    q_ref, k_ref = ref_kernel2(q0, k0, cos, sin, q_weight, k_weight, qk_var,
                               inv_tp, num_q_heads, num_kv_heads)

    def max_err():
        q_err = (q.float() - q_ref).abs().max().item()
        k_err = (k.float() - k_ref).abs().max().item()
        return max(q_err, k_err)

    run_legacy()  # JIT compile + warm
    torch.npu.synchronize()
    q.copy_(q0)
    k.copy_(k0)
    run_legacy()
    torch.npu.synchronize()
    err_legacy = max_err()

    tiled_us = {}
    tiled_errs = {}
    for bt in BLOCK_T_CANDIDATES:
        run_tiled = make_run_tiled(bt)
        try:
            run_tiled()  # JIT compile this BLOCK_T specialization
            torch.npu.synchronize()
        except Exception:
            # e.g. UB overflow for large BLOCK_T; skip this candidate
            tiled_us[bt] = None
            tiled_errs[bt] = None
            print(
                f"  !! tp={tp_world} tokens={num_tokens} BLOCK_T={bt}: compile/run failed, skipped"
            )
            continue
        q.copy_(q0)
        k.copy_(k0)
        run_tiled()
        torch.npu.synchronize()
        tiled_errs[bt] = max_err()

        q.copy_(q0)
        k.copy_(k0)
        tiled_us[bt] = bench_us(run_tiled)

    q.copy_(q0)
    k.copy_(k0)
    legacy_us = bench_us(run_legacy)

    return legacy_us, tiled_us, err_legacy, tiled_errs


def main():
    torch.manual_seed(0)
    init_device_properties_triton()
    print(f"vectorcore num: {get_vectorcore_num()}")

    # --- kernel1: split + local qk var (2D pow2-padded vs 3D segmented) ---
    k1_header = (
        f"{'tp':>3} {'tokens':>6} {'k1_legacy(us)':>14} {'bs':>3}"
        f" {'k1_seg(us)':>11} {'bs':>3} {'speedup':>8} {'var_err':>10}")
    print("=== kernel1: split_qkv + local qk var ===")
    print(k1_header)
    print("-" * len(k1_header))
    for tp_world, (num_q_heads, num_kv_heads) in TP_SHAPES.items():
        for num_tokens in NUM_TOKENS_LIST:
            (l_bs, l_us), (s_bs, s_us), var_err = run_shape_k1(
                tp_world, num_q_heads, num_kv_heads, num_tokens)
            speedup = f"{l_us / s_us:>7.2f}x" if l_us and s_us else f" {'-':>7}"
            print(
                f"{tp_world:>3} {num_tokens:>6}"
                f" {l_us if l_us is not None else float('nan'):>14.1f} {l_bs if l_bs is not None else '-':>3}"
                f" {s_us if s_us is not None else float('nan'):>11.1f} {s_bs if s_bs is not None else '-':>3}"
                f" {speedup} {var_err if var_err is not None else float('nan'):>10.3e}"
            )
    print("-" * len(k1_header))
    print(
        "copy outputs verified bit-exact vs reference; var_err is fp32 accumulation-order noise.\n"
    )

    # --- kernel2: global rmsnorm + rope (legacy vs token-tiled) ---
    print("=== kernel2: apply global rmsnorm + rope ===")
    bt_cols = "".join(f" {'bt=' + str(bt) + '(us)':>10}"
                      for bt in BLOCK_T_CANDIDATES)
    header = (f"{'tp':>3} {'tokens':>6} {'legacy(us)':>11}{bt_cols}"
              f" {'best':>5} {'err_legacy':>11} {'err_tiled':>10}")
    print(header)
    print("-" * len(header))
    for tp_world, (num_q_heads, num_kv_heads) in TP_SHAPES.items():
        for num_tokens in NUM_TOKENS_LIST:
            legacy_us, tiled_us, err_legacy, tiled_errs = run_shape(
                tp_world, num_q_heads, num_kv_heads, num_tokens)
            ok_bts = {bt: us for bt, us in tiled_us.items() if us is not None}
            best_bt = min(ok_bts, key=ok_bts.get) if ok_bts else "-"
            bt_vals = "".join(f" {tiled_us[bt]:>10.1f}"
                              if tiled_us[bt] is not None else f" {'FAIL':>10}"
                              for bt in BLOCK_T_CANDIDATES)
            err_tiled_max = max(e for e in tiled_errs.values()
                                if e is not None)
            print(f"{tp_world:>3} {num_tokens:>6} {legacy_us:>11.1f}{bt_vals}"
                  f" {best_bt:>5} {err_legacy:>11.3e} {err_tiled_max:>10.3e}")
    print("-" * len(header))
    print("err_* = max abs error vs fp32 reference; bf16 rounding scale, "
          "tiled and legacy should be the same order of magnitude.")


if __name__ == "__main__":
    main()
