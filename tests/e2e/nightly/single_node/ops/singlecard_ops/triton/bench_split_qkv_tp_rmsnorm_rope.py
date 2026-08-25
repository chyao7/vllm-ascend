# Benchmark for _apply_global_rmsnorm_kernel (MiniMax-M2.5 shapes),
# before/after comparison for the token-tiling change (BLOCK_T autotune).
#
# Not collected by pytest (bench_ prefix). Run directly on an NPU machine:
#   python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/bench_split_qkv_tp_rmsnorm_rope.py
#
# Measures per shape:
#   1. legacy kernel2 (single token per iteration, pre-change baseline)
#   2. tiled kernel2 (BLOCK_T autotuned, current implementation)
#   3. full op with legacy kernel2 (kernel1 + legacy kernel2, no custom-op dispatch)
#   4. the full split_qkv_tp_rmsnorm_rope custom op (kernel1 + tiled kernel2)
#   5. native CANN ops: split + npu_rms_norm + npu_rotary_mul + cat
# Also verifies bit-exact equality between legacy and tiled kernel2 outputs.

import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401  (registers torch.ops.vllm.split_qkv_tp_rmsnorm_rope)
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope import (
    _apply_global_rmsnorm_kernel,
    _split_qkv_and_compute_local_qk_var_kernel,
)
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
WARMUP_ITERS = 10
BENCH_ITERS = 50

# MiniMax-M2.5: 48 q heads, 8 kv heads. Local heads per TP rank:
#   TP=4 -> 12 q + 2 kv;  TP=8 -> 6 q + 1 kv
TP_SHAPES = {4: (12, 2), 8: (6, 1)}

# decode: batch * (1 + spec tokens); prefill: chunked prefill token budget
NUM_TOKENS_LIST = [128, 256, 512, 2048, 8192, 16384]


# ---------------------------------------------------------------------------
# Legacy kernel2: pre-tile version, one token per loop iteration.
# Kept here verbatim as the before-change baseline.
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
    program_token_end = min(program_token_offset + tokens_per_program, num_tokens)

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

        q_gv = tl.load(qk_global_var_ptr + token_offsets * 2, mask=token_mask, other=0.0).to(tl.float32)
        q_gv = q_gv * inv_tp_world
        k_gv = tl.load(qk_global_var_ptr + token_offsets * 2 + 1, mask=token_mask, other=0.0).to(tl.float32)
        k_gv = k_gv * inv_tp_world
        q_scale = 1.0 / tl.sqrt(q_gv + eps)
        k_scale = 1.0 / tl.sqrt(k_gv + eps)

        q_offsets = token_offsets[:, None, None] * q_cols + q_row_offsets[None, :, :]
        q_mask = token_mask[:, None, None]
        q_vals_raw = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        q_vals = q_vals_raw.to(tl.float32) * q_scale[:, None, None] * q_weight[None, :, :]

        k_offsets = token_offsets[:, None, None] * k_cols + k_row_offsets[None, :, :]
        k_mask = token_mask[:, None, None]
        k_vals_raw = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        k_vals = k_vals_raw.to(tl.float32) * k_scale[:, None, None] * k_weight[None, :, :]

        cs_offsets = token_offsets[:, None] * cs_row_stride + half_offsets[None, :]
        cs_mask = token_mask[:, None]
        cos_row = tl.load(cos_ptr + cs_offsets, mask=cs_mask, other=0.0).to(tl.float32)
        sin_row = tl.load(sin_ptr + cs_offsets, mask=cs_mask, other=0.0).to(tl.float32)

        q1 = extract_slice(q_vals, offsets=(0, 0, 0), sizes=(1, q_num_heads, HALF), strides=(1, 1, 1))
        q2 = extract_slice(q_vals, offsets=(0, 0, HALF), sizes=(1, q_num_heads, HALF), strides=(1, 1, 1))
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

        k1 = extract_slice(k_vals, offsets=(0, 0, 0), sizes=(1, k_num_heads, HALF), strides=(1, 1, 1))
        k2 = extract_slice(k_vals, offsets=(0, 0, HALF), sizes=(1, k_num_heads, HALF), strides=(1, 1, 1))
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

    q = torch.randn(num_tokens, q_cols, dtype=DTYPE, device=device)
    k = torch.randn(num_tokens, k_cols, dtype=DTYPE, device=device)
    # match runtime layout from update_cos_sin: [num_tokens, 2 * rotary_dim],
    # kernel only reads the first rotary_dim // 2 columns of each row.
    cos = torch.randn(num_tokens, 2 * ROTARY_DIM, dtype=DTYPE, device=device)
    sin = torch.randn(num_tokens, 2 * ROTARY_DIM, dtype=DTYPE, device=device)
    q_weight = torch.randn(q_cols, dtype=torch.float32, device=device)
    k_weight = torch.randn(k_cols, dtype=torch.float32, device=device)
    qk_var = torch.rand(num_tokens, 2, dtype=torch.float32, device=device) + 0.5

    grid = (min(num_tokens, get_vectorcore_num()),)

    def run_kernel2_new():
        _apply_global_rmsnorm_kernel[grid](
            q, k, cos, sin, cos.stride(0), q_weight, k_weight, qk_var,
            EPS, 1.0 / tp_world, num_tokens, q_cols, k_cols,
            num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM, ROTARY_DIM // 2,
        )

    def run_kernel2_legacy():
        _apply_global_rmsnorm_kernel_legacy[grid](
            q, k, cos, sin, cos.stride(0), q_weight, k_weight, qk_var,
            EPS, 1.0 / tp_world, num_tokens, q_cols, k_cols,
            num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM, ROTARY_DIM // 2,
        )

    # --- correctness: tiled kernel must be bit-exact vs legacy ---
    # Both kernels update q/k in place, so snapshot inputs and restore
    # between runs. The new kernel is autotuned; warm it up first so the
    # autotune search does not pollute the compared outputs.
    q0 = q.clone()
    k0 = k.clone()

    run_kernel2_legacy()
    torch.npu.synchronize()
    q_ref = q.clone()
    k_ref = k.clone()

    run_kernel2_new()  # triggers autotune on first call; output discarded
    torch.npu.synchronize()
    q.copy_(q0)
    k.copy_(k0)
    run_kernel2_new()
    torch.npu.synchronize()

    q_ok = torch.equal(q, q_ref)
    k_ok = torch.equal(k, k_ref)
    if not (q_ok and k_ok):
        q_diff = (q.float() - q_ref.float()).abs().max().item()
        k_diff = (k.float() - k_ref.float()).abs().max().item()
        print(f"  !! MISMATCH tp={tp_world} tokens={num_tokens}: "
              f"q_ok={q_ok} (max|d|={q_diff:.3e}) k_ok={k_ok} (max|d|={k_diff:.3e})")
    correct = q_ok and k_ok

    # --- perf ---
    q.copy_(q0)
    k.copy_(k0)
    legacy_us = bench_us(run_kernel2_legacy)
    q.copy_(q0)
    k.copy_(k0)
    new_us = bench_us(run_kernel2_new)

    # --- full op: kernel1 + kernel2 ---
    qkv = torch.randn(num_tokens, q_cols + 2 * k_cols, dtype=DTYPE, device=device)
    cos_half = cos[:, : ROTARY_DIM // 2].contiguous()
    sin_half = sin[:, : ROTARY_DIM // 2].contiguous()
    q_cols_pow2 = 1 << (q_cols - 1).bit_length()
    k_cols_pow2 = 1 << (k_cols - 1).bit_length()

    q_buf = torch.empty(num_tokens, q_cols, dtype=DTYPE, device=device)
    k_buf = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
    v_buf = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
    var_buf = torch.empty(num_tokens, 2, dtype=torch.float32, device=device)

    def run_full_op_legacy():
        _split_qkv_and_compute_local_qk_var_kernel[grid](
            qkv, q_buf, k_buf, v_buf, var_buf, num_tokens,
            q_cols, k_cols, q_cols_pow2, k_cols_pow2,
            q_cols + 2 * k_cols, 1.0 / q_cols, 1.0 / k_cols,
        )
        _apply_global_rmsnorm_kernel_legacy[grid](
            q_buf, k_buf, cos_half, sin_half, cos_half.stride(0),
            q_weight, k_weight, var_buf,
            EPS, 1.0, num_tokens, q_cols, k_cols,
            num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM, ROTARY_DIM // 2,
        )

    full_legacy_us = bench_us(run_full_op_legacy)

    def run_full_op_new():
        torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
            input=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_cols,
            kv_hidden_size=k_cols,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
            tp_world=1,
            cos=cos_half,
            sin=sin_half,
        )

    full_new_us = bench_us(run_full_op_new)

    # Native CANN path: split + rms_norm (full-hidden, tp_world=1) + rotary_mul + cat.
    q_weight_npu = torch.randn(q_cols, dtype=DTYPE, device=device)
    k_weight_npu = torch.randn(k_cols, dtype=DTYPE, device=device)
    cos_full = torch.cat((cos_half, cos_half), dim=-1).reshape(1, num_tokens, 1, ROTARY_DIM).contiguous()
    sin_full = torch.cat((sin_half, sin_half), dim=-1).reshape(1, num_tokens, 1, ROTARY_DIM).contiguous()

    def run_native():
        qn, kn, _ = torch.split(qkv, [q_cols, k_cols, k_cols], dim=-1)
        qn, _ = torch.ops.npu.npu_rms_norm(qn, q_weight_npu, EPS)
        kn, _ = torch.ops.npu.npu_rms_norm(kn, k_weight_npu, EPS)
        # npu_rotary_mul requires BSHD 4D layout to broadcast with cos/sin.
        q4 = qn.view(1, num_tokens, num_q_heads, HEAD_DIM)
        k4 = kn.view(1, num_tokens, num_kv_heads, HEAD_DIM)
        q_rot = torch_npu.npu_rotary_mul(q4[..., :ROTARY_DIM], cos_full, sin_full)
        k_rot = torch_npu.npu_rotary_mul(k4[..., :ROTARY_DIM], cos_full, sin_full)
        torch.cat((q_rot, q4[..., ROTARY_DIM:]), dim=-1)
        torch.cat((k_rot, k4[..., ROTARY_DIM:]), dim=-1)

    native_us = bench_us(run_native)

    moved_bytes = num_tokens * (2 * (q_cols + k_cols) * q.element_size() + ROTARY_DIM * q.element_size() + 8)
    bw_gbps = moved_bytes / (new_us * 1e-6) / 1e9
    speedup = legacy_us / new_us
    return legacy_us, new_us, speedup, bw_gbps, full_legacy_us, full_new_us, native_us, correct


def main():
    torch.manual_seed(0)
    init_device_properties_triton()
    print(f"vectorcore num: {get_vectorcore_num()}")
    header = (
        f"{'tp':>3} {'tokens':>6} {'k2_old(us)':>11} {'k2_new(us)':>11} {'speedup':>8}"
        f" {'k2_new GB/s':>12} {'full_old(us)':>13} {'full_new(us)':>13} {'native(us)':>11} {'exact':>6}"
    )
    print(header)
    print("-" * len(header))
    all_ok = True
    for tp_world, (num_q_heads, num_kv_heads) in TP_SHAPES.items():
        for num_tokens in NUM_TOKENS_LIST:
            legacy_us, new_us, speedup, bw_gbps, full_legacy_us, full_new_us, native_us, correct = run_shape(
                tp_world, num_q_heads, num_kv_heads, num_tokens
            )
            all_ok = all_ok and correct
            print(
                f"{tp_world:>3} {num_tokens:>6} {legacy_us:>11.1f} {new_us:>11.1f} {speedup:>7.2f}x"
                f" {bw_gbps:>12.0f} {full_legacy_us:>13.1f} {full_new_us:>13.1f} {native_us:>11.1f}"
                f" {'OK' if correct else 'FAIL':>6}"
            )
    print("-" * len(header))
    print(f"bit-exact vs legacy: {'ALL OK' if all_ok else 'MISMATCH FOUND'}")


if __name__ == "__main__":
    main()
