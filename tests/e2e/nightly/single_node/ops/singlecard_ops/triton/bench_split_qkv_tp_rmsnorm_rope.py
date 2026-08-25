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
# bt=1 is omitted: the single-token 2D specialization is pathologically slow
# (scalarized), and the legacy kernel already covers the BLOCK_T=1 baseline.
BLOCK_T_CANDIDATES = [2, 4, 8]

# MiniMax-M2.5: 48 q heads, 8 kv heads. Local heads per TP rank:
#   TP=4 -> 12 q + 2 kv;  TP=8 -> 6 q + 1 kv
TP_SHAPES = {4: (12, 2), 8: (6, 1)}

# decode: batch * (1 + spec tokens); prefill: chunked prefill token budget
NUM_TOKENS_LIST = [128, 256, 512, 2048, 8192, 16384]


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


# ---------------------------------------------------------------------------
# Tiled kernel2: BLOCK_T tokens per loop iteration, 2D layout.
# Rows are (token, head) pairs: row = local_token * num_heads + head, so all
# tiles are 2D and extract_slice/insert_slice stay on the 2D path (the CANN
# backend mis-plans UB for 3D slices with constexpr sizes -> ub overflow).
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
    program_token_end = min(program_token_offset + tokens_per_program, num_tokens)

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


def ref_kernel2(q0, k0, cos, sin, q_weight, k_weight, qk_var, inv_tp, num_q_heads, num_kv_heads):
    """fp32 PyTorch reference of kernel2 (rmsnorm scale + weight + neox RoPE)."""
    num_tokens = q0.shape[0]
    qf = q0.float().view(num_tokens, num_q_heads, HEAD_DIM)
    kf = k0.float().view(num_tokens, num_kv_heads, HEAD_DIM)
    q_scale = 1.0 / torch.sqrt(qk_var[:, 0] * inv_tp + EPS)
    k_scale = 1.0 / torch.sqrt(qk_var[:, 1] * inv_tp + EPS)
    qf = qf * q_scale[:, None, None] * q_weight.view(num_q_heads, HEAD_DIM)[None]
    kf = kf * k_scale[:, None, None] * k_weight.view(num_kv_heads, HEAD_DIM)[None]
    c = cos[:, :HALF].float()[:, None, :]
    s = sin[:, :HALF].float()[:, None, :]
    q1, q2 = qf[..., :HALF], qf[..., HALF : 2 * HALF]
    k1, k2 = kf[..., :HALF], kf[..., HALF : 2 * HALF]
    q_out = torch.cat([q1 * c - q2 * s, q2 * c + q1 * s, qf[..., 2 * HALF :]], dim=-1)
    k_out = torch.cat([k1 * c - k2 * s, k2 * c + k1 * s, kf[..., 2 * HALF :]], dim=-1)
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
    qk_var = torch.rand(num_tokens, 2, dtype=torch.float32, device=device) + 0.5

    grid = (min(num_tokens, get_vectorcore_num()),)

    def run_legacy():
        _apply_global_rmsnorm_kernel_legacy[grid](
            q, k, cos, sin, cos.stride(0), q_weight, k_weight, qk_var,
            EPS, inv_tp, num_tokens, q_cols, k_cols,
            num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM, HALF,
        )

    def make_run_tiled(block_t):
        def run():
            _apply_global_rmsnorm_kernel_tiled[grid](
                q, k, cos, sin, cos.stride(0), q_weight, k_weight, qk_var,
                EPS, inv_tp, num_tokens, q_cols, k_cols,
                num_q_heads, num_kv_heads, HEAD_DIM, ROTARY_DIM, HALF,
                BLOCK_T=block_t,
            )

        return run

    # --- correctness vs fp32 reference (kernels update q/k in place) ---
    q0 = q.clone()
    k0 = k.clone()
    q_ref, k_ref = ref_kernel2(q0, k0, cos, sin, q_weight, k_weight, qk_var, inv_tp, num_q_heads, num_kv_heads)

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
            print(f"  !! tp={tp_world} tokens={num_tokens} BLOCK_T={bt}: compile/run failed, skipped")
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
    bt_cols = "".join(f" {'bt=' + str(bt) + '(us)':>10}" for bt in BLOCK_T_CANDIDATES)
    header = (
        f"{'tp':>3} {'tokens':>6} {'legacy(us)':>11}{bt_cols}"
        f" {'best':>5} {'err_legacy':>11} {'err_tiled':>10}"
    )
    print(header)
    print("-" * len(header))
    for tp_world, (num_q_heads, num_kv_heads) in TP_SHAPES.items():
        for num_tokens in NUM_TOKENS_LIST:
            legacy_us, tiled_us, err_legacy, tiled_errs = run_shape(
                tp_world, num_q_heads, num_kv_heads, num_tokens
            )
            ok_bts = {bt: us for bt, us in tiled_us.items() if us is not None}
            best_bt = min(ok_bts, key=ok_bts.get) if ok_bts else "-"
            bt_vals = "".join(
                f" {tiled_us[bt]:>10.1f}" if tiled_us[bt] is not None else f" {'FAIL':>10}"
                for bt in BLOCK_T_CANDIDATES
            )
            err_tiled_max = max(e for e in tiled_errs.values() if e is not None)
            print(
                f"{tp_world:>3} {num_tokens:>6} {legacy_us:>11.1f}{bt_vals}"
                f" {best_bt:>5} {err_legacy:>11.3e} {err_tiled_max:>10.3e}"
            )
    print("-" * len(header))
    print("err_* = max abs error vs fp32 reference; bf16 rounding scale, "
          "tiled and legacy should be the same order of magnitude.")


if __name__ == "__main__":
    main()
