# Benchmark: KV-cache-fused split_qkv_tp_rmsnorm_rope (MiniMax-M2.5 shapes).
#
# Not collected by pytest (bench_ prefix). Run directly on an NPU machine:
#   python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/bench_split_qkv_tp_rmsnorm_rope_cache.py
#
# Legacy path (current M2.5 attention forward):
#   split_qkv_tp_rmsnorm_rope -> q/k/v contiguous buffers, then
#   npu_scatter_pa_kv_cache reads k/v back and scatters them into the
#   paged ND cache [num_blocks, block_size, num_kv_heads, head_dim].
# Fused path:
#   split_qkv_tp_rmsnorm_rope_cache scatters v (kernel1) and the
#   normed+roped k (kernel2) straight into the cache via slot_mapping,
#   so the contiguous k/v buffers and the scatter kernel disappear.
#
# The fused path must reproduce the legacy cache contents bit-exactly
# (pure data movement) and q bit-exactly (identical compute path).

import torch
import torch_npu  # noqa: F401
import vllm_ascend.ops  # noqa: F401  registers the custom ops

from vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope import (
    _apply_global_rmsnorm_cache_kv_kernel,
    _apply_global_rmsnorm_kernel,
    _split_qkv_and_compute_local_qk_var_kernel,
)
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

DTYPE = torch.bfloat16
HEAD_DIM = 128
ROTARY_DIM = 64
BLOCK_SIZE = 128
# (tp label, q heads per rank, kv heads per rank) for MiniMax-M2.5
TP_SHAPES = [(8, 6, 1), (4, 12, 2)]
NUM_TOKENS_LIST = [128, 256, 512, 2048, 8192, 16384]
WARMUP_ITERS = 10
BENCH_ITERS = 50
SENTINEL = 7.0
EPS = 1e-6


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


def run_case(tp, q_heads, kv_heads, num_tokens):
    device = "npu:0"
    q_cols = q_heads * HEAD_DIM
    k_cols = kv_heads * HEAD_DIM

    qkv = torch.randn(num_tokens, q_cols + 2 * k_cols, dtype=DTYPE, device=device)
    q_weight = torch.randn(q_cols, dtype=torch.float32, device=device) * 0.1 + 1.0
    k_weight = torch.randn(k_cols, dtype=torch.float32, device=device) * 0.1 + 1.0
    cos = torch.rand(num_tokens, ROTARY_DIM // 2, dtype=DTYPE, device=device)
    sin = torch.rand(num_tokens, ROTARY_DIM // 2, dtype=DTYPE, device=device)

    # Paged ND caches, twice the needed slots, scattered unique slot ids.
    num_blocks = 2 * ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) + 8
    cache_shape = (num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM)
    k_cache_legacy = torch.full(cache_shape, SENTINEL, dtype=DTYPE, device=device)
    v_cache_legacy = torch.full(cache_shape, SENTINEL, dtype=DTYPE, device=device)
    k_cache_fused = torch.full(cache_shape, SENTINEL, dtype=DTYPE, device=device)
    v_cache_fused = torch.full(cache_shape, SENTINEL, dtype=DTYPE, device=device)
    slot_mapping = torch.randperm(num_blocks * BLOCK_SIZE, device=device)[:num_tokens].to(torch.int32)

    def legacy():
        q, k, v = torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
            input=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_cols,
            kv_hidden_size=k_cols,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
            tp_world=1,
            cos=cos,
            sin=sin,
        )
        torch_npu.npu_scatter_pa_kv_cache(
            key=k.view(num_tokens, kv_heads, HEAD_DIM),
            value=v.view(num_tokens, kv_heads, HEAD_DIM),
            key_cache=k_cache_legacy,
            value_cache=v_cache_legacy,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )
        return q

    def fused():
        return torch.ops.vllm.split_qkv_tp_rmsnorm_rope_cache(
            input=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_cols,
            kv_hidden_size=k_cols,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
            tp_world=1,
            cos=cos,
            sin=sin,
            k_cache=k_cache_fused,
            v_cache=v_cache_fused,
            slot_mapping=slot_mapping,
        )

    q_legacy = legacy()
    q_fused = fused()
    torch.npu.synchronize()

    q_exact = torch.equal(q_legacy, q_fused)
    k_exact = torch.equal(k_cache_legacy, k_cache_fused)
    v_exact = torch.equal(v_cache_legacy, v_cache_fused)
    exact = q_exact and k_exact and v_exact

    t_fused = bench_us(fused)
    t_legacy = bench_us(legacy)
    t_scatter = bench_us(
        lambda: torch_npu.npu_scatter_pa_kv_cache(
            key=torch.empty(num_tokens, kv_heads, HEAD_DIM, dtype=DTYPE, device=device),
            value=torch.empty(num_tokens, kv_heads, HEAD_DIM, dtype=DTYPE, device=device),
            key_cache=k_cache_legacy,
            value_cache=v_cache_legacy,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        ))

    # Bytes the fused path skips: contiguous k/v write + read-back.
    saved_kb = 2 * (num_tokens * k_cols * 2) * 2 / 1024
    speedup = t_legacy / t_fused
    return t_fused, t_legacy, speedup, t_scatter, saved_kb, exact


def run_kernel_breakdown(tp, q_heads, kv_heads, num_tokens):
    """Per-kernel timing: kernel1 is shared by both paths; k2_legacy stores K
    contiguously while k2_fused scatters K and forwards V into the caches."""
    device = "npu:0"
    q_cols = q_heads * HEAD_DIM
    k_cols = kv_heads * HEAD_DIM

    qkv = torch.randn(num_tokens, q_cols + 2 * k_cols, dtype=DTYPE, device=device)
    q_weight = torch.randn(q_cols, dtype=torch.float32, device=device) * 0.1 + 1.0
    k_weight = torch.randn(k_cols, dtype=torch.float32, device=device) * 0.1 + 1.0
    cos = torch.rand(num_tokens, ROTARY_DIM // 2, dtype=DTYPE, device=device)
    sin = torch.rand(num_tokens, ROTARY_DIM // 2, dtype=DTYPE, device=device)

    num_blocks = 2 * ((num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) + 8
    cache_shape = (num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM)
    k_cache = torch.full(cache_shape, SENTINEL, dtype=DTYPE, device=device)
    v_cache = torch.full(cache_shape, SENTINEL, dtype=DTYPE, device=device)
    slot_mapping = torch.randperm(num_blocks * BLOCK_SIZE, device=device)[:num_tokens].to(torch.int32)

    q = torch.empty(num_tokens, q_cols, dtype=DTYPE, device=device)
    k = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
    v = torch.empty(num_tokens, k_cols, dtype=DTYPE, device=device)
    qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=device)

    grid = (min(num_tokens, get_vectorcore_num()),)
    q_cols_pow2 = 1 << (q_cols - 1).bit_length()
    k_cols_pow2 = 1 << (k_cols - 1).bit_length()
    q_inv = 1.0 / q_cols
    k_inv = 1.0 / k_cols

    def k1_legacy():
        _split_qkv_and_compute_local_qk_var_kernel[grid](
            qkv, q, k, v, qk_var, num_tokens, q_cols, k_cols, q_cols_pow2, k_cols_pow2,
            q_cols + 2 * k_cols, q_inv, k_inv)

    def k2_legacy():
        _apply_global_rmsnorm_kernel[grid](
            q, k, cos, sin, cos.stride(0), q_weight, k_weight, qk_var, EPS, 1.0,
            num_tokens, q_cols, k_cols, q_heads, kv_heads, HEAD_DIM, ROTARY_DIM,
            ROTARY_DIM // 2)

    def k2_fused():
        _apply_global_rmsnorm_cache_kv_kernel[grid](
            q, k, v, cos, sin, cos.stride(0), q_weight, k_weight, qk_var, k_cache,
            v_cache, slot_mapping, EPS, 1.0, num_tokens, q_cols, k_cols, q_heads,
            kv_heads, HEAD_DIM, ROTARY_DIM, ROTARY_DIM // 2)

    t_k1 = bench_us(k1_legacy)
    t_k2_legacy = bench_us(k2_legacy)
    t_k2_fused = bench_us(k2_fused)
    return t_k1, t_k2_legacy, t_k2_fused


def main():
    torch.manual_seed(0)
    init_device_properties_triton()
    print(f"vectorcore num: {get_vectorcore_num()}")
    header = (f"{'tp':>3} {'tokens':>6} {'fused(us)':>10} {'legacy(us)':>10} {'speedup':>8}"
              f" {'scatter(us)':>11} {'saved_kb':>9} {'exact':>6}")
    print(header)
    print("-" * len(header))
    for tp, q_heads, kv_heads in TP_SHAPES:
        for num_tokens in NUM_TOKENS_LIST:
            t_fused, t_legacy, speedup, t_scatter, saved_kb, exact = run_case(
                tp, q_heads, kv_heads, num_tokens)
            print(f"{tp:>3} {num_tokens:>6} {t_fused:>10.1f} {t_legacy:>10.1f}"
                  f" {speedup:>7.2f}x {t_scatter:>11.1f} {saved_kb:>9.0f}"
                  f" {'OK' if exact else 'FAIL':>6}")
    print("-" * len(header))
    print("legacy = split_qkv_tp_rmsnorm_rope + npu_scatter_pa_kv_cache;")
    print("exact  = q and both paged caches bit-exact vs the legacy path.")

    print("\n=== per-kernel breakdown (locate the scatter-store cost) ===")
    header2 = (f"{'tp':>3} {'tokens':>6} {'k1(us)':>8} {'k2_leg':>8} {'k2_fus':>8} {'k2_dlt':>8}")
    print(header2)
    print("-" * len(header2))
    for tp, q_heads, kv_heads in TP_SHAPES:
        for num_tokens in NUM_TOKENS_LIST:
            t1, t2l, t2f = run_kernel_breakdown(tp, q_heads, kv_heads, num_tokens)
            print(f"{tp:>3} {num_tokens:>6} {t1:>8.1f} {t2l:>8.1f} {t2f:>8.1f} {t2f - t2l:>+8.1f}")
    print("-" * len(header2))
    print("k1 = split + local var (shared by both paths)")
    print("k2 = global rmsnorm + rope; fused also scatters K and forwards V into cache")
    print("dlt > 0 is the in-kernel scatter cost; compare against scatter(us) above.")


if __name__ == "__main__":
    main()
