# Benchmark for _apply_global_rmsnorm_kernel (MiniMax-M2.5 shapes).
#
# Not collected by pytest (bench_ prefix). Run directly on an NPU machine:
#   python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/bench_split_qkv_tp_rmsnorm_rope.py
#
# Measures:
#   1. kernel2 (_apply_global_rmsnorm_kernel) alone
#   2. the full split_qkv_tp_rmsnorm_rope custom op (kernel1 + kernel2) for reference
# Reports per-call latency and effective HBM bandwidth of kernel2.

import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401  (registers torch.ops.vllm.split_qkv_tp_rmsnorm_rope)
from vllm_ascend.ops.triton.linearnorm.split_qkv_tp_rmsnorm_rope import (
    _apply_global_rmsnorm_kernel,
)
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

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

    def run_kernel2():
        _apply_global_rmsnorm_kernel[grid](
            q,
            k,
            cos,
            sin,
            cos.stride(0),
            q_weight,
            k_weight,
            qk_var,
            EPS,
            1.0 / tp_world,
            num_tokens,
            q_cols,
            k_cols,
            num_q_heads,
            num_kv_heads,
            HEAD_DIM,
            ROTARY_DIM,
            ROTARY_DIM // 2,
        )

    kernel2_us = bench_us(run_kernel2)

    qkv = torch.randn(num_tokens, q_cols + 2 * k_cols, dtype=DTYPE, device=device)
    cos_half = cos[:, : ROTARY_DIM // 2].contiguous()
    sin_half = sin[:, : ROTARY_DIM // 2].contiguous()

    def run_full_op():
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

    full_op_us = bench_us(run_full_op)

    moved_bytes = num_tokens * (2 * (q_cols + k_cols) * q.element_size() + ROTARY_DIM * q.element_size() + 8)
    bw_gbps = moved_bytes / (kernel2_us * 1e-6) / 1e9
    return kernel2_us, full_op_us, bw_gbps


def main():
    torch.manual_seed(0)
    init_device_properties_triton()
    print(f"vectorcore num: {get_vectorcore_num()}")
    header = f"{'tp':>3} {'tokens':>6} {'kernel2(us)':>12} {'full_op(us)':>12} {'kernel2 GB/s':>13}"
    print(header)
    print("-" * len(header))
    for tp_world, (num_q_heads, num_kv_heads) in TP_SHAPES.items():
        for num_tokens in NUM_TOKENS_LIST:
            kernel2_us, full_op_us, bw_gbps = run_shape(tp_world, num_q_heads, num_kv_heads, num_tokens)
            print(f"{tp_world:>3} {num_tokens:>6} {kernel2_us:>12.1f} {full_op_us:>12.1f} {bw_gbps:>13.0f}")


if __name__ == "__main__":
    main()
