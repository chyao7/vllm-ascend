# Benchmark for swiglu_quant (MiniMax-M2.5 w8a8 MoE shapes).
#
# Not collected by pytest (bench_ prefix). Run directly on an NPU machine:
#   python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/bench_swiglu_quant.py
#
# Compares the fused Triton swiglu_quant against the CANN two-op baseline
# (npu_swiglu + npu_dynamic_quant, the non-Triton fallback in moe_mlp.py),
# on the per-rank row counts the ALLGATHER EP8 path actually sees:
#   decode: 16 reqs * top6 = 96 rows;  prefill chunk: tokens * top6 / EP8.
# Row width is 2 * moe_intermediate_size = 3072 (non-power-of-2 on purpose).
#
# Correctness: dequantized outputs are compared against an fp32 reference.
# The Triton kernel truncates toward zero on int8 cast while CANN rounds to
# nearest, so a ~1-quantum difference between the two is expected and fine;
# both should sit at the quantization-noise scale (max|err| ~ scale/2).

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant
from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)

DTYPE = torch.bfloat16
MOE_INTER = 1536  # M2.5 moe_intermediate_size; gate_up output width = 2x
TOTAL_COLS = 2 * MOE_INTER
LOCAL_EXPERTS = 32  # 256 experts / EP8
NUM_ROWS_LIST = [96, 512, 3072, 6144, 12288, 24576, 49152]
WARMUP_ITERS = 10
BENCH_ITERS = 50


def ref_swiglu_quant(x):
    """fp32 reference: swiglu then per-row dynamic int8 quantization."""
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    sw = x1 * torch.sigmoid(x1) * x2
    scale = sw.abs().amax(dim=-1) / 127.0
    return sw, scale


def cann_swiglu_quant(x):
    """CANN two-op baseline (moe_mlp.py non-Triton fallback)."""
    sw = torch_npu.npu_swiglu(x)
    q, scale = torch_npu.npu_dynamic_quant(sw)
    return q, scale


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


def make_group_list(num_rows, device):
    """Evenly spread rows over LOCAL_EXPERTS experts (count form, type=1)."""
    base = num_rows // LOCAL_EXPERTS
    rem = num_rows % LOCAL_EXPERTS
    counts = [base + (1 if i < rem else 0) for i in range(LOCAL_EXPERTS)]
    return torch.tensor(counts, dtype=torch.int64, device=device)


def run_shape(num_rows):
    device = "npu:0"
    x = torch.randn(num_rows, TOTAL_COLS, dtype=DTYPE, device=device)
    group_list = make_group_list(num_rows, device)

    sw_ref, scale_ref = ref_swiglu_quant(x)

    def dequant_err(q, scale):
        return (q.float() * scale[:, None].float() - sw_ref).abs().max().item()

    # --- correctness ---
    q_t, s_t = swiglu_quant(x, group_list=group_list, group_list_type=1)
    torch.npu.synchronize()
    err_triton = dequant_err(q_t, s_t)

    q_c, s_c = cann_swiglu_quant(x)
    torch.npu.synchronize()
    err_cann = dequant_err(q_c, s_c)

    quant_noise = (scale_ref.max() / 2).item()

    # --- timing ---
    triton_us = bench_us(
        lambda: swiglu_quant(x, group_list=group_list, group_list_type=1))
    cann_us = bench_us(lambda: cann_swiglu_quant(x))

    # Triton fused: read 3072 bf16, write 1536 int8 + 4B scale per row.
    triton_bytes = num_rows * (TOTAL_COLS * 2 + MOE_INTER + 4)
    # CANN: swiglu reads 3072 bf16 writes 1536 bf16; quant reads it back and
    # writes 1536 int8 + 4B scale.
    cann_bytes = num_rows * (TOTAL_COLS * 2 + MOE_INTER * 2 * 2 + MOE_INTER +
                             4)
    triton_gbs = triton_bytes / (triton_us * 1e-6) / 1e9
    cann_gbs = cann_bytes / (cann_us * 1e-6) / 1e9

    return triton_us, cann_us, triton_gbs, cann_gbs, err_triton, err_cann, quant_noise


def main():
    torch.manual_seed(0)
    init_device_properties_triton()
    print(f"vectorcore num: {get_vectorcore_num()}")
    print(
        f"shape: rows x {TOTAL_COLS} (bf16) -> rows x {MOE_INTER} (int8) + fp32 scale"
    )
    header = (
        f"{'rows':>6} {'triton(us)':>11} {'cann(us)':>9} {'speedup':>8}"
        f" {'triton GB/s':>12} {'cann GB/s':>10} {'err_triton':>11} {'err_cann':>9} {'q_noise':>8}"
    )
    print(header)
    print("-" * len(header))
    for num_rows in NUM_ROWS_LIST:
        triton_us, cann_us, triton_gbs, cann_gbs, err_t, err_c, qn = run_shape(
            num_rows)
        print(
            f"{num_rows:>6} {triton_us:>11.1f} {cann_us:>9.1f} {cann_us / triton_us:>7.2f}x"
            f" {triton_gbs:>12.0f} {cann_gbs:>10.0f} {err_t:>11.3e} {err_c:>9.3e} {qn:>8.3e}"
        )
    print("-" * len(header))
    print("err_* = max abs error of dequantized output vs fp32 reference;")
    print(
        "q_noise = half quantization step. err within ~2x q_noise is healthy.")


if __name__ == "__main__":
    main()
