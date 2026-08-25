# Benchmark for torch_npu.npu_dynamic_quant (MiniMax-M2.5 w8a8 shapes).
#
# Not collected by pytest (bench_ prefix). Run directly on an NPU machine:
#   python tests/e2e/nightly/single_node/ops/singlecard_ops/triton/bench_dynamic_quant.py
#
# This op runs in the MoE prepare stage (prepare_finalize.py, W8A8 path):
# hidden_states [num_tokens, 3072] bf16 -> int8 + per-token fp32 scale,
# right before the EP AllGather, halving the communication volume.
# Row counts cover decode (16) up to a merged 64K-token prefill step.
#
# Correctness: dequantized output vs an fp32 reference; round-to-nearest
# means max|err| should sit at half a quantization step (scale/2).

import torch
import torch_npu  # noqa: F401

DTYPE = torch.bfloat16
HIDDEN = 3072  # M2.5 hidden_size
NUM_TOKENS_LIST = [16, 128, 512, 2048, 8192, 16384, 65536]
WARMUP_ITERS = 10
BENCH_ITERS = 50


def ref_dynamic_quant(x):
    xf = x.float()
    scale = xf.abs().amax(dim=-1) / 127.0
    return xf, scale


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


def run_shape(num_tokens):
    device = "npu:0"
    x = torch.randn(num_tokens, HIDDEN, dtype=DTYPE, device=device)
    xf, scale_ref = ref_dynamic_quant(x)

    q, scale = torch_npu.npu_dynamic_quant(x)
    torch.npu.synchronize()
    err = (q.float() * scale[:, None].float() - xf).abs().max().item()
    quant_noise = (scale_ref.max() / 2).item()

    us = bench_us(lambda: torch_npu.npu_dynamic_quant(x))

    # read 3072 bf16, write 3072 int8 + 4B scale per token
    nbytes = num_tokens * (HIDDEN * 2 + HIDDEN + 4)
    gbs = nbytes / (us * 1e-6) / 1e9
    return us, gbs, err, quant_noise


def main():
    torch.manual_seed(0)
    print(
        f"shape: tokens x {HIDDEN} (bf16) -> tokens x {HIDDEN} (int8) + fp32 scale"
    )
    header = f"{'tokens':>6} {'time(us)':>9} {'GB/s':>7} {'err':>10} {'q_noise':>9}"
    print(header)
    print("-" * len(header))
    for num_tokens in NUM_TOKENS_LIST:
        us, gbs, err, qn = run_shape(num_tokens)
        print(
            f"{num_tokens:>6} {us:>9.1f} {gbs:>7.0f} {err:>10.3e} {qn:>9.3e}")
    print("-" * len(header))
    print("err = max abs error of dequantized output vs fp32 reference;")
    print("healthy when err is within ~q_noise (half a quantization step).")


if __name__ == "__main__":
    main()
