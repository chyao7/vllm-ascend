#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone bench: ``torch_npu`` vs ``_C_ascend`` kv-quant sparse flash attention.

Loads custom op via ``enable_custom_op`` only (no ``adapt_patch`` / e2e conftest).

Production convention (``DeviceOperator.execute_kv_quant_sparse_flash_attention``)::

    query = cat(ql_nope, q_pe)   # TND [T, N, 576]
    key   = kv_merged            # PA_BSND packed int8 [Bn, Bs, 1, 656]
    value = kv_merged            # same packed buffer (no knope materialize)
    # Always ``torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention`` in-process.

Same-process conflict (this env)::

    * load ``vllm_ascend_C`` then call ``torch_npu`` → segfault
    * call ``torch_npu`` then load custom / call ``_C_ascend`` →
      ``binary bin not found`` / ACLNN tiling failure

So ``--backend both`` always runs the two backends in **separate subprocesses**.
Serving must never mix them in one worker (that was a prior enable_sparse_c8 bug).

Run (NPU required)::

    python benchmarks/scripts/bench_kv_quant_sparse_flash_attention.py --backend both
    python benchmarks/scripts/bench_kv_quant_sparse_flash_attention.py --backend torch_npu
    python benchmarks/scripts/bench_kv_quant_sparse_flash_attention.py --backend custom
    python benchmarks/scripts/bench_kv_quant_sparse_flash_attention.py --shapes
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch_npu

torch_npu.npu.config.allow_internal_format = True

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
NUM_ATTENTION_HEADS = 128
INDEX_TOPK = 2048
TILE_SIZE = 128
BLOCK_SIZE = 128
PACKED_KV_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM * 2 + (KV_LORA_RANK // TILE_SIZE) * 4
assert PACKED_KV_DIM == 656

WARMUP = 10
ITERS = 50

COMMON_KWARGS = dict(
    sparse_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=TILE_SIZE,
    key_quant_mode=2,
    value_quant_mode=2,
    rope_head_dim=QK_ROPE_HEAD_DIM,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def try_enable_custom_op() -> bool:
    """Load ``torch.ops._C_ascend`` without calling ``adapt_patch``."""
    try:
        log("[step] loading custom op via enable_custom_op() ...")
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            log("[warn] enable_custom_op() returned False; skip _C_ascend")
            return False
        if not hasattr(torch.ops, "_C_ascend"):
            log("[warn] torch.ops._C_ascend missing; skip _C_ascend")
            return False
        if not hasattr(torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention"):
            log("[warn] npu_kv_quant_sparse_flash_attention missing on _C_ascend; skip")
            return False
        log("[step] custom op ready")
        return True
    except Exception as e:
        log(f"[warn] failed to load custom op ({type(e).__name__}: {e}); skip _C_ascend")
        return False


def make_inputs(
    *,
    num_tokens: int,
    kv_seq: int,
    num_heads: int,
    block_size: int = BLOCK_SIZE,
    sparse_count: int = INDEX_TOPK,
    seed: int = 1024,
    value_mode: str = "packed",
) -> dict:
    torch.manual_seed(seed)
    device = "npu"

    if kv_seq % block_size != 0:
        raise ValueError(f"kv_seq ({kv_seq}) must be divisible by block_size ({block_size})")
    if sparse_count > kv_seq:
        raise ValueError(f"sparse_count ({sparse_count}) must be <= kv_seq ({kv_seq})")
    if value_mode not in ("knope", "packed"):
        raise ValueError(f"unknown value_mode={value_mode!r}")

    num_blocks = kv_seq // block_size
    kv_heads = 1
    scale_value = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) ** -0.5

    q_nope = torch.empty(
        (num_tokens, num_heads, KV_LORA_RANK), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    q_pe = torch.empty(
        (num_tokens, num_heads, QK_ROPE_HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    query = torch.cat((q_nope, q_pe), dim=-1)

    key_nope = torch.empty(
        (num_blocks, block_size, kv_heads, KV_LORA_RANK), dtype=torch.int8, device=device
    ).random_(-8, 8)
    key_rope = torch.empty(
        (num_blocks, block_size, kv_heads, QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    dequant_scale = torch.empty(
        (num_blocks, block_size, kv_heads, KV_LORA_RANK // TILE_SIZE),
        dtype=torch.float32,
        device=device,
    ).uniform_(0.1, 1.0)
    key = torch.cat(
        (
            key_nope,
            key_rope.contiguous().view(torch.int8),
            dequant_scale.contiguous().view(torch.int8),
        ),
        dim=-1,
    ).contiguous()
    assert key.shape[-1] == PACKED_KV_DIM

    value = key_nope.contiguous() if value_mode == "knope" else key

    sparse_indices = torch.full(
        (num_tokens, kv_heads, sparse_count), -1, dtype=torch.int32, device=device
    )
    for t in range(num_tokens):
        valid_end = kv_seq - num_tokens + t + 1
        take = min(sparse_count, valid_end)
        sparse_indices[t, 0, :take] = torch.randperm(valid_end, device=device, dtype=torch.int32)[
            :take
        ]

    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse_indices,
        "block_table": torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, num_blocks),
        "actual_seq_lengths_query": torch.tensor([num_tokens], dtype=torch.int32, device=device),
        "actual_seq_lengths_kv": torch.tensor([kv_seq], dtype=torch.int32, device=device),
        "scale_value": scale_value,
        "value_mode": value_mode,
        "value_dim": value.shape[-1],
    }


def run_torch_npu(inputs: dict):
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        sparse_indices=inputs["sparse_indices"],
        scale_value=inputs["scale_value"],
        block_table=inputs["block_table"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        **COMMON_KWARGS,
    )


def run_custom(inputs: dict, *, return_softmax_lse: bool = False):
    return torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        sparse_indices=inputs["sparse_indices"],
        scale_value=inputs["scale_value"],
        block_table=inputs["block_table"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        return_softmax_lse=return_softmax_lse,
        **COMMON_KWARGS,
    )


def unwrap_out(out):
    return out[0] if isinstance(out, tuple) else out


def bench(fn, inputs: dict, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(inputs)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(inputs)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def shape_tag(num_tokens: int, num_heads: int, kv_seq: int, value_dim: int, value_mode: str) -> str:
    return (
        f"T={num_tokens} N={num_heads} kv_seq={kv_seq} topk={INDEX_TOPK} "
        f"key_dim={PACKED_KV_DIM} value_dim={value_dim} value_mode={value_mode} "
        f"block={BLOCK_SIZE} layout=TND"
    )


def run_single_backend(
    *,
    backend: str,
    num_tokens: int,
    kv_seq: int,
    num_heads: int,
    value_mode: str,
    seed: int,
    warmup: int,
    iters: int,
    dump_path: str | None,
) -> None:
    assert backend in ("torch_npu", "custom")

    if backend == "custom":
        if not try_enable_custom_op():
            raise SystemExit("custom backend unavailable")

    inputs = make_inputs(
        num_tokens=num_tokens,
        kv_seq=kv_seq,
        num_heads=num_heads,
        value_mode=value_mode,
        seed=seed,
    )
    tag = shape_tag(num_tokens, num_heads, kv_seq, inputs["value_dim"], value_mode)

    if backend == "torch_npu":
        log(f"[step] torch_npu call ({tag}) ...")
        out = run_torch_npu(inputs)
        ms = bench(run_torch_npu, inputs, warmup=warmup, iters=iters)
        name = "torch_npu"
    else:
        log(f"[step] _C_ascend call ({tag}) ...")
        out = unwrap_out(run_custom(inputs, return_softmax_lse=False))
        ms = bench(
            lambda x: unwrap_out(run_custom(x, return_softmax_lse=False)),
            inputs,
            warmup=warmup,
            iters=iters,
        )
        name = "_C_ascend"

    assert out.dtype == torch.bfloat16
    log(f"[{name}] {tag} | {ms:.3f} ms  out={tuple(out.shape)}")

    if dump_path:
        torch.save(
            {
                "backend": backend,
                "ms": ms,
                "out": out.detach().float().cpu(),
                "shape": tuple(out.shape),
                "tag": tag,
            },
            dump_path,
        )
        log(f"[step] dumped result → {dump_path}")


def run_both_isolated(
    *,
    shapes: list[tuple[int, int, int]],
    value_mode: str,
    seed: int,
    warmup: int,
    iters: int,
) -> None:
    script = str(Path(__file__).resolve())
    log(
        "[note] --backend both uses subprocess isolation "
        "(same-process torch_npu ↔ _C_ascend conflicts on this stack)."
    )

    for num_tokens, kv_seq, num_heads in shapes:
        with tempfile.TemporaryDirectory(prefix="kv_quant_sfa_bench_") as td:
            npu_dump = str(Path(td) / "torch_npu.pt")
            custom_dump = str(Path(td) / "custom.pt")
            common = [
                sys.executable,
                script,
                "--t",
                str(num_tokens),
                "--kv",
                str(kv_seq),
                "--heads",
                str(num_heads),
                "--value-mode",
                value_mode,
                "--seed",
                str(seed),
                "--warmup",
                str(warmup),
                "--iters",
                str(iters),
            ]

            log(f"[step] subprocess torch_npu T={num_tokens} N={num_heads} kv={kv_seq} ...")
            r1 = subprocess.run(
                [*common, "--backend", "torch_npu", "--dump", npu_dump],
                check=False,
            )
            if r1.returncode != 0:
                raise SystemExit(f"torch_npu subprocess failed (code={r1.returncode})")

            log(f"[step] subprocess _C_ascend T={num_tokens} N={num_heads} kv={kv_seq} ...")
            r2 = subprocess.run(
                [*common, "--backend", "custom", "--dump", custom_dump],
                check=False,
            )
            if r2.returncode != 0:
                raise SystemExit(f"custom subprocess failed (code={r2.returncode})")

            npu = torch.load(npu_dump, map_location="cpu", weights_only=False)
            custom = torch.load(custom_dump, map_location="cpu", weights_only=False)
            assert npu["out"].shape == custom["out"].shape
            max_abs = (custom["out"] - npu["out"]).abs().max().item()
            ms_npu = float(npu["ms"])
            ms_custom = float(custom["ms"])
            faster = "_C_ascend" if ms_custom < ms_npu else "torch_npu"
            speedup = max(ms_custom, ms_npu) / max(min(ms_custom, ms_npu), 1e-9)
            log(
                f"[kv-quant SFA] {npu['tag']} | "
                f"_C_ascend={ms_custom:.3f} ms  torch_npu={ms_npu:.3f} ms  "
                f"faster={faster} (~{speedup:.2f}x)  max_abs_diff={max_abs:.6g}  "
                f"out={npu['shape']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=int, default=1, help="num_tokens (T)")
    parser.add_argument("--kv", type=int, default=4096, help="kv_seq")
    parser.add_argument("--heads", type=int, default=NUM_ATTENTION_HEADS, help="query heads N")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--seed", type=int, default=1024, help="shared RNG seed for both backends")
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="run a small shape sweep instead of a single shape",
    )
    parser.add_argument(
        "--backend",
        choices=("both", "torch_npu", "custom"),
        default="both",
        help="which API(s) to run (default: both, via subprocess)",
    )
    parser.add_argument(
        "--value-mode",
        choices=("knope", "packed"),
        default="packed",
        help="value tensor: packed=656 (default/production, same as key) or knope=512",
    )
    parser.add_argument(
        "--dump",
        default=None,
        help=argparse.SUPPRESS,  # internal: dump tensor+ms for parent compare
    )
    args = parser.parse_args()

    shapes = (
        [
            (1, 4096, NUM_ATTENTION_HEADS),
            (1, 16384, NUM_ATTENTION_HEADS),
            (1, 4096, 8),
            (1, 16384, 8),
            (16, 4096, 8),
            (64, 4096, 8),
        ]
        if args.shapes
        else [(args.t, args.kv, args.heads)]
    )

    if args.backend == "both":
        # Parent does not touch NPU ops; children do.
        run_both_isolated(
            shapes=shapes,
            value_mode=args.value_mode,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
        )
        return

    if not torch.npu.is_available():
        raise SystemExit("NPU is required")

    for num_tokens, kv_seq, num_heads in shapes:
        run_single_backend(
            backend=args.backend,
            num_tokens=num_tokens,
            kv_seq=kv_seq,
            num_heads=num_heads,
            value_mode=args.value_mode,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            dump_path=args.dump,
        )
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"[error] exception before exit:\n{sys.exc_info()[1]!r}")
        raise
