#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Micro-benchmark DeepSeek-V3.2 DSA sparse attention kernels on Ascend NPU.

This script benchmarks the core attention operators used by DeepSeek-V3.2:
  - A2/A3: ``npu_sparse_attn_sharedkv`` (+ metadata)
  - A5:    ``npu_kv_quant_sparse_attn_sharedkv`` (+ metadata)

It does NOT load the full model; shapes follow the DSA path in production
(TP-sharded heads, compress_ratio=4, index_topk=2048, head_dim=512).

Example:
  python benchmarks/ops/bench_deepseek32_attention.py --scenario all
  python benchmarks/ops/bench_deepseek32_attention.py --scenario decode --batch-size 4096 --seq-kv 8192
  python benchmarks/ops/bench_deepseek32_attention.py --scenario decode --grid
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

enable_custom_op()

# DeepSeek-V3.2 DSA defaults (per attention rank after TP).
# Full model: 128 heads, kv_lora_rank=512, index_topk=2048, sliding_window=128.
DEEPSEEK_V32 = {
    "num_heads_q": 64,  # typical per-rank heads under TP8 (128 / 2 DP ranks etc.)
    "num_heads_kv": 1,
    "head_dim": 512,
    "cmp_ratio": 4,
    "index_topk": 2048,  # DeepSeek-V3.2 hf_config.index_topk (QuantLightningIndexer + cmp_sparse_indices)
    "cmp_topk": 2048,    # metadata cmp_topk, same as index_topk in production
    "ori_win_left": 127,
    "ori_win_right": 0,
    "ori_mask_mode": 4,  # band / SWA
    "cmp_mask_mode": 3,  # rightDownCausal
    "sliding_window": 128,
    "rope_head_dim": 64,
    "tile_size": 64,
    "kv_quant_mode": 1,
    "layout_q": "TND",
    "layout_kv": "PA_ND",
}

KV_DIM_BF16 = 512
KV_DIM_FP8_PACKED = 640  # rope(64,bf16) + nope(448,fp8) + scale(7,fp8_e8m0) + pad


@dataclass
class BenchCase:
    name: str
    batch_size: int
    seq_q: int
    seq_kv: int
    cmp_ratio: int
    cmp_topk: int
    has_cmp_kv: bool


@dataclass
class BenchResult:
    case: str
    op: str
    device_type: str
    latency_ms: float
    q_tokens: int
    kv_tokens: int
    q_tokens_per_s: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V3.2 DSA attention ops")
    parser.add_argument(
        "--scenario",
        choices=["decode", "prefill", "scfa", "all"],
        default="all",
        help="decode=SWA only; prefill=long SWA; scfa=SWA+compressed sparse; all=run all",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (used when scenario is decode/prefill/scfa)")
    parser.add_argument("--seq-q", type=int, default=128, help="Query sequence length (prefill / scfa prefill case)")
    parser.add_argument("--seq-kv", type=int, default=8192, help="KV sequence length")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run preset batch/kv sweep instead of a single case from --batch-size/--seq-kv",
    )
    parser.add_argument("--num-heads-q", type=int, default=DEEPSEEK_V32["num_heads_q"])
    parser.add_argument("--cmp-ratio", type=int, default=DEEPSEEK_V32["cmp_ratio"])
    parser.add_argument(
        "--cmp-topk",
        type=int,
        default=DEEPSEEK_V32["index_topk"],
        help="Cmp sparse top-k (DeepSeek-V3.2 index_topk, default 2048)",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save JSON results")
    return parser.parse_args()


def _build_cases(args: argparse.Namespace) -> list[BenchCase]:
    cases: list[BenchCase] = []
    if args.scenario in {"decode", "all"}:
        if args.grid or args.scenario == "all":
            for bs in (1, 4, 8):
                for kv in (4096, 8192, 16384):
                    cases.append(
                        BenchCase(
                            name=f"decode_bs{bs}_kv{kv}",
                            batch_size=bs,
                            seq_q=1,
                            seq_kv=kv,
                            cmp_ratio=1,
                            cmp_topk=0,
                            has_cmp_kv=False,
                        )
                    )
        else:
            cases.append(
                BenchCase(
                    name=f"decode_bs{args.batch_size}_kv{args.seq_kv}",
                    batch_size=args.batch_size,
                    seq_q=1,
                    seq_kv=args.seq_kv,
                    cmp_ratio=1,
                    cmp_topk=0,
                    has_cmp_kv=False,
                )
            )
    if args.scenario in {"prefill", "all"}:
        if args.grid or args.scenario == "all":
            for bs in (1, 4):
                for sq in (128, 512):
                    cases.append(
                        BenchCase(
                            name=f"prefill_bs{bs}_sq{sq}_kv{args.seq_kv}",
                            batch_size=bs,
                            seq_q=sq,
                            seq_kv=args.seq_kv,
                            cmp_ratio=1,
                            cmp_topk=0,
                            has_cmp_kv=False,
                        )
                    )
        else:
            cases.append(
                BenchCase(
                    name=f"prefill_bs{args.batch_size}_sq{args.seq_q}_kv{args.seq_kv}",
                    batch_size=args.batch_size,
                    seq_q=args.seq_q,
                    seq_kv=args.seq_kv,
                    cmp_ratio=1,
                    cmp_topk=0,
                    has_cmp_kv=False,
                )
            )
    if args.scenario in {"scfa", "all"}:
        if args.grid or args.scenario == "all":
            for bs in (1, 4, 8):
                cases.append(
                    BenchCase(
                        name=f"scfa_bs{bs}_sq1_kv{args.seq_kv}",
                        batch_size=bs,
                        seq_q=1,
                        seq_kv=args.seq_kv,
                        cmp_ratio=args.cmp_ratio,
                        cmp_topk=args.cmp_topk,
                        has_cmp_kv=True,
                    )
                )
            cases.append(
                BenchCase(
                    name=f"scfa_prefill_bs{args.batch_size}_sq{args.seq_q}_kv{args.seq_kv}",
                    batch_size=args.batch_size,
                    seq_q=args.seq_q,
                    seq_kv=args.seq_kv,
                    cmp_ratio=args.cmp_ratio,
                    cmp_topk=args.cmp_topk,
                    has_cmp_kv=True,
                )
            )
        else:
            cases.append(
                BenchCase(
                    name=f"scfa_bs{args.batch_size}_sq{args.seq_q}_kv{args.seq_kv}",
                    batch_size=args.batch_size,
                    seq_q=args.seq_q,
                    seq_kv=args.seq_kv,
                    cmp_ratio=args.cmp_ratio,
                    cmp_topk=args.cmp_topk,
                    has_cmp_kv=True,
                )
            )
    if args.scenario not in {"decode", "prefill", "scfa", "all"}:
        cases.append(
            BenchCase(
                name="custom",
                batch_size=args.batch_size,
                seq_q=args.seq_q,
                seq_kv=args.seq_kv,
                cmp_ratio=args.cmp_ratio if args.cmp_ratio > 1 else 1,
                cmp_topk=args.cmp_topk,
                has_cmp_kv=args.cmp_ratio > 1,
            )
        )
    return cases


def _softmax_scale(head_dim: int) -> float:
    return head_dim**-0.5


def _make_block_table(batch_size: int, seq_kv: int, block_size: int, block_num: int, device: torch.device) -> torch.Tensor:
    blocks_per_seq = math.ceil(seq_kv / block_size)
    return torch.tensor(np.random.permutation(block_num), dtype=torch.int32, device=device).reshape(
        batch_size, -1
    )[:, :blocks_per_seq]


def _make_sparse_indices(
    total_q_tokens: int,
    num_heads_kv: int,
    topk: int,
    cmp_kv_len: int,
    seq_q: int,
    device: torch.device,
) -> torch.Tensor:
    upper = max(1, cmp_kv_len - seq_q + 1)
    k = min(topk, upper)
    idxs = random.sample(range(upper), k)
    if k < topk:
        idxs += [0] * (topk - k)
    return torch.tensor(
        [idxs for _ in range(total_q_tokens * num_heads_kv)],
        dtype=torch.int32,
        device=device,
    ).reshape(total_q_tokens, num_heads_kv, topk)


def _get_dsa_ops(
    use_kv_quant: bool,
    device: torch.device,
) -> tuple[Callable, dict, Callable, dict]:
    """Resolve DSA sparse-attention ops without importing DeviceOperator."""
    if use_kv_quant:
        return (
            torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata,
            {"kv_quant_mode": DEEPSEEK_V32["kv_quant_mode"]},
            torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv,
            {
                "kv_quant_mode": DEEPSEEK_V32["kv_quant_mode"],
                "tile_size": DEEPSEEK_V32["tile_size"],
                "rope_head_dim": DEEPSEEK_V32["rope_head_dim"],
            },
        )
    return (
        torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
        {"device": str(device)},
        torch.ops._C_ascend.npu_sparse_attn_sharedkv,
        {},
    )


def _prepare_tensors(
    case: BenchCase,
    args: argparse.Namespace,
    device: torch.device,
    use_kv_quant: bool,
) -> tuple[Callable[[], torch.Tensor], dict]:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    n1 = args.num_heads_q
    n2 = DEEPSEEK_V32["num_heads_kv"]
    dn = DEEPSEEK_V32["head_dim"]
    block_size = args.block_size
    b = case.batch_size
    s1 = case.seq_q
    s2 = case.seq_kv
    cmp_ratio = case.cmp_ratio
    topk = case.cmp_topk
    softmax_scale = _softmax_scale(dn)

    total_q = b * s1
    q = torch.randn(total_q, n1, dn, dtype=dtype, device=device)

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1, dtype=torch.int32, device=device)
    seqused_kv = torch.full((b,), s2, dtype=torch.int32, device=device)

    kv_dim = KV_DIM_FP8_PACKED if use_kv_quant else KV_DIM_BF16
    kv_dtype = torch.float8_e4m3fn if use_kv_quant else dtype

    ori_block_num = math.ceil(s2 / block_size) * b
    ori_block_table = _make_block_table(b, s2, block_size, ori_block_num, device)
    ori_kv = torch.randn(ori_block_num, block_size, n2, kv_dim, dtype=kv_dtype, device=device)

    cmp_kv = None
    cmp_block_table = None
    cmp_sparse_indices = None
    cu_seqlens_cmp_kv = None

    if case.has_cmp_kv:
        cmp_kv_len = s2 // cmp_ratio
        cmp_sparse_indices = _make_sparse_indices(total_q, n2, topk, cmp_kv_len, s1, device)
        cmp_block_num = math.ceil(cmp_kv_len / block_size) * b
        cmp_block_table = _make_block_table(b, cmp_kv_len, block_size, cmp_block_num, device)
        cmp_kv = torch.randn(cmp_block_num, block_size, n2, kv_dim, dtype=kv_dtype, device=device)
        cu_seqlens_cmp_kv = torch.zeros(b + 1, dtype=torch.int32, device=device)
        for i in range(1, b + 1):
            cu_seqlens_cmp_kv[i] = i * (cmp_kv_len // b)

    sinks = torch.rand(n1, dtype=torch.float32, device=device)

    meta_op, meta_extra_kwargs, attn_op, base_kwargs = _get_dsa_ops(use_kv_quant, device)
    meta_kwargs = {
        "num_heads_q": n1,
        "num_heads_kv": n2,
        "head_dim": dn,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
        "batch_size": b,
        "max_seqlen_q": s1,
        "max_seqlen_kv": s2,
        "cmp_topk": topk if case.has_cmp_kv else 0,
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": DEEPSEEK_V32["ori_mask_mode"],
        "cmp_mask_mode": DEEPSEEK_V32["cmp_mask_mode"],
        "ori_win_left": DEEPSEEK_V32["ori_win_left"],
        "ori_win_right": DEEPSEEK_V32["ori_win_right"],
        "layout_q": DEEPSEEK_V32["layout_q"],
        "layout_kv": DEEPSEEK_V32["layout_kv"],
        "has_ori_kv": True,
        "has_cmp_kv": case.has_cmp_kv,
        "device": str(device),
    }
    meta_kwargs.update(meta_extra_kwargs)

    metadata = meta_op(**meta_kwargs)

    attn_kwargs = dict(
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=None,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_q if s1 > 1 else None,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=None,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=DEEPSEEK_V32["ori_mask_mode"],
        cmp_mask_mode=DEEPSEEK_V32["cmp_mask_mode"],
        ori_win_left=DEEPSEEK_V32["ori_win_left"],
        ori_win_right=DEEPSEEK_V32["ori_win_right"],
        layout_q=DEEPSEEK_V32["layout_q"],
        layout_kv=DEEPSEEK_V32["layout_kv"],
        return_softmax_lse=False,
    )
    attn_kwargs.update(base_kwargs)

    def run_attn() -> torch.Tensor:
        out, _ = attn_op(q, **attn_kwargs)
        return out

    info = {
        "q_tokens": total_q,
        "kv_tokens": b * s2,
        "cmp_ratio": cmp_ratio,
        "has_cmp_kv": case.has_cmp_kv,
    }
    return run_attn, info


def _benchmark_npu(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(warmup + iters, dtype=np.float64)

    for i in range(warmup + iters):
        with torch.no_grad():
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    return float(np.min(times[warmup:]))


def _print_header(device_type: AscendDeviceType, op_name: str) -> None:
    print("=" * 88)
    print("DeepSeek-V3.2 DSA Attention Benchmark")
    print(f"  device_type : {device_type.name}")
    print(f"  attention op: {op_name}")
    print(f"  head_dim    : {DEEPSEEK_V32['head_dim']}")
    print(f"  kv layout   : {DEEPSEEK_V32['layout_kv']}")
    print("=" * 88)
    print(f"{'case':<32} {'latency(ms)':>12} {'q_tok/s':>14} {'q_tokens':>10} {'kv_tokens':>10}")
    print("-" * 88)


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.npu.is_available():
        print("ERROR: NPU is not available. Run this script on Ascend hardware.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_type = get_ascend_device_type()
    use_kv_quant = device_type == AscendDeviceType.A5
    op_name = (
        "npu_kv_quant_sparse_attn_sharedkv"
        if use_kv_quant
        else "npu_sparse_attn_sharedkv"
    )

    cases = _build_cases(args)
    _print_header(device_type, op_name)

    results: list[BenchResult] = []
    for case in cases:
        try:
            run_fn, info = _prepare_tensors(case, args, device, use_kv_quant)
            # metadata rebuild is cheap; include it in the timed region to match real forward.
            latency_ms = _benchmark_npu(run_fn, args.warmup, args.iters)
            q_tps = info["q_tokens"] / (latency_ms / 1000.0)
            result = BenchResult(
                case=case.name,
                op=op_name,
                device_type=device_type.name,
                latency_ms=latency_ms,
                q_tokens=info["q_tokens"],
                kv_tokens=info["kv_tokens"],
                q_tokens_per_s=q_tps,
            )
            results.append(result)
            print(
                f"{result.case:<32} {result.latency_ms:>12.3f} {result.q_tokens_per_s:>14.1f} "
                f"{result.q_tokens:>10} {result.kv_tokens:>10}"
            )
        except Exception as exc:  # noqa: BLE001 - benchmark script reports and continues
            print(f"{case.name:<32} FAILED: {exc}", file=sys.stderr)

    print("-" * 88)
    if not results:
        print("No benchmark results collected.", file=sys.stderr)
        return 2

    best = min(results, key=lambda r: r.latency_ms)
    print(f"Best case: {best.case} @ {best.latency_ms:.3f} ms ({best.q_tokens_per_s:.1f} q tok/s)")

    if args.json_out:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device_type": device_type.name,
            "op": op_name,
            "config": DEEPSEEK_V32,
            "args": vars(args),
            "results": [asdict(r) for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())