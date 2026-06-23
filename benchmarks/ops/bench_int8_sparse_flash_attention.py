#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Benchmark and accuracy-compare int8 vs bf16 sparse flash attention on Ascend NPU.

Targets ``csrc/attention/int8_sparse_flash_attention`` (910B only):
  - **bf16 baseline**: ``torch.ops._C_ascend.npu_sparse_flash_attention``
  - **int8**: ``torch.ops._C_ascend.npu_int8_sparse_flash_attention``

KV nope uses packed 910B sparse C8 layout: D=528 = 512 int8 + 4 fp32 per-tile scales.
Rope (D=64) stays bf16/fp16. Shapes follow production SFA (TND query, PA_BSND KV).

Example:
  python benchmarks/ops/bench_int8_sparse_flash_attention.py --scenario decode
  python benchmarks/ops/bench_int8_sparse_flash_attention.py --scenario all --grid
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

import vllm_ascend.platform  # noqa: F401
from vllm_ascend.attention.sfa_k_nope_pack import (
    K_NOPE_INT8_DIM,
    K_NOPE_PACKED_BYTES,
    quantize_k_nope_per_group,
)
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

enable_custom_op()

DEEPSEEK_V32_SFA = {
    "num_heads_q": 64,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "index_topk": 2048,
    "block_size": 128,
    "sparse_block_size": 1,
    "sparse_mode": 3,
    "attention_mode": 2,
    "layout_query": "TND",
    "layout_kv": "PA_BSND",
}

# Signal-relative error budgets (same spirit as tests/ut/attention/a2/test_sfa_v1_precision.py).
_SIG_FLOOR_FRAC = 0.5


@dataclass
class BenchCase:
    name: str
    batch_size: int
    seq_q: int
    seq_kv: int


@dataclass
class LatencyResult:
    case: str
    op: str
    latency_ms: float
    q_tokens: int
    kv_tokens: int
    q_tokens_per_s: float


@dataclass
class AccuracyResult:
    case: str
    max_abs_err: float
    mean_abs_err: float
    max_sig_rel_err: float
    mean_sig_rel_err: float
    max_rel_err_sig: float
    cosine_sim: float
    bf16_latency_ms: float
    int8_latency_ms: float
    speedup: float


@dataclass
class PreparedCase:
    ql_nope: torch.Tensor
    q_pe: torch.Tensor
    k_nope_bf16: torch.Tensor
    k_nope_int8: torch.Tensor
    k_rope_cache: torch.Tensor
    sparse_indices: torch.Tensor
    block_table: torch.Tensor
    cum_query_lens: torch.Tensor
    seq_lens: torch.Tensor
    scale: float
    q_tokens: int
    kv_tokens: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark int8 SFA vs bf16 SFA and compare output accuracy",
    )
    parser.add_argument(
        "--scenario",
        choices=["decode", "prefill", "all"],
        default="all",
        help="decode=sq1; prefill=sq>1; all=run preset decode+prefill groups",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-q", type=int, default=128, help="Query length (prefill)")
    parser.add_argument("--seq-kv", type=int, default=8192, help="KV length per sequence")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run preset batch/kv sweep instead of a single --batch-size/--seq-kv case",
    )
    parser.add_argument("--num-heads-q", type=int, default=DEEPSEEK_V32_SFA["num_heads_q"])
    parser.add_argument("--kv-lora-rank", type=int, default=DEEPSEEK_V32_SFA["kv_lora_rank"])
    parser.add_argument("--qk-rope-head-dim", type=int, default=DEEPSEEK_V32_SFA["qk_rope_head_dim"])
    parser.add_argument(
        "--sparse-count",
        type=int,
        default=DEEPSEEK_V32_SFA["index_topk"],
        help="Sparse top-k (hf_config index_topk, default 2048)",
    )
    parser.add_argument("--block-size", type=int, default=DEEPSEEK_V32_SFA["block_size"])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip accuracy comparison, only report latency",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Skip latency benchmark, only compare bf16 vs int8 outputs once",
    )
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
                        )
                    )
        else:
            cases.append(
                BenchCase(
                    name=f"decode_bs{args.batch_size}_kv{args.seq_kv}",
                    batch_size=args.batch_size,
                    seq_q=1,
                    seq_kv=args.seq_kv,
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
                        )
                    )
        else:
            cases.append(
                BenchCase(
                    name=f"prefill_bs{args.batch_size}_sq{args.seq_q}_kv{args.seq_kv}",
                    batch_size=args.batch_size,
                    seq_q=args.seq_q,
                    seq_kv=args.seq_kv,
                )
            )
    return cases


def _softmax_scale(kv_lora_rank: int, qk_rope_head_dim: int) -> float:
    return (kv_lora_rank + qk_rope_head_dim) ** -0.5


def _make_block_table(
    batch_size: int,
    seq_kv: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    blocks_per_seq = math.ceil(seq_kv / block_size)
    block_num = blocks_per_seq * batch_size + 1
    block_table = torch.zeros(batch_size, blocks_per_seq, dtype=torch.int32, device=device)
    next_block_id = 1
    for b in range(batch_size):
        for i in range(blocks_per_seq):
            block_table[b, i] = next_block_id
            next_block_id += 1
    return block_table, block_num


def _make_sparse_indices(
    batch_size: int,
    seq_q: int,
    seq_kv: int,
    sparse_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Causal top-k token indices; shape (T, 1, sparse_count), padded with -1."""
    total_q = batch_size * seq_q
    topk = torch.full((total_q, 1, sparse_count), -1, dtype=torch.int32, device=device)
    cum_q = 0
    for _b in range(batch_size):
        ctx_len = seq_kv - seq_q
        for j in range(seq_q):
            valid_end = min(ctx_len + j + 1, seq_kv)
            k = min(sparse_count, valid_end)
            if k == valid_end and valid_end <= sparse_count:
                idxs = list(range(valid_end))
            else:
                idxs = sorted(random.sample(range(valid_end), k)) if valid_end >= k else list(range(valid_end))
            if len(idxs) < sparse_count:
                idxs += [-1] * (sparse_count - len(idxs))
            topk[cum_q + j, 0, :] = torch.tensor(idxs, dtype=torch.int32, device=device)
        cum_q += seq_q
    return topk


def _quantize_kv_packed_per_tile(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack 512-dim k_nope rows into 528-byte int8 cache rows (910B sparse C8)."""
    *prefix, dim = kv_bf16.shape
    if dim != K_NOPE_INT8_DIM:
        raise ValueError(f"packed quant expects kv_lora_rank={K_NOPE_INT8_DIM}, got {dim}")
    flat = kv_bf16.reshape(-1, dim)
    packed = quantize_k_nope_per_group(flat).view(torch.int8)
    return packed.view(*prefix, K_NOPE_PACKED_BYTES)


def _prepare_case(
    case: BenchCase,
    args: argparse.Namespace,
    device: torch.device,
) -> PreparedCase:
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    n_heads = args.num_heads_q
    kv_lora_rank = args.kv_lora_rank
    rope_dim = args.qk_rope_head_dim
    block_size = args.block_size
    sparse_count = args.sparse_count
    b = case.batch_size
    s_q = case.seq_q
    s_kv = case.seq_kv
    scale = _softmax_scale(kv_lora_rank, rope_dim)

    total_q = b * s_q
    ql_nope = torch.randn(total_q, n_heads, kv_lora_rank, dtype=dtype, device=device)
    q_pe = torch.randn(total_q, n_heads, rope_dim, dtype=dtype, device=device)

    block_table, block_num = _make_block_table(b, s_kv, block_size, device)
    k_nope_bf16 = torch.randn(block_num, block_size, 1, kv_lora_rank, dtype=dtype, device=device)
    k_nope_int8 = _quantize_kv_packed_per_tile(k_nope_bf16)
    if k_nope_int8.dtype != torch.int8:
        raise TypeError(f"int8 KV must be torch.int8, got {k_nope_int8.dtype}")
    if k_nope_int8.shape[-1] != K_NOPE_PACKED_BYTES:
        raise ValueError(
            f"int8 KV last dim must be {K_NOPE_PACKED_BYTES}, got {k_nope_int8.shape[-1]}"
        )
    k_rope_cache = torch.randn(block_num, block_size, 1, rope_dim, dtype=dtype, device=device)
    sparse_indices = _make_sparse_indices(b, s_q, s_kv, sparse_count, device)

    cum_query_lens = torch.tensor(
        [s_q * (i + 1) for i in range(b)],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full((b,), s_kv, dtype=torch.int32, device=device)

    return PreparedCase(
        ql_nope=ql_nope,
        q_pe=q_pe,
        k_nope_bf16=k_nope_bf16,
        k_nope_int8=k_nope_int8,
        k_rope_cache=k_rope_cache,
        sparse_indices=sparse_indices,
        block_table=block_table,
        cum_query_lens=cum_query_lens,
        seq_lens=seq_lens,
        scale=scale,
        q_tokens=total_q,
        kv_tokens=b * s_kv,
    )


def _run_bf16_sfa(prepared: PreparedCase) -> torch.Tensor:
    attn_metadata = SimpleNamespace(block_table=prepared.block_table)
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=prepared.ql_nope,
        key=prepared.k_nope_bf16,
        value=prepared.k_nope_bf16,
        sparse_indices=prepared.sparse_indices,
        scale_value=prepared.scale,
        sparse_block_size=DEEPSEEK_V32_SFA["sparse_block_size"],
        block_table=attn_metadata.block_table,
        actual_seq_lengths_query=prepared.cum_query_lens,
        actual_seq_lengths_kv=prepared.seq_lens,
        query_rope=prepared.q_pe,
        key_rope=prepared.k_rope_cache,
        layout_query=DEEPSEEK_V32_SFA["layout_query"],
        layout_kv=DEEPSEEK_V32_SFA["layout_kv"],
        sparse_mode=DEEPSEEK_V32_SFA["sparse_mode"],
        attention_mode=DEEPSEEK_V32_SFA["attention_mode"],
    )
    return out


def _run_int8_sfa(prepared: PreparedCase) -> torch.Tensor:
    attn_metadata = SimpleNamespace(block_table=prepared.block_table)
    out, _, _ = torch.ops._C_ascend.npu_int8_sparse_flash_attention(
        query=prepared.ql_nope,
        key=prepared.k_nope_int8,
        value=prepared.k_nope_int8,
        sparse_indices=prepared.sparse_indices,
        scale_value=prepared.scale,
        block_table=attn_metadata.block_table,
        actual_seq_lengths_query=prepared.cum_query_lens,
        actual_seq_lengths_kv=prepared.seq_lens,
        query_rope=prepared.q_pe,
        key_rope=prepared.k_rope_cache,
        key_scale=1.0,
        key_offset=0.0,
        sparse_block_size=DEEPSEEK_V32_SFA["sparse_block_size"],
        layout_query=DEEPSEEK_V32_SFA["layout_query"],
        layout_kv=DEEPSEEK_V32_SFA["layout_kv"],
        sparse_mode=DEEPSEEK_V32_SFA["sparse_mode"],
        attention_mode=DEEPSEEK_V32_SFA["attention_mode"],
    )
    return out


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


def _compute_accuracy(
    bf16_out: torch.Tensor,
    int8_out: torch.Tensor,
) -> tuple[float, float, float, float, float, float]:
    ref = bf16_out.float()
    out = int8_out.float()
    diff = (out - ref).abs()
    ref_abs = ref.abs()

    peak = float(ref_abs.max())
    mean_ref_abs = float(ref_abs.mean())
    max_abs_err = float(diff.max())
    mean_abs_err = float(diff.mean())
    max_sig_rel_err = max_abs_err / peak if peak > 0 else 0.0
    mean_sig_rel_err = mean_abs_err / mean_ref_abs if mean_ref_abs > 0 else 0.0

    sig_floor = peak * _SIG_FLOOR_FRAC
    significant_mask = ref_abs >= sig_floor
    if significant_mask.any():
        per_elem_rel = diff[significant_mask] / ref_abs[significant_mask]
        max_rel_err_sig = float(per_elem_rel.max())
    else:
        max_rel_err_sig = 0.0

    flat_ref = ref.reshape(-1)
    flat_out = out.reshape(-1)
    cosine_sim = float(
        torch.nn.functional.cosine_similarity(flat_ref.unsqueeze(0), flat_out.unsqueeze(0)).item()
    )
    return max_abs_err, mean_abs_err, max_sig_rel_err, mean_sig_rel_err, max_rel_err_sig, cosine_sim


def _print_header(device_type: AscendDeviceType) -> None:
    print("=" * 100)
    print("DeepSeek-V3.2 Int8 vs BF16 Sparse Flash Attention")
    print("  op source    : csrc/attention/int8_sparse_flash_attention")
    print(f"  device_type  : {device_type.name}")
    print("  bf16 op      : npu_sparse_flash_attention")
    print("  int8 op      : npu_int8_sparse_flash_attention (910B packed D=528)")
    print(f"  kv_lora_rank : {DEEPSEEK_V32_SFA['kv_lora_rank']}")
    print(f"  rope_dim     : {DEEPSEEK_V32_SFA['qk_rope_head_dim']}")
    print(f"  index_topk   : {DEEPSEEK_V32_SFA['index_topk']}")
    print("=" * 100)


def _print_accuracy_header() -> None:
    print(
        f"{'case':<28} {'max_abs':>10} {'mean_abs':>10} {'max_rel%':>9} "
        f"{'cos_sim':>8} {'bf16_ms':>9} {'int8_ms':>9} {'speedup':>8}"
    )
    print("-" * 100)


def _print_latency_header() -> None:
    print(f"{'case':<32} {'op':<28} {'latency(ms)':>12} {'q_tok/s':>14} {'q_tokens':>10}")
    print("-" * 100)


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.npu.is_available():
        print("ERROR: NPU is not available. Run this script on Ascend hardware.", file=sys.stderr)
        return 1

    device_type = get_ascend_device_type()
    if device_type not in {AscendDeviceType.A2, AscendDeviceType.A3}:
        print(
            f"ERROR: npu_int8_sparse_flash_attention is built for ascend910b (A2/A3). "
            f"Current device: {device_type.name}.",
            file=sys.stderr,
        )
        return 1

    if not hasattr(torch.ops._C_ascend, "npu_int8_sparse_flash_attention"):
        print(
            "ERROR: npu_int8_sparse_flash_attention is not registered. "
            "Rebuild with: bash csrc/build_aclnn.sh $(pwd) ascend910b",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device)
    torch.npu.set_device(device)

    run_accuracy = not args.benchmark_only
    run_benchmark = not args.accuracy_only

    cases = _build_cases(args)
    _print_header(device_type)

    accuracy_results: list[AccuracyResult] = []
    latency_results: list[LatencyResult] = []

    if run_accuracy:
        _print_accuracy_header()

    for case in cases:
        try:
            prepared = _prepare_case(case, args, device)

            with torch.no_grad():
                bf16_out = _run_bf16_sfa(prepared)
                int8_out = _run_int8_sfa(prepared)

            if run_accuracy:
                (
                    max_abs_err,
                    mean_abs_err,
                    max_sig_rel_err,
                    mean_sig_rel_err,
                    max_rel_err_sig,
                    cosine_sim,
                ) = _compute_accuracy(bf16_out, int8_out)

                bf16_ms = 0.0
                int8_ms = 0.0
                speedup = 0.0
                if run_benchmark:
                    bf16_ms = _benchmark_npu(lambda: _run_bf16_sfa(prepared), args.warmup, args.iters)
                    int8_ms = _benchmark_npu(lambda: _run_int8_sfa(prepared), args.warmup, args.iters)
                    speedup = bf16_ms / int8_ms if int8_ms > 0 else 0.0

                acc = AccuracyResult(
                    case=case.name,
                    max_abs_err=max_abs_err,
                    mean_abs_err=mean_abs_err,
                    max_sig_rel_err=max_sig_rel_err,
                    mean_sig_rel_err=mean_sig_rel_err,
                    max_rel_err_sig=max_rel_err_sig,
                    cosine_sim=cosine_sim,
                    bf16_latency_ms=bf16_ms,
                    int8_latency_ms=int8_ms,
                    speedup=speedup,
                )
                accuracy_results.append(acc)
                print(
                    f"{acc.case:<28} {acc.max_abs_err:>10.4e} {acc.mean_abs_err:>10.4e} "
                    f"{acc.max_sig_rel_err * 100:>8.3f}% {acc.cosine_sim:>8.6f} "
                    f"{acc.bf16_latency_ms:>9.3f} {acc.int8_latency_ms:>9.3f} {acc.speedup:>8.2f}x"
                )

            elif run_benchmark:
                bf16_ms = _benchmark_npu(lambda: _run_bf16_sfa(prepared), args.warmup, args.iters)
                int8_ms = _benchmark_npu(lambda: _run_int8_sfa(prepared), args.warmup, args.iters)
                for op_name, latency_ms in (
                    ("npu_sparse_flash_attention", bf16_ms),
                    ("npu_int8_sparse_flash_attention", int8_ms),
                ):
                    latency_results.append(
                        LatencyResult(
                            case=case.name,
                            op=op_name,
                            latency_ms=latency_ms,
                            q_tokens=prepared.q_tokens,
                            kv_tokens=prepared.kv_tokens,
                            q_tokens_per_s=prepared.q_tokens / (latency_ms / 1000.0),
                        )
                    )

        except Exception as exc:  # noqa: BLE001 - benchmark script reports and continues
            print(f"{case.name:<28} FAILED: {exc}", file=sys.stderr)

    if run_benchmark and not run_accuracy and latency_results:
        print()
        _print_latency_header()
        for result in latency_results:
            print(
                f"{result.case:<32} {result.op:<28} {result.latency_ms:>12.3f} "
                f"{result.q_tokens_per_s:>14.1f} {result.q_tokens:>10}"
            )

    print("-" * 100)
    if not accuracy_results and not latency_results:
        print("No results collected.", file=sys.stderr)
        return 2

    if accuracy_results:
        worst = max(accuracy_results, key=lambda r: r.max_sig_rel_err)
        best_speed = max(
            (r for r in accuracy_results if r.speedup > 0),
            key=lambda r: r.speedup,
            default=None,
        )
        print(
            f"Accuracy vs bf16: worst max_rel={worst.max_sig_rel_err * 100:.3f}% "
            f"({worst.case}), mean_rel_avg="
            f"{sum(r.mean_sig_rel_err for r in accuracy_results) / len(accuracy_results) * 100:.3f}%"
        )
        if best_speed is not None:
            print(
                f"Best int8 speedup: {best_speed.speedup:.2f}x @ {best_speed.case} "
                f"(bf16={best_speed.bf16_latency_ms:.3f}ms, int8={best_speed.int8_latency_ms:.3f}ms)"
            )

    if args.json_out:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device_type": device_type.name,
            "op_source": "csrc/attention/int8_sparse_flash_attention",
            "kv_storage_dim": K_NOPE_PACKED_BYTES,
            "config": DEEPSEEK_V32_SFA,
            "args": vars(args),
            "accuracy": [asdict(r) for r in accuracy_results],
            "latency": [asdict(r) for r in latency_results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
