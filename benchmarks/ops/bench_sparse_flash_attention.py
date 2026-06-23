#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Micro-benchmark ``sparse_flash_attention`` (SFA) on Ascend NPU.

Benchmarks ``csrc/attention/sparse_flash_attention`` — the attention op used by
standard DeepSeek-V3.2 (``index_topk`` in config, no ``compress_ratios`` / SWA):
  - A2/A3: ``torch.ops._C_ascend.npu_sparse_flash_attention``
  - A5 (bf16/fp16 KV): ``torch_npu.npu_sparse_flash_attention``
  - A5 (fp8 KV, W8A8C8): ``torch_npu.npu_kv_quant_sparse_flash_attention``

Does NOT load the full model and does NOT run the Lightning Indexer (only the
attention kernel with pre-built ``sparse_indices``).

Example:
  python benchmarks/ops/bench_sparse_flash_attention.py --scenario decode --batch-size 1024 --seq-kv 8192
  python benchmarks/ops/bench_sparse_flash_attention.py --scenario all --grid
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
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

enable_custom_op()

# DeepSeek-V3.2 SFA defaults (per TP rank; full model has 128 Q heads).
# Matches hf_config: kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048.
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


@dataclass
class BenchCase:
    name: str
    batch_size: int
    seq_q: int
    seq_kv: int


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
    parser = argparse.ArgumentParser(description="Benchmark npu_sparse_flash_attention (DeepSeek-V3.2 SFA)")
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
    parser.add_argument(
        "--kv-quant",
        action="store_true",
        help="Use fp8 KV + npu_kv_quant_sparse_flash_attention (A5 W8A8C8 path)",
    )
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
            for bs in (1, 4, 8, 64, 256):
                for kv in (4096, 8192, 16384, 32768):
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
            for bs in (1, 4, 8):
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


def _resolve_op_name(use_kv_quant: bool, use_torch_npu: bool) -> str:
    if use_kv_quant:
        return "npu_kv_quant_sparse_flash_attention"
    if use_torch_npu:
        return "npu_sparse_flash_attention (torch_npu)"
    return "npu_sparse_flash_attention (_C_ascend)"


def _make_run_fn(
    *,
    use_kv_quant: bool,
    use_torch_npu: bool,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> Callable[[], torch.Tensor]:
    attn_metadata = SimpleNamespace(block_table=block_table)

    def _run_bf16() -> torch.Tensor:
        if use_torch_npu:
            out, _, _ = torch_npu.npu_sparse_flash_attention(
                query=ql_nope,
                key=k_nope_cache,
                value=k_nope_cache,
                sparse_indices=sparse_indices,
                scale_value=scale,
                sparse_block_size=DEEPSEEK_V32_SFA["sparse_block_size"],
                block_table=attn_metadata.block_table,
                actual_seq_lengths_query=cum_query_lens,
                actual_seq_lengths_kv=seq_lens,
                query_rope=q_pe,
                key_rope=k_rope_cache,
                layout_query=DEEPSEEK_V32_SFA["layout_query"],
                layout_kv=DEEPSEEK_V32_SFA["layout_kv"],
                sparse_mode=DEEPSEEK_V32_SFA["sparse_mode"],
                attention_mode=DEEPSEEK_V32_SFA["attention_mode"],
            )
            return out
        out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=k_nope_cache,
            value=k_nope_cache,
            sparse_indices=sparse_indices,
            scale_value=scale,
            sparse_block_size=DEEPSEEK_V32_SFA["sparse_block_size"],
            block_table=attn_metadata.block_table,
            actual_seq_lengths_query=cum_query_lens,
            actual_seq_lengths_kv=seq_lens,
            query_rope=q_pe,
            key_rope=k_rope_cache,
            layout_query=DEEPSEEK_V32_SFA["layout_query"],
            layout_kv=DEEPSEEK_V32_SFA["layout_kv"],
            sparse_mode=DEEPSEEK_V32_SFA["sparse_mode"],
            attention_mode=DEEPSEEK_V32_SFA["attention_mode"],
        )
        return out

    def _run_kv_quant() -> torch.Tensor:
        query = torch.cat([ql_nope, q_pe], dim=-1)
        return torch_npu.npu_kv_quant_sparse_flash_attention(
            query=query,
            key=k_nope_cache,
            value=k_nope_cache,
            sparse_indices=sparse_indices,
            scale_value=scale,
            sparse_block_size=DEEPSEEK_V32_SFA["sparse_block_size"],
            block_table=attn_metadata.block_table,
            actual_seq_lengths_query=cum_query_lens,
            actual_seq_lengths_kv=seq_lens,
            layout_query=DEEPSEEK_V32_SFA["layout_query"],
            layout_kv=DEEPSEEK_V32_SFA["layout_kv"],
            sparse_mode=DEEPSEEK_V32_SFA["sparse_mode"],
            attention_mode=DEEPSEEK_V32_SFA["attention_mode"],
            quant_scale_repo_mode=1,
            tile_size=128,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=DEEPSEEK_V32_SFA["qk_rope_head_dim"],
        )

    return _run_kv_quant if use_kv_quant else _run_bf16


def _prepare_tensors(
    case: BenchCase,
    args: argparse.Namespace,
    device: torch.device,
    use_kv_quant: bool,
) -> tuple[Callable[[], torch.Tensor], dict]:
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

    if use_kv_quant:
        kv_dtype = torch.float8_e4m3fn
        k_nope_cache = torch.randn(block_num, block_size, 1, kv_lora_rank, dtype=kv_dtype, device=device)
    else:
        k_nope_cache = torch.randn(block_num, block_size, 1, kv_lora_rank, dtype=dtype, device=device)
    k_rope_cache = torch.randn(block_num, block_size, 1, rope_dim, dtype=dtype, device=device)

    sparse_indices = _make_sparse_indices(b, s_q, s_kv, sparse_count, device)

    cum_query_lens = torch.tensor(
        [s_q * (i + 1) for i in range(b)],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full((b,), s_kv, dtype=torch.int32, device=device)

    use_torch_npu = get_ascend_device_type() == AscendDeviceType.A5
    run_fn = _make_run_fn(
        use_kv_quant=use_kv_quant,
        use_torch_npu=use_torch_npu,
        ql_nope=ql_nope,
        q_pe=q_pe,
        k_nope_cache=k_nope_cache,
        k_rope_cache=k_rope_cache,
        sparse_indices=sparse_indices,
        block_table=block_table,
        cum_query_lens=cum_query_lens,
        seq_lens=seq_lens,
        scale=scale,
    )
    info = {
        "q_tokens": total_q,
        "kv_tokens": b * s_kv,
        "sparse_count": sparse_count,
    }
    return run_fn, info


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
    print("DeepSeek-V3.2 Sparse Flash Attention (SFA) Benchmark")
    print(f"  device_type  : {device_type.name}")
    print(f"  attention op : {op_name}")
    print(f"  kv_lora_rank : {DEEPSEEK_V32_SFA['kv_lora_rank']}")
    print(f"  rope_dim     : {DEEPSEEK_V32_SFA['qk_rope_head_dim']}")
    print(f"  index_topk   : {DEEPSEEK_V32_SFA['index_topk']}")
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

    device_type = get_ascend_device_type()
    if args.kv_quant and device_type != AscendDeviceType.A5:
        print("ERROR: --kv-quant is only supported on A5.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    torch.npu.set_device(device)
    use_kv_quant = args.kv_quant
    use_torch_npu = device_type == AscendDeviceType.A5
    op_name = _resolve_op_name(use_kv_quant, use_torch_npu)

    cases = _build_cases(args)
    _print_header(device_type, op_name)

    results: list[BenchResult] = []
    for case in cases:
        try:
            run_fn, info = _prepare_tensors(case, args, device, use_kv_quant)
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
            "config": DEEPSEEK_V32_SFA,
            "args": vars(args),
            "results": [asdict(r) for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
