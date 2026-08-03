#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Performance analysis harness for ``npu_kv_quant_sparse_flash_attention``.

Aligned with Ascend C operator profiling guidance:
  - Collect: torch_npu.profiler (PipeUtilization) and/or msprof
  - Analyze: pipe util, block dim / core use, theory vs measured bound
  Docs:
    https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90/%E8%8E%B7%E5%8F%96%E6%80%A7%E8%83%BD%E6%95%B0%E6%8D%AE.md
    https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90/%E5%88%86%E6%9E%90%E6%80%A7%E8%83%BD%E6%95%B0%E6%8D%AE.md

Typical 910B2 DS32-C8 scenarios::

    # Default uses torch_npu (CANN op). Avoid _C_ascend if you see tilingKey miss.
    python benchmarks/scripts/analyze_kv_quant_sparse_flash_attention.py \\
        --scenario decode --batch 1 --kv 12288 --heads 128 --profile

    # 16-concurrency decode @12k
    python benchmarks/scripts/analyze_kv_quant_sparse_flash_attention.py \\
        --scenario decode --batch 16 --kv 12288 --heads 128 --profile

    # Prefill chunk
    python benchmarks/scripts/analyze_kv_quant_sparse_flash_attention.py \\
        --scenario prefill --batch 1 --t 2048 --kv 12288 --heads 128 --profile

    # Custom path only if your vllm_ascend_C + custom opp match host tiling:
    python benchmarks/scripts/analyze_kv_quant_sparse_flash_attention.py \\
        --scenario decode --batch 1 --kv 12288 --heads 128 --backend custom --profile

Notes:
  - Default ``--backend torch_npu``. Do **not** load ``_C_ascend`` in the same process.
  - ``not find tilingKey[578]`` means host tiling/custom kernel binary mismatch for
    ``KvQuantSparseFlashAttention``; use torch_npu or rebuild custom opp.
  - After ``--profile``, run ``analyse()`` on the generated ``*_ascend_pt`` dir,
    then inspect ``op_summary_*.csv`` / PipeUtilization.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path

import torch
import torch_npu

torch_npu.npu.config.allow_internal_format = True

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_TOPK = 2048
TILE_SIZE = 128
BLOCK_SIZE = 128
PACKED_KV_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM * 2 + (KV_LORA_RANK // TILE_SIZE) * 4
S2_BASE_SIZE = 512  # A2/A3 V_TEMPLATE outer tile
ASSERT_PACKED = PACKED_KV_DIM == 656

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
    try:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            return False
        return hasattr(torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention")
    except Exception as e:
        log(f"[warn] custom op unavailable: {type(e).__name__}: {e}")
        return False


def device_info() -> dict:
    if not torch.npu.is_available():
        raise SystemExit("NPU is required")
    props = torch.npu.get_device_properties(0)
    info = {
        "name": torch.npu.get_device_name(0),
        "total_memory_mb": getattr(props, "total_memory", None),
        "cube_core_num": getattr(props, "cube_core_num", None),
        "vector_core_num": getattr(props, "vector_core_num", None),
        "L2_cache_size": getattr(props, "L2_cache_size", None),
    }
    return info


def make_inputs(
    *,
    batch: int,
    q_tokens_per_req: int,
    kv_seq: int,
    num_heads: int,
    sparse_count: int = INDEX_TOPK,
    block_size: int = BLOCK_SIZE,
    seed: int = 1024,
) -> dict:
    """Build TND + PA_BSND packed-C8 inputs for B requests (same kv_seq each)."""
    assert ASSERT_PACKED
    torch.manual_seed(seed)
    device = "npu"

    if kv_seq % block_size != 0:
        raise ValueError(f"kv_seq ({kv_seq}) must be divisible by block_size ({block_size})")
    if sparse_count > kv_seq:
        raise ValueError(f"sparse_count ({sparse_count}) must be <= kv_seq ({kv_seq})")
    if q_tokens_per_req <= 0 or batch <= 0:
        raise ValueError("batch and q_tokens_per_req must be > 0")

    total_q = batch * q_tokens_per_req
    num_blocks = kv_seq // block_size
    kv_heads = 1
    scale_value = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) ** -0.5

    q_nope = torch.empty(
        (total_q, num_heads, KV_LORA_RANK), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    q_pe = torch.empty(
        (total_q, num_heads, QK_ROPE_HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    query = torch.cat((q_nope, q_pe), dim=-1)

    # Shared physical pages across batch for simplicity (still valid PA mapping).
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
    value = key

    sparse_indices = torch.full(
        (total_q, kv_heads, sparse_count), -1, dtype=torch.int32, device=device
    )
    for b in range(batch):
        for local_t in range(q_tokens_per_req):
            t = b * q_tokens_per_req + local_t
            # Causal-ish valid end for decode/prefill within this request.
            if q_tokens_per_req == 1:
                valid_end = kv_seq
            else:
                # Prefill: position local_t sees [0, kv_seq - q_tokens + local_t]
                valid_end = kv_seq - q_tokens_per_req + local_t + 1
            take = min(sparse_count, valid_end)
            sparse_indices[t, 0, :take] = torch.randperm(
                valid_end, device=device, dtype=torch.int32
            )[:take]

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(1, num_blocks)
    block_table = block_table.repeat(batch, 1)

    actual_seq_lengths_query = torch.arange(
        q_tokens_per_req,
        total_q + 1,
        q_tokens_per_req,
        dtype=torch.int32,
        device=device,
    )
    actual_seq_lengths_kv = torch.full((batch,), kv_seq, dtype=torch.int32, device=device)

    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse_indices,
        "block_table": block_table,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "scale_value": scale_value,
        "meta": {
            "batch": batch,
            "q_tokens_per_req": q_tokens_per_req,
            "total_q": total_q,
            "kv_seq": kv_seq,
            "num_heads": num_heads,
            "sparse_count": sparse_count,
            "block_size": block_size,
            "num_blocks": num_blocks,
        },
    }


def run_op(inputs: dict, *, backend: str):
    kwargs = dict(
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
    if backend == "custom":
        out = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention(
            **kwargs, return_softmax_lse=False
        )
        return out[0] if isinstance(out, tuple) else out
    return torch_npu.npu_kv_quant_sparse_flash_attention(**kwargs)


def estimate_work(meta: dict) -> dict:
    """Rough analytical model for DS32-C8 on A2 (outer S2 tile=512)."""
    t = meta["total_q"]
    g = meta["num_heads"]
    topk = meta["sparse_count"]
    # Decode / late prefill: each Q sees ~topk; early prefill is less — use topk upper bound.
    s2_tiles = math.ceil(topk / S2_BASE_SIZE)
    # Merged KV bytes written to workspace per Q (bf16 nope+rope), 4-buf not counted.
    merge_bytes_per_q = topk * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
    # Mm1 FLOPs ~ 2 * M * K * N with M=G, K=576, N=topk (MAC count order-of-magnitude).
    mm1_flops_per_q = 2 * g * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * topk
    mm2_flops_per_q = 2 * g * topk * KV_LORA_RANK
    return {
        "s1_tokens": t,
        "g_heads": g,
        "sparse_s2_per_q": topk,
        "s2_outer_tiles_per_q": s2_tiles,
        "expected_busy_cube_groups_cap": min(t, meta.get("cube_core_num") or t),
        "merge_bytes_total_mb": t * merge_bytes_per_q / (1024**2),
        "mm1_flops_total_g": t * mm1_flops_per_q / 1e9,
        "mm2_flops_total_g": t * mm2_flops_per_q / 1e9,
        "note": (
            "Kernel balances by Q-token count and does not split S2 across cores "
            "(FLASH_DECODE=0). batch/Q<=cube_cores => ~1 Q per Mix group."
        ),
    }


def bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def build_profiler(out_dir: Path):
    """Match vllm_ascend TorchNPUProfilerWrapper: Level1 + PipeUtilization."""
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
        experimental_config=experimental_config,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    )


def run_profile(fn, out_dir: Path, steps: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = build_profiler(out_dir)
    prof.start()
    try:
        for _ in range(steps):
            fn()
            torch.npu.synchronize()
    finally:
        prof.stop()
    log(f"[profile] raw traces under {out_dir}")
    log("[profile] next: analyse traces, e.g.")
    log("  python - <<'PY'")
    log("  from torch_npu.profiler.profiler import analyse")
    log(f"  analyse(r'{out_dir}')")
    log("  PY")
    log(
        "[profile] then inspect ASCEND_PROFILER_OUTPUT / op_summary_*.csv / "
        "PipeUtilization for aic_mac_ratio, aic_mte2_ratio, aiv_mte2_time, "
        "Block Dim, Duration (see Ascend 获取/分析性能数据)."
    )


def print_analysis_checklist(work: dict, ms: float, info: dict) -> None:
    log("\n========== Analysis checklist (Ascend profiling guide) ==========")
    log("1) PipeUtilization: which pipe is longest?")
    log("   - high aiv_mte2_*  => MergeKv discrete gather bound")
    log("   - high aic_mte2_*  => Cube loading merged K/V (GM/L2) bound")
    log("   - low aic_mac_ratio + low mte => sync / small-tile under-util")
    log("2) Tiling / Block Dim:")
    log(
        f"   - S1(Q tokens)={work['s1_tokens']}, cube_cores={info.get('cube_core_num')}, "
        f"expect busy groups ≈ {work['expected_busy_cube_groups_cap']}"
    )
    log(
        f"   - if Block Dim << {info.get('cube_core_num')}, core waste "
        "(decode low concurrency / no S2 split)"
    )
    log("3) Theory vs measured (order-of-magnitude only):")
    log(f"   - merged KV traffic ~ {work['merge_bytes_total_mb']:.2f} MB (bf16 nope+rope)")
    log(f"   - Mm1 FLOPs ~ {work['mm1_flops_total_g']:.2f} GFLOPs, Mm2 ~ {work['mm2_flops_total_g']:.2f} GFLOPs")
    log(f"   - measured avg latency = {ms:.3f} ms")
    log("4) Head overhead: for us-level ops, compare empty-kernel / few-iter Duration.")
    log("5) Simulator (optional): msOpProf / trace.json in chrome://tracing or MindStudio Insight.")
    log("================================================================\n")


def print_msprof_cmd(argv_tail: list[str], out_dir: Path) -> None:
    # Wrapper style suggested by Ascend '获取性能数据' (msprof around executable).
    script = str(Path(__file__).resolve())
    py = sys.executable
    inner = [py, script, *argv_tail]
    cmd = [
        "msprof",
        "--output",
        str(out_dir),
        "--aic-metrics=PipeUtilization",
        "--",
        *inner,
    ]
    log("[msprof] example command (adjust flags to your CANN version):")
    log("  " + " ".join(shlex.quote(c) for c in cmd))
    log(
        "[msprof] after run, check PipeUtilization.csv / op_summary for "
        "cube/vector/MTE ratios (获取性能数据): "
        "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90/%E8%8E%B7%E5%8F%96%E6%80%A7%E8%83%BD%E6%95%B0%E6%8D%AE.md"
    )


def resolve_scenario(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return (batch, q_tokens_per_req, kv_seq)."""
    if args.scenario == "decode":
        return args.batch, 1, args.kv
    if args.scenario == "prefill":
        return args.batch, args.t, args.kv
    # custom
    return args.batch, args.t, args.kv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("decode", "prefill", "custom"),
        default="decode",
        help="decode: T=batch*1; prefill: T=batch*t",
    )
    parser.add_argument("--batch", type=int, default=1, help="concurrency / request count")
    parser.add_argument("--t", type=int, default=2048, help="Q tokens per request (prefill/custom)")
    parser.add_argument("--kv", type=int, default=12288, help="KV seq length per request")
    parser.add_argument(
        "--heads",
        type=int,
        default=128,
        help=(
            "local Q heads G (query.shape[1]). Default 128 matches DS MLA full heads / "
            "existing bench. For TP-sharded local heads use 8/16/32/...; if custom op "
            "raises 'not find tilingKey', try --heads 128 or --backend torch_npu."
        ),
    )
    parser.add_argument("--topk", type=int, default=INDEX_TOPK)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--backend",
        choices=("custom", "torch_npu"),
        default="torch_npu",
        help=(
            "torch_npu: CANN built-in op (default; use for profiling when custom "
            "tilingKey is missing). custom: torch.ops._C_ascend (needs matching "
            "ASCEND_CUSTOM_OPP_PATH / rebuilt kernels)."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect torch_npu.profiler PipeUtilization traces",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="qsfa_prof",
        help="profiler output directory",
    )
    parser.add_argument(
        "--print-msprof",
        action="store_true",
        help="print an msprof wrapper command and exit",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default=None,
        help="optional path to dump timing + work estimate JSON",
    )
    args = parser.parse_args()

    batch, q_per_req, kv_seq = resolve_scenario(args)

    if args.print_msprof:
        out = Path(args.profile_dir)
        tail = [
            "--scenario",
            args.scenario,
            "--batch",
            str(batch),
            "--t",
            str(q_per_req),
            "--kv",
            str(kv_seq),
            "--heads",
            str(args.heads),
            "--topk",
            str(args.topk),
            "--backend",
            args.backend,
            "--warmup",
            str(min(args.warmup, 3)),
            "--iters",
            str(min(args.iters, 10)),
        ]
        print_msprof_cmd(tail, out)
        return

    info = device_info()
    log(f"[device] {json.dumps(info, ensure_ascii=False)}")

    if args.backend == "custom" and not try_enable_custom_op():
        raise SystemExit("custom backend unavailable; try --backend torch_npu in a clean process")

    inputs = make_inputs(
        batch=batch,
        q_tokens_per_req=q_per_req,
        kv_seq=kv_seq,
        num_heads=args.heads,
        sparse_count=args.topk,
        seed=args.seed,
    )
    meta = dict(inputs["meta"])
    meta["cube_core_num"] = info.get("cube_core_num")
    work = estimate_work(meta)

    log(
        f"[shape] scenario={args.scenario} B={batch} q_per_req={q_per_req} "
        f"T={meta['total_q']} G={args.heads} kv={kv_seq} topk={args.topk} "
        f"key_dim={PACKED_KV_DIM} s2_tile={S2_BASE_SIZE}"
    )
    log(f"[model] {json.dumps(work, ensure_ascii=False)}")

    def _once():
        return run_op(inputs, backend=args.backend)

    # Correctness smoke
    try:
        out = _once()
        torch.npu.synchronize()
    except Exception as e:
        msg = str(e)
        log(f"[error] op launch failed: {type(e).__name__}: {e}")
        if "tilingKey" in msg or "tiling" in msg.lower() or "EZ1001" in msg:
            log(
                "[hint] Host tiling produced a key not present in the loaded kernel binary.\n"
                "  This is a CANN custom-opp / vllm_ascend_C mismatch (not heads/shape).\n"
                "  Fix:\n"
                "    1) Open a NEW process (do not load _C_ascend) and run:\n"
                "         --backend torch_npu --heads 128 --profile\n"
                "    2) Or rebuild/install matching custom opp under\n"
                "         vllm_ascend/_cann_ops_custom/vendors/custom_transformer\n"
                "       so KvQuantSparseFlashAttention contains tilingKey 578\n"
                "       (TND + PA_BSND + V_TEMPLATE).\n"
                f"  Current: backend={args.backend} heads={args.heads} "
                f"T={meta['total_q']} kv={kv_seq}"
            )
        raise
    log(f"[smoke] out={tuple(out.shape)} dtype={out.dtype}")

    ms = bench(_once, warmup=args.warmup, iters=args.iters)
    log(f"[bench] backend={args.backend} avg={ms:.3f} ms  (warmup={args.warmup}, iters={args.iters})")

    print_analysis_checklist(work, ms, info)

    if args.profile:
        tag = f"{args.scenario}_b{batch}_t{q_per_req}_kv{kv_seq}_h{args.heads}_{args.backend}"
        prof_dir = Path(args.profile_dir) / tag
        # Short warmup outside profiler
        for _ in range(3):
            _once()
        torch.npu.synchronize()
        run_profile(_once, prof_dir, steps=5)

    if args.dump_json:
        payload = {
            "device": info,
            "meta": meta,
            "work": work,
            "backend": args.backend,
            "latency_ms": ms,
            "out_shape": list(out.shape),
        }
        Path(args.dump_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"[dump] wrote {args.dump_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(f"[error] {sys.exc_info()[1]!r}")
        raise
