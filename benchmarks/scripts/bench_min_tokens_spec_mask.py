#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbench for MinTokensLogitsProcessor.apply_with_spec_decode overhead.

Isolates per-verify-step cost of:
  1) CPU index expand (numpy)
  2) async H2D x2 (rows, toks)
  3) logits.index_put_(..., -inf)

This mirrors upstream vLLM ``apply_with_spec_decode`` used by
``AscendRejectionSampler.apply_logits_processors``.

Example (on Ascend NPU)::

    python benchmarks/scripts/bench_min_tokens_spec_mask.py \\
        --batch 16 --draft 3 --vocab 129280 --stop-tokens 2 \\
        --iters 200 --warmup 50 --tpot-ms 20

Interpretation:
  - If ``us/step`` << TPOT (e.g. <1% of --tpot-ms), H2D/index_put_ is not the TPOT cause.
  - Compare ``no_sync`` vs ``sync_each_step``: large gap means implicit sync bubbles in real serve.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np
import torch


def _async_tensor_h2d(
    data: list | np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Same contract as vllm.utils.torch_utils.async_tensor_h2d (no vllm import)."""
    if isinstance(data, np.ndarray):
        t = torch.from_numpy(data)
        if not t.is_pinned():
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
    else:
        t = torch.tensor(data, dtype=dtype, device="cpu")
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t.to(device=device, dtype=dtype, non_blocking=True)


def build_index(
    batch: int,
    draft: int,
    stop_tokens: list[int],
    remaining: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (rows, toks) like apply_with_spec_decode for uniform batch."""
    num_draft = np.full(batch, draft, dtype=np.int64)
    cumsum = np.concatenate([[0], np.cumsum(num_draft)])
    n_stop = len(stop_tokens)
    all_rows: list[np.ndarray] = []
    all_toks: list[np.ndarray] = []
    for req_idx in range(batch):
        n_mask = int(min(max(remaining, 0), num_draft[req_idx]))
        if n_mask <= 0:
            continue
        offset = cumsum[req_idx]
        row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
        all_rows.append(np.repeat(row_indices, n_stop))
        all_toks.append(np.tile(np.asarray(stop_tokens, dtype=np.int64), n_mask))
    if not all_rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(all_rows), np.concatenate(all_toks)


def run_once(
    logits: torch.Tensor,
    rows_np: np.ndarray,
    toks_np: np.ndarray,
    neg_inf: torch.Tensor,
    *,
    sync: bool,
) -> None:
    device = logits.device
    rows = _async_tensor_h2d(rows_np, device=device)
    toks = _async_tensor_h2d(toks_np, device=device)
    logits.index_put_((rows, toks), neg_inf)
    if sync:
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()


def timed_loop(
    logits: torch.Tensor,
    rows_np: np.ndarray,
    toks_np: np.ndarray,
    neg_inf: torch.Tensor,
    *,
    iters: int,
    warmup: int,
    sync_each_step: bool,
) -> list[float]:
    device = logits.device
    for _ in range(warmup):
        run_once(logits, rows_np, toks_np, neg_inf, sync=True)

    samples_ms: list[float] = []
    for _ in range(iters):
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once(logits, rows_np, toks_np, neg_inf, sync=sync_each_step)
        if not sync_each_step:
            if device.type == "npu":
                torch.npu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1e3)
    return samples_ms


def summarize(name: str, samples_ms: list[float], tpot_ms: float | None) -> dict[str, Any]:
    mean = statistics.fmean(samples_ms)
    p50 = statistics.median(samples_ms)
    p90 = sorted(samples_ms)[max(0, int(len(samples_ms) * 0.9) - 1)]
    out: dict[str, Any] = {
        "name": name,
        "mean_ms": mean,
        "mean_us": mean * 1e3,
        "p50_ms": p50,
        "p90_ms": p90,
        "iters": len(samples_ms),
    }
    if tpot_ms and tpot_ms > 0:
        out["pct_of_tpot"] = 100.0 * mean / tpot_ms
    return out


def pick_device(prefer: str) -> torch.device:
    if prefer == "npu" or (prefer == "auto" and hasattr(torch, "npu") and torch.npu.is_available()):
        import torch_npu  # noqa: F401

        return torch.device("npu:0")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=16, help="num requests in verify batch")
    p.add_argument("--draft", type=int, default=3, help="num_draft_tokens per request (MTP K)")
    p.add_argument("--vocab", type=int, default=129280, help="vocab size (DeepSeek ~129k)")
    p.add_argument("--stop-tokens", type=int, default=2, help="|all_stop_token_ids|")
    p.add_argument(
        "--remaining",
        type=int,
        default=10**9,
        help="min_tokens - current_len; large => mask all draft rows",
    )
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    p.add_argument(
        "--tpot-ms",
        type=float,
        default=None,
        help="your measured TPOT in ms; report overhead as percentage",
    )
    p.add_argument("--dump-json", type=str, default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    num_rows = args.batch * args.draft
    stop_tokens = list(range(args.stop_tokens))  # fake ids 0..K-1
    rows_np, toks_np = build_index(args.batch, args.draft, stop_tokens, args.remaining)
    index_elems = int(rows_np.size)

    print(
        f"[cfg] device={device} batch={args.batch} draft={args.draft} "
        f"vocab={args.vocab} stop_tokens={args.stop_tokens} "
        f"logits_rows={num_rows} index_elems={index_elems}"
    )

    logits = torch.zeros(num_rows, args.vocab, device=device, dtype=torch.float32)
    neg_inf = torch.tensor(-float("inf"), dtype=torch.float32, device=device)

    # Path A: host work + H2D + index_put, sync once at end of each timed iter
    # (closer to "async H2D overlapped", still includes kernel completion)
    samples_async = timed_loop(
        logits,
        rows_np,
        toks_np,
        neg_inf,
        iters=args.iters,
        warmup=args.warmup,
        sync_each_step=False,
    )
    # Path B: sync after every op — upper bound if serve hits a sync bubble
    samples_sync = timed_loop(
        logits,
        rows_np,
        toks_np,
        neg_inf,
        iters=args.iters,
        warmup=args.warmup,
        sync_each_step=True,
    )

    # Cached apply() style: H2D once, then only index_put_ each step
    rows_dev = _async_tensor_h2d(rows_np, device=device)
    toks_dev = _async_tensor_h2d(toks_np, device=device)
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(args.warmup):
        logits.index_put_((rows_dev, toks_dev), neg_inf)
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    samples_cached: list[float] = []
    for _ in range(args.iters):
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits.index_put_((rows_dev, toks_dev), neg_inf)
        if device.type == "npu":
            torch.npu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        samples_cached.append((time.perf_counter() - t0) * 1e3)

    results = [
        summarize("spec_mask_h2d_index_put", samples_async, args.tpot_ms),
        summarize("spec_mask_h2d_index_put_sync_each", samples_sync, args.tpot_ms),
        summarize("cached_index_put_only", samples_cached, args.tpot_ms),
    ]
    for r in results:
        line = (
            f"[{r['name']}] mean={r['mean_us']:.1f} us/step "
            f"(p50={r['p50_ms']*1e3:.1f} us, p90={r['p90_ms']*1e3:.1f} us)"
        )
        if "pct_of_tpot" in r:
            line += f"  → {r['pct_of_tpot']:.3f}% of TPOT={args.tpot_ms} ms"
        print(line)

    verdict = results[0]
    if args.tpot_ms and args.tpot_ms > 0:
        pct = verdict["pct_of_tpot"]
        if pct < 1.0:
            print(
                f"[verdict] ~{pct:.3f}% of TPOT → H2D/index_put_ is UNLIKELY the TPOT regression cause."
            )
        elif pct < 5.0:
            print(
                f"[verdict] ~{pct:.3f}% of TPOT → minor; check sync_each path and long-seq decode next."
            )
        else:
            print(
                f"[verdict] ~{pct:.3f}% of TPOT → meaningful; investigate H2D/sync on this device."
            )
    else:
        print(
            "[hint] pass --tpot-ms <your_tpot> to auto-judge. "
            "Also A/B serve with ignore_eos + fixed max_tokens, "
            "min_tokens=0 vs min_tokens=max_tokens."
        )

    if args.dump_json:
        payload = {
            "cfg": {
                "device": str(device),
                "batch": args.batch,
                "draft": args.draft,
                "vocab": args.vocab,
                "stop_tokens": args.stop_tokens,
                "index_elems": index_elems,
                "tpot_ms": args.tpot_ms,
            },
            "results": results,
        }
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"[dump] {args.dump_json}")


if __name__ == "__main__":
    main()
