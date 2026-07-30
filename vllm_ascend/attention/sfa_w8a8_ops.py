# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared W8A8 + int8 per-tile KV SFA decode operators (A2/A3).

Single source of truth for:
  1. A2/A3 C8 preprocess (mla_preprocess + per-tile KV pack/scatter) and A5 mxfp8 weight prep
  2. npu_mla_prolog_v3 kwargs (legacy / A5) and lightning indexer top-k (sparse_count=2048)
  3. npu_kv_quant_sparse_flash_attention vs npu_sparse_flash_attention

Used by BaseDeviceAdaptor (device_op.py) and NPU unit tests / perf benches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.attention.utils import (
    round_up,
    trans_rope_weight,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

INDEXER_SPARSE_COUNT = 2048

MLA_PROLOG_V3_TILE_SIZE = 128
MLA_PROLOG_V3_KV_SCALE_METADATA_BYTES = 4 * 4
# CANN MlaPrologV3 rope_mode: 0=interleave (GatherMask), 1=split-half NeoX (mla_preprocess style).
MLA_PROLOG_V3_ROPE_MODE_INTERLEAVE = 0
MLA_PROLOG_V3_ROPE_MODE_HALF = 1


def get_mla_prolog_v3_per_tile_kv_dim(kv_lora_rank: int, qk_rope_head_dim: int) -> int:
    """Per-tile KV cache last dim for npu_mla_prolog_v3 kv_cache_quant_mode=3 / merged Dtile."""
    return kv_lora_rank + qk_rope_head_dim * 2 + MLA_PROLOG_V3_KV_SCALE_METADATA_BYTES


def get_mlapo_query_num_heads(sfa_impl: Any) -> int:
    """Head count for MLAPO Q prolog; DSA-CP replicates full q_b_proj on each rank."""
    return getattr(sfa_impl, "local_num_heads", sfa_impl.num_heads)


def _transdata_nz_4d(nd_mat: torch.Tensor, block_size: tuple[int, int] = (16, 32)) -> torch.Tensor:
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    nd_mat = F.pad(nd_mat, (0, c - nd_mat.shape[1], 0, r - nd_mat.shape[0]))
    return torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    ).contiguous()


def cast_mlapo_nz_he_out(weight_he_out: torch.Tensor) -> torch.Tensor:
    """int8 [He, Out] -> FRACTAL_NZ; shared by A5 and A2/A3 npu_mla_prolog_v3."""
    return torch_npu.npu_format_cast(weight_he_out.contiguous(), ACL_FORMAT_FRACTAL_NZ)


def cast_prolog_v3_nz_hcq_out(weight_hcq_out: torch.Tensor) -> torch.Tensor:
    """int8 [Hcq, N*(D+Dr)] -> FRACTAL_NZ 4D transdata (weight_uq_qr)."""
    return torch_npu.npu_format_cast(_transdata_nz_4d(weight_hcq_out), ACL_FORMAT_FRACTAL_NZ)


def slice_fused_dkv_kr_weight(
    fused_qkv_weight: torch.Tensor,
    q_lora_rank: int,
) -> torch.Tensor:
    """ND int8 [He, Hckv+Dr] — same slice as A5 ``_process_weights_for_fused_mlapo_a5``."""
    return fused_qkv_weight[..., q_lora_rank:].contiguous()


def trans_rope_dkv_kr_weight(
    dkv_kr_weight: torch.Tensor,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """RoPE dim reorder on kv slice; required for A2 int8 prolog (interleave RoPE), not A5 mxfp8."""
    return trans_rope_weight(dkv_kr_weight.t(), qk_rope_head_dim).t().contiguous()


def prepare_prolog_v3_dkv_kr_weight(
    fused_qkv_weight: torch.Tensor, q_lora_rank: int, qk_rope_head_dim: int
) -> torch.Tensor:
    """A2/A3 int8 prolog: A5 slice + trans_rope (ND, before NZ cast)."""
    dkv_kr_nd = slice_fused_dkv_kr_weight(fused_qkv_weight, q_lora_rank)
    return trans_rope_dkv_kr_weight(dkv_kr_nd, qk_rope_head_dim)


def prepare_prolog_v3_dkv_kr_deq_scale(
    fused_deq_scale: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """Per-channel dequant for weight_dkv_kr; matches trans_rope on weight columns."""
    dkv_kr_dim = kv_lora_rank + qk_rope_head_dim
    dkv_deq = fused_deq_scale[q_lora_rank:].reshape(dkv_kr_dim, -1).contiguous()
    return trans_rope_weight(dkv_deq, qk_rope_head_dim).view(1, -1)


def prepare_prolog_v3_uq_qr_weight(
    q_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_lora_rank: int,
) -> torch.Tensor:
    n_uq_qr = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    wu = q_proj_weight.t().reshape(num_heads, qk_nope_head_dim + qk_rope_head_dim, q_lora_rank)
    wu = trans_rope_weight(wu, qk_rope_head_dim)
    return wu.reshape(n_uq_qr, q_lora_rank).t().contiguous()


def prepare_prolog_v3_uq_qr_deq_scale(
    q_proj_deq_scale: torch.Tensor, num_heads: int, qk_nope_head_dim: int, qk_rope_head_dim: int
) -> torch.Tensor:
    n_uq_qr = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    scale = q_proj_deq_scale.reshape(num_heads, qk_nope_head_dim + qk_rope_head_dim, -1)
    scale = trans_rope_weight(scale, qk_rope_head_dim)
    return scale.reshape(n_uq_qr).contiguous().view(1, -1)


def prepare_prolog_v3_dq_bias(
    fused_quant_bias: torch.Tensor,
    q_lora_rank: int,
) -> torch.Tensor:
    return fused_quant_bias[:q_lora_rank].to(torch.int32).contiguous().view(1, -1)


def prepare_prolog_v3_dkv_kr_bias(
    fused_quant_bias: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    dkv_kr_dim = kv_lora_rank + qk_rope_head_dim
    dkv_bias = fused_quant_bias[q_lora_rank:].reshape(dkv_kr_dim, -1).contiguous()
    return trans_rope_weight(dkv_bias, qk_rope_head_dim).view(1, -1).to(torch.int32)


def prepare_prolog_v3_uq_qr_bias(
    q_proj_quant_bias: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    n_uq_qr = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    bias = q_proj_quant_bias.reshape(num_heads, qk_nope_head_dim + qk_rope_head_dim, -1)
    bias = trans_rope_weight(bias, qk_rope_head_dim)
    return bias.reshape(n_uq_qr).contiguous().view(1, -1).to(torch.int32)


def slice_fused_dq_weight(
    fused_qkv_weight: torch.Tensor,
    q_lora_rank: int,
) -> torch.Tensor:
    """ND int8 [He, Hcq] — shared by A5 and A2/A3 prolog weight prep."""
    return fused_qkv_weight[..., :q_lora_rank].contiguous()


def prepare_mlapo_weight_dq(
    fused_qkv_weight: torch.Tensor,
    q_lora_rank: int,
) -> torch.Tensor:
    """Shared ``weight_dq``: A5 and A2 prolog_v3 use the same slice + FRACTAL_NZ cast."""
    return cast_mlapo_nz_he_out(slice_fused_dq_weight(fused_qkv_weight, q_lora_rank))


@dataclass
class MlapoPrologV3PreparedWeights:
    """NZ weights + per-channel dequant scales for ``weight_quant_mode=2`` (A2/A3 C8)."""

    weight_dq: torch.Tensor
    weight_uq_qr: torch.Tensor
    weight_dkv_kr: torch.Tensor
    dequant_scale_w_dq: torch.Tensor
    dequant_scale_w_uq_qr: torch.Tensor
    dequant_scale_w_dkv_kr: torch.Tensor
    bias_w_dq: torch.Tensor | None = None
    bias_w_dkv_kr: torch.Tensor | None = None
    bias_w_uq_qr: torch.Tensor | None = None


def prepare_mlapo_prolog_v3_weights(
    fused_qkv_weight: torch.Tensor,
    fused_deq_scale: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_deq_scale: torch.Tensor,
    *,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    fused_quant_bias: torch.Tensor | None = None,
    q_proj_quant_bias: torch.Tensor | None = None,
) -> MlapoPrologV3PreparedWeights:
    """Prepare NZ weights for ``npu_mla_prolog_v3`` ``weight_quant_mode=2``.

    Same ``weight_dq`` / dkv **slice** as A5; A2-only delta is ``trans_rope`` on dkv_kr (+ uq_qr layout).
    Note: ``enable_sparse_c8`` merged_dtile path uses ``_process_weights_for_fused_mlapo`` (wd_qkv), not this.
    """
    weight_dq = prepare_mlapo_weight_dq(fused_qkv_weight, q_lora_rank)
    weight_dkv_kr_nd = prepare_prolog_v3_dkv_kr_weight(
        fused_qkv_weight, q_lora_rank, qk_rope_head_dim
    )
    weight_uq_qr_nd = prepare_prolog_v3_uq_qr_weight(
        q_proj_weight, num_heads, qk_nope_head_dim, qk_rope_head_dim, q_lora_rank
    )
    bias_w_dq = None
    bias_w_dkv_kr = None
    bias_w_uq_qr = None
    if fused_quant_bias is not None:
        bias_w_dq = prepare_prolog_v3_dq_bias(fused_quant_bias, q_lora_rank)
        bias_w_dkv_kr = prepare_prolog_v3_dkv_kr_bias(
            fused_quant_bias, q_lora_rank, kv_lora_rank, qk_rope_head_dim
        )
    if q_proj_quant_bias is not None:
        bias_w_uq_qr = prepare_prolog_v3_uq_qr_bias(
            q_proj_quant_bias, num_heads, qk_nope_head_dim, qk_rope_head_dim
        )
    return MlapoPrologV3PreparedWeights(
        weight_dq=weight_dq,
        weight_uq_qr=cast_prolog_v3_nz_hcq_out(weight_uq_qr_nd),
        weight_dkv_kr=cast_mlapo_nz_he_out(weight_dkv_kr_nd),
        dequant_scale_w_dq=fused_deq_scale[:q_lora_rank].contiguous().view(1, -1),
        dequant_scale_w_dkv_kr=prepare_prolog_v3_dkv_kr_deq_scale(
            fused_deq_scale, q_lora_rank, kv_lora_rank, qk_rope_head_dim
        ),
        dequant_scale_w_uq_qr=prepare_prolog_v3_uq_qr_deq_scale(
            q_proj_deq_scale, num_heads, qk_nope_head_dim, qk_rope_head_dim
        ),
        bias_w_dq=bias_w_dq,
        bias_w_dkv_kr=bias_w_dkv_kr,
        bias_w_uq_qr=bias_w_uq_qr,
    )


def merged_kv_byte_offsets(kv_lora_rank: int, qk_rope_head_dim: int) -> tuple[int, int, int]:
    """Byte offsets in merged Dtile row: int8 k_nope | bf16 k_pe | per-tile scale meta."""
    k_nope_end = kv_lora_rank
    k_pe_end = k_nope_end + qk_rope_head_dim * 2
    scale_end = k_pe_end + MLA_PROLOG_V3_KV_SCALE_METADATA_BYTES
    return k_nope_end, k_pe_end, scale_end


def split_merged_kv_dtile(
    merged_kv: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split merged int8 Dtile into k_nope (int8), k_pe (bf16), scale metadata (int8).

    Layout per row (DeepSeek-V3.2: 512 + 128 + 16 = 656 bytes):
        [ int8 k_nope | bf16 k_pe | tile scale ]
    """
    k_nope_end, k_pe_end, scale_end = merged_kv_byte_offsets(kv_lora_rank, qk_rope_head_dim)
    prefix_shape = merged_kv.shape[:-1]
    k_nope = merged_kv[..., :k_nope_end]
    k_pe_bytes = merged_kv[..., k_nope_end:k_pe_end]
    k_pe = k_pe_bytes.view(torch.bfloat16).reshape(*prefix_shape, qk_rope_head_dim)
    scale_meta = merged_kv[..., k_pe_end:scale_end]
    return k_nope, k_pe, scale_meta


def _apply_k_nope_clip_alpha(
    k_nope: torch.Tensor,
    clip_alpha: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(clip_alpha, torch.Tensor):
        alpha = clip_alpha.to(device=k_nope.device, dtype=k_nope.dtype).reshape(())
        return k_nope * alpha
    if clip_alpha != 1.0:
        return k_nope * clip_alpha
    return k_nope


def _per_tile_symmetric_quant_k_nope_impl(
    k: torch.Tensor,
    tile_size: int,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-native per-tile symmetric int8 quant (bf16 on NPU, fp32 on CPU UT)."""
    k = k.to(compute_dtype)
    kv_dim = k.shape[-1]
    if kv_dim < tile_size:
        absmax = k.abs().amax(dim=-1).clamp(min=1e-8)
        quant = (k / absmax.unsqueeze(-1) * 127.0).round().clamp(-128, 127).to(torch.int8)
        descales = (absmax / 127.0).to(torch.float32).unsqueeze(-1)
        return quant, descales

    num_tiles = kv_dim // tile_size
    tiles = k.view(*k.shape[:-1], num_tiles, tile_size)
    absmax = tiles.abs().amax(dim=-1).clamp(min=1e-8)
    quant = (tiles / absmax.unsqueeze(-1) * 127.0).round().clamp(-128, 127).to(torch.int8)
    int8 = quant.reshape(*k.shape[:-1], kv_dim)
    descales = (absmax / 127.0).to(torch.float32)
    return int8, descales


def per_tile_symmetric_quant_k_nope(
    k_nope: torch.Tensor,
    *,
    clip_alpha: float | torch.Tensor = 1.0,
    tile_size: int = MLA_PROLOG_V3_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile symmetric int8 quant for ``kv_cache_quant_mode=3`` (CANN MlaProlog).

    For each tile of ``tile_size`` elements (last tile may be shorter when
    ``kv_lora_rank < tile_size``):
        absmax = max(|k|) * clip_alpha
        k_int8 = round(k / absmax * 127), clamped to [-128, 127]
        descale = absmax / 127   (stored in Dtile scale metadata)

    Uses device-native dtype (bf16 on NPU) — avoids ``.float()`` fp32 elementwise.
    """
    k = _apply_k_nope_clip_alpha(k_nope, clip_alpha)
    kv_dim = k.shape[-1]
    if kv_dim >= tile_size and kv_dim % tile_size != 0:
        raise ValueError(
            f"kv_lora_rank {kv_dim} must be < {tile_size} or divisible by tile_size {tile_size}"
        )
    compute_dtype = torch.float32 if k.device.type == "cpu" else k.dtype
    return _per_tile_symmetric_quant_k_nope_impl(k, tile_size, compute_dtype)


def _pad_per_tile_descales_for_dtile(
    descales: torch.Tensor,
    prefix_shape: tuple[int, ...],
    num_tiles: int,
) -> torch.Tensor:
    """Pad per-tile fp32 descales to 16-byte Dtile metadata (4 x fp32, zero-filled tail)."""
    num_fp32_slots = MLA_PROLOG_V3_KV_SCALE_METADATA_BYTES // 4
    scale_f32 = descales.to(torch.float32)
    if scale_f32.shape[-1] < num_fp32_slots:
        scale_f32 = F.pad(scale_f32, (0, num_fp32_slots - scale_f32.shape[-1]))
    return scale_f32[..., :num_fp32_slots]


def pack_merged_kv_dtile(
    k_nope_int8: torch.Tensor,
    k_pe: torch.Tensor,
    descales: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """Pack per-token int8 k_nope, bf16 k_pe, fp32 tile descales into merged Dtile row."""
    k_nope_end, k_pe_end, scale_end = merged_kv_byte_offsets(kv_lora_rank, qk_rope_head_dim)
    prefix = k_nope_int8.shape[:-1]
    k_int8 = k_nope_int8.reshape(*prefix, kv_lora_rank).to(torch.int8)
    k_pe_bf16 = k_pe.to(torch.bfloat16).reshape(*prefix, qk_rope_head_dim)
    k_pe_bytes = k_pe_bf16.contiguous().view(torch.int8).reshape(*prefix, qk_rope_head_dim * 2)
    scale_f32 = _pad_per_tile_descales_for_dtile(
        descales,
        prefix,
        max(1, kv_lora_rank // MLA_PROLOG_V3_TILE_SIZE),
    )
    scale_bytes = (
        scale_f32.contiguous()
        .view(torch.int8)
        .reshape(*prefix, MLA_PROLOG_V3_KV_SCALE_METADATA_BYTES)
    )
    merged = torch.cat([k_int8, k_pe_bytes, scale_bytes], dim=-1)
    assert merged.shape[-1] == scale_end, (
        f"packed Dtile dim {merged.shape[-1]} != expected {scale_end}"
    )
    return merged


_MERGED_DTILE_DEBUG_STAGE_NAMES: dict[int, str] = {
    100: "process_vector_k",
    110: "before_k_rmsrope",
    120: "after_k_rmsrope",
    130: "before_mm2_aiv",
    131: "after_mm2_aiv",
    200: "k_loop",
    220: "after_k_rms",
    230: "after_k_quant",
    240: "after_k_rope",
    250: "after_k_scatter",
    251: "after_k_row_write",
    400: "quant_start",
    401: "quant_tile",
    409: "quant_done",
}


def format_merged_dtile_debug_trace(debug_trace: torch.Tensor) -> str:
    """Format kernel GM trace (int32[0..3]) written by block0 vector core."""
    flat = debug_trace.reshape(-1).detach().cpu()
    if flat.numel() < 4:
        return f"debug_trace too small: numel={flat.numel()}"
    stage = int(flat[0].item())
    aux0 = int(flat[1].item())
    aux1 = int(flat[2].item())
    block = int(flat[3].item())
    name = _MERGED_DTILE_DEBUG_STAGE_NAMES.get(stage, "unknown")
    return f"stage={stage}({name}) aux0={aux0} aux1={aux1} block={block}"


def print_merged_dtile_debug_trace(debug_trace: torch.Tensor, *, where: str) -> None:
    print(f"[merged_dtile][{where}] {format_merged_dtile_debug_trace(debug_trace)}", flush=True)


def _invoke_mla_preprocess_merged_dtile(
    sfa_impl: Any,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_merged: torch.Tensor,
    slot_mapping: torch.Tensor,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_c: torch.Tensor,
    *,
    debug_trace: torch.Tensor | None = None,
) -> None:
    """Fused ``mla_preprocess_merged_dtile``: Q/K prolog + per-tile int8 merged Dtile KV write.

    Single custom op writes 656B/slot (512 int8 knope | 128 bf16 k_pe bytes | 16 fp32 scales)
    to paged ``kv_merged`` via ``slot_mapping`` (skips -1 padding slots).

    When ``debug_trace`` is set (int32, numel>=4), kernel writes stage markers for 507015 bisect.
    Also set ``VLLM_ASCEND_MERGED_DTILE_DEBUG=1`` and ``ASCEND_LAUNCH_BLOCKING=1``.
    """
    if getattr(sfa_impl, "wd_qkv", None) is None:
        raise RuntimeError(
            "mla_preprocess_merged_dtile requires mlapo preprocess weights (wd_qkv). "
            "For A2/A3 C8 layers, call _process_weights_for_fused_mlapo / "
            "_prepare_mla_preprocess_operands during weight loading."
        )
    if hidden_states.shape[0] == 0:
        return

    kv_merged = kv_merged.contiguous()
    if debug_trace is not None:
        debug_trace.fill_(-1)
    try:
        torch.ops._C_ascend.mla_preprocess_merged_dtile(
            hidden_states,
            sfa_impl.wd_qkv,
            sfa_impl.deq_scale_qkv,
            sfa_impl.gamma1,
            sfa_impl.beta1,
            sfa_impl.wu_q,
            sfa_impl.qb_deq_scl,
            sfa_impl.gamma2,
            cos,
            sin,
            sfa_impl.W_UK_T,
            kv_merged,
            slot_mapping,
            quant_scale0=sfa_impl.quant_scale0,
            quant_offset0=sfa_impl.quant_offset0,
            bias0=sfa_impl.quant_bias_qkv,
            quant_scale1=sfa_impl.quant_scale1,
            quant_offset1=sfa_impl.quant_offset1,
            bias1=sfa_impl.qb_qt_bias,
            ctkv_scale=sfa_impl.ctkv_scale,
            q_nope_scale=sfa_impl.q_nope_scale,
            # Must be allocated at weight-load time; do not use getattr(..., torch.tensor(...))
            # as the default is evaluated every forward and breaks ACL graph capture.
            k_nope_clip_alpha=sfa_impl.sfa_qsfa_k_nope_clip_alpha,
            debug_trace_out=debug_trace,
            q_out0=ql_nope,
            kv_cache_out=kv_merged,
            q_out1=q_pe,
            inner_out=q_c,
        )
    except Exception:
        if debug_trace is not None:
            try:
                torch.npu.synchronize()
            except RuntimeError:
                pass
            print_merged_dtile_debug_trace(debug_trace, where="python_after_crash")
        raise
    if debug_trace is not None:
        print_merged_dtile_debug_trace(debug_trace, where="python_after_run")


def _has_mla_preprocess_merged_dtile_rows_op() -> bool:
    return hasattr(torch.ops, "_C_ascend") and hasattr(torch.ops._C_ascend, "mla_preprocess_merged_dtile_rows")


def _invoke_mla_preprocess_merged_dtile_rows(
    sfa_impl: Any,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    dtile_rows_out: torch.Tensor,
    slot_mapping: torch.Tensor,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_c: torch.Tensor,
    *,
    debug_trace: torch.Tensor | None = None,
) -> None:
    """Fused ``mla_preprocess_merged_dtile_rows``: same Q/K prolog as merged_dtile, but writes
    contiguous ``[N, 656]`` dtile rows to ``dtile_rows_out`` (no paged-cache scatter).

    ``slot_mapping`` is still used to skip padding tokens (``-1`` entries).
    """
    if getattr(sfa_impl, "wd_qkv", None) is None:
        raise RuntimeError(
            "mla_preprocess_merged_dtile_rows requires mlapo preprocess weights (wd_qkv). "
            "For A2/A3 C8 layers, call _process_weights_for_fused_mlapo / "
            "_prepare_mla_preprocess_operands during weight loading."
        )
    if not _has_mla_preprocess_merged_dtile_rows_op():
        raise RuntimeError(
            "mla_preprocess_merged_dtile_rows is not registered; rebuild vllm_ascend with "
            "VLLM_ASCEND_BUILD_MLA_PREPROCESS_MERGED_DTILE=1."
        )
    if hidden_states.shape[0] == 0:
        return

    dtile_rows_out = dtile_rows_out.contiguous()
    if debug_trace is not None:
        debug_trace.fill_(-1)
    try:
        torch.ops._C_ascend.mla_preprocess_merged_dtile_rows(
            hidden_states,
            sfa_impl.wd_qkv,
            sfa_impl.deq_scale_qkv,
            sfa_impl.gamma1,
            sfa_impl.beta1,
            sfa_impl.wu_q,
            sfa_impl.qb_deq_scl,
            sfa_impl.gamma2,
            cos,
            sin,
            sfa_impl.W_UK_T,
            dtile_rows_out,
            slot_mapping,
            quant_scale0=sfa_impl.quant_scale0,
            quant_offset0=sfa_impl.quant_offset0,
            bias0=sfa_impl.quant_bias_qkv,
            quant_scale1=sfa_impl.quant_scale1,
            quant_offset1=sfa_impl.quant_offset1,
            bias1=sfa_impl.qb_qt_bias,
            ctkv_scale=sfa_impl.ctkv_scale,
            q_nope_scale=sfa_impl.q_nope_scale,
            k_nope_clip_alpha=sfa_impl.sfa_qsfa_k_nope_clip_alpha,
            debug_trace_out=debug_trace,
            q_out0=ql_nope,
            dtile_rows_out_ref=dtile_rows_out,
            q_out1=q_pe,
            inner_out=q_c,
        )
    except Exception:
        if debug_trace is not None:
            try:
                torch.npu.synchronize()
            except RuntimeError:
                pass
            print_merged_dtile_debug_trace(debug_trace, where="python_after_crash_rows")
        raise
    if debug_trace is not None:
        print_merged_dtile_debug_trace(debug_trace, where="python_after_run_rows")


def run_mla_preprocess_merged_dtile_op(
    sfa_impl: Any,
    hidden_states: torch.Tensor,
    kv_merged: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
    *,
    debug: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run **only** ``mla_preprocess_merged_dtile`` (no prolog_v3 / scatter / Python quant).

    Returns ``(ql_nope, q_pe, q_c)``; merged KV is written in-place to ``kv_merged``.
    """
    nheads = get_mlapo_query_num_heads(sfa_impl)
    hckv = sfa_impl.kv_lora_rank
    dr = sfa_impl.qk_rope_head_dim
    dtype = hidden_states.dtype
    device = hidden_states.device

    ql_nope = torch.empty((num_tokens, nheads, hckv), dtype=dtype, device=device)
    q_pe = torch.empty((num_tokens, nheads, dr), dtype=dtype, device=device)
    q_c = torch.empty((num_tokens, sfa_impl.q_lora_rank), dtype=dtype, device=device)

    enable_debug = (
        bool(int(os.getenv("VLLM_ASCEND_MERGED_DTILE_DEBUG", "0"))) if debug is None else debug
    )
    debug_trace = None
    if enable_debug:
        debug_trace = torch.full((4,), -1, dtype=torch.int32, device=device)
        print(
            "[merged_dtile] debug enabled: VLLM_ASCEND_MERGED_DTILE_DEBUG + "
            "recommend ASCEND_LAUNCH_BLOCKING=1",
            flush=True,
        )

    _invoke_mla_preprocess_merged_dtile(
        sfa_impl,
        hidden_states,
        cos,
        sin,
        kv_merged,
        slot_mapping,
        ql_nope,
        q_pe,
        q_c,
        debug_trace=debug_trace,
    )
    return ql_nope, q_pe, q_c


def mla_preprocess_write_per_tile_merged_kv(
    sfa_impl: Any,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_merged: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_c: torch.Tensor,
    *,
    dtile_rows_out: torch.Tensor | None = None,
) -> None:
    """``mla_preprocess`` fused per-tile int8 merged Dtile write (512+128+16 bytes).

    When ``dtile_rows_out`` is set, uses ``mla_preprocess_merged_dtile_rows`` to write
    contiguous ``[N, 656]`` rows (DSA-CP path). Otherwise scatters into paged ``kv_merged``.
    """
    if dtile_rows_out is not None:
        _invoke_mla_preprocess_merged_dtile_rows(
            sfa_impl,
            hidden_states,
            cos,
            sin,
            dtile_rows_out,
            slot_mapping,
            ql_nope,
            q_pe,
            q_c,
        )
        return
    _invoke_mla_preprocess_merged_dtile(
        sfa_impl,
        hidden_states,
        cos,
        sin,
        kv_merged,
        slot_mapping,
        ql_nope,
        q_pe,
        q_c,
    )


def resolve_mla_prolog_v3_kv_kr_caches(
    merged_kv: torch.Tensor,
    qk_rope_head_dim: int,
    sfa_impl: Any,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Resolve CANN prolog buffers for ``ckvkr_repo_mode=1`` (merged int8 Dtile).

    Production C8 write path (validated on A2/A3):
      - ``kv_cache``: merged int8 Dtile ``[..., 656]`` (512 k_nope | 128 k_pe bytes | 16 scale)
      - ``kr_cache``: empty placeholder ``(0, 0, Nkv, Dr)`` bf16
      - ``ckvkr_repo_mode=1``, ``kv_cache_quant_mode=3``

    Do **not** pass split ``k_nope(512)`` / ``k_pe(64)`` buffers to prolog write; CANN rejects
    them for per-tile quant. Use ``split_merged_kv_dtile()`` only after write (read/debug).
    """
    del sfa_impl
    kr_cache = make_empty_kr_cache(merged_kv, qk_rope_head_dim)
    return merged_kv, kr_cache, 1


def make_empty_kr_cache(merged_kv: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Placeholder kr_cache when ckvkr_repo_mode=1 (k_rope merged into kv_cache Dtile)."""
    return torch.zeros(0, 0, merged_kv.shape[-2], rope_dim, dtype=torch.bfloat16, device=merged_kv.device)


def quantize_hidden_for_w8a8_mla_prolog(
    hidden_bf16: torch.Tensor,
    sfa_impl: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor int8 activation quant aligned with ``mla_preprocess`` (``quant_scale0`` / ``quant_offset0``)."""
    flat = hidden_bf16.reshape(-1, hidden_bf16.shape[-1])
    scale = sfa_impl.quant_scale0.to(dtype=hidden_bf16.dtype, device=hidden_bf16.device)
    offset = sfa_impl.quant_offset0.to(dtype=hidden_bf16.dtype, device=hidden_bf16.device)
    scale_reciprocal = (1.0 / scale.to(torch.float32).clamp(min=1e-12)).to(
        dtype=hidden_bf16.dtype, device=hidden_bf16.device
    )
    token_x = torch_npu.npu_quantize(flat, scale_reciprocal, offset, torch.qint8, -1, False)
    # Per-tensor W8A8 + int32 bias: vector dequant uses weight scale only (``PpMatmulW8a8Aiv``).
    # Dynamic-quant paths pass real ``npu_dynamic_quant`` descales instead.
    dequant_scale_x = torch.ones(flat.shape[0], 1, dtype=torch.float32, device=hidden_bf16.device)
    return token_x.view(hidden_bf16.shape), dequant_scale_x


def dequant_int8_tensor(tensor: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    scale_f = scale.reshape(-1, 1).to(torch.float32)
    while scale_f.dim() < tensor.dim():
        scale_f = scale_f.unsqueeze(-1)
    return (tensor.to(torch.float32) * scale_f).to(out_dtype)


def mla_prolog_v3_cq_static_quant_kwargs(sfa_impl: Any) -> dict[str, torch.Tensor]:
    """Optional ``quant_scale_ckv`` / ``quant_scale_ckr`` for RmsNormCq static per-tensor quant.

    In per-tile KV mode (``kv_cache_quant_mode=3``) these optional inputs are not used for
    K-cache quant; pass shape ``[1]`` float tensors instead:
    ``quant_scale_ckv`` ← ``quant_scale1``, ``quant_scale_ckr`` ← ``quant_offset1``.
    Omit both to keep RmsNormCq dynamic quant (``smooth_scales_cq`` path).
    """
    quant_scale1 = getattr(sfa_impl, "quant_scale1", None)
    quant_offset1 = getattr(sfa_impl, "quant_offset1", None)
    if quant_scale1 is None or quant_offset1 is None:
        return {}
    device = quant_scale1.device
    return {
        "quant_scale_ckv": quant_scale1.reshape(1).to(dtype=torch.float32, device=device),
        "quant_scale_ckr": quant_offset1.reshape(1).to(dtype=torch.float32, device=device),
    }


def scatter_paged_kv_update(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    update: torch.Tensor,
) -> None:
    """Scatter into paged KV; skip slot_mapping entries < 0 (ACL graph padding).

    ``update[i]`` is written to physical cache row ``slot_mapping[i]`` (not row ``i``).
    Flatten ``cache`` to ``[num_slots, head_dim]`` so slot ids match paged KV layout.
    """
    num_tokens = update.shape[0]
    slots = slot_mapping[:num_tokens].to(torch.int64).reshape(-1)
    flat_cache = cache.reshape(-1, cache.shape[-1])
    indices = slots.view(-1, 1)
    update_flat = update.reshape(num_tokens, -1)
    # Always launch scatter on device; negative slot ids are skipped by the op.
    # Avoid valid.any()/valid.all() here — they sync to CPU and break ACL graph capture.
    torch.ops._C_ascend.npu_scatter_nd_update_v2(flat_cache, indices, update_flat)


def resolve_prolog_v3_scatter_slots(
    sfa_impl: Any,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Resolve physical paged-cache slot ids for KV scatter (padding = -1)."""
    del sfa_impl
    return slot_mapping[:num_tokens].reshape(-1)


def gather_merged_dtile_rows(
    kv_merged: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Read back per-token merged Dtile rows from paged cache for DSA-CP KV sync."""
    dtile_dim = kv_merged.shape[-1]
    flat = kv_merged.reshape(-1, dtile_dim)
    slots = slot_mapping[:num_rows].to(torch.int64).reshape(-1)
    valid = slots.ge(0)
    # Graph-safe: avoid valid.any() CPU sync during ACL graph capture.
    slots_safe = slots.clamp(min=0)
    gathered = flat.index_select(0, slots_safe)
    valid_mask = valid.unsqueeze(-1)
    return torch.where(valid_mask, gathered, torch.zeros_like(gathered))


def sync_merged_dtile_kv_across_dsa_cp(
    sfa_impl: Any,
    kv_cache: tuple,
    slot_mapping_cp: torch.Tensor,
    slot_mapping_global: torch.Tensor,
    local_num_pad: int,
    num_actual_tokens: int,
    tp_group: Any,
    *,
    async_op: bool = False,
    dtile_local_rows: torch.Tensor | None = None,
    ag_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Any | None]:
    """All-gather local merged Dtile rows; caller writes global slots after wait.

    When ``dtile_local_rows`` is provided (rows op output), skip read-back from paged cache.
    Pass ``ag_output`` (class-level pool) to avoid per-forward ``torch.empty`` allocation.
    """
    from vllm_ascend.distributed.utils import all_gather_async

    if dtile_local_rows is not None:
        dtile_local = dtile_local_rows[:local_num_pad]
    else:
        dtile_local = gather_merged_dtile_rows(kv_cache[0], slot_mapping_cp, local_num_pad)
    ag_num_tokens = local_num_pad * tp_group.world_size
    if ag_output is not None:
        ag_output = ag_output[:ag_num_tokens]
    dtile_global, handle = all_gather_async(
        dtile_local, tp_group, output=ag_output, async_op=async_op
    )
    return dtile_global, handle


def apply_gathered_merged_dtile_kv(
    sfa_impl: Any,
    kv_cache: tuple,
    slot_mapping_global: torch.Tensor,
    dtile_global: torch.Tensor,
    num_actual_tokens: int,
) -> None:
    """Scatter all-gathered merged Dtile rows into global paged KV slots."""
    write_packed_dtile_to_paged_cache(
        kv_cache[0],
        slot_mapping_global[:num_actual_tokens],
        dtile_global[:num_actual_tokens],
        sfa_impl,
    )


def write_packed_dtile_to_paged_cache(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    dtile_rows: torch.Tensor,
    sfa_impl: Any | None = None,
) -> None:
    """Write packed merged Dtile rows into paged cache at ``slot_mapping`` positions.

    Token ``i`` row (int8 k_nope | bf16 k_pe | scale meta) lands at physical slot
    ``slot_mapping[i]``, not at contiguous row ``i``.
    """
    num_tokens = dtile_rows.shape[0]
    if sfa_impl is not None:
        slots = resolve_prolog_v3_scatter_slots(sfa_impl, slot_mapping, num_tokens)
    else:
        slots = slot_mapping[:num_tokens].reshape(-1)
    scatter_paged_kv_update(cache, slots, dtile_rows)


def get_sparse_c8_dsa_cache_indices(kv_cache: tuple) -> tuple[int, int]:
    """Return (dsa_k_idx, dsa_k_scale_idx) for sparse C8 MLA KV layout."""
    if len(kv_cache) == 3:
        return 1, 2
    if len(kv_cache) == 4:
        return 2, 3
    raise ValueError(f"Unexpected sparse C8 kv_cache tuple length: {len(kv_cache)}")


def reshape_mla_prolog_v3_ql_nope(
    decode_q_nope: torch.Tensor,
    batch_size: int,
    num_heads: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    """``mla_prolog_v3`` query layout ``[B,S,N,H]`` or ``[B,N,H]`` -> ``[B,N,H]``."""
    if decode_q_nope.dim() == 4:
        if decode_q_nope.shape[1] != 1:
            raise ValueError(
                f"mla_prolog_v3 query seq dim must be 1, got shape {tuple(decode_q_nope.shape)}"
            )
        decode_q_nope = decode_q_nope[:, 0, ...]
    return decode_q_nope.reshape(batch_size, num_heads, kv_lora_rank)


def build_mla_prolog_v3_kwargs(
    sfa_impl: Any,
    hidden_states: torch.Tensor,
    kv_merged: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_input_tokens: int,
    *,
    kr_cache: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build ``mla_prolog_v3`` kwargs for A2/A3 W8A8 production.

    Activation: per-tensor ``quant_scale0`` / ``quant_offset0`` (``mla_preprocess`` aligned).
    RmsNormCq: static per-tensor ``quant_scale1`` / ``quant_offset1`` when present on
    ``sfa_impl`` (via ``mla_prolog_v3_cq_static_quant_kwargs``).
    """
    bsz = num_input_tokens
    slot_mapping = slot_mapping[:bsz]
    hidden_states_temp = hidden_states[:bsz].unsqueeze(1)
    cos = cos[:bsz, ...]
    sin = sin[:bsz, ...]

    cos_shape = cos.shape
    cos = cos.view(cos_shape[0], 1, cos_shape[-1])
    sin = sin.view(cos_shape[0], 1, cos_shape[-1])

    qk_rope_head_dim = cos_shape[-1]
    use_c8 = getattr(sfa_impl, "enable_sparse_c8", False)
    if kr_cache is None:
        kv_cache, kr_cache, ckvkr_repo_mode = resolve_mla_prolog_v3_kv_kr_caches(
            kv_merged, qk_rope_head_dim, sfa_impl
        )
    else:
        kv_cache = kv_merged
        ckvkr_repo_mode = 0
    token_x, dequant_scale_x = quantize_hidden_for_w8a8_mla_prolog(hidden_states_temp, sfa_impl)

    # Align with A5 prolog: pass slot_mapping as cache_index unchanged so padding
    # slots (-1) are skipped by npu_mla_prolog_v3 instead of clamping to slot 0.
    cache_index = slot_mapping[:bsz].view(bsz, -1).to(torch.int64)

    kwargs: dict[str, Any] = {
        "token_x": token_x,
        "dequant_scale_x": dequant_scale_x,
        "weight_dq": sfa_impl.weight_dq,
        "weight_uq_qr": sfa_impl.weight_uq_qr,
        "weight_uk": sfa_impl.W_UK_T,
        "weight_dkv_kr": sfa_impl.weight_dkv_kr,
        "rmsnorm_gamma_cq": sfa_impl.q_a_layernorm.weight.data,
        "rmsnorm_gamma_ckv": sfa_impl.kv_a_layernorm.weight.data,
        "rope_sin": sin,
        "rope_cos": cos,
        "kv_cache": kv_cache,
        "kr_cache": kr_cache,
        "cache_index": cache_index,
        "dequant_scale_w_dq": sfa_impl.dequant_scale_w_dq,
        "dequant_scale_w_uq_qr": sfa_impl.dequant_scale_w_uq_qr,
        "dequant_scale_w_dkv_kr": sfa_impl.dequant_scale_w_dkv_kr,
        "k_nope_clip_alpha": sfa_impl.sfa_qsfa_k_nope_clip_alpha,
        "cache_mode": "PA_BSND",
        "weight_quant_mode": 2,
        "kv_cache_quant_mode": 3 if use_c8 else 0,
        "query_quant_mode": 0,
        "ckvkr_repo_mode": ckvkr_repo_mode if use_c8 else 0,
        "quant_scale_repo_mode": 1 if use_c8 else 0,
        "query_norm_flag": True,
        "rope_mode": MLA_PROLOG_V3_ROPE_MODE_HALF,
    }
    kwargs.update(mla_prolog_v3_cq_static_quant_kwargs(sfa_impl))
    bias_w_dq = getattr(sfa_impl, "bias_w_dq", None)
    bias_w_dkv_kr = getattr(sfa_impl, "bias_w_dkv_kr", None)
    bias_w_uq_qr = getattr(sfa_impl, "bias_w_uq_qr", None)
    if bias_w_dq is not None:
        kwargs["bias_w_dq"] = bias_w_dq
    if bias_w_dkv_kr is not None:
        kwargs["bias_w_dkv_kr"] = bias_w_dkv_kr
    if bias_w_uq_qr is not None:
        kwargs["bias_w_uq_qr"] = bias_w_uq_qr
    return kwargs


def invoke_mla_prolog_v3(
    **prolog_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch ``mla_prolog_v3`` to vLLM Ascend custom op or ``torch_npu.npu_mla_prolog_v3``.

    Prefer ``torch.ops._C_ascend.mla_prolog_v3`` (vendored ``csrc/attention/mla_prolog_v3``,
    supports ``rope_mode``). Fallback strips ``rope_mode`` for older ``torch_npu`` schemas.
    """
    custom_ops = getattr(torch.ops, "_C_ascend", None)
    if custom_ops is not None and hasattr(custom_ops, "mla_prolog_v3"):
        return custom_ops.mla_prolog_v3(**prolog_kwargs)
    fallback_kwargs = {k: v for k, v in prolog_kwargs.items() if k != "rope_mode"}
    return torch_npu.npu_mla_prolog_v3(**fallback_kwargs)


# Deprecated alias; use invoke_mla_prolog_v3.
invoke_npu_mla_prolog_v3 = invoke_mla_prolog_v3


def _acquire_mla_query_buffers(
    sfa_impl: Any,
    num_tokens: int,
    nheads: int,
    hckv: int,
    dr: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse DSA-CP class-level Q pools when present; otherwise allocate.

    Non-DSA-CP / non-C8 configs leave the pools as ``None``, preserving the
    previous per-forward ``torch.empty`` behavior.
    """
    ql_pool = getattr(sfa_impl, "dsa_cp_ql_nope_pool", None)
    q_pe_pool = getattr(sfa_impl, "dsa_cp_q_pe_pool", None)
    q_c_pool = getattr(sfa_impl, "dsa_cp_q_c_pool", None)
    if (
        ql_pool is not None
        and q_pe_pool is not None
        and q_c_pool is not None
        and ql_pool.shape[0] >= num_tokens
        and ql_pool.shape[1] >= nheads
        and q_pe_pool.shape[0] >= num_tokens
        and q_pe_pool.shape[1] >= nheads
        and q_c_pool.shape[0] >= num_tokens
        and ql_pool.dtype == dtype
        and ql_pool.device == device
    ):
        return (
            ql_pool[:num_tokens, :nheads],
            q_pe_pool[:num_tokens, :nheads],
            q_c_pool[:num_tokens],
        )
    ql_nope = torch.empty((num_tokens, nheads, hckv), dtype=dtype, device=device)
    q_pe = torch.empty((num_tokens, nheads, dr), dtype=dtype, device=device)
    q_c = torch.empty((num_tokens, sfa_impl.q_lora_rank), dtype=dtype, device=device)
    return ql_nope, q_pe, q_c


def run_mla_prolog_v3_preprocess(
    sfa_impl: Any,
    hidden_states: torch.Tensor,
    kv_cache: tuple,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_input_tokens: int,
    num_valid_tokens: int | None = None,
    *,
    dtile_rows_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A2/A3 C8 preprocess via fused ``mla_preprocess`` merged Dtile KV write.

    Single custom op (``cache_mode=merged_dtile``): Q/K prolog + per-tile int8 quant
    + 656B Dtile scatter. Replaces decomposed quant/pack/scatter Python steps.
    uses only valid tokens; query outputs are zero-padded to ``num_input_tokens``.
    """
    if num_valid_tokens is None:
        num_valid_tokens = num_input_tokens
    num_cache_tokens = max(0, min(num_valid_tokens, num_input_tokens))

    hckv = sfa_impl.kv_lora_rank
    dr = sfa_impl.qk_rope_head_dim
    nheads = get_mlapo_query_num_heads(sfa_impl)
    dtype = hidden_states.dtype
    device = hidden_states.device

    ql_nope, q_pe, q_c = _acquire_mla_query_buffers(
        sfa_impl, num_input_tokens, nheads, hckv, dr, dtype, device
    )

    # DSA-CP ranks with no local tokens (local_num_valid=0) still carry padded
    # slot_mapping entries (-1). Skip the kernel — N=0 with bIndex=-1 triggers
    # "merged_dtile bIndex is out of range" in CANN host checks.
    if num_cache_tokens <= 0:
        ql_nope.zero_()
        q_pe.zero_()
        q_c.zero_()
        return hidden_states, ql_nope, q_pe, q_c

    mla_preprocess_write_per_tile_merged_kv(
        sfa_impl,
        hidden_states[:num_cache_tokens],
        cos[:num_cache_tokens],
        sin[:num_cache_tokens],
        kv_cache[0],
        slot_mapping[:num_cache_tokens],
        num_cache_tokens,
        ql_nope[:num_cache_tokens],
        q_pe[:num_cache_tokens],
        q_c[:num_cache_tokens],
        dtile_rows_out=dtile_rows_out[:num_cache_tokens] if dtile_rows_out is not None else None,
    )

    if num_cache_tokens < num_input_tokens:
        ql_nope[num_cache_tokens:].zero_()
        q_pe[num_cache_tokens:].zero_()
        q_c[num_cache_tokens:].zero_()

    return hidden_states, ql_nope, q_pe, q_c


def run_lightning_indexer_topk(
    *,
    enable_sparse_c8: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    query_dequant_scale: torch.Tensor | None = None,
    key_dequant_scale: torch.Tensor | None = None,
    query_shape_ori: tuple[int, ...] | None = None,
    use_torch_npu_lightning_indexer: bool = False,
) -> torch.Tensor:
    """Mirror BaseDeviceAdaptor.indexer_select_post_process top-k selection."""
    if enable_sparse_c8:
        assert query_dequant_scale is not None
        assert key_dequant_scale is not None
        assert query_shape_ori is not None
        weights = weights.to(torch.float16)
        # Defensive: callers should use DeviceOperator.prepare_dsa_indexer_* (A2 fp16;
        # A5 uses npu_quant_lightning_indexer with fp32 scales, not this path).
        query_dequant_scale = query_dequant_scale.to(torch.float16)
        key_dequant_scale = key_dequant_scale.to(torch.float16)
        # key cache scale is PA_BSND 4D; op expects rank-3 (key rank - 1).
        if key_dequant_scale.dim() == 4:
            key_dequant_scale = key_dequant_scale.squeeze(2)
        topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
            query=query.view(query_shape_ori),
            key=key,
            weights=weights,
            query_dequant_scale=query_dequant_scale.view(query_shape_ori[:-1]),
            key_dequant_scale=key_dequant_scale,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=INDEXER_SPARSE_COUNT,
            sparse_mode=3,
        )
        return topk_indices

    if use_torch_npu_lightning_indexer:
        topk_indices, _ = torch_npu.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=INDEXER_SPARSE_COUNT,
            sparse_mode=3,
        )
    else:
        topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=INDEXER_SPARSE_COUNT,
            sparse_mode=3,
        )
    return topk_indices


def run_kv_quant_sparse_flash_attention(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_merged: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    scale_value: float,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    """Standalone torch_npu kv-quant SFA helper for benches / isolated processes.

    Do **not** call this from the serving path after ``vllm_ascend_C`` is loaded:
    mixing ``torch_npu.npu_kv_quant_sparse_flash_attention`` with ``_C_ascend`` in the
    same process segfaults (see ``bench_kv_quant_sparse_flash_attention.py``).
    Production uses ``DeviceOperator.execute_kv_quant_sparse_flash_attention`` →
    ``torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention``.

    ``key`` and ``value`` both use the packed Dtile buffer (same as 0.22 serving).
    """
    query = torch.cat([ql_nope, q_pe], dim=-1)
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query,
        key=kv_merged,
        value=kv_merged,
        sparse_indices=topk_indices,
        scale_value=scale_value,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=MLA_PROLOG_V3_TILE_SIZE,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=qk_rope_head_dim,
    )


def run_sparse_flash_attention_bf16(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    """Mirror BaseDeviceAdaptor.execute_sparse_flash_attention_process bf16 branch."""
    attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=ql_nope,
        key=k_nope,
        value=k_nope,
        sparse_indices=topk_indices,
        scale_value=scale_value,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_key,
        query_rope=q_pe,
        key_rope=k_rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
    )
    return attn_output
