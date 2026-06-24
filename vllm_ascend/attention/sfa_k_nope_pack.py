# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack int8 k_nope + fp32 per-tile scales into kv_cache[0] for 910B sparse C8 SFA."""

from __future__ import annotations

import torch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

K_NOPE_INT8_DIM = 512
K_NOPE_TILE_SIZE = 128
K_NOPE_NUM_TILES = K_NOPE_INT8_DIM // K_NOPE_TILE_SIZE
K_NOPE_SCALE_METADATA_BYTES = K_NOPE_NUM_TILES * 4
K_NOPE_PACKED_BYTES = K_NOPE_INT8_DIM + K_NOPE_SCALE_METADATA_BYTES
INT8_MAX = 127.0


def use_910b_packed_k_nope_sparse_c8() -> bool:
    """True when 910B sparse C8 uses packed k_nope layout in kv_cache[0]."""
    return get_ascend_device_type() != AscendDeviceType.A5


def is_packed_k_nope_sparse_head_dim(
    sparse_head_dim: tuple[int, ...] | None,
    kv_lora_rank: int,
) -> bool:
    if sparse_head_dim is None or len(sparse_head_dim) != 3:
        return False
    packed_k_nope_dim, qk_rope_head_dim, _ = sparse_head_dim
    return (
        qk_rope_head_dim != 0
        and packed_k_nope_dim == kv_lora_rank + K_NOPE_SCALE_METADATA_BYTES
    )


def quantize_k_nope_per_group(k_nope: torch.Tensor) -> torch.Tensor:
    """Quantize k_nope to a packed uint8 row: 512 int8 + 4 fp32 scales.

    Args:
        k_nope: [..., kv_lora_rank] bf16/fp16/fp32

    Returns:
        Packed row [..., K_NOPE_PACKED_BYTES] with dtype uint8.
    """
    orig_shape = k_nope.shape[:-1]
    k_nope = k_nope.reshape(-1, K_NOPE_INT8_DIM)
    num_tokens = k_nope.shape[0]
    device = k_nope.device

    packed = torch.empty(
        num_tokens,
        K_NOPE_PACKED_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    k_int8 = packed[:, :K_NOPE_INT8_DIM].view(torch.int8)
    scales = packed[:, K_NOPE_INT8_DIM :].view(torch.float32)

    k_fp32 = k_nope.float()
    for tile_idx in range(K_NOPE_NUM_TILES):
        start = tile_idx * K_NOPE_TILE_SIZE
        end = start + K_NOPE_TILE_SIZE
        tile = k_fp32[:, start:end]
        tile_scale = torch.clamp(tile.abs().amax(dim=-1) / INT8_MAX, min=1e-12)
        scales[:, tile_idx] = tile_scale
        k_int8[:, start:end] = torch.round(tile / tile_scale.unsqueeze(-1)).clamp(
            -128, 127
        ).to(torch.int8)

    return packed.view(*orig_shape, K_NOPE_PACKED_BYTES)


def dequantize_packed_k_nope(packed: torch.Tensor) -> torch.Tensor:
    """Dequantize packed k_nope rows back to float32 [..., kv_lora_rank]."""
    orig_shape = packed.shape[:-1]
    packed = packed.reshape(-1, K_NOPE_PACKED_BYTES)
    k_int8 = packed[:, :K_NOPE_INT8_DIM].to(torch.float32)
    scales = packed[:, K_NOPE_INT8_DIM :].view(torch.float32)
    dequant = torch.empty_like(k_int8)
    for tile_idx in range(K_NOPE_NUM_TILES):
        start = tile_idx * K_NOPE_TILE_SIZE
        end = start + K_NOPE_TILE_SIZE
        dequant[:, start:end] = k_int8[:, start:end] * scales[:, tile_idx : tile_idx + 1]
    return dequant.view(*orig_shape, K_NOPE_INT8_DIM)


def scatter_packed_k_nope_to_cache(
    k_nope: torch.Tensor,
    packed_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Pack-quantize k_nope and scatter into kv_cache[0]."""
    import torch_npu

    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return

    k_nope = k_nope.reshape(num_tokens, -1)
    packed_rows = quantize_k_nope_per_group(k_nope).reshape(num_tokens, K_NOPE_PACKED_BYTES)
    torch_npu.npu_scatter_nd_update_(
        packed_cache.view(-1, K_NOPE_PACKED_BYTES),
        slot_mapping.view(-1, 1),
        packed_rows,
    )


def scatter_k_pe_to_cache(
    k_pe: torch.Tensor,
    k_pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter bf16 k_pe into kv_cache[1]."""
    import torch_npu

    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return

    k_pe = k_pe.reshape(num_tokens, -1)
    torch_npu.npu_scatter_nd_update_(
        k_pe_cache.view(-1, k_pe.shape[-1]),
        slot_mapping.view(-1, 1),
        k_pe,
    )
