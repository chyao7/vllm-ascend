# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.sfa_k_nope_pack import (
    K_NOPE_INT8_DIM,
    K_NOPE_PACKED_BYTES,
    K_NOPE_SCALE_METADATA_BYTES,
    dequantize_packed_k_nope,
    is_packed_k_nope_sparse_head_dim,
    quantize_k_nope_per_group,
)


def test_quantize_dequantize_roundtrip():
    k_nope = torch.randn(4, K_NOPE_INT8_DIM, dtype=torch.bfloat16)
    packed = quantize_k_nope_per_group(k_nope)
    assert packed.shape == (4, K_NOPE_PACKED_BYTES)
    assert packed.dtype == torch.uint8

    restored = dequantize_packed_k_nope(packed)
    assert restored.shape == (4, K_NOPE_INT8_DIM)
    torch.testing.assert_close(restored, k_nope.float(), rtol=0.05, atol=0.05)


def test_is_packed_k_nope_sparse_head_dim():
    assert is_packed_k_nope_sparse_head_dim((528, 64, 128), kv_lora_rank=512)
    assert not is_packed_k_nope_sparse_head_dim((512, 64, 128), kv_lora_rank=512)
    assert not is_packed_k_nope_sparse_head_dim((600, 0, 128), kv_lora_rank=512)


def test_packed_cache_row_layout():
    block_size = 4
    cache = torch.zeros(1, block_size, 1, K_NOPE_PACKED_BYTES, dtype=torch.int8)
    k_nope = torch.randn(block_size, K_NOPE_INT8_DIM, dtype=torch.bfloat16)
    packed = quantize_k_nope_per_group(k_nope).view(torch.int8)
    cache[0, :, 0, :] = packed
    restored = dequantize_packed_k_nope(cache[0, :, 0, :].view(torch.uint8))
    torch.testing.assert_close(restored, k_nope.float(), rtol=0.05, atol=0.05)
