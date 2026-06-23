# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.sfa_k_nope_pack import K_NOPE_PACKED_BYTES
from vllm_ascend.patch.platform.patch_kv_cache_interface import AscendMLAAttentionSpec


def test_910b_sparse_c8_page_size_and_ratio():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=K_NOPE_PACKED_BYTES + 64 + 128,
        sparse_head_dim=(K_NOPE_PACKED_BYTES, 64, 128),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        cache_sparse_c8=True,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
    )

    page_size = spec.page_size_bytes
    # 528 int8 + 128 bf16 rope + 128 int8 qli + 2 fp16 scale per token
    expected = 128 * (528 + 64 * 2 + 128 + 2)
    assert page_size == expected

    ratio = spec.sparse_kv_cache_ratio
    assert len(ratio) == 4
    assert ratio[3] is not None
    assert ratio[0] > ratio[1]
