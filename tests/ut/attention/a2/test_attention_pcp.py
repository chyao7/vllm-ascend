# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.attention_pcp import (
    AscendAttentionPCPImpl,
    AscendAttentionPCPMetadata,
    AscendAttentionPCPMetadataBuilder,
)
from vllm_ascend.attention.context_parallel.common_cp import _update_out_and_lse
from vllm_ascend.worker.pcp_utils import (
    local_kv_seq_lens,
    pcp_token_owner,
    split_tokens_for_pcp,
)


def test_gqa_pcp_extends_v1_backend_without_polluting_base_metadata() -> None:
    assert issubclass(AscendAttentionPCPImpl, AscendAttentionBackendImpl)
    assert issubclass(AscendAttentionPCPMetadataBuilder, AscendAttentionMetadataBuilder)
    assert AscendAttentionPCPMetadataBuilder.metadata_cls is AscendAttentionPCPMetadata
    assert not hasattr(AscendMetadata(), "query_positions")
    assert AscendAttentionPCPImpl.supports_pcp is True


def test_pcp_token_owner_interleave1_size2() -> None:
    positions = np.arange(8, dtype=np.int32)
    owners = pcp_token_owner(positions, pcp_size=2, interleave_size=1)
    np.testing.assert_array_equal(owners, [0, 1, 0, 1, 0, 1, 0, 1])


def test_pcp_local_kv_seq_lens_matches_interleave_formula() -> None:
    seq_lens = np.array([8, 7, 1], dtype=np.int32)
    rank0 = local_kv_seq_lens(seq_lens, pcp_size=2, pcp_rank=0, interleave_size=1)
    rank1 = local_kv_seq_lens(seq_lens, pcp_size=2, pcp_rank=1, interleave_size=1)
    np.testing.assert_array_equal(rank0, [4, 4, 1])
    np.testing.assert_array_equal(rank1, [4, 3, 0])
    np.testing.assert_array_equal(rank0 + rank1, seq_lens)


def test_split_prefill_tokens_keeps_owner_shard() -> None:
    positions = np.arange(8, dtype=np.int32)
    query_lens = np.array([8], dtype=np.int32)
    _, lens0, pos0, _, _ = split_tokens_for_pcp(positions, query_lens, pcp_rank=0, pcp_size=2)
    _, lens1, pos1, _, _ = split_tokens_for_pcp(positions, query_lens, pcp_rank=1, pcp_size=2)
    np.testing.assert_array_equal(pos0, [0, 2, 4, 6])
    np.testing.assert_array_equal(pos1, [1, 3, 5, 7])
    np.testing.assert_array_equal(lens0, [4])
    np.testing.assert_array_equal(lens1, [4])


def test_split_decode_tokens_are_replicated() -> None:
    positions = np.array([8], dtype=np.int32)
    query_lens = np.array([1], dtype=np.int32)
    _, lens0, pos0, _, _ = split_tokens_for_pcp(positions, query_lens, pcp_rank=0, pcp_size=2)
    _, lens1, pos1, _, _ = split_tokens_for_pcp(positions, query_lens, pcp_rank=1, pcp_size=2)
    np.testing.assert_array_equal(pos0, [8])
    np.testing.assert_array_equal(pos1, [8])
    np.testing.assert_array_equal(lens0, [1])
    np.testing.assert_array_equal(lens1, [1])


def test_pcp_partial_attention_merge_matches_weighted_reference() -> None:
    outputs = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
        ]
    ).reshape(2, 1, 1, 2)
    lse = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32).reshape(2, 1, 1, 1)

    output, merged_lse = _update_out_and_lse(outputs, lse)

    torch.testing.assert_close(output, torch.tensor([[[4.0, 6.0]]]))
    torch.testing.assert_close(merged_lse, torch.tensor([[[np.log(4.0)]]], dtype=torch.float32))
