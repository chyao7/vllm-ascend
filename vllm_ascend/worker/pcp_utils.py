#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Prefill Context Parallel helpers (PCP-only, dcp=1).

Token ownership uses the same interleaved layout as DCP:
    owner(x) = (x // interleave_size) % pcp_size

Prefill tokens stay on the owner rank. Decode tokens (query_len <=
decode_threshold) are replicated on every PCP rank so Q is available for
the local-KV + LSE-merge decode path.
"""

from __future__ import annotations

import numpy as np


def pcp_token_owner(
    positions: np.ndarray,
    pcp_size: int,
    interleave_size: int = 1,
) -> np.ndarray:
    """Return the PCP rank that stores KV for each global position."""
    if pcp_size <= 1:
        return np.zeros_like(positions, dtype=np.int32)
    return ((positions // interleave_size) % pcp_size).astype(np.int32)


def local_kv_seq_lens(
    seq_lens: np.ndarray,
    pcp_size: int,
    pcp_rank: int,
    interleave_size: int = 1,
) -> np.ndarray:
    """Interleave-aware number of KV tokens stored on ``pcp_rank``."""
    if pcp_size <= 1:
        return seq_lens.astype(np.int32)
    tiled = seq_lens.astype(np.int64)
    base = tiled // interleave_size // pcp_size * interleave_size
    remainder = tiled - base * pcp_size
    rank_offset = pcp_rank * interleave_size
    extra = np.clip(remainder - rank_offset, 0, interleave_size)
    return (base + extra).astype(np.int32)


def split_tokens_for_pcp(
    positions: np.ndarray,
    query_lens: np.ndarray,
    pcp_rank: int,
    pcp_size: int,
    interleave_size: int = 1,
    decode_threshold: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter a packed batch down to the tokens this PCP rank should compute.

    Returns:
        keep_idx: indices into the original packed token list
        new_query_lens: per-request local token counts (never zero)
        new_positions: global positions of kept tokens
        new_req_indices: request id for each kept token
        new_query_pos: 0..local_len-1 offsets within each request
    """
    num_reqs = int(query_lens.shape[0])
    if pcp_size <= 1 or num_reqs == 0:
        req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
        query_pos = np.concatenate(
            [np.arange(int(q), dtype=np.int32) for q in query_lens]
        ) if num_reqs else np.empty(0, dtype=np.int32)
        keep_idx = np.arange(positions.shape[0], dtype=np.int32)
        return keep_idx, query_lens.astype(np.int32), positions, req_indices, query_pos

    req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    owners = pcp_token_owner(positions, pcp_size, interleave_size)
    is_decode_req = query_lens <= decode_threshold
    is_decode_token = is_decode_req[req_indices]
    keep = is_decode_token | (owners == pcp_rank)

    keep_idx_list: list[int] = []
    new_query_lens = np.zeros(num_reqs, dtype=np.int32)
    new_positions_list: list[int] = []
    new_req_list: list[int] = []
    new_query_pos_list: list[int] = []

    token_offset = 0
    for req_i, qlen in enumerate(query_lens.tolist()):
        qlen = int(qlen)
        req_keep = np.flatnonzero(keep[token_offset : token_offset + qlen])
        if req_keep.size == 0:
            # Keep a dummy token so every rank still has this request.
            dummy = token_offset if qlen > 0 else 0
            keep_idx_list.append(int(dummy))
            new_positions_list.append(int(positions[dummy]) if qlen > 0 else 0)
            new_req_list.append(req_i)
            new_query_pos_list.append(0)
            new_query_lens[req_i] = 1
        else:
            local_pos = 0
            for local_i in req_keep.tolist():
                abs_i = token_offset + int(local_i)
                keep_idx_list.append(abs_i)
                new_positions_list.append(int(positions[abs_i]))
                new_req_list.append(req_i)
                new_query_pos_list.append(local_pos)
                local_pos += 1
            new_query_lens[req_i] = local_pos
        token_offset += qlen

    return (
        np.asarray(keep_idx_list, dtype=np.int32),
        new_query_lens,
        np.asarray(new_positions_list, dtype=positions.dtype),
        np.asarray(new_req_list, dtype=np.int32),
        np.asarray(new_query_pos_list, dtype=np.int32),
    )


def request_ids_from_query_lens(query_lens: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(query_lens.shape[0], dtype=np.int32), query_lens)
