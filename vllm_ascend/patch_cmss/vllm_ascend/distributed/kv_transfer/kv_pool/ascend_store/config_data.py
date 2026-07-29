from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from collections.abc import Callable, Mapping

_target = import_module('vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data')

globals().update({name: value for name, value in vars(_target).items() if not name.startswith("__")})

_original_pool_key_init = _target.PoolKey.__init__

_original_token_database_init = _target.ChunkedTokenDatabase.__init__

@dataclass(frozen=True)
class TPMismatchInfo:
    enabled: bool
    peer_tp_size: int
    effective_tp_size: int
    local_heads_per_rank: int
    effective_heads_per_rank: int
    num_sub_keys: int

def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default

def infer_tp_mismatch_info(
    kv_role: str,
    extra_config: Mapping[str, Any] | object,
    local_tp_size: int | object,
    num_kv_heads: int | object,
    use_mla: bool,
    use_hybrid: bool = False,
) -> TPMismatchInfo:
    local_tp_size = _as_positive_int(local_tp_size, 1)
    num_kv_heads = _as_positive_int(num_kv_heads, 1)
    peer_tp_size = local_tp_size
    if isinstance(extra_config, Mapping):
        peer_key = "prefill_tp_size" if kv_role == "kv_consumer" else "decode_tp_size"
        peer_tp_size = _as_positive_int(extra_config.get(peer_key, local_tp_size), local_tp_size)

    effective_tp_size = max(local_tp_size, peer_tp_size)
    enabled = (
        peer_tp_size != local_tp_size
        and not use_mla
        and not use_hybrid
        and num_kv_heads >= effective_tp_size
        and num_kv_heads % effective_tp_size == 0
    )
    local_heads_per_rank = num_kv_heads // local_tp_size if local_tp_size <= num_kv_heads else 1
    effective_heads_per_rank = num_kv_heads // effective_tp_size if enabled else local_heads_per_rank
    num_sub_keys = local_heads_per_rank // effective_heads_per_rank if enabled else 1
    return TPMismatchInfo(
        enabled=enabled,
        peer_tp_size=peer_tp_size,
        effective_tp_size=effective_tp_size,
        local_heads_per_rank=local_heads_per_rank,
        effective_heads_per_rank=effective_heads_per_rank,
        num_sub_keys=num_sub_keys,
    )

class PoolKey(_target.PoolKey):
            def __init__(self, key_metadata, chunk_hash, *, chunk_hash_bytes=None):
                _original_pool_key_init(self, key_metadata, chunk_hash)
                self.chunk_hash_bytes = chunk_hash_bytes

            def split_layers(self, num_layers: int) -> list[LayerPoolKey]:
                """Split the key into multiple keys for each layer"""
                keys = []
                for layer_id in range(num_layers):
                    key = LayerPoolKey(self.key_metadata, self.chunk_hash, layer_id)
                    key.chunk_hash_bytes = self.chunk_hash_bytes
                    keys.append(key)
                return keys

class ChunkedTokenDatabase(_target.ChunkedTokenDatabase):
            def __init__(
                self,
                metadata: list[KeyMetadata],
                block_size: list[int],
                partitions: list[int] | None,
                use_hybrid: bool = False,
                hash_block_size: int | None = None,
            ):
                _original_token_database_init(
                    self, metadata, block_size, partitions, use_hybrid, hash_block_size
                )
                self._key_prefix_cache: dict[tuple[int, str, str], str] = {}

            def _get_key_prefix(
                self,
                kv_cache_group_id: int,
                cache_role: str = "kv",
                cache_family: str | None = None,
            ) -> str:
                if cache_family is None:
                    cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
                cache_key = (kv_cache_group_id, cache_role, cache_family)
                prefix = self._key_prefix_cache.get(cache_key)
                if prefix is None:
                    group_metadata = self.metadata[kv_cache_group_id]
                    prefix = (
                        f"{group_metadata.model_name}"
                        f"@pcp{group_metadata.pcp_rank}@dcp{group_metadata.dcp_rank}"
                        f"@head_or_tp_rank:{group_metadata.head_or_tp_rank}"
                        f"@pp_rank:{group_metadata.pp_rank}"
                        f"@group:{kv_cache_group_id}"
                        f"@cache_role:{cache_role}"
                        f"@cache_family:{cache_family}@"
                    )
                    self._key_prefix_cache[cache_key] = prefix
                return prefix

            def _make_key_by_hash(
                self,
                chunk_hash: str,
                kv_cache_group_id: int = 0,
                cache_role: str = "kv",
                cache_family: str | None = None,
                chunk_hash_bytes: BlockHash | str | None = None,
                layer_id: int | None = None,
            ):
                assert self.metadata is not None
                if cache_family is None:
                    cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
                group_metadata = self.metadata[kv_cache_group_id]
                return PoolKey(
                    KeyMetadata(
                        model_name=group_metadata.model_name,
                        head_or_tp_rank=group_metadata.head_or_tp_rank,
                        pcp_rank=group_metadata.pcp_rank,
                        dcp_rank=group_metadata.dcp_rank,
                        pp_rank=group_metadata.pp_rank,
                        kv_cache_group_id=kv_cache_group_id,
                        cache_role=cache_role,
                        cache_family=cache_family,
                    ),
                    chunk_hash,
                    chunk_hash_bytes=chunk_hash_bytes,
                )

            def set_group_buffers(
                self,
                group_kv_caches_base_addr: dict[int, list[int]],
                group_block_len: dict[int, list[int]],
                group_block_stride: dict[int, list[int]] | None = None,
                cache_role: str = "kv",
                group_cache_families: dict[int, str] | None = None,
                group_num_layers: dict[int, int] | None = None,
            ) -> None:
                if cache_role == "state":
                    # Keep the interface for future explicit state groups, but this
                    # DSV4 branch stores compressor/indexer states in kv_caches.
                    pass
                else:
                    self.group_kv_caches_base_addr = group_kv_caches_base_addr
                    self.group_block_len = group_block_len
                    self.group_block_stride = group_block_stride or {}
                if group_cache_families is not None:
                    self.group_cache_families[cache_role] = group_cache_families.copy()
                    self._key_prefix_cache.clear()
                if group_num_layers is not None:
                    self.group_num_layers[cache_role] = group_num_layers.copy()

            def process_tokens(
                self,
                token_len: int,
                block_hashes: BlockHashList | list[str],
                mask_num: int = 0,
                kv_cache_group_id: int = 0,
                cache_role: str = "kv",
                cache_family: str | None = None,
                chunk_mask: Sequence[bool] | None = None,
                chunk_filter: Callable[[int], bool] | None = None,
            ) -> Iterable[tuple[int, int, PoolKey]]:
                """Process the tokens and return the corresponding cache engine keys."""
                for start_idx, end_idx, _key_string, hash_val in self.process_token_key_strings(
                    token_len,
                    block_hashes,
                    mask_num=mask_num,
                    kv_cache_group_id=kv_cache_group_id,
                    cache_role=cache_role,
                    cache_family=cache_family,
                    chunk_mask=chunk_mask,
                    chunk_filter=chunk_filter,
                ):
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(
                            block_hash_to_str(hash_val),
                            kv_cache_group_id=kv_cache_group_id,
                            cache_role=cache_role,
                            cache_family=cache_family,
                            chunk_hash_bytes=hash_val,
                        ),
                    )

            def process_token_key_strings(
                self,
                token_len: int,
                block_hashes: BlockHashList | list[str],
                mask_num: int = 0,
                kv_cache_group_id: int = 0,
                cache_role: str = "kv",
                cache_family: str | None = None,
                max_num: int | None = None,
                chunk_mask: Sequence[bool] | None = None,
                chunk_filter: Callable[[int], bool] | None = None,
            ) -> Iterable[tuple[int, int, str, BlockHash | str]]:
                if not block_hashes:
                    return
                base_block_size = self.get_block_size(kv_cache_group_id)
                if cache_family is None:
                    cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
                cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
                effective_block_size = base_block_size * cache_family_ratio
                grouped_hashes = get_block_hashes(
                    block_hashes,
                    effective_block_size,
                    self.hash_block_size,
                )
                if not grouped_hashes:
                    return
                prefix = self._get_key_prefix(kv_cache_group_id, cache_role, cache_family)
                lookup_end = token_len if max_num is None else min(token_len, max_num)
                max_chunks = cdiv(lookup_end, effective_block_size) if lookup_end > 0 else 0
                for chunk_id in range(min(len(grouped_hashes), max_chunks)):
                    if chunk_mask is not None and (chunk_id >= len(chunk_mask) or not chunk_mask[chunk_id]):
                        continue
                    start_token = chunk_id * effective_block_size
                    end_token = min(start_token + effective_block_size, lookup_end)
                    if start_token < mask_num:
                        continue
                    start_idx = start_token // cache_family_ratio
                    end_idx = end_token // cache_family_ratio
                    if end_idx <= start_idx:
                        continue
                    if chunk_filter is not None and not chunk_filter(start_idx):
                        continue
                    hash_val = grouped_hashes[chunk_id]
                    yield start_idx, end_idx, prefix + block_hash_to_str(hash_val), hash_val

            def process_tokens_with_block_ids(
                self,
                token_len: int,
                block_hashes: BlockHashList | list[str],
                block_ids: list[int],
                mask_num: int = 0,
                kv_cache_group_id: int = 0,
                skip_null_blocks: bool = False,
                cache_role: str = "kv",
                cache_family: str | None = None,
                chunk_mask: Sequence[bool] | None = None,
                chunk_filter: Callable[[int], bool] | None = None,
            ) -> Iterable[tuple[int, int, PoolKey, int]]:
                for start_idx, end_idx, _key_string, hash_val, block_id in self.process_token_key_strings_with_block_ids(
                    token_len,
                    block_hashes,
                    block_ids,
                    mask_num=mask_num,
                    kv_cache_group_id=kv_cache_group_id,
                    skip_null_blocks=skip_null_blocks,
                    cache_role=cache_role,
                    cache_family=cache_family,
                    chunk_mask=chunk_mask,
                    chunk_filter=chunk_filter,
                ):
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(
                            block_hash_to_str(hash_val),
                            kv_cache_group_id=kv_cache_group_id,
                            cache_role=cache_role,
                            cache_family=cache_family,
                            chunk_hash_bytes=hash_val,
                        ),
                        block_id,
                    )

            def process_token_key_strings_with_block_ids(
                self,
                token_len: int,
                block_hashes: BlockHashList | list[str],
                block_ids: list[int],
                mask_num: int = 0,
                kv_cache_group_id: int = 0,
                skip_null_blocks: bool = False,
                cache_role: str = "kv",
                cache_family: str | None = None,
                chunk_mask: Sequence[bool] | None = None,
                chunk_filter: Callable[[int], bool] | None = None,
            ) -> Iterable[tuple[int, int, str, BlockHash | str, int]]:
                if not block_hashes:
                    return
                base_block_size = self.get_block_size(kv_cache_group_id)
                if cache_family is None:
                    cache_family = self.group_cache_families.get(cache_role, {}).get(kv_cache_group_id, "default")
                cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
                effective_block_size = base_block_size * cache_family_ratio
                grouped_hashes = get_block_hashes(
                    block_hashes,
                    effective_block_size,
                    self.hash_block_size,
                )
                if not grouped_hashes:
                    return

                num_by_hashes = len(grouped_hashes)
                num_by_token_len = cdiv(token_len, effective_block_size) if token_len > 0 else 0
                num_logical_blocks = min(num_by_hashes, num_by_token_len)
                block_id_offset = max(num_logical_blocks - len(block_ids), 0)
                prefix = self._get_key_prefix(kv_cache_group_id, cache_role, cache_family)

                for chunk_id in range(num_logical_blocks):
                    if chunk_mask is not None and (chunk_id >= len(chunk_mask) or not chunk_mask[chunk_id]):
                        continue
                    start_token = chunk_id * effective_block_size
                    if start_token >= token_len:
                        break
                    end_token = min(start_token + effective_block_size, token_len)
                    if start_token < mask_num:
                        continue
                    start_idx = start_token // cache_family_ratio
                    end_idx = end_token // cache_family_ratio
                    if end_idx <= start_idx:
                        continue
                    if chunk_filter is not None and not chunk_filter(start_idx):
                        continue
                    block_idx = start_idx // base_block_size - block_id_offset
                    if block_idx < 0 or block_idx >= len(block_ids):
                        continue
                    block_id = block_ids[block_idx]
                    if skip_null_blocks and block_id <= 0:
                        continue
                    hash_val = grouped_hashes[chunk_id]
                    yield start_idx, end_idx, prefix + block_hash_to_str(hash_val), hash_val, block_id

def get_block_hashes(
    block_hashes: BlockHashList | list[str],
    group_block_size: int,
    hash_block_size: int,
) -> Sequence[BlockHash | str]:
    if group_block_size == hash_block_size:
        return block_hashes
    assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    return _LazyGroupedBlockHashList(block_hashes, group_block_size // hash_block_size)

class _LazyGroupedBlockHashList(Sequence[BlockHash]):
    def __init__(self, block_hashes: Sequence[BlockHash | str], scale_factor: int) -> None:
        self._block_hashes = block_hashes
        self._scale_factor = scale_factor
        self._length = len(block_hashes) // scale_factor
        self._cache: dict[int, BlockHash] = {}

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for idx in range(self._length):
            yield self[idx]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        cached = self._cache.get(index)
        if cached is None:
            start = index * self._scale_factor
            end = start + self._scale_factor
            cached = _rehash_block_hash_group(self._block_hashes[start:end])
            self._cache[index] = cached
        return cached

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return list(self) == list(other)
        return False

class LayerMultiBlockReqMeta(_target.LayerMultiBlockReqMeta):
            def __init__(
                self,
                req_id: str,
                keys: list[LayerPoolKey],
                starts: list[int],
                ends: list[int],
                block_ids_by_group: list[list[int]] | None = None,
                layer_id: int = 0,
                is_last_chunk: bool | None = True,
                current_event: torch.npu.Event | None = None,
                block_ids: list[int] | list[list[int]] | None = None,
                token_ids: list[int] | None = None,
                original_block_size: list[int] | int | None = None,
                block_hashes: Sequence[Any] | None = None,
                kv_cache_group_id: int = 0,
            ) -> None:
                self.req_id = req_id
                self.keys = keys
                self.starts = starts
                self.ends = ends
                if block_ids_by_group is None:
                    block_ids_by_group = normalize_block_ids_by_group(block_ids or [])
                self.block_ids_by_group = block_ids_by_group
                self.layer_id = layer_id
                self.is_last_chunk = is_last_chunk
                self.current_event = current_event
                self.token_ids = token_ids
                self.original_block_size = original_block_size
                self.block_hashes = [] if block_hashes is None else block_hashes
                self.kv_cache_group_id = kv_cache_group_id
