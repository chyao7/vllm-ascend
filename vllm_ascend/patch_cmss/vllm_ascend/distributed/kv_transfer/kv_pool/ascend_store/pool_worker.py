from __future__ import annotations

from importlib import import_module

_target = import_module('vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker')

globals().update({name: value for name, value in vars(_target).items() if not name.startswith("__")})

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import infer_tp_mismatch_info

_original_register_kv_caches = _target.KVPoolWorker.register_kv_caches

class KVPoolWorker(_target.KVPoolWorker):
            def start_load_kv(self, metadata: AscendConnectorMetadata):
                self.current_layer = 0
                self.layerwise_retrievers: list[Any] = []
                if self.use_layerwise:
                    self.next_layer_to_submit = 0
                    reset_attention_compute_start_gate()
                logger.debug("KV pool worker start_load_kv requests=%d", len(metadata.requests))
                if len(metadata.requests) == 0:
                    return
                if self.use_layerwise:
                    self.process_layer_data(metadata.requests)
                    return
                for request in metadata.requests:
                    load_spec = request.load_spec
                    if load_spec is None or not load_spec.can_load:  # load =0
                        logger.debug(
                            "KV pool worker skip get req=%s reason=%s",
                            request.req_id,
                            "no_load_spec" if load_spec is None else f"can_load={load_spec.can_load}",
                        )
                        continue
                    request.skip_null_blocks_by_group = self.group_uses_align_state
                    load_group_ids = request.kv_cache_group_ids or [0]
                    token_len = request.token_len_chunk
                    if (load_spec.kvpool_cached_tokens % self.cache_transfer_granularity != 0) and (
                        load_spec.kvpool_cached_tokens == token_len - 1
                    ):
                        token_len = request.load_spec.kvpool_cached_tokens + 1
                    else:
                        token_len = request.load_spec.kvpool_cached_tokens
                    request.load_spec.token_len = token_len
                    logger.debug(
                        "KV pool worker prepare get req=%s token_len_chunk=%d get_token_len=%d "
                        "vllm_cached=%d kvpool_cached=%d groups=%s load_async=%s",
                        request.req_id,
                        request.token_len_chunk,
                        token_len,
                        load_spec.vllm_cached_tokens,
                        load_spec.kvpool_cached_tokens,
                        load_group_ids,
                        self.load_async,
                    )
                    if self.tp_mismatch:
                        # tp_mismatch is restricted to non-hybrid -> single group.
                        group_block_size = self.grouped_block_size[0]
                        mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                        self._load_kv_tp_mismatch(
                            request.block_hashes,
                            request.block_ids_by_group[0],
                            token_len,
                            mask_num,
                        )
                    elif self.load_async:
                        self.kv_recv_thread.add_request(  # type: ignore[union-attr]
                            request,
                        )
                    else:
                        addr_list = []
                        size_list = []
                        key_list = []
                        block_id_list: list[int] = []
                        load_masks = self.token_database.load_mask(request.block_hashes, token_len)
                        for group_id in load_group_ids:
                            block_ids = request.block_ids_by_group[group_id]
                            group_block_size = self.grouped_block_size[group_id]
                            mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                            skip_null = group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]
                            for start, end, key, _block_hash, block_id in self.token_database.process_token_key_strings_with_block_ids(
                                token_len,
                                request.block_hashes,
                                block_ids,
                                mask_num,
                                kv_cache_group_id=group_id,
                                skip_null_blocks=skip_null,
                            ):
                                if not self.token_database.mask_allows_chunk(load_masks, group_id, start):
                                    continue
                                addr, size, block_id = self.token_database.prepare_value(
                                    start,
                                    end,
                                    block_ids,
                                    kv_cache_group_id=group_id,
                                    block_id=block_id,
                                )
                                key_list.append(key)
                                addr_list.append(addr)
                                size_list.append(size)
                                block_id_list.append(block_id)
                        if not key_list:
                            continue
                        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
                        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
                        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
                        block_id_list_c = (
                            block_id_list[self.tp_rank % len(block_id_list) :]
                            + block_id_list[: self.tp_rank % len(block_id_list)]
                        )
                        logger.debug(
                            "KV pool worker calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                            request.req_id,
                            token_len,
                            load_group_ids,
                            len(key_list_c),
                            key_list_c[:3],
                        )
                        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
                        if ret is not None and any(r != 0 for r in ret):
                            missing_block_ids = record_failed_blocks(
                                block_id_list_c,
                                ret,
                            )
                            if len(request.block_ids_by_group) == 1:
                                self._invalid_block_ids.update(missing_block_ids)
                            elif missing_block_ids:
                                logger.error(
                                    "KV load failed for hybrid request %s. "
                                    "Skip invalid-block fallback to avoid scheduler crash. "
                                    "failed_blocks=%s",
                                    request.req_id,
                                    missing_block_ids,
                                )
                        elif ret is None:
                            missing_block_ids = record_failed_blocks(
                                block_id_list_c,
                                [1] * len(block_id_list_c),
                            )
                            if len(request.block_ids_by_group) == 1:
                                self._invalid_block_ids.update(missing_block_ids)
                            elif missing_block_ids:
                                logger.error(
                                    "KV load failed for hybrid request %s. "
                                    "Skip invalid-block fallback to avoid scheduler crash. "
                                    "failed_blocks=%s",
                                    request.req_id,
                                    missing_block_ids,
                                )
                        logger.debug(
                            "KV pool worker backend get returned request=%s token_len=%d groups=%s keys=%d",
                            request.req_id,
                            token_len,
                            load_group_ids,
                            len(key_list_c),
                        )

            def lookup(
                self,
                token_len: int,
                block_hashes: list[BlockHash],
                kv_cache_group_ids: list[int] | None = None,
                use_layerwise: bool = False,
            ) -> int:
                """
                Checks the existence of KV cache of the tokens from the cache engine.
                :param tokens: the input tokens, with shape [seq_len]
                :return: An int indicating how many prefix tokens are cached.
                """
                try:
                    hits = []
                    kv_cache_group_ids = kv_cache_group_ids or [0]
                    coordinator_hit = self._lookup_with_coordinator(
                        token_len,
                        block_hashes,
                        kv_cache_group_ids,
                        use_layerwise,
                        include_all_ranks=False,
                    )
                    if coordinator_hit is not None:
                        return coordinator_hit
                    for group_id in kv_cache_group_ids:
                        end = 0
                        keys = []
                        starts = []
                        ends = []
                        if use_layerwise:
                            for start, end, key in self.token_database.process_tokens(
                                token_len,
                                block_hashes,
                                kv_cache_group_id=group_id,
                            ):
                                keys_multi_layer = key.split_layers(self.num_layers)
                                for item in keys_multi_layer:
                                    keys.append(item.to_string())
                                starts.append(start)
                                ends.append(end)
                        else:
                            for start, end, key_string, _ in self.token_database.process_token_key_strings(
                                token_len,
                                block_hashes,
                                kv_cache_group_id=group_id,
                            ):
                                keys.append(key_string)
                                starts.append(start)
                                ends.append(end)

                        if not keys:
                            hits.append(0)
                            continue

                        res = self.m_store.exists(keys)  # type: ignore[assignment]

                        if use_layerwise:
                            res = self.check_all_layers_exists(res, self.num_layers)
                        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                            hit_end = 0
                            for index in range(len(ends) - 1, -1, -1):
                                if (
                                    res[index] == 1  # type: ignore[index]
                                    and ends[index] % self.cache_transfer_granularity == 0
                                ):
                                    hit_end = ends[index]
                                    break
                        else:
                            hit_end = end
                            for index, value in enumerate(res):  # type: ignore[arg-type]
                                if value != 1:
                                    hit_end = 0
                                    for hit_index in range(index, 0, -1):
                                        if starts[hit_index] % self.cache_transfer_granularity == 0:
                                            hit_end = starts[hit_index]
                                            break
                                    break
                        hits.append(hit_end)
                except Exception as e:
                    logger.error(
                        "Remote connection failed in get_common_prefix_length. type=%s, error=%s. "
                        "Check network and remote store.",
                        type(e).__name__,
                        e,
                    )
                    return 0
                return min(hits) if hits else 0

            def get_group_tp_size(self, kv_cache_group_id: int):
                if self.tp_mismatch:
                    return self.effective_tp_size
                if self.group_uses_align_state[kv_cache_group_id]:
                    return self.tp_size
                return min(self.tp_size, self._get_group_num_kv_heads(kv_cache_group_id))

            def _make_sub_key_str(self, base_key, effective_rank: int) -> str:
                """Rewrite ``@head_or_tp_rank:<local>`` in base_key.to_string() to ``<effective_rank>``.

                Under TP mismatch, both sides address the pool at the effective_tp_size
                namespace rather than the local TP rank.
                """
                return self._replace_key_field(base_key.to_string(), "head_or_tp_rank", effective_rank)

            def _build_strided_addrs(self, block_id: int, token_count: int, sub_idx: int) -> tuple[list[int], list[int]]:
                """Build per-token (addr, size) pairs into local KV cache memory for one
                sub-key inside one block.

                KV cache layout: [num_block, block_size, num_kv_head_per_local_rank, head_dim].
                Heads of consecutive tokens are interleaved with token position, so a
                sub-slice of heads requires one transfer per token. Block stepping uses
                ``block_stride`` because the kernel may pad between blocks.
                """
                head_offset_bytes = sub_idx * self.sub_size_bytes
                addrs: list[int] = []
                sizes: list[int] = []
                # tp_mismatch is restricted to a single dense KV group -> group 0.
                group_addrs = self.group_kv_caches_base_addr[0]
                group_block_len = self.group_block_len[0]
                group_block_stride = self.group_block_stride[0]
                for base_addr, entry_block_len, entry_block_stride in zip(
                    group_addrs, group_block_len, group_block_stride, strict=True
                ):
                    entry_per_token_bytes = entry_block_len // self.block_size
                    block_base = base_addr + block_id * entry_block_stride
                    for t in range(token_count):
                        addrs.append(block_base + t * entry_per_token_bytes + head_offset_bytes)
                        sizes.append(self.sub_size_bytes)
                return addrs, sizes

            def _build_tp_mismatch_keys_and_addrs(
                self,
                block_hashes: list,
                block_ids: list[int],
                token_len: int,
                mask_num: int = 0,
            ) -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
                """Walk chunks x sub-keys; emit (keys, addrs, sizes, block_ids) for backend put/get.

                Each key represents one (chunk, sub_idx) pair. Its addrs/sizes cover all
                layer-entries x all tokens in the chunk, addressed at the head-slice
                owned by sub_idx within this rank's local cache.
                """
                all_keys: list[str] = []
                all_addrs: list[list[int]] = []
                all_sizes: list[list[int]] = []
                all_block_ids: list[int] = []
                for start, end, base_key, block_id in self.token_database.process_tokens_with_block_ids(
                    token_len,
                    block_hashes,
                    block_ids,
                    mask_num,
                ):
                    token_count = end - start
                    for sub_idx in range(self.num_sub_keys):
                        effective_rank = self.tp_rank * self.num_sub_keys + sub_idx
                        addrs, sizes = self._build_strided_addrs(block_id, token_count, sub_idx)
                        all_keys.append(self._make_sub_key_str(base_key, effective_rank))
                        all_addrs.append(addrs)
                        all_sizes.append(sizes)
                        all_block_ids.append(block_id)
                return all_keys, all_addrs, all_sizes, all_block_ids

            def _load_kv_tp_mismatch(
                self,
                block_hashes: list,
                block_ids: list[int],
                token_len: int,
                mask_num: int,
            ) -> None:
                keys, addrs, sizes, key_block_ids = self._build_tp_mismatch_keys_and_addrs(
                    block_hashes, block_ids, token_len, mask_num
                )
                if not keys:
                    return
                offset = self.tp_rank % len(keys)
                keys_c = keys[offset:] + keys[:offset]
                addrs_c = addrs[offset:] + addrs[:offset]
                sizes_c = sizes[offset:] + sizes[:offset]
                block_ids_c = key_block_ids[offset:] + key_block_ids[:offset]
                logger.debug(
                    "KV pool worker tp_mismatch get keys=%d sample_keys=%s",
                    len(keys_c),
                    keys_c[:3],
                )
                ret = self.m_store.get(keys_c, addrs_c, sizes_c)
                if ret is not None and any(r != 0 for r in ret):
                    missing_block_ids = record_failed_blocks(block_ids_c, ret)
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                elif ret is None:
                    missing_block_ids = record_failed_blocks(block_ids_c, [1] * len(block_ids_c))
                    with self._invalid_block_ids_lock:
                        self._invalid_block_ids.update(missing_block_ids)
                logger.debug(
                    "KV pool worker tp_mismatch get returned keys=%d",
                    len(keys_c),
                )

            def _store_kv_tp_mismatch(self, req_meta: ReqMeta) -> None:
                send_thread = self.kv_send_thread
                if send_thread is None:
                    return
                req_id = req_meta.req_id
                if not send_thread.is_stored_request(req_id):  # type: ignore[attr-defined]
                    return
                try:
                    token_len = req_meta.token_len_chunk
                    block_ids = req_meta.block_ids_by_group[0]
                    keys, addrs, sizes, _ = self._build_tp_mismatch_keys_and_addrs(
                        req_meta.block_hashes, block_ids, token_len, mask_num=0
                    )
                    if not keys:
                        return
                    exists_states = send_thread.lookup(keys)  # type: ignore[attr-defined]
                    missing_indices = [i for i, exists in enumerate(exists_states) if not exists]
                    if not missing_indices:
                        return
                    keys = [keys[i] for i in missing_indices]
                    addrs = [addrs[i] for i in missing_indices]
                    sizes = [sizes[i] for i in missing_indices]
                    if req_meta.current_event is not None:
                        req_meta.current_event.synchronize()
                    logger.debug(
                        "KV pool worker tp_mismatch put req=%s keys=%d sample_keys=%s",
                        req_id,
                        len(keys),
                        keys[:3],
                    )
                    self.m_store.put(keys, addrs, sizes)

                    if self.enable_kv_events:
                        event_block_size = (
                            req_meta.original_block_size[0]
                            if isinstance(req_meta.original_block_size, list)
                            else req_meta.original_block_size
                        )
                        stored_events: list[BlockStored] = []
                        prev_key = None
                        for idx, (start, end, _base_key) in enumerate(
                            self.token_database.process_tokens(token_len, req_meta.block_hashes)
                        ):
                            if idx >= len(req_meta.block_hashes):
                                break
                            block_hash = maybe_convert_block_hash(req_meta.block_hashes[idx])
                            token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
                            stored_events.append(
                                BlockStored(
                                    block_hashes=[block_hash],
                                    parent_block_hash=prev_key,
                                    token_ids=token_ids,
                                    block_size=event_block_size,
                                    lora_id=None,
                                    medium="cpu",
                                    lora_name=None,
                                )
                            )
                            prev_key = block_hash
                        if stored_events:
                            send_thread.update_kv_event(stored_events)  # type: ignore[attr-defined]
                finally:
                    send_thread.dec_stored_request(req_id)  # type: ignore[attr-defined]

            def _lookup_candidate_masks(self, token_len: int) -> tuple[list[bool], ...] | None:
                """Get store_mask for lookup pre-filtering.

                reachable_block_mask (which store_mask delegates to) computes its
                pattern from start_block=0 using absolute block positions. The pattern
                is position-stable across requests of different lengths:

                - For SWA/State groups (cache_family in {None, "default", "c1"}):
                  the mask marks only the tail blocks of each alignment segment as
                  cacheable. Since start_block is always 0, two requests of different
                  lengths share the same mask values in their overlapping region.

                - For c4/c128 groups (cache_family starts with "c"):
                  store_mask returns all-True, so no filtering is applied.

                - retention_interval (if set) adds subsampling on top of the base
                  pattern, but still keyed on absolute block_index, so it remains
                  stable across requests as long as the interval is not changed at
                  runtime (which requires a worker restart).

                Therefore it is safe to use store_mask as a hard filter for lookup:
                any chunk with mask=False could never have been stored by any previous
                request, so querying it would be a guaranteed miss.
                """
                if self.cache_coordinator is None:
                    return None
                try:
                    return self.cache_coordinator.store_mask(token_len, token_len)
                except AssertionError as exc:
                    logger.debug("Skip AscendStore lookup candidate mask for unaligned token_len=%d: %s", token_len, exc)
                    return None

            def _lookup_with_coordinator(
                self,
                token_len: int,
                block_hashes: list[BlockHash],
                kv_cache_group_ids: list[int],
                use_layerwise: bool,
                include_all_ranks: bool,
            ) -> int | None:
                if self.cache_coordinator is None or use_layerwise:
                    return None
                if sorted(kv_cache_group_ids) != list(range(self.num_kv_cache_groups)):
                    return None

                exists: set[tuple[int, bytes]] = set()
                lookup_masks = self._lookup_candidate_masks(token_len)

                all_keys: list[str] = []
                group_ranges: list[tuple[int, int, list[str], list[int]]] = []
                for group_id in kv_cache_group_ids:
                    keys: list[str] = []
                    chunk_hashes: list[str] = []
                    variant_counts: list[int] = []
                    group_mask = lookup_masks[group_id] if lookup_masks is not None and group_id < len(lookup_masks) else None
                    total_chunks = 0
                    skipped_chunks = 0
                    for _, _, key in self.token_database.process_tokens(
                        token_len,
                        block_hashes,
                        kv_cache_group_id=group_id,
                        chunk_mask=group_mask,
                    ):
                        variants = self._expand_lookup_key_variants(key.to_string(), group_id, include_all_ranks)
                        keys.extend(variants)
                        chunk_hashes.append(key.chunk_hash)
                        variant_counts.append(len(variants))

                    if group_mask is not None:
                        total_chunks = len(group_mask)
                        skipped_chunks = sum(1 for allowed in group_mask if not allowed)

                    group_start = len(all_keys)
                    all_keys.extend(keys)
                    group_ranges.append((group_start, len(keys), chunk_hashes, variant_counts))

                    logger.info(
                        "KV pool coordinator lookup group=%d token_len=%d keys=%d "
                        "mask_skipped=%d/%d sample_keys=%s",
                        group_id,
                        token_len,
                        len(keys),
                        skipped_chunks,
                        total_chunks,
                        keys[:3],
                    )

                if not all_keys:
                    _, hit_length = self.cache_coordinator.find_longest_cache_hit(
                        block_hashes, token_len, ExternalCachedBlockPool(exists), apply_eagle=False,
                    )
                    return hit_length

                all_res = self.m_store.exists(all_keys)  # type: ignore[assignment]

                for group_id, (g_start, g_count, chunk_hashes, variant_counts) in zip(kv_cache_group_ids, group_ranges):
                    if g_count == 0:
                        continue
                    res = all_res[g_start : g_start + g_count]  # type: ignore[index]
                    offset = 0
                    for chunk_hash, count in zip(chunk_hashes, variant_counts, strict=True):
                        values = res[offset : offset + count]  # type: ignore[index]
                        if values and all(value == 1 for value in values):
                            exists.add((group_id, block_hash_to_bytes(chunk_hash)))
                        offset += count
                    logger.info(
                        "KV pool coordinator lookup group=%d exists_chunks=%d/%d",
                        group_id,
                        sum(1 for group, _ in exists if group == group_id),
                        len(chunk_hashes),
                    )

                _, hit_length = self.cache_coordinator.find_longest_cache_hit(
                    block_hashes,
                    token_len,
                    ExternalCachedBlockPool(exists),
                    apply_eagle=False,
                )
                logger.info(
                    "KV pool coordinator lookup final token_len=%d groups=%s hit=%d total_keys=%d",
                    token_len,
                    kv_cache_group_ids,
                    hit_length,
                    len(all_keys),
                )
                return hit_length

            def lookup_scheduler(
                self,
                token_len: int,
                block_hashes: list[BlockHash],
                kv_cache_group_ids: list[int] | None = None,
                use_layerwise: bool = False,
                hbm_hit_tokens: int = 0,
            ) -> int:
                """
                Checks the existence of KV cache of the tokens from the cache engine.
                :param tokens: the input tokens, with shape [seq_len]
                :return: An int indicating how many prefix tokens are cached.
                """
                # The d19 lookup RPC forwards hbm_hit_tokens. Keep the verified
                # migrated behavior of checking the complete external prefix.
                try:
                    hits: list[list[int]] = []
                    max_hit_position = self.max_model_len
                    kv_cache_group_ids = kv_cache_group_ids or [0]
                    coordinator_hit = self._lookup_with_coordinator(
                        token_len,
                        block_hashes,
                        kv_cache_group_ids,
                        use_layerwise,
                        include_all_ranks=True,
                    )
                    if coordinator_hit is not None:
                        return coordinator_hit

                    lookup_masks = self._lookup_candidate_masks(token_len)

                    all_keys: list[str] = []
                    group_ranges: list[tuple[int, int, list[int], list[int], int, int]] = []
                    for group_id in kv_cache_group_ids:
                        keys = []
                        starts = []
                        ends = []
                        group_mask = lookup_masks[group_id] if lookup_masks is not None and group_id < len(lookup_masks) else None
                        if use_layerwise:
                            for start, end, key in self.token_database.process_tokens(
                                token_len,
                                block_hashes,
                                kv_cache_group_id=group_id,
                                chunk_mask=group_mask,
                            ):
                                keys_multi_layer = key.split_layers(self.num_layers)
                                for item in keys_multi_layer:
                                    keys.append(item.to_string())
                                starts.append(start)
                                ends.append(end)
                        else:
                            for start, end, key_string, _ in self.token_database.process_token_key_strings(
                                token_len,
                                block_hashes,
                                kv_cache_group_id=group_id,
                                chunk_mask=group_mask,
                            ):
                                keys.append(key_string)
                                starts.append(start)
                                ends.append(end)

                        if not keys:
                            return 0

                        multi_tp_keys = keys[:]
                        group_tp_size = self.get_group_tp_size(group_id)
                        for i in range(1, group_tp_size):
                            for item in keys:
                                new_str = self._replace_key_field(item, "head_or_tp_rank", i)
                                multi_tp_keys.append(new_str)

                        pp_base_keys = multi_tp_keys.copy()
                        for i in range(1, self.pp_size):
                            for item in pp_base_keys:
                                new_str = self._replace_key_field(item, "pp_rank", i)
                                multi_tp_keys.append(new_str)

                        num_block = len(keys)
                        if use_layerwise:
                            num_block = len(keys) // self.num_layers

                        group_start = len(all_keys)
                        all_keys.extend(multi_tp_keys)
                        group_ranges.append((group_start, len(multi_tp_keys), starts, ends, num_block, group_tp_size * self.pp_size))

                        logger.debug(
                            "KV pool lookup request token_len=%d group=%d keys=%d multi_tp_keys=%d "
                            "sample_keys=%s",
                            token_len,
                            group_id,
                            len(keys),
                            len(multi_tp_keys),
                            multi_tp_keys[:3],
                        )

                    if not all_keys:
                        return 0

                    all_res = self.m_store.exists(all_keys)  # type: ignore[assignment]

                    for group_id, (g_start, g_count, starts, ends, num_block, num_ranks) in zip(kv_cache_group_ids, group_ranges):
                        if g_count == 0:
                            hits.append([])
                            continue
                        res = all_res[g_start : g_start + g_count]  # type: ignore[index]
                        if use_layerwise:
                            res = self.check_all_layers_exists(res, self.num_layers)
                        multi_tp_values = [
                            res[i * num_block : (i + 1) * num_block]  # type: ignore[index]
                            for i in range(num_ranks)
                        ]
                        logger.info(
                            "KV pool lookup group=%d exists_count=%d/%d exists_sample=%s",
                            group_id,
                            sum(1 for value in res if value == 1),  # type: ignore[union-attr]
                            len(res),
                            list(res[: min(12, len(res))]),  # type: ignore[index]
                        )
                        if group_id < len(self.group_uses_align_state) and self.group_uses_align_state[group_id]:
                            group_hits = self.find_all_discontinuous_hit_positions(
                                multi_tp_values, ends, num_block, max_hit_position, self.cache_transfer_granularity
                            )
                        else:
                            group_hits = self.find_all_continuous_hit_positions(
                                multi_tp_values, ends, num_block, max_hit_position, self.cache_transfer_granularity
                            )
                        if not group_hits:
                            return 0
                        max_hit_position = min(max_hit_position, group_hits[-1])
                        hits.append(group_hits)
                        logger.info(
                            "KV pool scheduler lookup group=%d hit=%d token_len=%d",
                            group_id,
                            max_hit_position,
                            token_len,
                        )
                except Exception as e:
                    logger.error(
                        "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                        type(e).__name__,
                        e,
                    )
                    return 0
                final_hits = self._max_intersection_hit_position(hits)
                logger.debug(
                    "KV pool scheduler lookup final token_len=%d groups=%s hit=%d",
                    token_len,
                    kv_cache_group_ids,
                    final_hits,
                )
                return final_hits

_original_init_key_head_config = _target.KVPoolWorker._init_key_head_config

def _init_key_head_config(self, model_config, parallel_config) -> None:
    _original_init_key_head_config(self, model_config, parallel_config)
    info = infer_tp_mismatch_info(
        self.kv_role, self._extra_config, self.tp_size, self.num_kv_head,
        self.use_mla, self.use_hybrid,
    )
    self.peer_tp_size = info.peer_tp_size
    self.effective_tp_size = info.effective_tp_size
    self.tp_mismatch = info.enabled
    self.local_heads_per_rank = info.local_heads_per_rank
    self.effective_heads_per_rank = info.effective_heads_per_rank
    self.num_sub_keys = info.num_sub_keys
    if self.tp_mismatch and (self.use_sparse or self.use_layerwise or self.use_hybrid):
        raise ValueError("TP mismatch only supports non-sparse, non-layerwise, dense KV layouts")

KVPoolWorker._init_key_head_config = _init_key_head_config

def _register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    _original_register_kv_caches(self, kv_caches)
    if not self.tp_mismatch:
        return
    first_cache = self._as_cache_tuple(next(iter(kv_caches.values())))[0]
    self.elem_size = first_cache.element_size()
    self.head_dim = first_cache.shape[-1]
    self.per_token_bytes = self.group_block_len[0][0] // self.block_size
    self.sub_size_bytes = self.effective_heads_per_rank * self.head_dim * self.elem_size
    if self.kv_send_thread is not None:
        self.kv_send_thread.worker = self
    if self.kv_recv_thread is not None:
        self.kv_recv_thread.worker = self

KVPoolWorker.register_kv_caches = _register_kv_caches
