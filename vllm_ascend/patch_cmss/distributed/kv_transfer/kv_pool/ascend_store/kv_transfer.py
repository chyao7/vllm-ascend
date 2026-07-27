from __future__ import annotations

from importlib import import_module

from collections.abc import Callable, Mapping

_target = import_module('vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer')

globals().update({name: value for name, value in vars(_target).items() if not name.startswith("__")})

_original_sending_init = _target.KVCacheStoreSendingThread.__init__

_original_recving_init = _target.KVCacheStoreRecvingThread.__init__

class KVTransferThread(_target.KVTransferThread):
            def _process_token_key_strings_with_block_ids(
                self,
                token_len: int,
                block_hashes,
                block_ids: list[int],
                mask_num: int = 0,
                kv_cache_group_id: int = 0,
                skip_null_blocks: bool = False,
                cache_role: str = "kv",
            ):
                process_key_strings = getattr(self.token_database, "process_token_key_strings_with_block_ids", None)
                if process_key_strings is not None:
                    return process_key_strings(
                        token_len,
                        block_hashes,
                        block_ids,
                        mask_num=mask_num,
                        kv_cache_group_id=kv_cache_group_id,
                        skip_null_blocks=skip_null_blocks,
                        cache_role=cache_role,
                    )

                def iter_with_pool_keys():
                    for start, end, key, block_id in self._process_tokens_with_block_ids(
                        token_len,
                        block_hashes,
                        block_ids,
                        mask_num,
                        kv_cache_group_id=kv_cache_group_id,
                        skip_null_blocks=skip_null_blocks,
                        cache_role=cache_role,
                    ):
                        yield start, end, key.to_string(), getattr(key, "chunk_hash_bytes", key.chunk_hash), block_id

                return iter_with_pool_keys()

class KVCacheStoreSendingThread(_target.KVCacheStoreSendingThread):
            def _handle_request(self, req_meta: ReqMeta):
                if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
                    try:
                        self.worker._store_kv_tp_mismatch(req_meta)
                    finally:
                        self.request_queue.task_done()
                    return
                token_len = req_meta.token_len_chunk
                req_id = req_meta.req_id
                current_event = req_meta.current_event
                try:
                    if req_id not in self.stored_requests:
                        self.request_queue.task_done()
                        return

                    store_masks = self._store_mask(req_meta)
                    group_ids = req_meta.kv_cache_group_ids or [0]

                    group_collected: dict[int, tuple[list, list, list, list, list, list]] = {}
                    all_lookup_keys: list[str] = []
                    group_lookup_offsets: list[tuple[int, int, int]] = []

                    for group_id in group_ids:
                        starts = []
                        ends = []
                        keys = []
                        block_hashes = []
                        key_block_ids = []
                        block_ids = req_meta.block_ids_by_group[group_id]
                        group_block_size = self._get_block_size(group_id)

                        for start, end, key, block_hash, block_id in self._process_token_key_strings_with_block_ids(
                            token_len,
                            req_meta.block_hashes,
                            block_ids,
                            kv_cache_group_id=group_id,
                            skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                        ):
                            if not self._mask_allows_chunk(store_masks, group_id, start):
                                continue
                            starts.append(start)
                            ends.append(end)
                            keys.append(key)
                            block_hashes.append(block_hash)
                            key_block_ids.append(block_id)

                        if (
                            not self.dcp_size > 1
                            and not req_meta.disable_tp_key_sharding
                            and not self.group_uses_align_state[group_id]
                        ):
                            starts = starts[self.tp_rank % self.put_step :: self.put_step]
                            ends = ends[self.tp_rank % self.put_step :: self.put_step]
                            keys = keys[self.tp_rank % self.put_step :: self.put_step]
                            block_hashes = block_hashes[self.tp_rank % self.put_step :: self.put_step]
                            key_block_ids = key_block_ids[self.tp_rank % self.put_step :: self.put_step]

                        group_collected[group_id] = (starts, ends, keys, block_hashes, key_block_ids, block_ids)
                        group_lookup_offsets.append((group_id, len(all_lookup_keys), len(keys)))
                        all_lookup_keys.extend(keys)

                    all_exists = self.lookup(all_lookup_keys) if all_lookup_keys else []

                    all_keys: list[str] = []
                    all_addrs: list[list[int]] = []
                    all_sizes: list[list[int]] = []
                    all_stored_events: list[BlockStored] = []

                    for group_id, offset, count in group_lookup_offsets:
                        if count == 0:
                            continue
                        starts, ends, keys, block_hashes, key_block_ids, block_ids = group_collected[group_id]
                        group_block_size = self._get_block_size(group_id)

                        exists_states = all_exists[offset : offset + count]
                        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

                        if not missing_indices:
                            continue

                        starts = [starts[index] for index in missing_indices]
                        ends = [ends[index] for index in missing_indices]
                        keys = [keys[index] for index in missing_indices]
                        block_hashes = [block_hashes[index] for index in missing_indices]
                        key_block_ids = [key_block_ids[index] for index in missing_indices]

                        logger.info(
                            "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                            len(keys),
                            token_len // group_block_size,
                            len(missing_indices),
                            req_id,
                            group_id,
                        )
                        logger.debug(
                            "KV pool put request=%s group=%d token_len=%d keys=%d sample_keys=%s",
                            req_id,
                            group_id,
                            token_len,
                            len(keys),
                            keys[:3],
                        )

                        addrs = []
                        sizes = []
                        stored_events: list[BlockStored] = []
                        prev_key = None
                        new_block_hashes = [maybe_convert_block_hash(bh) for bh in block_hashes]
                        for index, start in enumerate(starts):
                            addr, size, _ = self._prepare_value(
                                start,
                                ends[index],
                                block_ids,
                                kv_cache_group_id=group_id,
                                block_id=key_block_ids[index],
                            )
                            addrs.append(addr)
                            sizes.append(size)

                            if self.enable_kv_event:
                                token_ids = req_meta.token_ids[start : ends[index]] if req_meta.token_ids is not None else None
                                block_size = (
                                    req_meta.original_block_size[group_id]
                                    if isinstance(req_meta.original_block_size, list)
                                    else req_meta.original_block_size
                                )
                                if block_size is not None:
                                    stored_event = BlockStored(
                                        block_hashes=[new_block_hashes[index]],
                                        parent_block_hash=prev_key,
                                        token_ids=token_ids,
                                        block_size=block_size,
                                        lora_id=None,
                                        medium="cpu",
                                        lora_name=None,
                                    )
                                    stored_events.append(stored_event)
                                    prev_key = new_block_hashes[index]
                                    logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

                        if self.kv_role == "kv_consumer":
                            keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                                keys,
                                addrs,
                                sizes,
                                kv_cache_group_id=group_id,
                            )

                        all_keys.extend(keys)
                        all_addrs.extend(addrs)
                        all_sizes.extend(sizes)
                        all_stored_events.extend(stored_events)

                    if all_keys:
                        if current_event is not None:
                            current_event.synchronize()
                        self.m_store.put(all_keys, all_addrs, all_sizes)

                        if self.enable_kv_event and all_stored_events:
                            self.update_kv_event(all_stored_events)
                finally:
                    self.mark_completed_events(req_meta.event_id)
                self.dec_stored_request(req_id)
                if self.stored_requests.get(req_id, -1) == 0:
                    self.delete_finished_stored_request(req_id)
                    self.set_finished_request(req_id)
                self.request_queue.task_done()

class KVCacheStoreRecvingThread(_target.KVCacheStoreRecvingThread):
            def _handle_request(self, req_meta: ReqMeta):
                try:
                    load_spec = req_meta.load_spec
                    req_id = req_meta.req_id
                    if load_spec is None:
                        logger.error("KV pool async recv request %s has no load spec; skip load.", req_id)
                        self.set_finished_request(req_id)
                        return
                    token_len = load_spec.token_len
                    if self.worker is not None and getattr(self.worker, "tp_mismatch", False):
                        group_block_size = self._get_block_size(0)
                        mask_num = load_spec.vllm_cached_tokens // group_block_size * group_block_size
                        self.worker._load_kv_tp_mismatch(
                            req_meta.block_hashes,
                            req_meta.block_ids_by_group[0],
                            token_len,
                            mask_num,
                        )
                        self.set_finished_request(req_id)
                        return
                    addr_list = []
                    size_list = []
                    key_list = []
                    block_id_list: list[int] = []
                    group_ids = req_meta.kv_cache_group_ids or [0]
                    load_masks = self._load_mask(req_meta, token_len)
                    for group_id in group_ids:
                        block_ids = req_meta.block_ids_by_group[group_id]
                        group_block_size = self._get_block_size(group_id)
                        mask_num = (
                            load_spec.vllm_cached_tokens
                            // group_block_size
                            * group_block_size
                        )
                        for start, end, key, _block_hash, block_id in self._process_token_key_strings_with_block_ids(
                            token_len,
                            req_meta.block_hashes,
                            block_ids,
                            mask_num,
                            kv_cache_group_id=group_id,
                            skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                        ):
                            if not self._mask_allows_chunk(load_masks, group_id, start):
                                continue
                            addr, size, block_id = self._prepare_value(
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
                        self.set_finished_request(req_id)
                        return
                    key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
                    addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
                    size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
                    block_id_list_c = (
                        block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
                    )
                    logger.debug(
                        "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
                        req_id,
                        token_len,
                        req_meta.kv_cache_group_ids or [0],
                        len(key_list_c),
                        key_list_c[:3],
                    )
                    ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
                    if ret is not None and any(r != 0 for r in ret):
                        missing_block_ids = record_failed_blocks(
                            block_id_list_c,
                            ret,
                        )
                        if len(req_meta.block_ids_by_group) == 1:
                            with self._invalid_block_ids_lock:
                                self._invalid_block_ids.update(missing_block_ids)
                        elif missing_block_ids:
                            logger.error(
                                "KV load failed for hybrid request %s. "
                                "Skip invalid-block fallback to avoid scheduler crash. "
                                "failed_blocks=%s",
                                req_id,
                                missing_block_ids,
                            )
                    elif ret is None:
                        missing_block_ids = record_failed_blocks(
                            block_id_list_c,
                            [1] * len(block_id_list_c),
                        )
                        if len(req_meta.block_ids_by_group) == 1:
                            with self._invalid_block_ids_lock:
                                self._invalid_block_ids.update(missing_block_ids)
                        elif missing_block_ids:
                            logger.error(
                                "KV load failed for hybrid request %s. "
                                "Skip invalid-block fallback to avoid scheduler crash. "
                                "failed_blocks=%s",
                                req_id,
                                missing_block_ids,
                            )
                    logger.debug(
                        "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
                        req_id,
                        token_len,
                        req_meta.kv_cache_group_ids or [0],
                        len(key_list_c),
                    )
                    self.set_finished_request(req_id)
                finally:
                    self.request_queue.task_done()

def _sending_init(self, *args, worker=None, **kwargs):
    _original_sending_init(self, *args, **kwargs)
    self.worker = worker

def _recving_init(self, *args, worker=None, **kwargs):
    _original_recving_init(self, *args, **kwargs)
    self.worker = worker

KVCacheStoreSendingThread.__init__ = _sending_init

KVCacheStoreRecvingThread.__init__ = _recving_init
