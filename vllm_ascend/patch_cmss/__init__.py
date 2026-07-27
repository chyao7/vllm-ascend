"""Minimal CMSS monkey-patches migrated from commit 94f098aa.

The source files hold migrated implementations, while the explicit member
map below ensures only symbols changed by that commit are installed. Target
version members not listed here remain untouched.
"""

from __future__ import annotations

from importlib import import_module
import inspect
from types import ModuleType

from vllm.logger import logger


_PATCH_MODULES = (
    "distributed.kv_transfer.kv_pool.ascend_store.config_data",
    "distributed.kv_transfer.kv_pool.ascend_store.kv_transfer",
    "distributed.kv_transfer.kv_pool.ascend_store.pool_worker",
)

_PATCH_MEMBERS = {
    "config_data": {
        "TPMismatchInfo", "_as_positive_int", "infer_tp_mismatch_info",
        "_LazyGroupedBlockHashList", "get_block_hashes",
        "PoolKey.__init__", "PoolKey.split_layers", "ChunkedTokenDatabase.__init__",
        "ChunkedTokenDatabase._get_key_prefix", "ChunkedTokenDatabase._make_key_by_hash",
        "ChunkedTokenDatabase.process_token_key_strings", "ChunkedTokenDatabase.process_token_key_strings_with_block_ids",
        "ChunkedTokenDatabase.process_tokens", "ChunkedTokenDatabase.process_tokens_with_block_ids",
        "ChunkedTokenDatabase.set_group_buffers", "LayerMultiBlockReqMeta.__init__",
    },
    "kv_transfer": {
        "KVTransferThread._process_token_key_strings_with_block_ids",
        "KVCacheStoreSendingThread.__init__", "KVCacheStoreSendingThread._handle_request",
        "KVCacheStoreRecvingThread.__init__", "KVCacheStoreRecvingThread._handle_request",
    },
    "pool_worker": {
        "KVPoolWorker._init_key_head_config", "KVPoolWorker.start_load_kv", "KVPoolWorker.register_kv_caches",
        "KVPoolWorker.get_group_tp_size", "KVPoolWorker.lookup",
        "KVPoolWorker.lookup_scheduler", "KVPoolWorker._lookup_with_coordinator",
        "KVPoolWorker._make_sub_key_str", "KVPoolWorker._build_strided_addrs",
        "KVPoolWorker._build_tp_mismatch_keys_and_addrs", "KVPoolWorker._load_kv_tp_mismatch",
        "KVPoolWorker._store_kv_tp_mismatch", "KVPoolWorker._lookup_candidate_masks",
    },
}

_APPLIED = False


def _merge_module(target: ModuleType, patch: ModuleType, module_name: str) -> int:
    """Install changed definitions without replacing target classes wholesale."""
    patched_count = 0
    for name, value in vars(patch).items():
        if name.startswith("__"):
            continue
        current = getattr(target, name, None)
        if inspect.isclass(value) and inspect.isclass(current):
            # Preserve fields and methods added by the target version while
            # replacing methods supplied by the migrated commit.
            for member_name, member in vars(value).items():
                if member_name.startswith("__") and member_name not in {"__init__", "__new__"}:
                    continue
                if f"{name}.{member_name}" not in _PATCH_MEMBERS[module_name]:
                    continue
                if inspect.isfunction(member) or isinstance(member, (classmethod, staticmethod, property)):
                    setattr(current, member_name, member)
                    patched_count += 1
            continue
        if name in _PATCH_MEMBERS[module_name]:
            setattr(target, name, value)
            patched_count += 1
    return patched_count


def apply() -> None:
    """Apply all migrated CMSS patches in dependency order."""
    global _APPLIED
    if _APPLIED:
        logger.info("CMSS monkey patch 94f098aa already applied; skipping.")
        return

    logger.info("Applying CMSS monkey patch 94f098aa.")
    total_count = 0
    for suffix in _PATCH_MODULES:
        patch = import_module(f"vllm_ascend.patch_cmss.{suffix}")
        target = import_module(f"vllm_ascend.{suffix}")
        module_name = suffix.rsplit(".", 1)[-1]
        patched_count = _merge_module(target, patch, module_name)
        total_count += patched_count
        logger.info(
            "CMSS monkey patch 94f098aa applied to %s: %d symbols.",
            suffix,
            patched_count,
        )

    _APPLIED = True
    logger.info("CMSS monkey patch 94f098aa applied successfully: %d symbols total.", total_count)


__all__ = ["apply"]
