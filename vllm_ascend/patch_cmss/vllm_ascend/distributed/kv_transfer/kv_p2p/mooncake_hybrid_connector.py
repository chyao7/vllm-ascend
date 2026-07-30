# SPDX-License-Identifier: Apache-2.0

"""Function-level monkey-patch migrated from commit 4026bfb0."""

from __future__ import annotations

from functools import wraps

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_hybrid_connector as target,
)


def _resolve_prefill_cached_tokens(prefill_stats) -> int | None:
    if prefill_stats is None:
        return None
    cached_tokens = getattr(prefill_stats, "num_cached_tokens", None)
    if cached_tokens is not None:
        return cached_tokens
    local_cached_tokens = getattr(prefill_stats, "num_local_cached_tokens", 0) or 0
    external_cached_tokens = getattr(prefill_stats, "num_external_cached_tokens", 0) or 0
    return local_cached_tokens + external_cached_tokens


def apply() -> None:
    scheduler_cls = target.MooncakeConnectorScheduler
    if getattr(scheduler_cls, "_cmss_4026bfb0_applied", False):
        return

    original_update_state_after_alloc = scheduler_cls.update_state_after_alloc
    original_request_finished_all_groups = scheduler_cls.request_finished_all_groups

    @wraps(original_update_state_after_alloc)
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params
        was_remote_prefill = bool(params and params.get("do_remote_prefill"))
        result = original_update_state_after_alloc(
            self, request, blocks, num_external_tokens
        )

        prefill_stats = request.prefill_stats
        if was_remote_prefill and prefill_stats is not None:
            remote_num_cached_tokens = params.get("remote_num_cached_tokens")
            if remote_num_cached_tokens is None:
                remote_num_cached_tokens = params.get("remote_cached_tokens")
            if remote_num_cached_tokens is not None:
                prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=prefill_stats.num_local_cached_tokens,
                    num_external_cached_tokens=remote_num_cached_tokens,
                )
                logger.info(
                    "CMSS 4026bfb0 corrected Mooncake cached tokens: "
                    "request_id=%s remote=%s prefill_stats=%s",
                    request.request_id,
                    remote_num_cached_tokens,
                    prefill_stats,
                )
        return result

    @wraps(original_request_finished_all_groups)
    def request_finished_all_groups(self, request, block_ids):
        delay_free_blocks, params = original_request_finished_all_groups(
            self, request, block_ids
        )
        if params is not None:
            remote_num_cached_tokens = _resolve_prefill_cached_tokens(
                request.prefill_stats
            )
            params["remote_num_cached_tokens"] = remote_num_cached_tokens
            logger.info(
                "CMSS 4026bfb0 exported Mooncake cached tokens: "
                "request_id=%s cached_tokens=%s",
                request.request_id,
                remote_num_cached_tokens,
            )
        return delay_free_blocks, params

    scheduler_cls.update_state_after_alloc = update_state_after_alloc
    scheduler_cls.request_finished_all_groups = request_finished_all_groups
    scheduler_cls._cmss_4026bfb0_applied = True
    logger.info(
        "CMSS monkey patch 4026bfb0 applied to "
        "MooncakeConnectorScheduler (2 methods)."
    )
