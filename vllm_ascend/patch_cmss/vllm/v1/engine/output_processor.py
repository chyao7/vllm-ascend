# SPDX-License-Identifier: Apache-2.0

"""Patch vllm.v1.engine.output_processor for remote cached tokens."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.engine import output_processor as target

logger = init_logger(__name__)


def resolve_remote_cached_tokens(kv_transfer_params) -> int | None:
    if not isinstance(kv_transfer_params, dict):
        return None
    for key in ("remote_num_cached_tokens", "remote_cached_tokens"):
        remote_cached_tokens = kv_transfer_params.get(key)
        if remote_cached_tokens is not None:
            return remote_cached_tokens
    return None


def _reconcile_prefill_stats(prefill_stats, remote_cached_tokens: int) -> bool:
    prompt_tokens = getattr(prefill_stats, "num_prompt_tokens", None)
    if prompt_tokens is None:
        return False

    local_cached_tokens = getattr(prefill_stats, "num_local_cached_tokens", 0) or 0
    total_cached_tokens = min(
        max(remote_cached_tokens, local_cached_tokens), prompt_tokens
    )
    external_cached_tokens = total_cached_tokens - local_cached_tokens
    if (
        getattr(prefill_stats, "num_cached_tokens", None) == total_cached_tokens
        and getattr(prefill_stats, "num_external_cached_tokens", None)
        == external_cached_tokens
    ):
        return False

    prefill_stats.set(
        num_prompt_tokens=prompt_tokens,
        num_local_cached_tokens=local_cached_tokens,
        num_external_cached_tokens=external_cached_tokens,
    )
    return True


def _patched_process_outputs(
    self,
    engine_core_outputs,
    engine_core_timestamp=None,
    iteration_stats=None,
):
    for engine_core_output in engine_core_outputs:
        prefill_stats = getattr(engine_core_output, "prefill_stats", None)
        if prefill_stats is None:
            continue
        remote_cached_tokens = resolve_remote_cached_tokens(
            getattr(engine_core_output, "kv_transfer_params", None)
        )
        if remote_cached_tokens is None:
            continue
        changed = _reconcile_prefill_stats(prefill_stats, remote_cached_tokens)
        logger.info(
            "CMSS 4026bfb0 reconciled output cached tokens: "
            "request_id=%s remote=%s total=%s changed=%s",
            getattr(engine_core_output, "request_id", None),
            remote_cached_tokens,
            prefill_stats.num_cached_tokens,
            changed,
        )

    return self._cmss_original_process_outputs(
        engine_core_outputs,
        engine_core_timestamp,
        iteration_stats,
    )


def apply() -> None:
    output_processor_cls = target.OutputProcessor
    if getattr(output_processor_cls, "_cmss_cached_tokens_patched", False):
        return

    output_processor_cls._cmss_original_process_outputs = (
        output_processor_cls.process_outputs
    )
    output_processor_cls.process_outputs = _patched_process_outputs
    output_processor_cls._cmss_cached_tokens_patched = True
    logger.info(
        "CMSS monkey patch 4026bfb0 applied to "
        "vllm.v1.engine.output_processor.OutputProcessor.process_outputs."
    )
