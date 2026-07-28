# SPDX-License-Identifier: Apache-2.0

"""Patch vLLM completion usage reporting for remote cached tokens."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from vllm.entrypoints.openai.completion import protocol
from vllm.entrypoints.openai.completion import serving as target
from vllm.logger import init_logger

from vllm_ascend.patch_cmss.vllm.v1.engine.output_processor import (
    resolve_remote_cached_tokens,
)

logger = init_logger(__name__)


def _resolve_output_cached_tokens(res) -> int | None:
    remote_cached_tokens = resolve_remote_cached_tokens(
        getattr(res, "kv_transfer_params", None)
    )
    if remote_cached_tokens is not None:
        return remote_cached_tokens
    return getattr(res, "num_cached_tokens", None)


def _set_prompt_tokens_details(usage, cached_tokens: int | None) -> None:
    if (
        usage is None
        or cached_tokens is None
        or getattr(usage, "prompt_tokens_details", None) is not None
    ):
        return
    usage.prompt_tokens_details = target.PromptTokenUsageInfo(
        cached_tokens=cached_tokens
    )


def _inject_prompt_tokens_details(data: str, cached_tokens: int | None) -> str:
    if cached_tokens is None or not data.startswith("data: "):
        return data

    payload = data[6:]
    if payload.endswith("\n\n"):
        payload = payload[:-2]
    if payload == "[DONE]":
        return data
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return data

    usage = chunk.get("usage")
    if not isinstance(usage, dict) or usage.get("prompt_tokens_details") is not None:
        return data
    usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _tracked_result_generator(
    result_generator: AsyncIterator, state: dict[str, int | None]
):
    async for prompt_idx, res in result_generator:
        cached_tokens = _resolve_output_cached_tokens(res)
        if cached_tokens is not None:
            state["cached_tokens"] = cached_tokens
        yield prompt_idx, res


async def _wrapped_stream_generator(
    self,
    request: protocol.CompletionRequest,
    engine_inputs,
    result_generator: AsyncIterator,
    request_id: str,
    created_time: int,
    model_name: str,
    num_prompts: int,
    tokenizer,
    request_metadata,
):
    state: dict[str, int | None] = {"cached_tokens": None}
    async for data in self._cmss_original_completion_stream_generator(
        request,
        engine_inputs,
        _tracked_result_generator(result_generator, state),
        request_id,
        created_time,
        model_name,
        num_prompts,
        tokenizer,
        request_metadata,
    ):
        yield _inject_prompt_tokens_details(data, state["cached_tokens"])

    if self.enable_prompt_tokens_details:
        _set_prompt_tokens_details(
            request_metadata.final_usage_info, state["cached_tokens"]
        )


def _wrapped_response(
    self,
    final_res_batch,
    request: protocol.CompletionRequest,
    request_id: str,
    created_time: int,
    model_name: str,
    tokenizer,
    request_metadata,
):
    response = self._cmss_original_completion_response(
        final_res_batch,
        request,
        request_id,
        created_time,
        model_name,
        tokenizer,
        request_metadata,
    )
    cached_tokens = None
    for final_res in final_res_batch:
        resolved_cached_tokens = _resolve_output_cached_tokens(final_res)
        if resolved_cached_tokens is not None:
            cached_tokens = resolved_cached_tokens

    if self.enable_prompt_tokens_details:
        _set_prompt_tokens_details(response.usage, cached_tokens)
        _set_prompt_tokens_details(
            request_metadata.final_usage_info, cached_tokens
        )
    return response


def apply() -> None:
    serving_cls = target.OpenAIServingCompletion
    if getattr(serving_cls, "_cmss_cached_tokens_patched", False):
        return

    serving_cls._cmss_original_completion_stream_generator = (
        serving_cls.completion_stream_generator
    )
    serving_cls._cmss_original_completion_response = (
        serving_cls.request_output_to_completion_response
    )
    serving_cls.completion_stream_generator = _wrapped_stream_generator
    serving_cls.request_output_to_completion_response = _wrapped_response
    serving_cls._cmss_cached_tokens_patched = True
    logger.info(
        "CMSS monkey patch 4026bfb0 applied to "
        "vllm.entrypoints.openai.completion.serving (2 methods)."
    )
