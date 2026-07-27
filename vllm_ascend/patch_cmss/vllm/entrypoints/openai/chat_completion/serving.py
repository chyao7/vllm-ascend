# SPDX-License-Identifier: Apache-2.0

"""Patch vLLM chat usage reporting for remote cached tokens."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from vllm.entrypoints.openai.chat_completion import protocol
from vllm.entrypoints.openai.chat_completion import serving as target
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
    async for res in result_generator:
        cached_tokens = _resolve_output_cached_tokens(res)
        if cached_tokens is not None:
            state["cached_tokens"] = cached_tokens
        yield res


async def _wrapped_stream_generator(
    self,
    request: protocol.ChatCompletionRequest,
    result_generator: AsyncIterator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata,
    reasoning_parser=None,
    **extra_kwargs,
):
    state: dict[str, int | None] = {"cached_tokens": None}
    async for data in self._cmss_original_chat_stream_generator(
        request,
        _tracked_result_generator(result_generator, state),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        reasoning_parser,
        **extra_kwargs,
    ):
        yield _inject_prompt_tokens_details(data, state["cached_tokens"])

    if self.enable_prompt_tokens_details:
        _set_prompt_tokens_details(
            request_metadata.final_usage_info, state["cached_tokens"]
        )


async def _wrapped_full_generator(
    self,
    request: protocol.ChatCompletionRequest,
    result_generator: AsyncIterator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata,
    reasoning_parser=None,
):
    state: dict[str, int | None] = {"cached_tokens": None}
    response = await self._cmss_original_chat_full_generator(
        request,
        _tracked_result_generator(result_generator, state),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        reasoning_parser,
    )

    if self.enable_prompt_tokens_details:
        if isinstance(response, protocol.ChatCompletionResponse):
            _set_prompt_tokens_details(response.usage, state["cached_tokens"])
        _set_prompt_tokens_details(
            request_metadata.final_usage_info, state["cached_tokens"]
        )
    return response


def apply() -> None:
    serving_cls = target.OpenAIServingChat
    if getattr(serving_cls, "_cmss_cached_tokens_patched", False):
        return

    serving_cls._cmss_original_chat_stream_generator = (
        serving_cls.chat_completion_stream_generator
    )
    serving_cls._cmss_original_chat_full_generator = (
        serving_cls.chat_completion_full_generator
    )
    serving_cls.chat_completion_stream_generator = _wrapped_stream_generator
    serving_cls.chat_completion_full_generator = _wrapped_full_generator
    serving_cls._cmss_cached_tokens_patched = True
    logger.info(
        "CMSS monkey patch 4026bfb0 applied to "
        "vllm.entrypoints.openai.chat_completion.serving (2 methods)."
    )
