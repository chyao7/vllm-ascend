"""CMSS monkey-patches targeting vLLM."""

from vllm_ascend.patch_cmss.vllm.entrypoints.openai.chat_completion.serving import (
    apply as apply_chat_serving_patch,
)
from vllm_ascend.patch_cmss.vllm.entrypoints.openai.completion.serving import (
    apply as apply_completion_serving_patch,
)
from vllm_ascend.patch_cmss.vllm.v1.engine.output_processor import (
    apply as apply_output_processor_patch,
)

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    apply_output_processor_patch()
    apply_chat_serving_patch()
    apply_completion_serving_patch()
    _APPLIED = True


__all__ = ["apply"]
