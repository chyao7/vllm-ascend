"""Central entry point for all CMSS monkey-patches."""

from vllm_ascend.patch_cmss.vllm import apply as apply_vllm_patches
from vllm_ascend.patch_cmss.vllm_ascend import apply as apply_vllm_ascend_patches

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    apply_vllm_patches()
    apply_vllm_ascend_patches()
    _APPLIED = True


__all__ = ["apply"]
