#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify int8 SFA Python binding and CANN custom-op install (no rg required)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check_torch_binding() -> bool:
    print("== Torch binding (vllm_ascend.vllm_ascend_C / torch.ops._C_ascend) ==")
    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: import torch: {exc}")
        return False

    try:
        import vllm_ascend.vllm_ascend_C as ext  # noqa: F401
    except ImportError as exc:
        print(f"FAIL: import vllm_ascend.vllm_ascend_C: {exc}")
        print("  Hint: pip install -e . --no-build-isolation  (COMPILE_CUSTOM_KERNELS=1)")
        return False

    ext_path = Path(ext.__file__).resolve()
    print(f"  extension .so : {ext_path}")
    print(f"  mtime         : {ext_path.stat().st_mtime:.0f}")

    has_op = hasattr(torch.ops._C_ascend, "npu_int8_sparse_flash_attention")
    print(f"  npu_int8_sparse_flash_attention registered: {has_op}")
    if not has_op:
        print("  FAIL: op not in torch.ops._C_ascend")
        return False
    return True


def _check_cann_custom_op(root: Path) -> bool:
    print()
    print("== CANN custom op (actual int8 kernel lives here, NOT in vllm_ascend_C) ==")
    vendor_root = root / "vllm_ascend" / "_cann_ops_custom"
    print(f"  install dir: {vendor_root}")

    if not vendor_root.is_dir():
        print("  FAIL: directory missing")
        print("  Hint: bash csrc/build_aclnn.sh $(pwd) ascend910b")
        return False

    entries = list(vendor_root.iterdir())
    if len(entries) <= 1 and all(p.name == ".gitkeep" for p in entries):
        print("  FAIL: only .gitkeep present; custom op not installed")
        print("  Hint: bash csrc/build_aclnn.sh $(pwd) ascend910b")
        return False

    # Search for Int8SparseFlashAttention artifacts
    hits: list[Path] = []
    for path in vendor_root.rglob("*"):
        if path.is_file() and "Int8Sparse" in path.name:
            hits.append(path)

    if hits:
        print(f"  Int8Sparse artifacts ({len(hits)}):")
        for path in sorted(hits)[:20]:
            print(f"    {path}  mtime={path.stat().st_mtime:.0f}")
        if len(hits) > 20:
            print(f"    ... and {len(hits) - 20} more")
    else:
        print("  WARN: no filename containing 'Int8Sparse' under install dir")
        so_files = list(vendor_root.rglob("*.so"))
        print(f"  .so files found: {len(so_files)}")
        for path in sorted(so_files)[:10]:
            print(f"    {path}")

    vector_src = (
        root
        / "csrc/attention/int8_sparse_flash_attention/op_kernel/arch22"
        / "int8_sparse_flash_attention_service_vector_mla.h"
    )
    if vector_src.is_file():
        src_mtime = vector_src.stat().st_mtime
        print(f"  vector_mla.h mtime: {src_mtime:.0f}  ({vector_src})")
        newest = max((p.stat().st_mtime for p in vendor_root.rglob("*") if p.is_file()), default=0.0)
        if newest and newest < src_mtime:
            print("  FAIL: install artifacts older than vector_mla.h — rebuild CANN op")
            return False
        if newest:
            print(f"  newest install artifact mtime: {newest:.0f}")
    return True


def _check_env() -> None:
    print()
    print("== Runtime env ==")
    for key in ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH", "ASCEND_HOME_PATH"):
        val = os.environ.get(key, "<unset>")
        print(f"  {key}: {val}")


def main() -> int:
    root = _repo_root()
    print(f"repo: {root}")
    ok_binding = _check_torch_binding()
    ok_cann = _check_cann_custom_op(root)
    _check_env()
    print()
    if ok_binding and ok_cann:
        print("OK: binding + CANN install look present. Run benchmark with --debug next.")
        return 0
    print("Some checks failed. Rebuild:")
    print("  bash csrc/build_aclnn.sh $(pwd) ascend910b")
    print("  pip install -e . --no-build-isolation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
