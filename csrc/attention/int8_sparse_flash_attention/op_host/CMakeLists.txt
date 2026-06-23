# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

function(__int8_sfa_sanitize_stale_aicpu_cust_targets)
    if (NOT DEFINED AICPU_CUST_OBJ_TARGETS OR AICPU_CUST_OBJ_TARGETS STREQUAL "")
        return()
    endif()
    set(_filtered "")
    foreach (_tgt IN LISTS AICPU_CUST_OBJ_TARGETS)
        if (TARGET "${_tgt}")
            list(APPEND _filtered "${_tgt}")
        endif()
    endforeach()
    if (NOT _filtered STREQUAL AICPU_CUST_OBJ_TARGETS)
        set(AICPU_CUST_OBJ_TARGETS "${_filtered}" CACHE INTERNAL
            "All aicpu cust obj targets" FORCE)
        message(STATUS "int8_sparse_flash_attention: pruned stale AICPU_CUST_OBJ_TARGETS")
    endif()
endfunction()

function(__int8_sfa_verify_kernel_headers)
    set(_arch22 "${CMAKE_CURRENT_LIST_DIR}/../op_kernel/arch22")
    set(_cube "${_arch22}/int8_sparse_flash_attention_service_cube_mla.h")
    set(_kernel "${_arch22}/int8_sparse_flash_attention_kernel_mla.h")
    set(_expected_cube_lines 1121)
    set(_expected_kernel_lines 1026)

    foreach (_hdr IN ITEMS "${_cube}" "${_kernel}")
        if (NOT EXISTS "${_hdr}")
            message(FATAL_ERROR "int8_sparse_flash_attention: missing kernel header ${_hdr}")
        endif()
        file(READ "${_hdr}" _content)
        if (_content MATCHES "INT8_SFA_CUBE_MLA_HEADER_MARKER")
            message(FATAL_ERROR
                "int8_sparse_flash_attention: stale header ${_hdr} still contains "
                "INT8_SFA_CUBE_MLA_HEADER_MARKER. Sync op_kernel/arch22/ before building.")
        endif()
        if (_content MATCHES "#error[^\n]*\\\\")
            message(FATAL_ERROR
                "int8_sparse_flash_attention: stale header ${_hdr} still contains multiline "
                "#error checks. Sync op_kernel/arch22/ before building.")
        endif()
    endforeach()

    execute_process(COMMAND wc -l "${_cube}" "${_kernel}"
        OUTPUT_VARIABLE _wc_output
        RESULT_VARIABLE _wc_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _wc_result EQUAL 0)
        message(FATAL_ERROR "int8_sparse_flash_attention: failed to run wc -l on kernel headers")
    endif()
    string(REPLACE "\n" ";" _wc_lines "${_wc_output}")
    foreach (_wc_line IN LISTS _wc_lines)
        if (_wc_line MATCHES "^[ \t]*([0-9]+)[ \t]+(.+)$")
            set(_line_count "${CMAKE_MATCH_1}")
            set(_path "${CMAKE_MATCH_2}")
            if (_path MATCHES "service_cube_mla\\.h$")
                set(_expected "${_expected_cube_lines}")
            elseif (_path MATCHES "kernel_mla\\.h$")
                set(_expected "${_expected_kernel_lines}")
            else()
                continue()
            endif()
            if (NOT _line_count EQUAL _expected)
                math(EXPR _fail_line "${_line_count} + 1")
                message(FATAL_ERROR
                    "int8_sparse_flash_attention: ${_path} has ${_line_count} lines, expected "
                    "${_expected}. Stale headers cause AscendC error 'unknown type name s' near line "
                    "${_fail_line}. Sync csrc/attention/int8_sparse_flash_attention/ to the build "
                    "host, then rm -rf csrc/build and rebuild.")
            endif()
        endif()
    endforeach()
    message(STATUS
        "int8_sparse_flash_attention: kernel headers OK "
        "(cube=${_expected_cube_lines}, kernel=${_expected_kernel_lines})")
endfunction()

function(__int8_sfa_register_src_copy_fixup)
    if (NOT DEFINED ASCEND_COMPUTE_UNIT)
        return()
    endif()
    foreach (_compute_unit ${ASCEND_COMPUTE_UNIT})
        string(STRIP "${_compute_unit}" _cu)
        if (_cu STREQUAL "")
            continue()
        endif()
        set(_src_copy_target "int8_sparse_flash_attention_${_cu}_src_copy")
        if (NOT TARGET ${_src_copy_target})
            continue()
        endif()
        if (DEFINED ASCEND_BINARY_OUT_DIR)
            set(_dst "${ASCEND_BINARY_OUT_DIR}/${_cu}/src/int8_sparse_flash_attention")
        else()
            set(_dst "${CMAKE_BINARY_DIR}/binary/${_cu}/src/int8_sparse_flash_attention")
        endif()
        set(_fixup_target "int8_sparse_flash_attention_${_cu}_pre_src_copy")
        if (NOT TARGET ${_fixup_target})
            add_custom_target(${_fixup_target}
                COMMAND ${CMAKE_COMMAND} -E rm -rf "${_dst}"
                COMMENT "Remove stale int8_sparse_flash_attention build src tree before src copy"
            )
        endif()
        add_dependencies(${_src_copy_target} ${_fixup_target})
        __int8_sfa_invalidate_build_cache("${_cu}" "${_dst}")
    endforeach()
endfunction()

function(__int8_sfa_invalidate_build_cache _cu _src_dst)
    set(_src_copy_done "${_src_dst}/int8_sparse_flash_attention_${_cu}_src_copy.done")
    if (EXISTS "${_src_copy_done}")
        file(REMOVE "${_src_copy_done}")
        message(STATUS "int8_sparse_flash_attention: removed stale ${_src_copy_done}")
    endif()
    if (DEFINED ASCEND_BINARY_OUT_DIR)
        set(_gen_dir "${ASCEND_BINARY_OUT_DIR}/${_cu}/gen")
    else()
        set(_gen_dir "${CMAKE_BINARY_DIR}/binary/${_cu}/gen")
    endif()
    if (NOT IS_DIRECTORY "${_gen_dir}")
        return()
    endif()
    file(GLOB _kernel_gen_dones "${_gen_dir}/int8_sparse_flash_attention_*.done")
    foreach (_done IN LISTS _kernel_gen_dones)
        file(REMOVE "${_done}")
        message(STATUS "int8_sparse_flash_attention: removed stale ${_done}")
    endforeach()
    file(GLOB _kernel_meta_dirs "${_gen_dir}/kernel_meta_Int8SparseFlashAttention_*")
    foreach (_meta_dir IN LISTS _kernel_meta_dirs)
        file(REMOVE_RECURSE "${_meta_dir}")
        message(STATUS "int8_sparse_flash_attention: removed stale ${_meta_dir}")
    endforeach()
endfunction()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/int8_sparse_flash_attention_def.cpp
    )
endif()

# Incremental builds may leave op_kernel as a symlink under build/binary/.../src/.
# add_ops_src_copy runs cp before rm, which fails when source and destination are the same inode.
if (BUILD_OPEN_PROJECT AND NOT BUILD_OPS_RTY_KERNEL)
    __int8_sfa_verify_kernel_headers()
    if (NOT DEFINED _INT8_SFA_SRC_COPY_FIXUP_DEFERRED)
        set(_INT8_SFA_SRC_COPY_FIXUP_DEFERRED TRUE CACHE INTERNAL "")
        cmake_language(DEFER CALL __int8_sfa_register_src_copy_fixup)
    endif()
    # AICPU_CUST_OBJ_TARGETS is CACHE INTERNAL and survives incremental configures.
    # gen_cust_aicpu_kernel_symbol() runs after custom_build; prune stale entries before it.
    if (NOT DEFINED _INT8_SFA_AICPU_CACHE_SANITIZE_DEFERRED)
        set(_INT8_SFA_AICPU_CACHE_SANITIZE_DEFERRED TRUE CACHE INTERNAL "")
        cmake_language(DEFER DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            CALL __int8_sfa_sanitize_stale_aicpu_cust_targets)
    endif()
endif()

add_ops_compile_options(
    OP_NAME Int8SparseFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE int8_sparse_flash_attention ACLNNTYPE aclnn_inner)
endif()
