/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MLA_PREPROCESS_MERGED_DTILE_H
#define MLA_PREPROCESS_MERGED_DTILE_H

#include "../../mla_preprocess/op_kernel/mla_preprocess.h"

constexpr uint8_t CACHE_MODE_MERGED_DTILE = 4;
// Build flag: merged Dtile K quant/scatter/UB (orthogonal to CACHE_MODE template param).
#ifndef MERGED_DTILE_BUILD
#define MERGED_DTILE_BUILD 1
#endif
// Per-tile quant tile size (mla_prolog_v3 tileSize=128).
constexpr uint32_t KV_PER_TILE_QUANT_SIZE = 128;
// GM dtile layout (ckvkr_repo_mode=1, quant_scale_repo_mode=1), stride = MERGED_DTILE_BYTES:
//   [0, headSizeCkv)                         int8 k_nope
//   [headSizeCkv, headSizeCkv + dr*2)        bf16 k_pe
//   [headSizeCkv + dr*2, dtileSize)          float per-tile descale (headSizeCkv/128)
constexpr uint32_t MERGED_DTILE_KNOPE_BYTES = SPLIT_RMSNRORM_SIZE_ONE;
constexpr uint32_t MERGED_DTILE_KPE_BYTE_LEN = SPLIT_RMSNRORM_SIZE_TWO * 2;
constexpr uint32_t MERGED_DTILE_SCALE_FLOATS = MERGED_DTILE_KNOPE_BYTES / KV_PER_TILE_QUANT_SIZE;
constexpr uint32_t MERGED_DTILE_SCALE_BYTES = MERGED_DTILE_SCALE_FLOATS * sizeof(float);
constexpr uint32_t MERGED_DTILE_KPE_BYTE_OFFSET = MERGED_DTILE_KNOPE_BYTES;
constexpr uint32_t MERGED_DTILE_SCALE_BYTE_OFFSET = MERGED_DTILE_KNOPE_BYTES + MERGED_DTILE_KPE_BYTE_LEN;
constexpr uint32_t MERGED_DTILE_BYTES = MERGED_DTILE_SCALE_BYTE_OFFSET + MERGED_DTILE_SCALE_BYTES;
// Legacy aliases used by existing scatter paths.
constexpr uint32_t MERGED_DTILE_PAYLOAD_BYTES = MERGED_DTILE_KPE_BYTE_OFFSET + MERGED_DTILE_KPE_BYTE_LEN;

// Per-tile quant scratch bytes are relative to quantShareTmpUb (placed after merged-row UB in ProcessVector).
constexpr uint32_t MERGED_SCRATCH_TMPFP16_BYTES = 1024;
constexpr uint32_t MERGED_SCRATCH_TMPFP16_BYTE_OFFSET = 0;
constexpr uint32_t MERGED_SCRATCH_QUANT_ABS_BYTE_OFFSET = MERGED_SCRATCH_TMPFP16_BYTES;
constexpr uint32_t MERGED_SCRATCH_QUANT_ABS_BYTES = SPLIT_RMSNRORM_SIZE_ONE * static_cast<uint32_t>(sizeof(float));
constexpr uint32_t MERGED_SCRATCH_QUANT_MAX_BYTE_OFFSET =
    MERGED_SCRATCH_QUANT_ABS_BYTE_OFFSET + MERGED_SCRATCH_QUANT_ABS_BYTES;
constexpr uint32_t MERGED_SCRATCH_QUANT_MAX_BYTES = 32;
constexpr uint32_t MERGED_SCRATCH_QUANT_WORK_BYTE_OFFSET =
    MERGED_SCRATCH_QUANT_MAX_BYTE_OFFSET + MERGED_SCRATCH_QUANT_MAX_BYTES;
constexpr uint32_t MERGED_SCRATCH_QUANT_WORK_BYTES = 512;
constexpr uint32_t MERGED_DTILE_ALIGN32 = 32;
constexpr uint32_t MERGED_PER_TILE_UB_SCRATCH_BYTES =
    ((MERGED_SCRATCH_QUANT_WORK_BYTE_OFFSET + MERGED_SCRATCH_QUANT_WORK_BYTES + MERGED_DTILE_ALIGN32 - 1) /
     MERGED_DTILE_ALIGN32) *
    MERGED_DTILE_ALIGN32;

// Bisect 507015: set to 0 in this header, rebuild, and re-run benchmark.
#ifndef MERGED_DTILE_ENABLE_PER_TILE_QUANT
#define MERGED_DTILE_ENABLE_PER_TILE_QUANT 1
#endif
#ifndef MERGED_DTILE_ENABLE_K_RMSROPE
#define MERGED_DTILE_ENABLE_K_RMSROPE 1
#endif
#ifndef MERGED_DTILE_ENABLE_KV_SCATTER
#define MERGED_DTILE_ENABLE_KV_SCATTER 1
#endif
#ifndef MERGED_DTILE_ENABLE_ROW_BUFFER
#define MERGED_DTILE_ENABLE_ROW_BUFFER 0
#endif
#if MERGED_DTILE_ENABLE_KV_SCATTER && MERGED_DTILE_ENABLE_ROW_BUFFER
#error "MERGED_DTILE_ENABLE_KV_SCATTER and MERGED_DTILE_ENABLE_ROW_BUFFER are mutually exclusive"
#endif
constexpr float KV_QUANT_MIN_ABSMAX = 1e-8f;

// GM trace via keycacheOutGm2 scratch (int32[0..3]). Set 0 to disable device writes.
#ifndef MERGED_DTILE_DEBUG_TRACE
#define MERGED_DTILE_DEBUG_TRACE 1
#endif
constexpr uint32_t MERGED_DTILE_DEBUG_TRACE_WORDS = 4;
// stage word0 | aux0 word1 | aux1 word2 | block word3
constexpr uint32_t MERGED_DTILE_DBG_PROCESS_VECTOR_K = 100;
constexpr uint32_t MERGED_DTILE_DBG_BEFORE_K_RMSROPE = 110;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_K_RMSROPE = 120;
constexpr uint32_t MERGED_DTILE_DBG_BEFORE_MM2_AIV = 130;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_MM2_AIV = 131;
constexpr uint32_t MERGED_DTILE_DBG_K_LOOP = 200;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_K_RMS = 220;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_K_QUANT = 230;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_K_ROPE = 240;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_K_SCATTER = 250;
constexpr uint32_t MERGED_DTILE_DBG_AFTER_K_ROW_WRITE = 251;
constexpr uint32_t MERGED_DTILE_DBG_QUANT_START = 400;
constexpr uint32_t MERGED_DTILE_DBG_QUANT_TILE = 401;
constexpr uint32_t MERGED_DTILE_DBG_QUANT_DONE = 409;

__aicore__ inline void MergedDtileDebugTrace(AscendC::GlobalTensor<int32_t> &traceGm, uint32_t stage,
                                             uint32_t aux0 = 0, uint32_t aux1 = 0)
{
#if MERGED_DTILE_DEBUG_TRACE
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
#ifdef __DAV_C220_VEC__
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
#endif
    traceGm.SetValue(0, static_cast<int32_t>(stage));
    traceGm.SetValue(1, static_cast<int32_t>(aux0));
    traceGm.SetValue(2, static_cast<int32_t>(aux1));
    traceGm.SetValue(3, static_cast<int32_t>(AscendC::GetBlockIdx()));
#endif
}

constexpr uint32_t KEY_BF16_MERGED_DTILE_INNER = 260 + 512;

#endif
