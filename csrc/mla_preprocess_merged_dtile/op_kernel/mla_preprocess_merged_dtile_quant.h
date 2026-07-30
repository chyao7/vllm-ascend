/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Per-tile symmetric int8 quant for merged Dtile K path (mla_prolog_v3 layout).
 * Matches mla_prolog QuantPerTile + PerTileClipWithAlpha + DynamicQuant:
 *   1) per-tile absmax
 *   2) clip to [-alpha*absmax, +alpha*absmax]
 *   3) scale = alpha * absmax / 127, quant = clip_value / scale
 */
#ifndef MLA_PREPROCESS_MERGED_DTILE_QUANT_H
#define MLA_PREPROCESS_MERGED_DTILE_QUANT_H

#include "kernel_operator.h"
#include "../../mla_preprocess/op_kernel/kernel/kernel_utils.h"
#include "mla_preprocess_merged_dtile.h"

namespace MLAPO_MERGED_DTILE {

constexpr uint32_t MERGED_QUANT_VEC_REPEAT_ELE = 64;
constexpr uint32_t MERGED_QUANT_VEC_REPEAT_BLOCK = 8;
constexpr float MERGED_QUANT_INV127 = 1.0f / 127.0f;

__aicore__ inline void MergedQuantTileAbsMax(const AscendC::LocalTensor<float> &dstLocal,
                                             const AscendC::LocalTensor<float> &workLocal,
                                             const AscendC::LocalTensor<float> &srcLocal, uint32_t count)
{
    const uint32_t repeat = count / MERGED_QUANT_VEC_REPEAT_ELE;
    const uint32_t tailNum = count % MERGED_QUANT_VEC_REPEAT_ELE;
    if (likely(repeat > 0)) {
        AscendC::WholeReduceMax(workLocal, srcLocal, MERGED_QUANT_VEC_REPEAT_ELE, repeat, 1, 1,
                                MERGED_QUANT_VEC_REPEAT_BLOCK, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
    }
    uint32_t reduceCount = repeat;
    if (unlikely(tailNum != 0)) {
        AscendC::WholeReduceMax(workLocal[repeat], srcLocal[count - tailNum], tailNum, 1, 1, 1,
                                MERGED_QUANT_VEC_REPEAT_BLOCK, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::PipeBarrier<PIPE_V>();
        reduceCount += 1;
    }
    AscendC::WholeReduceMax(dstLocal, workLocal, reduceCount, 1, 1, 1, MERGED_QUANT_VEC_REPEAT_BLOCK,
                            AscendC::ReduceOrder::ORDER_ONLY_VALUE);
}

__aicore__ inline void PerTileQuantKNormProlog(AscendC::LocalTensor<float> &rmsNormTensor,
                                               AscendC::LocalTensor<int8_t> &int8OutTensor,
                                               const AscendC::LocalTensor<uint8_t> &quantShareTmpUb, uint32_t kvDim,
                                               uint32_t tileSize, float clipAlpha,
                                               AscendC::GlobalTensor<int32_t> &debugTraceGm)
{
    const uint32_t numTiles = kvDim / tileSize;
    MergedDtileDebugTrace(debugTraceGm, MERGED_DTILE_DBG_QUANT_START, kvDim, tileSize);

    AscendC::LocalTensor<half> tmpfp16 =
        quantShareTmpUb[MERGED_SCRATCH_TMPFP16_BYTE_OFFSET].template ReinterpretCast<half>();
    AscendC::LocalTensor<float> absUb =
        quantShareTmpUb[MERGED_SCRATCH_QUANT_ABS_BYTE_OFFSET].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> aMax =
        quantShareTmpUb[MERGED_SCRATCH_QUANT_MAX_BYTE_OFFSET].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> reduceWork =
        quantShareTmpUb[MERGED_SCRATCH_QUANT_WORK_BYTE_OFFSET].template ReinterpretCast<float>();
    AscendC::LocalTensor<float> tileScalesOut =
        int8OutTensor[MERGED_DTILE_SCALE_BYTE_OFFSET].template ReinterpretCast<float>();

    for (uint32_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        const uint32_t offset = tileIdx * tileSize;
        MergedDtileDebugTrace(debugTraceGm, MERGED_DTILE_DBG_QUANT_TILE, tileIdx, offset);

        // PerTileClipWithAlpha: absmax per tile (before clip).
        AscendC::Abs(absUb, rmsNormTensor[offset], tileSize);
        AscendC::PipeBarrier<PIPE_V>();
        MergedQuantTileAbsMax(aMax, reduceWork, absUb, tileSize);
        AscendC::PipeBarrier<PIPE_V>();

        float absmax = aMax.GetValue(0);
        if (absmax < KV_QUANT_MIN_ABSMAX) {
            absmax = KV_QUANT_MIN_ABSMAX;
        }

        // Clip to [-alpha*absmax, +alpha*absmax].
        const float maxBound = clipAlpha * absmax;
        const float minBound = -clipAlpha * absmax;
        AscendC::Mins(rmsNormTensor[offset], rmsNormTensor[offset], maxBound, tileSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(rmsNormTensor[offset], rmsNormTensor[offset], minBound, tileSize);
        AscendC::PipeBarrier<PIPE_V>();

        // DynamicQuant: dequant scale = alpha * absmax / 127.
        const float descale = clipAlpha * absmax * MERGED_QUANT_INV127;
        tileScalesOut.SetValue(tileIdx, descale);
        const float invDescale = 1.0f / descale;
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID2);
        AscendC::Muls(rmsNormTensor[offset], rmsNormTensor[offset], invDescale, tileSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CastFrom32To16(tmpfp16, rmsNormTensor, kvDim);
    AscendC::PipeBarrier<PIPE_V>();
    CastFromF16ToI8(int8OutTensor, tmpfp16, static_cast<half>(-128), kvDim);
    AscendC::PipeBarrier<PIPE_V>();
    MergedDtileDebugTrace(debugTraceGm, MERGED_DTILE_DBG_QUANT_DONE, numTiles, 0);
}

}  // namespace MLAPO_MERGED_DTILE

#endif  // MLA_PREPROCESS_MERGED_DTILE_QUANT_H
