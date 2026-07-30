// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost.git
//   https://gitee.com/ascend/op-plugin.git
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#ifndef MLA_PREPROCESS_MERGED_DTILE_HOST_H
#define MLA_PREPROCESS_MERGED_DTILE_HOST_H

// Original mla_preprocess.h has no include guard. Skip re-include when the TU
// already pulled it in (torch_binding.cpp defines this before including us).
#ifndef VLLM_ASCEND_MLA_PREPROCESS_HOST_INCLUDED
#define VLLM_ASCEND_MLA_PREPROCESS_HOST_INCLUDED
#include "../../mla_preprocess/op_host/mla_preprocess.h"
#endif

namespace mlapo {

// Graph-safe tiling: stable GM pointer per N via slot bIndex=N-1 (same pattern as mla_preprocess).
// Covers graph padding batches up to 32k tokens (align with large max_num_batched_tokens sweeps).
constexpr uint32_t MERGED_DTILE_MAX_SUPPORT_TOKEN_NUMS = 32768;

std::tuple<at::Tensor, at::Tensor, uint32_t> mla_preprocess_merged_dtile_tiling(
    const at::Tensor &hiddenState,
    const at::Tensor &wdqkv,
    const at::Tensor &wuk,
    const at::Tensor &gamma1,
    const at::Tensor &cos)
{
    constexpr int cacheMode = 1;  // CACHE_MODE_KROPE_CTKV — same mm/rope path as mla_preprocess inner
    constexpr QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
    constexpr bool enableInnerOut = true;

    platform_ascendc::PlatformAscendC *platformAscendC = platform_ascendc::PlatformAscendCManager::GetInstance();

    struct PlatformInfo platformInfo;
    platformInfo.coreNum = platformAscendC->GetCoreNum();
    platformInfo.coreNumAic = platformAscendC->GetCoreNumAic();
    platformInfo.coreNumAiv = platformAscendC->GetCoreNumAiv();
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo.l2Size);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo.l0aSize);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo.l0bSize);
    platformAscendC->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo.l0cSize);

    int32_t N = hiddenState.sizes()[0];
    int32_t headNum = wuk.sizes()[0];
    uint32_t hiddenStateDim = hiddenState.sizes().back();

    uint32_t qkNopeHeadDim = wuk.sizes()[1];
    uint32_t kvLoraRank = wuk.sizes()[2];
    uint32_t qLoraRank = gamma1.sizes()[0];
    uint32_t qkRopeHeadDim = cos.sizes().back();

    OpParam opParam;
    opParam.hiddenStateDim = hiddenStateDim;
    opParam.N = N;
    opParam.headNum = headNum;
    opParam.cacheMode = cacheMode;
    opParam.quantMode = quantMode;
    opParam.inDtype = hiddenState.options().dtype();
    opParam.enableInnerOut = enableInnerOut;
    opParam.qLoraRank = qLoraRank;
    opParam.qkNopeHeadDim = qkNopeHeadDim;
    opParam.qkRopeHeadDim = qkRopeHeadDim;
    opParam.kvLoraRank = kvLoraRank;
    if (wdqkv.options().dtype() == at::kBFloat16 || wdqkv.options().dtype() == at::kHalf) {
        opParam.isWeightQuantized = 0;
    } else {
        opParam.isWeightQuantized = 1;
    }

    MlaTilingData tilingData;
    MlaPreprocessTiling mlaTiling(platformInfo, opParam, &tilingData);

    mlaTiling.Init();
    uint32_t blockDim = platformInfo.coreNumAic;

    uint64_t system_workspace_size = static_cast<uint64_t>(platformAscendC->GetLibApiWorkSpaceSize());
    uint64_t workspace_size = system_workspace_size + tilingData.userWorkspaceSize;
    auto options = at::TensorOptions().dtype(at::kByte).device(hiddenState.options().device());
    auto workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options);

    // Tiling: static device pool indexed by N (prolog_v3-style stable address, dynamic content).
    int32_t bIndex = N - 1;
    uint32_t tilingSize = sizeof(MlaTilingData);
    static auto global_tiling_data = at::empty(
        {static_cast<int64_t>(tilingSize * MERGED_DTILE_MAX_SUPPORT_TOKEN_NUMS)},
        at::TensorOptions().dtype(at::kByte).device(hiddenState.options().device()));
    TORCH_CHECK(
        bIndex >= 0 && bIndex < static_cast<int32_t>(MERGED_DTILE_MAX_SUPPORT_TOKEN_NUMS),
        "merged_dtile bIndex is out of range: ",
        bIndex,
        " (N=",
        N,
        ", max=",
        MERGED_DTILE_MAX_SUPPORT_TOKEN_NUMS,
        ")");
    aclrtMemcpy(
        global_tiling_data.data_ptr<uint8_t>() + (tilingSize * bIndex),
        tilingSize,
        &tilingData,
        tilingSize,
        ACL_MEMCPY_HOST_TO_DEVICE);
    at::Tensor tiling = at::from_blob(
        global_tiling_data.data_ptr<uint8_t>() + (tilingSize * bIndex),
        tilingSize,
        at::kByte);

    return std::make_tuple(workspace_tensor, tiling, blockDim);
}

}  // namespace mlapo

#endif  // MLA_PREPROCESS_MERGED_DTILE_HOST_H
