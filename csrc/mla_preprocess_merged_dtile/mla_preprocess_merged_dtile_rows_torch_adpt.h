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
#ifndef MLA_PREPROCESS_MERGED_DTILE_ROWS_TORCH_ADPT_H
#define MLA_PREPROCESS_MERGED_DTILE_ROWS_TORCH_ADPT_H

#include "../../ops.h"
#include "mla_preprocess_merged_dtile_torch_adpt.h"
#include "op_host/mla_preprocess_merged_dtile.h"

namespace vllm_ascend {

inline void CheckMergedDtileRowsBuffer(const at::Tensor &hiddenState, const at::Tensor &dtile_rows_out)
{
    constexpr int64_t kMergedDtileRowBytes = 656;
    const int64_t num_tokens = hiddenState.size(0);
    TORCH_CHECK(dtile_rows_out.scalar_type() == at::kChar || dtile_rows_out.scalar_type() == at::kByte,
                "dtile_rows_out must be int8 for merged_dtile rows");
    TORCH_CHECK(dtile_rows_out.dim() == 2, "dtile_rows_out must be [N, dtileSize], got dim=", dtile_rows_out.dim());
    TORCH_CHECK(dtile_rows_out.size(0) >= num_tokens,
                "dtile_rows_out rows must be >= num_tokens (", num_tokens, "), got ", dtile_rows_out.size(0));
    TORCH_CHECK(dtile_rows_out.size(1) == kMergedDtileRowBytes,
                "dtile_rows_out last dim must be ", kMergedDtileRowBytes, ", got ", dtile_rows_out.size(1));
    TORCH_CHECK(dtile_rows_out.is_contiguous(), "dtile_rows_out must be contiguous");
}

std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &> mla_preprocess_merged_dtile_rows(
    const at::Tensor &hiddenState, const at::Tensor &wdqkv, const c10::optional<at::Tensor> &descale0,
    const at::Tensor &gamma1, const c10::optional<at::Tensor> &beta1, const at::Tensor &wuq,
    const c10::optional<at::Tensor> &descale1, const at::Tensor &gamma2, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &wuk, const at::Tensor &dtile_rows_out, const at::Tensor &slotmapping,
    const c10::optional<at::Tensor> &quant_scale0, const c10::optional<at::Tensor> &quant_offset0,
    const c10::optional<at::Tensor> &bias0, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &quant_offset1, const c10::optional<at::Tensor> &bias1,
    const c10::optional<at::Tensor> &ctkv_scale, const c10::optional<at::Tensor> &q_nope_scale,
    const c10::optional<at::Tensor> &k_nope_clip_alpha, const c10::optional<at::Tensor> &debug_trace_out,
    at::Tensor &q_out0, at::Tensor &dtile_rows_out_ref, at::Tensor &q_out1, at::Tensor &inner_out)
{
    CheckMergedDtileRowsBuffer(hiddenState, dtile_rows_out);
    TORCH_CHECK(dtile_rows_out_ref.is_same(dtile_rows_out), "dtile_rows_out_ref must alias dtile_rows_out");

    at::Tensor Descale0 =
        descale0.has_value()
            ? descale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Descale1 =
        descale1.has_value()
            ? descale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Beta1 =
        beta1.has_value()
            ? beta1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale0 =
        quant_scale0.has_value()
            ? quant_scale0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_scale1 =
        quant_scale1.has_value()
            ? quant_scale1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset0 =
        quant_offset0.has_value()
            ? quant_offset0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Quant_offset1 =
        quant_offset1.has_value()
            ? quant_offset1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias0 =
        bias0.has_value()
            ? bias0.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor Bias1 =
        bias1.has_value()
            ? bias1.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor CtkvScale =
        ctkv_scale.has_value()
            ? ctkv_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor QnopeScale =
        q_nope_scale.has_value()
            ? q_nope_scale.value()
            : at::empty({1}, at::TensorOptions().dtype(at::kHalf).device(hiddenState.options().device()));
    at::Tensor KNopeClipAlpha =
        k_nope_clip_alpha.has_value()
            ? k_nope_clip_alpha.value()
            : at::full({1}, 1.0f, at::TensorOptions().dtype(at::kFloat).device(hiddenState.options().device()));

    auto [workspace_tensor, tiling, block_dim] =
        mlapo::mla_preprocess_merged_dtile_tiling(hiddenState, wdqkv, wuk, gamma1, cos);

    void *hidden_state_ptr = hiddenState.data_ptr();
    void *quant_scale0_ptr = Quant_scale0.data_ptr();
    void *quant_offset0_ptr = Quant_offset0.data_ptr();
    void *wdqkv_ptr = wdqkv.data_ptr();
    void *bias0_ptr = Bias0.data_ptr();
    void *gamma1_ptr = gamma1.data_ptr();
    void *beta1_ptr = Beta1.data_ptr();
    void *quant_scale1_ptr = Quant_scale1.data_ptr();
    void *quant_offset1_ptr = Quant_offset1.data_ptr();
    void *gamma2_ptr = gamma2.data_ptr();
    void *sin_ptr = sin.data_ptr();
    void *cos_ptr = cos.data_ptr();
    void *dtile_rows_ptr = dtile_rows_out.data_ptr();
    void *slotmapping_ptr = slotmapping.data_ptr();
    void *wuq_ptr = wuq.data_ptr();
    void *bias1_ptr = Bias1.data_ptr();
    void *wuk_ptr = wuk.data_ptr();
    void *descale0_ptr = Descale0.data_ptr();
    void *descale1_ptr = Descale1.data_ptr();
    void *ctkv_scale_ptr = CtkvScale.data_ptr();
    void *qnope_scale_ptr = QnopeScale.data_ptr();
    void *k_nope_clip_alpha_ptr = KNopeClipAlpha.data_ptr();
    void *q_out0_ptr = q_out0.data_ptr();
    void *dtile_rows_out_ptr = dtile_rows_out_ref.data_ptr();
    void *q_out1_ptr = q_out1.data_ptr();
    void *inner_out_ptr = inner_out.data_ptr();

    constexpr int64_t kMergedDtileRowBytes = 656;
    at::Tensor k_pe_scratch;
    if (debug_trace_out.has_value()) {
        k_pe_scratch = debug_trace_out.value();
        TORCH_CHECK(k_pe_scratch.scalar_type() == at::kInt,
                    "debug_trace_out must be int32 tensor for merged_dtile trace");
        TORCH_CHECK(k_pe_scratch.numel() >= 4, "debug_trace_out must have at least 4 int32 elements");
        if (MergedDtileDebugEnabled()) {
            k_pe_scratch.fill_(-1);
        }
    } else {
        k_pe_scratch = at::empty({kMergedDtileRowBytes}, hiddenState.options().dtype(at::kByte));
    }
    void *k_pe_scratch_ptr = k_pe_scratch.data_ptr();
    void *workspace_ptr = workspace_tensor.data_ptr();
    void *tiling_ptr = tiling.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("mla_preprocess_merged_dtile_rows");

    cmd.SetCustomHandler([stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr, gamma1_ptr,
                          beta1_ptr, quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, dtile_rows_ptr,
                          slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr,
                          qnope_scale_ptr, k_nope_clip_alpha_ptr, q_out0_ptr, dtile_rows_out_ptr, q_out1_ptr,
                          k_pe_scratch_ptr, inner_out_ptr, workspace_ptr, tiling_ptr, block_dim]() -> int {
        mla_preprocess_merged_dtile_rows_impl(
            stream, hidden_state_ptr, quant_scale0_ptr, quant_offset0_ptr, wdqkv_ptr, bias0_ptr, gamma1_ptr, beta1_ptr,
            quant_scale1_ptr, quant_offset1_ptr, gamma2_ptr, sin_ptr, cos_ptr, sin_ptr, cos_ptr, dtile_rows_ptr,
            slotmapping_ptr, wuq_ptr, bias1_ptr, wuk_ptr, descale0_ptr, descale1_ptr, ctkv_scale_ptr, qnope_scale_ptr,
            k_nope_clip_alpha_ptr, q_out0_ptr, dtile_rows_out_ptr, q_out1_ptr, k_pe_scratch_ptr, inner_out_ptr,
            workspace_ptr, tiling_ptr, block_dim);
        return 0;
    });
    cmd.Run();
    if (MergedDtileDebugEnabled()) {
        PrintMergedDtileDebugTrace(k_pe_scratch.reshape({-1}), "host_after_run_rows");
    }
    return std::forward_as_tuple(q_out0, dtile_rows_out_ref, q_out1, inner_out);
}

}  // namespace vllm_ascend

#endif
