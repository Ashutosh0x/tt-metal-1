// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw.hpp"

#include "device/cross_entropy_bw_device_operation.hpp"

namespace ttml::metal {

namespace {
// The kernel folds an upstream gradient of one value, or of one value per row, into its per-row scale.
bool can_fuse_upstream(const ttnn::Tensor& input, const ttnn::Tensor& grad) {
    if (grad.storage_type() != ttnn::StorageType::DEVICE || grad.layout() != ttnn::Layout::TILE ||
        (grad.dtype() != ttnn::DataType::BFLOAT16 && grad.dtype() != ttnn::DataType::FLOAT32) ||
        grad.memory_config() != ttnn::DRAM_MEMORY_CONFIG || grad.device() != input.device()) {
        return false;
    }
    if (grad.logical_shape() == ttnn::Shape({1, 1, 1, 1})) {
        return true;
    }
    auto per_row_shape = input.logical_shape();
    per_row_shape[-1] = 1U;
    return grad.logical_shape() == per_row_shape && grad.padded_shape()[-2] == input.padded_shape()[-2];
}
}  // namespace

ttnn::Tensor cross_entropy_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& target_tensor, const ttnn::Tensor& grad, float scaler) {
    if (can_fuse_upstream(input_tensor, grad)) {
        return ttnn::prim::ttml_cross_entropy_bw(input_tensor, target_tensor, scaler, std::nullopt, grad);
    }
    return ttnn::multiply(ttnn::prim::ttml_cross_entropy_bw(input_tensor, target_tensor, scaler), grad);
}

}  // namespace ttml::metal
