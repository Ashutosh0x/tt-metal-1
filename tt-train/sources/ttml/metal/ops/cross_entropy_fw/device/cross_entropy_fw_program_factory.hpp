// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "cross_entropy_fw_device_operation_types.hpp"
#include "metal/ops/log_softmax/row_softmax_program.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

// The collapse-resistant row-softmax program shared with the fused log_softmax (row_softmax_program.hpp).
struct CrossEntropyForwardProgramFactory {
    struct shared_variables_t {
        row_softmax::Handles handles;
        // DRAM exchange scratch when rows are split across cores; owned by the cached program.
        std::optional<ttnn::Tensor> scratch;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::cross_entropy_fw::device
