// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <optional>

#include "autograd/tensor.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core {

// Stores every trainable bf16 or fp32 parameter in dtype (BFLOAT16 or FLOAT32), cast from its current value. Frozen
// parameters and non-float tensors keep their dtype. With FLOAT32, fused AdamW keeps the fp32 master weights in the
// parameters themselves, and forward and backward run on their bf16 view.
//
// Load order: build the model, convert, then load weights and checkpoints, then create the optimizer. Every loader
// (checkpoints through assign(), the C++ safetensors loaders through serialization::assign_from_floats()) writes the
// dtype a tensor is stored in, so a value loaded after the conversion keeps its fp32 bits, while one loaded before it
// is rounded to bf16 and the conversion only widens it. The optimizer state follows the dtype it finds.
void store_trainable_parameters_as(const serialization::NamedParameters& parameters, ttnn::DataType dtype);

// Applies training_config.precision to the parameters: stores them in precision.master_dtype ("bfloat16" or
// "float32") with store_trainable_parameters_as(), whose contract applies. The block must be a map of known keys, and
// any other key or name is a TT_FATAL. Unset, or set to null, leaves the parameters in the dtype the model builds them
// in. Returns the dtype it applied, if any.
std::optional<ttnn::DataType> apply_precision_config(
    const YAML::Node& training_config, const serialization::NamedParameters& parameters);

}  // namespace ttml::core
