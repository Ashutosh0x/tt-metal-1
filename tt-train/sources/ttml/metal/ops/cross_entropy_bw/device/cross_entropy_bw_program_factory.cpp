// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cross_entropy_bw_program_factory.hpp"

namespace ttml::metal::ops::cross_entropy_bw::device {

namespace {
row_softmax::Buffers buffers_for(
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& output,
    const std::optional<ttnn::Tensor>& scratch) {
    return {
        .input = tensor_args.input.buffer(),
        .upstream = tensor_args.upstream.has_value() ? tensor_args.upstream->buffer() : nullptr,
        .target = tensor_args.target.buffer(),
        .output = output.buffer(),
        .scratch = scratch.has_value() ? scratch->buffer() : nullptr};
}
}  // namespace

CrossEntropyBackwardProgramFactory::cached_program_t CrossEntropyBackwardProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto layout = row_softmax::layout_for(input);
    std::optional<ttnn::Tensor> scratch;
    if (layout.chunks > 1) {
        scratch = ttnn::create_device_tensor(row_softmax::scratch_spec(layout), input.device());
    }
    // One row-major target page per batch-channel slice, 32 targets per tile row of the input.
    const auto& target = tensor_args.target;
    row_softmax::Options options{.kind = row_softmax::Kind::CrossEntropyBackward};
    options.scaler = operation_attributes.scaler;
    options.upstream_kind = upstream_kind(tensor_args);
    if (tensor_args.upstream.has_value()) {
        options.upstream_dtype = tensor_args.upstream->dtype();
    }
    options.target_page_bytes = target.logical_shape()[-1] * target.element_size();
    options.target_tiled_height = input.padded_shape()[-2] / tt::constants::TILE_HEIGHT;

    auto program = tt::tt_metal::CreateProgram(*input.device());
    const auto handles = row_softmax::create(program, options, input, buffers_for(tensor_args, output, scratch));
    return {std::move(program), {handles, std::move(scratch)}};
}

void CrossEntropyBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared = cached_program.shared_variables;
    row_softmax::override_addresses(
        cached_program.program, shared.handles, buffers_for(tensor_args, tensor_return_value, shared.scratch));
}

}  // namespace ttml::metal::ops::cross_entropy_bw::device
