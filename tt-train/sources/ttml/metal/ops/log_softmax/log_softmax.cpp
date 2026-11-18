// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "log_softmax.hpp"

#include <optional>
#include <variant>
#include <vector>

#include "metal/common/tensor_validation.hpp"
#include "row_softmax_program.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::log_softmax_device {
using row_softmax::layout_for;

struct LogSoftmaxAttributes {
    bool backward;
};
struct LogSoftmaxInputs {
    const ttnn::Tensor& input;
    std::optional<ttnn::Tensor> upstream;
    std::optional<ttnn::Tensor> state;
};
using Tensors = std::vector<ttnn::Tensor>;
using Specs = std::vector<tt::tt_metal::TensorSpec>;
struct LogSoftmaxFactory {
    struct shared_variables_t {
        row_softmax::Handles handles;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;
    static cached_program_t create(const LogSoftmaxAttributes&, const LogSoftmaxInputs&, Tensors&);
    static void override_runtime_arguments(
        cached_program_t&, const LogSoftmaxAttributes&, const LogSoftmaxInputs&, Tensors&);
};
struct LogSoftmaxOperation {
    using operation_attributes_t = LogSoftmaxAttributes;
    using tensor_args_t = LogSoftmaxInputs;
    using tensor_return_value_t = Tensors;
    using spec_return_value_t = Specs;
    using program_factory_t = std::variant<LogSoftmaxFactory>;
    static void validate_on_program_cache_miss(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        check_device_tensor(
            t.input,
            "LogSoftmax",
            "input",
            {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
        TT_FATAL(
            t.input.layout() == ttnn::Layout::TILE && t.input.logical_shape().rank() == 4,
            "Fused log_softmax expects rank-four TILE input");
        TT_FATAL(
            t.input.tensor_spec().tile().get_height() == 32 && t.input.tensor_spec().tile().get_width() == 32,
            "Fused log_softmax currently supports 32x32 tiles");
        TT_FATAL(t.input.memory_config() == ttnn::DRAM_MEMORY_CONFIG, "Fused log_softmax expects interleaved DRAM");
        TT_FATAL(
            t.input.logical_shape()[-1] > 0 && t.input.logical_shape()[-1] <= (1U << 24), "Invalid reduction width");
        if (a.backward) {
            TT_FATAL(t.upstream && t.state, "Fused log_softmax backward requires upstream and row state");
            check_device_tensor(
                *t.upstream,
                "LogSoftmax",
                "upstream",
                {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
            check_device_tensor(
                *t.state,
                "LogSoftmax",
                "state",
                {.dtypes = {ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
            TT_FATAL(
                t.upstream->logical_shape() == t.input.logical_shape() &&
                    t.upstream->padded_shape() == t.input.padded_shape() &&
                    t.upstream->memory_config() == t.input.memory_config() &&
                    t.upstream->layout() == t.input.layout() &&
                    t.upstream->tensor_spec().tile() == t.input.tensor_spec().tile(),
                "Upstream shape, layout and memory configuration must match input");
            TT_FATAL(
                t.upstream->device() == t.input.device() && t.state->device() == t.input.device(), "Device mismatch");
            TT_FATAL(
                t.state->dtype() == ttnn::DataType::FLOAT32 && t.state->layout() == ttnn::Layout::TILE,
                "Expected FP32 TILE row state");
            const uint32_t rows = t.input.physical_volume() / t.input.padded_shape()[-1];
            TT_FATAL(
                t.state->logical_shape() == ttnn::Shape({1, 1, rows, 96}) &&
                    t.state->memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
                    t.state->tensor_spec().tile() == t.input.tensor_spec().tile(),
                "Invalid fused log_softmax row state");
        }
    }
    static void validate_on_program_cache_hit(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        validate_on_program_cache_miss(a, t);
    }
    static Specs compute_output_specs(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        Specs result{t.input.tensor_spec()};
        const tt::tt_metal::TensorLayout fp32_layout(
            ttnn::DataType::FLOAT32, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
        const auto layout = layout_for(t.input);
        if (!a.backward) {
            result.emplace_back(ttnn::Shape({1, 1, layout.rows * 32, 96}), fp32_layout);
        }
        if (layout.chunks > 1) {
            // One page per (row, chunk) unit holding its packed partial row statistics.
            result.push_back(row_softmax::scratch_spec(layout));
        }
        return result;
    }
    static Tensors create_output_tensors(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        Tensors result;
        for (const auto& spec : compute_output_specs(a, t))
            result.push_back(ttnn::create_device_tensor(spec, t.input.device()));
        return result;
    }
    static ttsl::hash::hash_t compute_program_hash(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t) {
        return tt::tt_metal::operation::hash_operation<LogSoftmaxOperation>(
            a,
            t.input.dtype(),
            t.input.logical_shape(),
            t.input.padded_shape(),
            t.input.memory_config(),
            t.upstream ? t.upstream->dtype() : t.input.dtype());
    }
};
namespace {
row_softmax::Buffers buffers_for(const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    const size_t scratch_index = a.backward ? 1 : 2;
    return {
        .input = t.input.buffer(),
        .upstream = a.backward ? t.upstream->buffer() : nullptr,
        .state = a.backward ? t.state->buffer() : out[1].buffer(),
        .output = out[0].buffer(),
        .scratch = out.size() > scratch_index ? out[scratch_index].buffer() : nullptr};
}
}  // namespace

LogSoftmaxFactory::cached_program_t LogSoftmaxFactory::create(
    const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    auto program = tt::tt_metal::CreateProgram(*t.input.device());
    const row_softmax::Options options{
        .kind = a.backward ? row_softmax::Kind::LogSoftmaxBackward : row_softmax::Kind::LogSoftmaxForward,
        .upstream_dtype = a.backward ? t.upstream->dtype() : t.input.dtype()};
    const auto handles = row_softmax::create(program, options, t.input, buffers_for(a, t, out));
    return {std::move(program), {handles}};
}
void LogSoftmaxFactory::override_runtime_arguments(
    cached_program_t& cache, const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    row_softmax::override_addresses(cache.program, cache.shared_variables.handles, buffers_for(a, t, out));
}
}  // namespace ttml::metal::log_softmax_device
namespace ttml::metal {
std::vector<ttnn::Tensor> log_softmax_fw(const ttnn::Tensor& x) {
    using namespace log_softmax_device;
    auto result = ttnn::device_operation::launch<LogSoftmaxOperation>(
        LogSoftmaxAttributes{false}, LogSoftmaxInputs{x, std::nullopt, std::nullopt});
    result.resize(2);  // Drop the exchange scratch.
    return result;
}
ttnn::Tensor log_softmax_bw(const ttnn::Tensor& x, const ttnn::Tensor& dy, const ttnn::Tensor& state) {
    using namespace log_softmax_device;
    return ttnn::device_operation::launch<LogSoftmaxOperation>(
        LogSoftmaxAttributes{true}, LogSoftmaxInputs{x, dy, state})[0];
}
}  // namespace ttml::metal
