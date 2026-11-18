// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vocab_parallel_cross_entropy.hpp"

#include <limits>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

#include "metal/common/tensor_validation.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "row_softmax_program.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttml::metal::vocab_parallel_ce_device {

enum class Mode : uint8_t { Stats, Combine, Backward };

struct Attributes {
    Mode mode;
    std::optional<uint32_t> cluster_axis;
    float scaler = 1.F;  // Backward only.
};
struct Inputs {
    const ttnn::Tensor& input;  // Logits (stats, backward) or the gathered shard records (combine).
    std::optional<ttnn::Tensor> target;
    std::optional<ttnn::Tensor> state;
    std::optional<ttnn::Tensor> upstream;
};
using Tensors = std::vector<ttnn::Tensor>;
using Specs = std::vector<tt::tt_metal::TensorSpec>;

constexpr uint32_t kRecordWidth = 32;  // One FP32 tile column block per shard record.
constexpr uint32_t kStateWidth = 64;   // Two column-vector tiles (M, l) per tile row.
constexpr uint32_t kMaxShards = 32;    // Shards are combined as the 32 lanes of a tile.

row_softmax::Kind kind_of(Mode mode) {
    switch (mode) {
        case Mode::Stats: return row_softmax::Kind::VocabParallelStats;
        case Mode::Combine: return row_softmax::Kind::VocabParallelCombine;
        case Mode::Backward: return row_softmax::Kind::VocabParallelBackward;
    }
    TT_THROW("Unknown vocabulary-parallel cross-entropy mode");
}

// 0: no upstream gradient, 1: one value for every row, 2: one value per row.
uint32_t upstream_kind(const Inputs& t) {
    if (!t.upstream.has_value()) {
        return 0;
    }
    return t.upstream->logical_volume() == 1 ? 1 : 2;
}

ttnn::Shape with_width(const ttnn::Tensor& tensor, uint32_t width) {
    auto shape = tensor.logical_shape();
    shape[-1] = width;
    return shape;
}

void check_tiled_dram(const ttnn::Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.layout() == ttnn::Layout::TILE && tensor.logical_shape().rank() == 4 &&
            tensor.memory_config() == ttnn::DRAM_MEMORY_CONFIG && tensor.tensor_spec().tile().get_height() == 32 &&
            tensor.tensor_spec().tile().get_width() == 32,
        "VocabParallelCrossEntropy: {} must be a rank-4 interleaved DRAM tensor of 32x32 tiles",
        name);
}

// The reader walks one row-major target page per batch-channel slice of the logits; pin the target's page
// width and page count to the logits (see CrossEntropyBackward).
void check_target(const ttnn::Tensor& logits, const ttnn::Tensor& target) {
    check_device_tensor(
        target,
        "VocabParallelCrossEntropy",
        "target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});
    const auto& target_shape = target.logical_shape();
    TT_FATAL(
        target_shape[-1] == logits.logical_shape()[-2],
        "VocabParallelCrossEntropy: target inner dim ({}) must equal the logits sequence dim ({})",
        target_shape[-1],
        logits.logical_shape()[-2]);
    const auto& padded = logits.padded_shape();
    const uint64_t nc_pages = padded.volume() / (static_cast<uint64_t>(padded[-2]) * padded[-1]);
    TT_FATAL(
        target_shape.volume() / target_shape[-1] == nc_pages,
        "VocabParallelCrossEntropy: target must supply one page per logits batch-channel slice");
}

struct Factory {
    struct shared_variables_t {
        row_softmax::Handles handles;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    static cached_mesh_workload_t create_mesh_workload(
        const Attributes&, const ttnn::MeshCoordinateRangeSet&, const Inputs&, Tensors&);
    static void override_runtime_arguments(cached_mesh_workload_t&, const Attributes&, const Inputs&, Tensors&);
};

struct Operation {
    using operation_attributes_t = Attributes;
    using tensor_args_t = Inputs;
    using tensor_return_value_t = Tensors;
    using spec_return_value_t = Specs;
    using program_factory_t = std::variant<Factory>;

    static void validate_on_program_cache_miss(const Attributes& a, const Inputs& t) {
        const auto& input = t.input;
        check_device_tensor(
            input,
            "VocabParallelCrossEntropy",
            "input",
            {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
        check_tiled_dram(input, "input");
        auto* device = input.device();
        TT_FATAL(device != nullptr, "VocabParallelCrossEntropy: input must be on a (mesh) device");
        if (a.cluster_axis.has_value()) {
            TT_FATAL(
                *a.cluster_axis < device->shape().dims(),
                "VocabParallelCrossEntropy: cluster_axis {} is out of range for the mesh",
                *a.cluster_axis);
        }
        const uint32_t width = input.logical_shape()[-1];
        if (a.mode == Mode::Combine) {
            TT_FATAL(
                input.dtype() == ttnn::DataType::FLOAT32 && width % kRecordWidth == 0 && width > 0 &&
                    width / kRecordWidth <= kMaxShards,
                "VocabParallelCrossEntropy: combine expects FP32 records of {} columns per shard, at most {} shards",
                kRecordWidth,
                kMaxShards);
            return;
        }
        TT_FATAL(t.target.has_value(), "VocabParallelCrossEntropy: target required");
        check_target(input, *t.target);
        // Global column indices stay exact FP32 integers.
        const uint64_t shards = a.cluster_axis.has_value() ? device->shape()[*a.cluster_axis] : device->num_devices();
        TT_FATAL(
            width > 0 && static_cast<uint64_t>(width) * shards <= (1U << 24),
            "VocabParallelCrossEntropy: the full vocabulary ({} x {}) must be at most 2^24",
            shards,
            width);
        if (a.mode == Mode::Backward) {
            TT_FATAL(t.state.has_value(), "VocabParallelCrossEntropy: backward needs the row state");
            check_device_tensor(
                *t.state, "VocabParallelCrossEntropy", "state", {.dtypes = {ttnn::DataType::FLOAT32}});
            check_tiled_dram(*t.state, "state");
            TT_FATAL(
                t.state->logical_shape() == with_width(input, kStateWidth) &&
                    t.state->padded_shape()[-2] == input.padded_shape()[-2],
                "VocabParallelCrossEntropy: state must be {}, got {}",
                with_width(input, kStateWidth),
                t.state->logical_shape());
            if (t.upstream.has_value()) {
                check_device_tensor(
                    *t.upstream,
                    "VocabParallelCrossEntropy",
                    "upstream",
                    {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32},
                     .layout = ttnn::Layout::TILE,
                     .buffer_type = ttnn::BufferType::DRAM});
                TT_FATAL(
                    t.upstream->logical_shape() == ttnn::Shape({1, 1, 1, 1}) ||
                        (t.upstream->logical_shape() == with_width(input, 1) &&
                         t.upstream->padded_shape()[-2] == input.padded_shape()[-2]),
                    "VocabParallelCrossEntropy: upstream must be [1, 1, 1, 1] or {}",
                    with_width(input, 1));
            }
        }
    }
    static void validate_on_program_cache_hit(const Attributes& a, const Inputs& t) {
        validate_on_program_cache_miss(a, t);
    }
    static Specs compute_output_specs(const Attributes& a, const Inputs& t) {
        const tt::tt_metal::TensorLayout fp32(ttnn::DataType::FLOAT32, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG);
        switch (a.mode) {
            case Mode::Stats: {
                Specs result{tt::tt_metal::TensorSpec(with_width(t.input, kRecordWidth), fp32)};
                const auto layout = row_softmax::layout_for(t.input, row_softmax::Kind::VocabParallelStats);
                if (row_softmax::needs_exchange(layout, row_softmax::Kind::VocabParallelStats)) {
                    result.push_back(row_softmax::scratch_spec(layout));
                }
                return result;
            }
            case Mode::Combine:
                return {
                    tt::tt_metal::TensorSpec(with_width(t.input, 1), fp32),
                    tt::tt_metal::TensorSpec(with_width(t.input, kStateWidth), fp32)};
            case Mode::Backward: return {t.input.tensor_spec()};
        }
        TT_THROW("Unknown vocabulary-parallel cross-entropy mode");
    }
    static Tensors create_output_tensors(const Attributes& a, const Inputs& t) {
        Tensors result;
        for (const auto& spec : compute_output_specs(a, t))
            result.push_back(ttnn::create_device_tensor(spec, t.input.device()));
        return result;
    }
    static ttsl::hash::hash_t compute_program_hash(const Attributes& a, const Inputs& t) {
        // cluster_axis decides the program-to-coordinate mapping; value_or keeps nullopt distinct from axis 0.
        return tt::tt_metal::operation::hash_operation<Operation>(
            static_cast<uint32_t>(a.mode),
            a.cluster_axis.value_or(std::numeric_limits<uint32_t>::max()),
            a.scaler,
            t.input.dtype(),
            t.input.logical_shape(),
            t.input.padded_shape(),
            upstream_kind(t),
            t.upstream.has_value() ? t.upstream->dtype() : t.input.dtype(),
            t.target.has_value() ? t.target->logical_shape() : ttnn::Shape({1}));
    }
};

namespace {
row_softmax::Buffers buffers_for(const Attributes& a, const Inputs& t, Tensors& out) {
    row_softmax::Buffers b{.input = t.input.buffer(), .output = out[0].buffer()};
    switch (a.mode) {
        case Mode::Stats:
            b.target = t.target->buffer();
            b.scratch = out.size() > 1 ? out[1].buffer() : nullptr;
            break;
        case Mode::Combine: b.state = out[1].buffer(); break;
        case Mode::Backward:
            b.target = t.target->buffer();
            b.state = t.state->buffer();
            b.upstream = t.upstream.has_value() ? t.upstream->buffer() : nullptr;
            break;
    }
    return b;
}

tt::tt_metal::Program create_program(
    const Attributes& a, const Inputs& t, Tensors& out, uint32_t rank, row_softmax::Handles& handles) {
    row_softmax::Options options{.kind = kind_of(a.mode)};
    if (a.mode != Mode::Combine) {
        const auto& target = *t.target;
        options.first_v = rank * t.input.logical_shape()[-1];
        options.target_page_bytes = target.logical_shape()[-1] * target.element_size();
        options.target_tiled_height = t.input.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    }
    if (a.mode == Mode::Backward) {
        options.scaler = a.scaler;
        options.upstream_kind = upstream_kind(t);
        if (t.upstream.has_value()) {
            options.upstream_dtype = t.upstream->dtype();
        }
    }
    auto program = tt::tt_metal::CreateProgram(*t.input.device());
    handles = row_softmax::create(program, options, t.input, buffers_for(a, t, out));
    return program;
}
}  // namespace

Factory::cached_mesh_workload_t Factory::create_mesh_workload(
    const Attributes& a, const ttnn::MeshCoordinateRangeSet& tensor_coords, const Inputs& t, Tensors& out) {
    const auto mesh_shape = t.input.device()->shape();
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    const auto add = [&](const ttnn::MeshCoordinateRange& range, uint32_t rank) {
        shared_variables_t sv{};
        workload.add_program(range, create_program(a, t, out, rank, sv.handles));
        shared[range] = sv;
    };
    if (a.cluster_axis.has_value()) {
        // One program per TP rank: the hyperplane of the mesh fixing the TP coordinate.
        const uint32_t axis = *a.cluster_axis;
        auto slab_start = ttnn::MeshCoordinate::zero_coordinate(mesh_shape.dims());
        auto slab_end = ttnn::MeshCoordinate::zero_coordinate(mesh_shape.dims());
        for (size_t d = 0; d < mesh_shape.dims(); ++d) {
            slab_end[static_cast<int32_t>(d)] = mesh_shape[d] - 1U;
        }
        for (uint32_t rank = 0; rank < mesh_shape[axis]; ++rank) {
            slab_start[static_cast<int32_t>(axis)] = rank;
            slab_end[static_cast<int32_t>(axis)] = rank;
            const ttnn::MeshCoordinateRange slab{slab_start, slab_end};
            for (const auto& range : tensor_coords.ranges()) {
                if (auto inter = slab.intersection(range); inter.has_value()) {
                    add(*inter, rank);
                }
            }
        }
    } else {
        for (const auto& range : tensor_coords.ranges()) {
            for (const auto& coord : range) {
                add(ttnn::MeshCoordinateRange{coord}, static_cast<uint32_t>(coord.to_linear_index(mesh_shape)));
            }
        }
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void Factory::override_runtime_arguments(
    cached_mesh_workload_t& cached, const Attributes& a, const Inputs& t, Tensors& out) {
    const auto buffers = buffers_for(a, t, out);
    for (auto& [range, program] : cached.workload.get_programs()) {
        row_softmax::override_addresses(program, cached.shared_variables.at(range).handles, buffers);
    }
}

}  // namespace ttml::metal::vocab_parallel_ce_device

namespace ttml::metal {

namespace {
using vocab_parallel_ce_device::Attributes;
using vocab_parallel_ce_device::Inputs;
using vocab_parallel_ce_device::Mode;
using vocab_parallel_ce_device::Operation;

// One upstream value, or one per row, can be folded into the kernel's row scale.
bool can_fuse_upstream(const ttnn::Tensor& logits, const ttnn::Tensor& grad) {
    if (grad.storage_type() != ttnn::StorageType::DEVICE || grad.layout() != ttnn::Layout::TILE ||
        (grad.dtype() != ttnn::DataType::BFLOAT16 && grad.dtype() != ttnn::DataType::FLOAT32) ||
        grad.memory_config() != ttnn::DRAM_MEMORY_CONFIG || grad.device() != logits.device()) {
        return false;
    }
    if (grad.logical_shape() == ttnn::Shape({1, 1, 1, 1})) {
        return true;
    }
    auto per_row = logits.logical_shape();
    per_row[-1] = 1U;
    return grad.logical_shape() == per_row && grad.padded_shape()[-2] == logits.padded_shape()[-2];
}
}  // namespace

ttnn::Tensor vocab_parallel_ce_stats(
    const ttnn::Tensor& logits, const ttnn::Tensor& target, std::optional<uint32_t> cluster_axis) {
    return ttnn::device_operation::launch<Operation>(
        Attributes{.mode = Mode::Stats, .cluster_axis = cluster_axis},
        Inputs{logits, target, std::nullopt, std::nullopt})[0];  // Drops the exchange scratch.
}

std::array<ttnn::Tensor, 2> vocab_parallel_ce_combine(const ttnn::Tensor& gathered_stats) {
    auto result = ttnn::device_operation::launch<Operation>(
        Attributes{.mode = Mode::Combine}, Inputs{gathered_stats, std::nullopt, std::nullopt, std::nullopt});
    return {result[0], result[1]};
}

ttnn::Tensor vocab_parallel_ce_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& target,
    const ttnn::Tensor& state,
    const ttnn::Tensor& upstream,
    float scaler,
    std::optional<uint32_t> cluster_axis) {
    const Attributes attributes{.mode = Mode::Backward, .cluster_axis = cluster_axis, .scaler = scaler};
    if (can_fuse_upstream(logits, upstream)) {
        return ttnn::device_operation::launch<Operation>(attributes, Inputs{logits, target, state, upstream})[0];
    }
    return ttnn::multiply(
        ttnn::device_operation::launch<Operation>(attributes, Inputs{logits, target, state, std::nullopt})[0],
        upstream);
}

}  // namespace ttml::metal
