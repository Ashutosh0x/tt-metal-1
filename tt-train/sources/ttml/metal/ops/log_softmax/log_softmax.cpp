// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "log_softmax.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <variant>

#include "metal/common/program_utils.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

#ifndef TTML_LOG_SOFTMAX_KERNEL_ROOT
#define TTML_LOG_SOFTMAX_KERNEL_ROOT "tt-train/sources/ttml/metal/ops/log_softmax/kernels/"
#endif

namespace ttml::metal::log_softmax_device {
namespace {
constexpr uint32_t kMaxChunks = 32;      // Partial statistics are exchanged as the 32 lanes of a tile.
constexpr uint32_t kMinChunkTiles = 16;  // Below this the exchange costs more than the extra cores save.

struct Layout {
    uint32_t width_tiles;
    uint32_t rows;    // Rows of 32 physical rows.
    uint32_t chunks;  // Cores sharing one row.
    uint32_t chunk_tiles;
};

// Few rows: split each row's columns over several cores. Every core then holds exactly one (row, chunk)
// unit, so the per-row exchange never waits on a core that is busy with another row.
Layout layout_for(const ttnn::Tensor& input) {
    const auto& tile = input.tensor_spec().tile();
    const uint32_t width_tiles = input.padded_shape()[-1] / tile.get_width();
    const uint32_t rows = input.physical_volume() / (input.padded_shape()[-1] * tile.get_height());
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t cores = grid.x * grid.y;
    uint32_t chunks = 1;
    if (rows * 2 <= cores) {
        chunks = std::min({kMaxChunks, cores / rows, std::max(1U, width_tiles / kMinChunkTiles)});
    }
    const uint32_t chunk_tiles = (width_tiles + chunks - 1) / chunks;
    chunks = (width_tiles + chunk_tiles - 1) / chunk_tiles;
    return {width_tiles, rows, chunks, chunk_tiles};
}
}  // namespace

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
        tt::tt_metal::KernelHandle reader, writer;
        uint32_t cores, cores_y;
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
            result.emplace_back(ttnn::Shape({1, 1, 32, layout.rows * layout.chunks * 32}), fp32_layout);
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
LogSoftmaxFactory::cached_program_t LogSoftmaxFactory::create(
    const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    using namespace tt::tt_metal;
    const auto& input = t.input;
    auto* device = input.device();
    auto program = CreateProgram(*device);
    const auto layout = layout_for(input);
    const uint32_t width_tiles = layout.width_tiles, rows = layout.rows, chunks = layout.chunks;
    const auto grid = device->compute_with_storage_grid_size();
    // With chunks > 1 a unit is one (row, chunk) pair and every core takes at most one.
    auto [cores, all, group1, group2, units1, units2] = split_work_to_cores(grid, rows * chunks);
    TT_FATAL(chunks == 1 || (units1 == 1 && units2 <= 1), "Split rows need one unit per core");
    const auto fmt = datatype_to_dataformat_converter(input.dtype());
    const auto native_bytes = tt::tile_size(fmt), fp32_bytes = tt::tile_size(tt::DataFormat::Float32);
    // Stream rings hold two blocks of the largest block size; each core pushes blocks of 4, 2 or 1 tiles,
    // whichever divides its tile count, so every push stays aligned to the ring.
    constexpr uint32_t block = 4;
    const auto grad_dtype = a.backward ? t.upstream->dtype() : input.dtype();
    const auto grad_fmt = datatype_to_dataformat_converter(grad_dtype);
    // Bounded, double-buffered streams; scratch size is independent of vocabulary width.
    create_circular_buffer(program, all, 0, fmt, native_bytes, 2 * block);
    create_circular_buffer(program, all, 1, grad_fmt, tt::tile_size(grad_fmt), 2 * block);
    create_circular_buffer(program, all, 16, fmt, native_bytes, 2 * block);
    for (uint32_t cb : {2U, 3U, 4U, 5U, 6U, 7U, 10U, 11U, 12U, 13U, 14U, 15U, 17U, 18U, 19U, 20U, 21U, 22U})
        create_circular_buffer(program, all, cb, tt::DataFormat::Float32, fp32_bytes, 1);
    // Staging for the packed partial records of up to kMaxChunks cores (512 bytes each).
    create_circular_buffer(program, all, 24, tt::DataFormat::Float32, fp32_bytes, kMaxChunks * 512 / fp32_bytes);
    for (uint32_t cb : {8U, 9U, 23U}) create_circular_buffer(program, all, cb, tt::DataFormat::Float32, fp32_bytes, 3);
    const uint32_t arrive_sem = CreateSemaphore(program, all, 0);
    const uint32_t go_sem = CreateSemaphore(program, all, 0);

    const std::map<std::string, std::string> defines =
        a.backward ? std::map<std::string, std::string>{{"BACKWARD", "1"}} : std::map<std::string, std::string>{};
    const auto* state_buffer = a.backward ? t.state->buffer() : out[1].buffer();
    const size_t scratch_index = a.backward ? 1 : 2;
    // The scratch accessor is always compiled in; without a split it aliases the input and is never used.
    const auto* scratch_buffer = chunks > 1 ? out[scratch_index].buffer() : input.buffer();
    const uint32_t width = input.logical_shape()[-1];
    std::vector<uint32_t> ra{
        width_tiles,
        block,
        width,
        input.dtype() == ttnn::DataType::FLOAT32,
        grad_dtype == ttnn::DataType::FLOAT32,
        chunks,
        arrive_sem,
        go_sem};
    TensorAccessorArgs(input.buffer()).append_to(ra);
    if (a.backward) {
        TensorAccessorArgs(t.upstream->buffer()).append_to(ra);
        TensorAccessorArgs(t.state->buffer()).append_to(ra);
    }
    TensorAccessorArgs(scratch_buffer).append_to(ra);
    std::vector<uint32_t> wa{width_tiles, block, input.dtype() == ttnn::DataType::FLOAT32, chunks, arrive_sem};
    TensorAccessorArgs(out[0].buffer()).append_to(wa);
    TensorAccessorArgs(state_buffer).append_to(wa);
    TensorAccessorArgs(scratch_buffer).append_to(wa);
    auto reader = create_reader_kernel(program, all, ra, defines, TTML_LOG_SOFTMAX_KERNEL_ROOT "dataflow/reader.cpp");
    auto writer = create_writer_kernel(program, all, wa, defines, TTML_LOG_SOFTMAX_KERNEL_ROOT "dataflow/writer.cpp");
    // Route FP32 loads directly into DST: the FPU copy path truncates to TF32.
    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb = 2; cb < NUM_CIRCULAR_BUFFERS; ++cb) unpack_modes[cb] = UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[16] = UnpackToDestMode::Default;
    if (input.dtype() == ttnn::DataType::FLOAT32) {
        unpack_modes[0] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (grad_dtype == ttnn::DataType::FLOAT32) {
        unpack_modes[1] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto compute = CreateKernel(
        program,
        TTML_LOG_SOFTMAX_KERNEL_ROOT "compute/log_softmax.cpp",
        all,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode = false,
            .compile_args = {chunks},
            .defines = defines});

    std::vector<CoreCoord> unit_cores;
    for_each_core_with_work(
        cores, grid.y, group1, group2, units1, units2, [&](const CoreWork& w) { unit_cores.push_back(w.core); });
    for_each_core_with_work(cores, grid.y, group1, group2, units1, units2, [&](const CoreWork& w) {
        // Without a split, a core takes num_units whole rows; with one, its single unit is (row, chunk).
        const uint32_t row = chunks > 1 ? w.start / chunks : w.start;
        const uint32_t chunk = chunks > 1 ? w.start % chunks : 0;
        const uint32_t count = chunks > 1 ? 1 : w.num_units;
        const uint32_t begin = chunk * layout.chunk_tiles;
        const uint32_t end = std::min(width_tiles, begin + layout.chunk_tiles);
        const uint32_t core_block = (end - begin) % 4 == 0 ? 4 : ((end - begin) % 2 == 0 ? 2 : 1);
        // The row's chunk-0 core gathers arrivals; without a split nobody signals it.
        const auto leader = device->worker_core_from_logical_core(unit_cores[chunks > 1 ? row * chunks : w.index]);
        std::vector<uint32_t> reader_args{
            input.buffer()->address(),
            a.backward ? t.upstream->buffer()->address() : 0,
            state_buffer->address(),
            scratch_buffer->address(),
            count,
            row,
            begin,
            end,
            chunk,
            core_block};
        if (chunks > 1 && chunk == 0) {
            for (uint32_t g = 0; g < chunks; ++g) {
                const auto member = device->worker_core_from_logical_core(unit_cores[w.index + g]);
                reader_args.push_back(member.x);
                reader_args.push_back(member.y);
            }
        }
        SetRuntimeArgs(program, reader, w.core, reader_args);
        SetRuntimeArgs(
            program,
            writer,
            w.core,
            {out[0].buffer()->address(),
             state_buffer->address(),
             scratch_buffer->address(),
             count,
             row,
             begin,
             end,
             chunk,
             static_cast<uint32_t>(leader.x),
             static_cast<uint32_t>(leader.y),
             core_block});
        SetRuntimeArgs(program, compute, w.core, {count, begin, end});
    });
    return {std::move(program), {reader, writer, cores, grid.y}};
}
void LogSoftmaxFactory::override_runtime_arguments(
    cached_program_t& cache, const LogSoftmaxAttributes& a, const LogSoftmaxInputs& t, Tensors& out) {
    auto& r = GetRuntimeArgs(cache.program, cache.shared_variables.reader);
    auto& w = GetRuntimeArgs(cache.program, cache.shared_variables.writer);
    const auto* state_buffer = a.backward ? t.state->buffer() : out[1].buffer();
    const size_t scratch_index = a.backward ? 1 : 2;
    const auto* scratch_buffer = out.size() > scratch_index ? out[scratch_index].buffer() : t.input.buffer();
    for_each_core(cache.shared_variables.cores, cache.shared_variables.cores_y, [&](const tt::tt_metal::CoreCoord& c) {
        r[c.x][c.y][0] = t.input.buffer()->address();
        r[c.x][c.y][1] = a.backward ? t.upstream->buffer()->address() : 0;
        r[c.x][c.y][2] = state_buffer->address();
        r[c.x][c.y][3] = scratch_buffer->address();
        w[c.x][c.y][0] = out[0].buffer()->address();
        w[c.x][c.y][1] = state_buffer->address();
        w[c.x][c.y][2] = scratch_buffer->address();
    });
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
