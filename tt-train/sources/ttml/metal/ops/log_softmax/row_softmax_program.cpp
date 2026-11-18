// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "row_softmax_program.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <string>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <vector>

#include "metal/common/program_utils.hpp"

#ifndef TTML_LOG_SOFTMAX_KERNEL_ROOT
#define TTML_LOG_SOFTMAX_KERNEL_ROOT "tt-train/sources/ttml/metal/ops/log_softmax/kernels/"
#endif

namespace ttml::metal::row_softmax {
namespace {
constexpr uint32_t kMaxChunks = 32;      // Partial statistics are exchanged as the 32 lanes of a tile.
constexpr uint32_t kMinChunkTiles = 16;  // Below this the exchange costs more than the extra cores save.
constexpr uint32_t kTargetPageBytes = 32 * sizeof(uint32_t);  // One target per physical row of a tile row.

bool is_backward(Kind kind) {
    return kind == Kind::LogSoftmaxBackward || kind == Kind::CrossEntropyBackward;
}
bool is_cross_entropy(Kind kind) {
    return kind == Kind::CrossEntropyForward || kind == Kind::CrossEntropyBackward;
}
}  // namespace

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

tt::tt_metal::TensorSpec scratch_spec(const Layout& layout) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, 32, layout.rows * layout.chunks * 32}),
        tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32, ttnn::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG));
}

Handles create(
    tt::tt_metal::Program& program, const Options& options, const ttnn::Tensor& input, const Buffers& buffers) {
    using namespace tt::tt_metal;
    auto* device = input.device();
    const bool backward = is_backward(options.kind);
    const bool cross_entropy = is_cross_entropy(options.kind);
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
    const bool ce_upstream = options.kind == Kind::CrossEntropyBackward && options.upstream_kind != 0;
    const auto grad_dtype =
        options.kind == Kind::LogSoftmaxBackward || ce_upstream ? options.upstream_dtype : input.dtype();
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
    if (cross_entropy) {
        // 25: target logits (column 0) and 26: row targets for the writer, double-buffered so the reader can
        // start the next row; 27: column-0 mask; 28: aligned scratch for the 64-byte target-logit reads
        // (32 slots plus alignment slack).
        create_circular_buffer(program, all, 25, tt::DataFormat::Float32, fp32_bytes, 2);
        create_circular_buffer(program, all, 26, tt::DataFormat::Float32, kTargetPageBytes, 2);
        create_circular_buffer(program, all, 27, tt::DataFormat::Float32, fp32_bytes, 1);
        create_circular_buffer(program, all, 28, tt::DataFormat::Float32, fp32_bytes, 1);
    }
    if (ce_upstream) {
        // 29: per-row upstream gradient (column 0), 30: the raw upstream tile it is taken from.
        create_circular_buffer(program, all, 29, tt::DataFormat::Float32, fp32_bytes, 2);
        create_circular_buffer(program, all, 30, grad_fmt, tt::tile_size(grad_fmt), 1);
    }
    const uint32_t arrive_sem = CreateSemaphore(program, all, 0);
    const uint32_t go_sem = CreateSemaphore(program, all, 0);

    std::map<std::string, std::string> defines;
    if (backward) {
        defines["BACKWARD"] = "1";
    }
    if (cross_entropy) {
        defines["CROSS_ENTROPY"] = "1";
    }
    if (ce_upstream) {
        defines["UPSTREAM_GRAD"] = std::to_string(options.upstream_kind);
    }
    // The scratch accessor is always compiled in; without a split it aliases the input and is never used.
    auto* scratch = chunks > 1 ? buffers.scratch : buffers.input;
    TT_FATAL(scratch != nullptr, "Row-split programs need an exchange scratch buffer");
    auto* state = buffers.state != nullptr ? buffers.state : buffers.input;
    auto* target = buffers.target != nullptr ? buffers.target : buffers.input;
    const uint32_t width = input.logical_shape()[-1];
    std::vector<uint32_t> ra{
        width_tiles,
        block,
        width,
        input.dtype() == ttnn::DataType::FLOAT32,
        grad_dtype == ttnn::DataType::FLOAT32,
        chunks,
        arrive_sem,
        go_sem,
        options.target_page_bytes,
        options.target_tiled_height};
    TensorAccessorArgs(buffers.input).append_to(ra);
    if (options.kind == Kind::LogSoftmaxBackward) {
        TensorAccessorArgs(buffers.upstream).append_to(ra);
        TensorAccessorArgs(state).append_to(ra);
    }
    TensorAccessorArgs(scratch).append_to(ra);
    if (cross_entropy) {
        TensorAccessorArgs(target).append_to(ra);
    }
    if (ce_upstream) {
        TensorAccessorArgs(buffers.upstream).append_to(ra);
    }
    std::vector<uint32_t> wa{width_tiles, block, input.dtype() == ttnn::DataType::FLOAT32, chunks, arrive_sem};
    TensorAccessorArgs(buffers.output).append_to(wa);
    TensorAccessorArgs(state).append_to(wa);
    TensorAccessorArgs(scratch).append_to(wa);
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
            .compile_args = {chunks, std::bit_cast<uint32_t>(options.scaler)},
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
            buffers.input->address(),
            buffers.upstream != nullptr ? buffers.upstream->address() : 0,
            state->address(),
            scratch->address(),
            count,
            row,
            begin,
            end,
            chunk,
            core_block,
            target->address()};
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
            {buffers.output->address(),
             state->address(),
             scratch->address(),
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
    return {reader, writer, cores, grid.y};
}

void override_addresses(tt::tt_metal::Program& program, const Handles& handles, const Buffers& buffers) {
    auto& r = GetRuntimeArgs(program, handles.reader);
    auto& w = GetRuntimeArgs(program, handles.writer);
    // A scratch buffer belongs to the cached program (or is re-created with the output); without a split it
    // aliases the input, like state and target when the kind does not use them.
    const auto* state = buffers.state != nullptr ? buffers.state : buffers.input;
    const auto* scratch = buffers.scratch != nullptr ? buffers.scratch : buffers.input;
    const auto* target = buffers.target != nullptr ? buffers.target : buffers.input;
    for_each_core(handles.cores, handles.cores_y, [&](const tt::tt_metal::CoreCoord& c) {
        r[c.x][c.y][0] = buffers.input->address();
        r[c.x][c.y][1] = buffers.upstream != nullptr ? buffers.upstream->address() : 0;
        r[c.x][c.y][2] = state->address();
        r[c.x][c.y][3] = scratch->address();
        r[c.x][c.y][10] = target->address();
        w[c.x][c.y][0] = buffers.output->address();
        w[c.x][c.y][1] = state->address();
        w[c.x][c.y][2] = scratch->address();
    });
}

}  // namespace ttml::metal::row_softmax
