// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/tensor/tensor.hpp"

// Device program shared by the fused log_softmax and the fused cross-entropy. Both reduce each row of
// the last axis with the collapse-resistant statistics of arXiv:2610.31642, Section 2: one online pass
// keeps per-lane maximum, first-maximum index and the exponential tail without that maximum, combined
// into the row maximum m, first maximum k and log1p(tail) l in FP32.
//
//   LogSoftmaxForward:    y = (x - m) - l, plus the row state (m, k, l).
//   LogSoftmaxBackward:   dx from upstream dy and the row state; dx[k] rebuilt from the others (Eq. 7).
//   CrossEntropyForward:  loss = -((x[t] - m) - l), one value per row.
//   CrossEntropyBackward: dx = s * softmax(x), with dx[t] = s * expm1((x[t] - m) - l), where s is the scaler
//                         times the row's upstream gradient when one is given.
//
// Vocabulary-parallel cross-entropy (the vocabulary sharded across TP devices, shard r owning global columns
// [first_v, first_v + width) with first_v = r * width):
//   VocabParallelStats:    per shard and row, one FP32 tile row with columns (m, first_v + k, tail without
//                          x[k], x[t] or 0 when the target is on another shard): no log1p, no output tensor.
//   VocabParallelCombine:  input = those tiles all-gathered along the last axis ([.., 32 * shards]); combines
//                          the shards as lanes (one global first maximum, lowest shard on ties), l = log1p(T),
//                          loss = -((x[t] - M) - l) per row and the row state (M, l) as two column-vector tiles.
//   VocabParallelBackward: dx = s * exp(x - M - l) from that state, no statistics pass; the target's shard
//                          substitutes dx[t] = s * expm1((x[t] - M) - l) (Eq. 7 in closed form when t = k).
namespace ttml::metal::row_softmax {

enum class Kind : uint8_t {
    LogSoftmaxForward,
    LogSoftmaxBackward,
    CrossEntropyForward,
    CrossEntropyBackward,
    VocabParallelStats,
    VocabParallelCombine,
    VocabParallelBackward,
};

// Work split. A unit is one row of 32 physical rows; with few rows, each row's columns are split over
// `chunks` cores that exchange partial statistics.
struct Layout {
    uint32_t width_tiles;
    uint32_t rows;    // Rows of 32 physical rows.
    uint32_t chunks;  // Cores sharing one row.
    uint32_t chunk_tiles;
};
Layout layout_for(const ttnn::Tensor& input, Kind kind = Kind::LogSoftmaxForward);
// Whether the program exchanges partial statistics between the cores sharing a row (and needs a scratch).
bool needs_exchange(const Layout& layout, Kind kind);

// DRAM scratch for the exchange: one page per (row, chunk) unit. Only needed when chunks > 1.
tt::tt_metal::TensorSpec scratch_spec(const Layout& layout);

struct Options {
    Kind kind;
    ttnn::DataType upstream_dtype = ttnn::DataType::BFLOAT16;  // Upstream gradient dtype, when there is one.
    float scaler = 1.F;                                        // Cross-entropy backward only.
    // Cross-entropy backward: upstream gradient folded into the row scale s. 0: none, 1: one value
    // ([1, 1, 1, 1]), 2: one value per row ([N, C, H, 1], one tile per tile row of the input).
    uint32_t upstream_kind = 0;
    uint32_t target_page_bytes = 0;                            // Cross-entropy: bytes in one target page.
    uint32_t target_tiled_height = 0;                          // Cross-entropy: tile rows per target page.
    uint32_t first_v = 0;  // Vocabulary-parallel stats/backward: global column of this shard's first column.
};

// Buffers the program reads and writes. Null entries are unused by the kind; a null scratch aliases
// the input (never read without a split).
struct Buffers {
    tt::tt_metal::Buffer* input = nullptr;
    tt::tt_metal::Buffer* upstream = nullptr;
    tt::tt_metal::Buffer* state = nullptr;
    tt::tt_metal::Buffer* target = nullptr;
    tt::tt_metal::Buffer* output = nullptr;
    tt::tt_metal::Buffer* scratch = nullptr;
};

struct Handles {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    uint32_t cores;
    uint32_t cores_y;
};

Handles create(tt::tt_metal::Program& program, const Options& options, const ttnn::Tensor& input, const Buffers& buffers);
// Patches buffer addresses on a program-cache hit.
void override_addresses(tt::tt_metal::Program& program, const Handles& handles, const Buffers& buffers);

}  // namespace ttml::metal::row_softmax
