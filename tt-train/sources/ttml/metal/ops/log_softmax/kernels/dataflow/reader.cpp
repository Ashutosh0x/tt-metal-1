// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tile_constants.h"
#include "log_softmax_tile.hpp"
#ifdef CROSS_ENTROPY
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#endif

// Streams logits (and upstream gradients) for each unit: tile columns [begin, end) of one 32-row row.
// Forward: pass 1 (statistics) with tile padding set to -inf, then pass 2 (output).
// Cross-entropy: first reads each row's target and target logit (one aligned 64-byte read per row), then
// pass 1 as forward; backward streams the logits again for the gradient, padding at -inf so it gets zero.
// Backward: pass 1 reads only the upstream gradient, with the selected element of every row moved
// out into a column vector (cpivotdy) and replaced by zero; pass 2 reads logits and gradient.
// With chunks > 1, between the passes every core of a row waits until all of them have published their
// partial statistics, then gathers them as lanes of one tile per statistic.
// Vocabulary-parallel: targets are global, this shard owns [first_v, first_v + width). VP_STATS reads like the
// cross-entropy forward. VP_BACKWARD reads the row state (M, l) instead of a statistics pass. VP_COMBINE reads
// the all-gathered per-shard records of each row (one tile per shard) and hands them over as lanes.
void kernel_main() {
    constexpr uint32_t wt = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(2), input_fp32 = get_compile_time_arg_val(3);
    constexpr uint32_t grad_fp32 = get_compile_time_arg_val(4), chunks = get_compile_time_arg_val(5);
    constexpr uint32_t arrive_sem = get_compile_time_arg_val(6), go_sem = get_compile_time_arg_val(7);
    constexpr uint32_t target_page_bytes = get_compile_time_arg_val(8), target_tiled_h = get_compile_time_arg_val(9);
    const uint32_t x = get_arg_val<uint32_t>(0), dy = get_arg_val<uint32_t>(1), state = get_arg_val<uint32_t>(2);
    const uint32_t scratch = get_arg_val<uint32_t>(3), count = get_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(5), begin = get_arg_val<uint32_t>(6), end = get_arg_val<uint32_t>(7);
    // Each push is `block` tiles and block divides end - begin, so pushes never straddle the ring's end.
    const uint32_t chunk = get_arg_val<uint32_t>(8), block = get_arg_val<uint32_t>(9);
    [[maybe_unused]] const uint32_t first_v = get_arg_val<uint32_t>(11);
    constexpr auto xa = TensorAccessorArgs<10>();
    const auto input = TensorAccessor(xa, x);
#if defined(BACKWARD) && !defined(CROSS_ENTROPY)
    constexpr auto da = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto grad = TensorAccessor(da, dy);
    const auto stats = TensorAccessor(sa, state);
    constexpr uint32_t partials = 2;
#else
    constexpr auto pa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr uint32_t partials = 3;
#endif
#ifdef VP_COMBINE
    // Column-0 mask for the loss tile.
    DataflowBuffer target_x(25), first_col(27), record(28);
    first_col.reserve_back(1);
    log_softmax_tile::zero_fp32(first_col.get_write_ptr());
    for (uint32_t r = 0; r < 32; ++r)
        log_softmax_tile::store_fp32(first_col.get_write_ptr(), log_softmax_tile::offset(r, 0), 0x3F800000U);
    first_col.push_back(1);
#endif
#ifdef CROSS_ENTROPY
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto targets = TensorAccessor(ta, get_arg_val<uint32_t>(10));
    DataflowBuffer target_x(25), target_rows(26), first_col(27), gather(28);
    constexpr uint32_t elem_bytes = input_fp32 ? 4 : 2;
#ifndef BACKWARD
    // Column-0 mask for the loss tile.
    first_col.reserve_back(1);
    log_softmax_tile::zero_fp32(first_col.get_write_ptr());
    for (uint32_t r = 0; r < 32; ++r)
        log_softmax_tile::store_fp32(first_col.get_write_ptr(), log_softmax_tile::offset(r, 0), 0x3F800000U);
    first_col.push_back(1);
#endif
    // 32 slots of 64 bytes: the DRAM-aligned block around each row's target logit.
    gather.reserve_back(1);
    const uint32_t slots = (gather.get_write_ptr() + 63U) & ~63U;
#ifdef UPSTREAM_GRAD
    // Upstream gradient of the loss: one value (UPSTREAM_GRAD == 1) or one per row, one tile per tile row.
    constexpr auto ga = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto upstream = TensorAccessor(ga, dy);
    DataflowBuffer upstream_vec(29), upstream_tile(30);
#endif
#ifdef VP_BACKWARD
    // The row state accessor comes last.
#ifdef UPSTREAM_GRAD
    constexpr auto va = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
#else
    constexpr auto va = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
#endif
    const auto row_state = TensorAccessor(va, state);
#endif
#endif
    const Noc noc;
    DataflowBuffer xb(0), db(1), columns(2), sb(9), pivot_dy(20), staging(24);
    const uint32_t bytes = get_tile_size(0), grad_bytes = get_tile_size(1), fp32_bytes = get_tile_size(9);
    columns.reserve_back(1);
    auto ptr = reinterpret_cast<volatile tt_l1_ptr float *>(columns.get_write_ptr());
    constexpr uint32_t fh = tt::constants::FACE_HEIGHT, fw = tt::constants::FACE_WIDTH;
    for (uint32_t f = 0; f < 4; ++f)
        for (uint32_t r = 0; r < fh; ++r)
            for (uint32_t c = 0; c < fw; ++c) ptr[f * fh * fw + r * fw + c] = static_cast<float>((f % 2) * fw + c);
    columns.push_back(1);

    // Streams tiles [begin, end) of `row`; `patch(col, l1_address)` may edit each tile before it is pushed.
    const auto stream = [&](const auto &accessor, auto &dfb, uint32_t tile_bytes, uint32_t row, auto patch) {
        for (uint32_t col = begin; col < end; col += block) {
            const uint32_t n = end - col < block ? end - col : block;
            dfb.reserve_back(n);
            for (uint32_t i = 0; i < n; ++i)
                noc.async_read(
                    accessor, dfb, tile_bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * tile_bytes});
            noc.async_read_barrier();
            for (uint32_t i = 0; i < n; ++i) patch(col + i, dfb.get_write_ptr() + i * tile_bytes);
            dfb.push_back(n);
        }
    };
    [[maybe_unused]] const auto no_patch = [](uint32_t, uint32_t) {};
    // Tile padding never wins the maximum and contributes exp(-inf) = 0.
    [[maybe_unused]] const auto pad_neg_inf = [](uint32_t col, uint32_t address) {
        if constexpr (width % 32 != 0) {
            if (col == wt - 1) {
                for (uint32_t r = 0; r < 32; ++r)
                    for (uint32_t c = width % 32; c < 32; ++c)
                        log_softmax_tile::store_neg_inf<input_fp32>(address, log_softmax_tile::offset(r, c));
            }
        }
    };

    for (uint32_t row = start; row < start + count; ++row) {
#ifdef VP_COMBINE
        // Shard g's record (columns m, k, tail, x_t) becomes lane g of the three statistic tiles. Exactly one
        // shard holds a nonzero target logit (the others hold +0), so it is selected, not summed.
        DataflowBuffer m_lanes(17), k_lanes(18), t_lanes(19);
        m_lanes.reserve_back(1);
        k_lanes.reserve_back(1);
        t_lanes.reserve_back(1);
        target_x.reserve_back(1);
        log_softmax_tile::fill_fp32(m_lanes.get_write_ptr(), -__builtin_inff());
        log_softmax_tile::fill_fp32(k_lanes.get_write_ptr(), __builtin_inff());
        log_softmax_tile::zero_fp32(t_lanes.get_write_ptr());
        log_softmax_tile::zero_fp32(target_x.get_write_ptr());
        for (uint32_t g = 0; g < wt; ++g) {
            record.reserve_back(1);
            noc.async_read(input, record, fp32_bytes, {.page_id = row * wt + g}, {});
            noc.async_read_barrier();
            const uint32_t from = record.get_write_ptr();
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t lane = log_softmax_tile::offset(r, g);
                log_softmax_tile::copy_fp32(from, log_softmax_tile::offset(r, 0), m_lanes.get_write_ptr(), lane);
                log_softmax_tile::copy_fp32(from, log_softmax_tile::offset(r, 1), k_lanes.get_write_ptr(), lane);
                log_softmax_tile::copy_fp32(from, log_softmax_tile::offset(r, 2), t_lanes.get_write_ptr(), lane);
                const uint32_t x_t = log_softmax_tile::words(from)[log_softmax_tile::offset(r, 3)];
                if (x_t != 0U) {
                    log_softmax_tile::store_fp32(target_x.get_write_ptr(), log_softmax_tile::offset(r, 0), x_t);
                }
            }
        }
        m_lanes.push_back(1);
        k_lanes.push_back(1);
        t_lanes.push_back(1);
        target_x.push_back(1);
        continue;
#endif
#ifdef CROSS_ENTROPY
        // Targets of the 32 physical rows (out of range, including padded rows, becomes 0xFFFFFFFF) and their
        // logits as an FP32 column vector; an out-of-range target contributes a zero logit.
        target_rows.reserve_back(1);
        const uint32_t target_address = target_rows.get_write_ptr();
        read_target_indices_page_clamped(
            targets, target_address, row, target_tiled_h, target_page_bytes, 32 * sizeof(uint32_t));
        auto target = log_softmax_tile::words(target_address);
        for (uint32_t r = 0; r < 32; ++r) {
            // Shard-local column; targets on other shards (and invalid ones) wrap far above width.
            target[r] -= first_v;
            if (target[r] >= width) {
                target[r] = 0xFFFFFFFFU;
                continue;
            }
            const uint32_t at = log_softmax_tile::offset(r, target[r] % 32) * elem_bytes;
            noc_async_read(input.get_noc_addr(row * wt + target[r] / 32, at & ~63U), slots + r * 64, 64);
        }
        noc_async_read_barrier();
        target_x.reserve_back(1);
        log_softmax_tile::zero_fp32(target_x.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            if (target[r] != 0xFFFFFFFFU) {
                const uint32_t at = log_softmax_tile::offset(r, target[r] % 32) * elem_bytes;
                log_softmax_tile::store_fp32(
                    target_x.get_write_ptr(),
                    log_softmax_tile::offset(r, 0),
                    log_softmax_tile::load<input_fp32>(slots + r * 64 + (at & 63U), 0));
            }
        }
        target_x.push_back(1);
#ifdef BACKWARD
        target_rows.push_back(1);  // The writer substitutes each target's gradient.
#endif
#ifdef UPSTREAM_GRAD
        upstream_tile.reserve_back(1);
        noc.async_read(upstream, upstream_tile, get_tile_size(30), {.page_id = UPSTREAM_GRAD == 2 ? row : 0}, {});
        noc.async_read_barrier();
        upstream_vec.reserve_back(1);
        log_softmax_tile::zero_fp32(upstream_vec.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r)
            log_softmax_tile::store_fp32(
                upstream_vec.get_write_ptr(),
                log_softmax_tile::offset(r, 0),
                log_softmax_tile::load<grad_fp32>(
                    upstream_tile.get_write_ptr(), log_softmax_tile::offset(UPSTREAM_GRAD == 2 ? r : 0, 0)));
        upstream_vec.push_back(1);
#endif
#endif
#if defined(VP_BACKWARD)
        // Row state of the combined statistics: maximum M (page 2 * row) and log1p(tail) l (page 2 * row + 1).
        {
            DataflowBuffer max_vec(7), norm_vec(13);
            max_vec.reserve_back(1);
            norm_vec.reserve_back(1);
            noc.async_read(row_state, max_vec, fp32_bytes, {.page_id = 2 * row}, {});
            noc.async_read(row_state, norm_vec, fp32_bytes, {.page_id = 2 * row + 1}, {});
            noc.async_read_barrier();
            max_vec.push_back(1);
            norm_vec.push_back(1);
        }
#elif !defined(BACKWARD) || defined(CROSS_ENTROPY)
        stream(input, xb, bytes, row, pad_neg_inf);
#else
        // Row state: maximum, selected index and log1p(tail) column vectors.
        sb.reserve_back(3);
        for (uint32_t i = 0; i < 3; ++i)
            noc.async_read(stats, sb, fp32_bytes, {.page_id = row * 3 + i}, {.offset_bytes = i * fp32_bytes});
        noc.async_read_barrier();
        uint32_t pivot[32];
        log_softmax_tile::read_pivots(sb.get_write_ptr() + fp32_bytes, pivot);
        sb.push_back(3);
        pivot_dy.reserve_back(1);
        const uint32_t pivot_address = pivot_dy.get_write_ptr();
        log_softmax_tile::zero_fp32(pivot_address);
        stream(grad, db, grad_bytes, row, [&](uint32_t col, uint32_t address) {
            for (uint32_t r = 0; r < 32; ++r) {
                if (pivot[r] / 32 == col) {
                    const uint32_t at = log_softmax_tile::offset(r, pivot[r] % 32);
                    log_softmax_tile::store_fp32(
                        pivot_address, log_softmax_tile::offset(r, 0), log_softmax_tile::load<grad_fp32>(address, at));
                    log_softmax_tile::store_zero<grad_fp32>(address, at);
                }
            }
            if constexpr (width % 32 != 0) {
                if (col == wt - 1) {
                    for (uint32_t r = 0; r < 32; ++r)
                        for (uint32_t c = width % 32; c < 32; ++c)
                            log_softmax_tile::store_zero<grad_fp32>(address, log_softmax_tile::offset(r, c));
                }
            }
        });
        pivot_dy.push_back(1);
#endif
        if constexpr (chunks > 1) {
            const auto parts = TensorAccessor(pa, scratch);
            if (chunk == 0) {
                auto arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(arrive_sem));
                noc_semaphore_wait(arrive, chunks);
                noc_semaphore_set(arrive, 0);
                for (uint32_t g = 0; g < chunks; ++g) {
                    const uint32_t core_x = get_arg_val<uint32_t>(12 + 2 * g),
                                   core_y = get_arg_val<uint32_t>(13 + 2 * g);
                    noc_semaphore_inc(get_noc_addr(core_x, core_y, get_semaphore(go_sem)), 1);
                }
            }
            auto go = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(go_sem));
            noc_semaphore_wait(go, 1);
            noc_semaphore_set(go, 0);
            // Fetch every chunk's packed record with one barrier; statistic i of chunk g becomes lane g of tile i.
            staging.reserve_back((chunks * log_softmax_tile::kPartialBytes + fp32_bytes - 1) / fp32_bytes);
            const uint32_t records = staging.get_write_ptr();
            for (uint32_t g = 0; g < chunks; ++g)
                noc.async_read(
                    parts,
                    staging,
                    log_softmax_tile::kPartialBytes,
                    {.page_id = row * chunks + g},
                    {.offset_bytes = g * log_softmax_tile::kPartialBytes});
            noc.async_read_barrier();
            for (uint32_t i = 0; i < partials; ++i) {
                DataflowBuffer lanes(17 + i);
                lanes.reserve_back(1);
                const uint32_t lane_address = lanes.get_write_ptr();
#if !defined(BACKWARD) || defined(CROSS_ENTROPY)
                const float fill = i == 0 ? -__builtin_inff() : (i == 1 ? __builtin_inff() : 0.F);
#else
                const float fill = 0.F;
#endif
                log_softmax_tile::fill_fp32(lane_address, fill);
                for (uint32_t g = 0; g < chunks; ++g)
                    for (uint32_t r = 0; r < 32; ++r)
                        log_softmax_tile::copy_fp32(
                            records,
                            g * log_softmax_tile::kPartialWords + i * 32 + r,
                            lane_address,
                            log_softmax_tile::offset(r, g));
                lanes.push_back(1);
            }
        }
#if defined(CROSS_ENTROPY)
#ifdef BACKWARD
        stream(input, xb, bytes, row, pad_neg_inf);
#endif
#elif !defined(BACKWARD)
        stream(input, xb, bytes, row, no_patch);
#else
        for (uint32_t col = begin; col < end; col += block) {
            const uint32_t n = end - col < block ? end - col : block;
            xb.reserve_back(n);
            db.reserve_back(n);
            for (uint32_t i = 0; i < n; ++i) {
                noc.async_read(input, xb, bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * bytes});
                noc.async_read(grad, db, grad_bytes, {.page_id = row * wt + col + i}, {.offset_bytes = i * grad_bytes});
            }
            noc.async_read_barrier();
            xb.push_back(n);
            db.push_back(n);
        }
#endif
    }
}
