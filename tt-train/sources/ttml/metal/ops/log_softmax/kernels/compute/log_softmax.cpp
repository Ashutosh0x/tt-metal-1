// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/kernel_profiler.hpp"
#include "log_softmax_sfpu.h"

// Shared by the fused log_softmax and, with CROSS_ENTROPY, the fused cross-entropy (see row_softmax_program.hpp).
// A work unit is one row of 32 physical rows restricted to tile columns [begin, end). With `chunks` > 1
// the row is split across that many cores, each holding one unit; their partial statistics are exchanged
// through DRAM (see the dataflow kernels) and every core of the row combines them redundantly.
constexpr uint32_t chunks = get_compile_time_arg_val(0);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(1);  // Cross-entropy backward: s.
constexpr uint32_t cx = 0, cdy = 1, ccols = 2, cm = 3, cs = 4, cic = 5, ci = 6, cmax = 7, cstateout = 8, cstatein = 9,
                   cpivot = 10, ctail = 11, cd = 12, cnorm = 13, csum = 14, cothers = 15, cout = 16, cpart0 = 17,
                   cpart1 = 18, cpart2 = 19, cpivotdy = 20, cpivotgrad = 21, cpartout = 23, ctargetx = 25,
                   cfirstcol = 27, cupstream = 29;

inline void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
inline void fill(uint32_t dst, float value) {
    fill_tile_init();
    fill_tile(dst, value);
}
inline void sub_col(uint32_t dst, uint32_t vec) {
    sfpu_bcast_col_init();
    sfpu_sub_bcast_col(dst, vec);
}
inline void mul_col(uint32_t dst, uint32_t vec) {
    sfpu_bcast_col_init();
    sfpu_mul_bcast_col(dst, vec);
}
inline void exp_value(uint32_t dst) {
    exp_tile_init<false>();
    exp_tile<false>(dst);
}
// Packs DST tiles 0..n-1 into n consecutive tiles of `cb`. The caller has acquired DST.
inline void pack_n(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t i = 0; i < n; ++i) pack_tile(i, cb);
    tile_regs_release();
    cb_push_back(cb, n);
}
inline void pack_one(uint32_t reg, uint32_t cb) {
    cb_reserve_back(cb, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
template <PoolType type>
inline void reduce_row(uint32_t dst) {
    sfpu_reduce_init<type, DataFormat::Float32>();
    sfpu_reduce<type, DataFormat::Float32, ReduceDim::REDUCE_ROW>(dst);
}
inline void add_tiles(uint32_t a, uint32_t b, uint32_t out) {
    add_binary_tile_init();
    add_binary_tile(a, b, out);
}

// Writes column vector `index` of `cb` broadcast across all 32 columns of a tile in `out`, so the per-tile
// passes need no column broadcasts of their own.
inline void broadcast_col(uint32_t cb, uint32_t index, uint32_t out) {
    tile_regs_acquire();
    fill(0, 0.F);
    load(cb, index, 1);
    sfpu_bcast_col_init();
    sfpu_add_bcast_col(0, 1);
    pack_one(0, out);
}

#if !defined(BACKWARD) || defined(CROSS_ENTROPY)
// Combines 32 lanes of (maximum m, absolute first-maximum index i, tail t) per row into the row's
// maximum (cmax), first maximum index (cpivot) and tail excluding exactly that element (ctail):
//   M = max m;  K = min{i : m == M};  T = sum over lanes of t * exp(m - M) + [lane not K] exp(m - M).
// Lanes are either SFPU lanes of one core or, after the exchange, the cores sharing a row.
inline void combine(uint32_t cb_m, uint32_t cb_i, uint32_t cb_t) {
    tile_regs_acquire();
    load(cb_m, 0, 0);
    reduce_row<PoolType::MAX>(0);
    pack_one(0, cmax);
    cb_wait_front(cmax, 1);
    tile_regs_acquire();
    load(cb_m, 0, 0);
    load(cmax, 0, 1);
    sub_col(0, 1);
    pack_one(0, cd);
    cb_wait_front(cd, 1);
    tile_regs_acquire();
    load(cd, 0, 0);
    load(cb_i, 0, 1);
    log_softmax_mask<1, 1>(0, 1U << 25);
    reduce_row<PoolType::MIN>(0);
    pack_one(0, cpivot);
    cb_wait_front(cpivot, 1);
    tile_regs_acquire();
    load(cd, 0, 0);
    exp_value(0);
    load(cb_t, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 1);
    load(cb_i, 0, 2);
    load(cpivot, 0, 3);
    sub_col(2, 3);
    // Never add the selected e^0 = 1 to the tail (arXiv:2610.31642, Eq. 5).
    log_softmax_mask<3, 2>(0, 0);
    add_tiles(0, 1, 0);
    reduce_row<PoolType::SUM>(0);
    pack_one(0, ctail);
    cb_wait_front(ctail, 1);
    cb_pop_front(cd, 1);
}

// Pass 1 and the lane (and chunk) combine. Leaves the row maximum (cmax), first-maximum index (cpivot) and
// log1p(tail) (cnorm) at the front of their buffers, each a column vector.
inline void row_statistics(uint32_t begin, uint32_t end) {
    // Pass 1: online per-lane maximum, first-maximum tile column and rescaled tail.
    {
        DeviceZoneScopedN("LSM-FW-STATS");
        tile_regs_acquire();
        fill(0, -std::numeric_limits<float>::infinity());
        fill(1, 0.F);
        fill(2, std::numeric_limits<float>::infinity());
        reconfig_data_format_srca(cx);
        copy_init(cx);
        log_softmax_online_init();
        for (uint32_t col = begin; col < end; ++col) {
            cb_wait_front(cx, 1);
            copy_tile(cx, 0, 3);
            log_softmax_online(0, col);
            cb_pop_front(cx, 1);
        }
    }
    {
        DeviceZoneScopedN("LSM-FW-COMBINE");
        cb_reserve_back(cm, 1);
        cb_reserve_back(cs, 1);
        cb_reserve_back(cic, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cm);
        pack_tile(0, cm);
        pack_tile(1, cs);
        pack_tile(2, cic);
        tile_regs_release();
        cb_push_back(cm, 1);
        cb_push_back(cs, 1);
        cb_push_back(cic, 1);
        // Absolute index = 32 * tile column + lane column.
        cb_wait_front(cic, 1);
        tile_regs_acquire();
        load(cic, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x42000000U /* 32.0f */);
        load(ccols, 0, 1);
        add_tiles(0, 1, 0);
        pack_one(0, ci);
        cb_pop_front(cic, 1);
        cb_wait_front(cm, 1);
        cb_wait_front(cs, 1);
        cb_wait_front(ci, 1);
        combine(cm, ci, cs);
        cb_pop_front(cm, 1);
        cb_pop_front(cs, 1);
        cb_pop_front(ci, 1);
        if constexpr (chunks > 1) {
            // Publish this core's row statistics, then combine every core's, one per lane.
            tile_regs_acquire();
            load(cmax, 0, 0);
            load(cpivot, 0, 1);
            load(ctail, 0, 2);
            pack_n(cpartout, 3);
            cb_pop_front(cmax, 1);
            cb_pop_front(cpivot, 1);
            cb_pop_front(ctail, 1);
            cb_wait_front(cpart0, 1);
            cb_wait_front(cpart1, 1);
            cb_wait_front(cpart2, 1);
            combine(cpart0, cpart1, cpart2);
            cb_pop_front(cpart0, 1);
            cb_pop_front(cpart1, 1);
            cb_pop_front(cpart2, 1);
        }
        tile_regs_acquire();
        load(ctail, 0, 0);
        log1p_tile_init<false>();
        log1p_tile<false>(0);
        pack_one(0, cnorm);
        cb_pop_front(ctail, 1);
        cb_wait_front(cnorm, 1);
    }
}
#endif


#ifdef CROSS_ENTROPY
// Loss of each row from its target logit x_t (column 0 of ctargetx; zero for an out-of-range target):
// -y_t with y_t = (x_t - m) - log1p(tail), computed exactly as the log_softmax output. When the target is
// the selected maximum, x_t - m = 0 and the loss is log1p(tail) itself, however small.
inline void cross_entropy_loss() {
    DeviceZoneScopedN("CE-FW-LOSS");
    cb_wait_front(ctargetx, 1);
    tile_regs_acquire();
    load(ctargetx, 0, 0);
    load(cmax, 0, 1);
    sub_col(0, 1);
    load(cnorm, 0, 1);
    sub_col(0, 1);
    negative_tile_init();
    negative_tile(0);
    // Keep column 0 only, so the loss tile's padding columns are zero.
    load(cfirstcol, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    pack_one(0, cout);
    cb_pop_front(ctargetx, 1);
    cb_pop_front(cmax, 1);
    cb_pop_front(cpivot, 1);
    cb_pop_front(cnorm, 1);
}

#ifdef UPSTREAM_GRAD
// Multiplies the column vector in `dst` by each row's upstream gradient (cupstream, column 0).
inline void scale_by_upstream(uint32_t dst) {
    load(cupstream, 0, dst + 1);
    mul_binary_tile_init();
    mul_binary_tile(dst, dst + 1, dst);
}
#endif

// dx = s * softmax(x) = exp(x - m) * q with q = s * exp(-log1p(tail)), one exponential per element.
// The writer substitutes the target element with s * (p_t - 1) = s * expm1(y_t), y_t = (x_t - m) - log1p(tail):
// for a target at the selected maximum that is -s * tail / (1 + tail), the sum of the other gradients
// (arXiv:2610.31642, Eq. 7), without subtracting nearly equal values.
inline void cross_entropy_grad_rows(uint32_t begin, uint32_t end) {
    {
        DeviceZoneScopedN("CE-BW-TARGET");
        tile_regs_acquire();
        load(cnorm, 0, 0);
        negative_tile_init();
        negative_tile(0);
        exp_value(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scaler_bits);
#ifdef UPSTREAM_GRAD
        cb_wait_front(cupstream, 1);
        scale_by_upstream(0);
#endif
        pack_one(0, cd);
        cb_wait_front(cd, 1);
        broadcast_col(cd, 0, cs);
        cb_pop_front(cd, 1);
        cb_wait_front(ctargetx, 1);
        tile_regs_acquire();
        load(ctargetx, 0, 0);
        load(cmax, 0, 1);
        sub_col(0, 1);
        load(cnorm, 0, 1);
        sub_col(0, 1);
        expm1_tile_init();
        expm1_tile(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scaler_bits);
#ifdef UPSTREAM_GRAD
        scale_by_upstream(0);
        cb_pop_front(cupstream, 1);
#endif
        pack_one(0, cpivotgrad);
        cb_pop_front(ctargetx, 1);
        broadcast_col(cmax, 0, cm);
        cb_pop_front(cmax, 1);
        cb_pop_front(cpivot, 1);
        cb_pop_front(cnorm, 1);
        cb_wait_front(cm, 1);
        cb_wait_front(cs, 1);
    }
    DeviceZoneScopedN("CE-BW-EMIT");
    log_softmax_online_init();
    for (uint32_t col = begin; col < end; ++col) {
        cb_wait_front(cx, 1);
        tile_regs_acquire();
        load(cm, 0, 1);
        load(cs, 0, 2);
        load(cx, 0, 0);
        cross_entropy_grad(0);
        pack_one(0, cout);
        cb_pop_front(cx, 1);
    }
    cb_pop_front(cm, 1);
    cb_pop_front(cs, 1);
}
#endif

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0), begin = get_arg_val<uint32_t>(1), end = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(cx, cx, cout);
    cb_wait_front(ccols, 1);
#if defined(CROSS_ENTROPY) && !defined(BACKWARD)
    cb_wait_front(cfirstcol, 1);
#endif
    for (uint32_t row = 0; row < count; ++row) {
#if defined(CROSS_ENTROPY)
        row_statistics(begin, end);
#ifndef BACKWARD
        cross_entropy_loss();
#else
        cross_entropy_grad_rows(begin, end);
#endif
#elif !defined(BACKWARD)
        row_statistics(begin, end);
        tile_regs_acquire();
        load(cmax, 0, 0);
        load(cpivot, 0, 1);
        load(cnorm, 0, 2);
        pack_n(cstateout, 3);
        cb_pop_front(cpivot, 1);
        // Pass 2: emit native-precision output (x - m) - log1p(tail).
        DeviceZoneScopedN("LSM-FW-EMIT");
        broadcast_col(cmax, 0, cm);
        broadcast_col(cnorm, 0, cs);
        cb_pop_front(cmax, 1);
        cb_pop_front(cnorm, 1);
        cb_wait_front(cm, 1);
        cb_wait_front(cs, 1);
        log_softmax_online_init();
        for (uint32_t col = begin; col < end; ++col) {
            cb_wait_front(cx, 1);
            tile_regs_acquire();
            load(cm, 0, 1);
            load(cs, 0, 2);
            load(cx, 0, 0);
            log_softmax_output(0);
            pack_one(0, cout);
            cb_pop_front(cx, 1);
        }
        cb_pop_front(cm, 1);
        cb_pop_front(cs, 1);
#else
        // State: maximum, selected index, log1p(tail). The reader zeroes each row's selected upstream
        // element in pass 1 and hands it over separately (cpivotdy), so pass 1 is a plain sum.
        cb_wait_front(cstatein, 3);
        {
            DeviceZoneScopedN("LSM-BW-SUM");
            tile_regs_acquire();
            fill(0, 0.F);
            reconfig_data_format_srca(cdy);
            copy_init(cdy);
            add_binary_tile_init();
            for (uint32_t col = begin; col < end; ++col) {
                cb_wait_front(cdy, 1);
                copy_tile(cdy, 0, 1);
                add_binary_tile(0, 1, 0);
                cb_pop_front(cdy, 1);
            }
        }
        {
            DeviceZoneScopedN("LSM-BW-PIVOT");
            reduce_row<PoolType::SUM>(0);
            pack_one(0, cothers);
            cb_wait_front(cothers, 1);
            cb_wait_front(cpivotdy, 1);
            uint32_t pivot_dy = cpivotdy;
            if constexpr (chunks > 1) {
                tile_regs_acquire();
                load(cothers, 0, 0);
                load(cpivotdy, 0, 1);
                pack_n(cpartout, 2);
                cb_pop_front(cothers, 1);
                cb_pop_front(cpivotdy, 1);
                cb_wait_front(cpart0, 1);
                cb_wait_front(cpart1, 1);
                tile_regs_acquire();
                load(cpart0, 0, 0);
                reduce_row<PoolType::SUM>(0);
                pack_one(0, cothers);
                tile_regs_acquire();
                load(cpart1, 0, 0);
                reduce_row<PoolType::SUM>(0);
                pack_one(0, cd);
                cb_pop_front(cpart0, 1);
                cb_pop_front(cpart1, 1);
                cb_wait_front(cothers, 1);
                cb_wait_front(cd, 1);
                pivot_dy = cd;
            }
            // S = A + dy_k, where A sums the upstream gradient over every element except the selected one.
            tile_regs_acquire();
            load(cothers, 0, 0);
            load(pivot_dy, 0, 1);
            add_tiles(0, 1, 0);
            pack_one(0, csum);
            cb_wait_front(csum, 1);
            // Eq. (7) in closed form: dx_k = -sum_{j!=k} (dy_j - p_j S) = S * q - A with
            // q = sum_{j!=k} p_j = t / (1 + t) = -expm1(-log1p(t)), accurate when p_k rounds to one.
            tile_regs_acquire();
            load(cstatein, 2, 0);
            negative_tile_init();
            negative_tile(0);
            expm1_tile_init();
            expm1_tile(0);
            load(csum, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            load(cothers, 0, 1);
            add_tiles(0, 1, 0);
            negative_tile_init();
            negative_tile(0);
            pack_one(0, cpivotgrad);
            cb_pop_front(cothers, 1);
            cb_pop_front(pivot_dy, 1);
            // q = S * exp(-l) = S * p_k, so that softmax(x) * S = exp(x - m) * q.
            tile_regs_acquire();
            load(cstatein, 2, 0);
            negative_tile_init();
            negative_tile(0);
            exp_value(0);
            load(csum, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            pack_one(0, cnorm);
            cb_pop_front(csum, 1);
            cb_wait_front(cnorm, 1);
            broadcast_col(cstatein, 0, cm);
            broadcast_col(cnorm, 0, cs);
            cb_pop_front(cnorm, 1);
            cb_wait_front(cm, 1);
            cb_wait_front(cs, 1);
        }
        // Pass 2: dx = dy - exp(x - m) * q. The writer substitutes dx_k.
        DeviceZoneScopedN("LSM-BW-EMIT");
        log_softmax_online_init();
        for (uint32_t col = begin; col < end; ++col) {
            cb_wait_front(cx, 1);
            cb_wait_front(cdy, 1);
            tile_regs_acquire();
            load(cm, 0, 2);
            load(cs, 0, 3);
            load(cx, 0, 0);
            load(cdy, 0, 1);
            log_softmax_grad(0);
            pack_one(0, cout);
            cb_pop_front(cx, 1);
            cb_pop_front(cdy, 1);
        }
        cb_pop_front(cm, 1);
        cb_pop_front(cs, 1);
        cb_pop_front(cstatein, 3);
#endif
    }
#if defined(CROSS_ENTROPY) && !defined(BACKWARD)
    cb_pop_front(cfirstcol, 1);
#endif
    cb_pop_front(ccols, 1);
}
