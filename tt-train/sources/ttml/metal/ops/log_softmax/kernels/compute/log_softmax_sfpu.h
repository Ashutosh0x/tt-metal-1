// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
namespace ckernel::sfpu {
// All values and indices remain FP32. Column indices are exact through 2^24.
// A DST tile is 32 SFPU rows, so dst_reg[k * 32] addresses the tile k slots above the operand.
template <int mode, int index_offset>
inline void calculate_log_softmax_mask(uint32_t width) {
    const sfpi::vFloat limit = static_cast<float>(width);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vFloat index = sfpi::dst_reg[index_offset * 32];
        if constexpr (mode == 1) {
            // Keep the index only where the value is zero (the lane holds the row maximum).
            v_if(sfpi::abs(value) != 0.0f) {
                index = std::numeric_limits<float>::infinity();
            }
            v_endif;
            v_if(index >= limit) {
                index = std::numeric_limits<float>::infinity();
            }
            v_endif;
            value = index;
        } else {
            static_assert(mode == 3);
            // Zero the value where the index difference is zero (the selected maximum).
            v_if(sfpi::abs(index) == 0.0f) {
                value = 0.0f;
            }
            v_endif;
        }
        sfpi::dst_reg[0] = value;
        sfpi::dst_reg++;
    }
}

// The accurate FP32 exponential with its 32-bit range-reduction constants held in the three programmable
// constant registers (set by log_softmax_exp_constants), instead of two SFPLOADIs each on every row.
inline void log_softmax_exp_constants() {
    sfpi::vConstFloatPrgm0 = EXP_FP32_LOG2E;
    sfpi::vConstFloatPrgm1 = EXP_FP32_NEG_LN2_HI;
#ifdef ARCH_BLACKHOLE
    sfpi::vConstFloatPrgm2 = EXP_FP32_P1;
#else
    sfpi::vConstFloatPrgm2 = EXP_FP32_NEG_LN2_LO;
#endif
}
sfpi_inline sfpi::vFloat log_softmax_exp(sfpi::vFloat a) {
#ifdef ARCH_BLACKHOLE
    return _sfpu_exp_fp32_accurate_(
        a, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, EXP_FP32_P0, sfpi::vConstFloatPrgm2);
#else
    return _sfpu_exp_fp32_accurate_(a, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2);
#endif
}

// Online lane statistics (arXiv:2610.31642, Section 2, applied per SFPU lane). Tiles from the operand:
//   +0 running maximum m, +1 tail t = sum of exp(x - m) over every element seen except one first maximum,
//   +2 tile column of that first maximum, +3 the incoming logits x (padding already -inf).
// A strictly larger x becomes the new maximum and the old one joins the tail: t' = (t + 1) * exp(m - x).
// Ties take the else branch, so each lane keeps its first maximum and counts later ties in the tail.
// One exponential per element: exp(-|x - m|) serves both branches.
inline void calculate_log_softmax_online(uint32_t column) {
    const sfpi::vFloat tile_column = static_cast<float>(column);
#pragma GCC unroll 0
    for (int i = 0; i < 8; ++i) {
        sfpi::vFloat x = sfpi::dst_reg[96];
        sfpi::vFloat argument = sfpi::setsgn(x - sfpi::dst_reg[0], 1);
        // -inf - (-inf) is NaN; padding never contributes to the tail.
        v_if(x == -std::numeric_limits<float>::infinity()) {
            argument = -std::numeric_limits<float>::infinity();
        }
        v_endif;
        const sfpi::vFloat e = log_softmax_exp(argument);
        x = sfpi::dst_reg[96];
        sfpi::vFloat tail = sfpi::dst_reg[32];
        v_if(x > sfpi::dst_reg[0]) {
            tail = (tail + 1.0f) * e;
            sfpi::dst_reg[0] = x;
            sfpi::dst_reg[64] = tile_column;
        }
        v_else {
            tail = tail + e;
        }
        v_endif;
        sfpi::dst_reg[32] = tail;
        sfpi::dst_reg++;
    }
}

// Output: +0 logits x (overwritten), +1 row maximum m, +2 log1p(tail) l, both broadcast across the row.
// (x - m) is exact near the maximum, so the selected element gets exactly -l.
inline void calculate_log_softmax_output() {
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat m = sfpi::dst_reg[32];
        const sfpi::vFloat l = sfpi::dst_reg[64];
        sfpi::dst_reg[0] = (x - m) - l;
        sfpi::dst_reg++;
    }
}

// Gradient: +0 logits x (overwritten), +1 upstream dy, +2 row maximum m, +3 q = S * exp(-l), the last two
// broadcast across the row: dx = dy - exp(x - m) * q = dy - softmax(x) * S.
inline void calculate_log_softmax_grad() {
#pragma GCC unroll 0
    for (int i = 0; i < 8; ++i) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat m = sfpi::dst_reg[64];
        const sfpi::vFloat e = log_softmax_exp(x - m);
        const sfpi::vFloat dy = sfpi::dst_reg[32];
        const sfpi::vFloat q = sfpi::dst_reg[96];
        sfpi::dst_reg[0] = dy - e * q;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
namespace ckernel {
template <int mode, int index_offset = 1>
ALWI void log_softmax_mask(uint32_t dst, uint32_t width) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log_softmax_mask, (mode, index_offset), dst, VectorMode::RC, width));
}
// Shared by the custom operations below. Call it again after any other SFPU init, which may reprogram
// the constant registers the exponential reads.
ALWI void log_softmax_online_init() {
    MATH(SFPU_UNARY_INIT(unused));
    MATH((ckernel::sfpu::log_softmax_exp_constants()));
}
ALWI void log_softmax_output(uint32_t dst) {
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log_softmax_output, dst, VectorMode::RC));
}
ALWI void log_softmax_grad(uint32_t dst) {
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log_softmax_grad, dst, VectorMode::RC));
}
// Updates the lane statistics in DST tiles dst..dst+2 with the logits tile at dst+3.
ALWI void log_softmax_online(uint32_t dst, uint32_t column) {
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log_softmax_online, dst, VectorMode::RC, column));
}
}  // namespace ckernel
