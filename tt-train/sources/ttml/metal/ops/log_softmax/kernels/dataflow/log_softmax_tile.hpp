// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Element access inside one 32x32 tile in L1 (four 16x16 faces, row-major within a face), for the few
// scalar edits the data-movement kernels make: tile padding, the selected element of each row, and
// column-vector statistics. BF16 tiles hold the high half of the FP32 bit pattern.
namespace log_softmax_tile {
// One unit's partial row statistics as exchanged through DRAM: up to four 32-value column vectors
// (one per statistic) packed contiguously, rather than one 4 KiB tile per statistic.
constexpr uint32_t kPartialBytes = 512;
constexpr uint32_t kPartialWords = kPartialBytes / sizeof(uint32_t);
constexpr uint32_t offset(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
inline volatile tt_l1_ptr uint32_t* words(uint32_t base) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
}
inline volatile tt_l1_ptr uint16_t* halves(uint32_t base) {
    return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
}
// FP32 bit pattern of element `at`.
template <uint32_t fp32>
inline uint32_t load(uint32_t base, uint32_t at) {
    if constexpr (fp32) {
        return words(base)[at];
    } else {
        return static_cast<uint32_t>(halves(base)[at]) << 16;
    }
}
// Stores FP32 bits, rounding to nearest even for BF16 tiles.
template <uint32_t fp32>
inline void store(uint32_t base, uint32_t at, uint32_t bits) {
    if constexpr (fp32) {
        words(base)[at] = bits;
    } else {
        halves(base)[at] = static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
    }
}
template <uint32_t fp32>
inline void store_zero(uint32_t base, uint32_t at) {
    store<fp32>(base, at, 0U);
}
template <uint32_t fp32>
inline void store_neg_inf(uint32_t base, uint32_t at) {
    store<fp32>(base, at, 0xFF800000U);
}
inline void store_fp32(uint32_t base, uint32_t at, uint32_t bits) {
    words(base)[at] = bits;
}
inline void fill_fp32(uint32_t base, float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    for (uint32_t i = 0; i < 1024; ++i) words(base)[i] = bits;
}
inline void zero_fp32(uint32_t base) {
    fill_fp32(base, 0.F);
}
inline void copy_fp32(uint32_t from, uint32_t at_from, uint32_t to, uint32_t at_to) {
    words(to)[at_to] = words(from)[at_from];
}
// Selected maximum index of each row, stored as an exact FP32 integer in column 0.
inline void read_pivots(uint32_t base, uint32_t (&pivot)[32]) {
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t bits = words(base)[offset(r, 0)];
        const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127;
        if (bits & 0x80000000U || exponent < 0 || exponent > 23) {
            pivot[r] = exponent < 0 && !(bits & 0x80000000U) ? 0U : 0xFFFFFFFFU;  // Zero, or invalid (padding).
            continue;
        }
        pivot[r] = ((bits & 0x7FFFFFU) | 0x800000U) >> (23 - exponent);
    }
}
}  // namespace log_softmax_tile
