// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The per-launch captures of the tile-wide counters, which one core of the tile runs around the launch of all its
// cores: the NoC interface counters and the NoC router counters. Each compiles to nothing unless its option is on.

#pragma once

#include "internal/profiler/noc_counters.hpp"
#include "internal/profiler/router_counters.hpp"

#define TileCountersBegin()    \
    do {                       \
        NocCountersBegin();    \
        RouterCountersBegin(); \
    } while (0)

#define TileCountersEnd()    \
    do {                     \
        RouterCountersEnd(); \
        NocCountersEnd();    \
    } while (0)
