// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Teardown plan P28: a rank that is ending on a signal. Its own translation unit, independent of which context
// implementation (MPI or single-host) is in use: the flag is process-wide, and every context's collectives watch it.

#include "api/tt-metalium/distributed_context.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>

namespace tt::tt_metal::distributed::multihost {

namespace {
std::atomic<bool> g_terminating{false};
struct sigaction g_previous_sigterm{};
bool g_termination_handler_installed = false;

// Async-signal-safe: an atomic store, then the handler that was installed before ours.
extern "C" void on_sigterm(int signal_number) {
    g_terminating.store(true, std::memory_order_relaxed);
    if ((g_previous_sigterm.sa_flags & SA_SIGINFO) != 0) {
        return;  // a siginfo handler we cannot call with what we have; the flag is set, which is what matters
    }
    if (g_previous_sigterm.sa_handler == SIG_DFL) {
        (void)::sigaction(signal_number, &g_previous_sigterm, nullptr);
        ::raise(signal_number);  // the process ends as it would have without us
        return;
    }
    if (g_previous_sigterm.sa_handler != SIG_IGN) {
        g_previous_sigterm.sa_handler(signal_number);
    }
}
}  // namespace

void DistributedContext::mark_terminating() { g_terminating.store(true, std::memory_order_relaxed); }
bool DistributedContext::is_terminating() { return g_terminating.load(std::memory_order_relaxed); }

void DistributedContext::install_termination_handler() {
    if (g_termination_handler_installed || std::getenv("TT_METAL_SIMULATOR") != nullptr) {
        return;  // under a simulator, simulator_termination.cpp owns SIGTERM and exits through atexit
    }
    struct sigaction action{};
    action.sa_handler = on_sigterm;
    ::sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    if (::sigaction(SIGTERM, &action, &g_previous_sigterm) == 0) {
        g_termination_handler_installed = true;
    }
}

}  // namespace tt::tt_metal::distributed::multihost
