// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tt-metalium/distributed_context.hpp"

#include <atomic>
#include <cstdlib>
#include <csignal>
#include <array>

#include <tt_stl/assert.hpp>

#if defined(OPEN_MPI)
#include "mpi_distributed_context.hpp"
#else
#include "single_host_context.hpp"
#endif

namespace tt::tt_metal::distributed::multihost {

#if defined(OPEN_MPI)
using ContextImpl = MPIContext;
#else
using ContextImpl = SingleHostContext;
#endif

/* -------------------- factory for generic interface --------------------- */
void DistributedContext::create(int argc, char** argv) { ContextImpl::create(argc, argv); }

const ContextPtr& DistributedContext::get_current_world() { return ContextImpl::get_current_world(); }

ContextPtr DistributedContext::get_world_context() {
#if defined(OPEN_MPI)
    return MPIContext::get_world_context();
#else
    return ContextImpl::get_current_world();
#endif
}

void DistributedContext::set_current_world(const ContextPtr& ctx) { ContextImpl::set_current_world(ctx); }

Size DistributedContext::subcontext_size(SubcontextId subcontext_id) const {
    TT_FATAL(*subcontext_id == 0, "subcontext_id {} invalid for default single-context layout", *subcontext_id);
    return size();
}

ttsl::Span<const int> DistributedContext::subcontext_sizes() const {
    static thread_local std::array<int, 1> backing;
    backing[0] = *size();
    return {backing.data(), backing.size()};
}

Rank DistributedContext::local_to_world_rank(SubcontextId subcontext_id, Rank local_rank) const {
    TT_FATAL(*subcontext_id == 0, "subcontext_id {} invalid for default single-context layout", *subcontext_id);
    TT_FATAL(
        *local_rank >= 0 && *local_rank < *size(),
        "local_rank {} out of range for sub-context 0 (size {})",
        *local_rank,
        *size());
    return local_rank;
}

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

bool DistributedContext::is_initialized() { return ContextImpl::is_initialized(); }

DistributedContextId DistributedContext::id() const { return id_; }

/* -------------------- DistributedContext ID generation --------------------- */
DistributedContextId DistributedContext::generate_unique_id() {
    static std::size_t next_id = 0;
    return DistributedContextId(next_id++);
}

}  // namespace tt::tt_metal::distributed::multihost
