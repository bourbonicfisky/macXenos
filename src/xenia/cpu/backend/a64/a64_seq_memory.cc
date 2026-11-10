/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_sequences.h"

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_op.h"
#include "xenia/cpu/backend/a64/a64_seq_util.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/hir/instr.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/xex_module.h"

#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_switch.h>
#else
#include <thread>
#endif

DEFINE_int32(
    a64_delay_backoff_us, 0,
    "Guest spin-wait backoff. The guest's pause instruction (db16cyc) marks "
    "its spin loops: the Direct3D fence wait runs 32 of them between two "
    "reads of the fence. Once a guest thread has paused without a break for "
    "this many microseconds, every further 32 pauses give the core to any "
    "other runnable thread (a priority-depressing switch on macOS, a yield "
    "elsewhere) and continue at once when none is waiting. 0 = off: a pause "
    "is the host's yield instruction and nothing else.",
    "a64");
DEFINE_int32(
    a64_delay_backoff_sleep_us, 0,
    "With a64_delay_backoff_us: sleep this many microseconds per 32 pauses "
    "instead of offering the core, so a long guest spin stops using a core "
    "even when the host has idle ones. The guest then sees what it waits for "
    "up to this much later. 0 = offer the core only.",
    "a64");

DECLARE_bool(emit_mmio_aware_stores_for_recorded_exception_addresses);
DECLARE_bool(emit_inline_mmio_checks);

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

volatile int anchor_memory = 0;

static bool IsPossibleMMIOInstruction(A64Emitter& e, const hir::Instr* i) {
  if (!cvars::emit_mmio_aware_stores_for_recorded_exception_addresses) {
    return false;
  }
  uint32_t guest_address = i->GuestAddressFor();
  if (!guest_address) {
    return false;
  }

  auto* guest_module = e.GuestModule();
  if (!guest_module) {
    return false;
  }
  auto* flags = guest_module->GetInstructionAddressFlags(guest_address);
  return flags && flags->accessed_mmio;
}

// ============================================================================
// OPCODE_DELAY_EXECUTION
// ============================================================================
// XE_A64_DELAY_CENSUS=1 measures the guest's pause-marked spinning without
// changing it: how much thread time goes to continuous pause runs, by run
// length, logged every five seconds. A run's time includes its backoffs, so
// with the sleeping backoff the cores still spinning are spin minus backoff.
static bool DelayCensusEnabled() {
  static const bool s_enabled = [] {
    const char* v = std::getenv("XE_A64_DELAY_CENSUS");
    return v && v[0] == '1' && !v[1];
  }();
  return s_enabled;
}

// Pauses between two calls of DelayExecutionSlow. One pass of the Direct3D
// fence wait is 32 pauses.
static constexpr uint32_t kDelayBatch = 32;
// Two batches further apart than this belong to different spins.
static constexpr uint64_t kDelayRunGapNs = 50 * 1000;

// Finished spin runs by duration: under 100 us, 1 ms, 4 ms, 16 ms, and longer.
static constexpr size_t kDelayCensusBins = 5;
static std::atomic<uint64_t> delay_census_runs_[kDelayCensusBins];
static std::atomic<uint64_t> delay_census_run_ns_[kDelayCensusBins];
static std::atomic<uint64_t> delay_census_batches_;
static std::atomic<uint64_t> delay_census_backoffs_;
static std::atomic<uint64_t> delay_census_backoff_ns_;
static std::atomic<uint64_t> delay_census_next_log_ns_;
static std::atomic<uint64_t> delay_census_last_log_ns_;

static uint64_t DelayNowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

static void DelayCensusLog(uint64_t now_ns) {
  uint64_t next = delay_census_next_log_ns_.load(std::memory_order_relaxed);
  if (now_ns < next || !delay_census_next_log_ns_.compare_exchange_strong(
                           next, now_ns + 5'000'000'000ull,
                           std::memory_order_relaxed)) {
    return;
  }
  uint64_t last =
      delay_census_last_log_ns_.exchange(now_ns, std::memory_order_relaxed);
  if (!last) {
    return;
  }
  uint64_t runs[kDelayCensusBins], run_ns[kDelayCensusBins], total_ns = 0;
  for (size_t i = 0; i < kDelayCensusBins; ++i) {
    runs[i] = delay_census_runs_[i].exchange(0, std::memory_order_relaxed);
    run_ns[i] = delay_census_run_ns_[i].exchange(0, std::memory_order_relaxed);
    total_ns += run_ns[i];
  }
  double wall_ms = double(now_ns - last) / 1e6;
  XELOGI(
      "A64DelayCensus wall={:.0f}ms batches={} spin={:.0f}ms ({:.2f} cores) "
      "runs/ms by length <100us={}/{:.0f} <1ms={}/{:.0f} <4ms={}/{:.0f} "
      "<16ms={}/{:.0f} longer={}/{:.0f} backoffs={} backoff={:.0f}ms",
      wall_ms, delay_census_batches_.exchange(0, std::memory_order_relaxed),
      double(total_ns) / 1e6, double(total_ns) / 1e6 / wall_ms, runs[0],
      double(run_ns[0]) / 1e6, runs[1], double(run_ns[1]) / 1e6, runs[2],
      double(run_ns[2]) / 1e6, runs[3], double(run_ns[3]) / 1e6, runs[4],
      double(run_ns[4]) / 1e6,
      delay_census_backoffs_.exchange(0, std::memory_order_relaxed),
      double(delay_census_backoff_ns_.exchange(0, std::memory_order_relaxed)) /
          1e6);
}

// Every kDelayBatch pauses of a guest thread. x0 = PPCContext*.
static void DelayExecutionSlow(void* raw_context) {
  auto* bctx = reinterpret_cast<A64BackendContext*>(
      reinterpret_cast<intptr_t>(raw_context) - sizeof(A64BackendContext));
  const bool census = DelayCensusEnabled();
  uint64_t now_ns = DelayNowNs();
  if (now_ns - bctx->delay_last_ns > kDelayRunGapNs) {
    if (census && bctx->delay_last_ns) {
      uint64_t run_ns = bctx->delay_last_ns - bctx->delay_run_start_ns;
      size_t bin = run_ns < 100'000      ? 0
                   : run_ns < 1'000'000  ? 1
                   : run_ns < 4'000'000  ? 2
                   : run_ns < 16'000'000 ? 3
                                         : 4;
      delay_census_runs_[bin].fetch_add(1, std::memory_order_relaxed);
      delay_census_run_ns_[bin].fetch_add(run_ns, std::memory_order_relaxed);
    }
    bctx->delay_run_start_ns = now_ns;
  }
  const int64_t backoff_us = cvars::a64_delay_backoff_us;
  if (backoff_us > 0 &&
      now_ns - bctx->delay_run_start_ns >= uint64_t(backoff_us) * 1000) {
    const int64_t sleep_us = cvars::a64_delay_backoff_sleep_us;
#if XE_PLATFORM_MAC
    if (sleep_us > 0) {
      static const mach_timebase_info_data_t s_timebase = [] {
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        return info;
      }();
      mach_wait_until(mach_absolute_time() + uint64_t(sleep_us) * 1000 *
                                                 s_timebase.denom /
                                                 s_timebase.numer);
    } else {
      // As the Sleep(0) backoff in XThread::Delay: any runnable thread gets
      // the core, and with none waiting the spinner continues immediately.
      thread_switch(MACH_PORT_NULL, SWITCH_OPTION_DEPRESS, 1);
    }
#else
    if (sleep_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    } else {
      xe::threading::MaybeYield();
    }
#endif
    uint64_t after_ns = DelayNowNs();
    if (census) {
      delay_census_backoffs_.fetch_add(1, std::memory_order_relaxed);
      delay_census_backoff_ns_.fetch_add(after_ns - now_ns,
                                         std::memory_order_relaxed);
    }
    now_ns = after_ns;
  }
  bctx->delay_last_ns = now_ns;
  if (census) {
    delay_census_batches_.fetch_add(1, std::memory_order_relaxed);
    DelayCensusLog(now_ns);
  }
}

struct DELAY_EXECUTION
    : Sequence<DELAY_EXECUTION, I<OPCODE_DELAY_EXECUTION, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.yield();
    if (cvars::a64_delay_backoff_us <= 0 && !DelayCensusEnabled()) {
      return;
    }
    const uint32_t count_offset =
        static_cast<uint32_t>(offsetof(A64BackendContext, delay_count));
    auto& done = e.NewCachedLabel();
    e.ldr(e.w17, ptr(e.GetBackendCtxReg(), count_offset));
    e.add(e.w17, e.w17, 1);
    e.str(e.w17, ptr(e.GetBackendCtxReg(), count_offset));
    e.and_(e.w17, e.w17, kDelayBatch - 1);
    e.cbnz(e.w17, done);
    e.CallNativeSafe(reinterpret_cast<void*>(DelayExecutionSlow));
    e.L(done);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DELAY_EXECUTION, DELAY_EXECUTION);

// ============================================================================
// OPCODE_MEMORY_BARRIER
// ============================================================================
struct MEMORY_BARRIER
    : Sequence<MEMORY_BARRIER, I<OPCODE_MEMORY_BARRIER, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.dmb(Xbyak_aarch64::ISH);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MEMORY_BARRIER, MEMORY_BARRIER);

// ============================================================================
// OPCODE_LOAD_BARRIER
// ============================================================================
struct LOAD_BARRIER : Sequence<LOAD_BARRIER, I<OPCODE_LOAD_BARRIER, VoidOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    e.dmb(Xbyak_aarch64::ISHLD);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_BARRIER, LOAD_BARRIER);

// ============================================================================
// OPCODE_CACHE_CONTROL
// ============================================================================
struct CACHE_CONTROL
    : Sequence<CACHE_CONTROL,
               I<OPCODE_CACHE_CONTROL, VoidOp, I64Op, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    bool is_clflush = false, is_prefetch = false, is_prefetchw = false;
    switch (CacheControlType(i.instr->flags)) {
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_TOUCH:
        is_prefetch = true;
        break;
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_TOUCH_FOR_STORE:
        is_prefetchw = true;
        break;
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_STORE:
      case CacheControlType::CACHE_CONTROL_TYPE_DATA_STORE_AND_FLUSH:
        is_clflush = true;
        break;
      default:
        return;
    }
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.add(e.x0, e.GetMembaseReg(), addr);
    size_t cache_line_size = i.src2.value;
    if (is_clflush) {
      // dc civac, x0
      e.sys(0b011, 0b0111, 0b1110, 0b001, e.x0);
    }
    if (is_prefetch) {
      e.prfm(Xbyak_aarch64::PLDL1KEEP, ptr(e.x0));
    } else if (is_prefetchw) {
      e.prfm(Xbyak_aarch64::PSTL1KEEP, ptr(e.x0));
    }
    if (cache_line_size >= 128) {
      e.eor(e.x0, e.x0, 64);
      if (is_clflush) {
        // dc civac, x0
        e.sys(0b011, 0b0111, 0b1110, 0b001, e.x0);
      }
      if (is_prefetch) {
        e.prfm(Xbyak_aarch64::PLDL1KEEP, ptr(e.x0));
      } else if (is_prefetchw) {
        e.prfm(Xbyak_aarch64::PSTL1KEEP, ptr(e.x0));
      }
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CACHE_CONTROL, CACHE_CONTROL);

template <typename T, bool swap>
static void MMIOAwareStore(void* _ctx, unsigned int guestaddr, T value) {
  if (swap) {
    value = xe::byte_swap(value);
  }
  if (guestaddr >= 0xE0000000) {
    guestaddr += 0x1000;
  }
  auto ctx = reinterpret_cast<ppc::PPCContext*>(_ctx);
  auto gaddr = ctx->processor->memory()->LookupVirtualMappedRange(guestaddr);
  if (!gaddr) {
    *reinterpret_cast<T*>(ctx->virtual_membase + guestaddr) = value;
  } else {
    value = xe::byte_swap(value);
    gaddr->write(nullptr, gaddr->callback_context, guestaddr, value);
  }
}

template <typename T, bool swap>
static T MMIOAwareLoad(void* _ctx, unsigned int guestaddr) {
  T value;
  if (guestaddr >= 0xE0000000) {
    guestaddr += 0x1000;
  }
  auto ctx = reinterpret_cast<ppc::PPCContext*>(_ctx);
  auto gaddr = ctx->processor->memory()->LookupVirtualMappedRange(guestaddr);
  if (!gaddr) {
    value = *reinterpret_cast<T*>(ctx->virtual_membase + guestaddr);
    if (swap) {
      value = xe::byte_swap(value);
    }
  } else {
    // Handlers return what a byte-swapping load reads so a load without the
    // swap reverses it, matching the exception path.
    value = gaddr->read(nullptr, gaddr->callback_context, guestaddr);
    if (!swap) {
      value = xe::byte_swap(value);
    }
  }
  return value;
}

// ============================================================================
// OPCODE_LOAD
// ============================================================================
struct LOAD_I8 : Sequence<LOAD_I8, I<OPCODE_LOAD, I8Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.ldrb(i.dest, ptr(e.GetMembaseReg(), addr));
  }
};
struct LOAD_I16 : Sequence<LOAD_I16, I<OPCODE_LOAD, I16Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.ldrh(i.dest, ptr(e.GetMembaseReg(), addr));
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.rev16(i.dest, i.dest);
    }
  }
};
struct LOAD_I32 : Sequence<LOAD_I32, I<OPCODE_LOAD, I32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (IsPossibleMMIOInstruction(e, i.instr)) {
      void* mmio_fn = (void*)&MMIOAwareLoad<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareLoad<uint32_t, true>;
      }
      if (i.src1.is_constant) {
        e.mov(e.w1,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w1, WReg(i.src1.reg().getIdx()));
      }
      e.CallNativeSafe(mmio_fn);
      e.mov(i.dest, e.w0);
      return;
    }
    if (cvars::emit_inline_mmio_checks) {
      if (i.src1.is_constant) {
        e.mov(e.w17,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w17, WReg(i.src1.reg().getIdx()));
      }
      auto& normal_access = e.NewCachedLabel();
      auto& done = e.NewCachedLabel();
      e.mov(e.w0, 0x7FC00000u);
      e.cmp(e.w17, e.w0);
      e.b(LO, normal_access);
      e.mov(e.w0, 0x7FFFFFFFu);
      e.cmp(e.w17, e.w0);
      e.b(HI, normal_access);
      // MMIO path
      void* mmio_fn = (void*)&MMIOAwareLoad<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareLoad<uint32_t, true>;
      }
      e.mov(e.w1, e.w17);
      e.CallNativeSafe(mmio_fn);
      e.mov(i.dest, e.w0);
      e.b(done);
      e.L(normal_access);
      {
        auto addr = ComputeMemoryAddress(e, i.src1);
        e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
        if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
          e.rev(i.dest, i.dest);
        }
      }
      e.L(done);
    } else {
      auto addr = ComputeMemoryAddress(e, i.src1);
      e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        e.rev(i.dest, i.dest);
      }
    }
  }
};
struct LOAD_I64 : Sequence<LOAD_I64, I<OPCODE_LOAD, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.rev(i.dest, i.dest);
    }
  }
};
struct LOAD_F32 : Sequence<LOAD_F32, I<OPCODE_LOAD, F32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.ldr(e.w0, ptr(e.GetMembaseReg(), addr));
      e.rev(e.w0, e.w0);
      e.fmov(i.dest, e.w0);
    } else {
      e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
    }
  }
};
struct LOAD_F64 : Sequence<LOAD_F64, I<OPCODE_LOAD, F64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.ldr(e.x0, ptr(e.GetMembaseReg(), addr));
      e.rev(e.x0, e.x0);
      e.fmov(i.dest, e.x0);
    } else {
      e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
    }
  }
};
struct LOAD_V128 : Sequence<LOAD_V128, I<OPCODE_LOAD, V128Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      // Reverse bytes within each 32-bit word (PPC BE -> ARM64 LE).
      auto idx = i.dest.reg().getIdx();
      e.rev32(VReg16B(idx), VReg16B(idx));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD, LOAD_I8, LOAD_I16, LOAD_I32, LOAD_I64,
                     LOAD_F32, LOAD_F64, LOAD_V128);

// ============================================================================
// OPCODE_STORE
// ============================================================================
struct STORE_I8 : Sequence<STORE_I8, I<OPCODE_STORE, VoidOp, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.src2.is_constant) {
      e.mov(e.w17, static_cast<uint64_t>(i.src2.constant() & 0xFF));
      e.strb(e.w17, ptr(e.GetMembaseReg(), addr));
    } else {
      e.strb(i.src2, ptr(e.GetMembaseReg(), addr));
    }
  }
};
struct STORE_I16 : Sequence<STORE_I16, I<OPCODE_STORE, VoidOp, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      if (i.src2.is_constant) {
        uint16_t val = xe::byte_swap(static_cast<uint16_t>(i.src2.constant()));
        e.mov(e.w17, static_cast<uint64_t>(val));
      } else {
        e.rev16(e.w17, i.src2);
      }
      e.strh(e.w17, ptr(e.GetMembaseReg(), addr));
    } else {
      if (i.src2.is_constant) {
        e.mov(e.w17, static_cast<uint64_t>(i.src2.constant() & 0xFFFF));
        e.strh(e.w17, ptr(e.GetMembaseReg(), addr));
      } else {
        e.strh(i.src2, ptr(e.GetMembaseReg(), addr));
      }
    }
  }
};
struct STORE_I32 : Sequence<STORE_I32, I<OPCODE_STORE, VoidOp, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (IsPossibleMMIOInstruction(e, i.instr)) {
      void* mmio_fn = (void*)&MMIOAwareStore<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareStore<uint32_t, true>;
      }
      if (i.src1.is_constant) {
        e.mov(e.w1,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w1, WReg(i.src1.reg().getIdx()));
      }
      if (i.src2.is_constant) {
        e.mov(e.w2,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src2.constant())));
      } else {
        e.mov(e.w2, i.src2);
      }
      e.CallNativeSafe(mmio_fn);
      return;
    }
    if (cvars::emit_inline_mmio_checks) {
      if (i.src1.is_constant) {
        e.mov(e.w17,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w17, WReg(i.src1.reg().getIdx()));
      }
      auto& normal_access = e.NewCachedLabel();
      auto& done = e.NewCachedLabel();
      e.mov(e.w0, 0x7FC00000u);
      e.cmp(e.w17, e.w0);
      e.b(LO, normal_access);
      e.mov(e.w0, 0x7FFFFFFFu);
      e.cmp(e.w17, e.w0);
      e.b(HI, normal_access);
      // MMIO path — copy value to w2 before w1 in case src2 is in w1
      void* mmio_fn = (void*)&MMIOAwareStore<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareStore<uint32_t, true>;
      }
      if (i.src2.is_constant) {
        e.mov(e.w2,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src2.constant())));
      } else {
        e.mov(e.w2, i.src2);
      }
      e.mov(e.w1, e.w17);
      e.CallNativeSafe(mmio_fn);
      e.b(done);
      e.L(normal_access);
      {
        auto addr = ComputeMemoryAddress(e, i.src1);
        if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
          if (i.src2.is_constant) {
            uint32_t val =
                xe::byte_swap(static_cast<uint32_t>(i.src2.constant()));
            e.mov(e.w17, static_cast<uint64_t>(val));
          } else {
            e.rev(e.w17, i.src2);
          }
          e.str(e.w17, ptr(e.GetMembaseReg(), addr));
        } else {
          if (i.src2.is_constant) {
            e.mov(e.w17, static_cast<uint64_t>(
                             static_cast<uint32_t>(i.src2.constant())));
            e.str(e.w17, ptr(e.GetMembaseReg(), addr));
          } else {
            e.str(i.src2, ptr(e.GetMembaseReg(), addr));
          }
        }
      }
      e.L(done);
    } else {
      auto addr = ComputeMemoryAddress(e, i.src1);
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        if (i.src2.is_constant) {
          uint32_t val =
              xe::byte_swap(static_cast<uint32_t>(i.src2.constant()));
          e.mov(e.w17, static_cast<uint64_t>(val));
        } else {
          e.rev(e.w17, i.src2);
        }
        e.str(e.w17, ptr(e.GetMembaseReg(), addr));
      } else {
        if (i.src2.is_constant) {
          e.mov(e.w17, static_cast<uint64_t>(
                           static_cast<uint32_t>(i.src2.constant())));
          e.str(e.w17, ptr(e.GetMembaseReg(), addr));
        } else {
          e.str(i.src2, ptr(e.GetMembaseReg(), addr));
        }
      }
    }
  }
};
struct STORE_I64 : Sequence<STORE_I64, I<OPCODE_STORE, VoidOp, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      if (i.src2.is_constant) {
        uint64_t val = xe::byte_swap(static_cast<uint64_t>(i.src2.constant()));
        e.mov(e.x17, val);
      } else {
        e.rev(e.x17, i.src2);
      }
      e.str(e.x17, ptr(e.GetMembaseReg(), addr));
    } else {
      if (i.src2.is_constant) {
        e.mov(e.x17, static_cast<uint64_t>(i.src2.constant()));
        e.str(e.x17, ptr(e.GetMembaseReg(), addr));
      } else {
        e.str(i.src2, ptr(e.GetMembaseReg(), addr));
      }
    }
  }
};
struct STORE_F32 : Sequence<STORE_F32, I<OPCODE_STORE, VoidOp, I64Op, F32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      if (i.src2.is_constant) {
        uint32_t val =
            xe::byte_swap(static_cast<uint32_t>(i.src2.value->constant.i32));
        e.mov(e.w17, static_cast<uint64_t>(val));
      } else {
        e.fmov(e.w17, i.src2);
        e.rev(e.w17, e.w17);
      }
      e.str(e.w17, ptr(e.GetMembaseReg(), addr));
    } else {
      if (i.src2.is_constant) {
        e.mov(e.w17, static_cast<uint64_t>(i.src2.value->constant.i32));
        e.str(e.w17, ptr(e.GetMembaseReg(), addr));
      } else {
        e.str(i.src2, ptr(e.GetMembaseReg(), addr));
      }
    }
  }
};
struct STORE_F64 : Sequence<STORE_F64, I<OPCODE_STORE, VoidOp, I64Op, F64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      if (i.src2.is_constant) {
        uint64_t val =
            xe::byte_swap(static_cast<uint64_t>(i.src2.value->constant.i64));
        e.mov(e.x17, val);
      } else {
        e.fmov(e.x17, i.src2);
        e.rev(e.x17, e.x17);
      }
      e.str(e.x17, ptr(e.GetMembaseReg(), addr));
    } else {
      if (i.src2.is_constant) {
        e.mov(e.x17, static_cast<uint64_t>(i.src2.value->constant.i64));
        e.str(e.x17, ptr(e.GetMembaseReg(), addr));
      } else {
        e.str(i.src2, ptr(e.GetMembaseReg(), addr));
      }
    }
  }
};
struct STORE_V128
    : Sequence<STORE_V128, I<OPCODE_STORE, VoidOp, I64Op, V128Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // ComputeMemoryAddress may return x0, and LoadV128Const/SrcVReg clobber
    // x0, so save the address to x17 when we need to load a constant source.
    bool need_src_load =
        i.src2.is_constant ||
        (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP);
    auto addr = ComputeMemoryAddress(e, i.src1);
    if (need_src_load) {
      e.mov(e.x17, addr);
      addr = e.x17;
    }
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      // Reverse bytes within each 32-bit word, store via scratch v0.
      int idx = SrcVReg(e, i.src2, 0);
      e.rev32(VReg16B(0), VReg16B(idx));
      e.str(QReg(0), ptr(e.GetMembaseReg(), addr));
    } else {
      if (i.src2.is_constant) {
        LoadV128Const(e, 0, i.src2.constant());
        e.str(QReg(0), ptr(e.GetMembaseReg(), addr));
      } else {
        e.str(i.src2, ptr(e.GetMembaseReg(), addr));
      }
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE, STORE_I8, STORE_I16, STORE_I32, STORE_I64,
                     STORE_F32, STORE_F64, STORE_V128);

// ============================================================================
// OPCODE_LOAD_CLOCK
// ============================================================================
struct LOAD_CLOCK : Sequence<LOAD_CLOCK, I<OPCODE_LOAD_CLOCK, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // Call QueryGuestTickCount which updates the clock from host ticks.
    // Reading the cached pointer directly would return stale values for
    // consecutive mftb instructions.
    e.CallNative(reinterpret_cast<void*>(LoadClock));
    e.mov(i.dest, e.x0);
  }
  static uint64_t LoadClock(void* raw_context) {
    return Clock::QueryGuestTickCount();
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_CLOCK, LOAD_CLOCK);

// ============================================================================
// OPCODE_LOAD_OFFSET / OPCODE_STORE_OFFSET
// ============================================================================
struct LOAD_OFFSET_I8
    : Sequence<LOAD_OFFSET_I8, I<OPCODE_LOAD_OFFSET, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
    e.ldrb(i.dest, ptr(e.GetMembaseReg(), e.x0));
  }
};
struct LOAD_OFFSET_I16
    : Sequence<LOAD_OFFSET_I16, I<OPCODE_LOAD_OFFSET, I16Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
    e.ldrh(i.dest, ptr(e.GetMembaseReg(), e.x0));
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.rev16(i.dest, i.dest);
    }
  }
};
struct LOAD_OFFSET_I32
    : Sequence<LOAD_OFFSET_I32, I<OPCODE_LOAD_OFFSET, I32Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (IsPossibleMMIOInstruction(e, i.instr)) {
      void* mmio_fn = (void*)&MMIOAwareLoad<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareLoad<uint32_t, true>;
      }
      if (i.src1.is_constant) {
        e.mov(e.w1,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w1, WReg(i.src1.reg().getIdx()));
      }
      if (i.src2.is_constant) {
        e.mov(e.w17,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src2.constant())));
      } else {
        e.mov(e.w17, WReg(i.src2.reg().getIdx()));
      }
      e.add(e.w1, e.w1, e.w17);
      e.CallNativeSafe(mmio_fn);
      e.mov(i.dest, e.w0);
      return;
    }
    if (cvars::emit_inline_mmio_checks) {
      // Compute raw guest address (src1 + src2) in w17 for range check.
      if (i.src1.is_constant) {
        e.mov(e.w17,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w17, WReg(i.src1.reg().getIdx()));
      }
      if (i.src2.is_constant) {
        uint32_t offset = static_cast<uint32_t>(i.src2.constant());
        if (offset != 0) {
          e.mov(e.w0, static_cast<uint64_t>(offset));
          e.add(e.w17, e.w17, e.w0);
        }
      } else {
        e.add(e.w17, e.w17, WReg(i.src2.reg().getIdx()));
      }
      auto& normal_access = e.NewCachedLabel();
      auto& done = e.NewCachedLabel();
      e.mov(e.w0, 0x7FC00000u);
      e.cmp(e.w17, e.w0);
      e.b(LO, normal_access);
      e.mov(e.w0, 0x7FFFFFFFu);
      e.cmp(e.w17, e.w0);
      e.b(HI, normal_access);
      // MMIO path
      void* mmio_fn = (void*)&MMIOAwareLoad<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareLoad<uint32_t, true>;
      }
      e.mov(e.w1, e.w17);
      e.CallNativeSafe(mmio_fn);
      e.mov(i.dest, e.w0);
      e.b(done);
      e.L(normal_access);
      {
        AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
        e.ldr(i.dest, ptr(e.GetMembaseReg(), e.x0));
        if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
          e.rev(i.dest, i.dest);
        }
      }
      e.L(done);
    } else {
      AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
      e.ldr(i.dest, ptr(e.GetMembaseReg(), e.x0));
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        e.rev(i.dest, i.dest);
      }
    }
  }
};
struct LOAD_OFFSET_I64
    : Sequence<LOAD_OFFSET_I64, I<OPCODE_LOAD_OFFSET, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
    e.ldr(i.dest, ptr(e.GetMembaseReg(), e.x0));
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      e.rev(i.dest, i.dest);
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_OFFSET, LOAD_OFFSET_I8, LOAD_OFFSET_I16,
                     LOAD_OFFSET_I32, LOAD_OFFSET_I64);

struct STORE_OFFSET_I8
    : Sequence<STORE_OFFSET_I8,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I8Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
    if (i.src3.is_constant) {
      e.mov(e.w17, static_cast<uint64_t>(i.src3.constant() & 0xFF));
      e.strb(e.w17, ptr(e.GetMembaseReg(), e.x0));
    } else {
      e.strb(i.src3, ptr(e.GetMembaseReg(), e.x0));
    }
  }
};
struct STORE_OFFSET_I16
    : Sequence<STORE_OFFSET_I16,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I16Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      if (i.src3.is_constant) {
        uint16_t val = xe::byte_swap(static_cast<uint16_t>(i.src3.constant()));
        e.mov(e.w17, static_cast<uint64_t>(val));
      } else {
        e.rev16(e.w17, i.src3);
      }
      e.strh(e.w17, ptr(e.GetMembaseReg(), e.x0));
    } else {
      if (i.src3.is_constant) {
        e.mov(e.w17, static_cast<uint64_t>(i.src3.constant() & 0xFFFF));
        e.strh(e.w17, ptr(e.GetMembaseReg(), e.x0));
      } else {
        e.strh(i.src3, ptr(e.GetMembaseReg(), e.x0));
      }
    }
  }
};
struct STORE_OFFSET_I32
    : Sequence<STORE_OFFSET_I32,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    if (IsPossibleMMIOInstruction(e, i.instr)) {
      void* mmio_fn = (void*)&MMIOAwareStore<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareStore<uint32_t, true>;
      }
      if (i.src1.is_constant) {
        e.mov(e.w1,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w1, WReg(i.src1.reg().getIdx()));
      }
      if (i.src2.is_constant) {
        e.mov(e.w17,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src2.constant())));
      } else {
        e.mov(e.w17, WReg(i.src2.reg().getIdx()));
      }
      e.add(e.w1, e.w1, e.w17);
      if (i.src3.is_constant) {
        e.mov(e.w2,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src3.constant())));
      } else {
        e.mov(e.w2, i.src3);
      }
      e.CallNativeSafe(mmio_fn);
      return;
    }
    if (cvars::emit_inline_mmio_checks) {
      // Compute raw guest address (src1 + src2) in w17 for range check.
      if (i.src1.is_constant) {
        e.mov(e.w17,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src1.constant())));
      } else {
        e.mov(e.w17, WReg(i.src1.reg().getIdx()));
      }
      if (i.src2.is_constant) {
        uint32_t offset = static_cast<uint32_t>(i.src2.constant());
        if (offset != 0) {
          e.mov(e.w0, static_cast<uint64_t>(offset));
          e.add(e.w17, e.w17, e.w0);
        }
      } else {
        e.add(e.w17, e.w17, WReg(i.src2.reg().getIdx()));
      }
      auto& normal_access = e.NewCachedLabel();
      auto& done = e.NewCachedLabel();
      e.mov(e.w0, 0x7FC00000u);
      e.cmp(e.w17, e.w0);
      e.b(LO, normal_access);
      e.mov(e.w0, 0x7FFFFFFFu);
      e.cmp(e.w17, e.w0);
      e.b(HI, normal_access);
      // MMIO path — copy value to w2 before w1 in case src3 is in w1
      void* mmio_fn = (void*)&MMIOAwareStore<uint32_t, false>;
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        mmio_fn = (void*)&MMIOAwareStore<uint32_t, true>;
      }
      if (i.src3.is_constant) {
        e.mov(e.w2,
              static_cast<uint64_t>(static_cast<uint32_t>(i.src3.constant())));
      } else {
        e.mov(e.w2, i.src3);
      }
      e.mov(e.w1, e.w17);
      e.CallNativeSafe(mmio_fn);
      e.b(done);
      e.L(normal_access);
      {
        AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
        if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
          if (i.src3.is_constant) {
            uint32_t val =
                xe::byte_swap(static_cast<uint32_t>(i.src3.constant()));
            e.mov(e.w17, static_cast<uint64_t>(val));
          } else {
            e.rev(e.w17, i.src3);
          }
          e.str(e.w17, ptr(e.GetMembaseReg(), e.x0));
        } else {
          if (i.src3.is_constant) {
            e.mov(e.w17, static_cast<uint64_t>(
                             static_cast<uint32_t>(i.src3.constant())));
            e.str(e.w17, ptr(e.GetMembaseReg(), e.x0));
          } else {
            e.str(i.src3, ptr(e.GetMembaseReg(), e.x0));
          }
        }
      }
      e.L(done);
    } else {
      AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
      if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
        if (i.src3.is_constant) {
          uint32_t val =
              xe::byte_swap(static_cast<uint32_t>(i.src3.constant()));
          e.mov(e.w17, static_cast<uint64_t>(val));
        } else {
          e.rev(e.w17, i.src3);
        }
        e.str(e.w17, ptr(e.GetMembaseReg(), e.x0));
      } else {
        if (i.src3.is_constant) {
          e.mov(e.w17, static_cast<uint64_t>(
                           static_cast<uint32_t>(i.src3.constant())));
          e.str(e.w17, ptr(e.GetMembaseReg(), e.x0));
        } else {
          e.str(i.src3, ptr(e.GetMembaseReg(), e.x0));
        }
      }
    }
  }
};
struct STORE_OFFSET_I64
    : Sequence<STORE_OFFSET_I64,
               I<OPCODE_STORE_OFFSET, VoidOp, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    AddGuestMemoryOffset(e, ComputeMemoryAddress(e, i.src1), i.src2);
    if (i.instr->flags & LoadStoreFlags::LOAD_STORE_BYTE_SWAP) {
      if (i.src3.is_constant) {
        uint64_t val = xe::byte_swap(static_cast<uint64_t>(i.src3.constant()));
        e.mov(e.x17, val);
      } else {
        e.rev(e.x17, i.src3);
      }
      e.str(e.x17, ptr(e.GetMembaseReg(), e.x0));
    } else {
      if (i.src3.is_constant) {
        e.mov(e.x17, static_cast<uint64_t>(i.src3.constant()));
        e.str(e.x17, ptr(e.GetMembaseReg(), e.x0));
      } else {
        e.str(i.src3, ptr(e.GetMembaseReg(), e.x0));
      }
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE_OFFSET, STORE_OFFSET_I8, STORE_OFFSET_I16,
                     STORE_OFFSET_I32, STORE_OFFSET_I64);

// ============================================================================
// OPCODE_MEMSET
// ============================================================================
static const bool zva_enable = (xe_cpu_mrs(DCZID_EL0) & 0b1'0000) == 0;
static const uint64_t zva_length = (4ULL << (xe_cpu_mrs(DCZID_EL0) & 0b0'1111));

struct MEMSET_I64
    : Sequence<MEMSET_I64, I<OPCODE_MEMSET, VoidOp, I64Op, I8Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.is_constant);
    assert_true(i.src3.is_constant);
    assert_true(i.src2.constant() == 0);
    // memset(membase + guest_addr, 0, length)
    // Only used by dcbz/dcbz128: constant zero value, constant aligned size.
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.add(e.x0, e.GetMembaseReg(), addr);
    const uint64_t len = i.src3.constant();
    uint64_t off = 0;

    // Use `dc zva` if it writes more bytes at a time than STP
    if (zva_enable && len >= zva_length && zva_length > 16) {
      for (; off + zva_length <= len; off += zva_length) {
        // dc zva, x0
        e.sys(0b011, 0b0111, 0b0100, 0b001, e.x0);
        if (off + zva_length < len) {
          e.add(e.x0, e.x0, zva_length);
        }
      }
    }

    // Inline with STP xzr, xzr pairs (16 bytes each)
    for (; off + 16 <= len; off += 16) {
      e.stp(e.xzr, e.xzr, AdrPostImm(e.x0, 16));
    }
    // Handle remaining bytes (0-15)
    if (off + 8 <= len) {
      e.str(e.xzr, AdrPostImm(e.x0, 8));
      off += 8;
    }
    if (off + 4 <= len) {
      e.str(e.wzr, AdrPostImm(e.x0, 4));
      off += 4;
    }
    // Byte loop for any remaining 0-3 bytes
    for (; off + 1 <= len; off += 1) {
      e.strb(e.wzr, AdrPostImm(e.x0, 1));
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_MEMSET, MEMSET_I64);

// ============================================================================
// ============================================================================
// OPCODE_ATOMIC_COMPARE_EXCHANGE
// ============================================================================
struct ATOMIC_COMPARE_EXCHANGE_I32
    : Sequence<ATOMIC_COMPARE_EXCHANGE_I32,
               I<OPCODE_ATOMIC_COMPARE_EXCHANGE, I8Op, I64Op, I32Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    // Compute full host address (ldxr/stxr need base-only [Xn] addressing).
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.add(e.x4, e.GetMembaseReg(), addr);
    // src2 = expected (use w5), src3 = desired (use w6).
    if (i.src2.is_constant) {
      e.mov(e.w5,
            static_cast<uint64_t>(static_cast<uint32_t>(i.src2.constant())));
    } else {
      e.mov(e.w5, i.src2);
    }
    if (i.src3.is_constant) {
      e.mov(e.w6,
            static_cast<uint64_t>(static_cast<uint32_t>(i.src3.constant())));
    } else {
      e.mov(e.w6, i.src3);
    }

    if (e.IsFeatureEnabled(kA64EmitLSE)) {
      e.mov(e.w0, e.w5);
      e.casal(e.w5, e.w6, ptr(e.x4));
      e.cmp(e.w5, e.w0);
      e.cset(i.dest, Xbyak_aarch64::EQ);
      return;
    }

    auto& retry = e.NewCachedLabel();
    auto& fail = e.NewCachedLabel();
    auto& done = e.NewCachedLabel();
    e.L(retry);
    e.ldaxr(e.w2, ptr(e.x4));
    e.cmp(e.w2, e.w5);
    e.b(Xbyak_aarch64::NE, fail);
    e.stlxr(e.w3, e.w6, ptr(e.x4));
    e.cbnz(e.w3, retry);
    e.mov(i.dest, 1);
    e.b(done);
    e.L(fail);
    e.clrex(15);
    e.mov(i.dest, 0);
    e.L(done);
  }
};
struct ATOMIC_COMPARE_EXCHANGE_I64
    : Sequence<ATOMIC_COMPARE_EXCHANGE_I64,
               I<OPCODE_ATOMIC_COMPARE_EXCHANGE, I8Op, I64Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.add(e.x4, e.GetMembaseReg(), addr);
    if (i.src2.is_constant) {
      e.mov(e.x5, static_cast<uint64_t>(i.src2.constant()));
    } else {
      e.mov(e.x5, i.src2);
    }
    if (i.src3.is_constant) {
      e.mov(e.x6, static_cast<uint64_t>(i.src3.constant()));
    } else {
      e.mov(e.x6, i.src3);
    }

    if (e.IsFeatureEnabled(kA64EmitLSE)) {
      e.mov(e.x0, e.x5);
      e.casal(e.x5, e.x6, ptr(e.x4));
      e.cmp(e.x5, e.x0);
      e.cset(i.dest, Xbyak_aarch64::EQ);
      return;
    }

    auto& retry = e.NewCachedLabel();
    auto& fail = e.NewCachedLabel();
    auto& done = e.NewCachedLabel();
    e.L(retry);
    e.ldaxr(e.x2, ptr(e.x4));
    e.cmp(e.x2, e.x5);
    e.b(Xbyak_aarch64::NE, fail);
    e.stlxr(e.w3, e.x6, ptr(e.x4));
    e.cbnz(e.w3, retry);
    e.mov(i.dest, 1);
    e.b(done);
    e.L(fail);
    e.clrex(15);
    e.mov(i.dest, 0);
    e.L(done);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_ATOMIC_COMPARE_EXCHANGE,
                     ATOMIC_COMPARE_EXCHANGE_I32, ATOMIC_COMPARE_EXCHANGE_I64);

// ============================================================================
// OPCODE_LOAD_MMIO / OPCODE_STORE_MMIO
// ============================================================================
struct LOAD_MMIO_I32
    : Sequence<LOAD_MMIO_I32, I<OPCODE_LOAD_MMIO, I32Op, OffsetOp, OffsetOp>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto mmio_range = reinterpret_cast<MMIORange*>(i.src1.value);
    auto read_address = uint32_t(i.src2.value);
    // CallNativeSafe: thunk sets x0=PPCContext*, x1/x2/x3 pass through.
    // MMIOReadCallback(void* ppc_ctx, void* callback_ctx, uint32_t addr).
    e.mov(e.x1, uint64_t(mmio_range->callback_context));
    e.mov(e.w2, static_cast<uint64_t>(read_address));
    e.CallNativeSafe(reinterpret_cast<void*>(mmio_range->read));
    e.rev(e.w0, e.w0);
    e.mov(i.dest, e.w0);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_LOAD_MMIO, LOAD_MMIO_I32);

struct STORE_MMIO_I32
    : Sequence<STORE_MMIO_I32,
               I<OPCODE_STORE_MMIO, VoidOp, OffsetOp, OffsetOp, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    auto mmio_range = reinterpret_cast<MMIORange*>(i.src1.value);
    auto write_address = uint32_t(i.src2.value);
    // CallNativeSafe: thunk sets x0=PPCContext*, x1/x2/x3 pass through.
    // MMIOWriteCallback(void* ppc_ctx, void* callback_ctx, uint32_t addr,
    //                   uint32_t value).
    e.mov(e.x1, uint64_t(mmio_range->callback_context));
    e.mov(e.w2, static_cast<uint64_t>(write_address));
    if (i.src3.is_constant) {
      e.mov(e.w3, static_cast<uint64_t>(
                      xe::byte_swap(static_cast<uint32_t>(i.src3.constant()))));
    } else {
      e.mov(e.w3, i.src3);
      e.rev(e.w3, e.w3);
    }
    e.CallNativeSafe(reinterpret_cast<void*>(mmio_range->write));
  }
};
EMITTER_OPCODE_TABLE(OPCODE_STORE_MMIO, STORE_MMIO_I32);

// ============================================================================
// OPCODE_RESERVED_LOAD / OPCODE_RESERVED_STORE
// ============================================================================
// A per-granule generation counter makes a stwcx. on one thread invalidate
// other threads' reservations, which a compare-and-swap on the value alone
// cannot (ABA). These sequences share their state with the host helpers in
// a64_backend.cc, so a reservation taken by either can be completed by the
// other. Only x22-x28 and v4-v31 hold guest values, x0-x17 are scratch.
static constexpr uint32_t kReserveFlagBit = 1u << kA64BackendHasReserveBit;

static constexpr uint32_t BackendCtxOffset(size_t offset) {
  return static_cast<uint32_t>(offset);
}

// The guest (untranslated) address the reservation is keyed on. Non-constant
// sources sit in a callee-saved register ComputeMemoryAddress leaves alone.
static WReg GuestAddressReg(A64Emitter& e, const I64Op& src,
                            const WReg& scratch) {
  if (src.is_constant) {
    e.mov(scratch,
          static_cast<uint64_t>(static_cast<uint32_t>(src.constant())));
    return scratch;
  }
  return WReg(src.reg().getIdx());
}

static void EmitTryAcquireReservation(A64Emitter& e,
                                      const WReg& guest_address) {
  const auto bctx = e.GetBackendCtxReg();
  e.ldr(e.x8, ptr(bctx, BackendCtxOffset(
                            offsetof(A64BackendContext, reserve_helper_))));
  e.ubfx(e.w9, guest_address, A64_RESERVE_GRANULE_SHIFT,
         A64_RESERVE_ENTRY_BITS);
  e.add(e.x8, e.x8, e.x9, Xbyak_aarch64::LSL, 2);
  // snapshot the generation first, the acquire pins the value read after
  e.ldar(e.w10, ptr(e.x8));
  e.str(guest_address, ptr(bctx, BackendCtxOffset(offsetof(A64BackendContext,
                                                           reserve_address))));
  // lwarx replaces any reservation this thread already held
  e.ldr(e.w11, ptr(bctx, BackendCtxOffset(offsetof(A64BackendContext, flags))));
  e.orr(e.w11, e.w11, kReserveFlagBit);
  e.str(e.w11, ptr(bctx, BackendCtxOffset(offsetof(A64BackendContext, flags))));

  // a stwcx. has the granule claimed, AwaitReservation takes the generation
  // once it publishes
  auto& unclaimed = e.NewCachedLabel();
  auto& acquired = e.NewCachedLabel();
  e.tbz(e.w10, 0, unclaimed);
  e.mov(e.w1, guest_address);
  e.CallNativeSafe(reinterpret_cast<void*>(A64Backend::AwaitReservation));
  e.b(acquired);
  e.L(unclaimed);
  e.str(e.w10, ptr(bctx, BackendCtxOffset(
                             offsetof(A64BackendContext, reserve_generation))));
  e.L(acquired);
}

struct RESERVED_LOAD_I32
    : Sequence<RESERVED_LOAD_I32, I<OPCODE_RESERVED_LOAD, I32Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const WReg guest_address = GuestAddressReg(e, i.src1, e.w12);
    // the address after the acquire, whose wait clobbers the scratch registers
    EmitTryAcquireReservation(e, guest_address);
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
    e.mov(e.w0, i.dest);
    e.str(e.x0, ptr(e.GetBackendCtxReg(),
                    BackendCtxOffset(
                        offsetof(A64BackendContext, cached_reserve_value_))));
  }
};
struct RESERVED_LOAD_I64
    : Sequence<RESERVED_LOAD_I64, I<OPCODE_RESERVED_LOAD, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const WReg guest_address = GuestAddressReg(e, i.src1, e.w12);
    EmitTryAcquireReservation(e, guest_address);
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.ldr(i.dest, ptr(e.GetMembaseReg(), addr));
    e.str(i.dest, ptr(e.GetBackendCtxReg(),
                      BackendCtxOffset(
                          offsetof(A64BackendContext, cached_reserve_value_))));
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RESERVED_LOAD, RESERVED_LOAD_I32,
                     RESERVED_LOAD_I64);

// x1 = host address, x2 = value to store, both set up by the caller.
// Writes 1 to dest if the store landed, 0 otherwise.
static void EmitReservedStore(A64Emitter& e, const I8Op& dest,
                              const WReg& guest_address, bool bit64) {
  if (guest_address.getIdx() != e.w12.getIdx()) {
    e.mov(e.w12, guest_address);
  }
  e.mov(e.x9, reinterpret_cast<uint64_t>(
                  e.backend()->reserved_store_helper(bit64)));
  e.blr(e.x9);
  e.mov(WReg(dest.reg().getIdx()), e.w0);
}

struct RESERVED_STORE_I32
    : Sequence<RESERVED_STORE_I32,
               I<OPCODE_RESERVED_STORE, I8Op, I64Op, I32Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const WReg guest_address = GuestAddressReg(e, i.src1, e.w12);
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.add(e.x1, e.GetMembaseReg(), addr);
    if (i.src2.is_constant) {
      e.mov(e.w2,
            static_cast<uint64_t>(static_cast<uint32_t>(i.src2.constant())));
    } else {
      e.mov(e.w2, WReg(i.src2.reg().getIdx()));
    }
    EmitReservedStore(e, i.dest, guest_address, false);
  }
};
struct RESERVED_STORE_I64
    : Sequence<RESERVED_STORE_I64,
               I<OPCODE_RESERVED_STORE, I8Op, I64Op, I64Op>> {
  static void Emit(A64Emitter& e, const EmitArgType& i) {
    const WReg guest_address = GuestAddressReg(e, i.src1, e.w12);
    auto addr = ComputeMemoryAddress(e, i.src1);
    e.add(e.x1, e.GetMembaseReg(), addr);
    if (i.src2.is_constant) {
      e.mov(e.x2, static_cast<uint64_t>(i.src2.constant()));
    } else {
      e.mov(e.x2, XReg(i.src2.reg().getIdx()));
    }
    EmitReservedStore(e, i.dest, guest_address, true);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RESERVED_STORE, RESERVED_STORE_I32,
                     RESERVED_STORE_I64);

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
