/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/ppc/ppc_frontend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <utility>

#include "xenia/base/atomic.h"
#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/base/mutex.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/ppc/ppc_emit.h"
#include "xenia/cpu/ppc/ppc_opcode_info.h"
#include "xenia/cpu/ppc/ppc_translator.h"
#include "xenia/cpu/processor.h"

namespace xe {
namespace cpu {
namespace ppc {

void InitializeIfNeeded();
void CleanupOnShutdown();

void InitializeIfNeeded() {
  static bool has_initialized = false;
  if (has_initialized) {
    return;
  }
  has_initialized = true;

  RegisterEmitCategoryAltivec();
  RegisterEmitCategoryALU();
  RegisterEmitCategoryControl();
  RegisterEmitCategoryFPU();
  RegisterEmitCategoryMemory();

  atexit(CleanupOnShutdown);
}

void CleanupOnShutdown() {}

PPCFrontend::PPCFrontend(Processor* processor) : processor_(processor) {
  InitializeIfNeeded();
}

PPCFrontend::~PPCFrontend() {
  // Force cleanup now before we deinit.
  translator_pool_.Reset();
}

Memory* PPCFrontend::memory() const { return processor_->memory(); }

// Checks the state of the global lock and sets scratch to the current MSR
// value.
void CheckGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1) {
  auto global_mutex = reinterpret_cast<global_mutex_type*>(arg0);
  auto global_lock_count = reinterpret_cast<int32_t*>(arg1);
  std::lock_guard<global_mutex_type> lock(*global_mutex);
  ppc_context->scratch = *global_lock_count ? 0 : 0x8000;
}

// Enters the global lock. Safe to recursion.
void EnterGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1) {
  auto global_mutex = reinterpret_cast<global_mutex_type*>(arg0);
  auto global_lock_count = reinterpret_cast<int32_t*>(arg1);
  global_mutex->lock();
  xe::atomic_inc(global_lock_count);
}

// Leaves the global lock. Safe to recursion.
void LeaveGlobalLock(PPCContext* ppc_context, void* arg0, void* arg1) {
  auto global_mutex = reinterpret_cast<global_mutex_type*>(arg0);
  auto global_lock_count = reinterpret_cast<int32_t*>(arg1);
  auto new_lock_count = xe::atomic_dec(global_lock_count);
  assert_true(new_lock_count >= 0);
  global_mutex->unlock();
}

namespace {
struct GuestProbeRecord {
  uint64_t ns;
  uint32_t thread_id, pc, lr, r1, r3, r4, r5, r6, r7, r10, r11, r30, r31, w0,
      w1;
};
constexpr size_t kGuestProbeRing = 65536;
GuestProbeRecord g_guest_probes[kGuestProbeRing];
std::atomic<uint64_t> g_guest_probe_count{0};
std::atomic<bool> g_guest_probes_frozen{false};
}  // namespace

void FreezeGuestProbes() {
  g_guest_probes_frozen.store(true, std::memory_order_relaxed);
}

// XE_GUEST_PROBE_R31_RANGE=lo-hi (hex) records only hits whose r31 lies in
// [lo, hi): one lock or one object array out of a hot shared path.
static bool GuestProbeR31InRange(uint32_t r31) {
  static const std::pair<uint32_t, uint32_t> range = [] {
    const char* value = std::getenv("XE_GUEST_PROBE_R31_RANGE");
    if (!value) {
      return std::make_pair(uint32_t(0), uint32_t(0xFFFFFFFF));
    }
    char* end = nullptr;
    const uint32_t lo = uint32_t(std::strtoul(value, &end, 16));
    const uint32_t hi =
        end && *end == '-' ? uint32_t(std::strtoul(end + 1, nullptr, 16))
                           : lo + 1;
    return std::make_pair(lo, hi);
  }();
  return r31 >= range.first && r31 < range.second;
}

// The probed address arrives in scratch (written just before the call).
void GuestProbe(PPCContext* ppc_context, void* arg0, void* arg1) {
  if (g_guest_probes_frozen.load(std::memory_order_relaxed) ||
      !GuestProbeR31InRange(uint32_t(ppc_context->r[31]))) {
    return;
  }
  GuestProbeRecord r;
  r.ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
  r.thread_id = ppc_context->thread_id;
  r.pc = uint32_t(ppc_context->scratch);
  r.lr = uint32_t(ppc_context->lr);
  r.r1 = uint32_t(ppc_context->r[1]);
  r.r3 = uint32_t(ppc_context->r[3]);
  r.r4 = uint32_t(ppc_context->r[4]);
  r.r5 = uint32_t(ppc_context->r[5]);
  r.r6 = uint32_t(ppc_context->r[6]);
  r.r7 = uint32_t(ppc_context->r[7]);
  r.r10 = uint32_t(ppc_context->r[10]);
  r.r11 = uint32_t(ppc_context->r[11]);
  r.r30 = uint32_t(ppc_context->r[30]);
  r.r31 = uint32_t(ppc_context->r[31]);
  r.w0 = r.w1 = 0;
  if (r.r31 >= 0x10000 && r.r31 < 0xF0000000 && !(r.r31 & 3)) {
    // TranslateVirtual, not membase + address: the E0000000+ physical views
    // sit 0x1000 higher on 16 KB-page hosts.
    const uint8_t* p = ppc_context->TranslateVirtual(r.r31);
    r.w0 = xe::load_and_swap<uint32_t>(p);
    r.w1 = xe::load_and_swap<uint32_t>(p + 4);
  }
  const uint64_t n = g_guest_probe_count.fetch_add(1, std::memory_order_relaxed);
  g_guest_probes[n % kGuestProbeRing] = r;
}

std::string FormatGuestProbes(size_t max_records) {
  const uint64_t n = g_guest_probe_count.load(std::memory_order_acquire);
  const uint64_t count = std::min<uint64_t>({n, kGuestProbeRing, max_records});
  std::string out = fmt::format("GuestProbes: {} recorded, last {}\n", n, count);
  for (uint64_t i = n - count; i < n; ++i) {
    const GuestProbeRecord& r = g_guest_probes[i % kGuestProbeRing];
    out.append(fmt::format(
        "  {} t={:08X} pc={:08X} lr={:08X} r1={:08X} r3={:08X} r4={:08X} "
        "r5={:08X} r6={:08X} r7={:08X} r10={:08X} r11={:08X} r30={:08X} "
        "r31={:08X} "
        "[r31]={:08X} {:08X}\n",
        r.ns, r.thread_id, r.pc, r.lr, r.r1, r.r3, r.r4, r.r5, r.r6, r.r7,
        r.r10, r.r11, r.r30, r.r31, r.w0, r.w1));
  }
  return out;
}

void SyscallHandler(PPCContext* ppc_context, void* arg0, void* arg1) {
  uint64_t syscall_number = ppc_context->r[0];
  switch (syscall_number) {
    default:
      assert_unhandled_case(syscall_number);
      XELOGE("Unhandled syscall {}!", syscall_number);
      break;
#pragma warning(suppress : 4065)
  }
}

// jit_mtmsrd_atomic_lock: guest code brackets its lwarx/stwcx. sequences with
// mtmsrd r13 (interrupts off) ... mtmsrd rN (restore). Holding one host lock
// over that bracket makes the sequences mutually exclusive, so the
// value-compare stwcx. can't succeed where a real reservation would have been
// lost. Diagnostic only: a thread suspended inside the bracket stalls every
// other one.
namespace {
std::recursive_mutex g_atomic_section_mutex;
thread_local int32_t g_atomic_section_depth = 0;
}  // namespace

void EnterAtomicSection(PPCContext* ppc_context, void* arg0, void* arg1) {
  g_atomic_section_mutex.lock();
  ++g_atomic_section_depth;
}

void LeaveAtomicSection(PPCContext* ppc_context, void* arg0, void* arg1) {
  if (g_atomic_section_depth > 0) {
    --g_atomic_section_depth;
    g_atomic_section_mutex.unlock();
  }
}

bool PPCFrontend::Initialize() {
  void* arg0 = reinterpret_cast<void*>(&xe::global_critical_region::mutex());
  void* arg1 = reinterpret_cast<void*>(&builtins_.global_lock_count);
  builtins_.check_global_lock =
      processor_->DefineBuiltin("CheckGlobalLock", CheckGlobalLock, arg0, arg1);
  builtins_.enter_global_lock =
      processor_->DefineBuiltin("EnterGlobalLock", EnterGlobalLock, arg0, arg1);
  builtins_.leave_global_lock =
      processor_->DefineBuiltin("LeaveGlobalLock", LeaveGlobalLock, arg0, arg1);
  builtins_.syscall_handler = processor_->DefineBuiltin(
      "SyscallHandler", SyscallHandler, nullptr, nullptr);
  builtins_.guest_probe =
      processor_->DefineBuiltin("GuestProbe", GuestProbe, nullptr, nullptr);
  builtins_.enter_atomic_section = processor_->DefineBuiltin(
      "EnterAtomicSection", EnterAtomicSection, nullptr, nullptr);
  builtins_.leave_atomic_section = processor_->DefineBuiltin(
      "LeaveAtomicSection", LeaveAtomicSection, nullptr, nullptr);
  return true;
}

bool PPCFrontend::DeclareFunction(GuestFunction* function) {
  // chrispy: make sure we aren't declaring a function that is actually padding
  // data, this will mess up PPCScanner and is hard to debug wow, this halo
  // reach actually has branches into 0 opcodes, look into further
  // xenia_assert(*reinterpret_cast<const uint32_t*>(
  //                  this->memory()->TranslateVirtual(function->address())) !=
  //                  0);
  //  Could scan or something here.
  //  Could also check to see if it's a well-known function type and classify
  //  for later.
  //  Could also kick off a precompiler, since we know it's likely the function
  //  will be demanded soon.
  return true;
}

bool PPCFrontend::DefineFunction(GuestFunction* function,
                                 uint32_t debug_info_flags) {
  auto translator = translator_pool_.Allocate(this);
  bool result = translator->Translate(function, debug_info_flags);
  translator->Reset();
  translator_pool_.Release(translator);
  return result;
}

}  // namespace ppc
}  // namespace cpu
}  // namespace xe
