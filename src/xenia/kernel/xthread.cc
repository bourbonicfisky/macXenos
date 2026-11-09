/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xthread.h"

#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/thread_switch.h>
#endif

#if !XE_PLATFORM_WIN32
#include <pthread.h>
#include <signal.h>
#include <cstdlib>
#include <cstring>
#endif

#include "xenia/base/byte_stream.h"
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/processor.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/util/guest_suspend.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"

DEFINE_bool(ignore_thread_priorities, false,
            "Ignores game-specified thread priorities.", "Kernel");
UPDATE_from_bool(ignore_thread_priorities, 2026, 4, 9, 12, true);
DEFINE_bool(ignore_thread_affinities, true,
            "Ignores game-specified thread affinities.", "Kernel");

#if XE_PLATFORM_MAC
// macOS-only guest-thread QoS scheduling policy. On Apple Silicon the scheduler
// parks the emulator's busiest guest thread on efficiency cores (measured: the
// JC2 critical thread ran ~0% on P-cores) because Xenia sets no QoS class. These
// cvars let selected guest CPU threads self-tag an Apple QoS class so the
// scheduler keeps them on performance cores. Default 'all' since 2026-10-23:
// every sweep and published number was measured with it (the test machines'
// configs had it set), and a fresh M4 Pro with 'off' ran Mafia II at 35.6 fps
// vs 150.6 and GTA IV at 39.8 vs 60.0. Titles that measure better with 'off'
// get it from their built-in game config.
// Intel Macs have no efficiency cores, so the class only raises priority, and
// there 'all' cost the titles that wait on their own worker threads: Mac Pro
// (Xeon W-3245M, Radeon Pro 580X) Just Cause 2 26.7 -> 29.7 fps and Resident
// Evil 5 26.2 -> 30.0 with 'off', Skate 2 29.5 -> 30.0, GTA IV 42.2 -> 43.2,
// no title slower over 19 (2026-11-03). x86_64 builds default to 'off'.
#if XE_ARCH_AMD64
#define XE_GUEST_THREAD_QOS_DEFAULT "off"
#else
#define XE_GUEST_THREAD_QOS_DEFAULT "all"
#endif
DEFINE_string(
    guest_thread_qos, XE_GUEST_THREAD_QOS_DEFAULT,
    "macOS only: Apple QoS scheduling policy for guest CPU threads. "
    "'off' (no QoS calls; default on Intel Macs) | 'all' (default on Apple "
    "silicon, every guest XThread) | 'main' "
    "(entry-point thread only) | 'priority' (guest priority >= "
    "guest_thread_qos_priority_threshold). 'hot' (hot-after-warmup) is reserved "
    "but not yet implemented.",
    "CPU");
// The class default is the one every A/B measured (2026-10-15: Mafia II 25 ->
// 56 fps on the M1 Max, 34 -> 57 on the M4 with priority/all). On 4-P-core
// Macs every policy cost Just Cause 2 and Resident Evil 5 1-2 fps then; such
// titles set guest_thread_qos = "off" in their game config.
DEFINE_string(guest_thread_qos_class, "user_interactive",
              "macOS only: QoS class applied when guest_thread_qos selects a "
              "thread: 'user_initiated' | 'user_interactive' (default).",
              "CPU");
UPDATE_from_string(guest_thread_qos_class, 2026, 10, 16, 0, "user_initiated");
UPDATE_from_string(guest_thread_qos, 2026, 10, 23, 8, "off");
#if XE_ARCH_AMD64
UPDATE_from_string(guest_thread_qos, 2026, 11, 6, 0, "all");
#endif
DEFINE_int32(guest_thread_qos_priority_threshold, 1,
             "macOS only: with guest_thread_qos=priority, tag guest threads "
             "whose guest priority is >= this value.",
             "CPU");

// Map the guest_thread_qos policy to an Apple QoS class code for a guest thread
// (0 = none, 1 = user_initiated, 2 = user_interactive). Applied at thread
// creation via CreationParameters::qos_class in Create() below. It must be set
// at creation, not self-tagged after start: once a guest thread is placed in
// SCHED_FIFO by the priority mapping it is opted out of QoS management and a
// self-tag returns EPERM. Only XThreads run this, so GPU-command / host worker
// threads are never affected. hot-after-warmup is not implemented (would need a
// monitor plus pthread_override_qos_class_start_np).
static int32_t GuestThreadQosClass(bool main_thread, int32_t guest_priority) {
  const std::string& policy = cvars::guest_thread_qos;
  bool selected;
  if (policy == "all") {
    selected = true;
  } else if (policy == "main") {
    selected = main_thread;
  } else if (policy == "priority") {
    selected = guest_priority >= cvars::guest_thread_qos_priority_threshold;
  } else {
    if (policy == "hot") {
      static bool warned = false;
      if (!warned) {
        warned = true;
        XELOGW(
            "guest_thread_qos=hot is not implemented yet; no QoS applied. Use "
            "all/main/priority.");
      }
    }
    return 0;  // "off" (default), "hot", or unknown -> no QoS.
  }
  if (!selected) {
    return 0;
  }
  return cvars::guest_thread_qos_class == "user_interactive" ? 2 : 1;
}
#endif  // XE_PLATFORM_MAC

DEFINE_bool(
    guest_wait_profile, false,
    "CPU-perf analysis (Layer 1 Piece B): per-guest-thread wait attribution. "
    "Counts and times object blocking (event/mutant/semaphore/timer) vs "
    "Sleep(0)/yield spin-polling, and records which objects each thread blocks "
    "on. Dumped to the log at emulator shutdown. Default off (zero steady-state "
    "cost). Use it to find WHAT a dependency-bound guest thread waits on.",
    "CPU");

#if 0
DEFINE_int64(stack_size_multiplier_hack, 1,
             "A hack for games with setjmp/longjmp issues.", "Kernel");
DEFINE_int64(main_xthread_stack_size_multiplier_hack, 1,
             "A hack for games with setjmp/longjmp issues.", "Kernel");
#endif

DEFINE_int64(
    spin0_sleep_us, 0,
    "Diagnostic / candidate fix (default 0 = current MaybeYield): when a guest "
    "thread calls KeDelayExecutionThread(0)/Sleep(0) (non-alertable, the busy "
    "poll-loop yield), do a real NanoSleep of this many microseconds instead of "
    "sched_yield. Throttles the DOA5 load-stall poll loops that spam NtSetEvent "
    "and contend the kernel locks/scheduler. Try 100. Revert/redesign before any "
    "fix ships.",
    "Kernel");

DEFINE_int64(
    macos_spin0_backoff_us, 250,
    "macOS-only: adaptive busy-poll backoff. The guest's lock-free pump loops "
    "spam Sleep(0); on POSIX that is sched_yield, which keeps the spinner "
    "runnable and on Apple Silicon's asymmetric scheduler starves the guest job "
    "dispatcher (the DOA5 load-stall freeze). After a short burst of cheap "
    "yields, a runaway Sleep(0) spinner instead sleeps this many microseconds so "
    "the dispatcher gets the core. 0 disables (pure sched_yield, pre-fix "
    "behavior). 250 is the validated value; tune via 100/250/500.",
    "Kernel");

DEFINE_bool(
    macos_spin0_depress, true,
    "macOS-only: the Sleep(0) backoff gives the core away with a "
    "priority-depressing thread_switch (any runnable thread runs first, and "
    "the spinner resumes at once when none is waiting) instead of sleeping "
    "macos_spin0_backoff_us. A fixed sleep wakes a guest frame limiter that "
    "polls the performance counter between Sleep(0)s up to one slice late, "
    "and limiters that restart from the late time (Ultra Street Fighter IV) "
    "drift below 60 fps. false restores the sleep.",
    "Kernel");

#if XE_PLATFORM_MAC
// Adaptive Sleep(0) backoff tuning. After this many consecutive tight Sleep(0)
// calls, escalate from sched_yield to the bounded NanoSleep; reset the run if
// more than this many ms elapse between Sleep(0)s (thread isn't tight-spinning).
static constexpr uint32_t kSpin0YieldBurst = 8;
static constexpr uint64_t kSpin0ResetMs = 4;
#endif

namespace xe {
namespace kernel {

const uint32_t XAPC::kSize;
const uint32_t XAPC::kDummyKernelRoutine;
const uint32_t XAPC::kDummyRundownRoutine;
const uint32_t XAPC::kTimerKernelRoutine;

using namespace xe::literals;

// Atomic: guest threads and the emulator's host threads are constructed on
// different threads, and a duplicate id is visible to the guest (KTHREAD
// thread_id at 0x14C, which some titles' recursive locks use as the owner
// token, e.g. Skyrim's 82311B78).
std::atomic<uint32_t> next_xthread_id_{0};

XThread::XThread(KernelState* kernel_state)
    : XObject(kernel_state, kObjectType), guest_thread_(true) {}

XThread::XThread(KernelState* kernel_state, uint32_t stack_size,
                 uint32_t xapi_thread_startup, uint32_t start_address,
                 uint32_t start_context, uint32_t creation_flags,
                 bool guest_thread, bool main_thread, uint32_t guest_process)
    : XObject(kernel_state, kObjectType, !guest_thread),
      thread_id_(next_xthread_id_.fetch_add(1) + 1),
      guest_thread_(guest_thread),
      main_thread_(main_thread) {
  creation_params_.stack_size = stack_size;
  creation_params_.xapi_thread_startup = xapi_thread_startup;
  creation_params_.start_address = start_address;
  creation_params_.start_context = start_context;

  // top 8 bits = processor ID (or 0 for default)
  // bit 0 = 1 to create suspended
  creation_params_.creation_flags = creation_flags;

  // Adjust stack size - min of 16k.
  if (creation_params_.stack_size < 16 * 1024) {
    creation_params_.stack_size = 16 * 1024;
  }
  creation_params_.guest_process = guest_process;
  // The kernel does not take a reference. We must unregister in the dtor.
  kernel_state_->RegisterThread(this);
}

XThread::~XThread() {
  // Unregister first to prevent lookups while deleting.
  kernel_state_->UnregisterThread(this);

  // Notify processor of our impending destruction.
  emulator()->processor()->OnThreadDestroyed(thread_id_);

  thread_.reset();

  if (thread_state_) {
    delete thread_state_;
  }
  kernel_state()->memory()->SystemHeapFree(tls_static_address_);
  kernel_state()->memory()->SystemHeapFree(pcr_address_);
  FreeStack();

  if (thread_) {
    // TODO(benvanik): platform kill
    XELOGE("Thread disposed without exiting");
  }
}

thread_local XThread* current_xthread_tls_ = nullptr;

#if !XE_PLATFORM_WIN32
namespace {

pthread_key_t xthread_native_exit_release_key;
pthread_once_t xthread_native_exit_release_key_once = PTHREAD_ONCE_INIT;

void ReleaseRetainedXThreadOnNativeExit(void* value) {
  if (value) {
    static_cast<XThread*>(value)->Release();
  }
}

void CreateXThreadNativeExitReleaseKey() {
  int result = pthread_key_create(&xthread_native_exit_release_key,
                                  ReleaseRetainedXThreadOnNativeExit);
  assert_zero(result);
}

}  // namespace
#endif

#if !XE_PLATFORM_WIN32
// XE_THREAD_SUSPEND_LEGACY=1 restores the pre-2026-11-03 POSIX suspend rule
// (host held on every guest suspend, released only at a guest count of 0).
static bool LegacyGuestSuspendRule() {
  static const bool legacy = [] {
    const char* value = std::getenv("XE_THREAD_SUSPEND_LEGACY");
    return value && *value && std::strcmp(value, "0") != 0;
  }();
  return legacy;
}
#endif

// The guest thread (handle and return address) making a lifecycle call.
static std::string LifecycleCaller() {
  if (!XThread::IsInThread()) {
    return "host";
  }
  XThread* caller = XThread::GetCurrentThread();
  const auto* state = caller ? caller->thread_state() : nullptr;
  return fmt::format(
      "{:08X} lr={:08X}", caller ? caller->handle() : 0,
      state ? uint32_t(state->context()->lr) : 0);
}

bool XThread::IsInThread() { return Thread::IsInThread(); }

bool XThread::IsInThread(XThread* other) {
  return current_xthread_tls_ == other;
}

XThread* XThread::GetCurrentThread() {
  XThread* thread = reinterpret_cast<XThread*>(current_xthread_tls_);
  if (!thread) {
    assert_always("Attempting to use guest stuff from a non-guest thread.");
  }
  return thread;
}

uint32_t XThread::GetCurrentThreadHandle() {
  XThread* thread = XThread::GetCurrentThread();
  return thread->handle();
}

uint32_t XThread::GetCurrentThreadId() {
  XThread* thread = XThread::GetCurrentThread();
  return thread->guest_object<X_KTHREAD>()->thread_id;
}

uint32_t XThread::GetLastError() {
  XThread* thread = XThread::GetCurrentThread();
  return thread->last_error();
}

void XThread::SetLastError(uint32_t error_code) {
  XThread* thread = XThread::GetCurrentThread();
  thread->set_last_error(error_code);
}

uint32_t XThread::last_error() { return guest_object<X_KTHREAD>()->last_error; }

void XThread::set_last_error(uint32_t error_code) {
  guest_object<X_KTHREAD>()->last_error = error_code;
}

void XThread::set_name(const std::string_view name) {
  std::lock_guard<std::mutex> lock(thread_lock_);
  thread_name_ = fmt::format("{} ({:08X})", name, handle());
#if 0  // [thread-diag] Re-enable for thread-name lifetime probes.
#if XE_PLATFORM_MAC
  XELOGI(
      "[thread-diag] XThread::set_name caller_tid={} xthread_id={} "
      "handle={:08X} native_tid={} requested='{}' stored='{}'",
      xe::threading::current_thread_system_id(), thread_id_, handle(),
      thread_ ? thread_->system_id() : 0, name, thread_name_);
#endif
#endif

  if (thread_) {
    // May be getting set before the thread is created.
    // One the thread is ready it will handle it.
    thread_->set_name(thread_name_);
  }
}

std::string XThread::thread_name_snapshot() {
  std::lock_guard<std::mutex> lock(thread_lock_);
  return thread_name_;
}

static uint8_t next_cpu = 0;
static uint8_t GetFakeCpuNumber(uint8_t proc_mask) {
  // NOTE: proc_mask is logical processors, not physical processors or cores.
  if (!proc_mask) {
    // On Xbox 360, threads without an explicit processor assignment stay on
    // the same hardware thread as the parent.  Preserve this so that the
    // guest CPU assignment reflects the game's intent — parent-child thread
    // pairs that share a HW thread may rely on implicit serialization.
    // https://docs.microsoft.com/en-us/windows/win32/dxtecharts/coding-for-multiple-cores
    XThread* parent = current_xthread_tls_;
    if (parent) {
      return parent->active_cpu();
    }
    next_cpu = (next_cpu + 1) % 6;
    return next_cpu;
  }
  assert_false(proc_mask & 0xC0);

  uint8_t cpu_number = 7 - xe::lzcnt(proc_mask);
  assert_true(1 << cpu_number == proc_mask);
  assert_true(cpu_number < 6);
  return cpu_number;
}

void XThread::InitializeGuestObject() {
  auto guest_thread = guest_object<X_KTHREAD>();
  auto thread_guest_ptr = guest_object();
  guest_thread->header.type = X_DISPATCHER_FLAGS::DISPATCHER_THREAD;
  guest_thread->suspend_count =
      (creation_params_.creation_flags & X_CREATE_SUSPENDED) ? 1 : 0;

  guest_thread->mutants_list.flink_ptr = (thread_guest_ptr + 0x10);
  guest_thread->mutants_list.blink_ptr = (thread_guest_ptr + 0x10);

  auto timer_wait_header_list_entry = memory()->HostToGuestVirtual(
      &guest_thread->wait_timeout_timer.header.wait_list);
  guest_thread->wait_timeout_block.wait_list_entry.flink_ptr =
      timer_wait_header_list_entry;
  guest_thread->wait_timeout_block.wait_list_entry.blink_ptr =
      timer_wait_header_list_entry;
  guest_thread->wait_timeout_block.thread = thread_guest_ptr;
  guest_thread->wait_timeout_block.object =
      memory()->HostToGuestVirtual(&guest_thread->wait_timeout_timer);
  guest_thread->wait_timeout_block.wait_result_xstatus = X_STATUS_TIMEOUT;
  guest_thread->wait_timeout_block.wait_type = X_KWAIT_REASON::WaitAny;

  guest_thread->stack_base = (this->stack_base_);
  guest_thread->stack_limit = (this->stack_limit_);
  guest_thread->stack_kernel = (this->stack_base_ - 240);
  guest_thread->tls_address = (this->tls_dynamic_address_);
  guest_thread->thread_state = KTHREAD_STATE_INITIALIZED;
  uint32_t process_info_block_address =
      creation_params_.guest_process ? creation_params_.guest_process
                                     : this->kernel_state_->GetTitleProcess();

  X_KPROCESS* process =
      memory()->TranslateVirtual<X_KPROCESS*>(process_info_block_address);
  uint32_t kpcrb = pcr_address_ + offsetof(X_KPCR, prcb_data);

  auto process_type = process->process_type;
  guest_thread->process_type_dup = process_type;
  guest_thread->process_type = process_type;
  guest_thread->apc_lists[0].Initialize(memory());
  guest_thread->apc_lists[1].Initialize(memory());

  guest_thread->process_priority_class = process->process_priority_class;
  auto base_prio = process->default_thread_priority;
  guest_thread->base_priority_copy = base_prio;
  guest_thread->base_priority = base_prio;
  guest_thread->priority = base_prio;
  guest_thread->max_dynamic_priority = process->max_dynamic_priority;
  guest_thread->quantum = process->quantum;

  // Sync the host-side priority tracking to match the guest defaults.
  // Games may later override these via KeSetPriorityThread.
  priority_ = base_prio;
  base_priority_ = base_prio;

  guest_thread->a_prcb_ptr = kpcrb;
  guest_thread->another_prcb_ptr = kpcrb;

  guest_thread->may_queue_apcs = 1;
  guest_thread->msr_mask = 0xFDFFD7FF;
  guest_thread->process = process_info_block_address;
  guest_thread->stack_alloc_base = this->stack_base_;
  guest_thread->create_time = Clock::QueryGuestSystemTime();
  guest_thread->timer_list.flink_ptr = thread_guest_ptr + 324;
  guest_thread->timer_list.blink_ptr = thread_guest_ptr + 324;
  guest_thread->thread_id = this->thread_id_;
  guest_thread->start_address = this->creation_params_.start_address;
  guest_thread->unk_154.flink_ptr = thread_guest_ptr + 340;
  uint32_t v9 = thread_guest_ptr;
  guest_thread->last_error = 0;
  guest_thread->unk_154.blink_ptr = v9 + 340;
  guest_thread->creation_flags = this->creation_params_.creation_flags;
  // According to nukernel.
  // guest_thread->host_xthread_stash = reinterpret_cast<void*>(this);

  /*
   * not doing this right at all! we're not using our threads context, because
   * we may be on the host and have no underlying context. in reality we should
   * have a context and acquire any locks using that context!
   */
  auto context_here = thread_state_->context();
  auto old_irql = xboxkrnl::xeKeKfAcquireSpinLock(
      context_here, &process->thread_list_spinlock);

  // todo: acquire dispatcher lock here?

  util::XeInsertTailList(&process->thread_list, &guest_thread->process_threads,
                         context_here);
  process->thread_count += 1;
  // todo: release dispatcher lock here?
  xboxkrnl::xeKeKfReleaseSpinLock(context_here, &process->thread_list_spinlock,
                                  old_irql);
}

bool XThread::AllocateStack(uint32_t size) {
  auto heap = memory()->LookupHeap(kStackAddressRangeBegin);

  auto alignment = heap->page_size();
  auto padding = heap->page_size() * 2;  // Guard page size * 2
  size = xe::round_up(size, alignment);
  auto actual_size = size + padding;

  uint32_t address = 0;
  if (!heap->AllocRange(
          kStackAddressRangeBegin, kStackAddressRangeEnd, actual_size,
          alignment, kMemoryAllocationReserve | kMemoryAllocationCommit,
          kMemoryProtectRead | kMemoryProtectWrite, false, &address)) {
    return false;
  }

  stack_alloc_base_ = address;
  stack_alloc_size_ = actual_size;
  stack_limit_ = address + (padding / 2);
  stack_base_ = stack_limit_ + size;

  // Setup the guard pages
  heap->Protect(stack_alloc_base_, padding / 2, kMemoryProtectNoAccess);
  heap->Protect(stack_base_, padding / 2, kMemoryProtectNoAccess);

  return true;
}

void XThread::FreeStack() {
  if (stack_alloc_base_) {
    auto heap = memory()->LookupHeap(kStackAddressRangeBegin);
    heap->Release(stack_alloc_base_);

    stack_alloc_base_ = 0;
    stack_alloc_size_ = 0;
    stack_base_ = 0;
    stack_limit_ = 0;
  }
}

X_STATUS XThread::Create() {
  // Thread kernel object.
  if (!CreateNative<X_KTHREAD>()) {
    XELOGW("Unable to allocate thread object");
    return X_STATUS_NO_MEMORY;
  }

  // Allocate a stack.
  if (!AllocateStack(creation_params_.stack_size)) {
    return X_STATUS_NO_MEMORY;
  }

  // Allocate TLS block.
  // Games will specify a certain number of 4b slots that each thread will get.
  xex2_opt_tls_info* tls_header = nullptr;
  auto module = kernel_state()->GetExecutableModule();
  if (module) {
    module->GetOptHeader(XEX_HEADER_TLS_INFO, &tls_header);
  }

  constexpr uint32_t kDefaultTlsSlotCount = 1024;
  uint32_t tls_slots = kDefaultTlsSlotCount;
  uint32_t tls_extended_size = 0;
  if (tls_header && tls_header->slot_count) {
    tls_slots = tls_header->slot_count;
    tls_extended_size = tls_header->data_size;
  }

  // Allocate both the slots and the extended data.
  // Some TLS is compiled with the binary (declspec(thread)) vars. The game
  // will directly access those through 0(r13).
  uint32_t tls_slot_size = tls_slots * 4;
  tls_total_size_ = tls_slot_size + tls_extended_size;
  tls_static_address_ = memory()->SystemHeapAlloc(tls_total_size_);
  tls_dynamic_address_ = tls_static_address_ + tls_extended_size;
  if (!tls_static_address_) {
    XELOGW("Unable to allocate thread local storage block");
    return X_STATUS_NO_MEMORY;
  }

  // Zero all of TLS.
  memory()->Fill(tls_static_address_, tls_total_size_, 0);
  if (tls_extended_size) {
    // If game has extended data, copy in the default values.
    assert_not_zero(tls_header->raw_data_address);
    memory()->Copy(tls_static_address_, tls_header->raw_data_address,
                   tls_header->raw_data_size);
  }

  // Allocate thread state block from heap.
  // https://web.archive.org/web/20170704035330/https://www.microsoft.com/msj/archive/S2CE.aspx
  // This is set as r13 for user code and some special inlined Win32 calls
  // (like GetLastError/etc) will poke it directly.
  // We try to use it as our primary store of data just to keep things all
  // consistent.
  // 0x000: pointer to tls data
  // 0x100: pointer to TEB(?)
  // 0x10C: Current CPU(?)
  // 0x150: if >0 then error states don't get set (DPC active bool?)
  // TEB:
  // 0x14C: thread id
  // 0x160: last error
  // So, at offset 0x100 we have a 4b pointer to offset 200, then have the
  // structure.
  pcr_address_ = memory()->SystemHeapAlloc(0x2D8);
  if (!pcr_address_) {
    XELOGW("Unable to allocate thread state block");
    return X_STATUS_NO_MEMORY;
  }

  // Allocate processor thread state.
  // This is thread safe.
  thread_state_ = new cpu::ThreadState(kernel_state()->processor(), thread_id_,
                                       stack_base_, pcr_address_);
  XELOGI("XThread{:08X} ({:X}) Stack: {:08X}-{:08X}", handle(), thread_id_,
         stack_limit_, stack_base_);

  // Exports use this to get the kernel.
  thread_state_->context()->kernel_state = kernel_state_;

  uint8_t cpu_index = GetFakeCpuNumber(
      static_cast<uint8_t>(creation_params_.creation_flags >> 24));

  // Initialize the KTHREAD object.
  InitializeGuestObject();

  X_KPCR* pcr = memory()->TranslateVirtual<X_KPCR*>(pcr_address_);

  pcr->tls_ptr = tls_static_address_;
  pcr->pcr_ptr = pcr_address_;
  pcr->prcb_data.current_thread = guest_object();
  pcr->prcb = pcr_address_ + offsetof(X_KPCR, prcb_data);
  pcr->host_stash = reinterpret_cast<uint64_t>(thread_state_->context());
  pcr->stack_base_ptr = stack_base_;
  pcr->stack_end_ptr = stack_limit_;

  pcr->prcb_data.dpc_active = 0;  // DPC active bool?

  // Always retain when starting - the thread owns itself until exited.
  RetainHandle();

  xe::threading::Thread::CreationParameters params;

  params.create_suspended = true;

  params.stack_size = 16_MiB;  // Allocate a big host stack.
#if XE_PLATFORM_MAC
  // Assign an Apple QoS class (when the guest_thread_qos policy selects this
  // thread) at creation so the scheduler can keep it on P-cores; such threads
  // also skip the SCHED_FIFO mapping that would otherwise disable QoS.
  params.qos_class = GuestThreadQosClass(main_thread(), priority());
#endif
  thread_ = xe::threading::Thread::Create(params, [this]() {
    // Set thread ID override. This is used by logging.
    xe::threading::set_current_thread_id(handle());

    // Snapshot the guest thread name once under thread_lock_. This lambda runs
    // on the freshly-created thread; reading thread_name_ directly here would
    // race a concurrent XThread::set_name() reassigning it.
    const std::string thread_name = thread_name_snapshot();

    // Set name immediately, if we have one.
#if 0  // [thread-diag] Re-enable for thread startup lifetime probes.
#if XE_PLATFORM_MAC
    XELOGI(
        "[thread-diag] XThread::Create start naming caller_tid={} "
        "xthread_id={} handle={:08X} native_tid={} name='{}'",
        xe::threading::current_thread_system_id(), thread_id_, handle(),
        thread_ ? thread_->system_id() : 0, thread_name);
#endif
#endif
    thread_->set_name(thread_name);

    // Profiler needs to know about the thread.
#if 0  // [thread-diag] Re-enable for thread startup lifetime probes.
#if XE_PLATFORM_MAC
    XELOGI(
        "[thread-diag] XThread::Create profiler enter caller_tid={} "
        "xthread_id={} handle={:08X} native_tid={} name='{}'",
        xe::threading::current_thread_system_id(), thread_id_, handle(),
        thread_ ? thread_->system_id() : 0, thread_name);
#endif
#endif
    xe::Profiler::ThreadEnter(thread_name.c_str());

    // Execute user code.
    current_xthread_tls_ = this;
    current_thread_ = this;
    cpu::ThreadState::Bind(this->thread_state());
    running_ = true;
    Execute();
    running_ = false;
    current_thread_ = nullptr;
    current_xthread_tls_ = nullptr;

    xe::Profiler::ThreadExit();

    // Release the self-reference to the thread.
    ReleaseHandle();
  });

  if (!thread_) {
    // TODO(benvanik): translate error?
    XELOGE("CreateThread failed");
    return X_STATUS_NO_MEMORY;
  }

  // Set the thread name based on host ID (for easier debugging).
  if (thread_name_snapshot().empty()) {
    set_name(fmt::format("XThread{:04X}", thread_->system_id()));
  }

  if (creation_params_.creation_flags & 0x60) {
    thread_->set_priority(creation_params_.creation_flags & 0x20 ? 1 : 0);
  }

  // Assign the newly created thread to the logical processor, and also set up
  // the current CPU in KPCR and KTHREAD.
  SetActiveCpu(cpu_index);

  // Notify processor of our creation.
  emulator()->processor()->OnThreadCreated(handle(), thread_state_, this);

  if (xe::threading::ThreadLifecycleLogEnabled()) {
    XELOGI(
        "ThreadLifecycle create handle={:08X} tid={:X} host={:X} flags={:08X} "
        "start={:08X} guest_suspend={} by {}",
        handle(), thread_id_, thread_->system_id(),
        creation_params_.creation_flags, creation_params_.start_address,
        uint32_t(guest_object<X_KTHREAD>()->suspend_count), LifecycleCaller());
  }

  if ((creation_params_.creation_flags & X_CREATE_SUSPENDED) == 0) {
    // Start the thread now that we're all setup.
    thread_->Resume();
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XThread::Exit(int exit_code) {
  // This may only be called on the thread itself.
  assert_true(XThread::GetCurrentThread() == this);
  // TODO(chrispy): not sure if this order is correct, should it come after
  // apcs?
  auto kthread = guest_object<X_KTHREAD>();
  auto cpu_context = thread_state_->context();
  kthread->terminated = 1;

  // TODO(benvanik): dispatch events? waiters? etc?
  RundownAPCs();

  // Set exit code.
  kthread->header.signal_state = 1;
  kthread->exit_status = exit_code;

  auto kprocess = cpu_context->TranslateVirtual(kthread->process);

  uint32_t old_irql = xboxkrnl::xeKeKfAcquireSpinLock(
      cpu_context, &kprocess->thread_list_spinlock);

  util::XeRemoveEntryList(&kthread->process_threads, cpu_context);

  kprocess->thread_count = kprocess->thread_count - 1;

  xboxkrnl::xeKeKfReleaseSpinLock(cpu_context, &kprocess->thread_list_spinlock,
                                  old_irql);

  kernel_state()->OnThreadExit(this);

  // Notify processor of our exit.
  emulator()->processor()->OnThreadExit(thread_id_);

  // NOTE: unless PlatformExit fails, expect it to never return!
  current_xthread_tls_ = nullptr;
  current_thread_ = nullptr;
  xe::Profiler::ThreadExit();

  running_ = false;
#if !XE_PLATFORM_WIN32
  // Thread::Exit() still uses the POSIX thread-local current_thread_ to signal
  // the native Thread object. If this ReleaseHandle() drops the last guest
  // handle, keep the XThread and its thread_ wrapper alive until pthread exit.
  RetainUntilNativeThreadExit();
#endif
  ReleaseHandle();

  // NOTE: this does not return!
  xe::threading::Thread::Exit(exit_code);
  return X_STATUS_SUCCESS;
}

X_STATUS XThread::Terminate(int exit_code) {
  // TODO(benvanik): inform the profiler that this thread is exiting.
#if 0  // [thread-diag] Re-enable for self/external termination probes.
#if XE_PLATFORM_MAC
  XELOGI(
      "[thread-diag] XThread::Terminate caller_tid={} xthread_id={} "
      "handle={:08X} native_tid={} in_thread={} running={} exit_code={}",
      xe::threading::current_thread_system_id(), thread_id_, handle(),
      thread_ ? thread_->system_id() : 0, XThread::IsInThread(this), running_,
      exit_code);
#endif
#endif

  // Set exit code.
  X_KTHREAD* thread = guest_object<X_KTHREAD>();
  thread->header.signal_state = 1;
  thread->exit_status = exit_code;

  // Notify processor of our exit.
  emulator()->processor()->OnThreadExit(thread_id_);

  running_ = false;
  if (XThread::IsInThread(this)) {
#if !XE_PLATFORM_WIN32
    // See XThread::Exit(): Thread::Exit() needs the POSIX thread wrapper after
    // the guest handle release, so defer object destruction until native exit.
    RetainUntilNativeThreadExit();
#endif
    ReleaseHandle();
    xe::threading::Thread::Exit(exit_code);
  } else {
    thread_->Terminate(exit_code);
    ReleaseHandle();
  }

  return X_STATUS_SUCCESS;
}

void XThread::Execute() {
  execute_began_ = true;
  XELOGKERNEL("XThread::Execute thid {} (handle={:08X}, '{}', native={:08X})",
              thread_id_, handle(), thread_name_snapshot(), thread_->system_id());
  // Let the kernel know we are starting.
  kernel_state()->OnThreadExecute(this);

  // Dispatch any APCs that were queued before the thread was created first.
  DeliverAPCs();

  uint32_t address;
  std::vector<uint64_t> args;
  bool want_exit_code;
  int exit_code = 0;

  // If a XapiThreadStartup value is present, we use that as a trampoline.
  // Otherwise, we are a raw thread.
  if (creation_params_.xapi_thread_startup) {
    address = creation_params_.xapi_thread_startup;
    args.push_back(creation_params_.start_address);
    args.push_back(creation_params_.start_context);
    want_exit_code = false;
  } else {
    // Run user code.
    address = creation_params_.start_address;
    args.push_back(creation_params_.start_context);
    want_exit_code = true;
  }

  // Set up reentry mechanism for fiber-based stack switching.
  // When Reenter() is called (e.g., by KeSetCurrentStackPointers), it
  // unwinds back here to re-enter at a new guest address.
  //
  // On Linux, C++ exceptions are used so that DWARF unwind info (registered
  // for JIT code via __register_frame) allows proper destructor/RAII cleanup
  // through both JIT and host C++ frames.
  //
  // On Windows, setjmp/longjmp is used because MSVC's longjmp performs SEH
  // stack unwinding which already calls destructors.
  uint32_t next_address;
#if !XE_PLATFORM_WIN32
  try {
    exit_code = static_cast<int>(kernel_state()->processor()->Execute(
        thread_state_, address, args.data(), args.size()));
    next_address = 0;
  } catch (const FiberReentryException& e) {
#if XE_PLATFORM_LINUX
    // Ensure SIGRTMIN (used for thread suspend) is not left blocked.
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGRTMIN);
    pthread_sigmask(SIG_UNBLOCK, &set, nullptr);
#endif
    next_address = e.address;
  }

  while (next_address != 0) {
    try {
      kernel_state()->processor()->ExecuteRaw(thread_state_, next_address);
      next_address = 0;
      if (want_exit_code) {
        exit_code = static_cast<int>(thread_state_->context()->r[3]);
      }
    } catch (const FiberReentryException& e) {
#if XE_PLATFORM_LINUX
      sigset_t set;
      sigemptyset(&set);
      sigaddset(&set, SIGRTMIN);
      pthread_sigmask(SIG_UNBLOCK, &set, nullptr);
#endif
      next_address = e.address;
    }
  }
#else
  if (setjmp(reentry_jmp_buf_) != 0) {
    next_address = reentry_address_;
  } else {
    exit_code = static_cast<int>(kernel_state()->processor()->Execute(
        thread_state_, address, args.data(), args.size()));
    next_address = 0;
  }

  while (next_address != 0) {
    if (setjmp(reentry_jmp_buf_) != 0) {
      next_address = reentry_address_;
    } else {
      kernel_state()->processor()->ExecuteRaw(thread_state_, next_address);
      next_address = 0;
      if (want_exit_code) {
        exit_code = static_cast<int>(thread_state_->context()->r[3]);
      }
    }
  }
#endif

  // If we got here it means the execute completed without an exit being called.
  // Treat the return code as an implicit exit code (if desired).
  Exit(!want_exit_code ? 0 : exit_code);
}

void XThread::Reenter(uint32_t address) {
  // Called when the game switches fiber stacks (e.g., via
  // KeSetCurrentStackPointers in games like Forza Horizon 2).
  // Must unwind through all frames between here and Execute().
#if !XE_PLATFORM_WIN32
  // Throw a C++ exception that unwinds through JIT frames (using DWARF
  // .eh_frame info) and host frames (using compiler-generated DWARF),
  // calling destructors properly along the way.
  throw FiberReentryException{address};
#else
  reentry_address_ = address;
  std::longjmp(reentry_jmp_buf_, 1);
#endif
}

void XThread::EnterCriticalRegion() {
  guest_object<X_KTHREAD>()->apc_disable_count--;
}

void XThread::LeaveCriticalRegion() {
  auto kthread = guest_object<X_KTHREAD>();
  // this has nothing to do with user mode apcs!
  auto apc_disable_count = ++kthread->apc_disable_count;
}

void XThread::EnqueueApc(uint32_t normal_routine, uint32_t normal_context,
                         uint32_t arg1, uint32_t arg2, bool from_timer) {
  // don't use thread_state_ -> context() ! we're not running on the thread
  // we're enqueuing to
  uint32_t success = xboxkrnl::xeNtQueueApcThread(
      this->handle(), normal_routine, normal_context, arg1, arg2,
      cpu::ThreadState::Get()->context(), from_timer);

  xenia_assert(success == X_STATUS_SUCCESS);
}

void XThread::SetCurrentThread() { current_xthread_tls_ = this; }

void XThread::DeliverAPCs() {
  // https://www.drdobbs.com/inside-nts-asynchronous-procedure-call/184416590?pgno=1
  // https://www.drdobbs.com/inside-nts-asynchronous-procedure-call/184416590?pgno=7
  xboxkrnl::xeProcessUserApcs(thread_state_->context());
}

void XThread::RundownAPCs() {
  xboxkrnl::xeRundownApcs(thread_state_->context());
}

#if !XE_PLATFORM_WIN32
void XThread::RetainUntilNativeThreadExit() {
  int once_result =
      pthread_once(&xthread_native_exit_release_key_once,
                   CreateXThreadNativeExitReleaseKey);
  assert_zero(once_result);

  if (pthread_getspecific(xthread_native_exit_release_key) == this) {
    return;
  }

  Retain();
  int set_result = pthread_setspecific(xthread_native_exit_release_key, this);
  if (set_result != 0) {
    Release();
    assert_always("Unable to defer XThread release until native thread exit.");
  }
}
#endif

int32_t XThread::QueryPriority() { return thread_->priority(); }

// Map Xenon's 0-31 priority range across the available host priority levels.
// Priority 18 (0x12) is the Xenon real-time threshold — threads at or above
// it don't get quantum decay on real hardware.
static int32_t GuestPriorityToHost(int32_t guest_priority) {
  if (guest_priority >= 24) {
    return xe::threading::ThreadPriority::kHighest;
  } else if (guest_priority >= 17) {
    return xe::threading::ThreadPriority::kAboveNormal;
  } else if (guest_priority >= 10) {
    return xe::threading::ThreadPriority::kNormal;
  } else if (guest_priority >= 5) {
    return xe::threading::ThreadPriority::kBelowNormal;
  } else {
    return xe::threading::ThreadPriority::kLowest;
  }
}

void XThread::SetPriority(int32_t increment) {
  // Clamp to valid Xenon priority range.  Negative values can arrive via
  // KeSetBasePriorityThread (signed offset from process base).
  int32_t clamped = std::max(increment, 0);
  if (is_guest_thread()) {
    guest_object<X_KTHREAD>()->priority = static_cast<uint8_t>(clamped);
  }
  priority_ = clamped;
  base_priority_ = clamped;
  quantum_start_ms_ = Clock::QueryHostUptimeMillis();
  if (!cvars::ignore_thread_priorities) {
    thread_->set_priority(GuestPriorityToHost(clamped));
  }
}

void XThread::CheckQuantumAndDecay() {
  if (cvars::ignore_thread_priorities) {
    return;
  }
  // Real-time threads (current priority >= 0x12) don't decay on Xenon.
  if (priority_ >= 18) {
    return;
  }

  uint64_t now = Clock::QueryHostUptimeMillis();
  uint64_t elapsed = now - quantum_start_ms_;
  // On Xenon, the clock interrupt fires every ~1ms and decrements the
  // thread's quantum by 3.  The process quantum is 60, so it takes ~20ms
  // for quantum to expire.  When it does, the scheduler decays the
  // effective priority by exactly 1 and resets quantum.  We approximate
  // this by decaying 1 priority level per 20ms of elapsed wall-clock time.
  constexpr uint64_t kQuantumPeriodMs = 20;
  if (elapsed < kQuantumPeriodMs) {
    return;
  }

  int32_t decay_steps = static_cast<int32_t>(elapsed / kQuantumPeriodMs);
  // On the first decay step, drain the accumulated priority boost as well.
  // The real kernel computes: new_prio = priority - boost_accumulator - 1
  // then zeroes the accumulator.  Additional decay steps (if the timer
  // callback was late) each subtract 1 more.
  int32_t total_decay = boost_amount_ + decay_steps;
  boost_amount_ = 0;

  int32_t new_priority = priority_ - total_decay;
  if (new_priority < base_priority_) {
    new_priority = base_priority_;
  }
  if (new_priority != priority_) {
    priority_ = new_priority;
    if (is_guest_thread()) {
      guest_object<X_KTHREAD>()->priority = static_cast<uint8_t>(new_priority);
    }
    thread_->set_priority(GuestPriorityToHost(new_priority));
  }
  quantum_start_ms_ = now;
}

void XThread::BoostOnWake(int32_t increment) {
  if (cvars::ignore_thread_priorities) {
    return;
  }

  // Real-time threads (priority >= 0x12) just get their quantum reset.
  if (priority_ >= 18) {
    boost_amount_ = 0;
    quantum_start_ms_ = Clock::QueryHostUptimeMillis();
    return;
  }

  // Match the real kernel (xeEnqueueThreadPostWait):
  //   - Only apply boost if there is no pending decay (priority_decrement == 0)
  //     AND boost is not disabled on this thread.
  //   - Boosted priority = base + increment, clamped to max_priority_cap.
  //   - Only boost UP — never lower priority below its current value.
  bool apply_boost = false;
  if (increment > 0 && is_guest_thread()) {
    auto* kthread = guest_object<X_KTHREAD>();
    if (kthread->priority_decrement == 0 && !kthread->boost_disabled) {
      apply_boost = true;
    }
  } else if (increment > 0) {
    // Host threads (non-guest): apply boost unconditionally.
    apply_boost = true;
  }

  if (apply_boost) {
    int32_t boosted = base_priority_ + increment;
    // Clamp to the per-thread max dynamic priority cap.
    // For title threads this is 17 (just below real-time threshold).
    int32_t max_cap = 17;
    if (is_guest_thread()) {
      uint8_t guest_cap = guest_object<X_KTHREAD>()->max_dynamic_priority;
      if (guest_cap > 0) {
        max_cap = guest_cap;
      }
    }
    if (boosted > max_cap) {
      boosted = max_cap;
    }
    // Only boost UP, never lower.
    if (boosted > priority_) {
      priority_ = boosted;
      boost_amount_ = priority_ - base_priority_;
      if (is_guest_thread()) {
        guest_object<X_KTHREAD>()->priority = static_cast<uint8_t>(priority_);
      }
      thread_->set_priority(GuestPriorityToHost(priority_));
    }
  }

  quantum_start_ms_ = Clock::QueryHostUptimeMillis();
}

void XThread::SetAffinity(uint32_t affinity) {
  SetActiveCpu(GetFakeCpuNumber(affinity));
}

uint8_t XThread::active_cpu() const {
  const X_KPCR& pcr = *memory()->TranslateVirtual<const X_KPCR*>(pcr_address_);
  return pcr.prcb_data.current_cpu;
}

void XThread::SetActiveCpu(uint8_t cpu_index) {
  // May be called during thread creation - don't skip if current == new.

  assert_true(cpu_index < 6);

  X_KPCR& pcr = *memory()->TranslateVirtual<X_KPCR*>(pcr_address_);
  pcr.prcb_data.current_cpu = cpu_index;

  if (is_guest_thread()) {
    X_KTHREAD& thread_object =
        *memory()->TranslateVirtual<X_KTHREAD*>(guest_object());
    thread_object.current_cpu = cpu_index;
  }

  if (xe::threading::logical_processor_count() >= 6) {
    if (!cvars::ignore_thread_affinities) {
      thread_->set_affinity_mask(uint64_t(1) << cpu_index);
    }
  } else {
    // there no good reason why we need to log this... we don't perfectly
    // emulate the 360's scheduler in any way
    // XELOGW("Too few processor cores - scheduling will be wonky");
  }
}

bool XThread::GetTLSValue(uint32_t slot, uint32_t* value_out) {
  if (slot * 4 > tls_total_size_) {
    return false;
  }

  auto mem = memory()->TranslateVirtual(tls_dynamic_address_ + slot * 4);
  *value_out = xe::load_and_swap<uint32_t>(mem);
  return true;
}

bool XThread::SetTLSValue(uint32_t slot, uint32_t value) {
  if (slot * 4 >= tls_total_size_) {
    return false;
  }

  auto mem = memory()->TranslateVirtual(tls_dynamic_address_ + slot * 4);
  xe::store_and_swap<uint32_t>(mem, value);
  return true;
}

uint32_t XThread::suspend_count() {
  return guest_object<X_KTHREAD>()->suspend_count;
}

X_FILETIME XThread::creation_time() {
  return static_cast<X_FILETIME>(guest_object<X_KTHREAD>()->create_time);
}

uint32_t XThread::start_address() {
  return guest_object<X_KTHREAD>()->start_address;
}

X_STATUS XThread::Resume(uint32_t* out_suspend_count) {
  auto guest_thread = guest_object<X_KTHREAD>();
  uint32_t unused_host_suspend_count = 0;
  resume_calls_.fetch_add(1, std::memory_order_relaxed);

#if XE_PLATFORM_WIN32
  uint8_t previous_suspend_count =
      reinterpret_cast<std::atomic_uint8_t*>(&guest_thread->suspend_count)
          ->fetch_sub(1);
  if (out_suspend_count) {
    *out_suspend_count = previous_suspend_count;
  }

  if (thread_->Resume(&unused_host_suspend_count)) {
    return X_STATUS_SUCCESS;
  } else {
    return X_STATUS_UNSUCCESSFUL;
  }
#else
  uint32_t previous = 0;
  if (!LegacyGuestSuspendRule()) {
    // Release the host thread once per resume of a non-zero guest count, the
    // same number of times GuestSuspendHost held it (see guest_suspend.h).
    // Not parked by a suspend signal while holding suspend_mutex_ (Resume
    // needs it).
    xe::threading::SuspendDeferralScope suspend_deferral;
    std::lock_guard<std::mutex> lock(suspend_mutex_);
    previous =
        util::GuestResumeHost(&guest_thread->suspend_count, thread_.get());
    suspend_cv_.notify_all();
  } else {
    // Use mutex to protect suspend_count access and coordinate with
    // SelfSuspend.
    bool should_resume_host = false;
    {
      // Not parked by a suspend signal while holding suspend_mutex_ (Resume
      // needs it).
      xe::threading::SuspendDeferralScope suspend_deferral;
      std::lock_guard<std::mutex> lock(suspend_mutex_);
      previous = guest_thread->suspend_count;
      if (previous > 0) {
        guest_thread->suspend_count--;
      }
      should_resume_host = (guest_thread->suspend_count == 0);
      suspend_cv_.notify_all();
    }

    // Try to resume host thread if fully resumed (for non-self-suspended
    // case).
    if (should_resume_host) {
      thread_->Resume(&unused_host_suspend_count);
    }
  }
  if (out_suspend_count) {
    *out_suspend_count = previous;
  }
  if (xe::threading::ThreadLifecycleLogEnabled()) {
    XELOGI("ThreadLifecycle resume handle={:08X} guest_suspend {}->{} by {} {}",
           handle(), previous, uint32_t(guest_thread->suspend_count),
           LifecycleCaller(), thread_->DebugState());
  }
  return X_STATUS_SUCCESS;
#endif
}

X_STATUS XThread::Suspend(uint32_t* out_suspend_count) {
  // this normally holds the apc lock for the thread, because it queues a kernel
  // mode apc that does the actual suspension

  X_KTHREAD* guest_thread = guest_object<X_KTHREAD>();
  suspend_calls_.fetch_add(1, std::memory_order_relaxed);

#if !XE_PLATFORM_WIN32
  if (!LegacyGuestSuspendRule()) {
    if (XThread::IsInThread(this)) {
      // Parks on the guest count alone; Resume wakes it at 0.
      if (xe::threading::ThreadLifecycleLogEnabled()) {
        XELOGI("ThreadLifecycle self-suspend handle={:08X} via Suspend by {}",
               handle(), LifecycleCaller());
      }
      uint32_t previous = SelfSuspend();
      if (out_suspend_count) {
        *out_suspend_count = previous;
      }
      return X_STATUS_SUCCESS;
    }
    uint32_t previous = 0;
    {
      // Under the lock so a concurrent Resume sees the guest and host counts
      // change together.
      // Not parked by a suspend signal while holding suspend_mutex_ (Resume
      // needs it).
      xe::threading::SuspendDeferralScope suspend_deferral;
      std::lock_guard<std::mutex> lock(suspend_mutex_);
      previous =
          util::GuestSuspendHost(&guest_thread->suspend_count, thread_.get());
    }
    if (out_suspend_count) {
      *out_suspend_count = previous;
    }
    if (xe::threading::ThreadLifecycleLogEnabled()) {
      XELOGI(
          "ThreadLifecycle suspend handle={:08X} guest_suspend {}->{} by {} {}",
          handle(), previous, uint32_t(guest_thread->suspend_count),
          LifecycleCaller(), thread_->DebugState());
    }
    return X_STATUS_SUCCESS;
  }
#endif

  uint8_t previous_suspend_count =
      reinterpret_cast<std::atomic_uint8_t*>(&guest_thread->suspend_count)
          ->fetch_add(1);
  if (out_suspend_count) {
    *out_suspend_count = previous_suspend_count;
  }
  if (xe::threading::ThreadLifecycleLogEnabled()) {
    XELOGI(
        "ThreadLifecycle suspend(legacy) handle={:08X} guest_suspend {}->{} "
        "by {}",
        handle(), uint32_t(previous_suspend_count),
        uint32_t(guest_thread->suspend_count), LifecycleCaller());
  }
  // If we are suspending ourselves, we can't hold the lock.
  uint32_t unused_host_suspend_count = 0;

  // If we had suspend count wrap around and go back to 0 then thread is not
  // suspended.
  if (guest_thread->suspend_count == 0) {
    return X_STATUS_SUCCESS;
  }

  if (thread_->Suspend(&unused_host_suspend_count)) {
    return X_STATUS_SUCCESS;
  } else {
    return X_STATUS_UNSUCCESSFUL;
  }
}

#if !XE_PLATFORM_WIN32
uint32_t XThread::SelfSuspend() {
  auto guest_thread = guest_object<X_KTHREAD>();
  // Not parked by a suspend signal while holding suspend_mutex_ (Resume
  // needs it).
  xe::threading::SuspendDeferralScope suspend_deferral;
  std::unique_lock<std::mutex> lock(suspend_mutex_);
  uint32_t previous = guest_thread->suspend_count;
  guest_thread->suspend_count++;
  if (xe::threading::ThreadLifecycleLogEnabled()) {
    XELOGI("ThreadLifecycle self-suspend handle={:08X} guest_suspend {}->{}",
           handle(), previous, uint32_t(guest_thread->suspend_count));
  }
  suspend_cv_.wait(
      lock, [guest_thread]() { return guest_thread->suspend_count == 0; });
  return previous;
}
#endif  // !XE_PLATFORM_WIN32

std::string XThread::DebugSuspendState() {
  const X_KTHREAD* guest_thread = guest_object<X_KTHREAD>();
  return fmt::format(
      "gsusp={} flags={:08X}{} exec={} susp_calls={} res_calls={} {}",
      guest_thread ? uint32_t(guest_thread->suspend_count) : 0xFFFFFFFFu,
      creation_params_.creation_flags,
      (creation_params_.creation_flags & X_CREATE_SUSPENDED) ? "(suspended)"
                                                              : "",
      execute_began_.load(std::memory_order_relaxed) ? 1 : 0,
      suspend_calls_.load(std::memory_order_relaxed),
      resume_calls_.load(std::memory_order_relaxed),
      thread_ ? thread_->DebugState() : std::string("host[none]"));
}

X_STATUS XThread::Delay(uint32_t processor_mode, uint32_t alertable,
                        uint64_t interval) {
  int64_t timeout_ticks = interval;
  uint32_t timeout_ms;
  if (timeout_ticks > 0) {
    // Absolute time, based on January 1, 1601.
    // TODO(benvanik): convert time to relative time.
    assert_always();
    timeout_ms = 0;
  } else if (timeout_ticks < 0) {
    // Relative time.
    timeout_ms = uint32_t(-timeout_ticks / 10000);  // Ticks -> MS
  } else {
    timeout_ms = 0;
  }

  timeout_ms = Clock::ScaleGuestDurationMillis(timeout_ms);
  if (alertable) {
    const bool prof = cvars::guest_wait_profile;
    const auto sleep_start = prof ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    auto result =
        xe::threading::AlertableSleep(std::chrono::milliseconds(timeout_ms));
    if (prof) {
      NoteAlertableSleep(
          uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - sleep_start)
                       .count()));
    }
    switch (result) {
      default:
      case xe::threading::SleepResult::kSuccess:
        return X_STATUS_SUCCESS;
      case xe::threading::SleepResult::kAlerted:
        return X_STATUS_USER_APC;
    }
  } else {
    if (timeout_ms == 0) {
      if (cvars::spin0_sleep_us > 0) {
        // Diagnostic override: fixed NanoSleep on every Sleep(0) (A/B knob).
        xe::threading::NanoSleep(cvars::spin0_sleep_us * 1000);
        NoteBackoffSleep(uint64_t(cvars::spin0_sleep_us) * 1000);
      } else {
#if XE_PLATFORM_MAC
        // Adaptive busy-poll backoff (macOS ship fix), applied to ALL Sleep(0)
        // REGARDLESS of guest priority — the guest pump spinners that starve the
        // dispatcher are not all normal-priority, and the validated blunt fix
        // throttled every Sleep(0) (a per-priority gate let below-normal spinners
        // escape and the freeze returned). The guest's lock-free pump loops spam
        // Sleep(0); a bare sched_yield keeps the spinner runnable and on Apple
        // Silicon's asymmetric scheduler that starves the guest job dispatcher
        // (queue 83DF6880 orphans → DOA5 load-stall freeze). Keep a short burst
        // of consecutive Sleep(0)s cheap (yield), then de-schedule a runaway
        // spinner with a bounded sleep so the dispatcher gets the core; a
        // multi-ms gap between Sleep(0)s resets the run to stay low-latency.
        const int64_t backoff_us = cvars::macos_spin0_backoff_us;
        if (backoff_us > 0) {
          static thread_local uint32_t spin0_run = 0;
          static thread_local uint64_t spin0_last_ms = 0;
          const uint64_t now_ms = xe::Clock::QueryHostUptimeMillis();
          if (now_ms - spin0_last_ms > kSpin0ResetMs) {
            spin0_run = 0;
          }
          spin0_last_ms = now_ms;
          if (++spin0_run <= kSpin0YieldBurst) {
            xe::threading::MaybeYield();
            NoteYield();
          } else if (cvars::macos_spin0_depress) {
            // Depress this thread's priority for up to 1 ms and switch: the
            // starved dispatcher (or any other runnable thread) gets the core,
            // and with nothing waiting the spinner continues immediately, as
            // Sleep(0) does on the console.
            thread_switch(MACH_PORT_NULL, SWITCH_OPTION_DEPRESS, 1);
            NoteYield();
          } else {
            xe::threading::NanoSleep(backoff_us * 1000);
            NoteBackoffSleep(uint64_t(backoff_us) * 1000);
          }
        } else if (priority_ <= xe::threading::ThreadPriority::kBelowNormal) {
          xe::threading::NanoSleep(100);
          NoteBackoffSleep(100);
        } else {
          xe::threading::MaybeYield();
          NoteYield();
        }
#else
        if (priority_ <= xe::threading::ThreadPriority::kBelowNormal) {
          xe::threading::NanoSleep(100);
          NoteBackoffSleep(100);
        } else {
          xe::threading::MaybeYield();
          NoteYield();
        }
#endif
      }
    } else {
      xe::threading::Sleep(std::chrono::milliseconds(timeout_ms));
      NoteTimedDelay(uint64_t(timeout_ms) * 1000000ull);
    }
  }

  return X_STATUS_SUCCESS;
}

namespace {
const char* XObjectTypeName(XObject::Type type) {
  switch (type) {
    case XObject::Type::Event:
      return "Event";
    case XObject::Type::Mutant:
      return "Mutant";
    case XObject::Type::Semaphore:
      return "Semaphore";
    case XObject::Type::Timer:
      return "Timer";
    case XObject::Type::Thread:
      return "Thread";
    case XObject::Type::File:
      return "File";
    case XObject::Type::IOCompletion:
      return "IOCompletion";
    case XObject::Type::NotifyListener:
      return "NotifyListener";
    default:
      return "Object";
  }
}
}  // namespace

void XThread::NoteObjectWait(XObject* object, uint64_t blocked_ns,
                             bool timed_out) {
  if (!cvars::guest_wait_profile) {
    return;
  }
  guest_wait_stats_.object_wait_count.fetch_add(1, std::memory_order_relaxed);
  guest_wait_stats_.object_wait_ns.fetch_add(blocked_ns,
                                             std::memory_order_relaxed);
  if (timed_out) {
    guest_wait_stats_.object_timeout_count.fetch_add(1,
                                                     std::memory_order_relaxed);
  }
  if (!object) {
    return;
  }
  const uint64_t key =
      (uint64_t(static_cast<uint32_t>(object->type())) << 32) |
      uint32_t(object->handle());
  std::lock_guard<std::mutex> lock(thread_lock_);
  auto& agg = object_wait_breakdown_[key];
  if (agg.label.empty()) {
    const std::string& name = object->name();
    agg.label = fmt::format("{}:{:08X}{}{}", XObjectTypeName(object->type()),
                            uint32_t(object->handle()),
                            name.empty() ? "" : " ", name);
  }
  agg.count++;
  agg.ns += blocked_ns;
}

void XThread::NoteYield() {
  if (!cvars::guest_wait_profile) {
    return;
  }
  guest_wait_stats_.yield_count.fetch_add(1, std::memory_order_relaxed);
}

void XThread::NoteBackoffSleep(uint64_t requested_ns) {
  if (!cvars::guest_wait_profile) {
    return;
  }
  guest_wait_stats_.backoff_count.fetch_add(1, std::memory_order_relaxed);
  guest_wait_stats_.backoff_ns.fetch_add(requested_ns,
                                         std::memory_order_relaxed);
}

void XThread::NoteTimedDelay(uint64_t requested_ns) {
  if (!cvars::guest_wait_profile) {
    return;
  }
  guest_wait_stats_.timed_delay_count.fetch_add(1, std::memory_order_relaxed);
  guest_wait_stats_.timed_delay_ns.fetch_add(requested_ns,
                                             std::memory_order_relaxed);
}

void XThread::NoteAlertableSleep(uint64_t slept_ns) {
  if (!cvars::guest_wait_profile) {
    return;
  }
  guest_wait_stats_.alertable_count.fetch_add(1, std::memory_order_relaxed);
  guest_wait_stats_.alertable_ns.fetch_add(slept_ns, std::memory_order_relaxed);
}

std::vector<XThread::ObjWaitAgg> XThread::CopyObjectWaitBreakdown() {
  std::vector<ObjWaitAgg> out;
  std::lock_guard<std::mutex> lock(thread_lock_);
  out.reserve(object_wait_breakdown_.size());
  for (auto& kv : object_wait_breakdown_) {
    out.push_back(kv.second);
  }
  return out;
}

struct ThreadSavedState {
  uint32_t thread_id;
  bool is_main_thread;  // Is this the main thread?
  bool is_running;

  uint32_t apc_head;
  uint32_t tls_static_address;
  uint32_t tls_dynamic_address;
  uint32_t tls_total_size;
  uint32_t pcr_address;
  uint32_t stack_base;        // High address
  uint32_t stack_limit;       // Low address
  uint32_t stack_alloc_base;  // Allocation address
  uint32_t stack_alloc_size;  // Allocation size

  // Context (invalid if not running)
  struct {
    uint64_t lr;
    uint64_t ctr;
    uint64_t r[32];
    double f[32];
    vec128_t v[128];
    uint32_t cr[8];
    uint32_t fpscr;
    uint8_t xer_ca;
    uint8_t xer_ov;
    uint8_t xer_so;
    uint8_t vscr_sat;
    uint32_t pc;
  } context;
};

bool XThread::Save(ByteStream* stream) {
  if (!guest_thread_) {
    // Host XThreads are expected to be recreated on their own.
    return false;
  }

  XELOGD("XThread {:08X} serializing...", handle());

  uint32_t pc = 0;
  if (running_) {
    pc = emulator()->processor()->StepToGuestSafePoint(thread_id_);
    if (!pc) {
      XELOGE("XThread {:08X} failed to save: could not step to a safe point!",
             handle());
      assert_always();
      return false;
    }
  }

  if (!SaveObject(stream)) {
    return false;
  }

  stream->Write(kThreadSaveSignature);
  stream->Write(thread_name_);

  ThreadSavedState state;
  state.thread_id = thread_id_;
  state.is_main_thread = main_thread_;
  state.is_running = running_;
  state.tls_static_address = tls_static_address_;
  state.tls_dynamic_address = tls_dynamic_address_;
  state.tls_total_size = tls_total_size_;
  state.pcr_address = pcr_address_;
  state.stack_base = stack_base_;
  state.stack_limit = stack_limit_;
  state.stack_alloc_base = stack_alloc_base_;
  state.stack_alloc_size = stack_alloc_size_;

  if (running_) {
    // Context information
    auto context = thread_state_->context();
    state.context.lr = context->lr;
    state.context.ctr = context->ctr;
    std::memcpy(state.context.r, context->r, 32 * 8);
    std::memcpy(state.context.f, context->f, 32 * 8);
    std::memcpy(state.context.v, context->v, 128 * 16);
    state.context.cr[0] = context->cr0.value;
    state.context.cr[1] = context->cr1.value;
    state.context.cr[2] = context->cr2.value;
    state.context.cr[3] = context->cr3.value;
    state.context.cr[4] = context->cr4.value;
    state.context.cr[5] = context->cr5.value;
    state.context.cr[6] = context->cr6.value;
    state.context.cr[7] = context->cr7.value;
    state.context.fpscr = context->fpscr.value;
    state.context.xer_ca = context->xer_ca;
    state.context.xer_ov = context->xer_ov;
    state.context.xer_so = context->xer_so;
    state.context.vscr_sat = context->vscr_sat;
    state.context.pc = pc;
  }

  stream->Write(&state, sizeof(ThreadSavedState));
  return true;
}

object_ref<XThread> XThread::Restore(KernelState* kernel_state,
                                     ByteStream* stream) {
  // Kind-of a hack, but we need to set the kernel state outside of the object
  // constructor so it doesn't register a handle with the object table.
  auto thread = new XThread(nullptr);
  thread->kernel_state_ = kernel_state;

  if (!thread->RestoreObject(stream)) {
    return nullptr;
  }

  if (stream->Read<uint32_t>() != kThreadSaveSignature) {
    XELOGE("Could not restore XThread - invalid magic!");
    return nullptr;
  }

  XELOGD("XThread {:08X}", thread->handle());

  thread->thread_name_ = stream->Read<std::string>();

  ThreadSavedState state;
  stream->Read(&state, sizeof(ThreadSavedState));
  thread->thread_id_ = state.thread_id;
  thread->main_thread_ = state.is_main_thread;
  thread->running_ = state.is_running;
  thread->tls_static_address_ = state.tls_static_address;
  thread->tls_dynamic_address_ = state.tls_dynamic_address;
  thread->tls_total_size_ = state.tls_total_size;
  thread->pcr_address_ = state.pcr_address;
  thread->stack_base_ = state.stack_base;
  thread->stack_limit_ = state.stack_limit;
  thread->stack_alloc_base_ = state.stack_alloc_base;
  thread->stack_alloc_size_ = state.stack_alloc_size;

  // Register now that we know our thread ID.
  kernel_state->RegisterThread(thread);

  thread->thread_state_ =
      new cpu::ThreadState(kernel_state->processor(), thread->thread_id_,
                           thread->stack_base_, thread->pcr_address_);

  if (state.is_running) {
    auto context = thread->thread_state_->context();
    context->kernel_state = kernel_state;
    context->lr = state.context.lr;
    context->ctr = state.context.ctr;
    std::memcpy(context->r, state.context.r, 32 * 8);
    std::memcpy(context->f, state.context.f, 32 * 8);
    std::memcpy(context->v, state.context.v, 128 * 16);
    context->cr0.value = state.context.cr[0];
    context->cr1.value = state.context.cr[1];
    context->cr2.value = state.context.cr[2];
    context->cr3.value = state.context.cr[3];
    context->cr4.value = state.context.cr[4];
    context->cr5.value = state.context.cr[5];
    context->cr6.value = state.context.cr[6];
    context->cr7.value = state.context.cr[7];
    context->fpscr.value = state.context.fpscr;
    context->xer_ca = state.context.xer_ca;
    context->xer_ov = state.context.xer_ov;
    context->xer_so = state.context.xer_so;
    context->vscr_sat = state.context.vscr_sat;

    // Always retain when starting - the thread owns itself until exited.
    thread->RetainHandle();

    xe::threading::Thread::CreationParameters params;
    params.create_suspended = true;  // Not done restoring yet.
    params.stack_size = 16_MiB;
    thread->thread_ = xe::threading::Thread::Create(params, [thread, state]() {
      // Set thread ID override. This is used by logging.
      xe::threading::set_current_thread_id(thread->handle());

      // Snapshot the name once for stable native-name + profiler use, matching
      // the Create() startup path (avoids two separate reference reads racing a
      // concurrent rename).
      const std::string thread_name = thread->name();

      // Set name immediately, if we have one.
      thread->thread_->set_name(thread_name);

      // Profiler needs to know about the thread.
      xe::Profiler::ThreadEnter(thread_name.c_str());

      current_xthread_tls_ = thread;
      current_thread_ = thread;

      // Acquire any mutants
      for (auto mutant : thread->pending_mutant_acquires_) {
        uint64_t timeout = 0;
        auto status = mutant->Wait(0, 0, 0, &timeout);
        assert_true(status == X_STATUS_SUCCESS);
      }
      thread->pending_mutant_acquires_.clear();

      // Execute user code.
      thread->running_ = true;

      uint32_t pc = state.context.pc;
      thread->kernel_state_->processor()->ExecuteRaw(thread->thread_state_, pc);

      current_thread_ = nullptr;
      current_xthread_tls_ = nullptr;

      xe::Profiler::ThreadExit();

      // Release the self-reference to the thread.
      thread->ReleaseHandle();
    });
    assert_not_null(thread->thread_);

    // Notify processor we were recreated.
    thread->emulator()->processor()->OnThreadCreated(
        thread->handle(), thread->thread_state(), thread);
  }

  return object_ref<XThread>(thread);
}

XHostThread::XHostThread(KernelState* kernel_state, uint32_t stack_size,
                         uint32_t creation_flags, std::function<int()> host_fn,
                         uint32_t guest_process)
    : XThread(kernel_state, stack_size, 0, 0, 0, creation_flags, false, false,
              guest_process),
      host_fn_(host_fn) {
  // By default host threads are not debugger suspendable. If the thread runs
  // any guest code this must be overridden.
  can_debugger_suspend_ = false;
}

void XHostThread::Execute() {
  XELOGKERNEL(
      "XThread::Execute thid {} (handle={:08X}, '{}', native={:08X}, <host>)",
      thread_id_, handle(), thread_name_snapshot(), thread_->system_id());
  // Let the kernel know we are starting.
  kernel_state()->OnThreadExecute(this);
  int ret = host_fn_();

  // Exit.
  Exit(ret);
}

}  // namespace kernel
}  // namespace xe
