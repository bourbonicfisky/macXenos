/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xtimer.h"

#include <cstdlib>
#include <cstring>

#include "xenia/base/logging.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xthread.h"

namespace xe {
namespace kernel {

XTimer::XTimer(KernelState* kernel_state)
    : XObject(kernel_state, kObjectType) {}

XTimer::~XTimer() = default;

void XTimer::Initialize(uint32_t timer_type) {
  assert_false(timer_);
  switch (timer_type) {
    case 0:  // NotificationTimer
      timer_ = xe::threading::Timer::CreateManualResetTimer();
      manual_reset_ = true;
      break;
    case 1:  // SynchronizationTimer
      timer_ = xe::threading::Timer::CreateSynchronizationTimer();
      break;
    default:
      assert_always();
      break;
  }
  assert_not_null(timer_);
}

X_STATUS XTimer::SetTimer(int64_t due_time, uint32_t period_ms,
                          uint32_t routine, uint32_t routine_arg, bool resume) {
  using xe::chrono::WinSystemClock;
  using xe::chrono::XSystemClock;

  std::lock_guard<std::mutex> lock(timer_lock_);

  period_ms = Clock::ScaleGuestDurationMillis(period_ms);
  WinSystemClock::time_point due_tp;
  if (due_time < 0) {
    // Any timer implementation uses absolute times eventually, convert as early
    // as possible for increased accuracy
    auto after = xe::chrono::hundrednanoseconds(-due_time);
    due_tp = date::clock_cast<WinSystemClock>(XSystemClock::now() + after);
  } else {
    due_tp = date::clock_cast<WinSystemClock>(
        XSystemClock::from_file_time(due_time));
  }

  // Stash routine for callback.
  callback_thread_ = XThread::GetCurrentThread();
  callback_routine_ = routine;
  callback_routine_arg_ = routine_arg;

  // This callback will only be issued when the timer is fired.
  // Capture values by value to avoid racing with a future SetTimer() call.
  //
  // The requesting thread is captured by HANDLE and looked up when the timer
  // fires (2026-09-18): a raw XThread* dangles if the thread exits before its
  // timer fires, and it is null when NtSetTimer is issued from a context with
  // no current guest thread — Mass Effect (4D5307E8) crashed the
  // timer-queue thread with a null dereference in this lambda. Neither case
  // has a thread to run the APC on, so the fire is logged and dropped.
  std::function<void()> callback = nullptr;
  // XE_KERNEL_TIMER_APC=0: diagnostic rollback to the pre-2026-09-18 effect
  // (timer APCs never delivered), for A/B against titles that regress.
  static const bool s_timer_apc_off = [] {
    const char* v = std::getenv("XE_KERNEL_TIMER_APC");
    return v && std::strcmp(v, "0") == 0;
  }();
  if (callback_routine_ && !s_timer_apc_off) {
    if (!callback_thread_) {
      XELOGW(
          "XTimer::SetTimer: routine {:08X} requested with no current guest "
          "thread; the timer's APC will have nowhere to run",
          callback_routine_);
    }
    const uint32_t cb_thread_handle =
        callback_thread_ ? callback_thread_->handle() : 0;
    auto cb_routine = callback_routine_;
    auto cb_routine_arg = callback_routine_arg_;
    KernelState* cb_kernel_state = kernel_state();
    callback = [cb_kernel_state, cb_thread_handle, cb_routine,
                cb_routine_arg]() {
      // Queue APC to call back routine with (arg, low, high).
      // It'll be executed on the thread that requested the timer.
      uint64_t time = xe::Clock::QueryGuestSystemTime();
      uint32_t time_low = static_cast<uint32_t>(time);
      uint32_t time_high = static_cast<uint32_t>(time >> 32);
      // This runs on the host timer-queue thread, which has no guest thread
      // state; EnqueueApc needs one (the APC spinlock takes the caller's PCR
      // from the context). Hop to the kernel dispatch thread, as the deferred
      // overlapped completions do.
      cb_kernel_state->Dispatch([cb_kernel_state, cb_thread_handle, cb_routine,
                                 cb_routine_arg, time_low, time_high]() {
        auto cb_thread =
            cb_thread_handle
                ? cb_kernel_state->object_table()->LookupObject<XThread>(
                      cb_thread_handle)
                : nullptr;
        if (!cb_thread) {
          XELOGW(
              "XTimer fired for {:08X}({:08X}) but its thread (handle {:08X}) "
              "is gone or was never a guest thread; APC dropped",
              cb_routine, cb_routine_arg, cb_thread_handle);
          return;
        }
        XELOGI(
            "XTimer enqueuing timer callback to {:08X}({:08X}, {:08X}, {:08X})",
            cb_routine, cb_routine_arg, time_low, time_high);
        cb_thread->EnqueueApc(cb_routine, cb_routine_arg, time_low, time_high,
                              /*from_timer=*/true);
      });
    };
  }

  bool result;
  if (!period_ms) {
    result = timer_->SetOnceAt(due_tp, std::move(callback));
  } else {
    result = timer_->SetRepeatingAt(
        due_tp, std::chrono::milliseconds(period_ms), std::move(callback));
  }

  if (resume) {
    XThread::SetLastError(X_ERROR_NOT_SUPPORTED);
    return X_STATUS_TIMER_RESUME_IGNORED;
  }

  return result ? X_STATUS_SUCCESS : X_STATUS_UNSUCCESSFUL;
}

X_STATUS XTimer::Cancel() {
  std::lock_guard<std::mutex> lock(timer_lock_);
  return timer_->Cancel() ? X_STATUS_SUCCESS : X_STATUS_UNSUCCESSFUL;
}

bool XTimer::IsSignaled() {
  // A zero wait on the host timer would consume a synchronization timer's
  // signal so only a notification timer can be read.
  return manual_reset_ && xe::threading::Wait(timer_.get(), false,
                                              std::chrono::milliseconds(0)) ==
                              xe::threading::WaitResult::kSuccess;
}

}  // namespace kernel
}  // namespace xe
