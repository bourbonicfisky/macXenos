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
#include "xenia/cpu/thread_state.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_threading.h"
#include "xenia/kernel/xthread.h"

namespace xe {
namespace kernel {

XTimer::XTimer(KernelState* kernel_state)
    : XObject(kernel_state, kObjectType) {}

XTimer::~XTimer() {
  if (native_state_) {
    {
      std::lock_guard<std::mutex> lock(native_state_->mutex);
      ++native_state_->generation;
      native_state_->pending = false;
    }
    // Drain host callbacks before destroying the kernel object.
    timer_->Cancel();
    timer_.reset();
  }
}

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

void XTimer::InitializeNative(void* native_ptr, X_DISPATCH_HEADER* header) {
  Initialize(header->type - 8);
  native_state_ = std::make_shared<NativeState>();
  native_state_->guest_ptr = memory()->HostToGuestVirtual(native_ptr);
  native_state_->event =
      manual_reset_
          ? xe::threading::Event::CreateManualResetEvent(header->signal_state != 0)
          : xe::threading::Event::CreateAutoResetEvent(header->signal_state != 0);
  SetNativePointer(native_state_->guest_ptr, true);
}

xe::threading::WaitHandle* XTimer::GetWaitHandle() {
  // Native timers publish their guest state before waking a waiter. The host
  // timer is only the scheduler; its signal precedes its callback and cannot
  // keep repeated expirations in sync with guest signal consumption.
  if (native_state_) {
    return native_state_->event.get();
  }
  return timer_.get();
}

bool XTimer::SetNativeTimer(int64_t due_time, uint32_t period_ms, uint32_t dpc,
                           std::function<void()> callback) {
  using xe::chrono::WinSystemClock;
  using xe::chrono::XSystemClock;
  std::lock_guard<std::mutex> schedule_lock(timer_lock_);
  auto state = native_state_;
  uint64_t generation;
  bool previous;
  // An absolute time that has passed, down to 0, expires before the call
  // returns: the console's signal word is already 1 then, for both timer
  // types and for a periodic timer's first expiry (ke_timer_absolute probe).
  const bool expired =
      due_time >= 0 && due_time <= int64_t(Clock::QueryGuestSystemTime());
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    previous = state->pending;
    generation = ++state->generation;
    state->pending = true;
    state->event->Reset();
    auto timer = memory()->TranslateVirtual<X_KTIMER*>(state->guest_ptr);
    timer->header.inserted = 1;
    timer->header.signal_state = 0;
    timer->header.absolute = due_time >= 0;
    timer->due_time =
        due_time < 0
            ? Clock::QueryGuestSystemTime() - static_cast<uint64_t>(due_time)
            : static_cast<uint64_t>(due_time);
    timer->dpc = dpc;
    timer->period = period_ms;
    if (expired) {
      state->pending = period_ms != 0;
      timer->header.inserted = state->pending;
      state->event->Set();
      timer->header.signal_state = state->event->Query().state;
    }
  }
  if (expired) {
    if (callback) {
      callback();
    }
    if (!period_ms) {
      // Nothing left to schedule. Drop the schedule this one replaced; its
      // callback, if already on its way, sees the new generation and stops.
      timer_->Cancel();
      return previous;
    }
  }
  auto due = expired
                 ? date::clock_cast<WinSystemClock>(
                       XSystemClock::now() +
                       std::chrono::milliseconds(period_ms))
                 : due_time < 0
                 ? date::clock_cast<WinSystemClock>(
                       XSystemClock::now() +
                       xe::chrono::hundrednanoseconds(-due_time))
                 : date::clock_cast<WinSystemClock>(
                       XSystemClock::from_file_time(due_time));
  auto mem = memory();
  auto fire =
      [state, generation, mem, period_ms, callback = std::move(callback)] {
        {
          std::lock_guard<std::mutex> lock(state->mutex);
          if (generation != state->generation) {
            return;
          }
          state->pending = period_ms != 0;
          auto timer = mem->TranslateVirtual<X_KTIMER*>(state->guest_ptr);
          timer->header.inserted = state->pending;
          state->event->Set();
          timer->header.signal_state = state->event->Query().state;
        }
        if (callback) {
          callback();
        }
      };
  bool success = period_ms
                     ? timer_->SetRepeatingAt(
                           due, std::chrono::milliseconds(
                                    Clock::ScaleGuestDurationMillis(period_ms)),
                           std::move(fire))
                     : timer_->SetOnceAt(due, std::move(fire));
  if (!success) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->pending = false;
    ++state->generation;
    mem->TranslateVirtual<X_KTIMER*>(state->guest_ptr)->header.inserted = 0;
    XELOGE("Unable to schedule guest timer {:08X}", state->guest_ptr);
  }
  return previous;
}

bool XTimer::CancelNativeTimer() {
  std::lock_guard<std::mutex> schedule_lock(timer_lock_);
  bool previous;
  {
    std::lock_guard<std::mutex> lock(native_state_->mutex);
    previous = native_state_->pending;
    native_state_->pending = false;
    ++native_state_->generation;
    memory()->TranslateVirtual<X_KTIMER*>(native_state_->guest_ptr)
        ->header.inserted = 0;
  }
  // Cancellation removes the timer from the queue, without clearing a signal
  // that has already been delivered.
  timer_->Cancel();
  return previous;
}

void XTimer::WaitCallback() {
  if (native_state_) {
    std::lock_guard<std::mutex> lock(native_state_->mutex);
    auto timer = memory()->TranslateVirtual<X_KTIMER*>(native_state_->guest_ptr);
    // A periodic timer remains queued when a waiter consumes its signal. Use
    // the event's current state so a subsequent expiry is not cleared here.
    timer->header.signal_state = native_state_->event->Query().state;
  }
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
  } else if (due_time <= int64_t(Clock::QueryGuestSystemTime())) {
    // An absolute time that has passed, down to 0, expires at once.
    due_tp = date::clock_cast<WinSystemClock>(XSystemClock::now());
  } else {
    due_tp = date::clock_cast<WinSystemClock>(
        XSystemClock::from_file_time(due_time));
  }

  // The schedule this replaces takes its undelivered APC with it.
  RetractApc();

  // Stash routine for callback.
  callback_thread_ = XThread::GetCurrentThread();
  callback_thread_handle_ = callback_thread_ ? callback_thread_->handle() : 0;
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
    auto cb_generation = apc_generation_;
    const uint64_t cb_armed = cb_generation->load();
    callback = [cb_kernel_state, cb_thread_handle, cb_routine, cb_routine_arg,
                cb_generation, cb_armed]() {
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
                                 cb_routine_arg, time_low, time_high,
                                 cb_generation, cb_armed]() {
        if (cb_generation->load() != cb_armed) {
          // Cancelled or armed again since this expiry.
          return;
        }
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
        if (cb_generation->load() != cb_armed) {
          // The cancel ran while this was being queued.
          xboxkrnl::xeRemoveTimerApcs(cb_thread_handle, cb_routine,
                                      cb_routine_arg,
                                      cpu::ThreadState::Get()->context());
        }
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

void XTimer::RetractApc() {
  ++*apc_generation_;
  auto thread_state = cpu::ThreadState::Get();
  if (callback_routine_ && callback_thread_handle_ && thread_state) {
    xboxkrnl::xeRemoveTimerApcs(callback_thread_handle_, callback_routine_,
                                callback_routine_arg_,
                                thread_state->context());
  }
}

X_STATUS XTimer::Cancel() {
  std::lock_guard<std::mutex> lock(timer_lock_);
  X_STATUS status =
      timer_->Cancel() ? X_STATUS_SUCCESS : X_STATUS_UNSUCCESSFUL;
  RetractApc();
  return status;
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
