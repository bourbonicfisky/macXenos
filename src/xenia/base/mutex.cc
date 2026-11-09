/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2015 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/mutex.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

#include "xenia/base/logging.h"
#include "xenia/base/threading.h"
#if XE_PLATFORM_WIN32 == 1
#include "xenia/base/platform_win.h"
#else
#include <dlfcn.h>
#include <execinfo.h>
#endif
#if XE_PLATFORM_LINUX == 1
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace xe {
#if XE_PLATFORM_WIN32 == 1 && XE_ENABLE_FAST_WIN32_MUTEX == 1

// xe_global_mutex: recursive mutex via SRWLOCK
void xe_global_mutex::lock() {
  DWORD self = GetCurrentThreadId();
  if (owner_thread_ == self) {
    ++recursion_count_;
    return;
  }
  AcquireSRWLockExclusive(&srwlock_);
  owner_thread_ = self;
  recursion_count_ = 1;
}

void xe_global_mutex::unlock() {
  if (--recursion_count_ == 0) {
    owner_thread_ = 0;
    ReleaseSRWLockExclusive(&srwlock_);
  }
}

bool xe_global_mutex::try_lock() {
  DWORD self = GetCurrentThreadId();
  if (owner_thread_ == self) {
    ++recursion_count_;
    return true;
  }
  if (TryAcquireSRWLockExclusive(&srwlock_)) {
    owner_thread_ = self;
    recursion_count_ = 1;
    return true;
  }
  return false;
}

// xe_fast_mutex: non-recursive mutex via SRWLOCK
void xe_fast_mutex::lock() {
  DWORD self = GetCurrentThreadId();
  if (owner_thread_ == self) {
    assert_always("xe_fast_mutex: recursive lock detected");
  }
  AcquireSRWLockExclusive(&srwlock_);
  owner_thread_ = self;
}

void xe_fast_mutex::unlock() {
  owner_thread_ = 0;
  ReleaseSRWLockExclusive(&srwlock_);
}

bool xe_fast_mutex::try_lock() {
  if (TryAcquireSRWLockExclusive(&srwlock_)) {
    owner_thread_ = GetCurrentThreadId();
    return true;
  }
  return false;
}
#elif XE_PLATFORM_LINUX == 1 && XE_ENABLE_FAST_LINUX_MUTEX == 1

namespace {

inline int futex_wait(std::atomic<uint32_t>* addr, uint32_t expected) {
  return syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected, nullptr,
                 nullptr, 0);
}

inline int futex_wake(std::atomic<uint32_t>* addr, int count) {
  return syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, count, nullptr, nullptr,
                 0);
}

inline pid_t gettid() { return static_cast<pid_t>(syscall(SYS_gettid)); }

}  // namespace

// xe_global_mutex implementation (recursive)
void xe_global_mutex::lock() {
  pid_t self = gettid();

  // Fast path: check if we already own it (recursive lock)
  if (owner_.load(std::memory_order_relaxed) == self) {
    ++recursion_count_;
    return;
  }

  // Try to acquire with a simple CAS first (uncontended case)
  uint32_t expected = 0;
  if (XE_LIKELY(state_.compare_exchange_strong(
          expected, 1, std::memory_order_acquire, std::memory_order_relaxed))) {
    owner_.store(self, std::memory_order_relaxed);
    recursion_count_ = 1;
    return;
  }

  lock_slow();
}

void xe_global_mutex::lock_slow() {
  pid_t self = gettid();

  // Spin phase
  for (int i = 0; i < XE_LINUX_MUTEX_SPINCOUNT; ++i) {
#if XE_ARCH_AMD64 == 1
    _mm_pause();
#endif
    uint32_t expected = 0;
    if (state_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
      owner_.store(self, std::memory_order_relaxed);
      recursion_count_ = 1;
      return;
    }
  }

  // Slow path: use futex
  while (true) {
    // Mark as contended (state = 2) and wait
    uint32_t state = state_.exchange(2, std::memory_order_acquire);
    if (state == 0) {
      // We got the lock while marking contended
      owner_.store(self, std::memory_order_relaxed);
      recursion_count_ = 1;
      return;
    }

    // Wait on futex
    futex_wait(&state_, 2);

    // Try to acquire after wakeup
    uint32_t expected = 0;
    if (state_.compare_exchange_strong(expected, 2, std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
      owner_.store(self, std::memory_order_relaxed);
      recursion_count_ = 1;
      return;
    }
  }
}

void xe_global_mutex::unlock() {
  if (--recursion_count_ > 0) {
    return;  // Still have recursive locks
  }

  owner_.store(0, std::memory_order_relaxed);

  // If state was 2 (contended), we need to wake a waiter
  if (state_.exchange(0, std::memory_order_release) == 2) {
    futex_wake(&state_, 1);
  }
}

bool xe_global_mutex::try_lock() {
  pid_t self = gettid();

  // Check for recursive lock
  if (owner_.load(std::memory_order_relaxed) == self) {
    ++recursion_count_;
    return true;
  }

  uint32_t expected = 0;
  if (state_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                     std::memory_order_relaxed)) {
    owner_.store(self, std::memory_order_relaxed);
    recursion_count_ = 1;
    return true;
  }
  return false;
}

// xe_fast_mutex implementation (non-recursive)
void xe_fast_mutex::lock() {
  // Fast path: uncontended
  uint32_t expected = 0;
  if (XE_LIKELY(state_.compare_exchange_strong(
          expected, 1, std::memory_order_acquire, std::memory_order_relaxed))) {
    return;
  }

  lock_slow();
}

void xe_fast_mutex::lock_slow() {
  // Spin phase
  for (int i = 0; i < XE_LINUX_MUTEX_SPINCOUNT; ++i) {
#if XE_ARCH_AMD64 == 1
    _mm_pause();
#endif
    uint32_t expected = 0;
    if (state_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
      return;
    }
  }

  // Slow path: use futex
  while (true) {
    // Mark as contended (state = 2) and wait
    uint32_t state = state_.exchange(2, std::memory_order_acquire);
    if (state == 0) {
      // We got the lock while marking contended
      return;
    }

    // Wait on futex
    futex_wait(&state_, 2);

    // Try to acquire after wakeup
    uint32_t expected = 0;
    if (state_.compare_exchange_strong(expected, 2, std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
      return;
    }
  }
}

void xe_fast_mutex::unlock() {
  // If state was 2 (contended), we need to wake a waiter
  if (state_.exchange(0, std::memory_order_release) == 2) {
    futex_wake(&state_, 1);
  }
}

bool xe_fast_mutex::try_lock() {
  uint32_t expected = 0;
  return state_.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed);
}

#endif

#if !(XE_PLATFORM_WIN32 == 1 && XE_ENABLE_FAST_WIN32_MUTEX == 1) && \
    !(XE_PLATFORM_LINUX == 1 && XE_ENABLE_FAST_LINUX_MUTEX == 1)
// Diagnostic: XE_GLOBAL_LOCK_LOG_US=N logs every outermost acquisition that
// waited N us or longer (with the thread seen holding the lock) and every
// outermost hold of N us or longer (with the host frames that released it).
// Off when unset.
namespace {
int64_t GlobalLockLogThresholdUs() {
  static const int64_t threshold = [] {
    const char* value = std::getenv("XE_GLOBAL_LOCK_LOG_US");
    return value ? int64_t(std::strtoll(value, nullptr, 10)) : int64_t(-1);
  }();
  return threshold;
}
int64_t GlobalLockNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
std::atomic<uint32_t> global_lock_owner{0};
thread_local uint32_t global_lock_depth = 0;
thread_local int64_t global_lock_acquired_us = 0;
thread_local int64_t global_lock_wait_us = 0;
thread_local uint32_t global_lock_waited_on = 0;

void NoteGlobalLockAcquired(int64_t wait_us, uint32_t waited_on) {
  global_lock_owner.store(threading::current_thread_id(),
                          std::memory_order_relaxed);
  global_lock_depth = 1;
  global_lock_acquired_us = GlobalLockNowUs();
  global_lock_wait_us = wait_us;
  global_lock_waited_on = waited_on;
}

void LogGlobalLockRelease(int64_t threshold_us, int64_t hold_us) {
  if (global_lock_wait_us >= threshold_us) {
    XELOGI("GlobalLock: waited {} us (held by {:08X})", global_lock_wait_us,
           global_lock_waited_on);
  }
  if (hold_us >= threshold_us) {
    void* frames[8];
    int count = backtrace(frames, 8);
    std::string where;
    for (int i = 2; i < count; ++i) {
      Dl_info info;
      if (dladdr(frames[i], &info) && info.dli_sname) {
        where += fmt::format(" {}+{:X}", info.dli_sname,
                             uintptr_t(frames[i]) - uintptr_t(info.dli_saddr));
      } else {
        where += fmt::format(" {}", frames[i]);
      }
    }
    XELOGI("GlobalLock: held {} us, released at{}", hold_us, where);
  }
}
}  // namespace

// The deferral opens before the lock is taken: a suspend that lands while the
// thread waits for it would otherwise park the thread the moment it owns it.
void xe_global_mutex::lock() {
  threading::BeginSuspendDeferral();
  if (GlobalLockLogThresholdUs() < 0) {
    mutex_.lock();
    return;
  }
  if (global_lock_depth) {
    mutex_.lock();
    ++global_lock_depth;
    return;
  }
  if (mutex_.try_lock()) {
    NoteGlobalLockAcquired(0, 0);
    return;
  }
  const uint32_t waited_on =
      global_lock_owner.load(std::memory_order_relaxed);
  const int64_t wait_start_us = GlobalLockNowUs();
  mutex_.lock();
  NoteGlobalLockAcquired(GlobalLockNowUs() - wait_start_us, waited_on);
}

void xe_global_mutex::unlock() {
  const int64_t threshold_us = GlobalLockLogThresholdUs();
  if (threshold_us < 0 || --global_lock_depth) {
    mutex_.unlock();
    threading::EndSuspendDeferral();
    return;
  }
  const int64_t hold_us = GlobalLockNowUs() - global_lock_acquired_us;
  global_lock_owner.store(0, std::memory_order_relaxed);
  mutex_.unlock();
  threading::EndSuspendDeferral();
  LogGlobalLockRelease(threshold_us, hold_us);
}

bool xe_global_mutex::try_lock() {
  threading::BeginSuspendDeferral();
  if (mutex_.try_lock()) {
    if (GlobalLockLogThresholdUs() >= 0) {
      if (global_lock_depth) {
        ++global_lock_depth;
      } else {
        NoteGlobalLockAcquired(0, 0);
      }
    }
    return true;
  }
  threading::EndSuspendDeferral();
  return false;
}
#endif

global_mutex_type& global_critical_region::mutex() {
  static global_mutex_type global_mutex;
  return global_mutex;
}

}  // namespace xe
