/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/threading.h"

#include "xenia/base/assert.h"
#include "xenia/base/chrono_steady_cast.h"
#include "xenia/base/platform.h"
#include "xenia/base/threading_timer_queue.h"

#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/semaphore.h>
#endif
#include <signal.h>
#include <sys/resource.h>
#if XE_PLATFORM_MAC
#include <pthread/qos.h>
#endif
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <array>
#include <cstddef>
#include <ctime>

#include "logging.h"

#if XE_PLATFORM_ANDROID
#include <dlfcn.h>

#include "xenia/base/main_android.h"
#include "xenia/base/string_util.h"
#endif

#if XE_PLATFORM_LINUX
// SIGEV_THREAD_ID in timer_create(...) is a Linux extension
#define XE_HAS_SIGEV_THREAD_ID 1
#ifdef __GLIBC__
#define sigev_notify_thread_id _sigev_un._tid
#endif
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#define gettid() syscall(SYS_gettid)
#endif
#else
#define XE_HAS_SIGEV_THREAD_ID 0
#endif

#include "xenia/base/cvar.h"

DEFINE_bool(
    threading_blocking_waitany, true,
    "Event-driven waits on events and WaitMultiple(WaitAny) (default on). A "
    "waiter registers a latched wait-block with each handle; a handle's Signal "
    "hands the wake to the registered waiters under lock (every waiter of a "
    "manual-reset event, one for an auto-reset event), so a Set that is Reset "
    "before the woken threads run still releases them, as on a console. Off = "
    "a wait rechecks the event once it runs, and WaitAny polls every 1 ms, "
    "both of which lose such a Set. Advanced: turn off only to bisect a "
    "threading regression.",
    "Threading");
// 2026-11-10: default off -> on. With it off a manual-reset Set followed by a
// Reset released none of the waiters parked on the event.
UPDATE_from_bool(threading_blocking_waitany, 2026, 11, 10, 0, false);
DEFINE_bool(
    threading_blocking_waitall, true,
    "Event-driven WaitMultiple(WaitAll) (default on). The waiter registers a "
    "notify-only wait-block with every member; any member's Signal wakes it to "
    "re-run the all-signaled check under the member locks (nothing is consumed "
    "by the wake itself), with a 1 ms backstop re-check for members whose "
    "Signal is not hooked (mutant, timer, thread). Off = the previous 1 ms "
    "poll, which cost titles that join worker jobs every frame up to a "
    "millisecond per join (Resident Evil 5 ran at 12 fps instead of 30). "
    "Advanced: turn off only to bisect a threading regression.",
    "Threading");
// 2026-10-17: default off -> on after the regression check (21 titles over
// two machines, no loser; RE5 12.4 -> 29.1 fps, GTA IV +3.5).
UPDATE_from_bool(threading_blocking_waitall, 2026, 10, 17, 0, false);

#if XE_PLATFORM_MAC
DEFINE_string(
    macos_thread_cancel_mode, "async",
    "macOS diagnostic probe for guest-thread teardown. async is current "
    "behavior. deferred prevents pthread cancellation from landing at arbitrary "
    "instructions; it may convert heap corruption into a hang if a guest thread "
    "never reaches a cancellation point. Diagnostic only, not a shippable fix.",
    "Threading");
#endif

namespace xe {
namespace threading {

// Latched wait-block for the event-driven WaitMultiple(WaitAny) path. One per
// in-flight WaitMultiple call; registered with each member handle. A handle's
// Signal sets `satisfied`/`fired_index` under `m` and notifies `cv` — the latch
// survives a subsequent Reset, which is exactly what the poll path lost.
struct WaitBlock {
  std::mutex m;
  std::condition_variable cv;
  int fired_index = -1;
  bool satisfied = false;
  // WaitAll: a Signal only pokes the waiter (sets `pending`, notifies `cv`),
  // consumes nothing and leaves the block registered; the waiter re-runs the
  // all-signaled check itself and unregisters when done.
  bool notify_only = false;
  bool pending = false;
};

// A thread's user callbacks (Thread::QueueUserCallback) and the wake for its
// alertable waits. A callback queued while the thread is blocked in an
// alertable wait ends that wait with kUserCallback, and the thread then runs
// the queue itself. A signal cannot do this: its handler returns into the
// pthread_cond_wait it interrupted, so the wait never ended, and a guest APC
// queued from another thread stayed undelivered until the target began its
// next alertable wait (Quantum Theory parks its job threads in one for good
// and wakes them with empty APCs).
struct ThreadAlert {
  std::mutex m;
  std::deque<std::function<void()>> callbacks;  // under m
  // What the thread is blocked on while inside an alertable wait (under m).
  // Queue() notifies it holding its mutex, which orders the notify after the
  // waiter's check of `pending`.
  std::mutex* wait_mutex = nullptr;
  std::condition_variable* wait_cv = nullptr;
  // The thread was terminated from outside; its waits are gone (under m).
  bool closed = false;
  std::atomic<bool> pending{false};

  void Queue(std::function<void()> callback) {
    // m is one a resumer of this thread may need, on either side.
    SuspendDeferralScope suspend_deferral;
    std::lock_guard<std::mutex> lock(m);
    callbacks.push_back(std::move(callback));
    pending.store(true);
    if (wait_mutex) {
      std::lock_guard<std::mutex> wait_lock(*wait_mutex);
      wait_cv->notify_all();
    }
  }

  // Runs the queued callbacks, on the owning thread.
  void Run() {
    std::deque<std::function<void()>> run;
    {
      SuspendDeferralScope suspend_deferral;
      std::lock_guard<std::mutex> lock(m);
      run.swap(callbacks);
      pending.store(false);
    }
    for (auto& callback : run) {
      callback();
    }
  }

  void Close() {
    SuspendDeferralScope suspend_deferral;
    std::lock_guard<std::mutex> lock(m);
    closed = true;
    wait_mutex = nullptr;
    wait_cv = nullptr;
  }
};

// The calling thread's alert, or null for a thread this layer does not know.
static ThreadAlert* CurrentThreadAlert();

// The calling thread's alert while it is inside an alertable wait, else null.
thread_local ThreadAlert* alertable_wait_ = nullptr;

// For its scope, points the calling thread's alert at the mutex/condition pair
// a wait blocks on. Construct and destroy it holding neither that mutex nor a
// handle's: Queue() takes them after the alert's own.
class AlertArm {
 public:
  AlertArm(std::mutex* wait_mutex, std::condition_variable* wait_cv)
      : alert_(alertable_wait_) {
    if (alert_) {
      SuspendDeferralScope suspend_deferral;
      std::lock_guard<std::mutex> lock(alert_->m);
      if (!alert_->closed) {
        alert_->wait_mutex = wait_mutex;
        alert_->wait_cv = wait_cv;
      }
    }
  }
  ~AlertArm() {
    if (alert_) {
      SuspendDeferralScope suspend_deferral;
      std::lock_guard<std::mutex> lock(alert_->m);
      alert_->wait_mutex = nullptr;
      alert_->wait_cv = nullptr;
    }
  }
  AlertArm(const AlertArm&) = delete;
  AlertArm& operator=(const AlertArm&) = delete;

  // Read under the armed mutex by the wait's predicate.
  bool pending() const { return alert_ && alert_->pending.load(); }

 private:
  ThreadAlert* const alert_;
};

#if XE_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* android_libc_;
// API 26+.
static int (*android_pthread_getname_np_)(pthread_t pthread, char* buf,
                                          size_t n);

void AndroidInitialize() {
  if (xe::GetAndroidApiLevel() >= 26) {
    android_libc_ = dlopen("libc.so", RTLD_NOW);
    assert_not_null(android_libc_);
    if (android_libc_) {
      android_pthread_getname_np_ =
          reinterpret_cast<decltype(android_pthread_getname_np_)>(
              dlsym(android_libc_, "pthread_getname_np"));
      assert_not_null(android_pthread_getname_np_);
    }
  }
}

void AndroidShutdown() {
  android_pthread_getname_np_ = nullptr;
  if (android_libc_) {
    dlclose(android_libc_);
    android_libc_ = nullptr;
  }
}
#endif

template <typename _Rep, typename _Period>
timespec DurationToTimeSpec(std::chrono::duration<_Rep, _Period> duration) {
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  auto div = ldiv(nanoseconds.count(), 1000000000L);
  return timespec{div.quot, div.rem};
}

// Thread interruption is done using user-defined signals
// This implementation uses the SIGRTMAX - SIGRTMIN to signal to a thread
// gdb tip, for SIG = SIGRTMIN + SignalType : handle SIG nostop
// lldb tip, for SIG = SIGRTMIN + SignalType : process handle SIG -s false
enum class SignalType {
  kThreadSuspend,
#if XE_PLATFORM_ANDROID
  // pthread_cancel is not available on Android, using a signal handler for
  // simplified PTHREAD_CANCEL_ASYNCHRONOUS-like behavior - not disabling
  // cancellation currently, so should be enough.
  kThreadTerminate,
#endif
  k_Count
};

#if XE_PLATFORM_MAC
// macOS has no POSIX realtime signals; map the signal types onto the user
// signals SIGUSR1/SIGUSR2, which are consecutive.
#define XE_THREAD_SIGNAL_BASE SIGUSR1
#define XE_THREAD_SIGNAL_MAX SIGUSR2
#else
#define XE_THREAD_SIGNAL_BASE SIGRTMIN
#define XE_THREAD_SIGNAL_MAX SIGRTMAX
#endif

int GetSystemSignal(SignalType num) {
  auto result = XE_THREAD_SIGNAL_BASE + static_cast<int>(num);
  assert_true(result <= XE_THREAD_SIGNAL_MAX);
  return result;
}

SignalType GetSystemSignalType(int num) {
  return static_cast<SignalType>(num - XE_THREAD_SIGNAL_BASE);
}

std::array<std::atomic<bool>, static_cast<size_t>(SignalType::k_Count)>
    signal_handler_installed = {};

static void signal_handler(int signal, siginfo_t* info, void* context);

// XE_THREAD_SUSPEND_DEFER=0: park in the handler at once (the old behaviour).
// Read before the first suspend signal handler is installed.
static bool suspend_deferral_enabled_ = true;

// The semaphore a suspended thread parks on inside the suspend signal handler.
// macOS does not implement unnamed POSIX semaphores: sem_init fails with
// ENOSYS and every sem_wait on the result fails at once, so a "suspended"
// thread returned from the handler and kept running. Mach semaphores are
// plain kernel traps, safe to use from a signal handler.
class SuspendSemaphore {
 public:
  SuspendSemaphore() {
#if XE_PLATFORM_MAC
    valid_ = semaphore_create(mach_task_self(), &sem_, SYNC_POLICY_FIFO, 0) ==
             KERN_SUCCESS;
#else
    valid_ = sem_init(&sem_, 0, 0) == 0;
#endif
  }
  ~SuspendSemaphore() { Destroy(); }
  SuspendSemaphore(const SuspendSemaphore&) = delete;
  SuspendSemaphore& operator=(const SuspendSemaphore&) = delete;

  void Post() {
    if (!valid_) return;
#if XE_PLATFORM_MAC
    semaphore_signal(sem_);
#else
    sem_post(&sem_);
#endif
  }
  // False when the semaphore is gone or broken. An interruption returns true
  // like a wakeup, so the caller re-checks its condition: a suspend signal
  // that lands in a wait outside the handler (a self-suspend, or a deferred
  // suspend being served) runs a nested wait that takes the resume's post,
  // and retrying here would then wait for a post that never comes.
  bool Wait() {
    if (!valid_) return false;
#if XE_PLATFORM_MAC
    kern_return_t kr = semaphore_wait(sem_);
    return kr == KERN_SUCCESS || kr == KERN_ABORTED;
#else
    int ret = sem_wait(&sem_);
    return ret == 0 || errno == EINTR;
#endif
  }
  void Destroy() {
    if (!valid_) return;
    valid_ = false;
#if XE_PLATFORM_MAC
    semaphore_destroy(mach_task_self(), sem_);
#else
    sem_destroy(&sem_);
#endif
  }

 private:
#if XE_PLATFORM_MAC
  semaphore_t sem_ = 0;
#else
  sem_t sem_;
#endif
  std::atomic<bool> valid_{false};
};


void install_signal_handler(SignalType type) {
  if (type == SignalType::kThreadSuspend) {
    static const bool defer = [] {
      const char* value = std::getenv("XE_THREAD_SUSPEND_DEFER");
      return !(value && std::strcmp(value, "0") == 0);
    }();
    suspend_deferral_enabled_ = defer;
  }
  bool expected = false;
  if (!signal_handler_installed[static_cast<size_t>(type)]
           .compare_exchange_strong(expected, true)) {
    return;  // Already installed
  }
  struct sigaction action{};
  action.sa_flags = SA_SIGINFO | SA_RESTART;
  action.sa_sigaction = signal_handler;
  sigemptyset(&action.sa_mask);
  if (sigaction(GetSystemSignal(type), &action, nullptr) != 0) {
    // Failed to install, reset the flag
    signal_handler_installed[static_cast<size_t>(type)] = false;
  }
}

// TODO(dougvj)
void EnableAffinityConfiguration() {}

// The calling thread's id as system_id() reports it, for lifecycle logs
// (SYS_gettid is not a thread id on macOS).
static uint32_t LifecycleSelfId() {
#if XE_PLATFORM_MAC
  uint64_t tid = 0;
  pthread_threadid_np(nullptr, &tid);
  return static_cast<uint32_t>(tid);
#else
  return static_cast<uint32_t>(syscall(SYS_gettid));
#endif
}

// uint64_t ticks() { return mach_absolute_time(); }

uint32_t current_thread_system_id() {
  return static_cast<uint32_t>(syscall(SYS_gettid));
}

void MaybeYield() {
  sched_yield();
  __sync_synchronize();
}

void SyncMemory() { __sync_synchronize(); }

static void SleepFor(timespec rqtp) {
  timespec rmtp = {};
  auto p_rqtp = &rqtp;
  auto p_rmtp = &rmtp;
  int ret = 0;
  do {
    ret = nanosleep(p_rqtp, p_rmtp);
    // Swap requested for remaining in case of signal interruption
    // in which case, we start sleeping again for the remainder
    std::swap(p_rqtp, p_rmtp);
  } while (ret == -1 && errno == EINTR);
}

void Sleep(std::chrono::microseconds duration) {
  SleepFor(DurationToTimeSpec(duration));
}

void NanoSleep(int64_t duration) {
  // Not Sleep(nanoseconds(duration)): the conversion to microseconds
  // truncates, so anything under 1000 ns became nanosleep(0).
  if (duration <= 0) {
    return;
  }
  SleepFor({static_cast<time_t>(duration / 1000000000LL),
            static_cast<long>(duration % 1000000000LL)});
}

SleepResult AlertableSleep(std::chrono::microseconds duration) {
  ThreadAlert* alert = CurrentThreadAlert();
  if (!alert) {
    Sleep(duration);
    return SleepResult::kSuccess;
  }
  if (!alert->pending.load()) {
    if (duration.count() <= 0) {
      Sleep(duration);
    } else {
      std::mutex mutex;
      std::condition_variable cond;
      alertable_wait_ = alert;
      {
        AlertArm arm(&mutex, &cond);
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait_for(lock, duration, [&] { return arm.pending(); });
      }
      alertable_wait_ = nullptr;
    }
    if (!alert->pending.load()) {
      return SleepResult::kSuccess;
    }
  }
  alert->Run();
  return SleepResult::kAlerted;
}

TlsHandle AllocateTlsHandle() {
  auto key = static_cast<pthread_key_t>(-1);
  auto res = pthread_key_create(&key, nullptr);
  assert_zero(res);
  assert_true(key != static_cast<pthread_key_t>(-1));
  return static_cast<TlsHandle>(key);
}

bool FreeTlsHandle(TlsHandle handle) { return pthread_key_delete(handle) == 0; }

uintptr_t GetTlsValue(TlsHandle handle) {
  return reinterpret_cast<uintptr_t>(pthread_getspecific(handle));
}

bool SetTlsValue(TlsHandle handle, uintptr_t value) {
  return pthread_setspecific(handle, reinterpret_cast<void*>(value)) == 0;
}

class PosixConditionBase {
 public:
  PosixConditionBase() {
#if XE_PLATFORM_MAC
    // macOS does not support robust mutexes (no PTHREAD_MUTEX_ROBUST /
    // pthread_mutexattr_setrobust / pthread_mutex_consistent). Leave the mutex
    // as a plain default std::mutex; EOWNERDEAD recovery is not available, but
    // a plain mutex never reports it.
#else
    // Initialize as robust mutex to handle thread termination gracefully
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);

    // Get the native handle and set it as robust
    auto native_mutex = static_cast<pthread_mutex_t*>(mutex_.native_handle());
    pthread_mutex_destroy(native_mutex);      // Destroy default mutex
    pthread_mutex_init(native_mutex, &attr);  // Reinit as robust
    pthread_mutexattr_destroy(&attr);
#endif  // XE_PLATFORM_MAC
  }

  virtual ~PosixConditionBase() = default;
  virtual bool Signal() = 0;

  WaitResult Wait(std::chrono::milliseconds timeout) {
    // Route single waits on an event through the same latched wait-block path
    // as WaitAny so a Signal is delivered (and auto-reset consumed) exactly once
    // across single AND multi waiters — no transient loss, no steal. The other
    // objects stay signaled until a waiter takes them, so they keep the
    // condition variable, which their Signal notifies directly.
    if (cvars::threading_blocking_waitany && latches_single_waits()) {
      return WaitLatched(timeout);
    }

    // An alertable wait also ends when a user callback is queued on the
    // thread; the object, if signaled as well, is taken first.
    AlertArm alert(&mutex_, &cond_);
    auto predicate = [&] { return this->signaled() || alert.pending(); };

    // Handle robust mutex locking
    auto native_mutex = static_cast<pthread_mutex_t*>(mutex_.native_handle());
    int lock_result = pthread_mutex_lock(native_mutex);
    if (lock_result == EOWNERDEAD) {
      // Recover from dead owner
#if !XE_PLATFORM_MAC
      // macOS has no pthread_mutex_consistent; with a plain (non-robust) mutex
      // EOWNERDEAD never occurs, so this recovery path is unreachable there.
      pthread_mutex_consistent(native_mutex);
#endif  // !XE_PLATFORM_MAC
    } else if (lock_result != 0) {
      return WaitResult::kFailed;
    }

    std::unique_lock<std::mutex> lock(mutex_, std::adopt_lock);

    if (!predicate()) {
      if (timeout == std::chrono::milliseconds::max()) {
        cond_.wait(lock, predicate);
      } else {
        cond_.wait_for(lock, timeout, predicate);
      }
    }
    if (signaled()) {
      post_execution();
      return WaitResult::kSuccess;
    }
    return alert.pending() ? WaitResult::kUserCallback : WaitResult::kTimeout;
  }

  // A single wait through a wait-block: WaitAnyBlocking for one handle whose
  // every signal source delivers to the registered blocks (an event), so it
  // needs no backstop re-poll and a parked thread stays parked until the Set
  // or the timeout. A poll (zero timeout) only looks, under the lock.
  WaitResult WaitLatched(std::chrono::milliseconds timeout) {
    WaitBlock block;
    AlertArm alert(&block.m, &block.cv);
    {
      std::lock_guard<std::mutex> hl(mutex_);
      if (signaled()) {
        post_execution();
        return WaitResult::kSuccess;
      }
      if (alert.pending()) {
        return WaitResult::kUserCallback;
      }
      if (timeout == std::chrono::milliseconds::zero()) {
        return WaitResult::kTimeout;
      }
      wait_blocks_.push_back({&block, 0});
    }
    {
      std::unique_lock<std::mutex> bl(block.m);
      auto satisfied = [&] { return block.satisfied || alert.pending(); };
      if (timeout == std::chrono::milliseconds::max()) {
        block.cv.wait(bl, satisfied);
      } else {
        block.cv.wait_for(bl, timeout, satisfied);
      }
    }
    // Unregister under the handle's lock even when satisfied (the signaller
    // has erased the block by then): it holds that lock until it is done with
    // the block, which lives on this stack.
    {
      std::lock_guard<std::mutex> hl(mutex_);
      for (auto it = wait_blocks_.begin(); it != wait_blocks_.end(); ++it) {
        if (it->first == &block) {
          wait_blocks_.erase(it);
          break;
        }
      }
    }
    // Read the latch only now: a Set that arrived after the timeout but before
    // the unregistration was handed to (and, for an auto-reset event, consumed
    // for) this wait.
    std::lock_guard<std::mutex> bl(block.m);
    if (block.satisfied) {
      return WaitResult::kSuccess;
    }
    return alert.pending() ? WaitResult::kUserCallback : WaitResult::kTimeout;
  }

  static std::pair<WaitResult, size_t> WaitMultiple(
      std::vector<PosixConditionBase*>&& handles, bool wait_all,
      std::chrono::milliseconds timeout) {
    assert_true(!handles.empty());

    // For single handle, just use the normal Wait path
    if (handles.size() == 1) {
      auto result = handles[0]->Wait(timeout);
      return std::make_pair(result, 0);
    }

    // Event-driven WaitAny (candidate fix): register a latched wait-block with
    // each handle so a Signal hands the wake to a waiter under lock — no lost
    // sub-ms Set/Reset transients. A bounded backstop re-poll covers handle
    // types whose Signal path isn't yet hooked (e.g. mutant). WaitAll keeps the
    // poll below.
    if (!wait_all && cvars::threading_blocking_waitany) {
      return WaitAnyBlocking(handles, timeout);
    }
    if (wait_all && cvars::threading_blocking_waitall) {
      return WaitAllBlocking(handles, timeout);
    }

    // For multiple handles, we need to poll since we can't wait on multiple
    // condition variables simultaneously. This is a limitation of the POSIX
    // condition variable API.
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = (timeout == std::chrono::milliseconds::max())
                        ? std::chrono::steady_clock::time_point::max()
                        : start_time + timeout;

    while (true) {
      // Check all handles to see if any/all are signaled
      // Use try_lock to avoid deadlocks from lock ordering issues
      size_t first_signaled = std::numeric_limits<size_t>::max();
      bool condition_met = false;

      // Try to acquire all locks without blocking
      std::vector<std::unique_lock<std::mutex>> locks;
      locks.reserve(handles.size());
      bool all_locked = true;

      for (size_t i = 0; i < handles.size(); ++i) {
        // Try to lock, handling robust mutex EOWNERDEAD case
        auto native_mutex =
            static_cast<pthread_mutex_t*>(handles[i]->mutex_.native_handle());
        int result = pthread_mutex_trylock(native_mutex);

        if (result == 0 || result == EOWNERDEAD) {
          // Successfully acquired lock or recovered from dead owner
          if (result == EOWNERDEAD) {
            // Make mutex consistent after previous owner died
#if !XE_PLATFORM_MAC
            // macOS has no pthread_mutex_consistent; with a plain (non-robust)
            // mutex EOWNERDEAD never occurs, so this path is unreachable there.
            pthread_mutex_consistent(native_mutex);
#endif  // !XE_PLATFORM_MAC
          }
          locks.emplace_back(handles[i]->mutex_, std::adopt_lock);
        } else {
          // Couldn't acquire lock
          all_locked = false;
          break;
        }
      }

      // If we couldn't acquire all locks, release what we have and retry
      if (!all_locked) {
        locks.clear();
        std::this_thread::yield();
        continue;
      }

      // Now we have all locks, check the condition
      if (wait_all) {
        // For wait_all, check if ALL are signaled
        bool all_signaled = true;
        for (size_t i = 0; i < handles.size(); ++i) {
          if (!handles[i]->signaled()) {
            all_signaled = false;
            break;
          }
          if (first_signaled == std::numeric_limits<size_t>::max()) {
            first_signaled = i;
          }
        }
        condition_met = all_signaled;
      } else {
        // For wait_any, check if ANY is signaled
        for (size_t i = 0; i < handles.size(); ++i) {
          if (handles[i]->signaled()) {
            first_signaled = i;
            condition_met = true;
            break;
          }
        }
      }

      if (condition_met) {
        // Execute post_execution for the signaled handle(s)
        if (wait_all) {
          for (size_t i = 0; i < handles.size(); ++i) {
            handles[i]->post_execution();
          }
        } else {
          handles[first_signaled]->post_execution();
        }
        return std::make_pair(WaitResult::kSuccess, first_signaled);
      }

      // Release locks before sleeping
      locks.clear();

      if (alertable_wait_ && alertable_wait_->pending.load()) {
        return std::make_pair<WaitResult, size_t>(WaitResult::kUserCallback, 0);
      }

      // Check timeout
      auto now = std::chrono::steady_clock::now();
      if (now >= end_time) {
        return std::make_pair<WaitResult, size_t>(WaitResult::kTimeout, 0);
      }

      // Sleep for a short time before polling again
      auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(end_time - now);
      auto sleep_time = std::min(remaining, std::chrono::milliseconds(1));
      std::this_thread::sleep_for(sleep_time);
    }
  }

  // Event-driven WaitAny: register a latched WaitBlock with each handle, block
  // on its cv (woken by DeliverToWaitBlocksLocked from a handle's Signal), with
  // a bounded backstop re-poll for safety / un-hooked signal sources.
  static std::pair<WaitResult, size_t> WaitAnyBlocking(
      std::vector<PosixConditionBase*>& handles,
      std::chrono::milliseconds timeout) {
    const size_t n = handles.size();
    constexpr size_t kNone = std::numeric_limits<size_t>::max();
    WaitBlock block;
    AlertArm alert(&block.m, &block.cv);
    size_t fired = kNone;

    // Try-consume-if-already-signaled, else register, per handle (under its
    // mutex_, then block.m — the consistent lock order with Signal).
    auto try_take = [&](size_t i) -> bool {
      std::lock_guard<std::mutex> bl(block.m);
      if (!block.satisfied && handles[i]->signaled()) {
        block.satisfied = true;
        block.fired_index = static_cast<int>(i);
        handles[i]->post_execution();  // consume under handle lock
        return true;
      }
      return false;
    };
    // Atomic arm: lock ALL members (unique, sorted by address to avoid deadlock)
    // before the waiter is observable, so no Set/Reset on any member can slip
    // through a per-handle registration gap. Lock order handle-mutexes(sorted)
    // then block.m matches DeliverToWaitBlocksLocked.
    {
      std::vector<PosixConditionBase*> sorted(handles);
      std::sort(sorted.begin(), sorted.end());
      sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
      std::vector<std::unique_lock<std::mutex>> locks;
      locks.reserve(sorted.size());
      for (auto* h : sorted) {
        locks.emplace_back(h->mutex_);
      }
      for (size_t i = 0; i < n; ++i) {
        if (fired == kNone && try_take(i)) {
          fired = i;
        } else {
          handles[i]->wait_blocks_.push_back({&block, i});
        }
      }
    }  // release all member locks; registered waiter is now armed

    const auto end = (timeout == std::chrono::milliseconds::max())
                         ? std::chrono::steady_clock::time_point::max()
                         : std::chrono::steady_clock::now() + timeout;
    while (fired == kNone && !alert.pending()) {
      const auto now = std::chrono::steady_clock::now();
      auto backstop = std::chrono::milliseconds(4);
      if (end != std::chrono::steady_clock::time_point::max()) {
        if (now >= end) {
          break;  // timed out (checked before sleeping → no overshoot)
        }
        backstop = std::min(
            backstop,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - now));
      }
      {
        std::unique_lock<std::mutex> bl(block.m);
        if (!block.satisfied && !block.pending) {
          block.cv.wait_for(bl, backstop, [&] {
            return block.satisfied || block.pending || alert.pending();
          });
        }
        if (block.satisfied) {
          fired = static_cast<size_t>(block.fired_index);
          break;
        }
        block.pending = false;
      }
      // Re-poll: a member poked the block (PokeWaitBlocksLocked), or the
      // backstop ran out for a persistent signal whose source isn't hooked.
      for (size_t i = 0; i < n && fired == kNone; ++i) {
        std::lock_guard<std::mutex> hl(handles[i]->mutex_);
        if (try_take(i)) {
          fired = i;
        }
      }
    }

    // Unregister from all handles.
    for (size_t i = 0; i < n; ++i) {
      std::lock_guard<std::mutex> hl(handles[i]->mutex_);
      auto& v = handles[i]->wait_blocks_;
      for (auto it = v.begin(); it != v.end();) {
        if (it->first == &block) {
          it = v.erase(it);
        } else {
          ++it;
        }
      }
    }

    // A signaller can satisfy the block (and, for an auto-reset event or a
    // semaphore, consume the signal for it) any time before the unregistration
    // above, including after the wait gave up: a poll with a zero timeout is
    // registered only for an instant. The wake is this wait's, so take it;
    // returning a timeout here would drop a Set nobody else can see. No
    // signaller holds the block any more, so this read is the last word.
    if (fired == kNone) {
      std::lock_guard<std::mutex> bl(block.m);
      if (block.satisfied) {
        fired = static_cast<size_t>(block.fired_index);
      }
    }

    if (fired != kNone) {
      return std::make_pair(WaitResult::kSuccess, fired);
    }
    return std::make_pair(alert.pending() ? WaitResult::kUserCallback
                                          : WaitResult::kTimeout,
                          size_t(0));
  }

  // Event-driven WaitAll: one notify-only WaitBlock registered with every
  // member. The all-signaled check (and the consumption of every member on
  // success) runs under ALL member locks, taken in address order, exactly as
  // the poll did with try-locks; between checks the waiter sleeps on the
  // block's cv until a member's Signal pokes it, or 1 ms passes (the poll's
  // interval, for members whose Signal is not hooked).
  static std::pair<WaitResult, size_t> WaitAllBlocking(
      std::vector<PosixConditionBase*>& handles,
      std::chrono::milliseconds timeout) {
    const size_t n = handles.size();
    WaitBlock block;
    block.notify_only = true;
    AlertArm alert(&block.m, &block.cv);
    std::vector<PosixConditionBase*> sorted(handles);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    // Register with every member (under its lock) before the first check so
    // no Signal between the check and the sleep is missed.
    for (auto* h : sorted) {
      std::lock_guard<std::mutex> hl(h->mutex_);
      h->wait_blocks_.push_back({&block, 0});
    }

    const auto end = (timeout == std::chrono::milliseconds::max())
                         ? std::chrono::steady_clock::time_point::max()
                         : std::chrono::steady_clock::now() + timeout;
    WaitResult result = WaitResult::kTimeout;
    while (true) {
      {
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(sorted.size());
        for (auto* h : sorted) {
          locks.emplace_back(h->mutex_);
        }
        bool all_signaled = true;
        for (size_t i = 0; i < n; ++i) {
          if (!handles[i]->signaled()) {
            all_signaled = false;
            break;
          }
        }
        if (all_signaled) {
          for (size_t i = 0; i < n; ++i) {
            handles[i]->post_execution();
          }
          result = WaitResult::kSuccess;
          break;
        }
      }
      if (alert.pending()) {
        result = WaitResult::kUserCallback;
        break;
      }
      const auto now = std::chrono::steady_clock::now();
      auto backstop = std::chrono::milliseconds(1);
      if (end != std::chrono::steady_clock::time_point::max()) {
        if (now >= end) {
          break;
        }
        backstop = std::min(
            backstop,
            std::chrono::duration_cast<std::chrono::milliseconds>(end - now));
      }
      std::unique_lock<std::mutex> bl(block.m);
      if (!block.pending) {
        block.cv.wait_for(bl, backstop,
                          [&] { return block.pending || alert.pending(); });
      }
      block.pending = false;
    }

    for (auto* h : sorted) {
      std::lock_guard<std::mutex> hl(h->mutex_);
      auto& v = h->wait_blocks_;
      for (auto it = v.begin(); it != v.end();) {
        if (it->first == &block) {
          it = v.erase(it);
        } else {
          ++it;
        }
      }
    }
    return std::make_pair(result, size_t(0));
  }

  [[nodiscard]] virtual void* native_handle() const {
    return const_cast<std::condition_variable&>(cond_).native_handle();
  }

 protected:
  [[nodiscard]] inline virtual bool signaled() const = 0;
  inline virtual void post_execution() = 0;
  // Whether a single Wait() on this object goes through a wait-block. Only an
  // event needs it: its signal can be taken back (Reset, Pulse) before the
  // woken thread runs.
  [[nodiscard]] virtual bool latches_single_waits() const { return false; }

  // Called WITH mutex_ held by a signal source whose state the waiter has to
  // take for itself: a mutant is acquired by the waiting thread and a thread
  // handle is joined by it, neither can be done by the signaling thread. Wakes
  // the registered waiters to re-check now rather than at their backstop.
  void PokeWaitBlocksLocked() {
    for (auto& [b, index] : wait_blocks_) {
      std::lock_guard<std::mutex> bl(b->m);
      b->pending = true;
      b->cv.notify_all();
    }
  }

  // Called by a subclass Signal/Release WITH mutex_ held, after the signal
  // state has been raised. Hands the wake to registered WaitAny waiters,
  // latching satisfaction so a later Reset cannot revoke it. For consumable
  // objects (auto-reset event, semaphore) it consumes via post_execution() —
  // exactly once per available signal. Returns true if the signal was fully
  // consumed (caller should then NOT also notify single Wait() waiters).
  bool DeliverToWaitBlocksLocked() {
    for (auto it = wait_blocks_.begin(); it != wait_blocks_.end();) {
      if (!signaled()) {
        return true;  // consumed by a prior block this pass
      }
      WaitBlock* b = it->first;
      const size_t idx = it->second;
      if (b->notify_only) {
        // WaitAll waiter: poke it and leave the signal and the registration
        // alone (it consumes every member itself once all are signaled).
        std::lock_guard<std::mutex> bl(b->m);
        b->pending = true;
        b->cv.notify_all();
        ++it;
        continue;
      }
      bool consumed = false;
      {
        std::lock_guard<std::mutex> bl(b->m);
        if (b->satisfied) {
          ++it;
          continue;  // already satisfied via another handle
        }
        b->fired_index = static_cast<int>(idx);
        b->satisfied = true;
        b->cv.notify_all();
        post_execution();  // consume (auto-reset: signal_=false; sema: count--)
        consumed = true;
      }
      it = wait_blocks_.erase(it);  // this handle no longer owes this block
      if (consumed && !signaled()) {
        return true;  // signal fully consumed (auto-reset event)
      }
      // manual-reset (still signaled) → satisfy further blocks; semaphore with
      // remaining count → likewise continue.
    }
    return !signaled();
  }

  std::condition_variable cond_;
  std::mutex mutex_;
  // Registered WaitAny wait-blocks: {block, this handle's index in that wait}.
  std::vector<std::pair<WaitBlock*, size_t>> wait_blocks_;
};

// There really is no native POSIX handle for a single wait/signal construct
// pthreads is at a lower level with more handles for such a mechanism.
// This simple wrapper class functions as our handle and uses conditional
// variables for waits and signals.
template <typename T>
class PosixCondition {};

template <>
class PosixCondition<Event> : public PosixConditionBase {
 public:
  PosixCondition(bool manual_reset, bool initial_state)
      : signal_(initial_state), manual_reset_(manual_reset) {}
  ~PosixCondition() override = default;

  bool Signal() override {
    auto lock = std::unique_lock(mutex_);
    signal_ = true;
    if ((cvars::threading_blocking_waitany ||
         cvars::threading_blocking_waitall) &&
        DeliverToWaitBlocksLocked()) {
      // Consumed by a WaitAny waiter (auto-reset) — do not also wake a single
      // Wait() waiter, or the one signal would satisfy two.
      return true;
    }
    cond_.notify_all();
    return true;
  }

  void Reset() {
    auto lock = std::unique_lock(mutex_);
    signal_ = false;
  }

  // Releases the waiters parked right now (every one of a manual-reset event,
  // one of an auto-reset event) and leaves the event non-signaled. Needs the
  // waiters to be registered wait-blocks (threading_blocking_waitany).
  void Pulse() {
    auto lock = std::unique_lock(mutex_);
    signal_ = true;
    DeliverToWaitBlocksLocked();
    signal_ = false;
  }

  EventInfo Query() {
    auto lock = std::unique_lock(mutex_);
    // EVENT_BASIC_INFORMATION: NotificationEvent is 0, SynchronizationEvent 1.
    return {manual_reset_ ? 0u : 1u, signal_ ? 1u : 0u};
  }

 private:
  [[nodiscard]] bool signaled() const override { return signal_; }
  void post_execution() override {
    if (!manual_reset_) {
      signal_ = false;
    }
  }
  [[nodiscard]] bool latches_single_waits() const override { return true; }
  bool signal_;
  const bool manual_reset_;
};

template <>
class PosixCondition<Semaphore> final : public PosixConditionBase {
 public:
  PosixCondition(uint32_t initial_count, uint32_t maximum_count)
      : count_(initial_count), maximum_count_(maximum_count) {}

  bool Signal() override { return Release(1, nullptr); }

  bool Release(uint32_t release_count, int* out_previous_count) {
    auto lock = std::unique_lock(mutex_);
    // Validate that releasing would not exceed the maximum count
    if (count_ + release_count > maximum_count_) {
      return false;
    }
    if (out_previous_count) {
      *out_previous_count = count_;
    }
    count_ += release_count;
    if (cvars::threading_blocking_waitany ||
        cvars::threading_blocking_waitall) {
      DeliverToWaitBlocksLocked();  // hand counts to WaitAny/WaitAll waiters
    }
    cond_.notify_all();
    return true;
  }

 private:
  [[nodiscard]] bool signaled() const override { return count_ > 0; }
  void post_execution() override {
    count_--;
    cond_.notify_all();
  }
  // A release hands its counts to the threads already waiting, in order, as
  // the console kernel does. Counted with a condition variable, a thread that
  // came back to wait took them first: Ninja Gaiden II releases four job
  // tokens to its four workers, the one worker its job mask excludes woke
  // first, skipped, waited again and took all four, so the frame's job never
  // ran and the game froze in its opening (3 of 3 boots).
  // XE_SEMAPHORE_HANDOFF=0 restores the counted wait.
  [[nodiscard]] bool latches_single_waits() const override {
    static const bool s_handoff = [] {
      const char* value = std::getenv("XE_SEMAPHORE_HANDOFF");
      return !(value && value[0] == '0' && !value[1]);
    }();
    return s_handoff;
  }
  uint32_t count_;
  const uint32_t maximum_count_;
};

template <>
class PosixCondition<Mutant> final : public PosixConditionBase {
 public:
  explicit PosixCondition(bool initial_owner) : count_(0) {
    if (initial_owner) {
      count_ = 1;
      owner_ = std::this_thread::get_id();
    }
  }

  bool Signal() override { return Release(); }

  bool Release() {
    if (owner_ == std::this_thread::get_id() && count_ > 0) {
      auto lock = std::unique_lock(mutex_);
      --count_;
      // Free to be acquired by another thread
      if (count_ == 0) {
        PokeWaitBlocksLocked();
        cond_.notify_all();
      }
      return true;
    }
    return false;
  }

  [[nodiscard]] void* native_handle() const override {
    return const_cast<std::mutex&>(mutex_).native_handle();
  }

 private:
  [[nodiscard]] bool signaled() const override {
    return count_ == 0 || owner_ == std::this_thread::get_id();
  }
  void post_execution() override {
    count_++;
    owner_ = std::this_thread::get_id();
  }
  uint32_t count_;
  std::thread::id owner_;
};

template <>
class PosixCondition<Timer> final : public PosixConditionBase {
 public:
  explicit PosixCondition(bool manual_reset)
      : callback_(nullptr), signal_(false), manual_reset_(manual_reset) {}

  ~PosixCondition() override { Cancel(); }

  bool Signal() override {
    std::lock_guard lock(mutex_);
    signal_ = true;
    if ((cvars::threading_blocking_waitany ||
         cvars::threading_blocking_waitall) &&
        DeliverToWaitBlocksLocked()) {
      // An auto-reset timer, consumed by a WaitAny waiter.
      return true;
    }
    cond_.notify_all();
    return true;
  }

  void SetOnce(std::chrono::steady_clock::time_point due_time,
               std::function<void()> opt_callback) {
    Cancel();

    std::lock_guard lock(mutex_);

    callback_ = std::move(opt_callback);
    signal_ = false;
    wait_item_ = QueueTimerOnce(&CompletionRoutine, this, due_time);
  }

  void SetRepeating(std::chrono::steady_clock::time_point due_time,
                    std::chrono::milliseconds period,
                    std::function<void()> opt_callback) {
    Cancel();

    std::lock_guard lock(mutex_);

    callback_ = std::move(opt_callback);
    signal_ = false;
    wait_item_ =
        QueueTimerRecurring(&CompletionRoutine, this, due_time, period);
  }

  void Cancel() const {
    if (auto wait_item = wait_item_.lock()) {
      wait_item->Disarm();
    }
  }

  [[nodiscard]] void* native_handle() const override {
    assert_always();
    return nullptr;
  }

 private:
  static void CompletionRoutine(void* userdata) {
    assert_not_null(userdata);
    auto timer = static_cast<PosixCondition*>(userdata);
    timer->Signal();
    // As the callback may reset the timer, store local.
    std::function<void()> callback;
    {
      std::lock_guard lock(timer->mutex_);
      callback = timer->callback_;
    }
    if (callback) {
      callback();
    }
  }

  [[nodiscard]] bool signaled() const override { return signal_; }
  void post_execution() override {
    if (!manual_reset_) {
      signal_ = false;
    }
  }
  std::weak_ptr<TimerQueueWaitItem> wait_item_;
  std::function<void()> callback_;
  bool signal_;  // Protected by mutex_
  const bool manual_reset_;
};

struct ThreadStartData {
  std::function<void()> start_routine;
  bool create_suspended;
  Thread* thread_obj;
};

template <>
class PosixCondition<Thread> final : public PosixConditionBase {
  enum class State {
    kUninitialized,
    kRunning,
    kSuspended,
    kFinished,
  };

 public:
  PosixCondition()
      : thread_(0),
        signaled_(false),
        exit_code_(0),
        state_(State::kUninitialized),
        suspend_count_(0) {
#if XE_PLATFORM_ANDROID
    android_pre_api_26_name_[0] = '\0';
#endif
  }
  bool Initialize(Thread::CreationParameters params,
                  ThreadStartData* start_data) {
    start_data->create_suspended = params.create_suspended;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) {
      return false;
    }
    if (pthread_attr_setstacksize(&attr, params.stack_size) != 0) {
      pthread_attr_destroy(&attr);
      return false;
    }
    if (params.initial_priority != 0) {
      sched_param sched{};
      sched.sched_priority = params.initial_priority + 1;
      if (pthread_attr_setschedpolicy(&attr, SCHED_FIFO) != 0) {
        pthread_attr_destroy(&attr);
        return false;
      }
      if (pthread_attr_setschedparam(&attr, &sched) != 0) {
        pthread_attr_destroy(&attr);
        return false;
      }
    }
#if XE_PLATFORM_MAC
    // macOS: assign an Apple QoS class at creation so the scheduler can keep the
    // thread on performance cores. Doing this at creation (vs a post-start
    // pthread_set_qos_class_self_np) is what actually sticks, because once the
    // thread is placed in SCHED_FIFO by the priority mapping it is opted out of
    // QoS management (self-tag then returns EPERM). QoS-managed threads skip the
    // SCHED_FIFO mapping in set_priority() below.
    if (params.qos_class != 0) {
      qos_class_t qos = params.qos_class == 2 ? QOS_CLASS_USER_INTERACTIVE
                                              : QOS_CLASS_USER_INITIATED;
      if (pthread_attr_set_qos_class_np(&attr, qos, 0) == 0) {
        qos_managed_ = true;
      }
    }
#endif
    if (pthread_create(&thread_, &attr, ThreadStartRoutine, start_data) != 0) {
      pthread_attr_destroy(&attr);
      return false;
    }
    pthread_attr_destroy(&attr);
    return true;
  }

  /// Constructor for existing thread. This should only happen once called by
  /// Thread::GetCurrentThread() on the main thread
  explicit PosixCondition(pthread_t thread)
      : thread_(thread),
        tid_(static_cast<pid_t>(syscall(SYS_gettid))),
        signaled_(false),
        exit_code_(0),
        state_(State::kRunning),
        suspend_count_(0) {
#if XE_PLATFORM_ANDROID
    android_pre_api_26_name_[0] = '\0';
#endif
  }

  ~PosixCondition() override {
    // FIXME(RodoMa92): This causes random crashes.
    //  The proper way to handle them according to the webs is properly shutdown
    //  instead on relying on killing them using pthread_cancel.
    /*
    if (thread_ && !signaled_) {
#if XE_PLATFORM_ANDROID
      if (pthread_kill(thread_,
                       GetSystemSignal(SignalType::kThreadTerminate)) != 0) {
        assert_always();
      }
#else
      if (pthread_cancel(thread_) != 0) {
        assert_always();
      }
#endif
      if (pthread_join(thread_, nullptr) != 0) {
        assert_always();
      }
    }
    */
  }

  bool Signal() override { return true; }

  std::string name() const {
    WaitStarted();
    auto result = std::array<char, 17>{'\0'};
    SuspendDeferralScope suspend_deferral;
    std::unique_lock lock(state_mutex_);
    if (state_ != State::kUninitialized && state_ != State::kFinished) {
#if XE_PLATFORM_ANDROID
      // pthread_getname_np was added in API 26 - below that, store the name in
      // this object, which may be only modified through Xenia threading, but
      // should be enough in most cases.
      if (android_pthread_getname_np_) {
        if (android_pthread_getname_np_(thread_, result.data(),
                                        result.size() - 1) != 0) {
          assert_always();
        }
      } else {
        std::lock_guard<std::mutex> lock(android_pre_api_26_name_mutex_);
        std::strcpy(result.data(), android_pre_api_26_name_);
      }
#else
      if (pthread_getname_np(thread_, result.data(), result.size() - 1) != 0) {
        assert_always();
      }
#endif
    }
    return std::string(result.data());
  }

  void set_name(const std::string& name) const {
    WaitStarted();
    SuspendDeferralScope suspend_deferral;
    std::unique_lock<std::mutex> lock(state_mutex_);
    if (state_ != State::kUninitialized && state_ != State::kFinished) {
#if XE_PLATFORM_MAC
      // macOS pthread_setname_np only names the calling thread; other threads
      // cannot be renamed (name() still reads any thread via
      // pthread_getname_np).
      if (pthread_equal(pthread_self(), thread_)) {
        pthread_setname_np(std::string(name).c_str());
      }
#else
      pthread_setname_np(thread_, std::string(name).c_str());
#if XE_PLATFORM_ANDROID
      SetAndroidPreApi26Name(name);
#endif
#endif  // XE_PLATFORM_MAC
    }
  }

#if XE_PLATFORM_ANDROID
  void SetAndroidPreApi26Name(const std::string_view name) {
    if (android_pthread_getname_np_) {
      return;
    }
    std::lock_guard<std::mutex> lock(android_pre_api_26_name_mutex_);
    xe::string_util::copy_truncating(android_pre_api_26_name_, name,
                                     xe::countof(android_pre_api_26_name_));
  }
#endif

  uint32_t system_id() const {
#if XE_PLATFORM_MAC
    // pthread_t is an opaque pointer on macOS; use the kernel thread id.
    uint64_t tid = 0;
    pthread_threadid_np(thread_, &tid);
    return static_cast<uint32_t>(tid);
#else
    return static_cast<uint32_t>(thread_);
#endif  // XE_PLATFORM_MAC
  }

  uint64_t affinity_mask() const {
    WaitStarted();
#if XE_PLATFORM_MAC
    // macOS exposes no thread-affinity query API; report all cores available.
    return ~uint64_t(0);
#else
    cpu_set_t cpu_set;
#if XE_PLATFORM_ANDROID
    if (sched_getaffinity(pthread_gettid_np(thread_), sizeof(cpu_set_t),
                          &cpu_set) != 0) {
      assert_always();
    }
#else
    if (pthread_getaffinity_np(thread_, sizeof(cpu_set_t), &cpu_set) != 0) {
      assert_always();
    }
#endif
    uint64_t result = 0;
    auto cpu_count = std::min(CPU_SETSIZE, 64);
    for (auto i = 0u; i < cpu_count; i++) {
      // CPU_ISSET returns int, so `set << i` was an int shift: undefined from
      // bit 31 and wrapping above it.
      if (CPU_ISSET(i, &cpu_set)) {
        result |= uint64_t(1) << i;
      }
    }
    return result;
#endif  // XE_PLATFORM_MAC
  }

  void set_affinity_mask(uint64_t mask) const {
    WaitStarted();
#if XE_PLATFORM_MAC
    // macOS has no hard thread-affinity API (only advisory affinity hints);
    // treat as a best-effort no-op.
    (void)mask;
#else
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (auto i = 0u; i < 64; i++) {
      if (mask & (uint64_t(1) << i)) {
        CPU_SET(i, &cpu_set);
      }
    }
#if XE_PLATFORM_ANDROID
    if (sched_setaffinity(pthread_gettid_np(thread_), sizeof(cpu_set_t),
                          &cpu_set) != 0) {
      assert_always();
    }
#else
    if (pthread_setaffinity_np(thread_, sizeof(cpu_set_t), &cpu_set) != 0) {
      assert_always();
    }
#endif
#endif  // XE_PLATFORM_MAC
  }

  int priority() const {
    WaitStarted();
    if (!fifo_failed_) {
      int policy;
      sched_param param{};
      int ret = pthread_getschedparam(thread_, &policy, &param);
      if (ret != 0) {
        return -1;
      }
      return param.sched_priority;
    }
    // When using nice values, map back to the SCHED_FIFO range (1-32)
    // so callers see a consistent priority space.
    int nice_val = getpriority(PRIO_PROCESS, tid_);
    // nice -19..19 → fifo 32..1
    return 16 - nice_val;
  }

  void set_priority(int new_priority) const {
#if XE_PLATFORM_MAC
    // A thread given an Apple QoS class at creation must not be moved into
    // SCHED_FIFO: an explicit scheduling policy opts it out of QoS management on
    // Darwin, undoing the P-core hint. Let QoS govern its placement.
    if (qos_managed_) {
      return;
    }
#endif
    WaitStarted();
    if (!fifo_failed_) {
      // Try real-time SCHED_FIFO for best priority control.
      sched_param param{};
      param.sched_priority = new_priority;
      int res = pthread_setschedparam(thread_, SCHED_FIFO, &param);
      if (res == 0) {
        return;
      }
      if (res == EPERM) {
        fifo_failed_ = true;
      } else {
        XELOGW("Unexpected error {} while setting SCHED_FIFO priority", res);
        fifo_failed_ = true;
      }
    }
    // Fall back to nice values under SCHED_OTHER.
    // Map SCHED_FIFO range (1-32) to nice range (19 to -19).
    // Center: fifo 16 → nice 0.
    int nice_val = 16 - new_priority;
    // Clamp to valid nice range.
    if (nice_val < -20) {
      nice_val = -20;
    }
    if (nice_val > 19) {
      nice_val = 19;
    }
    if (tid_ > 0) {
      setpriority(PRIO_PROCESS, tid_, nice_val);
    }
  }

  void QueueUserCallback(std::function<void()> callback) {
    WaitStarted();
    alert_.Queue(std::move(callback));
  }

  ThreadAlert& alert() { return alert_; }

  bool Resume(uint32_t* out_previous_suspend_count = nullptr) {
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = 0;
    }
    WaitStarted();
    SuspendDeferralScope suspend_deferral;
    std::unique_lock lock(state_mutex_);
    if (ThreadLifecycleLogEnabled()) {
      XELOGI("ThreadLifecycle host-resume tid={:X} by={:X} count={} state={}",
             system_id(), LifecycleSelfId(), suspend_count_.load(),
             StateName(state_));
    }
    // Check if thread has any suspend count (Windows allows resume even if
    // running)
    if (suspend_count_ == 0) {
      return false;
    }
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = suspend_count_;
    }
    --suspend_count_;
    // If suspend count reaches 0, transition to running and wake the thread
    if (suspend_count_ == 0 && state_ == State::kSuspended) {
      state_ = State::kRunning;
      // Post to the semaphore to wake the thread from WaitSuspended.
      // Posting is async-signal-safe, so this is safe even if called
      // from unusual contexts.
      suspend_sem_.Post();
    }
    state_signal_.notify_all();
    return true;
  }

  bool Suspend(uint32_t* out_previous_suspend_count = nullptr) {
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = 0;
    }
    WaitStarted();

    // Check if we're trying to suspend ourselves
    bool is_current_thread = pthread_self() == thread_;
    bool already_suspended = false;

    {
      SuspendDeferralScope suspend_deferral;
      std::unique_lock lock(state_mutex_);
      if (out_previous_suspend_count) {
        *out_previous_suspend_count = suspend_count_;
      }
      already_suspended = (state_ == State::kSuspended);
      state_ = State::kSuspended;
      ++suspend_count_;
      if (ThreadLifecycleLogEnabled()) {
        XELOGI(
            "ThreadLifecycle host-suspend tid={:X} by={:X} count={} "
            "already={} self={} start_wait={}",
            system_id(), LifecycleSelfId(), suspend_count_.load(),
            already_suspended, is_current_thread, in_start_wait_.load());
      }
    }

    // If already suspended, just increment the count — don't send another
    // signal. A second pthread_kill while the thread is parked on the semaphore would
    // nest signal handlers and create multiple outstanding waits, but
    // Resume only posts once when count reaches 0.
    if (already_suspended) {
      return true;
    }

    if (is_current_thread) {
      // Self-suspension: Instead of sending a signal, directly call
      // WaitSuspended. This avoids the signal handler complexity for the
      // self-suspend case.
      WaitSuspended();
      return true;
    }

    int result =
        pthread_kill(thread_, GetSystemSignal(SignalType::kThreadSuspend));
    return result == 0;
  }

  void Terminate(int exit_code) {
    bool is_current_thread = pthread_self() == thread_;
#if 0  // [thread-diag] Re-enable for self/external termination probes.
#if XE_PLATFORM_MAC
    XELOGI(
        "[thread-diag] PosixCondition::Terminate entry caller_tid={} "
        "target_tid={} pthread={} current={} exit_code={}",
        current_thread_system_id(), system_id(),
        reinterpret_cast<void*>(thread_), is_current_thread, exit_code);
#endif
#endif
    {
      SuspendDeferralScope suspend_deferral;
      std::unique_lock lock(state_mutex_);
#if 0  // [thread-diag] Re-enable for self/external termination probes.
#if XE_PLATFORM_MAC
      XELOGI(
          "[thread-diag] PosixCondition::Terminate state caller_tid={} "
          "target_tid={} state={} current={} exit_code={}",
          current_thread_system_id(), system_id(), static_cast<int>(state_),
          is_current_thread, exit_code);
#endif
#endif
      if (state_ == State::kFinished) {
        if (is_current_thread) {
          // Another thread's Terminate() raced ours and already marked us
          // finished. Its pthread_cancel is not guaranteed to arrive, and
          // this loop burned a core waiting for it. Leave under our own
          // power instead.
          XELOGW("PosixThread::Terminate: thread was already finished");
          lock.unlock();
          pthread_exit(reinterpret_cast<void*>(exit_code));
        }
        return;
      }
      state_ = State::kFinished;
    }

    {
      std::lock_guard lock(mutex_);

      exit_code_ = exit_code;
      signaled_ = true;
      PokeWaitBlocksLocked();
      cond_.notify_all();
    }
    if (is_current_thread) {
      pthread_exit(reinterpret_cast<void*>(exit_code));
    }
    // The thread dies inside whatever wait it armed, without unwinding.
    alert_.Close();
#ifdef XE_PLATFORM_ANDROID
    if (pthread_kill(thread_, GetSystemSignal(SignalType::kThreadTerminate)) !=
        0) {
      assert_always();
    }
#else
    if (pthread_cancel(thread_) != 0) {
      assert_always();
    }
#endif
  }

  void WaitStarted() const {
    SuspendDeferralScope suspend_deferral;
    std::unique_lock lock(state_mutex_);
    state_signal_.wait(lock,
                       [this] { return state_ != State::kUninitialized; });
  }

  /// Set state to suspended and wait until it is reset by another thread.
  /// Parks on SuspendSemaphore, which is async-signal-safe, allowing this to be called
  /// from a signal handler (e.g., the SIGRTMIN suspend signal handler)
  /// without risking deadlock or heap corruption from non-reentrant
  /// mutex/condvar operations.
  void WaitSuspended() {
    // The suspend signal does not queue, and a Resume() whose post lands
    // before the signal is delivered leaves a token behind. Re-check the
    // count rather than trusting one token, or that stale post releases the
    // next suspend immediately.
    parked_.fetch_add(1, std::memory_order_relaxed);
    while (suspend_count_.load(std::memory_order_acquire) > 0) {
      if (!suspend_sem_.Wait()) {
        // The semaphore is gone (the thread has been joined). Do not spin.
        break;
      }
    }
    parked_.fetch_sub(1, std::memory_order_relaxed);
  }

  // Never blocks: the dump that calls this may run while a thread holds
  // state_mutex_.
  std::string DebugState() const {
    std::unique_lock lock(state_mutex_, std::try_to_lock);
    return fmt::format(
        "host[{} count={} start_wait={} body={} parked={}]",
        lock.owns_lock() ? StateName(state_) : "locked",
        suspend_count_.load(std::memory_order_relaxed),
        in_start_wait_.load(std::memory_order_relaxed),
        body_started_.load(std::memory_order_relaxed),
        parked_.load(std::memory_order_relaxed));
  }

  static const char* StateName(State state) {
    switch (state) {
      case State::kUninitialized:
        return "uninit";
      case State::kRunning:
        return "running";
      case State::kSuspended:
        return "suspended";
      case State::kFinished:
        return "finished";
    }
    return "?";
  }

  void* native_handle() const override {
    return reinterpret_cast<void*>(thread_);
  }

 private:
  static void* ThreadStartRoutine(void* parameter);
  bool signaled() const override { return signaled_; }
  void post_execution() override {
    // Every wait that a thread handle satisfies used to land here, so a
    // handle waited on twice joined the same thread twice, which is
    // undefined once the id can be reused, and tore down the suspend
    // semaphore each time rather than once after the join.
    bool expected = false;
    if (thread_ && joined_.compare_exchange_strong(expected, true)) {
      pthread_join(thread_, nullptr);
      suspend_sem_.Destroy();
    }
  }
  pthread_t thread_;
  pid_t tid_ = 0;                     // Kernel TID for setpriority() fallback
  mutable bool fifo_failed_ = false;  // True after SCHED_FIFO was rejected
  bool qos_managed_ = false;  // macOS: Apple QoS assigned at creation; skip
                              // SCHED_FIFO so QoS governs core placement.
  bool signaled_;
  int exit_code_;
  State state_;  // Protected by state_mutex_
  // Written under state_mutex_, but read from the suspend signal handler.
  std::atomic<uint32_t> suspend_count_;
  std::atomic<bool> joined_{false};  // Prevents a second pthread_join
  // Hang diagnostics (DebugState): parked before the start routine, start
  // routine entered, parked in WaitSuspended.
  std::atomic<bool> in_start_wait_{false};
  std::atomic<bool> body_started_{false};
  std::atomic<int32_t> parked_{0};
  SuspendSemaphore suspend_sem_;  // Async-signal-safe suspend/resume
  mutable std::mutex state_mutex_;
  mutable std::condition_variable state_signal_;
  ThreadAlert alert_;
#if XE_PLATFORM_ANDROID
  // Name accessible via name() on Android before API 26 which added
  // pthread_getname_np.
  mutable std::mutex android_pre_api_26_name_mutex_;
  char android_pre_api_26_name_[16];
#endif
};

class PosixWaitHandle {
 public:
  virtual ~PosixWaitHandle();
  virtual PosixConditionBase& condition() = 0;
};

// Out-of-line destructor to ensure proper RTTI/vtable emission
PosixWaitHandle::~PosixWaitHandle() = default;

// This wraps a condition object as our handle because posix has no single
// native handle for higher level concurrency constructs such as semaphores
template <typename T>
class PosixConditionHandle : public T, public PosixWaitHandle {
 public:
  PosixConditionHandle() = default;
  explicit PosixConditionHandle(bool);
  explicit PosixConditionHandle(pthread_t thread);
  PosixConditionHandle(bool manual_reset, bool initial_state);
  PosixConditionHandle(uint32_t initial_count, uint32_t maximum_count);
  ~PosixConditionHandle() override = default;

  PosixCondition<T>& condition() override { return handle_; }
  [[nodiscard]] void* native_handle() const override {
    return handle_.native_handle();
  }

 protected:
  PosixCondition<T> handle_;
  friend PosixCondition<T>;
};

template <>
PosixConditionHandle<Semaphore>::PosixConditionHandle(uint32_t initial_count,
                                                      uint32_t maximum_count)
    : handle_(initial_count, maximum_count) {}

template <>
PosixConditionHandle<Mutant>::PosixConditionHandle(bool initial_owner)
    : handle_(initial_owner) {}

template <>
PosixConditionHandle<Timer>::PosixConditionHandle(bool manual_reset)
    : handle_(manual_reset) {}

template <>
PosixConditionHandle<Event>::PosixConditionHandle(bool manual_reset,
                                                  bool initial_state)
    : handle_(manual_reset, initial_state) {}

template <>
PosixConditionHandle<Thread>::PosixConditionHandle(pthread_t thread)
    : handle_(thread) {}

WaitResult Wait(WaitHandle* wait_handle, bool is_alertable,
                std::chrono::milliseconds timeout) {
  auto posix_wait_handle = dynamic_cast<PosixWaitHandle*>(wait_handle);
  if (posix_wait_handle == nullptr) {
    return WaitResult::kFailed;
  }
  ThreadAlert* alert = is_alertable ? CurrentThreadAlert() : nullptr;
  alertable_wait_ = alert;
  auto result = posix_wait_handle->condition().Wait(timeout);
  alertable_wait_ = nullptr;
  if (result == WaitResult::kUserCallback) {
    alert->Run();
  }
  return result;
}

WaitResult SignalAndWait(WaitHandle* wait_handle_to_signal,
                         WaitHandle* wait_handle_to_wait_on, bool is_alertable,
                         std::chrono::milliseconds timeout) {
  auto result = WaitResult::kFailed;
  auto posix_wait_handle_to_signal =
      dynamic_cast<PosixWaitHandle*>(wait_handle_to_signal);
  auto posix_wait_handle_to_wait_on =
      dynamic_cast<PosixWaitHandle*>(wait_handle_to_wait_on);
  if (posix_wait_handle_to_signal == nullptr ||
      posix_wait_handle_to_wait_on == nullptr) {
    return WaitResult::kFailed;
  }
  ThreadAlert* alert = is_alertable ? CurrentThreadAlert() : nullptr;
  alertable_wait_ = alert;
  if (posix_wait_handle_to_signal->condition().Signal()) {
    result = posix_wait_handle_to_wait_on->condition().Wait(timeout);
  }
  alertable_wait_ = nullptr;
  if (result == WaitResult::kUserCallback) {
    alert->Run();
  }
  return result;
}

std::pair<WaitResult, size_t> WaitMultiple(WaitHandle* wait_handles[],
                                           size_t wait_handle_count,
                                           bool wait_all, bool is_alertable,
                                           std::chrono::milliseconds timeout) {
  std::vector<PosixConditionBase*> conditions;
  conditions.reserve(wait_handle_count);
  for (size_t i = 0u; i < wait_handle_count; ++i) {
    auto handle = dynamic_cast<PosixWaitHandle*>(wait_handles[i]);
    if (handle == nullptr) {
      return std::make_pair(WaitResult::kFailed, 0);
    }
    conditions.push_back(&handle->condition());
  }
  ThreadAlert* alert = is_alertable ? CurrentThreadAlert() : nullptr;
  alertable_wait_ = alert;
  auto result = PosixConditionBase::WaitMultiple(std::move(conditions),
                                                 wait_all, timeout);
  alertable_wait_ = nullptr;
  if (result.first == WaitResult::kUserCallback) {
    alert->Run();
  }
  return result;
}

class PosixEvent final : public PosixConditionHandle<Event> {
 public:
  PosixEvent(bool manual_reset, bool initial_state)
      : PosixConditionHandle(manual_reset, initial_state) {}
  ~PosixEvent() override = default;
  void Set() override { handle_.Signal(); }
  void Reset() override { handle_.Reset(); }
  EventInfo Query() override { return handle_.Query(); }
  void Pulse() override {
    using namespace std::chrono_literals;
    if (cvars::threading_blocking_waitany) {
      handle_.Pulse();
      return;
    }
    handle_.Signal();
    MaybeYield();
    Sleep(10us);
    handle_.Reset();
  }
};

std::unique_ptr<Event> Event::CreateManualResetEvent(bool initial_state) {
  return std::make_unique<PosixEvent>(true, initial_state);
}

std::unique_ptr<Event> Event::CreateAutoResetEvent(bool initial_state) {
  return std::make_unique<PosixEvent>(false, initial_state);
}

class PosixSemaphore final : public PosixConditionHandle<Semaphore> {
 public:
  PosixSemaphore(int initial_count, int maximum_count)
      : PosixConditionHandle(static_cast<uint32_t>(initial_count),
                             static_cast<uint32_t>(maximum_count)) {}
  ~PosixSemaphore() override = default;
  bool Release(int release_count, int* out_previous_count) override {
    if (release_count < 1) {
      return false;
    }
    return handle_.Release(static_cast<uint32_t>(release_count),
                           out_previous_count);
  }
};

std::unique_ptr<Semaphore> Semaphore::Create(int initial_count,
                                             int maximum_count) {
  if (initial_count < 0 || initial_count > maximum_count ||
      maximum_count <= 0) {
    return nullptr;
  }
  return std::make_unique<PosixSemaphore>(initial_count, maximum_count);
}

class PosixMutant final : public PosixConditionHandle<Mutant> {
 public:
  explicit PosixMutant(bool initial_owner)
      : PosixConditionHandle(initial_owner) {}
  ~PosixMutant() override = default;
  bool Release() override { return handle_.Release(); }
};

std::unique_ptr<Mutant> Mutant::Create(bool initial_owner) {
  return std::make_unique<PosixMutant>(initial_owner);
}

class PosixTimer final : public PosixConditionHandle<Timer> {
  using WClock_ = WClock_;
  using GClock_ = GClock_;

 public:
  explicit PosixTimer(bool manual_reset) : PosixConditionHandle(manual_reset) {}
  ~PosixTimer() override = default;

  bool SetOnceAfter(xe::chrono::hundrednanoseconds rel_time,
                    std::function<void()> opt_callback = nullptr) override {
    return SetOnceAt(GClock_::now() + rel_time, std::move(opt_callback));
  }
  bool SetOnceAt(WClock_::time_point due_time,
                 std::function<void()> opt_callback = nullptr) override {
    return SetOnceAt(date::clock_cast<GClock_>(due_time),
                     std::move(opt_callback));
  };
  bool SetOnceAt(GClock_::time_point due_time,
                 std::function<void()> opt_callback = nullptr) override {
    handle_.SetOnce(due_time, std::move(opt_callback));
    return true;
  }

  bool SetRepeatingAfter(
      xe::chrono::hundrednanoseconds rel_time, std::chrono::milliseconds period,
      std::function<void()> opt_callback = nullptr) override {
    return SetRepeatingAt(GClock_::now() + rel_time, period,
                          std::move(opt_callback));
  }
  bool SetRepeatingAt(WClock_::time_point due_time,
                      std::chrono::milliseconds period,
                      std::function<void()> opt_callback = nullptr) override {
    return SetRepeatingAt(date::clock_cast<GClock_>(due_time), period,
                          std::move(opt_callback));
  }
  bool SetRepeatingAt(GClock_::time_point due_time,
                      std::chrono::milliseconds period,
                      std::function<void()> opt_callback = nullptr) override {
    handle_.SetRepeating(due_time, period, std::move(opt_callback));
    return true;
  }
  bool Cancel() override {
    handle_.Cancel();
    return true;
  }
};

std::unique_ptr<Timer> Timer::CreateManualResetTimer() {
  return std::make_unique<PosixTimer>(true);
}

std::unique_ptr<Timer> Timer::CreateSynchronizationTimer() {
  return std::make_unique<PosixTimer>(false);
}

class PosixThread final : public PosixConditionHandle<Thread> {
 public:
  PosixThread() = default;
  explicit PosixThread(pthread_t thread) : PosixConditionHandle(thread) {}
  ~PosixThread() override = default;

  bool Initialize(CreationParameters params,
                  std::function<void()> start_routine) {
    auto start_data =
        new ThreadStartData({std::move(start_routine), false, this});
    return handle_.Initialize(params, start_data);
  }

  void set_name(std::string name) override {
    // Covers name_mutex_ as well as the state lock inside handle_.set_name().
    SuspendDeferralScope suspend_deferral;
    handle_.WaitStarted();
#if XE_PLATFORM_MAC
    // Serialize concurrent renames of the same thread object. The base
    // Thread::set_name() assigns the name_ std::string without a lock, so a
    // startup lambda naming itself while another thread runs XThread::set_name
    // can corrupt that string. handle_.set_name() is independently guarded by
    // state_mutex_, so this must be a distinct mutex to avoid a self-deadlock.
    std::lock_guard<std::mutex> name_lock(name_mutex_);
#endif
    Thread::set_name(name);
    if (name.length() > 15) {
      name = name.substr(0, 15);
    }
    handle_.set_name(name);
  }

  uint32_t system_id() const override { return handle_.system_id(); }

  uint64_t affinity_mask() override { return handle_.affinity_mask(); }
  void set_affinity_mask(uint64_t mask) override {
    handle_.set_affinity_mask(mask);
  }

  int priority() override { return handle_.priority(); }
  void set_priority(int new_priority) override {
    handle_.set_priority(new_priority);
  }

  void QueueUserCallback(std::function<void()> callback) override {
    handle_.QueueUserCallback(std::move(callback));
  }

  bool Resume(uint32_t* out_previous_suspend_count) override {
    return handle_.Resume(out_previous_suspend_count);
  }

  bool Suspend(uint32_t* out_previous_suspend_count) override {
    return handle_.Suspend(out_previous_suspend_count);
  }

  void Terminate(int exit_code) override { handle_.Terminate(exit_code); }

  void WaitSuspended() { handle_.WaitSuspended(); }

  std::string DebugState() const override { return handle_.DebugState(); }

#if XE_PLATFORM_MAC
 private:
  // Guards set_name() against concurrent renames of this thread object.
  std::mutex name_mutex_;
#endif
};

thread_local PosixThread* current_thread_ = nullptr;

static ThreadAlert* CurrentThreadAlert() {
  return current_thread_ ? &current_thread_->condition().alert() : nullptr;
}

// SuspendDeferralScope state. Only the owning thread and its own signal
// handler touch these; the handler reads the depth and sets the flag.
thread_local volatile sig_atomic_t suspend_deferral_depth_ = 0;
thread_local volatile sig_atomic_t suspend_deferred_ = 0;

SuspendDeferralScope::SuspendDeferralScope() { BeginSuspendDeferral(); }

SuspendDeferralScope::~SuspendDeferralScope() { EndSuspendDeferral(); }

void BeginSuspendDeferral() {
  suspend_deferral_depth_ = suspend_deferral_depth_ + 1;
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

void EndSuspendDeferral() {
  std::atomic_signal_fence(std::memory_order_seq_cst);
  suspend_deferral_depth_ = suspend_deferral_depth_ - 1;
  std::atomic_signal_fence(std::memory_order_seq_cst);
  // A request that arrived after the decrement saw depth 0 and parked in the
  // handler; one before it set the flag, read here.
  if (suspend_deferral_depth_ == 0 && suspend_deferred_) {
    suspend_deferred_ = 0;
    if (current_thread_) {
      if (ThreadLifecycleLogEnabled()) {
        XELOGI("ThreadLifecycle deferred suspend served tid={:X}",
               LifecycleSelfId());
      }
      current_thread_->WaitSuspended();
    }
  }
}

void* PosixCondition<Thread>::ThreadStartRoutine(void* parameter) {
#if !XE_PLATFORM_ANDROID
#if XE_PLATFORM_MAC
  int cancel_type = PTHREAD_CANCEL_ASYNCHRONOUS;
  if (cvars::macos_thread_cancel_mode == "deferred") {
    cancel_type = PTHREAD_CANCEL_DEFERRED;
  } else if (cvars::macos_thread_cancel_mode != "async") {
    XELOGW("Unknown macos_thread_cancel_mode='{}'; using async",
           cvars::macos_thread_cancel_mode);
  }
#if 0  // [thread-diag] Re-enable for cancel-mode validation probes.
  XELOGI("[thread-diag] pthread cancel mode {}",
         cancel_type == PTHREAD_CANCEL_DEFERRED ? "deferred" : "async");
#endif
#else
  int cancel_type = PTHREAD_CANCEL_ASYNCHRONOUS;
#endif
  if (pthread_setcanceltype(cancel_type, nullptr) != 0) {
    assert_always();
  }
#endif
  threading::set_name("");

  auto start_data = static_cast<ThreadStartData*>(parameter);
  assert_not_null(start_data);
  assert_not_null(start_data->thread_obj);

  auto thread = dynamic_cast<PosixThread*>(start_data->thread_obj);
  auto start_routine = std::move(start_data->start_routine);
  auto create_suspended = start_data->create_suspended;
  delete start_data;

  current_thread_ = thread;
  thread->handle_.tid_ = static_cast<pid_t>(syscall(SYS_gettid));
  {
    // The suspend count must be published with the state: Resume() waits only
    // for the state to leave kUninitialized, and a Resume() that ran between
    // the two found a count of 0, did nothing, and the count set afterwards
    // parked the thread forever. Every guest thread is created suspended and
    // resumed by its creator, so this hit at random (Gears of War 3 stuck on
    // "Loading profile": the loader thread never started).
    SuspendDeferralScope suspend_deferral;
    std::unique_lock lock(thread->handle_.state_mutex_);
    thread->handle_.state_ =
        create_suspended ? State::kSuspended : State::kRunning;
    if (create_suspended) {
      thread->handle_.suspend_count_ = 1;
    }
    thread->handle_.state_signal_.notify_all();
  }

  if (create_suspended) {
    thread->handle_.in_start_wait_ = true;
    SuspendDeferralScope suspend_deferral;
    std::unique_lock lock(thread->handle_.state_mutex_);
    thread->handle_.state_signal_.wait(
        lock, [thread] { return thread->handle_.suspend_count_ == 0; });
    thread->handle_.in_start_wait_ = false;
  }

  thread->handle_.body_started_ = true;
  if (ThreadLifecycleLogEnabled()) {
    XELOGI("ThreadLifecycle host-start tid={:X} create_suspended={}",
           LifecycleSelfId(), create_suspended);
  }
  start_routine();

  {
    SuspendDeferralScope suspend_deferral;
    std::unique_lock lock(thread->handle_.state_mutex_);
    thread->handle_.state_ = State::kFinished;
  }

  {
    std::unique_lock lock(thread->handle_.mutex_);
    thread->handle_.exit_code_ = 0;
    thread->handle_.signaled_ = true;
    thread->handle_.PokeWaitBlocksLocked();
    thread->handle_.cond_.notify_all();
  }

  current_thread_ = nullptr;
  return nullptr;
}

std::unique_ptr<Thread> Thread::Create(CreationParameters params,
                                       std::function<void()> start_routine) {
  install_signal_handler(SignalType::kThreadSuspend);
#if XE_PLATFORM_ANDROID
  install_signal_handler(SignalType::kThreadTerminate);
#endif
  auto thread = std::make_unique<PosixThread>();
  if (!thread->Initialize(params, std::move(start_routine))) {
    return nullptr;
  }
  assert_not_null(thread);
  return thread;
}

Thread* Thread::GetCurrentThread() {
  if (current_thread_) {
    return current_thread_;
  }

  // Should take this route only for threads not created by Thread::Create.
  // The only thread not created by Thread::Create should be the main thread.
  pthread_t handle = pthread_self();

  current_thread_ = new PosixThread(handle);
  // TODO(bwrsandman): Disabling deleting thread_local current thread to prevent
  //                   assert in destructor. Since this is thread local, the
  //                   "memory leaking" is controlled.
  // atexit([] { delete current_thread_; });

  return current_thread_;
}

void Thread::Exit(int exit_code) {
  if (current_thread_) {
    current_thread_->Terminate(exit_code);
  } else {
    // Should only happen with the main thread
    pthread_exit(reinterpret_cast<void*>(exit_code));
  }
  // Function must not return
  assert_always();
}

void set_name(const std::string_view name) {
#if XE_PLATFORM_MAC
  // macOS names only the calling thread and takes just the name.
  pthread_setname_np(std::string(name).c_str());
#else
  pthread_setname_np(pthread_self(), std::string(name).c_str());
#if XE_PLATFORM_ANDROID
  if (!android_pthread_getname_np_ && current_thread_) {
    current_thread_->condition().SetAndroidPreApi26Name(name);
  }
#endif
#endif  // XE_PLATFORM_MAC
}

static void signal_handler(int signal, siginfo_t* info, void* context) {
  switch (GetSystemSignalType(signal)) {
    case SignalType::kThreadSuspend: {
      if (!current_thread_) {
        // current_thread_ is NULL - this can happen if the signal arrives
        // before the thread has initialized or after it has exited
        return;
      }
      if (suspend_deferral_depth_ > 0 && suspend_deferral_enabled_) {
        // Holding a lock a resumer may need: park when the scope ends.
        suspend_deferred_ = 1;
        return;
      }
      current_thread_->WaitSuspended();
    } break;
#if XE_PLATFORM_ANDROID
    case SignalType::kThreadTerminate: {
      pthread_exit(reinterpret_cast<void*>(-1));
    } break;
#endif
    default:
      assert_always();
  }
}

}  // namespace threading
}  // namespace xe
