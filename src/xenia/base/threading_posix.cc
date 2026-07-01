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
#include <signal.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
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
    threading_blocking_waitany, false,
    "Candidate fix (default off = current 1ms poll): make WaitMultiple(WaitAny) "
    "event-driven instead of polling. Waiters register a latched wait-block with "
    "each handle; a handle's Signal hands the wake to one registered waiter "
    "under lock (auto-reset consumed exactly once), so a sub-millisecond "
    "Set-then-Reset transient is no longer lost (the DOA5 worker-pool deadlock). "
    "WaitAll still uses the poll. Regression-check broadly before shipping.",
    "Threading");

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
  kThreadUserCallback,
#if XE_PLATFORM_ANDROID
  // pthread_cancel is not available on Android, using a signal handler for
  // simplified PTHREAD_CANCEL_ASYNCHRONOUS-like behavior - not disabling
  // cancellation currently, so should be enough.
  kThreadTerminate,
#endif
  k_Count
};

#if XE_PLATFORM_MAC
// macOS has no POSIX realtime signals; map the (two) signal types onto the
// user signals SIGUSR1/SIGUSR2, which are consecutive.
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

void install_signal_handler(SignalType type) {
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

// uint64_t ticks() { return mach_absolute_time(); }

uint32_t current_thread_system_id() {
  return static_cast<uint32_t>(syscall(SYS_gettid));
}

void MaybeYield() {
  sched_yield();
  __sync_synchronize();
}

void SyncMemory() { __sync_synchronize(); }

void Sleep(std::chrono::microseconds duration) {
  timespec rqtp = DurationToTimeSpec(duration);
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

void NanoSleep(int64_t duration) { Sleep(std::chrono::nanoseconds(duration)); }

// TODO(bwrsandman) Implement by allowing alert interrupts from IO operations
thread_local bool alertable_state_ = false;
SleepResult AlertableSleep(std::chrono::microseconds duration) {
  alertable_state_ = true;
  Sleep(duration);
  alertable_state_ = false;
  return SleepResult::kSuccess;
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
    // Route single waits through the same latched wait-block path as WaitAny so
    // a Signal is delivered (and auto-reset consumed) exactly once across single
    // AND multi waiters — no transient loss, no steal. (Gated; off = old path.)
    if (cvars::threading_blocking_waitany) {
      std::vector<PosixConditionBase*> one{this};
      return WaitAnyBlocking(one, timeout).first;
    }

    bool executed;
    auto predicate = [this] { return this->signaled(); };

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

    if (predicate()) {
      executed = true;
    } else {
      if (timeout == std::chrono::milliseconds::max()) {
        cond_.wait(lock, predicate);
        executed = true;  // Did not time out;
      } else {
        executed = cond_.wait_for(lock, timeout, predicate);
      }
    }
    if (executed) {
      post_execution();
      return WaitResult::kSuccess;
    }
    return WaitResult::kTimeout;
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
    while (fired == kNone) {
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
        if (!block.satisfied) {
          block.cv.wait_for(bl, backstop, [&] { return block.satisfied; });
        }
        if (block.satisfied) {
          fired = static_cast<size_t>(block.fired_index);
          break;
        }
      }
      // Backstop re-poll for a persistent signal whose source isn't hooked.
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

    if (fired != kNone) {
      return std::make_pair(WaitResult::kSuccess, fired);
    }
    return std::make_pair(WaitResult::kTimeout, size_t(0));
  }

  [[nodiscard]] virtual void* native_handle() const {
    return const_cast<std::condition_variable&>(cond_).native_handle();
  }

 protected:
  [[nodiscard]] inline virtual bool signaled() const = 0;
  inline virtual void post_execution() = 0;

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
    if (cvars::threading_blocking_waitany && DeliverToWaitBlocksLocked()) {
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

 private:
  [[nodiscard]] bool signaled() const override { return signal_; }
  void post_execution() override {
    if (!manual_reset_) {
      signal_ = false;
    }
  }
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
    if (cvars::threading_blocking_waitany) {
      DeliverToWaitBlocksLocked();  // hand counts to WaitAny waiters
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
    sem_init(&suspend_sem_, 0, 0);
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
    sem_init(&suspend_sem_, 0, 0);
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
      auto set = CPU_ISSET(i, &cpu_set);
      result |= set << i;
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
      if (mask & (1 << i)) {
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
    std::unique_lock lock(callback_mutex_);
    user_callback_ = std::move(callback);
#if XE_PLATFORM_MAC
    // macOS has neither pthread_sigqueue nor sigqueue-to-thread, so no payload
    // can ride the signal. The handler runs on the target thread and recovers
    // this object from thread-local current_thread_ instead.
    pthread_kill(thread_, GetSystemSignal(SignalType::kThreadUserCallback));
#else
    sigval value{};
    value.sival_ptr = this;
#if XE_PLATFORM_ANDROID
    sigqueue(pthread_gettid_np(thread_),
             GetSystemSignal(SignalType::kThreadUserCallback), value);
#else
    pthread_sigqueue(thread_, GetSystemSignal(SignalType::kThreadUserCallback),
                     value);
#endif
#endif  // XE_PLATFORM_MAC
  }

  void CallUserCallback() const {
    std::unique_lock lock(callback_mutex_);
    user_callback_();
  }

  bool Resume(uint32_t* out_previous_suspend_count = nullptr) {
    if (out_previous_suspend_count) {
      *out_previous_suspend_count = 0;
    }
    WaitStarted();
    std::unique_lock lock(state_mutex_);
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
      // sem_post is async-signal-safe, so this is safe even if called
      // from unusual contexts.
      sem_post(&suspend_sem_);
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
      std::unique_lock lock(state_mutex_);
      if (out_previous_suspend_count) {
        *out_previous_suspend_count = suspend_count_;
      }
      already_suspended = (state_ == State::kSuspended);
      state_ = State::kSuspended;
      ++suspend_count_;
    }

    // If already suspended, just increment the count — don't send another
    // signal. A second pthread_kill while the thread is in sem_wait would
    // nest signal handlers and create multiple outstanding sem_waits, but
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
    {
      std::unique_lock lock(state_mutex_);
      if (state_ == State::kFinished) {
        if (is_current_thread) {
          // This is really bad. Some thread must have called Terminate() on us
          // just before we decided to terminate ourselves
          assert_always();
          for (;;) {
            // Wait for pthread_cancel() to actually happen.
          }
        }
        return;
      }
      state_ = State::kFinished;
    }

    {
      std::lock_guard lock(mutex_);

      exit_code_ = exit_code;
      signaled_ = true;
      cond_.notify_all();
    }
    if (is_current_thread) {
      pthread_exit(reinterpret_cast<void*>(exit_code));
    }
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
    std::unique_lock lock(state_mutex_);
    state_signal_.wait(lock,
                       [this] { return state_ != State::kUninitialized; });
  }

  /// Set state to suspended and wait until it is reset by another thread.
  /// Uses sem_wait which is async-signal-safe, allowing this to be called
  /// from a signal handler (e.g., the SIGRTMIN suspend signal handler)
  /// without risking deadlock or heap corruption from non-reentrant
  /// mutex/condvar operations.
  void WaitSuspended() {
    int ret;
    do {
      ret = sem_wait(&suspend_sem_);
    } while (ret == -1 && errno == EINTR);
  }

  void* native_handle() const override {
    return reinterpret_cast<void*>(thread_);
  }

 private:
  static void* ThreadStartRoutine(void* parameter);
  bool signaled() const override { return signaled_; }
  void post_execution() override {
    if (thread_) {
      pthread_join(thread_, nullptr);
    }
    sem_destroy(&suspend_sem_);
  }
  pthread_t thread_;
  pid_t tid_ = 0;                     // Kernel TID for setpriority() fallback
  mutable bool fifo_failed_ = false;  // True after SCHED_FIFO was rejected
  bool signaled_;
  int exit_code_;
  State state_;             // Protected by state_mutex_
  uint32_t suspend_count_;  // Protected by state_mutex_
  sem_t suspend_sem_;       // Async-signal-safe suspend/resume semaphore
  mutable std::mutex state_mutex_;
  mutable std::mutex callback_mutex_;
  mutable std::condition_variable state_signal_;
  std::function<void()> user_callback_;
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
  if (is_alertable) {
    alertable_state_ = true;
  }
  auto result = posix_wait_handle->condition().Wait(timeout);
  if (is_alertable) {
    alertable_state_ = false;
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
  if (is_alertable) {
    alertable_state_ = true;
  }
  if (posix_wait_handle_to_signal->condition().Signal()) {
    result = posix_wait_handle_to_wait_on->condition().Wait(timeout);
  }
  if (is_alertable) {
    alertable_state_ = false;
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
  if (is_alertable) {
    alertable_state_ = true;
  }
  auto result = PosixConditionBase::WaitMultiple(std::move(conditions),
                                                 wait_all, timeout);
  if (is_alertable) {
    alertable_state_ = false;
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
  EventInfo Query() override {
    EventInfo result{};
    assert_always();
    return result;
  }
  void Pulse() override {
    using namespace std::chrono_literals;
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

#if XE_PLATFORM_MAC
 private:
  // Guards set_name() against concurrent renames of this thread object.
  std::mutex name_mutex_;
#endif
};

thread_local PosixThread* current_thread_ = nullptr;

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
    std::unique_lock lock(thread->handle_.state_mutex_);
    thread->handle_.state_ =
        create_suspended ? State::kSuspended : State::kRunning;
    thread->handle_.state_signal_.notify_all();
  }

  if (create_suspended) {
    std::unique_lock lock(thread->handle_.state_mutex_);
    thread->handle_.suspend_count_ = 1;
    thread->handle_.state_signal_.wait(
        lock, [thread] { return thread->handle_.suspend_count_ == 0; });
  }

  start_routine();

  {
    std::unique_lock lock(thread->handle_.state_mutex_);
    thread->handle_.state_ = State::kFinished;
  }

  {
    std::unique_lock lock(thread->handle_.mutex_);
    thread->handle_.exit_code_ = 0;
    thread->handle_.signaled_ = true;
    thread->handle_.cond_.notify_all();
  }

  current_thread_ = nullptr;
  return nullptr;
}

std::unique_ptr<Thread> Thread::Create(CreationParameters params,
                                       std::function<void()> start_routine) {
  install_signal_handler(SignalType::kThreadSuspend);
  install_signal_handler(SignalType::kThreadUserCallback);
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
      current_thread_->WaitSuspended();
    } break;
    case SignalType::kThreadUserCallback: {
#if XE_PLATFORM_MAC
      // No sigval payload on macOS; the handler runs on the target thread, so
      // recover the thread object from thread-local state.
      assert_not_null(current_thread_);
      auto p_thread = &current_thread_->condition();
      if (alertable_state_) {
        p_thread->CallUserCallback();
      }
#else
      assert_not_null(info->si_value.sival_ptr);
      auto p_thread =
          static_cast<PosixCondition<Thread>*>(info->si_value.sival_ptr);
      if (alertable_state_) {
        p_thread->CallUserCallback();
      }
#endif  // XE_PLATFORM_MAC
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
