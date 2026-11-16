/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XTIMER_H_
#define XENIA_KERNEL_XTIMER_H_

#include <atomic>
#include <mutex>

#include "xenia/base/threading.h"
#include "xenia/kernel/xobject.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {

class XThread;

class XTimer : public XObject {
 public:
  static const XObject::Type kObjectType = XObject::Type::Timer;

  explicit XTimer(KernelState* kernel_state);
  ~XTimer() override;

  void Initialize(uint32_t timer_type);
  void InitializeNative(void* native_ptr, X_DISPATCH_HEADER* header);
  bool SetNativeTimer(int64_t due_time, uint32_t period_ms, uint32_t dpc,
                     std::function<void()> callback);
  bool CancelNativeTimer();

  X_STATUS SetTimer(int64_t due_time, uint32_t period_ms, uint32_t routine,
                    uint32_t routine_arg, bool resume);
  X_STATUS Cancel();
  bool IsSignaled();

 protected:
  xe::threading::WaitHandle* GetWaitHandle() override;
  void WaitCallback() override;

 private:
  std::unique_ptr<xe::threading::Timer> timer_;
  bool manual_reset_ = false;
  std::mutex timer_lock_;

  // Callback state outlives any in-flight host callback. Scheduling takes
  // timer_lock_; callbacks take only this lock, so Cancel can drain them.
  struct NativeState {
    std::mutex mutex;
    uint64_t generation = 0;
    uint32_t guest_ptr = 0;
    bool pending = false;
    std::unique_ptr<xe::threading::Event> event;
  };
  std::shared_ptr<NativeState> native_state_;

  // Takes back this timer's APC if it has expired but not run. Called with
  // timer_lock_ held, before a cancel or a new schedule.
  void RetractApc();

  XThread* callback_thread_ = nullptr;
  uint32_t callback_thread_handle_ = 0;
  uint32_t callback_routine_ = 0;
  uint32_t callback_routine_arg_ = 0;
  // Bumped by every cancel and every new schedule; a fire that was dispatched
  // under an older value queues nothing.
  std::shared_ptr<std::atomic<uint64_t>> apc_generation_ =
      std::make_shared<std::atomic<uint64_t>>(0);
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XTIMER_H_
