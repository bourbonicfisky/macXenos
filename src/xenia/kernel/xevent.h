/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XEVENT_H_
#define XENIA_KERNEL_XEVENT_H_

#include <atomic>
#include <mutex>

#include "xenia/base/threading.h"
#include "xenia/kernel/xobject.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {

// https://www.nirsoft.net/kernel_struct/vista/KEVENT.html
struct X_KEVENT {
  X_DISPATCH_HEADER header;
};
static_assert_size(X_KEVENT, 0x10);

class XEvent : public XObject {
 public:
  static const XObject::Type kObjectType = XObject::Type::Event;

  explicit XEvent(KernelState* kernel_state);
  ~XEvent() override;

  void Initialize(bool manual_reset, bool initial_state);
  void InitializeNative(void* native_ptr, X_DISPATCH_HEADER* header);

  int32_t Set(uint32_t priority_increment, bool wait);
  int32_t Pulse(uint32_t priority_increment, bool wait);
  int32_t Reset();
  void Query(uint32_t* out_type, uint32_t* out_state);
  void Clear();

  bool Save(ByteStream* stream) override;
  static object_ref<XEvent> Restore(KernelState* kernel_state,
                                    ByteStream* stream);

 protected:
  xe::threading::WaitHandle* GetWaitHandle() override { return event_.get(); }
  void SyncGuestStateBeforeWait() override;

 private:
  // Guest address of the title-owned KEVENT a notification event was built
  // from (InitializeNative), or 0: its header's signal_state is kept in step
  // with the host event (see SyncGuestStateBeforeWait).
  X_DISPATCH_HEADER* guest_header() const;
  void WriteGuestSignalState(uint32_t state);
  // The signal state the Ke calls report as the one from before. Needs
  // header_mutex_.
  int32_t SignalStateLocked() const;

  bool manual_reset_ = false;
  uint32_t guest_header_ptr_ = 0;
  // The header's signal_state as this event last stored or read it. A wait
  // whose header still holds it has nothing to sync and skips the lock.
  std::atomic<int32_t> header_state_seen_{0};
  // Orders header writes and host event changes against the wait-time sync,
  // so a sync never reads a header older than the host event.
  std::mutex header_mutex_;
  std::unique_ptr<xe::threading::Event> event_;
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XEVENT_H_
