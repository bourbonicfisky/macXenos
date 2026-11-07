/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xevent.h"

#include <cstdlib>

#include "xenia/base/byte_stream.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/kernel/xthread.h"

DECLARE_bool(io_ledger_diag);  // defined in xboxkrnl_io.cc

namespace xe {
namespace kernel {

XEvent::XEvent(KernelState* kernel_state)
    : XObject(kernel_state, kObjectType) {}

XEvent::~XEvent() = default;

void XEvent::Initialize(bool manual_reset, bool initial_state) {
  assert_false(event_);

  this->CreateNative<X_KEVENT>();

  if (manual_reset) {
    event_ = xe::threading::Event::CreateManualResetEvent(initial_state);
  } else {
    event_ = xe::threading::Event::CreateAutoResetEvent(initial_state);
  }
  assert_not_null(event_);
}

void XEvent::InitializeNative(void* native_ptr, X_DISPATCH_HEADER* header) {
  assert_false(event_);

  switch (header->type) {
    case 0x00:  // EventNotificationObject (manual reset)
      manual_reset_ = true;
      break;
    case 0x01:  // EventSynchronizationObject (auto reset)
      manual_reset_ = false;
      break;
    default:
      assert_always();
      return;
  }

  if (manual_reset_) {
    guest_header_ptr_ = memory()->HostToGuestVirtual(native_ptr);
  }
  bool initial_state = header->signal_state ? true : false;
  if (manual_reset_) {
    event_ = xe::threading::Event::CreateManualResetEvent(initial_state);
  } else {
    event_ = xe::threading::Event::CreateAutoResetEvent(initial_state);
  }
  assert_not_null(event_);
}

int32_t XEvent::Set(uint32_t priority_increment, bool wait) {
  if (cvars::io_ledger_diag) {
    auto* cur = XThread::GetCurrentThread();
    auto* ts = cpu::ThreadState::Get();
    xboxkrnl::IoLedgerOnEventSet(guest_object(), "XEvent::Set",
                                 cur ? cur->thread_id() : 0u,
                                 ts ? static_cast<uint32_t>(ts->context()->lr)
                                    : 0u);
  }
  set_priority_increment(priority_increment);
  if (guest_header_ptr_) {
    std::lock_guard<std::mutex> lock(header_mutex_);
    WriteGuestSignalState(1);
    event_->Set();
    return 1;
  }
  event_->Set();
  return 1;
}

int32_t XEvent::Pulse(uint32_t priority_increment, bool wait) {
  if (cvars::io_ledger_diag) {
    auto* cur = XThread::GetCurrentThread();
    auto* ts = cpu::ThreadState::Get();
    xboxkrnl::IoLedgerOnEventSet(guest_object(), "XEvent::Pulse",
                                 cur ? cur->thread_id() : 0u,
                                 ts ? static_cast<uint32_t>(ts->context()->lr)
                                    : 0u);
  }
  set_priority_increment(priority_increment);
  if (guest_header_ptr_) {
    std::lock_guard<std::mutex> lock(header_mutex_);
    event_->Pulse();
    WriteGuestSignalState(0);
    return 1;
  }
  event_->Pulse();
  return 1;
}

int32_t XEvent::Reset() {
  if (cvars::io_ledger_diag) {
    auto* cur = XThread::GetCurrentThread();
    auto* ts = cpu::ThreadState::Get();
    xboxkrnl::IoLedgerOnEventReset(guest_object(), cur ? cur->thread_id() : 0u,
                                   ts ? static_cast<uint32_t>(ts->context()->lr)
                                      : 0u);
  }
  if (guest_header_ptr_) {
    std::lock_guard<std::mutex> lock(header_mutex_);
    WriteGuestSignalState(0);
    event_->Reset();
    return 1;
  }
  event_->Reset();
  return 1;
}

X_DISPATCH_HEADER* XEvent::guest_header() const {
  return memory()->TranslateVirtual<X_DISPATCH_HEADER*>(guest_header_ptr_);
}

void XEvent::WriteGuestSignalState(uint32_t state) {
  guest_header()->signal_state = state;
}

// A notification KEVENT's state is its header, and titles clear one by
// writing signal_state themselves: the XDK inlines KeClearEvent as a single
// store. The host event only read the header when it was created, so a wait
// after such a clear returned at once. Dynasty Warriors 6's Musou worker
// clears its command events, signals its reply and waits again; the stale
// host event woke it with no command posted and it called through a null
// object (4 of 4 boots on the briefing screen). Set/Pulse/Reset keep the
// header current and each wait first takes the header's state.
// Synchronization (auto-reset) events are left alone: a released waiter
// consumes them on the host side, which the header cannot record.
void XEvent::SyncGuestStateBeforeWait() {
  static const bool enabled = [] {
    const char* v = std::getenv("XE_EVENT_HEADER_SYNC");
    return !(v && v[0] == '0' && !v[1]);
  }();
  if (!enabled || !guest_header_ptr_) {
    return;
  }
  std::lock_guard<std::mutex> lock(header_mutex_);
  if (guest_header()->signal_state) {
    event_->Set();
  } else {
    event_->Reset();
  }
}

void XEvent::Query(uint32_t* out_type, uint32_t* out_state) {
  auto [type, state] = event_->Query();

  *out_type = type;
  *out_state = state;
}
void XEvent::Clear() { Reset(); }

bool XEvent::Save(ByteStream* stream) {
  XELOGD("XEvent {:08X} ({})", handle(), manual_reset_ ? "manual" : "auto");
  SaveObject(stream);

  bool signaled = true;
  auto result =
      xe::threading::Wait(event_.get(), false, std::chrono::milliseconds(0));
  if (result == xe::threading::WaitResult::kSuccess) {
    signaled = true;
  } else if (result == xe::threading::WaitResult::kTimeout) {
    signaled = false;
  } else {
    assert_always();
  }

  if (signaled) {
    // Reset the event in-case it's an auto-reset.
    event_->Set();
  }

  stream->Write<bool>(signaled);
  stream->Write<bool>(manual_reset_);

  return true;
}

object_ref<XEvent> XEvent::Restore(KernelState* kernel_state,
                                   ByteStream* stream) {
  auto evt = new XEvent(nullptr);
  evt->kernel_state_ = kernel_state;

  evt->RestoreObject(stream);
  bool signaled = stream->Read<bool>();
  evt->manual_reset_ = stream->Read<bool>();

  if (evt->manual_reset_) {
    evt->event_ = xe::threading::Event::CreateManualResetEvent(false);
  } else {
    evt->event_ = xe::threading::Event::CreateAutoResetEvent(false);
  }
  assert_not_null(evt->event_);

  if (signaled) {
    evt->event_->Set();
  }

  return object_ref<XEvent>(evt);
}

}  // namespace kernel
}  // namespace xe
