/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_UTIL_GUEST_SUSPEND_H_
#define XENIA_KERNEL_UTIL_GUEST_SUSPEND_H_

#include <cstdint>

#include "xenia/base/threading.h"

namespace xe {
namespace kernel {
namespace util {

// Keeps a host thread's suspend count in step with the guest KTHREAD suspend
// count for suspends made from another thread. Each guest suspend holds the
// host thread once and each guest resume of a non-zero count releases it once,
// so a thread suspended N times runs again after N resumes.
//
// The legacy POSIX rule held the host thread on every suspend but released it
// only when the guest count reached zero. A thread suspended twice (a title's
// SuspendThread on a thread it created suspended, or two nested suspends) kept
// a host count of 1 after both resumes and never ran again: a thread created
// suspended stayed parked before its first guest instruction.
//
// Call both with the owning thread's suspend lock held, never on the thread
// being suspended (a self-suspend parks without the host thread). Return the
// previous guest count.
inline uint8_t GuestSuspendHost(uint8_t* guest_count,
                                threading::Thread* host_thread) {
  const uint8_t previous = *guest_count;
  if (previous == UINT8_MAX) {
    // Saturated; the old code wrapped to 0 and left the thread running.
    return previous;
  }
  *guest_count = uint8_t(previous + 1);
  host_thread->Suspend(nullptr);
  return previous;
}

inline uint8_t GuestResumeHost(uint8_t* guest_count,
                               threading::Thread* host_thread,
                               bool legacy_release_at_zero = false) {
  const uint8_t previous = *guest_count;
  if (previous) {
    *guest_count = uint8_t(previous - 1);
  }
  const bool release =
      legacy_release_at_zero ? *guest_count == 0 : previous != 0;
  if (release) {
    // False when the host count is already 0 (a self-suspended thread parks
    // on the guest count alone).
    host_thread->Resume(nullptr);
  }
  return previous;
}

}  // namespace util
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_UTIL_GUEST_SUSPEND_H_
