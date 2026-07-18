/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_UTIL_INPUT_NOTIFICATION_H_
#define XENIA_KERNEL_UTIL_INPUT_NOTIFICATION_H_

namespace xe::kernel {

// Broadcasts the guest input-device-change notification when a kernel is
// active. Kept outside HID so focused InputSystem tests don't need to link the
// complete kernel and UI implementation.
void BroadcastInputDevicesChanged();

}  // namespace xe::kernel

#endif  // XENIA_KERNEL_UTIL_INPUT_NOTIFICATION_H_
