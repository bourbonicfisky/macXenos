/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/util/input_notification.h"

#include "xenia/kernel/kernel_state.h"

namespace xe::kernel {

void BroadcastInputDevicesChanged() {
  if (KernelState* state = KernelState::shared()) {
    state->BroadcastNotification(kXNotificationSystemInputDevicesChanged, 0);
  }
}

}  // namespace xe::kernel
