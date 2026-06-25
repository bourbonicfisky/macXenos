/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_PROCESS_RELAUNCH_MAC_H_
#define XENIA_UI_PROCESS_RELAUNCH_MAC_H_

#include <string>

namespace xe {
namespace ui {

// Re-exec the current process, booting the given title via --target. Used by
// the emulation lifecycle controls so that switching titles (Load new game /
// Start / Reboot) runs in a pristine process - in-process relaunch corrupts
// guest memory because kernel_state_ persists across titles (see the
// add-emulation-lifecycle-controls design). The caller is responsible for
// flushing config + the host shader cache before calling this.
//
// macOS only. On success this does not return (the image is replaced). Returns
// false only if the relaunch failed.
bool RelaunchWithTarget(const std::string& target_path);

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_PROCESS_RELAUNCH_MAC_H_
