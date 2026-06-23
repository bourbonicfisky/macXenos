/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_WINDOWED_APP_CONTEXT_MAC_H_
#define XENIA_UI_WINDOWED_APP_CONTEXT_MAC_H_

#include "xenia/ui/windowed_app_context.h"

namespace xe {
namespace ui {

// macOS windowed app context driving an NSApplication run loop. Pending UI
// functions are dispatched to the main queue, which wakes the run loop.
class MacWindowedAppContext final : public WindowedAppContext {
 public:
  MacWindowedAppContext();
  ~MacWindowedAppContext();

  void NotifyUILoopOfPendingFunctions() override;
  void PlatformQuitFromUIThread() override;

  // Runs [NSApp run] until QuitFromUIThread is requested.
  void RunMainMacLoop();
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOWED_APP_CONTEXT_MAC_H_
