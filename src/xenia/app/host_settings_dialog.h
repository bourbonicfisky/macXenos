/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_HOST_SETTINGS_DIALOG_H_
#define XENIA_APP_HOST_SETTINGS_DIALOG_H_

#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"

namespace xe {
namespace app {

class EmulatorWindow;

// In-app "Host / Advanced Settings" dialog. Surfaces a curated allowlist of
// host/renderer cvars (NOT the whole registry - most cvars are dev/debug
// internals). Controls are rendered by cvar type and the explainer text is
// sourced from each cvar's description(), so help never drifts from the cvar
// definition. Edits persist through config::SaveConfig(). On macOS, when a
// title is loaded, an "Apply & Relaunch" button re-execs the current title so
// startup-only settings (e.g. mvk_config_defaults) take effect immediately.
class HostSettingsDialog final : public ui::ImGuiDialog {
 public:
  HostSettingsDialog(ui::ImGuiDrawer* imgui_drawer,
                     EmulatorWindow& emulator_window)
      : ui::ImGuiDialog(imgui_drawer), emulator_window_(emulator_window) {}

 protected:
  void OnDraw(ImGuiIO& io) override;
  void OnClose() override;

 private:
  EmulatorWindow& emulator_window_;
  double save_confirmation_disappearance_ = 0.0;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_HOST_SETTINGS_DIALOG_H_
