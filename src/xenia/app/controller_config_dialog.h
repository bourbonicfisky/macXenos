/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_CONTROLLER_CONFIG_DIALOG_H_
#define XENIA_APP_CONTROLLER_CONFIG_DIALOG_H_

#include "xenia/hid/input_driver.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"

namespace xe {
namespace app {

class EmulatorWindow;

// User-facing controller UI: lists the controllers in each player slot (with
// detected brand and name), and is the home for rebinding, glyphs, and slot
// assignment. The guest always receives an Xbox 360 pad regardless of what is
// shown here.
class ControllerConfigDialog final : public ui::ImGuiDialog {
 public:
  ControllerConfigDialog(ui::ImGuiDrawer* imgui_drawer,
                         EmulatorWindow& emulator_window)
      : ui::ImGuiDialog(imgui_drawer), emulator_window_(emulator_window) {}

 protected:
  void OnDraw(ImGuiIO& io) override;

 private:
  void DrawConnectedControllers();
  void DrawHotkeys();

  EmulatorWindow& emulator_window_;
};

// Human-readable brand label for a controller kind (text form of the glyph
// set; a graphical glyph set can replace/augment this later).
const char* ControllerKindLabel(hid::ControllerKind kind);

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_CONTROLLER_CONFIG_DIALOG_H_
