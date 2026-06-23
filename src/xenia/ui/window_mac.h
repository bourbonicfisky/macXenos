/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_WINDOW_MAC_H_
#define XENIA_UI_WINDOW_MAC_H_

#include <atomic>
#include <memory>
#include <string>

#include "xenia/ui/menu_item.h"
#include "xenia/ui/window.h"

namespace xe {
namespace ui {

class MacWindow : public Window {
 public:
  MacWindow(WindowedAppContext& app_context, const std::string_view title,
            uint32_t desired_logical_width, uint32_t desired_logical_height)
      : Window(app_context, title, desired_logical_width,
               desired_logical_height) {}
  ~MacWindow();

  // macOS reports sizes in points; 96 is treated as the unscaled baseline, with
  // the backing scale factor folded into GetLatestDpiImpl.
  uint32_t GetMediumDpi() const override { return 96; }

  // Called by the Cocoa window delegate (defined in the .mm).
  void HandleResize();
  void HandleNativeClose();
  void HandleFocus(bool has_focus);

  // Called by the content NSView on keyDown:/keyUp:. ns_event is an opaque
  // NSEvent* (this header is included from plain C++ too). Translates the native
  // event into Xenia key events and forwards printable characters to ImGui.
  void HandleKeyEvent(void* ns_event, bool is_down);

  // Called by the content NSView on mouse events. ns_event is an opaque
  // NSEvent*. |button| matches ui::MouseEvent::Button (0=none, 1=left, 2=right,
  // 3=middle); |action| is 0=down, 1=up, 2=move, 3=wheel. Translates to
  // window-local physical pixels and dispatches to the input listeners (ImGui),
  // so the in-window menu and dialogs are clickable.
  void HandleMouseEvent(void* ns_event, int button, int action);

 protected:
  uint32_t GetLatestDpiImpl() const override;

  bool OpenImpl() override;
  void RequestCloseImpl() override;
  void ApplyNewTitle() override;
  void FocusImpl() override;

  std::unique_ptr<Surface> CreateSurfaceImpl(
      Surface::TypeFlags allowed_types) override;
  void RequestPaintImpl() override;

 private:
  // Updates the Window's actual size from the native view's backing pixels.
  void UpdateSizeFromView();

  // Coalesces paint requests: at most one paint is queued on the main dispatch
  // queue at a time (see RequestPaintImpl). Atomic because RequestPaint may be
  // called from non-UI threads.
  std::atomic<bool> paint_scheduled_{false};

  // Opaque Objective-C pointers (this header is included from plain C++ too).
  void* window_ = nullptr;     // NSWindow*
  void* metal_view_ = nullptr; // NSView* hosting a CAMetalLayer
  void* delegate_ = nullptr;   // XeniaWindowDelegate*
};

// Menus are not yet implemented on macOS; this is a no-op placeholder mirroring
// AndroidMenuItem.
class MacMenuItem final : public MenuItem {
 public:
  MacMenuItem(Type type, const std::string& text, const std::string& hotkey,
              std::function<void()> callback)
      : MenuItem(type, text, hotkey, callback) {}
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOW_MAC_H_
