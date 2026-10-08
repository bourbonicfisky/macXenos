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
#include <functional>
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

  // Shows the standard macOS About panel (app icon + version from the bundle)
  // with a credits block of clickable links. Also invoked by the application
  // menu's About item.
  void ShowAboutDialog() override;

  // The window's content view (NSView*): a plain container holding the game's
  // Metal view, so other views (the Preferences overlay) can sit above it.
  void* native_container_view() const { return container_view_; }
  // The Metal view inside the container (NSView*), which has keyboard focus.
  void* native_game_view() const { return metal_view_; }

  // What the application menu's Settings… item (⌘,) does.
  void SetSettingsHandler(std::function<void()> handler) {
    settings_handler_ = std::move(handler);
  }
  void HandleSettingsRequested() {
    if (settings_handler_) {
      auto handler = settings_handler_;
      handler();
    }
  }

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
  void ApplyNewFullscreen() override;
  // Resizes the NSWindow content to the desired logical size (un-zooming first),
  // clamped to the screen's visible frame. See window.h.
  void SetDesiredLogicalSizeImpl() override;
  void FocusImpl() override;

  // Builds the native NSMenu tree from GetMainMenu() and installs it as the
  // app-global NSApp.mainMenu (prepending a standard application menu). The
  // macOS menu bar is global rather than per-window; with the single-window
  // emulator this is applied from the window that owns the menu.
  void ApplyNewMainMenu(MenuItem* old_main_menu) override;
  void CompleteMainMenuItemsUpdateImpl() override;

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

  // True while a programmatic SetDesiredLogicalSizeImpl resize is in flight, so
  // the resulting size feedback isn't mistaken for a user (kManual) resize and
  // doesn't overwrite the window_size_x/y cvars.
  bool applying_desired_size_ = false;

  // Opaque Objective-C pointers (this header is included from plain C++ too).
  void* window_ = nullptr;     // NSWindow*
  void* metal_view_ = nullptr; // NSView* hosting a CAMetalLayer
  void* container_view_ = nullptr;  // NSView*, the window's content view
  std::function<void()> settings_handler_;
  void* delegate_ = nullptr;   // XeniaWindowDelegate*
};

// Native macOS menu backing. Mirrors Win32MenuItem / GTKMenuItem: each node
// eagerly builds its native objects in the constructor and appends children as
// the cross-platform tree is assembled. kNormal/kPopup own an NSMenu; the item
// shown in the parent's NSMenu is the NSMenuItem (a separator, a leaf, or the
// submenu carrier for a popup).
class MacMenuItem final : public MenuItem {
 public:
  MacMenuItem(Type type, const std::string& text, const std::string& hotkey,
              std::function<void()> callback);
  ~MacMenuItem() override;

  // Opaque Objective-C pointers (this header is included from plain C++ too).
  void* ns_menu() const { return ns_menu_; }            // NSMenu*
  void* ns_menu_item() const { return ns_menu_item_; }  // NSMenuItem*

  void SetEnabled(bool enabled) override;

  using MenuItem::OnSelected;

 protected:
  void OnChildAdded(MenuItem* child_item) override;
  void OnChildRemoved(MenuItem* child_item) override;

 private:
  void* ns_menu_ = nullptr;       // NSMenu* (retained) for kNormal/kPopup
  void* ns_menu_item_ = nullptr;  // NSMenuItem* (retained) for popup/leaf/sep
  void* target_ = nullptr;        // XeniaMenuTarget* (retained) for kString
};

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_WINDOW_MAC_H_
