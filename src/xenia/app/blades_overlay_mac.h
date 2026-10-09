/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_BLADES_OVERLAY_MAC_H_
#define XENIA_APP_BLADES_OVERLAY_MAC_H_

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace xe {
namespace app {

// The Preferences blades, drawn by WebKit over the game.
//
// A WKWebView sits above the window's Metal view in the same container and
// loads the page from src/xenia/app/blades_ui/dist, which the build copies
// beside the binary as blades/ (and the packager into Contents/Resources).
// It is served from its own URL scheme, so the page's absolute paths (/fonts)
// resolve against that folder. Messages from the page arrive
// through `on_message`; PushState hands the page new state.
//
// Plain C++ interface (the .mm owns the Objective-C objects), so the
// emulator window can hold one without including AppKit.
// The Mac's preferred languages, most preferred first (System Settings >
// General > Language & Region), as BCP 47 tags: the dashboard's language
// follows them unless the user picks one.
std::vector<std::string> PreferredLanguages();

class BladesOverlay {
 public:
  struct Callbacks {
    // A JSON message posted by the page (see native.ts NativeMessage).
    std::function<void(const std::string& json)> on_message;
    // Called a few times a second while the overlay is visible.
    std::function<void()> on_tick;
  };

  // `container_view` is the window's content view (NSView*), `game_view` the
  // Metal view inside it that has keyboard focus when the overlay is hidden.
  // `title_icon_root` is served to the page as /title-icons/ (the games'
  // icons, EmulatorWindow::TitleIconPath). `sound_root` holds the downloaded
  // dashboard sounds (dashboard_sounds_mac.h); missing ones stay silent.
  BladesOverlay(void* container_view, void* game_view,
                const std::filesystem::path& title_icon_root,
                const std::filesystem::path& sound_root, Callbacks callbacks);
  ~BladesOverlay();

  BladesOverlay(const BladesOverlay&) = delete;
  BladesOverlay& operator=(const BladesOverlay&) = delete;

  // Where the built page lives, or empty if it was not shipped.
  static std::filesystem::path FindPageRoot();

  // Shows the overlay on one blade ("graphics", "controller", ...), loading
  // the page the first time.
  void Show(const std::string& blade_id);
  void Hide();
  bool visible() const { return visible_; }

  // Gives the page its whole state (a JSON object, native.ts NativeState).
  void PushState(const std::string& json);

  // Forwards a controller button to the page ("up", "a", "lb", "start"...).
  void PadButton(const char* button, bool down);

 private:
  void* impl_ = nullptr;  // XeniaBladesHost*
  bool visible_ = false;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_BLADES_OVERLAY_MAC_H_
