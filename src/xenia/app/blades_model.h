/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_BLADES_MODEL_H_
#define XENIA_APP_BLADES_MODEL_H_

#include <functional>
#include <map>
#include <string>

namespace xe {
namespace app {

class EmulatorWindow;

// The Preferences blades' view of the emulator (src/xenia/app/blades_ui).
//
// The page shows whatever BuildStateJson() describes and sends every change
// back as a small JSON message, which HandleMessage() applies: config
// settings (cvars) are overridden and saved at once, the console's own
// settings (XConfig) are written while no title holds them, and controller
// slots, bindings and hotkeys go to the input system and the emulator window.
// Nothing about the page's layout lives here; this is only state and actions.
class BladesModel {
 public:
  explicit BladesModel(EmulatorWindow& window);

  // Everything the page reads, as one JSON object (see native.ts).
  std::string BuildStateJson();

  enum class Result {
    kNone,      // Nothing changed that the page doesn't already show.
    kPush,      // Send the page the new state.
    kClose,     // The page asked to close Preferences.
    kRelaunch,  // Restart the running title (after the page has closed).
    kCloseThen,  // Close, then run TakeDeferred() (launching a game, a picker).
  };
  Result HandleMessage(const std::string& json);

  // Called a few times a second while the page is open. True when a
  // controller came or went, or a button capture finished, so the page needs
  // the new state.
  bool Poll();

  // Ends a button capture left running when the page closes.
  void CancelCapture();

  // The action a kCloseThen result runs once Preferences has closed.
  std::function<void()> TakeDeferred() { return std::move(deferred_); }

 private:
  std::string ControllersFingerprint();

  EmulatorWindow& window_;
  // Each setting's saved form when Preferences first opened, so the page can
  // tell which next-launch settings have changed since (native.ts `changed`).
  std::map<std::string, std::string> baseline_;
  std::string capturing_output_;
  int capturing_slot_ = -1;
  std::string last_controllers_;
  std::function<void()> deferred_;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_BLADES_MODEL_H_
