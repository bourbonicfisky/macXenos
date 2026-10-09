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

#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace xe {
namespace app {

class EmulatorWindow;
class StateWriter;

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
  // The games the signed-in profiles have played: their title IDs by
  // lower-case name (for history recorded before IDs were kept) and the first
  // player's achievements and gamerscore in each.
  struct PlayedTitles {
    struct Progress {
      uint32_t unlocked, total, score, max_score;
    };
    std::map<std::string, uint32_t> ids_by_name;
    std::map<uint32_t, Progress> progress;
  };

  // BuildStateJson's sections, each writing its own keys.
  void WriteRenderer(StateWriter& w);
  void WriteSettings(StateWriter& w);
  void WriteControllers(StateWriter& w);
  void WriteConsole(StateWriter& w);
  PlayedTitles CollectPlayedTitles();
  void WriteTitles(StateWriter& w, const PlayedTitles& played);
  void WriteMusicPlayer(StateWriter& w);
  void WriteProfiles(StateWriter& w);
  void WriteApp(StateWriter& w);

  std::string ControllersFingerprint();

  // A profile's games, gamerscore and achievements, as its gamercard shows
  // them. Read from the profile while it is signed in, and remembered
  // (profile_stats.toml in the storage root) so a signed-out profile still
  // shows its last ones.
  struct ProfileStats {
    uint32_t games = 0;
    uint32_t gamerscore = 0;
    uint32_t achievements = 0;
    bool operator==(const ProfileStats&) const = default;
  };
  const ProfileStats* StatsFor(uint64_t xuid);
  void LoadProfileStats();
  void SaveProfileStats();
  std::map<uint64_t, ProfileStats> profile_stats_;
  // The gamer-picture download's progress last sent to the page, in tenths
  // of a percent, so Poll pushes as it moves.
  int gamer_pictures_progress_sent_ = -1;
  bool profile_stats_loaded_ = false;

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
