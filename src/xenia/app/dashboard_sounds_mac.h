/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_DASHBOARD_SOUNDS_MAC_H_
#define XENIA_APP_DASHBOARD_SOUNDS_MAC_H_

#include <filesystem>
#include <functional>
#include <string>

namespace xe {
namespace app {

// The Blades dashboard's sounds are not shipped with the app. On request they
// are downloaded from a community sound pack ("Xbox 360 Blades" by squabbled,
// published on DeckThemes), checked against a pinned SHA-256, and unpacked
// into `folder` under the names the page asks for (focus.wav, select.wav...).
namespace dashboard_sounds {

// Where the pack comes from, for the page to credit.
extern const char* const kPackName;
extern const char* const kPackAuthor;
extern const char* const kPackSite;

// True when every sound the dashboard plays is in `folder`.
bool Present(const std::filesystem::path& folder);

// Downloads and installs the pack into `folder` in the background. `done` is
// called once, on a background thread, with an empty string on success or a
// short reason for the failure.
void Download(const std::filesystem::path& folder,
              std::function<void(std::string error)> done);

}  // namespace dashboard_sounds
}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_DASHBOARD_SOUNDS_MAC_H_
