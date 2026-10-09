/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_GAMER_PICTURES_MAC_H_
#define XENIA_APP_GAMER_PICTURES_MAC_H_

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace xe {
namespace app {

// The Xbox 360's gamer pictures, as a one-time download on request: the
// "Xbox 360 Profile Picture Collection" on the Internet Archive, a zip of the
// console's gamer-picture packages (content type 0x00020000). Nothing of it
// is shipped. Installing opens each package and keeps each picture that comes
// as a complete pair of the right sizes (64_<key>.png at 64x64, 32_<key>.png
// at 32x32, the two sizes a profile's picture is stored in), in the folder
// where XAM puts gamer pictures a game awards (content/0000000000000000/
// FFFE07D1/00020000/<title id>/, xam_user.cc). An index of what it installed
// is kept in <storage_root>/gamer_pictures.toml, for choosing a profile's
// picture from them.
namespace gamer_pictures {

extern const char* const kCollectionName;
extern const char* const kSite;
// The zip's size, for the page to say what the download costs.
extern const int64_t kDownloadBytes;

struct Summary {
  uint32_t pictures = 0;  // Distinct gamer pictures (each in two sizes).
  uint32_t titles = 0;    // Games and other titles they belong to.
};

// What an earlier install left, from its index.
std::optional<Summary> Installed(const std::filesystem::path& storage_root);

// Shared with the page while a download runs.
struct Progress {
  std::atomic<int64_t> received{0};
  std::atomic<int64_t> total{0};
  std::atomic<bool> installing{false};
};

// Downloads, checks (the archive's SHA-1) and installs the collection in the
// background. `done` is called once, on a background thread, with an empty
// error on success.
void Download(const std::filesystem::path& storage_root,
              const std::filesystem::path& content_root,
              std::shared_ptr<Progress> progress,
              std::function<void(std::string error, Summary summary)> done);

}  // namespace gamer_pictures
}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_GAMER_PICTURES_MAC_H_
