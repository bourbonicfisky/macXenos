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
#include <vector>

namespace xe {
namespace app {

// The Xbox 360's gamer pictures, as a one-time download on request: the
// "Xbox 360 Profile Picture Collection" on the Internet Archive, a zip of the
// console's gamer-picture packages (content type 0x00020000). Nothing of it
// is shipped. Installing opens each package and keeps each picture's 64x64
// (64_<key>.png) in the folder where XAM puts gamer pictures a game awards
// (content/0000000000000000/FFFE07D1/00020000/<title id>/, xam_user.cc).
// The packages' 32x32 copies are not kept: nothing reads them from there,
// and a profile's small picture is made from the 64x64 when it is chosen. An index of what it installed
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

// What an earlier install left, from its index. Also writes the index as
// gamer_pictures.json (titles, names, picture keys) for the page's picker,
// if an install left only the TOML.
std::optional<Summary> Installed(const std::filesystem::path& storage_root);

// A PNG's width and height from its header, or 0x0 if it isn't one.
std::pair<uint32_t, uint32_t> PngSize(const std::vector<uint8_t>& data);

// The 32x32 of a 64x64 gamer picture, as the console stored a profile's
// small picture (tile_32.png); empty if `png64` isn't a 64x64 PNG.
std::vector<uint8_t> SmallPicture(const std::vector<uint8_t>& png64);

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
