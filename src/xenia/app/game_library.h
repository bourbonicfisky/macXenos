/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_GAME_LIBRARY_H_
#define XENIA_APP_GAME_LIBRARY_H_

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xe {
namespace app {

// The folders that hold the user's games, and what is in them, for the
// dashboard's All Games list.
//
// A scan walks each folder a few levels deep and finds disc images (.iso,
// .zar), unpacked games (a folder with default.xex) and downloaded games
// (Arcade, Games on Demand and demo packages). It reads each one's title ID
// (and, for packages, its name and icon) without starting it, and keeps what
// it learned in game_index.toml so a later scan only reads new or changed
// files. Scans run on their own thread.
class GameLibrary {
 public:
  struct Game {
    std::filesystem::path path;  // What RunTitle launches.
    // The game's own name when the file carries one (packages), else the
    // file or folder name without its region and language tags.
    std::string name;
    uint32_t title_id = 0;  // 0 when it could not be read.
  };

  // `on_scanned` is called on the scan thread when a scan finishes.
  GameLibrary(std::filesystem::path storage_root,
              std::function<void()> on_scanned);
  ~GameLibrary();

  GameLibrary(const GameLibrary&) = delete;
  GameLibrary& operator=(const GameLibrary&) = delete;

  std::vector<std::filesystem::path> folders() const;
  // Both save the list and start a new scan.
  void AddFolder(const std::filesystem::path& folder);
  void RemoveFolder(const std::filesystem::path& folder);

  // Starts a scan unless one is running.
  void Rescan();
  bool scanning() const { return scanning_; }
  std::vector<Game> games() const;

 private:
  struct IndexEntry {
    uint64_t size = 0;
    int64_t mtime = 0;
    uint32_t title_id = 0;
    std::string name;
  };

  void LoadFolders();
  void SaveFolders() const;
  void Scan(std::vector<std::filesystem::path> folders);

  std::filesystem::path storage_root_;
  std::function<void()> on_scanned_;

  mutable std::mutex mutex_;
  std::vector<std::filesystem::path> folders_;
  std::vector<Game> games_;

  std::atomic<bool> scanning_{false};
  std::atomic<bool> rescan_requested_{false};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_GAME_LIBRARY_H_
