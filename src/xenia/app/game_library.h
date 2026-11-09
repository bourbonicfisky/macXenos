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
#include <chrono>
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
    // Which disc of a multi-disc game, from the "(Disc 2)" tag the name
    // drops; 0 for a single-disc game.
    int disc = 0;
  };

  // How far the running scan has got. It first lists the folders (`found`
  // grows, `total` is 0), then reads each game it found (`checked` of
  // `total`); games an earlier scan read and that have not changed are
  // checked straight away.
  struct ScanProgress {
    size_t found = 0;
    size_t checked = 0;
    size_t total = 0;
  };

  // `on_changed` is called on the scan thread while a scan runs (at most a
  // few times a second, for its progress) and when it finishes.
  GameLibrary(std::filesystem::path storage_root,
              std::function<void()> on_changed);
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
  ScanProgress scan_progress() const;
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
  // Calls on_changed_ unless it was called less than a moment ago.
  void ReportProgress(bool force);

  std::filesystem::path storage_root_;
  std::function<void()> on_changed_;

  mutable std::mutex mutex_;
  std::vector<std::filesystem::path> folders_;
  std::vector<Game> games_;

  std::atomic<bool> scanning_{false};
  std::atomic<bool> rescan_requested_{false};
  std::atomic<bool> stop_{false};
  std::atomic<size_t> scan_found_{0};
  std::atomic<size_t> scan_checked_{0};
  std::atomic<size_t> scan_total_{0};
  // Scan thread only.
  std::chrono::steady_clock::time_point last_report_;
  std::thread thread_;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_GAME_LIBRARY_H_
