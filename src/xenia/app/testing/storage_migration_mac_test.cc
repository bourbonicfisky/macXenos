/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/app/storage_migration_mac.h"

namespace xe {
namespace app {
namespace test {

namespace fs = std::filesystem;

namespace {

// A fresh folder per test case, removed afterwards.
struct Scratch {
  fs::path root;
  Scratch() {
    static int counter = 0;
    root = fs::temp_directory_path() /
           ("xe_storage_migration_test_" + std::to_string(getpid()) + "_" +
            std::to_string(counter++));
    fs::remove_all(root);
    fs::create_directories(root);
  }
  ~Scratch() {
    std::error_code ec;
    // Undo any permission a test took away so the folder can be removed.
    for (auto it = fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      chmod(it->path().c_str(), 0755);
    }
    chmod(root.c_str(), 0755);
    fs::remove_all(root, ec);
  }
};

void Write(const fs::path& path, const std::string& contents) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << contents;
}

std::string Read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

constexpr const char* kMarker = ".copied-from-legacy";

}  // namespace

TEST_CASE("Legacy storage: a fresh copy brings everything and finishes",
          "[storage_migration]") {
  Scratch s;
  const fs::path legacy = s.root / "legacy", root = s.root / "root";
  Write(legacy / "xenia-canary.config.toml", "old config");
  Write(legacy / "profiles" / "E000" / "Account", "account");
  Write(legacy / "content" / "E000" / "4D5307E6" / "00000001" / "save",
        "save");
  MigrationReport report = CopyLegacyStorage(legacy, root);
  REQUIRE(report.failures.empty());
  REQUIRE(report.complete);
  REQUIRE(fs::exists(root / kMarker));
  REQUIRE(Read(root / "xenia-canary.config.toml") == "old config");
  REQUIRE(Read(root / "profiles" / "E000" / "Account") == "account");
  REQUIRE(Read(root / "content" / "E000" / "4D5307E6" / "00000001" /
               "save") == "save");
  // The old folder stays as a backup.
  REQUIRE(Read(legacy / "profiles" / "E000" / "Account") == "account");
}

TEST_CASE("Legacy storage: an interrupted copy resumes on the next launch",
          "[storage_migration]") {
  Scratch s;
  const fs::path legacy = s.root / "legacy", root = s.root / "root";
  Write(legacy / "xenia-canary.config.toml", "old config");
  Write(legacy / "profiles" / "E000" / "Account", "account");
  Write(legacy / "content" / "E000" / "title" / "file", "content");
  // The first launch got as far as the config, and died inside the content
  // copy (its temporary name is left behind), before writing the marker.
  Write(root / "xenia-canary.config.toml", "old config");
  Write(root / "content.copying" / "E000" / "title" / "fi", "partial");
  MigrationReport report = CopyLegacyStorage(legacy, root);
  REQUIRE(report.failures.empty());
  REQUIRE(report.complete);
  REQUIRE(fs::exists(root / kMarker));
  REQUIRE(Read(root / "profiles" / "E000" / "Account") == "account");
  REQUIRE(Read(root / "content" / "E000" / "title" / "file") == "content");
  REQUIRE_FALSE(fs::exists(root / "content.copying"));
}

TEST_CASE("Legacy storage: files already at the destination are kept",
          "[storage_migration]") {
  Scratch s;
  const fs::path legacy = s.root / "legacy", root = s.root / "root";
  Write(legacy / "xenia-canary.config.toml", "old config");
  Write(legacy / "profiles" / "E000" / "Account", "old account");
  Write(legacy / "profiles" / "E000" / "Picture", "picture");
  Write(root / "xenia-canary.config.toml", "new config");
  Write(root / "profiles" / "E000" / "Account", "new account");
  MigrationReport report = CopyLegacyStorage(legacy, root);
  REQUIRE(report.failures.empty());
  REQUIRE(report.complete);
  REQUIRE(Read(root / "xenia-canary.config.toml") == "new config");
  REQUIRE(Read(root / "profiles" / "E000" / "Account") == "new account");
  // A folder on both sides is merged, not skipped.
  REQUIRE(Read(root / "profiles" / "E000" / "Picture") == "picture");
  REQUIRE(report.kept == 2);
}

TEST_CASE("Legacy storage: a denied read fails, reports, and retries",
          "[storage_migration]") {
  if (geteuid() == 0) {
    SUCCEED("permissions do not apply to root");
    return;
  }
  Scratch s;
  const fs::path legacy = s.root / "legacy", root = s.root / "root";
  Write(legacy / "xenia-canary.config.toml", "old config");
  Write(legacy / "profiles" / "E000" / "Account", "account");
  REQUIRE(chmod((legacy / "profiles").c_str(), 0) == 0);
  MigrationReport report = CopyLegacyStorage(legacy, root);
  REQUIRE_FALSE(report.failures.empty());
  REQUIRE_FALSE(report.complete);
  REQUIRE_FALSE(fs::exists(root / kMarker));
  REQUIRE_FALSE(fs::exists(root / "profiles" / "E000" / "Account"));
  // Fixed: the next launch finishes the job.
  REQUIRE(chmod((legacy / "profiles").c_str(), 0755) == 0);
  report = CopyLegacyStorage(legacy, root);
  REQUIRE(report.failures.empty());
  REQUIRE(report.complete);
  REQUIRE(Read(root / "profiles" / "E000" / "Account") == "account");
}

TEST_CASE("Legacy storage: a finished copy does not run again",
          "[storage_migration]") {
  Scratch s;
  const fs::path legacy = s.root / "legacy", root = s.root / "root";
  Write(legacy / "xenia-canary.config.toml", "old config");
  REQUIRE(CopyLegacyStorage(legacy, root).complete);
  Write(legacy / "late", "added later");
  fs::remove(root / "xenia-canary.config.toml");
  MigrationReport report = CopyLegacyStorage(legacy, root);
  REQUIRE(report.complete);
  REQUIRE(report.transferred == 0);
  REQUIRE_FALSE(fs::exists(root / "late"));
  REQUIRE_FALSE(fs::exists(root / "xenia-canary.config.toml"));
}

TEST_CASE("Saves: an existing empty save folder still receives the saves",
          "[storage_migration]") {
  Scratch s;
  const fs::path content = s.root / "content", saves = s.root / "Saves";
  const fs::path title = content / "E000" / "4D5307E6";
  Write(title / "00000001" / "slot", "save");
  Write(title / "Headers" / "00000001" / "slot.header", "header");
  Write(title / "00000002" / "dlc", "dlc");
  fs::create_directories(saves);
  MigrationReport report = MoveSavesToDefaultFolder(content, saves);
  REQUIRE(report.failures.empty());
  REQUIRE(report.complete);
  REQUIRE(Read(saves / "E000" / "4D5307E6" / "00000001" / "slot") == "save");
  REQUIRE(Read(saves / "E000" / "4D5307E6" / "Headers" / "00000001" /
               "slot.header") == "header");
  REQUIRE_FALSE(fs::exists(title / "00000001"));
  // Other content types stay where they are.
  REQUIRE(Read(title / "00000002" / "dlc") == "dlc");
}

TEST_CASE("Saves: an interrupted move resumes and keeps both sides intact",
          "[storage_migration]") {
  Scratch s;
  const fs::path content = s.root / "content", saves = s.root / "Saves";
  const fs::path from = content / "E000" / "4D5307E6" / "00000001";
  const fs::path to = saves / "E000" / "4D5307E6" / "00000001";
  // An earlier cross-volume move copied slot1 and died before removing it.
  Write(from / "slot1" / "data", "one");
  Write(from / "slot2" / "data", "two");
  Write(to / "slot1" / "data", "one");
  MigrationReport report = MoveSavesToDefaultFolder(content, saves);
  REQUIRE(report.failures.empty());
  REQUIRE(report.conflicts.empty());
  REQUIRE(report.complete);
  REQUIRE(Read(to / "slot1" / "data") == "one");
  REQUIRE(Read(to / "slot2" / "data") == "two");
  REQUIRE_FALSE(fs::exists(from));
}

TEST_CASE("Saves: a different file at the destination is never overwritten",
          "[storage_migration]") {
  Scratch s;
  const fs::path content = s.root / "content", saves = s.root / "Saves";
  const fs::path from = content / "E000" / "4D5307E6" / "00000001";
  const fs::path to = saves / "E000" / "4D5307E6" / "00000001";
  Write(from / "slot" / "data", "older");
  Write(from / "other" / "data", "other");
  Write(to / "slot" / "data", "newer");
  MigrationReport report = MoveSavesToDefaultFolder(content, saves);
  REQUIRE(report.failures.empty());
  REQUIRE(report.conflicts.size() == 1);
  REQUIRE_FALSE(report.complete);
  REQUIRE(Read(to / "slot" / "data") == "newer");
  REQUIRE(Read(from / "slot" / "data") == "older");
  REQUIRE(Read(to / "other" / "data") == "other");
  REQUIRE_FALSE(fs::exists(from / "other"));
}

TEST_CASE("Saves: a move to another volume copies, then removes",
          "[storage_migration]") {
  Scratch s;
  const fs::path content = s.root / "content", saves = s.root / "Saves";
  const fs::path from = content / "E000" / "4D5307E6" / "00000001";
  Write(from / "slot" / "data", "save");
  uint32_t renames = 0;
  RenameFunction cross_volume = [&](const fs::path&, const fs::path&,
                                    std::error_code& ec) {
    ++renames;
    ec = std::make_error_code(std::errc::cross_device_link);
  };
  MigrationReport report =
      MoveSavesToDefaultFolder(content, saves, cross_volume);
  REQUIRE(renames == 1);
  REQUIRE(report.failures.empty());
  REQUIRE(report.complete);
  REQUIRE(Read(saves / "E000" / "4D5307E6" / "00000001" / "slot" / "data") ==
          "save");
  REQUIRE_FALSE(fs::exists(from));
  REQUIRE_FALSE(
      fs::exists(saves / "E000" / "4D5307E6" / "00000001.copying"));
}

TEST_CASE("Saves: a destination that cannot be written fails and keeps them",
          "[storage_migration]") {
  if (geteuid() == 0) {
    SUCCEED("permissions do not apply to root");
    return;
  }
  Scratch s;
  const fs::path content = s.root / "content", saves = s.root / "Saves";
  const fs::path from = content / "E000" / "4D5307E6" / "00000001";
  Write(from / "slot" / "data", "save");
  fs::create_directories(saves);
  REQUIRE(chmod(saves.c_str(), 0555) == 0);
  MigrationReport report = MoveSavesToDefaultFolder(content, saves);
  REQUIRE_FALSE(report.failures.empty());
  REQUIRE_FALSE(report.complete);
  REQUIRE(Read(from / "slot" / "data") == "save");
  REQUIRE(chmod(saves.c_str(), 0755) == 0);
  report = MoveSavesToDefaultFolder(content, saves);
  REQUIRE(report.complete);
  REQUIRE(Read(saves / "E000" / "4D5307E6" / "00000001" / "slot" / "data") ==
          "save");
}

}  // namespace test
}  // namespace app
}  // namespace xe
