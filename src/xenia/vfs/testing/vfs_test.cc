/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2023 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "xenia/vfs/devices/host_path_device.h"
#include "xenia/vfs/devices/stfs_xbox.h"
#include "xenia/vfs/entry.h"
#include "xenia/vfs/file.h"

#include "third_party/catch/include/catch.hpp"

namespace xe::vfs::test {

TEST_CASE("STFS Decode date and time", "[stfs_decode]") {
  SECTION("10 June 2022 19:46:00 UTC - Decode") {
    const uint16_t date = 0x54CA;
    const uint16_t time = 0x9DBD;
    const uint64_t result = 132993639580000000;

    const uint64_t timestamp = decode_fat_timestamp(date, time);

    REQUIRE(timestamp == result);
  }
}

TEST_CASE("Host path entry found by its new name after a rename",
          "[vfs_rename]") {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("xenia-vfs-rename-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "item.tmp") << "data";

  {
    HostPathDevice device("\\Device\\Test", dir, false);
    REQUIRE(device.Initialize());
    Entry* entry = device.ResolvePath("item.tmp");
    REQUIRE(entry != nullptr);

    entry->Rename("cache:\\item.ipk");

    REQUIRE(entry->name() == "item.ipk");
    REQUIRE(device.ResolvePath("item.ipk") == entry);
    REQUIRE(device.ResolvePath("item.tmp") == nullptr);
    REQUIRE(std::filesystem::exists(dir / "item.ipk"));
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("Host deletion waits for the last open handle", "[host_file]") {
  struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xenia-vfs-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } temporary;
  REQUIRE(std::filesystem::create_directories(temporary.path / "dir"));
  std::ofstream(temporary.path / "shared.tmp") << "abc";
  std::ofstream(temporary.path / "dir" / "inner.tmp") << "abc";

  HostPathDevice device("test:", temporary.path, false);
  REQUIRE(device.Initialize());
  auto close = [](File* file) {
    if (file) {
      file->Destroy();
    }
  };
  auto open = [&](Entry* entry) {
    File* file = nullptr;
    REQUIRE(entry->Open(FileAccess::kGenericRead, &file) == X_STATUS_SUCCESS);
    return std::unique_ptr<File, decltype(close)>(file, close);
  };

  SECTION("A file is deleted once its last handle closes") {
    auto* entry = device.ResolvePath("shared.tmp");
    REQUIRE(entry);
    auto first = open(entry);
    auto second = open(entry);
    entry->SetForDeletion(true);
    REQUIRE_FALSE(entry->Delete());
    first.reset();
    REQUIRE(std::filesystem::exists(temporary.path / "shared.tmp"));
    std::array<uint8_t, 3> bytes{};
    size_t read = 0;
    REQUIRE(second->ReadSync(bytes, 0, &read) == X_STATUS_SUCCESS);
    REQUIRE(read == bytes.size());
    REQUIRE(bytes == std::array<uint8_t, 3>{'a', 'b', 'c'});
    second.reset();
    REQUIRE_FALSE(std::filesystem::exists(temporary.path / "shared.tmp"));
    REQUIRE(device.ResolvePath("shared.tmp") == nullptr);
  }

  SECTION("A directory is kept while a file inside it is open") {
    auto* directory = device.ResolvePath("dir");
    REQUIRE(directory);
    auto* inner = directory->GetChild("inner.tmp");
    REQUIRE(inner);
    auto file = open(inner);
    REQUIRE_FALSE(directory->Delete());
    REQUIRE(std::filesystem::exists(temporary.path / "dir" / "inner.tmp"));
    file.reset();
    REQUIRE(directory->Delete());
    REQUIRE_FALSE(std::filesystem::exists(temporary.path / "dir"));
    REQUIRE(device.ResolvePath("dir") == nullptr);
  }
}

}  // namespace xe::vfs::test
