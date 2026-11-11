/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2023 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "xenia/base/math.h"
#include "xenia/vfs/devices/host_path_device.h"
#include "xenia/vfs/devices/stfs_xbox.h"
#include "xenia/vfs/devices/xcontent_container_device.h"
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

TEST_CASE("Host path entry moves into the directory it is renamed into",
          "[vfs_rename]") {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("xenia-vfs-rename-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(dir / "pack");
  std::ofstream(dir / "temp.bin") << "data";

  {
    HostPathDevice device("\\Device\\Test", dir, false);
    REQUIRE(device.Initialize());
    Entry* entry = device.ResolvePath("temp.bin");
    REQUIRE(entry != nullptr);

    SECTION("into an existing directory") {
      entry->Rename("cache:\\pack\\item.arc");

      REQUIRE(device.ResolvePath("pack\\item.arc") == entry);
      REQUIRE(entry->parent() == device.ResolvePath("pack"));
      REQUIRE(entry->path() == "pack\\item.arc");
      REQUIRE(device.ResolvePath("item.arc") == nullptr);
      REQUIRE(device.ResolvePath("temp.bin") == nullptr);
      REQUIRE(std::filesystem::exists(dir / "pack" / "item.arc"));
    }

    SECTION("into a missing directory") {
      entry->Rename("cache:\\missing\\item.arc");

      REQUIRE(device.ResolvePath("temp.bin") == entry);
      REQUIRE(entry->name() == "temp.bin");
      REQUIRE(std::filesystem::exists(dir / "temp.bin"));
    }
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("Host path rename checks the target name", "[vfs_rename]") {
  struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xenia-vfs-rename-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } temporary;
  REQUIRE(std::filesystem::create_directories(temporary.path / "dir"));
  std::ofstream(temporary.path / "new.bin") << "new";
  std::ofstream(temporary.path / "old.bin") << "old";

  HostPathDevice device("test:", temporary.path, false);
  REQUIRE(device.Initialize());
  Entry* entry = device.ResolvePath("new.bin");
  Entry* old_entry = device.ResolvePath("old.bin");
  REQUIRE(entry);
  REQUIRE(old_entry);
  auto read = [&](const char* name) {
    std::string text;
    std::ifstream(temporary.path / name) >> text;
    return text;
  };

  SECTION("A missing directory is reported") {
    REQUIRE(entry->Rename("cache:\\missing\\item.bin") ==
            X_STATUS_OBJECT_PATH_NOT_FOUND);
  }

  SECTION("An existing file is kept without replace") {
    REQUIRE(entry->Rename("cache:\\OLD.BIN") == X_STATUS_OBJECT_NAME_COLLISION);
    REQUIRE(device.ResolvePath("old.bin") == old_entry);
    REQUIRE(device.ResolvePath("new.bin") == entry);
    REQUIRE(read("old.bin") == "old");
  }

  SECTION("An existing file is replaced with replace") {
    REQUIRE(entry->Rename("cache:\\old.bin", true) == X_STATUS_SUCCESS);
    REQUIRE(device.ResolvePath("old.bin") == entry);
    REQUIRE(device.ResolvePath("new.bin") == nullptr);
    REQUIRE(device.ResolvePath("")->child_count() == 2);
    REQUIRE(read("old.bin") == "new");
  }

  SECTION("An open file is not replaced") {
    File* file = nullptr;
    REQUIRE(old_entry->Open(FileAccess::kGenericRead, &file) ==
            X_STATUS_SUCCESS);
    REQUIRE(entry->Rename("cache:\\old.bin", true) == X_STATUS_ACCESS_DENIED);
    file->Destroy();
    REQUIRE(device.ResolvePath("old.bin") == old_entry);
    REQUIRE(read("old.bin") == "old");
  }

  SECTION("A directory is not replaced") {
    REQUIRE(entry->Rename("cache:\\dir", true) ==
            X_STATUS_OBJECT_NAME_COLLISION);
    REQUIRE(std::filesystem::is_directory(temporary.path / "dir"));
  }

  SECTION("Its own name is refused") {
    REQUIRE(entry->Rename("cache:\\NEW.BIN") == X_STATUS_OBJECT_NAME_COLLISION);
    REQUIRE(entry->Rename("cache:\\NEW.BIN", true) == X_STATUS_ACCESS_DENIED);
    REQUIRE(entry->name() == "new.bin");
  }
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

constexpr size_t kStfsBlockSize = 0x1000;

// Where data block |block| of a read-only STFS package below 28900 blocks
// lies, past the level 0 tables and the level 1 table that precede it.
static size_t StfsBlockOffset(size_t block) {
  const size_t base =
      xe::round_up(sizeof(XContentContainerHeader), kStfsBlockSize);
  return base +
         (block + block / 170 + 1 + (block >= 170 ? 1 : 0)) * kStfsBlockSize;
}

struct StfsTestPackage {
  std::vector<uint8_t> bytes;
  // Contiguous in blocks 1 to 172, across two hash tables.
  std::string icon;
  // In blocks 174 then 173.
  std::string data;
};

static StfsTestPackage MakeStfsPackage() {
  auto fill = [](size_t size, uint8_t seed) {
    std::string text(size, '\0');
    for (size_t i = 0; i < size; ++i) {
      text[i] = char(seed + i * 7 + i / kStfsBlockSize);
    }
    return text;
  };
  StfsTestPackage package;
  package.icon = fill(171 * kStfsBlockSize + 100, 1);
  package.data = fill(kStfsBlockSize + 50, 200);
  package.bytes.resize(StfsBlockOffset(175));
  auto write = [&](size_t offset, const void* data, size_t size) {
    std::memcpy(package.bytes.data() + offset, data, size);
  };

  auto header = std::make_unique<XContentContainerHeader>();
  std::memset(header.get(), 0, sizeof(XContentContainerHeader));
  header->content_header.magic = XContentPackageType::kPirs;
  header->content_header.header_size = sizeof(XContentContainerHeader);
  header->content_metadata.volume_type = XContentVolumeType::kStfs;
  auto& stfs = header->content_metadata.volume_descriptor.stfs;
  stfs.descriptor_length = sizeof(StfsVolumeDescriptor);
  stfs.flags.bits.read_only_format = 1;
  stfs.file_table_block_count = 1;
  stfs.set_file_table_block_number(0);
  stfs.total_block_count = 175;
  write(0, header.get(), sizeof(XContentContainerHeader));

  // The contiguous file's hash entries stay empty, it is read without them.
  StfsHashTable hashes;
  std::memset(&hashes, 0, sizeof(hashes));
  hashes.entries[0].set_level0_next_block(0xFFFFFF);
  write(StfsBlockOffset(0) - kStfsBlockSize, &hashes, sizeof(hashes));
  std::memset(&hashes, 0, sizeof(hashes));
  hashes.entries[174 % 170].set_level0_next_block(173);
  hashes.entries[173 % 170].set_level0_next_block(0xFFFFFF);
  write(StfsBlockOffset(170) - kStfsBlockSize, &hashes, sizeof(hashes));

  StfsDirectoryBlock directory;
  std::memset(&directory, 0, sizeof(directory));
  auto add = [&](size_t index, const char* name, bool contiguous,
                 uint32_t start_block, const std::string& contents) {
    auto& entry = directory.entries[index];
    std::memcpy(entry.name, name, std::strlen(name));
    entry.flags.name_length = uint8_t(std::strlen(name));
    entry.flags.contiguous = contiguous;
    const uint32_t blocks = uint32_t(
        xe::round_up(contents.size(), kStfsBlockSize) / kStfsBlockSize);
    entry.set_valid_data_blocks(blocks);
    entry.set_allocated_data_blocks(blocks);
    entry.set_start_block_number(start_block);
    entry.directory_index = 0xFFFF;
    entry.length = uint32_t(contents.size());
  };
  add(0, "icon.png", true, 1, package.icon);
  add(1, "data.bin", false, 174, package.data);
  write(StfsBlockOffset(0), &directory, sizeof(directory));

  for (size_t i = 0; i * kStfsBlockSize < package.icon.size(); ++i) {
    const size_t start = i * kStfsBlockSize;
    write(StfsBlockOffset(1 + i), package.icon.data() + start,
          std::min(kStfsBlockSize, package.icon.size() - start));
  }
  write(StfsBlockOffset(174), package.data.data(), kStfsBlockSize);
  write(StfsBlockOffset(173), package.data.data() + kStfsBlockSize,
        package.data.size() - kStfsBlockSize);
  return package;
}

TEST_CASE("STFS package mounted from its host file or another device",
          "[stfs]") {
  struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xenia-vfs-stfs-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } temporary;
  REQUIRE(std::filesystem::create_directories(temporary.path / "dir"));
  const auto package = MakeStfsPackage();
  auto write = [&](const char* name, size_t size) {
    std::ofstream(temporary.path / name, std::ios::binary)
        .write(reinterpret_cast<const char*>(package.bytes.data()), size);
  };
  write("package", package.bytes.size());
  // Ends before the file table block.
  write("short", StfsBlockOffset(0));

  auto read = [](Device& device, const char* name) {
    Entry* entry = device.ResolvePath(name);
    REQUIRE(entry);
    File* file = nullptr;
    REQUIRE(entry->Open(FileAccess::kFileReadData, &file) == X_STATUS_SUCCESS);
    std::string text(entry->size(), '\0');
    size_t bytes_read = 0;
    REQUIRE(
        file->ReadSync({reinterpret_cast<uint8_t*>(text.data()), text.size()},
                       0, &bytes_read) == X_STATUS_SUCCESS);
    file->Destroy();
    REQUIRE(bytes_read == text.size());
    return text;
  };

  SECTION("From its host file") {
    auto device = XContentContainerDevice::CreateContentDevice(
        "\\Device\\Package", temporary.path / "package");
    REQUIRE(device);
    REQUIRE(device->Initialize());
    REQUIRE(read(*device, "icon.png") == package.icon);
    REQUIRE(read(*device, "data.bin") == package.data);
  }

  SECTION("A host file that ends early does not mount") {
    auto device = XContentContainerDevice::CreateContentDevice(
        "\\Device\\Package", temporary.path / "short");
    REQUIRE(device);
    REQUIRE_FALSE(device->Initialize());
  }

  HostPathDevice disc("\\Device\\Disc", temporary.path, true);
  REQUIRE(disc.Initialize());

  SECTION("From a file on another device") {
    auto device = XContentContainerDevice::CreateContentDevice(
        "\\Device\\Package", disc.ResolvePath("package"));
    REQUIRE(device);
    REQUIRE(device->Initialize());
    REQUIRE(read(*device, "icon.png") == package.icon);
    REQUIRE(read(*device, "data.bin") == package.data);
  }

  SECTION("A file on another device that ends early does not mount") {
    auto device = XContentContainerDevice::CreateContentDevice(
        "\\Device\\Package", disc.ResolvePath("short"));
    REQUIRE(device);
    REQUIRE_FALSE(device->Initialize());
  }

  SECTION("A directory on another device is not a package") {
    REQUIRE_FALSE(XContentContainerDevice::CreateContentDevice(
        "\\Device\\Package", disc.ResolvePath("dir")));
  }
}

}  // namespace xe::vfs::test
