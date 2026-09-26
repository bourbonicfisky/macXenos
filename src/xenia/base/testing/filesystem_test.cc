/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/filesystem.h"

namespace xe::base::test {

namespace {

// A scratch file with known contents, deleted when the fixture goes away.
class ScratchFile {
 public:
  ScratchFile() {
    path_ = std::filesystem::temp_directory_path() /
            ("xenia_filesystem_test_" + std::to_string(counter_++) + ".bin");
    std::error_code ec;
    std::filesystem::remove(path_, ec);
    REQUIRE(filesystem::CreateEmptyFile(path_));
    auto handle = filesystem::FileHandle::OpenExisting(
        path_, filesystem::FileAccess::kGenericWrite);
    REQUIRE(handle);
    size_t written = 0;
    REQUIRE(handle->Write(0, kInitial, sizeof(kInitial), &written));
    REQUIRE(written == sizeof(kInitial));
  }
  ~ScratchFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  const std::filesystem::path& path() const { return path_; }

  static constexpr uint8_t kInitial[8] = {'i', 'n', 'i', 't', 'i', 'a', 'l', 0};

 private:
  static inline int counter_ = 0;
  std::filesystem::path path_;
};

std::vector<uint8_t> ReadAll(const std::filesystem::path& path) {
  auto handle = filesystem::FileHandle::OpenExisting(
      path, filesystem::FileAccess::kGenericRead);
  REQUIRE(handle);
  std::vector<uint8_t> bytes(64);
  size_t read = 0;
  REQUIRE(handle->Read(0, bytes.data(), bytes.size(), &read));
  bytes.resize(read);
  return bytes;
}

}  // namespace

// Guest titles open save files with GENERIC_READ | GENERIC_WRITE and expect a
// handle that both reads and writes (Viva Piñata garden saves were persisted
// as zero-byte files when this collapsed to a write-only descriptor).
TEST_CASE("FileHandle read-write access reads and writes", "[filesystem]") {
  ScratchFile file;
  auto handle = filesystem::FileHandle::OpenExisting(
      file.path(), filesystem::FileAccess::kGenericRead |
                       filesystem::FileAccess::kGenericWrite);
  REQUIRE(handle);

  uint8_t buffer[8] = {};
  size_t read = 0;
  REQUIRE(handle->Read(0, buffer, sizeof(buffer), &read));
  REQUIRE(read == sizeof(ScratchFile::kInitial));
  REQUIRE(std::memcmp(buffer, ScratchFile::kInitial, read) == 0);

  const uint8_t payload[4] = {'s', 'a', 'v', 'e'};
  size_t written = 0;
  REQUIRE(handle->Write(2, payload, sizeof(payload), &written));
  REQUIRE(written == sizeof(payload));

  auto bytes = ReadAll(file.path());
  REQUIRE(bytes.size() == 8);
  REQUIRE(std::memcmp(bytes.data(), "insavel", 7) == 0);
}

TEST_CASE("FileHandle file-data flags behave like the generic ones",
          "[filesystem]") {
  ScratchFile file;
  auto handle = filesystem::FileHandle::OpenExisting(
      file.path(), filesystem::FileAccess::kFileReadData |
                       filesystem::FileAccess::kFileWriteData);
  REQUIRE(handle);
  uint8_t buffer[8] = {};
  size_t read = 0;
  REQUIRE(handle->Read(0, buffer, sizeof(buffer), &read));
  REQUIRE(read == sizeof(ScratchFile::kInitial));
  const uint8_t payload[1] = {'X'};
  size_t written = 0;
  REQUIRE(handle->Write(0, payload, 1, &written));
  REQUIRE(written == 1);
  REQUIRE(ReadAll(file.path())[0] == 'X');
}

TEST_CASE("FileHandle GENERIC_ALL combined with other access opens",
          "[filesystem]") {
  ScratchFile file;
  auto handle = filesystem::FileHandle::OpenExisting(
      file.path(), filesystem::FileAccess::kGenericAll |
                       filesystem::FileAccess::kGenericWrite |
                       filesystem::FileAccess::kGenericRead);
  REQUIRE(handle);
  uint8_t buffer[8] = {};
  size_t read = 0;
  REQUIRE(handle->Read(0, buffer, sizeof(buffer), &read));
  REQUIRE(read == sizeof(ScratchFile::kInitial));
  const uint8_t payload[1] = {'Y'};
  size_t written = 0;
  REQUIRE(handle->Write(7, payload, 1, &written));
  REQUIRE(written == 1);
  REQUIRE(ReadAll(file.path())[7] == 'Y');
}

TEST_CASE("FileHandle write-only access still writes", "[filesystem]") {
  ScratchFile file;
  auto handle = filesystem::FileHandle::OpenExisting(
      file.path(), filesystem::FileAccess::kFileWriteData);
  REQUIRE(handle);
  const uint8_t payload[2] = {'W', 'O'};
  size_t written = 0;
  REQUIRE(handle->Write(0, payload, sizeof(payload), &written));
  REQUIRE(written == sizeof(payload));
  auto bytes = ReadAll(file.path());
  REQUIRE(bytes[0] == 'W');
  REQUIRE(bytes[1] == 'O');
}

TEST_CASE("FileHandle append-only access appends", "[filesystem]") {
  ScratchFile file;
  auto handle = filesystem::FileHandle::OpenExisting(
      file.path(), filesystem::FileAccess::kFileAppendData);
  REQUIRE(handle);
  const uint8_t payload[3] = {'a', 'p', 'p'};
  size_t written = 0;
  REQUIRE(handle->Write(0, payload, sizeof(payload), &written));
  REQUIRE(written == sizeof(payload));
  auto bytes = ReadAll(file.path());
  REQUIRE(bytes.size() == sizeof(ScratchFile::kInitial) + sizeof(payload));
  REQUIRE(std::memcmp(bytes.data() + sizeof(ScratchFile::kInitial), payload,
                      sizeof(payload)) == 0);
}

TEST_CASE("FileHandle read-only access rejects writes", "[filesystem]") {
  ScratchFile file;
  auto handle = filesystem::FileHandle::OpenExisting(
      file.path(), filesystem::FileAccess::kGenericRead);
  REQUIRE(handle);
  const uint8_t payload[1] = {'Z'};
  size_t written = 0;
  REQUIRE_FALSE(handle->Write(0, payload, 1, &written));
  REQUIRE(ReadAll(file.path())[0] == 'i');
}

}  // namespace xe::base::test
