/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/vfs/devices/null_device.h"
#include "xenia/vfs/devices/null_entry.h"
#include "xenia/base/filesystem.h"
#include "xenia/vfs/file.h"
#include "xenia/xbox.h"

namespace xe::vfs::test {

// The raw-partition stand-in must hand back what the guest wrote: Halo 3
// formats its cache filesystem on \Device\Harddisk0\Cache0/Cache1 and reads
// its own headers back before trusting the disk.
TEST_CASE("null device partitions remember what was written",
          "[null_device][store]") {
  NullDevice device("\\Device\\Harddisk0", {"\\Partition0", "\\Cache0"});
  REQUIRE(device.Initialize());
  Entry* entry = device.ResolvePath("\\Cache0");
  REQUIRE(entry != nullptr);
  File* file = nullptr;
  REQUIRE(entry->Open(FileAccess::kGenericRead | FileAccess::kGenericWrite | FileAccess::kFileReadData | FileAccess::kFileWriteData, &file) == X_STATUS_SUCCESS);
  REQUIRE(file != nullptr);

  std::vector<uint8_t> header(0x400, 0xA5);
  header[0] = 'X';
  size_t written = 0;
  CHECK(file->WriteSync(header, 0x800, &written) == X_STATUS_SUCCESS);
  CHECK(written == header.size());

  SECTION("read back returns the stored bytes and the full length") {
    std::vector<uint8_t> back(0x400, 0);
    size_t read = 0;
    CHECK(file->ReadSync(back, 0x800, &read) == X_STATUS_SUCCESS);
    CHECK(read == back.size());
    CHECK(std::memcmp(back.data(), header.data(), back.size()) == 0);
  }
  SECTION("never-written space reads as zeros, not garbage or a short read") {
    std::vector<uint8_t> back(0x1000, 0xFF);
    size_t read = 0;
    CHECK(file->ReadSync(back, 0x10000, &read) == X_STATUS_SUCCESS);
    CHECK(read == back.size());
    CHECK(std::count(back.begin(), back.end(), 0) == int64_t(back.size()));
  }
  SECTION("a runaway write is refused instead of eating host memory") {
    std::vector<uint8_t> big(16);
    size_t w = 0;
    CHECK(file->WriteSync(big, size_t(1) << 40, &w) == X_STATUS_DISK_FULL);
  }
  file->Destroy();
}

}  // namespace xe::vfs::test
