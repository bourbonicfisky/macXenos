/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/vfs/devices/null_file.h"

#include "xenia/vfs/devices/null_entry.h"

namespace xe {
namespace vfs {

NullFile::NullFile(uint32_t file_access, NullEntry* entry)
    : File(file_access, entry) {}

NullFile::~NullFile() = default;

void NullFile::Destroy() { delete this; }

// The null device stands in for the console's raw HDD partitions
// (\Device\Harddisk0\Partition0, Cache0, Cache1). Games' cache/STFC code
// formats a small filesystem there and reads its own headers back; a device
// that acknowledged writes but returned nothing made Halo 3 conclude the
// disk was broken ("Disc Read Error" after formatting Cache0/Cache1). The
// partition now remembers what was written: reads return the stored bytes
// (zeros beyond what was ever written), writes grow a bounded store.
// XE_NULL_DEVICE_STORE=0 restores the discard-everything behavior.
namespace {
bool NullDeviceStoreEnabled() {
  static const bool s_enabled = [] {
    const char* v = std::getenv("XE_NULL_DEVICE_STORE");
    return !v || v[0] != '0';
  }();
  return s_enabled;
}
// Geometry reported by NtDeviceIoControlFile is ~1 MiB per partition; leave
// generous headroom without letting a runaway writer eat host memory.
constexpr size_t kNullDeviceStoreCap = 64 * 1024 * 1024;
}  // namespace

X_STATUS NullFile::ReadSync(std::span<uint8_t> buffer, size_t byte_offset,
                            size_t* out_bytes_read) {
  if (!(file_access_ & FileAccess::kFileReadData)) {
    return X_STATUS_ACCESS_DENIED;
  }
  if (!NullDeviceStoreEnabled()) {
    return X_STATUS_SUCCESS;
  }
  auto& store = static_cast<NullEntry*>(entry_)->store();
  size_t n = buffer.size();
  std::memset(buffer.data(), 0, n);
  if (byte_offset < store.size()) {
    size_t avail = std::min(n, store.size() - byte_offset);
    std::memcpy(buffer.data(), store.data() + byte_offset, avail);
  }
  *out_bytes_read = n;
  return X_STATUS_SUCCESS;
}

X_STATUS NullFile::WriteSync(std::span<const uint8_t> buffer,
                             size_t byte_offset, size_t* out_bytes_written) {
  if (!(file_access_ &
        (FileAccess::kFileWriteData | FileAccess::kFileAppendData))) {
    return X_STATUS_ACCESS_DENIED;
  }
  if (!NullDeviceStoreEnabled()) {
    return X_STATUS_SUCCESS;
  }
  size_t end = byte_offset + buffer.size();
  if (end > kNullDeviceStoreCap || end < byte_offset) {
    return X_STATUS_DISK_FULL;
  }
  auto& store = static_cast<NullEntry*>(entry_)->store();
  if (store.size() < end) {
    store.resize(end, 0);
  }
  std::memcpy(store.data() + byte_offset, buffer.data(), buffer.size());
  *out_bytes_written = buffer.size();
  return X_STATUS_SUCCESS;
}

X_STATUS NullFile::SetLength(size_t length) {
  if (!(file_access_ & FileAccess::kFileWriteData)) {
    return X_STATUS_ACCESS_DENIED;
  }
  if (NullDeviceStoreEnabled() && length <= kNullDeviceStoreCap) {
    static_cast<NullEntry*>(entry_)->store().resize(length, 0);
  }
  return X_STATUS_SUCCESS;
}

}  // namespace vfs
}  // namespace xe
