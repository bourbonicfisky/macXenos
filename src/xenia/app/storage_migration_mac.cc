/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/storage_migration_mac.h"

#include <copyfile.h>
#include <cerrno>
#include <cstring>
#include <fstream>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/logging.h"

namespace xe {
namespace app {

namespace {

namespace fs = std::filesystem;

// Copies (APFS clones, so no extra space until something changes) a file or
// folder to `to`, through a temporary name so an interrupted copy never
// leaves a half-copied entry behind.
bool CloneEntry(const fs::path& from, const fs::path& to,
                MigrationReport& report) {
  fs::path partial = to;
  partial += ".copying";
  std::error_code ec;
  fs::remove_all(partial, ec);
  if (copyfile(from.c_str(), partial.c_str(), nullptr,
               COPYFILE_ALL | COPYFILE_RECURSIVE | COPYFILE_CLONE) != 0) {
    report.failures.push_back(fmt::format("Could not copy {} to {}: {}",
                                          from.string(), to.string(),
                                          std::strerror(errno)));
    fs::remove_all(partial, ec);
    return false;
  }
  fs::rename(partial, to, ec);
  if (ec) {
    report.failures.push_back(fmt::format("Could not rename {} to {}: {}",
                                          partial.string(), to.string(),
                                          ec.message()));
    fs::remove_all(partial, ec);
    return false;
  }
  return true;
}

// The entries of a folder, listed up front (the save move renames them out
// while it walks). False, with the failure noted, if it cannot be listed.
bool ListFolder(const fs::path& folder, std::vector<fs::path>& entries,
                MigrationReport& report) {
  entries.clear();
  std::error_code ec;
  fs::directory_iterator it(folder, ec);
  for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
    entries.push_back(it->path());
  }
  if (ec) {
    report.failures.push_back(
        fmt::format("Could not list {}: {}", folder.string(), ec.message()));
    return false;
  }
  return true;
}

// The type of `path` without following a symlink; not_found when absent,
// none (with the failure noted) when it cannot be checked.
fs::file_type EntryType(const fs::path& path, MigrationReport& report) {
  std::error_code ec;
  const fs::file_status status = fs::symlink_status(path, ec);
  if (status.type() == fs::file_type::not_found) {
    return fs::file_type::not_found;
  }
  if (ec) {
    report.failures.push_back(
        fmt::format("Could not check {}: {}", path.string(), ec.message()));
    return fs::file_type::none;
  }
  return status.type();
}

bool SameContents(const fs::path& a, const fs::path& b) {
  std::error_code ec_a, ec_b;
  const uintmax_t size = fs::file_size(a, ec_a);
  if (ec_a || size != fs::file_size(b, ec_b) || ec_b) {
    return false;
  }
  std::ifstream file_a(a, std::ios::binary), file_b(b, std::ios::binary);
  if (!file_a || !file_b) {
    return false;
  }
  char buffer_a[65536], buffer_b[65536];
  while (file_a && file_b) {
    file_a.read(buffer_a, sizeof(buffer_a));
    file_b.read(buffer_b, sizeof(buffer_b));
    if (file_a.gcount() != file_b.gcount() ||
        std::memcmp(buffer_a, buffer_b, size_t(file_a.gcount())) != 0) {
      return false;
    }
  }
  return file_a.eof() && file_b.eof();
}

// Copies what `to` lacks of `from`, recursing into folders both have; entries
// already at `to` are kept.
void MergeCopy(const fs::path& from, const fs::path& to,
               MigrationReport& report) {
  std::vector<fs::path> entries;
  if (!ListFolder(from, entries, report)) {
    return;
  }
  for (const fs::path& source : entries) {
    const fs::path destination = to / source.filename();
    const fs::file_type to_type = EntryType(destination, report);
    if (to_type == fs::file_type::none) {
      continue;
    }
    if (to_type == fs::file_type::not_found) {
      if (CloneEntry(source, destination, report)) {
        ++report.transferred;
      }
      continue;
    }
    if (to_type == fs::file_type::directory &&
        EntryType(source, report) == fs::file_type::directory) {
      MergeCopy(source, destination, report);
      continue;
    }
    ++report.kept;
  }
}

// Moves `from` to `to`, merging into what is already there: a source is only
// removed once the destination holds the same bytes.
void MoveEntry(const fs::path& from, const fs::path& to,
               const RenameFunction& rename, MigrationReport& report) {
  const fs::file_type to_type = EntryType(to, report);
  if (to_type == fs::file_type::none) {
    return;
  }
  std::error_code ec;
  if (to_type == fs::file_type::not_found) {
    fs::create_directories(to.parent_path(), ec);
    if (ec) {
      report.failures.push_back(fmt::format("Could not create {}: {}",
                                            to.parent_path().string(),
                                            ec.message()));
      return;
    }
    if (rename) {
      rename(from, to, ec);
    } else {
      fs::rename(from, to, ec);
    }
    if (!ec) {
      ++report.transferred;
      return;
    }
    // Another volume: copy, and remove the original only once copied.
    if (!CloneEntry(from, to, report)) {
      return;
    }
    ++report.transferred;
    fs::remove_all(from, ec);
    if (ec) {
      // The next launch finds the copy identical and removes this then.
      report.failures.push_back(fmt::format(
          "Copied {} but could not remove it: {}", from.string(), ec.message()));
    }
    return;
  }
  const fs::file_type from_type = EntryType(from, report);
  if (from_type == fs::file_type::directory &&
      to_type == fs::file_type::directory) {
    std::vector<fs::path> entries;
    if (!ListFolder(from, entries, report)) {
      return;
    }
    for (const fs::path& source : entries) {
      MoveEntry(source, to / source.filename(), rename, report);
    }
    // Only an emptied folder goes; anything left is reported above.
    fs::remove(from, ec);
    return;
  }
  if (from_type == fs::file_type::regular &&
      to_type == fs::file_type::regular && SameContents(from, to)) {
    // Copied by an earlier, interrupted move.
    ++report.kept;
    fs::remove(from, ec);
    if (ec) {
      report.failures.push_back(fmt::format(
          "Could not remove {} (already at {}): {}", from.string(),
          to.string(), ec.message()));
    }
    return;
  }
  report.conflicts.push_back(from);
}

}  // namespace

MigrationReport CopyLegacyStorage(const fs::path& legacy,
                                  const fs::path& storage_root) {
  MigrationReport report;
  const fs::path marker = storage_root / ".copied-from-legacy";
  std::error_code ec;
  if (!fs::is_directory(legacy, ec) || fs::exists(marker, ec)) {
    report.complete = true;
    return report;
  }
  XELOGI("Copying the old storage folder {} into {}", legacy, storage_root);
  fs::create_directories(storage_root, ec);
  if (ec) {
    report.failures.push_back(fmt::format(
        "Could not create {}: {}", storage_root.string(), ec.message()));
  } else {
    MergeCopy(legacy, storage_root, report);
  }
  if (report.failures.empty()) {
    std::ofstream out(marker);
    out << legacy.string() << "\n";
    out.close();
    if (out) {
      report.complete = true;
    } else {
      report.failures.push_back(
          fmt::format("Could not write {}", marker.string()));
    }
  }
  for (const std::string& failure : report.failures) {
    XELOGE("Storage migration: {}", failure);
  }
  if (report.complete) {
    XELOGI(
        "Old storage folder copied ({} copied, {} already there); {} is left "
        "as it was",
        report.transferred, report.kept, legacy);
  } else {
    XELOGW(
        "Old storage folder only partly copied ({} copied, {} failed); "
        "retrying next launch",
        report.transferred, report.failures.size());
  }
  return report;
}

MigrationReport MoveSavesToDefaultFolder(const fs::path& content_root,
                                         const fs::path& save_root,
                                         const RenameFunction& rename) {
  MigrationReport report;
  std::error_code ec;
  if (!fs::is_directory(content_root, ec)) {
    report.complete = true;
    return report;
  }
  const fs::path save_type = "00000001";
  std::vector<fs::path> xuids, titles;
  if (ListFolder(content_root, xuids, report)) {
    for (const fs::path& xuid : xuids) {
      if (EntryType(xuid, report) != fs::file_type::directory ||
          !ListFolder(xuid, titles, report)) {
        continue;
      }
      for (const fs::path& title : titles) {
        const fs::path relative = xuid.filename() / title.filename();
        for (const fs::path& part : {save_type, fs::path("Headers") / save_type}) {
          const fs::path from = title / part;
          if (EntryType(from, report) != fs::file_type::directory) {
            continue;
          }
          MoveEntry(from, save_root / relative / part, rename, report);
        }
      }
    }
  }
  report.complete = report.failures.empty() && report.conflicts.empty();
  for (const std::string& failure : report.failures) {
    XELOGE("Save migration: {}", failure);
  }
  for (const fs::path& conflict : report.conflicts) {
    XELOGW(
        "Save migration: left {} in place, a different file of that name is "
        "already in {}",
        conflict, save_root);
  }
  if (report.transferred || report.kept) {
    XELOGI("Moved {} save entries from {} to {} ({} already there)",
           report.transferred, content_root, save_root, report.kept);
  }
  return report;
}

}  // namespace app
}  // namespace xe
