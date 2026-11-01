/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_STORAGE_MIGRATION_MAC_H_
#define XENIA_APP_STORAGE_MIGRATION_MAC_H_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <system_error>
#include <vector>

namespace xe {
namespace app {

// What a migration did. Nothing is ever overwritten: an entry already at the
// destination is kept, and a source is only removed once the destination
// holds the same bytes.
struct MigrationReport {
  // Entries copied or moved into the destination.
  size_t transferred = 0;
  // Destination entries kept as they were (already there).
  size_t kept = 0;
  // Sources left in place because a different file of the same name is at
  // the destination (saves only).
  std::vector<std::filesystem::path> conflicts;
  // Everything that failed, one line each; empty when the migration finished.
  std::vector<std::string> failures;
  // The migration is finished (the legacy copy wrote its completion marker;
  // the save move found nothing left to move).
  bool complete = false;
};

// Test hook: the rename the save move tries before copying (a cross-volume
// rename fails, and the move then copies and removes).
using RenameFunction = std::function<void(const std::filesystem::path& from,
                                          const std::filesystem::path& to,
                                          std::error_code& ec)>;

// The storage root used to be ~/.local/share/Xenia, which Finder hides. Until
// a completed copy has written `.copied-from-legacy` into `storage_root`, the
// old folder is merged into it on every launch: entries missing at the
// destination are copied (APFS clones, through a temporary name so an
// interrupted copy never leaves a half-copied entry), folders present on both
// sides are merged, and files already at the destination are kept. The old
// folder is left untouched as a backup.
MigrationReport CopyLegacyStorage(const std::filesystem::path& legacy,
                                  const std::filesystem::path& storage_root);

// Saved games default to their own folder. The saves still under the content
// root (<xuid>/<title>/00000001 and Headers/00000001) move into it, keeping
// the same layout, on every launch that finds some: an interrupted or partly
// failed move resumes, and an existing (even empty) save folder is merged
// into rather than taken as "already moved".
MigrationReport MoveSavesToDefaultFolder(
    const std::filesystem::path& content_root,
    const std::filesystem::path& save_root,
    const RenameFunction& rename = nullptr);

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_STORAGE_MIGRATION_MAC_H_
