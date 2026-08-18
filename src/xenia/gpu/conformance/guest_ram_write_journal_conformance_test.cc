/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                            *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free) conformance for the guest-RAM rollback journal
// (OpenSpec fix-replay-guest-ram-rollback, task 2.3): first-write capture,
// capture-once under repeated writes, disjoint and overlapping ranges, and an
// empty range set producing no work. The journal is pure interval + byte
// logic with an injected translate hook, so these tests back it with a plain
// vector standing in for guest RAM.

#include <cstdint>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/guest_ram_write_journal.h"

namespace xe {
namespace gpu {
namespace test {

// 64 KiB stand-in for guest physical RAM, filled with a position-derived
// pattern so any restore error is visible at the exact byte.
class FakeRam {
 public:
  FakeRam() : bytes_(64 * 1024) { FillPattern(); }

  void FillPattern() {
    for (size_t i = 0; i < bytes_.size(); ++i) {
      bytes_[i] = uint8_t((i * 7 + 13) & 0xFF);
    }
  }

  uint8_t PatternAt(uint32_t i) const { return uint8_t((i * 7 + 13) & 0xFF); }

  void Overwrite(uint32_t base, uint32_t length, uint8_t value) {
    std::memset(bytes_.data() + base, value, length);
  }

  bool MatchesPattern(uint32_t base, uint32_t length) const {
    for (uint32_t i = 0; i < length; ++i) {
      if (bytes_[base + i] != PatternAt(base + i)) {
        return false;
      }
    }
    return true;
  }

  uint8_t at(uint32_t i) const { return bytes_[i]; }

  GuestRamWriteJournal::TranslateFn translate() {
    return [this](uint32_t base, uint32_t length) -> uint8_t* {
      if (uint64_t(base) + length > bytes_.size()) {
        return nullptr;
      }
      return bytes_.data() + base;
    };
  }

 private:
  std::vector<uint8_t> bytes_;
};

TEST_CASE("journal: first write captures, restore reverts", "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0x100, 0x40);
  ram.Overwrite(0x100, 0x40, 0xAA);
  REQUIRE(!ram.MatchesPattern(0x100, 0x40));

  REQUIRE(journal.RestoreAll() == 1);
  REQUIRE(ram.MatchesPattern(0x100, 0x40));
  REQUIRE(journal.entry_count() == 1);
  REQUIRE(journal.captured_bytes() == 0x40);
}

TEST_CASE("journal: repeated writes to the same range capture once",
          "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0x200, 0x20);
  ram.Overwrite(0x200, 0x20, 0xAA);
  // Second write of the same range: the capture must NOT refresh to the
  // post-first-write bytes.
  journal.CaptureBeforeWrite(0x200, 0x20);
  ram.Overwrite(0x200, 0x20, 0xBB);

  REQUIRE(journal.entry_count() == 1);
  REQUIRE(journal.captured_bytes() == 0x20);
  journal.RestoreAll();
  REQUIRE(ram.MatchesPattern(0x200, 0x20));
}

TEST_CASE("journal: disjoint ranges are independent entries", "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0x100, 0x10);
  ram.Overwrite(0x100, 0x10, 0xAA);
  journal.CaptureBeforeWrite(0x800, 0x30);
  ram.Overwrite(0x800, 0x30, 0xBB);

  REQUIRE(journal.entry_count() == 2);
  REQUIRE(journal.captured_bytes() == 0x40);
  REQUIRE(journal.RestoreAll() == 2);
  REQUIRE(ram.MatchesPattern(0x100, 0x10));
  REQUIRE(ram.MatchesPattern(0x800, 0x30));
  // Untouched neighborhood stays untouched.
  REQUIRE(ram.MatchesPattern(0x0F0, 0x10));
  REQUIRE(ram.MatchesPattern(0x110, 0x10));
}

TEST_CASE("journal: overlapping capture takes only the uncovered gap",
          "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  // Capture [0x100, 0x120), write it.
  journal.CaptureBeforeWrite(0x100, 0x20);
  ram.Overwrite(0x100, 0x20, 0xAA);
  // Overlapping capture [0x110, 0x140): only [0x120, 0x140) is new. The
  // overlap [0x110, 0x120) currently holds 0xAA — if the journal recaptured
  // it, restore would deliver 0xAA instead of the pre-frame pattern.
  journal.CaptureBeforeWrite(0x110, 0x30);
  ram.Overwrite(0x110, 0x30, 0xBB);

  REQUIRE(journal.captured_bytes() == 0x40);  // 0x20 + the 0x20 gap
  journal.RestoreAll();
  REQUIRE(ram.MatchesPattern(0x100, 0x40));
}

TEST_CASE("journal: capture fully inside a covered range is a no-op",
          "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0x100, 0x40);
  ram.Overwrite(0x100, 0x40, 0xAA);
  journal.CaptureBeforeWrite(0x110, 0x10);

  REQUIRE(journal.entry_count() == 1);
  REQUIRE(journal.captured_bytes() == 0x40);
  journal.RestoreAll();
  REQUIRE(ram.MatchesPattern(0x100, 0x40));
}

TEST_CASE("journal: capture spanning two entries takes the middle gap",
          "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0x100, 0x10);  // [0x100, 0x110)
  journal.CaptureBeforeWrite(0x120, 0x10);  // [0x120, 0x130)
  ram.Overwrite(0x100, 0x10, 0xAA);
  ram.Overwrite(0x120, 0x10, 0xAA);
  // Spanning capture [0x0F0, 0x140): new coverage is the prefix, the middle
  // gap, and the suffix.
  journal.CaptureBeforeWrite(0x0F0, 0x50);
  ram.Overwrite(0x0F0, 0x50, 0xBB);

  REQUIRE(journal.captured_bytes() == 0x50);
  journal.RestoreAll();
  REQUIRE(ram.MatchesPattern(0x0F0, 0x50));
}

TEST_CASE("journal: empty range set does no work", "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  REQUIRE(journal.RestoreAll() == 0);
  REQUIRE(journal.entry_count() == 0);
  REQUIRE(journal.captured_bytes() == 0);

  journal.CaptureBeforeWrite(0x100, 0);  // zero length: no entry
  REQUIRE(journal.entry_count() == 0);
}

TEST_CASE("journal: unmappable range is counted, not fatal", "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0xFFFF0, 0x20);  // past the fake RAM end
  REQUIRE(journal.entry_count() == 0);
  REQUIRE(journal.failed_captures() == 1);
  REQUIRE(journal.RestoreAll() == 0);
}

TEST_CASE("journal: clear drops all state", "[ram-journal]") {
  FakeRam ram;
  GuestRamWriteJournal journal(ram.translate());

  journal.CaptureBeforeWrite(0x100, 0x40);
  ram.Overwrite(0x100, 0x40, 0xAA);
  journal.Clear();

  REQUIRE(journal.entry_count() == 0);
  REQUIRE(journal.RestoreAll() == 0);
  // Cleared journal must not restore: the overwrite stays.
  REQUIRE(!ram.MatchesPattern(0x100, 0x40));
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
