/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the WATCH-ONLY admission contract used by
// SharedMemory::RequestWatchOnlyRange for consumers that read guest RAM
// directly (the Metal legacy texture untile).
//
// The rule under test: admit whole SYSTEM pages, and only those whose covered
// GUEST pages are ALL writable; skip unmapped holes anywhere in the extent
// (leading, trailing, interior) instead of rejecting the whole request. The
// live defect this replaces rejected the entire extent when its first or last
// page was unmapped, dropping whole textures to the null slot (300 null serves
// of one base in a live SF4 session).
//
// The system/guest page-size skew is load-bearing on Apple Silicon: system
// (== shared-memory) pages are 16 KiB while the small physical heap's guest
// pages are 4 KiB, so one system page covers four guest pages and protecting
// it traps writes to all four.

#include "xenia/gpu/watch_admission.h"

#include <cstdint>
#include <set>
#include <vector>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace {

constexpr uint32_t kGuestPageLog2 = 12;    // 4 KiB guest pages.
constexpr uint32_t kSystemPageLog2 = 14;   // 16 KiB host/system pages.
constexpr uint32_t kBufferSize = 512u << 20;

// Admits every guest page except those explicitly holed.
struct PageMap {
  std::set<uint32_t> unmapped_guest_pages;
  bool operator()(uint32_t guest_page) const {
    return unmapped_guest_pages.find(guest_page) ==
           unmapped_guest_pages.end();
  }
};

uint32_t Scan(uint32_t start, uint32_t length, const PageMap& map,
              std::vector<WatchAdmissionRun>& runs,
              uint32_t system_page_log2 = kSystemPageLog2) {
  runs.clear();
  return ComputeWatchAdmissionRuns(start, length, system_page_log2,
                                   kGuestPageLog2, kBufferSize, map, runs);
}

TEST_CASE("WatchAdmission: fully mapped extent admits one whole run",
          "[l0][watch-admission]") {
  PageMap map;
  std::vector<WatchAdmissionRun> runs;
  // 4 system pages starting system-page-aligned.
  uint32_t start = 0x10000000;
  uint32_t length = 4u << kSystemPageLog2;
  uint32_t admitted = Scan(start, length, map, runs);
  REQUIRE(runs.size() == 1);
  REQUIRE(runs[0].start == start);
  REQUIRE(runs[0].length == length);
  REQUIRE(admitted == length);
}

TEST_CASE("WatchAdmission: unmapped FIRST page no longer sinks the extent",
          "[l0][watch-admission]") {
  // The live failure mode: the whole request was rejected. Admission must
  // keep every mapped page after the hole.
  uint32_t start = 0x10000000;
  uint32_t length = 4u << kSystemPageLog2;
  PageMap map;
  map.unmapped_guest_pages.insert(start >> kGuestPageLog2);  // first guest page
  std::vector<WatchAdmissionRun> runs;
  uint32_t admitted = Scan(start, length, map, runs);
  REQUIRE(runs.size() == 1);
  // The first SYSTEM page is dropped whole (one of its guest pages is unmapped)
  // and the remaining three are admitted.
  REQUIRE(runs[0].start == start + (1u << kSystemPageLog2));
  REQUIRE(runs[0].length == 3u << kSystemPageLog2);
  REQUIRE(admitted == 3u << kSystemPageLog2);
}

TEST_CASE("WatchAdmission: unmapped LAST page keeps the leading pages",
          "[l0][watch-admission]") {
  uint32_t start = 0x10000000;
  uint32_t length = 4u << kSystemPageLog2;
  PageMap map;
  uint32_t last_byte = start + length - 1;
  map.unmapped_guest_pages.insert(last_byte >> kGuestPageLog2);
  std::vector<WatchAdmissionRun> runs;
  uint32_t admitted = Scan(start, length, map, runs);
  REQUIRE(runs.size() == 1);
  REQUIRE(runs[0].start == start);
  REQUIRE(runs[0].length == 3u << kSystemPageLog2);
  REQUIRE(admitted == 3u << kSystemPageLog2);
}

TEST_CASE("WatchAdmission: interior hole splits into two runs",
          "[l0][watch-admission]") {
  uint32_t start = 0x10000000;
  uint32_t length = 5u << kSystemPageLog2;
  PageMap map;
  // Hole inside the third system page.
  uint32_t hole_byte = start + (2u << kSystemPageLog2) + 4096;
  map.unmapped_guest_pages.insert(hole_byte >> kGuestPageLog2);
  std::vector<WatchAdmissionRun> runs;
  uint32_t admitted = Scan(start, length, map, runs);
  REQUIRE(runs.size() == 2);
  REQUIRE(runs[0].start == start);
  REQUIRE(runs[0].length == 2u << kSystemPageLog2);
  REQUIRE(runs[1].start == start + (3u << kSystemPageLog2));
  REQUIRE(runs[1].length == 2u << kSystemPageLog2);
  REQUIRE(admitted == 4u << kSystemPageLog2);
}

TEST_CASE("WatchAdmission: 16 KiB system page requires ALL four 4 KiB guest pages",
          "[l0][watch-admission]") {
  // Protection happens at system-page granularity, so a system page holding any
  // non-writable guest page must NOT be protected: the unprotect path keys off
  // the guest page table and would leave such a page trapping forever.
  uint32_t start = 0x10000000;
  uint32_t length = 1u << kSystemPageLog2;  // exactly one system page
  uint32_t first_guest_page = start >> kGuestPageLog2;
  for (uint32_t i = 0; i < 4; ++i) {
    PageMap map;
    map.unmapped_guest_pages.insert(first_guest_page + i);
    std::vector<WatchAdmissionRun> runs;
    uint32_t admitted = Scan(start, length, map, runs);
    INFO("unmapped guest page index within the system page: " << i);
    REQUIRE(runs.empty());
    REQUIRE(admitted == 0);
  }
  // All four mapped -> admitted.
  PageMap all_mapped;
  std::vector<WatchAdmissionRun> runs;
  REQUIRE(Scan(start, length, all_mapped, runs) == length);
  REQUIRE(runs.size() == 1);
}

TEST_CASE("WatchAdmission: degenerate and clamped extents",
          "[l0][watch-admission]") {
  PageMap map;
  std::vector<WatchAdmissionRun> runs;
  SECTION("zero length admits nothing") {
    REQUIRE(Scan(0x10000000, 0, map, runs) == 0);
    REQUIRE(runs.empty());
  }
  SECTION("start past the buffer admits nothing") {
    REQUIRE(Scan(kBufferSize, 4096, map, runs) == 0);
    REQUIRE(runs.empty());
  }
  SECTION("extent clamps to the buffer end") {
    uint32_t start = kBufferSize - (2u << kSystemPageLog2);
    uint32_t admitted = Scan(start, 16u << kSystemPageLog2, map, runs);
    REQUIRE(admitted == 2u << kSystemPageLog2);
    REQUIRE(runs.size() == 1);
    REQUIRE(runs[0].start == start);
  }
  SECTION("unaligned start covers its containing system page") {
    uint32_t start = 0x10000000 + 100;
    uint32_t admitted = Scan(start, 16, map, runs);
    REQUIRE(admitted == 1u << kSystemPageLog2);
    REQUIRE(runs[0].start == 0x10000000);
  }
  SECTION("every page unmapped admits nothing, and that is not an error") {
    PageMap none;
    // 4 system pages of 16 KiB span 16 guest pages of 4 KiB.
    for (uint32_t i = 0; i < 16; ++i) {
      none.unmapped_guest_pages.insert((0x10000000 >> kGuestPageLog2) + i);
    }
    REQUIRE(Scan(0x10000000, 4u << kSystemPageLog2, none, runs) == 0);
    REQUIRE(runs.empty());
  }
}

TEST_CASE("WatchAdmission: runs are ascending, disjoint and maximal",
          "[l0][watch-admission]") {
  uint32_t start = 0x10000000;
  uint32_t length = 12u << kSystemPageLog2;
  PageMap map;
  // Holes in system pages 1, 2 and 7 (0-based within the extent).
  for (uint32_t sys : {1u, 2u, 7u}) {
    map.unmapped_guest_pages.insert(
        (start + (sys << kSystemPageLog2)) >> kGuestPageLog2);
  }
  std::vector<WatchAdmissionRun> runs;
  uint32_t admitted = Scan(start, length, map, runs);
  REQUIRE(runs.size() == 3);
  uint32_t total = 0;
  for (size_t i = 0; i < runs.size(); ++i) {
    REQUIRE(runs[i].length > 0);
    REQUIRE((runs[i].start & ((1u << kSystemPageLog2) - 1)) == 0);
    REQUIRE((runs[i].length & ((1u << kSystemPageLog2) - 1)) == 0);
    total += runs[i].length;
    if (i) {
      // Ascending, disjoint, and separated by at least one skipped page
      // (otherwise they would have coalesced).
      REQUIRE(runs[i].start > runs[i - 1].start + runs[i - 1].length);
    }
  }
  REQUIRE(total == admitted);
  REQUIRE(admitted == 9u << kSystemPageLog2);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
