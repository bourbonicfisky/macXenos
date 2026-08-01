/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-depth-semantics 4.2 (L0): the PRODUCTION d2c classifier
// (metal_d2c_classifier.h) — the reviewer-specified case list: same-frame
// rule extraction, key mismatch, disjoint and partial ranges (split
// verdicts), intervening ownership changes (supersession), frame-boundary
// survival (cross-frame candidates measured but inert by default), and
// clear/reset invalidation. Identity is RenderTargetKey values throughout —
// pointer recycling cannot occur by construction.

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_d2c_classifier.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kColor = 0x1111;
constexpr uint32_t kDepth = 0x2222;
constexpr uint32_t kOtherColor = 0x3333;
constexpr uint32_t kOtherDepth = 0x4444;

uint32_t RealTiles(const std::vector<MetalD2cSegment>& segments) {
  uint32_t n = 0;
  for (const auto& s : segments) {
    if (s.verdict == MetalD2cVerdict::kRealTransfer) {
      n += s.end_tiles - s.start_tiles;
    }
  }
  return n;
}

TEST_CASE("same-frame rule: exact return classifies real, whole range",
          "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, /*swap=*/7, /*seq=*/1);
  auto segments = lineage.Classify(kDepth, kColor, 100, 120, 7, false);
  REQUIRE(segments.size() == 1);
  CHECK(segments[0].verdict == MetalD2cVerdict::kRealTransfer);
  CHECK(segments[0].start_tiles == 100);
  CHECK(segments[0].end_tiles == 120);
}

TEST_CASE("key mismatch never classifies real", "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, 7, 1);
  // Wrong color destination.
  CHECK(RealTiles(lineage.Classify(kDepth, kOtherColor, 100, 120, 7,
                                   false)) == 0);
  // Wrong depth source.
  CHECK(RealTiles(lineage.Classify(kOtherDepth, kColor, 100, 120, 7,
                                   false)) == 0);
}

TEST_CASE("disjoint and partial ranges split the verdict",
          "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, 7, 1);
  SECTION("disjoint: all persistence") {
    CHECK(RealTiles(lineage.Classify(kDepth, kColor, 120, 140, 7, false)) ==
          0);
  }
  SECTION("partial: only the proven intersection is real") {
    auto segments = lineage.Classify(kDepth, kColor, 110, 140, 7, false);
    REQUIRE(segments.size() == 2);
    CHECK(segments[0].start_tiles == 110);
    CHECK(segments[0].end_tiles == 120);
    CHECK(segments[0].verdict == MetalD2cVerdict::kRealTransfer);
    CHECK(segments[1].start_tiles == 120);
    CHECK(segments[1].end_tiles == 140);
    CHECK(segments[1].verdict == MetalD2cVerdict::kKeyLocalPersistence);
  }
  SECTION("interior: persistence on both flanks") {
    auto segments = lineage.Classify(kDepth, kColor, 90, 130, 7, false);
    REQUIRE(segments.size() == 3);
    CHECK(segments[0].verdict == MetalD2cVerdict::kKeyLocalPersistence);
    CHECK(segments[1].verdict == MetalD2cVerdict::kRealTransfer);
    CHECK(segments[1].start_tiles == 100);
    CHECK(segments[1].end_tiles == 120);
    CHECK(segments[2].verdict == MetalD2cVerdict::kKeyLocalPersistence);
  }
}

TEST_CASE("intervening ownership changes supersede the overlap only",
          "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, 7, 1);
  // A later conflicting transfer claims the middle of the lent range.
  lineage.SupersedeRange(105, 110);
  auto segments = lineage.Classify(kDepth, kColor, 100, 120, 7, false);
  CHECK(RealTiles(segments) == (105 - 100) + (120 - 110));
  // A later color->depth from ANOTHER color key supersedes its overlap too.
  lineage.RecordColorToDepth(kOtherColor, kDepth, 115, 120, 7, 2);
  segments = lineage.Classify(kDepth, kColor, 100, 120, 7, false);
  CHECK(RealTiles(segments) == (105 - 100) + (115 - 110));
  // ...and the OTHER color key's return proves exactly its own lend.
  CHECK(RealTiles(lineage.Classify(kDepth, kOtherColor, 100, 120, 7,
                                   false)) == 5);
}

TEST_CASE("frame boundary: survival is measured, not acted on",
          "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, /*swap=*/7, 1);
  // Next frame: same-frame rule refuses; the record survived unsuperseded.
  CHECK(RealTiles(lineage.Classify(kDepth, kColor, 100, 120, /*swap=*/8,
                                   false)) == 0);
  // Cross-frame flag would accept — this is the MEASURED candidate.
  CHECK(RealTiles(lineage.Classify(kDepth, kColor, 100, 120, 8, true)) == 20);
  CHECK(lineage.CountCrossFrameCandidateTiles(kDepth, kColor, 100, 120, 8) ==
        20);
  // Same-frame queries measure zero candidates (same-frame already real).
  CHECK(lineage.CountCrossFrameCandidateTiles(kDepth, kColor, 100, 120, 7) ==
        0);
}

TEST_CASE("clear/reset invalidation empties lineage",
          "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, 7, 1);
  lineage.RecordColorToDepth(kOtherColor, kOtherDepth, 300, 320, 7, 2);
  REQUIRE(lineage.record_count() == 2);
  lineage.InvalidateAll();
  CHECK(lineage.record_count() == 0);
  CHECK(RealTiles(lineage.Classify(kDepth, kColor, 100, 120, 7, false)) == 0);
  CHECK(RealTiles(lineage.Classify(kDepth, kColor, 100, 120, 7, true)) == 0);
}

TEST_CASE("re-lend after supersession restores provability",
          "[depth_semantics][d2c]") {
  MetalD2cLineage lineage;
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, 7, 1);
  lineage.SupersedeRange(100, 120);
  CHECK(RealTiles(lineage.Classify(kDepth, kColor, 100, 120, 7, false)) == 0);
  lineage.RecordColorToDepth(kColor, kDepth, 100, 120, 7, 3);
  CHECK(RealTiles(lineage.Classify(kDepth, kColor, 100, 120, 7, false)) == 20);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
