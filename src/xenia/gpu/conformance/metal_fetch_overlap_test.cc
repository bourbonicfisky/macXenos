/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the precise pending-writeback overlap decision.
// A fetched texture's guest storage is TWO disjoint ranges — base and mips at
// unrelated addresses. A writeback overlaps the fetch iff it intersects EITHER
// range and NEVER the gap between them; the previous conservative
// pitch*height*4 extent spanned that gap, causing false flushes. These pin the
// two-range logic; the range derivation from GetGuestTextureLayout is exercised
// by the (reverted) shadow gate and the four-trace gate.

#include "xenia/gpu/metal/metal_fetch_overlap.h"

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// base [0x1000,0x1400), mip [0x8000,0x8100); the gap 0x1400..0x8000 is NOT
// part of the texture.
FetchGuestRanges TwoRange() {
  FetchGuestRanges r;
  r.base_begin = 0x1000;
  r.base_end = 0x1400;
  r.mip_begin = 0x8000;
  r.mip_end = 0x8100;
  return r;
}

TEST_CASE("Overlap hits the base range", "[l0][metal][overlap]") {
  REQUIRE(WritebackOverlapsFetch(TwoRange(), 0x1200, 0x1300));
  REQUIRE(WritebackOverlapsFetch(TwoRange(), 0x0F00, 0x1100));  // straddles start
  REQUIRE(WritebackOverlapsFetch(TwoRange(), 0x1300, 0x1500));  // straddles end
}

TEST_CASE("Overlap hits the mip range", "[l0][metal][overlap]") {
  REQUIRE(WritebackOverlapsFetch(TwoRange(), 0x8040, 0x8080));
  REQUIRE(WritebackOverlapsFetch(TwoRange(), 0x7F00, 0x8010));
}

TEST_CASE("Overlap does NOT span the gap between base and mips",
          "[l0][metal][overlap]") {
  // A writeback entirely inside the base->mip gap must not overlap — this is
  // the exact false-flush the conservative extent produced.
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x2000, 0x3000));
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x1400, 0x8000));  // whole gap
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x7F00, 0x8000));  // up to mip
}

TEST_CASE("Overlap misses outside both ranges", "[l0][metal][overlap]") {
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x0000, 0x1000));  // before base
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x8100, 0x9000));  // after mips
}

TEST_CASE("Empty mip range is never matched", "[l0][metal][overlap]") {
  FetchGuestRanges r;
  r.base_begin = 0x1000;
  r.base_end = 0x1400;
  // mip_begin == mip_end == 0 (no mips).
  REQUIRE(WritebackOverlapsFetch(r, 0x1200, 0x1300));        // base still hits
  REQUIRE_FALSE(WritebackOverlapsFetch(r, 0x0000, 0x0000));  // degenerate wb
  REQUIRE_FALSE(WritebackOverlapsFetch(r, 0x2000, 0x3000));  // no mip range to hit
}

TEST_CASE("Half-open boundaries do not touch-overlap", "[l0][metal][overlap]") {
  // [wb) ending exactly at base_begin, or beginning exactly at base_end.
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x0800, 0x1000));
  REQUIRE_FALSE(WritebackOverlapsFetch(TwoRange(), 0x1400, 0x1800));
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
