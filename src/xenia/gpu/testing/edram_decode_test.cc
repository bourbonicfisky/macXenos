/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_analysis_decode.h"

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/xenos.h"

// Tier-0 drift guard for the headless trace-analysis EDRAM decoder
// (build-headless-trace-analysis-suite). If the tile addressing drifts from
// the GPU's edram.xesli, every downstream pixel compare is silently wrong.
// NOTE: the same check runs inside `trace-dump --decode` (RunEdramDecodeSelfTest)
// because standalone Catch binaries can be unrunnable in some sandboxes; this
// file is the CI-facing copy.

namespace xe {
namespace gpu {
namespace test {

static constexpr uint32_t k1X = uint32_t(xenos::MsaaSamples::k1X);

TEST_CASE("EDRAM pitch in tiles", "[edram_decode]") {
  // 80 samples per tile wide; 1280px 1X -> 16 tiles.
  REQUIRE(XeEdramPitchTiles(1280, k1X) == 16);
  REQUIRE(XeEdramPitchTiles(80, k1X) == 1);
  REQUIRE(XeEdramPitchTiles(81, k1X) == 2);   // rounds up
  // 4x MSAA doubles width in samples.
  REQUIRE(XeEdramPitchTiles(640, uint32_t(xenos::MsaaSamples::k4X)) == 16);
}

TEST_CASE("EDRAM 32bpp offset lands on known bytes", "[edram_decode]") {
  const uint32_t pitch = 16;  // 1280px
  auto off = [&](uint32_t x, uint32_t y) {
    return XeEdramSnapshotOffsetBytes(x, y, /*base_tiles=*/0, pitch, k1X,
                                      /*is_depth=*/false,
                                      /*format_ints_log2=*/0, /*sample=*/0);
  };
  REQUIRE(off(0, 0) == 0u);
  REQUIRE(off(79, 0) == 79u * 4u);          // last sample in tile row 0
  REQUIRE(off(0, 1) == 80u * 4u);           // next scanline within the tile
  REQUIRE(off(80, 0) == 1280u * 4u);        // crosses one tile in x
  REQUIRE(off(0, 16) == pitch * 1280u * 4u);  // crosses one tile row in y
}

TEST_CASE("EDRAM offsets are bijective (round-trip self-test)",
          "[edram_decode]") {
  // Same coverage the in-tool decode runs before every capture-decode.
  REQUIRE(RunEdramDecodeSelfTest());
}

TEST_CASE("EDRAM depth half-tile swap", "[edram_decode]") {
  // Depth tiles swap 40-sample halves; x=0 maps into the upper half.
  uint32_t color = XeEdramSnapshotOffsetBytes(0, 0, 0, 16, k1X, false, 0, 0);
  uint32_t depth = XeEdramSnapshotOffsetBytes(0, 0, 0, 16, k1X, true, 0, 0);
  REQUIRE(color == 0u);
  REQUIRE(depth == 40u * 4u);  // x 0 -> 40 within the tile
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
