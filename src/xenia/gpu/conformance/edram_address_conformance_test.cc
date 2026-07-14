/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 EDRAM tile/sample addressing oracles. These are the contracts the
// sample-accurate EDRAM redesign is built on ("every physical sample remains
// independently addressable across 1x/2x/4x") and the source of cross-title
// MSAA/64bpp incompatibilities. The CPU oracle here (XeEdramSnapshotOffsetBytes)
// mirrors the GPU-side edram.xesli XeEdramOffsetBytes; keeping them in lockstep
// is what a future Level-1 Metal-kernel test asserts.

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/trace_analysis_decode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

using xenos::MsaaSamples;

// The project's own comprehensive decode bijection/collision self-test.
TEST_CASE("EDRAM decode self-test (built-in bijection check)", "[l0][edram]") {
  REQUIRE(RunEdramDecodeSelfTest());
}

// XeEdramPitchTiles: an 80-sample-wide tile; width in samples doubles at 4x.
TEST_CASE("EDRAM pitch in tiles", "[l0][edram]") {
  REQUIRE(XeEdramPitchTiles(1280u, uint32_t(MsaaSamples::k1X)) == 16u);
  REQUIRE(XeEdramPitchTiles(1280u, uint32_t(MsaaSamples::k2X)) == 16u);  // y only
  REQUIRE(XeEdramPitchTiles(1280u, uint32_t(MsaaSamples::k4X)) == 32u);  // x doubles
  REQUIRE(XeEdramPitchTiles(1u, uint32_t(MsaaSamples::k1X)) == 1u);      // ceil
}

// The core contract: over a surface, every (pixel, physical sample) maps to a
// DISTINCT EDRAM byte and stays in the buffer. If this aliases, a 1x->4x
// ownership transfer or a sample-select resolve reads someone else's sample —
// exactly the "collapsed sample" class the redesign must eliminate.
static void CheckSampleBijection(MsaaSamples msaa, uint32_t num_samples,
                                 bool is_depth, uint32_t format_ints_log2) {
  const uint32_t w = 64, h = 32;  // small surface, several tiles
  // NOTE: XeEdramPitchTiles bakes in the 80-sample (32bpp) tile width and must
  // NOT be paired with 64bpp addressing, where the tile is 40 samples wide.
  // Derive the pitch from the real tile width so the tile-x index the offset
  // fn computes (sx / tile_w) stays inside the row pitch. Pairing the ÷80
  // helper with format_ints_log2=1 aliases across tile rows — the ladder
  // caught exactly that, so keep this derivation honest.
  const uint32_t tile_w_samples = 80u >> format_ints_log2;
  const uint32_t width_samples =
      w << (uint32_t(msaa) >= uint32_t(MsaaSamples::k4X) ? 1u : 0u);
  const uint32_t pitch_tiles =
      (width_samples + tile_w_samples - 1u) / tile_w_samples;
  std::vector<uint8_t> hit(size_t(xenos::kEdramSizeBytes), 0u);
  const uint32_t texel_bytes = 4u << format_ints_log2;  // 32bpp=4, 64bpp=8
  uint32_t placed = 0;
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      for (uint32_t s = 0; s < num_samples; ++s) {
        uint32_t off = XeEdramSnapshotOffsetBytes(
            x, y, /*base_tiles=*/0u, pitch_tiles, uint32_t(msaa), is_depth,
            format_ints_log2, s);
        REQUIRE(size_t(off) + texel_bytes <= hit.size());
        for (uint32_t b = 0; b < texel_bytes; ++b) {
          // No sample may land on a byte another sample already owns.
          REQUIRE(hit[size_t(off) + b] == 0u);
          hit[size_t(off) + b] = 1u;
        }
        ++placed;
      }
    }
  }
  REQUIRE(placed == w * h * num_samples);
}

TEST_CASE("EDRAM sample addressing is injective (1x/2x/4x, 32bpp color)",
          "[l0][edram][msaa]") {
  CheckSampleBijection(MsaaSamples::k1X, 1u, false, 0u);
  CheckSampleBijection(MsaaSamples::k2X, 2u, false, 0u);
  CheckSampleBijection(MsaaSamples::k4X, 4u, false, 0u);
}

TEST_CASE("EDRAM sample addressing is injective (depth + 64bpp)",
          "[l0][edram][msaa]") {
  // Depth applies the 40-column half-tile swap; still must be a bijection.
  CheckSampleBijection(MsaaSamples::k4X, 4u, true, 0u);
  // 64bpp halves the tile width (40 samples); 8 bytes/texel.
  CheckSampleBijection(MsaaSamples::k4X, 4u, false, 1u);
}

// The 40-column depth swap: within a 32bpp tile (80 samples wide), depth storage
// swaps the [0,40) and [40,80) column halves vs color. Pin that exact behavior.
TEST_CASE("EDRAM 40-column depth half-tile swap", "[l0][edram]") {
  const uint32_t pitch = XeEdramPitchTiles(80u, uint32_t(MsaaSamples::k1X));
  // Column 0 as depth lands where column 40 lands as color, and vice versa.
  uint32_t color_c0 = XeEdramSnapshotOffsetBytes(0, 0, 0, pitch, 0, false, 0, 0);
  uint32_t depth_c0 = XeEdramSnapshotOffsetBytes(0, 0, 0, pitch, 0, true, 0, 0);
  uint32_t color_c40 =
      XeEdramSnapshotOffsetBytes(40, 0, 0, pitch, 0, false, 0, 0);
  REQUIRE(depth_c0 == color_c40);
  REQUIRE(color_c0 != depth_c0);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
