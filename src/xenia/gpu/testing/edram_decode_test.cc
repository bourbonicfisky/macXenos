/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_analysis_decode.h"

#include <cstring>
#include <vector>

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

TEST_CASE("EDRAM 32bpp color decode round-trips synthetic EDRAM",
          "[edram_decode]") {
  // Tier-0 D7 guard: write a unique k_8_8_8_8 value at each pixel's computed
  // offset in a synthetic EDRAM, decode, and require exact recovery -- proves
  // the decoder re-tiles precisely what the offset formula placed.
  const uint32_t w = 96, h = 40;                 // spans multiple 80x16 tiles
  const uint32_t base = 5, pitch = XeEdramPitchTiles(w, k1X);
  std::vector<uint8_t> edram(xenos::kEdramSizeBytes, 0);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint32_t off = XeEdramSnapshotOffsetBytes(x, y, base, pitch, k1X, false,
                                                0, 0);
      REQUIRE(off + 4 <= edram.size());
      // Pack (x,y) into RGBA so each pixel is unique and channel order is
      // checkable: R=x, G=y, B=x^y, A=0xFF.
      uint32_t v = uint32_t(x & 0xFF) | (uint32_t(y & 0xFF) << 8) |
                   (uint32_t((x ^ y) & 0xFF) << 16) | (0xFFu << 24);
      std::memcpy(edram.data() + off, &v, sizeof(v));
    }
  }
  DecodedImage img;
  REQUIRE(DecodeEdramColor32bpp(edram.data(), edram.size(), base, pitch, k1X,
                                xenos::ColorRenderTargetFormat::k_8_8_8_8, w, h,
                                0, &img));
  REQUIRE(img.width == w);
  REQUIRE(img.height == h);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      size_t p = (size_t(y) * w + x) * 4;
      REQUIRE(img.rgba[p + 0] == uint8_t(x));
      REQUIRE(img.rgba[p + 1] == uint8_t(y));
      REQUIRE(img.rgba[p + 2] == uint8_t(x ^ y));
      REQUIRE(img.rgba[p + 3] == 0xFF);
    }
  }
}

TEST_CASE("EDRAM depth24 channel round-trips synthetic EDRAM",
          "[edram_decode]") {
  // The compare oracle diffs the raw depth24 channel byte-exact, so guard
  // that DecodeEdramDepth recovers the exact 24-bit depth the offset formula
  // placed (independent of the 8-bit grayscale PNG). Write a unique
  // depth/stencil word per pixel at its depth offset; require depth24 back.
  const uint32_t w = 96, h = 40;
  const uint32_t base = 3, pitch = XeEdramPitchTiles(w, k1X);
  std::vector<uint8_t> edram(xenos::kEdramSizeBytes, 0);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint32_t off = XeEdramSnapshotOffsetBytes(x, y, base, pitch, k1X,
                                                /*is_depth=*/true, 0, 0);
      REQUIRE(off + 4 <= edram.size());
      // depth in the high 24 bits, stencil in the low 8; make depth24 unique.
      uint32_t depth24 = (x << 12) | y;
      uint32_t word = (depth24 << 8) | 0xAB;
      std::memcpy(edram.data() + off, &word, sizeof(word));
    }
  }
  DecodedImage img;
  std::vector<uint32_t> depth24;
  REQUIRE(DecodeEdramDepth(edram.data(), edram.size(), base, pitch, k1X,
                           xenos::DepthRenderTargetFormat::kD24S8, w, h, 0,
                           &img, &depth24));
  REQUIRE(depth24.size() == size_t(w) * h);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      REQUIRE(depth24[size_t(y) * w + x] == ((x << 12) | y));
    }
  }
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
