/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Row-wise publication granule for guest-tiled destinations
// (openspec/changes/fix-metal-gpu-current-8bpp-publication).
//
// The GPU-current publication walk memcpys runs of texels it assumes are
// byte-contiguous in the guest tiled layout. This locks the assumption to the
// REAL address rule (texture_address::Tiled2D): for every texel size, a run
// of TiledPublishRunTexels(bpp) texels starting on a run boundary must be
// contiguous, and the flat 16-byte assumption must be shown NOT to hold for
// 8bpp (the measured Dead or Alive 5 half-stale k_8 mask).

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_tiled_publish_run.h"
#include "xenia/gpu/texture_address.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

bool RunIsContiguous(uint32_t bpp, uint32_t bpp_log2, uint32_t run,
                     uint32_t pitch) {
  for (int32_t y = 0; y < 64; ++y) {
    for (int32_t x = 0; x < int32_t(pitch); x += int32_t(run)) {
      int32_t base = texture_address::Tiled2D(x, y, pitch, bpp_log2);
      for (uint32_t i = 1; i < run; ++i) {
        if (texture_address::Tiled2D(x + int32_t(i), y, pitch, bpp_log2) !=
            base + int32_t(i * bpp)) {
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace

TEST_CASE("Tiled publish run: every texel size publishes a contiguous run",
          "[metal][resolve][publish-run]") {
  const uint32_t kPitch = 320;
  struct Case {
    uint32_t bpp;
    uint32_t bpp_log2;
    uint32_t expected_run;
  } cases[] = {{1, 0, 8}, {2, 1, 8}, {4, 2, 4}, {8, 3, 2}, {16, 4, 1}};
  for (const Case& c : cases) {
    uint32_t run = metal::TiledPublishRunTexels(c.bpp);
    REQUIRE(run == c.expected_run);
    REQUIRE(RunIsContiguous(c.bpp, c.bpp_log2, run, kPitch));
  }
}

TEST_CASE("Tiled publish run: the flat 16-byte rule breaks for 8bpp",
          "[metal][resolve][publish-run]") {
  // 16 texels of 8bpp are NOT contiguous: texels 8-15 sit 64 bytes away
  // (pipe bits), and bytes 8-15 after the run start belong to row y ^ 2.
  REQUIRE_FALSE(RunIsContiguous(1, 0, 16, 320));
  REQUIRE(texture_address::Tiled2D(8, 0, 320, 0) ==
          texture_address::Tiled2D(0, 0, 320, 0) + 64);
  REQUIRE(texture_address::Tiled2D(0, 2, 320, 0) ==
          texture_address::Tiled2D(0, 0, 320, 0) + 8);
  // Coverage: walking a row in 16-texel steps with 16-byte copies leaves
  // every texel with x bit 3 set unpublished.
  // One 32x32 macro tile of 8bpp texels is 1024 bytes; the walk touches
  // offsets inside it only.
  bool covered[1024] = {};
  for (int32_t x = 0; x < 32; x += 16) {
    int32_t off = texture_address::Tiled2D(x, 0, 320, 0);
    REQUIRE(off + 16 <= 1024);
    for (int32_t b = 0; b < 16; ++b) {
      covered[off + b] = true;
    }
  }
  for (int32_t x = 0; x < 32; ++x) {
    int32_t off = texture_address::Tiled2D(x, 0, 320, 0);
    REQUIRE(off < 1024);
    bool expect = ((x >> 3) & 1) == 0;
    REQUIRE(covered[off] == expect);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
