/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Packed base level placement for the legacy Metal texture cache
// (openspec/changes/diagnose-sonic-black-sky-columns, task 4).
//
// Exercises the REAL helper the cache uses (metal::ComputeMetalPackedBaseOffset)
// against the REAL shared packing rules (texture_util::GetPackedMipLevel /
// GetPackedMipOffset). The measured defect: an 8x8 DXT1 fetch with packed mips
// had its base read at block (0, 0) of the base tail, which holds the smaller
// levels — a uniform white sky gradient decoded as a black/white checker.

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_packed_base.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::ComputeMetalPackedBaseOffset;
using metal::MetalPackedBaseOffset;

constexpr auto k2D = xenos::DataDimension::k2DOrStacked;

}  // namespace

TEST_CASE("Metal packed base: 8x8 DXT1 base sits at the level-0 packed offset",
          "[metal][texture][packed-base]") {
  // The Sonic Generations sky gradient: 8x8, DXT1, packed mips declared.
  MetalPackedBaseOffset o = ComputeMetalPackedBaseOffset(
      true, k2D, 8, 8, xenos::TextureFormat::k_DXT1);
  REQUIRE(o.packed);
  // Must equal what the shared rule reports for level 0, not a re-derivation.
  uint32_t xb = 0, yb = 0, zb = 0;
  REQUIRE(texture_util::GetPackedMipOffset(8, 8, 1,
                                           xenos::TextureFormat::k_DXT1, 0, xb,
                                           yb, zb));
  REQUIRE(o.x_blocks == xb);
  REQUIRE(o.y_blocks == yb);
  // Square: laid out horizontally at 16 texels = 4 DXT1 blocks, row 0.
  REQUIRE(o.x_blocks == 4);
  REQUIRE(o.y_blocks == 0);
}

TEST_CASE("Metal packed base: uncompressed offsets are in texel blocks",
          "[metal][texture][packed-base]") {
  MetalPackedBaseOffset o = ComputeMetalPackedBaseOffset(
      true, k2D, 8, 8, xenos::TextureFormat::k_8_8_8_8);
  REQUIRE(o.packed);
  REQUIRE(o.x_blocks == 16);
  REQUIRE(o.y_blocks == 0);
}

TEST_CASE("Metal packed base: wider-than-tall packs vertically",
          "[metal][texture][packed-base]") {
  MetalPackedBaseOffset o = ComputeMetalPackedBaseOffset(
      true, k2D, 16, 4, xenos::TextureFormat::k_8_8_8_8);
  REQUIRE(o.packed);
  REQUIRE(o.x_blocks == 0);
  REQUIRE(o.y_blocks == 16);
}

TEST_CASE("Metal packed base: 16x16 is the last packed size, 32x32 is not",
          "[metal][texture][packed-base]") {
  REQUIRE(ComputeMetalPackedBaseOffset(true, k2D, 16, 16,
                                       xenos::TextureFormat::k_8_8_8_8)
              .packed);
  REQUIRE_FALSE(ComputeMetalPackedBaseOffset(true, k2D, 32, 32,
                                             xenos::TextureFormat::k_8_8_8_8)
                    .packed);
  // Shorter side decides.
  REQUIRE(ComputeMetalPackedBaseOffset(true, k2D, 256, 16,
                                       xenos::TextureFormat::k_8_8_8_8)
              .packed);
}

TEST_CASE("Metal packed base: no packed mips, 3D, or degenerate sizes read (0,0)",
          "[metal][texture][packed-base]") {
  REQUIRE_FALSE(ComputeMetalPackedBaseOffset(false, k2D, 8, 8,
                                             xenos::TextureFormat::k_DXT1)
                    .packed);
  REQUIRE_FALSE(ComputeMetalPackedBaseOffset(true, xenos::DataDimension::k3D,
                                             8, 8, xenos::TextureFormat::k_DXT1)
                    .packed);
  REQUIRE_FALSE(ComputeMetalPackedBaseOffset(true, k2D, 0, 8,
                                             xenos::TextureFormat::k_DXT1)
                    .packed);
  MetalPackedBaseOffset o = ComputeMetalPackedBaseOffset(
      false, k2D, 8, 8, xenos::TextureFormat::k_DXT1);
  REQUIRE(o.x_blocks == 0);
  REQUIRE(o.y_blocks == 0);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
