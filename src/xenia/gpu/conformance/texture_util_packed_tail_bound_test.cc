/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Gates for the canary texture_util layout ports (2026-09-28):
// - c332733af: GetTiledAddressUpperBound3D must be the exact end of the last
//   block any texel of the region lands in. Oracle = brute-force maximum of
//   texture_address::Tiled3D over every block, not a re-derived closed form.
// - ace153cb8: a fetch whose level 0 is already the packed mip tail has
//   mip_address 0; the mips live in the same tail at the base address, so
//   normalization must report the base page as the mip page.

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

uint64_t BruteForceUpperBound3D(uint32_t right, uint32_t bottom, uint32_t back,
                                uint32_t pitch_aligned, uint32_t height_aligned,
                                uint32_t bytes_per_block_log2) {
  int64_t last = 0;
  for (uint32_t z = 0; z < back; ++z) {
    for (uint32_t y = 0; y < bottom; ++y) {
      for (uint32_t x = 0; x < right; ++x) {
        last = std::max(last, texture_address::Tiled3D(
                                  int32_t(x), int32_t(y), int32_t(z),
                                  pitch_aligned, height_aligned,
                                  bytes_per_block_log2));
      }
    }
  }
  return uint64_t(last) + (uint64_t(1) << bytes_per_block_log2);
}

xenos::xe_gpu_texture_fetch_t MakeFetch2D(uint32_t width, uint32_t height,
                                          uint32_t base_page,
                                          uint32_t mip_page,
                                          uint32_t mip_max_level,
                                          bool packed_mips) {
  xenos::xe_gpu_texture_fetch_t fetch;
  std::memset(&fetch, 0, sizeof(fetch));
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  fetch.format = xenos::TextureFormat::k_DXT1;
  fetch.tiled = 1;
  fetch.pitch = 1;  // 32 texels
  fetch.size_2d.width = width - 1;
  fetch.size_2d.height = height - 1;
  fetch.base_address = base_page;
  fetch.mip_address = mip_page;
  fetch.mip_min_level = 0;
  fetch.mip_max_level = mip_max_level;
  fetch.packed_mips = packed_mips ? 1 : 0;
  return fetch;
}

}  // namespace

TEST_CASE("texture_util: tiled 3D upper bound is the exact last block end",
          "[texture_util][layout][tiled3d]") {
  struct Shape {
    uint32_t right, bottom, back, pitch_aligned, height_aligned;
  };
  // Pitch in the first and second 32-wide half of the 64-block period, X in
  // either half, Z inside and past the first 4 slices, non-tile-aligned
  // extents, and one width-exceeds-pitch region.
  const Shape shapes[] = {
      {1, 1, 1, 32, 32},     {32, 32, 4, 32, 32},   {33, 17, 5, 64, 32},
      {40, 32, 3, 96, 32},   {64, 64, 8, 64, 64},   {96, 40, 6, 96, 64},
      {100, 70, 9, 128, 96}, {127, 33, 2, 160, 64}, {16, 16, 16, 32, 32},
      {70, 10, 2, 32, 32},
  };
  for (uint32_t bpb_log2 = 0; bpb_log2 <= 4; ++bpb_log2) {
    for (const Shape& s : shapes) {
      INFO("bpb_log2=" << bpb_log2 << " right=" << s.right
                       << " bottom=" << s.bottom << " back=" << s.back
                       << " pitch=" << s.pitch_aligned
                       << " height=" << s.height_aligned);
      REQUIRE(texture_util::GetTiledAddressUpperBound3D(
                  s.right, s.bottom, s.back, s.pitch_aligned,
                  s.height_aligned, bpb_log2) ==
              BruteForceUpperBound3D(s.right, s.bottom, s.back,
                                     s.pitch_aligned, s.height_aligned,
                                     bpb_log2));
    }
  }
  REQUIRE(texture_util::GetTiledAddressUpperBound3D(0, 1, 1, 32, 32, 2) == 0);
}

TEST_CASE("texture_util: level-0 packed tail serves its mips from the base",
          "[texture_util][layout][packed-tail]") {
  uint32_t base_page = 0, mip_page = 0, mip_min = 0, mip_max = 0;

  SECTION("16x16 packed, mip_address 0: mips come from the base page") {
    auto fetch = MakeFetch2D(16, 16, 0x1234, 0, 4, true);
    REQUIRE(texture_util::GetPackedMipLevel(16, 16) == 0);
    texture_util::GetSubresourcesFromFetchConstant(
        fetch, nullptr, nullptr, nullptr, &base_page, &mip_page, &mip_min,
        &mip_max);
    REQUIRE(base_page == 0x1234);
    REQUIRE(mip_page == 0x1234);
    REQUIRE(mip_min == 0);
    REQUIRE(mip_max == 4);
    // The mip layout of that tail is populated at storage level 0, which is
    // what the legacy Metal upload reads levels >= packed_level through.
    auto layout = texture_util::GetGuestTextureLayout(
        fetch.dimension, fetch.pitch, 16, 16, 1, true, fetch.format, true,
        true, mip_max);
    REQUIRE(layout.packed_level == 0);
    REQUIRE(layout.mips[0].row_pitch_bytes != 0);
    REQUIRE(layout.mips_total_extent_bytes != 0);
  }

  SECTION("not packed: mip_address 0 still means base level only") {
    auto fetch = MakeFetch2D(16, 16, 0x1234, 0, 4, false);
    texture_util::GetSubresourcesFromFetchConstant(
        fetch, nullptr, nullptr, nullptr, &base_page, &mip_page, &mip_min,
        &mip_max);
    REQUIRE(mip_page == 0);
    REQUIRE(mip_max == 0);
  }

  SECTION("packed tail below level 0: mip_address 0 still drops the mips") {
    auto fetch = MakeFetch2D(64, 64, 0x1234, 0, 6, true);
    REQUIRE(texture_util::GetPackedMipLevel(64, 64) != 0);
    texture_util::GetSubresourcesFromFetchConstant(
        fetch, nullptr, nullptr, nullptr, &base_page, &mip_page, &mip_min,
        &mip_max);
    REQUIRE(mip_page == 0);
    REQUIRE(mip_max == 0);
  }

  SECTION("level 0 packed: the whole tail is at base_address") {
    // Real hardware reads the mips from the base tail even when mip_address
    // is nonzero (Canary 344fca758).
    auto fetch = MakeFetch2D(16, 16, 0x1234, 0x2000, 4, true);
    texture_util::GetSubresourcesFromFetchConstant(
        fetch, nullptr, nullptr, nullptr, &base_page, &mip_page, &mip_min,
        &mip_max);
    REQUIRE(mip_page == 0x1234);
    REQUIRE(mip_max == 4);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
