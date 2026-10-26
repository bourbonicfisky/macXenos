/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Texture format x dimension parity with Canary D3D12 (2026-10-26).
//
// The reference is D3D12TextureCache::host_formats_ at canary e6cd2e4bc
// (src/xenia/gpu/d3d12/d3d12_texture_cache.h): every guest format it maps to
// a real DXGI format must have a shared-route entry here (with Apple packed
// 16-bit and BC host formats available), and every format it leaves UNKNOWN
// is listed so a later upstream change shows up as a failing case rather than
// a silent gap. Also pins the block geometry the loaders dispatch over, the
// legacy key's storage-identity fields, and the planner shapes newly routed
// (1D rows, packed 3D mip tails).

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_mip_level_plan.h"
#include "xenia/gpu/metal/metal_shared_texture_format_table.h"
#include "xenia/gpu/metal/metal_texture_key.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

using F = xenos::TextureFormat;

// D3D12 host_formats_ rows with a non-UNKNOWN DXGI format.
const F kD3D12Mapped[] = {
    F::k_8,
    F::k_1_5_5_5,
    F::k_5_6_5,
    F::k_6_5_5,
    F::k_8_8_8_8,
    F::k_2_10_10_10,
    F::k_8_A,
    F::k_8_8,
    F::k_Cr_Y1_Cb_Y0_REP,
    F::k_Y1_Cr_Y0_Cb_REP,
    F::k_4_4_4_4,
    F::k_10_11_11,
    F::k_11_11_10,
    F::k_DXT1,
    F::k_DXT2_3,
    F::k_DXT4_5,
    F::k_24_8,
    F::k_24_8_FLOAT,
    F::k_16,
    F::k_16_16,
    F::k_16_16_16_16,
    F::k_16_EXPAND,
    F::k_16_16_EXPAND,
    F::k_16_16_16_16_EXPAND,
    F::k_16_FLOAT,
    F::k_16_16_FLOAT,
    F::k_16_16_16_16_FLOAT,
    F::k_32_FLOAT,
    F::k_32_32_FLOAT,
    F::k_32_32_32_32_FLOAT,
    F::k_DXN,
    F::k_8_8_8_8_AS_16_16_16_16,
    F::k_DXT1_AS_16_16_16_16,
    F::k_DXT2_3_AS_16_16_16_16,
    F::k_DXT4_5_AS_16_16_16_16,
    F::k_2_10_10_10_AS_16_16_16_16,
    F::k_10_11_11_AS_16_16_16_16,
    F::k_11_11_10_AS_16_16_16_16,
    F::k_DXT3A,
    F::k_DXT5A,
    F::k_CTX1,
    F::k_DXT3A_AS_1_1_1_1,
};

// D3D12 leaves these UNKNOWN (not sampleable on any upstream backend). k_8_B
// and k_8_8_8_8_A are UNKNOWN rows too, but the shared GetBaseFormat maps them
// to k_8 / k_8_8_8_8 before the host-format lookup, so D3D12 serves them
// through those rows; they are checked as aliases below.
const F kD3D12Unknown[] = {
    F::k_1_REVERSE,
    F::k_1,
    F::k_16_16_EDRAM,
    F::k_16_16_16_16_EDRAM,
    F::k_32,
    F::k_32_32,
    F::k_32_32_32_32,
    F::k_32_AS_8,
    F::k_32_AS_8_8,
    F::k_16_MPEG,
    F::k_16_16_MPEG,
    F::k_8_INTERLACED,
    F::k_32_AS_8_INTERLACED,
    F::k_32_AS_8_8_INTERLACED,
    F::k_16_INTERLACED,
    F::k_16_MPEG_INTERLACED,
    F::k_16_16_MPEG_INTERLACED,
    F::k_32_32_32_FLOAT,
    F::k_8_8_8_8_GAMMA_EDRAM,
    F::k_2_10_10_10_FLOAT_EDRAM,
};

struct DeviceCaps {
  DeviceCaps(bool packed16, bool bc) {
    saved_packed16 = SharedTexturePacked16Supported().load();
    saved_bc = SharedTextureBcSupported().load();
    SharedTexturePacked16Supported().store(packed16);
    SharedTextureBcSupported().store(bc);
  }
  ~DeviceCaps() {
    SharedTexturePacked16Supported().store(saved_packed16);
    SharedTextureBcSupported().store(saved_bc);
  }
  bool saved_packed16, saved_bc;
};

}  // namespace

TEST_CASE("Every D3D12-mapped texture format has a shared-route entry",
          "[texmatrix][formats]") {
  DeviceCaps caps(/*packed16=*/true, /*bc=*/true);
  for (F format : kD3D12Mapped) {
    INFO("format " << uint32_t(format));
    CHECK(FindSharedTextureFormat(format) != nullptr);
  }
}

TEST_CASE("D3D12 UNKNOWN formats stay unserved, aliases resolve to the base",
          "[texmatrix][formats]") {
  DeviceCaps caps(true, true);
  for (F format : kD3D12Unknown) {
    INFO("format " << uint32_t(format));
    CHECK(FindSharedTextureFormat(format) == nullptr);
  }
  CHECK(FindSharedTextureFormat(F::k_8_A) == FindSharedTextureFormat(F::k_8));
  CHECK(FindSharedTextureFormat(F::k_8_B) == FindSharedTextureFormat(F::k_8));
  CHECK(FindSharedTextureFormat(F::k_8_8_8_8_A) ==
        FindSharedTextureFormat(F::k_8_8_8_8));
}

TEST_CASE("Apple-only host formats drop out without packed 16-bit support",
          "[texmatrix][formats]") {
  DeviceCaps caps(/*packed16=*/false, /*bc=*/true);
  // B5G6R5 / ABGR4 / BGR5A1 exist only on Apple GPUs: the entries that need
  // them without an expanding sibling are unavailable on Intel/AMD Macs.
  CHECK(FindSharedTextureFormat(F::k_6_5_5) == nullptr);
  CHECK(FindSharedTextureFormat(F::k_DXT3A_AS_1_1_1_1) == nullptr);
  CHECK(FindSharedTextureFormat(F::k_1_5_5_5) == nullptr);
  // 4:4:4:4 and 5:6:5 keep their RGBA8 expansions.
  const SharedTextureFormatEntry* rgba4 = FindSharedTextureFormat(F::k_4_4_4_4);
  REQUIRE(rgba4 != nullptr);
  CHECK(rgba4->load_shader == SharedLoadFamily::k16bpbRgba4ToRgba8);
  const SharedTextureFormatEntry* r565 = FindSharedTextureFormat(F::k_5_6_5);
  REQUIRE(r565 != nullptr);
  CHECK(r565->load_shader == SharedLoadFamily::k16bpbR5G6B5ToRgba8);
  // Formats needing no Apple-only host format are unaffected.
  CHECK(FindSharedTextureFormat(F::k_Cr_Y1_Cb_Y0_REP) != nullptr);
  CHECK(FindSharedTextureFormat(F::k_32_32_32_32_FLOAT) != nullptr);
}

TEST_CASE("New entries carry the D3D12 host swizzle and block geometry",
          "[texmatrix][formats]") {
  DeviceCaps caps(true, true);
  const SharedTextureFormatEntry* e655 = FindSharedTextureFormat(F::k_6_5_5);
  REQUIRE(e655 != nullptr);
  CHECK(e655->host_swizzle == XE_GPU_MAKE_TEXTURE_SWIZZLE(R, B, G, G));
  CHECK(e655->host_format == kMtlPixelFormatB5G6R5Unorm);
  CHECK(SharedTextureHostBytesPerBlock(*e655) == 2u);
  CHECK(SharedTextureBlockWidth(*e655) == 1u);

  const SharedTextureFormatEntry* e1111 =
      FindSharedTextureFormat(F::k_DXT3A_AS_1_1_1_1);
  REQUIRE(e1111 != nullptr);
  CHECK(e1111->host_format == kMtlPixelFormatABGR4Unorm);
  CHECK(e1111->bytes_per_block == 8u);
  CHECK(SharedTextureHostBytesPerBlock(*e1111) == 2u);  // per host texel
  CHECK(SharedTextureHostExpand(*e1111) == 4u);
  CHECK(SharedTextureBlockWidth(*e1111) == 4u);
  CHECK(SharedTextureBlockHeight(*e1111) == 4u);

  for (F f422 : {F::k_Cr_Y1_Cb_Y0_REP, F::k_Y1_Cr_Y0_Cb_REP}) {
    const SharedTextureFormatEntry* e = FindSharedTextureFormat(f422);
    REQUIRE(e != nullptr);
    CHECK(e->host_swizzle == xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB);
    CHECK(e->bytes_per_block == 4u);
    CHECK(SharedTextureHostBytesPerBlock(*e) == 8u);  // two RGBA8 texels
    CHECK(SharedTextureHostExpand(*e) == 1u);
    CHECK(SharedTextureBlockWidth(*e) == 2u);
    CHECK(SharedTextureBlockHeight(*e) == 1u);
  }

  const SharedTextureFormatEntry* rgba32f =
      FindSharedTextureFormat(F::k_32_32_32_32_FLOAT);
  REQUIRE(rgba32f != nullptr);
  CHECK(rgba32f->load_shader == SharedLoadFamily::k128bpb);
  CHECK(rgba32f->bytes_per_block == 16u);
}

TEST_CASE("Legacy key distinguishes mip address, packed mips and stacked",
          "[texmatrix][texture_key]") {
  MetalTextureKeyInputs base;
  base.base = 0x03252000;
  base.width = 128;
  base.height = 128;
  base.pitch_texels = 128;
  base.format_tiled_endian = 0xF0000;
  base.swizzle = 0x688;
  base.mip_min_level = 0;
  base.mip_max_level = 3;
  base.mip_address = 0x3260;
  const uint64_t k = MakeMetalTextureKey(base);

  MetalTextureKeyInputs other_mip = base;
  other_mip.mip_address = 0x3270;
  CHECK(MakeMetalTextureKey(other_mip) != k);

  MetalTextureKeyInputs packed = base;
  packed.packed_mips = 1;
  CHECK(MakeMetalTextureKey(packed) != k);

  MetalTextureKeyInputs stacked = base;
  stacked.stacked = 1;
  CHECK(MakeMetalTextureKey(stacked) != k);

  MetalTextureKeyInputs same = base;
  CHECK(MakeMetalTextureKey(same) == k);
}

TEST_CASE("1D rows plan every level, zero fetch pitch included",
          "[texmatrix][mipplan]") {
  const uint32_t width = 256;
  const uint32_t max_level = xe::log2_floor(width);
  texture_util::TextureGuestLayout layout = texture_util::GetGuestTextureLayout(
      xenos::DataDimension::k1D, /*pitch_div32=*/0, width, 1, 1,
      /*tiled=*/false, F::k_8, /*has_packed_levels=*/false,
      /*has_base=*/true, max_level);
  REQUIRE(layout.max_level == max_level);
  MipLevelPlanInputs in;
  in.base_address = 0x10000000;
  in.mip_address = 0x11000000;
  in.width = width;
  in.height = 1;
  in.mip_max_level = layout.max_level;
  in.bytes_per_block = 1;
  in.block_dim = 1;
  in.is_1d = true;
  in.guest_buffer_size = 0x20000000;
  in.load_base = true;
  in.load_mips = true;
  std::vector<MipLevelPlan> plan;
  REQUIRE(BuildMipLevelPlan(in, layout, &plan));
  REQUIRE(plan.size() == size_t(max_level) + 1);
  for (const MipLevelPlan& p : plan) {
    CHECK(p.height == 1u);
    CHECK(p.width == std::max(width >> p.host_level, 1u));
    CHECK(p.pitch_blocks > 0u);
  }
  // Without is_1d the zero pitch is a planning error, as before.
  in.is_1d = false;
  CHECK_FALSE(BuildMipLevelPlan(in, layout, &plan));
}

TEST_CASE("Packed 3D mip tails plan with an origin inside the tail",
          "[texmatrix][mipplan]") {
  // 32x32x32 RGBA8 volume, full chain: levels from the packed level on share
  // the tail's storage and are told apart by their (x, y, z) origin.
  const uint32_t size = 32;
  const uint32_t max_level = xe::log2_floor(size);
  texture_util::TextureGuestLayout layout = texture_util::GetGuestTextureLayout(
      xenos::DataDimension::k3D, /*pitch_div32=*/1, size, size, size,
      /*tiled=*/true, F::k_8_8_8_8, /*has_packed_levels=*/true,
      /*has_base=*/true, max_level);
  REQUIRE(layout.packed_level != UINT32_MAX);
  MipLevelPlanInputs in;
  in.base_address = 0x10000000;
  in.mip_address = 0x11000000;
  in.width = size;
  in.height = size;
  in.array_size = size;
  in.is_3d = true;
  in.mip_max_level = layout.max_level;
  in.guest_buffer_size = 0x20000000;
  in.load_base = true;
  in.load_mips = true;
  std::vector<MipLevelPlan> plan;
  REQUIRE(BuildMipLevelPlan(in, layout, &plan));
  REQUIRE(plan.size() == size_t(layout.max_level) + 1);
  uint32_t tail_offset = 0;
  bool first_tail = true;
  uint32_t with_origin = 0;
  for (const MipLevelPlan& p : plan) {
    CHECK(p.depth == std::max(size >> p.host_level, 1u));
    if (p.host_level == 0 || p.host_level < layout.packed_level) {
      continue;
    }
    if (first_tail) {
      tail_offset = p.guest_offset;
      first_tail = false;
    } else {
      CHECK(p.guest_offset == tail_offset);
    }
    if (p.px || p.py || p.pz) {
      ++with_origin;
    }
    // The origin keeps the level inside the 32-aligned tail.
    CHECK(p.px + p.width <= 32u);
    CHECK(p.py + p.height <= 32u);
  }
  CHECK(with_origin >= 1u);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
