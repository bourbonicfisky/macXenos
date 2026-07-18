/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free): the strict-RGBA8 shadow-load logic — the shared
// eligibility predicate (every guard, positive and negative), the exact
// source-range computation, the aligned destination layout, and the
// deferred-comparison helper. This is the same header the production shadow
// path runs; no replicas.

#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_texture_load_shadow.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

xenos::xe_gpu_texture_fetch_t EligibleFetch() {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.type = xenos::FetchConstantType::kTexture;
  fetch.format = xenos::TextureFormat::k_8_8_8_8;
  fetch.sign_x = xenos::TextureSign::kUnsigned;
  fetch.sign_y = xenos::TextureSign::kUnsigned;
  fetch.sign_z = xenos::TextureSign::kUnsigned;
  fetch.sign_w = xenos::TextureSign::kUnsigned;
  fetch.tiled = 1;
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  fetch.stacked = 0;
  fetch.mip_min_level = 0;
  fetch.mip_max_level = 0;
  fetch.base_address = 0x1000;  // 4KiB units.
  return fetch;
}

constexpr uint32_t kW = 128, kH = 128, kPitch = 128;

TEST_CASE("shadow eligibility: strict RGBA8 positive case",
          "[l0][shadow]") {
  REQUIRE(IsStrictRgba8Eligible(EligibleFetch(), false, kW, kH, kPitch));
}

TEST_CASE("shadow eligibility: every negative guard rejects",
          "[l0][shadow]") {
  // Format aliases are NOT the class.
  {
    auto f = EligibleFetch();
    f.format = xenos::TextureFormat::k_8_8_8_8_A;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  {
    auto f = EligibleFetch();
    f.format = xenos::TextureFormat::k_DXT1;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  // Signed binding.
  REQUIRE_FALSE(IsStrictRgba8Eligible(EligibleFetch(), true, kW, kH, kPitch));
  // Any signed component.
  {
    auto f = EligibleFetch();
    f.sign_x = xenos::TextureSign::kSigned;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  {
    auto f = EligibleFetch();
    f.sign_w = xenos::TextureSign::kUnsignedBiased;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  // Linear (untiled).
  {
    auto f = EligibleFetch();
    f.tiled = 0;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  // Wrong dimension.
  {
    auto f = EligibleFetch();
    f.dimension = xenos::DataDimension::k3D;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  {
    auto f = EligibleFetch();
    f.dimension = xenos::DataDimension::kCube;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  // Stacked.
  {
    auto f = EligibleFetch();
    f.stacked = 1;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  // Mips beyond the base level.
  {
    auto f = EligibleFetch();
    f.mip_min_level = 1;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  {
    auto f = EligibleFetch();
    f.mip_max_level = 3;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  // Invalid base / dimensions / pitch.
  {
    auto f = EligibleFetch();
    f.base_address = 0;
    REQUIRE_FALSE(IsStrictRgba8Eligible(f, false, kW, kH, kPitch));
  }
  REQUIRE_FALSE(IsStrictRgba8Eligible(EligibleFetch(), false, 0, kH, kPitch));
  REQUIRE_FALSE(IsStrictRgba8Eligible(EligibleFetch(), false, kW, 0, kPitch));
  REQUIRE_FALSE(IsStrictRgba8Eligible(EligibleFetch(), false, kW, kH, 0));
}

TEST_CASE("shadow source range: exact bounds over the shader's padded read "
          "grid",
          "[l0][shadow]") {
  // Odd width: the shader pads x to 8-block thread strides; the range must
  // cover every 16-byte read group at Tiled2D(4g, y) — no more, no less.
  const uint32_t w = 41, h = 17, pitch = 64;
  auto r = ComputeStrictRgba8SourceRange(0x100000, w, h, pitch,
                                         512u << 20);
  REQUIRE(r.valid);
  uint32_t padded_w = 48;
  int32_t min_off = INT32_MAX, max_off = INT32_MIN;
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < padded_w; x += 4) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y), pitch, 2);
      min_off = std::min(min_off, off);
      max_off = std::max(max_off, off);
    }
  }
  REQUIRE(r.begin == 0x100000 + uint32_t(min_off));
  REQUIRE(r.end == 0x100000 + uint32_t(max_off) + 16);
  // Every read group lies inside the stated range.
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < padded_w; x += 4) {
      uint32_t addr = 0x100000 +
                      uint32_t(texture_address::Tiled2D(int32_t(x),
                                                        int32_t(y), pitch, 2));
      REQUIRE(addr >= r.begin);
      REQUIRE(addr + 16 <= r.end);
    }
  }
}

TEST_CASE("shadow source range: rejects out-of-bounds and degenerate shapes",
          "[l0][shadow]") {
  // Range past the end of physical memory is invalid, not clamped.
  auto r = ComputeStrictRgba8SourceRange((512u << 20) - 4096, 64, 64, 64,
                                         512u << 20);
  REQUIRE_FALSE(r.valid);
  REQUIRE_FALSE(
      ComputeStrictRgba8SourceRange(0x1000, 0, 64, 64, 512u << 20).valid);
  REQUIRE_FALSE(
      ComputeStrictRgba8SourceRange(0x1000, 64, 0, 64, 512u << 20).valid);
  REQUIRE_FALSE(
      ComputeStrictRgba8SourceRange(0x1000, 64, 64, 0, 512u << 20).valid);
}

TEST_CASE("shadow dest layout: 256-byte aligned rows, exact written extent",
          "[l0][shadow]") {
  {
    auto l = ComputeShadowDestLayout(41, 17);
    REQUIRE(l.logical_row_bytes == 164);
    REQUIRE(l.written_row_bytes == 192);  // round8(41)=48 blocks * 4.
    REQUIRE(l.host_pitch == 256);
    REQUIRE(l.rows == 17);
    REQUIRE(l.total_size == 256 * 17 + 256);
  }
  {
    // written_row_bytes exceeding a 256 multiple bumps the pitch.
    auto l = ComputeShadowDestLayout(63, 4);
    REQUIRE(l.written_row_bytes == 256);  // round8(63)=64 * 4.
    REQUIRE(l.host_pitch == 256);
  }
  {
    auto l = ComputeShadowDestLayout(65, 4);
    REQUIRE(l.written_row_bytes == 288);
    REQUIRE(l.host_pitch == 512);
  }
}

TEST_CASE("shadow compare: exact match, mismatch report, padding lanes "
          "ignored, guard violations",
          "[l0][shadow]") {
  const uint32_t w = 41, h = 3;
  auto l = ComputeShadowDestLayout(w, h);
  std::vector<uint8_t> cpu(size_t(w) * 4 * h);
  for (size_t i = 0; i < cpu.size(); ++i) {
    cpu[i] = uint8_t(i * 5 + 3);
  }
  std::vector<uint8_t> dest(l.total_size, kShadowDestSentinel);
  auto fill_dest_logical = [&] {
    for (uint32_t y = 0; y < h; ++y) {
      std::memcpy(&dest[size_t(y) * l.host_pitch],
                  &cpu[size_t(y) * l.logical_row_bytes],
                  l.logical_row_bytes);
      // Written-padding lanes hold arbitrary guest padding content.
      std::memset(&dest[size_t(y) * l.host_pitch + l.logical_row_bytes], 0xAB,
                  l.written_row_bytes - l.logical_row_bytes);
    }
  };
  fill_dest_logical();
  {
    auto r = CompareShadowDest(dest.data(), cpu.data(), l);
    REQUIRE(r.match);
    REQUIRE_FALSE(r.guard_violation);
    REQUIRE(r.bytes_compared == uint64_t(w) * 4 * h);
  }
  {
    // A flipped logical byte is a mismatch with exact location + values.
    dest[size_t(1) * l.host_pitch + 7] ^= 0x40;
    auto r = CompareShadowDest(dest.data(), cpu.data(), l);
    REQUIRE_FALSE(r.match);
    REQUIRE_FALSE(r.guard_violation);
    REQUIRE(r.first_mismatch_offset == l.host_pitch + 7);
    REQUIRE(r.expected_value == cpu[size_t(1) * l.logical_row_bytes + 7]);
    REQUIRE(r.actual_value ==
            uint8_t(cpu[size_t(1) * l.logical_row_bytes + 7] ^ 0x40));
    dest[size_t(1) * l.host_pitch + 7] ^= 0x40;
  }
  {
    // A write into the per-row guard region is a guard violation.
    dest[l.written_row_bytes] = 0x00;
    auto r = CompareShadowDest(dest.data(), cpu.data(), l);
    REQUIRE_FALSE(r.match);
    REQUIRE(r.guard_violation);
    REQUIRE(r.guard_offset == l.written_row_bytes);
    dest[l.written_row_bytes] = kShadowDestSentinel;
  }
  {
    // A write into the tail guard is a guard violation.
    dest[size_t(l.rows) * l.host_pitch + 5] = 0x11;
    auto r = CompareShadowDest(dest.data(), cpu.data(), l);
    REQUIRE(r.guard_violation);
    REQUIRE(r.guard_offset == uint32_t(size_t(l.rows) * l.host_pitch + 5));
    dest[size_t(l.rows) * l.host_pitch + 5] = kShadowDestSentinel;
  }
  {
    // Sanity after restores.
    auto r = CompareShadowDest(dest.data(), cpu.data(), l);
    REQUIRE(r.match);
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
