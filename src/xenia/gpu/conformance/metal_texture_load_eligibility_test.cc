/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the strict-RGBA8 GPU texture-load contract.
// Slices 3/4/5R proved this class byte-exact against the CPU untile across
// the recorded corpus (live JC2 12,480/12,480 + SF4 57,920/57,920 on clean
// runs); the 5R landing decision replays the mechanism through the shared
// TextureCache seam. This suite pins the three pure pieces that proof rests
// on so the replay inherits them unweakened:
//   1. the eligibility predicate — every guard is deliberate; weakening one
//      to increase engagement fails here first;
//   2. the EXACT source range the 32bpb load shader reads — the range the
//      seam validates/watches must cover every Tiled2D group access;
//   3. the 256-aligned destination layout the buffer->texture blit assumes.

#include "xenia/gpu/metal/metal_texture_load_eligibility.h"

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPhysicalSize = 512u << 20;

// A fetch that IS strict-RGBA8 eligible; each negative case below perturbs
// exactly one field so the guard under test is isolated.
xenos::xe_gpu_texture_fetch_t EligibleFetch() {
  xenos::xe_gpu_texture_fetch_t fetch = {};
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
  fetch.base_address = 0x11BE1;  // Any nonzero 4KiB-page base.
  return fetch;
}

bool Eligible(const xenos::xe_gpu_texture_fetch_t& fetch,
              bool is_signed = false, uint32_t width = 256,
              uint32_t height = 128, uint32_t pitch = 256) {
  return IsStrictRgba8Eligible(fetch, is_signed, width, height, pitch);
}

TEST_CASE("StrictRgba8Eligibility: baseline and gamma signs pass",
          "[texture_load_eligibility]") {
  auto fetch = EligibleFetch();
  REQUIRE(Eligible(fetch));
  // 5R amendment (user-approved 2026-07-18): gamma-corrected component
  // signs sample the same unorm bytes — eligible. The live JC2
  // resolve-backed pool is entirely gamma-signed.
  fetch.sign_x = xenos::TextureSign::kGamma;
  fetch.sign_y = xenos::TextureSign::kGamma;
  fetch.sign_z = xenos::TextureSign::kGamma;
  fetch.sign_w = xenos::TextureSign::kGamma;
  REQUIRE(Eligible(fetch));
}

TEST_CASE("StrictRgba8Eligibility: every guard falls one perturbation",
          "[texture_load_eligibility]") {
  SECTION("format must be k_8_8_8_8 exactly") {
    auto fetch = EligibleFetch();
    fetch.format = xenos::TextureFormat::k_8_8_8_8_A;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("signed binding excluded") {
    auto fetch = EligibleFetch();
    REQUIRE(!Eligible(fetch, /*is_signed=*/true));
  }
  SECTION("signed component sign excluded") {
    auto fetch = EligibleFetch();
    fetch.sign_y = xenos::TextureSign::kSigned;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("unsigned-biased component sign excluded") {
    auto fetch = EligibleFetch();
    fetch.sign_w = xenos::TextureSign::kUnsignedBiased;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("linear (untiled) excluded") {
    auto fetch = EligibleFetch();
    fetch.tiled = 0;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("non-2D dimension excluded") {
    auto fetch = EligibleFetch();
    fetch.dimension = xenos::DataDimension::k3D;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("stacked excluded") {
    auto fetch = EligibleFetch();
    fetch.stacked = 1;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("mips excluded (base mip only)") {
    auto fetch = EligibleFetch();
    fetch.mip_max_level = 1;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("zero base address excluded") {
    auto fetch = EligibleFetch();
    fetch.base_address = 0;
    REQUIRE(!Eligible(fetch));
  }
  SECTION("zero derived extents excluded") {
    auto fetch = EligibleFetch();
    REQUIRE(!Eligible(fetch, false, /*width=*/0, 128, 256));
    REQUIRE(!Eligible(fetch, false, 256, /*height=*/0, 256));
    REQUIRE(!Eligible(fetch, false, 256, 128, /*pitch=*/0));
  }
}

// Independent statement of the documented read set: the 32bpb shader reads
// [base + Tiled2D(4g, y), +16) for 4g < round_up(blocks_w, 8), y < blocks_h.
// The computed range must cover exactly that set — no group outside it, the
// min and max groups on its boundaries.
void CheckSourceRangeCoversReadSet(uint32_t base, uint32_t blocks_w,
                                   uint32_t blocks_h,
                                   uint32_t pitch_blocks_aligned) {
  TextureLoadSourceRange r = ComputeStrictRgba8SourceRange(
      base, blocks_w, blocks_h, pitch_blocks_aligned, kPhysicalSize);
  REQUIRE(r.valid);
  uint32_t padded_w = (blocks_w + 7u) & ~7u;
  uint32_t lo = UINT32_MAX;
  uint32_t hi = 0;
  for (uint32_t y = 0; y < blocks_h; ++y) {
    for (uint32_t x = 0; x < padded_w; x += 4) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y),
                                             pitch_blocks_aligned,
                                             /*bytes_per_block_log2=*/2);
      REQUIRE(off >= 0);
      uint32_t begin = base + uint32_t(off);
      REQUIRE(begin >= r.begin);
      REQUIRE(begin + 16 <= r.end);
      lo = std::min(lo, begin);
      hi = std::max(hi, begin + 16);
    }
  }
  // Tight bounds: the range is the hull of the read set, nothing more.
  REQUIRE(r.begin == lo);
  REQUIRE(r.end == hi);
}

TEST_CASE("StrictRgba8SourceRange: exact hull of the shader read set",
          "[texture_load_eligibility]") {
  // Aligned, odd-width (padded groups), odd-height, and sub-tile shapes;
  // pitch wider than width exercises the pitch-major tiled layout.
  CheckSourceRangeCoversReadSet(0x00100000, 256, 128, 256);
  CheckSourceRangeCoversReadSet(0x00100000, 33, 17, 64);
  CheckSourceRangeCoversReadSet(0x11BE1000, 640, 360, 640);
  CheckSourceRangeCoversReadSet(0x00004000, 8, 8, 32);
  CheckSourceRangeCoversReadSet(0x00004000, 5, 3, 32);
}

TEST_CASE("StrictRgba8SourceRange: rejects impossible inputs",
          "[texture_load_eligibility]") {
  SECTION("zero dimensions invalid") {
    REQUIRE(!ComputeStrictRgba8SourceRange(0, 0, 8, 32, kPhysicalSize).valid);
    REQUIRE(!ComputeStrictRgba8SourceRange(0, 8, 0, 32, kPhysicalSize).valid);
    REQUIRE(!ComputeStrictRgba8SourceRange(0, 8, 8, 0, kPhysicalSize).valid);
  }
  SECTION("range past physical memory invalid") {
    // A base near the top of physical memory whose tiled extent overruns it.
    uint32_t base = kPhysicalSize - 4096;
    REQUIRE(
        !ComputeStrictRgba8SourceRange(base, 256, 128, 256, kPhysicalSize)
             .valid);
  }
}

TEST_CASE("TextureLoadDestLayout: 256-aligned rows, exact guard split",
          "[texture_load_eligibility]") {
  SECTION("odd width pads to the 8-block stride then to 256") {
    TextureLoadDestLayout l = ComputeTextureLoadDestLayout(33, 17);
    REQUIRE(l.logical_row_bytes == 33u * 4);
    REQUIRE(l.written_row_bytes == 40u * 4);
    REQUIRE(l.host_pitch == 256u);
    REQUIRE(l.rows == 17u);
    REQUIRE(l.tail_guard_bytes == 256u);
    REQUIRE(l.total_size == 256u * 17 + 256u);
  }
  SECTION("invariants hold across shapes") {
    for (uint32_t w : {1u, 5u, 8u, 33u, 64u, 640u, 1280u}) {
      for (uint32_t h : {1u, 3u, 17u, 128u, 720u}) {
        TextureLoadDestLayout l = ComputeTextureLoadDestLayout(w, h);
        REQUIRE(l.logical_row_bytes <= l.written_row_bytes);
        REQUIRE(l.written_row_bytes <= l.host_pitch);
        REQUIRE(l.host_pitch % 256u == 0);
        REQUIRE(l.written_row_bytes % 32u == 0);  // round8(w) * 4.
        REQUIRE(l.rows == h);
        REQUIRE(l.total_size == l.host_pitch * l.rows + l.tail_guard_bytes);
      }
    }
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
