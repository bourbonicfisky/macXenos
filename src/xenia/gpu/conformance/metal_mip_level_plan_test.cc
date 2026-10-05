/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the OpenSpec 6.1 production mip-routing decision
// and per-level plan.
//
// The planner is the single authority production calls (BuildMipLevelPlan), so
// a rule dropped or weakened there fails here. These cases cover what the live
// fixtures cannot: exact packed-origin read sets, packed_level == 0, key/layout
// disagreement, and partial base-only / mip-only reloads. The 6.1-G0 shader
// tests prove the DISPATCH is byte-correct; they say nothing about which levels
// production plans or which bytes it claims, which is what this file pins.

#include "xenia/gpu/metal/metal_mip_level_plan.h"

#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/texture_util.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kBaseAddr = 0x10000000;
constexpr uint32_t kMipAddr = 0x11000000;
constexpr uint32_t kBufferSize = 0x20000000;  // SharedMemory::kBufferSize

// The shared helper asserts that the requested level count fits the
// dimensions, so clamp exactly as it would -- a test must not feed it an
// impossible chain.
uint32_t MaxLevelFor(uint32_t w, uint32_t h) {
  return xe::log2_floor(std::max(w, h));
}

texture_util::TextureGuestLayout LayoutFor(uint32_t w, uint32_t h,
                                          uint32_t max_level,
                                          bool packed = true) {
  uint32_t pitch_div32 = ((w + 31u) & ~31u) >> 5;
  return texture_util::GetGuestTextureLayout(
      xenos::DataDimension::k2DOrStacked, pitch_div32, w, h, 1, /*tiled=*/true,
      xenos::TextureFormat::k_8_8_8_8, /*has_packed_levels=*/packed,
      /*has_base=*/true, std::min(max_level, MaxLevelFor(w, h)));
}

MipLevelPlanInputs InputsFor(uint32_t w, uint32_t h, uint32_t max_level,
                             bool load_base = true, bool load_mips = true) {
  MipLevelPlanInputs in;
  in.base_address = kBaseAddr;
  in.mip_address = kMipAddr;
  in.width = w;
  in.height = h;
  in.mip_max_level = max_level;
  in.guest_buffer_size = kBufferSize;
  in.load_base = load_base;
  in.load_mips = load_mips;
  return in;
}

// ---------------------------------------------------------------------------
// Eligibility (the mipped sibling predicate)
// ---------------------------------------------------------------------------

xenos::xe_gpu_texture_fetch_t MippedFetch() {
  xenos::xe_gpu_texture_fetch_t f = {};
  f.type = xenos::FetchConstantType::kTexture;
  f.sign_x = f.sign_y = f.sign_z = f.sign_w = xenos::TextureSign::kUnsigned;
  f.format = xenos::TextureFormat::k_8_8_8_8;
  f.tiled = 1;
  f.dimension = xenos::DataDimension::k2DOrStacked;
  f.base_address = kBaseAddr >> 12;
  f.mip_address = kMipAddr >> 12;
  f.pitch = 8;  // 256 texels
  f.size_2d.width = 255;
  f.size_2d.height = 255;
  f.mip_min_level = 0;
  f.mip_max_level = 4;
  return f;
}

TEST_CASE("Mipped eligibility requires an unshifted level set", "[mipplan]") {
  auto f = MippedFetch();
  CHECK(IsStrictRgba8MipRouteEligible(f, false, 256, 256, 256));

  // mip_min != 0 is the mips-only class (2026-10-05): the chain still loads
  // whole from the mip pointer and the sampler clamps the LOD to [min, max],
  // so a shifted level set IS admitted (Red Dead's RGBA8 pyramids fetched
  // one level at a time). An inverted set (min above max) is not.
  auto shifted = f;
  shifted.mip_min_level = 1;
  CHECK(IsStrictRgba8MipRouteEligible(shifted, false, 256, 256, 256));
  auto inverted = f;
  inverted.mip_min_level = 5;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(inverted, false, 256, 256, 256));

  // No chain at all is the promoted single-mip class, not this one.
  auto flat = f;
  flat.mip_max_level = 0;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(flat, false, 256, 256, 256));

  // A chain with no mip pointer cannot be planned.
  auto no_mip_addr = f;
  no_mip_addr.mip_address = 0;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(no_mip_addr, false, 256, 256, 256));

  // Raw pitch zero stays on the legacy path (the key carries the raw pitch).
  auto no_pitch = f;
  no_pitch.pitch = 0;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(no_pitch, false, 256, 256, 256));
}

TEST_CASE("Mipped eligibility inherits every strict condition", "[mipplan]") {
  auto f = MippedFetch();
  auto signed_fetch = f;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(signed_fetch, true, 256, 256, 256));
  auto untiled = f;
  untiled.tiled = 0;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(untiled, false, 256, 256, 256));
  // k_8_8_8_8_AS_16_16_16_16 is a BYTE alias of k_8_8_8_8 (same bytes, same
  // decode — the legacy FormatPlan maps both to one plan) and is admitted
  // since the reopen widening; a genuinely value-changing format stays out.
  auto alias_format = f;
  alias_format.format = xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16;
  CHECK(IsStrictRgba8MipRouteEligible(alias_format, false, 256, 256, 256));
  auto wrong_format = f;
  wrong_format.format = xenos::TextureFormat::k_2_10_10_10;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(wrong_format, false, 256, 256, 256));
  auto stacked = f;
  stacked.stacked = 1;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(stacked, false, 256, 256, 256));
  auto cube = f;
  cube.dimension = xenos::DataDimension::kCube;
  CHECK_FALSE(IsStrictRgba8MipRouteEligible(cube, false, 256, 256, 256));
  // The promoted single-mip predicate is NOT weakened by the sibling.
  CHECK_FALSE(IsStrictRgba8Eligible(f, false, 256, 256, 256));
}

// ---------------------------------------------------------------------------
// Exact consumed ranges, including packed origins
// ---------------------------------------------------------------------------

TEST_CASE("Origin-aware range equals the shader's exact read set",
          "[mipplan]") {
  // At origin (0,0) the origin-aware helper must agree exactly with the proven
  // single-mip contract, so the mipped route cannot drift from the promoted one.
  for (uint32_t w : {8u, 32u, 96u, 256u}) {
    for (uint32_t h : {8u, 48u, 256u}) {
      TextureLoadSourceRange flat =
          ComputeStrictRgba8SourceRange(0, w, h, 256, UINT32_MAX);
      TextureLoadSourceRange org =
          ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, w, h, 256, UINT32_MAX);
      REQUIRE(flat.valid);
      REQUIRE(org.valid);
      CHECK(org.begin == flat.begin);
      CHECK(org.end == flat.end);
    }
  }
}

TEST_CASE("A packed origin does not claim the whole rectangle", "[mipplan]") {
  // The decisive property: scanning from the ORIGIN is exact, while scanning an
  // origin-to-end rectangle over-consumes -- it would also claim the bytes of
  // every larger level sharing the tail. A nonzero origin must therefore
  // produce a strictly smaller span than the rectangle form.
  const uint32_t px = 16, py = 16, w = 4, h = 4, pitch = 256;
  TextureLoadSourceRange exact = ComputeStrictRgba8SourceRangeAtOrigin(
      0, px, py, w, h, pitch, UINT32_MAX);
  TextureLoadSourceRange rect = ComputeStrictRgba8SourceRange(
      0, px + w, py + h, pitch, UINT32_MAX);
  REQUIRE(exact.valid);
  REQUIRE(rect.valid);
  uint32_t exact_span = exact.end - exact.begin;
  uint32_t rect_span = rect.end - rect.begin;
  CHECK(exact_span < rect_span);
  // And the exact span must cover every address the shader actually reads.
  uint32_t padded_w = (w + 7u) & ~7u;
  for (uint32_t y = py; y < py + h; ++y) {
    for (uint32_t x = px; x < px + padded_w; x += 4) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y), pitch, 2);
      REQUIRE(off >= 0);
      CHECK(uint32_t(off) >= exact.begin);
      CHECK(uint32_t(off) + 16u <= exact.end);
    }
  }
}

// ---------------------------------------------------------------------------
// The plan itself
// ---------------------------------------------------------------------------

TEST_CASE("A full chain plans every level from the shared layout",
          "[mipplan]") {
  texture_util::TextureGuestLayout layout = LayoutFor(256, 256, 4);
  std::vector<MipLevelPlan> plan;
  REQUIRE(BuildMipLevelPlan(InputsFor(256, 256, layout.max_level), layout,
                            &plan));
  REQUIRE(plan.size() == size_t(layout.max_level) + 1);
  for (uint32_t level = 0; level <= layout.max_level; ++level) {
    const MipLevelPlan& p = plan[level];
    CHECK(p.host_level == level);
    CHECK(p.width == std::max(256u >> level, 1u));
    CHECK(p.height == std::max(256u >> level, 1u));
    CHECK(p.pitch_blocks > 0);
    CHECK(p.consumed_end > p.consumed_begin);
    // Level 0 reads the base pointer; every mip level reads the mip pointer.
    CHECK(p.guest_offset >= (level == 0 ? kBaseAddr : kMipAddr));
  }
  // Packed levels carry a sub-tile origin; non-packed levels must not.
  for (const MipLevelPlan& p : plan) {
    bool is_packed =
        layout.packed_level != UINT32_MAX && p.host_level >= layout.packed_level;
    if (!is_packed) {
      CHECK(p.px == 0);
      CHECK(p.py == 0);
    }
  }
}

TEST_CASE("Packed tail levels share a store and differ by sub-tile origin",
          "[mipplan]") {
  texture_util::TextureGuestLayout layout = LayoutFor(256, 256, 8);
  REQUIRE(layout.packed_level != UINT32_MAX);
  std::vector<MipLevelPlan> plan;
  REQUIRE(BuildMipLevelPlan(InputsFor(256, 256, layout.max_level), layout,
                            &plan));
  uint32_t packed_with_origin = 0;
  uint32_t shared_offset = 0;
  bool first = true;
  for (const MipLevelPlan& p : plan) {
    if (p.host_level < layout.packed_level || p.host_level == 0) {
      continue;
    }
    // Every packed level reads the SAME stored region...
    if (first) {
      shared_offset = p.guest_offset;
      first = false;
    } else {
      CHECK(p.guest_offset == shared_offset);
    }
    if (p.px || p.py) {
      ++packed_with_origin;
    }
  }
  // ...and is distinguished by its sub-tile origin, so at least one packed
  // level past the tail's first must carry a nonzero origin.
  CHECK(packed_with_origin >= 1);
}

TEST_CASE("packed_level == 0 plans the base tail and the mips tail separately",
          "[mipplan]") {
  // Live-unexercised in every scripted title (JC2/LEGO/SF4 all reported 0), so
  // this is its only coverage. A small square with a chain puts the base itself
  // in the packed tail; the shared loader then stores a base tail at the base
  // address and a SEPARATE mips tail at the mip address.
  // 16x16 with a chain puts the base itself in the packed tail. REQUIRED, not
  // guarded: if the shared layout ever stops producing this case, the test must
  // FAIL so the base-tail path is re-examined, never silently skipped.
  texture_util::TextureGuestLayout layout = LayoutFor(16, 16, 4);
  REQUIRE(layout.packed_level == 0);
  std::vector<MipLevelPlan> plan;
  REQUIRE(BuildMipLevelPlan(InputsFor(16, 16, layout.max_level), layout, &plan));
  REQUIRE(plan.size() >= 2);
  // Level 0 comes from the BASE pointer, the mip levels from the MIP pointer --
  // the two tails are distinct regions, never one shared read.
  CHECK(plan[0].host_level == 0);
  CHECK(plan[0].guest_offset >= kBaseAddr);
  CHECK(plan[0].guest_offset < kMipAddr);
  for (size_t i = 1; i < plan.size(); ++i) {
    CHECK(plan[i].guest_offset >= kMipAddr);
  }
}

TEST_CASE("Partial reloads plan only the requested region", "[mipplan]") {
  texture_util::TextureGuestLayout layout = LayoutFor(256, 256, 4);
  // Base-only reload: exactly one level, reading the base pointer.
  std::vector<MipLevelPlan> base_only;
  REQUIRE(BuildMipLevelPlan(
      InputsFor(256, 256, layout.max_level, /*load_base=*/true,
                /*load_mips=*/false),
      layout, &base_only));
  REQUIRE(base_only.size() == 1);
  CHECK(base_only[0].host_level == 0);
  CHECK(base_only[0].guest_offset >= kBaseAddr);
  CHECK(base_only[0].guest_offset < kMipAddr);

  // Mip-only reload: the chain without level 0, all from the mip pointer.
  std::vector<MipLevelPlan> mips_only;
  REQUIRE(BuildMipLevelPlan(
      InputsFor(256, 256, layout.max_level, /*load_base=*/false,
                /*load_mips=*/true),
      layout, &mips_only));
  REQUIRE(mips_only.size() == size_t(layout.max_level));
  for (const MipLevelPlan& p : mips_only) {
    CHECK(p.host_level >= 1);
    CHECK(p.guest_offset >= kMipAddr);
  }

  // Neither region requested is not a load.
  std::vector<MipLevelPlan> none;
  CHECK_FALSE(BuildMipLevelPlan(
      InputsFor(256, 256, layout.max_level, false, false), layout, &none));
}

TEST_CASE("The plan refuses states the route cannot honor", "[mipplan]") {
  texture_util::TextureGuestLayout layout = LayoutFor(256, 256, 4);
  std::vector<MipLevelPlan> plan;

  // Key/layout level disagreement is a planning error, never a silent
  // truncation -- the descriptor is built from the same pair. (The layout here
  // is legal; only the KEY over-claims, which is exactly the divergence the
  // planner must refuse.)
  MipLevelPlanInputs mismatched = InputsFor(256, 256, layout.max_level + 1);
  CHECK_FALSE(BuildMipLevelPlan(mismatched, layout, &plan));
  MipLevelPlanInputs under = InputsFor(256, 256, layout.max_level - 1);
  CHECK_FALSE(BuildMipLevelPlan(under, layout, &plan));

  // A mip chain with no mip pointer.
  MipLevelPlanInputs no_mip = InputsFor(256, 256, layout.max_level);
  no_mip.mip_address = 0;
  CHECK_FALSE(BuildMipLevelPlan(no_mip, layout, &plan));

  // No base pointer.
  MipLevelPlanInputs no_base = InputsFor(256, 256, layout.max_level);
  no_base.base_address = 0;
  CHECK_FALSE(BuildMipLevelPlan(no_base, layout, &plan));

  // Addresses near the end of the guest buffer must be rejected on the 64-bit
  // bound, not wrapped into a bogus in-range offset.
  MipLevelPlanInputs near_end = InputsFor(256, 256, layout.max_level);
  near_end.base_address = kBufferSize - 0x1000;
  near_end.mip_address = kBufferSize - 0x1000;
  CHECK_FALSE(BuildMipLevelPlan(near_end, layout, &plan));

  MipLevelPlanInputs past_end = InputsFor(256, 256, layout.max_level);
  past_end.mip_address = kBufferSize + 0x1000;  // beyond the buffer entirely
  CHECK_FALSE(BuildMipLevelPlan(past_end, layout, &plan));
}

TEST_CASE("Every planned range lies inside the guest buffer", "[mipplan]") {
  for (uint32_t max_level : {1u, 4u, 8u}) {
    for (auto dims : {std::pair<uint32_t, uint32_t>{256, 256},
                      std::pair<uint32_t, uint32_t>{96, 48},
                      std::pair<uint32_t, uint32_t>{160, 96}}) {
      texture_util::TextureGuestLayout layout =
          LayoutFor(dims.first, dims.second, max_level);
      std::vector<MipLevelPlan> plan;
      if (!BuildMipLevelPlan(InputsFor(dims.first, dims.second,
                                       layout.max_level),
                             layout, &plan)) {
        continue;  // dimensions that cannot hold this chain
      }
      for (const MipLevelPlan& p : plan) {
        CHECK(p.consumed_begin < p.consumed_end);
        CHECK(p.consumed_end <= kBufferSize);
        CHECK(p.guest_offset < kBufferSize);
      }
    }
  }
}

TEST_CASE("The range memo changes cost, never the plan", "[mipplan]") {
  // The memo exists because the range scan is the planner's only non-trivial
  // cost (~0.53 ms for a 1280x720 level) and it is re-derived on EVERY load,
  // including the promoted single-mip class. That makes it a correctness risk
  // as well as a cost fix, so the equality is pinned rather than assumed.
  MipLevelRangeMemo memo;
  size_t entries_after_first_pass = 0;
  for (int pass = 0; pass < 2; ++pass) {
    for (uint32_t max_level : {0u, 1u, 4u, 8u}) {
      for (auto dims : {std::pair<uint32_t, uint32_t>{1280, 720},
                        std::pair<uint32_t, uint32_t>{256, 256},
                        std::pair<uint32_t, uint32_t>{96, 48},
                        std::pair<uint32_t, uint32_t>{47, 33},
                        std::pair<uint32_t, uint32_t>{16, 16}}) {
        texture_util::TextureGuestLayout layout =
            LayoutFor(dims.first, dims.second, max_level);
        MipLevelPlanInputs in =
            InputsFor(dims.first, dims.second, layout.max_level);
        std::vector<MipLevelPlan> plain;
        std::vector<MipLevelPlan> memoized;
        bool plain_ok = BuildMipLevelPlan(in, layout, &plain, nullptr);
        bool memo_ok = BuildMipLevelPlan(in, layout, &memoized, &memo);
        REQUIRE(plain_ok == memo_ok);
        if (!plain_ok) {
          continue;
        }
        REQUIRE(plain.size() == memoized.size());
        for (size_t i = 0; i < plain.size(); ++i) {
          CHECK(memoized[i].host_level == plain[i].host_level);
          CHECK(memoized[i].guest_offset == plain[i].guest_offset);
          CHECK(memoized[i].pitch_blocks == plain[i].pitch_blocks);
          CHECK(memoized[i].width == plain[i].width);
          CHECK(memoized[i].height == plain[i].height);
          CHECK(memoized[i].px == plain[i].px);
          CHECK(memoized[i].py == plain[i].py);
          CHECK(memoized[i].consumed_begin == plain[i].consumed_begin);
          CHECK(memoized[i].consumed_end == plain[i].consumed_end);
        }
      }
    }
    if (pass == 0) {
      entries_after_first_pass = memo.size();
      // It must actually be caching something, or the equality above is
      // vacuous and the cost fix is not real.
      CHECK(entries_after_first_pass > 0);
    } else {
      // A second identical pass adds no entries: the key covers everything the
      // scan depends on.
      CHECK(memo.size() == entries_after_first_pass);
    }
  }

  // A memo entry must not be reused across different origins or shapes.
  MipLevelRangeMemo origins;
  TextureLoadSourceRange at_zero = origins.Get(0, 0, 4, 4, 256);
  TextureLoadSourceRange at_sub = origins.Get(16, 16, 4, 4, 256);
  CHECK(origins.size() == 2u);
  CHECK(at_zero.valid);
  CHECK(at_sub.valid);
  CHECK(at_zero.begin != at_sub.begin);
  // And a cleared memo re-derives the same answers.
  origins.Clear();
  CHECK(origins.size() == 0u);
  TextureLoadSourceRange again = origins.Get(16, 16, 4, 4, 256);
  CHECK(again.begin == at_sub.begin);
  CHECK(again.end == at_sub.end);
}

TEST_CASE("The exact-range authority rejects impossible extents", "[mipplan]") {
  // Degenerate inputs.
  CHECK_FALSE(
      ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, 0, 4, 256, UINT32_MAX)
          .valid);
  CHECK_FALSE(
      ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, 4, 0, 256, UINT32_MAX)
          .valid);
  CHECK_FALSE(
      ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, 4, 4, 0, UINT32_MAX)
          .valid);
  // Extremes that would wrap a 32-bit intermediate or overflow the int32 the
  // tiled addresser takes: rejected, never wrapped into a bogus in-range span.
  CHECK_FALSE(ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, UINT32_MAX, 4, 256,
                                                    UINT32_MAX)
                  .valid);
  CHECK_FALSE(ComputeStrictRgba8SourceRangeAtOrigin(0, UINT32_MAX, 0, 8, 4, 256,
                                                    UINT32_MAX)
                  .valid);
  CHECK_FALSE(ComputeStrictRgba8SourceRangeAtOrigin(0, 0, UINT32_MAX, 8, 4, 256,
                                                    UINT32_MAX)
                  .valid);
  CHECK_FALSE(ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, 8, 4, UINT32_MAX,
                                                    UINT32_MAX)
                  .valid);
  // A valid extent whose span exceeds the declared physical size is rejected.
  CHECK_FALSE(
      ComputeStrictRgba8SourceRangeAtOrigin(0, 0, 0, 256, 256, 256, 16).valid);
  // And the single-mip entry point is the SAME authority at origin (0,0), so it
  // rejects identically.
  CHECK_FALSE(
      ComputeStrictRgba8SourceRange(0, UINT32_MAX, 4, 256, UINT32_MAX).valid);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
