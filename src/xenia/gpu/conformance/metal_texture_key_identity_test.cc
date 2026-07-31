/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Legacy Metal texture cache identity
// (openspec/changes/implement-metal-texture-key-mip-range, tasks 3.1-3.4).
//
// Exercises the REAL production key composition (metal::MakeMetalTextureKey)
// together with the REAL normalization
// (texture_util::GetSubresourcesFromFetchConstant). Neither is re-implemented
// here: a re-implemented key agrees with itself forever while production drifts.
//
// The measured defect these cases lock down: the key omitted the mip level
// range, so a fetch asking for level 1 could be served an entry built for level
// 0 — an entry that, because the upload only builds a chain when the effective
// maximum is non-zero, is single-level and does not contain level 1 at all.
// 52 such mis-serves were observed across the corpus, 24 of them harmful.

#include <cstdint>
#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_texture_key.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::MakeMetalTextureKey;
using metal::MetalTextureKeyInputs;

// A 2D fetch shaped like the Red Dead surfaces that collided: 128x128, a real
// base page and a real mip page, so the normalization does not collapse the
// range.
xenos::xe_gpu_texture_fetch_t MakeFetch(uint32_t base_page, uint32_t mip_page,
                                        uint32_t mip_min, uint32_t mip_max,
                                        uint32_t width_minus_1 = 127,
                                        uint32_t height_minus_1 = 127) {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  fetch.base_address = base_page;
  fetch.mip_address = mip_page;
  fetch.mip_min_level = mip_min;
  fetch.mip_max_level = mip_max;
  fetch.size_2d.width = width_minus_1;
  fetch.size_2d.height = height_minus_1;
  return fetch;
}

// Normalized effective range, exactly as production derives it.
void Normalize(const xenos::xe_gpu_texture_fetch_t& fetch, uint32_t* min_out,
               uint32_t* max_out) {
  texture_util::GetSubresourcesFromFetchConstant(
      fetch, nullptr, nullptr, nullptr, nullptr, nullptr, min_out, max_out);
}

// Key for a fetch, through the same two production pieces the cache uses.
uint64_t KeyFor(const xenos::xe_gpu_texture_fetch_t& fetch) {
  uint32_t min_level = 0, max_level = 0;
  Normalize(fetch, &min_level, &max_level);
  MetalTextureKeyInputs inputs;
  inputs.base = 0x03252000;
  inputs.width = 128;
  inputs.height = 128;
  inputs.pitch_texels = 128;
  inputs.format_tiled_endian = 0xF0000;
  inputs.swizzle = 0x688;
  inputs.mip_min_level = min_level;
  inputs.mip_max_level = max_level;
  return MakeMetalTextureKey(inputs);
}

MetalTextureKeyInputs BaseInputs() {
  MetalTextureKeyInputs inputs;
  inputs.base = 0x03252000;
  inputs.width = 128;
  inputs.height = 128;
  inputs.pitch_texels = 128;
  inputs.format_tiled_endian = 0xF0000;
  inputs.swizzle = 0x688;
  return inputs;
}

}  // namespace

TEST_CASE("Different normalized ranges produce different keys",
          "[texture_key]") {
  MetalTextureKeyInputs a = BaseInputs();
  MetalTextureKeyInputs b = BaseInputs();

  SECTION("the measured Red Dead collision shape: lv0..0 vs lv1..1") {
    a.mip_min_level = 0;
    a.mip_max_level = 0;
    b.mip_min_level = 1;
    b.mip_max_level = 1;
    CHECK(MakeMetalTextureKey(a) != MakeMetalTextureKey(b));
  }

  SECTION("lv1..1 vs lv2..2") {
    a.mip_min_level = 1;
    a.mip_max_level = 1;
    b.mip_min_level = 2;
    b.mip_max_level = 2;
    CHECK(MakeMetalTextureKey(a) != MakeMetalTextureKey(b));
  }

  SECTION("the minimum alone distinguishes") {
    a.mip_min_level = 0;
    a.mip_max_level = 2;
    b.mip_min_level = 1;
    b.mip_max_level = 2;
    CHECK(MakeMetalTextureKey(a) != MakeMetalTextureKey(b));
  }

  SECTION("the maximum alone distinguishes") {
    a.mip_min_level = 0;
    a.mip_max_level = 0;
    b.mip_min_level = 0;
    b.mip_max_level = 2;
    CHECK(MakeMetalTextureKey(a) != MakeMetalTextureKey(b));
  }

  SECTION("an identical range is an identical key") {
    a.mip_min_level = 1;
    a.mip_max_level = 3;
    b.mip_min_level = 1;
    b.mip_max_level = 3;
    CHECK(MakeMetalTextureKey(a) == MakeMetalTextureKey(b));
  }
}

TEST_CASE("Equivalent ranges share one entry", "[texture_key][normalization]") {
  // The whole reason the key takes the NORMALIZED range: raw register levels
  // that describe the same host texture must not split it.

  SECTION("no mip page collapses any raw range to the base level") {
    // Measured: every Just Cause 2 "collision" in the census had mip=00000000.
    // Raw fields said lv0..2 and lv0..0; both are the same single-level
    // texture, and keying raw bits would have manufactured a duplicate entry.
    auto wide = MakeFetch(/*base_page=*/0x3252, /*mip_page=*/0, 0, 2);
    auto narrow = MakeFetch(/*base_page=*/0x3252, /*mip_page=*/0, 0, 0);

    uint32_t wmin = 0, wmax = 0, nmin = 0, nmax = 0;
    Normalize(wide, &wmin, &wmax);
    Normalize(narrow, &nmin, &nmax);
    INFO("wide normalized " << wmin << ".." << wmax << ", narrow " << nmin
                            << ".." << nmax);
    CHECK(wmin == 0);
    CHECK(wmax == 0);
    CHECK(nmin == 0);
    CHECK(nmax == 0);
    CHECK(KeyFor(wide) == KeyFor(narrow));
  }

  SECTION("both ends are bounded by the size-derived maximum") {
    // A 128x128 texture has no level 9. Raw fields asking for one normalize to
    // the same range as fields asking for the real maximum.
    auto absurd = MakeFetch(0x3252, 0x3260, 0, 9);
    auto real_max = MakeFetch(0x3252, 0x3260, 0, 7);
    uint32_t amin = 0, amax = 0, rmin = 0, rmax = 0;
    Normalize(absurd, &amin, &amax);
    Normalize(real_max, &rmin, &rmax);
    INFO("absurd " << amin << ".." << amax << ", real " << rmin << ".." << rmax);
    CHECK(amax == rmax);
    CHECK(KeyFor(absurd) == KeyFor(real_max));
  }
}

TEST_CASE("Excluded fields are not direct key inputs",
          "[texture_key][exclusions]") {
  // The contract is exclusion as a DIRECT input. Indirect influence through the
  // normalized range is correct and is asserted separately below.

  SECTION("the key struct carries no excluded field") {
    // Two inputs identical in every member produce the same key; there is no
    // hidden field for mip address, packed mips, stacked identity or depth to
    // vary. This pins the composition itself.
    MetalTextureKeyInputs a = BaseInputs();
    MetalTextureKeyInputs b = BaseInputs();
    a.mip_min_level = b.mip_min_level = 1;
    a.mip_max_level = b.mip_max_level = 2;
    CHECK(MakeMetalTextureKey(a) == MakeMetalTextureKey(b));
  }

  SECTION("packed_mips does not reach the normalized range") {
    auto packed = MakeFetch(0x3252, 0x3260, 1, 1);
    packed.packed_mips = 1;
    auto unpacked = MakeFetch(0x3252, 0x3260, 1, 1);
    unpacked.packed_mips = 0;
    uint32_t pmin = 0, pmax = 0, umin = 0, umax = 0;
    Normalize(packed, &pmin, &pmax);
    Normalize(unpacked, &umin, &umax);
    CHECK(pmin == umin);
    CHECK(pmax == umax);
    CHECK(KeyFor(packed) == KeyFor(unpacked));
  }

  SECTION("mip address DOES legitimately change the key via the range") {
    // Not a leak: one fetch has no mip page, so its effective range collapses
    // to the base level while the other resolves wider. Requiring the same key
    // here would override the shared helper and reintroduce the
    // identity/content disagreement this capability closes.
    auto no_mip_page = MakeFetch(0x3252, 0x0000, 1, 2);
    auto with_mip_page = MakeFetch(0x3252, 0x3260, 1, 2);
    uint32_t nmin = 0, nmax = 0, wmin = 0, wmax = 0;
    Normalize(no_mip_page, &nmin, &nmax);
    Normalize(with_mip_page, &wmin, &wmax);
    INFO("no-page " << nmin << ".." << nmax << ", with-page " << wmin << ".."
                    << wmax);
    CHECK(nmax == 0);
    CHECK(wmax > 0);
    CHECK(KeyFor(no_mip_page) != KeyFor(with_mip_page));
  }
}

TEST_CASE("Raw register levels are not a substitute for the normalized range",
          "[texture_key][normalization]") {
  // Mutation target: production passing fetch.mip_min_level / mip_max_level
  // straight through. This case fails if it ever does, because the raw fields
  // here differ from the normalized result.
  auto fetch = MakeFetch(/*base_page=*/0x3252, /*mip_page=*/0, /*min=*/1,
                         /*max=*/2);
  uint32_t norm_min = 0, norm_max = 0;
  Normalize(fetch, &norm_min, &norm_max);

  INFO("raw 1..2, normalized " << norm_min << ".." << norm_max);
  REQUIRE(norm_min == 0);
  REQUIRE(norm_max == 0);
  CHECK(uint32_t(fetch.mip_min_level) != norm_min);
  CHECK(uint32_t(fetch.mip_max_level) != norm_max);

  MetalTextureKeyInputs normalized = BaseInputs();
  normalized.mip_min_level = norm_min;
  normalized.mip_max_level = norm_max;

  MetalTextureKeyInputs raw = BaseInputs();
  raw.mip_min_level = uint32_t(fetch.mip_min_level);
  raw.mip_max_level = uint32_t(fetch.mip_max_level);

  CHECK(MakeMetalTextureKey(normalized) != MakeMetalTextureKey(raw));
  CHECK(KeyFor(fetch) == MakeMetalTextureKey(normalized));
  CHECK(KeyFor(fetch) != MakeMetalTextureKey(raw));
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
