/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Synthetic coverage of the EXACT guest gamma-ramp classifier used by the
// Contract 1 engagement census. Exercises the production helpers directly.
//
// The property under test is that engagement cannot be manufactured: an
// unwritten ramp, an identity ramp, or a repeated identical write must never
// look like a real non-identity ramp change.

#include <cstdint>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_gamma_ramp_classify.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

std::vector<reg::DC_LUT_30_COLOR> MakeTable(bool scale_variant) {
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v =
        scale_variant ? GammaIdentityScale(i) : GammaIdentityReplicate(i);
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  return t;
}

std::vector<reg::DC_LUT_PWL_DATA> MakePWL(bool scale_variant) {
  std::vector<reg::DC_LUT_PWL_DATA> p(128 * 3);
  for (uint32_t i = 0; i < 128 * 3; ++i) {
    const uint32_t seg = i % 128u;
    const uint32_t base = (scale_variant ? GammaIdentityScale(seg << 1)
                                         : GammaIdentityReplicate(seg << 1))
                          << 6;
    p[i].value = 0;
    p[i].base = base;
    p[i].delta = 0;
  }
  return p;
}

}  // namespace

TEST_CASE("A complete standard identity ramp classifies as identity",
          "[metal][gamma][classify]") {
  // Both exact 8->10 widenings are admitted; neither is guessed at.
  REQUIRE(ClassifyGammaRamp256(MakeTable(false).data()) ==
          GammaRampState::kIdentity);
  REQUIRE(ClassifyGammaRamp256(MakeTable(true).data()) ==
          GammaRampState::kIdentity);
}

TEST_CASE("A complete PWL identity ramp classifies as identity",
          "[metal][gamma][classify]") {
  REQUIRE(ClassifyGammaRampPWL(MakePWL(false).data()) ==
          GammaRampState::kIdentity);
  REQUIRE(ClassifyGammaRampPWL(MakePWL(true).data()) ==
          GammaRampState::kIdentity);
}

TEST_CASE("An unwritten ramp is never-programmed, not identity",
          "[metal][gamma][classify]") {
  // The distinction matters: an all-zero ramp must not be counted as a title
  // having programmed an identity transfer function.
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  std::memset(t.data(), 0, sizeof(reg::DC_LUT_30_COLOR) * 256);
  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNeverProgrammed);

  std::vector<reg::DC_LUT_PWL_DATA> p(128 * 3);
  std::memset(p.data(), 0, sizeof(reg::DC_LUT_PWL_DATA) * 128 * 3);
  REQUIRE(ClassifyGammaRampPWL(p.data()) == GammaRampState::kNeverProgrammed);
}

TEST_CASE("A non-identity standard ramp is detected",
          "[metal][gamma][classify]") {
  auto t = MakeTable(false);
  // A single entry off by one LSB is a real transfer-function change. No
  // tolerance is applied anywhere, so this must be caught.
  t[128].color_10_green = t[128].color_10_green + 1;
  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
}

TEST_CASE("A non-identity PWL ramp is detected", "[metal][gamma][classify]") {
  auto p = MakePWL(false);
  p[64].base = p[64].base + 64;
  REQUIRE(ClassifyGammaRampPWL(p.data()) == GammaRampState::kNonIdentity);
}

TEST_CASE("A partial update is not yet a completed new ramp",
          "[metal][gamma][classify]") {
  // Mid-update the table is a mixture. Classification happens only at swap, so
  // what this asserts is that the CONTENT HASH changes as partial writes land
  // and settles when they stop -- the census counts a change per distinct
  // complete state, not per register write.
  auto t = MakeTable(false);
  const uint64_t settled = HashGammaRamp256(t.data());
  auto partial = t;
  partial[10].color_10_red = 999;
  REQUIRE(HashGammaRamp256(partial.data()) != settled);
  partial[10] = t[10];
  REQUIRE(HashGammaRamp256(partial.data()) == settled);
}

TEST_CASE("Repeated identical writes are not table changes",
          "[metal][gamma][classify]") {
  auto t = MakeTable(false);
  const uint64_t h1 = HashGammaRamp256(t.data());
  // Rewriting the same values must produce the same hash, so a title that
  // reprograms an unchanged ramp every frame yields zero changes.
  auto again = MakeTable(false);
  REQUIRE(HashGammaRamp256(again.data()) == h1);
}

TEST_CASE("Non-identity restored to identity is a real transition",
          "[metal][gamma][classify]") {
  auto t = MakeTable(false);
  const uint64_t identity_hash = HashGammaRamp256(t.data());
  t[5].color_10_blue = 1023;
  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
  const uint64_t altered_hash = HashGammaRamp256(t.data());
  REQUIRE(altered_hash != identity_hash);

  auto restored = MakeTable(false);
  REQUIRE(ClassifyGammaRamp256(restored.data()) == GammaRampState::kIdentity);
  REQUIRE(HashGammaRamp256(restored.data()) == identity_hash);
  REQUIRE(HashGammaRamp256(restored.data()) != altered_hash);
}

TEST_CASE("Standard and PWL state are independent",
          "[metal][gamma][classify]") {
  // A non-identity standard table must not make an untouched PWL ramp look
  // engaged, or a census could attribute engagement to the wrong mode.
  auto t = MakeTable(false);
  t[200].color_10_red = 7;
  std::vector<reg::DC_LUT_PWL_DATA> p(128 * 3);
  std::memset(p.data(), 0, sizeof(reg::DC_LUT_PWL_DATA) * 128 * 3);

  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
  REQUIRE(ClassifyGammaRampPWL(p.data()) == GammaRampState::kNeverProgrammed);
  REQUIRE(HashGammaRamp256(t.data()) != HashGammaRampPWL(p.data()));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
