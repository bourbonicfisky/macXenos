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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_gamma_ramp_classify.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

std::vector<reg::DC_LUT_30_COLOR> MakeTableOf(uint32_t (*f)(uint32_t)) {
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v = f(i);
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  return t;
}

// The ramp CommandProcessor::Initialize() actually seeds, byte for byte.
std::vector<reg::DC_LUT_30_COLOR> MakeReferenceDefaultTable() {
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v = i * 0x3FF / 0xFF;
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  return t;
}

}  // namespace

TEST_CASE("A complete standard identity ramp classifies as identity",
          "[metal][gamma][classify]") {
  // Both exact 8->10 widenings are admitted; neither is guessed at.
  REQUIRE(ClassifyGammaRamp256(MakeTableOf(GammaIdentityReplicate).data()) ==
          GammaRampState::kIdentity);
  REQUIRE(ClassifyGammaRamp256(MakeTableOf(GammaIdentityScale).data()) ==
          GammaRampState::kIdentity);
}

TEST_CASE("An unwritten ramp is never-programmed, not identity",
          "[metal][gamma][classify]") {
  // The distinction matters: an all-zero ramp must not be counted as a title
  // having programmed an identity transfer function.
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  std::memset(t.data(), 0, sizeof(reg::DC_LUT_30_COLOR) * 256);
  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNeverProgrammed);

}

TEST_CASE("A non-identity standard ramp is detected",
          "[metal][gamma][classify]") {
  auto t = MakeTableOf(GammaIdentityReplicate);
  // A single entry off by one LSB is a real transfer-function change. No
  // tolerance is applied anywhere, so this must be caught.
  t[128].color_10_green = t[128].color_10_green + 1;
  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
}

TEST_CASE("A partial update is not yet a completed new ramp",
          "[metal][gamma][classify]") {
  // Mid-update the table is a mixture. Classification happens only at swap, so
  // what this asserts is that the CONTENT HASH changes as partial writes land
  // and settles when they stop -- the census counts a change per distinct
  // complete state, not per register write.
  auto t = MakeTableOf(GammaIdentityReplicate);
  const uint64_t settled = HashGammaRamp256(t.data());
  auto partial = t;
  partial[10].color_10_red = 999;
  REQUIRE(HashGammaRamp256(partial.data()) != settled);
  partial[10] = t[10];
  REQUIRE(HashGammaRamp256(partial.data()) == settled);
}

TEST_CASE("Repeated identical writes are not table changes",
          "[metal][gamma][classify]") {
  auto t = MakeTableOf(GammaIdentityReplicate);
  const uint64_t h1 = HashGammaRamp256(t.data());
  // Rewriting the same values must produce the same hash, so a title that
  // reprograms an unchanged ramp every frame yields zero changes.
  auto again = MakeTableOf(GammaIdentityReplicate);
  REQUIRE(HashGammaRamp256(again.data()) == h1);
}

TEST_CASE("Non-identity restored to identity is a real transition",
          "[metal][gamma][classify]") {
  auto t = MakeTableOf(GammaIdentityReplicate);
  const uint64_t identity_hash = HashGammaRamp256(t.data());
  t[5].color_10_blue = 1023;
  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
  const uint64_t altered_hash = HashGammaRamp256(t.data());
  REQUIRE(altered_hash != identity_hash);

  auto restored = MakeTableOf(GammaIdentityReplicate);
  REQUIRE(ClassifyGammaRamp256(restored.data()) == GammaRampState::kIdentity);
  REQUIRE(HashGammaRamp256(restored.data()) == identity_hash);
  REQUIRE(HashGammaRamp256(restored.data()) != altered_hash);
}

TEST_CASE("Standard and PWL content stay independently identified",
          "[metal][gamma][classify]") {
  // A non-identity standard table must not make an untouched PWL ramp look
  // engaged, or a census could attribute engagement to the wrong mode. PWL is
  // identified by CONTENT HASH only -- the PWL classifier is quarantined, so
  // no PWL identity claim appears here.
  auto t = MakeTableOf(GammaIdentityReplicate);
  t[200].color_10_red = 7;
  std::vector<reg::DC_LUT_PWL_DATA> p(128 * 3);
  std::memset(p.data(), 0, sizeof(reg::DC_LUT_PWL_DATA) * 128 * 3);

  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
  REQUIRE(HashGammaRamp256(t.data()) != HashGammaRampPWL(p.data()));
}

TEST_CASE("The emulator's own default ramp classifies as grounded-linear identity",
          "[metal][gamma][classify]") {
  // THE REGRESSION THIS FILE EXISTS TO PREVENT. An earlier classifier tested
  // only bit-replication and rational scaling, so CommandProcessor's seeded
  // default -- which the emulator itself calls linear -- came back
  // NON-IDENTITY, manufacturing engagement out of a ramp no guest had touched.
  GammaIdentityVariant variant = GammaIdentityVariant::kNone;
  auto t = MakeReferenceDefaultTable();
  REQUIRE(ClassifyGammaRamp256(t.data(), &variant) ==
          GammaRampState::kIdentity);
  REQUIRE(variant == GammaIdentityVariant::kLinear);
}

TEST_CASE("The three identity formulas are genuinely distinct",
          "[metal][gamma][classify]") {
  // If these ever collapsed onto each other, testing all three would be
  // pointless and the grounded one could be silently replaced by a variant.
  uint32_t linear_vs_replicate = 0, linear_vs_scale = 0;
  for (uint32_t i = 0; i < 256; ++i) {
    if (GammaIdentityLinear(i) != GammaIdentityReplicate(i)) {
      ++linear_vs_replicate;
    }
    if (GammaIdentityLinear(i) != GammaIdentityScale(i)) {
      ++linear_vs_scale;
    }
  }
  REQUIRE(linear_vs_replicate == 126);
  REQUIRE(linear_vs_scale == 126);
  // Endpoints must still pin exactly under all three, or a ramp could be
  // "identity" while changing black or white.
  REQUIRE(GammaIdentityLinear(0) == 0);
  REQUIRE(GammaIdentityLinear(255) == 1023);
}

TEST_CASE("Accepted variants are reported as variants, not as the reference",
          "[metal][gamma][classify]") {
  GammaIdentityVariant variant = GammaIdentityVariant::kNone;
  REQUIRE(ClassifyGammaRamp256(MakeTableOf(GammaIdentityReplicate).data(),
                               &variant) == GammaRampState::kIdentity);
  REQUIRE(variant == GammaIdentityVariant::kReplicate);
  REQUIRE(ClassifyGammaRamp256(MakeTableOf(GammaIdentityScale).data(),
                               &variant) == GammaRampState::kIdentity);
  REQUIRE(variant == GammaIdentityVariant::kScale);
}

TEST_CASE("A real transfer curve is still non-identity under all three",
          "[metal][gamma][classify]") {
  // Modelled on the curve actually observed in LEGO Star Wars II: endpoints
  // pinned, midtones well below linear.
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v = uint32_t(1023.0 * std::pow(i / 255.0, 1.2) + 0.5);
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  GammaIdentityVariant variant = GammaIdentityVariant::kLinear;
  REQUIRE(ClassifyGammaRamp256(t.data(), &variant) ==
          GammaRampState::kNonIdentity);
  REQUIRE(variant == GammaIdentityVariant::kNone);
}

TEST_CASE("DC_LUTA_CONTROL selects the mode, with zero meaning standard",
          "[metal][gamma][selector]") {
  // Grounded in registers.h: the PWL increment "is nonzero, otherwise the
  // 256-entry table should be used instead".
  REQUIRE(GammaRampModeFromLutaControl(0x00000000u) ==
          GammaRampMode::kStandard256);
  // The two values the register documentation actually exhibits.
  REQUIRE(GammaRampModeFromLutaControl(0x00000003u) == GammaRampMode::kPWL);

  // Every other nonzero value must select PWL too, so an unexpected per-channel
  // field layout cannot be silently misread as "standard".
  for (uint32_t v : {0x1u, 0x2u, 0x4u, 0x7u, 0x100u, 0x00030000u, 0x80000000u}) {
    REQUIRE(GammaRampModeFromLutaControl(v) == GammaRampMode::kPWL);
  }
}

TEST_CASE("An unwritten selector is standard, not unknown",
          "[metal][gamma][selector]") {
  // register_table.inc records a title that never writes DC_LUTA_CONTROL yet
  // still expects the behavior of 0. Absence must therefore resolve to
  // standard rather than to an ambiguous third state that would block the gate.
  REQUIRE(GammaRampModeFromLutaControl(0u) == GammaRampMode::kStandard256);
}
}  // namespace metal
}  // namespace gpu
}  // namespace xe
