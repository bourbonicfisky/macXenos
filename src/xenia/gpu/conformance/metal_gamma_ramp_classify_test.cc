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

}  // namespace

TEST_CASE("A complete standard identity ramp classifies as identity",
          "[metal][gamma][classify]") {
  // Both exact 8->10 widenings are admitted; neither is guessed at.
  REQUIRE(ClassifyGammaRamp256(MakeTable(false).data()) ==
          GammaRampState::kIdentity);
  REQUIRE(ClassifyGammaRamp256(MakeTable(true).data()) ==
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
  auto t = MakeTable(false);
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

TEST_CASE("Standard and PWL content stay independently identified",
          "[metal][gamma][classify]") {
  // A non-identity standard table must not make an untouched PWL ramp look
  // engaged, or a census could attribute engagement to the wrong mode. PWL is
  // identified by CONTENT HASH only -- the PWL classifier is quarantined, so
  // no PWL identity claim appears here.
  auto t = MakeTable(false);
  t[200].color_10_red = 7;
  std::vector<reg::DC_LUT_PWL_DATA> p(128 * 3);
  std::memset(p.data(), 0, sizeof(reg::DC_LUT_PWL_DATA) * 128 * 3);

  REQUIRE(ClassifyGammaRamp256(t.data()) == GammaRampState::kNonIdentity);
  REQUIRE(HashGammaRamp256(t.data()) != HashGammaRampPWL(p.data()));
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
