/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-depth-semantics task 3.1: the host-depth ownership
// predicate's full decision matrix, on the PRODUCTION pure functions the
// backend calls (metal_depth_semantics_flags.h) — not a replica.
//
// The reviewer-specified matrix:
//   D24FS8 + PS conversion off -> true
//   D24FS8 + PS conversion on  -> false   (stage interlock, fails closed)
//   D24S8  -> false                        (never; no host precision to keep)
// plus the gate row: everything false while XE_METAL_HOST_DEPTH is off.

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_depth_semantics_flags.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr auto kD24S8 = xenos::DepthRenderTargetFormat::kD24S8;
constexpr auto kD24FS8 = xenos::DepthRenderTargetFormat::kD24FS8;

TEST_CASE("host-depth predicate matrix", "[depth_semantics][host_depth]") {
  // Gate off: always false, whatever else is set.
  CHECK_FALSE(MetalHostDepthEncodingDifferent(false, kD24FS8, false));
  CHECK_FALSE(MetalHostDepthEncodingDifferent(false, kD24FS8, true));
  CHECK_FALSE(MetalHostDepthEncodingDifferent(false, kD24S8, false));

  // Gate on — the reviewer matrix:
  CHECK(MetalHostDepthEncodingDifferent(true, kD24FS8, false));
  CHECK_FALSE(MetalHostDepthEncodingDifferent(true, kD24FS8, true));
  CHECK_FALSE(MetalHostDepthEncodingDifferent(true, kD24S8, false));
  CHECK_FALSE(MetalHostDepthEncodingDifferent(true, kD24S8, true));
}

TEST_CASE("F24 mode parse feeds the interlock correctly",
          "[depth_semantics][host_depth]") {
  // The predicate's f24_ps_engaged input comes from ParseMetalF24PsMode on
  // the SAME env string the command processor reads: every engaged spelling
  // must disable host-depth; off/unset/malformed must not.
  for (const char* engaged : {"1", "trunc", "round"}) {
    INFO(engaged);
    CHECK(MetalF24PsEngaged(ParseMetalF24PsMode(engaged)));
    CHECK_FALSE(MetalHostDepthEncodingDifferent(
        true, kD24FS8, MetalF24PsEngaged(ParseMetalF24PsMode(engaged))));
  }
  for (const char* off :
       {static_cast<const char*>(nullptr), "", "0"}) {
    CHECK_FALSE(MetalF24PsEngaged(ParseMetalF24PsMode(off)));
    CHECK(MetalHostDepthEncodingDifferent(
        true, kD24FS8, MetalF24PsEngaged(ParseMetalF24PsMode(off))));
  }
  // Malformed is its own token (caller warns once) and behaves as off.
  CHECK(ParseMetalF24PsMode("bogus") == MetalF24PsMode::kMalformed);
  CHECK_FALSE(MetalF24PsEngaged(ParseMetalF24PsMode("bogus")));
}

TEST_CASE("Effective F24 mode: env absent lets the shared cvar drive",
          "[depth_semantics][host_depth]") {
  CHECK(EffectiveMetalF24PsMode(nullptr, true) == MetalF24PsMode::kCvarDriven);
  CHECK(EffectiveMetalF24PsMode(nullptr, false) == MetalF24PsMode::kOff);
}

TEST_CASE("Effective F24 mode: env present overrides the shared cvar",
          "[depth_semantics][host_depth]") {
  // Explicit "0" force-disables even when the shared cvar is true.
  CHECK(EffectiveMetalF24PsMode("0", true) == MetalF24PsMode::kOff);
  CHECK(EffectiveMetalF24PsMode("", true) == MetalF24PsMode::kOff);
  // "1" is cvar-driven regardless of the cvar's value (mirrors
  // ParseMetalF24PsMode; the cvar itself is read later for round-vs-trunc).
  CHECK(EffectiveMetalF24PsMode("1", true) == MetalF24PsMode::kCvarDriven);
  CHECK(EffectiveMetalF24PsMode("1", false) == MetalF24PsMode::kCvarDriven);
  // trunc / round force a mode independent of the shared cvar.
  CHECK(EffectiveMetalF24PsMode("trunc", false) ==
        MetalF24PsMode::kTruncating);
  CHECK(EffectiveMetalF24PsMode("round", false) == MetalF24PsMode::kRounding);
  // Malformed is unaffected by the cvar value.
  CHECK(EffectiveMetalF24PsMode("bogus", true) == MetalF24PsMode::kMalformed);
  CHECK(EffectiveMetalF24PsMode("bogus", false) ==
        MetalF24PsMode::kMalformed);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
