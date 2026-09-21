/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// port-xenios-tessellation tasks 4.3/4.5/4.6 — device-free route contracts:
//
//   1. the XE_METAL_TESSELLATION gate parse matrix (strict, fail-closed,
//      exact depth-semantics-flag conventions);
//   2. the index/factor-stream admissibility predicate that pins the design
//      D5 endian DECISION (CPU-swap arena staging + published kNone);
//   3. the factor-value endian ORACLE: an asymmetric-byte guest factor pushed
//      through the REAL production staging primitive (SwapIndicesInto) and
//      the modeled in-shader transform (XeEndianSwap32 with the published
//      endian, asfloat, +1, clamp — tessellation_adaptive.vs.hlsl:4-16) must
//      produce the exact expected factor, and BOTH failure modes — double
//      swap (publishing the guest endian over a pre-swapped stream) and
//      un-swap (raw guest stream with kNone) — must fail loudly.

#include <cstdint>
#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_primitive_processor.h"
#include "xenia/gpu/metal/metal_tessellation_flags.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

TEST_CASE("XE_METAL_TESSELLATION gate parse: default on, exact 0 is the rollback",
          "[l0][tessellation][gate]") {
  // Promoted default-ON by implement-metal-host-tessellation (2026-09-20).
  CHECK(ParseMetalTessellationGate(nullptr) == MetalTessellationGate::kOn);
  CHECK(ParseMetalTessellationGate("") == MetalTessellationGate::kOn);
  CHECK(ParseMetalTessellationGate("1") == MetalTessellationGate::kOn);
  CHECK(ParseMetalTessellationGate("0") == MetalTessellationGate::kOff);
  // Everything else is malformed: warn-once at the caller, behaviorally the
  // default (on).
  CHECK(ParseMetalTessellationGate("2") == MetalTessellationGate::kMalformed);
  CHECK(ParseMetalTessellationGate("on") == MetalTessellationGate::kMalformed);
  CHECK(ParseMetalTessellationGate("true") ==
        MetalTessellationGate::kMalformed);
  CHECK(ParseMetalTessellationGate(" 1") == MetalTessellationGate::kMalformed);
  CHECK(ParseMetalTessellationGate("10") == MetalTessellationGate::kMalformed);
  CHECK(ParseMetalTessellationGate("00") == MetalTessellationGate::kMalformed);

  CHECK(MetalTessellationGateEngaged(MetalTessellationGate::kOn));
  CHECK_FALSE(MetalTessellationGateEngaged(MetalTessellationGate::kOff));
  CHECK(MetalTessellationGateEngaged(MetalTessellationGate::kMalformed));
}

TEST_CASE("tessellation index stream admissibility matches the staging helper",
          "[l0][tessellation][endian]") {
  // 16-bit streams: the shared PrimitiveProcessor normalizes the endian to
  // kNone/k8in16 before the route sees it; both are staged exactly.
  CHECK(IsMetalTessellationIndexStreamSupported(xenos::IndexFormat::kInt16,
                                                xenos::Endian::kNone));
  CHECK(IsMetalTessellationIndexStreamSupported(xenos::IndexFormat::kInt16,
                                                xenos::Endian::k8in16));
  // 32-bit streams (including adaptive factor streams): SwapIndicesInto's
  // 32-bit branch is the full byte reversal — exact for kNone/k8in32 only.
  CHECK(IsMetalTessellationIndexStreamSupported(xenos::IndexFormat::kInt32,
                                                xenos::Endian::kNone));
  CHECK(IsMetalTessellationIndexStreamSupported(xenos::IndexFormat::kInt32,
                                                xenos::Endian::k8in32));
  CHECK_FALSE(IsMetalTessellationIndexStreamSupported(
      xenos::IndexFormat::kInt32, xenos::Endian::k8in16));
  CHECK_FALSE(IsMetalTessellationIndexStreamSupported(
      xenos::IndexFormat::kInt32, xenos::Endian::k16in32));
}

// Models exactly what the prebuilt adaptive tessellation VS does to each
// fetched factor word (shaders/tessellation_adaptive.vs.hlsl:4-16):
// XeEndianSwap32 with the published xe_vertex_index_endian, asfloat, +1.0,
// clamp to the published range.
float EffectiveAdaptiveFactor(uint32_t bound_word,
                              xenos::Endian published_endian, float range_min,
                              float range_max) {
  uint32_t swapped = xenos::GpuSwap(bound_word, published_endian);
  float factor;
  std::memcpy(&factor, &swapped, sizeof(factor));
  factor += 1.0f;
  if (!(factor >= range_min)) factor = range_min;  // NaN -> min, like clamp.
  if (factor > range_max) factor = range_max;
  return factor;
}

TEST_CASE("adaptive factor endian oracle: CPU-swap + published kNone is exact,"
          " double-/un-swap fail loudly",
          "[l0][tessellation][endian]") {
  // A factor with four DISTINCT nonzero bytes so any wrong swap moves it.
  // asfloat(0x40913C5D) ~= 4.5386; expected effective factor = 5.5386...
  const uint32_t kFactorBits = 0x40913C5Du;
  float factor_value;
  std::memcpy(&factor_value, &kFactorBits, sizeof(factor_value));
  const float kRangeMin = 1.0f;   // VGT_HOS_MIN_TESS_LEVEL 0 + 1
  const float kRangeMax = 64.0f;  // VGT_HOS_MAX_TESS_LEVEL 63 + 1
  const float kExpected = factor_value + 1.0f;
  REQUIRE(kExpected > kRangeMin);
  REQUIRE(kExpected < kRangeMax);

  // The guest DMA stream is big-endian 32-bit words (k8in32): the raw word a
  // little-endian host reads from guest RAM is the byte reversal.
  const uint32_t kGuestRawWord = xe::byte_swap(kFactorBits);

  // PRODUCTION path: SwapIndicesInto (the same staging primitive the draw
  // uses) materializes the stream into the arena...
  uint32_t arena_word = 0;
  SwapIndicesInto(&arena_word, &kGuestRawWord, 1, xenos::IndexFormat::kInt32,
                  xenos::Endian::k8in32);
  CHECK(arena_word == kFactorBits);
  // ...and the route publishes vertex_index_endian = kNone, so the in-shader
  // swap is the identity: the effective factor is exact.
  CHECK(EffectiveAdaptiveFactor(arena_word, xenos::Endian::kNone, kRangeMin,
                                kRangeMax) == kExpected);

  // FAILURE MODE 1 — double swap: the arena stream is already host-order,
  // but the guest endian is published anyway. The shader swaps AGAIN and the
  // factor collapses to garbage (here: a huge float clamped to range max).
  CHECK(EffectiveAdaptiveFactor(arena_word, xenos::Endian::k8in32, kRangeMin,
                                kRangeMax) != kExpected);

  // FAILURE MODE 2 — un-swap: the raw guest stream is bound directly while
  // kNone is published; nobody swaps and the factor is garbage.
  CHECK(EffectiveAdaptiveFactor(kGuestRawWord, xenos::Endian::kNone, kRangeMin,
                                kRangeMax) != kExpected);

  // XeniOS's coherent alternative (raw stream + TRUE endian published) is
  // also value-exact — the corruption is only in MIXING conventions. Pinning
  // this documents that the D5 decision is about coherence, not correctness
  // of one shape.
  CHECK(EffectiveAdaptiveFactor(kGuestRawWord, xenos::Endian::k8in32,
                                kRangeMin, kRangeMax) == kExpected);
}

TEST_CASE("16-bit tessellation index staging round-trips k8in16",
          "[l0][tessellation][endian]") {
  const uint16_t kGuestRaw[3] = {xe::byte_swap(uint16_t(7)),
                                 xe::byte_swap(uint16_t(300)),
                                 xe::byte_swap(uint16_t(65535))};
  uint16_t staged[3] = {};
  SwapIndicesInto(staged, kGuestRaw, 3, xenos::IndexFormat::kInt16,
                  xenos::Endian::k8in16);
  CHECK(staged[0] == 7);
  CHECK(staged[1] == 300);
  CHECK(staged[2] == 65535);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
