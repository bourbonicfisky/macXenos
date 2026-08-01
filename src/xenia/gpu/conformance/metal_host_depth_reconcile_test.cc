/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-depth-semantics 3.3 (L0): the two-source reconciliation
// contract on the PRODUCTION oracle (metal_host_depth_reconcile.h).
//
//   pack(host) == guest  ->  host wins, BIT-EXACT (no decode round-trip)
//   pack(host) != guest  ->  guest wins (materialized 20e4, canonical *0.5)
// proven under BOTH rounding modes, including a midpoint value whose winner
// FLIPS with the mode — the case that proves the mode participates rather
// than being dead plumbing.

#include <cstdint>
#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_host_depth_reconcile.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

uint32_t BitsOf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

uint32_t Word(uint32_t d24, uint32_t stencil = 0xA5u) {
  return (d24 << 8) | stencil;
}

TEST_CASE("host wins when its pack reproduces the guest word, bit-exact",
          "[depth_semantics][host_depth][reconcile]") {
  for (bool rne : {false, true}) {
    INFO("round_to_nearest_even=" << rne);
    for (uint32_t code = 1; code < (1u << 24); code += 4099u) {
      // The host value the attachment would hold for this code, plus the
      // extra precision a host float32 legitimately carries beyond 20e4:
      // nudge the mantissa by one f32 ULP — still packs to the same code
      // (both modes: one f32 ULP is far below one 20e4 ULP for normals).
      float exact_host = xenos::Float20e4To32(code) * 0.5f;
      uint32_t nudged_bits = BitsOf(exact_host) + 1;
      float nudged_host;
      std::memcpy(&nudged_host, &nudged_bits, sizeof(nudged_host));
      if (MetalHostDepthPack(nudged_host, rne) != code) {
        continue;  // ULP crossed a code boundary (denormal floor) — skip
      }
      float out = MetalHostDepthReconcile(Word(code), nudged_host, rne);
      // Host preserved with ITS bits — not re-decoded to the code center.
      REQUIRE(BitsOf(out) == nudged_bits);
    }
  }
}

TEST_CASE("guest wins on any mismatch",
          "[depth_semantics][host_depth][reconcile]") {
  for (bool rne : {false, true}) {
    INFO("round_to_nearest_even=" << rne);
    for (uint32_t code = 0; code < (1u << 24); code += 9973u) {
      float host = xenos::Float20e4To32(code) * 0.5f;
      // Clobber the guest word by one code — the guest wrote the range.
      uint32_t clobbered = (code + 1) & 0xFFFFFFu;
      float out = MetalHostDepthReconcile(Word(clobbered), host, rne);
      float expected = xenos::Float20e4To32(clobbered) * 0.5f;
      REQUIRE(BitsOf(out) == BitsOf(expected));
    }
  }
}

TEST_CASE("the rounding mode decides the winner at a code midpoint",
          "[depth_semantics][host_depth][reconcile]") {
  // Adjacent normal codes with an ODD lower mantissa: the exact midpoint
  // packs DOWN under truncation and UP (to the even neighbor) under RNE.
  const uint32_t lower = (5u << 20) | 0x12345u;  // odd mantissa
  const uint32_t upper = lower + 1;              // even mantissa
  float f_lower = xenos::Float20e4To32(lower);
  float f_upper = xenos::Float20e4To32(upper);
  float midpoint_host = (f_lower + (f_upper - f_lower) * 0.5f) * 0.5f;

  // Truncation: packs to `lower` — host wins against a `lower` word, loses
  // against `upper`.
  CHECK(BitsOf(MetalHostDepthReconcile(Word(lower), midpoint_host, false)) ==
        BitsOf(midpoint_host));
  CHECK(BitsOf(MetalHostDepthReconcile(Word(upper), midpoint_host, false)) ==
        BitsOf(xenos::Float20e4To32(upper) * 0.5f));

  // RNE: packs to `upper` — the winners flip.
  CHECK(BitsOf(MetalHostDepthReconcile(Word(upper), midpoint_host, true)) ==
        BitsOf(midpoint_host));
  CHECK(BitsOf(MetalHostDepthReconcile(Word(lower), midpoint_host, true)) ==
        BitsOf(xenos::Float20e4To32(lower) * 0.5f));
}

TEST_CASE("two-source selection is dest-keyed and intersection-clipped",
          "[depth_semantics][host_depth][join]") {
  int source_a = 1, source_b = 2;
  std::vector<MetalTwoSourceRecord> records = {
      {/*dest_key=*/0xAAAA, &source_a, 0, 16},
      {/*dest_key=*/0xBBBB, &source_b, 8, 24},  // other dest, same tiles
  };

  SECTION("partial overlap clips to the exact intersection") {
    auto out = MetalSelectTwoSourceIntersections(records, 0xAAAA, 10, 20);
    REQUIRE(out.size() == 1);
    CHECK(out[0].start_tiles == 10);
    CHECK(out[0].end_tiles == 16);
    CHECK(out[0].host_source == &source_a);
  }

  SECTION("two destinations sharing tiles never cross-select") {
    // The 8551 defect class: dest A's materialize over tiles B also claims
    // must select ONLY A's record and A's source.
    auto for_a = MetalSelectTwoSourceIntersections(records, 0xAAAA, 8, 16);
    REQUIRE(for_a.size() == 1);
    CHECK(for_a[0].host_source == &source_a);
    auto for_b = MetalSelectTwoSourceIntersections(records, 0xBBBB, 8, 16);
    REQUIRE(for_b.size() == 1);
    CHECK(for_b[0].host_source == &source_b);
  }

  SECTION("an unknown destination selects nothing") {
    CHECK(MetalSelectTwoSourceIntersections(records, 0xCCCC, 0, 2048).empty());
  }

  SECTION("disjoint op range selects nothing") {
    CHECK(MetalSelectTwoSourceIntersections(records, 0xAAAA, 16, 32).empty());
  }
}

TEST_CASE("overlapping same-destination selections are dropped fail-closed",
          "[depth_semantics][host_depth][join]") {
  // The executor refreshes ALL lanes before ALL reconcile draws; overlapping
  // records would let the last refresh win both draws. The planner's
  // ownership walk is disjoint per destination, so overlap is an invariant
  // violation — the helper drops every record overlapping an earlier one and
  // reports it happened.
  int source_a = 1, source_b = 2, source_c = 3;
  std::vector<MetalTwoSourceRecord> disjoint = {
      {0xAAAA, &source_a, 0, 8}, {0xAAAA, &source_b, 8, 16}};
  CHECK_FALSE(MetalDropOverlappingTwoSourceRecords(&disjoint));
  REQUIRE(disjoint.size() == 2);

  std::vector<MetalTwoSourceRecord> overlapping = {
      {0xAAAA, &source_a, 0, 12},
      {0xAAAA, &source_b, 8, 16},   // overlaps the first -> dropped
      {0xAAAA, &source_c, 16, 20},  // disjoint -> kept
  };
  CHECK(MetalDropOverlappingTwoSourceRecords(&overlapping));
  REQUIRE(overlapping.size() == 2);
  CHECK(overlapping[0].host_source == &source_a);
  CHECK(overlapping[1].host_source == &source_c);
}

TEST_CASE("stencil never participates in reconciliation",
          "[depth_semantics][host_depth][reconcile]") {
  // Same depth bits, wildly different stencil bytes: identical outcome. The
  // caller always takes stencil from the guest word.
  float host = xenos::Float20e4To32(0x123456u) * 0.5f;
  float a = MetalHostDepthReconcile(Word(0x123456u, 0x00u), host, false);
  float b = MetalHostDepthReconcile(Word(0x123456u, 0xFFu), host, false);
  CHECK(BitsOf(a) == BitsOf(b));
  CHECK(BitsOf(a) == BitsOf(host));
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
