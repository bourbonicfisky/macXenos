/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// extend-metal-rasterizing-memexport-producers task 3.1 (Level 0 rung): the
// PURE gate parse, the producer-routing decision matrix, and the D2 stage
// masks. Device-free, environment-free -- these are the predicates the
// command processor consults before it binds a writable shared-memory UAV to a
// rasterizing draw, so a regression here is a regression in whether guest
// writes reach guest memory at all.

#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memexport_raster_flags.h"

namespace xe {
namespace gpu {
namespace test {

using metal::DecideMetalMemexportRasterRoute;
using metal::GetMetalMemexportRasterStageMask;
using metal::GetMetalMemexportRasterWritingStage;
using metal::kMetalMemexportRasterStageMesh;
using metal::kMetalMemexportRasterStageObject;
using metal::kMetalMemexportRasterStageVertex;
using metal::MetalMemexportRasterGate;
using metal::MetalMemexportRasterGateEngaged;
using metal::MetalMemexportRasterRoute;
using metal::MetalMemexportRasterRouteRoutes;
using metal::ParseMetalMemexportRasterGate;

TEST_CASE("XE_METAL_MEMEXPORT_RASTER parses with the house conventions",
          "[memexport][raster][gate]") {
  // Unset / empty / exact "0" are OFF (the exact-rollback arm).
  CHECK(ParseMetalMemexportRasterGate(nullptr) ==
        MetalMemexportRasterGate::kOff);
  CHECK(ParseMetalMemexportRasterGate("") == MetalMemexportRasterGate::kOff);
  CHECK(ParseMetalMemexportRasterGate("0") == MetalMemexportRasterGate::kOff);
  // Exact "1" is ON.
  CHECK(ParseMetalMemexportRasterGate("1") == MetalMemexportRasterGate::kOn);
  // Everything else is malformed -> the caller warns once, behavior is OFF.
  for (const char* value : {"2", "01", "1 ", " 1", "on", "true", "yes", "-1",
                            "0x1", "1\n"}) {
    INFO(value);
    CHECK(ParseMetalMemexportRasterGate(value) ==
          MetalMemexportRasterGate::kMalformed);
    CHECK_FALSE(
        MetalMemexportRasterGateEngaged(ParseMetalMemexportRasterGate(value)));
  }
  CHECK(MetalMemexportRasterGateEngaged(MetalMemexportRasterGate::kOn));
  CHECK_FALSE(MetalMemexportRasterGateEngaged(MetalMemexportRasterGate::kOff));
}

TEST_CASE("rasterizing memexport routing decides on inputs, not on the gate "
          "alone",
          "[memexport][raster][route]") {
  // No render target bound: not this class at all -- the already-promoted
  // VS-only/no-RT route owns the draw, gate or not.
  for (bool gate : {false, true}) {
    CHECK(DecideMetalMemexportRasterRoute(gate, /*render_target_bound=*/false,
                                          /*ranges=*/true, /*ps=*/false,
                                          /*tess=*/false) ==
          MetalMemexportRasterRoute::kNotRasterizing);
  }
  // Pixel-stage export is excluded REGARDLESS of the gate (spec requirement),
  // and its reason is identical in both arms.
  for (bool gate : {false, true}) {
    CHECK(DecideMetalMemexportRasterRoute(gate, true, /*ranges=*/true,
                                          /*ps=*/true, /*tess=*/false) ==
          MetalMemexportRasterRoute::kUnroutedPixelStage);
    // Even with no vertex ranges at all, the pixel-stage reason wins: it is
    // the more specific finding.
    CHECK(DecideMetalMemexportRasterRoute(gate, true, /*ranges=*/false,
                                          /*ps=*/true, /*tess=*/false) ==
          MetalMemexportRasterRoute::kUnroutedPixelStage);
  }
  // A shader that writes eM# but produced no VALID stream range has nothing to
  // prepare/publish/converge.
  for (bool gate : {false, true}) {
    CHECK(DecideMetalMemexportRasterRoute(gate, true, /*ranges=*/false,
                                          /*ps=*/false, /*tess=*/false) ==
          MetalMemexportRasterRoute::kUnroutedNoRanges);
  }
  // Tessellated producers stay fail-closed / out of scope in both arms.
  for (bool gate : {false, true}) {
    CHECK(DecideMetalMemexportRasterRoute(gate, true, /*ranges=*/true,
                                          /*ps=*/false, /*tess=*/true) ==
          MetalMemexportRasterRoute::kUnroutedTessellated);
  }
  // The one cell the gate owns.
  CHECK(DecideMetalMemexportRasterRoute(/*gate=*/false, true, true, false,
                                        false) ==
        MetalMemexportRasterRoute::kUnroutedGateOff);
  CHECK(DecideMetalMemexportRasterRoute(/*gate=*/true, true, true, false,
                                        false) ==
        MetalMemexportRasterRoute::kRouted);
  // Exactly one answer routes.
  CHECK(MetalMemexportRasterRouteRoutes(MetalMemexportRasterRoute::kRouted));
  for (MetalMemexportRasterRoute route :
       {MetalMemexportRasterRoute::kNotRasterizing,
        MetalMemexportRasterRoute::kUnroutedGateOff,
        MetalMemexportRasterRoute::kUnroutedPixelStage,
        MetalMemexportRasterRoute::kUnroutedTessellated,
        MetalMemexportRasterRoute::kUnroutedNoRanges}) {
    CHECK_FALSE(MetalMemexportRasterRouteRoutes(route));
  }
}

TEST_CASE("pixel-stage memexport routes under its own gate",
          "[memexport][raster][route][pixel]") {
  using metal::DecideMetalMemexportRoute;
  // No render target: never this class, whatever the gates.
  for (bool vg : {false, true}) {
    for (bool pg : {false, true}) {
      CHECK(DecideMetalMemexportRoute(vg, pg, /*rt=*/false, true, true, true,
                                      false) ==
            MetalMemexportRasterRoute::kNotRasterizing);
    }
  }
  // Pixel gate off: the old exclusion, identical whatever the vertex gate.
  for (bool vg : {false, true}) {
    CHECK(DecideMetalMemexportRoute(vg, /*pixel_gate=*/false, true,
                                    /*vs_ranges=*/false, /*ps=*/true,
                                    /*ps_ranges=*/true, false) ==
          MetalMemexportRasterRoute::kUnroutedPixelStage);
  }
  // Pixel gate on, PS-only exporter with ranges: routed without the vertex
  // raster gate.
  CHECK(DecideMetalMemexportRoute(/*vertex_gate=*/false, true, true, false,
                                  true, true, false) ==
        MetalMemexportRasterRoute::kRouted);
  // ...but not without a valid range.
  CHECK(DecideMetalMemexportRoute(false, true, true, false, true, false,
                                  false) ==
        MetalMemexportRasterRoute::kUnroutedNoRanges);
  // ...and never tessellated.
  CHECK(DecideMetalMemexportRoute(true, true, true, true, true, true, true) ==
        MetalMemexportRasterRoute::kUnroutedTessellated);
  // Both stages export: binding the UAV routes both, so the vertex gate is
  // required too.
  CHECK(DecideMetalMemexportRoute(/*vertex_gate=*/false, true, true,
                                  /*vs_ranges=*/true, true, true, false) ==
        MetalMemexportRasterRoute::kUnroutedGateOff);
  CHECK(DecideMetalMemexportRoute(true, true, true, true, true, true, false) ==
        MetalMemexportRasterRoute::kRouted);
  // With the pixel gate off, the seven-input decision is exactly the original
  // five-input one over every input.
  for (bool vg : {false, true}) {
    for (bool rt : {false, true}) {
      for (bool vr : {false, true}) {
        for (bool ps : {false, true}) {
          for (bool pr : {false, true}) {
            for (bool tess : {false, true}) {
              CHECK(DecideMetalMemexportRoute(vg, false, rt, vr, ps, pr,
                                              tess) ==
                    DecideMetalMemexportRasterRoute(vg, rt, vr, ps, tess));
            }
          }
        }
      }
    }
  }
}

TEST_CASE("gate-off routing is bit-identical across every other input",
          "[memexport][raster][route]") {
  // The OFF arm must never route, whatever else is true -- the property that
  // makes "gate off" an exact rollback.
  for (bool rt : {false, true}) {
    for (bool ranges : {false, true}) {
      for (bool ps : {false, true}) {
        for (bool tess : {false, true}) {
          CHECK_FALSE(MetalMemexportRasterRouteRoutes(
              DecideMetalMemexportRasterRoute(/*gate=*/false, rt, ranges, ps,
                                              tess)));
        }
      }
    }
  }
}

TEST_CASE("D2 stage masks: object-only writing stage under geometry emulation",
          "[memexport][raster][stages]") {
  // Plain rasterizing draw: the guest VS -- including its eM# stores -- is the
  // VERTEX stage.
  CHECK(GetMetalMemexportRasterStageMask(/*geometry_emulated=*/false) ==
        kMetalMemexportRasterStageVertex);
  CHECK(GetMetalMemexportRasterWritingStage(/*geometry_emulated=*/false) ==
        kMetalMemexportRasterStageVertex);
  // Geometry-emulated draw: the argument buffer is bound to BOTH mesh-path
  // stages, so the visibility mask spans object|mesh -- but only the OBJECT
  // stage runs the guest VS (the mesh stage is the generated GS, which
  // declares no UAV). Verified against the compiled pipeline: the geometry
  // object stage reuses the guest VS's DXBC verbatim
  // (metal_pipeline_cache.mm), and the generated GS source
  // (dxbc_geometry_shader.cc) contains no UAV declaration at all.
  CHECK(GetMetalMemexportRasterStageMask(/*geometry_emulated=*/true) ==
        (kMetalMemexportRasterStageObject | kMetalMemexportRasterStageMesh));
  CHECK(GetMetalMemexportRasterWritingStage(/*geometry_emulated=*/true) ==
        kMetalMemexportRasterStageObject);
  // The writing stage is always a subset of the visible mask -- a binding that
  // did not cover the writing stage would drop the exports silently.
  for (bool geometry : {false, true}) {
    CHECK((GetMetalMemexportRasterWritingStage(geometry) &
           GetMetalMemexportRasterStageMask(geometry)) ==
          GetMetalMemexportRasterWritingStage(geometry));
  }
}

TEST_CASE("tessellation factor-stream consumer extent boundary matrix",
          "[memexport][raster][consumer-extent]") {
  using metal::ComputeMetalIndexStreamExtent;
  using metal::MetalTessFactorStreamUnderCovered;
  constexpr uint32_t kBase = 0x1AFF8000u;

  // THE case this consumer range exists for: the tessellation route reads
  // host_draw_vertex_count elements, which EXCEEDS the guest draw's
  // index_count, so the generic guest-index interval under-covers the read and
  // adding the host extent is load-bearing rather than redundant.
  {
    auto guest = ComputeMetalIndexStreamExtent(kBase, 256, /*index_32bit=*/true);
    auto host = ComputeMetalIndexStreamExtent(kBase, 1024, true);
    REQUIRE(guest.valid);
    REQUIRE(host.valid);
    CHECK(guest.size == 1024);
    CHECK(host.size == 4096);
    CHECK(host.size > guest.size);
    CHECK(MetalTessFactorStreamUnderCovered(kBase, 256, 1024, true));
    // The measured VP shape: the producer exports 4096 bytes at 1AFF8000, and
    // the consumer's HOST extent must span all of them.
    CHECK(host.base == kBase);
    CHECK(uint64_t(host.base) + host.size == uint64_t(kBase) + 4096);
  }
  // host == guest: redundant but harmless (the range set normalizes).
  CHECK_FALSE(MetalTessFactorStreamUnderCovered(kBase, 1024, 1024, true));
  // host < guest: the guest interval already covers the read.
  CHECK_FALSE(MetalTessFactorStreamUnderCovered(kBase, 1024, 256, true));
  // Zero/absent streams are never dependencies and never enter a request.
  CHECK_FALSE(ComputeMetalIndexStreamExtent(kBase, 0, true).valid);
  CHECK_FALSE(ComputeMetalIndexStreamExtent(0, 1024, true).valid);
  CHECK_FALSE(MetalTessFactorStreamUnderCovered(kBase, 256, 0, true));
  CHECK_FALSE(MetalTessFactorStreamUnderCovered(0, 256, 1024, true));
  // A zero GUEST count with a real host count still under-covers: the guest
  // interval does not exist at all, so nothing else would add the range.
  CHECK(MetalTessFactorStreamUnderCovered(kBase, 0, 1024, true));
  // 16-bit streams halve the extent.
  {
    auto extent = ComputeMetalIndexStreamExtent(kBase, 1024,
                                                /*index_32bit=*/false);
    REQUIRE(extent.valid);
    CHECK(extent.size == 2048);
  }
  // End-of-address-space SATURATION, never wraparound: a wrapped interval
  // would become disjoint from the producer it depends on and silently drop
  // the dependency -- the precise failure this task exists to prevent.
  {
    auto extent = ComputeMetalIndexStreamExtent(0xFFFFFF00u, 1024, true);
    REQUIRE(extent.valid);
    CHECK(uint64_t(extent.base) + extent.size == 0xFFFFFFFFull);
    CHECK(extent.size == 0xFFu);
  }
  // Saturating to exactly the base leaves nothing addressable: not a range.
  CHECK_FALSE(ComputeMetalIndexStreamExtent(0xFFFFFFFFu, 1024, true).valid);
  // A count large enough to overflow 32-bit multiplication must saturate
  // rather than alias down to a small extent.
  {
    auto extent = ComputeMetalIndexStreamExtent(kBase, 0x40000000u, true);
    REQUIRE(extent.valid);
    CHECK(uint64_t(extent.base) + extent.size == 0xFFFFFFFFull);
  }
}

TEST_CASE("staleness backstop: a pending producer over the staged stream is "
          "always caught",
          "[memexport][raster][backstop]") {
  using metal::ComputeMetalIndexStreamExtent;
  using metal::MetalGuestRangesOverlap;
  using metal::MetalStreamReadWouldBeStale;
  constexpr uint32_t kBase = 0x1AFF8000u;
  auto extent = ComputeMetalIndexStreamExtent(kBase, 1024, true);  // +4096
  REQUIRE(extent.valid);
  REQUIRE(extent.size == 4096);

  // The measured VP shape: producer and consumer name the SAME range.
  CHECK(MetalStreamReadWouldBeStale(extent, kBase, 4096));
  // Partial overlaps in either direction are still stale reads -- a producer
  // owing even one byte of the staged stream must stop the read.
  CHECK(MetalStreamReadWouldBeStale(extent, kBase + 4092, 8));
  CHECK(MetalStreamReadWouldBeStale(extent, kBase - 4, 8));
  CHECK(MetalStreamReadWouldBeStale(extent, kBase + 2048, 16));
  // A producer strictly containing the stream, and one strictly inside it.
  CHECK(MetalStreamReadWouldBeStale(extent, kBase - 0x1000, 0x4000));
  CHECK(MetalStreamReadWouldBeStale(extent, kBase + 16, 16));
  // ADJACENCY IS NOT OVERLAP -- the backstop must not skip draws whose stream
  // merely abuts an unrelated export, or every neighbouring allocation would
  // become a spurious fail-closed skip.
  CHECK_FALSE(MetalStreamReadWouldBeStale(extent, kBase + 4096, 4096));
  CHECK_FALSE(MetalStreamReadWouldBeStale(extent, kBase - 4096, 4096));
  // Empty producers and an absent stream are never stale.
  CHECK_FALSE(MetalStreamReadWouldBeStale(extent, kBase, 0));
  CHECK_FALSE(MetalStreamReadWouldBeStale(
      ComputeMetalIndexStreamExtent(kBase, 0, true), kBase, 4096));
  // Overlap arithmetic must not wrap at the top of the address space.
  CHECK(MetalGuestRangesOverlap(0xFFFFF000u, 0x1000u, 0xFFFFF800u, 0x800u));
  CHECK_FALSE(MetalGuestRangesOverlap(0xFFFFF000u, 0x800u, 0xFFFFF800u, 0x800u));
  CHECK_FALSE(MetalGuestRangesOverlap(0u, 0x1000u, 0xFFFFF000u, 0x1000u));
}

TEST_CASE("consumer interval is added only when not already contained",
          "[memexport][raster][consumer-extent]") {
  using metal::ComputeMetalIndexStreamExtent;
  using metal::MetalExtentContains;
  constexpr uint32_t kBase = 0x1AFF8000u;

  // TODAY'S VP SHAPE: index_count == host_draw_vertex_count, so the two
  // extents are byte-identical, the guest-index interval already contains the
  // factor stream, and the addition is skipped -- the request set stays free
  // of redundant intervals. Structurally required, corpus-unengaged.
  auto guest_same = ComputeMetalIndexStreamExtent(kBase, 1024, true);
  auto host_same = ComputeMetalIndexStreamExtent(kBase, 1024, true);
  CHECK(MetalExtentContains(guest_same, host_same));

  // THE UNOBSERVED EXPANSION CASE: the backend reads MORE than the guest index
  // count, containment fails, and the interval must be added or the read is
  // uncovered (and the backstop would fail the draw closed).
  auto guest_small = ComputeMetalIndexStreamExtent(kBase, 256, true);
  auto host_big = ComputeMetalIndexStreamExtent(kBase, 1024, true);
  CHECK_FALSE(MetalExtentContains(guest_small, host_big));
  // Containment the other way (the backend reads less) is covered.
  CHECK(MetalExtentContains(host_big, guest_small));
  // No guest interval at all (auto-indexed draw): contains nothing, so the
  // addition always happens.
  CHECK_FALSE(MetalExtentContains(metal::MetalGuestStreamExtent{}, host_big));
  CHECK_FALSE(MetalExtentContains(host_big, metal::MetalGuestStreamExtent{}));
  // A smaller stream at a DIFFERENT base is not contained -- containment is
  // about bytes, not sizes.
  auto elsewhere = ComputeMetalIndexStreamExtent(kBase + 0x8000, 4, true);
  CHECK_FALSE(MetalExtentContains(host_big, elsewhere));
  // Partial overlap is not containment.
  auto straddling = ComputeMetalIndexStreamExtent(kBase + 4088, 8, true);
  CHECK_FALSE(MetalExtentContains(host_big, straddling));
}

TEST_CASE("join hash is mutation-pinned: one flipped bit must be caught",
          "[memexport][raster][join]") {
  using metal::MetalMemexportJoinHash;
  // The measured VP factor-stream size. A join hash that missed a single-byte
  // corruption would make the producer==consumer equality argument worthless,
  // so pin exactly that.
  constexpr uint32_t kSize = 4096;
  std::vector<uint8_t> bytes(kSize);
  for (uint32_t i = 0; i < kSize; ++i) {
    bytes[i] = uint8_t(i * 31u + 7u);
  }
  const uint64_t baseline = MetalMemexportJoinHash(bytes.data(), kSize);
  // Identical input reproduces the hash -- the join's other half.
  CHECK(MetalMemexportJoinHash(bytes.data(), kSize) == baseline);
  // Flip ONE BIT at the head, middle and tail. The tail matters most: a hash
  // that folded only a prefix would still pass a head-only test.
  for (uint32_t index : {0u, 1u, kSize / 2u, kSize - 2u, kSize - 1u}) {
    std::vector<uint8_t> mutated = bytes;
    mutated[index] ^= 0x01u;
    INFO("flipped bit in byte " << index);
    CHECK(MetalMemexportJoinHash(mutated.data(), kSize) != baseline);
  }
  // Truncation is a real failure mode (a short read) and must also be caught.
  CHECK(MetalMemexportJoinHash(bytes.data(), kSize - 1) != baseline);
  CHECK(MetalMemexportJoinHash(bytes.data(), 0) != baseline);
}

TEST_CASE("backstop fault injection parses conservatively",
          "[memexport][raster][backstop]") {
  using metal::MetalMemexportRasterFault;
  using metal::ParseMetalMemexportRasterFault;
  // Default and every unrecognized value mean NO fault: a typo must never
  // silently disable a safety mechanism.
  for (const char* value :
       {(const char*)nullptr, "", "0", "1", "yes", "no-tess-ranges", "TESS"}) {
    INFO(std::string(value ? value : "(null)"));
    CHECK(ParseMetalMemexportRasterFault(value) ==
          MetalMemexportRasterFault::kNone);
  }
  CHECK(ParseMetalMemexportRasterFault("no-tess-range") ==
        MetalMemexportRasterFault::kNoTessRange);
  CHECK(ParseMetalMemexportRasterFault("undersize-tess-range") ==
        MetalMemexportRasterFault::kUndersizeTessRange);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
