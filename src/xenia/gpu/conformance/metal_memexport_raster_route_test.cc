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

}  // namespace test
}  // namespace gpu
}  // namespace xe
