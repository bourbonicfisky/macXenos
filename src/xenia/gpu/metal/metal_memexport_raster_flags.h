/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MEMEXPORT_RASTER_FLAGS_H_
#define XENIA_GPU_METAL_METAL_MEMEXPORT_RASTER_FLAGS_H_

// extend-metal-rasterizing-memexport-producers tasks 2.1-2.4: the
// XE_METAL_MEMEXPORT_RASTER gate parse and the PURE producer-routing decision
// for memexport draws that ALSO rasterize, as functions of their inputs so the
// L0 suite pins the whole decision matrix without a device or an environment
// (the exact conventions of metal_tessellation_flags.h /
// metal_depth_semantics_flags.h).
//
// Gate contract (design D4): unset, empty, and exact "0" are OFF; exact "1" is
// ON; anything else is kMalformed so the caller can warn once, and
// behaviorally means OFF (fail closed -- the OFF arm must be bit-identical to
// a build without the feature).
//
// WHY A PURE PREDICATE: the routing decision decides whether guest writes
// reach guest memory at all. Every "unrouted" answer must carry a distinct
// reason so the unconditional gap report (metal_memexport_gap_report.h) and
// the draw-skip accounting can name it, and so the class accounting flips
// EXACTLY when a class is actually routed -- never merely because a gate was
// enabled.

#include <cstdint>
#include <cstring>

namespace xe {
namespace gpu {
namespace metal {

enum class MetalMemexportRasterGate { kOff, kOn, kMalformed };

// Pure parse of the XE_METAL_MEMEXPORT_RASTER value (nullptr = unset).
inline MetalMemexportRasterGate ParseMetalMemexportRasterGate(
    const char* value) {
  if (!value || !value[0] || !std::strcmp(value, "0")) {
    return MetalMemexportRasterGate::kOff;
  }
  if (!std::strcmp(value, "1")) {
    return MetalMemexportRasterGate::kOn;
  }
  return MetalMemexportRasterGate::kMalformed;
}

inline bool MetalMemexportRasterGateEngaged(MetalMemexportRasterGate gate) {
  return gate == MetalMemexportRasterGate::kOn;
}

// What this backend does with a memexport producer that also has a bound
// render target. Every non-routed answer is a distinct reason.
enum class MetalMemexportRasterRoute : uint32_t {
  // Not this producer class at all: no render target is bound, so the
  // already-promoted VS-only/no-RT route owns the draw.
  kNotRasterizing = 0,
  // Routed: the ORIGINAL draw rasterizes AND exports (design D1).
  kRouted,
  // Gate off -- today's behavior exactly (rasterize, drop the exports,
  // reason-coded by the gap report). The exact rollback.
  kUnroutedGateOff,
  // Pixel-shader export. Permanently out of scope for this change, gate or
  // not (spec: "Pixel-stage export remains excluded and reason-coded
  // regardless of the gate").
  kUnroutedPixelStage,
  // The draw is running the gated tessellation route. Tessellated producers
  // stay fail-closed/out of scope; the tessellation route itself skips such a
  // draw before it can reach here, so this is a belt-and-braces answer.
  kUnroutedTessellated,
  // The shader writes eM# but no VALID stream constant produced a range
  // (AddMemExportRanges rejected every stream). Nothing to prepare, publish,
  // or converge -- routing it would publish an empty extent.
  kUnroutedNoRanges,
};

inline const char* MetalMemexportRasterRouteName(
    MetalMemexportRasterRoute route) {
  switch (route) {
    case MetalMemexportRasterRoute::kNotRasterizing:
      return "not-rasterizing";
    case MetalMemexportRasterRoute::kRouted:
      return "routed";
    case MetalMemexportRasterRoute::kUnroutedGateOff:
      return "unrouted-gate-off";
    case MetalMemexportRasterRoute::kUnroutedPixelStage:
      return "unrouted-pixel-stage";
    case MetalMemexportRasterRoute::kUnroutedTessellated:
      return "unrouted-tessellated";
    case MetalMemexportRasterRoute::kUnroutedNoRanges:
      return "unrouted-no-ranges";
    default:
      return "unknown";
  }
}

inline bool MetalMemexportRasterRouteRoutes(MetalMemexportRasterRoute route) {
  return route == MetalMemexportRasterRoute::kRouted;
}

// The pure decision. Ordering is deliberate: the two permanent exclusions
// (pixel stage, tessellated) and the "no ranges" answer are decided BEFORE the
// gate, so their reason codes are identical in both arms and the gate never
// launders an out-of-scope producer into a routed one.
inline MetalMemexportRasterRoute DecideMetalMemexportRasterRoute(
    bool gate_engaged, bool render_target_bound,
    bool vertex_export_ranges_present, bool pixel_stage_exports,
    bool tessellated_draw) {
  if (!render_target_bound) {
    return MetalMemexportRasterRoute::kNotRasterizing;
  }
  if (pixel_stage_exports) {
    return MetalMemexportRasterRoute::kUnroutedPixelStage;
  }
  if (!vertex_export_ranges_present) {
    return MetalMemexportRasterRoute::kUnroutedNoRanges;
  }
  if (tessellated_draw) {
    return MetalMemexportRasterRoute::kUnroutedTessellated;
  }
  if (!gate_engaged) {
    return MetalMemexportRasterRoute::kUnroutedGateOff;
  }
  return MetalMemexportRasterRoute::kRouted;
}

// The stage mask a routed producer's WRITABLE shared-memory UAV must be
// visible on (design D2). Verified against the compiled pipeline for the
// geometry-emulated case: MSC compiles the guest VS -- including its eM#
// stores -- as the OBJECT stage (metal_pipeline_cache.mm's geometry VS stage
// reuses `vertex->translated_binary()` verbatim), while the mesh stage is the
// GENERATED DXBC geometry shader, which declares no UAV at all. The mesh bit
// is therefore a harmless superset kept only because this backend binds one
// top-level argument buffer to both mesh-path stages; the fragment stage never
// writes exports (pixel-stage export is out of scope).
enum MetalMemexportRasterStage : uint32_t {
  kMetalMemexportRasterStageVertex = 1u << 0,
  kMetalMemexportRasterStageObject = 1u << 1,
  kMetalMemexportRasterStageMesh = 1u << 2,
};

constexpr uint32_t GetMetalMemexportRasterStageMask(bool geometry_emulated) {
  return geometry_emulated ? (kMetalMemexportRasterStageObject |
                              kMetalMemexportRasterStageMesh)
                           : kMetalMemexportRasterStageVertex;
}

// The stage that actually executes the guest VS's export stores. Object-only
// under geometry emulation; the mesh stage runs the generated GS.
constexpr uint32_t GetMetalMemexportRasterWritingStage(bool geometry_emulated) {
  return geometry_emulated ? kMetalMemexportRasterStageObject
                           : kMetalMemexportRasterStageVertex;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_RASTER_FLAGS_H_
