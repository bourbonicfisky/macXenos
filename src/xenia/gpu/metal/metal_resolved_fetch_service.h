/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_RESOLVED_FETCH_SERVICE_H_
#define XENIA_GPU_METAL_METAL_RESOLVED_FETCH_SERVICE_H_

#include "xenia/gpu/metal/metal_exact_cover_predicate.h"
#include "xenia/gpu/metal/metal_scaled_cover_predicate.h"

namespace xe {
namespace gpu {
namespace metal {

// The ONE service classification for a texture fetch whose base address may
// be a registered resolve target (OpenSpec gpu-texture-loads slice 5R,
// design: docs/archive/evidence/metal_resolve_backed_texture_load_design.md). It composes
// the two EXISTING production predicates in GetTextureSlot's serve order —
// exact-cover first, then the bridge condition — so every consumer (the
// GetTextureSlot serve chain, the routed-load prepass, and any diagnostic)
// shares one decision and the guards are never duplicated.
enum class ResolvedFetchService {
  // The fetch base is not a registered resolve target: the ordinary
  // guest-memory path (cache/untile) applies.
  kNotResolveBacked,
  // Served GPU-side from the registered snapshot via the bridge condition
  // (depth, color-bridge A/B mode, never-RAM-backed dests, or the covered
  // R32F fast path incl. its derived 1x1 tip view). Never reads RAM.
  kBridgeSnapshot,
  // Served GPU-side from the snapshot via the exact-cover fast path
  // (ExactCoverSnapshotEligible: RAM-backed + newest writeback full-rect +
  // exact shape/pitch/endian + unsigned RGBA8). Never reads RAM.
  kExactCoverSnapshot,
  // RAM is authoritative for this fetch: today the CPU untile serves it;
  // under slice 5R it is the ONLY service class eligible for the GPU
  // load-shader route.
  kRamContent,
  // Draw resolution scaling only: a fetch that would otherwise be kRamContent,
  // served GPU-side from the snapshot because its guest RAM holds exactly the
  // snapshot's representative pixels (ScaledCoverSnapshotEligible). Never
  // reads RAM. Never selected at 1x.
  kScaledCoverSnapshot,
};

struct ResolvedFetchServiceInputs {
  bool has_resolved_target = false;
  // Built exactly the way GetTextureSlot builds it today (including the
  // stage-3 / color-bridge statics and the snapshot pixel-format checks).
  ExactCoverInputs exact = {};
  // The covered-R32F fast-path condition, computed by the caller exactly as
  // the serve chain computes it (incl. the XE_METAL_TIP_RAM_FETCH override).
  bool covered_r32f = false;
  // Scaled-cover inputs (scaled_session is false at 1x).
  ScaledCoverInputs scaled = {};
};

inline ResolvedFetchService ClassifyResolvedFetch(
    const ResolvedFetchServiceInputs& in) {
  if (!in.has_resolved_target) {
    return ResolvedFetchService::kNotResolveBacked;
  }
  // Serve order mirrors GetTextureSlot: exact-cover is evaluated first.
  if (ExactCoverSnapshotEligible(in.exact)) {
    return ResolvedFetchService::kExactCoverSnapshot;
  }
  if (in.exact.rt_is_depth || in.exact.color_bridge || !in.exact.ram_backed ||
      in.covered_r32f) {
    return ResolvedFetchService::kBridgeSnapshot;
  }
  if (ScaledCoverSnapshotEligible(in.scaled)) {
    return ResolvedFetchService::kScaledCoverSnapshot;
  }
  return ResolvedFetchService::kRamContent;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RESOLVED_FETCH_SERVICE_H_
