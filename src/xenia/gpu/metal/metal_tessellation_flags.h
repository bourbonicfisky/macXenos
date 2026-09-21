/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_TESSELLATION_FLAGS_H_
#define XENIA_GPU_METAL_METAL_TESSELLATION_FLAGS_H_

// port-xenios-tessellation task 4.6: the XE_METAL_TESSELLATION gate parse and
// the pure draw-route admissibility predicates, as PURE functions of their
// inputs so the L0 suite pins the decision matrix without a device or
// environment (the exact conventions of metal_depth_semantics_flags.h).
//
// Gate contract (design D6, the XE_METAL_F24_PS conventions): unset, empty,
// and exact "0" are OFF; exact "1" is ON; anything else is kMalformed so the
// caller can warn once, and behaviorally means OFF (fail closed — the OFF arm
// must be bit-identical to a build without the feature).

#include <cstring>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

enum class MetalTessellationGate { kOff, kOn, kMalformed };

// Pure parse of the XE_METAL_TESSELLATION value (nullptr = unset).
inline MetalTessellationGate ParseMetalTessellationGate(const char* value) {
  if (!value || !value[0] || !std::strcmp(value, "0")) {
    return MetalTessellationGate::kOff;
  }
  if (!std::strcmp(value, "1")) {
    return MetalTessellationGate::kOn;
  }
  return MetalTessellationGate::kMalformed;
}

inline bool MetalTessellationGateEngaged(MetalTessellationGate gate) {
  return gate == MetalTessellationGate::kOn;
}

// implement-metal-host-tessellation task 3.3: which implementation a
// tessellated draw takes once the gate admits it. XE_METAL_TESS_ROUTE unset,
// empty or exact "host" = the host tessellation route (design D1); exact
// "mesh" = the MSC object/mesh emulation route (retained ONLY for this
// change's A/B arms, deleted at promotion per design D10); anything else is
// kMalformed: warn once, behave as host.
enum class MetalTessellationRoute { kHost, kMesh, kMalformed };

inline MetalTessellationRoute ParseMetalTessellationRoute(const char* value) {
  if (!value || !value[0] || !std::strcmp(value, "host")) {
    return MetalTessellationRoute::kHost;
  }
  if (!std::strcmp(value, "mesh")) {
    return MetalTessellationRoute::kMesh;
  }
  return MetalTessellationRoute::kMalformed;
}

inline bool MetalTessellationRouteIsHost(MetalTessellationRoute route) {
  return route != MetalTessellationRoute::kMesh;
}

// Whether the tessellation route's CPU-swap arena staging (SwapIndicesInto)
// can reproduce the guest index/factor stream exactly (design D5 endian
// decision: the route CPU-swaps the stream into the upload arena and keeps
// publishing vertex_index_endian = kNone, so the prebuilt tess VS's in-shader
// XeEndianSwap32 is the identity).
//
// 16-bit streams: the shared PrimitiveProcessor already normalizes 32-bit
// swap modes on 16-bit indices down to kNone/k8in16, both of which
// SwapIndicesInto implements exactly.
// 32-bit streams (including the adaptive float factor stream): the staging
// helper's 32-bit branch implements only the full byte reversal, which is
// correct for kNone and k8in32 (the shapes seen from real guests) but NOT for
// k8in16/k16in32 on 32-bit words — those would silently corrupt factors, so
// the route fail-closed skips them instead (caught here, pinned by the L0
// endian oracle).
inline bool IsMetalTessellationIndexStreamSupported(xenos::IndexFormat format,
                                                    xenos::Endian endian) {
  if (format == xenos::IndexFormat::kInt16) {
    return true;
  }
  return endian == xenos::Endian::kNone || endian == xenos::Endian::k8in32;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TESSELLATION_FLAGS_H_
