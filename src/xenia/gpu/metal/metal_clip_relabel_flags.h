/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_CLIP_RELABEL_FLAGS_H_
#define XENIA_GPU_METAL_METAL_CLIP_RELABEL_FLAGS_H_

// fix-metal-geometry-clip-distance-bridge: the XE_METAL_CLIP_RELABEL gate.
//
// Candidate 1 (intermediate semantic legalization) relabels the clip-distance
// semantic on the OBJECT STAGE'S OUTPUT so it crosses the object->mesh hop as a
// private user parameter, which is what makes clip-1/clip-2 geometry pipelines
// materialize at all. It stays DEFAULT-OFF until the whole 4.x behavioral
// matrix passes: materialization is availability, not correctness, and a
// pipeline that creates and then clips WRONGLY is worse than one that fails
// loudly, because today's failure is at least visible in the skip accounting as
// a dropped-draw class.
//
// Parse follows the house convention (XE_METAL_TESSELLATION / XE_METAL_F24_PS):
// unset, empty and exact "0" are OFF; exact "1" is ON; anything else is
// kMalformed, which the caller warns about once and which behaves as OFF.
//
// NOTE, deliberate behaviour change: the original experiment sites tested
// `std::getenv(...) != nullptr`, so ANY value — including "0" — enabled the
// relabel. Strict parse means "0" now disables it, which is what every other
// gate in this backend does and what a reader would assume.
//
// Read PER CALL rather than cached in a function-local static: the gate is only
// consulted at geometry-pipeline creation (memoized per stage key), so the cost
// is irrelevant, and a cached static cannot be toggled between arms inside one
// test process — which is exactly what the 4.1 regression checkpoint needs to
// assert both directions.

#include <cstring>

namespace xe {
namespace gpu {
namespace metal {

enum class MetalClipRelabelGate { kOff, kOn, kMalformed };

inline MetalClipRelabelGate ParseMetalClipRelabelGate(const char* value) {
  if (!value || !value[0] || !std::strcmp(value, "0")) {
    return MetalClipRelabelGate::kOff;
  }
  if (!std::strcmp(value, "1")) {
    return MetalClipRelabelGate::kOn;
  }
  return MetalClipRelabelGate::kMalformed;
}

inline bool MetalClipRelabelGateEngaged(MetalClipRelabelGate gate) {
  return gate == MetalClipRelabelGate::kOn;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_CLIP_RELABEL_FLAGS_H_
