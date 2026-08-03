/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_RESIDENCY_STAGE_FLAGS_H_
#define XENIA_GPU_METAL_METAL_RESIDENCY_STAGE_FLAGS_H_

// port-xenios-tessellation task 5.3, candidate 1 (all-four-stage mask A/B):
// the XE_METAL_ALL_STAGE_RESIDENCY gate parse as a PURE function of its input,
// so the L0 suite pins the decision matrix without a device or environment.
//
// Gate contract (the XE_METAL_TESSELLATION / XE_METAL_F24_PS conventions):
// unset, empty, and exact "0" are OFF; exact "1" is ON; anything else is
// kMalformed so the caller can warn once, and behaviorally means OFF.
//
// WHAT IT DOES: widens the per-draw `useResource` stage mask from "exactly the
// stages this draw runs on" to XeniOS's unconditional all-graphics-stages
// shape (MetalAllGraphicsRenderStages()). Reference-fork comparison found
// XeniOS deliberately conservative here -- extra stage declarations cost
// nothing, while a wrongly-NARROW mask is a page fault -- whereas our per-draw
// mask is an optimization that depends on our stage model being exactly right.
//
// WHAT IT IS NOT: a fix. Widening the mask perturbs scheduling as well as
// residency, and a stochastic fault at ~4/7 occurrence can be suppressed by
// either. Per the change's pre-committed fix-validation rule, one fault
// FALSIFIES this candidate; >=5 clean matched-exposure runs make it a
// CANDIDATE only, and the mechanism -- why an unused Vertex stage would matter
// when no vertex stage runs under mesh-shader emulation -- must be established
// before anything lands permanently.

#include <cstring>

namespace xe {
namespace gpu {
namespace metal {

enum class MetalAllStageResidencyGate { kOff, kOn, kMalformed };

// Pure parse of the XE_METAL_ALL_STAGE_RESIDENCY value (nullptr = unset).
inline MetalAllStageResidencyGate ParseMetalAllStageResidencyGate(
    const char* value) {
  if (!value || !value[0] || !std::strcmp(value, "0")) {
    return MetalAllStageResidencyGate::kOff;
  }
  if (!std::strcmp(value, "1")) {
    return MetalAllStageResidencyGate::kOn;
  }
  return MetalAllStageResidencyGate::kMalformed;
}

inline bool MetalAllStageResidencyGateEngaged(MetalAllStageResidencyGate gate) {
  return gate == MetalAllStageResidencyGate::kOn;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RESIDENCY_STAGE_FLAGS_H_
