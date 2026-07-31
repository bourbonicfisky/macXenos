/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_INTEGER_SCALE_PUBLISH_H_
#define XENIA_GPU_METAL_METAL_INTEGER_SCALE_PUBLISH_H_

// Publication of per-fetch-constant integer texture scales into the shader
// system constants
// (openspec/changes/implement-unsigned-integer-texture-scaling).
//
// This is a separate header, rather than a loop inlined in the command
// processor, because its invariants are exactly the ones a rendered frame
// cannot show you:
//
//   - the value lands at the BINDING'S OWN fetch-constant index. The engaged
//     Just Cause 2 consumers sit at tf6 and tf10 (and tf4 in one trace), never
//     at slot 0, so an off-by-one or a "first binding wins" bug would still
//     look plausible in a frame while feeding the wrong shader lane.
//   - every slot NOT bound by the draw stays zero, which is what makes the
//     shader take its uniform early-out. A stray non-zero in a gap silently
//     rescales an unrelated fetch.
//   - a stage with no integer fetches contributes nothing, so vertex and pixel
//     stages sharing the array cannot corrupt each other.
//
// The command processor and the conformance suite call the same functions here,
// so a test cannot pass against a re-implementation of the rule.

#include <cstddef>
#include <cstdint>

#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// One entry per guest texture fetch constant.
inline constexpr size_t kMetalIntegerScaleSlotCount = 32;

// Clears the whole array. Callers that memset the entire system-constant block
// already satisfy this; it exists so the invariant is stated once and can be
// asserted directly.
inline void ResetIntegerScaleBits(uint32_t* scale_bits, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    scale_bits[i] = 0;
  }
}

// Publishes one bound texture's scale at its own fetch-constant index.
//
// Returns the value written, or 0 when the binding needs no rescaling — which
// is every fetch except an integer num_format on a fixed-point format. Out-of-
// range fetch constants are ignored rather than clamped: clamping would write a
// real scale onto an unrelated slot, which is worse than dropping it.
inline uint32_t PublishIntegerScaleBits(
    uint32_t fetch_constant, const xenos::xe_gpu_texture_fetch_t& fetch,
    uint8_t swizzled_signs, uint32_t* scale_bits, size_t count) {
  if (fetch_constant >= count) {
    return 0;
  }
  uint32_t scale = texture_util::IntegerScaleBits(
      fetch.format, uint32_t(fetch.num_format), fetch.swizzle, swizzled_signs);
  scale_bits[fetch_constant] = scale;
  return scale;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_INTEGER_SCALE_PUBLISH_H_
