/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_HOST_DEPTH_RECONCILE_H_
#define XENIA_GPU_METAL_METAL_HOST_DEPTH_RECONCILE_H_

// CPU oracle for Stage B two-source depth reconciliation
// (implement-metal-depth-semantics 3.3). Mirrors the MSL
// xe_host_depth_reconcile in metal_edram_msl.h; the L1 kernel case pins the
// two against each other, and this header is the L0 truth the kernel must
// match.
//
// CONTRACT (the reviewer-specified rule): the canonical guest 20e4 word is
// the authority. The host-precision float is used IF AND ONLY IF packing it
// (under the arm's rounding mode) reproduces the guest word's depth bits —
// i.e. the guest value is provably just the rounded shadow of this exact
// host value. Any mismatch — the guest wrote the range, another RT's pack
// produced the word, the lane is stale — makes the guest word win. Validity
// is pointwise; no ownership metadata exists to go stale.
//
// FLOAT24-ONLY by construction: the host-depth predicate
// (MetalHostDepthEncodingDifferent) is true only for kD24FS8. unorm24 host
// round-trips exactly within its precision, so kD24S8 never reaches
// reconciliation. Host domain is the depth attachment's [0,1] encoding of
// the guest [0,2) 20e4 range (the canonical *0.5 mapping).
//
// Stencil is NOT reconciled: the guest word's stencil byte is always
// authoritative (the host lane stores no stencil).

#include <cstdint>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Packs a host-domain depth value to guest 20e4 bits under the arm's
// rounding mode. *2.0f is exact (power of two); saturation handled by
// Float32To20e4.
inline uint32_t MetalHostDepthPack(float host_depth,
                                   bool round_to_nearest_even) {
  return xenos::Float32To20e4(host_depth * 2.0f, round_to_nearest_even);
}

// The reconciled host-domain depth for one sample.
inline float MetalHostDepthReconcile(uint32_t guest_word, float host_depth,
                                     bool round_to_nearest_even) {
  uint32_t guest_d24 = guest_word >> 8;
  if (MetalHostDepthPack(host_depth, round_to_nearest_even) == guest_d24) {
    return host_depth;  // bit-exact host preservation, not a re-decode
  }
  return xenos::Float20e4To32(guest_d24) * 0.5f;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_HOST_DEPTH_RECONCILE_H_
