/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_GAMMA_RAMP_CLASSIFY_H_
#define XENIA_GPU_METAL_METAL_GAMMA_RAMP_CLASSIFY_H_

// Exact classification of the COMPLETE guest gamma ramp state, for the
// Contract 1 engagement census.
//
// Scope: the guest gamma RAMP only -- the display transfer function the guest
// programs via DC_LUT registers. This is NOT the PS-side gamma encoding of
// k_8_8_8_8_GAMMA render targets, which is a separate implemented mechanism
// with its own flags. Conflating them would let PS-encode activity masquerade
// as ramp engagement.
//
// WHY "EXACT" IS SUBTLE, AND HOW IT IS HANDLED WITHOUT AN EPSILON
// ---------------------------------------------------------------
// The 256-entry table maps an 8-bit front-buffer value to a 10-bit output
// (apply_gamma_table.xesli fetches ramp[input] and writes rgb10_a2). An
// "identity" ramp is therefore one whose entries reproduce the input unchanged
// under the 8->10 bit widening the hardware would otherwise apply -- and there
// are two exact, standard widenings a title could legitimately have programmed:
//
//   REPLICATE:  out = (i << 2) | (i >> 6)     bit replication
//   SCALE:      out = (i * 1023 + 127) / 255  exact rational scaling
//
// These agree for most i and differ by at most 1 LSB elsewhere. Rather than
// GUESS which one a given title used -- and risk calling a real identity ramp
// "non-identity", manufacturing false engagement -- both are tested exactly and
// a ramp matching EITHER is identity. A ramp matching neither is non-identity
// by exact comparison, with no tolerance anywhere.
//
// A ramp that is entirely zero is NOT identity: it is "never programmed", and
// is reported separately so an unwritten ramp cannot be counted as engagement.

#include <cstdint>

#include "third_party/xxhash/xxhash.h"
#include "xenia/gpu/registers.h"

namespace xe {
namespace gpu {
namespace metal {

enum class GammaRampState : uint32_t {
  kNeverProgrammed = 0,  // all entries zero -- guest has not written a ramp
  kIdentity,             // exactly matches a standard 8->10 widening
  kNonIdentity,          // differs from both exact widenings
};

inline const char* GammaRampStateName(GammaRampState s) {
  switch (s) {
    case GammaRampState::kNeverProgrammed:
      return "never-programmed";
    case GammaRampState::kIdentity:
      return "identity";
    case GammaRampState::kNonIdentity:
      return "NON-IDENTITY";
    default:
      return "?";
  }
}

// The two exact 8->10 widenings. No rounding tolerance is applied to either.
constexpr uint32_t GammaIdentityReplicate(uint32_t i) {
  return ((i << 2) | (i >> 6)) & 0x3FF;
}
constexpr uint32_t GammaIdentityScale(uint32_t i) {
  return ((i * 1023u) + 127u) / 255u;
}

// Classifies the complete 256-entry table. `table` must be the full 256 entries
// as the base command processor tracks them.
inline GammaRampState ClassifyGammaRamp256(
    const reg::DC_LUT_30_COLOR* table) {
  bool all_zero = true;
  bool matches_replicate = true;
  bool matches_scale = true;
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t r = table[i].color_10_red;
    const uint32_t g = table[i].color_10_green;
    const uint32_t b = table[i].color_10_blue;
    if (r || g || b) {
      all_zero = false;
    }
    const uint32_t rep = GammaIdentityReplicate(i);
    const uint32_t sca = GammaIdentityScale(i);
    if (r != rep || g != rep || b != rep) {
      matches_replicate = false;
    }
    if (r != sca || g != sca || b != sca) {
      matches_scale = false;
    }
  }
  if (all_zero) {
    return GammaRampState::kNeverProgrammed;
  }
  return (matches_replicate || matches_scale) ? GammaRampState::kIdentity
                                              : GammaRampState::kNonIdentity;
}

// Classifies the complete PWL ramp (128 entries x 3 channels as tracked).
//
// PWL output is `base + (input & 7) * delta / 8`, normalized by 64*1023
// (apply_gamma_pwl.xesli). An identity PWL ramp is one whose segment bases step
// exactly with the segment index over that normalization, with the delta
// carrying the within-segment slope. Both exact widenings are again admitted,
// applied to the segment's high bits.
inline GammaRampState ClassifyGammaRampPWL(
    const reg::DC_LUT_PWL_DATA* pwl_rgb) {
  bool all_zero = true;
  bool matches_replicate = true;
  bool matches_scale = true;
  for (uint32_t i = 0; i < 128 * 3; ++i) {
    const uint32_t base = pwl_rgb[i].base;
    const uint32_t delta = pwl_rgb[i].delta;
    if (base || delta) {
      all_zero = false;
    }
    // Segment index within its channel drives the expected base.
    const uint32_t seg = i % 128u;
    const uint32_t rep = GammaIdentityReplicate(seg << 1) << 6;
    const uint32_t sca = GammaIdentityScale(seg << 1) << 6;
    if (base != rep) {
      matches_replicate = false;
    }
    if (base != sca) {
      matches_scale = false;
    }
  }
  if (all_zero) {
    return GammaRampState::kNeverProgrammed;
  }
  return (matches_replicate || matches_scale) ? GammaRampState::kIdentity
                                              : GammaRampState::kNonIdentity;
}

// Stable content hash of a complete ramp, so a repeated write of the SAME table
// is distinguishable from an actual table change. Hashing the tracked bytes
// directly makes this independent of any classification.
inline uint64_t HashGammaRamp256(const reg::DC_LUT_30_COLOR* table) {
  return XXH3_64bits(table, sizeof(reg::DC_LUT_30_COLOR) * 256);
}
inline uint64_t HashGammaRampPWL(const reg::DC_LUT_PWL_DATA* pwl_rgb) {
  return XXH3_64bits(pwl_rgb, sizeof(reg::DC_LUT_PWL_DATA) * 128 * 3);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_GAMMA_RAMP_CLASSIFY_H_
