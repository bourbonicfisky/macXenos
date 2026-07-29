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

// THE SELECTOR, grounded in the guest register contract rather than inferred.
//
// registers.h on DC_LUT_PWL_DATA: the PWL output is
// `base + (multiplier * delta) / 2^increment`, "where the increment is the
// value specified in DC_LUTA_CONTROL for the specific color channel ... the
// increment is nonzero, OTHERWISE THE 256-ENTRY TABLE SHOULD BE USED INSTEAD."
//
// So a zero increment selects the standard table and a nonzero one selects PWL.
// DC_LUTA_CONTROL defaults to 0, and register_table.inc records that a title
// which never writes it (535107D4) still expects the behavior of 0 -- so
// "unset" and "standard" are the same state, and absence is not ambiguity.
//
// Any nonzero raw value is treated as PWL. The M56 documents per-channel
// increment fields while this part appears to use a single set for all
// channels, so testing the whole dword rather than one field cannot silently
// misread an unexpected layout as standard. The census records the raw value
// alongside the mode precisely so such a layout surprise would be visible.
enum class GammaRampMode : uint32_t {
  kStandard256 = 0,
  kPWL,
};

inline const char* GammaRampModeName(GammaRampMode m) {
  return m == GammaRampMode::kStandard256 ? "standard-256" : "PWL";
}

inline GammaRampMode GammaRampModeFromLutaControl(uint32_t dc_luta_control) {
  return dc_luta_control == 0u ? GammaRampMode::kStandard256
                               : GammaRampMode::kPWL;
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

// PWL CLASSIFICATION IS DELIBERATELY ABSENT -- QUARANTINED, NOT MISSING.
//
// A prior `ClassifyGammaRampPWL` was REMOVED after the 2026-07-27 census proved
// it wrong: it reported NON-IDENTITY for five titles that had performed ZERO
// PWL writes, all sharing one byte-identical hash. That constant was the
// UNWRITTEN DEFAULT, so the classifier was manufacturing engagement out of
// state no guest had touched.
//
// It was doubly ungrounded. It assumed PWL segment bases step under the same
// 8->10 widenings as the standard table, when PWL semantics are
// `output = base + (multiplier * delta) / 2^increment` with the increment taken
// from DC_LUTA_CONTROL -- and the reference does not implement variable
// increments at all (apply_gamma_pwl.xesli hardcodes 2^3, with its own TODO).
//
// No PWL correctness claim may be made from this header. The grounded PWL
// signals are the WRITE COUNT and the selector, both of which the census
// records directly. A PWL classifier may only return once the increment
// contract is settled.

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
