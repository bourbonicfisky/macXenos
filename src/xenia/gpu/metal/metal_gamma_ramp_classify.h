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
// THE IDENTITY TEST IS GROUNDED IN THE EMULATOR'S OWN DEFAULT RAMP
// -----------------------------------------------------------------
// The 256-entry table maps an 8-bit front-buffer value to a 10-bit output
// (apply_gamma_table.xesli fetches ramp[input] and normalizes by 1023). An
// "identity" ramp is one that reproduces the input unchanged.
//
// The authoritative formula is stated by CommandProcessor::Initialize(), which
// seeds the table to "their default (LINEAR) values - taken from what games set
// when starting with the sRGB (return value 1) VdGetCurrentDisplayGamma":
//
//   LINEAR:  out = i * 0x3FF / 0xFF        (TRUNCATING integer division)
//
// That is the grounded reference: the emulator itself calls it linear and
// derives it from observed guest behavior.
//
// Two other exact 8->10 widenings are also accepted, because a guest could
// legitimately have programmed either and neither would alter the image:
//
//   REPLICATE:  out = (i << 2) | (i >> 6)     bit replication
//   SCALE:      out = (i * 1023 + 127) / 255  round-half-up rational scaling
//
// These are NOT interchangeable: LINEAR differs from each of the other two at
// 126 of 256 entries, because truncation and rounding disagree across half the
// range. An earlier revision of this header tested only REPLICATE and SCALE and
// therefore classified the emulator's own default-linear ramp as NON-IDENTITY.
// All three are now tested exactly, with no tolerance anywhere, and the census
// reports WHICH one matched so a grounded match is never conflated with an
// accepted-variant match.
//
// A ramp that is entirely zero is reported as "never programmed", but note that
// this state is UNREACHABLE at runtime: Initialize() seeds the linear ramp
// before any guest code runs, so an all-zero table only occurs in synthetic
// tests. A title that never writes the ramp presents the LINEAR default, which
// is an identity transfer -- not an absence of one.

#include <cstdint>

#include "third_party/xxhash/xxhash.h"
#include "xenia/gpu/registers.h"

namespace xe {
namespace gpu {
namespace metal {

enum class GammaRampState : uint32_t {
  kNeverProgrammed = 0,  // all entries zero -- synthetic only, see above
  kIdentity,             // exactly matches one of the three exact widenings
  kNonIdentity,          // matches none of them
};

// Which formula an identity ramp matched. kLinear is the grounded reference;
// the other two are accepted variants and are reported separately so they can
// never silently stand in for it.
enum class GammaIdentityVariant : uint32_t {
  kNone = 0,
  kLinear,
  kReplicate,
  kScale,
};

inline const char* GammaIdentityVariantName(GammaIdentityVariant v) {
  switch (v) {
    case GammaIdentityVariant::kLinear:
      return "linear-grounded";
    case GammaIdentityVariant::kReplicate:
      return "bit-replication";
    case GammaIdentityVariant::kScale:
      return "rational-scaling";
    default:
      return "none";
  }
}

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
// The grounded one: exactly CommandProcessor::Initialize()'s default ramp.
constexpr uint32_t GammaIdentityLinear(uint32_t i) {
  return (i * 0x3FFu) / 0xFFu;
}
constexpr uint32_t GammaIdentityReplicate(uint32_t i) {
  return ((i << 2) | (i >> 6)) & 0x3FF;
}
constexpr uint32_t GammaIdentityScale(uint32_t i) {
  return ((i * 1023u) + 127u) / 255u;
}

// Classifies the complete 256-entry table. `table` must be the full 256 entries
// as the base command processor tracks them.
inline GammaRampState ClassifyGammaRamp256(
    const reg::DC_LUT_30_COLOR* table,
    GammaIdentityVariant* variant_out = nullptr) {
  bool all_zero = true;
  bool matches_linear = true;
  bool matches_replicate = true;
  bool matches_scale = true;
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t r = table[i].color_10_red;
    const uint32_t g = table[i].color_10_green;
    const uint32_t b = table[i].color_10_blue;
    if (r || g || b) {
      all_zero = false;
    }
    const uint32_t lin = GammaIdentityLinear(i);
    const uint32_t rep = GammaIdentityReplicate(i);
    const uint32_t sca = GammaIdentityScale(i);
    if (r != lin || g != lin || b != lin) {
      matches_linear = false;
    }
    if (r != rep || g != rep || b != rep) {
      matches_replicate = false;
    }
    if (r != sca || g != sca || b != sca) {
      matches_scale = false;
    }
  }
  GammaIdentityVariant variant = GammaIdentityVariant::kNone;
  if (matches_linear) {
    variant = GammaIdentityVariant::kLinear;
  } else if (matches_replicate) {
    variant = GammaIdentityVariant::kReplicate;
  } else if (matches_scale) {
    variant = GammaIdentityVariant::kScale;
  }
  if (variant_out) {
    *variant_out = variant;
  }
  if (all_zero) {
    return GammaRampState::kNeverProgrammed;
  }
  return variant != GammaIdentityVariant::kNone ? GammaRampState::kIdentity
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
