/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <array>
#include <cstdint>
#include <ios>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_binding_layout.h"
#include "xenia/gpu/xenos.h"

// Level 0 (device-free) decision coverage for the Metal texture-binding
// sign-interpretation reachability contract
// (OpenSpec align-metal-texture-binding-signedness, tasks 2.2 and 2.3).
//
// Every case calls the PRODUCTION helper
// `IsMetalTextureSignInterpretationReachable` from
// `xenia/gpu/metal/metal_binding_layout.h`. Nothing here re-implements the
// packed-byte predicates: expectations are derived from the *translator's*
// per-component sign-selection contract in
// `src/xenia/gpu/dxbc_shader_translator_fetch.cc` (the `is_all_signed` /
// `is_any_signed` reduction for `kTextureFetch`, and the whole-byte
// `signs_byte != kSigned * 0b01010101` form for `kGetTextureComputedLod`),
// which is the oracle the helper must agree with. A test that re-derived the
// helper's own bit arithmetic would pass while production drifted.

namespace xe {
namespace gpu {
namespace metal {
namespace {

using xenos::TextureSign;

constexpr uint32_t kSignByteCount = 256;
constexpr uint32_t kComponentCount = 4;

// `texture_util::SwizzleSigns` packs component i into bits (i * 2)..(i * 2 + 1)
// (texture_util.cc: `signs |= uint8_t(sign) << (i * 2)`).
constexpr TextureSign ComponentSign(uint8_t packed_signs, uint32_t component) {
  return TextureSign((packed_signs >> (component * 2)) & 0b11);
}

constexpr uint8_t PackSigns(TextureSign c0, TextureSign c1, TextureSign c2,
                            TextureSign c3) {
  return uint8_t(uint32_t(c0) | (uint32_t(c1) << 2) | (uint32_t(c2) << 4) |
                 (uint32_t(c3) << 6));
}

constexpr uint8_t kAllSignedByte =
    PackSigns(TextureSign::kSigned, TextureSign::kSigned, TextureSign::kSigned,
              TextureSign::kSigned);

// ---- The translator's reduction, expressed over an arbitrary component set --
//
// `mask` is the set of components the reduction runs over -- the translated
// shader uses `used_result_nonzero_components`; the whole-byte case is
// mask == 0b1111. Mirrors dxbc_shader_translator_fetch.cc's `OpAnd` / `OpOr`
// over the per-component `sign == kSigned` flags.

constexpr bool TranslatorAllSigned(uint8_t packed_signs, uint32_t mask) {
  for (uint32_t i = 0; i < kComponentCount; ++i) {
    if ((mask & (1u << i)) && ComponentSign(packed_signs, i) !=
                                  TextureSign::kSigned) {
      return false;
    }
  }
  return true;
}

constexpr bool TranslatorAnySigned(uint8_t packed_signs, uint32_t mask) {
  for (uint32_t i = 0; i < kComponentCount; ++i) {
    if ((mask & (1u << i)) &&
        ComponentSign(packed_signs, i) == TextureSign::kSigned) {
      return true;
    }
  }
  return false;
}

// The interpretation the translated shader can actually sample for a given
// consumed-component subset: the unsigned sample is emitted under
// `OpIf(false, is_all_signed)` and the signed sample under
// `OpIf(true, is_any_signed)`.
constexpr bool TranslatorSelectsUnsigned(uint8_t packed_signs, uint32_t mask) {
  return !TranslatorAllSigned(packed_signs, mask);
}
constexpr bool TranslatorSelectsSigned(uint8_t packed_signs, uint32_t mask) {
  return TranslatorAnySigned(packed_signs, mask);
}

// The three sign classes named by the change's contract.
enum class SignClass { kAllSigned, kAllNonSigned, kMixed };

constexpr SignClass ClassifyByte(uint8_t packed_signs) {
  const bool all_signed = TranslatorAllSigned(packed_signs, 0b1111);
  const bool any_signed = TranslatorAnySigned(packed_signs, 0b1111);
  if (all_signed) {
    return SignClass::kAllSigned;
  }
  return any_signed ? SignClass::kMixed : SignClass::kAllNonSigned;
}

// Compile-time spot pins so the shape of the helper cannot silently invert.
static_assert(IsMetalTextureSignInterpretationReachable(kAllSignedByte, true),
              "all-signed must keep the signed interpretation");
static_assert(!IsMetalTextureSignInterpretationReachable(kAllSignedByte, false),
              "all-signed must drop the unsigned interpretation");
static_assert(!IsMetalTextureSignInterpretationReachable(0x00, true),
              "all-unsigned must drop the signed interpretation");
static_assert(IsMetalTextureSignInterpretationReachable(0x00, false),
              "all-unsigned must keep the unsigned interpretation");

}  // namespace

// ---------------------------------------------------------------------------
// Task 2.2 -- exhaustive cross-product, class outcomes, and non-vacuity.
// ---------------------------------------------------------------------------

TEST_CASE(
    "Metal texture sign reachability matches the translator over all 256 "
    "packed sign bytes",
    "[l0][metal][texture-signedness]") {
  uint32_t all_signed_bytes = 0;
  uint32_t all_non_signed_bytes = 0;
  uint32_t mixed_bytes = 0;
  uint32_t decisions = 0;

  for (uint32_t byte = 0; byte < kSignByteCount; ++byte) {
    const uint8_t signs = uint8_t(byte);
    INFO("packed sign byte 0x" << std::hex << byte);

    // Both interpretations, every byte: 256 x 2 = 512 decisions, each taken
    // from the production helper.
    const bool signed_reachable =
        IsMetalTextureSignInterpretationReachable(signs, true);
    const bool unsigned_reachable =
        IsMetalTextureSignInterpretationReachable(signs, false);
    decisions += 2;

    // The translator contract, over the whole byte.
    REQUIRE(signed_reachable == TranslatorSelectsSigned(signs, 0b1111));
    REQUIRE(unsigned_reachable == TranslatorSelectsUnsigned(signs, 0b1111));

    // No sign byte rejects both interpretations.
    REQUIRE((signed_reachable || unsigned_reachable));

    switch (ClassifyByte(signs)) {
      case SignClass::kAllSigned:
        ++all_signed_bytes;
        // All components signed -> the signed interpretation only.
        REQUIRE(signed_reachable);
        REQUIRE_FALSE(unsigned_reachable);
        break;
      case SignClass::kAllNonSigned:
        ++all_non_signed_bytes;
        // Any mixture of kUnsigned / kUnsignedBiased / kGamma -> the unsigned
        // interpretation only. Bias and gamma are applied after sampling and
        // never change which interpretation is sampled.
        REQUIRE_FALSE(signed_reachable);
        REQUIRE(unsigned_reachable);
        break;
      case SignClass::kMixed:
        ++mixed_bytes;
        // Mixed -> both are sampled and selected per component.
        REQUIRE(signed_reachable);
        REQUIRE(unsigned_reachable);
        break;
    }
  }

  // Exhaustiveness.
  REQUIRE(decisions == 512);
  REQUIRE(all_signed_bytes + all_non_signed_bytes + mixed_bytes ==
          kSignByteCount);

  // Non-vacuity: every class actually occurs, with the exact populations the
  // contract implies (1 all-signed byte, 3^4 all-non-signed bytes, the rest
  // mixed).
  REQUIRE(all_signed_bytes == 1);
  REQUIRE(all_non_signed_bytes == 81);
  REQUIRE(mixed_bytes == 174);
  REQUIRE(all_signed_bytes > 0);
  REQUIRE(all_non_signed_bytes > 0);
  REQUIRE(mixed_bytes > 0);
}

TEST_CASE("Metal texture sign reachability covers the named sign classes",
          "[l0][metal][texture-signedness]") {
  struct NamedCase {
    const char* name;
    uint8_t signs;
    bool signed_reachable;
    bool unsigned_reachable;
  };

  const std::array<NamedCase, 12> kCases = {{
      {"all-signed", kAllSignedByte, true, false},
      {"all-unsigned",
       PackSigns(TextureSign::kUnsigned, TextureSign::kUnsigned,
                 TextureSign::kUnsigned, TextureSign::kUnsigned),
       false, true},
      {"all-biased",
       PackSigns(TextureSign::kUnsignedBiased, TextureSign::kUnsignedBiased,
                 TextureSign::kUnsignedBiased, TextureSign::kUnsignedBiased),
       false, true},
      {"all-gamma",
       PackSigns(TextureSign::kGamma, TextureSign::kGamma, TextureSign::kGamma,
                 TextureSign::kGamma),
       false, true},
      {"gamma+unsigned+biased mixture",
       PackSigns(TextureSign::kGamma, TextureSign::kUnsigned,
                 TextureSign::kUnsignedBiased, TextureSign::kGamma),
       false, true},
      {"biased+gamma+unsigned mixture",
       PackSigns(TextureSign::kUnsignedBiased, TextureSign::kGamma,
                 TextureSign::kUnsigned, TextureSign::kUnsignedBiased),
       false, true},
      {"signed in component 0 only",
       PackSigns(TextureSign::kSigned, TextureSign::kUnsigned,
                 TextureSign::kUnsigned, TextureSign::kUnsigned),
       true, true},
      {"signed in component 3 only",
       PackSigns(TextureSign::kUnsigned, TextureSign::kUnsigned,
                 TextureSign::kUnsigned, TextureSign::kSigned),
       true, true},
      {"signed RGB, unsigned alpha",
       PackSigns(TextureSign::kSigned, TextureSign::kSigned,
                 TextureSign::kSigned, TextureSign::kUnsigned),
       true, true},
      {"signed + gamma mixture",
       PackSigns(TextureSign::kSigned, TextureSign::kGamma,
                 TextureSign::kSigned, TextureSign::kGamma),
       true, true},
      {"signed + biased mixture",
       PackSigns(TextureSign::kSigned, TextureSign::kUnsignedBiased,
                 TextureSign::kSigned, TextureSign::kUnsignedBiased),
       true, true},
      {"signed + gamma + biased + unsigned mixture",
       PackSigns(TextureSign::kSigned, TextureSign::kGamma,
                 TextureSign::kUnsignedBiased, TextureSign::kUnsigned),
       true, true},
  }};

  for (const NamedCase& c : kCases) {
    INFO(c.name);
    REQUIRE(IsMetalTextureSignInterpretationReachable(c.signs, true) ==
            c.signed_reachable);
    REQUIRE(IsMetalTextureSignInterpretationReachable(c.signs, false) ==
            c.unsigned_reachable);
  }

  // The three classes produce three DIFFERENT (signed, unsigned) outcomes --
  // the decision is discriminating, not a constant.
  const auto outcome = [](uint8_t signs) {
    return std::array<bool, 2>{
        IsMetalTextureSignInterpretationReachable(signs, true),
        IsMetalTextureSignInterpretationReachable(signs, false)};
  };
  const auto all_signed = outcome(kAllSignedByte);
  const auto all_non_signed =
      outcome(PackSigns(TextureSign::kGamma, TextureSign::kUnsigned,
                        TextureSign::kUnsignedBiased, TextureSign::kGamma));
  const auto mixed =
      outcome(PackSigns(TextureSign::kSigned, TextureSign::kGamma,
                        TextureSign::kUnsignedBiased, TextureSign::kUnsigned));
  REQUIRE(all_signed != all_non_signed);
  REQUIRE(all_signed != mixed);
  REQUIRE(all_non_signed != mixed);
}

TEST_CASE("Metal texture sign reachability never rejects both interpretations",
          "[l0][metal][texture-signedness]") {
  uint32_t both_reachable = 0;
  uint32_t signed_only = 0;
  uint32_t unsigned_only = 0;

  for (uint32_t byte = 0; byte < kSignByteCount; ++byte) {
    const uint8_t signs = uint8_t(byte);
    INFO("packed sign byte 0x" << std::hex << byte);
    const bool s = IsMetalTextureSignInterpretationReachable(signs, true);
    const bool u = IsMetalTextureSignInterpretationReachable(signs, false);
    REQUIRE((s || u));
    if (s && u) {
      ++both_reachable;
    } else if (s) {
      ++signed_only;
    } else {
      ++unsigned_only;
    }
  }

  // Non-vacuity for this case too: all three survivorship shapes occur, so the
  // "never both false" property is not satisfied by a trivially-true helper.
  REQUIRE(signed_only == 1);
  REQUIRE(unsigned_only == 81);
  REQUIRE(both_reachable == 174);
}

// ---------------------------------------------------------------------------
// Task 2.3 -- the conservatism (subset-superset) safety basis of the gate.
// ---------------------------------------------------------------------------

TEST_CASE(
    "Metal texture sign reachability is a superset of every translator "
    "component-subset selection",
    "[l0][metal][texture-signedness]") {
  uint32_t checked_pairs = 0;
  uint32_t strictly_wider = 0;

  for (uint32_t byte = 0; byte < kSignByteCount; ++byte) {
    const uint8_t signs = uint8_t(byte);
    const bool whole_signed =
        IsMetalTextureSignInterpretationReachable(signs, true);
    const bool whole_unsigned =
        IsMetalTextureSignInterpretationReachable(signs, false);

    // Every non-empty subset of the four result components, i.e. every value
    // `used_result_nonzero_components` can take at a fetch instruction.
    for (uint32_t mask = 1; mask < 16; ++mask) {
      INFO("packed sign byte 0x" << std::hex << byte
                                 << " component mask 0b" << std::hex << mask);
      const bool subset_signed = TranslatorSelectsSigned(signs, mask);
      const bool subset_unsigned = TranslatorSelectsUnsigned(signs, mask);

      // The safety basis: anything the shader can select, the whole-byte
      // predicate keeps. (A => B, i.e. never selected-but-rejected.)
      REQUIRE((!subset_signed || whole_signed));
      REQUIRE((!subset_unsigned || whole_unsigned));

      ++checked_pairs;
      if ((whole_signed && !subset_signed) ||
          (whole_unsigned && !subset_unsigned)) {
        ++strictly_wider;
      }
    }
  }

  REQUIRE(checked_pairs == 256 * 15);

  // Non-vacuity: the superset relation is genuinely one-directional -- there
  // exist (byte, subset) pairs where the whole-byte predicate retains an
  // interpretation the subset reduction would not select. If this were 0 the
  // superset assertions above would be an equality in disguise.
  REQUIRE(strictly_wider > 0);
}

TEST_CASE(
    "Metal texture sign reachability retains the kGetTextureComputedLod "
    "whole-byte selection",
    "[l0][metal][texture-signedness]") {
  // `kGetTextureComputedLod` queries exactly ONE interpretation, chosen by
  // `OpINE(signs_byte, kSigned * 0b01010101)`: the unsigned one if the byte is
  // not all-signed, otherwise the signed one. It reads the whole byte, so it
  // has no component-subset gap. Because one fetch constant may also be used by
  // an ordinary texture fetch, the binding-level helper remains conservative:
  // for a mixed byte it may retain the signed interpretation too. The property
  // required here is that it never rejects the one computed LOD selects.
  for (uint32_t byte = 0; byte < kSignByteCount; ++byte) {
    const uint8_t signs = uint8_t(byte);
    INFO("packed sign byte 0x" << std::hex << byte);
    const bool lod_uses_unsigned = signs != kAllSignedByte;
    if (lod_uses_unsigned) {
      REQUIRE(IsMetalTextureSignInterpretationReachable(signs, false));
    } else {
      REQUIRE(IsMetalTextureSignInterpretationReachable(signs, true));
    }
    // Whichever one it queries is reachable. The unsigned predicate happens to
    // be exact for this opcode; the signed predicate is intentionally a
    // superset for mixed bytes shared with ordinary fetches.
    REQUIRE(IsMetalTextureSignInterpretationReachable(signs, false) ==
            lod_uses_unsigned);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
