/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 conformance for unsigned fixed-point integer texture scaling
// (openspec/changes/implement-unsigned-integer-texture-scaling).
//
// Two contracts are checked here, both pure and device-free:
//   1. the format table's fixed-point classification and per-component widths;
//   2. texture_util::IntegerScaleBits packing (Canary's GetIntegerScaleBits
//      layout since fetch ABI 8: 6-bit fields with width and TextureSign, bit 24
//      normalized, bit 26 point sampled), including the ORDER in which output
//      components map back to source component widths.
//
// The engaged title class (JC2's k_8_8_8_8) has four equal component widths, so
// a swizzle/channel-order error in the derivation would be completely invisible
// there. Every ordering case below therefore uses an ASYMMETRIC format, where
// each component width is distinct and a mis-ordered result cannot coincide
// with the correct one.

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/texture_info.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

using xenos::TextureFormat;
using xenos::TextureSign;

namespace {

constexpr uint32_t kFractional = 0;
constexpr uint32_t kInteger = 1;

// Identity swizzle: output component i reads source component i.
constexpr uint32_t kSwizzleIdentity =
    (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);

uint32_t MakeSwizzle(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  return (x << 0) | (y << 3) | (z << 6) | (w << 9);
}

uint8_t MakeSigns(TextureSign a, TextureSign b, TextureSign c, TextureSign d) {
  return uint8_t((uint32_t(a) << 0) | (uint32_t(b) << 2) |
                 (uint32_t(c) << 4) | (uint32_t(d) << 6));
}

uint8_t UniformSigns(TextureSign sign) {
  return MakeSigns(sign, sign, sign, sign);
}

// A fetch constant with the given format, num_format and swizzle. Filters are
// linear unless point is requested, because zero-initialized filters are point
// and would set bit 26 and the narrow point rule in every case.
xenos::xe_gpu_texture_fetch_t MakeFetch(TextureFormat format,
                                        uint32_t num_format, uint32_t swizzle,
                                        bool point = false) {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.format = format;
  fetch.num_format = num_format;
  fetch.swizzle = swizzle;
  xenos::TextureFilter filter =
      point ? xenos::TextureFilter::kPoint : xenos::TextureFilter::kLinear;
  fetch.mag_filter = filter;
  fetch.min_filter = filter;
  fetch.mip_filter = filter;
  fetch.aniso_filter = xenos::AnisoFilter::kDisabled;
  return fetch;
}

uint32_t FieldOf(uint32_t scale_bits, uint32_t component) {
  return (scale_bits >>
          (component * texture_util::kIntegerScaleBitsPerComponent)) &
         ((1u << texture_util::kIntegerScaleBitsPerComponent) - 1);
}

TextureSign SignOf(uint32_t scale_bits, uint32_t component) {
  return TextureSign((FieldOf(scale_bits, component) >>
                      texture_util::kIntegerScaleSignShift) &
                     0b11);
}

// The integer num_format multiplier the shader forms from one component's
// field: 2^w - 1 for unsigned, 2^(w - 1) - 1 for signed and unsigned-biased.
uint32_t FieldToMultiplier(uint32_t scale_bits, uint32_t component) {
  uint32_t field = FieldOf(scale_bits, component);
  uint32_t width = (field & texture_util::kIntegerScaleWidthMask) + 1;
  TextureSign sign = SignOf(scale_bits, component);
  bool halved =
      sign == TextureSign::kSigned || sign == TextureSign::kUnsignedBiased;
  return (1u << (width - (halved ? 1 : 0))) - 1;
}

}  // namespace

TEST_CASE("Format table classifies fixed-point formats",
          "[integer_scale][format_oracle]") {
  SECTION("k_8_8_8_8 is fixed with 8-bit components") {
    const FormatInfo& info = *FormatInfo::Get(TextureFormat::k_8_8_8_8);
    REQUIRE(info.fixed);
    CHECK(info.component_bits[0] == 8);
    CHECK(info.component_bits[1] == 8);
    CHECK(info.component_bits[2] == 8);
    CHECK(info.component_bits[3] == 8);
  }

  SECTION("asymmetric fixed formats carry distinct per-component widths") {
    const FormatInfo& rgba1010102 =
        *FormatInfo::Get(TextureFormat::k_2_10_10_10);
    REQUIRE(rgba1010102.fixed);
    CHECK(rgba1010102.component_bits[0] == 10);
    CHECK(rgba1010102.component_bits[1] == 10);
    CHECK(rgba1010102.component_bits[2] == 10);
    CHECK(rgba1010102.component_bits[3] == 2);

    const FormatInfo& rgb565 = *FormatInfo::Get(TextureFormat::k_5_6_5);
    REQUIRE(rgb565.fixed);
    CHECK(rgb565.component_bits[0] == 5);
    CHECK(rgb565.component_bits[1] == 6);
    CHECK(rgb565.component_bits[2] == 5);
  }

  SECTION("the num_format=1 formats seen in the corpus that must NOT scale") {
    // Measured in docs/dev/metal_texture_integer_scale_pregate.md: these four all
    // appear with an integer num_format, and all must classify as not fixed so
    // they derive a zero scale.
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_24_8_FLOAT)->fixed);
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_32_FLOAT)->fixed);
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_16_16_16_16_FLOAT)->fixed);
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_16_16_16_16_EXPAND)->fixed);
  }

  SECTION("compressed formats are not fixed") {
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_DXT1)->fixed);
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_DXT2_3)->fixed);
    CHECK_FALSE(FormatInfo::Get(TextureFormat::k_DXT4_5)->fixed);
  }

  SECTION("a not-fixed format's component widths are never meaningful") {
    for (uint32_t f = 0; f < 64; ++f) {
      const FormatInfo& info = *FormatInfo::Get(f);
      if (info.fixed) {
        continue;
      }
      for (uint32_t i = 0; i < 4; ++i) {
        INFO("format index " << f << " component " << i);
        CHECK(info.component_bits[i] == 0);
      }
    }
  }
}

TEST_CASE("IntegerScaleBits returns zero when there is nothing to convert",
          "[integer_scale]") {
  SECTION("filtered fetch of a non-fixed format") {
    for (TextureFormat format :
         {TextureFormat::k_24_8_FLOAT, TextureFormat::k_32_FLOAT,
          TextureFormat::k_16_16_16_16_FLOAT,
          TextureFormat::k_16_16_16_16_EXPAND, TextureFormat::k_DXT1}) {
      INFO("format " << uint32_t(format));
      CHECK(texture_util::IntegerScaleBits(
                MakeFetch(format, kInteger, kSwizzleIdentity),
                UniformSigns(TextureSign::kUnsigned)) == 0);
      CHECK(texture_util::IntegerScaleBits(
                MakeFetch(format, kFractional, kSwizzleIdentity),
                UniformSigns(TextureSign::kUnsigned)) == 0);
    }
  }

  SECTION("integer fetch whose lanes are all constant 0/1") {
    uint32_t all_constant =
        MakeSwizzle(xenos::XE_GPU_TEXTURE_SWIZZLE_0,
                    xenos::XE_GPU_TEXTURE_SWIZZLE_1,
                    xenos::XE_GPU_TEXTURE_SWIZZLE_0,
                    xenos::XE_GPU_TEXTURE_SWIZZLE_1);
    CHECK(texture_util::IntegerScaleBits(
              MakeFetch(TextureFormat::k_8_8_8_8, kInteger, all_constant),
              UniformSigns(TextureSign::kUnsigned)) == 0);
  }

  SECTION("integer fetch of all-gamma components") {
    CHECK(texture_util::IntegerScaleBits(
              MakeFetch(TextureFormat::k_8_8_8_8, kInteger, kSwizzleIdentity),
              UniformSigns(TextureSign::kGamma)) == 0);
  }
}

TEST_CASE("IntegerScaleBits flags normalized and point-sampled fetches",
          "[integer_scale]") {
  SECTION("a filtered normalized unsigned fetch only requests Q16 rounding") {
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kFractional, kSwizzleIdentity),
        UniformSigns(TextureSign::kUnsigned));
    CHECK(bits == texture_util::kIntegerScaleNormalizedBit);
  }

  SECTION("point sampling sets bit 26 even for a non-fixed format") {
    CHECK(texture_util::IntegerScaleBits(
              MakeFetch(TextureFormat::k_32_FLOAT, kFractional,
                        kSwizzleIdentity, true),
              UniformSigns(TextureSign::kUnsigned)) ==
          texture_util::kIntegerScalePointSampledBit);
  }

  SECTION("a linear mip filter is not point sampled") {
    xenos::xe_gpu_texture_fetch_t fetch = MakeFetch(
        TextureFormat::k_8_8_8_8, kFractional, kSwizzleIdentity, true);
    fetch.mip_filter = xenos::TextureFilter::kLinear;
    CHECK_FALSE(texture_util::IntegerScaleBits(
                    fetch, UniformSigns(TextureSign::kUnsigned)) &
                texture_util::kIntegerScalePointSampledBit);
  }

  SECTION("point sampled 4 to 7 bit unsigned components carry their width") {
    // k_4_4_4_4 is JC2's engaged point-snap class (format 15).
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_4_4_4_4, kFractional, kSwizzleIdentity,
                  true),
        UniformSigns(TextureSign::kUnsigned));
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(FieldOf(bits, i) == 3);
    }
  }

  SECTION("point sampled 8 bit unsigned components carry no width") {
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kFractional, kSwizzleIdentity,
                  true),
        UniformSigns(TextureSign::kUnsigned));
    CHECK(bits == (texture_util::kIntegerScaleNormalizedBit |
                   texture_util::kIntegerScalePointSampledBit));
  }
}

TEST_CASE("IntegerScaleBits packs the engaged JC2 class", "[integer_scale]") {
  // k_8_8_8_8, integer num_format, all-unsigned signs: the one integer class
  // the 49-trace census found engaged. Every component must restore x255.
  uint32_t bits = texture_util::IntegerScaleBits(
      MakeFetch(TextureFormat::k_8_8_8_8, kInteger, kSwizzleIdentity),
      UniformSigns(TextureSign::kUnsigned));
  REQUIRE(bits != 0);
  CHECK_FALSE(bits & texture_util::kIntegerScaleNormalizedBit);
  for (uint32_t i = 0; i < 4; ++i) {
    INFO("component " << i);
    CHECK(FieldToMultiplier(bits, i) == 255);
  }
}

TEST_CASE("IntegerScaleBits follows the guest-visible component order",
          "[integer_scale][ordering]") {
  // k_2_10_10_10 is asymmetric: source widths are 10, 10, 10, 2. A swizzle that
  // moves the 2-bit component is the only way to catch a derivation that
  // resolves widths in the wrong direction, which JC2's uniform 8/8/8/8 class
  // could never expose.
  const uint8_t unsigned_signs = UniformSigns(TextureSign::kUnsigned);

  SECTION("identity swizzle keeps source widths in place") {
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_2_10_10_10, kInteger, kSwizzleIdentity),
        unsigned_signs);
    CHECK(FieldToMultiplier(bits, 0) == 1023);
    CHECK(FieldToMultiplier(bits, 1) == 1023);
    CHECK(FieldToMultiplier(bits, 2) == 1023);
    CHECK(FieldToMultiplier(bits, 3) == 3);
  }

  SECTION("reversed swizzle moves the narrow component to output 0") {
    // Output 0 reads source 3 (2 bits), output 3 reads source 0 (10 bits).
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_2_10_10_10, kInteger,
                  MakeSwizzle(3, 2, 1, 0)),
        unsigned_signs);
    CHECK(FieldToMultiplier(bits, 0) == 3);
    CHECK(FieldToMultiplier(bits, 1) == 1023);
    CHECK(FieldToMultiplier(bits, 2) == 1023);
    CHECK(FieldToMultiplier(bits, 3) == 1023);
  }

  SECTION("broadcasting one source component to every output") {
    // All four outputs read source 3, the 2-bit component.
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_2_10_10_10, kInteger,
                  MakeSwizzle(3, 3, 3, 3)),
        unsigned_signs);
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(FieldToMultiplier(bits, i) == 3);
    }
  }

  SECTION("a truly asymmetric format resolves each output independently") {
    // k_5_6_5 source widths are 5, 6, 5. Output 0 <- source 1 (6 bits),
    // output 1 <- source 0 (5 bits), output 2 <- constant 1, output 3 <-
    // source 2 (5 bits). Every field is distinguishable.
    uint32_t swizzle = MakeSwizzle(1, 0, xenos::XE_GPU_TEXTURE_SWIZZLE_1, 2);
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_5_6_5, kInteger, swizzle), unsigned_signs);
    CHECK(FieldToMultiplier(bits, 0) == 63);
    CHECK(FieldToMultiplier(bits, 1) == 31);
    CHECK(FieldOf(bits, 2) == 0);  // constant lane contributes nothing
    CHECK(FieldToMultiplier(bits, 3) == 31);
  }

  SECTION("a lane reading past the last stored component takes its width") {
    // k_5_6_5 stores three components. Upstream 052cb95f2 clamps source 3 to
    // the last stored one (blue, 5 bits), as the host swizzle does.
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_5_6_5, kInteger, MakeSwizzle(3, 3, 3, 3)),
        unsigned_signs);
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(FieldToMultiplier(bits, i) == 31);
    }
  }
}

TEST_CASE("IntegerScaleBits handles per-component signs independently",
          "[integer_scale]") {
  SECTION("signed components use one bit less of range") {
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kInteger, kSwizzleIdentity),
        UniformSigns(TextureSign::kSigned));
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(SignOf(bits, i) == TextureSign::kSigned);
      CHECK(FieldToMultiplier(bits, i) == 127);
    }
  }

  SECTION("an integer gamma component drops out while its neighbours do not") {
    uint8_t signs =
        MakeSigns(TextureSign::kUnsigned, TextureSign::kGamma,
                  TextureSign::kSigned, TextureSign::kUnsigned);
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kInteger, kSwizzleIdentity), signs);
    CHECK(FieldToMultiplier(bits, 0) == 255);
    CHECK(FieldOf(bits, 1) == 0);
    CHECK(FieldToMultiplier(bits, 2) == 127);
    CHECK(FieldToMultiplier(bits, 3) == 255);
  }

  SECTION("a normalized gamma component keeps its sign but no width") {
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kFractional, kSwizzleIdentity),
        UniformSigns(TextureSign::kGamma));
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(SignOf(bits, i) == TextureSign::kGamma);
      CHECK((FieldOf(bits, i) & texture_util::kIntegerScaleWidthMask) == 0);
    }
  }
}

TEST_CASE("Unsigned-biased components carry width for offset-binary decode",
          "[integer_scale][biased]") {
  // Upstream 2ddc5ef73 / c3cd8617b: biased components decode as
  // (n - 2^(w - 1)) / (2^(w - 1) - 1), so an 8-bit 128 returns 0, and integer
  // num_format scales them like signed. Engaged normalized in Halo 3 and
  // Halo 4 normal maps (value-census-20261012).

  SECTION("normalized biased components carry sign and width") {
    uint32_t bits = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kFractional, kSwizzleIdentity),
        UniformSigns(TextureSign::kUnsignedBiased));
    CHECK(bits & texture_util::kIntegerScaleNormalizedBit);
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(SignOf(bits, i) == TextureSign::kUnsignedBiased);
      CHECK((FieldOf(bits, i) & texture_util::kIntegerScaleWidthMask) == 7);
    }
  }

  SECTION("integer biased components scale like signed ones") {
    uint32_t biased = texture_util::IntegerScaleBits(
        MakeFetch(TextureFormat::k_8_8_8_8, kInteger, kSwizzleIdentity),
        UniformSigns(TextureSign::kUnsignedBiased));
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("component " << i);
      CHECK(FieldToMultiplier(biased, i) == 127);
    }
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
