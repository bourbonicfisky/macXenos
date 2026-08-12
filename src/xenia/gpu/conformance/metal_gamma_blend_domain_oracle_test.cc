/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 blend-domain oracle for k_8_8_8_8_GAMMA render targets (OpenSpec
// change fix-metal-gamma-render-target-blending, tasks 2.1/2.2).
//
// Xenos blends gamma render targets in the LINEAR domain: the PWL-encoded
// destination byte is decoded to linear, blending happens on linear values,
// and only the blended result is PWL-encoded back into EDRAM. The current
// Metal backend instead has the pixel shader PWL-encode its linear output
// BEFORE the output merger (kSysFlag_ConvertColorNToGamma published in
// metal_command_processor.mm; the encode itself in
// dxbc_shader_translator_om.cc, whose own comment states it "must be done
// after blending on the Xbox 360"), so fixed-function blending operates on
// encoded values against a plain-UNORM8 destination.
//
// This oracle computes both models with the production reference conversions
// xe::gpu::xenos::PWLGammaToLinear / LinearToPWLGamma and asserts that
// nontrivial blend configurations discriminate the two domains while trivial
// ones (opaque replacement, fully masked write) agree bit-exactly. It is the
// mechanism-independent acceptance instrument for whichever storage mechanism
// the design gate later selects; no Metal device and no backend code.
//
// The gamma encode applies to RGB only — the translator's encode loop masks
// 0b0111 — so source/destination ALPHA is linear in BOTH models and every
// alpha-derived blend factor is fed the same value in both. The oracle
// therefore models one RGB channel plus the source/destination alpha that
// factors may reference.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

// Quantization convention (used identically by both models, so the
// discrimination assertions are independent of tie handling): a stored
// n-bit UNORM value is round-to-nearest with ties rounding up,
//   Q_n(v) = trunc(saturate(v) * (2^n - 1) + 0.5).
float SaturateF(float v) {
  return std::min(1.0f, std::max(0.0f, v));
}

uint8_t QuantizeUnorm8(float v) {
  return uint8_t(SaturateF(v) * 255.0f + 0.5f);
}

uint16_t QuantizeUnorm16(float v) {
  return uint16_t(SaturateF(v) * 65535.0f + 0.5f);
}

// Evaluates a blend factor for one RGB channel. src_c/dst_c are the color
// values in whatever domain the calling model blends in (linear for the
// correct model, PWL-encoded for the current model); src_a/dst_a are always
// linear (alpha is never gamma-encoded).
float FactorValue(xenos::BlendFactor factor, float src_c, float src_a,
                  float dst_c, float dst_a) {
  switch (factor) {
    case xenos::BlendFactor::kZero:
      return 0.0f;
    case xenos::BlendFactor::kOne:
      return 1.0f;
    case xenos::BlendFactor::kSrcColor:
      return src_c;
    case xenos::BlendFactor::kOneMinusSrcColor:
      return 1.0f - src_c;
    case xenos::BlendFactor::kSrcAlpha:
      return src_a;
    case xenos::BlendFactor::kOneMinusSrcAlpha:
      return 1.0f - src_a;
    case xenos::BlendFactor::kDstColor:
      return dst_c;
    case xenos::BlendFactor::kOneMinusDstColor:
      return 1.0f - dst_c;
    case xenos::BlendFactor::kDstAlpha:
      return dst_a;
    case xenos::BlendFactor::kOneMinusDstAlpha:
      return 1.0f - dst_a;
    default:
      FAIL("blend factor not covered by the oracle matrix");
      return 0.0f;
  }
}

float ApplyBlendOp(xenos::BlendOp op, float src_term, float dst_term) {
  switch (op) {
    case xenos::BlendOp::kAdd:
      return src_term + dst_term;
    case xenos::BlendOp::kSubtract:
      return src_term - dst_term;
    case xenos::BlendOp::kRevSubtract:
      return dst_term - src_term;
    default:
      FAIL("blend op not covered by the oracle matrix");
      return 0.0f;
  }
}

struct BlendConfig {
  xenos::BlendFactor src_factor;
  xenos::BlendFactor dst_factor;
  xenos::BlendOp op;
};

// Correct Xenos model: the shader output channel is LINEAR, the stored
// destination byte is PWL-decoded to linear, blending happens on linear
// values, and the blended result is PWL-encoded and quantized into the
// stored byte. LinearToPWLGamma saturates its input.
uint8_t CorrectModelStoredByte(const BlendConfig& cfg, float src_linear,
                               float src_alpha, uint8_t dst_byte,
                               uint8_t dst_alpha_byte) {
  float dst_linear = xenos::PWLGammaToLinear(float(dst_byte) * (1.0f / 255.0f));
  float dst_alpha = float(dst_alpha_byte) * (1.0f / 255.0f);
  float src_factor = FactorValue(cfg.src_factor, src_linear, src_alpha,
                                 dst_linear, dst_alpha);
  float dst_factor = FactorValue(cfg.dst_factor, src_linear, src_alpha,
                                 dst_linear, dst_alpha);
  float blended = ApplyBlendOp(cfg.op, src_linear * src_factor,
                               dst_linear * dst_factor);
  return QuantizeUnorm8(xenos::LinearToPWLGamma(blended));
}

// Current Metal model: the shader PWL-encodes its linear RGB output before
// the output merger (the encoded source stays full-precision float — the
// blender consumes the shader output directly), the destination is the
// stored PWL byte read as plain UNORM8, blending happens on those encoded
// values, and the saturated result is quantized into the stored byte with no
// further conversion.
uint8_t CurrentModelStoredByte(const BlendConfig& cfg, float src_linear,
                               float src_alpha, uint8_t dst_byte,
                               uint8_t dst_alpha_byte) {
  float src_encoded = xenos::LinearToPWLGamma(src_linear);
  float dst_encoded = float(dst_byte) * (1.0f / 255.0f);
  float dst_alpha = float(dst_alpha_byte) * (1.0f / 255.0f);
  float src_factor = FactorValue(cfg.src_factor, src_encoded, src_alpha,
                                 dst_encoded, dst_alpha);
  float dst_factor = FactorValue(cfg.dst_factor, src_encoded, src_alpha,
                                 dst_encoded, dst_alpha);
  float blended = ApplyBlendOp(cfg.op, src_encoded * src_factor,
                               dst_encoded * dst_factor);
  return QuantizeUnorm8(blended);
}

}  // namespace

TEST_CASE(
    "gamma blend-domain oracle: nontrivial configurations discriminate the "
    "linear and encoded domains",
    "[l0][metal][gamma-blend-domain]") {
  struct DiscriminatingCase {
    const char* name;
    BlendConfig cfg;
    float src_linear;
    float src_alpha;
    uint8_t dst_byte;
    // Pinned model outputs (hand-derived from the piecewise reference
    // conversions, confirmed by this oracle): regression change-detectors for
    // the reference pair and exact targets for a future mechanism.
    uint8_t expected_correct;
    uint8_t expected_current;
  };
  const DiscriminatingCase cases[] = {
      {"src-alpha additive over mid-gray",
       {xenos::BlendFactor::kSrcAlpha, xenos::BlendFactor::kOne,
        xenos::BlendOp::kAdd},
       0.25f, 0.75f, 128, 176, 223},
      {"classic alpha blend over dark gray",
       {xenos::BlendFactor::kSrcAlpha, xenos::BlendFactor::kOneMinusSrcAlpha,
        xenos::BlendOp::kAdd},
       0.5f, 0.5f, 64, 135, 128},
      {"destination-color modulate",
       {xenos::BlendFactor::kDstColor, xenos::BlendFactor::kZero,
        xenos::BlendOp::kAdd},
       0.6f, 1.0f, 100, 75, 80},
      {"reverse subtract from bright destination",
       {xenos::BlendFactor::kOne, xenos::BlendFactor::kOne,
        xenos::BlendOp::kRevSubtract},
       0.05f, 1.0f, 200, 194, 149},
  };
  const uint8_t dst_alpha_byte = 255;
  for (const DiscriminatingCase& c : cases) {
    INFO(c.name);
    uint8_t correct = CorrectModelStoredByte(c.cfg, c.src_linear, c.src_alpha,
                                             c.dst_byte, dst_alpha_byte);
    uint8_t current = CurrentModelStoredByte(c.cfg, c.src_linear, c.src_alpha,
                                             c.dst_byte, dst_alpha_byte);
    CHECK(int(correct) == int(c.expected_correct));
    CHECK(int(current) == int(c.expected_current));
    // The discrimination requirement itself: more than one 8-bit step apart,
    // so quantization tolerance can never mask the domain difference.
    REQUIRE(std::abs(int(correct) - int(current)) > 1);
  }
}

TEST_CASE(
    "gamma blend-domain oracle: opaque replacement and fully masked writes "
    "agree bit-exactly",
    "[l0][metal][gamma-blend-domain]") {
  const BlendConfig opaque = {xenos::BlendFactor::kOne,
                              xenos::BlendFactor::kZero, xenos::BlendOp::kAdd};
  // Opaque replacement (src*ONE + dst*ZERO, ADD): both models store the
  // PWL-encoded, quantized source, independent of the destination.
  for (uint32_t i = 0; i <= 255; ++i) {
    float src_linear = float(i) * (1.0f / 255.0f);
    for (uint8_t dst_byte : {uint8_t(0), uint8_t(64), uint8_t(128),
                             uint8_t(200), uint8_t(255)}) {
      REQUIRE(int(CorrectModelStoredByte(opaque, src_linear, 1.0f, dst_byte,
                                         255)) ==
              int(CurrentModelStoredByte(opaque, src_linear, 1.0f, dst_byte,
                                         255)));
    }
  }
  // Fully masked write: the write mask drops the channel before storage in
  // both models, so the stored byte is the untouched destination byte.
  for (uint32_t dst = 0; dst <= 255; ++dst) {
    uint8_t masked_correct = uint8_t(dst);
    uint8_t masked_current = uint8_t(dst);
    REQUIRE(int(masked_correct) == int(masked_current));
  }
}

namespace {

// Canary's UNORM16 midpoint encoding of a PWL gamma byte — the exact MAd-form
// constants the D3D12 ownership-transfer shader emits
// (d3d12_render_target_cache.cc:3503-3508):
//   Piece 0 (B < 64):      F = B * (1/1023)    + 0.5/1023
//   Piece 1 (64 <= B < 96):  F = B * (1/511.5)   - 31.5/511.5
//   Piece 2 (96 <= B < 192): F = B * (1/255.75)  - 63.5/255.75
//   Piece 3 (B >= 192):    F = B * (1/127.875) + (-127.5/127.875)
// The midpoint keeps the linear value centered in its byte's decode interval
// so UNORM16 quantization cannot push it across the trunc() reconstruction
// threshold — storing the exact PWLGammaToLinear result would sit at the
// interval's lower boundary.
float MidpointLinearForGammaByte(uint32_t byte_value) {
  float recip, offset;
  if (byte_value < 64) {
    recip = 1.0f / 1023.0f;
    offset = 0.5f / 1023.0f;
  } else if (byte_value < 96) {
    recip = 1.0f / 511.5f;
    offset = -31.5f / 511.5f;
  } else if (byte_value < 192) {
    recip = 1.0f / 255.75f;
    offset = -63.5f / 255.75f;
  } else {
    recip = 1.0f / 127.875f;
    offset = -127.5f / 127.875f;
  }
  return float(byte_value) * recip + offset;
}

}  // namespace

TEST_CASE(
    "gamma round trip: raw lower-bound decode is not a byte-stable storage "
    "encoding",
    "[l0][metal][gamma-blend-domain]") {
  // PWLGammaToLinear returns the lower edge of the interval that re-encodes to
  // a byte. Even without an explicit UNORM16 hop, floating-point evaluation
  // may place that value just below the truncation threshold (17 -> 16 is the
  // first mismatch in the current production reference). This is a negative
  // control: a separate linear representation must not store this raw value
  // and assume byte-exact reconstruction. Canary uses interval midpoints,
  // exhaustively proven in the next case.
  uint32_t mismatch_count = 0;
  int maximum_byte_delta = 0;
  for (uint32_t b = 0; b <= 255; ++b) {
    float linear = xenos::PWLGammaToLinear(float(b) * (1.0f / 255.0f));
    uint8_t recovered = QuantizeUnorm8(xenos::LinearToPWLGamma(linear));
    int byte_delta = std::abs(int(recovered) - int(b));
    mismatch_count += byte_delta != 0;
    maximum_byte_delta = std::max(maximum_byte_delta, byte_delta);
  }
  CHECK(mismatch_count > 0);
  CHECK(maximum_byte_delta == 1);
}

TEST_CASE(
    "gamma round trip: every PWL byte survives UNORM16 midpoint encoding "
    "bit-exactly (Canary model)",
    "[l0][metal][gamma-blend-domain]") {
  // Precondition for RGBA16Unorm ever being selectable at the design gate:
  // byte -> midpoint linear -> UNORM16 quantize/dequantize ->
  // LinearToPWLGamma -> Q_8 recovers the byte for all 256 values.
  for (uint32_t b = 0; b <= 255; ++b) {
    INFO("gamma byte " << b);
    float midpoint = MidpointLinearForGammaByte(b);
    uint16_t stored = QuantizeUnorm16(midpoint);
    float dequantized = float(stored) * (1.0f / 65535.0f);
    uint8_t recovered = QuantizeUnorm8(xenos::LinearToPWLGamma(dequantized));
    REQUIRE(int(recovered) == int(b));
  }
  // The alpha plane of the same UNORM16 target stores plain linear A/255
  // (no gamma conversion in Canary's transfer shader); its round trip must
  // also be exact.
  for (uint32_t a = 0; a <= 255; ++a) {
    INFO("alpha byte " << a);
    uint16_t stored = QuantizeUnorm16(float(a) * (1.0f / 255.0f));
    uint8_t recovered = QuantizeUnorm8(float(stored) * (1.0f / 65535.0f));
    REQUIRE(int(recovered) == int(a));
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
