/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Legacy Metal texture cache: CPU texel decoders and host/guest swizzle
// composition (openspec/changes/implement-metal-texture-formats, task 2.4).
// The decoders are checked against the shared xenos conversions they must
// equal; the composition is checked against the Xenos swizzle semantics.

#include <cmath>
#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_texture_format_decode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

using metal::ComposeGuestHostSwizzle;
using metal::DecodeDepthFloatTexel;
using metal::DecodeDepthUnormTexel;

TEST_CASE("Metal depth unorm texel: word >> 8 through UNorm24To32",
          "[metal][texture][format-decode]") {
  REQUIRE(DecodeDepthUnormTexel(0x00000000u) == 0.0f);
  REQUIRE(DecodeDepthUnormTexel(0xFFFFFFFFu) == 1.0f);
  // Stencil byte (low 8 bits) is ignored.
  REQUIRE(DecodeDepthUnormTexel(0x000000FFu) == 0.0f);
  REQUIRE(DecodeDepthUnormTexel(0x800000AAu) ==
          xenos::UNorm24To32(0x800000u));
  for (uint32_t d = 0; d < 0x1000000u; d += 0x12345u) {
    REQUIRE(DecodeDepthUnormTexel((d << 8) | 0x5Au) == xenos::UNorm24To32(d));
  }
}

TEST_CASE("Metal depth float texel: word >> 8 through Float20e4To32",
          "[metal][texture][format-decode]") {
  REQUIRE(DecodeDepthFloatTexel(0x00000000u) == 0.0f);
  for (float f : {0.0f, 0.25f, 0.5f, 0.75f, 0.999f, 1.0f, 1.5f, 2.0f}) {
    uint32_t e = xenos::Float32To20e4(f, false);
    REQUIRE(DecodeDepthFloatTexel((e << 8) | 0x3Cu) == xenos::Float20e4To32(e));
  }
}

TEST_CASE("Metal swizzle composition: identity host keeps the guest swizzle",
          "[metal][texture][format-decode]") {
  for (uint32_t g = 0; g < 0x1000u; g += 0x49u) {
    REQUIRE(ComposeGuestHostSwizzle(g, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA) ==
            g);
  }
}

TEST_CASE("Metal swizzle composition: RGGG and RRRR redirect channel reads",
          "[metal][texture][format-decode]") {
  // Guest identity XYZW over an RGGG host reads R,G,G,G.
  REQUIRE(ComposeGuestHostSwizzle(xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
                                  xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG) ==
          xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG);
  // Guest .w over RRRR reads R.
  uint32_t guest_wwww = XE_GPU_MAKE_TEXTURE_SWIZZLE(A, A, A, A);
  REQUIRE(ComposeGuestHostSwizzle(guest_wwww,
                                  xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR) ==
          xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR);
  // Constant selections (0 = 4, 1 = 5) pass through untouched.
  uint32_t guest_const = (4u << 0) | (5u << 3) | (0u << 6) | (1u << 9);
  uint32_t composed =
      ComposeGuestHostSwizzle(guest_const, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG);
  REQUIRE(((composed >> 0) & 7u) == 4u);
  REQUIRE(((composed >> 3) & 7u) == 5u);
  REQUIRE(((composed >> 6) & 7u) == 0u);  // X -> R
  REQUIRE(((composed >> 9) & 7u) == 1u);  // Y -> G
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
