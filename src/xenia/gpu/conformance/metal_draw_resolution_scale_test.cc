/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Draw resolution scale configuration on Metal
// (metal_draw_resolution_scale.h): per-axis clamping to 1..3, and a session
// that renders at 1x1 until scaled rendering is enabled.

#include <algorithm>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_draw_resolution_scale.h"

namespace xe {
namespace gpu {
namespace test {

using metal::kMetalMaxDrawResolutionScale;
using metal::MetalScaledPixelShiftHostPixels;
using metal::ResolveMetalDrawResolutionScale;

TEST_CASE("Draw resolution scale: in-range values pass through",
          "[metal][draw_resolution_scale]") {
  auto scale = ResolveMetalDrawResolutionScale(2, 3, true);
  CHECK(scale.requested_x == 2);
  CHECK(scale.requested_y == 3);
  CHECK_FALSE(scale.clamped);
  CHECK(scale.x == 2);
  CHECK(scale.y == 3);
}

TEST_CASE("Draw resolution scale: each axis clamps to 1..3",
          "[metal][draw_resolution_scale]") {
  auto high = ResolveMetalDrawResolutionScale(7, 2, true);
  CHECK(high.requested_x == 3);
  CHECK(high.requested_y == 2);
  CHECK(high.clamped);
  auto low = ResolveMetalDrawResolutionScale(1, 0, true);
  CHECK(low.requested_x == 1);
  CHECK(low.requested_y == 1);
  CHECK(low.clamped);
  auto negative = ResolveMetalDrawResolutionScale(-4, 1, true);
  CHECK(negative.requested_x == 1);
  CHECK(negative.clamped);
}

TEST_CASE("Draw resolution scale: renders at 1x until scaling is enabled",
          "[metal][draw_resolution_scale]") {
  auto scale = ResolveMetalDrawResolutionScale(2, 2, false);
  CHECK(scale.requested_x == 2);
  CHECK(scale.requested_y == 2);
  CHECK(scale.x == 1);
  CHECK(scale.y == 1);
}

// D4a: with the shift, the representative host pixel (scale >> 1) of guest
// pixel g samples exactly g's 1x sample point, and the pixel position the
// translator gives it is exactly g.
TEST_CASE("Draw resolution scale: the representative pixel is 1x-exact",
          "[metal][draw_resolution_scale]") {
  CHECK(MetalScaledPixelShiftHostPixels(1) == 0.0f);
  CHECK(MetalScaledPixelShiftHostPixels(2) == 0.5f);
  CHECK(MetalScaledPixelShiftHostPixels(3) == 0.0f);
  for (uint32_t scale = 1; scale <= kMetalMaxDrawResolutionScale; ++scale) {
    float shift = MetalScaledPixelShiftHostPixels(scale);
    for (uint32_t g = 0; g < 4; ++g) {
      uint32_t host = g * scale + (scale >> 1);
      // The scene point the host pixel center samples, in guest pixels (the
      // 1x sample point of pixel g is g + 0.5).
      float sampled = (float(host) + 0.5f - shift) / float(scale);
      CHECK(sampled == float(g) + 0.5f);
      // The translator's pixel position.
      float position = std::max(
          (float(host) + 0.5f - shift) / float(scale) - 0.5f, 0.0f);
      CHECK(position == float(g));
    }
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
