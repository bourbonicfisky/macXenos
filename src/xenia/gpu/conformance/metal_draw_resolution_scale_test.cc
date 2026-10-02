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

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_draw_resolution_scale.h"

namespace xe {
namespace gpu {
namespace test {

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

}  // namespace test
}  // namespace gpu
}  // namespace xe
