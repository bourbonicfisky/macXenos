/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) coverage for the production Metal scissor attachment clamp.
// Guest register semantics are normalized by draw_util::GetScissor; these tests
// pin the Metal-specific second step that makes the rectangle valid for the
// active render attachments without expanding guest coverage.

#include <cstdint>
#include <limits>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_fixed_function_state.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

draw_util::Scissor Scissor(uint32_t x, uint32_t y, uint32_t width,
                           uint32_t height) {
  return {{x, y}, {width, height}};
}

reg::PA_SU_SC_MODE_CNTL CullMode(bool cull_front, bool cull_back,
                                 bool front_face_clockwise) {
  reg::PA_SU_SC_MODE_CNTL mode = {};
  mode.cull_front = cull_front;
  mode.cull_back = cull_back;
  mode.face = front_face_clockwise;
  return mode;
}

TEST_CASE("Metal polygon cull modes map without stale state",
          "[l0][metal][cull-front-face]") {
  MetalCullFrontFaceState none =
      GetMetalCullFrontFaceState(CullMode(false, false, false), true);
  CHECK(none.cull_mode == MetalCullMode::kNone);
  CHECK_FALSE(none.front_face_clockwise);
  CHECK_FALSE(none.culls_all());

  MetalCullFrontFaceState front =
      GetMetalCullFrontFaceState(CullMode(true, false, false), true);
  CHECK(front.cull_mode == MetalCullMode::kFront);
  CHECK_FALSE(front.front_face_clockwise);

  MetalCullFrontFaceState back =
      GetMetalCullFrontFaceState(CullMode(false, true, false), true);
  CHECK(back.cull_mode == MetalCullMode::kBack);
  CHECK_FALSE(back.front_face_clockwise);

  // Re-deriving no-cull after the two culling variants models both a state
  // transition on one encoder and the first draw of a newly opened encoder.
  MetalCullFrontFaceState reset =
      GetMetalCullFrontFaceState(CullMode(false, false, false), true);
  CHECK(reset.cull_mode == MetalCullMode::kNone);
  CHECK_FALSE(reset.front_face_clockwise);
}

TEST_CASE("Metal polygon winding follows Xenos face state",
          "[l0][metal][cull-front-face]") {
  CHECK_FALSE(GetMetalCullFrontFaceState(CullMode(false, false, false), true)
                  .front_face_clockwise);
  CHECK(GetMetalCullFrontFaceState(CullMode(false, false, true), true)
            .front_face_clockwise);
}

TEST_CASE("Metal both-face culling is a successful raster no-op",
          "[l0][metal][cull-front-face]") {
  MetalCullFrontFaceState state =
      GetMetalCullFrontFaceState(CullMode(true, true, true), true);
  CHECK(state.cull_mode == MetalCullMode::kAll);
  CHECK(state.front_face_clockwise);
  CHECK(state.culls_all());
}

TEST_CASE("Metal non-polygons always establish safe no-cull state",
          "[l0][metal][cull-front-face]") {
  MetalCullFrontFaceState state =
      GetMetalCullFrontFaceState(CullMode(true, true, true), false);
  CHECK(state.cull_mode == MetalCullMode::kNone);
  CHECK_FALSE(state.front_face_clockwise);
  CHECK_FALSE(state.culls_all());
}

TEST_CASE("Metal scissor preserves in-bounds coverage", "[l0][metal][scissor]") {
  MetalScissorRect rect =
      ClampScissorToRenderExtent(Scissor(10, 20, 30, 40), 1280, 720);
  CHECK(rect.x == 10);
  CHECK(rect.y == 20);
  CHECK(rect.width == 30);
  CHECK(rect.height == 40);
  CHECK_FALSE(rect.empty());
}

TEST_CASE("Metal scissor clips right and bottom coverage",
          "[l0][metal][scissor]") {
  MetalScissorRect rect =
      ClampScissorToRenderExtent(Scissor(1200, 700, 200, 100), 1280, 720);
  CHECK(rect.x == 1200);
  CHECK(rect.y == 700);
  CHECK(rect.width == 80);
  CHECK(rect.height == 20);
}

TEST_CASE("Metal scissor collapses fully out-of-bounds coverage",
          "[l0][metal][scissor]") {
  MetalScissorRect rect =
      ClampScissorToRenderExtent(Scissor(1400, 900, 50, 50), 1280, 720);
  CHECK(rect.x == 1280);
  CHECK(rect.y == 720);
  CHECK(rect.width == 0);
  CHECK(rect.height == 0);
  CHECK(rect.empty());
}

TEST_CASE("Metal scissor preserves empty guest coverage",
          "[l0][metal][scissor]") {
  CHECK(ClampScissorToRenderExtent(Scissor(10, 20, 0, 40), 1280, 720)
            .empty());
  CHECK(ClampScissorToRenderExtent(Scissor(10, 20, 30, 0), 1280, 720)
            .empty());
}

TEST_CASE("Metal scissor is empty for a zero-sized attachment",
          "[l0][metal][scissor]") {
  MetalScissorRect rect =
      ClampScissorToRenderExtent(Scissor(0, 0, 100, 100), 0, 0);
  CHECK(rect.x == 0);
  CHECK(rect.y == 0);
  CHECK(rect.empty());
}

TEST_CASE("Metal scissor end arithmetic does not wrap",
          "[l0][metal][scissor]") {
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
  MetalScissorRect rect =
      ClampScissorToRenderExtent(Scissor(kMax - 3, kMax - 7, 16, 32), kMax,
                                 kMax);
  CHECK(rect.x == kMax - 3);
  CHECK(rect.y == kMax - 7);
  CHECK(rect.width == 3);
  CHECK(rect.height == 7);
  CHECK_FALSE(rect.empty());
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
