/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) coverage for production Metal fixed-function state
// derivation. Guest register semantics remain shared; these tests pin the
// Metal-specific mappings, reset behavior, numeric conversion, and attachment
// clamping without requiring a device or Objective-C++.

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

TEST_CASE("Metal triangle fill mode follows visible dual-mode faces",
          "[l0][metal][fill-depth-bias]") {
  reg::PA_SU_SC_MODE_CNTL mode = {};
  mode.poly_mode = xenos::PolygonModeEnable::kDualMode;
  mode.polymode_front_ptype = xenos::PolygonType::kLines;
  mode.polymode_back_ptype = xenos::PolygonType::kTriangles;
  CHECK(GetMetalTriangleFillMode(mode, true) ==
        MetalTriangleFillMode::kLines);

  // A culled line face must not force the visible triangle face to wireframe.
  mode.cull_front = 1;
  CHECK(GetMetalTriangleFillMode(mode, true) ==
        MetalTriangleFillMode::kFill);

  // Metal has no point polygon fill, so preserve the non-fill intent as lines.
  mode.cull_front = 0;
  mode.polymode_front_ptype = xenos::PolygonType::kPoints;
  CHECK(GetMetalTriangleFillMode(mode, true) ==
        MetalTriangleFillMode::kLines);
}

TEST_CASE("Metal fill state resets across disabled and non-polygon draws",
          "[l0][metal][fill-depth-bias]") {
  reg::PA_SU_SC_MODE_CNTL mode = {};
  mode.poly_mode = xenos::PolygonModeEnable::kDualMode;
  mode.polymode_front_ptype = xenos::PolygonType::kLines;
  mode.polymode_back_ptype = xenos::PolygonType::kLines;
  CHECK(GetMetalTriangleFillMode(mode, true) ==
        MetalTriangleFillMode::kLines);

  mode.poly_mode = xenos::PolygonModeEnable::kDisabled;
  CHECK(GetMetalTriangleFillMode(mode, true) ==
        MetalTriangleFillMode::kFill);
  mode.poly_mode = xenos::PolygonModeEnable::kDualMode;
  CHECK(GetMetalTriangleFillMode(mode, false) ==
        MetalTriangleFillMode::kFill);
}

TEST_CASE("Metal constant depth bias depends on the guest depth format",
          "[l0][metal][fill-depth-bias]") {
  reg::PA_SU_SC_MODE_CNTL mode = {};
  mode.poly_offset_front_enable = 1;
  MetalFillDepthBiasState unorm = GetMetalFillDepthBiasState(
      mode, true, xenos::DepthRenderTargetFormat::kD24S8, 0.0f, 0.000002f);
  MetalFillDepthBiasState floating = GetMetalFillDepthBiasState(
      mode, true, xenos::DepthRenderTargetFormat::kD24FS8, 0.0f, 0.000002f);
  CHECK(unorm.depth_bias_constant == 34.0f);
  CHECK(floating.depth_bias_constant == 40.0f);
  CHECK(unorm.depth_bias_constant != floating.depth_bias_constant);
}

TEST_CASE("Metal slope bias uses preferred visible face and resolution scale",
          "[l0][metal][fill-depth-bias]") {
  reg::PA_SU_SC_MODE_CNTL mode = {};
  mode.poly_offset_front_enable = 1;
  mode.poly_offset_back_enable = 1;
  MetalFillDepthBiasState front = GetMetalFillDepthBiasState(
      mode, true, xenos::DepthRenderTargetFormat::kD24S8, 32.0f, 0.0f, 2);
  CHECK(front.depth_bias_slope == 4.0f);

  // The shared preferred-face selector returns the back-face value when the
  // front is culled; verify that value is converted without stale front state.
  mode.cull_front = 1;
  MetalFillDepthBiasState back = GetMetalFillDepthBiasState(
      mode, true, xenos::DepthRenderTargetFormat::kD24S8, 8.0f, 0.0f, 2);
  CHECK(back.depth_bias_slope == 1.0f);
}

TEST_CASE("Metal disabled and non-polygon bias state does not leak",
          "[l0][metal][fill-depth-bias]") {
  reg::PA_SU_SC_MODE_CNTL mode = {};
  MetalFillDepthBiasState polygon = GetMetalFillDepthBiasState(
      mode, true, xenos::DepthRenderTargetFormat::kD24S8, 0.0f, 0.0f);
  CHECK(polygon.depth_bias_constant == 0.0f);
  CHECK(polygon.depth_bias_slope == 0.0f);

  mode.poly_offset_front_enable = 1;
  // GetPreferredFacePolygonOffset returns zero for a non-polygon unless the
  // para enable is set, even if the front-face enable/value is nonzero.
  MetalFillDepthBiasState non_polygon = GetMetalFillDepthBiasState(
      mode, false, xenos::DepthRenderTargetFormat::kD24S8, 0.0f, 0.0f);
  CHECK(non_polygon.depth_bias_constant == 0.0f);
  CHECK(non_polygon.depth_bias_slope == 0.0f);

  mode.poly_offset_para_enable = 1;
  MetalFillDepthBiasState enabled_non_polygon = GetMetalFillDepthBiasState(
      mode, false, xenos::DepthRenderTargetFormat::kD24S8, 16.0f, 0.25f);
  CHECK(enabled_non_polygon.depth_bias_constant != 0.0f);
  CHECK(enabled_non_polygon.depth_bias_slope == 1.0f);
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
