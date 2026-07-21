/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Device-free coverage for the production Stage 19 normalization seams used
// by Metal's translated-shader system-constant upload.

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/metal/metal_fixed_function_state.h"

namespace xe {
namespace gpu {
namespace {

void SetFloatRegister(RegisterFile& regs, uint32_t index, float value) {
  regs[index] = xe::memory::Reinterpret<uint32_t>(value);
}

draw_util::ViewportInfo GetMetalViewport(
    xenos::DepthRenderTargetFormat depth_format, bool z_enable,
    bool pixel_shader_writes_depth) {
  RegisterFile regs;
  reg::RB_DEPTH_INFO depth_info = {};
  depth_info.depth_format = depth_format;
  regs[reg::RB_DEPTH_INFO::register_index] = depth_info.value;
  reg::RB_DEPTHCONTROL normalized_depth = {};
  normalized_depth.z_enable = z_enable;
  draw_util::GetViewportInfoArgs args = metal::GetMetalViewportInfoArgs(
      regs, normalized_depth, pixel_shader_writes_depth);
  draw_util::ViewportInfo viewport = {};
  draw_util::GetHostViewportInfo(&args, viewport);
  return viewport;
}

TEST_CASE("System constants tightly pack sparse user clip planes",
          "[l0][metal][system-constants]") {
  RegisterFile regs;
  reg::PA_CL_CLIP_CNTL clip_control = {};
  clip_control.ucp_ena = (1u << 1) | (1u << 3) | (1u << 5);
  regs[reg::PA_CL_CLIP_CNTL::register_index] = clip_control.value;
  for (uint32_t plane = 0; plane < 6; ++plane) {
    for (uint32_t component = 0; component < 4; ++component) {
      SetFloatRegister(regs, XE_GPU_REG_PA_CL_UCP_0_X + plane * 4 + component,
                       float(plane * 10 + component));
    }
  }

  draw_util::NormalizedUserClipPlanes normalized =
      draw_util::GetNormalizedUserClipPlanes(regs);
  REQUIRE(normalized.count == 3);
  for (uint32_t packed_plane = 0; packed_plane < 3; ++packed_plane) {
    uint32_t guest_plane = 1 + packed_plane * 2;
    for (uint32_t component = 0; component < 4; ++component) {
      CHECK(normalized.equations[packed_plane][component] ==
            float(guest_plane * 10 + component));
    }
  }
  CHECK(normalized.equations[3][0] == 0.0f);
}

TEST_CASE("System constants disable user clip planes with guest clipping",
          "[l0][metal][system-constants]") {
  RegisterFile regs;
  reg::PA_CL_CLIP_CNTL clip_control = {};
  clip_control.ucp_ena = 0b111111;
  clip_control.clip_disable = 1;
  regs[reg::PA_CL_CLIP_CNTL::register_index] = clip_control.value;
  SetFloatRegister(regs, XE_GPU_REG_PA_CL_UCP_0_X, 1.0f);

  draw_util::NormalizedUserClipPlanes normalized =
      draw_util::GetNormalizedUserClipPlanes(regs);
  CHECK(normalized.count == 0);
  CHECK(normalized.equations[0][0] == 0.0f);
}

TEST_CASE("System constants normalize point diameters and viewport axes",
          "[l0][metal][system-constants]") {
  RegisterFile regs;
  reg::VGT_DRAW_INITIATOR draw = {};
  draw.prim_type = xenos::PrimitiveType::kPointList;
  regs[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  reg::PA_SU_POINT_MINMAX point_minmax = {};
  point_minmax.min_size = 8;
  point_minmax.max_size = 24;
  regs[reg::PA_SU_POINT_MINMAX::register_index] = point_minmax.value;
  reg::PA_SU_POINT_SIZE point_size = {};
  point_size.width = 40;
  point_size.height = 16;
  regs[reg::PA_SU_POINT_SIZE::register_index] = point_size.value;
  draw_util::ViewportInfo viewport = {};
  viewport.xy_extent[0] = 256;
  viewport.xy_extent[1] = 128;

  draw_util::NormalizedPointParameters normalized =
      draw_util::GetNormalizedPointParameters(regs, viewport, 2, 4);
  CHECK(normalized.vertex_diameter_min == 1.0f);
  CHECK(normalized.vertex_diameter_max == 3.0f);
  CHECK(normalized.constant_diameter[0] == 5.0f);
  CHECK(normalized.constant_diameter[1] == 2.0f);
  CHECK(normalized.screen_diameter_to_ndc_radius[0] == 2.0f / 256.0f);
  CHECK(normalized.screen_diameter_to_ndc_radius[1] == 4.0f / 128.0f);
}

TEST_CASE("System constants keep point normalization finite and scoped",
          "[l0][metal][system-constants]") {
  RegisterFile regs;
  reg::VGT_DRAW_INITIATOR draw = {};
  draw.prim_type = xenos::PrimitiveType::kPointList;
  regs[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  draw_util::ViewportInfo viewport = {};

  draw_util::NormalizedPointParameters zero_extent =
      draw_util::GetNormalizedPointParameters(regs, viewport, 1, 2);
  CHECK(zero_extent.screen_diameter_to_ndc_radius[0] == 1.0f);
  CHECK(zero_extent.screen_diameter_to_ndc_radius[1] == 2.0f);

  draw.prim_type = xenos::PrimitiveType::kTriangleList;
  regs[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  draw_util::NormalizedPointParameters nonpoint =
      draw_util::GetNormalizedPointParameters(regs, viewport, 1, 1);
  CHECK(nonpoint.vertex_diameter_min == 0.0f);
  CHECK(nonpoint.vertex_diameter_max == 0.0f);
  CHECK(nonpoint.constant_diameter[0] == 0.0f);
  CHECK(nonpoint.constant_diameter[1] == 0.0f);
  CHECK(nonpoint.screen_diameter_to_ndc_radius[0] == 0.0f);
  CHECK(nonpoint.screen_diameter_to_ndc_radius[1] == 0.0f);
}

TEST_CASE("Metal system constants use shared primitive and typed index state",
          "[l0][metal][system-constants]") {
  RegisterFile regs;
  reg::VGT_DRAW_INITIATOR draw = {};
  draw.prim_type = xenos::PrimitiveType::kRectangleList;
  regs[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  reg::RB_DEPTH_INFO depth_info = {};
  depth_info.depth_format = xenos::DepthRenderTargetFormat::kD24FS8;
  regs[reg::RB_DEPTH_INFO::register_index] = depth_info.value;
  regs[reg::VGT_INDX_OFFSET::register_index] = 0xAB123456u;
  regs[reg::VGT_MIN_VTX_INDX::register_index] = 0xCD000010u;
  regs[reg::VGT_MAX_VTX_INDX::register_index] = 0xEF000020u;

  metal::MetalSystemConstantGuestState state =
      metal::GetMetalSystemConstantGuestState(regs);
  CHECK_FALSE(state.primitive_polygonal);
  CHECK(state.depth_float24);
  CHECK(state.vertex_index_offset == 0x123456u);
  CHECK(state.vertex_index_min == 0x10u);
  CHECK(state.vertex_index_max == 0x20u);

  draw.prim_type = xenos::PrimitiveType::kTriangleList;
  regs[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  depth_info.depth_format = xenos::DepthRenderTargetFormat::kD24S8;
  regs[reg::RB_DEPTH_INFO::register_index] = depth_info.value;
  state = metal::GetMetalSystemConstantGuestState(regs);
  CHECK(state.primitive_polygonal);
  CHECK_FALSE(state.depth_float24);
}

TEST_CASE("Metal viewport arguments keep one canonical float24 domain",
          "[l0][metal][system-constants]") {
  RegisterFile regs;
  reg::RB_DEPTH_INFO depth_info = {};
  depth_info.depth_format = xenos::DepthRenderTargetFormat::kD24FS8;
  regs[reg::RB_DEPTH_INFO::register_index] = depth_info.value;
  reg::RB_DEPTHCONTROL normalized_depth = {};
  normalized_depth.z_enable = 1;

  draw_util::GetViewportInfoArgs interpolated =
      metal::GetMetalViewportInfoArgs(regs, normalized_depth, false);
  CHECK(interpolated.normalized_depth_control.value == normalized_depth.value);
  CHECK_FALSE(interpolated.convert_z_to_float24);
  CHECK(interpolated.full_float24_in_0_to_1);
  CHECK_FALSE(interpolated.pixel_shader_writes_depth);
  CHECK(interpolated.depth_format == xenos::DepthRenderTargetFormat::kD24FS8);

  draw_util::GetViewportInfoArgs explicit_depth =
      metal::GetMetalViewportInfoArgs(regs, normalized_depth, true);
  CHECK(explicit_depth.full_float24_in_0_to_1);
  CHECK(explicit_depth.pixel_shader_writes_depth);

  depth_info.depth_format = xenos::DepthRenderTargetFormat::kD24S8;
  regs[reg::RB_DEPTH_INFO::register_index] = depth_info.value;
  draw_util::GetViewportInfoArgs unorm =
      metal::GetMetalViewportInfoArgs(regs, normalized_depth, false);
  CHECK(unorm.depth_format == xenos::DepthRenderTargetFormat::kD24S8);
  CHECK_FALSE(metal::GetMetalSystemConstantGuestState(regs).depth_float24);
}

TEST_CASE("Metal float24 viewport uses one host depth domain",
          "[l0][metal][system-constants]") {
  draw_util::ViewportInfo interpolated = GetMetalViewport(
      xenos::DepthRenderTargetFormat::kD24FS8, true, false);
  CHECK(interpolated.z_min == 0.0f);
  CHECK(interpolated.z_max == 0.5f);

  draw_util::ViewportInfo explicit_depth = GetMetalViewport(
      xenos::DepthRenderTargetFormat::kD24FS8, true, true);
  CHECK(explicit_depth.z_min == 0.0f);
  CHECK(explicit_depth.z_max == 0.5f);
}

TEST_CASE("Metal unorm and disabled depth avoid float24 viewport remap",
          "[l0][metal][system-constants]") {
  draw_util::ViewportInfo unorm = GetMetalViewport(
      xenos::DepthRenderTargetFormat::kD24S8, true, false);
  CHECK(unorm.z_min == 0.0f);
  CHECK(unorm.z_max == 1.0f);

  draw_util::ViewportInfo disabled = GetMetalViewport(
      xenos::DepthRenderTargetFormat::kD24FS8, false, false);
  CHECK(disabled.z_min == 0.0f);
  CHECK(disabled.z_max == 1.0f);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
