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

reg::RB_BLENDCONTROL BlendControl(
    xenos::BlendFactor color_source = xenos::BlendFactor::kOne,
    xenos::BlendFactor color_destination = xenos::BlendFactor::kZero,
    xenos::BlendOp color_operation = xenos::BlendOp::kAdd,
    xenos::BlendFactor alpha_source = xenos::BlendFactor::kOne,
    xenos::BlendFactor alpha_destination = xenos::BlendFactor::kZero,
    xenos::BlendOp alpha_operation = xenos::BlendOp::kAdd) {
  reg::RB_BLENDCONTROL control = {};
  control.color_srcblend = color_source;
  control.color_destblend = color_destination;
  control.color_comb_fcn = color_operation;
  control.alpha_srcblend = alpha_source;
  control.alpha_destblend = alpha_destination;
  control.alpha_comb_fcn = alpha_operation;
  return control;
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

TEST_CASE("Metal depth clipping follows guest clip-disable state",
          "[l0][metal][depth-clip]") {
  reg::PA_CL_CLIP_CNTL clip = {};
  CHECK(GetMetalDepthClipMode(clip) == MetalDepthClipMode::kClip);

  clip.clip_disable = 1;
  CHECK(GetMetalDepthClipMode(clip) == MetalDepthClipMode::kClamp);

  // Re-derivation must restore clipping on a later draw rather than leaking
  // the previous encoder's clamp state.
  clip.clip_disable = 0;
  CHECK(GetMetalDepthClipMode(clip) == MetalDepthClipMode::kClip);
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

TEST_CASE("Metal blend colors follow consecutive guest values",
          "[l0][metal][blend-stencil]") {
  reg::RB_DEPTHCONTROL depth = {};
  reg::RB_STENCILREFMASK stencil = {};
  reg::RB_STENCILREFMASK stencil_back = {};
  MetalBlendStencilState first = GetMetalBlendStencilState(
      0.125f, 0.25f, 0.5f, 1.0f, depth, true, stencil, stencil_back);
  CHECK(first.blend_color_red == 0.125f);
  CHECK(first.blend_color_green == 0.25f);
  CHECK(first.blend_color_blue == 0.5f);
  CHECK(first.blend_color_alpha == 1.0f);

  // Re-deriving all four values models both a consecutive draw transition and
  // the first draw on a newly opened encoder.
  MetalBlendStencilState second = GetMetalBlendStencilState(
      -1.0f, 2.0f, 0.0f, 0.75f, depth, true, stencil, stencil_back);
  CHECK(second.blend_color_red == -1.0f);
  CHECK(second.blend_color_green == 2.0f);
  CHECK(second.blend_color_blue == 0.0f);
  CHECK(second.blend_color_alpha == 0.75f);
}

TEST_CASE("Metal stencil references select normalized polygon faces",
          "[l0][metal][blend-stencil]") {
  reg::RB_DEPTHCONTROL depth = {};
  depth.stencil_enable = 1;
  depth.backface_enable = 1;
  reg::RB_STENCILREFMASK stencil = {};
  stencil.stencilref = 0x12;
  reg::RB_STENCILREFMASK stencil_back = {};
  stencil_back.stencilref = 0xE7;
  MetalBlendStencilState state = GetMetalBlendStencilState(
      0.0f, 0.0f, 0.0f, 0.0f, depth, true, stencil, stencil_back);
  CHECK(state.stencil_enabled);
  CHECK(state.stencil_reference_front == 0x12);
  CHECK(state.stencil_reference_back == 0xE7);
}

TEST_CASE("Metal non-polygons use the front stencil reference",
          "[l0][metal][blend-stencil]") {
  reg::RB_DEPTHCONTROL depth = {};
  depth.stencil_enable = 1;
  depth.backface_enable = 1;
  reg::RB_STENCILREFMASK stencil = {};
  stencil.stencilref = 0x34;
  reg::RB_STENCILREFMASK stencil_back = {};
  stencil_back.stencilref = 0xAB;
  MetalBlendStencilState state = GetMetalBlendStencilState(
      0.0f, 0.0f, 0.0f, 0.0f, depth, false, stencil, stencil_back);
  CHECK(state.stencil_enabled);
  CHECK(state.stencil_reference_front == 0x34);
  CHECK(state.stencil_reference_back == 0x34);

  depth.backface_enable = 0;
  state = GetMetalBlendStencilState(0.0f, 0.0f, 0.0f, 0.0f, depth, true,
                                    stencil, stencil_back);
  CHECK(state.stencil_reference_back == 0x34);
}

TEST_CASE("Metal color binding recognizes every MRT slot",
          "[l0][metal][mrt]") {
  CHECK_FALSE(HasMetalColorTarget(0));
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    CHECK(HasMetalColorTarget(0xFu << (i * 4u)));
    CHECK(HasMetalColorTarget(1u << (i * 4u + 3u)));
  }
  CHECK_FALSE(HasMetalColorTarget(1u << 16u));
}

TEST_CASE("Metal gamma conversion is derived per MRT slot",
          "[l0][metal][mrt][gamma]") {
  std::array<reg::RB_COLOR_INFO, xenos::kMaxColorRenderTargets> color_infos =
      {};
  for (auto& color_info : color_infos) {
    color_info.color_format = xenos::ColorRenderTargetFormat::k_8_8_8_8;
  }
  CHECK(GetMetalGammaRenderTargetMask(color_infos) == 0u);
  color_infos[1].color_format =
      xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
  color_infos[3].color_format =
      xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
  CHECK(GetMetalGammaRenderTargetMask(color_infos) == 0b1010u);
  color_infos[0].color_format =
      xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
  color_infos[2].color_format =
      xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
  CHECK(GetMetalGammaRenderTargetMask(color_infos) == 0b1111u);
}

TEST_CASE("Metal MRT blend state preserves sparse and mixed write nibbles",
          "[l0][metal][mrt][blend]") {
  reg::RB_BLENDCONTROL opaque = BlendControl();

  constexpr uint32_t kRt1OnlyMask = 0x00F0u;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    MetalColorAttachmentBlendState state = GetMetalColorAttachmentBlendState(
        opaque.value, kRt1OnlyMask >> (i * 4u));
    CHECK(state.write_mask == (i == 1 ? 0xFu : 0u));
    CHECK_FALSE(state.blending_required);
  }

  constexpr uint32_t kMixedMask = 0xA05Fu;
  constexpr uint32_t kExpectedMasks[xenos::kMaxColorRenderTargets] = {
      0xFu, 0x5u, 0x0u, 0xAu};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    MetalColorAttachmentBlendState state =
        GetMetalColorAttachmentBlendState(opaque.value, kMixedMask >> (i * 4u));
    CHECK(state.write_mask == kExpectedMasks[i]);
  }
}

TEST_CASE("Metal dormant MRT attachments disable writes and blending",
          "[l0][metal][mrt][blend]") {
  reg::RB_BLENDCONTROL nontrivial = BlendControl(
      xenos::BlendFactor::kSrcAlpha, xenos::BlendFactor::kOneMinusSrcAlpha,
      xenos::BlendOp::kAdd, xenos::BlendFactor::kDstAlpha,
      xenos::BlendFactor::kOneMinusDstAlpha, xenos::BlendOp::kRevSubtract);
  MetalColorAttachmentBlendState state =
      GetMetalColorAttachmentBlendState(nontrivial.value, 0);
  CHECK(state.write_mask == 0);
  CHECK_FALSE(state.blending_required);
}

TEST_CASE("Metal opaque MRT replacement does not enable blending",
          "[l0][metal][mrt][blend]") {
  MetalColorAttachmentBlendState opaque =
      GetMetalColorAttachmentBlendState(BlendControl().value, 0xFu);
  CHECK(opaque.write_mask == 0xFu);
  CHECK(opaque.rgb.is_opaque_replacement());
  CHECK(opaque.alpha.is_opaque_replacement());
  CHECK_FALSE(opaque.blending_required);

  reg::RB_BLENDCONTROL color_blend = BlendControl(
      xenos::BlendFactor::kSrcAlpha, xenos::BlendFactor::kOneMinusSrcAlpha);
  MetalColorAttachmentBlendState color_state =
      GetMetalColorAttachmentBlendState(color_blend.value, 0xFu);
  CHECK(color_state.blending_required);

  reg::RB_BLENDCONTROL alpha_blend = BlendControl(
      xenos::BlendFactor::kOne, xenos::BlendFactor::kZero, xenos::BlendOp::kAdd,
      xenos::BlendFactor::kOne, xenos::BlendFactor::kOne, xenos::BlendOp::kAdd);
  MetalColorAttachmentBlendState alpha_state =
      GetMetalColorAttachmentBlendState(alpha_blend.value, 0xFu);
  CHECK(alpha_state.blending_required);
}

TEST_CASE("Metal MRT blend equations remain independent per slot",
          "[l0][metal][mrt][blend]") {
  const std::array<reg::RB_BLENDCONTROL, xenos::kMaxColorRenderTargets>
      controls = {
          BlendControl(),
          BlendControl(xenos::BlendFactor::kSrcAlpha,
                       xenos::BlendFactor::kOneMinusSrcAlpha),
          BlendControl(xenos::BlendFactor::kDstColor,
                       xenos::BlendFactor::kOneMinusDstColor,
                       xenos::BlendOp::kSubtract),
          BlendControl(
              xenos::BlendFactor::kConstantColor,
              xenos::BlendFactor::kOneMinusConstantColor,
              xenos::BlendOp::kRevSubtract, xenos::BlendFactor::kConstantColor,
              xenos::BlendFactor::kOneMinusConstantColor, xenos::BlendOp::kMax),
      };
  constexpr uint32_t kMixedMask = 0xA05Fu;
  std::array<MetalColorAttachmentBlendState, xenos::kMaxColorRenderTargets>
      states;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    states[i] = GetMetalColorAttachmentBlendState(controls[i].value,
                                                  kMixedMask >> (i * 4u));
  }

  CHECK_FALSE(states[0].blending_required);
  CHECK(states[1].blending_required);
  CHECK(states[1].rgb.source_factor == MetalBlendFactor::kSourceAlpha);
  CHECK(states[1].rgb.destination_factor ==
        MetalBlendFactor::kOneMinusSourceAlpha);
  CHECK(states[2].write_mask == 0);
  CHECK_FALSE(states[2].blending_required);
  CHECK(states[3].blending_required);
  CHECK(states[3].rgb.source_factor == MetalBlendFactor::kBlendColor);
  CHECK(states[3].rgb.operation == MetalBlendOperation::kReverseSubtract);
  CHECK(states[3].alpha.source_factor == MetalBlendFactor::kBlendAlpha);
  CHECK(states[3].alpha.operation == MetalBlendOperation::kMax);
}

TEST_CASE("Metal alpha blend equations normalize color factors",
          "[l0][metal][mrt][blend]") {
  struct AlphaFactorCase {
    xenos::BlendFactor guest;
    MetalBlendFactor normalized;
  };
  constexpr AlphaFactorCase kCases[] = {
      {xenos::BlendFactor::kSrcColor, MetalBlendFactor::kSourceAlpha},
      {xenos::BlendFactor::kOneMinusSrcColor,
       MetalBlendFactor::kOneMinusSourceAlpha},
      {xenos::BlendFactor::kDstColor, MetalBlendFactor::kDestinationAlpha},
      {xenos::BlendFactor::kOneMinusDstColor,
       MetalBlendFactor::kOneMinusDestinationAlpha},
      {xenos::BlendFactor::kConstantColor, MetalBlendFactor::kBlendAlpha},
      {xenos::BlendFactor::kOneMinusConstantColor,
       MetalBlendFactor::kOneMinusBlendAlpha},
  };
  for (const AlphaFactorCase& factor_case : kCases) {
    reg::RB_BLENDCONTROL control = BlendControl(
        xenos::BlendFactor::kOne, xenos::BlendFactor::kZero,
        xenos::BlendOp::kAdd, factor_case.guest, factor_case.guest);
    MetalColorAttachmentBlendState state =
        GetMetalColorAttachmentBlendState(control.value, 0xFu);
    CHECK(state.alpha.source_factor == factor_case.normalized);
    CHECK(state.alpha.destination_factor == factor_case.normalized);
  }
}

TEST_CASE("Metal unknown blend state fails closed deterministically",
          "[l0][metal][mrt][blend]") {
  reg::RB_BLENDCONTROL control = BlendControl();
  control.color_srcblend = xenos::BlendFactor(31);
  control.color_destblend = xenos::BlendFactor(30);
  control.color_comb_fcn = xenos::BlendOp(7);
  control.alpha_srcblend = xenos::BlendFactor(29);
  control.alpha_destblend = xenos::BlendFactor(28);
  control.alpha_comb_fcn = xenos::BlendOp(6);
  MetalColorAttachmentBlendState state =
      GetMetalColorAttachmentBlendState(control.value, 0x1Fu);
  CHECK(state.write_mask == 0xFu);
  CHECK(state.rgb.source_factor == MetalBlendFactor::kZero);
  CHECK(state.rgb.destination_factor == MetalBlendFactor::kZero);
  CHECK(state.rgb.operation == MetalBlendOperation::kAdd);
  CHECK(state.alpha.source_factor == MetalBlendFactor::kZero);
  CHECK(state.alpha.destination_factor == MetalBlendFactor::kZero);
  CHECK(state.alpha.operation == MetalBlendOperation::kAdd);
  CHECK(state.blending_required);
}

TEST_CASE("Metal disabled stencil state has deterministic safe references",
          "[l0][metal][blend-stencil]") {
  reg::RB_DEPTHCONTROL depth = {};
  reg::RB_STENCILREFMASK stencil = {};
  stencil.stencilref = 0x55;
  reg::RB_STENCILREFMASK stencil_back = {};
  stencil_back.stencilref = 0xAA;
  MetalBlendStencilState state = GetMetalBlendStencilState(
      0.0f, 0.0f, 0.0f, 0.0f, depth, true, stencil, stencil_back);
  CHECK_FALSE(state.stencil_enabled);
  CHECK(state.stencil_reference_front == 0);
  CHECK(state.stencil_reference_back == 0);
}

TEST_CASE("Metal raster sample state matches pipeline and shader ABI",
          "[l0][metal][pipeline-raster]") {
  reg::RB_COLORCONTROL color = {};

  MetalPipelineRasterState one = GetMetalPipelineRasterState(1u, color);
  CHECK(one.raster_sample_count == 1u);
  CHECK(one.pipeline_key().raster_sample_count == 1u);
  CHECK(one.sample_count_log2_x == 0u);
  CHECK(one.sample_count_log2_y == 0u);

  MetalPipelineRasterState two = GetMetalPipelineRasterState(2u, color);
  CHECK(two.pipeline_key().raster_sample_count == 2u);
  CHECK(two.sample_count_log2_x == 0u);
  CHECK(two.sample_count_log2_y == 1u);

  MetalPipelineRasterState four = GetMetalPipelineRasterState(4u, color);
  CHECK(four.pipeline_key().raster_sample_count == 4u);
  CHECK(four.sample_count_log2_x == 1u);
  CHECK(four.sample_count_log2_y == 1u);

  MetalPipelineRasterState malformed = GetMetalPipelineRasterState(8u, color);
  CHECK(malformed.pipeline_key().raster_sample_count == 1u);
  CHECK(malformed.sample_count_log2_x == 0u);
  CHECK(malformed.sample_count_log2_y == 0u);
}

TEST_CASE("Metal alpha-to-mask is dynamic and does not fragment pipelines",
          "[l0][metal][pipeline-raster]") {
  reg::RB_COLORCONTROL color = {};
  color.alpha_to_mask_offset0 = 3u;
  color.alpha_to_mask_offset1 = 1u;
  color.alpha_to_mask_offset2 = 0u;
  color.alpha_to_mask_offset3 = 2u;

  MetalPipelineRasterState disabled = GetMetalPipelineRasterState(4u, color);
  CHECK(disabled.alpha_to_mask == 0u);

  color.alpha_to_mask_enable = 1u;
  MetalPipelineRasterState enabled = GetMetalPipelineRasterState(4u, color);
  CHECK(enabled.alpha_to_mask == 0x187u);
  CHECK(enabled.pipeline_key() == disabled.pipeline_key());
  CHECK(enabled.sample_count_log2_x == disabled.sample_count_log2_x);
  CHECK(enabled.sample_count_log2_y == disabled.sample_count_log2_y);

  // A later disabled draw must clear the packed offsets rather than inherit
  // the preceding shader constant.
  color.alpha_to_mask_enable = 0u;
  CHECK(GetMetalPipelineRasterState(4u, color).alpha_to_mask == 0u);
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
