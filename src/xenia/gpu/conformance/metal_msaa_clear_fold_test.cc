/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Device-free qualification of the MSAA fast-clear fold
// (metal_msaa_clear_fold.h): the runtime clear shape folds into the 1x view
// with the right scale and RB_SURFACE_INFO, and every disqualifying state
// refuses on its own with the matching reason.

#include <cstring>
#include <memory>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_msaa_clear_fold.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::MsaaClearFoldRefusal;

struct ClearState {
  std::unique_ptr<RegisterFile> regs = std::make_unique<RegisterFile>();
  uint64_t vs = metal::kMsaaClearFoldVertexShader;
  uint64_t ps = metal::kMsaaClearFoldPixelShader;
  xenos::PrimitiveType primitive = xenos::PrimitiveType::kRectangleList;
  uint32_t index_count = 3;
  reg::RB_DEPTHCONTROL depth{};
  uint32_t color_mask = 0;

  ClearState(xenos::MsaaSamples msaa, uint32_t pitch) {
    std::memset(regs->values, 0, sizeof(regs->values));
    reg::RB_SURFACE_INFO surface{};
    surface.surface_pitch = pitch;
    surface.msaa_samples = msaa;
    regs->values[XE_GPU_REG_RB_SURFACE_INFO] = surface.value;
    reg::RB_MODECONTROL mode{};
    mode.edram_mode = xenos::EdramMode::kColorDepth;
    regs->values[XE_GPU_REG_RB_MODECONTROL] = mode.value;
    // Viewport transform off (all PA_CL_VTE_CNTL enables zero).
    regs->values[XE_GPU_REG_PA_CL_VTE_CNTL] = 0;
    // Replace blending on every colour target.
    reg::RB_BLENDCONTROL blend{};
    blend.color_srcblend = xenos::BlendFactor::kOne;
    blend.color_destblend = xenos::BlendFactor::kZero;
    blend.color_comb_fcn = xenos::BlendOp::kAdd;
    blend.alpha_srcblend = xenos::BlendFactor::kOne;
    blend.alpha_destblend = xenos::BlendFactor::kZero;
    blend.alpha_comb_fcn = xenos::BlendOp::kAdd;
    for (uint32_t i = 0; i < 4; ++i) {
      regs->values[reg::RB_BLENDCONTROL::rt_register_indices[i]] = blend.value;
    }
    // Depth ALWAYS + write + stencil ALWAYS/REPLACE (the Gears 3 depth clear).
    depth.z_enable = 1;
    depth.z_write_enable = 1;
    depth.zfunc = xenos::CompareFunction::kAlways;
    depth.stencil_enable = 1;
    depth.stencilfunc = xenos::CompareFunction::kAlways;
    depth.stencilfail = xenos::StencilOp::kKeep;
    depth.stencilzpass = xenos::StencilOp::kReplace;
    depth.stencilzfail = xenos::StencilOp::kKeep;
  }

  metal::MsaaClearFoldDecision Qualify() const {
    return metal::QualifyMsaaClearFold(*regs, vs, ps, primitive, index_count,
                                       depth, color_mask);
  }
};

}  // namespace

TEST_CASE("MSAA clear fold: the runtime 4x depth clear folds into 1x",
          "[metal][msaa_clear_fold]") {
  ClearState clear(xenos::MsaaSamples::k4X, 640);
  auto decision = clear.Qualify();
  REQUIRE(decision.fold());
  CHECK(decision.scale_x_log2 == 1);
  CHECK(decision.scale_y_log2 == 1);
  reg::RB_SURFACE_INFO surface_1x;
  surface_1x.value = decision.surface_info_1x;
  CHECK(surface_1x.msaa_samples == xenos::MsaaSamples::k1X);
  // Same tile pitch: 640 4x pixels = 1280 samples = 1280 1x pixels.
  CHECK(surface_1x.surface_pitch == 1280);
}

TEST_CASE("MSAA clear fold: 2x doubles rows only",
          "[metal][msaa_clear_fold]") {
  ClearState clear(xenos::MsaaSamples::k2X, 1280);
  auto decision = clear.Qualify();
  REQUIRE(decision.fold());
  CHECK(decision.scale_x_log2 == 0);
  CHECK(decision.scale_y_log2 == 1);
  reg::RB_SURFACE_INFO surface_1x;
  surface_1x.value = decision.surface_info_1x;
  CHECK(surface_1x.surface_pitch == 1280);
}

TEST_CASE("MSAA clear fold: stencil-only and colour clears fold",
          "[metal][msaa_clear_fold]") {
  ClearState stencil_only(xenos::MsaaSamples::k4X, 640);
  stencil_only.depth.z_enable = 0;
  stencil_only.depth.z_write_enable = 0;
  CHECK(stencil_only.Qualify().fold());
  ClearState colour(xenos::MsaaSamples::k4X, 640);
  colour.color_mask = 0xF;
  CHECK(colour.Qualify().fold());
}

TEST_CASE("MSAA clear fold: each disqualifier refuses with its reason",
          "[metal][msaa_clear_fold]") {
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.vs = 0x1234;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kShaderPair);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.primitive = xenos::PrimitiveType::kTriangleList;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kPrimitive);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.index_count = 6;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kPrimitive);
  }
  {
    ClearState c(xenos::MsaaSamples::k1X, 1280);
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kSingleSample);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    reg::PA_CL_VTE_CNTL vte{};
    vte.vport_x_scale_ena = 1;
    c.regs->values[XE_GPU_REG_PA_CL_VTE_CNTL] = vte.value;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kViewportTransform);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    reg::RB_MODECONTROL mode{};
    mode.edram_mode = xenos::EdramMode::kCopy;
    c.regs->values[XE_GPU_REG_RB_MODECONTROL] = mode.value;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kEdramMode);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.depth.zfunc = xenos::CompareFunction::kLessEqual;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kDepthFunction);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.depth.stencilfunc = xenos::CompareFunction::kEqual;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kStencil);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.depth.stencilzpass = xenos::StencilOp::kIncrementClamp;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kStencil);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.depth.backface_enable = 1;
    c.depth.stencilfunc_bf = xenos::CompareFunction::kNotEqual;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kStencil);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.color_mask = 0xF;
    reg::RB_BLENDCONTROL blend{};
    blend.color_srcblend = xenos::BlendFactor::kOne;
    blend.color_destblend = xenos::BlendFactor::kOne;
    c.regs->values[reg::RB_BLENDCONTROL::rt_register_indices[0]] = blend.value;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kBlending);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 640);
    reg::RB_COLORCONTROL color_control{};
    color_control.alpha_test_enable = 1;
    color_control.alpha_func = xenos::CompareFunction::kGreater;
    c.regs->values[XE_GPU_REG_RB_COLORCONTROL] = color_control.value;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kAlphaTest);
  }
  {
    ClearState c(xenos::MsaaSamples::k4X, 0);
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kPitch);
  }
  {
    // Gamma colour: the canonical round trip quantizes the host value.
    ClearState c(xenos::MsaaSamples::k4X, 640);
    c.color_mask = 0xF;
    reg::RB_COLOR_INFO color_info{};
    color_info.color_format = xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA;
    c.regs->values[reg::RB_COLOR_INFO::rt_register_indices[0]] =
        color_info.value;
    CHECK(c.Qualify().refusal == MsaaClearFoldRefusal::kColourFormat);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
