/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_FIXED_FUNCTION_STATE_H_
#define XENIA_GPU_METAL_METAL_FIXED_FUNCTION_STATE_H_

#include <algorithm>
#include <cstdint>

#include "xenia/gpu/draw_util.h"

namespace xe {
namespace gpu {
namespace metal {

// Device-independent form of MTLScissorRect. Kept free of Metal types so the
// exact production attachment-clamp logic is covered by the L0 conformance
// binary without requiring a device or Objective-C++.
struct MetalScissorRect {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;

  bool empty() const { return width == 0 || height == 0; }
};

// Device-independent representation of Metal's dynamic face-raster state.
// kAll is a draw no-op because Metal has no front-and-back cull enum.
enum class MetalCullMode : uint32_t {
  kNone,
  kFront,
  kBack,
  kAll,
};

struct MetalCullFrontFaceState {
  MetalCullMode cull_mode;
  bool front_face_clockwise;

  bool culls_all() const { return cull_mode == MetalCullMode::kAll; }
};

// Device-independent representation of Metal's dynamic triangle fill and
// depth-bias state. Metal has no point polygon fill mode, so Xenos point fill
// is conservatively represented as lines, matching the shared D3D12 fallback.
enum class MetalTriangleFillMode : uint32_t {
  kFill,
  kLines,
};

struct MetalFillDepthBiasState {
  MetalTriangleFillMode triangle_fill_mode;
  float depth_bias_constant;
  float depth_bias_slope;
};

// Device-independent representation of Metal's per-draw blend color and
// stencil reference state. References are zeroed while stencil is disabled so
// the helper has a deterministic safe state; an applicable stencil draw always
// establishes its complete front/back values on the encoder.
struct MetalBlendStencilState {
  float blend_color_red;
  float blend_color_green;
  float blend_color_blue;
  float blend_color_alpha;
  bool stencil_enabled;
  uint32_t stencil_reference_front;
  uint32_t stencil_reference_back;
};

// Xenos culling and faceness apply only to polygonal primitives. Lines,
// points, and rectangle-list geometry must establish safe no-cull state on
// every draw so a preceding triangle draw can't leak encoder state into them.
inline MetalCullFrontFaceState GetMetalCullFrontFaceState(
    reg::PA_SU_SC_MODE_CNTL mode, bool primitive_polygonal) {
  if (!primitive_polygonal) {
    return {MetalCullMode::kNone, false};
  }
  MetalCullMode cull_mode = MetalCullMode::kNone;
  if (mode.cull_front && mode.cull_back) {
    cull_mode = MetalCullMode::kAll;
  } else if (mode.cull_front) {
    cull_mode = MetalCullMode::kFront;
  } else if (mode.cull_back) {
    cull_mode = MetalCullMode::kBack;
  }
  return {cull_mode, mode.face != 0};
}

// Xenos dual polygon mode may select fill independently for front and back
// faces. Metal has one fill mode for the draw, so use the most conservative
// non-culled face: either points or lines becomes line fill. Reserved/disabled
// polygon modes and non-polygon primitives explicitly restore solid fill.
inline MetalTriangleFillMode GetMetalTriangleFillMode(
    reg::PA_SU_SC_MODE_CNTL mode, bool primitive_polygonal) {
  if (!primitive_polygonal ||
      mode.poly_mode != xenos::PolygonModeEnable::kDualMode) {
    return MetalTriangleFillMode::kFill;
  }
  xenos::PolygonType polygon_type = xenos::PolygonType::kTriangles;
  if (!mode.cull_front) {
    polygon_type = std::min(polygon_type, mode.polymode_front_ptype);
  }
  if (!mode.cull_back) {
    polygon_type = std::min(polygon_type, mode.polymode_back_ptype);
  }
  return polygon_type == xenos::PolygonType::kTriangles
             ? MetalTriangleFillMode::kFill
             : MetalTriangleFillMode::kLines;
}

// Converts the shared preferred-face polygon offset to a complete Metal value.
// The caller obtains scale/offset through GetPreferredFacePolygonOffset so
// culling and the distinct non-polygon enable have one guest-semantics owner.
// The shared D3D10 conversion preserves useful guest depth separation for
// unorm24 and float24 host depth representations.
inline MetalFillDepthBiasState GetMetalFillDepthBiasState(
    reg::PA_SU_SC_MODE_CNTL mode, bool primitive_polygonal,
    xenos::DepthRenderTargetFormat depth_format, float polygon_offset_scale,
    float polygon_offset, uint32_t draw_resolution_scale = 1) {
  return {
      GetMetalTriangleFillMode(mode, primitive_polygonal),
      float(draw_util::GetD3D10IntegerPolygonOffset(
          depth_format, polygon_offset)),
      polygon_offset_scale * xenos::kPolygonOffsetScaleSubpixelUnit *
          float(std::max(draw_resolution_scale, uint32_t(1))),
  };
}

// Metal supports separate front/back stencil references. Xenos back-face
// stencil state applies only to polygonal primitives when enabled in the
// normalized depth control; lines, points, and other non-polygons use the
// front reference even if the raw back-face-enable bit was set.
inline MetalBlendStencilState GetMetalBlendStencilState(
    float blend_color_red, float blend_color_green, float blend_color_blue,
    float blend_color_alpha,
    reg::RB_DEPTHCONTROL normalized_depth_control, bool primitive_polygonal,
    reg::RB_STENCILREFMASK stencil_ref_mask,
    reg::RB_STENCILREFMASK stencil_ref_mask_back) {
  bool stencil_enabled = normalized_depth_control.stencil_enable != 0;
  uint32_t stencil_reference_front =
      stencil_enabled ? stencil_ref_mask.stencilref : 0;
  uint32_t stencil_reference_back = stencil_reference_front;
  if (stencil_enabled && primitive_polygonal &&
      normalized_depth_control.backface_enable) {
    stencil_reference_back = stencil_ref_mask_back.stencilref;
  }
  return {blend_color_red,
          blend_color_green,
          blend_color_blue,
          blend_color_alpha,
          stencil_enabled,
          stencil_reference_front,
          stencil_reference_back};
}

// Intersects the shared guest-normalized scissor with the active Metal render
// extent. Widened end arithmetic is required because a malformed or extreme
// guest rectangle must collapse/clip, never wrap and expose unrelated pixels.
inline MetalScissorRect ClampScissorToRenderExtent(
    const draw_util::Scissor& scissor, uint32_t render_width,
    uint32_t render_height) {
  uint64_t x_begin = std::min<uint64_t>(scissor.offset[0], render_width);
  uint64_t y_begin = std::min<uint64_t>(scissor.offset[1], render_height);
  uint64_t x_end =
      std::min<uint64_t>(uint64_t(scissor.offset[0]) + scissor.extent[0],
                         render_width);
  uint64_t y_end =
      std::min<uint64_t>(uint64_t(scissor.offset[1]) + scissor.extent[1],
                         render_height);
  x_end = std::max(x_end, x_begin);
  y_end = std::max(y_end, y_begin);
  return {uint32_t(x_begin), uint32_t(y_begin), uint32_t(x_end - x_begin),
          uint32_t(y_end - y_begin)};
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_FIXED_FUNCTION_STATE_H_
