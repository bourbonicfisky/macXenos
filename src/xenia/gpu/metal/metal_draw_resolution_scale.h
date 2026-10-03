/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_DRAW_RESOLUTION_SCALE_H_
#define XENIA_GPU_METAL_METAL_DRAW_RESOLUTION_SCALE_H_

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

// Draw resolution scaling on the native Metal backend
// (implement-metal-draw-resolution-scaling). The scale is an integer per axis,
// read once per session from draw_resolution_scale_x/y and clamped to
// 1..kMetalMaxDrawResolutionScale (memory: the canonical EDRAM, render targets
// and resolve snapshots all grow by x * y).
constexpr uint32_t kMetalMaxDrawResolutionScale = 3;

struct MetalDrawResolutionScale {
  // The configured scale after clamping.
  uint32_t requested_x = 1;
  uint32_t requested_y = 1;
  // Whether either configured value was outside
  // 1..kMetalMaxDrawResolutionScale.
  bool clamped = false;
  // The scale the session renders at. Stays 1x1 until scaled rendering is
  // enabled, which it is only once render targets, EDRAM, resolves and texture
  // serving support it.
  uint32_t x = 1;
  uint32_t y = 1;
};

constexpr uint32_t ClampMetalDrawResolutionScaleAxis(int32_t value) {
  return value < 1 ? 1u
                   : value > int32_t(kMetalMaxDrawResolutionScale)
                         ? kMetalMaxDrawResolutionScale
                         : uint32_t(value);
}

constexpr MetalDrawResolutionScale ResolveMetalDrawResolutionScale(
    int32_t config_x, int32_t config_y, bool scaled_rendering_enabled) {
  MetalDrawResolutionScale scale;
  scale.requested_x = ClampMetalDrawResolutionScaleAxis(config_x);
  scale.requested_y = ClampMetalDrawResolutionScaleAxis(config_y);
  scale.clamped = int32_t(scale.requested_x) != config_x ||
                  int32_t(scale.requested_y) != config_y;
  if (scaled_rendering_enabled) {
    scale.x = scale.requested_x;
    scale.y = scale.requested_y;
  }
  return scale;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_DRAW_RESOLUTION_SCALE_H_
