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
