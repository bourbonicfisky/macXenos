/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_TEXTURE_FORMAT_DECODE_H_
#define XENIA_GPU_METAL_METAL_TEXTURE_FORMAT_DECODE_H_

#include <cstdint>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// CPU texel transforms for guest formats the legacy Metal texture cache serves
// through a decoded host format, written as the exact expressions of the
// shared load shaders so the CPU path and the D3D12/Vulkan GPU path agree:
//   texture_load_depth_unorm.cs.xesl: UNorm24To32(word >> 8)
//   texture_load_depth_float.cs.xesl: Float20e4To32(word >> 8)
// The word is the guest 32-bit texel already brought to host byte order.

inline float DecodeDepthUnormTexel(uint32_t word) {
  return xenos::UNorm24To32(word >> 8u);
}

inline float DecodeDepthFloatTexel(uint32_t word) {
  return xenos::Float20e4To32(word >> 8u);
}

// Composes the host-format channel swizzle (the shared tables' RRRR / RGGG /
// RGBA mapping of Xenos components onto host channels) with the guest fetch
// swizzle. Both are 12-bit Xenos swizzles (3 bits per output component:
// 0-3 = X/Y/Z/W, 4 = 0, 5 = 1). A guest component that selects a channel is
// redirected through the host swizzle; constant selections pass through.
inline uint32_t ComposeGuestHostSwizzle(uint32_t guest_swizzle,
                                        uint32_t host_swizzle) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t g = (guest_swizzle >> (3 * i)) & 0b111;
    if (g < 4) {
      g = (host_swizzle >> (3 * g)) & 0b111;
    }
    out |= g << (3 * i);
  }
  return out;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_FORMAT_DECODE_H_
