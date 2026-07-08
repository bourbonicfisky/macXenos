/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_
#define XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

// Minimal render-target cache for the Metal backend bring-up. For now it owns a
// single color MTLTexture that IssueDraw renders into and IssueSwap presents,
// bypassing the guest frontbuffer / EDRAM-copy path (that arrives once real
// draws land). Metal objects are held as opaque void* (retained) so the header
// stays plain C++.
class MetalRenderTargetCache {
 public:
  // MTLPixelFormatBGRA8Unorm — matches the drawable and the present blit so a
  // straight copy into the presenter's mailbox works.
  static const uint64_t kColorFormat;

  explicit MetalRenderTargetCache(void* device);
  ~MetalRenderTargetCache();

  // Ensures the color target is exactly width x height, (re)creating on change.
  // Returns the id<MTLTexture> as void*, or nullptr on failure.
  void* EnsureColorTarget(uint32_t width, uint32_t height);

  void* color_target() const { return color_target_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

 private:
  void* device_;                  // id<MTLDevice>, not owned.
  void* color_target_ = nullptr;  // id<MTLTexture>, retained.
  uint32_t width_ = 0;
  uint32_t height_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_
