/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_render_target_cache.h"

#import <Metal/Metal.h>

#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {

const uint64_t MetalRenderTargetCache::kColorFormat = MTLPixelFormatBGRA8Unorm;

MetalRenderTargetCache::MetalRenderTargetCache(void* device)
    : device_(device) {}

MetalRenderTargetCache::~MetalRenderTargetCache() {
  if (color_target_) {
    id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)color_target_;
    (void)released;
    color_target_ = nullptr;
  }
}

void* MetalRenderTargetCache::EnsureColorTarget(uint32_t width,
                                                uint32_t height) {
  if (color_target_ && width_ == width && height_ == height) {
    return color_target_;
  }
  if (color_target_) {
    id<MTLTexture> old = (__bridge_transfer id<MTLTexture>)color_target_;
    (void)old;
    color_target_ = nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:(MTLPixelFormat)kColorFormat
                                   width:width
                                  height:height
                               mipmapped:NO];
  // Rendered into by IssueDraw and copied out to the presenter mailbox by
  // IssueSwap.
  desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalRenderTargetCache: failed to create {}x{} color target", width,
           height);
    return nullptr;
  }
  color_target_ = (__bridge_retained void*)texture;
  width_ = width;
  height_ = height;
  return color_target_;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
