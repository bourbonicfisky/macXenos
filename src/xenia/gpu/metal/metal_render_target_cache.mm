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
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

const uint64_t MetalRenderTargetCache::kColorFormat = MTLPixelFormatBGRA8Unorm;
const uint64_t MetalRenderTargetCache::kDepthFormat = MTLPixelFormatDepth32Float;

namespace {
// Guest color format -> host MTLPixelFormat. BGRA8 for the 8888 formats
// (matches the present blit; GAMMA is treated as linear for now — a known
// approximation, revisit with sRGB handling); real float/packed formats get
// float/packed targets so render data survives (k_32_FLOAT into BGRA8 was
// destroying JC2's lighting inputs).
MTLPixelFormat ColorGuestFormatToMetal(uint32_t guest_format) {
  switch (static_cast<xenos::ColorRenderTargetFormat>(guest_format)) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      return MTLPixelFormatBGRA8Unorm;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_AS_10_10_10_10:
      return MTLPixelFormatRGB10A2Unorm;
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT:
    case xenos::ColorRenderTargetFormat::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      return MTLPixelFormatRGBA16Float;
    case xenos::ColorRenderTargetFormat::k_16_16:
      return MTLPixelFormatRG16Snorm;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16:
      return MTLPixelFormatRGBA16Snorm;
    case xenos::ColorRenderTargetFormat::k_16_16_FLOAT:
      return MTLPixelFormatRG16Float;
    case xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT:
      return MTLPixelFormatRGBA16Float;
    case xenos::ColorRenderTargetFormat::k_32_FLOAT:
      return MTLPixelFormatR32Float;
    case xenos::ColorRenderTargetFormat::k_32_32_FLOAT:
      return MTLPixelFormatRG32Float;
    default:
      return MTLPixelFormatBGRA8Unorm;
  }
}
}  // namespace

uint32_t MetalRenderTargetCache::ColorTargetKey(uint32_t edram_base,
                                                uint32_t guest_format) {
  return (edram_base << 12) | uint32_t(ColorGuestFormatToMetal(guest_format));
}

MetalRenderTargetCache::MetalRenderTargetCache(void* device)
    : device_(device) {}

MetalRenderTargetCache::~MetalRenderTargetCache() {
  for (auto& pair : color_targets_) {
    if (pair.second.texture) {
      id<MTLTexture> released =
          (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  color_targets_.clear();
  for (auto& pair : depth_targets_) {
    if (pair.second.texture) {
      id<MTLTexture> released =
          (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  depth_targets_.clear();
  for (auto& pair : resolve_snapshots_) {
    if (pair.second.texture) {
      id<MTLTexture> released =
          (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  resolve_snapshots_.clear();
}

void* MetalRenderTargetCache::EnsureResolveSnapshot(uint32_t dest_base,
                                                    uint32_t width,
                                                    uint32_t height,
                                                    uint64_t pixel_format) {
  // Scene-scale depth resolves arrive as vertical strips (1280x384 + 1280x336)
  // of ONE 720-row buffer that shaders sample full-height from the first
  // strip's base; allocate the first strip full-height so the later strip can
  // land inside it (FindContainingSnapshot).
  if (pixel_format == uint64_t(MTLPixelFormatR32Float) && width >= 1280 &&
      height < 720) {
    height = 720;
  }
  auto it = resolve_snapshots_.find(dest_base);
  if (it != resolve_snapshots_.end() && it->second.texture &&
      it->second.width >= width && it->second.height >= height &&
      it->second.pixel_format == pixel_format) {
    return it->second.texture;
  }
  if (it != resolve_snapshots_.end() && it->second.texture) {
    id<MTLTexture> old = (__bridge_transfer id<MTLTexture>)it->second.texture;
    (void)old;
    it->second.texture = nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  // 2D array so the texture cache (ForceTextureArray reads) can sample it.
  // ShaderRead|RenderTarget: a color snapshot is a blit destination + sampled;
  // the depth-conversion snapshot is a render target of the conversion pass +
  // sampled.
  desc.textureType = MTLTextureType2DArray;
  desc.pixelFormat = (MTLPixelFormat)pixel_format;
  desc.width = width;
  desc.height = height;
  desc.arrayLength = 1;
  desc.mipmapLevelCount = 1;
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalRenderTargetCache: failed to create {}x{} resolve snapshot",
           width, height);
    return nullptr;
  }
  resolve_snapshots_[dest_base] =
      Snapshot{(__bridge_retained void*)texture, width, height, pixel_format};
  return resolve_snapshots_[dest_base].texture;
}

void* MetalRenderTargetCache::GetResolveSnapshot(uint32_t dest_base) const {
  auto it = resolve_snapshots_.find(dest_base);
  return it != resolve_snapshots_.end() ? it->second.texture : nullptr;
}

void* MetalRenderTargetCache::FindContainingSnapshot(
    uint32_t dest_base, uint32_t* y_offset_out) const {
  for (const auto& pair : resolve_snapshots_) {
    const Snapshot& snap = pair.second;
    if (!snap.texture || dest_base <= pair.first) {
      continue;
    }
    // Tightly-packed guest extent (pitch == width * 4bpp — true for the 32bpp
    // scene-depth strips this exists for).
    uint32_t row_bytes = snap.width * 4;
    uint32_t extent = row_bytes * snap.height;
    uint32_t offset = dest_base - pair.first;
    if (offset < extent && (offset % row_bytes) == 0) {
      if (y_offset_out) {
        *y_offset_out = offset / row_bytes;
      }
      return snap.texture;
    }
  }
  return nullptr;
}

void* MetalRenderTargetCache::GetDepthTarget(uint32_t edram_base) {
  auto it = depth_targets_.find(edram_base);
  if (it != depth_targets_.end()) {
    return it->second.texture;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:(MTLPixelFormat)kDepthFormat
                                   width:kDepthWidth
                                  height:kDepthHeight
                               mipmapped:NO];
  // ShaderRead so the depth-resolve conversion pass can sample it (depth2d) to
  // produce the R32Float snapshot deferred lighting reads back.
  desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalRenderTargetCache: failed to create depth target base={}",
           edram_base);
    return nullptr;
  }
  Target target;
  target.texture = (__bridge_retained void*)texture;
  target.pixel_format = kDepthFormat;
  depth_targets_.emplace(edram_base, target);
  XELOGI("MetalRenderTargetCache: new depth target base={}", edram_base);
  return target.texture;
}

void* MetalRenderTargetCache::FindDepthTarget(uint32_t edram_base) const {
  auto it = depth_targets_.find(edram_base);
  return it != depth_targets_.end() ? it->second.texture : nullptr;
}

void* MetalRenderTargetCache::GetColorTarget(uint32_t edram_base,
                                             uint32_t guest_format,
                                             uint64_t* pixel_format_out) {
  // Key by (EDRAM base, HOST pixel format), mirroring the mainline resource-
  // format keying: k_8_8_8_8 and k_8_8_8_8_GAMMA are the same EDRAM bits and
  // must share one texture (draws bind GAMMA, resolves often declare non-GAMMA
  // of the same target), while genuinely different storage (k_32_FLOAT) stays
  // distinct.
  MTLPixelFormat host_format = ColorGuestFormatToMetal(guest_format);
  uint32_t key = ColorTargetKey(edram_base, guest_format);
  auto it = color_targets_.find(key);
  if (it != color_targets_.end()) {
    if (pixel_format_out) {
      *pixel_format_out = it->second.pixel_format;
    }
    if (it->second.pixel_format == kColorFormat) {
      presentable_color_target_ = it->second.texture;
    }
    return it->second.texture;
  }
  MTLPixelFormat pixel_format = host_format;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:pixel_format
                                   width:kColorWidth
                                  height:kColorHeight
                               mipmapped:NO];
  // Rendered into by IssueDraw; blitted to resolve snapshots by IssueCopy and
  // (BGRA8 only) to the presenter mailbox by IssueSwap.
  desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalRenderTargetCache: failed to create color target base={} fmt={}",
           edram_base, guest_format);
    return nullptr;
  }
  Target target;
  target.texture = (__bridge_retained void*)texture;
  target.pixel_format = uint64_t(pixel_format);
  color_targets_.emplace(key, target);
  if (pixel_format_out) {
    *pixel_format_out = target.pixel_format;
  }
  if (target.pixel_format == kColorFormat) {
    presentable_color_target_ = target.texture;
  }
  XELOGI("MetalRenderTargetCache: new color target base={} fmt={} mtl={}",
         edram_base, guest_format, uint32_t(pixel_format));
  return target.texture;
}

void* MetalRenderTargetCache::FindColorTarget(uint32_t edram_base,
                                              uint32_t guest_format,
                                              uint64_t* pixel_format_out) const {
  uint32_t key = ColorTargetKey(edram_base, guest_format);
  auto it = color_targets_.find(key);
  if (it == color_targets_.end()) {
    return nullptr;
  }
  if (pixel_format_out) {
    *pixel_format_out = it->second.pixel_format;
  }
  return it->second.texture;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
