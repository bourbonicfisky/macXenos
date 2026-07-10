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

#include <cstring>
#include <memory>

#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/gpu/metal/metal_command_processor.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

const uint64_t MetalRenderTargetCache::kColorFormat = MTLPixelFormatBGRA8Unorm;
const uint64_t MetalRenderTargetCache::kDepthFormat =
    MTLPixelFormatDepth32Float;

namespace {
// Guest color RESOURCE format (gamma already collapsed by the base class since
// IsGammaFormatHostStorageSeparate()==false) -> host MTLPixelFormat.
MTLPixelFormat ColorResourceFormatToMetal(
    xenos::ColorRenderTargetFormat format) {
  switch (format) {
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

MetalRenderTargetCache::MetalRenderTarget::~MetalRenderTarget() {
  if (texture_) {
    id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)texture_;
    (void)released;
    texture_ = nullptr;
  }
}

MetalRenderTargetCache::MetalRenderTargetCache(
    const RegisterFile& register_file, const Memory& memory,
    TraceWriter* trace_writer, MetalCommandProcessor& command_processor,
    void* device)
    : RenderTargetCache(register_file, memory, trace_writer, 1, 1),
      command_processor_(command_processor),
      device_(device) {}

MetalRenderTargetCache::~MetalRenderTargetCache() { Shutdown(); }

bool MetalRenderTargetCache::Initialize() {
  InitializeCommon();
  return true;
}

void MetalRenderTargetCache::Shutdown() {
  for (auto& pair : resolve_snapshots_) {
    if (pair.second.texture) {
      id<MTLTexture> released =
          (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  resolve_snapshots_.clear();
  presentable_color_target_ = nullptr;
  // Destroys all MetalRenderTargets (releasing their textures), then the
  // common ownership state.
  DestroyAllRenderTargets(true);
  ShutdownCommon();
}

RenderTargetCache::RenderTarget* MetalRenderTargetCache::CreateRenderTarget(
    RenderTargetKey key) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLPixelFormat pixel_format =
      key.is_depth ? (MTLPixelFormat)kDepthFormat
                   : ColorResourceFormatToMetal(key.GetColorFormat());
  // Bring-up: all textures are single-sample regardless of the key's MSAA
  // (JC2 scene passes are 4x). Size at the 1x pixel dimensions so a 4x key
  // isn't created half-width, and 4x<->1x transfers become plain copies.
  uint32_t width = RenderTargetKey::GetWidth(key.pitch_tiles_at_32bpp,
                                             xenos::MsaaSamples::k1X);
  uint32_t height =
      GetRenderTargetHeight(key.pitch_tiles_at_32bpp, xenos::MsaaSamples::k1X);
  if (!width || !height) {
    return nullptr;
  }
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:pixel_format
                                   width:width
                                  height:height
                               mipmapped:NO];
  // Sampled by the depth-convert / snapshot-blit paths and by transfers.
  desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalRenderTargetCache: failed to create RT {}",
           key.GetDebugName());
    return nullptr;
  }
  XELOGI("MetalRenderTargetCache: new RT {} ({}x{} mtl={})", key.GetDebugName(),
         width, height, uint32_t(pixel_format));
  return new MetalRenderTarget(key, (__bridge_retained void*)texture,
                               uint64_t(pixel_format));
}

bool MetalRenderTargetCache::Update(bool is_rasterization_done,
                                    reg::RB_DEPTHCONTROL normalized_depth_control,
                                    uint32_t normalized_color_mask,
                                    const Shader& vertex_shader) {
  if (!RenderTargetCache::Update(is_rasterization_done,
                                 normalized_depth_control,
                                 normalized_color_mask, vertex_shader)) {
    return false;
  }
  RenderTarget* const* accumulated = last_update_accumulated_render_targets();
  PerformTransfers(accumulated, last_update_transfers());
  // Track the presentable (BGRA8) color target for the pre-resolve fallback.
  for (uint32_t i = 1; i <= xenos::kMaxColorRenderTargets; ++i) {
    auto* rt = static_cast<MetalRenderTarget*>(accumulated[i]);
    if (rt && rt->pixel_format() == kColorFormat) {
      presentable_color_target_ = rt->texture();
      break;
    }
  }
  return true;
}

void* MetalRenderTargetCache::current_depth_texture() const {
  auto* rt = static_cast<MetalRenderTarget*>(
      last_update_accumulated_render_targets()[0]);
  return rt ? rt->texture() : nullptr;
}

void* MetalRenderTargetCache::current_color_texture(
    uint32_t index, uint64_t* pixel_format_out) const {
  auto* rt = static_cast<MetalRenderTarget*>(
      last_update_accumulated_render_targets()[1 + index]);
  if (!rt) {
    return nullptr;
  }
  if (pixel_format_out) {
    *pixel_format_out = rt->pixel_format();
  }
  return rt->texture();
}

uint64_t MetalRenderTargetCache::current_attachment_key() const {
  RenderTarget* const* accumulated = last_update_accumulated_render_targets();
  uint64_t key = 0;
  for (uint32_t i = 0; i <= xenos::kMaxColorRenderTargets; ++i) {
    key = key * 1000003u ^
          (accumulated[i] ? uint64_t(accumulated[i]->key().key) + 1 : 0);
  }
  return key;
}

void MetalRenderTargetCache::PerformTransfers(
    RenderTarget* const* render_targets,
    const std::vector<Transfer>* transfer_lists) {
  bool any = false;
  for (uint32_t i = 0; i <= xenos::kMaxColorRenderTargets && !any; ++i) {
    any = render_targets[i] && !transfer_lists[i].empty();
  }
  if (!any) {
    return;
  }
  // Transfers are encoded as standalone blits at a pass boundary before the
  // draw's encoder opens.
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)
      command_processor_.PrepareTransferCommandBuffer();
  if (!cb) {
    return;
  }
  id<MTLBlitCommandEncoder> blit = nil;
  for (uint32_t i = 0; i <= xenos::kMaxColorRenderTargets; ++i) {
    auto* dest = static_cast<MetalRenderTarget*>(render_targets[i]);
    if (!dest) {
      continue;
    }
    RenderTargetKey dest_key = dest->key();
    for (const Transfer& transfer : transfer_lists[i]) {
      auto* source = static_cast<MetalRenderTarget*>(transfer.source);
      if (!source) {
        continue;
      }
      RenderTargetKey src_key = source->key();
      // Bring-up scope: same host pixel format and same pitch — the tile span
      // maps to identical rectangles in both textures (all textures are
      // single-sample and 1x-sized, so MSAA-differing keys copy plainly).
      // Reinterpreting transfers (format/bpp/color<->depth changes) are logged
      // and skipped until JC2 demands them.
      if (source->pixel_format() != dest->pixel_format() ||
          src_key.GetPitchTiles() != dest_key.GetPitchTiles()) {
        static uint32_t s_skips = 0;
        if (s_skips++ < 24) {
          XELOGW("MetalRenderTargetCache: SKIP transfer {} -> {} tiles [{},{})",
                 src_key.GetDebugName(), dest_key.GetDebugName(),
                 transfer.start_tiles, transfer.end_tiles);
        }
        continue;
      }
      // Rectangles at 1x sample density (our textures are all single-sample).
      Transfer::Rectangle dest_rects[Transfer::kMaxRectanglesWithCutout];
      uint32_t dest_rect_count = transfer.GetRectangles(
          dest_key.base_tiles, dest_key.GetPitchTiles(),
          xenos::MsaaSamples::k1X, dest_key.Is64bpp(), dest_rects);
      Transfer::Rectangle src_rects[Transfer::kMaxRectanglesWithCutout];
      uint32_t src_rect_count = Transfer::GetRangeRectangles(
          transfer.start_tiles, transfer.end_tiles, src_key.base_tiles,
          src_key.GetPitchTiles(), xenos::MsaaSamples::k1X, src_key.Is64bpp(),
          src_rects);
      if (dest_rect_count != src_rect_count) {
        continue;  // Shouldn't happen with equal pitch; be safe.
      }
      if (!blit) {
        blit = [cb blitCommandEncoder];
      }
      id<MTLTexture> src_tex = (__bridge id<MTLTexture>)source->texture();
      id<MTLTexture> dst_tex = (__bridge id<MTLTexture>)dest->texture();
      for (uint32_t r = 0; r < dest_rect_count; ++r) {
        const auto& sr = src_rects[r];
        const auto& dr = dest_rects[r];
        uint32_t w = std::min(sr.width_pixels, dr.width_pixels);
        uint32_t h = std::min(sr.height_pixels, dr.height_pixels);
        // Clamp to actual texture extents (heights may differ).
        w = std::min<uint32_t>(
            w, std::min(src_tex.width - sr.x_pixels, dst_tex.width - dr.x_pixels));
        if (sr.y_pixels >= src_tex.height || dr.y_pixels >= dst_tex.height) {
          continue;
        }
        h = std::min<uint32_t>(
            h, std::min(src_tex.height - sr.y_pixels,
                        dst_tex.height - dr.y_pixels));
        if (!w || !h) {
          continue;
        }
        [blit copyFromTexture:src_tex
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(sr.x_pixels, sr.y_pixels, 0)
                   sourceSize:MTLSizeMake(w, h, 1)
                    toTexture:dst_tex
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(dr.x_pixels, dr.y_pixels, 0)];
      }
    }
  }
  if (blit) {
    [blit endEncoding];
  }
}

void* MetalRenderTargetCache::GetResolveSourceTexture(
    uint32_t edram_base, uint32_t row_length_used, uint32_t rows,
    uint32_t pitch_tiles, bool is_depth, uint64_t* pixel_format_out) {
  resolve_dump_rectangles_.clear();
  GetResolveCopyRectanglesToDump(edram_base, row_length_used, rows, pitch_tiles,
                                 resolve_dump_rectangles_);
  for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
    auto* rt = static_cast<MetalRenderTarget*>(rect.render_target);
    if (!rt || rt->key().is_depth != uint32_t(is_depth)) {
      continue;
    }
    if (pixel_format_out) {
      *pixel_format_out = rt->pixel_format();
    }
    return rt->texture();
  }
  return nullptr;
}

void* MetalRenderTargetCache::EnsureResolveSnapshot(uint32_t dest_base,
                                                    uint32_t width,
                                                    uint32_t height,
                                                    uint64_t pixel_format) {
  // Scene-scale depth resolves arrive as vertical strips of ONE 720-row buffer
  // sampled full-height from the first strip's base; allocate full-height so
  // the later strip lands inside (FindContainingSnapshot).
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

void MetalRenderTargetCache::RestoreEdramSnapshot(const void* snapshot) {
  // Full-EDRAM 1280-pitch k_8_8_8_8 render target owning all tiles.
  RenderTarget* full_rt = PrepareFullEdram1280xRenderTargetForSnapshotRestoration(
      xenos::ColorRenderTargetFormat::k_8_8_8_8);
  if (!full_rt) {
    XELOGW("MetalRenderTargetCache: EDRAM snapshot restore RT unavailable");
    return;
  }
  auto* rt = static_cast<MetalRenderTarget*>(full_rt);
  id<MTLTexture> texture = (__bridge id<MTLTexture>)rt->texture();
  // Reorder the tile-major snapshot (2048 tiles of 80x16 32bpp samples) into
  // the 1280-wide (16 tiles per row) row-major layout.
  uint32_t width = uint32_t(texture.width);    // 1280
  uint32_t height = uint32_t(texture.height);  // covers 2048 tiles
  const uint32_t tile_w = xenos::kEdramTileWidthSamples;    // 80
  const uint32_t tile_h = xenos::kEdramTileHeightSamples;   // 16
  uint32_t tiles_per_row = width / tile_w;                  // 16
  uint32_t row_bytes = width * 4;
  std::unique_ptr<uint8_t[]> reordered(new uint8_t[size_t(row_bytes) * height]);
  std::memset(reordered.get(), 0, size_t(row_bytes) * height);
  const uint8_t* src = static_cast<const uint8_t*>(snapshot);
  uint32_t tile_count = xenos::kEdramTileCount;
  for (uint32_t tile = 0; tile < tile_count; ++tile) {
    uint32_t tx = (tile % tiles_per_row) * tile_w;
    uint32_t ty = (tile / tiles_per_row) * tile_h;
    if (ty + tile_h > height) {
      break;
    }
    const uint8_t* tile_src = src + size_t(tile) * tile_w * tile_h * 4;
    for (uint32_t row = 0; row < tile_h; ++row) {
      std::memcpy(reordered.get() + size_t(ty + row) * row_bytes + tx * 4,
                  tile_src + size_t(row) * tile_w * 4, tile_w * 4);
    }
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  id<MTLBuffer> staging =
      [device newBufferWithBytes:reordered.get()
                          length:size_t(row_bytes) * height
                         options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)
      command_processor_.PrepareTransferCommandBuffer();
  if (!staging || !cb) {
    return;
  }
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromBuffer:staging
             sourceOffset:0
        sourceBytesPerRow:row_bytes
      sourceBytesPerImage:size_t(row_bytes) * height
               sourceSize:MTLSizeMake(width, height, 1)
                toTexture:texture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
  [blit endEncoding];
  XELOGI("MetalRenderTargetCache: EDRAM snapshot restored into {}",
         full_rt->key().GetDebugName());
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
