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
      if (source->pixel_format() == dest->pixel_format()) {
        // Same host format (any pitch/MSAA — all textures are 1x-sized): copy
        // tile-by-tile. Each EDRAM tile is an 80x16 (40x16 at 64bpp) pixel
        // block at a computable position in each texture's layout.
        if (!blit) {
          blit = [cb blitCommandEncoder];
        }
        id<MTLTexture> src_tex = (__bridge id<MTLTexture>)source->texture();
        id<MTLTexture> dst_tex = (__bridge id<MTLTexture>)dest->texture();
        uint32_t tile_w = src_key.Is64bpp() ? 40 : 80;
        uint32_t src_pitch = src_key.GetPitchTiles();
        uint32_t dst_pitch = dest_key.GetPitchTiles();
        for (uint32_t tile = transfer.start_tiles; tile < transfer.end_tiles;
             ++tile) {
          uint32_t sl = tile - src_key.base_tiles;
          uint32_t dl = tile - dest_key.base_tiles;
          uint32_t sx = (sl % src_pitch) * tile_w, sy = (sl / src_pitch) * 16;
          uint32_t dx = (dl % dst_pitch) * tile_w, dy = (dl / dst_pitch) * 16;
          if (sx + tile_w > src_tex.width || sy + 16 > src_tex.height ||
              dx + tile_w > dst_tex.width || dy + 16 > dst_tex.height) {
            continue;
          }
          [blit copyFromTexture:src_tex
                    sourceSlice:0
                    sourceLevel:0
                   sourceOrigin:MTLOriginMake(sx, sy, 0)
                     sourceSize:MTLSizeMake(tile_w, 16, 1)
                      toTexture:dst_tex
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(dx, dy, 0)];
        }
      } else if (dest_key.is_depth && !src_key.is_depth &&
                 source->pixel_format() == kColorFormat) {
        // Color (raw EDRAM bits in a BGRA8 texture, e.g. the restored
        // snapshot) reinterpreted as depth: unpack the 24-bit guest depth and
        // write host D32F via a depth-only pass.
        if (blit) {
          [blit endEncoding];
          blit = nil;
        }
        PerformColorToDepthTransfer((__bridge const void*)cb, source, dest,
                                    transfer);
      } else if (!dest_key.is_depth && src_key.is_depth &&
                 dest->pixel_format() == kColorFormat) {
        // Depth reinterpreted as raw-bits color (a color pass reuses tiles a
        // depth RT last owned): encode host D32F back to the guest 24-bit
        // pattern and write it as BGRA8 bytes.
        if (blit) {
          [blit endEncoding];
          blit = nil;
        }
        PerformDepthToColorTransfer((__bridge const void*)cb, source, dest,
                                    transfer);
      } else {
        static uint32_t s_skips = 0;
        if (s_skips++ < 24) {
          XELOGW("MetalRenderTargetCache: SKIP transfer {} -> {} tiles [{},{})",
                 src_key.GetDebugName(), dest_key.GetDebugName(),
                 transfer.start_tiles, transfer.end_tiles);
        }
      }
    }
  }
  if (blit) {
    [blit endEncoding];
  }
}

void MetalRenderTargetCache::PerformColorToDepthTransfer(
    const void* command_buffer, RenderTarget* source_rt, RenderTarget* dest_rt,
    const Transfer& transfer) {
  auto* source = static_cast<MetalRenderTarget*>(source_rt);
  auto* dest = static_cast<MetalRenderTarget*>(dest_rt);
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  if (!transfer_c2d_pipeline_) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct XferParams {
        uint src_base; uint src_pitch; uint dst_base; uint dst_pitch;
        uint start_tiles; uint end_tiles; uint is_float24; uint pad;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_xfer_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      struct DepthOut { float depth [[depth(any)]]; };
      // 20e4 (4-bit exponent, 20-bit mantissa, [0,2)) guest float24 -> float32.
      static float xe_f20e4_to_f32(uint f24) {
        f24 &= 0xFFFFFFu;
        if (f24 == 0u) return 0.0;
        uint mantissa = f24 & 0xFFFFFu;
        uint exponent = f24 >> 20;
        if (exponent == 0u) {
          uint lzcnt = clz(mantissa) - (32u - 21u);
          exponent = uint(1 - int(lzcnt));
          mantissa = (mantissa << lzcnt) & 0xFFFFFu;
        }
        return as_type<float>(((exponent + 112u) << 23) | (mantissa << 3));
      }
      fragment DepthOut xe_xfer_c2d_fs(VOut in [[stage_in]],
                                       texture2d<float> src [[texture(0)]],
                                       constant XferParams& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint dl = (dp.y / 16u) * p.dst_pitch + (dp.x / 80u);
        uint tile = p.dst_base + dl;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        uint2 sp = uint2((sl % p.src_pitch) * 80u + (dp.x % 80u),
                         (sl / p.src_pitch) * 16u + (dp.y % 16u));
        float4 c = src.read(sp, 0);
        // BGRA8 texel bytes back to the raw EDRAM word; depth in bits [8,32).
        uint bits = uint(round(c.b * 255.0)) | (uint(round(c.g * 255.0)) << 8) |
                    (uint(round(c.r * 255.0)) << 16) |
                    (uint(round(c.a * 255.0)) << 24);
        uint d24 = bits >> 8;
        DepthOut o;
        o.depth = p.is_float24 != 0u ? xe_f20e4_to_f32(d24)
                                     : float(d24) / 16777215.0;
        return o;
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: c2d transfer library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = [lib newFunctionWithName:@"xe_xfer_vs"];
    d.fragmentFunction = [lib newFunctionWithName:@"xe_xfer_c2d_fs"];
    d.depthAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) {
      XELOGE("MetalRenderTargetCache: c2d transfer pipeline failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    transfer_c2d_pipeline_ = (__bridge_retained void*)ps;
    MTLDepthStencilDescriptor* ds = [[MTLDepthStencilDescriptor alloc] init];
    ds.depthCompareFunction = MTLCompareFunctionAlways;
    ds.depthWriteEnabled = YES;
    transfer_c2d_depth_state_ =
        (__bridge_retained void*)[device newDepthStencilStateWithDescriptor:ds];
  }
  if (!transfer_c2d_pipeline_ || !transfer_c2d_depth_state_) {
    return;
  }
  struct XferParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, is_float24, pad;
  } params{source->key().base_tiles,     source->key().GetPitchTiles(),
           dest->key().base_tiles,       dest->key().GetPitchTiles(),
           transfer.start_tiles,         transfer.end_tiles,
           uint32_t(dest->key().GetDepthFormat() ==
                    xenos::DepthRenderTargetFormat::kD24FS8),
           0};
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.depthAttachment.texture = (__bridge id<MTLTexture>)dest->texture();
  pass.depthAttachment.loadAction = MTLLoadActionLoad;
  pass.depthAttachment.storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                  transfer_c2d_pipeline_];
  [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)
                                transfer_c2d_depth_state_];
  [enc setFragmentTexture:(__bridge id<MTLTexture>)source->texture() atIndex:0];
  [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [enc endEncoding];
}

void MetalRenderTargetCache::PerformDepthToColorTransfer(
    const void* command_buffer, RenderTarget* source_rt, RenderTarget* dest_rt,
    const Transfer& transfer) {
  auto* source = static_cast<MetalRenderTarget*>(source_rt);
  auto* dest = static_cast<MetalRenderTarget*>(dest_rt);
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  if (!transfer_d2c_pipeline_) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct XferParams {
        uint src_base; uint src_pitch; uint dst_base; uint dst_pitch;
        uint start_tiles; uint end_tiles; uint is_float24; uint pad;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_xfer_d2c_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      // float32 -> guest 20e4 float24 (truncating).
      static uint xe_f32_to_f20e4(float f) {
        if (!(f > 0.0)) return 0u;
        uint u = as_type<uint>(f);
        if (u >= 0x40000000u) return 0xFFFFFFu;  // >= 2.0 saturates.
        if (u < 0x38800000u) {
          // Denormal in 20e4: shift the implicit-1 mantissa down.
          uint shift = min(113u - (u >> 23), 24u);
          u = (0x800000u | (u & 0x7FFFFFu)) >> shift;
        } else {
          u -= 0x38000000u;
        }
        return (u >> 3) & 0xFFFFFFu;
      }
      fragment float4 xe_xfer_d2c_fs(VOut in [[stage_in]],
                                     depth2d<float> src [[texture(0)]],
                                     constant XferParams& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint dl = (dp.y / 16u) * p.dst_pitch + (dp.x / 80u);
        uint tile = p.dst_base + dl;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        uint2 sp = uint2((sl % p.src_pitch) * 80u + (dp.x % 80u),
                         (sl / p.src_pitch) * 16u + (dp.y % 16u));
        float d = src.read(sp);
        uint d24 = p.is_float24 != 0u
                       ? xe_f32_to_f20e4(d)
                       : uint(round(saturate(d) * 16777215.0));
        uint bits = d24 << 8;  // stencil (low byte) not emulated.
        return float4(float(bits & 0xFFu), float((bits >> 8) & 0xFFu),
                      float((bits >> 16) & 0xFFu), float((bits >> 24) & 0xFFu)) /
               255.0;
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: d2c transfer library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = [lib newFunctionWithName:@"xe_xfer_d2c_vs"];
    d.fragmentFunction = [lib newFunctionWithName:@"xe_xfer_d2c_fs"];
    d.colorAttachments[0].pixelFormat = (MTLPixelFormat)kColorFormat;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) {
      XELOGE("MetalRenderTargetCache: d2c transfer pipeline failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    transfer_d2c_pipeline_ = (__bridge_retained void*)ps;
  }
  if (!transfer_d2c_pipeline_) {
    return;
  }
  struct XferParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, is_float24, pad;
  } params{source->key().base_tiles,     source->key().GetPitchTiles(),
           dest->key().base_tiles,       dest->key().GetPitchTiles(),
           transfer.start_tiles,         transfer.end_tiles,
           uint32_t(source->key().GetDepthFormat() ==
                    xenos::DepthRenderTargetFormat::kD24FS8),
           0};
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest->texture();
  pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                  transfer_d2c_pipeline_];
  [enc setFragmentTexture:(__bridge id<MTLTexture>)source->texture() atIndex:0];
  [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [enc endEncoding];
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

bool MetalRenderTargetCache::ResolveDepthToSnapshot(
    const void* command_buffer, uint32_t span_base, uint32_t row_length_used,
    uint32_t rows, uint32_t span_pitch, bool is_float24, void* dest_snapshot,
    uint32_t width, uint32_t height, uint32_t dest_y_offset) {
  if (!command_buffer || !dest_snapshot) {
    return false;
  }
  // The span can be owned PIECEWISE by several render targets; convert each
  // owner's rectangle from its texture.
  resolve_dump_rectangles_.clear();
  GetResolveCopyRectanglesToDump(span_base, row_length_used, rows, span_pitch,
                                 resolve_dump_rectangles_);
  if (resolve_dump_rectangles_.empty()) {
    return false;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  if (!resolve_c2f_pipeline_) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct Params {
        uint src_base; uint src_pitch; uint span_base; uint span_pitch;
        uint y_offset; uint is_float24; uint pad0; uint pad1;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_c2f_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      static float xe_f20e4_to_f32(uint f24) {
        f24 &= 0xFFFFFFu;
        if (f24 == 0u) return 0.0;
        uint mantissa = f24 & 0xFFFFFu;
        uint exponent = f24 >> 20;
        if (exponent == 0u) {
          uint lzcnt = clz(mantissa) - (32u - 21u);
          exponent = uint(1 - int(lzcnt));
          mantissa = (mantissa << lzcnt) & 0xFFFFFu;
        }
        return as_type<float>(((exponent + 112u) << 23) | (mantissa << 3));
      }
      fragment float xe_c2f_fs(VOut in [[stage_in]],
                               texture2d<float> src [[texture(0)]],
                               constant Params& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint ry = dp.y - p.y_offset;  // row within the resolve region.
        uint tile = p.span_base + (ry / 16u) * p.span_pitch + (dp.x / 80u);
        uint sl = tile - p.src_base;
        uint2 sp = uint2((sl % p.src_pitch) * 80u + (dp.x % 80u),
                         (sl / p.src_pitch) * 16u + (ry % 16u));
        float4 c = src.read(sp, 0);
        uint bits = uint(round(c.b * 255.0)) | (uint(round(c.g * 255.0)) << 8) |
                    (uint(round(c.r * 255.0)) << 16) |
                    (uint(round(c.a * 255.0)) << 24);
        uint d24 = bits >> 8;
        return p.is_float24 != 0u ? xe_f20e4_to_f32(d24)
                                  : float(d24) / 16777215.0;
      }
      fragment float xe_d2f_fs(VOut in [[stage_in]],
                               depth2d<float> src [[texture(0)]],
                               constant Params& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint ry = dp.y - p.y_offset;
        uint tile = p.span_base + (ry / 16u) * p.span_pitch + (dp.x / 80u);
        uint sl = tile - p.src_base;
        uint2 sp = uint2((sl % p.src_pitch) * 80u + (dp.x % 80u),
                         (sl / p.src_pitch) * 16u + (ry % 16u));
        return src.read(sp);
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: resolve library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return false;
    }
    auto make_pipeline = [&](NSString* fs_name) -> void* {
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_c2f_vs"];
      d.fragmentFunction = [lib newFunctionWithName:fs_name];
      d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
      NSError* perr = nil;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&perr];
      if (!ps) {
        XELOGE("MetalRenderTargetCache: resolve pipeline {} failed: {}",
               fs_name.UTF8String,
               perr ? perr.localizedDescription.UTF8String : "?");
        return nullptr;
      }
      return (__bridge_retained void*)ps;
    };
    resolve_c2f_pipeline_ = make_pipeline(@"xe_c2f_fs");
    resolve_d2f_pipeline_ = make_pipeline(@"xe_d2f_fs");
  }
  if (!resolve_c2f_pipeline_ || !resolve_d2f_pipeline_) {
    return false;
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest_snapshot;
  pass.colorAttachments[0].slice = 0;
  pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  bool any = false;
  for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
    auto* owner = static_cast<MetalRenderTarget*>(rect.render_target);
    if (!owner) {
      continue;
    }
    bool owner_is_depth = owner->key().is_depth != 0;
    if (!owner_is_depth && owner->pixel_format() != kColorFormat) {
      continue;  // Raw-bits decode only understands 32bpp color for now.
    }
    // The rect's tile rows within the span -> destination pixel rows.
    uint32_t y0 = rect.row_first * 16;
    if (y0 >= height) {
      continue;
    }
    uint32_t h = std::min(rect.rows * 16, height - y0);
    struct Params {
      uint32_t src_base, src_pitch, span_base, span_pitch;
      uint32_t y_offset, is_float24, pad0, pad1;
    } params{owner->key().base_tiles, owner->key().GetPitchTiles(),
             span_base,               span_pitch,
             dest_y_offset,           uint32_t(is_float24),
             0,                       0};
    [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)(
                                    owner_is_depth ? resolve_d2f_pipeline_
                                                   : resolve_c2f_pipeline_)];
    MTLViewport vp{0.0, double(dest_y_offset + y0), double(width), double(h),
                   0.0, 1.0};
    [enc setViewport:vp];
    // Viewports don't clip (the fullscreen triangle overflows 2x); scissor
    // does — without it each rect would stomp the rows below it.
    id<MTLTexture> dest_tex = (__bridge id<MTLTexture>)dest_snapshot;
    uint32_t sy = std::min<uint32_t>(dest_y_offset + y0,
                                     uint32_t(dest_tex.height));
    uint32_t sh =
        std::min<uint32_t>(h, uint32_t(dest_tex.height) - sy);
    if (!sh) {
      continue;
    }
    [enc setScissorRect:MTLScissorRect{0, sy,
                                       std::min<NSUInteger>(width,
                                                            dest_tex.width),
                                       sh}];
    [enc setFragmentTexture:(__bridge id<MTLTexture>)owner->texture()
                    atIndex:0];
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    any = true;
    static uint32_t s_logs = 0;
    if (s_logs++ < 16) {
      XELOGI(
          "MetalRenderTargetCache: depth resolve rect rows[{}+{}] from {}",
          rect.row_first, rect.rows, owner->key().GetDebugName());
    }
  }
  [enc endEncoding];
  return any;
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
  {
    // Diagnostic: sample the snapshot content (is it real data or a fill
    // pattern?). Words from the depth region (tiles ~16+) and a simple sum.
    const uint32_t* words = static_cast<const uint32_t*>(snapshot);
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 4096; ++i) {
      sum += words[i * 512];
    }
    XELOGI(
        "MetalRenderTargetCache: EDRAM snapshot sample w[0]={:08X} "
        "w[20K]={:08X} w[500K]={:08X} w[2M]={:08X} sum16M={:016X}",
        words[0], words[20000], words[500000], words[2000000], sum);
  }
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
