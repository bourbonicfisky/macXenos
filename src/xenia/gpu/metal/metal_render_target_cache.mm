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

const uint64_t MetalRenderTargetCache::kColorFormat = MTLPixelFormatRGBA8Unorm;
// Combined depth+stencil: the guest's D24S8/D24FS8 stencil plane is real state
// (JC2 gates its half-screen passes on it); a stencil-less depth RT silently
// zeroes the stencil byte on every EDRAM round-trip.
const uint64_t MetalRenderTargetCache::kDepthFormat =
    MTLPixelFormatDepth32Float_Stencil8;

namespace {
// Guest color RESOURCE format (gamma already collapsed by the base class since
// IsGammaFormatHostStorageSeparate()==false) -> host MTLPixelFormat.
MTLPixelFormat ColorResourceFormatToMetal(
    xenos::ColorRenderTargetFormat format) {
  switch (format) {
    case xenos::ColorRenderTargetFormat::k_8_8_8_8:
    case xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA:
      return MTLPixelFormatRGBA8Unorm;
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
      return MTLPixelFormatRGBA8Unorm;
  }
}
}  // namespace

namespace {
// Sample-space -> pixel shifts for a key (Xenos MSAA: 2x doubles vertical
// samples, 4x doubles both).
void GetMsaaShifts(uint32_t msaa_samples, uint32_t& x_shift,
                   uint32_t& y_shift) {
  x_shift = msaa_samples >= uint32_t(xenos::MsaaSamples::k4X) ? 1 : 0;
  y_shift = msaa_samples >= uint32_t(xenos::MsaaSamples::k2X) ? 1 : 0;
}
}  // namespace

MetalRenderTargetCache::MetalRenderTarget::~MetalRenderTarget() {
  if (texture_) {
    // Defer, don't free: in-flight encoders may still reference this texture
    // (see the class comment). The cache drains after the frame's GPU work
    // completes.
    cache_.deferred_texture_releases_.emplace_back(
        MetalCommandProcessor::S7TagsOff()
            ? 0
            : cache_.command_processor_.CurrentEncodingSubmission(),
        texture_);
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
  // Canonical per-sample EDRAM buffer (source of truth for the sample-accurate
  // path). Allocated only when gated on, so the default collapsed path stays
  // byte-for-byte unchanged. See openspec implement-sample-accurate-metal-edram
  // task 2.1.
  sample_edram_enabled_ = std::getenv("XE_METAL_SAMPLE_EDRAM") != nullptr;
  if (sample_edram_enabled_) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    id<MTLBuffer> buffer =
        [device newBufferWithLength:kCanonicalEdramSize
                            options:MTLResourceStorageModePrivate];
    if (!buffer) {
      XELOGE(
          "MetalRenderTargetCache: canonical EDRAM allocation failed ({} MB); "
          "falling back to the collapsed path",
          kCanonicalEdramSize / (1024 * 1024));
      sample_edram_enabled_ = false;
    } else {
      buffer.label = @"xe_canonical_edram";
      canonical_edram_ = (__bridge_retained void*)buffer;
      XELOGI(
          "MetalRenderTargetCache: canonical per-sample EDRAM enabled "
          "({} MB private buffer)",
          kCanonicalEdramSize / (1024 * 1024));
    }
  }
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
  if (canonical_edram_) {
    id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)canonical_edram_;
    (void)released;
    canonical_edram_ = nullptr;
  }
  if (canonical_materialize_pipeline_) {
    id<MTLComputePipelineState> released =
        (__bridge_transfer id<MTLComputePipelineState>)canonical_materialize_pipeline_;
    (void)released;
    canonical_materialize_pipeline_ = nullptr;
  }
  if (canonical_materialize_library_) {
    id<MTLLibrary> released =
        (__bridge_transfer id<MTLLibrary>)canonical_materialize_library_;
    (void)released;
    canonical_materialize_library_ = nullptr;
  }
  presentable_color_target_ = nullptr;
  // Destroys all MetalRenderTargets (retiring their textures to the deferred
  // list), then the common ownership state; drain LAST so those retirements
  // are actually freed.
  DestroyAllRenderTargets(true);
  ShutdownCommon();
  DrainDeferredReleases(UINT64_MAX);
}

void MetalRenderTargetCache::ClearTextureAtCreation(void* texture,
                                                    bool is_depth) {
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)
      command_processor_.PrepareTransferCommandBuffer();
  if (!cb || !texture) {
    return;
  }
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  if (is_depth) {
    pass.depthAttachment.texture = (__bridge id<MTLTexture>)texture;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = 0.0;  // Reverse-z far (JC2 convention).
    pass.stencilAttachment.texture = (__bridge id<MTLTexture>)texture;
    pass.stencilAttachment.loadAction = MTLLoadActionClear;
    pass.stencilAttachment.storeAction = MTLStoreActionStore;
    pass.stencilAttachment.clearStencil = 0;
  } else {
    pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)texture;
    pass.colorAttachments[0].slice = 0;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  }
  id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
  enc.label = @"rtcache:clear-at-creation";
  [enc endEncoding];
}

RenderTargetCache::RenderTarget* MetalRenderTargetCache::CreateRenderTarget(
    RenderTargetKey key) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLPixelFormat pixel_format =
      key.is_depth ? (MTLPixelFormat)kDepthFormat
                   : ColorResourceFormatToMetal(key.GetColorFormat());
  // Mainline sizing: pitch_tiles_at_32bpp is SAMPLE-space (a 4x key's pitch is
  // doubled), GetWidth()/GetRenderTargetHeight() convert to msaa-aware PIXEL
  // dimensions. Textures are still single-sample (no host MSAA) — MSAA'd guest
  // surfaces just render aliased at their pixel size; all tile math converts
  // sample coordinates to pixels with the key's shifts.
  uint32_t width = key.GetWidth();
  uint32_t height =
      GetRenderTargetHeight(key.pitch_tiles_at_32bpp, key.msaa_samples);
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
  ClearTextureAtCreation((__bridge void*)texture, key.is_depth != 0);
  return new MetalRenderTarget(key, *this, (__bridge_retained void*)texture,
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
      uint32_t sxs, sys, dxs, dys;
      GetMsaaShifts(uint32_t(src_key.msaa_samples), sxs, sys);
      GetMsaaShifts(uint32_t(dest_key.msaa_samples), dxs, dys);
      if (source->pixel_format() == dest->pixel_format() && sxs == dxs &&
          sys == dys) {
        // Same host format + same MSAA (equal pixel block sizes): copy
        // tile-by-tile. An EDRAM tile is 80x16 SAMPLES = an
        // (80>>xs)x(16>>ys)-pixel block at a computable position per layout.
        if (!blit) {
          blit = [cb blitCommandEncoder];
          blit.label = @"rtcache:transfer-blit";
        }
        static uint32_t s_blit_logs = 0;
        if (s_blit_logs++ < 256) {
          XELOGW("MetalXferBlit @draw {}: {} -> {} tiles [{},{})",
                 command_processor_.trace_draw_index(),
                 src_key.GetDebugName(), dest_key.GetDebugName(),
                 transfer.start_tiles, transfer.end_tiles);
        }
        id<MTLTexture> src_tex = (__bridge id<MTLTexture>)source->texture();
        id<MTLTexture> dst_tex = (__bridge id<MTLTexture>)dest->texture();
        uint32_t tile_w_samples = src_key.Is64bpp() ? 40 : 80;
        uint32_t bw = tile_w_samples >> sxs, bh = 16 >> sys;
        uint32_t src_pitch = src_key.GetPitchTiles();
        uint32_t dst_pitch = dest_key.GetPitchTiles();
        for (uint32_t tile = transfer.start_tiles; tile < transfer.end_tiles;
             ++tile) {
          uint32_t sl = tile - src_key.base_tiles;
          uint32_t dl = tile - dest_key.base_tiles;
          uint32_t sx = ((sl % src_pitch) * tile_w_samples) >> sxs,
                   sy = ((sl / src_pitch) * 16) >> sys;
          uint32_t dx = ((dl % dst_pitch) * tile_w_samples) >> dxs,
                   dy = ((dl / dst_pitch) * 16) >> dys;
          if (sx + bw > src_tex.width || sy + bh > src_tex.height ||
              dx + bw > dst_tex.width || dy + bh > dst_tex.height) {
            continue;
          }
          [blit copyFromTexture:src_tex
                    sourceSlice:0
                    sourceLevel:0
                   sourceOrigin:MTLOriginMake(sx, sy, 0)
                     sourceSize:MTLSizeMake(bw, bh, 1)
                      toTexture:dst_tex
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(dx, dy, 0)];
        }
      } else if (source->pixel_format() == dest->pixel_format() &&
                 src_key.is_depth == dest_key.is_depth &&
                 !src_key.Is64bpp() && !dest_key.Is64bpp()) {
        // Same host format, different MSAA sample density: per-tile pixel
        // blocks have different sizes, so remap through a sampling pass.
        if (blit) {
          [blit endEncoding];
          blit = nil;
        }
        PerformCrossMsaaTransfer((__bridge const void*)cb, source, dest,
                                 transfer);
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
        // Remember who lent these tiles: if the SAME color key takes them
        // back this frame, that d2c must transport the real bits (below).
        c2d_pingpong_[dest] = C2dPingPong{source, transfer.start_tiles,
                                          transfer.end_tiles,
                                          command_processor_.swap_count()};
      } else if (!dest_key.is_depth && src_key.is_depth &&
                 dest->pixel_format() == kColorFormat) {
        // Depth reinterpreted as raw-bits color (a color pass reuses tiles a
        // depth RT last owned). DEFAULT: SKIP — the color key keeps its own
        // previous content ("key-local EDRAM persistence"). Both consumer
        // patterns want that: JC2's tonemap draws BLEND AGAINST DESTINATION
        // over these tiles (writing zeros here rendered the casino's dark
        // rectangles; real depth bits render as garbage the multiply
        // amplifies), and the composite's depth-owned resolve spans read
        // whatever the chain last produced, exactly like the mainline truth
        // leg. XE_METAL_D2C_REAL=1 encodes the real depth bits instead
        // (hardware-accurate raw reinterpret, for titles that genuinely
        // read them).
        static const bool s_d2c_real = std::getenv("XE_METAL_D2C_REAL");
        // Same-frame ping-pong exception: this color key lent exactly these
        // tiles to that depth key THIS frame (Sonic's DoF pass writes a
        // packed motion-vector/CoC map as 4x depth bits over the blur
        // buffer's tiles and the composite decodes the raw bits — skipping
        // left stale scene pixels to be decoded as gather vectors: the
        // melted mip / white screen).
        bool pingpong_return = false;
        auto pp = c2d_pingpong_.find(source);
        if (pp != c2d_pingpong_.end() &&
            pp->second.src_color_rt == (const void*)dest &&
            pp->second.swap == command_processor_.swap_count() &&
            transfer.start_tiles < pp->second.end_tiles &&
            transfer.end_tiles > pp->second.start_tiles) {
          pingpong_return = true;
          static uint32_t s_pp_logs = 0;
          if (s_pp_logs++ < 16) {
            XELOGW("MetalXferD2C @draw {}: ping-pong return {} -> {} tiles "
                   "[{},{}) — real bit transport",
                   command_processor_.trace_draw_index(),
                   src_key.GetDebugName(), dest_key.GetDebugName(),
                   transfer.start_tiles, transfer.end_tiles);
          }
        }
        if (s_d2c_real || pingpong_return) {
          if (blit) {
            [blit endEncoding];
            blit = nil;
          }
          PerformDepthToColorTransfer((__bridge const void*)cb, source, dest,
                                      transfer, pingpong_return);
        }
      } else if (!src_key.is_depth && !dest_key.is_depth &&
                 !src_key.Is64bpp() && !dest_key.Is64bpp() &&
                 ((source->pixel_format() == kColorFormat &&
                   dest->pixel_format() == uint64_t(MTLPixelFormatR32Float)) ||
                  (source->pixel_format() == uint64_t(MTLPixelFormatR32Float) &&
                   dest->pixel_format() == kColorFormat))) {
        // Color<->color 32bpp REINTERPRET: the EDRAM word is identical, only
        // the host container differs (BGRA8 bytes vs f32 bits). JC2's
        // auto-exposure pyramid renders into k_32_FLOAT-keyed tiles that then
        // alias k_8_8_8_8 keys (and vice versa); this was the last log-skipped
        // transfer class — the luminance content stayed stranded in the other
        // key's texture and the exposure loop read zeros (the black flicker).
        if (blit) {
          [blit endEncoding];
          blit = nil;
        }
        PerformColorReinterpretTransfer((__bridge const void*)cb, source, dest,
                                        transfer);
      } else if (!src_key.is_depth && !dest_key.is_depth &&
                 !src_key.Is64bpp() && !dest_key.Is64bpp()) {
        // General 32bpp color->color transfer: encode the source texel to the
        // raw EDRAM word (per source host format), decode per dest format,
        // with the full tile remap (handles reinterpret AND cross-MSAA in one
        // pass). Covers the combos the specific handlers don't — Sonic's HDR
        // chain runs on k_2_10_10_10_FLOAT (host RGBA16F, 7e3-packed words)
        // and every one of its ownership handoffs to/from 8888 was skipped,
        // so the bloom pyramid read stale content (the scrambled top band).
        if (blit) {
          [blit endEncoding];
          blit = nil;
        }
        PerformColorConvertTransfer((__bridge const void*)cb, source, dest,
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

void MetalRenderTargetCache::PerformCrossMsaaTransfer(
    const void* command_buffer, RenderTarget* source_rt, RenderTarget* dest_rt,
    const Transfer& transfer) {
  auto* source = static_cast<MetalRenderTarget*>(source_rt);
  auto* dest = static_cast<MetalRenderTarget*>(dest_rt);
  // The half-height double-stack forensics: which draw triggers a cross-MSAA
  // transfer over which tiles (throttled; replay has dozens per frame).
  static uint32_t s_x2x_logs = 0;
  if (s_x2x_logs++ < 256) {
    id<MTLTexture> st = (__bridge id<MTLTexture>)source->texture();
    id<MTLTexture> dt = (__bridge id<MTLTexture>)dest->texture();
    XELOGW("MetalX2X @draw {}: {} ({}x{} tex={}) -> {} ({}x{} tex={}) "
           "tiles [{},{})",
           command_processor_.trace_draw_index(),
           source->key().GetDebugName(), st.width, st.height,
           (void*)source->texture(), dest->key().GetDebugName(), dt.width,
           dt.height, (void*)dest->texture(), transfer.start_tiles,
           transfer.end_tiles);
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  bool is_depth = dest->key().is_depth != 0;
  // XE_METAL_X2X_PROBE=1: after the first depth transfer, flush and read
  // back dest vs source pixels (verifies the copy lands). =2: at the SECOND
  // depth transfer, read the SOURCE (the msaa RT after the guest draw on it)
  // BEFORE encoding anything — locates where transferred content gets wiped.
  if (!transfer_x2x_library_) {
    // Same tile arithmetic as the reinterpret transfers; host->host, so no
    // raw-word decode and no depth 40-column swap.
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct XferParams {
        uint src_base; uint src_pitch; uint dst_base; uint dst_pitch;
        uint start_tiles; uint end_tiles; uint stencil_bit; uint shifts;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_x2x_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      // shifts: dst_x | dst_y<<4 | src_x<<8 | src_y<<12 (sample->pixel).
      static uint2 xe_x2x_src(uint2 dp, constant XferParams& p,
                              thread bool& in_range) {
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint dl = (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        uint tile = p.dst_base + dl;
        in_range = tile >= p.start_tiles && tile < p.end_tiles;
        uint sl = tile - p.src_base;
        uint2 ss = uint2((sl % p.src_pitch) * 80u + (ds.x % 80u),
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        return uint2(ss.x >> ((p.shifts >> 8) & 15u),
                     ss.y >> ((p.shifts >> 12) & 15u));
      }
      fragment float4 xe_x2x_col_fs(VOut in [[stage_in]],
                                    texture2d<float> src [[texture(0)]],
                                    constant XferParams& p [[buffer(0)]]) {
        bool in_range;
        uint2 sp = xe_x2x_src(uint2(in.pos.xy), p, in_range);
        if (!in_range) {
          discard_fragment();
        }
        return src.read(sp, 0);
      }
      struct DepthOut { float depth [[depth(any)]]; };
      fragment DepthOut xe_x2x_dep_fs(VOut in [[stage_in]],
                                      depth2d<float> src [[texture(0)]],
                                      constant XferParams& p [[buffer(0)]]) {
        bool in_range;
        uint2 sp = xe_x2x_src(uint2(in.pos.xy), p, in_range);
        if (!in_range) {
          discard_fragment();
        }
        DepthOut o;
        o.depth = src.read(sp);
        return o;
      }
      // Stencil bit-plane pass (source stencil sampled via an X32_Stencil8
      // view bound as texture2d<uint>); same masked-REPLACE scheme as c2d.
      fragment void xe_x2x_sten_fs(VOut in [[stage_in]],
                                   texture2d<uint> src [[texture(0)]],
                                   constant XferParams& p [[buffer(0)]]) {
        bool in_range;
        uint2 sp = xe_x2x_src(uint2(in.pos.xy), p, in_range);
        if (!in_range) {
          discard_fragment();
        }
        if ((src.read(sp, 0).r & p.stencil_bit) == 0u) {
          discard_fragment();
        }
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: x2x transfer library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    transfer_x2x_library_ = (__bridge_retained void*)lib;
  }
  id<MTLLibrary> lib = (__bridge id<MTLLibrary>)transfer_x2x_library_;
  id<MTLRenderPipelineState> pipeline = nil;
  if (is_depth) {
    if (!transfer_x2x_depth_pipeline_) {
      NSError* err = nil;
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_x2x_vs"];
      d.fragmentFunction = [lib newFunctionWithName:@"xe_x2x_dep_fs"];
      d.depthAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
      d.stencilAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&err];
      if (!ps) {
        XELOGE("MetalRenderTargetCache: x2x depth pipeline failed: {}",
               err ? err.localizedDescription.UTF8String : "?");
        return;
      }
      transfer_x2x_depth_pipeline_ = (__bridge_retained void*)ps;
      // The depth pass also zeroes the rect's stencil (REPLACE ref=0) as the
      // base the bit passes below OR the source stencil into.
      MTLDepthStencilDescriptor* ds = [[MTLDepthStencilDescriptor alloc] init];
      ds.depthCompareFunction = MTLCompareFunctionAlways;
      ds.depthWriteEnabled = YES;
      MTLStencilDescriptor* zero_sten = [[MTLStencilDescriptor alloc] init];
      zero_sten.stencilCompareFunction = MTLCompareFunctionAlways;
      zero_sten.depthStencilPassOperation = MTLStencilOperationReplace;
      zero_sten.stencilFailureOperation = MTLStencilOperationReplace;
      zero_sten.depthFailureOperation = MTLStencilOperationReplace;
      zero_sten.writeMask = 0xFF;
      ds.frontFaceStencil = zero_sten;
      ds.backFaceStencil = zero_sten;
      transfer_x2x_depth_state_ =
          (__bridge_retained void*)[device newDepthStencilStateWithDescriptor:ds];
      MTLRenderPipelineDescriptor* sd =
          [[MTLRenderPipelineDescriptor alloc] init];
      sd.vertexFunction = [lib newFunctionWithName:@"xe_x2x_vs"];
      sd.fragmentFunction = [lib newFunctionWithName:@"xe_x2x_sten_fs"];
      sd.depthAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
      sd.stencilAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
      id<MTLRenderPipelineState> sps =
          [device newRenderPipelineStateWithDescriptor:sd error:&err];
      if (sps) {
        transfer_x2x_sten_pipeline_ = (__bridge_retained void*)sps;
      } else {
        XELOGE("MetalRenderTargetCache: x2x stencil pipeline failed: {}",
               err ? err.localizedDescription.UTF8String : "?");
      }
      EnsureStencilBitStates();
    }
    pipeline =
        (__bridge id<MTLRenderPipelineState>)transfer_x2x_depth_pipeline_;
  } else {
    uint64_t fmt = dest->pixel_format();
    auto it = transfer_x2x_color_pipelines_.find(fmt);
    if (it == transfer_x2x_color_pipelines_.end()) {
      NSError* err = nil;
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_x2x_vs"];
      d.fragmentFunction = [lib newFunctionWithName:@"xe_x2x_col_fs"];
      d.colorAttachments[0].pixelFormat = (MTLPixelFormat)fmt;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&err];
      if (!ps) {
        XELOGE("MetalRenderTargetCache: x2x color pipeline {} failed: {}", fmt,
               err ? err.localizedDescription.UTF8String : "?");
      }
      it = transfer_x2x_color_pipelines_
               .emplace(fmt, ps ? (__bridge_retained void*)ps : nullptr)
               .first;
    }
    pipeline = (__bridge id<MTLRenderPipelineState>)it->second;
  }
  if (!pipeline) {
    return;
  }
  uint32_t sxs, sys, dxs, dys;
  GetMsaaShifts(uint32_t(source->key().msaa_samples), sxs, sys);
  GetMsaaShifts(uint32_t(dest->key().msaa_samples), dxs, dys);
  struct XferParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, stencil_bit, shifts;
  } params{source->key().base_tiles, source->key().GetPitchTiles(),
           dest->key().base_tiles,   dest->key().GetPitchTiles(),
           transfer.start_tiles,     transfer.end_tiles,
           0,                        dxs | (dys << 4) | (sxs << 8) | (sys << 12)};
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  if (is_depth) {
    pass.depthAttachment.texture = (__bridge id<MTLTexture>)dest->texture();
    pass.depthAttachment.loadAction = MTLLoadActionLoad;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.stencilAttachment.texture = (__bridge id<MTLTexture>)dest->texture();
    pass.stencilAttachment.loadAction = MTLLoadActionLoad;
    pass.stencilAttachment.storeAction = MTLStoreActionStore;
  } else {
    pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest->texture();
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  }
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  enc.label = @"rtcache:transfer-crossmsaa";
  [enc setRenderPipelineState:pipeline];
  if (is_depth) {
    [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)
                                  transfer_x2x_depth_state_];
    [enc setStencilReferenceValue:0];  // depth pass zeroes the rect's stencil.
  }
  [enc setFragmentTexture:(__bridge id<MTLTexture>)source->texture() atIndex:0];
  [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  // Default ON: the earlier "x2x stencil makes it worse" verdict was an
  // artifact of the dump's msaa-shift bug (enum-vs-count). With the dump
  // fixed, stencil transfer takes 7282 draw-25 567K -> 327K diff dwords and
  // draw 900 reaches FULL EDRAM parity (0 tiles). XE_METAL_X2X_STEN_OFF=1
  // disables for A/B.
  static const bool s_x2x_sten_off = std::getenv("XE_METAL_X2X_STEN_OFF");
  if (is_depth && !s_x2x_sten_off && transfer_x2x_sten_pipeline_ &&
      stencil_bit_states_[0]) {
    // Transfer the source stencil plane: 8 masked REPLACE passes reading an
    // X32_Stencil8 view of the source (texture2d<uint>, .r = stencil).
    id<MTLTexture> src_tex = (__bridge id<MTLTexture>)source->texture();
    id<MTLTexture> sten_view = [src_tex
        newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    static uint32_t s_view_logs = 0;
    if (!sten_view && s_view_logs++ < 4) {
      XELOGE("MetalX2X: X32_Stencil8 view creation FAILED (usage flags?)");
    }
    if (sten_view) {
      [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      transfer_x2x_sten_pipeline_];
      [enc setStencilReferenceValue:0xFF];
      [enc setFragmentTexture:sten_view atIndex:0];
      for (uint32_t bit = 0; bit < 8; ++bit) {
        params.stencil_bit = 1u << bit;
        [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)
                                      stencil_bit_states_[bit]];
        [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:3];
      }
    }
  }
  [enc endEncoding];
}


// General color->color transfer with format conversion through the raw EDRAM
// word. Source texel -> 32-bit word (per source host format) -> dest texel
// (per dest format), with the same dest-pixel -> tile -> source-pixel remap
// as the cross-MSAA pass, so one pipeline handles reinterpret, cross-MSAA,
// and both combined. Format classes: 0=RGBA8 (bytes are the word),
// 1=R32Float (bits are the word), 2=RGBA16F holding 7e3-packed
// k_2_10_10_10_FLOAT (10:10:10 7e3 floats + 2-bit alpha),
// 3=RGB10A2Unorm (raw 2:10:10:10 unorm).
void MetalRenderTargetCache::PerformColorConvertTransfer(
    const void* command_buffer, RenderTarget* source_rt, RenderTarget* dest_rt,
    const Transfer& transfer) {
  auto* source = static_cast<MetalRenderTarget*>(source_rt);
  auto* dest = static_cast<MetalRenderTarget*>(dest_rt);
  auto fmt_class = [](uint64_t pf) -> int32_t {
    switch ((MTLPixelFormat)pf) {
      case MTLPixelFormatRGBA8Unorm:
        return 0;
      case MTLPixelFormatR32Float:
        return 1;
      case MTLPixelFormatRGBA16Float:
        return 2;
      case MTLPixelFormatRGB10A2Unorm:
        return 3;
      default:
        return -1;
    }
  };
  int32_t src_fmt = fmt_class(source->pixel_format());
  int32_t dst_fmt = fmt_class(dest->pixel_format());
  if (src_fmt < 0 || dst_fmt < 0) {
    static uint32_t s_conv_skips = 0;
    if (s_conv_skips++ < 8) {
      XELOGW("MetalRenderTargetCache: conv transfer unsupported format {} -> {}",
             source->pixel_format(), dest->pixel_format());
    }
    return;
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  if (!transfer_conv_library_) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct ConvParams {
        uint src_base; uint src_pitch; uint dst_base; uint dst_pitch;
        uint start_tiles; uint end_tiles; uint src_fmt; uint shifts;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_conv_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      // Xenos 7e3 (7-bit mantissa, 3-bit exponent, bias -124 vs f32): the
      // packed channel format of k_2_10_10_10_FLOAT EDRAM words.
      static float xe_7e3_to_f32(uint v) {
        v &= 0x3FFu;
        if (v == 0u) { return 0.0; }
        uint m = v & 0x7Fu;
        int e = int(v >> 7);
        if (e == 0) {
          // Denormal: normalize into the f32 encoding.
          int lz = clz(m) - 24;  // m in [1,0x7F] -> lz in [1,7]
          e = 1 - lz;
          m = (m << uint(lz)) & 0x7Fu;
        }
        return as_type<float>(uint(e + 124) << 23 | (m << 16));
      }
      static uint xe_f32_to_7e3(float f) {
        if (!(f > 0.0)) { return 0u; }
        if (f >= 31.875) { return 0x3FFu; }
        uint u = as_type<uint>(f);
        int e = int((u >> 23) & 0xFFu) - 124;
        uint m = (u >> 16) & 0x7Fu;
        uint round = (u >> 15) & 1u;
        if (e <= 0) {
          uint shift = uint(1 - e);
          if (shift > 8u) { return 0u; }
          uint full = 0x80u | m;
          uint r = (full >> shift) + ((full >> (shift - 1u)) & 1u);
          return min(r, 0x7Fu);
        }
        m += round;
        if (m > 0x7Fu) {
          m = 0u;
          ++e;
          if (e >= 8) { return 0x3FFu; }
        }
        return (uint(e) << 7) | m;
      }
      static uint xe_encode(float4 c, uint fmt) {
        switch (fmt) {
          default:
          case 0u:  // RGBA8: storage bytes ARE the guest word.
            return uint(round(saturate(c.r) * 255.0)) |
                   (uint(round(saturate(c.g) * 255.0)) << 8) |
                   (uint(round(saturate(c.b) * 255.0)) << 16) |
                   (uint(round(saturate(c.a) * 255.0)) << 24);
          case 1u:  // R32F: bits are the word.
            return as_type<uint>(c.r);
          case 2u:  // RGBA16F container of 7e3 floats + 2-bit alpha.
            return xe_f32_to_7e3(c.r) | (xe_f32_to_7e3(c.g) << 10) |
                   (xe_f32_to_7e3(c.b) << 20) |
                   (uint(round(saturate(c.a) * 3.0)) << 30);
          case 3u:  // RGB10A2 unorm raw.
            return uint(round(saturate(c.r) * 1023.0)) |
                   (uint(round(saturate(c.g) * 1023.0)) << 10) |
                   (uint(round(saturate(c.b) * 1023.0)) << 20) |
                   (uint(round(saturate(c.a) * 3.0)) << 30);
        }
      }
      static float4 xe_decode(uint w, uint fmt) {
        switch (fmt) {
          default:
          case 0u:
            return float4(float(w & 0xFFu), float((w >> 8) & 0xFFu),
                          float((w >> 16) & 0xFFu), float(w >> 24)) /
                   255.0;
          case 1u:
            return float4(as_type<float>(w), 0.0, 0.0, 0.0);
          case 2u:
            return float4(xe_7e3_to_f32(w), xe_7e3_to_f32(w >> 10),
                          xe_7e3_to_f32(w >> 20), float(w >> 30) / 3.0);
          case 3u:
            return float4(float(w & 0x3FFu) / 1023.0,
                          float((w >> 10) & 0x3FFu) / 1023.0,
                          float((w >> 20) & 0x3FFu) / 1023.0,
                          float(w >> 30) / 3.0);
        }
      }
      fragment float4 xe_conv_fs(VOut in [[stage_in]],
                                 texture2d<float> src [[texture(0)]],
                                 constant ConvParams& p [[buffer(0)]],
                                 constant uint& dst_fmt [[buffer(1)]]) {
        uint2 dp = uint2(in.pos.xy);
        // shifts: dst_x | dst_y<<4 | src_x<<8 | src_y<<12 (sample->pixel).
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint dl = (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        uint tile = p.dst_base + dl;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        uint2 ss = uint2((sl % p.src_pitch) * 80u + (ds.x % 80u),
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        uint2 sp = uint2(ss.x >> ((p.shifts >> 8) & 15u),
                         ss.y >> ((p.shifts >> 12) & 15u));
        float4 c = src.read(sp, 0);
        return xe_decode(xe_encode(c, p.src_fmt), dst_fmt);
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: conv transfer library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    transfer_conv_library_ = (__bridge_retained void*)lib;
  }
  uint64_t pipe_key = dest->pixel_format();
  auto it = transfer_conv_pipelines_.find(pipe_key);
  if (it == transfer_conv_pipelines_.end()) {
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)transfer_conv_library_;
    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = [lib newFunctionWithName:@"xe_conv_vs"];
    d.fragmentFunction = [lib newFunctionWithName:@"xe_conv_fs"];
    d.colorAttachments[0].pixelFormat = (MTLPixelFormat)dest->pixel_format();
    NSError* err = nil;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) {
      XELOGE("MetalRenderTargetCache: conv transfer pipeline {} failed: {}",
             dest->pixel_format(),
             err ? err.localizedDescription.UTF8String : "?");
    }
    it = transfer_conv_pipelines_
             .emplace(pipe_key, ps ? (__bridge_retained void*)ps : nullptr)
             .first;
  }
  if (!it->second) {
    return;
  }
  static uint32_t s_conv_logs = 0;
  if (s_conv_logs++ < 64) {
    XELOGW("MetalXferConv @draw {}: {} -> {} tiles [{},{}) fmt {}->{}",
           command_processor_.trace_draw_index(),
           source->key().GetDebugName(), dest->key().GetDebugName(),
           transfer.start_tiles, transfer.end_tiles, src_fmt, dst_fmt);
  }
  uint32_t sxs, sys, dxs, dys;
  GetMsaaShifts(uint32_t(source->key().msaa_samples), sxs, sys);
  GetMsaaShifts(uint32_t(dest->key().msaa_samples), dxs, dys);
  struct ConvParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, src_fmt, shifts;
  } params{source->key().base_tiles, source->key().GetPitchTiles(),
           dest->key().base_tiles,   dest->key().GetPitchTiles(),
           transfer.start_tiles,     transfer.end_tiles,
           uint32_t(src_fmt),
           dxs | (dys << 4) | (sxs << 8) | (sys << 12)};
  uint32_t dst_fmt_u = uint32_t(dst_fmt);
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest->texture();
  pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  enc.label = @"rtcache:transfer-convert";
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)it->second];
  [enc setFragmentTexture:(__bridge id<MTLTexture>)source->texture() atIndex:0];
  [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
  [enc setFragmentBytes:&dst_fmt_u length:sizeof(dst_fmt_u) atIndex:1];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [enc endEncoding];
}

// Lazily creates the 8 per-bit stencil-write states shared by the c2d and x2x
// stencil rebuild passes: func ALWAYS, REPLACE on pass (the fragment shader
// discards where the source bit is clear), write mask = that bit.
void MetalRenderTargetCache::EnsureStencilBitStates() {
  if (stencil_bit_states_[0]) {
    return;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  for (uint32_t bit = 0; bit < 8; ++bit) {
    MTLDepthStencilDescriptor* bds = [[MTLDepthStencilDescriptor alloc] init];
    bds.depthCompareFunction = MTLCompareFunctionAlways;
    bds.depthWriteEnabled = NO;
    MTLStencilDescriptor* bsten = [[MTLStencilDescriptor alloc] init];
    bsten.stencilCompareFunction = MTLCompareFunctionAlways;
    bsten.depthStencilPassOperation = MTLStencilOperationReplace;
    bsten.stencilFailureOperation = MTLStencilOperationKeep;
    bsten.depthFailureOperation = MTLStencilOperationReplace;
    bsten.writeMask = 1u << bit;
    bds.frontFaceStencil = bsten;
    bds.backFaceStencil = bsten;
    stencil_bit_states_[bit] =
        (__bridge_retained void*)[device newDepthStencilStateWithDescriptor:bds];
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
        uint start_tiles; uint end_tiles; uint is_float24; uint shifts;
        uint stencil_bit; uint pad0; uint pad1; uint pad2;
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
        // shifts: dst_x | dst_y<<4 | src_x<<8 | src_y<<12 (sample->pixel).
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint dl = (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        uint tile = p.dst_base + dl;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        // Depth EDRAM addressing swaps the 40-sample column halves of each
        // tile (see the reference dump shader): the raw word for host sample
        // x lives at x^40-ish (self-inverse +-40).
        uint dtx = ds.x % 80u;
        uint stx = dtx < 40u ? dtx + 40u : dtx - 40u;
        uint2 ss = uint2((sl % p.src_pitch) * 80u + stx,
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        uint2 sp = uint2(ss.x >> ((p.shifts >> 8) & 15u),
                         ss.y >> ((p.shifts >> 12) & 15u));
        float4 c = src.read(sp, 0);
        // Canonical channel order: logical .r holds guest byte0 (RGBA8
        // storage == guest EDRAM bytes); depth in bits [8,32).
        uint bits = uint(round(c.r * 255.0)) | (uint(round(c.g * 255.0)) << 8) |
                    (uint(round(c.b * 255.0)) << 16) |
                    (uint(round(c.a * 255.0)) << 24);
        uint d24 = bits >> 8;
        DepthOut o;
        // Canonical unorm24 decode (pixel_formats.xesli XeUNorm24To32):
        // exact exponent-shift division by 2^24 with the >>23 bias trick;
        // /16777215.0 is off by an ULP for common values like 0xC00000.
        o.depth = p.is_float24 != 0u
                      ? xe_f20e4_to_f32(d24)
                      : float(d24 + (d24 >> 23)) * (1.0 / 16777216.0);
        return o;
      }
      // Stencil bit-plane pass: survives (and lets the stencil REPLACE op set
      // writemask's bit) only where the source word's stencil byte has
      // p.stencil_bit set. No depth/color output; Metal has no shader stencil
      // export, so the byte transfers as 8 masked REPLACE passes.
      fragment void xe_xfer_c2d_sten_fs(VOut in [[stage_in]],
                                        texture2d<float> src [[texture(0)]],
                                        constant XferParams& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint dl = (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        uint tile = p.dst_base + dl;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        uint dtx = ds.x % 80u;
        uint stx = dtx < 40u ? dtx + 40u : dtx - 40u;
        uint2 ss = uint2((sl % p.src_pitch) * 80u + stx,
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        uint2 sp = uint2(ss.x >> ((p.shifts >> 8) & 15u),
                         ss.y >> ((p.shifts >> 12) & 15u));
        float4 c = src.read(sp, 0);
        uint stencil = uint(round(c.r * 255.0));  // guest byte0 = stencil.
        if ((stencil & p.stencil_bit) == 0u) {
          discard_fragment();
        }
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
    d.stencilAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) {
      XELOGE("MetalRenderTargetCache: c2d transfer pipeline failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    transfer_c2d_pipeline_ = (__bridge_retained void*)ps;
    // Depth pass state: also zeroes the transferred rect's stencil (REPLACE
    // ref=0, full write mask) as the base the bit passes OR into.
    MTLDepthStencilDescriptor* ds = [[MTLDepthStencilDescriptor alloc] init];
    ds.depthCompareFunction = MTLCompareFunctionAlways;
    ds.depthWriteEnabled = YES;
    MTLStencilDescriptor* zero_sten = [[MTLStencilDescriptor alloc] init];
    zero_sten.stencilCompareFunction = MTLCompareFunctionAlways;
    zero_sten.depthStencilPassOperation = MTLStencilOperationReplace;
    zero_sten.stencilFailureOperation = MTLStencilOperationReplace;
    zero_sten.depthFailureOperation = MTLStencilOperationReplace;
    zero_sten.writeMask = 0xFF;
    ds.frontFaceStencil = zero_sten;
    ds.backFaceStencil = zero_sten;
    transfer_c2d_depth_state_ =
        (__bridge_retained void*)[device newDepthStencilStateWithDescriptor:ds];
    // Stencil bit-plane pipeline + one state per bit.
    MTLRenderPipelineDescriptor* sd =
        [[MTLRenderPipelineDescriptor alloc] init];
    sd.vertexFunction = [lib newFunctionWithName:@"xe_xfer_vs"];
    sd.fragmentFunction = [lib newFunctionWithName:@"xe_xfer_c2d_sten_fs"];
    sd.depthAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
    sd.stencilAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
    id<MTLRenderPipelineState> sps =
        [device newRenderPipelineStateWithDescriptor:sd error:&err];
    if (!sps) {
      XELOGE("MetalRenderTargetCache: c2d stencil pipeline failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    transfer_c2d_sten_pipeline_ = (__bridge_retained void*)sps;
    EnsureStencilBitStates();
  }
  if (!transfer_c2d_pipeline_ || !transfer_c2d_depth_state_ ||
      !transfer_c2d_sten_pipeline_) {
    return;
  }
  uint32_t sxs, sys, dxs, dys;
  GetMsaaShifts(uint32_t(source->key().msaa_samples), sxs, sys);
  GetMsaaShifts(uint32_t(dest->key().msaa_samples), dxs, dys);
  struct XferParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, is_float24, shifts;
    uint32_t stencil_bit, pad0, pad1, pad2;
  } params{source->key().base_tiles,     source->key().GetPitchTiles(),
           dest->key().base_tiles,       dest->key().GetPitchTiles(),
           transfer.start_tiles,         transfer.end_tiles,
           uint32_t(dest->key().GetDepthFormat() ==
                    xenos::DepthRenderTargetFormat::kD24FS8),
           dxs | (dys << 4) | (sxs << 8) | (sys << 12),
           0, 0, 0, 0};
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.depthAttachment.texture = (__bridge id<MTLTexture>)dest->texture();
  pass.depthAttachment.loadAction = MTLLoadActionLoad;
  pass.depthAttachment.storeAction = MTLStoreActionStore;
  pass.stencilAttachment.texture = (__bridge id<MTLTexture>)dest->texture();
  pass.stencilAttachment.loadAction = MTLLoadActionLoad;
  pass.stencilAttachment.storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                  transfer_c2d_pipeline_];
  enc.label = @"rtcache:transfer-c2d";
  [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)
                                transfer_c2d_depth_state_];
  [enc setStencilReferenceValue:0];  // depth pass zeroes the rect's stencil.
  [enc setFragmentTexture:(__bridge id<MTLTexture>)source->texture() atIndex:0];
  [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  // Rebuild the guest stencil byte from the source color's byte0, one masked
  // REPLACE pass per bit (no shader stencil export in Metal).
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                  transfer_c2d_sten_pipeline_];
  [enc setStencilReferenceValue:0xFF];
  for (uint32_t bit = 0; bit < 8; ++bit) {
    if (!stencil_bit_states_[bit]) {
      break;
    }
    params.stencil_bit = 1u << bit;
    [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)
                                  stencil_bit_states_[bit]];
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  }
  [enc endEncoding];
}

void MetalRenderTargetCache::PerformDepthToColorTransfer(
    const void* command_buffer, RenderTarget* source_rt, RenderTarget* dest_rt,
    const Transfer& transfer, bool real_bits) {
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
        uint start_tiles; uint end_tiles; uint is_float24; uint shifts;
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
                                     texture2d<uint> sten [[texture(1)]],
                                     constant XferParams& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint dl = (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        uint tile = p.dst_base + dl;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        // Inverse of the depth EDRAM 40-column swap: this raw-word position
        // holds the host depth sample from the other 40-column half.
        uint dtx = ds.x % 80u;
        uint stx = dtx < 40u ? dtx + 40u : dtx - 40u;
        uint2 ss = uint2((sl % p.src_pitch) * 80u + stx,
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        uint2 sp = uint2(ss.x >> ((p.shifts >> 8) & 15u),
                         ss.y >> ((p.shifts >> 12) & 15u));
        float d = src.read(sp);
        // rint = roundEven: the canonical unorm24 encode (the Vulkan dump
        // shader uses GLSL RoundEven; round-half-away differs at exact .5).
        uint d24 = p.is_float24 != 0u
                       ? xe_f32_to_f20e4(d)
                       : uint(rint(saturate(d) * 16777215.0));
        // DEFAULT: zero words. Rationale: a color pass claiming depth-owned
        // tiles only ever CONSUMES this content when the real producer draws
        // were lost (trace replay's stale-VB degenerate quads) — live, the
        // guest overwrites the tiles before reading. The stable Vulkan
        // reference zero-fills this case, and real f20e4 depth bytes here
        // read as saturated blue fog in JC2's composite (81.9 vs 49.9 MAE).
        // p.shifts bit 17 = probe: encode the REAL depth bits (d24<<8, with
        // bit 16 setting stencil FF instead of 0).
        // Bit 18 (0x40000): transport the REAL stencil plane — in Sonic's
        // packed vector map the stencil byte is part of each vector's value
        // (packed formats center at 128; a zero byte decodes as a large
        // negative component = the horizontal echo drag).
        uint stencil = (p.shifts & 0x40000u) != 0u
                           ? (sten.read(sp).r & 0xFFu)
                           : ((p.shifts & 0x10000u) != 0u ? 0xFFu : 0u);
        uint bits = (p.shifts & 0x20000u) != 0u ? ((d24 << 8) | stencil) : 0u;
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
  uint32_t sxs, sys, dxs, dys;
  GetMsaaShifts(uint32_t(source->key().msaa_samples), sxs, sys);
  GetMsaaShifts(uint32_t(dest->key().msaa_samples), dxs, dys);
  // A/B probes: XE_METAL_D2C_REAL=1 encodes the real depth bits instead of
  // the default zero words (see the shader comment); XE_METAL_D2C_STENCILFF=1
  // additionally writes 0xFF in the stencil byte (hardware-capture-like;
  // measured WORSE vs Vulkan truth).
  static const uint32_t s_stencil0_bit =
      (std::getenv("XE_METAL_D2C_STENCILFF") ? 0x10000u : 0u) |
      (std::getenv("XE_METAL_D2C_REAL") ? 0x20000u : 0u);
  // Ping-pong bit transport: the caller proved this transfer returns tiles
  // the dest color key lent THIS frame — the guest wrote real payload bits
  // through the depth plane (Sonic's vector map). Zero words here decode as
  // maximum-magnitude vectors (packed formats center at 0x8000): the
  // full-screen echo/streak haze.
  const uint32_t encode_bits =
      s_stencil0_bit | (real_bits ? (0x20000u | 0x40000u) : 0u);
  struct XferParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, is_float24, shifts;
  } params{source->key().base_tiles,     source->key().GetPitchTiles(),
           dest->key().base_tiles,       dest->key().GetPitchTiles(),
           transfer.start_tiles,         transfer.end_tiles,
           uint32_t(source->key().GetDepthFormat() ==
                    xenos::DepthRenderTargetFormat::kD24FS8),
           dxs | (dys << 4) | (sxs << 8) | (sys << 12) | encode_bits};
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest->texture();
  pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                  transfer_d2c_pipeline_];
  enc.label = @"rtcache:transfer-d2c";
  id<MTLTexture> src_tex = (__bridge id<MTLTexture>)source->texture();
  [enc setFragmentTexture:src_tex atIndex:0];
  // Stencil plane view (bit 18 real-stencil transport). The shader statically
  // references the texture, so bind it even when the bit is off.
  id<MTLTexture> sten_view =
      [src_tex newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
  [enc setFragmentTexture:sten_view atIndex:1];
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
  // Ownership forensics (rides XE_METAL_PROBE_RESOLVE, any value): the span
  // and EVERY owner rect, not just the dominant — the transferless 2x->1x
  // ownership flip hides in this list.
  static const bool s_probe_owners =
      std::getenv("XE_METAL_PROBE_RESOLVE") != nullptr;
  if (s_probe_owners) {
    XELOGW(
        "MetalOwners @draw {}: span base={} rowlen={} rows={} pitch={} depth={}",
        command_processor_.trace_draw_index(), edram_base, row_length_used,
        rows, pitch_tiles, is_depth);
    for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
      auto* rt = static_cast<MetalRenderTarget*>(rect.render_target);
      XELOGW("MetalOwners   rows[{}+{}] owner={}", rect.row_first, rect.rows,
             rt ? rt->key().GetDebugName().c_str() : "(null)");
    }
  }
  // The span can be owned piecewise; pick the DOMINANT owner (most tiles) —
  // e.g. a 64-tile quarter-scene resolve whose first 4 tiles belong to a tiny
  // R32Float exposure RT must not take its snapshot format from the sliver.
  MetalRenderTarget* best = nullptr;
  uint32_t best_tiles = 0;
  for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
    auto* rt = static_cast<MetalRenderTarget*>(rect.render_target);
    if (!rt || rt->key().is_depth != uint32_t(is_depth)) {
      continue;
    }
    uint32_t tiles = rect.rows * row_length_used;
    if (tiles > best_tiles) {
      best_tiles = tiles;
      best = rt;
    }
  }
  if (!best) {
    return nullptr;
  }
  if (pixel_format_out) {
    *pixel_format_out = best->pixel_format();
  }
  return best->texture();
}

void MetalRenderTargetCache::PerformColorReinterpretTransfer(
    const void* command_buffer, RenderTarget* source_rt, RenderTarget* dest_rt,
    const Transfer& transfer) {
  auto* source = static_cast<MetalRenderTarget*>(source_rt);
  auto* dest = static_cast<MetalRenderTarget*>(dest_rt);
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  bool to_float = dest->pixel_format() == uint64_t(MTLPixelFormatR32Float);
  if (!transfer_reint_c2f_pipeline_ || !transfer_reint_f2c_pipeline_) {
    // Same tile arithmetic as the c2d/d2c reinterprets, but color tiles have
    // no 40-sample column swap; the EDRAM 32-bit word itself is preserved.
    // Byte order matches the c2d reconstruction: word = b|g<<8|r<<16|a<<24.
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct XferParams {
        uint src_base; uint src_pitch; uint dst_base; uint dst_pitch;
        uint start_tiles; uint end_tiles; uint unused; uint shifts;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_xfer_reint_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      static uint2 xe_reint_src_pixel(uint2 dp, constant XferParams& p,
                                      thread bool& in_range) {
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint dl = (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        uint tile = p.dst_base + dl;
        in_range = tile >= p.start_tiles && tile < p.end_tiles;
        uint sl = tile - p.src_base;
        uint2 ss = uint2((sl % p.src_pitch) * 80u + (ds.x % 80u),
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        return uint2(ss.x >> ((p.shifts >> 8) & 15u),
                     ss.y >> ((p.shifts >> 12) & 15u));
      }
      // BGRA8 raw bytes -> the same EDRAM word as f32 bits.
      fragment float4 xe_xfer_c2f_fs(VOut in [[stage_in]],
                                     texture2d<float> src [[texture(0)]],
                                     constant XferParams& p [[buffer(0)]]) {
        bool in_range;
        uint2 sp = xe_reint_src_pixel(uint2(in.pos.xy), p, in_range);
        if (!in_range || sp.x >= src.get_width() || sp.y >= src.get_height()) {
          discard_fragment();
        }
        float4 c = src.read(sp, 0);
        // Canonical: logical .r = guest byte0 (RGBA8 storage).
        uint bits = uint(round(c.r * 255.0)) | (uint(round(c.g * 255.0)) << 8) |
                    (uint(round(c.b * 255.0)) << 16) |
                    (uint(round(c.a * 255.0)) << 24);
        return float4(as_type<float>(bits), 0.0, 0.0, 0.0);
      }
      // f32 bits -> the same EDRAM word as BGRA8 raw bytes.
      fragment float4 xe_xfer_f2c_fs(VOut in [[stage_in]],
                                     texture2d<float> src [[texture(0)]],
                                     constant XferParams& p [[buffer(0)]]) {
        bool in_range;
        uint2 sp = xe_reint_src_pixel(uint2(in.pos.xy), p, in_range);
        if (!in_range || sp.x >= src.get_width() || sp.y >= src.get_height()) {
          discard_fragment();
        }
        uint bits = as_type<uint>(src.read(sp, 0).x);
        // Canonical: guest byte0 -> logical .r (matches d2c).
        return float4(float(bits & 0xFFu), float((bits >> 8) & 0xFFu),
                      float((bits >> 16) & 0xFFu), float(bits >> 24)) / 255.0;
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: reinterpret transfer library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    auto make_pipeline = [&](NSString* fs, MTLPixelFormat rt_format) -> void* {
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_xfer_reint_vs"];
      d.fragmentFunction = [lib newFunctionWithName:fs];
      d.colorAttachments[0].pixelFormat = rt_format;
      NSError* perr = nil;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&perr];
      if (!ps) {
        XELOGE("MetalRenderTargetCache: reinterpret pipeline failed: {}",
               perr ? perr.localizedDescription.UTF8String : "?");
        return nullptr;
      }
      return (__bridge_retained void*)ps;
    };
    transfer_reint_c2f_pipeline_ =
        make_pipeline(@"xe_xfer_c2f_fs", MTLPixelFormatR32Float);
    transfer_reint_f2c_pipeline_ =
        make_pipeline(@"xe_xfer_f2c_fs", (MTLPixelFormat)kColorFormat);
  }
  void* pipeline =
      to_float ? transfer_reint_c2f_pipeline_ : transfer_reint_f2c_pipeline_;
  if (!pipeline) {
    return;
  }
  uint32_t sxs, sys, dxs, dys;
  GetMsaaShifts(uint32_t(source->key().msaa_samples), sxs, sys);
  GetMsaaShifts(uint32_t(dest->key().msaa_samples), dxs, dys);
  struct XferParams {
    uint32_t src_base, src_pitch, dst_base, dst_pitch;
    uint32_t start_tiles, end_tiles, unused, shifts;
  } params{source->key().base_tiles, source->key().GetPitchTiles(),
           dest->key().base_tiles,   dest->key().GetPitchTiles(),
           transfer.start_tiles,     transfer.end_tiles,
           0,                        dxs | (dys << 4) | (sxs << 8) | (sys << 12)};
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest->texture();
  pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> enc =
      [cb renderCommandEncoderWithDescriptor:pass];
  enc.label = @"rtcache:transfer-reinterpret";
  [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pipeline];
  [enc setFragmentTexture:(__bridge id<MTLTexture>)source->texture() atIndex:0];
  [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [enc endEncoding];
}

bool MetalRenderTargetCache::ResolveDepthToSnapshot(
    const void* command_buffer, uint32_t span_base, uint32_t row_length_used,
    uint32_t rows, uint32_t span_pitch, uint32_t span_msaa_samples,
    bool is_float24, void* dest_snapshot, uint32_t width, uint32_t height,
    uint32_t dest_y_offset) {
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
        uint y_offset; uint is_float24; uint shifts; uint pad1;
      };
      // shifts: src_x | src_y<<4 | span_x<<8 | span_y<<12 (sample->pixel).
      static uint2 xe_span_sample(uint2 dp, uint y_offset, uint shifts) {
        return uint2(dp.x << ((shifts >> 8) & 15u),
                     (dp.y - y_offset) << ((shifts >> 12) & 15u));
      }
      static uint2 xe_src_pixel(uint sl, uint src_pitch, uint2 ds,
                                uint shifts) {
        uint2 ss = uint2((sl % src_pitch) * 80u + (ds.x % 80u),
                         (sl / src_pitch) * 16u + (ds.y % 16u));
        return uint2(ss.x >> (shifts & 15u), ss.y >> ((shifts >> 4) & 15u));
      }
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
        uint2 ds = xe_span_sample(dp, p.y_offset, p.shifts);
        uint tile = p.span_base + (ds.y / 16u) * p.span_pitch + (ds.x / 80u);
        uint sl = tile - p.src_base;
        // Depth EDRAM 40-column swap: the word for this span sample is in
        // the other 40-column half of the tile.
        uint2 dsw = uint2((ds.x / 80u) * 80u +
                              ((ds.x % 80u) < 40u ? (ds.x % 80u) + 40u
                                                  : (ds.x % 80u) - 40u),
                          ds.y);
        uint2 sp = xe_src_pixel(sl, p.src_pitch, dsw, p.shifts);
        float4 c = src.read(sp, 0);
        uint bits = uint(round(c.b * 255.0)) | (uint(round(c.g * 255.0)) << 8) |
                    (uint(round(c.r * 255.0)) << 16) |
                    (uint(round(c.a * 255.0)) << 24);
        uint d24 = bits >> 8;
        // Canonical unorm24 decode (see XeUNorm24To32).
        return p.is_float24 != 0u
                   ? xe_f20e4_to_f32(d24)
                   : float(d24 + (d24 >> 23)) * (1.0 / 16777216.0);
      }
      fragment float xe_d2f_fs(VOut in [[stage_in]],
                               depth2d<float> src [[texture(0)]],
                               constant Params& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint2 ds = xe_span_sample(dp, p.y_offset, p.shifts);
        uint tile = p.span_base + (ds.y / 16u) * p.span_pitch + (ds.x / 80u);
        uint sl = tile - p.src_base;
        uint2 sp = xe_src_pixel(sl, p.src_pitch, ds, p.shifts);
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
  enc.label = @"rtcache:resolve-depth-snapshot";
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
    // The rect's tile rows within the span -> destination pixel rows (span
    // sample rows >> span y-shift).
    uint32_t oxs, oys, pxs, pys;
    GetMsaaShifts(uint32_t(owner->key().msaa_samples), oxs, oys);
    // Owner shifts govern both sides (aliased single-sample textures; see the
    // color resolve).
    pxs = oxs;
    pys = oys;
    (void)span_msaa_samples;
    uint32_t y0 = (rect.row_first * 16) >> pys;
    if (y0 >= height) {
      continue;
    }
    uint32_t h = std::min((rect.rows * 16) >> pys, height - y0);
    struct Params {
      uint32_t src_base, src_pitch, span_base, span_pitch;
      uint32_t y_offset, is_float24, shifts, pad1;
    } params{owner->key().base_tiles, owner->key().GetPitchTiles(),
             span_base,               span_pitch,
             dest_y_offset,           uint32_t(is_float24),
             oxs | (oys << 4) | (pxs << 8) | (pys << 12), 0};
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

bool MetalRenderTargetCache::ResolveColorToSnapshot(
    const void* command_buffer, uint32_t span_base, uint32_t row_length_used,
    uint32_t rows, uint32_t span_pitch, uint32_t span_msaa_samples,
    void* dest_snapshot, uint64_t dest_pixel_format, uint32_t width,
    uint32_t height, uint32_t dest_y_offset, float exp_bias_scale) {
  if (!command_buffer || !dest_snapshot) {
    return false;
  }
  resolve_dump_rectangles_.clear();
  GetResolveCopyRectanglesToDump(span_base, row_length_used, rows, span_pitch,
                                 resolve_dump_rectangles_);
  if (resolve_dump_rectangles_.empty()) {
    return false;
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  id<MTLTexture> dest_tex = (__bridge id<MTLTexture>)dest_snapshot;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  // One sampling pass per owner rect instead of per-tile blits: hundreds of
  // copyFromTexture calls each created a driver texture view — the #2 cost
  // center in gameplay profiles (after the shared-memory memcmp).
  if (!resolve_copy_library_) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct Params {
        uint src_base; uint src_pitch; uint span_base; uint span_pitch;
        uint start_tiles; uint end_tiles; uint y_offset; uint shifts;
        uint tile_w; float exp_bias_scale; uint pad2; uint pad3;
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_rcp_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      // shifts: dst_x | dst_y<<4 | src_x<<8 | src_y<<12 (sample->pixel).
      fragment float4 xe_rcp_fs(VOut in [[stage_in]],
                                texture2d<float> src [[texture(0)]],
                                constant Params& p [[buffer(0)]]) {
        uint2 dp = uint2(in.pos.xy);
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         (dp.y - p.y_offset) << ((p.shifts >> 4) & 15u));
        uint col = ds.x / p.tile_w;
        // Past the span's tile row width: would alias into the NEXT tile row.
        if (col >= p.span_pitch) {
          discard_fragment();
        }
        uint tile = p.span_base + (ds.y / 16u) * p.span_pitch + col;
        if (tile < p.start_tiles || tile >= p.end_tiles) {
          discard_fragment();
        }
        uint sl = tile - p.src_base;
        uint2 ss = uint2((sl % p.src_pitch) * p.tile_w + (ds.x % p.tile_w),
                         (sl / p.src_pitch) * 16u + (ds.y % 16u));
        uint2 sp = uint2(ss.x >> ((p.shifts >> 8) & 15u),
                         ss.y >> ((p.shifts >> 12) & 15u));
        // Owner texture bounds (partial edge tiles).
        if (sp.x >= src.get_width() || sp.y >= src.get_height()) {
          discard_fragment();
        }
        return src.read(sp, 0) * p.exp_bias_scale;
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: resolve-copy library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return false;
    }
    resolve_copy_library_ = (__bridge_retained void*)lib;
  }
  id<MTLRenderPipelineState> pipeline = nil;
  {
    auto it = resolve_copy_pipelines_.find(dest_pixel_format);
    if (it == resolve_copy_pipelines_.end()) {
      id<MTLLibrary> lib = (__bridge id<MTLLibrary>)resolve_copy_library_;
      NSError* err = nil;
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_rcp_vs"];
      d.fragmentFunction = [lib newFunctionWithName:@"xe_rcp_fs"];
      d.colorAttachments[0].pixelFormat = (MTLPixelFormat)dest_pixel_format;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&err];
      if (!ps) {
        XELOGE("MetalRenderTargetCache: resolve-copy pipeline {} failed: {}",
               dest_pixel_format,
               err ? err.localizedDescription.UTF8String : "?");
      }
      it = resolve_copy_pipelines_
               .emplace(dest_pixel_format,
                        ps ? (__bridge_retained void*)ps : nullptr)
               .first;
    }
    pipeline = (__bridge id<MTLRenderPipelineState>)it->second;
  }
  if (!pipeline) {
    return false;
  }
  bool any = false;
  for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
    auto* owner = static_cast<MetalRenderTarget*>(rect.render_target);
    if (!owner || owner->key().is_depth) {
      continue;
    }
    // Metal performs the numeric color conversion when the sampled float4 is
    // written to the destination attachment. These are the host containers
    // used by Xenos color RTs and resolve destinations; in particular, Red
    // Dead resolves RGBA16F HDR EDRAM into its RGBA8 frontbuffer here.
    auto numeric_color_format = [](uint64_t format) {
      return format == kColorFormat ||
             format == uint64_t(MTLPixelFormatR32Float) ||
             format == uint64_t(MTLPixelFormatRG16Float) ||
             format == uint64_t(MTLPixelFormatRGBA16Float) ||
             format == uint64_t(MTLPixelFormatRGB10A2Unorm);
    };
    if (!numeric_color_format(owner->pixel_format()) ||
        !numeric_color_format(dest_pixel_format)) {
      static uint32_t s_skip_convert_logs = 0;
      if (s_skip_convert_logs++ < 8) {
        XELOGW(
            "MetalRenderTargetCache: SKIP color resolve owner fmt {} -> dest "
            "fmt {} (unsupported conversion)",
            owner->pixel_format(), dest_pixel_format);
      }
      continue;
    }
    uint32_t owner_base = owner->key().base_tiles;
    uint32_t owner_pitch = owner->key().GetPitchTiles();
    uint32_t tile_w = owner->key().Is64bpp() ? 40 : 80;
    // Owner shifts govern BOTH sides of the mapping: our textures are aliased
    // single-sample, so an MSAA surface already stores one texel per resolved
    // pixel — a guest 2x->1x resolve is a 1:1 pixel copy in our model. (The
    // declared span msaa can differ from the owner's; the physical layout is
    // the owner's.)
    uint32_t oxs, oys;
    GetMsaaShifts(uint32_t(owner->key().msaa_samples), oxs, oys);
    (void)span_msaa_samples;
    static uint32_t s_owner_logs = 0;
    if (s_owner_logs++ < 12) {
      XELOGI("MetalRenderTargetCache: color resolve owner {} rt={:X} rows[{}+{}]",
             owner->key().GetDebugName(), uintptr_t(owner->texture()),
             rect.row_first, rect.rows);
    }
    struct Params {
      uint32_t src_base, src_pitch, span_base, span_pitch;
      uint32_t start_tiles, end_tiles, y_offset, shifts;
      uint32_t tile_w;
      float exp_bias_scale;
      uint32_t pad2, pad3;
    } params{owner_base,
             owner_pitch,
             span_base,
             span_pitch,
             span_base + rect.row_first * span_pitch,
             span_base + (rect.row_first + rect.rows) * span_pitch,
             dest_y_offset,
             oxs | (oys << 4) | (oxs << 8) | (oys << 12),
             tile_w,
             exp_bias_scale,
             0,
             0};
    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = dest_tex;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:pass];
    enc.label = @"rtcache:resolve-color-snapshot";
    [enc setRenderPipelineState:pipeline];
    // Scissor to this rect's destination rows (the tile-range discard handles
    // exact bounds; the scissor keeps other strips/padding untouched).
    uint32_t sy0 = dest_y_offset + ((rect.row_first * 16) >> oys);
    uint32_t sy1 = dest_y_offset + (((rect.row_first + rect.rows) * 16) >> oys);
    sy0 = std::min<uint32_t>(sy0, uint32_t(dest_tex.height));
    sy1 = std::min<uint32_t>(sy1, uint32_t(dest_tex.height));
    if (sy1 <= sy0) {
      [enc endEncoding];
      continue;
    }
    MTLScissorRect scissor;
    scissor.x = 0;
    scissor.width = dest_tex.width;
    scissor.y = sy0;
    scissor.height = sy1 - sy0;
    [enc setScissorRect:scissor];
    [enc setFragmentTexture:(__bridge id<MTLTexture>)owner->texture()
                    atIndex:0];
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    any = true;
  }
  return any;
}

void MetalRenderTargetCache::PerformResolveClear(
    const void* command_buffer, uint32_t span_base, uint32_t row_length_used,
    uint32_t rows, uint32_t span_pitch, bool is_depth, uint32_t clear_value) {
  if (!command_buffer) {
    return;
  }
  resolve_dump_rectangles_.clear();
  GetResolveCopyRectanglesToDump(span_base, row_length_used, rows, span_pitch,
                                 resolve_dump_rectangles_);
  if (resolve_dump_rectangles_.empty()) {
    return;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  if (!clear_color_pipeline_) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct ClearParams {
        uint dst_base; uint dst_pitch; uint start_tiles; uint end_tiles;
        uint shifts; uint pad0; uint pad1; uint pad2;
        float4 color;  // or .x = depth.
      };
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_clear_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      static bool xe_clear_out_of_span(uint2 dp, constant ClearParams& p) {
        uint2 ds = uint2(dp.x << (p.shifts & 15u),
                         dp.y << ((p.shifts >> 4) & 15u));
        uint tile = p.dst_base + (ds.y / 16u) * p.dst_pitch + (ds.x / 80u);
        return tile < p.start_tiles || tile >= p.end_tiles;
      }
      fragment float4 xe_clear_color_fs(VOut in [[stage_in]],
                                        constant ClearParams& p [[buffer(0)]]) {
        if (xe_clear_out_of_span(uint2(in.pos.xy), p)) discard_fragment();
        return p.color;
      }
      struct DepthOut { float depth [[depth(any)]]; };
      fragment DepthOut xe_clear_depth_fs(VOut in [[stage_in]],
                                          constant ClearParams& p
                                          [[buffer(0)]]) {
        if (xe_clear_out_of_span(uint2(in.pos.xy), p)) discard_fragment();
        DepthOut o;
        o.depth = p.color.x;
        return o;
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: clear library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return;
    }
    {
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_clear_vs"];
      d.fragmentFunction = [lib newFunctionWithName:@"xe_clear_color_fs"];
      d.colorAttachments[0].pixelFormat = (MTLPixelFormat)kColorFormat;
      NSError* perr = nil;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&perr];
      if (ps) {
        clear_color_pipeline_ = (__bridge_retained void*)ps;
      }
    }
    {
      MTLRenderPipelineDescriptor* d =
          [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_clear_vs"];
      d.fragmentFunction = [lib newFunctionWithName:@"xe_clear_depth_fs"];
      d.depthAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
      d.stencilAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
      NSError* perr = nil;
      id<MTLRenderPipelineState> ps =
          [device newRenderPipelineStateWithDescriptor:d error:&perr];
      if (ps) {
        clear_depth_pipeline_ = (__bridge_retained void*)ps;
      }
      MTLDepthStencilDescriptor* ds = [[MTLDepthStencilDescriptor alloc] init];
      ds.depthCompareFunction = MTLCompareFunctionAlways;
      ds.depthWriteEnabled = YES;
      // The guest clear dword is depth24<<8 | stencil: the same rect clear
      // also REPLACEs stencil with the encoder's reference (clear & 0xFF).
      MTLStencilDescriptor* sten = [[MTLStencilDescriptor alloc] init];
      sten.stencilCompareFunction = MTLCompareFunctionAlways;
      sten.depthStencilPassOperation = MTLStencilOperationReplace;
      sten.stencilFailureOperation = MTLStencilOperationReplace;
      sten.depthFailureOperation = MTLStencilOperationReplace;
      sten.writeMask = 0xFF;
      ds.frontFaceStencil = sten;
      ds.backFaceStencil = sten;
      clear_depth_state_ =
          (__bridge_retained void*)[device
              newDepthStencilStateWithDescriptor:ds];
    }
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
  for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
    auto* owner = static_cast<MetalRenderTarget*>(rect.render_target);
    if (!owner || owner->key().is_depth != uint32_t(is_depth)) {
      continue;
    }
    uint32_t oxs, oys;
    GetMsaaShifts(uint32_t(owner->key().msaa_samples), oxs, oys);
    struct ClearParams {
      uint32_t dst_base, dst_pitch, start_tiles, end_tiles;
      uint32_t shifts, pad0, pad1, pad2;
      float color[4];
    } params{owner->key().base_tiles,
             owner->key().GetPitchTiles(),
             span_base,
             span_base + rows * span_pitch,
             oxs | (oys << 4),
             0,
             0,
             0,
             {0, 0, 0, 0}};
    if (is_depth) {
      if (!clear_depth_pipeline_ || !clear_depth_state_) {
        return;
      }
      // Raw EDRAM depth word: depth24 in the high bits (unorm24; float24
      // clears are typically 0 or max, both map fine through unorm here).
      uint32_t clear_d24 = clear_value >> 8;
      // Canonical unorm24 decode (see XeUNorm24To32).
      params.color[0] =
          float(clear_d24 + (clear_d24 >> 23)) * (1.0f / 16777216.0f);
      MTLRenderPassDescriptor* pass =
          [MTLRenderPassDescriptor renderPassDescriptor];
      pass.depthAttachment.texture = (__bridge id<MTLTexture>)owner->texture();
      pass.depthAttachment.loadAction = MTLLoadActionLoad;
      pass.depthAttachment.storeAction = MTLStoreActionStore;
      pass.stencilAttachment.texture =
          (__bridge id<MTLTexture>)owner->texture();
      pass.stencilAttachment.loadAction = MTLLoadActionLoad;
      pass.stencilAttachment.storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc =
          [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      clear_depth_pipeline_];
      enc.label = @"rtcache:resolve-clear-depth";
      [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)
                                    clear_depth_state_];
      [enc setStencilReferenceValue:clear_value & 0xFF];
      [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
    } else {
      if (!clear_color_pipeline_ || owner->pixel_format() != kColorFormat) {
        continue;
      }
      // Raw 8888 EDRAM word -> canonical logical channels (guest byte0 = .r,
      // RGBA8 storage == guest bytes; matches d2c/c2d/c2f/f2c and the dump).
      params.color[0] = float(clear_value & 0xFF) / 255.0f;          // R
      params.color[1] = float((clear_value >> 8) & 0xFF) / 255.0f;   // G
      params.color[2] = float((clear_value >> 16) & 0xFF) / 255.0f;  // B
      params.color[3] = float(clear_value >> 24) / 255.0f;           // A
      MTLRenderPassDescriptor* pass =
          [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture =
          (__bridge id<MTLTexture>)owner->texture();
      pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc =
          [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      clear_color_pipeline_];
      enc.label = @"rtcache:resolve-clear-color";
      [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
    }
    static uint32_t s_logs = 0;
    if (s_logs++ < 12) {
      XELOGI("MetalRenderTargetCache: resolve clear {} = {:08X} into {}",
             is_depth ? "depth" : "color", clear_value,
             owner->key().GetDebugName());
    }
  }
}

void* MetalRenderTargetCache::EnsureResolveSnapshot(uint32_t dest_base,
                                                    uint32_t width,
                                                    uint32_t height,
                                                    uint64_t pixel_format) {
  // Scene-scale resolves (depth AND color, e.g. the 720p albedo) arrive as
  // vertical strips of ONE 720-row buffer sampled full-height from the first
  // strip's base; allocate full-height so the later strip lands inside
  // (FindContainingSnapshot).
  if (width >= 1280 && height < 720) {
    height = 720;
  }
  // Key by (base, format): JC2 resolves color (BGRA8) and depth (R32F) to
  // the same guest base — a single-slot map destroyed/recreated the snapshot
  // EVERY resolve, and each new texture burned a bindless view slot until
  // the heap ran dry a few minutes into gameplay (black samples, then a
  // frozen present).
  uint64_t snap_key = (uint64_t(dest_base) << 1) |
                      (pixel_format == uint64_t(MTLPixelFormatR32Float) ? 1 : 0);
  auto it = resolve_snapshots_.find(snap_key);
  if (it != resolve_snapshots_.end() && it->second.texture &&
      it->second.width >= width && it->second.height >= height &&
      it->second.pixel_format == pixel_format) {
    return it->second.texture;
  }
  if (it != resolve_snapshots_.end() && it->second.texture) {
    // Do NOT free the old texture now: draws already encoded into the open frame
    // command buffer still reference it by bindless gpuAddress, and freeing it
    // mid-frame GPU-page-faults when they execute (the "GPU froze" hang). Retire
    // it to the deferred list; the command processor frees it after commit+wait.
    deferred_texture_releases_.emplace_back(
        MetalCommandProcessor::S7TagsOff()
            ? 0
            : command_processor_.CurrentEncodingSubmission(),
        it->second.texture);
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
  ClearTextureAtCreation((__bridge void*)texture, /*is_depth=*/false);
  resolve_snapshots_[snap_key] =
      Snapshot{(__bridge_retained void*)texture, width, height, pixel_format};
  return resolve_snapshots_[snap_key].texture;
}

void MetalRenderTargetCache::DrainDeferredReleases(
    uint64_t completed_submission) {
  auto it = std::remove_if(
      deferred_texture_releases_.begin(), deferred_texture_releases_.end(),
      [&](const std::pair<uint64_t, void*>& entry) {
        if (entry.first > completed_submission) {
          return false;
        }
        id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)entry.second;
        (void)released;
        return true;
      });
  deferred_texture_releases_.erase(it, deferred_texture_releases_.end());
}

void* MetalRenderTargetCache::GetResolveSnapshot(uint32_t dest_base) const {
  // Composite key: base<<1 | is_r32f. Callers (present, RT-as-texture
  // bridge) want the COLOR snapshot; fall back to depth only if that's all
  // there is.
  auto it = resolve_snapshots_.find(uint64_t(dest_base) << 1);
  if (it != resolve_snapshots_.end() && it->second.texture) {
    return it->second.texture;
  }
  it = resolve_snapshots_.find((uint64_t(dest_base) << 1) | 1);
  return it != resolve_snapshots_.end() ? it->second.texture : nullptr;
}

void* MetalRenderTargetCache::FindContainingSnapshot(
    uint32_t dest_base, uint32_t* y_offset_out) const {
  for (const auto& pair : resolve_snapshots_) {
    const Snapshot& snap = pair.second;
    uint32_t snap_base = uint32_t(pair.first >> 1);
    if (!snap.texture || dest_base <= snap_base) {
      continue;
    }
    uint32_t row_bytes = snap.width * 4;
    uint32_t extent = row_bytes * snap.height;
    uint32_t offset = dest_base - snap_base;
    if (offset < extent && (offset % row_bytes) == 0) {
      if (y_offset_out) {
        *y_offset_out = offset / row_bytes;
      }
      return snap.texture;
    }
  }
  return nullptr;
}

void MetalRenderTargetCache::GetEdramOwnersForDump(
    std::vector<EdramOwnerDump>& out) {
  out.clear();
  // The whole EDRAM as one 16-tile-pitch span (matching the snapshot-restore
  // layout and the offline decoder's expectations).
  resolve_dump_rectangles_.clear();
  GetResolveCopyRectanglesToDump(0, 16, xenos::kEdramTileCount / 16, 16,
                                 resolve_dump_rectangles_);
  for (const ResolveCopyDumpRectangle& rect : resolve_dump_rectangles_) {
    auto* rt = static_cast<MetalRenderTarget*>(rect.render_target);
    if (!rt) {
      continue;
    }
    EdramOwnerDump dump;
    dump.texture = rt->texture();
    dump.base_tiles = rt->key().base_tiles;
    dump.pitch_tiles = rt->key().GetPitchTiles();
    dump.msaa_samples = uint32_t(rt->key().msaa_samples);
    dump.is_depth = rt->key().is_depth;
    dump.is_float24_depth =
        rt->key().is_depth &&
        rt->key().GetDepthFormat() == xenos::DepthRenderTargetFormat::kD24FS8;
    dump.pixel_format = rt->pixel_format();
    dump.is_64bpp = uint32_t(rt->key().Is64bpp());
    dump.row_first = rect.row_first;
    dump.rows = rect.rows;
    out.push_back(dump);
  }
}

uint32_t MetalRenderTargetCache::GetTraceDrawIndexForOwnershipTrace() const {
  return command_processor_.trace_draw_index();
}

bool MetalRenderTargetCache::EnsureCanonicalMaterializePipeline() {
  if (canonical_materialize_pipeline_) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  if (!canonical_materialize_library_) {
    // Mirror of the proven materialize_8888 kernel (see the L1 conformance
    // suite). Reads a canonical EDRAM word at each pixel's physical offset and
    // decodes the four byte lanes to an RGBA8Unorm texel. SurfaceParams.pad
    // carries the canonical word count so an over-dispatch can't read OOB.
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct SurfaceParams { uint base, pitch, msaa, is_depth, fmt_log2, words; };
      inline uint xe_edram_offset_bytes(uint px, uint py, uint base, uint pitch,
                                        uint msaa, uint is_depth, uint fmt_log2,
                                        uint sample) {
        uint2 s = uint2(px, py) << uint2(msaa >= 2u ? 1u : 0u,
                                         msaa >= 1u ? 1u : 0u);
        s += (uint2(sample, sample) >> uint2(1u, 0u)) & 1u;
        uint2 tsize32 = uint2(80u, 16u);
        uint2 tsize = tsize32 >> uint2(fmt_log2, 0u);
        uint2 toff = s / tsize;
        uint b = base + toff.y * pitch + toff.x;
        s -= toff * tsize;
        if (is_depth != 0u) {
          uint half_w = tsize.x >> 1u;
          s.x = (s.x >= half_w) ? (s.x - half_w) : (s.x + half_w);
        }
        uint address = b * (tsize32.x * tsize32.y) +
                       ((s.y * tsize.x + s.x) << fmt_log2);
        return address << 2;
      }
      kernel void materialize_8888(
          device const uint* canonical [[buffer(0)]],
          texture2d<float, access::write> tex [[texture(0)]],
          constant SurfaceParams& sp [[buffer(1)]],
          uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
        uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch,
                                         sp.msaa, 0u, 0u, 0u);
        uint idx = off >> 2;
        uint w = (idx < sp.words) ? canonical[idx] : 0u;
        float4 c = float4(w & 0xFFu, (w >> 8) & 0xFFu, (w >> 16) & 0xFFu,
                          (w >> 24) & 0xFFu) * (1.0 / 255.0);
        tex.write(c, gid);
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      XELOGE("MetalRenderTargetCache: canonical materialize library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return false;
    }
    canonical_materialize_library_ = (__bridge_retained void*)lib;
  }
  id<MTLLibrary> lib = (__bridge id<MTLLibrary>)canonical_materialize_library_;
  NSError* err = nil;
  id<MTLComputePipelineState> pso = [device
      newComputePipelineStateWithFunction:[lib newFunctionWithName:@"materialize_8888"]
                                    error:&err];
  if (!pso) {
    XELOGE("MetalRenderTargetCache: canonical materialize pipeline failed: {}",
           err ? err.localizedDescription.UTF8String : "?");
    return false;
  }
  canonical_materialize_pipeline_ = (__bridge_retained void*)pso;
  return true;
}

void MetalRenderTargetCache::RunCanonicalRestoreSelfCheck(
    const void* snapshot, const void* reordered_reference, uint32_t width,
    uint32_t height, uint32_t row_bytes) {
  if (!EnsureCanonicalMaterializePipeline()) {
    return;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  id<MTLBuffer> canonical = (__bridge id<MTLBuffer>)canonical_edram_;
  id<MTLComputePipelineState> pso =
      (__bridge id<MTLComputePipelineState>)canonical_materialize_pipeline_;
  // The snapshot is tile-major physical order == the canonical layout, so
  // populating canonical is a straight copy (blit staging -> private buffer).
  id<MTLBuffer> staging =
      [device newBufferWithBytes:snapshot
                          length:kCanonicalEdramSize
                         options:MTLResourceStorageModeShared];
  // Scratch host attachment: RGBA8Unorm, shared so we can read it back and
  // compare. tiles-per-row = width/80 gives the pitch in tiles (16 for 1280).
  MTLTextureDescriptor* td = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> scratch = [device newTextureWithDescriptor:td];
  if (!staging || !scratch) {
    return;
  }
  struct SurfaceParamsCPU {
    uint32_t base, pitch, msaa, is_depth, fmt_log2, words;
  } sp{0u, width / xenos::kEdramTileWidthSamples, 0u, 0u, 0u,
       kCanonicalEdramSize / 4u};
  id<MTLBuffer> sp_buf = [device newBufferWithBytes:&sp
                                            length:sizeof(sp)
                                           options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    blit.label = @"rtcache:canonical-populate";
    [blit copyFromBuffer:staging
             sourceOffset:0
                 toBuffer:canonical
        destinationOffset:0
                     size:kCanonicalEdramSize];
    [blit endEncoding];
  }
  {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    enc.label = @"rtcache:canonical-materialize-selfcheck";
    [enc setComputePipelineState:pso];
    [enc setBuffer:canonical offset:0 atIndex:0];
    [enc setTexture:scratch atIndex:0];
    [enc setBuffer:sp_buf offset:0 atIndex:1];
    NSUInteger tw = pso.threadExecutionWidth;
    NSUInteger th = pso.maxTotalThreadsPerThreadgroup / tw;
    [enc dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    [enc endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) {
    XELOGW("MetalRenderTargetCache: canonical self-check GPU error: {}",
           cb.error.localizedDescription.UTF8String);
    return;
  }
  // Compare the materialized scratch bytes against the legacy CPU reorder. Both
  // are raw snapshot bytes per pixel, so byte-equality is the exact invariant.
  std::unique_ptr<uint8_t[]> got(new uint8_t[size_t(row_bytes) * height]);
  [scratch getBytes:got.get()
        bytesPerRow:row_bytes
         fromRegion:MTLRegionMake2D(0, 0, width, height)
        mipmapLevel:0];
  const uint8_t* ref = static_cast<const uint8_t*>(reordered_reference);
  size_t total = size_t(row_bytes) * height;
  size_t first_bad = total;
  uint64_t bad_count = 0;
  for (size_t i = 0; i < total; ++i) {
    if (got[i] != ref[i]) {
      if (first_bad == total) first_bad = i;
      ++bad_count;
    }
  }
  if (bad_count == 0) {
    XELOGI(
        "MetalRenderTargetCache: canonical materialize self-check PASS "
        "({}x{}, {} bytes byte-exact vs legacy reorder)",
        width, height, total);
  } else {
    XELOGE(
        "MetalRenderTargetCache: canonical materialize self-check FAIL: {} / {} "
        "bytes differ, first at byte {} (px {},{})",
        bad_count, total, first_bad, (first_bad / 4) % width,
        (first_bad / 4) / width);
  }
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
  blit.label = @"rtcache:edram-restore";
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
  // Gated, additive: exercise + self-check the canonical materialize path on
  // this real snapshot without touching the restored RT above.
  if (sample_edram_enabled_ && canonical_edram_) {
    RunCanonicalRestoreSelfCheck(snapshot, reordered.get(), width, height,
                                 row_bytes);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
