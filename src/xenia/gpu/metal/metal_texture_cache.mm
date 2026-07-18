/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/metal/metal_exact_cover_predicate.h"
#include "xenia/gpu/metal/metal_resolved_fetch_service.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <vector>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/memory.h"

#include "metal_irconverter_runtime.h"

namespace xe {
namespace gpu {
namespace metal {

namespace {
// Guest physical memory is 512 MiB; guard reads to it.
constexpr uint32_t kPhysicalSize = 512u * 1024 * 1024;

// Describes how a guest texture format maps to a host upload. `bytes_per_block`
// / `block_log2` / `block_texels` are the compressed-block geometry (1x1 for
// uncompressed 8_8_8_8). `host_format` is the Metal pixel format we upload to;
// for compressed formats we hand the raw (endian-corrected) blocks to the
// hardware BC decoder, for 8_8_8_8 we build linear RGBA8.
struct FormatPlan {
  bool supported;
  bool compressed;
  uint32_t bytes_per_block;
  uint32_t block_log2;   // log2(bytes_per_block), for the tiled addresser.
  uint32_t block_texels;  // 1 (uncompressed) or 4 (DXT).
  MTLPixelFormat host_format;
  // If true, the untiled 16bpp k_4_4_4_4 texels are CPU-expanded to RGBA8 before
  // upload (host_format is RGBA8Unorm) so the same swizzle path as k_8_8_8_8
  // applies. Avoids the channel-reorder load shader XeniOS/Vulkan use.
  bool decode_4444 = false;
  // If true, the untiled 16bpp single-channel k_16 texels are expanded to RGBA8
  // (high byte -> R=G=B, A=255); the fetch swizzle (typically R,R,R,1) selects R.
  bool decode_r16 = false;
  // SNORM variant served to SIGNED texture bindings (the DXBC translator makes
  // signed fetches a SEPARATE binding expecting -1..1 values; guest signed
  // 8-bit texels are two's-complement, i.e. exactly host snorm bytes).
  MTLPixelFormat host_format_signed = MTLPixelFormatInvalid;
};

FormatPlan PlanForFormat(xenos::TextureFormat format, bool bc_supported) {
  switch (format) {
    case xenos::TextureFormat::k_8_8_8_8:
    // AS_16_16_16_16 (fmt 50): a memory-layout alias of 8888 — same bytes,
    // same decode (JC2's temporal composite fetches its history buffer with
    // it).
    case xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16:
      return {true,  false, 4, 2, 1, MTLPixelFormatRGBA8Unorm,
              false, false, MTLPixelFormatRGBA8Snorm};
    case xenos::TextureFormat::k_8_8:
      // Two-channel (bump/normal maps: swizzle typically (X,X,X,Y), the
      // shader reconstructs Z).
      return {true,  false, 2, 1, 1, MTLPixelFormatRG8Unorm,
              false, false, MTLPixelFormatRG8Snorm};
    case xenos::TextureFormat::k_8:
      return {true,  false, 1, 0, 1, MTLPixelFormatR8Unorm,
              false, false, MTLPixelFormatR8Snorm};
    case xenos::TextureFormat::k_32_FLOAT:
      // Raw f32 words (the resolve dest format of the luminance pyramid).
      return {true, false, 4, 2, 1, MTLPixelFormatR32Float};
    case xenos::TextureFormat::k_16_16_16_16_EXPAND:
    case xenos::TextureFormat::k_16_16_16_16_FLOAT:
      // 64bpp RGBA16F. Sonic's upscale draw reads its 1x1 EXPOSURE and a
      // 220x180 HDR buffer with this — the NULL fetch fed the tonemap a
      // garbage exposure (white-out candidate). Guest halfwords swap with
      // k8in16 (SwapEndian handles 2-byte units across the 8-byte block).
      // EXPAND is the same half-float storage, with filtering semantics that
      // don't alter the upload representation (Red Dead's full-frame HDR
      // chain declares its resolved RGBA16F surfaces through this alias).
      return {true, false, 8, 3, 1, MTLPixelFormatRGBA16Float};
    case xenos::TextureFormat::k_16_16_EXPAND:
    case xenos::TextureFormat::k_16_16_FLOAT:
      return {true, false, 4, 2, 1, MTLPixelFormatRG16Float};
    case xenos::TextureFormat::k_16_EXPAND:
    case xenos::TextureFormat::k_16_FLOAT:
      return {true, false, 2, 1, 1, MTLPixelFormatR16Float};
    case xenos::TextureFormat::k_4_4_4_4:
      // 16bpp RGBA4, decoded to RGBA8 on the CPU (see decode_4444). block_log2 =
      // log2(2 source bytes) = 1.
      return {true, false, 2, 1, 1, MTLPixelFormatRGBA8Unorm, true, false};
    case xenos::TextureFormat::k_16:
      // 16bpp single channel at FULL precision. (Was decode_r16 high-byte
      // RGBA8 expansion — 8-bit quantization banded JC2's 1024x3072 k_16
      // shadow map into visible stripes on every shadow receiver.)
      return {true,  false, 2, 1, 1, MTLPixelFormatR16Unorm,
              false, false, MTLPixelFormatR16Snorm};
    case xenos::TextureFormat::k_DXT1:
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
      return {bc_supported, true, 8, 3, 4, MTLPixelFormatBC1_RGBA};
    case xenos::TextureFormat::k_DXT2_3:
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
      return {bc_supported, true, 16, 4, 4, MTLPixelFormatBC2_RGBA};
    case xenos::TextureFormat::k_DXT4_5:
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
      return {bc_supported, true, 16, 4, 4, MTLPixelFormatBC3_RGBA};
    default:
      return {false, false, 0, 0, 0, MTLPixelFormatInvalid};
  }
}

// Map one guest 3-bit swizzle component (X/Y/Z/W/0/1, values 6/7 clamped to
// 0/1) to a Metal swizzle channel. Our host formats (RGBA8, BC1/2/3_RGBA) all
// store channels in identity RGBA order, so the host-format swizzle is identity
// and the guest fetch swizzle maps straight through.
MTLTextureSwizzle GuestSwizzleComponentToMetal(uint32_t comp) {
  switch (comp) {
    case 0:
      return MTLTextureSwizzleRed;
    case 1:
      return MTLTextureSwizzleGreen;
    case 2:
      return MTLTextureSwizzleBlue;
    case 3:
      return MTLTextureSwizzleAlpha;
    case 4:
    case 6:  // 6/7 are invalid; clamp to constant 0/1 like GuestToHostSwizzle.
      return MTLTextureSwizzleZero;
    default:  // 5, 7
      return MTLTextureSwizzleOne;
  }
}

// True if the 12-bit guest swizzle is the identity RGBA mapping (xyzw), so no
// swizzled view is needed.
bool GuestSwizzleIsIdentity(uint32_t guest_swizzle) {
  return (guest_swizzle & 0xFFF) == ((0u) | (1u << 3) | (2u << 6) | (3u << 9));
}

// Convert a block (or texel) of `bytes` from the guest endianness to host
// little-endian in place. Guest 32bpp is k8in32; guest DXT is normally k8in16
// (each 16-bit word byte-swapped), but honor whatever the fetch declares.
void SwapEndian(uint8_t* data, uint32_t bytes, xenos::Endian endian) {
  switch (endian) {
    case xenos::Endian::k8in16:
      for (uint32_t i = 0; i + 1 < bytes; i += 2) {
        std::swap(data[i], data[i + 1]);
      }
      break;
    case xenos::Endian::k8in32:
      for (uint32_t i = 0; i + 3 < bytes; i += 4) {
        std::swap(data[i], data[i + 3]);
        std::swap(data[i + 1], data[i + 2]);
      }
      break;
    case xenos::Endian::k16in32:
      for (uint32_t i = 0; i + 3 < bytes; i += 4) {
        std::swap(data[i], data[i + 2]);
        std::swap(data[i + 1], data[i + 3]);
      }
      break;
    default:
      break;
  }
}
}  // namespace

MetalTextureCache::MetalTextureCache(void* device, Memory* memory,
                                     SharedMemory* shared_memory,
                                     void* view_heap, uint32_t view_heap_slots)
    : device_(device),
      memory_(memory),
      shared_memory_(shared_memory),
      view_heap_(view_heap),
      view_heap_slots_(view_heap_slots) {
  id<MTLBuffer> heap = (__bridge id<MTLBuffer>)view_heap_;
  view_heap_contents_ = [heap contents];
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_;
  bc_supported_ = [mtl_device supportsBCTextureCompression];
  if (shared_memory_) {
    shared_memory_watch_ = shared_memory_->RegisterGlobalWatch(
        SharedMemoryGlobalWatchCallback, this);
  }
  XELOGI("MetalTextureCache: BC (DXT) texture compression {}",
         bc_supported_ ? "supported" : "UNSUPPORTED (DXT stays gray)");
}

MetalTextureCache::~MetalTextureCache() {
  if (shared_memory_watch_) {
    shared_memory_->UnregisterGlobalWatch(shared_memory_watch_);
    shared_memory_watch_ = nullptr;
  }
  for (auto& pair : cache_) {
    if (pair.second.texture) {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  cache_.clear();
}

void MetalTextureCache::SetTracePlayback() {
  if (trace_playback_) {
    return;
  }
  trace_playback_ = true;
  if (shared_memory_watch_) {
    shared_memory_->UnregisterGlobalWatch(shared_memory_watch_);
    shared_memory_watch_ = nullptr;
  }
  std::lock_guard<std::mutex> lock(pending_invalidations_mutex_);
  pending_invalidations_.clear();
}

void MetalTextureCache::SharedMemoryGlobalWatchCallback(
    [[maybe_unused]] const global_unique_lock_type& global_lock, void* context,
    uint32_t address_first, uint32_t address_last,
    [[maybe_unused]] bool invalidated_by_gpu) {
  static_cast<MetalTextureCache*>(context)->QueueInvalidation(address_first,
                                                               address_last);
}

void MetalTextureCache::QueueInvalidation(uint32_t address_first,
                                          uint32_t address_last) {
  if (address_last < address_first) {
    return;
  }
  std::lock_guard<std::mutex> lock(pending_invalidations_mutex_);
  // Memory invalidation normally arrives in page runs. Coalesce adjacent runs
  // here so a decoder rewriting a frame plane produces one GPU-thread lookup,
  // not one per protected page.
  if (!pending_invalidations_.empty()) {
    auto& last = pending_invalidations_.back();
    uint64_t last_start = last.first;
    uint64_t last_end = uint64_t(last.first) + last.second;
    uint64_t new_start = address_first;
    uint64_t new_end = uint64_t(address_last) + uint64_t(1);
    if (new_start <= last_end && new_end >= last_start) {
      uint64_t merged_start = std::min(last_start, new_start);
      uint64_t merged_end =
          std::max(last_end, new_end);
      last.first = uint32_t(merged_start);
      last.second = uint32_t(merged_end - merged_start);
      return;
    }
  }
  pending_invalidations_.emplace_back(address_first,
                                      address_last - address_first + 1);
}

void MetalTextureCache::DrainPendingInvalidations() {
  std::vector<std::pair<uint32_t, uint32_t>> invalidations;
  {
    std::lock_guard<std::mutex> lock(pending_invalidations_mutex_);
    invalidations.swap(pending_invalidations_);
  }
  uint32_t evicted = 0;
  for (const auto& range : invalidations) {
    evicted += InvalidateRange(range.first, range.second);
  }
  static uint32_t s_cpu_invalidation_logs = 0;
  if (evicted && s_cpu_invalidation_logs++ < 16) {
    XELOGI("MetalTextureCache: CPU/GPU writes refreshed {} cached texture{}",
           evicted, evicted == 1 ? "" : "s");
  }
}

// --- Snapshot-bridge temporal-loop lineage probe (XE_METAL_LINEAGE=<base hex>,
// e.g. 11849000 = JC2's composite temporal history buffer). One probe that
// tags every event on the watched base with the shared draw-index clock, a raw
// guest-RAM content hash, and the reader's cache HIT/MISS + RAM/bridge decision,
// so the whole Frame-N-resolve -> RAM-scatter -> Frame-N+1-fetch chain is
// visible in one run and the failing edge is identified (not iterated visually).
static uint32_t xe_lineage_base() {
  static uint32_t b = [] {
    const char* v = std::getenv("XE_METAL_LINEAGE");
    return v ? uint32_t(strtoul(v, nullptr, 16)) : 0u;
  }();
  return b;
}
static uint32_t xe_lineage_fnv(const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 16777619u;
  return h;
}

// The single texture-key definition — the routed-load prepass and the draw
// path MUST agree byte-for-byte or the route degrades to a safe CPU-path
// miss (never corruption).
static uint64_t MakeTextureKey(const xenos::xe_gpu_texture_fetch_t& fetch,
                               bool is_signed, uint32_t base, uint32_t width,
                               uint32_t height, uint32_t pitch_texels) {
  struct KeyBytes {
    uint32_t base, width, height, pitch;
    uint32_t format_tiled_endian;
  } kb{base, width, height, pitch_texels,
       (uint32_t(fetch.format) << 16) | (uint32_t(is_signed) << 15) |
           (uint32_t(fetch.dimension) << 12) | (uint32_t(fetch.tiled) << 8) |
           uint32_t(fetch.endianness)};
  return XXH3_64bits(&kb, sizeof(kb));
}


// Shared service-classification input builder (slice 5R): the ONE place the
// exact-cover and bridge inputs are computed. GetTextureSlot's serve chain
// and the routed-load prepass both classify through this — the guards live
// in ClassifyResolvedFetch/ExactCoverSnapshotEligible and are never
// duplicated. Returns false when the base has no registered resolve target.
bool MetalTextureCache::BuildResolvedServiceInputs(
    const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed,
    ResolvedFetchServiceInputs* out) const {
  // Stage 3 is DEFAULT ON (XE_METAL_STAGE3=0 disables) — keep in sync with
  // the writeback-cap reader in metal_command_processor.mm.
  static const bool s_stage3 = [] {
    const char* v = std::getenv("XE_METAL_STAGE3");
    return !v || v[0] != '0';
  }();
  static const bool s_color_bridge =
      !s_stage3 || std::getenv("XE_METAL_COLOR_BRIDGE") != nullptr;
  static const bool s_tip_ram_fetch =
      std::getenv("XE_METAL_TIP_RAM_FETCH") != nullptr;
  out->has_resolved_target = false;
  auto rit = resolved_targets_.find(fetch.base_address << 12);
  if (rit == resolved_targets_.end() || !rit->second.texture) {
    return false;
  }
  const ResolvedTarget& rt = rit->second;
  uint32_t fetch_w = fetch.size_2d.width + 1;
  uint32_t fetch_h = fetch.size_2d.height + 1;
  // Covered-fetch fast path: R32Float ONLY (see the serve chain's rationale;
  // covered RGBA8 snapshots regressed the gate and need the RAM path).
  bool covered_r32f =
      !rt.is_depth && rt.width >= fetch_w && rt.height >= fetch_h &&
      ((__bridge id<MTLTexture>)rt.texture).pixelFormat ==
          MTLPixelFormatR32Float;
  // DIAGNOSTIC (stabilize 6.6): XE_METAL_TIP_RAM_FETCH=1 forces 1x1 R32F
  // fetches (the exposure tips) down the RAM path.
  if (s_tip_ram_fetch && covered_r32f && fetch_w == 1 && fetch_h == 1 &&
      !rt.is_depth && !s_color_bridge && rt.ram_backed) {
    covered_r32f = false;
  }
  id<MTLTexture> exact_snap = (__bridge id<MTLTexture>)rt.texture;
  out->exact = ExactCoverInputs{
      rt.is_depth,
      s_color_bridge,
      rt.ram_backed,
      rt.full_rect,
      is_signed,
      exact_snap.pixelFormat == MTLPixelFormatRGBA8Unorm,
      fetch.format,
      bool(fetch.tiled),
      fetch.dimension,
      bool(fetch.stacked),
      fetch.mip_min_level,
      fetch.mip_max_level,
      fetch_w,
      fetch_h,
      uint32_t(fetch.pitch) << 5,
      rt.width,
      rt.height,
      uint32_t(fetch.endianness),
      rt.wb_endian};
  out->covered_r32f = covered_r32f;
  out->has_resolved_target = true;
  return true;
}

ResolvedFetchService MetalTextureCache::ClassifyFetchService(
    const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed) const {
  ResolvedFetchServiceInputs in;
  BuildResolvedServiceInputs(fetch, is_signed, &in);
  return ClassifyResolvedFetch(in);
}

uint32_t MetalTextureCache::GetTextureSlot(
    const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed) {
  if (!trace_playback_) {
    DrainPendingInvalidations();
  }
  // Diagnostic: log why a fetch resolves to the null (gray) texture. The
  // gameplay gray screen is a composite pass sampling slot 0; this shows the
  // fetch (base/format/dimension) so we know whether it's an unsupported format,
  // a non-2D dimension, or a resolve-backed base with no data.
  static uint32_t s_null_log = 0;
  auto log_null = [&](const char* reason) {
    if (s_null_log < 300) {
      ++s_null_log;
      XELOGI(
          "MetalTex NULL ({}): type={} dim={} fmt={} base={:08X} {}x{} "
          "tiled={} swz={:03X}",
          reason, uint32_t(fetch.type), uint32_t(fetch.dimension),
          uint32_t(fetch.format), fetch.base_address << 12,
          fetch.size_2d.width + 1, fetch.size_2d.height + 1,
          uint32_t(fetch.tiled), fetch.swizzle & 0xFFF);
    }
  };
  if (fetch.type != xenos::FetchConstantType::kTexture) {
    return 0;
  }
  // Shader-executor harness (XE_METAL_SYNTH_FETCH=hexbase,R,G,B,A): serve a
  // constant-color texture for that base — a known input pushed through the
  // REAL pipeline (VS, interpolators, sampler, translated PS). Reading the
  // consuming pass's output then fingerprints any channel permutation/scale
  // the translated shader introduces.
  {
    struct SynthFetch {
      uint32_t base = 0;
      uint8_t rgba[4] = {0, 0, 0, 0};
    };
    static const SynthFetch s_synth = [] {
      SynthFetch s;
      if (const char* v = std::getenv("XE_METAL_SYNTH_FETCH")) {
        unsigned b = 0, r = 0, g = 0, bl = 0, a = 255;
        if (sscanf(v, "%x,%u,%u,%u,%u", &b, &r, &g, &bl, &a) >= 4) {
          s.base = b;
          s.rgba[0] = uint8_t(r);
          s.rgba[1] = uint8_t(g);
          s.rgba[2] = uint8_t(bl);
          s.rgba[3] = uint8_t(a);
        }
      }
      return s;
    }();
    if (s_synth.base && (fetch.base_address << 12) == s_synth.base) {
      static void* s_synth_tex = nullptr;
      static uint32_t s_synth_slot = 0;
      if (!s_synth_tex) {
        // Match the fetch's DECLARED dimensions: unnormalized-coordinate
        // fetches read at pixel coords sized for the real texture — an
        // undersized stand-in reads out of bounds (undefined garbage).
        uint32_t sw = fetch.size_2d.width + 1;
        uint32_t sh = fetch.size_2d.height + 1;
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
        MTLTextureDescriptor* d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                         width:sw
                                        height:sh
                                     mipmapped:NO];
        d.textureType = MTLTextureType2DArray;
        d.arrayLength = 1;
        d.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> tex = [device newTextureWithDescriptor:d];
        std::vector<uint8_t> px(size_t(sw) * sh * 4);
        for (size_t i = 0; i < size_t(sw) * sh; ++i) {
          std::memcpy(&px[i * 4], s_synth.rgba, 4);
        }
        [tex replaceRegion:MTLRegionMake2D(0, 0, sw, sh)
               mipmapLevel:0
                     slice:0
                 withBytes:px.data()
               bytesPerRow:sw * 4
             bytesPerImage:size_t(sw) * sh * 4];
        s_synth_tex = (__bridge_retained void*)tex;
        s_synth_slot = AllocateSlot();
        auto* entries =
            static_cast<IRDescriptorTableEntry*>(view_heap_contents_);
        IRDescriptorTableSetTexture(&entries[s_synth_slot],
                                    (__bridge id<MTLTexture>)s_synth_tex, 0.0f,
                                    0);
        XELOGW("MetalSynthFetch: base {:08X} -> constant ({},{},{},{}) slot {}",
               s_synth.base, s_synth.rgba[0], s_synth.rgba[1], s_synth.rgba[2],
               s_synth.rgba[3], s_synth_slot);
      }
      active_textures_.push_back(s_synth_tex);
      return s_synth_slot;
    }
  }
  // RT-as-texture bridge — STAGE 3 hybrid. Served for:
  //   - DEPTH (no guest-RAM encoding of host depth yet);
  //   - color bases whose resolves never enqueued a RAM writeback (dest
  //     formats the writer can't encode — 64bpp HDR etc.);
  //   - RAM-backed color bases whose registered snapshot FULLY COVERS the
  //     fetch (single-buffer dests: JC2's luminance-pyramid levels, SF4's
  //     per-column dests). The snapshot holds exactly what the writeback
  //     scatters, and sampling it is GPU-ordered — the RAM path for these
  //     forced a mid-frame commit+wait PER CONSUMER (JC2: 39 commits/frame,
  //     21.5 -> 12.7 fps when stage 3 went default-ON).
  // Everything else (multi-writeback assemblies, base reuse, strips — the
  // registered dims don't cover the fetch there) reads guest RAM: JC2's
  // final composite reads LAST frame's 1280x720 image while the base's
  // snapshot was re-registered 1280x384 by the mid-frame G-buffer dump —
  // coverage fails -> RAM, exactly the case the bridge can't represent.
  // XE_METAL_COLOR_BRIDGE=1 restores bridge-for-all-color for A/B.
  {
    auto rit = resolved_targets_.find(fetch.base_address << 12);
    if (rit != resolved_targets_.end() && rit->second.texture) {
      const ResolvedTarget& rt = rit->second;
      uint32_t fetch_w = fetch.size_2d.width + 1;
      uint32_t fetch_h = fetch.size_2d.height + 1;
      // ONE shared decision (slice 5R): exact-cover and bridge conditions
      // are composed by ClassifyResolvedFetch in this serve order; the
      // routed prepass consults the identical classification.
      ResolvedFetchServiceInputs service_in;
      BuildResolvedServiceInputs(fetch, is_signed, &service_in);
      ResolvedFetchService service = ClassifyResolvedFetch(service_in);
      bool covered_r32f = service_in.covered_r32f;
      if (service == ResolvedFetchService::kExactCoverSnapshot) {
        // Exact-cover RGBA8 fast path: the snapshot is provably the same
        // content the RAM upload would produce (per-condition rationale in
        // metal_exact_cover_predicate.h and the design doc).
        if (xe_lineage_base() &&
            (fetch.base_address << 12) == xe_lineage_base()) {
          XELOGW(
              "MetalLineage EXACT-COVER draw={} base={:08X} served snapshot "
              "{}x{} endian={} (RAM upload avoided)",
              lineage_draw_, fetch.base_address << 12, rt.width, rt.height,
              rt.wb_endian);
        }
        active_textures_.push_back(rt.texture);
        return rt.slot;
      }
      if (service == ResolvedFetchService::kBridgeSnapshot) {
        // XE_METAL_TIP_SNAP_VIEW: a 1x1 guest fetch of a larger snapshot must
        // read guest texel (0,0); binding the full WxH snapshot lands the
        // shader's UV mid-texture (8x8 exposure tip -> texel 4,4 — the JC2
        // steady-dark mechanism). Serve the derived 1x1 blit instead;
        // GPU-ordered like the snapshot itself.
        if (covered_r32f && fetch_w == 1 && fetch_h == 1 &&
            (rt.width > 1 || rt.height > 1) && rt.tip_texture) {
          if (xe_lineage_base() &&
              (fetch.base_address << 12) == xe_lineage_base()) {
            XELOGW(
                "MetalLineage TIP-VIEW draw={} base={:08X} served derived 1x1 "
                "(snapshot texel 0,0; rt={}x{})",
                lineage_draw_, fetch.base_address << 12, rt.width, rt.height);
          }
          active_textures_.push_back(rt.tip_texture);
          return rt.tip_slot;
        }
        if (xe_lineage_base() && (fetch.base_address << 12) == xe_lineage_base()) {
          XELOGW(
              "MetalLineage BRIDGE draw={} base={:08X} served SNAPSHOT rt={}x{} "
              "fetch={}x{} is_depth={} ram_backed={} covered_r32f={} "
              "(RAM bypassed)",
              lineage_draw_, fetch.base_address << 12, rt.width, rt.height,
              fetch_w, fetch_h, rt.is_depth, rt.ram_backed, covered_r32f);
        }
        active_textures_.push_back(rt.texture);
        return rt.slot;
      }
      // kRamContent: RAM authoritative — fall through to the cache/untile
      // path below (and, under slice 5R, the routed GPU load).
    }
  }
  // 2D (or stacked), 3D and CUBE are handled; 3D is sampled as its slice 0 as
  // a 2D texture (the guest gpu_3d_to_2d_texture behavior). Cubes upload all
  // 6 faces (guest layout: faces sequential, each a 32x32-block-aligned tiled
  // slice) into an MTLTextureTypeCubeArray — MSC's ForceTextureArray turns
  // the translated TextureCube SRV into texturecube_array. 1D is skipped.
  bool is_3d = fetch.dimension == xenos::DataDimension::k3D;
  bool is_cube = fetch.dimension == xenos::DataDimension::kCube;
  if (fetch.dimension != xenos::DataDimension::k2DOrStacked && !is_3d &&
      !is_cube) {
    log_null("dim");
    return 0;
  }
  FormatPlan plan = PlanForFormat(fetch.format, bc_supported_);
  if (!plan.supported) {
    log_null("fmt");
    return 0;
  }
  if (is_signed && plan.host_format_signed != MTLPixelFormatInvalid) {
    plan.host_format = plan.host_format_signed;
  }

  uint32_t width = (is_3d ? fetch.size_3d.width : fetch.size_2d.width) + 1;
  uint32_t height = (is_3d ? fetch.size_3d.height : fetch.size_2d.height) + 1;
  uint32_t base = fetch.base_address << 12;
  uint32_t pitch_texels = fetch.pitch << 5;  // guest pitch is texels/32.
  if (!base || !width || !height || width > 8192 || height > 8192 ||
      base >= kPhysicalSize) {
    log_null("bounds");
    return 0;
  }
  if (!pitch_texels) {
    pitch_texels = (width + 31) & ~31u;
  }

  uint64_t key = MakeTextureKey(fetch, is_signed, base, width, height,
                                pitch_texels);
  auto it = cache_.find(key);
  // Lineage probe: this fetch of the watched base is the composite's temporal
  // read. A cache HIT means the untile is SKIPPED and the STALE cached upload is
  // served -- the freshly-resolved guest RAM never reaches the composite (the
  // invalidation/ordering break). ramHash is a strided hash of the guest RAM the
  // untile WOULD read; compare across frames vs the DRAIN ramHash to see whether
  // this reads last frame's resolve (correct temporal) or something stale.
  if (xe_lineage_base() && base == xe_lineage_base()) {
    const uint8_t* g =
        static_cast<const uint8_t*>(memory_->TranslatePhysical(base));
    uint32_t rh = 0;
    if (g) {
      size_t region = std::min<size_t>(size_t(pitch_texels) * height * 4u,
                                       size_t(kPhysicalSize - base));
      uint32_t h = 2166136261u;
      for (size_t i = 0; i < region; i += 37) h = (h ^ g[i]) * 16777619u;
      rh = h;
    }
    bool ram_backed = false;
    auto rit = resolved_targets_.find(base);
    if (rit != resolved_targets_.end()) ram_backed = rit->second.ram_backed;
    XELOGW(
        "MetalLineage FETCH draw={} base={:08X} {}x{} pitch={} fmt={} tiled={} "
        "endian={} cache={} ram_backed={} ramHash={:08x}",
        lineage_draw_, base, width, height, pitch_texels,
        uint32_t(fetch.format), uint32_t(fetch.tiled),
        uint32_t(fetch.endianness), it != cache_.end() ? "HIT" : "MISS",
        ram_backed, rh);
  }
  if (it != cache_.end()) {
    active_textures_.push_back(it->second.texture);
    return it->second.slot;
  }
  if (next_slot_ >= view_heap_slots_ && free_slots_.empty()) {
    return 0;
  }

  const uint8_t* guest =
      static_cast<const uint8_t*>(memory_->TranslatePhysical(base));
  if (!guest) {
    return 0;
  }

  bool tiled = fetch.tiled != 0;
  xenos::Endian endian = fetch.endianness;
  // Calibration override (XE_METAL_HISTORY_ENDIAN=0..3): endian applied when
  // untiling fetches of RESOLVE-WRITTEN bases — hardware's resolve write
  // order vs artist-texture order is being determined empirically.
  static const int s_history_endian = [] {
    const char* v = std::getenv("XE_METAL_HISTORY_ENDIAN");
    return v ? atoi(v) : -1;
  }();
  if (s_history_endian >= 0 &&
      resolved_targets_.find(base) != resolved_targets_.end()) {
    endian = xenos::Endian(s_history_endian & 3);
  }
  uint32_t max_src_bytes = kPhysicalSize - base;

  // Block grid (1x1 for uncompressed). The tiled addresser works in block
  // coordinates; the pitch it wants is in blocks, 32-block aligned.
  uint32_t block_texels = plan.block_texels;
  uint32_t bpb = plan.bytes_per_block;
  uint32_t blocks_w = (width + block_texels - 1) / block_texels;
  uint32_t blocks_h = (height + block_texels - 1) / block_texels;
  uint32_t pitch_blocks = pitch_texels / block_texels;
  uint32_t pitch_blocks_aligned = (pitch_blocks + 31u) & ~31u;

  uint32_t blocks_h_aligned = (blocks_h + 31u) & ~31u;
  size_t base_source_bytes =
      is_cube ? size_t(pitch_blocks_aligned) * blocks_h_aligned * bpb * 6
              : size_t(pitch_blocks_aligned) * blocks_h * bpb;
  uint32_t watched_base_bytes = uint32_t(std::min<size_t>(
      base_source_bytes, size_t(kPhysicalSize - base)));
  // NOTE: this cache deliberately does NOT call SharedMemory::RequestRange to
  // register these pages for CPU write-watching. RequestRange does host-GPU
  // buffer residency + guest-page write protection (EnsureHostGpuMemoryAllocated
  // -> UploadRanges), a memory model this bring-up cache doesn't otherwise use:
  // it reads guest RAM directly below. Enabling it CORRUPTED that direct read —
  // font-atlas glyphs rendered as noise blocks and the HDR/exposure chain blew
  // out to white live (SF4 434307FC), because protecting/relocating the source
  // pages tore the concurrent untile read. Live-bisected 2026-07-14: this call
  // was the sole cause; removing it renders SF4 clean. It was added to refresh
  // FMV/video textures the CPU rewrites in place — that refresh now needs a
  // Metal-safe page watch (physical access callbacks) instead; tracked
  // separately. The retire path (QueueInvalidation/InvalidateRange) stays for
  // that future watch and for trace playback's explicit restores.

  // Untile / copy into a linear block-major staging buffer (host byte order).
  // Rows are independent — run them concurrently: stage-3 re-uploads big
  // resolve dests every frame (SF4: 1280x720 each frame on the GPU thread),
  // and the single-threaded per-block loop was a measurable slice of the
  // live frame.
  std::vector<uint8_t> linear(size_t(blocks_w) * blocks_h * bpb);
  auto untile_row = [&](size_t by) {
    for (uint32_t bx = 0; bx < blocks_w; ++bx) {
      uint32_t src_off;
      if (tiled) {
        src_off = uint32_t(texture_address::Tiled2D(
            int32_t(bx), int32_t(by), pitch_blocks_aligned, plan.block_log2));
      } else {
        src_off = uint32_t(by * pitch_blocks + bx) * bpb;
      }
      uint8_t* dst = &linear[(by * blocks_w + bx) * bpb];
      if (src_off + bpb <= max_src_bytes) {
        std::memcpy(dst, guest + src_off, bpb);
        SwapEndian(dst, bpb, endian);
      }
    }
  };
  if (size_t(blocks_w) * blocks_h * bpb >= 512 * 1024) {
    dispatch_apply(blocks_h, DISPATCH_APPLY_AUTO,
                   ^(size_t by) { untile_row(by); });
  } else {
    for (uint32_t by = 0; by < blocks_h; ++by) {
      untile_row(by);
    }
  }


  // Data + row stride handed to Metal. k_4_4_4_4 is CPU-expanded here from its
  // 16bpp linear form to RGBA8 (each 4-bit channel -> 8 bits via *17), in the
  // guest's channel order (ascending nibble -> ascending byte), so the same
  // fetch-swizzle-view path as k_8_8_8_8 makes channels correct.
  const uint8_t* upload_data = linear.data();
  uint32_t upload_bytes_per_row = blocks_w * bpb;
  // Calibration (XE_METAL_DUMP_FETCH=<hex base>): print center texels of the
  // untiled+endian-swapped upload for that base — channel-permutation ground
  // truth against a reference render of the same content.
  static const uint32_t s_dump_fetch = [] {
    const char* v = std::getenv("XE_METAL_DUMP_FETCH");
    return v ? uint32_t(strtoul(v, nullptr, 16)) : 0u;
  }();
  if (s_dump_fetch && base == s_dump_fetch) {
    if (bpb == 4) {
      auto texel = [&](uint32_t x, uint32_t y) {
        const uint8_t* t = &linear[(size_t(y) * blocks_w + x) * 4];
        return fmt::format("({},{},{},{})", t[0], t[1], t[2], t[3]);
      };
      XELOGW("MetalFetchDump {:08X} fmt={} {}x{} swz={:03X}: c={} q1={} q3={}",
             base, uint32_t(fetch.format), width, height, fetch.swizzle & 0xFFF,
             texel(width / 2, height / 2), texel(width / 4, height / 4),
             texel(3 * width / 4, 3 * height / 4));
    } else if (bpb == 2) {
      // 16bpp (k_16 LUTs etc.): print a ramp of raw 16-bit values.
      std::string vals;
      for (uint32_t x = 0; x < width; x += std::max(1u, width / 16)) {
        const uint8_t* t = &linear[size_t(x) * 2];
        vals += fmt::format(" {:04X}", uint32_t(t[0]) | (uint32_t(t[1]) << 8));
      }
      XELOGW("MetalFetchDump {:08X} fmt={} {}x{} swz={:03X} endian={}:{}",
             base, uint32_t(fetch.format), width, height, fetch.swizzle & 0xFFF,
             uint32_t(endian), vals);
    }
  }
  // XE_METAL_DUMP_FETCH_PNG=<hex base>: dump the whole untiled+swapped upload
  // as /tmp/xe_fetch_<base>.ppm (bytes 0,1,2 as RGB). Run once reader-only
  // (trace-restored RAM = hardware content) and once with stage-3 writebacks
  // (our content) — the visual diff attributes divergence to the producer.
  static const uint32_t s_dump_fetch_png = [] {
    const char* v = std::getenv("XE_METAL_DUMP_FETCH_PNG");
    return v ? uint32_t(strtoul(v, nullptr, 16)) : 0u;
  }();
  if (s_dump_fetch_png && base == s_dump_fetch_png && bpb == 4) {
    std::string path = fmt::format("/tmp/xe_fetch_{:08X}.ppm", base);
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
      fprintf(f, "P6\n%u %u\n255\n", blocks_w, blocks_h);
      for (size_t i = 0; i < size_t(blocks_w) * blocks_h; ++i) {
        fwrite(&linear[i * 4], 1, 3, f);
      }
      fclose(f);
      XELOGW("MetalFetchPNG: wrote {} ({}x{})", path, blocks_w, blocks_h);
    }
  }
  std::vector<uint8_t> decoded;
  if (plan.decode_4444 || plan.decode_r16) {
    decoded.resize(size_t(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        size_t si = (size_t(y) * blocks_w + x) * bpb;
        uint16_t v = uint16_t(linear[si] | (linear[si + 1] << 8));
        uint8_t* o = &decoded[(size_t(y) * width + x) * 4];
        if (plan.decode_r16) {
          uint8_t r = uint8_t(v >> 8);  // high byte, unorm
          o[0] = o[1] = o[2] = r;
          o[3] = 255;
        } else {  // decode_4444
          o[0] = uint8_t(((v >> 0) & 0xF) * 17);
          o[1] = uint8_t(((v >> 4) & 0xF) * 17);
          o[2] = uint8_t(((v >> 8) & 0xF) * 17);
          o[3] = uint8_t(((v >> 12) & 0xF) * 17);
        }
      }
    }
    upload_data = decoded.data();
    upload_bytes_per_row = width * 4;
  }

  // Bring-up diagnostic: a texture that decodes to all-zero bytes renders pure
  // black. The most likely source is a render-target-backed surface (e.g. the
  // resolved scene) whose guest memory we never populated because IssueCopy is
  // still a stub. Flag it so we can tell "black texture" from a shader bug.
  static uint32_t s_tex_log = 0;
  bool all_zero = true;
  for (uint8_t b : linear) {
    if (b) {
      all_zero = false;
      break;
    }
  }
  if (s_tex_log < 24) {
    ++s_tex_log;
    XELOGI(
        "MetalTex: fmt={} {}x{} base={:08X} pitch_t={} tiled={} endian={} "
        "swz={:03X} slot={} {}",
        uint32_t(fetch.format), width, height, base, pitch_texels, tiled,
        uint32_t(endian), fetch.swizzle & 0xFFF, next_slot_,
        all_zero ? "ALL-ZERO(black)" : "hasdata");
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  // MUST be a 2D ARRAY texture: our DXBC translator emits texture2d_array reads
  // and MSC is compiled with IRCompatibilityFlagForceTextureArray, so a plain
  // 2D texture sampled through an array read returns garbage (flat sampling).
  // Cubes likewise must be CUBE ARRAYS (texturecube_array in the MSL).
  if (is_cube && (width != height || plan.decode_4444 || plan.decode_r16)) {
    log_null("cube-shape");
    return 0;
  }
  // Mip chain: mips live at fetch.mip_address with their own layout
  // (pow2-rounded, 32-block-aligned, small levels packed into a tail —
  // texture_util::GetGuestTextureLayout). Serving mip 0 only point/linear
  // aliases every minified texture (JC2 cloth moire). 2D only for now.
  uint32_t mip_addr = fetch.mip_address << 12;
  uint32_t host_levels = 1;
  // A/B: XE_METAL_NO_MIPS=1 disables mip upload; XE_METAL_NO_PACKED_MIPS=1
  // uploads only the levels before the packed tail.
  static const bool s_no_mips = std::getenv("XE_METAL_NO_MIPS") != nullptr;
  static const bool s_no_packed_mips =
      std::getenv("XE_METAL_NO_PACKED_MIPS") != nullptr;
  static const bool s_mips_dxt_only =
      std::getenv("XE_METAL_MIPS_DXT_ONLY") != nullptr;
  static const bool s_mips_uncomp_only =
      std::getenv("XE_METAL_MIPS_UNCOMP_ONLY") != nullptr;
  bool mip_format_allowed = !(s_mips_dxt_only && !plan.compressed) &&
                            !(s_mips_uncomp_only && plan.compressed);
  if (!s_no_mips && mip_format_allowed && !is_cube && !is_3d && mip_addr &&
      mip_addr < kPhysicalSize && fetch.mip_max_level > 0 &&
      !plan.decode_4444 && !plan.decode_r16) {
    host_levels = std::min<uint32_t>(
        uint32_t(fetch.mip_max_level) + 1,
        xe::log2_floor(std::max(width, height)) + 1);
    if (s_no_packed_mips && fetch.packed_mips) {
      host_levels = std::min(
          host_levels, std::max(1u, texture_util::GetPackedMipLevel(width, height)));
    }
    // Content guard: trace replays don't carry mip memory (the capture never
    // recorded it) — the restored pages are ZEROS, and uploading them blacks
    // out every minified surface. If the first 4KB at mip_address is
    // all-zero, serve the base level only (live RAM has real mips; a
    // genuinely black mip tail just loses nothing visible).
    const uint32_t* probe = reinterpret_cast<const uint32_t*>(
        memory_->TranslatePhysical(mip_addr));
    if (probe) {
      uint32_t probe_words =
          std::min<uint32_t>(1024, (kPhysicalSize - mip_addr) / 4);
      bool any_nonzero = false;
      for (uint32_t i = 0; i < probe_words; ++i) {
        if (probe[i]) {
          any_nonzero = true;
          break;
        }
      }
      if (!any_nonzero) {
        host_levels = 1;
      }
    } else {
      host_levels = 1;
    }
  }
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = is_cube ? MTLTextureTypeCubeArray : MTLTextureType2DArray;
  desc.pixelFormat = plan.host_format;
  desc.width = width;
  desc.height = height;
  desc.arrayLength = 1;  // For cube arrays this is the CUBE count (6 slices).
  desc.mipmapLevelCount = host_levels;
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return 0;
  }
  // For BC formats bytesPerRow is a row of blocks; MTLRegion is still in texels
  // and Metal handles partial edge blocks. Slice 0 of the single array layer.
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
                   slice:0
               withBytes:upload_data
             bytesPerRow:upload_bytes_per_row
           bytesPerImage:0];
  // Cube faces 1..5: guest layout is face-major — each face is a full tiled
  // (or linear) slice whose block grid is padded to 32x32 blocks
  // (texture_extent.cc guest rules); re-untile each face at its offset.
  uint32_t face_stride = 0;
  if (is_cube) {
    face_stride = pitch_blocks_aligned * blocks_h_aligned * bpb;
    for (uint32_t face = 1; face < 6; ++face) {
      uint32_t face_base = face * face_stride;
      for (uint32_t by = 0; by < blocks_h; ++by) {
        for (uint32_t bx = 0; bx < blocks_w; ++bx) {
          uint32_t src_off;
          if (tiled) {
            src_off = face_base + uint32_t(texture_address::Tiled2D(
                                      int32_t(bx), int32_t(by),
                                      pitch_blocks_aligned, plan.block_log2));
          } else {
            src_off = face_base + (by * pitch_blocks + bx) * bpb;
          }
          uint8_t* dst = &linear[(size_t(by) * blocks_w + bx) * bpb];
          if (src_off + bpb <= max_src_bytes) {
            std::memcpy(dst, guest + src_off, bpb);
            SwapEndian(dst, bpb, endian);
          }
        }
      }
      [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                 mipmapLevel:0
                       slice:face
                   withBytes:linear.data()
                 bytesPerRow:upload_bytes_per_row
               bytesPerImage:0];
    }
  }
  uint32_t entry_mip_bytes = 0;
  if (host_levels > 1) {
    texture_util::TextureGuestLayout layout = texture_util::GetGuestTextureLayout(
        fetch.dimension, fetch.pitch, width, height, 1, tiled, fetch.format,
        fetch.packed_mips != 0, /*has_base=*/true, fetch.mip_max_level);
    entry_mip_bytes = std::min<uint32_t>(layout.mips_total_extent_bytes,
                                         kPhysicalSize - mip_addr);
    // Mips live at mip_address, NOT relative to the base level — read through
    // their own translation with their own bounds.
    const uint8_t* mip_guest =
        static_cast<const uint8_t*>(memory_->TranslatePhysical(mip_addr));
    uint32_t max_mip_bytes = kPhysicalSize - mip_addr;
    std::vector<uint8_t> mip_linear;
    for (uint32_t level = 1; mip_guest && level < host_levels; ++level) {
      uint32_t lw = std::max(1u, width >> level);
      uint32_t lh = std::max(1u, height >> level);
      uint32_t lbw = (lw + block_texels - 1) / block_texels;
      uint32_t lbh = (lh + block_texels - 1) / block_texels;
      uint32_t storage_level = level;
      uint32_t xb = 0, yb = 0, zb = 0;
      if (layout.packed_level != UINT32_MAX && level >= layout.packed_level) {
        storage_level = layout.packed_level;
        texture_util::GetPackedMipOffset(width, height, 1, fetch.format, level,
                                         xb, yb, zb);
      }
      if (storage_level >= xenos::kTextureMaxMips) {
        break;
      }
      const auto& lvl = layout.mips[storage_level];
      if (!lvl.row_pitch_bytes) {
        break;
      }
      uint32_t level_base = layout.mip_offsets_bytes[storage_level];
      uint32_t lpitch_blocks = lvl.row_pitch_bytes / bpb;
      mip_linear.assign(size_t(lbw) * lbh * bpb, 0);
      for (uint32_t by = 0; by < lbh; ++by) {
        for (uint32_t bx = 0; bx < lbw; ++bx) {
          uint32_t src_off;
          if (tiled) {
            src_off = level_base +
                      uint32_t(texture_address::Tiled2D(
                          int32_t(bx + xb), int32_t(by + yb), lpitch_blocks,
                          plan.block_log2));
          } else {
            src_off = level_base + (by + yb) * lvl.row_pitch_bytes +
                      (bx + xb) * bpb;
          }
          uint8_t* dst = &mip_linear[(size_t(by) * lbw + bx) * bpb];
          if (src_off + bpb <= max_mip_bytes) {
            std::memcpy(dst, mip_guest + src_off, bpb);
            SwapEndian(dst, bpb, endian);
          }
        }
      }
      // XE_METAL_MIP_TRACE=1: sample the untiled level's bytes (mip
      // corruption forensics — plausible-DXT vs noise vs zeros).
      static const bool s_mip_trace = std::getenv("XE_METAL_MIP_TRACE");
      static uint32_t s_mip_trace_count = 0;
      if (s_mip_trace && level == 1 && s_mip_trace_count < 8) {
        ++s_mip_trace_count;
        const uint8_t* l0 = linear.data();
        const uint8_t* l1 = mip_linear.data();
        size_t mid1 = (mip_linear.size() / 2) & ~15ull;
        // First nonzero byte offset in the raw guest mip region — reveals
        // where the data actually lives vs where the layout math looks.
        uint32_t first_nz = 0xFFFFFFFFu;
        uint32_t scan_len = std::min<uint32_t>(0x100000, kPhysicalSize - mip_addr);
        const uint8_t* raw = mip_guest;
        for (uint32_t i = 0; i < scan_len; ++i) {
          if (raw[i]) {
            first_nz = i;
            break;
          }
        }
        XELOGW("MetalMipTrace   first_nonzero at mip_addr+{:#x}", first_nz);
        XELOGW(
            "MetalMipTrace base={:08X} fmt={} {}x{} mip_addr={:08X} "
            "packed_lvl={} off1={} pitch1={} L0[0..8]={:02X}{:02X}{:02X}{:02X}"
            "{:02X}{:02X}{:02X}{:02X} L1[0..8]={:02X}{:02X}{:02X}{:02X}{:02X}"
            "{:02X}{:02X}{:02X} L1[mid]={:02X}{:02X}{:02X}{:02X}{:02X}{:02X}"
            "{:02X}{:02X}",
            base, uint32_t(fetch.format), width, height, mip_addr,
            layout.packed_level, layout.mip_offsets_bytes[storage_level],
            lpitch_blocks, l0[0], l0[1], l0[2], l0[3], l0[4], l0[5], l0[6],
            l0[7], l1[0], l1[1], l1[2], l1[3], l1[4], l1[5], l1[6], l1[7],
            l1[mid1], l1[mid1 + 1], l1[mid1 + 2], l1[mid1 + 3], l1[mid1 + 4],
            l1[mid1 + 5], l1[mid1 + 6], l1[mid1 + 7]);
      }
      [texture replaceRegion:MTLRegionMake2D(0, 0, lw, lh)
                 mipmapLevel:level
                       slice:0
                   withBytes:mip_linear.data()
                 bytesPerRow:lbw * bpb
               bytesPerImage:0];
    }
  }

  return FinalizeTexture(key, fetch, (__bridge void*)texture,
                         uint64_t(plan.host_format), is_cube, host_levels,
                         base, watched_base_bytes,
                         entry_mip_bytes ? mip_addr : 0, entry_mip_bytes);
}

// The SINGLE finalize tail for both the CPU-untile path and the slice-4
// routed GPU-load path: swizzle view, bindless slot, trace-read record and
// cache entry are identical regardless of which path produced the bytes —
// so trace ranges and cache/lifetime semantics cannot diverge by route.
// texture_bridge is an id<MTLTexture> at +0 (caller holds a strong ref
// through the call).
uint32_t MetalTextureCache::FinalizeTexture(
    uint64_t key, const xenos::xe_gpu_texture_fetch_t& fetch,
    void* texture_bridge, uint64_t host_format, bool is_cube,
    uint32_t host_levels, uint32_t base, uint32_t guest_bytes,
    uint32_t mip_base, uint32_t mip_bytes) {
  id<MTLTexture> texture = (__bridge id<MTLTexture>)texture_bridge;
  // The texture unit applies the fetch-constant swizzle (X/Y/Z/W/0/1 per output
  // channel) when sampling. Bake it into a swizzled texture VIEW so the shader
  // reads the correct channels (matches the Vulkan backend's host_swizzle). A
  // wrong swizzle here reads alpha from the wrong channel, and alpha-test then
  // discards the whole draw.
  uint32_t guest_swizzle = fetch.swizzle & 0xFFF;
  id<MTLTexture> sampled = texture;
  if (!GuestSwizzleIsIdentity(guest_swizzle)) {
    MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
        GuestSwizzleComponentToMetal((guest_swizzle >> 0) & 0b111),
        GuestSwizzleComponentToMetal((guest_swizzle >> 3) & 0b111),
        GuestSwizzleComponentToMetal((guest_swizzle >> 6) & 0b111),
        GuestSwizzleComponentToMetal((guest_swizzle >> 9) & 0b111));
    id<MTLTexture> view = [texture
        newTextureViewWithPixelFormat:MTLPixelFormat(host_format)
                          textureType:is_cube ? MTLTextureTypeCubeArray
                                              : MTLTextureType2DArray
                               levels:NSMakeRange(0, host_levels)
                               slices:NSMakeRange(0, is_cube ? 6 : 1)
                              swizzle:channels];
    if (view) {
      sampled = view;
    }
  }

  uint32_t slot = AllocateSlot();
  auto* entries = static_cast<IRDescriptorTableEntry*>(view_heap_contents_);
  IRDescriptorTableSetTexture(&entries[slot], sampled, 0.0f, 0);
  // Retain the sampled view (or the base texture if identity) for the descriptor
  // and residency. A view keeps its base alive, so this covers both.
  void* texture_ptr = (__bridge_retained void*)sampled;
  ++upload_count_;
  // Frame-trace recording: a replay must restore this texture's guest bytes
  // before the fetch that triggered this upload. No-op unless tracing.
  if (trace_writer_) {
    trace_writer_->WriteMemoryRead(base, guest_bytes);
    if (mip_bytes) {
      trace_writer_->WriteMemoryRead(mip_base, mip_bytes);
    }
  }
  cache_.emplace(key, Entry{texture_ptr, slot, base, guest_bytes,
                            mip_base, mip_bytes});
  active_textures_.push_back(texture_ptr);
  return slot;
}

void MetalTextureCache::WriteAllCachedTexturesToTrace(
    TraceWriter& trace_writer) const {
  for (const auto& pair : cache_) {
    const Entry& entry = pair.second;
    if (entry.guest_bytes) {
      trace_writer.WriteMemoryRead(entry.guest_base, entry.guest_bytes);
    }
    if (entry.mip_bytes) {
      trace_writer.WriteMemoryRead(entry.mip_base, entry.mip_bytes);
    }
  }
}

void MetalTextureCache::RegisterResolvedTarget(uint32_t base, void* texture,
                                               uint32_t width, uint32_t height,
                                               bool is_depth) {
  if (!texture) {
    return;
  }
  auto* entries = static_cast<IRDescriptorTableEntry*>(view_heap_contents_);
  auto it = resolved_targets_.find(base);
  if (it != resolved_targets_.end()) {
    // A new snapshot generation or extent invalidates the exact-cover
    // state: RAM equivalence must be re-established by the next writeback.
    if (it->second.texture != texture || it->second.width != width ||
        it->second.height != height) {
      it->second.full_rect = false;
    }
    it->second.width = width;
    it->second.height = height;
    if (it->second.texture != texture) {
      // FAULT INJECTOR (XE_METAL_FAULT_INJECT=1): restore the old in-place
      // repoint, which reproduces the GPU page fault deliberately — used to
      // verify fault attribution (S2) and auto-recovery (S6) end-to-end.
      static const bool s_fault_inject =
          std::getenv("XE_METAL_FAULT_INJECT") != nullptr;
      if (s_fault_inject) {
        it->second.texture = texture;
        IRDescriptorTableSetTexture(&entries[it->second.slot],
                                    (__bridge id<MTLTexture>)texture, 0.0f, 0);
        return;
      }
      // The snapshot texture was recreated (grew/reformatted). DON'T repoint the
      // existing slot in place: draws already encoded this frame reference it by
      // gpuAddress and made the OLD texture resident on their (now closed)
      // encoders. Repointing makes them sample the new texture, which is NOT
      // resident there -> GPU page fault (the "GPU froze" hang), and even when
      // resident, the wrong generation (the tile stipple). Instead give the new
      // generation a FRESH slot; the old slot keeps pointing at the old texture
      // for the in-flight draws and is reclaimed after the frame drains.
      uint32_t new_slot = AllocateSlot();
      if (!new_slot) {
        return;  // heap exhausted; keep the old slot rather than corrupt.
      }
      stale_slots_.emplace_back(RetireSubmission(), it->second.slot);
      IRDescriptorTableSetTexture(&entries[new_slot],
                                  (__bridge id<MTLTexture>)texture, 0.0f, 0);
      it->second.texture = texture;
      it->second.slot = new_slot;
    }
    return;
  }
  uint32_t slot = AllocateSlot();
  if (!slot) {
    return;
  }
  id<MTLTexture> tex = (__bridge id<MTLTexture>)texture;
  IRDescriptorTableSetTexture(&entries[slot], tex, 0.0f, 0);
  resolved_targets_[base] = ResolvedTarget{texture, width, height, slot, is_depth};
  XELOGI("MetalTex: resolved RT registered base={:08X} {}x{} -> slot {}", base,
         width, height, slot);
}

void MetalTextureCache::RegisterResolvedTipView(uint32_t base, void* texture) {
  if (!texture) {
    return;
  }
  auto it = resolved_targets_.find(base);
  if (it == resolved_targets_.end() || it->second.tip_texture == texture) {
    return;  // no registered target yet, or stable object already slotted.
  }
  uint32_t new_slot = AllocateSlot();
  if (!new_slot) {
    return;  // heap exhausted; keep serving the full snapshot.
  }
  if (it->second.tip_slot) {
    // Same page-fault rule as RegisterResolvedTarget: in-flight draws hold the
    // old texture by gpuAddress — retire the slot, never repoint in place.
    stale_slots_.emplace_back(RetireSubmission(), it->second.tip_slot);
  }
  auto* entries = static_cast<IRDescriptorTableEntry*>(view_heap_contents_);
  IRDescriptorTableSetTexture(&entries[new_slot],
                              (__bridge id<MTLTexture>)texture, 0.0f, 0);
  it->second.tip_texture = texture;
  it->second.tip_slot = new_slot;
  XELOGI("MetalTex: tip view registered base={:08X} -> slot {}", base,
         new_slot);
}

bool MetalTextureCache::BridgeServes(uint32_t base, uint32_t width,
                                     uint32_t height) const {
  auto it = resolved_targets_.find(base);
  if (it == resolved_targets_.end() || !it->second.texture) {
    return false;
  }
  const ResolvedTarget& rt = it->second;
  if (rt.is_depth || !rt.ram_backed) {
    return true;
  }
  return rt.width >= width && rt.height >= height &&
         ((__bridge id<MTLTexture>)rt.texture).pixelFormat ==
             MTLPixelFormatR32Float;
}

void MetalTextureCache::MarkResolvedTargetRamBacked(uint32_t base,
                                                    uint32_t wb_width,
                                                    uint32_t wb_height,
                                                    bool full_rect_geometry,
                                                    uint32_t wb_endian) {
  auto it = resolved_targets_.find(base);
  if (it == resolved_targets_.end()) {
    return;
  }
  if (!it->second.ram_backed) {
    it->second.ram_backed = true;
    XELOGI("MetalTex: resolved RT base={:08X} now RAM-backed (bridge retired)",
           base);
  }
  // The newest writeback governs: a full-extent write makes RAM equal to the
  // snapshot regardless of history; anything partial/offset means RAM is a
  // composition the snapshot cannot represent.
  it->second.full_rect = full_rect_geometry &&
                         wb_width == it->second.width &&
                         wb_height == it->second.height;
  it->second.wb_endian = wb_endian;
}

void MetalTextureCache::AppendAllTextures(std::vector<void*>& out) const {
  for (const auto& pair : cache_) {
    if (pair.second.texture) {
      out.push_back(pair.second.texture);
    }
  }
  for (const auto& pair : resolved_targets_) {
    if (pair.second.texture) {
      out.push_back(pair.second.texture);
    }
    if (pair.second.tip_texture) {
      out.push_back(pair.second.tip_texture);
    }
  }
}

uint32_t MetalTextureCache::AllocateSlot() {
  if (!free_slots_.empty()) {
    uint32_t slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
  }
  if (next_slot_ >= view_heap_slots_) {
    // Exhaustion was SILENT here while consumers degraded to stale/null
    // textures (the zero-exposure black flicker) — scream, throttled.
    static uint32_t s_exhausted_logs = 0;
    if (s_exhausted_logs++ < 8 || (s_exhausted_logs & 1023) == 0) {
      XELOGE("MetalTextureCache: BINDLESS HEAP EXHAUSTED ({} slots, {} stale)",
             view_heap_slots_, uint32_t(stale_slots_.size()));
    }
    return 0;  // exhausted (0 == null texture).
  }
  return next_slot_++;
}

void MetalTextureCache::ReclaimStaleSlots(uint64_t completed_submission) {
  // Free ONLY entries whose last-referencing submission has completed —
  // later in-flight submissions still read the rest (freeing them was the
  // async-S8 page-fault flicker). Descriptors are overwritten on the next
  // AllocateSlot that hands a freed slot out.
  auto slot_it = std::remove_if(
      stale_slots_.begin(), stale_slots_.end(),
      [&](const std::pair<uint64_t, uint32_t>& entry) {
        if (entry.first > completed_submission) {
          return false;
        }
        free_slots_.push_back(entry.second);
        return true;
      });
  stale_slots_.erase(slot_it, stale_slots_.end());
  auto tex_it = std::remove_if(
      retired_textures_.begin(), retired_textures_.end(),
      [&](const std::pair<uint64_t, void*>& entry) {
        if (entry.first > completed_submission) {
          return false;
        }
        id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)entry.second;
        (void)released;
        return true;
      });
  retired_textures_.erase(tex_it, retired_textures_.end());
}

uint32_t MetalTextureCache::InvalidateRange(uint32_t base, uint32_t length) {
  uint32_t evicted = 0;
  uint32_t end = base + length;
  for (auto it = cache_.begin(); it != cache_.end();) {
    const Entry& e = it->second;
    if (e.guest_base < end && e.guest_base + e.guest_bytes > base) {
      // Retire, don't free: in-flight draws may still reference the texture
      // (bindless) and the slot; both drain once the submissions encoded
      // against them complete.
      retired_textures_.emplace_back(RetireSubmission(), e.texture);
      stale_slots_.emplace_back(RetireSubmission(), e.slot);
      it = cache_.erase(it);
      ++evicted;
    } else {
      ++it;
    }
  }
  return evicted;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
