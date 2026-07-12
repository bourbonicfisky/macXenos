/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_texture_cache.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <vector>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/texture_address.h"
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
                                     void* view_heap, uint32_t view_heap_slots)
    : device_(device),
      memory_(memory),
      view_heap_(view_heap),
      view_heap_slots_(view_heap_slots) {
  id<MTLBuffer> heap = (__bridge id<MTLBuffer>)view_heap_;
  view_heap_contents_ = [heap contents];
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_;
  bc_supported_ = [mtl_device supportsBCTextureCompression];
  XELOGI("MetalTextureCache: BC (DXT) texture compression {}",
         bc_supported_ ? "supported" : "UNSUPPORTED (DXT stays gray)");
}

MetalTextureCache::~MetalTextureCache() {
  for (auto& pair : cache_) {
    if (pair.second.texture) {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)pair.second.texture;
      (void)released;
    }
  }
  cache_.clear();
}

uint32_t MetalTextureCache::GetTextureSlot(
    const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed) {
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
  // RT-as-texture bridge — STAGE 3: served for DEPTH only (no guest-RAM
  // encoding of host depth yet). COLOR fetches read guest RAM, which resolves
  // now write back at every size: the bridge's per-base snapshot cannot
  // represent base reuse, strip assembly, or temporal (previous-frame) reads
  // — JC2's final composite reads LAST frame's image and got the current
  // frame's G-buffer strip + garbage instead (the stipple/darkness).
  // XE_METAL_COLOR_BRIDGE=1 restores the old behavior for A/B.
  {
    static const bool s_stage3 = std::getenv("XE_METAL_STAGE3") != nullptr;
    static const bool s_color_bridge =
        !s_stage3 || std::getenv("XE_METAL_COLOR_BRIDGE") != nullptr;
    auto rit = resolved_targets_.find(fetch.base_address << 12);
    if (rit != resolved_targets_.end() && rit->second.texture &&
        (rit->second.is_depth || s_color_bridge)) {
      active_textures_.push_back(rit->second.texture);
      return rit->second.slot;
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

  struct KeyBytes {
    uint32_t base, width, height, pitch;
    uint32_t format_tiled_endian;
  } kb{base, width, height, pitch_texels,
       (uint32_t(fetch.format) << 16) | (uint32_t(is_signed) << 15) |
           (uint32_t(fetch.dimension) << 12) | (uint32_t(fetch.tiled) << 8) |
           uint32_t(fetch.endianness)};
  uint64_t key = XXH3_64bits(&kb, sizeof(kb));
  auto it = cache_.find(key);
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

  // Untile / copy into a linear block-major staging buffer (host byte order).
  std::vector<uint8_t> linear(size_t(blocks_w) * blocks_h * bpb);
  for (uint32_t by = 0; by < blocks_h; ++by) {
    for (uint32_t bx = 0; bx < blocks_w; ++bx) {
      uint32_t src_off;
      if (tiled) {
        src_off = uint32_t(texture_address::Tiled2D(
            int32_t(bx), int32_t(by), pitch_blocks_aligned, plan.block_log2));
      } else {
        src_off = (by * pitch_blocks + bx) * bpb;
      }
      uint8_t* dst = &linear[(size_t(by) * blocks_w + bx) * bpb];
      if (src_off + bpb <= max_src_bytes) {
        std::memcpy(dst, guest + src_off, bpb);
        SwapEndian(dst, bpb, endian);
      }
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
  MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
  desc.textureType = is_cube ? MTLTextureTypeCubeArray : MTLTextureType2DArray;
  desc.pixelFormat = plan.host_format;
  desc.width = width;
  desc.height = height;
  desc.arrayLength = 1;  // For cube arrays this is the CUBE count (6 slices).
  desc.mipmapLevelCount = 1;
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
    uint32_t blocks_v_aligned = (blocks_h + 31u) & ~31u;
    face_stride = pitch_blocks_aligned * blocks_v_aligned * bpb;
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
        newTextureViewWithPixelFormat:plan.host_format
                          textureType:is_cube ? MTLTextureTypeCubeArray
                                              : MTLTextureType2DArray
                               levels:NSMakeRange(0, 1)
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
  // Guest byte extent (conservative: aligned pitch over all block rows; all
  // 6 face slices for cubes) for resolve-writeback invalidation.
  uint32_t guest_bytes = uint32_t(std::min<size_t>(
      is_cube ? size_t(face_stride) * 6
              : size_t(pitch_blocks_aligned) * blocks_h * bpb,
      max_src_bytes));
  ++upload_count_;
  cache_.emplace(key, Entry{texture_ptr, slot, base, guest_bytes});
  active_textures_.push_back(texture_ptr);
  return slot;
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
      stale_slots_.push_back(it->second.slot);
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
  }
}

uint32_t MetalTextureCache::AllocateSlot() {
  if (!free_slots_.empty()) {
    uint32_t slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
  }
  if (next_slot_ >= view_heap_slots_) {
    return 0;  // exhausted (0 == null texture).
  }
  return next_slot_++;
}

void MetalTextureCache::ReclaimStaleSlots() {
  // The retired slots' in-flight draws have drained (caller waited on the frame
  // command buffer), so the slots can be reused. Their descriptors are
  // overwritten on the next AllocateSlot that hands them out.
  free_slots_.insert(free_slots_.end(), stale_slots_.begin(), stale_slots_.end());
  stale_slots_.clear();
  for (void* tex : retired_textures_) {
    id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)tex;
    (void)released;
  }
  retired_textures_.clear();
}

uint32_t MetalTextureCache::InvalidateRange(uint32_t base, uint32_t length) {
  uint32_t evicted = 0;
  uint32_t end = base + length;
  for (auto it = cache_.begin(); it != cache_.end();) {
    const Entry& e = it->second;
    if (e.guest_base < end && e.guest_base + e.guest_bytes > base) {
      // Retire, don't free: in-flight draws may still reference the texture
      // (bindless) and the slot; both drain after the next commit + wait.
      retired_textures_.push_back(e.texture);
      stale_slots_.push_back(e.slot);
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
