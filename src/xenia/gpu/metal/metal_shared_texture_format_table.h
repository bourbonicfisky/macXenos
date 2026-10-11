/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHARED_TEXTURE_FORMAT_TABLE_H_
#define XENIA_GPU_METAL_METAL_SHARED_TEXTURE_FORMAT_TABLE_H_

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Per-guest-format routing entry for the shared-buffer GPU texture load route
// (optimize-metal-gpu-resident-resolve-coherence, stage 1). One row per
// xenos::TextureFormat the route serves; anything not listed stays on the
// legacy CPU cache.
//
// Host formats, swizzles and signed views mirror the legacy cache's
// PlanForFormat (the byte oracle every routed class is proven against) and
// upstream D3D12's host_formats_ table. Values are MTLPixelFormat raw values
// so the header stays Objective-C free; 0 (MTLPixelFormatInvalid) = none.
// Upstream texture-load family (bit-copy families only in tranche 1); the
// shared cache maps these onto TextureCache::LoadShaderIndex, which is a
// protected member of the base class.
enum class SharedLoadFamily : uint32_t {
  k8bpb,
  k16bpb,
  k32bpb,
  k64bpb,
  // 16bpp 4:4:4:4 expanded to RGBA8 (texture_load_r4g4b4a4_r8g8b8a8): the
  // host block is 4 bytes, twice the guest's. For GPUs without ABGR4Unorm.
  k16bpbRgba4ToRgba8,
};

struct SharedTextureFormatEntry {
  xenos::TextureFormat format;
  // Load shader family (non-scaled); the direct RGBA8 pipeline is selected by
  // `direct` instead.
  SharedLoadFamily load_shader;
  uint32_t host_format;         // MTLPixelFormat for unsigned/float bindings.
  uint32_t host_format_signed;  // MTLPixelFormat view for signed bindings, or 0.
  uint32_t host_swizzle;        // xenos XE_GPU_TEXTURE_SWIZZLE_* composition.
  uint32_t bytes_per_block;     // Guest bytes per block (== host bytes
                                // except for the expanding families).
  bool direct;                  // texture_load_32bpb_direct (writes texture).
};

// MTLPixelFormat raw values (Metal/MTLPixelFormat.h), duplicated here so the
// table is usable from C++ translation units and tests.
enum : uint32_t {
  kMtlPixelFormatInvalid = 0,
  kMtlPixelFormatR8Unorm = 10,
  kMtlPixelFormatABGR4Unorm = 42,
  kMtlPixelFormatR8Snorm = 12,
  kMtlPixelFormatR16Unorm = 20,
  kMtlPixelFormatR16Snorm = 22,
  kMtlPixelFormatR16Float = 25,
  kMtlPixelFormatRG8Unorm = 30,
  kMtlPixelFormatRG8Snorm = 32,
  kMtlPixelFormatR32Float = 55,
  kMtlPixelFormatRG16Unorm = 60,
  kMtlPixelFormatRG16Snorm = 62,
  kMtlPixelFormatRG16Float = 65,
  kMtlPixelFormatRGBA8Unorm = 70,
  kMtlPixelFormatRGBA8Snorm = 72,
  kMtlPixelFormatRGB10A2Unorm = 90,
  kMtlPixelFormatRGBA16Unorm = 110,
  kMtlPixelFormatRGBA16Snorm = 112,
  kMtlPixelFormatRGBA16Float = 115,
};

inline const SharedTextureFormatEntry* SharedTextureFormatTable(
    size_t* count_out) {
  static const SharedTextureFormatEntry kTable[] = {
      // The originally promoted class (C1T-R): direct pipeline.
      {xenos::TextureFormat::k_8_8_8_8, SharedLoadFamily::k32bpb,
       kMtlPixelFormatRGBA8Unorm, kMtlPixelFormatRGBA8Snorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 4, true},
      {xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16,
       SharedLoadFamily::k32bpb, kMtlPixelFormatRGBA8Unorm,
       kMtlPixelFormatRGBA8Snorm, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 4, true},
      // Tranche 1: bit-copy families (endian swap only), the resolve-producible
      // formats the untile timer named on JC2 / Sonic / DOA5.
      {xenos::TextureFormat::k_8, SharedLoadFamily::k8bpb,
       kMtlPixelFormatR8Unorm, kMtlPixelFormatR8Snorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 1, false},
      {xenos::TextureFormat::k_8_8, SharedLoadFamily::k16bpb,
       kMtlPixelFormatRG8Unorm, kMtlPixelFormatRG8Snorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 2, false},
      {xenos::TextureFormat::k_16, SharedLoadFamily::k16bpb,
       kMtlPixelFormatR16Unorm, kMtlPixelFormatR16Snorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 2, false},
      {xenos::TextureFormat::k_16_16, SharedLoadFamily::k32bpb,
       kMtlPixelFormatRG16Unorm, kMtlPixelFormatRG16Snorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 4, false},
      {xenos::TextureFormat::k_16_16_16_16, SharedLoadFamily::k64bpb,
       kMtlPixelFormatRGBA16Unorm, kMtlPixelFormatRGBA16Snorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 8, false},
      {xenos::TextureFormat::k_16_FLOAT, SharedLoadFamily::k16bpb,
       kMtlPixelFormatR16Float, kMtlPixelFormatR16Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 2, false},
      {xenos::TextureFormat::k_16_EXPAND, SharedLoadFamily::k16bpb,
       kMtlPixelFormatR16Float, kMtlPixelFormatR16Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 2, false},
      {xenos::TextureFormat::k_16_16_FLOAT, SharedLoadFamily::k32bpb,
       kMtlPixelFormatRG16Float, kMtlPixelFormatRG16Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 4, false},
      {xenos::TextureFormat::k_16_16_EXPAND,
       SharedLoadFamily::k32bpb, kMtlPixelFormatRG16Float,
       kMtlPixelFormatRG16Float, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 4, false},
      {xenos::TextureFormat::k_16_16_16_16_FLOAT,
       SharedLoadFamily::k64bpb, kMtlPixelFormatRGBA16Float,
       kMtlPixelFormatRGBA16Float, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 8,
       false},
      {xenos::TextureFormat::k_16_16_16_16_EXPAND,
       SharedLoadFamily::k64bpb, kMtlPixelFormatRGBA16Float,
       kMtlPixelFormatRGBA16Float, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 8,
       false},
      {xenos::TextureFormat::k_32_FLOAT, SharedLoadFamily::k32bpb,
       kMtlPixelFormatR32Float, kMtlPixelFormatR32Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 4, false},
      // Same bit layout as RGB10A2; no snorm sibling in Metal, a signed fetch
      // keeps the unorm view (legacy plan does the same).
      {xenos::TextureFormat::k_2_10_10_10, SharedLoadFamily::k32bpb,
       kMtlPixelFormatRGB10A2Unorm, kMtlPixelFormatRGB10A2Unorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 4, false},
      {xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16,
       SharedLoadFamily::k32bpb, kMtlPixelFormatRGB10A2Unorm,
       kMtlPixelFormatRGB10A2Unorm, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 4,
       false},
      // 4:4:4:4 (2026-10-04, Red Dead Redemption: 49 resolve-backed CPU
      // decodes per frame behind 26 blocking consumer waits at its title
      // scene). After the fetch-endian swap the guest word holds R in bits
      // 0-3 through A in bits 12-15 (the legacy decode_4444 oracle); Metal's
      // ABGR4Unorm reads A from bits 0-3 through R from bits 12-15 (device
      // probe), so the host swizzle reverses the channels. No snorm sibling.
      // XE_METAL_TEX_ROUTE_4444=0 drops this entry (legacy CPU decode).
      {xenos::TextureFormat::k_4_4_4_4, SharedLoadFamily::k16bpb,
       kMtlPixelFormatABGR4Unorm, kMtlPixelFormatABGR4Unorm,
       XE_GPU_MAKE_TEXTURE_SWIZZLE(A, B, G, R), 2, false},
      // The same texels expanded to RGBA8 on the GPU where ABGR4Unorm does
      // not exist (Intel/AMD): without it Red Dead's ~34 resolve-backed
      // 4:4:4:4 fetches a frame fell back to blocking CPU decodes (Mac Pro
      // cinematic 2.7 fps, 2026-10-10). Selected only without packed16.
      {xenos::TextureFormat::k_4_4_4_4, SharedLoadFamily::k16bpbRgba4ToRgba8,
       kMtlPixelFormatRGBA8Unorm, kMtlPixelFormatRGBA8Unorm,
       // The shader's nibble order (R = bits 4-7, G = 0-3, B = 12-15,
       // A = 8-11 of the swapped word) to guest X/Y/Z/W = bits 0-3 ... 12-15,
       // the order the ABGR4 entry and the legacy decode use.
       XE_GPU_MAKE_TEXTURE_SWIZZLE(G, R, A, B), 2, false},
  };
  if (count_out) {
    *count_out = sizeof(kTable) / sizeof(kTable[0]);
  }
  return kTable;
}

// Packed 16-bit pixel formats (ABGR4Unorm and the 5:6:5 / 1:5:5:5 family)
// exist only on Apple GPU families: on an Intel/AMD Mac a texture descriptor
// with one fails validation (Red Dead on the RX 6900 XT aborted at its first
// 4:4:4:4 texture). The shared texture cache records the device's answer at
// initialization; entries that need the formats are dropped without it.
// Host bytes per block of an entry's textures.
inline uint32_t SharedTextureHostBytesPerBlock(
    const SharedTextureFormatEntry& entry) {
  return entry.load_shader == SharedLoadFamily::k16bpbRgba4ToRgba8
             ? 4u
             : entry.bytes_per_block;
}

// Mipped 4:4:4:4 keys take the RGBA4 -> RGBA8 expansion (non-Apple GPUs):
// on three Red Dead traces it is pixel-identical to the ABGR4 route, while the
// legacy CPU decode (base level only) differs by up to 139. An earlier
// level-1 mismatch against guest memory was a bad oracle, not the shader.
// XE_METAL_RGBA4_EXPAND_MIPS=0 returns mipped keys to the legacy decode.
inline bool SharedRgba4ExpandMipsEnabled() {
  static const bool s_enabled = [] {
    const char* v = std::getenv("XE_METAL_RGBA4_EXPAND_MIPS");
    return !(v && v[0] == '0');
  }();
  return s_enabled;
}

inline std::atomic<bool>& SharedTexturePacked16Supported() {
  static std::atomic<bool> supported{true};
  return supported;
}

// XE_METAL_TEX_ROUTE_FORMATS=0 restricts the table to the originally promoted
// RGBA8 class (exact rollback of the stage-1 widening).
inline bool SharedTextureRouteFormatsWidened() {
  static const bool s_widened = [] {
    const char* v = std::getenv("XE_METAL_TEX_ROUTE_FORMATS");
    return !(v && v[0] == '0' && !v[1]);
  }();
  return s_widened;
}

inline const SharedTextureFormatEntry* FindSharedTextureFormat(
    xenos::TextureFormat format) {
  size_t count = 0;
  const SharedTextureFormatEntry* table = SharedTextureFormatTable(&count);
  for (size_t i = 0; i < count; ++i) {
    if (table[i].format != format) {
      continue;
    }
    if (!table[i].direct && !SharedTextureRouteFormatsWidened()) {
      return nullptr;
    }
    if (table[i].format == xenos::TextureFormat::k_4_4_4_4) {
      static const bool s_4444 = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_4444");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_4444) {
        return nullptr;
      }
      // ABGR4Unorm with packed16, the RGBA8 expansion entry without.
      const bool packed16 =
          SharedTexturePacked16Supported().load(std::memory_order_relaxed);
      const bool expands =
          table[i].load_shader == SharedLoadFamily::k16bpbRgba4ToRgba8;
      if (packed16 == expands) {
        continue;
      }
    }
    return &table[i];
  }
  return nullptr;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHARED_TEXTURE_FORMAT_TABLE_H_
