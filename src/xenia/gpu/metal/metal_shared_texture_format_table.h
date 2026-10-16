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
  // 16bpp 5:6:5 expanded to opaque RGBA8 (texture_load_r5g6b5_r8g8b8a8), for
  // GPUs without B5G6R5Unorm. Apple GPUs keep 5:6:5 on the legacy cache.
  k16bpbR5G6B5ToRgba8,
  // CTX1 (Xenos two-channel normal-map compression, 8-byte 4x4 blocks)
  // decoded to RG8 texels (texture_load_ctx1). The legacy cache decodes it on
  // the CPU at the base level only.
  kCtx1ToRg8,
  // 32bpp packed 11:11:10 / 10:11:11 unorm expanded to RGBA16 (the upstream
  // texture_load_r11g11b10_rgba16 / r10g11b11 shaders): the host block is 8
  // bytes, twice the guest's. Signed bindings take an RGBA16Snorm view of
  // the same bytes (the exact snorm decode is a separate shader the view
  // cannot express; the HDR chains that use these formats fetch unsigned).
  kR11G11B10ToRgba16,
  kR10G11B11ToRgba16,
  // DXT3A (the DXT3 alpha block alone, 8-byte 4x4 blocks) expanded to R8
  // texels (texture_load_dxt3a), as the legacy CPU decode does at the base
  // level only.
  kDxt3aToR8,
  // 16-byte 4x4 block copy (texture_load_128bpb) for DXT3 / DXT5 / DXN kept
  // compressed on the host (BC2 / BC3 / BC5); DXT1 / DXT5A take k64bpb
  // (BC1 / BC4). 2026-10-16: Rage's 16 MB linear DXT5 megatexture page cache
  // was a CPU untile every frame.
  k128bpb,
  // 32-bit depth texels (k_24_8 unorm24 / k_24_8_FLOAT 20e4) converted to
  // R32Float on the GPU (texture_load_depth_unorm / _float, the same rule the
  // legacy CPU decode implements). 2026-10-16: Rage's 768x768 24_8_FLOAT
  // shadow maps were 5 ms/frame of CPU decode.
  kDepthUnormToR32,
  kDepthFloatToR32,
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
  kMtlPixelFormatB5G6R5Unorm = 40,
  kMtlPixelFormatABGR4Unorm = 42,
  kMtlPixelFormatR8Snorm = 12,
  kMtlPixelFormatR16Unorm = 20,
  kMtlPixelFormatR16Snorm = 22,
  kMtlPixelFormatR16Float = 25,
  kMtlPixelFormatRG8Unorm = 30,
  kMtlPixelFormatRG8Snorm = 32,
  kMtlPixelFormatR32Float = 55,
  kMtlPixelFormatRG32Float = 105,
  kMtlPixelFormatRG16Unorm = 60,
  kMtlPixelFormatRG16Snorm = 62,
  kMtlPixelFormatRG16Float = 65,
  kMtlPixelFormatRGBA8Unorm = 70,
  kMtlPixelFormatRGBA8Snorm = 72,
  kMtlPixelFormatRGB10A2Unorm = 90,
  kMtlPixelFormatRGBA16Unorm = 110,
  kMtlPixelFormatRGBA16Snorm = 112,
  kMtlPixelFormatRGBA16Float = 115,
  kMtlPixelFormatBC1_RGBA = 130,
  kMtlPixelFormatBC2_RGBA = 132,
  kMtlPixelFormatBC3_RGBA = 134,
  kMtlPixelFormatBC4_RUnorm = 140,
  kMtlPixelFormatBC4_RSnorm = 141,
  kMtlPixelFormatBC5_RGUnorm = 142,
  kMtlPixelFormatBC5_RGSnorm = 143,
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
      // Two-channel 32-bit float (Tomb Raider's 1x1 luminance readback; off
      // the route its draw was a RAM consumer of the resolve: commit + wait).
      {xenos::TextureFormat::k_32_32_FLOAT, SharedLoadFamily::k64bpb,
       kMtlPixelFormatRG32Float, kMtlPixelFormatRG32Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 8, false},
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
      // 5:6:5 expanded to RGBA8 on the GPU where B5G6R5Unorm does not exist
      // (Intel/AMD). The legacy CPU decode served it at the base level only
      // (Skate 2 on the Mac Pro). The shader writes guest R/G/B to host R/G/B
      // with alpha 1; the shared table reads alpha from blue (RGBB), as the
      // legacy decode does. Selected only without packed16.
      // XE_METAL_TEX_ROUTE_565=0 drops this entry (legacy CPU decode).
      {xenos::TextureFormat::k_5_6_5, SharedLoadFamily::k16bpbR5G6B5ToRgba8,
       kMtlPixelFormatRGBA8Unorm, kMtlPixelFormatRGBA8Unorm,
       XE_GPU_MAKE_TEXTURE_SWIZZLE(R, G, B, B), 2, false},
      // 5:6:5 native (Apple, packed16), 2026-10-16: the raw 16-bit copy into
      // B5G6R5Unorm with the legacy plan's host swizzle (guest R sits in the
      // low bits, Metal's B field; alpha from the high field, as the legacy
      // CPU path reads it), so routed and legacy serve the same bits. Until
      // now Apple GPUs kept 5:6:5 on the CPU (Rage: two linear mip chains,
      // 6.7 ms/frame). Selected only with packed16; XE_METAL_TEX_ROUTE_565=0
      // drops it.
      {xenos::TextureFormat::k_5_6_5, SharedLoadFamily::k16bpb,
       kMtlPixelFormatB5G6R5Unorm, kMtlPixelFormatB5G6R5Unorm,
       XE_GPU_MAKE_TEXTURE_SWIZZLE(B, G, R, R), 2, false},
      // CTX1 decoded to RG8 on the GPU, mips included (Halo 3 materials: 134
      // mipped uploads in a 40 s sweep served base-only by the legacy CPU
      // decode, on every GPU). RGGG like the legacy plan; no snorm sibling,
      // as in the legacy plan. XE_METAL_TEX_ROUTE_CTX1=0 drops this entry.
      {xenos::TextureFormat::k_CTX1, SharedLoadFamily::kCtx1ToRg8,
       kMtlPixelFormatRG8Unorm, kMtlPixelFormatRG8Unorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 8, false},
      // Packed 11:11:10 / 10:11:11 (resolve-backed HDR chains: Tomb Raider's
      // 320x180 / 80x48 / 20x12 downsamples) expanded to RGBA16 on the GPU.
      // Off the route these are RAM consumers of the resolve, and every draw
      // sampling one paid a commit + GPU wait (58 drains, 25 ms a frame).
      // XE_METAL_TEX_ROUTE_1110=0 drops these entries.
      {xenos::TextureFormat::k_11_11_10_AS_16_16_16_16,
       SharedLoadFamily::kR11G11B10ToRgba16, kMtlPixelFormatRGBA16Unorm,
       kMtlPixelFormatRGBA16Snorm, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 4,
       false},
      {xenos::TextureFormat::k_10_11_11_AS_16_16_16_16,
       SharedLoadFamily::kR10G11B11ToRgba16, kMtlPixelFormatRGBA16Unorm,
       kMtlPixelFormatRGBA16Snorm, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 4,
       false},
      // DXT3A expanded to R8 on the GPU, mips included (Halo 3, Viva Pinata).
      // RRRR like the legacy plan; no snorm sibling.
      // XE_METAL_TEX_ROUTE_DXT3A=0 drops this entry.
      {xenos::TextureFormat::k_DXT3A, SharedLoadFamily::kDxt3aToR8,
       kMtlPixelFormatR8Unorm, kMtlPixelFormatR8Unorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 8, false},
      // Block-compressed formats kept compressed on the host (BC1-BC5, native
      // on Apple GPUs; the legacy plan uploads the same blocks after a CPU
      // untile). 8- or 16-byte block copies with the 64bpb / 128bpb loaders;
      // host swizzles as the legacy plan (DXT5A RRRR, DXN RGGG with a snorm
      // sibling). Selected only where the device reports BC support;
      // XE_METAL_TEX_ROUTE_BC=0 drops them. 2026-10-16, Rage.
      {xenos::TextureFormat::k_DXT1, SharedLoadFamily::k64bpb,
       kMtlPixelFormatBC1_RGBA, kMtlPixelFormatBC1_RGBA,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 8, false},
      {xenos::TextureFormat::k_DXT1_AS_16_16_16_16, SharedLoadFamily::k64bpb,
       kMtlPixelFormatBC1_RGBA, kMtlPixelFormatBC1_RGBA,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 8, false},
      {xenos::TextureFormat::k_DXT2_3, SharedLoadFamily::k128bpb,
       kMtlPixelFormatBC2_RGBA, kMtlPixelFormatBC2_RGBA,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 16, false},
      {xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16, SharedLoadFamily::k128bpb,
       kMtlPixelFormatBC2_RGBA, kMtlPixelFormatBC2_RGBA,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 16, false},
      {xenos::TextureFormat::k_DXT4_5, SharedLoadFamily::k128bpb,
       kMtlPixelFormatBC3_RGBA, kMtlPixelFormatBC3_RGBA,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 16, false},
      {xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16, SharedLoadFamily::k128bpb,
       kMtlPixelFormatBC3_RGBA, kMtlPixelFormatBC3_RGBA,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA, 16, false},
      {xenos::TextureFormat::k_DXT5A, SharedLoadFamily::k64bpb,
       kMtlPixelFormatBC4_RUnorm, kMtlPixelFormatBC4_RUnorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 8, false},
      {xenos::TextureFormat::k_DXN, SharedLoadFamily::k128bpb,
       kMtlPixelFormatBC5_RGUnorm, kMtlPixelFormatBC5_RGSnorm,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG, 16, false},
      // Depth texels to R32Float on the GPU (RRRR like the legacy plan).
      // XE_METAL_TEX_ROUTE_DEPTH=0 drops both.
      {xenos::TextureFormat::k_24_8, SharedLoadFamily::kDepthUnormToR32,
       kMtlPixelFormatR32Float, kMtlPixelFormatR32Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 4, false},
      {xenos::TextureFormat::k_24_8_FLOAT, SharedLoadFamily::kDepthFloatToR32,
       kMtlPixelFormatR32Float, kMtlPixelFormatR32Float,
       xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR, 4, false},
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
// For the block-decoding families (CTX1, DXT3A) this is bytes per host TEXEL,
// the unit their load shaders write and the shared LoadShaderInfo reports.
inline uint32_t SharedTextureHostBytesPerBlock(
    const SharedTextureFormatEntry& entry) {
  switch (entry.load_shader) {
    case SharedLoadFamily::k16bpbRgba4ToRgba8:
    case SharedLoadFamily::k16bpbR5G6B5ToRgba8:
      return 4u;
    case SharedLoadFamily::kR11G11B10ToRgba16:
    case SharedLoadFamily::kR10G11B11ToRgba16:
      return 8u;
    case SharedLoadFamily::kCtx1ToRg8:
      return 2u;
    case SharedLoadFamily::kDxt3aToR8:
      return 1u;
    default:
      return entry.bytes_per_block;
  }
}

// The 4x4 block-compressed guest formats (decoded or kept compressed).
inline bool IsSharedTextureBlockCompressedFormat(xenos::TextureFormat format) {
  switch (format) {
    case xenos::TextureFormat::k_DXT1:
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT2_3:
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT4_5:
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
    case xenos::TextureFormat::k_DXT5A:
    case xenos::TextureFormat::k_DXN:
    case xenos::TextureFormat::k_CTX1:
    case xenos::TextureFormat::k_DXT3A:
      return true;
    default:
      return false;
  }
}

// Guest block edge in texels: 4 for the 4x4 block-compressed formats, 1 for
// everything else.
inline uint32_t SharedTextureBlockDim(const SharedTextureFormatEntry& entry) {
  return IsSharedTextureBlockCompressedFormat(entry.format) ? 4u : 1u;
}

// Host texels per guest block edge: the decoding families (CTX1, DXT3A)
// write 4x4 host texels per guest block; the BC-preserving families and
// every uncompressed format write one host block/texel per guest block.
inline uint32_t SharedTextureHostExpand(const SharedTextureFormatEntry& entry) {
  return entry.load_shader == SharedLoadFamily::kCtx1ToRg8 ||
                 entry.load_shader == SharedLoadFamily::kDxt3aToR8
             ? 4u
             : 1u;
}

// Whether the entry's host texture is a BC (compressed) pixel format: no
// shader-write usage, blits address block rows.
inline bool SharedTextureHostIsBc(const SharedTextureFormatEntry& entry) {
  return entry.host_format >= kMtlPixelFormatBC1_RGBA &&
         entry.host_format <= kMtlPixelFormatBC5_RGSnorm;
}

// BC texture compression support, recorded by the shared cache at
// initialization (MTLDevice supportsBCTextureCompression: every Apple
// Silicon Mac; not every Intel/AMD part). Entries with BC host formats are
// dropped without it.
inline std::atomic<bool>& SharedTextureBcSupported() {
  static std::atomic<bool> supported{false};
  return supported;
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
    if (table[i].format == xenos::TextureFormat::k_11_11_10_AS_16_16_16_16 ||
        table[i].format == xenos::TextureFormat::k_10_11_11_AS_16_16_16_16) {
      static const bool s_1110 = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_1110");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_1110) {
        return nullptr;
      }
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
    if (table[i].format == xenos::TextureFormat::k_CTX1) {
      static const bool s_ctx1 = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_CTX1");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_ctx1) {
        return nullptr;
      }
    }
    if (table[i].format == xenos::TextureFormat::k_DXT3A) {
      static const bool s_dxt3a = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_DXT3A");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_dxt3a) {
        return nullptr;
      }
    }
    if (table[i].format == xenos::TextureFormat::k_5_6_5) {
      static const bool s_565 = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_565");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_565) {
        return nullptr;
      }
      // Two 5:6:5 entries: the native B5G6R5 one with packed16, the RGBA8
      // expansion without.
      const bool packed16 =
          SharedTexturePacked16Supported().load(std::memory_order_relaxed);
      const bool native = table[i].load_shader == SharedLoadFamily::k16bpb;
      if (packed16 != native) {
        continue;
      }
    }
    if (table[i].load_shader == SharedLoadFamily::kDepthUnormToR32 ||
        table[i].load_shader == SharedLoadFamily::kDepthFloatToR32) {
      static const bool s_depth = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_DEPTH");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_depth) {
        return nullptr;
      }
    }
    if (SharedTextureHostIsBc(table[i])) {
      static const bool s_bc = [] {
        const char* v = std::getenv("XE_METAL_TEX_ROUTE_BC");
        return !(v && v[0] == '0' && !v[1]);
      }();
      if (!s_bc ||
          !SharedTextureBcSupported().load(std::memory_order_relaxed)) {
        return nullptr;
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
