/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_EDRAM_MSL_H_
#define XENIA_GPU_METAL_METAL_EDRAM_MSL_H_

namespace xe {
namespace gpu {
namespace metal {

// The canonical color/depth/stencil EDRAM materialize/pack MSL, extracted as a
// SHARED production source. It is compiled verbatim by BOTH the conformance harness
// (metal_kernel_conformance_test.mm, which appends its extra kernels after it)
// and the live render-target cache's cross-MSAA path, so the tested helpers and
// the live path are byte-identical and cannot drift -- the ladder's core
// guarantee. Every function here is proven by the [l1][metal][...][depth] cases:
// float24 over the full [0,2) via *0.5//*2, unorm24 exact incl. near-1.0, the
// complete (d24<<8)|stencil word via bit-plane stencil, per-sample 4x, and the
// mixed 4x->1x->4x cross-MSAA chain.
//
// Self-contained (has the metal_stdlib prelude, the shared addresser, and
// SurfaceParams) so it compiles as a standalone MTLLibrary for the live path.
//
// SurfaceParams.is_depth encodes BOTH depth-ness and the guest format for the
// depth kernels: 2 = float24 (kD24FS8, [0,2) via *0.5//*2), 1 = unorm24
// (kD24S8, [0,1]); any nonzero value triggers the 40-column depth swap in the
// addresser. Color kernels pass is_depth = 0.
static constexpr const char* kEdramMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline uint2 xe_edram_msaa_shifts(uint msaa) {
  return uint2(msaa >= 2u ? 1u : 0u, msaa >= 1u ? 1u : 0u);
}
// Metal's native sample indices use the D3D10-standard position order, while
// Xenos EDRAM sample slots follow the sub-pixel storage order consumed by
// xe_edram_offset_bytes. Convert only at the real multisample attachment
// boundary: native 2x bottom/top -> guest top/bottom, and native 4x
// TL/TR/BL/BR -> guest TL/BL/TR/BR. Canonical and array-backed callers already
// carry guest sample indices and must not use this conversion.
inline uint xe_metal_sample_to_guest(uint msaa, uint sample) {
  if (msaa == 1u) {
    return sample ^ 1u;
  }
  if (msaa == 2u) {
    return sample == 1u ? 2u : (sample == 2u ? 1u : sample);
  }
  return sample;
}
inline uint xe_edram_depth_swap_x(uint x, uint tile_width) {
  uint half_width = tile_width >> 1u;
  return x >= half_width ? x - half_width : x + half_width;
}

// Shared addresser: edram.xesli XeEdramOffsetBytes at resolution_scale=(1,1),
// wrap=false. One MSL definition used by BOTH the offset-parity kernel and the
// canonical pack/unpack kernels, so they cannot drift. msaa: 0=1x,1=2x,2=4x.
inline uint xe_edram_offset_bytes(uint px, uint py, uint base, uint pitch,
                                  uint msaa, uint is_depth, uint fmt_log2,
                                  uint sample) {
  uint2 s = uint2(px, py) << xe_edram_msaa_shifts(msaa);
  s += (uint2(sample, sample) >> uint2(1u, 0u)) & 1u;
  uint2 tsize32 = uint2(80u, 16u);
  uint2 tsize = tsize32 >> uint2(fmt_log2, 0u);
  uint2 toff = s / tsize;
  uint b = base + toff.y * pitch + toff.x;
  s -= toff * tsize;
  if (is_depth != 0u) {
    s.x = xe_edram_depth_swap_x(s.x, tsize.x);
  }
  uint address = b * (tsize32.x * tsize32.y) +
                 ((s.y * tsize.x + s.x) << fmt_log2);
  return address << 2;
}

// Per-surface constants for pack/unpack (one surface per dispatch). `words` is
// the canonical buffer size in 32-bit words, so a bounds guard can skip an
// over-dispatch instead of reading/writing out of range.
struct SurfaceParams { uint base, pitch, msaa, is_depth, fmt_log2, words; };

inline float4 xe_unpack_8888(uint word) {
  return float4(word & 0xFFu, (word >> 8u) & 0xFFu,
                (word >> 16u) & 0xFFu, (word >> 24u) & 0xFFu) *
         (1.0 / 255.0);
}
inline uint xe_pack_8888(float4 color) {
  uint4 bytes = uint4(round(color * 255.0)) & 0xFFu;
  return bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u);
}

// Color storage classes used by the pinned Katamari + Red Dead corpus. Values
// are the Xenos ColorRenderTargetFormat enum after GetStorageColorFormat:
// 0 = 8_8_8_8 (gamma has the same raw storage), 3 = 2_10_10_10_FLOAT,
// 4 = 16_16 fixed, 14 = 32_FLOAT.
inline float xe_7e3_to_f32(uint value) {
  value &= 0x3FFu;
  if (value == 0u) return 0.0;
  uint mantissa = value & 0x7Fu;
  uint exponent = value >> 7u;
  if (exponent == 0u) {
    uint lz = clz(mantissa) - 24u;
    exponent = 1u - lz;
    mantissa = (mantissa << lz) & 0x7Fu;
  }
  return as_type<float>(((exponent + 124u) << 23u) | (mantissa << 16u));
}
inline uint xe_f32_to_7e3(float value) {
  if (!(value > 0.0)) return 0u;
  if (value >= 31.875) return 0x3FFu;
  uint bits = as_type<uint>(value);
  int exponent = int((bits >> 23u) & 0xFFu) - 124;
  uint mantissa = (bits >> 16u) & 0x7Fu;
  uint round_up = (bits >> 15u) & 1u;
  if (exponent <= 0) {
    uint shift = uint(1 - exponent);
    if (shift > 8u) return 0u;
    uint full = 0x80u | mantissa;
    uint result = (full >> shift) + ((full >> (shift - 1u)) & 1u);
    return min(result, 0x7Fu);
  }
  mantissa += round_up;
  if (mantissa > 0x7Fu) {
    mantissa = 0u;
    ++exponent;
    if (exponent >= 8) return 0x3FFu;
  }
  return (uint(exponent) << 7u) | mantissa;
}
inline float4 xe_unpack_color32(uint word, uint format) {
  switch (format) {
    default:
    case 0u:
      return xe_unpack_8888(word);
    case 2u:
      return float4(float(word & 0x3FFu) / 1023.0,
                    float((word >> 10u) & 0x3FFu) / 1023.0,
                    float((word >> 20u) & 0x3FFu) / 1023.0,
                    float(word >> 30u) / 3.0);
    case 3u:
      return float4(xe_7e3_to_f32(word), xe_7e3_to_f32(word >> 10u),
                    xe_7e3_to_f32(word >> 20u), float(word >> 30u) / 3.0);
    case 4u: {
      int2 fixed = int2(as_type<int>(word << 16u) >> 16,
                        as_type<int>(word) >> 16);
      return float4(float2(fixed) * (32.0 / 32767.0), 0.0, 1.0);
    }
    case 6u:
      return float4(float2(as_type<half2>(word)), 0.0, 0.0);
    case 14u:
      return float4(as_type<float>(word), 0.0, 0.0, 0.0);
  }
}
inline int xe_f32_to_fixed16(float value) {
  // -32768 decodes slightly below -32. Preserve that otherwise-unreachable raw
  // code during materialize->pack, while guest draw output at exactly -32 uses
  // the normal Xenos signed rounding and becomes -32767.
  if (value < -32.0) return -32768;
  value = clamp(value, -32.0, 32.0);
  float scaled = value * (32767.0 / 32.0);
  return int(scaled + (scaled >= 0.0 ? 0.5 : -0.5));
}
inline uint xe_pack_color32(float4 color, uint format) {
  switch (format) {
    default:
    case 0u:
      return xe_pack_8888(color);
    case 2u:
      return uint(round(saturate(color.r) * 1023.0)) |
             (uint(round(saturate(color.g) * 1023.0)) << 10u) |
             (uint(round(saturate(color.b) * 1023.0)) << 20u) |
             (uint(round(saturate(color.a) * 3.0)) << 30u);
    case 3u:
      return xe_f32_to_7e3(color.r) | (xe_f32_to_7e3(color.g) << 10u) |
             (xe_f32_to_7e3(color.b) << 20u) |
             (uint(round(saturate(color.a) * 3.0)) << 30u);
    case 4u: {
      int2 fixed = int2(xe_f32_to_fixed16(color.r),
                        xe_f32_to_fixed16(color.g));
      return (uint(fixed.x) & 0xFFFFu) | (uint(fixed.y) << 16u);
    }
    case 14u:
      return as_type<uint>(color.r);
  }
}
inline float4 xe_unpack_color64(uint2 words, uint format) {
  switch (format) {
    default:
    case 5u: {
      int4 fixed = int4(as_type<int>(words.x << 16u) >> 16,
                        as_type<int>(words.x) >> 16,
                        as_type<int>(words.y << 16u) >> 16,
                        as_type<int>(words.y) >> 16);
      return float4(fixed) * (32.0 / 32767.0);
    }
    case 7u:
      return float4(as_type<half4>(words));
    case 15u:
      return float4(as_type<float>(words.x), as_type<float>(words.y), 0.0,
                    0.0);
  }
}
inline uint2 xe_pack_color64(float4 color, uint format) {
  switch (format) {
    default:
    case 5u: {
      int4 fixed = int4(xe_f32_to_fixed16(color.r),
                        xe_f32_to_fixed16(color.g),
                        xe_f32_to_fixed16(color.b),
                        xe_f32_to_fixed16(color.a));
      return uint2((uint(fixed.x) & 0xFFFFu) |
                       (uint(fixed.y) << 16u),
                   (uint(fixed.z) & 0xFFFFu) |
                       (uint(fixed.w) << 16u));
    }
    case 7u:
      return as_type<uint2>(half4(color));
    case 15u:
      return uint2(as_type<uint>(color.r), as_type<uint>(color.g));
  }
}
inline uint4 xe_unpack_color32_raw(uint word, uint format) {
  if (format == 6u) {
    return uint4(word & 0xFFFFu, word >> 16u, 0u, 0u);
  }
  return uint4(word, 0u, 0u, 0u);
}
inline uint xe_pack_color32_raw(uint4 color, uint format) {
  if (format == 6u) {
    return (color.x & 0xFFFFu) | ((color.y & 0xFFFFu) << 16u);
  }
  return color.x;
}
inline uint4 xe_unpack_color64_raw(uint2 words, uint format) {
  if (format == 7u) {
    return uint4(words.x & 0xFFFFu, words.x >> 16u,
                 words.y & 0xFFFFu, words.y >> 16u);
  }
  return uint4(words.x, words.y, 0u, 0u);
}
inline uint2 xe_pack_color64_raw(uint4 color, uint format) {
  if (format == 7u) {
    return uint2((color.x & 0xFFFFu) | ((color.y & 0xFFFFu) << 16u),
                 (color.z & 0xFFFFu) | ((color.w & 0xFFFFu) << 16u));
  }
  return color.xy;
}
inline uint xe_pack_depth_stencil(uint depth24, uint stencil) {
  return ((depth24 & 0xFFFFFFu) << 8u) | (stencil & 0xFFu);
}
inline uint xe_depth24(uint word) { return word >> 8u; }
inline uint xe_stencil(uint word) { return word & 0xFFu; }
inline uint xe_stencil_bit_mask(uint bit_index) {
  return 1u << (bit_index & 7u);
}

struct AuthorityRangeParams {
  uint start_tiles;
  uint end_tiles;
};
inline uint xe_authority_tile_of(uint px, uint py, constant SurfaceParams& sp,
                                 uint sample) {
  uint2 sample_xy = uint2(px, py) << xe_edram_msaa_shifts(sp.msaa);
  sample_xy += (uint2(sample, sample) >> uint2(1u, 0u)) & 1u;
  uint tile_width_samples = 80u >> sp.fmt_log2;
  return sp.base + (sample_xy.y / 16u) * sp.pitch +
         sample_xy.x / tile_width_samples;
}
inline bool xe_authority_in_range(uint px, uint py,
                                  constant SurfaceParams& sp, uint sample,
                                  constant AuthorityRangeParams& range) {
  uint tile = xe_authority_tile_of(px, py, sp, sample);
  return tile >= range.start_tiles && tile < range.end_tiles;
}

// Canonical RGBA8Unorm materialize/pack kernels. These are shared verbatim by
// the live cache and the Metal conformance suite.
kernel void materialize_8888(
    device const uint* canonical [[buffer(0)]],
    texture2d<float, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint idx = off >> 2u;
  uint word = idx < sp.words ? canonical[idx] : 0u;
  tex.write(xe_unpack_8888(word), gid);
}
kernel void pack_8888(
    texture2d<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint idx = off >> 2u;
  if (idx < sp.words) canonical[idx] = xe_pack_8888(tex.read(gid));
}

// Generic single-sample color working-view conversion for the formats used by
// the deterministic corpus. The attachment format supplies the lossless host
// container (RGBA8Unorm, RGBA16Float, RG32Float, or R32Float); `format` supplies
// the guest raw-word interpretation.
kernel void materialize_color(
    device const uint* canonical [[buffer(0)]],
    texture2d<float, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint idx = off >> 2u;
  tex.write(xe_unpack_color32(idx < sp.words ? canonical[idx] : 0u, format),
            gid);
}
kernel void pack_color(
    texture2d<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint idx = off >> 2u;
  if (idx < sp.words) canonical[idx] = xe_pack_color32(tex.read(gid), format);
}
kernel void materialize_color_range(
    device const uint* canonical [[buffer(0)]],
    texture2d<float, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint idx = off >> 2u;
  tex.write(xe_unpack_color32(idx < sp.words ? canonical[idx] : 0u, format),
            gid);
}
kernel void pack_color_range(
    texture2d<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint idx = off >> 2u;
  if (idx < sp.words) canonical[idx] = xe_pack_color32(tex.read(gid), format);
}
kernel void materialize_color64_range(
    device const uint* canonical [[buffer(0)]],
    texture2d<float, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 1u, 0u) >> 2u;
  uint2 words = idx + 1u < sp.words
                    ? uint2(canonical[idx], canonical[idx + 1u])
                    : uint2(0u);
  tex.write(xe_unpack_color64(words, format), gid);
}
kernel void pack_color64_range(
    texture2d<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 1u, 0u) >> 2u;
  if (idx + 1u < sp.words) {
    uint2 words = xe_pack_color64(tex.read(gid), format);
    canonical[idx] = words.x;
    canonical[idx + 1u] = words.y;
  }
}
kernel void materialize_color_raw_range(
    device const uint* canonical [[buffer(0)]],
    texture2d<uint, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u) >> 2u;
  uint word = idx < sp.words ? canonical[idx] : 0u;
  tex.write(xe_unpack_color32_raw(word, format), gid);
}
kernel void pack_color_raw_range(
    texture2d<uint, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u) >> 2u;
  if (idx < sp.words) {
    canonical[idx] = xe_pack_color32_raw(tex.read(gid), format);
  }
}
kernel void materialize_color64_raw_range(
    device const uint* canonical [[buffer(0)]],
    texture2d<uint, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 1u, 0u) >> 2u;
  uint2 words = idx + 1u < sp.words
                    ? uint2(canonical[idx], canonical[idx + 1u])
                    : uint2(0u);
  tex.write(xe_unpack_color64_raw(words, format), gid);
}
kernel void pack_color64_raw_range(
    texture2d<uint, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      !xe_authority_in_range(gid.x, gid.y, sp, 0u, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 1u, 0u) >> 2u;
  if (idx + 1u < sp.words) {
    uint2 words = xe_pack_color64_raw(tex.read(gid), format);
    canonical[idx] = words.x;
    canonical[idx + 1u] = words.y;
  }
}

// Array-slice per-sample representation used by the isolated compute oracle.
kernel void materialize_8888_msaa(
    device const uint* canonical [[buffer(0)]],
    texture2d_array<float, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_array_size()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
  uint idx = off >> 2u;
  uint word = idx < sp.words ? canonical[idx] : 0u;
  tex.write(xe_unpack_8888(word), uint2(gid.x, gid.y), gid.z);
}
kernel void pack_8888_msaa(
    texture2d_array<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_array_size()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
  uint idx = off >> 2u;
  if (idx < sp.words) {
    canonical[idx] = xe_pack_8888(tex.read(uint2(gid.x, gid.y), gid.z));
  }
}

// Real multisample attachment representation used by the live path: sample-
// rate materialize and explicit per-sample compute packing.
struct ColorVOut { float4 pos [[position]]; };
vertex ColorVOut xe_color_vs(uint vid [[vertex_id]]) {
  ColorVOut out;
  float2 t = float2(float((vid << 1) & 2), float(vid & 2));
  out.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
  return out;
}
fragment float4 materialize_ms_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint sample [[sample_id]]) {
  uint2 pixel = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sample);
  uint off = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 0u, guest_sample);
  uint idx = off >> 2u;
  return xe_unpack_8888(idx < sp.words ? canonical[idx] : 0u);
}
kernel void pack_ms(
    texture2d_ms<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples()) return;
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, guest_sample);
  uint idx = off >> 2u;
  if (idx < sp.words) {
    canonical[idx] = xe_pack_8888(tex.read(uint2(gid.x, gid.y), gid.z));
  }
}

// Format-generic real-MSAA color working-view conversions. The fragment runs
// once per covered sample because it consumes [[sample_id]], preserving every
// Xenos physical sample in the actual multisample attachment. The compute pack
// explicitly reads each sample back into the canonical physical layout.
fragment float4 materialize_color_ms_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    uint sample [[sample_id]]) {
  uint2 pixel = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sample);
  uint off = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 0u, guest_sample);
  uint idx = off >> 2u;
  return xe_unpack_color32(idx < sp.words ? canonical[idx] : 0u, format);
}
kernel void pack_color_ms(
    texture2d_ms<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples()) return;
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, guest_sample);
  uint idx = off >> 2u;
  if (idx < sp.words) {
    canonical[idx] =
        xe_pack_color32(tex.read(uint2(gid.x, gid.y), gid.z), format);
  }
}
fragment float4 materialize_color_ms_range_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint sample [[sample_id]]) {
  uint2 pixel = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sample);
  if (!xe_authority_in_range(pixel.x, pixel.y, sp, guest_sample, range)) {
    discard_fragment();
  }
  uint off = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 0u, guest_sample);
  uint idx = off >> 2u;
  return xe_unpack_color32(idx < sp.words ? canonical[idx] : 0u, format);
}
kernel void pack_color_ms_range(
    texture2d_ms<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples() ||
      !xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, guest_sample);
  uint idx = off >> 2u;
  if (idx < sp.words) {
    canonical[idx] =
        xe_pack_color32(tex.read(uint2(gid.x, gid.y), gid.z), format);
  }
}
fragment float4 materialize_color64_ms_range_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint sample [[sample_id]]) {
  uint2 pixel = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sample);
  if (!xe_authority_in_range(pixel.x, pixel.y, sp, guest_sample, range)) {
    discard_fragment();
  }
  uint idx = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 1u, guest_sample) >> 2u;
  uint2 words = idx + 1u < sp.words
                    ? uint2(canonical[idx], canonical[idx + 1u])
                    : uint2(0u);
  return xe_unpack_color64(words, format);
}
kernel void pack_color64_ms_range(
    texture2d_ms<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples() ||
      !xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 1u, guest_sample) >> 2u;
  if (idx + 1u < sp.words) {
    uint2 words = xe_pack_color64(
        tex.read(uint2(gid.x, gid.y), gid.z), format);
    canonical[idx] = words.x;
    canonical[idx + 1u] = words.y;
  }
}
fragment uint4 materialize_color_raw_ms_range_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint sample [[sample_id]]) {
  uint2 pixel = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sample);
  if (!xe_authority_in_range(pixel.x, pixel.y, sp, guest_sample, range)) {
    discard_fragment();
  }
  uint idx = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 0u, guest_sample) >> 2u;
  uint word = idx < sp.words ? canonical[idx] : 0u;
  return xe_unpack_color32_raw(word, format);
}
kernel void pack_color_raw_ms_range(
    texture2d_ms<uint, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples() ||
      !xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, guest_sample) >> 2u;
  if (idx < sp.words) {
    canonical[idx] = xe_pack_color32_raw(
        tex.read(uint2(gid.x, gid.y), gid.z), format);
  }
}
fragment uint4 materialize_color64_raw_ms_range_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint sample [[sample_id]]) {
  uint2 pixel = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sample);
  if (!xe_authority_in_range(pixel.x, pixel.y, sp, guest_sample, range)) {
    discard_fragment();
  }
  uint idx = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 1u, guest_sample) >> 2u;
  uint2 words = idx + 1u < sp.words
                    ? uint2(canonical[idx], canonical[idx + 1u])
                    : uint2(0u);
  return xe_unpack_color64_raw(words, format);
}
kernel void pack_color64_raw_ms_range(
    texture2d_ms<uint, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant uint& format [[buffer(2)]],
    constant AuthorityRangeParams& range [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples() ||
      !xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) return;
  uint idx = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 1u, guest_sample) >> 2u;
  if (idx + 1u < sp.words) {
    uint2 words = xe_pack_color64_raw(
        tex.read(uint2(gid.x, gid.y), gid.z), format);
    canonical[idx] = words.x;
    canonical[idx + 1u] = words.y;
  }
}

inline float xe_f20e4_to_f32(uint f24) {
  f24 &= 0xFFFFFFu;
  if (f24 == 0u) return 0.0;
  uint mantissa = f24 & 0xFFFFFu;
  uint exponent = f24 >> 20;
  if (exponent == 0u) {           // denormal: same normalization as k_20e4
    uint lz = clz(mantissa) - 11u;
    exponent = 1u - lz;
    mantissa = (mantissa << lz) & 0xFFFFFu;
  }
  return as_type<float>(((exponent + 112u) << 23) | (mantissa << 3));
}
inline uint xe_f32_to_f20e4(float f) {  // inverse; mirrors the d2c encode
  if (!(f > 0.0)) return 0u;
  uint u = as_type<uint>(f);
  if (u >= 0x40000000u) return 0xFFFFFFu;  // >= 2.0 saturates
  if (u < 0x38800000u) {                   // denormal in 20e4
    uint shift = min(113u - (u >> 23), 24u);
    u = (0x800000u | (u & 0x7FFFFFu)) >> shift;
  } else {
    u -= 0x38000000u;
  }
  return (u >> 3) & 0xFFFFFFu;
}
// UNorm24 (kD24S8): the exact Xenia mapping. The numerator n24+(n24>>23) is an
// integer <= 2^24 (exactly representable in f32) and the *2^-24 only shifts the
// exponent, so EVERY decoded value is an exact f32 -- no near-1.0 collision.
inline float xe_unorm24_to_f32(uint n24) {
  n24 &= 0xFFFFFFu;
  return float(n24 + (n24 >> 23)) * (1.0 / 16777216.0);
}
inline uint xe_f32_to_unorm24(float d) {
  // EXACT inverse of xe_unorm24_to_f32: decode is (n24 + (n24>>23)) * 2^-24, so
  // d * 2^24 reconstructs the integer n24+b EXACTLY (power-of-two multiply, no
  // rounding), then remove the +b bias (b = 1 above the 0.5 boundary). Using
  // *2^24 (not *(2^24-1)) avoids the round-even tie that dropped near-boundary
  // values by 1.
  uint t = min(uint(rint(saturate(d) * 16777216.0)), 16777216u);
  return (t > 8388608u ? t - 1u : t) & 0xFFFFFFu;
}
// Decode a guest depth word's d24 to the internal host depth-attachment domain.
// float24 (fmt 2) spans guest [0,2): pack it to [0,1] with *0.5 so a Metal
// depth attachment holds the whole domain; unorm24 (fmt 1) is already [0,1].
// pack applies the inverse (*2 for float24).
inline float xe_depth_word_to_host(uint word, uint fmt) {
  uint d24 = xe_depth24(word);
  return fmt == 2u ? xe_f20e4_to_f32(d24) * 0.5 : xe_unorm24_to_f32(d24);
}
// Resolve snapshots are guest-visible numeric textures, not internal depth
// attachments. Decode them in the guest domain so a D24FS8 value g sampled
// after a resolve is still g rather than the host attachment encoding 0.5*g.
inline float xe_depth_word_to_guest(uint word, uint fmt) {
  uint d24 = xe_depth24(word);
  return fmt == 2u ? xe_f20e4_to_f32(d24) : xe_unorm24_to_f32(d24);
}
inline uint xe_host_to_depth_d24(float d, uint fmt) {
  return fmt == 2u ? xe_f32_to_f20e4(d * 2.0) : xe_f32_to_unorm24(d);
}
struct DepthVOut { float4 pos [[position]]; };
vertex DepthVOut xe_depth_vs(uint vid [[vertex_id]]) {
  DepthVOut o;
  float2 t = float2(float((vid << 1) & 2), float(vid & 2));
  o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
  return o;
}
struct DepthFragOut { float depth [[depth(any)]]; };
fragment DepthFragOut materialize_depth_fs(
    DepthVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]]) {
  uint2 gid = uint2(in.pos.xy);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, 0u);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  DepthFragOut o;
  o.depth = xe_depth_word_to_host(w, sp.is_depth);  // is_depth: 2=float24,1=unorm24
  return o;
}
// Stencil materialize: no shader stencil export in Metal, so the guest stencil
// byte is rebuilt one bit-plane at a time -- this fragment discards where the
// source word's stencil bit `bit` is clear, and the caller's REPLACE state ORs
// the bit in where it survives (mirrors the c2d stencil rebuild). NOTE: the
// caller's depth pass must REPLACE stencil to 0 first, so a PARTIAL (scissored,
// loadAction:Load) write fully replaces instead of OR-ing over prior stencil.
fragment void materialize_sten_fs(DepthVOut in [[stage_in]],
                                  device const uint* canonical [[buffer(0)]],
                                  constant SurfaceParams& sp [[buffer(1)]],
                                  constant uint& bit [[buffer(2)]]) {
  uint2 gid = uint2(in.pos.xy);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, 0u);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  if ((xe_stencil(w) & bit) == 0u) discard_fragment();
}
// Pack emits the COMPLETE depth/stencil word: (d24 << 8) | stencil. Depth from
// the depth2d view, stencil from the X32_Stencil8 view of the same attachment.
kernel void pack_depth(depth2d<float, access::read> tex [[texture(0)]],
                      texture2d<uint, access::read> sten [[texture(1)]],
                      device uint* canonical [[buffer(0)]],
                      constant SurfaceParams& sp [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, 0u);
  uint idx = off >> 2;
  if (idx >= sp.words) return;
  float d = tex.read(gid);
  uint s = sten.read(gid).r & 0xFFu;
  canonical[idx] =
      xe_pack_depth_stencil(xe_host_to_depth_d24(d, sp.is_depth), s);
}
// Per-sample 4x/2x depth+stencil: the same materialize/pack, but SAMPLE-AWARE.
// Materialize is sample-rate ([[sample_id]] forces per-sample execution) so each
// sample gets the canonical word at its own (pixel, sample) offset; pack reads
// depth2d_ms/texture2d_ms per sample. Reuses xe_depth_word_to_host /
// xe_host_to_depth_d24 / xe_edram_offset_bytes verbatim (the live path must too).
fragment DepthFragOut materialize_depth_ms_fs(
    DepthVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint sid [[sample_id]]) {
  uint2 gid = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sid);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, guest_sample);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  DepthFragOut o;
  o.depth = xe_depth_word_to_host(w, sp.is_depth);
  return o;
}
fragment void materialize_sten_ms_fs(DepthVOut in [[stage_in]],
                                     device const uint* canonical [[buffer(0)]],
                                     constant SurfaceParams& sp [[buffer(1)]],
                                     constant uint& bit [[buffer(2)]],
                                     uint sid [[sample_id]]) {
  uint2 gid = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sid);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, guest_sample);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  if ((xe_stencil(w) & bit) == 0u) discard_fragment();
}
kernel void pack_depth_ms(depth2d_ms<float, access::read> tex [[texture(0)]],
                         texture2d_ms<uint, access::read> sten [[texture(1)]],
                         device uint* canonical [[buffer(0)]],
                         constant SurfaceParams& sp [[buffer(1)]],
                         uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples()) return;
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, guest_sample);
  uint idx = off >> 2;
  if (idx >= sp.words) return;
  float d = tex.read(uint2(gid.x, gid.y), gid.z);
  uint s = sten.read(uint2(gid.x, gid.y), gid.z).r & 0xFFu;
  canonical[idx] =
      xe_pack_depth_stencil(xe_host_to_depth_d24(d, sp.is_depth), s);
}
fragment DepthFragOut materialize_depth_ms_range_fs(
    DepthVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant AuthorityRangeParams& range [[buffer(2)]],
    uint sid [[sample_id]]) {
  uint2 gid = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sid);
  if (!xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) {
    discard_fragment();
  }
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, guest_sample);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  DepthFragOut o;
  o.depth = xe_depth_word_to_host(w, sp.is_depth);
  return o;
}
fragment void materialize_sten_ms_range_fs(
    DepthVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant AuthorityRangeParams& range [[buffer(2)]],
    constant uint& bit [[buffer(3)]],
    uint sid [[sample_id]]) {
  uint2 gid = uint2(in.pos.xy);
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, sid);
  if (!xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) {
    discard_fragment();
  }
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, guest_sample);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  if ((xe_stencil(w) & bit) == 0u) discard_fragment();
}
kernel void pack_depth_ms_range(
    depth2d_ms<float, access::read> tex [[texture(0)]],
    texture2d_ms<uint, access::read> sten [[texture(1)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    constant AuthorityRangeParams& range [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint guest_sample = xe_metal_sample_to_guest(sp.msaa, gid.z);
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples() ||
      !xe_authority_in_range(gid.x, gid.y, sp, guest_sample, range)) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, guest_sample);
  uint idx = off >> 2;
  if (idx >= sp.words) return;
  float d = tex.read(uint2(gid.x, gid.y), gid.z);
  uint s = sten.read(uint2(gid.x, gid.y), gid.z).r & 0xFFu;
  canonical[idx] =
      xe_pack_depth_stencil(xe_host_to_depth_d24(d, sp.is_depth), s);
}

// Canonical resolve source. Destination formatting is performed by the Metal
// color attachment selected from RB_COPY_DEST_INFO; the existing writeback
// scatter then applies guest tiling, offsets, channel swap, and endian.
struct CanonicalResolveParams {
  SurfaceParams surface;
  uint source_x;
  uint source_y;
  uint dest_y;
  uint sample_select;
  uint color_format;
  uint dest_swap;
  uint dest_red_only;
  float exp_bias_scale;
  uint width;
  uint height;
  uint dest_format;
  uint dest_bpp_log2;
  uint dest_endian;
  uint raw_copy;
};
inline float4 xe_resolve_read_color(device const uint* canonical,
                                    constant CanonicalResolveParams& p,
                                    uint2 pixel, uint sample) {
  uint idx = xe_edram_offset_bytes(
                 pixel.x, pixel.y, p.surface.base, p.surface.pitch,
                 p.surface.msaa, 0u, p.surface.fmt_log2, sample) >>
             2u;
  // ResolveInfo permits the EDRAM base plus the rectangle offset to wrap the
  // physical 2048-tile ring. Other canonical conversion kernels are dispatched
  // over already-split physical ranges, but resolves address the guest surface
  // directly and therefore must wrap here.
  idx %= p.surface.words;
  if (p.surface.fmt_log2 != 0u) {
    uint2 words = uint2(canonical[idx],
                        canonical[(idx + 1u) % p.surface.words]);
    return xe_unpack_color64(words, p.color_format);
  }
  uint word = idx < p.surface.words ? canonical[idx] : 0u;
  return xe_unpack_color32(word, p.color_format);
}
inline float4 xe_resolve_color_value(device const uint* canonical,
                                     constant CanonicalResolveParams& p,
                                     uint2 source) {
  float4 value;
  switch (p.sample_select) {
    case 1u:
    case 2u:
    case 3u:
      value = xe_resolve_read_color(canonical, p, source, p.sample_select);
      break;
    case 4u:
      value = (xe_resolve_read_color(canonical, p, source, 0u) +
               xe_resolve_read_color(canonical, p, source, 1u)) *
              0.5;
      break;
    case 5u:
      value = (xe_resolve_read_color(canonical, p, source, 2u) +
               xe_resolve_read_color(canonical, p, source, 3u)) *
              0.5;
      break;
    case 6u:
      value = (xe_resolve_read_color(canonical, p, source, 0u) +
               xe_resolve_read_color(canonical, p, source, 1u) +
               xe_resolve_read_color(canonical, p, source, 2u) +
               xe_resolve_read_color(canonical, p, source, 3u)) *
              0.25;
      break;
    default:
      value = xe_resolve_read_color(canonical, p, source, 0u);
      break;
  }
  value *= p.exp_bias_scale;
  if (p.dest_swap == 0u) {
    return value;
  }
  if (p.dest_red_only != 0u) {
    // The resolve red-only packing path selects blue on swap only when the
    // source format actually stores a blue component. R/RG sources retain R.
    if (p.color_format == 0u || p.color_format == 2u ||
        p.color_format == 3u || p.color_format == 5u ||
        p.color_format == 7u) {
      value.r = value.b;
    } else if (p.color_format == 15u) {
      // Swapped 64bpp red-only resolves advance to the second stored dword;
      // for 32_32_FLOAT that is the G component.
      value.r = value.g;
    }
    return value;
  }
  return value.bgra;
}

// Guest resolve destination packing. Keep these switches in lockstep with
// pixel_formats.xesli XePack{16,32,64}bpp* and resolve_full_128bpp.xesli.
inline uint xe_resolve_pack_5551(float4 v) {
  uint4 n = uint4(saturate(v) * float4(31.0, 31.0, 31.0, 1.0) + 0.5);
  return n.r | (n.g << 5u) | (n.b << 10u) | (n.a << 15u);
}
inline uint xe_resolve_pack_565(float3 v) {
  uint3 n = uint3(saturate(v) * float3(31.0, 63.0, 31.0) + 0.5);
  return n.r | (n.g << 5u) | (n.b << 11u);
}
inline uint xe_resolve_pack_556(float3 v) {
  uint3 n = uint3(saturate(v) * float3(31.0, 31.0, 63.0) + 0.5);
  return n.r | (n.g << 5u) | (n.b << 10u);
}
inline uint xe_resolve_pack_4444(float4 v) {
  uint4 n = uint4(saturate(v) * 15.0 + 0.5);
  return n.r | (n.g << 4u) | (n.b << 8u) | (n.a << 12u);
}
inline uint xe_resolve_pack_1010102(float4 v) {
  uint4 n = uint4(saturate(v) * float4(1023.0, 1023.0, 1023.0, 3.0) + 0.5);
  return n.r | (n.g << 10u) | (n.b << 20u) | (n.a << 30u);
}
inline uint xe_resolve_pack_111110(float3 v) {
  uint3 n = uint3(saturate(v) * float3(2047.0, 2047.0, 1023.0) + 0.5);
  return n.r | (n.g << 11u) | (n.b << 22u);
}
inline uint xe_resolve_pack_101111(float3 v) {
  uint3 n = uint3(saturate(v) * float3(1023.0, 2047.0, 2047.0) + 0.5);
  return n.r | (n.g << 10u) | (n.b << 21u);
}
inline uint xe_resolve_pack_16(float v) {
  return uint(saturate(v) * 65535.0 + 0.5);
}
inline uint xe_resolve_pack_16_16(float2 v) {
  uint2 n = uint2(saturate(v) * 65535.0 + 0.5);
  return n.x | (n.y << 16u);
}
inline uint2 xe_resolve_pack_16_16_16_16(float4 v) {
  uint4 n = uint4(saturate(v) * 65535.0 + 0.5);
  return uint2(n.x | (n.y << 16u), n.z | (n.w << 16u));
}
inline ushort xe_resolve_pack_16bpp(float4 v, uint format) {
  switch (format) {
    case 3u: return ushort(xe_resolve_pack_5551(v));
    case 4u: return ushort(xe_resolve_pack_565(v.rgb));
    case 5u: return ushort(xe_resolve_pack_556(v.rgb));
    case 10u: {
      uint2 n = uint2(saturate(v.rg) * 255.0 + 0.5);
      return ushort(n.x | (n.y << 8u));
    }
    case 15u: return ushort(xe_resolve_pack_4444(v));
    case 24u: return ushort(xe_resolve_pack_16(v.r));
    default: return as_type<ushort>(half(v.r));
  }
}
inline uint xe_resolve_pack_32bpp(float4 v, uint format) {
  switch (format) {
    case 6u:
    case 14u:
    case 50u: {
      uint4 n = uint4(saturate(v) * 255.0 + 0.5);
      return n.r | (n.g << 8u) | (n.b << 16u) | (n.a << 24u);
    }
    case 7u:
    case 54u: return xe_resolve_pack_1010102(v);
    case 16u:
    case 55u: return xe_resolve_pack_111110(v.rgb);
    case 17u:
    case 56u: return xe_resolve_pack_101111(v.rgb);
    case 13u:
    case 25u: return xe_resolve_pack_16_16(v.rg);
    case 31u: return as_type<uint>(half2(v.rg));
    default: return as_type<uint>(v.r);
  }
}
inline uint2 xe_resolve_pack_64bpp(float4 v, uint format) {
  switch (format) {
    case 21u:
    case 26u: return xe_resolve_pack_16_16_16_16(v);
    case 32u: return as_type<uint2>(half4(v));
    default: return as_type<uint2>(v.rg);
  }
}
inline uint xe_resolve_endian32(uint v, uint endian) {
  if (endian == 1u || endian == 2u) {
    v = ((v & 0x00FF00FFu) << 8u) | ((v & 0xFF00FF00u) >> 8u);
  }
  if (endian == 2u || endian == 3u) v = (v << 16u) | (v >> 16u);
  return v;
}
inline uint2 xe_resolve_endian64(uint2 v, uint endian) {
  if (endian == 4u) {
    v = v.yx;
    endian = 2u;
  }
  return uint2(xe_resolve_endian32(v.x, endian),
               xe_resolve_endian32(v.y, endian));
}
inline uint4 xe_resolve_endian128(uint4 v, uint endian) {
  if (endian == 5u) {
    v = v.wzyx;
    endian = 2u;
  }
  uint2 lo = xe_resolve_endian64(v.xy, endian);
  uint2 hi = xe_resolve_endian64(v.zw, endian);
  return uint4(lo, hi);
}
inline uint4 xe_resolve_read_raw(device const uint* canonical,
                                 constant CanonicalResolveParams& p,
                                 uint2 source) {
  uint sample = min(p.sample_select, 3u);
  uint idx = xe_edram_offset_bytes(
                 source.x, source.y, p.surface.base, p.surface.pitch,
                 p.surface.msaa, 0u, p.surface.fmt_log2, sample) >> 2u;
  idx %= p.surface.words;
  uint4 value = uint4(canonical[idx], 0u, 0u, 0u);
  if (p.surface.fmt_log2 != 0u) {
    value.y = canonical[(idx + 1u) % p.surface.words];
  }
  if (p.dest_swap != 0u) {
    if (p.surface.fmt_log2 == 0u) {
      if (p.color_format == 0u) {
        value.x = (value.x & 0xFF00FF00u) |
                  ((value.x & 0xFFu) << 16u) |
                  ((value.x >> 16u) & 0xFFu);
      } else if (p.color_format == 2u) {
        value.x = (value.x & 0xC00FFC00u) |
                  ((value.x & 0x3FFu) << 20u) |
                  ((value.x >> 20u) & 0x3FFu);
      }
    } else if (p.color_format == 7u) {
      ushort4 h = as_type<ushort4>(value.xy);
      value.xy = as_type<uint2>(h.bgra);
    }
  }
  return value;
}

kernel void resolve_canonical_color_packed(
    device const uint* canonical [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant CanonicalResolveParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) return;
  uint2 source = uint2(p.source_x + gid.x, p.source_y + gid.y);
  uint byte_offset = (gid.y * p.width + gid.x) << p.dest_bpp_log2;
  if (p.raw_copy != 0u) {
    uint4 raw = xe_resolve_read_raw(canonical, p, source);
    if (p.dest_bpp_log2 == 2u) {
      *reinterpret_cast<device uint*>(destination + byte_offset) =
          xe_resolve_endian32(raw.x, p.dest_endian);
    } else {
      *reinterpret_cast<device uint2*>(destination + byte_offset) =
          xe_resolve_endian64(raw.xy, p.dest_endian);
    }
    return;
  }
  float4 value = xe_resolve_color_value(canonical, p, source);
  switch (p.dest_bpp_log2) {
    case 0u:
      destination[byte_offset] = uchar(saturate(value.r) * 255.0 + 0.5);
      break;
    case 1u: {
      ushort packed = xe_resolve_pack_16bpp(value, p.dest_format);
      if (p.dest_endian == 1u) packed = ushort((packed << 8u) | (packed >> 8u));
      *reinterpret_cast<device ushort*>(destination + byte_offset) = packed;
      break;
    }
    case 2u:
      *reinterpret_cast<device uint*>(destination + byte_offset) =
          xe_resolve_endian32(xe_resolve_pack_32bpp(value, p.dest_format),
                              p.dest_endian);
      break;
    case 3u:
      *reinterpret_cast<device uint2*>(destination + byte_offset) =
          xe_resolve_endian64(xe_resolve_pack_64bpp(value, p.dest_format),
                              p.dest_endian);
      break;
    default:
      *reinterpret_cast<device uint4*>(destination + byte_offset) =
          xe_resolve_endian128(as_type<uint4>(value), p.dest_endian);
      break;
  }
}
fragment float4 resolve_canonical_color_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant CanonicalResolveParams& p [[buffer(1)]]) {
  uint2 dest = uint2(in.pos.xy);
  if (dest.x >= p.width || dest.y < p.dest_y ||
      dest.y - p.dest_y >= p.height) {
    discard_fragment();
  }
  uint2 source = uint2(p.source_x + dest.x,
                       p.source_y + dest.y - p.dest_y);
  return xe_resolve_color_value(canonical, p, source);
}
fragment float4 resolve_canonical_depth_fs(
    ColorVOut in [[stage_in]],
    device const uint* canonical [[buffer(0)]],
    constant CanonicalResolveParams& p [[buffer(1)]]) {
  uint2 dest = uint2(in.pos.xy);
  if (dest.x >= p.width || dest.y < p.dest_y ||
      dest.y - p.dest_y >= p.height) {
    discard_fragment();
  }
  uint2 source = uint2(p.source_x + dest.x,
                       p.source_y + dest.y - p.dest_y);
  uint idx = xe_edram_offset_bytes(
                 source.x, source.y, p.surface.base, p.surface.pitch,
                 p.surface.msaa, p.surface.is_depth, 0u,
                 p.sample_select) >>
             2u;
  uint word = canonical[idx % p.surface.words];
  return float4(xe_depth_word_to_guest(word, p.surface.is_depth), 0.0, 0.0,
                0.0);
}
kernel void resolve_canonical_depth_raw(
    device const uint* canonical [[buffer(0)]],
    device uint* destination [[buffer(1)]],
    constant CanonicalResolveParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) {
    return;
  }
  uint2 source = uint2(p.source_x + gid.x, p.source_y + gid.y);
  uint idx = xe_edram_offset_bytes(
                 source.x, source.y, p.surface.base, p.surface.pitch,
                 p.surface.msaa, p.surface.is_depth, 0u, p.sample_select) >>
             2u;
  // Depth resolve RAM is a raw copy. Preserve the selected physical D24S8 or
  // D24FS8 word, including stencil, while the separate R32F snapshot above is
  // the decoded sampling view used by Metal shaders.
  destination[gid.y * p.width + gid.x] =
      canonical[idx % p.surface.words];
}
// Canonical clear scatter. One gid is one logical pixel and one physical MSAA
// sample; it writes the clear word(s) straight into the surface's own sample
// layout (including depth's 40-column swap via surface.is_depth), so a cleared
// canonical EDRAM matches what a resolve of the same surface would read back.
struct CanonicalClearParams {
  SurfaceParams surface;
  uint source_x;
  uint source_y;
  uint width;
  uint height;
  uint clear_lo;
  uint clear_hi;
};
kernel void resolve_canonical_clear(
    device uint* canonical [[buffer(0)]],
    constant CanonicalClearParams& p [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height ||
      gid.z >= (1u << p.surface.msaa)) {
    return;
  }
  uint idx = xe_edram_offset_bytes(
                 p.source_x + gid.x, p.source_y + gid.y, p.surface.base,
                 p.surface.pitch, p.surface.msaa, p.surface.is_depth,
                 p.surface.fmt_log2, gid.z) >>
             2u;
  idx %= p.surface.words;
  canonical[idx] = p.clear_lo;
  if (p.surface.fmt_log2 != 0u) {
    canonical[(idx + 1u) % p.surface.words] = p.clear_hi;
  }
}
)MSL";

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_EDRAM_MSL_H_
