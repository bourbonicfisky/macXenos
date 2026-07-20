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
inline uint xe_pack_depth_stencil(uint depth24, uint stencil) {
  return ((depth24 & 0xFFFFFFu) << 8u) | (stencil & 0xFFu);
}
inline uint xe_depth24(uint word) { return word >> 8u; }
inline uint xe_stencil(uint word) { return word & 0xFFu; }
inline uint xe_stencil_bit_mask(uint bit_index) {
  return 1u << (bit_index & 7u);
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
  uint off = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 0u, sample);
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
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
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
  uint off = xe_edram_offset_bytes(pixel.x, pixel.y, sp.base, sp.pitch,
                                   sp.msaa, 0u, 0u, sample);
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
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
  uint idx = off >> 2u;
  if (idx < sp.words) {
    canonical[idx] = xe_pack_color32(tex.read(uint2(gid.x, gid.y), gid.z),
                                     format);
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
// Decode a guest depth word's d24 to a host [0,1] depth value. float24 (fmt 2)
// spans guest [0,2): pack it to [0,1] with *0.5 so a Metal depth attachment
// (which clamps to [0,1]) holds the WHOLE domain; unorm24 (fmt 1) is already
// [0,1]. pack applies the inverse (*2 for float24).
inline float xe_depth_word_to_host(uint word, uint fmt) {
  uint d24 = xe_depth24(word);
  return fmt == 2u ? xe_f20e4_to_f32(d24) * 0.5 : xe_unorm24_to_f32(d24);
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
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, sid);
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
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, sid);
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
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, gid.z);
  uint idx = off >> 2;
  if (idx >= sp.words) return;
  float d = tex.read(uint2(gid.x, gid.y), gid.z);
  uint s = sten.read(uint2(gid.x, gid.y), gid.z).r & 0xFFu;
  canonical[idx] =
      xe_pack_depth_stencil(xe_host_to_depth_d24(d, sp.is_depth), s);
}
)MSL";

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_EDRAM_MSL_H_
