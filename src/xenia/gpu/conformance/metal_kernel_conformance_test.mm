/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 of the Metal conformance ladder: run the guest format decoders as
// Metal COMPUTE kernels and assert they agree BIT-FOR-BIT with the Level-0 CPU
// oracles (xenos.cc) over a swept input space. L0 proves the math is right on
// the CPU; L1 proves Metal's compiler + int/float semantics reproduce that same
// math on-device. A divergence here (fast-math reassociation, fma contraction,
// denormal flush, shift UB) is exactly the silent, per-title corruption class
// the native backend must never ship -- so it must fail in this isolated,
// device-only harness, not in a game frame.
//
// The kernels below are hand-mirrors of the CPU oracles. They are deliberately
// kept textually close to xenos.cc so a future edit to either side that breaks
// parity trips this test. When the real backend's MSL helpers (edram.xesli,
// resolve/copy kernels) are promoted here, they replace these mirrors one by
// one, each still checked against the same CPU oracle.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/trace_analysis_decode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

// CPU reference for the tile-range content hash (mirror of xe_hash_mix /
// k_tile_hash in the MSL below). The live ownership-transition manifest hashes
// dst tile ranges before/after each transfer; this is the oracle proving the
// GPU reduction is content-addressed and thread-order-independent.
inline uint32_t XeHashMix(uint32_t w, uint32_t i) {
  uint32_t h = w;
  h ^= i * 0x9E3779B9u;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h;
}
inline uint32_t XeTileHash(const std::vector<uint32_t>& words) {
  uint32_t acc = 0u;
  for (uint32_t i = 0; i < words.size(); ++i) acc += XeHashMix(words[i], i);
  return acc;
}

// Inline MSL: one kernel per format decoder. Each reads a uint code from `in`
// and writes the RESULT BITS to `out` (float results are stored via as_type so
// the comparison is exact, never through a float round-trip).
static const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Shared addresser: edram.xesli XeEdramOffsetBytes at resolution_scale=(1,1),
// wrap=false. One MSL definition used by BOTH the offset-parity kernel and the
// canonical pack/unpack kernels, so they cannot drift. msaa: 0=1x,1=2x,2=4x.
inline uint xe_edram_offset_bytes(uint px, uint py, uint base, uint pitch,
                                  uint msaa, uint is_depth, uint fmt_log2,
                                  uint sample) {
  uint2 s = uint2(px, py) << uint2(msaa >= 2u ? 1u : 0u,   // x doubles @4x
                                   msaa >= 1u ? 1u : 0u);   // y doubles @2x
  s += (uint2(sample, sample) >> uint2(1u, 0u)) & 1u;
  uint2 tsize32 = uint2(80u, 16u);
  uint2 tsize = tsize32 >> uint2(fmt_log2, 0u);
  uint2 toff = s / tsize;
  uint b = base + toff.y * pitch + toff.x;
  s -= toff * tsize;
  if (is_depth != 0u) {
    uint half_w = tsize.x >> 1u;  // 'half' is a reserved MSL type name
    s.x = (s.x >= half_w) ? (s.x - half_w) : (s.x + half_w);
  }
  uint address = b * (tsize32.x * tsize32.y) +
                 ((s.y * tsize.x + s.x) << fmt_log2);
  return address << 2;
}

// Per-surface constants for pack/unpack (one surface per dispatch). `words` is
// the canonical buffer size in 32-bit words, so a bounds guard can skip an
// over-dispatch instead of reading/writing out of range.
struct SurfaceParams { uint base, pitch, msaa, is_depth, fmt_log2, words; };
// One (pixel, sample) work item.
struct Coord { uint px, py, sample; };

// PACK: write each sample's raw 32-bit word into the canonical EDRAM buffer at
// its physical offset. This is the word-level core of the resolve/ownership
// materialize that will be wired into MetalRenderTargetCache behind the gate.
kernel void pack_words(device const Coord* coords [[buffer(0)]],
                      device const uint* values [[buffer(1)]],
                      device uint* canonical [[buffer(2)]],
                      constant SurfaceParams& sp [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  Coord c = coords[i];
  uint off = xe_edram_offset_bytes(c.px, c.py, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, sp.fmt_log2, c.sample);
  canonical[off >> 2] = values[i];  // off is bytes; >>2 -> word index
}

// UNPACK: read each sample back from the canonical buffer at its physical
// offset. pack->unpack must be the identity (also re-proves address
// injectivity on real hardware: a collision would clobber a word).
kernel void unpack_words(device const Coord* coords [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        device const uint* canonical [[buffer(2)]],
                        constant SurfaceParams& sp [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  Coord c = coords[i];
  uint off = xe_edram_offset_bytes(c.px, c.py, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, sp.fmt_log2, c.sample);
  out[i] = canonical[off >> 2];
}

// MATERIALIZE (canonical -> host attachment) for k_8_8_8_8 into an RGBA8Unorm
// texture: decode the EDRAM word's four byte lanes to normalized float4 and
// write the resolved pixel. 1x/single-sample. This is the exact 3.1 operation
// the cache will run to reconstruct a host render target from canonical EDRAM.
kernel void materialize_8888(device const uint* canonical [[buffer(0)]],
                            texture2d<float, access::write> tex [[texture(0)]],
                            constant SurfaceParams& sp [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  uint w = canonical[off >> 2];
  float4 c = float4(w & 0xFFu, (w >> 8) & 0xFFu, (w >> 16) & 0xFFu,
                    (w >> 24) & 0xFFu) * (1.0 / 255.0);
  tex.write(c, gid);
}

// PACK (host attachment -> canonical) for k_8_8_8_8: the inverse -- read the
// RGBA8Unorm texel, re-encode the four byte lanes into the EDRAM word. Exact
// for 8-bit unorm (byte/255 rounds back to byte). materialize->pack == identity.
kernel void pack_8888(texture2d<float, access::read> tex [[texture(0)]],
                     device uint* canonical [[buffer(0)]],
                     constant SurfaceParams& sp [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
  float4 c = tex.read(gid);
  uint4 b = uint4(round(c * 255.0)) & 0xFFu;
  uint w = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, 0u);
  canonical[off >> 2] = w;
}

// PER-SAMPLE MSAA materialize/pack (task 3.3, the sample-collapse fix). A true
// MSAA texture can't be per-sample-written from compute, so each of the 2x/4x
// samples lives in its own slice of a texture2d_array (sample s -> slice s).
// gid.z is the sample index; the canonical offset carries it. This is what
// keeps the samples the collapsed 1x path throws away.
kernel void materialize_8888_msaa(
    device const uint* canonical [[buffer(0)]],
    texture2d_array<float, access::write> tex [[texture(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_array_size()) return;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  float4 c = float4(w & 0xFFu, (w >> 8) & 0xFFu, (w >> 16) & 0xFFu,
                    (w >> 24) & 0xFFu) * (1.0 / 255.0);
  tex.write(c, uint2(gid.x, gid.y), gid.z);
}
kernel void pack_8888_msaa(
    texture2d_array<float, access::read> tex [[texture(0)]],
    device uint* canonical [[buffer(0)]],
    constant SurfaceParams& sp [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_array_size()) return;
  float4 c = tex.read(uint2(gid.x, gid.y), gid.z);
  uint4 b = uint4(round(c * 255.0)) & 0xFFu;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
  uint idx = off >> 2;
  if (idx < sp.words) canonical[idx] = b.x | (b.y << 8) | (b.z << 16) |
                                       (b.w << 24);
}

// The LIVE MSAA representation (task 3.3 wiring): a real texture2d_ms
// attachment. Materialize is a fullscreen RENDER pass with sample-rate shading
// -- [[sample_id]] forces per-sample execution, so each sample writes its own
// canonical word. This is the only way to get per-sample data into a true MSAA
// attachment (compute can't per-sample-write one).
struct FsOut { float4 pos [[position]]; };
vertex FsOut xe_fs_vs(uint vid [[vertex_id]]) {
  FsOut o;
  float2 t = float2(float((vid << 1) & 2), float(vid & 2));
  o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
  return o;
}
fragment float4 materialize_ms_fs(FsOut in [[stage_in]],
                                 device const uint* canonical [[buffer(0)]],
                                 constant SurfaceParams& sp [[buffer(1)]],
                                 uint sid [[sample_id]]) {
  uint2 p = uint2(in.pos.xy);
  uint off = xe_edram_offset_bytes(p.x, p.y, sp.base, sp.pitch, sp.msaa, 0u, 0u,
                                   sid);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  return float4(w & 0xFFu, (w >> 8) & 0xFFu, (w >> 16) & 0xFFu,
                (w >> 24) & 0xFFu) * (1.0 / 255.0);
}
// Pack from the real MSAA attachment: compute reads each sample via
// texture2d_ms.read(coord, sample). This is the live resolve/ownership pack.
kernel void pack_ms(texture2d_ms<float, access::read> tex [[texture(0)]],
                   device uint* canonical [[buffer(0)]],
                   constant SurfaceParams& sp [[buffer(1)]],
                   uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= tex.get_width() || gid.y >= tex.get_height() ||
      gid.z >= tex.get_num_samples()) return;
  float4 c = tex.read(uint2(gid.x, gid.y), gid.z);
  uint4 b = uint4(round(c * 255.0)) & 0xFFu;
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   0u, 0u, gid.z);
  uint idx = off >> 2;
  if (idx < sp.words) canonical[idx] = b.x | (b.y << 8) | (b.z << 16) |
                                       (b.w << 24);
}

kernel void k_gpuswap(device const uint* in [[buffer(0)]],
                      device uint* out [[buffer(1)]],
                      constant uint& mode [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
  uint v = in[i];
  // Mirror of xenos::GpuSwapInline (mode: 0=None,1=8in16,2=8in32,3=16in32).
  uint r;
  if (mode == 1u) {
    r = ((v << 8) & 0xFF00FF00u) | ((v >> 8) & 0x00FF00FFu);
  } else if (mode == 2u) {
    r = ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) |
        ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);
  } else if (mode == 3u) {
    r = ((v >> 16) & 0xFFFFu) | (v << 16);
  } else {
    r = v;
  }
  out[i] = r;
}

kernel void k_7e3(device const uint* in [[buffer(0)]],
                 device uint* out [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
  uint f10 = in[i] & 0x3FFu;
  if (f10 == 0u) { out[i] = 0u; return; }  // 0.0f bits
  uint mantissa = f10 & 0x7Fu;
  uint exponent = f10 >> 7;
  if (exponent == 0u) {
    uint lz = clz(mantissa) - 24u;         // (32 - 8)
    exponent = 1u - lz;                    // wraps like the CPU cast
    mantissa = (mantissa << lz) & 0x7Fu;
  }
  out[i] = ((exponent + 124u) << 23) | (mantissa << 3);
}

kernel void k_20e4(device const uint* in [[buffer(0)]],
                  device uint* out [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
  uint f24 = in[i] & 0xFFFFFFu;
  if (f24 == 0u) { out[i] = 0u; return; }
  uint mantissa = f24 & 0xFFFFFu;
  uint exponent = f24 >> 20;
  if (exponent == 0u) {
    uint lz = clz(mantissa) - 11u;         // (32 - 21)
    exponent = 1u - lz;
    mantissa = (mantissa << lz) & 0xFFFFFu;
  }
  out[i] = ((exponent + 112u) << 23) | (mantissa << 3);
}

kernel void k_unorm24(device const uint* in [[buffer(0)]],
                     device uint* out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
  uint n24 = in[i] & 0xFFFFFFu;
  // Exact: scale is 2^-24, so no rounding regardless of fast-math.
  float r = float(n24 + (n24 >> 23)) * (1.0 / 16777216.0);
  out[i] = as_type<uint>(r);
}

// EDRAM tile/sample byte offset -- a faithful transcription of the SHIPPING
// GPU addresser edram.xesli XeEdramOffsetBytes at resolution_scale=(1,1),
// wrap=false. Checked against the CPU oracle XeEdramSnapshotOffsetBytes so the
// two independent implementations of the one addressing convention are proven
// to agree (task 2.2). msaa: 0=1x,1=2x,2=4x (Xenos enum).
struct EdramIn {
  uint px, py, base, pitch, msaa, is_depth, fmt_log2, sample;
};
kernel void k_edram_offset(device const EdramIn* in [[buffer(0)]],
                          device uint* out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  EdramIn e = in[i];
  out[i] = xe_edram_offset_bytes(e.px, e.py, e.base, e.pitch, e.msaa,
                                 e.is_depth, e.fmt_log2, e.sample);
}

// The SHIPPING resolve-copy tile walk, transcribed from
// MetalRenderTargetCache xe_rcp_fs (.mm:1745-1753) at y_offset=0. It runs in
// the collapsed single-sample model (no per-sample, no depth swap), but it
// assigns each dest pixel to an EDRAM tile with the SAME arithmetic the
// canonical addresser uses. Proving the two agree on tile membership is the
// invariant the canonical resolve wiring depends on (both must place a pixel in
// the same tile). Reuses EdramIn; sample/is_depth are ignored (origin sample).
kernel void k_rcp_tile(device const EdramIn* in [[buffer(0)]],
                      device uint* out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  EdramIn e = in[i];
  uint tile_w = 80u >> e.fmt_log2;
  uint2 ds = uint2(e.px << (e.msaa >= 2u ? 1u : 0u),   // x doubles @4x
                   e.py << (e.msaa >= 1u ? 1u : 0u));   // y doubles @2x
  uint col = ds.x / tile_w;
  out[i] = e.base + (ds.y / 16u) * e.pitch + col;
}

// Tile-range content hash primitive (L3b ownership-transition manifest). An
// order-INDEPENDENT (atomic-add), position-SENSITIVE (index folded in)
// reduction over 32-bit EDRAM words. The live transition manifest will hash a
// destination tile range before and after each transfer to detect value
// corruption, so the reduction must be reproducible regardless of thread order
// yet change if a word lands at the wrong position. Avalanche mix so distinct
// content maps to distinct sums with high probability.
inline uint xe_hash_mix(uint w, uint i) {
  uint h = w;
  h ^= i * 0x9E3779B9u;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  h ^= h >> 16;
  return h;
}
kernel void k_tile_hash(device const uint* words [[buffer(0)]],
                       device atomic_uint* out [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  atomic_fetch_add_explicit(out, xe_hash_mix(words[i], i),
                            memory_order_relaxed);
}
// Same reduction, but sourced from a host attachment -- proves hashing the
// RGBA8Unorm/R32Float texture equals hashing its raw EDRAM words, the invariant
// the live path relies on to hash owned tiles without a materialize step.
// mode 0 = RGBA8Unorm (pack the four unorm8 lanes, matching pack_8888);
// mode 1 = R32Float (raw float bits).
kernel void k_tile_hash_tex(texture2d<float, access::read> tex [[texture(0)]],
                           device atomic_uint* out [[buffer(0)]],
                           constant uint& mode [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
  uint W = tex.get_width();
  if (gid.x >= W || gid.y >= tex.get_height()) return;
  float4 v = tex.read(gid);
  uint word;
  if (mode == 0u) {
    uint4 b = uint4(round(v * 255.0)) & 0xFFu;
    word = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
  } else {
    word = as_type<uint>(v.x);
  }
  uint i = gid.y * W + gid.x;
  atomic_fetch_add_explicit(out, xe_hash_mix(word, i), memory_order_relaxed);
}

// Depth canonical materialize/pack substrate for the cross-MSAA slice. The JC2
// hot range [720,1440) is kD24FS8 (float24, 20e4): its 20-bit mantissa fits
// D32Float's 23 bits, so it round-trips through a host depth texture LOSSLESSLY
// (unlike unorm24/kD24S8 near 1.0 -- a separate precision case). Materialize
// writes SV_Depth in a fragment shader (depth textures can't be compute-written);
// pack reads depth2d in compute. The canonical word is (d24 << 8) | stencil;
// this depth-only rung ignores stencil (stencil plane is the next rung).
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
  uint d24 = word >> 8;
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
// the bit in where it survives (mirrors the c2d stencil rebuild).
fragment void materialize_sten_fs(DepthVOut in [[stage_in]],
                                  device const uint* canonical [[buffer(0)]],
                                  constant SurfaceParams& sp [[buffer(1)]],
                                  constant uint& bit [[buffer(2)]]) {
  uint2 gid = uint2(in.pos.xy);
  uint off = xe_edram_offset_bytes(gid.x, gid.y, sp.base, sp.pitch, sp.msaa,
                                   sp.is_depth, 0u, 0u);
  uint idx = off >> 2;
  uint w = (idx < sp.words) ? canonical[idx] : 0u;
  if (((w & 0xFFu) & bit) == 0u) discard_fragment();
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
  canonical[idx] = (xe_host_to_depth_d24(d, sp.is_depth) << 8) | s;
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
  if (((w & 0xFFu) & bit) == 0u) discard_fragment();
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
  canonical[idx] = (xe_host_to_depth_d24(d, sp.is_depth) << 8) | s;
}
)MSL";

// Mirrors of the MSL pack/unpack structs -- layouts must match exactly or the
// round-trip corrupts (a stride mismatch would fail loudly, by design).
struct Coord {
  uint32_t px, py, sample;
};
struct SurfaceParams {
  uint32_t base, pitch, msaa, is_depth, fmt_log2, words;
};

// A minimal compute harness: compile the library once, dispatch one kernel over
// a code vector, return the output bits. All buffers are shared storage so the
// readback is a plain memcpy.
struct MetalCompute {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;

  // Returns false ONLY when there is no Metal device (a legitimate headless-CI
  // skip). If a device IS present but the shared MSL library fails to compile,
  // that is a HARD FAILURE, not a skip -- a broken kernel silently skipping
  // every L1 case is precisely the false-green this ladder exists to prevent.
  bool Init() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    queue = [dev newCommandQueue];
    NSError* err = nil;
    lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                            options:nil
                              error:&err];
    if (!lib) {
      FAIL("device present but MSL library failed to compile: " <<
           (err ? err.localizedDescription.UTF8String : "unknown"));
    }
    return true;
  }

  // Runs `name` over `codes`; `mode` is an optional scalar bound at buffer(2).
  std::vector<uint32_t> Run(const char* name,
                            const std::vector<uint32_t>& codes,
                            bool has_mode = false, uint32_t mode = 0) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    const size_t n = codes.size();
    const size_t bytes = n * sizeof(uint32_t);
    id<MTLBuffer> in_buf = [dev newBufferWithBytes:codes.data()
                                            length:bytes
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [dev newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> mode_buf = has_mode
        ? [dev newBufferWithBytes:&mode length:sizeof(mode)
                          options:MTLResourceStorageModeShared]
        : nil;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:in_buf offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    if (has_mode) [enc setBuffer:mode_buf offset:0 atIndex:2];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(n);
    std::memcpy(out.data(), out_buf.contents, bytes);
    return out;
  }

  // Runs `name` over `count` input elements of arbitrary struct type at
  // buffer(0); returns `count` uint results from buffer(1).
  std::vector<uint32_t> RunRaw(const char* name, const void* in_data,
                               size_t in_bytes, size_t count) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);
    id<MTLBuffer> in_buf = [dev newBufferWithBytes:in_data
                                            length:in_bytes
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf =
        [dev newBufferWithLength:count * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:in_buf offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > count) tg = count ? count : 1;
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(count);
    std::memcpy(out.data(), out_buf.contents, count * sizeof(uint32_t));
    return out;
  }

  // Packs `values` into a real EDRAM-sized canonical buffer at each coord's
  // physical offset, then unpacks them back -- the word-level resolve/ownership
  // materialize round-trip against the ACTUAL buffer the redesign will use.
  // Returns the unpacked values; identity <=> correct addressing + no aliasing.
  std::vector<uint32_t> RoundTripCanonical(const std::vector<Coord>& coords,
                                           const std::vector<uint32_t>& values,
                                           const struct SurfaceParams& sp,
                                           size_t canonical_bytes) {
    const size_t n = coords.size();
    id<MTLBuffer> coord_buf =
        [dev newBufferWithBytes:coords.data()
                         length:n * sizeof(Coord)
                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> val_buf =
        [dev newBufferWithBytes:values.data()
                         length:n * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf =
        [dev newBufferWithLength:n * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);  // not zero-guaranteed
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(struct SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    id<MTLComputePipelineState> pack =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_words"]
                                           error:&err];
    REQUIRE(pack != nil);
    id<MTLComputePipelineState> unpack =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"unpack_words"]
                                           error:&err];
    REQUIRE(unpack != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    // Two separate encoders => the pack writes are ordered-before the unpack
    // reads (encoder boundary is a barrier on the shared canonical buffer).
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pack];
      [enc setBuffer:coord_buf offset:0 atIndex:0];
      [enc setBuffer:val_buf offset:0 atIndex:1];
      [enc setBuffer:canonical offset:0 atIndex:2];
      [enc setBuffer:sp_buf offset:0 atIndex:3];
      [enc dispatchThreads:MTLSizeMake(n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
              pack.maxTotalThreadsPerThreadgroup > n
                  ? (n ? n : 1)
                  : pack.maxTotalThreadsPerThreadgroup,
              1, 1)];
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:unpack];
      [enc setBuffer:coord_buf offset:0 atIndex:0];
      [enc setBuffer:out_buf offset:0 atIndex:1];
      [enc setBuffer:canonical offset:0 atIndex:2];
      [enc setBuffer:sp_buf offset:0 atIndex:3];
      [enc dispatchThreads:MTLSizeMake(n, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
              unpack.maxTotalThreadsPerThreadgroup > n
                  ? (n ? n : 1)
                  : unpack.maxTotalThreadsPerThreadgroup,
              1, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(n);
    std::memcpy(out.data(), out_buf.contents, n * sizeof(uint32_t));
    return out;
  }

  // Materialize k_8_8_8_8 canonical words into a host RGBA8Unorm texture, then
  // pack the texture back to canonical -- the round-trip the cache's 3.1
  // materialize/pack will perform. `words` holds one word per (y*w+x); returns
  // the re-packed words read back from the canonical buffer at the same offsets.
  std::vector<uint32_t> RoundTripMaterialize8888(uint32_t w, uint32_t h,
                                                 const struct SurfaceParams& sp,
                                                 const std::vector<uint32_t>& words,
                                                 size_t canonical_bytes) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    // Seed canonical at each pixel's physical offset (CPU oracle offset).
    uint32_t* cwords = static_cast<uint32_t*>(canonical.contents);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t off = XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch,
                                                  sp.msaa, false, 0u, 0u);
        cwords[off >> 2] = words[y * w + x];
      }
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:w
                                    height:h
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(struct SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    id<MTLComputePipelineState> mat = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"materialize_8888"]
                                      error:&err];
    REQUIRE(mat != nil);
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_8888"]
                                      error:&err];
    REQUIRE(pk != nil);
    auto dispatch2d = [&](id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso) {
      [enc setComputePipelineState:pso];
      NSUInteger tw = pso.threadExecutionWidth;
      NSUInteger th = pso.maxTotalThreadsPerThreadgroup / tw;
      [enc dispatchThreads:MTLSizeMake(w, h, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    };
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setTexture:tex atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      dispatch2d(enc, mat);
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setTexture:tex atIndex:0];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      dispatch2d(enc, pk);
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(size_t(w) * h);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t off = XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch,
                                                  sp.msaa, false, 0u, 0u);
        out[y * w + x] = cwords[off >> 2];
      }
    }
    return out;
  }

  // Per-sample MSAA variant: seed canonical with a distinct word for every
  // (pixel, sample), materialize into a texture2d_array (slice per sample),
  // pack it back, and return the re-packed words. Identity <=> all 2x/4x
  // samples survive independently -- the sample-collapse fix (task 3.3).
  // `values` is indexed [(y*w + x) * num_samples + s].
  std::vector<uint32_t> RoundTripMaterializeMsaa(uint32_t w, uint32_t h,
                                                 uint32_t num_samples,
                                                 const struct SurfaceParams& sp,
                                                 const std::vector<uint32_t>& values,
                                                 size_t canonical_bytes) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* cwords = static_cast<uint32_t*>(canonical.contents);
    auto seed_offset = [&](uint32_t x, uint32_t y, uint32_t s) {
      return XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch, sp.msaa, false,
                                        0u, s) >> 2;
    };
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          cwords[seed_offset(x, y, s)] = values[(y * w + x) * num_samples + s];
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DArray;
    td.pixelFormat = MTLPixelFormatRGBA8Unorm;
    td.width = w;
    td.height = h;
    td.arrayLength = num_samples;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(struct SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    id<MTLComputePipelineState> mat = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"materialize_8888_msaa"]
                                      error:&err];
    REQUIRE(mat != nil);
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_8888_msaa"]
                                      error:&err];
    REQUIRE(pk != nil);
    auto dispatch3d = [&](id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso) {
      [enc setComputePipelineState:pso];
      NSUInteger tw = pso.threadExecutionWidth;
      NSUInteger th = pso.maxTotalThreadsPerThreadgroup / tw;
      [enc dispatchThreads:MTLSizeMake(w, h, num_samples)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
    };
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setTexture:tex atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      dispatch3d(enc, mat);
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setTexture:tex atIndex:0];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      dispatch3d(enc, pk);
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(size_t(w) * h * num_samples);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          out[(y * w + x) * num_samples + s] = cwords[seed_offset(x, y, s)];
    return out;
  }

  // The LIVE representation round-trip: materialize canonical into a REAL MSAA
  // texture via a sample-rate render pass, then pack it back via compute
  // (texture2d_ms.read per sample). Proves the exact technique the cache wiring
  // will use to render per-sample and resolve samples. `values` indexed
  // [(y*w+x)*num_samples + s]; returns the re-packed words.
  std::vector<uint32_t> RoundTripMsaaReal(uint32_t w, uint32_t h,
                                          uint32_t num_samples,
                                          const struct SurfaceParams& sp,
                                          const std::vector<uint32_t>& values,
                                          size_t canonical_bytes) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* cwords = static_cast<uint32_t*>(canonical.contents);
    auto seed_offset = [&](uint32_t x, uint32_t y, uint32_t s) {
      return XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch, sp.msaa, false,
                                        0u, s) >> 2;
    };
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          cwords[seed_offset(x, y, s)] = values[(y * w + x) * num_samples + s];
    // Real MSAA attachment (private; MSAA can't be shared). Store all samples.
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DMultisample;
    td.pixelFormat = MTLPixelFormatRGBA8Unorm;
    td.width = w;
    td.height = h;
    td.sampleCount = num_samples;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> ms = [dev newTextureWithDescriptor:td];
    REQUIRE(ms != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(struct SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = [lib newFunctionWithName:@"xe_fs_vs"];
    rpd.fragmentFunction = [lib newFunctionWithName:@"materialize_ms_fs"];
    rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    rpd.rasterSampleCount = num_samples;
    id<MTLRenderPipelineState> rps =
        [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
    REQUIRE(rps != nil);
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_ms"]
                                      error:&err];
    REQUIRE(pk != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    // Materialize: sample-rate render pass into the MSAA texture.
    {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = ms;
      rp.colorAttachments[0].loadAction = MTLLoadActionClear;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;  // keep samples
      rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      id<MTLRenderCommandEncoder> enc =
          [cb renderCommandEncoderWithDescriptor:rp];
      [enc setRenderPipelineState:rps];
      [enc setFragmentBuffer:canonical offset:0 atIndex:0];
      [enc setFragmentBuffer:sp_buf offset:0 atIndex:1];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
    }
    // Pack: compute reads each sample of the MSAA texture back to canonical.
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pk];
      [enc setTexture:ms atIndex:0];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      NSUInteger tw = pk.threadExecutionWidth;
      NSUInteger th = pk.maxTotalThreadsPerThreadgroup / tw;
      [enc dispatchThreads:MTLSizeMake(w, h, num_samples)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(size_t(w) * h * num_samples);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          out[(y * w + x) * num_samples + s] = cwords[seed_offset(x, y, s)];
    return out;
  }

  // Depth canonical round-trip: seed canonical with a d24 word at each pixel's
  // DEPTH offset, materialize into a host D32Float depth texture via an SV_Depth
  // render pass, pack it back with pack_depth, and return the re-packed canonical
  // words. Identity <=> the format's decode/encode (sp.is_depth: 2=float24 with
  // the *0.5//*2 [0,2) mapping, 1=unorm24) and depth addressing are exact
  // inverses through real host depth storage -- the substrate the cross-MSAA
  // slice needs for JC2's kD24FS8/kD24S8 buffers.
  std::vector<uint32_t> RoundTripDepth(uint32_t w, uint32_t h,
                                       const struct SurfaceParams& sp,
                                       const std::vector<uint32_t>& words,
                                       size_t canonical_bytes) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* cwords = static_cast<uint32_t*>(canonical.contents);
    auto doff = [&](uint32_t x, uint32_t y) {
      return XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch, sp.msaa,
                                        /*is_depth=*/true, 0u, 0u) >> 2;
    };
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        cwords[doff(x, y)] = words[y * w + x];  // full (d24<<8)|stencil word
    // Combined depth+stencil attachment (guest word = (d24<<8)|stencil), so the
    // round-trip carries the full word, not just depth.
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                     width:w
                                    height:h
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;  // depth/stencil can't be shared.
    id<MTLTexture> depth = [dev newTextureWithDescriptor:td];
    REQUIRE(depth != nil);
    // X32_Stencil8 view for reading the stencil plane as texture2d<uint>.
    id<MTLTexture> sten_view =
        [depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    REQUIRE(sten_view != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(struct SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = [lib newFunctionWithName:@"xe_depth_vs"];
    rpd.fragmentFunction = [lib newFunctionWithName:@"materialize_depth_fs"];
    rpd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    rpd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    id<MTLRenderPipelineState> rps =
        [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
    REQUIRE(rps != nil);
    MTLRenderPipelineDescriptor* spd = [[MTLRenderPipelineDescriptor alloc] init];
    spd.vertexFunction = [lib newFunctionWithName:@"xe_depth_vs"];
    spd.fragmentFunction = [lib newFunctionWithName:@"materialize_sten_fs"];
    spd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    spd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    id<MTLRenderPipelineState> sps =
        [dev newRenderPipelineStateWithDescriptor:spd error:&err];
    REQUIRE(sps != nil);
    // Depth pass: write depth, leave stencil untouched (cleared to 0 by load).
    MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction = MTLCompareFunctionAlways;
    dsd.depthWriteEnabled = YES;
    id<MTLDepthStencilState> dss = [dev newDepthStencilStateWithDescriptor:dsd];
    // One REPLACE-masked state per stencil bit (sets bit where undiscarded).
    id<MTLDepthStencilState> bit_states[8];
    for (uint32_t b = 0; b < 8; ++b) {
      MTLDepthStencilDescriptor* bds = [[MTLDepthStencilDescriptor alloc] init];
      bds.depthCompareFunction = MTLCompareFunctionAlways;
      bds.depthWriteEnabled = NO;
      MTLStencilDescriptor* bs = [[MTLStencilDescriptor alloc] init];
      bs.stencilCompareFunction = MTLCompareFunctionAlways;
      bs.depthStencilPassOperation = MTLStencilOperationReplace;
      bs.writeMask = 1u << b;
      bds.frontFaceStencil = bs;
      bds.backFaceStencil = bs;
      bit_states[b] = [dev newDepthStencilStateWithDescriptor:bds];
    }
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_depth"]
                                      error:&err];
    REQUIRE(pk != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {  // materialize: depth via SV_Depth, then stencil one bit-plane at a time.
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = depth;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.clearDepth = 0.0;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      rp.stencilAttachment.texture = depth;
      rp.stencilAttachment.loadAction = MTLLoadActionClear;
      rp.stencilAttachment.clearStencil = 0;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc =
          [cb renderCommandEncoderWithDescriptor:rp];
      [enc setRenderPipelineState:rps];
      [enc setDepthStencilState:dss];
      [enc setFragmentBuffer:canonical offset:0 atIndex:0];
      [enc setFragmentBuffer:sp_buf offset:0 atIndex:1];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc setRenderPipelineState:sps];
      [enc setStencilReferenceValue:0xFF];
      [enc setFragmentBuffer:canonical offset:0 atIndex:0];
      [enc setFragmentBuffer:sp_buf offset:0 atIndex:1];
      for (uint32_t b = 0; b < 8; ++b) {
        uint32_t bit = 1u << b;
        [enc setDepthStencilState:bit_states[b]];
        [enc setFragmentBytes:&bit length:sizeof(bit) atIndex:2];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      }
      [enc endEncoding];
    }
    {  // pack: compute reads depth + stencil views back to the complete word.
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pk];
      [enc setTexture:depth atIndex:0];
      [enc setTexture:sten_view atIndex:1];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      NSUInteger tw = pk.threadExecutionWidth;
      NSUInteger th = pk.maxTotalThreadsPerThreadgroup / tw;
      [enc dispatchThreads:MTLSizeMake(w, h, 1)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(size_t(w) * h);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x) out[y * w + x] = cwords[doff(x, y)];
    return out;
  }

  // Per-sample MSAA depth+stencil round-trip: a REAL num_samples-sample
  // Depth32Float_Stencil8 attachment, materialized sample-rate and packed
  // per sample. `words` is indexed [(y*w + x) * num_samples + s]; returns the
  // re-packed complete words in the same order. Identity <=> every sample's
  // depth AND stencil survive independently (the 4x sample dimension the
  // collapsed host loses).
  std::vector<uint32_t> RoundTripDepthMsaa(uint32_t w, uint32_t h,
                                           uint32_t num_samples,
                                           const struct SurfaceParams& sp,
                                           const std::vector<uint32_t>& words,
                                           size_t canonical_bytes) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* cwords = static_cast<uint32_t*>(canonical.contents);
    auto doff = [&](uint32_t x, uint32_t y, uint32_t s) {
      return XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch, sp.msaa,
                                        /*is_depth=*/true, 0u, s) >> 2;
    };
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          cwords[doff(x, y, s)] = words[(y * w + x) * num_samples + s];
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DMultisample;
    td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    td.width = w;
    td.height = h;
    td.sampleCount = num_samples;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth = [dev newTextureWithDescriptor:td];
    REQUIRE(depth != nil);
    id<MTLTexture> sten_view =
        [depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    REQUIRE(sten_view != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(struct SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = [lib newFunctionWithName:@"xe_depth_vs"];
    rpd.fragmentFunction = [lib newFunctionWithName:@"materialize_depth_ms_fs"];
    rpd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    rpd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    rpd.rasterSampleCount = num_samples;
    id<MTLRenderPipelineState> rps =
        [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
    REQUIRE(rps != nil);
    MTLRenderPipelineDescriptor* spd = [[MTLRenderPipelineDescriptor alloc] init];
    spd.vertexFunction = [lib newFunctionWithName:@"xe_depth_vs"];
    spd.fragmentFunction = [lib newFunctionWithName:@"materialize_sten_ms_fs"];
    spd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    spd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    spd.rasterSampleCount = num_samples;
    id<MTLRenderPipelineState> sps =
        [dev newRenderPipelineStateWithDescriptor:spd error:&err];
    REQUIRE(sps != nil);
    MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction = MTLCompareFunctionAlways;
    dsd.depthWriteEnabled = YES;
    id<MTLDepthStencilState> dss = [dev newDepthStencilStateWithDescriptor:dsd];
    id<MTLDepthStencilState> bit_states[8];
    for (uint32_t b = 0; b < 8; ++b) {
      MTLDepthStencilDescriptor* bds = [[MTLDepthStencilDescriptor alloc] init];
      bds.depthCompareFunction = MTLCompareFunctionAlways;
      bds.depthWriteEnabled = NO;
      MTLStencilDescriptor* bs = [[MTLStencilDescriptor alloc] init];
      bs.stencilCompareFunction = MTLCompareFunctionAlways;
      bs.depthStencilPassOperation = MTLStencilOperationReplace;
      bs.writeMask = 1u << b;
      bds.frontFaceStencil = bs;
      bds.backFaceStencil = bs;
      bit_states[b] = [dev newDepthStencilStateWithDescriptor:bds];
    }
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_depth_ms"]
                                      error:&err];
    REQUIRE(pk != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = depth;
      rp.depthAttachment.loadAction = MTLLoadActionClear;
      rp.depthAttachment.clearDepth = 0.0;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      rp.stencilAttachment.texture = depth;
      rp.stencilAttachment.loadAction = MTLLoadActionClear;
      rp.stencilAttachment.clearStencil = 0;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc =
          [cb renderCommandEncoderWithDescriptor:rp];
      [enc setRenderPipelineState:rps];
      [enc setDepthStencilState:dss];
      [enc setFragmentBuffer:canonical offset:0 atIndex:0];
      [enc setFragmentBuffer:sp_buf offset:0 atIndex:1];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc setRenderPipelineState:sps];
      [enc setStencilReferenceValue:0xFF];
      [enc setFragmentBuffer:canonical offset:0 atIndex:0];
      [enc setFragmentBuffer:sp_buf offset:0 atIndex:1];
      for (uint32_t b = 0; b < 8; ++b) {
        uint32_t bit = 1u << b;
        [enc setDepthStencilState:bit_states[b]];
        [enc setFragmentBytes:&bit length:sizeof(bit) atIndex:2];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      }
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pk];
      [enc setTexture:depth atIndex:0];
      [enc setTexture:sten_view atIndex:1];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      NSUInteger tw = pk.threadExecutionWidth;
      NSUInteger th = pk.maxTotalThreadsPerThreadgroup / tw;
      [enc dispatchThreads:MTLSizeMake(w, h, num_samples)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> out(size_t(w) * h * num_samples);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          out[(y * w + x) * num_samples + s] = cwords[doff(x, y, s)];
    return out;
  }

  // Reduces `words` with k_tile_hash into a single atomic-accumulated hash.
  uint32_t HashWordsBuffer(const std::vector<uint32_t>& words) {
    const uint32_t count = uint32_t(words.size());
    id<MTLBuffer> in_buf =
        [dev newBufferWithBytes:words.data()
                         length:std::max<size_t>(1u, words.size()) * 4u
                        options:MTLResourceStorageModeShared];
    uint32_t zero = 0u;
    id<MTLBuffer> out_buf = [dev newBufferWithBytes:&zero
                                             length:4u
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> cnt_buf = [dev newBufferWithBytes:&count
                                             length:4u
                                            options:MTLResourceStorageModeShared];
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tile_hash"]
                                      error:&err];
    REQUIRE(pso != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:in_buf offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    [enc setBuffer:cnt_buf offset:0 atIndex:2];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > count) tg = count ? count : 1;
    [enc dispatchThreads:MTLSizeMake(count ? count : 1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    return *static_cast<const uint32_t*>(out_buf.contents);
  }

  // Uploads `words` (w*h of them, row-major) into a wxh host texture and hashes
  // it with k_tile_hash_tex. mode 0 = RGBA8Unorm, 1 = R32Float. Proves that
  // hashing the attachment == hashing the raw words (HashWordsBuffer).
  uint32_t HashTexture(const std::vector<uint32_t>& words, uint32_t w,
                       uint32_t h, uint32_t mode) {
    MTLPixelFormat pf =
        mode == 0u ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatR32Float;
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pf
                                     width:w
                                    height:h
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
    REQUIRE(tex != nil);
    // The 4 raw bytes per word ARE the little-endian texel bits for both
    // formats (RGBA8Unorm: R,G,B,A lanes; R32Float: the f32 bit pattern).
    [tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
           mipmapLevel:0
             withBytes:words.data()
           bytesPerRow:w * 4u];
    uint32_t zero = 0u;
    id<MTLBuffer> out_buf = [dev newBufferWithBytes:&zero
                                             length:4u
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> mode_buf = [dev newBufferWithBytes:&mode
                                              length:4u
                                             options:MTLResourceStorageModeShared];
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tile_hash_tex"]
                                      error:&err];
    REQUIRE(pso != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setTexture:tex atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:0];
    [enc setBuffer:mode_buf offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(w, h, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    return *static_cast<const uint32_t*>(out_buf.contents);
  }
};

// Mirror of the MSL EdramIn struct (std layout, 8 x uint32).
struct EdramIn {
  uint32_t px, py, base, pitch, msaa, is_depth, fmt_log2, sample;
};

// Count and report the first few mismatches so a break is diagnosable.
static uint32_t CountMismatches(const std::vector<uint32_t>& codes,
                                const std::vector<uint32_t>& gpu,
                                const std::vector<uint32_t>& cpu) {
  uint32_t bad = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    if (gpu[i] != cpu[i]) {
      if (bad < 8) {
        WARN("code=0x" << std::hex << codes[i] << " gpu=0x" << gpu[i]
                       << " cpu=0x" << cpu[i]);
      }
      ++bad;
    }
  }
  return bad;
}

}  // namespace

TEST_CASE("Metal device is available for L1", "[l1][metal]") {
  MetalCompute mc;
  if (!mc.Init()) {
    WARN("No Metal device / MSL compile failed -- skipping L1 (expected in a "
         "headless CI without a GPU).");
    SUCCEED("L1 skipped: no Metal device");
    return;
  }
  SUCCEED("Metal device present");
}

TEST_CASE("Metal 7e3 decode matches CPU oracle bit-for-bit", "[l1][metal][float7e3]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<uint32_t> codes;
  for (uint32_t c = 0; c < (1u << 10); ++c) codes.push_back(c);  // full 10-bit space
  std::vector<uint32_t> cpu(codes.size());
  for (size_t i = 0; i < codes.size(); ++i) {
    float f = xenos::Float7e3To32(codes[i]);
    std::memcpy(&cpu[i], &f, 4);
  }
  auto gpu = mc.Run("k_7e3", codes);
  REQUIRE(CountMismatches(codes, gpu, cpu) == 0u);
}

TEST_CASE("Metal 20e4 decode matches CPU oracle bit-for-bit", "[l1][metal][float24]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<uint32_t> codes;
  // 24-bit space is 16M; sweep a dense stride plus all denormals (exp==0).
  for (uint32_t c = 0; c < (1u << 20); ++c) codes.push_back(c);        // all denormals
  for (uint32_t c = 0; c < (1u << 24); c += 251u) codes.push_back(c);  // normals stride
  std::vector<uint32_t> cpu(codes.size());
  for (size_t i = 0; i < codes.size(); ++i) {
    float f = xenos::Float20e4To32(codes[i]);
    std::memcpy(&cpu[i], &f, 4);
  }
  auto gpu = mc.Run("k_20e4", codes);
  REQUIRE(CountMismatches(codes, gpu, cpu) == 0u);
}

TEST_CASE("Metal unorm24 decode matches CPU oracle bit-for-bit", "[l1][metal][unorm24]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<uint32_t> codes;
  for (uint32_t c = 0; c < (1u << 24); c += 97u) codes.push_back(c);
  codes.push_back(0xFFFFFFu);
  codes.push_back(0xC00000u);  // the 2_10_10_10 clear the oracle comment calls out
  std::vector<uint32_t> cpu(codes.size());
  for (size_t i = 0; i < codes.size(); ++i) {
    float f = xenos::UNorm24To32(codes[i]);
    std::memcpy(&cpu[i], &f, 4);
  }
  auto gpu = mc.Run("k_unorm24", codes);
  REQUIRE(CountMismatches(codes, gpu, cpu) == 0u);
}

TEST_CASE("Metal GpuSwap matches CPU oracle bit-for-bit", "[l1][metal][endian]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<uint32_t> codes;
  for (uint32_t c = 0; c < 65536u; ++c) codes.push_back(c * 65537u);  // spread 32 bits
  struct { xenos::Endian e; uint32_t mode; } modes[] = {
      {xenos::Endian::kNone, 0u},
      {xenos::Endian::k8in16, 1u},
      {xenos::Endian::k8in32, 2u},
      {xenos::Endian::k16in32, 3u},
  };
  for (auto m : modes) {
    std::vector<uint32_t> cpu(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
      cpu[i] = xenos::GpuSwap(codes[i], m.e);
    }
    auto gpu = mc.Run("k_gpuswap", codes, /*has_mode=*/true, m.mode);
    REQUIRE(CountMismatches(codes, gpu, cpu) == 0u);
  }
}

// The redesign-critical one: the SHIPPING GPU addresser edram.xesli
// XeEdramOffsetBytes (transcribed to MSL, resolution_scale=1, wrap=false) must
// agree with the CPU oracle XeEdramSnapshotOffsetBytes for EVERY sample across
// 1x/2x/4x, color/depth, 32/64bpp. This is the CPU<->GPU convention guarantee
// the canonical per-sample EDRAM buffer's correctness rests on (task 2.2): if
// the two disagree on where a sample lives, materialize/pack corrupts it.
TEST_CASE("Metal EDRAM offset matches CPU oracle over the parameter space",
          "[l1][metal][edram]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<EdramIn> in;
  // Sweep a spread of pixels through several tiles, every MSAA level with its
  // valid sample count, color+depth, 32+64bpp, and a couple of base/pitch
  // values. Equality holds for any inputs if the transcription is faithful, so
  // physical validity isn't required -- only coverage of the branches.
  const uint32_t bases[] = {0u, 7u};
  const uint32_t pitches[] = {2u, 16u};
  for (uint32_t msaa = 0; msaa <= 2; ++msaa) {
    uint32_t nsamp = (msaa == 0) ? 1u : (msaa == 1) ? 2u : 4u;
    for (uint32_t fmt = 0; fmt <= 1; ++fmt) {
      for (uint32_t is_depth = 0; is_depth <= 1; ++is_depth) {
        for (uint32_t base : bases) {
          for (uint32_t pitch : pitches) {
            for (uint32_t py = 0; py < 40; py += 3) {
              for (uint32_t px = 0; px < 170; px += 3) {
                for (uint32_t s = 0; s < nsamp; ++s) {
                  in.push_back(
                      {px, py, base, pitch, msaa, is_depth, fmt, s});
                }
              }
            }
          }
        }
      }
    }
  }
  std::vector<uint32_t> cpu(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const EdramIn& e = in[i];
    cpu[i] = XeEdramSnapshotOffsetBytes(e.px, e.py, e.base, e.pitch, e.msaa,
                                        e.is_depth != 0u, e.fmt_log2, e.sample);
  }
  auto gpu = mc.RunRaw("k_edram_offset", in.data(),
                       in.size() * sizeof(EdramIn), in.size());
  uint32_t bad = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    if (gpu[i] != cpu[i]) {
      if (bad < 8) {
        const EdramIn& e = in[i];
        WARN("px=" << e.px << " py=" << e.py << " msaa=" << e.msaa
                   << " depth=" << e.is_depth << " fmt=" << e.fmt_log2
                   << " s=" << e.sample << " base=" << e.base
                   << " pitch=" << e.pitch << " gpu=" << gpu[i]
                   << " cpu=" << cpu[i]);
      }
      ++bad;
    }
  }
  REQUIRE(bad == 0u);
}

// The shipping resolve-copy tile walk (Metal-inline math) must assign each
// pixel to the SAME EDRAM tile the canonical addresser does, or the canonical
// resolve wiring would pack a pixel into a different tile than the collapsed
// path resolved it from. The canonical tile index is the addresser's byte
// offset / (80*16 words) / 4. Third transcription of the spec, now pinned.
TEST_CASE("Metal resolve-copy tile walk agrees with canonical tile assignment",
          "[l1][metal][resolvecopy]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  const uint32_t kTileWords =
      xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples;  // 1280
  std::vector<EdramIn> in;
  const uint32_t bases[] = {0u, 11u};
  for (uint32_t msaa = 0; msaa <= 2; ++msaa) {
    for (uint32_t fmt = 0; fmt <= 1; ++fmt) {
      for (uint32_t base : bases) {
        const uint32_t w = 160, h = 40;
        const uint32_t pitch =
            [&] {
              uint32_t tile_w = 80u >> fmt;
              uint32_t ws = w << (msaa >= 2u ? 1u : 0u);
              return (ws + tile_w - 1u) / tile_w;
            }();
        for (uint32_t py = 0; py < h; py += 2) {
          for (uint32_t px = 0; px < w; px += 2) {
            in.push_back({px, py, base, pitch, msaa, 0u, fmt, 0u});
          }
        }
      }
    }
  }
  std::vector<uint32_t> cpu(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const EdramIn& e = in[i];
    uint32_t off = XeEdramSnapshotOffsetBytes(e.px, e.py, e.base, e.pitch,
                                              e.msaa, /*is_depth=*/false,
                                              e.fmt_log2, /*sample=*/0u);
    cpu[i] = (off >> 2) / kTileWords;  // byte offset -> word -> tile index
  }
  auto gpu = mc.RunRaw("k_rcp_tile", in.data(), in.size() * sizeof(EdramIn),
                       in.size());
  uint32_t bad = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    if (gpu[i] != cpu[i]) {
      if (bad < 8) {
        const EdramIn& e = in[i];
        WARN("px=" << e.px << " py=" << e.py << " msaa=" << e.msaa << " fmt="
                   << e.fmt_log2 << " base=" << e.base << " pitch=" << e.pitch
                   << " rcp_tile=" << gpu[i] << " canonical_tile=" << cpu[i]);
      }
      ++bad;
    }
  }
  REQUIRE(bad == 0u);
}

// Consistent tile pitch for a surface (matches L0's derivation): the 64bpp tile
// is 40 samples wide, so it CANNOT use XeEdramPitchTiles (which bakes in 80).
static uint32_t PitchFor(uint32_t w, uint32_t msaa, uint32_t fmt_log2) {
  uint32_t tile_w = 80u >> fmt_log2;
  uint32_t width_samples = w << (msaa >= 2u ? 1u : 0u);  // 4x doubles x
  return (width_samples + tile_w - 1u) / tile_w;
}

// The 2.3 groundwork: run the canonical EDRAM pack/unpack against a REAL
// 10 MB Metal buffer and prove pack->unpack is the identity over a whole
// surface's samples. Identity <=> the shipping addresser indexes the buffer
// correctly AND every sample is injective (a collision would clobber a word and
// break identity). This is the last agent-testable rung before the kernel is
// wired into MetalRenderTargetCache::ResolveColorToSnapshot behind the gate.
TEST_CASE("Canonical EDRAM pack->unpack round-trips a surface's samples",
          "[l1][metal][canonical]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct Cfg {
    uint32_t w, h, base, msaa, is_depth, fmt_log2;
    const char* name;
  } cfgs[] = {
      {48, 24, 0, 0, 0, 0, "1x color 32bpp"},
      {48, 24, 0, 1, 0, 0, "2x color 32bpp"},
      {48, 24, 0, 2, 0, 0, "4x color 32bpp"},
      {48, 24, 13, 2, 0, 0, "4x color 32bpp base=13"},
      {48, 24, 0, 2, 1, 0, "4x depth (40-col swap)"},
      {48, 24, 0, 2, 0, 1, "4x 64bpp"},
  };
  for (const Cfg& cfg : cfgs) {
    const uint32_t nsamp = cfg.msaa == 0 ? 1u : cfg.msaa == 1 ? 2u : 4u;
    const uint32_t pitch = PitchFor(cfg.w, cfg.msaa, cfg.fmt_log2);
    std::vector<Coord> coords;
    std::vector<uint32_t> values;
    for (uint32_t y = 0; y < cfg.h; ++y) {
      for (uint32_t x = 0; x < cfg.w; ++x) {
        for (uint32_t s = 0; s < nsamp; ++s) {
          coords.push_back({x, y, s});
          values.push_back(uint32_t(values.size()) + 1u);  // distinct, nonzero
        }
      }
    }
    // Independently bound-check every offset so the kernel never writes OOB.
    uint32_t max_end = 0;
    for (const Coord& c : coords) {
      uint32_t off = XeEdramSnapshotOffsetBytes(c.px, c.py, cfg.base, pitch,
                                                cfg.msaa, cfg.is_depth != 0u,
                                                cfg.fmt_log2, c.sample);
      uint32_t end = off + (4u << cfg.fmt_log2);
      if (end > max_end) max_end = end;
    }
    REQUIRE(max_end <= xenos::kEdramSizeBytes);

    SurfaceParams sp{cfg.base, pitch, cfg.msaa, cfg.is_depth, cfg.fmt_log2, 0u};
    auto out = mc.RoundTripCanonical(coords, values, sp, xenos::kEdramSizeBytes);
    uint32_t bad = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      if (out[i] != values[i]) {
        if (bad < 8) {
          WARN(cfg.name << ": coord(" << coords[i].px << "," << coords[i].py
                        << ",s" << coords[i].sample << ") expected "
                        << values[i] << " got " << out[i]);
        }
        ++bad;
      }
    }
    INFO(cfg.name);
    REQUIRE(bad == 0u);
  }
}

// The materialize/pack precursor to the live cache wiring (task 3.1): seed
// canonical k_8_8_8_8 words, materialize them into a host RGBA8Unorm texture,
// pack the texture back to canonical, and require the round-trip is the
// identity for arbitrary 32-bit words. Exercises real compute texture
// read/write + the byte-lane encode/decode + canonical addressing together --
// the exact machinery MetalRenderTargetCache will port, proven headlessly.
TEST_CASE("Canonical k_8_8_8_8 materialize->pack round-trips through a texture",
          "[l1][metal][materialize]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct Cfg { uint32_t w, h, base; } cfgs[] = {
      {64, 32, 0}, {80, 16, 0}, {130, 40, 9}};
  for (const Cfg& cfg : cfgs) {
    const uint32_t pitch = PitchFor(cfg.w, 0u, 0u);  // 1x color 32bpp
    std::vector<uint32_t> words(size_t(cfg.w) * cfg.h);
    for (size_t i = 0; i < words.size(); ++i) {
      // Arbitrary 32-bit words: every byte lane must survive (RGBA8Unorm stores
      // each byte exactly, so any word round-trips iff the plumbing is correct).
      words[i] = uint32_t(i * 2654435761u) ^ 0x0F1E2D3Cu;
    }
    SurfaceParams sp{cfg.base, pitch, 0u, 0u, 0u, 0u};
    auto out = mc.RoundTripMaterialize8888(cfg.w, cfg.h, sp, words,
                                           xenos::kEdramSizeBytes);
    uint32_t bad = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      if (out[i] != words[i]) {
        if (bad < 8) {
          WARN("pixel " << (i % cfg.w) << "," << (i / cfg.w) << " expected "
                        << words[i] << " got " << out[i]);
        }
        ++bad;
      }
    }
    INFO("w=" << cfg.w << " h=" << cfg.h << " base=" << cfg.base);
    REQUIRE(bad == 0u);
  }
}

// The sample-collapse fix, proven (task 3.3): materialize per-sample canonical
// into a texture2d_array (one slice per 2x/4x sample), pack it back, and
// require every sample survives independently. This is exactly what the
// collapsed 1x host path cannot do -- the accuracy loss behind Red Dead /
// Katamari MSAA corruption. Distinct per-sample seed words make a collapse
// (any sample overwriting another) impossible to miss.
TEST_CASE("Canonical per-sample MSAA materialize->pack keeps all samples",
          "[l1][metal][materialize][msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct Cfg { uint32_t w, h, base, msaa, samples; const char* name; } cfgs[] = {
      {48, 24, 0, 1u, 2u, "2x"},
      {48, 24, 0, 2u, 4u, "4x"},
      {40, 16, 7, 2u, 4u, "4x base=7"},
  };
  for (const Cfg& cfg : cfgs) {
    const uint32_t pitch = PitchFor(cfg.w, cfg.msaa, 0u);
    std::vector<uint32_t> values(size_t(cfg.w) * cfg.h * cfg.samples);
    for (uint32_t y = 0; y < cfg.h; ++y)
      for (uint32_t x = 0; x < cfg.w; ++x)
        for (uint32_t s = 0; s < cfg.samples; ++s)
          // Distinct across every (pixel, sample) so a collapse is detectable.
          values[(y * cfg.w + x) * cfg.samples + s] =
              (uint32_t((y * cfg.w + x) * cfg.samples + s) * 2654435761u) ^
              0x5A17C0DEu;
    SurfaceParams sp{cfg.base, pitch, cfg.msaa, 0u, 0u,
                     uint32_t(xenos::kEdramSizeBytes / 4u)};
    auto out = mc.RoundTripMaterializeMsaa(cfg.w, cfg.h, cfg.samples, sp, values,
                                           xenos::kEdramSizeBytes);
    uint32_t bad = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      if (out[i] != values[i]) {
        if (bad < 8) WARN(cfg.name << " idx " << i << " expected " << values[i]
                                   << " got " << out[i]);
        ++bad;
      }
    }
    INFO(cfg.name);
    REQUIRE(bad == 0u);
  }
}

// The LIVE technique proof (task 3.3 wiring precursor): render per-sample into a
// REAL texture2d_ms attachment via sample-rate shading, then pack each sample
// back via compute. This is exactly what the cache will do -- draws render into
// a real MSAA attachment, resolves/ownership read individual samples -- so
// proving the round-trip here de-risks the rendering technique before it
// touches the draw path. Distinct per-sample seeds catch any sample loss.
TEST_CASE("Real MSAA attachment per-sample render->pack round-trips",
          "[l1][metal][msaa][live]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct Cfg { uint32_t w, h, base, msaa, samples; const char* name; } cfgs[] = {
      {48, 24, 0, 1u, 2u, "2x"},
      {48, 24, 0, 2u, 4u, "4x"},
      {40, 16, 5, 2u, 4u, "4x base=5"},
  };
  for (const Cfg& cfg : cfgs) {
    const uint32_t pitch = PitchFor(cfg.w, cfg.msaa, 0u);
    std::vector<uint32_t> values(size_t(cfg.w) * cfg.h * cfg.samples);
    for (uint32_t y = 0; y < cfg.h; ++y)
      for (uint32_t x = 0; x < cfg.w; ++x)
        for (uint32_t s = 0; s < cfg.samples; ++s)
          values[(y * cfg.w + x) * cfg.samples + s] =
              (uint32_t((y * cfg.w + x) * cfg.samples + s) * 2246822519u) ^
              0x1234ABCDu;
    SurfaceParams sp{cfg.base, pitch, cfg.msaa, 0u, 0u,
                     uint32_t(xenos::kEdramSizeBytes / 4u)};
    auto out = mc.RoundTripMsaaReal(cfg.w, cfg.h, cfg.samples, sp, values,
                                    xenos::kEdramSizeBytes);
    uint32_t bad = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      if (out[i] != values[i]) {
        if (bad < 8) WARN(cfg.name << " idx " << i << " expected " << values[i]
                                   << " got " << out[i]);
        ++bad;
      }
    }
    INFO(cfg.name);
    REQUIRE(bad == 0u);
  }
}

// Depth substrate for the cross-MSAA slice. Both guest depth encodings must
// round-trip losslessly through a REAL host D32Float depth texture -- the
// invariant the cross-MSAA-as-canonical-relabel rests on for JC2's kD24FS8/
// kD24S8 buffers. Runs the FULL domain of each format (Codex: the [0,1] sweep
// was not the whole domain; unorm24 must be proven, not declared lossy).
namespace {
// Round-trips `d24` through a wxh 1x depth tile of the given format and reports
// how many complete words fail to survive. fmt: 2=float24 (kD24FS8),
// 1=unorm24 (kD24S8). A DISTINCT stencil byte is folded into each pixel's word
// so the full (d24<<8)|stencil round-trip (incl. the 8 bit-plane passes) is
// exercised, not just depth.
uint32_t DepthRoundTripBad(MetalCompute& mc, const std::vector<uint32_t>& d24,
                           uint32_t fmt) {
  const uint32_t w = 80, h = 16;  // one 32bpp tile at 1x (1280 pixels).
  const uint32_t pitch = PitchFor(w, 0u, 0u);
  SurfaceParams sp{0u, pitch, 0u, fmt, 0u,
                   uint32_t(xenos::kEdramSizeBytes / 4u)};
  std::vector<uint32_t> words(d24.size());
  for (uint32_t i = 0; i < d24.size(); ++i)
    words[i] = ((d24[i] & 0xFFFFFFu) << 8) | ((i * 37u + 1u) & 0xFFu);
  auto out = mc.RoundTripDepth(w, h, sp, words, xenos::kEdramSizeBytes);
  uint32_t bad = 0;
  for (uint32_t i = 0; i < words.size(); ++i) {
    if (out[i] != words[i]) {
      if (bad < 8)
        WARN("fmt " << fmt << " idx " << i << " expected " << words[i]
                    << " got " << out[i]);
      ++bad;
    }
  }
  return bad;
}

// Per-sample 4x variant: one 4x tile (40x8 px x 4 samples = 1280 words), a
// DISTINCT (d24, stencil) per (pixel, sample). Returns failed-word count.
uint32_t MsaaDepthBad(MetalCompute& mc, uint32_t fmt) {
  const uint32_t w = 40, h = 8, ns = 4;
  const uint32_t pitch = PitchFor(w, /*4x=*/2u, 0u);
  SurfaceParams sp{0u, pitch, /*4x=*/2u, fmt, 0u,
                   uint32_t(xenos::kEdramSizeBytes / 4u)};
  std::vector<uint32_t> words(size_t(w) * h * ns);
  for (uint32_t i = 0; i < words.size(); ++i) {
    uint32_t d = (i * 2654435761u) & 0xFFFFFFu;   // distinct depth per sample
    uint32_t s = (i * 37u + 1u) & 0xFFu;          // distinct stencil per sample
    words[i] = (d << 8) | s;
  }
  auto out = mc.RoundTripDepthMsaa(w, h, ns, sp, words, xenos::kEdramSizeBytes);
  uint32_t bad = 0;
  for (uint32_t i = 0; i < words.size(); ++i) {
    if (out[i] != words[i]) {
      if (bad < 8)
        WARN("fmt " << fmt << " idx " << i << " expected " << words[i]
                    << " got " << out[i]);
      ++bad;
    }
  }
  return bad;
}
}  // namespace

TEST_CASE("Canonical kD24FS8 (float24) depth+stencil word round-trips over [0,2)",
          "[l1][metal][materialize][depth]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Sweep all 24 bits, so exponent 15 (guest depth in [1,2)) is included: the
  // *0.5//*2 mapping keeps it inside the [0,1] attachment instead of clamping.
  std::vector<uint32_t> d24(80u * 16u);
  for (uint32_t i = 0; i < d24.size(); ++i)
    d24[i] = uint32_t(uint64_t(i) * 0xFFFFFFu / (d24.size() - 1));
  d24[0] = 0u;
  d24[1] = 0xFFFFFFu;  // just under 2.0 -- the value the [0,1]-only test clamped
  d24[2] = 0xF00000u;  // exactly 1.0 (the old ceiling)
  d24[3] = 0xF00001u;  // just over 1.0 -- must now survive
  REQUIRE(DepthRoundTripBad(mc, d24, /*float24=*/2u) == 0u);
}

TEST_CASE("Canonical kD24S8 (unorm24) depth+stencil word round-trips exactly near 1.0",
          "[l1][metal][materialize][depth]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Adversarial: the values my "24>23 bits" claim said would collide. Densely
  // fill the top of the range and the 0.5 exponent boundary with ADJACENT n24
  // (step 1) -- if any two collide through D32Float, the count drops and this
  // fails. 1280 pixels: 640 near 1.0, 640 straddling 0x800000.
  std::vector<uint32_t> d24(80u * 16u);
  for (uint32_t i = 0; i < 640u; ++i) d24[i] = 0xFFFFFFu - i;        // near 1.0
  for (uint32_t i = 0; i < 640u; ++i) d24[640u + i] = 0x800000u - 320u + i;
  REQUIRE(DepthRoundTripBad(mc, d24, /*unorm24=*/1u) == 0u);
  // And a full-range stride, including 0 and the exact top.
  std::vector<uint32_t> sweep(80u * 16u);
  for (uint32_t i = 0; i < sweep.size(); ++i)
    sweep[i] = uint32_t(uint64_t(i) * 0xFFFFFFu / (sweep.size() - 1));
  REQUIRE(DepthRoundTripBad(mc, sweep, /*unorm24=*/1u) == 0u);
}

TEST_CASE("Canonical 4x per-sample depth+stencil round-trips every sample",
          "[l1][metal][materialize][depth][msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Both guest depth formats, all four samples independent: the sample
  // dimension the collapsed host cannot hold and the cross-MSAA slice depends
  // on. Distinct depth AND stencil per (pixel, sample) makes any collapse or
  // cross-sample leak fail loudly.
  REQUIRE(MsaaDepthBad(mc, /*float24=*/2u) == 0u);
  REQUIRE(MsaaDepthBad(mc, /*unorm24=*/1u) == 0u);
}

// ---------------------------------------------------------------------------
// L3b hash primitive: the tile-range content hash the live ownership-transition
// manifest will use to detect value corruption (before/after a transfer). Prove
// here, headless, three properties the live path depends on:
//   (1) the GPU reduction matches the CPU oracle bit-for-bit;
//   (2) it is thread-order-independent yet position-sensitive;
//   (3) hashing a host attachment == hashing its raw EDRAM words.
namespace {
std::vector<uint32_t> SpreadWords(uint32_t n) {
  std::vector<uint32_t> v(n);
  for (uint32_t i = 0; i < n; ++i)
    v[i] = (i * 2654435761u) ^ (0xB5297A4Du + (i << 3));  // spread all 32 bits
  return v;
}
}  // namespace

TEST_CASE("Metal tile-hash buffer reduction matches the CPU oracle",
          "[l1][metal][hash]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  for (uint32_t n : {1u, 2u, 7u, 64u, 1000u, 4096u}) {
    auto words = SpreadWords(n);
    INFO("n=" << n);
    REQUIRE(mc.HashWordsBuffer(words) == XeTileHash(words));
  }
}

TEST_CASE("Metal tile-hash is content-addressed (a single changed word moves it)",
          "[l1][metal][hash]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  auto words = SpreadWords(512);
  uint32_t h0 = mc.HashWordsBuffer(words);
  // Flip one bit of one word: the hash must change (corruption is observable).
  words[123] ^= 1u;
  REQUIRE(mc.HashWordsBuffer(words) != h0);
  // Move a word to a different position (swap two): position-sensitivity means
  // the hash changes even though the multiset of words is identical.
  auto swapped = SpreadWords(512);
  std::swap(swapped[10], swapped[400]);
  REQUIRE(mc.HashWordsBuffer(swapped) != XeTileHash(SpreadWords(512)));
}

TEST_CASE("Metal tile-hash of an RGBA8Unorm attachment == hashing its words",
          "[l1][metal][hash][texture]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  const uint32_t w = 40, h = 16;  // one 32bpp EDRAM tile's pixel block at 1x.
  auto words = SpreadWords(w * h);
  // RGBA8Unorm is a bijection on 32-bit words (each byte lane is exact unorm8),
  // so hashing the attachment must equal hashing the raw words.
  REQUIRE(mc.HashTexture(words, w, h, 0u) == XeTileHash(words));
  REQUIRE(mc.HashTexture(words, w, h, 0u) == mc.HashWordsBuffer(words));
}

TEST_CASE("Metal tile-hash of an R32Float attachment == hashing its words",
          "[l1][metal][hash][texture]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  const uint32_t w = 40, h = 16;
  // Use FINITE-float bit patterns: NaN payloads can be canonicalized on a
  // texture read, which is not a hash bug but would break bit-exactness. Real
  // EDRAM luminance words the reinterpret path carries are finite.
  std::vector<uint32_t> words(w * h);
  for (uint32_t i = 0; i < words.size(); ++i) {
    float f = (float(i) - 128.0f) * 0.375f + 1e-3f;
    std::memcpy(&words[i], &f, 4);
  }
  REQUIRE(mc.HashTexture(words, w, h, 1u) == XeTileHash(words));
  REQUIRE(mc.HashTexture(words, w, h, 1u) == mc.HashWordsBuffer(words));
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
