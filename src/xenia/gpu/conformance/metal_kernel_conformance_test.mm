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
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/edram_layout.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/metal/metal_edram_authority.h"
#include "xenia/gpu/metal/metal_host_depth_reconcile.h"
#include "xenia/gpu/metal/metal_edram_msl.h"
#include "xenia/gpu/metal/metal_upload_arena.h"
#include "xenia/gpu/trace_analysis_decode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::ComputeCanonicalRangeTextureHeight;
using metal::ComputeCanonicalRangeTextureRowStart;
using metal::ComputeMsaaCompanionAllocationHeight;
using metal::MetalEdramAuthorityTracker;
using metal::ParseMetalEnvironmentFlag;

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
// The extra conformance kernels appended AFTER the shared kEdramMsl
// (metal_edram_msl.h). The metal_stdlib prelude, the shared addresser
// xe_edram_offset_bytes, and struct SurfaceParams all come from that shared
// source -- the SAME one the live cross-MSAA path compiles -- so the test and
// the live path cannot drift. Init() concatenates kEdramMsl + this.
static const char* kKernelSource = R"MSL(
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

kernel void k_resolve_color_value(
    device const uint* canonical [[buffer(0)]],
    device uint4* out [[buffer(1)]],
    constant CanonicalResolveParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.width || gid.y >= p.height) return;
  float4 value = xe_resolve_color_value(
      canonical, p, uint2(p.source_x + gid.x, p.source_y + gid.y));
  out[gid.y * p.width + gid.x] = as_type<uint4>(value);
}

// Color materialize/pack helpers and kernels live in the shared kEdramMsl
// production source prepended by Init(). The tests below invoke those exact
// functions, so the live cache and the conformance oracle cannot drift.

// Stage B (implement-metal-depth-semantics 3.3): pin the RNE f20e4 encoder
// and the host-depth reconcile against their CPU oracles. Reconcile consumes
// interleaved pairs (word, host-bits) so the unary harness carries it.
kernel void k_f16ext_unpack(device const uint* in [[buffer(0)]],
                            device uint* out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
  out[i] = as_type<uint>(xe_f16_ext_to_f32(in[i]));
}
kernel void k_f16ext_pack(device const uint* in [[buffer(0)]],
                          device uint* out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  out[i] = xe_f32_to_f16_ext(as_type<float>(in[i]));
}
kernel void k_f20e4_rne(device const uint* in [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  out[i] = xe_f32_to_f20e4_rne(as_type<float>(in[i]));
}
// One thread per ELEMENT (matching the harness grid): even threads own a
// (word, host) pair and write the reconciled bits; odd threads write 0.
kernel void k_host_depth_reconcile_trunc(
    device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i & 1u) { out[i] = 0u; return; }
  out[i] = as_type<uint>(
      xe_host_depth_reconcile(in[i], as_type<float>(in[i + 1u]), 0u));
}
kernel void k_host_depth_reconcile_rne(
    device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i & 1u) { out[i] = 0u; return; }
  out[i] = as_type<uint>(
      xe_host_depth_reconcile(in[i], as_type<float>(in[i + 1u]), 1u));
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
  out[i] = ((exponent + 124u) << 23) | (mantissa << 16);
}

kernel void k_metal_sample_to_guest(device const uint* in [[buffer(0)]],
                                    device uint* out [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
  uint packed = in[i];
  out[i] = xe_metal_sample_to_guest(packed >> 8u, packed & 0xFFu);
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

kernel void k_depth_word_to_host(device const uint* in [[buffer(0)]],
                                 device uint* out [[buffer(1)]],
                                 constant uint& fmt [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  out[i] = as_type<uint>(xe_depth_word_to_host(in[i], fmt));
}

kernel void k_depth_word_to_guest(device const uint* in [[buffer(0)]],
                                  device uint* out [[buffer(1)]],
                                  constant uint& fmt [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
  out[i] = as_type<uint>(xe_depth_word_to_guest(in[i], fmt));
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

// Stage B guest-texture tiling. The CPU writeback scatter addresses every
// resolve destination texel with texture_address::Tiled2D, so the GPU scatter
// is only correct if this port agrees for every input the resolve path can
// produce. One disagreeing bit scrambles the destination.
struct Tiled2DIn {
  uint x, y, pitch_aligned, bpp_log2;
};
kernel void k_tiled2d_offset(device const Tiled2DIn* in [[buffer(0)]],
                             device uint* out [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  Tiled2DIn t = in[i];
  out[i] = xe_tiled2d_offset(t.x, t.y, t.pitch_aligned, t.bpp_log2);
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

// Color/depth/stencil materialize/pack kernels plus their addressing and format
// helpers live in the shared kEdramMsl
// (metal_edram_msl.h), prepended by Init() -- the SAME source the live
// cross-MSAA path compiles. The [depth] cases below exercise exactly that source.
)MSL";

// Mirrors of the MSL pack/unpack structs -- layouts must match exactly or the
// round-trip corrupts (a stride mismatch would fail loudly, by design).
struct Coord {
  uint32_t px, py, sample;
};
using SurfaceParams = EdramSurfaceParams;
struct CanonicalResolveParamsHost {
  SurfaceParams surface;
  uint32_t source_x, source_y, dest_y, sample_select;
  uint32_t color_format;
  uint32_t dest_swap;
  uint32_t dest_red_only;
  float exp_bias_scale;
  uint32_t width, height;
  uint32_t dest_format, dest_bpp_log2, dest_endian, raw_copy;
  // Stage B tiled scatter (metal_edram_msl.h CanonicalResolveParams). These
  // MUST be mirrored here even though no conformance case enables the scatter:
  // the struct is a constant-buffer layout, so a host mirror shorter than the
  // MSL struct makes the kernel read `scatter_enable` from whatever follows the
  // upload. That is not hypothetical — it is exactly how this mirror broke the
  // packed-resolve case when the fields were added (garbage 0xcdcdcdcd read as
  // a nonzero enable, sending every write to a wrong address).
  uint32_t scatter_enable;
  uint32_t dest_base, dest_pitch_aligned;
  uint32_t dest_offset_x, dest_offset_y, dest_extent_bytes;
  // copy_dest_number (fixed-point pack semantics) and the integer-destination
  // component widths for float snapshots; both 0 = unsigned fraction, the
  // pack every case below expects unless it sets dest_number itself.
  uint32_t dest_number;
  uint32_t dest_int_bits;
};
static_assert(sizeof(CanonicalResolveParamsHost) == 28u * sizeof(uint32_t));

// Host mirror of the shared MSL CanonicalClearParams (metal_edram_msl.h). The
// clear scatter writes one clear word (two for 64bpp) at each declared
// (pixel, sample) physical offset -- so the layout must match dword-for-dword.
struct CanonicalClearParamsHost {
  SurfaceParams surface;  // base, pitch, msaa, is_depth, fmt_log2, words
  uint32_t source_x, source_y, width, height;
  uint32_t clear_lo, clear_hi;
};
static_assert(sizeof(CanonicalClearParamsHost) == 12u * sizeof(uint32_t));

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
    // Prepend the SHARED production depth/stencil MSL (the same source the live
    // cross-MSAA path compiles) so the test exercises the live helpers, not a
    // copy. The extra conformance kernels in kKernelSource follow it.
    std::string combined = std::string(xe::gpu::metal::kEdramMsl);
    combined += kKernelSource;
    lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:combined.c_str()]
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
                                           const SurfaceParams& sp,
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
                                            length:sizeof(SurfaceParams)
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

  std::vector<uint32_t> ResolveRawDepth(
      uint32_t w, uint32_t h, const SurfaceParams& sp,
      uint32_t sample_select, const std::vector<uint32_t>& values,
      size_t canonical_bytes) {
    REQUIRE(values.size() == size_t(w) * h);
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* canonical_words =
        static_cast<uint32_t*>(canonical.contents);
    const uint32_t word_count = uint32_t(canonical_bytes / sizeof(uint32_t));
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t word = XeEdramSnapshotOffsetBytes(
                            x, y, sp.base, sp.pitch, sp.msaa, true, 0u,
                            sample_select) >>
                        2u;
        canonical_words[word % word_count] = values[y * w + x];
      }
    }
    id<MTLBuffer> out =
        [dev newBufferWithLength:values.size() * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    CanonicalResolveParamsHost params{sp, 0u, 0u, 0u, sample_select,
                                      0u, 0u, 0u, 1.0f, w, h,
                                      0u, 0u, 0u, 0u};
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [dev
        newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"resolve_canonical_depth_raw"]
                                      error:&error];
    REQUIRE(pipeline != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:canonical offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(w, h, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> result(values.size());
    std::memcpy(result.data(), out.contents,
                result.size() * sizeof(uint32_t));
    return result;
  }

  std::vector<uint32_t> ResolveColorOne(
      const SurfaceParams& sp, uint32_t sample_select, bool dest_swap,
      bool dest_red_only, uint32_t color_format, float exp_bias_scale,
      const std::array<uint32_t, 4>& sample_words,
      size_t canonical_bytes,
      const std::array<uint32_t, 4>* second_sample_words = nullptr) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* words = static_cast<uint32_t*>(canonical.contents);
    const uint32_t word_count = uint32_t(canonical_bytes / 4u);
    for (uint32_t sample = 0; sample < 4u; ++sample) {
      uint32_t offset = XeEdramSnapshotOffsetBytes(
                            0u, 0u, sp.base, sp.pitch, sp.msaa, false,
                            sp.format_ints_log2, sample) >>
                        2u;
      words[offset % word_count] = sample_words[sample];
      if (second_sample_words) {
        words[(offset + 1u) % word_count] = (*second_sample_words)[sample];
      }
    }
    id<MTLBuffer> out =
        [dev newBufferWithLength:4u * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];
    CanonicalResolveParamsHost params{sp, 0u, 0u, 0u, sample_select, 0u,
                                      uint32_t(dest_swap),
                                      uint32_t(dest_red_only), exp_bias_scale,
                                      1u, 1u, 0u, 0u, 0u, 0u};
    params.color_format = color_format;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [dev
        newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"k_resolve_color_value"]
                                      error:&error];
    REQUIRE(pipeline != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:canonical offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::vector<uint32_t> result(4u);
    std::memcpy(result.data(), out.contents, 4u * sizeof(uint32_t));
    return result;
  }

  std::array<uint8_t, 16> ResolveColorPackedOne(
      const SurfaceParams& sp, uint32_t sample_select, bool dest_swap,
      uint32_t color_format, float exp_bias_scale, uint32_t dest_format,
      uint32_t dest_bpp_log2, uint32_t dest_endian, bool raw_copy,
      const std::array<uint32_t, 4>& sample_words,
      const std::array<uint32_t, 4>* second_sample_words = nullptr,
      uint32_t dest_number = 0u) {
    const size_t canonical_bytes = xenos::kEdramSizeBytes;
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* words = static_cast<uint32_t*>(canonical.contents);
    const uint32_t word_count = uint32_t(canonical_bytes / 4u);
    for (uint32_t sample = 0; sample < 4u; ++sample) {
      uint32_t offset = XeEdramSnapshotOffsetBytes(
                            0u, 0u, sp.base, sp.pitch, sp.msaa, false,
                            sp.format_ints_log2, sample) >>
                        2u;
      words[offset % word_count] = sample_words[sample];
      if (second_sample_words) {
        words[(offset + 1u) % word_count] = (*second_sample_words)[sample];
      }
    }
    id<MTLBuffer> out =
        [dev newBufferWithLength:16u options:MTLResourceStorageModeShared];
    std::memset(out.contents, 0xCD, 16u);
    CanonicalResolveParamsHost params{
        sp, 0u, 0u, 0u, sample_select, color_format, uint32_t(dest_swap),
        uint32_t(dest_bpp_log2 == 0u),
        exp_bias_scale, 1u, 1u, dest_format, dest_bpp_log2, dest_endian,
        uint32_t(raw_copy)};
    params.dest_number = dest_number;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [dev
        newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"resolve_canonical_color_packed"]
                                      error:&error];
    REQUIRE(pipeline != nil);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:canonical offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    std::array<uint8_t, 16> result;
    std::memcpy(result.data(), out.contents, result.size());
    return result;
  }

  // Materialize k_8_8_8_8 canonical words into a host RGBA8Unorm texture, then
  // pack the texture back to canonical -- the round-trip the cache's 3.1
  // materialize/pack will perform. `words` holds one word per (y*w+x); returns
  // the re-packed words read back from the canonical buffer at the same offsets.
  std::vector<uint32_t> RoundTripMaterialize8888(uint32_t w, uint32_t h,
                                                 const SurfaceParams& sp,
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
                                            length:sizeof(SurfaceParams)
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

  std::vector<uint32_t> RoundTripMaterializeColor(
      uint32_t w, uint32_t h, const SurfaceParams& sp,
      const std::vector<uint32_t>& words, size_t canonical_bytes,
      MTLPixelFormat pixel_format, uint32_t color_format) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    REQUIRE(canonical != nil);
    std::memset(canonical.contents, 0, canonical_bytes);
    uint32_t* cwords = static_cast<uint32_t*>(canonical.contents);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t off = XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch,
                                                  sp.msaa, false, 0u, 0u);
        cwords[off >> 2] = words[y * w + x];
      }
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:pixel_format
                                     width:w
                                    height:h
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
    REQUIRE(tex != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    id<MTLComputePipelineState> mat = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"materialize_color"]
                                      error:&err];
    REQUIRE(mat != nil);
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_color"]
                                      error:&err];
    REQUIRE(pk != nil);
    auto dispatch = [&](id<MTLComputeCommandEncoder> enc,
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
      [enc setBytes:&color_format length:sizeof(color_format) atIndex:2];
      dispatch(enc, mat);
      [enc endEncoding];
    }
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setTexture:tex atIndex:0];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      [enc setBytes:&color_format length:sizeof(color_format) atIndex:2];
      dispatch(enc, pk);
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
                                                 const SurfaceParams& sp,
                                                 const std::vector<uint32_t>& values,
                                                 size_t canonical_bytes) {
    id<MTLBuffer> canonical =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    REQUIRE(canonical != nil);
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
                                            length:sizeof(SurfaceParams)
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
                                          const SurfaceParams& sp,
                                          const std::vector<uint32_t>& values,
                                          size_t canonical_bytes,
                                          MTLPixelFormat pixel_format,
                                          uint32_t color_format) {
    id<MTLBuffer> canonical_in =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> canonical_out =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    REQUIRE(canonical_in != nil);
    REQUIRE(canonical_out != nil);
    std::memset(canonical_in.contents, 0, canonical_bytes);
    std::memset(canonical_out.contents, 0xCD, canonical_bytes);
    uint32_t* input_words = static_cast<uint32_t*>(canonical_in.contents);
    uint32_t* output_words = static_cast<uint32_t*>(canonical_out.contents);
    auto seed_offset = [&](uint32_t x, uint32_t y, uint32_t s) {
      return XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch, sp.msaa, false,
                                        0u, s) >> 2;
    };
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (uint32_t s = 0; s < num_samples; ++s)
          input_words[seed_offset(x, y, s)] =
              values[(y * w + x) * num_samples + s];
    // Real MSAA attachment (private; MSAA can't be shared). Store all samples.
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DMultisample;
    td.pixelFormat = pixel_format;
    td.width = w;
    td.height = h;
    td.sampleCount = num_samples;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> ms = [dev newTextureWithDescriptor:td];
    REQUIRE(ms != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    NSError* err = nil;
    MTLRenderPipelineDescriptor* rpd = [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = [lib newFunctionWithName:@"xe_color_vs"];
    rpd.fragmentFunction =
        [lib newFunctionWithName:@"materialize_color_ms_fs"];
    rpd.colorAttachments[0].pixelFormat = pixel_format;
    rpd.rasterSampleCount = num_samples;
    id<MTLRenderPipelineState> rps =
        [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
    REQUIRE(rps != nil);
    id<MTLComputePipelineState> pk = [dev
        newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"pack_color_ms"]
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
      [enc setFragmentBuffer:canonical_in offset:0 atIndex:0];
      [enc setFragmentBuffer:sp_buf offset:0 atIndex:1];
      [enc setFragmentBytes:&color_format
                     length:sizeof(color_format)
                    atIndex:2];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
    }
    // Pack: compute reads each sample of the MSAA texture back to canonical.
    {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pk];
      [enc setTexture:ms atIndex:0];
      [enc setBuffer:canonical_out offset:0 atIndex:0];
      [enc setBuffer:sp_buf offset:0 atIndex:1];
      [enc setBytes:&color_format length:sizeof(color_format) atIndex:2];
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
          out[(y * w + x) * num_samples + s] =
              output_words[seed_offset(x, y, s)];
    return out;
  }

  // Range synchronization through the same 32/64bpp kernels used by the live
  // cache. Input and output canonical buffers are deliberately separate: only
  // pixels in [start_tile, end_tile) may be copied, while all other output
  // words must retain `sentinel`.
  std::vector<uint32_t> RoundTripColorRange(
      uint32_t w, uint32_t h, uint32_t num_samples, const SurfaceParams& sp,
      const std::vector<uint32_t>& values, size_t canonical_bytes,
      MTLPixelFormat pixel_format, uint32_t color_format,
      uint32_t start_tile, uint32_t end_tile, uint32_t sentinel) {
    REQUIRE(sp.format_ints_log2 <= 1u);
    const uint32_t words_per_sample = 1u << sp.format_ints_log2;
    REQUIRE(values.size() ==
            size_t(w) * h * num_samples * words_per_sample);
    id<MTLBuffer> canonical_in =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> canonical_out =
        [dev newBufferWithLength:canonical_bytes
                         options:MTLResourceStorageModeShared];
    REQUIRE(canonical_in != nil);
    REQUIRE(canonical_out != nil);
    std::memset(canonical_in.contents, 0, canonical_bytes);
    uint32_t* in_words = static_cast<uint32_t*>(canonical_in.contents);
    uint32_t* out_words = static_cast<uint32_t*>(canonical_out.contents);
    std::fill_n(out_words, canonical_bytes / sizeof(uint32_t), sentinel);
    auto word_offset = [&](uint32_t x, uint32_t y, uint32_t s) {
      return XeEdramSnapshotOffsetBytes(x, y, sp.base, sp.pitch, sp.msaa,
                                        false, sp.format_ints_log2, s) >> 2;
    };
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        for (uint32_t s = 0; s < num_samples; ++s) {
          size_t value_index =
              ((size_t(y) * w + x) * num_samples + s) * words_per_sample;
          uint32_t offset = word_offset(x, y, s);
          for (uint32_t word = 0; word < words_per_sample; ++word) {
            in_words[offset + word] = values[value_index + word];
          }
        }
      }
    }

    struct RangeParams {
      uint32_t start_tiles, end_tiles;
    } range{start_tile, end_tile};
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(SurfaceParams)
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> range_buf =
        [dev newBufferWithBytes:&range
                         length:sizeof(range)
                        options:MTLResourceStorageModeShared];
    REQUIRE(sp_buf != nil);
    REQUIRE(range_buf != nil);
    NSError* err = nil;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    REQUIRE(cb != nil);
    const bool wide = sp.format_ints_log2 != 0u;
    const bool raw_view = color_format == 6u || color_format == 7u ||
                          color_format == 14u || color_format == 15u;
    const MTLPixelFormat sync_pixel_format =
        color_format == 6u    ? MTLPixelFormatRG16Uint
        : color_format == 7u  ? MTLPixelFormatRGBA16Uint
        : color_format == 14u ? MTLPixelFormatR32Uint
        : color_format == 15u ? MTLPixelFormatRG32Uint
                              : pixel_format;
    if (num_samples == 1u) {
      MTLTextureDescriptor* td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:pixel_format
                                       width:w
                                      height:h
                                   mipmapped:NO];
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                 MTLTextureUsagePixelFormatView;
      td.storageMode = MTLStorageModeShared;
      id<MTLTexture> storage_texture = [dev newTextureWithDescriptor:td];
      REQUIRE(storage_texture != nil);
      id<MTLTexture> texture =
          raw_view ? [storage_texture
                         newTextureViewWithPixelFormat:sync_pixel_format]
                   : storage_texture;
      REQUIRE(texture != nil);
      id<MTLComputePipelineState> materialize = [dev
          newComputePipelineStateWithFunction:
              [lib newFunctionWithName:
                       raw_view
                           ? (wide ? @"materialize_color64_raw_range"
                                   : @"materialize_color_raw_range")
                           : (wide ? @"materialize_color64_range"
                                   : @"materialize_color_range")]
                                        error:&err];
      REQUIRE(materialize != nil);
      id<MTLComputePipelineState> pack = [dev
          newComputePipelineStateWithFunction:
              [lib newFunctionWithName:
                       raw_view
                           ? (wide ? @"pack_color64_raw_range"
                                   : @"pack_color_raw_range")
                           : (wide ? @"pack_color64_range"
                                   : @"pack_color_range")]
                                        error:&err];
      REQUIRE(pack != nil);
      auto dispatch = [&](id<MTLComputeCommandEncoder> encoder,
                          id<MTLComputePipelineState> pipeline) {
        [encoder setComputePipelineState:pipeline];
        NSUInteger tw = pipeline.threadExecutionWidth;
        NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
        // Range kernels take their first texture row at buffer(7).
        static const uint32_t kXeRowStartZero = 0;
        [encoder setBytes:&kXeRowStartZero length:sizeof(kXeRowStartZero) atIndex:7];
        [encoder dispatchThreads:MTLSizeMake(w, h, 1u)
            threadsPerThreadgroup:MTLSizeMake(tw, th, 1u)];
      };
      {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setBuffer:canonical_in offset:0 atIndex:0];
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:sp_buf offset:0 atIndex:1];
        [encoder setBytes:&color_format length:sizeof(color_format) atIndex:2];
        [encoder setBuffer:range_buf offset:0 atIndex:3];
        dispatch(encoder, materialize);
        [encoder endEncoding];
      }
      {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:canonical_out offset:0 atIndex:0];
        [encoder setBuffer:sp_buf offset:0 atIndex:1];
        [encoder setBytes:&color_format length:sizeof(color_format) atIndex:2];
        [encoder setBuffer:range_buf offset:0 atIndex:3];
        dispatch(encoder, pack);
        [encoder endEncoding];
      }
    } else {
      MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
      td.textureType = MTLTextureType2DMultisample;
      td.pixelFormat = pixel_format;
      td.width = w;
      td.height = h;
      td.sampleCount = num_samples;
      td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                 MTLTextureUsagePixelFormatView;
      td.storageMode = MTLStorageModePrivate;
      id<MTLTexture> storage_texture = [dev newTextureWithDescriptor:td];
      REQUIRE(storage_texture != nil);
      id<MTLTexture> texture =
          raw_view ? [storage_texture
                         newTextureViewWithPixelFormat:sync_pixel_format]
                   : storage_texture;
      REQUIRE(texture != nil);
      MTLRenderPipelineDescriptor* rpd =
          [[MTLRenderPipelineDescriptor alloc] init];
      rpd.vertexFunction = [lib newFunctionWithName:@"xe_color_vs"];
      rpd.fragmentFunction =
          [lib newFunctionWithName:
                   raw_view
                       ? (wide ? @"materialize_color64_raw_ms_range_fs"
                               : @"materialize_color_raw_ms_range_fs")
                       : (wide ? @"materialize_color64_ms_range_fs"
                               : @"materialize_color_ms_range_fs")];
      rpd.colorAttachments[0].pixelFormat = sync_pixel_format;
      rpd.rasterSampleCount = num_samples;
      id<MTLRenderPipelineState> materialize =
          [dev newRenderPipelineStateWithDescriptor:rpd error:&err];
      REQUIRE(materialize != nil);
      id<MTLComputePipelineState> pack = [dev
          newComputePipelineStateWithFunction:
              [lib newFunctionWithName:
                       raw_view
                           ? (wide ? @"pack_color64_raw_ms_range"
                                   : @"pack_color_raw_ms_range")
                           : (wide ? @"pack_color64_ms_range"
                                   : @"pack_color_ms_range")]
                                        error:&err];
      REQUIRE(pack != nil);
      {
        MTLRenderPassDescriptor* rp =
            [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = texture;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        id<MTLRenderCommandEncoder> encoder =
            [cb renderCommandEncoderWithDescriptor:rp];
        [encoder setRenderPipelineState:materialize];
        [encoder setFragmentBuffer:canonical_in offset:0 atIndex:0];
        [encoder setFragmentBuffer:sp_buf offset:0 atIndex:1];
        [encoder setFragmentBytes:&color_format
                           length:sizeof(color_format)
                          atIndex:2];
        [encoder setFragmentBuffer:range_buf offset:0 atIndex:3];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
        [encoder endEncoding];
      }
      {
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        [encoder setComputePipelineState:pack];
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:canonical_out offset:0 atIndex:0];
        [encoder setBuffer:sp_buf offset:0 atIndex:1];
        [encoder setBytes:&color_format length:sizeof(color_format) atIndex:2];
        [encoder setBuffer:range_buf offset:0 atIndex:3];
        NSUInteger tw = pack.threadExecutionWidth;
        NSUInteger th = pack.maxTotalThreadsPerThreadgroup / tw;
        // Range kernels take their first texture row at buffer(7).
        static const uint32_t kXeRowStartZero = 0;
        [encoder setBytes:&kXeRowStartZero length:sizeof(kXeRowStartZero) atIndex:7];
        [encoder dispatchThreads:MTLSizeMake(w, h, num_samples)
            threadsPerThreadgroup:MTLSizeMake(tw, th, 1u)];
        [encoder endEncoding];
      }
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);

    std::vector<uint32_t> result(values.size());
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        for (uint32_t s = 0; s < num_samples; ++s) {
          size_t value_index =
              ((size_t(y) * w + x) * num_samples + s) * words_per_sample;
          uint32_t offset = word_offset(x, y, s);
          for (uint32_t word = 0; word < words_per_sample; ++word) {
            result[value_index + word] = out_words[offset + word];
          }
        }
      }
    }
    return result;
  }

  // Depth canonical round-trip: seed canonical with a d24 word at each pixel's
  // DEPTH offset, materialize into a host D32Float depth texture via an SV_Depth
  // render pass, pack it back with pack_depth, and return the re-packed canonical
  // words. Identity <=> the format's decode/encode (sp.is_depth: 2=float24 with
  // the *0.5//*2 [0,2) mapping, 1=unorm24) and depth addressing are exact
  // inverses through real host depth storage -- the substrate the cross-MSAA
  // slice needs for JC2's kD24FS8/kD24S8 buffers.
  std::vector<uint32_t> RoundTripDepth(uint32_t w, uint32_t h,
                                       const SurfaceParams& sp,
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
    td.usage =
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;  // depth/stencil can't be shared.
    id<MTLTexture> depth = [dev newTextureWithDescriptor:td];
    REQUIRE(depth != nil);
    // X32_Stencil8 view for reading the stencil plane as texture2d<uint>.
    id<MTLTexture> sten_view =
        [depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    REQUIRE(sten_view != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(SurfaceParams)
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
                                           const SurfaceParams& sp,
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
    td.usage =
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth = [dev newTextureWithDescriptor:td];
    REQUIRE(depth != nil);
    id<MTLTexture> sten_view =
        [depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    REQUIRE(sten_view != nil);
    id<MTLBuffer> sp_buf = [dev newBufferWithBytes:&sp
                                            length:sizeof(SurfaceParams)
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

  // Mixed 4x -> 1x -> 4x relabel chain through ONE shared canonical buffer, with
  // a write through the 4x view (the seed materialize) and a SCISSORED write
  // through the 1x view. Returns the final tile words (offsets 0..1279). The
  // caller CPU-replays the same physical-slot writes to get the expected result:
  // identity <=> the relabel preserves untouched samples AND propagates a 1x
  // write to the 4x view -- the live cross-MSAA slice's exact contract. Fixed to
  // one tile: 4x = 40x8 px x 4 samples, 1x = 80x16 px, both 1280 slots.
  std::vector<uint32_t> CrossMsaaChain(uint32_t fmt,
                                       const std::vector<uint32_t>& a4x,
                                       const std::vector<uint32_t>& c1x,
                                       uint32_t rx0, uint32_t ry0, uint32_t rx1,
                                       uint32_t ry1, size_t canonical_bytes) {
    const uint32_t w4 = 40, h4 = 8, ns = 4, w1 = 80, h1 = 16;
    // Tile pitch (PitchFor is defined after this struct): one tile wide for both.
    const uint32_t p4 = ((w4 << 1) + 79u) / 80u, p1 = (w1 + 79u) / 80u;
    SurfaceParams sp4{0u, p4, 2u, fmt, 0u, uint32_t(canonical_bytes / 4)};
    SurfaceParams sp1{0u, p1, 0u, fmt, 0u, uint32_t(canonical_bytes / 4)};
    auto off4 = [&](uint32_t x, uint32_t y, uint32_t s) {
      return XeEdramSnapshotOffsetBytes(x, y, 0u, p4, 2u, true, 0u, s) >> 2;
    };
    auto off1 = [&](uint32_t x, uint32_t y) {
      return XeEdramSnapshotOffsetBytes(x, y, 0u, p1, 0u, true, 0u, 0u) >> 2;
    };
    auto mkbuf = [&](const std::vector<uint32_t>& src4,
                     const std::vector<uint32_t>* src1) {
      id<MTLBuffer> b = [dev newBufferWithLength:canonical_bytes
                                         options:MTLResourceStorageModeShared];
      std::memset(b.contents, 0, canonical_bytes);
      uint32_t* cw = static_cast<uint32_t*>(b.contents);
      if (!src4.empty())
        for (uint32_t y = 0; y < h4; ++y)
          for (uint32_t x = 0; x < w4; ++x)
            for (uint32_t s = 0; s < ns; ++s)
              cw[off4(x, y, s)] = src4[(y * w4 + x) * ns + s];
      if (src1)
        for (uint32_t y = 0; y < h1; ++y)
          for (uint32_t x = 0; x < w1; ++x) cw[off1(x, y)] = (*src1)[y * w1 + x];
      return b;
    };
    id<MTLBuffer> seedA = mkbuf(a4x, nullptr);
    id<MTLBuffer> cC = mkbuf({}, &c1x);
    id<MTLBuffer> canon = [dev newBufferWithLength:canonical_bytes
                                           options:MTLResourceStorageModeShared];
    std::memset(canon.contents, 0, canonical_bytes);
    id<MTLBuffer> sp4b = [dev newBufferWithBytes:&sp4 length:sizeof(sp4)
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> sp1b = [dev newBufferWithBytes:&sp1 length:sizeof(sp1)
                                         options:MTLResourceStorageModeShared];
    // Attachments: one 4x and one 1x Depth32Float_Stencil8, reused across passes.
    auto mktex = [&](uint32_t w, uint32_t h, uint32_t samples) {
      MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
      td.textureType = samples > 1 ? MTLTextureType2DMultisample
                                   : MTLTextureType2D;
      td.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      td.width = w; td.height = h; td.sampleCount = samples;
      td.usage =
          MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
      td.storageMode = MTLStorageModePrivate;
      return [dev newTextureWithDescriptor:td];
    };
    id<MTLTexture> t4 = mktex(w4, h4, ns), t1 = mktex(w1, h1, 1);
    id<MTLTexture> v4 = [t4 newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    id<MTLTexture> v1 = [t1 newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    NSError* err = nil;
    auto pipe = [&](NSString* dep, NSString* sten, uint32_t samples) {
      MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
      d.vertexFunction = [lib newFunctionWithName:@"xe_depth_vs"];
      d.fragmentFunction = [lib newFunctionWithName:dep];
      d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      d.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      d.rasterSampleCount = samples;
      id<MTLRenderPipelineState> dp =
          [dev newRenderPipelineStateWithDescriptor:d error:&err];
      d.fragmentFunction = [lib newFunctionWithName:sten];
      id<MTLRenderPipelineState> sp =
          [dev newRenderPipelineStateWithDescriptor:d error:&err];
      return std::make_pair(dp, sp);
    };
    auto pipe4 = pipe(@"materialize_depth_ms_fs", @"materialize_sten_ms_fs", ns);
    auto pipe1 = pipe(@"materialize_depth_fs", @"materialize_sten_fs", 1);
    // Depth pass ALSO zeroes the stencil (REPLACE ref 0, full mask) so a
    // materialize fully REPLACES stencil even on a loadAction:Load scissored
    // write -- the bit-plane passes only SET bits, so without this a 1x write
    // over existing stencil would OR instead of replace (mirrors the c2d depth
    // state). Caller sets stencilRef 0 before the depth draw, 0xFF before bits.
    MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction = MTLCompareFunctionAlways;
    dsd.depthWriteEnabled = YES;
    MTLStencilDescriptor* zs = [[MTLStencilDescriptor alloc] init];
    zs.stencilCompareFunction = MTLCompareFunctionAlways;
    zs.depthStencilPassOperation = MTLStencilOperationReplace;
    zs.writeMask = 0xFF;
    dsd.frontFaceStencil = zs; dsd.backFaceStencil = zs;
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
      bds.frontFaceStencil = bs; bds.backFaceStencil = bs;
      bit_states[b] = [dev newDepthStencilStateWithDescriptor:bds];
    }
    id<MTLComputePipelineState> pk4 = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_depth_ms"]
                                      error:&err];
    id<MTLComputePipelineState> pk1 = [dev
        newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pack_depth"]
                                      error:&err];
    REQUIRE(pipe4.first); REQUIRE(pipe1.first); REQUIRE(pk4); REQUIRE(pk1);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    // Materialize `src` into `att` (depth + 8 stencil bit-planes). load=Clear
    // starts fresh; load=Load + a scissor writes only the rect (a view write).
    auto materialize = [&](id<MTLTexture> att, std::pair<id<MTLRenderPipelineState>,
                           id<MTLRenderPipelineState>> ps, id<MTLBuffer> src,
                           id<MTLBuffer> spb, bool clear, const MTLScissorRect* sc) {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = att;
      rp.depthAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
      rp.depthAttachment.clearDepth = 0.0;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      rp.stencilAttachment.texture = att;
      rp.stencilAttachment.loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
      rp.stencilAttachment.clearStencil = 0;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      if (sc) [e setScissorRect:*sc];
      [e setRenderPipelineState:ps.first];
      [e setDepthStencilState:dss];
      [e setStencilReferenceValue:0];  // depth pass zeroes stencil (REPLACE 0)
      [e setFragmentBuffer:src offset:0 atIndex:0];
      [e setFragmentBuffer:spb offset:0 atIndex:1];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e setRenderPipelineState:ps.second];
      [e setStencilReferenceValue:0xFF];
      [e setFragmentBuffer:src offset:0 atIndex:0];
      [e setFragmentBuffer:spb offset:0 atIndex:1];
      for (uint32_t b = 0; b < 8; ++b) {
        uint32_t bit = 1u << b;
        [e setDepthStencilState:bit_states[b]];
        [e setFragmentBytes:&bit length:sizeof(bit) atIndex:2];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      }
      [e endEncoding];
    };
    auto pack = [&](id<MTLTexture> att, id<MTLTexture> sview,
                    id<MTLComputePipelineState> pk, id<MTLBuffer> spb, uint32_t w,
                    uint32_t h, uint32_t samples) {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pk];
      [e setTexture:att atIndex:0];
      [e setTexture:sview atIndex:1];
      [e setBuffer:canon offset:0 atIndex:0];
      [e setBuffer:spb offset:0 atIndex:1];
      NSUInteger tw = pk.threadExecutionWidth;
      NSUInteger th = pk.maxTotalThreadsPerThreadgroup / tw;
      [e dispatchThreads:MTLSizeMake(w, h, samples)
          threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
      [e endEncoding];
    };
    // 1) Write A through the 4x view, pack to canonical.
    materialize(t4, pipe4, seedA, sp4b, true, nullptr);
    pack(t4, v4, pk4, sp4b, w4, h4, ns);
    // 2) Relabel to 1x (materialize from canonical), then a SCISSORED write of C
    //    through the 1x view, then pack back to canonical.
    materialize(t1, pipe1, canon, sp1b, true, nullptr);
    MTLScissorRect sc{rx0, ry0, rx1 - rx0, ry1 - ry0};
    materialize(t1, pipe1, cC, sp1b, false, &sc);
    pack(t1, v1, pk1, sp1b, w1, h1, 1);
    // 3) Relabel back to 4x and pack (must be lossless after the 1x write).
    materialize(t4, pipe4, canon, sp4b, true, nullptr);
    pack(t4, v4, pk4, sp4b, w4, h4, ns);
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    const uint32_t* cw = static_cast<const uint32_t*>(canon.contents);
    std::vector<uint32_t> out(1280);
    for (uint32_t i = 0; i < 1280; ++i) out[i] = cw[i];
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
struct Tiled2DIn {
  uint32_t x, y, pitch_aligned, bpp_log2;
};
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

TEST_CASE("Metal binding upload exhaustion fails without publishing stale output",
          "[l1][metal][binding-parity][failure]") {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    SUCCEED("no device");
    return;
  }

  constexpr uint32_t kArenaSize = 256;
  metal::MetalUploadArena arena((__bridge void*)device, kArenaSize);
  REQUIRE(arena.Initialize());

  metal::MetalUploadAllocation first{};
  REQUIRE(arena.Allocate(kArenaSize, kArenaSize, &first));
  REQUIRE(first.buffer != nullptr);
  REQUIRE(first.cpu_ptr != nullptr);

  // IssueDraw zero-initializes every required upload result and exits before
  // encoding when Allocate returns false. Pin the allocator half of that
  // contract on-device: exhaustion is reported synchronously and does not
  // replace the caller's output with an earlier live allocation.
  metal::MetalUploadAllocation failed{};
  CHECK_FALSE(arena.Allocate(sizeof(uint32_t), alignof(uint32_t), &failed));
  CHECK(failed.buffer == nullptr);
  CHECK(failed.cpu_ptr == nullptr);
  CHECK(failed.gpu_address == 0);
  CHECK(failed.offset == 0);
}

TEST_CASE("Metal native MSAA sample IDs map to Xenos storage order",
          "[l1][metal][msaa][sample-order]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }

  // Query the device instead of assuming an undocumented sample-ID order.
  // Metal's default locations on Apple GPUs are the D3D10 standard locations
  // used by the Vulkan backend: 2x is bottom/top, and 4x is TL/TR/BL/BR.
  MTLSamplePosition positions_2x[2];
  MTLSamplePosition positions_4x[4];
  [mc.dev getDefaultSamplePositions:positions_2x count:2u];
  [mc.dev getDefaultSamplePositions:positions_4x count:4u];
  CHECK(positions_2x[0].y > positions_2x[1].y);
  CHECK(positions_4x[0].x < 0.5f);
  CHECK(positions_4x[0].y < 0.5f);
  CHECK(positions_4x[1].x > 0.5f);
  CHECK(positions_4x[1].y < 0.5f);
  CHECK(positions_4x[2].x < 0.5f);
  CHECK(positions_4x[2].y > 0.5f);
  CHECK(positions_4x[3].x > 0.5f);
  CHECK(positions_4x[3].y > 0.5f);

  // Xenos slots are T/B for 2x and TL/BL/TR/BR for 4x. Exercise the shared
  // production MSL helper directly so a materialize+pack round trip cannot
  // hide an equal and opposite permutation at its two boundaries.
  const std::vector<uint32_t> native = {
      (1u << 8u) | 0u, (1u << 8u) | 1u,
      (2u << 8u) | 0u, (2u << 8u) | 1u,
      (2u << 8u) | 2u, (2u << 8u) | 3u};
  const std::vector<uint32_t> expected = {1u, 0u, 0u, 2u, 1u, 3u};
  CHECK(mc.Run("k_metal_sample_to_guest", native) == expected);
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

TEST_CASE("Metal extended-range float16 decode matches CPU oracle bit-for-bit",
          "[l1][metal][float16ext]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<uint32_t> codes;
  for (uint32_t c = 0; c < (1u << 16); ++c) codes.push_back(c);
  std::vector<uint32_t> cpu(codes.size());
  for (size_t i = 0; i < codes.size(); ++i) {
    float f = xenos::Float16ExtendedRangeTo32(codes[i]);
    std::memcpy(&cpu[i], &f, 4);
  }
  auto gpu = mc.Run("k_f16ext_unpack", codes);
  REQUIRE(CountMismatches(codes, gpu, cpu) == 0u);
}

TEST_CASE("Metal extended-range float16 encode matches CPU oracle bit-for-bit",
          "[l1][metal][float16ext]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Every decoded code (fixed points), the midpoints between adjacent codes
  // (rounding ties, including the 65504/65520 hand-off into exponent 31),
  // values beyond the guest range, Inf and NaN.
  std::vector<uint32_t> inputs;
  auto push_f = [&](float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    inputs.push_back(u);
  };
  for (uint32_t c = 0; c < 0x8000u; ++c) {
    float lo = xenos::Float16ExtendedRangeTo32(c);
    float hi = xenos::Float16ExtendedRangeTo32(c + 1);
    push_f(lo);
    push_f(-lo);
    push_f(lo + (hi - lo) * 0.5f);
    push_f(lo + (hi - lo) * 0.75f);
  }
  push_f(65519.0f);
  push_f(65520.0f);
  push_f(131008.0f);
  push_f(131009.0f);
  push_f(1.0e9f);
  push_f(-1.0e9f);
  push_f(INFINITY);
  push_f(-INFINITY);
  push_f(NAN);
  push_f(-0.0f);
  push_f(1.0e-9f);
  std::vector<uint32_t> cpu(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    float f;
    std::memcpy(&f, &inputs[i], 4);
    cpu[i] = xenos::Float32ToFloat16ExtendedRange(f);
  }
  auto gpu = mc.Run("k_f16ext_pack", inputs);
  REQUIRE(CountMismatches(inputs, gpu, cpu) == 0u);
}

TEST_CASE("Metal RNE 20e4 encode matches CPU oracle bit-for-bit",
          "[l1][metal][float24][depth_semantics]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Encode every decoded code (exactness fixed points) plus midpoints between
  // adjacent codes (the rounding decision points) plus saturation inputs.
  std::vector<uint32_t> inputs;
  auto push_f = [&](float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    inputs.push_back(u);
  };
  for (uint32_t c = 0; c < (1u << 24); c += 509u) {
    float lo = xenos::Float20e4To32(c);
    float hi = xenos::Float20e4To32(c + 1);
    push_f(lo);
    push_f(lo + (hi - lo) * 0.5f);  // exact midpoint (tie)
    push_f(lo + (hi - lo) * 0.75f);
  }
  push_f(32.0f);
  push_f(1.0e6f);
  push_f(0.0f);
  std::vector<uint32_t> cpu(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    float f;
    std::memcpy(&f, &inputs[i], 4);
    cpu[i] = xenos::Float32To20e4(f, /*round_to_nearest_even=*/true);
  }
  auto gpu = mc.Run("k_f20e4_rne", inputs);
  REQUIRE(CountMismatches(inputs, gpu, cpu) == 0u);
}

TEST_CASE("Metal host-depth reconcile matches the CPU oracle, both modes",
          "[l1][metal][depth_semantics][host_depth]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Interleaved (guest word, host bits) pairs: exact matches (host wins),
  // one-code clobbers (guest wins), one-ULP host nudges (host wins), and the
  // midpoint whose winner flips with the mode.
  std::vector<uint32_t> pairs;
  auto push_pair = [&](uint32_t d24, float host) {
    uint32_t hbits;
    std::memcpy(&hbits, &host, 4);
    pairs.push_back((d24 << 8) | 0xA5u);
    pairs.push_back(hbits);
  };
  for (uint32_t c = 1; c < (1u << 24); c += 1021u) {
    float host = xenos::Float20e4To32(c) * 0.5f;
    push_pair(c, host);                       // exact -> host wins
    push_pair((c + 1) & 0xFFFFFFu, host);     // clobbered -> guest wins
  }
  const uint32_t lower = (5u << 20) | 0x12345u;
  float mid = (xenos::Float20e4To32(lower) +
               (xenos::Float20e4To32(lower + 1) -
                xenos::Float20e4To32(lower)) *
                   0.5f) *
              0.5f;
  push_pair(lower, mid);
  push_pair(lower + 1, mid);
  for (bool rne : {false, true}) {
    INFO("rne=" << rne);
    std::vector<uint32_t> cpu(pairs.size(), 0u);
    for (size_t i = 0; i + 1 < pairs.size(); i += 2) {
      float host;
      std::memcpy(&host, &pairs[i + 1], 4);
      float r = metal::MetalHostDepthReconcile(pairs[i], host, rne);
      std::memcpy(&cpu[i], &r, 4);
    }
    auto gpu = mc.Run(
        rne ? "k_host_depth_reconcile_rne" : "k_host_depth_reconcile_trunc",
        pairs);
    REQUIRE(CountMismatches(pairs, gpu, cpu) == 0u);
  }
}

TEST_CASE("lane refresh installs the SELECTED source's provenance",
          "[l1][metal][depth_semantics][host_depth][provenance]") {
  // The positive provenance oracle: host depths A and B pack to the SAME
  // guest word but differ in float precision. The lane previously contains B
  // (another plan's value); the shared plan selects a source whose
  // attachment holds A. After the production lane_store_ms_range refresh,
  // every covered lane slot must hold A's bits — so reconciliation returns
  // A, never the stale-but-predicate-satisfying B.
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { SUCCEED("no device"); return; }

  const uint32_t kCode = 0x123456u;
  float value_a = xenos::Float20e4To32(kCode) * 0.5f;  // the selected source
  uint32_t bits_a, bits_b;
  std::memcpy(&bits_a, &value_a, 4);
  bits_b = bits_a + 1;  // +1 ULP: same 20e4 code under truncation
  float value_b;
  std::memcpy(&value_b, &bits_b, 4);
  REQUIRE(xenos::Float32To20e4(value_a * 2.0f, false) == kCode);
  REQUIRE(xenos::Float32To20e4(value_b * 2.0f, false) == kCode);

  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:
                                         xe::gpu::metal::kEdramMsl]
                   options:nil
                     error:&error];
  {
    std::string library_error =
        error ? error.localizedDescription.UTF8String : "(none)";
    INFO(library_error);
    REQUIRE(library != nil);
  }
  id<MTLComputePipelineState> lane_store = [device
      newComputePipelineStateWithFunction:
          [library newFunctionWithName:@"lane_store_ms_range"]
                                    error:&error];
  REQUIRE(lane_store != nil);

  // 80x16 pixels at 2x = 2560 samples = exactly tiles [0,2).
  MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
  desc.textureType = MTLTextureType2DMultisample;
  desc.pixelFormat = MTLPixelFormatDepth32Float;
  desc.width = 80;
  desc.height = 16;
  desc.sampleCount = 2;
  desc.usage = MTLTextureUsageRenderTarget;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> depth = [device newTextureWithDescriptor:desc];
  REQUIRE(depth != nil);

  constexpr uint32_t kWords = 2 * 1280;
  id<MTLBuffer> lane =
      [device newBufferWithLength:kWords * 4
                          options:MTLResourceStorageModeShared];
  float* lane_cpu = static_cast<float*>(lane.contents);
  for (uint32_t i = 0; i < kWords; ++i) lane_cpu[i] = value_b;  // stale B

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  MTLRenderPassDescriptor* clear_pass =
      [MTLRenderPassDescriptor renderPassDescriptor];
  clear_pass.depthAttachment.texture = depth;
  clear_pass.depthAttachment.loadAction = MTLLoadActionClear;
  clear_pass.depthAttachment.storeAction = MTLStoreActionStore;
  clear_pass.depthAttachment.clearDepth = double(value_a);
  [[command renderCommandEncoderWithDescriptor:clear_pass] endEncoding];

  struct { uint32_t base, pitch, msaa, is_depth, fmt_log2, words; } sp{
      0u, 1u, 1u /*2x*/, 2u /*float24*/, 0u, kWords};
  struct { uint32_t start_tiles, end_tiles; } range{0u, 2u};
  id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
  [compute setComputePipelineState:lane_store];
  [compute setTexture:depth atIndex:0];
  [compute setBuffer:lane offset:0 atIndex:0];
  [compute setBytes:&sp length:sizeof(sp) atIndex:1];
  [compute setBytes:&range length:sizeof(range) atIndex:2];
  // Range kernels take their first texture row at buffer(7).
  static const uint32_t kXeRowStartZero = 0;
  [compute setBytes:&kXeRowStartZero length:sizeof(kXeRowStartZero) atIndex:7];
  [compute dispatchThreads:MTLSizeMake(80, 16, 2)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [compute endEncoding];
  [command commit];
  [command waitUntilCompleted];
  REQUIRE(command.status == MTLCommandBufferStatusCompleted);

  uint32_t stale_b = 0, holds_a = 0, other = 0;
  const uint32_t* lane_bits = static_cast<const uint32_t*>(lane.contents);
  for (uint32_t i = 0; i < kWords; ++i) {
    if (lane_bits[i] == bits_a) ++holds_a;
    else if (lane_bits[i] == bits_b) ++stale_b;
    else ++other;
  }
  INFO("A=" << holds_a << " staleB=" << stale_b << " other=" << other);
  CHECK(holds_a == kWords);   // every covered slot carries the SELECTED source
  CHECK(stale_b == 0);        // no stale provenance survives the refresh
  CHECK(other == 0);

  // And the oracle: with the lane refreshed to A, reconciliation returns A —
  // bit-exact — for the guest word both values pack to.
  uint32_t word = (kCode << 8) | 0x5Au;
  float reconciled = metal::MetalHostDepthReconcile(word, lane_cpu[0], false);
  uint32_t reconciled_bits;
  std::memcpy(&reconciled_bits, &reconciled, 4);
  CHECK(reconciled_bits == bits_a);
  CHECK(reconciled_bits != bits_b);
}

TEST_CASE("depth-test discriminator: preserved host precision changes the outcome",
          "[l1][metal][depth_semantics][host_depth][discriminator]") {
  // The spec's corrected oracle: EDRAM hashes CANNOT distinguish preserved
  // host precision from guest-decoded depth (both repack to the same 20e4
  // word). This is the end-to-end discriminator instead: materialize through
  // the PRODUCTION range fragment (reconcile on vs off), then depth-test a
  // quad at a comparison depth D strictly between host A and the decoded
  // value — the two paths must produce OPPOSITE pass/fail results.
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { SUCCEED("no device"); return; }

  const uint32_t kCode = 0x234567u;
  float decoded = xenos::Float20e4To32(kCode) * 0.5f;  // pure guest path
  uint32_t decoded_bits;
  std::memcpy(&decoded_bits, &decoded, 4);
  // f32 carries exactly 3 mantissa bits beyond 20e4 for normal codes: 8 f32
  // ULPs per code. +4 stays strictly inside kCode under truncation.
  uint32_t bits_a = decoded_bits + 4;  // host A: above decoded, same code
  float value_a;
  std::memcpy(&value_a, &bits_a, 4);
  REQUIRE(xenos::Float32To20e4(value_a * 2.0f, false) == kCode);
  float depth_d = decoded + (value_a - decoded) * 0.5f;  // strictly between
  REQUIRE(depth_d > decoded);
  REQUIRE(depth_d < value_a);

  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:
                                         xe::gpu::metal::kEdramMsl]
                   options:nil
                     error:&error];
  REQUIRE(library != nil);
  static const char* kQuadSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct QOut { float4 pos [[position]]; };
vertex QOut disc_vs(uint vid [[vertex_id]], constant float& depth [[buffer(0)]]) {
  QOut o;
  float2 t = float2(float((vid << 1) & 2), float(vid & 2));
  o.pos = float4(t * 2.0 - 1.0, depth, 1.0);
  return o;
}
fragment float4 disc_fs() { return float4(1.0, 0.0, 0.0, 1.0); }
)MSL";
  id<MTLLibrary> quad_library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:kQuadSource]
                   options:nil
                     error:&error];
  REQUIRE(quad_library != nil);

  constexpr uint32_t kWords = 2 * 1280;
  id<MTLBuffer> canonical =
      [device newBufferWithLength:kWords * 4
                          options:MTLResourceStorageModeShared];
  uint32_t* canonical_cpu = static_cast<uint32_t*>(canonical.contents);
  for (uint32_t i = 0; i < kWords; ++i) canonical_cpu[i] = (kCode << 8) | 0x5A;
  id<MTLBuffer> lane =
      [device newBufferWithLength:kWords * 4
                          options:MTLResourceStorageModeShared];
  float* lane_cpu = static_cast<float*>(lane.contents);
  for (uint32_t i = 0; i < kWords; ++i) lane_cpu[i] = value_a;

  MTLTextureDescriptor* depth_desc = [MTLTextureDescriptor new];
  depth_desc.textureType = MTLTextureType2DMultisample;
  depth_desc.pixelFormat = MTLPixelFormatDepth32Float;
  depth_desc.width = 80;
  depth_desc.height = 16;
  depth_desc.sampleCount = 2;
  depth_desc.usage = MTLTextureUsageRenderTarget;
  depth_desc.storageMode = MTLStorageModePrivate;
  MTLTextureDescriptor* color_desc = [MTLTextureDescriptor new];
  color_desc.textureType = MTLTextureType2DMultisample;
  color_desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  color_desc.width = 80;
  color_desc.height = 16;
  color_desc.sampleCount = 2;
  color_desc.usage = MTLTextureUsageRenderTarget;
  color_desc.storageMode = MTLStorageModePrivate;
  MTLTextureDescriptor* resolve_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:80
                                  height:16
                               mipmapped:NO];
  resolve_desc.usage = MTLTextureUsageRenderTarget;

  MTLRenderPipelineDescriptor* mat_desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  mat_desc.vertexFunction = [library newFunctionWithName:@"xe_depth_vs"];
  mat_desc.fragmentFunction =
      [library newFunctionWithName:@"materialize_depth_ms_range_fs"];
  mat_desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  mat_desc.rasterSampleCount = 2;
  id<MTLRenderPipelineState> mat_pipeline =
      [device newRenderPipelineStateWithDescriptor:mat_desc error:&error];
  {
    std::string pipeline_error =
        error ? error.localizedDescription.UTF8String : "(none)";
    INFO(pipeline_error);
    REQUIRE(mat_pipeline != nil);
  }
  MTLRenderPipelineDescriptor* quad_desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  quad_desc.vertexFunction = [quad_library newFunctionWithName:@"disc_vs"];
  quad_desc.fragmentFunction = [quad_library newFunctionWithName:@"disc_fs"];
  quad_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  quad_desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  quad_desc.rasterSampleCount = 2;
  id<MTLRenderPipelineState> quad_pipeline =
      [device newRenderPipelineStateWithDescriptor:quad_desc error:&error];
  REQUIRE(quad_pipeline != nil);

  MTLDepthStencilDescriptor* write_desc = [MTLDepthStencilDescriptor new];
  write_desc.depthCompareFunction = MTLCompareFunctionAlways;
  write_desc.depthWriteEnabled = YES;
  id<MTLDepthStencilState> write_state =
      [device newDepthStencilStateWithDescriptor:write_desc];
  MTLDepthStencilDescriptor* test_desc = [MTLDepthStencilDescriptor new];
  test_desc.depthCompareFunction = MTLCompareFunctionLess;
  test_desc.depthWriteEnabled = NO;
  id<MTLDepthStencilState> test_state =
      [device newDepthStencilStateWithDescriptor:test_desc];

  struct { uint32_t base, pitch, msaa, is_depth, fmt_log2, words; } sp{
      0u, 1u, 1u, 2u, 0u, kWords};
  struct { uint32_t start_tiles, end_tiles; } range{0u, 2u};
  id<MTLCommandQueue> queue = [device newCommandQueue];

  auto run_arm = [&](uint32_t hl_mode) -> double {
    id<MTLTexture> depth = [device newTextureWithDescriptor:depth_desc];
    id<MTLTexture> color = [device newTextureWithDescriptor:color_desc];
    id<MTLTexture> resolved = [device newTextureWithDescriptor:resolve_desc];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    // Pass 1: PRODUCTION materialize (reconcile per hl_mode).
    MTLRenderPassDescriptor* mat_pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    mat_pass.depthAttachment.texture = depth;
    mat_pass.depthAttachment.loadAction = MTLLoadActionClear;
    mat_pass.depthAttachment.storeAction = MTLStoreActionStore;
    mat_pass.depthAttachment.clearDepth = 0.0;
    id<MTLRenderCommandEncoder> mat_encoder =
        [command renderCommandEncoderWithDescriptor:mat_pass];
    [mat_encoder setRenderPipelineState:mat_pipeline];
    [mat_encoder setDepthStencilState:write_state];
    [mat_encoder setFragmentBuffer:canonical offset:0 atIndex:0];
    [mat_encoder setFragmentBytes:&sp length:sizeof(sp) atIndex:1];
    [mat_encoder setFragmentBytes:&range length:sizeof(range) atIndex:2];
    [mat_encoder setFragmentBytes:&hl_mode length:sizeof(hl_mode) atIndex:3];
    [mat_encoder setFragmentBuffer:lane offset:0 atIndex:4];
    [mat_encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
    [mat_encoder endEncoding];
    // Pass 2: depth-tested quad at D (LESS, no write), red on pass.
    MTLRenderPassDescriptor* quad_pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    quad_pass.colorAttachments[0].texture = color;
    quad_pass.colorAttachments[0].resolveTexture = resolved;
    quad_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    quad_pass.colorAttachments[0].storeAction =
        MTLStoreActionMultisampleResolve;
    quad_pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    quad_pass.depthAttachment.texture = depth;
    quad_pass.depthAttachment.loadAction = MTLLoadActionLoad;
    quad_pass.depthAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> quad_encoder =
        [command renderCommandEncoderWithDescriptor:quad_pass];
    [quad_encoder setRenderPipelineState:quad_pipeline];
    [quad_encoder setDepthStencilState:test_state];
    [quad_encoder setVertexBytes:&depth_d length:sizeof(depth_d) atIndex:0];
    [quad_encoder drawPrimitives:MTLPrimitiveTypeTriangle
                     vertexStart:0
                     vertexCount:3];
    [quad_encoder endEncoding];
    // Read back the resolved color.
    id<MTLBuffer> readback =
        [device newBufferWithLength:80 * 16 * 4
                            options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:resolved
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(80, 16, 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:80 * 4
 destinationBytesPerImage:80 * 16 * 4];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    const uint8_t* pixels = static_cast<const uint8_t*>(readback.contents);
    uint32_t red = 0;
    for (uint32_t i = 0; i < 80 * 16; ++i) {
      if (pixels[i * 4] > 128) ++red;
    }
    return double(red) / (80.0 * 16.0);
  };

  double host_arm = run_arm(2u);   // reconcile: attachment holds A
  double guest_arm = run_arm(0u);  // pure: attachment holds decoded
  INFO("host-arm red fraction " << host_arm << ", guest-arm " << guest_arm);
  // D < A -> LESS passes everywhere under preserved host precision.
  CHECK(host_arm > 0.99);
  // D > decoded -> LESS fails everywhere under the pure guest path.
  CHECK(guest_arm < 0.01);
}

TEST_CASE("Stencil-export depth materialize equals depth + bit-plane passes",
          "[l1][metal][canonical][depth][stencil_export]") {
  // The production 2x/4x ranged materialize has two encodings: the depth pass
  // (stencil REPLACEd to 0) followed by eight masked bit-plane passes, and a
  // single pass exporting [[stencil]] under the same REPLACE state. Both must
  // leave every sample's depth bits and stencil byte identical, inside and
  // outside the tile range, for both depth formats.
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { SUCCEED("no device"); return; }
  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:
                                         xe::gpu::metal::kEdramMsl]
                   options:nil
                     error:&error];
  REQUIRE(library != nil);
  static const char* kReadSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void read_ds_ms(depth2d_ms<float, access::read> d [[texture(0)]],
                       texture2d_ms<uint, access::read> s [[texture(1)]],
                       device uint* out [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= d.get_width() || gid.y >= d.get_height()) return;
  uint n = d.get_num_samples();
  for (uint i = 0; i < n; ++i) {
    uint idx = ((gid.y * d.get_width() + gid.x) * n + i) * 2u;
    out[idx] = as_type<uint>(d.read(gid, i));
    out[idx + 1u] = s.read(gid, i).r;
  }
}
)MSL";
  id<MTLLibrary> read_library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:kReadSource]
                   options:nil
                     error:&error];
  REQUIRE(read_library != nil);
  id<MTLComputePipelineState> read_pipeline = [device
      newComputePipelineStateWithFunction:[read_library
                                              newFunctionWithName:@"read_ds_ms"]
                                    error:&error];
  REQUIRE(read_pipeline != nil);

  MTLDepthStencilDescriptor* dsd = [MTLDepthStencilDescriptor new];
  dsd.depthCompareFunction = MTLCompareFunctionAlways;
  dsd.depthWriteEnabled = YES;
  MTLStencilDescriptor* zs = [MTLStencilDescriptor new];
  zs.stencilCompareFunction = MTLCompareFunctionAlways;
  zs.depthStencilPassOperation = MTLStencilOperationReplace;
  zs.writeMask = 0xFF;
  dsd.frontFaceStencil = zs;
  dsd.backFaceStencil = zs;
  id<MTLDepthStencilState> depth_state =
      [device newDepthStencilStateWithDescriptor:dsd];
  id<MTLDepthStencilState> bit_states[8];
  for (uint32_t b = 0; b < 8; ++b) {
    MTLDepthStencilDescriptor* bds = [MTLDepthStencilDescriptor new];
    bds.depthCompareFunction = MTLCompareFunctionAlways;
    bds.depthWriteEnabled = NO;
    MTLStencilDescriptor* bs = [MTLStencilDescriptor new];
    bs.stencilCompareFunction = MTLCompareFunctionAlways;
    bs.depthStencilPassOperation = MTLStencilOperationReplace;
    bs.writeMask = 1u << b;
    bds.frontFaceStencil = bs;
    bds.backFaceStencil = bs;
    bit_states[b] = [device newDepthStencilStateWithDescriptor:bds];
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];

  for (uint32_t samples : {2u, 4u}) {
    for (uint32_t is_depth : {1u, 2u}) {
      MTLRenderPipelineDescriptor* rpd = [MTLRenderPipelineDescriptor new];
      rpd.vertexFunction = [library newFunctionWithName:@"xe_depth_vs"];
      rpd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      rpd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
      rpd.rasterSampleCount = samples;
      rpd.fragmentFunction =
          [library newFunctionWithName:@"materialize_depth_ms_range_fs"];
      id<MTLRenderPipelineState> depth_pipeline =
          [device newRenderPipelineStateWithDescriptor:rpd error:&error];
      rpd.fragmentFunction =
          [library newFunctionWithName:@"materialize_sten_ms_range_fs"];
      id<MTLRenderPipelineState> sten_pipeline =
          [device newRenderPipelineStateWithDescriptor:rpd error:&error];
      rpd.fragmentFunction =
          [library newFunctionWithName:@"materialize_depth_sten_ms_range_fs"];
      id<MTLRenderPipelineState> export_pipeline =
          [device newRenderPipelineStateWithDescriptor:rpd error:&error];
      REQUIRE(depth_pipeline != nil);
      REQUIRE(sten_pipeline != nil);
      REQUIRE(export_pipeline != nil);

      // Two tiles wide, three tall (pitch 2); the range covers tiles [1, 4)
      // so both in-range and discarded samples are checked.
      const uint32_t msaa_log2 = samples == 4u ? 2u : 1u;
      const uint32_t width = samples == 4u ? 80u : 160u;
      const uint32_t height = samples == 4u ? 24u : 24u;
      const uint32_t words = 6u * 1280u;
      id<MTLBuffer> canonical =
          [device newBufferWithLength:words * 4
                              options:MTLResourceStorageModeShared];
      uint32_t* canonical_cpu = static_cast<uint32_t*>(canonical.contents);
      uint32_t lcg = 0x1234567u + samples * 31u + is_depth;
      for (uint32_t i = 0; i < words; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        uint32_t d24 = lcg >> 8;
        if (is_depth == 1u) d24 &= 0xFFFFFFu;
        // float24 stays in the representable [0,2) code space.
        if (is_depth == 2u) d24 &= 0x7FFFFFu;
        lcg = lcg * 1664525u + 1013904223u;
        canonical_cpu[i] = (d24 << 8) | (lcg >> 24);
      }
      id<MTLBuffer> lane =
          [device newBufferWithLength:words * 4
                              options:MTLResourceStorageModeShared];
      struct { uint32_t base, pitch, msaa, is_depth, fmt_log2, words; } sp{
          0u, 2u, msaa_log2, is_depth, 0u, words};
      struct { uint32_t start_tiles, end_tiles; } range{1u, 4u};
      uint32_t hl_mode = 0u;

      MTLTextureDescriptor* td = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                       width:width
                                      height:height
                                   mipmapped:NO];
      td.textureType = MTLTextureType2DMultisample;
      td.sampleCount = samples;
      td.storageMode = MTLStorageModePrivate;
      td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                 MTLTextureUsagePixelFormatView;

      auto run_arm = [&](bool stencil_export) -> std::vector<uint32_t> {
        id<MTLTexture> texture = [device newTextureWithDescriptor:td];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        MTLRenderPassDescriptor* pass =
            [MTLRenderPassDescriptor renderPassDescriptor];
        pass.depthAttachment.texture = texture;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = 0.625;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.stencilAttachment.texture = texture;
        pass.stencilAttachment.loadAction = MTLLoadActionClear;
        pass.stencilAttachment.clearStencil = 0xA5;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> encoder =
            [command renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:stencil_export ? export_pipeline
                                                       : depth_pipeline];
        [encoder setDepthStencilState:depth_state];
        [encoder setStencilReferenceValue:0];
        [encoder setFragmentBuffer:canonical offset:0 atIndex:0];
        [encoder setFragmentBytes:&sp length:sizeof(sp) atIndex:1];
        [encoder setFragmentBytes:&range length:sizeof(range) atIndex:2];
        [encoder setFragmentBytes:&hl_mode length:sizeof(hl_mode) atIndex:3];
        [encoder setFragmentBuffer:lane offset:0 atIndex:4];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
        if (!stencil_export) {
          [encoder setRenderPipelineState:sten_pipeline];
          [encoder setStencilReferenceValue:0xFF];
          for (uint32_t b = 0; b < 8; ++b) {
            uint32_t bit = 1u << b;
            [encoder setDepthStencilState:bit_states[b]];
            [encoder setFragmentBytes:&bit length:sizeof(bit) atIndex:3];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:3];
          }
        }
        [encoder endEncoding];
        id<MTLTexture> stencil_view =
            [texture newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
        const size_t count = size_t(width) * height * samples * 2u;
        id<MTLBuffer> out =
            [device newBufferWithLength:count * 4
                                options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
        [compute setComputePipelineState:read_pipeline];
        [compute setTexture:texture atIndex:0];
        [compute setTexture:stencil_view atIndex:1];
        [compute setBuffer:out offset:0 atIndex:0];
        [compute dispatchThreads:MTLSizeMake(width, height, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [compute endEncoding];
        [command commit];
        [command waitUntilCompleted];
        REQUIRE(command.status == MTLCommandBufferStatusCompleted);
        const uint32_t* data = static_cast<const uint32_t*>(out.contents);
        return std::vector<uint32_t>(data, data + count);
      };

      std::vector<uint32_t> legacy = run_arm(false);
      std::vector<uint32_t> exported = run_arm(true);
      size_t mismatches = 0, in_range = 0, untouched = 0;
      float clear_depth = 0.625f;
      uint32_t clear_bits;
      std::memcpy(&clear_bits, &clear_depth, 4);
      for (size_t i = 0; i < legacy.size(); i += 2) {
        if (legacy[i] != exported[i] || legacy[i + 1] != exported[i + 1]) {
          ++mismatches;
        }
        if (legacy[i] == clear_bits && legacy[i + 1] == 0xA5u) {
          ++untouched;
        } else {
          ++in_range;
        }
      }
      INFO("samples " << samples << " is_depth " << is_depth << " in-range "
                      << in_range << " untouched " << untouched);
      CHECK(mismatches == 0);
      // The range must exercise both written and discarded samples.
      CHECK(in_range > 0);
      CHECK(untouched > 0);
    }
  }
}

TEST_CASE("Canonical depth resolve exposes guest rather than attachment domain",
          "[l1][metal][canonical][resolve][depth]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }

  std::vector<uint32_t> words;
  for (uint32_t d24 = 0u; d24 < (1u << 24); d24 += 251u) {
    words.push_back((d24 << 8u) | 0x5Au);
  }
  words.push_back(0x000000A5u);
  words.push_back(0x800000A5u);
  words.push_back(0xFFFFFFA5u);

  std::vector<uint32_t> guest_float24(words.size());
  std::vector<uint32_t> host_float24(words.size());
  std::vector<uint32_t> guest_unorm24(words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    uint32_t d24 = words[i] >> 8u;
    float guest_float = xenos::Float20e4To32(d24);
    float host_float = guest_float * 0.5f;
    float guest_unorm = xenos::UNorm24To32(d24);
    std::memcpy(&guest_float24[i], &guest_float, sizeof(uint32_t));
    std::memcpy(&host_float24[i], &host_float, sizeof(uint32_t));
    std::memcpy(&guest_unorm24[i], &guest_unorm, sizeof(uint32_t));
  }

  CHECK(mc.Run("k_depth_word_to_guest", words, true, 2u) == guest_float24);
  CHECK(mc.Run("k_depth_word_to_host", words, true, 2u) == host_float24);
  CHECK(mc.Run("k_depth_word_to_guest", words, true, 1u) == guest_unorm24);
  CHECK(mc.Run("k_depth_word_to_host", words, true, 1u) == guest_unorm24);
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
// Stage B (implement-metal-gpu-visible-resolve-authority, task 2.3): the GPU
// scatter addresses guest memory with the MSL port of texture_address::Tiled2D.
// The CPU writeback scatter is the shipping oracle, so the port must agree
// EXACTLY -- a single wrong bit in the bank/pipe interleave scrambles every
// tiled fetch of the resolve destination, and the failure looks like corrupted
// textures rather than a wrong address.
TEST_CASE("Metal tiled2D offset matches the CPU scatter oracle",
          "[l1][metal][scatter]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  std::vector<Tiled2DIn> in;
  // Every bpp the resolve destination path can produce (1,2,4,8,16 bytes ->
  // log2 0..4), several 32-aligned pitches, and enough of the x/y space to
  // cross macro tiles and exercise every bank/pipe/y_lsb combination.
  // 32-block aligned, as the CPU addresser requires (it asserts on this).
  const uint32_t pitches[] = {32u, 64u, 320u, 1280u, 2048u};
  for (uint32_t bpp_log2 = 0; bpp_log2 <= 4u; ++bpp_log2) {
    for (uint32_t pitch : pitches) {
      for (uint32_t y = 0; y < 80u; ++y) {
        for (uint32_t x = 0; x < 80u; ++x) {
          in.push_back({x, y, pitch, bpp_log2});
        }
      }
    }
  }
  std::vector<uint32_t> cpu(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const Tiled2DIn& t = in[i];
    cpu[i] = uint32_t(texture_address::Tiled2D(int32_t(t.x), int32_t(t.y),
                                               t.pitch_aligned, t.bpp_log2));
  }
  auto gpu = mc.RunRaw("k_tiled2d_offset", in.data(),
                       in.size() * sizeof(Tiled2DIn), in.size());
  REQUIRE(gpu.size() == cpu.size());
  uint32_t bad = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    if (gpu[i] != cpu[i]) {
      if (bad < 8) {
        const Tiled2DIn& t = in[i];
        WARN("x=" << t.x << " y=" << t.y << " pitch=" << t.pitch_aligned
                  << " bpp_log2=" << t.bpp_log2 << " gpu=" << gpu[i]
                  << " cpu=" << cpu[i]);
      }
      ++bad;
    }
  }
  REQUIRE(bad == 0u);
}

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

TEST_CASE("Metal environment flags preserve strict default and rollback",
          "[l0][metal][authority]") {
  CHECK_FALSE(ParseMetalEnvironmentFlag(nullptr, false));
  CHECK(ParseMetalEnvironmentFlag(nullptr, true));
  CHECK(ParseMetalEnvironmentFlag("1", false));
  CHECK(ParseMetalEnvironmentFlag("1", true));
  CHECK_FALSE(ParseMetalEnvironmentFlag("0", false));
  CHECK_FALSE(ParseMetalEnvironmentFlag("0", true));
  CHECK_FALSE(ParseMetalEnvironmentFlag("", true));
  CHECK_FALSE(ParseMetalEnvironmentFlag("true", true));
  CHECK_FALSE(ParseMetalEnvironmentFlag("10", true));
}

TEST_CASE("Metal canonical range work height excludes untouched lower rows",
          "[l0][metal][authority]") {
  CHECK(ComputeCanonicalRangeTextureHeight(0u, 16u, 0u, 16u, 2048u) ==
        16u);
  CHECK(ComputeCanonicalRangeTextureHeight(0u, 16u, 1u, 17u, 1024u) ==
        16u);
  CHECK(ComputeCanonicalRangeTextureHeight(174u, 2u, 1u, 178u, 8192u) ==
        16u);
  CHECK(ComputeCanonicalRangeTextureHeight(174u, 2u, 1u, 9999u, 8192u) ==
        8192u);
  // Rows above the first tile row of a range: none inside the first row, one
  // 16-sample tile row at 1x, halved by the 2x MSAA sample shift, doubled by a
  // vertical draw resolution scale of 2, and clamped to the texture.
  CHECK(ComputeCanonicalRangeTextureRowStart(0u, 16u, 0u, 0u, 2048u) == 0u);
  CHECK(ComputeCanonicalRangeTextureRowStart(0u, 16u, 0u, 15u, 2048u) == 0u);
  CHECK(ComputeCanonicalRangeTextureRowStart(0u, 16u, 0u, 16u, 2048u) == 16u);
  CHECK(ComputeCanonicalRangeTextureRowStart(0u, 16u, 1u, 32u, 1024u) == 16u);
  CHECK(ComputeCanonicalRangeTextureRowStart(174u, 2u, 1u, 178u, 8192u) ==
        16u);
  CHECK(ComputeCanonicalRangeTextureRowStart(0u, 16u, 0u, 16u, 4096u, 2u) ==
        32u);
  CHECK(ComputeCanonicalRangeTextureRowStart(0u, 16u, 0u, 9999u, 64u) == 64u);
  CHECK(ComputeCanonicalRangeTextureHeight(174u, 0u, 1u, 178u, 8192u) == 0u);
  CHECK(ComputeCanonicalRangeTextureHeight(174u, 2u, 1u, 174u, 8192u) == 0u);
}

TEST_CASE("Metal MSAA companion height matches its resolve texture",
          "[l0][metal][authority]") {
  // Metal validates multisample and resolve attachments at identical sizes,
  // so any non-empty requested range allocates the full companion height.
  CHECK(ComputeMsaaCompanionAllocationHeight(0u, 0u, 8192u, 8u) == 0u);
  CHECK(ComputeMsaaCompanionAllocationHeight(0u, 1u, 8192u, 8u) == 8192u);
  CHECK(ComputeMsaaCompanionAllocationHeight(0u, 721u, 8192u, 8u) ==
        8192u);

  // Existing height and row-growth quantum cannot relax the invariant.
  CHECK(ComputeMsaaCompanionAllocationHeight(728u, 720u, 8192u, 8u) ==
        8192u);
  CHECK(ComputeMsaaCompanionAllocationHeight(728u, 729u, 8192u, 8u) ==
        8192u);
  CHECK(ComputeMsaaCompanionAllocationHeight(728u, 2000u, 8192u, 8u) ==
        8192u);

  CHECK(ComputeMsaaCompanionAllocationHeight(4096u, 4097u, 8192u, 8u) ==
        8192u);
  CHECK(ComputeMsaaCompanionAllocationHeight(8192u, 100u, 8192u, 8u) ==
        8192u);
  CHECK(ComputeMsaaCompanionAllocationHeight(0u, UINT32_MAX, 8192u, 8u) ==
        8192u);
  CHECK(ComputeMsaaCompanionAllocationHeight(0x80000000u, 0x80000001u,
                                             UINT32_MAX, 8u) == UINT32_MAX);
  CHECK(ComputeMsaaCompanionAllocationHeight(7u, 9u, 17u, 0u) == 17u);
  CHECK(ComputeMsaaCompanionAllocationHeight(7u, 9u, 0u, 8u) == 0u);
}

TEST_CASE("Metal EDRAM authority synchronizes only overlapping dirty tiles",
          "[l1][metal][authority]") {
  using Location = MetalEdramAuthorityTracker::Location;
  using SyncKind = MetalEdramAuthorityTracker::SyncKind;
  constexpr uint64_t kAttachmentA = 0xA1u;
  constexpr uint64_t kAttachmentB = 0xB2u;

  MetalEdramAuthorityTracker tracker;
  tracker.MarkCanonical(0u, 64u, 1u);
  tracker.MarkAttachment(kAttachmentA, 10u, 20u, 2u);

  // B consumes [15,25): only A's overlap is packed, then exactly the requested
  // interval is materialized. A's non-overlapping [10,15) stays authoritative.
  auto operations =
      tracker.SynchronizeAttachment(kAttachmentB, 15u, 25u, 3u);
  REQUIRE(operations.size() == 2u);
  CHECK(operations[0].kind == SyncKind::kPackToCanonical);
  CHECK(operations[0].attachment == kAttachmentA);
  CHECK(operations[0].start_tiles == 15u);
  CHECK(operations[0].end_tiles == 20u);
  CHECK(operations[0].producer_order == 2u);
  CHECK(operations[0].consumer_order == 3u);
  CHECK(operations[1].kind == SyncKind::kMaterializeFromCanonical);
  CHECK(operations[1].attachment == kAttachmentB);
  CHECK(operations[1].start_tiles == 15u);
  CHECK(operations[1].end_tiles == 25u);
  CHECK(tracker.tile(12u).location == Location::kAttachment);
  CHECK(tracker.tile(12u).authority_attachment == kAttachmentA);
  CHECK(tracker.tile(17u).location == Location::kCanonical);
  CHECK(tracker.tile(17u).clean_attachment == kAttachmentB);

  // Reusing B is compatible and emits no redundant synchronization.
  CHECK(tracker.SynchronizeAttachment(kAttachmentB, 15u, 25u, 4u).empty());
  // Replacing B's Metal texture invalidates its clean-view cache, so the same
  // canonical words must be materialized into the new allocation.
  tracker.InvalidateAttachment(kAttachmentB);
  CHECK(tracker.tile(17u).clean_attachment == 0u);
  operations = tracker.SynchronizeAttachment(kAttachmentB, 15u, 25u, 5u);
  REQUIRE(operations.size() == 1u);
  CHECK(operations[0].kind == SyncKind::kMaterializeFromCanonical);
  CHECK(operations[0].attachment == kAttachmentB);
  CHECK(operations[0].start_tiles == 15u);
  CHECK(operations[0].end_tiles == 25u);
  // A resolve clear is a new canonical producer. Even if A was retained as a
  // clean cache by the preceding pack, MarkCanonical must invalidate it so the
  // next A consumer rematerializes the cleared words rather than reusing the
  // pre-clear attachment contents.
  tracker.MarkCanonical(15u, 20u, 6u);
  operations = tracker.SynchronizeAttachment(kAttachmentA, 15u, 20u, 7u);
  REQUIRE(operations.size() == 1u);
  CHECK(operations[0].kind == SyncKind::kMaterializeFromCanonical);
  CHECK(operations[0].attachment == kAttachmentA);
  CHECK(operations[0].start_tiles == 15u);
  CHECK(operations[0].end_tiles == 20u);
  // A canonical consumer of a smaller overlap packs only that subrange.
  operations = tracker.SynchronizeCanonical(10u, 12u, 8u);
  REQUIRE(operations.size() == 1u);
  CHECK(operations[0].kind == SyncKind::kPackToCanonical);
  CHECK(operations[0].start_tiles == 10u);
  CHECK(operations[0].end_tiles == 12u);
  CHECK(tracker.tile(12u).location == Location::kAttachment);

  // If a caller retires a dirty view without synchronizing first, never leave
  // authority pointing at deallocated content. The defensive branch degrades
  // those tiles to unknown rather than allowing a stale pack later.
  tracker.MarkAttachment(kAttachmentA, 40u, 42u, 9u);
  tracker.InvalidateAttachment(kAttachmentA);
  CHECK(tracker.tile(40u).location == Location::kUnknown);
  CHECK(tracker.tile(40u).authority_attachment == 0u);
  CHECK(tracker.tile(40u).clean_attachment == 0u);
}

TEST_CASE("Metal EDRAM authority preserves submission order and batches sources",
          "[l1][metal][authority]") {
  using Location = MetalEdramAuthorityTracker::Location;
  using SyncKind = MetalEdramAuthorityTracker::SyncKind;
  constexpr uint64_t kAttachmentA = 0xA1u;
  constexpr uint64_t kAttachmentB = 0xB2u;
  constexpr uint64_t kAttachmentC = 0xC3u;

  MetalEdramAuthorityTracker tracker;
  tracker.MarkCanonical(0u, 64u, 1u);
  tracker.MarkAttachment(kAttachmentA, 30u, 34u, 2u);
  tracker.MarkAttachment(kAttachmentB, 34u, 38u, 3u);
  auto operations =
      tracker.SynchronizeAttachment(kAttachmentC, 32u, 36u, 4u);
  REQUIRE(operations.size() == 3u);
  CHECK(operations[0].kind == SyncKind::kPackToCanonical);
  CHECK(operations[0].attachment == kAttachmentA);
  CHECK(operations[0].start_tiles == 32u);
  CHECK(operations[0].end_tiles == 34u);
  CHECK(operations[1].kind == SyncKind::kPackToCanonical);
  CHECK(operations[1].attachment == kAttachmentB);
  CHECK(operations[1].start_tiles == 34u);
  CHECK(operations[1].end_tiles == 36u);
  CHECK(operations[2].kind == SyncKind::kMaterializeFromCanonical);
  CHECK(operations[2].attachment == kAttachmentC);
  CHECK(operations[2].start_tiles == 32u);
  CHECK(operations[2].end_tiles == 36u);

  // A delayed producer event and an older consumer cannot overwrite or pack a
  // tile whose newer canonical write has already been observed.
  tracker.MarkCanonical(33u, 35u, 8u);
  tracker.MarkAttachment(kAttachmentA, 33u, 35u, 7u);
  CHECK(tracker.tile(33u).location == Location::kCanonical);
  CHECK(tracker.tile(33u).submission_order == 8u);
  CHECK(tracker.SynchronizeAttachment(kAttachmentB, 33u, 35u, 7u).empty());
  operations = tracker.SynchronizeAttachment(kAttachmentB, 33u, 35u, 9u);
  REQUIRE(operations.size() == 1u);
  CHECK(operations[0].kind == SyncKind::kMaterializeFromCanonical);
  CHECK(operations[0].start_tiles == 33u);
  CHECK(operations[0].end_tiles == 35u);

  // Range clamping never emits an operation outside physical EDRAM.
  tracker.MarkAttachment(kAttachmentA, xenos::kEdramTileCount - 2u,
                         xenos::kEdramTileCount + 10u, 10u);
  operations = tracker.SynchronizeCanonical(
      xenos::kEdramTileCount - 1u, xenos::kEdramTileCount + 20u, 11u);
  REQUIRE(operations.size() == 1u);
  CHECK(operations[0].start_tiles == xenos::kEdramTileCount - 1u);
  CHECK(operations[0].end_tiles == xenos::kEdramTileCount);
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

TEST_CASE("Canonical raw depth resolve preserves sample, stencil, and wrap",
          "[l1][metal][canonical][resolve]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  constexpr uint32_t w = 16u;
  constexpr uint32_t h = 16u;
  constexpr uint32_t sample = 2u;
  const uint32_t pitch = PitchFor(w, 2u, 0u);
  SurfaceParams sp{xenos::kEdramTileCount - 1u, pitch, 2u, 1u, 0u,
                   uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))};
  std::vector<uint32_t> expected(size_t(w) * h);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      // Distinct depth and stencil bits make any float round-trip or sample
      // replication fail byte-exactly.
      uint32_t i = y * w + x;
      expected[i] = (((i * 0x10203u) ^ 0x5A1234u) & 0xFFFFFFu) << 8u |
                    ((i * 37u + 0x5Bu) & 0xFFu);
    }
  }
  auto actual = mc.ResolveRawDepth(w, h, sp, sample, expected,
                                   xenos::kEdramSizeBytes);
  REQUIRE(actual == expected);
}

TEST_CASE("Canonical color resolve selects, averages, biases, and swaps",
          "[l1][metal][canonical][resolve]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  const uint32_t pitch = PitchFor(1u, 2u, 0u);
  SurfaceParams sp{0u, pitch, 2u, 0u, 0u,
                   uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))};
  const std::array<uint32_t, 4> words = {
      0x281E140Au, 0x50463C32u, 0x786E645Au, 0xA0968C82u};
  struct Selection { uint32_t value, first, count; } selections[] = {
      {0u, 0u, 1u}, {1u, 1u, 1u}, {2u, 2u, 1u}, {3u, 3u, 1u},
      {4u, 0u, 2u}, {5u, 2u, 2u}, {6u, 0u, 4u}};
  for (const Selection& selection : selections) {
    auto bits = mc.ResolveColorOne(sp, selection.value, /*dest_swap=*/true,
                                   /*dest_red_only=*/false,
                                   /*color_format=*/0u,
                                   /*exp_bias_scale=*/2.0f, words,
                                   xenos::kEdramSizeBytes);
    float actual[4];
    std::memcpy(actual, bits.data(), sizeof(actual));
    float expected[4] = {};
    for (uint32_t i = 0; i < selection.count; ++i) {
      uint32_t word = words[selection.first + i];
      expected[0] += float(word & 0xFFu);
      expected[1] += float((word >> 8u) & 0xFFu);
      expected[2] += float((word >> 16u) & 0xFFu);
      expected[3] += float((word >> 24u) & 0xFFu);
    }
    for (float& lane : expected) {
      lane *= 2.0f / (255.0f * float(selection.count));
    }
    std::swap(expected[0], expected[2]);
    INFO("sample select " << selection.value);
    for (uint32_t lane = 0; lane < 4u; ++lane) {
      CHECK(std::abs(actual[lane] - expected[lane]) < 1.0e-6f);
    }
  }
  // A red-only R32 destination must not turn an R/RG source into zero when
  // copy_dest_swap is set, while an RGBA source selects its stored blue lane.
  std::array<uint32_t, 4> r32_words;
  float r32_value = 3.5f;
  uint32_t r32_bits;
  std::memcpy(&r32_bits, &r32_value, sizeof(r32_bits));
  r32_words.fill(r32_bits);
  auto r32_result = mc.ResolveColorOne(
      SurfaceParams{0u, pitch, 2u, 0u, 0u,
                    uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))},
      0u, /*dest_swap=*/true, /*dest_red_only=*/true,
      /*color_format=*/14u, 1.0f, r32_words, xenos::kEdramSizeBytes);
  float r32_actual;
  std::memcpy(&r32_actual, r32_result.data(), sizeof(r32_actual));
  CHECK(r32_actual == r32_value);
  auto rgba_red_result = mc.ResolveColorOne(
      sp, 0u, /*dest_swap=*/true, /*dest_red_only=*/true,
      /*color_format=*/0u, 1.0f, words, xenos::kEdramSizeBytes);
  float rgba_red_actual;
  std::memcpy(&rgba_red_actual, rgba_red_result.data(),
              sizeof(rgba_red_actual));
  CHECK(std::abs(rgba_red_actual - (30.0f / 255.0f)) < 1.0e-6f);
  float rg_r = 1.25f;
  float rg_g = 4.5f;
  uint32_t rg_r_bits, rg_g_bits;
  std::memcpy(&rg_r_bits, &rg_r, sizeof(rg_r_bits));
  std::memcpy(&rg_g_bits, &rg_g, sizeof(rg_g_bits));
  std::array<uint32_t, 4> rg_r_words, rg_g_words;
  rg_r_words.fill(rg_r_bits);
  rg_g_words.fill(rg_g_bits);
  auto rg_result = mc.ResolveColorOne(
      SurfaceParams{0u, PitchFor(1u, 2u, 1u), 2u, 0u, 1u,
                    uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))},
      0u, /*dest_swap=*/true, /*dest_red_only=*/true,
      /*color_format=*/15u, 1.0f, rg_r_words, xenos::kEdramSizeBytes,
      &rg_g_words);
  float rg_actual;
  std::memcpy(&rg_actual, rg_result.data(), sizeof(rg_actual));
  CHECK(rg_actual == rg_g);
}

TEST_CASE("Canonical packed color resolve matches guest formats and Endian128",
          "[l1][metal][canonical][resolve][packed]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  const uint32_t word_count =
      uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t));
  auto read16 = [](const std::array<uint8_t, 16>& bytes) {
    uint16_t value;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
  };
  auto read32 = [](const std::array<uint8_t, 16>& bytes, uint32_t word = 0u) {
    uint32_t value;
    std::memcpy(&value, bytes.data() + word * sizeof(uint32_t), sizeof(value));
    return value;
  };
  auto swap32 = [](uint32_t value, uint32_t endian) {
    if (endian == 1u || endian == 2u) {
      value = ((value & 0x00FF00FFu) << 8u) |
              ((value & 0xFF00FF00u) >> 8u);
    }
    if (endian == 2u || endian == 3u) {
      value = (value << 16u) | (value >> 16u);
    }
    return value;
  };
  auto swap64 = [&](std::array<uint32_t, 2> value, uint32_t endian) {
    if (endian == 4u) {
      std::swap(value[0], value[1]);
      endian = 2u;
    }
    value[0] = swap32(value[0], endian);
    value[1] = swap32(value[1], endian);
    return value;
  };
  auto swap128 = [&](std::array<uint32_t, 4> value, uint32_t endian) {
    if (endian == 5u) {
      std::reverse(value.begin(), value.end());
      endian = 2u;
    }
    auto lo = swap64({value[0], value[1]}, endian);
    auto hi = swap64({value[2], value[3]}, endian);
    return std::array<uint32_t, 4>{lo[0], lo[1], hi[0], hi[1]};
  };

  SurfaceParams rgba_sp{0u, PitchFor(1u, 2u, 0u), 2u, 0u, 0u,
                        word_count};
  std::array<uint32_t, 4> samples{0xFF000010u, 0xFF000020u,
                                  0xFF000040u, 0xFF000080u};
  auto r8 = mc.ResolveColorPackedOne(rgba_sp, 4u, false, 0u, 2.0f,
                                     2u, 0u, 5u, false, samples);
  CHECK(r8[0] == 48u);

  samples.fill(0xFF0080FFu);
  auto rgb565 = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 1.0f,
                                         4u, 1u, 1u, false, samples);
  CHECK(read16(rgb565) == 0x1F04u);

  samples.fill(0x44332211u);
  constexpr uint32_t swapped_8888 = 0x44112233u;
  for (uint32_t endian = 0; endian <= 3u; ++endian) {
    auto packed = mc.ResolveColorPackedOne(rgba_sp, 0u, true, 0u, 1.0f,
                                           6u, 2u, endian, false, samples);
    INFO("32bpp endian " << endian);
    CHECK(read32(packed) == swap32(swapped_8888, endian));

    // Bitwise-equivalent resolves take the raw-copy fast path live. The raw
    // word must still honor copy_dest_swap before endian packing; otherwise a
    // later Z,Y,X,W texture fetch applies the only R/B exchange and composites
    // the entire image with red and blue reversed.
    auto raw_packed = mc.ResolveColorPackedOne(
        rgba_sp, 0u, true, 0u, 1.0f, 6u, 2u, endian, true, samples);
    INFO("32bpp raw-copy endian " << endian);
    CHECK(read32(raw_packed) == swap32(swapped_8888, endian));
  }

  SurfaceParams half_sp{0u, PitchFor(1u, 2u, 1u), 2u, 0u, 1u,
                        word_count};
  std::array<uint32_t, 4> half_rg, half_ba;
  half_rg.fill(0x40003C00u);
  half_ba.fill(0x44004200u);
  const std::array<uint32_t, 2> half_expected{0x40003C00u, 0x44004200u};
  for (uint32_t endian = 0; endian <= 5u; ++endian) {
    auto packed = mc.ResolveColorPackedOne(
        half_sp, 0u, false, 7u, 1.0f, 32u, 3u, endian, false, half_rg,
        &half_ba);
    auto expected = swap64(half_expected, endian);
    INFO("64bpp endian " << endian);
    CHECK(read32(packed, 0u) == expected[0]);
    CHECK(read32(packed, 1u) == expected[1]);
  }

  samples.fill(0xFFFFFFFFu);
  const std::array<uint32_t, 4> float_one{0x3F800000u, 0x3F800000u,
                                         0x3F800000u, 0x3F800000u};
  for (uint32_t endian = 0; endian <= 5u; ++endian) {
    auto packed = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 1.0f,
                                           38u, 4u, endian, false, samples);
    auto expected = swap128(float_one, endian);
    INFO("128bpp endian " << endian);
    for (uint32_t word = 0; word < 4u; ++word) {
      CHECK(read32(packed, word) == expected[word]);
    }
  }

  SurfaceParams r32_sp{0u, PitchFor(1u, 2u, 0u), 2u, 0u, 0u, word_count};
  std::array<uint32_t, 4> nan_samples{0x7FC00001u, 0x7FC00002u,
                                      0x7FC12345u, 0x7FC00004u};
  auto raw = mc.ResolveColorPackedOne(r32_sp, 2u, false, 14u, 1.0f,
                                      36u, 2u, 0u, true, nan_samples);
  CHECK(read32(raw) == 0x7FC12345u);

  // Red-only swap is specific to the 8bpp resolve shader. Wider conversion
  // paths swap the full float4 before packing, so R/RG -> R32 selects the
  // absent blue lane (zero). k_8 + swap is A8 (Canary 2b3f0cb45): it selects
  // the source alpha, which a 32_32_FLOAT source lacks, so it reads 1.0.
  float quarter = 0.25f;
  float half = 0.5f;
  uint32_t quarter_bits, half_bits;
  std::memcpy(&quarter_bits, &quarter, sizeof(quarter_bits));
  std::memcpy(&half_bits, &half, sizeof(half_bits));
  std::array<uint32_t, 4> r_words, g_words;
  r_words.fill(quarter_bits);
  g_words.fill(half_bits);
  auto converted_r32 = mc.ResolveColorPackedOne(
      r32_sp, 0u, true, 14u, 1.0f, 36u, 2u, 0u, false, r_words);
  CHECK(read32(converted_r32) == 0u);
  SurfaceParams rg_sp{0u, PitchFor(1u, 2u, 1u), 2u, 0u, 1u, word_count};
  auto converted_r8 = mc.ResolveColorPackedOne(
      rg_sp, 0u, true, 15u, 1.0f, 2u, 0u, 0u, false, r_words, &g_words);
  CHECK(converted_r8[0] == 255u);
  // 8_8_8_8 -> k_8 + swap takes alpha (0x80), not blue (0x40); unswapped it
  // takes red (0x10).
  std::array<uint32_t, 4> rgba8_words;
  rgba8_words.fill(0x80402010u);
  auto a8 = mc.ResolveColorPackedOne(r32_sp, 0u, true, 0u, 1.0f, 2u, 0u, 0u,
                                     false, rgba8_words);
  CHECK(a8[0] == 0x80u);
  auto l8 = mc.ResolveColorPackedOne(r32_sp, 0u, false, 0u, 1.0f, 2u, 0u, 0u,
                                     false, rgba8_words);
  CHECK(l8[0] == 0x10u);
}


TEST_CASE("Canonical packed color resolve honors copy_dest_number integer",
          "[l1][metal][canonical][resolve][packed][number]") {
  // Crysis 3 resolves its 7e3 HDR scene with exp_bias +6 into an unsigned
  // INTEGER 11:11:10 destination (RB_COPY_DEST_INFO copy_dest_number 2) and
  // fetches it back as integer with exp_adjust -6: the stored word is
  // round(value * 64), not saturate(value * 64) * 2047. The unsigned-fraction
  // pack saturated the whole scene copy to 0xFFFFFFFF (the washout).
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  const uint32_t word_count =
      uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t));
  auto read32 = [](const std::array<uint8_t, 16>& bytes, uint32_t word = 0u) {
    uint32_t value;
    std::memcpy(&value, bytes.data() + word * sizeof(uint32_t), sizeof(value));
    return value;
  };
  SurfaceParams rgba_sp{0u, PitchFor(1u, 2u, 0u), 2u, 0u, 0u, word_count};
  std::array<uint32_t, 4> samples;
  // 8888 source A=0xFF B=0x33 G=0x22 R=0x11 (word is ABGR little-endian).
  samples.fill(0xFF332211u);
  // Unsigned fraction (dest_number 0): every channel * 64 saturates.
  auto fraction = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 64.0f,
                                           16u, 2u, 0u, false, samples,
                                           nullptr, 0u);
  CHECK(read32(fraction) == 0xFFFFFFFFu);
  // Unsigned integer (dest_number 2): round(c / 255 * 64) per channel.
  // R 0x11 -> 4.27 -> 4, G 0x22 -> 8.53 -> 9, B 0x33 -> 12.8 -> 13.
  auto integer = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 64.0f,
                                          16u, 2u, 0u, false, samples,
                                          nullptr, 2u);
  CHECK(read32(integer) == (4u | (9u << 11u) | (13u << 22u)));
  // Integer saturates at the component maximum, not at 1.0: * 65536 into
  // 11:11:10 clamps R to 2047, G to 2047, B to 1023.
  auto clamped = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 65536.0f,
                                          16u, 2u, 0u, false, samples,
                                          nullptr, 2u);
  CHECK(read32(clamped) == 0xFFFFFFFFu);
  // 16_16_16_16 integer with bias 2048 (Crysis 3's luminance chain):
  // R 0x11 -> 136.5 -> 137, G 0x22 -> 273.1 -> 273, B 0x33 -> 409.6 -> 410,
  // A 0xFF -> 2048.
  auto wide = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 2048.0f,
                                       26u, 3u, 0u, false, samples,
                                       nullptr, 2u);
  CHECK(read32(wide, 0u) == (137u | (273u << 16u)));
  CHECK(read32(wide, 1u) == (410u | (2048u << 16u)));
  // Signed fraction (dest_number 1) packs with the positive endpoint 127.
  auto snorm = mc.ResolveColorPackedOne(rgba_sp, 0u, false, 0u, 1.0f,
                                        6u, 2u, 0u, false, samples,
                                        nullptr, 1u);
  // 0x11/255*127 = 8.47 -> 8, 0x22 -> 16.9 -> 17, 0x33 -> 25.4 -> 25,
  // 0xFF -> 127.
  CHECK(read32(snorm) == (8u | (17u << 8u) | (25u << 16u) | (127u << 24u)));
}
TEST_CASE("Canonical resolve clear writes only the declared surface samples",
          "[l1][metal][canonical][resolve][clear]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [mc.dev
      newComputePipelineStateWithFunction:
          [mc.lib newFunctionWithName:@"resolve_canonical_clear"]
                                    error:&error];
  REQUIRE(pipeline != nil);
  constexpr uint32_t sentinel = 0xDEADBEEFu;
  const uint32_t word_count =
      uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t));
  struct Config {
    SurfaceParams surface;
    uint32_t source_x, source_y, width, height, clear_lo, clear_hi;
    const char* name;
  } configs[] = {
      {{20u, PitchFor(13u, 0u, 0u), 0u, 0u, 0u, word_count},
       3u, 5u, 13u, 7u, 0x10203040u, 0u, "1x color"},
      {{40u, PitchFor(15u, 1u, 0u), 1u, 1u, 0u, word_count},
       1u, 3u, 15u, 9u, 0xFEDCBA79u, 0u, "2x depth+stencil"},
      {{xenos::kEdramTileCount - 1u, PitchFor(16u, 2u, 0u), 2u, 0u,
        0u, word_count},
       0u, 8u, 16u, 8u, 0xA1B2C3D4u, 0u, "4x color wrap"},
      {{100u, PitchFor(16u, 2u, 0u), 2u, 1u, 0u, word_count},
       8u, 0u, 16u, 8u, 0x123456A5u, 0u, "4x depth+stencil"},
      {{200u, PitchFor(16u, 2u, 1u), 2u, 0u, 1u, word_count},
       0u, 0u, 16u, 8u, 0x01020304u, 0xA0B0C0D0u, "4x 64bpp color"},
  };
  for (const Config& config : configs) {
    id<MTLBuffer> canonical =
        [mc.dev newBufferWithLength:xenos::kEdramSizeBytes
                            options:MTLResourceStorageModeShared];
    uint32_t* actual = static_cast<uint32_t*>(canonical.contents);
    std::fill(actual, actual + word_count, sentinel);
    std::vector<uint32_t> expected(word_count, sentinel);
    const uint32_t sample_count = 1u << config.surface.msaa;
    for (uint32_t y = 0; y < config.height; ++y) {
      for (uint32_t x = 0; x < config.width; ++x) {
        for (uint32_t sample = 0; sample < sample_count; ++sample) {
          uint32_t index =
              (XeEdramSnapshotOffsetBytes(
                   config.source_x + x, config.source_y + y,
                   config.surface.base, config.surface.pitch,
                   config.surface.msaa, config.surface.is_depth != 0u,
                   config.surface.format_ints_log2, sample) >>
               2u) %
              word_count;
          expected[index] = config.clear_lo;
          if (config.surface.format_ints_log2 != 0u) {
            expected[(index + 1u) % word_count] = config.clear_hi;
          }
        }
      }
    }
    CanonicalClearParamsHost params{
        config.surface, config.source_x, config.source_y, config.width,
        config.height, config.clear_lo, config.clear_hi};
    id<MTLCommandBuffer> cb = [mc.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:canonical offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(config.width, config.height,
                                         sample_count)
        threadsPerThreadgroup:MTLSizeMake(8u, 8u, 1u)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);
    uint32_t bad = 0u;
    uint32_t first_bad = 0u;
    for (uint32_t i = 0; i < word_count; ++i) {
      if (actual[i] != expected[i]) {
        if (!bad) first_bad = i;
        ++bad;
      }
    }
    INFO(config.name << ": first bad word " << first_bad);
    CHECK(bad == 0u);
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

TEST_CASE("Canonical corpus color formats round-trip through lossless views",
          "[l1][metal][materialize][color]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct Cfg {
    const char* name;
    uint32_t format;
    MTLPixelFormat pixel_format;
  } cfgs[] = {
      {"8_8_8_8_GAMMA", 1u, MTLPixelFormatRGBA16Unorm},
      {"2_10_10_10_FLOAT", 3u, MTLPixelFormatRGBA16Float},
      {"16_16", 4u, MTLPixelFormatRG32Float},
      {"32_FLOAT", 14u, MTLPixelFormatR32Float},
  };
  constexpr uint32_t w = 83u, h = 19u, base = 11u;
  const uint32_t pitch = PitchFor(w, 0u, 0u);
  SurfaceParams sp{base, pitch, 0u, 0u, 0u,
                   uint32_t(xenos::kEdramSizeBytes / 4u)};
  for (const Cfg& cfg : cfgs) {
    std::vector<uint32_t> words(size_t(w) * h);
    for (size_t i = 0; i < words.size(); ++i) {
      if (cfg.format == 1u) {
        // Odd multipliers cover every byte value in every lane. The RGB
        // midpoint materialization must survive RGBA16Unorm quantization and
        // PWL re-encoding exactly; alpha remains ordinary linear UNORM.
        uint32_t byte = uint32_t(i) & 0xFFu;
        words[i] = byte | (((byte * 73u) & 0xFFu) << 8u) |
                   (((byte * 151u) & 0xFFu) << 16u) |
                   (((byte * 199u) & 0xFFu) << 24u);
      } else if (cfg.format == 3u) {
        uint32_t r = uint32_t(i * 17u) & 0x3FFu;
        uint32_t g = uint32_t(i * 37u + 1u) & 0x3FFu;
        uint32_t b = uint32_t(i * 73u + 127u) & 0x3FFu;
        words[i] = r | (g << 10u) | (b << 20u) |
                   ((uint32_t(i) & 3u) << 30u);
      } else if (cfg.format == 4u) {
        // Exercise all signed quadrants, including the otherwise unreachable
        // -32768 endpoint that the shared converter preserves explicitly.
        uint32_t lo = uint32_t(i * 40503u + 0x8000u) & 0xFFFFu;
        uint32_t hi = uint32_t(i * 25173u + 0x7FFFu) & 0xFFFFu;
        words[i] = lo | (hi << 16u);
      } else {
        float value = float(int32_t(i % 257u) - 128) * 0.25f;
        std::memcpy(&words[i], &value, sizeof(value));
      }
    }
    auto out = mc.RoundTripMaterializeColor(
        w, h, sp, words, xenos::kEdramSizeBytes, cfg.pixel_format, cfg.format);
    uint32_t bad = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      if (out[i] != words[i]) {
        if (bad < 8) {
          WARN(cfg.name << " pixel " << (i % w) << "," << (i / w)
                        << " expected " << words[i] << " got " << out[i]);
        }
        ++bad;
      }
    }
    INFO(cfg.name);
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
// touches the draw path. Every corpus storage format is exercised through its
// lossless working view, and distinct per-sample seeds catch any sample loss.
TEST_CASE("Real MSAA attachment per-sample render->pack round-trips",
          "[l1][metal][msaa][live]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct SurfaceCfg {
    uint32_t w, h, base, msaa, samples;
    const char* name;
  } surface_cfgs[] = {
      {48, 24, 0, 1u, 2u, "2x"},
      {40, 16, 5, 2u, 4u, "4x base=5"},
  };
  struct FormatCfg {
    const char* name;
    uint32_t format;
    MTLPixelFormat pixel_format;
  } format_cfgs[] = {
      {"8_8_8_8", 0u, MTLPixelFormatRGBA8Unorm},
      {"8_8_8_8_GAMMA", 1u, MTLPixelFormatRGBA16Unorm},
      {"2_10_10_10_FLOAT", 3u, MTLPixelFormatRGBA16Float},
      {"16_16", 4u, MTLPixelFormatRG32Float},
      {"32_FLOAT", 14u, MTLPixelFormatR32Float},
  };
  for (const SurfaceCfg& surface : surface_cfgs) {
    const uint32_t pitch = PitchFor(surface.w, surface.msaa, 0u);
    SurfaceParams sp{surface.base, pitch, surface.msaa, 0u, 0u,
                     uint32_t(xenos::kEdramSizeBytes / 4u)};
    for (const FormatCfg& format : format_cfgs) {
      std::vector<uint32_t> values(size_t(surface.w) * surface.h *
                                   surface.samples);
      for (size_t i = 0; i < values.size(); ++i) {
        if (format.format == 0u) {
          values[i] = uint32_t(i * 2246822519u) ^ 0x1234ABCDu;
        } else if (format.format == 1u) {
          uint32_t byte = uint32_t(i) & 0xFFu;
          values[i] = byte | (((byte * 73u) & 0xFFu) << 8u) |
                      (((byte * 151u) & 0xFFu) << 16u) |
                      (((byte * 199u) & 0xFFu) << 24u);
        } else if (format.format == 3u) {
          uint32_t r = uint32_t(i * 17u) & 0x3FFu;
          uint32_t g = uint32_t(i * 37u + 1u) & 0x3FFu;
          uint32_t b = uint32_t(i * 73u + 127u) & 0x3FFu;
          values[i] = r | (g << 10u) | (b << 20u) |
                      ((uint32_t(i) & 3u) << 30u);
        } else if (format.format == 4u) {
          uint32_t lo = uint32_t(i * 40503u + 0x8000u) & 0xFFFFu;
          uint32_t hi = uint32_t(i * 25173u + 0x7FFFu) & 0xFFFFu;
          values[i] = lo | (hi << 16u);
        } else {
          float value = float(int32_t(i % 257u) - 128) * 0.25f;
          std::memcpy(&values[i], &value, sizeof(value));
        }
      }
      auto out = mc.RoundTripMsaaReal(
          surface.w, surface.h, surface.samples, sp, values,
          xenos::kEdramSizeBytes, format.pixel_format, format.format);
      uint32_t bad = 0;
      for (size_t i = 0; i < values.size(); ++i) {
        if (out[i] != values[i]) {
          if (bad < 8) {
            WARN(surface.name << " " << format.name << " idx " << i
                              << " expected " << values[i] << " got "
                              << out[i]);
          }
          ++bad;
        }
      }
      INFO(surface.name << " " << format.name);
      REQUIRE(bad == 0u);
    }
  }
}

TEST_CASE("Canonical 64bpp range transfers preserve both words per sample",
          "[l1][metal][materialize][color64][range][msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct SurfaceCfg {
    uint32_t msaa, samples;
    const char* name;
  } surface_cfgs[] = {
      {0u, 1u, "1x"}, {1u, 2u, "2x"}, {2u, 4u, "4x"}};
  struct FormatCfg {
    const char* name;
    uint32_t format;
    MTLPixelFormat pixel_format;
  } format_cfgs[] = {
      {"16_16_16_16", 5u, MTLPixelFormatRGBA32Float},
      {"16_16_16_16_FLOAT", 7u, MTLPixelFormatRGBA16Float},
      {"32_32_FLOAT", 15u, MTLPixelFormatRG32Float},
  };
  constexpr uint32_t w = 48u, h = 24u, base = 5u;
  constexpr uint32_t range_start = base, range_end = base + 1u;
  constexpr uint32_t sentinel = 0xD15EA5EDu;
  constexpr uint32_t tile_bytes = 80u * 16u * sizeof(uint32_t);
  for (const SurfaceCfg& surface : surface_cfgs) {
    const uint32_t pitch = PitchFor(w, surface.msaa, 1u);
    SurfaceParams sp{base, pitch, surface.msaa, 0u, 1u,
                     uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))};
    for (const FormatCfg& format : format_cfgs) {
      const size_t sample_count = size_t(w) * h * surface.samples;
      std::vector<uint32_t> values(sample_count * 2u);
      for (size_t i = 0; i < sample_count; ++i) {
        if (format.format == 5u) {
          // Every signed-16 lane is exactly representable in the float32
          // working view, including the -32768 endpoint.
          values[i * 2u] = uint32_t(i * 40503u + 0x8000u) |
                            (uint32_t(i * 25173u + 0x7FFFu) << 16u);
          values[i * 2u + 1u] = uint32_t(i * 32771u + 0xFFFFu) |
                                 (uint32_t(i * 16411u) << 16u);
        } else if (format.format == 7u) {
          // Raw half lanes, including explicit signaling/quiet NaN payloads,
          // catch float-path canonicalization instead of merely testing finite
          // values that happen to survive a numeric round trip.
          values[i * 2u] = uint32_t(i * 40503u + 0x7C01u) |
                            (uint32_t(i * 25173u + 0x7E55u) << 16u);
          values[i * 2u + 1u] = uint32_t(i * 32771u + 0xFC01u) |
                                 (uint32_t(i * 16411u + 0x3555u) << 16u);
        } else {
          // Raw float words, likewise forcing both NaN kinds and payloads.
          values[i * 2u] = uint32_t(i * 2654435761u) ^ 0x7F800001u;
          values[i * 2u + 1u] = uint32_t(i * 2246822519u) ^ 0x7FC12345u;
        }
      }
      auto out = mc.RoundTripColorRange(
          w, h, surface.samples, sp, values, xenos::kEdramSizeBytes,
          format.pixel_format, format.format, range_start, range_end, sentinel);
      uint32_t bad = 0u;
      uint32_t outside_bad = 0u;
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          for (uint32_t s = 0; s < surface.samples; ++s) {
            const size_t sample_index =
                (size_t(y) * w + x) * surface.samples + s;
            const size_t value_index = sample_index * 2u;
            uint32_t offset = XeEdramSnapshotOffsetBytes(
                x, y, base, pitch, surface.msaa, false, 1u, s);
            uint32_t tile = offset / tile_bytes;
            const bool in_range = tile >= range_start && tile < range_end;
            for (uint32_t word = 0; word < 2u; ++word) {
              uint32_t expected =
                  in_range ? values[value_index + word] : sentinel;
              if (out[value_index + word] != expected) {
                if (bad < 8u) {
                  WARN(surface.name << " " << format.name << " pixel " << x
                                    << "," << y << " sample " << s << " word "
                                    << word << " tile " << tile << " expected "
                                    << expected << " got "
                                    << out[value_index + word]);
                }
                ++bad;
                if (!in_range) {
                  ++outside_bad;
                }
              }
            }
          }
        }
      }
      INFO(surface.name << " " << format.name);
      CHECK(outside_bad == 0u);
      REQUIRE(bad == 0u);
    }
  }
}

TEST_CASE("Canonical gamma range transfers preserve PWL words per sample",
          "[l1][metal][gamma][materialize][color32][range][msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct SurfaceCfg {
    uint32_t msaa, samples;
    const char* name;
  } surface_cfgs[] = {
      {0u, 1u, "1x"}, {1u, 2u, "2x"}, {2u, 4u, "4x"}};
  constexpr uint32_t w = 88u, h = 20u, base = 9u;
  constexpr uint32_t range_start = base, range_end = base + 1u;
  constexpr uint32_t sentinel = 0xA11A51A5u;
  constexpr uint32_t tile_bytes = 80u * 16u * sizeof(uint32_t);
  for (const SurfaceCfg& surface : surface_cfgs) {
    const uint32_t pitch = PitchFor(w, surface.msaa, 0u);
    SurfaceParams sp{base, pitch, surface.msaa, 0u, 0u,
                     uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))};
    const size_t sample_count = size_t(w) * h * surface.samples;
    std::vector<uint32_t> values(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
      const uint32_t byte = uint32_t(i) & 0xFFu;
      values[i] = byte | (((byte * 73u) & 0xFFu) << 8u) |
                  (((byte * 151u) & 0xFFu) << 16u) |
                  (((byte * 199u) & 0xFFu) << 24u);
    }
    auto out = mc.RoundTripColorRange(
        w, h, surface.samples, sp, values, xenos::kEdramSizeBytes,
        MTLPixelFormatRGBA16Unorm, 1u, range_start, range_end, sentinel);
    uint32_t bad = 0u;
    uint32_t outside_bad = 0u;
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        for (uint32_t s = 0; s < surface.samples; ++s) {
          const size_t index = (size_t(y) * w + x) * surface.samples + s;
          const uint32_t offset = XeEdramSnapshotOffsetBytes(
              x, y, base, pitch, surface.msaa, false, 0u, s);
          const uint32_t tile = offset / tile_bytes;
          const bool in_range = tile >= range_start && tile < range_end;
          const uint32_t expected = in_range ? values[index] : sentinel;
          if (out[index] != expected) {
            if (bad < 8u) {
              WARN(surface.name << " pixel " << x << "," << y << " sample "
                                << s << " tile " << tile << " expected "
                                << expected << " got " << out[index]);
            }
            ++bad;
            if (!in_range) {
              ++outside_bad;
            }
          }
        }
      }
    }
    INFO(surface.name);
    CHECK(outside_bad == 0u);
    REQUIRE(bad == 0u);
  }
}

TEST_CASE("Canonical float-storage range transfers preserve raw 32-bit words",
          "[l1][metal][materialize][color32][raw][range][msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  struct SurfaceCfg {
    uint32_t msaa, samples;
    const char* name;
  } surface_cfgs[] = {
      {0u, 1u, "1x"}, {1u, 2u, "2x"}, {2u, 4u, "4x"}};
  struct FormatCfg {
    const char* name;
    uint32_t format;
    MTLPixelFormat pixel_format;
  } format_cfgs[] = {
      {"16_16_FLOAT", 6u, MTLPixelFormatRG16Float},
      {"32_FLOAT", 14u, MTLPixelFormatR32Float},
  };
  constexpr uint32_t w = 88u, h = 20u, base = 9u;
  constexpr uint32_t range_start = base, range_end = base + 1u;
  constexpr uint32_t sentinel = 0xA11A51A5u;
  constexpr uint32_t tile_bytes = 80u * 16u * sizeof(uint32_t);
  for (const SurfaceCfg& surface : surface_cfgs) {
    const uint32_t pitch = PitchFor(w, surface.msaa, 0u);
    SurfaceParams sp{base, pitch, surface.msaa, 0u, 0u,
                     uint32_t(xenos::kEdramSizeBytes / sizeof(uint32_t))};
    for (const FormatCfg& format : format_cfgs) {
      const size_t sample_count = size_t(w) * h * surface.samples;
      std::vector<uint32_t> values(sample_count);
      for (size_t i = 0; i < sample_count; ++i) {
        values[i] = format.format == 6u
                        ? (uint32_t(i * 40503u + 0x7C01u) |
                           (uint32_t(i * 25173u + 0xFE55u) << 16u))
                        : (uint32_t(i * 2654435761u) ^ 0x7F812345u);
      }
      auto out = mc.RoundTripColorRange(
          w, h, surface.samples, sp, values, xenos::kEdramSizeBytes,
          format.pixel_format, format.format, range_start, range_end, sentinel);
      uint32_t bad = 0u;
      uint32_t outside_bad = 0u;
      for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
          for (uint32_t s = 0; s < surface.samples; ++s) {
            const size_t index = (size_t(y) * w + x) * surface.samples + s;
            uint32_t offset = XeEdramSnapshotOffsetBytes(
                x, y, base, pitch, surface.msaa, false, 0u, s);
            uint32_t tile = offset / tile_bytes;
            const bool in_range = tile >= range_start && tile < range_end;
            uint32_t expected = in_range ? values[index] : sentinel;
            if (out[index] != expected) {
              if (bad < 8u) {
                WARN(surface.name << " " << format.name << " pixel " << x
                                  << "," << y << " sample " << s << " tile "
                                  << tile << " expected " << expected << " got "
                                  << out[index]);
              }
              ++bad;
              if (!in_range) {
                ++outside_bad;
              }
            }
          }
        }
      }
      INFO(surface.name << " " << format.name);
      CHECK(outside_bad == 0u);
      REQUIRE(bad == 0u);
    }
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

// Per-sample 2x/4x variant: one physical EDRAM tile, with a DISTINCT (d24,
// stencil) per (pixel, sample). Returns failed-word count.
uint32_t MsaaDepthBad(MetalCompute& mc, uint32_t fmt, uint32_t msaa) {
  const uint32_t w = msaa == 2u ? 40u : 80u;
  const uint32_t h = 8u;
  const uint32_t ns = 1u << msaa;
  const uint32_t pitch = PitchFor(w, msaa, 0u);
  SurfaceParams sp{0u, pitch, msaa, fmt, 0u,
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

// Drives the mixed 4x->1x->4x chain and checks the GPU's final canonical against
// a CPU replay of the SAME physical-slot writes (A via 4x offsets, then C via
// the 1x rect offsets). Returns the failed-slot count.
uint32_t CrossMsaaBad(MetalCompute& mc, uint32_t fmt) {
  const uint32_t w4 = 40, h4 = 8, ns = 4, w1 = 80, h1 = 16;
  const uint32_t p4 = PitchFor(w4, 2u, 0u), p1 = PitchFor(w1, 0u, 0u);
  std::vector<uint32_t> a4x(size_t(w4) * h4 * ns), c1x(size_t(w1) * h1);
  for (uint32_t i = 0; i < a4x.size(); ++i)
    a4x[i] = ((i * 2654435761u & 0xFFFFFFu) << 8) | ((i * 37u + 1u) & 0xFFu);
  for (uint32_t i = 0; i < c1x.size(); ++i)
    c1x[i] = ((i * 40503u & 0xFFFFFFu) << 8) | ((i * 101u + 7u) & 0xFFu);
  const uint32_t rx0 = 48, ry0 = 4, rx1 = 72, ry1 = 12;  // 1x-space write rect
  auto out = mc.CrossMsaaChain(fmt, a4x, c1x, rx0, ry0, rx1, ry1,
                               xenos::kEdramSizeBytes);
  auto off4 = [&](uint32_t x, uint32_t y, uint32_t s) {
    return XeEdramSnapshotOffsetBytes(x, y, 0u, p4, 2u, true, 0u, s) >> 2;
  };
  auto off1 = [&](uint32_t x, uint32_t y) {
    return XeEdramSnapshotOffsetBytes(x, y, 0u, p1, 0u, true, 0u, 0u) >> 2;
  };
  std::vector<uint32_t> exp(1280, 0);
  for (uint32_t y = 0; y < h4; ++y)
    for (uint32_t x = 0; x < w4; ++x)
      for (uint32_t s = 0; s < ns; ++s)
        exp[off4(x, y, s)] = a4x[(y * w4 + x) * ns + s];
  for (uint32_t y = ry0; y < ry1; ++y)
    for (uint32_t x = rx0; x < rx1; ++x) exp[off1(x, y)] = c1x[y * w1 + x];
  uint32_t bad = 0;
  for (uint32_t i = 0; i < 1280; ++i) {
    if (out[i] != exp[i]) {
      if (bad < 8)
        WARN("fmt " << fmt << " slot " << i << " expected " << exp[i] << " got "
                    << out[i]);
      ++bad;
    }
  }
  return bad;
}

// Round-trips a depth surface at ARBITRARY geometry (nonzero base, any pitch,
// multi-tile, 1x or 4x) and reports failed complete-words. This is what the
// single-tile base=0 pitch=1 cases don't cover: JC2's real [720,1440) buffer is
// base 720, pitch 16, spanning many tiles. Distinct depth+stencil per element.
uint32_t GeomDepthBad(MetalCompute& mc, uint32_t fmt, uint32_t base,
                      uint32_t pitch, uint32_t w, uint32_t h, uint32_t msaa,
                      uint32_t ns) {
  SurfaceParams sp{base, pitch, msaa, fmt, 0u,
                   uint32_t(xenos::kEdramSizeBytes / 4u)};
  std::vector<uint32_t> words(size_t(w) * h * ns);
  for (uint32_t i = 0; i < words.size(); ++i)
    words[i] = ((i * 2654435761u & 0xFFFFFFu) << 8) | ((i * 37u + 1u) & 0xFFu);
  auto out = ns > 1
                 ? mc.RoundTripDepthMsaa(w, h, ns, sp, words, xenos::kEdramSizeBytes)
                 : mc.RoundTripDepth(w, h, sp, words, xenos::kEdramSizeBytes);
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

TEST_CASE("Canonical 2x/4x per-sample depth+stencil round-trips every sample",
          "[l1][metal][materialize][depth][msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // Both guest depth formats, every 2x/4x sample independent: the sample
  // dimension the collapsed host cannot hold and the cross-MSAA slice depends
  // on. Distinct depth AND stencil per (pixel, sample) makes any collapse or
  // cross-sample leak fail loudly.
  REQUIRE(MsaaDepthBad(mc, /*float24=*/2u, /*2x=*/1u) == 0u);
  REQUIRE(MsaaDepthBad(mc, /*unorm24=*/1u, /*2x=*/1u) == 0u);
  REQUIRE(MsaaDepthBad(mc, /*float24=*/2u, /*4x=*/2u) == 0u);
  REQUIRE(MsaaDepthBad(mc, /*unorm24=*/1u, /*4x=*/2u) == 0u);
}

TEST_CASE("Cross-MSAA 4x->1x->4x preserves samples and propagates a 1x write",
          "[l1][metal][materialize][depth][msaa][cross_msaa]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // The live slice's exact contract, through REAL attachments: write A via the
  // 4x view, relabel to 1x, write C into a sub-rect via the 1x view, relabel
  // back to 4x. Every untouched sample must survive AND the 1x write must show
  // up at 4x. Checked against a CPU replay of the same physical-slot writes.
  REQUIRE(CrossMsaaBad(mc, /*float24=*/2u) == 0u);
  REQUIRE(CrossMsaaBad(mc, /*unorm24=*/1u) == 0u);
}

TEST_CASE("Canonical depth round-trips at JC2 geometry (base 720, pitch 16, multi-tile)",
          "[l1][metal][materialize][depth][geometry]") {
  MetalCompute mc;
  if (!mc.Init()) { SUCCEED("no device"); return; }
  // JC2's [720,1440) buffer: base 720, pitch 16. The region is only 2 tiles
  // wide, so pitch 16 != region-width-in-tiles -> non-contiguous tile rows
  // (row 1 jumps 16 tiles, not 2), exercising the pitch arithmetic the single-
  // tile base=0 pitch=1 cases can't. 1x = 160x32 px (2x2 tiles), 4x = 80x16 px
  // x 4 samples (2x2 tiles). Both guest formats, complete depth+stencil word.
  REQUIRE(GeomDepthBad(mc, /*float24=*/2u, 720u, 16u, 160u, 32u, 0u, 1u) == 0u);
  REQUIRE(GeomDepthBad(mc, /*unorm24=*/1u, 720u, 16u, 160u, 32u, 0u, 1u) == 0u);
  REQUIRE(GeomDepthBad(mc, /*float24=*/2u, 720u, 16u, 80u, 16u, 2u, 4u) == 0u);
  REQUIRE(GeomDepthBad(mc, /*unorm24=*/1u, 720u, 16u, 80u, 16u, 2u, 4u) == 0u);
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
