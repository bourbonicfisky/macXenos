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

// Per-surface constants for pack/unpack (one surface per dispatch).
struct SurfaceParams { uint base, pitch, msaa, is_depth, fmt_log2, pad; };
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
)MSL";

// Mirrors of the MSL pack/unpack structs -- layouts must match exactly or the
// round-trip corrupts (a stride mismatch would fail loudly, by design).
struct Coord {
  uint32_t px, py, sample;
};
struct SurfaceParams {
  uint32_t base, pitch, msaa, is_depth, fmt_log2, pad;
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

}  // namespace test
}  // namespace gpu
}  // namespace xe
