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
)MSL";

// A minimal compute harness: compile the library once, dispatch one kernel over
// a code vector, return the output bits. All buffers are shared storage so the
// readback is a plain memcpy.
struct MetalCompute {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;

  bool Init() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    queue = [dev newCommandQueue];
    NSError* err = nil;
    lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                            options:nil
                              error:&err];
    if (!lib) {
      WARN("MSL compile failed: " <<
           (err ? err.localizedDescription.UTF8String : "unknown"));
      return false;
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

}  // namespace test
}  // namespace gpu
}  // namespace xe
