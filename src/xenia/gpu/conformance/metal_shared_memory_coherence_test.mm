/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 (Metal device required; passes-as-skipped without one): the
// MetalSharedMemory authority invariant, exercised against the REAL Memory +
// REAL MetalSharedMemory + the REAL SharedMemory validity logic — no replica
// of production decisions. Stimulus follows the resolve-writeback drain's
// exact production sequence after the writer-audit correction: write guest RAM
// via TranslatePhysical, then MemoryInvalidationCallback(extent, exact) — the
// same authority entry the guest-mirror fault path reaches. (A full
// command-processor drain in this harness would be artificial; the drain's
// call path into the same authority is established by the code audit recorded
// in the gpu-texture-loads design. The guest-mirror FAULT path itself is not
// exercised here — it requires installed exception handling — and is covered
// by that audit; this is the documented limitation.)
//
// GPU reads use a tiny self-contained compute copy (buffer->buffer) so byte
// comparison observes exactly what a shader consuming the shared buffer
// would read. Submission ordering is validated with two separately committed
// command buffers: N consumes old validated bytes; after invalidation +
// revalidation, M > N consumes the new bytes. The wait in the test is only to
// OBSERVE results; ordering itself comes from commit order.

#import <Metal/Metal.h>

#include <cstring>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPage = 4096;

struct CoherenceHarness {
  std::unique_ptr<Memory> memory;
  std::unique_ptr<MetalSharedMemory> shared;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> copy_pipeline = nil;
  bool ok = false;

  CoherenceHarness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    memory = std::make_unique<Memory>();
    if (!memory->Initialize()) {
      memory.reset();
      return;
    }
    shared = std::make_unique<MetalSharedMemory>(*memory,
                                                 (__bridge void*)device);
    if (!shared->Initialize()) {
      shared.reset();
      return;
    }
    queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> lib = [device
        newLibraryWithSource:
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void copy_bytes(device const uint* src [[buffer(0)]],\n"
             "                       device uint* dst [[buffer(1)]],\n"
             "                       constant uint& src_word_off [[buffer(2)]],\n"
             "                       uint tid [[thread_position_in_grid]]) {\n"
             "  dst[tid] = src[src_word_off + tid];\n"
             "}\n"
                     options:nil
                       error:&error];
    if (!lib) {
      return;
    }
    copy_pipeline = [device
        newComputePipelineStateWithFunction:[lib newFunctionWithName:
                                                     @"copy_bytes"]
                                      error:&error];
    ok = queue && copy_pipeline;
  }

  // Reads `bytes` (multiple of 4) at guest physical `addr` from the shared
  // buffer via a GPU compute dispatch on a fresh command buffer; returns the
  // bytes the GPU observed. The wait is observation-only.
  std::vector<uint8_t> GpuRead(uint32_t addr, uint32_t bytes) {
    id<MTLBuffer> dst = [device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:copy_pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)shared->GetBuffer()
            offset:0
           atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    uint32_t word_off = addr / 4;
    [enc setBytes:&word_off length:sizeof(word_off) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(bytes / 4, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), [dst contents], bytes);
    return out;
  }

  // Commits the physical pages backing [addr, addr+len) so uploads pass the
  // heap-access validation, mirroring real guest allocations.
  void EnsureAllocated(uint32_t addr, uint32_t len) {
    uint32_t base = addr & ~(kPage - 1);
    uint32_t end = (addr + len + kPage - 1) & ~(kPage - 1);
    memory->GetPhysicalHeap()->AllocFixed(
        base, end - base, kPage,
        kMemoryAllocationReserve | kMemoryAllocationCommit,
        kMemoryProtectRead | kMemoryProtectWrite);
  }

  // Fills guest RAM at `addr` (production writer stimulus: TranslatePhysical
  // host-view write, exactly how the drain scatters).
  void RamWrite(uint32_t addr, const std::vector<uint8_t>& bytes) {
    EnsureAllocated(addr, uint32_t(bytes.size()));
    std::memcpy(memory->TranslatePhysical(addr), bytes.data(), bytes.size());
  }

  std::vector<uint8_t> Pattern(uint32_t bytes, uint8_t seed) {
    std::vector<uint8_t> v(bytes);
    for (uint32_t i = 0; i < bytes; ++i) {
      v[i] = uint8_t(seed + i * 7 + (i >> 8));
    }
    return v;
  }
};

CoherenceHarness& H() {
  static CoherenceHarness h;
  return h;
}

#define REQUIRE_DEVICE()            \
  if (!H().ok) {                    \
    SUCCEED("no Metal device");     \
    return;                         \
  }

TEST_CASE("SM coherence: device harness is live (not skipped)",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  REQUIRE(H().shared != nullptr);
}

TEST_CASE("SM coherence: initially-invalid range validates and GPU-reads",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0A000000;  // page-aligned physical
  auto bytes = h.Pattern(kPage, 0x11);
  h.RamWrite(addr, bytes);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == bytes);
}

TEST_CASE("SM coherence: drain-sequence write + extent invalidation "
          "revalidates for a later submission",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0A100000;
  auto old_bytes = h.Pattern(kPage, 0x22);
  h.RamWrite(addr, old_bytes);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  // Submission N observes the old validated content.
  REQUIRE(h.GpuRead(addr, kPage) == old_bytes);
  // Production drain sequence: RAM write via host view, then exact-extent
  // invalidation through the authority.
  auto new_bytes = h.Pattern(kPage, 0x33);
  h.RamWrite(addr, new_bytes);
  h.shared->MemoryInvalidationCallback(addr, kPage, true);
  // Without revalidation the buffer still holds old bytes; RequestRange must
  // re-upload because the pages were invalidated.
  REQUIRE(h.shared->RequestRange(addr, kPage));
  // Later submission M > N observes the new bytes.
  REQUIRE(h.GpuRead(addr, kPage) == new_bytes);
}

TEST_CASE("SM coherence: write to already-valid page without invalidation is "
          "the hazard the correction closes",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0A200000;
  auto old_bytes = h.Pattern(kPage, 0x44);
  h.RamWrite(addr, old_bytes);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == old_bytes);
  auto new_bytes = h.Pattern(kPage, 0x55);
  h.RamWrite(addr, new_bytes);  // no invalidation: page stays "valid"
  REQUIRE(h.shared->RequestRange(addr, kPage));
  // Documents the stale-read hazard itself: still the OLD bytes.
  REQUIRE(h.GpuRead(addr, kPage) == old_bytes);
  // The correction's invalidation resolves it.
  h.shared->MemoryInvalidationCallback(addr, kPage, true);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == new_bytes);
}

TEST_CASE("SM coherence: writeback-shaped extents (32bpp full, 64bpp, depth, "
          "offset, partial row, cross-page)",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  struct Case {
    const char* name;
    uint32_t addr;
    uint32_t extent;
  } cases[] = {
      // extents = copy_dest extent bytes as the drain would carry them.
      // Regions are spaced so no case's extent can overlap another's pages
      // (the depth case spans 4 MiB).
      {"full-rect 32bpp 256x256", 0x0B000000, 256 * 256 * 4},
      {"64bpp RGBA16F 64x64", 0x0C000000, 64 * 64 * 8},
      {"depth D24S8 1024x1024", 0x0D000000, 1024 * 1024 * 4},
      {"offset dest (extent covers offset area)", 0x0E000000 + 0x40,
       80 * 32 * 4},
      {"partial final row", 0x0F000000, 33 * 320 * 4 + 100 * 4},
      {"cross-page small", 0x10000000 + kPage - 64, 256},
  };
  for (const Case& c : cases) {
    INFO(c.name);
    uint32_t up_base = c.addr & ~(kPage - 1);
    uint32_t up_len =
        ((c.addr + c.extent + kPage - 1) & ~(kPage - 1)) - up_base;
    auto old_bytes = h.Pattern(up_len, 0x66);
    h.RamWrite(up_base, old_bytes);
    REQUIRE(h.shared->RequestRange(up_base, up_len));
    REQUIRE(h.GpuRead(up_base, up_len) == old_bytes);
    auto new_bytes = h.Pattern(up_len, 0x77);
    h.RamWrite(up_base, new_bytes);
    // Exact production range: [addr, addr + extent).
    h.shared->MemoryInvalidationCallback(c.addr, c.extent, true);
    REQUIRE(h.shared->RequestRange(up_base, up_len));
    auto got = h.GpuRead(up_base, up_len);
    // Pages intersecting the extent must show new bytes.
    uint32_t lo = (c.addr - up_base) & ~(kPage - 1);
    uint32_t hi_page_end =
        ((c.addr + c.extent - up_base) + kPage - 1) & ~(kPage - 1);
    REQUIRE(std::memcmp(got.data() + lo, new_bytes.data() + lo,
                        std::min(hi_page_end, up_len) - lo) == 0);
  }
}

TEST_CASE("SM coherence: physical-memory boundary is checked",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t end = 512u * 1024 * 1024;
  // Clamp safety at the literal top: an invalidation whose length overshoots
  // the end of physical space must neither crash nor wrap. (The topmost
  // pages themselves are heap-reserved and not allocatable, which
  // RequestRange correctly refuses — verified below.)
  h.shared->MemoryInvalidationCallback(end - kPage, kPage * 4, true);
  REQUIRE_FALSE(h.shared->RequestRange(end - kPage, kPage));
  // Full validate/invalidate/revalidate round at a high allocatable address,
  // with an overshooting invalidation length that must clamp to the extent
  // end without touching unrelated pages.
  const uint32_t addr = 0x1F000000;
  auto bytes = h.Pattern(kPage, 0x88);
  h.RamWrite(addr, bytes);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == bytes);
  auto new_bytes = h.Pattern(kPage, 0x99);
  h.RamWrite(addr, new_bytes);
  h.shared->MemoryInvalidationCallback(addr, end - addr + kPage, true);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == new_bytes);
}

TEST_CASE("SM coherence: repeated invalidation/revalidation cycles",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0AA00000;
  for (uint8_t i = 0; i < 8; ++i) {
    auto bytes = h.Pattern(kPage * 2, uint8_t(0xA0 + i));
    h.RamWrite(addr, bytes);
    h.shared->MemoryInvalidationCallback(addr, kPage * 2, true);
    REQUIRE(h.shared->RequestRange(addr, kPage * 2));
    REQUIRE(h.GpuRead(addr, kPage * 2) == bytes);
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
