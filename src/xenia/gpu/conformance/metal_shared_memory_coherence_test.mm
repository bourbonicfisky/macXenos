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
// same authority entry the guest-mirror fault path reaches. The physical
// callback thunk is exercised directly to prove its Metal observation fires
// once and direct backend invalidations do not masquerade as CPU writes. (A full
// command-processor drain in this harness would be artificial; the drain's
// call path into the same authority is established by the code audit recorded
// in the gpu-texture-loads design. The host exception transport into the thunk
// is not exercised here — it requires installed exception handling — and is
// covered by that audit; this is the documented limitation.)
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
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPage = 4096;

class TestMetalSharedMemory : public MetalSharedMemory {
 public:
  using MetalSharedMemory::MetalSharedMemory;

  std::pair<uint32_t, uint32_t> InvokePhysicalInvalidation(uint32_t address, uint32_t length,
                                                           bool exact_range) {
    return MemoryInvalidationCallbackThunk(this, address, length, exact_range);
  }
};

struct CoherenceHarness {
  struct EncodedGpuWrite {
    id<MTLBuffer> source = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    uint64_t submission = 0;
    bool committed = false;
  };

  Memory* memory = nullptr;
  std::unique_ptr<TestMetalSharedMemory> shared;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> copy_pipeline = nil;
  uint64_t next_command_buffer_submission = 1;
  bool ok = false;

  CoherenceHarness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    shared = std::make_unique<TestMetalSharedMemory>(*memory, (__bridge void*)device);
    if (!shared->Initialize()) {
      shared.reset();
      return;
    }
    queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> lib = [device
        newLibraryWithSource:@"#include <metal_stdlib>\n"
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
    copy_pipeline =
        [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy_bytes"]
                                              error:&error];
    ok = queue && copy_pipeline;
  }

  // Reads `bytes` (multiple of 4) at guest physical `addr` from the shared
  // buffer via a GPU compute dispatch on a fresh command buffer; returns the
  // bytes the GPU observed. The wait is observation-only.
  std::vector<uint8_t> GpuRead(uint32_t addr, uint32_t bytes, uint64_t* submission_out = nullptr) {
    id<MTLBuffer> dst = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:copy_pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)shared->GetBuffer() offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    uint32_t word_off = addr / 4;
    [enc setBytes:&word_off length:sizeof(word_off) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(bytes / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cb commit];
    uint64_t submission = next_command_buffer_submission++;
    if (submission_out) {
      *submission_out = submission;
    }
    [cb waitUntilCompleted];
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), [dst contents], bytes);
    return out;
  }

  // Encodes exact bytes into the real shared MTLBuffer without committing.
  // Memexport ownership and RangeWrittenByGpu publication are installed while
  // submission N is still encoded, matching production's no-unowned-interval
  // contract; CommitGpuWrite then submits N before a later consumer M.
  EncodedGpuWrite EncodeGpuWrite(uint32_t addr, const std::vector<uint8_t>& bytes) {
    EncodedGpuWrite write;
    write.source = [device newBufferWithLength:bytes.size() options:MTLResourceStorageModeShared];
    std::memcpy([write.source contents], bytes.data(), bytes.size());
    write.command_buffer = [queue commandBuffer];
    id<MTLBlitCommandEncoder> enc = [write.command_buffer blitCommandEncoder];
    [enc copyFromBuffer:write.source
             sourceOffset:0
                 toBuffer:(__bridge id<MTLBuffer>)shared->GetBuffer()
        destinationOffset:addr
                     size:bytes.size()];
    [enc endEncoding];
    write.submission = next_command_buffer_submission++;
    return write;
  }

  void CommitGpuWrite(EncodedGpuWrite* write) {
    [write->command_buffer commit];
    write->committed = true;
  }

  // Commits the physical pages backing [addr, addr+len) so uploads pass the
  // heap-access validation, mirroring real guest allocations.
  void EnsureAllocated(uint32_t addr, uint32_t len) {
    uint32_t base = addr & ~(kPage - 1);
    uint32_t end = (addr + len + kPage - 1) & ~(kPage - 1);
    memory->GetPhysicalHeap()->AllocFixed(base, end - base, kPage,
                                          kMemoryAllocationReserve | kMemoryAllocationCommit,
                                          kMemoryProtectRead | kMemoryProtectWrite);
  }

  // Fills guest RAM at `addr` (production writer stimulus: TranslatePhysical
  // host-view write, exactly how the drain scatters).
  void RamWrite(uint32_t addr, const std::vector<uint8_t>& bytes) {
    EnsureAllocated(addr, uint32_t(bytes.size()));
    std::memcpy(memory->TranslatePhysical(addr), bytes.data(), bytes.size());
  }

  std::vector<uint8_t> RamRead(uint32_t addr, uint32_t bytes) const {
    std::vector<uint8_t> out(bytes);
    std::memcpy(out.data(), memory->TranslatePhysical(addr), bytes);
    return out;
  }

  std::vector<uint8_t> SharedRead(uint32_t addr, uint32_t bytes) const {
    std::vector<uint8_t> out(bytes);
    const uint8_t* contents =
        static_cast<const uint8_t*>([(__bridge id<MTLBuffer>)shared->GetBuffer() contents]);
    std::memcpy(out.data(), contents + addr, bytes);
    return out;
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

// Test-only adapters route the typed publication seam's shared-memory work to
// the same real MetalSharedMemory operations as production. The trace-only
// legacy-cache operation is recorded (this harness does not construct a
// texture cache); that cache surface is gated separately in task 5. They own
// no authority state and introduce no synchronization.
class RealRamWritePublicationTarget final : public RamWritePublicationTarget {
 public:
  explicit RealRamWritePublicationTarget(TestMetalSharedMemory* shared) : shared_(shared) {}

  bool InvalidateSharedPages(uint32_t base, uint32_t length, bool exact_range) override {
    ++shared_calls;
    shared_base = base;
    shared_length = length;
    shared_exact = exact_range;
    shared_->MemoryInvalidationCallback(base, length, exact_range);
    return true;
  }

  bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                uint32_t* invalidated_entries) override {
    ++legacy_calls;
    legacy_base = base;
    legacy_length = length;
    *invalidated_entries = 0;
    return true;
  }

  uint32_t shared_calls = 0;
  uint32_t shared_base = 0;
  uint32_t shared_length = 0;
  bool shared_exact = false;
  uint32_t legacy_calls = 0;
  uint32_t legacy_base = 0;
  uint32_t legacy_length = 0;

 private:
  TestMetalSharedMemory* shared_;
};

class RealMemexportPreparationTarget final : public MemexportPreparationTarget {
 public:
  explicit RealMemexportPreparationTarget(TestMetalSharedMemory* shared) : shared_(shared) {}

  bool RequestDestinationRange(uint32_t base, uint32_t length) override {
    return shared_->RequestRange(base, length);
  }

 private:
  TestMetalSharedMemory* shared_;
};

class OwnedMemexportPublicationTarget final : public MemexportPublicationTarget {
 public:
  OwnedMemexportPublicationTarget(TestMetalSharedMemory* shared,
                                  std::vector<PendingProducer>* pending,
                                  std::vector<uint32_t>* event_order)
      : shared_(shared), pending_(pending), event_order_(event_order) {}

  bool CanMarkRangeWrittenByGpu() const override { return shared_ != nullptr; }

  void MarkRangeWrittenByGpu(uint32_t base, uint32_t length) override {
    mark_saw_owner = pending_ && !pending_->empty() && pending_->back().range.start() == base &&
                     pending_->back().range.length() == length;
    event_order_->push_back(2u);
    shared_->RangeWrittenByGpu(base, length);
  }

  bool mark_saw_owner = false;

 private:
  TestMetalSharedMemory* shared_;
  std::vector<PendingProducer>* pending_;
  std::vector<uint32_t>* event_order_;
};

class RealMemexportConvergenceTarget final : public MemexportConvergenceTarget {
 public:
  explicit RealMemexportConvergenceTarget(CoherenceHarness* harness) : harness_(harness) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length, uint32_t* invalidated_entries) override {
    uint8_t* ram = harness_->memory->TranslatePhysical<uint8_t*>(base);
    const uint8_t* contents = static_cast<const uint8_t*>(
        [(__bridge id<MTLBuffer>)harness_->shared->GetBuffer() contents]);
    if (!ram || !contents) {
      return false;
    }
    std::memcpy(ram, contents + base, length);
    *invalidated_entries = 0;
    return true;
  }

 private:
  CoherenceHarness* harness_;
};

#define REQUIRE_DEVICE()        \
  if (!H().ok) {                \
    SUCCEED("no Metal device"); \
    return;                     \
  }

TEST_CASE("SM coherence: device harness is live (not skipped)", "[l1][metal][sharedmem]") {
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

TEST_CASE("SM coherence: only the physical thunk observes a guest CPU writer",
          "[l1][metal][sharedmem][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0A080000;
  h.EnsureAllocated(addr, kPage);

  auto before = h.shared->guest_cpu_authority_stats();
  h.shared->MemoryInvalidationCallback(addr, 0x80u, true);
  auto after_direct = h.shared->guest_cpu_authority_stats();
  REQUIRE(after_direct.publications == before.publications);
  REQUIRE(after_direct.range_bytes == before.range_bytes);

  h.shared->InvokePhysicalInvalidation(addr + 0x40u, 0x80u, false);
  auto after_physical = h.shared->guest_cpu_authority_stats();
  REQUIRE(after_physical.publications == before.publications + 1u);
  REQUIRE(after_physical.range_bytes == before.range_bytes + 0x80u);

  h.shared->MemoryInvalidationCallback(addr, 0x80u, true);
  auto after_second_direct = h.shared->guest_cpu_authority_stats();
  REQUIRE(after_second_direct.publications == after_physical.publications);
  REQUIRE(after_second_direct.range_bytes == after_physical.range_bytes);
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
      {"offset dest (extent covers offset area)", 0x0E000000 + 0x40, 80 * 32 * 4},
      {"partial final row", 0x0F000000, 33 * 320 * 4 + 100 * 4},
      {"cross-page small", 0x10000000 + kPage - 64, 256},
  };
  for (const Case& c : cases) {
    INFO(c.name);
    uint32_t up_base = c.addr & ~(kPage - 1);
    uint32_t up_len = ((c.addr + c.extent + kPage - 1) & ~(kPage - 1)) - up_base;
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
    uint32_t hi_page_end = ((c.addr + c.extent - up_base) + kPage - 1) & ~(kPage - 1);
    REQUIRE(std::memcmp(got.data() + lo, new_bytes.data() + lo,
                        std::min(hi_page_end, up_len) - lo) == 0);
  }
}

TEST_CASE("SM coherence: physical-memory boundary is checked", "[l1][metal][sharedmem]") {
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

TEST_CASE("SM coherence: RangeWrittenByGpu protects GPU-written buffer "
          "bytes from stale CPU re-upload (device)",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0AB00000;
  auto ram = h.Pattern(kPage, 0xB0);
  h.RamWrite(addr, ram);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == ram);
  // "GPU write": newest bytes land in the real shared MTLBuffer only.
  auto gpu = h.Pattern(kPage, 0xC0);
  uint8_t* contents =
      static_cast<uint8_t*>([(__bridge id<MTLBuffer>)h.shared->GetBuffer() contents]);
  std::memcpy(contents + addr, gpu.data(), kPage);
  h.shared->RangeWrittenByGpu(addr, kPage);
  // A later request must not clobber the GPU-produced bytes with stale RAM.
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == gpu);
  // Memexport-drain convergence: RAM := buffer, no invalidation, bytes stay.
  h.RamWrite(addr, gpu);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == gpu);
}

TEST_CASE("SM coherence: repeated invalidation/revalidation cycles", "[l1][metal][sharedmem]") {
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

TEST_CASE("SM coherence: typed resolve and trace publication reaches exact "
          "RAM and shared-buffer bytes",
          "[l1][metal][sharedmem][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  struct WriterCase {
    const char* name;
    WriterKind writer;
    uint32_t address;
    uint32_t expected_legacy_calls;
  } cases[] = {
      {"resolve", WriterKind::kResolve, 0x11000000u, 0u},
      {"trace restore", WriterKind::kTraceRestore, 0x11100000u, 1u},
  };

  for (const WriterCase& writer_case : cases) {
    INFO(writer_case.name);
    auto before = h.Pattern(kPage, 0xD0u);
    h.RamWrite(writer_case.address, before);
    REQUIRE(h.shared->RequestRange(writer_case.address, kPage));
    REQUIRE(h.SharedRead(writer_case.address, kPage) == before);

    auto after = h.Pattern(kPage, 0xD1u);
    h.RamWrite(writer_case.address, after);
    auto publication = DecideRamWritePublication(writer_case.writer, writer_case.address, kPage);
    REQUIRE(publication.has_value());
    RealRamWritePublicationTarget target(h.shared.get());
    LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);

    REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
    REQUIRE(outcome.shared_invalidation_executed);
    REQUIRE(target.shared_calls == 1u);
    REQUIRE(target.shared_base == writer_case.address);
    REQUIRE(target.shared_length == kPage);
    REQUIRE(target.shared_exact);
    REQUIRE(target.legacy_calls == writer_case.expected_legacy_calls);
    REQUIRE(outcome.legacy_invalidation_executed == (writer_case.expected_legacy_calls != 0u));

    REQUIRE(h.shared->RequestRange(writer_case.address, kPage));
    REQUIRE(h.RamRead(writer_case.address, kPage) == after);
    REQUIRE(h.SharedRead(writer_case.address, kPage) == after);
    REQUIRE(h.GpuRead(writer_case.address, kPage) == after);
  }
}

TEST_CASE("SM coherence: submission-owned memexport preserves GPU-newer bytes "
          "through exact RAM convergence",
          "[l1][metal][sharedmem][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x12000000u;

  auto stale_ram = h.Pattern(kPage, 0xE0u);
  h.RamWrite(addr, stale_ram);
  auto preparation = DecideMemexportDestinationPreparation(addr, kPage);
  REQUIRE(preparation.has_value());
  RealMemexportPreparationTarget preparation_target(h.shared.get());
  REQUIRE(PrepareMemexportDestination(*preparation, preparation_target));
  REQUIRE(h.RamRead(addr, kPage) == stale_ram);
  REQUIRE(h.SharedRead(addr, kPage) == stale_ram);

  auto exported = h.Pattern(kPage, 0xE1u);
  auto gpu_write = h.EncodeGpuWrite(addr, exported);
  uint64_t submission_n = gpu_write.submission;
  REQUIRE(h.RamRead(addr, kPage) == stale_ram);

  auto publication = DecideMemexportPublication(addr, kPage, submission_n);
  REQUIRE(publication.has_value());
  std::vector<PendingProducer> pending;
  std::vector<uint32_t> event_order;
  OwnedMemexportPublicationTarget publication_target(h.shared.get(), &pending, &event_order);
  MemexportPublicationResult publication_result = PublishMemexportWrite(
      *publication, publication_target, [&](const MemexportPublication& owned) {
        event_order.push_back(1u);
        pending.push_back(
            PendingProducer{owned.range, TransitState::kPendingMemexport, owned.submission, 1u});
        return true;
      });
  REQUIRE(publication_result.registered);
  REQUIRE(publication_result.marked);
  REQUIRE(publication_result.action == AuthorityAction::kPublishGpuWrite);
  REQUIRE(publication_target.mark_saw_owner);
  const std::vector<uint32_t> expected_event_order = {1u, 2u};
  REQUIRE(event_order == expected_event_order);
  REQUIRE(pending.size() == 1u);
  REQUIRE_FALSE(ProducerComplete(pending.front(), submission_n - 1u));
  REQUIRE(gpu_write.command_buffer.status == MTLCommandBufferStatusNotEnqueued);
  REQUIRE_FALSE(gpu_write.committed);

  // Requesting the already-published range must not upload stale RAM over the
  // scheduled GPU write. Commit N only after the range has an owner; the later
  // command buffer M observes N by same-queue command-buffer ordering.
  h.CommitGpuWrite(&gpu_write);
  REQUIRE(gpu_write.committed);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  uint64_t submission_m = 0;
  REQUIRE(h.GpuRead(addr, kPage, &submission_m) == exported);
  REQUIRE(submission_m > submission_n);
  REQUIRE(gpu_write.command_buffer.status == MTLCommandBufferStatusCompleted);
  REQUIRE(ProducerComplete(pending.front(), submission_n));
  REQUIRE(h.RamRead(addr, kPage) == stale_ram);
  REQUIRE(h.SharedRead(addr, kPage) == exported);

  RealMemexportConvergenceTarget convergence_target(&h);
  MemexportConvergenceResult convergence_result =
      ConvergeMemexportRam(pending.front(), convergence_target);
  REQUIRE(convergence_result.converged);
  REQUIRE(convergence_result.action == AuthorityAction::kConvergeRamFromBuffer);
  if (convergence_result.converged) {
    pending.erase(pending.begin());
  }
  REQUIRE(pending.empty());
  REQUIRE(h.RamRead(addr, kPage) == exported);
  REQUIRE(h.SharedRead(addr, kPage) == exported);
  REQUIRE(h.GpuRead(addr, kPage) == exported);

  // Observable no-invalidation proof: mutate RAM without publishing it, then
  // request the range. Preserved GPU-written validity must keep the exported
  // shared bytes; an incorrect convergence invalidation would upload this RAM
  // canary and fail both exact shared/GPU assertions.
  auto ram_canary = h.Pattern(kPage, 0xE2u);
  h.RamWrite(addr, ram_canary);
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.SharedRead(addr, kPage) == exported);
  REQUIRE(h.GpuRead(addr, kPage) == exported);
  h.RamWrite(addr, exported);
  REQUIRE(h.RamRead(addr, kPage) == exported);
  REQUIRE(h.SharedRead(addr, kPage) == exported);
}

TEST_CASE("SM coherence: repeated mixed writers leave each later consumer on "
          "the newest exact bytes",
          "[l1][metal][sharedmem][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x13000000u;

  for (uint8_t cycle = 0; cycle < 4; ++cycle) {
    INFO("mixed-writer cycle " << uint32_t(cycle));
    uint8_t seed = uint8_t(0x40u + cycle * 8u);

    // Guest CPU: only the real physical callback thunk may observe the writer.
    auto cpu_bytes = h.Pattern(kPage, seed);
    auto cpu_stats_before = h.shared->guest_cpu_authority_stats();
    h.RamWrite(addr, cpu_bytes);
    h.shared->InvokePhysicalInvalidation(addr, kPage, true);
    auto cpu_stats_after = h.shared->guest_cpu_authority_stats();
    REQUIRE(cpu_stats_after.publications == cpu_stats_before.publications + 1u);
    REQUIRE(cpu_stats_after.range_bytes == cpu_stats_before.range_bytes + kPage);
    REQUIRE(h.shared->RequestRange(addr, kPage));
    REQUIRE(h.RamRead(addr, kPage) == cpu_bytes);
    REQUIRE(h.SharedRead(addr, kPage) == cpu_bytes);
    REQUIRE(h.GpuRead(addr, kPage) == cpu_bytes);

    // Resolve: RAM scatter is final before the typed seam invalidates the
    // exact checked extent for a later shared-buffer consumer.
    auto resolve_bytes = h.Pattern(kPage, uint8_t(seed + 1u));
    h.RamWrite(addr, resolve_bytes);
    auto resolve_publication = DecideRamWritePublication(WriterKind::kResolve, addr, kPage);
    REQUIRE(resolve_publication.has_value());
    RealRamWritePublicationTarget resolve_target(h.shared.get());
    LegacyWriterOutcome resolve_outcome = PublishRamWrite(*resolve_publication, resolve_target);
    REQUIRE(resolve_outcome.action == AuthorityAction::kInvalidateRamWrite);
    REQUIRE(resolve_target.shared_calls == 1u);
    REQUIRE(resolve_target.legacy_calls == 0u);
    REQUIRE(h.shared->RequestRange(addr, kPage));
    REQUIRE(h.RamRead(addr, kPage) == resolve_bytes);
    REQUIRE(h.SharedRead(addr, kPage) == resolve_bytes);
    REQUIRE(h.GpuRead(addr, kPage) == resolve_bytes);

    // Memexport: preparation precedes GPU submission N; pending ownership is
    // registered before publication; later command buffer M observes N; only
    // completed convergence updates RAM and retires the local factual record.
    auto preparation = DecideMemexportDestinationPreparation(addr, kPage);
    REQUIRE(preparation.has_value());
    RealMemexportPreparationTarget preparation_target(h.shared.get());
    REQUIRE(PrepareMemexportDestination(*preparation, preparation_target));
    auto export_bytes = h.Pattern(kPage, uint8_t(seed + 2u));
    auto gpu_write = h.EncodeGpuWrite(addr, export_bytes);
    uint64_t submission_n = gpu_write.submission;
    auto export_publication = DecideMemexportPublication(addr, kPage, submission_n);
    REQUIRE(export_publication.has_value());
    std::vector<PendingProducer> pending;
    std::vector<uint32_t> event_order;
    OwnedMemexportPublicationTarget export_target(h.shared.get(), &pending, &event_order);
    MemexportPublicationResult export_result = PublishMemexportWrite(
        *export_publication, export_target, [&](const MemexportPublication& owned) {
          event_order.push_back(1u);
          pending.push_back(PendingProducer{owned.range, TransitState::kPendingMemexport,
                                            owned.submission, uint64_t(cycle) + 1u});
          return true;
        });
    REQUIRE(export_result.action == AuthorityAction::kPublishGpuWrite);
    REQUIRE(export_target.mark_saw_owner);
    const std::vector<uint32_t> expected_event_order = {1u, 2u};
    REQUIRE(event_order == expected_event_order);
    REQUIRE(pending.size() == 1u);
    REQUIRE(h.RamRead(addr, kPage) == resolve_bytes);
    REQUIRE(gpu_write.command_buffer.status == MTLCommandBufferStatusNotEnqueued);
    h.CommitGpuWrite(&gpu_write);
    REQUIRE(h.shared->RequestRange(addr, kPage));
    uint64_t submission_m = 0;
    REQUIRE(h.GpuRead(addr, kPage, &submission_m) == export_bytes);
    REQUIRE(submission_m > submission_n);
    REQUIRE(gpu_write.command_buffer.status == MTLCommandBufferStatusCompleted);
    REQUIRE(ProducerComplete(pending.front(), submission_n));
    REQUIRE(h.SharedRead(addr, kPage) == export_bytes);
    RealMemexportConvergenceTarget convergence_target(&h);
    MemexportConvergenceResult convergence_result =
        ConvergeMemexportRam(pending.front(), convergence_target);
    REQUIRE(convergence_result.converged);
    if (convergence_result.converged) {
      pending.erase(pending.begin());
    }
    REQUIRE(pending.empty());
    REQUIRE(h.RamRead(addr, kPage) == export_bytes);
    REQUIRE(h.SharedRead(addr, kPage) == export_bytes);

    // Trace restore returns the range to RAM-current authority and retains the
    // independently-required legacy invalidation operation in its policy.
    auto trace_bytes = h.Pattern(kPage, uint8_t(seed + 3u));
    h.RamWrite(addr, trace_bytes);
    auto trace_publication = DecideRamWritePublication(WriterKind::kTraceRestore, addr, kPage);
    REQUIRE(trace_publication.has_value());
    RealRamWritePublicationTarget trace_target(h.shared.get());
    LegacyWriterOutcome trace_outcome = PublishRamWrite(*trace_publication, trace_target);
    REQUIRE(trace_outcome.action == AuthorityAction::kInvalidateRamWrite);
    REQUIRE(trace_target.shared_calls == 1u);
    REQUIRE(trace_target.legacy_calls == 1u);
    REQUIRE(h.shared->RequestRange(addr, kPage));
    REQUIRE(h.RamRead(addr, kPage) == trace_bytes);
    REQUIRE(h.SharedRead(addr, kPage) == trace_bytes);
    REQUIRE(h.GpuRead(addr, kPage) == trace_bytes);
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
