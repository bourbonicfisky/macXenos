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
// callback thunk is exercised directly to prove the functional shared-memory
// invalidation used by guest CPU writes. Direct backend publications are
// covered at their explicit typed call sites. (A full command-processor drain
// in this harness would be artificial; the drain's
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

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPage = 4096;

// Deterministic byte pattern independent of any device. Used by the device-free
// boundary-clamp surface (which must run even with no Metal device) and mirrored
// by CoherenceHarness::Pattern for the on-device cells.
std::vector<uint8_t> SeededPattern(uint32_t bytes, uint8_t seed) {
  std::vector<uint8_t> v(bytes);
  for (uint32_t i = 0; i < bytes; ++i) {
    v[i] = uint8_t(seed + i * 7 + (i >> 8));
  }
  return v;
}

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
    return SeededPattern(bytes, seed);
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

// --- Task 6.1 factual four-writer / four-consumer matrix helpers -----------
//
// These route BOTH sides of every writer x consumer cell through the real
// production seams (DecideRamWritePublication/PublishRamWrite, the memexport
// preparation/publication/ConvergeMemexportRam ports, DecideConsumer, and the
// shared DrainSelectedProducers executor) against the real Memory +
// MetalSharedMemory + GpuRead/RamRead corridors. No production decision is
// reimplemented here: the helpers only supply stimulus and read back exact
// bytes, so each cell asserts the writer's own byte pattern for that consumer
// rather than inferring a cell from a shared mechanism.

// Executes writer W's real production publication of `bytes` over [addr, len).
// Guest CPU flows only through the physical-callback thunk; resolve and trace
// restore flow through the typed PublishRamWrite seam (with the writer's exact
// legacy-invalidation policy asserted); memexport prepares the destination,
// encodes submission N, registers its pending record BEFORE RangeWrittenByGpu,
// then commits and awaits N so the record is complete. A memexport leaves its
// pending record in `pending` and returns N; the RAM writers leave none and
// return 0 (no completed submission to await).
uint64_t ApplyWriter(CoherenceHarness& h, WriterKind writer, uint32_t addr,
                     uint32_t len, const std::vector<uint8_t>& bytes,
                     std::vector<PendingProducer>* pending) {
  switch (writer) {
    case WriterKind::kGuestCpu:
      h.RamWrite(addr, bytes);
      h.shared->InvokePhysicalInvalidation(addr, len, true);
      return 0u;
    case WriterKind::kResolve:
    case WriterKind::kTraceRestore: {
      h.RamWrite(addr, bytes);
      auto publication = DecideRamWritePublication(writer, addr, len);
      REQUIRE(publication.has_value());
      RealRamWritePublicationTarget target(h.shared.get());
      RamWritePublicationOutcome outcome = PublishRamWrite(*publication, target);
      REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
      REQUIRE(outcome.shared_invalidation_executed);
      REQUIRE(target.shared_calls == 1u);
      REQUIRE(target.legacy_calls ==
              (writer == WriterKind::kTraceRestore ? 1u : 0u));
      return 0u;
    }
    case WriterKind::kMemexport:
    default: {
      auto preparation = DecideMemexportDestinationPreparation(addr, len);
      REQUIRE(preparation.has_value());
      RealMemexportPreparationTarget preparation_target(h.shared.get());
      REQUIRE(PrepareMemexportDestination(*preparation, preparation_target));
      auto gpu_write = h.EncodeGpuWrite(addr, bytes);
      uint64_t submission_n = gpu_write.submission;
      std::vector<uint32_t> event_order;
      OwnedMemexportPublicationTarget publication_target(h.shared.get(), pending,
                                                         &event_order);
      auto publication = DecideMemexportPublication(addr, len, submission_n);
      REQUIRE(publication.has_value());
      MemexportPublicationResult result = PublishMemexportWrite(
          *publication, publication_target,
          [&](const MemexportPublication& owned) {
            event_order.push_back(1u);
            pending->push_back({owned.range, TransitState::kPendingMemexport,
                                owned.submission, 1u});
            return true;
          });
      REQUIRE(result.marked);
      REQUIRE(publication_target.mark_saw_owner);
      const std::vector<uint32_t> expected_event_order = {1u, 2u};
      REQUIRE(event_order == expected_event_order);
      h.CommitGpuWrite(&gpu_write);
      [gpu_write.command_buffer waitUntilCompleted];
      REQUIRE(gpu_write.command_buffer.status ==
              MTLCommandBufferStatusCompleted);
      return submission_n;
    }
  }
}

struct CellResult {
  AuthorityAction action = AuthorityAction::kNoDependency;
  size_t selected = 0;
  std::vector<uint8_t> observed;
};

// Applies writer W, then observes [c_addr, c_len) through consumer C's real
// corridor, routed through the real DecideConsumer + DrainSelectedProducers
// seam. `request_addr` is the address the consumer presents (optionally the
// bit-29 physical alias); it must normalize to `c_addr`. Any pending memexport
// record is converged through the real ConvergeMemexportRam port and retired by
// its unique authority sequence; the RAM writers leave nothing to drain. Shared-
// buffer consumers observe via a GPU compute read of the shared buffer, RAM
// consumers via a direct guest-RAM read.
CellResult WriteAndConsume(CoherenceHarness& h, WriterKind writer,
                           ConsumerKind consumer, uint32_t w_addr,
                           uint32_t w_len, const std::vector<uint8_t>& bytes,
                           uint32_t c_addr, uint32_t c_len,
                           uint32_t request_addr) {
  std::vector<PendingProducer> pending;
  uint64_t completed = ApplyWriter(h, writer, w_addr, w_len, bytes, &pending);

  const uint32_t physical = xenos::CpuToGpu(request_addr);
  REQUIRE(physical == c_addr);
  GuestRangeSet requested;
  requested.Add(GuestRange::FromExactBaseLength(physical, c_len));
  ConsumerDecision decision =
      DecideConsumer(consumer, requested, pending, completed);
  REQUIRE(decision.revalidate_gpu_bytes == ConsumerReadsSharedBuffer(consumer));

  RealMemexportConvergenceTarget convergence_target(&h);
  auto retire_by_sequence = [&](uint64_t sequence) {
    for (auto it = pending.begin(); it != pending.end(); ++it) {
      if (it->sequence == sequence) {
        pending.erase(it);
        return true;
      }
    }
    return false;
  };
  ProducerDrainOutcome drain = DrainSelectedProducers(
      decision, [&](const PendingProducer& selected) {
        ProducerDrainOutcome one;
        MemexportConvergenceResult converged =
            ConvergeMemexportRam(selected, convergence_target);
        REQUIRE(converged.converged);
        REQUIRE(converged.action == AuthorityAction::kConvergeRamFromBuffer);
        one.memexport_records = 1;
        one.drained_records = 1;
        REQUIRE(retire_by_sequence(selected.sequence));
        return one;
      });
  REQUIRE(drain.drained_records == decision.selected.size());

  CellResult result;
  result.action = decision.action;
  result.selected = decision.selected.size();
  if (ConsumerReadsSharedBuffer(consumer)) {
    REQUIRE(h.shared->RequestRange(c_addr, c_len));
    result.observed = h.GpuRead(c_addr, c_len);
  } else {
    result.observed = h.RamRead(c_addr, c_len);
  }
  return result;
}

constexpr WriterKind kAllWriters[] = {
    WriterKind::kGuestCpu, WriterKind::kResolve, WriterKind::kMemexport,
    WriterKind::kTraceRestore};
constexpr ConsumerKind kAllConsumers[] = {
    ConsumerKind::kTexture, ConsumerKind::kVertex, ConsumerKind::kWait,
    ConsumerKind::kHostCpuRead};

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

TEST_CASE("SM coherence: watch-only admission uploads nothing and validates "
          "nothing",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  // Never-requested page: the shared buffer cannot already hold its bytes.
  const uint32_t addr = 0x0A200000;
  auto bytes = h.Pattern(kPage, 0x44);
  h.RamWrite(addr, bytes);
  // Admission works in whole SYSTEM pages (16 KiB here) and requires every
  // covered 4 KiB guest page to be mapped, so commit the containing system
  // page — the 4 KiB RamWrite above only backs one quarter of it.
  h.EnsureAllocated(addr, xe::memory::page_size());
  // Mapped guest memory is admissible for watching...
  REQUIRE(h.shared->RequestWatchOnlyRange(addr, kPage) > 0);
  // ...but admission must NOT upload: the buffer still lacks the RAM bytes.
  REQUIRE(h.GpuRead(addr, kPage) != bytes);
  // ...and must NOT mark the pages valid, or this RequestRange would skip the
  // upload the shared-buffer consumer genuinely needs and read stale bytes.
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == bytes);
}

TEST_CASE("SM coherence: watch-only admission leaves an already-valid range "
          "intact",
          "[l1][metal][sharedmem]") {
  REQUIRE_DEVICE();
  auto& h = H();
  const uint32_t addr = 0x0A280000;
  auto bytes = h.Pattern(kPage, 0x55);
  h.RamWrite(addr, bytes);
  h.EnsureAllocated(addr, xe::memory::page_size());
  REQUIRE(h.shared->RequestRange(addr, kPage));
  REQUIRE(h.GpuRead(addr, kPage) == bytes);
  // Watch-only admission over a valid range must neither invalidate nor
  // re-upload — it only widens which pages notify on a CPU write.
  REQUIRE(h.shared->RequestWatchOnlyRange(addr, kPage) > 0);
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
    RamWritePublicationOutcome outcome = PublishRamWrite(*publication, target);

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

TEST_CASE("SM coherence: aliased same-frame consumers observe the newest "
          "selected producer bytes",
          "[l1][metal][sharedmem][authority][same-frame][bytes][alias]") {
  REQUIRE_DEVICE();
  auto& h = H();
  struct ConsumerCase {
    const char* name;
    ConsumerKind kind;
    uint32_t address;
  } cases[] = {
      {"texture shared-byte corridor", ConsumerKind::kTexture, 0x12400000u},
      {"vertex shared-byte corridor", ConsumerKind::kVertex, 0x12500000u},
      {"wait RAM corridor", ConsumerKind::kWait, 0x12600000u},
      {"host CPU-read RAM corridor", ConsumerKind::kHostCpuRead, 0x12700000u},
  };

  for (uint8_t case_index = 0; case_index < std::size(cases);
       ++case_index) {
    const ConsumerCase& consumer_case = cases[case_index];
    INFO(consumer_case.name);
    const uint32_t address = consumer_case.address;
    const uint32_t disjoint_address = address + 0x8000u;
    auto stale = h.Pattern(kPage, uint8_t(0x90u + case_index * 3u));
    auto resolved = h.Pattern(kPage, uint8_t(0x91u + case_index * 3u));
    auto exported = h.Pattern(kPage, uint8_t(0x92u + case_index * 3u));
    h.RamWrite(address, stale);
    REQUIRE(h.shared->RequestRange(address, kPage));
    REQUIRE(h.GpuRead(address, kPage) == stale);

    // The newer memexport is encoded and owns its pending range before the
    // command buffer is committed. The older resolve and one disjoint record
    // model the real cross-kind pending snapshot seen by DecideConsumer.
    auto preparation =
        DecideMemexportDestinationPreparation(address, kPage);
    REQUIRE(preparation.has_value());
    RealMemexportPreparationTarget preparation_target(h.shared.get());
    REQUIRE(PrepareMemexportDestination(*preparation, preparation_target));
    auto gpu_write = h.EncodeGpuWrite(address, exported);
    std::vector<PendingProducer> pending = {
        {*GuestRange::FromBaseLength(address, kPage),
         TransitState::kPendingResolve, gpu_write.submission, 1u},
    };
    std::vector<uint32_t> event_order;
    OwnedMemexportPublicationTarget export_target(h.shared.get(), &pending,
                                                   &event_order);
    auto export_publication =
        DecideMemexportPublication(address, kPage, gpu_write.submission);
    REQUIRE(export_publication.has_value());
    MemexportPublicationResult export_result = PublishMemexportWrite(
        *export_publication, export_target,
        [&](const MemexportPublication& owned) {
          event_order.push_back(1u);
          pending.push_back({owned.range, TransitState::kPendingMemexport,
                             owned.submission, 2u});
          return true;
        });
    REQUIRE(export_result.registered);
    REQUIRE(export_result.marked);
    REQUIRE(export_target.mark_saw_owner);
    pending.push_back({*GuestRange::FromBaseLength(disjoint_address, kPage),
                       TransitState::kPendingResolve, gpu_write.submission,
                       3u});

    // PM4 and fetch descriptors may carry the bit-29 physical alias. The
    // authority receives the normalized physical range, just as the routed
    // production call sites do.
    const uint32_t aliased_address = address | 0x20000000u;
    const uint32_t physical_address = xenos::CpuToGpu(aliased_address);
    REQUIRE(physical_address == address);
    GuestRangeSet requested;
    requested.Add(
        GuestRange::FromExactBaseLength(physical_address, kPage));
    ConsumerDecision decision = DecideConsumer(
        consumer_case.kind, requested, pending,
        /*completed_submission=*/0u);
    REQUIRE(decision.action == AuthorityAction::kAwaitProducer);
    REQUIRE(decision.await);
    REQUIRE(decision.drain);
    REQUIRE(decision.selected.size() == 2u);
    REQUIRE(decision.selected[0].state == TransitState::kPendingResolve);
    REQUIRE(decision.selected[1].state == TransitState::kPendingMemexport);
    REQUIRE(decision.selected[0].sequence < decision.selected[1].sequence);
    REQUIRE(decision.revalidate_gpu_bytes ==
            ConsumerReadsSharedBuffer(consumer_case.kind));

    h.CommitGpuWrite(&gpu_write);
    [gpu_write.command_buffer waitUntilCompleted];
    REQUIRE(gpu_write.command_buffer.status ==
            MTLCommandBufferStatusCompleted);
    REQUIRE(ProducerComplete(decision.selected[0], gpu_write.submission));
    REQUIRE(ProducerComplete(decision.selected[1], gpu_write.submission));

    // Drain the selected records through the SAME shared production executor
    // (DrainSelectedProducers) used by MetalCommandProcessor, rather than
    // replaying the selection order by hand. The callback drives the real
    // publication / convergence seams for one record and, on success, retires
    // the matching LOCAL pending record by its unique authority sequence. The
    // disjoint record is never selected, so it remains without any manual erase.
    RealMemexportConvergenceTarget convergence_target(&h);
    auto retire_by_sequence = [&](uint64_t sequence) {
      for (auto it = pending.begin(); it != pending.end(); ++it) {
        if (it->sequence == sequence) {
          pending.erase(it);
          return true;
        }
      }
      return false;
    };
    ProducerDrainOutcome drain = DrainSelectedProducers(
        decision, [&](const PendingProducer& selected) {
          ProducerDrainOutcome one;
          if (selected.state == TransitState::kPendingResolve) {
            // The resolve's checked scatter deposits its bytes to RAM, then the
            // typed seam publishes the exact extent.
            h.RamWrite(selected.range.start(), resolved);
            auto resolve_publication = DecideRamWritePublication(
                WriterKind::kResolve, selected.range.start(),
                selected.range.length());
            REQUIRE(resolve_publication.has_value());
            RealRamWritePublicationTarget resolve_target(h.shared.get());
            RamWritePublicationOutcome resolve_outcome =
                PublishRamWrite(*resolve_publication, resolve_target);
            REQUIRE(resolve_outcome.action ==
                    AuthorityAction::kInvalidateRamWrite);
            one.resolve_records = 1;
          } else {
            MemexportConvergenceResult convergence_result =
                ConvergeMemexportRam(selected, convergence_target);
            REQUIRE(convergence_result.converged);
            REQUIRE(convergence_result.action ==
                    AuthorityAction::kConvergeRamFromBuffer);
            one.memexport_records = 1;
            one.invalidated_entries = convergence_result.invalidated_entries;
          }
          one.drained_records = 1;
          REQUIRE(retire_by_sequence(selected.sequence));
          return one;
        });
    REQUIRE(drain.drained_records == decision.selected.size());
    REQUIRE(drain.resolve_records == 1u);
    REQUIRE(drain.memexport_records == 1u);
    REQUIRE(h.RamRead(address, kPage) == exported);
    REQUIRE(h.SharedRead(address, kPage) == exported);

    if (ConsumerReadsSharedBuffer(consumer_case.kind)) {
      REQUIRE(h.shared->RequestRange(address, kPage));
      REQUIRE(h.GpuRead(address, kPage) == exported);
    } else {
      REQUIRE(h.RamRead(address, kPage) == exported);
    }

    // Only the selected records retired via the shared executor; the disjoint
    // record was never selected and remains pending without any manual erase.
    REQUIRE(pending.size() == 1u);
    REQUIRE(pending.front().range.start() == disjoint_address);
  }
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

    // Guest CPU: the real physical callback thunk publishes the writer.
    auto cpu_bytes = h.Pattern(kPage, seed);
    h.RamWrite(addr, cpu_bytes);
    h.shared->InvokePhysicalInvalidation(addr, kPage, true);
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
    RamWritePublicationOutcome resolve_outcome = PublishRamWrite(*resolve_publication, resolve_target);
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
    RamWritePublicationOutcome trace_outcome = PublishRamWrite(*trace_publication, trace_target);
    REQUIRE(trace_outcome.action == AuthorityAction::kInvalidateRamWrite);
    REQUIRE(trace_target.shared_calls == 1u);
    REQUIRE(trace_target.legacy_calls == 1u);
    REQUIRE(h.shared->RequestRange(addr, kPage));
    REQUIRE(h.RamRead(addr, kPage) == trace_bytes);
    REQUIRE(h.SharedRead(addr, kPage) == trace_bytes);
    REQUIRE(h.GpuRead(addr, kPage) == trace_bytes);
  }
}

// --- Task 6.1: the complete four-writer x four-consumer factual matrix ------
//
// The tests above prove individual mechanisms; task 6.1 asks the whole grid
// directly. The prior evidence was insufficient on its own: the aliased same-
// frame case masks resolve under a newer memexport (resolve's bytes are never
// asserted for any consumer), and the guest-CPU / trace-restore proofs read
// through generic GpuRead/RamRead rather than each of the four consumer
// corridors. The following cases assert, for EVERY writer x consumer cell, that
// the consumer observes the writer's own exact byte pattern through that
// consumer's real corridor. No cell is credited from a shared mechanism.

TEST_CASE("SM coherence: four-writer x four-consumer exact byte matrix "
          "(bit-29 aliased request)",
          "[l1][metal][sharedmem][authority][matrix][bytes]") {
  REQUIRE_DEVICE();
  auto& h = H();
  uint32_t addr = 0x14000000u;
  for (uint8_t wi = 0; wi < std::size(kAllWriters); ++wi) {
    for (uint8_t ci = 0; ci < std::size(kAllConsumers); ++ci) {
      const WriterKind writer = kAllWriters[wi];
      const ConsumerKind consumer = kAllConsumers[ci];
      const uint32_t cell = addr;
      addr += 0x00100000u;  // 1 MiB per cell keeps every cell on its own pages.
      INFO(WriterKindName(writer) << " -> " << ConsumerKindName(consumer));

      // Baseline: distinct stale bytes, validated. A cell that failed to publish
      // (or that masked this writer under another) would read back this stale
      // pattern and fail the exact comparison below.
      auto stale = h.Pattern(kPage, uint8_t(0x10u + wi * 16u + ci));
      auto written = h.Pattern(kPage, uint8_t(0x80u + wi * 16u + ci));
      h.RamWrite(cell, stale);
      REQUIRE(h.shared->RequestRange(cell, kPage));
      REQUIRE(h.GpuRead(cell, kPage) == stale);

      // Consumer presents the range through the bit-29 physical alias; the
      // authority receives the normalized physical range (aliased coverage).
      const uint32_t aliased = cell | 0x20000000u;
      auto writer_range = GuestRange::FromBaseLength(cell, kPage);
      auto consumer_range =
          GuestRange::FromBaseLength(xenos::CpuToGpu(aliased), kPage);
      REQUIRE(writer_range.has_value());
      REQUIRE(consumer_range.has_value());
      REQUIRE(writer_range->Classify(*consumer_range) ==
              GuestRange::Relation::kExact);
      CellResult r = WriteAndConsume(h, writer, consumer, cell, kPage, written,
                                     cell, kPage, aliased);
      REQUIRE(r.observed == written);
      REQUIRE(h.RamRead(cell, kPage) == written);
      if (writer == WriterKind::kMemexport) {
        REQUIRE(r.action == AuthorityAction::kDrainMatchingRecords);
        REQUIRE(r.selected == 1u);
      } else {
        REQUIRE(r.action == AuthorityAction::kNoDependency);
        REQUIRE(r.selected == 0u);
      }
    }
  }
}

TEST_CASE("SM coherence: four-writer x four-consumer partial and repeated "
          "overlap byte matrix",
          "[l1][metal][sharedmem][authority][matrix][bytes]") {
  REQUIRE_DEVICE();
  auto& h = H();
  uint32_t addr = 0x15000000u;
  const uint32_t w_len = kPage * 4u;     // writer covers [base, base+4p).
  const uint32_t union_len = kPage * 5u; // includes the consumer-only tail.
  for (uint8_t wi = 0; wi < std::size(kAllWriters); ++wi) {
    for (uint8_t ci = 0; ci < std::size(kAllConsumers); ++ci) {
      const WriterKind writer = kAllWriters[wi];
      const ConsumerKind consumer = kAllConsumers[ci];
      const uint32_t base = addr;
      addr += 0x00100000u;
      const uint32_t sub = base + 3u * kPage;
      const uint32_t consumer_len = 2u * kPage;
      INFO(WriterKindName(writer) << " -> " << ConsumerKindName(consumer));

      // ---- partial: writer=[base,base+4p), consumer=[base+3p,base+5p).
      // They share one page, while the consumer's final page is a stale canary.
      // Neither range contains the other, so this is true kPartial overlap.
      auto writer_range = GuestRange::FromBaseLength(base, w_len);
      auto consumer_range = GuestRange::FromBaseLength(sub, consumer_len);
      REQUIRE(writer_range.has_value());
      REQUIRE(consumer_range.has_value());
      REQUIRE(writer_range->Classify(*consumer_range) ==
              GuestRange::Relation::kPartial);
      auto stale = h.Pattern(union_len, uint8_t(0x20u + wi * 16u + ci));
      h.RamWrite(base, stale);
      REQUIRE(h.shared->RequestRange(base, union_len));
      auto written = h.Pattern(w_len, uint8_t(0xA0u + wi * 16u + ci));
      CellResult r = WriteAndConsume(h, writer, consumer, base, w_len, written,
                                     sub, consumer_len, sub);
      std::vector<uint8_t> expected(written.begin() + 3u * kPage,
                                    written.end());
      expected.insert(expected.end(), stale.begin() + 4u * kPage, stale.end());
      REQUIRE(r.observed == expected);
      REQUIRE(h.RamRead(base, w_len) == written);
      REQUIRE(h.RamRead(base + w_len, kPage) ==
              std::vector<uint8_t>(stale.begin() + w_len, stale.end()));
      if (writer == WriterKind::kMemexport) {
        REQUIRE(r.action == AuthorityAction::kDrainMatchingRecords);
        REQUIRE(r.selected == 1u);
      } else {
        REQUIRE(r.action == AuthorityAction::kNoDependency);
      }

      // ---- repeated: the same writer publishes again over the same range;
      // the overlap must land on the NEWEST bytes while the consumer-only tail
      // remains the original stale canary.
      auto rewritten = h.Pattern(w_len, uint8_t(0xC0u + wi * 16u + ci));
      CellResult r2 = WriteAndConsume(h, writer, consumer, base, w_len,
                                      rewritten, sub, consumer_len, sub);
      std::vector<uint8_t> expected2(rewritten.begin() + 3u * kPage,
                                     rewritten.end());
      expected2.insert(expected2.end(), stale.begin() + 4u * kPage,
                       stale.end());
      REQUIRE(r2.observed == expected2);
      REQUIRE(h.RamRead(base, w_len) == rewritten);
      REQUIRE(h.RamRead(base + w_len, kPage) ==
              std::vector<uint8_t>(stale.begin() + w_len, stale.end()));
    }
  }
}

TEST_CASE("SM coherence: four-writer x four-consumer adjacent and disjoint "
          "non-dependency byte matrix",
          "[l1][metal][sharedmem][authority][matrix][bytes]") {
  REQUIRE_DEVICE();
  auto& h = H();
  uint32_t addr = 0x16000000u;
  for (uint8_t wi = 0; wi < std::size(kAllWriters); ++wi) {
    for (uint8_t ci = 0; ci < std::size(kAllConsumers); ++ci) {
      for (uint8_t variant = 0; variant < 2; ++variant) {
        const WriterKind writer = kAllWriters[wi];
        const ConsumerKind consumer = kAllConsumers[ci];
        const bool disjoint = variant != 0;
        const uint32_t consumer_addr = addr;
        addr += 0x00080000u;  // 512 KiB per (cell, variant).
        // Adjacent: writer's range begins exactly where the consumer's ends
        // (half-open touch, zero shared bytes). Disjoint: a page gap between
        // them. Neither is a dependency; the writer's bytes must not bleed into
        // the consumer range.
        const uint32_t writer_addr =
            disjoint ? consumer_addr + kPage * 4u : consumer_addr + kPage;
        auto writer_range = GuestRange::FromBaseLength(writer_addr, kPage);
        auto consumer_range =
            GuestRange::FromBaseLength(consumer_addr, kPage);
        REQUIRE(writer_range.has_value());
        REQUIRE(consumer_range.has_value());
        REQUIRE(consumer_range->Classify(*writer_range) ==
                (disjoint ? GuestRange::Relation::kDisjoint
                          : GuestRange::Relation::kAdjacent));
        INFO(WriterKindName(writer)
             << " -> " << ConsumerKindName(consumer)
             << (disjoint ? " (disjoint)" : " (adjacent)"));

        auto baseline = h.Pattern(kPage, uint8_t(0x30u + wi * 16u + ci));
        auto neighbor = h.Pattern(kPage, uint8_t(0xB0u + wi * 16u + ci));
        auto neighbor_ram =
            h.Pattern(kPage, uint8_t(0x50u + wi * 16u + ci));
        h.RamWrite(consumer_addr, baseline);
        REQUIRE(h.shared->RequestRange(consumer_addr, kPage));
        REQUIRE(h.GpuRead(consumer_addr, kPage) == baseline);
        // Memexport preparation requests the destination from guest RAM before
        // the GPU write. Allocate and seed it just as a real guest allocation
        // would; RAM-current writers overwrite this seed in ApplyWriter.
        h.RamWrite(writer_addr, neighbor_ram);

        std::vector<PendingProducer> pending;
        ApplyWriter(h, writer, writer_addr, kPage, neighbor, &pending);

        // The consumer's own range is decided against whatever pending record
        // the writer left. Completed high-water is maxed so that if a record
        // were wrongly selected it would drain (not merely await) -- yet a non-
        // overlapping request must select nothing regardless.
        GuestRangeSet requested;
        requested.Add(GuestRange::FromExactBaseLength(consumer_addr, kPage));
        ConsumerDecision decision = DecideConsumer(
            consumer, requested, pending, /*completed_submission=*/~0ull);
        REQUIRE(decision.action == AuthorityAction::kNoDependency);
        REQUIRE(decision.selected.empty());
        REQUIRE(pending.size() ==
                (writer == WriterKind::kMemexport ? 1u : 0u));

        // The consumer still observes exactly its own baseline: the adjacent /
        // disjoint writer did not bleed into the consumer range.
        REQUIRE(h.RamRead(consumer_addr, kPage) == baseline);
        if (ConsumerReadsSharedBuffer(consumer)) {
          REQUIRE(h.shared->RequestRange(consumer_addr, kPage));
          REQUIRE(h.GpuRead(consumer_addr, kPage) == baseline);
        } else {
          REQUIRE(h.RamRead(consumer_addr, kPage) == baseline);
        }
        // And the writer genuinely wrote its neighbor (so the non-bleed above is
        // a real separation, not a writer that silently did nothing). Memexport
        // lands in the shared buffer; the RAM writers land in guest RAM.
        if (writer == WriterKind::kMemexport) {
          REQUIRE(h.SharedRead(writer_addr, kPage) == neighbor);
          REQUIRE(h.RamRead(writer_addr, kPage) == neighbor_ram);
        } else {
          REQUIRE(h.RamRead(writer_addr, kPage) == neighbor);
        }
      }
    }
  }
}

// --- Task 6.1 boundary cell: device-free RAM + shared-buffer world ----------
//
// The literal 512 MiB top cannot be allocated on the real device (the topmost
// guest pages are heap-reserved and RequestRange refuses them), so the boundary
// cell cannot ride the on-device CoherenceHarness. This sparse in-memory world
// models only the byte transport and validity for a small window abutting the
// domain top. It owns NO authority policy: range normalization, selection,
// action, and ordering come from the production decisions and ports below. The
// existing high-address Metal test separately proves the real shared-memory and
// GPU-compute transport. Out-of-domain model access is refused, never wrapped.
class BoundaryByteWorld {
 public:
  BoundaryByteWorld(uint32_t base, uint32_t size)
      : base_(base), ram_(size), shared_(size), valid_(size, false) {}

  bool Contains(uint32_t addr, uint32_t length) const {
    return addr >= base_ && uint64_t(addr - base_) + length <= ram_.size();
  }

  // Host-view guest-RAM deposit: the RAM-current writer stimulus (resolve/trace
  // scatter, guest CPU store) and the memexport destination seed.
  void RamWrite(uint32_t addr, const std::vector<uint8_t>& bytes) {
    REQUIRE(Contains(addr, uint32_t(bytes.size())));
    std::copy(bytes.begin(), bytes.end(), ram_.begin() + (addr - base_));
  }
  std::vector<uint8_t> RamRead(uint32_t addr, uint32_t length) const {
    REQUIRE(Contains(addr, length));
    const auto first = ram_.begin() + (addr - base_);
    return std::vector<uint8_t>(first, first + length);
  }

  // A completed GPU deposit into the shared buffer (the memexport write): lands
  // in the buffer and leaves the range valid, as a finished blit would.
  void SharedWrite(uint32_t addr, const std::vector<uint8_t>& bytes) {
    REQUIRE(Contains(addr, uint32_t(bytes.size())));
    std::copy(bytes.begin(), bytes.end(), shared_.begin() + (addr - base_));
    SetValid(addr, uint32_t(bytes.size()), true);
  }
  std::vector<uint8_t> SharedRead(uint32_t addr, uint32_t length) const {
    REQUIRE(Contains(addr, length));
    const auto first = shared_.begin() + (addr - base_);
    return std::vector<uint8_t>(first, first + length);
  }

  // Mirrors SharedMemory::MemoryInvalidationCallback: marks the range invalid so
  // the next RequestRange re-uploads. An out-of-domain length is ignored (the
  // real callback clamps), never wrapped into low pages.
  void InvalidateShared(uint32_t addr, uint32_t length) {
    if (!Contains(addr, length)) {
      return;
    }
    SetValid(addr, length, false);
  }

  // Mirrors SharedMemory::RequestRange: re-uploads any invalid bytes from RAM,
  // marks them valid, and refuses an out-of-domain request rather than clamp it.
  bool RequestRange(uint32_t addr, uint32_t length) {
    if (!Contains(addr, length)) {
      return false;
    }
    for (uint32_t i = 0; i < length; ++i) {
      uint32_t off = addr - base_ + i;
      if (!valid_[off]) {
        shared_[off] = ram_[off];
        valid_[off] = true;
      }
    }
    return true;
  }

  // The memexport convergence copy: shared buffer -> guest RAM, never touching
  // the buffer it read from.
  bool CopyBufferToRam(uint32_t addr, uint32_t length) {
    if (!Contains(addr, length)) {
      return false;
    }
    for (uint32_t i = 0; i < length; ++i) {
      ram_[addr - base_ + i] = shared_[addr - base_ + i];
    }
    return true;
  }

 private:
  void SetValid(uint32_t addr, uint32_t length, bool v) {
    for (uint32_t i = 0; i < length; ++i) {
      valid_[addr - base_ + i] = v;
    }
  }

  uint32_t base_;
  std::vector<uint8_t> ram_;
  std::vector<uint8_t> shared_;
  std::vector<bool> valid_;
};

// Device-free seam targets for the boundary cell. Each routes a typed port to
// the world's modeled RAM/shared storage and owns no authority state.
class BoundaryRamWriteTarget final : public RamWritePublicationTarget {
 public:
  explicit BoundaryRamWriteTarget(BoundaryByteWorld* world) : world_(world) {}
  bool InvalidateSharedPages(uint32_t base, uint32_t length, bool) override {
    ++shared_calls;
    world_->InvalidateShared(base, length);
    return true;
  }
  bool InvalidateLegacyTextures(uint32_t, uint32_t,
                                uint32_t* invalidated_entries) override {
    ++legacy_calls;
    *invalidated_entries = 0;
    return true;
  }
  uint32_t shared_calls = 0;
  uint32_t legacy_calls = 0;

 private:
  BoundaryByteWorld* world_;
};

class BoundaryPreparationTarget final : public MemexportPreparationTarget {
 public:
  explicit BoundaryPreparationTarget(BoundaryByteWorld* world) : world_(world) {}
  bool RequestDestinationRange(uint32_t base, uint32_t length) override {
    return world_->RequestRange(base, length);
  }

 private:
  BoundaryByteWorld* world_;
};

class BoundaryPublicationTarget final : public MemexportPublicationTarget {
 public:
  bool CanMarkRangeWrittenByGpu() const override { return true; }
  void MarkRangeWrittenByGpu(uint32_t base, uint32_t length) override {
    ++marks;
    marked_base = base;
    marked_length = length;
  }
  uint32_t marks = 0;
  uint32_t marked_base = 0;
  uint32_t marked_length = 0;
};

class BoundaryConvergenceTarget final : public MemexportConvergenceTarget {
 public:
  explicit BoundaryConvergenceTarget(BoundaryByteWorld* world) : world_(world) {}
  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries) override {
    *invalidated_entries = 0;
    return world_->CopyBufferToRam(base, length);
  }

 private:
  BoundaryByteWorld* world_;
};

// Publishes writer W's clamped `payload` over [start, start+requested_len) through
// W's real production seam against the boundary world. Guest CPU is published by
// the physical-memory callback (modeled as a direct shared invalidation), then
// PublishRamWrite is exercised to confirm its no-op policy adds nothing; resolve
// and trace restore flow through the typed PublishRamWrite seam with each writer's
// exact legacy-invalidation policy asserted; memexport prepares the destination,
// deposits its submission-owned bytes into the shared buffer, then registers its
// pending record BEFORE marking the range GPU-written. Memexport leaves its
// pending record in `pending` and returns submission 1; the RAM writers leave
// none and return 0. Every seam's range is asserted to clamp to `clamped_len`.
uint64_t ApplyBoundaryWriter(BoundaryByteWorld& world, WriterKind writer,
                             uint32_t start, uint32_t requested_len,
                             uint32_t clamped_len,
                             const std::vector<uint8_t>& payload,
                             std::vector<PendingProducer>* pending) {
  switch (writer) {
    case WriterKind::kGuestCpu: {
      world.RamWrite(start, payload);
      world.InvalidateShared(start, clamped_len);  // the physical callback.
      auto publication = DecideRamWritePublication(writer, start, requested_len);
      REQUIRE(publication.has_value());
      REQUIRE(publication->range.start() == start);
      REQUIRE(publication->range.length() == clamped_len);  // clamped, no wrap.
      BoundaryRamWriteTarget target(&world);
      RamWritePublicationOutcome outcome = PublishRamWrite(*publication, target);
      REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
      // Guest CPU policy: the seam re-publishes nothing (the callback already did).
      REQUIRE(target.shared_calls == 0u);
      REQUIRE(target.legacy_calls == 0u);
      REQUIRE_FALSE(outcome.shared_invalidation_executed);
      return 0u;
    }
    case WriterKind::kResolve:
    case WriterKind::kTraceRestore: {
      world.RamWrite(start, payload);
      auto publication = DecideRamWritePublication(writer, start, requested_len);
      REQUIRE(publication.has_value());
      REQUIRE(publication->range.start() == start);
      REQUIRE(publication->range.length() == clamped_len);  // clamped, no wrap.
      BoundaryRamWriteTarget target(&world);
      RamWritePublicationOutcome outcome = PublishRamWrite(*publication, target);
      REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
      REQUIRE(outcome.shared_invalidation_executed);
      REQUIRE(target.shared_calls == 1u);
      REQUIRE(target.legacy_calls ==
              (writer == WriterKind::kTraceRestore ? 1u : 0u));
      return 0u;
    }
    case WriterKind::kMemexport:
    default: {
      auto preparation =
          DecideMemexportDestinationPreparation(start, requested_len);
      REQUIRE(preparation.has_value());
      REQUIRE(preparation->range.length() == clamped_len);  // clamped, no wrap.
      BoundaryPreparationTarget preparation_target(&world);
      REQUIRE(PrepareMemexportDestination(*preparation, preparation_target));
      world.SharedWrite(start, payload);  // the submission-owned GPU write.
      auto publication = DecideMemexportPublication(start, requested_len, 1u);
      REQUIRE(publication.has_value());
      REQUIRE(publication->range.start() == start);
      REQUIRE(publication->range.length() == clamped_len);
      BoundaryPublicationTarget publication_target;
      MemexportPublicationResult result = PublishMemexportWrite(
          *publication, publication_target,
          [&](const MemexportPublication& owned) {
            pending->push_back({owned.range, TransitState::kPendingMemexport,
                                owned.submission, 1u});
            return true;
          });
      REQUIRE(result.marked);
      REQUIRE(result.action == AuthorityAction::kPublishGpuWrite);
      REQUIRE(publication_target.marks == 1u);
      REQUIRE(publication_target.marked_length == clamped_len);
      return 1u;
    }
  }
}

TEST_CASE("Authority matrix: four-writer x four-consumer 512 MiB "
          "boundary-clamped "
          "byte surface (device-free, paired with the real high-boundary test)",
          "[l0][authority][matrix][boundary][bytes]") {
  // The literal 512 MiB top cannot be allocated on the real device: the topmost
  // guest pages are heap-reserved and RequestRange refuses them (see "SM
  // coherence: physical-memory boundary is checked", which pairs a real GPU read
  // at a high allocatable address with an overshooting invalidation length).
  // This device-free surface completes the boundary cell for EVERY writer x
  // consumer by driving normalization, publication, selection, and convergence
  // through the production typed decisions and ports used by the other matrix
  // tests (DecideRamWritePublication/PublishRamWrite, the memexport ports,
  // DecideConsumer, and DrainSelectedProducers). The sparse world supplies only
  // the otherwise-unbackable literal-tail byte transport. GuestRange clamps the
  // overshoot to kGuestPhysicalSize, and the modeled shared/RAM storage class
  // observes exactly the in-domain clamped extent, never a wrapped or out-of-
  // domain byte. Actual Metal transport is the paired high-boundary test above;
  // no device is required for this literal-tail seam proof.
  const uint32_t kDomain = uint32_t(GuestRange::kGuestPhysicalSize);
  const uint32_t window = kPage * 8u;
  const uint32_t base_lo = kDomain - window;  // world spans [base_lo, kDomain).
  const uint32_t start = kDomain - kPage;     // last in-domain page.
  const uint32_t requested_len = kPage * 4u;  // three-page overshoot past top.
  for (uint8_t wi = 0; wi < std::size(kAllWriters); ++wi) {
    for (uint8_t ci = 0; ci < std::size(kAllConsumers); ++ci) {
      const WriterKind writer = kAllWriters[wi];
      const ConsumerKind consumer = kAllConsumers[ci];
      INFO(WriterKindName(writer) << " -> " << ConsumerKindName(consumer));

      // The seam's factory clamps the overshoot to the single in-domain page.
      auto w_range = GuestRange::FromBaseLength(start, requested_len);
      REQUIRE(w_range.has_value());
      REQUIRE(w_range->start() == start);
      REQUIRE(w_range->end() == kDomain);   // clamped end, no 32-bit wrap.
      REQUIRE(w_range->length() == kPage);  // only the in-domain page survives.
      // A host dereference that demands the full overshoot must be REJECTED, not
      // silently clamped, so a consumer never dereferences past the domain.
      REQUIRE_FALSE(
          GuestRange::FromExactBaseLength(start, requested_len).has_value());
      const uint32_t clamped_len = w_range->length();

      // Seed the whole window stale and validated so a cell that failed to
      // publish (or clamped to the wrong extent/offset) reads back this stale
      // pattern. Writer/consumer identity is in the payload seed.
      BoundaryByteWorld world(base_lo, window);
      auto stale = SeededPattern(window, uint8_t(0x10u + wi * 16u + ci));
      world.RamWrite(base_lo, stale);
      REQUIRE(world.RequestRange(base_lo, window));
      REQUIRE(world.SharedRead(base_lo, window) == stale);
      auto payload = SeededPattern(clamped_len, uint8_t(0x40u + wi * 16u + ci));

      std::vector<PendingProducer> pending;
      uint64_t completed = ApplyBoundaryWriter(
          world, writer, start, requested_len, clamped_len, payload, &pending);
      REQUIRE(pending.size() == (writer == WriterKind::kMemexport ? 1u : 0u));

      // The consumer presents exactly the in-domain clamped page as an exact
      // range (accepted precisely because it no longer overshoots the domain).
      GuestRangeSet requested;
      REQUIRE(requested.Add(GuestRange::FromExactBaseLength(start, clamped_len)));
      ConsumerDecision decision =
          DecideConsumer(consumer, requested, pending, completed);
      REQUIRE(decision.revalidate_gpu_bytes ==
              ConsumerReadsSharedBuffer(consumer));

      // Any pending memexport record converges through the real seam and is
      // retired by its unique authority sequence; the RAM writers leave nothing.
      BoundaryConvergenceTarget convergence_target(&world);
      ProducerDrainOutcome drain = DrainSelectedProducers(
          decision, [&](const PendingProducer& selected) {
            ProducerDrainOutcome one;
            MemexportConvergenceResult converged =
                ConvergeMemexportRam(selected, convergence_target);
            REQUIRE(converged.converged);
            REQUIRE(converged.action == AuthorityAction::kConvergeRamFromBuffer);
            one.memexport_records = 1;
            one.drained_records = 1;
            for (auto it = pending.begin(); it != pending.end(); ++it) {
              if (it->sequence == selected.sequence) {
                pending.erase(it);
                break;
              }
            }
            return one;
          });
      REQUIRE(drain.drained_records == decision.selected.size());

      if (writer == WriterKind::kMemexport) {
        REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
        REQUIRE(decision.selected.size() == 1u);
      } else {
        REQUIRE(decision.action == AuthorityAction::kNoDependency);
        REQUIRE(decision.selected.empty());
      }
      REQUIRE(pending.empty());  // every selected record retired.

      // Observe through the matching sparse storage class: shared-buffer
      // consumers (texture/vertex) revalidate their exact bytes from RAM before
      // reading the buffer; guest-RAM consumers (wait/host CPU read) read RAM
      // directly. The paired high-boundary case supplies the actual Metal GPU
      // transport half of this composed boundary proof.
      std::vector<uint8_t> observed;
      if (ConsumerReadsSharedBuffer(consumer)) {
        REQUIRE(world.RequestRange(start, clamped_len));
        observed = world.SharedRead(start, clamped_len);
      } else {
        observed = world.RamRead(start, clamped_len);
      }
      REQUIRE(observed == payload);
      // Guest RAM holds the converged clamped bytes for every writer.
      REQUIRE(world.RamRead(start, clamped_len) == payload);

      // No out-of-domain / wrapped access: every byte below the clamped page is
      // still the stale seed in BOTH RAM and the shared buffer.
      const uint32_t prefix_len = start - base_lo;
      std::vector<uint8_t> stale_prefix(stale.begin(),
                                        stale.begin() + prefix_len);
      REQUIRE(world.RamRead(base_lo, prefix_len) == stale_prefix);
      REQUIRE(world.SharedRead(base_lo, prefix_len) == stale_prefix);
    }
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
