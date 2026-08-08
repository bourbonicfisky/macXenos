/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 (Metal device required; passes-as-skipped without one): the
// pre-gate-metal-zero-copy-shared-memory task 6.2 DEVICE PROBES.
//
// This is a NARROW addition, not the binary's first GPU work. Plenty of Metal
// conformance suites encode real GPU commands, the shared-memory coherence
// suite among them (it compute-reads the shared buffer and blit-writes into
// it). What is missing is a specific intersection, and only that intersection
// is claimed here.
//
// The gap is on the zero-copy side. The group-4 (convergence/upload) and
// group-5 (overlap) suites are the ones that run the gate's TWO BACKINGS
// against the real MetalSharedMemory over the real Memory -- and they encode no
// GPU commands at all. The production upload and convergence paths they drive
// do move real bytes; what is simulated in them is the GUEST'S OWN producer and
// consumer, a memcpy into `[buffer contents]` standing in for the memexport UAV
// store and a memcpy back out standing in for a shader read. That is enough to
// pin the accounting and the lifecycle, and it is NOT enough for task 6.2,
// whose whole subject is what the GPU actually does with an aliased allocation:
// those memcpys would produce identical numbers on a host where the GPU could
// not see guest RAM at all. Conversely the suites that DO encode GPU work run
// the mirror backing only, and in shapes (compute read, blit write) that leave
// the vertex stage, buffer-backed textures, and the guest fault path untested
// against an alias.
//
// So what this file adds is both backings AND real GPU work, in the producer
// and consumer shapes production actually uses:
//
//   P1  CPU-to-GPU bytes -- a guest CPU write, published through the real
//       physical-invalidation entry, then requested through the production
//       upload seam and READ BY A COMPUTE KERNEL out of the shared buffer.
//   P2  memexport GPU-to-CPU bytes after a guest-visible boundary -- a COMPUTE
//       KERNEL writes the export payload into the shared buffer, the record is
//       registered through the real publication seam before commit, the command
//       buffer is committed and waited (the boundary), and the real gated
//       convergence then publishes to guest RAM.
//   P3  memexport-to-vfetch and memexport-to-texture GPU consumption -- the
//       GPU-exported bytes are consumed again by the GPU, once through a real
//       VERTEX stage that indexes the shared buffer exactly as a translated
//       vfetch does, and once through a linear MTLTexture created ON the shared
//       buffer and sampled by a kernel.
//   P4  physical write-watch after alias creation -- a REAL guest write through
//       the guest virtual mirror, faulting through the installed access
//       violation handler into the production thunk. This is the one question
//       the group-5 suite explicitly could not answer: it drives the thunk
//       directly and records "no real fault transport" as its limitation. The
//       risk being probed is concrete -- `newBufferWithBytesNoCopy` wires guest
//       RAM into the Metal device, and if that defeated `mprotect` the whole
//       write-watch invalidation model would silently stop working under alias
//       mode.
//   P5  ordered resolve/memexport overlap -- an older resolve and a newer
//       GPU-written memexport over the SAME bytes, ordered by the production
//       destination-preparation seam (which retires the older producer BEFORE
//       the store is encoded) and drained through the real executor, with the
//       final bytes read back by the GPU. This is the one probe whose outcome
//       is not backing-symmetric by accident: on the mirror a late resolve
//       drain is undone by memexport convergence copying buffer -> RAM over it,
//       while on an alias that convergence is a self-copy with nothing left to
//       restore, so the writer order has to be enforced rather than survived.
//
// Every probe runs BOTH backings and compares them. The alias arm is never
// asserted as zero-copy on a host that rejected the wrap: `aliases()` is read
// back from the shared memory, and a host that fell back is reported and
// compared as the mirror it actually is.
//
// Normal launch defaults are untouched: the gate is exact-off and every arm
// sets it explicitly around one Initialize, restoring the previous environment.

#import <Metal/Metal.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/memory.h"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_zero_copy_contract.h"
#include "xenia/gpu/metal/metal_zero_copy_overlap.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// Own pages, 1 MiB apart, in a region no other suite in this binary touches, so
// a neighbouring fixture's write-watch state cannot bleed into an upload count.
// (Group 4 owns 0x1A......, group 5 owns 0x1B......, the coherence suite owns
// 0x12....... )
constexpr uint32_t kUploadProbeBase = 0x1C000000u;
constexpr uint32_t kExportProbeBase = 0x1C100000u;
constexpr uint32_t kVfetchProbeBase = 0x1C200000u;
constexpr uint32_t kTextureProbeBase = 0x1C300000u;
constexpr uint32_t kWatchProbeBase = 0x1C400000u;
constexpr uint32_t kOrderProbeBase = 0x1C500000u;

constexpr uint32_t kPage = 4096u;  // guest allocation granularity.
// The guest VIRTUAL mirror of the physical domain. Write watches are installed
// on the virtual physical heaps (vA/vC/vE), never on the physical membase view
// the alias wraps, so a probe that wants a REAL fault has to write here.
constexpr uint32_t kVirtualPhysicalBase = 0xA0000000u;

uint32_t HostPage() { return uint32_t(xe::memory::page_size()); }

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* previous = std::getenv(name);
    had_previous_ = previous != nullptr;
    if (had_previous_) {
      previous_ = previous;
    }
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (had_previous_) {
      setenv(name_, previous_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_previous_ = false;
  std::string previous_;
};

Memory& Guest() { return *conformance::TestPhysicalMemory(); }

std::vector<uint8_t> SeededPattern(uint32_t bytes, uint8_t seed) {
  std::vector<uint8_t> v(bytes);
  for (uint32_t i = 0; i < bytes; ++i) {
    v[i] = uint8_t(seed + i * 7 + (i >> 8));
  }
  return v;
}

// The GPU side of every probe, built once. Nothing here replicates a production
// decision -- these are the shader stages that stand in for the guest's own
// consumers and producers, and they touch the shared buffer only through
// MetalSharedMemory::GetBuffer().
struct ProbeGpu {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> read_words = nil;
  id<MTLComputePipelineState> write_words = nil;
  id<MTLComputePipelineState> read_texels = nil;
  id<MTLRenderPipelineState> vfetch = nil;
  bool ok = false;

  ProbeGpu() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void zc_read_words(device const uint* src [[buffer(0)]],\n"
             "                          device uint* dst [[buffer(1)]],\n"
             "                          constant uint& word_off [[buffer(2)]],\n"
             "                          uint tid [[thread_position_in_grid]]) {\n"
             "  dst[tid] = src[word_off + tid];\n"
             "}\n"
             "kernel void zc_write_words(device const uint* src [[buffer(0)]],\n"
             "                           device uint* dst [[buffer(1)]],\n"
             "                           constant uint& word_off [[buffer(2)]],\n"
             "                           uint tid [[thread_position_in_grid]]) {\n"
             "  dst[word_off + tid] = src[tid];\n"
             "}\n"
             "kernel void zc_read_texels(\n"
             "    texture2d<uint, access::read> tex [[texture(0)]],\n"
             "    device uint* dst [[buffer(0)]],\n"
             "    uint2 tid [[thread_position_in_grid]]) {\n"
             "  uint4 texel = tex.read(tid);\n"
             "  dst[tid.y * tex.get_width() + tid.x] =\n"
             "      (texel.x & 0xFFu) | ((texel.y & 0xFFu) << 8) |\n"
             "      ((texel.z & 0xFFu) << 16) | ((texel.w & 0xFFu) << 24);\n"
             "}\n"
             "struct ZcVertexOut {\n"
             "  float4 position [[position]];\n"
             "  float point_size [[point_size]];\n"
             "  uint value [[flat]];\n"
             "};\n"
             "vertex ZcVertexOut zc_vfetch_vs(\n"
             "    device const uint* stream [[buffer(0)]],\n"
             "    constant uint& word_off [[buffer(1)]],\n"
             "    constant uint& count [[buffer(2)]],\n"
             "    uint vid [[vertex_id]]) {\n"
             "  ZcVertexOut out;\n"
             "  float x = (float(vid) + 0.5) / float(count) * 2.0 - 1.0;\n"
             "  out.position = float4(x, 0.0, 0.0, 1.0);\n"
             "  out.point_size = 1.0;\n"
             "  out.value = stream[word_off + vid];\n"
             "  return out;\n"
             "}\n"
             "fragment uint zc_vfetch_fs(ZcVertexOut in [[stage_in]]) {\n"
             "  return in.value;\n"
             "}\n"
                     options:nil
                       error:&error];
    if (!library) {
      return;
    }
    read_words = [device
        newComputePipelineStateWithFunction:[library
                                                newFunctionWithName:@"zc_read_"
                                                                    @"words"]
                                      error:&error];
    write_words = [device
        newComputePipelineStateWithFunction:[library
                                                newFunctionWithName:@"zc_write_"
                                                                    @"words"]
                                      error:&error];
    read_texels = [device
        newComputePipelineStateWithFunction:[library
                                                newFunctionWithName:@"zc_read_"
                                                                    @"texels"]
                                      error:&error];
    MTLRenderPipelineDescriptor* render_descriptor =
        [[MTLRenderPipelineDescriptor alloc] init];
    render_descriptor.vertexFunction =
        [library newFunctionWithName:@"zc_vfetch_vs"];
    render_descriptor.fragmentFunction =
        [library newFunctionWithName:@"zc_vfetch_fs"];
    render_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
    vfetch = [device newRenderPipelineStateWithDescriptor:render_descriptor
                                                    error:&error];
    ok = queue && read_words && write_words && read_texels && vfetch;
  }
};

ProbeGpu& Gpu() {
  static ProbeGpu gpu;
  return gpu;
}

// One initialized MetalSharedMemory with the gate set for this arm, plus the
// counting overlap providers the alias observer needs to classify anything.
// Alias state is read back from the shared memory rather than assumed from the
// request: a host that rejects the wrap runs the mirror in BOTH arms.
struct ProbeArm {
  std::unique_ptr<MetalSharedMemory> shared;
  bool ok = false;
  // Stands in for the command processor's encoding/completed submission
  // counters. Both are plain values the probe advances explicitly, so no probe
  // can accidentally read completion out of the same source that claimed it.
  uint64_t encoding_submission = 0;
  uint64_t completed_submission = 0;

  explicit ProbeArm(bool request_alias) {
    if (!Gpu().ok) {
      return;
    }
    Memory* memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    ScopedEnv gate(kZeroCopyGateEnvVar, request_alias ? "1" : nullptr);
    ScopedEnv force(kZeroCopyForceRejectEnvVar, nullptr);
    ScopedEnv force_observer(kZeroCopyForceObserverRejectEnvVar, nullptr);
    shared = std::make_unique<MetalSharedMemory>(
        *memory, (__bridge void*)Gpu().device);
    // Installed before Initialize, exactly as MetalCommandProcessor installs
    // them, so the observer is armed before the physical callback exists.
    shared->SetOverlapSubmissionProvider([this]() { return encoding_submission; });
    shared->SetOverlapCompletedSubmissionProvider(
        [this]() { return completed_submission; });
    ok = shared->Initialize();
    if (!ok) {
      shared.reset();
    }
  }

  bool aliases() const { return shared && shared->is_zero_copy(); }
  id<MTLBuffer> buffer() const {
    return shared ? (__bridge id<MTLBuffer>)shared->GetBuffer() : nil;
  }
};

// Commits the physical pages behind a range so it looks like a real guest
// allocation; UploadRanges validates heap access before copying.
void EnsureAllocated(uint32_t address, uint32_t length) {
  uint32_t base = address & ~(kPage - 1);
  uint32_t end = (address + length + kPage - 1) & ~(kPage - 1);
  Guest().GetPhysicalHeap()->AllocFixed(
      base, end - base, kPage,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite);
}

void RamWrite(uint32_t address, const std::vector<uint8_t>& bytes) {
  EnsureAllocated(address, uint32_t(bytes.size()));
  std::memcpy(Guest().TranslatePhysical<uint8_t*>(address), bytes.data(),
              bytes.size());
}

std::vector<uint8_t> RamRead(uint32_t address, uint32_t length) {
  std::vector<uint8_t> out(length);
  std::memcpy(out.data(), Guest().TranslatePhysical<uint8_t*>(address), length);
  return out;
}

// --- Real GPU corridors ----------------------------------------------------

// What a shader consuming the shared buffer observes: a compute kernel copies
// `length` bytes (multiple of 4) out of the shared buffer at guest physical
// `address` into a fresh staging buffer. The wait is observation-only.
std::vector<uint8_t> GpuReadBuffer(const ProbeArm& arm, uint32_t address,
                                   uint32_t length) {
  id<MTLBuffer> staging = [Gpu().device newBufferWithLength:length
                                                    options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command_buffer = [Gpu().queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:Gpu().read_words];
  [encoder setBuffer:arm.buffer() offset:0 atIndex:0];
  [encoder setBuffer:staging offset:0 atIndex:1];
  uint32_t word_offset = address / 4;
  [encoder setBytes:&word_offset length:sizeof(word_offset) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(length / 4, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  std::vector<uint8_t> out(length);
  std::memcpy(out.data(), [staging contents], length);
  return out;
}

// A real GPU write into the shared buffer, encoded but NOT committed: the
// caller registers the pending record before crossing the boundary, which is
// the ordering production uses.
struct EncodedGpuExport {
  id<MTLCommandBuffer> command_buffer = nil;
  uint64_t submission = 0;
};

EncodedGpuExport EncodeGpuExport(ProbeArm& arm, uint32_t address,
                                 const std::vector<uint8_t>& bytes) {
  id<MTLBuffer> staging =
      [Gpu().device newBufferWithLength:bytes.size()
                                options:MTLResourceStorageModeShared];
  std::memcpy([staging contents], bytes.data(), bytes.size());
  EncodedGpuExport encoded;
  encoded.command_buffer = [Gpu().queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder =
      [encoded.command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:Gpu().write_words];
  [encoder setBuffer:staging offset:0 atIndex:0];
  [encoder setBuffer:arm.buffer() offset:0 atIndex:1];
  uint32_t word_offset = address / 4;
  [encoder setBytes:&word_offset length:sizeof(word_offset) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(bytes.size() / 4, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [encoder endEncoding];
  encoded.submission = ++arm.encoding_submission;
  return encoded;
}

// The memexport-to-vfetch corridor: a REAL vertex stage indexes the shared
// buffer by vertex id -- which is exactly the shape a translated vfetch takes
// on this backend, a `device const uint*` at a buffer slot -- and the values it
// fetched are written out per pixel. Returns the fetched words as bytes.
std::vector<uint8_t> GpuVertexFetch(const ProbeArm& arm, uint32_t address,
                                    uint32_t length) {
  const uint32_t count = length / 4;
  MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                   width:count
                                  height:1
                               mipmapped:NO];
  descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  descriptor.storageMode = MTLStorageModeShared;
  id<MTLTexture> target = [Gpu().device newTextureWithDescriptor:descriptor];

  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);

  id<MTLCommandBuffer> command_buffer = [Gpu().queue commandBuffer];
  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:pass];
  [encoder setRenderPipelineState:Gpu().vfetch];
  [encoder setVertexBuffer:arm.buffer() offset:0 atIndex:0];
  uint32_t word_offset = address / 4;
  [encoder setVertexBytes:&word_offset length:sizeof(word_offset) atIndex:1];
  [encoder setVertexBytes:&count length:sizeof(count) atIndex:2];
  [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:count];
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  std::vector<uint8_t> out(length);
  [target getBytes:out.data()
        bytesPerRow:length
         fromRegion:MTLRegionMake2D(0, 0, count, 1)
        mipmapLevel:0];
  return out;
}

// The memexport-to-texture corridor: a linear MTLTexture created ON the shared
// buffer at the exported address (so in alias mode the texture's storage IS
// guest RAM), sampled by a kernel. Returns the texel bytes, or an empty vector
// when this device does not admit a linear texture at this alignment.
std::vector<uint8_t> GpuTextureFetch(const ProbeArm& arm, uint32_t address,
                                     uint32_t width, uint32_t height) {
  const uint32_t bytes_per_row = width * 4;
  const NSUInteger alignment = [Gpu().device
      minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA8Uint];
  if (alignment == 0 || (address % alignment) != 0 ||
      (bytes_per_row % alignment) != 0) {
    return {};
  }
  MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                   width:width
                                  height:height
                               mipmapped:NO];
  descriptor.usage = MTLTextureUsageShaderRead;
  descriptor.storageMode = arm.buffer().storageMode;
  id<MTLTexture> texture = [arm.buffer() newTextureWithDescriptor:descriptor
                                                           offset:address
                                                      bytesPerRow:bytes_per_row];
  if (!texture) {
    return {};
  }
  const uint32_t length = bytes_per_row * height;
  id<MTLBuffer> staging =
      [Gpu().device newBufferWithLength:length
                                options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> command_buffer = [Gpu().queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:Gpu().read_texels];
  [encoder setTexture:texture atIndex:0];
  [encoder setBuffer:staging offset:0 atIndex:0];
  [encoder dispatchThreads:MTLSizeMake(width, height, 1)
      threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  std::vector<uint8_t> out(length);
  std::memcpy(out.data(), [staging contents], length);
  return out;
}

// --- Production seam adapters ----------------------------------------------

// The production convergence port, wired exactly as MetalCommandProcessor's
// MetalMemexportConvergenceTarget wires it: the shared memory owns the
// copy-or-alias decision and reports the bytes it moved.
class SharedMemoryConvergenceTarget final : public MemexportConvergenceTarget {
 public:
  explicit SharedMemoryConvergenceTarget(MetalSharedMemory* shared)
      : shared_(shared) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries,
                       uint64_t* bytes_copied) override {
    ++calls;
    *invalidated_entries = 0;
    SharedBufferConvergenceOutcome published =
        shared_->ConvergeRangeToRam(base, length);
    *bytes_copied = published.bytes_copied;
    alias_skipped = published.alias_skipped;
    return published.converged;
  }

  uint32_t calls = 0;
  bool alias_skipped = false;

 private:
  MetalSharedMemory* shared_;
};

class SharedMemoryPreparationTarget final : public MemexportPreparationTarget {
 public:
  explicit SharedMemoryPreparationTarget(MetalSharedMemory* shared)
      : shared_(shared) {}
  bool RequestDestinationRange(uint32_t base, uint32_t length) override {
    return shared_->RequestRange(base, length);
  }

 private:
  MetalSharedMemory* shared_;
};

class SharedMemoryPublicationTarget final : public MemexportPublicationTarget {
 public:
  explicit SharedMemoryPublicationTarget(MetalSharedMemory* shared)
      : shared_(shared) {}
  bool CanMarkRangeWrittenByGpu() const override { return shared_ != nullptr; }
  void MarkRangeWrittenByGpu(uint32_t base, uint32_t length) override {
    shared_->RangeWrittenByGpu(base, length);
  }

 private:
  MetalSharedMemory* shared_;
};

class SharedMemoryRamWriteTarget final : public RamWritePublicationTarget {
 public:
  explicit SharedMemoryRamWriteTarget(MetalSharedMemory* shared)
      : shared_(shared) {}
  bool InvalidateSharedPages(uint32_t base, uint32_t length,
                             bool exact_range) override {
    ++shared_calls;
    shared_->MemoryInvalidationCallback(base, length, exact_range);
    return true;
  }
  bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                uint32_t* invalidated_entries) override {
    ++legacy_calls;
    *invalidated_entries = 0;
    return true;
  }

  uint32_t shared_calls = 0;
  uint32_t legacy_calls = 0;

 private:
  MetalSharedMemory* shared_;
};

// What the production destination-preparation seam established for one export.
// Reported separately from the bytes so a probe can assert the ORDER the seam
// enforced rather than infer it from the final content.
struct MemexportOrderingEvidence {
  MemexportDestinationOrdering ordering =
      MemexportDestinationOrdering::kNotEvaluated;
  size_t selected = 0;
  bool await = false;
  TransitState first_selected_state = TransitState::kRamCurrent;
  uint64_t first_selected_sequence = 0;
  ProducerDrainOutcome drain;
};

// How a probe retires the older overlapping producers the ordering seam
// selected. Left empty by probes that register no older producer: an empty
// function retires NOTHING, so a selection it did not expect yields
// kOlderProducersRetained and fails the REQUIRE below instead of quietly
// encoding an unordered store.
using RetireOlderProducersFn =
    std::function<ProducerDrainOutcome(const ConsumerDecision&)>;

// One GPU-written memexport, produced and made final exactly the way production
// does: prepare the destination THROUGH THE ORDERED SEAM (retire every older
// overlapping producer, then request the pages), encode the shader store,
// register the pending record BEFORE the commit, then cross the guest-visible
// boundary (commit + wait) and advance the completed high-water mark. Returns
// the owning submission and appends the pending record.
uint64_t GpuMemexport(ProbeArm& arm, uint32_t address,
                      const std::vector<uint8_t>& bytes,
                      std::vector<PendingProducer>* pending, uint64_t sequence,
                      const RetireOlderProducersFn& retire_older = {},
                      MemexportOrderingEvidence* ordering_out = nullptr) {
  const uint32_t length = uint32_t(bytes.size());
  EnsureAllocated(address, length);

  std::optional<MemexportDestinationPreparation> preparation =
      DecideMemexportDestinationPreparation(address, length);
  REQUIRE(preparation.has_value());
  std::vector<MemexportDestinationPreparation> preparations{*preparation};
  SharedMemoryPreparationTarget preparation_target(arm.shared.get());
  // The SAME production entry point MetalCommandProcessor's
  // prepare_memexport_destinations calls: the writer-before-writer ordering and
  // the page request as one step, with the request unreachable when the
  // ordering is unsatisfied.
  MemexportDestinationPreparationOutcome prepared =
      PrepareOrderedMemexportDestinations(
          preparations, *pending, arm.completed_submission, preparation_target,
          [&](const ConsumerDecision& selected) {
            return retire_older ? retire_older(selected)
                                : ProducerDrainOutcome{};
          });
  if (ordering_out) {
    ordering_out->ordering = prepared.ordering;
    ordering_out->selected = prepared.decision.selected.size();
    ordering_out->await = prepared.decision.await;
    if (!prepared.decision.selected.empty()) {
      ordering_out->first_selected_state =
          prepared.decision.selected.front().state;
      ordering_out->first_selected_sequence =
          prepared.decision.selected.front().sequence;
    }
    ordering_out->drain = prepared.drain;
  }
  REQUIRE(prepared.ordered());
  REQUIRE(prepared.prepared());
  REQUIRE(prepared.requested_ranges == 1u);

  EncodedGpuExport encoded = EncodeGpuExport(arm, address, bytes);

  SharedMemoryPublicationTarget publication_target(arm.shared.get());
  std::optional<MemexportPublication> publication =
      DecideMemexportPublication(address, length, encoded.submission);
  REQUIRE(publication.has_value());
  MemexportPublicationResult result = PublishMemexportWrite(
      *publication, publication_target,
      [&](const MemexportPublication& owned) {
        pending->push_back({owned.range, TransitState::kPendingMemexport,
                            owned.submission, sequence});
        return true;
      });
  REQUIRE(result.registered);
  REQUIRE(result.marked);

  // THE GUEST-VISIBLE BOUNDARY. Nothing the shader wrote is final before this,
  // aliasing included: a shared allocation still holds only what the shader has
  // stored so far.
  [encoded.command_buffer commit];
  [encoded.command_buffer waitUntilCompleted];
  REQUIRE(encoded.command_buffer.status == MTLCommandBufferStatusCompleted);
  arm.completed_submission = encoded.submission;
  return encoded.submission;
}

// Drains one selected record through the real shared gate and retires it.
struct DrainEvidence {
  uint32_t target_calls = 0;
  uint64_t bytes_copied = 0;
  bool alias_skipped = false;
  uint32_t drained_records = 0;
};

DrainEvidence DrainMemexports(ProbeArm& arm, const ConsumerDecision& decision,
                              std::vector<PendingProducer>* pending,
                              uint64_t completed) {
  DrainEvidence evidence;
  SharedMemoryConvergenceTarget target(arm.shared.get());
  auto retire_by_sequence = [&](uint64_t sequence) {
    for (auto it = pending->begin(); it != pending->end(); ++it) {
      if (it->sequence == sequence) {
        pending->erase(it);
        return true;
      }
    }
    return false;
  };
  ProducerDrainOutcome drain = DrainSelectedProducers(
      decision, [&](const PendingProducer& selected) {
        ProducerDrainOutcome one;
        GatedMemexportConvergence gated =
            ConvergeCompletedMemexportRam(selected, completed, target);
        REQUIRE(gated.attempted());
        REQUIRE(gated.retires());
        evidence.bytes_copied += gated.result.bytes_copied;
        one.drained_records = 1;
        one.memexport_records = 1;
        one.memexport_bytes_copied = gated.result.bytes_copied;
        REQUIRE(retire_by_sequence(selected.sequence));
        return one;
      });
  evidence.target_calls = target.calls;
  evidence.alias_skipped = target.alias_skipped;
  evidence.drained_records = drain.drained_records;
  return evidence;
}

// The guest virtual mirror's own allocation, made once for the whole binary:
// AllocFixed over an already-reserved range fails, and re-reserving per arm
// would turn a second arm into a false negative.
bool EnsureWatchPageAllocated(uint32_t address, uint32_t length) {
  static bool allocated = [&]() {
    const uint32_t virtual_address = kVirtualPhysicalBase + address;
    BaseHeap* heap = Guest().LookupHeap(virtual_address);
    if (!heap) {
      return false;
    }
    return heap->AllocFixed(
        virtual_address, length, 64u * 1024u,
        kMemoryAllocationReserve | kMemoryAllocationCommit,
        kMemoryProtectRead | kMemoryProtectWrite);
  }();
  return allocated;
}

bool HostPageProtectionIsReadOnly(uint32_t virtual_address) {
  void* host = Guest().TranslateVirtual<void*>(virtual_address);
  size_t length = 1;
  xe::memory::PageAccess access = xe::memory::PageAccess::kNoAccess;
  if (!xe::memory::QueryProtect(host, length, access)) {
    return false;
  }
  return access == xe::memory::PageAccess::kReadOnly;
}

#define REQUIRE_GPU()                                                       \
  if (!Gpu().ok) {                                                          \
    SUCCEED("no Metal device; zero-copy device probes are skipped");        \
    return;                                                                 \
  }

}  // namespace

TEST_CASE("Zero-copy probe: the device probe harness is live (not skipped)",
          "[l1][metal][zerocopy][probe]") {
  if (!Gpu().ok) {
    WARN("no Metal device; zero-copy task 6.2 device probes are skipped");
  }
  SUCCEED();
}

// --- P1: CPU-to-GPU bytes ---------------------------------------------------

TEST_CASE("Zero-copy probe: a guest CPU write is read by the GPU out of both "
          "backings, and only the mirror copies it",
          "[l1][metal][zerocopy][probe][cpu-to-gpu]") {
  REQUIRE_GPU();
  const uint32_t length = 2 * HostPage();
  const std::vector<uint8_t> written = SeededPattern(length, 0x4B);

  struct ArmEvidence {
    bool aliased = false;
    uint64_t copied_delta = 0;
    uint64_t aliased_delta = 0;
    std::vector<uint8_t> gpu_observed;
  };
  ArmEvidence evidence[2];

  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    const bool request_alias = arm_index == 1;
    INFO("alias requested: " << request_alias);
    ProbeArm arm(request_alias);
    REQUIRE(arm.ok);
    evidence[arm_index].aliased = arm.aliases();

    // The production guest-CPU corridor: RAM is written, then the physical
    // invalidation entry publishes it, then the consumer requests the range.
    RamWrite(kUploadProbeBase, written);
    arm.shared->MemoryInvalidationCallback(kUploadProbeBase, length,
                                           /*exact_range=*/true);
    const uint64_t copied_before = arm.shared->upload_bytes_copied();
    const uint64_t aliased_before = arm.shared->upload_bytes_aliased();
    REQUIRE(arm.shared->RequestRange(kUploadProbeBase, length));
    evidence[arm_index].copied_delta =
        arm.shared->upload_bytes_copied() - copied_before;
    evidence[arm_index].aliased_delta =
        arm.shared->upload_bytes_aliased() - aliased_before;

    // THE PROBE: a shader, not a memcpy through `contents`, reports what the
    // GPU can see. A host whose device could not actually reach guest RAM would
    // fail exactly here and nowhere in the group-4 suite.
    evidence[arm_index].gpu_observed =
        GpuReadBuffer(arm, kUploadProbeBase, length);
  }

  CHECK_FALSE(evidence[0].aliased);
  CHECK(evidence[0].copied_delta == uint64_t(length));
  CHECK(evidence[0].aliased_delta == 0u);
  CHECK(evidence[0].gpu_observed == written);

  if (!evidence[1].aliased) {
    WARN("host does not alias guest RAM; the alias CPU-to-GPU arm is the "
         "mirror path and is not asserted as zero-copy");
    CHECK(evidence[1].gpu_observed == written);
    return;
  }
  CHECK(evidence[1].copied_delta == 0u);
  CHECK(evidence[1].aliased_delta == uint64_t(length));
  CHECK(evidence[1].gpu_observed == written);
  CHECK(evidence[1].gpu_observed == evidence[0].gpu_observed);
}

// --- P2: memexport GPU-to-CPU after a guest-visible boundary ---------------

TEST_CASE("Zero-copy probe: shader-written memexport bytes reach guest RAM "
          "after the boundary, with the mirror copying and the alias not",
          "[l1][metal][zerocopy][probe][gpu-to-cpu]") {
  REQUIRE_GPU();
  const uint32_t length = kPage;
  const std::vector<uint8_t> filler = SeededPattern(length, 0x22);
  const std::vector<uint8_t> exported = SeededPattern(length, 0x77);

  struct ArmEvidence {
    bool aliased = false;
    std::vector<uint8_t> ram_before;
    std::vector<uint8_t> ram_after;
    std::vector<uint8_t> gpu_after;
    uint64_t bytes_copied = 0;
    bool alias_skipped = false;
    uint32_t target_calls = 0;
    uint32_t drained_records = 0;
  };
  ArmEvidence evidence[2];

  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    const bool request_alias = arm_index == 1;
    INFO("alias requested: " << request_alias);
    ProbeArm arm(request_alias);
    REQUIRE(arm.ok);
    evidence[arm_index].aliased = arm.aliases();

    // Prime RAM with a filler the export must overwrite, and make the mirror
    // hold it too, so the mirror's post-convergence bytes cannot be the filler
    // by accident.
    RamWrite(kExportProbeBase, filler);
    arm.shared->MemoryInvalidationCallback(kExportProbeBase, length, true);
    REQUIRE(arm.shared->RequestRange(kExportProbeBase, length));

    std::vector<PendingProducer> pending;
    const uint64_t submission =
        GpuMemexport(arm, kExportProbeBase, exported, &pending, /*sequence=*/1u);
    REQUIRE(pending.size() == 1u);

    // What guest RAM holds after the boundary but BEFORE convergence. In the
    // mirror this is still the filler; in the alias the shader stored straight
    // into guest RAM. That difference is the backing, so it is captured rather
    // than asserted, and the claim below ("RAM ends holding what the SHADER
    // wrote") is the one that means the same thing in both arms.
    evidence[arm_index].ram_before = RamRead(kExportProbeBase, length);

    GuestRangeSet requested;
    requested.Add(GuestRange::FromExactBaseLength(kExportProbeBase, length));
    ConsumerDecision decision = DecideConsumer(
        ConsumerKind::kHostCpuRead, requested, pending, submission);
    REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
    REQUIRE_FALSE(decision.await);

    DrainEvidence drain = DrainMemexports(arm, decision, &pending, submission);
    REQUIRE(pending.empty());
    evidence[arm_index].bytes_copied = drain.bytes_copied;
    evidence[arm_index].alias_skipped = drain.alias_skipped;
    evidence[arm_index].target_calls = drain.target_calls;
    evidence[arm_index].drained_records = drain.drained_records;
    evidence[arm_index].ram_after = RamRead(kExportProbeBase, length);
    evidence[arm_index].gpu_after = GpuReadBuffer(arm, kExportProbeBase, length);
  }

  // Mirror: the shader wrote the buffer only, so RAM still held the filler at
  // the boundary and convergence moved exactly the record's length.
  CHECK_FALSE(evidence[0].aliased);
  CHECK(evidence[0].ram_before == filler);
  CHECK(evidence[0].ram_after == exported);
  CHECK(evidence[0].gpu_after == exported);
  CHECK(evidence[0].bytes_copied == uint64_t(length));
  CHECK_FALSE(evidence[0].alias_skipped);
  CHECK(evidence[0].target_calls == 1u);
  CHECK(evidence[0].drained_records == 1u);

  if (!evidence[1].aliased) {
    WARN("host does not alias guest RAM; the alias memexport arm is the mirror "
         "path and is not asserted as zero-copy");
    CHECK(evidence[1].ram_after == exported);
    return;
  }

  // Alias: the shader's store landed in guest RAM directly, the same record
  // retired through the same gate, and ZERO bytes moved -- yet the guest ends
  // on byte-identical content.
  CHECK(evidence[1].ram_before == exported);
  CHECK(evidence[1].ram_after == exported);
  CHECK(evidence[1].gpu_after == exported);
  CHECK(evidence[1].bytes_copied == 0u);
  CHECK(evidence[1].alias_skipped);
  CHECK(evidence[1].target_calls == 1u);
  CHECK(evidence[1].drained_records == 1u);
  CHECK(evidence[1].ram_after == evidence[0].ram_after);
}

// --- P3: memexport-to-vfetch and memexport-to-texture GPU consumption ------

TEST_CASE("Zero-copy probe: a real vertex stage fetches shader-exported bytes "
          "out of both backings",
          "[l1][metal][zerocopy][probe][vfetch]") {
  REQUIRE_GPU();
  const uint32_t length = 256u;  // 64 fetched words.
  const std::vector<uint8_t> exported = SeededPattern(length, 0x53);
  std::vector<uint8_t> fetched[2];
  bool aliased[2] = {false, false};

  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    const bool request_alias = arm_index == 1;
    INFO("alias requested: " << request_alias);
    ProbeArm arm(request_alias);
    REQUIRE(arm.ok);
    aliased[arm_index] = arm.aliases();

    RamWrite(kVfetchProbeBase, SeededPattern(length, 0x01));
    arm.shared->MemoryInvalidationCallback(kVfetchProbeBase, length, true);
    REQUIRE(arm.shared->RequestRange(kVfetchProbeBase, length));

    std::vector<PendingProducer> pending;
    const uint64_t submission =
        GpuMemexport(arm, kVfetchProbeBase, exported, &pending, 1u);
    GuestRangeSet requested;
    requested.Add(GuestRange::FromExactBaseLength(kVfetchProbeBase, length));
    ConsumerDecision decision =
        DecideConsumer(ConsumerKind::kVertex, requested, pending, submission);
    REQUIRE(decision.revalidate_gpu_bytes);
    DrainMemexports(arm, decision, &pending, submission);
    REQUIRE(pending.empty());

    // The vertex consumer's production entry (RequestConsumedRange is what a
    // vfetch reaches; it is also the alias enrollment seam), then the fetch
    // itself on the GPU.
    REQUIRE(arm.shared->RequestConsumedRange(kVfetchProbeBase, length));
    fetched[arm_index] = GpuVertexFetch(arm, kVfetchProbeBase, length);
  }

  CHECK_FALSE(aliased[0]);
  CHECK(fetched[0] == exported);
  if (!aliased[1]) {
    WARN("host does not alias guest RAM; the alias vfetch arm is the mirror "
         "path");
  }
  CHECK(fetched[1] == exported);
  CHECK(fetched[1] == fetched[0]);
}

TEST_CASE("Zero-copy probe: a texture created on the shared buffer reads "
          "shader-exported bytes out of both backings",
          "[l1][metal][zerocopy][probe][texture]") {
  REQUIRE_GPU();
  const uint32_t width = 64u;
  const uint32_t height = 4u;
  const uint32_t length = width * height * 4u;
  const std::vector<uint8_t> exported = SeededPattern(length, 0x8E);
  std::vector<uint8_t> sampled[2];
  bool aliased[2] = {false, false};
  bool admitted = true;

  for (int arm_index = 0; arm_index < 2 && admitted; ++arm_index) {
    const bool request_alias = arm_index == 1;
    INFO("alias requested: " << request_alias);
    ProbeArm arm(request_alias);
    REQUIRE(arm.ok);
    aliased[arm_index] = arm.aliases();

    RamWrite(kTextureProbeBase, SeededPattern(length, 0x02));
    arm.shared->MemoryInvalidationCallback(kTextureProbeBase, length, true);
    REQUIRE(arm.shared->RequestRange(kTextureProbeBase, length));

    std::vector<PendingProducer> pending;
    const uint64_t submission =
        GpuMemexport(arm, kTextureProbeBase, exported, &pending, 1u);
    GuestRangeSet requested;
    requested.Add(GuestRange::FromExactBaseLength(kTextureProbeBase, length));
    ConsumerDecision decision =
        DecideConsumer(ConsumerKind::kTexture, requested, pending, submission);
    REQUIRE(decision.revalidate_gpu_bytes);
    DrainMemexports(arm, decision, &pending, submission);
    REQUIRE(pending.empty());

    REQUIRE(arm.shared->RequestConsumedRange(kTextureProbeBase, length));
    sampled[arm_index] =
        GpuTextureFetch(arm, kTextureProbeBase, width, height);
    if (sampled[arm_index].empty()) {
      admitted = false;
    }
  }

  if (!admitted) {
    WARN("this device does not admit a linear texture over the shared buffer "
         "at this alignment; the memexport-to-texture probe did not run");
    return;
  }
  CHECK_FALSE(aliased[0]);
  CHECK(sampled[0] == exported);
  if (!aliased[1]) {
    WARN("host does not alias guest RAM; the alias texture arm is the mirror "
         "path");
  }
  CHECK(sampled[1] == exported);
  CHECK(sampled[1] == sampled[0]);
}

// --- P4: physical write-watch after alias creation -------------------------

TEST_CASE("Zero-copy probe: a real guest write faults and invalidates after "
          "the alias buffer has been created",
          "[l1][metal][zerocopy][probe][write-watch]") {
  REQUIRE_GPU();
  const uint32_t length = 2 * HostPage();
  const uint32_t virtual_address = kVirtualPhysicalBase + kWatchProbeBase;
  if (!EnsureWatchPageAllocated(kWatchProbeBase, 64u * 1024u)) {
    WARN("could not reserve the guest virtual mirror for the write-watch "
         "probe; it did not run");
    return;
  }
  const std::vector<uint8_t> first = SeededPattern(length, 0x15);
  const std::vector<uint8_t> second = SeededPattern(length, 0x9C);

  struct ArmEvidence {
    bool aliased = false;
    bool protection_armed = false;
    uint64_t copied_delta = 0;
    uint64_t aliased_delta = 0;
    uint64_t cpu_write_callbacks = 0;
    std::vector<uint8_t> ram_after_write;
    std::vector<uint8_t> gpu_after_write;
  };
  ArmEvidence evidence[2];

  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    const bool request_alias = arm_index == 1;
    INFO("alias requested: " << request_alias);
    ProbeArm arm(request_alias);
    REQUIRE(arm.ok);
    evidence[arm_index].aliased = arm.aliases();

    uint8_t* guest_virtual = Guest().TranslateVirtual<uint8_t*>(virtual_address);
    REQUIRE(guest_virtual != nullptr);
    std::memcpy(guest_virtual, first.data(), first.size());

    // Validating the range is what arms the watch: MakeRangeValid enables the
    // physical access callbacks, which mprotect the guest virtual mirror down
    // to read-only. In alias mode Metal is holding those very bytes as buffer
    // storage at the same time.
    REQUIRE(arm.shared->RequestRange(kWatchProbeBase, length));
    evidence[arm_index].protection_armed =
        HostPageProtectionIsReadOnly(virtual_address);

    // THE PROBE: a genuine guest store through the virtual mirror. If the
    // protection is live this traps into the installed access-violation
    // handler, which reaches the production physical-invalidation thunk. No
    // part of this path is stubbed.
    const uint64_t callbacks_before =
        arm.shared->zero_copy_overlap_snapshot().cpu_write_callbacks;
    std::memcpy(guest_virtual, second.data(), second.size());
    evidence[arm_index].cpu_write_callbacks =
        arm.shared->zero_copy_overlap_snapshot().cpu_write_callbacks -
        callbacks_before;
    evidence[arm_index].ram_after_write = RamRead(kWatchProbeBase, length);

    // Whether the watch actually invalidated: a re-request re-presents exactly
    // the pages the fault dropped. Had the fault not reached shared memory the
    // pages would still be valid and both deltas would be zero.
    const uint64_t copied_before = arm.shared->upload_bytes_copied();
    const uint64_t aliased_before = arm.shared->upload_bytes_aliased();
    REQUIRE(arm.shared->RequestRange(kWatchProbeBase, length));
    evidence[arm_index].copied_delta =
        arm.shared->upload_bytes_copied() - copied_before;
    evidence[arm_index].aliased_delta =
        arm.shared->upload_bytes_aliased() - aliased_before;
    evidence[arm_index].gpu_after_write =
        GpuReadBuffer(arm, kWatchProbeBase, length);
  }

  // Mirror: the established behaviour, recorded here as the control the alias
  // arm is compared against rather than assumed.
  CHECK_FALSE(evidence[0].aliased);
  CHECK(evidence[0].protection_armed);
  CHECK(evidence[0].ram_after_write == second);
  CHECK(evidence[0].copied_delta == uint64_t(length));
  CHECK(evidence[0].gpu_after_write == second);

  if (!evidence[1].aliased) {
    WARN("host does not alias guest RAM; the alias write-watch arm is the "
         "mirror path and proves nothing about a wrapped allocation");
    return;
  }

  // Alias: wrapping guest RAM in a Metal buffer did NOT defeat the write watch.
  // The page was still protected, the store still trapped, the production thunk
  // still ran (the observer counted the callback), the range was invalidated,
  // and the GPU sees the rewritten bytes.
  CHECK(evidence[1].protection_armed);
  CHECK(evidence[1].cpu_write_callbacks >= 1u);
  CHECK(evidence[1].ram_after_write == second);
  CHECK(evidence[1].copied_delta == 0u);
  CHECK(evidence[1].aliased_delta == uint64_t(length));
  CHECK(evidence[1].gpu_after_write == second);
  CHECK(evidence[1].gpu_after_write == evidence[0].gpu_after_write);
}

// --- P5: ordered resolve/memexport overlap ---------------------------------

TEST_CASE("Zero-copy probe: an older resolve and a newer GPU memexport over "
          "the same bytes retire in sequence order in both backings",
          "[l1][metal][zerocopy][probe][ordered-overlap]") {
  REQUIRE_GPU();
  const uint32_t length = kPage;
  const std::vector<uint8_t> stale = SeededPattern(length, 0x30);
  const std::vector<uint8_t> resolved = SeededPattern(length, 0x31);
  const std::vector<uint8_t> exported = SeededPattern(length, 0x32);

  struct ArmEvidence {
    bool aliased = false;
    // The writer-before-writer ordering seam, run at destination preparation
    // BEFORE the store was encoded.
    MemexportOrderingEvidence ordering;
    // RAM as it stood the instant that ordering completed. This is what makes
    // "the older writer really went first" a measurement rather than an
    // inference from the final bytes: if the resolve had not landed here, the
    // final content would be right for the wrong reason.
    std::vector<uint8_t> ram_after_ordering;
    // The consumer barrier that follows the export.
    size_t selected = 0;
    TransitState selected_state = TransitState::kRamCurrent;
    // Totals across BOTH seams: one resolve retired at the ordering barrier,
    // one memexport retired at the consumer barrier.
    uint32_t resolve_records = 0;
    uint32_t memexport_records = 0;
    uint64_t memexport_bytes_copied = 0;
    std::vector<uint8_t> ram_final;
    std::vector<uint8_t> gpu_final;
    size_t pending_left = 0;
  };
  ArmEvidence evidence[2];

  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    const bool request_alias = arm_index == 1;
    INFO("alias requested: " << request_alias);
    ProbeArm arm(request_alias);
    REQUIRE(arm.ok);
    evidence[arm_index].aliased = arm.aliases();

    RamWrite(kOrderProbeBase, stale);
    arm.shared->MemoryInvalidationCallback(kOrderProbeBase, length, true);
    REQUIRE(arm.shared->RequestRange(kOrderProbeBase, length));

    // The OLDER writer: a resolve that owes RAM its scattered bytes, recorded
    // as a pending producer exactly as the command processor records one.
    std::vector<PendingProducer> pending;
    pending.push_back({*GuestRange::FromExactBaseLength(kOrderProbeBase, length),
                       TransitState::kPendingResolve,
                       /*submission=*/0u, /*sequence=*/1u});

    // One drain implementation, shared by the ordering barrier below and the
    // consumer barrier after the export, so neither seam gets a private notion
    // of what retiring a record means.
    SharedMemoryConvergenceTarget convergence_target(arm.shared.get());
    auto retire_by_sequence = [&](uint64_t sequence) {
      for (auto it = pending.begin(); it != pending.end(); ++it) {
        if (it->sequence == sequence) {
          pending.erase(it);
          return true;
        }
      }
      return false;
    };
    auto drain_one = [&](const PendingProducer& selected) {
      ProducerDrainOutcome one;
      if (selected.state == TransitState::kPendingResolve) {
        RamWrite(selected.range.start(), resolved);
        std::optional<RamWritePublication> publication =
            DecideRamWritePublication(WriterKind::kResolve,
                                      selected.range.start(),
                                      selected.range.length());
        REQUIRE(publication.has_value());
        SharedMemoryRamWriteTarget resolve_target(arm.shared.get());
        RamWritePublicationOutcome outcome =
            PublishRamWrite(*publication, resolve_target);
        REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
        one.resolve_records = 1;
      } else {
        GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
            selected, arm.completed_submission, convergence_target);
        REQUIRE(gated.attempted());
        REQUIRE(gated.retires());
        one.memexport_records = 1;
        one.memexport_bytes_copied = gated.result.bytes_copied;
      }
      one.drained_records = 1;
      REQUIRE(retire_by_sequence(selected.sequence));
      return one;
    };

    // The NEWER writer: a real shader store into the same bytes. Its
    // destination preparation is what retires the resolve above -- the store is
    // not encoded, and the destination pages are not even requested, until that
    // older producer has published. This stands in for
    // MetalCommandProcessor::ResolvePendingProducerDependencies, which performs
    // the same selective drain behind its commit/wait boundary.
    RetireOlderProducersFn retire_older =
        [&](const ConsumerDecision& decision) {
          ProducerDrainOutcome drain = DrainSelectedProducers(decision,
                                                              drain_one);
          evidence[arm_index].ram_after_ordering =
              RamRead(kOrderProbeBase, length);
          return drain;
        };
    const uint64_t submission =
        GpuMemexport(arm, kOrderProbeBase, exported, &pending, /*sequence=*/2u,
                     retire_older, &evidence[arm_index].ordering);
    // The resolve retired at the ordering barrier; only the export remains.
    REQUIRE(pending.size() == 1u);
    evidence[arm_index].resolve_records +=
        evidence[arm_index].ordering.drain.resolve_records;
    evidence[arm_index].memexport_records +=
        evidence[arm_index].ordering.drain.memexport_records;

    GuestRangeSet requested;
    requested.Add(GuestRange::FromExactBaseLength(kOrderProbeBase, length));
    ConsumerDecision decision =
        DecideConsumer(ConsumerKind::kVertex, requested, pending, submission);
    REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
    REQUIRE(decision.selected.size() == 1u);
    evidence[arm_index].selected = decision.selected.size();
    evidence[arm_index].selected_state = decision.selected[0].state;

    ProducerDrainOutcome drain = DrainSelectedProducers(decision, drain_one);
    REQUIRE(drain.drained_records == decision.selected.size());
    evidence[arm_index].resolve_records += drain.resolve_records;
    evidence[arm_index].memexport_records += drain.memexport_records;
    evidence[arm_index].memexport_bytes_copied = drain.memexport_bytes_copied;
    evidence[arm_index].pending_left = pending.size();
    evidence[arm_index].ram_final = RamRead(kOrderProbeBase, length);

    REQUIRE(arm.shared->RequestConsumedRange(kOrderProbeBase, length));
    evidence[arm_index].gpu_final = GpuReadBuffer(arm, kOrderProbeBase, length);
  }

  for (int arm_index = 0; arm_index < 2; ++arm_index) {
    INFO("arm " << arm_index << " aliased: " << evidence[arm_index].aliased);
    // The ordering seam saw exactly the older resolve, and retired it, before
    // the export existed. An empty selection here would mean the seam never
    // looked -- which is why it has its own verdict.
    CHECK(evidence[arm_index].ordering.ordering ==
          MemexportDestinationOrdering::kOlderProducersRetired);
    CHECK(evidence[arm_index].ordering.selected == 1u);
    CHECK(evidence[arm_index].ordering.first_selected_state ==
          TransitState::kPendingResolve);
    CHECK(evidence[arm_index].ordering.first_selected_sequence == 1u);
    CHECK(evidence[arm_index].ordering.drain.drained_records == 1u);
    CHECK(evidence[arm_index].ram_after_ordering == resolved);
    // ... and the export is then the only record the consumer barrier finds.
    CHECK(evidence[arm_index].selected == 1u);
    CHECK(evidence[arm_index].selected_state ==
          TransitState::kPendingMemexport);
    CHECK(evidence[arm_index].resolve_records == 1u);
    CHECK(evidence[arm_index].memexport_records == 1u);
    CHECK(evidence[arm_index].pending_left == 0u);
    // The newer writer wins both corridors: the resolve drained first, then the
    // memexport published over it. Ordering is decided by the same sequence in
    // both backings -- aliasing must not reorder anything.
    CHECK(evidence[arm_index].ram_final == exported);
    CHECK(evidence[arm_index].gpu_final == exported);
    CHECK_FALSE(evidence[arm_index].ram_final == resolved);
  }
  CHECK(evidence[0].memexport_bytes_copied == uint64_t(length));
  CHECK(evidence[0].ram_final == evidence[1].ram_final);
  CHECK(evidence[0].gpu_final == evidence[1].gpu_final);
  if (!evidence[1].aliased) {
    WARN("host does not alias guest RAM; the alias ordered-overlap arm is the "
         "mirror path and is not asserted as zero-copy");
    return;
  }
  CHECK(evidence[1].memexport_bytes_copied == 0u);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
