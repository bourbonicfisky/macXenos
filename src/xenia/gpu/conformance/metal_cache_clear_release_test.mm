/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-cache-clearing section 3: what a clear actually releases.
//
// What is covered here:
//
//   1. The boundary ORDERING ALGORITHM. A loop shaped like
//      ExecuteDeferredCacheClear, driven by the real admission predicate,
//      records the sequence of operations it produces, so "release ran before
//      publish" is a test failure rather than a code-reading exercise.
//
//      SCOPE LIMIT, stated plainly: this exercises the decision sequence, NOT
//      the production drain. It creates no memexport or render-target
//      writeback and never observes guest RAM, so it is NOT the factual
//      publication-before-release proof task 3.1 requires -- that needs an
//      outstanding producer landing real bytes in RAM across a real clear.
//      3.1 stays open on that gap.
//
//   2. Bridge classification: which resolve-snapshot bridges a clear may
//      release, decided from the production ram_backed/is_depth state set
//      through the production registration calls.
//
//   3. Post-clear consistency of the real MetalTextureCache: no cached texture
//      or retire-list entry survives, every freed bindless slot describes the
//      null texture rather than a released one, retained bridges keep their
//      slots, and the cache repopulates afterwards.
//
// Textures, bindless descriptors, and the view heap are real Metal objects,
// and every entry is created through the production GetTextureSlot path. The
// external-slot API used below is the exact adapter MetalSharedTextureCache's
// ReleaseBindingSlot retires through, so the cross-cache ordering rule (shared
// subclass releases into the legacy pools, legacy clear then drains them) is
// exercised with production calls rather than a replica.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "metal_irconverter_runtime.h"
#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_cache_clear_admission.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPage = 4096;
constexpr uint32_t kExtent = 0x2000;

void EnsureAllocated(Memory& memory, uint32_t base, uint32_t length) {
  uint32_t first = base & ~(kPage - 1);
  uint32_t last = (base + length + kPage - 1) & ~(kPage - 1);
  memory.GetPhysicalHeap()->AllocFixed(first, last - first, kPage,
                                       kMemoryAllocationReserve | kMemoryAllocationCommit,
                                       kMemoryProtectRead | kMemoryProtectWrite);
}

void FillMemory(Memory& memory, uint32_t base, uint32_t length, uint8_t seed) {
  EnsureAllocated(memory, base, length);
  uint8_t* bytes = memory.TranslatePhysical<uint8_t*>(base);
  REQUIRE(bytes != nullptr);
  for (uint32_t i = 0; i < length; ++i) {
    bytes[i] = uint8_t(seed + i * 7u + (i >> 8));
  }
}

xenos::xe_gpu_texture_fetch_t MakeFetch(uint32_t base) {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.type = xenos::FetchConstantType::kTexture;
  fetch.sign_x = xenos::TextureSign::kUnsigned;
  fetch.sign_y = xenos::TextureSign::kUnsigned;
  fetch.sign_z = xenos::TextureSign::kUnsigned;
  fetch.sign_w = xenos::TextureSign::kUnsigned;
  fetch.pitch = 2;  // 64 texels.
  fetch.tiled = 0;
  fetch.format = xenos::TextureFormat::k_8_8_8_8;
  fetch.endianness = xenos::Endian::kNone;
  fetch.base_address = base >> 12;
  fetch.size_2d.width = 63;
  fetch.size_2d.height = 31;
  fetch.swizzle = 0x688u;
  fetch.mip_min_level = 0;
  fetch.mip_max_level = 0;
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  return fetch;
}

struct ReleaseHarness {
  id<MTLDevice> device = nil;
  Memory* memory = nullptr;
  bool ok = false;

  ReleaseHarness() {
    device = MTLCreateSystemDefaultDevice();
    memory = conformance::TestPhysicalMemory();
    ok = device != nil && memory != nullptr;
  }
};

ReleaseHarness& H() {
  static ReleaseHarness harness;
  return harness;
}

#define REQUIRE_DEVICE()        \
  if (!H().ok) {                \
    SUCCEED("no Metal device"); \
    return;                     \
  }

// A freed slot must describe the NULL TEXTURE -- the state initialization puts
// every slot in so any index a shader can form is a valid resource. An empty
// (zeroed) entry would be an invalid resource reference, which is a worse
// outcome than the stale one the clear is removing.
bool DescriptorIsNull(id<MTLBuffer> heap, uint32_t slot, id<MTLTexture> null_texture) {
  const auto* entries = static_cast<const IRDescriptorTableEntry*>(heap.contents);
  IRDescriptorTableEntry expected;
  std::memset(&expected, 0, sizeof(expected));
  IRDescriptorTableSetTexture(&expected, null_texture, 0.0f, 0);
  return std::memcmp(&entries[slot], &expected, sizeof(expected)) == 0;
}

id<MTLTexture> MakeNullTexture(id<MTLDevice> device) {
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:1
                                                        height:1
                                                     mipmapped:NO];
  return [device newTextureWithDescriptor:desc];
}

// ---------------------------------------------------------------------------
// 1. The boundary ordering algorithm (decision sequence only -- see the scope
//    limit at the top of this file; this is not the 3.1 factual proof).
// ---------------------------------------------------------------------------

// A boundary driven by the REAL predicate, mirroring the shape of
// ExecuteDeferredCacheClear: each decision performs its action and re-decides,
// and every action is appended to a trace so the resulting ORDER is testable.
struct BoundaryTrace {
  std::vector<std::string> steps;
  bool released = false;

  // Returns true if the clear completed.
  bool Run(CacheClearAdmissionInputs in, bool publication_converges = true) {
    for (int pass = 0; pass < 4; ++pass) {
      switch (DecideCacheClearAdmission(in)) {
        case CacheClearAdmission::kEndSubmissionFirst:
          steps.push_back("quiesce");
          in.encoder_open = false;
          in.has_unsubmitted_work = false;
          in.completed_submission = in.last_submitted_submission;
          break;
        case CacheClearAdmission::kPublishPendingFirst:
          steps.push_back("publish");
          if (publication_converges) {
            in.has_pending_producers = false;
          } else {
            in.publication_failed = true;
          }
          break;
        case CacheClearAdmission::kExecute:
          steps.push_back("release");
          released = true;
          return true;
        case CacheClearAdmission::kDefer:
          steps.push_back("defer");
          return false;
        case CacheClearAdmission::kNoRequest:
          steps.push_back("none");
          return false;
      }
    }
    steps.push_back("gaveup");
    return false;
  }
};

CacheClearAdmissionInputs BoundaryInputs() {
  CacheClearAdmissionInputs in{};
  in.request_latched = true;
  in.last_submitted_submission = 7;
  in.completed_submission = 7;
  in.can_end_submission = true;
  in.can_publish_pending = true;
  return in;
}

TEST_CASE("The boundary sequence publishes before it releases", "[cacheclear]") {
  CacheClearAdmissionInputs in = BoundaryInputs();
  in.encoder_open = true;
  in.completed_submission = 3;  // work still in flight
  in.has_pending_producers = true;

  BoundaryTrace trace;
  REQUIRE(trace.Run(in));
  // The whole contract in one assertion: quiesce, then publish, then release.
  CHECK(trace.steps == std::vector<std::string>{"quiesce", "publish", "release"});
}

TEST_CASE("Release never precedes publication in any reachable boundary", "[cacheclear]") {
  // Every combination of the two hazardous starting conditions, so the
  // ordering claim does not rest on the one path above.
  for (int bits = 0; bits < 8; ++bits) {
    CacheClearAdmissionInputs in = BoundaryInputs();
    in.encoder_open = (bits & 1) != 0;
    in.has_unsubmitted_work = (bits & 2) != 0;
    in.has_pending_producers = (bits & 4) != 0;
    if (in.encoder_open || in.has_unsubmitted_work) {
      in.completed_submission = 3;
    }

    BoundaryTrace trace;
    REQUIRE(trace.Run(in));
    REQUIRE(trace.released);
    auto release_at = std::find(trace.steps.begin(), trace.steps.end(), "release");
    auto publish_at = std::find(trace.steps.begin(), trace.steps.end(), "publish");
    if (in.has_pending_producers) {
      REQUIRE(publish_at != trace.steps.end());
      CHECK(publish_at < release_at);
    }
    // Quiescing always precedes release too.
    auto quiesce_at = std::find(trace.steps.begin(), trace.steps.end(), "quiesce");
    if (in.encoder_open || in.has_unsubmitted_work) {
      REQUIRE(quiesce_at != trace.steps.end());
      CHECK(quiesce_at < release_at);
    }
  }
}

TEST_CASE("A boundary whose publication fails releases nothing", "[cacheclear]") {
  CacheClearAdmissionInputs in = BoundaryInputs();
  in.has_pending_producers = true;

  BoundaryTrace trace;
  CHECK_FALSE(trace.Run(in, /*publication_converges=*/false));
  CHECK_FALSE(trace.released);
  // It attempts publication once and then defers -- it must not spin on the
  // same failing drain within one boundary.
  CHECK(trace.steps == std::vector<std::string>{"publish", "defer"});
}

// ---------------------------------------------------------------------------
// 2. Bridge classification, from production registration state.
// ---------------------------------------------------------------------------

TEST_CASE("Only RAM-backed color resolve bridges are released", "[cacheclear]") {
  REQUIRE_DEVICE();
  auto& h = H();

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  id<MTLTexture> null_texture = MakeNullTexture(h.device);
  REQUIRE(null_texture != nil);
  MetalTextureCache cache((__bridge void*)h.device, h.memory, nullptr, (__bridge void*)heap, 64);
  cache.SetNullTexture((__bridge void*)null_texture);
  cache.SetSubmissionProvider([]() { return uint64_t(9); });

  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:8
                                                        height:8
                                                     mipmapped:NO];
  id<MTLTexture> snapshot = [h.device newTextureWithDescriptor:desc];
  REQUIRE(snapshot != nil);

  constexpr uint32_t kRamBackedColor = 0x0D400000;
  constexpr uint32_t kUnbackedColor = 0x0D404000;
  constexpr uint32_t kRamBackedDepth = 0x0D408000;

  // A RAM-backed color dest: a writeback was enqueued, so guest RAM carries
  // the content and the bridge is derived.
  cache.RegisterResolvedTarget(kRamBackedColor, (__bridge void*)snapshot, 8, 8,
                               /*is_depth=*/false,
                               /*requires_fetch_swizzle=*/false, 0, 256);
  cache.MarkResolvedTargetRamBacked(kRamBackedColor, 8, 8, true, 0);
  // A color dest the writeback path skipped: the snapshot is the only copy.
  cache.RegisterResolvedTarget(kUnbackedColor, (__bridge void*)snapshot, 8, 8,
                               /*is_depth=*/false,
                               /*requires_fetch_swizzle=*/false, 0, 256);
  // A depth dest, RAM-backed or not, is served from the snapshot.
  cache.RegisterResolvedTarget(kRamBackedDepth, (__bridge void*)snapshot, 8, 8,
                               /*is_depth=*/true,
                               /*requires_fetch_swizzle=*/false, 0, 256);
  cache.MarkResolvedTargetRamBacked(kRamBackedDepth, 8, 8, true, 0);
  REQUIRE(cache.resolved_bridge_count() == 3u);

  std::vector<std::pair<uint32_t, void*>> released;
  cache.ReleaseDerivedResolveBridges(&released);

  // Exactly the derived one, reported with the texture THAT bridge named so
  // the render-target cache can verify identity before releasing anything.
  REQUIRE(released.size() == 1u);
  CHECK(released[0].first == kRamBackedColor);
  CHECK(released[0].second == (__bridge void*)snapshot);
  CHECK(cache.resolved_bridge_count() == 2u);
  // The retention invariant the command processor asserts after every clear.
  CHECK(cache.retained_bridges_are_authoritative());
}

TEST_CASE("Overwriting a snapshot in place starts a new content generation", "[cacheclear]") {
  // The case object identity cannot see. A resolve REUSES the existing
  // snapshot whenever it is large enough, so ResolveColorToSnapshot overwrites
  // the content with the pointer and dimensions unchanged, and
  // RegisterResolvedTarget runs before that resolve's writeback decision. If
  // "same texture, same extent" counted as the same generation, a skipped or
  // failed writeback would inherit the previous generation's RAM-backed
  // status and the clear would release content guest RAM never received.
  REQUIRE_DEVICE();
  auto& h = H();

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  id<MTLTexture> null_texture = MakeNullTexture(h.device);
  REQUIRE(null_texture != nil);
  MetalTextureCache cache((__bridge void*)h.device, h.memory, nullptr, (__bridge void*)heap, 64);
  cache.SetNullTexture((__bridge void*)null_texture);
  cache.SetSubmissionProvider([]() { return uint64_t(3); });

  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:8
                                                        height:8
                                                     mipmapped:NO];
  id<MTLTexture> a = [h.device newTextureWithDescriptor:desc];
  REQUIRE(a != nil);

  constexpr uint32_t kBase = 0x0D600000;

  // Generation 1: resolved into A, written back -> derived.
  cache.RegisterResolvedTarget(kBase, (__bridge void*)a, 8, 8, false, false, 0, 256);
  cache.MarkResolvedTargetRamBacked(kBase, 8, 8, true, 0);

  // Generation 2: the SAME object at the SAME extent, re-resolved in place,
  // with no writeback. Nothing about identity or size changed.
  cache.RegisterResolvedTarget(kBase, (__bridge void*)a, 8, 8, false, false, 0, 256);

  std::vector<std::pair<uint32_t, void*>> released;
  cache.ReleaseDerivedResolveBridges(&released);
  CHECK(released.empty());
  CHECK(cache.resolved_bridge_count() == 1u);
  CHECK(cache.retained_bridges_are_authoritative());

  // Once THIS generation's writeback lands, it is derived again.
  cache.MarkResolvedTargetRamBacked(kBase, 8, 8, true, 0);
  cache.ReleaseDerivedResolveBridges(&released);
  REQUIRE(released.size() == 1u);
  CHECK(released[0].first == kBase);
  CHECK(released[0].second == (__bridge void*)a);
}

TEST_CASE("A newer snapshot generation is not released on an older writeback", "[cacheclear]") {
  // The same hazard where the snapshot object itself is replaced (a resolve
  // needing a larger texture). ram_backed is set once and never cleared, and
  // it drives the SERVE path, so it cannot be repurposed as "this snapshot's
  // content is in RAM".
  REQUIRE_DEVICE();
  auto& h = H();

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  id<MTLTexture> null_texture = MakeNullTexture(h.device);
  REQUIRE(null_texture != nil);
  MetalTextureCache cache((__bridge void*)h.device, h.memory, nullptr, (__bridge void*)heap, 64);
  cache.SetNullTexture((__bridge void*)null_texture);
  cache.SetSubmissionProvider([]() { return uint64_t(3); });

  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:8
                                                        height:8
                                                     mipmapped:NO];
  id<MTLTexture> first = [h.device newTextureWithDescriptor:desc];
  id<MTLTexture> second = [h.device newTextureWithDescriptor:desc];
  REQUIRE(first != nil);
  REQUIRE(second != nil);

  constexpr uint32_t kBase = 0x0D500000;

  // Generation 1: resolved, written back. Derived -- releasable.
  cache.RegisterResolvedTarget(kBase, (__bridge void*)first, 8, 8, false, false, 0, 256);
  cache.MarkResolvedTargetRamBacked(kBase, 8, 8, true, 0);
  {
    std::vector<std::pair<uint32_t, void*>> released;
    MetalTextureCache probe((__bridge void*)h.device, h.memory, nullptr, (__bridge void*)heap, 64);
    // (sanity: the same state in a fresh cache classifies as derived)
    probe.SetNullTexture((__bridge void*)null_texture);
    probe.SetSubmissionProvider([]() { return uint64_t(3); });
    probe.RegisterResolvedTarget(kBase, (__bridge void*)first, 8, 8, false, false, 0, 256);
    probe.MarkResolvedTargetRamBacked(kBase, 8, 8, true, 0);
    probe.ReleaseDerivedResolveBridges(&released);
    REQUIRE(released.size() == 1u);
  }

  // Generation 2: a new snapshot texture at the same base, NO writeback.
  cache.RegisterResolvedTarget(kBase, (__bridge void*)second, 8, 8, false, false, 0, 256);

  std::vector<std::pair<uint32_t, void*>> released;
  cache.ReleaseDerivedResolveBridges(&released);

  // Nothing may be released: this generation's content is not in guest RAM.
  CHECK(released.empty());
  CHECK(cache.resolved_bridge_count() == 1u);
  CHECK(cache.retained_bridges_are_authoritative());

  // Once THIS generation is written back, it becomes derived again.
  cache.MarkResolvedTargetRamBacked(kBase, 8, 8, true, 0);
  cache.ReleaseDerivedResolveBridges(&released);
  REQUIRE(released.size() == 1u);
  CHECK(released[0].first == kBase);
  // And it names the CURRENT texture, not the one the first writeback covered.
  CHECK(released[0].second == (__bridge void*)second);
}

// ---------------------------------------------------------------------------
// 3. Post-clear state of the real legacy cache.
// ---------------------------------------------------------------------------

TEST_CASE("A clear releases cached textures, slots, and descriptors", "[cacheclear]") {
  REQUIRE_DEVICE();
  auto& h = H();

  constexpr uint32_t kBaseA = 0x0D200000;
  constexpr uint32_t kBaseB = 0x0D204000;
  constexpr uint32_t kBaseC = 0x0D208000;
  FillMemory(*h.memory, kBaseA, kExtent, 0x11);
  FillMemory(*h.memory, kBaseB, kExtent, 0x22);
  FillMemory(*h.memory, kBaseC, kExtent, 0x33);

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  id<MTLTexture> null_texture = MakeNullTexture(h.device);
  REQUIRE(null_texture != nil);
  auto* init_entries = static_cast<IRDescriptorTableEntry*>(heap.contents);
  for (uint32_t i = 0; i < 64; ++i) {
    IRDescriptorTableSetTexture(&init_entries[i], null_texture, 0.0f, 0);
  }
  MetalTextureCache cache((__bridge void*)h.device, h.memory, nullptr, (__bridge void*)heap, 64);
  cache.SetNullTexture((__bridge void*)null_texture);
  // The submission every retirement is tagged with. Deliberately far ahead of
  // anything "completed": a clear must drain the retire lists regardless,
  // because it only runs when every submission has finished.
  cache.SetSubmissionProvider([]() { return uint64_t(500); });

  uint32_t slot_a = cache.GetTextureSlot(MakeFetch(kBaseA), false);
  uint32_t slot_b = cache.GetTextureSlot(MakeFetch(kBaseB), false);
  uint32_t slot_c = cache.GetTextureSlot(MakeFetch(kBaseC), false);
  REQUIRE(slot_a != 0u);
  REQUIRE(slot_b != 0u);
  REQUIRE(slot_c != 0u);
  REQUIRE(cache.cached_texture_count() == 3u);
  REQUIRE_FALSE(DescriptorIsNull(heap, slot_a, null_texture));

  // Evict one through the production path so a retire list is non-empty at
  // clear time -- the state where "drain on clear" actually matters.
  REQUIRE(cache.InvalidateRange(kBaseC, kExtent,
                                MetalTextureCache::InvalidationScope::kAllRepresentations) == 1u);
  // Content revalidation parks the evicted texture (its slot is retired as
  // before); the retire list receives it only once its bytes prove changed.
  // Change them, then re-fetch: the parked texture retires and a fresh one
  // uploads, which is the non-empty retire list this clear case needs.
  REQUIRE(cache.parked_texture_count() == 1u);
  REQUIRE(cache.retired_texture_count() == 0u);
  REQUIRE(cache.slots_stale() == 1u);
  REQUIRE(cache.cached_texture_count() == 2u);
  FillMemory(*h.memory, kBaseC, kExtent, 0x44);
  REQUIRE(cache.GetTextureSlot(MakeFetch(kBaseC), false) != 0u);
  REQUIRE(cache.parked_texture_count() == 0u);
  REQUIRE(cache.retired_texture_count() == 1u);
  REQUIRE(cache.cached_texture_count() == 3u);

  // A slot+view retired exactly the way MetalSharedTextureCache's
  // ReleaseBindingSlot retires them: through this adapter, into these pools.
  // The clear must drain these too, or the shared subclass's release would be
  // stranded in a pool that was just emptied.
  uint32_t external_slot = cache.AllocateExternalSlot();
  REQUIRE(external_slot != 0u);
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:4
                                                        height:4
                                                     mipmapped:NO];
  id<MTLTexture> external_texture = [h.device newTextureWithDescriptor:desc];
  REQUIRE(external_texture != nil);
  void* external_bridge = (__bridge_retained void*)external_texture;
  cache.BindExternalSlot(external_slot, external_bridge);
  REQUIRE_FALSE(DescriptorIsNull(heap, external_slot, null_texture));
  cache.RetireExternalSlot(external_slot);
  cache.RetireExternalTexture(external_bridge);
  REQUIRE(cache.retired_texture_count() == 2u);

  uint32_t slots_used_before = cache.slots_used();
  uint32_t free_before = cache.slots_free();

  cache.ClearCache();

  // Nothing cached, nothing queued.
  CHECK(cache.cached_texture_count() == 0u);
  CHECK(cache.retired_texture_count() == 0u);
  CHECK(cache.slots_stale() == 0u);
  // Every slot came back, so allocation returns toward its baseline. The
  // allocator high-water mark itself does not move, because the retained
  // resolve bridge keeps its slots and recycling them would alias.
  CHECK(cache.slots_used() == slots_used_before);
  CHECK(cache.slots_free() > free_before);
  // No freed slot may still describe a texture that no longer exists, and it
  // must hold the valid null descriptor rather than an empty one.
  CHECK(DescriptorIsNull(heap, slot_a, null_texture));
  CHECK(DescriptorIsNull(heap, slot_b, null_texture));
  CHECK(DescriptorIsNull(heap, external_slot, null_texture));

  // And the cache is usable again: a clear is a lifetime event, not a reset.
  uint32_t reloaded = cache.GetTextureSlot(MakeFetch(kBaseA), false);
  CHECK(reloaded != 0u);
  CHECK(cache.cached_texture_count() == 1u);
}

TEST_CASE("Repeated clears neither leak nor grow the slot allocator", "[cacheclear]") {
  REQUIRE_DEVICE();
  auto& h = H();

  constexpr uint32_t kBase = 0x0D300000;
  FillMemory(*h.memory, kBase, kExtent, 0x44);

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  id<MTLTexture> null_texture = MakeNullTexture(h.device);
  REQUIRE(null_texture != nil);
  MetalTextureCache cache((__bridge void*)h.device, h.memory, nullptr, (__bridge void*)heap, 64);
  cache.SetNullTexture((__bridge void*)null_texture);
  cache.SetSubmissionProvider([]() { return uint64_t(1); });

  uint32_t high_water = 0;
  for (int i = 0; i < 8; ++i) {
    REQUIRE(cache.GetTextureSlot(MakeFetch(kBase), false) != 0u);
    REQUIRE(cache.cached_texture_count() == 1u);
    cache.ClearCache();
    REQUIRE(cache.cached_texture_count() == 0u);
    if (i == 0) {
      high_water = cache.slots_used();
    } else {
      // Slots are recycled through the free list rather than newly allocated,
      // so repeated clear/repopulate cycles cannot walk the heap to
      // exhaustion.
      CHECK(cache.slots_used() == high_water);
    }
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
