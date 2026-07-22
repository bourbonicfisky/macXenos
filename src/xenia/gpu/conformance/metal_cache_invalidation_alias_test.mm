/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Tasks 5.2-5.3: factual cache alias, lifetime, and writer-matrix coverage.
// Every legacy entry is created through MetalTextureCache::GetTextureSlot, and
// the production InvalidateRange scan or SharedMemory watch fan-out is the only
// overlap oracle. Real Metal textures and bindless descriptor slots are used.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/texture_cache.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kPage = 4096;
constexpr uint32_t kLegacyTextureExtent = 0x2000;
constexpr uint32_t kTarget = 0x0D000000;
constexpr uint32_t kMipCarrierBase = 0x0E000000;

void EnsureAllocated(Memory& memory, uint32_t base, uint32_t length) {
  uint32_t first = base & ~(kPage - 1);
  uint32_t last = (base + length + kPage - 1) & ~(kPage - 1);
  memory.GetPhysicalHeap()->AllocFixed(
      first, last - first, kPage,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite);
}

void FillMemory(Memory& memory, uint32_t base, uint32_t length, uint8_t seed) {
  EnsureAllocated(memory, base, length);
  uint8_t* bytes = memory.TranslatePhysical<uint8_t*>(base);
  REQUIRE(bytes != nullptr);
  for (uint32_t i = 0; i < length; ++i) {
    bytes[i] = uint8_t(seed + i * 13u + (i >> 8));
  }
}

xenos::xe_gpu_texture_fetch_t MakeLinearFetch(
    uint32_t base, xenos::TextureFormat format, uint32_t swizzle = 0x688u,
    uint32_t mip_base = 0, uint32_t mip_max_level = 0) {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.type = xenos::FetchConstantType::kTexture;
  fetch.sign_x = xenos::TextureSign::kUnsigned;
  fetch.sign_y = xenos::TextureSign::kUnsigned;
  fetch.sign_z = xenos::TextureSign::kUnsigned;
  fetch.sign_w = xenos::TextureSign::kUnsigned;
  fetch.pitch = 2;  // 64 texels (pitch is in 32-texel units).
  fetch.tiled = 0;
  fetch.format = format;
  fetch.endianness = xenos::Endian::kNone;
  fetch.base_address = base >> 12;
  fetch.size_2d.width = 63;
  fetch.size_2d.height = 31;
  fetch.swizzle = swizzle;
  fetch.mip_min_level = 0;
  fetch.mip_max_level = mip_max_level;
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  fetch.packed_mips = 0;
  fetch.mip_address = mip_base >> 12;
  return fetch;
}

// Byte transport is deliberately trivial; page validity, watch registration,
// page rounding, one-shot firing, and re-arming all remain the real
// SharedMemory / TextureCache implementation used by MetalSharedTextureCache.
class AliasSharedMemory final : public SharedMemory {
 public:
  explicit AliasSharedMemory(Memory& memory) : SharedMemory(memory) {}
  bool Initialize() { return InitializeCommon(); }
  uint32_t PageSize() const { return 1u << page_size_log2(); }

 protected:
  bool UploadRanges(const std::pair<uint32_t, uint32_t>* ranges,
                    uint32_t range_count) override {
    for (uint32_t i = 0; i < range_count; ++i) {
      MakeRangeValid(ranges[i].first << page_size_log2(),
                     ranges[i].second << page_size_log2(), false);
    }
    return true;
  }
};

class WatchTextureCache final : public TextureCache {
 public:
  class WatchTexture final : public TextureCache::Texture {
   public:
    WatchTexture(WatchTextureCache& cache, const TextureKey& key)
        : Texture(cache, key) {}
  };

  WatchTextureCache(const RegisterFile& register_file, SharedMemory& shared)
      : TextureCache(register_file, shared, 1, 1) {}

  WatchTexture* Load(const xenos::xe_gpu_texture_fetch_t& fetch) {
    TextureKey key;
    uint8_t signs = 0;
    BindingInfoFromFetchConstant(fetch, key, &signs);
    if (!key.is_valid) {
      return nullptr;
    }
    Texture* texture = FindOrCreateTexture(key);
    if (!texture) {
      return nullptr;
    }
    Texture* one[] = {texture};
    LoadTexturesData(one, 1);
    return static_cast<WatchTexture*>(texture);
  }

  void Reload(WatchTexture* texture) {
    Texture* one[] = {texture};
    LoadTexturesData(one, 1);
  }

  uint32_t load_count() const { return load_count_; }

 protected:
  uint32_t GetHostFormatSwizzle(TextureKey) const override {
    return xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }
  uint32_t GetMaxHostTextureWidthHeight(
      xenos::DataDimension) const override {
    return xenos::kTexture2DCubeMaxWidthHeight;
  }
  uint32_t GetMaxHostTextureDepthOrArraySize(
      xenos::DataDimension) const override {
    return 1;
  }
  std::unique_ptr<Texture> CreateTexture(TextureKey key) override {
    return std::make_unique<WatchTexture>(*this, key);
  }
  bool LoadTextureDataFromResidentMemoryImpl(Texture&, bool, bool) override {
    ++load_count_;
    return true;
  }

 private:
  uint32_t load_count_ = 0;
};

xenos::xe_gpu_texture_fetch_t MakeSharedRouteFetch(uint32_t base,
                                                   uint32_t width) {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.type = xenos::FetchConstantType::kTexture;
  fetch.sign_x = xenos::TextureSign::kUnsigned;
  fetch.sign_y = xenos::TextureSign::kUnsigned;
  fetch.sign_z = xenos::TextureSign::kUnsigned;
  fetch.sign_w = xenos::TextureSign::kUnsigned;
  fetch.pitch = 1;  // 32 texels.
  fetch.tiled = 1;
  fetch.format = xenos::TextureFormat::k_8_8_8_8;
  fetch.endianness = xenos::Endian::kNone;
  fetch.base_address = base >> 12;
  fetch.size_2d.width = width - 1;
  fetch.size_2d.height = 15;
  fetch.swizzle = 0x688u;
  fetch.mip_max_level = 0;
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  return fetch;
}

class RealCacheRamWriteTarget final : public RamWritePublicationTarget {
 public:
  RealCacheRamWriteTarget(SharedMemory& shared, MetalTextureCache& legacy)
      : shared_(shared), legacy_(legacy) {}

  bool InvalidateSharedPages(uint32_t base, uint32_t length,
                             bool exact_range) override {
    shared_.MemoryInvalidationCallback(base, length, exact_range);
    return true;
  }
  bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                uint32_t* invalidated_entries) override {
    *invalidated_entries = legacy_.InvalidateRange(base, length);
    return true;
  }

 private:
  SharedMemory& shared_;
  MetalTextureCache& legacy_;
};

struct LegacyCacheHarness {
  id<MTLDevice> device = nil;
  id<MTLBuffer> view_heap = nil;
  Memory* memory = nullptr;
  std::unique_ptr<MetalTextureCache> cache;
  bool ok = false;

  LegacyCacheHarness() {
    device = MTLCreateSystemDefaultDevice();
    memory = conformance::TestPhysicalMemory();
    if (!device || !memory) {
      return;
    }
    view_heap = [device newBufferWithLength:16 * 1024
                                   options:MTLResourceStorageModeShared];
    if (!view_heap) {
      return;
    }
    cache = std::make_unique<MetalTextureCache>(
        (__bridge void*)device, memory, nullptr, (__bridge void*)view_heap, 32);
    ok = cache != nullptr;
  }
};

LegacyCacheHarness& H() {
  static LegacyCacheHarness harness;
  return harness;
}

#define REQUIRE_DEVICE()        \
  if (!H().ok) {                \
    SUCCEED("no Metal device"); \
    return;                     \
  }

TEST_CASE(
    "Cache alias 5.2: real legacy keys and mip placement retire together "
    "without premature slot reuse",
    "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();

  FillMemory(*h.memory, kTarget, 0x4000, 0x11);
  FillMemory(*h.memory, kMipCarrierBase, 0x2000, 0x31);
  FillMemory(*h.memory, kTarget + 0x10000, 0x2000, 0x51);
  FillMemory(*h.memory, kTarget + 0x20000, 0x2000, 0x61);
  FillMemory(*h.memory, kTarget + 0x30000, 0x2000, 0x71);

  auto alias_a = MakeLinearFetch(kTarget, xenos::TextureFormat::k_8_8_8_8);
  auto alias_b = alias_a;
  auto alias_c =
      MakeLinearFetch(kTarget, xenos::TextureFormat::k_8_8_8_8, 0x60Au);
  auto alias_d = MakeLinearFetch(kTarget, xenos::TextureFormat::k_32_FLOAT);
  auto mip_carrier = MakeLinearFetch(kMipCarrierBase,
                                     xenos::TextureFormat::k_8_8_8_8,
                                     0x688u, kTarget, 1);
  auto adjacent = MakeLinearFetch(kTarget + 0x2000,
                                  xenos::TextureFormat::k_8_8_8_8);
  auto disjoint = MakeLinearFetch(kTarget + 0x10000,
                                  xenos::TextureFormat::k_8_8_8_8);

  std::vector<uint32_t> slots;
  slots.push_back(h.cache->GetTextureSlot(alias_a, false));
  slots.push_back(h.cache->GetTextureSlot(alias_b, true));
  slots.push_back(h.cache->GetTextureSlot(alias_c, false));
  slots.push_back(h.cache->GetTextureSlot(alias_d, false));
  slots.push_back(h.cache->GetTextureSlot(mip_carrier, false));
  slots.push_back(h.cache->GetTextureSlot(adjacent, false));
  slots.push_back(h.cache->GetTextureSlot(disjoint, false));

  for (uint32_t slot : slots) {
    REQUIRE(slot != 0u);
  }
  auto sorted_slots = slots;
  std::sort(sorted_slots.begin(), sorted_slots.end());
  REQUIRE(std::adjacent_find(sorted_slots.begin(), sorted_slots.end()) ==
          sorted_slots.end());
  REQUIRE(h.cache->ConsumeUploadCount() == 7u);

  std::vector<void*> live;
  h.cache->AppendAllTextures(live);
  REQUIRE(live.size() == 7u);

  REQUIRE(h.cache->GetTextureSlot(adjacent, false) == slots[5]);
  REQUIRE(h.cache->GetTextureSlot(disjoint, false) == slots[6]);
  REQUIRE(h.cache->ConsumeUploadCount() == 0u);

  h.cache->SetSubmissionProvider([]() { return uint64_t(7); });
  REQUIRE(h.cache->InvalidateRange(
              kTarget + 0x100, 0x20,
              MetalTextureCache::InvalidationScope::kAllRepresentations) ==
          5u);

  live.clear();
  h.cache->AppendAllTextures(live);
  REQUIRE(live.size() == 2u);
  REQUIRE(h.cache->slots_stale() == 5u);
  REQUIRE(h.cache->slots_free() == 0u);

  // A new entry submitted while the retired aliases are still in flight must
  // not receive any of their slots, even though those keys are no longer live.
  auto before_reclaim =
      MakeLinearFetch(kTarget + 0x20000, xenos::TextureFormat::k_8_8_8_8);
  uint32_t before_reclaim_slot =
      h.cache->GetTextureSlot(before_reclaim, false);
  REQUIRE(before_reclaim_slot != 0u);
  REQUIRE(std::find(slots.begin(), slots.begin() + 5,
                    before_reclaim_slot) == slots.begin() + 5);
  REQUIRE(h.cache->ConsumeUploadCount() == 1u);

  h.cache->ReclaimStaleSlots(6);
  REQUIRE(h.cache->slots_stale() == 5u);
  REQUIRE(h.cache->slots_free() == 0u);
  h.cache->ReclaimStaleSlots(7);
  REQUIRE(h.cache->slots_stale() == 0u);
  REQUIRE(h.cache->slots_free() == 5u);

  // Once submission 7 completes, a new entry must be able to reuse one of the
  // retired alias slots. This observes the allocator, not only its counters.
  auto after_reclaim =
      MakeLinearFetch(kTarget + 0x30000, xenos::TextureFormat::k_8_8_8_8);
  uint32_t after_reclaim_slot = h.cache->GetTextureSlot(after_reclaim, false);
  REQUIRE(after_reclaim_slot != 0u);
  REQUIRE(std::find(slots.begin(), slots.begin() + 5,
                    after_reclaim_slot) != slots.begin() + 5);
  REQUIRE(h.cache->slots_free() == 4u);
  REQUIRE(h.cache->ConsumeUploadCount() == 1u);

  REQUIRE(h.cache->GetTextureSlot(adjacent, false) == slots[5]);
  REQUIRE(h.cache->GetTextureSlot(disjoint, false) == slots[6]);
  REQUIRE(h.cache->ConsumeUploadCount() == 0u);

  // Leave the process-wide harness with no live or retired entries so this
  // case is order-independent with the resolved-target case below.
  REQUIRE(h.cache->InvalidateRange(
              kTarget, 0x40000,
              MetalTextureCache::InvalidationScope::kAllRepresentations) ==
          4u);
  h.cache->ReclaimStaleSlots(UINT64_MAX);
  REQUIRE(h.cache->slots_stale() == 0u);
}

TEST_CASE(
    "Cache alias 5.2: scoped resolve invalidation preserves its bridge while "
    "an independent writer retires it",
    "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBridgeBase = 0x0F000000;

  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                MTLPixelFormatRGBA8Unorm
                                                     width:16
                                                    height:16
                                                 mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> bridge = [h.device newTextureWithDescriptor:descriptor];
  REQUIRE(bridge != nil);

  h.cache->SetSubmissionProvider([]() { return uint64_t(11); });
  uint32_t stale_before = h.cache->slots_stale();
  h.cache->RegisterResolvedTarget(kBridgeBase, (__bridge void*)bridge, 16, 16,
                                  false, false, kBridgeBase, kPage);
  std::vector<void*> live;
  h.cache->AppendAllTextures(live);
  REQUIRE(std::find(live.begin(), live.end(), (__bridge void*)bridge) !=
          live.end());

  REQUIRE(h.cache->InvalidateRange(
              kBridgeBase + 4, 4,
              MetalTextureCache::InvalidationScope::kRamUploadsOnly) == 0u);
  live.clear();
  h.cache->AppendAllTextures(live);
  REQUIRE(std::find(live.begin(), live.end(), (__bridge void*)bridge) !=
          live.end());
  REQUIRE(h.cache->slots_stale() == stale_before);

  REQUIRE(h.cache->InvalidateRange(
              kBridgeBase + 4, 4,
              MetalTextureCache::InvalidationScope::kAllRepresentations) ==
          1u);
  live.clear();
  h.cache->AppendAllTextures(live);
  REQUIRE(std::find(live.begin(), live.end(), (__bridge void*)bridge) ==
          live.end());
  REQUIRE(h.cache->slots_stale() == stale_before + 1u);

  h.cache->ReclaimStaleSlots(10);
  REQUIRE(h.cache->slots_stale() == stale_before + 1u);
  h.cache->ReclaimStaleSlots(11);
  REQUIRE(h.cache->slots_stale() == stale_before);
}

TEST_CASE(
    "Cache alias 5.2: partial-page and GPU-written notifications fan out to "
    "real legacy and shared watch subscribers",
    "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x11000000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  uint32_t page = shared->PageSize();
  // Shared-memory aliases are page-scoped, while the real 64x32 RGBA8 legacy
  // fetch records an 8 KiB range. Keep the two adjacency boundaries factual on
  // both 4 KiB Intel and 16 KiB Apple Silicon hosts.
  uint32_t legacy_adjacent_base =
      kBase + std::max(page, kLegacyTextureExtent);
  uint32_t legacy_disjoint_base = legacy_adjacent_base + 0x10000;
  FillMemory(*h.memory, kBase, page * 3, 0x71);
  FillMemory(*h.memory, legacy_adjacent_base, kLegacyTextureExtent, 0x81);
  FillMemory(*h.memory, legacy_disjoint_base, kLegacyTextureExtent, 0x91);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  MetalTextureCache legacy((__bridge void*)h.device, h.memory, shared.get(),
                           (__bridge void*)heap, 64);
  legacy.SetSubmissionProvider([]() { return uint64_t(31); });

  auto* shared_a = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  auto* shared_b = shared_cache.Load(MakeSharedRouteFetch(kBase, 32));
  auto* shared_adjacent =
      shared_cache.Load(MakeSharedRouteFetch(kBase + page, 16));
  auto* shared_disjoint =
      shared_cache.Load(MakeSharedRouteFetch(kBase + page * 2, 16));
  REQUIRE(shared_a != nullptr);
  REQUIRE(shared_b != nullptr);
  REQUIRE(shared_adjacent != nullptr);
  REQUIRE(shared_disjoint != nullptr);
  REQUIRE_FALSE(shared_a->base_outdated_lockless());
  REQUIRE_FALSE(shared_b->base_outdated_lockless());
  REQUIRE_FALSE(shared_adjacent->base_outdated_lockless());
  REQUIRE_FALSE(shared_disjoint->base_outdated_lockless());

  auto legacy_a = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  auto legacy_b = legacy_a;
  auto legacy_adjacent =
      MakeLinearFetch(legacy_adjacent_base, xenos::TextureFormat::k_8_8_8_8);
  auto legacy_disjoint =
      MakeLinearFetch(legacy_disjoint_base, xenos::TextureFormat::k_8_8_8_8);
  REQUIRE(legacy.GetTextureSlot(legacy_a, false) != 0u);
  REQUIRE(legacy.GetTextureSlot(legacy_b, true) != 0u);
  uint32_t adjacent_slot = legacy.GetTextureSlot(legacy_adjacent, false);
  uint32_t disjoint_slot = legacy.GetTextureSlot(legacy_disjoint, false);
  REQUIRE(adjacent_slot != 0u);
  REQUIRE(disjoint_slot != 0u);
  REQUIRE(legacy.ConsumeUploadCount() == 4u);

  struct GlobalProbe {
    uint32_t calls = 0;
    uint32_t first = 0;
    uint32_t last = 0;
  } probe;
  auto probe_callback = [](const global_unique_lock_type&, void* context,
                           uint32_t first, uint32_t last, bool) {
    auto* value = static_cast<GlobalProbe*>(context);
    ++value->calls;
    value->first = first;
    value->last = last;
  };
  auto probe_watch = shared->RegisterGlobalWatch(probe_callback, &probe);

  // The logical write is only 32 bytes, but SharedMemory's real page fan-out
  // invalidates both canonical shared entries whose watched extents occupy the
  // same host page. Page-adjacent and disjoint entries survive.
  shared->MemoryInvalidationCallback(kBase + 0x100, 0x20, true);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());
  REQUIRE_FALSE(shared_adjacent->base_outdated_lockless());
  REQUIRE_FALSE(shared_disjoint->base_outdated_lockless());
  REQUIRE(probe.calls == 1u);
  REQUIRE(probe.first == kBase);
  REQUIRE(probe.last == kBase + page - 1);

  // The legacy callback queues only. A real cache lookup drains that queue on
  // the GPU-thread path and retires both aliased keys.
  REQUIRE(legacy.GetTextureSlot(legacy_adjacent, false) == adjacent_slot);
  std::vector<void*> legacy_live;
  legacy.AppendAllTextures(legacy_live);
  REQUIRE(legacy_live.size() == 2u);
  REQUIRE(legacy.GetTextureSlot(legacy_disjoint, false) == disjoint_slot);

  shared_cache.Reload(shared_a);
  shared_cache.Reload(shared_b);
  REQUIRE_FALSE(shared_a->base_outdated_lockless());
  REQUIRE_FALSE(shared_b->base_outdated_lockless());
  REQUIRE(legacy.GetTextureSlot(legacy_a, false) != 0u);
  REQUIRE(legacy.GetTextureSlot(legacy_b, true) != 0u);
  REQUIRE(legacy.ConsumeUploadCount() == 2u);

  // GPU-written publication uses the same real FireWatches path and must
  // reach both subscriber classes before marking the shared bytes valid.
  shared->RangeWrittenByGpu(kBase + 0x200, 0x20);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());
  REQUIRE_FALSE(shared_adjacent->base_outdated_lockless());
  REQUIRE_FALSE(shared_disjoint->base_outdated_lockless());
  REQUIRE(probe.calls == 2u);
  REQUIRE(legacy.GetTextureSlot(legacy_adjacent, false) == adjacent_slot);
  legacy_live.clear();
  legacy.AppendAllTextures(legacy_live);
  REQUIRE(legacy_live.size() == 2u);

  shared->UnregisterGlobalWatch(probe_watch);
  legacy.ReclaimStaleSlots(UINT64_MAX);
}

TEST_CASE(
    "Cache alias 5.2: trace restore direct legacy leg survives watch teardown "
    "and shares publication with range subscribers",
    "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x12000000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  uint32_t page = shared->PageSize();
  uint32_t legacy_adjacent_base =
      kBase + std::max(page, kLegacyTextureExtent);
  FillMemory(*h.memory, kBase, page * 2, 0x91);
  FillMemory(*h.memory, legacy_adjacent_base, kLegacyTextureExtent, 0xA1);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  MetalTextureCache legacy((__bridge void*)h.device, h.memory, shared.get(),
                           (__bridge void*)heap, 64);
  legacy.SetSubmissionProvider([]() { return uint64_t(41); });

  auto* shared_a = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  auto* shared_b = shared_cache.Load(MakeSharedRouteFetch(kBase, 32));
  auto* shared_adjacent =
      shared_cache.Load(MakeSharedRouteFetch(kBase + page, 16));
  REQUIRE(shared_a != nullptr);
  REQUIRE(shared_b != nullptr);
  REQUIRE(shared_adjacent != nullptr);

  auto legacy_a = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  auto legacy_b = legacy_a;
  auto legacy_adjacent =
      MakeLinearFetch(legacy_adjacent_base, xenos::TextureFormat::k_8_8_8_8);
  REQUIRE(legacy.GetTextureSlot(legacy_a, false) != 0u);
  REQUIRE(legacy.GetTextureSlot(legacy_b, true) != 0u);
  uint32_t adjacent_slot = legacy.GetTextureSlot(legacy_adjacent, false);
  REQUIRE(adjacent_slot != 0u);
  REQUIRE(legacy.ConsumeUploadCount() == 3u);

  legacy.SetTracePlayback();
  shared->MemoryInvalidationCallback(kBase + 0x100, 0x20, true);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());
  REQUIRE_FALSE(shared_adjacent->base_outdated_lockless());

  // The legacy global watch is gone and trace mode deliberately doesn't drain
  // queued live invalidations. The negative control proves all three entries
  // remain until the typed trace publication executes its direct legacy leg.
  REQUIRE(legacy.GetTextureSlot(legacy_adjacent, false) == adjacent_slot);
  std::vector<void*> legacy_live;
  legacy.AppendAllTextures(legacy_live);
  REQUIRE(legacy_live.size() == 3u);

  shared_cache.Reload(shared_a);
  shared_cache.Reload(shared_b);
  auto publication = DecideRamWritePublication(WriterKind::kTraceRestore,
                                                kBase + 0x100, 0x20);
  REQUIRE(publication.has_value());
  RealCacheRamWriteTarget target(*shared, legacy);
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE(outcome.shared_invalidation_executed);
  REQUIRE(outcome.legacy_invalidation_executed);
  REQUIRE(outcome.invalidated_entries == 2u);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());
  REQUIRE_FALSE(shared_adjacent->base_outdated_lockless());

  legacy_live.clear();
  legacy.AppendAllTextures(legacy_live);
  REQUIRE(legacy_live.size() == 1u);
  REQUIRE(legacy.GetTextureSlot(legacy_adjacent, false) == adjacent_slot);

  legacy.ReclaimStaleSlots(UINT64_MAX);
}

// Task 5.3 compares actual affected entry sets. Direct-only reference caches
// and watch-driven caches start with the same real Metal entries; survivor
// membership and retirement counts are compared after each production event.
// SharedMemory notifications are page-granular, so a watch set may be a safe
// conservative superset of the exact direct set on a 16 KiB host. Resolve and
// trace are different: their legacy-global watch set is empty by design, so
// their scoped direct hooks remain independently required.

id<MTLTexture> MakeBridgeTexture(id<MTLDevice> device) {
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:16
                                                        height:16
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  return [device newTextureWithDescriptor:descriptor];
}

std::vector<void*> LiveTextures(const MetalTextureCache& cache) {
  std::vector<void*> live;
  cache.AppendAllTextures(live);
  return live;
}

bool ContainsTexture(const std::vector<void*>& live, void* texture) {
  return std::find(live.begin(), live.end(), texture) != live.end();
}

class MatrixMemexportConvergenceTarget final : public MemexportConvergenceTarget {
 public:
  MatrixMemexportConvergenceTarget(Memory& memory, MetalTextureCache* legacy)
      : memory_(memory), legacy_(legacy) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length, uint32_t* invalidated_entries) override {
    uint8_t* dest = memory_.TranslatePhysical<uint8_t*>(base);
    if (!dest) {
      return false;
    }
    for (uint32_t i = 0; i < length; ++i) {
      dest[i] = uint8_t(0xC0u + i);
    }
    *invalidated_entries = legacy_ ? legacy_->InvalidateRange(base, length) : 0;
    return true;
  }

 private:
  Memory& memory_;
  MetalTextureCache* legacy_;
};

TEST_CASE("Cache alias 5.3: guest CPU and memexport publication watch sets contain "
          "the exact direct set and expose only page-granular conservative extras",
          "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x13000000;
  constexpr uint32_t kAdjacent = kBase + kLegacyTextureExtent;
  constexpr uint32_t kDisjoint = kBase + 0x10000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  uint32_t page = shared->PageSize();
  FillMemory(*h.memory, kBase, 0x12000, 0xB1);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  auto* shared_a = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  auto* shared_b = shared_cache.Load(MakeSharedRouteFetch(kBase, 32));
  REQUIRE(shared_a != nullptr);
  REQUIRE(shared_b != nullptr);

  id<MTLBuffer> watch_heap = [h.device newBufferWithLength:16 * 1024
                                                   options:MTLResourceStorageModeShared];
  id<MTLBuffer> direct_heap = [h.device newBufferWithLength:16 * 1024
                                                    options:MTLResourceStorageModeShared];
  REQUIRE(watch_heap != nil);
  REQUIRE(direct_heap != nil);
  MetalTextureCache watch_cache((__bridge void*)h.device, h.memory, shared.get(),
                                (__bridge void*)watch_heap, 64);
  MetalTextureCache direct_cache((__bridge void*)h.device, h.memory, nullptr,
                                 (__bridge void*)direct_heap, 64);
  watch_cache.SetSubmissionProvider([]() { return uint64_t(51); });
  direct_cache.SetSubmissionProvider([]() { return uint64_t(51); });

  auto alias = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  auto adjacent = MakeLinearFetch(kAdjacent, xenos::TextureFormat::k_8_8_8_8);
  auto disjoint = MakeLinearFetch(kDisjoint, xenos::TextureFormat::k_8_8_8_8);
  auto seed = [&](MetalTextureCache& cache) {
    std::vector<uint32_t> slots = {
        cache.GetTextureSlot(alias, false), cache.GetTextureSlot(alias, true),
        cache.GetTextureSlot(adjacent, false), cache.GetTextureSlot(disjoint, false)};
    for (uint32_t slot : slots) {
      REQUIRE(slot != 0u);
    }
    REQUIRE(cache.ConsumeUploadCount() == 4u);
    return slots;
  };
  auto watch_slots = seed(watch_cache);
  auto direct_slots = seed(direct_cache);

  id<MTLTexture> bridge = MakeBridgeTexture(h.device);
  REQUIRE(bridge != nil);
  watch_cache.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                     kLegacyTextureExtent);
  direct_cache.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                      kLegacyTextureExtent);
  REQUIRE(LiveTextures(watch_cache).size() == 5u);
  REQUIRE(LiveTextures(direct_cache).size() == 5u);

  SECTION("guest CPU physical callback") {
    auto policy = RamWritePolicyFor(WriterKind::kGuestCpu);
    REQUIRE_FALSE(policy.execute_shared_invalidation);
    REQUIRE_FALSE(policy.execute_legacy_invalidation);
    shared->MemoryInvalidationCallback(kBase + 0x100, 0x20, true);
  }
  SECTION("memexport GPU publication") { shared->RangeWrittenByGpu(kBase + 0x100, 0x20); }

  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());

  // The exact reference removes the two keys and overlapping bridge, but not
  // the exactly adjacent or disjoint entries.
  REQUIRE(direct_cache.InvalidateRange(kBase + 0x100, 0x20,
                                       MetalTextureCache::InvalidationScope::kAllRepresentations) ==
          3u);
  REQUIRE(direct_cache.GetTextureSlot(adjacent, false) == direct_slots[2]);
  REQUIRE(direct_cache.GetTextureSlot(disjoint, false) == direct_slots[3]);
  auto direct_live = LiveTextures(direct_cache);
  REQUIRE(direct_live.size() == 2u);
  REQUIRE_FALSE(ContainsTexture(direct_live, (__bridge void*)bridge));

  // A disjoint lookup drains the real legacy global queue without recreating
  // a target alias. On 16 KiB hosts the page also covers the exact-adjacent
  // 8 KiB entry; on 4 KiB hosts it does not. This conservative superset is
  // page validity, not a missed cache subscriber.
  REQUIRE(watch_cache.GetTextureSlot(disjoint, false) == watch_slots[3]);
  auto watch_live = LiveTextures(watch_cache);
  uint32_t expected_watch_live = page > kLegacyTextureExtent ? 1u : 2u;
  REQUIRE(watch_live.size() == expected_watch_live);
  REQUIRE_FALSE(ContainsTexture(watch_live, (__bridge void*)bridge));
  if (page <= kLegacyTextureExtent) {
    REQUIRE(watch_cache.GetTextureSlot(adjacent, false) == watch_slots[2]);
  }
  REQUIRE(watch_cache.InvalidateRange(kBase + 0x100, 0x20,
                                      MetalTextureCache::InvalidationScope::kAllRepresentations) ==
          0u);
  REQUIRE(watch_cache.slots_stale() == (page > kLegacyTextureExtent ? 4u : 3u));
  REQUIRE(direct_cache.slots_stale() == 3u);

  watch_cache.InvalidateRange(kBase, 0x20000);
  direct_cache.InvalidateRange(kBase, 0x20000);
  watch_cache.ReclaimStaleSlots(UINT64_MAX);
  direct_cache.ReclaimStaleSlots(UINT64_MAX);
}

TEST_CASE("Cache alias 5.3: resolve direct RAM-only retirement precedes suppressed "
          "global publication and preserves an overlapping bridge",
          "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x14000000;
  constexpr uint32_t kSurvivor = kBase + 0x10000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  FillMemory(*h.memory, kBase, 0x12000, 0xC1);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  auto* shared_a = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  auto* shared_b = shared_cache.Load(MakeSharedRouteFetch(kBase, 32));
  REQUIRE(shared_a != nullptr);
  REQUIRE(shared_b != nullptr);

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  MetalTextureCache legacy((__bridge void*)h.device, h.memory, shared.get(), (__bridge void*)heap,
                           64);
  legacy.SetSubmissionProvider([]() { return uint64_t(61); });
  auto alias = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  auto survivor = MakeLinearFetch(kSurvivor, xenos::TextureFormat::k_8_8_8_8);
  REQUIRE(legacy.GetTextureSlot(alias, false) != 0u);
  REQUIRE(legacy.GetTextureSlot(alias, true) != 0u);
  uint32_t survivor_slot = legacy.GetTextureSlot(survivor, false);
  REQUIRE(survivor_slot != 0u);
  REQUIRE(legacy.ConsumeUploadCount() == 3u);

  id<MTLTexture> bridge = MakeBridgeTexture(h.device);
  REQUIRE(bridge != nil);
  legacy.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                kLegacyTextureExtent);
  REQUIRE(LiveTextures(legacy).size() == 4u);

  // Production order: the writeback drain retires RAM uploads directly, then
  // typed publication invalidates shared pages with global fan-out suppressed.
  REQUIRE(legacy.InvalidateRange(kBase + 0x100, 0x20,
                                 MetalTextureCache::InvalidationScope::kRamUploadsOnly) == 2u);
  auto live = LiveTextures(legacy);
  REQUIRE(live.size() == 2u);
  REQUIRE(ContainsTexture(live, (__bridge void*)bridge));
  REQUIRE(legacy.GetTextureSlot(survivor, false) == survivor_slot);

  shared->MemoryInvalidationCallback(kBase + 0x100, 0x20, true,
                                     /*fire_global_watches=*/false);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());
  REQUIRE(legacy.GetTextureSlot(survivor, false) == survivor_slot);
  live = LiveTextures(legacy);
  REQUIRE(live.size() == 2u);
  REQUIRE(ContainsTexture(live, (__bridge void*)bridge));

  // kAll over the same overlapping range retires the bridge nontrivially; the
  // survivor is cleaned separately. Neither slot may be reclaimed early.
  REQUIRE(legacy.InvalidateRange(kBase + 0x100, 0x20) == 1u);
  REQUIRE(legacy.InvalidateRange(kSurvivor, kLegacyTextureExtent) == 1u);
  REQUIRE(legacy.slots_stale() == 4u);
  legacy.ReclaimStaleSlots(60);
  REQUIRE(legacy.slots_stale() == 4u);
  legacy.ReclaimStaleSlots(61);
  REQUIRE(legacy.slots_stale() == 0u);
  REQUIRE(legacy.slots_free() == 4u);
}

TEST_CASE("Cache alias 5.3: memexport full lifecycle direct convergence and "
          "publication-watch candidate retire identical entries before reuse",
          "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x15000000;
  constexpr uint32_t kDisjoint = kBase + 0x10000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  uint32_t page = shared->PageSize();
  uint32_t adjacent_base = kBase + std::max(page, kLegacyTextureExtent);
  FillMemory(*h.memory, kBase, 0x12000, 0xD1);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  auto* shared_a = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  auto* shared_b = shared_cache.Load(MakeSharedRouteFetch(kBase, 32));
  REQUIRE(shared_a != nullptr);
  REQUIRE(shared_b != nullptr);

  id<MTLBuffer> current_heap = [h.device newBufferWithLength:16 * 1024
                                                     options:MTLResourceStorageModeShared];
  id<MTLBuffer> candidate_heap = [h.device newBufferWithLength:16 * 1024
                                                       options:MTLResourceStorageModeShared];
  REQUIRE(current_heap != nil);
  REQUIRE(candidate_heap != nil);
  MetalTextureCache current((__bridge void*)h.device, h.memory, shared.get(),
                            (__bridge void*)current_heap, 64);
  MetalTextureCache candidate((__bridge void*)h.device, h.memory, shared.get(),
                              (__bridge void*)candidate_heap, 64);
  current.SetSubmissionProvider([]() { return uint64_t(71); });
  candidate.SetSubmissionProvider([]() { return uint64_t(71); });

  auto alias = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  auto adjacent = MakeLinearFetch(adjacent_base, xenos::TextureFormat::k_8_8_8_8);
  auto disjoint = MakeLinearFetch(kDisjoint, xenos::TextureFormat::k_8_8_8_8);
  auto seed = [&](MetalTextureCache& cache) {
    std::vector<uint32_t> slots = {
        cache.GetTextureSlot(alias, false), cache.GetTextureSlot(alias, true),
        cache.GetTextureSlot(adjacent, false), cache.GetTextureSlot(disjoint, false)};
    for (uint32_t slot : slots) {
      REQUIRE(slot != 0u);
    }
    REQUIRE(cache.ConsumeUploadCount() == 4u);
    return slots;
  };
  auto current_slots = seed(current);
  auto candidate_slots = seed(candidate);
  id<MTLTexture> bridge = MakeBridgeTexture(h.device);
  REQUIRE(bridge != nil);
  current.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                 kLegacyTextureExtent);
  candidate.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                   kLegacyTextureExtent);

  // Production publication happens before convergence and queues both legacy
  // global watches while firing the shared per-range subscribers.
  shared->RangeWrittenByGpu(kBase + 0x100, 0x20);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());

  auto range = GuestRange::FromBaseLength(kBase + 0x100, 0x20);
  REQUIRE(range.has_value());
  PendingProducer producer{*range, TransitState::kPendingMemexport,
                           /*submission=*/71};
  MatrixMemexportConvergenceTarget current_target(*h.memory, &current);
  auto current_result = ConvergeMemexportRam(producer, current_target);
  REQUIRE(current_result.converged);
  REQUIRE(current_result.invalidated_entries == 3u);
  REQUIRE(LiveTextures(current).size() == 2u);

  // Candidate 5.4 shape: bytes converge but the earlier publication watch is
  // the only cache invalidation. The next cache lookup drains that queue before
  // it can serve any target texture.
  MatrixMemexportConvergenceTarget candidate_target(*h.memory, nullptr);
  auto candidate_result = ConvergeMemexportRam(producer, candidate_target);
  REQUIRE(candidate_result.converged);
  REQUIRE(candidate_result.invalidated_entries == 0u);
  REQUIRE(LiveTextures(candidate).size() == 5u);
  for (uint32_t i = 0; i < 0x20; ++i) {
    REQUIRE(h.memory->TranslatePhysical<uint8_t*>(kBase + 0x100)[i] == uint8_t(0xC0u + i));
  }

  REQUIRE(current.GetTextureSlot(disjoint, false) == current_slots[3]);
  REQUIRE(candidate.GetTextureSlot(disjoint, false) == candidate_slots[3]);
  REQUIRE(current.GetTextureSlot(adjacent, false) == current_slots[2]);
  REQUIRE(candidate.GetTextureSlot(adjacent, false) == candidate_slots[2]);
  auto current_live = LiveTextures(current);
  auto candidate_live = LiveTextures(candidate);
  REQUIRE(current_live.size() == 2u);
  REQUIRE(candidate_live.size() == 2u);
  REQUIRE_FALSE(ContainsTexture(current_live, (__bridge void*)bridge));
  REQUIRE_FALSE(ContainsTexture(candidate_live, (__bridge void*)bridge));
  REQUIRE(current.slots_stale() == 3u);
  REQUIRE(candidate.slots_stale() == 3u);
  REQUIRE(current.InvalidateRange(kBase + 0x100, 0x20) == 0u);
  REQUIRE(candidate.InvalidateRange(kBase + 0x100, 0x20) == 0u);

  current.ReclaimStaleSlots(70);
  candidate.ReclaimStaleSlots(70);
  REQUIRE(current.slots_stale() == 3u);
  REQUIRE(candidate.slots_stale() == 3u);
  current.ReclaimStaleSlots(71);
  candidate.ReclaimStaleSlots(71);
  REQUIRE(current.slots_stale() == 0u);
  REQUIRE(candidate.slots_stale() == 0u);

  current.InvalidateRange(adjacent_base, 0x20000);
  candidate.InvalidateRange(adjacent_base, 0x20000);
  current.ReclaimStaleSlots(UINT64_MAX);
  candidate.ReclaimStaleSlots(UINT64_MAX);
}

TEST_CASE("Cache alias 5.3: memexport trace playback fires shared range fan-out but "
          "leaves an empty legacy-global set, so direct convergence stays required",
          "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x17000000;
  constexpr uint32_t kSurvivor = kBase + 0x10000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  FillMemory(*h.memory, kBase, 0x12000, 0xF1);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  auto* shared_a = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  auto* shared_b = shared_cache.Load(MakeSharedRouteFetch(kBase, 32));
  REQUIRE(shared_a != nullptr);
  REQUIRE(shared_b != nullptr);

  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  MetalTextureCache legacy((__bridge void*)h.device, h.memory, shared.get(), (__bridge void*)heap,
                           64);
  legacy.SetSubmissionProvider([]() { return uint64_t(91); });
  auto alias = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  auto survivor = MakeLinearFetch(kSurvivor, xenos::TextureFormat::k_8_8_8_8);
  // Two overlapping RAM aliases differing only by fetch sign, plus a disjoint
  // survivor whose extent never intersects the memexport write range.
  REQUIRE(legacy.GetTextureSlot(alias, false) != 0u);
  REQUIRE(legacy.GetTextureSlot(alias, true) != 0u);
  uint32_t survivor_slot = legacy.GetTextureSlot(survivor, false);
  REQUIRE(survivor_slot != 0u);
  REQUIRE(legacy.ConsumeUploadCount() == 3u);

  id<MTLTexture> bridge = MakeBridgeTexture(h.device);
  REQUIRE(bridge != nil);
  legacy.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                kLegacyTextureExtent);
  REQUIRE(LiveTextures(legacy).size() == 4u);

  // Trace playback permanently unregisters the legacy global watch and makes
  // GetTextureSlot skip DrainPendingInvalidations. A replayed memexport GPU
  // publication therefore fires only the shared per-range subscribers; the
  // legacy target set is left entirely intact, and even a drain-attempting
  // lookup cannot retire it.
  legacy.SetTracePlayback();
  shared->RangeWrittenByGpu(kBase + 0x100, 0x20);
  REQUIRE(shared_a->base_outdated_lockless());
  REQUIRE(shared_b->base_outdated_lockless());
  REQUIRE(legacy.GetTextureSlot(survivor, false) == survivor_slot);
  REQUIRE(LiveTextures(legacy).size() == 4u);

  // With the watch set empty under trace playback, direct convergence is the
  // only path that retires the target aliases and the overlapping bridge. Task
  // 5.4 must therefore retain this memexport convergence direct leg globally
  // unless production conditionally restores the legacy watch (or otherwise
  // proves a watch equivalent) that survives SetTracePlayback's teardown.
  auto range = GuestRange::FromBaseLength(kBase + 0x100, 0x20);
  REQUIRE(range.has_value());
  PendingProducer producer{*range, TransitState::kPendingMemexport,
                           /*submission=*/91};
  MatrixMemexportConvergenceTarget target(*h.memory, &legacy);
  auto result = ConvergeMemexportRam(producer, target);
  REQUIRE(result.converged);
  REQUIRE(result.invalidated_entries == 3u);
  for (uint32_t i = 0; i < 0x20; ++i) {
    REQUIRE(h.memory->TranslatePhysical<uint8_t*>(kBase + 0x100)[i] == uint8_t(0xC0u + i));
  }
  REQUIRE(legacy.GetTextureSlot(survivor, false) == survivor_slot);
  auto live = LiveTextures(legacy);
  REQUIRE(live.size() == 1u);
  REQUIRE_FALSE(ContainsTexture(live, (__bridge void*)bridge));
  REQUIRE(legacy.slots_stale() == 3u);

  // The retired target keys carry submission 91; reclaim must wait for it to
  // complete before any target slot can be reused, exactly as the normal-mode
  // convergence path above.
  legacy.ReclaimStaleSlots(90);
  REQUIRE(legacy.slots_stale() == 3u);
  legacy.ReclaimStaleSlots(91);
  REQUIRE(legacy.slots_stale() == 0u);
  REQUIRE(legacy.slots_free() == 3u);

  legacy.InvalidateRange(kSurvivor, kLegacyTextureExtent);
  legacy.ReclaimStaleSlots(UINT64_MAX);
}

TEST_CASE("Cache alias 5.3: trace restore has shared range fan-out but an empty "
          "legacy-global set, so its direct all-representations leg is required",
          "[l1][metal][cache][alias][authority]") {
  REQUIRE_DEVICE();
  auto& h = H();
  constexpr uint32_t kBase = 0x16000000;

  auto shared = std::make_unique<AliasSharedMemory>(*h.memory);
  REQUIRE(shared->Initialize());
  FillMemory(*h.memory, kBase, shared->PageSize() * 2, 0xE1);

  RegisterFile register_file;
  WatchTextureCache shared_cache(register_file, *shared);
  auto* shared_entry = shared_cache.Load(MakeSharedRouteFetch(kBase, 16));
  REQUIRE(shared_entry != nullptr);
  id<MTLBuffer> heap = [h.device newBufferWithLength:16 * 1024
                                             options:MTLResourceStorageModeShared];
  REQUIRE(heap != nil);
  MetalTextureCache legacy((__bridge void*)h.device, h.memory, shared.get(), (__bridge void*)heap,
                           64);
  legacy.SetSubmissionProvider([]() { return uint64_t(81); });
  auto alias = MakeLinearFetch(kBase, xenos::TextureFormat::k_8_8_8_8);
  REQUIRE(legacy.GetTextureSlot(alias, false) != 0u);
  REQUIRE(legacy.GetTextureSlot(alias, true) != 0u);
  REQUIRE(legacy.ConsumeUploadCount() == 2u);
  id<MTLTexture> bridge = MakeBridgeTexture(h.device);
  REQUIRE(bridge != nil);
  legacy.RegisterResolvedTarget(kBase, (__bridge void*)bridge, 16, 16, false, false, kBase,
                                kLegacyTextureExtent);
  REQUIRE(LiveTextures(legacy).size() == 3u);

  legacy.SetTracePlayback();
  auto publication = DecideRamWritePublication(WriterKind::kTraceRestore, kBase + 0x100, 0x20);
  REQUIRE(publication.has_value());
  shared->MemoryInvalidationCallback(kBase + 0x100, 0x20, publication->policy.exact_range);
  REQUIRE(shared_entry->base_outdated_lockless());
  REQUIRE(LiveTextures(legacy).size() == 3u);

  // SetTracePlayback removed the legacy global subscriber, so only the shared
  // range set fired. The direct all-representations leg must retire both RAM
  // aliases and the overlapping bridge. Task 5.2 separately executes the full
  // PublishRamWrite ordering; this case isolates and enumerates the two sets.
  RealCacheRamWriteTarget target(*shared, legacy);
  uint32_t direct_invalidated = 0;
  REQUIRE(target.InvalidateLegacyTextures(kBase + 0x100, 0x20, &direct_invalidated));
  REQUIRE(direct_invalidated == 3u);
  REQUIRE(LiveTextures(legacy).empty());
  REQUIRE(legacy.slots_stale() == 3u);
  legacy.ReclaimStaleSlots(80);
  REQUIRE(legacy.slots_stale() == 3u);
  legacy.ReclaimStaleSlots(81);
  REQUIRE(legacy.slots_stale() == 0u);
}
}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
