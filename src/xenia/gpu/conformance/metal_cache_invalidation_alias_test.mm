/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Task 5.2: factual legacy-cache alias and lifetime coverage. Every entry is
// created through MetalTextureCache::GetTextureSlot, and the production
// InvalidateRange scan is the only overlap oracle. A real Metal device is
// required because entries own real textures and bindless descriptor slots.

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

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
