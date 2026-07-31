/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// POSITIVE byte gate for the mip-range key change
// (openspec/changes/implement-metal-texture-key-mip-range, task 4.3).
//
// Zero wrong-range cache hits proves the bad alias disappeared. It does NOT
// prove the replacement entry contains the bytes the fetch asked for — an entry
// keyed correctly but populated wrongly would satisfy the absence check and
// still serve the wrong image. This closes that gap positively, through the
// PRODUCTION cache, on a shape taken from the measured Red Dead collisions:
// a fetch whose normalized effective range is lv1..1.
//
// Provenance is the discriminator. The base level and the mip source are seeded
// with different constants, so a host level populated from the wrong guest
// region is unambiguous rather than a subtle numeric difference. This avoids
// re-implementing the guest mip layout in the test, which would only assert
// that the test agrees with itself.

#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

#include "test_shared_physical_memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kBase = 0x2000000;
constexpr uint32_t kMipBase = 0x2100000;
constexpr uint8_t kBaseFill = 0xAA;
constexpr uint8_t kMipFill = 0xBB;

void EnsureAllocated(Memory& memory, uint32_t base, uint32_t length) {
  auto heap = memory.LookupHeap(base);
  if (!heap) return;
  heap->AllocFixed(base, length, 4096,
                   kMemoryAllocationReserve | kMemoryAllocationCommit,
                   kMemoryProtectRead | kMemoryProtectWrite);
}

void FillMemory(Memory& memory, uint32_t base, uint32_t length, uint8_t value) {
  EnsureAllocated(memory, base, length);
  auto* p = memory.TranslatePhysical(base);
  if (p) std::memset(p, value, length);
}

// A linear, unswizzled 64x32 k_8_8_8_8 fetch — the shape the Red Dead
// collisions took, reduced to the smallest form that still carries a mip chain.
xenos::xe_gpu_texture_fetch_t MakeFetch(uint32_t mip_min, uint32_t mip_max) {
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
  fetch.base_address = kBase >> 12;
  fetch.mip_address = kMipBase >> 12;
  fetch.size_2d.width = 63;
  fetch.size_2d.height = 31;
  fetch.swizzle = 0x688u;
  fetch.mip_min_level = mip_min;
  fetch.mip_max_level = mip_max;
  fetch.dimension = xenos::DataDimension::k2DOrStacked;
  fetch.packed_mips = 0;
  return fetch;
}

void Normalized(const xenos::xe_gpu_texture_fetch_t& fetch, uint32_t* min_out,
                uint32_t* max_out) {
  texture_util::GetSubresourcesFromFetchConstant(
      fetch, nullptr, nullptr, nullptr, nullptr, nullptr, min_out, max_out);
}

struct Harness {
  id<MTLDevice> device = nil;
  id<MTLBuffer> view_heap = nil;
  Memory* memory = nullptr;
  std::unique_ptr<MetalTextureCache> cache;
  bool ok = false;

  Harness() {
    device = MTLCreateSystemDefaultDevice();
    memory = conformance::TestPhysicalMemory();
    if (!device || !memory) return;
    view_heap = [device newBufferWithLength:16 * 1024
                                    options:MTLResourceStorageModeShared];
    if (!view_heap) return;
    cache = std::make_unique<MetalTextureCache>(
        (__bridge void*)device, memory, nullptr, (__bridge void*)view_heap, 32);
    ok = cache != nullptr;
  }
};

Harness& H() {
  static Harness harness;
  return harness;
}

// Reads one host mip level back and reports the dominant byte. Constant seeding
// makes that sufficient to identify which guest region populated the level.
bool ReadLevelByte(id<MTLTexture> texture, uint32_t level, uint8_t* out) {
  if (!texture || level >= texture.mipmapLevelCount) return false;
  uint32_t w = std::max<uint32_t>(1u, uint32_t(texture.width) >> level);
  uint32_t h = std::max<uint32_t>(1u, uint32_t(texture.height) >> level);
  std::vector<uint8_t> bytes(size_t(w) * h * 4);
  [texture getBytes:bytes.data()
        bytesPerRow:w * 4
      bytesPerImage:0
         fromRegion:MTLRegionMake2D(0, 0, w, h)
        mipmapLevel:level
              slice:0];
  *out = bytes.empty() ? 0 : bytes[0];
  return true;
}

}  // namespace

TEST_CASE("Key mip range 4.3: the lv1..1 entry is populated from the mip source",
          "[l1][metal][texture_key][bytes]") {
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  // Seed distinct provenance for the two guest regions.
  FillMemory(*h.memory, kBase, 0x8000, kBaseFill);
  FillMemory(*h.memory, kMipBase, 0x8000, kMipFill);

  auto minified = MakeFetch(/*mip_min=*/1, /*mip_max=*/1);
  uint32_t norm_min = 0, norm_max = 0;
  Normalized(minified, &norm_min, &norm_max);

  SECTION("the request normalizes to lv1..1") {
    // Guards the premise: a mip page IS present, so the range does not collapse.
    CHECK(norm_min == 1);
    CHECK(norm_max == 1);
  }

  SECTION("the admitted entry is distinct from the base-only entry") {
    // The key must record the range: a lv0..0 request must not be answered with
    // the lv1..1 entry, which is precisely the measured defect.
    auto base_only = MakeFetch(/*mip_min=*/0, /*mip_max=*/0);
    uint32_t base_slot = h.cache->GetTextureSlot(base_only, false);
    uint32_t minified_slot = h.cache->GetTextureSlot(minified, false);
    REQUIRE(base_slot != 0);
    REQUIRE(minified_slot != 0);
    CHECK(base_slot != minified_slot);
  }

  SECTION("the host level carries the MIP source bytes, not the base bytes") {
    h.cache->ResetActive();
    uint32_t slot = h.cache->GetTextureSlot(minified, false);
    REQUIRE(slot != 0);
    REQUIRE_FALSE(h.cache->active_textures().empty());
    id<MTLTexture> texture =
        (__bridge id<MTLTexture>)h.cache->active_textures().back();
    REQUIRE(texture != nil);

    // A lv1..1 request must produce a texture that actually carries level 1.
    INFO("host mip levels " << texture.mipmapLevelCount);
    REQUIRE(texture.mipmapLevelCount >= 2);

    uint8_t level0 = 0, level1 = 0;
    REQUIRE(ReadLevelByte(texture, 0, &level0));
    REQUIRE(ReadLevelByte(texture, 1, &level1));
    INFO("level0 byte " << std::hex << uint32_t(level0) << ", level1 byte "
                        << uint32_t(level1));

    // Provenance, which is the whole point: level 0 comes from the base region
    // and level 1 from the mip region. Before this change a lv1..1 fetch could
    // be served a single-level entry built from the base alone.
    CHECK(level0 == kBaseFill);
    CHECK(level1 == kMipFill);
    CHECK(level0 != level1);
  }

  SECTION("re-requesting hits the same correctly populated entry") {
    h.cache->ResetActive();
    uint32_t first = h.cache->GetTextureSlot(minified, false);
    REQUIRE(first != 0);
    void* first_texture = h.cache->active_textures().back();

    h.cache->ResetActive();
    uint32_t second = h.cache->GetTextureSlot(minified, false);
    REQUIRE(second != 0);
    void* second_texture = h.cache->active_textures().back();

    CHECK(first == second);
    CHECK(first_texture == second_texture);

    uint8_t level1 = 0;
    REQUIRE(ReadLevelByte((__bridge id<MTLTexture>)second_texture, 1, &level1));
    CHECK(level1 == kMipFill);
  }
}

TEST_CASE("Key mip range 6.x: the split siblings invalidate independently",
          "[l1][metal][texture_key][lifecycle]") {
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  // The two siblings the key split creates for one guest texture: a base-only
  // entry (no mip data uploaded) and a minified entry that carries BOTH the
  // base level and the mip level. Their invalidation footprints therefore
  // differ, and that difference is the point of testing them as siblings rather
  // than by aggregate count.
  auto base_only = MakeFetch(/*mip_min=*/0, /*mip_max=*/0);
  auto minified = MakeFetch(/*mip_min=*/1, /*mip_max=*/1);

  auto admit_both = [&]() {
    FillMemory(*h.memory, kBase, 0x8000, kBaseFill);
    FillMemory(*h.memory, kMipBase, 0x8000, kMipFill);
    h.cache->ClearCache();
    h.cache->ResetActive();
    uint32_t a = h.cache->GetTextureSlot(base_only, false);
    uint32_t b = h.cache->GetTextureSlot(minified, false);
    REQUIRE(a != 0);
    REQUIRE(b != 0);
    REQUIRE(a != b);
    REQUIRE(h.cache->cached_texture_count() == 2u);
  };

  // Provenance after re-admission, so each event is followed by a
  // correctness check rather than only a count check.
  auto check_provenance = [&]() {
    h.cache->ResetActive();
    REQUIRE(h.cache->GetTextureSlot(minified, false) != 0);
    id<MTLTexture> t =
        (__bridge id<MTLTexture>)h.cache->active_textures().back();
    REQUIRE(t != nil);
    REQUIRE(t.mipmapLevelCount >= 2);
    uint8_t l0 = 0, l1 = 0;
    REQUIRE(ReadLevelByte(t, 0, &l0));
    REQUIRE(ReadLevelByte(t, 1, &l1));
    CHECK(l0 == kBaseFill);
    CHECK(l1 == kMipFill);
  };

  SECTION("a base-page write invalidates BOTH siblings") {
    admit_both();
    // Both entries uploaded host level 0 from the base region, so both must go.
    uint32_t evicted = h.cache->InvalidateRange(kBase, 0x1000);
    CHECK(evicted == 2u);
    CHECK(h.cache->cached_texture_count() == 0u);
    check_provenance();
  }

  SECTION("a mip-page write invalidates ONLY the minified sibling") {
    admit_both();
    // The base-only entry uploaded no mip data at all — its recorded mip range
    // is empty — so evicting it here would be unnecessary work, not safety.
    uint32_t evicted = h.cache->InvalidateRange(kMipBase, 0x1000);
    CHECK(evicted == 1u);
    CHECK(h.cache->cached_texture_count() == 1u);

    // The survivor must be the base-only entry: re-requesting it is a HIT that
    // does not grow the cache, while the minified one must be rebuilt.
    h.cache->ResetActive();
    REQUIRE(h.cache->GetTextureSlot(base_only, false) != 0);
    CHECK(h.cache->cached_texture_count() == 1u);
    check_provenance();
    CHECK(h.cache->cached_texture_count() == 2u);
  }

  SECTION("ClearCache releases both siblings and frees their slots") {
    admit_both();
    h.cache->ClearCache();
    CHECK(h.cache->cached_texture_count() == 0u);
    // Slots return through the free list rather than by resetting the
    // allocator, which is the documented behavior; re-admission must therefore
    // succeed and repopulate correct provenance.
    check_provenance();
  }

  SECTION("repeated clear/re-admit cycles are stable") {
    for (int cycle = 0; cycle < 3; ++cycle) {
      INFO("cycle " << cycle);
      admit_both();
      CHECK(h.cache->cached_texture_count() == 2u);
      h.cache->ClearCache();
      CHECK(h.cache->cached_texture_count() == 0u);
    }
    admit_both();
    check_provenance();
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
