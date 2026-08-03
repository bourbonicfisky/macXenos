/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 (Metal device required): what ReclaimStaleSlots leaves in a freed slot's
// view-heap descriptor, for port-xenios-tessellation task 5.3 candidate 3.
//
// THE DEFECT UNDER TEST. Reclaim releases the retired texture and pushes the
// slot onto the free list WITHOUT rewriting its descriptor, relying on
// "descriptors are overwritten on the next AllocateSlot that hands a freed slot
// out". Between reclaim and that reallocation, a LIVE bindless heap holds a
// descriptor naming deallocated memory, so safety depends on no shader ever
// indexing a free slot -- an assumption about shader behaviour, not a lifetime
// guarantee. ClearCache's free_slot already rejects exactly this reasoning for
// the clear boundary and rebinds to the null texture; the per-frame reclaim
// path does not.
//
// Every assertion below reads the REAL view-heap buffer the production cache
// writes through -- no model of the descriptor, no test-only accessor. The
// heap is a shared MTLBuffer of IRDescriptorTableEntry, so a slot's descriptor
// is compared byte-for-byte against a reference entry built by the same
// IRDescriptorTableSetTexture the production path uses.
//
// MUTATION PIN: "reclaim writes the null descriptor" and "reclaim in the
// default mode does NOT" are asserted as a PAIR. Deleting the rebind from
// ReclaimStaleSlots fails the first; making it unconditional fails the second.
// A one-sided test would pass against a build that nulled every slot always,
// which would destroy the default arm's bit-identity.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#import <Metal/Metal.h>

#include "metal_irconverter_runtime.h"
#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_slot_reclaim_flags.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kHeapSlots = 4096;

struct SlotEnv {
  id<MTLDevice> device = nil;
  Memory* memory = nullptr;
  std::unique_ptr<MetalSharedMemory> shared_memory;
  id<MTLBuffer> view_heap = nil;
  id<MTLTexture> null_texture = nil;
  id<MTLTexture> live_texture = nil;
  std::unique_ptr<MetalTextureCache> cache;
  uint64_t submission = 1;
  bool ok = false;

  SlotEnv() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    shared_memory =
        std::make_unique<MetalSharedMemory>(*memory, (__bridge void*)device);
    if (!shared_memory->Initialize()) {
      return;
    }
    view_heap =
        [device newBufferWithLength:kHeapSlots * sizeof(IRDescriptorTableEntry)
                            options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                  MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    null_texture = [device newTextureWithDescriptor:d];
    live_texture = [device newTextureWithDescriptor:d];
    if (!view_heap || !null_texture || !live_texture) {
      return;
    }
    cache = std::make_unique<MetalTextureCache>((__bridge void*)device, memory,
                                                shared_memory.get(),
                                                (__bridge void*)view_heap,
                                                kHeapSlots);
    cache->SetNullTexture((__bridge void*)null_texture);
    cache->SetSubmissionProvider([this]() { return submission; });
    ok = true;
  }

  IRDescriptorTableEntry* entries() const {
    return static_cast<IRDescriptorTableEntry*>(view_heap.contents);
  }
  IRDescriptorTableEntry entry(uint32_t slot) const { return entries()[slot]; }
};

IRDescriptorTableEntry ReferenceEntryFor(id<MTLTexture> texture) {
  IRDescriptorTableEntry e = {};
  IRDescriptorTableSetTexture(&e, texture, 0.0f, 0);
  return e;
}

bool SameEntry(const IRDescriptorTableEntry& a,
               const IRDescriptorTableEntry& b) {
  return std::memcmp(&a, &b, sizeof(IRDescriptorTableEntry)) == 0;
}

// Drives one retire+reclaim cycle under an explicit mode and reports what the
// freed slot's descriptor holds afterwards. The environment variable is the
// production selector, so the test exercises the same path a live arm does.
struct ReclaimOutcome {
  uint32_t slot = 0;
  IRDescriptorTableEntry after_retire = {};
  IRDescriptorTableEntry after_reclaim = {};
};

ReclaimOutcome RunReclaim(SlotEnv& env, const char* mode_value,
                          uint64_t completed) {
  if (mode_value) {
    setenv("XE_METAL_SLOT_RECLAIM", mode_value, 1);
  } else {
    unsetenv("XE_METAL_SLOT_RECLAIM");
  }
  ReclaimOutcome out;
  out.slot = env.cache->AllocateExternalSlot();
  REQUIRE(out.slot != 0);
  env.cache->BindExternalSlot(out.slot, (__bridge void*)env.live_texture);
  env.cache->RetireExternalSlot(out.slot);
  out.after_retire = env.entry(out.slot);
  env.cache->ReclaimStaleSlots(completed);
  out.after_reclaim = env.entry(out.slot);
  unsetenv("XE_METAL_SLOT_RECLAIM");
  return out;
}

TEST_CASE("Slot reclaim mode parses strictly and fails closed to current",
          "[l0][metal][slot-reclaim]") {
  CHECK(ParseMetalSlotReclaimMode(nullptr) == MetalSlotReclaimMode::kCurrent);
  CHECK(ParseMetalSlotReclaimMode("") == MetalSlotReclaimMode::kCurrent);
  CHECK(ParseMetalSlotReclaimMode("current") ==
        MetalSlotReclaimMode::kCurrent);
  CHECK(ParseMetalSlotReclaimMode("widen") ==
        MetalSlotReclaimMode::kWidenWindow);
  CHECK(ParseMetalSlotReclaimMode("null") ==
        MetalSlotReclaimMode::kNullOnReclaim);
  for (const char* bad : {"1", "NULL", "Null", "true", "widen ", "nullify"}) {
    CHECK(ParseMetalSlotReclaimMode(bad) == MetalSlotReclaimMode::kMalformed);
    // Malformed must behave as CURRENT -- never as an untested mode.
    CHECK(MetalSlotReclaimModeEffective(ParseMetalSlotReclaimMode(bad)) ==
          MetalSlotReclaimMode::kCurrent);
  }
  CHECK(MetalSlotReclaimModeEffective(MetalSlotReclaimMode::kNullOnReclaim) ==
        MetalSlotReclaimMode::kNullOnReclaim);
}

TEST_CASE("Reclaim writes the null descriptor only in null mode",
          "[l1][device][metal][slot-reclaim]") {
  SlotEnv env;
  if (!env.ok) {
    WARN("no Metal device / shared memory; skipping");
    return;
  }
  const IRDescriptorTableEntry null_ref = ReferenceEntryFor(env.null_texture);
  const IRDescriptorTableEntry live_ref = ReferenceEntryFor(env.live_texture);
  REQUIRE_FALSE(SameEntry(null_ref, live_ref));

  SECTION("default mode leaves the released texture's descriptor in place") {
    // This is the DEFECT, asserted as current behaviour so the fix is a
    // visible change rather than a claim -- and so an unconditional rebind
    // (which would break the default arm's bit-identity) fails here.
    ReclaimOutcome out = RunReclaim(env, nullptr, /*completed=*/8);
    CHECK(SameEntry(out.after_retire, live_ref));
    CHECK(SameEntry(out.after_reclaim, live_ref));
    CHECK_FALSE(SameEntry(out.after_reclaim, null_ref));
  }

  SECTION("explicit current mode matches unset") {
    ReclaimOutcome out = RunReclaim(env, "current", /*completed=*/8);
    CHECK(SameEntry(out.after_reclaim, live_ref));
  }

  SECTION("malformed value fails closed to current") {
    ReclaimOutcome out = RunReclaim(env, "nullify", /*completed=*/8);
    CHECK(SameEntry(out.after_reclaim, live_ref));
  }

  SECTION("null mode rebinds the freed slot to the null texture") {
    // MUTATION PIN: deleting the rebind from ReclaimStaleSlots fails this.
    ReclaimOutcome out = RunReclaim(env, "null", /*completed=*/8);
    CHECK(SameEntry(out.after_retire, live_ref));
    CHECK(SameEntry(out.after_reclaim, null_ref));
  }
}

TEST_CASE("Reclaim never touches a slot whose submission is still in flight",
          "[l1][device][metal][slot-reclaim]") {
  SlotEnv env;
  if (!env.ok) {
    WARN("no Metal device / shared memory; skipping");
    return;
  }
  const IRDescriptorTableEntry null_ref = ReferenceEntryFor(env.null_texture);
  const IRDescriptorTableEntry live_ref = ReferenceEntryFor(env.live_texture);

  // Retired while encoding submission 9; only submission 8 has completed.
  env.submission = 9;
  setenv("XE_METAL_SLOT_RECLAIM", "null", 1);
  uint32_t slot = env.cache->AllocateExternalSlot();
  REQUIRE(slot != 0);
  env.cache->BindExternalSlot(slot, (__bridge void*)env.live_texture);
  env.cache->RetireExternalSlot(slot);

  env.cache->ReclaimStaleSlots(/*completed=*/8);
  // A LIVE slot must never be nulled: submission 9 may still be reading it, and
  // nulling early would be the same class of bug as freeing early (the async-S8
  // page-fault flicker).
  CHECK(SameEntry(env.entry(slot), live_ref));
  CHECK_FALSE(SameEntry(env.entry(slot), null_ref));

  // Once it completes, the same slot is nulled.
  env.cache->ReclaimStaleSlots(/*completed=*/9);
  CHECK(SameEntry(env.entry(slot), null_ref));
  unsetenv("XE_METAL_SLOT_RECLAIM");
}

TEST_CASE("Reallocation overwrites the null descriptor",
          "[l1][device][metal][slot-reclaim]") {
  SlotEnv env;
  if (!env.ok) {
    WARN("no Metal device / shared memory; skipping");
    return;
  }
  const IRDescriptorTableEntry null_ref = ReferenceEntryFor(env.null_texture);
  const IRDescriptorTableEntry live_ref = ReferenceEntryFor(env.live_texture);

  ReclaimOutcome out = RunReclaim(env, "null", /*completed=*/8);
  REQUIRE(SameEntry(out.after_reclaim, null_ref));

  // The freed slot must come back and be fully rewritten -- nulling must not
  // strand a slot or leave the null descriptor visible to a real binding.
  setenv("XE_METAL_SLOT_RECLAIM", "null", 1);
  uint32_t reused = env.cache->AllocateExternalSlot();
  CHECK(reused == out.slot);
  env.cache->BindExternalSlot(reused, (__bridge void*)env.live_texture);
  CHECK(SameEntry(env.entry(reused), live_ref));
  CHECK_FALSE(SameEntry(env.entry(reused), null_ref));
  unsetenv("XE_METAL_SLOT_RECLAIM");
}

TEST_CASE("Widen mode holds the slot out of the free list, then releases it",
          "[l1][device][metal][slot-reclaim]") {
  SlotEnv env;
  if (!env.ok) {
    WARN("no Metal device / shared memory; skipping");
    return;
  }
  setenv("XE_METAL_SLOT_RECLAIM", "widen", 1);
  uint32_t slot = env.cache->AllocateExternalSlot();
  REQUIRE(slot != 0);
  env.cache->BindExternalSlot(slot, (__bridge void*)env.live_texture);
  env.cache->RetireExternalSlot(slot);

  // Reclaimed, but deliberately NOT returned to the free list yet: the probe's
  // whole purpose is that the dangling descriptor survives longer.
  env.cache->ReclaimStaleSlots(/*completed=*/8);
  uint32_t next = env.cache->AllocateExternalSlot();
  CHECK(next != slot);

  // After the hold expires the slot returns, so the probe cannot leak slots.
  env.cache->ReclaimStaleSlots(
      /*completed=*/8 + kMetalSlotReclaimWidenHoldSubmissions + 1);
  uint32_t recycled = env.cache->AllocateExternalSlot();
  CHECK(recycled == slot);
  unsetenv("XE_METAL_SLOT_RECLAIM");
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
