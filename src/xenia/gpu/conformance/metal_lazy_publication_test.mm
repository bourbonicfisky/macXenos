/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Stage 2 of optimize-metal-gpu-resident-resolve-coherence, L0 cases:
//  - the physical-heap READ watch (memory.cc): a no-access page's first CPU
//    read through the virtual view calls the read callback with kRead and the
//    read then completes; a write calls kWrite; a release calls kDiscard;
//  - MetalLazyPublication::OnAccess copies exactly the touched page of an
//    owner's tiled extent, byte-identical to the eager tiled walk.

#include <atomic>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_lazy_publication.h"
#include "xenia/gpu/metal/metal_tiled_publish_run.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

struct ReadWatchRecorder {
  std::vector<std::pair<Memory::PhysicalAccess, std::pair<uint32_t, uint32_t>>>
      calls;
  static void Thunk(void* context, uint32_t start, uint32_t length,
                    Memory::PhysicalAccess access) {
    auto* self = static_cast<ReadWatchRecorder*>(context);
    self->calls.emplace_back(access, std::make_pair(start, length));
  }
  size_t Count(Memory::PhysicalAccess access) const {
    size_t n = 0;
    for (const auto& c : calls) {
      n += c.first == access ? 1 : 0;
    }
    return n;
  }
};

}  // namespace

TEST_CASE("lazy publication: physical read watch fires on the first CPU read",
          "[l0][metal][lazypub]") {
  Memory* memory = conformance::TestPhysicalMemory();
  REQUIRE(memory);
  BaseHeap* heap = memory->LookupHeap(0xA0000000u);
  REQUIRE(heap);
  uint32_t guest_address = 0;
  // Two host pages (arm64 macOS: 16 KiB each) so the range spans a watch
  // block boundary case as well.
  const uint32_t kSize = 64 * 1024;
  REQUIRE(heap->Alloc(kSize, 64 * 1024,
                      kMemoryAllocationReserve | kMemoryAllocationCommit,
                      kMemoryProtectRead | kMemoryProtectWrite, false,
                      &guest_address));
  const uint32_t physical = memory->GetPhysicalAddress(guest_address);
  uint8_t* alias = memory->TranslatePhysical<uint8_t*>(physical);
  REQUIRE(alias);
  // The "GPU-resident" bytes: written through the physical alias, which the
  // watch never protects.
  for (uint32_t i = 0; i < kSize; ++i) {
    alias[i] = uint8_t(i * 3 + 1);
  }

  ReadWatchRecorder recorder;
  void* handle =
      memory->RegisterPhysicalMemoryReadCallback(&ReadWatchRecorder::Thunk,
                                                 &recorder);
  REQUIRE(handle);
  memory->EnablePhysicalMemoryAccessCallbacks(physical, kSize, false, true);

  // First read through the virtual (protected) view: faults, the callback
  // runs (twice: once with the global lock released, once under it), the
  // read completes with the aliased bytes.
  volatile uint8_t* view = memory->TranslateVirtual<uint8_t*>(guest_address);
  uint8_t observed = view[4096 * 3 + 5];
  CHECK(observed == uint8_t((4096 * 3 + 5) * 3 + 1));
  REQUIRE(recorder.Count(Memory::PhysicalAccess::kRead) >= 1);
  CHECK(recorder.calls.front().second.first <= physical + 4096 * 3);
  CHECK(recorder.calls.front().second.first +
            recorder.calls.front().second.second >
        physical + 4096 * 3);
  // Second read of the same host page: watch cleared, no further callback.
  size_t reads_before = recorder.calls.size();
  observed = view[4096 * 3 + 6];
  CHECK(observed == uint8_t((4096 * 3 + 6) * 3 + 1));
  CHECK(recorder.calls.size() == reads_before);

  // A write to a still-watched host page publishes first (kWrite) and then
  // proceeds; the byte lands in the alias.
  const uint32_t write_offset = kSize - 4096;  // last host page, unread yet
  view[write_offset] = 0xEE;
  CHECK(alias[write_offset] == 0xEE);
  CHECK(recorder.Count(Memory::PhysicalAccess::kWrite) >= 1);

  // Releasing the allocation discards whatever a read watch still owed.
  size_t discards_before = recorder.Count(Memory::PhysicalAccess::kDiscard);
  REQUIRE(heap->Release(guest_address));
  CHECK(recorder.Count(Memory::PhysicalAccess::kDiscard) >= discards_before);
  memory->UnregisterPhysicalMemoryReadCallback(handle);
}

TEST_CASE("lazy publication: OnAccess copies the touched page of a tiled owner",
          "[l0][metal][lazypub]") {
  // Owner: a 256x64 RGBA8 tiled rectangle at guest 0x10000, pitch 256.
  MetalLazyPublication table;
  MetalLazyPublication::Owner owner;
  owner.authority_sequence = 7;
  owner.submission = 3;
  owner.guest_addr = 0x10000;
  owner.width = 256;
  owner.height = 64;
  owner.bpp = 4;
  owner.dest_pitch_aligned = 256;
  owner.dest_offset_x = 0;
  owner.dest_offset_y = 0;
  owner.dest_extent_bytes = 256 * 64 * 4;
  owner.range_bytes = owner.dest_extent_bytes;
  uint32_t first_page = 0, page_count = 0;
  table.Arm(owner, &first_page, &page_count);
  CHECK(first_page == (0x10000u >> 12));
  CHECK(page_count == (owner.range_bytes + 4095) / 4096);

  // "Shared buffer" and "RAM" images covering the extent.
  const uint32_t image_bytes = 0x10000 + owner.range_bytes;
  std::vector<uint8_t> shared(image_bytes), ram(image_bytes, 0);
  for (uint32_t i = 0; i < image_bytes; ++i) {
    shared[i] = uint8_t(i * 5 + 3);
  }
  // Reference: the eager tiled walk restricted to one page.
  const uint32_t page = (0x10000u >> 12) + 2;
  const uint32_t page_start = page << 12, page_end = page_start + 4096;
  std::vector<uint8_t> expected(image_bytes, 0);
  const uint32_t run_texels = TiledPublishRunTexels(4);
  for (uint32_t y = 0; y < owner.height; ++y) {
    for (uint32_t x = 0; x < owner.width;) {
      uint32_t run = std::min(run_texels - (x & (run_texels - 1)),
                              owner.width - x);
      uint32_t off = uint32_t(texture_address::Tiled2D(int32_t(x), int32_t(y),
                                                       256, 2));
      uint32_t abs0 = owner.guest_addr + off, abs1 = abs0 + run * 4;
      uint32_t c0 = std::max(abs0, page_start), c1 = std::min(abs1, page_end);
      if (c0 < c1) {
        std::memcpy(&expected[c0], &shared[c0], c1 - c0);
      }
      x += run;
    }
  }

  std::vector<uint64_t> waited;
  uint32_t remaining = table.OnAccess(
      page_start + 100, 8,
      [&](uint64_t submission) {
        waited.push_back(submission);
        return true;
      },
      shared.data(),
      [&](uint32_t address) { return ram.data() + address; });
  CHECK(remaining == 0);
  REQUIRE(waited.size() == 1);
  CHECK(waited[0] == 3);
  CHECK(std::memcmp(ram.data(), expected.data(), image_bytes) == 0);
  // Exactly that page left the table; the rest of the owner is still armed.
  CHECK(table.armed_page_count() == page_count - 1);
  CHECK(table.Owns(7));
  // An owner that cannot be waited for (uncommitted) stays armed and counts
  // as a stale read.
  uint32_t stale = table.OnAccess(
      page_start + 4096, 4, [&](uint64_t) { return false; }, shared.data(),
      [&](uint32_t address) { return ram.data() + address; });
  CHECK(stale == 1);
  CHECK(table.armed_page_count() == page_count - 1);
  CHECK(table.Retire(7) == page_count - 1);
  CHECK(!table.Owns(7));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
