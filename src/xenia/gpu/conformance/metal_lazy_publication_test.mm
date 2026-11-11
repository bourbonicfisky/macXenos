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
#include <thread>
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

  // A write to a still-watched host page publishes first (kWriteUnlocked, then
  // kWriteNoWait under the global lock) and then proceeds; the byte lands in
  // the alias.
  const uint32_t write_offset = kSize - 4096;  // last host page, unread yet
  view[write_offset] = 0xEE;
  CHECK(alias[write_offset] == 0xEE);
  CHECK(recorder.Count(Memory::PhysicalAccess::kWriteUnlocked) >= 1);

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

TEST_CASE("lazy publication: OnAccess publishes the completed older owners",
          "[l0][metal][lazypub]") {
  // Two linear owners sharing page 0x20: A (completed, submission 1) covers
  // [0x20000, 0x20800), B (uncommitted, submission 2) covers [0x20400,
  // 0x20C00). A read must see A's output (the newest finished producer, as
  // hardware would before B executes) and leave the page armed for B.
  MetalLazyPublication table;
  MetalLazyPublication::Owner a;
  a.authority_sequence = 11;
  a.submission = 1;
  a.guest_addr = 0x20000;
  a.range_bytes = 0x800;
  a.linear = true;
  MetalLazyPublication::Owner b = a;
  b.authority_sequence = 12;
  b.submission = 2;
  b.guest_addr = 0x20400;
  table.Arm(a, nullptr, nullptr);
  table.Arm(b, nullptr, nullptr);

  const uint32_t image_bytes = 0x21000;
  std::vector<uint8_t> shared(image_bytes), ram(image_bytes, 0);
  for (uint32_t i = 0; i < image_bytes; ++i) {
    shared[i] = uint8_t(i * 7 + 1);
  }
  uint32_t remaining = table.OnAccess(
      0x20100, 4, [&](uint64_t submission) { return submission == 1; },
      shared.data(), [&](uint32_t address) { return ram.data() + address; });
  CHECK(remaining == 1);
  CHECK(std::memcmp(ram.data() + 0x20000, shared.data() + 0x20000, 0x800) ==
        0);
  for (uint32_t i = 0x20800; i < 0x20C00; ++i) {
    REQUIRE(ram[i] == 0);  // B's own bytes are not published yet.
  }
  CHECK(!table.Owns(11));
  CHECK(table.Owns(12));
  CHECK(table.stats().stale_reads == 1);
  CHECK(table.stats().stale_partial_pages == 1);

  // B completes: the next read publishes it.
  remaining = table.OnAccess(
      0x20100, 4, [&](uint64_t) { return true; }, shared.data(),
      [&](uint32_t address) { return ram.data() + address; });
  CHECK(remaining == 0);
  CHECK(std::memcmp(ram.data() + 0x20400, shared.data() + 0x20400, 0x800) ==
        0);
  CHECK(!table.Owns(12));
}

TEST_CASE("lazy publication: a page published during the wait is not copied "
          "again",
          "[l0][metal][lazypub]") {
  MetalLazyPublication table;
  MetalLazyPublication::Owner owner;
  owner.authority_sequence = 21;
  owner.submission = 1;
  owner.guest_addr = 0x30000;
  owner.range_bytes = 0x1000;
  owner.linear = true;
  table.Arm(owner, nullptr, nullptr);

  const uint32_t image_bytes = 0x31000;
  std::vector<uint8_t> shared(image_bytes), ram(image_bytes, 0);
  for (uint32_t i = 0; i < image_bytes; ++i) {
    shared[i] = uint8_t(i * 11 + 5);
  }
  const uint8_t cpu_byte = uint8_t(~shared[0x30010]);
  // While the access waits for the submission, the drain publishes the
  // record and retires it, and the guest (the page is open now) writes.
  uint32_t remaining = table.OnAccess(
      0x30100, 4,
      [&](uint64_t) {
        std::memcpy(ram.data() + 0x30000, shared.data() + 0x30000, 0x1000);
        CHECK(table.Retire(21) == 1);
        ram[0x30010] = cpu_byte;
        return true;
      },
      shared.data(), [&](uint32_t address) { return ram.data() + address; });
  CHECK(remaining == 0);
  CHECK(ram[0x30010] == cpu_byte);
  CHECK(table.stats().already_published == 1);
  CHECK(table.stats().fault_pages_copied == 0);

  // Superseded during the wait: a newer record owns the page. The access
  // still gets the record it waited for, and the page stays armed.
  MetalLazyPublication::Owner older = owner;
  older.authority_sequence = 31;
  older.guest_addr = 0x30000;
  older.range_bytes = 0x800;
  MetalLazyPublication::Owner newer = older;
  newer.authority_sequence = 32;
  newer.submission = 2;
  table.Arm(older, nullptr, nullptr);
  std::fill(ram.begin(), ram.end(), uint8_t(0));
  remaining = table.OnAccess(
      0x30100, 4,
      [&](uint64_t submission) {
        CHECK(submission == 1);
        CHECK(table.Retire(31) == 1);
        table.Arm(newer, nullptr, nullptr);
        return true;
      },
      shared.data(), [&](uint32_t address) { return ram.data() + address; });
  CHECK(remaining == 0);
  CHECK(std::memcmp(ram.data() + 0x30000, shared.data() + 0x30000, 0x800) ==
        0);
  CHECK(table.stats().superseded_copies == 1);
  CHECK(table.Owns(32));
}

TEST_CASE("lazy publication: two accesses to one page copy it once",
          "[l0][metal][lazypub]") {
  MetalLazyPublication table;
  MetalLazyPublication::Owner owner;
  owner.authority_sequence = 41;
  owner.submission = 1;
  owner.guest_addr = 0x50000;
  owner.range_bytes = 0x1000;
  owner.linear = true;
  table.Arm(owner, nullptr, nullptr);

  const uint32_t image_bytes = 0x51000;
  std::vector<uint8_t> shared(image_bytes), ram(image_bytes, 0);
  for (uint32_t i = 0; i < image_bytes; ++i) {
    shared[i] = uint8_t(i * 13 + 7);
  }
  // Both accesses take their snapshot before either copies.
  std::atomic<int> waiting{0};
  auto wait = [&](uint64_t) {
    ++waiting;
    while (waiting.load() < 2) {
      std::this_thread::yield();
    }
    return true;
  };
  auto translate = [&](uint32_t address) { return ram.data() + address; };
  uint32_t remaining_other = 1;
  std::thread other([&] {
    remaining_other =
        table.OnAccess(0x50800, 4, wait, shared.data(), translate);
  });
  uint32_t remaining = table.OnAccess(0x50100, 4, wait, shared.data(),
                                      translate);
  other.join();
  CHECK(remaining == 0);
  CHECK(remaining_other == 0);
  CHECK(std::memcmp(ram.data() + 0x50000, shared.data() + 0x50000, 0x1000) ==
        0);
  CHECK(table.stats().fault_pages_copied == 1);
  CHECK(table.stats().already_published == 1);
  CHECK(!table.Owns(41));
}

TEST_CASE("lazy publication: OldestOwnerBefore finds older overlapping owners",
          "[l0][metal][lazypub]") {
  MetalLazyPublication table;
  auto linear = [](uint64_t sequence, uint32_t addr, uint32_t bytes) {
    MetalLazyPublication::Owner owner;
    owner.authority_sequence = sequence;
    owner.submission = 1;
    owner.guest_addr = addr;
    owner.range_bytes = bytes;
    owner.linear = true;
    return owner;
  };
  // Partial-page owners accumulate; a whole-page linear owner would replace.
  table.Arm(linear(5, 0x10100, 0x1E00), nullptr, nullptr);
  table.Arm(linear(3, 0x10800, 0x100), nullptr, nullptr);
  table.Arm(linear(9, 0x30000, 0x100), nullptr, nullptr);

  // Overlapping and older: the oldest of the two wins.
  REQUIRE(table.OldestOwnerBefore(0x10800, 0x80, 7) == 3);
  REQUIRE(table.OldestOwnerBefore(0x11000, 0x80, 7) == 5);
  // Not older (equal or newer sequence) is never reported.
  REQUIRE(table.OldestOwnerBefore(0x11000, 0x80, 5) == 0);
  REQUIRE(table.OldestOwnerBefore(0x30000, 0x100, 9) == 0);
  // Same page, disjoint bytes: byte overlap, not page overlap, decides.
  REQUIRE(table.OldestOwnerBefore(0x30800, 0x100, 10) == 0);
  REQUIRE(table.OldestOwnerBefore(0x300F0, 0x100, 10) == 9);
  // Adjacent past the end.
  REQUIRE(table.OldestOwnerBefore(0x11F00, 0x100, 7) == 0);
  REQUIRE(table.OldestOwnerBefore(0x11000, 0, 7) == 0);
  // Retired owners stop being reported.
  table.Retire(3);
  REQUIRE(table.OldestOwnerBefore(0x10800, 0x80, 7) == 5);
  table.Retire(5);
  REQUIRE(table.OldestOwnerBefore(0x10800, 0x80, 7) == 0);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
