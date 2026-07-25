/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the ZPD visibility slot-allocation policy.
//
// MetalZPDVisibilityPool delegates every index/generation decision to
// MetalZPDSlotPool, so the invariants pinned here are the ones production runs.
// The one that matters for guest-visible correctness is generation safety: a
// resolve queued against a slot that has since been recycled must be dropped,
// not accumulated into whichever report owns that slot now. The offset
// invariants matter because Metal permits a given visibility-buffer offset to
// be selected only once per render pass.

#include "xenia/gpu/metal/metal_zpd_slot_pool.h"

#include <set>
#include <vector>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

struct Slot {
  uint32_t index;
  uint32_t generation;
  size_t offset;
};

Slot AcquireOrFail(MetalZPDSlotPool& pool) {
  Slot slot{};
  REQUIRE(pool.Acquire(slot.index, slot.generation, slot.offset));
  return slot;
}

TEST_CASE("ZPD slot pool hands out distinct slots and offsets", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(8);
  REQUIRE(pool.capacity() == 8);
  REQUIRE(pool.free_count() == 8);
  REQUIRE(pool.acquired_count() == 0);

  std::set<uint32_t> indices;
  std::set<size_t> offsets;
  for (uint32_t i = 0; i < 8; ++i) {
    Slot slot = AcquireOrFail(pool);
    REQUIRE(slot.index < 8);
    // Every live slot must own a distinct buffer offset; two segments sharing
    // one offset would both be invalid under Metal's once-per-pass rule.
    REQUIRE(indices.insert(slot.index).second);
    REQUIRE(offsets.insert(slot.offset).second);
    REQUIRE(slot.offset == MetalZPDSlotPool::OffsetForIndex(slot.index));
    REQUIRE(slot.offset == size_t(slot.index) * sizeof(uint64_t));
  }
  REQUIRE(pool.acquired_count() == 8);
  REQUIRE_FALSE(pool.has_free_indices());
}

TEST_CASE("ZPD slot pool reports exhaustion without corrupting state",
          "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(2);
  Slot a = AcquireOrFail(pool);
  Slot b = AcquireOrFail(pool);

  uint32_t index = 123;
  uint32_t generation = 456;
  size_t offset = 789;
  REQUIRE_FALSE(pool.Acquire(index, generation, offset));
  REQUIRE(index == MetalZPDSlotPool::kInvalidIndex);
  REQUIRE(generation == 0);
  REQUIRE(offset == 0);

  // A failed acquire must not have disturbed the live slots.
  REQUIRE(pool.acquired_count() == 2);
  REQUIRE(pool.IsGenerationCurrent(a.index, a.generation));
  REQUIRE(pool.IsGenerationCurrent(b.index, b.generation));

  // Releasing one makes exactly one available again.
  REQUIRE(pool.Release(a.index, a.generation));
  REQUIRE(pool.free_count() == 1);
  Slot c = AcquireOrFail(pool);
  REQUIRE(c.index == a.index);
  REQUIRE_FALSE(pool.has_free_indices());
}

TEST_CASE("ZPD slot generation invalidates a recycled slot", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(4);
  Slot first = AcquireOrFail(pool);
  REQUIRE(pool.IsGenerationCurrent(first.index, first.generation));

  REQUIRE(pool.Release(first.index, first.generation));
  // The moment the slot is released its old identity is stale, so a resolve
  // still queued against it is dropped rather than misattributed.
  REQUIRE_FALSE(pool.IsGenerationCurrent(first.index, first.generation));

  // Reacquire until the same physical index comes back around.
  Slot reused{};
  bool found = false;
  for (uint32_t i = 0; i < 4 && !found; ++i) {
    Slot slot = AcquireOrFail(pool);
    if (slot.index == first.index) {
      reused = slot;
      found = true;
    }
  }
  REQUIRE(found);
  REQUIRE(reused.generation != first.generation);
  REQUIRE(reused.offset == first.offset);
  // The new owner is current; the previous owner's handle stays stale.
  REQUIRE(pool.IsGenerationCurrent(reused.index, reused.generation));
  REQUIRE_FALSE(pool.IsGenerationCurrent(first.index, first.generation));
}

TEST_CASE("ZPD slot pool rejects a double release", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(4);
  Slot slot = AcquireOrFail(pool);

  REQUIRE(pool.Release(slot.index, slot.generation));
  REQUIRE(pool.free_count() == 4);

  // A second release with the same handle must change nothing; accepting it
  // would push a duplicate index and hand one offset to two live segments.
  REQUIRE_FALSE(pool.Release(slot.index, slot.generation));
  REQUIRE(pool.free_count() == 4);
}

TEST_CASE("ZPD slot pool rejects releasing a slot nobody holds", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(4);

  // A never-acquired slot still carries the initial generation. Trusting the
  // generation alone would accept this and duplicate the index.
  REQUIRE_FALSE(pool.Release(0, 0));
  REQUIRE(pool.free_count() == 4);

  Slot slot = AcquireOrFail(pool);
  // Stale generation on a live slot is equally rejected.
  REQUIRE_FALSE(pool.Release(slot.index, slot.generation + 1));
  REQUIRE(pool.acquired_count() == 1);
  REQUIRE(pool.IsGenerationCurrent(slot.index, slot.generation));
}

TEST_CASE("ZPD slot pool rejects out-of-range indices", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(4);
  REQUIRE_FALSE(pool.Release(4, 0));
  REQUIRE_FALSE(pool.Release(MetalZPDSlotPool::kInvalidIndex, 0));
  REQUIRE_FALSE(pool.IsGenerationCurrent(4, 0));
  REQUIRE_FALSE(pool.IsGenerationCurrent(MetalZPDSlotPool::kInvalidIndex, 0));
  REQUIRE(pool.free_count() == 4);
}

TEST_CASE("ZPD slot pool survives a full acquire/release cycle", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(16);

  // Several full generations of churn: every cycle must return the pool to a
  // complete free list with no leaked or duplicated index.
  for (uint32_t cycle = 0; cycle < 4; ++cycle) {
    std::vector<Slot> live;
    while (pool.has_free_indices()) {
      live.push_back(AcquireOrFail(pool));
    }
    REQUIRE(live.size() == 16);
    for (const Slot& slot : live) {
      REQUIRE(pool.Release(slot.index, slot.generation));
    }
    REQUIRE(pool.free_count() == 16);
    REQUIRE(pool.acquired_count() == 0);

    std::set<uint32_t> distinct;
    for (const Slot& slot : live) {
      distinct.insert(slot.index);
    }
    REQUIRE(distinct.size() == 16);
  }
}

TEST_CASE("ZPD slot pool capacity math matches the buffer layout", "[zpdpool]") {
  REQUIRE(MetalZPDSlotPool::kSlotSize == sizeof(uint64_t));
  REQUIRE(MetalZPDSlotPool::BufferSizeForCapacity(0) == 0);
  REQUIRE(MetalZPDSlotPool::BufferSizeForCapacity(1) == sizeof(uint64_t));
  REQUIRE(MetalZPDSlotPool::BufferSizeForCapacity(8192) ==
          size_t(8192) * sizeof(uint64_t));
  // The last slot must fit inside the buffer the owner allocates for it.
  REQUIRE(MetalZPDSlotPool::OffsetForIndex(8191) +
              MetalZPDSlotPool::kSlotSize ==
          MetalZPDSlotPool::BufferSizeForCapacity(8192));
}

TEST_CASE("ZPD slot pool with zero capacity never acquires", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(0);
  REQUIRE(pool.capacity() == 0);
  REQUIRE_FALSE(pool.has_free_indices());

  uint32_t index = 0;
  uint32_t generation = 0;
  size_t offset = 0;
  REQUIRE_FALSE(pool.Acquire(index, generation, offset));
  REQUIRE(index == MetalZPDSlotPool::kInvalidIndex);
}

TEST_CASE("ZPD slot pool Clear drops all state", "[zpdpool]") {
  MetalZPDSlotPool pool;
  pool.Reset(4);
  Slot slot = AcquireOrFail(pool);
  pool.Clear();

  REQUIRE(pool.capacity() == 0);
  REQUIRE(pool.free_count() == 0);
  REQUIRE_FALSE(pool.has_free_indices());
  // Handles from before the teardown must not validate against the empty pool.
  REQUIRE_FALSE(pool.IsGenerationCurrent(slot.index, slot.generation));
  REQUIRE_FALSE(pool.Release(slot.index, slot.generation));
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
