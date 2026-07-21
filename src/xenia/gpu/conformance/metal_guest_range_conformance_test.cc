/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free) conformance for the cross-writer memory-authority guest
// range type (OpenSpec implement-metal-cross-writer-memory-authority, tasks 2.1
// and 2.2). These are the wrap/overflow/boundary contracts that the audit's
// blocker 7 says no production routing may depend on until proven here:
//   * zero-length rejection and end-of-memory clamping with no 32-bit wrap;
//   * exact / partial / adjacent (NOT overlap) / disjoint / alias overlap;
//   * range-set operations that preserve disjoint base+mip extents instead of
//     collapsing them into one conservative envelope.
// Pure math + Catch2; links no Metal device and no command processor.

#include <cstdint>
#include <optional>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_guest_range.h"

namespace xe {
namespace gpu {
namespace metal {
namespace test {

using Relation = GuestRange::Relation;

static constexpr uint64_t kEnd = GuestRange::kGuestPhysicalSize;  // 0x20000000

TEST_CASE("GuestRange domain is the fixed 512 MiB physical space", "[l0][authority]") {
  REQUIRE(GuestRange::kGuestPhysicalSize == 0x20000000ull);
  REQUIRE(GuestRange::kGuestPhysicalSize == uint64_t(SharedMemory::kBufferSize));
}

TEST_CASE("GuestRange rejects zero-length input", "[l0][authority]") {
  REQUIRE_FALSE(GuestRange::FromBaseLength(0x1000u, 0u).has_value());
  REQUIRE_FALSE(GuestRange::FromBaseLength(0u, 0u).has_value());
  // Empty explicit bounds are rejected too.
  REQUIRE_FALSE(GuestRange::FromBounds(0x1000u, 0x1000u).has_value());
  REQUIRE_FALSE(GuestRange::FromBounds(0x2000u, 0x1000u).has_value());
}

TEST_CASE("GuestRange rejects a base at or past end of memory", "[l0][authority]") {
  REQUIRE_FALSE(GuestRange::FromBaseLength(uint32_t(kEnd), 0x100u).has_value());
  REQUIRE_FALSE(GuestRange::FromBounds(kEnd, kEnd + 0x100u).has_value());
  // The last representable byte is still a valid one-byte range.
  auto last = GuestRange::FromBaseLength(uint32_t(kEnd) - 1u, 1u);
  REQUIRE(last.has_value());
  REQUIRE(last->start() == uint32_t(kEnd) - 1u);
  REQUIRE(last->end() == uint32_t(kEnd));
  REQUIRE(last->length() == 1u);
}

TEST_CASE("GuestRange clamps at end of memory without 32-bit wrap", "[l0][authority]") {
  // base + length overflows 32 bits (0xFFF00000 + 0x00200000 wraps to 0x100000
  // in 32-bit math). The clamped, overflow-safe result is [base, 512 MiB).
  auto wrap = GuestRange::FromBaseLength(0xFFF00000u, 0x00200000u);
  // base is already past 512 MiB, so there is no in-bounds byte at all.
  REQUIRE_FALSE(wrap.has_value());

  // A base just below the boundary with a length that overflows 32-bit end
  // arithmetic must still clamp to exactly 512 MiB, never a small wrapped value.
  auto clamped = GuestRange::FromBaseLength(uint32_t(kEnd) - 0x1000u, 0xFFFFFFFFu);
  REQUIRE(clamped.has_value());
  REQUIRE(clamped->start() == uint32_t(kEnd) - 0x1000u);
  REQUIRE(clamped->end() == uint32_t(kEnd));
  REQUIRE(clamped->length() == 0x1000u);

  // Explicit bounds that run past the domain clamp the end only.
  auto over = GuestRange::FromBounds(0x1000u, kEnd + 0x5000u);
  REQUIRE(over.has_value());
  REQUIRE(over->start() == 0x1000u);
  REQUIRE(over->end() == uint32_t(kEnd));
}

TEST_CASE("GuestRange classifies exact ranges", "[l0][authority]") {
  auto a = GuestRange::FromBaseLength(0x1000u, 0x400u);
  auto b = GuestRange::FromBaseLength(0x1000u, 0x400u);
  REQUIRE(a.has_value());
  REQUIRE(b.has_value());
  REQUIRE(*a == *b);
  REQUIRE(a->Classify(*b) == Relation::kExact);
  REQUIRE(a->Overlaps(*b));
}

TEST_CASE("GuestRange classifies partial overlap", "[l0][authority]") {
  auto a = GuestRange::FromBounds(0x1000u, 0x2000u);
  auto b = GuestRange::FromBounds(0x1800u, 0x2800u);
  REQUIRE(a->Overlaps(*b));
  REQUIRE(a->Classify(*b) == Relation::kPartial);
  REQUIRE(b->Classify(*a) == Relation::kPartial);
  auto isect = a->Intersection(*b);
  REQUIRE(isect.has_value());
  REQUIRE(isect->start() == 0x1800u);
  REQUIRE(isect->end() == 0x2000u);
}

TEST_CASE("GuestRange classifies containment", "[l0][authority]") {
  auto outer = GuestRange::FromBounds(0x1000u, 0x4000u);
  auto inner = GuestRange::FromBounds(0x2000u, 0x3000u);
  REQUIRE(outer->Contains(*inner));
  REQUIRE_FALSE(inner->Contains(*outer));
  REQUIRE(outer->Classify(*inner) == Relation::kContains);
  REQUIRE(inner->Classify(*outer) == Relation::kContainedBy);
}

TEST_CASE("GuestRange treats adjacency as non-overlap", "[l0][authority]") {
  // b begins exactly where a ends: touching, zero shared bytes.
  auto a = GuestRange::FromBounds(0x1000u, 0x2000u);
  auto b = GuestRange::FromBounds(0x2000u, 0x3000u);
  REQUIRE_FALSE(a->Overlaps(*b));
  REQUIRE(a->Adjacent(*b));
  REQUIRE(b->Adjacent(*a));
  REQUIRE(a->Classify(*b) == Relation::kAdjacent);
  REQUIRE(b->Classify(*a) == Relation::kAdjacent);
  // Adjacency shares no bytes, so there is no intersection range.
  REQUIRE_FALSE(a->Intersection(*b).has_value());
}

TEST_CASE("GuestRange classifies disjoint ranges", "[l0][authority]") {
  auto a = GuestRange::FromBounds(0x1000u, 0x2000u);
  auto b = GuestRange::FromBounds(0x3000u, 0x4000u);  // gap [0x2000,0x3000)
  REQUIRE_FALSE(a->Overlaps(*b));
  REQUIRE_FALSE(a->Adjacent(*b));
  REQUIRE(a->Classify(*b) == Relation::kDisjoint);
  REQUIRE_FALSE(a->Intersection(*b).has_value());
}

TEST_CASE("GuestRangeSet preserves aliased (identical) consumer ranges",
          "[l0][authority][alias]") {
  // Multiple consumers naming the same bytes: stored verbatim, then coalesced.
  GuestRangeSet set;
  REQUIRE(set.Add(GuestRange::FromBaseLength(0x8000u, 0x1000u)));
  REQUIRE(set.Add(GuestRange::FromBaseLength(0x8000u, 0x1000u)));
  REQUIRE(set.Add(GuestRange::FromBaseLength(0x8000u, 0x1000u)));
  REQUIRE(set.size() == 3u);
  set.Normalize();
  REQUIRE(set.size() == 1u);
  REQUIRE(set.ranges()[0].start() == 0x8000u);
  REQUIRE(set.ranges()[0].end() == 0x9000u);
  REQUIRE(set.CoveredBytes() == 0x1000u);
}

TEST_CASE("GuestRangeSet preserves disjoint base/mip texture extents",
          "[l0][authority][alias]") {
  // A texture supplies its precise base-level and packed-mip extents. They are
  // separated by a gap and MUST survive normalization as two ranges, not one
  // conservative envelope spanning the hole.
  GuestRangeSet set;
  REQUIRE(set.Add(GuestRange::FromBaseLength(0x10000u, 0x8000u)));   // base level
  REQUIRE(set.Add(GuestRange::FromBaseLength(0x30000u, 0x2000u)));   // mip tail
  set.Normalize();
  REQUIRE(set.size() == 2u);
  REQUIRE(set.ranges()[0].start() == 0x10000u);
  REQUIRE(set.ranges()[0].end() == 0x18000u);
  REQUIRE(set.ranges()[1].start() == 0x30000u);
  REQUIRE(set.ranges()[1].end() == 0x32000u);
  // Covered bytes count each extent once; the gap is NOT covered.
  REQUIRE(set.CoveredBytes() == 0x8000u + 0x2000u);
}

TEST_CASE("GuestRangeSet coalesces touching and overlapping ranges",
          "[l0][authority]") {
  GuestRangeSet set;
  set.Add(*GuestRange::FromBounds(0x1000u, 0x2000u));
  set.Add(*GuestRange::FromBounds(0x2000u, 0x3000u));  // adjacent -> merges
  set.Add(*GuestRange::FromBounds(0x2800u, 0x3800u));  // overlaps -> merges
  set.Add(*GuestRange::FromBounds(0x5000u, 0x6000u));  // disjoint -> separate
  set.Normalize();
  REQUIRE(set.size() == 2u);
  REQUIRE(set.ranges()[0].start() == 0x1000u);
  REQUIRE(set.ranges()[0].end() == 0x3800u);
  REQUIRE(set.ranges()[1].start() == 0x5000u);
  REQUIRE(set.ranges()[1].end() == 0x6000u);
}

TEST_CASE("GuestRangeSet overlap queries ignore adjacency", "[l0][authority]") {
  GuestRangeSet producers;
  producers.Add(*GuestRange::FromBounds(0x1000u, 0x2000u));
  producers.Add(*GuestRange::FromBounds(0x4000u, 0x5000u));

  // A consumer that only touches a producer at an endpoint has no dependency.
  auto adjacent_consumer = GuestRange::FromBounds(0x2000u, 0x3000u);
  REQUIRE_FALSE(producers.OverlapsAny(*adjacent_consumer));
  REQUIRE(producers.CollectOverlapping(*adjacent_consumer).empty());

  // A disjoint consumer in the gap has no dependency.
  auto gap_consumer = GuestRange::FromBounds(0x3000u, 0x3800u);
  REQUIRE_FALSE(producers.OverlapsAny(*gap_consumer));

  // A consumer that shares a byte selects only the intersecting producer.
  auto real_consumer = GuestRange::FromBounds(0x1800u, 0x1900u);
  REQUIRE(producers.OverlapsAny(*real_consumer));
  auto selected = producers.CollectOverlapping(*real_consumer);
  REQUIRE(selected.size() == 1u);
  REQUIRE(selected[0].start() == 0x1000u);
  REQUIRE(selected[0].end() == 0x2000u);
}

TEST_CASE("GuestRangeSet selective query spans multiple producers",
          "[l0][authority]") {
  GuestRangeSet producers;
  producers.Add(*GuestRange::FromBounds(0x1000u, 0x2000u));  // resolve record
  producers.Add(*GuestRange::FromBounds(0x3000u, 0x4000u));  // memexport record
  producers.Add(*GuestRange::FromBounds(0x9000u, 0xA000u));  // unrelated record

  // A wide consumer overlapping the first two but not the third selects exactly
  // those two -- the disjoint third adds no drain.
  auto consumer = GuestRange::FromBounds(0x1800u, 0x3800u);
  auto selected = producers.CollectOverlapping(*consumer);
  REQUIRE(selected.size() == 2u);
  REQUIRE(selected[0].start() == 0x1000u);
  REQUIRE(selected[1].start() == 0x3000u);
}

TEST_CASE("GuestRangeSet cross-set overlap detects and clears aliases",
          "[l0][authority][alias]") {
  GuestRangeSet consumer;
  consumer.Add(*GuestRange::FromBounds(0x10000u, 0x18000u));
  consumer.Add(*GuestRange::FromBounds(0x30000u, 0x32000u));

  GuestRangeSet overlapping;
  overlapping.Add(*GuestRange::FromBounds(0x31000u, 0x40000u));  // hits the mip
  REQUIRE(consumer.OverlapsAny(overlapping));

  GuestRangeSet disjoint;
  disjoint.Add(*GuestRange::FromBounds(0x18000u, 0x20000u));  // adjacent to base
  disjoint.Add(*GuestRange::FromBounds(0x20000u, 0x30000u));  // fills the gap
  REQUIRE_FALSE(consumer.OverlapsAny(disjoint));
}

TEST_CASE("GuestRange end-of-memory ranges classify and query normally",
          "[l0][authority]") {
  auto tail = GuestRange::FromBounds(kEnd - 0x1000u, kEnd);
  auto probe = GuestRange::FromBounds(kEnd - 0x400u, kEnd + 0x400u);  // clamped
  REQUIRE(tail.has_value());
  REQUIRE(probe.has_value());
  REQUIRE(probe->end() == uint32_t(kEnd));
  REQUIRE(tail->Overlaps(*probe));
  REQUIRE(tail->Classify(*probe) == Relation::kContains);

  GuestRangeSet set;
  set.Add(*tail);
  REQUIRE(set.OverlapsAny(*probe));
}

}  // namespace test
}  // namespace gpu
}  // namespace metal
}  // namespace xe
