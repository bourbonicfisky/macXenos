/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 matrix for the pure memexport consumer classifier
// (pre-gate-metal-memexport-gpu-consumers task 2.2). Device-free, pure math.
//
// EVERY CASE ASSERTS INTERVAL SHAPE, NOT JUST BYTE TOTALS. That is the whole
// design of this file, and it is not a stylistic preference — task 2.1 shipped
// with a boundary-sweep bug that collapsed a four-interval partition into one
// "uncovered" interval while still reporting `IsExactPartition() == true`,
// because the byte accounting stayed self-consistent. Totals, counts and the
// exactness flag were all satisfied by a completely wrong answer.
//
// So the helper below compares the FULL ordered interval list — start, length,
// authority and writer sequence for every entry — and §"shape guard" contains a
// case constructed so that the byte totals are correct under the old bug and
// only the shape reveals it.

#include <cstdint>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memexport_consumer_classifier.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::ClassifiedInterval;
using metal::ClassifyLatestWriterIntervals;
using metal::ConsumerReadsSharedBuffer;
using metal::ConsumerRequiresRam;
using metal::GpuDirectEligibleBytes;
using metal::GuestRange;
using metal::GuestRangeSet;
using metal::IntervalAuthority;
using metal::IntervalAuthorityName;
using metal::IntervalClassification;
using metal::IntervalIsGpuDirectEligible;
using metal::PendingProducer;
using metal::PhysicalConsumerClass;
using metal::TransitState;

GuestRangeSet Request(uint32_t base, uint32_t length) {
  GuestRangeSet set;
  set.Add(*GuestRange::FromBaseLength(base, length));
  return set;
}

PendingProducer Producer(uint32_t base, uint32_t length, TransitState state,
                         uint64_t sequence) {
  // Aggregate-initialised: GuestRange has no default constructor, so
  // PendingProducer has none either.
  return PendingProducer{*GuestRange::FromBaseLength(base, length), state,
                         /*submission=*/1, sequence};
}

PendingProducer Memexport(uint32_t base, uint32_t length, uint64_t sequence) {
  return Producer(base, length, TransitState::kPendingMemexport, sequence);
}
PendingProducer Resolve(uint32_t base, uint32_t length, uint64_t sequence) {
  return Producer(base, length, TransitState::kPendingResolve, sequence);
}

struct Expected {
  uint32_t base;
  uint32_t length;
  IntervalAuthority authority;
  uint64_t sequence;
};

std::string Describe(const IntervalClassification& actual) {
  std::string out;
  for (const ClassifiedInterval& interval : actual.intervals) {
    out += "[" + std::to_string(interval.range.start()) + "+" +
           std::to_string(interval.range.length()) + " " +
           IntervalAuthorityName(interval.authority) + " seq=" +
           std::to_string(interval.writer_sequence) + "] ";
  }
  return out.empty() ? "(none)" : out;
}

// Asserts the COMPLETE ordered shape. Byte totals and the exactness flag are
// checked too, but they are the weakest assertions here, not the point.
void RequireShape(const IntervalClassification& actual,
                  const std::vector<Expected>& expected) {
  INFO("actual: " << Describe(actual));
  REQUIRE(actual.intervals.size() == expected.size());
  uint64_t expected_bytes = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    INFO("interval " << i);
    CHECK(actual.intervals[i].range.start() == expected[i].base);
    CHECK(actual.intervals[i].range.length() == expected[i].length);
    CHECK(actual.intervals[i].authority == expected[i].authority);
    CHECK(actual.intervals[i].writer_sequence == expected[i].sequence);
    expected_bytes += expected[i].length;
  }
  CHECK(actual.classified_bytes == expected_bytes);
  CHECK(actual.IsExactPartition());
  // Intervals must be ordered and disjoint — a partition, not a bag.
  for (size_t i = 1; i < actual.intervals.size(); ++i) {
    const uint64_t previous_end =
        uint64_t(actual.intervals[i - 1].range.start()) +
        actual.intervals[i - 1].range.length();
    CHECK(previous_end <= actual.intervals[i].range.start());
  }
}

constexpr IntervalAuthority kUncovered = IntervalAuthority::kUncovered;
constexpr IntervalAuthority kMemexport =
    IntervalAuthority::kLatestPendingMemexport;
constexpr IntervalAuthority kResolve = IntervalAuthority::kLatestPendingResolve;

}  // namespace

// ---------------------------------------------------------------------------
// Shape guard — the case that would have caught the 2.1 sweep bug
// ---------------------------------------------------------------------------

TEST_CASE("Byte totals cannot substitute for interval shape",
          "[memexport_classifier][shape]") {
  // Under the 2.1 boundary-sweep bug every producer edge after the first was
  // compared against the previously pushed cut instead of the request, so the
  // whole request collapsed into ONE uncovered interval — while
  // requested_bytes, classified_bytes and IsExactPartition() all still agreed.
  //
  // This case therefore asserts what only shape can express: the same 1024
  // bytes, split four ways, with two different writers and two uncovered gaps.
  const IntervalClassification actual = ClassifyLatestWriterIntervals(
      Request(0x1000, 0x400),
      {Memexport(0x1100, 0x100, 10), Resolve(0x1180, 0x100, 20)});

  // The weak assertions the bug satisfied:
  CHECK(actual.requested_bytes == 0x400);
  CHECK(actual.classified_bytes == 0x400);
  CHECK(actual.IsExactPartition());

  // The assertion it could not:
  RequireShape(actual, {{0x1000, 0x100, kUncovered, 0},
                        {0x1100, 0x080, kMemexport, 10},
                        {0x1180, 0x100, kResolve, 20},
                        {0x1280, 0x180, kUncovered, 0}});
}

// ---------------------------------------------------------------------------
// Supersession — both directions, decided by sequence and never by kind
// ---------------------------------------------------------------------------

TEST_CASE("A newer resolve supersedes an older memexport",
          "[memexport_classifier][supersession]") {
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x2000, 0x100),
                   {Memexport(0x2000, 0x100, 5), Resolve(0x2000, 0x100, 9)}),
               {{0x2000, 0x100, kResolve, 9}});
}

TEST_CASE("A newer memexport supersedes an older resolve",
          "[memexport_classifier][supersession]") {
  // The reverse direction must work identically. A classifier that ranked by
  // writer KIND rather than sequence would pass the case above and fail here.
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x2000, 0x100),
                   {Resolve(0x2000, 0x100, 5), Memexport(0x2000, 0x100, 9)}),
               {{0x2000, 0x100, kMemexport, 9}});
}

TEST_CASE("Registration order in the input does not decide authority",
          "[memexport_classifier][supersession]") {
  // Same two producers, opposite vector order, identical verdict: the sequence
  // decides, not the position in the list.
  const std::vector<Expected> expected = {{0x2000, 0x100, kMemexport, 9}};
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x2000, 0x100),
                   {Resolve(0x2000, 0x100, 5), Memexport(0x2000, 0x100, 9)}),
               expected);
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x2000, 0x100),
                   {Memexport(0x2000, 0x100, 9), Resolve(0x2000, 0x100, 5)}),
               expected);
}

// ---------------------------------------------------------------------------
// Coverage geometry
// ---------------------------------------------------------------------------

TEST_CASE("Partial overlap is exact at both edges",
          "[memexport_classifier][overlap]") {
  // Producer covers the middle only; the bytes on either side are uncovered
  // and must not be absorbed into the covered interval.
  RequireShape(ClassifyLatestWriterIntervals(Request(0x3000, 0x300),
                                             {Memexport(0x3100, 0x100, 4)}),
               {{0x3000, 0x100, kUncovered, 0},
                {0x3100, 0x100, kMemexport, 4},
                {0x3200, 0x100, kUncovered, 0}});
}

TEST_CASE("A producer wider than the request is clipped to the request",
          "[memexport_classifier][overlap]") {
  RequireShape(ClassifyLatestWriterIntervals(Request(0x4100, 0x100),
                                             {Memexport(0x4000, 0x400, 2)}),
               {{0x4100, 0x100, kMemexport, 2}});
}

TEST_CASE("Mixed ownership splits into one interval per writer",
          "[memexport_classifier][overlap]") {
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x5000, 0x300),
                   {Memexport(0x5000, 0x100, 1), Resolve(0x5100, 0x100, 2),
                    Memexport(0x5200, 0x100, 3)}),
               {{0x5000, 0x100, kMemexport, 1},
                {0x5100, 0x100, kResolve, 2},
                {0x5200, 0x100, kMemexport, 3}});
}

TEST_CASE("A single uncovered byte is its own interval",
          "[memexport_classifier][uncovered]") {
  // The smallest possible gap must survive: an off-by-one that swallowed it
  // would report full coverage of a range that has an unowned byte in it.
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x6000, 0x201),
                   {Memexport(0x6000, 0x100, 1), Memexport(0x6101, 0x100, 2)}),
               {{0x6000, 0x100, kMemexport, 1},
                {0x6100, 0x001, kUncovered, 0},
                {0x6101, 0x100, kMemexport, 2}});
}

TEST_CASE("Abutting producers stay split when the writer differs",
          "[memexport_classifier][adjacency]") {
  // Same authority, adjacent ranges, DIFFERENT sequences: two writers, so two
  // intervals. Coalescing them would erase which record owns which bytes,
  // which is what a retirement decision depends on.
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x7000, 0x200),
                   {Memexport(0x7000, 0x100, 1), Memexport(0x7100, 0x100, 2)}),
               {{0x7000, 0x100, kMemexport, 1},
                {0x7100, 0x100, kMemexport, 2}});
}

TEST_CASE("One producer spanning a cut coalesces into a single interval",
          "[memexport_classifier][adjacency]") {
  // A second producer's edge falls inside the first's span. The elementary
  // intervals on either side of that cut have the SAME latest writer, so they
  // must merge rather than appear as two rows for one record.
  RequireShape(ClassifyLatestWriterIntervals(
                   Request(0x8000, 0x200),
                   {Memexport(0x8000, 0x200, 5), Resolve(0x8100, 0x100, 1)}),
               {{0x8000, 0x200, kMemexport, 5}});
}

TEST_CASE("Disjoint producers leave the request wholly uncovered",
          "[memexport_classifier][disjoint]") {
  RequireShape(ClassifyLatestWriterIntervals(Request(0x9000, 0x100),
                                             {Memexport(0xA000, 0x100, 3)}),
               {{0x9000, 0x100, kUncovered, 0}});
}

TEST_CASE("An empty request is an exact partition of nothing",
          "[memexport_classifier][empty]") {
  const IntervalClassification actual =
      ClassifyLatestWriterIntervals(GuestRangeSet(), {Memexport(0x100, 8, 1)});
  RequireShape(actual, {});
  CHECK(actual.requested_bytes == 0);
  CHECK(actual.IsExactPartition());
}

TEST_CASE("No pending producers leaves the request uncovered, not empty",
          "[memexport_classifier][empty]") {
  RequireShape(ClassifyLatestWriterIntervals(Request(0xB000, 0x40), {}),
               {{0xB000, 0x40, kUncovered, 0}});
}

TEST_CASE("A request at the top of the address space saturates cleanly",
          "[memexport_classifier][saturation]") {
  const uint32_t top =
      uint32_t(GuestRange::kGuestPhysicalSize) - 0x100;
  RequireShape(
      ClassifyLatestWriterIntervals(Request(top, 0x100),
                                    {Memexport(top, 0x100, 7)}),
      {{top, 0x100, kMemexport, 7}});
}

TEST_CASE("A producer running past the end of the address space is clipped",
          "[memexport_classifier][saturation]") {
  const uint32_t top = uint32_t(GuestRange::kGuestPhysicalSize) - 0x80;
  // FromBaseLength clamps, so the producer covers only the in-bounds tail.
  RequireShape(ClassifyLatestWriterIntervals(Request(top, 0x80),
                                             {Memexport(top, 0x1000, 3)}),
               {{top, 0x80, kMemexport, 3}});
}

// ---------------------------------------------------------------------------
// Physical consumer class and eligibility
// ---------------------------------------------------------------------------

TEST_CASE("Shared-buffer consumers are eligible where CPU consumers are not",
          "[memexport_classifier][consumer]") {
  // The classification is identical; only eligibility differs. This is the
  // kVertex split from task 1.3 expressed as a decision: the same request can
  // carry shared vfetch bytes and CPU-staged index/factor bytes, and the label
  // alone cannot tell them apart.
  const IntervalClassification actual = ClassifyLatestWriterIntervals(
      Request(0xC000, 0x200), {Memexport(0xC000, 0x100, 4)});
  RequireShape(actual, {{0xC000, 0x100, kMemexport, 4},
                        {0xC100, 0x100, kUncovered, 0}});

  CHECK(GpuDirectEligibleBytes(actual, PhysicalConsumerClass::kSharedVfetch) ==
        0x100);
  CHECK(GpuDirectEligibleBytes(
            actual, PhysicalConsumerClass::kSharedTextureLoad) == 0x100);
  CHECK(GpuDirectEligibleBytes(actual,
                               PhysicalConsumerClass::kCpuIndexStream) == 0);
  CHECK(GpuDirectEligibleBytes(
            actual, PhysicalConsumerClass::kCpuTessFactorStream) == 0);
  CHECK(GpuDirectEligibleBytes(
            actual, PhysicalConsumerClass::kLegacyTextureCpuRead) == 0);
}

TEST_CASE("A newer resolve makes even a shared consumer ineligible",
          "[memexport_classifier][consumer]") {
  const IntervalClassification actual = ClassifyLatestWriterIntervals(
      Request(0xD000, 0x100),
      {Memexport(0xD000, 0x100, 1), Resolve(0xD000, 0x100, 2)});
  RequireShape(actual, {{0xD000, 0x100, kResolve, 2}});
  CHECK(GpuDirectEligibleBytes(actual, PhysicalConsumerClass::kSharedVfetch) ==
        0);
}

TEST_CASE("Uncovered bytes are never eligible for any consumer",
          "[memexport_classifier][consumer]") {
  const IntervalClassification actual =
      ClassifyLatestWriterIntervals(Request(0xE000, 0x100), {});
  for (PhysicalConsumerClass consumer :
       {PhysicalConsumerClass::kSharedVfetch,
        PhysicalConsumerClass::kSharedTextureLoad,
        PhysicalConsumerClass::kCpuIndexStream,
        PhysicalConsumerClass::kHostRead,
        PhysicalConsumerClass::kFrameEndPublication}) {
    CHECK(GpuDirectEligibleBytes(actual, consumer) == 0);
  }
}

TEST_CASE("The consumer partition is total and non-overlapping",
          "[memexport_classifier][consumer]") {
  // Every class is exactly one of: reads the shared buffer, requires RAM, or
  // reads no guest bytes. A class added later without a decision fails here
  // rather than defaulting into a category.
  const PhysicalConsumerClass all[] = {
      PhysicalConsumerClass::kSharedVfetch,
      PhysicalConsumerClass::kSharedTextureLoad,
      PhysicalConsumerClass::kCpuIndexStream,
      PhysicalConsumerClass::kCpuTessFactorStream,
      PhysicalConsumerClass::kLegacyTextureCpuRead,
      PhysicalConsumerClass::kWaitPredicate,
      PhysicalConsumerClass::kHostRead,
      PhysicalConsumerClass::kTextureCacheHit,
      PhysicalConsumerClass::kFrameEndPublication,
      PhysicalConsumerClass::kLifecyclePublication,
  };
  for (PhysicalConsumerClass consumer : all) {
    INFO("consumer " << metal::PhysicalConsumerClassName(consumer));
    const int categories = int(ConsumerReadsSharedBuffer(consumer)) +
                           int(ConsumerRequiresRam(consumer)) +
                           int(metal::ConsumerReadsNoGuestBytes(consumer));
    CHECK(categories == 1);
  }
  // Publication boundaries are RAM-required, never a separate escape hatch.
  CHECK(ConsumerRequiresRam(PhysicalConsumerClass::kFrameEndPublication));
  CHECK(ConsumerRequiresRam(PhysicalConsumerClass::kLifecyclePublication));
  CHECK(metal::ConsumerIsPublicationBoundary(
      PhysicalConsumerClass::kFrameEndPublication));
  CHECK(metal::ConsumerIsPublicationBoundary(
      PhysicalConsumerClass::kLifecyclePublication));
}

// ---------------------------------------------------------------------------
// Multi-range requests
// ---------------------------------------------------------------------------

TEST_CASE("A producer cannot fragment an unrelated requested range",
          "[memexport_classifier][shape]") {
  // Two disjoint requested ranges; the producer touches only the first. The
  // second must come back as one interval, not split at the producer's edges.
  GuestRangeSet request;
  request.Add(*GuestRange::FromBaseLength(0xF000, 0x200));
  request.Add(*GuestRange::FromBaseLength(0x10000, 0x100));
  RequireShape(
      ClassifyLatestWriterIntervals(request, {Memexport(0xF080, 0x80, 6)}),
      {{0xF000, 0x080, kUncovered, 0},
       {0xF080, 0x080, kMemexport, 6},
       {0xF100, 0x100, kUncovered, 0},
       {0x10000, 0x100, kUncovered, 0}});
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
