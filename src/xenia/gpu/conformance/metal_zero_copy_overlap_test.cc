/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free): the in-flight CPU rewrite evidence model --
// pre-gate-metal-zero-copy-shared-memory tasks 5.1-5.4.
//
// The device half (a real MetalSharedMemory, the real physical invalidation
// thunk, the real providers) lives in metal_zero_copy_overlap_device_test.mm
// and is skipped on a host with no Metal device. THESE cases are mandatory
// everywhere, and they are where the rules themselves are pinned, because
// almost every rule that matters here is unreachable on a device:
//
//   * the completed/in-flight boundary needs a submission that has completed
//     to exactly the stamp -- a timing coincidence a device test cannot force;
//   * counter saturation is at 2^64, which no run can count to;
//   * a range that starts past the guest physical domain is not something the
//     backend can ask for.
//
// As values they are all enumerable in microseconds, and production calls these
// exact functions, so a rule deleted in production fails here.

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_zero_copy_overlap.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// A small domain so page arithmetic is checkable by hand: 64 pages of 4 KiB.
constexpr uint32_t kPageLog2 = 12;
constexpr uint32_t kPageBytes = 1u << kPageLog2;
constexpr uint32_t kPages = 64;
constexpr uint64_t kDomainBytes = uint64_t(kPages) << kPageLog2;

// Every declared class, checked for completeness rather than asserted in prose.
constexpr std::array kOverlapClasses = {
    ZeroCopyOverlapClass::kIdle,
    ZeroCopyOverlapClass::kCompletedReader,
    ZeroCopyOverlapClass::kInFlightReader,
};
static_assert(kOverlapClasses.size() == kZeroCopyOverlapClassCount,
              "add the new ZeroCopyOverlapClass enumerator to kOverlapClasses");

}  // namespace

TEST_CASE("Zero-copy overlap vocabulary: every class is a distinct named value "
          "and the sentinel is never one",
          "[l0][metal][zerocopy][overlap]") {
  for (uint32_t i = 0; i < kOverlapClasses.size(); ++i) {
    // Contiguous 0..kCount-1, so the sweep above cannot silently shrink.
    CHECK(static_cast<uint32_t>(kOverlapClasses[i]) == i);
    CHECK(IsZeroCopyOverlapClassValue(kOverlapClasses[i]));
    CHECK(std::string(ZeroCopyOverlapClassName(kOverlapClasses[i])) !=
          "unknown");
  }
  CHECK(ZeroCopyOverlapClassName(ZeroCopyOverlapClass::kIdle) ==
        std::string("idle"));
  CHECK(ZeroCopyOverlapClassName(ZeroCopyOverlapClass::kCompletedReader) ==
        std::string("completed_reader"));
  CHECK(ZeroCopyOverlapClassName(ZeroCopyOverlapClass::kInFlightReader) ==
        std::string("in_flight_reader"));

  // kCount is a bound, never an observation, and never carries a name.
  CHECK_FALSE(IsZeroCopyOverlapClassValue(ZeroCopyOverlapClass::kCount));
  CHECK(ZeroCopyOverlapClassName(ZeroCopyOverlapClass::kCount) ==
        std::string("unknown"));
  CHECK_FALSE(IsZeroCopyOverlapClassValue(
      static_cast<ZeroCopyOverlapClass>(kZeroCopyOverlapClassCount + 7)));
}

TEST_CASE("Zero-copy overlap classification: idle, completed, and unfinished "
          "are three different facts",
          "[l0][metal][zerocopy][overlap]") {
  // Stamp zero is the idle sentinel and is decided BEFORE any comparison. It
  // must stay idle against every completed mark, including zero -- a page
  // nobody ever claimed is not "a reader that finished".
  CHECK(ClassifyZeroCopyOverlap(0, 0) == ZeroCopyOverlapClass::kIdle);
  CHECK(ClassifyZeroCopyOverlap(0, 1) == ZeroCopyOverlapClass::kIdle);
  CHECK(ClassifyZeroCopyOverlap(0, UINT64_MAX) == ZeroCopyOverlapClass::kIdle);

  // Strictly newer than completed -> the rewrite raced an unfinished reader.
  CHECK(ClassifyZeroCopyOverlap(1, 0) == ZeroCopyOverlapClass::kInFlightReader);
  CHECK(ClassifyZeroCopyOverlap(9, 8) == ZeroCopyOverlapClass::kInFlightReader);
  CHECK(ClassifyZeroCopyOverlap(UINT64_MAX, UINT64_MAX - 1) ==
        ZeroCopyOverlapClass::kInFlightReader);

  // THE BOUNDARY. stamp == completed means that reader is the newest one the
  // GPU has FINISHED, so this is not a race. A `>=` here would report every
  // just-completed reader as in flight, and the counter would be nonzero on
  // every run -- which is the same as having no counter.
  CHECK(ClassifyZeroCopyOverlap(1, 1) ==
        ZeroCopyOverlapClass::kCompletedReader);
  CHECK(ClassifyZeroCopyOverlap(8, 8) ==
        ZeroCopyOverlapClass::kCompletedReader);
  CHECK(ClassifyZeroCopyOverlap(8, 9) ==
        ZeroCopyOverlapClass::kCompletedReader);
  CHECK(ClassifyZeroCopyOverlap(1, UINT64_MAX) ==
        ZeroCopyOverlapClass::kCompletedReader);
}

TEST_CASE("Zero-copy overlap counters saturate instead of wrapping",
          "[l0][metal][zerocopy][overlap]") {
  // The saturation boundary is 2^64: no run, and no test, can reach it by
  // counting. It is pinned here as a value, and production accumulates through
  // this exact function -- a counter that wrapped would read as a small number,
  // i.e. as "almost nothing overlapped", which is the one claim this pre-gate
  // exists to measure.
  CHECK(ZeroCopySaturatingAdd(0, 0) == 0u);
  CHECK(ZeroCopySaturatingAdd(0, 5) == 5u);
  CHECK(ZeroCopySaturatingAdd(5, 7) == 12u);
  CHECK(ZeroCopySaturatingAdd(UINT64_MAX - 1, 1) == UINT64_MAX);
  CHECK(ZeroCopySaturatingAdd(UINT64_MAX - 1, 2) == UINT64_MAX);
  CHECK(ZeroCopySaturatingAdd(UINT64_MAX - 3, 1000) == UINT64_MAX);
  CHECK(ZeroCopySaturatingAdd(UINT64_MAX, 1) == UINT64_MAX);
  CHECK(ZeroCopySaturatingAdd(UINT64_MAX, UINT64_MAX) == UINT64_MAX);
  CHECK(ZeroCopySaturatingAdd(1, UINT64_MAX) == UINT64_MAX);

  // The atomic form production uses, at the same boundary.
  std::atomic<uint64_t> counter{UINT64_MAX - 2};
  ZeroCopySaturatingAdd(counter, 1);
  CHECK(counter.load() == UINT64_MAX - 1);
  ZeroCopySaturatingAdd(counter, 10);
  CHECK(counter.load() == UINT64_MAX);
  ZeroCopySaturatingAdd(counter, 1);
  CHECK(counter.load() == UINT64_MAX);
}

TEST_CASE("Zero-copy overlap page span: first, last, and clamped ranges",
          "[l0][metal][zerocopy][overlap]") {
  auto span = [](uint64_t start, uint64_t length) {
    return ClampZeroCopyPageSpan(start, length, kPageLog2, kPages);
  };

  // The FIRST page, from its exact base and from a byte inside it.
  CHECK(span(0, 1).first_page == 0u);
  CHECK(span(0, 1).page_count == 1u);
  CHECK_FALSE(span(0, 1).clamped);
  CHECK(span(1, 1).first_page == 0u);
  CHECK(span(kPageBytes - 1, 1).page_count == 1u);
  // One byte either side of a page boundary is two pages.
  CHECK(span(kPageBytes - 1, 2).first_page == 0u);
  CHECK(span(kPageBytes - 1, 2).page_count == 2u);

  // The LAST page, and a range ending exactly at the domain end -- which is in
  // range, not clamped. Hardening must not shrink the legitimate domain.
  CHECK(span(kDomainBytes - kPageBytes, kPageBytes).first_page == kPages - 1);
  CHECK(span(kDomainBytes - kPageBytes, kPageBytes).page_count == 1u);
  CHECK_FALSE(span(kDomainBytes - kPageBytes, kPageBytes).clamped);
  CHECK(span(0, kDomainBytes).page_count == kPages);
  CHECK_FALSE(span(0, kDomainBytes).clamped);

  // Running PAST the end: clamped to the domain, never past the fixed table.
  CHECK(span(kDomainBytes - kPageBytes, kPageBytes + 1).page_count == 1u);
  CHECK(span(kDomainBytes - kPageBytes, kPageBytes + 1).clamped);
  CHECK(span(0, kDomainBytes + 1).page_count == kPages);
  CHECK(span(0, kDomainBytes + 1).clamped);

  // Starting AT or past the end observes nothing, and says it was clamped.
  CHECK(span(kDomainBytes, kPageBytes).page_count == 0u);
  CHECK(span(kDomainBytes, kPageBytes).clamped);
  CHECK(span(kDomainBytes * 4, 1).page_count == 0u);
  CHECK(span(kDomainBytes * 4, 1).clamped);

  // Overflow-safe: start + length is never formed, so a maximal length from a
  // valid base is an ordinary clamp rather than a wrap into page 0.
  CHECK(span(kPageBytes, UINT64_MAX).first_page == 1u);
  CHECK(span(kPageBytes, UINT64_MAX).page_count == kPages - 1);
  CHECK(span(kPageBytes, UINT64_MAX).clamped);
  CHECK(span(UINT64_MAX - 1, UINT64_MAX).page_count == 0u);
  CHECK(span(UINT64_MAX, UINT64_MAX).page_count == 0u);

  // Degenerate inputs observe nothing and are NOT reported as clamped: an empty
  // request was not truncated, it was empty.
  CHECK(span(0, 0).page_count == 0u);
  CHECK_FALSE(span(0, 0).clamped);
  CHECK(ClampZeroCopyPageSpan(0, kPageBytes, kPageLog2, 0).page_count == 0u);
  // A shift at or past the width of the domain product is rejected outright.
  CHECK(ClampZeroCopyPageSpan(0, kPageBytes, 32, kPages).page_count == 0u);
  CHECK(ClampZeroCopyPageSpan(0, kPageBytes, 64, kPages).page_count == 0u);
}

TEST_CASE("Zero-copy overlap observer: an uninitialized observer is inert and "
          "silent",
          "[l0][metal][zerocopy][overlap]") {
  // This is the structural half of the exact-off/fallback silence requirement:
  // mirror mode never initializes the observer, so its production entry points
  // do nothing even when a caller drives them.
  ZeroCopyOverlapObserver observer;
  CHECK_FALSE(observer.active());
  observer.StampConsumer(0, kPageBytes, 7);
  observer.ObserveCpuWrite(0, kPageBytes, 1);

  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK_FALSE(snapshot.active);
  CHECK(snapshot.stamp_calls == 0u);
  CHECK(snapshot.stamped_pages == 0u);
  CHECK(snapshot.cpu_write_callbacks == 0u);
  CHECK(snapshot.cpu_write_pages == 0u);
  CHECK(snapshot.in_flight_cpu_write_overlap == 0u);
  CHECK(snapshot.sample_count == 0u);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(snapshot));
}

TEST_CASE("Zero-copy overlap observer: a forced rejection leaves exactly what "
          "a real arming failure leaves",
          "[l0][metal][zerocopy][overlap]") {
  // The injected rejection (kZeroCopyForceObserverRejectEnvVar in production)
  // exists because a genuine failure here is an allocation failure over a
  // 512 MiB domain, which no test can produce on demand -- so without it the
  // "an unobservable alias is not accepted" rule could only be reviewed, never
  // run. It must therefore be indistinguishable from the real thing: refused,
  // inactive, and inert.
  ZeroCopyOverlapObserver observer;
  CHECK_FALSE(observer.Initialize(kDomainBytes, kPageLog2,
                                  /*force_rejection=*/true));
  CHECK_FALSE(observer.active());
  observer.StampConsumer(0, kPageBytes, 7);
  observer.ObserveCpuWrite(0, kPageBytes, 1);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(observer.Snapshot()));

  // It is checked FIRST, so it rejects inputs that would otherwise succeed and
  // does not depend on the domain being bad.
  CHECK_FALSE(observer.Initialize(kDomainBytes, kPageLog2, true));
  CHECK_FALSE(observer.active());

  // And it is not sticky: the same observer arms normally without it, so the
  // rejection is the injected input rather than a latched state.
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  CHECK(observer.active());
  CHECK(observer.Snapshot().page_count == kPages);

  // Arming over an ALREADY ACTIVE observer with the injection still refuses,
  // and leaves it inactive rather than half-armed with the previous table.
  CHECK_FALSE(observer.Initialize(kDomainBytes, kPageLog2, true));
  CHECK_FALSE(observer.active());
  CHECK(observer.Snapshot().page_count == 0u);
  CHECK(observer.Snapshot().page_size_log2 == 0u);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(observer.Snapshot()));

  // The default is off: an ordinary two-argument call is never a rejection.
  ZeroCopyOverlapObserver defaulted;
  CHECK(defaulted.Initialize(kDomainBytes, kPageLog2));
  CHECK(defaulted.active());
}

TEST_CASE("Zero-copy overlap report schedule: first boundary, interval, and "
          "the degenerate cadences",
          "[l0][metal][zerocopy][overlap]") {
  // The reporter runs in the command processor, which is not linked here, so
  // the SCHEDULE is extracted as a pure predicate and pinned as a value rule.
  // It is driven by the swap FINALIZATION index -- every IssueSwap return path,
  // including the suppressed-present and no-presenter exits -- because a title
  // stuck in persistent publication failure never advances swap_count_, and
  // that is exactly the run whose overlap evidence matters most.
  constexpr uint64_t kInterval = 512;

  // Index 0 is "no frame boundary has happened yet": never due. Reporting there
  // would print an empty snapshot as if it were a measured frame.
  CHECK_FALSE(ZeroCopyOverlapReportDue(0, kInterval));
  // The FIRST boundary always reports, so a run that ends after a handful of
  // frames still leaves a line.
  CHECK(ZeroCopyOverlapReportDue(1, kInterval));

  // Just before, at, and just after the interval. Index 513 is the second due
  // boundary because the first one was index 1; 512 is not, and neither is 514.
  CHECK_FALSE(ZeroCopyOverlapReportDue(kInterval - 1, kInterval));
  CHECK_FALSE(ZeroCopyOverlapReportDue(kInterval, kInterval));
  CHECK(ZeroCopyOverlapReportDue(kInterval + 1, kInterval));
  CHECK_FALSE(ZeroCopyOverlapReportDue(kInterval + 2, kInterval));

  // Later intervals keep the same phase rather than drifting.
  CHECK(ZeroCopyOverlapReportDue(2 * kInterval + 1, kInterval));
  CHECK(ZeroCopyOverlapReportDue(10 * kInterval + 1, kInterval));
  CHECK_FALSE(ZeroCopyOverlapReportDue(10 * kInterval, kInterval));

  // Exactly one due index per interval-sized window, and it is the first one:
  // the cadence is bounded, so a long run cannot turn the evidence into a log.
  uint32_t due = 0;
  for (uint64_t i = 1; i <= 4 * kInterval; ++i) {
    if (ZeroCopyOverlapReportDue(i, kInterval)) {
      ++due;
      CHECK(((i - 1) % kInterval) == 0u);
    }
  }
  CHECK(due == 4u);

  // An interval of 0 is a division by zero, and is treated as NEVER rather
  // than as always: a misconfigured cadence must not turn a bounded report
  // into a per-frame log on the GPU thread.
  for (uint64_t i = 0; i < 8; ++i) {
    INFO("finalization index " << i);
    CHECK_FALSE(ZeroCopyOverlapReportDue(i, 0));
  }
  // An interval of 1 is every boundary -- degenerate but well-defined, and
  // still not true at index 0.
  CHECK_FALSE(ZeroCopyOverlapReportDue(0, 1));
  for (uint64_t i = 1; i < 8; ++i) {
    CHECK(ZeroCopyOverlapReportDue(i, 1));
  }

  // Usable at compile time, like every other rule in this header.
  static_assert(ZeroCopyOverlapReportDue(1, 512));
  static_assert(!ZeroCopyOverlapReportDue(0, 512));
  static_assert(!ZeroCopyOverlapReportDue(512, 512));
  static_assert(ZeroCopyOverlapReportDue(513, 512));
  static_assert(!ZeroCopyOverlapReportDue(1, 0));
}

TEST_CASE("Zero-copy overlap silence: an inactive snapshot that counted "
          "something is NOT silent",
          "[l0][metal][zerocopy][overlap]") {
  // Both halves of the rule are required. A run that advanced a counter with no
  // observer installed is a defect in the model, and reporting it as a clean
  // control arm would hide exactly what the two-source split exists to catch.
  ZeroCopyOverlapSnapshot empty;
  CHECK(ZeroCopyOverlapSnapshotIsSilent(empty));

  ZeroCopyOverlapSnapshot active_and_empty;
  active_and_empty.active = true;
  CHECK_FALSE(ZeroCopyOverlapSnapshotIsSilent(active_and_empty));

  // Every counter, individually: none of them may be ignorable.
  auto not_silent = [](ZeroCopyOverlapSnapshot s) {
    return !ZeroCopyOverlapSnapshotIsSilent(s);
  };
  ZeroCopyOverlapSnapshot s;
  s = empty; s.stamp_calls = 1;                     CHECK(not_silent(s));
  s = empty; s.stamped_pages = 1;                   CHECK(not_silent(s));
  s = empty; s.zero_submission_stamp_pages = 1;     CHECK(not_silent(s));
  s = empty; s.cpu_write_callbacks = 1;             CHECK(not_silent(s));
  s = empty; s.cpu_write_pages = 1;                 CHECK(not_silent(s));
  s = empty; s.clamped_cpu_writes = 1;              CHECK(not_silent(s));
  s = empty; s.idle_pages = 1;                      CHECK(not_silent(s));
  s = empty; s.completed_reader_pages = 1;          CHECK(not_silent(s));
  s = empty; s.in_flight_cpu_write_overlap = 1;     CHECK(not_silent(s));
  s = empty; s.sample_count = 1;                    CHECK(not_silent(s));
}

TEST_CASE("Zero-copy overlap observer: a write to an unclaimed page is idle",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  CHECK(observer.active());

  // No consumer ever enrolled this page. Even with a completed mark of zero --
  // i.e. nothing has finished either -- this is not an overlap.
  observer.ObserveCpuWrite(4 * kPageBytes, kPageBytes, 0);
  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK(snapshot.cpu_write_callbacks == 1u);
  CHECK(snapshot.cpu_write_pages == 1u);
  CHECK(snapshot.idle_pages == 1u);
  CHECK(snapshot.completed_reader_pages == 0u);
  CHECK(snapshot.in_flight_cpu_write_overlap == 0u);
  CHECK(snapshot.sample_count == 0u);
  CHECK_FALSE(ZeroCopyOverlapSnapshotIsSilent(snapshot));
}

TEST_CASE("Zero-copy overlap observer: a write to a page whose reader "
          "completed is not an overlap",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  observer.StampConsumer(4 * kPageBytes, kPageBytes, /*submission=*/5);
  // Completed has reached the enrolling submission exactly: its reads are done.
  observer.ObserveCpuWrite(4 * kPageBytes, kPageBytes, /*completed=*/5);
  // And past it.
  observer.ObserveCpuWrite(4 * kPageBytes, kPageBytes, /*completed=*/6);

  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK(snapshot.stamp_calls == 1u);
  CHECK(snapshot.stamped_pages == 1u);
  CHECK(snapshot.zero_submission_stamp_pages == 0u);
  CHECK(snapshot.cpu_write_callbacks == 2u);
  CHECK(snapshot.cpu_write_pages == 2u);
  CHECK(snapshot.completed_reader_pages == 2u);
  CHECK(snapshot.idle_pages == 0u);
  CHECK(snapshot.in_flight_cpu_write_overlap == 0u);
  CHECK(snapshot.sample_count == 0u);
}

TEST_CASE("Zero-copy overlap observer: a write to a page with an unfinished "
          "reader is the stop signal",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  observer.StampConsumer(4 * kPageBytes, kPageBytes, /*submission=*/9);
  observer.ObserveCpuWrite(4 * kPageBytes, kPageBytes, /*completed=*/8);

  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK(snapshot.in_flight_cpu_write_overlap == 1u);
  CHECK(snapshot.idle_pages == 0u);
  CHECK(snapshot.completed_reader_pages == 0u);
  REQUIRE(snapshot.sample_count == 1u);
  CHECK(snapshot.samples[0].page_index == 4u);
  CHECK(snapshot.samples[0].consumer_submission == 9u);
  CHECK(snapshot.samples[0].completed_submission == 8u);
}

TEST_CASE("Zero-copy overlap observer: the newest reader wins the page",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  // Three consumers claim the same page out of order. An older enrollment must
  // not be able to lower the page's claim, or a CPU write would be compared
  // against a submission that already finished while a newer one is still
  // reading the same bytes.
  observer.StampConsumer(4 * kPageBytes, kPageBytes, /*submission=*/9);
  observer.StampConsumer(4 * kPageBytes, kPageBytes, /*submission=*/3);
  observer.StampConsumer(4 * kPageBytes, kPageBytes, /*submission=*/7);

  observer.ObserveCpuWrite(4 * kPageBytes, kPageBytes, /*completed=*/8);
  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK(snapshot.stamp_calls == 3u);
  CHECK(snapshot.stamped_pages == 3u);
  CHECK(snapshot.in_flight_cpu_write_overlap == 1u);
  REQUIRE(snapshot.sample_count == 1u);
  CHECK(snapshot.samples[0].consumer_submission == 9u);
}

TEST_CASE("Zero-copy overlap observer: an idle-sentinel enrollment claims "
          "nothing and says so",
          "[l0][metal][zerocopy][overlap]") {
  // This is the guard against the wrong execution source. The S7-governed
  // consumed-range provider returns 0 on every default launch, and 0 is the
  // idle sentinel -- so wiring the overlap stamps to it would make every page
  // read idle and the overlap counter permanently zero for a reason that has
  // nothing to do with safety. Counting those enrollments means the report can
  // say the run is not evidence, instead of looking clean.
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  observer.StampConsumer(4 * kPageBytes, kPageBytes, /*submission=*/0);
  observer.ObserveCpuWrite(4 * kPageBytes, kPageBytes, /*completed=*/0);

  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK(snapshot.stamp_calls == 1u);
  CHECK(snapshot.stamped_pages == 1u);
  CHECK(snapshot.zero_submission_stamp_pages == 1u);
  // The page is genuinely unclaimed, so the write is idle -- and the nonzero
  // sentinel count is the only thing that distinguishes this from a real
  // no-overlap run.
  CHECK(snapshot.idle_pages == 1u);
  CHECK(snapshot.in_flight_cpu_write_overlap == 0u);
}

TEST_CASE("Zero-copy overlap observer: one callback uses one completed mark "
          "for every page it touches",
          "[l0][metal][zerocopy][overlap]") {
  // The caller loads the high-water mark once and hands it down, so the
  // observer structurally cannot re-read it per page. Sampling per page would
  // let a submission complete mid-range and split one physical write into two
  // different verdicts -- a page reported finished only because the loop
  // reached it late.
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  observer.StampConsumer(8 * kPageBytes, kPageBytes, /*submission=*/5);
  observer.StampConsumer(9 * kPageBytes, kPageBytes, /*submission=*/6);
  observer.StampConsumer(10 * kPageBytes, kPageBytes, /*submission=*/7);

  // One callback across all three pages, against completed = 6.
  observer.ObserveCpuWrite(8 * kPageBytes, 3 * kPageBytes, /*completed=*/6);
  ZeroCopyOverlapSnapshot snapshot = observer.Snapshot();
  CHECK(snapshot.cpu_write_callbacks == 1u);
  CHECK(snapshot.cpu_write_pages == 3u);
  // 5 and 6 finished under this one mark; only 7 is still in flight.
  CHECK(snapshot.completed_reader_pages == 2u);
  CHECK(snapshot.in_flight_cpu_write_overlap == 1u);
  REQUIRE(snapshot.sample_count == 1u);
  CHECK(snapshot.samples[0].page_index == 10u);
  CHECK(snapshot.samples[0].completed_submission == 6u);
  // Every observed page is in exactly one bucket.
  CHECK(snapshot.idle_pages + snapshot.completed_reader_pages +
            snapshot.in_flight_cpu_write_overlap ==
        snapshot.cpu_write_pages);
}

TEST_CASE("Zero-copy overlap observer: boundary and clamped writes are "
          "observed exactly once each",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  // First page and last page of the domain, both claimed by an unfinished
  // submission. The guest chooses the fault range, not the backend, so the ends
  // of the domain are ordinary inputs.
  observer.StampConsumer(0, kPageBytes, /*submission=*/4);
  observer.StampConsumer(kDomainBytes - kPageBytes, kPageBytes,
                         /*submission=*/4);

  observer.ObserveCpuWrite(0, kPageBytes, /*completed=*/3);
  observer.ObserveCpuWrite(kDomainBytes - kPageBytes, kPageBytes,
                           /*completed=*/3);
  ZeroCopyOverlapSnapshot ends = observer.Snapshot();
  CHECK(ends.in_flight_cpu_write_overlap == 2u);
  CHECK(ends.clamped_cpu_writes == 0u);
  CHECK(ends.sample_count == 2u);

  // A write straddling the end: the in-domain part is observed, the rest is
  // clamped away, and the fixed table is never indexed past its capacity.
  observer.ObserveCpuWrite(kDomainBytes - kPageBytes, 4 * kPageBytes,
                           /*completed=*/3);
  ZeroCopyOverlapSnapshot straddle = observer.Snapshot();
  CHECK(straddle.cpu_write_callbacks == 3u);
  CHECK(straddle.cpu_write_pages == 3u);  // one more page, not four
  CHECK(straddle.clamped_cpu_writes == 1u);
  CHECK(straddle.in_flight_cpu_write_overlap == 3u);

  // Entirely out of domain: counted as a callback, clamped, and observed as
  // nothing at all.
  observer.ObserveCpuWrite(kDomainBytes, kPageBytes, /*completed=*/3);
  observer.ObserveCpuWrite(UINT64_MAX - 3, 8, /*completed=*/3);
  ZeroCopyOverlapSnapshot outside = observer.Snapshot();
  CHECK(outside.cpu_write_callbacks == 5u);
  CHECK(outside.cpu_write_pages == 3u);
  CHECK(outside.clamped_cpu_writes == 3u);
  CHECK(outside.in_flight_cpu_write_overlap == 3u);

  // A stamp outside the domain claims nothing either.
  observer.StampConsumer(kDomainBytes, kPageBytes, /*submission=*/99);
  CHECK(observer.Snapshot().stamp_calls == 2u);
}

TEST_CASE("Zero-copy overlap observer: samples are deduplicated and bounded",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));

  // The same page overlapping many times keeps ONE slot. A title that rewrites
  // one page every frame would otherwise fill every slot with one page's story
  // and hide every other page that overlapped.
  observer.StampConsumer(2 * kPageBytes, kPageBytes, /*submission=*/9);
  for (uint32_t i = 0; i < 32; ++i) {
    observer.ObserveCpuWrite(2 * kPageBytes, kPageBytes, /*completed=*/8);
  }
  ZeroCopyOverlapSnapshot repeated = observer.Snapshot();
  CHECK(repeated.in_flight_cpu_write_overlap == 32u);  // the COUNT still moves
  CHECK(repeated.sample_count == 1u);                  // the samples do not
  CHECK(repeated.samples[0].page_index == 2u);

  // Distinct pages fill the fixed slots and then stop. The counter keeps
  // counting; retention does not grow with the number of observations.
  for (uint32_t page = 16; page < 16 + 32; ++page) {
    observer.StampConsumer(uint64_t(page) << kPageLog2, kPageBytes,
                           /*submission=*/9);
    observer.ObserveCpuWrite(uint64_t(page) << kPageLog2, kPageBytes,
                             /*completed=*/8);
  }
  ZeroCopyOverlapSnapshot many = observer.Snapshot();
  CHECK(many.in_flight_cpu_write_overlap == 64u);
  CHECK(many.sample_count == kZeroCopyOverlapSampleSlots);
  // Every retained slot describes a DIFFERENT page.
  for (uint32_t i = 0; i < many.sample_count; ++i) {
    for (uint32_t j = i + 1; j < many.sample_count; ++j) {
      CHECK(many.samples[i].page_index != many.samples[j].page_index);
    }
  }
}

TEST_CASE("Zero-copy overlap observer: shutdown clears the evidence and "
          "re-initialization starts clean",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  observer.StampConsumer(kPageBytes, kPageBytes, /*submission=*/9);
  observer.ObserveCpuWrite(kPageBytes, kPageBytes, /*completed=*/8);
  REQUIRE(observer.Snapshot().in_flight_cpu_write_overlap == 1u);

  // Shutdown is what the shared memory runs AFTER unregistering its physical
  // invalidation callback, so the evidence must be read before it -- which is
  // why the command processor's final report happens before shared_memory_ is
  // destroyed.
  observer.Shutdown();
  CHECK_FALSE(observer.active());
  CHECK(ZeroCopyOverlapSnapshotIsSilent(observer.Snapshot()));

  // A second Initialize starts from zero, including the stamp table: a stale
  // stamp would make the next run's first CPU write look like an overlap.
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  observer.ObserveCpuWrite(kPageBytes, kPageBytes, /*completed=*/0);
  ZeroCopyOverlapSnapshot reinitialized = observer.Snapshot();
  CHECK(reinitialized.idle_pages == 1u);
  CHECK(reinitialized.in_flight_cpu_write_overlap == 0u);
  CHECK(reinitialized.sample_count == 0u);
}

TEST_CASE("Zero-copy overlap observer: the fixed table is sized by the domain "
          "and rejects impossible geometry",
          "[l0][metal][zerocopy][overlap]") {
  ZeroCopyOverlapObserver observer;
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  CHECK(observer.Snapshot().page_count == kPages);
  CHECK(observer.Snapshot().page_size_log2 == kPageLog2);

  // The real geometry: 512 MiB of guest physical domain over a 16 KiB Apple
  // Silicon host page, and over a 4 KiB page. Bounded either way -- the table
  // is the domain divided by the page, never per-request retention.
  REQUIRE(observer.Initialize(512u * 1024 * 1024, 14));
  CHECK(observer.Snapshot().page_count == 32768u);
  REQUIRE(observer.Initialize(512u * 1024 * 1024, 12));
  CHECK(observer.Snapshot().page_count == 131072u);

  // Geometry that cannot describe a table leaves the observer INACTIVE rather
  // than partially armed. Every refusal is checked for GEOMETRY as well as for
  // activity: `active()` reads the table pointer, so a run that committed a
  // page count and then failed to get the table would still report inactive
  // while the snapshot advertised a table size nothing backs. Reported geometry
  // and the table it describes are the same fact or the report is fiction.
  CHECK_FALSE(observer.Initialize(0, kPageLog2));
  CHECK_FALSE(observer.active());
  CHECK(observer.Snapshot().page_count == 0u);
  CHECK(observer.Snapshot().page_size_log2 == 0u);
  CHECK_FALSE(observer.Initialize(kDomainBytes, 32));
  CHECK_FALSE(observer.active());
  CHECK(observer.Snapshot().page_count == 0u);
  CHECK(observer.Snapshot().page_size_log2 == 0u);
  CHECK_FALSE(observer.Initialize(kPageBytes - 1, kPageLog2));
  CHECK_FALSE(observer.active());
  CHECK(observer.Snapshot().page_count == 0u);
  CHECK(observer.Snapshot().page_size_log2 == 0u);

  // The same invariant from the other direction: a refusal that FOLLOWS a
  // successful arming must not leave the previous run's geometry standing in
  // for the table it just released.
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  CHECK(observer.Snapshot().page_count == kPages);
  CHECK_FALSE(observer.Initialize(0, kPageLog2));
  CHECK_FALSE(observer.active());
  CHECK(observer.Snapshot().page_count == 0u);
  CHECK(observer.Snapshot().page_size_log2 == 0u);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(observer.Snapshot()));
}

TEST_CASE("Zero-copy overlap observer: arming refuses by RETURN VALUE, and the "
          "table allocation is the untestable half",
          "[l0][metal][zerocopy][overlap]") {
  // Initialize's contract is that an observer which cannot be armed says so by
  // returning false, so MetalSharedMemory can release the alias candidate and
  // install the ordinary mirror. That contract has two arming failures behind
  // it and this suite can only run one of them:
  //
  //   * REACHABLE HERE -- unusable geometry and the diagnostic injection, both
  //     covered above and both plain early returns.
  //   * NOT REACHABLE HERE -- the host failing to allocate the stamp table.
  //     Producing a real allocation failure for a 512 MiB domain on demand
  //     would need a test-only allocator seam in the production path, which
  //     costs more than it proves. What is pinned instead is the SHAPE the
  //     allocation-failure path shares with the reachable ones: the observer is
  //     inactive, carries no geometry, and its production entries are inert --
  //     the caller's fallback keys off the return value and nothing else.
  //
  // The proof limit is therefore explicit: the allocation is non-throwing and
  // null-checked BY CONSTRUCTION (see Initialize), and reviewed, not executed.
  // What this case forbids is the state a throwing or unchecked allocation
  // would leave behind -- committed geometry with no table under it.
  ZeroCopyOverlapObserver observer;

  // Both reachable refusals are indistinguishable to the caller: same return
  // value, same inert observer, same empty snapshot.
  for (bool forced : {false, true}) {
    ZeroCopyOverlapSnapshot snapshot;
    if (forced) {
      CHECK_FALSE(observer.Initialize(kDomainBytes, kPageLog2, true));
    } else {
      CHECK_FALSE(observer.Initialize(0, kPageLog2));
    }
    CHECK_FALSE(observer.active());
    // Inert: the production entries do nothing whatever a caller asks of them,
    // so a rejected observer can never contribute a page to the evidence.
    observer.StampConsumer(0, kDomainBytes, 9);
    observer.ObserveCpuWrite(0, kDomainBytes, 1);
    snapshot = observer.Snapshot();
    CHECK_FALSE(snapshot.active);
    CHECK(snapshot.page_count == 0u);
    CHECK(snapshot.page_size_log2 == 0u);
    CHECK(ZeroCopyOverlapSnapshotIsSilent(snapshot));
  }

  // And a successful arming is the only thing that produces the opposite: a
  // table, its geometry, and a live observation. Without this the case above
  // would pass on an observer that can never arm at all.
  REQUIRE(observer.Initialize(kDomainBytes, kPageLog2));
  CHECK(observer.active());
  observer.StampConsumer(0, kPageBytes, 9);
  observer.ObserveCpuWrite(0, kPageBytes, 1);
  ZeroCopyOverlapSnapshot armed = observer.Snapshot();
  CHECK(armed.active);
  CHECK(armed.page_count == kPages);
  CHECK(armed.page_size_log2 == kPageLog2);
  CHECK(armed.in_flight_cpu_write_overlap == 1u);
  CHECK_FALSE(ZeroCopyOverlapSnapshotIsSilent(armed));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
