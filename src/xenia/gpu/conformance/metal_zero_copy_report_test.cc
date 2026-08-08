/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free) conformance for the Metal zero-copy RUNTIME COST
// report: pre-gate-metal-zero-copy-shared-memory group 6 (tasks 6.3/6.4).
//
// The reporter itself is a log line and cannot be asserted here. What CAN be
// pinned -- and what the evidence rests on -- is the vocabulary underneath it:
//
//   * the exact-"1" gate, which is what makes an unset, "0", or malformed run
//     silent, and which is deliberately the SAME rule the alias gate parses
//     with (two spellings of "enabled" is how one arm of an A/B ends up
//     carrying diagnostics the other does not);
//   * the counter set, enumerated exhaustively from its own kCount so a new
//     counter cannot be added without being named;
//   * the meter's arithmetic: inert while disabled, saturating rather than
//     wrapping, and per-counter independent, so one production site can never
//     move another's number;
//   * the counting DISCIPLINE the production sites follow -- retirements
//     counted as events and bytes as bytes, so an aliased backing that moves
//     zero bytes still shows its retirements and a retained record shows
//     neither.
//
// Completeness is pinned at COMPILE time: the sweep runs over a std::array
// sized by the enum's own kCount and static_asserted to hold exactly its values
// 0..kCount-1, so adding an enumerator without listing it here fails the build
// rather than silently shrinking the sweep.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_zero_copy_report.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

template <typename T, size_t N>
constexpr bool IsExhaustive(const std::array<T, N>& values) {
  for (size_t i = 0; i < N; ++i) {
    if (values[i] != static_cast<T>(i)) {
      return false;
    }
  }
  return true;
}

constexpr std::array<ZeroCopyCostCounter, kZeroCopyCostCounterCount>
    kAllCounters = {
        ZeroCopyCostCounter::kSubmissionCommits,
        ZeroCopyCostCounter::kSyncCommitWaits,
        ZeroCopyCostCounter::kWaitForSubmissionCalls,
        ZeroCopyCostCounter::kWaitForSubmissionBlocked,
        ZeroCopyCostCounter::kWaitForSubmissionTimeouts,
        ZeroCopyCostCounter::kMemexportDestinationPreparations,
        ZeroCopyCostCounter::kMemexportNoOlderFastPaths,
        ZeroCopyCostCounter::kMemexportOrderingBarriers,
        ZeroCopyCostCounter::kMemexportSelectedPredecessors,
        ZeroCopyCostCounter::kMemexportOrderingRetained,
        ZeroCopyCostCounter::kMemexportPreparationFailures,
        ZeroCopyCostCounter::kMemexportRetirements,
        ZeroCopyCostCounter::kMemexportConvergenceBytesCopied,
};
static_assert(IsExhaustive(kAllCounters),
              "kAllCounters must list every ZeroCopyCostCounter value exactly "
              "once, in declaration order");

}  // namespace

TEST_CASE("Zero-copy cost gate: only the exact value \"1\" enables the report",
          "[l0][metal][zerocopy][cost]") {
  // Off is the default and every near miss stays off. A gate that accepted
  // "true", "01", " 1" or any nonempty value would make "the cost line is not
  // being emitted" unfalsifiable from the environment alone, which is what the
  // control arms of this pre-gate rest on.
  static constexpr const char* kDisabled[] = {
      "",   "0",   "00",  "01",  " 1",  "1 ",  "1\n", "1\t", "+1",
      "-1", "1.0", "10",  "11",  "2",   "on",  "ON",  "yes", "true",
      "TRUE", "y", "enable", "enabled", "\0011"};
  for (const char* value : kDisabled) {
    INFO("value=\"" << value << "\"");
    CHECK_FALSE(IsZeroCopyReportEnabled(value));
  }
  CHECK_FALSE(IsZeroCopyReportEnabled(nullptr));  // unset
  CHECK(IsZeroCopyReportEnabled("1"));

  // ONE rule, not two: the cost gate and the alias gate must agree on every
  // input, so a run cannot be "enabled" for one and not the other.
  for (const char* value : kDisabled) {
    INFO("value=\"" << value << "\"");
    CHECK(IsZeroCopyReportEnabled(value) == IsZeroCopyGateEnabled(value));
  }
  CHECK(IsZeroCopyReportEnabled(nullptr) == IsZeroCopyGateEnabled(nullptr));
  CHECK(IsZeroCopyReportEnabled("1") == IsZeroCopyGateEnabled("1"));

  // Usable in a constant expression, so the production gate costs nothing at
  // runtime beyond the single getenv it parses.
  static_assert(IsZeroCopyReportEnabled("1"));
  static_assert(!IsZeroCopyReportEnabled("11"));
  static_assert(!IsZeroCopyReportEnabled(nullptr));
  static_assert(!IsZeroCopyReportEnabled(""));

  // The environment variable is its own name, distinct from every other
  // zero-copy gate: sharing one would make the diagnostic and the backing
  // decision inseparable.
  CHECK(std::string(kZeroCopyReportEnvVar) == "XE_METAL_ZERO_COPY_REPORT");
  CHECK(std::string(kZeroCopyReportEnvVar) !=
        std::string(kZeroCopyGateEnvVar));
  CHECK(std::string(kZeroCopyReportEnvVar) !=
        std::string(kZeroCopyForceRejectEnvVar));
  CHECK(std::string(kZeroCopyReportEnvVar) !=
        std::string(kZeroCopyForceObserverRejectEnvVar));
}

TEST_CASE("Zero-copy cost vocabulary: complete and distinctly named",
          "[l0][metal][zerocopy][cost]") {
  static_assert(kAllCounters.size() == kZeroCopyCostCounterCount);

  // kCount is a bound, never a counter.
  static_assert(!IsZeroCopyCostCounterValue(ZeroCopyCostCounter::kCount));
  static_assert(!IsZeroCopyCostCounterValue(
      static_cast<ZeroCopyCostCounter>(kZeroCopyCostCounterCount + 7)));
  for (ZeroCopyCostCounter counter : kAllCounters) {
    CHECK(IsZeroCopyCostCounterValue(counter));
  }

  std::vector<std::string> names;
  for (ZeroCopyCostCounter counter : kAllCounters) {
    std::string name = ZeroCopyCostCounterName(counter);
    INFO("counter=" << name);
    CHECK(!name.empty());
    CHECK(name != "unknown");
    names.push_back(name);
  }
  // Distinct: two counters sharing a label would make the evidence line
  // ambiguous exactly where two production operations must stay apart (a
  // barrier vs a fast path, a wait vs a wait that blocked).
  for (size_t i = 0; i < names.size(); ++i) {
    for (size_t j = i + 1; j < names.size(); ++j) {
      INFO(names[i] << " vs " << names[j]);
      CHECK(names[i] != names[j]);
    }
  }

  // The sentinel and anything past it reach the out-of-range bucket rather
  // than aliasing a real counter's name.
  for (uint32_t past : {uint32_t(0), uint32_t(1), uint32_t(7)}) {
    INFO("kCount + " << past);
    CHECK(std::string(ZeroCopyCostCounterName(static_cast<ZeroCopyCostCounter>(
              kZeroCopyCostCounterCount + past))) == "unknown");
  }
}

TEST_CASE("Zero-copy cost meter: disabled is inert",
          "[l0][metal][zerocopy][cost]") {
  // The default (unset / "0" / malformed) arm. Silence is STRUCTURAL: the meter
  // refuses to advance, so an off run cannot produce a line by a caller
  // forgetting to check a flag.
  ZeroCopyCostMeter meter;
  CHECK_FALSE(meter.enabled());
  for (ZeroCopyCostCounter counter : kAllCounters) {
    meter.Count(counter);
    meter.Add(counter, 1000);
  }
  ZeroCopyCostSnapshot snapshot = meter.Snapshot();
  CHECK_FALSE(snapshot.enabled);
  for (ZeroCopyCostCounter counter : kAllCounters) {
    INFO("counter=" << ZeroCopyCostCounterName(counter));
    CHECK(snapshot.Get(counter) == 0);
  }
  CHECK(ZeroCopyCostSnapshotIsSilent(snapshot));

  // Enabling later starts from zero, and disabling again freezes the values
  // rather than clearing them -- the run's evidence is not erased by a toggle.
  meter.SetEnabled(true);
  CHECK(meter.enabled());
  meter.Count(ZeroCopyCostCounter::kSubmissionCommits);
  meter.SetEnabled(false);
  meter.Count(ZeroCopyCostCounter::kSubmissionCommits);
  snapshot = meter.Snapshot();
  CHECK(snapshot.Get(ZeroCopyCostCounter::kSubmissionCommits) == 1);
  // Counted-but-disabled is NOT silent: the reporter has to be able to say so
  // instead of printing nothing and calling it a clean control arm.
  CHECK_FALSE(ZeroCopyCostSnapshotIsSilent(snapshot));
}

TEST_CASE("Zero-copy cost meter: an armed run is never silent",
          "[l0][metal][zerocopy][cost]") {
  // An enabled run that did nothing still has to emit its line: on a control
  // arm, "the backend ran and moved nothing" and "nothing was measured" are
  // different claims, and only the first is evidence.
  ZeroCopyCostMeter meter;
  meter.SetEnabled(true);
  ZeroCopyCostSnapshot snapshot = meter.Snapshot();
  CHECK(snapshot.enabled);
  CHECK_FALSE(ZeroCopyCostSnapshotIsSilent(snapshot));
  for (ZeroCopyCostCounter counter : kAllCounters) {
    CHECK(snapshot.Get(counter) == 0);
  }
}

TEST_CASE("Zero-copy cost meter: counters are independent",
          "[l0][metal][zerocopy][cost]") {
  // One production site must never be able to move another site's number.
  ZeroCopyCostMeter meter;
  meter.SetEnabled(true);
  for (size_t i = 0; i < kAllCounters.size(); ++i) {
    meter.Add(kAllCounters[i], uint64_t(i) + 1);
  }
  ZeroCopyCostSnapshot snapshot = meter.Snapshot();
  for (size_t i = 0; i < kAllCounters.size(); ++i) {
    INFO("counter=" << ZeroCopyCostCounterName(kAllCounters[i]));
    CHECK(snapshot.Get(kAllCounters[i]) == uint64_t(i) + 1);
  }

  // Count() is exactly +1 on its own counter.
  meter.Count(ZeroCopyCostCounter::kMemexportOrderingBarriers);
  ZeroCopyCostSnapshot after = meter.Snapshot();
  for (size_t i = 0; i < kAllCounters.size(); ++i) {
    uint64_t expected = uint64_t(i) + 1;
    if (kAllCounters[i] == ZeroCopyCostCounter::kMemexportOrderingBarriers) {
      ++expected;
    }
    INFO("counter=" << ZeroCopyCostCounterName(kAllCounters[i]));
    CHECK(after.Get(kAllCounters[i]) == expected);
  }

  // Adding nothing is a no-op, and an undefined counter is ignored rather than
  // indexing past the array or landing on a real counter.
  meter.Add(ZeroCopyCostCounter::kSubmissionCommits, 0);
  meter.Add(ZeroCopyCostCounter::kCount, 999);
  meter.Add(static_cast<ZeroCopyCostCounter>(kZeroCopyCostCounterCount + 3),
            999);
  ZeroCopyCostSnapshot unchanged = meter.Snapshot();
  for (ZeroCopyCostCounter counter : kAllCounters) {
    INFO("counter=" << ZeroCopyCostCounterName(counter));
    CHECK(unchanged.Get(counter) == after.Get(counter));
  }
  // Reading an undefined counter yields zero, never whatever followed the
  // array.
  CHECK(unchanged.Get(ZeroCopyCostCounter::kCount) == 0);
  CHECK(unchanged.Get(static_cast<ZeroCopyCostCounter>(
            kZeroCopyCostCounterCount + 3)) == 0);
}

TEST_CASE("Zero-copy cost meter: totals saturate rather than wrap",
          "[l0][metal][zerocopy][cost]") {
  // A session-long byte total that wrapped would read as a SMALL number -- i.e.
  // as "almost no copying happened", which is precisely the claim this evidence
  // exists to test. Saturation makes an overflowed counter obviously pinned
  // instead of quietly flattering the alias arm.
  ZeroCopyCostMeter meter;
  meter.SetEnabled(true);
  const ZeroCopyCostCounter kBytes =
      ZeroCopyCostCounter::kMemexportConvergenceBytesCopied;
  meter.Add(kBytes, UINT64_MAX - 4);
  CHECK(meter.Snapshot().Get(kBytes) == UINT64_MAX - 4);
  meter.Add(kBytes, 3);
  CHECK(meter.Snapshot().Get(kBytes) == UINT64_MAX - 1);
  meter.Add(kBytes, 10);
  CHECK(meter.Snapshot().Get(kBytes) == UINT64_MAX);
  meter.Add(kBytes, UINT64_MAX);
  CHECK(meter.Snapshot().Get(kBytes) == UINT64_MAX);
  meter.Count(kBytes);
  CHECK(meter.Snapshot().Get(kBytes) == UINT64_MAX);
  // Saturating one counter leaves the others alone.
  CHECK(meter.Snapshot().Get(ZeroCopyCostCounter::kMemexportRetirements) == 0);

  // The shared scalar rule the meter is built on, checked at compile time.
  static_assert(ZeroCopySaturatingAdd(UINT64_MAX, 1) == UINT64_MAX);
  static_assert(ZeroCopySaturatingAdd(UINT64_MAX - 1, 1) == UINT64_MAX);
  static_assert(ZeroCopySaturatingAdd(0, 0) == 0);
}

TEST_CASE("Zero-copy cost meter: reset clears values, not the request",
          "[l0][metal][zerocopy][cost]") {
  // A session boundary resets what was measured. It must not reset WHETHER
  // measuring was requested: the environment was parsed once, at startup, and a
  // title relaunch cannot be allowed to silently turn the evidence off.
  ZeroCopyCostMeter meter;
  meter.SetEnabled(true);
  for (ZeroCopyCostCounter counter : kAllCounters) {
    meter.Add(counter, 7);
  }
  meter.ResetCounters();
  ZeroCopyCostSnapshot snapshot = meter.Snapshot();
  CHECK(snapshot.enabled);
  for (ZeroCopyCostCounter counter : kAllCounters) {
    INFO("counter=" << ZeroCopyCostCounterName(counter));
    CHECK(snapshot.Get(counter) == 0);
  }
  CHECK_FALSE(ZeroCopyCostSnapshotIsSilent(snapshot));
}

TEST_CASE(
    "Zero-copy cost discipline: retirements and bytes are counted separately",
    "[l0][metal][zerocopy][cost]") {
  // The rule the production convergence site follows, stated as a value
  // transform so it is pinned independently of the site: a retirement is an
  // EVENT, the bytes are BYTES, and the two are counted at the same place
  // without either being a function of the other.
  //
  // This is what lets the mirror and alias arms be compared honestly: both
  // retire the same records, and only one of them copies.
  struct Record {
    bool retires;
    uint64_t bytes_copied;
  };
  auto drain = [](ZeroCopyCostMeter& meter, const std::vector<Record>& records) {
    for (const Record& record : records) {
      if (!record.retires) {
        // A retained record -- a short drain, an unproven boundary, a failed
        // copy -- never reaches the counting site at all.
        continue;
      }
      meter.Count(ZeroCopyCostCounter::kMemexportRetirements);
      meter.Add(ZeroCopyCostCounter::kMemexportConvergenceBytesCopied,
                record.bytes_copied);
    }
  };

  // Mirror backing: three records retire, each physically copying its length.
  ZeroCopyCostMeter mirror;
  mirror.SetEnabled(true);
  drain(mirror, {{true, 2048}, {true, 2048}, {true, 4096}});
  CHECK(mirror.Snapshot().Get(ZeroCopyCostCounter::kMemexportRetirements) == 3);
  CHECK(mirror.Snapshot().Get(
            ZeroCopyCostCounter::kMemexportConvergenceBytesCopied) == 8192);

  // Alias backing: the SAME three records retire, having moved zero bytes. A
  // zero-byte convergence still retires -- that is the whole point of the
  // alias -- so a byte-derived retirement count would report the alias arm as
  // having published nothing.
  ZeroCopyCostMeter alias;
  alias.SetEnabled(true);
  drain(alias, {{true, 0}, {true, 0}, {true, 0}});
  CHECK(alias.Snapshot().Get(ZeroCopyCostCounter::kMemexportRetirements) == 3);
  CHECK(alias.Snapshot().Get(
            ZeroCopyCostCounter::kMemexportConvergenceBytesCopied) == 0);

  // A retained record is neither a retirement nor bytes, on either backing.
  ZeroCopyCostMeter short_drain;
  short_drain.SetEnabled(true);
  drain(short_drain, {{true, 4096}, {false, 4096}, {false, 0}});
  CHECK(short_drain.Snapshot().Get(
            ZeroCopyCostCounter::kMemexportRetirements) == 1);
  CHECK(short_drain.Snapshot().Get(
            ZeroCopyCostCounter::kMemexportConvergenceBytesCopied) == 4096);
}

TEST_CASE("Zero-copy cost discipline: barriers are not consultations",
          "[l0][metal][zerocopy][cost]") {
  // The ordering counters model the three outcomes the destination-preparation
  // seam can reach, and only the outcome that RAN a drain counts as a barrier.
  // A preparation whose snapshot selected nothing consulted the snapshot and
  // stopped; counting it as a barrier would make the alias arm's ordering cost
  // look identical to the mirror's no matter what actually happened.
  ZeroCopyCostMeter meter;
  meter.SetEnabled(true);

  // Two preparations: one with no older producer, one that drained two.
  meter.Count(ZeroCopyCostCounter::kMemexportDestinationPreparations);
  meter.Count(ZeroCopyCostCounter::kMemexportNoOlderFastPaths);

  meter.Count(ZeroCopyCostCounter::kMemexportDestinationPreparations);
  meter.Count(ZeroCopyCostCounter::kMemexportOrderingBarriers);
  meter.Add(ZeroCopyCostCounter::kMemexportSelectedPredecessors, 2);

  // A third whose drain came up short: a barrier DID run, its predecessors were
  // selected, and the destination stayed unprepared.
  meter.Count(ZeroCopyCostCounter::kMemexportDestinationPreparations);
  meter.Count(ZeroCopyCostCounter::kMemexportOrderingBarriers);
  meter.Add(ZeroCopyCostCounter::kMemexportSelectedPredecessors, 1);
  meter.Count(ZeroCopyCostCounter::kMemexportOrderingRetained);
  meter.Count(ZeroCopyCostCounter::kMemexportPreparationFailures);

  // A fourth that never reached an ordering decision at all: the draw produced
  // no normalizable destination range, so the seam returns kNotEvaluated
  // without consulting the snapshot. It is a preparation and a failure, and it
  // is NEITHER a fast path nor a barrier -- which is exactly why the reading
  // rule below is an inequality.
  meter.Count(ZeroCopyCostCounter::kMemexportDestinationPreparations);
  meter.Count(ZeroCopyCostCounter::kMemexportPreparationFailures);

  ZeroCopyCostSnapshot snapshot = meter.Snapshot();
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportDestinationPreparations) ==
        4);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportNoOlderFastPaths) == 1);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportOrderingBarriers) == 2);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportSelectedPredecessors) == 3);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportOrderingRetained) == 1);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportPreparationFailures) == 2);
  // The reading rule for the line: fast paths and barriers are disjoint subsets
  // of the preparations, and their sum is a LOWER bound, not an identity. The
  // remainder is the unevaluated case above -- a preparation that never got as
  // far as an ordering verdict. Stating this as an identity would make a run
  // containing such draws look like a run with miscounted barriers.
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportNoOlderFastPaths) +
            snapshot.Get(ZeroCopyCostCounter::kMemexportOrderingBarriers) <=
        snapshot.Get(ZeroCopyCostCounter::kMemexportDestinationPreparations));
  // Retained outcomes are a subset of the barriers that ran (a barrier is what
  // can come up short) and of the preparation failures (a retained destination
  // is never prepared).
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportOrderingRetained) <=
        snapshot.Get(ZeroCopyCostCounter::kMemexportOrderingBarriers));
  CHECK(snapshot.Get(ZeroCopyCostCounter::kMemexportOrderingRetained) <=
        snapshot.Get(ZeroCopyCostCounter::kMemexportPreparationFailures));
}

TEST_CASE("Zero-copy cost discipline: waits separate calls, blocks, timeouts",
          "[l0][metal][zerocopy][cost]") {
  // Three facts about the same seam. A call satisfied by the already-completed
  // fast path costs nothing; a call that blocked cost a stall; a call that
  // timed out means the GPU wedged and the run's other numbers describe a
  // broken session rather than a fast one.
  ZeroCopyCostMeter meter;
  meter.SetEnabled(true);
  // Four calls: two satisfied immediately, two blocked, one of which expired.
  for (int i = 0; i < 4; ++i) {
    meter.Count(ZeroCopyCostCounter::kWaitForSubmissionCalls);
  }
  meter.Add(ZeroCopyCostCounter::kWaitForSubmissionBlocked, 2);
  meter.Count(ZeroCopyCostCounter::kWaitForSubmissionTimeouts);

  ZeroCopyCostSnapshot snapshot = meter.Snapshot();
  CHECK(snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionCalls) == 4);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionBlocked) == 2);
  CHECK(snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionTimeouts) == 1);
  // Blocked calls are a subset of calls, and timeouts a subset of blocked
  // calls. The production sites establish this by construction (each is counted
  // inside the previous one's branch); asserting it here documents which
  // containment the evidence line may be read with.
  //
  // What this L0 case CANNOT establish is that a counted block really entered
  // the condition-variable wait: the meter sees a number, not the seam. That
  // half is a locking property of MetalCommandProcessor -- the completion
  // high-water advances under the same mutex the waiter re-checks and waits
  // under -- and it is stated where it is enforced (PublishCompletedSubmission
  // / WaitForSubmission), because a value test that restated it here would only
  // be asserting its own arithmetic.
  CHECK(snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionBlocked) <=
        snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionCalls));
  CHECK(snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionTimeouts) <=
        snapshot.Get(ZeroCopyCostCounter::kWaitForSubmissionBlocked));

  // Commits and their synchronous waits are likewise separate: the sync-wait
  // count is bounded by the commit count, never derived from the mode flag.
  meter.Add(ZeroCopyCostCounter::kSubmissionCommits, 9);
  meter.Add(ZeroCopyCostCounter::kSyncCommitWaits, 9);
  snapshot = meter.Snapshot();
  CHECK(snapshot.Get(ZeroCopyCostCounter::kSyncCommitWaits) <=
        snapshot.Get(ZeroCopyCostCounter::kSubmissionCommits));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
