/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Mutation gate for the memexport census verifier
// (pre-gate-metal-memexport-gpu-consumers task 2.4). Device-free L0.
//
// A census is only worth believing if a wrong census is REJECTED, and rejected
// for a NAMED reason — "the seam is missing" and "the seam lied about its
// class" need different repairs, so one boolean cannot serve. Each case below
// applies exactly one mutation to an otherwise-accepted run and requires the
// specific verdict for it.
//
// The mutations are the four defects task 2.3 actually shipped with, plus the
// two liveness/accounting failures that would let an empty or unjustified run
// pass: a removed class, a duplicated range, a CPU-index relabel, a suppressed
// heartbeat, an unattributed entry, and a retirement with no copied bytes.

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memexport_census_verify.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::CensusObservation;
using metal::CensusRunSummary;
using metal::CensusSeamId;
using metal::CensusVerdict;
using metal::CensusVerdictName;
using metal::ClassifiedInterval;
using metal::GuestRange;
using metal::GuestRangeSet;
using metal::IntervalAuthority;
using metal::PhysicalConsumerClass;
using metal::VerifyCensusRun;

ClassifiedInterval Interval(uint32_t base, uint32_t length,
                            IntervalAuthority authority, uint64_t sequence) {
  ClassifiedInterval interval;
  interval.range = *GuestRange::FromBaseLength(base, length);
  interval.authority = authority;
  interval.writer_sequence = sequence;
  return interval;
}

CensusObservation SharedVfetchObservation() {
  CensusObservation observation;
  observation.seam = CensusSeamId::kVertexSharedVfetch;
  observation.reported_class = PhysicalConsumerClass::kSharedVfetch;
  observation.requested.Add(*GuestRange::FromBaseLength(0x1000, 0x200));
  observation.classified = {
      Interval(0x1000, 0x100, IntervalAuthority::kLatestPendingMemexport, 7),
      Interval(0x1100, 0x100, IntervalAuthority::kUncovered, 0)};
  observation.retirements = 1;
  observation.bytes_copied = 0x100;
  return observation;
}

CensusObservation CpuIndexObservation() {
  CensusObservation observation;
  observation.seam = CensusSeamId::kVertexCpuIndex;
  observation.reported_class = PhysicalConsumerClass::kCpuIndexStream;
  observation.requested.Add(*GuestRange::FromBaseLength(0x2000, 0x80));
  observation.classified = {
      Interval(0x2000, 0x80, IntervalAuthority::kUncovered, 0)};
  return observation;
}

// A run the verifier must ACCEPT. Every mutation below starts from this.
CensusRunSummary AcceptedRun() {
  CensusRunSummary run;
  run.heartbeats = 12;
  run.observations = {SharedVfetchObservation(), CpuIndexObservation()};
  run.executed_seams = {CensusSeamId::kVertexSharedVfetch,
                        CensusSeamId::kVertexCpuIndex};
  return run;
}

}  // namespace

TEST_CASE("A well-formed census run is accepted",
          "[census_verify]") {
  // The positive control. Without it every rejection below would be satisfied
  // by a verifier that rejects everything.
  const auto result = VerifyCensusRun(AcceptedRun());
  INFO("verdict: " << CensusVerdictName(result.verdict) << " — "
                   << result.detail);
  CHECK(result.accepted());
}

TEST_CASE("A removed class is caught as missing_seam",
          "[census_verify][mutation]") {
  // MUTATION: the CPU-index physical operation still ran, but no seam observed
  // it. This is task 2.3's actual defect — the class was declared and never
  // emitted — and it is invisible to any check that only inspects the
  // observations that DO exist.
  CensusRunSummary run = AcceptedRun();
  run.observations.pop_back();  // drop the CPU-index observation
  // executed_seams still records that the operation happened.
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kMissingSeam);
}

TEST_CASE("An unattributed entry is caught as unknown_attribution",
          "[census_verify][mutation]") {
  CensusRunSummary run = AcceptedRun();
  run.unattributed_entries = 1;
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kUnknownAttribution);
}

TEST_CASE("A duplicated range is caught before normalization",
          "[census_verify][mutation]") {
  // MUTATION: the same bytes claimed twice. Checked BEFORE normalizing,
  // because normalizing first merges the duplicate into its neighbour and
  // yields a clean union — which is exactly how double-counting hides.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].classified.push_back(
      Interval(0x1000, 0x100, IntervalAuthority::kLatestPendingMemexport, 7));
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kDuplicateOrOverlap);
}

TEST_CASE("A partially overlapping range is caught too",
          "[census_verify][mutation]") {
  // Not just exact duplicates: a half-overlap is still double-counted bytes.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].classified.push_back(
      Interval(0x1080, 0x100, IntervalAuthority::kUncovered, 0));
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kDuplicateOrOverlap);
}

TEST_CASE("A CPU-index relabel is caught as seam_class_mismatch",
          "[census_verify][mutation]") {
  // MUTATION: the CPU index stream reported as GPU-shared. This is precisely
  // what task 2.3's vertex seam does today — it labels the whole combined
  // request kSharedVfetch — so this case is the acceptance gate for that
  // repair, not a hypothetical.
  CensusRunSummary run = AcceptedRun();
  run.observations[1].reported_class = PhysicalConsumerClass::kSharedVfetch;
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kSeamClassMismatch);
}

TEST_CASE("A texture route relabel is caught the same way",
          "[census_verify][mutation]") {
  // The 1.3 texture split, as an acceptance check: the legacy CPU fallback
  // reported as the shared direct load.
  CensusObservation legacy;
  legacy.seam = CensusSeamId::kTextureLegacyCpu;
  legacy.reported_class = PhysicalConsumerClass::kSharedTextureLoad;  // wrong
  legacy.requested.Add(*GuestRange::FromBaseLength(0x3000, 0x40));
  legacy.classified = {Interval(0x3000, 0x40, IntervalAuthority::kUncovered, 0)};

  CensusRunSummary run = AcceptedRun();
  run.observations.push_back(legacy);
  run.executed_seams.push_back(CensusSeamId::kTextureLegacyCpu);
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kSeamClassMismatch);
}

TEST_CASE("A suppressed heartbeat is caught as heartbeat_missing",
          "[census_verify][mutation]") {
  // MUTATION: the run produced no liveness signal. Checked FIRST, because an
  // empty run satisfies every other check vacuously — "nothing was wrong" and
  // "nothing happened" must not look alike.
  CensusRunSummary run = AcceptedRun();
  run.heartbeats = 0;
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kHeartbeatMissing);
}

TEST_CASE("An empty run cannot pass by having nothing to check",
          "[census_verify][mutation]") {
  // The degenerate case the heartbeat check exists for.
  CensusRunSummary run;
  const auto result = VerifyCensusRun(run);
  CHECK(result.verdict == CensusVerdict::kHeartbeatMissing);
}

TEST_CASE("A shape-wrong union is caught even when byte totals agree",
          "[census_verify][mutation][shape]") {
  // MUTATION: the classified intervals sum to the requested byte count but
  // cover DIFFERENT addresses. Task 2.2 proved totals cannot see this, and
  // task 2.3's live check compared totals under a comment claiming union
  // equality — so this is the acceptance gate for that repair.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].classified = {
      Interval(0x9000, 0x100, IntervalAuthority::kLatestPendingMemexport, 7),
      Interval(0x9100, 0x100, IntervalAuthority::kUncovered, 0)};
  // Same 0x200 bytes, entirely the wrong range.
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kUnionMismatch);
}

TEST_CASE("A short union is caught", "[census_verify][mutation][shape]") {
  CensusRunSummary run = AcceptedRun();
  run.observations[0].classified.pop_back();  // drops the uncovered tail
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kUnionMismatch);
}

TEST_CASE("A retirement with no copied bytes is caught",
          "[census_verify][mutation]") {
  // MUTATION: records retired without the copy that justifies them. Task 2.3
  // could not detect this because bytes_copied was written only by an unused
  // scope, so copied=0 beside nonzero retirements was ambiguous rather than
  // wrong.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].bytes_copied = 0;
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kCopiedRetiredMismatch);
}

TEST_CASE("Zero retirements with zero copied bytes is fine",
          "[census_verify]") {
  // The complement: a consumer that retired nothing owes no copy. Without this
  // the rule above would reject every disjoint consumer.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].retirements = 0;
  run.observations[0].bytes_copied = 0;
  CHECK(VerifyCensusRun(run).accepted());
}

TEST_CASE("Every seam has a declared expected class",
          "[census_verify]") {
  // The expected-class table is what makes a relabel detectable, so a seam
  // added later without an entry must fail here rather than silently inheriting
  // whatever the default is.
  for (size_t i = 0; i < size_t(CensusSeamId::kCount); ++i) {
    const CensusSeamId seam = CensusSeamId(i);
    INFO("seam " << metal::CensusSeamIdName(seam));
    // The metadata guard is the one seam with no consumer class: it reads no
    // guest bytes and is recorded on its own counters.
    if (seam == CensusSeamId::kMetadataGuard) {
      continue;
    }
    CensusObservation observation;
    observation.seam = seam;
    observation.reported_class = metal::ExpectedClassForSeam(seam);
    observation.requested.Add(*GuestRange::FromBaseLength(0x5000, 0x10));
    observation.classified = {
        Interval(0x5000, 0x10, IntervalAuthority::kUncovered, 0)};
    CensusRunSummary run;
    run.heartbeats = 1;
    run.observations = {observation};
    run.executed_seams = {seam};
    CHECK(VerifyCensusRun(run).accepted());
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
