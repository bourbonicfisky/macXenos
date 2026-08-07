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
using metal::CensusRequestBalance;
using metal::CensusRunSummary;
using metal::CensusSeamId;
using metal::CensusVerdict;
using metal::CensusVerdictName;
using metal::CensusVerifyResult;
using metal::CloseCensusRequest;
using metal::VerifyCensusRequestBalances;
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
  observation.memexport_retirements = 1;
  observation.memexport_copies = 1;
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
  run.observations[0].memexport_copies = 0;  // retired without a completed copy
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kCopiedRetiredMismatch);
}

TEST_CASE("Many retirements behind one copy is caught",
          "[census_verify][mutation]") {
  // The old rule was `retirements && !bytes_copied`, which accepted ten
  // retirements justified by a single byte. The invariant is now a COUNT
  // equality, so the imbalance is visible.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].memexport_retirements = 10;
  run.observations[0].memexport_copies = 1;
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kCopiedRetiredMismatch);
}

TEST_CASE("A resolve retirement needs no memexport copy",
          "[census_verify]") {
  // The old rule FALSELY REJECTED this: a resolve retirement legitimately has
  // no memexport memcpy behind it, so a single combined counter could not
  // express the invariant without over-rejecting resolves.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].memexport_retirements = 0;
  run.observations[0].memexport_copies = 0;
  run.observations[0].resolve_retirements = 3;
  run.observations[0].bytes_copied = 0;
  const auto result = VerifyCensusRun(run);
  INFO("verdict: " << CensusVerdictName(result.verdict) << " — "
                   << result.detail);
  CHECK(result.accepted());
}

TEST_CASE("A live but blind run is caught as engagement_missing",
          "[census_verify][mutation]") {
  // Distinct from heartbeat_missing: the harness ran, the seams saw nothing.
  // Those are different repairs, so they are different verdicts.
  CensusRunSummary run;
  run.heartbeats = 9;
  const auto result = VerifyCensusRun(run);
  INFO(result.detail);
  CHECK(result.verdict == CensusVerdict::kEngagementMissing);
}

TEST_CASE("Zero retirements with zero copied bytes is fine",
          "[census_verify]") {
  // The complement: a consumer that retired nothing owes no copy. Without this
  // the rule above would reject every disjoint consumer.
  CensusRunSummary run = AcceptedRun();
  run.observations[0].memexport_retirements = 0;
  run.observations[0].memexport_copies = 0;
  run.observations[0].bytes_copied = 0;
  CHECK(VerifyCensusRun(run).accepted());
}

// ---------------------------------------------------------------------------
// The classless (metadata guard) rule, and seam-id validity
//
// Five cases, because the previous round left this rule with NO durable pin:
// the completeness test skipped the metadata guard, and the verifier treated
// ANY seam with no expected class as a valid classless seam — which admitted
// the kCount sentinel and out-of-range ids. A run consisting solely of a bogus
// seam was accepted.
// ---------------------------------------------------------------------------

namespace {
CensusRunSummary GuardRun(CensusObservation observation) {
  CensusRunSummary run;
  run.heartbeats = 1;
  run.observations = {observation};
  run.executed_seams = {observation.seam};
  return run;
}
CensusObservation BareGuard() {
  CensusObservation observation;
  observation.seam = CensusSeamId::kMetadataGuard;
  return observation;  // no class, no requested range, no classified range
}
}  // namespace

TEST_CASE("A classless zero-byte metadata guard is accepted",
          "[census_verify][classless]") {
  const auto result = VerifyCensusRun(GuardRun(BareGuard()));
  INFO("verdict: " << CensusVerdictName(result.verdict) << " — "
                   << result.detail);
  CHECK(result.accepted());
}

TEST_CASE("A metadata guard claiming a class is rejected",
          "[census_verify][classless]") {
  CensusObservation guard = BareGuard();
  guard.reported_class = PhysicalConsumerClass::kHostRead;
  CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
        CensusVerdict::kSeamClassMismatch);
}

TEST_CASE("A metadata guard carrying a requested range is rejected",
          "[census_verify][classless]") {
  // The guard reads record METADATA. Letting it carry bytes would fold a
  // zero-byte guard into a corridor's byte accounting and make it look like
  // convergence cost, which task 1.3 established it is not.
  CensusObservation guard = BareGuard();
  guard.requested.Add(*GuestRange::FromBaseLength(0x400, 0x10));
  CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
        CensusVerdict::kSeamClassMismatch);
}

TEST_CASE("A metadata guard carrying a classified range is rejected",
          "[census_verify][classless]") {
  CensusObservation guard = BareGuard();
  guard.classified = {
      Interval(0x400, 0x10, IntervalAuthority::kUncovered, 0)};
  CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
        CensusVerdict::kSeamClassMismatch);
}

TEST_CASE("A metadata guard reporting any accounting is rejected",
          "[census_verify][classless]") {
  // The classless branch used to `continue` before the copy/retirement checks,
  // so a zero-byte guard could report convergence accounting it cannot have
  // performed. Each field is pinned INDEPENDENTLY: a single combined check
  // would pass if only one field were wrongly permitted.
  {
    CensusObservation guard = BareGuard();
    guard.memexport_copies = 5;
    CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
          CensusVerdict::kCopiedRetiredMismatch);
  }
  {
    CensusObservation guard = BareGuard();
    guard.memexport_retirements = 5;
    CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
          CensusVerdict::kCopiedRetiredMismatch);
  }
  {
    CensusObservation guard = BareGuard();
    guard.resolve_retirements = 5;
    CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
          CensusVerdict::kCopiedRetiredMismatch);
  }
  {
    CensusObservation guard = BareGuard();
    guard.bytes_copied = 4096;
    CHECK(VerifyCensusRun(GuardRun(guard)).verdict ==
          CensusVerdict::kCopiedRetiredMismatch);
  }
}

TEST_CASE("The sentinel and out-of-range seam ids are rejected",
          "[census_verify][classless]") {
  // Both were ACCEPTED before this round: the verifier inferred "classless"
  // from a missing expected-class entry, and kCount has no entry.
  CensusObservation sentinel;
  sentinel.seam = CensusSeamId::kCount;
  CHECK(VerifyCensusRun(GuardRun(sentinel)).verdict ==
        CensusVerdict::kInvalidSeam);

  CensusObservation bogus;
  bogus.seam = static_cast<CensusSeamId>(99);
  CHECK(VerifyCensusRun(GuardRun(bogus)).verdict ==
        CensusVerdict::kInvalidSeam);
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
    if (seam == CensusSeamId::kCount) {
      continue;  // sentinel, not a seam
    }
    if (metal::IsClasslessSeam(seam)) {
      // Not skipped silently: the classless seams are pinned by their own
      // five cases above, and this asserts the allow-list and the table agree.
      CHECK_FALSE(metal::ExpectedClassForSeam(seam).has_value());
      continue;
    }
    // The expectation comes from an INDEPENDENT table, not from the function
    // under test — reading it from ExpectedClassForSeam made this tautological,
    // so a newly unhandled seam compared equal to the old default and passed.
    const auto declared = metal::ExpectedClassForSeam(seam);
    REQUIRE(declared.has_value());
    CensusObservation observation;
    observation.seam = seam;
    observation.reported_class = *declared;
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


// ---------------------------------------------------------------------------
// Per-request balance (D2a). Aggregation is only legal over balanced requests.
// ---------------------------------------------------------------------------

TEST_CASE("Cancelling per-request errors are rejected despite equal totals",
          "[metal][memexport][census]") {
  // The concrete cancellation: request A drops an observation, request B
  // invents one. Per-seam totals are 2 == 2 and a total-only check passes.
  const std::vector<CensusRequestBalance> cancelling = {
      {CensusSeamId::kVertexSharedVfetch, 2, 1},
      {CensusSeamId::kVertexSharedVfetch, 0, 1},
  };

  uint64_t executed_total = 0;
  uint64_t observed_total = 0;
  for (const CensusRequestBalance& balance : cancelling) {
    executed_total += balance.executed;
    observed_total += balance.observed;
  }
  // Pin the premise: without per-request closure this run looks clean.
  REQUIRE(executed_total == observed_total);

  const CensusVerifyResult result = VerifyCensusRequestBalances(cancelling);
  REQUIRE_FALSE(result.accepted());
  REQUIRE(result.verdict == CensusVerdict::kMissingSeam);
}

TEST_CASE("Per-request balance separates the two directions",
          "[metal][memexport][census]") {
  // A dropped observation and an invented one need DIFFERENT repairs, so they
  // must not share a verdict.
  REQUIRE(CloseCensusRequest({CensusSeamId::kVertexSharedVfetch, 3, 1}) ==
          CensusVerdict::kMissingSeam);
  REQUIRE(CloseCensusRequest({CensusSeamId::kVertexSharedVfetch, 0, 1}) ==
          CensusVerdict::kUnjustifiedObservation);
  REQUIRE(CloseCensusRequest({CensusSeamId::kVertexSharedVfetch, 2, 2}) ==
          CensusVerdict::kAccepted);
}

TEST_CASE("A zero-work request balances", "[metal][memexport][census]") {
  // Nothing executed and nothing observed is a legal request, not a hole.
  REQUIRE(CloseCensusRequest({CensusSeamId::kTextureCacheHit, 0, 0}) ==
          CensusVerdict::kAccepted);
}

TEST_CASE("Per-request balance rejects an undeclared seam",
          "[metal][memexport][census]") {
  REQUIRE(CloseCensusRequest({CensusSeamId::kCount, 1, 1}) ==
          CensusVerdict::kInvalidSeam);
  REQUIRE(CloseCensusRequest({static_cast<CensusSeamId>(9999), 1, 1}) ==
          CensusVerdict::kInvalidSeam);
}

TEST_CASE("Removing an observation hook fails with execution preserved",
          "[metal][memexport][census]") {
  // The D2a mutation. Because the counters are emitted at independent
  // locations, deleting the observation hook leaves `executed` standing and the
  // request fails locally. A single marker writing both would zero both and
  // this would pass.
  CensusRequestBalance mutated{CensusSeamId::kTextureSharedDirectLoad, 4, 4};
  REQUIRE(CloseCensusRequest(mutated) == CensusVerdict::kAccepted);

  mutated.observed = 0;
  REQUIRE(mutated.executed == 4);
  REQUIRE(CloseCensusRequest(mutated) == CensusVerdict::kMissingSeam);
}

TEST_CASE("An unbalanced request fails before run-level aggregation",
          "[metal][memexport][census]") {
  // Built on a run that is otherwise ACCEPTED, so the verdict below can only
  // come from the balance check. (Heartbeat and engagement gate earlier by
  // design — a bare run fails those first and would never reach here, which is
  // what an earlier version of this test wrongly asserted against.)
  CensusRunSummary healthy = GuardRun(BareGuard());
  REQUIRE(VerifyCensusRun(healthy).accepted());

  healthy.request_balances = {{CensusSeamId::kVertexSharedVfetch, 2, 1}};
  const CensusVerifyResult result = VerifyCensusRun(healthy);
  REQUIRE_FALSE(result.accepted());
  REQUIRE(result.verdict == CensusVerdict::kMissingSeam);
}

TEST_CASE("Balance precedes the aggregate stages in the reported verdict",
          "[metal][memexport][census]") {
  // A run that is BOTH unbalanced and has an unattributed entry must name the
  // unbalanced request. The aggregate failure is downstream of the local one,
  // and reporting it would send the repair to the wrong place.
  CensusRunSummary run = GuardRun(BareGuard());
  run.request_balances = {{CensusSeamId::kVertexSharedVfetch, 0, 1}};
  run.unattributed_entries = 3;
  const CensusVerifyResult result = VerifyCensusRun(run);
  REQUIRE(result.verdict == CensusVerdict::kUnjustifiedObservation);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
