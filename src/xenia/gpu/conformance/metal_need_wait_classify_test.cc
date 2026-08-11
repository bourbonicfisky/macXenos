/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free) conformance for need-based consumer waits
// (OpenSpec optimize-need-based-consumer-waits, design D1/D1c, gate D5).
//
// What these cases are FOR: the Route A classification decides whether the
// consumer corridor may skip its unconditional synchronous boundary. A wrong
// verdict does not fail loudly -- it publishes guest bytes the GPU may still be
// writing -- so the rule is a pure function precisely so every case, including
// the two exact boundary conditions (`hw == tag`, `cur == tag`), can be
// enumerated here rather than being reachable only through a live corridor.
//
// Covered:
//   * the four D1 cases (untagged fallback, publish-immediate, narrow wait with
//     a commit, narrow wait without one) and both boundary conditions;
//   * the fail-closed inputs: empty selection, and a tag claiming to be newer
//     than the open command buffer (design D1d's `tag <= cur` invariant);
//   * the D1a deferred check -- an untagged record of EITHER producer kind
//     (kPendingResolve as well as kPendingMemexport) is refused, both by the
//     classification and by the shared authority gate under kNone evidence,
//     plus the full admission table the resolve-writeback half of
//     DrainPendingWritebacks now evaluates per record;
//   * the case-3 timeout contract: publication fails closed and the open buffer
//     is intentionally left uncommitted when it held no selected producer;
//   * a mutation battery over the classification and evidence rules;
//   * the engagement meter's counting, saturation and banner discipline.
//
// Pure C++ + Catch2; links no Metal device and no command processor. The
// authority-gate half calls the SAME production functions the command processor
// calls -- there is no test-only copy of any rule.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_need_wait_classify.h"

namespace xe {
namespace gpu {
namespace metal {
namespace test {

namespace {

PendingProducer MakeProducer(uint32_t base, uint32_t size, TransitState state,
                             uint64_t submission, uint64_t sequence = 1) {
  return PendingProducer{*GuestRange::FromBaseLength(base, size), state,
                         submission, sequence};
}

// A selection of tagged memexport producers, sequence-ordered like the
// authority layer's own selected set.
std::vector<PendingProducer> Tagged(std::vector<uint64_t> tags,
                                    TransitState state =
                                        TransitState::kPendingMemexport) {
  std::vector<PendingProducer> out;
  uint32_t base = 0x10000u;
  uint64_t sequence = 1;
  for (uint64_t tag : tags) {
    out.push_back(MakeProducer(base, 0x400u, state, tag, sequence));
    base += 0x1000u;
    ++sequence;
  }
  return out;
}

// The ONE thing these cases model that production owns: the ordering of the
// need-based branch's side effects (commit only when a selected tag is in the
// open buffer, then wait, then fail closed on timeout). Everything it decides
// with -- the classification and the publication predicate -- is the real
// production function, so what is replicated here is three lines of sequencing,
// not a rule.
struct NeedBasedCorridorTrace {
  uint32_t commits = 0;
  uint32_t waits = 0;
  uint64_t waited_for = 0;
  bool published = false;
  bool complete = true;
  MemexportBoundaryEvidence evidence = MemexportBoundaryEvidence::kNone;
};

// `wait_succeeds` false models WaitForSubmission returning false (timeout).
// `drained_records` is what the drain reported, so the publication verdict is
// computed by the production predicate rather than asserted.
NeedBasedCorridorTrace RunNeedBasedBranch(
    const std::vector<PendingProducer>& selected,
    const NeedBasedWaitClassification& classification, bool wait_succeeds,
    uint32_t drained_records) {
  NeedBasedCorridorTrace trace;
  if (!classification.takes_need_path()) {
    return trace;  // the caller would have taken the boundary path instead.
  }
  if (classification.decision == NeedBasedWaitDecision::kNarrowWait) {
    if (classification.must_commit_open_buffer) {
      ++trace.commits;
    }
    ++trace.waits;
    trace.waited_for = classification.wait_submission;
    if (!wait_succeeds) {
      // Fail closed: no drain, no evidence, incomplete outcome.
      trace.complete = false;
      return trace;
    }
  }
  trace.published = true;
  ConsumerDecision decision;
  decision.selected = selected;
  ProducerDrainOutcome drain;
  drain.drained_records = drained_records;
  trace.complete = ConsumerPublicationComplete(decision, drain);
  return trace;
}

}  // namespace

// --- D1 case 1: any untagged record falls back -----------------------------

TEST_CASE("NeedWait: an untagged record falls back to the full boundary",
          "[l0][needwait]") {
  // The untagged test must be over the WHOLE selection, not a per-record early
  // exit a later tagged record could override, so the untagged record is placed
  // first, last, and alone in turn.
  const std::vector<std::vector<uint64_t>> shapes = {
      {0}, {0, 5}, {5, 0}, {5, 0, 7}};
  for (const std::vector<uint64_t>& tags : shapes) {
    INFO("tags size " << tags.size());
    NeedBasedWaitClassification c = ClassifyNeedBasedWait(
        Tagged(tags), /*completed_high_water=*/100u,
        /*open_buffer_submission=*/101u);
    REQUIRE(c.decision == NeedBasedWaitDecision::kFallbackUntagged);
    REQUIRE(c.fallback_cause == NeedBasedWaitFallbackCause::kUntaggedRecord);
    REQUIRE_FALSE(c.takes_need_path());
    // A fallback names no wait and no commit: the boundary path owns both.
    REQUIRE(c.wait_submission == 0u);
    REQUIRE_FALSE(c.must_commit_open_buffer);
    REQUIRE(c.selected_total == uint32_t(tags.size()));
  }
}

TEST_CASE("NeedWait: the untagged refusal covers BOTH producer kinds",
          "[l0][needwait][memexport]") {
  // Design D1a's deferred check, FIRST layer. The classification refuses an
  // untagged record whatever kind it is -- it reads the submission tag and
  // never TransitState -- so an untagged kPendingResolve record is never handed
  // to the need-based (kNone) path in the first place. The second layer (the
  // shared authority gate the production drain evaluates per record) is the
  // next two cases.
  for (TransitState state :
       {TransitState::kPendingResolve, TransitState::kPendingMemexport}) {
    NeedBasedWaitClassification c =
        ClassifyNeedBasedWait(Tagged({0}, state), /*completed=*/50u,
                              /*open_buffer_submission=*/51u);
    REQUIRE(c.decision == NeedBasedWaitDecision::kFallbackUntagged);
    REQUIRE(c.fallback_cause == NeedBasedWaitFallbackCause::kUntaggedRecord);
  }
  // Mixed kinds, one of them untagged: still a fallback.
  std::vector<PendingProducer> mixed = {
      MakeProducer(0x20000u, 0x400u, TransitState::kPendingResolve, 0u, 1u),
      MakeProducer(0x21000u, 0x400u, TransitState::kPendingMemexport, 9u, 2u)};
  NeedBasedWaitClassification c = ClassifyNeedBasedWait(mixed, 50u, 51u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kFallbackUntagged);
}

TEST_CASE("NeedWait: the shared authority gate refuses an untagged record under "
          "kNone for either producer kind",
          "[l0][needwait][memexport]") {
  // The second half of the same protection, in the production gate itself: even
  // if a classification bug routed an untagged record onto the need-based path,
  // kNone evidence fails it CLOSED rather than publishing it. This is the
  // defense-in-depth claim design D1 makes; it is checked, not asserted.
  //
  // `DecideMemexportConvergenceGate` is the predicate BOTH halves of
  // MetalCommandProcessor::DrainPendingWritebacks call -- memexport records
  // through ConvergeCompletedMemexportRam, resolve writebacks directly, both
  // paired with MemexportConvergenceGateAttempts to decide admission. It is
  // named for the seam it was written at, but it reads only the submission tag
  // and the caller's evidence, never TransitState, which is what makes one rule
  // cover two record kinds.
  for (TransitState state :
       {TransitState::kPendingResolve, TransitState::kPendingMemexport}) {
    PendingProducer untagged =
        MakeProducer(0x30000u, 0x400u, state, /*submission=*/0u);
    for (uint64_t completed : {uint64_t(0), uint64_t(1), uint64_t(1u << 20)}) {
      REQUIRE(DecideMemexportConvergenceGate(
                  untagged, completed, MemexportBoundaryEvidence::kNone) ==
              MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary);
      REQUIRE_FALSE(MemexportConvergenceGateAttempts(
          DecideMemexportConvergenceGate(untagged, completed,
                                         MemexportBoundaryEvidence::kNone)));
    }
    // ...and the boundary path's evidence is what makes the SAME record
    // publishable, which is why the fallback must stay byte-for-byte.
    REQUIRE(DecideMemexportConvergenceGate(
                untagged, /*completed=*/0u,
                MemexportBoundaryEvidence::kSynchronousBoundaryCompleted) ==
            MemexportConvergenceGate::kAttemptedSynchronousBoundary);
  }
}

TEST_CASE(
    "NeedWait: the resolve-writeback admission table is the shared gate, "
    "whole",
    "[l0][needwait][memexport][authority]") {
  // The FULL rule the resolve half of DrainPendingWritebacks evaluates per
  // record, enumerated. It used to test `wb.submission > completed` alone,
  // which reads submission 0 as `0 <= completed` and therefore ADMITTED an
  // untagged record with no proof whatsoever -- the one publication the
  // memexport half of the same function refuses. The two rows that pin that
  // down are the untagged ones; the tagged rows pin the byte-equivalence the
  // change had to preserve, because they are the only shape the async default
  // actually produces.
  struct Row {
    const char* what;
    uint64_t submission;
    uint64_t completed;
    MemexportBoundaryEvidence evidence;
    MemexportConvergenceGate expected;
    bool admitted;
  };
  const Row rows[] = {
      // Tagged, retired: admitted on its own tag, evidence irrelevant. The
      // need-based path's kPublishImmediate and post-narrow-wait shapes.
      {"tagged retired (kNone)", 7u, 9u, MemexportBoundaryEvidence::kNone,
       MemexportConvergenceGate::kAttemptedCompletedSubmission, true},
      {"tagged retired at the mark", 9u, 9u, MemexportBoundaryEvidence::kNone,
       MemexportConvergenceGate::kAttemptedCompletedSubmission, true},
      {"tagged retired (boundary evidence)", 7u, 9u,
       MemexportBoundaryEvidence::kSynchronousBoundaryCompleted,
       MemexportConvergenceGate::kAttemptedCompletedSubmission, true},
      // Tagged, still in flight: REFUSED even under boundary evidence. A
      // caller cannot talk an unfinished submission past its own high-water
      // mark; the record is retained for a later drain.
      {"tagged in flight (kNone)", 10u, 9u, MemexportBoundaryEvidence::kNone,
       MemexportConvergenceGate::kNotAttemptedIncompleteSubmission, false},
      {"tagged in flight (boundary evidence)", 10u, 9u,
       MemexportBoundaryEvidence::kSynchronousBoundaryCompleted,
       MemexportConvergenceGate::kNotAttemptedIncompleteSubmission, false},
      // Untagged: decidable ONLY by the caller's proof. `completed` is large
      // here on purpose -- under the old predicate 0 > completed was false and
      // this row published.
      {"untagged without evidence", 0u, 1u << 20,
       MemexportBoundaryEvidence::kNone,
       MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary, false},
      {"untagged under the full boundary", 0u, 1u << 20,
       MemexportBoundaryEvidence::kSynchronousBoundaryCompleted,
       MemexportConvergenceGate::kAttemptedSynchronousBoundary, true},
      // ...including with nothing completed at all: the proof is the caller's
      // boundary, never an inference from the mark.
      {"untagged under the boundary, nothing completed", 0u, 0u,
       MemexportBoundaryEvidence::kSynchronousBoundaryCompleted,
       MemexportConvergenceGate::kAttemptedSynchronousBoundary, true},
  };
  for (const Row& row : rows) {
    INFO(row.what);
    PendingProducer resolve = MakeProducer(
        0x40000u, 0x800u, TransitState::kPendingResolve, row.submission);
    const MemexportConvergenceGate gate = DecideMemexportConvergenceGate(
        resolve, row.completed, row.evidence);
    REQUIRE(gate == row.expected);
    // Admission is the gate paired with MemexportConvergenceGateAttempts --
    // the production drain's exact expression, not a restatement of it.
    REQUIRE(MemexportConvergenceGateAttempts(gate) == row.admitted);
    // Kind-independence, stated as a checked fact rather than a comment: the
    // same row decides identically for a memexport record.
    PendingProducer memexport = MakeProducer(
        0x40000u, 0x800u, TransitState::kPendingMemexport, row.submission);
    REQUIRE(DecideMemexportConvergenceGate(memexport, row.completed,
                                           row.evidence) == gate);
  }
}

// --- D1 case 2: everything retired -> publish immediately ------------------

TEST_CASE("NeedWait: all selected tags retired publishes immediately",
          "[l0][needwait]") {
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(Tagged({3, 7, 5}), /*completed_high_water=*/9u,
                            /*open_buffer_submission=*/10u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kPublishImmediate);
  REQUIRE(c.takes_need_path());
  REQUIRE(c.fallback_cause == NeedBasedWaitFallbackCause::kNotFallback);
  // No commit, no wait: the two facts the whole change exists to produce.
  REQUIRE_FALSE(c.must_commit_open_buffer);
  REQUIRE(c.wait_submission == 0u);
  REQUIRE(c.selected_total == 3u);
  REQUIRE(c.tagged_total == 3u);
  REQUIRE(c.already_complete_total == 3u);
  REQUIRE(c.max_selected_tag == 7u);
}

TEST_CASE("NeedWait: BOUNDARY hw == tag is retired, hw == tag - 1 is not",
          "[l0][needwait][boundary]") {
  // The high-water mark is INCLUSIVE (ProducerComplete: tag <= completed). One
  // off-by-one here is the difference between publishing a finished export and
  // publishing one still in flight.
  NeedBasedWaitClassification at_mark =
      ClassifyNeedBasedWait(Tagged({12}), /*completed=*/12u,
                            /*open_buffer_submission=*/13u);
  REQUIRE(at_mark.decision == NeedBasedWaitDecision::kPublishImmediate);
  REQUIRE(at_mark.already_complete_total == 1u);

  NeedBasedWaitClassification one_short =
      ClassifyNeedBasedWait(Tagged({12}), /*completed=*/11u,
                            /*open_buffer_submission=*/13u);
  REQUIRE(one_short.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE(one_short.already_complete_total == 0u);
  REQUIRE(one_short.wait_submission == 12u);
  // 12 != 13: the producer is in an ALREADY COMMITTED buffer, so no commit.
  REQUIRE_FALSE(one_short.must_commit_open_buffer);
}

// --- D1 case 3: narrow wait, with and without the commit -------------------

TEST_CASE("NeedWait: a narrow wait waits for the MAX selected tag only",
          "[l0][needwait]") {
  // Waiting for the max is what makes this narrow: every smaller selected tag
  // is retired by the same wait, and nothing outside the selection is waited on.
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(Tagged({4, 20, 11}), /*completed=*/10u,
                            /*open_buffer_submission=*/25u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE(c.wait_submission == 20u);
  REQUIRE(c.max_selected_tag == 20u);
  REQUIRE(c.already_complete_total == 1u);  // only tag 4
  REQUIRE_FALSE(c.must_commit_open_buffer);
}

TEST_CASE("NeedWait: BOUNDARY cur == tag is the commit case",
          "[l0][needwait][boundary]") {
  // A producer still being ENCODED in the open buffer can only be awaited after
  // that buffer is committed; anything else would wait for a submission that
  // will never be handed to the GPU.
  NeedBasedWaitClassification in_open =
      ClassifyNeedBasedWait(Tagged({9, 14}), /*completed=*/8u,
                            /*open_buffer_submission=*/14u);
  REQUIRE(in_open.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE(in_open.must_commit_open_buffer);
  REQUIRE(in_open.wait_submission == 14u);

  // Same selection, buffer one submission further along: already committed.
  NeedBasedWaitClassification committed =
      ClassifyNeedBasedWait(Tagged({9, 14}), /*completed=*/8u,
                            /*open_buffer_submission=*/15u);
  REQUIRE(committed.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE_FALSE(committed.must_commit_open_buffer);
}

TEST_CASE("NeedWait: with NO open buffer the no-commit branch applies",
          "[l0][needwait]") {
  // `open_buffer_submission == 0` is the caller's factual "no buffer open"
  // (production reads it only under the frame_command_buffer_ guard). It must
  // never be read as a submission id that could equal a tag.
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(Tagged({6}), /*completed=*/2u,
                            /*open_buffer_submission=*/0u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE_FALSE(c.must_commit_open_buffer);
  REQUIRE(c.wait_submission == 6u);

  // ...and it must not be mistaken for an untagged record either way round.
  NeedBasedWaitClassification retired =
      ClassifyNeedBasedWait(Tagged({6}), /*completed=*/6u,
                            /*open_buffer_submission=*/0u);
  REQUIRE(retired.decision == NeedBasedWaitDecision::kPublishImmediate);
}

// --- Fail-closed inputs ----------------------------------------------------

TEST_CASE("NeedWait: an empty selection fails closed and is named",
          "[l0][needwait]") {
  // Unreachable from the corridor (an empty selection early-outs first), but
  // representable -- and "nothing selected" must never read as "everything is
  // already retired, publish".
  NeedBasedWaitClassification c = ClassifyNeedBasedWait({}, 100u, 101u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kFallbackUntagged);
  REQUIRE(c.fallback_cause == NeedBasedWaitFallbackCause::kEmptySelection);
  REQUIRE_FALSE(c.takes_need_path());
  REQUIRE(c.selected_total == 0u);
}

TEST_CASE("NeedWait: a tag newer than the open buffer fails closed (D1d)",
          "[l0][needwait]") {
  // The open buffer's submission is the highest id that exists, so `tag > cur`
  // is structurally impossible. It is therefore an ambiguity, not a wait
  // target: waiting for it would block until the timeout and then fail anyway.
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(Tagged({30}), /*completed=*/5u,
                            /*open_buffer_submission=*/29u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kFallbackUntagged);
  REQUIRE(c.fallback_cause ==
          NeedBasedWaitFallbackCause::kTagAheadOfOpenBuffer);
  REQUIRE_FALSE(c.takes_need_path());

  // Equality is the COMMIT case, not an anomaly -- the invariant is `tag <= cur`.
  NeedBasedWaitClassification equal =
      ClassifyNeedBasedWait(Tagged({29}), /*completed=*/5u,
                            /*open_buffer_submission=*/29u);
  REQUIRE(equal.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE(equal.must_commit_open_buffer);

  // An untagged record outranks the anomaly: the expected steady fallback must
  // not be reported as an unexplained one.
  NeedBasedWaitClassification both =
      ClassifyNeedBasedWait(Tagged({0, 30}), /*completed=*/5u,
                            /*open_buffer_submission=*/29u);
  REQUIRE(both.fallback_cause == NeedBasedWaitFallbackCause::kUntaggedRecord);
}

// --- The case-3 timeout contract (D1 / D5 precommit) ------------------------

TEST_CASE("NeedWait: a timed-out narrow wait fails closed and leaves the open "
          "buffer uncommitted when it held no selected producer",
          "[l0][needwait][timeout]") {
  // This side effect is deliberately NEW: the historical path always committed
  // before waiting, so "unchanged" cannot be inferred here -- it has to be
  // stated and checked.
  std::vector<PendingProducer> selected = Tagged({9});
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(selected, /*completed=*/2u,
                            /*open_buffer_submission=*/20u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kNarrowWait);
  REQUIRE_FALSE(c.must_commit_open_buffer);

  NeedBasedCorridorTrace timed_out = RunNeedBasedBranch(
      selected, c, /*wait_succeeds=*/false, /*drained_records=*/0u);
  REQUIRE(timed_out.waits == 1u);
  REQUIRE(timed_out.waited_for == 9u);
  REQUIRE(timed_out.commits == 0u);  // the intended new side effect
  REQUIRE_FALSE(timed_out.published);
  REQUIRE_FALSE(timed_out.complete);
  REQUIRE(timed_out.evidence == MemexportBoundaryEvidence::kNone);
}

TEST_CASE("NeedWait: a timed-out narrow wait in the COMMIT case still committed",
          "[l0][needwait][timeout]") {
  // The commit is not conditional on the wait succeeding: it is what makes the
  // wait satisfiable at all, so it must already have happened.
  std::vector<PendingProducer> selected = Tagged({20});
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(selected, /*completed=*/2u,
                            /*open_buffer_submission=*/20u);
  REQUIRE(c.must_commit_open_buffer);
  NeedBasedCorridorTrace timed_out = RunNeedBasedBranch(
      selected, c, /*wait_succeeds=*/false, /*drained_records=*/0u);
  REQUIRE(timed_out.commits == 1u);
  REQUIRE_FALSE(timed_out.complete);
  REQUIRE_FALSE(timed_out.published);
}

TEST_CASE("NeedWait: a short drain on the need path fails the consumer closed",
          "[l0][needwait]") {
  // Publication failure is the SAME predicate the boundary path uses -- a
  // narrower wait may not weaken it. Two of three records retired is a failure,
  // not a partial success.
  std::vector<PendingProducer> selected = Tagged({3, 4, 5});
  NeedBasedWaitClassification c =
      ClassifyNeedBasedWait(selected, /*completed=*/5u,
                            /*open_buffer_submission=*/6u);
  REQUIRE(c.decision == NeedBasedWaitDecision::kPublishImmediate);
  REQUIRE(RunNeedBasedBranch(selected, c, true, /*drained_records=*/2u)
              .complete == false);
  REQUIRE(RunNeedBasedBranch(selected, c, true, /*drained_records=*/3u)
              .complete == true);
  // Immediate publication takes neither a commit nor a wait.
  NeedBasedCorridorTrace ok = RunNeedBasedBranch(selected, c, true, 3u);
  REQUIRE(ok.commits == 0u);
  REQUIRE(ok.waits == 0u);
}

// --- The tagged paths pass kNone, and kNone is sufficient for them ---------

TEST_CASE("NeedWait: kNone admits a retired tagged record and refuses an "
          "in-flight one",
          "[l0][needwait][memexport]") {
  // Design D1a: the tagged half of the gate never consults evidence, which is
  // exactly why the need-based paths owe none. If this ever stopped holding,
  // the change's whole safety argument would go with it.
  PendingProducer tagged =
      MakeProducer(0x40000u, 0x400u, TransitState::kPendingMemexport, 7u);
  REQUIRE(DecideMemexportConvergenceGate(tagged, /*completed=*/7u,
                                         MemexportBoundaryEvidence::kNone) ==
          MemexportConvergenceGate::kAttemptedCompletedSubmission);
  REQUIRE(DecideMemexportConvergenceGate(tagged, /*completed=*/6u,
                                         MemexportBoundaryEvidence::kNone) ==
          MemexportConvergenceGate::kNotAttemptedIncompleteSubmission);
  // ...and boundary evidence cannot talk an in-flight tagged record past its own
  // high-water mark either.
  REQUIRE(DecideMemexportConvergenceGate(
              tagged, /*completed=*/6u,
              MemexportBoundaryEvidence::kSynchronousBoundaryCompleted) ==
          MemexportConvergenceGate::kNotAttemptedIncompleteSubmission);
}

// --- Vocabulary completeness ----------------------------------------------

TEST_CASE("NeedWait: the decision vocabulary is complete and named",
          "[l0][needwait]") {
  REQUIRE(kNeedBasedWaitDecisions.size() == kNeedBasedWaitDecisionCount);
  uint32_t need_path_values = 0;
  for (NeedBasedWaitDecision d : kNeedBasedWaitDecisions) {
    REQUIRE(IsNeedBasedWaitDecisionValue(d));
    std::string name = NeedBasedWaitDecisionName(d);
    REQUIRE_FALSE(name.empty());
    REQUIRE(name != "unknown");
    if (NeedBasedWaitDecisionTakesNeedPath(d)) {
      ++need_path_values;
    }
  }
  REQUIRE(need_path_values == 2u);  // publish-immediate and narrow-wait only
  // kCount is a bound, never an observation, and can never license the new path.
  REQUIRE_FALSE(IsNeedBasedWaitDecisionValue(NeedBasedWaitDecision::kCount));
  REQUIRE_FALSE(
      NeedBasedWaitDecisionTakesNeedPath(NeedBasedWaitDecision::kCount));
  REQUIRE(std::string(NeedBasedWaitDecisionName(
              NeedBasedWaitDecision::kCount)) == "unknown");
  // A default-constructed classification is the pre-feature boundary.
  NeedBasedWaitClassification fresh;
  REQUIRE(fresh.decision == NeedBasedWaitDecision::kFallbackUntagged);
  REQUIRE_FALSE(fresh.takes_need_path());
}

TEST_CASE("NeedWait: every fallback cause is named",
          "[l0][needwait]") {
  for (uint32_t i = 0; i < kNeedBasedWaitFallbackCauseCount; ++i) {
    std::string name =
        NeedBasedWaitFallbackCauseName(NeedBasedWaitFallbackCause(i));
    REQUIRE_FALSE(name.empty());
    REQUIRE(name != "unknown");
  }
  REQUIRE(std::string(NeedBasedWaitFallbackCauseName(
              NeedBasedWaitFallbackCause::kCount)) == "unknown");
}

TEST_CASE("NeedWait: the flag's env var name is the one the gate arms set",
          "[l0][needwait]") {
  // Two spellings of "is this enabled" is how a control arm ends up running
  // with a different configuration than the arm it is compared against.
  REQUIRE(std::string(kNeedBasedWaitsEnvVar) == "XE_METAL_NEED_BASED_WAITS");
}

// --- Mutation battery ------------------------------------------------------

TEST_CASE("NeedWait mutation: every input dimension changes the verdict it "
          "must change, and nothing else",
          "[l0][needwait][mutation]") {
  // Base: two tagged producers, both retired, open buffer beyond both.
  const std::vector<PendingProducer> base = Tagged({4, 8});
  const uint64_t hw = 8u;
  const uint64_t cur = 9u;
  REQUIRE(ClassifyNeedBasedWait(base, hw, cur).decision ==
          NeedBasedWaitDecision::kPublishImmediate);

  struct Mutation {
    const char* name;
    std::vector<PendingProducer> selected;
    uint64_t hw;
    uint64_t cur;
    NeedBasedWaitDecision expected;
    bool expect_commit;
  };
  const std::vector<Mutation> mutations = {
      {"zero one tag", Tagged({0, 8}), hw, cur,
       NeedBasedWaitDecision::kFallbackUntagged, false},
      {"zero the other tag", Tagged({4, 0}), hw, cur,
       NeedBasedWaitDecision::kFallbackUntagged, false},
      {"drop the high-water below the max tag", base, 7u, cur,
       NeedBasedWaitDecision::kNarrowWait, false},
      {"drop the high-water below BOTH tags", base, 3u, cur,
       NeedBasedWaitDecision::kNarrowWait, false},
      {"pull the open buffer onto the max tag", base, 7u, 8u,
       NeedBasedWaitDecision::kNarrowWait, true},
      {"pull the open buffer onto the LOWER tag", Tagged({4, 8}), 3u, 4u,
       // 8 > cur=4: structurally impossible, so it fails closed rather than
       // committing for a tag that cannot exist.
       NeedBasedWaitDecision::kFallbackUntagged, false},
      {"close the open buffer", base, 7u, 0u,
       NeedBasedWaitDecision::kNarrowWait, false},
      {"empty the selection", {}, hw, cur,
       NeedBasedWaitDecision::kFallbackUntagged, false},
      {"raise the high-water far past both", base, 1000u, 1001u,
       NeedBasedWaitDecision::kPublishImmediate, false},
  };
  for (const Mutation& m : mutations) {
    INFO("mutation: " << m.name);
    NeedBasedWaitClassification c =
        ClassifyNeedBasedWait(m.selected, m.hw, m.cur);
    REQUIRE(c.decision == m.expected);
    REQUIRE(c.must_commit_open_buffer == m.expect_commit);
    // Invariant across EVERY mutation: a fallback never names a wait target,
    // and a need-path verdict never names one it did not select.
    if (c.takes_need_path()) {
      REQUIRE(c.tagged_total == c.selected_total);
      if (c.decision == NeedBasedWaitDecision::kNarrowWait) {
        REQUIRE(c.wait_submission == c.max_selected_tag);
        REQUIRE(c.wait_submission != 0u);
      } else {
        REQUIRE(c.wait_submission == 0u);
      }
    } else {
      REQUIRE(c.wait_submission == 0u);
      REQUIRE_FALSE(c.must_commit_open_buffer);
    }
  }
}

TEST_CASE("NeedWait mutation: the evidence rules cannot be weakened",
          "[l0][needwait][mutation][memexport]") {
  // Mutating the evidence value, the tag and the high-water independently: the
  // ONLY publishable combinations are (tagged AND retired) or (untagged AND
  // synchronous-boundary evidence). Anything else must not reach the target.
  for (uint64_t tag : {uint64_t(0), uint64_t(5)}) {
    for (uint64_t completed : {uint64_t(0), uint64_t(4), uint64_t(5)}) {
      for (MemexportBoundaryEvidence evidence : kMemexportBoundaryEvidences) {
        PendingProducer p = MakeProducer(
            0x50000u, 0x400u, TransitState::kPendingMemexport, tag);
        const bool expect_attempt =
            tag != 0
                ? tag <= completed
                : evidence ==
                      MemexportBoundaryEvidence::kSynchronousBoundaryCompleted;
        INFO("tag " << tag << " completed " << completed << " evidence "
                    << MemexportBoundaryEvidenceName(evidence));
        REQUIRE(MemexportConvergenceGateAttempts(DecideMemexportConvergenceGate(
                    p, completed, evidence)) == expect_attempt);
      }
    }
  }
}

// --- The engagement meter --------------------------------------------------

TEST_CASE("NeedWait meter: counts verdicts at the verdict, not the branch",
          "[l0][needwait][meter]") {
  NeedBasedWaitMeter meter;
  REQUIRE_FALSE(meter.AnyActivity());
  REQUIRE_FALSE(meter.Engaged());

  meter.CountRequest();
  REQUIRE(meter.AnyActivity());
  // A request counted with no classification is the flag-OFF control shape: a
  // denominator with no engagement, which must NOT read as engagement.
  REQUIRE_FALSE(meter.Engaged());
  REQUIRE(meter.classified == 0u);

  meter.CountRequest();
  meter.RecordClassification(
      ClassifyNeedBasedWait(Tagged({4, 8}), /*hw=*/8u, /*cur=*/9u));
  REQUIRE(meter.publish_immediate == 1u);
  REQUIRE(meter.classified == 1u);
  REQUIRE(meter.selected_total == 2u);
  REQUIRE(meter.tagged_total == 2u);
  REQUIRE(meter.already_complete_total == 2u);
  REQUIRE(meter.Engaged());

  meter.CountRequest();
  meter.RecordClassification(
      ClassifyNeedBasedWait(Tagged({4, 8}), /*hw=*/3u, /*cur=*/8u));
  REQUIRE(meter.tagged_waits == 1u);
  REQUIRE(meter.commit_only == 1u);

  meter.CountRequest();
  meter.RecordClassification(
      ClassifyNeedBasedWait(Tagged({0}), /*hw=*/3u, /*cur=*/8u));
  REQUIRE(meter.fallback_untagged == 1u);
  REQUIRE(meter.fallback_tag_ahead == 0u);

  meter.CountRequest();
  meter.RecordClassification(
      ClassifyNeedBasedWait(Tagged({30}), /*hw=*/3u, /*cur=*/8u));
  REQUIRE(meter.fallback_tag_ahead == 1u);
  REQUIRE(meter.fallback_untagged == 1u);  // the anomaly did not hide in it

  meter.RecordClassification(ClassifyNeedBasedWait({}, 3u, 8u));
  REQUIRE(meter.fallback_empty == 1u);

  // Every classified request landed in exactly one verdict bucket.
  REQUIRE(meter.publish_immediate + meter.tagged_waits +
              meter.fallback_untagged + meter.fallback_tag_ahead +
              meter.fallback_empty ==
          meter.classified);
  REQUIRE(meter.requests == 5u);
}

TEST_CASE("NeedWait meter: blocked and timed-out waits are counted apart",
          "[l0][needwait][meter]") {
  NeedBasedWaitMeter meter;
  meter.RecordWait(/*wait_ns=*/1000u, /*did_block=*/false, /*timed_out=*/false);
  REQUIRE(meter.tagged_wait_blocked == 0u);
  REQUIRE(meter.tagged_wait_timeouts == 0u);
  REQUIRE(meter.wait_ns_total == 1000u);
  REQUIRE(meter.wait_ns_max == 1000u);

  meter.RecordWait(5000u, /*did_block=*/true, /*timed_out=*/false);
  REQUIRE(meter.tagged_wait_blocked == 1u);
  REQUIRE(meter.wait_ns_total == 6000u);
  REQUIRE(meter.wait_ns_max == 5000u);

  meter.RecordWait(400u, /*did_block=*/true, /*timed_out=*/true);
  REQUIRE(meter.tagged_wait_blocked == 2u);
  REQUIRE(meter.tagged_wait_timeouts == 1u);
  // A shorter wait must not lower the max.
  REQUIRE(meter.wait_ns_max == 5000u);
}

TEST_CASE("NeedWait meter: counters saturate rather than wrap",
          "[l0][needwait][meter]") {
  // A wrapped session total would read as "almost nothing happened", which is
  // precisely the claim a gate arm is testing.
  NeedBasedWaitMeter meter;
  meter.wait_ns_total = UINT64_MAX - 10u;
  meter.RecordWait(100u, false, false);
  REQUIRE(meter.wait_ns_total == UINT64_MAX);
  meter.requests = UINT64_MAX;
  meter.CountRequest();
  REQUIRE(meter.requests == UINT64_MAX);
}

TEST_CASE("NeedWait meter: the report line is the flag's banner",
          "[l0][needwait][meter]") {
  // D5 reads the flag state from this line. An all-zero ON line and an OFF line
  // must not be the same text, or an arm could not prove which mode it ran in.
  NeedBasedWaitMeter meter;
  char on_buf[512];
  char off_buf[512];
  meter.Format(on_buf, sizeof(on_buf), /*enabled=*/true);
  meter.Format(off_buf, sizeof(off_buf), /*enabled=*/false);
  REQUIRE(std::string(on_buf).find("flag=on") != std::string::npos);
  REQUIRE(std::string(off_buf).find("flag=off") != std::string::npos);
  REQUIRE(std::string(on_buf) != std::string(off_buf));

  // Truncation must still leave a valid C string, never uninitialized stack.
  char tiny[8];
  std::memset(tiny, 0x7F, sizeof(tiny));
  size_t written = meter.Format(tiny, sizeof(tiny), true);
  REQUIRE(written < sizeof(tiny));
  REQUIRE(std::strlen(tiny) < sizeof(tiny));

  // A zero-capacity buffer writes nothing and reports nothing.
  REQUIRE(meter.Format(tiny, 0, true) == 0u);
}

}  // namespace test
}  // namespace metal
}  // namespace gpu
}  // namespace xe
