/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_VERIFY_H_
#define XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_VERIFY_H_

// Pure census verifier for the memexport GPU-consumer pre-gate (task 2.4).
//
// Task 2.3 built the census. This decides whether a census RUN may be believed,
// and it exists because 2.3 shipped with four defects that its own output
// looked fine under:
//
//   * the whole vertex request was labelled shared-vfetch, so the CPU index and
//     tessellation-factor streams were reported as GPU-shared — the
//     "CPU index relabelled GPU-shared" mutation, present in the baseline;
//   * the whole texture request was labelled shared-texture-load before route
//     selection, so the legacy-RAM, cache-hit and resolve/bridge classes were
//     declared but never emitted;
//   * the live "union equality" check called IsExactPartition(), which compares
//     two byte TOTALS — the exact weakness task 2.2 demonstrated, reproduced in
//     the live path under a comment claiming otherwise;
//   * bytes_copied was only written by an unused scope, so copied=0 beside
//     nonzero retirements was not evidence of anything.
//
// Every failure below is NAMED. A verifier that returns one boolean cannot tell
// "the seam is missing" from "the seam lied about its class", and those need
// different repairs.

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xenia/gpu/metal/metal_guest_range.h"
#include "xenia/gpu/metal/metal_memexport_consumer_classifier.h"

namespace xe {
namespace gpu {
namespace metal {

// A stable identity for each physical operation the census observes. Stable so
// a seam's EXPECTED class can be validated independently of what the seam
// claimed at runtime — the check that catches a relabel.
enum class CensusSeamId {
  kVertexSharedVfetch,
  kVertexCpuIndex,
  kVertexCpuTessFactor,
  kTextureSharedDirectLoad,
  kTextureLegacyCpu,
  kTextureCacheHit,
  kTextureResolveSnapshot,
  kWaitPredicate,
  kHostRead,
  kFrameEndPublication,
  kLifecyclePublication,
  kMetadataGuard,
  kCount,
};

const char* CensusSeamIdName(CensusSeamId seam);

// The class a given seam MUST report. Declared here, independently of the
// production call site, so a seam that reports something else is a detectable
// mismatch rather than the new truth.
// Returns nullopt for a seam with no declared class. Deliberately NOT a
// defaulted value: the previous version returned kHostRead for anything
// unhandled, and the completeness test read its expectation from this same
// function — so a newly added seam was compared against the default and passed.
// A seam id that names a real physical operation. kCount is a sentinel and
// anything outside the enum is a corrupt id; neither may be treated as a seam.
constexpr bool IsDeclaredSeam(CensusSeamId seam) {
  return size_t(seam) < size_t(CensusSeamId::kCount);
}

// The ONLY seam permitted to carry no consumer class: it inspects pending-record
// metadata and reads no guest bytes. Written as an explicit allow-list rather
// than "whatever ExpectedClassForSeam has no entry for", because the latter
// silently admitted the kCount sentinel and any out-of-range id as valid
// classless seams.
constexpr bool IsClasslessSeam(CensusSeamId seam) {
  return seam == CensusSeamId::kMetadataGuard;
}

constexpr std::optional<PhysicalConsumerClass> ExpectedClassForSeam(
    CensusSeamId seam) {
  switch (seam) {
    case CensusSeamId::kVertexSharedVfetch:
      return PhysicalConsumerClass::kSharedVfetch;
    case CensusSeamId::kVertexCpuIndex:
      return PhysicalConsumerClass::kCpuIndexStream;
    case CensusSeamId::kVertexCpuTessFactor:
      return PhysicalConsumerClass::kCpuTessFactorStream;
    case CensusSeamId::kTextureSharedDirectLoad:
      return PhysicalConsumerClass::kSharedTextureLoad;
    case CensusSeamId::kTextureLegacyCpu:
      return PhysicalConsumerClass::kLegacyTextureCpuRead;
    case CensusSeamId::kTextureCacheHit:
      return PhysicalConsumerClass::kTextureCacheHit;
    case CensusSeamId::kTextureResolveSnapshot:
      return PhysicalConsumerClass::kResolveSnapshot;
    case CensusSeamId::kWaitPredicate:
      return PhysicalConsumerClass::kWaitPredicate;
    case CensusSeamId::kHostRead:
      return PhysicalConsumerClass::kHostRead;
    case CensusSeamId::kFrameEndPublication:
      return PhysicalConsumerClass::kFrameEndPublication;
    case CensusSeamId::kLifecyclePublication:
      return PhysicalConsumerClass::kLifecyclePublication;
    case CensusSeamId::kMetadataGuard:
    case CensusSeamId::kCount:
      return std::nullopt;  // no consumer class: reads no guest bytes
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Per-request balance (D2a). Multiplicities are compared PER REQUEST, before
// anything is aggregated.
//
// Per-seam totals cancel errors. Two requests on one seam:
//     request A: executed 2, observed 1
//     request B: executed 0, observed 1
//     aggregate: executed 2, observed 2   <-- passes, and both are wrong
// A missing observation in A is paid for by an unjustified observation in B,
// and the aggregate reports a clean partition. So each one-snapshot decision
// closes with its OWN equality, and only a balanced request contributes to the
// bounded totals.
// ---------------------------------------------------------------------------
struct CensusRequestBalance {
  CensusSeamId seam = CensusSeamId::kHostRead;
  // Emitted at INDEPENDENT locations per D2a: `executed` at the physical
  // operation, `observed` at the classifier. Removing the observation hook must
  // leave `executed` intact — if one marker wrote both, deleting it would drive
  // both to zero and this equality would pass on a seam nobody observed.
  uint64_t executed = 0;
  uint64_t observed = 0;

  bool balanced() const { return executed == observed; }
};

// Closes one request (defined below, after CensusVerdict). Returns the verdict
// for THAT request; the caller must not aggregate an unbalanced one.

// How a request ended. A legitimate early return between the dependency query
// and the physical read must say so EXPLICITLY; silently dropping the token is
// a named failure, not a quiet success.
enum class CensusCloseOutcome {
  kCompleted,            // ran through to the physical read
  kAbandonedBeforeRead,  // explicit early exit, nothing read
  // Observation already happened (classification runs in the dependency query)
  // and the draw was then canceled — e.g. a drain failure — so the physical
  // read never occurred. This is a REAL path at the vertex seam, so it is a
  // named non-aggregating outcome rather than a generic missing_seam.
  kCanceledAfterObservation,
};

// Named failures. Each names a DIFFERENT repair.
enum class CensusVerdict {
  kAccepted,
  kMissingSeam,          // a physical operation ran with no seam observing it
  kUnknownAttribution,   // an observation reached the census with no class
  kDuplicateOrOverlap,   // two observations claim the same bytes
  kSeamClassMismatch,    // a seam reported a class it is not allowed to report
  kUnionMismatch,        // classified ranges are not the requested ranges
  kHeartbeatMissing,     // the run produced no liveness signal at all
  kEngagementMissing,    // the run was alive but observed nothing
  kInvalidSeam,          // a seam id outside the declared set (incl. kCount)
  kUnjustifiedObservation, // observed > executed for one request
  kUnclosedRequest,      // a token was discarded without an explicit Close
  kDuplicateClose,       // a token was closed more than once
  kNoRequestsOpened,     // an engaged run enrolled no requests at all
  kAbandonedWithWork,    // closed as "no read performed" but work was recorded
  kCopiedRetiredMismatch,// retirements without the bytes that justify them
  kCanceledAfterObservation, // observed, then the draw was canceled before the
                             // physical read (a real control-flow path, not a
                             // defect); never aggregates
  kPostCloseEmission,    // a counter was emitted on an already-closed token
  kCanceledWithoutObservation, // canceled-after-observation, but nothing was
                               // observed -- the classifier hook is missing
  kRejectionTableMismatch, // per-verdict counters disagree with the count
  kDuplicateSnapshotSeam,  // the same (snapshot, seam) child opened twice
  kCount,                // sentinel; not a verdict
};

inline constexpr size_t kCensusVerdictCount = size_t(CensusVerdict::kCount);

// Slots that may legally hold a rejection. kAccepted and the cancellation
// outcome are NOT rejections; a count in either slot means the recording path
// diverged from the counting path.
inline constexpr bool IsRejectionVerdict(CensusVerdict verdict) {
  return verdict != CensusVerdict::kAccepted &&
         verdict != CensusVerdict::kCanceledAfterObservation &&
         verdict != CensusVerdict::kCount;
}

const char* CensusVerdictName(CensusVerdict verdict);

// One observation as the verifier sees it. Deliberately carries the RAW
// intervals rather than a byte total, because totals were exactly what hid the
// 2.3 defects.
struct CensusObservation {
  CensusSeamId seam = CensusSeamId::kHostRead;
  // OPTIONAL, because not every observed physical operation reads guest bytes.
  // The metadata guard has no consumer class at all: it inspects pending-record
  // metadata. Requiring a class for it made the seam unrepresentable — omitting
  // the observation failed missing_seam, and supplying one failed
  // seam_class_mismatch, so the guard could not be recorded either way.
  std::optional<PhysicalConsumerClass> reported_class;
  GuestRangeSet requested;
  std::vector<ClassifiedInterval> classified;
  // Retirements split by writer kind. A resolve retirement legitimately has NO
  // memexport memcpy behind it, so one combined counter cannot express the
  // invariant without either over-rejecting resolves or under-checking
  // memexports.
  uint64_t memexport_retirements = 0;
  uint64_t memexport_copies = 0;   // COUNT of copies, not bytes
  uint64_t resolve_retirements = 0;
  // Magnitude evidence only; never the invariant, because one byte can justify
  // many retirements.
  uint64_t bytes_copied = 0;
};

// Accounting emitted by the live ledger. Carried on the run so the verifier can
// reject a run that never enrolled anything, which is the difference between
// "closure is enforced" and "closure is available".
struct CensusLedgerStats {
  uint64_t opened = 0;
  uint64_t accepted = 0;   // closed and balanced
  uint64_t rejected = 0;   // closed and unbalanced
  uint64_t canceled = 0;   // closed as canceled-after-observation
  uint64_t duplicate_closes = 0;
  uint64_t silently_abandoned = 0;  // token destroyed while still open
  uint64_t post_close_emissions = 0;
  // Bounded (one slot per verdict, no per-request retention). Without this the
  // ledger kept only a rejection COUNT and the run report collapsed every
  // rejection to missing_seam, so unjustified_observation, invalid_seam and
  // abandoned_with_work could not survive into the live report.
  uint64_t rejected_by_verdict[kCensusVerdictCount] = {};

  uint64_t duplicate_snapshot_seams = 0;

  bool balances() const { return opened == accepted + rejected + canceled; }

  // The first recorded rejection verdict. Returns nullopt for "no information",
  // which is NOT the same as kAccepted: falling back to kAccepted turned an
  // inconsistent table (rejected=1, every counter zero) into a PASSING run.
  std::optional<CensusVerdict> FirstRejectionVerdict() const {
    for (size_t i = 0; i < kCensusVerdictCount; ++i) {
      if (rejected_by_verdict[i]) {
        return CensusVerdict(i);
      }
    }
    return std::nullopt;
  }

  uint64_t RejectionTableTotal() const {
    uint64_t total = 0;
    for (size_t i = 0; i < kCensusVerdictCount; ++i) {
      total += rejected_by_verdict[i];
    }
    return total;
  }
};

struct CensusRunSummary {
  CensusLedgerStats ledger;
  // Per-request balances, checked before aggregation (see
  // VerifyCensusRequestBalances). Bounded in the live census: a request that
  // balances contributes to totals and is not retained.
  std::vector<CensusRequestBalance> request_balances;
  std::vector<CensusObservation> observations;
  uint64_t heartbeats = 0;
  uint64_t unattributed_entries = 0;
  // Physical operations the run performed, whether or not a seam observed them.
  std::vector<CensusSeamId> executed_seams;
};

struct CensusVerifyResult {
  CensusVerdict verdict = CensusVerdict::kAccepted;
  std::string detail;
  bool accepted() const { return verdict == CensusVerdict::kAccepted; }
};

// Checks OVERLAP BEFORE NORMALIZATION. Normalizing first would merge a
// duplicated range into its neighbour and report a clean union — which is
// precisely how a double-counted observation hides.
inline bool CensusIntervalsOverlapBeforeNormalize(
    const std::vector<ClassifiedInterval>& intervals) {
  for (size_t i = 0; i < intervals.size(); ++i) {
    const uint64_t a_begin = intervals[i].range.start();
    const uint64_t a_end = a_begin + intervals[i].range.length();
    for (size_t j = i + 1; j < intervals.size(); ++j) {
      const uint64_t b_begin = intervals[j].range.start();
      const uint64_t b_end = b_begin + intervals[j].range.length();
      if (a_begin < b_end && b_begin < a_end) {
        return true;
      }
    }
  }
  return false;
}

// TRUE union equality: compares RANGES, not byte totals. This is the check the
// live path must use; IsExactPartition() compares two sums and cannot see a
// partition that is the wrong shape.
inline bool CensusUnionEqualsRequest(
    const std::vector<ClassifiedInterval>& intervals,
    const GuestRangeSet& requested) {
  GuestRangeSet rebuilt;
  for (const ClassifiedInterval& interval : intervals) {
    rebuilt.Add(interval.range);
  }
  rebuilt.Normalize();
  GuestRangeSet expected = requested;
  expected.Normalize();
  if (rebuilt.ranges().size() != expected.ranges().size()) {
    return false;
  }
  for (size_t i = 0; i < rebuilt.ranges().size(); ++i) {
    if (rebuilt.ranges()[i].start() != expected.ranges()[i].start() ||
        rebuilt.ranges()[i].length() != expected.ranges()[i].length()) {
      return false;
    }
  }
  return true;
}

inline CensusVerdict CloseCensusRequest(const CensusRequestBalance& balance);

// ---------------------------------------------------------------------------
// Token lifecycle: Begin(snapshot) -> independent emissions -> Close -> totals.
//
// The token COORDINATES the two counters; it never emits both. RecordExecuted
// is called from the physical-operation site and RecordObserved from the
// classifier, so deleting the observation hook leaves `executed` standing (D2a).
//
// The destructor is the enforcement: a token that goes out of scope unclosed —
// an early return between the dependency query and the physical read — records
// a silent abandonment that fails the run by name.
// ---------------------------------------------------------------------------
class CensusRequestLedger {
 public:
  static_assert(size_t(CensusSeamId::kCount) <= 32,
                "Snapshot::seam_mask_ is a uint32_t");
  class Token {
   public:
    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;
    Token(Token&& other) noexcept
        : ledger_(other.ledger_),
          request_id_(other.request_id_),
          balance_(other.balance_),
          closed_(other.closed_) {
      other.ledger_ = nullptr;  // moved-from tokens must not report abandonment
      other.closed_ = true;
    }
    // DELETED: move assignment could overwrite a still-open destination token,
    // silently losing its enrollment. Construction is the only transfer.
    Token& operator=(Token&&) = delete;
    ~Token() {
      if (ledger_ && !closed_) {
        ++ledger_->stats_.silently_abandoned;
      }
    }

    // Emitted at the physical operation. Independent of RecordObserved.
    void RecordExecuted(uint64_t count = 1) {
      if (SealedForEmission()) return;
      balance_.executed += count;
    }
    // Emitted at the classifier. Independent of RecordExecuted.
    void RecordObserved(uint64_t count = 1) {
      if (SealedForEmission()) return;
      balance_.observed += count;
    }

    uint64_t request_id() const { return request_id_; }

    const CensusRequestBalance& balance() const { return balance_; }
    bool closed() const { return closed_; }

   private:
    friend class CensusRequestLedger;
    Token(CensusRequestLedger* ledger, uint64_t request_id, CensusSeamId seam)
        : ledger_(ledger), request_id_(request_id) {
      balance_.seam = seam;
    }
    // A closed token is SEALED: a late emission is recorded as a lifecycle
    // failure rather than mutating a balance that has already been aggregated.
    bool SealedForEmission() {
      if (!closed_) return false;
      if (ledger_) ++ledger_->stats_.post_close_emissions;
      return true;
    }
    CensusRequestLedger* ledger_ = nullptr;
    uint64_t request_id_ = 0;
    CensusRequestBalance balance_;
    bool closed_ = false;
  };

  // One SNAPSHOT can fan out to several seams (the vertex three-way split), so
  // tokens are (snapshot, seam) children sharing one request identity. Each
  // child still closes on its own balance: one token carrying several seams
  // would let one seam's hole be covered by another seam's surplus.
  //
  // Children are opened THROUGH this object rather than by passing a raw id, so
  // an unissued snapshot id is unrepresentable rather than merely detected. The
  // seam mask makes a duplicate (snapshot, seam) child detectable; without it
  // two balanced children for one seam silently doubled that seam's totals.
  class Snapshot {
   public:
    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;
    Snapshot(Snapshot&& other) noexcept
        : ledger_(other.ledger_), id_(other.id_), seam_mask_(other.seam_mask_) {
      other.ledger_ = nullptr;
    }
    Snapshot& operator=(Snapshot&&) = delete;

    Token Begin(CensusSeamId seam) {
      const uint32_t bit =
          IsDeclaredSeam(seam) ? (uint32_t(1) << uint32_t(seam)) : 0;
      if (bit && (seam_mask_ & bit)) {
        ++ledger_->stats_.duplicate_snapshot_seams;
      }
      seam_mask_ |= bit;
      return ledger_->BeginChild(id_, seam);
    }

    uint64_t id() const { return id_; }

   private:
    friend class CensusRequestLedger;
    Snapshot(CensusRequestLedger* ledger, uint64_t id)
        : ledger_(ledger), id_(id) {}
    CensusRequestLedger* ledger_ = nullptr;
    uint64_t id_ = 0;
    uint32_t seam_mask_ = 0;
  };

  Snapshot BeginSnapshot() { return Snapshot(this, ++next_request_id_); }

  // Single-seam convenience: its own snapshot identity.
  Token Begin(CensusSeamId seam) { return BeginChild(++next_request_id_, seam); }

  // The ONLY path that contributes to the totals.
  CensusVerdict Close(Token& token, CensusCloseOutcome outcome) {
    if (token.closed_) {
      ++stats_.duplicate_closes;
      return CensusVerdict::kDuplicateClose;
    }
    token.closed_ = true;
    if (outcome == CensusCloseOutcome::kCanceledAfterObservation) {
      // Observation without a read is EXPECTED here: classification runs in the
      // dependency query and the draw can be canceled afterwards. Named,
      // counted, and deliberately non-aggregating -- it must not become part of
      // the executed/observed totals, and it is not a defect.
      if (token.balance_.executed) {
        // A read did happen, so "canceled before the read" is false.
        return RecordRejected(CensusVerdict::kAbandonedWithWork);
      }
      if (!token.balance_.observed) {
        // Cancellation is only meaningful AFTER an observation. Accepting the
        // 0/0 case let a canceled draw with its classifier hook removed pass.
        return RecordRejected(CensusVerdict::kCanceledWithoutObservation);
      }
      ++stats_.canceled;
      return CensusVerdict::kCanceledAfterObservation;
    }
    if (outcome == CensusCloseOutcome::kAbandonedBeforeRead &&
        (token.balance_.executed || token.balance_.observed)) {
      // "Nothing was read" and "work was recorded" cannot both be true.
      return RecordRejected(CensusVerdict::kAbandonedWithWork);
    }
    const CensusVerdict verdict = CloseCensusRequest(token.balance_);
    if (verdict == CensusVerdict::kAccepted) {
      ++stats_.accepted;
      executed_total_ += token.balance_.executed;
      observed_total_ += token.balance_.observed;
    } else {
      RecordRejected(verdict);
    }
    return verdict;
  }

  // The ONE place a rejection is recorded, so the count and the table cannot
  // drift apart.
  CensusVerdict RecordRejected(CensusVerdict verdict) {
    ++stats_.rejected;
    ++stats_.rejected_by_verdict[size_t(verdict)];
    return verdict;
  }

  const CensusLedgerStats& stats() const { return stats_; }
  uint64_t executed_total() const { return executed_total_; }
  uint64_t observed_total() const { return observed_total_; }

 private:
  Token BeginChild(uint64_t request_id, CensusSeamId seam) {
    ++stats_.opened;
    return Token(this, request_id, seam);
  }

  friend class Snapshot;
  CensusLedgerStats stats_;
  uint64_t next_request_id_ = 0;
  uint64_t executed_total_ = 0;
  uint64_t observed_total_ = 0;
};

inline CensusVerdict CloseCensusRequest(const CensusRequestBalance& balance) {
  if (!IsDeclaredSeam(balance.seam)) {
    return CensusVerdict::kInvalidSeam;
  }
  // BOTH directions. `executed > observed` means a physical operation ran
  // unobserved; `observed > executed` means the census invented an observation
  // for work that did not happen. Only one of the two is a missing hook, and
  // conflating them would send the repair to the wrong place.
  if (balance.executed > balance.observed) {
    return CensusVerdict::kMissingSeam;
  }
  if (balance.observed > balance.executed) {
    return CensusVerdict::kUnjustifiedObservation;
  }
  return CensusVerdict::kAccepted;
}

// Aggregation is only legal over balanced requests. Any unbalanced request
// fails HERE, before its counts can be summed against another request's
// opposite error.
inline CensusVerifyResult VerifyCensusRequestBalances(
    const std::vector<CensusRequestBalance>& balances) {
  CensusVerifyResult result;
  for (const CensusRequestBalance& balance : balances) {
    const CensusVerdict verdict = CloseCensusRequest(balance);
    if (verdict != CensusVerdict::kAccepted) {
      result.verdict = verdict;
      result.detail =
          std::string(CensusSeamIdName(balance.seam)) + " request unbalanced: executed=" +
          std::to_string(balance.executed) + " observed=" +
          std::to_string(balance.observed);
      return result;
    }
  }
  return result;
}

inline CensusVerifyResult VerifyCensusRun(const CensusRunSummary& run) {
  CensusVerifyResult result;
  auto fail = [&result](CensusVerdict verdict, std::string detail) {
    result.verdict = verdict;
    result.detail = std::move(detail);
    return result;
  };

  // Liveness first: a run with no heartbeat proves nothing, and every other
  // check below would pass vacuously on an empty run.
  if (!run.heartbeats) {
    return fail(CensusVerdict::kHeartbeatMissing,
                "no heartbeat: the run produced no liveness signal, so an "
                "empty census cannot be distinguished from one that never ran");
  }

  // Alive but blind: a run that observed nothing is not evidence of "nothing
  // to observe". Separate from the heartbeat so the two failures name
  // different repairs — the harness versus the seams.
  if (run.observations.empty() && run.executed_seams.empty()) {
    return fail(CensusVerdict::kEngagementMissing,
                "the run produced a heartbeat but observed no physical "
                "operation; an empty census is not a finding");
  }

  // The lifecycle is enforced BEFORE the balances are read, because an empty
  // `request_balances` is otherwise indistinguishable from "production never
  // enrolled a request" — absence read as permission, again. An engaged run
  // that opened nothing is a missing-enrollment failure, not a clean run.
  if (run.ledger.silently_abandoned) {
    return fail(CensusVerdict::kUnclosedRequest,
                std::to_string(run.ledger.silently_abandoned) +
                    " request token(s) discarded without an explicit Close");
  }
  if (run.ledger.duplicate_closes) {
    return fail(CensusVerdict::kDuplicateClose,
                std::to_string(run.ledger.duplicate_closes) +
                    " token(s) closed more than once");
  }
  if (!run.ledger.balances()) {
    return fail(CensusVerdict::kUnclosedRequest,
                "ledger does not balance: opened=" +
                    std::to_string(run.ledger.opened) + " accepted=" +
                    std::to_string(run.ledger.accepted) + " rejected=" +
                    std::to_string(run.ledger.rejected));
  }
  if (!run.ledger.opened) {
    return fail(CensusVerdict::kNoRequestsOpened,
                "run observed work but enrolled no requests; removing the "
                "Close path must not read as a clean run");
  }
  if (run.ledger.post_close_emissions) {
    return fail(CensusVerdict::kPostCloseEmission,
                std::to_string(run.ledger.post_close_emissions) +
                    " counter emission(s) on already-closed token(s)");
  }
  if (run.ledger.duplicate_snapshot_seams) {
    return fail(CensusVerdict::kDuplicateSnapshotSeam,
                std::to_string(run.ledger.duplicate_snapshot_seams) +
                    " duplicate (snapshot, seam) child token(s)");
  }
  // The rejection table is validated BEFORE it is read. An inconsistent table
  // is its own failure, never an acceptance.
  for (size_t i = 0; i < kCensusVerdictCount; ++i) {
    if (run.ledger.rejected_by_verdict[i] && !IsRejectionVerdict(CensusVerdict(i))) {
      return fail(CensusVerdict::kRejectionTableMismatch,
                  std::string("rejection counted in the non-rejection slot ") +
                      CensusVerdictName(CensusVerdict(i)));
    }
  }
  if (run.ledger.RejectionTableTotal() != run.ledger.rejected) {
    return fail(CensusVerdict::kRejectionTableMismatch,
                "rejection counters sum to " +
                    std::to_string(run.ledger.RejectionTableTotal()) +
                    " but rejected=" + std::to_string(run.ledger.rejected));
  }
  if (run.ledger.rejected) {
    // The REAL verdict, not a stand-in. Collapsing these to missing_seam sent
    // every rejection to the wrong repair.
    const std::optional<CensusVerdict> actual = run.ledger.FirstRejectionVerdict();
    if (!actual) {
      return fail(CensusVerdict::kRejectionTableMismatch,
                  "rejections recorded with no verdict attributed");
    }
    return fail(*actual, std::to_string(run.ledger.rejected) +
                             " request(s) closed with " +
                             CensusVerdictName(*actual));
  }

  // Per-request balance BEFORE anything aggregated. A run whose per-seam totals
  // agree can still contain a missing observation in one request cancelled by
  // an unjustified one in another.
  {
    const CensusVerifyResult balance_result =
        VerifyCensusRequestBalances(run.request_balances);
    if (!balance_result.accepted()) {
      return balance_result;
    }
  }

  if (run.unattributed_entries) {
    return fail(CensusVerdict::kUnknownAttribution,
                "an observation reached the census with no seam class");
  }

  // Seam ids are validated BEFORE anything is inferred from them. Previously a
  // seam whose expected class was nullopt was treated as a valid classless
  // seam, which admitted the kCount sentinel and any out-of-range id — a run
  // consisting entirely of a bogus seam was ACCEPTED.
  for (CensusSeamId executed : run.executed_seams) {
    if (!IsDeclaredSeam(executed)) {
      return fail(CensusVerdict::kInvalidSeam,
                  "executed seam id is not a declared seam");
    }
  }
  for (const CensusObservation& observation : run.observations) {
    if (!IsDeclaredSeam(observation.seam)) {
      return fail(CensusVerdict::kInvalidSeam,
                  "observation carries a seam id that is not a declared seam");
    }
  }

  // Every physical operation the run performed must have been observed.
  for (CensusSeamId executed : run.executed_seams) {
    bool observed = false;
    for (const CensusObservation& observation : run.observations) {
      if (observation.seam == executed) {
        observed = true;
        break;
      }
    }
    if (!observed) {
      return fail(CensusVerdict::kMissingSeam,
                  std::string("physical operation ran unobserved: ") +
                      CensusSeamIdName(executed));
    }
  }

  for (const CensusObservation& observation : run.observations) {
    const std::string where = CensusSeamIdName(observation.seam);

    // A seam may only report the class it is defined to be.
    const std::optional<PhysicalConsumerClass> expected =
        ExpectedClassForSeam(observation.seam);
    if (!expected && !IsClasslessSeam(observation.seam)) {
      // A declared seam with no expected class that is not on the classless
      // allow-list is a table gap, not a licence to skip validation.
      return fail(CensusVerdict::kSeamClassMismatch,
                  where + " has no declared expected class and is not a "
                          "permitted classless seam");
    }
    if (!expected) {
      // A CLASSLESS seam (the metadata guard). It must carry no class and no
      // bytes: it reads record metadata only, and letting it carry a range
      // would fold a zero-byte guard into a corridor's byte accounting — which
      // task 1.3 established is a content effect, not convergence cost.
      if (observation.reported_class) {
        return fail(CensusVerdict::kSeamClassMismatch,
                    where + " is classless but reported " +
                        PhysicalConsumerClassName(*observation.reported_class));
      }
      if (!observation.requested.ranges().empty() ||
          !observation.classified.empty()) {
        return fail(CensusVerdict::kSeamClassMismatch,
                    where + " is a zero-byte guard but carried ranges");
      }
      // EVERY accounting field must be zero. The classless branch previously
      // `continue`d before the copy/retirement checks, so a guard reporting
      // copies, retirements or bytes was accepted — a zero-byte guard would
      // have contributed convergence accounting it cannot possibly have
      // performed. Checked here rather than by falling through, because the
      // invariant is "all zero", not the memexport count equality below.
      if (observation.memexport_copies || observation.memexport_retirements ||
          observation.resolve_retirements || observation.bytes_copied) {
        return fail(CensusVerdict::kCopiedRetiredMismatch,
                    where + " is a zero-byte guard but reported accounting: "
                            "copies=" +
                        std::to_string(observation.memexport_copies) +
                        " memexport_retirements=" +
                        std::to_string(observation.memexport_retirements) +
                        " resolve_retirements=" +
                        std::to_string(observation.resolve_retirements) +
                        " bytes=" + std::to_string(observation.bytes_copied));
      }
      continue;  // a validated classless observation carries nothing else
    }
    if (!observation.reported_class) {
      return fail(CensusVerdict::kSeamClassMismatch,
                  where + " reported no class but one is required");
    }
    if (*observation.reported_class != *expected) {
      return fail(CensusVerdict::kSeamClassMismatch,
                  where + " reported " +
                      PhysicalConsumerClassName(*observation.reported_class) +
                      ", expected " + PhysicalConsumerClassName(*expected));
    }

    if (CensusIntervalsOverlapBeforeNormalize(observation.classified)) {
      return fail(CensusVerdict::kDuplicateOrOverlap,
                  where + " has overlapping classified intervals");
    }

    if (!CensusUnionEqualsRequest(observation.classified,
                                  observation.requested)) {
      return fail(CensusVerdict::kUnionMismatch,
                  where + " classified ranges are not the requested ranges");
    }

    // EXACT invariant: every retired memexport record must correspond to one
    // completed copy. The previous rule was `retirements && !bytes_copied`,
    // which accepted ten retirements behind a single byte AND falsely rejected
    // a legitimate resolve retirement that has no memexport memcpy at all.
    // Byte count is retained as magnitude evidence, not as the invariant.
    if (observation.memexport_copies != observation.memexport_retirements) {
      return fail(CensusVerdict::kCopiedRetiredMismatch,
                  where + " retired " +
                      std::to_string(observation.memexport_retirements) +
                      " memexport records against " +
                      std::to_string(observation.memexport_copies) +
                      " completed copies");
    }
    if (observation.memexport_copies && !observation.bytes_copied) {
      return fail(CensusVerdict::kCopiedRetiredMismatch,
                  where + " reported completed copies of zero bytes");
    }
  }
  return result;
}

inline const char* CensusSeamIdName(CensusSeamId seam) {
  switch (seam) {
    case CensusSeamId::kVertexSharedVfetch: return "vertex/shared-vfetch";
    case CensusSeamId::kVertexCpuIndex: return "vertex/cpu-index";
    case CensusSeamId::kVertexCpuTessFactor: return "vertex/cpu-tess-factor";
    case CensusSeamId::kTextureSharedDirectLoad: return "texture/shared-direct";
    case CensusSeamId::kTextureLegacyCpu: return "texture/legacy-cpu";
    case CensusSeamId::kTextureCacheHit: return "texture/cache-hit";
    case CensusSeamId::kTextureResolveSnapshot:
      return "texture/resolve-snapshot";
    case CensusSeamId::kWaitPredicate: return "wait-predicate";
    case CensusSeamId::kHostRead: return "host-read";
    case CensusSeamId::kFrameEndPublication: return "frame-end-publication";
    case CensusSeamId::kLifecyclePublication: return "lifecycle-publication";
    case CensusSeamId::kMetadataGuard: return "metadata-guard";
    default: return "unknown-seam";
  }
}

inline const char* CensusVerdictName(CensusVerdict verdict) {
  switch (verdict) {
    case CensusVerdict::kAccepted: return "accepted";
    case CensusVerdict::kMissingSeam: return "missing_seam";
    case CensusVerdict::kUnknownAttribution: return "unknown_attribution";
    case CensusVerdict::kDuplicateOrOverlap: return "duplicate_or_overlap";
    case CensusVerdict::kSeamClassMismatch: return "seam_class_mismatch";
    case CensusVerdict::kUnionMismatch: return "union_mismatch";
    case CensusVerdict::kHeartbeatMissing: return "heartbeat_missing";
    case CensusVerdict::kEngagementMissing: return "engagement_missing";
    case CensusVerdict::kInvalidSeam: return "invalid_seam";
    case CensusVerdict::kUnjustifiedObservation:
      return "unjustified_observation";
    case CensusVerdict::kUnclosedRequest: return "unclosed_request";
    case CensusVerdict::kDuplicateClose: return "duplicate_close";
    case CensusVerdict::kNoRequestsOpened: return "no_requests_opened";
    case CensusVerdict::kAbandonedWithWork: return "abandoned_with_work";
    case CensusVerdict::kCanceledAfterObservation:
      return "canceled_after_observation";
    case CensusVerdict::kPostCloseEmission: return "post_close_emission";
    case CensusVerdict::kCanceledWithoutObservation:
      return "canceled_without_observation";
    case CensusVerdict::kRejectionTableMismatch:
      return "rejection_table_mismatch";
    case CensusVerdict::kDuplicateSnapshotSeam:
      return "duplicate_snapshot_seam";
    case CensusVerdict::kCount: return "invalid(count)";
    case CensusVerdict::kCopiedRetiredMismatch:
      return "copied_retired_mismatch";
    default: return "unknown";
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_VERIFY_H_
