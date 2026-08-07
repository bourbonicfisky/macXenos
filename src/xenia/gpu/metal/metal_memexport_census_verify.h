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
  kCopiedRetiredMismatch,// retirements without the bytes that justify them
};

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

struct CensusRunSummary {
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
    case CensusVerdict::kCopiedRetiredMismatch:
      return "copied_retired_mismatch";
    default: return "unknown";
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_VERIFY_H_
