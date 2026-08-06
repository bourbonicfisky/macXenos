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
constexpr PhysicalConsumerClass ExpectedClassForSeam(CensusSeamId seam) {
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
    case CensusSeamId::kWaitPredicate:
      return PhysicalConsumerClass::kWaitPredicate;
    case CensusSeamId::kHostRead:
      return PhysicalConsumerClass::kHostRead;
    case CensusSeamId::kFrameEndPublication:
      return PhysicalConsumerClass::kFrameEndPublication;
    case CensusSeamId::kLifecyclePublication:
      return PhysicalConsumerClass::kLifecyclePublication;
    default:
      return PhysicalConsumerClass::kHostRead;
  }
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
  kCopiedRetiredMismatch,// retirements without the bytes that justify them
};

const char* CensusVerdictName(CensusVerdict verdict);

// One observation as the verifier sees it. Deliberately carries the RAW
// intervals rather than a byte total, because totals were exactly what hid the
// 2.3 defects.
struct CensusObservation {
  CensusSeamId seam = CensusSeamId::kHostRead;
  PhysicalConsumerClass reported_class = PhysicalConsumerClass::kHostRead;
  GuestRangeSet requested;
  std::vector<ClassifiedInterval> classified;
  uint64_t retirements = 0;
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

  if (run.unattributed_entries) {
    return fail(CensusVerdict::kUnknownAttribution,
                "an observation reached the census with no seam class");
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
    if (observation.reported_class != ExpectedClassForSeam(observation.seam)) {
      return fail(CensusVerdict::kSeamClassMismatch,
                  where + " reported " +
                      PhysicalConsumerClassName(observation.reported_class) +
                      ", expected " +
                      PhysicalConsumerClassName(
                          ExpectedClassForSeam(observation.seam)));
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

    // Retirements must be justified by copied bytes. A retirement with no copy
    // is a record dropped without its data reaching RAM.
    if (observation.retirements && !observation.bytes_copied) {
      return fail(CensusVerdict::kCopiedRetiredMismatch,
                  where + " retired records without copying any bytes");
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
    case CensusVerdict::kCopiedRetiredMismatch:
      return "copied_retired_mismatch";
    default: return "unknown";
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_VERIFY_H_
