/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MEMEXPORT_CONSUMER_CLASSIFIER_H_
#define XENIA_GPU_METAL_METAL_MEMEXPORT_CONSUMER_CLASSIFIER_H_

// Pure transient interval classifier for the memexport GPU-consumer pre-gate
// (task 2.1). Latest-writer authority per interval, plus the physical consumer
// class the pre-gate's partition requires.
//
// PURE AND TRANSIENT, BY CONSTRUCTION:
//   * owns NO persistent authority state — every call takes the producer list
//     and the request as inputs and returns a value; nothing is remembered
//     between calls, and nothing here is a cache;
//   * alters NO production request and NO control flow — it takes const inputs
//     and returns a description. It cannot retire a record, converge a range,
//     issue a commit, or change what a draw does. The census (2.3) observes
//     with it; production does not consult it.
//
// WHY A SEPARATE CLASSIFIER RATHER THAN READING ConsumerKind. Task 1.3
// established that the existing labels do not partition physical readers:
// kVertex covers both shared-buffer vfetch and CPU-staged index/factor streams,
// kHostCpuRead covers true host reads AND two different publication boundaries,
// and kTexture covers a shared-buffer compute load, two legacy CPU paths and a
// zero-byte cache hit. A partition built from those labels would look exact and
// be wrong, so the physical class is modelled here explicitly.
//
// WHAT IT DELIBERATELY DOES NOT DO. It does not decide whether a direct GPU
// read is SAFE. Eligibility here means "the latest writer over these bytes is a
// pending memexport in the shared buffer, and this consumer reads that buffer".
// Ordering (task 1.5) and the byte/lifetime oracle (group 3) are separate and
// neither is implied by an eligible verdict.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "xenia/gpu/metal/metal_guest_range.h"
#include "xenia/gpu/metal/metal_memory_authority.h"

namespace xe {
namespace gpu {
namespace metal {

// The physical reader, as established by task 1.3. This is an INPUT — which
// code is about to read the bytes — never something inferred from a range.
enum class PhysicalConsumerClass {
  // Reads the shared Metal buffer on the GPU.
  kSharedVfetch,         // translated vertex fetch
  kSharedTextureLoad,    // strict-RGBA8 direct load compute dispatch
  // Reads guest RAM on the CPU.
  kCpuIndexStream,       // guest index stream staged at encode time
  kCpuTessFactorStream,  // tessellation factor/index stream staged at encode
  kLegacyTextureCpuRead, // legacy texture fallback (base level or mip chain)
  kWaitPredicate,        // WAIT_REG_MEM / memory predicate poll
  kHostRead,             // host/trace/debug/save read
  // Reads no guest bytes at all.
  kTextureCacheHit,
  // Boundaries. Mandatory, and outside the avoidable-opportunity denominator.
  kFrameEndPublication,
  kLifecyclePublication,
};

// Reads the shared Metal buffer, so it CAN observe GPU-newer bytes without RAM
// convergence. Necessary for GPU-direct eligibility, never sufficient.
constexpr bool ConsumerReadsSharedBuffer(PhysicalConsumerClass consumer) {
  return consumer == PhysicalConsumerClass::kSharedVfetch ||
         consumer == PhysicalConsumerClass::kSharedTextureLoad;
}

// Reads guest RAM on the CPU, so it REQUIRES convergence. The spec forbids
// classifying any of these as GPU-direct.
constexpr bool ConsumerRequiresRam(PhysicalConsumerClass consumer) {
  switch (consumer) {
    case PhysicalConsumerClass::kCpuIndexStream:
    case PhysicalConsumerClass::kCpuTessFactorStream:
    case PhysicalConsumerClass::kLegacyTextureCpuRead:
    case PhysicalConsumerClass::kWaitPredicate:
    case PhysicalConsumerClass::kHostRead:
    case PhysicalConsumerClass::kFrameEndPublication:
    case PhysicalConsumerClass::kLifecyclePublication:
      return true;
    default:
      return false;
  }
}

// A publication boundary rather than a consumer. Counted, never counted as
// avoidable.
constexpr bool ConsumerIsPublicationBoundary(PhysicalConsumerClass consumer) {
  return consumer == PhysicalConsumerClass::kFrameEndPublication ||
         consumer == PhysicalConsumerClass::kLifecyclePublication;
}

// Reads zero guest bytes. Must still appear in the partition, with a zero
// range, or the consult tally understates what was examined.
constexpr bool ConsumerReadsNoGuestBytes(PhysicalConsumerClass consumer) {
  return consumer == PhysicalConsumerClass::kTextureCacheHit;
}

const char* PhysicalConsumerClassName(PhysicalConsumerClass consumer);

// The latest pending writer over one interval. "Latest" is decided by
// `sequence` — chronological registration order — and NEVER by writer kind, so
// memexport-then-resolve and resolve-then-memexport both resolve correctly.
enum class IntervalAuthority {
  kUncovered,             // no pending producer covers these bytes
  kLatestPendingMemexport,  // latest covering writer is a pending memexport
  kLatestPendingResolve,    // latest covering writer is a pending resolve
};

const char* IntervalAuthorityName(IntervalAuthority authority);

struct ClassifiedInterval {
  // GuestRange has no default constructor by design (an invalid range must be
  // unrepresentable), so an interval is always built from a validated range.
  GuestRange range = *GuestRange::FromBaseLength(0, 1);
  IntervalAuthority authority = IntervalAuthority::kUncovered;
  // Sequence of the latest covering writer; 0 when uncovered.
  uint64_t writer_sequence = 0;

  bool operator==(const ClassifiedInterval& other) const {
    return range.start() == other.range.start() &&
           range.length() == other.range.length() &&
           authority == other.authority &&
           writer_sequence == other.writer_sequence;
  }
};

// The result: a set of disjoint intervals in ascending address order that
// EXACTLY partitions the requested bytes. Exactness is carried as data rather
// than assumed, because the pre-gate's requirement is "zero unknown,
// double-counted, or silently omitted entries" and a partition that merely
// looks complete is the failure mode.
struct IntervalClassification {
  std::vector<ClassifiedInterval> intervals;
  uint64_t requested_bytes = 0;
  uint64_t classified_bytes = 0;

  bool IsExactPartition() const {
    return requested_bytes == classified_bytes;
  }
  uint64_t BytesWithAuthority(IntervalAuthority authority) const {
    uint64_t total = 0;
    for (const ClassifiedInterval& interval : intervals) {
      if (interval.authority == authority) {
        total += interval.range.length();
      }
    }
    return total;
  }
};

// Latest-writer classification over `request_ranges`, given the pending
// producers. Producers whose state is neither pending-resolve nor
// pending-memexport are ignored; both are considered, because supersession must
// work in both directions.
//
// Complexity is O(boundaries * producers), with both bounded by the pending set
// and the request — this runs in a census, not on a frame path.
inline IntervalClassification ClassifyLatestWriterIntervals(
    const GuestRangeSet& request_ranges,
    const std::vector<PendingProducer>& producers) {
  IntervalClassification result;

  GuestRangeSet normalized = request_ranges;
  normalized.Normalize();
  for (const GuestRange& range : normalized.ranges()) {
    result.requested_bytes += range.length();
  }
  if (normalized.ranges().empty()) {
    // Empty input is an exact partition of nothing, not an error.
    return result;
  }

  // Boundary sweep: every producer edge that falls strictly inside a requested
  // range can split it. Collected per requested range so a producer far away
  // cannot fragment an unrelated interval.
  for (const GuestRange& requested : normalized.ranges()) {
    // Bounds captured BEFORE the loop: comparing against cuts.front()/back()
    // while pushing into `cuts` compares each producer edge against the
    // previously pushed edge instead of the request, which silently drops every
    // cut after the first and collapses the whole request into one interval.
    const uint64_t request_begin = requested.start();
    const uint64_t request_end =
        uint64_t(requested.start()) + requested.length();
    std::vector<uint64_t> cuts;
    cuts.push_back(request_begin);
    cuts.push_back(request_end);
    for (const PendingProducer& producer : producers) {
      const uint64_t producer_begin = producer.range.start();
      const uint64_t producer_end =
          uint64_t(producer.range.start()) + producer.range.length();
      if (producer_begin > request_begin && producer_begin < request_end) {
        cuts.push_back(producer_begin);
      }
      if (producer_end > request_begin && producer_end < request_end) {
        cuts.push_back(producer_end);
      }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    for (size_t i = 0; i + 1 < cuts.size(); ++i) {
      const uint64_t begin = cuts[i];
      const uint64_t end = cuts[i + 1];
      if (end <= begin) {
        continue;
      }
      // Latest covering writer over [begin, end) — strictly by sequence.
      const PendingProducer* latest = nullptr;
      for (const PendingProducer& producer : producers) {
        if (producer.state != TransitState::kPendingResolve &&
            producer.state != TransitState::kPendingMemexport) {
          continue;
        }
        const uint64_t producer_begin = producer.range.start();
        const uint64_t producer_end =
            uint64_t(producer.range.start()) + producer.range.length();
        if (producer_begin > begin || producer_end < end) {
          continue;  // does not cover the whole elementary interval
        }
        if (!latest || producer.sequence > latest->sequence) {
          latest = &producer;
        }
      }

      ClassifiedInterval interval;
      std::optional<GuestRange> range = GuestRange::FromBounds(begin, end);
      if (!range) {
        continue;  // saturated away entirely; contributes no bytes
      }
      interval.range = *range;
      if (latest) {
        interval.authority =
            latest->state == TransitState::kPendingMemexport
                ? IntervalAuthority::kLatestPendingMemexport
                : IntervalAuthority::kLatestPendingResolve;
        interval.writer_sequence = latest->sequence;
      }

      // Coalesce with the previous interval when the authority AND the writer
      // are identical and the ranges abut. Two adjacent producers with
      // different sequences stay separate: they are different writers even if
      // the verdict happens to match.
      if (!result.intervals.empty()) {
        ClassifiedInterval& previous = result.intervals.back();
        const uint64_t previous_end =
            uint64_t(previous.range.start()) + previous.range.length();
        if (previous_end == interval.range.start() &&
            previous.authority == interval.authority &&
            previous.writer_sequence == interval.writer_sequence) {
          std::optional<GuestRange> merged = GuestRange::FromBounds(
              uint64_t(previous.range.start()),
              uint64_t(interval.range.start()) + interval.range.length());
          if (merged) {
            previous.range = *merged;
            result.classified_bytes += interval.range.length();
            continue;
          }
        }
      }
      result.classified_bytes += interval.range.length();
      result.intervals.push_back(interval);
    }
  }
  return result;
}

// An interval is GPU-direct eligible only when EVERY byte of it has a latest
// covering writer that is a pending memexport backed by the shared buffer, AND
// the consumer physically reads that buffer.
//
// Deliberately written as a conjunction of positives: an interval with no
// covering writer, mixed ownership (which the classifier has already split),
// or a newer resolve is not eligible, and a CPU consumer is never eligible no
// matter what the authority says.
constexpr bool IntervalIsGpuDirectEligible(const ClassifiedInterval& interval,
                                           PhysicalConsumerClass consumer) {
  return interval.authority == IntervalAuthority::kLatestPendingMemexport &&
         ConsumerReadsSharedBuffer(consumer);
}

// Total eligible bytes for one consumer over a classification. The complement
// (requested minus eligible) is what would still need convergence, and the two
// must sum to the request — which `IsExactPartition` is what proves.
inline uint64_t GpuDirectEligibleBytes(
    const IntervalClassification& classification,
    PhysicalConsumerClass consumer) {
  uint64_t total = 0;
  for (const ClassifiedInterval& interval : classification.intervals) {
    if (IntervalIsGpuDirectEligible(interval, consumer)) {
      total += interval.range.length();
    }
  }
  return total;
}

inline const char* PhysicalConsumerClassNameImpl(
    PhysicalConsumerClass consumer) {
  switch (consumer) {
    case PhysicalConsumerClass::kSharedVfetch:
      return "shared-vfetch";
    case PhysicalConsumerClass::kSharedTextureLoad:
      return "shared-texture-load";
    case PhysicalConsumerClass::kCpuIndexStream:
      return "cpu-index-stream";
    case PhysicalConsumerClass::kCpuTessFactorStream:
      return "cpu-tess-factor-stream";
    case PhysicalConsumerClass::kLegacyTextureCpuRead:
      return "legacy-texture-cpu-read";
    case PhysicalConsumerClass::kWaitPredicate:
      return "wait-predicate";
    case PhysicalConsumerClass::kHostRead:
      return "host-read";
    case PhysicalConsumerClass::kTextureCacheHit:
      return "texture-cache-hit";
    case PhysicalConsumerClass::kFrameEndPublication:
      return "frame-end-publication";
    case PhysicalConsumerClass::kLifecyclePublication:
      return "lifecycle-publication";
  }
  return "unknown";
}

inline const char* PhysicalConsumerClassName(PhysicalConsumerClass consumer) {
  return PhysicalConsumerClassNameImpl(consumer);
}

inline const char* IntervalAuthorityName(IntervalAuthority authority) {
  switch (authority) {
    case IntervalAuthority::kUncovered:
      return "uncovered";
    case IntervalAuthority::kLatestPendingMemexport:
      return "latest-pending-memexport";
    case IntervalAuthority::kLatestPendingResolve:
      return "latest-pending-resolve";
  }
  return "unknown";
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_CONSUMER_CLASSIFIER_H_
