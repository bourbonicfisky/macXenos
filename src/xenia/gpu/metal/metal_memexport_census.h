/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_H_
#define XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_H_

// TEMPORARY observational census for the memexport GPU-consumer pre-gate
// (task 2.3). Exact-off, env-gated (`XE_METAL_MEMEXPORT_CENSUS=1`), removed at
// closeout.
//
// IT CHANGES NO CONTROL FLOW. Every entry point records and returns void. It
// never decides whether to converge, commit, wait, drain, retire or skip; it
// observes what production already decided. The classification it computes
// (via the pure task-2.1 classifier) is written to counters and read by nobody
// in production.
//
// THE CLASS COMES FROM THE CALL SITE, NEVER FROM ConsumerKind. Task 1.3
// established that the existing labels do not partition physical readers, so a
// census keyed on them would report a complete partition that is wrong. Each
// production seam opens a scope naming what it physically is.
//
// WHAT IT MUST NOT COLLAPSE. Task 1.4 established four separations that a
// single "commit cost" number destroys, and each is a distinct counter here:
//   * consultations vs commits — a selected-empty decision costs no GPU
//     boundary, and a non-empty one costs a commit only when a frame command
//     buffer existed;
//   * boundary wait vs required-submission wait, each attempted AND blocked —
//     neither implies the other, and a wait that did not block is not a stall;
//   * successful vs failed drains — retirement must equal successful copies,
//     never attempts;
//   * four invalidation ORIGINS — the GPU-write publication watch fires before
//     any convergence, so attributing it to convergence would credit this
//     corridor with work it does not do.
//
// EXACTNESS IS ASSERTED ON THE INTERVAL UNION, NOT ON BYTE TOTALS. Task 2.2
// demonstrated that a wrong partition can keep byte totals self-consistent, so
// the census compares the classified union against the production request and
// counts any mismatch rather than trusting a sum.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/gpu/metal/metal_guest_range.h"
#include "xenia/gpu/metal/metal_memexport_consumer_classifier.h"

namespace xe {
namespace gpu {
namespace metal {

// Where an invalidation came from (task 1.4 §4). Kept separate because the
// earliest source is not convergence at all.
enum class InvalidationOrigin {
  kPublicationWatch,      // RangeWrittenByGpu at producer publication
  kMidFrameConvergence,   // a consumer forced a drain
  kFrameEndConvergence,   // swap publication
  kCacheClearConvergence, // lifecycle publication
  kCount,
};

inline const char* InvalidationOriginName(InvalidationOrigin origin) {
  switch (origin) {
    case InvalidationOrigin::kPublicationWatch:
      return "publication-watch";
    case InvalidationOrigin::kMidFrameConvergence:
      return "mid-frame-convergence";
    case InvalidationOrigin::kFrameEndConvergence:
      return "frame-end-convergence";
    case InvalidationOrigin::kCacheClearConvergence:
      return "cache-clear-convergence";
    default:
      return "unknown";
  }
}

inline constexpr size_t kMemexportCensusClassCount = 11;

// Per-physical-corridor accumulators. Every field is an ACTUAL observation, not
// an inference from another field.
struct MemexportCensusCorridor {
  std::atomic<uint64_t> consultations{0};       // every seam entry, incl. empty
  std::atomic<uint64_t> zero_byte_consultations{0};
  std::atomic<uint64_t> selected_empty{0};      // no GPU boundary at all
  std::atomic<uint64_t> requested_bytes{0};
  std::atomic<uint64_t> classified_bytes{0};
  std::atomic<uint64_t> eligible_bytes{0};      // GPU-direct eligible
  std::atomic<uint64_t> selected_records{0};

  std::atomic<uint64_t> had_frame_command_buffer{0};
  std::atomic<uint64_t> commits{0};             // measured delta, never inferred

  std::atomic<uint64_t> boundary_wait_attempted{0};
  std::atomic<uint64_t> boundary_wait_blocked{0};
  std::atomic<uint64_t> required_wait_attempted{0};
  std::atomic<uint64_t> required_wait_blocked{0};

  std::atomic<uint64_t> drains_succeeded{0};
  std::atomic<uint64_t> drains_failed{0};
  std::atomic<uint64_t> bytes_copied{0};
  std::atomic<uint64_t> retirements{0};

  // Both must stay 0. Nonzero means the partition is not exact and no verdict
  // may be drawn from this run.
  std::atomic<uint64_t> unknown_attributions{0};
  std::atomic<uint64_t> union_mismatches{0};
};

class MetalMemexportCensus {
 public:
  static bool Enabled() {
    static const bool enabled = [] {
      const char* value = std::getenv("XE_METAL_MEMEXPORT_CENSUS");
      return value && value[0] == '1';
    }();
    return enabled;
  }

  MemexportCensusCorridor& Corridor(PhysicalConsumerClass consumer) {
    return corridors_[size_t(consumer)];
  }

  void NoteInvalidation(InvalidationOrigin origin, uint64_t entries) {
    invalidations_[size_t(origin)].fetch_add(entries, std::memory_order_relaxed);
    invalidation_events_[size_t(origin)].fetch_add(1,
                                                   std::memory_order_relaxed);
  }

  // The legacy `:5629` staleness guard: reads record METADATA, no guest bytes,
  // adds no commit or wait, but CAN skip a draw. Recorded so it is neither
  // silently omitted nor counted as convergence cost.
  void NoteMetadataGuardConsult(bool skipped_draw) {
    metadata_guard_consults_.fetch_add(1, std::memory_order_relaxed);
    if (skipped_draw) {
      metadata_guard_draw_skips_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void NoteHeartbeat() {
    heartbeats_.fetch_add(1, std::memory_order_relaxed);
  }

  // Engagement: any corridor that consulted at all. A census reporting zero
  // engagement proves nothing about the workload and must not be read as
  // "no opportunity".
  bool AnyEngagement() const {
    for (const MemexportCensusCorridor& corridor : corridors_) {
      if (corridor.consultations.load(std::memory_order_relaxed)) {
        return true;
      }
    }
    return metadata_guard_consults_.load(std::memory_order_relaxed) != 0;
  }

  uint64_t TotalUnknowns() const {
    uint64_t total = 0;
    for (const MemexportCensusCorridor& corridor : corridors_) {
      total += corridor.unknown_attributions.load(std::memory_order_relaxed) +
               corridor.union_mismatches.load(std::memory_order_relaxed);
    }
    return total;
  }

  std::string Report(const char* phase) const {
    std::string out = fmt::format(
        "MemexportCensus|phase={}|heartbeats={}|engaged={}|unknowns={}",
        phase, heartbeats_.load(std::memory_order_relaxed),
        AnyEngagement() ? 1 : 0, TotalUnknowns());
    for (size_t i = 0; i < kMemexportCensusClassCount; ++i) {
      const MemexportCensusCorridor& c = corridors_[i];
      const uint64_t consults = c.consultations.load(std::memory_order_relaxed);
      if (!consults) {
        continue;  // a corridor never reached adds no row, but AnyEngagement
                   // above still distinguishes "nothing ran" from "nothing to do"
      }
      out += fmt::format(
          "\n  {}|consults={}|zero_byte={}|selected_empty={}|sel_records={}"
          "|req_bytes={}|cls_bytes={}|eligible_bytes={}"
          "|had_buf={}|commits={}"
          "|boundary_wait={}/{}|required_wait={}/{}"
          "|drain_ok={}|drain_fail={}|copied={}|retired={}"
          "|unknown={}|union_mismatch={}",
          PhysicalConsumerClassName(PhysicalConsumerClass(i)), consults,
          c.zero_byte_consultations.load(std::memory_order_relaxed),
          c.selected_empty.load(std::memory_order_relaxed),
          c.selected_records.load(std::memory_order_relaxed),
          c.requested_bytes.load(std::memory_order_relaxed),
          c.classified_bytes.load(std::memory_order_relaxed),
          c.eligible_bytes.load(std::memory_order_relaxed),
          c.had_frame_command_buffer.load(std::memory_order_relaxed),
          c.commits.load(std::memory_order_relaxed),
          c.boundary_wait_blocked.load(std::memory_order_relaxed),
          c.boundary_wait_attempted.load(std::memory_order_relaxed),
          c.required_wait_blocked.load(std::memory_order_relaxed),
          c.required_wait_attempted.load(std::memory_order_relaxed),
          c.drains_succeeded.load(std::memory_order_relaxed),
          c.drains_failed.load(std::memory_order_relaxed),
          c.bytes_copied.load(std::memory_order_relaxed),
          c.retirements.load(std::memory_order_relaxed),
          c.unknown_attributions.load(std::memory_order_relaxed),
          c.union_mismatches.load(std::memory_order_relaxed));
    }
    for (size_t i = 0; i < size_t(InvalidationOrigin::kCount); ++i) {
      out += fmt::format("\n  invalidation|{}|events={}|entries={}",
                         InvalidationOriginName(InvalidationOrigin(i)),
                         invalidation_events_[i].load(std::memory_order_relaxed),
                         invalidations_[i].load(std::memory_order_relaxed));
    }
    out += fmt::format(
        "\n  metadata_guard|consults={}|draw_skips={} (zero-byte; no commit, no "
        "wait; skips are a CONTENT effect, not convergence cost)",
        metadata_guard_consults_.load(std::memory_order_relaxed),
        metadata_guard_draw_skips_.load(std::memory_order_relaxed));
    return out;
  }

 private:
  MemexportCensusCorridor corridors_[kMemexportCensusClassCount];
  std::atomic<uint64_t> invalidations_[size_t(InvalidationOrigin::kCount)] = {};
  std::atomic<uint64_t>
      invalidation_events_[size_t(InvalidationOrigin::kCount)] = {};
  std::atomic<uint64_t> metadata_guard_consults_{0};
  std::atomic<uint64_t> metadata_guard_draw_skips_{0};
  std::atomic<uint64_t> heartbeats_{0};
};

inline MetalMemexportCensus& GetMetalMemexportCensus() {
  static MetalMemexportCensus census;
  return census;
}

// Opened at a production seam to name what that seam PHYSICALLY is. Nothing
// here reads ConsumerKind. The scope is inert unless the census is enabled.
//
// It records the classification against the production request and checks the
// classified union for equality with it — the exactness requirement — without
// influencing what production then does with that request.
class MetalMemexportCensusScope {
 public:
  MetalMemexportCensusScope(PhysicalConsumerClass consumer,
                            const GuestRangeSet& request,
                            const std::vector<PendingProducer>& producers)
      : consumer_(consumer) {
    if (!MetalMemexportCensus::Enabled()) {
      return;
    }
    active_ = true;
    MemexportCensusCorridor& corridor =
        GetMetalMemexportCensus().Corridor(consumer);
    corridor.consultations.fetch_add(1, std::memory_order_relaxed);

    const IntervalClassification classification =
        ClassifyLatestWriterIntervals(request, producers);
    corridor.requested_bytes.fetch_add(classification.requested_bytes,
                                       std::memory_order_relaxed);
    corridor.classified_bytes.fetch_add(classification.classified_bytes,
                                        std::memory_order_relaxed);
    corridor.eligible_bytes.fetch_add(
        GpuDirectEligibleBytes(classification, consumer),
        std::memory_order_relaxed);
    if (!classification.requested_bytes) {
      corridor.zero_byte_consultations.fetch_add(1, std::memory_order_relaxed);
    }
    // UNION equality, not a byte sum: task 2.2 showed a wrong partition can
    // keep totals self-consistent.
    if (!classification.IsExactPartition() ||
        !UnionMatchesRequest(classification, request)) {
      corridor.union_mismatches.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Recorded by the executor instrumentation; all observations, no decisions.
  void NoteSelected(size_t selected_records, bool selected_empty) {
    if (!active_) return;
    MemexportCensusCorridor& corridor =
        GetMetalMemexportCensus().Corridor(consumer_);
    corridor.selected_records.fetch_add(selected_records,
                                        std::memory_order_relaxed);
    if (selected_empty) {
      corridor.selected_empty.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void NoteBoundary(bool had_frame_command_buffer, uint64_t commit_delta,
                    bool boundary_wait_attempted, bool boundary_wait_blocked) {
    if (!active_) return;
    MemexportCensusCorridor& corridor =
        GetMetalMemexportCensus().Corridor(consumer_);
    if (had_frame_command_buffer) {
      corridor.had_frame_command_buffer.fetch_add(1,
                                                  std::memory_order_relaxed);
    }
    corridor.commits.fetch_add(commit_delta, std::memory_order_relaxed);
    if (boundary_wait_attempted) {
      corridor.boundary_wait_attempted.fetch_add(1, std::memory_order_relaxed);
    }
    if (boundary_wait_blocked) {
      corridor.boundary_wait_blocked.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void NoteRequiredWait(bool attempted, bool blocked) {
    if (!active_) return;
    MemexportCensusCorridor& corridor =
        GetMetalMemexportCensus().Corridor(consumer_);
    if (attempted) {
      corridor.required_wait_attempted.fetch_add(1, std::memory_order_relaxed);
    }
    if (blocked) {
      corridor.required_wait_blocked.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void NoteDrain(bool succeeded, uint64_t bytes_copied, uint64_t retirements) {
    if (!active_) return;
    MemexportCensusCorridor& corridor =
        GetMetalMemexportCensus().Corridor(consumer_);
    if (succeeded) {
      corridor.drains_succeeded.fetch_add(1, std::memory_order_relaxed);
    } else {
      corridor.drains_failed.fetch_add(1, std::memory_order_relaxed);
    }
    corridor.bytes_copied.fetch_add(bytes_copied, std::memory_order_relaxed);
    corridor.retirements.fetch_add(retirements, std::memory_order_relaxed);
  }

  PhysicalConsumerClass consumer() const { return consumer_; }
  bool active() const { return active_; }

 private:
  // The classified intervals must union to exactly the requested set. Compared
  // as ranges, not as a byte count.
  static bool UnionMatchesRequest(const IntervalClassification& classification,
                                  const GuestRangeSet& request) {
    GuestRangeSet normalized = request;
    normalized.Normalize();
    GuestRangeSet rebuilt;
    for (const ClassifiedInterval& interval : classification.intervals) {
      rebuilt.Add(interval.range);
    }
    rebuilt.Normalize();
    if (rebuilt.ranges().size() != normalized.ranges().size()) {
      return false;
    }
    for (size_t i = 0; i < rebuilt.ranges().size(); ++i) {
      if (rebuilt.ranges()[i].start() != normalized.ranges()[i].start() ||
          rebuilt.ranges()[i].length() != normalized.ranges()[i].length()) {
        return false;
      }
    }
    return true;
  }

  PhysicalConsumerClass consumer_;
  bool active_ = false;
};

// Names what the CURRENT production seam physically is, for the executor
// instrumentation to read. Set by an RAII marker at each call site so the class
// comes from the seam rather than from ConsumerKind, and restored on exit so
// nesting cannot leak a class into an unrelated corridor.
//
// A thread-local pointer, because the executor is reached only from the GPU
// thread's call sites and must not gain a parameter it would otherwise ignore
// when the census is off.
inline PhysicalConsumerClass*& CurrentMemexportCensusClassSlot() {
  static thread_local PhysicalConsumerClass* slot = nullptr;
  return slot;
}

class MetalMemexportCensusSeam {
 public:
  explicit MetalMemexportCensusSeam(PhysicalConsumerClass consumer)
      : consumer_(consumer),
        previous_(CurrentMemexportCensusClassSlot()) {
    CurrentMemexportCensusClassSlot() = &consumer_;
  }
  ~MetalMemexportCensusSeam() {
    CurrentMemexportCensusClassSlot() = previous_;
  }

  // The class of the innermost open seam, or nullopt when none is open — which
  // is itself an observation: an unattributed executor entry is an UNKNOWN, not
  // something to guess a class for.
  static const PhysicalConsumerClass* Current() {
    return CurrentMemexportCensusClassSlot();
  }

 private:
  PhysicalConsumerClass consumer_;
  PhysicalConsumerClass* previous_;
};

// An executor entry with no open seam. Counted against a dedicated bucket so it
// can never be silently folded into a real corridor.
inline void NoteUnattributedExecutorEntry() {
  if (!MetalMemexportCensus::Enabled()) {
    return;
  }
  GetMetalMemexportCensus()
      .Corridor(PhysicalConsumerClass::kHostRead)
      .unknown_attributions.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_H_
