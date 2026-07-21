/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MEMORY_AUTHORITY_H_
#define XENIA_GPU_METAL_METAL_MEMORY_AUTHORITY_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xenia/base/logging.h"
#include "xenia/gpu/metal/metal_guest_range.h"

namespace xe {
namespace gpu {
namespace metal {

// Typed, non-owning cross-writer memory-authority decision seam (OpenSpec
// implement-metal-cross-writer-memory-authority, tasks 2.3 and 2.4).
//
// This is the coordination contract the audit calls for: the same pending
// producer state (pending_writebacks_ / pending_memexports_) is today
// interpreted differently by the vertex, texture, and WAIT_REG_MEM call sites
// (raw 32-bit end arithmetic, unfiltered vs range-scoped drains). This header
// gives all of them ONE typed vocabulary and ONE pure decision function so the
// difference between the intended selective contract and today's legacy action
// becomes measurable rather than implicit.
//
// Design constraints this file honors (docs/metal_cross_writer_memory_authority
// _audit.md):
//   * Device-free and pure. No Metal, no command processor, no device handle.
//   * Non-owning. The decision functions read a SNAPSHOT of the existing
//     pending-producer records; they never store, mutate, or own them, and add
//     NO second version bitmap, address map, cache-key map, or independent
//     pending queue. SharedMemory validity and the command processor's pending
//     queues remain the sole sources of truth.
//   * Range-set aware. Consumers supply their precise GuestRangeSet (a texture's
//     disjoint base/mip extents, an aliased/multi-consumer set); the decision
//     selects ONLY intersecting producer records. Adjacent/disjoint ranges
//     choose no dependency (GuestRange treats adjacency as non-overlap).
//   * Incrementally routed. AuthorityObserver remains telemetry only, while the
//     RAM-writer publication seam below may execute the existing publication
//     operations for explicitly promoted writers. It owns no authority state.

// Which of the four writers produced (or will produce) the bytes.
enum class WriterKind {
  kGuestCpu,      // guest CPU store; RAM is authoritative, published by the
                  // physical-memory invalidation callback.
  kResolve,       // color/depth resolve scatter to guest RAM.
  kMemexport,     // shader memory export into the shared-memory buffer.
  kTraceRestore,  // trace playback restoring bytes directly to guest RAM.
};

// Which host/GPU consumer is preparing to read the range.
enum class ConsumerKind {
  kTexture,      // texture fetch (reads shared-memory / RAM-derived content).
  kVertex,       // vertex/index fetch (reads the shared-memory buffer).
  kWait,         // WAIT_REG_MEM and memory-predicate polls (reads guest RAM).
  kHostCpuRead,  // any other audited host CPU read of guest RAM.
};

// The bounded authority state of a range. RAM-current is the steady state; the
// two transit states are the only sanctioned GPU-newer-than-RAM conditions, and
// each MUST be owned by a submission-scoped pending record.
enum class TransitState {
  kRamCurrent,       // guest RAM holds the guest-visible bytes.
  kPendingResolve,   // a completed/queued resolve owes RAM its scattered bytes.
  kPendingMemexport,  // the shared buffer holds newer bytes owed to RAM.
};

// The action the authority contract selects. These match the design's explicit
// action set (design.md Decision 1 / Decision 5).
enum class AuthorityAction {
  kNoDependency,            // consumer disjoint from every pending producer:
                           // no commit, wait, or drain.
  kAwaitProducer,          // an overlapping producer's owning submission has not
                           // completed; await it, then drain.
  kDrainMatchingRecords,   // overlapping producers are complete; converge only
                           // the intersecting records to RAM.
  kInvalidateRamWrite,     // a RAM writer publishes its final extent by
                           // invalidating shared pages + overlapping caches.
  kRequestValidatedGpuBytes,  // a shared-buffer consumer revalidates its exact
                              // bytes from RAM after producers converged.
  kPublishGpuWrite,        // memexport marks the range GPU-written and registers
                           // its pending record as ONE ordered transition.
  kConvergeRamFromBuffer,  // a completed memexport copies buffer->RAM and
                           // retires its record without clobbering the buffer.
};

// A non-owning snapshot of one pending GPU producer record as it currently
// exists in pending_writebacks_ / pending_memexports_. The seam reads these to
// decide dependencies; it never owns or mutates the underlying record.
struct PendingProducer {
  GuestRange range;
  TransitState state;   // kPendingResolve or kPendingMemexport.
  uint64_t submission;  // owning submission id (0 == pre-submission / S7-off).
  uint64_t sequence = 0;  // chronological writer registration order.
};

// True once a producer's owning submission has finished on the GPU, using the
// same completed-submission high-water mark the command processor tracks.
inline bool ProducerComplete(const PendingProducer& producer,
                             uint64_t completed_submission) {
  // Submission 0 is the S7-tags-off / not-yet-owned sentinel. The producer is
  // still pending and must cross the legacy commit+wait boundary before RAM
  // convergence; treating 0 <= 0 as completion would under-predict that wait.
  return producer.submission != 0 &&
         producer.submission <= completed_submission;
}

// True when a consumer reads the shared-memory buffer (and therefore must
// revalidate its exact bytes from RAM after any dependency resolves). Wait and
// host-CPU reads read guest RAM directly and never revalidate GPU bytes.
inline bool ConsumerReadsSharedBuffer(ConsumerKind consumer) {
  return consumer == ConsumerKind::kTexture || consumer == ConsumerKind::kVertex;
}

// The full decision for one consumer preparing one requested range set against
// the current pending-producer snapshot. `selected` holds ONLY the producers
// that actually intersect the request, in snapshot order -- the selective-drain
// set. Disjoint/adjacent producers are never selected.
struct ConsumerDecision {
  ConsumerKind consumer = ConsumerKind::kTexture;
  AuthorityAction action = AuthorityAction::kNoDependency;
  GuestRangeSet requested;
  std::vector<PendingProducer> selected;
  bool await = false;                  // any selected producer incomplete.
  bool drain = false;                  // at least one producer to converge.
  bool revalidate_gpu_bytes = false;   // consumer reads the shared buffer.
  uint64_t max_wait_submission = 0;    // highest submission to await (0 if none).

  // Bytes actually depended on (covered by the selected producers), for
  // telemetry. Zero when there is no dependency.
  uint64_t selected_bytes() const {
    GuestRangeSet set;
    for (const PendingProducer& p : selected) {
      set.Add(p.range);
    }
    return set.CoveredBytes();
  }
};

// Producer-state-specific request ranges. Most consumers read the same ranges
// regardless of writer state and use Uniform(). Texture bridge service is the
// one factual exception: a bridge-served resolve is already GPU-ordered, while
// memexport at that address still needs convergence.
struct ConsumerRequest {
  GuestRangeSet resolve_ranges;
  GuestRangeSet memexport_ranges;

  static ConsumerRequest Uniform(const GuestRangeSet& ranges) {
    return {ranges, ranges};
  }

  const GuestRangeSet& ForProducer(TransitState state) const {
    return state == TransitState::kPendingMemexport ? memexport_ranges
                                                    : resolve_ranges;
  }

  GuestRangeSet Combined() const {
    GuestRangeSet combined = resolve_ranges;
    for (const GuestRange& range : memexport_ranges.ranges()) {
      combined.Add(range);
    }
    combined.Normalize();
    return combined;
  }
};

// The pure consumer decision. Selects only producers intersecting `requested`,
// classifies the resulting action, and reports whether an await/drain/revalidate
// is required. No side effects, no ownership, no device.
inline ConsumerDecision DecideConsumer(
    ConsumerKind consumer, const ConsumerRequest& request,
    const std::vector<PendingProducer>& producers,
    uint64_t completed_submission) {
  ConsumerDecision decision;
  decision.consumer = consumer;
  decision.requested = request.Combined();
  decision.revalidate_gpu_bytes = ConsumerReadsSharedBuffer(consumer);
  for (const PendingProducer& producer : producers) {
    if (!request.ForProducer(producer.state).OverlapsAny(producer.range)) {
      continue;  // disjoint or merely adjacent: not a dependency.
    }
    decision.selected.push_back(producer);
    if (!ProducerComplete(producer, completed_submission)) {
      decision.await = true;
      if (producer.submission > decision.max_wait_submission) {
        decision.max_wait_submission = producer.submission;
      }
    }
  }
  std::stable_sort(decision.selected.begin(), decision.selected.end(),
                   [](const PendingProducer& a, const PendingProducer& b) {
                     return a.sequence < b.sequence;
                   });
  if (decision.selected.empty()) {
    decision.action = AuthorityAction::kNoDependency;
    decision.drain = false;
  } else {
    decision.drain = true;
    decision.action = decision.await ? AuthorityAction::kAwaitProducer
                                     : AuthorityAction::kDrainMatchingRecords;
  }
  return decision;
}

inline ConsumerDecision DecideConsumer(
    ConsumerKind consumer, const GuestRangeSet& requested,
    const std::vector<PendingProducer>& producers,
    uint64_t completed_submission) {
  return DecideConsumer(consumer, ConsumerRequest::Uniform(requested),
                        producers, completed_submission);
}

// The pure writer-publication decision: how a writer's newly-final bytes reach
// their steady authority. Guest CPU, resolve, and trace-restore publish RAM
// (invalidate shared pages + caches); memexport publishes a GPU-written transit
// range owned by its pending record.
inline AuthorityAction DecideWriterPublication(WriterKind writer) {
  switch (writer) {
    case WriterKind::kMemexport:
      return AuthorityAction::kPublishGpuWrite;
    case WriterKind::kGuestCpu:
    case WriterKind::kResolve:
    case WriterKind::kTraceRestore:
    default:
      return AuthorityAction::kInvalidateRamWrite;
  }
}

// The transit state a writer's range occupies while a pending record owns it
// (kRamCurrent for the RAM writers whose publication is immediate).
inline TransitState WriterTransitState(WriterKind writer) {
  switch (writer) {
    case WriterKind::kResolve:
      return TransitState::kPendingResolve;
    case WriterKind::kMemexport:
      return TransitState::kPendingMemexport;
    case WriterKind::kGuestCpu:
    case WriterKind::kTraceRestore:
    default:
      return TransitState::kRamCurrent;
  }
}

inline const char* WriterKindName(WriterKind w) {
  switch (w) {
    case WriterKind::kGuestCpu: return "guest_cpu";
    case WriterKind::kResolve: return "resolve";
    case WriterKind::kMemexport: return "memexport";
    case WriterKind::kTraceRestore: return "trace_restore";
    default: return "?";
  }
}
inline const char* ConsumerKindName(ConsumerKind c) {
  switch (c) {
    case ConsumerKind::kTexture: return "texture";
    case ConsumerKind::kVertex: return "vertex";
    case ConsumerKind::kWait: return "wait";
    case ConsumerKind::kHostCpuRead: return "host_cpu_read";
    default: return "?";
  }
}
inline const char* AuthorityActionName(AuthorityAction a) {
  switch (a) {
    case AuthorityAction::kNoDependency: return "no_dependency";
    case AuthorityAction::kAwaitProducer: return "await_producer";
    case AuthorityAction::kDrainMatchingRecords: return "drain_matching";
    case AuthorityAction::kInvalidateRamWrite: return "invalidate_ram";
    case AuthorityAction::kRequestValidatedGpuBytes: return "request_gpu_bytes";
    case AuthorityAction::kPublishGpuWrite: return "publish_gpu_write";
    case AuthorityAction::kConvergeRamFromBuffer: return "converge_ram";
    default: return "?";
  }
}

// What the current legacy call site ACTUALLY did for a consumer, captured so the
// observer can compare it against the seam's prediction. This is an observation
// of existing behavior -- it never drives behavior.
struct LegacyConsumerOutcome {
  bool overlap = false;             // legacy detected an overlapping producer.
  bool committed = false;           // a command buffer was actually committed.
  bool waited = false;              // that submission was actually awaited.
  bool drained = false;             // legacy ran DrainPendingWritebacks.
  bool drain_range_scoped = false;  // filtered drain (vs unfiltered all-records).
  uint32_t drained_records = 0;     // records legacy retired, when known.
};

// Map a legacy consumer outcome to the action vocabulary so agreement can be
// judged on the same axis as the prediction.
inline AuthorityAction LegacyConsumerAction(const LegacyConsumerOutcome& o) {
  if (!o.overlap) {
    return AuthorityAction::kNoDependency;
  }
  return (o.committed || o.waited) ? AuthorityAction::kAwaitProducer
                                   : AuthorityAction::kDrainMatchingRecords;
}

// Independently observed writer-side legacy action. The observer compares this
// with the seam prediction; callers must not derive it from the same decision
// function or agreement would be tautological.
struct LegacyWriterOutcome {
  AuthorityAction action = AuthorityAction::kNoDependency;
  uint32_t invalidated_entries = 0;
  bool committed = false;
  bool waited = false;
  bool shared_invalidation_executed = false;
  bool legacy_invalidation_executed = false;
};

// Factual deltas returned by the existing synchronization and drain operations.
// Returning these changes no behavior; it only lets observer call sites report
// what happened rather than infer it from an overlap branch.
struct LegacySyncOutcome {
  bool committed = false;
  bool waited = false;
};
struct LegacyDrainOutcome {
  uint32_t drained_records = 0;
  uint32_t resolve_records = 0;
  uint32_t memexport_records = 0;
  uint32_t invalidated_entries = 0;
};

// Existing RAM-write publication operations, exposed through a thin device-free
// port so the typed seam can execute them and tests can substitute a recorder.
// The port exposes invalidation only -- never watch registration or authority
// storage -- so routing through it cannot add a second CPU watch.
struct RamWritePublicationTarget {
  virtual ~RamWritePublicationTarget() = default;

  // Invalidate the shared-memory pages backing [base, base+length).
  virtual bool InvalidateSharedPages(uint32_t base, uint32_t length,
                                     bool exact_range) = 0;

  // Invalidate overlapping legacy texture-cache entries, returning the number
  // retired. Only writers whose policy requires the independently-needed direct
  // invalidation (trace restore during the phase-5 equivalence period) call it.
  virtual bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                        uint32_t* invalidated_entries) = 0;
};

// Source-appropriate RAM-write publication policy. Guest-CPU publication is
// observed after the sole physical-memory callback has already performed the
// invalidation, so executing the seam must not invalidate it a second time.
struct RamWritePublicationPolicy {
  bool execute_shared_invalidation = true;
  bool exact_range = true;
  bool execute_legacy_invalidation = false;
};

inline RamWritePublicationPolicy RamWritePolicyFor(WriterKind writer) {
  switch (writer) {
    case WriterKind::kGuestCpu:
      // The sole physical callback has already invalidated shared pages and
      // fired watches before this publication is observed.
      return {/*shared=*/false, /*exact=*/false, /*legacy=*/false};
    case WriterKind::kTraceRestore:
      // Exact publication plus the independently-required direct legacy texture
      // invalidation retained through the phase-5 equivalence period (audit
      // blocker 3: trace playback unregisters the legacy global watch, so its
      // direct InvalidateRange is not yet redundant).
      return {/*shared=*/true, /*exact=*/true, /*legacy=*/true};
    case WriterKind::kResolve:
      // Exact publication of the checked scatter extent; legacy fan-out is the
      // global watch. Routed in task 3.2; the policy lives here so the seam
      // speaks one vocabulary across writers.
      return {/*shared=*/true, /*exact=*/true, /*legacy=*/false};
    case WriterKind::kMemexport:
    default:
      // Memexport does not publish RAM by invalidation; it converges RAM from
      // the buffer under its pending record (task 3.3). The seam performs no RAM
      // invalidation for it.
      return {/*shared=*/false, /*exact=*/true, /*legacy=*/false};
  }
}

// One normalized RAM-writer publication decision. It is a transient value, not
// persistent authority. The factory rejects empty/out-of-domain input and
// clamps the end before any backend operation sees the range.
struct RamWritePublication {
  WriterKind writer;
  GuestRange range;
  AuthorityAction action;
  RamWritePublicationPolicy policy;
};

inline std::optional<RamWritePublication> DecideRamWritePublication(
    WriterKind writer, uint32_t base, uint32_t length) {
  std::optional<GuestRange> range = GuestRange::FromBaseLength(base, length);
  if (!range) {
    return std::nullopt;
  }
  return RamWritePublication{writer, *range, DecideWriterPublication(writer),
                             RamWritePolicyFor(writer)};
}

// Execute a normalized decision through existing backend operations. Guest CPU
// decisions intentionally execute nothing because the physical callback has
// already published them. This function owns no state and adds no
// synchronization.
inline LegacyWriterOutcome PublishRamWrite(
    const RamWritePublication& publication,
    RamWritePublicationTarget& target) {
  LegacyWriterOutcome outcome;
  bool required_operations_executed = true;
  if (publication.policy.execute_shared_invalidation) {
    outcome.shared_invalidation_executed = target.InvalidateSharedPages(
        publication.range.start(), publication.range.length(),
        publication.policy.exact_range);
    required_operations_executed &= outcome.shared_invalidation_executed;
  }
  if (publication.policy.execute_legacy_invalidation) {
    outcome.legacy_invalidation_executed = target.InvalidateLegacyTextures(
        publication.range.start(), publication.range.length(),
        &outcome.invalidated_entries);
    required_operations_executed &= outcome.legacy_invalidation_executed;
  }
  outcome.action = required_operations_executed
                       ? publication.action
                       : AuthorityAction::kNoDependency;
  return outcome;
}

// Observer-only telemetry (task 2.4). Bounded, throttled, non-owning. It holds
// ONLY counters and a tiny disagreement sample -- never authority state -- and
// it executes nothing. Callers feed it a prediction and the legacy outcome; it
// tallies agreement and emits a throttled diagnostic on divergence.
class AuthorityObserver {
 public:
  // Per-consumer and per-writer sampling budget. The first `kInitialSamples`
  // events of each kind are always recorded; after that only 1 in
  // `kSampleStride` is, so steady-state cost stays flat and logs stay quiet.
  static constexpr uint32_t kInitialSamples = 64;
  static constexpr uint32_t kSampleStride = 1024;
  // How many divergences to log verbatim before switching to counters only.
  static constexpr uint32_t kMaxDivergenceLogs = 32;

  // True when the next consumer/writer event of this kind should build a full
  // snapshot and record. Cheap: one counter increment + modulo. Call sites gate
  // the (allocating) snapshot build behind this so unsampled events are ~free.
  bool ShouldSampleConsumer(ConsumerKind consumer) {
    uint32_t n = consumer_seen_[static_cast<size_t>(consumer)]++;
    return n < kInitialSamples || (n % kSampleStride) == 0;
  }
  bool ShouldSampleWriter(WriterKind writer) {
    uint32_t n = writer_seen_[static_cast<size_t>(writer)]++;
    return n < kInitialSamples || (n % kSampleStride) == 0;
  }

  // Record one consumer decision alongside the legacy outcome that actually ran.
  void RecordConsumer(const ConsumerDecision& predicted,
                      const LegacyConsumerOutcome& legacy) {
    size_t ci = static_cast<size_t>(predicted.consumer);
    ++consumer_records_[ci];
    action_count_[static_cast<size_t>(predicted.action)]++;
    if (predicted.await) {
      ++consumer_await_[ci];
    }
    if (predicted.drain) {
      ++consumer_drain_[ci];
      predicted_drained_records_[ci] += predicted.selected.size();
    }
    if (legacy.committed) {
      ++legacy_commit_[ci];
    }
    if (legacy.waited) {
      ++legacy_wait_[ci];
    }
    if (legacy.drained) {
      ++legacy_drain_[ci];
      legacy_drained_records_[ci] += legacy.drained_records;
    }
    AuthorityAction legacy_action = LegacyConsumerAction(legacy);
    bool agree = legacy_action == predicted.action;
    // Every attempted drain must retire exactly the selected records. This
    // catches over-broad unfiltered drains, under-retirement after a failed
    // wait/copy, and mismatches in nominally range-scoped paths.
    bool drain_count_divergence =
        predicted.drain && legacy.drained &&
        legacy.drained_records != predicted.selected.size();
    if (agree && !drain_count_divergence) {
      ++consumer_agree_[ci];
    } else {
      ++consumer_disagree_[ci];
      if (divergence_logs_++ < kMaxDivergenceLogs) {
        XELOGI(
            "MetalAuthority[observer] consumer={} predicted={} legacy={} "
            "selected={} selected_bytes={} requested_ranges={} "
            "requested_bytes={} legacy_records={} legacy_scoped={} "
            "(divergence)",
            ConsumerKindName(predicted.consumer),
            AuthorityActionName(predicted.action),
            AuthorityActionName(legacy_action), predicted.selected.size(),
            predicted.selected_bytes(), predicted.requested.size(),
            predicted.requested.CoveredBytes(), legacy.drained_records,
            legacy.drain_range_scoped);
      }
    }
  }

  // Record one writer publication alongside the independently observed legacy
  // outcome. Invalidation counts are factual Entry retirements returned by the
  // direct legacy cache call; deferred watch fan-out remains a separately
  // measured phase-5 surface and is not fabricated here.
  void RecordWriter(WriterKind writer, const GuestRangeSet& ranges,
                    AuthorityAction predicted_action,
                    const LegacyWriterOutcome& legacy) {
    size_t wi = static_cast<size_t>(writer);
    ++writer_records_[wi];
    action_count_[static_cast<size_t>(predicted_action)]++;
    writer_range_count_[wi] += ranges.size();
    writer_range_bytes_[wi] += ranges.CoveredBytes();
    writer_invalidated_entries_[wi] += legacy.invalidated_entries;
    if (legacy.committed) {
      ++writer_commits_[wi];
    }
    if (legacy.waited) {
      ++writer_waits_[wi];
    }
    if (predicted_action != legacy.action) {
      ++writer_disagree_[wi];
      if (divergence_logs_++ < kMaxDivergenceLogs) {
        XELOGW(
            "MetalAuthority[observer] writer={} predicted={} legacy={} "
            "ranges={} bytes={} (policy divergence)",
            WriterKindName(writer), AuthorityActionName(predicted_action),
            AuthorityActionName(legacy.action), ranges.size(),
            ranges.CoveredBytes());
      }
    } else {
      ++writer_agree_[wi];
    }
  }

  // Emit a one-line summary per consumer/writer kind. Throttled by the caller
  // (e.g. once per swap interval); safe to call at any point.
  void ReportSummary() const {
    for (size_t c = 0; c < kConsumerCount; ++c) {
      if (!consumer_records_[c]) {
        continue;
      }
      XELOGI(
          "MetalAuthority[observer] consumer={} records={} agree={} "
          "disagree={} await={} drain={} legacy_commits={} legacy_waits={} "
          "legacy_drain={} "
          "pred_drained_recs={} legacy_drained_recs={}",
          ConsumerKindName(static_cast<ConsumerKind>(c)), consumer_records_[c],
          consumer_agree_[c], consumer_disagree_[c], consumer_await_[c],
          consumer_drain_[c], legacy_commit_[c], legacy_wait_[c],
          legacy_drain_[c], predicted_drained_records_[c],
          legacy_drained_records_[c]);
    }
    for (size_t w = 0; w < kWriterCount; ++w) {
      if (!writer_records_[w]) {
        continue;
      }
      XELOGI(
          "MetalAuthority[observer] writer={} records={} agree={} disagree={} "
          "ranges={} bytes={} invalidated_entries={} commits={} waits={}",
          WriterKindName(static_cast<WriterKind>(w)), writer_records_[w],
          writer_agree_[w], writer_disagree_[w], writer_range_count_[w],
          writer_range_bytes_[w],
          writer_invalidated_entries_[w], writer_commits_[w],
          writer_waits_[w]);
    }
  }

  // Accessors for the conformance tests (no device, no log noise).
  uint64_t consumer_records(ConsumerKind c) const {
    return consumer_records_[static_cast<size_t>(c)];
  }
  uint64_t consumer_agree(ConsumerKind c) const {
    return consumer_agree_[static_cast<size_t>(c)];
  }
  uint64_t consumer_disagree(ConsumerKind c) const {
    return consumer_disagree_[static_cast<size_t>(c)];
  }
  uint64_t writer_records(WriterKind w) const {
    return writer_records_[static_cast<size_t>(w)];
  }
  uint64_t writer_agree(WriterKind w) const {
    return writer_agree_[static_cast<size_t>(w)];
  }
  uint64_t writer_range_count(WriterKind w) const {
    return writer_range_count_[static_cast<size_t>(w)];
  }
  uint64_t writer_range_bytes(WriterKind w) const {
    return writer_range_bytes_[static_cast<size_t>(w)];
  }
  uint64_t action_count(AuthorityAction a) const {
    return action_count_[static_cast<size_t>(a)];
  }
  uint64_t predicted_drained_records(ConsumerKind c) const {
    return predicted_drained_records_[static_cast<size_t>(c)];
  }
  uint64_t predicted_drained_records() const {
    uint64_t total = 0;
    for (uint64_t count : predicted_drained_records_) {
      total += count;
    }
    return total;
  }
  uint64_t legacy_drained_records(ConsumerKind c) const {
    return legacy_drained_records_[static_cast<size_t>(c)];
  }
  uint64_t legacy_drained_records() const {
    uint64_t total = 0;
    for (uint64_t count : legacy_drained_records_) {
      total += count;
    }
    return total;
  }

 private:
  static constexpr size_t kConsumerCount = 4;
  static constexpr size_t kWriterCount = 4;
  static constexpr size_t kActionCount = 7;

  // Sampling counters (mutable across the throttle checks).
  uint32_t consumer_seen_[kConsumerCount] = {};
  uint32_t writer_seen_[kWriterCount] = {};

  // Consumer tallies.
  uint64_t consumer_records_[kConsumerCount] = {};
  uint64_t consumer_agree_[kConsumerCount] = {};
  uint64_t consumer_disagree_[kConsumerCount] = {};
  uint64_t consumer_await_[kConsumerCount] = {};
  uint64_t consumer_drain_[kConsumerCount] = {};
  uint64_t legacy_commit_[kConsumerCount] = {};
  uint64_t legacy_wait_[kConsumerCount] = {};
  uint64_t legacy_drain_[kConsumerCount] = {};

  // Writer tallies.
  uint64_t writer_records_[kWriterCount] = {};
  uint64_t writer_agree_[kWriterCount] = {};
  uint64_t writer_disagree_[kWriterCount] = {};
  uint64_t writer_range_count_[kWriterCount] = {};
  uint64_t writer_range_bytes_[kWriterCount] = {};
  uint64_t writer_invalidated_entries_[kWriterCount] = {};
  uint64_t writer_commits_[kWriterCount] = {};
  uint64_t writer_waits_[kWriterCount] = {};

  // Action histogram + drain-breadth accounting.
  uint64_t action_count_[kActionCount] = {};
  uint64_t predicted_drained_records_[kConsumerCount] = {};
  uint64_t legacy_drained_records_[kConsumerCount] = {};

  uint32_t divergence_logs_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMORY_AUTHORITY_H_
