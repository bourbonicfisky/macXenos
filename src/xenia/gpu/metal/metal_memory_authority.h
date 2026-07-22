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
#include <optional>
#include <vector>

#include "xenia/gpu/metal/metal_guest_range.h"

namespace xe {
namespace gpu {
namespace metal {

// Typed, non-owning cross-writer memory-authority decision and execution seam
// (OpenSpec implement-metal-cross-writer-memory-authority).
//
// This is the coordination contract the audit calls for: the same pending
// producer state (pending_writebacks_ / pending_memexports_) was previously
// interpreted independently by the vertex, texture, and WAIT_REG_MEM call
// sites. This header gives every promoted writer and consumer one typed
// vocabulary and one pure decision function for selective convergence.
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
//   * Non-authoritative execution. The RAM-writer publication and consumer
//     dependency seams below execute the existing publication/drain operations
//     for the promoted writers and consumers. They own no authority state.

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
  // still pending and must cross the synchronous commit+wait boundary before
  // RAM
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

// The pure consumer decision. Seeds selection only from producers intersecting
// `requested`, then closes over older overlapping predecessors so whole-record
// convergence cannot invert writer chronology. Classifies the resulting action
// and reports whether await/drain/revalidate is required. No side effects, no
// ownership, no device.
inline ConsumerDecision DecideConsumer(
    ConsumerKind consumer, const ConsumerRequest& request,
    const std::vector<PendingProducer>& producers,
    uint64_t completed_submission) {
  ConsumerDecision decision;
  decision.consumer = consumer;
  decision.requested = request.Combined();
  decision.revalidate_gpu_bytes = ConsumerReadsSharedBuffer(consumer);
  std::vector<bool> selected(producers.size(), false);
  for (size_t i = 0; i < producers.size(); ++i) {
    const PendingProducer& producer = producers[i];
    if (!request.ForProducer(producer.state).OverlapsAny(producer.range)) {
      continue;  // disjoint or merely adjacent: not a dependency.
    }
    selected[i] = true;
  }

  // Convergence applies whole producer records, not only their intersection
  // with the consumer request. Close the selected set transitively over every
  // older overlapping producer: otherwise an unselected predecessor could be
  // drained later and overwrite a newer record already converged here. This is
  // especially important for bridge service, where the initial request may
  // omit resolves, but it is writer-kind-independent (M1 -> R2 -> M3 chains
  // need all three). A frontier makes the closure O(selected * producers), and
  // resolve-only bridge traffic keeps its fast path because it seeds nothing.
  std::vector<size_t> closure_frontier;
  closure_frontier.reserve(producers.size());
  for (size_t i = 0; i < producers.size(); ++i) {
    if (selected[i]) {
      closure_frontier.push_back(i);
    }
  }
  for (size_t frontier_index = 0; frontier_index < closure_frontier.size();
       ++frontier_index) {
    const PendingProducer& newer =
        producers[closure_frontier[frontier_index]];
    for (size_t older_index = 0; older_index < producers.size();
         ++older_index) {
      const PendingProducer& older = producers[older_index];
      if (selected[older_index] || older.sequence >= newer.sequence ||
          !older.range.Overlaps(newer.range)) {
        continue;
      }
      selected[older_index] = true;
      closure_frontier.push_back(older_index);
    }
  }

  for (size_t i = 0; i < producers.size(); ++i) {
    if (!selected[i]) {
      continue;
    }
    const PendingProducer& producer = producers[i];
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
// Factual writer-side publication result returned by PublishRamWrite. `action`
// is the promoted outcome the caller checks (kInvalidateRamWrite on success);
// the remaining fields report which existing invalidation operations actually
// ran so callers report facts rather than infer them.
struct RamWritePublicationOutcome {
  AuthorityAction action = AuthorityAction::kNoDependency;
  uint32_t invalidated_entries = 0;
  bool shared_invalidation_executed = false;
  bool legacy_invalidation_executed = false;
};

// Factual result of producer convergence used by the promoted selective-drain
// executor and its behavioral tests.
struct ProducerDrainOutcome {
  uint32_t drained_records = 0;
  uint32_t resolve_records = 0;
  uint32_t memexport_records = 0;
  uint32_t invalidated_entries = 0;
};

// The one publication-convergence predicate shared by production and the
// conformance harness: a RAM consumer's publication is complete only when the
// selective drain retired exactly one pending record for every selected
// producer. A short retirement (the stop-on-first-failure policy above halting,
// or a failed wait yielding an empty drain) leaves the failing record and its
// untouched selected suffix pending, so the consumer must fail closed.
//
// MetalCommandProcessor::ResolvePendingProducerDependencies uses this to set
// ProducerDependencyOutcome::complete, so the bool the swap's full-domain
// publication consumes as its present/suppress gate is exactly this function of
// the decision and drain. Owning no state, it lets the device-free tests assert
// the SAME gate production presents on, rather than an independently recomputed
// availability flag.
inline bool ConsumerPublicationComplete(const ConsumerDecision& decision,
                                        const ProducerDrainOutcome& drain) {
  return drain.drained_records == decision.selected.size();
}

// The selective-drain executor shared by production
// (MetalCommandProcessor::DrainSelectedPendingProducers) and the task-4.5
// conformance executors. It walks the seam's sequence-ordered selected records,
// drains each through the caller-supplied one-record callback, merges the
// factual ProducerDrainOutcome counts, and stops the moment a record fails to
// retire exactly one pending record.
//
// Stopping on the first non-unit retirement preserves authority chronology: a
// retained older record must never be followed by a newer selected writer, or a
// later retry of the older record could overwrite the newer bytes and invert
// writer order. The caller fails the consumer closed while the failing record
// and the untouched selected suffix stay pending.
//
// This helper owns NO state and adds no queue, flag, ownership, or title rule.
// The callback performs the actual convergence against whatever pending queue it
// closes over, matching each selected record back by its unique authority
// sequence; the selection itself remains a transient value.
template <typename DrainOneProducer>
inline ProducerDrainOutcome DrainSelectedProducers(
    const ConsumerDecision& decision, DrainOneProducer&& drain_one) {
  ProducerDrainOutcome combined;
  for (const PendingProducer& producer : decision.selected) {
    ProducerDrainOutcome one = drain_one(producer);
    combined.drained_records += one.drained_records;
    combined.resolve_records += one.resolve_records;
    combined.memexport_records += one.memexport_records;
    combined.invalidated_entries += one.invalidated_entries;
    if (one.drained_records != 1) {
      break;
    }
  }
  return combined;
}

// RAM-write publication operations, exposed through a thin device-free
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
  // invalidation (trace restore, whose legacy global watch is unregistered by
  // SetTracePlayback) call it. Task 5.3's writer-matrix comparison proved this
  // leg is durably required, not a temporary phase-5 measure: with the global
  // watch gone the watch fan-out is empty, so nothing else retires these
  // entries.
  virtual bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                        uint32_t* invalidated_entries) = 0;
};

// Source-appropriate RAM-write publication policy. The sole physical-memory
// callback already performs guest-CPU publication, so a typed model decision
// must not invalidate the same write a second time.
struct RamWritePublicationPolicy {
  bool execute_shared_invalidation = true;
  bool exact_range = true;
  bool execute_legacy_invalidation = false;
};

inline RamWritePublicationPolicy RamWritePolicyFor(WriterKind writer) {
  switch (writer) {
    case WriterKind::kGuestCpu:
      // The sole physical callback has already invalidated shared pages and
      // fired watches; the typed policy executes no duplicate operation.
      return {/*shared=*/false, /*exact=*/false, /*legacy=*/false};
    case WriterKind::kTraceRestore:
      // Exact publication plus the independently-required direct legacy texture
      // invalidation. Trace playback unregisters the legacy global watch, so
      // watch fan-out cannot retire these entries and the direct InvalidateRange
      // is durably required. Task 5.3's writer-matrix comparison proved this
      // leg is not removable (the global-watch set is empty after
      // SetTracePlayback), so it is retained by design rather than pending a
      // later equivalence proof.
      return {/*shared=*/true, /*exact=*/true, /*legacy=*/true};
    case WriterKind::kResolve:
      // Exact publication of the checked scatter extent. The resolve drain
      // performs the scoped direct legacy RAM-upload invalidation separately;
      // this publication suppresses duplicate global fan-out while still
      // firing shared-cache per-range watches. Routed in task 3.2; the policy
      // lives here so the seam speaks one vocabulary across writers.
      return {/*shared=*/true, /*exact=*/true, /*legacy=*/false};
    case WriterKind::kMemexport:
    default:
      // Memexport does not publish RAM by invalidation; it converges RAM from
      // the buffer under its pending record (task 3.3). The seam performs no RAM
      // invalidation for it.
      return {/*shared=*/false, /*exact=*/true, /*legacy=*/false};
  }
}

// --- Memexport destination preparation / publication / convergence (task 3.3)
//
// Memexport is the one writer whose bytes exist in the shared buffer before
// they exist in guest RAM (design.md Decision 2's "pending memexport" transit
// state). Three distinct typed steps route through this seam:
//   1. Destination preparation: request the destination pages BEFORE the GPU
//      write so a stale CPU-RAM upload cannot race ahead of the export.
//   2. Publication: after preflighting the marker, the seam registers the
//      submission-scoped pending record first and then calls RangeWrittenByGpu
//      in the SAME ordered step. No interval is marked GPU-newer before it has
//      an owner.
//   3. Convergence: a completed pending record's exact bytes are copied
//      shared-buffer -> RAM. The port never touches shared-buffer validity;
//      the buffer remains the authoritative source the copy read from. The
//      caller retires the pending record only when the copy reports success.

// Requests the destination range's pages before the GPU write.
struct MemexportPreparationTarget {
  virtual ~MemexportPreparationTarget() = default;
  virtual bool RequestDestinationRange(uint32_t base, uint32_t length) = 0;
};

// One normalized memexport destination-preparation decision.
struct MemexportDestinationPreparation {
  GuestRange range;
};

inline std::optional<MemexportDestinationPreparation>
DecideMemexportDestinationPreparation(uint32_t base, uint32_t length) {
  std::optional<GuestRange> range = GuestRange::FromBaseLength(base, length);
  if (!range) {
    return std::nullopt;
  }
  return MemexportDestinationPreparation{*range};
}

// Executes the destination-preparation decision through the existing
// SharedMemory::RequestRange operation. Returns the factual RequestRange
// result -- false means the exports have nowhere to land and the caller must
// treat the draw the same as today (skip it), never invent success.
inline bool PrepareMemexportDestination(
    const MemexportDestinationPreparation& preparation,
    MemexportPreparationTarget& target) {
  return target.RequestDestinationRange(preparation.range.start(),
                                        preparation.range.length());
}

// Marks a memexport range GPU-written.
struct MemexportPublicationTarget {
  virtual ~MemexportPublicationTarget() = default;
  virtual bool CanMarkRangeWrittenByGpu() const = 0;
  virtual void MarkRangeWrittenByGpu(uint32_t base, uint32_t length) = 0;
};

// One normalized memexport GPU-write publication decision, owned by the
// exporting submission.
struct MemexportPublication {
  GuestRange range;
  uint64_t submission;
};

inline std::optional<MemexportPublication> DecideMemexportPublication(
    uint32_t base, uint32_t length, uint64_t submission) {
  std::optional<GuestRange> range = GuestRange::FromBaseLength(base, length);
  if (!range) {
    return std::nullopt;
  }
  return MemexportPublication{*range, submission};
}

// Factual result of the ordered ownership + GPU-write publication.
struct MemexportPublicationResult {
  bool registered = false;
  bool marked = false;
  AuthorityAction action = AuthorityAction::kNoDependency;
};

template <typename RegisterPending>
inline MemexportPublicationResult PublishMemexportWrite(
    const MemexportPublication& publication,
    MemexportPublicationTarget& target, RegisterPending&& register_pending) {
  MemexportPublicationResult result;
  if (!target.CanMarkRangeWrittenByGpu()) {
    return result;
  }
  // Ownership is deliberately installed BEFORE the GPU-written notification.
  // If record allocation/registration fails, the marker is never called.
  result.registered = register_pending(publication);
  if (!result.registered) {
    return result;
  }
  target.MarkRangeWrittenByGpu(publication.range.start(),
                               publication.range.length());
  result.marked = true;
  result.action = AuthorityAction::kPublishGpuWrite;
  return result;
}

// Copies one completed pending memexport record's exact bytes from the
// shared buffer to guest RAM. The port MUST NOT invalidate or overwrite the
// shared buffer -- it is the authoritative source of the converged bytes --
// and MUST report false (rather than partially converge) when the RAM
// mapping or shared-buffer contents pointer is unavailable, so the caller
// retains the pending record for retry instead of losing GPU-newer bytes.
struct MemexportConvergenceTarget {
  virtual ~MemexportConvergenceTarget() = default;
  virtual bool CopyBufferToRam(uint32_t base, uint32_t length,
                               uint32_t* invalidated_entries) = 0;
};

// Factual result of one convergence attempt. `converged` gates record
// retirement: the caller must retain the pending record for retry whenever
// this is false, and increment drain counters only when it is true.
struct MemexportConvergenceResult {
  bool converged = false;
  AuthorityAction action = AuthorityAction::kNoDependency;
  uint32_t invalidated_entries = 0;
};

inline MemexportConvergenceResult ConvergeMemexportRam(
    const PendingProducer& producer, MemexportConvergenceTarget& target) {
  MemexportConvergenceResult result;
  result.converged = target.CopyBufferToRam(
      producer.range.start(), producer.range.length(),
      &result.invalidated_entries);
  result.action = result.converged ? AuthorityAction::kConvergeRamFromBuffer
                                   : AuthorityAction::kNoDependency;
  return result;
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
inline RamWritePublicationOutcome PublishRamWrite(
    const RamWritePublication& publication,
    RamWritePublicationTarget& target) {
  RamWritePublicationOutcome outcome;
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

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMORY_AUTHORITY_H_
