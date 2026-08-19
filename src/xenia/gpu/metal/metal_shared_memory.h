/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_
#define XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "xenia/gpu/metal/metal_zero_copy_contract.h"
#include "xenia/gpu/metal/metal_zero_copy_overlap.h"
#include "xenia/gpu/shared_memory.h"

namespace xe {
class Memory;
namespace gpu {
class TraceWriter;
namespace metal {

// Factual result of one shared-buffer -> guest-RAM convergence performed by the
// shared memory itself. Every field is recorded at the physical operation, not
// derived from the backing mode: `bytes_copied` is assigned at the memcpy that
// moved them, and `alias_skipped` is set only where the destination was PROVEN
// to be the source. A caller can therefore report what happened instead of
// restating what it configured.
struct SharedBufferConvergenceOutcome {
  // Gates pending-record retirement. False means nothing was published and the
  // caller must retain its record for retry.
  bool converged = false;
  // Bytes this call actually moved. Zero on the proven alias path.
  uint64_t bytes_copied = 0;
  // The copy was suppressed because destination and source are one allocation.
  bool alias_skipped = false;
};

// Real xe::gpu::SharedMemory subclass over a single 512 MiB shared-storage
// MTLBuffer where buffer offset == guest physical byte address, so the
// translated vertex shader's vfetch reads straight out of it. The base class
// provides page-granular validity tracking with CPU write-watch invalidation:
// uploads happen only for pages actually written since their last use. (The
// bring-up shim this replaces memcmp'd entire fetch ranges per draw — the
// terrain mega-streams are tens of MB, which became tens of GB of comparison
// per frame once real gameplay rendered.) The buffer is CPU-visible, so
// UploadRanges is a plain memcpy — no staging, no GPU commands.
//
// Zero-copy pre-gate (default off): when XE_METAL_ZERO_COPY_SHARED_MEMORY is
// exactly "1" the buffer may instead WRAP the guest physical view itself
// (newBufferWithBytesNoCopy), so buffer bytes and guest RAM are one allocation.
// Every other value, and every failed contract check, takes the mirror path
// above unchanged. See metal_zero_copy_contract.h. In that mode the upload and
// memexport-convergence copies become no-ops over bytes that are already in
// place; nothing else about validity, tracing, hazard flushing, writer
// ordering, or record retirement changes.
class MetalSharedMemory : public SharedMemory {
 public:

  MetalSharedMemory(Memory& memory, void* device);
  ~MetalSharedMemory() override;

  bool Initialize();
  void Shutdown();

  // Frame-trace recording: every upload's source range is written into the
  // open trace (uploads read guest RAM, which is exactly the data a replay
  // must restore). WriteMemoryRead self-guards when no trace is open.
  void SetTraceWriter(TraceWriter* trace_writer) {
    trace_writer_ = trace_writer;
  }
  // Frame-trace init: invalidates every non-GPU-written page (so ranges the
  // traced frame consumes re-upload and get recorded) and writes collected
  // GPU-written ranges straight from guest RAM (CPU-authoritative here).
  void WriteTraceInitialState(TraceWriter& trace_writer);

  // id<MTLBuffer> (shared storage), not owned by the caller.
  void* GetBuffer() const { return buffer_; }

  // Whether the installed buffer aliases guest RAM rather than mirroring it.
  // Decided once in Initialize and never transitioned afterwards: consumers
  // must not have to ask whether the backing changed mid-run. False until a
  // successful Initialize and again after Shutdown.
  bool is_zero_copy() const { return is_zero_copy_; }
  // The facts the single startup line was derived from, kept for conformance
  // and for the pre-gate's control arms (an exact-off run must show a
  // completely untouched observation).
  const ZeroCopyObservation& zero_copy_observation() const {
    return zero_copy_observation_;
  }

  // Frame-batching correctness: every draw in the open frame command buffer
  // reads the buffer's state at COMMIT, not at encode. When a re-upload is
  // about to land in pages a draw encoded this frame has already consumed
  // (the intra-frame dynamic rewrite: UI glyph rings, particles), the flush
  // callback (commit + wait, set by the command processor) runs FIRST so the
  // encoded draws read the old data.
  void SetHazardFlushCallback(std::function<void()> callback) {
    hazard_flush_callback_ = std::move(callback);
  }

  // RequestRange plus recording [start, start+length) as consumed by the
  // current frame for the hazard detection above.
  bool RequestConsumedRange(uint32_t start, uint32_t length);

  // After a GPU-current publication the walked bytes are IDENTICAL in the
  // buffer and RAM (the publication just copied buffer -> RAM, and the
  // extent's padding was committed from RAM at dispatch). Mark the pages
  // valid-not-gpu-written so consumers stop re-uploading bytes the buffer
  // already holds (the 4.2 probes measured +45% tex uploads/frame from the
  // publication's blanket invalidation). Watches for the content CHANGE were
  // already fired by the publication's invalidate; this only restores page
  // validity.
  void MarkRangeRamEqual(uint32_t start, uint32_t length) {
    MakeRangeValid(start, length, /*written_by_gpu=*/false);
  }

  // S8: the submission the currently-open command buffer will get — consumed
  // ranges are tagged with it so ReleaseConsumedRanges frees exactly the
  // ranges whose reader has completed (a blanket clear would drop hazard
  // protection for LATER in-flight submissions).
  void SetSubmissionProvider(std::function<uint64_t()> provider) {
    submission_provider_ = std::move(provider);
  }
  void ReleaseConsumedRanges(uint64_t completed_submission);

  // --- Zero-copy in-flight CPU rewrite evidence (pre-gate tasks 5.1-5.4) ----
  //
  // Two providers, because the evidence needs two INDEPENDENT sources (design
  // D6 / D2a). Both are installed once, before Initialize, and are only read
  // afterwards: the faulting guest thread invokes one of them and must not race
  // an assignment.

  // EXECUTION. Deliberately NOT the S7-governed submission_provider_ above.
  // That one returns `S7TagsOff() ? 0 : CurrentEncodingSubmission()`, and
  // S7TagsOff() is true on every default launch, so it reports 0 — which is the
  // idle sentinel. Reusing it would stamp every page with "nobody is reading
  // this", and the overlap counter could then only ever be zero, for a reason
  // that has nothing to do with whether an overlap happened. This provider must
  // report the real encoding submission in every mode.
  void SetOverlapSubmissionProvider(std::function<uint64_t()> provider) {
    overlap_submission_provider_ = std::move(provider);
  }
  // COMPLETION, supplied separately from the stamps so no single source both
  // records the claim and decides whether it is still outstanding. Read once
  // per invalidation callback, on the faulting guest thread, so it must be an
  // atomic load and nothing more.
  void SetOverlapCompletedSubmissionProvider(
      std::function<uint64_t()> provider) {
    overlap_completed_submission_provider_ = std::move(provider);
  }
  // Bounded evidence for the lifecycle reporter. Inert (and reported as
  // inactive) unless alias mode installed the observer.
  //
  // Taken under the physical-invalidation callback lock, so the counters it
  // returns are one coherent moment rather than a set of independently-timed
  // reads that could, say, show more classified pages than observed callbacks.
  // Bounded and allocation-free inside the lock: a fixed-size struct copy and
  // no logging (the caller formats after it returns).
  ZeroCopyOverlapSnapshot zero_copy_overlap_snapshot() const;

  // Forgets the consumed-range set (call after the frame command buffer has
  // been committed and waited on).
  void ResetFrameRanges() { consumed_ranges_.clear(); }  // sync-mode/shutdown
  // Post-clear evidence: submission-tagged hazard records still held. At a
  // quiesced boundary every submission has completed, so ReleaseConsumedRanges
  // (run by the pumped completion actions) should already have emptied this --
  // a claim the clear verifies rather than assumes.
  uint32_t consumed_range_count() const {
    return uint32_t(consumed_ranges_.size());
  }

  // Publishes one completed memexport's exact bytes from the shared buffer to
  // guest RAM (the byte-moving half of the typed MemexportConvergenceTarget
  // port; the caller still owns completion gating, cache invalidation and
  // record retirement). It NEVER touches shared-buffer validity -- the buffer
  // is the authoritative source of the converged bytes.
  //
  // This lives here, on the shared memory that owns the backing, because it is
  // the only place that can PROVE whether destination and source are one
  // allocation. A caller deciding that from is_zero_copy() would be inferring a
  // physical fact from a mode; this returns what the operation actually did.
  // A claimed alias whose destination is not the source fails (converged
  // false), so a wrong alias verdict retains the record instead of silently
  // publishing nothing.
  SharedBufferConvergenceOutcome ConvergeRangeToRam(uint32_t base,
                                                    uint32_t length);

  // Upload copy accounting, updated at the memcpy / alias-skip site itself
  // (never from the mode). Saturating rather than wrapping: an overflowed
  // counter that read as a small number would be indistinguishable from
  // "almost no copying happened", which is exactly the claim the zero-copy
  // pre-gate has to measure.
  uint64_t upload_bytes_copied() const { return upload_bytes_copied_; }
  uint64_t upload_bytes_aliased() const { return upload_bytes_aliased_; }

  // Cache-clear release (implement-metal-cache-clearing step 5): the inherited
  // clear (which fires every watch and revalidates page state, keeping
  // GPU-written data) plus the Metal-only hazard records the base cannot know
  // about.
  void ClearCache() override;

 protected:
  bool UploadRanges(const std::pair<uint32_t, uint32_t>* upload_page_ranges,
                    uint32_t num_upload_ranges) override;

  // The guest-CPU write observation half of the overlap evidence. Invoked by
  // the base class's physical-invalidation THUNK only, so the backend's own
  // publications (which call MemoryInvalidationCallback directly) are not
  // counted as CPU writes. Runs on the faulting guest thread.
  void ObservePhysicalCpuWrite(uint32_t physical_address_start,
                               uint32_t length) override;

 private:
  TraceWriter* trace_writer_ = nullptr;
  void* device_;                        // id<MTLDevice>, not owned.
  void* buffer_ = nullptr;              // id<MTLBuffer>, retained.
  uint8_t* buffer_contents_ = nullptr;  // CPU-visible pointer into buffer_.
  bool is_zero_copy_ = false;
  ZeroCopyObservation zero_copy_observation_;
  uint64_t upload_bytes_copied_ = 0;
  uint64_t upload_bytes_aliased_ = 0;

  // Byte ranges consumed by draws encoded in the open frame command buffer.
  struct ConsumedRange {
    uint32_t first;   // start
    uint32_t second;  // end
    uint64_t submission;
  };
  std::vector<ConsumedRange> consumed_ranges_;
  std::function<uint64_t()> submission_provider_;
  std::function<void()> hazard_flush_callback_;

  // Fixed-size (512 MiB / host page) and installed ONLY in alias mode, which is
  // what makes every other arm structurally silent rather than silent by
  // convention. Arming it is a POSTCONDITION of accepting the alias, not a
  // consequence: if it cannot be armed the alias candidate is released and the
  // mirror is installed instead, because an unwatched alias would make "no
  // overlap was reported" mean "nobody was looking".
  ZeroCopyOverlapObserver zero_copy_overlap_;
  std::function<uint64_t()> overlap_submission_provider_;
  std::function<uint64_t()> overlap_completed_submission_provider_;
};

// Result of one guest-RAM alias attempt. `buffer` is a retained id<MTLBuffer>
// (or nullptr); the caller owns it and must transfer or release it.
struct ZeroCopyAliasAttempt {
  void* buffer = nullptr;
  ZeroCopyPrecondition precondition = ZeroCopyPrecondition::kNotEvaluated;
  ZeroCopyDeviceStep device_step = ZeroCopyDeviceStep::kNotAttempted;
};

// The whole alias decision for one (device, base, length) request: device-free
// preconditions, then the single-VM-region proof, then
// newBufferWithBytesNoCopy with shared storage and Metal's DEFAULT tracked
// hazard mode, then the contents-identity postcondition. Nothing is returned
// unless every step passed; a buffer that fails the postcondition is released
// here rather than handed back.
//
// Exposed rather than kept private to Initialize because the rejection classes
// are only reachable on a real device with hostile inputs (an unaligned base,
// a length past maxBufferLength, an mprotect-split region) that the production
// initializer, by construction, never produces.
ZeroCopyAliasAttempt TryCreateZeroCopyAliasBuffer(void* device, void* base,
                                                  uint64_t length,
                                                  bool force_rejection);

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_
