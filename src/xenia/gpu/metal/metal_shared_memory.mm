/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_shared_memory.h"

#include <algorithm>

#include "xenia/gpu/trace_writer.h"

#import <Metal/Metal.h>

#include <cstdlib>
#include <cstring>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/memory.h"

DECLARE_bool(gpu_allow_invalid_upload_range);

namespace xe {
namespace gpu {
namespace metal {

namespace {

// Apple's newBufferWithBytesNoCopy contract requires the wrapped allocation to
// occupy ONE virtual-memory region. QueryProtect is this project's wrapper over
// mach_vm_region (see base/memory_posix.cc): it reports the region containing
// `base` and, in `length`, the distance from `base` to that region's end. So
// the whole half-open interval is inside one region exactly when that distance
// covers it. Checked before the Metal call because a wrap that spans a region
// boundary is undefined rather than merely unsupported -- Metal is under no
// obligation to fail it.
bool IsSingleVmRegion(void* base, uint64_t length) {
  size_t region_remaining = 0;
  xe::memory::PageAccess access = xe::memory::PageAccess::kNoAccess;
  if (!xe::memory::QueryProtect(base, region_remaining, access)) {
    return false;
  }
  return uint64_t(region_remaining) >= length;
}

// Byte counters here are diagnostics that run for the whole life of a session
// and are read as evidence. A wrapped total would read as a small number, i.e.
// as "almost nothing was copied" -- the exact claim this pre-gate measures --
// so they saturate instead.
uint64_t SaturatingAdd(uint64_t accumulated, uint64_t added) {
  return added > UINT64_MAX - accumulated ? UINT64_MAX : accumulated + added;
}

}  // namespace

ZeroCopyAliasAttempt TryCreateZeroCopyAliasBuffer(void* device_handle,
                                                  void* base, uint64_t length,
                                                  bool force_rejection) {
  ZeroCopyAliasAttempt attempt;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;

  ZeroCopyContractInputs inputs;
  inputs.base_address = uint64_t(reinterpret_cast<uintptr_t>(base));
  inputs.length = length;
  inputs.host_page_size = uint64_t(xe::memory::page_size());
  inputs.device_max_buffer_length = device ? uint64_t([device maxBufferLength])
                                           : 0;
  inputs.force_rejection = force_rejection;
  attempt.precondition = EvaluateZeroCopyPreconditions(inputs);
  if (attempt.precondition != ZeroCopyPrecondition::kSatisfied) {
    return attempt;
  }

  if (!IsSingleVmRegion(base, length)) {
    attempt.device_step = ZeroCopyDeviceStep::kVmRegionRejected;
    return attempt;
  }

  // Shared storage with Metal's DEFAULT hazard tracking. Untracked hazards are
  // explicitly NOT taken here: the pre-gate changes where bytes live, not who
  // orders in-flight GPU work, and the driver's tracking is what keeps the
  // existing encoder ordering honest. No write-combined cache mode either --
  // this mapping is ordinary cached guest RAM the CPU reads, and describing it
  // otherwise would be a lie about memory Metal does not own.
  id<MTLBuffer> buffer =
      [device newBufferWithBytesNoCopy:base
                                length:size_t(length)
                               options:MTLResourceStorageModeShared |
                                       MTLResourceHazardTrackingModeDefault
                           deallocator:nil];
  if (!buffer) {
    attempt.device_step = ZeroCopyDeviceStep::kAllocationRejected;
    return attempt;
  }
  // The postcondition that makes this an alias at all (see
  // ZeroCopyContentsAreAlias).
  if (!ZeroCopyContentsAreAlias([buffer contents], base)) {
    attempt.device_step = ZeroCopyDeviceStep::kContentsRejected;
    buffer = nil;  // ARC releases the partial result; nothing escapes.
    return attempt;
  }
  attempt.device_step = ZeroCopyDeviceStep::kAccepted;
  attempt.buffer = (__bridge_retained void*)buffer;
  return attempt;
}

MetalSharedMemory::MetalSharedMemory(Memory& memory, void* device)
    : SharedMemory(memory), device_(device) {}

MetalSharedMemory::~MetalSharedMemory() { Shutdown(); }

bool MetalSharedMemory::Initialize() {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

  // Exact opt-in. Parsed once, here, and recorded independently of what the
  // attempt below concludes.
  zero_copy_observation_ = ZeroCopyObservation();
  zero_copy_observation_.requested =
      IsZeroCopyGateEnabled(std::getenv(kZeroCopyGateEnvVar));

  id<MTLBuffer> buffer = nil;
  if (zero_copy_observation_.requested) {
    ZeroCopyAliasAttempt attempt = TryCreateZeroCopyAliasBuffer(
        device_, memory().TranslatePhysical(0), kBufferSize,
        IsZeroCopyGateEnabled(std::getenv(kZeroCopyForceRejectEnvVar)));
    zero_copy_observation_.precondition = attempt.precondition;
    zero_copy_observation_.device_step = attempt.device_step;
    if (attempt.buffer) {
      id<MTLBuffer> candidate = (__bridge_transfer id<MTLBuffer>)attempt.buffer;
      // OBSERVER READINESS IS A POSTCONDITION OF ACCEPTING THE ALIAS.
      //
      // Mirroring guest RAM gives the backend a snapshot; aliasing removes it,
      // so an alias nobody is watching cannot be distinguished from an alias
      // nobody raced. Ignoring this result would leave is_zero_copy_ true over
      // an inactive, permanently empty observer, and the overlap report treats
      // an empty snapshot as the required silence -- i.e. the run would read as
      // clean evidence precisely because no evidence was being collected.
      //
      // So a failure here REJECTS the candidate rather than degrading it: the
      // alias buffer is released, the pre-existing mirror is allocated below,
      // and the verdict is the named observer_rejected step. It is deliberately
      // not fatal -- the exact fallback this pre-gate already relies on is
      // available, and failing the whole emulator over a diagnostic that is
      // default-off would be a worse answer than not aliasing.
      //
      // The table is allocated HERE, before InitializeCommon registers the
      // physical invalidation callback, because the faulting guest thread that
      // fills it may not allocate.
      if (zero_copy_overlap_.Initialize(
              kBufferSize, page_size_log2(),
              IsZeroCopyGateEnabled(
                  std::getenv(kZeroCopyForceObserverRejectEnvVar)))) {
        buffer = candidate;
      } else {
        // Leave nothing half-armed: an inactive observer is inert, so its
        // stamping and observation entries do nothing whatever a caller asks.
        zero_copy_overlap_.Shutdown();
        candidate = nil;  // ARC releases the alias; guest RAM is untouched.
        zero_copy_observation_.device_step =
            ZeroCopyDeviceStep::kObserverRejected;
      }
    }
  }

  if (!buffer) {
    // The pre-existing mirror allocation, unchanged: shared storage so uploads
    // are plain host memcpys (unified memory on Apple Silicon); write-combined
    // is fine because nothing reads it from the CPU.
    buffer = [device newBufferWithLength:kBufferSize
                                 options:MTLResourceStorageModeShared |
                                         MTLResourceCPUCacheModeWriteCombined];
    if (!buffer) {
      XELOGE("MetalSharedMemory: failed to allocate {} MiB shared buffer",
             kBufferSize / (1024 * 1024));
      return false;
    }
  }
  buffer_ = (__bridge_retained void*)buffer;
  buffer_contents_ = static_cast<uint8_t*>([buffer contents]);

  // Second, independent source for the verdict: re-derived from the buffer
  // actually installed, not from the attempt record. "It aliases" means the
  // installed bytes ARE guest RAM -- a claim the mirror allocation can never
  // satisfy, whatever the attempt believed.
  zero_copy_observation_.alias_buffer_installed =
      buffer_contents_ != nullptr &&
      buffer_contents_ == static_cast<uint8_t*>(memory().TranslatePhysical(0));
  is_zero_copy_ = zero_copy_observation_.alias_buffer_installed;

  // The observer is armed if and only if the INSTALLED backing aliases guest
  // RAM -- re-checked here rather than assumed, because the arming above keyed
  // off the attempt while is_zero_copy_ comes from the independent
  // re-derivation. If the two disagree, the mirror is what the GPU actually
  // reads, and an armed observer over a mirror would be alias-only work inside
  // a control arm. Disarm; the startup line reports the disagreement as
  // inconsistent_observation, which is the loud half of this pair.
  if (zero_copy_overlap_.active() && !is_zero_copy_) {
    zero_copy_overlap_.Shutdown();
  }

  // One bounded line, and only when the probe was requested: an exact-off run
  // must produce no zero-copy observation at all.
  if (zero_copy_observation_.requested) {
    ZeroCopyStartupReport report =
        DeriveZeroCopyStartupReport(zero_copy_observation_);
    XELOGI(
        "MetalSharedMemory: zero-copy requested, verdict {} "
        "(precondition {}, device step {})",
        ZeroCopyStartupReportName(report),
        ZeroCopyPreconditionName(zero_copy_observation_.precondition),
        ZeroCopyDeviceStepName(zero_copy_observation_.device_step));
  }

  InitializeCommon();
  return true;
}

void MetalSharedMemory::Shutdown() {
  ShutdownCommon();
  // Strictly AFTER ShutdownCommon, which unregisters the physical invalidation
  // callback: releasing the fixed stamp table while a guest thread could still
  // enter ObservePhysicalCpuWrite would be a use-after-free. Note the evidence
  // dies here, so any final report must be taken BEFORE the shared memory is
  // destroyed (MetalCommandProcessor::ShutdownContext does).
  zero_copy_overlap_.Shutdown();
  consumed_ranges_.clear();
  if (buffer_) {
    id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)buffer_;
    (void)released;
    buffer_ = nullptr;
    buffer_contents_ = nullptr;
  }
  // In alias mode the released buffer never owned the bytes (deallocator nil),
  // so guest RAM outlives it untouched.
  is_zero_copy_ = false;
  zero_copy_observation_ = ZeroCopyObservation();
}

bool MetalSharedMemory::RequestConsumedRange(uint32_t start, uint32_t length) {
  if (!length) {
    return true;
  }
  // RequestRange uploads only the pages invalidated since their last use; a
  // hazard flush (see UploadRanges) may run inside.
  bool result = RequestRange(start, length);
  // Task 5.1: enrol this range's HOST pages as claimed by the submission being
  // encoded right now. Two conditions, both load-bearing:
  //   * `is_zero_copy_` -- mirror mode must not even ASK the provider for a
  //     submission, because the pre-gate's control arms prove silence by
  //     counting provider calls, not by reading a counter a mirror run would
  //     leave at zero either way.
  //   * `result` -- a failed request served no consumer, so recording one would
  //     manufacture an enrollment that never happened and make later overlap
  //     counts unfalsifiable in the loud direction.
  // This is deliberately the REQUEST seam, so enrollment is conservative: a
  // draw that later fails still leaves the page claimed. Over-stamping makes
  // the stop signal too loud, which is the only safe direction for it to err.
  if (is_zero_copy_ && result) {
    // The provider is read BEFORE the lock: it runs on this same GPU thread and
    // only reads the encoding submission counter, so there is nothing for the
    // lock to order, and holding the callback lock across foreign code would
    // put an unbounded body inside the guest fault path's critical region.
    uint64_t submission =
        overlap_submission_provider_ ? overlap_submission_provider_() : 0;
    // THE SYNCHRONIZATION EDGE. The stamp is published here and consumed on the
    // faulting guest thread inside the physical invalidation callback. Relaxed
    // atomics on the slot alone give no happens-before between the two: a
    // callback that begins after this enrollment could still legally read the
    // old zero stamp and classify a genuinely raced page as idle -- silence in
    // the one direction this pre-gate may not err in.
    //
    // Taking the SAME lock the callback dispatcher already holds makes the
    // order causal. A callback that ran while holding it happened BEFORE this
    // enrollment and is entitled to see the page idle; a callback that acquires
    // it after this scope MUST see the stamp. The faulting thread takes no new
    // lock for this -- it already holds the only one involved -- and still does
    // just one immutable provider load plus bounded atomics.
    //
    // Strictly AFTER RequestRange, which may run a hazard flush (commit + wait)
    // and takes this region itself: holding it across that would invert nothing
    // but would park an unbounded GPU wait inside the guest's critical region.
    // Only the bounded atomic stamp is inside.
    auto invalidation_lock =
        memory().LockPhysicalMemoryInvalidationCallbacks();
    zero_copy_overlap_.StampConsumer(start, length, submission);
  }
  uint32_t end = start + length;
  uint64_t submission = submission_provider_ ? submission_provider_() : 0;
  for (auto& range : consumed_ranges_) {
    if (start >= range.first && end <= range.second) {
      // Already covered — but the NEWER submission now also reads it; keep
      // the range protected until the newest reader completes.
      range.submission = std::max(range.submission, submission);
      return result;
    }
  }
  consumed_ranges_.push_back(ConsumedRange{start, end, submission});
  return result;
}

void MetalSharedMemory::ReleaseConsumedRanges(uint64_t completed_submission) {
  consumed_ranges_.erase(
      std::remove_if(consumed_ranges_.begin(), consumed_ranges_.end(),
                     [&](const ConsumedRange& range) {
                       return range.submission <= completed_submission;
                     }),
      consumed_ranges_.end());
}

ZeroCopyOverlapSnapshot MetalSharedMemory::zero_copy_overlap_snapshot() const {
  // Same lock as the enrollment and the callback, so what the reporter prints
  // is one coherent moment instead of counters sampled at different instants
  // (which could show, say, more classified pages than observed callbacks and
  // read as a defect in the model). Bounded and allocation-free inside: a
  // fixed-size struct copy, no logging -- the caller formats after this
  // returns, never inside the guest fault path's critical region.
  auto invalidation_lock = memory().LockPhysicalMemoryInvalidationCallbacks();
  return zero_copy_overlap_.Snapshot();
}

void MetalSharedMemory::ObservePhysicalCpuWrite(
    uint32_t physical_address_start, uint32_t length) {
  // Alias guard (task 5.2). Exact-off and every fallback arm return here
  // BEFORE any provider is invoked, so "the control arm did no zero-copy work"
  // is observable at the provider rather than inferred from an empty counter.
  if (!is_zero_copy_) {
    return;
  }
  // ONE completed high-water mark for the whole callback, loaded once from the
  // independent completion source. Sampling it per page would let a submission
  // complete mid-range and split one physical write across two different
  // verdicts, so a page could be reported finished only because the loop
  // reached it late.
  //
  // Everything below this line runs on the faulting guest thread WITH the
  // global critical region already held (PhysicalHeap::TriggerCallbacks walks
  // the callback list under it), i.e. on the critical path of every guest
  // fault: one acquire load through an immutable provider, then bounded atomic
  // work in the observer. No wait, no log, no allocation, NO NEW LOCK, no
  // command-processor mutation, no other command-processor API.
  //
  // The stamps this reads are published under that SAME already-held lock by
  // RequestConsumedRange, so the ordering is causal rather than hopeful: this
  // callback either ran before an enrollment (and may see the page idle, which
  // was true when it ran) or acquired the lock after it (and must see the
  // stamp). Nothing extra is paid here for that edge -- the lock is already in
  // hand -- which is why the producer, not the consumer, is the side that had
  // to change.
  uint64_t completed_submission = overlap_completed_submission_provider_
                                      ? overlap_completed_submission_provider_()
                                      : 0;
  zero_copy_overlap_.ObserveCpuWrite(physical_address_start, length,
                                     completed_submission);
}

void MetalSharedMemory::ClearCache() {
  // Hazard records name byte ranges consumed by draws in a submission, so
  // their only purpose is protecting encoded-but-unfinished readers. At the
  // clear boundary there are none, and ReleaseConsumedRanges (driven by the
  // pumped completion actions) should already have emptied the set -- which
  // the caller VERIFIES before calling this, so clearing here cannot erase
  // the evidence for that check.
  ResetFrameRanges();
  SharedMemory::ClearCache();
}

SharedBufferConvergenceOutcome MetalSharedMemory::ConvergeRangeToRam(
    uint32_t base, uint32_t length) {
  SharedBufferConvergenceOutcome outcome;
  if (!buffer_contents_) {
    // No backing at all (pre-Initialize / post-Shutdown). Report the factual
    // disagreement so the caller retains its pending record rather than
    // retiring bytes that were never published.
    return outcome;
  }
  // Domain checks BEFORE any pointer arithmetic. `base + length` is formed
  // nowhere below in a way that could wrap: the subtraction is on values
  // already proven to be in range. A zero length is rejected rather than
  // silently "succeeding" -- an empty publication that reported converged would
  // retire a record whose bytes nothing ever moved, and the caller cannot tell
  // that apart from a real alias skip.
  if (!length || base >= kBufferSize || length > kBufferSize - base) {
    return outcome;
  }
  uint8_t* destination = static_cast<uint8_t*>(memory().TranslatePhysical(base));
  if (!destination) {
    return outcome;
  }
  const uint8_t* source = buffer_contents_ + base;
  if (is_zero_copy_) {
    // A claimed alias is not evidence. Skipping the copy is licensed only by
    // the destination BEING the source at this exact range; if the two
    // disagree, the copy would be a real publication and suppressing it would
    // silently lose the export, so this fails and the caller retries.
    if (!ZeroCopySelfCopyIsProven(is_zero_copy_, destination, source)) {
      return outcome;
    }
    outcome.alias_skipped = true;
    outcome.bytes_copied = 0;  // proven self-copy: no byte moved.
    outcome.converged = true;
    return outcome;
  }
  std::memcpy(destination, source, length);
  outcome.bytes_copied = length;  // recorded AT the copy that moved them.
  outcome.converged = true;
  return outcome;
}

void MetalSharedMemory::WriteTraceInitialState(TraceWriter& trace_writer) {
  PrepareForTraceDownload();
  for (const auto& range : trace_download_ranges()) {
    trace_writer.WriteMemoryRead(range.first, range.second);
  }
  ReleaseTraceDownloadRanges();
}

bool MetalSharedMemory::UploadRanges(
    const std::pair<uint32_t, uint32_t>* upload_page_ranges,
    uint32_t num_upload_ranges) {
  if (!num_upload_ranges) {
    return true;
  }
  if (!buffer_contents_) {
    return false;
  }
  // Only invalidated pages reach here, so any overlap with a range a draw in
  // the open frame command buffer consumed is a genuine intra-frame rewrite:
  // flush (commit + wait) BEFORE the new bytes land so those draws read the
  // old data. The callback also resets consumed_ranges_.
  bool hazard = false;
  // Experiment (XE_METAL_FLUSH_ALL_UPLOADS): flush before ANY mid-frame
  // invalidating upload once a draw has consumed shared memory this frame,
  // ignoring the overlap test. Over-approximates per-draw serialization for the
  // shared-memory path (no arena coupling) — an in-agent A/B for whether the
  // stipple is shared-memory upload staleness at all.
  static const bool s_flush_all =
      std::getenv("XE_METAL_FLUSH_ALL_UPLOADS") != nullptr;
  if (s_flush_all && !consumed_ranges_.empty()) {
    hazard = true;
  }
  for (uint32_t i = 0; i < num_upload_ranges && !hazard; ++i) {
    uint32_t range_start = upload_page_ranges[i].first << page_size_log2();
    uint32_t range_end = range_start
                         + (upload_page_ranges[i].second << page_size_log2());
    for (const auto& consumed : consumed_ranges_) {
      if (range_start < consumed.second && range_end > consumed.first) {
        hazard = true;
        break;
      }
    }
  }
  if (hazard && hazard_flush_callback_) {
    static uint32_t s_hazard_flushes = 0;
    if (s_hazard_flushes++ < 8) {
      XELOGI("MetalSharedMemory: intra-frame rewrite hazard flush #{}",
             s_hazard_flushes);
    }
    hazard_flush_callback_();
  }
  for (uint32_t i = 0; i < num_upload_ranges; ++i) {
    uint32_t upload_start = upload_page_ranges[i].first << page_size_log2();
    uint32_t upload_length = upload_page_ranges[i].second << page_size_log2();
    if (!upload_length) {
      continue;
    }
    if (!cvars::gpu_allow_invalid_upload_range) {
      uint32_t last_page_start =
          upload_start + upload_length - (1u << page_size_log2());
      if (memory().GetPhysicalHeap()->QueryRangeAccess(
              upload_start, upload_start) == xe::memory::PageAccess::kNoAccess ||
          memory().GetPhysicalHeap()->QueryRangeAccess(
              last_page_start, last_page_start) ==
              xe::memory::PageAccess::kNoAccess) {
        XELOGE("MetalSharedMemory: invalid upload range {:08X}+{:X}",
               upload_start, upload_length);
        return false;
      }
    }
    // Validate BEFORE the memcpy so a CPU write racing the copy re-invalidates
    // the pages rather than being lost (base-class contract). Alias mode does
    // not change this: the pages are equally "no longer owed an upload", and
    // moving the validation inside the copy branch would leave alias mode
    // re-walking the same pages forever.
    MakeRangeValid(upload_start, upload_length, false);
    // The ONLY alias-mode difference (task 4.1). The CPU already wrote these
    // bytes into the backing the GPU reads, so the copy below would have
    // identical source and destination -- undefined as a memcpy and pointless
    // as work. Validity above and trace recording below are deliberately
    // OUTSIDE this branch: neither is a copy, and both are still owed.
    if (is_zero_copy_) {
      upload_bytes_aliased_ =
          SaturatingAdd(upload_bytes_aliased_, upload_length);
    } else {
      std::memcpy(buffer_contents_ + upload_start,
                  memory().TranslatePhysical(upload_start), upload_length);
      // Counted AT the copy, so the number is what moved, not what the mode
      // predicts would move.
      upload_bytes_copied_ =
          SaturatingAdd(upload_bytes_copied_, upload_length);
    }
    // Frame-trace recording: the uploaded source bytes are exactly what a
    // replay must restore before this range's consumers run. No-op unless a
    // trace file is open.
    if (trace_writer_) {
      trace_writer_->WriteMemoryRead(upload_start, upload_length);
    }
  }
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
