/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_TEXTURE_CACHE_H_
#define XENIA_GPU_METAL_METAL_TEXTURE_CACHE_H_

#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "xenia/gpu/metal/metal_guest_range.h"
#include "xenia/gpu/metal/metal_resolved_fetch_service.h"
#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/xenos.h"

namespace xe {
class Memory;
namespace gpu {
class TraceWriter;
namespace metal {

// Bring-up minimal texture cache. Given a guest texture fetch constant, produces
// a host MTLTexture (CPU-untiled) and assigns it a slot in the bindless view
// heap, returning that slot so the draw path can write it into the b4
// descriptor-indices cbuffer. Supports only what's needed to get the first real
// textures on screen: 2D k_8_8_8_8 and DXT1/DXT2_3/DXT4_5 (hardware BC on
// Apple Silicon), linear or tiled. Everything else returns slot 0 (the
// null/gray texture).
//
// Deliberately NOT the proper xe::gpu::TextureCache subclass (no shared-memory
// watches, no LRU, no mips, no format table, re-parses each draw but
// caches the host texture by guest key). It exists to prove the texture path;
// the real TextureCache subclass (needs a SharedMemory subclass first) is the
// productionization step.
class MetalTextureCache {
 public:
  // device: id<MTLDevice>. view_heap: id<MTLBuffer> of IRDescriptorTableEntry
  // (the bindless view heap owned by the command processor). Slot 0 is reserved
  // for the null texture.
  MetalTextureCache(void* device, Memory* memory, SharedMemory* shared_memory,
                    void* view_heap, uint32_t view_heap_slots);
  ~MetalTextureCache();

  // Returns the bindless view-heap slot for the texture described by `fetch`,
  // uploading + caching the host texture on first use. Returns 0 (null) if the
  // format/dimension isn't supported yet or the fetch is invalid. Records the
  // resolved host texture in the active list for residency. `is_signed`
  // requests the SNORM view variant (signed fetches are separate translated
  // bindings expecting -1..1 texels).
  uint32_t GetTextureSlot(const xenos::xe_gpu_texture_fetch_t& fetch,
                          bool is_signed = false);

  // The ONE resolve-backed service classification for a fetch whose base may
  // be a registered resolve target. The serve chain in GetTextureSlot (and
  // any future consumer deciding how a resolve-backed fetch is served)
  // classifies through this — one decision, no duplicated exact-cover/bridge
  // guards. Stable within a draw (resolved_targets_ mutates only at
  // IssueCopy).
  ResolvedFetchService ClassifyFetchService(
      const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed) const;

  // Lineage probe (XE_METAL_LINEAGE): the command processor stamps the current
  // draw index here each draw so the reader's fetch events share the writer's
  // clock; only read when the probe is active.
  uint32_t lineage_draw_ = 0;

  // RT-as-texture bridge: register a resolved render-target snapshot (owned by
  // the render-target cache) at its guest resolve destination base. A later
  // texture fetch whose base matches samples that snapshot instead of the
  // (never-populated) guest memory, feeding deferred / composite passes. The
  // slot is allocated once per base and reused; the snapshot's contents update
  // in place, so re-registering the same base/texture is a cheap no-op.
  void RegisterResolvedTarget(uint32_t base, void* texture, uint32_t width,
                              uint32_t height, bool is_depth,
                              bool requires_fetch_swizzle,
                              uint32_t guest_extent_start,
                              uint32_t guest_extent_length);

  // Stage 3 hybrid: a resolve at `base` enqueued a guest-RAM writeback, so
  // RAM is authoritative for fetches of this base (the bridge snapshot is a
  // fallback for dests the writer can't encode — e.g. 64bpp HDR).
  void MarkResolvedTargetRamBacked(uint32_t base, uint32_t wb_width,
                                   uint32_t wb_height, bool full_rect_geometry,
                                   uint32_t wb_endian);
  // XE_METAL_TIP_SNAP_VIEW: attach the derived 1x1 texel-(0,0) copy to the
  // registered resolve target at base (allocates/repoints its bindless slot).
  void RegisterResolvedTipView(uint32_t base, void* texture);

  // True if a fetch of (base, width, height) will be served by the bridge
  // snapshot rather than guest RAM — the command processor skips the
  // pending-writeback flush (a full mid-frame commit+wait) for such fetches.
  // Must mirror the serving rule in the fetch path's bridge block (covered
  // fast path is R32Float only — see the comment there). Defined in the .mm
  // (needs the MTLTexture bridge cast).
  bool BridgeServes(uint32_t base, uint32_t width, uint32_t height) const;

  // Host textures (id<MTLTexture>) referenced since the last ResetActive, to be
  // made resident on the encoder.
  const std::vector<void*>& active_textures() const { return active_textures_; }
  void ResetActive() {
    active_textures_.clear();
    binding_allocation_failed_ = false;
  }
  bool binding_allocation_failed() const { return binding_allocation_failed_; }
  void ClearBindingAllocationFailure() { binding_allocation_failed_ = false; }

  // Appends EVERY live texture a bindless slot can reach (guest cache entries
  // + resolved targets) for the per-encoder residency sweep: a shader indexing
  // a slot whose texture was only made resident on an EARLIER encoder (stale
  // descriptor index, prior-generation slot) page-faults otherwise — the
  // attributed live faults ('draws@N' encoders) are exactly this class.
  void AppendAllTextures(std::vector<void*>& out) const;

  // Returns bindless slots retired by submissions <= `completed_submission`
  // (a resolved target got a new texture generation -> new slot) to the free
  // list, and releases textures retired by InvalidateRange in that window.
  // Entries retired while ENCODING a later, still-in-flight submission stay
  // retired — freeing them page-faults the in-flight draws (the async-S8
  // black flicker). Pass UINT64_MAX at shutdown.
  void ReclaimStaleSlots(uint64_t completed_submission);

  // Tags retires with the submission that may last reference them.
  void SetSubmissionProvider(std::function<uint64_t()> provider) {
    submission_provider_ = std::move(provider);
  }

  // Stage 3: a resolve wrote [base, base+length) of guest RAM — evict cached
  // textures overlapping it so the next fetch re-uploads fresh content.
  // Evicted textures/slots are retired (freed via ReclaimStaleSlots after the
  // frame's GPU work completes). Returns entries evicted.
  enum class InvalidationScope {
    // An independent RAM writer supersedes every cached representation of the
    // bytes, including an older resolve-backed snapshot bridge.
    kAllRepresentations,
    // The resolve publishing these bytes owns the matching snapshot, so only
    // RAM-derived uploads are stale; preserve the GPU-ordered bridge.
    kRamUploadsOnly,
  };
  uint32_t InvalidateRange(
      uint32_t base, uint32_t length,
      InvalidationScope scope = InvalidationScope::kAllRepresentations);

  // Trace playback restores guest ranges explicitly and calls InvalidateRange
  // for each restore. It doesn't need live CPU page watching, which would add
  // a redundant protected-page copy to every replayed texture upload.
  void SetTracePlayback();

  // Frame-trace recording: uploads record their guest source ranges into an
  // open trace; WriteAllCachedTexturesToTrace records the ranges of WARM
  // entries (they won't re-upload during the traced frame, but the replay
  // must still restore their data). Guest RAM holds their current content.
  void SetTraceWriter(TraceWriter* trace_writer) {
    trace_writer_ = trace_writer;
  }
  void WriteAllCachedTexturesToTrace(TraceWriter& trace_writer) const;

  // Telemetry: bindless-slot pressure (frame stats). Exhaustion is silent
  // at the allocation sites — surfacing it here caught the slot leak that
  // zeroed the exposure pyramid ~1 minute into every session.
  // The view-heap null texture (id<MTLTexture>, not owned). Initialization
  // binds EVERY slot to it so any index the shader can form is a valid
  // resource; a freed slot must be returned to that state rather than left
  // describing a released texture.
  void SetNullTexture(void* null_texture) { null_texture_ = null_texture; }

  // Cache-clear release (implement-metal-cache-clearing step 5). Frees every
  // cached texture, returns its bindless slot to the free list, rebinds the
  // freed view-heap descriptors to the null texture, and drains the
  // submission-gated retire lists outright.
  //
  // Direct release rather than retirement is correct ONLY because the deferred
  // clear runs at a fully quiesced boundary -- encoder closed, every
  // submission complete, completion actions pumped -- which the admission
  // predicate is what guarantees. Never call this from anywhere else.
  //
  // Call ReleaseDerivedResolveBridges BEFORE this: it retires bridge slots and
  // views into the same pools this drains.
  void ClearCache();

  // Cache-clear step: release the resolve-snapshot bridges that are DERIVED,
  // and report (destination base, snapshot texture) for each so the
  // render-target cache can release exactly the texture THIS bridge named.
  //
  // The texture identity is reported, not just the base, because
  // EnsureResolveSnapshot replaces the texture at a key when a later resolve
  // needs a larger one (retiring the old to the deferred list). A bridge can
  // therefore hold an older pointer than the snapshot currently at its base,
  // and releasing by base alone would destroy the NEWER snapshot that the
  // present path or another consumer may still be using.
  //
  // Classification:
  //   RELEASED  generation_ram_backed && !is_depth -- a guest-RAM writeback
  //             for THIS snapshot generation has been enqueued, and the clear
  //             publishes every pending writeback before this runs, so guest
  //             RAM holds the content. These are exactly the bridges whose
  //             serve path already reads RAM rather than the snapshot.
  //   RETAINED  !generation_ram_backed -- this generation's content was never
  //             written to guest RAM (either no writeback was ever enqueued
  //             for the destination -- the 64bpp/HDR dests the writer skips --
  //             or a newer snapshot replaced the one that was written back).
  //             The snapshot is then the ONLY copy of that content, and
  //             releasing it would destroy guest-visible data.
  //   RETAINED  is_depth -- depth bridges are served to fetches from the
  //             snapshot even when a writeback exists, so releasing one would
  //             change which content a fetch observes.
  // After a clear, every remaining bridge satisfies
  // (!generation_ram_backed || is_depth).
  void ReleaseDerivedResolveBridges(
      std::vector<std::pair<uint32_t, void*>>* released_out);

  // Replay-boundary retirement (fix-metal-trace-replay-state-reset, carrier
  // C2): retires EVERY resolved-target bridge -- slots, tip slot, and
  // cache-owned swizzle views into the retire pools ClearCache drains -- and
  // erases the map, with NO live retention rules. The clear path's retention
  // above ("the snapshot is the only copy of guest-visible content") is
  // exactly inverted under replay restoration: authoritative RAM + EDRAM are
  // about to be re-delivered by the trace, so a snapshot that is the only
  // copy of PRE-RESTORE content is precisely the state that must die.
  // Snapshot/tip textures are owned elsewhere and are released by their
  // owners at the same seam. Call ONLY from the quiesced replay-reset
  // boundary (MetalCommandProcessor::ResetDerivedReplayState), followed by
  // ClearCache() to drain what this retires.
  void RetireAllResolvedTargets();
  // Post-clear consistency evidence (a clear must return these toward their
  // post-initialization baseline, and leave nothing queued).
  uint32_t cached_texture_count() const { return uint32_t(cache_.size()); }
  uint32_t retired_texture_count() const {
    return uint32_t(retired_textures_.size());
  }
  uint32_t resolved_bridge_count() const {
    return uint32_t(resolved_targets_.size());
  }
  // True when every retained bridge satisfies the retention invariant above.
  bool retained_bridges_are_authoritative() const {
    for (const auto& pair : resolved_targets_) {
      if (pair.second.generation_ram_backed && !pair.second.is_depth) {
        return false;
      }
    }
    return true;
  }

  uint32_t slots_used() const { return next_slot_; }
  uint32_t slots_free() const { return uint32_t(free_slots_.size()); }
  uint32_t slots_stale() const { return uint32_t(stale_slots_.size()); }
  uint32_t slots_capacity() const { return view_heap_slots_; }


  // Telemetry: uploads performed since the last query (frame stats).
  uint32_t ConsumeUploadCount() {
    uint32_t n = upload_count_;
    upload_count_ = 0;
    return n;
  }

  // --- C1T-R shared-subclass adapter (single slot/residency authority) ---
  // MetalSharedTextureCache allocates, binds, and retires bindless view-heap
  // slots and registers per-draw residency exclusively through these; a
  // second allocator over the same heap is forbidden (seam audit, proof 1).
  uint32_t AllocateExternalSlot() { return AllocateSlot(); }
  // Points `slot` at a retained id<MTLTexture> the caller keeps alive.
  void BindExternalSlot(uint32_t slot, void* texture_bridge);
  void RetireExternalSlot(uint32_t slot) {
    stale_slots_.emplace_back(RetireSubmission(), slot);
  }
  // Takes ownership of a retained id<MTLTexture> (+1); released once the
  // submissions that may reference it complete (same pool as evictions).
  void RetireExternalTexture(void* texture_bridge) {
    retired_textures_.emplace_back(RetireSubmission(), texture_bridge);
  }
  // Per-draw residency registration for a texture served by the subclass.
  void AddActiveTexture(void* texture_bridge) {
    active_textures_.push_back(texture_bridge);
  }

 private:
  static void SharedMemoryGlobalWatchCallback(
      const global_unique_lock_type& global_lock, void* context,
      uint32_t address_first, uint32_t address_last, bool invalidated_by_gpu);
  void QueueInvalidation(uint32_t address_first, uint32_t address_last);
  void DrainPendingInvalidations();

  // Allocates a bindless view-heap slot, reusing a reclaimed one if available.
  // Returns 0 (the null slot) if the heap is exhausted.
  uint32_t AllocateSlot();

  void* device_;   // id<MTLDevice>, not owned.
  Memory* memory_;  // not owned.
  SharedMemory* shared_memory_;  // not owned.
  SharedMemory::GlobalWatchHandle shared_memory_watch_ = nullptr;
  std::mutex pending_invalidations_mutex_;
  // Guest CPU writes may arrive from any emulated thread. The shared-memory
  // callback only queues their ranges; cache mutation and Metal object
  // retirement stay on the GPU thread in DrainPendingInvalidations.
  std::vector<std::pair<uint32_t, uint32_t>> pending_invalidations_;
  bool trace_playback_ = false;
  void* view_heap_;  // id<MTLBuffer>, not owned.
  void* view_heap_contents_;  // IRDescriptorTableEntry* (CPU pointer).
  void* null_texture_ = nullptr;  // id<MTLTexture>, not owned; may be null.
  uint32_t view_heap_slots_;
  uint32_t next_slot_ = 1;  // 0 == null texture.
  // Distinguishes an intentional null descriptor (unsupported/invalid guest
  // fetch) from failure to create a required Metal object or bindless slot.
  // IssueDraw skips only the latter before emission.
  bool binding_allocation_failed_ = false;
  // Slots freed when a resolved target was repointed to a new texture; the old
  // slot keeps pointing at the old texture until the submissions encoded
  // against it complete, then moves to free_slots_ for reuse. Tagged with the
  // last submission that may reference them.
  // Single writer for every view-heap descriptor. All nine former direct
  // IRDescriptorTableSetTexture sites route through this, so the descriptor
  // write has exactly one implementation.
  void WriteSlotDescriptor(uint32_t slot, void* texture_bridge);
  std::vector<std::pair<uint64_t, uint32_t>> stale_slots_;
  std::vector<uint32_t> free_slots_;
  // Textures evicted by InvalidateRange, released once their submissions
  // complete.
  std::vector<std::pair<uint64_t, void*>> retired_textures_;
  std::function<uint64_t()> submission_provider_;
  bool BuildResolvedServiceInputs(const xenos::xe_gpu_texture_fetch_t& fetch,
                                  bool is_signed,
                                  ResolvedFetchServiceInputs* out) const;
  // Shared finalize tail (swizzle view, slot, trace record, cache entry) —
  // the ONE place upload finalization semantics live. texture_bridge:
  // id<MTLTexture> at +0. host_format: MTLPixelFormat.
  uint32_t FinalizeTexture(uint64_t key,
                           const xenos::xe_gpu_texture_fetch_t& fetch,
                           void* texture_bridge, uint64_t host_format,
                           bool is_cube, uint32_t host_levels, uint32_t base,
                           uint32_t guest_bytes, uint32_t mip_base,
                           uint32_t mip_bytes);
  uint64_t RetireSubmission() const {
    return submission_provider_ ? submission_provider_() : UINT64_MAX;
  }
  uint32_t upload_count_ = 0;
  TraceWriter* trace_writer_ = nullptr;  // not owned; may be null.
  bool bc_supported_ = false;  // Apple Silicon Metal decodes BC1/2/3 in HW.

  struct Entry {
    void* texture;  // id<MTLTexture>, retained.
    uint32_t slot;
    // Guest byte range the texture was uploaded from — resolve writebacks
    // into this range invalidate the entry (stage 3: fetches read guest RAM).
    uint32_t guest_base;
    uint32_t guest_bytes;
    // Mip range (0/0 when single-level) for invalidation + trace recording.
    uint32_t mip_base = 0;
    uint32_t mip_bytes = 0;
  };
  std::unordered_map<uint64_t, Entry> cache_;
  std::vector<void*> active_textures_;

  struct ResolvedTarget {
    void* texture;  // id<MTLTexture>, owned by the render-target cache.
    uint32_t width;
    uint32_t height;
    uint32_t slot;
    // Exact normalized guest destination extent of the resolve represented by
    // this snapshot. Used to retire the bridge when a later independent RAM
    // writer overlaps any byte, including a partial/aliased write.
    uint32_t guest_extent_start;
    uint32_t guest_extent_length;
    // Stage 3: DEPTH bridges and non-RAM-backed color bridges are served to
    // fetches; RAM-backed color reads guest RAM (exact-address placement).
    bool is_depth;
    // True when the snapshot contains guest resolve-destination channel
    // representation, so sampling it must apply the texture fetch swizzle just
    // like the RAM upload path. Legacy resolve snapshots are already logical
    // host color and must not be decoded a second time.
    bool requires_fetch_swizzle;
    // True once any resolve at this base enqueued a guest-RAM writeback: the
    // writer could encode the dest, so RAM (not the per-base snapshot) is
    // the truth. Stays false for dests the writer skips (64bpp HDR etc.),
    // which keep the bridge (Sonic's tonemap chain went flat gray without
    // this fallback under default-ON stage 3).
    bool ram_backed = false;
    // Cache-clear classification ONLY (implement-metal-cache-clearing): does
    // the CURRENT content generation's data exist in guest RAM?
    //
    // Distinct from ram_backed, which is sticky for the life of the base and
    // drives the serve path, and reset on a BROADER condition than full_rect:
    // every successful RegisterResolvedTarget clears this, whereas full_rect
    // resets only when the texture or extent changes. The conditions differ on
    // purpose. A resolve reuses the existing snapshot object whenever it is
    // large enough, so ResolveColorToSnapshot can overwrite the content in
    // place with the pointer and dimensions unchanged -- still a new
    // generation, and registration happens before that resolve's writeback
    // decision.
    //
    // Without this, a base whose earlier resolve wrote back and whose later
    // resolve skipped or failed its writeback would still read
    // ram_backed == true, and a clear would release a snapshot holding content
    // guest RAM never received.
    bool generation_ram_backed = false;
    // XE_METAL_TIP_SNAP_VIEW: derived 1x1 copy of the snapshot's texel (0,0),
    // blitted at resolve time by the command processor (which owns the
    // texture). A 1x1 guest fetch must read guest texel (0,0), but binding
    // the full WxH snapshot puts the shader's UV mid-texture (8x8 tip ->
    // texel 4,4) — the JC2 exposure-darkness mechanism.
    void* tip_texture = nullptr;  // id<MTLTexture>, owned by the CP.
    uint32_t tip_slot = 0;
    // Exact-cover reader: true while the NEWEST guest-RAM writeback at this
    // base wrote the full registered extent at offset (0,0) with
    // pitch == width. RAM then holds scatter(snapshot) — any channel swap
    // is baked into the snapshot by the resolve shader itself, and equal
    // fetch/writeback endian makes untile cancel the scatter — so for a
    // fetch of the same extent and pitch the snapshot IS the RAM upload's
    // content and can be served directly. Offset/partial resolves (SF4's
    // column writes) clear this and keep the exact-address RAM path.
    bool full_rect = false;
    uint32_t wb_endian = 0;
    // Per-swizzle sampled views for the exact-cover fast path. Serving the raw
    // snapshot slot (rt.slot) bypasses the fetch swizzle the ordinary RAM
    // upload bakes into a texture view, so a nonidentity guest swizzle would
    // read the wrong channels. Identity RGBA reuses the raw slot; every other
    // 12-bit guest swizzle gets its own channel-swizzled MTLTexture view in a
    // fresh bindless slot, cached here. The views are ours (retained +1);
    // retired wholesale (slot -> stale_slots_, view -> retired_textures_) when
    // the snapshot texture generation changes, and released at destruction.
    struct SwizzleView {
      void* view;  // id<MTLTexture>, retained by us.
      uint32_t slot;
    };
    std::unordered_map<uint32_t, SwizzleView> swizzle_views;
    // Per-swizzle sampled views of the derived 1x1 tip_texture (bridge 1x1
    // fetch path). Kept separate from swizzle_views because they alias a
    // DIFFERENT source object with its own lifetime: they retire when the tip
    // texture generation changes (RegisterResolvedTipView), not when the full
    // snapshot changes.
    std::unordered_map<uint32_t, SwizzleView> tip_swizzle_views;
  };
  // Keyed by guest resolve destination base (fetch.base_address << 12).
  std::unordered_map<uint32_t, ResolvedTarget> resolved_targets_;

  // Retires descriptors and cache-owned views without releasing the snapshot
  // or tip texture objects, whose ownership remains outside this cache.
  void RetireResolvedTarget(ResolvedTarget& target);

  // Returns the bindless slot for a channel-swizzled sampled view of `source`
  // (a resolved snapshot or its 1x1 tip), creating and caching it in `views`
  // keyed by the 12-bit guest swizzle, and marks it active for residency. The
  // caller must have already checked the swizzle is non-identity. Returns 0 on
  // view creation or heap exhaustion — the caller then picks the safe fallback
  // (RAM re-upload for RAM-backed color, null for bridge-only sources) rather
  // than bind the unswizzled raw slot with the wrong channels.
  uint32_t GetSwizzledSnapshotSlot(
      void* source, uint32_t guest_swizzle,
      std::unordered_map<uint32_t, ResolvedTarget::SwizzleView>& views);
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_CACHE_H_
