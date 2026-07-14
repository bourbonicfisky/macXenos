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

  // RT-as-texture bridge: register a resolved render-target snapshot (owned by
  // the render-target cache) at its guest resolve destination base. A later
  // texture fetch whose base matches samples that snapshot instead of the
  // (never-populated) guest memory, feeding deferred / composite passes. The
  // slot is allocated once per base and reused; the snapshot's contents update
  // in place, so re-registering the same base/texture is a cheap no-op.
  void RegisterResolvedTarget(uint32_t base, void* texture, uint32_t width,
                              uint32_t height, bool is_depth);

  // Stage 3 hybrid: a resolve at `base` enqueued a guest-RAM writeback, so
  // RAM is authoritative for fetches of this base (the bridge snapshot is a
  // fallback for dests the writer can't encode — e.g. 64bpp HDR).
  void MarkResolvedTargetRamBacked(uint32_t base);

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
  void ResetActive() { active_textures_.clear(); }

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
  uint32_t InvalidateRange(uint32_t base, uint32_t length);

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
  uint32_t view_heap_slots_;
  uint32_t next_slot_ = 1;  // 0 == null texture.
  // Slots freed when a resolved target was repointed to a new texture; the old
  // slot keeps pointing at the old texture until the submissions encoded
  // against it complete, then moves to free_slots_ for reuse. Tagged with the
  // last submission that may reference them.
  std::vector<std::pair<uint64_t, uint32_t>> stale_slots_;
  std::vector<uint32_t> free_slots_;
  // Textures evicted by InvalidateRange, released once their submissions
  // complete.
  std::vector<std::pair<uint64_t, void*>> retired_textures_;
  std::function<uint64_t()> submission_provider_;
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
    // Stage 3: DEPTH bridges and non-RAM-backed color bridges are served to
    // fetches; RAM-backed color reads guest RAM (exact-address placement).
    bool is_depth;
    // True once any resolve at this base enqueued a guest-RAM writeback: the
    // writer could encode the dest, so RAM (not the per-base snapshot) is
    // the truth. Stays false for dests the writer skips (64bpp HDR etc.),
    // which keep the bridge (Sonic's tonemap chain went flat gray without
    // this fallback under default-ON stage 3).
    bool ram_backed = false;
  };
  // Keyed by guest resolve destination base (fetch.base_address << 12).
  std::unordered_map<uint32_t, ResolvedTarget> resolved_targets_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_CACHE_H_
