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
#include <unordered_map>
#include <vector>

#include "xenia/gpu/xenos.h"

namespace xe {
class Memory;
namespace gpu {
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
  MetalTextureCache(void* device, Memory* memory, void* view_heap,
                    uint32_t view_heap_slots);
  ~MetalTextureCache();

  // Returns the bindless view-heap slot for the texture described by `fetch`,
  // uploading + caching the host texture on first use. Returns 0 (null) if the
  // format/dimension isn't supported yet or the fetch is invalid. Records the
  // resolved host texture in the active list for residency.
  uint32_t GetTextureSlot(const xenos::xe_gpu_texture_fetch_t& fetch);

  // RT-as-texture bridge: register a resolved render-target snapshot (owned by
  // the render-target cache) at its guest resolve destination base. A later
  // texture fetch whose base matches samples that snapshot instead of the
  // (never-populated) guest memory, feeding deferred / composite passes. The
  // slot is allocated once per base and reused; the snapshot's contents update
  // in place, so re-registering the same base/texture is a cheap no-op.
  void RegisterResolvedTarget(uint32_t base, void* texture, uint32_t width,
                              uint32_t height);

  // Host textures (id<MTLTexture>) referenced since the last ResetActive, to be
  // made resident on the encoder.
  const std::vector<void*>& active_textures() const { return active_textures_; }
  void ResetActive() { active_textures_.clear(); }

  // Returns bindless slots retired this frame (a resolved target got a new
  // texture generation -> new slot) to the free list. Safe only after the
  // frame's GPU work has completed: earlier draws still referenced the retired
  // slot. Called by the command processor after commit + wait.
  void ReclaimStaleSlots();

 private:
  // Allocates a bindless view-heap slot, reusing a reclaimed one if available.
  // Returns 0 (the null slot) if the heap is exhausted.
  uint32_t AllocateSlot();

  void* device_;   // id<MTLDevice>, not owned.
  Memory* memory_;  // not owned.
  void* view_heap_;  // id<MTLBuffer>, not owned.
  void* view_heap_contents_;  // IRDescriptorTableEntry* (CPU pointer).
  uint32_t view_heap_slots_;
  uint32_t next_slot_ = 1;  // 0 == null texture.
  // Slots freed when a resolved target was repointed to a new texture; the old
  // slot keeps pointing at the old texture until in-flight draws drain, then
  // moves to free_slots_ for reuse (bounds heap growth without in-place update).
  std::vector<uint32_t> stale_slots_;
  std::vector<uint32_t> free_slots_;
  bool bc_supported_ = false;  // Apple Silicon Metal decodes BC1/2/3 in HW.

  struct Entry {
    void* texture;  // id<MTLTexture>, retained.
    uint32_t slot;
  };
  std::unordered_map<uint64_t, Entry> cache_;
  std::vector<void*> active_textures_;

  struct ResolvedTarget {
    void* texture;  // id<MTLTexture>, owned by the render-target cache.
    uint32_t width;
    uint32_t height;
    uint32_t slot;
  };
  // Keyed by guest resolve destination base (fetch.base_address << 12).
  std::unordered_map<uint32_t, ResolvedTarget> resolved_targets_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_CACHE_H_
