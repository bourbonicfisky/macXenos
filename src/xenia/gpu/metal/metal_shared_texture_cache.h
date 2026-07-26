/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHARED_TEXTURE_CACHE_H_
#define XENIA_GPU_METAL_METAL_SHARED_TEXTURE_CACHE_H_

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "xenia/gpu/register_file.h"
#include "xenia/gpu/texture_cache.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
class TraceWriter;
namespace metal {

class MetalSharedMemory;
class MetalTextureCache;

// C1T-R (OpenSpec implement-metal-gpu-texture-loads): the real
// xe::gpu::TextureCache subclass serving ONLY strict tiled single-mip 2D
// k_8_8_8_8 kRamContent cache misses through the proven direct-to-tiled
// compute load (texture_load_32bpb_direct) — the canonical route for that
// class. The legacy MetalTextureCache remains the
// universal fallback and the single bindless-slot authority (this class
// allocates/retires slots and registers residency through its narrow
// external-slot adapter). Seam contract: docs/
// metal_texture_cache_integration_adr.md "C1T-R Phase 0.2 seam audit" —
// classification stays ClassifyResolvedFetch + the eligibility header;
// preparation runs at the pre-encoder post-overlap/drain seam on the
// existing frame command buffer; serving is keyed by fetch-constant index
// against the per-draw eligible mask the command processor computed, so
// prep-time and serve-time can never disagree.
class MetalSharedTextureCache : public TextureCache {
 public:
  MetalSharedTextureCache(const RegisterFile& register_file,
                          MetalSharedMemory& shared_memory, void* device,
                          MetalTextureCache* legacy_adapter);
  ~MetalSharedTextureCache() override;

  // Builds the direct-load pipeline. Returns false (route unusable, callers
  // must not create the instance's route) on any failure.
  bool Initialize();

  // Command-processor wiring (all GPU-thread).
  // Returns the frame id<MTLCommandBuffer> at a pass boundary (the existing
  // PrepareTransferCommandBuffer), or nullptr.
  void SetPrepareTransferCallback(std::function<void*()> callback) {
    prepare_transfer_ = std::move(callback);
  }
  // The submission id the currently-open frame command buffer will get.
  void SetSubmissionProvider(std::function<uint64_t()> provider) {
    submission_provider_ = std::move(provider);
  }
  // True if the given submission faulted (lost command buffer): content
  // produced by GPU work in it is garbage and must not be served (M4).
  void SetSubmissionFaultQuery(std::function<bool(uint64_t)> query) {
    submission_fault_query_ = std::move(query);
  }

  // Pre-encoder prepass (after the overlap commit/wait/drain gateway, before
  // BeginFrameEncoder): forces re-parse of the masked fetch constants (this
  // backend has no register-write notification plumbing; per-draw re-parse
  // matches the legacy cache's cost class and the base key comparison
  // prevents redundant loads) and runs the base RequestTextures flow, whose
  // LoadTextureDataFromResidentMemoryImpl encodes the direct dispatch on the
  // frame command buffer.
  void PrepareEligible(uint32_t eligible_mask);

  // Serve at descriptor-marshal time: the bindless slot for the prepared
  // fetch-constant index, or 0 when unserved (caller falls back to the
  // legacy path). On success marks the texture used and registers encoder
  // residency through the legacy adapter.
  uint32_t GetServedSlot(uint32_t fetch_constant_index);

  // M4 (mandatory): called by the command processor when fault publication
  // records lost submissions. Every texture whose last load rode one of
  // them is invalidated THROUGH the shared-memory authority (watch fires,
  // texture goes outdated, pages revalidate from intact RAM, next use
  // reloads) — eager and independent of the bounded faulted-submission set
  // the serve-time query consults as defense in depth.
  void OnSubmissionsFaulted(const std::vector<uint64_t>& submissions);

  // Trace capture (M5, revised by gate-C evidence): the route DISENGAGES for
  // the captured frame (the command processor's prepass checks
  // trace_writer_.is_open()), so the legacy CPU path serves and records the
  // exact CPU-oracle read-set during the frame. The one remaining
  // contribution is this warm sweep — the exact analog of the legacy
  // warm-entry sweep for textures the subclass owns: it MUST run BEFORE
  // MetalSharedMemory::WriteTraceInitialState (whose page invalidation
  // fires the watches and outdates everything) and records only textures
  // that were up-to-date at that point, with the legacy extent expression.
  void WriteAllCachedTexturesToTrace(TraceWriter& trace_writer);

  // Residency sweep companion of MetalTextureCache::AppendAllTextures:
  // every texture a live subclass-bound slot can reach.
  void AppendAllTextures(std::vector<void*>& out) const;

  // Cache-clear release (implement-metal-cache-clearing step 5).
  //
  // The inherited TextureCache::ClearCache is NOT sufficient here, which is
  // why this override exists. The base clear resets the bindings and calls
  // UpdateTextureBindingsImpl, whose implementation PARKS rather than releases
  // a binding whose texture is gone -- correct for the per-swap BeginFrame
  // reset it was written for -- and then destroys the textures. That would
  // leave every slot holding a dangling Texture*, a retained view over a
  // destroyed base texture, and a legacy bindless slot still bound to it.
  //
  // ORDERING: this releases slots and views back into the legacy cache's
  // submission-gated pools, so it MUST run before MetalTextureCache::ClearCache
  // drains those pools. Reversing the two would strand the retirements in a
  // pool that was already emptied.
  void ClearCache() override;
  // Post-clear evidence: how many fetch-constant slots still express a
  // binding (a texture, a view, or a legacy slot).
  uint32_t bound_slot_count() const;

 protected:
  bool IsSignedVersionSeparateForFormat(TextureKey key) const override {
    // k_8_8_8_8 signed is a view over the same bytes; signed bindings are
    // route-ineligible anyway.
    return false;
  }
  uint32_t GetHostFormatSwizzle(TextureKey key) const override {
    // The direct load produces guest channel order in RGBA8Unorm.
    return xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }
  uint32_t GetMaxHostTextureWidthHeight(
      xenos::DataDimension dimension) const override {
    return xenos::kTexture2DCubeMaxWidthHeight;
  }
  uint32_t GetMaxHostTextureDepthOrArraySize(
      xenos::DataDimension dimension) const override {
    return 1;  // Routed class is plain 2D only.
  }
  std::unique_ptr<Texture> CreateTexture(TextureKey key) override;
  bool LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                             bool load_mips) override;
  void UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) override;

 private:
  class MetalTexture;

  // Per-fetch-constant slot state mirroring the base texture_bindings_:
  // which texture+swizzle the bound descriptor currently expresses.
  struct BindingSlot {
    const Texture* texture = nullptr;
    uint32_t host_swizzle = 0;
    void* sampled_view = nullptr;  // id<MTLTexture> (2DArray view), retained.
    uint32_t slot = 0;             // 0 = unbound.
  };
  void ReleaseBindingSlot(BindingSlot& binding_slot);

  void* device_;                        // id<MTLDevice>, not owned.
  MetalTextureCache* legacy_;           // slot/residency adapter, not owned.
  void* direct_pipeline_ = nullptr;     // id<MTLComputePipelineState>, retained.
  std::function<void*()> prepare_transfer_;
  std::function<uint64_t()> submission_provider_;
  std::function<bool(uint64_t)> submission_fault_query_;
  uint64_t last_begun_submission_ = 0;

  BindingSlot binding_slots_[xenos::kTextureFetchConstantCount];

  // Exact consumed source range per shape (Tiled2D min/max scan is ~1 call
  // per 4 texels; cache it like the proven slice-3 route did).
  struct ShapeKey {
    uint32_t w, h, pitch;
    bool operator==(const ShapeKey& o) const {
      return w == o.w && h == o.h && pitch == o.pitch;
    }
  };
  struct ShapeKeyHasher {
    size_t operator()(const ShapeKey& k) const {
      return (size_t(k.w) * 0x9E3779B1u) ^ (size_t(k.h) << 16) ^ k.pitch;
    }
  };
  struct ShapeRange {
    uint32_t begin_off, end_off;
    bool valid;
  };
  std::unordered_map<ShapeKey, ShapeRange, ShapeKeyHasher> shape_ranges_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHARED_TEXTURE_CACHE_H_
