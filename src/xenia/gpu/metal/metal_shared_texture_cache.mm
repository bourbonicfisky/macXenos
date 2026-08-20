/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_shared_texture_cache.h"

#import <Metal/Metal.h>

#include "xenia/base/math.h"
#include "xenia/gpu/metal/metal_load_shader_pipelines.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/metal/metal_mip_level_plan.h"
#include "xenia/gpu/metal/metal_texture_load_eligibility.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {

// Store convention: the direct load writes the 2D base texture; consuming
// guest shaders sample texture2d_array (MSC ForceTextureArray), so the
// descriptor slot binds a 2DArray view with the binding's host swizzle —
// the same view mechanism the legacy FinalizeTexture uses.
static MTLTextureSwizzle HostSwizzleComponentToMetal(uint32_t component) {
  switch (component) {
    case xenos::XE_GPU_TEXTURE_SWIZZLE_R:
      return MTLTextureSwizzleRed;
    case xenos::XE_GPU_TEXTURE_SWIZZLE_G:
      return MTLTextureSwizzleGreen;
    case xenos::XE_GPU_TEXTURE_SWIZZLE_B:
      return MTLTextureSwizzleBlue;
    case xenos::XE_GPU_TEXTURE_SWIZZLE_A:
      return MTLTextureSwizzleAlpha;
    case xenos::XE_GPU_TEXTURE_SWIZZLE_0:
      return MTLTextureSwizzleZero;
    case xenos::XE_GPU_TEXTURE_SWIZZLE_1:
    default:
      return MTLTextureSwizzleOne;
  }
}

class MetalSharedTextureCache::MetalTexture final
    : public TextureCache::Texture {
 public:
  MetalTexture(MetalSharedTextureCache& cache, const TextureKey& key,
               void* mtl_texture)
      : Texture(cache, key), mtl_texture_(mtl_texture) {
    // Full chain, not just the base level: LRU eviction and the memory limits
    // are driven by this number, so a mipped texture must not under-report.
    uint64_t bytes = 0;
    uint32_t levels = key.mip_max_level + 1;
    for (uint32_t level = 0; level < levels; ++level) {
      bytes += uint64_t(std::max(key.GetWidth() >> level, 1u)) *
               std::max(key.GetHeight() >> level, 1u) * 4u;
    }
    SetHostMemoryUsage(bytes);
  }
  ~MetalTexture() override {
    auto& cache = static_cast<MetalSharedTextureCache&>(texture_cache());
    // Submission-gated release through the single legacy retirement pool:
    // the load encoder referenced this texture directly and views bound in
    // descriptor slots retain it; destruction is LRU/completion-gated, but
    // the retire pool is the uniformly safe release point.
    if (cache.legacy_) {
      cache.legacy_->RetireExternalTexture(mtl_texture_);
    } else {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)mtl_texture_;
      (void)released;
    }
  }
  void* mtl_texture() const { return mtl_texture_; }

  // Base and mip load provenance are tracked SEPARATELY so a later partial
  // reload of one region cannot hide an earlier faulted load of the other.
  uint64_t last_load_submission = 0;      // base region
  uint64_t last_mip_load_submission = 0;  // mip chain

 private:
  void* mtl_texture_;  // id<MTLTexture>, 2D base, retained.
};

MetalSharedTextureCache::MetalSharedTextureCache(
    const RegisterFile& register_file, MetalSharedMemory& shared_memory,
    void* device, MetalTextureCache* legacy_adapter)
    : TextureCache(register_file, shared_memory, 1, 1),
      device_(device),
      legacy_(legacy_adapter) {}

MetalSharedTextureCache::~MetalSharedTextureCache() {
  for (BindingSlot& binding_slot : binding_slots_) {
    ReleaseBindingSlot(binding_slot);
  }
  // Destroy all textures HERE, while the derived object is still alive:
  // ~MetalTexture casts texture_cache() back to MetalSharedTextureCache and
  // reads legacy_, which is undefined once the base destructor runs (the
  // base header documents exactly this pattern). The base destructor then
  // clears an empty map.
  DestroyAllTextures(true);
  if (direct_pipeline_) {
    id<MTLComputePipelineState> released =
        (__bridge_transfer id<MTLComputePipelineState>)direct_pipeline_;
    (void)released;
    direct_pipeline_ = nullptr;
  }
}

bool MetalSharedTextureCache::Initialize() {
  direct_pipeline_ = MetalLoadShaderPipelines::CreateDirectRgba8Pipeline(device_);
  return direct_pipeline_ != nullptr;
}

bool MetalSharedTextureCache::PlanLevels(const Texture& texture, bool load_base,
                                         bool load_mips,
                                         std::vector<MipLevelPlan>* out) {
  TextureKey key = texture.key();
  MipLevelPlanInputs in;
  in.base_address = uint32_t(key.base_page) << 12;
  in.mip_address = uint32_t(key.mip_page) << 12;
  in.width = key.GetWidth();
  in.height = key.GetHeight();
  in.mip_max_level = key.mip_max_level;
  in.guest_buffer_size = SharedMemory::kBufferSize;
  in.load_base = load_base;
  in.load_mips = load_mips;
  // The single planning authority (metal_mip_level_plan.h), shared with
  // conformance so production and tests can never diverge. The memo caches
  // only the per-shape range scan, so it changes cost, never the plan.
  return BuildMipLevelPlan(in, texture.guest_layout(), out, &range_memo_);
}

std::unique_ptr<TextureCache::Texture> MetalSharedTextureCache::CreateTexture(
    TextureKey key) {
  // Strict routed-class guard (C1T-R): anything else is unsupported here and
  // falls back to the legacy cache. Mirrors IsStrictRgba8Eligible at key
  // granularity, plus pitch != 0 (the key carries the raw fetch pitch; the
  // legacy align32(width) default is not representable) and the mip_page
  // agreement checked below.
  if (key.dimension != xenos::DataDimension::k2DOrStacked ||
      (key.format != xenos::TextureFormat::k_8_8_8_8 &&
       key.format !=
           xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16) ||  // byte alias
      !key.tiled || key.signed_separate || key.scaled_resolve ||
      key.depth_or_array_size_minus_1 != 0 || key.base_page == 0 ||
      key.pitch == 0) {
    return nullptr;
  }
  // The strict MIPPED sibling is served whenever the chain has its own guest
  // pointer; the single-mip class stays exactly base-only. Each class refuses
  // the other's mip_page shape, so they remain mutually exclusive.
  bool mipped = key.mip_max_level != 0;
  if (mipped) {
    if (key.mip_page == 0) {
      return nullptr;
    }
  } else if (key.mip_page != 0) {
    return nullptr;
  }
  uint32_t width = key.GetWidth();
  uint32_t height = key.GetHeight();
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:width
                                  height:height
                               mipmapped:mipped ? YES : NO];
  desc.storageMode = MTLStorageModePrivate;
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  if (mipped) {
    // Level count from the shared layout's CLAMPED max_level, and the key must
    // agree with it -- the planner cross-checks the same pair, so a descriptor
    // and a plan can never disagree about how many levels exist.
    texture_util::TextureGuestLayout layout = key.GetGuestLayout();
    if (layout.max_level != key.mip_max_level) {
      return nullptr;
    }
    desc.mipmapLevelCount = layout.max_level + 1;
    // The load writes through level-scoped views; the G0 foundation fixture
    // establishes that this usage bit is part of that contract.
    desc.usage |= MTLTextureUsagePixelFormatView;
  }
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return nullptr;
  }
  return std::unique_ptr<Texture>(
      new MetalTexture(*this, key, (__bridge_retained void*)texture));
}

bool MetalSharedTextureCache::LoadTextureDataFromResidentMemoryImpl(
    Texture& texture, bool load_base, bool load_mips) {
  MetalTexture& metal_texture = static_cast<MetalTexture&>(texture);
  TextureKey key = texture.key();
  bool mipped = key.mip_max_level != 0;
  if (!mipped) {
    // The promoted single-mip class: base-only by CreateTexture's guard.
    if (!load_base || load_mips) {
      return false;
    }
  }
  if (!load_base && !load_mips) {
    return false;
  }

  // ---- Content guard, the equivalent of the legacy path's: captures never
  // record mip memory, so a trace replay restores all-zero mip pages, and
  // loading those blacks out every minified surface. The legacy cache probes
  // the first 4 KiB at mip_address and serves base-only when it is entirely
  // zero; the routed path must apply the SAME rule or the two diverge on
  // replays and on any live all-zero mip region. Direct replay non-engagement
  // telemetry is evidence, not a substitute -- older or future traces could
  // engage. New-capture mip recording is unaffected (that is the warm sweep,
  // which only records regions already up to date).
  if (mipped && load_mips) {
    uint32_t mip_addr = uint32_t(key.mip_page) << 12;
    // Bound FIRST: the subtraction below underflows for an out-of-range
    // address, which would produce an enormous probe length.
    if (mip_addr >= SharedMemory::kBufferSize) {
      return false;
    }
    const uint32_t* probe = reinterpret_cast<const uint32_t*>(
        shared_memory().guest_memory().TranslatePhysical(mip_addr));
    if (!probe) {
      return false;
    }
    uint32_t probe_words = std::min<uint32_t>(
        1024, (SharedMemory::kBufferSize - mip_addr) / 4);
    bool any_nonzero = false;
    for (uint32_t i = 0; i < probe_words; ++i) {
      if (probe[i]) {
        any_nonzero = true;
        break;
      }
    }
    if (!any_nonzero) {
      // Zero mip region: decline the whole load so the legacy path applies its
      // own guard and serves the base level only. Atomic fallback.
      return false;
    }
  }

  // ---- Plan EVERY requested level and validate EVERY consumed range BEFORE
  // any encoding, so an unsatisfiable plan falls back to the legacy cache
  // atomically instead of half-loading a texture. ----
  std::vector<MipLevelPlan> plan;
  if (!PlanLevels(texture, load_base, load_mips, &plan)) {
    return false;
  }
  auto& metal_shared_memory = static_cast<MetalSharedMemory&>(shared_memory());
  for (const MipLevelPlan& p : plan) {
    // Register each exact read range as CONSUMED by the open frame so a later
    // same-frame rewrite triggers the existing hazard flush.
    //
    // ATOMICITY, precisely: if a later request fails, earlier ones stay
    // registered and we return before ANY encoding, so no texture content is
    // half-written and the legacy path serves the whole texture. A surplus
    // consumed range is conservative by construction -- it can only cause an
    // extra hazard flush, never a missed one -- so the guarantee is "no partial
    // GPU work", not "no partial bookkeeping".
    if (!metal_shared_memory.RequestConsumedRange(
            p.consumed_begin, p.consumed_end - p.consumed_begin)) {
      return false;
    }
  }

  // ---- Pre-create and validate EVERY level view BEFORE opening the encoder.
  // A late newTextureView failure must not leave earlier dispatches encoded, so
  // view creation cannot be interleaved with dispatching. ----
  id<MTLTexture> base_texture =
      (__bridge id<MTLTexture>)metal_texture.mtl_texture();
  if (!base_texture) {
    return false;
  }
  // RAII: every early exit after pre-creation releases the retained views
  // exactly once. `release_on_scope_exit` is cleared only after the views have
  // been handed to the submission-gated retirement pool, so a view is never
  // both retired and released.
  std::vector<void*> level_views;  // retained id<MTLTexture>, parallel to plan
  bool release_on_scope_exit = true;
  struct ViewGuard {
    std::vector<void*>* views;
    bool* armed;
    ~ViewGuard() {
      if (!*armed) {
        return;
      }
      for (void* v : *views) {
        id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)v;
        (void)released;
      }
      views->clear();
    }
  } view_guard{&level_views, &release_on_scope_exit};
  bool views_ok = true;
  if (mipped) {
    level_views.reserve(plan.size());
    for (const MipLevelPlan& p : plan) {
      id<MTLTexture> view = [base_texture
          newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
                            textureType:MTLTextureType2D
                                 levels:NSMakeRange(p.host_level, 1)
                                 slices:NSMakeRange(0, 1)];
      if (!view) {
        views_ok = false;
        break;
      }
      level_views.push_back((__bridge_retained void*)view);
    }
    if (!views_ok) {
      // The guard releases what we made; nothing was encoded, so the legacy
      // path serves the whole texture.
      return false;
    }
  }

  // ---- ONE PrepareTransferCommandBuffer per texture load; ALL levels batched
  // as dispatches on that existing frame command buffer. ----
  void* command_buffer_ptr = prepare_transfer_ ? prepare_transfer_() : nullptr;
  if (!command_buffer_ptr || !direct_pipeline_) {
    return false;
  }
  id<MTLCommandBuffer> command_buffer =
      (__bridge id<MTLCommandBuffer>)command_buffer_ptr;
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (!encoder) {
    return false;
  }
  encoder.label = mipped ? @"shared-texcache direct load (mip chain)"
                         : @"shared-texcache direct load";
  [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)
                                       direct_pipeline_];
  [encoder setBuffer:(__bridge id<MTLBuffer>)metal_shared_memory.GetBuffer()
              offset:0
             atIndex:MetalLoadShaderPipelines::kBufferIndexSource];
  MTLSize group = MTLSizeMake(MetalLoadShaderPipelines::kThreadgroupSizeX,
                              MetalLoadShaderPipelines::kThreadgroupSizeY, 1);
  for (size_t i = 0; i < plan.size(); ++i) {
    const MipLevelPlan& p = plan[i];
    // Ephemeral level-scoped write view, pre-created above: level N of the base
    // aliased as a single-level image so the unchanged image_store primitive
    // can target it. Deliberately NOT cached -- a per-level view cache would be
    // new lifetime state and a fresh instance of the stale-reference class.
    // A non-mipped texture needs no view at all.
    id<MTLTexture> dest =
        mipped ? (__bridge id<MTLTexture>)level_views[i] : base_texture;
    MetalLoadShaderPipelines::LoadConstants constants = {};
    constants.is_tiled_3d_endian_scale = 1u | (uint32_t(key.endianness) << 2);
    constants.guest_offset = p.guest_offset;
    constants.guest_pitch_aligned = p.pitch_blocks;
    constants.size_blocks[0] = p.width;
    constants.size_blocks[1] = p.height;
    constants.size_blocks[2] = 1;
    // The packed sub-tile origin rides host_offset (px low 16 / py high 16) --
    // the production-zero field the 6.1-G0 foundation proved. Zero for base and
    // non-packed levels, keeping those byte-identical to the promoted route.
    constants.host_offset = (p.px & 0xFFFFu) | ((p.py & 0xFFFFu) << 16);
    constants.host_pitch = ((p.width * 4u) + 255u) & ~255u;
    constants.height_texels = p.height;
    [encoder setBytes:&constants
               length:sizeof(constants)
              atIndex:MetalLoadShaderPipelines::kBufferIndexPushConstants];
    [encoder setTexture:dest atIndex:1];
    uint32_t threads_x = (p.width + 7u) / 8u;  // 1 thread = 8 texels along X.
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      (threads_x + group.width - 1) /
                                          group.width,
                                      (p.height + group.height - 1) /
                                          group.height,
                                      1)
            threadsPerThreadgroup:group];
  }
  [encoder endEncoding];

  // The GPU reads each view for the life of this submission; retire them all
  // through the single submission-gated legacy pool, then DISARM the guard so
  // ownership transfers exactly once.
  release_on_scope_exit = false;
  for (void* v : level_views) {
    if (legacy_) {
      legacy_->RetireExternalTexture(v);
    } else {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)v;
      (void)released;
    }
  }
  level_views.clear();

  uint64_t submission = submission_provider_ ? submission_provider_() : 0;
  if (load_base) {
    metal_texture.last_load_submission = submission;
  }
  if (load_mips) {
    metal_texture.last_mip_load_submission = submission;
  }
  // Keep the texture out of LRU destruction while its load is in flight
  // (completion actions can pump mid-frame on upload-ring pressure).
  texture.MarkAsUsed();

  // No trace recording here (M5 revised by gate-C evidence): the route
  // disengages while a capture is open, so a load never encodes during a
  // captured frame and the legacy path records the CPU-oracle read-set.
  return true;
}

void MetalSharedTextureCache::UpdateTextureBindingsImpl(
    uint32_t fetch_constant_mask) {
  uint32_t bits = fetch_constant_mask;
  uint32_t index;
  while (xe::bit_scan_forward(bits, &index)) {
    bits = xe::clear_lowest_bit(bits);
    BindingSlot& binding_slot = binding_slots_[index];
    const TextureBinding* binding = GetValidTextureBinding(index);
    const Texture* texture = binding ? binding->texture : nullptr;
    if (!texture) {
      // Park rather than release: BeginFrame resets ALL bindings each swap,
      // and releasing here would churn a fresh slot+view per warm texture
      // per frame. A parked slot is never served (the serve check requires a
      // valid matching binding) and is retired when a different binding
      // arrives or at destruction; its view pins at most 32 textures.
      continue;
    }
    if (binding_slot.texture == texture &&
        binding_slot.host_swizzle == binding->host_swizzle &&
        binding_slot.slot) {
      continue;  // Descriptor already expresses this binding.
    }
    ReleaseBindingSlot(binding_slot);
    const MetalTexture* metal_texture =
        static_cast<const MetalTexture*>(texture);
    id<MTLTexture> base_texture =
        (__bridge id<MTLTexture>)metal_texture->mtl_texture();
    // Always a view: the 2D->2DArray type change is required even for the
    // identity swizzle (guest shaders sample texture2d_array).
    uint32_t swizzle = binding->host_swizzle;
    MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
        HostSwizzleComponentToMetal((swizzle >> 0) & 0b111),
        HostSwizzleComponentToMetal((swizzle >> 3) & 0b111),
        HostSwizzleComponentToMetal((swizzle >> 6) & 0b111),
        HostSwizzleComponentToMetal((swizzle >> 9) & 0b111));
    // FULL mip range: a mipped texture served through a single-level view
    // samples only mip 0, which would silently break guest LOD selection.
    NSUInteger level_count = base_texture.mipmapLevelCount;
    id<MTLTexture> view = [base_texture
        newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
                          textureType:MTLTextureType2DArray
                               levels:NSMakeRange(0, level_count)
                              slices:NSMakeRange(0, 1)
                             swizzle:channels];
    if (!view || !legacy_) {
      continue;  // Unserved; the legacy path serves this index.
    }
    uint32_t slot = legacy_->AllocateExternalSlot();
    if (!slot) {
      continue;  // Heap exhausted; fall back rather than corrupt slot 0.
    }
    void* view_ptr = (__bridge_retained void*)view;
    legacy_->BindExternalSlot(slot, view_ptr);
    binding_slot.texture = texture;
    binding_slot.host_swizzle = swizzle;
    binding_slot.sampled_view = view_ptr;
    binding_slot.slot = slot;
  }
}

void MetalSharedTextureCache::ClearCache() {
  // Release the descriptor state FIRST, while the textures it names are still
  // alive. The base clear below resets the bindings, and this cache's
  // UpdateTextureBindingsImpl deliberately parks a texture-less binding
  // instead of releasing it, so nothing else would ever retire these.
  for (BindingSlot& binding_slot : binding_slots_) {
    ReleaseBindingSlot(binding_slot);
  }
  // Derived scan results keyed by texture shape; meaningless once the textures
  // are gone, and a clear must not leave state behind that a growth check
  // would see.
  range_memo_.Clear();
  TextureCache::ClearCache();
}

void MetalSharedTextureCache::ReleaseBindingSlot(BindingSlot& binding_slot) {
  if (!binding_slot.slot && !binding_slot.sampled_view) {
    binding_slot = BindingSlot{};
    return;
  }
  if (legacy_) {
    // In-flight draws may still index the slot / sample the view: retire
    // both through the single submission-gated legacy pools.
    if (binding_slot.slot) {
      legacy_->RetireExternalSlot(binding_slot.slot);
    }
    if (binding_slot.sampled_view) {
      legacy_->RetireExternalTexture(binding_slot.sampled_view);
    }
    if (binding_slot.signed_slot) {
      legacy_->RetireExternalSlot(binding_slot.signed_slot);
    }
    if (binding_slot.signed_view) {
      legacy_->RetireExternalTexture(binding_slot.signed_view);
    }
  } else {
    if (binding_slot.sampled_view) {
      id<MTLTexture> released =
          (__bridge_transfer id<MTLTexture>)binding_slot.sampled_view;
      (void)released;
    }
    if (binding_slot.signed_view) {
      id<MTLTexture> released =
          (__bridge_transfer id<MTLTexture>)binding_slot.signed_view;
      (void)released;
    }
  }
  binding_slot = BindingSlot{};
}

void MetalSharedTextureCache::PrepareEligible(uint32_t eligible_mask) {
  if (!eligible_mask) {
    return;
  }
  uint64_t submission = submission_provider_ ? submission_provider_() : 0;
  if (submission > last_begun_submission_) {
    last_begun_submission_ = submission;
    BeginSubmission(submission);
  }
  // Per-draw re-parse of exactly the masked constants (no register-write
  // notification exists in this backend; the base key comparison keeps
  // unchanged bindings load-free).
  uint32_t bits = eligible_mask;
  uint32_t index;
  while (xe::bit_scan_forward(bits, &index)) {
    bits = xe::clear_lowest_bit(bits);
    TextureFetchConstantWritten(index);
  }
  RequestTextures(eligible_mask);
}

uint32_t MetalSharedTextureCache::GetServedSlot(uint32_t fetch_constant_index,
                                                bool is_signed) {
  if (fetch_constant_index >= xenos::kTextureFetchConstantCount) {
    return 0;
  }
  BindingSlot& binding_slot = binding_slots_[fetch_constant_index];
  const TextureBinding* binding = GetValidTextureBinding(fetch_constant_index);
  if (!binding || !binding->texture || !binding_slot.slot ||
      binding_slot.texture != binding->texture) {
    return 0;
  }
  Texture* texture = binding->texture;
  if (texture->base_outdated_lockless() || texture->mips_outdated_lockless()) {
    // Invalidated between prep and serve — the CPU path serves current RAM.
    return 0;
  }
  MetalTexture* metal_texture = static_cast<MetalTexture*>(texture);
  if (submission_fault_query_) {
    // M4: content that rode a lost command buffer must never be served. BOTH
    // regions are consulted, because base and mip loads carry independent
    // submissions -- checking only the base would make the claimed independent
    // provenance false and could serve a texture whose mip chain is garbage.
    // The next invalidation (recurring resolve dests re-invalidate every frame)
    // reloads, and the CPU path covers until then.
    bool base_faulted =
        metal_texture->last_load_submission &&
        submission_fault_query_(metal_texture->last_load_submission);
    bool mip_faulted =
        metal_texture->last_mip_load_submission &&
        submission_fault_query_(metal_texture->last_mip_load_submission);
    if (base_faulted || mip_faulted) {
      return 0;
    }
  }
  if (is_signed) {
    // Lazy signed serve: an RGBA8Snorm view of the SAME loaded bytes, with
    // the SAME swizzle/type/levels as the unsigned view. Created on first
    // signed use; retired with the binding slot.
    if (!binding_slot.signed_slot) {
      const MetalTexture* mt = static_cast<const MetalTexture*>(texture);
      id<MTLTexture> base_texture = (__bridge id<MTLTexture>)mt->mtl_texture();
      uint32_t swizzle = binding_slot.host_swizzle;
      MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
          HostSwizzleComponentToMetal((swizzle >> 0) & 0b111),
          HostSwizzleComponentToMetal((swizzle >> 3) & 0b111),
          HostSwizzleComponentToMetal((swizzle >> 6) & 0b111),
          HostSwizzleComponentToMetal((swizzle >> 9) & 0b111));
      id<MTLTexture> view = [base_texture
          newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Snorm
                            textureType:MTLTextureType2DArray
                                 levels:NSMakeRange(
                                            0, base_texture.mipmapLevelCount)
                                 slices:NSMakeRange(0, 1)
                                swizzle:channels];
      if (!view || !legacy_) {
        return 0;  // Unserved; legacy path handles the signed binding.
      }
      uint32_t slot = legacy_->AllocateExternalSlot();
      if (!slot) {
        return 0;
      }
      void* view_ptr = (__bridge_retained void*)view;
      legacy_->BindExternalSlot(slot, view_ptr);
      binding_slot.signed_view = view_ptr;
      binding_slot.signed_slot = slot;
    }
    texture->MarkAsUsed();
    if (legacy_) {
      legacy_->AddActiveTexture(binding_slot.signed_view);
    }
    return binding_slot.signed_slot;
  }
  texture->MarkAsUsed();
  if (legacy_) {
    legacy_->AddActiveTexture(binding_slot.sampled_view);
  }
  return binding_slot.slot;
}

void MetalSharedTextureCache::OnSubmissionsFaulted(
    const std::vector<uint64_t>& submissions) {
  if (submissions.empty()) {
    return;
  }
  ForEachTexture([&](Texture& texture) {
    MetalTexture& metal_texture = static_cast<MetalTexture&>(texture);
    // BOTH regions are checked independently: base and mip loads carry their
    // own submission, so a later partial reload of one cannot hide an earlier
    // faulted load of the other.
    bool base_faulted = false;
    bool mip_faulted = false;
    for (uint64_t submission : submissions) {
      if (metal_texture.last_load_submission &&
          metal_texture.last_load_submission == submission) {
        base_faulted = true;
      }
      if (metal_texture.last_mip_load_submission &&
          metal_texture.last_mip_load_submission == submission) {
        mip_faulted = true;
      }
    }
    if (!base_faulted && !mip_faulted) {
      return;
    }
    // The load rode a lost command buffer: its content is garbage while
    // base state says up-to-date. Invalidate through the ONE authority —
    // the watch fires, the texture goes outdated, and the next use
    // revalidates + reloads from intact RAM (also evicts any overlapping
    // legacy entry via its own global watch).
    TextureKey key = texture.key();
    if (base_faulted) {
      shared_memory().MemoryInvalidationCallback(
          key.base_page << 12, texture.GetGuestBaseSize(), true);
      metal_texture.last_load_submission = 0;
    }
    if (mip_faulted) {
      shared_memory().MemoryInvalidationCallback(
          key.mip_page << 12, texture.GetGuestMipsSize(), true);
      metal_texture.last_mip_load_submission = 0;
    }
  });
}

void MetalSharedTextureCache::WriteAllCachedTexturesToTrace(
    TraceWriter& trace_writer) {
  // Pre-WriteTraceInitialState only (the caller guarantees the order): a
  // texture up-to-date HERE is exactly a texture the legacy cache would
  // still hold a warm entry for, and the record value matches the legacy
  // warm-entry sweep's. Outdated textures have no legacy analog (the
  // immediate drain-time InvalidateRange evicted the entry) and reload
  // during the captured frame through the legacy path instead.
  ForEachTexture([&](Texture& texture) {
    TextureKey key = texture.key();
    if (!texture.base_outdated_lockless()) {
      uint32_t base = key.base_page << 12;
      trace_writer.WriteMemoryRead(
          base, ComputeWatchedBaseBytes(uint32_t(key.pitch) << 5,
                                        key.GetHeight(), 4, base,
                                        SharedMemory::kBufferSize));
    }
    // 6.1: the mip chain is a SEPARATE up-to-date region at its own guest
    // pointer, so a warm mipped texture must record those bytes too or a
    // replay would restore a texture whose levels were never captured.
    if (key.mip_max_level && key.mip_page &&
        !texture.mips_outdated_lockless()) {
      trace_writer.WriteMemoryRead(uint32_t(key.mip_page) << 12,
                                   texture.GetGuestMipsSize());
    }
  });
}

void MetalSharedTextureCache::AppendAllTextures(
    std::vector<void*>& out) const {
  for (const BindingSlot& binding_slot : binding_slots_) {
    if (binding_slot.sampled_view) {
      out.push_back(binding_slot.sampled_view);
    }
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
