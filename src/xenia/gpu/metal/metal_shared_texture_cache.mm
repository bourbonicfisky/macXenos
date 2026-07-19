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
#include "xenia/gpu/metal/metal_texture_load_eligibility.h"
#include "xenia/gpu/trace_writer.h"

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
    SetHostMemoryUsage(uint64_t(key.GetWidth()) * key.GetHeight() * 4);
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

  uint64_t last_load_submission = 0;

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

std::unique_ptr<TextureCache::Texture> MetalSharedTextureCache::CreateTexture(
    TextureKey key) {
  // Strict routed-class guard (C1T-R): anything else is unsupported here and
  // falls back to the legacy cache. Mirrors IsStrictRgba8Eligible at key
  // granularity, plus pitch != 0 (the key carries the raw fetch pitch; the
  // legacy align32(width) default is not representable) and mip_page == 0
  // (single-mip base-only).
  if (key.dimension != xenos::DataDimension::k2DOrStacked ||
      key.format != xenos::TextureFormat::k_8_8_8_8 || !key.tiled ||
      key.signed_separate || key.scaled_resolve || key.mip_max_level != 0 ||
      key.mip_page != 0 || key.depth_or_array_size_minus_1 != 0 ||
      key.base_page == 0 || key.pitch == 0) {
    return nullptr;
  }
  uint32_t width = key.GetWidth();
  uint32_t height = key.GetHeight();
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.storageMode = MTLStorageModePrivate;
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    return nullptr;
  }
  return std::unique_ptr<Texture>(
      new MetalTexture(*this, key, (__bridge_retained void*)texture));
}

bool MetalSharedTextureCache::LoadTextureDataFromResidentMemoryImpl(
    Texture& texture, bool load_base, bool load_mips) {
  // The routed class is single-mip base-only by CreateTexture's guard.
  if (!load_base || load_mips) {
    return false;
  }
  MetalTexture& metal_texture = static_cast<MetalTexture&>(texture);
  TextureKey key = texture.key();
  uint32_t base = key.base_page << 12;
  uint32_t blocks_w = key.GetWidth();
  uint32_t blocks_h = key.GetHeight();
  // fetch pitch is texels/32, so pitch_texels is already 32-aligned; block ==
  // texel for 32bpp.
  uint32_t pitch_blocks_aligned = uint32_t(key.pitch) << 5;

  // Exact consumed source range (the proven contract's Tiled2D min/max over
  // the padded read grid), cached per shape.
  ShapeKey shape{blocks_w, blocks_h, pitch_blocks_aligned};
  auto shape_it = shape_ranges_.find(shape);
  if (shape_it == shape_ranges_.end()) {
    TextureLoadSourceRange r = ComputeStrictRgba8SourceRange(
        0, blocks_w, blocks_h, pitch_blocks_aligned, UINT32_MAX);
    shape_it = shape_ranges_
                   .emplace(shape, ShapeRange{r.begin, r.end, r.valid})
                   .first;
  }
  const ShapeRange& range = shape_it->second;
  if (!range.valid ||
      uint64_t(base) + range.end_off > SharedMemory::kBufferSize) {
    return false;
  }

  // The base-class LoadTextureData already ran RequestRange over the shared
  // guest layout extent (page revalidation from post-scatter RAM); this
  // registers the exact read range as CONSUMED by the open frame so a later
  // same-frame rewrite triggers the existing hazard flush — pre-encoder, the
  // same seam class the vfetch path uses.
  auto& metal_shared_memory = static_cast<MetalSharedMemory&>(shared_memory());
  if (!metal_shared_memory.RequestConsumedRange(
          base + range.begin_off, range.end_off - range.begin_off)) {
    return false;
  }

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
  encoder.label = @"shared-texcache direct load";
  MetalLoadShaderPipelines::LoadConstants constants = {};
  constants.is_tiled_3d_endian_scale = 1u | (uint32_t(key.endianness) << 2);
  constants.guest_offset = base;
  constants.guest_pitch_aligned = pitch_blocks_aligned;
  constants.size_blocks[0] = blocks_w;
  constants.size_blocks[1] = blocks_h;
  constants.size_blocks[2] = 1;
  // Host fields are unused by the direct (image-store) variant; keep the
  // buffer-shape values for ABI-shape consistency with the proven G0 case.
  constants.host_offset = 0;
  constants.host_pitch = ((blocks_w * 4u) + 255u) & ~255u;
  constants.height_texels = blocks_h;
  [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)
                                       direct_pipeline_];
  [encoder setBytes:&constants
             length:sizeof(constants)
            atIndex:MetalLoadShaderPipelines::kBufferIndexPushConstants];
  [encoder setTexture:(__bridge id<MTLTexture>)metal_texture.mtl_texture()
              atIndex:1];
  [encoder setBuffer:(__bridge id<MTLBuffer>)metal_shared_memory.GetBuffer()
              offset:0
             atIndex:MetalLoadShaderPipelines::kBufferIndexSource];
  MTLSize group = MTLSizeMake(MetalLoadShaderPipelines::kThreadgroupSizeX,
                              MetalLoadShaderPipelines::kThreadgroupSizeY, 1);
  uint32_t threads_x = (blocks_w + 7u) / 8u;  // 1 thread = 8 texels along X.
  [encoder dispatchThreadgroups:MTLSizeMake(
                                    (threads_x + group.width - 1) / group.width,
                                    (blocks_h + group.height - 1) /
                                        group.height,
                                    1)
          threadsPerThreadgroup:group];
  [encoder endEncoding];

  metal_texture.last_load_submission =
      submission_provider_ ? submission_provider_() : 0;
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
    id<MTLTexture> view = [base_texture
        newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
                          textureType:MTLTextureType2DArray
                               levels:NSMakeRange(0, 1)
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
  } else if (binding_slot.sampled_view) {
    id<MTLTexture> released =
        (__bridge_transfer id<MTLTexture>)binding_slot.sampled_view;
    (void)released;
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

uint32_t MetalSharedTextureCache::GetServedSlot(uint32_t fetch_constant_index) {
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
  if (submission_fault_query_ && metal_texture->last_load_submission &&
      submission_fault_query_(metal_texture->last_load_submission)) {
    // M4: content rode a lost command buffer — never serve it; the next
    // invalidation (recurring resolve dests re-invalidate every frame)
    // reloads, and the CPU path covers until then.
    return 0;
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
    if (!metal_texture.last_load_submission) {
      return;
    }
    for (uint64_t submission : submissions) {
      if (metal_texture.last_load_submission != submission) {
        continue;
      }
      // The load rode a lost command buffer: its content is garbage while
      // base state says up-to-date. Invalidate through the ONE authority —
      // the watch fires, the texture goes outdated, and the next use
      // revalidates + reloads from intact RAM (also evicts any overlapping
      // legacy entry via its own global watch).
      TextureKey key = texture.key();
      shared_memory().MemoryInvalidationCallback(
          key.base_page << 12, texture.GetGuestBaseSize(), true);
      metal_texture.last_load_submission = 0;
      return;
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
    if (texture.base_outdated_lockless()) {
      return;
    }
    TextureKey key = texture.key();
    uint32_t base = key.base_page << 12;
    trace_writer.WriteMemoryRead(
        base, ComputeWatchedBaseBytes(uint32_t(key.pitch) << 5,
                                      key.GetHeight(), 4, base,
                                      SharedMemory::kBufferSize));
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
