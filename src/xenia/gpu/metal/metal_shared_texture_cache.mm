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

// The C++-visible format table stores MTLPixelFormat raw values; pin them.
static_assert(MTLPixelFormatR8Unorm == xe::gpu::metal::kMtlPixelFormatR8Unorm, "");
static_assert(MTLPixelFormatABGR4Unorm == xe::gpu::metal::kMtlPixelFormatABGR4Unorm, "");
static_assert(MTLPixelFormatR8Snorm == xe::gpu::metal::kMtlPixelFormatR8Snorm, "");
static_assert(MTLPixelFormatR16Unorm == xe::gpu::metal::kMtlPixelFormatR16Unorm, "");
static_assert(MTLPixelFormatR16Snorm == xe::gpu::metal::kMtlPixelFormatR16Snorm, "");
static_assert(MTLPixelFormatR16Float == xe::gpu::metal::kMtlPixelFormatR16Float, "");
static_assert(MTLPixelFormatRG8Unorm == xe::gpu::metal::kMtlPixelFormatRG8Unorm, "");
static_assert(MTLPixelFormatRG8Snorm == xe::gpu::metal::kMtlPixelFormatRG8Snorm, "");
static_assert(MTLPixelFormatR32Float == xe::gpu::metal::kMtlPixelFormatR32Float, "");
static_assert(MTLPixelFormatRG16Unorm == xe::gpu::metal::kMtlPixelFormatRG16Unorm, "");
static_assert(MTLPixelFormatRG16Snorm == xe::gpu::metal::kMtlPixelFormatRG16Snorm, "");
static_assert(MTLPixelFormatRG16Float == xe::gpu::metal::kMtlPixelFormatRG16Float, "");
static_assert(MTLPixelFormatRGBA8Unorm == xe::gpu::metal::kMtlPixelFormatRGBA8Unorm, "");
static_assert(MTLPixelFormatRGBA8Snorm == xe::gpu::metal::kMtlPixelFormatRGBA8Snorm, "");
static_assert(MTLPixelFormatRGB10A2Unorm == xe::gpu::metal::kMtlPixelFormatRGB10A2Unorm, "");
static_assert(MTLPixelFormatRGBA16Unorm == xe::gpu::metal::kMtlPixelFormatRGBA16Unorm, "");
static_assert(MTLPixelFormatRGBA16Snorm == xe::gpu::metal::kMtlPixelFormatRGBA16Snorm, "");
static_assert(MTLPixelFormatRGBA16Float == xe::gpu::metal::kMtlPixelFormatRGBA16Float, "");

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include "xenia/base/logging.h"
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
    const SharedTextureFormatEntry* entry = FindSharedTextureFormat(key.format);
    const uint32_t bpb = entry ? entry->bytes_per_block : 4u;
    for (uint32_t level = 0; level < levels; ++level) {
      bytes += uint64_t(std::max(key.GetWidth() >> level, 1u)) *
               std::max(key.GetHeight() >> level, 1u) * bpb;
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
    void* device, MetalTextureCache* legacy_adapter,
    uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y)
    : TextureCache(register_file, shared_memory, draw_resolution_scale_x,
                   draw_resolution_scale_y),
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
  family_pipelines_.Shutdown();
  for (ScratchBuffer& sb : scratch_) {
    if (sb.buffer) {
      id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)sb.buffer;
      (void)released;
    }
  }
  scratch_.clear();
  if (direct_pipeline_) {
    id<MTLComputePipelineState> released =
        (__bridge_transfer id<MTLComputePipelineState>)direct_pipeline_;
    (void)released;
    direct_pipeline_ = nullptr;
  }
}

bool MetalSharedTextureCache::Initialize() {
  SharedTexturePacked16Supported().store(
      [(__bridge id<MTLDevice>)device_ supportsFamily:MTLGPUFamilyApple1],
      std::memory_order_relaxed);
  direct_pipeline_ = MetalLoadShaderPipelines::CreateDirectRgba8Pipeline(device_);
  if (!direct_pipeline_) {
    return false;
  }
  if (SharedTextureRouteFormatsWidened()) {
    if (!family_pipelines_.Initialize(device_)) {
      XELOGW(
          "MetalSharedTextureCache: texture-load family pipelines failed to "
          "build; only the RGBA8 direct class is routed");
    }
  }
  return true;
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
  const SharedTextureFormatEntry* entry = FindSharedTextureFormat(key.format);
  if (!entry || (!entry->direct && !family_pipelines_.initialized())) {
    return nullptr;
  }
  if (key.dimension != xenos::DataDimension::k2DOrStacked || !key.tiled ||
      key.signed_separate || key.scaled_resolve ||
      key.depth_or_array_size_minus_1 != 0 || key.pitch == 0) {
    return nullptr;
  }
  // The strict MIPPED sibling is served whenever the chain has its own guest
  // pointer; the single-mip class stays exactly base-only. Each class refuses
  // the other's mip_page shape, so they remain mutually exclusive.
  //
  // A mips-only key (base_page 0: the fetch's minimum level is above 0, so
  // the base cache zeroes the base page and the guest layout has no base
  // extent) is the D3D12/Vulkan contract for such fetches: levels 1..max load
  // from the mip chain, level 0 of the host texture is never written and
  // never sampled (the sampler's lodMinClamp is the fetch's minimum level,
  // and base-map pins to it). Red Dead Redemption fetches mips 1 and 2 of
  // its resolve-backed 4:4:4:4 pyramids alone (2026-10-04;
  // XE_METAL_TEX_ROUTE_MINMIP=0 keeps such fetches off the route).
  bool mipped = key.mip_max_level != 0;
  if (mipped) {
    if (key.mip_page == 0) {
      return nullptr;
    }
    if (key.base_page == 0 && !SharedRouteMinMipEnabled()) {
      return nullptr;
    }
  } else if (key.mip_page != 0 || key.base_page == 0) {
    return nullptr;
  }
  uint32_t width = key.GetWidth();
  uint32_t height = key.GetHeight();
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormat(entry->host_format)
                                   width:width
                                  height:height
                               mipmapped:mipped ? YES : NO];
  desc.storageMode = MTLStorageModePrivate;
  // PixelFormatView: the signed serve creates an RGBA8Snorm view of this
  // Unorm texture — a FORMAT-CASTING view, which Metal only defines when the
  // base texture declares this usage. Without it the cast view's sampling is
  // formally undefined (descriptor A/B for the 10395 corner artifact).
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsagePixelFormatView;
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

namespace {
// Bounded per-reason log of shared-route load declines (the prepass then
// reports the fetch as UNSERVED and drains it for the legacy path).
template <typename Key>  // TextureCache::TextureKey (protected; deduced)
void NoteDecline(const char* reason, const Key& key) {
  static std::unordered_map<std::string, uint32_t> s_counts;
  static std::mutex s_mutex;
  std::lock_guard<std::mutex> lock(s_mutex);
  uint32_t n = ++s_counts[reason];
  if (n <= 4 || (n & 1023) == 0) {
    XELOGW("MetalSharedTex DECLINE {} #{} fmt={} base={:08X} {}x{} mips={} "
           "mip_base={:08X}",
           reason, n, uint32_t(key.format), uint32_t(key.base_page) << 12,
           key.GetWidth(), key.GetHeight(), key.mip_max_level,
           uint32_t(key.mip_page) << 12);
  }
}
}  // namespace

bool MetalSharedTextureCache::LoadTextureDataFromResidentMemoryImpl(
    Texture& texture, bool load_base, bool load_mips) {
  MetalTexture& metal_texture = static_cast<MetalTexture&>(texture);
  TextureKey key = texture.key();
  bool mipped = key.mip_max_level != 0;
  if (!mipped) {
    // The promoted single-mip class: base-only by CreateTexture's guard.
    if (!load_base || load_mips) {
      NoteDecline("single-mip-load-shape", key);
      return false;
    }
  }
  if (!load_base && !load_mips) {
    NoteDecline("no-levels", key);
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
  static const bool s_zero_guard_live =
      std::getenv("XE_METAL_MIP_ZERO_GUARD_LIVE") != nullptr;
  if (mipped && load_mips && (trace_playback_ || s_zero_guard_live)) {
    uint32_t mip_addr = uint32_t(key.mip_page) << 12;
    // Bound FIRST: the subtraction below underflows for an out-of-range
    // address, which would produce an enormous probe length.
    if (mip_addr >= SharedMemory::kBufferSize) {
      NoteDecline("mip-address-out-of-range", key);
      return false;
    }
    const uint32_t* probe = reinterpret_cast<const uint32_t*>(
        shared_memory().guest_memory().TranslatePhysical(mip_addr));
    if (!probe) {
      NoteDecline("mip-probe-untranslatable", key);
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
      NoteDecline("zero-mip-region", key);
      return false;
    }
  }

  // ---- Plan EVERY requested level and validate EVERY consumed range BEFORE
  // any encoding, so an unsatisfiable plan falls back to the legacy cache
  // atomically instead of half-loading a texture. ----
  std::vector<MipLevelPlan> plan;
  if (!PlanLevels(texture, load_base, load_mips, &plan)) {
    NoteDecline("plan-failed", key);
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
      NoteDecline("consumed-range-refused", key);
      return false;
    }
  }
  const SharedTextureFormatEntry* entry = FindSharedTextureFormat(key.format);
  if (!entry) {
    NoteDecline("format-not-in-table", key);
    return false;
  }
  if (!entry->direct) {
    // The planner's exact range scan is the 32bpp tiled walk; for the other
    // block sizes register the conservative whole tiled extent of each level
    // (32-row aligned) as consumed. A superset can only add a hazard flush.
    for (const MipLevelPlan& p : plan) {
      uint32_t pitch_blocks = (p.pitch_blocks * 4u) / entry->bytes_per_block;
      uint64_t rows = uint64_t(((p.py + p.height) + 31u) & ~31u);
      uint64_t bytes = rows * pitch_blocks * entry->bytes_per_block;
      uint64_t end = std::min<uint64_t>(uint64_t(p.guest_offset) + bytes,
                                        SharedMemory::kBufferSize);
      if (end <= p.guest_offset ||
          !metal_shared_memory.RequestConsumedRange(
              p.guest_offset, uint32_t(end - p.guest_offset))) {
        NoteDecline("family-consumed-range-refused", key);
        return false;
      }
    }
  }

  if (entry->direct) {
    if (!LoadLevelsDirect(texture, plan, mipped)) {
      NoteDecline("direct-load-failed", key);
      return false;
    }
  } else if (!LoadLevelsFamily(texture, *entry, plan)) {
    NoteDecline("family-load-failed", key);
    return false;
  }

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

bool MetalSharedTextureCache::LoadLevelsDirect(
    Texture& texture, const std::vector<MipLevelPlan>& plan, bool mipped) {
  MetalTexture& metal_texture = static_cast<MetalTexture&>(texture);
  TextureKey key = texture.key();
  auto& metal_shared_memory = static_cast<MetalSharedMemory&>(shared_memory());
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
  // XE_METAL_ROUTE_LOAD_LOG=1: every direct load in encode order (bounded),
  // to interleave with the scatter / draw logs when bisecting staleness.
  static const bool s_load_log = std::getenv("XE_METAL_ROUTE_LOAD_LOG") != nullptr;
  if (s_load_log) {
    static uint32_t s_load_logs = 0;
    if (s_load_logs++ < 400) {
      XELOGI("MetalRouteLoad #{} base={:08X} {}x{} levels={} submission={}",
             s_load_logs, uint32_t(key.base_page) << 12, key.GetWidth(),
             key.GetHeight(), plan.size(),
             submission_provider_ ? submission_provider_() : 0);
    }
  }
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
  return true;
}

bool MetalSharedTextureCache::AllocateScratch(uint32_t size,
                                              uint32_t alignment,
                                              uint64_t submission,
                                              void** buffer_out,
                                              uint32_t* offset_out) {
  auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
  // 1. Bump inside the buffer already stamped with this submission.
  for (ScratchBuffer& sb : scratch_) {
    if (sb.submission != submission || !sb.buffer) {
      continue;
    }
    uint32_t head = align_up(sb.head, alignment);
    if (uint64_t(head) + size <= sb.size) {
      sb.head = head + size;
      *buffer_out = sb.buffer;
      *offset_out = head;
      return true;
    }
  }
  // 2. Reuse a buffer whose submission the GPU has finished with. A zero
  //    submission id (S7 tags off) can never be proven complete, so such
  //    buffers are never reused.
  const uint64_t completed =
      completed_submission_provider_ ? completed_submission_provider_() : 0;
  for (ScratchBuffer& sb : scratch_) {
    if (!sb.buffer || sb.submission == submission || sb.submission == 0 ||
        sb.submission > completed || size > sb.size) {
      continue;
    }
    sb.submission = submission;
    sb.head = size;
    *buffer_out = sb.buffer;
    *offset_out = 0;
    return true;
  }
  // 3. Grow the ring, bounded; beyond the bound the caller falls back.
  if (scratch_.size() >= kScratchBufferMax) {
    return false;
  }
  uint32_t buffer_size = std::max(kScratchBufferSize, align_up(size, 1u << 20));
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  id<MTLBuffer> buffer =
      [device newBufferWithLength:buffer_size
                          options:MTLResourceStorageModePrivate];
  if (!buffer) {
    return false;
  }
  buffer.label = @"shared-texcache load scratch";
  ScratchBuffer sb;
  sb.buffer = (__bridge_retained void*)buffer;
  sb.size = buffer_size;
  sb.head = size;
  sb.submission = submission;
  scratch_.push_back(sb);
  *buffer_out = sb.buffer;
  *offset_out = 0;
  return true;
}

// Buffer-writing families (everything but the direct RGBA8 class): each level
// is untiled by the upstream load shader into a linear scratch region laid
// out as the host texture rows (256-aligned pitch), then blitted into the
// private texture level in the same command buffer. Metal's default hazard
// tracking orders the compute writes before the blit reads.
bool MetalSharedTextureCache::LoadLevelsFamily(
    Texture& texture, const SharedTextureFormatEntry& entry,
    const std::vector<MipLevelPlan>& plan) {
  MetalTexture& metal_texture = static_cast<MetalTexture&>(texture);
  TextureKey key = texture.key();
  auto& metal_shared_memory = static_cast<MetalSharedMemory&>(shared_memory());
  void* pipeline_ptr = nullptr;
  TextureCache::LoadShaderIndex load_shader_index = kLoadShaderIndexUnknown;
  {
    const char* wanted = nullptr;
    switch (entry.load_shader) {
      case SharedLoadFamily::k8bpb:
        wanted = "texture_load_8bpb_cs";
        load_shader_index = kLoadShaderIndex8bpb;
        break;
      case SharedLoadFamily::k16bpb:
        wanted = "texture_load_16bpb_cs";
        load_shader_index = kLoadShaderIndex16bpb;
        break;
      case SharedLoadFamily::k32bpb:
        wanted = "texture_load_32bpb_cs";
        load_shader_index = kLoadShaderIndex32bpb;
        break;
      case SharedLoadFamily::k64bpb:
        wanted = "texture_load_64bpb_cs";
        load_shader_index = kLoadShaderIndex64bpb;
        break;
    }
    if (!wanted) {
      return false;
    }
    int index = family_pipelines_.FindIndex(wanted);
    if (index < 0) {
      return false;
    }
    pipeline_ptr = family_pipelines_.pipeline(size_t(index));
  }
  if (!pipeline_ptr) {
    return false;
  }
  const TextureCache::LoadShaderInfo& info =
      GetLoadShaderInfo(load_shader_index);
  const uint32_t bpb = entry.bytes_per_block;
  if (info.bytes_per_host_block != bpb) {
    return false;  // Tranche-1 families are bit copies: guest block == host.
  }
  const uint32_t blocks_per_thread = 1u << info.guest_x_blocks_per_thread_log2;

  // Plan every level's scratch region before encoding anything (atomic
  // fallback, like the direct path).
  struct LevelDispatch {
    MetalLoadShaderPipelines::LoadConstants constants;
    void* scratch = nullptr;
    uint32_t scratch_offset = 0;
    uint32_t host_pitch = 0;
    uint32_t region_w = 0;
    uint32_t region_h = 0;
    uint32_t threads_x = 0;
  };
  std::vector<LevelDispatch> dispatches;
  dispatches.reserve(plan.size());
  const uint64_t submission = submission_provider_ ? submission_provider_() : 0;
  for (const MipLevelPlan& p : plan) {
    LevelDispatch d;
    // The shader reads blocks [0, region) relative to the level's guest
    // offset; a packed sub-tile origin (px, py) is covered by dispatching the
    // enclosing region and blitting the sub-rectangle out of it.
    d.region_w = p.px + p.width;
    d.region_h = p.py + p.height;
    uint32_t written_row =
        ((d.region_w + blocks_per_thread - 1u) / blocks_per_thread) *
        blocks_per_thread * bpb;
    d.host_pitch = (written_row + 255u) & ~255u;
    uint64_t dest_bytes = uint64_t(d.host_pitch) * d.region_h + 256u;
    if (dest_bytes > kScratchBufferSize) {
      return false;
    }
    if (!AllocateScratch(uint32_t(dest_bytes), 256, submission, &d.scratch,
                         &d.scratch_offset)) {
      return false;
    }
    // The planner derives pitch_blocks at 32bpp; rescale to this format's
    // block size (pitch_bytes == pitch_blocks * 4 by construction).
    uint32_t pitch_blocks = (p.pitch_blocks * 4u) / bpb;
    d.constants = {};
    d.constants.is_tiled_3d_endian_scale =
        1u | (uint32_t(key.endianness) << 2);
    d.constants.guest_offset = p.guest_offset;
    d.constants.guest_pitch_aligned = pitch_blocks;
    d.constants.size_blocks[0] = d.region_w;
    d.constants.size_blocks[1] = d.region_h;
    d.constants.size_blocks[2] = 1;
    d.constants.host_offset = 0;
    d.constants.host_pitch = d.host_pitch;
    d.constants.height_texels = d.region_h;
    d.threads_x = (d.region_w + blocks_per_thread - 1u) / blocks_per_thread;
    dispatches.push_back(d);
  }

  void* command_buffer_ptr = prepare_transfer_ ? prepare_transfer_() : nullptr;
  if (!command_buffer_ptr) {
    return false;
  }
  id<MTLCommandBuffer> command_buffer =
      (__bridge id<MTLCommandBuffer>)command_buffer_ptr;
  id<MTLTexture> base_texture =
      (__bridge id<MTLTexture>)metal_texture.mtl_texture();
  if (!base_texture) {
    return false;
  }
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (!encoder) {
    return false;
  }
  encoder.label = @"shared-texcache family load";
  [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)
                                       pipeline_ptr];
  [encoder setBuffer:(__bridge id<MTLBuffer>)metal_shared_memory.GetBuffer()
              offset:0
             atIndex:MetalLoadShaderPipelines::kBufferIndexSource];
  MTLSize group = MTLSizeMake(MetalLoadShaderPipelines::kThreadgroupSizeX,
                              MetalLoadShaderPipelines::kThreadgroupSizeY,
                              MetalLoadShaderPipelines::kThreadgroupSizeZ);
  for (const LevelDispatch& d : dispatches) {
    [encoder setBytes:&d.constants
               length:sizeof(d.constants)
              atIndex:MetalLoadShaderPipelines::kBufferIndexPushConstants];
    [encoder setBuffer:(__bridge id<MTLBuffer>)d.scratch
                offset:d.scratch_offset
               atIndex:MetalLoadShaderPipelines::kBufferIndexDest];
    [encoder dispatchThreadgroups:MTLSizeMake(
                                      (d.threads_x + group.width - 1) /
                                          group.width,
                                      (d.region_h + group.height - 1) /
                                          group.height,
                                      1)
            threadsPerThreadgroup:group];
  }
  [encoder endEncoding];

  id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
  if (!blit) {
    return false;
  }
  blit.label = @"shared-texcache family load blit";
  for (size_t i = 0; i < plan.size(); ++i) {
    const MipLevelPlan& p = plan[i];
    const LevelDispatch& d = dispatches[i];
    NSUInteger source_offset =
        d.scratch_offset + NSUInteger(p.py) * d.host_pitch + NSUInteger(p.px) * bpb;
    [blit copyFromBuffer:(__bridge id<MTLBuffer>)d.scratch
               sourceOffset:source_offset
          sourceBytesPerRow:d.host_pitch
        sourceBytesPerImage:NSUInteger(d.host_pitch) * p.height
                 sourceSize:MTLSizeMake(p.width, p.height, 1)
                  toTexture:base_texture
           destinationSlice:0
           destinationLevel:p.host_level
          destinationOrigin:MTLOriginMake(0, 0, 0)];
  }
  // XE_METAL_DUMP_ROUTE=<hex base>: read every loaded level of this routed
  // texture back after the load (host rows, bpb bytes per texel) to
  // /tmp/xe_route_<base>_L<level>_<n>.bin -- the texel-level oracle against
  // the legacy decode's XE_METAL_DUMP_FETCH dumps of the same base.
  static const uint32_t s_dump_route = [] {
    const char* v = std::getenv("XE_METAL_DUMP_ROUTE");
    return v ? uint32_t(strtoul(v, nullptr, 16)) : 0u;
  }();
  if (s_dump_route && (uint32_t(key.base_page) << 12) == s_dump_route) {
    static uint32_t s_n = 0;
    const uint32_t n = s_n++;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
    for (const MipLevelPlan& p0 : plan) {
      NSUInteger row = NSUInteger(p0.width) * bpb;
      id<MTLBuffer> rb = [device newBufferWithLength:row * p0.height
                                             options:MTLResourceStorageModeShared];
      [blit copyFromTexture:base_texture
                  sourceSlice:0
                  sourceLevel:p0.host_level
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(p0.width, p0.height, 1)
                     toBuffer:rb
            destinationOffset:0
       destinationBytesPerRow:row
     destinationBytesPerImage:row * p0.height];
      std::string path = fmt::format("/tmp/xe_route_{:08X}_L{}_{}.bin",
                                     s_dump_route, p0.host_level, n);
      uint32_t w = p0.width, h = p0.height, lvl = p0.host_level,
               go = p0.guest_offset, px = p0.px, py = p0.py,
               pb = p0.pitch_blocks;
      [command_buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        if (FILE* f = fopen(path.c_str(), "wb")) {
          fwrite([rb contents], 1, row * h, f);
          fclose(f);
        }
        XELOGW("MetalRouteDump {} L{} {}x{} bpb={} guest_offset={:08X} px={} py={} pitch_blocks32={}",
               path, lvl, w, h, bpb, go, px, py, pb);
      }];
    }
  }
  [blit endEncoding];
  static uint32_t s_family_logs = 0;
  static const bool s_family_log_all =
      std::getenv("XE_METAL_FAMILY_LOAD_LOG") != nullptr;
  if (s_family_logs++ < 16 || s_family_log_all) {
    XELOGI("MetalSharedTex FAMILY-LOAD fmt={} base={:08X} {}x{} levels={} bpb={}",
           uint32_t(key.format), key.base_page << 12, key.GetWidth(),
           key.GetHeight(), plan.size(), bpb);
  }
  family_load_count_ += 1;
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
    const SharedTextureFormatEntry* entry =
        FindSharedTextureFormat(texture->key().format);
    MTLPixelFormat view_format =
        entry ? MTLPixelFormat(entry->host_format) : MTLPixelFormatRGBA8Unorm;
    id<MTLTexture> view = [base_texture
        newTextureViewWithPixelFormat:view_format
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
      const SharedTextureFormatEntry* entry =
          FindSharedTextureFormat(texture->key().format);
      MTLPixelFormat signed_format =
          entry && entry->host_format_signed
              ? MTLPixelFormat(entry->host_format_signed)
              : MTLPixelFormatRGBA8Snorm;
      id<MTLTexture> view = [base_texture
          newTextureViewWithPixelFormat:signed_format
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
      static uint32_t s_signed_serve_logs = 0;
      if (s_signed_serve_logs++ < 16) {
        TextureKey key = texture->key();
        XELOGI(
            "MetalSharedTex SIGNED-SERVE fc={} base={:08X} {}x{} pitch={} "
            "mips={} fmt={}",
            fetch_constant_index, key.base_page << 12, key.GetWidth(),
            key.GetHeight(), uint32_t(key.pitch) << 5, key.mip_max_level,
            uint32_t(key.format));
      }
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

const char* MetalSharedTextureCache::UnservedReason(
    uint32_t fetch_constant_index, bool is_signed) {
  if (fetch_constant_index >= xenos::kTextureFetchConstantCount) {
    return "index";
  }
  BindingSlot& binding_slot = binding_slots_[fetch_constant_index];
  const TextureBinding* binding = GetValidTextureBinding(fetch_constant_index);
  if (!binding) {
    return "no-valid-binding";
  }
  if (!binding->texture) {
    return "no-texture";
  }
  if (!binding_slot.slot) {
    return "no-slot";
  }
  if (binding_slot.texture != binding->texture) {
    return "slot-texture-mismatch";
  }
  if (binding->texture->base_outdated_lockless()) {
    return "base-outdated";
  }
  if (binding->texture->mips_outdated_lockless()) {
    return "mips-outdated";
  }
  if (is_signed && !binding_slot.signed_slot) {
    return "signed-view";
  }
  return "served";
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
