/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_command_processor.h"
#include "xenia/gpu/metal/metal_gamma_ramp_shader.h"
#include "xenia/gpu/metal/metal_present_convert_shader.h"

#import <Metal/Metal.h>

#include <mach/mach.h>
#include <objc/message.h>

// Public ObjC ABI for manual autorelease-pool scopes that span calls (the
// per-frame pool cycled at IssueSwap can't be an @autoreleasepool block).
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <set>
#include <cstring>
#include <chrono>
#include <exception>
#include <string>
#include <unordered_set>
#include <vector>

// Headers required by the PM4 execute-loop template included at the bottom of
// this file (mirrors the generic set vulkan_command_processor.cc pulls in).
#include "third_party/xxhash/xxhash.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/profiling.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/edram_layout.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/metal/metal_binding_layout.h"
#include "xenia/gpu/metal/metal_cache_clear_admission.h"
#include "xenia/gpu/metal/metal_edram_authority.h"
#include "xenia/gpu/metal/metal_fetch_overlap.h"
#include "xenia/gpu/metal/metal_fixed_function_state.h"
#include "xenia/gpu/metal/metal_integer_scale_publish.h"
#include "xenia/gpu/metal/metal_memexport_gap_report.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_primitive_processor.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_shared_texture_cache.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/metal/metal_texture_load_eligibility.h"
#include "xenia/gpu/metal/metal_upload_arena.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/resolve_writeback_log.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"
#include "xenia/gpu/xenos_zpd_report.h"
#include "xenia/kernel/user_module.h"
#include "xenia/ui/metal/metal_presenter.h"
#include "xenia/ui/metal/metal_provider.h"
#include "xenia/ui/presenter.h"

// Metal Shader Converter runtime (declarations only; the implementation TU is
// ir_runtime_impl.mm). Provides IRDescriptorTableEntry, the kIR*BindPoint slot
// constants, IRDescriptorTableSet*, and the IRRuntimeDraw* helpers.
#include "metal_irconverter_runtime.h"

DEFINE_bool(
    metal_present_format_convert, true,
    "When the selected swap source's pixel format differs from the guest "
    "output format, convert it per-channel instead of replicating the red "
    "channel. Enabled by default: the red-replication path was a debug "
    "visualization that a real title reaches -- Halo 3 presents an "
    "RGBA16Float source for every swap, and its guest image is fully "
    "chromatic, so it was being flattened to grey. Set false to restore the "
    "previous output exactly.",
    "GPU");
// Without this, the flip above reaches NOBODY who has already launched Xenia:
// the config file persists every cvar, so a stored `false` -- written while
// false was the default -- wins over the new compiled default. Measured, not
// assumed: the first default-on run still reported the cvar as false.
//
// UPDATE_from_bool migrates configs that still hold the OLD DEFAULT while
// preserving any value a user chose deliberately, which is exactly the intent
// here: adopt the fix, but never override someone who turned it off on purpose.
UPDATE_from_bool(metal_present_format_convert, 2026, 7, 29, 13, false);

DEFINE_bool(
    metal_guest_gamma_ramp, true,
    "Apply the guest's standard 256-entry gamma ramp to the presented guest "
    "output on Metal. Enabled by default: five calibrated titles program a "
    "complete non-identity ramp that is present at every swap, and ignoring it "
    "presented an untransformed image. Set false to restore the previous "
    "output exactly -- no ramp buffer is created, no upload occurs, and the "
    "swap source reaches the mailbox by the same blit as before.",
    "GPU");
// The compiled default alone reaches nobody who has already launched Xenia --
// measured during the C2-a promotion, where a persisted `false` beat a new
// `true` default. The hour is deliberately LATER than the C2-a update
// (2026-07-29 13) so configs that recorded this cvar as false while it was
// experimental also adopt the promoted default; a value a user set on purpose
// is still preserved.
UPDATE_from_bool(metal_guest_gamma_ramp, 2026, 7, 29, 14, false);

namespace xe {
namespace gpu {
namespace metal {

namespace {

// Process physical footprint in MiB, or 0.0 if the query fails. One sampler
// for every consumer (frame telemetry and cache-clear resource evidence) so
// they cannot drift into measuring subtly different things. Internal linkage:
// there is no header or API consumer outside this translation unit.
double ProcessFootprintMiB() {
  task_vm_info_data_t vm_info;
  mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&vm_info),
                &vm_count) != KERN_SUCCESS) {
    return 0.0;
  }
  return double(vm_info.phys_footprint) / (1024.0 * 1024.0);
}

// Concrete task-3.1 RamWritePublicationTarget: forwards the seam's two
// invalidation primitives to the operations that already exist on this
// command processor's shared memory / legacy texture cache. Holds no state of
// its own and adds no new watch -- it is a thin adapter so PublishRamWrite can
// EXECUTE a writer's publication instead of only predicting it.
class MetalRamWritePublicationTarget : public metal::RamWritePublicationTarget {
 public:
  MetalRamWritePublicationTarget(SharedMemory* shared_memory,
                                 metal::MetalTextureCache* texture_cache,
                                 metal::MetalTextureCache::InvalidationScope
                                     texture_invalidation_scope =
                                         metal::MetalTextureCache::
                                             InvalidationScope::
                                                 kAllRepresentations)
      : shared_memory_(shared_memory),
        texture_cache_(texture_cache),
        texture_invalidation_scope_(texture_invalidation_scope) {}

  bool InvalidateSharedPages(uint32_t base, uint32_t length,
                             bool exact_range) override {
    if (!shared_memory_) {
      return false;
    }
    // Resolve convergence invalidates the shared pages and fires per-range
    // watches so the shared texture cache reloads, but the matching legacy
    // snapshot bridge must survive. The preceding resolve-drain call site
    // invalidates legacy RAM uploads with kRamUploadsOnly, so suppress only
    // duplicate global-watch fan-out for this self-publication.
    bool fire_global_watches =
        texture_invalidation_scope_ == metal::MetalTextureCache::
                                           InvalidationScope::
                                               kAllRepresentations;
    shared_memory_->MemoryInvalidationCallback(base, length, exact_range,
                                               fire_global_watches);
    return true;
  }

  bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                uint32_t* invalidated_entries) override {
    if (!texture_cache_) {
      return false;
    }
    *invalidated_entries = texture_cache_->InvalidateRange(
        base, length, texture_invalidation_scope_);
    return true;
  }

 private:
  SharedMemory* shared_memory_;
  metal::MetalTextureCache* texture_cache_;
  metal::MetalTextureCache::InvalidationScope texture_invalidation_scope_;
};

// Concrete task-3.3 memexport destination-preparation target: forwards to the
// existing SharedMemory::RequestRange the direct call site used before.
class MetalMemexportPreparationTarget
    : public metal::MemexportPreparationTarget {
 public:
  explicit MetalMemexportPreparationTarget(SharedMemory* shared_memory)
      : shared_memory_(shared_memory) {}

  bool RequestDestinationRange(uint32_t base, uint32_t length) override {
    return shared_memory_ && shared_memory_->RequestRange(base, length);
  }

 private:
  SharedMemory* shared_memory_;
};

// Concrete task-3.3 memexport GPU-write publication target: forwards to the
// existing SharedMemory::RangeWrittenByGpu the direct call site used before.
class MetalMemexportPublicationTarget
    : public metal::MemexportPublicationTarget {
 public:
  explicit MetalMemexportPublicationTarget(SharedMemory* shared_memory)
      : shared_memory_(shared_memory) {}

  bool CanMarkRangeWrittenByGpu() const override {
    return shared_memory_ != nullptr;
  }

  void MarkRangeWrittenByGpu(uint32_t base, uint32_t length) override {
    shared_memory_->RangeWrittenByGpu(base, length);
  }

 private:
  SharedMemory* shared_memory_;
};

// Concrete task-3.3 memexport RAM-convergence target: copies the completed
// export's exact bytes from the shared buffer to guest RAM and invalidates
// overlapping legacy texture-cache entries (the same direct call the drain
// used before). It never touches shared-memory page validity -- the shared
// buffer is the authoritative source of the converged bytes and is left
// exactly as-is.
class MetalMemexportConvergenceTarget
    : public metal::MemexportConvergenceTarget {
 public:
  MetalMemexportConvergenceTarget(Memory* memory,
                                  MetalSharedMemory* shared_memory,
                                  metal::MetalTextureCache* texture_cache)
      : memory_(memory),
        shared_memory_(shared_memory),
        texture_cache_(texture_cache) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries) override {
    if (!memory_ || !shared_memory_) {
      return false;
    }
    uint8_t* dest = memory_->TranslatePhysical<uint8_t*>(base);
    id<MTLBuffer> shared_buffer =
        (__bridge id<MTLBuffer>)shared_memory_->GetBuffer();
    const uint8_t* contents =
        static_cast<const uint8_t*>(shared_buffer.contents);
    if (!dest || !contents) {
      return false;
    }
    std::memcpy(dest, contents + base, length);
    if (texture_cache_) {
      // Task 5.3 proved the earlier RangeWrittenByGpu watch is equivalent in
      // normal mode, but trace playback unregisters that global watch and
      // skips queue draining. Keep this direct retirement globally so replayed
      // memexports cannot leave aliases or resolved bridges stale.
      *invalidated_entries = texture_cache_->InvalidateRange(base, length);
    }
    return true;
  }

 private:
  Memory* memory_;
  MetalSharedMemory* shared_memory_;
  metal::MetalTextureCache* texture_cache_;
};

// Snapshot-bridge lineage probe (XE_METAL_LINEAGE=<base hex>): writer-side half.
// Same base + strided-hash as the reader (metal_texture_cache.mm) so the
// resolve/writeback/drain events on the watched base share the fetch clock.
uint32_t xe_lineage_base() {
  static uint32_t b = [] {
    const char* v = std::getenv("XE_METAL_LINEAGE");
    return v ? uint32_t(strtoul(v, nullptr, 16)) : 0u;
  }();
  return b;
}
uint32_t xe_lineage_stridehash(const void* p, size_t n) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i += 37) h = (h ^ b[i]) * 16777619u;
  return h;
}

// Precise guest storage ranges of a texture fetch: base level(s) and mip chain
// as two DISJOINT ranges (mip_address is a separate pointer). Derived from the
// shared GetGuestTextureLayout, so compressed/tiled/linear/packed-mip/dimension
// semantics match the upload path exactly. Ends use 64-bit math clamped to
// UINT32_MAX so no address arithmetic wraps.
metal::FetchGuestRanges ComputeFetchGuestRanges(
    const xenos::xe_gpu_texture_fetch_t& f) {
  metal::FetchGuestRanges r;
  uint32_t w = 1, h = 1, d = 1;
  switch (f.dimension) {
    case xenos::DataDimension::k1D:
      w = f.size_1d.width + 1;
      break;
    case xenos::DataDimension::k2DOrStacked:
      w = f.size_2d.width + 1;
      h = f.size_2d.height + 1;
      d = f.stacked ? (f.size_2d.stack_depth + 1) : 1;
      break;
    case xenos::DataDimension::k3D:
      w = f.size_3d.width + 1;
      h = f.size_3d.height + 1;
      d = f.size_3d.depth + 1;
      break;
    case xenos::DataDimension::kCube:
      w = f.size_2d.width + 1;
      h = f.size_2d.height + 1;
      d = 6;
      break;
  }
  auto clamp_end = [](uint32_t begin, uint32_t size) -> uint32_t {
    uint64_t e = uint64_t(begin) + uint64_t(size);
    return e > 0xFFFFFFFFull ? 0xFFFFFFFFu : uint32_t(e);
  };
  texture_util::TextureGuestLayout lay = texture_util::GetGuestTextureLayout(
      f.dimension, f.pitch, w, h, d, f.tiled != 0, f.format,
      f.packed_mips != 0, /*has_base=*/true, f.mip_max_level);
  uint32_t base_addr = f.base_address << 12;
  r.base_begin = base_addr;
  r.base_end = clamp_end(base_addr, lay.base.level_data_extent_bytes);
  if (f.mip_max_level > 0 && lay.mips_total_extent_bytes) {
    uint32_t mip_addr = f.mip_address << 12;
    r.mip_begin = mip_addr;
    r.mip_end = clamp_end(mip_addr, lay.mips_total_extent_bytes);
  }
  return r;
}

// Entries inside the small fixed system view table.
constexpr uint32_t kSystemViewSharedMemorySRV = 0;
constexpr uint32_t kSystemViewNullUAV = 1;
// Shared memory as a writable UAV (u0, space0) for memexport draws — the
// top-level AB slot kAbUavSpace0 points here only while encoding a draw whose
// shader actually exports (non-exporting draws keep the null entry).
constexpr uint32_t kSystemViewSharedMemoryUAV = 2;
constexpr uint32_t kSystemViewEntryCount = 3;

// Sized for bring-up; a real build would grow these on demand.
constexpr uint32_t kViewBindlessHeapSlots = 4096;
constexpr uint32_t kSamplerBindlessHeapSlots = 256;
constexpr uint32_t kUploadArenaSize = 32u * 1024 * 1024;
// XE_METAL_ARENA_SIZE=<bytes>: ring-stress override — a small ring forces
// constant wrapping/reclaim pressure in trace replay, turning any ring
// lifetime bug into a deterministic headless corruption vs the baselines.
static uint32_t UploadArenaSize() {
  const char* v = std::getenv("XE_METAL_ARENA_SIZE");
  return v ? uint32_t(strtoul(v, nullptr, 10)) : kUploadArenaSize;
}

bool MapPrimitiveType(xenos::PrimitiveType type, MTLPrimitiveType* out) {
  switch (type) {
    case xenos::PrimitiveType::kPointList:
      *out = MTLPrimitiveTypePoint;
      return true;
    case xenos::PrimitiveType::kLineList:
      *out = MTLPrimitiveTypeLine;
      return true;
    case xenos::PrimitiveType::kLineStrip:
      *out = MTLPrimitiveTypeLineStrip;
      return true;
    case xenos::PrimitiveType::kTriangleList:
      *out = MTLPrimitiveTypeTriangle;
      return true;
    case xenos::PrimitiveType::kTriangleStrip:
      *out = MTLPrimitiveTypeTriangleStrip;
      return true;
    case xenos::PrimitiveType::kRectangleList:
      // 3 guest vertices define a rectangle whose 4th corner (v3 = v0 + v2 - v1)
      // the guest GPU synthesizes. IssueDraw expands rect-lists through MSC
      // geometry emulation (the GS emits the true 4-corner rect) BEFORE reaching
      // MapPrimitiveType, so this is only hit as the fallback when the device
      // lacks mesh-shader support (or there's no pixel shader). Drawing 3 verts as
      // one triangle covers only half the screen with degenerate UVs, so skip
      // (render nothing) rather than draw garbage. See openspec
      // reduce-apple-gpu-frametime handoff.
      return false;
    case xenos::PrimitiveType::kTriangleFan:
      // No host triangle-fan primitive; IssueDraw expands it to a triangle list
      // with a generated index buffer. Report the target host primitive here.
      *out = MTLPrimitiveTypeTriangle;
      return true;
    default:
      // Quads and patches need primitive-processor conversion or geometry
      // emulation. Guest line loops are also intentionally not mapped here:
      // PP3-auto maps only the shared processor's host kLineStrip result.
      return false;
  }
}

// SwapIndicesInto (guest index byte-swap into the host arena) and IsPp1Routable
// (the routing predicate) are the shared production contracts, defined once in
// metal_primitive_processor.h so the device-free conformance test exercises the
// same code rather than a replica.

}  // namespace

MetalCommandProcessor::MetalCommandProcessor(
    MetalGraphicsSystem* graphics_system, kernel::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state) {}
MetalCommandProcessor::~MetalCommandProcessor() = default;

bool MetalCommandProcessor::GetOrCreateSamplerSlot(uint32_t key, uint32_t* out_slot) {
  *out_slot = 0;
  auto it = sampler_slot_cache_.find(key);
  if (it != sampler_slot_cache_.end()) {
    *out_slot = it->second;
    return true;
  }
  if (next_sampler_slot_ >= kSamplerBindlessHeapSlots ||
      !sampler_bindless_heap_) {
    return false;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider())
          ->device();
  MTLSamplerDescriptor* d = [[MTLSamplerDescriptor alloc] init];
  auto minmag = [](uint32_t f) {
    // TextureFilter: 0=point, 1=linear, 2=base-map (no mip; min/mag linear
    // gives the least-surprising look), 3 resolved before packing.
    return f == 0 ? MTLSamplerMinMagFilterNearest
                  : MTLSamplerMinMagFilterLinear;
  };
  d.magFilter = minmag(key & 3);
  d.minFilter = minmag((key >> 2) & 3);
  uint32_t mip = (key >> 4) & 3;
  // Textures are single-mip for now; point/linear only matter once mip
  // chains upload. kBaseMap (2) = not mipmapped.
  d.mipFilter = mip == 1 ? MTLSamplerMipFilterLinear
                         : (mip == 0 ? MTLSamplerMipFilterNearest
                                     : MTLSamplerMipFilterNotMipmapped);
  auto addr = [](uint32_t c) {
    switch (xenos::ClampMode(c)) {
      case xenos::ClampMode::kRepeat:
        return MTLSamplerAddressModeRepeat;
      case xenos::ClampMode::kMirroredRepeat:
        return MTLSamplerAddressModeMirrorRepeat;
      case xenos::ClampMode::kMirrorClampToEdge:
      case xenos::ClampMode::kMirrorClampToHalfway:
      case xenos::ClampMode::kMirrorClampToBorder:
        return MTLSamplerAddressModeMirrorClampToEdge;
      case xenos::ClampMode::kClampToBorder:
        return MTLSamplerAddressModeClampToBorderColor;
      case xenos::ClampMode::kClampToEdge:
      case xenos::ClampMode::kClampToHalfway:
      default:
        return MTLSamplerAddressModeClampToEdge;
    }
  };
  d.sAddressMode = addr((key >> 6) & 7);
  d.tAddressMode = addr((key >> 9) & 7);
  d.rAddressMode = addr((key >> 12) & 7);
  uint32_t aniso = (key >> 15) & 7;
  if (aniso) {
    d.maxAnisotropy = std::min(16u, 1u << (aniso - 1));
  }
  id<MTLSamplerState> state = [device newSamplerStateWithDescriptor:d];
  if (!state) {
    return false;
  }
  uint32_t slot = next_sampler_slot_++;
  auto* entries = static_cast<IRDescriptorTableEntry*>(
      [(__bridge id<MTLBuffer>)sampler_bindless_heap_ contents]);
  IRDescriptorTableSetSampler(&entries[slot], state, 0.0f);
  sampler_states_.push_back((__bridge_retained void*)state);
  sampler_slot_cache_.emplace(key, slot);
  *out_slot = slot;
  return true;
}

void MetalCommandProcessor::TracePlaybackWroteMemory(uint32_t base_ptr,
                                                     uint32_t length) {
  // Trace restore publishes bytes straight to RAM. Route the publication
  // through the typed task-3.1 seam: it invalidates the shared-memory pages
  // the trace stream just rewrote (so the next RequestConsumedRange
  // re-uploads them, mirroring the Vulkan backend) using the SAME
  // MemoryInvalidationCallback this call site used directly before.
  //
  // The direct MetalTextureCache::InvalidateRange call is durably RETAINED:
  // trace playback unregisters the legacy global watch, so the seam's policy
  // for kTraceRestore also requests legacy invalidation rather than relying on
  // watch fan-out. Task 5.3's writer-matrix comparison proved this leg is not
  // redundant (the legacy global-watch set is empty after SetTracePlayback), so
  // it stays by design. SetTracePlayback() must still run first regardless of
  // range count.
  if (texture_cache_) {
    texture_cache_->SetTracePlayback();
  }
  auto publication = metal::DecideRamWritePublication(
      metal::WriterKind::kTraceRestore, base_ptr, length);
  if (publication) {
    MetalRamWritePublicationTarget target(shared_memory_.get(),
                                          texture_cache_.get());
    metal::PublishRamWrite(*publication, target);
  }
}

void MetalCommandProcessor::RestoreEdramSnapshot(const void* snapshot) {
  // Trace playback: seed the EDRAM ownership + contents captured before the
  // frame (JC2 resolves the previous frame's depth/color early in the frame).
  if (render_target_cache_) {
    render_target_cache_->RestoreEdramSnapshot(snapshot);
  }
}

bool MetalCommandProcessor::DumpEdramToBuffer(
    std::vector<uint8_t>& edram_out) {
  if (!render_target_cache_) {
    return false;
  }
  // Flush all pending GPU work so texture contents are final. Mid-frame this
  // is a commit + wait; the next draw's BeginFrameEncoder starts a fresh
  // command buffer and LOADs the (persistent) render-target contents.
  EndFrameEncoderAndWait();
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue()) {
    return false;
  }
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)provider->command_queue();
  if (render_target_cache_->sample_edram_enabled()) {
    // Canonical mode capture contract: snapshots, full dumps, and the base
    // command processor's per-tile/sample hashes all read the same private raw
    // physical-layout buffer. Attachment writes begin advancing it in tasks
    // 3.x; until then this intentionally exposes restored canonical state.
    if (!render_target_cache_->PrepareCanonicalEdramForReadback()) {
      return false;
    }
    // The pack operations above use the ordinary frame command buffer so they
    // remain ordered after the producing draws. Complete them before the dump
    // command buffer reads the private canonical buffer.
    EndFrameEncoderAndWait();
    return render_target_cache_->DumpCanonicalEdram((__bridge void*)queue,
                                                     edram_out);
  }
  std::vector<MetalRenderTargetCache::EdramOwnerDump> owners;
  render_target_cache_->GetEdramOwnersForDump(owners);
  // The EDRAM image: 2048 tiles x 80x16 samples x 4 bytes, tile-major.
  constexpr uint32_t kTileBytes = 80 * 16 * 4;
  edram_out.assign(xenos::kEdramSizeBytes, 0);
  std::vector<uint8_t>& edram = edram_out;
  static const bool s_dump_owner_log = std::getenv("XE_METAL_DUMP_OWNER_LOG");
  for (const auto& owner : owners) {
    if (owner.is_64bpp) {
      continue;  // 64bpp encode not implemented in the dump yet.
    }
    if (s_dump_owner_log) {
      id<MTLTexture> ot = (__bridge id<MTLTexture>)owner.texture;
      XELOGW("MetalDumpOwner: tex={} {}x{} depth={} msaa={} base={} pitch={} "
             "rows[{},{})",
             owner.texture, ot.width, ot.height, owner.is_depth,
             owner.msaa_samples, owner.base_tiles, owner.pitch_tiles,
             owner.row_first, owner.row_first + owner.rows);
    }
    // Per-owner autorelease pool: with per-draw capture (--capture_edram_*)
    // this runs thousands of times on a pool-less replay thread; the
    // autoreleased command buffer/encoder otherwise retain the multi-MB
    // staging buffers until the process OOMs (SIGKILL ~2600 draws in).
    @autoreleasepool {
    id<MTLTexture> tex = (__bridge id<MTLTexture>)owner.texture;
    uint32_t tw = uint32_t(tex.width), th = uint32_t(tex.height);
    // Read the whole owner texture back once. Combined depth-stencil owners
    // read as two plane blits: depth floats (4B/px) + stencil bytes (1B/px,
    // appended after the depth plane in the staging buffer).
    bool combined_ds =
        owner.is_depth && tex.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
    id<MTLDevice> dev = tex.device;
    size_t depth_plane_bytes = size_t(tw) * th * 4;
    id<MTLBuffer> staging =
        [dev newBufferWithLength:depth_plane_bytes +
                                 (combined_ds ? size_t(tw) * th : 0)
                         options:MTLResourceStorageModeShared];
    if (!staging) {
      continue;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if (combined_ds) {
      [blit copyFromTexture:tex
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(tw, th, 1)
                   toBuffer:staging
          destinationOffset:0
     destinationBytesPerRow:tw * 4
   destinationBytesPerImage:depth_plane_bytes
                    options:MTLBlitOptionDepthFromDepthStencil];
      [blit copyFromTexture:tex
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(tw, th, 1)
                   toBuffer:staging
          destinationOffset:depth_plane_bytes
     destinationBytesPerRow:tw
   destinationBytesPerImage:size_t(tw) * th
                    options:MTLBlitOptionStencilFromDepthStencil];
    } else {
      [blit copyFromTexture:tex
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(tw, th, 1)
                   toBuffer:staging
          destinationOffset:0
     destinationBytesPerRow:tw * 4
   destinationBytesPerImage:depth_plane_bytes];
    }
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint8_t* pixels = static_cast<const uint8_t*>(staging.contents);
    const uint8_t* stencil_plane =
        combined_ds ? pixels + depth_plane_bytes : nullptr;
    // msaa_samples is the xenos::MsaaSamples ENUM (k1X=0, k2X=1, k4X=2), not a
    // sample count: 2x+ doubles vertical, 4x also doubles horizontal. The old
    // >=3/>=2 count-style thresholds read 4x owners with xs=0 — the right half
    // of every sample row fell out of bounds and dumped as zeros (the phantom
    // "half-screen" divergence at the 7282/6756 draw-25 MSAA clears).
    EdramMsaaShifts msaa_shifts = GetEdramMsaaShifts(owner.msaa_samples);
    // The rect covers full-EDRAM 16-tile rows [row_first, row_first+rows).
    for (uint32_t er = owner.row_first; er < owner.row_first + owner.rows;
         ++er) {
      for (uint32_t ec = 0; ec < 16; ++ec) {
        uint32_t tile = er * 16 + ec;
        if (tile < owner.base_tiles) {
          continue;
        }
        uint32_t local = tile - owner.base_tiles;
        if (owner.pitch_tiles == 0 || local / owner.pitch_tiles >= 100000) {
          continue;
        }
        uint8_t* tile_out = edram.data() + size_t(tile) * kTileBytes;
        // Per sample of the 80x16 tile: source pixel = sample >> shifts.
        // Depth EDRAM addressing swaps each tile's 40-sample column halves
        // (see the reference dump shader): word position sx holds the host
        // sample from the other half.
        for (uint32_t sy = 0; sy < 16; ++sy) {
          for (uint32_t sx = 0; sx < 80; ++sx) {
            uint32_t host_sx = owner.is_depth
                                   ? EdramDepthSwapX(
                                         sx, xenos::kEdramTileWidthSamples)
                                   : sx;
            uint32_t sample_x = (local % owner.pitch_tiles) * 80 + host_sx;
            uint32_t sample_y = (local / owner.pitch_tiles) * 16 + sy;
            uint32_t px = sample_x >> msaa_shifts.x;
            uint32_t py = sample_y >> msaa_shifts.y;
            uint32_t word = 0;
            if (px < tw && py < th) {
              const uint8_t* p = pixels + (size_t(py) * tw + px) * 4;
              if (owner.is_depth) {
                float d;
                std::memcpy(&d, p, 4);
                // unorm24 encode must be roundEven(d * 0xFFFFFF) to match
                // the guest-visible conversion (see the Vulkan dump shader:
                // +0.5-and-truncate is explicitly the WRONG conversion, and
                // biases the whole depth plane by up to 2 ULP24).
                uint32_t d24 =
                    owner.is_float24_depth
                        ? xenos::Float32To20e4(d, false)
                        : uint32_t(std::rint(
                              std::min(1.0f, std::max(0.0f, d)) *
                              16777215.0f));
                uint32_t stencil =
                    stencil_plane ? stencil_plane[size_t(py) * tw + px] : 0u;
                word = PackEdramDepthStencil(d24, stencil);
              } else if (owner.pixel_format == uint64_t(MTLPixelFormatR32Float)) {
                std::memcpy(&word, p, 4);
              } else {
                // Canonical RGBA8 storage: bytes ARE the guest EDRAM word.
                word = PackEdramColor8888(p[0], p[1], p[2], p[3]);
              }
            }
            std::memcpy(tile_out + (size_t(sy) * 80 + sx) * 4, &word, 4);
          }
        }
      }
    }
    }  // autoreleasepool
  }
  return true;
}

// Frame command buffers carry per-encoder execution status so a GPU fault
// names the encoder that caused it (labels are set at each encoder open). The
// alternative — a bare [queue commandBuffer] — reports only "Caused GPU
// Address Fault Error" with no attribution.
static id<MTLCommandBuffer> MakeAttributedCommandBuffer(
    id<MTLCommandQueue> queue) {
  MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
  desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
  return [queue commandBufferWithDescriptor:desc];
}

// Logs the per-encoder execution status carried by a faulted command buffer
// (requires MakeAttributedCommandBuffer): which encoders faulted vs completed.
static void LogCommandBufferFault(id<MTLCommandBuffer> command_buffer,
                                  const char* site, uint32_t swap_count) {
  NSError* err = command_buffer.error;
  XELOGE("Metal GPU FAULT ({}, swap#{}): {} (code {})", site, swap_count,
         err.localizedDescription ? err.localizedDescription.UTF8String
                                  : "(no description)",
         err ? long(err.code) : -1L);
  NSArray<id<MTLCommandBufferEncoderInfo>>* infos =
      err.userInfo[MTLCommandBufferEncoderInfoErrorKey];
  for (id<MTLCommandBufferEncoderInfo> info in infos) {
    const char* state = "unknown";
    switch (info.errorState) {
      case MTLCommandEncoderErrorStateCompleted: state = "completed"; break;
      case MTLCommandEncoderErrorStateAffected: state = "AFFECTED"; break;
      case MTLCommandEncoderErrorStateUnknown: state = "unknown"; break;
      case MTLCommandEncoderErrorStatePending: state = "pending"; break;
      case MTLCommandEncoderErrorStateFaulted: state = "FAULTED"; break;
    }
    // Only the interesting ones — a frame can have hundreds of completed
    // encoders.
    if (info.errorState != MTLCommandEncoderErrorStateCompleted) {
      XELOGE("  encoder '{}': {}",
             info.label ? info.label.UTF8String : "(unlabeled)", state);
    }
  }
}

void* MetalCommandProcessor::PrepareTransferCommandBuffer() {
  // A transfer is a pass boundary: close the open render encoder; ensure the
  // frame command buffer exists so the blits order before the next pass.
  EndRenderEncoder();
  if (!frame_command_buffer_) {
    auto* provider =
        static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
    if (!provider || !provider->command_queue()) {
      return nullptr;
    }
    if (S7ArenaResetOff() && upload_arena_) {
      upload_arena_->Reset();
    }
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)provider->command_queue();
    id<MTLCommandBuffer> command_buffer = MakeAttributedCommandBuffer(queue);
    frame_command_buffer_ = (__bridge_retained void*)command_buffer;
  }
  return frame_command_buffer_;
}

// ---------------------------------------------------------------------------
// DEBUG TOGGLES (environment variables) — the Metal backend is under active
// bring-up; these enable fast in-agent/on-device bisection without rebuilds.
//   XE_METAL_DISABLE_GEO=1     rect/point geometry emulation off (skip fallback)
//   XE_METAL_NO_SIGNS=1        texture signedness/gamma decode off
//   XE_METAL_NO_CENTROID=1     force center interpolation (no centroid at MSAA)
//   XE_METAL_NO_SMALL_DEPTH=1  depth test off for small-pitch (lighting) draws
//   XE_METAL_NO_WRITEBACK=1    small-resolve guest-memory writeback off
//   XE_METAL_WB_HOLD=1         enable diagnostic exposure hold-last-value (A/B;
//                              default off so canonical zero stays exact)
//   XE_METAL_SERIALIZE_RESOLVES=1  commit+wait after each resolve (hazard A/B)
//   XE_METAL_NO_RESOLVE_COMMIT=1   disable the default commit-before-resolve
//   XE_METAL_STAGE3=0         disable stage 3 (guest-RAM resolve reads/writes,
//                             DEFAULT ON since the copy_dest_swap fix: gate
//                             at-or-better on every trace and SF4's column
//                             resolves need exact-address placement)
//   XE_METAL_COLOR_BRIDGE=1    restore the color RT-as-texture bridge (A/B;
//                             default = fetches read guest RAM, stage 3)
//   XE_METAL_FAULT_INJECT=1    deliberately reproduce the bindless page fault
//   XE_METAL_EXPO_TRACE=hash  per-swap exposure-loop telemetry (MetalExpo
//                             lines: wb value / tonemap-PS c0 / present luma)
//   XE_METAL_DUMP_RT=N        dump bound color RT texture to /tmp at draw N
//   XE_METAL_CHAIN_TRACE=hex,hex  per-swap 4px fingerprint of each base's
//                             snapshot (MetalChain lines; find what alternates)
//   XE_METAL_FLUSH_ALL_UPLOADS=1   flush before any mid-frame upload (hazard A/B)
//   XE_METAL_SAMPLE_EDRAM_SELF_CHECK=1  validate a restored canonical snapshot
//                             with a full 10 MiB dump/materialize/pack pass
//   XE_METAL_PRESENT_BASE=hex  present that resolved snapshot + exact readback
//   XE_METAL_PEEK=hex,hex,...  log guest-RAM words at addresses (ground truth)
//   XE_METAL_DUMP_PS=hexhash   dump that PS + paired VS ucode disasm to /tmp
// Diagnostics logged: MetalRTConfig (unique RT configs), MetalZCfg (depth
// configs), MetalLightProducer (small-pitch draw inputs), MetalRTConsumer
// (snapshot-sampling passes), MetalB1 (packed-vs-direct constants),
// MetalDepthResolve / resolve-owner / resolve-clear / writeback one-shots.
// ---------------------------------------------------------------------------

// Crash-capture: JC2-on-Metal aborts recurrently via _objc_terminate — an
// uncaught ObjC/Metal exception (e.g. an MTL validation error thrown from a
// background thread) whose MESSAGE is lost in the bare abort. ObjC exceptions
// are thrown through the C++ ABI on arm64, so std::current_exception() captures
// the NSException and it can be @caught and logged. Installed once at Metal
// backend setup; chains to the previous handler so behavior is otherwise
// unchanged. Pair with MTL_DEBUG_LAYER=1 (and MallocGuardEdges for the
// separate double-free track) when hunting the abort on-device.
namespace {
std::terminate_handler g_prev_terminate = nullptr;
void MetalTerminateHandler() {
  std::exception_ptr eptr = std::current_exception();
  if (eptr) {
    // @catch only accepts ObjC pointer types; route everything else back
    // through a nested C++ catch to recover the std::exception message.
    @try {
      std::rethrow_exception(eptr);
    } @catch (NSException* ex) {
      XELOGE("Metal FATAL: uncaught NSException {}: {}",
             ex.name ? ex.name.UTF8String : "(null)",
             ex.reason ? ex.reason.UTF8String : "(null)");
      fprintf(stderr, "Metal FATAL: uncaught NSException %s: %s\n",
              ex.name ? ex.name.UTF8String : "(null)",
              ex.reason ? ex.reason.UTF8String : "(null)");
      for (NSString* frame in ex.callStackSymbols) {
        XELOGE("  {}", frame.UTF8String);
      }
    } @catch (...) {
      try {
        std::rethrow_exception(eptr);
      } catch (const std::exception& e) {
        XELOGE("Metal FATAL: uncaught std::exception: {}", e.what());
      } catch (...) {
        XELOGE("Metal FATAL: uncaught non-standard exception");
      }
    }
  }
  if (g_prev_terminate) {
    g_prev_terminate();
  }
  std::abort();
}
void InstallMetalCrashCapture() {
  static bool s_installed = false;
  if (s_installed) {
    return;
  }
  s_installed = true;
  g_prev_terminate = std::set_terminate(&MetalTerminateHandler);
}
}  // namespace

bool MetalCommandProcessor::SetupContext() {
  // Effective submission/lifetime mode banner: config sessions were getting
  // misidentified from fps profiles alone — the log states the truth.
  XELOGI(
      "MetalCommandProcessor: MODE submission={} arena={} ({} MB) retire-tags={}",
      (std::getenv("XE_METAL_SYNC_WAIT") || !std::getenv("XE_METAL_ASYNC"))
          ? "SYNC-WAIT"
          : "async",
      S7ArenaResetOff() ? "reset-per-buffer" : "ring",
      UploadArenaSize() / (1024 * 1024), S7TagsOff() ? "OFF" : "on");
  if (!CommandProcessor::SetupContext()) {
    return false;
  }
  InstallMetalCrashCapture();
  // Surface active debug toggles prominently so logs are self-describing.
  static const char* kDebugEnvs[] = {
      "XE_METAL_DISABLE_GEO",    "XE_METAL_NO_SIGNS",
      "XE_METAL_NO_CENTROID",    "XE_METAL_NO_SMALL_DEPTH",
      "XE_METAL_NO_WRITEBACK",   "XE_METAL_WB_HOLD",
      "XE_METAL_SERIALIZE_RESOLVES", "XE_METAL_FLUSH_ALL_UPLOADS",
      "XE_METAL_FAULT_INJECT",  "XE_METAL_EXPO_TRACE",
      "XE_METAL_CHAIN_TRACE",   "XE_METAL_NO_RESOLVE_COMMIT",
      "XE_METAL_COLOR_BRIDGE",  "XE_METAL_STAGE3",
      "XE_METAL_SAMPLE_EDRAM",  "XE_METAL_SAMPLE_EDRAM_RASTER",
      "XE_METAL_CROSSMSAA_SHADOW",
      "XE_METAL_SAMPLE_EDRAM_FULL_SYNC",
      "XE_METAL_SAMPLE_EDRAM_SELF_CHECK",
      "XE_METAL_CANONICAL_DEPTH",
      "XE_METAL_MEMEXPORT",
      "XE_METAL_DUMP_RT",
      "XE_METAL_PRESENT_BASE",   "XE_METAL_PEEK",
      "XE_METAL_DUMP_PS",
  };
  for (const char* env : kDebugEnvs) {
    if (const char* value = std::getenv(env)) {
      XELOGW("MetalCommandProcessor: DEBUG TOGGLE {}={}", env, value);
    }
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  void* device = provider ? provider->device() : nullptr;
  if (!device) {
    XELOGE("MetalCommandProcessor: no Metal device from provider");
    return false;
  }
  pipeline_cache_ = std::make_unique<MetalPipelineCache>(device);
  if (!pipeline_cache_->Initialize()) {
    XELOGE("MetalCommandProcessor: pipeline cache init failed");
    return false;
  }
  render_target_cache_ = std::make_unique<MetalRenderTargetCache>(
      *register_file_, *memory_, &trace_writer_, *this, device);
  if (!render_target_cache_->Initialize()) {
    XELOGE("MetalCommandProcessor: render target cache init failed");
    return false;
  }
  shared_memory_ = std::make_unique<MetalSharedMemory>(*memory_, device);
  shared_memory_->SetTraceWriter(&trace_writer_);
  shared_memory_->SetSubmissionProvider([this]() {
    return S7TagsOff() ? 0 : CurrentEncodingSubmission();
  });
  if (!shared_memory_->Initialize()) {
    XELOGE("MetalCommandProcessor: shared memory init failed");
    return false;
  }
  // Intra-frame rewrite hazard: commit + wait the open frame so already-
  // encoded draws consume the pre-rewrite bytes (also resets the consumed-
  // range set). Runs from inside RequestConsumedRange, which is only called
  // before the draw's encoder opens.
  // Must WAIT: the caller is about to memcpy over shared-buffer ranges that
  // in-flight submissions read.
  shared_memory_->SetHazardFlushCallback([this]() { EndFrameEncoderAndWait(); });
  // PP1 + PP2-auto: the shared PrimitiveProcessor normalization authority for
  // routed native restart-disabled line/triangle list/strip draws and
  // auto-indexed triangle fans. Depends on shared_memory_ (constructed above).
  // A failure leaves primitive_processor_ null and every draw falls back to the
  // inline path (behavior-preserving), including auto fans.
  primitive_processor_ = std::make_unique<MetalPrimitiveProcessor>(*register_file_, *memory_,
                                                                   trace_writer_, *shared_memory_);
  if (!primitive_processor_->Initialize()) {
    XELOGE("MetalCommandProcessor: primitive processor init failed; routing "
           "disabled, all draws stay on the inline path");
    primitive_processor_.reset();
  }
  if (!InitializeDrawResources(device)) {
    XELOGE("MetalCommandProcessor: draw resource init failed");
    return false;
  }
  return true;
}

bool MetalCommandProcessor::InitializeDrawResources(void* device_ptr) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;

  // Mesh-shader support gates the geometry-emulation (rectangle/point/quad) path.
  // Apple Silicon is Apple7+; Mac2 covers AMD on macOS. Mirrors XeniOS.
  mesh_shader_supported_ = [device supportsFamily:MTLGPUFamilyApple7] ||
                           [device supportsFamily:MTLGPUFamilyMac2];
  // Diagnostic A/B: XE_METAL_DISABLE_GEO=1 forces the rect-list SKIP fallback so
  // the geometry-emulation output can be compared against no-geometry.
  if (std::getenv("XE_METAL_DISABLE_GEO")) {
    mesh_shader_supported_ = false;
  }

  upload_arena_ = std::make_unique<MetalUploadArena>(device_ptr, UploadArenaSize());
  if (!upload_arena_->Initialize()) {
    return false;
  }

  // Small buffer standing in for inactive CBV slots so MSC never dereferences a
  // null gpuAddress.
  id<MTLBuffer> null_buffer =
      [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
  if (!null_buffer) {
    return false;
  }
  std::memset([null_buffer contents], 0, 256);
  null_buffer_ = (__bridge_retained void*)null_buffer;

  // 1x1 null texture + default sampler so unbound bindless samples are safe.
  // Bring-up diagnostic: fill it mid-gray (not black) so geometry that samples
  // still-unimplemented textures shows up as visible gray shapes instead of
  // rendering black — proving the geometry is there and the gap is the texture
  // cache. (Turns black post-process composites gray.)
  // MUST be a 2D ARRAY texture (see MetalTextureCache): the guest shaders sample
  // texture2d_array, so a plain 2D null texture would sample as garbage.
  MTLTextureDescriptor* null_desc = [[MTLTextureDescriptor alloc] init];
  null_desc.textureType = MTLTextureType2DArray;
  null_desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  null_desc.width = 1;
  null_desc.height = 1;
  null_desc.arrayLength = 1;
  null_desc.mipmapLevelCount = 1;
  null_desc.usage = MTLTextureUsageShaderRead;
  null_desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> null_texture = [device newTextureWithDescriptor:null_desc];
  if (!null_texture) {
    return false;
  }
  uint8_t gray_pixel[4] = {128, 128, 128, 255};
  [null_texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                  mipmapLevel:0
                        slice:0
                    withBytes:gray_pixel
                  bytesPerRow:4
                bytesPerImage:0];
  null_texture_ = (__bridge_retained void*)null_texture;
  id<MTLSamplerState> null_sampler =
      [device newSamplerStateWithDescriptor:[[MTLSamplerDescriptor alloc] init]];
  null_sampler_ = (__bridge_retained void*)null_sampler;

  // Bindless view/sampler heaps (arrays of IRDescriptorTableEntry). Fill every
  // slot with the null texture/sampler so any index is valid.
  id<MTLBuffer> view_heap = [device
      newBufferWithLength:kViewBindlessHeapSlots * sizeof(IRDescriptorTableEntry)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeWriteCombined];
  id<MTLBuffer> sampler_heap = [device
      newBufferWithLength:kSamplerBindlessHeapSlots *
                          sizeof(IRDescriptorTableEntry)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeWriteCombined];
  if (!view_heap || !sampler_heap) {
    return false;
  }
  auto* view_entries =
      static_cast<IRDescriptorTableEntry*>([view_heap contents]);
  for (uint32_t i = 0; i < kViewBindlessHeapSlots; ++i) {
    IRDescriptorTableSetTexture(&view_entries[i], null_texture, 0.0f, 0);
  }
  auto* sampler_entries =
      static_cast<IRDescriptorTableEntry*>([sampler_heap contents]);
  for (uint32_t i = 0; i < kSamplerBindlessHeapSlots; ++i) {
    IRDescriptorTableSetSampler(&sampler_entries[i], null_sampler, 0.0f);
  }
  view_bindless_heap_ = (__bridge_retained void*)view_heap;
  sampler_bindless_heap_ = (__bridge_retained void*)sampler_heap;

  // Fixed system view table: entry 0 = shared-memory SRV, entry 1 = null UAV.
  id<MTLBuffer> system_view = [device
      newBufferWithLength:kSystemViewEntryCount * sizeof(IRDescriptorTableEntry)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeWriteCombined];
  if (!system_view) {
    return false;
  }
  auto* system_entries =
      static_cast<IRDescriptorTableEntry*>([system_view contents]);
  id<MTLBuffer> shared_buffer =
      (__bridge id<MTLBuffer>)shared_memory_->GetBuffer();
  IRDescriptorTableSetBuffer(&system_entries[kSystemViewSharedMemorySRV],
                             [shared_buffer gpuAddress],
                             MetalSharedMemory::kBufferSize);
  IRDescriptorTableSetBuffer(&system_entries[kSystemViewNullUAV],
                             [null_buffer gpuAddress], 256);
  IRDescriptorTableSetBuffer(&system_entries[kSystemViewSharedMemoryUAV],
                             [shared_buffer gpuAddress],
                             MetalSharedMemory::kBufferSize);
  system_view_tables_ = (__bridge_retained void*)system_view;

  texture_cache_ = std::make_unique<MetalTextureCache>(
      device_ptr, memory_, shared_memory_.get(), view_bindless_heap_,
      kViewBindlessHeapSlots);
  texture_cache_->SetTraceWriter(&trace_writer_);
  // The same null texture every heap slot is initialized to, so a slot freed
  // by a cache clear returns to a valid resource rather than an empty entry.
  texture_cache_->SetNullTexture(null_texture_);
  texture_cache_->SetSubmissionProvider([this]() {
    return S7TagsOff() ? 0 : CurrentEncodingSubmission();
  });
  // C1T-R shared TextureCache subclass route (canonical for strict tiled
  // single-mip 2D k_8_8_8_8 kRamContent misses; the legacy cache serves
  // everything else and remains the universal fallback). All wiring uses
  // existing mechanisms only (seam audit M1-M5): the legacy cache is
  // the slot/residency/retirement authority, loads encode through
  // PrepareTransferCommandBuffer, lifetime keys off the same submission ids,
  // and faulted submissions are queried from the existing GPU-thread fault
  // recording.
  shared_texture_cache_ = std::make_unique<MetalSharedTextureCache>(
      *register_file_, *shared_memory_, device_ptr, texture_cache_.get());
  shared_texture_cache_->SetPrepareTransferCallback(
      [this]() { return PrepareTransferCommandBuffer(); });
  // NOT gated on S7TagsOff: these ids drive the subclass's BeginSubmission
  // advance, LRU gating, and M4 faulted-submission matching — equality
  // bookkeeping that needs real monotonic ids in both sync and async
  // modes. (Retirement tags still flow through the legacy adapter's
  // S7-governed RetireSubmission.)
  shared_texture_cache_->SetSubmissionProvider(
      [this]() { return CurrentEncodingSubmission(); });
  shared_texture_cache_->SetSubmissionFaultQuery([this](uint64_t submission) {
    return faulted_submissions_.count(submission) != 0;
  });
  if (!shared_texture_cache_->Initialize()) {
    XELOGE(
        "MetalCommandProcessor: shared texture cache init failed; route "
        "disabled, legacy cache serves everything");
    shared_texture_cache_.reset();
  }
  return true;
}

void MetalCommandProcessor::ShutdownDrawResources() {
  // The shared-route subclass retires slots/textures through texture_cache_;
  // destroy it first.
  shared_texture_cache_.reset();
  texture_cache_.reset();  // references view_bindless_heap_, free first.
  auto release = [](void*& p) {
    if (p) {
      id obj = (__bridge_transfer id)p;
      (void)obj;
      p = nullptr;
    }
  };
  release(view_bindless_heap_);
  release(sampler_bindless_heap_);
  release(system_view_tables_);
  release(null_buffer_);
  release(null_texture_);
  release(null_sampler_);
  for (auto& pair : depth_stencil_states_) {
    release(pair.second);
  }
  depth_stencil_states_.clear();
  for (void*& state : sampler_states_) {
    release(state);
  }
  sampler_states_.clear();
  sampler_slot_cache_.clear();
  next_sampler_slot_ = 1;
  upload_arena_.reset();
}

void MetalCommandProcessor::ShutdownContext() {
  if (frame_open_) {
    EndFrameEncoderAndWait();
  }
  PumpCompletedActions();
  // Drops any resolve still queued and releases the visibility buffer; the
  // reports they belonged to are abandoned by the shared reset path.
  ShutdownZPDQueryResources();
  ShutdownDrawResources();
  // PrimitiveProcessor: destroy before shared_memory_ (it holds a reference).
  primitive_processor_.reset();
  shared_memory_.reset();
  render_target_cache_.reset();
  pipeline_cache_.reset();
  CommandProcessor::ShutdownContext();
  // Drain the final frame's pool (pushed at the last IssueSwap; the teardown
  // above autoreleased into it). Runs on the same GPU thread as IssueSwap.
  if (frame_autorelease_pool_) {
    objc_autoreleasePoolPop(frame_autorelease_pool_);
    frame_autorelease_pool_ = nullptr;
  }
}

// Depth buffer is GATED OFF for now. When enabled it clears depth to 1.0 (far)
// assuming standard 0..1 depth, but many Xbox 360 titles use reverse-z (clear 0,
// "greater" test), so the fixed clear discards everything and 2D/menus go black.
// Re-enable only once the guest depth-clear value + z-direction are emulated
// (read from the guest's depth clear, not assumed). Keeps the pipeline's depth
// attachment format and the render pass consistent (both on or both off).
static constexpr bool kDepthEnabled = true;

void* MetalCommandProcessor::BeginFrameEncoder() {
  // Attach whatever the RT cache accumulated for the current draw (Update()
  // ran in IssueDraw). The attachment key detects guest pass changes.
  if (!memexport_pass_ &&
      !render_target_cache_->PrepareCurrentRasterViews()) {
    return nullptr;
  }
  uint64_t want_key = render_target_cache_->current_attachment_key();
  if (dummy_depth_pass_) {
    // Scratch-depth passes are distinct from same-color passes without one:
    // their pipelines carry a depth pixel format the plain pass lacks.
    want_key ^= 0x9E3779B97F4A7C15ull;
  }
  if (memexport_pass_) {
    // Memexport draws run in their own attachment-less pass; never share an
    // encoder with attachment passes.
    want_key = 0x4D454D4558504F52ull;  // 'MEMEXPOR'
  }
  bool reuse_encoder = frame_open_ && encoder_attachment_key_ == want_key;
  if (reuse_encoder && ZPDSegmentPending()) {
    // A report can open, and a closed segment can become pending again, while
    // this encoder is already current -- a path that would otherwise never
    // reach the segment resume below, because a reused encoder returns early.
    // Mafia II's "dead" frame is exactly this shape: its whole report lifetime
    // falls inside one unchanged attachment key.
    if (!zpd_visibility_pool_.is_initialized()) {
      EnsureZPDQueryResources();
    }
    if (encoder_has_zpd_visibility_) {
      OpenQuerySegment(false);
    } else if (zpd_visibility_pool_.is_initialized()) {
      // The encoder predates the pool. Metal only accepts a visibility buffer
      // on the descriptor, so the encoder has to be rebuilt to carry one.
      // Bounded: every encoder created afterwards carries it, so this happens
      // once per pool lifetime rather than per draw.
      reuse_encoder = false;
    }
    // Pool unavailable: keep the encoder rather than rebuilding it forever.
  }
  if (frame_open_) {
    if (reuse_encoder) {
      return frame_render_encoder_;
    }
    EndRenderEncoder();
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue() || !render_target_cache_) {
    return nullptr;
  }
  uint32_t raster_sample_count =
      memexport_pass_ ? 1u : render_target_cache_->current_raster_sample_count();
  void* color_resolves[xenos::kMaxColorRenderTargets] = {};
  void* color_targets[xenos::kMaxColorRenderTargets] = {};
  void* primary_color_target = nullptr;
  bool any_color_target = false;
  if (!memexport_pass_) {
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      color_resolves[i] =
          render_target_cache_->current_color_texture(i, nullptr);
      color_targets[i] =
          render_target_cache_->current_color_raster_texture(i, nullptr);
      if (color_targets[i]) {
        any_color_target = true;
        if (!primary_color_target) {
          primary_color_target = color_targets[i];
        }
      }
    }
  }
  void* depth_resolve =
      (kDepthEnabled && !memexport_pass_)
          ? render_target_cache_->current_depth_texture()
          : nullptr;
  void* depth = (kDepthEnabled && !memexport_pass_)
                    ? render_target_cache_->current_depth_raster_texture()
                    : nullptr;
  if (!any_color_target && !depth && !memexport_pass_) {
    // Nothing to attach (the caller should have skipped such draws).
    return nullptr;
  }
  // The command buffer lives until the next commit boundary; the upload ring
  // reclaims per-submission from the completion pump (S8) — no reset here.
  id<MTLCommandBuffer> command_buffer;
  bool new_frame = frame_command_buffer_ == nullptr;
  if (new_frame) {
    if (S7ArenaResetOff() && upload_arena_) {
      upload_arena_->Reset();
    }
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)provider->command_queue();
    command_buffer = MakeAttributedCommandBuffer(queue);
    frame_command_buffer_ = (__bridge_retained void*)command_buffer;
  } else {
    command_buffer = (__bridge id<MTLCommandBuffer>)frame_command_buffer_;
  }
  // Per-target clear policy: each logical target clears on its FIRST use in a
  // frame and loads afterwards. Same-target multi-pass accumulation (JC2's HUD
  // drawn over the composited front buffer after its resolve) keeps working,
  // while distinct targets no longer bleed into each other through a shared
  // texture (each has its own now).
  MTLRenderPassDescriptor* pass =
      [MTLRenderPassDescriptor renderPassDescriptor];
  if (memexport_pass_) {
    // Memexport-only draws: exports go to the shared-memory UAV; the raster
    // output is irrelevant but Metal requires a non-void VS return only when
    // rasterizing, and MSC shaders always return stage-out — so rasterize
    // into a small throwaway scratch depth (DontCare load/store), reusing
    // the dummy-depth texture pool.
    uint64_t dk = (64ull << 20) | 64ull;
    id<MTLTexture> scratch = nil;
    auto dit = dummy_depth_textures_.find(dk);
    if (dit != dummy_depth_textures_.end()) {
      scratch = (__bridge id<MTLTexture>)dit->second;
    } else {
      MTLTextureDescriptor* dd = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:(MTLPixelFormat)
                                                 MetalRenderTargetCache::
                                                     kDepthFormat
                                       width:64
                                      height:64
                                   mipmapped:NO];
      dd.usage = MTLTextureUsageRenderTarget;
      dd.storageMode = MTLStorageModePrivate;
      id<MTLDevice> dev = (__bridge id<MTLDevice>)
          static_cast<xe::ui::metal::MetalProvider*>(
              graphics_system_->provider())->device();
      scratch = [dev newTextureWithDescriptor:dd];
      scratch.label = @"cp:memexport-scratch-depth";
      dummy_depth_textures_.emplace(dk, (__bridge_retained void*)scratch);
    }
    if (!scratch) {
      return nullptr;
    }
    pass.depthAttachment.texture = scratch;
    pass.depthAttachment.loadAction = MTLLoadActionDontCare;
    pass.depthAttachment.storeAction = MTLStoreActionDontCare;
    pass.stencilAttachment.texture = scratch;
    pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
    pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
  }
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (!color_targets[i]) {
      continue;
    }
    // Always LOAD: textures are cleared at creation; content comes from
    // ownership transfers / persistence, clears from guest resolve clears.
    pass.colorAttachments[i].texture =
        (__bridge id<MTLTexture>)color_targets[i];
    pass.colorAttachments[i].loadAction = MTLLoadActionLoad;
    if (raster_sample_count > 1u) {
      pass.colorAttachments[i].resolveTexture =
          (__bridge id<MTLTexture>)color_resolves[i];
      pass.colorAttachments[i].storeAction =
          MTLStoreActionStoreAndMultisampleResolve;
    } else {
      pass.colorAttachments[i].storeAction = MTLStoreActionStore;
    }
  }
  // Depth attachment: cleared to the GUEST's depth-clear value (RB_DEPTH_CLEAR,
  // 24-bit depth in the high bits), not a fixed 1.0. Reverse-z titles (JC2
  // menus) clear to 0 and test "greater"; a hardcoded 1.0 made every fragment
  // fail z>=1.0 -> black. Reading the guest value serves both directions.
  if (depth) {
    pass.depthAttachment.texture = (__bridge id<MTLTexture>)depth;
    pass.depthAttachment.loadAction = MTLLoadActionLoad;
    if (raster_sample_count > 1u) {
      pass.depthAttachment.resolveTexture =
          (__bridge id<MTLTexture>)depth_resolve;
      pass.depthAttachment.depthResolveFilter =
          MTLMultisampleDepthResolveFilterSample0;
      pass.depthAttachment.storeAction =
          MTLStoreActionStoreAndMultisampleResolve;
    } else {
      pass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    pass.stencilAttachment.texture = (__bridge id<MTLTexture>)depth;
    pass.stencilAttachment.loadAction = MTLLoadActionLoad;
    if (raster_sample_count > 1u) {
      pass.stencilAttachment.resolveTexture =
          (__bridge id<MTLTexture>)depth_resolve;
      pass.stencilAttachment.stencilResolveFilter =
          MTLMultisampleStencilResolveFilterSample0;
      pass.stencilAttachment.storeAction =
          MTLStoreActionStoreAndMultisampleResolve;
    } else {
      pass.stencilAttachment.storeAction = MTLStoreActionStore;
    }
  } else if (primary_color_target && dummy_depth_pass_) {
    // The draw's PS exports depth with no depth target bound — attach a
    // throwaway scratch depth so the pipeline is valid (see dummy_depth in
    // IssueDraw). Never loaded, never stored; content is irrelevant.
    id<MTLTexture> rt_tex =
        (__bridge id<MTLTexture>)primary_color_target;
    uint64_t dk = (1ull << 63) | (uint64_t(rt_tex.width) << 32) |
                  (uint64_t(rt_tex.height) << 8) | raster_sample_count;
    id<MTLTexture> dummy_tex = nil;
    auto dit = dummy_depth_textures_.find(dk);
    if (dit != dummy_depth_textures_.end()) {
      dummy_tex = (__bridge id<MTLTexture>)dit->second;
    } else {
      MTLTextureDescriptor* dd = [[MTLTextureDescriptor alloc] init];
      dd.textureType = raster_sample_count > 1u
                           ? MTLTextureType2DMultisample
                           : MTLTextureType2D;
      dd.pixelFormat =
          (MTLPixelFormat)MetalRenderTargetCache::kDepthFormat;
      dd.width = rt_tex.width;
      dd.height = rt_tex.height;
      dd.sampleCount = raster_sample_count;
      dd.usage = MTLTextureUsageRenderTarget;
      dd.storageMode = MTLStorageModePrivate;
      dummy_tex = [rt_tex.device newTextureWithDescriptor:dd];
      dummy_tex.label = @"cp:dummy-depth";
      dummy_depth_textures_.emplace(dk, (__bridge_retained void*)dummy_tex);
    }
    if (dummy_tex) {
      pass.depthAttachment.texture = dummy_tex;
      pass.depthAttachment.loadAction = MTLLoadActionDontCare;
      pass.depthAttachment.storeAction = MTLStoreActionDontCare;
      pass.stencilAttachment.texture = dummy_tex;
      pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
      pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
    }
  }
  // ZPD occlusion queries: Metal takes the visibility buffer on the DESCRIPTOR
  // and offers no way to attach one to a live encoder, so it must be decided
  // here. The pool is created lazily on the first encoder built while a guest
  // report is open -- a title that never issues a query allocates nothing and
  // every encoder below is byte-for-byte what it was before this change.
  if (ZPDSegmentPending() && !zpd_visibility_pool_.is_initialized()) {
    EnsureZPDQueryResources();
  }
  id<MTLBuffer> zpd_visibility_buffer =
      (__bridge id<MTLBuffer>)zpd_visibility_pool_.visibility_buffer();
  if (zpd_visibility_buffer) {
    pass.visibilityResultBuffer = zpd_visibility_buffer;
  }

  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:pass];
  // Fault attribution: names this encoder in per-encoder execution status.
  encoder.label = [NSString
      stringWithFormat:@"draws@%u rt=%016llX", draw_count_,
                       (unsigned long long)want_key];
  [encoder setCullMode:MTLCullModeNone];
  // Residency sweep: make EVERY texture any live bindless slot can reach
  // resident on this encoder in one batched call. The per-draw useResource
  // loop covers only that draw's own fetches; a shader indexing any OTHER
  // live slot (stale descriptor index, prior-generation snapshot slot) reads
  // a non-resident texture -> GPU page fault (the attributed 'draws@N'
  // faults). Textures created after this sweep are covered by the per-draw
  // loop as before.
  if (texture_cache_) {
    resident_sweep_scratch_.clear();
    texture_cache_->AppendAllTextures(resident_sweep_scratch_);
    if (shared_texture_cache_) {
      shared_texture_cache_->AppendAllTextures(resident_sweep_scratch_);
    }
    if (!resident_sweep_scratch_.empty()) {
      MTLRenderStages sweep_stages =
          MTLRenderStageVertex | MTLRenderStageFragment;
      if (mesh_shader_supported_) {
        sweep_stages |= MTLRenderStageObject | MTLRenderStageMesh;
      }
      // void* -> id<MTLTexture> in place (toll-free bridge array).
      [encoder useResources:reinterpret_cast<id<MTLResource> const*>(
                                resident_sweep_scratch_.data())
                      count:resident_sweep_scratch_.size()
                      usage:MTLResourceUsageRead
                     stages:sweep_stages];
    }
  }
  frame_render_encoder_ = (__bridge_retained void*)encoder;
  frame_open_ = true;
  encoder_attachment_key_ = want_key;
  encoder_has_zpd_visibility_ = zpd_visibility_buffer != nil;
  // A new encoder is the next valid opportunity for a report whose segment is
  // waiting to open -- either because the guest issued BEGIN with no eligible
  // encoder, or because the previous segment was closed at the last encoder
  // boundary. This is the Metal analogue of Vulkan reopening at render-pass
  // begin. can_close_submission is false: nothing here may commit the command
  // buffer this encoder was just created on.
  if (encoder_has_zpd_visibility_ && ZPDSegmentPending()) {
    OpenQuerySegment(false);
  }
  return frame_render_encoder_;
}

// XE_METAL_S7_OFF=1: live-bisect kill switch — restores pre-S7 lifetime
// semantics (retires/writebacks tagged 0 = reclaimable at the next
// completion, upload ring hard-reset at every new command buffer like the
// old bump arena). Only sound combined with XE_METAL_SYNC_WAIT=1.
// Split variants for narrowing: XE_METAL_S7_OFF_ARENA (ring reset only),
// XE_METAL_S7_OFF_TAGS (zero retire tags only).
bool MetalCommandProcessor::S7LifetimeOff() {
  // DEFAULT ON (pre-S7 semantics) until the async lifetime race is found:
  // the S7/S8 fast path (28.5 vs 21 fps live) faults ~1/500 swaps in
  // gameplay — a timing race that hides under shader validation and hasn't
  // reproduced in replay (even 16x ring stress / loop stress). XE_METAL_ASYNC=1
  // opts back into the full async+ring+tags path for the hunt.
  static const bool s_async = std::getenv("XE_METAL_ASYNC") != nullptr;
  static const bool s_off =
      std::getenv("XE_METAL_S7_OFF") != nullptr || !s_async;
  return s_off;
}

bool MetalCommandProcessor::S7ArenaResetOff() {
  static const bool s_off = std::getenv("XE_METAL_S7_OFF_ARENA") != nullptr;
  return s_off || S7LifetimeOff();
}

bool MetalCommandProcessor::S7TagsOff() {
  static const bool s_off = std::getenv("XE_METAL_S7_OFF_TAGS") != nullptr;
  return s_off || S7LifetimeOff();
}

void MetalCommandProcessor::PublishCompletedSubmission(uint64_t submission) {
  uint64_t prev = completed_submission_.load(std::memory_order_relaxed);
  while (prev < submission &&
         !completed_submission_.compare_exchange_weak(
             prev, submission, std::memory_order_release,
             std::memory_order_relaxed)) {
  }
  // Wake WaitForSubmission (S8). The empty lock pairs with the waiter's
  // predicate re-check so a wake between check and wait isn't lost.
  { std::lock_guard<std::mutex> lock(completion_mutex_); }
  completion_cv_.notify_all();
}

bool MetalCommandProcessor::WaitForSubmission(uint64_t submission,
                                              uint32_t timeout_ms) {
  if (completed_submission_.load(std::memory_order_acquire) >= submission) {
    return true;
  }
  std::unique_lock<std::mutex> lock(completion_mutex_);
  bool ok = completion_cv_.wait_for(
      lock, std::chrono::milliseconds(timeout_ms), [&] {
        return completed_submission_.load(std::memory_order_acquire) >=
               submission;
      });
  if (!ok) {
    XELOGE(
        "MetalCommandProcessor: TIMEOUT waiting for submission {} "
        "(completed {}) — GPU wedged?",
        submission, completed_submission_.load(std::memory_order_acquire));
  }
  return ok;
}

void MetalCommandProcessor::DrainFaultedCommandBuffers() {
  std::vector<void*> faulted;
  std::vector<uint64_t> faulted_ids;
  {
    std::lock_guard<std::mutex> lock(fault_mutex_);
    faulted.swap(faulted_command_buffers_);
    faulted_ids.swap(faulted_submission_ids_);
  }
  for (uint64_t id : faulted_ids) {
    faulted_submissions_.insert(id);
    // Bounded: fault-attribution evidence, not a growing ledger.
    if (faulted_submissions_.size() > 256) {
      faulted_submissions_.clear();
      faulted_submissions_.insert(id);
    }
  }
  // C1T-R M4 (mandatory): eagerly invalidate routed textures whose load rode
  // a lost submission — through the shared-memory authority, independent of
  // the bounded fault set above.
  if (shared_texture_cache_ && !faulted_ids.empty()) {
    shared_texture_cache_->OnSubmissionsFaulted(faulted_ids);
  }
  for (void* cb_ptr : faulted) {
    id<MTLCommandBuffer> command_buffer =
        (__bridge_transfer id<MTLCommandBuffer>)cb_ptr;
    static uint32_t s_cb_errors = 0;
    ++s_cb_errors;
    if (s_cb_errors <= 32 || (s_cb_errors & 255) == 0) {
      LogCommandBufferFault(command_buffer, "frame commit", swap_count_);
      if (s_cb_errors > 32) {
        XELOGE("Metal GPU FAULT: {} total this session", s_cb_errors);
      }
    }
    // AUTO-RECOVERY: error code 4 (access revoked) LATCHES — recreate the
    // queue (GPU thread; the provider/presenter aren't touched from the
    // completion-handler thread). Throttled against fault storms.
    if (command_buffer.error && command_buffer.error.code == 4) {
      static uint32_t s_last_recovery_swap = 0;
      if (swap_count_ - s_last_recovery_swap > 120 ||
          s_last_recovery_swap == 0) {
        s_last_recovery_swap = swap_count_;
        auto* provider = static_cast<xe::ui::metal::MetalProvider*>(
            graphics_system_->provider());
        if (provider && provider->RecreateQueue()) {
          if (auto* presenter = static_cast<xe::ui::metal::MetalPresenter*>(
                  graphics_system_->presenter())) {
            presenter->SetCommandQueue(provider->command_queue());
          }
          XELOGE(
              "Metal GPU FAULT: queue revoked -> RECREATED (swap#{}); "
              "display should resume",
              swap_count_);
        }
      }
    }
  }
}

bool MetalCommandProcessor::ArenaAllocate(uint32_t size, uint32_t alignment,
                                          MetalUploadAllocation* out) {
  if (!upload_arena_) {
    return false;
  }
  if (upload_arena_->Allocate(size, alignment, out)) {
    return true;
  }
  // Ring full: space is owned by in-flight submissions. Wait for the oldest
  // one, reclaim, retry once. (Waiting for ALL would serialize; the oldest
  // frees the ring tail.)
  uint64_t completed = completed_submission_.load(std::memory_order_acquire);
  if (completed < submission_counter_) {
    if (WaitForSubmission(completed + 1)) {
      upload_arena_->Reclaim(
          completed_submission_.load(std::memory_order_acquire));
      PumpCompletedActions();
      return upload_arena_->Allocate(size, alignment, out);
    }
  }
  return false;
}

void MetalCommandProcessor::PumpCompletedActions() {
  // GPU-thread only: completion_actions_ isn't synchronized; only the
  // published counter crosses threads.
  DrainFaultedCommandBuffers();
  uint64_t completed = completed_submission_.load(std::memory_order_acquire);
  while (!completion_actions_.empty() &&
         completion_actions_.front().first <= completed) {
    auto action = std::move(completion_actions_.front().second);
    completion_actions_.pop_front();
    action();
  }
  // Query slots are submission-gated the same way: a visibility counter is
  // only readable once the command buffer that wrote it has completed.
  PumpQueryResolves();
}

void MetalCommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  // Latch only. The request arrives through GraphicsSystem::ClearCaches ->
  // CallInThread (F5 / GPU menu) or from a trace player rewinding to an
  // earlier command, so it can land mid-encoder with submissions in flight.
  // Releasing here would free resources the GPU is still reading. D3D12 makes
  // the same choice for the same reason.
  cache_clear_requested_ = true;
}

void MetalCommandProcessor::ExecuteDeferredCacheClear(bool publication_failed) {
  // No-request hot path: a single predictable branch. No commit, no wait, no
  // allocation for a title that never requests a clear.
  if (!cache_clear_requested_) {
    return;
  }
  // At most three passes -- quiesce, publish, execute. Each pass satisfies at
  // most one outstanding precondition; anything still unsatisfied leaves the
  // request LATCHED for the next boundary rather than dropping it or
  // releasing early.
  for (int pass = 0; pass < 3; ++pass) {
    metal::CacheClearAdmissionInputs in{};
    in.request_latched = cache_clear_requested_;
    in.encoder_open = frame_open_;
    in.has_unsubmitted_work = frame_command_buffer_ != nullptr;
    in.last_submitted_submission = submission_counter_;
    in.completed_submission =
        completed_submission_.load(std::memory_order_acquire);
    in.has_pending_producers =
        !pending_writebacks_.empty() || !pending_memexports_.empty();
    in.publication_failed = publication_failed;
    in.can_end_submission = true;
    in.can_publish_pending = true;
    switch (metal::DecideCacheClearAdmission(in)) {
      case metal::CacheClearAdmission::kEndSubmissionFirst:
        // Steps 1-3: end the encoder, commit, await every submission, and pump
        // completion actions so submission-gated retirement finishes before
        // anything is released. At the swap boundary the frame's own
        // EndFrameEncoderAndWait has already done this, so this is a no-op
        // there; it is what makes the routine safe from any other boundary.
        EndFrameEncoderAndWait();
        continue;
      case metal::CacheClearAdmission::kPublishPendingFirst: {
        // Step 4: completing a submission makes a drain POSSIBLE without
        // performing it. A queued memexport or render-target writeback is
        // guest-visible data that has not yet reached RAM -- authoritative
        // state under the cross-writer authority contract, not derived cache
        // content -- so it must land before anything is released. Same
        // full-domain publication the swap performs, through the same
        // chronological authority executor.
        metal::GuestRangeSet publication_domain;
        publication_domain.Add(metal::GuestRange::FromBaseLength(
            0, static_cast<uint32_t>(metal::GuestRange::kGuestPhysicalSize)));
        if (!ResolveRamConsumerDependencies(metal::ConsumerKind::kHostCpuRead,
                                            publication_domain)) {
          // Fail closed: keep the pending set intact and the request latched.
          publication_failed = true;
        }
        continue;
      }
      case metal::CacheClearAdmission::kExecute:
        // Step 5: every precondition holds -- encoder closed, every submission
        // complete, completion actions pumped, and all GPU-produced guest data
        // published to RAM. Only now may anything be released.
        if (!ReleaseCachesForClear()) {
          // A precondition failed and nothing was released, so the request is
          // NOT satisfied: leave it latched and do not count it.
          return;
        }
        // A GUEST logical report open across this clear (BEGIN seen, END not
        // yet) is the one ZPD state a clear could strand: the pool and its
        // in-flight resolves are retained, but the report must still retire
        // afterwards. Record only handles actually observed in that state --
        // never a merely in-flight resolve, which the boundary has already
        // drained -- so PumpQueryResolves can report the matching retirement.
        // Ordinary titles never populate this set.
        if (zpd_active_segment_.logical_active &&
            zpd_active_segment_.report_handle != kInvalidReportHandle) {
          zpd_handles_spanning_clear_.insert(zpd_active_segment_.report_handle);
          XELOGI("MetalZPD: report handle={} spans cache clear (swap #{})",
                 zpd_active_segment_.report_handle, swap_count_);
        }
        // Step 6: the request is satisfied. Consume the latch and publish the
        // completed clear (release, so the release work above is visible to
        // whichever thread reads the count).
        cache_clear_requested_ = false;
        cache_clears_executed_.fetch_add(1, std::memory_order_release);
        return;
      case metal::CacheClearAdmission::kDefer:
      case metal::CacheClearAdmission::kNoRequest:
        return;
    }
  }
}

bool MetalCommandProcessor::ReleaseCachesForClear() {
  // Step 5 of the deferred cache-clear ordering. Reached only through
  // ExecuteDeferredCacheClear's kExecute decision, which is what guarantees
  // the device is idle and every GPU-produced guest write has already reached
  // RAM. Every release below is therefore immediate rather than retired.
  //
  // DURABLE PIPELINE STORAGE IS DELIBERATELY NOT TOUCHED HERE (task 3.3f of
  // implement-metal-shader-pipeline-lifetime). The promoted cache-clearing
  // decision already retains pipelines and MSC-translated shaders by design and
  // records that shader/pipeline lifetime is not covered by it, so the on-disk
  // store follows the same rule as its in-memory counterparts. A clear releases
  // DERIVED state that can be rebuilt from guest memory; the durable store is
  // not derived from guest memory at all, and erasing it would turn a
  // cache-control request into a permanent cold start for that title — every
  // stall the census measured, paid again, with no guest-visible benefit.
  // Deleting it is a separate, separately gated behavior if it is ever wanted.
  //
  // ALL preconditions are checked FIRST, before a single object is released,
  // so a failure is reported with everything still intact rather than half way
  // through a partial release. Both of these describe submission-gated state
  // that step 3's completion pumping must already have drained at a boundary
  // where every submission has completed; a survivor of either would name
  // resources this function is about to free.
  uint32_t queued_actions = uint32_t(completion_actions_.size());
  uint32_t consumed_ranges =
      shared_memory_ ? shared_memory_->consumed_range_count() : 0;
  if (queued_actions || consumed_ranges) {
    XELOGE(
        "MetalCommandProcessor: cache clear aborted -- {} completion action(s) "
        "and {} consumed range(s) still held at a quiesced boundary; "
        "submission-gated retirement did not drain",
        queued_actions, consumed_ranges);
    assert_always();
    return false;  // nothing released; the request stays latched.
  }

  // Resource-return evidence (task 4.5). Sampled only on the clear path, so a
  // title that never requests one pays nothing. Footprint is read through the
  // shared sampler both here and after the release, so the pair is measured
  // the same way -- see the log below for what it can and cannot show.
  struct ClearResourceSample {
    uint32_t textures = 0;
    uint32_t bridges = 0;
    uint32_t snapshots = 0;
    uint32_t slots_used = 0;
    uint32_t slots_free = 0;
    uint32_t slots_stale = 0;
    double footprint_mib = 0.0;
  };
  auto sample_resources = [this]() {
    ClearResourceSample s;
    if (texture_cache_) {
      s.textures = texture_cache_->cached_texture_count();
      s.bridges = texture_cache_->resolved_bridge_count();
      s.slots_used = texture_cache_->slots_used();
      s.slots_free = texture_cache_->slots_free();
      s.slots_stale = texture_cache_->slots_stale();
    }
    if (render_target_cache_) {
      s.snapshots = render_target_cache_->resolve_snapshot_count();
    }
    s.footprint_mib = ProcessFootprintMiB();
    return s;
  };
  const ClearResourceSample before = sample_resources();

  // The order is load-bearing, not cosmetic: every step that RETIRES into the
  // legacy cache's pools must run before the step that DRAINS them.

  // 1. The shared TextureCache subclass FIRST: it retires its binding slots
  //    and views into the LEGACY cache's pools.
  if (shared_texture_cache_) {
    shared_texture_cache_->ClearCache();
  }
  // 2. Classify the resolve-snapshot bridges and release the derived ones
  //    (current-generation RAM-backed color: the clear published their
  //    writebacks in step 4, so guest RAM holds THIS generation's content and
  //    their serve path already reads it).
  //    This also retires into the legacy pools, so it precedes the drain.
  //    Bridges that are !generation_ram_backed (this content generation was
  //    never written to guest RAM, so the snapshot is the only copy) or
  //    is_depth (served from the snapshot regardless of RAM) are retained.
  std::vector<std::pair<uint32_t, void*>> released_bridges;
  if (texture_cache_) {
    texture_cache_->ReleaseDerivedResolveBridges(&released_bridges);
  }
  // 3. Release exactly the snapshot textures those bridges named, plus the
  //    command-processor-owned tip textures and per-base resolve metadata for
  //    the same bases. Nothing else's snapshot is touched.
  uint32_t released_snapshots = 0;
  std::vector<uint32_t> released_snapshot_bases;
  if (render_target_cache_ && !released_bridges.empty()) {
    released_snapshots = render_target_cache_->ReleaseResolveSnapshotsForBases(
        released_bridges, &released_snapshot_bases);
  }
  // Per-base state is keyed off what was ACTUALLY released, never off what was
  // requested. A skipped identity mismatch means a newer snapshot generation
  // lives at that base, and its tip texture, RAM colour layout, snapshot
  // generation, and held writeback content belong to that generation -- they
  // must survive whole, exactly like the snapshot itself.
  for (uint32_t base : released_snapshot_bases) {
    auto tip_it = tip_view_textures_.find(base);
    if (tip_it != tip_view_textures_.end()) {
      if (tip_it->second) {
        id<MTLTexture> released =
            (__bridge_transfer id<MTLTexture>)tip_it->second;
        (void)released;
      }
      tip_view_textures_.erase(tip_it);
    }
    ram_color_layouts_.erase(base);
    resolve_snapshot_generation_.erase(base);
    held_writeback_content_.erase(base);
  }
  // 4. The legacy cache DRAIN: cached textures, their descriptors, the slot
  //    pools, and both retire lists -- including everything steps 1 and 2 just
  //    retired into them.
  if (texture_cache_) {
    texture_cache_->ClearCache();
  }
  // 5. Render targets: drops derived state keyed by render-target identity and
  //    drains deferred releases, then the shared clear keeps every target still
  //    owning EDRAM data. Canonical EDRAM is untouched.
  if (render_target_cache_) {
    render_target_cache_->ClearCache();
  }
  // 6. Shared memory. The consumed-range precondition was verified at the top,
  //    before anything was released; re-check here only to catch a release
  //    path above that somehow recorded a new hazard range.
  if (shared_memory_) {
    uint32_t consumed_now = shared_memory_->consumed_range_count();
    if (consumed_now) {
      XELOGE(
          "MetalCommandProcessor: {} consumed range(s) appeared DURING the "
          "cache-clear release -- a release path is recording hazard state",
          consumed_now);
      assert_always();
    }
    shared_memory_->ClearCache();
  }
  // 7. Metal-owned transient pools with no shared equivalent. The arena reset
  //    is a hard ring reset -- documented as safe only when nothing is in
  //    flight, which is exactly what this boundary proves. It is unconditional:
  //    the S7 arena-reset bisect switch governs per-frame reclamation policy,
  //    not whether a cache clear releases the pool.
  if (upload_arena_) {
    upload_arena_->Reset();
  }
  for (auto& pair : dummy_depth_textures_) {
    if (pair.second) {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)pair.second;
      (void)released;
    }
  }
  dummy_depth_textures_.clear();
  frame_cleared_targets_.clear();  // per-frame clear bookkeeping.

  // NOT released, and each for a stated reason:
  //   - pipeline_cache_ (pipeline states + MSC-translated shaders): the D3D12
  //     contract retains what live pipelines reference; destroying them would
  //     make this a translation-layer reset and risk post-clear draw skips.
  //   - sampler_states_/sampler_slot_cache_ and depth_stencil_states_:
  //     immutable state objects of the same class as pipelines.
  //   - the ZPD visibility pool, zpd_resolves_in_flight_, and the active
  //     segment: D3D12's clear does not touch its query pool either, and a
  //     logical report can span the request.
  //   - canonical EDRAM and EDRAM-owning render targets: authoritative guest
  //     state, retained by the shared clear's own rule.

  // Retire-list consistency: nothing may remain queued against a submission
  // whose resources were just released. Completion actions are pumped by step
  // 3 at a boundary where every submission has completed, so this must already
  // be empty -- verify rather than assume, since a survivor would run against
  // caches that no longer hold what it expects.
  // Re-check: nothing above may have enqueued new submission-gated work.
  if (!completion_actions_.empty()) {
    XELOGE(
        "MetalCommandProcessor: cache clear ENQUEUED {} completion action(s) "
        "while releasing -- a release path is retiring against a submission "
        "that will never be revisited",
        uint32_t(completion_actions_.size()));
    assert_always();
  }
  // Every retained bridge must satisfy the documented retention invariant.
  if (texture_cache_ && !texture_cache_->retained_bridges_are_authoritative()) {
    XELOGE(
        "MetalCommandProcessor: cache clear retained a current-generation "
        "RAM-backed color resolve bridge (generation_ram_backed) -- derived "
        "state was classified as authoritative");
    assert_always();
  }
  // One machine-parsable before->after line per executed clear. The ENUMERABLE
  // cache counters are the evidence that derived state was actually returned:
  // they are exact, owned by this backend, and bounded by construction.
  //
  // footprint_mib is reported for completeness and is deliberately NOT an
  // acceptance criterion. Releasing a Metal object returns it to the driver's
  // and the allocator's pools, which need not shrink the process's physical
  // footprint at that instant or ever; a flat or rising number here is not
  // evidence that a cache retained anything, and a falling one would not prove
  // it released everything. What the counters prove is cache-level return to
  // baseline; what footprint cannot prove is either direction of that claim.
  const ClearResourceSample after = sample_resources();
  XELOGI(
      "MetalCacheClearStats swap={} textures={}->{} "
      "bridges={}->{}(released={},retained={}) "
      "snapshots={}->{}(released={}) slots_used={}->{} slots_free={}->{} "
      "slots_stale={}->{} slots_capacity={} footprint_mib={:.2f}->{:.2f}",
      swap_count_, before.textures, after.textures, before.bridges,
      after.bridges, uint32_t(released_bridges.size()), after.bridges,
      before.snapshots, after.snapshots, released_snapshots, before.slots_used,
      after.slots_used, before.slots_free, after.slots_free,
      before.slots_stale, after.slots_stale,
      texture_cache_ ? texture_cache_->slots_capacity() : 0,
      before.footprint_mib, after.footprint_mib);
  return true;
}

void MetalCommandProcessor::EnsureZPDQueryResources() {
  if (GetZPDMode() == ZPDMode::kFake) {
    return;
  }
  auto* provider = static_cast<xe::ui::metal::MetalProvider*>(
      graphics_system_->provider());
  if (!provider || !provider->device()) {
    return;
  }
  zpd_visibility_pool_.EnsureInitialized(provider->device(),
                                         kZPDQueryPoolCapacity);
}

void MetalCommandProcessor::ShutdownZPDQueryResources() {
  zpd_resolves_in_flight_.clear();
  zpd_handles_spanning_clear_.clear();
  zpd_query_open_ = false;
  encoder_has_zpd_visibility_ = false;
  zpd_visibility_pool_.Shutdown();
}

bool MetalCommandProcessor::IsZPDQueryPoolReady() const {
  return zpd_visibility_pool_.is_initialized();
}

bool MetalCommandProcessor::ZPDSegmentPending() const {
  return GetZPDMode() != ZPDMode::kFake && !zpd_force_fake_fallback_ &&
         zpd_active_segment_.logical_active &&
         zpd_active_segment_.segment_pending_begin;
}

bool MetalCommandProcessor::CanOpenZPDQuery() const {
  // Only "a render encoder exists". Deliberately NOT "and it carries the
  // visibility buffer": the shared OpenQuerySegment tests this BEFORE calling
  // EnsureZPDQueryResources, so requiring the buffer here would stop the pool
  // from ever being created, and would also skip the fake-fallback latch that
  // a failed pool creation is supposed to trigger. Whether this particular
  // encoder can carry a query is decided in OpenZPDQuery, which defers.
  return frame_open_ && frame_render_encoder_ != nullptr;
}

CommandProcessor::QueryOpenResult MetalCommandProcessor::OpenZPDQuery(
    ReportHandle report_handle, bool can_close_submission) {
  // Drain first, so the admission decision sees slots freed by already
  // completed submissions rather than declaring a false exhaustion.
  if (!zpd_visibility_pool_.has_free_indices()) {
    PumpQueryResolves();
  }

  ZPDMode mode = GetZPDMode();
  ZPDQueryAdmissionInputs admission;
  admission.pool_ready = zpd_visibility_pool_.is_initialized();
  admission.query_already_open = zpd_query_open_;
  admission.encoder_can_carry_query =
      CanOpenZPDQuery() && encoder_has_zpd_visibility_;
  admission.has_free_slot = zpd_visibility_pool_.has_free_indices();
  admission.fast_mode = mode == ZPDMode::kFast || mode == ZPDMode::kFastAlt;
  admission.has_pending_resolve = !zpd_resolves_in_flight_.empty();
  admission.oldest_pending_submission =
      admission.has_pending_resolve ? zpd_resolves_in_flight_.front().submission
                                    : 0;
  admission.current_encoding_submission = CurrentEncodingSubmission();
  admission.can_close_submission = can_close_submission;

  switch (DecideZPDQueryAdmission(admission)) {
    case ZPDQueryAdmission::kAcquire:
      break;
    case ZPDQueryAdmission::kDefer:
      return QueryOpenResult::kDeferred;
    case ZPDQueryAdmission::kDeferAfterClosingSubmission:
      EndFrameEncoder();
      return QueryOpenResult::kDeferred;
    case ZPDQueryAdmission::kPoolExhausted:
      return QueryOpenResult::kPoolExhausted;
    case ZPDQueryAdmission::kFailed:
      return QueryOpenResult::kFailed;
    case ZPDQueryAdmission::kWaitForOldestResolve:
      if (!WaitForSubmission(admission.oldest_pending_submission)) {
        return QueryOpenResult::kFailed;
      }
      PumpQueryResolves();
      if (!zpd_visibility_pool_.has_free_indices()) {
        return QueryOpenResult::kPoolExhausted;
      }
      break;
  }

  uint32_t index = 0;
  uint32_t generation = 0;
  size_t offset = 0;
  if (!zpd_visibility_pool_.Acquire(index, generation, offset)) {
    return QueryOpenResult::kFailed;
  }
  id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)frame_render_encoder_;
  [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                            offset:offset];
  zpd_active_index_ = index;
  zpd_active_generation_ = generation;
  zpd_query_open_ = true;
  return QueryOpenResult::kOpened;
}

bool MetalCommandProcessor::RetireActiveZPDQuery(ReportHandle report_handle,
                                                 uint64_t* out_submission) {
  if (!zpd_query_open_) {
    return false;
  }
  // If the encoder already ended, counting stopped with it and there is
  // nothing to disable -- but the slot still has a pending GPU write, so it
  // must be queued either way.
  if (frame_open_ && frame_render_encoder_ && encoder_has_zpd_visibility_) {
    id<MTLRenderCommandEncoder> encoder =
        (__bridge id<MTLRenderCommandEncoder>)frame_render_encoder_;
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
  }

  MetalZPDResolve resolve;
  // The command buffer being encoded now is the one that will write this slot.
  resolve.submission = CurrentEncodingSubmission();
  resolve.index = zpd_active_index_;
  resolve.generation = zpd_active_generation_;
  resolve.report_handle = report_handle;
  zpd_resolves_in_flight_.push_back(resolve);
  if (out_submission) {
    *out_submission = resolve.submission;
  }
  zpd_query_open_ = false;
  return true;
}

bool MetalCommandProcessor::CloseZPDQuery(ReportHandle report_handle,
                                          uint64_t& out_submission) {
  return RetireActiveZPDQuery(report_handle, &out_submission);
}

bool MetalCommandProcessor::DiscardZPDQuery() {
  // kInvalidReportHandle: the slot is reclaimed on completion but its count is
  // attributed to nothing.
  return RetireActiveZPDQuery(kInvalidReportHandle, nullptr);
}

void MetalCommandProcessor::PumpQueryResolves() {
  if (!zpd_visibility_pool_.is_initialized()) {
    return;
  }
  uint64_t completed = completed_submission_.load(std::memory_order_acquire);
  while (!zpd_resolves_in_flight_.empty() &&
         zpd_resolves_in_flight_.front().submission <= completed) {
    MetalZPDResolve resolve = zpd_resolves_in_flight_.front();
    zpd_resolves_in_flight_.pop_front();
    if (!zpd_visibility_pool_.IsGenerationCurrent(resolve.index,
                                                  resolve.generation)) {
      // The slot was recycled out from under this resolve; attributing it now
      // would corrupt whichever report owns the slot.
      continue;
    }
    // A faulted submission still publishes as completed while its GPU work
    // never ran. The slot was zeroed at acquire, so such a report reads 0
    // rather than stale content.
    uint64_t raw_samples = zpd_visibility_pool_.Read(resolve.index);
    zpd_visibility_pool_.Release(resolve.index, resolve.generation);
    if (resolve.report_handle != kInvalidReportHandle) {
      XELOGD("MetalZPD: resolved handle={} slot={} submission={} samples={}",
             resolve.report_handle, resolve.index, resolve.submission,
             raw_samples);
      OnZPDQueryResolved(resolve.report_handle, raw_samples);
      // Cache-clear span observability. The set is empty for every title that
      // never spans a clear, so an ordinary resolve pays one branch.
      if (!zpd_handles_spanning_clear_.empty()) {
        auto spanning = zpd_handles_spanning_clear_.find(resolve.report_handle);
        if (spanning != zpd_handles_spanning_clear_.end() &&
            logical_zpd_reports_.find(resolve.report_handle) ==
                logical_zpd_reports_.end()) {
          // The report that was live across a clear has now completed the
          // shared retirement path (accumulated, committed, and erased) rather
          // than being stranded by the release.
          XELOGI("MetalZPD: clear-spanning report handle={} retired",
                 resolve.report_handle);
          zpd_handles_spanning_clear_.erase(spanning);
        }
      }
    }
  }
}

bool MetalCommandProcessor::AwaitQueryResolve(ReportHandle report_handle,
                                              uint64_t wait_for_submission) {
  if (GetZPDMode() == ZPDMode::kFake) {
    return false;
  }
  PumpQueryResolves();

  auto report = logical_zpd_reports_.find(report_handle);
  if (report == logical_zpd_reports_.end()) {
    return true;
  }
  if (report->second.pending_segments == 0 && report->second.ended) {
    return true;
  }
  if (!wait_for_submission) {
    return false;
  }

  if (wait_for_submission >= CurrentEncodingSubmission()) {
    if (!frame_command_buffer_) {
      return false;
    }
    // The segment is in the command buffer still open; commit it so it can
    // complete. Strict mode's stall backstops bound how often this happens.
    EndFrameEncoder();
  }
  if (wait_for_submission >
          completed_submission_.load(std::memory_order_acquire) &&
      !WaitForSubmission(wait_for_submission)) {
    return false;
  }
  PumpQueryResolves();

  report = logical_zpd_reports_.find(report_handle);
  return report == logical_zpd_reports_.end() ||
         (report->second.pending_segments == 0 && report->second.ended);
}

void MetalCommandProcessor::EndRenderEncoder() {
  if (!frame_open_) {
    return;
  }
  // The encoder is a query-segment boundary (Metal's analogue of a Vulkan
  // render pass): close the active segment while setVisibilityResultMode can
  // still be issued. The logical report survives and reopens on the next
  // visibility-carrying encoder.
  if (zpd_query_open_) {
    CloseQuerySegment();
  }
  id<MTLRenderCommandEncoder> encoder =
      (__bridge_transfer id<MTLRenderCommandEncoder>)frame_render_encoder_;
  frame_render_encoder_ = nullptr;
  frame_open_ = false;
  encoder_has_zpd_visibility_ = false;
  [encoder endEncoding];
}

void MetalCommandProcessor::EndFrameEncoder() {
  EndRenderEncoder();
  if (!frame_command_buffer_) {
    return;
  }
  id<MTLCommandBuffer> command_buffer =
      (__bridge_transfer id<MTLCommandBuffer>)frame_command_buffer_;
  frame_command_buffer_ = nullptr;
  ++frame_commit_count_;
  draws_since_commit_ = 0;
  // S7/S8: assign this commit its submission id; the completion handler
  // publishes it and records faults (drained + logged + auto-recovered on
  // the GPU thread from PumpCompletedActions — a fault otherwise fails the
  // command buffer SILENTLY and reads on-screen as a freeze).
  uint64_t submission = ++submission_counter_;
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    if (cb.status == MTLCommandBufferStatusError) {
      std::lock_guard<std::mutex> lock(fault_mutex_);
      faulted_command_buffers_.push_back((__bridge_retained void*)cb);
      faulted_submission_ids_.push_back(submission);
    }
    PublishCompletedSubmission(submission);
  }];
  // Stamp the upload ring: everything allocated since the previous mark is
  // owned by this submission until it completes.
  if (upload_arena_) {
    upload_arena_->MarkSubmission(submission);
  }
  [command_buffer commit];
  // S8: NO synchronous wait — the GPU pipelines behind CPU encoding.
  // XE_METAL_SYNC_WAIT=1 restores the pre-S8 commit+wait for A/B.
  static const bool s_sync_wait =
      std::getenv("XE_METAL_SYNC_WAIT") != nullptr ||
      std::getenv("XE_METAL_ASYNC") == nullptr;
  if (s_sync_wait) {
    [command_buffer waitUntilCompleted];
    PublishCompletedSubmission(submission);
  } else {
    // Bound the pipeline depth: cap in-flight submissions so a fast CPU
    // can't queue unbounded GPU work (memory + latency + ring pressure).
    constexpr uint64_t kMaxInFlightSubmissions = 8;
    if (submission > kMaxInFlightSubmissions) {
      WaitForSubmission(submission - kMaxInFlightSubmissions);
    }
  }
  // Reclamation runs once THIS submission has completed: snapshot textures
  // retired mid-frame are then no longer referenced by any in-flight draw,
  // bindless slots can recycle, the upload-ring region frees, and later
  // uploads may rewrite any shared-memory range the submission consumed.
  completion_actions_.emplace_back(submission, [this, submission]() {
    if (upload_arena_) {
      upload_arena_->Reclaim(submission);
    }
    if (render_target_cache_) {
      render_target_cache_->DrainDeferredReleases(submission);
    }
    if (shared_texture_cache_) {
      // LRU destruction gated on completed submissions; retired slots/views
      // land in texture_cache_'s pools reclaimed below/next cycle.
      shared_texture_cache_->CompletedSubmissionUpdated(submission);
    }
    if (texture_cache_) {
      texture_cache_->ReclaimStaleSlots(submission);
    }
    if (shared_memory_) {
      shared_memory_->ReleaseConsumedRanges(submission);
    }
  });
  PumpCompletedActions();
}

void MetalCommandProcessor::EndFrameEncoderAndWait() {
  // For sites that read GPU results on the CPU right after the commit
  // (writeback drains, guest fence waits, EDRAM/RT readbacks, shutdown):
  // full synchronous semantics.
  bool had_buffer = frame_command_buffer_ != nullptr;
  EndFrameEncoder();
  if (had_buffer && WaitForSubmission(submission_counter_)) {
    PumpCompletedActions();
  }
}

Shader* MetalCommandProcessor::LoadShader(xenos::ShaderType shader_type,
                                          uint32_t guest_address,
                                          const uint32_t* host_address,
                                          uint32_t dword_count) {
  if (!pipeline_cache_) {
    return nullptr;
  }
  return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
}

// Bring-up diagnostic toggle: replace the guest pixel shader with a solid color
// so we can tell whether the vertex side (positions/constants) is producing
// visible geometry independent of the guest pixel shader / textures.
static constexpr bool kDebugSolidFragment = false;

// Returns a cached id<MTLRenderPipelineState> pairing the given guest vertex
// function with a hardcoded solid-color fragment function.
static void* GetDebugSolidPipeline(
    void* device_ptr, void* vs_function, void*& debug_solid_fragment,
    std::unordered_map<void*, void*>& cache) {
  auto it = cache.find(vs_function);
  if (it != cache.end()) {
    return it->second;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
  if (!debug_solid_fragment) {
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      fragment float4 solid_fs(constant float4& c [[buffer(8)]]) { return c; }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    id<MTLFunction> fs = lib ? [lib newFunctionWithName:@"solid_fs"] : nil;
    if (!fs) {
      cache.emplace(vs_function, nullptr);
      return nullptr;
    }
    debug_solid_fragment = (__bridge_retained void*)fs;
  }
  MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
  d.vertexFunction = (__bridge id<MTLFunction>)vs_function;
  d.fragmentFunction = (__bridge id<MTLFunction>)debug_solid_fragment;
  d.colorAttachments[0].pixelFormat =
      (MTLPixelFormat)MetalRenderTargetCache::kColorFormat;
  NSError* err = nil;
  id<MTLRenderPipelineState> ps =
      [device newRenderPipelineStateWithDescriptor:d error:&err];
  void* ps_ptr = ps ? (__bridge_retained void*)ps : nullptr;
  cache.emplace(vs_function, ps_ptr);
  return ps_ptr;
}

bool MetalCommandProcessor::IssueDraw(xenos::PrimitiveType prim_type,
                                      uint32_t index_count,
                                      IndexBufferInfo* index_buffer_info,
                                      bool major_mode_explicit) {
  // An EDRAM "copy mode" draw is not geometry — it's a resolve (EDRAM ->
  // guest). The D3D12/Vulkan backends dispatch this from IssueDraw; without the
  // check our backend treated resolves as normal draws and IssueCopy never fired
  // (the gameplay gray screen: the scene is resolved to a guest texture we never
  // populate, so the composite samples the null/gray texture).
  if (register_file_->Get<reg::RB_MODECONTROL>().edram_mode ==
      xenos::EdramMode::kCopy) {
    return IssueCopy();
  }

  auto* vertex_shader = static_cast<MetalShader*>(active_vertex_shader());
  auto* pixel_shader = static_cast<MetalShader*>(active_pixel_shader());
  if (!vertex_shader || !pipeline_cache_ || !upload_arena_) {
    ++draw_skip_count_;
    return true;
  }


  // Rectangle and quad lists (fullscreen/composite surfaces) and point lists
  // (light-volume sprites — plain Metal points can't expand size/texcoords,
  // and their pipelines fail on the xespritetexcoord0 VS/PS mismatch) can't be
  // drawn directly; expand them via MSC geometry emulation, which needs a
  // mesh-shader-capable device and a pixel shader for the fragment stage.
  PipelineGeometryShader geometry_type = PipelineGeometryShader::kNone;
  if (prim_type == xenos::PrimitiveType::kRectangleList) {
    geometry_type = PipelineGeometryShader::kRectangleList;
  } else if (prim_type == xenos::PrimitiveType::kPointList) {
    geometry_type = PipelineGeometryShader::kPointList;
  } else if (prim_type == xenos::PrimitiveType::kQuadList) {
    geometry_type = PipelineGeometryShader::kQuadList;
  }
  bool use_geometry_emulation = geometry_type != PipelineGeometryShader::kNone &&
                                mesh_shader_supported_ && pixel_shader != nullptr;

  RegisterFile& regs = *register_file_;
  // PP3-auto line loops are intentionally absent from the guest primitive
  // mapping: they must first normalize through the shared PrimitiveProcessor
  // to the already-mapped host kLineStrip topology.
  bool auto_line_loop_route = IsAutoLineLoopRoutable(regs, prim_type);
  // PP4: enabled 16-bit DMA strips whose reset value already equals Metal's
  // fixed 0xFFFF sentinel need normalization fields only, never conversion.
  bool fixed_reset_strip_route = IsFixedResetStripRoutable(regs, prim_type);
  // A skipped draw is not an error (the base loop treats false as failure); we
  // return true and just render nothing so the game keeps advancing.
  MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
  if (!use_geometry_emulation && !auto_line_loop_route && !MapPrimitiveType(prim_type, &mtl_prim)) {
    ++draw_skip_count_;
    return true;
  }

  if (!AreMetalConstantBankBasesSupported(regs[XE_GPU_REG_SQ_VS_CONST],
                                          regs[XE_GPU_REG_SQ_PS_CONST])) {
    static uint32_t invalid_constant_bank_logs = 0;
    if (invalid_constant_bank_logs++ < 16) {
      XELOGW("MetalCommandProcessor: skipping draw with unsupported constant "
             "banks VS={:08X} PS={:08X}",
             regs[XE_GPU_REG_SQ_VS_CONST], regs[XE_GPU_REG_SQ_PS_CONST]);
    }
    ++draw_skip_count_;
    return true;
  }

  // Analyze microcode (once) so constant maps / bindings are available.
  StringBuffer ucode_disasm;
  if (!vertex_shader->is_ucode_analyzed()) {
    vertex_shader->AnalyzeUcode(ucode_disasm);
  }
  if (pixel_shader && !pixel_shader->is_ucode_analyzed()) {
    pixel_shader->AnalyzeUcode(ucode_disasm);
  }
  // Diagnostic: XE_METAL_DUMP_PS=<hex ucode hash> writes that pixel shader's
  // ucode disassembly to /tmp/xe_ps_<hash>.txt once, for single-shader
  // forensics (what inputs drive the color output).
  if (const char* dump_ps = std::getenv("XE_METAL_DUMP_PS")) {
    uint64_t dump_hash = strtoull(dump_ps, nullptr, 16);
    // Matches the PS hash OR the VS hash (several VSes share one PS).
    if ((pixel_shader && pixel_shader->ucode_data_hash() == dump_hash) ||
        vertex_shader->ucode_data_hash() == dump_hash) {
      static bool s_dumped = false;
      if (!s_dumped) {
        s_dumped = true;
        auto write_disasm = [](const char* kind, uint64_t hash,
                               const std::string& text) {
          std::string path = fmt::format("/tmp/xe_{}_{:016X}.txt", kind, hash);
          FILE* f = fopen(path.c_str(), "w");
          if (f) {
            fwrite(text.data(), 1, text.size(), f);
            fclose(f);
            XELOGI("MetalCommandProcessor: dumped {} disasm to {}", kind, path);
          }
        };
        write_disasm("ps", pixel_shader->ucode_data_hash(),
                     pixel_shader->ucode_disassembly());
        // The paired vertex shader (feeds the PS's interpolators).
        write_disasm("vs", vertex_shader->ucode_data_hash(),
                     vertex_shader->ucode_disassembly());
        // Raw ucode words too — the shader-executor harness's input.
        auto write_ucode = [](const char* kind, uint64_t hash,
                              const std::vector<uint32_t>& words) {
          std::string path =
              fmt::format("/tmp/xe_{}_{:016X}.ucode.bin", kind, hash);
          FILE* f = fopen(path.c_str(), "wb");
          if (f) {
            fwrite(words.data(), sizeof(uint32_t), words.size(), f);
            fclose(f);
            XELOGI("MetalCommandProcessor: dumped {} ucode ({} words) to {}",
                   kind, words.size(), path);
          }
        };
        if (pixel_shader) {
          write_ucode("ps", pixel_shader->ucode_data_hash(),
                      pixel_shader->ucode_data());
        }
        write_ucode("vs", vertex_shader->ucode_data_hash(),
                    vertex_shader->ucode_data());
      }
    }
  }

  // --- Guest render-target binding. The normalized color mask folds
  // RB_MODECONTROL + RB_COLOR_MASK + the PS's static writes; each enabled
  // color slot's EDRAM base/format and RB_DEPTH_INFO key the per-target views
  // in the RT cache. ---
  uint32_t normalized_color_mask =
      pixel_shader ? draw_util::GetNormalizedColorMask(
                         regs, pixel_shader->writes_color_targets())
                   : 0;
  reg::RB_DEPTHCONTROL norm_depth = draw_util::GetNormalizedDepthControl(regs);
  bool depth_bound = norm_depth.z_enable || norm_depth.stencil_enable;
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  // Write-provenance probe (XE_METAL_DEPTH_PROVENANCE): the pack-shader-
  // INDEPENDENT answer to "does the guest ever write real depth into the 4x
  // view?" — read straight from guest registers, so it can't be fooled by a
  // pack misread the way a host-texture nonzero count can. For the cross-MSAA
  // depth bypass this is the authority question: if no draw ever binds a 4x
  // depth surface with z-write, the 4x view carries no guest content and the
  // resample to/from it is pure churn; if some do, the empty shadow reads are a
  // pack artifact and the bypass would drop real depth. Throttled; grep
  // MetalDepthWrite and count by msaa/zwrite over depth_base 720.
  if (depth_bound) {
    static const bool s_depth_prov = std::getenv("XE_METAL_DEPTH_PROVENANCE");
    if (s_depth_prov) {
      static uint32_t s_prov_logs = 0;
      if (s_prov_logs++ < 4096) {
        uint32_t msaa = uint32_t(regs.Get<reg::RB_SURFACE_INFO>().msaa_samples);
        XELOGW("MetalDepthWrite @draw {}: base={}t msaa={}x fmt={} z_enable={} "
               "z_write={} sten_enable={}",
               trace_draw_index(), uint32_t(rb_depth_info.depth_base),
               1u << msaa,
               uint32_t(rb_depth_info.depth_format),
               uint32_t(norm_depth.z_enable),
               uint32_t(norm_depth.z_write_enable),
               uint32_t(norm_depth.stencil_enable));
      }
    }
  }
  bool color_bound = HasMetalColorTarget(normalized_color_mask);
  std::array<reg::RB_COLOR_INFO, xenos::kMaxColorRenderTargets> color_infos;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    color_infos[i] = regs.Get<reg::RB_COLOR_INFO>(
        reg::RB_COLOR_INFO::rt_register_indices[i]);
  }
  const reg::RB_COLOR_INFO& rt0_info = color_infos[0];
  {
    // Log each unique RT configuration once (kept from the MRT-proof pass; it
    // is the ground truth for the per-target model).
    uint64_t cfg_key =
        (uint64_t(normalized_color_mask) << 33) ^
        (color_bound ? regs[reg::RB_COLOR_INFO::rt_register_indices[0]] : 0) ^
        (depth_bound ? (uint64_t(rb_depth_info.value) | (1ull << 32)) : 0);
    static std::unordered_set<uint64_t> s_seen_rt_configs;
    if (s_seen_rt_configs.insert(cfg_key).second) {
      XELOGI(
          "MetalRTConfig: color={} (base={} fmt={} mask={:X} expbias={}) "
          "depth={} (base={} fmt={}) ncm={:04X}",
          color_bound, uint32_t(rt0_info.color_base),
          uint32_t(rt0_info.color_format), normalized_color_mask & 0b1111,
          int32_t(rt0_info.color_exp_bias), depth_bound,
          uint32_t(rb_depth_info.depth_base),
          uint32_t(rb_depth_info.depth_format), normalized_color_mask);
    }
  }
  // Memexport: draws that bind neither color nor depth but whose VS has
  // valid export streams execute as vertex-only rasterization-disabled
  // pipelines in an attachment-less pass, writing the shared-memory UAV
  // (JC2's HDR/lighting chain: 3x2MB point-list exports per frame).
  // XE_METAL_MEMEXPORT_LOG=1 logs executions and remaining skips.
  static const bool s_memexport_log =
      std::getenv("XE_METAL_MEMEXPORT_LOG") != nullptr;
  // The validated VS-only route is part of the default pipeline. Exact
  // XE_METAL_MEMEXPORT=0 retains the pre-promotion skip path as a rollback;
  // explicit 1 enables and malformed values fail closed through the same
  // strict environment contract as canonical EDRAM.
  static const bool s_memexport_exec = metal::ParseMetalEnvironmentFlag(
      std::getenv("XE_METAL_MEMEXPORT"), /*default_enabled=*/true);
  bool memexport_draw = false;
  std::vector<draw_util::MemExportRange> memexport_ranges;
  if (s_memexport_exec && vertex_shader->memexport_eM_written()) {
    draw_util::AddMemExportRanges(regs, *vertex_shader, memexport_ranges);
  }
  // Report memexport producer classes this backend does not route. Bounded and
  // deduplicated by MemexportGapReporter; unconditional by design, because the
  // symptom is missing guest data with nothing to attribute it to, and a
  // switch-gated report cannot catch what nobody is yet looking for.
  {
    const bool rt_bound_for_export = color_bound || (kDepthEnabled && depth_bound);
    const bool ps_writes_eM =
        pixel_shader && pixel_shader->memexport_eM_written();
    if (rt_bound_for_export && (!memexport_ranges.empty() || ps_writes_eM)) {
      const MemexportProducerClass producer_class =
          ps_writes_eM ? MemexportProducerClass::kPixelStage
                       : (use_geometry_emulation
                              ? MemexportProducerClass::kRasterizingVertexGeometry
                              : MemexportProducerClass::kRasterizingVertex);
      auto report = [&](uint32_t base_dwords, uint32_t size_bytes) {
        switch (memexport_gap_reporter_.Observe(producer_class, base_dwords,
                                                size_bytes)) {
          case MemexportGapReporter::Action::kReport:
            // Guest addresses and producer classification only.
            XELOGW(
                "MetalMemexportUnrouted class={} guest_base=0x{:08X} size={} "
                "-- UNPUBLISHED: this backend routes only the vertex-stage "
                "no-render-target class, so these writes never reach guest "
                "memory",
                MemexportProducerClassName(producer_class), base_dwords << 2,
                size_bytes);
            break;
          case MemexportGapReporter::Action::kEmitSuppressionSummary:
            XELOGW(
                "MetalMemexportUnrouted: reached {} distinct unrouted export "
                "reports this title; further reports suppressed",
                MemexportGapReporter::kMaxDistinctReports);
            break;
          case MemexportGapReporter::Action::kSuppressed:
            break;
        }
      };
      if (memexport_ranges.empty()) {
        // A pixel-stage export whose ranges production never gathers still has
        // a class worth reporting once; key it on a zero range.
        report(0u, 0u);
      } else {
        for (const auto& r : memexport_ranges) {
          report(r.base_address_dwords, r.size_bytes);
        }
      }
    }
  }

  if (!color_bound && !(kDepthEnabled && depth_bound)) {
    bool ps_exports = pixel_shader && pixel_shader->memexport_eM_written();
    if (memexport_ranges.empty() || ps_exports) {
      // Nothing to render and nothing exportable via the v1 path (plain
      // no-op draw, or a rasterizing PS export which v1 doesn't run).
      if (s_memexport_log) {
        if (ps_exports) {
          XELOGW("MetalMemexportSkip draw={} PS-export (unsupported in v1)",
                 trace_draw_index_);
        } else {
          static uint32_t s_plain_skips = 0;
          if (s_plain_skips++ < 8) {
            XELOGW("MetalMemexportSkip draw={} NO-RT draw without memexport "
                   "(plain skip)",
                   trace_draw_index_);
          }
        }
      }
      ++draw_skip_count_;
      return true;
    }
    // Commit the destination pages; failure means the exports have nowhere
    // to land — skip like the reference backends fail the draw. Routed
    // through the typed task-3.3 destination-preparation seam (normalized
    // range + the same RequestRange call the direct call site used before).
    bool ranges_ok = true;
    MetalMemexportPreparationTarget preparation_target(
        shared_memory_.get());
    for (const auto& range : memexport_ranges) {
      uint32_t range_base = range.base_address_dwords << 2;
      auto preparation = metal::DecideMemexportDestinationPreparation(
          range_base, range.size_bytes);
      bool requested =
          preparation && metal::PrepareMemexportDestination(
                              *preparation, preparation_target);
      if (!requested) {
        XELOGE("MetalMemexport: failed to request range {:08X}(+{})",
               range_base, range.size_bytes);
        ranges_ok = false;
      }
    }
    if (!ranges_ok) {
      ++draw_skip_count_;
      return true;
    }
    // Rasterization is disabled on this path: the pixel shader (if any) and
    // geometry emulation are irrelevant; the guest VS runs per-vertex as a
    // plain vertex function.
    memexport_draw = true;
    pixel_shader = nullptr;
    use_geometry_emulation = false;
    if (!MapPrimitiveType(prim_type, &mtl_prim)) {
      if (s_memexport_log) {
        XELOGW("MetalMemexportSkip draw={} unmappable prim {} for export",
               trace_draw_index_, uint32_t(prim_type));
      }
      ++draw_skip_count_;
      return true;
    }
    if (s_memexport_log) {
      for (const auto& range : memexport_ranges) {
        XELOGW(
            "MetalMemexportExec draw={} VS shader={:016X} dest={:08X} "
            "bytes={} prim={} n={}",
            trace_draw_index_, vertex_shader->ucode_data_hash(),
            range.base_address_dwords << 2, range.size_bytes,
            uint32_t(prim_type), index_count);
      }
      // The export VS's index math runs off these — log once per draw so the
      // CPU-side simulation of the ucode has ground-truth inputs.
      auto logc = [&](uint32_t creg) {
        const float* c = reinterpret_cast<const float*>(
            &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + creg * 4]);
        XELOGW("MetalMemexportExec   c{}=({:g}, {:g}, {:g}, {:g})", creg, c[0],
               c[1], c[2], c[3]);
      };
      logc(0);
      logc(1);
      logc(2);
      logc(253);
      logc(254);
      logc(255);
    }
  }
  // DIAGNOSTIC: the quarter-res lighting producer — log draws into small-pitch
  // color RTs (the inferred-lighting buffer) with everything they sample.
  if (color_bound && pixel_shader &&
      regs.Get<reg::RB_SURFACE_INFO>().surface_pitch <= 400) {
    static std::unordered_set<uint64_t> s_seen_light_producer;
    if (s_seen_light_producer.insert(pixel_shader->ucode_data_hash()).second &&
        s_seen_light_producer.size() <= 16) {
      XELOGI(
          "MetalLightProducer: ps={:016X} pitch={} fmt={} prim={} n={} geo={} "
          "blend={:08X} mask={:X} rt={:X} samples:",
          pixel_shader->ucode_data_hash(),
          uint32_t(regs.Get<reg::RB_SURFACE_INFO>().surface_pitch),
          uint32_t(rt0_info.color_format), uint32_t(prim_type), index_count,
          use_geometry_emulation, regs[XE_GPU_REG_RB_BLENDCONTROL0],
          normalized_color_mask & 0b1111,
          uintptr_t(render_target_cache_->current_color_texture(0, nullptr)));
      for (const auto& binding :
           pixel_shader->GetTextureBindingsAfterTranslation()) {
        auto f = regs.GetTextureFetch(binding.fetch_constant);
        uint32_t fetch_base = f.base_address << 12;
        XELOGI("MetalLightProducer:   fc={} base={:08X}{} fmt={} {}x{}",
               binding.fetch_constant, fetch_base,
               render_target_cache_->GetResolveSnapshot(fetch_base)
                   ? " [SNAPSHOT]"
                   : "",
               uint32_t(f.format), f.size_2d.width + 1, f.size_2d.height + 1);
      }
    }
  }
  // DIAGNOSTIC (lighting-term ~0): log, once per (pixel shader, RT config),
  // every draw whose PS samples a RESOLVED SNAPSHOT (the RT-consuming passes —
  // deferred lighting, composite, post) with what it samples and where it
  // writes, so the black lighting can be traced through the pass graph.
  if (color_bound && pixel_shader && render_target_cache_) {
    bool samples_snapshot = false;
    for (const auto& binding :
         pixel_shader->GetTextureBindingsAfterTranslation()) {
      auto f = regs.GetTextureFetch(binding.fetch_constant);
      if (render_target_cache_->GetResolveSnapshot(f.base_address << 12)) {
        samples_snapshot = true;
        break;
      }
    }
    if (samples_snapshot) {
      static std::unordered_set<uint64_t> s_seen_consumer_ps;
      uint64_t seen_key = pixel_shader->ucode_data_hash() ^
                          (uint64_t(rt0_info.value) << 1);
      if (s_seen_consumer_ps.insert(seen_key).second) {
        XELOGI(
            "MetalRTConsumer: ps={:016X} -> color(base={} fmt={}) prim={} "
            "depth={} zfunc={} samples:",
            pixel_shader->ucode_data_hash(), uint32_t(rt0_info.color_base),
            uint32_t(rt0_info.color_format), uint32_t(prim_type), depth_bound,
            uint32_t(norm_depth.zfunc));
        for (const auto& binding :
             pixel_shader->GetTextureBindingsAfterTranslation()) {
          auto f = regs.GetTextureFetch(binding.fetch_constant);
          uint32_t fetch_base = f.base_address << 12;
          bool is_snap =
              render_target_cache_->GetResolveSnapshot(fetch_base) != nullptr;
          XELOGI(
              "MetalRTConsumer:   fc={} base={:08X}{} fmt={} {}x{} "
              "swz={:03X} signs={:02X}",
              binding.fetch_constant, fetch_base, is_snap ? " [SNAPSHOT]" : "",
              uint32_t(f.format), f.size_2d.width + 1, f.size_2d.height + 1,
              f.swizzle & 0xFFF, texture_util::SwizzleSigns(f));
        }
      }
    }
  }
  // Task 4.2: collect every draw-input vertex range before any CPU-side draw
  // estimation, trace capture, primitive processing, or Metal encoding reads
  // it. The exact DMA index interval participates in the same kVertex request
  // because this backend materializes indices directly from guest RAM. Only
  // vfetch ranges are revalidated in the shared buffer below; indices remain a
  // RAM consumer.
  const Shader::ConstantRegisterMap& vs_map =
      vertex_shader->constant_register_map();
  struct VertexFetchDependency {
    uint32_t fetch_constant;
    xenos::xe_gpu_vertex_fetch_t fetch;
    metal::GuestRange range;
  };
  std::vector<VertexFetchDependency> vertex_fetch_dependencies;
  metal::GuestRangeSet vertex_dependency_ranges;
  for (uint32_t i = 0; i < xe::countof(vs_map.vertex_fetch_bitmap); ++i) {
    uint32_t bits = vs_map.vertex_fetch_bitmap[i];
    uint32_t j;
    while (xe::bit_scan_forward(bits, &j)) {
      bits = xe::clear_lowest_bit(bits);
      uint32_t fetch_constant = i * 32 + j;
      xenos::xe_gpu_vertex_fetch_t fetch =
          regs.GetVertexFetch(fetch_constant);
      if (fetch.type != xenos::FetchConstantType::kVertex &&
          fetch.type != xenos::FetchConstantType::kInvalidVertex) {
        continue;
      }
      uint32_t fetch_base = fetch.address << 2;
      auto range = metal::GuestRange::FromBounds(
          fetch_base,
          uint64_t(fetch_base) + uint64_t(fetch.size) * sizeof(uint32_t));
      if (!range) {
        continue;
      }
      vertex_fetch_dependencies.push_back(
          {fetch_constant, fetch, *range});
      vertex_dependency_ranges.Add(*range);
    }
  }
  if (index_buffer_info && index_buffer_info->guest_base && index_count) {
    uint32_t index_size_bytes =
        index_buffer_info->format == xenos::IndexFormat::kInt32 ? 4 : 2;
    vertex_dependency_ranges.Add(metal::GuestRange::FromBounds(
        index_buffer_info->guest_base,
        uint64_t(index_buffer_info->guest_base) +
            uint64_t(index_count) * index_size_bytes));
  }
  vertex_dependency_ranges.Normalize();

  if (!vertex_dependency_ranges.empty() &&
      (!pending_writebacks_.empty() || !pending_memexports_.empty())) {
    metal::ConsumerDecision authority_decision =
        QueryPendingProducerDependencies(
            metal::ConsumerKind::kVertex,
            metal::ConsumerRequest::Uniform(vertex_dependency_ranges));
    ProducerDependencyOutcome dependency_outcome =
        ResolvePendingProducerDependencies(authority_decision);
    if (!dependency_outcome.complete) {
      XELOGE(
          "MetalCommandProcessor: vertex dependency convergence retired {} "
          "of {} selected producers; skipping draw",
          dependency_outcome.drain.drained_records,
          authority_decision.selected.size());
      ++draw_skip_count_;
      return true;
    }
  }

  // VB sentinel probe (XE_METAL_VB_SENTINEL=<draw>): overwrite the draw's
  // vertex floats with per-dword sentinels z = (w+1)*8 + 0.5 so a consumer
  // like frc(z*c254) encodes WHICH dword the translated vfetch actually
  // read into the output color (o1.y = (w+1)/8 + 0.0078 -> G ~= 7,15,22,
  // 29,36,44 * tex.a for w=0..5). Settles offset-vs-endian in the
  // light-VS zero-output hunt empirically.
  static const int64_t s_vb_sentinel = [] {
    const char* v = std::getenv("XE_METAL_VB_SENTINEL");
    return v ? strtoll(v, nullptr, 10) : -1ll;
  }();
  if (s_vb_sentinel >= 0 &&
      int64_t(trace_draw_index_) - 1 == s_vb_sentinel && vertex_shader) {
    for (const auto& vb : vertex_shader->vertex_bindings()) {
      xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(vb.fetch_constant);
      uint32_t vaddr = vf.address << 2;
      uint32_t stride = vb.stride_words ? vb.stride_words : 4;
      if (!vaddr || !vf.size) continue;
      uint32_t* vdata =
          reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(vaddr));
      uint32_t vcount = std::min<uint32_t>(vf.size / stride, 8);
      for (uint32_t vtx = 0; vtx < vcount; ++vtx) {
        // Keep dwords 0-2 (position) intact — sentinel only the tail
        // (uv/packed attributes) so the quad still rasterizes.
        for (uint32_t w = 3; w < stride && w < 8; ++w) {
          float sentinel = float(w + 1) * 8.0f + 0.5f;
          uint32_t bits;
          std::memcpy(&bits, &sentinel, 4);
          vdata[vtx * stride + w] =
              xenos::GpuSwapInline(bits, xenos::Endian(vf.endian));
        }
      }
      if (shared_memory_) {
        shared_memory_->MemoryInvalidationCallback(vaddr, vf.size * 4, true);
      }
      XELOGW("MetalVBSentinel: draw {} fc={} addr={:08X} patched {} verts",
             s_vb_sentinel, vb.fetch_constant, vaddr, vcount);
    }
  }

  // Revalidate each exact vfetch interval only after all selected producer
  // records have converged to RAM and after the diagnostic writer above. This
  // is still before RenderTargetCache::Update and BeginFrameEncoder, so both
  // CPU shader interpretation and the consuming Metal draw see the same bytes.
  static const char* s_log_vf = std::getenv("XE_METAL_LOG_VF");
  bool log_vf =
      s_log_vf &&
      vertex_shader->ucode_data_hash() == strtoull(s_log_vf, nullptr, 16);
  static uint32_t s_log_vf_count = 0;
  for (const VertexFetchDependency& dependency :
       vertex_fetch_dependencies) {
    if (log_vf && s_log_vf_count < 24) {
      ++s_log_vf_count;
      const uint32_t* words = reinterpret_cast<const uint32_t*>(
          memory_->TranslatePhysical(dependency.range.start()));
      uint32_t sampled_words[4] = {};
      if (words) {
        std::memcpy(sampled_words, words,
                    std::min<size_t>(dependency.range.length(),
                                     sizeof(sampled_words)));
      }
      XELOGW("MetalVF draw={} slot={} type={} addr={:08X} size_dw={} "
             "endian={} words={:08X} {:08X} {:08X} {:08X}",
             trace_draw_index_, dependency.fetch_constant,
             uint32_t(dependency.fetch.type), dependency.range.start(),
             uint32_t(dependency.fetch.size),
             uint32_t(dependency.fetch.endian), sampled_words[0],
             sampled_words[1], sampled_words[2], sampled_words[3]);
    }
    if (!shared_memory_->RequestConsumedRange(dependency.range.start(),
                                               dependency.range.length())) {
      XELOGE("MetalCommandProcessor: failed to validate vertex range "
             "{:08X}(+{})",
             dependency.range.start(), dependency.range.length());
      ++draw_skip_count_;
      return true;
    }
  }
  // Frame-trace recording: this backend reads guest indices directly from
  // RAM (not through shared memory), so record the range for replay. No-op
  // unless a trace file is open.
  if (index_buffer_info && index_buffer_info->guest_base && index_count) {
    uint32_t index_size_bytes =
        index_buffer_info->format == xenos::IndexFormat::kInt32 ? 4 : 2;
    trace_writer_.WriteMemoryRead(index_buffer_info->guest_base,
                                  size_t(index_count) * index_size_bytes);
  }
  // Base-class EDRAM ownership update: derives the RenderTargetKeys from the
  // registers, creates/accumulates the bound host RTs, and computes+executes
  // the ownership transfers this draw needs (stage 2 of the render chain).
  // Memexport-only draws touch no EDRAM: skip the update (the next normal
  // draw re-runs it) and the attachment checks below.
  if (!memexport_draw &&
      !render_target_cache_->Update(color_bound || depth_bound, norm_depth,
                                    normalized_color_mask, *vertex_shader)) {
    ++draw_skip_count_;
    return true;
  }
  // A/B probe: XE_METAL_NO_SMALL_DEPTH=1 disables the depth test on
  // small-pitch (quarter-res lighting) draws to isolate the light buffer's
  // zero output (depth-test vs shader-math).
  static const bool s_no_small_depth = std::getenv("XE_METAL_NO_SMALL_DEPTH");
  if (s_no_small_depth && depth_bound &&
      regs.Get<reg::RB_SURFACE_INFO>().surface_pitch <= 400) {
    norm_depth.value = 0;
    depth_bound = false;
    if (!render_target_cache_->Update(color_bound, norm_depth,
                                      normalized_color_mask, *vertex_shader)) {
      ++draw_skip_count_;
      return true;
    }
  }
  // The truth of what's bound now comes from the accumulated RTs.
  uint64_t color_pixel_formats[xenos::kMaxColorRenderTargets] = {};
  bool have_color = false;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    color_pixel_formats[i] = uint64_t(MTLPixelFormatInvalid);
    if (!memexport_draw &&
        render_target_cache_->current_color_texture(
            i, &color_pixel_formats[i]) != nullptr) {
      have_color = true;
    }
  }
  depth_bound = !memexport_draw && kDepthEnabled &&
                render_target_cache_->current_depth_texture() != nullptr;
  if (!have_color && !depth_bound && !memexport_draw) {
    ++draw_skip_count_;
    return true;
  }
  // Diagnostic: XE_METAL_PRECLEAR_RT=<trace draw index> clears the bound
  // color-0 RT to magenta right before encoding that draw. Combined with
  // XE_METAL_DUMP_RT on the same index, the post-draw dump shows the draw's
  // raster coverage exactly: any non-magenta pixel was written BY this draw.
  static const int64_t s_preclear_rt_at = [] {
    const char* v = std::getenv("XE_METAL_PRECLEAR_RT");
    return v ? strtoll(v, nullptr, 10) : -1ll;
  }();
  if (s_preclear_rt_at >= 0 &&
      int64_t(trace_draw_index_) - 1 == s_preclear_rt_at && have_color) {
    EndFrameEncoderAndWait();
    void* rt_ptr = render_target_cache_->current_color_texture(0, nullptr);
    if (rt_ptr) {
      id<MTLCommandQueue> pq = (__bridge id<MTLCommandQueue>)
          static_cast<xe::ui::metal::MetalProvider*>(
              graphics_system_->provider())->command_queue();
      id<MTLCommandBuffer> pcb = [pq commandBuffer];
      MTLRenderPassDescriptor* pd = [MTLRenderPassDescriptor renderPassDescriptor];
      pd.colorAttachments[0].texture = (__bridge id<MTLTexture>)rt_ptr;
      pd.colorAttachments[0].loadAction = MTLLoadActionClear;
      pd.colorAttachments[0].storeAction = MTLStoreActionStore;
      // Alpha 0 so post-draw alpha == the accumulated SRC alpha (additive
      // light passes: out = src*srcA + dst — separating "src small" from
      // "alpha small" in the zero-contribution hunt).
      pd.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 0.0, 1.0, 0.0);
      id<MTLRenderCommandEncoder> pe = [pcb renderCommandEncoderWithDescriptor:pd];
      [pe endEncoding];
      [pcb commit];
      [pcb waitUntilCompleted];
      XELOGW("XE_METAL_PRECLEAR_RT: cleared RT to magenta before draw {}",
             s_preclear_rt_at);
    }
  }

  // Translate to MTLFunctions (cached per modification).
  auto sq_program_cntl = regs.Get<reg::SQ_PROGRAM_CNTL>();

  // Interpolator mask + PsParamGen destination: which VS-output interpolators
  // (texcoords etc.) the PS consumes, and whether the PS needs the
  // auto-generated pixel-position/point-coord register. This MUST be threaded
  // into BOTH shader modifications and MUST match between them, or the DXBC
  // translator wires VS outputs to PS inputs wrong -> scrambled/scaled texcoords
  // (the splash quad sampled only a sub-rect of its texture = the on-screen
  // zoom). Mirrors the D3D12 backend's UpdateBindings path exactly.
  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask =
      pixel_shader ? (vertex_shader->writes_interpolators() &
                      pixel_shader->GetInterpolatorInputMask(
                          sq_program_cntl, regs.Get<reg::SQ_CONTEXT_MISC>(),
                          ps_param_gen_pos))
                   : 0;

  DxbcShaderTranslator::Modification vs_mod_decoded(
      pipeline_cache_->shader_translator()->GetDefaultVertexShaderModification(
          vertex_shader->GetDynamicAddressableRegisterCount(
              sq_program_cntl.vs_num_reg),
          Shader::HostVertexShaderType::kVertex));
  vs_mod_decoded.vertex.interpolator_mask = interpolator_mask;
  {
    auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
    uint32_t user_clip_planes =
        pa_cl_clip_cntl.clip_disable ? 0 : pa_cl_clip_cntl.ucp_ena;
    vs_mod_decoded.vertex.user_clip_plane_count =
        xe::bit_count(user_clip_planes);
    vs_mod_decoded.vertex.user_clip_plane_cull =
        uint32_t(user_clip_planes && pa_cl_clip_cntl.ucp_cull_only_ena);
    vs_mod_decoded.vertex.vertex_kill_and = uint32_t(
        (vertex_shader->writes_point_size_edge_flag_kill_vertex() & 0b100) &&
        !pa_cl_clip_cntl.vtx_kill_or);
    vs_mod_decoded.vertex.output_point_size = uint32_t(
        (vertex_shader->writes_point_size_edge_flag_kill_vertex() & 0b001) &&
        regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type ==
            xenos::PrimitiveType::kPointList);
  }
  uint64_t vs_mod = vs_mod_decoded.value;
  auto* vs_translation = static_cast<MetalShader::MetalTranslation*>(
      vertex_shader->GetOrCreateTranslation(vs_mod));
  if (!pipeline_cache_->TranslateShader(*vs_translation)) {
    ++draw_skip_count_;
    return true;
  }
  MetalShader::MetalTranslation* ps_translation = nullptr;
  uint64_t ps_mod_value = 0;
  if (pixel_shader) {
    DxbcShaderTranslator::Modification ps_mod_decoded(
        pipeline_cache_->shader_translator()->GetDefaultPixelShaderModification(
            pixel_shader->GetDynamicAddressableRegisterCount(
                sq_program_cntl.ps_num_reg)));
    ps_mod_decoded.pixel.interpolator_mask = interpolator_mask;
    // A/B: XE_METAL_NO_CENTROID=1 forces center sampling. Gameplay surfaces are
    // 4x MSAA (menus 1x), and at 4x the PS modification requests CENTROID
    // interpolation on some inputs — if MSC mis-links centroid-decorated PS
    // inputs against the (non-centroid) VS outputs, lighting interpolators read
    // garbage.
    static const bool s_no_centroid = std::getenv("XE_METAL_NO_CENTROID");
    ps_mod_decoded.pixel.interpolators_centroid =
        s_no_centroid ? 0
                      : (interpolator_mask &
                         ~xenos::GetInterpolatorSamplingPattern(
                             regs.Get<reg::RB_SURFACE_INFO>().msaa_samples,
                             regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
                             regs.Get<reg::SQ_INTERPOLATOR_CNTL>()
                                 .sampling_pattern));
    if (ps_param_gen_pos < xenos::kMaxInterpolators) {
      ps_mod_decoded.pixel.param_gen_enable = 1;
      ps_mod_decoded.pixel.param_gen_interpolator = ps_param_gen_pos;
      ps_mod_decoded.pixel.param_gen_point =
          uint32_t(regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type ==
                   xenos::PrimitiveType::kPointList);
    }
    // The RT0 MIN/MAX-blend premultiply factors default to kZero in a zeroed
    // modification, which forces RT0 rgb/a to 0 (pure black) in the non-ROV
    // output path. The neutral value is kOne (no premultiply); MIN/MAX blend
    // handling comes with the blend state later.
    ps_mod_decoded.pixel.rt0_blend_rgb_factor_for_premult =
        xenos::BlendFactor::kOne;
    ps_mod_decoded.pixel.rt0_blend_a_factor_for_premult =
        xenos::BlendFactor::kOne;
    uint64_t ps_mod = ps_mod_decoded.value;
    ps_mod_value = ps_mod;
    ps_translation = static_cast<MetalShader::MetalTranslation*>(
        pixel_shader->GetOrCreateTranslation(ps_mod));
    if (!pipeline_cache_->TranslateShader(*ps_translation)) {
      ps_translation = nullptr;
    }
  }

  void* pipeline;
  // Guest PS exports depth but this draw binds no depth target (z + stencil
  // disabled). Metal rejects a depth-writing fragment function without a
  // depth attachment — the draw used to NULL-PIPELINE skip (Sonic's
  // fullscreen bloom composite 1088: skipped -> the resolve exported stale
  // mip debris = the top-band scramble). Attach a scratch throwaway depth
  // target instead; the default encoder depth state (test Always, write off)
  // gives hardware semantics: with z disabled, oDepth is ignored.
  // XE_METAL_NO_DUMMY_DEPTH=1 restores the old NULL-PIPELINE skip for A/B
  // (e.g. Sonic: the skip hid a white-out from bad composite inputs).
  static const bool s_no_dummy_depth =
      std::getenv("XE_METAL_NO_DUMMY_DEPTH") != nullptr;
  bool dummy_depth =
      !s_no_dummy_depth && pixel_shader && pixel_shader->writes_depth() &&
      !(kDepthEnabled && depth_bound);
  uint64_t pipeline_depth_format =
      ((kDepthEnabled && depth_bound) || dummy_depth)
          ? MetalRenderTargetCache::kDepthFormat
          : uint64_t(MTLPixelFormatInvalid);
  // Geometry-emulation config (VS output size + GS max input primitives per mesh
  // threadgroup), filled by the geometry pipeline and passed to the draw call.
  uint32_t gs_vertex_size = 0;
  uint32_t gs_max_input_primitives = 0;
  // The pipeline sample count must exactly match the selected raster working
  // views. The default and sample-only gates remain single-sample; task 4.1's
  // strict raster gate selects the common real 2x/4x attachment count.
  MetalPipelineRasterState pipeline_raster_state =
      GetMetalPipelineRasterState(
          memexport_draw ? 1u
                         : render_target_cache_->current_raster_sample_count(),
          regs.Get<reg::RB_COLORCONTROL>());
  uint32_t raster_sample_count =
      pipeline_raster_state.pipeline_key().raster_sample_count;
  uint32_t blend_controls[xenos::kMaxColorRenderTargets] = {};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    blend_controls[i] =
        regs.Get<reg::RB_BLENDCONTROL>(
                reg::RB_BLENDCONTROL::rt_register_indices[i])
            .value;
  }
  if (memexport_draw) {
    // Vertex-only, rasterization disabled, zero attachments.
    pipeline = pipeline_cache_->GetOrCreateMemexportPipelineState(vs_translation);
  } else if (kDebugSolidFragment) {
    pipeline = GetDebugSolidPipeline(
        static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider())
            ->device(),
        vs_translation->metal_function(), debug_solid_fragment_,
        debug_solid_pipelines_);
  } else if (use_geometry_emulation) {
    pipeline = pipeline_cache_->GetOrCreateGeometryPipelineState(
        vs_translation, ps_translation, geometry_type,
        DxbcShaderTranslator::Modification(vs_mod),
        DxbcShaderTranslator::Modification(ps_mod_value), color_pixel_formats,
        pipeline_depth_format, pipeline_raster_state.pipeline_key(),
        blend_controls,
        normalized_color_mask,
        &gs_vertex_size, &gs_max_input_primitives);
  } else {
    pipeline = pipeline_cache_->GetOrCreateRenderPipelineState(
        vs_translation, ps_translation, color_pixel_formats,
        pipeline_depth_format, pipeline_raster_state.pipeline_key(),
        blend_controls,
        normalized_color_mask);
  }
  if (!pipeline) {
    if (use_geometry_emulation) {
      ++geo_pipeline_fail_count_;
    }
    static std::unordered_set<uint64_t> s_null_pipeline_ps;
    if (pixel_shader &&
        s_null_pipeline_ps.insert(pixel_shader->ucode_data_hash()).second &&
        s_null_pipeline_ps.size() <= 16) {
      XELOGW("MetalCommandProcessor: NULL PIPELINE skip ps={:016X} fmt={:X}",
             pixel_shader->ucode_data_hash(), color_pixel_formats[0]);
    }
    ++draw_skip_count_;
    return true;
  }

  // STAGE 3 lazy writeback flush: if this draw SAMPLES a range a pending
  // resolve writeback covers (same-frame consumer — e.g. each luminance
  // pyramid level reads the previous level's resolve), commit + drain so the
  // fetch's guest-RAM upload sees the resolved content. MUST run before the
  // encoder opens (a flush here would kill an open encoder — the
  // setViewport-on-ended-encoder crash class). Cross-frame consumers cost
  // nothing: their writebacks drained at a prior swap.
  if (!pending_writebacks_.empty() || !pending_memexports_.empty()) {
    auto collect_texture_ranges = [&](MetalShader* shader,
                                      metal::ConsumerRequest& request) {
      if (!shader) return;
      for (const auto& binding : shader->GetTextureBindingsAfterTranslation()) {
        auto f = regs.GetTextureFetch(binding.fetch_constant);
        uint32_t fbase = f.base_address << 12;
        if (!fbase) continue;

        metal::FetchGuestRanges rng = ComputeFetchGuestRanges(f);
        auto base_range =
            metal::GuestRange::FromBounds(rng.base_begin, rng.base_end);
        auto mip_range =
            metal::GuestRange::FromBounds(rng.mip_begin, rng.mip_end);

        // Every RAM-backed fetch can depend on memexport. A bridge-served
        // resolve is already GPU-ordered, while memexport at the same address
        // still needs convergence because the bridge does not contain it.
        request.memexport_ranges.Add(base_range);
        if (rng.mip_end > rng.mip_begin) {
          request.memexport_ranges.Add(mip_range);
        }
        bool bridge_served =
            texture_cache_ &&
            texture_cache_->BridgeServes(fbase, f.size_2d.width + 1,
                                          f.size_2d.height + 1);
        if (!bridge_served) {
          request.resolve_ranges.Add(base_range);
          if (rng.mip_end > rng.mip_begin) {
            request.resolve_ranges.Add(mip_range);
          }
        }
      }
    };
    metal::ConsumerRequest texture_request;
    collect_texture_ranges(pixel_shader, texture_request);
    collect_texture_ranges(vertex_shader, texture_request);
    metal::ConsumerDecision authority_decision =
        QueryPendingProducerDependencies(metal::ConsumerKind::kTexture,
                                         texture_request);
    // Task 4.3 routes both texture-cache implementations through the common
    // selected-record executor: one boundary, selective drain, before either
    // cache consumes the RAM-derived bytes below.
    ProducerDependencyOutcome dependency_outcome =
        ResolvePendingProducerDependencies(authority_decision);
    if (!dependency_outcome.complete) {
      XELOGE(
          "MetalCommandProcessor: texture dependency convergence retired {} "
          "of {} selected producers; skipping draw",
          dependency_outcome.drain.drained_records,
          authority_decision.selected.size());
      ++draw_skip_count_;
      return true;
    }
  }

  // C1T-R prepass: prepare eligible strict-RGBA8 kRamContent
  // misses through the shared TextureCache subclass at the proven
  // post-overlap/drain pre-encoder seam. The mask computed here is THE
  // per-draw routing decision — descriptor marshalling serves exactly these
  // indices from the subclass (seam audit M2: prep and serve can never
  // disagree), and classification/eligibility are the single existing
  // contracts (ClassifyFetchService + metal_texture_load_eligibility.h).
  draw_shared_route_mask_ = 0;
  // Capture disengage: while a frame trace is being recorded the CPU path
  // serves everything, so the capture records exactly the CPU-oracle
  // read-set — recorded ranges are IDENTICAL to route-off by construction
  // (a routed prep during capture would add page-granular shared-memory
  // upload records the CPU path never emits). Replays re-engage the route
  // from restored RAM.
  if (shared_texture_cache_ && texture_cache_ && !trace_writer_.is_open()) {
    auto accumulate_eligible = [&](MetalShader* shader) {
      if (!shader) {
        return;
      }
      for (const auto& binding : shader->GetTextureBindingsAfterTranslation()) {
        if (binding.is_signed ||
            binding.fetch_constant >= xenos::kTextureFetchConstantCount ||
            (draw_shared_route_mask_ & (1u << binding.fetch_constant))) {
          continue;
        }
        auto fetch = regs.GetTextureFetch(binding.fetch_constant);
        if (fetch.type != xenos::FetchConstantType::kTexture) {
          continue;
        }
        uint32_t fetch_width = fetch.size_2d.width + 1;
        uint32_t fetch_height = fetch.size_2d.height + 1;
        uint32_t fetch_pitch_texels = fetch.pitch
                                          ? (uint32_t(fetch.pitch) << 5)
                                          : ((fetch_width + 31u) & ~31u);
        // Two mutually exclusive strict classes: the single-mip predicate keeps
        // deciding its own class unchanged, and the mipped sibling decides the
        // chain-bearing one.
        if (!metal::IsStrictRgba8RouteEligible(fetch, false, fetch_width,
                                               fetch_height,
                                               fetch_pitch_texels) &&
            !metal::IsStrictRgba8MipRouteEligible(fetch, false, fetch_width,
                                                  fetch_height,
                                                  fetch_pitch_texels)) {
          continue;
        }
        if (texture_cache_->ClassifyFetchService(fetch, false) !=
            ResolvedFetchService::kRamContent) {
          continue;
        }
        draw_shared_route_mask_ |= 1u << binding.fetch_constant;
      }
    };
    accumulate_eligible(vertex_shader);
    accumulate_eligible(pixel_shader);
    if (draw_shared_route_mask_) {
      shared_texture_cache_->PrepareEligible(draw_shared_route_mask_);
    }
  }

  // PP1 + PP2-auto + PP3-auto + PP4-fixed: normalize the routed draws through
  // the shared
  // PrimitiveProcessor — the single normalization authority for the routed set.
  // PP1 routes native restart-disabled line/triangle list/strip draws;
  // PP2-auto additionally routes auto-indexed triangle fans (to the built-in
  // fan table), PP3-auto routes auto-indexed line loops to a host line strip
  // plus a closing-index remap, and PP4-fixed routes only matching 16-bit DMA
  // strip reset without conversion. This runs at the proven pre-encoder seam
  // (before BeginFrameEncoder below); the Metal override of
  // RequestGuestIndexSharedMemoryRange is a no-op, so Process performs no
  // shared-memory RequestRange — avoiding the unwanted shared-buffer upload and
  // its intra-frame hazard flush (commit/wait). The routed set reads no guest
  // index memory inside Process (PP1 pass-through and auto fans alike), so it
  // emits no trace record of its own; the existing explicit WriteMemoryRead of
  // any DMA index range above preserves capture (auto fans/loops have no such
  // range).
  // A routed draw skipped downstream is harmless. The bind/draw is emitted from
  // pp_result below.
  bool pp_routed = false;
  // PP2-auto: auto-indexed triangle fans additionally route through Process to
  // the shared kHostBuiltinForAuto fan table (center-first order, byte/order-
  // identical to the inline auto-fan expansion). DMA/reset fans stay inline.
  bool auto_fan_route = IsAutoFanRoutable(regs, prim_type);
  PrimitiveProcessor::ProcessingResult pp_result{};
  if (primitive_processor_ && !use_geometry_emulation &&
      (IsPp1Routable(regs, prim_type) || auto_fan_route ||
       auto_line_loop_route || fixed_reset_strip_route) &&
      IsPp1DrawViewConsistent(
          regs, index_count, index_buffer_info != nullptr,
          index_buffer_info ? index_buffer_info->guest_base : 0,
          index_buffer_info ? index_buffer_info->format : xenos::IndexFormat::kInt16,
          index_buffer_info ? index_buffer_info->endianness : xenos::Endian::kNone,
          index_buffer_info ? index_buffer_info->count : 0)) {
    if (primitive_processor_->Process(pp_result)) {
      if (fixed_reset_strip_route) {
        // Accept only the non-converting fixed-sentinel result proven by the
        // PP4 L0 matrix and live census. Any future shared result-shape change
        // fails closed; converted buffers are not owned by this slice.
        bool fixed_reset_shape =
            pp_result.host_primitive_type == prim_type &&
            pp_result.host_vertex_shader_type ==
                Shader::HostVertexShaderType::kVertex &&
            pp_result.host_draw_vertex_count == index_count &&
            pp_result.host_primitive_reset_enabled &&
            pp_result.index_buffer_type ==
                PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
            pp_result.host_index_format == xenos::IndexFormat::kInt16;
        if (fixed_reset_shape) {
          pp_routed = true;
        } else {
          XELOGE(
              "MetalPrimitiveProcessor: routed fixed-reset strip did not "
              "produce the kGuestDMA/fixed-16 shape (guest_prim {} host_prim "
              "{} vs_type {} count {}/{} reset {} ibt {} format {}); "
              "skipping.",
              uint32_t(prim_type), uint32_t(pp_result.host_primitive_type),
              uint32_t(pp_result.host_vertex_shader_type),
              pp_result.host_draw_vertex_count, index_count,
              pp_result.host_primitive_reset_enabled,
              uint32_t(pp_result.index_buffer_type),
              uint32_t(pp_result.host_index_format));
          assert_always();
        }
      } else if (auto_line_loop_route) {
        // Accept exactly the auto-loop shared shape. count <= 1 deliberately
        // produces a zero-count no-draw result with closing index 0.
        uint32_t expected_count = PrimitiveProcessor::GetLineLoopStripIndexCount(index_count);
        uint32_t expected_closing_index = expected_count ? expected_count - 1 : 0;
        bool auto_loop_shape =
            pp_result.host_primitive_type == xenos::PrimitiveType::kLineStrip &&
            pp_result.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex &&
            !pp_result.host_primitive_reset_enabled &&
            pp_result.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kNone &&
            pp_result.host_draw_vertex_count == expected_count &&
            pp_result.line_loop_closing_index == expected_closing_index;
        if (auto_loop_shape) {
          pp_routed = true;
        } else {
          XELOGE("MetalPrimitiveProcessor: routed auto line loop did not produce "
                 "the kNone/kLineStrip shape (host_prim {} vs_type {} count {}/{} "
                 "closing {}/{} reset {} ibt {}); skipping.",
                 uint32_t(pp_result.host_primitive_type),
                 uint32_t(pp_result.host_vertex_shader_type), pp_result.host_draw_vertex_count,
                 expected_count, pp_result.line_loop_closing_index, expected_closing_index,
                 pp_result.host_primitive_reset_enabled, uint32_t(pp_result.index_buffer_type));
          assert_always();
        }
      } else if (auto_fan_route) {
        // Accept exactly the auto-fan built-in shape: a triangle list, plain
        // vertex shader, no reset, kHostBuiltinForAuto, with the expected
        // expanded count (0 for count<3 -> a no-draw success below).
        bool builtin_fan_shape =
            pp_result.host_primitive_type == xenos::PrimitiveType::kTriangleList &&
            pp_result.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex &&
            !pp_result.host_primitive_reset_enabled &&
            pp_result.index_buffer_type ==
                PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto &&
            pp_result.host_draw_vertex_count ==
                PrimitiveProcessor::GetTriangleFanListIndexCount(index_count);
        if (builtin_fan_shape) {
          pp_routed = true;
        } else {
          XELOGE("MetalPrimitiveProcessor: routed auto fan did not produce the "
                 "kHostBuiltinForAuto shape (host_prim {} vs_type {} count {}/{} "
                 "reset {} ibt {}); staying inline.",
                 uint32_t(pp_result.host_primitive_type),
                 uint32_t(pp_result.host_vertex_shader_type),
                 pp_result.host_draw_vertex_count,
                 PrimitiveProcessor::GetTriangleFanListIndexCount(index_count),
                 pp_result.host_primitive_reset_enabled,
                 uint32_t(pp_result.index_buffer_type));
          assert_always();
        }
      } else {
        // Fail-closed: only accept the exact PP1 pass-through shape. Anything
        // else (a conversion, tessellation, reset, count/topology change, or an
        // unexpected index-buffer type) means the routed set assumptions were
        // violated; leave pp_routed false so the inline path stays authoritative.
        PrimitiveProcessor::ProcessedIndexBufferType expected_ibt =
            index_buffer_info ? PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA
                              : PrimitiveProcessor::ProcessedIndexBufferType::kNone;
        bool pass_through_shape =
            pp_result.host_primitive_type == prim_type &&
            pp_result.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex &&
            pp_result.host_draw_vertex_count == index_count &&
            !pp_result.host_primitive_reset_enabled &&
            pp_result.index_buffer_type == expected_ibt;
        if (pass_through_shape) {
          pp_routed = true;
        } else {
          XELOGE("MetalPrimitiveProcessor: routed draw did not produce the PP1 "
                 "pass-through shape (guest_prim {} host_prim {} vs_type {} "
                 "count {}/{} reset {} ibt {}); staying inline.",
                 uint32_t(prim_type), uint32_t(pp_result.host_primitive_type),
                 uint32_t(pp_result.host_vertex_shader_type),
                 pp_result.host_draw_vertex_count, index_count,
                 pp_result.host_primitive_reset_enabled,
                 uint32_t(pp_result.index_buffer_type));
          assert_always();
        }
      }
    }
  }

  // Auto line loops have no valid inline Metal fallback: the guest topology is
  // intentionally unmapped. If view consistency, Process, or the exact shape
  // guard rejected the route, preserve the prior behavior and skip.
  if (auto_line_loop_route && !pp_routed) {
    ++draw_skip_count_;
    return true;
  }
  // A degenerate loop is a successful no-draw. Skip before opening an encoder
  // or allocating constants, and never expose the synthetic closing vertex.
  if (auto_line_loop_route && pp_result.host_draw_vertex_count == 0) {
    ++draw_skip_count_;
    return true;
  }

  dummy_depth_pass_ = dummy_depth;
  memexport_pass_ = memexport_draw;
  id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)BeginFrameEncoder();
  if (!encoder) {
    ++draw_skip_count_;
    return true;
  }

  // --- Marshal the constant buffers (b0-b4), matching the D3D12 backend. ---
  MetalUploadAllocation system_alloc{};
  DxbcShaderTranslator::SystemConstants system_constants;
  std::memset(&system_constants, 0, sizeof(system_constants));
  {
    MetalSystemConstantGuestState system_constant_guest_state =
        GetMetalSystemConstantGuestState(regs);
    // Viewport -> ndc_scale/offset (position-critical: zeroing collapses output).
    draw_util::ViewportInfo viewport_info;
    // x_max/y_max is the maximum host viewport extent, NOT the render-target
    // size. Both reference backends pass the device max (D3D12:
    // D3D12_VIEWPORT_BOUNDS_MAX, Vulkan: maxViewportDimensions) and let the
    // render target clip an oversized guest viewport. Passing 1280x720 here
    // instead made GetHostViewportInfo *rescale* (zoom-crop) any guest viewport
    // wider/taller than the RT into it — the splash-screen "zoom". Use the max
    // guest RT extent (8192, already the clip-disabled cap and within Metal's
    // 16384 viewport limit) so guest content beyond 1280x720 is clipped, not
    // scaled.
    draw_util::GetViewportInfoArgs vp_args = GetMetalViewportInfoArgs(
        regs, norm_depth, pixel_shader && pixel_shader->writes_depth());
    draw_util::GetHostViewportInfo(&vp_args, viewport_info);
    for (uint32_t i = 0; i < 3; ++i) {
      system_constants.ndc_scale[i] = viewport_info.ndc_scale[i];
      system_constants.ndc_offset[i] = viewport_info.ndc_offset[i];
    }
    draw_util::NormalizedUserClipPlanes user_clip_planes =
        draw_util::GetNormalizedUserClipPlanes(regs);
    std::memcpy(system_constants.user_clip_planes,
                user_clip_planes.equations,
                sizeof(system_constants.user_clip_planes));
    draw_util::NormalizedPointParameters point_parameters =
        draw_util::GetNormalizedPointParameters(regs, viewport_info, 1, 1);
    system_constants.point_vertex_diameter_min =
        point_parameters.vertex_diameter_min;
    system_constants.point_vertex_diameter_max =
        point_parameters.vertex_diameter_max;
    system_constants.point_constant_diameter[0] =
        point_parameters.constant_diameter[0];
    system_constants.point_constant_diameter[1] =
        point_parameters.constant_diameter[1];
    system_constants.point_screen_diameter_to_ndc_radius[0] =
        point_parameters.screen_diameter_to_ndc_radius[0];
    system_constants.point_screen_diameter_to_ndc_radius[1] =
        point_parameters.screen_diameter_to_ndc_radius[1];
    // Position-relevant flags.
    auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
    uint32_t flags = 0;
    if (memexport_draw) {
      // The translator predicates every eA/eM store on this flag (exports
      // compile in but no-op without it — the "stores did not land" oracle).
      flags |= DxbcShaderTranslator::kSysFlag_SharedMemoryIsUAV;
    }
    if (pa_cl_vte_cntl.vtx_xy_fmt) flags |= DxbcShaderTranslator::kSysFlag_XYDividedByW;
    if (pa_cl_vte_cntl.vtx_z_fmt) flags |= DxbcShaderTranslator::kSysFlag_ZDividedByW;
    if (pa_cl_vte_cntl.vtx_w0_fmt) flags |= DxbcShaderTranslator::kSysFlag_WNotReciprocal;
    if (system_constant_guest_state.primitive_polygonal) {
      flags |= DxbcShaderTranslator::kSysFlag_PrimitivePolygonal;
    }
    if (prim_type == xenos::PrimitiveType::kLineList ||
        prim_type == xenos::PrimitiveType::kLineStrip) {
      flags |= DxbcShaderTranslator::kSysFlag_PrimitiveLine;
    }
    if (system_constant_guest_state.depth_float24) {
      flags |= DxbcShaderTranslator::kSysFlag_DepthFloat24;
    }
    // Alpha test: the compare function is a 3-bit (less|equal|greater) mask in
    // the flags. Leaving it 0 means kNever -> the pixel shader discards EVERY
    // fragment. When alpha test is disabled it must be kAlways (pass all).
    auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
    xenos::CompareFunction alpha_test_function =
        rb_colorcontrol.alpha_test_enable ? rb_colorcontrol.alpha_func
                                          : xenos::CompareFunction::kAlways;
    // Gamma write conversion: a k_8_8_8_8_GAMMA render target stores piecewise-
    // linear gamma; the PS must encode its linear output (matches D3D12).
    // Without this the scene was written LINEAR and every downstream pass that
    // gamma-decodes it (composite, bloom) darkened it (0.5 -> ~0.22),
    // compounding into the near-black gameplay image.
    // A/B (XE_METAL_NO_GAMMA_ENCODE=1): skip the PS linear->gamma encode on
    // gamma RTs — the exposure/history-loop stability probe (gamma decode
    // NO_SIGNS flattens the oscillation; this tests the encode leg).
    static const bool s_no_gamma_encode =
        std::getenv("XE_METAL_NO_GAMMA_ENCODE") != nullptr;
    if (!s_no_gamma_encode && color_bound) {
      flags |= GetMetalGammaRenderTargetMask(color_infos)
               << DxbcShaderTranslator::kSysFlag_ConvertColor0ToGamma_Shift;
    }
    flags |= uint32_t(alpha_test_function)
             << DxbcShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;
    system_constants.flags = flags;
    system_constants.alpha_test_reference =
        regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
    // The DXBC translator implements the exact Xenos alpha-to-mask thresholds,
    // dither offsets, and coverage output in the pixel shader. Populate its
    // dynamic ABI from the same normalized count used by the pipeline and
    // attachments; Metal hardware alpha-to-coverage stays off to avoid
    // filtering the shader-produced mask a second time.
    system_constants.sample_count_log2[0] =
        pipeline_raster_state.sample_count_log2_x;
    system_constants.sample_count_log2[1] =
        pipeline_raster_state.sample_count_log2_y;
    system_constants.alpha_to_mask = pipeline_raster_state.alpha_to_mask;
    // Texture signedness / gamma (8 bits per fetch constant): tells the PS/VS
    // to decode gamma-signed fetches (TextureSign::kGamma) to linear on sample
    // and handle signed formats. Left zeroed, every gamma texture (incl. the
    // resolved scene the composite samples) was read raw.
    // Unsigned fixed-point integer texture scaling
    // (openspec/changes/implement-unsigned-integer-texture-scaling). DEFAULT
    // OFF: with the switch off no scale is ever published, so every fetch reads
    // a zero packed value and takes the shader's uniform early-out, leaving
    // output identical to before the change. Promotion to default-on is a
    // separate evidence-backed decision.
    static const bool integer_texture_scaling_enabled =
        std::getenv("XE_METAL_INTEGER_TEXTURE_SCALE") != nullptr;
    auto fill_texture_signs = [&](MetalShader* shader) {
      if (!shader) return;
      for (const auto& binding : shader->GetTextureBindingsAfterTranslation()) {
        uint32_t fc = binding.fetch_constant;
        auto fetch = regs.GetTextureFetch(fc);
        uint32_t signs = texture_util::SwizzleSigns(fetch);
        system_constants.texture_swizzled_signs[fc >> 2] |=
            signs << ((fc & 3) * 8);
        // Integer num_format on a fixed-point format needs the guest's
        // unnormalized range restored in the shader. Derive it HERE, beside the
        // signs, from the same fetch constant.
        //
        // Deliberately not routed through shared TextureCache::RequestTextures
        // / TextureBinding the way upstream does: MetalTextureCache is a
        // standalone class, not a shared-TextureCache subclass, and shared
        // RequestTextures only ever runs for the C1T-R eligible subset. A
        // census probe placed there reported two binding classes for an entire
        // Just Cause 2 frame — a false zero that would have closed this work
        // wrongly. This loop, over GetTextureBindingsAfterTranslation, is the
        // site that sees EVERY Metal binding.
        //
        // fetch.swizzle is passed as the swizzle, matching SwizzleSigns above
        // exactly: both resolve output component i back to the same source
        // component, so the scale field for a lane always describes the same
        // component as that lane's sign. They are consumed together, in the
        // same shader, one immediately after the other.
        if (integer_texture_scaling_enabled) {
          uint32_t scale_bits = PublishIntegerScaleBits(
              fc, fetch, uint8_t(signs),
              system_constants.texture_integer_scale_bits,
              kMetalIntegerScaleSlotCount);
          // TEMPORARY acceptance probe (XE_INTEGER_SCALE_REPORT=1), removed
          // before closure: reports what is actually published per slot, and
          // the per-component multiplier the shader will form from it.
          static const bool s_scale_report =
              std::getenv("XE_INTEGER_SCALE_REPORT") != nullptr;
          if (s_scale_report) {
            uint64_t site = (uint64_t(shader->ucode_data_hash()) << 8) | fc;
            static std::set<uint64_t> s_reported;
            if (s_reported.insert(site).second) {
              uint32_t mul[4];
              for (uint32_t c = 0; c < 4; ++c) {
                uint32_t field = (scale_bits >> (c * 5)) & 0x1F;
                uint32_t width = (field & 0xF) + 1;
                uint32_t sgn = (field & 0x10) ? 1 : 0;
                mul[c] = field ? ((1u << (width - sgn)) - 1u) : 0u;
              }
              auto filter_name = [](xenos::TextureFilter f) {
                switch (f) {
                  case xenos::TextureFilter::kPoint: return "point";
                  case xenos::TextureFilter::kLinear: return "linear";
                  case xenos::TextureFilter::kBaseMap: return "basemap";
                  default: return "usefetchconst";
                }
              };
              XELOGI(
                  "IntegerScaleReport: stage={} shader={:016X} slot={} "
                  "format={} num_format={} signs={:02X} swizzle={:03X} "
                  "packed={:08X} mul={},{},{},{} mag={} min={} mip={} "
                  "aniso={} mip_min={} mip_max={} mip_addr={:X}",
                  shader->type() == xenos::ShaderType::kVertex ? "VS" : "PS",
                  shader->ucode_data_hash(), fc,
                  FormatInfo::GetName(uint32_t(fetch.format)),
                  uint32_t(fetch.num_format), signs, uint32_t(fetch.swizzle),
                  scale_bits, mul[0], mul[1], mul[2], mul[3],
                  filter_name(fetch.mag_filter), filter_name(fetch.min_filter),
                  filter_name(fetch.mip_filter), uint32_t(fetch.aniso_filter),
                  uint32_t(fetch.mip_min_level),
                  uint32_t(fetch.mip_max_level),
                  uint32_t(fetch.mip_address) << 12);
            }
          }
        }
        // Gamma-parity diagnostic: the game's signs describe the raw bytes
        // hardware resolves preserve — compare against the resolve source's
        // gamma-ness we recorded (see IssueCopy).
        static const bool s_gamma_parity = std::getenv("XE_METAL_GAMMA_PARITY");
        if (s_gamma_parity) {
          uint32_t fbase = fetch.base_address << 12;
          auto it = gamma_parity_resolves_.find(fbase);
          if (it != gamma_parity_resolves_.end()) {
            bool fetch_gamma = (signs & 0b11) == 0b11;  // component 0 kGamma
            if (fetch_gamma != it->second &&
                gamma_parity_reported_.insert(fbase).second) {
              XELOGW(
                  "MetalGammaParity MISMATCH base={:08X}: resolve source "
                  "{}GAMMA but fetch signs={:02X} ({}gamma) — byte domain "
                  "disagreement",
                  fbase, it->second ? "" : "NOT-", signs,
                  fetch_gamma ? "" : "not-");
            } else if (gamma_parity_reported_.insert(fbase ^ 0x80000000u)
                           .second) {
              XELOGI("MetalGammaParity ok base={:08X}: src_gamma={} signs={:02X}",
                     fbase, it->second, signs);
            }
          }
        }
      }
    };
    // A/B: XE_METAL_NO_SIGNS=1 leaves signs zeroed (all-unsigned, no gamma
    // decode) to isolate sign-driven fetch behavior (signed-SRV selection).
    if (!std::getenv("XE_METAL_NO_SIGNS")) {
      fill_texture_signs(vertex_shader);
      fill_texture_signs(pixel_shader);
    }
    // Color exponent bias multiplies the pixel output. Metal stores the fixed
    // 16-bit formats in SNORM host targets, which truncate to -1..1, so unlike
    // a backend configured for full-range fixed16 storage we must not apply
    // the additional /32 remap here. Doing so starves temporal luminance
    // passes and made SF4 compensate into a live white-out. The guest exponent
    // is still required for HDR formats (notably Red Dead's 7e3 chain).
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      system_constants.color_exp_bias[i] =
          std::ldexp(1.0f, int32_t(color_infos[i].color_exp_bias));
    }
    // Index/vfetch helpers used by the vertex shader. The endian here must
    // describe the index buffer AS BOUND, not the guest's: our draw paths
    // CPU-swap the guest indices into the upload arena (SwapIndicesInto /
    // the fan expansion), so the value reaching SV_VertexID is already
    // host-order. Passing the guest endianness made the translated VS swap
    // it AGAIN — every indexed draw fetched byte-swapped vertex slots
    // (terrain chunks as giant wedges; chunks whose swapped indices left
    // the lattice rasterized nothing).
    system_constants.vertex_index_endian = xenos::Endian::kNone;
    system_constants.vertex_index_offset =
        system_constant_guest_state.vertex_index_offset;
    system_constants.vertex_index_min =
        system_constant_guest_state.vertex_index_min;
    system_constants.vertex_index_max =
        system_constant_guest_state.vertex_index_max;
    // PP3-auto: only the synthetic final vertex of a routed line loop remaps to
    // guest vertex 0. memset above keeps this 0 (a shader no-op) for every
    // non-loop draw.
    if (auto_line_loop_route) {
      system_constants.line_loop_closing_index = pp_result.line_loop_closing_index;
    }

    // Diagnostic (rides XE_METAL_EXPO_TRACE=<ps hash>): log the host viewport
    // + window-offset registers for draws using that pixel shader — the
    // tonemap half-screen blocks landed at different texture rows while the
    // guest reuses the same EDRAM tiles for both halves.
    static const uint64_t s_vp_ps_hash = [] {
      const char* v = std::getenv("XE_METAL_EXPO_TRACE");
      return v ? strtoull(v, nullptr, 16) : 0ull;
    }();
    // XE_METAL_VP_RANGE=A,B additionally logs EVERY draw in that trace-draw
    // range (the window-offset placement audit over the post-tonemap chain).
    static const auto s_vp_range = []() -> std::pair<int64_t, int64_t> {
      const char* v = std::getenv("XE_METAL_VP_RANGE");
      if (!v) return {-1, -1};
      int64_t a = 0, b = 0;
      if (sscanf(v, "%lld,%lld", &a, &b) != 2) return {-1, -1};
      return {a, b};
    }();
    int64_t vp_draw_index = int64_t(trace_draw_index_) - 1;
    bool vp_log = (s_vp_ps_hash && pixel_shader &&
                   pixel_shader->ucode_data_hash() == s_vp_ps_hash) ||
                  (vp_draw_index >= s_vp_range.first &&
                   vp_draw_index <= s_vp_range.second);
    if (vp_log) {
      auto win_off = regs.Get<reg::PA_SC_WINDOW_OFFSET>();
      auto sc_mode = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
      auto clip = regs.Get<reg::PA_CL_CLIP_CNTL>();
      XELOGW(
          "MetalVP draw {}: prim={} geo={} vp=({},{} {}x{}) win_off=({},{}) "
          "vtx_wofs_en={} clip_dis={} ndc_y=(s{} o{}) ps={:08X} blend={:08X} "
          "mask={:X} imask={:X} pgen={}",
          vp_draw_index, uint32_t(prim_type), use_geometry_emulation ? 1 : 0,
          viewport_info.xy_offset[0], viewport_info.xy_offset[1],
          viewport_info.xy_extent[0], viewport_info.xy_extent[1],
          int32_t(win_off.window_x_offset), int32_t(win_off.window_y_offset),
          uint32_t(sc_mode.vtx_window_offset_enable),
          uint32_t(clip.clip_disable), viewport_info.ndc_scale[1],
          viewport_info.ndc_offset[1],
          pixel_shader ? uint32_t(pixel_shader->ucode_data_hash() >> 32) : 0,
          regs[XE_GPU_REG_RB_BLENDCONTROL0], normalized_color_mask & 0xF,
          interpolator_mask, ps_param_gen_pos);
      // The draw's pixel-shader fetches: base + format + snapshot-hit, the
      // composite-input audit.
      if (pixel_shader && render_target_cache_) {
        for (const auto& binding :
             pixel_shader->GetTextureBindingsAfterTranslation()) {
          auto f = regs.GetTextureFetch(binding.fetch_constant);
          uint32_t fetch_base = f.base_address << 12;
          XELOGW("MetalVP   fetch fc={} base={:08X}{} fmt={} {}x{} swz={:03X} "
                 "endian={} tiled={} pitch={}",
                 binding.fetch_constant, fetch_base,
                 render_target_cache_->GetResolveSnapshot(fetch_base)
                     ? " [SNAP]"
                     : "",
                 uint32_t(f.format), f.size_2d.width + 1, f.size_2d.height + 1,
                 f.swizzle & 0xFFF, uint32_t(f.endianness), uint32_t(f.tiled),
                 uint32_t(f.pitch) << 5);
        }
      }
      // Geometry-side audit (zero-raster dissection): VTE transform bits,
      // index state, and the first vertices of each VS vertex binding read
      // straight from guest RAM (byteswapped) — shows where positions land.
      XELOGW("MetalVP   vte={:08X} idx={} cnt={} vgt_ofs={} min={} max={}",
             regs[XE_GPU_REG_PA_CL_VTE_CNTL], index_buffer_info != nullptr,
             index_count, regs[XE_GPU_REG_VGT_INDX_OFFSET],
             regs[XE_GPU_REG_VGT_MIN_VTX_INDX],
             regs[XE_GPU_REG_VGT_MAX_VTX_INDX]);
      {
        const float* vs_c255 = reinterpret_cast<const float*>(
            &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + 255 * 4]);
        XELOGW("MetalVP   vs_c255=({:g} {:g} {:g} {:g})", vs_c255[0],
               vs_c255[1], vs_c255[2], vs_c255[3]);
      }
      if (vertex_shader) {
        for (const auto& vb : vertex_shader->vertex_bindings()) {
          xenos::xe_gpu_vertex_fetch_t vf =
              regs.GetVertexFetch(vb.fetch_constant);
          uint32_t vaddr = vf.address << 2;
          XELOGW("MetalVP   vbind fc={} addr={:08X} size_dw={} endian={} "
                 "stride_dw={}",
                 vb.fetch_constant, vaddr, uint32_t(vf.size),
                 uint32_t(vf.endian), vb.stride_words);
          if (vaddr && vf.size) {
            const uint32_t* vdata = reinterpret_cast<const uint32_t*>(
                memory_->TranslatePhysical(vaddr));
            uint32_t stride = vb.stride_words ? vb.stride_words : 4;
            for (uint32_t vtx = 0;
                 vtx < 4 && (vtx + 1) * stride <= vf.size; ++vtx) {
              std::string words;
              for (uint32_t w = 0; w < stride && w < 8; ++w) {
                uint32_t raw = xenos::GpuSwap(vdata[vtx * stride + w],
                                              xenos::Endian(vf.endian));
                float fv;
                std::memcpy(&fv, &raw, 4);
                words += fmt::format(" {:g}[{:08X}]", fv, raw);
              }
              XELOGW("MetalVP     v{}:{}", vtx, words);
            }
          }
        }
      }
    }

    bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);

    // Set the host viewport from the integer part. Memexport passes have a
    // 1x1 attachment-less render area and rasterization disabled — the guest
    // viewport would exceed the pass bounds (validation error) and shades
    // nothing anyway.
    if (!memexport_draw && viewport_info.xy_extent[0] &&
        viewport_info.xy_extent[1]) {
      MTLViewport vp;
      vp.originX = viewport_info.xy_offset[0];
      vp.originY = viewport_info.xy_offset[1];
      vp.width = viewport_info.xy_extent[0];
      vp.height = viewport_info.xy_extent[1];
      vp.znear = viewport_info.z_min;
      vp.zfar = viewport_info.z_max;
      [encoder setViewport:vp];
    }
    // Apply Xenos polygon culling and front-face winding per normal draw.
    // Metal encoder state persists across draws, so non-polygons explicitly
    // restore no-cull/CCW rather than inheriting the preceding triangle state.
    // Both guest faces culled is a successful raster no-op. Memexport bypasses
    // raster state because its vertex-stage shared-memory stores must execute.
    if (!memexport_draw) {
      MetalDepthClipMode depth_clip_mode = GetMetalDepthClipMode(
          regs.Get<reg::PA_CL_CLIP_CNTL>());
      [encoder setDepthClipMode:depth_clip_mode == MetalDepthClipMode::kClamp
                                    ? MTLDepthClipModeClamp
                                    : MTLDepthClipModeClip];
      MetalCullFrontFaceState face_state = GetMetalCullFrontFaceState(
          regs.Get<reg::PA_SU_SC_MODE_CNTL>(), primitive_polygonal);
      if (face_state.culls_all()) {
        return true;
      }
      MTLCullMode cull_mode = MTLCullModeNone;
      if (face_state.cull_mode == MetalCullMode::kFront) {
        cull_mode = MTLCullModeFront;
      } else if (face_state.cull_mode == MetalCullMode::kBack) {
        cull_mode = MTLCullModeBack;
      }
      [encoder setCullMode:cull_mode];
      [encoder
          setFrontFacingWinding:face_state.front_face_clockwise
                                    ? MTLWindingClockwise
                                    : MTLWindingCounterClockwise];

      // Apply a complete fill/depth-bias value every draw so state cannot leak
      // across guest transitions or newly opened render encoders. The local
      // Metal path currently renders at 1x; the helper keeps resolution scale
      // explicit for the eventual scaled-render-target path.
      float polygon_offset_scale = 0.0f;
      float polygon_offset = 0.0f;
      draw_util::GetPreferredFacePolygonOffset(
          regs, primitive_polygonal, polygon_offset_scale, polygon_offset);
      MetalFillDepthBiasState fill_bias = GetMetalFillDepthBiasState(
          regs.Get<reg::PA_SU_SC_MODE_CNTL>(), primitive_polygonal,
          regs.Get<reg::RB_DEPTH_INFO>().depth_format, polygon_offset_scale,
          polygon_offset, 1);
      [encoder
          setTriangleFillMode:fill_bias.triangle_fill_mode ==
                                      MetalTriangleFillMode::kLines
                                  ? MTLTriangleFillModeLines
                                  : MTLTriangleFillModeFill];
      [encoder setDepthBias:fill_bias.depth_bias_constant
                 slopeScale:fill_bias.depth_bias_slope
                      clamp:0.0f];
    }
    // Stage 17: apply the guest scissor on every normal raster draw. Shared
    // draw_util owns Xenos window/screen/offset/pitch semantics; the Metal-only
    // helper then intersects that result with every active attachment so the
    // MTLScissorRect is validation-safe without expanding guest coverage.
    // Memexport is intentionally excluded: its vertex-stage stores must execute
    // even though raster output is irrelevant.
    if (!memexport_draw) {
      uint32_t render_width = UINT32_MAX;
      uint32_t render_height = UINT32_MAX;
      bool have_render_extent = false;
      auto intersect_render_extent = [&](void* texture_ptr) {
        if (!texture_ptr) {
          return;
        }
        id<MTLTexture> texture = (__bridge id<MTLTexture>)texture_ptr;
        render_width = std::min(render_width, uint32_t(texture.width));
        render_height = std::min(render_height, uint32_t(texture.height));
        have_render_extent = true;
      };
      if (have_color) {
        for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
          intersect_render_extent(
              render_target_cache_->current_color_texture(i, nullptr));
        }
      }
      if (depth_bound) {
        intersect_render_extent(render_target_cache_->current_depth_texture());
      }
      if (!have_render_extent) {
        ++draw_skip_count_;
        return true;
      }
      draw_util::Scissor guest_scissor;
      draw_util::GetScissor(regs, guest_scissor);
      MetalScissorRect metal_scissor = ClampScissorToRenderExtent(
          guest_scissor, render_width, render_height);
      if (metal_scissor.empty()) {
        // Empty guest coverage is a successful no-op, not a backend failure.
        return true;
      }
      MTLScissorRect scissor_rect{metal_scissor.x, metal_scissor.y,
                                  metal_scissor.width, metal_scissor.height};
      [encoder setScissorRect:scissor_rect];
    }
    // Diagnostic (scene depth stuck at clear): log each unique depth
    // configuration of depth-bound draws once — z range, func, write, target.
    if (depth_bound) {
      uint64_t zcfg = (uint64_t(norm_depth.value) << 32) ^
                      uint64_t(viewport_info.z_min * 1000) ^
                      (uint64_t(viewport_info.z_max * 1000) << 10) ^
                      (render_target_cache_->current_attachment_key() << 20);
      static std::unordered_set<uint64_t> s_seen_zcfg;
      if (s_seen_zcfg.insert(zcfg).second && s_seen_zcfg.size() <= 24) {
        XELOGI(
            "MetalZCfg: z=({},{}) zfunc={} zwrite={} ext=({},{}) color={} "
            "clear={:08X}",
            viewport_info.z_min, viewport_info.z_max,
            uint32_t(norm_depth.zfunc), uint32_t(norm_depth.z_write_enable),
            viewport_info.xy_extent[0], viewport_info.xy_extent[1], have_color,
            regs[XE_GPU_REG_RB_DEPTH_CLEAR]);
      }
    }
    // Bring-up diagnostic: log the transform of the first few draws of the
    // session (any primitive type) to catch off-screen / degenerate output.
    if (tri_log_count_ < 12) {
      ++tri_log_count_;
      XELOGI(
          "MetalCommandProcessor: DRAW {} prim={} idx={} clipdis={} imask={:X} "
          "pgen={} ext=({},{}) z=({},{}) ndc_s=({},{},{}) ndc_o=({},{},{}) "
          "flags={:X} ps_xlat={} depth={}",
          tri_log_count_, uint32_t(prim_type), index_buffer_info != nullptr,
          uint32_t(regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable),
          interpolator_mask, ps_param_gen_pos, viewport_info.xy_extent[0],
          viewport_info.xy_extent[1], viewport_info.z_min, viewport_info.z_max,
          viewport_info.ndc_scale[0], viewport_info.ndc_scale[1],
          viewport_info.ndc_scale[2], viewport_info.ndc_offset[0],
          viewport_info.ndc_offset[1], viewport_info.ndc_offset[2],
          system_constants.flags, ps_translation != nullptr, depth_bound);
    }
  }
  if (!upload_arena_->Allocate(sizeof(system_constants), 256, &system_alloc)) {
    ++draw_skip_count_;
    return true;
  }
  std::memcpy(system_alloc.cpu_ptr, &system_constants, sizeof(system_constants));

  // Float constants, packed per the shader's float_bitmap (b1). Vertex + pixel.
  auto marshal_floats = [&](const Shader::ConstantRegisterMap& map, uint32_t reg_base,
                            MetalUploadAllocation* out) -> bool {
    uint32_t count = GetMetalPackedFloatConstantCount(map);
    uint32_t bytes = sizeof(float) * 4 * std::max(count, 1u);
    if (!upload_arena_->Allocate(bytes, 256, out)) {
      return false;
    }
    return PackMetalFloatConstants(regs, map, reg_base, out->cpu_ptr, bytes);
  };
  MetalUploadAllocation float_vertex_alloc{};
  if (!marshal_floats(vs_map, XE_GPU_REG_SHADER_CONSTANT_000_X,
                      &float_vertex_alloc)) {
    ++draw_skip_count_;
    return true;
  }
  // Forensics (with XE_METAL_DUMP_PS): log the packed VS b1 upload once —
  // bitmap words + the packed values for the HIGH constants (bitmap word 3,
  // c192-255: the terrain lighting bank) + direct register reads to compare.
  if (const char* dump_ps2 = std::getenv("XE_METAL_DUMP_PS")) {
    if (pixel_shader &&
        pixel_shader->ucode_data_hash() == strtoull(dump_ps2, nullptr, 16)) {
      static bool s_logged_b1 = false;
      if (!s_logged_b1) {
        s_logged_b1 = true;
        XELOGI("MetalB1: vs bitmap {:016X} {:016X} {:016X} {:016X}",
               vs_map.float_bitmap[0], vs_map.float_bitmap[1],
               vs_map.float_bitmap[2], vs_map.float_bitmap[3]);
        const float* packed = static_cast<const float*>(
            static_cast<const void*>(float_vertex_alloc.cpu_ptr));
        uint32_t packed_idx = 0;
        for (uint32_t i = 0; i < 4; ++i) {
          uint64_t entry = vs_map.float_bitmap[i];
          uint32_t index;
          while (xe::bit_scan_forward(entry, &index)) {
            entry = xe::clear_lowest_bit(entry);
            uint32_t creg = i * 64 + index;
            if (creg >= 224 || creg < 12) {
              const float* direct = reinterpret_cast<const float*>(
                  &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (i << 8) +
                        (index << 2)]);
              XELOGI(
                  "MetalB1: c{} packed[{}]=({}, {}, {}, {}) direct=({}, {}, "
                  "{}, {})",
                  creg, packed_idx, packed[packed_idx * 4],
                  packed[packed_idx * 4 + 1], packed[packed_idx * 4 + 2],
                  packed[packed_idx * 4 + 3], direct[0], direct[1], direct[2],
                  direct[3]);
            }
            ++packed_idx;
          }
        }
      }
    }
  }
  MetalUploadAllocation float_pixel_alloc{};
  if (pixel_shader) {
    if (!marshal_floats(pixel_shader->constant_register_map(),
                        XE_GPU_REG_SHADER_CONSTANT_256_X, &float_pixel_alloc)) {
      ++draw_skip_count_;
      return true;
    }
    // Exposure telemetry (XE_METAL_EXPO_TRACE=<ps hash hex>): capture the
    // tonemap shader's first float constants — the exposure value the game
    // computed FROM our luminance writebacks and fed back into rendering.
    static const uint64_t s_expo_ps_hash = [] {
      const char* v = std::getenv("XE_METAL_EXPO_TRACE");
      return v ? strtoull(v, nullptr, 16) : 0ull;
    }();
    if (s_expo_ps_hash && float_pixel_alloc.cpu_ptr &&
        pixel_shader->ucode_data_hash() == s_expo_ps_hash) {
      std::memcpy(expo_tonemap_c0_, float_pixel_alloc.cpu_ptr,
                  sizeof(expo_tonemap_c0_));
      expo_tonemap_seen_ = true;
      // Also log the full PACKED float-constant buffer once (the c255-class
      // weight forensics: is the last packed register marshaled right?).
      static bool s_packed_logged = false;
      if (!s_packed_logged) {
        s_packed_logged = true;
        const auto& map = pixel_shader->constant_register_map();
        uint32_t packed = 0;
        for (uint32_t i = 0; i < 4; ++i) {
          packed += xe::bit_count(map.float_bitmap[i]);
        }
        const float* fl = static_cast<const float*>(float_pixel_alloc.cpu_ptr);
        std::string line;
        for (uint32_t i = 0; i < packed; ++i) {
          line += fmt::format(" [{}]=({} {} {} {})", i, fl[i * 4], fl[i * 4 + 1],
                              fl[i * 4 + 2], fl[i * 4 + 3]);
        }
        XELOGW("MetalPackedC ps={:016X} count={}:{}",
               pixel_shader->ucode_data_hash(), packed, line);
      }
    } else if (s_expo_ps_hash && float_pixel_alloc.cpu_ptr &&
               pixel_shader->ucode_data_hash() != s_expo_ps_hash) {
      // Tonemap-hash DISCOVERY: the real ucode hash is 64-bit; if the supplied
      // XE_METAL_EXPO_TRACE didn't match any shader, dump each unique pixel
      // shader's hash + first float constants once (capped). The tonemap is the
      // full-screen draw whose constants track exposure -- set EXPO_TRACE to its
      // {:016X} hash to capture expc. grep MetalExpoCandidate.
      static std::unordered_set<uint64_t> s_seen_expo_ps;
      if (s_seen_expo_ps.size() < 64 &&
          s_seen_expo_ps.insert(pixel_shader->ucode_data_hash()).second) {
        const float* fl = static_cast<const float*>(float_pixel_alloc.cpu_ptr);
        XELOGW(
            "MetalExpoCandidate ps={:016X} c0=({:.4f},{:.4f},{:.4f},{:.4f})",
            pixel_shader->ucode_data_hash(), fl[0], fl[1], fl[2], fl[3]);
      }
    }
  }

  // Bool/loop (b2), fetch (b3) — verbatim register blocks.
  auto marshal_block = [&](uint32_t reg_base, uint32_t bytes,
                           MetalUploadAllocation* out) -> bool {
    if (!upload_arena_->Allocate(bytes, 256, out)) return false;
    std::memcpy(out->cpu_ptr, &regs[reg_base], bytes);
    return true;
  };
  MetalUploadAllocation bool_loop_alloc{};
  MetalUploadAllocation fetch_alloc{};
  if (!marshal_block(XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031, kMetalBoolLoopConstantsSize,
                     &bool_loop_alloc) ||
      !marshal_block(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0, kMetalFetchConstantsSize,
                     &fetch_alloc)) {
    ++draw_skip_count_;
    return true;
  }

  // Descriptor indices (b4): for each texture binding, resolve its guest fetch
  // constant to a bindless view-heap slot (real texture, or 0=null/gray if the
  // format isn't supported yet); for each sampler binding, a sampler-heap slot
  // built from the binding's filters (fetch-constant filters where the shader
  // says kUseFetchConst) + the fetch constant's clamp modes. Texture and
  // sampler bindings share one bindless index space (disjoint indices).
  texture_cache_->ResetActive();
  texture_cache_->lineage_draw_ = trace_draw_index_;  // lineage-probe clock
  auto marshal_desc_indices = [&](MetalShader* shader,
                                  MetalUploadAllocation* out) -> bool {
    // The cbuffer must cover every FLAT INDEX the translated shader reads:
    // [0, max bindless_descriptor_index]. The translator's index space can be
    // SPARSE (JC2's exposure shader assigns 1..6 with no 0: 4 textures + 2
    // samplers but max index 6), so sizing by list-count under-allocates and
    // the shader reads adjacent arena bytes as a descriptor-heap index — a
    // wild texture pointer and a GPU page fault (the live JC2
    // lost-command-buffer storm). Unassigned gaps read 0 = the null
    // texture/sampler slot, which is always valid.
    MetalDescriptorIndexPlan plan;
    if (shader) {
      plan = GetMetalDescriptorIndexPlan(shader->GetTextureBindingsAfterTranslation(),
                                         shader->GetSamplerBindingsAfterTranslation());
      if (!plan.valid) {
        XELOGW("MetalCommandProcessor: skipping draw with invalid {} descriptor "
               "index layout",
               shader->type() == xenos::ShaderType::kPixel ? "pixel" : "vertex");
        return false;
      }
    }
    const uint32_t count = plan.count;
    if (!upload_arena_->Allocate(count * sizeof(uint32_t), 256, out)) {
      return false;
    }
    auto* indices = static_cast<uint32_t*>(out->cpu_ptr);
    if (!InitializeMetalDescriptorIndices(plan, indices, count * sizeof(uint32_t))) {
      return false;
    }
    if (shader) {
      // XE_METAL_TEX_PROBE=<draw>: log every texture binding's fetch fields +
      // resolved bindless slot at that trace draw (slot 0 = null texture ->
      // the shader samples zero; the light-draw zero-output suspect).
      static const int64_t s_tex_probe = [] {
        const char* v = std::getenv("XE_METAL_TEX_PROBE");
        return v ? strtoll(v, nullptr, 10) : -1;
      }();
      bool probe = s_tex_probe >= 0 &&
                   int64_t(trace_draw_index_) - 1 == s_tex_probe;
      for (const auto& binding : shader->GetTextureBindingsAfterTranslation()) {
        if (binding.bindless_descriptor_index < count) {
          auto fetch = regs.GetTextureFetch(binding.fetch_constant);
          // C1T-R serve: indices the prepass prepared come from the shared
          // subclass; anything unserved (prep failure, invalidated between
          // prep and serve, faulted load) falls to the legacy path
          // unchanged.
          uint32_t routed_slot = 0;
          if (!binding.is_signed && shared_texture_cache_ &&
              (draw_shared_route_mask_ & (1u << binding.fetch_constant))) {
            routed_slot =
                shared_texture_cache_->GetServedSlot(binding.fetch_constant);
          }
          uint32_t texture_slot = routed_slot;
          if (!texture_slot) {
            // A failed shared-cache candidate may still have a correct legacy
            // fallback. Only propagate failure if that fallback also can't
            // create the required object or heap slot.
            texture_cache_->ClearBindingAllocationFailure();
            texture_slot = texture_cache_->GetTextureSlot(fetch, binding.is_signed);
            if (texture_cache_->binding_allocation_failed()) {
              XELOGW("MetalCommandProcessor: skipping draw after texture "
                     "binding allocation failure");
              return false;
            }
          }
          indices[binding.bindless_descriptor_index] = texture_slot;
          if (probe) {
            XELOGI(
                "MetalTexProbe draw={} {} fc={} descidx={} -> slot={} "
                "type={} fmt={} base={:08X} size={}x{} endian={} tiled={} "
                "stacked={} packedmips={} miplevels={}-{}",
                s_tex_probe, shader->type() == xenos::ShaderType::kPixel ? "PS" : "VS",
                binding.fetch_constant, binding.bindless_descriptor_index,
                indices[binding.bindless_descriptor_index],
                uint32_t(fetch.dimension), uint32_t(fetch.format),
                fetch.base_address << 12, fetch.size_2d.width + 1,
                fetch.size_2d.height + 1, uint32_t(fetch.endianness),
                uint32_t(fetch.tiled), uint32_t(fetch.stacked),
                uint32_t(fetch.packed_mips), fetch.mip_min_level,
                fetch.mip_max_level);
          }
        }
      }
      for (const auto& sb : shader->GetSamplerBindingsAfterTranslation()) {
        if (sb.bindless_descriptor_index >= count) {
          continue;
        }
        auto f = regs.GetTextureFetch(sb.fetch_constant);
        auto resolve_filter = [](xenos::TextureFilter binding_filter,
                                 uint32_t fetch_filter) {
          return binding_filter == xenos::TextureFilter::kUseFetchConst
                     ? fetch_filter
                     : uint32_t(binding_filter);
        };
        uint32_t mag = resolve_filter(sb.mag_filter, uint32_t(f.mag_filter));
        uint32_t mn = resolve_filter(sb.min_filter, uint32_t(f.min_filter));
        uint32_t mip = resolve_filter(sb.mip_filter, uint32_t(f.mip_filter));
        uint32_t an = sb.aniso_filter == xenos::AnisoFilter::kUseFetchConst
                          ? uint32_t(f.aniso_filter)
                          : uint32_t(sb.aniso_filter);
        uint32_t skey = (mag & 3) | ((mn & 3) << 2) | ((mip & 3) << 4) |
                        ((uint32_t(f.clamp_x) & 7) << 6) |
                        ((uint32_t(f.clamp_y) & 7) << 9) |
                        ((uint32_t(f.clamp_z) & 7) << 12) | ((an & 7) << 15);
        uint32_t sampler_slot;
        if (!GetOrCreateSamplerSlot(skey, &sampler_slot)) {
          XELOGW("MetalCommandProcessor: skipping draw after sampler binding "
                 "allocation failure");
          return false;
        }
        indices[sb.bindless_descriptor_index] = sampler_slot;
      }
    }
    return true;
  };
  MetalUploadAllocation desc_idx_vertex_alloc{};
  MetalUploadAllocation desc_idx_pixel_alloc{};
  if (!marshal_desc_indices(vertex_shader, &desc_idx_vertex_alloc) ||
      !marshal_desc_indices(pixel_shader, &desc_idx_pixel_alloc)) {
    ++draw_skip_count_;
    return true;
  }

  // --- Build the top-level argument buffer from the root-layout authority. ---
  id<MTLBuffer> view_heap = (__bridge id<MTLBuffer>)view_bindless_heap_;
  id<MTLBuffer> sampler_heap = (__bridge id<MTLBuffer>)sampler_bindless_heap_;
  id<MTLBuffer> system_view = (__bridge id<MTLBuffer>)system_view_tables_;
  id<MTLBuffer> null_buffer = (__bridge id<MTLBuffer>)null_buffer_;
  uint64_t view_heap_gpu = [view_heap gpuAddress];
  uint64_t sampler_heap_gpu = [sampler_heap gpuAddress];
  uint64_t system_view_gpu = [system_view gpuAddress];
  uint64_t null_gpu = [null_buffer gpuAddress];

  MetalUploadAllocation ab_alloc{};
  if (!upload_arena_->Allocate(kMetalBindingSlotCount * sizeof(uint64_t), 256, &ab_alloc)) {
    ++draw_skip_count_;
    return true;
  }
  MetalBindingAddresses binding_addresses;
  binding_addresses.null_buffer = null_gpu;
  binding_addresses.view_heap = view_heap_gpu;
  binding_addresses.sampler_heap = sampler_heap_gpu;
  binding_addresses.shared_memory_srv =
      system_view_gpu + kSystemViewSharedMemorySRV * sizeof(IRDescriptorTableEntry);
  binding_addresses.null_uav =
      system_view_gpu + kSystemViewNullUAV * sizeof(IRDescriptorTableEntry);
  binding_addresses.shared_memory_uav =
      system_view_gpu + kSystemViewSharedMemoryUAV * sizeof(IRDescriptorTableEntry);
  binding_addresses.system_constants = system_alloc.gpu_address;
  binding_addresses.float_vertex = float_vertex_alloc.gpu_address;
  binding_addresses.bool_loop = bool_loop_alloc.gpu_address;
  binding_addresses.fetch = fetch_alloc.gpu_address;
  binding_addresses.descriptor_indices_vertex = desc_idx_vertex_alloc.gpu_address;
  binding_addresses.float_pixel = float_pixel_alloc.gpu_address;
  binding_addresses.descriptor_indices_pixel = desc_idx_pixel_alloc.gpu_address;
  std::array<uint64_t, kMetalBindingSlotCount> argument_buffer =
      BuildMetalTopLevelArgumentBuffer(binding_addresses, pixel_shader != nullptr, memexport_draw);
  std::memcpy(ab_alloc.cpu_ptr, argument_buffer.data(), sizeof(argument_buffer));

  // --- Bind. ---
  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pipeline];

  // Apply complete blend/stencil dynamic values on every normal draw. Metal
  // encoder state persists across draws, while a new encoder begins with its
  // own defaults, so even blend colors that a particular pipeline doesn't
  // consume are established rather than assumed. Memexport has no raster
  // output and stays outside this state. The helper also prevents raw
  // back-face stencil state from leaking into non-polygon primitives.
  reg::RB_STENCILREFMASK stencil_ref_mask =
      regs.Get<reg::RB_STENCILREFMASK>();
  reg::RB_STENCILREFMASK stencil_ref_mask_back =
      regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  MetalBlendStencilState blend_stencil = GetMetalBlendStencilState(
      regs.Get<float>(XE_GPU_REG_RB_BLEND_RED),
      regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
      regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE),
      regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA), norm_depth,
      draw_util::IsPrimitivePolygonal(regs), stencil_ref_mask,
      stencil_ref_mask_back);
  if (!memexport_draw) {
    [encoder setBlendColorRed:blend_stencil.blend_color_red
                        green:blend_stencil.blend_color_green
                         blue:blend_stencil.blend_color_blue
                        alpha:blend_stencil.blend_color_alpha];
  }
  if (kDepthEnabled && depth_bound) {
    // Only when the pass has a depth attachment — setting a depth-test state on
    // an encoder without one is a Metal validation error. norm_depth already
    // collapsed "always, no write" to disabled.
    void* depth_state = GetOrCreateDepthStencilState(
        norm_depth.value, stencil_ref_mask.value, stencil_ref_mask_back.value);
    if (depth_state) {
      [encoder
          setDepthStencilState:(__bridge id<MTLDepthStencilState>)depth_state];
      if (blend_stencil.stencil_enabled) {
        [encoder
            setStencilFrontReferenceValue:
                blend_stencil.stencil_reference_front
                      backReferenceValue:
                          blend_stencil.stencil_reference_back];
      }
    }
  }
  if (kDebugSolidFragment) {
    // Per-vertex-shader color so distinct draw types show as distinct regions:
    // many colors = real structured scene geometry, one flat color = a single
    // fullscreen quad covering everything.
    uintptr_t p = reinterpret_cast<uintptr_t>(vs_translation->metal_function());
    uint32_t h = uint32_t(p * 2654435761u);
    float debug_color[4] = {0.2f + 0.8f * ((h & 0xFF) / 255.0f),
                            0.2f + 0.8f * (((h >> 8) & 0xFF) / 255.0f),
                            0.2f + 0.8f * (((h >> 16) & 0xFF) / 255.0f), 1.0f};
    [encoder setFragmentBytes:debug_color length:sizeof(debug_color) atIndex:8];
  }
  id<MTLBuffer> ab_buffer = (__bridge id<MTLBuffer>)ab_alloc.buffer;
  // Geometry emulation runs the guest VS as the OBJECT stage and the generated GS
  // as the MESH stage, so the argument buffer + bindless heaps bind to
  // object/mesh instead of vertex. The fragment stage is unchanged.
  auto bind_arg_buffer = [&](id<MTLBuffer> buf, NSUInteger off, NSUInteger idx) {
    if (use_geometry_emulation) {
      [encoder setObjectBuffer:buf offset:off atIndex:idx];
      [encoder setMeshBuffer:buf offset:off atIndex:idx];
    } else {
      [encoder setVertexBuffer:buf offset:off atIndex:idx];
    }
    [encoder setFragmentBuffer:buf offset:off atIndex:idx];
  };
  bind_arg_buffer(ab_buffer, ab_alloc.offset, kIRArgumentBufferBindPoint);
  bind_arg_buffer(view_heap, 0, kIRDescriptorHeapBindPoint);
  bind_arg_buffer(sampler_heap, 0, kIRSamplerHeapBindPoint);
  if (use_geometry_emulation) {
    // MSC manual: the object stage expects an IRRuntimeVertexBuffers table at
    // kIRVertexBufferBindPoint. Our guest VS fetches in-shader from shared memory
    // (zero stage-in inputs), so a zeroed table satisfies the binding without
    // supplying any real vertex streams.
    IRRuntimeVertexBuffers vertex_buffers = {};
    [encoder setObjectBytes:vertex_buffers
                     length:sizeof(vertex_buffers)
                    atIndex:kIRVertexBufferBindPoint];
  }

  // Residency for everything the shaders reach through gpuAddresses.
  id<MTLBuffer> shared_buffer =
      (__bridge id<MTLBuffer>)shared_memory_->GetBuffer();
  id<MTLTexture> null_texture = (__bridge id<MTLTexture>)null_texture_;
  uint32_t binding_stage_mask = GetMetalBindingStageMask(use_geometry_emulation);
  MTLRenderStages stages = 0;
  if (binding_stage_mask & kMetalBindingStageVertex) {
    stages |= MTLRenderStageVertex;
  }
  if (binding_stage_mask & kMetalBindingStageObject) {
    stages |= MTLRenderStageObject;
  }
  if (binding_stage_mask & kMetalBindingStageMesh) {
    stages |= MTLRenderStageMesh;
  }
  if (binding_stage_mask & kMetalBindingStageFragment) {
    stages |= MTLRenderStageFragment;
  }
  [encoder useResource:shared_buffer
                 usage:memexport_draw
                           ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                           : MTLResourceUsageRead
                stages:stages];
  [encoder useResource:ab_buffer usage:MTLResourceUsageRead stages:stages];
  [encoder useResource:view_heap usage:MTLResourceUsageRead stages:stages];
  [encoder useResource:sampler_heap usage:MTLResourceUsageRead stages:stages];
  [encoder useResource:system_view usage:MTLResourceUsageRead stages:stages];
  [encoder useResource:null_buffer usage:MTLResourceUsageRead stages:stages];
  [encoder useResource:null_texture usage:MTLResourceUsageRead stages:stages];
  for (void* tex : texture_cache_->active_textures()) {
    [encoder useResource:(__bridge id<MTLTexture>)tex
                   usage:MTLResourceUsageRead
                  stages:stages];
  }

  auto mark_raster_targets_written = [&] {
    render_target_cache_->MarkCurrentRasterTargetsWritten(
        normalized_color_mask,
        norm_depth.z_write_enable || norm_depth.stencil_enable);
  };

  // --- Draw. ---
  // Rectangle/point/quad geometry emulation: the guest VS runs
  // as the object stage and a generated GS as the mesh stage; the runtime
  // dispatches the object+mesh threadgroups. rect-list maps to a triangle input
  // topology (the GS synthesizes the 4th corner). See openspec
  // reduce-apple-gpu-frametime handoff.
  if (use_geometry_emulation) {
    IRRuntimeGeometryPipelineConfig geometry_config = {};
    geometry_config.gsVertexSizeInBytes = gs_vertex_size;
    geometry_config.gsMaxInputPrimitivesPerMeshThreadgroup =
        gs_max_input_primitives;
    // The runtime primitive matches the GS input topology: rectangle lists
    // feed triangles, point sprites feed points, and quad lists feed the
    // line-with-adjacency geometry shader through the line primitive family.
    IRRuntimePrimitiveType geometry_primitive;
    switch (geometry_type) {
      case PipelineGeometryShader::kPointList:
        geometry_primitive = IRRuntimePrimitiveTypePoint;
        break;
      case PipelineGeometryShader::kQuadList:
        // The generated geometry shader declares line-with-adjacency. Tell
        // the runtime the full four-vertex primitive type too — passing the
        // ordinary two-vertex line type makes its object stage assemble pairs
        // while the mesh stage consumes groups of four, joining unrelated
        // quad vertices into long wedges (Red Dead's foliage draw 705).
        geometry_primitive = IRRuntimePrimitiveTypeLineWithAdj;
        break;
      case PipelineGeometryShader::kRectangleList:
      default:
        geometry_primitive = IRRuntimePrimitiveTypeTriangle;
        break;
    }
    if (index_buffer_info && index_buffer_info->guest_base) {
      uint32_t index_size =
          index_buffer_info->format == xenos::IndexFormat::kInt32 ? 4 : 2;
      uint32_t total = index_count * index_size;
      const void* guest_indices =
          memory_->TranslatePhysical(index_buffer_info->guest_base);
      MetalUploadAllocation index_alloc{};
      if (!guest_indices || !upload_arena_->Allocate(total, 16, &index_alloc)) {
        ++draw_skip_count_;
        return true;
      }
      SwapIndicesInto(index_alloc.cpu_ptr, guest_indices, index_count,
                      index_buffer_info->format, index_buffer_info->endianness);
      id<MTLBuffer> index_buffer = (__bridge id<MTLBuffer>)index_alloc.buffer;
      [encoder useResource:index_buffer usage:MTLResourceUsageRead stages:stages];
      MTLIndexType index_type =
          index_size == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
      // The runtime reads the index buffer by gpuAddress; our data lives at the
      // arena byte offset, expressed here as an element start index.
      uint32_t start_index = uint32_t(index_alloc.offset / index_size);
      IRRuntimeDrawIndexedPrimitivesGeometryEmulation(
          encoder, geometry_primitive, index_type, index_buffer,
          geometry_config, /*instanceCount=*/1, index_count, start_index,
          /*baseVertex=*/0, /*baseInstance=*/0);
    } else {
      IRRuntimeDrawPrimitivesGeometryEmulation(
          encoder, geometry_primitive, geometry_config,
          /*instanceCount=*/1, index_count, /*baseVertex=*/0,
          /*baseInstance=*/0);
    }
    mark_raster_targets_written();
    frame_has_render_ = true;
    ++draw_count_;
    ++geo_draw_count_;
    return true;
  }

  // Triangle fans have no host primitive on Metal: expand to a triangle list
  // with a generated 32-bit index buffer. Fan (v0, v1, v2, v3, ...) becomes
  // triangles (v0,v1,v2), (v0,v2,v3), ... indexing the same vertices, so it
  // works whether the guest draw is indexed or sequential (auto).
  if (prim_type == xenos::PrimitiveType::kTriangleFan && !pp_routed) {
    if (index_count < 3) {
      ++draw_skip_count_;
      return true;
    }
    uint32_t host_index_count = (index_count - 2) * 3;
    MetalUploadAllocation fan_alloc{};
    if (!upload_arena_->Allocate(host_index_count * sizeof(uint32_t), 16,
                                 &fan_alloc)) {
      ++draw_skip_count_;
      return true;
    }
    // Source vertex indices: the (endian-swapped) guest index buffer, or the
    // sequential 0..N-1 for an auto draw.
    std::vector<uint32_t> src(index_count);
    if (index_buffer_info && index_buffer_info->guest_base) {
      const void* gi = memory_->TranslatePhysical(index_buffer_info->guest_base);
      if (!gi) {
        ++draw_skip_count_;
        return true;
      }
      bool i32 = index_buffer_info->format == xenos::IndexFormat::kInt32;
      xenos::Endian ie = index_buffer_info->endianness;
      for (uint32_t i = 0; i < index_count; ++i) {
        if (i32) {
          uint32_t v = static_cast<const uint32_t*>(gi)[i];
          src[i] = ie == xenos::Endian::kNone ? v : xe::byte_swap(v);
        } else {
          uint16_t v = static_cast<const uint16_t*>(gi)[i];
          bool swap = ie == xenos::Endian::k8in16 || ie == xenos::Endian::k8in32;
          src[i] = swap ? xe::byte_swap(v) : v;
        }
      }
    } else {
      for (uint32_t i = 0; i < index_count; ++i) {
        src[i] = i;
      }
    }
    uint32_t* host_idx = static_cast<uint32_t*>(fan_alloc.cpu_ptr);
    uint32_t o = 0;
    for (uint32_t i = 1; i + 1 < index_count; ++i) {
      host_idx[o++] = src[0];
      host_idx[o++] = src[i];
      host_idx[o++] = src[i + 1];
    }
    id<MTLBuffer> fan_ib = (__bridge id<MTLBuffer>)fan_alloc.buffer;
    [encoder useResource:fan_ib usage:MTLResourceUsageRead stages:stages];
    IRRuntimeDrawIndexedPrimitives(encoder, MTLPrimitiveTypeTriangle,
                                   host_index_count, MTLIndexTypeUInt32, fan_ib,
                                   fan_alloc.offset, /*instanceCount=*/1,
                                   /*baseVertex=*/0, /*baseInstance=*/0);
    mark_raster_targets_written();
    frame_has_render_ = true;
    ++draw_count_;
    return true;
  }
  if (pp_routed) {
    // Routed draw: the shared PrimitiveProcessor is the normalization authority.
    // Bind/draw from pp_result. host_primitive_type is either one of the four
    // PP1/PP4 native classes (host == guest, non-tessellated), kTriangleList
    // for a PP2-auto fan, or kLineStrip for a PP3-auto loop, so
    // MapPrimitiveType always succeeds. The index MECHANISM is unchanged:
    // PP1 and PP4 kGuestDMA CPU-byte-swap the guest indices into the arena
    // using the NORMALIZED host_index_format / host_shader_index_endian
    // (byte-identical to the inline SwapIndicesInto); PP2-auto copies the
    // built-in fan-table slice into the arena (below).
    MTLPrimitiveType pp_prim = MTLPrimitiveTypeTriangle;
    MapPrimitiveType(pp_result.host_primitive_type, &pp_prim);
    uint32_t host_count = pp_result.host_draw_vertex_count;
    if (pp_result.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA) {
      uint32_t index_size = pp_result.host_index_format == xenos::IndexFormat::kInt32 ? 4 : 2;
      uint32_t total = host_count * index_size;
      const void* guest_indices = memory_->TranslatePhysical(pp_result.guest_index_base);
      MetalUploadAllocation index_alloc{};
      if (!guest_indices || !upload_arena_->Allocate(total, 16, &index_alloc)) {
        ++draw_skip_count_;
        return true;
      }
      SwapIndicesInto(index_alloc.cpu_ptr, guest_indices, host_count, pp_result.host_index_format,
                      pp_result.host_shader_index_endian);
      id<MTLBuffer> index_buffer = (__bridge id<MTLBuffer>)index_alloc.buffer;
      [encoder useResource:index_buffer usage:MTLResourceUsageRead stages:stages];
      MTLIndexType index_type = index_size == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
      IRRuntimeDrawIndexedPrimitives(encoder, pp_prim, host_count, index_type, index_buffer,
                                     index_alloc.offset,
                                     /*instanceCount=*/1, /*baseVertex=*/0,
                                     /*baseInstance=*/0);
    } else if (pp_result.index_buffer_type ==
               PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
      // PP1 auto pass-through or PP3-auto line loop: no host index byte stream.
      IRRuntimeDrawPrimitives(encoder, pp_prim, /*vertexStart=*/uint64_t(0), uint64_t(host_count));
    } else if (pp_result.index_buffer_type ==
               PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto) {
      // PP2-auto triangle fan: the shared built-in 16-bit fan table expands the
      // auto-indexed fan to a triangle list in center-first order (byte- and
      // order-identical to the retired inline auto-fan expansion). Materialize
      // the needed slice into the transient upload arena (same lifetime as the
      // inline path) and draw UInt16. count<3 -> host_count 0 -> no-draw skip,
      // matching the inline path exactly.
      if (host_count == 0) {
        ++draw_skip_count_;
        return true;
      }
      uint32_t total = host_count * uint32_t(sizeof(uint16_t));
      MetalUploadAllocation index_alloc{};
      if (!upload_arena_->Allocate(total, 16, &index_alloc)) {
        ++draw_skip_count_;
        return true;
      }
      if (!primitive_processor_->CopyBuiltinIndexData(
              pp_result.host_index_buffer_handle, total, index_alloc.cpu_ptr)) {
        // Out-of-bounds slice (never expected: num_indices is 16-bit and the
        // table covers UINT16_MAX). Fail closed to a skip rather than read past.
        ++draw_skip_count_;
        return true;
      }
      id<MTLBuffer> index_buffer = (__bridge id<MTLBuffer>)index_alloc.buffer;
      [encoder useResource:index_buffer usage:MTLResourceUsageRead stages:stages];
      IRRuntimeDrawIndexedPrimitives(encoder, pp_prim, host_count,
                                     MTLIndexTypeUInt16, index_buffer,
                                     index_alloc.offset, /*instanceCount=*/1,
                                     /*baseVertex=*/0, /*baseInstance=*/0);
    } else {
      // Unreachable: the pre-encoder shape guards only set pp_routed for
      // kGuestDMA / kNone (PP1/PP3/PP4) or kHostBuiltinForAuto (PP2-auto).
      // Never
      // silently fall into an auto draw for an unexpected index-buffer type.
      XELOGE("MetalPrimitiveProcessor: routed draw reached emission with "
             "unexpected index_buffer_type {}; skipping.",
             uint32_t(pp_result.index_buffer_type));
      assert_always();
      ++draw_skip_count_;
      return true;
    }
  } else if (index_buffer_info && index_buffer_info->guest_base) {
    uint32_t index_size =
        index_buffer_info->format == xenos::IndexFormat::kInt32 ? 4 : 2;
    uint32_t total = index_count * index_size;
    const void* guest_indices =
        memory_->TranslatePhysical(index_buffer_info->guest_base);
    MetalUploadAllocation index_alloc{};
    if (!guest_indices ||
        !upload_arena_->Allocate(total, 16, &index_alloc)) {
      ++draw_skip_count_;
      return true;
    }
    SwapIndicesInto(index_alloc.cpu_ptr, guest_indices, index_count,
                    index_buffer_info->format, index_buffer_info->endianness);
    id<MTLBuffer> index_buffer = (__bridge id<MTLBuffer>)index_alloc.buffer;
    [encoder useResource:index_buffer usage:MTLResourceUsageRead stages:stages];
    MTLIndexType index_type = index_size == 4 ? MTLIndexTypeUInt32
                                              : MTLIndexTypeUInt16;
    IRRuntimeDrawIndexedPrimitives(encoder, mtl_prim, index_count, index_type,
                                   index_buffer, index_alloc.offset,
                                   /*instanceCount=*/1, /*baseVertex=*/0,
                                   /*baseInstance=*/0);
  } else {
    IRRuntimeDrawPrimitives(encoder, mtl_prim, /*vertexStart=*/uint64_t(0),
                            uint64_t(index_count));
  }

  mark_raster_targets_written();

  if (memexport_draw) {
    // The exports now exist (GPU-ordered) in the shared-memory buffer. Mark
    // the ranges GPU-written (watches + stale-texture invalidation, and so
    // later RequestRange calls don't upload stale CPU RAM over them) and
    // queue the RAM drain for when this submission completes. Routed through
    // the typed task-3.3 publication seam: RangeWrittenByGpu and the pending-
    // record registration form ONE ordered transition. The seam preflights the
    // marker, registers ownership FIRST, then marks the same normalized range
    // GPU-written, so no interval is GPU-newer before it has an owner.
    uint64_t export_submission = S7TagsOff() ? 0 : CurrentEncodingSubmission();
    MetalMemexportPublicationTarget publication_target(
        shared_memory_.get());
    for (const auto& range : memexport_ranges) {
      uint32_t range_base = range.base_address_dwords << 2;
      auto publication = metal::DecideMemexportPublication(
          range_base, range.size_bytes, export_submission);
      metal::MemexportPublicationResult publish_result;
      if (publication) {
        publish_result = metal::PublishMemexportWrite(
            *publication, publication_target,
            [this](const metal::MemexportPublication& owned) {
              pending_memexports_.push_back(PendingMemexport{
                  owned.range.start(), owned.range.length(), owned.submission,
                  ++memory_authority_sequence_});
              return true;
            });
      }
      if (!publish_result.marked) {
        XELOGE(
            "MetalMemexport: failed to publish GPU write {:08X}(+{}); "
            "no GPU-written notification was issued",
            range_base, range.size_bytes);
      }
    }
  }

  frame_has_render_ = true;
  ++draw_count_;
  ++draws_since_commit_;
  if (draw_count_ == 1 || (draw_count_ % 2000) == 0) {
    XELOGI("MetalCommandProcessor: encoded draw #{} (prim {}, {} indices, {})",
           draw_count_, uint32_t(prim_type), index_count,
           index_buffer_info ? "indexed" : "auto");
  }
  // Diagnostic: XE_METAL_DUMP_DEPTH=<trace draw index> dumps the bound depth
  // texture (D32F) to /tmp/xe_depth_<index>.ppm as sqrt-toned grayscale right
  // after that draw encodes — splits "depth VALUES wrong" from "the d2c
  // reinterpret scrambles correct values" (smooth image = values fine).
  static const int64_t s_dump_depth_at = [] {
    const char* v = std::getenv("XE_METAL_DUMP_DEPTH");
    return v ? strtoll(v, nullptr, 10) : -1ll;
  }();
  if (s_dump_depth_at >= 0 && int64_t(trace_draw_index_) - 1 == s_dump_depth_at) {
    EndFrameEncoderAndWait();
    void* dt_ptr = render_target_cache_->current_depth_texture();
    if (dt_ptr) {
      id<MTLTexture> dt = (__bridge id<MTLTexture>)dt_ptr;
      uint32_t tw = uint32_t(dt.width), th = uint32_t(dt.height);
      id<MTLBuffer> rb =
          [dt.device newBufferWithLength:size_t(tw) * th * 4
                                 options:MTLResourceStorageModeShared];
      id<MTLCommandQueue> dq = (__bridge id<MTLCommandQueue>)
          static_cast<xe::ui::metal::MetalProvider*>(
              graphics_system_->provider())->command_queue();
      if (rb && dq) {
        id<MTLCommandBuffer> dcb = [dq commandBuffer];
        id<MTLBlitCommandEncoder> dbl = [dcb blitCommandEncoder];
        [dbl copyFromTexture:dt
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(tw, th, 1)
                     toBuffer:rb
            destinationOffset:0
       destinationBytesPerRow:tw * 4
     destinationBytesPerImage:size_t(tw) * th * 4];
        [dbl endEncoding];
        [dcb commit];
        [dcb waitUntilCompleted];
        std::string path = fmt::format("/tmp/xe_depth_{}.ppm", s_dump_depth_at);
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
          fprintf(f, "P6\n%u %u\n255\n", tw, th);
          const float* px = static_cast<const float*>(rb.contents);
          for (size_t i = 0; i < size_t(tw) * th; ++i) {
            float v = px[i];
            v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
            uint8_t g = uint8_t(std::sqrt(v) * 255.f + 0.5f);
            uint8_t rgb[3] = {g, g, g};
            fwrite(rgb, 1, 3, f);
          }
          fclose(f);
          XELOGW("XE_METAL_DUMP_DEPTH: wrote {} ({}x{})", path, tw, th);
        }
      }
    }
  }
  // Diagnostic: XE_METAL_DUMP_RT=<trace draw index> dumps the bound color-0
  // RT texture to /tmp/xe_rt_<index>.ppm right after that draw encodes —
  // TEXTURE-space ground truth vs the EDRAM-mapped view of the capture
  // pipeline (splits "content mislaid in the texture" from "the dump/EDRAM
  // mapping mislays it").
  static const int64_t s_dump_rt_at = [] {
    const char* v = std::getenv("XE_METAL_DUMP_RT");
    return v ? strtoll(v, nullptr, 10) : -1ll;
  }();
  if (s_dump_rt_at >= 0 && int64_t(trace_draw_index_) - 1 == s_dump_rt_at) {
    EndFrameEncoderAndWait();
    void* rt_ptr = render_target_cache_->current_color_texture(0, nullptr);
    if (rt_ptr) {
      id<MTLTexture> rt_tex = (__bridge id<MTLTexture>)rt_ptr;
      uint32_t tw = uint32_t(rt_tex.width), th = uint32_t(rt_tex.height);
      id<MTLBuffer> rb =
          [rt_tex.device newBufferWithLength:size_t(tw) * th * 4
                                     options:MTLResourceStorageModeShared];
      id<MTLCommandQueue> dq = (__bridge id<MTLCommandQueue>)
          static_cast<xe::ui::metal::MetalProvider*>(
              graphics_system_->provider())->command_queue();
      if (rb && dq) {
        id<MTLCommandBuffer> dcb = [dq commandBuffer];
        id<MTLBlitCommandEncoder> dbl = [dcb blitCommandEncoder];
        [dbl copyFromTexture:rt_tex
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(tw, th, 1)
                     toBuffer:rb
            destinationOffset:0
       destinationBytesPerRow:tw * 4
     destinationBytesPerImage:size_t(tw) * th * 4];
        [dbl endEncoding];
        [dcb commit];
        [dcb waitUntilCompleted];
        std::string path = fmt::format("/tmp/xe_rt_{}.ppm", s_dump_rt_at);
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
          fprintf(f, "P6\n%u %u\n255\n", tw, th);
          const uint8_t* px = static_cast<const uint8_t*>(rb.contents);
          uint64_t a_sum = 0;
          uint8_t a_min = 255, a_max = 0;
          for (size_t i = 0; i < size_t(tw) * th; ++i) {
            fwrite(px + i * 4, 1, 3, f);  // RGBA8 storage: bytes are R,G,B.
            uint8_t a = px[i * 4 + 3];
            a_sum += a;
            a_min = std::min(a_min, a);
            a_max = std::max(a_max, a);
          }
          fclose(f);
          XELOGW("XE_METAL_DUMP_RT: wrote {} ({}x{}) alpha min={} max={} mean={:.2f}",
                 path, tw, th, a_min, a_max,
                 double(a_sum) / (double(tw) * th));
        }
      }
    }
  }
  return true;
}

static MTLCompareFunction ToMetalCompareFunction(xenos::CompareFunction f) {
  switch (f) {
    case xenos::CompareFunction::kNever:
      return MTLCompareFunctionNever;
    case xenos::CompareFunction::kLess:
      return MTLCompareFunctionLess;
    case xenos::CompareFunction::kEqual:
      return MTLCompareFunctionEqual;
    case xenos::CompareFunction::kLessEqual:
      return MTLCompareFunctionLessEqual;
    case xenos::CompareFunction::kGreater:
      return MTLCompareFunctionGreater;
    case xenos::CompareFunction::kNotEqual:
      return MTLCompareFunctionNotEqual;
    case xenos::CompareFunction::kGreaterEqual:
      return MTLCompareFunctionGreaterEqual;
    default:
      return MTLCompareFunctionAlways;
  }
}

static MTLStencilOperation ToMetalStencilOperation(xenos::StencilOp op) {
  switch (op) {
    case xenos::StencilOp::kKeep:
      return MTLStencilOperationKeep;
    case xenos::StencilOp::kZero:
      return MTLStencilOperationZero;
    case xenos::StencilOp::kReplace:
      return MTLStencilOperationReplace;
    case xenos::StencilOp::kIncrementClamp:
      return MTLStencilOperationIncrementClamp;
    case xenos::StencilOp::kDecrementClamp:
      return MTLStencilOperationDecrementClamp;
    case xenos::StencilOp::kInvert:
      return MTLStencilOperationInvert;
    case xenos::StencilOp::kIncrementWrap:
      return MTLStencilOperationIncrementWrap;
    case xenos::StencilOp::kDecrementWrap:
      return MTLStencilOperationDecrementWrap;
    default:
      return MTLStencilOperationKeep;
  }
}

// Builds (and caches) an MTLDepthStencilState from RB_DEPTHCONTROL plus the
// read/write masks of RB_STENCILREFMASK(_BF). The stencil REFERENCE is encoder
// state (setStencilReferenceValue at the draw), not part of this object. When
// depth test is disabled the state passes everything and writes nothing.
void* MetalCommandProcessor::GetOrCreateDepthStencilState(
    uint32_t depth_control, uint32_t stencil_ref_mask,
    uint32_t stencil_ref_mask_bf) {
  reg::RB_DEPTHCONTROL dc;
  dc.value = depth_control;
  reg::RB_STENCILREFMASK srm;
  srm.value = stencil_ref_mask;
  reg::RB_STENCILREFMASK srm_bf;
  srm_bf.value = dc.backface_enable ? stencil_ref_mask_bf : stencil_ref_mask;
  uint32_t masks = 0;
  if (dc.stencil_enable) {
    masks = (srm.stencilmask) | (srm.stencilwritemask << 8) |
            (srm_bf.stencilmask << 16) | (srm_bf.stencilwritemask << 24);
  }
  // Depth bits + all stencil fields of RB_DEPTHCONTROL + the masks.
  uint64_t key =
      (uint64_t(masks) << 32) |
      (dc.value & ~0x8u);  // bit 3 is unused padding on Xenos.
  auto it = depth_stencil_states_.find(key);
  if (it != depth_stencil_states_.end()) {
    return it->second;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider) {
    return nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)provider->device();
  MTLDepthStencilDescriptor* desc = [[MTLDepthStencilDescriptor alloc] init];
  desc.depthCompareFunction = dc.z_enable
                                  ? ToMetalCompareFunction(dc.zfunc)
                                  : MTLCompareFunctionAlways;
  desc.depthWriteEnabled = dc.z_enable && dc.z_write_enable;
  if (dc.stencil_enable) {
    MTLStencilDescriptor* front = [[MTLStencilDescriptor alloc] init];
    front.stencilCompareFunction = ToMetalCompareFunction(dc.stencilfunc);
    front.stencilFailureOperation = ToMetalStencilOperation(dc.stencilfail);
    front.depthStencilPassOperation =
        ToMetalStencilOperation(dc.stencilzpass);
    front.depthFailureOperation = ToMetalStencilOperation(dc.stencilzfail);
    front.readMask = srm.stencilmask;
    front.writeMask = srm.stencilwritemask;
    desc.frontFaceStencil = front;
    MTLStencilDescriptor* back = [[MTLStencilDescriptor alloc] init];
    if (dc.backface_enable) {
      back.stencilCompareFunction = ToMetalCompareFunction(dc.stencilfunc_bf);
      back.stencilFailureOperation =
          ToMetalStencilOperation(dc.stencilfail_bf);
      back.depthStencilPassOperation =
          ToMetalStencilOperation(dc.stencilzpass_bf);
      back.depthFailureOperation =
          ToMetalStencilOperation(dc.stencilzfail_bf);
    } else {
      back.stencilCompareFunction = front.stencilCompareFunction;
      back.stencilFailureOperation = front.stencilFailureOperation;
      back.depthStencilPassOperation = front.depthStencilPassOperation;
      back.depthFailureOperation = front.depthFailureOperation;
    }
    back.readMask = srm_bf.stencilmask;
    back.writeMask = srm_bf.stencilwritemask;
    desc.backFaceStencil = back;
  }
  id<MTLDepthStencilState> state =
      [device newDepthStencilStateWithDescriptor:desc];
  void* state_ptr = (__bridge_retained void*)state;
  depth_stencil_states_.emplace(key, state_ptr);
  return state_ptr;
}

bool MetalCommandProcessor::ConvertDepthToSnapshot(void* depth_texture,
                                                   void* dest_snapshot,
                                                   uint32_t width,
                                                   uint32_t height,
                                                   uint32_t dest_y_offset) {
  if (!frame_command_buffer_) {
    return false;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider) {
    return false;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)provider->device();
  if (!depth_convert_pipeline_) {
    // Exact per-pixel read (snapshot pixel (x,y) == depth-texture pixel (x,y)):
    // the depth texture (1280x1024, shadow-map-capable) is LARGER than most
    // snapshots (1280x720 scene), so normalized-uv sampling squashed the whole
    // texture into the snapshot; and a presenter-style Y-flip mirrored depth
    // vertically relative to the (unflipped, blit-copied) color snapshots —
    // together feeding deferred lighting wrong positions everywhere.
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct VOut { float4 pos [[position]]; };
      vertex VOut xe_depth_convert_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        return o;
      }
      fragment float xe_depth_convert_fs(VOut in [[stage_in]],
                                         depth2d<float> d [[texture(0)]]) {
        return d.read(uint2(in.pos.xy));
      }
    )METAL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
      XELOGE("MetalCommandProcessor: depth-convert library failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return false;
    }
    MTLRenderPipelineDescriptor* d = [[MTLRenderPipelineDescriptor alloc] init];
    d.vertexFunction = [lib newFunctionWithName:@"xe_depth_convert_vs"];
    d.fragmentFunction = [lib newFunctionWithName:@"xe_depth_convert_fs"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatR32Float;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) {
      XELOGE("MetalCommandProcessor: depth-convert pipeline failed: {}",
             err ? err.localizedDescription.UTF8String : "?");
      return false;
    }
    depth_convert_pipeline_ = (__bridge_retained void*)ps;
  }
  id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)frame_command_buffer_;
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)dest_snapshot;
  pass.colorAttachments[0].slice = 0;
  // Load, not DontCare: a multi-strip resolve converts into a SUB-RECT of a
  // full-height snapshot (dest_y_offset), and the other strip's rows must
  // survive this pass.
  pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> encoder =
      [cb renderCommandEncoderWithDescriptor:pass];
  encoder.label = @"cp:depth-convert-snapshot";
  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      depth_convert_pipeline_];
  // The viewport clips the fullscreen triangle to the strip's rows; the FS
  // reads the depth texture at the destination pixel coordinate, so source and
  // destination rows align automatically (both are frame-absolute).
  MTLViewport vp{0.0, double(dest_y_offset), double(width), double(height), 0.0,
                 1.0};
  [encoder setViewport:vp];
  [encoder setFragmentTexture:(__bridge id<MTLTexture>)depth_texture atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [encoder endEncoding];
  return true;
}

bool MetalCommandProcessor::IssueCopy() {
  // First slice of the EDRAM->guest resolve (the gameplay gray-screen blocker):
  // decode the resolve with the reusable draw_util helper and record the guest
  // destination base so the texture cache can bridge it to our color RT. No data
  // is copied yet -- this pass logs what JC2 resolves so the RT-as-texture bridge
  // can be built against real addresses/formats.
  // GetResolveInfo decodes the resolve rectangle from vf0 on the host. The
  // D3D9 convention says the CPU normally writes these vertices, but the
  // authority contract must cover the factual address rather than assume its
  // writer. Resolve an aliased pending producer before the host dereference.
  xenos::xe_gpu_vertex_fetch_t resolve_fetch =
      register_file_->GetVertexFetch(0);
  if (resolve_fetch.type == xenos::FetchConstantType::kVertex &&
      resolve_fetch.size == 3 * 2) {
    uint32_t resolve_vertices_address =
        xenos::CpuToGpu(resolve_fetch.address * sizeof(uint32_t));
    if (!PrepareGuestMemoryForHostRead(
            resolve_vertices_address,
            resolve_fetch.size * sizeof(uint32_t))) {
      XELOGE(
          "MetalCommandProcessor: failed to converge resolve vf0 {:08X}+{}",
          resolve_vertices_address,
          resolve_fetch.size * uint32_t(sizeof(uint32_t)));
      return false;
    }
  }

  draw_util::ResolveInfo resolve_info;
  if (!draw_util::GetResolveInfo(*register_file_, *memory_, trace_writer_,
                                 /*draw_resolution_scale_x=*/1,
                                 /*draw_resolution_scale_y=*/1,
                                 /*fixed_rg16_truncated_to_minus_1_to_1=*/false,
                                 /*fixed_rgba16_truncated_to_minus_1_to_1=*/false,
                                 resolve_info)) {
    return true;
  }
  // Commit encoded draws BEFORE processing the resolve (default-on;
  // XE_METAL_NO_RESOLVE_COMMIT=1 restores the old whole-frame buffer for A/B).
  // Found via the resolve-owner probe: sessions WITHOUT a submission boundary
  // here page-faulted 60-84x/session (the recurring 'draws@N' +
  // 'rtcache:resolve-color-snapshot' fault) and every faulted command buffer
  // DISCARDED its encoded work — scene color lost -> the black flicker bursts.
  // With the boundary: 0 faults, no flicker, ~4 commits/frame at an unchanged
  // 28.4 fps (measured live). This is also the XeniOS submission shape:
  // command buffers split at hazard boundaries.
  // LARGE resolves only: a boundary at EVERY resolve measured 39 commits/frame
  // -> 23.1 fps (-19%), while the proven-0-fault probe configuration bounded
  // only the scene-sized resolve. 1MB splits the scene G-buffers/frontbuffer
  // (1.9-3.7MB) from the pyramid/quarter chain (4-256KB).
  static const bool s_no_resolve_commit =
      std::getenv("XE_METAL_NO_RESOLVE_COMMIT") != nullptr;
  constexpr uint32_t kResolveCommitMinBytes = 1024 * 1024;
  if (!s_no_resolve_commit && draws_since_commit_ > 0 &&
      resolve_info.copy_dest_extent_length >= kResolveCommitMinBytes) {
    EndFrameEncoder();
  }
  // CRITICAL: the resolve paths below all guard on frame_command_buffer_. If
  // the frame was committed (just above, or by a mid-frame hazard flush,
  // writeback wait, --serialize_draw_range), the buffer is NULL and every
  // resolve op would be SILENTLY SKIPPED — the consumer then samples a stale
  // or never-written snapshot (missing scene / transient black blips; found
  // by the draw-range bisect: one commit after the last scene draw dropped
  // the scene resolve and lost the whole frame). Recreate it up front.
  PrepareTransferCommandBuffer();
  // RT-as-texture bridge. A color resolve is the boundary of a render pass: blit
  // what the pass drew into ITS color target (looked up by the resolve's EDRAM
  // base + format — the same key the draws bound it with) out into a
  // per-destination snapshot texture registered at the resolve's guest base, so
  // a later deferred / composite pass that fetches this base samples the real
  // rendered pixels (its guest RAM is never populated by our backend).
  if (!resolve_info.IsCopyingDepth() && resolve_info.copy_dest_extent_length) {
    last_color_resolve_base_ = resolve_info.copy_dest_base;
    last_color_resolve_start_ = resolve_info.copy_dest_extent_start;
    last_color_resolve_length_ = resolve_info.copy_dest_extent_length;

    uint32_t rw = resolve_info.coordinate_info.width_div_8 * 8;
    uint32_t rh = resolve_info.height_div_8 * 8;
    if (rw && rh && render_target_cache_ && texture_cache_ &&
        frame_command_buffer_) {
      // Close the current pass's encoder so its stores are visible to the blit,
      // then copy the color RT region into the snapshot on the frame command
      // buffer (Metal orders the blit after the render encoder and before the
      // later composite encoder that samples the snapshot).
      EndRenderEncoder();
      // Ownership truth: ask the base class which render target owns the
      // resolve's EDRAM tiles. The snapshot's pixel format mirrors the
      // source's so the blit is format-compatible.
      uint32_t span_base, span_row_length, span_rows, span_pitch;
      resolve_info.GetCopyEdramTileSpan(span_base, span_row_length, span_rows,
                                        span_pitch);
      // The snapshot format mirrors the first color owner's; the copy itself is
      // per-owner + per-tile with FULL layout remapping (owner pitch vs the
      // resolve's linear span -- a naive rect blit scrambles any owner whose
      // pitch differs, e.g. 4x-MSAA keys twice as wide). Strips coalesce like
      // depth (the guest splits 720p buffers into adjacent strip resolves).
      uint64_t src_pixel_format = 0;
      void* probe = render_target_cache_->GetResolveSourceTexture(
          span_base, span_row_length, span_rows, span_pitch, /*is_depth=*/false,
          &src_pixel_format);
      const bool canonical_resolve =
          render_target_cache_->sample_edram_raster_enabled();
      if (probe || canonical_resolve) {
        // Diagnostic: XE_METAL_PROBE_RESOLVE=<hex dest base> reads back the
        // OWNER texture's top-left region at resolve time (forces a mid-frame
        // commit) — splits "the producing draw rendered zero" from "the
        // resolve remap reads the wrong place".
        static const char* s_probe_resolve =
            std::getenv("XE_METAL_PROBE_RESOLVE");
        if (probe && s_probe_resolve &&
            uint32_t(strtoul(s_probe_resolve, nullptr, 16)) ==
                resolve_info.copy_dest_base) {
          EndFrameEncoderAndWait();
          PrepareTransferCommandBuffer();
          id<MTLTexture> owner_tex = (__bridge id<MTLTexture>)probe;
          uint32_t pw = std::min<uint32_t>(uint32_t(owner_tex.width), 64);
          id<MTLBuffer> pstage =
              [owner_tex.device newBufferWithLength:pw * 4 * 4
                                            options:MTLResourceStorageModeShared];
          if (pstage) {
            id<MTLCommandQueue> pq =
                (__bridge id<MTLCommandQueue>)static_cast<xe::ui::metal::MetalProvider*>(
                    graphics_system_->provider())->command_queue();
            id<MTLCommandBuffer> pcb = [pq commandBuffer];
            id<MTLBlitCommandEncoder> pbl = [pcb blitCommandEncoder];
            [pbl copyFromTexture:owner_tex
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(pw, 4, 1)
                         toBuffer:pstage
                destinationOffset:0
           destinationBytesPerRow:pw * 4
         destinationBytesPerImage:pw * 4 * 4];
            [pbl endEncoding];
            [pcb commit];
            [pcb waitUntilCompleted];
            const uint32_t* w = static_cast<const uint32_t*>(pstage.contents);
            XELOGW(
                "MetalProbeResolve {:08X}: owner {}x{} row0 {:08X} {:08X} "
                "{:08X} {:08X} row2 {:08X} {:08X}",
                resolve_info.copy_dest_base, uint32_t(owner_tex.width),
                uint32_t(owner_tex.height), w[0], w[1], w[8], w[32],
                w[pw * 2], w[pw * 2 + 8]);
          }
        }
        // Honor the GUEST's copy_dest_format: the Xenos resolve CONVERTS. The
        // auto-exposure luminance pyramid resolves k_8_8_8_8 EDRAM into
        // k_32_FLOAT (fmt 36) textures the game (and the next pyramid level)
        // reads as f32 — registering a BGRA8-content snapshot instead made
        // consumers decode BGRA8 bytes as floats (denormals ≈ 0): pyramid
        // levels 2+ rendered genuinely black and the CPU exposure loop read
        // ~0 (the black flicker/darkness). R32Float attachments keep .r of
        // the sampled float4 — exactly the hardware's 8888→k_32_FLOAT rule.
        uint64_t dest_pixel_format = src_pixel_format;
        switch (resolve_info.copy_dest_info.copy_dest_format) {
          case xenos::ColorFormat::k_8_8_8_8:
          case xenos::ColorFormat::k_8_8_8_8_A:
          case xenos::ColorFormat::k_8_8_8_8_AS_16_16_16_16:
            dest_pixel_format = MetalRenderTargetCache::kColorFormat;
            break;
          case xenos::ColorFormat::k_16_16_FLOAT:
            dest_pixel_format = uint64_t(MTLPixelFormatRG16Float);
            break;
          case xenos::ColorFormat::k_16_16_16_16_FLOAT:
            dest_pixel_format = uint64_t(MTLPixelFormatRGBA16Float);
            break;
          case xenos::ColorFormat::k_32_FLOAT:
            dest_pixel_format = uint64_t(MTLPixelFormatR32Float);
            break;
          default:
            break;
        }
        uint32_t strip_y_offset = 0;
        void* parent_snap = render_target_cache_->FindContainingSnapshot(
            resolve_info.copy_dest_base, &strip_y_offset);
        // Half-pass placement audit (XE_METAL_RESOLVE_LOG=1): the guest splits
        // 720p into EDRAM-capacity halves; each half's resolve must land at
        // its own dest rows. Log the raw placement inputs for big resolves.
        static const bool s_resolve_log =
            std::getenv("XE_METAL_RESOLVE_LOG") != nullptr;
        // XE_METAL_RESOLVE_LOG=2: log ALL color resolves (incl. small mips).
        static const bool s_resolve_log_all = [] {
          const char* v = std::getenv("XE_METAL_RESOLVE_LOG");
          return v && v[0] == '2';
        }();
        if ((s_resolve_log && rw * rh * 4 >= 1024 * 1024) ||
            s_resolve_log_all) {
          XELOGW(
              "MetalResolve @draw {}: dest base={:08X} extent=[{:08X},+{}] "
              "rect={}x{} edram_base={} pitch={} msaa={} sample={} dest_pitch={} "
              "off_x={} off_y={} dest_fmt={} swap={} endian={} exp_bias={} "
              "parent={} strip_y={}",
              trace_draw_index_, resolve_info.copy_dest_base,
              resolve_info.copy_dest_extent_start,
              resolve_info.copy_dest_extent_length, rw, rh,
              resolve_info.color_edram_info.base_tiles,
              resolve_info.color_edram_info.pitch_tiles,
              uint32_t(resolve_info.color_edram_info.msaa_samples),
              uint32_t(resolve_info.copy_dest_coordinate_info
                           .copy_sample_select),
              uint32_t(resolve_info.copy_dest_coordinate_info.pitch_aligned_div_32)
                  * 32,
              uint32_t(resolve_info.copy_dest_coordinate_info.offset_x_div_8)
                  * 8,
              uint32_t(resolve_info.copy_dest_coordinate_info.offset_y_div_8)
                  * 8,
              uint32_t(resolve_info.copy_dest_info.copy_dest_format),
              uint32_t(resolve_info.copy_dest_info.copy_dest_swap),
              uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
              int32_t(resolve_info.copy_dest_info.copy_dest_exp_bias),
              parent_snap != nullptr, strip_y_offset);
        }
        // A resolve must NEVER join a parent strip of a different format:
        // "render in the snapshot's REAL format" would silently skip the
        // guest-requested conversion. JC2 exposed this first with an
        // 8888->float luminance resolve whose raw color bytes were consumed
        // as floats; Red Dead adds the inverse HDR-float->8888 frontbuffer
        // case. Keep mixed-format destinations as independent snapshots.
        if (parent_snap &&
            uint64_t(((__bridge id<MTLTexture>)parent_snap).pixelFormat) !=
                dest_pixel_format) {
          parent_snap = nullptr;
          strip_y_offset = 0;
        }
        // A parent-strip join is only valid when the strip actually FITS:
        // same width, and the y range inside the parent texture. SF4's intro
        // double-buffers 1280-pitch column dests 0x11000 apart — the second
        // frame's 544x720 resolve "joined" the first frame's 544-wide
        // snapshot at strip_y=32 (69632/544/4 = exactly 32 rows), rendering
        // 720 rows into a 720-tall texture: the bottom 32 rows clipped and
        // the writeback blit read out of bounds -> stale bottom-of-column
        // bands live (the lower-right smear zones) and every-other-frame
        // ghosting. Reject the join; the resolve gets its own snapshot and
        // stage-3 RAM assembly handles placement.
        if (parent_snap) {
          id<MTLTexture> parent_tex = (__bridge id<MTLTexture>)parent_snap;
          if (rw != uint32_t(parent_tex.width) ||
              strip_y_offset + rh > uint32_t(parent_tex.height)) {
            parent_snap = nullptr;
            strip_y_offset = 0;
          }
        }
        void* snap = parent_snap;
        bool is_new_snapshot = false;
        if (!snap) {
          snap = render_target_cache_->EnsureResolveSnapshot(
              resolve_info.copy_dest_base, rw, rh, dest_pixel_format);
          is_new_snapshot = snap != nullptr;
          strip_y_offset = 0;
        }
        // A coalesced parent strip may predate the format decision — always
        // render in the snapshot texture's REAL format.
        if (snap) {
          dest_pixel_format =
              uint64_t(((__bridge id<MTLTexture>)snap).pixelFormat);
        }
        float exp_bias_scale = std::ldexp(
            1.0f,
            int32_t(resolve_info.copy_dest_info.copy_dest_exp_bias));
        // Raw-copy eligibility is shared resolve semantics, not a Metal
        // decision: ask draw_util rather than re-deriving the condition here,
        // so shared changes to it cannot silently skip this backend.
        const bool canonical_raw_copy = resolve_info.IsColorRawCopySafe();
        bool resolved = false;
        bool canonical_prepared = false;
        if (canonical_resolve) {
          canonical_prepared =
              render_target_cache_->PrepareCanonicalEdramForResolve(
                  span_base, span_row_length, span_rows, span_pitch);
          if (snap && canonical_prepared) {
            resolved = render_target_cache_->ResolveCanonicalToSnapshot(
                         frame_command_buffer_, /*is_depth=*/false,
                         resolve_info.color_edram_info.base_tiles,
                         resolve_info.color_edram_info.pitch_tiles,
                         uint32_t(resolve_info.color_edram_info.msaa_samples),
                         resolve_info.color_edram_info.format,
                         resolve_info.color_edram_info.format_is_64bpp != 0,
                         uint32_t(resolve_info.coordinate_info
                                      .edram_offset_x_div_8)
                             << 3,
                         uint32_t(resolve_info.coordinate_info
                                      .edram_offset_y_div_8)
                             << 3,
                         uint32_t(resolve_info.copy_dest_coordinate_info
                                      .copy_sample_select),
                         resolve_info.copy_dest_info.copy_dest_swap != 0,
                         canonical_raw_copy,
                         snap, dest_pixel_format, rw, rh, strip_y_offset,
                         exp_bias_scale);
          }
        } else if (snap) {
          resolved = render_target_cache_->ResolveColorToSnapshot(
              frame_command_buffer_, span_base, span_row_length, span_rows,
              span_pitch,
              uint32_t(resolve_info.color_edram_info.msaa_samples), snap,
              dest_pixel_format, rw, rh, strip_y_offset, exp_bias_scale);
        }
        if (resolved) {
          // Advance provenance for every successful resolve, including a
          // coalesced parent-strip update. Only an encoded color writeback
          // below stamps its layout with this generation; skipped or failed
          // writeback therefore leaves the previous layout provably stale.
          ++resolve_snapshot_generation_[resolve_info.copy_dest_base];
        }
        if (resolved && !parent_snap) {
          // Register (or re-register) every own-snapshot resolve: the entry's
          // dims must track the LAST resolve to the base — base reuse (JC2's
          // mid-frame G-buffer dump into the history base) downgrades them so
          // the covered-fetch bridge fast path correctly falls back to RAM
          // for the composite's full-height temporal read. Parent-strip joins
          // must NOT register: the strip base would alias the parent texture
          // without its y-offset.
          texture_cache_->RegisterResolvedTarget(resolve_info.copy_dest_base,
                                                 snap, rw, rh,
                                                 /*is_depth=*/false,
                                                 canonical_resolve,
                                                 resolve_info.copy_dest_extent_start,
                                                 resolve_info.copy_dest_extent_length);
          // XE_METAL_TIP_SNAP_VIEW (stabilize 6.6 experiment, default off):
          // maintain a derived 1x1 R32F copy of the snapshot's guest texel
          // (0,0) so 1x1 fetches (the exposure tips) sample the correct
          // texel instead of the middle of the full snapshot. One 4-byte
          // GPU-ordered blit per tip resolve; no wait, no control flow.
          static const bool s_tip_snap_view =
              std::getenv("XE_METAL_TIP_SNAP_VIEW") != nullptr;
          if (s_tip_snap_view &&
              dest_pixel_format == uint64_t(MTLPixelFormatR32Float)) {
            void*& tip = tip_view_textures_[resolve_info.copy_dest_base];
            if (!tip) {
              MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
              td.textureType = MTLTextureType2DArray;
              td.pixelFormat = MTLPixelFormatR32Float;
              td.width = 1;
              td.height = 1;
              td.arrayLength = 1;
              td.mipmapLevelCount = 1;
              td.usage = MTLTextureUsageShaderRead;
              // Shared: the drain updates this texture CPU-side from the
              // POST-HOLD guest word, so the zero-luminance guard reaches the
              // path the tonemap actually samples. A GPU blit from the
              // snapshot bypassed the guard: one sun-blowout zero tile
              // latched exposure black (most view angles black live).
              td.storageMode = MTLStorageModeShared;
              id<MTLTexture> t = [((__bridge id<MTLTexture>)snap).device
                  newTextureWithDescriptor:td];
              tip = (__bridge_retained void*)t;
            }
            if (tip) {
              texture_cache_->RegisterResolvedTipView(
                  resolve_info.copy_dest_base, tip);
            }
          }
          if (xe_lineage_base() &&
              resolve_info.copy_dest_base == xe_lineage_base()) {
            XELOGW(
                "MetalLineage RESOLVE-REG draw={} base={:08X} {}x{} fmt={} "
                "(re-registers dims; coverage for the composite fetch)",
                trace_draw_index_, resolve_info.copy_dest_base, rw, rh,
                dest_pixel_format == MetalRenderTargetCache::kColorFormat
                    ? 0u
                    : uint32_t(dest_pixel_format));
          }
        }
        static const bool s_resolve_log2 =
            std::getenv("XE_METAL_RESOLVE_LOG") != nullptr;
        if (s_resolve_log2 && (!resolved || (!is_new_snapshot && !parent_snap))) {
          XELOGW(
              "MetalResolve @draw {} COLOR {:08X} {}x{}: resolved={} snap={} "
              "new={} parent={} (bridge {})",
              trace_draw_index_, resolve_info.copy_dest_base, rw, rh,
              resolved, snap != nullptr, is_new_snapshot,
              parent_snap != nullptr,
              resolved && is_new_snapshot ? "registered" : "NOT registered");
        }
        // Stage-3 slice: SMALL resolves also write back into REAL guest memory
        // at frame end. JC2's auto-exposure loop is CPU-side: the game reads
        // the resolved 1x1 luminance from RAM to compute the next frame's
        // exposure constant — snapshot-only resolves leave that RAM unwritten.
        static const bool s_no_writeback =
            std::getenv("XE_METAL_NO_WRITEBACK") != nullptr;
        // STAGE 3 (default ON; XE_METAL_STAGE3=0 disables): write back at
        // EVERY size — the composite's temporal history buffer (3.7MB) and
        // the G-buffer dests all live in guest RAM; fetches read RAM (color
        // bridge retired). Viable since the copy_dest_swap fix: the scatter
        // now matches hardware channel order, and SF4's x-offset column
        // resolves REQUIRE exact-address placement (the bridge can't).
        static const bool s_stage3 = [] {
          const char* v = std::getenv("XE_METAL_STAGE3");
          return !v || v[0] != '0';
        }();
        const uint64_t wb_cap =
            s_stage3 ? UINT32_MAX : 256u * 1024u;
        const uint32_t dest_format =
            uint32_t(resolve_info.copy_dest_info.copy_dest_format);
        const FormatInfo& wb_format_info =
            *FormatInfo::Get(xenos::TextureFormat(dest_format));
        const uint32_t wb_bpp = wb_format_info.bits_per_pixel >> 3u;
        const uint32_t wb_bpp_log2 = xe::log2_floor(wb_bpp);
        const bool canonical_packed = canonical_resolve && canonical_prepared;
        const bool legacy_blit_format =
            dest_pixel_format == MetalRenderTargetCache::kColorFormat ||
            dest_pixel_format == uint64_t(MTLPixelFormatR32Float) ||
            dest_pixel_format == uint64_t(MTLPixelFormatRGBA16Float);
        const bool writeback_source_available =
            canonical_packed || (resolved && snap && legacy_blit_format);
        if (writeback_source_available && !s_no_writeback &&
            uint64_t(rw) * rh * wb_bpp <= wb_cap &&
            (canonical_packed || legacy_blit_format)) {
          id<MTLDevice> wb_device = (__bridge id<MTLDevice>)
              static_cast<xe::ui::metal::MetalProvider*>(
                  graphics_system_->provider())
                  ->device();
          id<MTLBuffer> staging =
              [wb_device newBufferWithLength:size_t(rw) * rh * wb_bpp
                                     options:MTLResourceStorageModeShared];
          if (staging) {
            bool writeback_encoded = false;
            if (canonical_packed) {
              writeback_encoded =
                  render_target_cache_->ResolveCanonicalColorToBuffer(
                      frame_command_buffer_,
                      resolve_info.color_edram_info.base_tiles,
                      resolve_info.color_edram_info.pitch_tiles,
                      uint32_t(resolve_info.color_edram_info.msaa_samples),
                      resolve_info.color_edram_info.format,
                      resolve_info.color_edram_info.format_is_64bpp != 0,
                      uint32_t(resolve_info.coordinate_info
                                   .edram_offset_x_div_8)
                          << 3,
                      uint32_t(resolve_info.coordinate_info
                                   .edram_offset_y_div_8)
                          << 3,
                      uint32_t(resolve_info.copy_dest_coordinate_info
                                   .copy_sample_select),
                      resolve_info.copy_dest_info.copy_dest_swap != 0,
                      exp_bias_scale, dest_format, wb_bpp_log2,
                      uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
                      canonical_raw_copy, (__bridge void*)staging, rw, rh);
            } else {
              id<MTLCommandBuffer> wb_cb =
                  (__bridge id<MTLCommandBuffer>)frame_command_buffer_;
              id<MTLBlitCommandEncoder> wb_blit = [wb_cb blitCommandEncoder];
              wb_blit.label = @"cp:writeback-blit";
              [wb_blit copyFromTexture:(__bridge id<MTLTexture>)snap
                            sourceSlice:0
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, strip_y_offset, 0)
                             sourceSize:MTLSizeMake(rw, rh, 1)
                               toBuffer:staging
                      destinationOffset:0
                 destinationBytesPerRow:rw * wb_bpp
               destinationBytesPerImage:rw * rh * wb_bpp];
              [wb_blit endEncoding];
              writeback_encoded = true;
            }
            const auto& dc = resolve_info.copy_dest_coordinate_info;
            uint32_t extent_bytes =
                resolve_info.copy_dest_extent_start - resolve_info.copy_dest_base +
                resolve_info.copy_dest_extent_length;
            // copy_dest_swap only applies to multi-channel color packing
            // (resolve.xesli writes .bgra); R32Float has nothing to swap.
            uint32_t wb_swap_rb =
                dest_pixel_format == MetalRenderTargetCache::kColorFormat &&
                !canonical_resolve &&
                resolve_info.copy_dest_info.copy_dest_swap;
            if (writeback_encoded) {
              pending_writebacks_.push_back(PendingWriteback{
                  (__bridge_retained void*)staging,
                  resolve_info.copy_dest_base,
                  rw,
                  rh,
                  wb_bpp,
                  canonical_packed
                      ? 0u
                      : uint32_t(
                            resolve_info.copy_dest_info.copy_dest_endian),
                  uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
                  uint32_t(dc.pitch_aligned_div_32) << 5,
                  uint32_t(dc.offset_x_div_8) << 3,
                  uint32_t(dc.offset_y_div_8) << 3,
                  extent_bytes,
                  uint32_t(dc.height_aligned_div_32) << 5,
                  uint32_t(resolve_info.copy_dest_info.copy_dest_slice),
                  uint32_t(resolve_info.copy_dest_info.copy_dest_array),
                  wb_swap_rb,
                  uint32_t(resolve_info.copy_dest_info.copy_dest_swap),
                  uint32_t(canonical_packed),
                  S7TagsOff() ? 0 : CurrentEncodingSubmission(),
                  dest_format,
                  trace_draw_index_,
                  swap_count_,
                  /*depth_encode=*/0u,
                  ++memory_authority_sequence_});
            }
            if (writeback_encoded &&
                dest_pixel_format == MetalRenderTargetCache::kColorFormat) {
              ram_color_layouts_[resolve_info.copy_dest_base] = RamColorLayout{
                  uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
                  uint32_t(resolve_info.copy_dest_info.copy_dest_swap),
                  resolve_snapshot_generation_[resolve_info.copy_dest_base]};
            }
            // Stage-3 hybrid: RAM now holds (will hold) this dest — fetches
            // of the base must read RAM, not the bridge snapshot. Full-rect
            // geometry (offset 0,0 and pitch == width) lets the exact-cover
            // reader prove snapshot/RAM value equivalence for equal-extent
            // fetches.
            if (writeback_encoded && texture_cache_) {
              texture_cache_->MarkResolvedTargetRamBacked(
                  resolve_info.copy_dest_base, rw, rh,
                  dc.offset_x_div_8 == 0 && dc.offset_y_div_8 == 0 &&
                      (uint32_t(dc.pitch_aligned_div_32) << 5) == rw,
                  uint32_t(resolve_info.copy_dest_info.copy_dest_endian));
            }
            if (writeback_encoded && xe_lineage_base() &&
                resolve_info.copy_dest_base == xe_lineage_base()) {
              XELOGW(
                  "MetalLineage WB-ENQUEUE draw={} base={:08X} {}x{} bpp={} "
                  "endian={} swap_rb={} pitch={} off=({},{}) (queued -> RAM at "
                  "drain; ram_backed=true)",
                  trace_draw_index_, resolve_info.copy_dest_base, rw, rh, wb_bpp,
                  uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
                  wb_swap_rb, uint32_t(dc.pitch_aligned_div_32) << 5,
                  uint32_t(dc.offset_x_div_8) << 3,
                  uint32_t(dc.offset_y_div_8) << 3);
            }
          }
        }
      } else {
        static uint32_t s_missing_color_rt_logs = 0;
        if (s_missing_color_rt_logs++ < 8) {
          XELOGW(
              "MetalCommandProcessor: color resolve base={} fmt={} has no "
              "rendered target",
              resolve_info.color_original_base,
              uint32_t(resolve_info.color_edram_info.format));
        }
      }
    }
  } else if (resolve_info.IsCopyingDepth() &&
             resolve_info.copy_dest_extent_length) {
    // Depth resolve: deferred lighting samples the depth buffer as a texture to
    // reconstruct position (JC2/Sonic sample fmt 22/23 -> we returned null ->
    // black lighting). Our depth RT is Depth32Float, which Metal only samples as
    // depth2d while MSC shaders declare texture2d_array<float>; convert it to an
    // R32Float snapshot with a small fullscreen pass, and register that. The
    // source is the depth target the pass rendered (by the resolve's EDRAM
    // base), e.g. the shadow map and the scene depth are distinct textures.
    uint32_t rw = resolve_info.coordinate_info.width_div_8 * 8;
    uint32_t rh = resolve_info.height_div_8 * 8;
    if (rw && rh && kDepthEnabled && render_target_cache_ && texture_cache_ &&
        frame_command_buffer_) {
      EndRenderEncoder();
      uint32_t span_base, span_row_length, span_rows, span_pitch;
      resolve_info.GetCopyEdramTileSpan(span_base, span_row_length, span_rows,
                                        span_pitch);
      // Multi-strip resolve: if this strip's destination lands inside an
      // existing (full-height) snapshot, convert into that snapshot at the row
      // offset — shaders sample the FIRST strip's base as the whole buffer.
      uint32_t strip_y_offset = 0;
      void* parent_snap = render_target_cache_->FindContainingSnapshot(
          resolve_info.copy_dest_base, &strip_y_offset);
      static const bool s_resolve_log_d =
          std::getenv("XE_METAL_RESOLVE_LOG") != nullptr;
      if (s_resolve_log_d) {
        XELOGW(
            "MetalResolve @draw {} DEPTH: dest base={:08X} extent=[{:08X},+{}] "
            "rect={}x{} edram_base={} msaa={} parent={} parent_fmt={} strip_y={}",
            trace_draw_index_, resolve_info.copy_dest_base,
            resolve_info.copy_dest_extent_start,
            resolve_info.copy_dest_extent_length, rw, rh,
            resolve_info.depth_edram_info.base_tiles,
            uint32_t(resolve_info.depth_edram_info.msaa_samples),
            parent_snap != nullptr,
            parent_snap
                ? uint64_t(((__bridge id<MTLTexture>)parent_snap).pixelFormat)
                : 0,
            strip_y_offset);
      }
      // Address containment alone doesn't prove that this resolve is a
      // vertical strip of the earlier snapshot. Tiled column resolves may
      // have nearby adjusted bases whose byte delta happens to look like a row
      // offset. Joining those clips the bottom and makes later columns sample
      // stale/repeated depth. Require the same concrete texture compatibility
      // and fit used by the color path.
      if (parent_snap) {
        id<MTLTexture> parent_tex = (__bridge id<MTLTexture>)parent_snap;
        if (parent_tex.pixelFormat != MTLPixelFormatR32Float ||
            rw != uint32_t(parent_tex.width) ||
            strip_y_offset + rh > uint32_t(parent_tex.height)) {
          parent_snap = nullptr;
          strip_y_offset = 0;
        }
      }
      void* snap = parent_snap;
      bool is_new_snapshot = false;
      if (!snap) {
        snap = render_target_cache_->EnsureResolveSnapshot(
            resolve_info.copy_dest_base, rw, rh, MTLPixelFormatR32Float);
        is_new_snapshot = snap != nullptr;
        strip_y_offset = 0;
      }
      // Per-owner conversion (the span can be owned piecewise): depth owners
      // read directly, color owners (raw EDRAM bits, e.g. the restored trace
      // snapshot) decode the guest 24-bit depth.
      bool is_float24 = resolve_info.depth_edram_info.format ==
                        uint32_t(xenos::DepthRenderTargetFormat::kD24FS8);
      bool resolved = false;
      if (snap && render_target_cache_->sample_edram_raster_enabled()) {
        resolved = render_target_cache_->PrepareCanonicalEdramForResolve(
                       span_base, span_row_length, span_rows, span_pitch) &&
                   render_target_cache_->ResolveCanonicalToSnapshot(
                       frame_command_buffer_, /*is_depth=*/true,
                       resolve_info.depth_edram_info.base_tiles,
                       resolve_info.depth_edram_info.pitch_tiles,
                       uint32_t(resolve_info.depth_edram_info.msaa_samples),
                       resolve_info.depth_edram_info.format,
                       /*is_64bpp=*/false,
                       uint32_t(resolve_info.coordinate_info
                                    .edram_offset_x_div_8)
                           << 3,
                       uint32_t(resolve_info.coordinate_info
                                    .edram_offset_y_div_8)
                           << 3,
                       uint32_t(resolve_info.copy_dest_coordinate_info
                                    .copy_sample_select),
                       /*dest_swap=*/false,
                       /*raw_copy=*/false,
                       snap, MTLPixelFormatR32Float, rw, rh, strip_y_offset,
                       1.0f);
      } else if (snap) {
        resolved = render_target_cache_->ResolveDepthToSnapshot(
            frame_command_buffer_, span_base, span_row_length, span_rows,
            span_pitch,
            uint32_t(resolve_info.depth_edram_info.msaa_samples), is_float24,
            snap, rw, rh, strip_y_offset);
      }
      if (resolved) {
        // A depth resolve cannot produce color layout metadata. Advance
        // provenance even when it reuses an existing snapshot so any prior
        // color layout at this base is rejected.
        ++resolve_snapshot_generation_[resolve_info.copy_dest_base];
      }
      if (resolved && is_new_snapshot) {
        texture_cache_->RegisterResolvedTarget(resolve_info.copy_dest_base,
                                               snap, rw, rh, /*is_depth=*/true,
                                               /*requires_fetch_swizzle=*/true,
                                               resolve_info.copy_dest_extent_start,
                                               resolve_info.copy_dest_extent_length);
      }
      // DEPTH RAM writeback: texture consumers get the snapshot (bridge),
      // but VFETCH consumers read guest RAM — JC2's memexport light-atlas VS
      // vfetches the 1024x1024 depth resolve at 121E1000, which stayed
      // permanently zero without this (the black-ambient root, one level
      // above the skipped exports). Blit the R32F snapshot to staging; the
      // canonical path stages the selected raw D24S8/D24FS8 word (including
      // stencil); the legacy fallback still packs its R32F snapshot.
      static const bool s_no_writeback_d =
          std::getenv("XE_METAL_NO_WRITEBACK") != nullptr;
      if (resolved && !s_no_writeback_d) {
        id<MTLDevice> wb_device = ((__bridge id<MTLTexture>)snap).device;
        id<MTLBuffer> staging =
            [wb_device newBufferWithLength:size_t(rw) * rh * 4u
                                   options:MTLResourceStorageModeShared];
        if (staging) {
          const bool canonical_depth_writeback =
              render_target_cache_->sample_edram_raster_enabled();
          bool writeback_encoded = false;
          if (canonical_depth_writeback) {
            writeback_encoded =
                render_target_cache_->ResolveCanonicalDepthToBuffer(
                    frame_command_buffer_,
                    resolve_info.depth_edram_info.base_tiles,
                    resolve_info.depth_edram_info.pitch_tiles,
                    uint32_t(resolve_info.depth_edram_info.msaa_samples),
                    resolve_info.depth_edram_info.format,
                    uint32_t(resolve_info.coordinate_info
                                 .edram_offset_x_div_8)
                        << 3,
                    uint32_t(resolve_info.coordinate_info
                                 .edram_offset_y_div_8)
                        << 3,
                    uint32_t(resolve_info.copy_dest_coordinate_info
                                 .copy_sample_select),
                    (__bridge void*)staging, rw, rh);
          } else {
            id<MTLCommandBuffer> wb_cb =
                (__bridge id<MTLCommandBuffer>)frame_command_buffer_;
            id<MTLBlitCommandEncoder> wb_blit = [wb_cb blitCommandEncoder];
            wb_blit.label = @"cp:depth-writeback-blit";
            [wb_blit copyFromTexture:(__bridge id<MTLTexture>)snap
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, strip_y_offset, 0)
                           sourceSize:MTLSizeMake(rw, rh, 1)
                             toBuffer:staging
                    destinationOffset:0
               destinationBytesPerRow:rw * 4
             destinationBytesPerImage:rw * rh * 4];
            [wb_blit endEncoding];
            writeback_encoded = true;
          }
          const auto& dc = resolve_info.copy_dest_coordinate_info;
          uint32_t extent_bytes = resolve_info.copy_dest_extent_start -
                                  resolve_info.copy_dest_base +
                                  resolve_info.copy_dest_extent_length;
          if (writeback_encoded) {
            pending_writebacks_.push_back(PendingWriteback{
                (__bridge_retained void*)staging,
                resolve_info.copy_dest_base, rw, rh, 4u,
                uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
                uint32_t(resolve_info.copy_dest_info.copy_dest_endian),
                uint32_t(dc.pitch_aligned_div_32) << 5,
                uint32_t(dc.offset_x_div_8) << 3,
                uint32_t(dc.offset_y_div_8) << 3, extent_bytes,
                uint32_t(dc.height_aligned_div_32) << 5,
                uint32_t(resolve_info.copy_dest_info.copy_dest_slice),
                uint32_t(resolve_info.copy_dest_info.copy_dest_array),
                /*swap_rb=*/0u,
                /*layout_swap_rb=*/0u,
                /*packed_bytes=*/0u,
                S7TagsOff() ? 0 : CurrentEncodingSubmission(),
                uint32_t(resolve_info.copy_dest_info.copy_dest_format),
                trace_draw_index_, swap_count_,
                /*depth_encode=*/canonical_depth_writeback
                    ? 0u
                    : (is_float24 ? 2u : 1u),
                ++memory_authority_sequence_});
          }
        }
      }
    }
  }
  // Resolve CLEARS: guests set render-target base values this way (JC2's
  // quarter-res scene multiplies by DESTINATION — an unset base of 0 blackens
  // the chain; shadow cascades clear depth between renders).
  if ((resolve_info.IsClearingColor() || resolve_info.IsClearingDepth()) &&
      render_target_cache_ && frame_command_buffer_) {
    EndRenderEncoder();
    uint32_t span_base, span_row_length, span_rows, span_pitch;
    resolve_info.GetCopyEdramTileSpan(span_base, span_row_length, span_rows,
                                      span_pitch);
    static const bool s_clear_log =
        std::getenv("XE_METAL_RESOLVE_LOG") != nullptr;
    if (s_clear_log) {
      XELOGW(
          "MetalResolveClear @draw {}: color={} depth={} span base={} "
          "rowlen={} rows={} pitch={} c_edram(base={} pitch={} msaa={}) "
          "d_edram(base={} pitch={} msaa={}) clearval c={:08X} d={:08X}",
          trace_draw_index_, uint32_t(resolve_info.IsClearingColor()),
          uint32_t(resolve_info.IsClearingDepth()), span_base, span_row_length,
          span_rows, span_pitch, resolve_info.color_edram_info.base_tiles,
          resolve_info.color_edram_info.pitch_tiles,
          uint32_t(resolve_info.color_edram_info.msaa_samples),
          resolve_info.depth_edram_info.base_tiles,
          resolve_info.depth_edram_info.pitch_tiles,
          uint32_t(resolve_info.depth_edram_info.msaa_samples),
          resolve_info.rb_color_clear, resolve_info.rb_depth_clear);
    }
    // Each clear must use ITS OWN surface's tile span: GetCopyEdramTileSpan
    // returns the span of the surface being COPIED (color unless
    // IsCopyingDepth), but a dual clear also names the OTHER surface via its
    // own edram_info. Passing the color span (base 768) to the depth clear
    // found no depth owners there and silently did nothing — the guest's
    // depth never cleared between the 720p half-passes, and the composite's
    // depth tests ran against the previous half's stale depth (the hard
    // half-screen seam: sky/objects failing GE tests only below y=384).
    const bool canonical_clear =
        render_target_cache_->sample_edram_raster_enabled();
    const uint32_t clear_source_x =
        uint32_t(resolve_info.coordinate_info.edram_offset_x_div_8) << 3;
    const uint32_t clear_source_y =
        uint32_t(resolve_info.coordinate_info.edram_offset_y_div_8) << 3;
    const uint32_t clear_width =
        uint32_t(resolve_info.coordinate_info.width_div_8) << 3;
    const uint32_t clear_height = resolve_info.height_div_8 << 3;
    // Match the reference backends' command order. If the declared depth and
    // color spans alias, the later color clear is the final physical producer.
    if (resolve_info.IsClearingDepth()) {
      uint32_t dbase, drowlen, drows;
      draw_util::GetResolveEdramTileSpan(resolve_info.depth_edram_info,
                                         resolve_info.coordinate_info,
                                         resolve_info.height_div_8, dbase,
                                         drowlen, drows);
      bool cleared = canonical_clear &&
                     render_target_cache_->PerformCanonicalResolveClear(
                         frame_command_buffer_, dbase, drowlen, drows,
                         resolve_info.depth_edram_info.pitch_tiles,
                         resolve_info.depth_edram_info.base_tiles,
                         resolve_info.depth_edram_info.pitch_tiles,
                         uint32_t(
                             resolve_info.depth_edram_info.msaa_samples),
                         /*is_depth=*/true, /*is_64bpp=*/false,
                         clear_source_x, clear_source_y, clear_width,
                         clear_height, resolve_info.rb_depth_clear,
                         resolve_info.rb_depth_clear);
      if (!cleared) {
        render_target_cache_->PerformResolveClear(
            frame_command_buffer_, dbase, drowlen, drows,
            resolve_info.depth_edram_info.pitch_tiles, /*is_depth=*/true,
            resolve_info.rb_depth_clear);
      }
    }
    if (resolve_info.IsClearingColor()) {
      uint32_t cbase, crowlen, crows;
      draw_util::GetResolveEdramTileSpan(resolve_info.color_edram_info,
                                         resolve_info.coordinate_info,
                                         resolve_info.height_div_8, cbase,
                                         crowlen, crows);
      bool cleared = canonical_clear &&
                     render_target_cache_->PerformCanonicalResolveClear(
                         frame_command_buffer_, cbase, crowlen, crows,
                         resolve_info.color_edram_info.pitch_tiles,
                         resolve_info.color_edram_info.base_tiles,
                         resolve_info.color_edram_info.pitch_tiles,
                         uint32_t(
                             resolve_info.color_edram_info.msaa_samples),
                         /*is_depth=*/false,
                         resolve_info.color_edram_info.format_is_64bpp != 0,
                         clear_source_x, clear_source_y, clear_width,
                         clear_height, resolve_info.rb_color_clear,
                         resolve_info.rb_color_clear_lo);
      if (!cleared) {
        render_target_cache_->PerformResolveClear(
            frame_command_buffer_, cbase, crowlen, crows,
            resolve_info.color_edram_info.pitch_tiles, /*is_depth=*/false,
            resolve_info.rb_color_clear);
      }
    }
  }
  // Gamma-parity diagnostic (XE_METAL_GAMMA_PARITY=1): record each color
  // resolve's dest base -> whether its EDRAM SOURCE format was
  // k_8_8_8_8_GAMMA. Raw resolves preserve the byte domain, so a consumer
  // fetch's gamma signs must MATCH the source's gamma-ness; a mismatch means
  // our derived RT format (and thus the PS gamma-encode flag) disagrees with
  // the game's own declaration — the non-cancelling leg of the history
  // cycle (the temporal-oscillator hunt).
  static const bool s_gamma_parity = std::getenv("XE_METAL_GAMMA_PARITY");
  if (s_gamma_parity && !resolve_info.IsCopyingDepth()) {
    bool src_gamma = resolve_info.color_edram_info.format ==
                     uint32_t(xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA);
    gamma_parity_resolves_[resolve_info.copy_dest_base] = src_gamma;
  }
  static uint32_t s_copy_count = 0;
  if (s_copy_count++ < 48) {
    XELOGI(
        "MetalCommandProcessor::IssueCopy dest_base={:08X} extent=[{:08X}+{}] "
        "depth={} fmt={} w_div8={} h_div8={}",
        resolve_info.copy_dest_base, resolve_info.copy_dest_extent_start,
        resolve_info.copy_dest_extent_length, resolve_info.IsCopyingDepth(),
        uint32_t(resolve_info.copy_dest_info.copy_dest_format),
        resolve_info.coordinate_info.width_div_8, resolve_info.height_div_8);
  }
  // Experiment (XE_METAL_SERIALIZE_RESOLVES): commit+wait after each resolve so
  // every snapshot blit fully completes before any later draw samples it. Far
  // cheaper than per-draw serialization (dozens of resolves/frame). If the tile
  // stipple clears under this, the defect is a resolve->sample inter-pass GPU
  // hazard (bindless-sampled snapshots not ordered after their resolve blit).
  static const bool s_serialize_resolves =
      std::getenv("XE_METAL_SERIALIZE_RESOLVES") != nullptr;
  if (s_serialize_resolves) {
    EndFrameEncoderAndWait();
  }
  return true;
}

void MetalCommandProcessor::InitializeShaderStorage(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
    std::function<void()> completion_callback) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking,
                                            nullptr);
  // Title/session boundary: start a fresh unrouted-export reporting session.
  // Touches only the reporter's own counters -- no cache state, no guest state.
  memexport_gap_reporter_.Reset();
  // Title/session boundary: the base class re-seeds its default ramp without
  // invoking the write hook, so re-arm rather than trusting the flag to have
  // survived.
  gamma_ramp_dirty_ = true;
  if (pipeline_cache_) {
    pipeline_cache_->InitializeDurableStorage(cache_root, title_id);
  }
  if (completion_callback) {
    completion_callback();
  }
}



// The base implementation is preserved by calling through, so guest ramp
// tracking is unchanged.
void MetalCommandProcessor::OnGammaRamp256EntryTableValueWritten() {
  CommandProcessor::OnGammaRamp256EntryTableValueWritten();
  // Mark stale so the next swap re-uploads. Deliberately unconditional on the
  // switch: the flag is a single bool with no allocation behind it, and gating
  // it would let a ramp written while disabled be missed if the switch were
  // ever flipped at runtime. Nothing is uploaded until IssueSwap looks.
  gamma_ramp_dirty_ = true;
}

void MetalCommandProcessor::OnGammaRampPWLValueWritten() {
  CommandProcessor::OnGammaRampPWLValueWritten();
}

void MetalCommandProcessor::InitializeTrace() {
  // Warm-entry and shared-memory trace initialization read arbitrary guest RAM
  // ranges. Converge every pending producer once before either sweep so the
  // trace never captures an older RAM version. Do this before writing any
  // initial trace command so a convergence failure leaves no partial capture.
  // With no pending producers this query performs no commit, wait, or drain.
  if (!PrepareGuestMemoryForHostRead(
          0, static_cast<uint32_t>(SharedMemory::kBufferSize))) {
    XELOGE("MetalCommandProcessor: failed to converge RAM for trace start");
    AbortTrace("failed to converge Metal guest RAM at trace start");
    return;
  }
  // Base: all register values + gamma ramps.
  CommandProcessor::InitializeTrace();
  // Guest RAM is CPU-authoritative for this backend (uploads read it
  // directly, resolves write back to it through the CPU) — no GPU downloads
  // are needed. PrepareForTraceDownload invalidates every non-GPU-written
  // shared-memory page so all ranges the traced frame consumes re-upload
  // during it (and get recorded by the UploadRanges hook); any collected
  // GPU-written ranges are written straight from RAM.
  // Shared-subclass warm sweep MUST precede WriteTraceInitialState: its
  // page invalidation fires the watches and outdates every routed texture,
  // and the sweep's "would the legacy cache hold a warm entry" condition is
  // the pre-fire up-to-date state. (The route itself disengages for the
  // captured frame — see the IssueDraw prepass — so everything else about
  // the capture is the pure CPU-path recording.)
  if (shared_texture_cache_) {
    shared_texture_cache_->WriteAllCachedTexturesToTrace(trace_writer_);
  }
  if (shared_memory_) {
    shared_memory_->WriteTraceInitialState(trace_writer_);
  }
  // Warm texture-cache entries won't re-upload during the traced frame —
  // record their guest ranges now (RAM holds their current content).
  if (texture_cache_) {
    texture_cache_->WriteAllCachedTexturesToTrace(trace_writer_);
  }
  // EDRAM: encode the current render-target ownership back to guest tile
  // bytes (the replay seeds its RT caches from this snapshot).
  std::vector<uint8_t> edram;
  if (DumpEdramToBuffer(edram) && edram.size() >= xenos::kEdramSizeBytes) {
    trace_writer_.WriteEdramSnapshot(edram.data());
  }
}

std::string MetalCommandProcessor::GetCaptureDeviceInfoJson() const {
  auto* provider = static_cast<xe::ui::metal::MetalProvider*>(
      graphics_system_->provider());
  id<MTLDevice> device =
      provider ? (__bridge id<MTLDevice>)provider->device() : nil;
  if (!device) {
    return std::string();
  }
  const char* device_name =
      device.name.UTF8String ? device.name.UTF8String : "unknown";
  return fmt::format(
      "{{\"device_name\":\"{}\",\"registry_id\":{},"
      "\"unified_memory\":{},\"low_power\":{},\"removable\":{},"
      "\"recommended_max_working_set_size\":{}}}",
      device_name, uint64_t(device.registryID),
      device.hasUnifiedMemory ? "true" : "false",
      device.isLowPower ? "true" : "false",
      device.isRemovable ? "true" : "false",
      uint64_t(device.recommendedMaxWorkingSetSize));
}

void* MetalCommandProcessor::GetRamFrontbufferTexture(uint32_t base,
                                                      uint32_t width,
                                                      uint32_t height) {
  if (!base || !width || !height || width > 2048 || height > 2048) {
    return nullptr;
  }
  const uint8_t* guest = memory_->TranslatePhysical<const uint8_t*>(base);
  if (!guest) {
    return nullptr;
  }
  // The frontbuffer is a tiled k_8_8_8_8 surface (a resolve dest). Bound the
  // readable range by the tiled extent (rows padded to the 32-row macro tile).
  uint32_t pitch_aligned = (width + 31u) & ~31u;
  uint32_t rows_padded = (height + 31u) & ~31u;
  uint64_t range_bytes = uint64_t(pitch_aligned) * rows_padded * 4;
  constexpr uint64_t kPhysicalMemorySize = 512ull * 1024 * 1024;
  if (uint64_t(base) + range_bytes > kPhysicalMemorySize) {
    return nullptr;
  }
  // The decode below depends on the resolve layout (presence, endian,
  // copy_dest_swap) recorded for this base, not just the raw bytes. Fold that
  // metadata AND the snapshot generation into the reuse key so identical guest
  // bytes decoded under changed metadata can't reuse a stale decoded texture.
  auto layout_it = ram_color_layouts_.find(base);
  bool has_resolve_layout = layout_it != ram_color_layouts_.end();
  RamColorLayout resolve_layout =
      has_resolve_layout ? layout_it->second : RamColorLayout{};
  struct DecodeKey {
    uint64_t bytes_hash;
    uint32_t present;
    uint32_t endian;
    uint32_t swap_rb;
    uint32_t generation;
  } decode_key{XXH3_64bits(guest, size_t(range_bytes)),
               has_resolve_layout ? 1u : 0u, resolve_layout.endian,
               resolve_layout.swap_rb, resolve_layout.generation};
  uint64_t hash = XXH3_64bits(&decode_key, sizeof(decode_key));
  id<MTLTexture> tex = (__bridge id<MTLTexture>)fb_ram_texture_;
  if (tex && fb_ram_width_ == width && fb_ram_height_ == height &&
      fb_ram_content_hash_ == hash) {
    return fb_ram_texture_;  // unchanged frame (video below refresh rate).
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  id<MTLDevice> device =
      provider ? (__bridge id<MTLDevice>)provider->device() : nil;
  if (!device) {
    return nullptr;
  }
  if (!tex || fb_ram_width_ != width || fb_ram_height_ != height) {
    if (fb_ram_texture_) {
      CFRelease(fb_ram_texture_);
      fb_ram_texture_ = nullptr;
    }
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:height
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    tex = [device newTextureWithDescriptor:desc];
    if (!tex) {
      return nullptr;
    }
    tex.label = @"cp:ram-frontbuffer";
    fb_ram_texture_ = (__bridge_retained void*)tex;
    fb_ram_width_ = width;
    fb_ram_height_ = height;
  }
  fb_ram_content_hash_ = hash;
  // Untile guest RAM -> linear RGBA. GPU resolve destinations carry their
  // exact endian / copy_dest_swap packing, which is reversed here. Buffers
  // without a recorded resolve layout are CPU-produced scanout (FMV) in
  // [A][R][G][B] byte order, retained via the legacy rotate-right-8 path.
  std::vector<uint32_t> linear(size_t(width) * height);
  uint32_t* out = linear.data();
  auto untile_row = [&](size_t y) {
    for (uint32_t x = 0; x < width; ++x) {
      uint32_t off = uint32_t(texture_address::Tiled2D(
          int32_t(x), int32_t(y), pitch_aligned, 2));
      uint32_t v;
      std::memcpy(&v, guest + off, 4);
      if (has_resolve_layout) {
        if (resolve_layout.endian <=
            uint32_t(xenos::Endian128::k16in32)) {
          v = xenos::GpuSwapInline(
              v, xenos::Endian(resolve_layout.endian));
        }
        if (resolve_layout.swap_rb) {
          v = (v & 0xFF00FF00u) | ((v & 0xFFu) << 16) |
              ((v >> 16) & 0xFFu);
        }
        out[y * width + x] = v;
      } else {
        out[y * width + x] = (v >> 8) | (v << 24);
      }
    }
  };
  dispatch_apply(height, DISPATCH_APPLY_AUTO,
                 ^(size_t y) { untile_row(y); });
  [tex replaceRegion:MTLRegionMake2D(0, 0, width, height)
         mipmapLevel:0
           withBytes:linear.data()
         bytesPerRow:size_t(width) * 4];
  static uint32_t s_fb_ram_logs = 0;
  if (s_fb_ram_logs++ < 8) {
    XELOGI("MetalCommandProcessor: RAM frontbuffer present {}x{} @ {:08X}",
           width, height, base);
  }
  return fb_ram_texture_;
}

std::vector<metal::PendingProducer>
MetalCommandProcessor::SnapshotPendingProducers() const {
  // Non-owning read of the two existing pending-producer queues into the
  // device-free authority vocabulary. Ranges are normalized/clamped by
  // GuestRange; a record whose extent has no in-bounds byte is skipped (it can
  // never be a real dependency). The existing records carry one cross-kind
  // registration sequence so the seam sees chronological producer order even
  // though the two backing queues are stored and drained separately.
  std::vector<metal::PendingProducer> producers;
  producers.reserve(pending_writebacks_.size() + pending_memexports_.size());
  for (const PendingWriteback& wb : pending_writebacks_) {
    auto range = metal::GuestRange::FromBaseLength(wb.guest_addr,
                                                   wb.range_bytes());
    if (range) {
      producers.push_back({*range, metal::TransitState::kPendingResolve,
                           wb.submission, wb.authority_sequence});
    }
  }
  for (const PendingMemexport& me : pending_memexports_) {
    auto range =
        metal::GuestRange::FromBaseLength(me.base_address, me.size_bytes);
    if (range) {
      producers.push_back({*range, metal::TransitState::kPendingMemexport,
                           me.submission, me.authority_sequence});
    }
  }
  std::stable_sort(producers.begin(), producers.end(),
                   [](const metal::PendingProducer& a,
                      const metal::PendingProducer& b) {
                     return a.sequence < b.sequence;
                   });
  return producers;
}

metal::ConsumerDecision
MetalCommandProcessor::QueryPendingProducerDependencies(
    metal::ConsumerKind consumer,
    const metal::ConsumerRequest& request) const {
  // Task 4.1's one multi-range query. It selects only intersecting records
  // from the two existing producer queues and owns no authority state. The
  // caller executes the promoted one-boundary commit/wait and selective
  // drain policy for the returned decision.
  return metal::DecideConsumer(
      consumer, request, SnapshotPendingProducers(),
      completed_submission_.load(std::memory_order_acquire));
}

metal::ProducerDrainOutcome MetalCommandProcessor::DrainPendingWritebacks(
    uint32_t filter_base, uint32_t filter_size,
    uint64_t filter_authority_sequence) {
  metal::ProducerDrainOutcome outcome;
  auto filter_range =
      metal::GuestRange::FromBaseLength(filter_base, filter_size);
  auto in_filter = [&](uint32_t base, uint32_t size,
                       uint64_t authority_sequence) {
    if (filter_authority_sequence &&
        authority_sequence != filter_authority_sequence) {
      return false;
    }
    auto producer_range = metal::GuestRange::FromBaseLength(base, size);
    return filter_range && producer_range &&
           filter_range->Overlaps(*producer_range);
  };
  // Resolve guest-memory writebacks (the CPU-side exposure loop and any
  // fence-after-copy data). S7: a staging buffer is final only once its
  // submission completed — process those, keep the rest pending (with the
  // synchronous EndFrameEncoder wait every entry is completed by the time
  // any drain site runs, so this is behavior-neutral until S8).
  uint64_t completed = completed_submission_.load(std::memory_order_acquire);
  // Memexport drains first: exports are linear and already guest-endian in
  // the shared-memory buffer (MTLStorageModeShared), so RAM coherence is a
  // plain memcpy once the exporting submission has completed. Invalidate
  // cached texture uploads overlapping the range so the next fetch re-reads.
  if (!pending_memexports_.empty()) {
    std::vector<PendingMemexport> still_pending_exports;
    const bool exact_sequence = filter_authority_sequence != 0;
    auto export_begin = pending_memexports_.begin();
    auto export_end = pending_memexports_.end();
    if (exact_sequence) {
      export_begin = std::lower_bound(
          pending_memexports_.begin(), pending_memexports_.end(),
          filter_authority_sequence,
          [](const PendingMemexport& pending, uint64_t sequence) {
            return pending.authority_sequence < sequence;
          });
      export_end =
          export_begin != pending_memexports_.end() &&
                  export_begin->authority_sequence == filter_authority_sequence
              ? export_begin + 1
              : export_begin;
    } else {
      still_pending_exports.reserve(pending_memexports_.size());
    }
    bool retire_exact_export = false;
    id<MTLBuffer> me_shared_buffer = shared_memory_
                                         ? (__bridge id<MTLBuffer>)
                                               shared_memory_->GetBuffer()
                                         : nil;
    const uint8_t* me_contents =
        static_cast<const uint8_t*>(me_shared_buffer.contents);
    MetalMemexportConvergenceTarget convergence_target(
        memory_, shared_memory_.get(), texture_cache_.get());
    for (auto export_it = export_begin; export_it != export_end; ++export_it) {
      const PendingMemexport& me = *export_it;
      if (me.submission > completed ||
          !in_filter(me.base_address, me.size_bytes,
                     me.authority_sequence)) {
        if (!exact_sequence) {
          still_pending_exports.push_back(me);
        }
        continue;
      }
      std::optional<metal::GuestRange> range =
          metal::GuestRange::FromBaseLength(me.base_address, me.size_bytes);
      if (!range) {
        // Invalid decision (no in-bounds byte): unreachable in practice,
        // since both the destination-preparation and publication seams
        // already reject out-of-domain ranges before a record is
        // registered, but retain rather than drop so a record is never
        // silently discarded outside the explicit success path.
        if (!exact_sequence) {
          still_pending_exports.push_back(me);
        }
        continue;
      }
      static uint32_t s_me_drains = 0;
      // First 8, then a 3-drain sample every 512 (~once per 170 frames at 3
      // exports/frame) — steady-state gameplay content stays observable.
      bool me_log = s_me_drains < 8 || (s_me_drains & 511) < 3;
      ++s_me_drains;
      uint32_t old_hash = 0, new_hash = 0;
      if (me_log && me_contents) {
        // Store-landing oracle: if the buffer content equals the RAM it was
        // primed from, the shader's UAV stores never landed (silent
        // RAM->buffer->RAM round-trip) — a binding problem, not a content
        // problem. Read before the seam's copy so this never races the
        // convergence write into the same guest bytes.
        uint8_t* me_dest_peek =
            memory_->TranslatePhysical<uint8_t*>(me.base_address);
        if (me_dest_peek) {
          old_hash = xe_lineage_stridehash(me_dest_peek, me.size_bytes);
        }
        new_hash = xe_lineage_stridehash(me_contents + me.base_address,
                                         me.size_bytes);
      }
      // Routed through the typed task-3.3 convergence seam: exact
      // shared-buffer bytes -> RAM, no shared-buffer invalidation, and the
      // record is retired only when the copy itself reports success.
      metal::PendingProducer producer{*range,
                                      metal::TransitState::kPendingMemexport,
                                      me.submission, me.authority_sequence};
      metal::MemexportConvergenceResult result =
          metal::ConvergeMemexportRam(producer, convergence_target);
      if (!result.converged) {
        // Missing RAM mapping or shared-buffer contents: a factual
        // disagreement, not a silent drop -- retain the record for retry.
        if (!exact_sequence) {
          still_pending_exports.push_back(me);
        }
        static uint32_t s_failed_convergences = 0;
        if (s_failed_convergences++ < 8) {
          XELOGE(
              "MetalMemexport: convergence failed for {:08X}(+{}); "
              "retaining pending record for retry",
              me.base_address, me.size_bytes);
        }
        continue;
      }
      ++outcome.drained_records;
      ++outcome.memexport_records;
      outcome.invalidated_entries += result.invalidated_entries;
      retire_exact_export = exact_sequence;
      if (me_log && me_contents) {
        const uint32_t* buf_words = reinterpret_cast<const uint32_t*>(
            me_contents + me.base_address);
        uint32_t nonzero = 0;
        uint32_t word_count = me.size_bytes / 4;
        for (uint32_t w = 0; w < word_count; ++w) {
          nonzero += buf_words[w] != 0;
        }
        XELOGI("MetalMemexport: drained {:08X}(+{}) ramHash {:08x} -> "
               "bufHash {:08x} nonzero {}/{} w0={:08X} w1={:08X}{}",
               me.base_address, me.size_bytes, old_hash, new_hash, nonzero,
               word_count, buf_words[0], buf_words[1],
               old_hash == new_hash ? " (UNCHANGED — stores did not land?)"
                                    : " (exported content differs)");
      }
    }
    if (retire_exact_export) {
      pending_memexports_.erase(export_begin);
    } else if (!exact_sequence) {
      pending_memexports_ = std::move(still_pending_exports);
    }
  }
  std::vector<PendingWriteback> still_pending;
  const bool exact_sequence = filter_authority_sequence != 0;
  auto writeback_begin = pending_writebacks_.begin();
  auto writeback_end = pending_writebacks_.end();
  if (exact_sequence) {
    writeback_begin = std::lower_bound(
        pending_writebacks_.begin(), pending_writebacks_.end(),
        filter_authority_sequence,
        [](const PendingWriteback& pending, uint64_t sequence) {
          return pending.authority_sequence < sequence;
        });
    writeback_end =
        writeback_begin != pending_writebacks_.end() &&
                writeback_begin->authority_sequence == filter_authority_sequence
            ? writeback_begin + 1
            : writeback_begin;
  } else {
    still_pending.reserve(pending_writebacks_.size());
  }
  bool retire_exact_writeback = false;
  for (auto writeback_it = writeback_begin; writeback_it != writeback_end;
       ++writeback_it) {
    PendingWriteback& wb = *writeback_it;
    if (wb.submission > completed ||
        !in_filter(wb.guest_addr, wb.range_bytes(),
                   wb.authority_sequence)) {
      if (!exact_sequence) {
        still_pending.push_back(wb);
      }
      continue;
    }
    uint32_t invalidated_entries = 0;
    metal::RamWritePublicationOutcome publish_outcome;
    // Borrow the retained staging buffer until RAM publication succeeds. A
    // failed mapping/invalidation must keep both the record and its staging
    // ownership so a later drain can retry rather than silently retiring
    // GPU-newer data.
    id<MTLBuffer> staging = (__bridge id<MTLBuffer>)wb.staging;
    uint8_t* dest = memory_->TranslatePhysical<uint8_t*>(wb.guest_addr);
    if (dest && staging) {
      const uint32_t* src_words =
          static_cast<const uint32_t*>(staging.contents);
      const uint8_t* src_bytes =
          static_cast<const uint8_t*>(staging.contents);
      uint32_t count = wb.width * wb.height;
      const uint32_t words_per_texel = (wb.bpp + 3u) / 4u;
      uint64_t content_sum = 0;
      if (wb.bpp < 4u) {
        for (uint32_t i = 0; i < count * wb.bpp; ++i) {
          content_sum += src_bytes[i];
        }
      } else {
        for (uint32_t i = 0; i < count * words_per_texel; ++i) {
          content_sum += src_words[i] & 0xFFFFFF;  // ignore alpha
        }
      }
      // Scatter through the TILED addresser: resolve dests are tiled guest
      // textures (32bpp: block == pixel, block_log2 == 2). A linear memcpy
      // makes every tiled fetch of the dest read block-scrambled garbage.
      uint32_t pitch_aligned =
          wb.dest_pitch_aligned ? wb.dest_pitch_aligned : ((wb.width + 31) & ~31u);
      uint32_t height_aligned = wb.dest_height_aligned
                                    ? wb.dest_height_aligned
                                    : ((wb.height + 31u) & ~31u);
      const uint32_t block_log2 = xe::log2_floor(wb.bpp);
      auto tiled_offset = [&](uint32_t x, uint32_t y) {
        if (wb.dest_is_3d) {
          return uint32_t(texture_address::Tiled3D(
              int32_t(wb.dest_offset_x + x), int32_t(wb.dest_offset_y + y),
              int32_t(wb.dest_slice), pitch_aligned, height_aligned,
              int32_t(block_log2)));
        }
        return uint32_t(texture_address::Tiled2D(
            int32_t(wb.dest_offset_x + x), int32_t(wb.dest_offset_y + y),
            pitch_aligned, int32_t(block_log2)));
      };
      auto scatter_row = [&](const uint8_t* bytes, size_t y) {
        for (uint32_t x = 0; x < wb.width; ++x) {
          uint32_t off = tiled_offset(x, uint32_t(y));
          if (wb.dest_extent_bytes && off + wb.bpp > wb.dest_extent_bytes) {
            continue;
          }
          const uint8_t* texel_bytes =
              bytes + (y * wb.width + x) * wb.bpp;
          if (wb.packed_bytes) {
            std::memcpy(dest + off, texel_bytes, wb.bpp);
            continue;
          }
          const uint32_t* texel =
              reinterpret_cast<const uint32_t*>(texel_bytes);
          for (uint32_t w = 0; w < words_per_texel; ++w) {
            uint32_t source_word = w;
            uint32_t endian32 = wb.endian;
            if (wb.bpp == 8u && wb.endian ==
                                    uint32_t(xenos::Endian128::k8in64)) {
              source_word ^= 1u;
              endian32 = uint32_t(xenos::Endian128::k8in32);
            } else if (endian32 >
                       uint32_t(xenos::Endian128::k16in32)) {
              endian32 = uint32_t(xenos::Endian128::kNone);
            }
            uint32_t v = texel[source_word];
            if (wb.depth_encode) {
              // Staging holds R32Float depth from the snapshot conversion;
              // pack to the guest D24S8 word (stencil 0 in v1 — extend from
              // the d2c stencil plane if a title needs it in RAM).
              float df;
              std::memcpy(&df, &v, sizeof(df));
              uint32_t d24 = wb.depth_encode == 2
                                 ? xenos::Float32To20e4(df, true)
                                 : xenos::Float32ToUNorm24(df);
              v = PackEdramDepthStencil(d24, 0u);
            }
            if (wb.swap_rb) {
              // RB_COPY_DEST_INFO.copy_dest_swap: hardware packs .bgra
              // (resolve.xesli) — staging is RGBA8, exchange bytes 0/2.
              // (32bpp color only; never set for R32F/RGBA16F.)
              v = (v & 0xFF00FF00u) | ((v & 0xFFu) << 16) |
                  ((v >> 16) & 0xFFu);
            }
            v = xenos::GpuSwapInline(v, xenos::Endian(endian32));
            std::memcpy(dest + off + w * 4, &v, 4);
          }
        }
      };
      // Rows scatter to disjoint tiled addresses — run them concurrently for
      // the big stage-3 writebacks (SF4 scatters several MB per frame on the
      // GPU thread; the single-threaded loop was a visible slice of the live
      // frame).
      auto scatter_bytes = [&](const uint8_t* bytes) {
        if (size_t(wb.width) * wb.height >= 128 * 1024) {
          dispatch_apply(wb.height, DISPATCH_APPLY_AUTO,
                         ^(size_t y) { scatter_row(bytes, y); });
        } else {
          for (uint32_t y = 0; y < wb.height; ++y) {
            scatter_row(bytes, y);
          }
        }
      };
      scatter_bytes(src_bytes);
      static uint32_t s_wb_logs = 0;
      if (s_wb_logs++ < 8) {
        XELOGI("MetalCommandProcessor: wrote back {}x{} resolve to {:08X}",
               wb.width, wb.height, wb.guest_addr);
      }
      // STAGE 3: the writeback changed guest RAM — evict cached textures
      // uploaded from that range so the next fetch re-uploads fresh content.
      // Task 5.3 proved this scoped direct leg is required: the following
      // shared publication suppresses global fan-out to preserve the
      // overlapping resolved bridge, so no legacy watch can replace it.
      if (texture_cache_) {
        invalidated_entries =
            texture_cache_->InvalidateRange(
                wb.guest_addr, wb.range_bytes(),
                metal::MetalTextureCache::InvalidationScope::kRamUploadsOnly);
        outcome.invalidated_entries += invalidated_entries;
      }
      if (xe_lineage_base() && wb.guest_addr == xe_lineage_base()) {
        // RAM now holds this frame's scattered resolve. This ramHash is what a
        // subsequent fetch's untile WOULD read; compare it to the FETCH ramHash
        // and to the NEXT frame's DRAIN to place the temporal read/write order.
        uint32_t rh = xe_lineage_stridehash(dest, size_t(count) * wb.bpp);
        XELOGW(
            "MetalLineage DRAIN base={:08X} {}x{} bytes={} invalidate=[{:08X},+{}] "
            "ramHash={:08x} (scattered -> RAM, cache evicted)",
            wb.guest_addr, wb.width, wb.height, count * wb.bpp, wb.guest_addr,
            wb.range_bytes(), rh);
      }
      // Exposure telemetry: remember the smallest writeback this frame (the
      // pyramid tip the game's exposure controller actually reads).
      if (wb.bpp >= 4u && count <= 256 &&
          (expo_last_wb_words_ == 0 || count <= expo_last_wb_words_)) {
        expo_last_wb_words_ = count;
        float f;
        std::memcpy(&f, &src_words[0], sizeof(f));
        expo_last_wb_value_ = f;
      }
      // Address-filtered exposure probe (Codex): the "smallest writeback" wb is
      // ambiguous when several tiny writebacks exist. Log EVERY <=256-word
      // writeback with its guest ADDRESS + first float, so the true luminance
      // destination and its per-swap trajectory are unambiguous (grep
      // MetalExpoWB, group by addr). Gated by XE_METAL_EXPO_TRACE.
      static const bool s_expo_addr =
          std::getenv("XE_METAL_EXPO_TRACE") != nullptr;
      if (s_expo_addr && wb.bpp >= 4u && count <= 256) {
        float f0, f1;
        std::memcpy(&f0, &src_words[0], sizeof(f0));
        std::memcpy(&f1, count > 1 ? &src_words[1] : &src_words[0], sizeof(f1));
        // Also print the HOST VIRTUAL address the guest PPC read lands on
        // (virtual_membase_ + ea + heap host_address_offset). This is the
        // address to hand an LLDB hardware READ watchpoint -- the JIT loads
        // guest RAM through the virtual mirror, NOT the physical_membase mirror
        // that TranslatePhysical/`dest` uses, so a watch must target this one.
        // Given live: `watchpoint set expression -w read -s 4096 -- <hostva>`
        // on a tip (12EA2000 R32F / 1390B000 8888) catches the exposure read
        // and its guest+host PC. raw32 is the first packed guest word (compare
        // Metal vs Vulkan AFTER endian scatter, not the pre-scatter float).
        // hostva = the VIRTUAL-mirror host address the guest PPC load lands on
        // (virtual_membase_ + ea + heap offset). THIS is the LLDB hardware READ
        // watchpoint target -- the JIT reads guest RAM through the virtual
        // mirror, not the physical_membase mirror our scatter writes.
        const void* host_va = memory_->TranslateVirtual<const void*>(
            wb.guest_addr);
        // raw32 = the word we actually scattered into guest RAM, read back from
        // the PHYSICAL mirror (`dest`, texel (0,0) == tiled offset 0) AFTER the
        // GpuSwapInline endian pack. THIS is the post-endian word to compare
        // Metal vs Vulkan, not the pre-scatter src float. (Read from `dest`, not
        // host_va: in the trace-dump harness the virtual mirror does not reflect
        // the physical write; live they alias, so the guest observes this word.)
        uint32_t raw32;
        std::memcpy(&raw32, dest, sizeof(raw32));
        XELOGI(
            "MetalExpoWB swap#{} addr={:08X} words={} v0={:.6f} v1={:.6f} "
            "hostva={} raw32_scattered={:08X}",
            swap_count_, wb.guest_addr, count, f0, f1, host_va, raw32);
      }
      // Diagnostic-only exposure hold. This must never be default behavior:
      // zero, opaque black, alpha-only color, and zero depth are all valid
      // canonical resolve results and guest RAM must observe them exactly.
      static const bool s_wb_hold =
          std::getenv("XE_METAL_WB_HOLD") != nullptr;
      const bool small = wb.bpp >= 4u && count <= 256;
      if (s_wb_hold && small && content_sum == 0) {
        static uint32_t s_zero_wb = 0;
        ++s_zero_wb;
        auto held = held_writeback_content_.find(wb.guest_addr);
        if (held != held_writeback_content_.end() &&
            held->second.size() == count * words_per_texel) {
          // Re-write the last good luminance instead of the black tile. The
          // held words are PRE-swap staging content; scatter like the normal
          // path.
          scatter_bytes(reinterpret_cast<const uint8_t*>(
              held->second.data()));
          if (s_zero_wb <= 24 || (s_zero_wb & 255) == 0) {
            XELOGW("MetalWB HOLD {}x{} @ {:08X} (swap#{}, #{}) — held last value",
                   wb.width, wb.height, wb.guest_addr, swap_count_, s_zero_wb);
          }
        } else if (s_zero_wb <= 24 || (s_zero_wb & 255) == 0) {
          XELOGW("MetalWB ZERO {}x{} @ {:08X} (swap#{}, #{}) — no held value",
                 wb.width, wb.height, wb.guest_addr, swap_count_, s_zero_wb);
        }
      } else if (s_wb_hold && small) {
        // Non-zero small writeback: remember it as the fallback for future
        // zero-content resolves at this address (pre-swap staging words; the
        // hold path re-scatters them like a normal drain).
        auto& held = held_writeback_content_[wb.guest_addr];
        held.assign(src_words, src_words + count * words_per_texel);
      }
      // TIP_SNAP_VIEW: refresh the derived 1x1 exposure view from the
      // POST-HOLD guest word. Updating here (CPU, Shared storage, after the
      // zero-hold ran) instead of a GPU blit from the snapshot puts the
      // guard in the exact path the tonemap samples — a zero resolve tile
      // can no longer latch exposure black. Runs on the GPU thread after
      // submission completion and before the next frame encodes: no race.
      {
        auto tip_it = tip_view_textures_.find(wb.guest_addr);
        if (tip_it != tip_view_textures_.end() && tip_it->second &&
            wb.bpp == 4) {
          uint32_t tip_off = tiled_offset(0u, 0u);
          if (!wb.dest_extent_bytes || tip_off + 4 <= wb.dest_extent_bytes) {
            uint32_t tip_word;
            std::memcpy(&tip_word, dest + tip_off, sizeof(tip_word));
            // dest holds the post-endian packed word; the texture needs the
            // logical float — undo the scatter's endian pack.
            uint32_t layout_endian = wb.layout_endian;
            if (layout_endian <=
                uint32_t(xenos::Endian128::k16in32)) {
              tip_word = xenos::GpuSwapInline(
                  tip_word, xenos::Endian(layout_endian));
            }
            id<MTLTexture> tip_tex = (__bridge id<MTLTexture>)tip_it->second;
            [tip_tex replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                       mipmapLevel:0
                             slice:0
                         withBytes:&tip_word
                       bytesPerRow:4
                     bytesPerImage:4];
          }
        }
      }
      // Backend-neutral resolve-writeback record (stabilize task 6.2), after
      // the scatter (and any zero-hold re-scatter) so word0/hash are what a
      // guest CPU load observes. Diagnostic-only; no sync, no control flow.
      if (IsResolveWritebackLogEnabled()) {
        uint32_t rec_extent =
            wb.dest_extent_bytes ? wb.dest_extent_bytes : count * wb.bpp;
        uint32_t first_off = tiled_offset(0u, 0u);
        uint32_t word0 = 0;
        if (first_off + 4 <= rec_extent) {
          std::memcpy(&word0, dest + first_off, sizeof(word0));
        }
        ResolveWritebackRecord record;
        record.backend = "metal";
        record.frame = swap_count_;
        record.sequence = wb.draw_index;
        record.guest_address = wb.guest_addr;
        record.width = wb.width;
        record.height = wb.height;
        record.format = wb.dest_format;
        record.endian = wb.layout_endian;
        record.swap = wb.layout_swap_rb;
        record.extent_bytes = rec_extent;
        record.raw_word0 = word0;
        record.guest_hash = HashResolveWritebackExtent(dest, rec_extent);
        record.host_virtual =
            memory_->TranslateVirtual<const void*>(wb.guest_addr);
        record.readback_age_frames = int32_t(swap_count_ - wb.enqueue_swap);
        LogResolveWriteback(record);
      }
      // Writer-audit correction (see the gpu-texture-loads design): the tiled
      // scatter — and any zero-hold re-scatter or tip refresh above — wrote
      // guest RAM through the host physical view, which the guest-mirror
      // write protection does not cover, so the shared-memory authority never
      // sees these writes on its own. Route the publication through the typed
      // task-3.2 seam using the checked destination extent
      // ([guest_addr, guest_addr + dest_extent_bytes); enqueue guarantees the
      // extent is nonzero), after ALL RAM writes for this record, so a later
      // shared-memory consumer revalidates the pages. The shared BUFFER was
      // not written — invalidation, never RangeWrittenByGpu.
      auto publication = metal::DecideRamWritePublication(
          metal::WriterKind::kResolve, wb.guest_addr, wb.dest_extent_bytes);
      if (publication) {
        MetalRamWritePublicationTarget target(
            shared_memory_.get(), texture_cache_.get(),
            metal::MetalTextureCache::InvalidationScope::kRamUploadsOnly);
        publish_outcome = metal::PublishRamWrite(*publication, target);
      }
    }
    if (publish_outcome.action !=
        metal::AuthorityAction::kInvalidateRamWrite) {
      static uint32_t s_failed_resolve_publications = 0;
      if (s_failed_resolve_publications++ < 8) {
        XELOGE(
            "MetalCommandProcessor: retaining unpublished resolve {:08X}+{} "
            "(dest={}, staging={}, shared={})",
            wb.guest_addr, wb.dest_extent_bytes, dest != nullptr,
            staging != nil, shared_memory_ != nullptr);
      }
      if (!exact_sequence) {
        still_pending.push_back(wb);
      }
      continue;
    }
    ++outcome.drained_records;
    ++outcome.resolve_records;
    id<MTLBuffer> released_staging =
        (__bridge_transfer id<MTLBuffer>)wb.staging;
    (void)released_staging;
    wb.staging = nullptr;
    retire_exact_writeback = exact_sequence;
  }
  if (retire_exact_writeback) {
    pending_writebacks_.erase(writeback_begin);
  } else if (!exact_sequence) {
    pending_writebacks_ = std::move(still_pending);
  }
  return outcome;
}

metal::ProducerDrainOutcome
MetalCommandProcessor::DrainSelectedPendingProducers(
    const metal::ConsumerDecision& decision) {
  // DecideConsumer returns selected records in their cross-kind registration
  // order. Delegate the loop/merge/stop-on-failure policy to the shared
  // selective-drain executor and, per selected record, match it back to the
  // existing queues by its unique authority sequence and reuse the established
  // convergence implementation. This adds no queue or ownership state: the selection is a
  // transient value, successful convergence still retires the original pending
  // record, and a failed convergence keeps this record and the untouched suffix
  // pending so a later retry cannot invert authority order.
  return metal::DrainSelectedProducers(
      decision, [&](const metal::PendingProducer& producer) {
        return DrainPendingWritebacks(producer.range.start(),
                                      producer.range.length(),
                                      producer.sequence);
      });
}

MetalCommandProcessor::ProducerDependencyOutcome
MetalCommandProcessor::ResolvePendingProducerDependencies(
    const metal::ConsumerDecision& decision) {
  ProducerDependencyOutcome outcome;
  if (decision.selected.empty()) {
    // Early no-op: a disjoint (or no-pending) consumer adds no commit, wait, or
    // drain.
    return outcome;
  }

  // One completion boundary for the whole consumer request. A producer in the
  // current command buffer is tagged with CurrentEncodingSubmission (or zero
  // in the diagnostic tags-off mode), so capture the real boundary before the
  // commit and prove it completed independently of the record tag.
  uint64_t boundary_submission =
      frame_command_buffer_ ? CurrentEncodingSubmission() : 0;
  EndFrameEncoderAndWait();
  uint64_t required_submission =
      std::max(boundary_submission, decision.max_wait_submission);
  if (required_submission >
      completed_submission_.load(std::memory_order_acquire)) {
    bool waited = WaitForSubmission(required_submission);
    if (!waited) {
      // Failed-wait path: the commit already happened but the drain did not.
      outcome.complete = false;
      return outcome;
    }
    PumpCompletedActions();
  }

  outcome.drain = DrainSelectedPendingProducers(decision);
  outcome.complete =
      metal::ConsumerPublicationComplete(decision, outcome.drain);
  return outcome;
}

bool MetalCommandProcessor::ResolveRamConsumerDependencies(
    metal::ConsumerKind consumer,
    const metal::GuestRangeSet& requested_ranges) {
  if (requested_ranges.empty()) {
    return true;
  }

  metal::ConsumerDecision decision = QueryPendingProducerDependencies(
      consumer, metal::ConsumerRequest::Uniform(requested_ranges));
  ProducerDependencyOutcome outcome =
      ResolvePendingProducerDependencies(decision);

  if (!outcome.complete) {
    XELOGE(
        "MetalCommandProcessor: {} RAM dependency convergence retired {} of "
        "{} selected producers",
        metal::ConsumerKindName(consumer), outcome.drain.drained_records,
        decision.selected.size());
    return false;
  }
  return true;
}

bool MetalCommandProcessor::PrepareGuestMemoryForHostRead(
    uint32_t guest_address, uint32_t length) {
  if (!length) {
    return true;
  }
  std::optional<metal::GuestRange> exact_range =
      metal::GuestRange::FromExactBaseLength(guest_address, length);
  if (!exact_range) {
    XELOGE(
        "MetalCommandProcessor: rejected out-of-domain host read {:08X}+{}",
        guest_address, length);
    return false;
  }
  metal::GuestRangeSet requested;
  requested.Add(exact_range);
  return ResolveRamConsumerDependencies(metal::ConsumerKind::kHostCpuRead,
                                        requested);
}

bool MetalCommandProcessor::PrepareGuestMemoryForCompletionExposure() {
  // Completion notifications do not identify the range the guest CPU will
  // inspect next. Publish every existing deferred producer before exposing the
  // signal; the query is still a strict no-op when the pending set is empty.
  metal::GuestRangeSet requested;
  requested.Add(metal::GuestRange::FromBaseLength(
      0, static_cast<uint32_t>(SharedMemory::kBufferSize)));
  return ResolveRamConsumerDependencies(metal::ConsumerKind::kHostCpuRead,
                                        requested);
}

bool MetalCommandProcessor::FlushPendingWritebacksForWait(
    uint32_t guest_address) {
  // A wait blocks the command stream before swap, so an intersecting deferred
  // producer must converge now. The common executor retires only the selected
  // records; disjoint resolve/memexport traffic keeps its normal swap cadence.
  metal::GuestRangeSet requested;
  requested.Add(metal::GuestRange::FromBaseLength(guest_address, 4));
  return ResolveRamConsumerDependencies(metal::ConsumerKind::kWait,
                                        requested);
}

void MetalCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr,
                                      uint32_t frontbuffer_width,
                                      uint32_t frontbuffer_height) {
  // Cycle the GPU-thread autorelease pool once per frame: drain everything
  // autoreleased while encoding the frame just ended (command buffers,
  // encoders, pass descriptors, label strings — the thread has no run-loop
  // pool, so without this they accumulate for the whole session; a multi-hour
  // session leaked to system-wide memory pressure). Pop-then-push so this
  // swap's own allocations land in the fresh pool.
  if (frame_autorelease_pool_) {
    objc_autoreleasePoolPop(frame_autorelease_pool_);
  }
  frame_autorelease_pool_ = objc_autoreleasePoolPush();
  if (shared_texture_cache_) {
    // Base-class per-frame reset so a failed creation is retried rather than
    // sticking as a permanently invalid binding.
    shared_texture_cache_->BeginFrame();
  }
  // Diagnostic: XE_METAL_PEEK=addr1,addr2,... (hex) logs the first guest-RAM
  // words at each address — ground truth for what a buffer SHOULD hold (traces
  // carry the previous frame's memory; our backend never writes guest RAM).
  if (const char* peek = std::getenv("XE_METAL_PEEK")) {
    static bool s_peeked = false;
    if (!s_peeked) {
      s_peeked = true;
      char* dup = strdup(peek);
      for (char* tok = strtok(dup, ","); tok; tok = strtok(nullptr, ",")) {
        uint32_t raw_address = uint32_t(strtoul(tok, nullptr, 16));
        uint32_t physical_address = xenos::CpuToGpu(raw_address);
        bool base_in_range =
            metal::GuestRange::FromExactBaseLength(physical_address, 16)
                .has_value();
        bool base_ready =
            base_in_range &&
            PrepareGuestMemoryForHostRead(physical_address, 16);
        uint64_t middle_address_64 =
            uint64_t(physical_address) + 0x60000u;
        bool middle_in_range =
            middle_address_64 <= UINT32_MAX &&
            metal::GuestRange::FromExactBaseLength(
                static_cast<uint32_t>(middle_address_64), 16)
                .has_value();
        uint32_t middle_address = static_cast<uint32_t>(middle_address_64);
        bool middle_ready =
            !middle_in_range ||
            PrepareGuestMemoryForHostRead(middle_address, 16);
        if (!base_ready || !middle_ready) {
          XELOGE("MetalPeek {:08X}: failed to converge diagnostic range",
                 physical_address);
          continue;
        }
        const uint32_t* p = reinterpret_cast<const uint32_t*>(
            memory_->TranslatePhysical(physical_address));
        if (p) {
          // Middle of the buffer too (offset +0x60000 = row ~300 of a 1280-wide
          // 32bpp image) to skip letterbox/border rows.
          const uint32_t* q =
              middle_in_range
                  ? reinterpret_cast<const uint32_t*>(
                        memory_->TranslatePhysical(middle_address))
                  : nullptr;
          XELOGI(
              "MetalPeek {:08X}: {:08X} {:08X} {:08X} {:08X} | +60000: {:08X} "
              "{:08X} {:08X} {:08X}",
              physical_address, p[0], p[1], p[2], p[3], q ? q[0] : 0,
              q ? q[1] : 0, q ? q[2] : 0, q ? q[3] : 0);
        } else {
          XELOGI("MetalPeek {:08X}: unmapped", physical_address);
        }
      }
      free(dup);
    }
  }
  // Flush the frame's draws + resolve blits before presenting. Unconditional:
  // when the frame's LAST GPU work is a resolve (or a skipped draw), the render
  // encoder is already closed (frame_open_ false) but the frame command buffer
  // still holds every draw and snapshot blit — gating the commit on frame_open_
  // silently discarded the whole frame (uniform magenta present).
  // The swap is the frame's ONE synchronous point (S8): the writeback drain
  // below reads staging buffers, and the wait doubles as frame pacing.
  EndFrameEncoderAndWait();

  // Black-flicker forensics: a swap with no draws since the previous one
  // presents stale/black content — log which swaps do that (throttled).
  if (!frame_has_render_) {
    static uint32_t s_empty_swaps = 0;
    ++s_empty_swaps;
    if (s_empty_swaps <= 16 || (s_empty_swaps % 256) == 0) {
      XELOGW("MetalCommandProcessor: swap #{} with NO rendered draws (#{})",
             swap_count_, s_empty_swaps);
    }
  }

  // Publish every deferred producer to guest RAM before present. Route the
  // swap's full-domain publication through the same chronological authority
  // decision/selected-record executor every other consumer uses, rather than
  // the cross-kind grouped unfiltered DrainPendingWritebacks() (which converged
  // ALL memexports before ALL resolves regardless of registration order). For
  // overlapping records that inversion could publish a newer memexport and then
  // let an older resolve overwrite it; the selective drain walks the records in
  // authority-sequence order so the newest overlapping writer always lands
  // last. Its stop-on-first-failed-retirement policy retains a failed record
  // and the untouched later suffix instead of retiring a newer record over an
  // older still-pending predecessor. The request spans the whole 512 MiB guest
  // physical domain (buffer offset == guest physical address), so full
  // publication intent is preserved: every pending record intersects and is
  // selected. The frame's single synchronous EndFrameEncoderAndWait above
  // already completed every submission, so this adds no wait for a disjoint or
  // empty pending set. kHostCpuRead expresses a RAM-publication boundary, so it
  // adds no shared-buffer revalidation and no second queue.
  metal::GuestRangeSet swap_publication_domain;
  swap_publication_domain.Add(metal::GuestRange::FromBaseLength(
      0, static_cast<uint32_t>(metal::GuestRange::kGuestPhysicalSize)));
  XE_UNLIKELY_IF(!ResolveRamConsumerDependencies(
      metal::ConsumerKind::kHostCpuRead, swap_publication_domain)) {
    // Fail closed: full-domain publication did not converge every selected
    // producer (the shared executor's stop-on-first-failed-retirement policy
    // halted, or a wait failed). Presenting now would expose stale RAM for the
    // failed record and its untouched suffix while both stay pending. Suppress
    // this frame's present and leave the pending set intact so the next swap
    // republishes the whole overlap together -- the same retry the shared
    // fail-closed contract gives every other RAM consumer. Reset per-frame
    // render state exactly like the no-presenter/no-queue early exits below so
    // the empty-swap forensics and next frame stay consistent.
    XELOGE(
        "MetalCommandProcessor: swap #{} suppressed present -- full-domain RAM "
        "publication did not fully converge; retrying next frame",
        swap_count_);
    frame_has_render_ = false;
    // A latched cache clear must NOT run on a frame whose guest data failed to
    // land: releasing now would discard the records the retry depends on. The
    // request stays latched for the next swap.
    ExecuteDeferredCacheClear(/*publication_failed=*/true);
    return;
  }

  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    frame_has_render_ = false;
    // Publication converged and the device is quiesced, so a latched clear is
    // eligible even though this frame never presents (headless boundaries).
    ExecuteDeferredCacheClear(/*publication_failed=*/false);
    return;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue()) {
    frame_has_render_ = false;
    ExecuteDeferredCacheClear(/*publication_failed=*/false);
    return;
  }
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)provider->command_queue();

  uint32_t count = swap_count_++;
  // Frame-cost telemetry: commits per frame is the frame-split count (each is
  // a full commit + waitUntilCompleted stall — hazard flushes, captures,
  // writeback waits). Logged every 128 swaps with the wall-clock frame rate.
  {
    static uint32_t s_last_commits = 0;
    static uint32_t s_last_draws = 0;
    static std::chrono::steady_clock::time_point s_last_time =
        std::chrono::steady_clock::now();
    if ((count & 127) == 0 && count) {
      auto now = std::chrono::steady_clock::now();
      float secs =
          std::chrono::duration<float>(now - s_last_time).count();
      // Memory-leak telemetry: physical footprint every stats tick — a
      // steadily climbing number is a leak long before the system pauses.
      double footprint_mb = ProcessFootprintMiB();
      XELOGI(
          "MetalFrameStats swap#{}: {:.1f} fps, {:.2f} commits/frame, "
          "{:.0f} draws/frame, {:.0f} MB footprint, {:.1f} tex-uploads/frame, "
          "slots {}/{} (free {}, stale {})",
          count, 128.0f / std::max(secs, 0.001f),
          float(frame_commit_count_ - s_last_commits) / 128.0f,
          float(draw_count_ - s_last_draws) / 128.0f, footprint_mb,
          texture_cache_ ? float(texture_cache_->ConsumeUploadCount()) / 128.0f
                         : 0.0f,
          texture_cache_ ? texture_cache_->slots_used() : 0,
          texture_cache_ ? texture_cache_->slots_capacity() : 0,
          texture_cache_ ? texture_cache_->slots_free() : 0,
          texture_cache_ ? texture_cache_->slots_stale() : 0);
      s_last_commits = frame_commit_count_;
      s_last_draws = draw_count_;
      s_last_time = now;
    }
  }
  // Cycle-transfer instrument (XE_METAL_DUMP_SNAPSHOT=<hexbase>): dump the
  // resolve snapshot for that base at EVERY swap to
  // /tmp/xe_snap_<base>_<n>.ppm — across --replay_loops this yields the
  // texel series v_n whose pairwise map v_{n+1}=f(v_n) IS the temporal
  // cycle's transfer function (the oscillator hunt).
  static const uint32_t s_dump_snapshot = [] {
    const char* v = std::getenv("XE_METAL_DUMP_SNAPSHOT");
    return v ? uint32_t(strtoul(v, nullptr, 16)) : 0u;
  }();
  if (s_dump_snapshot && render_target_cache_) {
    void* snap = render_target_cache_->GetResolveSnapshot(s_dump_snapshot);
    if (snap) {
      static uint32_t s_snap_n = 0;
      id<MTLTexture> st = (__bridge id<MTLTexture>)snap;
      uint32_t tw = uint32_t(st.width), th = uint32_t(st.height);
      id<MTLBuffer> rb =
          [st.device newBufferWithLength:size_t(tw) * th * 4
                                 options:MTLResourceStorageModeShared];
      auto* provider = static_cast<xe::ui::metal::MetalProvider*>(
          graphics_system_->provider());
      if (rb && provider && provider->command_queue()) {
        id<MTLCommandQueue> dq =
            (__bridge id<MTLCommandQueue>)provider->command_queue();
        id<MTLCommandBuffer> dcb = [dq commandBuffer];
        id<MTLBlitCommandEncoder> dbl = [dcb blitCommandEncoder];
        [dbl copyFromTexture:st
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(tw, th, 1)
                     toBuffer:rb
            destinationOffset:0
       destinationBytesPerRow:tw * 4
     destinationBytesPerImage:size_t(tw) * th * 4];
        [dbl endEncoding];
        [dcb commit];
        [dcb waitUntilCompleted];
        std::string path = fmt::format("/tmp/xe_snap_{:08X}_{}.ppm",
                                       s_dump_snapshot, s_snap_n++);
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
          fprintf(f, "P6\n%u %u\n255\n", tw, th);
          const uint8_t* px = static_cast<const uint8_t*>(rb.contents);
          for (size_t i = 0; i < size_t(tw) * th; ++i) {
            fwrite(px + i * 4, 1, 3, f);
          }
          fclose(f);
        }
      }
    }
  }
  id<MTLTexture> source_rt = nil;
  uint32_t present_width = 1280;
  uint32_t present_height = 720;
  // Prefer the snapshot the composite pass resolved into the frontbuffer base:
  // it's the guaranteed final image (the live shared RT may have been re-cleared
  // by a HUD pass drawn after the composite resolve).
  void* fb_snapshot = render_target_cache_
                          ? render_target_cache_->GetResolveSnapshot(frontbuffer_ptr)
                          : nullptr;
  // Some games assemble the frontbuffer with adjacent column resolves. The
  // per-base snapshot only contains the first column (F.E.A.R.: 768x720),
  // while stage-3 writebacks place every column at its exact address in guest
  // RAM. Presenting that partial snapshot crops / magnifies the frame. When
  // RAM is authoritative and the snapshot doesn't cover the requested guest
  // extent, prefer a full-size RAM frontbuffer upload.
  uint32_t requested_present_width =
      frontbuffer_width ? frontbuffer_width : 1280;
  uint32_t requested_present_height =
      frontbuffer_height ? frontbuffer_height : 720;
  bool partial_ram_backed_snapshot = false;
  bool canonical_ram_backed_snapshot = false;
  if (fb_snapshot && texture_cache_) {
    id<MTLTexture> snapshot_texture = (__bridge id<MTLTexture>)fb_snapshot;
    bool snapshot_covers_frontbuffer =
        snapshot_texture.width >= requested_present_width &&
        snapshot_texture.height >= requested_present_height;
    partial_ram_backed_snapshot =
        !snapshot_covers_frontbuffer &&
        !texture_cache_->BridgeServes(frontbuffer_ptr,
                                      requested_present_width,
                                      requested_present_height);
    // A canonical color snapshot is the resolve destination representation:
    // copy_dest_swap is already packed into its channels. That representation
    // is correct for texture fetches (their guest swizzle decodes it), but it
    // is not directly displayable. Once its writeback has drained, the RAM
    // frontbuffer path is the shared decode oracle: it reverses the recorded
    // endian and copy_dest_swap before presenting. Legacy snapshots do not
    // bake copy_dest_swap into the texture, so they retain the direct path.
    // Only trust the canonical RAM layout when it was recorded for the base's
    // CURRENT snapshot generation. ram_color_layouts_ entries are never erased,
    // so a base once resolved as canonical color but re-registered this frame
    // by a different (legacy/depth) path would otherwise select the canonical
    // RAM decode over the correct snapshot, presenting stale/mis-decoded bytes.
    auto layout_gen_it = ram_color_layouts_.find(frontbuffer_ptr);
    if (render_target_cache_->sample_edram_raster_enabled() &&
        layout_gen_it != ram_color_layouts_.end()) {
      auto gen_it = resolve_snapshot_generation_.find(frontbuffer_ptr);
      canonical_ram_backed_snapshot =
          gen_it != resolve_snapshot_generation_.end() &&
          layout_gen_it->second.generation == gen_it->second;
    }
  }
  bool prefer_ram_frontbuffer =
      partial_ram_backed_snapshot || canonical_ram_backed_snapshot;
  // Diagnostic: XE_METAL_PRESENT_BASE=<hex guest base> presents that resolved
  // snapshot instead of the frontbuffer, to inspect an intermediate scene RT
  // (G-buffer / pre-tonemap / bloom) via the dumped image.
  if (const char* dbg_base = std::getenv("XE_METAL_PRESENT_BASE")) {
    uint32_t base = uint32_t(std::strtoul(dbg_base, nullptr, 16));
    void* snap = render_target_cache_
                     ? render_target_cache_->GetResolveSnapshot(base)
                     : nullptr;
    if (snap) {
      source_rt = (__bridge id<MTLTexture>)snap;
      present_width = uint32_t(source_rt.width);
      present_height = uint32_t(source_rt.height);
      XELOGI("MetalCommandProcessor: DEBUG presenting snapshot base {:08X} ({}x{})",
             base, present_width, present_height);
      // Read back a few texels so values are exact, not eyeballed.
      id<MTLDevice> dev = source_rt.device;
      uint32_t rb_w = uint32_t(source_rt.width);
      id<MTLBuffer> rb =
          [dev newBufferWithLength:rb_w * 4 options:MTLResourceStorageModeShared];
      if (rb) {
        id<MTLCommandBuffer> rcb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> rbl = [rcb blitCommandEncoder];
        [rbl copyFromTexture:source_rt
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, source_rt.height / 2, 0)
                   sourceSize:MTLSizeMake(rb_w, 1, 1)
                     toBuffer:rb
            destinationOffset:0
       destinationBytesPerRow:rb_w * 4
     destinationBytesPerImage:rb_w * 4];
        [rbl endEncoding];
        [rcb commit];
        [rcb waitUntilCompleted];
        const uint32_t* words = static_cast<const uint32_t*>(rb.contents);
        const float* floats = reinterpret_cast<const float*>(rb.contents);
        XELOGI(
            "MetalCommandProcessor: DEBUG row[h/2] raw x0={:08X} x64={:08X} "
            "x320={:08X} x640={:08X} x1000={:08X} asf x320={} x640={}",
            words[0], words[64 % rb_w], words[320 % rb_w], words[640 % rb_w],
            words[1000 % rb_w], floats[320 % rb_w], floats[640 % rb_w]);
      }
    } else {
      XELOGW("MetalCommandProcessor: DEBUG present base {:08X} has no snapshot",
             base);
    }
  }
  // Which branch chose the present source — a transient black BLIP (distinct
  // from the exposure spiral) is a swap presenting a different/stale source
  // than its neighbors (fb snapshot missing for one frame -> fallback target,
  // or the fb snapshot pointer flipping to a freshly-cleared generation). The
  // MetalFlicker probe logs this at collapse events to fingerprint the blip.
  const char* present_source_path = "none";
  if (source_rt) {
    // already selected by the debug override
    present_source_path = "debug-base";
  } else if (fb_snapshot && !prefer_ram_frontbuffer) {
    source_rt = (__bridge id<MTLTexture>)fb_snapshot;
    // Use the snapshot's real extent so the present blit can't overrun it.
    present_width = uint32_t(source_rt.width);
    present_height = uint32_t(source_rt.height);
    present_source_path = "fb-snapshot";
  } else if (prefer_ram_frontbuffer) {
    void* ram_fb = GetRamFrontbufferTexture(
        frontbuffer_ptr, requested_present_width, requested_present_height);
    if (ram_fb) {
      source_rt = (__bridge id<MTLTexture>)ram_fb;
      present_width = uint32_t(source_rt.width);
      present_height = uint32_t(source_rt.height);
      present_source_path = canonical_ram_backed_snapshot
                                ? "ram-frontbuffer-canonical-snapshot"
                                : "ram-frontbuffer-partial-snapshot";
    } else {
      // Keep the old partial presentation as a last-resort fallback rather
      // than dropping the frame entirely if the RAM upload can't be made.
      source_rt = (__bridge id<MTLTexture>)fb_snapshot;
      present_width = uint32_t(source_rt.width);
      present_height = uint32_t(source_rt.height);
      present_source_path = "fb-snapshot-ram-fallback-failed";
    }
  } else if (frame_has_render_ && render_target_cache_ &&
             render_target_cache_->presentable_color_target()) {
    // Pre-resolve fallback (splash/menus before any frontbuffer snapshot): the
    // most recent BGRA8 color target drawn into. Present the guest frontbuffer
    // extent, not the (taller, shadow-map-capable) texture extent.
    source_rt = (__bridge id<MTLTexture>)
                    render_target_cache_->presentable_color_target();
    present_width = frontbuffer_width ? frontbuffer_width : 1280;
    present_height = frontbuffer_height ? frontbuffer_height : 720;
    present_source_path = "presentable-fallback";
  } else {
    // No snapshot AND nothing rendered this frame: the frontbuffer content
    // lives only in guest RAM (CPU-decoded FMV) — the RequestSwapTexture
    // pattern. Upload and present it instead of the test pattern.
    void* ram_fb = GetRamFrontbufferTexture(
        frontbuffer_ptr, frontbuffer_width ? frontbuffer_width : 1280,
        frontbuffer_height ? frontbuffer_height : 720);
    if (ram_fb) {
      source_rt = (__bridge id<MTLTexture>)ram_fb;
      present_width = uint32_t(source_rt.width);
      present_height = uint32_t(source_rt.height);
      present_source_path = "ram-frontbuffer";
    }
  }
  if (count == 0 || (count % 60) == 0) {
    XELOGI(
        "MetalCommandProcessor: IssueSwap #{} (RT {}x{}, guest fb {}x{}, ptr "
        "{:08X}, {}; draws {} skipped {}; geo-draws {} geo-pipeline-fails {})",
        count, present_width, present_height, frontbuffer_width,
        frontbuffer_height, frontbuffer_ptr,
        source_rt ? present_source_path : "test-pattern", draw_count_,
        draw_skip_count_, geo_draw_count_, geo_pipeline_fail_count_);
  }

  // MetalFlicker probe (testing framework): async center-strip readback of
  // the presented texture; each swap reads the PREVIOUS swap's strip (a full
  // frame of latency guarantees completion without a wait) and logs sudden
  // luminance collapses/recoveries — timestamps the black frames so they can
  // be correlated with hazard flushes / resolves / writebacks in the log.
  if (source_rt && source_rt.pixelFormat == MTLPixelFormatRGBA8Unorm) {
    static id<MTLBuffer> s_luma_rb = nil;
    static uint32_t s_luma_w = 0;
    static bool s_luma_pending = false;
    static float s_last_luma = -1.0f;
    static uint32_t s_flicker_logs = 0;
    // Identity of the frame whose strip is currently pending readback (set when
    // its strip was blitted a swap ago) — so a collapse names the BLIP's source.
    static const void* s_measured_src = nullptr;
    static const char* s_measured_path = "none";
    static uint32_t s_measured_fb = 0;
    static uint32_t s_measured_swap = 0;
    if (s_luma_pending && s_luma_rb) {
      const uint8_t* px = static_cast<const uint8_t*>(s_luma_rb.contents);
      uint64_t sum = 0;
      uint32_t n = s_luma_w * 4;
      for (uint32_t i = 0; i < n; ++i) {
        sum += px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2];
      }
      float luma = float(sum) / (3.0f * n);
      // Exposure-loop telemetry: one line per swap with all three loop stages
      // — luminance we wrote back, exposure constant the game fed back into
      // the tonemap, luma that came out. NOTE: luma lags one swap (async strip
      // readback); align by shifting the luma column when analyzing.
      static const bool s_expo_trace =
          std::getenv("XE_METAL_EXPO_TRACE") != nullptr;
      if (s_expo_trace) {
        XELOGI(
            "MetalExpo swap#{}: wb={:.6f} ({}w) expc=({:.5f},{:.5f},{:.5f},"
            "{:.5f}){} luma={:.1f}",
            count, expo_last_wb_value_, expo_last_wb_words_,
            expo_tonemap_c0_[0], expo_tonemap_c0_[1], expo_tonemap_c0_[2],
            expo_tonemap_c0_[3], expo_tonemap_seen_ ? "" : "[stale]", luma);
        expo_last_wb_words_ = 0;
        expo_tonemap_seen_ = false;
      }
      if (s_last_luma >= 0.0f && s_flicker_logs < 64 &&
          ((s_last_luma > 8.0f && luma < 0.35f * s_last_luma) ||
           (luma > 8.0f && s_last_luma < 0.35f * luma))) {
        ++s_flicker_logs;
        // Report the present source of the collapsed frame: a transient blip
        // shows up as a source pointer/path that differs from the good frames
        // (missing fb snapshot -> fallback, or a freshly-cleared snapshot).
        XELOGW(
            "MetalFlicker swap#{}: luma {:.1f} -> {:.1f} | blip source "
            "path={} tex={} fb={:08X} (measured swap#{})",
            count, s_last_luma, luma, s_measured_path, s_measured_src,
            s_measured_fb, s_measured_swap);
      }
      s_last_luma = luma;
    }
    uint32_t w = uint32_t(source_rt.width);
    uint32_t h = uint32_t(source_rt.height);
    if (w >= 64 && h >= 8) {
      if (!s_luma_rb || s_luma_w != w) {
        s_luma_rb = [source_rt.device
            newBufferWithLength:size_t(w) * 4 * 4
                        options:MTLResourceStorageModeShared];
        s_luma_w = w;
      }
      if (s_luma_rb) {
        // This swap's source becomes the identity measured next iteration.
        s_measured_src = (__bridge const void*)source_rt;
        s_measured_path = present_source_path;
        s_measured_fb = frontbuffer_ptr;
        s_measured_swap = count;
        id<MTLCommandBuffer> lcb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> lbl = [lcb blitCommandEncoder];
        [lbl copyFromTexture:source_rt
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, h / 2 - 2, 0)
                   sourceSize:MTLSizeMake(w, 4, 1)
                     toBuffer:s_luma_rb
            destinationOffset:0
       destinationBytesPerRow:w * 4
     destinationBytesPerImage:size_t(w) * 4 * 4];
        [lbl endEncoding];
        [lcb commit];
        s_luma_pending = true;
      }
    }
  }

  // Chain-trace probe (XE_METAL_CHAIN_TRACE=hex,hex,...): per-swap content
  // fingerprint of the listed guest-base snapshots — a center 4-pixel strip
  // of each, read back async (one swap of lag, same pattern as the luma
  // probe). With constant exposure constants but an alternating presented
  // frame (the MetalExpo finding), walking these fingerprints upstream finds
  // the FIRST pipeline stage whose content alternates — the flicker origin.
  {
    static std::vector<uint32_t> s_chain_bases = [] {
      std::vector<uint32_t> bases;
      if (const char* env = std::getenv("XE_METAL_CHAIN_TRACE")) {
        char* dup = strdup(env);
        for (char* tok = strtok(dup, ","); tok; tok = strtok(nullptr, ",")) {
          bases.push_back(uint32_t(strtoul(tok, nullptr, 16)));
        }
        free(dup);
      }
      return bases;
    }();
    if (!s_chain_bases.empty() && render_target_cache_) {
      static id<MTLBuffer> s_chain_rb = nil;
      static bool s_chain_pending = false;
      constexpr uint32_t kPixelsPerBase = 4;
      const uint32_t stride = kPixelsPerBase * 4;
      if (s_chain_pending && s_chain_rb) {
        const uint32_t* words = static_cast<const uint32_t*>(s_chain_rb.contents);
        std::string line;
        for (size_t i = 0; i < s_chain_bases.size(); ++i) {
          line += fmt::format(" {:08X}={:08X},{:08X},{:08X},{:08X}",
                              s_chain_bases[i], words[i * kPixelsPerBase],
                              words[i * kPixelsPerBase + 1],
                              words[i * kPixelsPerBase + 2],
                              words[i * kPixelsPerBase + 3]);
        }
        XELOGI("MetalChain swap#{}:{}", count, line);
      }
      if (!s_chain_rb) {
        id<MTLDevice> dev = (__bridge id<MTLDevice>)
            static_cast<xe::ui::metal::MetalProvider*>(
                graphics_system_->provider())->device();
        s_chain_rb =
            [dev newBufferWithLength:s_chain_bases.size() * stride
                             options:MTLResourceStorageModeShared];
      }
      if (s_chain_rb) {
        id<MTLCommandBuffer> ccb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> cbl = [ccb blitCommandEncoder];
        cbl.label = @"cp:chain-trace";
        for (size_t i = 0; i < s_chain_bases.size(); ++i) {
          void* snap = render_target_cache_->GetResolveSnapshot(s_chain_bases[i]);
          if (!snap) {
            continue;
          }
          id<MTLTexture> tex = (__bridge id<MTLTexture>)snap;
          uint32_t cx = uint32_t(tex.width) / 2, cy = uint32_t(tex.height) / 2;
          if (cx + kPixelsPerBase > tex.width) {
            cx = 0;
          }
          [cbl copyFromTexture:tex
                    sourceSlice:0
                    sourceLevel:0
                   sourceOrigin:MTLOriginMake(cx, cy, 0)
                     sourceSize:MTLSizeMake(kPixelsPerBase, 1, 1)
                       toBuffer:s_chain_rb
              destinationOffset:i * stride
         destinationBytesPerRow:stride
       destinationBytesPerImage:stride];
        }
        [cbl endEncoding];
        [ccb commit];
        s_chain_pending = true;
      }
    }
  }

  double t = double(count % 120) / 120.0;
  double pattern_r = t, pattern_g = 1.0 - t, pattern_b = 0.5;

  auto refresher =
      [&](ui::Presenter::GuestOutputRefreshContext& context) -> bool {
    auto& metal_context =
        static_cast<xe::ui::metal::MetalGuestOutputRefreshContext&>(context);
    id<MTLTexture> mailbox =
        (__bridge id<MTLTexture>)metal_context.target_texture();
    if (!mailbox) {
      return false;
    }
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (source_rt && source_rt.pixelFormat != mailbox.pixelFormat) {
      // Debug visualization (XE_METAL_PRESENT_BASE of a non-BGRA8 snapshot,
      // e.g. an R32Float depth snapshot): a blit between different formats is
      // invalid, render grayscale (.r) into the mailbox instead.
      static void* s_gray_pipeline = nullptr;   // id<MTLRenderPipelineState>.
      static void* s_convert_pipeline = nullptr;
      if (!s_gray_pipeline) {
        id<MTLDevice> dev = mailbox.device;
        NSString* src = kMetalPresentConvertShader;
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil
                                                 error:&err];
        if (lib) {
          MTLRenderPipelineDescriptor* d =
              [[MTLRenderPipelineDescriptor alloc] init];
          d.vertexFunction = [lib newFunctionWithName:@"xe_gray_vs"];
          d.fragmentFunction = [lib newFunctionWithName:@"xe_gray_fs"];
          d.colorAttachments[0].pixelFormat = mailbox.pixelFormat;
          id<MTLRenderPipelineState> ps =
              [dev newRenderPipelineStateWithDescriptor:d error:&err];
          if (ps) {
            s_gray_pipeline = (__bridge_retained void*)ps;
          }
          d.fragmentFunction = [lib newFunctionWithName:@"xe_convert_fs"];
          id<MTLRenderPipelineState> cps =
              [dev newRenderPipelineStateWithDescriptor:d error:&err];
          if (cps) {
            s_convert_pipeline = (__bridge_retained void*)cps;
          }
        }
      }
      const bool use_convert =
          cvars::metal_present_format_convert && s_convert_pipeline;
      if (s_gray_pipeline || use_convert) {
        MTLRenderPassDescriptor* pass =
            [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = mailbox;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
        id<MTLRenderCommandEncoder> enc =
            [command_buffer renderCommandEncoderWithDescriptor:pass];
        [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)(
                                        use_convert ? s_convert_pipeline
                                                    : s_gray_pipeline)];
        [enc setFragmentTexture:source_rt atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:3];
        [enc endEncoding];
      }
    } else if (source_rt) {
      // Standard 256-entry guest gamma ramp, applied AFTER the swap source has
      // been selected (source_rt is already chosen above, by either the
      // resolved-snapshot or the RAM/CPU-decoded FMV path) and BEFORE the
      // presenter's scaling and UI composition, which happen later against the
      // mailbox. Both sources therefore reach the identical ramp stage, and no
      // UI pixel can be touched by it.
      //
      // Shape: blit the selected source into a private pre-ramp copy exactly as
      // before, then render that copy into the mailbox through the ramp. The
      // mailbox has no ShaderWrite usage, so it cannot be read-modify-written
      // in place, and rewriting the blit itself would have coupled the ramp to
      // swap-source selection.
      bool gamma_applied = false;
      if (cvars::metal_guest_gamma_ramp) {
        id<MTLDevice> gamma_device = mailbox.device;
        // Ramp buffer: 256 dwords, exactly the guest register layout.
        if (!gamma_ramp_buffer_) {
          id<MTLBuffer> buf =
              [gamma_device newBufferWithLength:kGammaRampTableBytes
                                        options:MTLResourceStorageModeShared];
          if (buf) {
            buf.label = @"cp:gamma-ramp-256";
            gamma_ramp_buffer_ = (__bridge_retained void*)buf;
          }
        }
        if (!gamma_ramp_pipeline_) {
          NSError* err = nil;
          id<MTLLibrary> lib =
              [gamma_device newLibraryWithSource:kMetalGammaRampShader
                                         options:nil
                                           error:&err];
          if (lib) {
            MTLRenderPipelineDescriptor* d =
                [[MTLRenderPipelineDescriptor alloc] init];
            d.vertexFunction = [lib newFunctionWithName:@"xe_gamma_vs"];
            d.fragmentFunction = [lib newFunctionWithName:@"xe_gamma_fs"];
            d.colorAttachments[0].pixelFormat = mailbox.pixelFormat;
            id<MTLRenderPipelineState> ps =
                [gamma_device newRenderPipelineStateWithDescriptor:d
                                                             error:&err];
            if (ps) {
              gamma_ramp_pipeline_ = (__bridge_retained void*)ps;
            } else {
              XELOGE("MetalGamma: ramp pipeline creation failed: {}",
                     err ? err.localizedDescription.UTF8String : "unknown");
            }
          }
        }
        // Pre-ramp copy, resized with the guest output.
        if (gamma_ramp_source_texture_ &&
            (gamma_ramp_source_width_ != present_width ||
             gamma_ramp_source_height_ != present_height)) {
          id<MTLTexture> old_tex =
              (__bridge_transfer id<MTLTexture>)gamma_ramp_source_texture_;
          (void)old_tex;
          gamma_ramp_source_texture_ = nullptr;
        }
        if (!gamma_ramp_source_texture_) {
          MTLTextureDescriptor* td = [MTLTextureDescriptor
              texture2DDescriptorWithPixelFormat:mailbox.pixelFormat
                                           width:present_width
                                          height:present_height
                                       mipmapped:NO];
          td.usage = MTLTextureUsageShaderRead;
          td.storageMode = MTLStorageModePrivate;
          id<MTLTexture> tex = [gamma_device newTextureWithDescriptor:td];
          if (tex) {
            tex.label = @"cp:gamma-pre-ramp-source";
            gamma_ramp_source_texture_ = (__bridge_retained void*)tex;
            gamma_ramp_source_width_ = present_width;
            gamma_ramp_source_height_ = present_height;
          }
        }
        id<MTLBuffer> ramp_buffer = (__bridge id<MTLBuffer>)gamma_ramp_buffer_;
        id<MTLTexture> pre_ramp =
            (__bridge id<MTLTexture>)gamma_ramp_source_texture_;
        id<MTLRenderPipelineState> ramp_pipeline =
            (__bridge id<MTLRenderPipelineState>)gamma_ramp_pipeline_;
        if (ramp_buffer && pre_ramp && ramp_pipeline &&
            source_rt.pixelFormat == mailbox.pixelFormat) {
          // BUFFER LIFETIME INVARIANT: this refresher commits and then
          // waitUntilCompleted's before returning (see the end of this lambda),
          // and nothing else writes the ramp buffer. So no command buffer that
          // reads the ramp can still be in flight when the next swap mutates
          // it, and a single immutable-per-frame buffer is sufficient -- no
          // ring, no per-frame versioning. The assertion below fails loudly if
          // a future change removes that wait.
          assert_false(gamma_ramp_upload_in_flight_);
          if (gamma_ramp_dirty_) {
            std::memcpy(ramp_buffer.contents, gamma_ramp_256_entry_table(),
                        kGammaRampTableBytes);
            gamma_ramp_dirty_ = false;
            ++gamma_ramp_uploads_;
          }
          id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
          [blit copyFromTexture:source_rt
                    sourceSlice:0
                    sourceLevel:0
                   sourceOrigin:MTLOriginMake(0, 0, 0)
                     sourceSize:MTLSizeMake(present_width, present_height, 1)
                      toTexture:pre_ramp
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
          [blit endEncoding];
          MTLRenderPassDescriptor* pass =
              [MTLRenderPassDescriptor renderPassDescriptor];
          pass.colorAttachments[0].texture = mailbox;
          pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
          pass.colorAttachments[0].storeAction = MTLStoreActionStore;
          id<MTLRenderCommandEncoder> enc =
              [command_buffer renderCommandEncoderWithDescriptor:pass];
          [enc setRenderPipelineState:ramp_pipeline];
          [enc setFragmentTexture:pre_ramp atIndex:0];
          [enc setFragmentBuffer:ramp_buffer offset:0 atIndex:0];
          [enc drawPrimitives:MTLPrimitiveTypeTriangle
                  vertexStart:0
                  vertexCount:3];
          [enc endEncoding];
          gamma_ramp_upload_in_flight_ = true;
          ++gamma_ramp_applications_;
          gamma_applied = true;
        }
      }
      if (!gamma_applied) {
        // Exact rollback path: byte-for-byte the pre-change behavior.
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
        [blit copyFromTexture:source_rt
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(present_width, present_height, 1)
                    toTexture:mailbox
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
      }
    } else {
      MTLRenderPassDescriptor* pass =
          [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = mailbox;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      pass.colorAttachments[0].clearColor =
          MTLClearColorMake(pattern_r, pattern_g, pattern_b, 1.0);
      id<MTLRenderCommandEncoder> encoder =
          [command_buffer renderCommandEncoderWithDescriptor:pass];
      [encoder endEncoding];
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    // The wait above is what makes a single ramp buffer safe; clearing here
    // ties the invariant to the wait rather than to a comment.
    gamma_ramp_upload_in_flight_ = false;
    return true;
  };
  presenter->RefreshGuestOutput(present_width, present_height, /*aspect_x=*/16,
                                /*aspect_y=*/9, refresher);
  frame_has_render_ = false;

  // The frame boundary, and the Metal analogue of D3D12 executing its deferred
  // clear inside EndSubmission when the frame closes. Everything the ordering
  // contract requires has already happened for this frame: the encoder ended,
  // every submission completed and its completion actions ran
  // (EndFrameEncoderAndWait above), all pending memexports and render-target
  // writebacks were published to guest RAM, and the presenter's own command
  // buffer committed and was waited on inside the refresher. A frame that
  // never requests a clear reaches only the leading branch of this call.
  ExecuteDeferredCacheClear(/*publication_failed=*/false);
}

// Compile this backend's own copy of the PM4 execute loop, statically bound to
// MetalCommandProcessor so its Issue*/LoadShader overrides are the ones called
// (mirrors the vulkan/d3d12 backends; included inside the namespace).
#define COMMAND_PROCESSOR MetalCommandProcessor
#include "../pm4_command_processor_implement.h"

}  // namespace metal
}  // namespace gpu
}  // namespace xe
