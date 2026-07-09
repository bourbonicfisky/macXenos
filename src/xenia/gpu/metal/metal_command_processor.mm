/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_command_processor.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

// Headers required by the PM4 execute-loop template included at the bottom of
// this file (mirrors the generic set vulkan_command_processor.cc pulls in).
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/profiling.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_texture_cache.h"
#include "xenia/gpu/metal/metal_upload_arena.h"
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

namespace xe {
namespace gpu {
namespace metal {

namespace {

// Top-level argument buffer: 32 uint64 gpuAddress entries, one per root-signature
// parameter, in the exact order MetalShaderConverter::CreateXbox360RootSignature
// appends them.
constexpr uint32_t kTopLevelABSlots = 32;
enum AbSlot : uint32_t {
  kAbSrvSpace0 = 0,   // shared memory SRV (t0, space0)
  kAbSrvSpace1 = 1,
  kAbSrvSpace2 = 2,
  kAbSrvSpace3 = 3,
  kAbSrvSpace10 = 4,
  kAbUavSpace0 = 5,
  kAbUavSpace1 = 6,
  kAbUavSpace2 = 7,
  kAbUavSpace3 = 8,
  kAbSamplerSpace0 = 9,
  kAbCbvSystem = 10,        // b0
  kAbCbvFloatVertex = 11,   // b1 (vertex)
  kAbCbvBoolLoop = 12,      // b2
  kAbCbvFetchVertex = 13,   // b3 (vertex)
  kAbCbvDescIdxVertex = 14, // b4 (vertex)
  kAbCbvFloatHull = 15,
  kAbCbvFetchHull = 16,
  kAbCbvDescIdxHull = 17,
  kAbCbvFloatDomain = 18,
  kAbCbvFetchDomain = 19,
  kAbCbvDescIdxDomain = 20,
  kAbCbvFloatPixel = 21,    // b1 (pixel)
  kAbCbvFetchPixel = 22,    // b3 (pixel)
  kAbCbvDescIdxPixel = 23,  // b4 (pixel)
  kAbCbvFunctionConstants = 24,
};

// Entries inside the small fixed system view table.
constexpr uint32_t kSystemViewSharedMemorySRV = 0;
constexpr uint32_t kSystemViewNullUAV = 1;
constexpr uint32_t kSystemViewEntryCount = 2;

// Sized for bring-up; a real build would grow these on demand.
constexpr uint32_t kViewBindlessHeapSlots = 4096;
constexpr uint32_t kSamplerBindlessHeapSlots = 256;
constexpr uint32_t kUploadArenaSize = 32u * 1024 * 1024;

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
      // Quads, line loops, patches need primitive-processor conversion or
      // geometry emulation — skipped in the bring-up slice.
      return false;
  }
}

bool IsPolygonalPrimitive(xenos::PrimitiveType type) {
  return type == xenos::PrimitiveType::kTriangleList ||
         type == xenos::PrimitiveType::kTriangleStrip ||
         type == xenos::PrimitiveType::kTriangleFan ||
         type == xenos::PrimitiveType::kRectangleList;
}

// Byte-swaps guest indices into a host (little-endian) copy for Metal.
void SwapIndicesInto(void* dst, const void* src, uint32_t count,
                     xenos::IndexFormat format, xenos::Endian endian) {
  if (format == xenos::IndexFormat::kInt16) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    uint16_t* d = static_cast<uint16_t*>(dst);
    bool swap = endian == xenos::Endian::k8in16 ||
                endian == xenos::Endian::k8in32;
    for (uint32_t i = 0; i < count; ++i) {
      d[i] = swap ? xe::byte_swap(s[i]) : s[i];
    }
  } else {
    const uint32_t* s = static_cast<const uint32_t*>(src);
    uint32_t* d = static_cast<uint32_t*>(dst);
    for (uint32_t i = 0; i < count; ++i) {
      d[i] = endian == xenos::Endian::kNone ? s[i] : xe::byte_swap(s[i]);
    }
  }
}

}  // namespace

MetalCommandProcessor::MetalCommandProcessor(
    MetalGraphicsSystem* graphics_system, kernel::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state) {}
MetalCommandProcessor::~MetalCommandProcessor() = default;

void MetalCommandProcessor::TracePlaybackWroteMemory(uint32_t base_ptr,
                                                     uint32_t length) {}

void MetalCommandProcessor::RestoreEdramSnapshot(const void* snapshot) {}

bool MetalCommandProcessor::SetupContext() {
  if (!CommandProcessor::SetupContext()) {
    return false;
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
  render_target_cache_ = std::make_unique<MetalRenderTargetCache>(device);
  shared_memory_ = std::make_unique<MetalSharedMemory>(device, memory_);
  if (!shared_memory_->Initialize()) {
    XELOGE("MetalCommandProcessor: shared memory init failed");
    return false;
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

  upload_arena_ = std::make_unique<MetalUploadArena>(device_ptr, kUploadArenaSize);
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
  system_view_tables_ = (__bridge_retained void*)system_view;

  texture_cache_ = std::make_unique<MetalTextureCache>(
      device_ptr, memory_, view_bindless_heap_, kViewBindlessHeapSlots);
  return true;
}

void MetalCommandProcessor::ShutdownDrawResources() {
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
  upload_arena_.reset();
}

void MetalCommandProcessor::ShutdownContext() {
  if (frame_open_) {
    EndFrameEncoder();
  }
  ShutdownDrawResources();
  shared_memory_.reset();
  render_target_cache_.reset();
  pipeline_cache_.reset();
  CommandProcessor::ShutdownContext();
}

// Depth buffer is GATED OFF for now. When enabled it clears depth to 1.0 (far)
// assuming standard 0..1 depth, but many Xbox 360 titles use reverse-z (clear 0,
// "greater" test), so the fixed clear discards everything and 2D/menus go black.
// Re-enable only once the guest depth-clear value + z-direction are emulated
// (read from the guest's depth clear, not assumed). Keeps the pipeline's depth
// attachment format and the render pass consistent (both on or both off).
static constexpr bool kDepthEnabled = true;

void* MetalCommandProcessor::BeginFrameEncoder() {
  if (frame_open_) {
    return frame_render_encoder_;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue() || !render_target_cache_) {
    return nullptr;
  }
  void* rt = render_target_cache_->EnsureColorTarget(1280, 720);
  if (!rt) {
    return nullptr;
  }
  void* depth =
      kDepthEnabled ? render_target_cache_->EnsureDepthTarget(1280, 720) : nullptr;
  // The command buffer lives for the whole frame; create it (and reset the
  // per-frame upload arena) on the first encoder open of the frame.
  id<MTLCommandBuffer> command_buffer;
  bool new_frame = frame_command_buffer_ == nullptr;
  if (new_frame) {
    upload_arena_->Reset();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)provider->command_queue();
    command_buffer = [queue commandBuffer];
    frame_command_buffer_ = (__bridge_retained void*)command_buffer;
  } else {
    command_buffer = (__bridge id<MTLCommandBuffer>)frame_command_buffer_;
  }
  // Clear at the start of a frame and after every resolve (a pass boundary);
  // otherwise load so the passes' draws accumulate into the RT.
  // Only clear on a genuine new frame; on mid-frame reopens LOAD the retained
  // shared-RT content. JC2's final HUD pass re-renders into the already-composited
  // front-buffer base and expects to draw OVER the resolved scene; clearing at
  // every resolve boundary (the old reopen_clear_ behavior) threw the whole scene
  // away, leaving only the clear color + HUD in the present. Loading fixes that.
  // KNOWN LIMITATION: this is blunt — passes that render UNRELATED targets into
  // the shared RT now over-accumulate (mild bleed on some frames). The correct
  // model honors the guest's actual clears via EDRAM-tile<->RAM-base tracking;
  // until that lands, load-and-accumulate is the net-better default. reopen_clear_
  // is retained (still cleared below) for when that model arrives.
  bool clear = new_frame;
  reopen_clear_ = false;
  MTLRenderPassDescriptor* pass =
      [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)rt;
  pass.colorAttachments[0].loadAction =
      clear ? MTLLoadActionClear : MTLLoadActionLoad;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  // Dark green base: if a draw lands nothing, we still see "our RT".
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.20, 0.08, 1.0);
  // Depth attachment: cleared at each pass boundary to the GUEST's depth-clear
  // value (RB_DEPTH_CLEAR, 24-bit depth in the high bits), not a fixed 1.0.
  // Reverse-z titles (JC2 menus) clear to 0 and test "greater"; a hardcoded 1.0
  // made every fragment fail z>=1.0 -> black. Reading the guest value serves both
  // directions.
  if (depth) {
    uint32_t rb_depth_clear = (*register_file_)[XE_GPU_REG_RB_DEPTH_CLEAR];
    double clear_depth_value = double(rb_depth_clear >> 8) / double(0xFFFFFF);
    pass.depthAttachment.texture = (__bridge id<MTLTexture>)depth;
    pass.depthAttachment.loadAction =
        clear ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    pass.depthAttachment.clearDepth = clear_depth_value;
  }
  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:pass];
  [encoder setCullMode:MTLCullModeNone];
  frame_render_encoder_ = (__bridge_retained void*)encoder;
  frame_open_ = true;
  return frame_render_encoder_;
}

void MetalCommandProcessor::EndRenderEncoder() {
  if (!frame_open_) {
    return;
  }
  id<MTLRenderCommandEncoder> encoder =
      (__bridge_transfer id<MTLRenderCommandEncoder>)frame_render_encoder_;
  frame_render_encoder_ = nullptr;
  frame_open_ = false;
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
  reopen_clear_ = false;
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
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

  // Rectangle lists (JC2's deferred composite/lighting/post passes are true
  // 4-corner rects) can't be drawn as plain triangles; expand them via MSC
  // geometry emulation, which needs a mesh-shader-capable device and a pixel
  // shader for the fragment stage. Point lists can join this path later; for now
  // only rect-list is routed here (everything else still goes through the plain
  // primitive map, which skips unsupported types).
  PipelineGeometryShader geometry_type = PipelineGeometryShader::kNone;
  if (prim_type == xenos::PrimitiveType::kRectangleList) {
    geometry_type = PipelineGeometryShader::kRectangleList;
  }
  bool use_geometry_emulation = geometry_type != PipelineGeometryShader::kNone &&
                                mesh_shader_supported_ && pixel_shader != nullptr;

  // A skipped draw is not an error (the base loop treats false as failure); we
  // return true and just render nothing so the game keeps advancing.
  MTLPrimitiveType mtl_prim = MTLPrimitiveTypeTriangle;
  if (!use_geometry_emulation && !MapPrimitiveType(prim_type, &mtl_prim)) {
    ++draw_skip_count_;
    return true;
  }
  RegisterFile& regs = *register_file_;

  // Analyze microcode (once) so constant maps / bindings are available.
  StringBuffer ucode_disasm;
  if (!vertex_shader->is_ucode_analyzed()) {
    vertex_shader->AnalyzeUcode(ucode_disasm);
  }
  if (pixel_shader && !pixel_shader->is_ucode_analyzed()) {
    pixel_shader->AnalyzeUcode(ucode_disasm);
  }

  // DIAGNOSTIC (MRT-vs-single-RT proof, openspec reduce-apple-gpu-frametime):
  // log each unique render-target configuration once — the active color-target
  // set (normalized mask folds RB_MODECONTROL + RB_COLOR_MASK + the PS's static
  // writes) with each RT's EDRAM tile base + format, plus the bound depth
  // base/format. Decides whether the multi-target RT cache must bind multiple
  // simultaneous attachments (true MRT) or distinct per-pass single targets.
  {
    uint32_t normalized_color_mask =
        pixel_shader ? draw_util::GetNormalizedColorMask(
                           regs, pixel_shader->writes_color_targets())
                     : 0;
    reg::RB_DEPTHCONTROL norm_depth = draw_util::GetNormalizedDepthControl(regs);
    bool depth_bound = norm_depth.z_enable || norm_depth.stencil_enable;
    auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
    // Unique-config key: color mask + each active RT's info + depth info.
    uint64_t cfg_key = normalized_color_mask;
    uint32_t color_rt_count = 0;
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (!(normalized_color_mask & (0b1111u << (4 * i)))) continue;
      ++color_rt_count;
      cfg_key = cfg_key * 1000003u ^
                regs[reg::RB_COLOR_INFO::rt_register_indices[i]];
    }
    cfg_key = cfg_key * 1000003u ^
              (depth_bound ? (uint64_t(rb_depth_info.value) | (1ull << 32)) : 0);
    static std::unordered_set<uint64_t> s_seen_rt_configs;
    if (s_seen_rt_configs.insert(cfg_key).second) {
      std::string rt_desc;
      for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
        if (!(normalized_color_mask & (0b1111u << (4 * i)))) continue;
        auto ci = regs.Get<reg::RB_COLOR_INFO>(
            reg::RB_COLOR_INFO::rt_register_indices[i]);
        rt_desc += fmt::format(" RT{}[base={} fmt={} mask={:X}]", i,
                               uint32_t(ci.color_base),
                               uint32_t(ci.color_format),
                               (normalized_color_mask >> (4 * i)) & 0b1111);
      }
      XELOGI(
          "MetalRTConfig: colors={}{}{} depth={} (base={} fmt={}) ncm={:04X}",
          color_rt_count, rt_desc.empty() ? " (none)" : "", rt_desc,
          depth_bound, uint32_t(rb_depth_info.depth_base),
          uint32_t(rb_depth_info.depth_format), normalized_color_mask);
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
    ps_mod_decoded.pixel.interpolators_centroid =
        interpolator_mask &
        ~xenos::GetInterpolatorSamplingPattern(
            regs.Get<reg::RB_SURFACE_INFO>().msaa_samples,
            regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
            regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);
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
  // Geometry-emulation config (VS output size + GS max input primitives per mesh
  // threadgroup), filled by the geometry pipeline and passed to the draw call.
  uint32_t gs_vertex_size = 0;
  uint32_t gs_max_input_primitives = 0;
  if (kDebugSolidFragment) {
    pipeline = GetDebugSolidPipeline(
        static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider())
            ->device(),
        vs_translation->metal_function(), debug_solid_fragment_,
        debug_solid_pipelines_);
  } else if (use_geometry_emulation) {
    uint32_t blend_control = regs[XE_GPU_REG_RB_BLENDCONTROL0];
    uint32_t color_write_mask = regs.Get<reg::RB_COLOR_MASK>().value & 0b1111;
    pipeline = pipeline_cache_->GetOrCreateGeometryPipelineState(
        vs_translation, ps_translation, geometry_type,
        DxbcShaderTranslator::Modification(vs_mod),
        DxbcShaderTranslator::Modification(ps_mod_value),
        MetalRenderTargetCache::kColorFormat,
        kDepthEnabled ? MetalRenderTargetCache::kDepthFormat
                      : uint64_t(MTLPixelFormatInvalid),
        blend_control, color_write_mask, &gs_vertex_size,
        &gs_max_input_primitives);
  } else {
    uint32_t blend_control = regs[XE_GPU_REG_RB_BLENDCONTROL0];
    uint32_t color_write_mask = regs.Get<reg::RB_COLOR_MASK>().value & 0b1111;
    pipeline = pipeline_cache_->GetOrCreateRenderPipelineState(
        vs_translation, ps_translation, MetalRenderTargetCache::kColorFormat,
        kDepthEnabled ? MetalRenderTargetCache::kDepthFormat
                      : uint64_t(MTLPixelFormatInvalid),
        blend_control, color_write_mask);
  }
  if (!pipeline) {
    if (use_geometry_emulation) {
      ++geo_pipeline_fail_count_;
    }
    ++draw_skip_count_;
    return true;
  }

  id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)BeginFrameEncoder();
  if (!encoder) {
    ++draw_skip_count_;
    return true;
  }

  // Upload the vertex data ranges referenced by the vertex shader's vfetches.
  const Shader::ConstantRegisterMap& vs_map =
      vertex_shader->constant_register_map();
  for (uint32_t i = 0; i < xe::countof(vs_map.vertex_fetch_bitmap); ++i) {
    uint32_t bits = vs_map.vertex_fetch_bitmap[i];
    uint32_t j;
    while (xe::bit_scan_forward(bits, &j)) {
      bits = xe::clear_lowest_bit(bits);
      xenos::xe_gpu_vertex_fetch_t fetch = regs.GetVertexFetch(i * 32 + j);
      if (fetch.type != xenos::FetchConstantType::kVertex &&
          fetch.type != xenos::FetchConstantType::kInvalidVertex) {
        continue;
      }
      shared_memory_->UploadRange(fetch.address << 2, fetch.size << 2);
    }
  }

  // --- Marshal the constant buffers (b0-b4), matching the D3D12 backend. ---
  MetalUploadAllocation system_alloc{};
  DxbcShaderTranslator::SystemConstants system_constants;
  std::memset(&system_constants, 0, sizeof(system_constants));
  {
    // Viewport -> ndc_scale/offset (position-critical: zeroing collapses output).
    draw_util::ViewportInfo viewport_info;
    draw_util::GetViewportInfoArgs vp_args;
    // x_max/y_max is the maximum host viewport extent, NOT the render-target
    // size. Both reference backends pass the device max (D3D12:
    // D3D12_VIEWPORT_BOUNDS_MAX, Vulkan: maxViewportDimensions) and let the
    // render target clip an oversized guest viewport. Passing 1280x720 here
    // instead made GetHostViewportInfo *rescale* (zoom-crop) any guest viewport
    // wider/taller than the RT into it — the splash-screen "zoom". Use the max
    // guest RT extent (8192, already the clip-disabled cap and within Metal's
    // 16384 viewport limit) so guest content beyond 1280x720 is clipped, not
    // scaled.
    vp_args.Setup(1, 1, divisors::MagicDiv(1), divisors::MagicDiv(1),
                  /*origin_bottom_left=*/true,
                  xenos::kTexture2DCubeMaxWidthHeight,
                  xenos::kTexture2DCubeMaxWidthHeight,
                  /*allow_reverse_z=*/true, regs.Get<reg::RB_DEPTHCONTROL>(),
                  /*convert_z_to_float24=*/false,
                  /*full_float24_in_0_to_1=*/false,
                  /*pixel_shader_writes_depth=*/false);
    vp_args.SetupRegisterValues(regs);
    draw_util::GetHostViewportInfo(&vp_args, viewport_info);
    for (uint32_t i = 0; i < 3; ++i) {
      system_constants.ndc_scale[i] = viewport_info.ndc_scale[i];
      system_constants.ndc_offset[i] = viewport_info.ndc_offset[i];
    }
    // Position-relevant flags.
    auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
    uint32_t flags = 0;
    if (pa_cl_vte_cntl.vtx_xy_fmt) flags |= DxbcShaderTranslator::kSysFlag_XYDividedByW;
    if (pa_cl_vte_cntl.vtx_z_fmt) flags |= DxbcShaderTranslator::kSysFlag_ZDividedByW;
    if (pa_cl_vte_cntl.vtx_w0_fmt) flags |= DxbcShaderTranslator::kSysFlag_WNotReciprocal;
    if (IsPolygonalPrimitive(prim_type)) {
      flags |= DxbcShaderTranslator::kSysFlag_PrimitivePolygonal;
    }
    if (prim_type == xenos::PrimitiveType::kLineList ||
        prim_type == xenos::PrimitiveType::kLineStrip) {
      flags |= DxbcShaderTranslator::kSysFlag_PrimitiveLine;
    }
    // Alpha test: the compare function is a 3-bit (less|equal|greater) mask in
    // the flags. Leaving it 0 means kNever -> the pixel shader discards EVERY
    // fragment. When alpha test is disabled it must be kAlways (pass all).
    auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
    xenos::CompareFunction alpha_test_function =
        rb_colorcontrol.alpha_test_enable ? rb_colorcontrol.alpha_func
                                          : xenos::CompareFunction::kAlways;
    flags |= uint32_t(alpha_test_function)
             << DxbcShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;
    system_constants.flags = flags;
    system_constants.alpha_test_reference =
        regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
    // Color exponent bias multiplies the pixel output; its neutral value is
    // 1.0 (2^0), NOT 0 -- a zeroed struct here would multiply every color by
    // zero and render pure black. (Full per-RT bias from RB_COLOR_INFO later.)
    for (uint32_t i = 0; i < 4; ++i) {
      system_constants.color_exp_bias[i] = 1.0f;
    }
    // Index/vfetch helpers used by the vertex shader.
    xenos::Endian index_endian =
        index_buffer_info ? index_buffer_info->endianness : xenos::Endian::kNone;
    system_constants.vertex_index_endian = index_endian;
    system_constants.vertex_index_offset = regs[XE_GPU_REG_VGT_INDX_OFFSET];
    system_constants.vertex_index_min = regs[XE_GPU_REG_VGT_MIN_VTX_INDX];
    system_constants.vertex_index_max = regs[XE_GPU_REG_VGT_MAX_VTX_INDX];

    // Set the host viewport from the integer part.
    if (viewport_info.xy_extent[0] && viewport_info.xy_extent[1]) {
      MTLViewport vp;
      vp.originX = viewport_info.xy_offset[0];
      vp.originY = viewport_info.xy_offset[1];
      vp.width = viewport_info.xy_extent[0];
      vp.height = viewport_info.xy_extent[1];
      vp.znear = viewport_info.z_min;
      vp.zfar = viewport_info.z_max;
      [encoder setViewport:vp];
    }
    // Bring-up diagnostic: log the transform of the first few draws of the
    // session (any primitive type) to catch off-screen / degenerate output.
    if (tri_log_count_ < 12) {
      ++tri_log_count_;
      XELOGI(
          "MetalCommandProcessor: DRAW {} prim={} idx={} clipdis={} imask={:X} "
          "pgen={} ext=({},{}) ndc_s=({},{},{}) ndc_o=({},{},{}) flags={:X} "
          "ps_xlat={}",
          tri_log_count_, uint32_t(prim_type), index_buffer_info != nullptr,
          uint32_t(regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable),
          interpolator_mask, ps_param_gen_pos, viewport_info.xy_extent[0],
          viewport_info.xy_extent[1], viewport_info.ndc_scale[0],
          viewport_info.ndc_scale[1], viewport_info.ndc_scale[2],
          viewport_info.ndc_offset[0], viewport_info.ndc_offset[1],
          viewport_info.ndc_offset[2], system_constants.flags,
          ps_translation != nullptr);
    }
  }
  if (!upload_arena_->Allocate(sizeof(system_constants), 256, &system_alloc)) {
    ++draw_skip_count_;
    return true;
  }
  std::memcpy(system_alloc.cpu_ptr, &system_constants, sizeof(system_constants));

  // Float constants, packed per the shader's float_bitmap (b1). Vertex + pixel.
  auto marshal_floats = [&](const Shader::ConstantRegisterMap& map,
                            uint32_t reg_base,
                            MetalUploadAllocation* out) -> bool {
    uint32_t count = 0;
    for (uint32_t i = 0; i < 4; ++i) count += xe::bit_count(map.float_bitmap[i]);
    uint32_t bytes = sizeof(float) * 4 * std::max(count, 1u);
    if (!upload_arena_->Allocate(bytes, 256, out)) return false;
    uint8_t* dst = static_cast<uint8_t*>(out->cpu_ptr);
    for (uint32_t i = 0; i < 4; ++i) {
      uint64_t entry = map.float_bitmap[i];
      uint32_t index;
      while (xe::bit_scan_forward(entry, &index)) {
        entry = xe::clear_lowest_bit(entry);
        std::memcpy(dst, &regs[reg_base + (i << 8) + (index << 2)],
                    4 * sizeof(float));
        dst += 4 * sizeof(float);
      }
    }
    return true;
  };
  MetalUploadAllocation float_vertex_alloc{};
  if (!marshal_floats(vs_map, XE_GPU_REG_SHADER_CONSTANT_000_X,
                      &float_vertex_alloc)) {
    ++draw_skip_count_;
    return true;
  }
  MetalUploadAllocation float_pixel_alloc{};
  if (pixel_shader) {
    if (!marshal_floats(pixel_shader->constant_register_map(),
                        XE_GPU_REG_SHADER_CONSTANT_256_X, &float_pixel_alloc)) {
      ++draw_skip_count_;
      return true;
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
  if (!marshal_block(XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031,
                     (8 + 32) * sizeof(uint32_t), &bool_loop_alloc) ||
      !marshal_block(XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0,
                     32 * 6 * sizeof(uint32_t), &fetch_alloc)) {
    ++draw_skip_count_;
    return true;
  }

  // Descriptor indices (b4): for each texture binding, resolve its guest fetch
  // constant to a bindless view-heap slot (real texture, or 0=null/gray if the
  // format isn't supported yet). Samplers stay at slot 0 (the null sampler).
  texture_cache_->ResetActive();
  auto marshal_desc_indices = [&](MetalShader* shader,
                                  MetalUploadAllocation* out) -> bool {
    uint32_t count = 1;
    if (shader) {
      count = std::max<uint32_t>(
          uint32_t(shader->GetTextureBindingsAfterTranslation().size() +
                   shader->GetSamplerBindingsAfterTranslation().size()),
          1u);
    }
    if (!upload_arena_->Allocate(count * sizeof(uint32_t), 256, out)) {
      return false;
    }
    auto* indices = static_cast<uint32_t*>(out->cpu_ptr);
    std::memset(indices, 0, count * sizeof(uint32_t));
    if (shader) {
      for (const auto& binding : shader->GetTextureBindingsAfterTranslation()) {
        if (binding.bindless_descriptor_index < count) {
          indices[binding.bindless_descriptor_index] =
              texture_cache_->GetTextureSlot(
                  regs.GetTextureFetch(binding.fetch_constant));
        }
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

  // --- Build the top-level argument buffer (32 gpuAddresses). ---
  id<MTLBuffer> view_heap = (__bridge id<MTLBuffer>)view_bindless_heap_;
  id<MTLBuffer> sampler_heap = (__bridge id<MTLBuffer>)sampler_bindless_heap_;
  id<MTLBuffer> system_view = (__bridge id<MTLBuffer>)system_view_tables_;
  id<MTLBuffer> null_buffer = (__bridge id<MTLBuffer>)null_buffer_;
  uint64_t view_heap_gpu = [view_heap gpuAddress];
  uint64_t sampler_heap_gpu = [sampler_heap gpuAddress];
  uint64_t system_view_gpu = [system_view gpuAddress];
  uint64_t null_gpu = [null_buffer gpuAddress];

  MetalUploadAllocation ab_alloc{};
  if (!upload_arena_->Allocate(kTopLevelABSlots * sizeof(uint64_t), 256,
                               &ab_alloc)) {
    ++draw_skip_count_;
    return true;
  }
  uint64_t* ab = static_cast<uint64_t*>(ab_alloc.cpu_ptr);
  for (uint32_t i = 0; i < kTopLevelABSlots; ++i) ab[i] = null_gpu;
  ab[kAbSrvSpace0] =
      system_view_gpu + kSystemViewSharedMemorySRV * sizeof(IRDescriptorTableEntry);
  ab[kAbSrvSpace1] = view_heap_gpu;
  ab[kAbSrvSpace2] = view_heap_gpu;
  ab[kAbSrvSpace3] = view_heap_gpu;
  ab[kAbSrvSpace10] = view_heap_gpu;
  ab[kAbUavSpace0] =
      system_view_gpu + kSystemViewNullUAV * sizeof(IRDescriptorTableEntry);
  ab[kAbUavSpace1] = view_heap_gpu;
  ab[kAbUavSpace2] = view_heap_gpu;
  ab[kAbUavSpace3] = view_heap_gpu;
  ab[kAbSamplerSpace0] = sampler_heap_gpu;
  ab[kAbCbvSystem] = system_alloc.gpu_address;
  ab[kAbCbvFloatVertex] = float_vertex_alloc.gpu_address;
  ab[kAbCbvBoolLoop] = bool_loop_alloc.gpu_address;
  ab[kAbCbvFetchVertex] = fetch_alloc.gpu_address;
  ab[kAbCbvDescIdxVertex] = desc_idx_vertex_alloc.gpu_address;
  if (pixel_shader) {
    ab[kAbCbvFloatPixel] = float_pixel_alloc.gpu_address;
    ab[kAbCbvFetchPixel] = fetch_alloc.gpu_address;
    ab[kAbCbvDescIdxPixel] = desc_idx_pixel_alloc.gpu_address;
  }

  // --- Bind. ---
  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)pipeline];
  if (kDepthEnabled) {
    // Normalized depth control disables depth entirely when the pass isn't a
    // depth pass (edram_mode not kColorDepth/kDepthOnly) and collapses redundant
    // "always, no write" to disabled — so 2D/menu draws don't get depth-tested
    // against a buffer they never meant to use.
    void* depth_state = GetOrCreateDepthStencilState(
        draw_util::GetNormalizedDepthControl(*register_file_).value);
    if (depth_state) {
      [encoder
          setDepthStencilState:(__bridge id<MTLDepthStencilState>)depth_state];
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
  MTLRenderStages stages =
      use_geometry_emulation
          ? (MTLRenderStageObject | MTLRenderStageMesh | MTLRenderStageFragment)
          : (MTLRenderStageVertex | MTLRenderStageFragment);
  [encoder useResource:shared_buffer usage:MTLResourceUsageRead stages:stages];
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

  // --- Draw. ---
  // Rectangle-list (and later point/quad) geometry emulation: the guest VS runs
  // as the object stage and a generated GS as the mesh stage; the runtime
  // dispatches the object+mesh threadgroups. rect-list maps to a triangle input
  // topology (the GS synthesizes the 4th corner). See openspec
  // reduce-apple-gpu-frametime handoff.
  if (use_geometry_emulation) {
    IRRuntimeGeometryPipelineConfig geometry_config = {};
    geometry_config.gsVertexSizeInBytes = gs_vertex_size;
    geometry_config.gsMaxInputPrimitivesPerMeshThreadgroup =
        gs_max_input_primitives;
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
          encoder, IRRuntimePrimitiveTypeTriangle, index_type, index_buffer,
          geometry_config, /*instanceCount=*/1, index_count, start_index,
          /*baseVertex=*/0, /*baseInstance=*/0);
    } else {
      IRRuntimeDrawPrimitivesGeometryEmulation(
          encoder, IRRuntimePrimitiveTypeTriangle, geometry_config,
          /*instanceCount=*/1, index_count, /*baseVertex=*/0,
          /*baseInstance=*/0);
    }
    frame_has_render_ = true;
    ++draw_count_;
    ++geo_draw_count_;
    return true;
  }

  // Triangle fans have no host primitive on Metal: expand to a triangle list
  // with a generated 32-bit index buffer. Fan (v0, v1, v2, v3, ...) becomes
  // triangles (v0,v1,v2), (v0,v2,v3), ... indexing the same vertices, so it
  // works whether the guest draw is indexed or sequential (auto).
  if (prim_type == xenos::PrimitiveType::kTriangleFan) {
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
    frame_has_render_ = true;
    ++draw_count_;
    return true;
  }
  if (index_buffer_info && index_buffer_info->guest_base) {
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

  frame_has_render_ = true;
  ++draw_count_;
  if (draw_count_ == 1 || (draw_count_ % 2000) == 0) {
    XELOGI("MetalCommandProcessor: encoded draw #{} (prim {}, {} indices, {})",
           draw_count_, uint32_t(prim_type), index_count,
           index_buffer_info ? "indexed" : "auto");
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

// Builds (and caches) an MTLDepthStencilState from RB_DEPTHCONTROL. Stencil is
// not implemented yet; only z_enable / z_write_enable / zfunc are honored. When
// depth test is disabled the state passes everything and writes nothing.
void* MetalCommandProcessor::GetOrCreateDepthStencilState(uint32_t depth_control) {
  // Only the depth-relevant bits matter for the key (mask out stencil fields).
  reg::RB_DEPTHCONTROL dc;
  dc.value = depth_control;
  uint32_t key = (dc.z_enable ? 1u : 0u) | (dc.z_write_enable ? 2u : 0u) |
                 (uint32_t(dc.zfunc) << 2);
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
  id<MTLDepthStencilState> state =
      [device newDepthStencilStateWithDescriptor:desc];
  void* state_ptr = (__bridge_retained void*)state;
  depth_stencil_states_.emplace(key, state_ptr);
  return state_ptr;
}

bool MetalCommandProcessor::ConvertDepthToSnapshot(void* depth_texture,
                                                   void* dest_snapshot,
                                                   uint32_t width,
                                                   uint32_t height) {
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
    NSString* src = @R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      struct VOut { float4 pos [[position]]; float2 uv; };
      vertex VOut xe_depth_convert_vs(uint vid [[vertex_id]]) {
        VOut o;
        float2 t = float2(float((vid << 1) & 2), float(vid & 2));
        o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
        o.uv = float2(t.x, 1.0 - t.y);
        return o;
      }
      fragment float xe_depth_convert_fs(VOut in [[stage_in]],
                                         depth2d<float> d [[texture(0)]]) {
        constexpr sampler s(coord::normalized, filter::nearest,
                            address::clamp_to_edge);
        return d.sample(s, in.uv);
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
  pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> encoder =
      [cb renderCommandEncoderWithDescriptor:pass];
  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      depth_convert_pipeline_];
  MTLViewport vp{0.0, 0.0, double(width), double(height), 0.0, 1.0};
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
  draw_util::ResolveInfo resolve_info;
  if (!draw_util::GetResolveInfo(*register_file_, *memory_, trace_writer_,
                                 /*draw_resolution_scale_x=*/1,
                                 /*draw_resolution_scale_y=*/1,
                                 /*fixed_rg16_truncated_to_minus_1_to_1=*/false,
                                 /*fixed_rgba16_truncated_to_minus_1_to_1=*/false,
                                 resolve_info)) {
    return true;
  }
  // RT-as-texture bridge. A color resolve is the boundary of a render pass: blit
  // what we've drawn into the shared color RT out into a per-destination snapshot
  // texture and register it at the resolve's guest base, so a later deferred /
  // composite pass that fetches this base samples the real rendered pixels (its
  // guest RAM is never populated by our backend). Then flag the next pass to
  // start from a fresh clear.
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
      void* snap = render_target_cache_->EnsureResolveSnapshot(
          resolve_info.copy_dest_base, rw, rh,
          MetalRenderTargetCache::kColorFormat);
      void* color_rt = render_target_cache_->color_target();
      if (snap && color_rt) {
        uint32_t cw = render_target_cache_->width();
        uint32_t ch = render_target_cache_->height();
        uint32_t bw = rw < cw ? rw : cw;
        uint32_t bh = rh < ch ? rh : ch;
        id<MTLCommandBuffer> cb =
            (__bridge id<MTLCommandBuffer>)frame_command_buffer_;
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:(__bridge id<MTLTexture>)color_rt
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(bw, bh, 1)
                    toTexture:(__bridge id<MTLTexture>)snap
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        texture_cache_->RegisterResolvedTarget(resolve_info.copy_dest_base, snap,
                                                rw, rh);
      }
      reopen_clear_ = true;
    }
  } else if (resolve_info.IsCopyingDepth() &&
             resolve_info.copy_dest_extent_length) {
    // Depth resolve: deferred lighting samples the depth buffer as a texture to
    // reconstruct position (JC2/Sonic sample fmt 22/23 -> we returned null ->
    // black lighting). Our depth RT is Depth32Float, which Metal only samples as
    // depth2d while MSC shaders declare texture2d_array<float>; convert it to an
    // R32Float snapshot with a small fullscreen pass, and register that.
    uint32_t rw = resolve_info.coordinate_info.width_div_8 * 8;
    uint32_t rh = resolve_info.height_div_8 * 8;
    if (rw && rh && kDepthEnabled && render_target_cache_ && texture_cache_ &&
        frame_command_buffer_) {
      EndRenderEncoder();
      void* snap = render_target_cache_->EnsureResolveSnapshot(
          resolve_info.copy_dest_base, rw, rh, MTLPixelFormatR32Float);
      void* depth_rt = render_target_cache_->depth_target();
      if (snap && depth_rt && ConvertDepthToSnapshot(depth_rt, snap, rw, rh)) {
        texture_cache_->RegisterResolvedTarget(resolve_info.copy_dest_base, snap,
                                                rw, rh);
      }
      reopen_clear_ = true;
    }
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
  return true;
}

void MetalCommandProcessor::InitializeTrace() {}

void MetalCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr,
                                      uint32_t frontbuffer_width,
                                      uint32_t frontbuffer_height) {
  // Flush the frame's draws into the render target before presenting it.
  if (frame_open_) {
    EndFrameEncoder();
  }

  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    frame_has_render_ = false;
    return;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue()) {
    frame_has_render_ = false;
    return;
  }
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)provider->command_queue();

  uint32_t count = swap_count_++;
  id<MTLTexture> source_rt = nil;
  uint32_t present_width = 1280;
  uint32_t present_height = 720;
  // Prefer the snapshot the composite pass resolved into the frontbuffer base:
  // it's the guaranteed final image (the live shared RT may have been re-cleared
  // by a HUD pass drawn after the composite resolve).
  void* fb_snapshot = render_target_cache_
                          ? render_target_cache_->GetResolveSnapshot(frontbuffer_ptr)
                          : nullptr;
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
    } else {
      XELOGW("MetalCommandProcessor: DEBUG present base {:08X} has no snapshot",
             base);
    }
  }
  if (source_rt) {
    // already selected by the debug override
  } else if (fb_snapshot) {
    source_rt = (__bridge id<MTLTexture>)fb_snapshot;
    // Use the snapshot's real extent so the present blit can't overrun it.
    present_width = uint32_t(source_rt.width);
    present_height = uint32_t(source_rt.height);
  } else if (frame_has_render_ && render_target_cache_ &&
             render_target_cache_->color_target()) {
    source_rt = (__bridge id<MTLTexture>)render_target_cache_->color_target();
    present_width = render_target_cache_->width();
    present_height = render_target_cache_->height();
  }
  if (count == 0 || (count % 60) == 0) {
    XELOGI(
        "MetalCommandProcessor: IssueSwap #{} (RT {}x{}, guest fb {}x{}, ptr "
        "{:08X}, {}; draws {} skipped {}; geo-draws {} geo-pipeline-fails {})",
        count, present_width, present_height, frontbuffer_width,
        frontbuffer_height, frontbuffer_ptr,
        source_rt ? "render target" : "test pattern", draw_count_,
        draw_skip_count_, geo_draw_count_, geo_pipeline_fail_count_);
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
    if (source_rt) {
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
    return true;
  };
  presenter->RefreshGuestOutput(present_width, present_height, /*aspect_x=*/16,
                                /*aspect_y=*/9, refresher);
  frame_has_render_ = false;
}

// Compile this backend's own copy of the PM4 execute loop, statically bound to
// MetalCommandProcessor so its Issue*/LoadShader overrides are the ones called
// (mirrors the vulkan/d3d12 backends; included inside the namespace).
#define COMMAND_PROCESSOR MetalCommandProcessor
#include "../pm4_command_processor_implement.h"

}  // namespace metal
}  // namespace gpu
}  // namespace xe
