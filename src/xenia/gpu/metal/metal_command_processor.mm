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
#include <cstring>
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
    default:
      // Fans, quads, rectangles, patches need primitive-processor conversion or
      // geometry emulation — skipped in the bring-up slice.
      return false;
  }
}

bool IsPolygonalPrimitive(xenos::PrimitiveType type) {
  return type == xenos::PrimitiveType::kTriangleList ||
         type == xenos::PrimitiveType::kTriangleStrip ||
         type == xenos::PrimitiveType::kTriangleFan;
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
  upload_arena_->Reset();
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)provider->command_queue();
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  MTLRenderPassDescriptor* pass =
      [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)rt;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  // Dark green base: if a draw lands nothing, we still see "our RT".
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.20, 0.08, 1.0);
  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:pass];
  [encoder setCullMode:MTLCullModeNone];
  frame_command_buffer_ = (__bridge_retained void*)command_buffer;
  frame_render_encoder_ = (__bridge_retained void*)encoder;
  frame_open_ = true;
  return frame_render_encoder_;
}

void MetalCommandProcessor::EndFrameEncoder() {
  if (!frame_open_) {
    return;
  }
  id<MTLRenderCommandEncoder> encoder =
      (__bridge_transfer id<MTLRenderCommandEncoder>)frame_render_encoder_;
  id<MTLCommandBuffer> command_buffer =
      (__bridge_transfer id<MTLCommandBuffer>)frame_command_buffer_;
  frame_render_encoder_ = nullptr;
  frame_command_buffer_ = nullptr;
  frame_open_ = false;
  [encoder endEncoding];
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
  // A skipped draw is not an error (the base loop treats false as failure); we
  // return true and just render nothing so the game keeps advancing.
  MTLPrimitiveType mtl_prim;
  if (!MapPrimitiveType(prim_type, &mtl_prim)) {
    ++draw_skip_count_;
    return true;
  }
  auto* vertex_shader = static_cast<MetalShader*>(active_vertex_shader());
  auto* pixel_shader = static_cast<MetalShader*>(active_pixel_shader());
  if (!vertex_shader || !pipeline_cache_ || !upload_arena_) {
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

  // Translate to MTLFunctions (cached per modification).
  auto sq_program_cntl = regs.Get<reg::SQ_PROGRAM_CNTL>();
  uint64_t vs_mod =
      pipeline_cache_->shader_translator()->GetDefaultVertexShaderModification(
          vertex_shader->GetDynamicAddressableRegisterCount(
              sq_program_cntl.vs_num_reg),
          Shader::HostVertexShaderType::kVertex);
  auto* vs_translation = static_cast<MetalShader::MetalTranslation*>(
      vertex_shader->GetOrCreateTranslation(vs_mod));
  if (!pipeline_cache_->TranslateShader(*vs_translation)) {
    ++draw_skip_count_;
    return true;
  }
  MetalShader::MetalTranslation* ps_translation = nullptr;
  if (pixel_shader) {
    DxbcShaderTranslator::Modification ps_mod_decoded(
        pipeline_cache_->shader_translator()->GetDefaultPixelShaderModification(
            pixel_shader->GetDynamicAddressableRegisterCount(
                sq_program_cntl.ps_num_reg)));
    // The RT0 MIN/MAX-blend premultiply factors default to kZero in a zeroed
    // modification, which forces RT0 rgb/a to 0 (pure black) in the non-ROV
    // output path. The neutral value is kOne (no premultiply); MIN/MAX blend
    // handling comes with the blend state later.
    ps_mod_decoded.pixel.rt0_blend_rgb_factor_for_premult =
        xenos::BlendFactor::kOne;
    ps_mod_decoded.pixel.rt0_blend_a_factor_for_premult =
        xenos::BlendFactor::kOne;
    uint64_t ps_mod = ps_mod_decoded.value;
    ps_translation = static_cast<MetalShader::MetalTranslation*>(
        pixel_shader->GetOrCreateTranslation(ps_mod));
    if (!pipeline_cache_->TranslateShader(*ps_translation)) {
      ps_translation = nullptr;
    }
  }

  void* pipeline;
  if (kDebugSolidFragment) {
    pipeline = GetDebugSolidPipeline(
        static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider())
            ->device(),
        vs_translation->metal_function(), debug_solid_fragment_,
        debug_solid_pipelines_);
  } else {
    pipeline = pipeline_cache_->GetOrCreateRenderPipelineState(
        vs_translation, ps_translation, MetalRenderTargetCache::kColorFormat,
        MTLPixelFormatInvalid);
  }
  if (!pipeline) {
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
    vp_args.Setup(1, 1, divisors::MagicDiv(1), divisors::MagicDiv(1),
                  /*origin_bottom_left=*/true, 1280, 720,
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
    // Diagnostic: dump the position transform + first vfetch for the first
    // indexed triangle-list draws (the "real geometry" ones), regardless of
    // when they occur, to catch off-screen / degenerate output.
    if (index_buffer_info && prim_type == xenos::PrimitiveType::kTriangleList &&
        tri_log_count_ < 12) {
      ++tri_log_count_;
      xenos::xe_gpu_vertex_fetch_t vf0{};
      for (uint32_t i = 0; i < xe::countof(vs_map.vertex_fetch_bitmap); ++i) {
        uint32_t b = vs_map.vertex_fetch_bitmap[i];
        uint32_t j;
        if (xe::bit_scan_forward(b, &j)) {
          vf0 = regs.GetVertexFetch(i * 32 + j);
          break;
        }
      }
      // Vertex float-constant count + first used vec4, to verify b1 is sane
      // (not zero/garbage, which would compute black / off-screen).
      uint32_t vs_float_count = 0;
      for (uint32_t i = 0; i < 4; ++i) {
        vs_float_count += xe::bit_count(vs_map.float_bitmap[i]);
      }
      float first_vec4[4] = {0, 0, 0, 0};
      for (uint32_t i = 0; i < 4; ++i) {
        uint32_t idx;
        uint64_t bm = vs_map.float_bitmap[i];
        if (xe::bit_scan_forward(bm, &idx)) {
          std::memcpy(
              first_vec4,
              &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (i << 8) + (idx << 2)],
              sizeof(first_vec4));
          break;
        }
      }
      XELOGI(
          "MetalCommandProcessor: TRI {} ext=({},{}) ndc_s=({},{},{}) "
          "ndc_o=({},{},{}) flags={:X} ps_xlat={} vf0 addr={:08X} size={} "
          "vs_floats={} first=({},{},{},{})",
          tri_log_count_, viewport_info.xy_extent[0], viewport_info.xy_extent[1],
          viewport_info.ndc_scale[0], viewport_info.ndc_scale[1],
          viewport_info.ndc_scale[2], viewport_info.ndc_offset[0],
          viewport_info.ndc_offset[1], viewport_info.ndc_offset[2],
          system_constants.flags, ps_translation != nullptr, vf0.address << 2,
          vf0.size << 2, vs_float_count, first_vec4[0], first_vec4[1],
          first_vec4[2], first_vec4[3]);
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
  [encoder setVertexBuffer:ab_buffer
                    offset:ab_alloc.offset
                   atIndex:kIRArgumentBufferBindPoint];
  [encoder setFragmentBuffer:ab_buffer
                      offset:ab_alloc.offset
                     atIndex:kIRArgumentBufferBindPoint];
  [encoder setVertexBuffer:view_heap offset:0 atIndex:kIRDescriptorHeapBindPoint];
  [encoder setFragmentBuffer:view_heap
                      offset:0
                     atIndex:kIRDescriptorHeapBindPoint];
  [encoder setVertexBuffer:sampler_heap
                    offset:0
                   atIndex:kIRSamplerHeapBindPoint];
  [encoder setFragmentBuffer:sampler_heap
                      offset:0
                     atIndex:kIRSamplerHeapBindPoint];

  // Residency for everything the shaders reach through gpuAddresses.
  id<MTLBuffer> shared_buffer =
      (__bridge id<MTLBuffer>)shared_memory_->GetBuffer();
  id<MTLTexture> null_texture = (__bridge id<MTLTexture>)null_texture_;
  MTLRenderStages stages = MTLRenderStageVertex | MTLRenderStageFragment;
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

bool MetalCommandProcessor::IssueCopy() {
  // Resolve/copy is still a stub (no EDRAM->guest resolve). Count invocations so
  // we can tell whether the game relies on resolved render targets (which would
  // explain black surfaces sampled as textures with all-zero guest memory).
  static uint32_t s_copy_count = 0;
  if ((s_copy_count++ % 300) == 0) {
    XELOGI("MetalCommandProcessor::IssueCopy stub invoked (count={})",
             s_copy_count);
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
  if (frame_has_render_ && render_target_cache_ &&
      render_target_cache_->color_target()) {
    source_rt = (__bridge id<MTLTexture>)render_target_cache_->color_target();
    present_width = render_target_cache_->width();
    present_height = render_target_cache_->height();
  }
  if (count == 0 || (count % 60) == 0) {
    XELOGI(
        "MetalCommandProcessor: IssueSwap #{} ({}x{}, ptr {:08X}, {}; draws {} "
        "skipped {})",
        count, present_width, present_height, frontbuffer_ptr,
        source_rt ? "render target" : "test pattern", draw_count_,
        draw_skip_count_);
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
