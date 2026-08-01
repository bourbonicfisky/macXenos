/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// port-xenios-tessellation task 4.5 — the on-device encode smoke:
//
// Builds a tessellation-emulation pipeline through the PRODUCTION
// MetalPipelineCache, then encodes patch draws into a REAL render command
// encoder with the FULL group-4 binding assembly — the top-level argument
// buffer from the pure builder (hull/domain slots populated), the AB bound
// again at kIRArgumentBufferHullDomainBindPoint on object+mesh, the
// tessellator tables loaded via IRRuntimeLoadTessellatorTables and bound to
// BOTH stages at kIRRuntimeTessellatorTablesBindPoint with residency marked —
// and dispatches through IRRuntimeDraw[Indexed]PatchesTessellationEmulation
// with the config rebuilt verbatim from the TessellationPipeline mirror.
// Acceptance: the command buffer completes with no error (run with Metal
// validation enabled when the environment provides it; the harness itself
// does not force MTL_SHADER_VALIDATION).

#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/metal_binding_layout.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/xenos.h"

#include "metal_irconverter_runtime.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

using metal::BuildMetalTopLevelArgumentBuffer;
using metal::kMetalBindingSlotCount;
using metal::MetalBindingAddresses;
using metal::MetalBindingSlot;
using metal::MetalBindingSlotIndex;
using metal::MetalPipelineCache;
using metal::MetalPipelineRasterKey;
using metal::MetalShader;
using HostVertexShaderType = Shader::HostVertexShaderType;

// Same nondegenerate guest vertex fixture as the pipeline-cache case.
const uint32_t kGuestVertexUcode[] = {
    0x00001002, 0x00001000, 0xC2000000, 0x00001003,
    0x00002000, 0x00000000, 0xC8020000, 0x006C6C00,
    0xC1000000, 0xC88F803E, 0x00000000, 0xC2000000};

struct SmokeHarness {
  id<MTLDevice> device = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  bool ok = false;
  std::string error;

  SmokeHarness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      error = "no Metal device";
      return;
    }
    cache = std::make_unique<MetalPipelineCache>((__bridge void*)device);
    if (!cache->Initialize()) {
      error = "MetalPipelineCache::Initialize failed";
      return;
    }
    ok = true;
  }

  MetalShader::MetalTranslation* MakeDomainTranslation(
      HostVertexShaderType host_vertex_shader_type) {
    MetalShader* shader = cache->LoadShader(
        xenos::ShaderType::kVertex, kGuestVertexUcode,
        uint32_t(sizeof(kGuestVertexUcode) / sizeof(uint32_t)));
    if (!shader->is_ucode_analyzed()) {
      StringBuffer disasm;
      shader->AnalyzeUcode(disasm);
    }
    DxbcShaderTranslator::Modification modification(
        cache->shader_translator()->GetDefaultVertexShaderModification(
            shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters),
            host_vertex_shader_type));
    return static_cast<MetalShader::MetalTranslation*>(
        shader->GetOrCreateTranslation(modification.value));
  }
};

id<MTLBuffer> NewZeroBuffer(id<MTLDevice> device, NSUInteger length,
                            NSString* label) {
  id<MTLBuffer> buffer =
      [device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if (buffer) {
    std::memset([buffer contents], 0, length);
    buffer.label = label;
  }
  return buffer;
}

}  // namespace

TEST_CASE(
    "tessellation encode smoke: full binding assembly + patch dispatch "
    "completes on-device",
    "[tessellation][encode][device]") {
  SmokeHarness h;
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  id<MTLDevice> device = h.device;

  // --- Pipeline via the production cache (triangle cell, depth-only PS). ---
  uint64_t colors[xenos::kMaxColorRenderTargets];
  uint32_t blends[xenos::kMaxColorRenderTargets] = {};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    colors[i] = uint64_t(MTLPixelFormatInvalid);
  }
  colors[0] = uint64_t(MTLPixelFormatRGBA8Unorm);
  uint64_t depth_format = uint64_t(MTLPixelFormatDepth32Float_Stencil8);
  MetalShader::MetalTranslation* tri_domain =
      h.MakeDomainTranslation(HostVertexShaderType::kTriangleDomainCPIndexed);
  REQUIRE(tri_domain != nullptr);
  const MetalPipelineCache::TessellationPipeline* tess =
      h.cache->GetOrCreateTessellationPipelineState(
          tri_domain, /*pixel=*/nullptr,
          HostVertexShaderType::kTriangleDomainCPIndexed,
          xenos::TessellationMode::kDiscrete,
          xenos::PrimitiveType::kTriangleList, colors, depth_format,
          MetalPipelineRasterKey{/*raster_sample_count=*/1}, blends,
          /*color_write_mask=*/0xF);
  REQUIRE(tess != nullptr);
  REQUIRE(tess->pipeline != nullptr);

  // --- Tessellator tables: the same allocate-once + load the CP performs. ---
  uint64_t tables_size = IRRuntimeTessellatorTablesSize();
  CHECK(tables_size > 0);
  id<MTLBuffer> tessellator_tables =
      [device newBufferWithLength:tables_size
                          options:MTLResourceStorageModeShared];
  REQUIRE(tessellator_tables != nil);
  tessellator_tables.label = @"Tessellator Tables (smoke)";
  IRRuntimeLoadTessellatorTables(tessellator_tables);

  // --- Render targets. ---
  MTLTextureDescriptor* color_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:64
                                  height:64
                               mipmapped:NO];
  color_desc.usage = MTLTextureUsageRenderTarget;
  color_desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> color_rt = [device newTextureWithDescriptor:color_desc];
  MTLTextureDescriptor* depth_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                   width:64
                                  height:64
                               mipmapped:NO];
  depth_desc.usage = MTLTextureUsageRenderTarget;
  depth_desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> depth_rt = [device newTextureWithDescriptor:depth_desc];
  REQUIRE(color_rt != nil);
  REQUIRE(depth_rt != nil);

  // --- Constant/descriptor backing, mirroring the CP's fixed resources. ---
  // System constants: factor range published exactly as the CP does
  // (VGT_HOS_{MIN,MAX}_TESS_LEVEL + 1), vertex_index_endian kNone (the D5
  // decision: the bound stream is pre-swapped, in-shader swap is identity).
  DxbcShaderTranslator::SystemConstants system_constants;
  std::memset(&system_constants, 0, sizeof(system_constants));
  system_constants.tessellation_factor_range_min = 1.0f;
  system_constants.tessellation_factor_range_max = 4.0f;
  system_constants.vertex_index_endian = xenos::Endian::kNone;
  system_constants.vertex_index_max = 0xFFFFFFu;
  id<MTLBuffer> system_buffer = NewZeroBuffer(
      device, sizeof(system_constants), @"System Constants (smoke)");
  REQUIRE(system_buffer != nil);
  std::memcpy([system_buffer contents], &system_constants,
              sizeof(system_constants));
  // One zeroed bank stands in for float/bool-loop/fetch/descriptor-indices.
  id<MTLBuffer> cbv_scratch = NewZeroBuffer(device, 4096, @"CBV (smoke)");
  id<MTLBuffer> null_buffer = NewZeroBuffer(device, 256, @"Null (smoke)");
  REQUIRE(cbv_scratch != nil);
  REQUIRE(null_buffer != nil);

  // Null texture + sampler so every descriptor-heap slot is dereferenceable.
  MTLTextureDescriptor* null_tex_desc = [[MTLTextureDescriptor alloc] init];
  null_tex_desc.textureType = MTLTextureType2DArray;
  null_tex_desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
  null_tex_desc.width = 1;
  null_tex_desc.height = 1;
  null_tex_desc.arrayLength = 1;
  null_tex_desc.usage = MTLTextureUsageShaderRead;
  null_tex_desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> null_texture = [device newTextureWithDescriptor:null_tex_desc];
  id<MTLSamplerState> null_sampler = [device
      newSamplerStateWithDescriptor:[[MTLSamplerDescriptor alloc] init]];
  REQUIRE(null_texture != nil);
  REQUIRE(null_sampler != nil);
  constexpr uint32_t kHeapSlots = 8;
  id<MTLBuffer> view_heap = NewZeroBuffer(
      device, kHeapSlots * sizeof(IRDescriptorTableEntry), @"View heap");
  id<MTLBuffer> sampler_heap = NewZeroBuffer(
      device, kHeapSlots * sizeof(IRDescriptorTableEntry), @"Sampler heap");
  id<MTLBuffer> system_views = NewZeroBuffer(
      device, 3 * sizeof(IRDescriptorTableEntry), @"System views");
  REQUIRE(view_heap != nil);
  REQUIRE(sampler_heap != nil);
  REQUIRE(system_views != nil);
  auto* view_entries =
      static_cast<IRDescriptorTableEntry*>([view_heap contents]);
  auto* sampler_entries =
      static_cast<IRDescriptorTableEntry*>([sampler_heap contents]);
  for (uint32_t i = 0; i < kHeapSlots; ++i) {
    IRDescriptorTableSetTexture(&view_entries[i], null_texture, 0.0f, 0);
    IRDescriptorTableSetSampler(&sampler_entries[i], null_sampler, 0.0f);
  }
  // Entry 0: a shared-memory-SRV stand-in; 1: null UAV; 2: UAV stand-in.
  id<MTLBuffer> guest_scratch =
      NewZeroBuffer(device, 64 * 1024, @"Guest scratch");
  REQUIRE(guest_scratch != nil);
  auto* system_view_entries =
      static_cast<IRDescriptorTableEntry*>([system_views contents]);
  IRDescriptorTableSetBuffer(&system_view_entries[0],
                             [guest_scratch gpuAddress], 64 * 1024);
  IRDescriptorTableSetBuffer(&system_view_entries[1],
                             [null_buffer gpuAddress], 256);
  IRDescriptorTableSetBuffer(&system_view_entries[2],
                             [guest_scratch gpuAddress], 64 * 1024);

  // --- The top-level argument buffer from the PURE production builder, with
  // the tessellation hull/domain aliasing engaged. ---
  MetalBindingAddresses addresses;
  addresses.null_buffer = [null_buffer gpuAddress];
  addresses.view_heap = [view_heap gpuAddress];
  addresses.sampler_heap = [sampler_heap gpuAddress];
  addresses.shared_memory_srv = [system_views gpuAddress];
  addresses.null_uav =
      [system_views gpuAddress] + sizeof(IRDescriptorTableEntry);
  addresses.shared_memory_uav =
      [system_views gpuAddress] + 2 * sizeof(IRDescriptorTableEntry);
  addresses.system_constants = [system_buffer gpuAddress];
  addresses.float_vertex = [cbv_scratch gpuAddress];
  addresses.bool_loop = [cbv_scratch gpuAddress];
  addresses.fetch = [cbv_scratch gpuAddress];
  addresses.descriptor_indices_vertex = [cbv_scratch gpuAddress];
  addresses.float_pixel = 0;
  addresses.descriptor_indices_pixel = 0;
  std::array<uint64_t, kMetalBindingSlotCount> argument_buffer =
      BuildMetalTopLevelArgumentBuffer(addresses, /*pixel_shader_present=*/
                                       false,
                                       /*memexport_draw=*/false,
                                       /*tessellation_draw=*/true);
  // The binding oracle on the real assembly: hull/domain slots populated.
  CHECK(argument_buffer[MetalBindingSlotIndex(
            MetalBindingSlot::kCbvFloatHull)] == addresses.float_vertex);
  CHECK(argument_buffer[MetalBindingSlotIndex(
            MetalBindingSlot::kCbvDescriptorIndicesDomain)] ==
        addresses.descriptor_indices_vertex);
  id<MTLBuffer> ab_buffer = NewZeroBuffer(
      device, sizeof(argument_buffer), @"Top-level AB (smoke)");
  REQUIRE(ab_buffer != nil);
  std::memcpy([ab_buffer contents], argument_buffer.data(),
              sizeof(argument_buffer));

  // Pre-swapped 16-bit index stream {0,1,2} for the indexed patch draw.
  id<MTLBuffer> index_buffer =
      NewZeroBuffer(device, 3 * sizeof(uint16_t), @"Patch indices (smoke)");
  REQUIRE(index_buffer != nil);
  auto* indices = static_cast<uint16_t*>([index_buffer contents]);
  indices[0] = 0;
  indices[1] = 1;
  indices[2] = 2;

  // --- Encode: the exact group-4 bind sequence. ---
  id<MTLCommandQueue> queue = [device newCommandQueue];
  REQUIRE(queue != nil);
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  MTLRenderPassDescriptor* pass =
      [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = color_rt;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.depthAttachment.texture = depth_rt;
  pass.depthAttachment.loadAction = MTLLoadActionClear;
  pass.depthAttachment.storeAction = MTLStoreActionDontCare;
  pass.stencilAttachment.texture = depth_rt;
  pass.stencilAttachment.loadAction = MTLLoadActionClear;
  pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:pass];
  REQUIRE(encoder != nil);

  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      tess->pipeline];
  auto bind_all_stages = [&](id<MTLBuffer> buffer, NSUInteger offset,
                             NSUInteger index) {
    [encoder setObjectBuffer:buffer offset:offset atIndex:index];
    [encoder setMeshBuffer:buffer offset:offset atIndex:index];
    [encoder setFragmentBuffer:buffer offset:offset atIndex:index];
  };
  bind_all_stages(ab_buffer, 0, kIRArgumentBufferBindPoint);
  bind_all_stages(view_heap, 0, kIRDescriptorHeapBindPoint);
  bind_all_stages(sampler_heap, 0, kIRSamplerHeapBindPoint);
  // Hull/domain AB alias + tessellator tables on BOTH object and mesh.
  [encoder setObjectBuffer:ab_buffer
                    offset:0
                   atIndex:kIRArgumentBufferHullDomainBindPoint];
  [encoder setMeshBuffer:ab_buffer
                  offset:0
                 atIndex:kIRArgumentBufferHullDomainBindPoint];
  [encoder setObjectBuffer:tessellator_tables
                    offset:0
                   atIndex:kIRRuntimeTessellatorTablesBindPoint];
  [encoder setMeshBuffer:tessellator_tables
                  offset:0
                 atIndex:kIRRuntimeTessellatorTablesBindPoint];
  IRRuntimeVertexBuffers vertex_buffers = {};
  [encoder setObjectBytes:vertex_buffers
                   length:sizeof(vertex_buffers)
                  atIndex:kIRVertexBufferBindPoint];
  MTLRenderStages stages =
      MTLRenderStageObject | MTLRenderStageMesh | MTLRenderStageFragment;
  for (id<MTLBuffer> resource :
       {tessellator_tables, ab_buffer, view_heap, sampler_heap, system_views,
        system_buffer, cbv_scratch, null_buffer, guest_scratch,
        index_buffer}) {
    [encoder useResource:resource usage:MTLResourceUsageRead stages:stages];
  }
  [encoder useResource:null_texture usage:MTLResourceUsageRead stages:stages];

  // Config rebuilt VERBATIM from the pipeline mirror (the group-4 contract).
  IRRuntimeTessellationPipelineConfig config = {};
  config.outputPrimitiveType =
      IRRuntimeTessellatorOutputPrimitive(tess->output_primitive_type);
  config.vsOutputSizeInBytes = tess->vs_output_size_in_bytes;
  config.gsMaxInputPrimitivesPerMeshThreadgroup =
      tess->gs_max_input_primitives_per_mesh_threadgroup;
  config.hsMaxPatchesPerObjectThreadgroup =
      tess->hs_max_patches_per_object_threadgroup;
  config.hsInputControlPointCount = tess->hs_input_control_point_count;
  config.hsMaxObjectThreadsPerThreadgroup =
      tess->hs_max_object_threads_per_threadgroup;
  config.hsMaxTessellationFactor = tess->hs_max_tessellation_factor;
  config.gsInstanceCount = tess->gs_instance_count;
  REQUIRE(config.hsInputControlPointCount == 3);

  // Non-indexed patch draw (one 3-CP patch)...
  IRRuntimeDrawPatchesTessellationEmulation(
      encoder, IRRuntimePrimitiveType3ControlPointPatchlist, config,
      /*instanceCount=*/1, /*vertexCountPerInstance=*/3, /*baseInstance=*/0,
      /*baseVertex=*/0);
  // ...and the indexed entry point over the pre-swapped stream.
  IRRuntimeDrawIndexedPatchesTessellationEmulation(
      encoder, IRRuntimePrimitiveType3ControlPointPatchlist,
      MTLIndexTypeUInt16, index_buffer, config, /*instanceCount=*/1,
      /*indexCountPerInstance=*/3, /*baseInstance=*/0, /*baseVertex=*/0,
      /*startIndex=*/0);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  std::string command_buffer_error =
      command_buffer.error
          ? [[command_buffer.error localizedDescription] UTF8String]
          : "no error";
  INFO(command_buffer_error);
  CHECK(command_buffer.status == MTLCommandBufferStatusCompleted);
  CHECK(command_buffer.error == nil);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
