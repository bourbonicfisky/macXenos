/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// PP3 L1 conformance: execute the shared closing-index vertex remap after the
// complete Xenia microcode -> DXBC -> DXIL -> Metal Shader Converter chain.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

#include "metal_irconverter_runtime.h"
#include "third_party/catch/include/catch.hpp"
#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 8;
constexpr uint32_t kTopLevelArgumentCount = 32;
constexpr uint32_t kSystemConstantsArgument = 10;

// Minimal fetch-free Xbox 360 vertex program. It receives the translated guest
// vertex index in r0.x and exports (r0.x, 0, 0, 1) as position. The instruction
// shape is from an authoritative trace shader, with only the position sources
// reduced from the already-zero r1 to r0. Keeping real microcode here exercises
// DxbcShaderTranslator rather than pinning a generated DXBC artifact.
alignas(uint32_t) constexpr uint8_t kVertexUcode[] = {
    0x03, 0x10, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC4, 0x04, 0x10, 0x00, 0x00,
    0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC2, 0x05, 0x10, 0x00, 0x00, 0x00, 0x12, 0x06, 0x10,
    0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2,
    0x00, 0xC0, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2, 0x3E, 0xC0, 0x89, 0xC8,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xE2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};
static_assert(sizeof(kVertexUcode) == 96);

struct RenderResult {
  bool ok = false;
  std::vector<uint8_t> rgba;
  std::string error;
};

id<MTLFunction> BuildWhiteFragment(id<MTLDevice> device, id<MTLLibrary>* library_out,
                                   std::string& error_out) {
  static NSString* source = @"#include <metal_stdlib>\n"
                             "using namespace metal;\n"
                             "fragment float4 pp3_white() { return float4(1.0); }\n";
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
  if (!library) {
    error_out = error ? [[error localizedDescription] UTF8String]
                      : "native fragment library creation failed";
    return nil;
  }
  id<MTLFunction> function = [library newFunctionWithName:@"pp3_white"];
  if (!function) {
    error_out = "native fragment function creation failed";
    return nil;
  }
  *library_out = library;
  return function;
}

RenderResult RenderLineStrip(id<MTLDevice> device, id<MTLRenderPipelineState> pipeline,
                             uint32_t closing_index) {
  RenderResult result;

  MTLTextureDescriptor* texture_desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:kWidth
                                                        height:kHeight
                                                     mipmapped:NO];
  texture_desc.storageMode = MTLStorageModeShared;
  texture_desc.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> texture = [device newTextureWithDescriptor:texture_desc];
  id<MTLBuffer> system_buffer =
      [device newBufferWithLength:sizeof(DxbcShaderTranslator::SystemConstants)
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> null_buffer = [device newBufferWithLength:4096
                                                  options:MTLResourceStorageModeShared];
  id<MTLBuffer> argument_buffer =
      [device newBufferWithLength:kTopLevelArgumentCount * sizeof(uint64_t)
                          options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!texture || !system_buffer || !null_buffer || !argument_buffer || !queue) {
    result.error = "Metal test resource allocation failed";
    return result;
  }

  auto* constants = static_cast<DxbcShaderTranslator::SystemConstants*>([system_buffer contents]);
  std::memset(constants, 0, sizeof(*constants));
  constants->flags = DxbcShaderTranslator::kSysFlag_WNotReciprocal;
  constants->line_loop_closing_index = closing_index;
  constants->vertex_index_endian = xenos::Endian::kNone;
  constants->vertex_index_min = 0;
  constants->vertex_index_max = xenos::kVertexIndexMask;
  constants->ndc_scale[0] = 0.5f;
  constants->ndc_scale[1] = 1.0f;
  constants->ndc_scale[2] = 1.0f;
  constants->ndc_offset[0] = -0.5f;

  std::memset([null_buffer contents], 0, [null_buffer length]);
  auto* arguments = static_cast<uint64_t*>([argument_buffer contents]);
  for (uint32_t i = 0; i < kTopLevelArgumentCount; ++i) {
    arguments[i] = [null_buffer gpuAddress];
  }
  arguments[kSystemConstantsArgument] = [system_buffer gpuAddress];

  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = texture;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLRenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
  [encoder setRenderPipelineState:pipeline];
  [encoder setViewport:MTLViewport{0.0, 0.0, double(kWidth), double(kHeight), 0.0, 1.0}];
  [encoder setVertexBuffer:argument_buffer offset:0 atIndex:kIRArgumentBufferBindPoint];
  [encoder setVertexBuffer:null_buffer offset:0 atIndex:kIRDescriptorHeapBindPoint];
  [encoder setVertexBuffer:null_buffer offset:0 atIndex:kIRSamplerHeapBindPoint];
  [encoder useResource:argument_buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
  [encoder useResource:system_buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
  [encoder useResource:null_buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
  IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeLineStrip, uint64_t(0), uint64_t(3));
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if ([command_buffer status] == MTLCommandBufferStatusError) {
    NSError* error = [command_buffer error];
    result.error =
        error ? [[error localizedDescription] UTF8String] : "line-strip command buffer failed";
    return result;
  }

  result.rgba.resize(kWidth * kHeight * 4);
  [texture getBytes:result.rgba.data()
        bytesPerRow:kWidth * 4
         fromRegion:MTLRegionMake2D(0, 0, kWidth, kHeight)
        mipmapLevel:0];
  result.ok = true;
  return result;
}

uint32_t CountLitRightPixels(const std::vector<uint8_t>& rgba) {
  uint32_t count = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 37; x < kWidth; ++x) {
      if (rgba[(y * kWidth + x) * 4] != 0) {
        ++count;
      }
    }
  }
  return count;
}

std::string DescribeLitPixels(const std::vector<uint8_t>& rgba) {
  uint32_t count = 0;
  uint32_t min_x = kWidth;
  uint32_t max_x = 0;
  uint32_t min_y = kHeight;
  uint32_t max_y = 0;
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      if (rgba[(y * kWidth + x) * 4] == 0) {
        continue;
      }
      ++count;
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
    }
  }
  if (!count) {
    return "no lit pixels";
  }
  return fmt::format("{} lit pixels, bounds x={}..{} y={}..{}", count, min_x, max_x, min_y, max_y);
}

TEST_CASE("PP3 closing index survives DXBC to DXIL to MSC and executes",
          "[l1][metal][pp3auto][shader]") {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    MetalPipelineCache cache((__bridge void*)device);
    REQUIRE(cache.Initialize());
    // Shader::DumpUcode writes the already-normalized host-native dwords. State
    // that provenance explicitly so this fixture is not guest-byte-swapped a
    // second time by MetalShader's default big-endian input contract.
    MetalShader guest_shader(xenos::ShaderType::kVertex, 0,
                             reinterpret_cast<const uint32_t*>(kVertexUcode),
                             sizeof(kVertexUcode) / sizeof(uint32_t), std::endian::little);
    MetalShader* shader = &guest_shader;
    StringBuffer disassembly;
    shader->AnalyzeUcode(disassembly);
    INFO(disassembly.to_string());

    DxbcShaderTranslator::Modification modification(
        cache.shader_translator()->GetDefaultVertexShaderModification(
            shader->GetDynamicAddressableRegisterCount(xenos::kMaxShaderTempRegisters),
            Shader::HostVertexShaderType::kVertex));
    modification.vertex.interpolator_mask = 0;
    auto* translation = static_cast<MetalShader::MetalTranslation*>(
        shader->GetOrCreateTranslation(modification.value));
    REQUIRE(cache.TranslateShader(*translation));
    REQUIRE(translation->has_metal_function());

    std::string error;
    id<MTLLibrary> fragment_library = nil;
    id<MTLFunction> fragment = BuildWhiteFragment(device, &fragment_library, error);
    INFO(error);
    REQUIRE(fragment != nil);

    MTLRenderPipelineDescriptor* pipeline_desc = [[MTLRenderPipelineDescriptor alloc] init];
    pipeline_desc.vertexFunction = (__bridge id<MTLFunction>)translation->metal_function();
    pipeline_desc.fragmentFunction = fragment;
    pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    NSError* pipeline_error = nil;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:pipeline_desc error:&pipeline_error];
    std::string pipeline_message =
        pipeline_error ? [[pipeline_error localizedDescription] UTF8String] : "";
    INFO(pipeline_message);
    REQUIRE(pipeline != nil);

    RenderResult control = RenderLineStrip(device, pipeline, 0);
    INFO(control.error);
    REQUIRE(control.ok);
    RenderResult closed = RenderLineStrip(device, pipeline, 2);
    INFO(closed.error);
    REQUIRE(closed.ok);

    uint32_t control_right = CountLitRightPixels(control.rgba);
    uint32_t closed_right = CountLitRightPixels(closed.rgba);
    INFO("control " << DescribeLitPixels(control.rgba));
    INFO("closed " << DescribeLitPixels(closed.rgba));
    INFO("control right-side pixels " << control_right);
    INFO("closed right-side pixels " << closed_right);
    CHECK(control_right > 0);
    CHECK(closed_right == 0);
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
