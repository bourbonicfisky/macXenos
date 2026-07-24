/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// PP4 Level 1: prove the native Metal fixed-index primitive-restart contract
// used by the admitted 16-bit DMA strip route. This intentionally exercises a
// real indexed render pass. The two visible ranges prove that non-sentinel
// indices survive, while the empty center corridor proves 0xFFFF separates the
// ranges instead of producing a connecting primitive.

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace test {

namespace {

constexpr uint32_t kRestartWidth = 96;
constexpr uint32_t kRestartHeight = 32;

static const char* kPrimitiveRestartMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RestartVertexOut {
  float4 position [[position]];
};

vertex RestartVertexOut primitive_restart_vs(
    const device packed_float2* positions [[buffer(0)]],
    uint vertex_id [[vertex_id]]) {
  RestartVertexOut out;
  out.position = float4(float2(positions[vertex_id]), 0.0, 1.0);
  return out;
}

fragment half4 primitive_restart_fs() {
  return half4(1.0, 1.0, 1.0, 1.0);
}
)MSL";

struct Position {
  float x;
  float y;
};

struct RestartImage {
  std::vector<uint8_t> rgba;

  bool IsLit(uint32_t x, uint32_t y) const {
    return rgba[(size_t(y) * kRestartWidth + x) * 4] != 0;
  }

  uint32_t CountLit(uint32_t x_begin, uint32_t x_end, uint32_t y_begin, uint32_t y_end) const {
    uint32_t count = 0;
    for (uint32_t y = y_begin; y < y_end; ++y) {
      for (uint32_t x = x_begin; x < x_end; ++x) {
        count += IsLit(x, y) ? 1 : 0;
      }
    }
    return count;
  }
};

RestartImage RenderRestartStrip(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                id<MTLRenderPipelineState> pipeline,
                                MTLPrimitiveType primitive_type,
                                const std::vector<Position>& positions,
                                const std::vector<uint16_t>& indices) {
  id<MTLBuffer> vertex_buffer = [device newBufferWithBytes:positions.data()
                                                    length:positions.size() * sizeof(Position)
                                                   options:MTLResourceStorageModeShared];
  REQUIRE(vertex_buffer != nil);
  id<MTLBuffer> index_buffer = [device newBufferWithBytes:indices.data()
                                                   length:indices.size() * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
  REQUIRE(index_buffer != nil);

  MTLTextureDescriptor* texture_descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:kRestartWidth
                                                        height:kRestartHeight
                                                     mipmapped:NO];
  texture_descriptor.usage = MTLTextureUsageRenderTarget;
  texture_descriptor.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [device newTextureWithDescriptor:texture_descriptor];
  REQUIRE(texture != nil);

  MTLRenderPassDescriptor* render_pass = [MTLRenderPassDescriptor renderPassDescriptor];
  render_pass.colorAttachments[0].texture = texture;
  render_pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  render_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  render_pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);

  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  REQUIRE(command_buffer != nil);
  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:render_pass];
  REQUIRE(encoder != nil);
  [encoder setRenderPipelineState:pipeline];
  [encoder setVertexBuffer:vertex_buffer offset:0 atIndex:0];
  [encoder drawIndexedPrimitives:primitive_type
                      indexCount:indices.size()
                       indexType:MTLIndexTypeUInt16
                     indexBuffer:index_buffer
               indexBufferOffset:0];
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  REQUIRE(command_buffer.status == MTLCommandBufferStatusCompleted);
  REQUIRE(command_buffer.error == nil);

  RestartImage image;
  image.rgba.resize(size_t(kRestartWidth) * kRestartHeight * 4);
  [texture getBytes:image.rgba.data()
        bytesPerRow:kRestartWidth * 4
         fromRegion:MTLRegionMake2D(0, 0, kRestartWidth, kRestartHeight)
        mipmapLevel:0];
  return image;
}

}  // namespace

TEST_CASE("Metal fixed 16-bit restart separates native strip ranges",
          "[l1][metal][pp4][primitive-reset][fixed]") {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    REQUIRE(queue != nil);

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:kPrimitiveRestartMsl];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    const char* error_string = error ? error.localizedDescription.UTF8String : "";
    INFO(error_string);
    REQUIRE(library != nil);
    MTLRenderPipelineDescriptor* pipeline_descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipeline_descriptor.vertexFunction = [library newFunctionWithName:@"primitive_restart_vs"];
    pipeline_descriptor.fragmentFunction = [library newFunctionWithName:@"primitive_restart_fs"];
    pipeline_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:pipeline_descriptor error:&error];
    error_string = error ? error.localizedDescription.UTF8String : "";
    INFO(error_string);
    REQUIRE(pipeline != nil);

    SECTION("line strip") {
      const std::vector<Position> positions = {
          {-0.90f, 0.0f}, {-0.30f, 0.0f}, {0.30f, 0.0f}, {0.90f, 0.0f}};
      const std::vector<uint16_t> indices = {0, 1, UINT16_MAX, 2, 3};
      RestartImage image = RenderRestartStrip(device, queue, pipeline, MTLPrimitiveTypeLineStrip,
                                              positions, indices);

      CHECK(image.CountLit(4, 35, 0, kRestartHeight) >= 20);
      CHECK(image.CountLit(61, 92, 0, kRestartHeight) >= 20);
      CHECK(image.CountLit(42, 54, 0, kRestartHeight) == 0);
    }

    SECTION("triangle strip") {
      const std::vector<Position> positions = {{-0.90f, -0.60f}, {-0.90f, 0.60f}, {-0.30f, -0.60f},
                                               {-0.30f, 0.60f},  {0.30f, -0.60f}, {0.30f, 0.60f},
                                               {0.90f, -0.60f},  {0.90f, 0.60f}};
      const std::vector<uint16_t> indices = {0, 1, 2, 3, UINT16_MAX, 4, 5, 6, 7};
      RestartImage image = RenderRestartStrip(device, queue, pipeline,
                                              MTLPrimitiveTypeTriangleStrip, positions, indices);

      CHECK(image.CountLit(4, 35, 5, 27) >= 400);
      CHECK(image.CountLit(61, 92, 5, 27) >= 400);
      CHECK(image.CountLit(42, 54, 0, kRestartHeight) == 0);
    }
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
