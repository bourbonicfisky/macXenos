/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Target-device feature gate for the candidate linear gamma render-target
// storage. This deliberately tests Metal itself rather than emulator behavior:
// RGBA16Unorm must be renderable, loadable, storeable, and fixed-function
// blendable before it may be selected as a host render-target format.

#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr char kGammaBlendFeatureShader[] = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOutput {
  float4 position [[position]];
};

vertex VertexOutput gamma_blend_feature_vs(uint vertex_id [[vertex_id]]) {
  const float2 positions[] = {
      float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
  VertexOutput output;
  output.position = float4(positions[vertex_id], 0.0, 1.0);
  return output;
}

fragment float4 gamma_blend_feature_fs() {
  return float4(0.75, 0.25, 0.625, 0.5);
}
)";

constexpr std::array<uint16_t, 4> kDestination = {10000, 30000, 50000,
                                                   60000};
constexpr std::array<double, 4> kSource = {0.75, 0.25, 0.625, 0.5};

uint16_t QuantizeUnorm16(double value) {
  return uint16_t(std::lround(value * 65535.0));
}

}  // namespace

TEST_CASE("RGBA16Unorm render targets support fixed-function blending",
          "[metal][gamma][blend][device]") {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      WARN("No Metal device is available; skipping the target-device gate");
      return;
    }
    INFO("Metal device: " << [[device name] UTF8String]);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    REQUIRE(queue != nil);

    NSError* error = nil;
    NSString* shader_source =
        [NSString stringWithUTF8String:kGammaBlendFeatureShader];
    id<MTLLibrary> library = [device newLibraryWithSource:shader_source
                                                  options:nil
                                                    error:&error];
    INFO("Metal shader error: "
         << (error ? [[error localizedDescription] UTF8String] : "none"));
    REQUIRE(library != nil);

    MTLRenderPipelineDescriptor* pipeline_descriptor =
        [[MTLRenderPipelineDescriptor alloc] init];
    pipeline_descriptor.vertexFunction =
        [library newFunctionWithName:@"gamma_blend_feature_vs"];
    pipeline_descriptor.fragmentFunction =
        [library newFunctionWithName:@"gamma_blend_feature_fs"];

    MTLRenderPipelineColorAttachmentDescriptor* color_attachment =
        pipeline_descriptor.colorAttachments[0];
    color_attachment.pixelFormat = MTLPixelFormatRGBA16Unorm;
    color_attachment.blendingEnabled = YES;
    color_attachment.rgbBlendOperation = MTLBlendOperationAdd;
    color_attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    color_attachment.destinationRGBBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    // Pin alpha independently so the check proves the configured alpha path,
    // not an incidental consequence of the RGB equation.
    color_attachment.alphaBlendOperation = MTLBlendOperationAdd;
    color_attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    color_attachment.destinationAlphaBlendFactor = MTLBlendFactorZero;

    error = nil;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:pipeline_descriptor
                                               error:&error];
    INFO("Metal pipeline error: "
         << (error ? [[error localizedDescription] UTF8String] : "none"));
    REQUIRE(pipeline != nil);

    MTLTextureDescriptor* texture_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Unorm
                                     width:1
                                    height:1
                                 mipmapped:NO];
    texture_descriptor.usage = MTLTextureUsageRenderTarget;
    texture_descriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture =
        [device newTextureWithDescriptor:texture_descriptor];
    REQUIRE(texture != nil);
    [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
               mipmapLevel:0
                 withBytes:kDestination.data()
               bytesPerRow:sizeof(kDestination)];

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    REQUIRE(command_buffer != nil);
    MTLRenderPassDescriptor* render_pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    render_pass.colorAttachments[0].texture = texture;
    render_pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    render_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:render_pass];
    REQUIRE(encoder != nil);
    [encoder setRenderPipelineState:pipeline];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:3];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    INFO("Metal command-buffer error: "
         << (command_buffer.error
                 ? [[command_buffer.error localizedDescription] UTF8String]
                 : "none"));
    REQUIRE(command_buffer.status == MTLCommandBufferStatusCompleted);

    std::array<uint16_t, 4> actual = {};
    [texture getBytes:actual.data()
          bytesPerRow:sizeof(actual)
           fromRegion:MTLRegionMake2D(0, 0, 1, 1)
          mipmapLevel:0];

    // Metal may round at the fragment-to-blend and blend-to-UNORM boundaries.
    // Two UNORM16 LSBs is tight enough to distinguish the requested equation
    // while allowing either valid rounding order.
    constexpr int32_t kRoundingTolerance = 2;
    for (size_t channel = 0; channel < 3; ++channel) {
      const double destination = double(kDestination[channel]) / 65535.0;
      const uint16_t expected =
          QuantizeUnorm16(kSource[channel] * kSource[3] +
                          destination * (1.0 - kSource[3]));
      CAPTURE(channel, actual[channel], expected);
      CHECK(std::abs(int32_t(actual[channel]) - int32_t(expected)) <=
            kRoundingTolerance);
    }
    const uint16_t expected_alpha = QuantizeUnorm16(kSource[3]);
    CAPTURE(actual[3], expected_alpha);
    CHECK(std::abs(int32_t(actual[3]) - int32_t(expected_alpha)) <=
          kRoundingTolerance);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
