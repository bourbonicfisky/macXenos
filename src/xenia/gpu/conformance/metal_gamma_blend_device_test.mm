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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_edram_msl.h"
#include "xenia/gpu/xenos.h"

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

fragment float4 gamma_blend_integrated_fs(
    constant float4& source [[buffer(0)]]) {
  return source;
}
)";

constexpr std::array<uint16_t, 4> kDestination = {10000, 30000, 50000,
                                                   60000};
constexpr std::array<double, 4> kSource = {0.75, 0.25, 0.625, 0.5};

uint16_t QuantizeUnorm16(double value) {
  return uint16_t(std::lround(value * 65535.0));
}

uint8_t QuantizeUnorm8(float value) {
  return uint8_t(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

uint8_t ColorByte(uint32_t word, uint32_t channel) {
  return uint8_t(word >> (channel * 8u));
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

TEST_CASE(
    "production gamma materialize and pack bracket linear-domain blending",
    "[metal][gamma][blend][device][integration]") {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      WARN("No Metal device is available; skipping the integrated gamma test");
      return;
    }
    INFO("Metal device: " << [[device name] UTF8String]);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    REQUIRE(queue != nil);

    // Compile the exact production EDRAM conversion source, adding only the
    // constant-source draw used between its materialize and pack kernels.
    std::string combined_source(kEdramMsl);
    combined_source += kGammaBlendFeatureShader;
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString
                                 stringWithUTF8String:combined_source.c_str()]
                      options:nil
                        error:&error];
    INFO("Metal shader error: "
         << (error ? [[error localizedDescription] UTF8String] : "none"));
    REQUIRE(library != nil);

    id<MTLFunction> materialize_function =
        [library newFunctionWithName:@"materialize_color"];
    id<MTLFunction> pack_function =
        [library newFunctionWithName:@"pack_color"];
    REQUIRE(materialize_function != nil);
    REQUIRE(pack_function != nil);
    error = nil;
    id<MTLComputePipelineState> materialize_pipeline =
        [device newComputePipelineStateWithFunction:materialize_function
                                              error:&error];
    INFO("Metal materialize pipeline error: "
         << (error ? [[error localizedDescription] UTF8String] : "none"));
    REQUIRE(materialize_pipeline != nil);
    error = nil;
    id<MTLComputePipelineState> pack_pipeline =
        [device newComputePipelineStateWithFunction:pack_function
                                              error:&error];
    INFO("Metal pack pipeline error: "
         << (error ? [[error localizedDescription] UTF8String] : "none"));
    REQUIRE(pack_pipeline != nil);

    auto make_draw_pipeline = [&](bool blending_enabled) {
      MTLRenderPipelineDescriptor* descriptor =
          [[MTLRenderPipelineDescriptor alloc] init];
      descriptor.vertexFunction =
          [library newFunctionWithName:@"gamma_blend_feature_vs"];
      descriptor.fragmentFunction =
          [library newFunctionWithName:@"gamma_blend_integrated_fs"];
      MTLRenderPipelineColorAttachmentDescriptor* attachment =
          descriptor.colorAttachments[0];
      attachment.pixelFormat = MTLPixelFormatRGBA16Unorm;
      attachment.blendingEnabled = blending_enabled ? YES : NO;
      attachment.rgbBlendOperation = MTLBlendOperationAdd;
      attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      attachment.destinationRGBBlendFactor =
          MTLBlendFactorOneMinusSourceAlpha;
      attachment.alphaBlendOperation = MTLBlendOperationAdd;
      attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
      attachment.destinationAlphaBlendFactor = MTLBlendFactorZero;
      NSError* pipeline_error = nil;
      id<MTLRenderPipelineState> pipeline =
          [device newRenderPipelineStateWithDescriptor:descriptor
                                                 error:&pipeline_error];
      INFO("Metal integrated draw pipeline error: "
           << (pipeline_error
                   ? [[pipeline_error localizedDescription] UTF8String]
                   : "none"));
      REQUIRE(pipeline != nil);
      return pipeline;
    };
    id<MTLRenderPipelineState> blend_pipeline = make_draw_pipeline(true);
    id<MTLRenderPipelineState> replace_pipeline = make_draw_pipeline(false);

    MTLTextureDescriptor* texture_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Unorm
                                     width:1
                                    height:1
                                 mipmapped:NO];
    texture_descriptor.usage = MTLTextureUsageShaderRead |
                               MTLTextureUsageShaderWrite |
                               MTLTextureUsageRenderTarget;
    texture_descriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture =
        [device newTextureWithDescriptor:texture_descriptor];
    REQUIRE(texture != nil);

    id<MTLBuffer> canonical =
        [device newBufferWithLength:sizeof(uint32_t)
                           options:MTLResourceStorageModeShared];
    REQUIRE(canonical != nil);
    struct SurfaceParamsHost {
      uint32_t base, pitch, msaa, is_depth, fmt_log2, words;
    } surface = {0u, 1u, 0u, 0u, 0u, 1u};
    static_assert(sizeof(SurfaceParamsHost) == 6u * sizeof(uint32_t));
    const uint32_t gamma_format =
        uint32_t(xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA);
    const std::array<float, 4> source = {0.42f, 0.18f, 0.73f, 0.35f};

    auto run_case = [&](uint32_t destination,
                        id<MTLRenderPipelineState> draw_pipeline) {
      *static_cast<uint32_t*>(canonical.contents) = destination;
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      REQUIRE(command_buffer != nil);
      {
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        REQUIRE(encoder != nil);
        [encoder setComputePipelineState:materialize_pipeline];
        [encoder setBuffer:canonical offset:0 atIndex:0];
        [encoder setTexture:texture atIndex:0];
        [encoder setBytes:&surface length:sizeof(surface) atIndex:1];
        [encoder setBytes:&gamma_format
                   length:sizeof(gamma_format)
                  atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
      }
      {
        MTLRenderPassDescriptor* render_pass =
            [MTLRenderPassDescriptor renderPassDescriptor];
        render_pass.colorAttachments[0].texture = texture;
        render_pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        render_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> encoder =
            [command_buffer renderCommandEncoderWithDescriptor:render_pass];
        REQUIRE(encoder != nil);
        [encoder setRenderPipelineState:draw_pipeline];
        [encoder setFragmentBytes:source.data()
                           length:sizeof(source)
                          atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
        [encoder endEncoding];
      }
      {
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        REQUIRE(encoder != nil);
        [encoder setComputePipelineState:pack_pipeline];
        [encoder setTexture:texture atIndex:0];
        [encoder setBuffer:canonical offset:0 atIndex:0];
        [encoder setBytes:&surface length:sizeof(surface) atIndex:1];
        [encoder setBytes:&gamma_format
                   length:sizeof(gamma_format)
                  atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
      }
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      INFO("Metal command-buffer error: "
           << (command_buffer.error
                   ? [[command_buffer.error localizedDescription] UTF8String]
                   : "none"));
      REQUIRE(command_buffer.status == MTLCommandBufferStatusCompleted);
      return *static_cast<const uint32_t*>(canonical.contents);
    };

    // Guest bytes are R=120, G=80, B=200, A=160 in the canonical dword.
    constexpr uint32_t kDestinationWord = 0xA0C85078u;
    const uint32_t blended = run_case(kDestinationWord, blend_pipeline);
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const float destination_linear = xenos::PWLGammaToLinear(
          float(ColorByte(kDestinationWord, channel)) * (1.0f / 255.0f));
      const float blended_linear =
          source[channel] * source[3] + destination_linear * (1.0f - source[3]);
      const uint8_t expected =
          QuantizeUnorm8(xenos::LinearToPWLGamma(blended_linear));
      const uint8_t actual = ColorByte(blended, channel);
      CAPTURE(channel, actual, expected);
      CHECK(std::abs(int(actual) - int(expected)) <= 1);
    }
    const uint8_t expected_alpha = QuantizeUnorm8(source[3]);
    CHECK(int(ColorByte(blended, 3)) == int(expected_alpha));

    const uint32_t replaced = run_case(kDestinationWord, replace_pipeline);
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const uint8_t expected =
          QuantizeUnorm8(xenos::LinearToPWLGamma(source[channel]));
      const uint8_t actual = ColorByte(replaced, channel);
      CAPTURE(channel, actual, expected);
      CHECK(int(actual) == int(expected));
    }
    CHECK(int(ColorByte(replaced, 3)) == int(expected_alpha));
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
