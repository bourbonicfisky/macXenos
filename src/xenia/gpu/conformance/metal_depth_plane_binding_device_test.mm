/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Target-device probe for reduce-metal-draw-pass-traffic D2: a draw pass that
// binds only the depth plane of a Depth32Float_Stencil8 texture (stencil
// attachment nil, pipeline stencilAttachmentPixelFormat Invalid). Two
// questions about Metal itself, not about emulator behaviour:
//  1. Does the unbound stencil plane keep its contents (the content
//     guarantee the lever rests on)? Asserted.
//  2. Are the depth values a depth-writing draw produces bit-identical to
//     the same draw in a pass that binds both planes? Reported (WARN) and
//     asserted, because the corpus showed RDR 8551 depth words moving when
//     stencil was left unbound.

#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kSize = 256;

constexpr char kShader[] = R"(
#include <metal_stdlib>
using namespace metal;
struct VertexOutput { float4 position [[position]]; };
// One perspective triangle (w varies per vertex so depth interpolation is
// perspective-correct and exercises the rasterizer's depth math).
vertex VertexOutput depth_plane_vs(uint vid [[vertex_id]]) {
  const float3 ndc[3] = {float3(-1.0, -1.0, 0.1), float3(1.0, -1.0, 0.7),
                         float3(0.0, 1.0, 0.4)};
  const float w[3] = {1.0, 2.5, 1.7};
  VertexOutput o;
  o.position = float4(ndc[vid].xy * w[vid], ndc[vid].z * w[vid], w[vid]);
  return o;
}
fragment float4 depth_plane_fs() { return float4(0.5, 0.25, 0.75, 1.0); }
)";

struct Arm {
  bool bind_stencil;
  float bias_constant;
  float bias_slope;
  std::vector<float> depth;
  std::vector<uint8_t> stencil;
};

bool RunArm(id<MTLDevice> device, id<MTLCommandQueue> queue,
            id<MTLLibrary> library, Arm& arm) {
  MTLTextureDescriptor* dd = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                   width:kSize
                                  height:kSize
                               mipmapped:NO];
  dd.usage = MTLTextureUsageRenderTarget;
  dd.storageMode = MTLStorageModePrivate;
  id<MTLTexture> ds = [device newTextureWithDescriptor:dd];
  MTLTextureDescriptor* cd = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:kSize
                                  height:kSize
                               mipmapped:NO];
  cd.usage = MTLTextureUsageRenderTarget;
  cd.storageMode = MTLStorageModePrivate;
  id<MTLTexture> color = [device newTextureWithDescriptor:cd];
  if (!ds || !color) {
    return false;
  }

  MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
  pd.vertexFunction = [library newFunctionWithName:@"depth_plane_vs"];
  pd.fragmentFunction = [library newFunctionWithName:@"depth_plane_fs"];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  pd.stencilAttachmentPixelFormat = arm.bind_stencil
                                        ? MTLPixelFormatDepth32Float_Stencil8
                                        : MTLPixelFormatInvalid;
  NSError* error = nil;
  id<MTLRenderPipelineState> pipeline =
      [device newRenderPipelineStateWithDescriptor:pd error:&error];
  INFO("pipeline error: "
       << (error ? [[error localizedDescription] UTF8String] : "none"));
  if (!pipeline) {
    return false;
  }
  MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
  dsd.depthCompareFunction = MTLCompareFunctionLess;
  dsd.depthWriteEnabled = YES;
  id<MTLDepthStencilState> ds_state =
      [device newDepthStencilStateWithDescriptor:dsd];

  id<MTLCommandBuffer> cb = [queue commandBuffer];
  // Pass 1: clear both planes and the colour (binds everything).
  {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = color;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.depthAttachment.texture = ds;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    rp.stencilAttachment.texture = ds;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.clearStencil = 0x5A;
    rp.stencilAttachment.storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    [e endEncoding];
  }
  // Pass 2: the depth-writing draw, with or without the stencil plane bound.
  {
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = color;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.depthAttachment.texture = ds;
    rp.depthAttachment.loadAction = MTLLoadActionLoad;
    rp.depthAttachment.storeAction = MTLStoreActionStore;
    if (arm.bind_stencil) {
      rp.stencilAttachment.texture = ds;
      rp.stencilAttachment.loadAction = MTLLoadActionLoad;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
    [e setRenderPipelineState:pipeline];
    [e setDepthStencilState:ds_state];
    [e setDepthBias:arm.bias_constant slopeScale:arm.bias_slope clamp:0.0f];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
  }
  // Read both planes back.
  id<MTLBuffer> depth_buf =
      [device newBufferWithLength:kSize * kSize * 4
                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> stencil_buf =
      [device newBufferWithLength:kSize * kSize
                          options:MTLResourceStorageModeShared];
  {
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:ds
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(kSize, kSize, 1)
                    toBuffer:depth_buf
           destinationOffset:0
      destinationBytesPerRow:kSize * 4
    destinationBytesPerImage:kSize * kSize * 4
                     options:MTLBlitOptionDepthFromDepthStencil];
    [b copyFromTexture:ds
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(kSize, kSize, 1)
                    toBuffer:stencil_buf
           destinationOffset:0
      destinationBytesPerRow:kSize
    destinationBytesPerImage:kSize * kSize
                     options:MTLBlitOptionStencilFromDepthStencil];
    [b endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];
  INFO("command buffer error: "
       << (cb.error ? [[cb.error localizedDescription] UTF8String] : "none"));
  if (cb.status != MTLCommandBufferStatusCompleted) {
    return false;
  }
  arm.depth.assign(static_cast<const float*>(depth_buf.contents),
                   static_cast<const float*>(depth_buf.contents) +
                       kSize * kSize);
  arm.stencil.assign(static_cast<const uint8_t*>(stencil_buf.contents),
                     static_cast<const uint8_t*>(stencil_buf.contents) +
                         kSize * kSize);
  return true;
}

}  // namespace

TEST_CASE("Depth-plane-only pass: stencil preserved, depth bit-identical",
          "[metal][pass-traffic][depth-plane][device]") {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      WARN("No Metal device is available; skipping the depth-plane probe");
      return;
    }
    INFO("Metal device: " << [[device name] UTF8String]);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    REQUIRE(queue != nil);
    NSError* error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kShader]
                             options:nil
                               error:&error];
    INFO("shader error: "
         << (error ? [[error localizedDescription] UTF8String] : "none"));
    REQUIRE(library != nil);

    const float biases[][2] = {{0.0f, 0.0f}, {4.0f, 1.5f}, {-16.0f, 0.0f}};
    for (const auto& bias : biases) {
      Arm both{true, bias[0], bias[1]};
      Arm depth_only{false, bias[0], bias[1]};
      REQUIRE(RunArm(device, queue, library, both));
      REQUIRE(RunArm(device, queue, library, depth_only));

      // 1. The unbound stencil plane keeps the cleared value everywhere.
      uint32_t stencil_changed = 0;
      for (uint8_t s : depth_only.stencil) {
        stencil_changed += s != 0x5A;
      }
      CAPTURE(bias[0], bias[1], stencil_changed);
      CHECK(stencil_changed == 0);

      // 2. Depth values: covered samples, differing samples, largest delta.
      uint32_t covered = 0, differing = 0;
      float max_delta = 0.0f;
      uint32_t first_diff = UINT32_MAX;
      for (uint32_t i = 0; i < kSize * kSize; ++i) {
        if (both.depth[i] != 1.0f || depth_only.depth[i] != 1.0f) {
          ++covered;
        }
        if (both.depth[i] != depth_only.depth[i]) {
          ++differing;
          max_delta = std::max(max_delta,
                               std::fabs(both.depth[i] - depth_only.depth[i]));
          first_diff = std::min(first_diff, i);
        }
      }
      WARN("depth-plane probe bias=(" << bias[0] << "," << bias[1]
           << "): covered=" << covered << " differing=" << differing
           << " max_delta=" << max_delta << " first=" << first_diff
           << (first_diff != UINT32_MAX
                   ? " both=" + std::to_string(both.depth[first_diff]) +
                         " depth_only=" +
                         std::to_string(depth_only.depth[first_diff])
                   : std::string()));
      CAPTURE(covered, differing, max_delta);
      CHECK(covered > 1000);
      CHECK(differing == 0);
    }
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
