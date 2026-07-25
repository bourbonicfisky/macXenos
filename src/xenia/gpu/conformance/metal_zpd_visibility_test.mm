/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 ******************************************************************************
 */

// Level 1 (Metal device): the visibility-counting mechanism that backs guest
// ZPD reports, exercised through the PRODUCTION MetalZPDVisibilityPool rather
// than a test-local replica.
//
// The guest contract is "z pass done": a report counts samples that survive the
// depth test, not samples the rasterizer merely covered. These cases pin that
// meaning on the device, plus the two Metal rules the backend design depends
// on -- the visibility buffer must be attached before the encoder exists, and
// each segment must own a distinct buffer offset within a pass.
//
// Geometry is chosen so the expected counts are EXACT, not approximate: the
// render target is 64x64 with 1x sampling and quads land on pixel boundaries,
// so a full-viewport quad covers exactly 4096 samples and a half-viewport quad
// exactly 2048. An approximate count would not distinguish "counting works"
// from "counting is off by a fill-rule edge".

#import <Metal/Metal.h>

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_zpd_visibility_pool.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint32_t kTargetSize = 64;
constexpr uint64_t kFullCoverage = uint64_t(kTargetSize) * kTargetSize;
constexpr uint64_t kHalfCoverage = kFullCoverage / 2;
constexpr uint32_t kTestCapacity = 8;

static const char* kVisibilityMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VisibilityVertexOut {
  float4 position [[position]];
};

vertex VisibilityVertexOut visibility_vs(
    const device packed_float3* positions [[buffer(0)]],
    uint vertex_id [[vertex_id]]) {
  VisibilityVertexOut out;
  float3 p = float3(positions[vertex_id]);
  out.position = float4(p.xy, p.z, 1.0);
  return out;
}

fragment half4 visibility_fs() {
  return half4(1.0, 1.0, 1.0, 1.0);
}
)MSL";

struct Position {
  float x;
  float y;
  float z;
};

// A quad as two triangles. `x_end` of 0.0 covers exactly the left half.
std::vector<Position> Quad(float x_begin, float x_end, float z) {
  return {{x_begin, -1.0f, z}, {x_end, -1.0f, z}, {x_begin, 1.0f, z},
          {x_end, -1.0f, z},   {x_end, 1.0f, z},  {x_begin, 1.0f, z}};
}

id<MTLBuffer> MakeVertexBuffer(id<MTLDevice> device,
                               const std::vector<Position>& positions) {
  return [device newBufferWithBytes:positions.data()
                             length:positions.size() * sizeof(Position)
                            options:MTLResourceStorageModeShared];
}

struct Harness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLRenderPipelineState> pipeline = nil;
  id<MTLDepthStencilState> depth_write_less = nil;
  id<MTLDepthStencilState> depth_test_less = nil;
  id<MTLTexture> color = nil;
  id<MTLTexture> depth = nil;
};

Harness MakeHarness() {
  Harness h;
  h.device = MTLCreateSystemDefaultDevice();
  REQUIRE(h.device != nil);
  h.queue = [h.device newCommandQueue];
  REQUIRE(h.queue != nil);

  NSError* error = nil;
  id<MTLLibrary> library =
      [h.device newLibraryWithSource:[NSString stringWithUTF8String:kVisibilityMsl]
                             options:nil
                               error:&error];
  const char* error_string = error ? error.localizedDescription.UTF8String : "";
  INFO(error_string);
  REQUIRE(library != nil);

  MTLRenderPipelineDescriptor* pipeline_descriptor =
      [[MTLRenderPipelineDescriptor alloc] init];
  pipeline_descriptor.vertexFunction =
      [library newFunctionWithName:@"visibility_vs"];
  pipeline_descriptor.fragmentFunction =
      [library newFunctionWithName:@"visibility_fs"];
  pipeline_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  pipeline_descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  h.pipeline = [h.device newRenderPipelineStateWithDescriptor:pipeline_descriptor
                                                        error:&error];
  error_string = error ? error.localizedDescription.UTF8String : "";
  INFO(error_string);
  REQUIRE(h.pipeline != nil);

  MTLDepthStencilDescriptor* write_descriptor =
      [[MTLDepthStencilDescriptor alloc] init];
  write_descriptor.depthCompareFunction = MTLCompareFunctionLess;
  write_descriptor.depthWriteEnabled = YES;
  h.depth_write_less =
      [h.device newDepthStencilStateWithDescriptor:write_descriptor];
  REQUIRE(h.depth_write_less != nil);

  MTLDepthStencilDescriptor* test_descriptor =
      [[MTLDepthStencilDescriptor alloc] init];
  test_descriptor.depthCompareFunction = MTLCompareFunctionLess;
  test_descriptor.depthWriteEnabled = NO;
  h.depth_test_less =
      [h.device newDepthStencilStateWithDescriptor:test_descriptor];
  REQUIRE(h.depth_test_less != nil);

  MTLTextureDescriptor* color_descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:kTargetSize
                                  height:kTargetSize
                               mipmapped:NO];
  color_descriptor.usage = MTLTextureUsageRenderTarget;
  color_descriptor.storageMode = MTLStorageModePrivate;
  h.color = [h.device newTextureWithDescriptor:color_descriptor];
  REQUIRE(h.color != nil);

  MTLTextureDescriptor* depth_descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                   width:kTargetSize
                                  height:kTargetSize
                               mipmapped:NO];
  depth_descriptor.usage = MTLTextureUsageRenderTarget;
  depth_descriptor.storageMode = MTLStorageModePrivate;
  h.depth = [h.device newTextureWithDescriptor:depth_descriptor];
  REQUIRE(h.depth != nil);
  return h;
}

// Builds the pass the backend builds: the visibility buffer is attached to the
// DESCRIPTOR, before any encoder exists.
MTLRenderPassDescriptor* MakePass(const Harness& h, id<MTLBuffer> visibility) {
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = h.color;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  pass.depthAttachment.texture = h.depth;
  pass.depthAttachment.loadAction = MTLLoadActionClear;
  pass.depthAttachment.storeAction = MTLStoreActionDontCare;
  pass.depthAttachment.clearDepth = 1.0;
  pass.visibilityResultBuffer = visibility;
  return pass;
}

void DrawQuad(id<MTLRenderCommandEncoder> encoder, id<MTLBuffer> vertices) {
  [encoder setVertexBuffer:vertices offset:0 atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

void CommitAndWait(id<MTLCommandBuffer> command_buffer) {
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  const char* error_string =
      command_buffer.error ? command_buffer.error.localizedDescription.UTF8String
                           : "";
  INFO(error_string);
  REQUIRE(command_buffer.status == MTLCommandBufferStatusCompleted);
}

}  // namespace

TEST_CASE("Metal ZPD pool allocates a visibility buffer on the device",
          "[l1][metal][zpd]") {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    MetalZPDVisibilityPool pool;
    REQUIRE(pool.EnsureInitialized((__bridge void*)device, kTestCapacity));
    REQUIRE(pool.is_initialized());
    REQUIRE(pool.capacity() == kTestCapacity);
    REQUIRE(pool.free_count() == kTestCapacity);

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)pool.visibility_buffer();
    REQUIRE(buffer != nil);
    REQUIRE(buffer.length ==
            MetalZPDSlotPool::BufferSizeForCapacity(kTestCapacity));

    // Idempotent for an unchanged capacity: the same buffer is kept, so an
    // encoder already built against it stays valid.
    REQUIRE(pool.EnsureInitialized((__bridge void*)device, kTestCapacity));
    REQUIRE((__bridge id<MTLBuffer>)pool.visibility_buffer() == buffer);

    pool.Shutdown();
    REQUIRE_FALSE(pool.is_initialized());
    REQUIRE(pool.visibility_buffer() == nullptr);
  }
}

TEST_CASE("Metal ZPD counting reports exact covered sample counts",
          "[l1][metal][zpd]") {
  @autoreleasepool {
    Harness h = MakeHarness();
    MetalZPDVisibilityPool pool;
    REQUIRE(pool.EnsureInitialized((__bridge void*)h.device, kTestCapacity));
    id<MTLBuffer> visibility =
        (__bridge id<MTLBuffer>)pool.visibility_buffer();

    uint32_t index = 0;
    uint32_t generation = 0;
    size_t offset = 0;
    REQUIRE(pool.Acquire(index, generation, offset));

    id<MTLBuffer> full = MakeVertexBuffer(h.device, Quad(-1.0f, 1.0f, 0.5f));
    id<MTLCommandBuffer> command_buffer = [h.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:MakePass(h, visibility)];
    [encoder setRenderPipelineState:h.pipeline];
    [encoder setDepthStencilState:h.depth_write_less];
    [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                              offset:offset];
    DrawQuad(encoder, full);
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    [encoder endEncoding];
    CommitAndWait(command_buffer);

    CHECK(pool.Read(index) == kFullCoverage);
  }
}

TEST_CASE("Metal ZPD counting excludes depth-failing samples",
          "[l1][metal][zpd]") {
  @autoreleasepool {
    Harness h = MakeHarness();
    MetalZPDVisibilityPool pool;
    REQUIRE(pool.EnsureInitialized((__bridge void*)h.device, kTestCapacity));
    id<MTLBuffer> visibility =
        (__bridge id<MTLBuffer>)pool.visibility_buffer();

    uint32_t index = 0;
    uint32_t generation = 0;
    size_t offset = 0;
    REQUIRE(pool.Acquire(index, generation, offset));

    // A near occluder over the left half, then a farther full-viewport quad
    // under the counted segment. Only the unoccluded right half passes depth,
    // which is exactly what a guest ZPD report means.
    id<MTLBuffer> occluder = MakeVertexBuffer(h.device, Quad(-1.0f, 0.0f, 0.25f));
    id<MTLBuffer> tested = MakeVertexBuffer(h.device, Quad(-1.0f, 1.0f, 0.75f));

    id<MTLCommandBuffer> command_buffer = [h.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:MakePass(h, visibility)];
    [encoder setRenderPipelineState:h.pipeline];
    [encoder setDepthStencilState:h.depth_write_less];
    DrawQuad(encoder, occluder);

    [encoder setDepthStencilState:h.depth_test_less];
    [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                              offset:offset];
    DrawQuad(encoder, tested);
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    [encoder endEncoding];
    CommitAndWait(command_buffer);

    CHECK(pool.Read(index) == kHalfCoverage);
  }
}

TEST_CASE("Metal ZPD segments at distinct offsets do not cross-contaminate",
          "[l1][metal][zpd]") {
  @autoreleasepool {
    Harness h = MakeHarness();
    MetalZPDVisibilityPool pool;
    REQUIRE(pool.EnsureInitialized((__bridge void*)h.device, kTestCapacity));
    id<MTLBuffer> visibility =
        (__bridge id<MTLBuffer>)pool.visibility_buffer();

    uint32_t first_index = 0, first_generation = 0;
    uint32_t second_index = 0, second_generation = 0;
    size_t first_offset = 0, second_offset = 0;
    REQUIRE(pool.Acquire(first_index, first_generation, first_offset));
    REQUIRE(pool.Acquire(second_index, second_generation, second_offset));
    // Metal permits a given offset only once per pass, so two live segments in
    // one pass must hold different slots.
    REQUIRE(first_offset != second_offset);

    id<MTLBuffer> full = MakeVertexBuffer(h.device, Quad(-1.0f, 1.0f, 0.5f));
    id<MTLBuffer> half = MakeVertexBuffer(h.device, Quad(-1.0f, 0.0f, 0.5f));

    id<MTLCommandBuffer> command_buffer = [h.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:MakePass(h, visibility)];
    [encoder setRenderPipelineState:h.pipeline];
    [encoder setDepthStencilState:h.depth_test_less];

    [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                              offset:first_offset];
    DrawQuad(encoder, full);
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];

    // A draw between the segments must land in neither.
    DrawQuad(encoder, full);

    [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                              offset:second_offset];
    DrawQuad(encoder, half);
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    [encoder endEncoding];
    CommitAndWait(command_buffer);

    CHECK(pool.Read(first_index) == kFullCoverage);
    CHECK(pool.Read(second_index) == kHalfCoverage);
  }
}

TEST_CASE("Metal ZPD slot reads zero when its segment never counted",
          "[l1][metal][zpd]") {
  @autoreleasepool {
    Harness h = MakeHarness();
    MetalZPDVisibilityPool pool;
    REQUIRE(pool.EnsureInitialized((__bridge void*)h.device, kTestCapacity));
    id<MTLBuffer> visibility =
        (__bridge id<MTLBuffer>)pool.visibility_buffer();

    uint32_t index = 0;
    uint32_t generation = 0;
    size_t offset = 0;
    REQUIRE(pool.Acquire(index, generation, offset));

    // Draws occur, but counting is never enabled for this slot. The acquire
    // zeroing is what makes this a definite 0 rather than stale content.
    id<MTLBuffer> full = MakeVertexBuffer(h.device, Quad(-1.0f, 1.0f, 0.5f));
    id<MTLCommandBuffer> command_buffer = [h.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:MakePass(h, visibility)];
    [encoder setRenderPipelineState:h.pipeline];
    [encoder setDepthStencilState:h.depth_test_less];
    DrawQuad(encoder, full);
    [encoder endEncoding];
    CommitAndWait(command_buffer);

    CHECK(pool.Read(index) == 0);
  }
}

TEST_CASE("Metal ZPD recycled slot is zeroed before reuse", "[l1][metal][zpd]") {
  @autoreleasepool {
    Harness h = MakeHarness();
    MetalZPDVisibilityPool pool;
    REQUIRE(pool.EnsureInitialized((__bridge void*)h.device, kTestCapacity));
    id<MTLBuffer> visibility =
        (__bridge id<MTLBuffer>)pool.visibility_buffer();
    id<MTLBuffer> full = MakeVertexBuffer(h.device, Quad(-1.0f, 1.0f, 0.5f));

    uint32_t index = 0;
    uint32_t generation = 0;
    size_t offset = 0;
    REQUIRE(pool.Acquire(index, generation, offset));

    id<MTLCommandBuffer> command_buffer = [h.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:MakePass(h, visibility)];
    [encoder setRenderPipelineState:h.pipeline];
    [encoder setDepthStencilState:h.depth_test_less];
    [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting
                              offset:offset];
    DrawQuad(encoder, full);
    [encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    [encoder endEncoding];
    CommitAndWait(command_buffer);
    REQUIRE(pool.Read(index) == kFullCoverage);

    // Release and reacquire the same physical slot: the count from the previous
    // occupant must not survive into the next report.
    REQUIRE(pool.Release(index, generation));
    uint32_t reused_index = 0;
    uint32_t reused_generation = 0;
    size_t reused_offset = 0;
    bool found = false;
    for (uint32_t i = 0; i < kTestCapacity && !found; ++i) {
      REQUIRE(pool.Acquire(reused_index, reused_generation, reused_offset));
      found = reused_index == index;
    }
    REQUIRE(found);
    CHECK(pool.Read(index) == 0);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
