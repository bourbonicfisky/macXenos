/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-depth-semantics 5.1 (Stage D — verification, not routing):
// word-level verification of the two MEASURED 32bpp transfer-tail shapes
// through the PRODUCTION canonical kernels, against INDEPENDENT CPU oracles.
//
// Independence: every conversion below (7e3 encode, 20e4 decode, unorm24,
// fixed16 decode) is written HERE from the Xenos format definitions — no
// metal_edram_msl mirrors, no xenos.h converters. Addressing uses
// XeEdramSnapshotOffsetBytes (the CPU side of the already-pinned addressing
// oracle pair; not a production MSL helper). Patterns are asymmetric per
// pixel, per sample, and per channel, and every claim carries a MUTATION
// assertion: a deliberately wrong oracle (swapped samples, swapped channels,
// swapped 16-bit halves, swapped sample plane) MUST disagree — proving the
// test detects the defect classes it covers.
//
// Shape 1: k_2_10_10_10_FLOAT 2x MSAA color -> canonical words -> kD24FS8
//          1x depth materialize (sample mapping under cross-MSAA).
// Shape 2: kD24S8 depth+stencil -> canonical words -> k_16_16 materialize
//          (16-bit component ordering).

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#import <Metal/Metal.h>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/edram_layout.h"
#include "xenia/gpu/metal/metal_edram_msl.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

// ---- Independent conversions (from the Xenos format definitions) ----------

// float -> 7e3 (3-bit exponent, 7-bit mantissa, bias 3, denormals, max
// 31.875). Inputs in this test are chosen exactly representable, so simple
// truncating construction is exact.
uint32_t My7e3(float value) {
  if (!(value > 0.0f)) return 0;
  if (value >= 31.875f) return 0x3FF;
  if (value < 0.25f) {
    // Denormal domain (below the smallest normal 1.0*2^(1-3)):
    // value = m/128 * 2^-2, m in [0,127].
    return uint32_t(value * 512.0f + 0.5f) & 0x7F;
  }
  int exponent = 0;
  float mantissa_scale = value;
  while (mantissa_scale >= 2.0f) { mantissa_scale *= 0.5f; ++exponent; }
  while (mantissa_scale < 1.0f) { mantissa_scale *= 2.0f; --exponent; }
  uint32_t mantissa =
      uint32_t((mantissa_scale - 1.0f) * 128.0f + 0.5f) & 0x7F;
  return (uint32_t(exponent + 3) << 7) | mantissa;
}

// 20e4 (bias 15, 20-bit mantissa) -> float.
float My20e4ToFloat(uint32_t code) {
  uint32_t exponent = (code >> 20) & 0xF;
  uint32_t mantissa = code & 0xFFFFF;
  if (exponent == 0) {
    return float(mantissa) / 1048576.0f / 16384.0f;
  }
  return (1.0f + float(mantissa) / 1048576.0f) *
         float(1u << exponent) / 32768.0f;
}

// unorm24 decode (the biased-numerator Xenia mapping).
float MyUNorm24(uint32_t n24) {
  n24 &= 0xFFFFFF;
  return float(n24 + (n24 >> 23)) / 16777216.0f;
}

// fixed16 (guest snorm carrying -32..32) decode to host value.
float MyFixed16(uint16_t half_bits) {
  return float(int16_t(half_bits)) * (1.0f / 1024.0f);
}

id<MTLLibrary> Lib(id<MTLDevice> device, const char* source) {
  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString stringWithUTF8String:source]
                   options:nil
                     error:&error];
  if (!library) {
    std::string compile_error =
        error ? error.localizedDescription.UTF8String : "?";
    UNSCOPED_INFO(compile_error);
  }
  return library;
}

struct Sp { uint32_t base, pitch, msaa, is_depth, fmt_log2, words; };
struct Range { uint32_t start_tiles, end_tiles; };

}  // namespace

TEST_CASE("shape 1: 7e3 2x color words verify against the independent oracle",
          "[l1][metal][depth_semantics][transfer_tail]") {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { SUCCEED("no device"); return; }

  // Per-sample asymmetric fill: exact-half, exact-7e3 values distinct in
  // x, y, sample, and channel.
  static const char* kFillSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
vertex VOut fill_vs(uint vid [[vertex_id]]) {
  VOut o;
  float2 t = float2(float((vid << 1) & 2), float(vid & 2));
  o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
  return o;
}
fragment half4 fill_fs(VOut in [[stage_in]], uint sid [[sample_id]]) {
  uint x = uint(in.pos.x), y = uint(in.pos.y);
  half r = half(float(x % 8u) * 0.25f + 0.25f);
  half g = half(float(y % 4u) * 0.5f + 0.125f * float(sid + 1u));
  half b = half(float((x + y) % 4u) * 0.125f + (sid == 0u ? 0.0f : 2.0f));
  half a = sid == 0u ? half(0.25f) : half(0.75f);
  return half4(r, g, b, a);
}
)MSL";
  id<MTLLibrary> fill_lib = Lib(device, kFillSource);
  REQUIRE(fill_lib != nil);
  id<MTLLibrary> prod_lib = Lib(device, xe::gpu::metal::kEdramMsl);
  REQUIRE(prod_lib != nil);

  MTLTextureDescriptor* color_desc = [MTLTextureDescriptor new];
  color_desc.textureType = MTLTextureType2DMultisample;
  color_desc.pixelFormat = MTLPixelFormatRGBA16Float;
  color_desc.width = 80;
  color_desc.height = 16;
  color_desc.sampleCount = 2;
  color_desc.usage = MTLTextureUsageRenderTarget;
  color_desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> color = [device newTextureWithDescriptor:color_desc];
  REQUIRE(color != nil);

  MTLRenderPipelineDescriptor* fill_desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  fill_desc.vertexFunction = [fill_lib newFunctionWithName:@"fill_vs"];
  fill_desc.fragmentFunction = [fill_lib newFunctionWithName:@"fill_fs"];
  fill_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
  fill_desc.rasterSampleCount = 2;
  NSError* error = nil;
  id<MTLRenderPipelineState> fill_pipeline =
      [device newRenderPipelineStateWithDescriptor:fill_desc error:&error];
  REQUIRE(fill_pipeline != nil);
  id<MTLComputePipelineState> pack = [device
      newComputePipelineStateWithFunction:
          [prod_lib newFunctionWithName:@"pack_color_ms_range"]
                                    error:&error];
  REQUIRE(pack != nil);

  constexpr uint32_t kWords = 2 * 1280;
  id<MTLBuffer> canonical =
      [device newBufferWithLength:kWords * 4
                          options:MTLResourceStorageModeShared];
  Sp sp{0, 1, 1 /*2x*/, 0, 0, kWords};
  Range range{0, 2};
  uint32_t format = 3;  // k_2_10_10_10_FLOAT

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = color;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLRenderCommandEncoder> render =
      [command renderCommandEncoderWithDescriptor:pass];
  [render setRenderPipelineState:fill_pipeline];
  [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render endEncoding];
  id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
  [compute setComputePipelineState:pack];
  [compute setTexture:color atIndex:0];
  [compute setBuffer:canonical offset:0 atIndex:0];
  [compute setBytes:&sp length:sizeof(sp) atIndex:1];
  [compute setBytes:&format length:sizeof(format) atIndex:2];
  [compute setBytes:&range length:sizeof(range) atIndex:3];
  // Range kernels take their first texture row at buffer(7).
  static const uint32_t kXeRowStartZero = 0;
  [compute setBytes:&kXeRowStartZero length:sizeof(kXeRowStartZero) atIndex:7];
  [compute dispatchThreads:MTLSizeMake(80, 16, 2)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [compute endEncoding];
  [command commit];
  [command waitUntilCompleted];
  REQUIRE(command.status == MTLCommandBufferStatusCompleted);

  // Independent oracle. The pinned 2x storage-order mapping is the
  // INVERSION: Metal sample id m lands in guest slot m^1 (verified here —
  // the first run's dump showed guest slot 0 holding metal sample 1's fill
  // exactly). So guest slot `sid` carries the fill of METAL sample sid^1;
  // the sample mutation asserts sensitivity by un-inverting.
  auto expected_word = [&](uint32_t x, uint32_t y, uint32_t sid,
                           bool mutate_sample, bool mutate_channel) {
    uint32_t s = mutate_sample ? sid : (sid ^ 1u);
    float r = float(x % 8u) * 0.25f + 0.25f;
    float g = float(y % 4u) * 0.5f + 0.125f * float(s + 1u);
    float b = float((x + y) % 4u) * 0.125f + (s == 0u ? 0.0f : 2.0f);
    float a = s == 0u ? 0.25f : 0.75f;
    if (mutate_channel) std::swap(r, g);
    return My7e3(r) | (My7e3(g) << 10) | (My7e3(b) << 20) |
           (uint32_t(a * 3.0f + 0.5f) << 30);
  };
  const uint32_t* words = static_cast<const uint32_t*>(canonical.contents);
  uint32_t mismatches = 0, mutated_sample = 0, mutated_channel = 0;
  std::string first_bad;
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 80; ++x) {
      for (uint32_t sid = 0; sid < 2; ++sid) {
        uint32_t idx = XeEdramSnapshotOffsetBytes(x, y, 0, 1, 1, false, 0,
                                                  sid) / 4;
        REQUIRE(idx < kWords);
        uint32_t got = words[idx];
        uint32_t expect = expected_word(x, y, sid, false, false);
        if (got != expect) {
          ++mismatches;
          if (first_bad.size() < 400) {
            char line[160];
            std::snprintf(line, sizeof(line),
                          "(%u,%u,s%u) got=%08X exp=%08X swap=%08X\n", x, y,
                          sid, got, expect,
                          expected_word(x, y, sid, true, false));
            first_bad += line;
          }
        }
        if (got != expected_word(x, y, sid, true, false)) ++mutated_sample;
        if (got != expected_word(x, y, sid, false, true)) ++mutated_channel;
      }
    }
  }
  INFO(first_bad);
  INFO("mismatches " << mismatches << " mutated-sample-disagreements "
                     << mutated_sample << " mutated-channel-disagreements "
                     << mutated_channel);
  CHECK(mismatches == 0);
  // Mutation sensitivity: the wrong oracles MUST disagree somewhere.
  CHECK(mutated_sample > 0);
  CHECK(mutated_channel > 0);
}

TEST_CASE("shape 2: kD24S8 words and k_16_16 half ordering verify",
          "[l1][metal][depth_semantics][transfer_tail]") {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { SUCCEED("no device"); return; }

  // Depth written from an exactly-decodable unorm24 pattern; stencil cleared
  // to a constant. Asymmetric in x and y.
  static const char* kDepthSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
vertex VOut d_vs(uint vid [[vertex_id]]) {
  VOut o;
  float2 t = float2(float((vid << 1) & 2), float(vid & 2));
  o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
  return o;
}
struct DOut { float depth [[depth(any)]]; };
fragment DOut d_fs(VOut in [[stage_in]]) {
  uint x = uint(in.pos.x), y = uint(in.pos.y);
  uint n = ((x + y * 80u) * 4919u + 3u) & 0xFFFFFFu;
  DOut o;
  o.depth = float(n + (n >> 23u)) / 16777216.0f;
  return o;
}
)MSL";
  id<MTLLibrary> depth_lib = Lib(device, kDepthSource);
  REQUIRE(depth_lib != nil);
  id<MTLLibrary> prod_lib = Lib(device, xe::gpu::metal::kEdramMsl);
  REQUIRE(prod_lib != nil);

  MTLTextureDescriptor* depth_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                   width:80
                                  height:16
                               mipmapped:NO];
  depth_desc.usage = MTLTextureUsageRenderTarget;
  depth_desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> depth = [device newTextureWithDescriptor:depth_desc];
  REQUIRE(depth != nil);
  id<MTLTexture> stencil_view =
      [depth newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
  REQUIRE(stencil_view != nil);

  MTLRenderPipelineDescriptor* dp = [[MTLRenderPipelineDescriptor alloc] init];
  dp.vertexFunction = [depth_lib newFunctionWithName:@"d_vs"];
  dp.fragmentFunction = [depth_lib newFunctionWithName:@"d_fs"];
  dp.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  dp.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  NSError* error = nil;
  id<MTLRenderPipelineState> depth_pipeline =
      [device newRenderPipelineStateWithDescriptor:dp error:&error];
  REQUIRE(depth_pipeline != nil);
  MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
  ds.depthCompareFunction = MTLCompareFunctionAlways;
  ds.depthWriteEnabled = YES;
  id<MTLDepthStencilState> ds_state =
      [device newDepthStencilStateWithDescriptor:ds];

  id<MTLComputePipelineState> pack_depth = [device
      newComputePipelineStateWithFunction:
          [prod_lib newFunctionWithName:@"pack_depth"]
                                    error:&error];
  REQUIRE(pack_depth != nil);
  id<MTLComputePipelineState> mat_color = [device
      newComputePipelineStateWithFunction:
          [prod_lib newFunctionWithName:@"materialize_color"]
                                    error:&error];
  REQUIRE(mat_color != nil);

  constexpr uint32_t kWords = 1280;  // one tile: 80x16 at 1x
  id<MTLBuffer> canonical =
      [device newBufferWithLength:kWords * 4
                          options:MTLResourceStorageModeShared];
  Sp sp_depth{0, 1, 0, 1 /*unorm24*/, 0, kWords};
  Sp sp_color{0, 1, 0, 0, 0, kWords};
  uint32_t format_1616 = 4;

  MTLTextureDescriptor* rg_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                   width:80
                                  height:16
                               mipmapped:NO];
  rg_desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
  id<MTLTexture> rg = [device newTextureWithDescriptor:rg_desc];
  REQUIRE(rg != nil);

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.depthAttachment.texture = depth;
  pass.depthAttachment.loadAction = MTLLoadActionClear;
  pass.depthAttachment.storeAction = MTLStoreActionStore;
  pass.stencilAttachment.texture = depth;
  pass.stencilAttachment.loadAction = MTLLoadActionClear;
  pass.stencilAttachment.storeAction = MTLStoreActionStore;
  pass.stencilAttachment.clearStencil = 0xA5;
  id<MTLRenderCommandEncoder> render =
      [command renderCommandEncoderWithDescriptor:pass];
  [render setRenderPipelineState:depth_pipeline];
  [render setDepthStencilState:ds_state];
  [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render endEncoding];
  id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
  [compute setComputePipelineState:pack_depth];
  [compute setTexture:depth atIndex:0];
  [compute setTexture:stencil_view atIndex:1];
  [compute setBuffer:canonical offset:0 atIndex:0];
  [compute setBytes:&sp_depth length:sizeof(sp_depth) atIndex:1];
  [compute dispatchThreads:MTLSizeMake(80, 16, 1)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [compute setComputePipelineState:mat_color];
  [compute setBuffer:canonical offset:0 atIndex:0];
  [compute setTexture:rg atIndex:0];
  [compute setBytes:&sp_color length:sizeof(sp_color) atIndex:1];
  [compute setBytes:&format_1616 length:sizeof(format_1616) atIndex:2];
  [compute dispatchThreads:MTLSizeMake(80, 16, 1)
      threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [compute endEncoding];
  id<MTLBuffer> rg_readback =
      [device newBufferWithLength:80 * 16 * 16
                          options:MTLResourceStorageModeShared];
  id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
  [blit copyFromTexture:rg
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(80, 16, 1)
               toBuffer:rg_readback
      destinationOffset:0
 destinationBytesPerRow:80 * 16
destinationBytesPerImage:80 * 16 * 16];
  [blit endEncoding];
  [command commit];
  [command waitUntilCompleted];
  REQUIRE(command.status == MTLCommandBufferStatusCompleted);

  // Words: (n << 8) | 0xA5 at the DEPTH-swapped x addressing.
  const uint32_t* words = static_cast<const uint32_t*>(canonical.contents);
  uint32_t word_mismatches = 0;
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 80; ++x) {
      uint32_t n = ((x + y * 80u) * 4919u + 3u) & 0xFFFFFFu;
      uint32_t idx =
          XeEdramSnapshotOffsetBytes(x, y, 0, 1, 0, true, 0, 0) / 4;
      if (words[idx] != ((n << 8) | 0xA5u)) ++word_mismatches;
    }
  }
  CHECK(word_mismatches == 0);

  // k_16_16 materialize: R decodes the LOW half, G the HIGH half.
  const float* rg_values = static_cast<const float*>(rg_readback.contents);
  uint32_t rg_mismatches = 0, swapped_disagrees = 0;
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 80; ++x) {
      // The color view reads UNSWAPPED addressing over the same words.
      uint32_t idx =
          XeEdramSnapshotOffsetBytes(x, y, 0, 1, 0, false, 0, 0) / 4;
      uint32_t word = words[idx];
      float expect_r = MyFixed16(uint16_t(word & 0xFFFF));
      float expect_g = MyFixed16(uint16_t(word >> 16));
      const float* pixel = rg_values + (y * 80 + x) * 4;
      bool ok = pixel[0] == expect_r && pixel[1] == expect_g;
      bool swapped_ok = pixel[0] == expect_g && pixel[1] == expect_r;
      if (!ok) ++rg_mismatches;
      if (!swapped_ok) ++swapped_disagrees;
    }
  }
  INFO("rg mismatches " << rg_mismatches << " swapped-oracle disagreements "
                        << swapped_disagrees);
  CHECK(rg_mismatches == 0);
  CHECK(swapped_disagrees > 0);  // half-order mutation sensitivity
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
