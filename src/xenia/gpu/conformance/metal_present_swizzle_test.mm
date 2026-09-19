/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// On-device conformance for the swap-texture swizzle applied before
// presentation (the RequestSwapTexture contract: the guest present shader's
// fetch constant 0 swizzle). Uses the PRODUCTION shader via the shared header.
//
// Expectations are analytic: known texel bytes through a known guest swizzle
// encoding must produce exactly the selected bytes. Every case uses
// per-channel distinct values, because a red/blue exchange (the defect this
// pass fixes on Forza Horizon 2 / Halo 3 / ODST 10:10:10:2 frontbuffers,
// 2026-09-18) is invisible to a greyscale fixture.

#import <Metal/Metal.h>

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_present_swizzle_shader.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

struct Rgba {
  uint8_t r, g, b, a;
  bool operator==(const Rgba& o) const {
    return r == o.r && g == o.g && b == o.b && a == o.a;
  }
};

constexpr uint32_t Swizzle(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  return x | (y << 3) | (z << 6) | (w << 9);
}
constexpr uint32_t kIdentity = Swizzle(0, 1, 2, 3);  // 0x688
constexpr uint32_t kZyxOne = Swizzle(2, 1, 0, 5);    // 0xA0A, the live value

// Runs the production shader over one row of `pixels` with `swizzle`. The
// source is a 2D array (one slice) when `array_source`, like resolve
// snapshots, else a plain 2D texture. Empty when no device is available.
std::vector<Rgba> RunOnDevice(const std::vector<Rgba>& pixels,
                              uint32_t swizzle, bool array_source) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return {};
    }
    const uint32_t width = uint32_t(pixels.size());
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kMetalPresentSwizzleShader
                                              options:nil
                                                error:&err];
    REQUIRE(lib != nil);
    MTLRenderPipelineDescriptor* pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"xe_swz_vs"];
    pd.fragmentFunction = [lib
        newFunctionWithName:(array_source ? @"xe_swz_fs_array"
                                          : @"xe_swz_fs_2d")];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:pd error:&err];
    REQUIRE(ps != nil);

    MTLTextureDescriptor* sd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:1
                                 mipmapped:NO];
    if (array_source) {
      sd.textureType = MTLTextureType2DArray;
      sd.arrayLength = 1;
    }
    sd.usage = MTLTextureUsageShaderRead;
    sd.storageMode = MTLStorageModeShared;
    id<MTLTexture> src = [device newTextureWithDescriptor:sd];
    [src replaceRegion:MTLRegionMake2D(0, 0, width, 1)
           mipmapLevel:0
                 slice:0
             withBytes:pixels.data()
           bytesPerRow:width * 4
         bytesPerImage:width * 4];

    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:1
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModeShared;
    id<MTLTexture> dst = [device newTextureWithDescriptor:dd];

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:ps];
    [enc setFragmentTexture:src atIndex:0];
    uint32_t swz = swizzle;
    [enc setFragmentBytes:&swz length:sizeof(swz) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::vector<Rgba> out(width);
    [dst getBytes:out.data()
        bytesPerRow:width * 4
         fromRegion:MTLRegionMake2D(0, 0, width, 1)
        mipmapLevel:0];
    return out;
  }
}

std::vector<Rgba> Fixture() {
  return {{10, 20, 30, 40}, {200, 150, 100, 50}, {255, 0, 128, 0},
          {0, 255, 1, 255}};
}

// The CPU statement of the contract: 0..3 select R/G/B/A, 4 is zero, 5 is
// one, 6/7 clamp to zero/one like GuestToHostSwizzle.
Rgba Oracle(const Rgba& c, uint32_t swizzle) {
  auto pick = [&](uint32_t s) -> uint8_t {
    if (s > 5) s -= 2;
    switch (s) {
      case 0: return c.r;
      case 1: return c.g;
      case 2: return c.b;
      case 3: return c.a;
      case 4: return 0;
      default: return 255;
    }
  };
  return {pick(swizzle & 7), pick((swizzle >> 3) & 7), pick((swizzle >> 6) & 7),
          pick((swizzle >> 9) & 7)};
}

void CheckSwizzle(uint32_t swizzle, bool array_source) {
  auto in = Fixture();
  auto out = RunOnDevice(in, swizzle, array_source);
  if (out.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(out.size() == in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    INFO("swizzle " << std::hex << swizzle << " array=" << array_source
                    << " texel " << i);
    CHECK(out[i] == Oracle(in[i], swizzle));
  }
}

}  // namespace

TEST_CASE("Identity swizzle is byte-exact", "[metal][present][swizzle]") {
  CheckSwizzle(kIdentity, false);
  CheckSwizzle(kIdentity, true);
}

TEST_CASE("Z,Y,X,1 (the live swap swizzle) exchanges red and blue and forces "
          "alpha opaque",
          "[metal][present][swizzle]") {
  // Explicit expected bytes, independent of the oracle, for the value every
  // censused title uses (0xA0A).
  auto out = RunOnDevice(Fixture(), kZyxOne, true);
  if (out.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(out.size() == 4);
  CHECK(out[0] == Rgba{30, 20, 10, 255});
  CHECK(out[1] == Rgba{100, 150, 200, 255});
  CHECK(out[2] == Rgba{128, 0, 255, 255});
  CHECK(out[3] == Rgba{1, 255, 0, 255});
  CheckSwizzle(kZyxOne, false);
}

TEST_CASE("Constant selectors produce zero and one, including the invalid "
          "6/7 encodings",
          "[metal][present][swizzle]") {
  CheckSwizzle(Swizzle(4, 5, 4, 5), true);
  CheckSwizzle(Swizzle(6, 7, 0, 1), true);
  CheckSwizzle(Swizzle(3, 3, 3, 3), false);
}

TEST_CASE("Every single-channel selection is exact",
          "[metal][present][swizzle]") {
  for (uint32_t s = 0; s < 4; ++s) {
    CheckSwizzle(Swizzle(s, s, s, s), true);
  }
  CheckSwizzle(Swizzle(3, 2, 1, 0), false);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
