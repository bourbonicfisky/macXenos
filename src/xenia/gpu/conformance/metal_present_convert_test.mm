/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// On-device conformance for the format-mismatch present conversion.
//
// Uses the PRODUCTION shader via the shared header, compared against the CPU
// oracle in metal_present_convert.h. The cases cover what the live Halo 3
// capture could not: negatives, values above one, and non-finite inputs, none
// of which appeared in that source (it was fully finite and in [0, 0.15]).
//
// The regression this exists to prevent is specific: the path being replaced
// replicated the RED channel to all three outputs, which on a real title
// discarded a dark purple image into grey. A greyscale fixture cannot detect
// that, so every case here uses per-channel distinct values.

#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_present_convert.h"
#include "xenia/gpu/metal/metal_present_convert_shader.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

struct Rgba8 {
  uint8_t r, g, b, a;
  bool operator==(const Rgba8& o) const {
    return r == o.r && g == o.g && b == o.b && a == o.a;
  }
};

// float -> IEEE half, for building an RGBA16Float source like Halo 3's.
uint16_t ToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
  const uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) {  // Inf / NaN
    return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
  }
  if (exp <= 0) {
    return uint16_t(sign);
  }
  if (exp >= 0x1F) {
    return uint16_t(sign | 0x7C00u);
  }
  return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

// Runs the production conversion shader over one row of RGBA16Float texels.
// Empty if no device is available, so callers skip rather than fail.
std::vector<Rgba8> RunShader(const std::vector<float>& rgba,
                            NSString* fragment_name) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return {};
    }
    const uint32_t width = uint32_t(rgba.size() / 4);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:kMetalPresentConvertShader
                             options:nil
                               error:&err];
    REQUIRE(lib != nil);
    MTLRenderPipelineDescriptor* pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"xe_gray_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:fragment_name];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:pd error:&err];
    REQUIRE(ps != nil);

    // The production shader samples a 2D ARRAY, matching resolve snapshots.
    MTLTextureDescriptor* sd = [[MTLTextureDescriptor alloc] init];
    sd.textureType = MTLTextureType2DArray;
    sd.pixelFormat = MTLPixelFormatRGBA16Float;
    sd.width = width;
    sd.height = 1;
    sd.arrayLength = 1;
    sd.usage = MTLTextureUsageShaderRead;
    sd.storageMode = MTLStorageModeShared;
    id<MTLTexture> src = [device newTextureWithDescriptor:sd];
    std::vector<uint16_t> halves(rgba.size());
    for (size_t i = 0; i < rgba.size(); ++i) {
      halves[i] = ToHalf(rgba[i]);
    }
    [src replaceRegion:MTLRegionMake2D(0, 0, width, 1)
           mipmapLevel:0
                 slice:0
             withBytes:halves.data()
           bytesPerRow:width * 8
         bytesPerImage:width * 8];

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
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::vector<Rgba8> out(width);
    [dst getBytes:out.data()
        bytesPerRow:width * 4
         fromRegion:MTLRegionMake2D(0, 0, width, 1)
        mipmapLevel:0];
    return out;
  }
}

// The shader only ever sees the HALF, so the oracle must be fed the same
// round-tripped value. Comparing against the original float would charge the
// implementation for this fixture's precision, not for its own behavior.
float HalfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0x1F) {
    out = sign | 0x7F800000u | (mant << 13);
  } else if (exp == 0) {
    if (!mant) {
      out = sign;
    } else {
      int e = -1;
      uint32_t m = mant;
      do {
        ++e;
        m <<= 1;
      } while (!(m & 0x400u));
      out = sign | (uint32_t(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

float RoundTripHalf(float f) { return HalfToFloat(ToHalf(f)); }

std::vector<Rgba8> RunConvert(const std::vector<float>& rgba) {
  return RunShader(rgba, @"xe_convert_fs");
}

// The legacy grayscale path, retained as the exact rollback target.
std::vector<Rgba8> RunLegacyGray(const std::vector<float>& rgba) {
  return RunShader(rgba, @"xe_gray_fs");
}

std::vector<Rgba8> OracleOver(const std::vector<float>& rgba) {
  std::vector<Rgba8> out(rgba.size() / 4);
  for (size_t i = 0; i < out.size(); ++i) {
    const PresentConvertPixel p = PresentConvertOracle(
        RoundTripHalf(rgba[i * 4]), RoundTripHalf(rgba[i * 4 + 1]),
        RoundTripHalf(rgba[i * 4 + 2]));
    out[i] = {p.r, p.g, p.b, p.a};
  }
  return out;
}

}  // namespace

TEST_CASE("Per-channel conversion matches the CPU oracle",
          "[metal][present][convert]") {
  // Every texel has three distinct channels, because the bug being fixed
  // replicated red and a greyscale fixture could not see it.
  std::vector<float> in;
  for (int i = 0; i < 64; ++i) {
    in.push_back(float(i) / 64.0f);
    in.push_back(float(i) / 128.0f);
    in.push_back(float(i) / 256.0f);
    in.push_back(1.0f);
  }
  const auto got = RunConvert(in);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got == OracleOver(in));
}

TEST_CASE("The Halo 3 measured range converts exactly",
          "[metal][present][convert]") {
  // The actual component ranges captured from Halo 3's swap source: a dark
  // purple that the previous path rendered grey.
  const std::vector<float> in{0.0684f, 0.0000f, 0.0547f, 1.0f,
                              0.1407f, 0.0401f, 0.1281f, 1.0f,
                              0.1000f, 0.0200f, 0.0900f, 1.0f};
  const auto got = RunConvert(in);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got == OracleOver(in));
  // And the result must not be grey, which is the whole point.
  REQUIRE_FALSE(got[0].r == got[0].g);
  REQUIRE_FALSE(got[0].g == got[0].b);
}

TEST_CASE("Out-of-range values clamp rather than wrap",
          "[metal][present][convert]") {
  const std::vector<float> in{-1.0f, 2.0f,  0.5f,  1.0f,
                              -0.0f, 1.0f,  0.0f,  1.0f,
                              8.0f,  -8.0f, 1.5f,  1.0f};
  const auto got = RunConvert(in);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got == OracleOver(in));
  REQUIRE(got[0].r == 0);
  REQUIRE(got[0].g == 255);
  REQUIRE(got[2].b == 255);
}

TEST_CASE("Non-finite inputs follow the D3D rule", "[metal][present][convert]") {
  // Halo 3's source contained none of these, so the live capture could not
  // exercise them; the contract still has to be pinned.
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();
  const std::vector<float> in{nan_v, 0.5f,   inf_v, 1.0f,
                              -inf_v, nan_v, 0.25f, 1.0f};
  const auto got = RunConvert(in);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got[0].r == 0);    // NaN -> 0
  REQUIRE(got[0].b == 255);  // +Inf -> 1.0
  REQUIRE(got[1].r == 0);    // -Inf -> 0
  REQUIRE(got[1].g == 0);    // NaN -> 0
  REQUIRE(got == OracleOver(in));
}

TEST_CASE("Alpha is forced opaque", "[metal][present][convert]") {
  const std::vector<float> in{0.25f, 0.5f, 0.75f, 0.0f};
  const auto got = RunConvert(in);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got[0].a == 255);
}

TEST_CASE("The rollback path still replicates red, exactly",
          "[metal][present][convert][rollback]") {
  // Pins the ROLLBACK target at the shader level, so "explicit rollback
  // reproduces the previous output" stays verified after all capture
  // instrumentation was removed. Both shaders come from the production header,
  // so this exercises the shipped rollback path rather than a description of
  // it.
  const std::vector<float> in{0.0684f, 0.0000f, 0.0547f, 1.0f,
                              0.1407f, 0.0401f, 0.1281f, 1.0f};
  const auto got = RunLegacyGray(in);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    const uint8_t red = PresentConvertChannelToUnorm8(RoundTripHalf(in[i * 4]));
    REQUIRE(got[i].r == red);
    REQUIRE(got[i].g == red);
    REQUIRE(got[i].b == red);
    REQUIRE(got[i].a == 255);
  }
  // And the two paths must genuinely differ on this chromatic input, or the
  // rollback distinction would be untested.
  const auto converted = RunConvert(in);
  REQUIRE_FALSE(got == converted);
}

TEST_CASE("The oracle rejects red-channel replication",
          "[metal][present][convert]") {
  // Pins the defect itself: for a chromatic source, replicating red is not the
  // oracle's answer. If this ever passes, the bug is back.
  const PresentConvertPixel p = PresentConvertOracle(0.0684f, 0.0f, 0.0547f);
  REQUIRE_FALSE(p.r == p.g);
  REQUIRE_FALSE(p.r == p.b);
  REQUIRE(p.g == 0);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
