/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// On-device conformance for the piecewise-linear (PWL) guest gamma ramp.
//
// The shader is the PRODUCTION one from metal_pwl_gamma.h, compiled with the
// production options, and every case compares absolute values against that
// header's CPU oracle. Asymmetric per-channel curves are used throughout: a
// red/blue reversal or a wrong `* 3 + channel` stride is invisible with a
// greyscale table.

#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_pwl_gamma.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

struct Rgba8 {
  uint8_t r, g, b, a;
};

using PwlTable = std::vector<reg::DC_LUT_PWL_DATA>;  // 128 x R,G,B

// A PWL table approximating `f` per channel the way a title builds one: base
// at each entry's first input, delta to the next entry, both with the low six
// bits clear (hardwired to zero).
PwlTable TableOf(double (*fr)(double), double (*fg)(double),
                 double (*fb)(double)) {
  PwlTable t(kPwlGammaEntries * 3);
  double (*fs[3])(double) = {fr, fg, fb};
  for (uint32_t i = 0; i < kPwlGammaEntries; ++i) {
    for (uint32_t c = 0; c < 3; ++c) {
      const double y0 = fs[c](i * 8 / 1023.0);
      const double y1 = fs[c](std::min(1023u, (i + 1) * 8) / 1023.0);
      const uint32_t base = uint32_t(y0 * 1023.0 * 64.0) & ~0x3Fu;
      const uint32_t next = uint32_t(y1 * 1023.0 * 64.0) & ~0x3Fu;
      reg::DC_LUT_PWL_DATA e;
      e.value = 0;
      e.base = std::min(base, 0xFFC0u);
      e.delta = next > base ? std::min(next - base, 0xFFC0u) : 0u;
      t[i * 3 + c] = e;
    }
  }
  return t;
}

double Gamma22(double x) { return std::pow(x, 2.2); }
double Linear(double x) { return x; }
double Gamma045(double x) { return std::pow(x, 0.45); }

PwlTable AsymmetricTable() { return TableOf(Gamma22, Linear, Gamma045); }

// Runs the production shader. `array` selects the one-slice-array RGBA16Float
// source (the resolved-snapshot path) or the 2D RGBA8 source (the same-format
// path). Returns the RGBA8 mailbox pixels; empty if no device.
std::vector<Rgba8> RunOnDevice(const std::vector<float>& rgb, bool array,
                               const PwlTable& table) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return {};
    }
    const uint32_t width = uint32_t(rgb.size() / 3);
    NSError* err = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:kMetalPwlGammaShader
                             options:MetalPwlGammaCompileOptions()
                               error:&err];
    REQUIRE(lib != nil);
    MTLRenderPipelineDescriptor* pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"xe_pwl_vs"];
    pd.fragmentFunction =
        [lib newFunctionWithName:array ? @"xe_pwl_fs_array" : @"xe_pwl_fs_2d"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:pd error:&err];
    REQUIRE(ps != nil);

    MTLTextureDescriptor* sd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:array ? MTLPixelFormatRGBA16Float
                                                 : MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:1
                                 mipmapped:NO];
    sd.usage = MTLTextureUsageShaderRead;
    sd.storageMode = MTLStorageModeShared;
    if (array) {
      sd.textureType = MTLTextureType2DArray;
      sd.arrayLength = 1;
    }
    id<MTLTexture> src = [device newTextureWithDescriptor:sd];
    if (array) {
      std::vector<__fp16> px(width * 4);
      for (uint32_t i = 0; i < width; ++i) {
        px[i * 4 + 0] = __fp16(rgb[i * 3 + 0]);
        px[i * 4 + 1] = __fp16(rgb[i * 3 + 1]);
        px[i * 4 + 2] = __fp16(rgb[i * 3 + 2]);
        px[i * 4 + 3] = __fp16(1.0f);
      }
      [src replaceRegion:MTLRegionMake2D(0, 0, width, 1)
             mipmapLevel:0
                   slice:0
               withBytes:px.data()
             bytesPerRow:width * 8
           bytesPerImage:width * 8];
    } else {
      std::vector<uint8_t> px(width * 4);
      for (uint32_t i = 0; i < width; ++i) {
        for (uint32_t c = 0; c < 3; ++c) {
          px[i * 4 + c] = uint8_t(std::nearbyint(rgb[i * 3 + c] * 255.0f));
        }
        px[i * 4 + 3] = 255;
      }
      [src replaceRegion:MTLRegionMake2D(0, 0, width, 1)
             mipmapLevel:0
               withBytes:px.data()
             bytesPerRow:width * 4];
    }

    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:1
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModeShared;
    id<MTLTexture> dst = [device newTextureWithDescriptor:dd];
    id<MTLBuffer> buf =
        [device newBufferWithBytes:table.data()
                            length:kPwlGammaTableBytes
                           options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:ps];
    [enc setFragmentTexture:src atIndex:0];
    [enc setFragmentBuffer:buf offset:0 atIndex:0];
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

// The value the shader actually reads for `v` from each source format.
float AsStored(float v, bool array) {
  if (array) {
    return float(__fp16(v));
  }
  return std::nearbyint(v * 255.0f) / 255.0f;
}

// Compares device output with the oracle; returns the mismatch count and
// reports the first few.
uint32_t CompareToOracle(const std::vector<float>& rgb, bool array,
                         const PwlTable& table) {
  const std::vector<Rgba8> got = RunOnDevice(rgb, array, table);
  if (got.empty()) {
    WARN("no Metal device; skipped");
    return 0;
  }
  uint32_t mismatches = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const PwlGammaPixel want = ApplyPwlGammaOracle(
        table.data(), AsStored(rgb[i * 3 + 0], array),
        AsStored(rgb[i * 3 + 1], array), AsStored(rgb[i * 3 + 2], array));
    if (got[i].r != want.r || got[i].g != want.g || got[i].b != want.b ||
        got[i].a != 255) {
      if (mismatches++ < 8) {
        UNSCOPED_INFO("pixel " << i << ": got " << int(got[i].r) << ","
                               << int(got[i].g) << "," << int(got[i].b) << ","
                               << int(got[i].a) << " want " << int(want.r)
                               << "," << int(want.g) << "," << int(want.b));
      }
    }
  }
  return mismatches;
}

// Every 10-bit level on all three channels.
std::vector<float> AllLevels10() {
  std::vector<float> rgb(1024 * 3);
  for (uint32_t i = 0; i < 1024; ++i) {
    rgb[i * 3 + 0] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = i / 1023.0f;
  }
  return rgb;
}

}  // namespace

TEST_CASE("PWL: asymmetric curves over every 10-bit level match the oracle",
          "[metal][gamma][pwl]") {
  // Catches a red/blue reversal, a wrong channel stride and an off-by-one
  // entry index, all invisible with a greyscale table.
  REQUIRE(CompareToOracle(AllLevels10(), /*array=*/true, AsymmetricTable()) ==
          0);
}

TEST_CASE("PWL: the multiplier interpolates inside an entry",
          "[metal][gamma][pwl]") {
  // Base 0 everywhere and a large delta: every output comes from the
  // (in10 & 7) * delta / 8 term, so ignoring it (a plain table lookup) or
  // using the wrong low bits fails.
  PwlTable t(kPwlGammaEntries * 3);
  for (auto& e : t) {
    e.value = 0;
    e.delta = 0xFFC0;
  }
  REQUIRE(CompareToOracle(AllLevels10(), /*array=*/true, t) == 0);
  // And the oracle itself really varies inside one entry.
  REQUIRE(PwlGammaChannel(t.data(), 1.0f / 1023.0f, 0) !=
          PwlGammaChannel(t.data(), 2.0f / 1023.0f, 0));
}

TEST_CASE("PWL: NaN, negative and over-range inputs clamp like UNORM",
          "[metal][gamma][pwl]") {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> rgb = {nan, -0.5f, 2.0f, 0.0f, 1.0f, 65504.0f,
                            -65504.0f, 0.5f, nan};
  REQUIRE(CompareToOracle(rgb, /*array=*/true, AsymmetricTable()) == 0);
}

TEST_CASE("PWL: the 2D 8-bit source path matches the oracle",
          "[metal][gamma][pwl]") {
  std::vector<float> rgb(256 * 3);
  for (uint32_t i = 0; i < 256; ++i) {
    rgb[i * 3 + 0] = i / 255.0f;
    rgb[i * 3 + 1] = (255 - i) / 255.0f;
    rgb[i * 3 + 2] = ((i * 37) & 255) / 255.0f;
  }
  REQUIRE(CompareToOracle(rgb, /*array=*/false, AsymmetricTable()) == 0);
}

TEST_CASE("PWL: the upload layout is the guest register layout",
          "[metal][gamma][pwl]") {
  // 128 entries x R,G,B, one DC_LUT_PWL_DATA dword each: base in bits 0:15,
  // delta in bits 16:31, as the shader unpacks them.
  REQUIRE(kPwlGammaTableBytes == 128 * 3 * 4);
  reg::DC_LUT_PWL_DATA e;
  e.value = 0;
  e.base = 0x1240;
  e.delta = 0x0380;
  REQUIRE((e.value & 0xFFFFu) == 0x1240u);
  REQUIRE((e.value >> 16) == 0x0380u);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
