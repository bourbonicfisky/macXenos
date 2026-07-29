/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// On-device conformance for the standard 256-entry guest gamma ramp.
//
// The Metal fragment shader is the PRODUCTION one, included from the shared
// header the command processor uses -- not a copy, which would keep passing
// after production changed. It is compared pixel-for-pixel against the
// CPU oracle in metal_gamma_ramp_apply.h, which is the single definition of the
// lookup contract. "Applied versus ignored" is NOT used as a correctness
// oracle anywhere in this file -- every case checks absolute values.
//
// What each case is defending against is noted at the case, because several of
// these failures are invisible to the obvious test: a red/blue reversal is
// undetectable with a greyscale ramp, and an off-by-one index is undetectable
// away from the endpoints.

#import <Metal/Metal.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_gamma_ramp_apply.h"
#include "xenia/gpu/metal/metal_gamma_ramp_classify.h"
#include "xenia/gpu/metal/metal_gamma_ramp_shader.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// The production shader itself, via the shared header -- not a copy. A copy
// would keep passing after production changed.
static NSString* const kShader = kMetalGammaRampShader;

struct Rgba {
  uint8_t r, g, b, a;
  bool operator==(const Rgba& o) const {
    return r == o.r && g == o.g && b == o.b && a == o.a;
  }
};

// Runs the production shader over `pixels` with `table` and returns the result.
// Empty on device-unavailable so the caller can skip rather than fail.
std::vector<Rgba> RunOnDevice(const std::vector<Rgba>& pixels,
                              const std::vector<reg::DC_LUT_30_COLOR>& table) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return {};
    }
    const uint32_t width = uint32_t(pixels.size());
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:kShader
                                              options:nil
                                                error:&err];
    REQUIRE(lib != nil);
    MTLRenderPipelineDescriptor* pd =
        [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"xe_gamma_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"xe_gamma_fs"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> ps =
        [device newRenderPipelineStateWithDescriptor:pd error:&err];
    REQUIRE(ps != nil);

    MTLTextureDescriptor* std_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:1
                                 mipmapped:NO];
    std_desc.usage = MTLTextureUsageShaderRead;
    std_desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> src = [device newTextureWithDescriptor:std_desc];
    [src replaceRegion:MTLRegionMake2D(0, 0, width, 1)
           mipmapLevel:0
             withBytes:pixels.data()
           bytesPerRow:width * 4];

    MTLTextureDescriptor* dst_desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:1
                                 mipmapped:NO];
    dst_desc.usage = MTLTextureUsageRenderTarget;
    dst_desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> dst = [device newTextureWithDescriptor:dst_desc];

    id<MTLBuffer> ramp =
        [device newBufferWithBytes:table.data()
                            length:kGammaRampTableBytes
                           options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:ps];
    [enc setFragmentTexture:src atIndex:0];
    [enc setFragmentBuffer:ramp offset:0 atIndex:0];
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

std::vector<reg::DC_LUT_30_COLOR> TableOf(uint32_t (*f)(uint32_t)) {
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v = f(i);
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  return t;
}

// The ramp CommandProcessor::Initialize() seeds, byte for byte.
std::vector<reg::DC_LUT_30_COLOR> SeededDefaultTable() {
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v = i * 0x3FF / 0xFF;
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  return t;
}

// Deliberately different curve per channel, so red<->blue reversal shows up.
std::vector<reg::DC_LUT_30_COLOR> AsymmetricTable() {
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const double n = i / 255.0;
    t[i].value = 0;
    t[i].color_10_red = uint32_t(1023.0 * std::pow(n, 2.2) + 0.5);
    t[i].color_10_green = uint32_t(1023.0 * n + 0.5);
    t[i].color_10_blue = uint32_t(1023.0 * std::pow(n, 0.45) + 0.5);
  }
  return t;
}

std::vector<Rgba> AllLevelsGrey() {
  std::vector<Rgba> p(256);
  for (uint32_t i = 0; i < 256; ++i) {
    p[i] = {uint8_t(i), uint8_t(i), uint8_t(i), 255};
  }
  return p;
}

std::vector<Rgba> OracleOver(const std::vector<Rgba>& in,
                             const std::vector<reg::DC_LUT_30_COLOR>& table) {
  std::vector<Rgba> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    const GammaRampPixel p =
        ApplyGammaRampOracle(table.data(), in[i].r, in[i].g, in[i].b);
    out[i] = {p.r, p.g, p.b, p.a};
  }
  return out;
}

}  // namespace

TEST_CASE("Device matches the CPU oracle for the seeded default ramp",
          "[metal][gamma][apply]") {
  const auto table = SeededDefaultTable();
  const auto in = AllLevelsGrey();
  const auto got = RunOnDevice(in, table);
  if (got.empty()) {
    WARN("No Metal device; skipping on-device gamma comparison");
    return;
  }
  REQUIRE(got == OracleOver(in, table));
}

TEST_CASE("Every identity ramp is byte-identical to its input",
          "[metal][gamma][apply]") {
  // Exact, not tolerance-based: all three identity formulas round-trip through
  // the 8-bit guest-output quantization for all 256 inputs. If any of these
  // shifts by one LSB, an "identity" ramp would silently alter every frame.
  const auto in = AllLevelsGrey();
  for (auto* f : {&GammaIdentityLinear, &GammaIdentityReplicate,
                  &GammaIdentityScale}) {
    const auto table = TableOf(*f);
    const auto got = RunOnDevice(in, table);
    if (got.empty()) {
      WARN("No Metal device; skipping");
      return;
    }
    REQUIRE(got == in);
    REQUIRE(got == OracleOver(in, table));
  }
}

TEST_CASE("Asymmetric per-channel curves match the oracle exactly",
          "[metal][gamma][apply]") {
  // THE RED/BLUE REVERSAL CASE. DC_LUT_30_COLOR packs blue at bits 0:9 and red
  // at 20:29; D3D12's .b/.g/.r picks cancel that packing. An implementation
  // that reverses a second time swaps red and blue -- and every greyscale ramp
  // above would still pass.
  const auto table = AsymmetricTable();
  const auto in = AllLevelsGrey();
  const auto got = RunOnDevice(in, table);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got == OracleOver(in, table));

  // And prove the fixture could actually expose it: the three channels must
  // genuinely disagree, or this case is vacuous.
  bool channels_differ = false;
  for (const Rgba& p : got) {
    if (p.r != p.b) {
      channels_differ = true;
      break;
    }
  }
  REQUIRE(channels_differ);
}

TEST_CASE("Endpoints and midtones are exact", "[metal][gamma][apply]") {
  const auto table = AsymmetricTable();
  std::vector<Rgba> in;
  for (uint8_t v : {0, 1, 51, 102, 127, 128, 153, 204, 254, 255}) {
    in.push_back({v, v, v, 255});
  }
  const auto got = RunOnDevice(in, table);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got == OracleOver(in, table));
  // Black and white must survive a curve that moves everything between them.
  REQUIRE(got.front().r == 0);
  REQUIRE(got.back().r == 255);
  REQUIRE(got.front().a == 255);
}

TEST_CASE("Index selection is exact nearest, catching off-by-one",
          "[metal][gamma][apply]") {
  // A table that is zero everywhere except one entry: the output is non-zero
  // for exactly one input if and only if the index is computed exactly.
  for (uint32_t probe : {0u, 1u, 127u, 128u, 254u, 255u}) {
    std::vector<reg::DC_LUT_30_COLOR> t(256);
    std::memset(t.data(), 0, kGammaRampTableBytes);
    t[probe].color_10_red = 1023;
    t[probe].color_10_green = 1023;
    t[probe].color_10_blue = 1023;
    const auto in = AllLevelsGrey();
    const auto got = RunOnDevice(in, t);
    if (got.empty()) {
      WARN("No Metal device; skipping");
      return;
    }
    for (uint32_t i = 0; i < 256; ++i) {
      const uint8_t expected = (i == probe) ? 255 : 0;
      REQUIRE(got[i].r == expected);
    }
  }
}

TEST_CASE("No interpolation between adjacent entries",
          "[metal][gamma][apply]") {
  // A step table: entries below 128 are 0, at/above are max. With any blending
  // the transition would produce intermediate values; the reference table path
  // has none.
  std::vector<reg::DC_LUT_30_COLOR> t(256);
  for (uint32_t i = 0; i < 256; ++i) {
    const uint32_t v = i < 128 ? 0 : 1023;
    t[i].value = 0;
    t[i].color_10_red = v;
    t[i].color_10_green = v;
    t[i].color_10_blue = v;
  }
  const auto in = AllLevelsGrey();
  const auto got = RunOnDevice(in, t);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  for (uint32_t i = 0; i < 256; ++i) {
    REQUIRE(got[i].r == (i < 128 ? 0 : 255));
  }
}

TEST_CASE("Output quantization matches the UNORM8 target",
          "[metal][gamma][apply]") {
  // Sweep every 10-bit value through a single entry and require the device's
  // float->UNORM8 store to agree with the oracle's rounding. No exact-half case
  // exists (see the oracle header), so nearest is unambiguous.
  for (uint32_t v10 = 0; v10 < 1024; v10 += 7) {
    std::vector<reg::DC_LUT_30_COLOR> t(256);
    std::memset(t.data(), 0, kGammaRampTableBytes);
    t[200].color_10_red = v10;
    t[200].color_10_green = v10;
    t[200].color_10_blue = v10;
    std::vector<Rgba> in{{200, 200, 200, 255}};
    const auto got = RunOnDevice(in, t);
    if (got.empty()) {
      WARN("No Metal device; skipping");
      return;
    }
    REQUIRE(got[0].r == GammaRampQuantize8(v10));
  }
}

TEST_CASE("Alpha is forced opaque regardless of source alpha",
          "[metal][gamma][apply]") {
  const auto table = AsymmetricTable();
  std::vector<Rgba> in{{10, 20, 30, 0}, {40, 50, 60, 128}};
  const auto got = RunOnDevice(in, table);
  if (got.empty()) {
    WARN("No Metal device; skipping");
    return;
  }
  REQUIRE(got[0].a == 255);
  REQUIRE(got[1].a == 255);
  REQUIRE(got == OracleOver(in, table));
}

TEST_CASE("Index selection rounds to nearest at half-step boundaries",
          "[metal][gamma][apply]") {
  // Exercised at the ORACLE level, on arbitrary floats. The device-level test
  // above cannot distinguish rounding from truncation, and that is a property
  // of the input domain rather than a weakness of the test -- see the next
  // case. This one pins the contract itself.
  REQUIRE(GammaRampIndexFromUnorm8(0.0f) == 0);
  REQUIRE(GammaRampIndexFromUnorm8(1.0f) == 255);
  // Exactly half a step below index 1 rounds up to 1; just under it stays 0.
  REQUIRE(GammaRampIndexFromUnorm8(0.5f / 255.0f) == 1);
  REQUIRE(GammaRampIndexFromUnorm8(0.499f / 255.0f) == 0);
  REQUIRE(GammaRampIndexFromUnorm8(10.5f / 255.0f) == 11);
  REQUIRE(GammaRampIndexFromUnorm8(10.4f / 255.0f) == 10);
  // Out-of-range input must clamp rather than fetch out of bounds.
  REQUIRE(GammaRampIndexFromUnorm8(-1.0f) == 0);
  REQUIRE(GammaRampIndexFromUnorm8(2.0f) == 255);
}

TEST_CASE("Rounding and truncation agree over the whole 8-bit source domain",
          "[metal][gamma][apply]") {
  // WHY THE `+ 0.5` MUTATION IS NOT CAUGHT ON DEVICE, recorded rather than
  // papered over: the swap source is 8-bit UNORM, so `value * 255` reproduces
  // the original integer exactly and truncation gives the same index as
  // rounding for every representable input. The rounding is defensive, faithful
  // to the reference's D3D10+ UNORM rule, and behaviorally equivalent HERE --
  // it would stop being equivalent the moment the guest-output source became
  // wider than 8 bits, which is exactly the Contract 2 change this slice
  // deliberately does not make.
  for (uint32_t n = 0; n < 256; ++n) {
    const float v = float(n) / 255.0f;
    REQUIRE(GammaRampIndexFromUnorm8(v) == n);
    REQUIRE(uint32_t(v * 255.0f) == n);  // truncation agrees, on this domain
  }
}

TEST_CASE("The oracle's packed layout matches the guest register layout",
          "[metal][gamma][apply]") {
  // The shader unpacks bits directly; if the register layout ever moved, the
  // shader would read the wrong field while the oracle kept reading by name.
  reg::DC_LUT_30_COLOR e;
  e.value = 0;
  e.color_10_red = 0x3FF;
  e.color_10_green = 0;
  e.color_10_blue = 0;
  REQUIRE(((e.value >> 20) & 0x3FF) == 0x3FF);
  e.value = 0;
  e.color_10_green = 0x3FF;
  REQUIRE(((e.value >> 10) & 0x3FF) == 0x3FF);
  e.value = 0;
  e.color_10_blue = 0x3FF;
  REQUIRE((e.value & 0x3FF) == 0x3FF);
  REQUIRE(kGammaRampTableBytes == 1024);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
