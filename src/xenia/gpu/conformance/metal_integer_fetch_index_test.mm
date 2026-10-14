/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 (on-device) acceptance oracle for unsigned fixed-point integer
// texture scaling (openspec/changes/implement-unsigned-integer-texture-scaling,
// task 7.3).
//
// The L0 suite proves the packed scale is derived correctly, and a live probe
// proves the backend publishes multiplier 255 at the 13 implicated Just Cause 2
// sites. Those leave exactly one link unproven, and it is the link the whole
// claim rests on:
//
//   does sampling an MTLPixelFormatRGBA8Unorm texture, scaling by the published
//   multiplier, and truncating actually recover the guest's integer index?
//
// Deriving that on paper requires assuming the sampler returns exactly
// byte/255. That assumption is only safe because the 13 sites were measured to
// be POINT sampled with no mips (mag=point min=point mip=basemap aniso=0,
// mip_address=0) — a filtered fetch would interpolate and truncation would floor
// an interpolated value instead. This test pins both halves: it samples with a
// point sampler, and it feeds the exact byte domain measured in the guest LUT
// at 0x110C3000 (0-31 in trace 534307E7_6756, 0-64 in 534307E7_4218).
//
// The scale arithmetic below is the same sequence the DXBC fetch path emits
// for an integer num_format since fetch ABI 8 (Canary's layout): unpack a
// 6-bit field, width = (field & 0xF) + 1, sign = (field >> 4) & 3, multiplier
// = (1 << (width - (signed or biased))) - 1, multiply, round when the fetch
// constant is point sampled (bit 26), then the guest's trunc.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

// Mirrors the DXBC fetch emission. `scale_bits` is the packed value
// texture_util::IntegerScaleBits produces and the backend publishes; the branch
// on it being zero is the production uniform early-out.
constexpr const char* kSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void integer_fetch_index(
    texture2d<float, access::sample> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    device const uint& scale_bits [[buffer(0)]],
    device const uint2* coords [[buffer(1)]],
    device uint4* out_indices [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint2 c = coords[gid];
  // Point-sample at the texel centre, the way a point-filtered guest fetch of
  // an unnormalized 128x128 lookup lands.
  float2 uv = (float2(c) + 0.5f) / float2(tex.get_width(), tex.get_height());
  float4 s = tex.sample(samp, uv);

  // Bit 24 (normalized num_format) is clear for the integer class here.
  if ((scale_bits & ((1u << 26) - 1u)) != 0u &&
      (scale_bits & (1u << 24)) == 0u) {
    uint4 fields = (uint4(scale_bits) >> uint4(0u, 6u, 12u, 18u)) & 0x3Fu;
    uint4 width = (fields & 0xFu) + 1u;
    uint4 sign = (fields >> 4u) & 3u;
    uint4 halved = select(uint4(0u), uint4(1u), sign - 1u < 2u);
    float4 mul = float4((uint4(1u) << (width - halved)) - 1u);
    float4 offset = min(mul, 0.5f) - 0.5f;
    s = s * max(mul, 0.5f) + offset;
    if ((scale_bits & (1u << 26)) != 0u) {
      s = rint(s);
    }
  }

  // The guest's own decode: trunc, then use as an index.
  out_indices[gid] = uint4(trunc(s));
}
)MSL";

struct DeviceFixture {
  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> lib = nil;

  // False ONLY when there is no Metal device. A present device whose library
  // fails to compile is a hard failure — a silently skipping L1 case is the
  // false-green this ladder exists to prevent.
  bool Init() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) return false;
    queue = [dev newCommandQueue];
    NSError* err = nil;
    lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                            options:nil
                              error:&err];
    if (!lib) {
      FAIL("device present but MSL failed to compile: "
           << (err ? err.localizedDescription.UTF8String : "unknown"));
    }
    return true;
  }

  // Uploads `texels` (one byte per channel, RGBA8Unorm) into a width x height
  // texture, runs the kernel over every texel, and returns the truncated index
  // of channel 0 for each.
  std::vector<uint32_t> Run(const std::vector<uint8_t>& texels, uint32_t width,
                            uint32_t height, uint32_t scale_bits) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:width
                                    height:height
                                 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
    REQUIRE(tex != nil);
    [tex replaceRegion:MTLRegionMake2D(0, 0, width, height)
           mipmapLevel:0
             withBytes:texels.data()
           bytesPerRow:width * 4];

    // Point sampling, clamped — matching the measured guest state at the 13
    // sites (mag=point, min=point, mip=basemap, aniso off, no mip chain).
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterNearest;
    sd.magFilter = MTLSamplerMinMagFilterNearest;
    sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> samp = [dev newSamplerStateWithDescriptor:sd];

    std::vector<uint32_t> coords;
    coords.reserve(size_t(width) * height * 2);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        coords.push_back(x);
        coords.push_back(y);
      }
    }
    const size_t n = size_t(width) * height;

    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"integer_fetch_index"];
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    REQUIRE(pso != nil);

    id<MTLBuffer> scale_buf =
        [dev newBufferWithBytes:&scale_bits
                         length:sizeof(scale_bits)
                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> coord_buf =
        [dev newBufferWithBytes:coords.data()
                         length:coords.size() * sizeof(uint32_t)
                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf =
        [dev newBufferWithLength:n * 4 * sizeof(uint32_t)
                         options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setTexture:tex atIndex:0];
    [enc setSamplerState:samp atIndex:0];
    [enc setBuffer:scale_buf offset:0 atIndex:0];
    [enc setBuffer:coord_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, n);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg ? tg : 1, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);

    std::vector<uint32_t> raw(n * 4);
    std::memcpy(raw.data(), out_buf.contents, raw.size() * sizeof(uint32_t));
    std::vector<uint32_t> indices(n);
    for (size_t i = 0; i < n; ++i) {
      indices[i] = raw[i * 4];
    }
    return indices;
  }
};

// The packed scale the backend was measured to publish at all 13 sites:
// k_8_8_8_8, integer num_format, all-unsigned -> multiplier 255 per component.
uint32_t JC2ScaleBits() {
  // mag=point min=point mip=basemap aniso=0, as measured at the 13 sites.
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.format = xenos::TextureFormat::k_8_8_8_8;
  fetch.num_format = 1;
  fetch.swizzle = (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);
  fetch.mag_filter = xenos::TextureFilter::kPoint;
  fetch.min_filter = xenos::TextureFilter::kPoint;
  fetch.mip_filter = xenos::TextureFilter::kBaseMap;
  fetch.aniso_filter = xenos::AnisoFilter::kDisabled;
  return texture_util::IntegerScaleBits(fetch, /*swizzled_signs=*/0);
}

}  // namespace

TEST_CASE("Point-sampled RGBA8Unorm x255 recovers the guest integer index",
          "[integer_scale][device]") {
  DeviceFixture f;
  if (!f.Init()) {
    WARN("no Metal device; skipping on-device integer fetch index test");
    return;
  }

  REQUIRE(JC2ScaleBits() == 0x041C71C7u);

  // Every byte value the guest LUT can hold, so the result is not an argument
  // about the particular values measured in one frame.
  const uint32_t kWidth = 16, kHeight = 16;
  std::vector<uint8_t> texels(size_t(kWidth) * kHeight * 4, 0);
  for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
    uint8_t v = uint8_t(i);  // 0..255
    texels[i * 4 + 0] = v;
    texels[i * 4 + 1] = v;
    texels[i * 4 + 2] = v;
    texels[i * 4 + 3] = v;
  }

  SECTION("scale off: every index collapses to zero — the defect") {
    std::vector<uint32_t> idx = f.Run(texels, kWidth, kHeight, 0);
    REQUIRE(idx.size() == size_t(kWidth) * kHeight);
    // Normalized delivery: byte/255 < 1 for every byte below 255, so trunc
    // yields 0. This reproduces exactly the collapse measured in JC2, where the
    // LUT tops out at 64.
    for (uint32_t i = 0; i < 255; ++i) {
      INFO("byte " << i);
      CHECK(idx[i] == 0);
    }
    // Only an exact 1.0 survives.
    CHECK(idx[255] == 1);
  }

  SECTION("scale on: the index equals the stored byte, for all 256 values") {
    std::vector<uint32_t> idx = f.Run(texels, kWidth, kHeight, JC2ScaleBits());
    REQUIRE(idx.size() == size_t(kWidth) * kHeight);
    for (uint32_t i = 0; i < 256; ++i) {
      INFO("byte " << i);
      CHECK(idx[i] == i);
    }
  }
}

TEST_CASE("The measured JC2 LUT byte domain reproduces as indices",
          "[integer_scale][device]") {
  DeviceFixture f;
  if (!f.Init()) {
    WARN("no Metal device; skipping on-device integer fetch index test");
    return;
  }

  // The domains measured in the guest LUT at 0x110C3000
  // (docs/dev/archive/closed-investigations/metal_integer_fetch_consumer_investigation.md): trace 6756 tops out
  // at 31, trace 4218 at 64.
  struct Case {
    const char* trace;
    uint32_t max_value;
  };
  const Case cases[] = {{"534307E7_6756", 31}, {"534307E7_4218", 64}};

  for (const Case& c : cases) {
    INFO("trace " << c.trace);
    const uint32_t kWidth = 16, kHeight = 16;
    std::vector<uint8_t> texels(size_t(kWidth) * kHeight * 4, 0);
    std::set<uint32_t> expected;
    for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
      uint8_t v = uint8_t(i % (c.max_value + 1));
      texels[i * 4 + 0] = v;
      texels[i * 4 + 1] = v;
      texels[i * 4 + 2] = v;
      texels[i * 4 + 3] = v;
      expected.insert(v);
    }

    std::vector<uint32_t> off = f.Run(texels, kWidth, kHeight, 0);
    std::set<uint32_t> off_domain(off.begin(), off.end());
    // Pinned to a single index — the guest's per-texel selection is inert.
    CHECK(off_domain.size() == 1);
    CHECK(*off_domain.begin() == 0);

    std::vector<uint32_t> on = f.Run(texels, kWidth, kHeight, JC2ScaleBits());
    std::set<uint32_t> on_domain(on.begin(), on.end());
    CHECK(on_domain == expected);
    CHECK(*on_domain.rbegin() == c.max_value);
    CHECK(on_domain.size() == size_t(c.max_value) + 1);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
