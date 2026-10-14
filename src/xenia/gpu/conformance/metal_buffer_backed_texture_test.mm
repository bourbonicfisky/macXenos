/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 (on-device): C1P G0a — the buffer-backed-texture mechanism microcase.
// Proves, in isolation (no emulator cache/routing), that an RGBA8 MTLTexture
// created OVER a persistent MTLBuffer arena written by the REAL retained
// texture-load compute pipeline samples byte-identically to an ordinary
// MTLTexture holding the same texels, across encoder boundaries in one
// command buffer with no intermediate CPU wait — and measures the sampling
// cost of the buffer-backed view against the frozen G0 threshold.
//
// These cases prove the FUNCTIONAL mechanism only (device-REQUIRED: they
// fail, never skip, without the Metal device). The C1P G0a sampling-COST
// measurement is NOT gated here — wall-clock GPU timing is too high-variance
// for the deterministic suite. It was measured under the frozen threshold
// (paired ratio median AND p95 <= 1.10) and FAILED (median ~1.10-1.15, p95
// ~1.16 across replicates: buffer-backed linear sampling is ~10-15% slower
// than an optimally-tiled texture on the M4), killing C1P; see the G0
// outcome in docs/dev/archive/evidence/metal_texture_cache_integration_adr.md.

#include <algorithm>
#include <cstring>
#include <vector>

#import <Metal/Metal.h>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_load_shader_pipelines.h"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace {

using metal::MetalLoadShaderPipelines;
using LoadConstants = MetalLoadShaderPipelines::LoadConstants;

constexpr uint32_t kRowAlign = 256;  // The retained dest-layout row alignment.

struct BufViewHarness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  MetalLoadShaderPipelines pipelines;
  id<MTLComputePipelineState> sample_pipeline = nil;
  id<MTLComputePipelineState> filter_pipeline = nil;
  bool ok = false;

  BufViewHarness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    queue = [device newCommandQueue];
    if (!queue || !pipelines.Initialize((__bridge void*)device)) {
      return;
    }
    NSString* src = @R"MSL(
      #include <metal_stdlib>
      using namespace metal;
      // Exact texel reads: out[gid] = tex.read(gid).
      kernel void read_texels(texture2d<float, access::read> tex [[texture(0)]],
                              device uchar4* out [[buffer(0)]],
                              constant uint2& dims [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) return;
        float4 v = tex.read(gid);
        out[gid.y * dims.x + gid.x] = uchar4(round(v * 255.0f));
      }
      // Filtered samples at quarter-texel offsets: forces real bilinear.
      kernel void sample_filtered(texture2d<float, access::sample> tex
                                      [[texture(0)]],
                                  device uchar4* out [[buffer(0)]],
                                  constant uint2& dims [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dims.x || gid.y >= dims.y) return;
        constexpr sampler s(coord::normalized, address::clamp_to_edge,
                            filter::linear);
        float2 uv = (float2(gid) + float2(0.75f, 0.25f)) / float2(dims);
        float4 v = tex.sample(s, uv);
        out[gid.y * dims.x + gid.x] = uchar4(round(v * 255.0f));
      }
    )MSL";
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil
                                                error:&err];
    if (!lib) {
      return;
    }
    auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
      NSError* perr = nil;
      id<MTLFunction> fn = [lib newFunctionWithName:name];
      return fn ? [device newComputePipelineStateWithFunction:fn error:&perr]
                : nil;
    };
    sample_pipeline = make(@"read_texels");
    filter_pipeline = make(@"sample_filtered");
    ok = sample_pipeline && filter_pipeline;
  }
};

BufViewHarness& H() {
  static BufViewHarness h;
  return h;
}

#define REQUIRE_DEVICE()                                        \
  do {                                                          \
    if (!H().device) {                                          \
      FAIL("No Metal device: these cases must run on the M4."); \
    }                                                           \
  } while (0)

// Builds tiled guest bytes + the expected linear host bytes via the
// PRODUCTION helpers (Tiled2D + GpuSwap) — the same oracle the dispatch
// conformance uses.
struct GuestImage {
  std::vector<uint8_t> tiled;     // source buffer content (guest layout)
  std::vector<uint8_t> expected;  // linear rows at host_pitch
  uint32_t width, height, pitch_aligned, host_pitch;
};

GuestImage MakeGuestImage(uint32_t width, uint32_t height,
                          xenos::Endian endian) {
  GuestImage img;
  img.width = width;
  img.height = height;
  img.pitch_aligned = (width + 31u) & ~31u;
  img.host_pitch = ((width * 4u) + (kRowAlign - 1)) & ~(kRowAlign - 1);
  int32_t max_off = 0;
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y),
                                             img.pitch_aligned, 2);
      max_off = std::max(max_off, off);
    }
  }
  img.tiled.resize(uint32_t(max_off) + 16);
  for (size_t i = 0; i < img.tiled.size(); ++i) {
    img.tiled[i] = uint8_t(i * 5 + (i >> 7) * 11 + 3);
  }
  img.expected.assign(size_t(img.host_pitch) * height, 0);
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y),
                                             img.pitch_aligned, 2);
      uint32_t word;
      std::memcpy(&word, &img.tiled[off], 4);
      word = xenos::GpuSwap(word, endian);
      std::memcpy(&img.expected[size_t(y) * img.host_pitch + x * 4u], &word,
                  4);
    }
  }
  return img;
}

// Runs the full mechanism once: load-dispatch into a PRIVATE arena at
// arena_offset -> buffer-backed texture view -> cross-encoder texel read of
// BOTH the view and a regular reference texture (blitted from the expected
// bytes) -> guard readback. ONE command buffer, no intermediate CPU wait.
struct MechanismResult {
  std::vector<uint8_t> view_texels, ref_texels, guards;
  bool completed = false;
};

MechanismResult RunMechanism(const GuestImage& img, xenos::Endian endian,
                             uint32_t arena_offset, uint32_t arena_size,
                             bool filtered) {
  auto& h = H();
  MechanismResult r;
  int load_index = h.pipelines.FindIndex("texture_load_32bpb_cs");
  if (load_index < 0) {
    return r;
  }
  id<MTLBuffer> source =
      [h.device newBufferWithBytes:img.tiled.data()
                            length:img.tiled.size()
                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> arena =
      [h.device newBufferWithLength:arena_size
                            options:MTLResourceStorageModePrivate];
  id<MTLBuffer> expected_src =
      [h.device newBufferWithBytes:img.expected.data()
                            length:img.expected.size()
                           options:MTLResourceStorageModeShared];
  MTLTextureDescriptor* ref_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:img.width
                                  height:img.height
                               mipmapped:NO];
  ref_desc.storageMode = MTLStorageModePrivate;
  ref_desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> ref_tex = [h.device newTextureWithDescriptor:ref_desc];

  MTLTextureDescriptor* view_desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:img.width
                                  height:img.height
                               mipmapped:NO];
  view_desc.storageMode = MTLStorageModePrivate;
  view_desc.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> view_tex = [arena newTextureWithDescriptor:view_desc
                                                     offset:arena_offset
                                                bytesPerRow:img.host_pitch];
  if (!source || !arena || !expected_src || !ref_tex || !view_tex) {
    return r;
  }
  uint32_t texel_bytes = img.width * img.height * 4u;
  id<MTLBuffer> out_view =
      [h.device newBufferWithLength:texel_bytes
                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> out_ref =
      [h.device newBufferWithLength:texel_bytes
                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> guard_rb =
      [h.device newBufferWithLength:arena_size
                            options:MTLResourceStorageModeShared];

  LoadConstants constants = {};
  // Packing per the shipped dispatch conformance: tiled bit 0, endian bits
  // [3:2] (texture_load.xesli field layout).
  constants.is_tiled_3d_endian_scale = 1u | (uint32_t(endian) << 2);
  constants.guest_offset = 0;
  constants.guest_pitch_aligned = img.pitch_aligned;
  constants.size_blocks[0] = img.width;
  constants.size_blocks[1] = img.height;
  constants.size_blocks[2] = 1;
  constants.host_offset = arena_offset;
  constants.host_pitch = img.host_pitch;
  constants.height_texels = img.height;

  id<MTLCommandBuffer> cmd = [h.queue commandBuffer];
  // Encoder 0: fill the WHOLE arena with a nonzero sentinel — freshly
  // allocated private-buffer contents are NOT a valid guard oracle.
  constexpr uint8_t kSentinel = 0xA5;
  {
    id<MTLBlitCommandEncoder> fill = [cmd blitCommandEncoder];
    [fill fillBuffer:arena range:NSMakeRange(0, arena_size) value:kSentinel];
    [fill endEncoding];
  }
  // Encoder 1: the REAL load pipeline writes the private arena.
  {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)
                                     h.pipelines.pipeline(size_t(load_index))];
    [enc setBytes:&constants
           length:sizeof(constants)
          atIndex:MetalLoadShaderPipelines::kBufferIndexPushConstants];
    [enc setBuffer:arena
            offset:0
           atIndex:MetalLoadShaderPipelines::kBufferIndexDest];
    [enc setBuffer:source
            offset:0
           atIndex:MetalLoadShaderPipelines::kBufferIndexSource];
    MTLSize group = MTLSizeMake(MetalLoadShaderPipelines::kThreadgroupSizeX,
                                MetalLoadShaderPipelines::kThreadgroupSizeY,
                                1);
    uint32_t threads_x = ((img.width + 7u) / 8u);
    MTLSize groups = MTLSizeMake((threads_x + group.width - 1) / group.width,
                                 (img.height + group.height - 1) / group.height,
                                 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
    [enc endEncoding];
  }
  // Encoder 2 (blit): reference texture from the CPU-expected linear bytes.
  {
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:expected_src
               sourceOffset:0
          sourceBytesPerRow:img.host_pitch
        sourceBytesPerImage:img.expected.size()
                 sourceSize:MTLSizeMake(img.width, img.height, 1)
                  toTexture:ref_tex
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
  }
  // Encoder 3: cross-encoder sampling of BOTH textures — no CPU wait between
  // the arena write and this read; Metal hazard tracking must order them.
  {
    id<MTLComputePipelineState> ps =
        filtered ? h.filter_pipeline : h.sample_pipeline;
    uint32_t dims[2] = {img.width, img.height};
    for (int pass = 0; pass < 2; ++pass) {
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:ps];
      [enc setTexture:(pass == 0 ? view_tex : ref_tex) atIndex:0];
      [enc setBuffer:(pass == 0 ? out_view : out_ref) offset:0 atIndex:0];
      [enc setBytes:dims length:sizeof(dims) atIndex:1];
      MTLSize group = MTLSizeMake(8, 8, 1);
      MTLSize groups = MTLSizeMake((img.width + 7) / 8, (img.height + 7) / 8,
                                   1);
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
      [enc endEncoding];
    }
  }
  // Encoder 4: guard readback of the whole arena.
  {
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:arena
            sourceOffset:0
                toBuffer:guard_rb
       destinationOffset:0
                    size:arena_size];
    [blit endEncoding];
  }
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted || cmd.error != nil) {
    return r;
  }
  r.completed = true;
  r.view_texels.assign(static_cast<uint8_t*>(out_view.contents),
                       static_cast<uint8_t*>(out_view.contents) + texel_bytes);
  r.ref_texels.assign(static_cast<uint8_t*>(out_ref.contents),
                      static_cast<uint8_t*>(out_ref.contents) + texel_bytes);
  r.guards.assign(static_cast<uint8_t*>(guard_rb.contents),
                  static_cast<uint8_t*>(guard_rb.contents) + arena_size);
  return r;
}

TEST_CASE("buffer-backed RGBA8: device linear-texture alignment admits the "
          "retained 256-aligned dest layout",
          "[l1][metal][bufview]") {
  REQUIRE_DEVICE();
  NSUInteger align = [H().device
      minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm];
  INFO("device minimumLinearTextureAlignment(RGBA8Unorm) = " << align);
  REQUIRE(align > 0);
  // The retained layout writes 256-aligned rows and 256-aligned offsets; the
  // device requirement must divide that, or the mechanism is infeasible.
  REQUIRE(kRowAlign % align == 0);
}

TEST_CASE("buffer-backed RGBA8: real load-shader write then cross-encoder "
          "texel reads match a regular texture exactly",
          "[l1][metal][bufview]") {
  REQUIRE_DEVICE();
  REQUIRE(H().ok);
  struct Shape {
    uint32_t w, h;
    xenos::Endian endian;
  };
  // Representative live class shape, an odd shape, and both retained endian
  // modes for the 32bpb class.
  const Shape shapes[] = {
      {320, 180, xenos::Endian::k8in32},
      {47, 33, xenos::Endian::k8in32},
      {320, 180, xenos::Endian::k8in16},
      {1280, 720, xenos::Endian::k8in32},  // full-frame live-class shape
  };
  for (const Shape& sh : shapes) {
    INFO("shape " << sh.w << "x" << sh.h << " endian "
                  << uint32_t(sh.endian));
    GuestImage img = MakeGuestImage(sh.w, sh.h, sh.endian);
    uint32_t arena_offset = 4096;  // 256-aligned nonzero offset.
    uint32_t written = img.host_pitch * img.height;
    uint32_t arena_size = arena_offset + written + 4096;
    MechanismResult r1 =
        RunMechanism(img, sh.endian, arena_offset, arena_size, false);
    REQUIRE(r1.completed);
    REQUIRE(r1.view_texels == r1.ref_texels);
    // Expected-content check (not just view==ref): texel (x,y) equals the
    // production-helper bytes.
    for (uint32_t y = 0; y < img.height; y += 7) {
      for (uint32_t x = 0; x < img.width; x += 5) {
        const uint8_t* got = &r1.view_texels[(size_t(y) * img.width + x) * 4];
        const uint8_t* want =
            &img.expected[size_t(y) * img.host_pitch + x * 4u];
        if (std::memcmp(got, want, 4) != 0) {
          FAIL("texel " << x << "," << y << " differs from production oracle");
        }
      }
    }
    // Guard bytes vs the blitted sentinel (0xA5): pre-offset, post-extent,
    // and per-row tails. Per the retained dest layout the shader WRITES
    // round_up(blocks_w, 8) blocks per row (full thread strides); guard
    // territory per row begins at written_row_bytes, not width*4.
    constexpr uint8_t kSentinel = 0xA5;
    uint32_t written_row_bytes = (((sh.w + 7u) & ~7u)) * 4u;
    for (uint32_t i = 0; i < arena_offset; ++i) {
      if (r1.guards[i] != kSentinel) FAIL("pre-offset guard byte " << i);
    }
    for (uint32_t i = arena_offset + written; i < arena_size; ++i) {
      if (r1.guards[i] != kSentinel) FAIL("post-extent guard byte " << i);
    }
    if (written_row_bytes < img.host_pitch) {
      for (uint32_t y = 0; y < img.height; ++y) {
        const uint8_t* row = &r1.guards[arena_offset + y * img.host_pitch];
        for (uint32_t i = written_row_bytes; i < img.host_pitch; ++i) {
          if (row[i] != kSentinel) {
            FAIL("row " << y << " tail guard byte " << i);
          }
        }
      }
    }
    // Repeat determinism.
    MechanismResult r2 =
        RunMechanism(img, sh.endian, arena_offset, arena_size, false);
    REQUIRE(r2.completed);
    REQUIRE(r2.view_texels == r1.view_texels);
  }
}

TEST_CASE("buffer-backed RGBA8: filtered sampling parity with a regular "
          "texture",
          "[l1][metal][bufview]") {
  REQUIRE_DEVICE();
  REQUIRE(H().ok);
  GuestImage img = MakeGuestImage(320, 180, xenos::Endian::k8in32);
  uint32_t arena_offset = 256;
  uint32_t arena_size = arena_offset + img.host_pitch * img.height + 256;
  MechanismResult r =
      RunMechanism(img, xenos::Endian::k8in32, arena_offset, arena_size, true);
  REQUIRE(r.completed);
  REQUIRE(r.view_texels == r.ref_texels);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
