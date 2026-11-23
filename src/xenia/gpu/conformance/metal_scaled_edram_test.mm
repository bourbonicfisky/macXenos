/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Canonical EDRAM at a draw resolution scale
// (implement-metal-draw-resolution-scaling): the shared kEdramMsl compiled with
// the live cache's scaled tile macros must
// - address exactly as edram.xesli XeEdramOffsetBytes does at that scale,
//   injectively and in bounds over every host sample of a surface (so the
//   pack/materialize kernels, which index by host pixel, round-trip),
// - resolve each guest pixel from its representative host pixel, and
// - clear every host sample of a guest rectangle and nothing else.

#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/edram_layout.h"
#include "xenia/gpu/metal/metal_edram_msl.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

// edram.xesli XeEdramOffsetBytes with a resolution scale, wrap=false: the
// independent CPU reference.
uint32_t ScaledEdramOffsetBytes(uint32_t px, uint32_t py, uint32_t base,
                                uint32_t pitch, uint32_t msaa, bool is_depth,
                                uint32_t fmt_log2, uint32_t sample,
                                uint32_t scale_x, uint32_t scale_y) {
  EdramMsaaShifts shifts = GetEdramMsaaShifts(msaa);
  uint32_t sx = (px << shifts.x) + ((sample >> 1u) & 1u);
  uint32_t sy = (py << shifts.y) + (sample & 1u);
  uint32_t tile32_w = xenos::kEdramTileWidthSamples * scale_x;
  uint32_t tile32_h = xenos::kEdramTileHeightSamples * scale_y;
  uint32_t tile_w = tile32_w >> fmt_log2;
  uint32_t tx = sx / tile_w, ty = sy / tile32_h;
  base += ty * pitch + tx;
  sx -= tx * tile_w;
  sy -= ty * tile32_h;
  if (is_depth) {
    uint32_t half = tile_w >> 1u;
    sx = sx >= half ? sx - half : sx + half;
  }
  return (base * (tile32_w * tile32_h) + ((sy * tile_w + sx) << fmt_log2)) *
         4u;
}

constexpr const char* kScaledKernels = R"MSL(
struct EdramIn {
  uint px, py, base, pitch, msaa, is_depth, fmt_log2, sample;
};
kernel void k_scaled_edram_offset(device const EdramIn* in [[buffer(0)]],
                                  device uint* out [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
  EdramIn e = in[i];
  out[i] = xe_edram_offset_bytes(e.px, e.py, e.base, e.pitch, e.msaa,
                                 e.is_depth, e.fmt_log2, e.sample);
}
kernel void k_resolve_host_pixel(device const uint2* in [[buffer(0)]],
                                 device uint2* out [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
  out[i] = xe_resolve_host_pixel(in[i]);
}
)MSL";

struct EdramIn {
  uint32_t px, py, base, pitch, msaa, is_depth, fmt_log2, sample;
};

struct ScaledMsl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;

  bool Init(uint32_t scale_x, uint32_t scale_y) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return false;
    }
    queue = [device newCommandQueue];
    // The same macros MetalRenderTargetCache passes for the session scale.
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.preprocessorMacros = @{
      @"XE_EDRAM_TILE_WIDTH_SAMPLES" : [NSString
          stringWithFormat:@"%uu", xenos::kEdramTileWidthSamples * scale_x],
      @"XE_EDRAM_TILE_HEIGHT_SAMPLES" : [NSString
          stringWithFormat:@"%uu", xenos::kEdramTileHeightSamples * scale_y],
    };
    std::string source = std::string(metal::kEdramMsl) + kScaledKernels;
    NSError* error = nil;
    library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                     options:options
                       error:&error];
    if (!library) {
      FAIL("scaled kEdramMsl failed to compile: "
           << (error ? error.localizedDescription.UTF8String : "unknown"));
    }
    return true;
  }

  id<MTLComputePipelineState> Pipeline(const char* name) {
    NSError* error = nil;
    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    REQUIRE(function != nil);
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    REQUIRE(pipeline != nil);
    return pipeline;
  }

  // One input element per thread at buffer(0), one output at buffer(1).
  std::vector<uint8_t> Run(const char* name, const void* in, size_t in_bytes,
                           size_t count, size_t out_element_bytes) {
    id<MTLComputePipelineState> pipeline = Pipeline(name);
    id<MTLBuffer> in_buffer =
        [device newBufferWithBytes:in
                            length:in_bytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer =
        [device newBufferWithLength:count * out_element_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in_buffer offset:0 atIndex:0];
    [encoder setBuffer:out_buffer offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    REQUIRE(command_buffer.error == nil);
    std::vector<uint8_t> out(count * out_element_bytes);
    std::memcpy(out.data(), out_buffer.contents, out.size());
    return out;
  }
};

const uint32_t kScales[][2] = {{2, 2}, {3, 3}, {2, 1}};

}  // namespace

TEST_CASE("Scaled EDRAM addresser matches edram.xesli at the scale",
          "[l1][metal][draw_resolution_scale]") {
  for (const auto& scale : kScales) {
    ScaledMsl msl;
    if (!msl.Init(scale[0], scale[1])) {
      SUCCEED("no device");
      return;
    }
    std::vector<EdramIn> in;
    for (uint32_t msaa = 0; msaa <= 2; ++msaa) {
      uint32_t samples = msaa == 0 ? 1u : msaa == 1 ? 2u : 4u;
      for (uint32_t fmt = 0; fmt <= 1; ++fmt) {
        for (uint32_t is_depth = 0; is_depth <= 1; ++is_depth) {
          for (uint32_t pitch : {2u, 16u}) {
            for (uint32_t py = 0; py < 40 * scale[1]; py += 3) {
              for (uint32_t px = 0; px < 170 * scale[0]; px += 5) {
                for (uint32_t s = 0; s < samples; ++s) {
                  in.push_back({px, py, 7u, pitch, msaa, is_depth, fmt, s});
                }
              }
            }
          }
        }
      }
    }
    auto bytes = msl.Run("k_scaled_edram_offset", in.data(),
                         in.size() * sizeof(EdramIn), in.size(), 4);
    const uint32_t* gpu = reinterpret_cast<const uint32_t*>(bytes.data());
    uint32_t bad = 0;
    for (size_t i = 0; i < in.size(); ++i) {
      const EdramIn& e = in[i];
      uint32_t cpu = ScaledEdramOffsetBytes(e.px, e.py, e.base, e.pitch,
                                            e.msaa, e.is_depth != 0,
                                            e.fmt_log2, e.sample, scale[0],
                                            scale[1]);
      bad += gpu[i] != cpu;
    }
    INFO("scale " << scale[0] << "x" << scale[1]);
    REQUIRE(bad == 0u);
  }
}

TEST_CASE("Scaled EDRAM addresser is injective over a surface's host samples",
          "[l1][metal][draw_resolution_scale]") {
  for (const auto& scale : kScales) {
    // Every host sample of a guest surface maps to a distinct, in-bounds word
    // of the scaled canonical EDRAM, so pack (host -> canonical) followed by
    // materialize (canonical -> host) is the identity.
    for (uint32_t msaa = 0; msaa <= 2; ++msaa) {
      for (uint32_t fmt = 0; fmt <= 1; ++fmt) {
        for (uint32_t is_depth = 0; is_depth <= 1; ++is_depth) {
          const uint32_t guest_w = 64, guest_h = 32;
          const uint32_t pitch = XeEdramPitchTiles(guest_w, msaa, fmt);
          EdramMsaaShifts shifts = GetEdramMsaaShifts(msaa);
          const uint32_t samples = 1u << (shifts.x + shifts.y);
          const uint32_t words = xenos::kEdramSizeBytes / 4u * scale[0] *
                                 scale[1];
          std::set<uint32_t> seen;
          uint32_t collisions = 0, out_of_bounds = 0;
          for (uint32_t py = 0; py < guest_h * scale[1]; ++py) {
            for (uint32_t px = 0; px < guest_w * scale[0]; ++px) {
              for (uint32_t s = 0; s < samples; ++s) {
                uint32_t word =
                    ScaledEdramOffsetBytes(px, py, 3u, pitch, msaa,
                                           is_depth != 0, fmt, s, scale[0],
                                           scale[1]) /
                    4u;
                for (uint32_t w = 0; w < (1u << fmt); ++w) {
                  out_of_bounds += word + w >= words;
                  collisions += !seen.insert(word + w).second;
                }
              }
            }
          }
          INFO("scale " << scale[0] << "x" << scale[1] << " msaa " << msaa
                        << " fmt_log2 " << fmt << " depth " << is_depth);
          REQUIRE(collisions == 0u);
          REQUIRE(out_of_bounds == 0u);
        }
      }
    }
  }
}

TEST_CASE("Scaled resolve reads the representative host pixel",
          "[l1][metal][draw_resolution_scale]") {
  for (const auto& scale : kScales) {
    ScaledMsl msl;
    if (!msl.Init(scale[0], scale[1])) {
      SUCCEED("no device");
      return;
    }
    std::vector<uint32_t> in;
    for (uint32_t y = 0; y < 9; ++y) {
      for (uint32_t x = 0; x < 13; ++x) {
        in.push_back(x);
        in.push_back(y);
      }
    }
    size_t count = in.size() / 2;
    auto bytes = msl.Run("k_resolve_host_pixel", in.data(),
                         in.size() * sizeof(uint32_t), count, 8);
    const uint32_t* out = reinterpret_cast<const uint32_t*>(bytes.data());
    uint32_t bad = 0;
    for (size_t i = 0; i < count; ++i) {
      uint32_t ex = in[i * 2] * scale[0] + (scale[0] >> 1);
      uint32_t ey = in[i * 2 + 1] * scale[1] + (scale[1] >> 1);
      bad += out[i * 2] != ex || out[i * 2 + 1] != ey;
    }
    INFO("scale " << scale[0] << "x" << scale[1]);
    REQUIRE(bad == 0u);
  }
}

TEST_CASE("Scaled canonical clear covers exactly the rectangle's host samples",
          "[l1][metal][draw_resolution_scale]") {
  for (const auto& scale : kScales) {
    ScaledMsl msl;
    if (!msl.Init(scale[0], scale[1])) {
      SUCCEED("no device");
      return;
    }
    for (uint32_t msaa : {0u, 2u}) {
      for (uint32_t is_depth : {0u, 1u}) {
        struct SurfaceParams {
          uint32_t base, pitch, msaa, is_depth, fmt_log2, words;
        };
        struct ClearParams {
          SurfaceParams surface;
          uint32_t source_x, source_y, width, height, clear_lo, clear_hi;
        };
        const uint32_t words =
            xenos::kEdramSizeBytes / 4u * scale[0] * scale[1];
        const uint32_t pitch = XeEdramPitchTiles(96, msaa, 0);
        ClearParams params{{5u, pitch, msaa, is_depth, 0u, words},
                           11u, 6u, 23u, 9u, 0xC1EA12u, 0u};
        id<MTLBuffer> canonical =
            [msl.device newBufferWithLength:size_t(words) * 4u
                                    options:MTLResourceStorageModeShared];
        std::memset(canonical.contents, 0, size_t(words) * 4u);
        id<MTLComputePipelineState> pipeline =
            msl.Pipeline("resolve_canonical_clear");
        id<MTLCommandBuffer> command_buffer = [msl.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:canonical offset:0 atIndex:0];
        [encoder setBytes:&params length:sizeof(params) atIndex:1];
        // The live dispatch: the guest rectangle at the draw resolution scale.
        uint32_t host_w = params.width * scale[0];
        uint32_t host_h = params.height * scale[1];
        [encoder dispatchThreadgroups:MTLSizeMake((host_w + 7) / 8,
                                                  (host_h + 7) / 8,
                                                  1u << msaa)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        REQUIRE(command_buffer.error == nil);

        std::set<uint32_t> expected;
        for (uint32_t y = 0; y < host_h; ++y) {
          for (uint32_t x = 0; x < host_w; ++x) {
            for (uint32_t s = 0; s < (1u << msaa); ++s) {
              expected.insert(
                  ScaledEdramOffsetBytes(
                      params.source_x * scale[0] + x,
                      params.source_y * scale[1] + y, params.surface.base,
                      pitch, msaa, is_depth != 0, 0u, s, scale[0], scale[1]) /
                  4u);
            }
          }
        }
        const uint32_t* data = static_cast<const uint32_t*>(canonical.contents);
        uint32_t missing = 0, stray = 0;
        for (uint32_t i = 0; i < words; ++i) {
          bool cleared = data[i] == params.clear_lo;
          bool want = expected.count(i) != 0;
          missing += want && !cleared;
          stray += cleared && !want;
        }
        INFO("scale " << scale[0] << "x" << scale[1] << " msaa " << msaa
                      << " depth " << is_depth);
        REQUIRE(missing == 0u);
        REQUIRE(stray == 0u);
      }
    }
  }
}

TEST_CASE("Packed color snapshots retain every scaled subpixel",
          "[l1][metal][draw_resolution_scale][packed_snapshot]") {
  for (uint32_t scale : {1u, 2u, 3u}) {
    ScaledMsl msl;
    REQUIRE(msl.Init(scale, scale));
    const uint32_t width = 2 * scale, height = 2 * scale;
    std::vector<uint32_t> expected(width * height);
    std::vector<uint32_t> edram(80 * 16 * scale * scale);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint32_t i = y * width + x;
        // Distinct 10-bit channels and all four alpha codes, including ends.
        const uint32_t word = ((i * 29) & 1023) |
            (((1023 - i * 17) & 1023) << 10) |
            (((i * 113) & 1023) << 20) | ((i & 3) << 30);
        expected[i] = word;
        edram[ScaledEdramOffsetBytes(x, y, 0, 1, 0, false, 0, 0,
                                    scale, scale) / 4] = word;
      }
    }
    id<MTLBuffer> source = [msl.device newBufferWithBytes:edram.data()
        length:edram.size() * 4 options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2DArray;
    desc.pixelFormat = MTLPixelFormatRGB10A2Unorm;
    desc.width = width;
    desc.height = height;
    desc.arrayLength = 1;
    desc.storageMode = MTLStorageModeShared;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> output = [msl.device newTextureWithDescriptor:desc];
    REQUIRE(output != nil);
    desc.textureType = MTLTextureType2D;
    id<MTLTexture> direct_source = [msl.device newTextureWithDescriptor:desc];
    REQUIRE(direct_source != nil);
    [direct_source replaceRegion:MTLRegionMake2D(0, 0, width, height)
        mipmapLevel:0 withBytes:expected.data() bytesPerRow:width * 4];
    // Madden reinterprets an HDR attachment as packed UNORM at resolve time.
    // Preserve the 7e3 words through the direct source's RGBA16Float container.
    desc.pixelFormat = MTLPixelFormatRGBA16Float;
    id<MTLTexture> hdr_source = [msl.device newTextureWithDescriptor:desc];
    REQUIRE(hdr_source != nil);
    std::vector<_Float16> hdr(expected.size() * 4);
    for (size_t i = 0; i < expected.size(); ++i) {
      for (uint32_t c = 0; c < 3; ++c) {
        hdr[i * 4 + c] = _Float16(xenos::Float7e3To32(
            (expected[i] >> (c * 10)) & 1023));
      }
      hdr[i * 4 + 3] = _Float16(float(expected[i] >> 30) / 3.0f);
    }
    [hdr_source replaceRegion:MTLRegionMake2D(0, 0, width, height)
        mipmapLevel:0 withBytes:hdr.data() bytesPerRow:width * 8];
    // CanonicalResolveParams (29 words), identical to the production layout.
    uint32_t params[29] = {};
    params[1] = 1;  // pitch in tiles
    params[5] = uint32_t(edram.size());
    params[10] = 2;  // source k_2_10_10_10
    float bias = 1.0f;
    std::memcpy(&params[13], &bias, 4);
    params[14] = 2;  // guest width
    params[15] = 2;  // guest height
    params[16] = 7;  // destination packed 10:10:10:2
    uint32_t origin[2] = {}, rt_format = 2;
    for (uint32_t source_kind : {0u, 1u, 2u}) {
      bool direct = source_kind != 0;
      rt_format = source_kind == 2 ? 3 : 2;
      auto pipeline = msl.Pipeline(direct ? "resolve_direct_color_cs_1x"
                                         : "resolve_canonical_color_cs");
      auto command = [msl.queue commandBuffer];
      auto encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:source offset:0 atIndex:0];
      [encoder setBytes:params length:sizeof(params) atIndex:1];
      [encoder setBytes:origin length:sizeof(origin) atIndex:2];
      [encoder setBytes:&rt_format length:sizeof(rt_format) atIndex:3];
      [encoder setTexture:output atIndex:0];
      if (direct) {
        [encoder setTexture:source_kind == 2 ? hdr_source : direct_source
                    atIndex:1];
      }
      [encoder dispatchThreads:MTLSizeMake(width, height, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      REQUIRE(command.error == nil);
      std::vector<uint32_t> actual(expected.size());
      [output getBytes:actual.data() bytesPerRow:width * 4
          bytesPerImage:width * height * 4
          fromRegion:MTLRegionMake2D(0, 0, width, height)
          mipmapLevel:0 slice:0];
      INFO("scale=" << scale << " source_kind=" << source_kind);
      CHECK(actual == expected);
    }
  }
}

TEST_CASE("Direct extended color and fused resolves match canonical at scale",
          "[l1][metal][draw_resolution_scale][direct_fused]") {
  for (uint32_t scale : {1u, 2u, 3u}) {
    ScaledMsl msl;
    REQUIRE(msl.Init(scale, scale));
    const uint32_t gw = 16, gh = 8;
    const uint32_t hw = gw * scale, hh = gh * scale;
    const uint32_t pitch_tiles = 1;
    const uint32_t edram_words = 80 * 16 * scale * scale;
    // Test 64bpp raw-view k_16_16_16_16_FLOAT (format 7): RGBA16Uint texture
    // packed into canonical EDRAM vs direct 64_raw_1x and fused+fwd kernels.
    std::vector<uint16_t> raw_tex(hw * hh * 4);
    std::vector<uint32_t> edram(edram_words, 0u);
    for (uint32_t y = 0; y < hh; ++y) {
      for (uint32_t x = 0; x < hw; ++x) {
        uint32_t i = y * hw + x;
        uint16_t h0 = uint16_t(0x3C00u + ((i * 13u) & 0x3FFu));
        uint16_t h1 = uint16_t(0x3800u + ((i * 29u) & 0x3FFu));
        uint16_t h2 = uint16_t(0x3400u + ((i * 47u) & 0x3FFu));
        uint16_t h3 = 0x3C00u;
        raw_tex[i * 4 + 0] = h0;
        raw_tex[i * 4 + 1] = h1;
        raw_tex[i * 4 + 2] = h2;
        raw_tex[i * 4 + 3] = h3;
        uint32_t off = ScaledEdramOffsetBytes(x, y, 0, pitch_tiles, 0, false, 1,
                                              0, scale, scale) /
                       4u;
        edram[off] = uint32_t(h0) | (uint32_t(h1) << 16u);
        edram[off + 1] = uint32_t(h2) | (uint32_t(h3) << 16u);
      }
    }
    id<MTLBuffer> canonical =
        [msl.device newBufferWithBytes:edram.data()
                                length:edram.size() * 4u
                               options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = MTLPixelFormatRGBA16Uint;
    td.width = hw;
    td.height = hh;
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> direct_tex = [msl.device newTextureWithDescriptor:td];
    REQUIRE(direct_tex != nil);
    [direct_tex replaceRegion:MTLRegionMake2D(0, 0, hw, hh)
                  mipmapLevel:0
                    withBytes:raw_tex.data()
                  bytesPerRow:hw * 8];

    auto make_snap = [&]() {
      MTLTextureDescriptor* sd = [[MTLTextureDescriptor alloc] init];
      sd.textureType = MTLTextureType2DArray;
      sd.pixelFormat = MTLPixelFormatRGBA16Float;
      sd.width = hw;
      sd.height = hh;
      sd.arrayLength = 1;
      sd.storageMode = MTLStorageModeShared;
      sd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      return [msl.device newTextureWithDescriptor:sd];
    };
    auto make_fwd = [&]() {
      MTLTextureDescriptor* fd = [[MTLTextureDescriptor alloc] init];
      fd.textureType = MTLTextureType2D;
      fd.pixelFormat = MTLPixelFormatRGBA16Uint;
      fd.width = gw;
      fd.height = gh;
      fd.storageMode = MTLStorageModeShared;
      fd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      return [msl.device newTextureWithDescriptor:fd];
    };
    const uint32_t dest_extent = 32u * 32u * 8u;
    auto make_buf = [&]() {
      id<MTLBuffer> b =
          [msl.device newBufferWithLength:dest_extent
                                  options:MTLResourceStorageModeShared];
      std::memset(b.contents, 0, dest_extent);
      return b;
    };
    id<MTLTexture> snap_ref = make_snap();
    id<MTLTexture> snap_dir = make_snap();
    id<MTLTexture> snap_fused = make_snap();
    id<MTLTexture> fwd_ref = make_fwd();
    id<MTLTexture> fwd_fused = make_fwd();
    id<MTLBuffer> buf_ref = make_buf();
    id<MTLBuffer> buf_dir = make_buf();
    id<MTLBuffer> buf_fused = make_buf();

    uint32_t params[29] = {};
    params[0] = 0;            // base
    params[1] = pitch_tiles;  // pitch
    params[2] = 0;            // msaa 1x
    params[3] = 0;            // is_depth
    params[4] = 1;            // fmt_log2 (64bpp)
    params[5] = edram_words;
    params[10] = 7;  // color_format k_16_16_16_16_FLOAT
    params[11] = 1;  // dest_swap
    float bias = 1.0f;
    std::memcpy(&params[13], &bias, 4);
    params[14] = gw;
    params[15] = gh;
    params[16] = 32;  // k_16_16_16_16_FLOAT texture dest
    params[17] = 3;   // dest_bpp_log2 = 8 bytes
    params[18] = 2;   // 8in32 endian
    params[19] = 0;   // raw_copy = 0
    params[20] = 1;   // scatter_enable
    params[21] = 0;   // dest_base
    params[22] = 32;  // dest_pitch_aligned
    params[25] = dest_extent;
    uint32_t origin[2] = {0, 0};
    uint32_t rt_format = 7;
    uint32_t snap_pack_1010102 = 0;
    uint32_t fwd_params[5] = {2u, 2u, gw, gh, 0u};

    id<MTLCommandBuffer> cb = [msl.queue commandBuffer];
    // Reference: canonical cs + canonical packed_fwd.
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline("resolve_canonical_color_cs")];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBytes:params length:sizeof(params) atIndex:1];
      [enc setBytes:origin length:sizeof(origin) atIndex:2];
      [enc setTexture:snap_ref atIndex:0];
      [enc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline(
                                       "resolve_canonical_color_packed_fwd")];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:buf_ref offset:0 atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:2];
      [enc setBytes:fwd_params length:sizeof(fwd_params) atIndex:5];
      [enc setTexture:fwd_ref atIndex:2];
      [enc dispatchThreads:MTLSizeMake(gw, gh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    // Direct unfused: direct 64_raw_1x cs + packed.
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline(
                                       "resolve_direct_color_cs_64_raw_1x")];
      [enc setTexture:direct_tex atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:1];
      [enc setBytes:origin length:sizeof(origin) atIndex:2];
      [enc setBytes:&rt_format length:sizeof(rt_format) atIndex:3];
      [enc setTexture:snap_dir atIndex:0];
      [enc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline(
                                       "resolve_direct_color_packed_64_raw_1x")];
      [enc setTexture:direct_tex atIndex:0];
      [enc setBuffer:buf_dir offset:0 atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:2];
      [enc setBytes:&rt_format length:sizeof(rt_format) atIndex:3];
      [enc dispatchThreads:MTLSizeMake(gw, gh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    // Direct fused + forward in one dispatch.
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:
               msl.Pipeline("resolve_direct_color_fused_fwd_64_raw_1x")];
      [enc setTexture:direct_tex atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:1];
      [enc setBytes:origin length:sizeof(origin) atIndex:2];
      [enc setBytes:&rt_format length:sizeof(rt_format) atIndex:3];
      [enc setBuffer:buf_fused offset:0 atIndex:4];
      [enc setBytes:fwd_params length:sizeof(fwd_params) atIndex:5];
      [enc setBytes:&snap_pack_1010102
             length:sizeof(snap_pack_1010102)
            atIndex:6];
      [enc setTexture:snap_fused atIndex:0];
      [enc setTexture:fwd_fused atIndex:2];
      [enc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);

    std::vector<uint8_t> s_ref(hw * hh * 8), s_dir(hw * hh * 8),
        s_fus(hw * hh * 8);
    [snap_ref getBytes:s_ref.data()
           bytesPerRow:hw * 8
         bytesPerImage:hw * hh * 8
            fromRegion:MTLRegionMake2D(0, 0, hw, hh)
           mipmapLevel:0
                 slice:0];
    [snap_dir getBytes:s_dir.data()
           bytesPerRow:hw * 8
         bytesPerImage:hw * hh * 8
            fromRegion:MTLRegionMake2D(0, 0, hw, hh)
           mipmapLevel:0
                 slice:0];
    [snap_fused getBytes:s_fus.data()
             bytesPerRow:hw * 8
           bytesPerImage:hw * hh * 8
              fromRegion:MTLRegionMake2D(0, 0, hw, hh)
             mipmapLevel:0
                   slice:0];
    std::vector<uint8_t> f_ref(gw * gh * 8), f_fus(gw * gh * 8);
    [fwd_ref getBytes:f_ref.data()
          bytesPerRow:gw * 8
           fromRegion:MTLRegionMake2D(0, 0, gw, gh)
          mipmapLevel:0];
    [fwd_fused getBytes:f_fus.data()
            bytesPerRow:gw * 8
             fromRegion:MTLRegionMake2D(0, 0, gw, gh)
            mipmapLevel:0];
    INFO("scale=" << scale);
    CHECK(s_dir == s_ref);
    CHECK(s_fus == s_ref);
    CHECK(std::memcmp(buf_dir.contents, buf_ref.contents, dest_extent) == 0);
    CHECK(std::memcmp(buf_fused.contents, buf_ref.contents, dest_extent) == 0);
    CHECK(f_fus == f_ref);
  }
}

TEST_CASE("Direct depth and fused depth resolves match canonical at scale",
          "[l1][metal][draw_resolution_scale][direct_depth]") {
  for (uint32_t scale : {1u, 2u, 3u}) {
    ScaledMsl msl;
    REQUIRE(msl.Init(scale, scale));
    const uint32_t gw = 16, gh = 8;
    const uint32_t hw = gw * scale, hh = gh * scale;
    const uint32_t edram_words = 80 * 16 * scale * scale;
    // Create private Depth32Float_Stencil8 attachment, populate it via a
    // render pass (varying depth + stencil), pack to canonical, then compare
    // direct depth cs/raw/fused vs canonical cs/raw bit-for-bit.
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                     width:hw
                                    height:hh
                                 mipmapped:NO];
    dd.storageMode = MTLStorageModePrivate;
    dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
               MTLTextureUsagePixelFormatView;
    id<MTLTexture> depth_tex = [msl.device newTextureWithDescriptor:dd];
    REQUIRE(depth_tex != nil);
    id<MTLTexture> stencil_view =
        [depth_tex newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    REQUIRE(stencil_view != nil);

    // Fill canonical EDRAM with distinct D24FS8 words, materialize into
    // depth_tex, then compare direct resolve against canonical resolve.
    std::vector<uint32_t> init_edram(edram_words, 0u);
    for (uint32_t y = 0; y < hh; ++y) {
      for (uint32_t x = 0; x < hw; ++x) {
        uint32_t i = y * hw + x;
        uint32_t d24 = 0x400000u + ((i * 997u) & 0x3FFFFFu);
        uint32_t sten = (i * 37u + 11u) & 0xFFu;
        uint32_t off = ScaledEdramOffsetBytes(x, y, 0, 1, 0, true, 0, 0, scale,
                                              scale) /
                       4u;
        init_edram[off] = (d24 << 8u) | sten;
      }
    }
    id<MTLBuffer> canonical =
        [msl.device newBufferWithBytes:init_edram.data()
                                length:init_edram.size() * 4u
                               options:MTLResourceStorageModeShared];
    // Populate depth_tex with varying per-pixel depth and stencil via
    // materialize_depth_sten_ms_range_fs (msaa=0, sample_count=1) so every
    // pixel has distinct depth24 and stencil8 values, then pack to canonical.
    MTLRenderPipelineDescriptor* rpd =
        [[MTLRenderPipelineDescriptor alloc] init];
    rpd.vertexFunction = [msl.library newFunctionWithName:@"xe_depth_vs"];
    rpd.fragmentFunction =
        [msl.library newFunctionWithName:@"materialize_depth_sten_ms_range_fs"];
    rpd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    rpd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    rpd.rasterSampleCount = 1;
    NSError* rpe = nil;
    id<MTLRenderPipelineState> mat_pso =
        [msl.device newRenderPipelineStateWithDescriptor:rpd error:&rpe];
    REQUIRE(mat_pso != nil);
    MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction = MTLCompareFunctionAlways;
    dsd.depthWriteEnabled = YES;
    MTLStencilDescriptor* sd_s = [[MTLStencilDescriptor alloc] init];
    sd_s.stencilCompareFunction = MTLCompareFunctionAlways;
    sd_s.depthStencilPassOperation = MTLStencilOperationReplace;
    sd_s.writeMask = 0xFF;
    dsd.frontFaceStencil = sd_s;
    dsd.backFaceStencil = sd_s;
    id<MTLDepthStencilState> dss =
        [msl.device newDepthStencilStateWithDescriptor:dsd];
    {
      MTLRenderPassDescriptor* rp =
          [MTLRenderPassDescriptor renderPassDescriptor];
      rp.depthAttachment.texture = depth_tex;
      rp.depthAttachment.loadAction = MTLLoadActionDontCare;
      rp.depthAttachment.storeAction = MTLStoreActionStore;
      rp.stencilAttachment.texture = depth_tex;
      rp.stencilAttachment.loadAction = MTLLoadActionDontCare;
      rp.stencilAttachment.storeAction = MTLStoreActionStore;
      id<MTLCommandBuffer> cb0 = [msl.queue commandBuffer];
      id<MTLRenderCommandEncoder> renc =
          [cb0 renderCommandEncoderWithDescriptor:rp];
      uint32_t sp[6] = {0, 1, 0, 2, 0, edram_words};
      uint32_t rng[2] = {0, 2048};
      uint32_t hl = 0;
      id<MTLBuffer> dummy_lane =
          [msl.device newBufferWithLength:16
                                  options:MTLResourceStorageModeShared];
      [renc setRenderPipelineState:mat_pso];
      [renc setDepthStencilState:dss];
      [renc setFragmentBuffer:canonical offset:0 atIndex:0];
      [renc setFragmentBytes:sp length:sizeof(sp) atIndex:1];
      [renc setFragmentBytes:rng length:sizeof(rng) atIndex:2];
      [renc setFragmentBytes:&hl length:sizeof(hl) atIndex:3];
      [renc setFragmentBuffer:dummy_lane offset:0 atIndex:4];
      [renc drawPrimitives:MTLPrimitiveTypeTriangle
               vertexStart:0
               vertexCount:3];
      [renc endEncoding];
      auto penc = [cb0 computeCommandEncoder];
      [penc setComputePipelineState:msl.Pipeline("pack_depth")];
      [penc setTexture:depth_tex atIndex:0];
      [penc setTexture:stencil_view atIndex:1];
      [penc setBuffer:canonical offset:0 atIndex:0];
      [penc setBytes:sp length:sizeof(sp) atIndex:1];
      [penc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [penc endEncoding];
      [cb0 commit];
      [cb0 waitUntilCompleted];
      REQUIRE(cb0.error == nil);
    }

    auto make_snap = [&]() {
      MTLTextureDescriptor* sd = [[MTLTextureDescriptor alloc] init];
      sd.textureType = MTLTextureType2DArray;
      sd.pixelFormat = MTLPixelFormatR32Float;
      sd.width = hw;
      sd.height = hh;
      sd.arrayLength = 1;
      sd.storageMode = MTLStorageModeShared;
      sd.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      return [msl.device newTextureWithDescriptor:sd];
    };
    const uint32_t dest_extent = 32u * 32u * 4u;
    auto make_buf = [&]() {
      id<MTLBuffer> b =
          [msl.device newBufferWithLength:dest_extent
                                  options:MTLResourceStorageModeShared];
      std::memset(b.contents, 0, dest_extent);
      return b;
    };
    id<MTLTexture> snap_ref = make_snap();
    id<MTLTexture> snap_dir = make_snap();
    id<MTLTexture> snap_fused = make_snap();
    id<MTLBuffer> buf_ref = make_buf();
    id<MTLBuffer> buf_dir = make_buf();
    id<MTLBuffer> buf_fused = make_buf();

    uint32_t params[29] = {};
    params[0] = 0;  // base
    params[1] = 1;  // pitch
    params[2] = 0;  // msaa 1x
    params[3] = 2;  // is_depth = kD24FS8
    params[4] = 0;
    params[5] = edram_words;
    float bias = 1.0f;
    std::memcpy(&params[13], &bias, 4);
    params[14] = gw;
    params[15] = gh;
    params[17] = 2;  // 32bpp
    params[18] = 2;  // 8in32 endian
    params[19] = 1;  // raw
    params[20] = 1;  // scatter
    params[22] = 32;
    params[25] = dest_extent;
    uint32_t origin[2] = {0, 0};

    id<MTLCommandBuffer> cb = [msl.queue commandBuffer];
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline("resolve_canonical_depth_cs")];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBytes:params length:sizeof(params) atIndex:1];
      [enc setBytes:origin length:sizeof(origin) atIndex:2];
      [enc setTexture:snap_ref atIndex:0];
      [enc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline("resolve_canonical_depth_raw")];
      [enc setBuffer:canonical offset:0 atIndex:0];
      [enc setBuffer:buf_ref offset:0 atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(gw, gh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline("resolve_direct_depth_cs_1x")];
      [enc setTexture:depth_tex atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:1];
      [enc setBytes:origin length:sizeof(origin) atIndex:2];
      [enc setTexture:snap_dir atIndex:0];
      [enc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline("resolve_direct_depth_raw_1x")];
      [enc setTexture:depth_tex atIndex:0];
      [enc setTexture:stencil_view atIndex:1];
      [enc setBuffer:buf_dir offset:0 atIndex:1];
      [enc setBytes:params length:sizeof(params) atIndex:2];
      [enc dispatchThreads:MTLSizeMake(gw, gh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    {
      auto enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:msl.Pipeline(
                                       "resolve_direct_depth_fused_1x")];
      [enc setTexture:depth_tex atIndex:1];
      [enc setTexture:stencil_view atIndex:3];
      [enc setBytes:params length:sizeof(params) atIndex:1];
      [enc setBytes:origin length:sizeof(origin) atIndex:2];
      [enc setBuffer:buf_fused offset:0 atIndex:4];
      [enc setTexture:snap_fused atIndex:0];
      [enc dispatchThreads:MTLSizeMake(hw, hh, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
    }
    [cb commit];
    [cb waitUntilCompleted];
    REQUIRE(cb.error == nil);

    std::vector<uint32_t> s_ref(hw * hh), s_dir(hw * hh), s_fus(hw * hh);
    [snap_ref getBytes:s_ref.data()
           bytesPerRow:hw * 4
         bytesPerImage:hw * hh * 4
            fromRegion:MTLRegionMake2D(0, 0, hw, hh)
           mipmapLevel:0
                 slice:0];
    [snap_dir getBytes:s_dir.data()
           bytesPerRow:hw * 4
         bytesPerImage:hw * hh * 4
            fromRegion:MTLRegionMake2D(0, 0, hw, hh)
           mipmapLevel:0
                 slice:0];
    [snap_fused getBytes:s_fus.data()
             bytesPerRow:hw * 4
           bytesPerImage:hw * hh * 4
              fromRegion:MTLRegionMake2D(0, 0, hw, hh)
             mipmapLevel:0
                   slice:0];
    INFO("scale=" << scale);
    CHECK(s_dir == s_ref);
    CHECK(s_fus == s_ref);
    CHECK(std::memcmp(buf_dir.contents, buf_ref.contents, dest_extent) == 0);
    CHECK(std::memcmp(buf_fused.contents, buf_ref.contents, dest_extent) == 0);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
