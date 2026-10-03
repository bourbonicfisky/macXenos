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

}  // namespace test
}  // namespace gpu
}  // namespace xe
