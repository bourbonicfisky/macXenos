/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 (on-device): the Metal texture-load shader INFRASTRUCTURE — generated
// metallib bytecode -> MTLLibrary -> entry_xe -> MTLComputePipelineState ->
// isolated deterministic dispatch. No emulator texture cache, shared memory
// or gameplay path is engaged; resources are plain MTLBuffers.
//
// The dispatch oracle is the PRODUCTION CPU texture-layout math — the same
// helpers the bring-up CPU untile path uses (texture_address::Tiled2D for
// guest addressing, xenos::GpuSwap for endian) — not a test-only replica of
// the shader.

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

// Host mirror of the MSL push_const_struct_xe for the texture-load family
// (texture_load.xesli). MSL uint3 is 16-byte aligned AND 16-byte sized, so
// xe_texture_load_host_offset lands at offset 32, not 28.
struct LoadConstants {
  uint32_t is_tiled_3d_endian_scale;               // 0
  uint32_t guest_offset;                           // 4
  uint32_t guest_pitch_aligned;                    // 8
  uint32_t guest_z_stride_block_rows_aligned;      // 12
  uint32_t size_blocks[3];                         // 16
  uint32_t pad_uint3_w;                            // 28 (uint3 padding)
  uint32_t host_offset;                            // 32
  uint32_t host_pitch;                             // 36
  uint32_t height_texels;                          // 40
  uint32_t pad_tail;                               // 44 (struct align 16)
};
static_assert(sizeof(LoadConstants) == 48, "MSL push-const ABI");
static_assert(offsetof(LoadConstants, host_offset) == 32, "MSL uint3 size");

struct DispatchHarness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  MetalLoadShaderPipelines pipelines;
  bool ok = false;

  DispatchHarness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return;
    }
    queue = [device newCommandQueue];
    if (!queue) {
      return;
    }
    ok = pipelines.Initialize((__bridge void*)device);
  }

  // Encodes one texture-load dispatch and waits for completion. Returns
  // false if the command buffer reports an error (Metal fault).
  bool Dispatch(void* pipeline_handle, const LoadConstants& constants,
                id<MTLBuffer> dest, id<MTLBuffer> source,
                uint32_t threads_x, uint32_t threads_y, uint32_t threads_z) {
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_handle;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBytes:&constants
           length:sizeof(constants)
          atIndex:MetalLoadShaderPipelines::kBufferIndexPushConstants];
    [enc setBuffer:dest
            offset:0
           atIndex:MetalLoadShaderPipelines::kBufferIndexDest];
    [enc setBuffer:source
            offset:0
           atIndex:MetalLoadShaderPipelines::kBufferIndexSource];
    MTLSize group = MTLSizeMake(MetalLoadShaderPipelines::kThreadgroupSizeX,
                                MetalLoadShaderPipelines::kThreadgroupSizeY,
                                MetalLoadShaderPipelines::kThreadgroupSizeZ);
    MTLSize groups = MTLSizeMake(
        (threads_x + group.width - 1) / group.width,
        (threads_y + group.height - 1) / group.height,
        (threads_z + group.depth - 1) / group.depth);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted && cmd.error == nil;
  }
};

DispatchHarness& H() {
  static DispatchHarness h;
  return h;
}

#define REQUIRE_DEVICE()                                        \
  do {                                                          \
    if (!H().device) {                                          \
      FAIL("No Metal device: these cases must run on the M4."); \
    }                                                           \
  } while (0)

TEST_CASE("load shaders: full family builds libraries and compute pipelines "
          "on device",
          "[l1][metal][loadshader]") {
  REQUIRE_DEVICE();
  auto& h = H();
  REQUIRE(h.ok);
  size_t declared_count;
  MetalLoadShaderPipelines::shaders(&declared_count);
  REQUIRE(declared_count == 59);  // Complete texture_load_*.cs.xesl family.
  REQUIRE(h.pipelines.count() == declared_count);
  for (size_t i = 0; i < declared_count; ++i) {
    INFO("shader index " << i);
    REQUIRE(h.pipelines.pipeline(i) != nullptr);
  }
  REQUIRE(h.pipelines.FindIndex("texture_load_32bpb_cs") >= 0);
  REQUIRE(h.pipelines.FindIndex("no_such_shader") == -1);
}

TEST_CASE("load shaders: zero-work dispatch completes without fault and "
          "writes nothing",
          "[l1][metal][loadshader]") {
  REQUIRE_DEVICE();
  auto& h = H();
  REQUIRE(h.ok);
  int index = h.pipelines.FindIndex("texture_load_32bpb_cs");
  REQUIRE(index >= 0);

  constexpr uint32_t kDestSize = 4096;
  id<MTLBuffer> dest = [h.device newBufferWithLength:kDestSize
                                             options:MTLResourceStorageModeShared];
  id<MTLBuffer> source = [h.device newBufferWithLength:4096
                                               options:MTLResourceStorageModeShared];
  std::memset(dest.contents, 0xCD, kDestSize);

  LoadConstants constants = {};
  // size_blocks == 0: every thread exits on the bounds guard.
  REQUIRE(h.Dispatch(h.pipelines.pipeline(size_t(index)), constants, dest,
                     source, 4, 32, 1));
  const uint8_t* bytes = static_cast<const uint8_t*>(dest.contents);
  for (uint32_t i = 0; i < kDestSize; ++i) {
    if (bytes[i] != 0xCD) {
      FAIL("dest byte " << i << " modified by zero-work dispatch");
    }
  }
  SUCCEED();
}

TEST_CASE("load shaders: deterministic tiled 32bpb dispatch matches the "
          "production CPU layout helpers exactly",
          "[l1][metal][loadshader]") {
  REQUIRE_DEVICE();
  auto& h = H();
  REQUIRE(h.ok);
  int index = h.pipelines.FindIndex("texture_load_32bpb_cs");
  REQUIRE(index >= 0);

  // Guest layout: tiled 2D, 48x33 blocks of 4 bytes, pitch aligned to 64
  // blocks (32-block tiling alignment), endian 8-in-32. 48 is a multiple of
  // the shader's 8-blocks-per-thread stride so the written extent is exact;
  // 33 rows exercises the second macro-tile row and the Y bounds guard.
  constexpr uint32_t kWidthBlocks = 48;
  constexpr uint32_t kHeightBlocks = 33;
  constexpr uint32_t kPitchAligned = 64;
  constexpr uint32_t kBpb = 4;
  constexpr uint32_t kBpbLog2 = 2;
  const xenos::Endian kEndian = xenos::Endian::k8in32;

  // Guest source sized by the production tiled-address helper itself: the
  // maximum Tiled2D offset over the grid (plus one uint4 read) bounds every
  // byte the shader may touch.
  int32_t max_offset = 0;
  for (uint32_t y = 0; y < kHeightBlocks; ++y) {
    for (uint32_t x = 0; x < kWidthBlocks; ++x) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y),
                                             kPitchAligned, kBpbLog2);
      REQUIRE(off >= 0);
      if (off > max_offset) {
        max_offset = off;
      }
    }
  }
  const uint32_t kGuestOffset = 256;  // Nonzero base offset in the buffer.
  uint32_t source_size = kGuestOffset + uint32_t(max_offset) + 16;
  std::vector<uint8_t> guest(source_size);
  for (uint32_t i = 0; i < source_size; ++i) {
    guest[i] = uint8_t(i * 7 + (i >> 8) * 13 + 1);
  }

  // Host destination: rows of kWidthBlocks*4 bytes at a larger pitch, below
  // a nonzero host offset — everything outside the written extent must keep
  // its sentinel.
  constexpr uint32_t kHostOffset = 128;
  constexpr uint32_t kHostPitch = 256;  // > 48*4 = 192: 64 guard bytes/row.
  const uint32_t dest_size = kHostOffset + kHeightBlocks * kHostPitch + 512;

  // Expected bytes via the PRODUCTION helpers: for each block,
  // host[host_offset + x*4 + y*host_pitch] =
  //     GpuSwap(source_u32[guest_offset + Tiled2D(x, y)], endian).
  std::vector<uint8_t> expected(dest_size, 0xCD);
  for (uint32_t y = 0; y < kHeightBlocks; ++y) {
    for (uint32_t x = 0; x < kWidthBlocks; ++x) {
      uint32_t src_off = kGuestOffset +
                         uint32_t(texture_address::Tiled2D(
                             int32_t(x), int32_t(y), kPitchAligned, kBpbLog2));
      uint32_t word;
      std::memcpy(&word, &guest[src_off], sizeof(word));
      word = xenos::GpuSwap(word, kEndian);
      uint32_t dst_off = kHostOffset + x * kBpb + y * kHostPitch;
      std::memcpy(&expected[dst_off], &word, sizeof(word));
    }
  }

  id<MTLBuffer> source = [h.device newBufferWithLength:source_size
                                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> dest = [h.device newBufferWithLength:dest_size
                                             options:MTLResourceStorageModeShared];
  std::memcpy(source.contents, guest.data(), source_size);
  std::memset(dest.contents, 0xCD, dest_size);

  LoadConstants constants = {};
  constants.is_tiled_3d_endian_scale =
      1u | (uint32_t(kEndian) << 2);  // tiled, 2D, endian bits [3:2].
  constants.guest_offset = kGuestOffset;
  constants.guest_pitch_aligned = kPitchAligned;
  constants.guest_z_stride_block_rows_aligned = 0;
  constants.size_blocks[0] = kWidthBlocks;
  constants.size_blocks[1] = kHeightBlocks;
  constants.size_blocks[2] = 1;
  constants.host_offset = kHostOffset;
  constants.host_pitch = kHostPitch;
  constants.height_texels = kHeightBlocks;

  // One thread per 8 blocks along X.
  REQUIRE(h.Dispatch(h.pipelines.pipeline(size_t(index)), constants, dest,
                     source, kWidthBlocks / 8, kHeightBlocks, 1));

  const uint8_t* got = static_cast<const uint8_t*>(dest.contents);
  // Exact byte comparison over the whole destination: written extent must
  // match the CPU helpers, and every guard byte (before host_offset, the
  // per-row tail, and past the last row) must still be the sentinel.
  size_t first_mismatch = size_t(-1);
  for (size_t i = 0; i < dest_size; ++i) {
    if (got[i] != expected[i]) {
      first_mismatch = i;
      break;
    }
  }
  if (first_mismatch != size_t(-1)) {
    FAIL("dest mismatch at byte "
         << first_mismatch << ": got 0x" << std::hex
         << unsigned(got[first_mismatch]) << " expected 0x"
         << unsigned(expected[first_mismatch]));
  }
  SUCCEED();

  // Determinism: a second identical dispatch over a re-sentineled buffer
  // must produce identical bytes.
  std::memset(dest.contents, 0xCD, dest_size);
  REQUIRE(h.Dispatch(h.pipelines.pipeline(size_t(index)), constants, dest,
                     source, kWidthBlocks / 8, kHeightBlocks, 1));
  REQUIRE(std::memcmp(dest.contents, expected.data(), dest_size) == 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
