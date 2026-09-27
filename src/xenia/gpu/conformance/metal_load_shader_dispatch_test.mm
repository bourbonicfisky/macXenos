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

// The production packing definition — the single ABI source of truth.
using LoadConstants = MetalLoadShaderPipelines::LoadConstants;
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

// ---- Shape/endian sweep: the load-shader infrastructure under odd,
// non-8-aligned dimensions, 256-byte-aligned destination pitches (the
// buffer-to-texture transfer layout), and every 32bpb endian mode — each
// byte-checked against the production CPU layout helpers with sentinel
// guards outside the shader's writable extent, twice (determinism). ----

static void RunShapeCase(uint32_t blocks_w, uint32_t blocks_h,
                         uint32_t pitch_aligned, xenos::Endian endian) {
  auto& h = H();
  REQUIRE(h.ok);
  int index = h.pipelines.FindIndex("texture_load_32bpb_cs");
  REQUIRE(index >= 0);

  // Destination layout: rows at a 256-aligned pitch; the shader writes
  // round8(blocks_w) blocks per row (full 8-block thread strides), of which
  // blocks_w*4 bytes are logical content; everything else must stay 0xC5.
  const uint32_t logical_row = blocks_w * 4;
  const uint32_t written_row = ((blocks_w + 7u) & ~7u) * 4;
  const uint32_t host_pitch = (written_row + 255u) & ~255u;
  const uint32_t dest_size = host_pitch * blocks_h + 256;

  // Source sized from the shader's exact padded read grid via the
  // production tiled addresser.
  const uint32_t padded_w = (blocks_w + 7u) & ~7u;
  int32_t max_off = 0;
  for (uint32_t y = 0; y < blocks_h; ++y) {
    for (uint32_t x = 0; x < padded_w; x += 4) {
      max_off = std::max(max_off, texture_address::Tiled2D(
                                      int32_t(x), int32_t(y), pitch_aligned, 2));
    }
  }
  const uint32_t source_size = uint32_t(max_off) + 16;
  std::vector<uint8_t> guest(source_size);
  for (uint32_t i = 0; i < source_size; ++i) {
    guest[i] = uint8_t(i * 11 + (i >> 9) * 7 + 3);
  }
  std::vector<uint8_t> cpu(size_t(blocks_w) * 4 * blocks_h);
  for (uint32_t y = 0; y < blocks_h; ++y) {
    for (uint32_t x = 0; x < blocks_w; ++x) {
      uint32_t src = uint32_t(texture_address::Tiled2D(
          int32_t(x), int32_t(y), pitch_aligned, 2));
      uint32_t word;
      std::memcpy(&word, &guest[src], 4);
      word = xenos::GpuSwap(word, endian);
      std::memcpy(&cpu[(size_t(y) * blocks_w + x) * 4], &word, 4);
    }
  }

  id<MTLBuffer> source =
      [h.device newBufferWithLength:source_size
                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> dest =
      [h.device newBufferWithLength:dest_size
                            options:MTLResourceStorageModeShared];
  std::memcpy(source.contents, guest.data(), source_size);

  LoadConstants constants = {};
  constants.is_tiled_3d_endian_scale = 1u | (uint32_t(endian) << 2);
  constants.guest_pitch_aligned = pitch_aligned;
  constants.size_blocks[0] = blocks_w;
  constants.size_blocks[1] = blocks_h;
  constants.size_blocks[2] = 1;
  constants.host_pitch = host_pitch;
  constants.height_texels = blocks_h;

  for (int pass = 0; pass < 2; ++pass) {
    std::memset(dest.contents, 0xC5, dest_size);
    REQUIRE(h.Dispatch(h.pipelines.pipeline(size_t(index)), constants, dest,
                       source, (blocks_w + 7) / 8, blocks_h, 1));
    const uint8_t* got = static_cast<const uint8_t*>(dest.contents);
    INFO("shape " << blocks_w << "x" << blocks_h << " pitch " << pitch_aligned
                  << " endian " << uint32_t(endian) << " pass " << pass);
    for (uint32_t y = 0; y < blocks_h; ++y) {
      const uint8_t* row = got + size_t(y) * host_pitch;
      if (std::memcmp(row, &cpu[size_t(y) * logical_row], logical_row) != 0) {
        FAIL("logical mismatch in row " << y);
      }
      for (uint32_t i = written_row; i < host_pitch; ++i) {
        if (row[i] != 0xC5) {
          FAIL("guard byte touched at row " << y << " offset " << i);
        }
      }
    }
    const uint8_t* tail = got + size_t(blocks_h) * host_pitch;
    for (uint32_t i = 0; i < 256; ++i) {
      if (tail[i] != 0xC5) {
        FAIL("tail guard touched at " << i);
      }
    }
  }
  SUCCEED();
}

// ---- Stage-1 widening (optimize-metal-gpu-resident-resolve-coherence):
// the bit-copy families the shared route now serves. Oracle = the guest
// dword(s) containing each block, GpuSwap'ed with the fetch endian, then the
// block's bytes taken at their offset inside the dword -- the hardware /
// upstream-shader semantics (endian swaps act on 16/32-bit units of guest
// memory, so an 8bpb or 16bpb block's neighbours participate). ----

static void RunFamilyShapeCase(const char* family, uint32_t bpb,
                               uint32_t block_log2,
                               uint32_t blocks_per_thread, uint32_t blocks_w,
                               uint32_t blocks_h, uint32_t pitch_aligned,
                               xenos::Endian endian) {
  auto& h = H();
  REQUIRE(h.ok);
  int index = h.pipelines.FindIndex(family);
  REQUIRE(index >= 0);

  const uint32_t logical_row = blocks_w * bpb;
  const uint32_t padded_w =
      ((blocks_w + blocks_per_thread - 1u) / blocks_per_thread) *
      blocks_per_thread;
  const uint32_t written_row = padded_w * bpb;
  const uint32_t host_pitch = (written_row + 255u) & ~255u;
  const uint32_t dest_size = host_pitch * blocks_h + 256;

  int32_t max_off = 0;
  for (uint32_t y = 0; y < blocks_h; ++y) {
    for (uint32_t x = 0; x < padded_w; ++x) {
      max_off = std::max(max_off, texture_address::Tiled2D(
                                      int32_t(x), int32_t(y), pitch_aligned,
                                      block_log2));
    }
  }
  const uint32_t source_size = ((uint32_t(max_off) + bpb + 31u) & ~15u) + 32;
  std::vector<uint8_t> guest(source_size);
  for (uint32_t i = 0; i < source_size; ++i) {
    guest[i] = uint8_t(i * 13 + (i >> 7) * 5 + 1);
  }
  std::vector<uint8_t> cpu(size_t(blocks_w) * bpb * blocks_h);
  for (uint32_t y = 0; y < blocks_h; ++y) {
    for (uint32_t x = 0; x < blocks_w; ++x) {
      uint32_t src = uint32_t(texture_address::Tiled2D(
          int32_t(x), int32_t(y), pitch_aligned, block_log2));
      uint8_t* dst = &cpu[(size_t(y) * blocks_w + x) * bpb];
      // Per-block swap, exactly the legacy cache's SwapEndian and the
      // shaders' XeEndianSwap16/32/64: a swap unit wider than the block is a
      // no-op (8bpb never swaps; 16bpb only honours 8in16).
      std::memcpy(dst, &guest[src], bpb);
      switch (endian) {
        case xenos::Endian::k8in16:
          for (uint32_t i = 0; i + 1 < bpb; i += 2) {
            std::swap(dst[i], dst[i + 1]);
          }
          break;
        case xenos::Endian::k8in32:
          for (uint32_t i = 0; i + 3 < bpb; i += 4) {
            std::swap(dst[i], dst[i + 3]);
            std::swap(dst[i + 1], dst[i + 2]);
          }
          break;
        case xenos::Endian::k16in32:
          for (uint32_t i = 0; i + 3 < bpb; i += 4) {
            std::swap(dst[i], dst[i + 2]);
            std::swap(dst[i + 1], dst[i + 3]);
          }
          break;
        default:
          break;
      }
    }
  }

  id<MTLBuffer> source =
      [h.device newBufferWithLength:source_size
                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> dest =
      [h.device newBufferWithLength:dest_size
                            options:MTLResourceStorageModeShared];
  std::memcpy(source.contents, guest.data(), source_size);

  LoadConstants constants = {};
  constants.is_tiled_3d_endian_scale = 1u | (uint32_t(endian) << 2);
  constants.guest_pitch_aligned = pitch_aligned;
  constants.size_blocks[0] = blocks_w;
  constants.size_blocks[1] = blocks_h;
  constants.size_blocks[2] = 1;
  constants.host_pitch = host_pitch;
  constants.height_texels = blocks_h;

  for (int pass = 0; pass < 2; ++pass) {
    std::memset(dest.contents, 0xC5, dest_size);
    REQUIRE(h.Dispatch(h.pipelines.pipeline(size_t(index)), constants, dest,
                       source, padded_w / blocks_per_thread, blocks_h, 1));
    const uint8_t* got = static_cast<const uint8_t*>(dest.contents);
    INFO(family << " shape " << blocks_w << "x" << blocks_h << " pitch "
                << pitch_aligned << " endian " << uint32_t(endian) << " pass "
                << pass);
    for (uint32_t y = 0; y < blocks_h; ++y) {
      const uint8_t* row = got + size_t(y) * host_pitch;
      if (std::memcmp(row, &cpu[size_t(y) * logical_row], logical_row) != 0) {
        FAIL("logical mismatch in row " << y);
      }
      for (uint32_t i = written_row; i < host_pitch; ++i) {
        if (row[i] != 0xC5) {
          FAIL("guard byte touched at row " << y << " offset " << i);
        }
      }
    }
    const uint8_t* tail = got + size_t(blocks_h) * host_pitch;
    for (uint32_t i = 0; i < 256; ++i) {
      if (tail[i] != 0xC5) {
        FAIL("tail guard touched at " << i);
      }
    }
  }
  SUCCEED();
}

TEST_CASE("load shaders: 16bpb family matches the legacy per-block untile",
          "[l1][metal][loadshader][family]") {
  REQUIRE_DEVICE();
  for (xenos::Endian e : {xenos::Endian::kNone, xenos::Endian::k8in16,
                          xenos::Endian::k8in32, xenos::Endian::k16in32}) {
    RunFamilyShapeCase("texture_load_16bpb_cs", 2, 1, 16, 64, 16, 64, e);
    RunFamilyShapeCase("texture_load_16bpb_cs", 2, 1, 16, 41, 17, 64, e);
    RunFamilyShapeCase("texture_load_16bpb_cs", 2, 1, 16, 100, 7, 128, e);
  }
}

TEST_CASE("load shaders: 64bpb family matches the legacy per-block untile",
          "[l1][metal][loadshader][family]") {
  REQUIRE_DEVICE();
  for (xenos::Endian e : {xenos::Endian::kNone, xenos::Endian::k8in16,
                          xenos::Endian::k8in32, xenos::Endian::k16in32}) {
    RunFamilyShapeCase("texture_load_64bpb_cs", 8, 3, 4, 32, 16, 32, e);
    RunFamilyShapeCase("texture_load_64bpb_cs", 8, 3, 4, 41, 17, 64, e);
    RunFamilyShapeCase("texture_load_64bpb_cs", 8, 3, 4, 100, 7, 128, e);
  }
}

TEST_CASE("load shaders: 8bpb family matches the legacy per-block untile",
          "[l1][metal][loadshader][family]") {
  REQUIRE_DEVICE();
  for (xenos::Endian e : {xenos::Endian::kNone, xenos::Endian::k8in16,
                          xenos::Endian::k8in32, xenos::Endian::k16in32}) {
    RunFamilyShapeCase("texture_load_8bpb_cs", 1, 0, 16, 64, 16, 64, e);
    RunFamilyShapeCase("texture_load_8bpb_cs", 1, 0, 16, 41, 17, 64, e);
    RunFamilyShapeCase("texture_load_8bpb_cs", 1, 0, 16, 100, 7, 128, e);
  }
}

TEST_CASE("load shaders: odd non-aligned dimensions and pitches",
          "[l1][metal][loadshader]") {
  REQUIRE_DEVICE();
  RunShapeCase(41, 17, 64, xenos::Endian::k8in32);
  RunShapeCase(33, 47, 64, xenos::Endian::k8in32);
  RunShapeCase(100, 7, 128, xenos::Endian::k8in32);
}

TEST_CASE("load shaders: every 32bpb endian mode matches the CPU helper",
          "[l1][metal][loadshader]") {
  REQUIRE_DEVICE();
  RunShapeCase(24, 8, 32, xenos::Endian::kNone);
  RunShapeCase(24, 8, 32, xenos::Endian::k8in16);
  RunShapeCase(24, 8, 32, xenos::Endian::k8in32);
  RunShapeCase(24, 8, 32, xenos::Endian::k16in32);
}
}  // namespace
}  // namespace gpu
}  // namespace xe
