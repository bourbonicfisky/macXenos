/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_TEXTURE_LOAD_ELIGIBILITY_H_
#define XENIA_GPU_METAL_METAL_TEXTURE_LOAD_ELIGIBILITY_H_

#include <algorithm>
#include <cstdint>

#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/xenos.h"

// Pure (device-free) strict-RGBA8 GPU texture-load contract, proven byte-exact
// against the CPU untile across the recorded corpus (OpenSpec
// implement-metal-gpu-texture-loads, slices 3/4/5R): the eligibility
// predicate, the EXACT guest source range the 32bpb load shader reads, and
// the 256-aligned buffer-to-texture destination layout. The 5R landing
// decision replays the GPU load through the shared TextureCache seam
// (LoadTextureDataFromResidentMemoryImpl); this header carries the proven
// contract to that seam. Guarded by
// conformance/metal_texture_load_eligibility_test.cc — weakening a condition
// there fails the suite.

namespace xe {
namespace gpu {
namespace metal {

// ---- Strict first class: k_8_8_8_8, unsigned/gamma, tiled, 2D, base mip ----

// One shared predicate for every strict-RGBA8 GPU-load consumer.
// Every condition is deliberate; do NOT weaken to increase engagement.
// width/height/pitch_texels are the derived values the texture cache uses
// (pitch defaulted to align32(width) when the fetch pitch is zero).
inline bool IsStrictRgba8Eligible(const xenos::xe_gpu_texture_fetch_t& fetch,
                                  bool is_signed, uint32_t width,
                                  uint32_t height, uint32_t pitch_texels) {
  // k_8_8_8_8 exactly — no aliases (k_8_8_8_8_A/AS_8_8_8_8 etc. differ).
  if (fetch.format != xenos::TextureFormat::k_8_8_8_8) {
    return false;
  }
  // Unsigned binding only (signed fetches are a separate SNORM-view
  // binding). Per-component signs may be kUnsigned or kGamma —
  // both sample the same unorm BYTES (gamma is a sampling-time transfer
  // curve in the translated shader, not an upload transform), and the
  // shipped exact-cover snapshot reader serves this class without
  // consulting component signs at all. kSigned/kUnsignedBiased components
  // change value interpretation classes and stay excluded.
  // (5R amendment, user-approved 2026-07-18: the entire live JC2
  // resolve-backed pool is gamma-signed; unsigned-only excluded it.)
  if (is_signed) {
    return false;
  }
  auto sign_ok = [](xenos::TextureSign s) {
    return s == xenos::TextureSign::kUnsigned ||
           s == xenos::TextureSign::kGamma;
  };
  if (!sign_ok(fetch.sign_x) || !sign_ok(fetch.sign_y) ||
      !sign_ok(fetch.sign_z) || !sign_ok(fetch.sign_w)) {
    return false;
  }
  if (!fetch.tiled) {
    return false;
  }
  if (fetch.dimension != xenos::DataDimension::k2DOrStacked) {
    return false;
  }
  if (fetch.stacked) {
    return false;
  }
  if (fetch.mip_min_level != 0 || fetch.mip_max_level != 0) {
    return false;
  }
  if (!fetch.base_address || !width || !height || !pitch_texels) {
    return false;
  }
  return true;
}

// ---- Exact source range ----

// The 32bpb load shader reads 16-byte groups at Tiled2D(4g, y) byte offsets
// (each 4 consecutive x blocks are byte-contiguous; the second uint4 of a
// thread is the g+1 group via the local-XOR step), for x groups covering the
// grid padded to the shader's 8-blocks-per-thread stride, and every y row
// (the y bounds guard is exact). The exact read set is therefore
//   { [base + Tiled2D(4g, y), +16) : 4g < round_up(blocks_w, 8),
//                                    y < blocks_h }
// computed with the SAME shared tiled-address helper the CPU untile uses —
// never a pitch*height*bpb approximation.
struct TextureLoadSourceRange {
  uint32_t begin = 0;  // Absolute guest byte address, inclusive.
  uint32_t end = 0;    // Absolute guest byte address, exclusive.
  bool valid = false;
};

inline TextureLoadSourceRange ComputeStrictRgba8SourceRange(
    uint32_t base, uint32_t blocks_w, uint32_t blocks_h,
    uint32_t pitch_blocks_aligned, uint32_t physical_size) {
  TextureLoadSourceRange r;
  if (!blocks_w || !blocks_h || !pitch_blocks_aligned) {
    return r;
  }
  uint32_t padded_w = (blocks_w + 7u) & ~7u;
  int32_t min_off = INT32_MAX;
  int32_t max_off = INT32_MIN;
  for (uint32_t y = 0; y < blocks_h; ++y) {
    for (uint32_t x = 0; x < padded_w; x += 4) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y),
                                             pitch_blocks_aligned,
                                             /*bytes_per_block_log2=*/2);
      if (off < 0) {
        return r;
      }
      min_off = std::min(min_off, off);
      max_off = std::max(max_off, off);
    }
  }
  uint64_t begin = uint64_t(base) + uint64_t(min_off);
  uint64_t end = uint64_t(base) + uint64_t(max_off) + 16;
  if (end > physical_size || begin >= end) {
    return r;
  }
  r.begin = uint32_t(begin);
  r.end = uint32_t(end);
  r.valid = true;
  return r;
}

// ---- Destination layout ----

// Rows are 256-byte aligned (buffer-to-texture transfer alignment); the
// shader writes round_up(blocks_w, 8) blocks per row (full 8-block thread
// strides), of which only blocks_w * 4 bytes are logical texture content.
// Beyond written_row_bytes and past the last row, bytes are guard territory
// the dispatch must never touch.
struct TextureLoadDestLayout {
  uint32_t logical_row_bytes = 0;  // blocks_w * 4 — texture content.
  uint32_t written_row_bytes = 0;  // round8(blocks_w) * 4 — shader-written.
  uint32_t host_pitch = 0;         // align256(written_row_bytes).
  uint32_t rows = 0;               // blocks_h (exact y guard).
  uint32_t tail_guard_bytes = 0;
  uint32_t total_size = 0;
};

inline TextureLoadDestLayout ComputeTextureLoadDestLayout(uint32_t blocks_w,
                                                          uint32_t blocks_h) {
  TextureLoadDestLayout l;
  l.logical_row_bytes = blocks_w * 4;
  l.written_row_bytes = ((blocks_w + 7u) & ~7u) * 4;
  l.host_pitch = (l.written_row_bytes + 255u) & ~255u;
  l.rows = blocks_h;
  l.tail_guard_bytes = 256;
  l.total_size = l.host_pitch * l.rows + l.tail_guard_bytes;
  return l;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_LOAD_ELIGIBILITY_H_
