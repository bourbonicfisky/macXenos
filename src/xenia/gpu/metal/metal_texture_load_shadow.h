/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_TEXTURE_LOAD_SHADOW_H_
#define XENIA_GPU_METAL_METAL_TEXTURE_LOAD_SHADOW_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/xenos.h"

// Pure (device-free) logic for the strict-RGBA8 GPU texture-load SHADOW
// experiment (OpenSpec implement-metal-gpu-texture-loads, slice 3): the
// shared eligibility predicate (also the eventual routed-path predicate),
// the EXACT guest source range the 32bpb load shader reads, the aligned
// shadow destination layout, and the deferred byte comparison. Everything
// here is testable without a Metal device; the .mm integration only
// orchestrates.

namespace xe {
namespace gpu {
namespace metal {

// ---- Strict first class: k_8_8_8_8, unsigned, tiled, 2D, base mip only ----

// One shared predicate for the shadow path and the eventual routed path.
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
struct ShadowSourceRange {
  uint32_t begin = 0;  // Absolute guest byte address, inclusive.
  uint32_t end = 0;    // Absolute guest byte address, exclusive.
  bool valid = false;
};

inline ShadowSourceRange ComputeStrictRgba8SourceRange(
    uint32_t base, uint32_t blocks_w, uint32_t blocks_h,
    uint32_t pitch_blocks_aligned, uint32_t physical_size) {
  ShadowSourceRange r;
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

// ---- Shadow destination layout ----

// Rows are 256-byte aligned (buffer-to-texture transfer alignment, matching
// the future routed path); the shader writes round_up(blocks_w, 8) blocks
// per row (full 8-block thread strides), of which only blocks_w * 4 bytes
// are logical texture content. Beyond written_row_bytes and past the last
// row, bytes are guard territory the dispatch must never touch.
struct ShadowDestLayout {
  uint32_t logical_row_bytes = 0;  // blocks_w * 4 — compared.
  uint32_t written_row_bytes = 0;  // round8(blocks_w) * 4 — shader-written.
  uint32_t host_pitch = 0;         // align256(written_row_bytes).
  uint32_t rows = 0;               // blocks_h (exact y guard).
  uint32_t tail_guard_bytes = 0;
  uint32_t total_size = 0;
};

inline ShadowDestLayout ComputeShadowDestLayout(uint32_t blocks_w,
                                                uint32_t blocks_h) {
  ShadowDestLayout l;
  l.logical_row_bytes = blocks_w * 4;
  l.written_row_bytes = ((blocks_w + 7u) & ~7u) * 4;
  l.host_pitch = (l.written_row_bytes + 255u) & ~255u;
  l.rows = blocks_h;
  l.tail_guard_bytes = 256;
  l.total_size = l.host_pitch * l.rows + l.tail_guard_bytes;
  return l;
}

constexpr uint8_t kShadowDestSentinel = 0xC5;

// ---- Deferred comparison ----

struct ShadowCompareResult {
  uint64_t bytes_compared = 0;
  bool match = true;
  // First logical mismatch (valid when !match && !guard_violation).
  uint32_t first_mismatch_offset = 0;  // Offset into the dest buffer.
  uint8_t expected_value = 0;
  uint8_t actual_value = 0;
  // A write outside the shader's writable extent — infrastructure bug, not
  // a content mismatch.
  bool guard_violation = false;
  uint32_t guard_offset = 0;
};

// Compares the GPU dest bytes against the CPU untile's linear output
// (blocks_w*4-byte rows, exactly what the authoritative path uploads).
// Logical bytes must match; written-padding lanes (guest padding content,
// [logical_row_bytes, written_row_bytes)) are neither compared nor guarded;
// everything else must still hold the sentinel.
inline ShadowCompareResult CompareShadowDest(const uint8_t* dest,
                                             const uint8_t* cpu_linear,
                                             const ShadowDestLayout& l) {
  ShadowCompareResult r;
  for (uint32_t y = 0; y < l.rows; ++y) {
    const uint8_t* row = dest + size_t(y) * l.host_pitch;
    const uint8_t* expect = cpu_linear + size_t(y) * l.logical_row_bytes;
    for (uint32_t i = 0; i < l.logical_row_bytes; ++i) {
      if (row[i] != expect[i]) {
        if (r.match) {
          r.match = false;
          r.first_mismatch_offset = uint32_t(size_t(y) * l.host_pitch + i);
          r.expected_value = expect[i];
          r.actual_value = row[i];
        }
        break;  // One report per row is enough; totals stay aggregate.
      }
    }
    r.bytes_compared += l.logical_row_bytes;
    for (uint32_t i = l.written_row_bytes; i < l.host_pitch; ++i) {
      if (row[i] != kShadowDestSentinel) {
        r.guard_violation = true;
        r.guard_offset = uint32_t(size_t(y) * l.host_pitch + i);
        r.match = false;
        return r;
      }
    }
  }
  const uint8_t* tail = dest + size_t(l.rows) * l.host_pitch;
  for (uint32_t i = 0; i < l.tail_guard_bytes; ++i) {
    if (tail[i] != kShadowDestSentinel) {
      r.guard_violation = true;
      r.guard_offset = uint32_t(size_t(l.rows) * l.host_pitch + i);
      r.match = false;
      return r;
    }
  }
  return r;
}

// ---- Staged shadow request (texture cache -> command processor) ----

struct TextureLoadShadowStage {
  uint32_t base = 0;
  uint32_t blocks_w = 0;
  uint32_t blocks_h = 0;
  uint32_t pitch_blocks_aligned = 0;
  uint32_t endian = 0;  // xenos::Endian as uint (shader endian_32 bits).
  ShadowSourceRange source_range;
  // XXH3 of the raw guest bytes in source_range at snapshot time; re-checked
  // against the shared-memory buffer content at arm time so a source that
  // mutated between the CPU untile and the deferred validation is dropped
  // (counted, never dispatched) instead of producing a false mismatch.
  uint64_t source_hash = 0;
  // The authoritative CPU untile output (blocks_w*4-byte rows), copied AFTER
  // the untile fully completed and BEFORE any shared-memory call.
  std::vector<uint8_t> cpu_linear;
};

// Aggregate diagnostics (no per-texture spam).
struct TextureLoadShadowStats {
  uint64_t eligible_uploads = 0;
  uint64_t staged = 0;
  uint64_t dropped_over_budget = 0;
  uint64_t dropped_source_mutated = 0;
  uint64_t dropped_request_failed = 0;
  uint64_t dropped_submission_fault = 0;
  uint64_t dropped_no_command_buffer = 0;
  uint64_t dispatches_completed = 0;
  uint64_t bytes_compared = 0;
  uint64_t exact_matches = 0;
  uint64_t mismatches = 0;
  uint64_t guard_violations = 0;
  // First mismatch ever seen (sticky).
  bool have_first_mismatch = false;
  uint32_t first_mismatch_base = 0;
  uint32_t first_mismatch_offset = 0;
  uint8_t first_mismatch_expected = 0;
  uint8_t first_mismatch_actual = 0;
  // Pending shadow resources (staged + armed + in-flight).
  uint64_t pending_bytes = 0;
  uint64_t pending_entries = 0;
  uint64_t high_water_bytes = 0;
  uint64_t high_water_entries = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_LOAD_SHADOW_H_
