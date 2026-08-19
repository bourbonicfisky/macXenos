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

// C1T-R route eligibility: the shared-subclass route additionally requires a
// nonzero fetch pitch — the shared TextureKey carries the RAW fetch pitch,
// so the legacy cache's align32(width) default for pitch==0 fetches is not
// representable there; such fetches stay on the legacy CPU path. This is a
// NARROWING for one consumer, never a weakening of the contract above.
inline bool IsStrictRgba8RouteEligible(
    const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed, uint32_t width,
    uint32_t height, uint32_t pitch_texels) {
  return IsStrictRgba8Eligible(fetch, is_signed, width, height, pitch_texels) &&
         fetch.pitch != 0;
}

// ---- SIGNED-binding sibling (reopen-metal-gpu-visible-resolve-authority) ----
//
// A SIGNED k_8_8_8_8 binding samples the SAME guest bytes through an
// RGBA8Snorm VIEW (Xenos 8-bit signed texels are two's-complement — exactly
// host snorm bytes; the legacy plan's host_format_signed encodes the same
// equivalence). Deliberately a separate predicate so the strict class above
// is not weakened: every structural condition is identical, the differences
// are that the binding is signed and per-component signs may additionally be
// kSigned. kUnsignedBiased stays excluded everywhere — it is a value
// transform, not a reinterpreting view.
inline bool IsStrictRgba8SignedRouteEligible(
    const xenos::xe_gpu_texture_fetch_t& fetch, uint32_t width,
    uint32_t height, uint32_t pitch_texels) {
  if (fetch.format != xenos::TextureFormat::k_8_8_8_8) {
    return false;
  }
  auto sign_ok = [](xenos::TextureSign s) {
    return s != xenos::TextureSign::kUnsignedBiased;
  };
  if (!sign_ok(fetch.sign_x) || !sign_ok(fetch.sign_y) ||
      !sign_ok(fetch.sign_z) || !sign_ok(fetch.sign_w)) {
    return false;
  }
  if (!fetch.tiled || fetch.dimension != xenos::DataDimension::k2DOrStacked ||
      fetch.stacked) {
    return false;
  }
  // Single-mip shape ONLY (exactly the strict class's mip condition). A
  // mipped signed fetch served through a view of a single-level entry would
  // clamp every minified sample to level 0 — measured as a 125k-px SF4
  // regression when this predicate briefly admitted mipped shapes.
  if (fetch.mip_min_level != 0 || fetch.mip_max_level != 0) {
    return false;
  }
  if (!fetch.base_address || !width || !height || !pitch_texels) {
    return false;
  }
  return fetch.pitch != 0;
}

// ---- Mipped sibling class (OpenSpec 6.1) ----

// The MIPPED sibling of the strict class. Deliberately a separate predicate:
// IsStrictRgba8Eligible's single-mip rejection above is pinned by conformance
// and is NOT weakened, because the canonical single-mip route keeps depending
// on it exactly as promoted.
//
// Every non-mip condition is identical to the strict predicate (evaluated by
// masking the mip levels off and delegating, so the two can never drift), plus:
//   - mip_min_level == 0 EXPLICITLY. A nonzero min level means the guest
//     samples a shifted level set, which this slice does not express.
//   - mip_max_level > 0, i.e. there is actually a chain to load.
//   - a nonzero mip_address: the chain lives at its own guest pointer, and the
//     per-level plan reads levels 1..max from there.
//   - a nonzero fetch pitch, for the same reason the single-mip route requires
//     it (the shared TextureKey carries the raw pitch).
inline bool IsStrictRgba8MipRouteEligible(
    const xenos::xe_gpu_texture_fetch_t& fetch, bool is_signed, uint32_t width,
    uint32_t height, uint32_t pitch_texels) {
  if (fetch.mip_min_level != 0 || fetch.mip_max_level == 0) {
    return false;
  }
  if (!fetch.mip_address || !fetch.pitch) {
    return false;
  }
  xenos::xe_gpu_texture_fetch_t as_base = fetch;
  as_base.mip_min_level = 0;
  as_base.mip_max_level = 0;
  return IsStrictRgba8Eligible(as_base, is_signed, width, height, pitch_texels);
}

// ---- Trace-record extent ----

// The guest byte extent the legacy cache records for a non-cube texture
// upload (watched_base_bytes in MetalTextureCache::GetTextureSlot) and the
// warm-entry trace sweep. Factored here as the documented single expression
// of that contract. (The C1T-R route needs no matching record: it disengages
// for captured frames, so the legacy path records the CPU-oracle read-set —
// the gate-C identical-ranges property holds by construction.)
inline uint32_t ComputeWatchedBaseBytes(uint32_t pitch_blocks_aligned,
                                        uint32_t blocks_h, uint32_t bpb,
                                        uint32_t base,
                                        uint32_t physical_size) {
  uint64_t bytes = uint64_t(pitch_blocks_aligned) * blocks_h * bpb;
  uint64_t max_bytes = physical_size > base ? physical_size - base : 0;
  return uint32_t(std::min<uint64_t>(bytes, max_bytes));
}

// ---- Conservative source envelope ----

// The 32bpb load shader reads 16-byte groups at Tiled2D(4g, y) byte offsets
// (each 4 consecutive x blocks are byte-contiguous; the second uint4 of a
// thread is the g+1 group via the local-XOR step), for x groups covering the
// grid padded to the shader's 8-blocks-per-thread stride, and every y row
// (the y bounds guard is exact). The exact read set is
//   { [base + Tiled2D(4g, y), +16) : 4g < round_up(blocks_w, 8),
//                                    y < blocks_h }
// but TextureLoadSourceRange stores only the minimum/maximum ENVELOPE around
// that set. Tiled addressing may leave holes inside it. The envelope is still
// computed with the SAME shared tiled-address helper the CPU untile uses —
// never a pitch*height*bpb approximation — and is conservatively safe for
// RequestConsumedRange and hazard planning.
struct TextureLoadSourceRange {
  uint32_t begin = 0;  // Absolute guest byte address, inclusive.
  uint32_t end = 0;    // Absolute guest byte address, exclusive.
  bool valid = false;
};

// Origin-aware envelope calculation. The 32bpb direct shader dispatches
// ceil(w/8) threads along x, each covering 8 blocks as two 4-block groups, and
// reads Tiled2D(px + 4k, py + y) -- so for a packed sub-mip the read set starts
// AT the sub-tile origin, not at (0,0). Scanning from the origin avoids claiming
// the bytes of every larger level sharing the tail, though the returned min/max
// pair remains a conservative envelope rather than the exact group union.
//
// All intermediates are 64-bit and every value that reaches Tiled2D is proven
// to fit in int32, so a hostile or corrupt fetch cannot wrap into a bogus
// in-range span.
inline TextureLoadSourceRange ComputeStrictRgba8SourceRangeAtOrigin(
    uint32_t base, uint32_t px, uint32_t py, uint32_t blocks_w,
    uint32_t blocks_h, uint32_t pitch_blocks_aligned, uint32_t physical_size) {
  TextureLoadSourceRange r;
  if (!blocks_w || !blocks_h || !pitch_blocks_aligned) {
    return r;
  }
  uint64_t padded_w = (uint64_t(blocks_w) + 7u) & ~uint64_t(7u);
  uint64_t x_end = uint64_t(px) + padded_w;
  uint64_t y_end = uint64_t(py) + uint64_t(blocks_h);
  // Everything handed to Tiled2D must be a non-negative int32.
  if (x_end > uint64_t(INT32_MAX) || y_end > uint64_t(INT32_MAX) ||
      pitch_blocks_aligned > uint32_t(INT32_MAX)) {
    return r;
  }
  int64_t min_off = INT64_MAX;
  int64_t max_off = INT64_MIN;
  for (uint64_t y = py; y < y_end; ++y) {
    for (uint64_t x = px; x < x_end; x += 4) {
      int32_t off = texture_address::Tiled2D(int32_t(x), int32_t(y),
                                             pitch_blocks_aligned,
                                             /*bytes_per_block_log2=*/2);
      if (off < 0) {
        return r;
      }
      min_off = std::min<int64_t>(min_off, off);
      max_off = std::max<int64_t>(max_off, off);
    }
  }
  if (min_off > max_off) {
    return r;
  }
  uint64_t begin = uint64_t(base) + uint64_t(min_off);
  uint64_t end = uint64_t(base) + uint64_t(max_off) + 16u;
  if (end > uint64_t(physical_size) || begin >= end) {
    return r;
  }
  r.begin = uint32_t(begin);
  r.end = uint32_t(end);
  r.valid = true;
  return r;
}

// The single-mip form: the origin-aware envelope at origin (0,0). Kept as a
// named entry point because the promoted route and its pinned conformance test
// call it, but it is NOT a second implementation.
inline TextureLoadSourceRange ComputeStrictRgba8SourceRange(
    uint32_t base, uint32_t blocks_w, uint32_t blocks_h,
    uint32_t pitch_blocks_aligned, uint32_t physical_size) {
  return ComputeStrictRgba8SourceRangeAtOrigin(base, 0, 0, blocks_w, blocks_h,
                                               pitch_blocks_aligned,
                                               physical_size);
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
