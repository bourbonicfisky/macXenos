/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_EDRAM_LAYOUT_H_
#define XENIA_GPU_EDRAM_LAYOUT_H_

#include <cstdint>

#include "xenia/gpu/xenos.h"

// CPU-side canonical Xenos EDRAM layout and word helpers. The Metal equivalents
// live together in metal/metal_edram_msl.h and are checked against these
// by the L1 conformance suite. Keep all snapshot, diagnostic, and canonical-
// buffer code on these helpers rather than carrying local addressing formulas.

namespace xe {
namespace gpu {

struct EdramMsaaShifts {
  uint32_t x;
  uint32_t y;
};

constexpr EdramMsaaShifts GetEdramMsaaShifts(uint32_t msaa_samples) {
  return {msaa_samples >= uint32_t(xenos::MsaaSamples::k4X) ? 1u : 0u,
          msaa_samples >= uint32_t(xenos::MsaaSamples::k2X) ? 1u : 0u};
}

constexpr uint32_t EdramDepthSwapX(uint32_t tile_sample_x,
                                   uint32_t tile_width_samples) {
  uint32_t half_width = tile_width_samples >> 1u;
  return tile_sample_x >= half_width ? tile_sample_x - half_width
                                     : tile_sample_x + half_width;
}

// Byte offset of one sample in the raw 10 MiB tile-major EDRAM image.
// Mirrors shaders/edram.xesli XeEdramOffsetBytes at resolution_scale=(1,1),
// wrap=false.
constexpr uint32_t XeEdramSnapshotOffsetBytes(
    uint32_t pixel_x, uint32_t pixel_y, uint32_t base_tiles,
    uint32_t pitch_tiles, uint32_t msaa_samples, bool is_depth,
    uint32_t format_ints_log2, uint32_t pixel_sample_index) {
  EdramMsaaShifts shifts = GetEdramMsaaShifts(msaa_samples);
  uint32_t sample_x = (pixel_x << shifts.x) +
                      ((pixel_sample_index >> 1u) & 1u);
  uint32_t sample_y =
      (pixel_y << shifts.y) + (pixel_sample_index & 1u);
  uint32_t tile_width_samples =
      xenos::kEdramTileWidthSamples >> format_ints_log2;
  uint32_t tile_x = sample_x / tile_width_samples;
  uint32_t tile_y = sample_y / xenos::kEdramTileHeightSamples;
  base_tiles += tile_y * pitch_tiles + tile_x;
  sample_x -= tile_x * tile_width_samples;
  sample_y -= tile_y * xenos::kEdramTileHeightSamples;
  if (is_depth) {
    sample_x = EdramDepthSwapX(sample_x, tile_width_samples);
  }
  uint32_t word_index =
      base_tiles *
          (xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples) +
      ((sample_y * tile_width_samples + sample_x) << format_ints_log2);
  return word_index * sizeof(uint32_t);
}

constexpr uint32_t XeEdramPitchTiles(uint32_t surface_pitch_pixels,
                                     uint32_t msaa_samples,
                                     uint32_t format_ints_log2 = 0u) {
  EdramMsaaShifts shifts = GetEdramMsaaShifts(msaa_samples);
  uint32_t width_samples = surface_pitch_pixels << shifts.x;
  uint32_t tile_width_samples =
      xenos::kEdramTileWidthSamples >> format_ints_log2;
  return (width_samples + tile_width_samples - 1u) / tile_width_samples;
}

// Canonical 32bpp 4x sample plane of a tile-local raw dword. This is the same
// parity classification used by XEDH v2 diagnostics.
constexpr uint32_t EdramSamplePlaneForTileWord(uint32_t tile_word) {
  uint32_t x = tile_word % xenos::kEdramTileWidthSamples;
  uint32_t y = tile_word / xenos::kEdramTileWidthSamples;
  return ((x & 1u) << 1u) | (y & 1u);
}

constexpr uint32_t PackEdramColor8888(uint32_t red, uint32_t green,
                                      uint32_t blue, uint32_t alpha) {
  return (red & 0xFFu) | ((green & 0xFFu) << 8u) |
         ((blue & 0xFFu) << 16u) | ((alpha & 0xFFu) << 24u);
}

constexpr uint32_t GetEdramColor8888Channel(uint32_t word,
                                            uint32_t channel) {
  return (word >> ((channel & 3u) * 8u)) & 0xFFu;
}

constexpr uint32_t PackEdramDepthStencil(uint32_t depth24,
                                         uint32_t stencil) {
  return ((depth24 & 0xFFFFFFu) << 8u) | (stencil & 0xFFu);
}

constexpr uint32_t GetEdramDepth24(uint32_t word) { return word >> 8u; }
constexpr uint32_t GetEdramStencil(uint32_t word) { return word & 0xFFu; }
constexpr uint32_t GetEdramStencilBitMask(uint32_t bit_index) {
  return 1u << (bit_index & 7u);
}

// CPU mirror of the canonical Metal SurfaceParams constant-buffer ABI.
struct EdramSurfaceParams {
  uint32_t base;
  uint32_t pitch;
  uint32_t msaa;
  // 0=color, 1=D24S8, 2=D24FS8. Any nonzero value applies the depth swap.
  uint32_t is_depth;
  uint32_t format_ints_log2;
  uint32_t words;
};

// Whether a canonical EDRAM materialize range [start_tiles, end_tiles) safely
// allows MTLLoadActionDontCare on a tight materialize render pass of height
// `work_height` targeting an attachment of `texture_height`.
//
// On Apple Silicon TBDR GPUs, render passes load and store attachments in
// hardware tiles (e.g. 32 px tall). When work_height < texture_height and
// lands mid-tile, MTLLoadActionDontCare leaves rows [work_height,
// next_tile_boundary) undefined in tile memory and MTLStoreActionStore writes
// them back over live rows below the pass. Require work_height ==
// texture_height so DontCare is only used when the pass covers the entire
// attachment; partial-height passes keep MTLLoadActionLoad while still
// bounding pass.renderTargetHeight to work_height.
constexpr bool CanonicalMaterializeRangeAllowsDontCare(
    uint32_t base_tiles, uint32_t pitch_tiles, uint32_t msaa_samples,
    uint32_t start_tiles, uint32_t end_tiles, uint64_t work_height,
    uint64_t texture_height, uint32_t scale_y) {
  if (!pitch_tiles || !work_height || work_height != texture_height ||
      start_tiles > base_tiles) {
    return false;
  }
  const EdramMsaaShifts shifts = GetEdramMsaaShifts(msaa_samples);
  const uint64_t tile_row_height =
      uint64_t(xenos::kEdramTileHeightSamples >> shifts.y) * scale_y;
  if (!tile_row_height) {
    return false;
  }
  const uint64_t tile_rows =
      (work_height + tile_row_height - 1u) / tile_row_height;
  return uint64_t(end_tiles) >=
         uint64_t(base_tiles) + tile_rows * uint64_t(pitch_tiles);
}
static_assert(sizeof(EdramSurfaceParams) == 6u * sizeof(uint32_t));

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_EDRAM_LAYOUT_H_
