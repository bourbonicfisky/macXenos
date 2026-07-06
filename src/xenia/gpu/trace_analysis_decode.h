/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_TRACE_ANALYSIS_DECODE_H_
#define XENIA_GPU_TRACE_ANALYSIS_DECODE_H_

#include <cstdint>
#include <filesystem>
#include <vector>

#include "xenia/gpu/xenos.h"

// In-tree authoritative decoder for the headless trace-analysis suite
// (build-headless-trace-analysis-suite, design D7). Turns a raw EDRAM
// snapshot (draw_NNNNNN.edram.bin, produced by --capture_dir/--capture_*) plus
// its manifest row into color/depth images, reusing the exact Xenos EDRAM tile
// addressing (mirrors src/xenia/gpu/shaders/edram.xesli) rather than
// reimplementing tiling in Python.

namespace xe {
namespace gpu {

// Byte offset of one sample into a raw EDRAM snapshot. Faithful C++ mirror of
// edram.xesli `XeEdramOffsetBytes` for resolution_scale = 1 (headless capture
// is unscaled: InitializeTraceSubmitDownloads refuses scaled draws).
uint32_t XeEdramSnapshotOffsetBytes(uint32_t pixel_x, uint32_t pixel_y,
                                    uint32_t base_tiles, uint32_t pitch_tiles,
                                    uint32_t msaa_samples, bool is_depth,
                                    uint32_t format_ints_log2,
                                    uint32_t pixel_sample_index);

// EDRAM tile pitch (in tiles) for a color/depth surface of the given guest
// surface pitch (in pixels) at the given MSAA level.
uint32_t XeEdramPitchTiles(uint32_t surface_pitch_pixels,
                           uint32_t msaa_samples);

// A decoded RGBA8 image.
struct DecodedImage {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<uint8_t> rgba;  // width*height*4, row-major.
};

// Decodes a 32bpp color render target from an EDRAM snapshot into RGBA8 for one
// MSAA sample. Returns false on bad params / out-of-range offsets.
bool DecodeEdramColor32bpp(const uint8_t* edram, size_t edram_size,
                           uint32_t base_tiles, uint32_t pitch_tiles,
                           uint32_t msaa_samples,
                           xenos::ColorRenderTargetFormat format,
                           uint32_t width, uint32_t height,
                           uint32_t sample_index, DecodedImage* out);

// Decodes a depth render target from an EDRAM snapshot into an RGBA8 grayscale
// visualization (depth as luminance) for one MSAA sample. `format` selects
// unorm24 (kD24S8) vs float24 20e4 (kD24FS8). Returns false on bad params.
bool DecodeEdramDepth(const uint8_t* edram, size_t edram_size,
                      uint32_t base_tiles, uint32_t pitch_tiles,
                      uint32_t msaa_samples,
                      xenos::DepthRenderTargetFormat format, uint32_t width,
                      uint32_t height, uint32_t sample_index, DecodedImage* out);

// Round-trip drift guard (design: Tier 0). Verifies the offset formula is a
// bijection over a covered region for representative (msaa, format, depth)
// combinations -- i.e. capture -> offset -> re-tile == input. Returns true on
// success; logs the first failing case otherwise. Runnable inside trace-dump
// (which executes in agent/CI sessions), unlike a standalone test binary.
bool RunEdramDecodeSelfTest();

// Decodes every draw_*.edram.bin in a capture directory using params from its
// manifest.jsonl, writing color_drawNNNNNN.png next to each snapshot. Returns
// the number of snapshots decoded, or -1 on fatal error (e.g. self-test fail).
int DecodeCaptureDirectory(const std::filesystem::path& capture_dir,
                           uint32_t width, uint32_t height,
                           uint32_t sample_index);

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_TRACE_ANALYSIS_DECODE_H_
