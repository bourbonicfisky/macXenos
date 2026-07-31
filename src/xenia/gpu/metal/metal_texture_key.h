/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_TEXTURE_KEY_H_
#define XENIA_GPU_METAL_METAL_TEXTURE_KEY_H_

// Legacy Metal texture cache identity
// (openspec/changes/implement-metal-texture-key-mip-range).
//
// This is the smallest production-owned piece of the key: the composition
// itself, with no fetch-constant parsing and no cache state. It exists as a
// header so conformance can exercise the REAL key construction instead of
// re-implementing it — a re-implemented key would agree with itself forever
// while production drifted.
//
// The caller supplies the NORMALIZED EFFECTIVE mip range (from
// texture_util::GetSubresourcesFromFetchConstant), never raw fetch fields.
// Normalization is what lets equivalent ranges share one entry: an absent mip
// page collapses the range to the base level, and both ends are bounded by the
// size-derived maximum. Passing raw register levels here is a defect, and the
// conformance suite mutation-tests exactly that substitution.

#include <cstdint>

#include "third_party/xxhash/xxhash.h"

namespace xe {
namespace gpu {
namespace metal {

// Every field that distinguishes one host texture's storage from another.
//
// Deliberately absent as DIRECT inputs, each with zero measured engagement
// across the trace corpus: the raw mip address, packed-mips state, stacked
// identity, and 3D/stack depth. The first and, for 3D, the last still reach the
// key INDIRECTLY through the normalized range, which is correct — see the
// capability spec.
struct MetalTextureKeyInputs {
  uint32_t base = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pitch_texels = 0;
  // Packed format/signedness/dimensionality/tiling/endianness, as the cache
  // composes it.
  uint32_t format_tiled_endian = 0;
  // The 12-bit guest fetch swizzle baked into the sampled view.
  uint32_t swizzle = 0;
  // NORMALIZED effective range. Not the raw fetch fields.
  uint32_t mip_min_level = 0;
  uint32_t mip_max_level = 0;
};

inline uint64_t MakeMetalTextureKey(const MetalTextureKeyInputs& inputs) {
  struct KeyBytes {
    uint32_t base, width, height, pitch;
    uint32_t format_tiled_endian;
    uint32_t swizzle;
    uint32_t mip_level_range;
  } kb{inputs.base,
       inputs.width,
       inputs.height,
       inputs.pitch_texels,
       inputs.format_tiled_endian,
       inputs.swizzle,
       (inputs.mip_min_level << 8) | (inputs.mip_max_level & 0xFF)};
  return XXH3_64bits(&kb, sizeof(kb));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_TEXTURE_KEY_H_
