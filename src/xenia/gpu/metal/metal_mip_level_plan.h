/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MIP_LEVEL_PLAN_H_
#define XENIA_GPU_METAL_METAL_MIP_LEVEL_PLAN_H_

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "xenia/base/math.h"
#include "xenia/gpu/metal/metal_texture_load_eligibility.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

// Pure (device-free) per-level plan for the OpenSpec 6.1 mipped sibling of the
// promoted strict-RGBA8 direct-load route. Extracted as a free function so it
// is the SINGLE planning authority: production calls it, and conformance tests
// it directly rather than through a device harness.
//
// Every per-level (offset, pitch, size, packed sub-tile) value is derived from
// the shared layout helpers (GetGuestTextureLayout + GetPackedMipOffset) -- the
// same selection D3D12TextureCache::LoadTextureDataFromResidentMemoryImpl
// makes. No tiled, packing or endian math is re-implemented here.

namespace xe {
namespace gpu {
namespace metal {

struct MipLevelPlan {
  uint32_t host_level = 0;
  uint32_t guest_offset = 0;    // absolute guest byte address of the level
  uint32_t pitch_blocks = 0;    // row pitch in blocks (== texels at 32bpp)
  uint32_t width = 0;           // level texels
  uint32_t height = 0;
  uint32_t px = 0;              // packed sub-tile block origin
  uint32_t py = 0;
  uint32_t consumed_begin = 0;  // exact read range, absolute, inclusive
  uint32_t consumed_end = 0;    // exclusive
};

// Memo for the exact-read-range scan, which is the planner's only non-trivial
// cost: ComputeStrictRgba8SourceRangeAtOrigin evaluates Tiled2D once per
// 4-block group over the padded grid, measured at ~0.53 ms for a 1280x720
// level on an M4. The result depends ONLY on (px, py, blocks_w, blocks_h,
// pitch_blocks) -- never on the guest address -- so it is identical for every
// texture of a shape and the promoted single-mip route re-derives the same
// answer on every miss without this.
//
// It is a pure memo, not a second implementation: a miss calls the one
// authority and stores what it returned, so a memoized plan is equal to an
// unmemoized plan by construction (and pinned by conformance). Passing nullptr
// is always valid and always gives the same plan.
class MipLevelRangeMemo {
 public:
  TextureLoadSourceRange Get(uint32_t px, uint32_t py, uint32_t blocks_w,
                             uint32_t blocks_h, uint32_t pitch_blocks) {
    Key key{px, py, blocks_w, blocks_h, pitch_blocks};
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      return it->second;
    }
    // Relative to offset 0 and unbounded: the caller applies the texture's own
    // guest address and the real buffer bound, which are NOT part of the key.
    TextureLoadSourceRange range = ComputeStrictRgba8SourceRangeAtOrigin(
        0, px, py, blocks_w, blocks_h, pitch_blocks, UINT32_MAX);
    entries_.emplace(key, range);
    return range;
  }

  // Derived scan results keyed by shape; cheap to recompute and meaningless
  // once the textures are gone.
  void Clear() { entries_.clear(); }
  size_t size() const { return entries_.size(); }

 private:
  struct Key {
    uint32_t px, py, w, h, pitch;
    bool operator==(const Key& o) const {
      return px == o.px && py == o.py && w == o.w && h == o.h &&
             pitch == o.pitch;
    }
  };
  struct KeyHasher {
    size_t operator()(const Key& k) const {
      size_t hash = size_t(k.w) * 0x9E3779B1u;
      hash ^= size_t(k.h) << 16;
      hash ^= size_t(k.pitch) * 0x85EBCA6Bu;
      hash ^= (size_t(k.px) << 8) ^ (size_t(k.py) << 24);
      return hash;
    }
  };
  std::unordered_map<Key, TextureLoadSourceRange, KeyHasher> entries_;
};

// Inputs the planner needs, all already available on the shared TextureKey and
// its layout. Keeping them explicit is what makes the planner device-free.
struct MipLevelPlanInputs {
  uint32_t base_address = 0;  // key.base_page << 12
  uint32_t mip_address = 0;   // key.mip_page << 12
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mip_max_level = 0;  // from the KEY
  uint32_t guest_buffer_size = 0;
  bool load_base = false;
  bool load_mips = false;
};

// Builds the COMPLETE plan for the requested regions. Returns false -- meaning
// the caller must fall back to the legacy cache atomically, before encoding
// anything -- if any level cannot be planned or validated.
//
// `layout` must be the texture's own guest layout, and `in.mip_max_level` the
// KEY's level count; the two are cross-checked so a descriptor built from one
// can never disagree with a plan built from the other.
//
// `memo` is an optional pure cache of the per-shape read-range scan (see
// MipLevelRangeMemo). It changes cost only, never the plan.
inline bool BuildMipLevelPlan(const MipLevelPlanInputs& in,
                              const texture_util::TextureGuestLayout& layout,
                              std::vector<MipLevelPlan>* out,
                              MipLevelRangeMemo* memo = nullptr) {
  if (!out || (!in.load_base && !in.load_mips)) {
    return false;
  }
  out->clear();
  // Key/layout agreement: the layout clamps max_level to what the dimensions
  // can actually hold, so a key claiming more levels than the layout stores is
  // a planning error rather than something to silently truncate.
  if (in.mip_max_level != layout.max_level) {
    return false;
  }
  uint32_t packed = layout.packed_level;

  auto plan_one = [&](uint32_t level, uint64_t guest_offset,
                      uint32_t pitch_bytes, bool is_packed) -> bool {
    MipLevelPlan p;
    p.host_level = level;
    p.pitch_blocks = pitch_bytes / 4u;  // 32bpp: block == texel == 4 bytes
    p.width = std::max(in.width >> level, 1u);
    p.height = std::max(in.height >> level, 1u);
    if (is_packed) {
      uint32_t px = 0, py = 0, pz = 0;
      texture_util::GetPackedMipOffset(in.width, in.height, 1,
                                       xenos::TextureFormat::k_8_8_8_8, level,
                                       px, py, pz);
      p.px = px;
      p.py = py;
    }
    if (!p.pitch_blocks || !p.width || !p.height) {
      return false;
    }
    // 64-bit throughout, validated, THEN narrowed: a guest pointer plus a mip
    // offset can exceed 32 bits before any bounds check would see it.
    if (guest_offset >= in.guest_buffer_size) {
      return false;
    }
    p.guest_offset = uint32_t(guest_offset);
    TextureLoadSourceRange r =
        memo ? memo->Get(p.px, p.py, p.width, p.height, p.pitch_blocks)
             : ComputeStrictRgba8SourceRangeAtOrigin(0, p.px, p.py, p.width,
                                                     p.height, p.pitch_blocks,
                                                     UINT32_MAX);
    if (!r.valid) {
      return false;
    }
    uint64_t begin = guest_offset + uint64_t(r.begin);
    uint64_t end = guest_offset + uint64_t(r.end);
    if (end > uint64_t(in.guest_buffer_size) || begin >= end) {
      return false;
    }
    p.consumed_begin = uint32_t(begin);
    p.consumed_end = uint32_t(end);
    out->push_back(p);
    return true;
  };

  if (in.load_base) {
    if (!in.base_address) {
      return false;
    }
    // packed_level == 0 means the BASE itself lives in a packed mip tail stored
    // at the base address; the shared loader keeps a SEPARATE mips tail at the
    // mip address. Level 0 is the tail's largest level, so its sub-tile origin
    // is whatever GetPackedMipOffset reports for level 0.
    if (!plan_one(0, uint64_t(in.base_address), layout.base.row_pitch_bytes,
                  packed == 0)) {
      return false;
    }
  }
  if (in.load_mips) {
    if (!in.mip_max_level || !in.mip_address) {
      return false;
    }
    for (uint32_t level = 1; level <= in.mip_max_level; ++level) {
      // The layout entry this level is STORED in: its own for a non-packed
      // level, the packed tail's entry for every packed level.
      uint32_t store =
          packed == UINT32_MAX ? level : std::min(level, packed);
      if (store >= xe::countof(layout.mips)) {
        return false;
      }
      bool is_packed = packed != UINT32_MAX && level >= packed;
      if (!plan_one(level,
                    uint64_t(in.mip_address) +
                        uint64_t(layout.mip_offsets_bytes[store]),
                    layout.mips[store].row_pitch_bytes, is_packed)) {
        return false;
      }
    }
  }
  return !out->empty();
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MIP_LEVEL_PLAN_H_
