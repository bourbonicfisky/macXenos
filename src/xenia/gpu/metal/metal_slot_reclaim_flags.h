/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SLOT_RECLAIM_FLAGS_H_
#define XENIA_GPU_METAL_METAL_SLOT_RECLAIM_FLAGS_H_

// port-xenios-tessellation task 5.3, candidate 3: ONE binary, THREE explicit
// modes for what ReclaimStaleSlots does to a freed bindless slot, selected by
// XE_METAL_SLOT_RECLAIM. Default is current behaviour, so an unset environment
// is bit-identical to a build without this flag.
//
//   unset / "" / "current"  kCurrent
//       Today's behaviour. The texture is released and the slot is pushed onto
//       free_slots_ WITHOUT rewriting its view-heap descriptor, on the stated
//       assumption that "descriptors are overwritten on the next AllocateSlot
//       that hands a freed slot out". Between reclaim and that reallocation a
//       LIVE bindless heap holds a descriptor naming deallocated memory.
//
//   "widen"                 kWidenWindow
//       DIAGNOSTIC. Releases the texture as usual but HOLDS the slot out of
//       free_slots_ for an extra span of submissions, so the dangling
//       descriptor persists longer before a reallocation overwrites it. This
//       is a ONE-DIRECTIONAL probe: a higher fault rate is strong support for
//       the dangling-descriptor mechanism, but **a flat rate does NOT falsify
//       it** — the natural window may already cover the vulnerable access, and
//       the experiment may lack the power to separate the two rates. Never
//       record a flat widen result as a refutation.
//
//   "null"                  kNullOnReclaim
//       CANDIDATE FIX. Rebinds the slot to the null texture BEFORE it enters
//       free_slots_, exactly as ClearCache's free_slot already does, so a
//       freed slot describes a valid resource instead of a released one.
//       Submission gating is unchanged: only entries whose recorded
//       last-referencing submission has completed are touched at all.
//
// Anything else is kMalformed: the caller warns once and behaves as kCurrent
// (fail closed to today's behaviour, never to an untested mode).

#include <cstdint>
#include <cstring>

namespace xe {
namespace gpu {
namespace metal {

enum class MetalSlotReclaimMode {
  kCurrent,
  kWidenWindow,
  kNullOnReclaim,
  kMalformed,
};

inline MetalSlotReclaimMode ParseMetalSlotReclaimMode(const char* value) {
  if (!value || !value[0] || !std::strcmp(value, "current")) {
    return MetalSlotReclaimMode::kCurrent;
  }
  if (!std::strcmp(value, "widen")) {
    return MetalSlotReclaimMode::kWidenWindow;
  }
  if (!std::strcmp(value, "null")) {
    return MetalSlotReclaimMode::kNullOnReclaim;
  }
  return MetalSlotReclaimMode::kMalformed;
}

// Behavioural mode, with kMalformed folded to kCurrent.
inline MetalSlotReclaimMode MetalSlotReclaimModeEffective(
    MetalSlotReclaimMode mode) {
  return mode == MetalSlotReclaimMode::kMalformed ? MetalSlotReclaimMode::kCurrent
                                                  : mode;
}

// How many submissions past completion the widen probe holds a freed slot out
// of the free list. Submissions run ~20 per frame here, so 16 is roughly one
// extra frame of dangling window -- long enough to matter, short enough that
// the 4096-slot heap is not starved by held slots under eviction churn.
constexpr uint64_t kMetalSlotReclaimWidenHoldSubmissions = 16;

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SLOT_RECLAIM_FLAGS_H_
