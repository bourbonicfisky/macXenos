/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_RESIDENCY_AUDIT_H_
#define XENIA_GPU_METAL_METAL_RESIDENCY_AUDIT_H_

// port-xenios-tessellation task 5.3 — the FINAL attribution slice.
//
// THE ONE QUESTION: for a tessellation-enabled render encoder, does what the
// GPU could DEREFERENCE match what Metal was TOLD IS RESIDENT?
//
// Everything here is a PURE function of numeric records, so the whole decision
// matrix is pinned at L0 without a device, and each classification is
// mutation-pinned independently. The capture side owns only recording.
//
// NUMERIC ONLY, BY CONSTRUCTION. These records deliberately cannot hold an
// Objective-C reference: retaining a texture to describe it later would extend
// its lifetime and SUPPRESS the fault being measured. A resource is a monotonic
// uint64 id; a slot is a monotonic generation. Pointer reuse is survived by
// erasing the address->id mapping at release, so a later texture at a recycled
// address gets a NEW id rather than inheriting a dead one.
//
// SCOPE DISCIPLINE: argument-buffer allocation generation/offset are carried as
// CONTEXT only. The arena's lifetime question was closed by the read-only audit
// (docs/metal_tessellation_arena_lifetime_audit.md); these fields exist to
// describe a record, not to reopen it. There are deliberately no arena tickets
// and no argument-buffer content hashing here.

#include <cstdint>
#include <vector>

namespace xe {
namespace gpu {
namespace metal {

// Stage bits mirror MetalBindingStageMask; kept as a plain mask so this header
// stays free of Metal types.
struct MetalEncodedSlot {
  uint32_t slot = 0;
  // Identity of the resource the DESCRIPTOR named when the draw encoded it.
  uint64_t resource_id = 0;
  uint64_t generation = 0;
  // Stages the draw actually needs this resource on.
  uint32_t required_stage_mask = 0;
};

// One useResource declaration made on the encoder.
struct MetalDeclaredResource {
  uint64_t resource_id = 0;
  // Slot generation observed at declaration time (the residency sweep runs at
  // encoder open; a per-draw declaration runs later).
  uint64_t generation = 0;
  uint32_t declared_stage_mask = 0;
};

struct MetalEncoderResidencyRecord {
  uint64_t submission = 0;
  uint64_t episode = 0;
  uint64_t rt_hash = 0;
  uint32_t draws_at_open = 0;
  bool tessellation = false;
  // Context only -- see the scope note above.
  uint64_t ab_allocation_generation = 0;
  uint64_t ab_offset = 0;
  std::vector<MetalEncodedSlot> encoded;
  std::vector<MetalDeclaredResource> declared;
};

enum class MetalResidencyVerdict : uint32_t {
  kConsistent = 0,
  // The encoded resource was never declared resident on this encoder at all.
  kResourceNotDeclared = 1 << 0,
  // Declared, but the declaration named a DIFFERENT generation of the slot --
  // i.e. the slot was rewritten between descriptor encoding and the sweep.
  kGenerationMismatch = 1 << 1,
  // Correct resource declared, but missing a stage the draw requires.
  kMissingRequiredStage = 1 << 2,
};

inline uint32_t operator|(MetalResidencyVerdict a, MetalResidencyVerdict b) {
  return uint32_t(a) | uint32_t(b);
}

// The whole classification, as a pure function. Returns a bitmask of
// MetalResidencyVerdict values; 0 means every encoded slot was declared with a
// matching generation and a sufficient stage mask.
inline uint32_t ClassifyMetalEncoderResidency(
    const MetalEncoderResidencyRecord& record) {
  uint32_t verdict = 0;
  for (const MetalEncodedSlot& slot : record.encoded) {
    if (!slot.resource_id) {
      // A slot naming nothing is not a residency question.
      continue;
    }
    const MetalDeclaredResource* match = nullptr;
    bool any_id_match = false;
    for (const MetalDeclaredResource& decl : record.declared) {
      if (decl.resource_id != slot.resource_id) {
        continue;
      }
      any_id_match = true;
      // Prefer a declaration whose generation agrees; a stale-generation
      // declaration is reported rather than silently accepted.
      if (decl.generation == slot.generation) {
        match = &decl;
        break;
      }
    }
    if (!any_id_match) {
      verdict |= uint32_t(MetalResidencyVerdict::kResourceNotDeclared);
      continue;
    }
    if (!match) {
      verdict |= uint32_t(MetalResidencyVerdict::kGenerationMismatch);
      continue;
    }
    if ((slot.required_stage_mask & ~match->declared_stage_mask) != 0) {
      verdict |= uint32_t(MetalResidencyVerdict::kMissingRequiredStage);
    }
  }
  return verdict;
}

inline bool MetalResidencyRecordIsConsistent(
    const MetalEncoderResidencyRecord& record) {
  return ClassifyMetalEncoderResidency(record) == 0;
}

// In-flight capture store. FAIL-LOUD ON EXHAUSTION rather than evicting: a
// fault storm is exactly when eviction would destroy the evidence, and an
// unfinished record silently dropped would read as "no mismatch found".
class MetalResidencyCaptureStore {
 public:
  explicit MetalResidencyCaptureStore(size_t capacity) : capacity_(capacity) {}

  // Returns false (and records nothing) when full. The caller must report the
  // overflow; it must never be treated as a successful capture.
  bool Add(const MetalEncoderResidencyRecord& record) {
    if (records_.size() >= capacity_) {
      ++dropped_;
      return false;
    }
    records_.push_back(record);
    return true;
  }

  // Retires every record for submissions at or below `completed`, returning
  // them for emission. Unfinished records are never removed.
  std::vector<MetalEncoderResidencyRecord> Drain(uint64_t completed) {
    std::vector<MetalEncoderResidencyRecord> out;
    std::vector<MetalEncoderResidencyRecord> keep;
    for (auto& r : records_) {
      if (r.submission <= completed) {
        out.push_back(r);
      } else {
        keep.push_back(r);
      }
    }
    records_.swap(keep);
    return out;
  }

  size_t size() const { return records_.size(); }
  uint64_t dropped() const { return dropped_; }
  bool healthy() const { return dropped_ == 0; }

 private:
  size_t capacity_;
  uint64_t dropped_ = 0;
  std::vector<MetalEncoderResidencyRecord> records_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RESIDENCY_AUDIT_H_
