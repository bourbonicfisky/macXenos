/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_PRESENT_CENSUS_H_
#define XENIA_GPU_METAL_METAL_PRESENT_CENSUS_H_

// Contract 2 discovery: what the presentation path actually does, measured
// rather than inferred from source.
//
// The gamma work established one fact that motivates this: across all five
// calibrated titles, EVERY sampled swap presented via
// ram-frontbuffer-canonical-snapshot, and the resolved-snapshot path never
// appeared. That was sampled every 60 swaps, so it is suggestive rather than
// exhaustive -- this census counts every swap instead, so "never" can be
// distinguished from "never sampled".
//
// It answers, per title and over a WHOLE run including boot and FMV rather than
// only a settled scene:
//   - which present source each swap used, exhaustively;
//   - what source format and size the swap source had;
//   - how guest frontbuffer size relates to the presented size;
//   - how often either changed, since a size that changes mid-run is what makes
//     resize handling a real question rather than a hypothetical one.
//
// Bounded by construction: a fixed set of counters plus a small table of
// distinct observed values. It cannot grow with run length, and it stores no
// guest memory contents -- only sizes, formats, and path identities.
//
// PURELY OBSERVATIONAL. It reads state that already exists at the point the
// swap source has been chosen, and emits nothing into presentation, bindings,
// or synchronization.

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

class MetalPresentCensus {
 public:
  // Distinct present-source path identities tracked. The backend has seven
  // labels today; the spare slots exist so a new one shows up as itself rather
  // than silently merging into another.
  static constexpr uint32_t kMaxPaths = 12;
  // Distinct (width, height) pairs recorded before further ones are only
  // counted. Small on purpose: the question is whether the size is STABLE, and
  // a handful of distinct values already answers it.
  static constexpr uint32_t kMaxSizes = 8;
  static constexpr uint32_t kMaxFormats = 8;

  void ObserveSwap(const char* path, uint32_t source_format,
                   uint32_t present_width, uint32_t present_height,
                   uint32_t frontbuffer_width, uint32_t frontbuffer_height) {
    ++swaps_;
    BumpPath(path);
    BumpFormat(source_format);
    BumpSize(present_sizes_, present_size_counts_, present_size_distinct_,
             present_width, present_height);
    BumpSize(frontbuffer_sizes_, frontbuffer_size_counts_,
             frontbuffer_size_distinct_, frontbuffer_width, frontbuffer_height);
    // A presented size that differs from the guest's own frontbuffer size means
    // something in the chain is scaling; counting it separates "scaling always
    // happens" from "scaling happens sometimes", which need different work.
    if (present_width != frontbuffer_width ||
        present_height != frontbuffer_height) {
      ++swaps_size_mismatched_;
    }
  }

  uint64_t swaps() const { return swaps_; }
  uint64_t swaps_size_mismatched() const { return swaps_size_mismatched_; }
  uint32_t path_distinct() const { return path_distinct_; }
  const char* path_name(uint32_t i) const { return path_names_[i]; }
  uint64_t path_count(uint32_t i) const { return path_counts_[i]; }
  uint32_t format_distinct() const { return format_distinct_; }
  uint32_t format_value(uint32_t i) const { return formats_[i]; }
  uint64_t format_count(uint32_t i) const { return format_counts_[i]; }
  uint32_t present_size_distinct() const { return present_size_distinct_; }
  uint32_t present_size_w(uint32_t i) const { return present_sizes_[i] >> 16; }
  uint32_t present_size_h(uint32_t i) const {
    return present_sizes_[i] & 0xFFFFu;
  }
  uint64_t present_size_count(uint32_t i) const {
    return present_size_counts_[i];
  }
  uint32_t frontbuffer_size_distinct() const {
    return frontbuffer_size_distinct_;
  }
  uint32_t frontbuffer_size_w(uint32_t i) const {
    return frontbuffer_sizes_[i] >> 16;
  }
  uint32_t frontbuffer_size_h(uint32_t i) const {
    return frontbuffer_sizes_[i] & 0xFFFFu;
  }
  uint64_t frontbuffer_size_count(uint32_t i) const {
    return frontbuffer_size_counts_[i];
  }

  void Reset() { *this = MetalPresentCensus(); }

 private:
  // Path identities are string literals from a fixed set of sites, so pointer
  // identity is sufficient and avoids a per-swap string compare. A literal that
  // is pooled with another simply merges two names that are already equal.
  void BumpPath(const char* path) {
    for (uint32_t i = 0; i < path_distinct_; ++i) {
      if (path_names_[i] == path) {
        ++path_counts_[i];
        return;
      }
    }
    if (path_distinct_ < kMaxPaths) {
      path_names_[path_distinct_] = path;
      path_counts_[path_distinct_] = 1;
      ++path_distinct_;
    } else {
      ++path_overflow_;
    }
  }

  void BumpFormat(uint32_t format) {
    for (uint32_t i = 0; i < format_distinct_; ++i) {
      if (formats_[i] == format) {
        ++format_counts_[i];
        return;
      }
    }
    if (format_distinct_ < kMaxFormats) {
      formats_[format_distinct_] = format;
      format_counts_[format_distinct_] = 1;
      ++format_distinct_;
    }
  }

  static void BumpSize(uint32_t* sizes, uint64_t* counts, uint32_t& distinct,
                       uint32_t w, uint32_t h) {
    const uint32_t key = (w << 16) | (h & 0xFFFFu);
    for (uint32_t i = 0; i < distinct; ++i) {
      if (sizes[i] == key) {
        ++counts[i];
        return;
      }
    }
    if (distinct < kMaxSizes) {
      sizes[distinct] = key;
      counts[distinct] = 1;
      ++distinct;
    }
  }

  uint64_t swaps_ = 0;
  uint64_t swaps_size_mismatched_ = 0;
  const char* path_names_[kMaxPaths] = {};
  uint64_t path_counts_[kMaxPaths] = {};
  uint32_t path_distinct_ = 0;
  uint64_t path_overflow_ = 0;
  uint32_t formats_[kMaxFormats] = {};
  uint64_t format_counts_[kMaxFormats] = {};
  uint32_t format_distinct_ = 0;
  uint32_t present_sizes_[kMaxSizes] = {};
  uint64_t present_size_counts_[kMaxSizes] = {};
  uint32_t present_size_distinct_ = 0;
  uint32_t frontbuffer_sizes_[kMaxSizes] = {};
  uint64_t frontbuffer_size_counts_[kMaxSizes] = {};
  uint32_t frontbuffer_size_distinct_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PRESENT_CENSUS_H_
