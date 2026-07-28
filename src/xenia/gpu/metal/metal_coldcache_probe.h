/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_COLDCACHE_PROBE_H_
#define XENIA_GPU_METAL_METAL_COLDCACHE_PROBE_H_

// TEMPORARY DIAGNOSTIC — task group 0 of
// openspec/changes/implement-metal-async-pipeline-creation.
//
// Measures the COLD-CACHE LAUNCH baseline: a launch whose title M1 cache is
// empty. NOT a "first-ever title launch" — the fixture runs against existing
// configuration and save state.
//
// The three pools are timed SEPARATELY and never summed into one "creation"
// number, because they are addressed by different mechanisms and only one of
// them is M3's:
//
//   TRANSLATION  ucode -> DXBC -> DXIL -> MSC -> MTLFunction.  <- M3's target
//                Removed on a warm launch by M1 (promoted).
//   PSTATE       newRenderPipelineState / IRRuntime geometry pipeline.
//                NOT M3's target. Measured state-dependent (0.089-12.384 ms
//                mean for the same set) and closed NO-GO as M2. Reported here
//                only so it can be EXCLUDED from M3's justification.
//   GEOMSTAGE    MSC object/mesh stage conversion. Also M1 territory.
//
// Per-frame concentration is tracked for TRANSLATION specifically: a run total
// says nothing about whether the cost is a hitch.
//
// REMOVAL CONDITION (task 6.3): this file and its call sites are deleted at
// closeout regardless of outcome. Header-only, in no CMakeLists.
//
// Enable: XE_METAL_COLDCACHE_PROBE=1

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {
namespace coldcache_probe {

enum class Pool : uint32_t {
  kTranslation = 0,  // M3's target pool.
  kPState,           // M2's pool — excluded from M3's justification.
  kGeomStage,        // M1 territory.
  kCount,
};

inline const char* PoolName(Pool p) {
  switch (p) {
    case Pool::kTranslation:
      return "TRANSLATION";
    case Pool::kPState:
      return "PSTATE";
    case Pool::kGeomStage:
      return "GEOMSTAGE";
    default:
      return "?";
  }
}

inline constexpr double kThresholds[4] = {5.0, 1000.0 / 60.0, 33.3, 100.0};

class State {
 public:
  static State& Get() {
    static State instance;
    return instance;
  }
  bool enabled() const { return enabled_; }

  void Record(Pool pool, double ms, bool m1_hit) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t i = uint32_t(pool);
    ++count_[i];
    total_ms_[i] += ms;
    max_ms_[i] = std::max(max_ms_[i], ms);
    if (pool == Pool::kTranslation) {
      // Separating hits from misses is what makes "cold cache" checkable
      // rather than asserted: a cold run should be nearly all misses.
      if (m1_hit) {
        ++m1_hits_;
        m1_hit_ms_ += ms;
      } else {
        ++m1_misses_;
      }
      if (frame_ms_.size() <= frame_index_) {
        frame_ms_.resize(frame_index_ + 1, 0.0);
      }
      frame_ms_[frame_index_] += ms;
    }
  }

  void AdvanceFrame() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++frame_index_;
    if ((frame_index_ % 512u) == 0u) {
      ReportLocked();
    }
  }

  void Report() {
    std::lock_guard<std::mutex> lock(mutex_);
    ReportLocked();
  }

 private:
  State() {
    const char* v = std::getenv("XE_METAL_COLDCACHE_PROBE");
    enabled_ = v && v[0] && v[0] != '0';
    if (enabled_) {
      XELOGI(
          "MetalColdCache: ENABLED (cold-cache launch baseline; pools timed "
          "separately, never summed)");
    }
  }

  void ReportLocked() {
    for (uint32_t i = 0; i < uint32_t(Pool::kCount); ++i) {
      if (!count_[i]) {
        continue;
      }
      XELOGI(
          "MetalColdCache {:<11} n={:<5} total={:9.2f} ms  mean={:7.3f} ms  "
          "max={:8.3f} ms",
          PoolName(Pool(i)), count_[i], total_ms_[i],
          total_ms_[i] / double(count_[i]), max_ms_[i]);
    }
    XELOGI(
        "MetalColdCache   M1 translation: misses={} hits={} ({:.2f} ms in hits)"
        "  <- a cold-cache launch should be nearly all misses",
        m1_misses_, m1_hits_, m1_hit_ms_);

    // Per-frame concentration, TRANSLATION only. This is the hitch claim.
    uint32_t over[4] = {};
    uint32_t frames_with_any = 0;
    double worst = 0.0;
    uint32_t worst_frame = 0;
    for (size_t f = 0; f < frame_ms_.size(); ++f) {
      if (frame_ms_[f] > 0.0) {
        ++frames_with_any;
      }
      if (frame_ms_[f] > worst) {
        worst = frame_ms_[f];
        worst_frame = uint32_t(f);
      }
      for (int t = 0; t < 4; ++t) {
        if (frame_ms_[f] > kThresholds[t]) {
          ++over[t];
        }
      }
    }
    XELOGI(
        "MetalColdCache   TRANSLATION per-frame: frames={} with-any={} "
        "worst frame#{} = {:.3f} ms",
        frame_index_, frames_with_any, worst_frame, worst);
    for (int t = 0; t < 4; ++t) {
      XELOGI("MetalColdCache   TRANSLATION frames over {:6.2f} ms: {}",
             kThresholds[t], over[t]);
    }
  }

  bool enabled_ = false;
  std::mutex mutex_;
  std::array<uint64_t, 3> count_ = {};
  std::array<double, 3> total_ms_ = {};
  std::array<double, 3> max_ms_ = {};
  uint64_t m1_hits_ = 0;
  uint64_t m1_misses_ = 0;
  double m1_hit_ms_ = 0.0;
  uint32_t frame_index_ = 0;
  std::vector<double> frame_ms_;
};

inline bool Enabled() { return State::Get().enabled(); }
inline void AdvanceFrame() {
  if (Enabled()) {
    State::Get().AdvanceFrame();
  }
}
inline void Report() {
  if (Enabled()) {
    State::Get().Report();
  }
}

class Scope {
 public:
  explicit Scope(Pool pool) : pool_(pool), enabled_(Enabled()) {
    if (enabled_) {
      start_ = std::chrono::steady_clock::now();
    }
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() {
    if (!enabled_) {
      return;
    }
    State::Get().Record(pool_,
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start_)
                            .count(),
                        m1_hit_);
  }
  void set_m1_hit() { m1_hit_ = true; }

 private:
  Pool pool_;
  bool enabled_;
  bool m1_hit_ = false;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace coldcache_probe
}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_COLDCACHE_PROBE_H_
