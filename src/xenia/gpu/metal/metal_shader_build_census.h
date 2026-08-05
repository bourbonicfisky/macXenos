/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHADER_BUILD_CENSUS_H_
#define XENIA_GPU_METAL_METAL_SHADER_BUILD_CENSUS_H_

// Per-stage cost census for the shader/pipeline build chain
// (adopt-xenios-shader-artifact-caching task 1.1 — the baseline every caching
// stage is measured against).
//
// WHY THIS IS A PERMANENT COUNTER AND NOT A ONE-OFF SCRIPT. 1.1's baseline has
// to be re-measurable after each caching stage lands, on the same definitions,
// or the deltas in 2.4 / 3.3 / 5.6 are comparisons across measurement methods
// rather than across builds. The counters are the definition.
//
// WHAT IT SEPARATES, AND WHY EACH SPLIT EARNS ITS PLACE:
//
//   dxbc        stage 1, microcode -> DXBC. Runs on a metallib cache HIT too
//               (it keeps translated_binary() populated for the geometry
//               stages), so it is the floor a metallib hit cannot go below.
//   dxil        stage 2, DXBC -> DXIL. THE STAGE THE DXIL CACHE REMOVES.
//               Also carries the fresh-IDxbcConverter-per-Convert cost the
//               2026-08-02 fix added, which is the concrete performance
//               justification for caching its result.
//   msc         stage 3, DXIL -> metallib (Metal Shader Converter).
//   fn          metallib -> MTLFunction on the device.
//   pso         newRenderPipelineState — host pipeline state.
//   gpso        the geometry-emulation mesh pipeline (IR runtime).
//
// COLD vs FIRST-ENCOUNTER. Build work before the first swap is startup cost;
// build work after it is a stall inside a live frame, which is what the async
// stage (5.x) exists to move off the frame path. They are counted separately
// because a change can improve one and worsen the other. Per-frame build time
// is bucketed at 8/16/33 ms rather than averaged: a mean hides exactly the
// single 200 ms hitch a player notices.
//
// Always on. A build event is rare (hundreds per launch, not per frame) and
// costs two clock reads, so there is no gate to forget to set — and a baseline
// nobody can reproduce without special flags is not a baseline.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

#include "third_party/fmt/include/fmt/format.h"

namespace xe {
namespace gpu {
namespace metal {

class MetalShaderBuildCensus {
 public:
  struct Stage {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> ns{0};
    std::atomic<uint64_t> max_ns{0};
    std::atomic<uint64_t> bytes{0};

    void Add(uint64_t elapsed_ns, uint64_t produced_bytes = 0) {
      count.fetch_add(1, std::memory_order_relaxed);
      ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
      bytes.fetch_add(produced_bytes, std::memory_order_relaxed);
      uint64_t previous = max_ns.load(std::memory_order_relaxed);
      while (elapsed_ns > previous &&
             !max_ns.compare_exchange_weak(previous, elapsed_ns,
                                           std::memory_order_relaxed)) {
      }
    }
    double total_ms() const {
      return double(ns.load(std::memory_order_relaxed)) / 1e6;
    }
    double max_ms() const {
      return double(max_ns.load(std::memory_order_relaxed)) / 1e6;
    }
    double mean_ms() const {
      const uint64_t n = count.load(std::memory_order_relaxed);
      return n ? total_ms() / double(n) : 0.0;
    }
  };

  Stage dxbc;
  Stage dxil;
  Stage msc;
  Stage function;
  Stage pso;
  Stage geometry_pso;

  // Durable-store outcomes. Hits and misses are kept apart from the stage
  // counters above because a hit is the ABSENCE of stage cost, and folding
  // them together would let a cache that never worked look like a cheap one.
  std::atomic<uint64_t> metallib_hits{0};
  std::atomic<uint64_t> metallib_misses{0};
  std::atomic<uint64_t> metallib_load_ns{0};
  std::atomic<uint64_t> metallib_stores{0};
  std::atomic<uint64_t> metallib_store_ns{0};
  std::atomic<uint64_t> metallib_store_bytes{0};
  // A hit whose blob the device refused. Not a miss: a miss is cold, this is a
  // stored artifact that is wrong, and the two must never share a counter.
  std::atomic<uint64_t> metallib_rejected{0};

  // Frame attribution.
  std::atomic<uint64_t> swaps{0};
  std::atomic<uint64_t> startup_build_ns{0};    // before the first swap
  std::atomic<uint64_t> stall_build_ns{0};      // after it
  std::atomic<uint64_t> stall_events{0};
  std::atomic<uint64_t> stall_frames{0};        // frames with any build work
  std::atomic<uint64_t> worst_frame_ns{0};
  std::atomic<uint64_t> frames_over_8ms{0};
  std::atomic<uint64_t> frames_over_16ms{0};
  std::atomic<uint64_t> frames_over_33ms{0};
  // Wall clock from census construction (process start, near enough — it is
  // constructed on first build event or first swap, whichever comes first) to
  // the first swap. "Total startup time" in the 1.1 sense.
  std::atomic<uint64_t> first_swap_ns{0};

  MetalShaderBuildCensus() : origin_(std::chrono::steady_clock::now()) {}

  // Every build event routes through here so frame attribution cannot drift
  // from the per-stage totals.
  void RecordBuild(Stage& stage, uint64_t elapsed_ns, uint64_t bytes = 0) {
    stage.Add(elapsed_ns, bytes);
    if (swaps.load(std::memory_order_relaxed) == 0) {
      startup_build_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
      return;
    }
    stall_build_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    stall_events.fetch_add(1, std::memory_order_relaxed);
    frame_build_ns_.fetch_add(elapsed_ns, std::memory_order_relaxed);
  }

  // Called once per presented frame. Closes the current frame's accumulator:
  // per-frame build cost is what a player experiences, and it is not
  // recoverable from a total.
  void NoteSwap() {
    const uint64_t previous_swaps = swaps.fetch_add(1, std::memory_order_relaxed);
    if (previous_swaps == 0) {
      first_swap_ns.store(uint64_t(std::chrono::duration_cast<
                                       std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() -
                                       origin_)
                                       .count()),
                          std::memory_order_relaxed);
    }
    const uint64_t frame_ns = frame_build_ns_.exchange(0, std::memory_order_relaxed);
    if (!frame_ns) {
      return;
    }
    stall_frames.fetch_add(1, std::memory_order_relaxed);
    uint64_t worst = worst_frame_ns.load(std::memory_order_relaxed);
    while (frame_ns > worst &&
           !worst_frame_ns.compare_exchange_weak(worst, frame_ns,
                                                 std::memory_order_relaxed)) {
    }
    if (frame_ns > 8000000ull) frames_over_8ms.fetch_add(1, std::memory_order_relaxed);
    if (frame_ns > 16000000ull) frames_over_16ms.fetch_add(1, std::memory_order_relaxed);
    if (frame_ns > 33000000ull) frames_over_33ms.fetch_add(1, std::memory_order_relaxed);
  }

  // One grep-able line per report. `phase` names why it was emitted, because a
  // periodic sample and a shutdown total are not interchangeable numbers.
  std::string Report(const char* phase) const {
    const uint64_t hits = metallib_hits.load(std::memory_order_relaxed);
    const uint64_t misses = metallib_misses.load(std::memory_order_relaxed);
    const uint64_t lookups = hits + misses;
    return fmt::format(
        "MetalShaderCensus|phase={}|swaps={}|first_swap_ms={:.1f}"
        "|dxbc={}/{:.1f}ms|dxil={}/{:.1f}ms/max{:.1f}|msc={}/{:.1f}ms/max{:.1f}"
        "|fn={}/{:.1f}ms|pso={}/{:.1f}ms/max{:.1f}|gpso={}/{:.1f}ms"
        "|mlib_hit={}|mlib_miss={}|mlib_hitrate={:.1f}%|mlib_rejected={}"
        "|mlib_load_ms={:.1f}|mlib_store={}/{:.1f}ms/{}KiB"
        "|startup_build_ms={:.1f}|stall_build_ms={:.1f}|stall_events={}"
        "|stall_frames={}|worst_frame_ms={:.1f}|f>8ms={}|f>16ms={}|f>33ms={}",
        phase, swaps.load(std::memory_order_relaxed),
        double(first_swap_ns.load(std::memory_order_relaxed)) / 1e6,
        dxbc.count.load(std::memory_order_relaxed), dxbc.total_ms(),
        dxil.count.load(std::memory_order_relaxed), dxil.total_ms(),
        dxil.max_ms(),
        msc.count.load(std::memory_order_relaxed), msc.total_ms(), msc.max_ms(),
        function.count.load(std::memory_order_relaxed), function.total_ms(),
        pso.count.load(std::memory_order_relaxed), pso.total_ms(), pso.max_ms(),
        geometry_pso.count.load(std::memory_order_relaxed),
        geometry_pso.total_ms(),
        hits, misses,
        lookups ? 100.0 * double(hits) / double(lookups) : 0.0,
        metallib_rejected.load(std::memory_order_relaxed),
        double(metallib_load_ns.load(std::memory_order_relaxed)) / 1e6,
        metallib_stores.load(std::memory_order_relaxed),
        double(metallib_store_ns.load(std::memory_order_relaxed)) / 1e6,
        metallib_store_bytes.load(std::memory_order_relaxed) / 1024,
        double(startup_build_ns.load(std::memory_order_relaxed)) / 1e6,
        double(stall_build_ns.load(std::memory_order_relaxed)) / 1e6,
        stall_events.load(std::memory_order_relaxed),
        stall_frames.load(std::memory_order_relaxed),
        double(worst_frame_ns.load(std::memory_order_relaxed)) / 1e6,
        frames_over_8ms.load(std::memory_order_relaxed),
        frames_over_16ms.load(std::memory_order_relaxed),
        frames_over_33ms.load(std::memory_order_relaxed));
  }

 private:
  std::chrono::steady_clock::time_point origin_;
  std::atomic<uint64_t> frame_build_ns_{0};
};

inline MetalShaderBuildCensus& GetMetalShaderBuildCensus() {
  static MetalShaderBuildCensus census;
  return census;
}

// Times a scoped region into a stage. `bytes` is filled in by the caller after
// the work, so the artifact size travels with its cost.
class MetalShaderBuildTimer {
 public:
  explicit MetalShaderBuildTimer(MetalShaderBuildCensus::Stage& stage)
      : stage_(stage), start_(std::chrono::steady_clock::now()) {}
  ~MetalShaderBuildTimer() {
    if (dismissed_) {
      return;
    }
    const uint64_t elapsed = uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_)
            .count());
    GetMetalShaderBuildCensus().RecordBuild(stage_, elapsed, bytes_);
  }

  void set_bytes(uint64_t bytes) { bytes_ = bytes; }
  // A failed build is not a build. Dismissing keeps failure cost out of the
  // per-stage means without hiding it — the failure is already logged loudly
  // at the call site.
  void Dismiss() { dismissed_ = true; }

 private:
  MetalShaderBuildCensus::Stage& stage_;
  std::chrono::steady_clock::time_point start_;
  uint64_t bytes_ = 0;
  bool dismissed_ = false;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHADER_BUILD_CENSUS_H_
