/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_PIPELINE_CENSUS_H_
#define XENIA_GPU_METAL_METAL_PIPELINE_CENSUS_H_

// TEMPORARY DIAGNOSTIC — task group 1 of
// openspec/changes/implement-metal-shader-pipeline-lifetime.
//
// Purpose: answer "how often does pipeline creation actually stall a frame, and
// in which titles" BEFORE any durable cache or creation thread is written. The
// change's D1 decision permits a recorded negative as the outcome.
//
// REMOVAL CONDITION (task 8.1): this whole file, its five Scope call sites in
// metal_pipeline_cache.mm, and the two call sites in metal_command_processor.mm
// are deleted before any mechanism is promoted. It is header-only and appears in
// no CMakeLists, so removal is deleting the file plus those seven lines.
//
// Contract (spec: "Diagnostic instrumentation is temporary and default-off"):
// default-off; when enabled it only reads a clock around calls that already
// happen, so the set of pipelines created and the frames they are created on are
// identical with it on or off.
//
// Enable:  XE_METAL_PIPELINE_CENSUS=1
// Budget:  XE_METAL_PIPELINE_CENSUS_BUDGET_MS=<float>   (default 16.67)
//
// Reports:
//   - a running line every 128 swaps (a live title is usually killed rather than
//     shut down cleanly, so the periodic line is the primary evidence);
//   - a full per-run report at MetalPipelineCache::Shutdown.

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
namespace pipeline_census {

// The five creation sites that can block a draw. They are separated because
// they are not cured by the same mechanism: an MTLBinaryArchive covers host
// pipeline-state creation, while the translation chain (DXBC -> DXIL -> MSC ->
// metallib) would need a separate durable shader store. Reporting one blended
// number would make the go/no-go decision on the wrong mechanism.
enum class Event : uint32_t {
  kShaderTranslate = 0,  // ucode -> DXBC -> DXIL -> metallib -> MTLFunction
  kRenderPipeline,       // newRenderPipelineStateWithDescriptor:error:
  kGeometryVertexStage,  // guest VS -> MSC object stage (+ stage-in)
  kGeometryShaderStage,  // generated DXBC GS -> MSC mesh stage
  kGeometryPipeline,     // IRRuntimeNewGeometryEmulationPipeline
  kCount,
};

inline const char* EventName(Event event) {
  switch (event) {
    case Event::kShaderTranslate:
      return "shader-translate";
    case Event::kRenderPipeline:
      return "render-pipeline";
    case Event::kGeometryVertexStage:
      return "geometry-vs-stage";
    case Event::kGeometryShaderStage:
      return "geometry-gs-stage";
    case Event::kGeometryPipeline:
      return "geometry-pipeline";
    default:
      return "unknown";
  }
}

struct EventStats {
  uint64_t count = 0;
  uint64_t failed = 0;
  double total_ms = 0.0;
  double max_ms = 0.0;
};

// Cold-start buckets (task 1.4): how much of a run's creation time lands in the
// first N frames, which is exactly what a durable cache would remove.
inline constexpr std::array<uint32_t, 6> kColdStartFrames = {1,  2,  5,
                                                             30, 60, 300};

// Stall thresholds reported side by side, so the reader can apply the one that
// matches the run's measured frame interval instead of assuming 60 fps.
inline constexpr std::array<double, 4> kThresholds = {5.0, 1000.0 / 60.0, 33.3,
                                                      100.0};

class State {
 public:
  static State& Get() {
    static State instance;
    return instance;
  }

  bool enabled() const { return enabled_; }
  double budget_ms() const { return budget_ms_; }

  void Record(Event event, double ms, bool succeeded) {
    std::lock_guard<std::mutex> lock(mutex_);
    EventStats& stats = events_[static_cast<uint32_t>(event)];
    ++stats.count;
    if (!succeeded) {
      ++stats.failed;
    }
    stats.total_ms += ms;
    stats.max_ms = std::max(stats.max_ms, ms);
    if (frame_ms_.size() <= frame_index_) {
      frame_ms_.resize(frame_index_ + 1, 0.0);
      frame_count_.resize(frame_index_ + 1, 0u);
    }
    frame_ms_[frame_index_] += ms;
    ++frame_count_[frame_index_];
  }

  // Called at IssueSwap. Frame 0 is everything before the first swap, i.e. boot
  // and the first presented frame's setup — the cold-start bucket that a durable
  // cache is supposed to remove.
  void AdvanceFrame() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_index_ < kMaxFrames) {
      ++frame_index_;
    }
    if ((frame_index_ & 127u) == 0u && frame_index_) {
      LogRunningLocked();
    }
    // A live title is normally killed, so Shutdown's report often never runs.
    // Emit the full one periodically as well — otherwise the worst-frame list
    // and the cold-start split, which are the actual task 1.2/1.4 evidence, are
    // lost on every live run.
    if ((frame_index_ % 512u) == 0u && frame_index_) {
      ReportLocked("periodic");
    }
  }

  void Report(const char* reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReportLocked(reason);
  }

 private:
  void ReportLocked(const char* reason) {
    double run_total_ms = 0.0;
    uint64_t run_total_count = 0;
    for (const EventStats& stats : events_) {
      run_total_ms += stats.total_ms;
      run_total_count += stats.count;
    }
    XELOGI(
        "MetalPipelineCensus REPORT ({}): frames={}, creations={}, "
        "total={:.2f} ms, budget={:.2f} ms",
        reason ? reason : "", frame_index_, run_total_count, run_total_ms,
        budget_ms_);
    if (!run_total_count) {
      XELOGI("MetalPipelineCensus   no creation events recorded");
      return;
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(Event::kCount); ++i) {
      const EventStats& stats = events_[i];
      if (!stats.count) {
        continue;
      }
      XELOGI(
          "MetalPipelineCensus   {:<18} n={:<6} failed={:<4} total={:9.2f} ms "
          "({:5.1f}%)  mean={:7.3f} ms  max={:8.3f} ms",
          EventName(static_cast<Event>(i)), stats.count, stats.failed,
          stats.total_ms, 100.0 * stats.total_ms / std::max(run_total_ms, 1e-9),
          stats.total_ms / double(stats.count), stats.max_ms);
    }

    // Per-frame distribution (task 1.2): which frames spent more than a frame
    // budget inside creation. This — not the run total — is the stall claim.
    //
    // Reported against several thresholds on purpose. A single 16.67 ms bar
    // silently assumes the backend is near 60 fps; where it is not, that bar
    // both over- and under-states the stall, so the reader picks the threshold
    // that matches the measured frame interval for the run.
    uint32_t frames_with_any = 0;
    std::array<uint32_t, kThresholds.size()> over = {};
    for (size_t f = 0; f < frame_ms_.size(); ++f) {
      if (frame_count_[f]) {
        ++frames_with_any;
      }
      for (size_t t = 0; t < kThresholds.size(); ++t) {
        if (frame_ms_[f] > kThresholds[t]) {
          ++over[t];
        }
      }
    }
    const double frames = std::max<uint32_t>(frame_index_, 1u);
    XELOGI("MetalPipelineCensus   frames with any creation={} ({:.2f}% of {})",
           frames_with_any, 100.0 * frames_with_any / frames, frame_index_);
    for (size_t t = 0; t < kThresholds.size(); ++t) {
      XELOGI(
          "MetalPipelineCensus   frames over {:6.2f} ms in creation: {:<6} "
          "({:.3f}%)",
          kThresholds[t], over[t], 100.0 * over[t] / frames);
    }

    // The worst frames by name, so a stall can be correlated with what the
    // title was doing (a transition is the hypothesised worst case, task 1.3).
    std::vector<uint32_t> worst;
    worst.reserve(frame_ms_.size());
    for (uint32_t f = 0; f < uint32_t(frame_ms_.size()); ++f) {
      if (frame_ms_[f] > 0.0) {
        worst.push_back(f);
      }
    }
    std::partial_sort(
        worst.begin(),
        worst.begin() + std::min<size_t>(worst.size(), kWorstFramesReported),
        worst.end(),
        [this](uint32_t a, uint32_t b) { return frame_ms_[a] > frame_ms_[b]; });
    const size_t reported =
        std::min<size_t>(worst.size(), kWorstFramesReported);
    for (size_t i = 0; i < reported; ++i) {
      const uint32_t f = worst[i];
      XELOGI("MetalPipelineCensus   worst frame#{:<6} {:8.3f} ms in {} creations",
             f, frame_ms_[f], frame_count_[f]);
    }

    // Cold start (task 1.4).
    for (uint32_t n : kColdStartFrames) {
      double sum = 0.0;
      uint64_t count = 0;
      for (uint32_t f = 0; f < n && f < frame_ms_.size(); ++f) {
        sum += frame_ms_[f];
        count += frame_count_[f];
      }
      XELOGI(
          "MetalPipelineCensus   first {:<4} frames: {:9.2f} ms ({:5.1f}% of "
          "run) in {} creations",
          n, sum, 100.0 * sum / std::max(run_total_ms, 1e-9), count);
    }
  }

 private:
  static constexpr uint32_t kMaxFrames = 1u << 20;
  static constexpr size_t kWorstFramesReported = 16;

  State() {
    const char* v = std::getenv("XE_METAL_PIPELINE_CENSUS");
    enabled_ = v && v[0] && v[0] != '0';
    const char* b = std::getenv("XE_METAL_PIPELINE_CENSUS_BUDGET_MS");
    budget_ms_ = b ? std::atof(b) : 0.0;
    if (!(budget_ms_ > 0.0)) {
      budget_ms_ = 1000.0 / 60.0;
    }
    if (enabled_) {
      XELOGI(
          "MetalPipelineCensus: ENABLED (diagnostic, frame budget {:.2f} ms) — "
          "creation set and timing of creation are unchanged",
          budget_ms_);
    }
  }

  void LogRunningLocked() {
    double run_total_ms = 0.0;
    uint64_t run_total_count = 0;
    for (const EventStats& stats : events_) {
      run_total_ms += stats.total_ms;
      run_total_count += stats.count;
    }
    uint32_t over_budget = 0;
    for (double ms : frame_ms_) {
      if (ms > budget_ms_) {
        ++over_budget;
      }
    }
    XELOGI(
        "MetalPipelineCensus swap#{}: {} creations, {:.2f} ms total "
        "(xlat {:.2f} / rpstate {:.2f} / geom {:.2f}), {} frames over "
        "{:.2f} ms",
        frame_index_, run_total_count, run_total_ms,
        events_[uint32_t(Event::kShaderTranslate)].total_ms,
        events_[uint32_t(Event::kRenderPipeline)].total_ms,
        events_[uint32_t(Event::kGeometryVertexStage)].total_ms +
            events_[uint32_t(Event::kGeometryShaderStage)].total_ms +
            events_[uint32_t(Event::kGeometryPipeline)].total_ms,
        over_budget, budget_ms_);
  }

  bool enabled_ = false;
  double budget_ms_ = 1000.0 / 60.0;
  std::mutex mutex_;
  std::array<EventStats, static_cast<size_t>(Event::kCount)> events_ = {};
  uint32_t frame_index_ = 0;
  std::vector<double> frame_ms_;
  std::vector<uint32_t> frame_count_;
};

inline bool Enabled() { return State::Get().enabled(); }

inline void AdvanceFrame() {
  if (Enabled()) {
    State::Get().AdvanceFrame();
  }
}

inline void Report(const char* reason) {
  if (Enabled()) {
    State::Get().Report(reason);
  }
}

// RAII timer around one creation site. Constructing it when the census is off
// costs a predicted-taken branch and nothing else; it never touches the clock.
class Scope {
 public:
  // `result_flag`, when given, is read at scope exit — for a function with many
  // early failure returns, point it at a local the success path sets.
  explicit Scope(Event event, const bool* result_flag = nullptr)
      : event_(event), enabled_(Enabled()), result_flag_(result_flag) {
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
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start_)
                          .count();
    const bool succeeded = result_flag_ ? *result_flag_ : succeeded_;
    State::Get().Record(event_, ms, succeeded);
  }

  // A failed creation still consumed wall time on the frame that wanted it, so
  // it is recorded — but separately, because a failure is not work a cache
  // would ever avoid.
  void set_failed() { succeeded_ = false; }

 private:
  Event event_;
  bool enabled_;
  const bool* result_flag_ = nullptr;
  bool succeeded_ = true;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace pipeline_census
}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_CENSUS_H_
