/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_H_
#define XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_H_

// TEMPORARY DIAGNOSTIC — task 1.1 of
// openspec/changes/complete-metal-memexport-producers.
//
// Classifies every draw whose shader writes eM into producer strata, so scope
// follows measured engagement rather than the parity ledger.
//
// STRATA — the control is deliberately separated from the gate classes:
//
//   CONTROL_VS_NORT   VS export, no render target. This is the PROMOTED route
//                     and JC2's known 0x11BE1000 producer. It exists here ONLY
//                     to prove the probe sees what it should. It MUST NOT
//                     satisfy the GO gate for any other stratum.
//
//   GATE_VS_RASTER    VS export with a bound RT, NOT geometry-emulated.
//                     Currently collected then DROPPED.
//   GATE_VS_RASTER_GEOM  Same, but on a geometry-emulated primitive. Reported
//                     SEPARATELY even though semantically part of rasterizing
//                     VS export: if only this stratum engages, the generated
//                     object/mesh path needs its own implementation and
//                     acceptance slice, not a shared one.
//   GATE_PS           Pixel-stage export. Currently refused outright.
//
// The probe only reads state the draw path already computed; it does not alter
// which draws execute.
//
// REMOVAL CONDITION (task 4.1): this file and its call site are deleted at
// closeout regardless of outcome. Header-only, in no CMakeLists.
//
// Enable: XE_METAL_MEMEXPORT_CENSUS=1

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <set>
#include <utility>

#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {
namespace memexport_census {

enum class Stratum : uint32_t {
  kControlVsNoRt = 0,      // promoted route — CONTROL ONLY
  kGateVsRaster,           // rasterizing VS export, not geometry-emulated
  kGateVsRasterGeom,       // rasterizing VS export on a geometry-emulated prim
  kGatePs,                 // pixel-stage export
  kCount,
};

inline const char* StratumName(Stratum s) {
  switch (s) {
    case Stratum::kControlVsNoRt:
      return "CONTROL_VS_NORT";
    case Stratum::kGateVsRaster:
      return "GATE_VS_RASTER";
    case Stratum::kGateVsRasterGeom:
      return "GATE_VS_RASTER_GEOM";
    case Stratum::kGatePs:
      return "GATE_PS";
    default:
      return "?";
  }
}

inline bool IsControl(Stratum s) { return s == Stratum::kControlVsNoRt; }

class State {
 public:
  static State& Get() {
    static State instance;
    return instance;
  }
  bool enabled() const { return enabled_; }

  void Record(Stratum s, uint32_t base_dwords, uint32_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t i = uint32_t(s);
    ++draws_[i];
    bytes_[i] += size_bytes;
    if (size_bytes) {
      ranges_[i].insert({base_dwords, size_bytes});
    }
  }

  void RecordDrawOnly(Stratum s) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++draws_[uint32_t(s)];
  }

  void AdvanceFrame() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++frames_;
    // 64, not 512: a trace replay presents a single frame, so a long cadence
    // reports nothing at all. Live runs still get periodic output.
    if ((frames_ % 64u) == 0u) {
      ReportLocked();
    }
  }

  void Report() {
    std::lock_guard<std::mutex> lock(mutex_);
    ReportLocked();
  }

 private:
  State() {
    const char* v = std::getenv("XE_METAL_MEMEXPORT_CENSUS");
    enabled_ = v && v[0] && v[0] != '0';
    if (enabled_) {
      XELOGI(
          "MetalMemexportCensus: ENABLED (classification only; does not alter "
          "which draws execute)");
    }
  }

  void ReportLocked() {
    XELOGI("MetalMemexportCensus REPORT frames={}", frames_);
    for (uint32_t i = 0; i < uint32_t(Stratum::kCount); ++i) {
      const Stratum s = Stratum(i);
      if (!draws_[i]) {
        continue;
      }
      XELOGI(
          "MetalMemexportCensus   {:<20} draws={:<8} ranges={:<5} bytes={:<12}"
          "{}",
          StratumName(s), draws_[i], ranges_[i].size(), bytes_[i],
          IsControl(s) ? "  [CONTROL — must NOT satisfy any GO gate]" : "");
      // Range bases matter: a stratum engaged by one recurring destination is a
      // different proposition from one scattered across many.
      uint32_t shown = 0;
      for (const auto& r : ranges_[i]) {
        if (shown++ >= 4) {
          break;
        }
        XELOGI("MetalMemexportCensus       base=0x{:08X} size={}",
               r.first << 2, r.second);
      }
    }
    const uint64_t gate_draws = draws_[uint32_t(Stratum::kGateVsRaster)] +
                                draws_[uint32_t(Stratum::kGateVsRasterGeom)] +
                                draws_[uint32_t(Stratum::kGatePs)];
    XELOGI(
        "MetalMemexportCensus   GATE STRATA TOTAL draws={} (control excluded)",
        gate_draws);
  }

  bool enabled_ = false;
  std::mutex mutex_;
  std::array<uint64_t, 4> draws_ = {};
  std::array<uint64_t, 4> bytes_ = {};
  std::array<std::set<std::pair<uint32_t, uint32_t>>, 4> ranges_;
  uint64_t frames_ = 0;
};

inline bool Enabled() { return State::Get().enabled(); }
inline void Record(Stratum s, uint32_t base_dwords, uint32_t size_bytes) {
  if (Enabled()) {
    State::Get().Record(s, base_dwords, size_bytes);
  }
}
inline void RecordDrawOnly(Stratum s) {
  if (Enabled()) {
    State::Get().RecordDrawOnly(s);
  }
}
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

}  // namespace memexport_census
}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_MEMEXPORT_CENSUS_H_
