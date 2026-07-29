/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Direct coverage of the PRODUCTION reporting policy for memexport producer
// classes this backend does not route.
//
// This exercises `MemexportGapReporter` itself -- the same type
// `MetalCommandProcessor` holds and calls -- not a test-only replica. The
// policy is deliberately free of logging and Metal types so it is testable
// without a device.
//
// The behavioral contract under test:
//   - detection is unconditional; no switch controls it
//   - the first sighting of a (class, guest range) reports
//   - an identical sighting is suppressed
//   - distinct sightings report up to a hard cap
//   - exceeding the cap emits EXACTLY ONE suppression summary
//   - everything after that stays suppressed
//   - producer classes are distinguishable from one another
//   - reset begins a fresh session
//
// Build-presence of a log string is NOT the acceptance gate; this is.

#include <cstdint>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memexport_gap_report.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

using Action = MemexportGapReporter::Action;
constexpr uint32_t kCap = MemexportGapReporter::kMaxDistinctReports;

}  // namespace

TEST_CASE("First unique dropped range reports", "[metal][memexport][gap]") {
  MemexportGapReporter r;
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, 0x11BE1000 >> 2,
                    2097152) == Action::kReport);
  REQUIRE(r.distinct_reported() == 1);
  REQUIRE(r.suppressed() == 0);
}

TEST_CASE("An identical range is suppressed", "[metal][memexport][gap]") {
  MemexportGapReporter r;
  const uint32_t base = 0x11BE1000 >> 2;
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, base,
                    2097152) == Action::kReport);
  // Same class, same guest range -- one finding, not two.
  for (int i = 0; i < 100; ++i) {
    REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, base,
                      2097152) == Action::kSuppressed);
  }
  REQUIRE(r.distinct_reported() == 1);
  REQUIRE(r.suppressed() == 100);
  REQUIRE_FALSE(r.summary_emitted());
}

TEST_CASE("Distinct ranges report through the configured cap",
          "[metal][memexport][gap]") {
  MemexportGapReporter r;
  for (uint32_t i = 0; i < kCap; ++i) {
    REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, 0x1000 + i,
                      256) == Action::kReport);
  }
  REQUIRE(r.distinct_reported() == kCap);
  REQUIRE_FALSE(r.summary_emitted());
}

TEST_CASE("Exceeding the cap produces exactly one suppression summary",
          "[metal][memexport][gap]") {
  MemexportGapReporter r;
  for (uint32_t i = 0; i < kCap; ++i) {
    REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, 0x1000 + i,
                      256) == Action::kReport);
  }
  // The first distinct sighting past the cap announces suppression, once.
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, 0x2000, 256) ==
          Action::kEmitSuppressionSummary);

  // Every further distinct sighting stays quiet -- no second summary.
  int summaries = 0;
  for (uint32_t i = 0; i < 200; ++i) {
    if (r.Observe(MemexportProducerClass::kRasterizingVertex, 0x3000 + i,
                  256) == Action::kEmitSuppressionSummary) {
      ++summaries;
    }
  }
  REQUIRE(summaries == 0);
  REQUIRE(r.summary_emitted());
  REQUIRE(r.distinct_reported() == kCap);
}

TEST_CASE("Each producer classification is distinguishable",
          "[metal][memexport][gap]") {
  MemexportGapReporter r;
  const uint32_t base = 0x11BE1000 >> 2;
  // The SAME guest range reached by a different producer class is a different
  // finding: the classes need different work to support.
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, base, 4096) ==
          Action::kReport);
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertexGeometry, base,
                    4096) == Action::kReport);
  REQUIRE(r.Observe(MemexportProducerClass::kPixelStage, base, 4096) ==
          Action::kReport);
  REQUIRE(r.distinct_reported() == 3);

  // And each remains individually deduplicated.
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertexGeometry, base,
                    4096) == Action::kSuppressed);
  REQUIRE(r.distinct_reported() == 3);
}

TEST_CASE("Size participates in the dedup key", "[metal][memexport][gap]") {
  MemexportGapReporter r;
  const uint32_t base = 0x4000;
  REQUIRE(r.Observe(MemexportProducerClass::kPixelStage, base, 256) ==
          Action::kReport);
  REQUIRE(r.Observe(MemexportProducerClass::kPixelStage, base, 512) ==
          Action::kReport);
  REQUIRE(r.distinct_reported() == 2);
}

TEST_CASE("Reset begins a fresh reporting session", "[metal][memexport][gap]") {
  MemexportGapReporter r;
  for (uint32_t i = 0; i < kCap + 10; ++i) {
    r.Observe(MemexportProducerClass::kRasterizingVertex, 0x1000 + i, 256);
  }
  REQUIRE(r.summary_emitted());
  REQUIRE(r.distinct_reported() == kCap);

  r.Reset();

  REQUIRE(r.distinct_reported() == 0);
  REQUIRE(r.suppressed() == 0);
  REQUIRE_FALSE(r.summary_emitted());
  // A range reported in the previous session reports again in the new one --
  // otherwise a defect present across two titles would be invisible in the
  // second.
  REQUIRE(r.Observe(MemexportProducerClass::kRasterizingVertex, 0x1000, 256) ==
          Action::kReport);
}

TEST_CASE("No diagnostic switch controls reporting", "[metal][memexport][gap]") {
  // The policy type has no environment or cvar input by construction: a fresh
  // reporter reports regardless of process configuration. This is the property
  // that lets the condition be discovered by someone who does not already
  // suspect memexport.
  MemexportGapReporter r;
  REQUIRE(r.Observe(MemexportProducerClass::kPixelStage, 0x9000, 128) ==
          Action::kReport);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
