/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Direct coverage of the PRODUCTION skip-classification policy behind
// MetalCommandProcessor::RecordDrawSkip -- the same type the command processor
// holds, not a test-only replica.
//
// The behavioral contract under test:
//   - every reason has a fixed severity; kInfo reasons are never actionable
//   - the first sighting of a site reports (one-shot log cue), repeats do not
//   - a record freezes draw index and primitive/shader identity at FIRST
//     occurrence and only its count moves afterwards
//   - frame scope resets at EndFrame; session scope survives it
//   - frame_actionable_count() counts warning+error only -- the trace-trigger
//     arming condition must never fire on expected pruning
//   - the actionable-only report omits kInfo sites; the full report keeps them
//   - ResetSession() begins a fresh session
//
// Build-presence of a log string is NOT the acceptance gate; this is.

#include <cstdint>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_draw_skip_accounting.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

TEST_CASE("severity mapping is fixed and info is never actionable",
          "[draw_skip]") {
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kExpectedNoEffect) ==
        MetalDrawSkipSeverity::kInfo);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kDiagnosticRollback) ==
        MetalDrawSkipSeverity::kInfo);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kUnsupportedCapability) ==
        MetalDrawSkipSeverity::kWarning);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kInvalidGuestState) ==
        MetalDrawSkipSeverity::kWarning);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kTranslationFailure) ==
        MetalDrawSkipSeverity::kError);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kPipelineFailure) ==
        MetalDrawSkipSeverity::kError);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kResourceFailure) ==
        MetalDrawSkipSeverity::kError);
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kInternalInvariant) ==
        MetalDrawSkipSeverity::kError);
}

TEST_CASE("first sighting reports once and identity freezes", "[draw_skip]") {
  MetalDrawSkipAccounting accounting;
  CHECK(accounting.Record("pipeline-null",
                          MetalDrawSkipReason::kPipelineFailure,
                          /*draw_index=*/41, /*prim_type=*/4,
                          /*vs_hash=*/0x1111, /*ps_hash=*/0x2222));
  CHECK_FALSE(accounting.Record("pipeline-null",
                                MetalDrawSkipReason::kPipelineFailure,
                                /*draw_index=*/99, /*prim_type=*/6,
                                /*vs_hash=*/0x3333, /*ps_hash=*/0x4444));
  const auto& records = accounting.session_records();
  REQUIRE(records.count("pipeline-null") == 1);
  const auto& record = records.at("pipeline-null");
  CHECK(record.count == 2);
  CHECK(record.first_draw_index == 41);
  CHECK(record.first_prim_type == 4);
  CHECK(record.first_vs_hash == 0x1111);
  CHECK(record.first_ps_hash == 0x2222);
}

TEST_CASE("frame scope resets at EndFrame, session scope survives",
          "[draw_skip]") {
  MetalDrawSkipAccounting accounting;
  accounting.Record("prim-unmappable",
                    MetalDrawSkipReason::kUnsupportedCapability, 7, 11, 0, 0);
  CHECK(accounting.frame_total_count() == 1);
  CHECK(accounting.session_total_count() == 1);
  accounting.EndFrame();
  CHECK(accounting.frame_total_count() == 0);
  CHECK(accounting.frame_records().empty());
  CHECK(accounting.session_total_count() == 1);
  REQUIRE(accounting.session_records().count("prim-unmappable") == 1);
  // A new frame's sighting of a known site is not a first-in-session report.
  CHECK_FALSE(accounting.Record("prim-unmappable",
                                MetalDrawSkipReason::kUnsupportedCapability, 8,
                                11, 0, 0));
  CHECK(accounting.frame_total_count() == 1);
  CHECK(accounting.session_records().at("prim-unmappable").count == 2);
}

TEST_CASE("actionable count excludes expected pruning", "[draw_skip]") {
  MetalDrawSkipAccounting accounting;
  accounting.Record("empty-render-extent",
                    MetalDrawSkipReason::kExpectedNoEffect, 1, 4, 0, 0);
  accounting.Record("empty-render-extent",
                    MetalDrawSkipReason::kExpectedNoEffect, 2, 4, 0, 0);
  accounting.Record("rt-cache-update-nosmalldepth",
                    MetalDrawSkipReason::kDiagnosticRollback, 3, 4, 0, 0);
  CHECK(accounting.frame_total_count() == 3);
  CHECK(accounting.frame_actionable_count() == 0);
  accounting.Record("vs-translate", MetalDrawSkipReason::kTranslationFailure,
                    4, 4, 0xAA, 0);
  accounting.Record("constant-banks",
                    MetalDrawSkipReason::kUnsupportedCapability, 5, 4, 0, 0);
  CHECK(accounting.frame_total_count() == 5);
  CHECK(accounting.frame_actionable_count() == 2);
}

TEST_CASE("actionable-only report omits info sites, full report keeps them",
          "[draw_skip]") {
  MetalDrawSkipAccounting accounting;
  accounting.Record("no-bound-targets", MetalDrawSkipReason::kExpectedNoEffect,
                    1, 4, 0, 0);
  accounting.Record("pipeline-null", MetalDrawSkipReason::kPipelineFailure, 2,
                    4, 0xBEEF, 0xF00D);
  std::string actionable = accounting.FormatReport(/*frame_scope=*/true,
                                                   /*actionable_only=*/true);
  CHECK(actionable.find("pipeline-null") != std::string::npos);
  CHECK(actionable.find("pipeline-failure") != std::string::npos);
  CHECK(actionable.find("no-bound-targets") == std::string::npos);
  std::string full = accounting.FormatReport(/*frame_scope=*/true,
                                             /*actionable_only=*/false);
  CHECK(full.find("no-bound-targets") != std::string::npos);
  CHECK(full.find("expected-no-effect") != std::string::npos);
  // Count and first-draw fields appear in the record text.
  CHECK(full.find("=1 first@2") != std::string::npos);
}

TEST_CASE("tessellation reasons are distinct, correctly tiered, and named",
          "[draw_skip][tessellation]") {
  // port-xenios-tessellation task 4.7: an unsupported guest tessellation
  // shape is guest-behavior triage (warning, like other capability gaps); a
  // missing tessellator-tables buffer is an init invariant this backend
  // controls (error). Neither may collide with the generic reasons.
  CHECK(MetalDrawSkipSeverityFor(
            MetalDrawSkipReason::kTessellationUnsupportedShape) ==
        MetalDrawSkipSeverity::kWarning);
  CHECK(MetalDrawSkipSeverityFor(
            MetalDrawSkipReason::kTessellationResourceMissing) ==
        MetalDrawSkipSeverity::kError);
  CHECK(std::string(MetalDrawSkipReasonName(
            MetalDrawSkipReason::kTessellationUnsupportedShape)) ==
        "tessellation-unsupported-shape");
  CHECK(std::string(MetalDrawSkipReasonName(
            MetalDrawSkipReason::kTessellationResourceMissing)) ==
        "tessellation-resource-missing");
}

TEST_CASE("shader-pending skips are expected, not actionable",
          "[draw_skip]") {
  // A draw held back while its shader translates in the background is the
  // designed behavior of metal_async_shader_translation, so it must never arm
  // a trace trigger or count as an actionable skip.
  CHECK(MetalDrawSkipSeverityFor(MetalDrawSkipReason::kShaderPending) ==
        MetalDrawSkipSeverity::kInfo);
  CHECK(std::string(MetalDrawSkipReasonName(
            MetalDrawSkipReason::kShaderPending)) == "shader-pending");
  MetalDrawSkipAccounting accounting;
  accounting.Record("vs-translate", MetalDrawSkipReason::kShaderPending, 1, 4,
                    0xABCD, 0);
  CHECK(accounting.frame_actionable_count() == 0);
}

TEST_CASE("tessellation pipeline-failure repeats are quieted per site",
          "[draw_skip][tessellation]") {
  // Task 4.7 repeat-quieting shape: the first negative-cache-suppressed
  // tessellated draw is the one-shot log cue; a storm of repeats (many draws
  // suppressed by one failed creation) never re-reports, but stays counted
  // and actionable for the trace trigger.
  MetalDrawSkipAccounting accounting;
  CHECK(accounting.Record("tess-pipeline-null",
                          MetalDrawSkipReason::kPipelineFailure,
                          /*draw_index=*/12, /*prim_type=*/17,
                          /*vs_hash=*/0xD0D0, /*ps_hash=*/0));
  for (uint32_t i = 0; i < 64; ++i) {
    CHECK_FALSE(accounting.Record("tess-pipeline-null",
                                  MetalDrawSkipReason::kPipelineFailure,
                                  13 + i, 17, 0xD0D0, 0));
  }
  const auto& record = accounting.session_records().at("tess-pipeline-null");
  CHECK(record.count == 65);
  CHECK(record.first_draw_index == 12);
  CHECK(accounting.frame_actionable_count() == 65);
  // A distinct tessellation site reports independently (per-site quieting,
  // not per-reason).
  CHECK(accounting.Record("tess-tables-missing",
                          MetalDrawSkipReason::kTessellationResourceMissing,
                          90, 17, 0xD0D0, 0));
}

TEST_CASE("ResetSession begins a fresh session", "[draw_skip]") {
  MetalDrawSkipAccounting accounting;
  accounting.Record("encoder-unavailable",
                    MetalDrawSkipReason::kResourceFailure, 1, 4, 0, 0);
  accounting.ResetSession();
  CHECK(accounting.session_total_count() == 0);
  CHECK(accounting.frame_total_count() == 0);
  CHECK(accounting.session_records().empty());
  CHECK(accounting.Record("encoder-unavailable",
                          MetalDrawSkipReason::kResourceFailure, 2, 4, 0, 0));
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
