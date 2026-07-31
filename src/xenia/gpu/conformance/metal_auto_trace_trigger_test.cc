/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Direct coverage of the PRODUCTION auto-trace arming policy -- the same type
// MetalCommandProcessor evaluates at swap, not a test-only replica.
//
// The behavioral contract under test:
//   - disabled config never fires, whatever the inputs
//   - a GPU fault fires and outranks the skip condition in the reason token
//   - actionable skips fire at >= threshold; threshold 0 disables the skip
//     condition entirely (fault-only arming)
//   - the FIRST fire is never delayed by cooldown; later fires are suppressed
//     until cooldown_frames have elapsed, then allowed again
//   - max_captures is a hard session budget; 0 means never fire
//
// The caller-side rule (never arm while a trace is pending/streaming) lives
// in IssueSwap, not here -- this type only answers "would this frame fire".

#include <cstdint>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_auto_trace_trigger.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

MetalAutoTraceConfig EnabledConfig() {
  MetalAutoTraceConfig config;
  config.enabled = true;
  config.skip_threshold = 1;
  config.on_gpu_fault = true;
  config.cooldown_frames = 10;
  config.max_captures = 2;
  return config;
}

TEST_CASE("disabled trigger never fires", "[auto_trace]") {
  MetalAutoTraceConfig config = EnabledConfig();
  config.enabled = false;
  MetalAutoTraceTrigger trigger(config);
  CHECK_FALSE(trigger.Evaluate(1, 100, 5).fire);
  CHECK(trigger.fired_count() == 0);
}

TEST_CASE("gpu fault fires and outranks skips", "[auto_trace]") {
  MetalAutoTraceTrigger trigger(EnabledConfig());
  auto decision = trigger.Evaluate(1, /*actionable_skips=*/50,
                                   /*gpu_faults=*/1);
  REQUIRE(decision.fire);
  CHECK(std::string(decision.reason) == "gpu-fault");
}

TEST_CASE("skip threshold fires at >= and respects threshold 0",
          "[auto_trace]") {
  MetalAutoTraceConfig config = EnabledConfig();
  config.skip_threshold = 5;
  MetalAutoTraceTrigger trigger(config);
  CHECK_FALSE(trigger.Evaluate(1, 4, 0).fire);
  auto decision = trigger.Evaluate(2, 5, 0);
  REQUIRE(decision.fire);
  CHECK(std::string(decision.reason) == "skip-threshold");

  config.skip_threshold = 0;  // fault-only arming
  MetalAutoTraceTrigger fault_only(config);
  CHECK_FALSE(fault_only.Evaluate(1, 1000, 0).fire);
  CHECK(fault_only.Evaluate(2, 0, 1).fire);
}

TEST_CASE("first fire is immediate, cooldown gates the second",
          "[auto_trace]") {
  MetalAutoTraceConfig config = EnabledConfig();
  config.max_captures = 5;
  MetalAutoTraceTrigger trigger(config);
  // Fires on the very first frame despite cooldown_frames = 10.
  CHECK(trigger.Evaluate(1, 1, 0).fire);
  // Suppressed until 10 frames after the fire...
  CHECK_FALSE(trigger.Evaluate(5, 1, 0).fire);
  CHECK_FALSE(trigger.Evaluate(10, 1, 0).fire);
  // ...and allowed again at exactly cooldown distance.
  CHECK(trigger.Evaluate(11, 1, 0).fire);
  CHECK(trigger.fired_count() == 2);
}

TEST_CASE("max_captures is a hard budget", "[auto_trace]") {
  MetalAutoTraceTrigger trigger(EnabledConfig());  // budget 2, cooldown 10
  CHECK(trigger.Evaluate(1, 1, 0).fire);
  CHECK(trigger.Evaluate(20, 1, 0).fire);
  // Budget exhausted: even a GPU fault no longer fires.
  CHECK_FALSE(trigger.Evaluate(100, 1, 1).fire);
  CHECK(trigger.fired_count() == 2);

  MetalAutoTraceConfig config = EnabledConfig();
  config.max_captures = 0;
  MetalAutoTraceTrigger never(config);
  CHECK_FALSE(never.Evaluate(1, 1, 1).fire);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
