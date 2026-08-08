/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Device-free accounting tests for the seam-timing accumulator
// (optimize-native-metal-frametime task 3.1 contract). Synthetic ("fake
// clock") nanosecond values only -- no real timers, no Metal device, no
// assertions on wall-clock behavior.

#include <cstring>
#include <set>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_seam_timing.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

TEST_CASE("seam timing: boundary reasons are exhaustive, named, and unique",
          "[seamtiming]") {
  // No default reason exists by construction (the parameter is default-free);
  // this pins the other half: every enumerator below kCount has a distinct
  // non-null name, so a new reason cannot land without naming itself, and
  // kCount itself is not a nameable reason.
  std::set<std::string> names;
  for (uint32_t i = 0; i < uint32_t(SeamBoundaryReason::kCount); ++i) {
    const char* name = SeamBoundaryReasonName(SeamBoundaryReason(i));
    REQUIRE(name != nullptr);
    REQUIRE(name[0] != '\0');
    CAPTURE(i, name);
    REQUIRE(names.insert(name).second);
  }
  REQUIRE(SeamBoundaryReasonName(SeamBoundaryReason::kCount) == nullptr);
}

TEST_CASE("seam timing: consumer-kind count matches named kinds",
          "[seamtiming]") {
  // Pinned to the enum's own kCount sentinel. Every index below it must name
  // a REAL kind: non-null, non-empty, unique, and NOT the "?" fallback --
  // "?" would otherwise let an unnamed kind pass a naive non-null check.
  // The sentinel itself must map to the fallback, proving the loop bound is
  // exactly the named range.
  REQUIRE(kSeamConsumerKindCount == uint32_t(ConsumerKind::kCount));
  std::set<std::string> names;
  for (uint32_t i = 0; i < kSeamConsumerKindCount; ++i) {
    const char* name = ConsumerKindName(ConsumerKind(i));
    REQUIRE(name != nullptr);
    REQUIRE(name[0] != '\0');
    CAPTURE(i, name);
    REQUIRE(std::string(name) != "?");
    REQUIRE(names.insert(name).second);
  }
  CHECK(std::string(ConsumerKindName(ConsumerKind::kCount)) == "?");
}

TEST_CASE("seam timing: checked consumer indexing clamps out-of-range kinds",
          "[seamtiming]") {
  SeamTimingAccumulator acc;
  // The kCount sentinel (and anything past it) must clamp to the last real
  // slot rather than write past the array.
  auto& sentinel_slot = acc.ConsumerSlot(ConsumerKind::kCount);
  auto& last_real_slot =
      acc.consumer[SeamTimingAccumulator::kConsumerCount - 1];
  CHECK(&sentinel_slot == &last_real_slot);
  auto& real_slot = acc.ConsumerSlot(ConsumerKind::kTexture);
  CHECK(&real_slot == &acc.consumer[uint32_t(ConsumerKind::kTexture)]);
}

TEST_CASE("seam timing: stage totals record count, sum, and max",
          "[seamtiming]") {
  SeamStageTotals t;
  t.Record(100);
  t.Record(50);
  t.Record(300);
  CHECK(t.count == 3);
  CHECK(t.total_ns == 450);
  CHECK(t.max_ns == 300);
  t.Reset();
  CHECK(t.count == 0);
  CHECK(t.total_ns == 0);
  CHECK(t.max_ns == 0);
}

TEST_CASE("seam timing: fake-clock stage decomposition stays nested-separate",
          "[seamtiming]") {
  // Simulates one consumer request: inclusive 10ms containing a 2ms required
  // wait and a 3ms drain, plus the global boundary's 1ms close / 6ms gpu-wait.
  // The invariant under test is bookkeeping separation: the accumulator holds
  // inclusive and nested stages side by side and nothing sums an outer stage
  // with its contents.
  SeamTimingAccumulator acc;
  auto& r = acc.reason[uint32_t(SeamBoundaryReason::kConsumerDependency)];
  ++r.boundaries;
  r.close.Record(1'000'000);
  r.gpu_wait.Record(6'000'000);
  r.retire.Record(200'000);    // post-completion bookkeeping, its own stage
  r.post_wait.Record(10'000);  // pre-satisfied wrapper wait: near-zero
  auto& c = acc.consumer[uint32_t(ConsumerKind::kVertex)];
  ++c.consultations;
  ++c.non_empty;
  ++c.cb_present_commits;
  ++c.wait_requested;
  ++c.wait_blocked;
  c.required_wait.Record(2'000'000);
  c.drain.Record(3'000'000);
  c.inclusive.Record(10'000'000);
  ++c.completed;
  ++c.binding[uint32_t(SeamBindingTerm::kBoundary)];

  // Nested stages retain their own values; inclusive retains its own; no
  // field aggregates another.
  CHECK(c.required_wait.total_ns == 2'000'000);
  CHECK(c.drain.total_ns == 3'000'000);
  CHECK(c.inclusive.total_ns == 10'000'000);
  CHECK(c.inclusive.total_ns >=
        c.required_wait.total_ns + c.drain.total_ns);  // sanity of the fake
  CHECK(r.gpu_wait.total_ns == 6'000'000);
  CHECK(r.close.total_ns == 1'000'000);
  CHECK(r.retire.total_ns == 200'000);

  CHECK(acc.AnyActivity());
  acc.Reset();
  CHECK_FALSE(acc.AnyActivity());
  CHECK(c.inclusive.count == 0);
  CHECK(r.gpu_wait.count == 0);
  CHECK(c.binding[uint32_t(SeamBindingTerm::kBoundary)] == 0);
}

TEST_CASE("seam timing: empty early-outs and noop closes stay apart from work",
          "[seamtiming]") {
  SeamTimingAccumulator acc;
  auto& c = acc.consumer[uint32_t(ConsumerKind::kHostCpuRead)];
  c.consultations = 5;
  c.empty_early_outs = 4;
  c.non_empty = 1;
  auto& r = acc.reason[uint32_t(SeamBoundaryReason::kSwapPacing)];
  r.noop_calls = 3;
  // No boundaries recorded: noop closes alone must not read as boundaries.
  CHECK(r.boundaries == 0);
  CHECK(c.consultations - c.empty_early_outs == c.non_empty);
  CHECK(acc.AnyActivity());
}

TEST_CASE("seam timing: an inactive side formats to a valid EMPTY string",
          "[seamtiming]") {
  // Both buffers are always passed to the logger; a side with no activity
  // must yield "\0", never uninitialized stack memory. Poison the buffers to
  // prove the formatter writes the terminator itself.
  SeamTimingAccumulator acc;
  char buf[256];
  std::memset(buf, 0x7f, sizeof(buf));
  size_t n = acc.FormatReasons(buf, sizeof(buf));
  CHECK(n == 0);
  REQUIRE(buf[0] == '\0');
  std::memset(buf, 0x7f, sizeof(buf));
  n = acc.FormatConsumers(buf, sizeof(buf));
  CHECK(n == 0);
  REQUIRE(buf[0] == '\0');

  // One-sided activity: consumers active, reasons still empty-and-valid.
  auto& c = acc.consumer[uint32_t(ConsumerKind::kWait)];
  ++c.consultations;
  std::memset(buf, 0x7f, sizeof(buf));
  n = acc.FormatReasons(buf, sizeof(buf));
  CHECK(n == 0);
  REQUIRE(buf[0] == '\0');
  CHECK(acc.FormatConsumers(buf, sizeof(buf)) > 0);

  // And the mirror: reasons active, consumers empty-and-valid.
  SeamTimingAccumulator acc2;
  ++acc2.reason[uint32_t(SeamBoundaryReason::kSwapPacing)].boundaries;
  std::memset(buf, 0x7f, sizeof(buf));
  n = acc2.FormatConsumers(buf, sizeof(buf));
  CHECK(n == 0);
  REQUIRE(buf[0] == '\0');
  CHECK(acc2.FormatReasons(buf, sizeof(buf)) > 0);
}

TEST_CASE("seam timing: formatter emits only non-zero entries",
          "[seamtiming]") {
  SeamTimingAccumulator acc;
  char buf[2048];
  // Empty accumulator formats to nothing.
  size_t n = acc.FormatReasons(buf, sizeof(buf));
  CHECK(n == 0);
  n = acc.FormatConsumers(buf, sizeof(buf));
  CHECK(n == 0);

  auto& r = acc.reason[uint32_t(SeamBoundaryReason::kResolveSizeSplit)];
  ++r.boundaries;
  r.close.Record(1'500'000);   // 1500us
  r.gpu_wait.Record(4'000'000);  // 4000us
  n = acc.FormatReasons(buf, sizeof(buf));
  REQUIRE(n > 0);
  std::string reasons(buf);
  CHECK(reasons.find("resolve-size-split=1") != std::string::npos);
  CHECK(reasons.find("close 1500us") != std::string::npos);
  CHECK(reasons.find("gpuwait 4000us") != std::string::npos);
  // Untouched reasons are absent -- the line stays bounded.
  CHECK(reasons.find("swap-pacing") == std::string::npos);
  CHECK(reasons.find("consumer-dependency") == std::string::npos);

  auto& c = acc.consumer[uint32_t(ConsumerKind::kMemexportDestination)];
  ++c.consultations;
  ++c.non_empty;
  c.selected_memexport = 2;
  c.drained_records = 2;
  c.copied_bytes = 4096;
  c.inclusive.Record(7'000'000);
  ++c.completed;
  c.binding[uint32_t(SeamBindingTerm::kZero)] = 2;
  n = acc.FormatConsumers(buf, sizeof(buf));
  REQUIRE(n > 0);
  std::string consumers(buf);
  CHECK(consumers.find("sel r0/m2") != std::string::npos);
  CHECK(consumers.find("bytes 4096") != std::string::npos);
  CHECK(consumers.find("incl 7000us") != std::string::npos);
  // The both-terms-zero binding bucket is distinct from tie in the output.
  CHECK(consumers.find("bind b0/p0/t0/z2") != std::string::npos);
  CHECK(consumers.find("texture") == std::string::npos);
}

TEST_CASE("seam timing: formatter clamps to caller storage without overflow",
          "[seamtiming]") {
  SeamTimingAccumulator acc;
  for (uint32_t i = 0; i < uint32_t(SeamBoundaryReason::kCount); ++i) {
    auto& r = acc.reason[i];
    r.boundaries = 1000 + i;
    r.noop_calls = i;
    r.close.Record(123'456'789);
    r.gpu_wait.Record(987'654'321);
    r.post_wait.Record(11'111);
    r.post_wait_blocked = i;
  }
  char small[64];
  size_t n = acc.FormatReasons(small, sizeof(small));
  CHECK(n <= sizeof(small));
  // NUL-terminated even when clamped.
  CHECK(std::strlen(small) < sizeof(small));
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
