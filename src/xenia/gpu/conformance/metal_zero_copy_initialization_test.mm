/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 (Metal device required; passes-as-skipped without one): the zero-copy
// shared-memory INITIALIZATION probe -- pre-gate-metal-zero-copy-shared-memory
// task 3.4.
//
// The device-free contract is pinned in metal_zero_copy_contract_test.cc. What
// only a real device can answer is whether THIS host's 512 MiB guest physical
// view actually satisfies Apple's newBufferWithBytesNoCopy contract, and
// whether a rejected attempt genuinely lands on the mirror. Both arms run
// against the REAL Memory + REAL MetalSharedMemory: no replica of the
// initialization decision.
//
// Normal launch defaults are untouched. The gate is exact-off, so every arm
// here sets it explicitly around one Initialize and restores the prior
// environment afterwards.

#import <Metal/Metal.h>

#include <cstdlib>
#include <memory>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/memory.h"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_zero_copy_contract.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// Sets an environment variable for the duration of one arm and restores the
// exact previous state (including "was unset"), so arms cannot leak into each
// other or into the rest of the suite.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* previous = std::getenv(name);
    had_previous_ = previous != nullptr;
    if (had_previous_) {
      previous_ = previous;
    }
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (had_previous_) {
      setenv(name_, previous_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_previous_ = false;
  std::string previous_;
};

id<MTLDevice> Device() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

// One initialized MetalSharedMemory over the process-wide test Memory. Torn
// down at scope exit so the next arm starts from a clean write-watch
// registration.
struct InitArm {
  std::unique_ptr<MetalSharedMemory> shared;
  bool ok = false;

  InitArm() {
    if (!Device()) {
      return;
    }
    Memory* memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    shared = std::make_unique<MetalSharedMemory>(*memory,
                                                 (__bridge void*)Device());
    ok = shared->Initialize();
    if (!ok) {
      shared.reset();
    }
  }
};

}  // namespace

TEST_CASE("Zero-copy init: device harness is live (not skipped)",
          "[l1][metal][zerocopy]") {
  // Recorded rather than required: the suite runs device-free in restricted
  // environments, and a silent skip there must not read as a pass here.
  if (!Device()) {
    WARN("no Metal device; zero-copy device arms are skipped");
  }
  SUCCEED();
}

TEST_CASE("Zero-copy init: exact-off uses the mirror and observes nothing",
          "[l1][metal][zerocopy]") {
  if (!Device()) {
    return;
  }
  ScopedEnv gate(kZeroCopyGateEnvVar, nullptr);
  ScopedEnv force(kZeroCopyForceRejectEnvVar, nullptr);
  InitArm arm;
  REQUIRE(arm.ok);

  CHECK_FALSE(arm.shared->is_zero_copy());
  // The mirror buffer is a distinct allocation from guest RAM.
  const void* contents = [(__bridge id<MTLBuffer>)arm.shared->GetBuffer() contents];
  CHECK(contents != conformance::TestPhysicalMemory()->TranslatePhysical(0));

  // No alias-only work: every recorded fact is still at its untouched default.
  const ZeroCopyObservation& o = arm.shared->zero_copy_observation();
  CHECK_FALSE(o.requested);
  CHECK(o.precondition == ZeroCopyPrecondition::kNotEvaluated);
  CHECK(o.device_step == ZeroCopyDeviceStep::kNotAttempted);
  CHECK_FALSE(o.alias_buffer_installed);
  CHECK(DeriveZeroCopyStartupReport(o) ==
        ZeroCopyStartupReport::kNotRequested);
}

TEST_CASE("Zero-copy init: a malformed gate value is exactly off",
          "[l1][metal][zerocopy]") {
  if (!Device()) {
    return;
  }
  // "true" is the value a permissive parser would accept; it must behave
  // identically to unset, down to the untouched observation.
  ScopedEnv gate(kZeroCopyGateEnvVar, "true");
  ScopedEnv force(kZeroCopyForceRejectEnvVar, nullptr);
  InitArm arm;
  REQUIRE(arm.ok);

  CHECK_FALSE(arm.shared->is_zero_copy());
  const ZeroCopyObservation& o = arm.shared->zero_copy_observation();
  CHECK_FALSE(o.requested);
  CHECK(o.precondition == ZeroCopyPrecondition::kNotEvaluated);
  CHECK(o.device_step == ZeroCopyDeviceStep::kNotAttempted);
  CHECK(DeriveZeroCopyStartupReport(o) ==
        ZeroCopyStartupReport::kNotRequested);
}

TEST_CASE("Zero-copy init: forced rejection falls back to the mirror",
          "[l1][metal][zerocopy]") {
  if (!Device()) {
    return;
  }
  // The requested-but-rejected arm. Without it the only non-aliasing outcome
  // reachable on a conforming host is "never asked", which cannot show that
  // the rejection path itself completes normally on the mirror.
  ScopedEnv gate(kZeroCopyGateEnvVar, "1");
  ScopedEnv force(kZeroCopyForceRejectEnvVar, "1");
  InitArm arm;
  REQUIRE(arm.ok);

  CHECK_FALSE(arm.shared->is_zero_copy());
  const void* contents = [(__bridge id<MTLBuffer>)arm.shared->GetBuffer() contents];
  CHECK(contents != conformance::TestPhysicalMemory()->TranslatePhysical(0));

  const ZeroCopyObservation& o = arm.shared->zero_copy_observation();
  CHECK(o.requested);
  CHECK(o.precondition == ZeroCopyPrecondition::kForcedRejection);
  // Rejected before any device call, so no partial alias resource exists.
  CHECK(o.device_step == ZeroCopyDeviceStep::kNotAttempted);
  CHECK_FALSE(o.alias_buffer_installed);
  CHECK(DeriveZeroCopyStartupReport(o) ==
        ZeroCopyStartupReport::kPreconditionRejected);
}

TEST_CASE("Zero-copy init: the local 512 MiB guest view against the contract",
          "[l1][metal][zerocopy]") {
  if (!Device()) {
    return;
  }
  ScopedEnv gate(kZeroCopyGateEnvVar, "1");
  ScopedEnv force(kZeroCopyForceRejectEnvVar, nullptr);
  InitArm arm;
  REQUIRE(arm.ok);

  const ZeroCopyObservation& o = arm.shared->zero_copy_observation();
  CHECK(o.requested);
  ZeroCopyStartupReport report = DeriveZeroCopyStartupReport(o);
  // The report must never be self-contradictory, whichever way this host goes.
  CHECK(report != ZeroCopyStartupReport::kInconsistentObservation);
  // A requested run must have evaluated its own contract.
  CHECK(o.precondition != ZeroCopyPrecondition::kNotEvaluated);

  // This is the open local question the pre-gate exists to answer, so the
  // outcome is RECORDED, not asserted: a host whose physical view spans several
  // VM regions is a legitimate NO for aliasing, not a test failure.
  WARN("local zero-copy verdict: "
       << ZeroCopyStartupReportName(report) << " (precondition "
       << ZeroCopyPreconditionName(o.precondition) << ", device step "
       << ZeroCopyDeviceStepName(o.device_step) << ")");

  const void* contents = [(__bridge id<MTLBuffer>)arm.shared->GetBuffer() contents];
  const void* guest_base =
      conformance::TestPhysicalMemory()->TranslatePhysical(0);
  if (report == ZeroCopyStartupReport::kAliasActive) {
    // Accepted aliasing: the buffer's bytes ARE guest RAM, and is_zero_copy()
    // agrees with the installed backing rather than with the request.
    CHECK(arm.shared->is_zero_copy());
    CHECK(contents == guest_base);
    CHECK(o.precondition == ZeroCopyPrecondition::kSatisfied);
    CHECK(o.device_step == ZeroCopyDeviceStep::kAccepted);
    CHECK(o.alias_buffer_installed);

    // A guest RAM write is visible through the buffer with no upload, and the
    // reverse, because there is one allocation. Uses the last page of the
    // domain so it cannot collide with other suites' fixtures.
    const uint32_t probe_addr = SharedMemory::kBufferSize - 4096;
    uint8_t* ram = static_cast<uint8_t*>(
        conformance::TestPhysicalMemory()->TranslatePhysical(probe_addr));
    uint8_t* view = static_cast<uint8_t*>(const_cast<void*>(contents)) +
                    probe_addr;
    CHECK(ram == view);
    const uint8_t saved = ram[0];
    ram[0] = uint8_t(saved + 0x5A);
    CHECK(view[0] == ram[0]);
    ram[0] = saved;
  } else {
    // Every rejection route must have completed the mirror allocation
    // normally, left is_zero_copy() false, and released any partial alias.
    CHECK_FALSE(arm.shared->is_zero_copy());
    CHECK(contents != guest_base);
    CHECK_FALSE(o.alias_buffer_installed);
  }
}

TEST_CASE("Zero-copy attempt: hostile inputs reject on the real device",
          "[l1][metal][zerocopy]") {
  if (!Device()) {
    return;
  }
  void* device = (__bridge void*)Device();
  void* guest_base = conformance::TestPhysicalMemory()->TranslatePhysical(0);
  const uint64_t page = uint64_t(xe::memory::page_size());

  SECTION("null base") {
    ZeroCopyAliasAttempt a =
        TryCreateZeroCopyAliasBuffer(device, nullptr, SharedMemory::kBufferSize,
                                     false);
    CHECK(a.precondition == ZeroCopyPrecondition::kNullPointer);
    CHECK(a.device_step == ZeroCopyDeviceStep::kNotAttempted);
    CHECK(a.buffer == nullptr);
  }
  SECTION("empty length") {
    ZeroCopyAliasAttempt a =
        TryCreateZeroCopyAliasBuffer(device, guest_base, 0, false);
    CHECK(a.precondition == ZeroCopyPrecondition::kEmptyLength);
    CHECK(a.buffer == nullptr);
  }
  SECTION("misaligned base") {
    ZeroCopyAliasAttempt a = TryCreateZeroCopyAliasBuffer(
        device, static_cast<uint8_t*>(guest_base) + 1,
        SharedMemory::kBufferSize, false);
    CHECK(a.precondition == ZeroCopyPrecondition::kPointerAlignment);
    CHECK(a.device_step == ZeroCopyDeviceStep::kNotAttempted);
    CHECK(a.buffer == nullptr);
  }
  SECTION("misaligned length") {
    ZeroCopyAliasAttempt a = TryCreateZeroCopyAliasBuffer(
        device, guest_base, SharedMemory::kBufferSize + 1, false);
    CHECK(a.precondition == ZeroCopyPrecondition::kLengthAlignment);
    CHECK(a.buffer == nullptr);
  }
  SECTION("past the device buffer-length limit") {
    // Rejected by our own check against the device's reported maximum, so the
    // driver never sees a request it would have to diagnose.
    const uint64_t too_long =
        (uint64_t([Device() maxBufferLength]) + page) & ~(page - 1);
    ZeroCopyAliasAttempt a =
        TryCreateZeroCopyAliasBuffer(device, guest_base, too_long, false);
    CHECK(a.precondition == ZeroCopyPrecondition::kDeviceLimit);
    CHECK(a.device_step == ZeroCopyDeviceStep::kNotAttempted);
    CHECK(a.buffer == nullptr);
  }
  SECTION("forced rejection") {
    ZeroCopyAliasAttempt a = TryCreateZeroCopyAliasBuffer(
        device, guest_base, SharedMemory::kBufferSize, true);
    CHECK(a.precondition == ZeroCopyPrecondition::kForcedRejection);
    CHECK(a.device_step == ZeroCopyDeviceStep::kNotAttempted);
    CHECK(a.buffer == nullptr);
  }
  SECTION("an mprotect-split interval is rejected before the Metal call") {
    // A real multi-VM-region span, built on this host: reserve two pages, then
    // change the protection of the second so the kernel splits the entry. This
    // is the precise shape Apple's single-region requirement excludes, and the
    // one the production initializer must detect on a host whose guest mapping
    // has already been fragmented.
    const size_t span = size_t(page) * 2;
    void* region = xe::memory::AllocFixed(
        nullptr, span, xe::memory::AllocationType::kReserveCommit,
        xe::memory::PageAccess::kReadWrite);
    REQUIRE(region != nullptr);
    const bool split = xe::memory::Protect(
        static_cast<uint8_t*>(region) + page, size_t(page),
        xe::memory::PageAccess::kReadOnly);
    REQUIRE(split);

    ZeroCopyAliasAttempt a =
        TryCreateZeroCopyAliasBuffer(device, region, uint64_t(span), false);
    CHECK(a.precondition == ZeroCopyPrecondition::kSatisfied);
    CHECK(a.device_step == ZeroCopyDeviceStep::kVmRegionRejected);
    CHECK(a.buffer == nullptr);

    // The first page alone IS one region, so the rejection above is about the
    // span and not about this mapping being unwrappable in principle.
    ZeroCopyAliasAttempt single =
        TryCreateZeroCopyAliasBuffer(device, region, uint64_t(page), false);
    CHECK(single.precondition == ZeroCopyPrecondition::kSatisfied);
    CHECK(single.device_step == ZeroCopyDeviceStep::kAccepted);
    REQUIRE(single.buffer != nullptr);
    {
      id<MTLBuffer> wrapped = (__bridge_transfer id<MTLBuffer>)single.buffer;
      // The postcondition production relies on: contents ARE the wrapped
      // pages, and the buffer is in Metal's default tracked hazard mode.
      CHECK([wrapped contents] == region);
      CHECK([wrapped hazardTrackingMode] == MTLHazardTrackingModeTracked);
      CHECK([wrapped storageMode] == MTLStorageModeShared);
      CHECK([wrapped length] == page);
    }

    xe::memory::Protect(static_cast<uint8_t*>(region) + page, size_t(page),
                        xe::memory::PageAccess::kReadWrite);
    xe::memory::DeallocFixed(region, 0,
                             xe::memory::DeallocationType::kRelease);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
