/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 1 (Metal device required; passes-as-skipped without one): the in-flight
// CPU rewrite evidence WIRING -- pre-gate-metal-zero-copy-shared-memory tasks
// 5.1-5.3.
//
// The model's rules are pinned device-free in metal_zero_copy_overlap_test.cc.
// What only the real object can answer is whether the two sources are actually
// connected to the production seams:
//
//   * does a successful alias RequestConsumedRange enrol the pages it served?
//   * does the PHYSICAL invalidation thunk -- the exact entry the guest's
//     access-violation path reaches -- observe that enrollment?
//   * does a BACKEND publication, which calls MemoryInvalidationCallback
//     directly, stay out of the CPU-write evidence?
//   * do exact-off and forced-fallback runs stay silent, without so much as
//     asking a provider for a submission?
//   * does an alias whose OBSERVER cannot be armed fall back to the mirror
//     rather than aliasing unwatched?
//   * are the two seams serialized by the physical-invalidation callback lock,
//     and do they agree about a page across threads?
//
// Everything runs the REAL MetalSharedMemory over the REAL Memory. The thunk is
// reached through a subclass that exposes the production static, not through a
// replica of it, so a mutation that unhooks the observation from the thunk is
// visible here.
//
// Normal launch defaults are untouched: the gate is exact-off and every arm
// sets it explicitly around one Initialize, restoring the previous environment
// afterwards.

#import <Metal/Metal.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/memory.h"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/metal/metal_zero_copy_contract.h"
#include "xenia/gpu/metal/metal_zero_copy_overlap.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// Own pages in a region no other suite in this binary touches, so a
// neighbouring fixture's write-watch state cannot bleed into a page stamp.
constexpr uint32_t kOverlapBase = 0x1B000000u;
constexpr uint32_t kSilenceBase = 0x1B100000u;
constexpr uint32_t kBackendBase = 0x1B200000u;
constexpr uint32_t kFailBase = 0x1B300000u;
constexpr uint32_t kOrderBase = 0x1B400000u;
constexpr uint32_t kPage = 4096u;  // guest allocation granularity.
uint32_t HostPage() { return uint32_t(xe::memory::page_size()); }

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

Memory& Guest() { return *conformance::TestPhysicalMemory(); }

void EnsureAllocated(uint32_t address, uint32_t length) {
  uint32_t base = address & ~(kPage - 1);
  uint32_t end = (address + length + kPage - 1) & ~(kPage - 1);
  Guest().GetPhysicalHeap()->AllocFixed(
      base, end - base, kPage,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite);
}

// The production dispatch boundary, reached rather than reimplemented.
// MemoryInvalidationCallbackThunk is the static Memory itself calls on a guest
// access violation; exposing it from a subclass is what lets these cases drive
// the SAME entry a page fault does, without installing a second watch or
// faulting for real (which would make the counts depend on which pages the host
// happened to have protected).
class ThunkExposingSharedMemory final : public MetalSharedMemory {
 public:
  ThunkExposingSharedMemory(Memory& memory, void* device)
      : MetalSharedMemory(memory, device) {}

  std::pair<uint32_t, uint32_t> InvokePhysicalWriteThunk(uint32_t start,
                                                         uint32_t length,
                                                         bool exact_range) {
    return MemoryInvalidationCallbackThunk(this, start, length, exact_range);
  }
};

// One initialized shared memory plus COUNTING overlap providers. The counters
// are the point: the control arms prove silence by showing the providers were
// never asked, which a zero observation counter alone could not do (a mirror
// run leaves it at zero whether the guard works or not).
struct OverlapArm {
  std::unique_ptr<ThunkExposingSharedMemory> shared;
  bool ok = false;
  uint32_t submission_calls = 0;
  uint32_t completed_calls = 0;
  uint64_t submission_value = 0;
  uint64_t completed_value = 0;

  OverlapArm(bool request_alias, bool force_reject,
             bool force_observer_reject = false) {
    if (!Device()) {
      return;
    }
    Memory* memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    ScopedEnv gate(kZeroCopyGateEnvVar, request_alias ? "1" : nullptr);
    ScopedEnv force(kZeroCopyForceRejectEnvVar, force_reject ? "1" : nullptr);
    // The observer-arming injection. Separate from the contract's forced
    // rejection because it engages LATER -- after Metal has handed back a
    // genuine alias -- which is the only way to reach the observer-rejection
    // fallback on a host that satisfies the contract.
    ScopedEnv force_observer(kZeroCopyForceObserverRejectEnvVar,
                             force_observer_reject ? "1" : nullptr);
    shared = std::make_unique<ThunkExposingSharedMemory>(
        *memory, (__bridge void*)Device());
    // Installed BEFORE Initialize, exactly as MetalCommandProcessor installs
    // them, so the observer is armed before the physical callback exists.
    shared->SetOverlapSubmissionProvider([this]() {
      ++submission_calls;
      return submission_value;
    });
    shared->SetOverlapCompletedSubmissionProvider([this]() {
      ++completed_calls;
      return completed_value;
    });
    ok = shared->Initialize();
    if (!ok) {
      shared.reset();
    }
  }

  bool aliases() const { return shared && shared->is_zero_copy(); }
  ZeroCopyOverlapSnapshot snapshot() const {
    return shared ? shared->zero_copy_overlap_snapshot()
                  : ZeroCopyOverlapSnapshot();
  }
  ZeroCopyStartupReport report() const {
    return shared ? DeriveZeroCopyStartupReport(shared->zero_copy_observation())
                  : ZeroCopyStartupReport::kInconsistentObservation;
  }
  // Whether the installed buffer's bytes are guest RAM, read from the BUFFER
  // rather than from is_zero_copy(): the fallback claim is about which
  // allocation the GPU reads, so it must not be checked against the flag that
  // is supposed to describe it.
  bool installed_backing_is_guest_ram() const {
    if (!shared || !shared->GetBuffer()) {
      return false;
    }
    return [(__bridge id<MTLBuffer>)shared->GetBuffer() contents] ==
           conformance::TestPhysicalMemory()->TranslatePhysical(0);
  }
};

// Whether THIS host reaches an accepted alias at all. The observer-rejection
// fallback only exists after Metal has said yes, so on a host that rejects
// earlier (a physical view spanning several VM regions is a legitimate no)
// there is nothing to fall back FROM and the arm proves nothing. Probed with a
// scoped arm that is destroyed before the real one is built, so only one shared
// memory is ever registered at a time.
bool HostAcceptsAlias() {
  OverlapArm probe(/*request_alias=*/true, /*force_reject=*/false);
  return probe.ok && probe.aliases();
}

// Host pages an exactly-aligned request of `length` bytes covers.
uint32_t HostPagesFor(uint32_t length) { return length / HostPage(); }

}  // namespace

TEST_CASE("Zero-copy overlap: the device harness is live (not skipped)",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    WARN("no Metal device; zero-copy overlap device arms are skipped");
  }
  SUCCEED();
}

TEST_CASE("Zero-copy overlap: a successful alias RequestConsumedRange enrols "
          "its host pages and the physical thunk observes them",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false);
  REQUIRE(arm.ok);
  if (!arm.aliases()) {
    WARN("host does not alias guest RAM; the overlap wiring arm has no alias "
         "to observe and is not asserted");
    return;
  }
  const uint32_t length = 2 * HostPage();
  const uint32_t pages = HostPagesFor(length);
  EnsureAllocated(kOverlapBase, length);

  // EXECUTION source: the real production request seam, with the submission
  // currently being encoded.
  arm.submission_value = 9;
  ZeroCopyOverlapSnapshot before_stamp = arm.snapshot();
  REQUIRE(arm.shared->RequestConsumedRange(kOverlapBase, length));
  ZeroCopyOverlapSnapshot after_stamp = arm.snapshot();
  CHECK(after_stamp.active);
  CHECK(after_stamp.stamp_calls - before_stamp.stamp_calls == 1u);
  CHECK(after_stamp.stamped_pages - before_stamp.stamped_pages == pages);
  // The alias arm asked the EXECUTION provider, and it reported a real
  // submission rather than the idle sentinel.
  CHECK(arm.submission_calls >= 1u);
  CHECK(after_stamp.zero_submission_stamp_pages == 0u);

  // OBSERVATION source: the physical invalidation thunk, with the enrolling
  // submission still unfinished.
  arm.completed_value = 8;
  const uint32_t completed_calls_before = arm.completed_calls;
  ZeroCopyOverlapSnapshot before_write = arm.snapshot();
  arm.shared->InvokePhysicalWriteThunk(kOverlapBase, length,
                                       /*exact_range=*/true);
  ZeroCopyOverlapSnapshot after_write = arm.snapshot();

  CHECK(after_write.cpu_write_callbacks - before_write.cpu_write_callbacks ==
        1u);
  CHECK(after_write.cpu_write_pages - before_write.cpu_write_pages == pages);
  CHECK(after_write.in_flight_cpu_write_overlap -
            before_write.in_flight_cpu_write_overlap ==
        pages);
  CHECK(after_write.completed_reader_pages -
            before_write.completed_reader_pages ==
        0u);
  // ONE completed high-water mark for the whole callback, however many pages it
  // spans -- the provider is asked exactly once.
  CHECK(arm.completed_calls - completed_calls_before == 1u);
  REQUIRE(after_write.sample_count >= 1u);
  CHECK(after_write.samples[0].consumer_submission == 9u);
  CHECK(after_write.samples[0].completed_submission == 8u);

  // And the converse, through the same two seams: once the enrolling submission
  // has completed, the same physical write is not an overlap.
  arm.completed_value = 9;
  ZeroCopyOverlapSnapshot before_safe = arm.snapshot();
  arm.shared->InvokePhysicalWriteThunk(kOverlapBase, length,
                                       /*exact_range=*/true);
  ZeroCopyOverlapSnapshot after_safe = arm.snapshot();
  CHECK(after_safe.in_flight_cpu_write_overlap ==
        before_safe.in_flight_cpu_write_overlap);
  CHECK(after_safe.completed_reader_pages - before_safe.completed_reader_pages ==
        pages);
}

TEST_CASE("Zero-copy overlap: a backend publication is not a guest CPU write",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false);
  REQUIRE(arm.ok);
  if (!arm.aliases()) {
    WARN("host does not alias guest RAM; the backend-publication arm has no "
         "alias to observe and is not asserted");
    return;
  }
  const uint32_t length = HostPage();
  EnsureAllocated(kBackendBase, length);
  arm.submission_value = 5;
  REQUIRE(arm.shared->RequestConsumedRange(kBackendBase, length));
  arm.completed_value = 4;

  // The backend's own publications (resolve, memexport, trace restore) call the
  // NONVIRTUAL MemoryInvalidationCallback directly. They are GPU writes wearing
  // the invalidation call's name; counting them would make the evidence
  // describe the backend observing itself, and every frame would "overlap".
  const uint32_t completed_calls_before = arm.completed_calls;
  ZeroCopyOverlapSnapshot before = arm.snapshot();
  arm.shared->MemoryInvalidationCallback(kBackendBase, length,
                                         /*exact_range=*/true);
  arm.shared->MemoryInvalidationCallback(kBackendBase, length,
                                         /*exact_range=*/true,
                                         /*fire_global_watches=*/false);
  ZeroCopyOverlapSnapshot after = arm.snapshot();

  CHECK(after.cpu_write_callbacks == before.cpu_write_callbacks);
  CHECK(after.cpu_write_pages == before.cpu_write_pages);
  CHECK(after.in_flight_cpu_write_overlap == before.in_flight_cpu_write_overlap);
  // Not even a completion read: the observation hook was never entered.
  CHECK(arm.completed_calls == completed_calls_before);

  // The SAME range through the thunk is observed, so the difference above is
  // the dispatch boundary and not the range.
  arm.shared->InvokePhysicalWriteThunk(kBackendBase, length,
                                       /*exact_range=*/true);
  ZeroCopyOverlapSnapshot thunked = arm.snapshot();
  CHECK(thunked.cpu_write_callbacks - before.cpu_write_callbacks == 1u);
  CHECK(thunked.in_flight_cpu_write_overlap -
            before.in_flight_cpu_write_overlap ==
        HostPagesFor(length));
  CHECK(arm.completed_calls - completed_calls_before == 1u);
}

TEST_CASE("Zero-copy overlap: a failed request enrols nothing",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false);
  REQUIRE(arm.ok);
  if (!arm.aliases()) {
    WARN("host does not alias guest RAM; the failed-request arm is not "
         "asserted");
    return;
  }
  // A range the guest never committed: UploadRanges rejects it, so RequestRange
  // fails and no consumer was served. Recording an enrollment anyway would
  // manufacture a claim that never happened -- and every later CPU write to
  // those pages would then report an overlap against work that was never
  // encoded.
  arm.submission_value = 12;
  ZeroCopyOverlapSnapshot before = arm.snapshot();
  const bool served = arm.shared->RequestConsumedRange(kFailBase, HostPage());
  ZeroCopyOverlapSnapshot after = arm.snapshot();
  if (served) {
    WARN("the unallocated probe range was serviceable on this host; the "
         "failed-request arm proves nothing here");
    return;
  }
  CHECK(after.stamp_calls == before.stamp_calls);
  CHECK(after.stamped_pages == before.stamped_pages);

  // Nothing is claimed, so a physical write to the same pages is idle.
  arm.completed_value = 0;
  arm.shared->InvokePhysicalWriteThunk(kFailBase, HostPage(),
                                       /*exact_range=*/true);
  ZeroCopyOverlapSnapshot observed = arm.snapshot();
  CHECK(observed.in_flight_cpu_write_overlap ==
        before.in_flight_cpu_write_overlap);
  CHECK(observed.idle_pages - before.idle_pages == HostPagesFor(HostPage()));
}

TEST_CASE("Zero-copy overlap: an alias whose observer cannot be armed installs "
          "the mirror instead of aliasing unobserved",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  if (!HostAcceptsAlias()) {
    WARN("host does not alias guest RAM; there is no accepted alias for the "
         "observer-rejection fallback to reject, and this arm is not asserted");
    return;
  }
  // The failure this rule exists against: arming the observer fails, the alias
  // stays installed anyway, and every later run reports zero overlaps because
  // NOBODY IS LOOKING -- which the silence rule then reads as a clean arm.
  // Absence of evidence presented as permission is the one outcome this
  // pre-gate may not produce.
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false,
                 /*force_observer_reject=*/true);
  REQUIRE(arm.ok);

  // 1. The alias candidate was released and the ordinary mirror installed.
  //    Checked at the BUFFER, not at the flag that claims to describe it.
  CHECK_FALSE(arm.aliases());
  CHECK_FALSE(arm.installed_backing_is_guest_ram());

  // 2. A precise, named verdict. Not `accepted` (nothing is aliased), not
  //    `contents_rejected` (Metal handed back a real alias; blaming the device
  //    would be false), and not `inconsistent_observation` (this is a decision
  //    the backend made deliberately, not a model that disagrees with itself).
  const ZeroCopyObservation& o = arm.shared->zero_copy_observation();
  CHECK(o.requested);
  CHECK(o.precondition == ZeroCopyPrecondition::kSatisfied);
  CHECK(o.device_step == ZeroCopyDeviceStep::kObserverRejected);
  CHECK_FALSE(o.alias_buffer_installed);
  CHECK(arm.report() == ZeroCopyStartupReport::kObserverRejected);
  CHECK(arm.report() != ZeroCopyStartupReport::kAliasActive);
  CHECK(arm.report() != ZeroCopyStartupReport::kContentsIdentityRejected);
  CHECK(arm.report() != ZeroCopyStartupReport::kInconsistentObservation);
  CHECK(std::string(ZeroCopyDeviceStepName(o.device_step)) ==
        "observer_rejected");
  CHECK(std::string(ZeroCopyStartupReportName(arm.report())) ==
        "observer_rejected");

  // 3. The observer is inactive and the snapshot is silent -- the same state
  //    every other fallback arm produces, so the reporter emits nothing.
  ZeroCopyOverlapSnapshot snapshot = arm.snapshot();
  CHECK_FALSE(snapshot.active);
  CHECK(snapshot.page_count == 0u);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(snapshot));

  // 4. And it behaves like a mirror run at the SEAMS, proven at the providers:
  //    a fallback that kept asking for submissions would be alias-only work
  //    performed in a run that is not aliasing.
  const uint32_t length = HostPage();
  EnsureAllocated(kSilenceBase, length);
  arm.submission_value = 9;
  arm.completed_value = 8;
  REQUIRE(arm.shared->RequestConsumedRange(kSilenceBase, length));
  arm.shared->InvokePhysicalWriteThunk(kSilenceBase, length,
                                       /*exact_range=*/true);
  arm.shared->MemoryInvalidationCallback(kSilenceBase, length,
                                         /*exact_range=*/true);
  CHECK(arm.submission_calls == 0u);
  CHECK(arm.completed_calls == 0u);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(arm.snapshot()));
}

TEST_CASE("Zero-copy overlap: exact-off and forced fallback are silent and "
          "never ask an overlap provider",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  // Two control arms: the gate unset, and the gate on with the forced-rejection
  // arm engaged. Both must install the mirror, and both must do NO zero-copy
  // work -- proven at the provider, because a counter left at zero is what a
  // broken guard and a working guard both look like.
  struct Arm {
    const char* name;
    bool request_alias;
    bool force_reject;
    bool force_observer_reject;
    // What an untouched exact-off run must still report. The observer injection
    // is a diagnostic on the ALIAS path, so a run that never requested the
    // alias must not read it, act on it, or record anything because of it --
    // exact-off has to stay bit-for-bit the same launch it was.
    ZeroCopyStartupReport expected;
  };
  const Arm arms[] = {
      {"exact-off", false, false, false, ZeroCopyStartupReport::kNotRequested},
      {"exact-off with the observer injection set", false, false, true,
       ZeroCopyStartupReport::kNotRequested},
      {"forced fallback", true, true, false,
       ZeroCopyStartupReport::kPreconditionRejected},
      {"forced fallback with the observer injection set", true, true, true,
       ZeroCopyStartupReport::kPreconditionRejected},
  };
  const uint32_t length = HostPage();
  EnsureAllocated(kSilenceBase, length);

  for (const Arm& configuration : arms) {
    INFO("control arm: " << configuration.name);
    OverlapArm arm(configuration.request_alias, configuration.force_reject,
                   configuration.force_observer_reject);
    REQUIRE(arm.ok);
    // The premise of a control arm: the mirror is installed.
    REQUIRE_FALSE(arm.aliases());
    CHECK_FALSE(arm.installed_backing_is_guest_ram());
    // The contract rejection happens BEFORE the observer step, so the observer
    // injection never changes a control arm's verdict.
    CHECK(arm.report() == configuration.expected);
    CHECK(arm.shared->zero_copy_observation().device_step !=
          ZeroCopyDeviceStep::kObserverRejected);

    arm.submission_value = 9;
    arm.completed_value = 8;
    REQUIRE(arm.shared->RequestConsumedRange(kSilenceBase, length));
    arm.shared->InvokePhysicalWriteThunk(kSilenceBase, length,
                                         /*exact_range=*/true);
    arm.shared->MemoryInvalidationCallback(kSilenceBase, length,
                                           /*exact_range=*/true);

    // Not asked, not counted, and nothing for a report to print.
    CHECK(arm.submission_calls == 0u);
    CHECK(arm.completed_calls == 0u);
    ZeroCopyOverlapSnapshot snapshot = arm.snapshot();
    CHECK_FALSE(snapshot.active);
    CHECK(snapshot.stamp_calls == 0u);
    CHECK(snapshot.stamped_pages == 0u);
    CHECK(snapshot.cpu_write_callbacks == 0u);
    CHECK(snapshot.cpu_write_pages == 0u);
    CHECK(snapshot.in_flight_cpu_write_overlap == 0u);
    CHECK(snapshot.sample_count == 0u);
    CHECK(ZeroCopyOverlapSnapshotIsSilent(snapshot));
  }
}

TEST_CASE("Zero-copy overlap: a lifecycle snapshot is serialized against the "
          "physical-invalidation callback lock",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  // The evidence is written on a faulting guest thread and read on the GPU
  // thread. Reading it outside the lock that serializes the callbacks would
  // return counters sampled at different instants -- a snapshot that could show
  // more classified pages than observed callbacks, and so read as a defect in
  // the model rather than as the tearing it is.
  //
  // What is asserted here is MUTUAL EXCLUSION against a real second thread
  // holding that exact lock through the narrow Memory API. It is the load-
  // bearing half of the claim: the API returns the same region the dispatcher
  // holds, so joining it is what buys any ordering at all.
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false);
  REQUIRE(arm.ok);

  std::atomic<bool> holding{false};
  std::atomic<bool> reader_ready{false};
  std::atomic<bool> released{false};
  std::thread holder([&]() {
    auto lock = Guest().LockPhysicalMemoryInvalidationCallbacks();
    holding.store(true, std::memory_order_release);
    while (!reader_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    // Long enough that a reader which does NOT take the lock will have finished
    // and read `released` as false. A reader that DOES take it cannot proceed
    // until the store below is published by the unlock.
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    released.store(true, std::memory_order_release);
  });
  while (!holding.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  reader_ready.store(true, std::memory_order_release);
  ZeroCopyOverlapSnapshot snapshot = arm.shared->zero_copy_overlap_snapshot();
  // The whole assertion: the snapshot could not have been taken while the lock
  // was held elsewhere. This can only fail early, never late -- a descheduled
  // reader makes the case weaker, never flaky.
  CHECK(released.load(std::memory_order_acquire));
  holder.join();

  // And it is still a real snapshot, not an empty struct returned to dodge the
  // lock: an aliasing arm reports the armed observer.
  if (arm.aliases()) {
    CHECK(snapshot.active);
    CHECK(snapshot.page_count > 0u);
  } else {
    CHECK(ZeroCopyOverlapSnapshotIsSilent(snapshot));
  }
}

TEST_CASE("Zero-copy overlap: a callback before enrollment sees idle, one "
          "after it sees the claim",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false);
  REQUIRE(arm.ok);
  if (!arm.aliases()) {
    WARN("host does not alias guest RAM; the enrollment-ordering arm has no "
         "alias to observe and is not asserted");
    return;
  }
  const uint32_t length = HostPage();
  const uint32_t pages = HostPagesFor(length);
  EnsureAllocated(kOrderBase, length);

  // Both observations run on a SEPARATE thread that takes the physical-
  // invalidation callback lock itself, exactly as PhysicalHeap::TriggerCallbacks
  // does before dispatching. Enrollment runs on this thread, under the same
  // lock, which is what gives the two a causal order instead of a hopeful one.
  //
  // Honest about what this pins: std::thread creation and join are themselves
  // synchronization edges, so this is a WIRING and CLASSIFICATION pin -- that
  // the two seams agree about a page across threads -- not a demonstration of
  // the C++ memory model. The ordering rule itself is argued from the lock, in
  // metal_shared_memory.mm and metal_zero_copy_overlap.h.
  auto observe_under_callback_lock = [&]() {
    std::thread guest([&]() {
      auto lock = Guest().LockPhysicalMemoryInvalidationCallbacks();
      arm.shared->InvokePhysicalWriteThunk(kOrderBase, length,
                                           /*exact_range=*/true);
    });
    guest.join();
  };

  // BEFORE: nothing has claimed the page, so idle is not a missed overlap --
  // it is the truthful answer for a callback that really did run first.
  arm.completed_value = 20;
  ZeroCopyOverlapSnapshot before_early = arm.snapshot();
  observe_under_callback_lock();
  ZeroCopyOverlapSnapshot after_early = arm.snapshot();
  CHECK(after_early.cpu_write_callbacks - before_early.cpu_write_callbacks == 1u);
  CHECK(after_early.idle_pages - before_early.idle_pages == pages);
  CHECK(after_early.in_flight_cpu_write_overlap ==
        before_early.in_flight_cpu_write_overlap);

  // ENROLL on this thread, under the same lock.
  arm.submission_value = 21;
  REQUIRE(arm.shared->RequestConsumedRange(kOrderBase, length));

  // AFTER: a callback that acquires the lock once the enrollment has released
  // it must see the claim. Reporting idle here would be the silent failure this
  // whole group exists to prevent -- a genuinely raced page counted as safe.
  ZeroCopyOverlapSnapshot before_late = arm.snapshot();
  observe_under_callback_lock();
  ZeroCopyOverlapSnapshot after_late = arm.snapshot();
  CHECK(after_late.cpu_write_callbacks - before_late.cpu_write_callbacks == 1u);
  CHECK(after_late.in_flight_cpu_write_overlap -
            before_late.in_flight_cpu_write_overlap ==
        pages);
  CHECK(after_late.idle_pages == before_late.idle_pages);
}

TEST_CASE("Zero-copy overlap: the evidence dies with the backing, so a final "
          "report must precede shutdown",
          "[l1][metal][zerocopy][overlap]") {
  if (!Device()) {
    return;
  }
  OverlapArm arm(/*request_alias=*/true, /*force_reject=*/false);
  REQUIRE(arm.ok);
  if (!arm.aliases()) {
    WARN("host does not alias guest RAM; the shutdown arm is not asserted");
    return;
  }
  const uint32_t length = HostPage();
  EnsureAllocated(kOverlapBase, length);
  arm.submission_value = 3;
  arm.completed_value = 2;
  REQUIRE(arm.shared->RequestConsumedRange(kOverlapBase, length));
  arm.shared->InvokePhysicalWriteThunk(kOverlapBase, length,
                                       /*exact_range=*/true);
  REQUIRE(arm.snapshot().in_flight_cpu_write_overlap >= 1u);

  // Shutdown releases the observer (it has to: the physical callback is
  // unregistered first, and the fixed table cannot outlive the object). So the
  // run's last chance to read this is BEFORE the shared memory goes away --
  // which is why MetalCommandProcessor::ShutdownContext reports before
  // shared_memory_.reset(), not after.
  arm.shared->Shutdown();
  ZeroCopyOverlapSnapshot after_shutdown = arm.snapshot();
  CHECK_FALSE(after_shutdown.active);
  CHECK(ZeroCopyOverlapSnapshotIsSilent(after_shutdown));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
