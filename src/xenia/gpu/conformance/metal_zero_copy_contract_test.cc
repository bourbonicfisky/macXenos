/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free) conformance for the Metal zero-copy shared-memory
// contract: pre-gate-metal-zero-copy-shared-memory task 2.2.
//
// Three separable contracts, each exhaustively enumerated rather than sampled:
//   * the exact-"1" environment gate,
//   * the overflow-safe device-free precondition evaluator,
//   * the two-source startup observation -> report derivation.
//
// The observation cases sweep the FULL cross product of every recorded fact
// (2 x 10 x 5 x 2 = 200 combinations) and assert the report for each, so a
// rule that stops being load-bearing cannot hide behind a hand-picked case
// list. Enum completeness is pinned at COMPILE time: each sweep runs over a
// std::array sized by the enum's own kCount and static_asserted to hold exactly
// its values 0..kCount-1, so adding an enumerator without enumerating it here
// fails the build rather than silently shrinking the sweep.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/metal/metal_zero_copy_contract.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr uint64_t kPage = 4096;
// A byte with a stable address, so the self-copy identity rule can also be
// checked in a constant expression (its address is all the rule reads).
constexpr uint8_t kIdentityProbe = 0;
constexpr uint64_t kBufferSize = 512ull * 1024 * 1024;
// A plausible Apple Silicon maxBufferLength; only its relation to `length`
// matters to the contract.
constexpr uint64_t kDeviceLimit = 4ull * 1024 * 1024 * 1024;

// Every value of each enum, in declaration order. The array LENGTH is the
// enum's own kCount, so a new enumerator added before kCount that is not also
// listed here is a compile error at the initializer -- the completeness pin is
// the type, not a hand-updated number. IsExhaustive below additionally requires
// the listed values to be the enum's values 0..kCount-1 with no repeats or
// omissions, which a same-length but wrong list would otherwise satisfy.
template <typename T, size_t N>
constexpr bool IsExhaustive(const std::array<T, N>& values) {
  for (size_t i = 0; i < N; ++i) {
    if (values[i] != static_cast<T>(i)) {
      return false;
    }
  }
  return true;
}

constexpr std::array<ZeroCopyPrecondition, kZeroCopyPreconditionCount>
    kAllPreconditions = {
        ZeroCopyPrecondition::kNotEvaluated,
        ZeroCopyPrecondition::kSatisfied,
        ZeroCopyPrecondition::kForcedRejection,
        ZeroCopyPrecondition::kNullPointer,
        ZeroCopyPrecondition::kEmptyLength,
        ZeroCopyPrecondition::kInvalidPageSize,
        ZeroCopyPrecondition::kPointerAlignment,
        ZeroCopyPrecondition::kLengthAlignment,
        ZeroCopyPrecondition::kAddressOverflow,
        ZeroCopyPrecondition::kDeviceLimit,
};
static_assert(IsExhaustive(kAllPreconditions));

constexpr std::array<ZeroCopyDeviceStep, kZeroCopyDeviceStepCount>
    kAllDeviceSteps = {
        ZeroCopyDeviceStep::kNotAttempted,
        ZeroCopyDeviceStep::kVmRegionRejected,
        ZeroCopyDeviceStep::kAllocationRejected,
        ZeroCopyDeviceStep::kContentsRejected,
        ZeroCopyDeviceStep::kObserverRejected,
        ZeroCopyDeviceStep::kAccepted,
};
static_assert(IsExhaustive(kAllDeviceSteps));

constexpr std::array<ZeroCopyStartupReport, kZeroCopyStartupReportCount>
    kAllReports = {
        ZeroCopyStartupReport::kNotRequested,
        ZeroCopyStartupReport::kPreconditionRejected,
        ZeroCopyStartupReport::kVmRegionRejected,
        ZeroCopyStartupReport::kAllocationRejected,
        ZeroCopyStartupReport::kContentsIdentityRejected,
        ZeroCopyStartupReport::kObserverRejected,
        ZeroCopyStartupReport::kAliasActive,
        ZeroCopyStartupReport::kInconsistentObservation,
};
static_assert(IsExhaustive(kAllReports));

// A request that satisfies every device-free condition; individual cases below
// perturb exactly one field so the failing check is unambiguous.
ZeroCopyContractInputs GoodInputs() {
  ZeroCopyContractInputs in;
  in.base_address = 0x1'0000'0000ull;  // page-aligned
  in.length = kBufferSize;
  in.host_page_size = kPage;
  in.device_max_buffer_length = kDeviceLimit;
  in.force_rejection = false;
  return in;
}

// The report the model is expected to produce, computed independently of
// DeriveZeroCopyStartupReport so the exhaustive sweep below is a comparison
// against a restatement of the rules rather than against itself.
ZeroCopyStartupReport ExpectedReport(const ZeroCopyObservation& o) {
  const bool untouched =
      o.precondition == ZeroCopyPrecondition::kNotEvaluated &&
      o.device_step == ZeroCopyDeviceStep::kNotAttempted &&
      !o.alias_buffer_installed;
  if (!o.requested) {
    return untouched ? ZeroCopyStartupReport::kNotRequested
                     : ZeroCopyStartupReport::kInconsistentObservation;
  }
  if (o.precondition == ZeroCopyPrecondition::kNotEvaluated) {
    return ZeroCopyStartupReport::kInconsistentObservation;
  }
  if (o.precondition != ZeroCopyPrecondition::kSatisfied) {
    const bool clean = o.device_step == ZeroCopyDeviceStep::kNotAttempted &&
                       !o.alias_buffer_installed;
    return clean ? ZeroCopyStartupReport::kPreconditionRejected
                 : ZeroCopyStartupReport::kInconsistentObservation;
  }
  if (o.device_step == ZeroCopyDeviceStep::kAccepted) {
    return o.alias_buffer_installed
               ? ZeroCopyStartupReport::kAliasActive
               : ZeroCopyStartupReport::kInconsistentObservation;
  }
  if (o.alias_buffer_installed) {
    return ZeroCopyStartupReport::kInconsistentObservation;
  }
  switch (o.device_step) {
    case ZeroCopyDeviceStep::kNotAttempted:
      return ZeroCopyStartupReport::kInconsistentObservation;
    case ZeroCopyDeviceStep::kVmRegionRejected:
      return ZeroCopyStartupReport::kVmRegionRejected;
    case ZeroCopyDeviceStep::kAllocationRejected:
      return ZeroCopyStartupReport::kAllocationRejected;
    case ZeroCopyDeviceStep::kContentsRejected:
      return ZeroCopyStartupReport::kContentsIdentityRejected;
    case ZeroCopyDeviceStep::kObserverRejected:
      // The alias was real but unobservable, so it was released: a distinct
      // verdict, and specifically NOT kContentsIdentityRejected (the device
      // said yes) and NOT kAliasActive (nothing is aliased).
      return ZeroCopyStartupReport::kObserverRejected;
    case ZeroCopyDeviceStep::kAccepted:
      return ZeroCopyStartupReport::kAliasActive;
  }
  return ZeroCopyStartupReport::kInconsistentObservation;
}

}  // namespace

TEST_CASE("Zero-copy gate: only the exact value \"1\" enables the probe",
          "[l0][metal][zerocopy]") {
  CHECK(IsZeroCopyGateEnabled("1"));

  // Unset is the default state of every normal launch.
  CHECK_FALSE(IsZeroCopyGateEnabled(nullptr));
  // Empty, explicit-off, and the near misses that a "nonempty means on" or a
  // "starts with 1" parser would wrongly accept.
  for (const char* value : {"", "0", "01", "1 ", " 1", "10", "11", "1\n", "1\t",
                            "true", "TRUE", "True", "yes", "on", "enabled",
                            "-1", "+1", "1.0", "2", "y", "Y"}) {
    INFO("gate value: '" << value << "'");
    CHECK_FALSE(IsZeroCopyGateEnabled(value));
  }

  // Both control arms use the same exact-off parser, so the three gates cannot
  // drift into different acceptance rules.
  CHECK(IsZeroCopyGateEnabled("1"));
  CHECK_FALSE(IsZeroCopyGateEnabled("0"));

  // The gate names are part of the contract: the pre-gate's control arms and
  // its evidence are keyed to these exact strings.
  CHECK(std::string(kZeroCopyGateEnvVar) == "XE_METAL_ZERO_COPY_SHARED_MEMORY");
  CHECK(std::string(kZeroCopyForceRejectEnvVar) ==
        "XE_METAL_ZERO_COPY_FORCE_REJECT");
  CHECK(std::string(kZeroCopyForceObserverRejectEnvVar) ==
        "XE_METAL_ZERO_COPY_FORCE_OBSERVER_REJECT");
  // Three DISTINCT names. Two injections sharing one variable would make the
  // observer-rejection arm unreachable without also engaging the contract
  // rejection, and the two land on different verdicts.
  CHECK(std::string(kZeroCopyGateEnvVar) !=
        std::string(kZeroCopyForceRejectEnvVar));
  CHECK(std::string(kZeroCopyGateEnvVar) !=
        std::string(kZeroCopyForceObserverRejectEnvVar));
  CHECK(std::string(kZeroCopyForceRejectEnvVar) !=
        std::string(kZeroCopyForceObserverRejectEnvVar));
}

TEST_CASE("Zero-copy preconditions: the accepted request", "[l0][metal][zerocopy]") {
  CHECK(EvaluateZeroCopyPreconditions(GoodInputs()) ==
        ZeroCopyPrecondition::kSatisfied);

  // Boundary acceptances: length exactly at the device limit, and a base of
  // exactly one page. Both must pass -- the checks are `>` and `%`, not `>=`.
  ZeroCopyContractInputs at_limit = GoodInputs();
  at_limit.length = at_limit.device_max_buffer_length;
  CHECK(EvaluateZeroCopyPreconditions(at_limit) ==
        ZeroCopyPrecondition::kSatisfied);

  ZeroCopyContractInputs low_base = GoodInputs();
  low_base.base_address = kPage;
  CHECK(EvaluateZeroCopyPreconditions(low_base) ==
        ZeroCopyPrecondition::kSatisfied);

  // Larger page sizes (16 KiB Apple Silicon) still divide the 512 MiB length.
  ZeroCopyContractInputs big_page = GoodInputs();
  big_page.host_page_size = 16384;
  CHECK(EvaluateZeroCopyPreconditions(big_page) ==
        ZeroCopyPrecondition::kSatisfied);
}

TEST_CASE("Zero-copy preconditions: every rejection class", "[l0][metal][zerocopy]") {
  SECTION("null guest pointer") {
    ZeroCopyContractInputs in = GoodInputs();
    in.base_address = 0;
    CHECK(EvaluateZeroCopyPreconditions(in) ==
          ZeroCopyPrecondition::kNullPointer);
  }
  SECTION("empty length") {
    ZeroCopyContractInputs in = GoodInputs();
    in.length = 0;
    CHECK(EvaluateZeroCopyPreconditions(in) ==
          ZeroCopyPrecondition::kEmptyLength);
  }
  SECTION("unusable host page size") {
    for (uint64_t page : {uint64_t(0), uint64_t(3), uint64_t(4095),
                          uint64_t(6144), uint64_t(12288)}) {
      INFO("page size: " << page);
      ZeroCopyContractInputs in = GoodInputs();
      in.host_page_size = page;
      CHECK(EvaluateZeroCopyPreconditions(in) ==
            ZeroCopyPrecondition::kInvalidPageSize);
    }
  }
  SECTION("pointer misalignment") {
    for (uint64_t offset : {uint64_t(1), uint64_t(8), uint64_t(4095)}) {
      INFO("base offset: " << offset);
      ZeroCopyContractInputs in = GoodInputs();
      in.base_address += offset;
      CHECK(EvaluateZeroCopyPreconditions(in) ==
            ZeroCopyPrecondition::kPointerAlignment);
    }
  }
  SECTION("length misalignment") {
    for (uint64_t offset : {uint64_t(1), uint64_t(8), uint64_t(4095)}) {
      INFO("length offset: " << offset);
      ZeroCopyContractInputs in = GoodInputs();
      in.length += offset;
      CHECK(EvaluateZeroCopyPreconditions(in) ==
            ZeroCopyPrecondition::kLengthAlignment);
    }
  }
  SECTION("address overflow") {
    // A page-aligned base near the top of the address space with a
    // page-aligned length that does not fit. Computing base + length here
    // would wrap and silently pass a naive `base + length < base` test only by
    // luck; the evaluator subtracts instead.
    ZeroCopyContractInputs in = GoodInputs();
    in.base_address = UINT64_MAX - kPage + 1;  // page-aligned
    in.length = kPage * 2;
    in.device_max_buffer_length = UINT64_MAX;
    CHECK(EvaluateZeroCopyPreconditions(in) ==
          ZeroCopyPrecondition::kAddressOverflow);

    // An interval ending EXACTLY at 2^64 is also rejected: its one-past-the-end
    // address is not representable, so no caller could form the half-open range
    // the VM-region check and Metal both work in.
    ZeroCopyContractInputs to_the_top = in;
    to_the_top.length = kPage;
    CHECK(EvaluateZeroCopyPreconditions(to_the_top) ==
          ZeroCopyPrecondition::kAddressOverflow);

    // One page lower, the end address is representable and the request stands.
    ZeroCopyContractInputs highest_valid = in;
    highest_valid.base_address = UINT64_MAX - (kPage * 2) + 1;
    highest_valid.length = kPage;
    CHECK(EvaluateZeroCopyPreconditions(highest_valid) ==
          ZeroCopyPrecondition::kSatisfied);
  }
  SECTION("device buffer-length limit") {
    ZeroCopyContractInputs in = GoodInputs();
    in.device_max_buffer_length = in.length - kPage;
    CHECK(EvaluateZeroCopyPreconditions(in) ==
          ZeroCopyPrecondition::kDeviceLimit);

    // A device reporting no limit at all cannot host any nonzero wrap.
    ZeroCopyContractInputs no_device = GoodInputs();
    no_device.device_max_buffer_length = 0;
    CHECK(EvaluateZeroCopyPreconditions(no_device) ==
          ZeroCopyPrecondition::kDeviceLimit);
  }
  SECTION("forced rejection outranks an otherwise valid request") {
    ZeroCopyContractInputs in = GoodInputs();
    in.force_rejection = true;
    CHECK(EvaluateZeroCopyPreconditions(in) ==
          ZeroCopyPrecondition::kForcedRejection);

    // It is a control arm, not a diagnosis: it must not relabel a request that
    // was already invalid for a real reason.
    ZeroCopyContractInputs also_bad = in;
    also_bad.base_address = 0;
    CHECK(EvaluateZeroCopyPreconditions(also_bad) ==
          ZeroCopyPrecondition::kForcedRejection);
  }
  SECTION("a request may fail several ways; the reported reason is stable") {
    ZeroCopyContractInputs in;  // all-default: null, empty, no page size
    CHECK(EvaluateZeroCopyPreconditions(in) ==
          ZeroCopyPrecondition::kNullPointer);
  }
}

TEST_CASE("Zero-copy postcondition: contents identity", "[l0][metal][zerocopy]") {
  uint8_t bytes[2] = {0, 0};
  void* base = &bytes[0];
  void* elsewhere = &bytes[1];

  CHECK(ZeroCopyContentsAreAlias(base, base));

  // A buffer that came back pointing at other memory is a copy, not an alias.
  CHECK_FALSE(ZeroCopyContentsAreAlias(elsewhere, base));
  // Null on either side is never an alias -- in particular, two nulls must not
  // compare "equal" into an accepted wrap.
  CHECK_FALSE(ZeroCopyContentsAreAlias(nullptr, base));
  CHECK_FALSE(ZeroCopyContentsAreAlias(base, nullptr));
  CHECK_FALSE(ZeroCopyContentsAreAlias(nullptr, nullptr));
}

TEST_CASE("Zero-copy self-copy: suppressing a copy needs identity, not a mode",
          "[l0][metal][zerocopy][convergence]") {
  // Task 4.2's guard, enumerated as a value because a host that ACCEPTED the
  // alias cannot reach the disagreement: there, contents + base is
  // TranslatePhysical(base) for every in-domain base, so no device arm can show
  // the identity conjunct is load-bearing. It is exhaustive over the four
  // (alias_installed x identity) combinations plus the null pairs.
  uint8_t bytes[2] = {0, 0};
  void* here = &bytes[0];
  void* elsewhere = &bytes[1];

  // The ONLY licensed skip: the mode says alias AND the destination IS the
  // source.
  CHECK(ZeroCopySelfCopyIsProven(true, here, here));

  // Mirror mode: two distinct allocations, so the copy is a real publication
  // even where the addresses happen to be presented as equal.
  CHECK_FALSE(ZeroCopySelfCopyIsProven(false, here, here));
  CHECK_FALSE(ZeroCopySelfCopyIsProven(false, elsewhere, here));

  // A CLAIMED alias whose addresses disagree. Skipping here would silently drop
  // the publication, so the rule must reject and let the caller fail closed.
  CHECK_FALSE(ZeroCopySelfCopyIsProven(true, elsewhere, here));
  CHECK_FALSE(ZeroCopySelfCopyIsProven(true, here, elsewhere));

  // Null on either side is never proof, and two nulls must not compare "equal"
  // into a suppressed copy.
  CHECK_FALSE(ZeroCopySelfCopyIsProven(true, nullptr, here));
  CHECK_FALSE(ZeroCopySelfCopyIsProven(true, here, nullptr));
  CHECK_FALSE(ZeroCopySelfCopyIsProven(true, nullptr, nullptr));

  // Usable at compile time, like the rest of the contract.
  static_assert(ZeroCopySelfCopyIsProven(true, &kIdentityProbe,
                                         &kIdentityProbe));
  static_assert(!ZeroCopySelfCopyIsProven(false, &kIdentityProbe,
                                          &kIdentityProbe));
}

TEST_CASE("Zero-copy observation: the seven startup verdicts",
          "[l0][metal][zerocopy]") {
  SECTION("not requested") {
    ZeroCopyObservation o;  // untouched
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kNotRequested);
  }
  SECTION("precondition rejected") {
    ZeroCopyObservation o;
    o.requested = true;
    o.precondition = ZeroCopyPrecondition::kPointerAlignment;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kPreconditionRejected);

    // The forced-fallback arm lands in the same verdict class.
    o.precondition = ZeroCopyPrecondition::kForcedRejection;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kPreconditionRejected);
  }
  SECTION("VM-region rejected") {
    ZeroCopyObservation o;
    o.requested = true;
    o.precondition = ZeroCopyPrecondition::kSatisfied;
    o.device_step = ZeroCopyDeviceStep::kVmRegionRejected;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kVmRegionRejected);
  }
  SECTION("allocation rejected") {
    ZeroCopyObservation o;
    o.requested = true;
    o.precondition = ZeroCopyPrecondition::kSatisfied;
    o.device_step = ZeroCopyDeviceStep::kAllocationRejected;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kAllocationRejected);
  }
  SECTION("contents-identity rejected") {
    ZeroCopyObservation o;
    o.requested = true;
    o.precondition = ZeroCopyPrecondition::kSatisfied;
    o.device_step = ZeroCopyDeviceStep::kContentsRejected;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kContentsIdentityRejected);
  }
  SECTION("observer rejected") {
    // The device said YES and the backend still refused: a real alias that no
    // in-flight CPU rewrite observer could be armed over is released, because
    // an unwatched alias makes "no overlap was reported" mean "nobody was
    // looking". It gets its own verdict, distinct from every device rejection
    // and from the accepted one.
    ZeroCopyObservation o;
    o.requested = true;
    o.precondition = ZeroCopyPrecondition::kSatisfied;
    o.device_step = ZeroCopyDeviceStep::kObserverRejected;
    o.alias_buffer_installed = false;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kObserverRejected);
    CHECK(DeriveZeroCopyStartupReport(o) !=
          ZeroCopyStartupReport::kContentsIdentityRejected);
    CHECK(DeriveZeroCopyStartupReport(o) !=
          ZeroCopyStartupReport::kAliasActive);
    CHECK(DeriveZeroCopyStartupReport(o) !=
          ZeroCopyStartupReport::kInconsistentObservation);

    // The fallback is the WHOLE rule: rejecting the observer while keeping the
    // alias installed is the state this step exists to make impossible, and is
    // never quietly reported as a rejection.
    o.alias_buffer_installed = true;
    CHECK(DeriveZeroCopyStartupReport(o) ==
          ZeroCopyStartupReport::kInconsistentObservation);
  }
  SECTION("alias active") {
    ZeroCopyObservation o;
    o.requested = true;
    o.precondition = ZeroCopyPrecondition::kSatisfied;
    o.device_step = ZeroCopyDeviceStep::kAccepted;
    o.alias_buffer_installed = true;
    CHECK(DeriveZeroCopyStartupReport(o) == ZeroCopyStartupReport::kAliasActive);
  }
}

TEST_CASE("Zero-copy observation: the attempt cannot be its own verdict",
          "[l0][metal][zerocopy]") {
  // Accepted by the attempt, but the installed backing is not guest RAM. If the
  // report were derived from the attempt alone this would read "alias_active"
  // -- a run claiming zero-copy while the mirror is what the GPU reads.
  ZeroCopyObservation claimed;
  claimed.requested = true;
  claimed.precondition = ZeroCopyPrecondition::kSatisfied;
  claimed.device_step = ZeroCopyDeviceStep::kAccepted;
  claimed.alias_buffer_installed = false;
  CHECK(DeriveZeroCopyStartupReport(claimed) ==
        ZeroCopyStartupReport::kInconsistentObservation);

  // The mirror image: guest RAM is installed but no attempt says why.
  ZeroCopyObservation unexplained;
  unexplained.requested = true;
  unexplained.precondition = ZeroCopyPrecondition::kSatisfied;
  unexplained.device_step = ZeroCopyDeviceStep::kAllocationRejected;
  unexplained.alias_buffer_installed = true;
  CHECK(DeriveZeroCopyStartupReport(unexplained) ==
        ZeroCopyStartupReport::kInconsistentObservation);

  // Requested, preconditions held, nothing attempted.
  ZeroCopyObservation skipped;
  skipped.requested = true;
  skipped.precondition = ZeroCopyPrecondition::kSatisfied;
  CHECK(DeriveZeroCopyStartupReport(skipped) ==
        ZeroCopyStartupReport::kInconsistentObservation);

  // Requested but the contract was never evaluated at all.
  ZeroCopyObservation unevaluated;
  unevaluated.requested = true;
  CHECK(DeriveZeroCopyStartupReport(unevaluated) ==
        ZeroCopyStartupReport::kInconsistentObservation);

  // An exact-off run that nonetheless performed alias work: every field that
  // could only be set by the alias path is a violation on its own.
  for (ZeroCopyPrecondition precondition : kAllPreconditions) {
    for (ZeroCopyDeviceStep step : kAllDeviceSteps) {
      for (bool installed : {false, true}) {
        if (precondition == ZeroCopyPrecondition::kNotEvaluated &&
            step == ZeroCopyDeviceStep::kNotAttempted && !installed) {
          continue;  // the legitimately untouched observation
        }
        ZeroCopyObservation o;
        o.requested = false;
        o.precondition = precondition;
        o.device_step = step;
        o.alias_buffer_installed = installed;
        INFO("precondition " << ZeroCopyPreconditionName(precondition)
                             << ", step " << ZeroCopyDeviceStepName(step)
                             << ", installed " << installed);
        CHECK(DeriveZeroCopyStartupReport(o) ==
              ZeroCopyStartupReport::kInconsistentObservation);
      }
    }
  }
}

TEST_CASE("Zero-copy observation: exhaustive fact cross product",
          "[l0][metal][zerocopy]") {
  // 2 x 10 x 6 x 2 = 240 observations, every one asserted. Also confirms that
  // exactly one combination yields kAliasActive: aliasing must be reachable by
  // precisely one path through the recorded facts.
  uint32_t total = 0;
  uint32_t alias_active = 0;
  uint32_t not_requested = 0;
  for (bool requested : {false, true}) {
    for (ZeroCopyPrecondition precondition : kAllPreconditions) {
      for (ZeroCopyDeviceStep step : kAllDeviceSteps) {
        for (bool installed : {false, true}) {
          ZeroCopyObservation o;
          o.requested = requested;
          o.precondition = precondition;
          o.device_step = step;
          o.alias_buffer_installed = installed;
          INFO("requested " << requested << ", precondition "
                            << ZeroCopyPreconditionName(precondition)
                            << ", step " << ZeroCopyDeviceStepName(step)
                            << ", installed " << installed);
          ZeroCopyStartupReport report = DeriveZeroCopyStartupReport(o);
          CHECK(report == ExpectedReport(o));
          ++total;
          if (report == ZeroCopyStartupReport::kAliasActive) {
            ++alias_active;
          }
          if (report == ZeroCopyStartupReport::kNotRequested) {
            ++not_requested;
          }
        }
      }
    }
  }
  CHECK(total == 240);
  CHECK(alias_active == 1);
  CHECK(not_requested == 1);
}

TEST_CASE("Zero-copy vocabulary: enums are complete and distinctly named",
          "[l0][metal][zerocopy]") {
  // Completeness is pinned at compile time by the IsExhaustive static_asserts
  // on the arrays above -- adding an enumerator before kCount without listing
  // it there fails the build. The counts are derived from kCount, so what is
  // still worth asserting here is that they agree with the arrays actually
  // swept by every case in this file.
  static_assert(kAllPreconditions.size() == kZeroCopyPreconditionCount);
  static_assert(kAllDeviceSteps.size() == kZeroCopyDeviceStepCount);
  static_assert(kAllReports.size() == kZeroCopyStartupReportCount);

  // kCount is a bound, not a value: it is invalid, and so is anything past it.
  static_assert(!IsZeroCopyPreconditionValue(ZeroCopyPrecondition::kCount));
  static_assert(!IsZeroCopyDeviceStepValue(ZeroCopyDeviceStep::kCount));
  static_assert(!IsZeroCopyStartupReportValue(ZeroCopyStartupReport::kCount));
  static_assert(!IsZeroCopyPreconditionValue(static_cast<ZeroCopyPrecondition>(
      kZeroCopyPreconditionCount + 7)));
  static_assert(!IsZeroCopyDeviceStepValue(
      static_cast<ZeroCopyDeviceStep>(kZeroCopyDeviceStepCount + 7)));
  static_assert(!IsZeroCopyStartupReportValue(
      static_cast<ZeroCopyStartupReport>(kZeroCopyStartupReportCount + 7)));
  for (ZeroCopyPrecondition v : kAllPreconditions) {
    CHECK(IsZeroCopyPreconditionValue(v));
  }
  for (ZeroCopyDeviceStep v : kAllDeviceSteps) {
    CHECK(IsZeroCopyDeviceStepValue(v));
  }
  for (ZeroCopyStartupReport v : kAllReports) {
    CHECK(IsZeroCopyStartupReportValue(v));
  }

  // Every declared value maps to a nonempty name that is not the out-of-range
  // "unknown" bucket. Names may repeat ACROSS enums (a VM-region step and a
  // VM-region report describe the same fact) but never within one.
  for (ZeroCopyPrecondition v : kAllPreconditions) {
    std::string name = ZeroCopyPreconditionName(v);
    CHECK(!name.empty());
    CHECK(name != "unknown");
  }
  for (ZeroCopyDeviceStep v : kAllDeviceSteps) {
    std::string name = ZeroCopyDeviceStepName(v);
    CHECK(!name.empty());
    CHECK(name != "unknown");
  }
  for (ZeroCopyStartupReport v : kAllReports) {
    std::string name = ZeroCopyStartupReportName(v);
    CHECK(!name.empty());
    CHECK(name != "unknown");
  }

  auto unique_within = [](const std::vector<std::string>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
      for (size_t j = i + 1; j < v.size(); ++j) {
        if (v[i] == v[j]) {
          return false;
        }
      }
    }
    return true;
  };
  std::vector<std::string> precondition_names;
  for (ZeroCopyPrecondition v : kAllPreconditions) {
    precondition_names.push_back(ZeroCopyPreconditionName(v));
  }
  std::vector<std::string> step_names;
  for (ZeroCopyDeviceStep v : kAllDeviceSteps) {
    step_names.push_back(ZeroCopyDeviceStepName(v));
  }
  std::vector<std::string> report_names;
  for (ZeroCopyStartupReport v : kAllReports) {
    report_names.push_back(ZeroCopyStartupReportName(v));
  }
  CHECK(unique_within(precondition_names));
  CHECK(unique_within(step_names));
  CHECK(unique_within(report_names));

  // The kCount sentinel and any value past it reach the "unknown" bucket
  // rather than aliasing a real name (the name helpers handle kCount by
  // breaking to the fall-through return, and have no default case).
  for (uint32_t past : {uint32_t(0), uint32_t(1), uint32_t(7)}) {
    INFO("kCount + " << past);
    CHECK(std::string(ZeroCopyPreconditionName(
              static_cast<ZeroCopyPrecondition>(kZeroCopyPreconditionCount +
                                                past))) == "unknown");
    CHECK(std::string(ZeroCopyDeviceStepName(static_cast<ZeroCopyDeviceStep>(
              kZeroCopyDeviceStepCount + past))) == "unknown");
    CHECK(std::string(ZeroCopyStartupReportName(
              static_cast<ZeroCopyStartupReport>(kZeroCopyStartupReportCount +
                                                 past))) == "unknown");
  }

  // A sentinel or out-of-range recorded fact must NOT be reported as a
  // success, on either axis, even when every other fact spells out the one
  // combination that would otherwise read "alias_active".
  for (uint32_t past : {uint32_t(0), uint32_t(1), uint32_t(7)}) {
    INFO("kCount + " << past);
    ZeroCopyObservation bad_step;
    bad_step.requested = true;
    bad_step.precondition = ZeroCopyPrecondition::kSatisfied;
    bad_step.device_step =
        static_cast<ZeroCopyDeviceStep>(kZeroCopyDeviceStepCount + past);
    bad_step.alias_buffer_installed = true;
    CHECK(DeriveZeroCopyStartupReport(bad_step) ==
          ZeroCopyStartupReport::kInconsistentObservation);

    ZeroCopyObservation bad_precondition;
    bad_precondition.requested = true;
    bad_precondition.precondition = static_cast<ZeroCopyPrecondition>(
        kZeroCopyPreconditionCount + past);
    bad_precondition.device_step = ZeroCopyDeviceStep::kAccepted;
    bad_precondition.alias_buffer_installed = true;
    CHECK(DeriveZeroCopyStartupReport(bad_precondition) ==
          ZeroCopyStartupReport::kInconsistentObservation);

    // Not-requested runs get the same treatment: an undefined recorded fact is
    // never the untouched observation kNotRequested describes.
    ZeroCopyObservation unrequested;
    unrequested.device_step =
        static_cast<ZeroCopyDeviceStep>(kZeroCopyDeviceStepCount + past);
    CHECK(DeriveZeroCopyStartupReport(unrequested) ==
          ZeroCopyStartupReport::kInconsistentObservation);
  }

  // Whatever the report derivation returns is itself always a declared value.
  CHECK(IsZeroCopyStartupReportValue(
      DeriveZeroCopyStartupReport(ZeroCopyObservation())));
}

TEST_CASE("Zero-copy contract: the evaluator is usable at compile time",
          "[l0][metal][zerocopy]") {
  // constexpr-ness is what lets the production initializer keep these checks
  // as plain value code with no runtime cost, and what keeps them free of any
  // device or environment dependency.
  constexpr ZeroCopyContractInputs kAccepted = {0x1000, 0x1000, 0x1000, 0x10000,
                                                false};
  static_assert(EvaluateZeroCopyPreconditions(kAccepted) ==
                ZeroCopyPrecondition::kSatisfied);
  constexpr ZeroCopyContractInputs kMisaligned = {0x1001, 0x1000, 0x1000,
                                                 0x10000, false};
  static_assert(EvaluateZeroCopyPreconditions(kMisaligned) ==
                ZeroCopyPrecondition::kPointerAlignment);
  static_assert(IsZeroCopyGateEnabled("1"));
  static_assert(!IsZeroCopyGateEnabled("10"));
  SUCCEED();
}

// ---------------------------------------------------------------------------
// The memexport RAM-convergence gate (task 4.2).
//
// Untagged records (submission 0) are the DEFAULT on this backend, not a
// diagnostic: S7LifetimeOff() is true unless XE_METAL_ASYNC is set, S7TagsOff()
// includes it, and the publication seam registers `S7TagsOff() ? 0 : ...`. So
// the ordinary launch registers every memexport with submission 0, and their
// finality comes from the synchronous commit-and-wait the consumer path
// performs. These cases pin both halves: a tagged record is decided by its own
// submission and NOTHING else, and an untagged record is publishable only under
// explicit caller-supplied synchronous-boundary evidence.
// ---------------------------------------------------------------------------

namespace {

constexpr std::array<MemexportConvergenceGate, kMemexportConvergenceGateCount>
    kAllConvergenceGates = {
        MemexportConvergenceGate::kNotAttemptedIncompleteSubmission,
        MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary,
        MemexportConvergenceGate::kAttemptedCompletedSubmission,
        MemexportConvergenceGate::kAttemptedSynchronousBoundary,
};
static_assert(IsExhaustive(kAllConvergenceGates));
// The production array must itself be complete and contiguous -- it is what the
// production switches and this sweep are both pinned to, so a same-length but
// reordered list would otherwise pass.
static_assert(kMemexportConvergenceGates.size() ==
              kMemexportConvergenceGateCount);
static_assert(IsExhaustive(kMemexportConvergenceGates));

constexpr std::array<MemexportBoundaryEvidence, kMemexportBoundaryEvidenceCount>
    kAllBoundaryEvidence = {
        MemexportBoundaryEvidence::kNone,
        MemexportBoundaryEvidence::kSynchronousBoundaryCompleted,
};
static_assert(IsExhaustive(kAllBoundaryEvidence));
static_assert(kMemexportBoundaryEvidences.size() ==
              kMemexportBoundaryEvidenceCount);
static_assert(IsExhaustive(kMemexportBoundaryEvidences));

// Ordering is the claim, so both the observation and the publication append to
// ONE log. Two independent counters could not tell "before" from "after".
enum class ConvergenceEvent { kPreAttempt, kTarget };

class RecordingConvergenceTarget final : public MemexportConvergenceTarget {
 public:
  explicit RecordingConvergenceTarget(std::vector<ConvergenceEvent>* log)
      : log_(log) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries,
                       uint64_t* bytes_copied) override {
    ++calls;
    log_->push_back(ConvergenceEvent::kTarget);
    *invalidated_entries = 7;
    *bytes_copied = length;
    return succeed;
  }

  uint32_t calls = 0;
  bool succeed = true;

 private:
  std::vector<ConvergenceEvent>* log_;
};

PendingProducer GateProducer(uint64_t submission) {
  std::optional<GuestRange> range =
      GuestRange::FromExactBaseLength(0x2000u, 0x1000u);
  REQUIRE(range.has_value());
  return PendingProducer{*range, TransitState::kPendingMemexport, submission,
                         /*sequence=*/1u};
}

}  // namespace

TEST_CASE("Memexport gate vocabulary: complete, contiguous, distinctly named, "
          "and never attempted outside its own values",
          "[l0][metal][zerocopy][convergence]") {
  // Every real value names itself, and no real name is the out-of-range bucket.
  std::vector<std::string> gate_names;
  for (MemexportConvergenceGate gate : kAllConvergenceGates) {
    CHECK(IsMemexportConvergenceGateValue(gate));
    const std::string name = MemexportConvergenceGateName(gate);
    CHECK(name != "unknown");
    gate_names.push_back(name);
  }
  CHECK(gate_names ==
        std::vector<std::string>{"not_attempted_incomplete_submission",
                                 "not_attempted_untagged_without_boundary",
                                 "attempted_completed_submission",
                                 "attempted_synchronous_boundary"});

  // Exactly which values are attempts, pinned by name rather than by position.
  CHECK_FALSE(MemexportConvergenceGateAttempts(
      MemexportConvergenceGate::kNotAttemptedIncompleteSubmission));
  CHECK_FALSE(MemexportConvergenceGateAttempts(
      MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary));
  CHECK(MemexportConvergenceGateAttempts(
      MemexportConvergenceGate::kAttemptedCompletedSubmission));
  CHECK(MemexportConvergenceGateAttempts(
      MemexportConvergenceGate::kAttemptedSynchronousBoundary));

  // kCount and anything past it are not values: unnamed, and never an attempt.
  // A publication licensed by a value the vocabulary does not define would be a
  // publication nothing decided.
  CHECK_FALSE(
      IsMemexportConvergenceGateValue(MemexportConvergenceGate::kCount));
  CHECK(std::string(MemexportConvergenceGateName(
            MemexportConvergenceGate::kCount)) == "unknown");
  CHECK_FALSE(
      MemexportConvergenceGateAttempts(MemexportConvergenceGate::kCount));
  for (uint32_t past = kMemexportConvergenceGateCount;
       past < kMemexportConvergenceGateCount + 4; ++past) {
    const auto out_of_range = static_cast<MemexportConvergenceGate>(past);
    CHECK_FALSE(IsMemexportConvergenceGateValue(out_of_range));
    CHECK(std::string(MemexportConvergenceGateName(out_of_range)) == "unknown");
    CHECK_FALSE(MemexportConvergenceGateAttempts(out_of_range));
    GatedMemexportConvergence forged;
    forged.gate = out_of_range;
    forged.result.converged = true;  // even a "successful" result cannot retire
    CHECK_FALSE(forged.attempted());
    CHECK_FALSE(forged.retires());
  }

  // The evidence vocabulary, to the same standard.
  std::vector<std::string> evidence_names;
  for (MemexportBoundaryEvidence evidence : kAllBoundaryEvidence) {
    CHECK(IsMemexportBoundaryEvidenceValue(evidence));
    evidence_names.push_back(MemexportBoundaryEvidenceName(evidence));
  }
  CHECK(evidence_names ==
        std::vector<std::string>{"none", "synchronous_boundary_completed"});
  CHECK_FALSE(
      IsMemexportBoundaryEvidenceValue(MemexportBoundaryEvidence::kCount));
  CHECK(std::string(MemexportBoundaryEvidenceName(
            MemexportBoundaryEvidence::kCount)) == "unknown");
  CHECK(std::string(MemexportBoundaryEvidenceName(
            static_cast<MemexportBoundaryEvidence>(
                kMemexportBoundaryEvidenceCount + 1))) == "unknown");

  // A default-constructed gated result is a NON-attempt, so a caller that never
  // ran the gate cannot read it as a licensed publication.
  GatedMemexportConvergence unrun;
  CHECK_FALSE(unrun.attempted());
  CHECK_FALSE(unrun.retires());
  CHECK(IsMemexportConvergenceGateValue(unrun.gate));
}

TEST_CASE("Memexport gate: a tagged record is decided by its own submission, "
          "an untagged one only by caller evidence",
          "[l0][metal][zerocopy][convergence]") {
  for (MemexportBoundaryEvidence evidence : kAllBoundaryEvidence) {
    INFO("evidence " << MemexportBoundaryEvidenceName(evidence));

    // Tagged and complete: attempted, and NOT because of the evidence -- the
    // same verdict holds under both evidence values.
    CHECK(DecideMemexportConvergenceGate(GateProducer(4u), 4u, evidence) ==
          MemexportConvergenceGate::kAttemptedCompletedSubmission);
    CHECK(DecideMemexportConvergenceGate(GateProducer(4u), 9u, evidence) ==
          MemexportConvergenceGate::kAttemptedCompletedSubmission);

    // Tagged and incomplete: NOT attempted, and boundary evidence must not be
    // able to talk it past its own high-water mark.
    CHECK(DecideMemexportConvergenceGate(GateProducer(9u), 8u, evidence) ==
          MemexportConvergenceGate::kNotAttemptedIncompleteSubmission);
    CHECK(DecideMemexportConvergenceGate(GateProducer(1u), 0u, evidence) ==
          MemexportConvergenceGate::kNotAttemptedIncompleteSubmission);
  }

  // Untagged -- the DEFAULT record shape. Without evidence it is unproven, no
  // matter how high the completed high-water mark is: a mark says nothing about
  // a record that never named a submission.
  for (uint64_t completed : {0u, 1u, 1000u}) {
    CHECK(DecideMemexportConvergenceGate(GateProducer(0u), completed,
                                         MemexportBoundaryEvidence::kNone) ==
          MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary);
    CHECK(DecideMemexportConvergenceGate(
              GateProducer(0u), completed,
              MemexportBoundaryEvidence::kSynchronousBoundaryCompleted) ==
          MemexportConvergenceGate::kAttemptedSynchronousBoundary);
  }

  // An evidence value the vocabulary does not define is not the boundary proof.
  CHECK(DecideMemexportConvergenceGate(
            GateProducer(0u), 5u,
            static_cast<MemexportBoundaryEvidence>(
                kMemexportBoundaryEvidenceCount + 3)) ==
        MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary);

  // ProducerComplete itself is untouched: it never reads 0 as completed, which
  // is why the untagged proof has to arrive from outside instead of being
  // inferred inside the generic tagged predicate.
  CHECK_FALSE(ProducerComplete(GateProducer(0u), 0u));
  CHECK_FALSE(ProducerComplete(GateProducer(0u), 1000u));
  CHECK(ProducerComplete(GateProducer(4u), 4u));
}

TEST_CASE("Memexport convergence: the target is reached on exactly the two "
          "proven paths and on neither unproven one",
          "[l0][metal][zerocopy][convergence]") {
  // Untagged, no evidence: nothing is published and nothing is read.
  {
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
        GateProducer(0u), /*completed=*/100u, MemexportBoundaryEvidence::kNone,
        target);
    CHECK(gated.gate ==
          MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary);
    CHECK_FALSE(gated.attempted());
    CHECK_FALSE(gated.retires());
    CHECK(target.calls == 0u);
    CHECK(gated.result.bytes_copied == 0u);
    CHECK(gated.result.action == AuthorityAction::kNoDependency);
  }

  // Tagged, incomplete: same, decided by the tag.
  {
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
        GateProducer(9u), /*completed=*/8u,
        MemexportBoundaryEvidence::kSynchronousBoundaryCompleted, target);
    CHECK(gated.gate ==
          MemexportConvergenceGate::kNotAttemptedIncompleteSubmission);
    CHECK(target.calls == 0u);
    CHECK_FALSE(gated.retires());
  }

  // Untagged WITH the boundary proof: called exactly once, and it retires.
  {
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
        GateProducer(0u), /*completed=*/0u,
        MemexportBoundaryEvidence::kSynchronousBoundaryCompleted, target);
    CHECK(gated.gate ==
          MemexportConvergenceGate::kAttemptedSynchronousBoundary);
    CHECK(gated.attempted());
    CHECK(gated.retires());
    CHECK(target.calls == 1u);
    CHECK(gated.result.bytes_copied == 0x1000u);
    CHECK(gated.result.invalidated_entries == 7u);
    CHECK(gated.result.action == AuthorityAction::kConvergeRamFromBuffer);
  }

  // Tagged and complete: called exactly once, and it retires.
  {
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
        GateProducer(4u), /*completed=*/4u, MemexportBoundaryEvidence::kNone,
        target);
    CHECK(gated.gate ==
          MemexportConvergenceGate::kAttemptedCompletedSubmission);
    CHECK(gated.retires());
    CHECK(target.calls == 1u);
  }

  // The three-argument convenience overload supplies kNone, so it is SAFE by
  // default: an untagged record cannot be published through it.
  {
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    GatedMemexportConvergence gated =
        ConvergeCompletedMemexportRam(GateProducer(0u), 100u, target);
    CHECK(gated.gate ==
          MemexportConvergenceGate::kNotAttemptedUntaggedWithoutBoundary);
    CHECK(target.calls == 0u);
    // ...and it still publishes a tagged, completed record.
    GatedMemexportConvergence tagged =
        ConvergeCompletedMemexportRam(GateProducer(4u), 4u, target);
    CHECK(tagged.retires());
    CHECK(target.calls == 1u);
  }

  // An attempted convergence whose target FAILS does not retire, and the
  // failure is distinguishable from never having been attempted.
  {
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    target.succeed = false;
    GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
        GateProducer(0u), 0u,
        MemexportBoundaryEvidence::kSynchronousBoundaryCompleted, target);
    CHECK(gated.attempted());
    CHECK_FALSE(gated.result.converged);
    CHECK_FALSE(gated.retires());
    CHECK(target.calls == 1u);
  }
}

TEST_CASE("Memexport convergence: the pre-attempt observation runs exactly "
          "once, only on a proven path, and strictly before publication",
          "[l0][metal][zerocopy][convergence]") {
  // The observation exists to read the GPU-written export bytes. Reading them
  // for a record whose finality is unproven would sample an export that may
  // still be in flight and report it as a failed store, so the two unproven
  // paths must not invoke it AT ALL -- not merely discard its result.
  struct Arm {
    const char* name;
    uint64_t submission;
    uint64_t completed;
    MemexportBoundaryEvidence evidence;
    bool attempts;
  };
  const Arm arms[] = {
      {"untagged without boundary", 0u, 100u, MemexportBoundaryEvidence::kNone,
       false},
      {"tagged incomplete", 9u, 8u,
       MemexportBoundaryEvidence::kSynchronousBoundaryCompleted, false},
      {"untagged with boundary", 0u, 0u,
       MemexportBoundaryEvidence::kSynchronousBoundaryCompleted, true},
      {"tagged complete", 4u, 4u, MemexportBoundaryEvidence::kNone, true},
  };

  for (const Arm& arm : arms) {
    INFO(arm.name);
    std::vector<ConvergenceEvent> log;
    RecordingConvergenceTarget target(&log);
    uint32_t pre_attempt_calls = 0;
    GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
        GateProducer(arm.submission), arm.completed, arm.evidence, target,
        [&]() {
          ++pre_attempt_calls;
          log.push_back(ConvergenceEvent::kPreAttempt);
        });
    CHECK(gated.attempted() == arm.attempts);
    if (!arm.attempts) {
      CHECK(pre_attempt_calls == 0u);
      CHECK(target.calls == 0u);
      CHECK(log.empty());
      continue;
    }
    CHECK(pre_attempt_calls == 1u);
    CHECK(target.calls == 1u);
    // Order, not just presence: the observation must precede publication, or it
    // would be describing bytes the publication had already moved.
    CHECK(log == std::vector<ConvergenceEvent>{ConvergenceEvent::kPreAttempt,
                                               ConvergenceEvent::kTarget});
  }

  // A failing target does not change the count or the order -- the observation
  // is licensed by the GATE, not by the outcome.
  std::vector<ConvergenceEvent> log;
  RecordingConvergenceTarget failing(&log);
  failing.succeed = false;
  uint32_t pre_attempt_calls = 0;
  GatedMemexportConvergence gated = ConvergeCompletedMemexportRam(
      GateProducer(4u), 4u, MemexportBoundaryEvidence::kNone, failing, [&]() {
        ++pre_attempt_calls;
        log.push_back(ConvergenceEvent::kPreAttempt);
      });
  CHECK_FALSE(gated.retires());
  CHECK(pre_attempt_calls == 1u);
  CHECK(log == std::vector<ConvergenceEvent>{ConvergenceEvent::kPreAttempt,
                                             ConvergenceEvent::kTarget});
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
