/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (device-free) conformance for the cross-writer memory-authority
// decision seam (OpenSpec implement-metal-cross-writer-memory-authority, tasks
// 2.3 and 2.4). Exercises the pure decision functions and the observer-only
// telemetry against the full writer/consumer/action matrix:
//   * every ConsumerKind x pending-producer arrangement -> AuthorityAction;
//   * multiple pending records with selective (intersecting-only) selection;
//   * aliased / disjoint base+mip range sets;
//   * submission ordering (await vs drain by completed-submission high-water);
//   * disjoint no-op (no commit, wait, or drain);
//   * writer publication policy;
//   * typed observer agreement/disagreement accounting.
// Pure C++ + Catch2; links no Metal device and no command processor.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memory_authority.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace test {

namespace {

PendingProducer MakeProducer(uint32_t base, uint32_t size, TransitState state,
                             uint64_t submission, uint64_t sequence = 0) {
  return PendingProducer{*GuestRange::FromBaseLength(base, size), state,
                         submission, sequence};
}

GuestRangeSet SingleRange(uint32_t base, uint32_t size) {
  GuestRangeSet set;
  set.Add(GuestRange::FromBaseLength(base, size));
  return set;
}

class RecordingRamWriteTarget : public RamWritePublicationTarget {
 public:
  bool InvalidateSharedPages(uint32_t base, uint32_t length,
                             bool exact_range) override {
    ++shared_calls;
    shared_base = base;
    shared_length = length;
    shared_exact = exact_range;
    return shared_result;
  }

  bool InvalidateLegacyTextures(uint32_t base, uint32_t length,
                                uint32_t* invalidated_entries) override {
    ++legacy_calls;
    legacy_base = base;
    legacy_length = length;
    *invalidated_entries = legacy_entries;
    return legacy_result;
  }

  uint32_t shared_calls = 0;
  uint32_t shared_base = 0;
  uint32_t shared_length = 0;
  bool shared_exact = false;
  bool shared_result = true;
  uint32_t legacy_calls = 0;
  uint32_t legacy_base = 0;
  uint32_t legacy_length = 0;
  bool legacy_result = true;
  uint32_t legacy_entries = 3;
};

}  // namespace

TEST_CASE("Authority: disjoint consumer selects no dependency", "[l0][authority]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 5),
      MakeProducer(0x8000u, 0x1000u, TransitState::kPendingMemexport, 5),
  };
  // Consumer in the gap [0x2000,0x8000): disjoint from both.
  auto decision = DecideConsumer(ConsumerKind::kVertex,
                                 SingleRange(0x4000u, 0x400u), producers,
                                 /*completed=*/5);
  REQUIRE(decision.action == AuthorityAction::kNoDependency);
  REQUIRE(decision.selected.empty());
  REQUIRE_FALSE(decision.await);
  REQUIRE_FALSE(decision.drain);
  REQUIRE(decision.selected_bytes() == 0u);
}

TEST_CASE("Authority: adjacency is not a dependency", "[l0][authority]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 3),
  };
  // Consumer begins exactly where the producer ends: touching, zero bytes.
  auto decision = DecideConsumer(ConsumerKind::kTexture,
                                 SingleRange(0x2000u, 0x1000u), producers,
                                 /*completed=*/3);
  REQUIRE(decision.action == AuthorityAction::kNoDependency);
  REQUIRE(decision.selected.empty());
}

TEST_CASE("Authority: complete overlapping producer drains matching records",
          "[l0][authority]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x2000u, TransitState::kPendingResolve, 4),
  };
  auto decision = DecideConsumer(ConsumerKind::kVertex,
                                 SingleRange(0x1800u, 0x400u), producers,
                                 /*completed=*/4);  // submission 4 finished.
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(decision.selected.size() == 1u);
  REQUIRE_FALSE(decision.await);
  REQUIRE(decision.drain);
  REQUIRE(decision.revalidate_gpu_bytes);  // vertex reads the shared buffer.
  REQUIRE(decision.max_wait_submission == 0u);
}

TEST_CASE("Authority: incomplete overlapping producer awaits its submission",
          "[l0][authority]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x2000u, TransitState::kPendingMemexport, 9),
  };
  auto decision = DecideConsumer(ConsumerKind::kTexture,
                                 SingleRange(0x1800u, 0x400u), producers,
                                 /*completed=*/7);  // submission 9 not done.
  REQUIRE(decision.action == AuthorityAction::kAwaitProducer);
  REQUIRE(decision.await);
  REQUIRE(decision.drain);
  REQUIRE(decision.max_wait_submission == 9u);
}

TEST_CASE("Authority: submission zero is conservatively incomplete",
          "[l0][authority]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 0),
  };
  auto decision = DecideConsumer(ConsumerKind::kWait,
                                 SingleRange(0x1000u, 4u), producers,
                                 /*completed=*/0);
  REQUIRE(decision.action == AuthorityAction::kAwaitProducer);
  REQUIRE(decision.await);
  REQUIRE(decision.max_wait_submission == 0u);
}

TEST_CASE("Authority: selective drain across multiple pending records",
          "[l0][authority]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 2),
      MakeProducer(0x3000u, 0x1000u, TransitState::kPendingMemexport, 2),
      MakeProducer(0x9000u, 0x1000u, TransitState::kPendingResolve, 2),
  };
  // Consumer overlaps the first two but not the third disjoint record.
  auto decision = DecideConsumer(ConsumerKind::kWait,
                                 SingleRange(0x1800u, 0x2000u), producers,
                                 /*completed=*/2);
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(decision.selected.size() == 2u);
  REQUIRE(decision.selected[0].range.start() == 0x1000u);
  REQUIRE(decision.selected[1].range.start() == 0x3000u);
  REQUIRE_FALSE(decision.revalidate_gpu_bytes);  // wait reads guest RAM.
  // The third record is disjoint and must never be drained by this consumer.
  REQUIRE(decision.selected_bytes() == 0x1000u + 0x1000u);
}

TEST_CASE("Authority: wait word leaves adjacent and disjoint producers pending",
          "[l0][authority][cpu-read]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 4u, TransitState::kPendingResolve, 3, 1),
      MakeProducer(0x1004u, 4u, TransitState::kPendingMemexport, 3, 2),
      MakeProducer(0x8000u, 4u, TransitState::kPendingResolve, 3, 3),
  };
  auto decision = DecideConsumer(ConsumerKind::kWait,
                                 SingleRange(0x1000u, 4u), producers, 3);
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(decision.selected.size() == 1u);
  REQUIRE(decision.selected[0].sequence == 1u);
  REQUIRE_FALSE(decision.revalidate_gpu_bytes);
}

TEST_CASE("Authority: PM4 RAM consumer normalizes the bit-29 physical alias",
          "[l0][authority][cpu-read][alias]") {
  constexpr uint32_t kAliasedPacketAddress = 0x20001003u;
  uint32_t physical_address =
      xenos::CpuToGpu(kAliasedPacketAddress & ~uint32_t(0x3));
  REQUIRE(physical_address == 0x1000u);

  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 4u, TransitState::kPendingMemexport, 3, 1),
      MakeProducer(0x2000u, 4u, TransitState::kPendingResolve, 3, 2),
  };
  auto decision = DecideConsumer(ConsumerKind::kWait,
                                 SingleRange(physical_address, 4u), producers,
                                 /*completed_submission=*/3);
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(decision.selected.size() == 1u);
  REQUIRE(decision.selected[0].sequence == 1u);
}

TEST_CASE("Authority: exact host spans reject clamped end-boundary reads",
          "[l0][authority][cpu-read][boundary]") {
  constexpr uint32_t kGuestEnd =
      static_cast<uint32_t>(GuestRange::kGuestPhysicalSize);
  auto exact_end = GuestRange::FromExactBaseLength(kGuestEnd - 16u, 16u);
  REQUIRE(exact_end);
  REQUIRE(exact_end->end() == kGuestEnd);

  REQUIRE_FALSE(
      GuestRange::FromExactBaseLength(kGuestEnd - 4u, 16u).has_value());
  REQUIRE_FALSE(GuestRange::FromExactBaseLength(kGuestEnd, 4u).has_value());
  REQUIRE_FALSE(GuestRange::FromExactBaseLength(0x1000u, 0u).has_value());

  uint32_t alias_physical = xenos::CpuToGpu(0x20001000u);
  auto alias_exact = GuestRange::FromExactBaseLength(alias_physical, 16u);
  REQUIRE(alias_exact);
  REQUIRE(alias_exact->start() == 0x1000u);
  REQUIRE(alias_exact->length() == 16u);
}

TEST_CASE("Authority: completion and bulk host boundary select every producer",
          "[l0][authority][cpu-read][completion]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x100u, TransitState::kPendingResolve, 7, 3),
      MakeProducer(0x9000u, 0x200u, TransitState::kPendingMemexport, 6, 1),
      MakeProducer(0x1FFFF000u, 0x1000u, TransitState::kPendingResolve, 7, 2),
  };
  auto decision = DecideConsumer(
      ConsumerKind::kHostCpuRead,
      SingleRange(0, static_cast<uint32_t>(GuestRange::kGuestPhysicalSize)),
      producers, 6);
  REQUIRE(decision.action == AuthorityAction::kAwaitProducer);
  REQUIRE(decision.selected.size() == 3u);
  REQUIRE(decision.selected[0].sequence == 1u);
  REQUIRE(decision.selected[1].sequence == 2u);
  REQUIRE(decision.selected[2].sequence == 3u);
  REQUIRE(decision.max_wait_submission == 7u);
}

TEST_CASE("Authority: mixed producers retain chronological order",
          "[l0][authority][ordering]") {
  // The physical queues may be presented grouped by kind. The seam orders the
  // selected dependencies by the sequence carried on the existing records.
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 4, 20),
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingMemexport, 4, 10),
  };
  auto decision = DecideConsumer(ConsumerKind::kHostCpuRead,
                                 SingleRange(0x1000u, 0x1000u), producers, 4);
  REQUIRE(decision.selected.size() == 2u);
  REQUIRE(decision.selected[0].state == TransitState::kPendingMemexport);
  REQUIRE(decision.selected[1].state == TransitState::kPendingResolve);
}

TEST_CASE("Authority: aliased base/mip range set selects the right producers",
          "[l0][authority][alias]") {
  // A texture supplies its precise base and mip extents as two disjoint ranges.
  GuestRangeSet requested;
  requested.Add(GuestRange::FromBaseLength(0x10000u, 0x8000u));  // base level.
  requested.Add(GuestRange::FromBaseLength(0x30000u, 0x2000u));  // mip tail.

  std::vector<PendingProducer> producers = {
      MakeProducer(0x31000u, 0x800u, TransitState::kPendingResolve, 6),  // mip.
      MakeProducer(0x20000u, 0x1000u, TransitState::kPendingMemexport, 6),  // gap.
      MakeProducer(0x14000u, 0x800u, TransitState::kPendingResolve, 6),  // base.
  };
  auto decision =
      DecideConsumer(ConsumerKind::kTexture, requested, producers,
                     /*completed=*/6);
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  // The gap producer (0x20000) is NOT selected; base + mip producers are.
  REQUIRE(decision.selected.size() == 2u);
  REQUIRE(decision.selected[0].range.start() == 0x31000u);
  REQUIRE(decision.selected[1].range.start() == 0x14000u);
  REQUIRE(decision.requested.size() == 2u);
  REQUIRE(decision.requested.CoveredBytes() == 0xA000u);
}

TEST_CASE("Authority: texture bridge excludes newer resolve but not memexport",
          "[l0][authority][bridge]") {
  ConsumerRequest request;
  // A bridge-served texture contributes no resolve-read range, but its exact
  // bytes remain a memexport dependency because memexport is not in the bridge.
  request.memexport_ranges.Add(
      GuestRange::FromBaseLength(0x4000u, 0x1000u));
  std::vector<PendingProducer> producers = {
      MakeProducer(0x4000u, 0x1000u, TransitState::kPendingMemexport, 3, 10),
      MakeProducer(0x4000u, 0x1000u, TransitState::kPendingResolve, 3, 11),
  };
  auto decision = DecideConsumer(ConsumerKind::kTexture, request, producers, 3);
  REQUIRE(decision.selected.size() == 1u);
  REQUIRE(decision.selected[0].state == TransitState::kPendingMemexport);
  REQUIRE(decision.requested.CoveredBytes() == 0x1000u);
}

TEST_CASE(
    "Authority: bridge selects older resolve before newer overlapping "
    "memexport",
    "[l0][authority][bridge][mixed-writer]") {
  ConsumerRequest request;
  request.memexport_ranges.Add(
      GuestRange::FromBaseLength(0x4000u, 0x1000u));
  std::vector<PendingProducer> producers = {
      MakeProducer(0x4000u, 0x1000u, TransitState::kPendingResolve, 4, 20),
      MakeProducer(0x4800u, 0x800u, TransitState::kPendingMemexport, 5, 21),
  };
  auto decision = DecideConsumer(ConsumerKind::kTexture, request, producers, 5);
  REQUIRE(decision.selected.size() == 2u);
  REQUIRE(decision.selected[0].state == TransitState::kPendingResolve);
  REQUIRE(decision.selected[0].sequence == 20u);
  REQUIRE(decision.selected[1].state == TransitState::kPendingMemexport);
  REQUIRE(decision.selected[1].sequence == 21u);
}

TEST_CASE("Authority: mixed-writer closure covers whole memexport record",
          "[l0][authority][bridge][mixed-writer]") {
  ConsumerRequest request;
  // The consumer reads only the memexport tail. Convergence nevertheless
  // copies the full memexport record, so the older resolve at its head must be
  // retired first or it could overwrite the newer head bytes on a later drain.
  request.memexport_ranges.Add(
      GuestRange::FromBaseLength(0x5000u, 0x100u));
  std::vector<PendingProducer> producers = {
      MakeProducer(0x4000u, 0x100u, TransitState::kPendingResolve, 6, 30),
      MakeProducer(0x4000u, 0x1100u, TransitState::kPendingMemexport, 7, 31),
  };
  auto decision = DecideConsumer(ConsumerKind::kTexture, request, producers, 7);
  REQUIRE(decision.selected.size() == 2u);
  REQUIRE(decision.selected[0].sequence == 30u);
  REQUIRE(decision.selected[1].sequence == 31u);
}

TEST_CASE("Authority: producer ordering closure is transitive across kinds",
          "[l0][authority][mixed-writer]") {
  ConsumerRequest request;
  request.memexport_ranges.Add(
      GuestRange::FromBaseLength(0x5000u, 0x100u));
  std::vector<PendingProducer> producers = {
      MakeProducer(0x4000u, 0x100u, TransitState::kPendingMemexport, 1, 40),
      MakeProducer(0x4000u, 0x2000u, TransitState::kPendingResolve, 2, 41),
      MakeProducer(0x5000u, 0x100u, TransitState::kPendingMemexport, 3, 42),
  };
  auto decision = DecideConsumer(ConsumerKind::kTexture, request, producers, 3);
  REQUIRE(decision.selected.size() == 3u);
  REQUIRE(decision.selected[0].sequence == 40u);
  REQUIRE(decision.selected[1].sequence == 41u);
  REQUIRE(decision.selected[2].sequence == 42u);
}

TEST_CASE("Authority: submission ordering N produces, M consumes",
          "[l0][authority]") {
  // Producer registered on submission N=10. A consumer on the same range while
  // completed high-water is 9 must await; once 10 completes it only drains.
  std::vector<PendingProducer> producers = {
      MakeProducer(0x5000u, 0x1000u, TransitState::kPendingMemexport, 10),
  };
  auto before = DecideConsumer(ConsumerKind::kHostCpuRead,
                               SingleRange(0x5000u, 0x1000u), producers,
                               /*completed=*/9);
  REQUIRE(before.action == AuthorityAction::kAwaitProducer);
  REQUIRE(before.max_wait_submission == 10u);

  auto after = DecideConsumer(ConsumerKind::kHostCpuRead,
                              SingleRange(0x5000u, 0x1000u), producers,
                              /*completed=*/10);
  REQUIRE(after.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE_FALSE(after.await);
}

TEST_CASE("Authority: end-of-memory consumer clamps and still selects",
          "[l0][authority]") {
  const uint32_t kEnd = uint32_t(GuestRange::kGuestPhysicalSize);
  std::vector<PendingProducer> producers = {
      MakeProducer(kEnd - 0x1000u, 0x1000u, TransitState::kPendingResolve, 1),
  };
  GuestRangeSet requested;
  requested.Add(GuestRange::FromBounds(kEnd - 0x400u, kEnd + 0x400u));  // clamped
  auto decision = DecideConsumer(ConsumerKind::kVertex, requested, producers,
                                 /*completed=*/1);
  REQUIRE(decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(decision.selected.size() == 1u);
}

TEST_CASE("Authority: writer publication policy", "[l0][authority]") {
  REQUIRE(DecideWriterPublication(WriterKind::kGuestCpu) ==
          AuthorityAction::kInvalidateRamWrite);
  REQUIRE(DecideWriterPublication(WriterKind::kResolve) ==
          AuthorityAction::kInvalidateRamWrite);
  REQUIRE(DecideWriterPublication(WriterKind::kTraceRestore) ==
          AuthorityAction::kInvalidateRamWrite);
  REQUIRE(DecideWriterPublication(WriterKind::kMemexport) ==
          AuthorityAction::kPublishGpuWrite);

  REQUIRE(WriterTransitState(WriterKind::kResolve) ==
          TransitState::kPendingResolve);
  REQUIRE(WriterTransitState(WriterKind::kMemexport) ==
          TransitState::kPendingMemexport);
  REQUIRE(WriterTransitState(WriterKind::kGuestCpu) ==
          TransitState::kRamCurrent);
  REQUIRE(WriterTransitState(WriterKind::kTraceRestore) ==
          TransitState::kRamCurrent);
}

TEST_CASE("Authority: trace restore executes normalized RAM publication",
          "[l0][authority][publication]") {
  const uint32_t kEnd = uint32_t(GuestRange::kGuestPhysicalSize);
  auto publication = DecideRamWritePublication(
      WriterKind::kTraceRestore, kEnd - 0x100u, 0x400u);
  REQUIRE(publication);
  REQUIRE(publication->range.start() == kEnd - 0x100u);
  REQUIRE(publication->range.length() == 0x100u);
  REQUIRE(publication->action == AuthorityAction::kInvalidateRamWrite);

  RecordingRamWriteTarget target;
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE(target.shared_calls == 1u);
  REQUIRE(target.shared_base == kEnd - 0x100u);
  REQUIRE(target.shared_length == 0x100u);
  REQUIRE(target.shared_exact);
  REQUIRE(target.legacy_calls == 1u);
  REQUIRE(target.legacy_base == kEnd - 0x100u);
  REQUIRE(target.legacy_length == 0x100u);
  REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
  REQUIRE(outcome.invalidated_entries == target.legacy_entries);
  REQUIRE(outcome.shared_invalidation_executed);
  REQUIRE(outcome.legacy_invalidation_executed);
}

TEST_CASE("Authority: guest CPU observation never duplicates physical "
          "invalidation",
          "[l0][authority][publication]") {
  auto publication =
      DecideRamWritePublication(WriterKind::kGuestCpu, 0x1234u, 0x80u);
  REQUIRE(publication);
  REQUIRE_FALSE(publication->policy.execute_shared_invalidation);
  REQUIRE_FALSE(publication->policy.execute_legacy_invalidation);

  RecordingRamWriteTarget target;
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE(target.shared_calls == 0u);
  REQUIRE(target.legacy_calls == 0u);
  REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
}

TEST_CASE("Authority: missing publication operation cannot report agreement",
          "[l0][authority][publication]") {
  auto publication = DecideRamWritePublication(
      WriterKind::kTraceRestore, 0x4000u, 0x1000u);
  REQUIRE(publication);
  RecordingRamWriteTarget target;
  target.shared_result = false;
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE_FALSE(outcome.shared_invalidation_executed);
  REQUIRE(outcome.legacy_invalidation_executed);
  REQUIRE(outcome.action == AuthorityAction::kNoDependency);
}

TEST_CASE("Authority: empty and out-of-domain RAM publications are rejected",
          "[l0][authority][publication]") {
  const uint32_t kEnd = uint32_t(GuestRange::kGuestPhysicalSize);
  REQUIRE_FALSE(
      DecideRamWritePublication(WriterKind::kTraceRestore, 0x1000u, 0u));
  REQUIRE_FALSE(
      DecideRamWritePublication(WriterKind::kGuestCpu, kEnd, 0x1000u));
}

// Task 3.2: completed resolve publication routes through the same typed seam
// used by trace restore (task 3.1), using the checked destination extent
// (metal_command_processor.mm's wb.guest_addr / wb.dest_extent_bytes) rather
// than RangeWrittenByGpu. Resolve's policy executes shared-page invalidation
// only -- legacy texture-cache fan-out for resolve remains the global watch,
// unlike trace restore's independently-required direct call (audit blocker 3
// does not apply to resolve).
TEST_CASE("Authority: resolve executes normalized RAM publication",
          "[l0][authority][publication]") {
  auto publication =
      DecideRamWritePublication(WriterKind::kResolve, 0x20000u, 0x800u);
  REQUIRE(publication);
  REQUIRE(publication->range.start() == 0x20000u);
  REQUIRE(publication->range.length() == 0x800u);
  REQUIRE(publication->action == AuthorityAction::kInvalidateRamWrite);
  REQUIRE(publication->policy.execute_shared_invalidation);
  REQUIRE(publication->policy.exact_range);
  REQUIRE_FALSE(publication->policy.execute_legacy_invalidation);

  RecordingRamWriteTarget target;
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE(target.shared_calls == 1u);
  REQUIRE(target.shared_base == 0x20000u);
  REQUIRE(target.shared_length == 0x800u);
  REQUIRE(target.shared_exact);
  // Resolve's policy never calls the legacy-texture port; cache invalidation
  // for resolve stays on the existing STAGE-3 direct call at the drain site,
  // not the seam's independently-required trace-restore path.
  REQUIRE(target.legacy_calls == 0u);
  REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
  REQUIRE(outcome.shared_invalidation_executed);
  REQUIRE_FALSE(outcome.legacy_invalidation_executed);
  REQUIRE(outcome.invalidated_entries == 0u);
}

TEST_CASE(
    "Authority: resolve publication uses the checked/clamped destination "
    "extent",
    "[l0][authority][publication]") {
  const uint32_t kEnd = uint32_t(GuestRange::kGuestPhysicalSize);
  // A checked scatter destination extent that runs past the 512 MiB guest
  // physical boundary (e.g. a corrupted/edge dest_extent_bytes) must clamp
  // rather than publish an out-of-bounds or wrapped range.
  auto publication = DecideRamWritePublication(WriterKind::kResolve,
                                               kEnd - 0x40u, 0x1000u);
  REQUIRE(publication);
  REQUIRE(publication->range.start() == kEnd - 0x40u);
  REQUIRE(publication->range.length() == 0x40u);

  RecordingRamWriteTarget target;
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE(target.shared_base == kEnd - 0x40u);
  REQUIRE(target.shared_length == 0x40u);
  REQUIRE(outcome.action == AuthorityAction::kInvalidateRamWrite);
}

TEST_CASE(
    "Authority: resolve publication with zero checked extent is rejected",
    "[l0][authority][publication]") {
  // wb.dest_extent_bytes == 0 must never reach InvalidateSharedPages as a
  // publication -- the enqueue side guarantees nonzero, but the seam itself
  // must not fabricate a publication for an empty range.
  REQUIRE_FALSE(
      DecideRamWritePublication(WriterKind::kResolve, 0x9000u, 0u));
}

TEST_CASE(
    "Authority: resolve publication reports no dependency when the target "
    "operation fails",
    "[l0][authority][publication]") {
  // A null/missing RAM mapping must not be falsely published as an executed
  // invalidate-RAM action (metal_command_processor.mm's MetalRamWritePublica-
  // tionTarget returns false when shared_memory_ is null).
  auto publication =
      DecideRamWritePublication(WriterKind::kResolve, 0x30000u, 0x200u);
  REQUIRE(publication);
  RecordingRamWriteTarget target;
  target.shared_result = false;
  LegacyWriterOutcome outcome = PublishRamWrite(*publication, target);
  REQUIRE(target.shared_calls == 1u);
  REQUIRE_FALSE(outcome.shared_invalidation_executed);
  REQUIRE(outcome.action == AuthorityAction::kNoDependency);
}

TEST_CASE("Authority: full consumer x producer action matrix", "[l0][authority]") {
  const ConsumerKind consumers[] = {ConsumerKind::kTexture, ConsumerKind::kVertex,
                                    ConsumerKind::kWait,
                                    ConsumerKind::kHostCpuRead};
  for (ConsumerKind consumer : consumers) {
    // Disjoint -> no dependency.
    {
      std::vector<PendingProducer> p = {
          MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 1)};
      auto d = DecideConsumer(consumer, SingleRange(0x9000u, 0x1000u), p, 1);
      REQUIRE(d.action == AuthorityAction::kNoDependency);
      REQUIRE(d.revalidate_gpu_bytes == ConsumerReadsSharedBuffer(consumer));
    }
    // Overlap + complete -> drain.
    {
      std::vector<PendingProducer> p = {
          MakeProducer(0x1000u, 0x1000u, TransitState::kPendingMemexport, 1)};
      auto d = DecideConsumer(consumer, SingleRange(0x1000u, 0x1000u), p, 1);
      REQUIRE(d.action == AuthorityAction::kDrainMatchingRecords);
    }
    // Overlap + incomplete -> await.
    {
      std::vector<PendingProducer> p = {
          MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 5)};
      auto d = DecideConsumer(consumer, SingleRange(0x1000u, 0x1000u), p, 4);
      REQUIRE(d.action == AuthorityAction::kAwaitProducer);
    }
  }
}

TEST_CASE("Authority observer: agreement when legacy matches prediction",
          "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  // Prediction: overlap + incomplete -> await. The factual vertex path commits,
  // waits, then performs its range-scoped drain, so the actions agree.
  std::vector<PendingProducer> p = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 2)};
  auto decision = DecideConsumer(ConsumerKind::kVertex,
                                 SingleRange(0x1000u, 0x1000u), p, 1);
  LegacyConsumerOutcome legacy;
  legacy.overlap = true;
  legacy.committed = true;
  legacy.waited = true;
  legacy.drained = true;
  legacy.drain_range_scoped = true;
  legacy.drained_records = 1;
  obs.RecordConsumer(decision, legacy);
  REQUIRE(obs.consumer_records(ConsumerKind::kVertex) == 1u);
  REQUIRE(obs.consumer_agree(ConsumerKind::kVertex) == 1u);
  REQUIRE(obs.consumer_disagree(ConsumerKind::kVertex) == 0u);
  REQUIRE(obs.predicted_drained_records() == 1u);
  REQUIRE(obs.legacy_drained_records() == 1u);
}

TEST_CASE("Authority observer: scope divergence recorded for unfiltered drain",
          "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  // The texture and WAIT paths drain ALL records on any overlap (blocker 6).
  // The seam predicts a selective drain of only the intersecting record, so the
  // top-level action agrees but drain breadth diverges -- record it.
  std::vector<PendingProducer> p = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 2),
      MakeProducer(0x9000u, 0x1000u, TransitState::kPendingResolve, 2),
  };
  auto decision = DecideConsumer(ConsumerKind::kWait,
                                 SingleRange(0x1000u, 0x1000u), p, 1);
  REQUIRE(decision.selected.size() == 1u);  // only the intersecting record.
  LegacyConsumerOutcome legacy;
  legacy.overlap = true;
  legacy.committed = true;
  legacy.waited = true;
  legacy.drained = true;
  legacy.drain_range_scoped = false;  // legacy drained ALL records.
  legacy.drained_records = 2;
  obs.RecordConsumer(decision, legacy);
  REQUIRE(obs.consumer_disagree(ConsumerKind::kWait) == 1u);
  REQUIRE(obs.consumer_agree(ConsumerKind::kWait) == 0u);
  // The seam would drain one record; legacy drained two.
  REQUIRE(obs.predicted_drained_records() == 1u);
  REQUIRE(obs.legacy_drained_records() == 2u);
}

TEST_CASE("Authority observer: unfiltered identical breadth is agreement",
          "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  std::vector<PendingProducer> p = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 2),
  };
  auto decision = DecideConsumer(ConsumerKind::kWait,
                                 SingleRange(0x1000u, 4u), p, 1);
  LegacyConsumerOutcome legacy;
  legacy.overlap = true;
  legacy.committed = true;
  legacy.waited = true;
  legacy.drained = true;
  legacy.drain_range_scoped = false;
  legacy.drained_records = 1;
  obs.RecordConsumer(decision, legacy);
  REQUIRE(obs.consumer_agree(ConsumerKind::kWait) == 1u);
  REQUIRE(obs.consumer_disagree(ConsumerKind::kWait) == 0u);
}

TEST_CASE("Authority observer: range-scoped under-retirement diverges",
          "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  std::vector<PendingProducer> p = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 2),
      MakeProducer(0x1800u, 0x1000u, TransitState::kPendingMemexport, 2),
  };
  auto decision = DecideConsumer(ConsumerKind::kVertex,
                                 SingleRange(0x1800u, 0x100u), p, 1);
  REQUIRE(decision.selected.size() == 2u);
  LegacyConsumerOutcome legacy;
  legacy.overlap = true;
  legacy.committed = true;
  legacy.waited = true;
  legacy.drained = true;
  legacy.drain_range_scoped = true;
  legacy.drained_records = 1;
  obs.RecordConsumer(decision, legacy);
  REQUIRE(obs.consumer_disagree(ConsumerKind::kVertex) == 1u);
}

TEST_CASE("Authority observer: routed host read reports selective convergence",
          "[l0][authority][cpu-read][telemetry]") {
  AuthorityObserver obs;
  std::vector<PendingProducer> producers = {
      MakeProducer(0x1000u, 0x100u, TransitState::kPendingResolve, 4, 1),
      MakeProducer(0x9000u, 0x100u, TransitState::kPendingMemexport, 4, 2),
  };
  auto decision = DecideConsumer(ConsumerKind::kHostCpuRead,
                                 SingleRange(0x1000u, 4u), producers, 4);
  REQUIRE(decision.selected.size() == 1u);
  LegacyConsumerOutcome actual;
  actual.overlap = true;
  actual.drained = true;
  actual.drain_range_scoped = true;
  actual.drained_records = 1;
  obs.RecordConsumer(decision, actual);
  REQUIRE(obs.consumer_agree(ConsumerKind::kHostCpuRead) == 1u);
  REQUIRE(obs.consumer_disagree(ConsumerKind::kHostCpuRead) == 0u);
  REQUIRE(obs.predicted_drained_records() == 1u);
  REQUIRE(obs.legacy_drained_records() == 1u);
}

TEST_CASE("Authority observer: no-dependency agreement adds no drain",
          "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  std::vector<PendingProducer> p = {
      MakeProducer(0x1000u, 0x1000u, TransitState::kPendingResolve, 1)};
  auto decision = DecideConsumer(ConsumerKind::kTexture,
                                 SingleRange(0x9000u, 0x1000u), p, 1);
  LegacyConsumerOutcome legacy;  // legacy also found no overlap: all false.
  obs.RecordConsumer(decision, legacy);
  REQUIRE(obs.consumer_agree(ConsumerKind::kTexture) == 1u);
  REQUIRE(obs.action_count(AuthorityAction::kNoDependency) == 1u);
  REQUIRE(obs.predicted_drained_records() == 0u);
}

TEST_CASE("Authority observer: writer publication agreement", "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  GuestRangeSet memexport_ranges = SingleRange(0x4000u, 0x1000u);
  GuestRangeSet resolve_ranges = SingleRange(0x8000u, 0x2000u);
  obs.RecordWriter(WriterKind::kMemexport,
                   memexport_ranges,
                   DecideWriterPublication(WriterKind::kMemexport),
                   {/*action=*/AuthorityAction::kPublishGpuWrite,
                    /*invalidated_entries=*/3, /*committed=*/false,
                    /*waited=*/false});
  obs.RecordWriter(WriterKind::kResolve,
                   resolve_ranges,
                   DecideWriterPublication(WriterKind::kResolve),
                   {/*action=*/AuthorityAction::kInvalidateRamWrite,
                    /*invalidated_entries=*/1, /*committed=*/true,
                    /*waited=*/true});
  REQUIRE(obs.writer_records(WriterKind::kMemexport) == 1u);
  REQUIRE(obs.writer_agree(WriterKind::kMemexport) == 1u);
  REQUIRE(obs.writer_agree(WriterKind::kResolve) == 1u);
  REQUIRE(obs.writer_range_count(WriterKind::kMemexport) == 1u);
  REQUIRE(obs.writer_range_bytes(WriterKind::kMemexport) == 0x1000u);
  REQUIRE(obs.writer_range_bytes(WriterKind::kResolve) == 0x2000u);
  REQUIRE(obs.action_count(AuthorityAction::kPublishGpuWrite) == 1u);
  REQUIRE(obs.action_count(AuthorityAction::kInvalidateRamWrite) == 1u);
}

TEST_CASE("Authority observer: sampling throttle bounds recorded events",
          "[l0][authority][telemetry]") {
  AuthorityObserver obs;
  uint32_t sampled = 0;
  for (uint32_t i = 0; i < AuthorityObserver::kInitialSamples * 4; ++i) {
    if (obs.ShouldSampleConsumer(ConsumerKind::kVertex)) {
      ++sampled;
    }
  }
  // The first kInitialSamples always sample; after that only 1/kSampleStride.
  REQUIRE(sampled >= AuthorityObserver::kInitialSamples);
  REQUIRE(sampled < AuthorityObserver::kInitialSamples * 4);
}

// --- Task 3.3: memexport destination preparation / publication / RAM
// convergence ------------------------------------------------------------

namespace {

class RecordingMemexportPreparationTarget
    : public MemexportPreparationTarget {
 public:
  bool RequestDestinationRange(uint32_t base, uint32_t length) override {
    ++calls;
    requested_base = base;
    requested_length = length;
    return result;
  }

  uint32_t calls = 0;
  uint32_t requested_base = 0;
  uint32_t requested_length = 0;
  bool result = true;
};

class RecordingMemexportPublicationTarget
    : public MemexportPublicationTarget {
 public:
  bool CanMarkRangeWrittenByGpu() const override { return can_mark; }

  void MarkRangeWrittenByGpu(uint32_t base, uint32_t length) override {
    ++calls;
    marked_base = base;
    marked_length = length;
    event_order.push_back(2u);
  }

  uint32_t calls = 0;
  uint32_t marked_base = 0;
  uint32_t marked_length = 0;
  bool can_mark = true;
  std::vector<uint32_t> event_order;
};

class RecordingMemexportConvergenceTarget
    : public MemexportConvergenceTarget {
 public:
  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries) override {
    ++calls;
    copied_base = base;
    copied_length = length;
    *invalidated_entries = invalidated_entries_to_report;
    return result;
  }

  uint32_t calls = 0;
  uint32_t copied_base = 0;
  uint32_t copied_length = 0;
  uint32_t invalidated_entries_to_report = 0;
  bool result = true;
};

}  // namespace

TEST_CASE("Authority: memexport destination preparation executes RequestRange"
          " for the normalized exact range",
          "[l0][authority][memexport]") {
  auto preparation = DecideMemexportDestinationPreparation(0x5000u, 0x800u);
  REQUIRE(preparation.has_value());
  REQUIRE(preparation->range.start() == 0x5000u);
  REQUIRE(preparation->range.length() == 0x800u);

  RecordingMemexportPreparationTarget target;
  bool requested = PrepareMemexportDestination(*preparation, target);
  REQUIRE(requested);
  REQUIRE(target.calls == 1u);
  REQUIRE(target.requested_base == 0x5000u);
  REQUIRE(target.requested_length == 0x800u);
}

TEST_CASE("Authority: memexport destination preparation clamps to the guest "
          "physical boundary",
          "[l0][authority][memexport]") {
  uint32_t kEnd = static_cast<uint32_t>(GuestRange::kGuestPhysicalSize);
  auto preparation =
      DecideMemexportDestinationPreparation(kEnd - 0x100u, 0x1000u);
  REQUIRE(preparation.has_value());
  REQUIRE(preparation->range.start() == kEnd - 0x100u);
  REQUIRE(preparation->range.end() == kEnd);
  REQUIRE(preparation->range.length() == 0x100u);  // clamped, no wraparound.
}

TEST_CASE("Authority: memexport destination preparation rejects zero-length "
          "and out-of-domain input",
          "[l0][authority][memexport]") {
  REQUIRE_FALSE(DecideMemexportDestinationPreparation(0x1000u, 0u));
  uint32_t kEnd = static_cast<uint32_t>(GuestRange::kGuestPhysicalSize);
  REQUIRE_FALSE(DecideMemexportDestinationPreparation(kEnd, 0x100u));
}

TEST_CASE("Authority: memexport destination preparation reports a factual "
          "failure when RequestRange fails",
          "[l0][authority][memexport]") {
  auto preparation = DecideMemexportDestinationPreparation(0x9000u, 0x400u);
  REQUIRE(preparation.has_value());
  RecordingMemexportPreparationTarget target;
  target.result = false;
  REQUIRE_FALSE(PrepareMemexportDestination(*preparation, target));
  REQUIRE(target.calls == 1u);
}

TEST_CASE("Authority: memexport publication marks GPU-written for the exact "
          "submission-owned range",
          "[l0][authority][memexport]") {
  auto publication = DecideMemexportPublication(0x6000u, 0x2000u, 42u);
  REQUIRE(publication.has_value());
  REQUIRE(publication->range.start() == 0x6000u);
  REQUIRE(publication->range.length() == 0x2000u);
  REQUIRE(publication->submission == 42u);

  RecordingMemexportPublicationTarget target;
  uint32_t registered_base = 0;
  uint32_t registered_length = 0;
  uint64_t registered_submission = 0;
  MemexportPublicationResult result = PublishMemexportWrite(
      *publication, target,
      [&](const MemexportPublication& owned) {
        target.event_order.push_back(1u);
        registered_base = owned.range.start();
        registered_length = owned.range.length();
        registered_submission = owned.submission;
        return true;
      });
  REQUIRE(result.registered);
  REQUIRE(result.marked);
  REQUIRE(result.action == AuthorityAction::kPublishGpuWrite);
  REQUIRE(target.calls == 1u);
  REQUIRE(target.marked_base == 0x6000u);
  REQUIRE(target.marked_length == 0x2000u);
  REQUIRE(registered_base == 0x6000u);
  REQUIRE(registered_length == 0x2000u);
  REQUIRE(registered_submission == 42u);
  const std::vector<uint32_t> expected_order = {1u, 2u};
  REQUIRE(target.event_order == expected_order);
}

TEST_CASE("Authority: memexport publication forms one ordered transition -- "
          "failed preflight cannot register or mark",
          "[l0][authority][memexport]") {
  auto publication = DecideMemexportPublication(0x7000u, 0x400u, 7u);
  REQUIRE(publication.has_value());
  RecordingMemexportPublicationTarget target;
  target.can_mark = false;
  uint32_t registrations = 0;
  MemexportPublicationResult result = PublishMemexportWrite(
      *publication, target,
      [&](const MemexportPublication&) {
        ++registrations;
        return true;
      });
  REQUIRE_FALSE(result.registered);
  REQUIRE_FALSE(result.marked);
  REQUIRE(result.action == AuthorityAction::kNoDependency);
  REQUIRE(registrations == 0u);
  REQUIRE(target.calls == 0u);
  REQUIRE(target.event_order.empty());
}

TEST_CASE("Authority: failed memexport registration never marks GPU-written",
          "[l0][authority][memexport]") {
  auto publication = DecideMemexportPublication(0x7400u, 0x400u, 8u);
  REQUIRE(publication.has_value());
  RecordingMemexportPublicationTarget target;
  MemexportPublicationResult result = PublishMemexportWrite(
      *publication, target,
      [&](const MemexportPublication&) {
        target.event_order.push_back(1u);
        return false;
      });
  REQUIRE_FALSE(result.registered);
  REQUIRE_FALSE(result.marked);
  REQUIRE(result.action == AuthorityAction::kNoDependency);
  REQUIRE(target.calls == 0u);
  const std::vector<uint32_t> expected_order = {1u};
  REQUIRE(target.event_order == expected_order);
}

TEST_CASE("Authority: memexport RAM convergence copies exact bytes and "
          "retires on success without touching the shared buffer",
          "[l0][authority][memexport]") {
  PendingProducer producer =
      MakeProducer(0x10000u, 0x1000u, TransitState::kPendingMemexport, 5u, 1u);
  RecordingMemexportConvergenceTarget target;
  target.invalidated_entries_to_report = 2u;
  MemexportConvergenceResult result = ConvergeMemexportRam(producer, target);
  REQUIRE(result.converged);
  REQUIRE(result.action == AuthorityAction::kConvergeRamFromBuffer);
  REQUIRE(result.invalidated_entries == 2u);
  REQUIRE(target.calls == 1u);
  REQUIRE(target.copied_base == 0x10000u);
  REQUIRE(target.copied_length == 0x1000u);
  // The port contract exposes only a RAM-directed copy -- no shared-buffer
  // invalidation primitive is reachable from this seam at all, so a
  // convergence call structurally cannot clobber the buffer it read from.
}

TEST_CASE("Authority: memexport RAM convergence failure reports a factual "
          "disagreement so the caller retains the pending record",
          "[l0][authority][memexport]") {
  PendingProducer producer =
      MakeProducer(0x20000u, 0x800u, TransitState::kPendingMemexport, 9u, 2u);
  RecordingMemexportConvergenceTarget target;
  target.result = false;  // missing RAM mapping or shared-buffer contents.
  MemexportConvergenceResult result = ConvergeMemexportRam(producer, target);
  REQUIRE_FALSE(result.converged);
  REQUIRE(result.action == AuthorityAction::kNoDependency);
  REQUIRE(result.invalidated_entries == 0u);

  // The caller's retention policy: a failed convergence must retain the
  // record for retry rather than drop it or retire it.
  std::vector<PendingProducer> still_pending;
  uint32_t drained_records = 0;
  if (!result.converged) {
    still_pending.push_back(producer);
  } else {
    ++drained_records;
  }
  REQUIRE(still_pending.size() == 1u);
  REQUIRE(drained_records == 0u);
}

TEST_CASE("Authority: memexport convergence drain counters increment only "
          "for successful retirement",
          "[l0][authority][memexport]") {
  std::vector<PendingProducer> producers = {
      MakeProducer(0x30000u, 0x100u, TransitState::kPendingMemexport, 1u, 1u),
      MakeProducer(0x31000u, 0x100u, TransitState::kPendingMemexport, 1u, 2u),
  };
  RecordingMemexportConvergenceTarget ok_target;
  RecordingMemexportConvergenceTarget failing_target;
  failing_target.result = false;

  uint32_t drained = 0;
  std::vector<PendingProducer> retained;
  MemexportConvergenceResult first =
      ConvergeMemexportRam(producers[0], ok_target);
  if (first.converged) {
    ++drained;
  } else {
    retained.push_back(producers[0]);
  }
  MemexportConvergenceResult second =
      ConvergeMemexportRam(producers[1], failing_target);
  if (second.converged) {
    ++drained;
  } else {
    retained.push_back(producers[1]);
  }
  REQUIRE(drained == 1u);
  REQUIRE(retained.size() == 1u);
  REQUIRE(retained.front().range == producers[1].range);
}

// --- Task 4.5: same-frame byte-level consumption matrix -------------------
//
// The tests above assert the seam's ACTION and the SET of selected producers.
// Task 4.5 asks the stronger question: after the seam's selected producers
// converge, does the consumer read the exact newest bytes? To answer factually
// this section adds a minimal device-free byte store. The store owns NO policy:
// WHICH producers converge and in WHAT order comes entirely from the real
// DecideConsumer output (its sequence-sorted `selected`); memexport producers
// converge through the real ConvergeMemexportRam seam. The store only moves the
// bytes those decisions name, then the test asserts the resulting RAM image.
// Newest-writer-wins therefore falls out of the seam's ordering, not out of any
// logic reimplemented here.

namespace {

// A single guest-RAM window shared by every task 4.5 case. All producer and
// consumer ranges below live inside [kFrameRamBase, kFrameRamBase + kFrameSize).
constexpr uint32_t kFrameRamBase = 0x1000u;
constexpr uint32_t kFrameSize = 0x10000u;
constexpr uint8_t kStaleRam = 0x11u;  // baseline "RAM has the old bytes" fill.

// A pending producer paired with the exact bytes it owes RAM over its range.
// For a resolve these are the scattered destination bytes; for a memexport they
// are the shared-buffer contents the convergence copy reads.
struct ProducerBytes {
  PendingProducer producer;
  std::vector<uint8_t> payload;  // payload[i] belongs to range.start() + i.
  // When false the convergence copy reports failure without moving bytes,
  // modelling a missing RAM mapping / shared-buffer pointer so the shared
  // drain executor's stop-on-failure policy can be exercised at the byte level.
  bool converges = true;
};

ProducerBytes MakeProducerBytes(uint32_t base, uint32_t size,
                                TransitState state, uint64_t submission,
                                uint64_t sequence, uint8_t fill) {
  return ProducerBytes{MakeProducer(base, size, state, submission, sequence),
                       std::vector<uint8_t>(size, fill)};
}

const ProducerBytes* FindBySequence(const std::vector<ProducerBytes>& world,
                                    uint64_t sequence) {
  for (const ProducerBytes& pb : world) {
    if (pb.producer.sequence == sequence) {
      return &pb;
    }
  }
  return nullptr;
}

// The byte-moving body of the real MemexportConvergenceTarget port: copies a
// producer's staged shared-buffer bytes into the RAM image. The seam still
// decides the base/length; this only performs the copy it is handed and never
// touches the source buffer, honoring the convergence contract.
class ByteStoreConvergenceTarget : public MemexportConvergenceTarget {
 public:
  ByteStoreConvergenceTarget(std::vector<uint8_t>& ram, uint32_t ram_base,
                             const ProducerBytes& producer)
      : ram_(ram), ram_base_(ram_base), producer_(producer) {}

  bool CopyBufferToRam(uint32_t base, uint32_t length,
                       uint32_t* invalidated_entries) override {
    *invalidated_entries = 0;
    // Model a missing RAM mapping / shared-buffer contents pointer: report the
    // failure the convergence contract expects instead of moving bytes, so the
    // caller retains the pending record.
    if (!producer_.converges) {
      return false;
    }
    // Explicit bounds validation: a malformed future range must fail
    // diagnostically rather than index the RAM image or payload out of bounds
    // with unchecked vector arithmetic. All widths are promoted to 64-bit so the
    // end computation itself cannot wrap.
    uint32_t payload_base = producer_.producer.range.start();
    REQUIRE(base >= ram_base_);
    REQUIRE(base >= payload_base);
    REQUIRE(uint64_t(base - ram_base_) + length <= ram_.size());
    REQUIRE(uint64_t(base - payload_base) + length <= producer_.payload.size());
    for (uint32_t i = 0; i < length; ++i) {
      ram_[base - ram_base_ + i] = producer_.payload[base - payload_base + i];
    }
    return true;
  }

 private:
  std::vector<uint8_t>& ram_;
  uint32_t ram_base_;
  const ProducerBytes& producer_;
};

// Drain one selected producer's owed bytes into the RAM image and report the
// factual LegacyDrainOutcome the shared executor merges. Memexport routes
// through the real convergence seam; resolve deposits its checked scatter extent
// the way the writeback drain copies a completed resolve (no device-free resolve
// byte seam exists, so the store performs the copy the seam's selection
// ordered). A non-converging producer returns a zero outcome so the shared
// stop-on-failure policy retains it and the untouched suffix.
LegacyDrainOutcome ApplyProducerBytes(const ProducerBytes& pb,
                                      std::vector<uint8_t>& ram) {
  LegacyDrainOutcome outcome;
  if (pb.producer.state == TransitState::kPendingMemexport) {
    ByteStoreConvergenceTarget target(ram, kFrameRamBase, pb);
    MemexportConvergenceResult result = ConvergeMemexportRam(pb.producer, target);
    if (!result.converged) {
      return outcome;  // drained_records stays 0 -> executor stops.
    }
    REQUIRE(result.action == AuthorityAction::kConvergeRamFromBuffer);
    outcome.memexport_records = 1;
    outcome.invalidated_entries = result.invalidated_entries;
  } else {
    // Explicit bounds validation on the resolve copy: reject a malformed future
    // range diagnostically instead of using unchecked vector arithmetic.
    uint32_t base = pb.producer.range.start();
    uint32_t length = pb.producer.range.length();
    REQUIRE(base >= kFrameRamBase);
    REQUIRE(uint64_t(base - kFrameRamBase) + length <= ram.size());
    REQUIRE(length <= pb.payload.size());
    for (uint32_t i = 0; i < length; ++i) {
      ram[base - kFrameRamBase + i] = pb.payload[i];
    }
    outcome.resolve_records = 1;
  }
  outcome.drained_records = 1;
  return outcome;
}

// How a modeled await resolves. Production (ResolvePendingProducerDependencies)
// waits for the required submission and only then drains; a failed wait returns
// incomplete and fails the draw closed. There is no in-frame state in which an
// awaiting consumer reads stale bytes -- it either waits+drains, or observes
// nothing at all.
enum class WaitModel {
  kSucceeds,  // the awaited submissions complete; the consumer drains.
  kFails,     // the wait fails; the consumer produces no byte observation.
};

// The factual result of one consumer operation. `available` is the byte-level
// analogue of ResolveRamConsumerDependencies returning true: the consumer got a
// converged byte image only when the (possibly awaited) drain retired every
// selected record. When false, `ram` must not be treated as an observation.
struct ConsumerObservation {
  ConsumerDecision decision;
  LegacyDrainOutcome drain;
  bool available = false;
  std::vector<uint8_t> ram;
};

// Materialize the guest-RAM bytes a consumer observes: run the real
// DecideConsumer over the world's pending records, then converge the selected
// producers through the SAME shared production executor (DrainSelectedProducers)
// used by MetalCommandProcessor. Await semantics mirror production exactly:
//   * no await -> drain immediately;
//   * await + successful wait -> the awaited submissions complete and the SAME
//     consumer operation drains and returns the newest bytes;
//   * await + failed wait -> no bytes are observed and the consumer is
//     unavailable (available == false), never a stale byte image.
// `available` additionally requires the executor to retire every selected
// record, so a mid-drain stop-on-failure also yields an unavailable consumer
// with the untouched suffix left stale.
ConsumerObservation ObserveConsumerBytes(
    ConsumerKind consumer, const ConsumerRequest& request,
    const std::vector<ProducerBytes>& world, uint64_t completed_submission,
    WaitModel wait_model = WaitModel::kSucceeds) {
  std::vector<PendingProducer> snapshot;
  snapshot.reserve(world.size());
  for (const ProducerBytes& pb : world) {
    snapshot.push_back(pb.producer);
  }
  ConsumerObservation observation;
  observation.decision =
      DecideConsumer(consumer, request, snapshot, completed_submission);
  std::vector<uint8_t> ram(kFrameSize, kStaleRam);

  if (observation.decision.await && wait_model == WaitModel::kFails) {
    // Failed wait: production returns incomplete and fails the draw closed. No
    // convergence runs and no bytes are observed.
    observation.available = false;
    return observation;
  }

  // Either no await, or a successful wait completed the awaited submissions.
  // Drive producer application through the shared executor and check the actual
  // merged drain outcome rather than replaying the selection order by hand.
  observation.drain = DrainSelectedProducers(
      observation.decision, [&](const PendingProducer& selected) {
        const ProducerBytes* pb = FindBySequence(world, selected.sequence);
        REQUIRE(pb != nullptr);
        return ApplyProducerBytes(*pb, ram);
      });
  observation.available =
      observation.drain.drained_records == observation.decision.selected.size();
  observation.ram = std::move(ram);
  return observation;
}

ConsumerRequest UniformRequest(uint32_t base, uint32_t size) {
  return ConsumerRequest::Uniform(SingleRange(base, size));
}

// Assert every byte of [base, base+size) equals `value`. Bounds are validated
// (64-bit end, no wrap) so a malformed future range fails diagnostically instead
// of iterating past the RAM image with unchecked vector arithmetic.
void RequireRangeBytes(const std::vector<uint8_t>& ram, uint32_t base,
                       uint32_t size, uint8_t value) {
  REQUIRE(base >= kFrameRamBase);
  REQUIRE(uint64_t(base - kFrameRamBase) + size <= ram.size());
  const auto first = ram.begin() + (base - kFrameRamBase);
  REQUIRE(std::all_of(first, first + size,
                      [value](uint8_t byte) { return byte == value; }));
}

}  // namespace

TEST_CASE("Authority bytes: same-frame resolve is seen whole by a vertex fetch",
          "[l0][authority][same-frame][bytes]") {
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x2000u, 0x1000u, TransitState::kPendingResolve,
                        /*submission=*/5, /*sequence=*/1, /*fill=*/0xA0u),
  };
  auto obs = ObserveConsumerBytes(ConsumerKind::kVertex,
                                  UniformRequest(0x2000u, 0x1000u), world,
                                  /*completed=*/5);
  REQUIRE(obs.available);
  REQUIRE(obs.decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(obs.decision.revalidate_gpu_bytes);  // vertex reads the shared buffer.
  REQUIRE(obs.decision.selected_bytes() == 0x1000u);
  // The consumer observes the resolved bytes, not stale RAM.
  RequireRangeBytes(obs.ram, 0x2000u, 0x1000u, 0xA0u);
  RequireRangeBytes(obs.ram, 0x1000u, 0x1000u, kStaleRam);  // outside: untouched.
}

TEST_CASE(
    "Authority bytes: same-frame memexport is seen whole by a texture fetch",
    "[l0][authority][same-frame][bytes]") {
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x4000u, 0x1000u, TransitState::kPendingMemexport,
                        /*submission=*/3, /*sequence=*/1, /*fill=*/0xB0u),
  };
  auto obs = ObserveConsumerBytes(ConsumerKind::kTexture,
                                  UniformRequest(0x4000u, 0x1000u), world,
                                  /*completed=*/3);
  REQUIRE(obs.available);
  REQUIRE(obs.decision.action == AuthorityAction::kDrainMatchingRecords);
  // Bytes arrive through the real ConvergeMemexportRam seam (buffer -> RAM).
  RequireRangeBytes(obs.ram, 0x4000u, 0x1000u, 0xB0u);
}

TEST_CASE(
    "Authority bytes: every consumer kind converges identical same-frame bytes",
    "[l0][authority][same-frame][bytes]") {
  // A resolve and a memexport in the same frame, with a stale gap between them.
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x2000u, 0x1000u, TransitState::kPendingResolve,
                        /*submission=*/4, /*sequence=*/1, /*fill=*/0xC1u),
      MakeProducerBytes(0x5000u, 0x1000u, TransitState::kPendingMemexport,
                        /*submission=*/4, /*sequence=*/2, /*fill=*/0xC2u),
  };
  const ConsumerKind kinds[] = {ConsumerKind::kTexture, ConsumerKind::kVertex,
                                ConsumerKind::kWait, ConsumerKind::kHostCpuRead};
  for (ConsumerKind kind : kinds) {
    auto obs = ObserveConsumerBytes(kind, UniformRequest(0x2000u, 0x4000u),
                                    world, /*completed=*/4);
    REQUIRE(obs.available);
    REQUIRE(obs.decision.action == AuthorityAction::kDrainMatchingRecords);
    REQUIRE(obs.decision.selected.size() == 2u);
    RequireRangeBytes(obs.ram, 0x2000u, 0x1000u, 0xC1u);  // resolve bytes.
    RequireRangeBytes(obs.ram, 0x3000u, 0x2000u, kStaleRam);  // gap: no producer.
    RequireRangeBytes(obs.ram, 0x5000u, 0x1000u, 0xC2u);  // memexport bytes.
  }
}

TEST_CASE(
    "Authority bytes: multiple pending records select intersecting bytes and "
    "leave a disjoint record untouched",
    "[l0][authority][same-frame][bytes]") {
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x2000u, 0x800u, TransitState::kPendingResolve,
                        /*submission=*/2, /*sequence=*/1, /*fill=*/0xD1u),
      MakeProducerBytes(0x2800u, 0x800u, TransitState::kPendingMemexport,
                        /*submission=*/2, /*sequence=*/2, /*fill=*/0xD2u),
      // Disjoint third record: never touched by this consumer.
      MakeProducerBytes(0x8000u, 0x1000u, TransitState::kPendingResolve,
                        /*submission=*/2, /*sequence=*/3, /*fill=*/0xD3u),
  };
  auto obs = ObserveConsumerBytes(ConsumerKind::kWait,
                                  UniformRequest(0x2000u, 0x1000u), world,
                                  /*completed=*/2);
  REQUIRE(obs.available);
  REQUIRE(obs.decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(obs.decision.selected.size() == 2u);  // the two intersecting records.
  RequireRangeBytes(obs.ram, 0x2000u, 0x800u, 0xD1u);
  RequireRangeBytes(obs.ram, 0x2800u, 0x800u, 0xD2u);
  // The disjoint record's RAM is never converged: it still reads stale bytes,
  // proving the selective (not global) drain at the byte level.
  RequireRangeBytes(obs.ram, 0x8000u, 0x1000u, kStaleRam);
}

TEST_CASE(
    "Authority bytes: chronological newest-writer-wins where records overlap",
    "[l0][authority][same-frame][bytes]") {
  // An older resolve and a newer memexport overlap on [0x2800, 0x3000). Applied
  // in the seam's sequence order the memexport lands last and wins that span.
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x2000u, 0x1000u, TransitState::kPendingResolve,
                        /*submission=*/4, /*sequence=*/10, /*fill=*/0xE1u),
      MakeProducerBytes(0x2800u, 0x1000u, TransitState::kPendingMemexport,
                        /*submission=*/4, /*sequence=*/20, /*fill=*/0xE2u),
  };
  auto obs = ObserveConsumerBytes(ConsumerKind::kHostCpuRead,
                                  UniformRequest(0x2000u, 0x1800u), world,
                                  /*completed=*/4);
  REQUIRE(obs.available);
  REQUIRE(obs.decision.selected.size() == 2u);
  REQUIRE(obs.decision.selected[0].sequence == 10u);  // older applied first.
  REQUIRE(obs.decision.selected[1].sequence == 20u);  // newer applied last.
  RequireRangeBytes(obs.ram, 0x2000u, 0x800u, 0xE1u);   // resolve-only head.
  RequireRangeBytes(obs.ram, 0x2800u, 0x1000u, 0xE2u);  // overlap + tail: newest.
}

TEST_CASE(
    "Authority bytes: closure pulls the older resolve so newest bytes still win "
    "when only the memexport tail is requested",
    "[l0][authority][same-frame][bytes]") {
  // Bridge-style request: the texture contributes only its memexport range, but
  // whole-record convergence must retire the older overlapping resolve first or
  // it could later clobber the newer overlap bytes. Assert the byte outcome.
  ConsumerRequest request;
  request.memexport_ranges.Add(GuestRange::FromBaseLength(0x2800u, 0x1000u));
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x2000u, 0x1000u, TransitState::kPendingResolve,
                        /*submission=*/6, /*sequence=*/10, /*fill=*/0xF1u),
      MakeProducerBytes(0x2800u, 0x1000u, TransitState::kPendingMemexport,
                        /*submission=*/6, /*sequence=*/20, /*fill=*/0xF2u),
  };
  auto obs = ObserveConsumerBytes(ConsumerKind::kTexture, request, world,
                                  /*completed=*/6);
  REQUIRE(obs.available);
  REQUIRE(obs.decision.selected.size() == 2u);  // closure pulled the resolve.
  REQUIRE(obs.decision.selected[0].sequence == 10u);
  REQUIRE(obs.decision.selected[1].sequence == 20u);
  RequireRangeBytes(obs.ram, 0x2000u, 0x800u, 0xF1u);   // resolve head survives.
  RequireRangeBytes(obs.ram, 0x2800u, 0x1000u, 0xF2u);  // newest memexport wins.
}

TEST_CASE(
    "Authority bytes: bit-29 aliased WAIT_REG_MEM word converges its whole "
    "record and retains the disjoint record",
    "[l0][authority][same-frame][bytes][alias]") {
  // WAIT_REG_MEM polls exactly one 32-bit word. Its packet address carries the
  // bit-29 (0x2000'0000) physical alias; the routed call site aligns it to 4 and
  // normalizes it before the authority sees it. The request is therefore exactly
  // 4 bytes -- matching production -- yet whole-record convergence still applies
  // the entire selected memexport record, and the disjoint physical record must
  // neither be selected nor converged.
  constexpr uint32_t kAliasedWaitAddress = 0x20001003u;
  uint32_t physical_address =
      xenos::CpuToGpu(kAliasedWaitAddress & ~uint32_t(0x3));
  REQUIRE(physical_address == 0x1000u);

  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x1000u, 0x40u, TransitState::kPendingMemexport,
                        /*submission=*/3, /*sequence=*/1, /*fill=*/0xA1u),
      MakeProducerBytes(0x2000u, 0x40u, TransitState::kPendingResolve,
                        /*submission=*/3, /*sequence=*/2, /*fill=*/0xA2u),
  };
  auto obs = ObserveConsumerBytes(ConsumerKind::kWait,
                                  UniformRequest(physical_address, 4u), world,
                                  /*completed=*/3);
  REQUIRE(obs.available);
  REQUIRE(obs.decision.action == AuthorityAction::kDrainMatchingRecords);
  REQUIRE(obs.decision.selected.size() == 1u);
  REQUIRE(obs.decision.selected[0].sequence == 1u);  // the physical record.
  REQUIRE(obs.decision.selected[0].range.length() == 0x40u);  // whole record.
  REQUIRE(obs.decision.requested.CoveredBytes() == 4u);  // exact WAIT word.
  REQUIRE_FALSE(obs.decision.revalidate_gpu_bytes);  // wait reads guest RAM.
  // The exact 4-byte WAIT word reads the newest physical byte.
  RequireRangeBytes(obs.ram, physical_address, 4u, 0xA1u);
  // Whole selected record converges, not just the 4-byte intersection.
  RequireRangeBytes(obs.ram, 0x1000u, 0x40u, 0xA1u);
  // The disjoint physical record is neither selected nor converged: retained.
  RequireRangeBytes(obs.ram, 0x2000u, 0x40u, kStaleRam);
}

TEST_CASE(
    "Authority bytes: an awaiting consumer waits and drains in one operation, "
    "or observes nothing -- never stale bytes",
    "[l0][authority][same-frame][bytes]") {
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x3000u, 0x1000u, TransitState::kPendingMemexport,
                        /*submission=*/10, /*sequence=*/1, /*fill=*/0xB5u),
  };

  // Submission 10 has not completed (high-water 9), so the consumer must await.
  // A FAILED wait converges nothing and produces no byte observation at all --
  // production fails the draw closed rather than reading stale RAM. There is no
  // in-frame state in which an awaiting consumer observes the old bytes.
  auto failed = ObserveConsumerBytes(ConsumerKind::kTexture,
                                     UniformRequest(0x3000u, 0x1000u), world,
                                     /*completed=*/9, WaitModel::kFails);
  REQUIRE(failed.decision.action == AuthorityAction::kAwaitProducer);
  REQUIRE(failed.decision.max_wait_submission == 10u);
  REQUIRE_FALSE(failed.available);       // no consumer availability.
  REQUIRE(failed.ram.empty());           // no byte observation.
  REQUIRE(failed.drain.drained_records == 0u);

  // A SUCCESSFUL wait completes submission 10 and, in the SAME consumer
  // operation, drains the selected record and returns the newest bytes.
  auto awaited = ObserveConsumerBytes(ConsumerKind::kTexture,
                                      UniformRequest(0x3000u, 0x1000u), world,
                                      /*completed=*/9, WaitModel::kSucceeds);
  REQUIRE(awaited.decision.action == AuthorityAction::kAwaitProducer);
  REQUIRE(awaited.decision.max_wait_submission == 10u);
  REQUIRE(awaited.available);
  REQUIRE(awaited.drain.drained_records == 1u);
  RequireRangeBytes(awaited.ram, 0x3000u, 0x1000u, 0xB5u);

  // The non-awaited path (high-water already 10) drains identically.
  auto ready = ObserveConsumerBytes(ConsumerKind::kTexture,
                                    UniformRequest(0x3000u, 0x1000u), world,
                                    /*completed=*/10);
  REQUIRE(ready.available);
  REQUIRE(ready.decision.action == AuthorityAction::kDrainMatchingRecords);
  RequireRangeBytes(ready.ram, 0x3000u, 0x1000u, 0xB5u);
}

TEST_CASE(
    "Authority bytes: the shared drain executor stops on the first failed "
    "convergence and leaves the selected suffix and disjoint record untouched",
    "[l0][authority][same-frame][bytes]") {
  // Three intersecting records plus a disjoint one. The middle memexport record
  // cannot converge (missing RAM/buffer mapping). The shared executor
  // (DrainSelectedProducers) must apply the first record, stop at the failing
  // second, and never touch the selected suffix or the disjoint record -- so a
  // later retry of an older record cannot invert authority order.
  std::vector<ProducerBytes> world = {
      MakeProducerBytes(0x2000u, 0x800u, TransitState::kPendingResolve,
                        /*submission=*/2, /*sequence=*/1, /*fill=*/0xC1u),
      MakeProducerBytes(0x2800u, 0x800u, TransitState::kPendingMemexport,
                        /*submission=*/2, /*sequence=*/2, /*fill=*/0xC2u),
      MakeProducerBytes(0x3000u, 0x800u, TransitState::kPendingResolve,
                        /*submission=*/2, /*sequence=*/3, /*fill=*/0xC3u),
      // Disjoint record: never selected.
      MakeProducerBytes(0x8000u, 0x800u, TransitState::kPendingResolve,
                        /*submission=*/2, /*sequence=*/4, /*fill=*/0xC4u),
  };
  world[1].converges = false;  // the middle record fails to converge.

  auto obs = ObserveConsumerBytes(ConsumerKind::kWait,
                                  UniformRequest(0x2000u, 0x1800u), world,
                                  /*completed=*/2);
  REQUIRE(obs.decision.selected.size() == 3u);  // the three intersecting.
  // The executor retired exactly one record before stopping.
  REQUIRE(obs.drain.drained_records == 1u);
  REQUIRE(obs.drain.resolve_records == 1u);
  REQUIRE_FALSE(obs.available);  // not every selected record retired.
  // Only the first record's bytes landed.
  RequireRangeBytes(obs.ram, 0x2000u, 0x800u, 0xC1u);
  // The failing record and the untouched selected suffix stay stale.
  RequireRangeBytes(obs.ram, 0x2800u, 0x800u, kStaleRam);
  RequireRangeBytes(obs.ram, 0x3000u, 0x800u, kStaleRam);
  // The disjoint record is never touched.
  RequireRangeBytes(obs.ram, 0x8000u, 0x800u, kStaleRam);
}

}  // namespace test
}  // namespace gpu
}  // namespace metal
}  // namespace xe
