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

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_memory_authority.h"

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

}  // namespace test
}  // namespace gpu
}  // namespace metal
}  // namespace xe
