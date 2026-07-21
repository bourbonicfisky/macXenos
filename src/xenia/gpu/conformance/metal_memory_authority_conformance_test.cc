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

TEST_CASE("Authority: texture bridge excludes resolve but not memexport",
          "[l0][authority][bridge]") {
  ConsumerRequest request;
  // A bridge-served texture contributes no resolve-read range, but its exact
  // bytes remain a memexport dependency because memexport is not in the bridge.
  request.memexport_ranges.Add(
      GuestRange::FromBaseLength(0x4000u, 0x1000u));
  std::vector<PendingProducer> producers = {
      MakeProducer(0x4000u, 0x1000u, TransitState::kPendingResolve, 3),
      MakeProducer(0x4000u, 0x1000u, TransitState::kPendingMemexport, 3),
  };
  auto decision = DecideConsumer(ConsumerKind::kTexture, request, producers, 3);
  REQUIRE(decision.selected.size() == 1u);
  REQUIRE(decision.selected[0].state == TransitState::kPendingMemexport);
  REQUIRE(decision.requested.CoveredBytes() == 0x1000u);
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

}  // namespace test
}  // namespace gpu
}  // namespace metal
}  // namespace xe
