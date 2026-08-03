/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// GPU fault EPISODE grouping (Level 0). These rules decide how many times we
// say "the GPU broke", and until now they could only be exercised when a
// stochastic hardware fault happened to occur -- roughly a coin flip per live
// run. Pinning them here makes the accounting verifiable without a fault.

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_fault_episode.h"

namespace xe {
namespace gpu {
namespace test {

using metal::FaultEpisodeSummary;
using metal::FaultEpisodeTracker;

TEST_CASE("a single failure opens one episode rooted at its serial",
          "[gpu-fault][episode]") {
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  CHECK(tracker.NoteMember(100, 0, &closed) == 1);
  CHECK_FALSE(closed.valid);  // Nothing was closed by the first member.
  CHECK((tracker.open_episodes() != 0));
  FaultEpisodeSummary summary = tracker.CloseAny();
  REQUIRE(summary.valid);
  CHECK(summary.episode == 1);
  CHECK(summary.root_serial == 100);
  CHECK(summary.failed_submissions == 1);
  CHECK_FALSE((tracker.open_episodes() != 0));
}

TEST_CASE("collateral in the same epoch joins ONE episode",
          "[gpu-fault][episode]") {
  // The measured VP shape: one root failure, many collateral completions.
  // These must NOT be reported as many independent faults.
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  for (uint64_t serial : {500u, 501u, 502u, 503u, 504u}) {
    CHECK(tracker.NoteMember(serial, 0, &closed) == 1);
    CHECK_FALSE(closed.valid);
  }
  FaultEpisodeSummary summary = tracker.CloseAny();
  REQUIRE(summary.valid);
  CHECK(summary.episode == 1);              // ONE episode...
  CHECK(summary.failed_submissions == 5);   // ...with a blast radius of 5.
  CHECK(summary.root_serial == 500);
  CHECK(tracker.episode_count() == 1);
}

TEST_CASE("out-of-order completions revise the root downward",
          "[gpu-fault][episode]") {
  // Completion callbacks may arrive out of submission order, so the
  // first-REPORTED failure is not necessarily the earliest one. The root must
  // be the lowest serial, whenever it shows up.
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  tracker.NoteMember(900, 0, &closed);   // reported first
  tracker.NoteMember(880, 0, &closed);   // earlier submission, later callback
  tracker.NoteMember(910, 0, &closed);
  tracker.NoteMember(875, 0, &closed);   // earlier still
  FaultEpisodeSummary summary = tracker.CloseAny();
  REQUIRE(summary.valid);
  CHECK(summary.root_serial == 875);
  CHECK(summary.failed_submissions == 4);
  // A later HIGHER serial must never raise the root.
  FaultEpisodeTracker rising;
  rising.NoteMember(10, 0, &closed);
  rising.NoteMember(99, 0, &closed);
  CHECK(rising.CloseAny().root_serial == 10);
}

TEST_CASE("an epoch change starts a new episode and closes the previous one",
          "[gpu-fault][episode]") {
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  tracker.NoteMember(200, 0, &closed);
  tracker.NoteMember(201, 0, &closed);
  CHECK_FALSE(closed.valid);
  // Queue revoked and recreated.
  tracker.BumpEpoch();
  CHECK(tracker.epoch() == 1);
  // The next failure belongs to a NEW episode, and closing the old one is
  // reported through `closed` so the caller can emit its summary.
  CHECK(tracker.NoteMember(300, 1, &closed) == 2);
  // Nothing is force-retired: the old epoch's episode stays open so a LATE
  // callback carrying epoch 0 can still join it.
  CHECK_FALSE(closed.valid);
  CHECK(tracker.open_episodes() == 2);
  FaultEpisodeSummary first = tracker.CloseEpoch(0);
  REQUIRE(first.valid);
  CHECK(first.episode == 1);
  CHECK(first.epoch == 0);
  CHECK(first.root_serial == 200);
  CHECK(first.failed_submissions == 2);
  FaultEpisodeSummary second = tracker.CloseAny();
  REQUIRE(second.valid);
  CHECK(second.episode == 2);
  CHECK(second.epoch == 1);
  CHECK(second.root_serial == 300);
  CHECK(tracker.episode_count() == 2);
}

TEST_CASE("bumping the epoch without a fault opens no episode",
          "[gpu-fault][episode]") {
  // A queue recreate with no failures must not manufacture an episode.
  FaultEpisodeTracker tracker;
  tracker.BumpEpoch();
  tracker.BumpEpoch();
  CHECK_FALSE((tracker.open_episodes() != 0));
  CHECK(tracker.episode_count() == 0);
  CHECK_FALSE(tracker.CloseAny().valid);
}

TEST_CASE("closing twice is harmless and never double-counts",
          "[gpu-fault][episode]") {
  // Close() is called both on queue-recreate and at teardown; the second must
  // not emit a phantom episode.
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  tracker.NoteMember(7, 0, &closed);
  CHECK(tracker.CloseAny().valid);
  CHECK_FALSE(tracker.CloseAny().valid);
  CHECK(tracker.episode_count() == 1);
  // Reopening after a close is a genuinely new episode.
  CHECK(tracker.NoteMember(8, 0, &closed) == 2);
}

TEST_CASE("episode ids are monotonic across many epochs",
          "[gpu-fault][episode]") {
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  uint64_t previous = 0;
  for (uint64_t i = 0; i < 8; ++i) {
    uint64_t id = tracker.NoteMember(1000 + i * 10, i, &closed);
    CHECK(id > previous);
    previous = id;
    tracker.BumpEpoch();
    tracker.CloseEpoch(i);
  }
  CHECK(tracker.episode_count() == 8);
  CHECK(tracker.epoch() == 8);
}

TEST_CASE("a zero serial never becomes the root",
          "[gpu-fault][episode]") {
  // Serial 0 is the "unknown/unowned" sentinel elsewhere in the backend; it
  // must not win the min() and mask the real root.
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary closed;
  tracker.NoteMember(42, 0, &closed);
  tracker.NoteMember(0, 0, &closed);
  FaultEpisodeSummary summary = tracker.CloseAny();
  REQUIRE(summary.valid);
  CHECK(summary.root_serial == 42);
  CHECK(summary.failed_submissions == 2);  // Still counted as a member.
}

TEST_CASE("a LATE callback after queue recreation joins its ORIGINAL episode",
          "[gpu-fault][episode]") {
  // THE REGRESSION THIS SHAPE EXISTS FOR. A submission issued under epoch 0
  // can have its completion callback drained after the queue was revoked and
  // recreated (epoch 1). Keying membership on the DRAIN-TIME epoch would file
  // it under the new episode, splitting one real event in two and inventing a
  // second "fault". Membership keys on the submission's OWN epoch instead.
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary retired;
  CHECK(tracker.NoteMember(500, 0, &retired) == 1);
  tracker.BumpEpoch();  // queue revoked -> recreated
  CHECK(tracker.epoch() == 1);
  // Late callback for a submission issued under the OLD epoch.
  CHECK(tracker.NoteMember(499, 0, &retired) == 1);  // same episode
  CHECK_FALSE(retired.valid);
  // A genuinely new failure under the new epoch is a DIFFERENT episode.
  CHECK(tracker.NoteMember(600, 1, &retired) == 2);
  FaultEpisodeSummary old_episode = tracker.CloseEpoch(0);
  REQUIRE(old_episode.valid);
  CHECK(old_episode.episode == 1);
  CHECK(old_episode.failed_submissions == 2);
  CHECK(old_episode.root_serial == 499);  // late lower serial became the root
  CHECK(tracker.CloseEpoch(1).episode == 2);
}

TEST_CASE("in-flight capacity exhaustion retires loudly, never silently drops",
          "[gpu-fault][episode]") {
  FaultEpisodeTracker tracker;
  FaultEpisodeSummary retired;
  for (uint64_t e = 0; e < FaultEpisodeTracker::kMaxOpenEpisodes; ++e) {
    tracker.NoteMember(100 + e, e, &retired);
    CHECK_FALSE(retired.valid);
  }
  CHECK(tracker.open_episodes() == FaultEpisodeTracker::kMaxOpenEpisodes);
  // One more epoch forces a retirement -- and it must be REPORTED, so
  // attribution is never lost without a record.
  tracker.NoteMember(999, 99, &retired);
  REQUIRE(retired.valid);
  CHECK(retired.episode == 1);       // oldest retired
  CHECK(retired.root_serial == 100);
  CHECK(tracker.open_episodes() == FaultEpisodeTracker::kMaxOpenEpisodes);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
