/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the deferred cache-clear admission decision.
//
// The states that matter are the ones a live command processor reaches only by
// accident of timing -- a ClearCaches() call-in landing mid-encoder, landing
// with submissions in flight, or landing with a resolve writeback still queued
// for guest RAM -- and each of them, decided wrongly, frees memory the GPU is
// reading or discards guest-visible data that never arrived. Enumerating them
// as values is exact; reproducing them live is not.
//
// MetalCommandProcessor builds these inputs and calls the same predicate, so a
// rule dropped in production fails here.

#include "xenia/gpu/metal/metal_cache_clear_admission.h"

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// The state at the frame boundary the clear is designed to run at: request
// latched, encoder closed, every submission complete, nothing left to publish.
CacheClearAdmissionInputs Executable() {
  CacheClearAdmissionInputs in{};
  in.request_latched = true;
  in.encoder_open = false;
  in.last_submitted_submission = 12;
  in.completed_submission = 12;
  in.has_pending_producers = false;
  in.publication_failed = false;
  in.can_end_submission = true;
  in.can_publish_pending = true;
  return in;
}

TEST_CASE("Cache clear does nothing without a request", "[cacheclear]") {
  CacheClearAdmissionInputs in = Executable();
  in.request_latched = false;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kNoRequest);

  // The no-request answer must not depend on any other input: this is the hot
  // path every title that never presses F5 takes, and it must be free.
  in.encoder_open = true;
  in.has_unsubmitted_work = true;
  in.completed_submission = 0;
  in.has_pending_producers = true;
  in.publication_failed = true;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kNoRequest);
}

TEST_CASE("Uncommitted encoded work blocks release", "[cacheclear]") {
  // The state the submission comparison alone cannot see: no encoder is open
  // and every submitted buffer has completed, but the frame command buffer
  // holds encoded draws that reference the resources a release would free.
  CacheClearAdmissionInputs in = Executable();
  in.encoder_open = false;
  in.has_unsubmitted_work = true;
  CHECK(DecideCacheClearAdmission(in) ==
        CacheClearAdmission::kEndSubmissionFirst);

  in.can_end_submission = false;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kDefer);
}

TEST_CASE("Cache clear executes only when fully quiesced", "[cacheclear]") {
  CHECK(DecideCacheClearAdmission(Executable()) ==
        CacheClearAdmission::kExecute);
}

TEST_CASE("A request arriving mid-encoder defers rather than releasing",
          "[cacheclear]") {
  CacheClearAdmissionInputs in = Executable();
  in.encoder_open = true;
  // The call site may close the submission: end the encoder and wait first.
  CHECK(DecideCacheClearAdmission(in) ==
        CacheClearAdmission::kEndSubmissionFirst);

  // A call site that may not end a submission can only wait for a later
  // boundary. It must never fall through to a release.
  in.can_end_submission = false;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kDefer);
}

TEST_CASE("A request with work in flight never releases", "[cacheclear]") {
  CacheClearAdmissionInputs in = Executable();
  in.encoder_open = false;
  in.last_submitted_submission = 12;
  in.completed_submission = 11;  // one submission still reading resources.
  CHECK(DecideCacheClearAdmission(in) ==
        CacheClearAdmission::kEndSubmissionFirst);

  in.can_end_submission = false;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kDefer);

  // Completion caught up: the in-flight objection is gone.
  in.completed_submission = 12;
  in.can_end_submission = true;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kExecute);
}

TEST_CASE("A pending writeback must be published before any release",
          "[cacheclear]") {
  // Waiting is not publishing. Every submission has completed here, which is
  // what makes the drain POSSIBLE -- it does not perform it, and the queued
  // record is guest-visible data that has not yet reached RAM.
  CacheClearAdmissionInputs in = Executable();
  in.has_pending_producers = true;
  CHECK(DecideCacheClearAdmission(in) ==
        CacheClearAdmission::kPublishPendingFirst);

  // A call site that cannot drain must defer, not release.
  in.can_publish_pending = false;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kDefer);
}

TEST_CASE("A failed publication keeps the request latched", "[cacheclear]") {
  // The boundary published and did not converge every producer. Releasing now
  // would drop the record the retry depends on.
  CacheClearAdmissionInputs in = Executable();
  in.publication_failed = true;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kDefer);

  // And it must not ask for ANOTHER publication in the same frame: the
  // records that failed to converge are exactly the ones still pending, so a
  // retry here would repeat the same drain and its waits against unchanged
  // state. The failure outranks the pending-producer branch.
  in.has_pending_producers = true;
  in.can_publish_pending = true;
  CHECK(DecideCacheClearAdmission(in) == CacheClearAdmission::kDefer);
}

TEST_CASE("No unquiesced state can ever reach execution", "[cacheclear]") {
  // The invariant, over every combination of the hazardous inputs rather than
  // over the cases picked above: kExecute implies encoder closed, all work
  // complete, nothing pending, and no failed publication.
  for (int bits = 0; bits < 128; ++bits) {
    CacheClearAdmissionInputs in{};
    in.request_latched = true;
    in.encoder_open = (bits & 1) != 0;
    in.last_submitted_submission = 9;
    in.completed_submission = (bits & 2) ? 9 : 4;
    in.has_pending_producers = (bits & 4) != 0;
    in.publication_failed = (bits & 8) != 0;
    in.can_end_submission = (bits & 16) != 0;
    in.can_publish_pending = (bits & 32) != 0;
    in.has_unsubmitted_work = (bits & 64) != 0;

    CacheClearAdmission decision = DecideCacheClearAdmission(in);
    if (decision == CacheClearAdmission::kExecute) {
      CHECK_FALSE(in.encoder_open);
      CHECK_FALSE(in.has_unsubmitted_work);
      CHECK(in.completed_submission >= in.last_submitted_submission);
      CHECK_FALSE(in.has_pending_producers);
      CHECK_FALSE(in.publication_failed);
    }
    // A failed publication never asks for another publication attempt at the
    // same boundary, whatever else is true.
    if (in.publication_failed &&
        in.completed_submission >= in.last_submitted_submission &&
        !in.encoder_open && !in.has_unsubmitted_work) {
      CHECK(decision == CacheClearAdmission::kDefer);
    }
    // And a latched request is never answered with kNoRequest, which is the
    // only outcome that would silently drop it.
    CHECK(decision != CacheClearAdmission::kNoRequest);
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
