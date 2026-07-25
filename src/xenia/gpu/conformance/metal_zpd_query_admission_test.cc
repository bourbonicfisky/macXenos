/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the ZPD query admission decision, above all its
// pool-exhaustion behavior.
//
// Exhaustion is the hardest state to reach for real -- it needs 8192
// concurrent segments on a live command processor -- and the most dangerous to
// get wrong: the strict path can block, and a wrong branch there deadlocks the
// guest rather than degrading it. MetalCommandProcessor::OpenZPDQuery builds
// these inputs and calls the same predicate, so a rule dropped in production
// fails here.

#include "xenia/gpu/metal/metal_zpd_query_admission.h"

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// A healthy open: pool ready, eligible encoder, a slot free.
ZPDQueryAdmissionInputs Admissible() {
  ZPDQueryAdmissionInputs in{};
  in.pool_ready = true;
  in.query_already_open = false;
  in.encoder_can_carry_query = true;
  in.has_free_slot = true;
  in.fast_mode = true;
  in.has_pending_resolve = false;
  in.oldest_pending_submission = 0;
  in.current_encoding_submission = 10;
  in.can_close_submission = false;
  return in;
}

// Exhausted, with one resolve pending in an ALREADY COMMITTED submission --
// the state where waiting can actually free a slot.
ZPDQueryAdmissionInputs ExhaustedWithCommittedResolve() {
  ZPDQueryAdmissionInputs in = Admissible();
  in.has_free_slot = false;
  in.fast_mode = false;
  in.has_pending_resolve = true;
  in.oldest_pending_submission = 7;
  in.current_encoding_submission = 10;
  return in;
}

TEST_CASE("ZPD admission opens when a slot is free", "[zpdadmit]") {
  CHECK(DecideZPDQueryAdmission(Admissible()) == ZPDQueryAdmission::kAcquire);

  // Strict mode with a free slot must not wait for anything.
  ZPDQueryAdmissionInputs strict = Admissible();
  strict.fast_mode = false;
  strict.has_pending_resolve = true;
  strict.oldest_pending_submission = 3;
  CHECK(DecideZPDQueryAdmission(strict) == ZPDQueryAdmission::kAcquire);
}

TEST_CASE("ZPD admission fails on an unusable pool or a double open",
          "[zpdadmit]") {
  ZPDQueryAdmissionInputs no_pool = Admissible();
  no_pool.pool_ready = false;
  CHECK(DecideZPDQueryAdmission(no_pool) == ZPDQueryAdmission::kFailed);

  // Opening a second segment would strand the first slot's offset for the rest
  // of the pass, which Metal cannot recover from.
  ZPDQueryAdmissionInputs already_open = Admissible();
  already_open.query_already_open = true;
  CHECK(DecideZPDQueryAdmission(already_open) == ZPDQueryAdmission::kFailed);

  // An unusable pool outranks everything else, including a free slot.
  ZPDQueryAdmissionInputs both = Admissible();
  both.pool_ready = false;
  both.query_already_open = true;
  CHECK(DecideZPDQueryAdmission(both) == ZPDQueryAdmission::kFailed);
}

TEST_CASE("ZPD admission defers when the encoder cannot carry a query",
          "[zpdadmit]") {
  // No encoder, or one built before the pool existed. Deferral (not failure)
  // is what lets BeginFrameEncoder rebuild the encoder and retry.
  ZPDQueryAdmissionInputs in = Admissible();
  in.encoder_can_carry_query = false;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kDefer);

  // Deferral outranks exhaustion: with no eligible encoder there is nothing to
  // wait for, so strict mode must not block here.
  in.has_free_slot = false;
  in.fast_mode = false;
  in.has_pending_resolve = true;
  in.oldest_pending_submission = 1;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kDefer);
}

TEST_CASE("ZPD admission reports exhaustion in fast modes rather than waiting",
          "[zpdadmit]") {
  // Fast modes trade accuracy for forward progress; the shared code
  // substitutes a minimal accumulated value instead of stalling the guest.
  ZPDQueryAdmissionInputs in = ExhaustedWithCommittedResolve();
  in.fast_mode = true;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kPoolExhausted);

  // Even with a committed resolve pending and permission to close.
  in.can_close_submission = true;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kPoolExhausted);
}

TEST_CASE("ZPD admission waits for the oldest committed resolve in strict mode",
          "[zpdadmit]") {
  CHECK(DecideZPDQueryAdmission(ExhaustedWithCommittedResolve()) ==
        ZPDQueryAdmission::kWaitForOldestResolve);

  // The boundary: a resolve one submission below the one being encoded has
  // been committed, so it can complete on its own.
  ZPDQueryAdmissionInputs boundary = ExhaustedWithCommittedResolve();
  boundary.oldest_pending_submission = 9;
  boundary.current_encoding_submission = 10;
  CHECK(DecideZPDQueryAdmission(boundary) ==
        ZPDQueryAdmission::kWaitForOldestResolve);
}

TEST_CASE("ZPD admission never waits on an uncommitted submission",
          "[zpdadmit]") {
  // The freeing resolve is in the command buffer still being encoded. Blocking
  // on it without committing first is the deadlock this branch exists to
  // prevent.
  ZPDQueryAdmissionInputs in = ExhaustedWithCommittedResolve();
  in.oldest_pending_submission = 10;
  in.current_encoding_submission = 10;

  in.can_close_submission = true;
  CHECK(DecideZPDQueryAdmission(in) ==
        ZPDQueryAdmission::kDeferAfterClosingSubmission);

  // Call sites that may not commit (encoder open in progress) simply retry.
  in.can_close_submission = false;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kDefer);

  // A resolve tagged beyond the current submission is equally unwaitable.
  in.oldest_pending_submission = 11;
  in.can_close_submission = true;
  CHECK(DecideZPDQueryAdmission(in) ==
        ZPDQueryAdmission::kDeferAfterClosingSubmission);
}

TEST_CASE("ZPD admission reports exhaustion when nothing is in flight",
          "[zpdadmit]") {
  // Strict mode with no pending resolve: no slot can ever come back by
  // waiting, so it must report exhaustion instead of blocking forever.
  ZPDQueryAdmissionInputs in = ExhaustedWithCommittedResolve();
  in.has_pending_resolve = false;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kPoolExhausted);

  in.can_close_submission = true;
  CHECK(DecideZPDQueryAdmission(in) == ZPDQueryAdmission::kPoolExhausted);
}

TEST_CASE("ZPD admission is a total function over its inputs", "[zpdadmit]") {
  // Every reachable combination must yield a decision, and strict mode must
  // never choose to block on work that has not been committed.
  for (int bits = 0; bits < 256; ++bits) {
    ZPDQueryAdmissionInputs in{};
    in.pool_ready = bits & 1;
    in.query_already_open = bits & 2;
    in.encoder_can_carry_query = bits & 4;
    in.has_free_slot = bits & 8;
    in.fast_mode = bits & 16;
    in.has_pending_resolve = bits & 32;
    in.can_close_submission = bits & 64;
    in.oldest_pending_submission = (bits & 128) ? 10 : 5;
    in.current_encoding_submission = 10;

    ZPDQueryAdmission decision = DecideZPDQueryAdmission(in);
    if (decision == ZPDQueryAdmission::kWaitForOldestResolve) {
      // Waiting is only ever chosen when the wait can actually be satisfied.
      CHECK(in.has_pending_resolve);
      CHECK(in.oldest_pending_submission < in.current_encoding_submission);
      CHECK_FALSE(in.fast_mode);
      CHECK(in.pool_ready);
      CHECK_FALSE(in.has_free_slot);
    }
    if (decision == ZPDQueryAdmission::kAcquire) {
      CHECK(in.pool_ready);
      CHECK_FALSE(in.query_already_open);
      CHECK(in.encoder_can_carry_query);
      CHECK(in.has_free_slot);
    }
    if (decision == ZPDQueryAdmission::kDeferAfterClosingSubmission) {
      CHECK(in.can_close_submission);
    }
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
