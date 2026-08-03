/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_FAULT_EPISODE_H_
#define XENIA_GPU_METAL_METAL_FAULT_EPISODE_H_

// GPU fault EPISODE accounting, as a pure state machine so the grouping rules
// are pinned by the L0 suite instead of being validated only when a stochastic
// hardware fault happens to occur.
//
// WHY EPISODES EXIST: one GPU failure makes every in-flight submission complete
// with an error, and a revoked queue fails everything until it is recreated. So
// a session can log dozens of completion errors and many FAULTED encoder lines
// from a handful of real events. Measured on Viva Pinata (both_r1): 45
// completion-error headers across 19 swaps carried 5 FAULTED encoder lines and
// 2 queue revokes -- most plausibly ~2 episodes with a large collateral radius.
//
// Fault magnitude therefore needs FOUR counts, none of which substitutes for
// another: EPISODES, distinct failed submission serials, fault-bearing swaps,
// and FAULTED encoder lines.
//
// MEMBERSHIP RULE: an episode is keyed by the QUEUE EPOCH (bumped on every
// revoke/recreate) -- all failures observed within one epoch are one episode.
// The ROOT is the LOWEST failed submission serial in the episode, NOT the
// first-reported one: completion callbacks may arrive out of submission order,
// so a lower serial reporting later is the truer root and revises it downward.

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

struct FaultEpisodeSummary {
  uint64_t episode = 0;
  uint64_t epoch = 0;
  uint64_t root_serial = 0;      // Lowest failed serial: the one to root-cause.
  uint64_t failed_submissions = 0;  // Members, i.e. blast radius.
  bool valid = false;
};

class FaultEpisodeTracker {
 public:
  // Assigns `submission` to an episode, opening one if the epoch changed.
  // Returns the episode id. If this closes a previous episode, `closed`
  // receives its summary so the caller can emit it.
  uint64_t NoteMember(uint64_t submission, FaultEpisodeSummary* closed) {
    if (closed) {
      *closed = FaultEpisodeSummary();
    }
    if (open_ && open_epoch_ != epoch_) {
      if (closed) {
        *closed = Current();
      }
      Reset();
    }
    if (!open_) {
      open_ = true;
      open_id_ = ++counter_;
      open_epoch_ = epoch_;
      root_serial_ = submission;
      members_ = 0;
    }
    // Out-of-order completion: revise the root downward.
    if (submission && submission < root_serial_) {
      root_serial_ = submission;
    }
    ++members_;
    return open_id_;
  }

  // Closes the open episode (queue recreated, session end). Returns an invalid
  // summary when nothing is open, so closing twice is harmless.
  FaultEpisodeSummary Close() {
    if (!open_) {
      return FaultEpisodeSummary();
    }
    FaultEpisodeSummary summary = Current();
    Reset();
    return summary;
  }

  // Queue revoke/recreate. Does NOT itself close the episode -- the caller
  // closes explicitly so the summary is emitted with a reason.
  void BumpEpoch() { ++epoch_; }

  uint64_t epoch() const { return epoch_; }
  uint64_t episode_count() const { return counter_; }
  bool open() const { return open_; }

 private:
  FaultEpisodeSummary Current() const {
    FaultEpisodeSummary s;
    s.episode = open_id_;
    s.epoch = open_epoch_;
    s.root_serial = root_serial_;
    s.failed_submissions = members_;
    s.valid = true;
    return s;
  }
  void Reset() {
    open_ = false;
    open_id_ = 0;
    root_serial_ = 0;
    members_ = 0;
  }

  uint64_t epoch_ = 0;
  uint64_t counter_ = 0;
  bool open_ = false;
  uint64_t open_id_ = 0;
  uint64_t open_epoch_ = 0;
  uint64_t root_serial_ = 0;
  uint64_t members_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_FAULT_EPISODE_H_
