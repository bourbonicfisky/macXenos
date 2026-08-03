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
// MEMBERSHIP RULE: an episode is keyed by the QUEUE EPOCH THE SUBMISSION WAS
// SUBMITTED UNDER -- never the epoch current when its callback is drained.
// Callbacks can arrive late, after the queue has already been revoked and
// recreated; keying on drain-time epoch would split one episode across two.
// Each submission carries its own epoch and episodes are tracked per-epoch
// concurrently.
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
  // Concurrently-open epochs. Small: late callbacks trail the current epoch by
  // at most a frame or two in practice. Overflow retires the OLDEST episode
  // (emitting its summary) rather than dropping attribution silently.
  static constexpr uint32_t kMaxOpenEpisodes = 8;

  // Assigns a failed `submission`, SUBMITTED under `submit_epoch`, to that
  // epoch's episode. Returns the episode id. If admitting it retired an older
  // episode, `retired` receives that summary so the caller can emit it.
  uint64_t NoteMember(uint64_t submission, uint64_t submit_epoch,
                      FaultEpisodeSummary* retired) {
    if (retired) {
      *retired = FaultEpisodeSummary();
    }
    Episode* episode = Find(submit_epoch);
    if (!episode) {
      if (count_ == kMaxOpenEpisodes) {
        uint32_t oldest = 0;
        for (uint32_t i = 1; i < count_; ++i) {
          if (open_[i].id < open_[oldest].id) {
            oldest = i;
          }
        }
        if (retired) {
          *retired = SummaryOf(open_[oldest]);
        }
        open_[oldest] = open_[--count_];
      }
      episode = &open_[count_++];
      episode->id = ++counter_;
      episode->epoch = submit_epoch;
      episode->root_serial = submission;
      episode->members = 0;
    }
    // Out-of-order completion: a lower serial is the truer root. Serial 0 is
    // the unknown/unowned sentinel elsewhere and must never win the min().
    if (submission &&
        (!episode->root_serial || submission < episode->root_serial)) {
      episode->root_serial = submission;
    }
    ++episode->members;
    return episode->id;
  }

  // Closes one epoch's episode, if open.
  FaultEpisodeSummary CloseEpoch(uint64_t epoch) {
    for (uint32_t i = 0; i < count_; ++i) {
      if (open_[i].epoch == epoch) {
        FaultEpisodeSummary summary = SummaryOf(open_[i]);
        open_[i] = open_[--count_];
        return summary;
      }
    }
    return FaultEpisodeSummary();
  }

  // Drains one open episode (teardown / end-of-session reporting).
  FaultEpisodeSummary CloseAny() {
    if (!count_) {
      return FaultEpisodeSummary();
    }
    FaultEpisodeSummary summary = SummaryOf(open_[count_ - 1]);
    --count_;
    return summary;
  }

  void BumpEpoch() { ++epoch_; }
  uint64_t epoch() const { return epoch_; }
  uint64_t episode_count() const { return counter_; }
  uint32_t open_episodes() const { return count_; }

 private:
  struct Episode {
    uint64_t id = 0;
    uint64_t epoch = 0;
    uint64_t root_serial = 0;
    uint64_t members = 0;
  };
  Episode* Find(uint64_t epoch) {
    for (uint32_t i = 0; i < count_; ++i) {
      if (open_[i].epoch == epoch) {
        return &open_[i];
      }
    }
    return nullptr;
  }
  static FaultEpisodeSummary SummaryOf(const Episode& e) {
    FaultEpisodeSummary s;
    s.episode = e.id;
    s.epoch = e.epoch;
    s.root_serial = e.root_serial;
    s.failed_submissions = e.members;
    s.valid = true;
    return s;
  }

  Episode open_[kMaxOpenEpisodes];
  uint32_t count_ = 0;
  uint64_t epoch_ = 0;
  uint64_t counter_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_FAULT_EPISODE_H_
