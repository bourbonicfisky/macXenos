/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// MetalEdramAuthorityTracker::version() contract, which lets
// MetalRenderTargetCache::PrepareCurrentRasterViews skip a repeat
// SynchronizeAttachment: if the version still equals its value right after
// SynchronizeAttachment(A, r), calling SynchronizeAttachment(A, r) again must
// return no operations and change no tile. Checked on random mixed operation
// sequences whose orders come from one increasing counter (the production
// canonical_authority_order_ discipline), including the per-draw
// MarkAttachment re-stamp that must NOT bump the version.

#include <cstdint>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_edram_authority.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using Tracker = metal::MetalEdramAuthorityTracker;
constexpr uint32_t kWindow = 96;  // dense overlap within a small tile window.

bool SameTiles(const Tracker& t, const std::vector<Tracker::TileState>& snap) {
  for (uint32_t i = 0; i < kWindow; ++i) {
    const auto& a = t.tile(i);
    const auto& b = snap[i];
    if (a.location != b.location ||
        a.authority_attachment != b.authority_attachment ||
        a.clean_attachment != b.clean_attachment ||
        a.submission_order != b.submission_order) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST_CASE("EDRAM authority: unchanged version proves a repeat attachment sync "
          "is a no-op",
          "[metal][edram-authority][version]") {
  uint64_t skippable_checks = 0;
  uint64_t restamps_without_bump = 0;
  for (uint32_t seed = 1; seed <= 200; ++seed) {
    std::mt19937 rng(seed);
    Tracker tracker;
    uint64_t order = 0;
    // (attachment, start, end) -> version right after its last sync.
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>, uint64_t> memo;
    auto pick_range = [&](uint32_t& s, uint32_t& e) {
      s = rng() % kWindow;
      e = s + 1 + rng() % (kWindow - s);
    };
    // A few fixed "render targets" so repeats of the same (A, r) are common.
    const struct { uint64_t a; uint32_t s, e; } targets[] = {
        {1, 0, 40}, {2, 30, 70}, {3, 60, 96}, {1, 20, 60}, {2, 0, 96}};
    for (int step = 0; step < 400; ++step) {
      uint32_t op = rng() % 100;
      if (op < 45) {
        // SynchronizeAttachment on a target (the per-draw prepare).
        const auto& t = targets[rng() % 5];
        auto key = std::make_tuple(t.a, t.s, t.e);
        auto it = memo.find(key);
        const bool skippable =
            it != memo.end() && it->second == tracker.version();
        std::vector<Tracker::TileState> snap(kWindow);
        for (uint32_t i = 0; i < kWindow; ++i) snap[i] = tracker.tile(i);
        auto ops = tracker.SynchronizeAttachment(t.a, t.s, t.e, ++order);
        if (skippable) {
          ++skippable_checks;
          INFO("seed " << seed << " step " << step);
          REQUIRE(ops.empty());
          REQUIRE(SameTiles(tracker, snap));
        }
        memo[key] = tracker.version();
      } else if (op < 75) {
        // Draw into a target: MarkAttachment over its range (often a pure
        // re-stamp right after its own sync).
        const auto& t = targets[rng() % 5];
        uint64_t before = tracker.version();
        bool all_owned = true;
        for (uint32_t i = t.s; i < t.e; ++i) {
          const auto& st = tracker.tile(i);
          all_owned &= st.location == Tracker::Location::kAttachment &&
                       st.authority_attachment == t.a &&
                       st.clean_attachment == t.a;
        }
        tracker.MarkAttachment(t.a, t.s, t.e, ++order);
        if (all_owned) {
          REQUIRE(tracker.version() == before);
          ++restamps_without_bump;
        }
      } else if (op < 85) {
        uint32_t s, e;
        pick_range(s, e);
        tracker.MarkCanonical(s, e, ++order);
      } else if (op < 93) {
        uint32_t s, e;
        pick_range(s, e);
        tracker.SynchronizeCanonical(s, e, ++order);
      } else if (op < 99) {
        tracker.InvalidateAttachment(1 + rng() % 3);
      } else {
        tracker.ResetUnknown();
      }
    }
  }
  // The property must actually have been exercised, re-stamps included.
  REQUIRE(skippable_checks > 1000);
  REQUIRE(restamps_without_bump > 100);
}

TEST_CASE("EDRAM authority: skipping a same-version re-mark changes nothing",
          "[metal][edram-authority][version]") {
  // MarkCurrentRasterTargetsWritten skips MarkAttachment(A, r) when A marked r
  // last and the version is unchanged since. Run a faithful tracker and a
  // skipping tracker in lockstep: every sync must return the same operations
  // (kind, attachment, range) and every tile the same location and owners.
  // Only submission-order stamps may differ; nothing outside the tracker
  // reads them and every comparison is against a newer counter value.
  uint64_t skipped_marks = 0;
  for (uint32_t seed = 1; seed <= 200; ++seed) {
    std::mt19937 rng(seed);
    Tracker faithful, skipping;
    uint64_t order = 0;
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>, uint64_t> mark_memo;
    auto pick_range = [&](uint32_t& s, uint32_t& e) {
      s = rng() % kWindow;
      e = s + 1 + rng() % (kWindow - s);
    };
    const struct { uint64_t a; uint32_t s, e; } targets[] = {
        {1, 0, 40}, {2, 30, 70}, {3, 60, 96}, {1, 20, 60}, {2, 0, 96}};
    auto same_ops = [](const std::vector<Tracker::SyncOperation>& x,
                       const std::vector<Tracker::SyncOperation>& y) {
      if (x.size() != y.size()) return false;
      for (size_t i = 0; i < x.size(); ++i) {
        if (x[i].kind != y[i].kind || x[i].attachment != y[i].attachment ||
            x[i].start_tiles != y[i].start_tiles ||
            x[i].end_tiles != y[i].end_tiles) {
          return false;
        }
      }
      return true;
    };
    for (int step = 0; step < 400; ++step) {
      uint32_t op = rng() % 100;
      if (op < 40) {
        const auto& t = targets[rng() % 5];
        uint64_t o = ++order;
        auto a = faithful.SynchronizeAttachment(t.a, t.s, t.e, o);
        auto b = skipping.SynchronizeAttachment(t.a, t.s, t.e, o);
        INFO("seed " << seed << " step " << step);
        REQUIRE(same_ops(a, b));
      } else if (op < 75) {
        const auto& t = targets[rng() % 5];
        uint64_t o = ++order;
        faithful.MarkAttachment(t.a, t.s, t.e, o);
        auto key = std::make_tuple(t.a, t.s, t.e);
        auto it = mark_memo.find(key);
        if (it != mark_memo.end() && it->second == skipping.version()) {
          ++skipped_marks;
        } else {
          skipping.MarkAttachment(t.a, t.s, t.e, o);
          mark_memo[key] = skipping.version();
        }
      } else if (op < 85) {
        uint32_t s, e;
        pick_range(s, e);
        uint64_t o = ++order;
        faithful.MarkCanonical(s, e, o);
        skipping.MarkCanonical(s, e, o);
      } else if (op < 93) {
        uint32_t s, e;
        pick_range(s, e);
        uint64_t o = ++order;
        auto a = faithful.SynchronizeCanonical(s, e, o);
        auto b = skipping.SynchronizeCanonical(s, e, o);
        REQUIRE(same_ops(a, b));
      } else if (op < 99) {
        uint64_t victim = 1 + rng() % 3;
        faithful.InvalidateAttachment(victim);
        skipping.InvalidateAttachment(victim);
      } else {
        faithful.ResetUnknown();
        skipping.ResetUnknown();
      }
      for (uint32_t i = 0; i < kWindow; ++i) {
        const auto& x = faithful.tile(i);
        const auto& y = skipping.tile(i);
        INFO("seed " << seed << " step " << step << " tile " << i);
        REQUIRE(x.location == y.location);
        REQUIRE(x.authority_attachment == y.authority_attachment);
        REQUIRE(x.clean_attachment == y.clean_attachment);
      }
    }
  }
  REQUIRE(skipped_marks > 1000);
}

TEST_CASE("EDRAM authority: foreign writes into a range bump the version",
          "[metal][edram-authority][version]") {
  Tracker tracker;
  uint64_t order = 0;
  tracker.MarkAttachment(1, 0, 10, ++order);
  tracker.SynchronizeAttachment(1, 0, 10, ++order);
  uint64_t v = tracker.version();
  // Same attachment re-stamp: no bump.
  tracker.MarkAttachment(1, 0, 10, ++order);
  REQUIRE(tracker.version() == v);
  // Another attachment takes tiles 5..8: bump, and the next sync of 1 packs.
  tracker.MarkAttachment(2, 5, 8, ++order);
  REQUIRE(tracker.version() != v);
  auto ops = tracker.SynchronizeAttachment(1, 0, 10, ++order);
  REQUIRE_FALSE(ops.empty());
  // Canonical write bumps; a later sync of 1 materializes.
  v = tracker.version();
  tracker.MarkCanonical(0, 3, ++order);
  REQUIRE(tracker.version() != v);
  REQUIRE_FALSE(tracker.SynchronizeAttachment(1, 0, 10, ++order).empty());
  // ResetUnknown always bumps.
  v = tracker.version();
  tracker.ResetUnknown();
  REQUIRE(tracker.version() != v);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
