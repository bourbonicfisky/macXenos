/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                            *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_GUEST_RAM_WRITE_JOURNAL_H_
#define XENIA_GPU_GUEST_RAM_WRITE_JOURNAL_H_

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace xe {
namespace gpu {

// Copy-on-first-write journal of guest-RAM ranges a replayed frame writes
// (fix-replay-guest-ram-rollback design D1/D2). The trace player owns one and
// feeds it every backend-reported pre-write notification; before each replay
// loop N>0 it restores the captured pre-frame bytes so the frame's inputs are
// as captured (design D3), making `--edram_determinism` sound for fixtures
// whose frames write guest RAM (resolve writeback, memexport convergence, ZPD
// reports, Vulkan readback resolve).
//
// Purely interval + byte logic: guest-physical-to-host translation is
// injected, so the class is unit-testable device-free against a plain buffer
// (conformance L0). Entries never overlap — a capture request covering
// already-captured bytes captures only the uncovered gaps, which is exactly
// first-write-wins: the first capture of any byte holds its PRE-FRAME value,
// and later loops' captures of new ranges still see pre-frame content because
// every restore has already put the covered bytes back.
//
// Single-threaded by contract: all captures and restores run on the
// command-processor thread (the writer sites and TracePlayer's playback entry
// both live there).
class GuestRamWriteJournal {
 public:
  // Returns the host pointer for `length` bytes at guest physical `base`, or
  // null if the range cannot be mapped (a null skips that capture and counts
  // it in failed_captures() rather than faulting).
  using TranslateFn = std::function<uint8_t*(uint32_t base, uint32_t length)>;

  explicit GuestRamWriteJournal(TranslateFn translate)
      : translate_(std::move(translate)) {}

  // Called immediately BEFORE `length` bytes at `base` are overwritten.
  // Captures the current (pre-write) contents of every not-yet-captured byte
  // in the range; already-captured bytes are left untouched.
  void CaptureBeforeWrite(uint32_t base, uint32_t length) {
    if (!length) {
      return;
    }
    uint64_t end = uint64_t(base) + length;
    // Gather uncovered gaps first, then capture: capturing inserts entries,
    // which must not perturb the scan.
    std::vector<std::pair<uint32_t, uint32_t>> gaps;
    uint64_t cursor = base;
    auto it = entries_.upper_bound(base);
    if (it != entries_.begin()) {
      auto prev = std::prev(it);
      uint64_t prev_end = uint64_t(prev->first) + prev->second.size();
      if (prev_end > cursor) {
        cursor = prev_end;  // prefix already captured
      }
    }
    while (cursor < end) {
      while (it != entries_.end() &&
             uint64_t(it->first) + it->second.size() <= cursor) {
        ++it;
      }
      uint64_t gap_end = end;
      if (it != entries_.end() && it->first < end) {
        if (uint64_t(it->first) <= cursor) {
          cursor = uint64_t(it->first) + it->second.size();
          continue;
        }
        gap_end = it->first;
      }
      if (gap_end > cursor) {
        gaps.emplace_back(uint32_t(cursor), uint32_t(gap_end - cursor));
        cursor = gap_end;
      }
    }
    for (const auto& gap : gaps) {
      CaptureRange(gap.first, gap.second);
    }
  }

  // Copies every captured pre-image back to guest RAM. Entries are RETAINED:
  // their pre-frame bytes stay authoritative for every later loop, and
  // CaptureBeforeWrite finding them covered is what makes capture
  // first-write-only. Returns the number of ranges restored.
  size_t RestoreAll() {
    size_t restored = 0;
    for (const auto& entry : entries_) {
      uint8_t* host =
          translate_(entry.first, uint32_t(entry.second.size()));
      if (!host) {
        ++failed_restores_;
        continue;
      }
      std::memcpy(host, entry.second.data(), entry.second.size());
      ++restored;
    }
    return restored;
  }

  // Drops all captured state (used when playback moves to a different frame:
  // the pre-images belong to the old frame).
  void Clear() {
    entries_.clear();
    captured_bytes_ = 0;
    failed_captures_ = 0;
    failed_restores_ = 0;
  }

  size_t entry_count() const { return entries_.size(); }
  uint64_t captured_bytes() const { return captured_bytes_; }
  uint32_t failed_captures() const { return failed_captures_; }
  uint32_t failed_restores() const { return failed_restores_; }

 private:
  void CaptureRange(uint32_t base, uint32_t length) {
    uint8_t* host = translate_(base, length);
    if (!host) {
      ++failed_captures_;
      return;
    }
    std::vector<uint8_t> pre_image(host, host + length);
    captured_bytes_ += length;
    entries_.emplace(base, std::move(pre_image));
  }

  TranslateFn translate_;
  // base -> pre-image bytes; invariant: entries never overlap.
  std::map<uint32_t, std::vector<uint8_t>> entries_;
  uint64_t captured_bytes_ = 0;
  uint32_t failed_captures_ = 0;
  uint32_t failed_restores_ = 0;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_GUEST_RAM_WRITE_JOURNAL_H_
