/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_WATCH_ADMISSION_H_
#define XENIA_GPU_WATCH_ADMISSION_H_

#include <algorithm>
#include <cstdint>
#include <vector>

// Pure (allocation-light, device-free) admission scan for WATCH-ONLY CPU-write
// notification over a guest byte extent.
//
// Motivation: a consumer that reads guest RAM DIRECTLY (the Metal legacy
// texture untile) needs to learn when the CPU rewrites those bytes in place,
// but must NOT drag in the shared GPU buffer's upload/validity contract — that
// contract belongs to consumers who actually read the shared buffer. Ordinary
// RequestRange/UploadRanges semantics are therefore left untouched, and this
// scan feeds a separate registration path that only enables invalidation
// callbacks.
//
// Two granularities matter and they differ on Apple Silicon:
//   * SYSTEM pages (== shared-memory pages, 16 KiB here) are the unit that host
//     memory protection can actually protect;
//   * GUEST heap pages (4 KiB for the small physical heap) are the unit whose
//     mapping/protection state the guest controls.
// Protecting a system page whose covered guest pages are not ALL writable can
// trap writes the unprotect path will not release (the guest page table says
// the page was never protected), so a system page is admitted only when every
// guest page it covers is writable. Unmapped holes anywhere in the extent —
// leading, trailing, or interior — are skipped rather than failing the whole
// request: a partially admitted extent still watches everything it can, and a
// completely unadmitted extent is not an error, only an absence of watching.

namespace xe {
namespace gpu {

struct WatchAdmissionRun {
  uint32_t start = 0;   // Byte address, system-page aligned.
  uint32_t length = 0;  // Bytes, whole system pages.
};

// Appends the system-page-aligned runs of [start, start+length) that may safely
// have CPU-write invalidation callbacks enabled, and returns the total admitted
// byte count. `is_guest_page_writable(guest_page_index)` reports whether one
// guest page is currently mapped AND writable. Runs are contiguous, ascending,
// and maximal (adjacent admissible system pages coalesce).
template <typename IsGuestPageWritableFn>
inline uint32_t ComputeWatchAdmissionRuns(
    uint32_t start, uint32_t length, uint32_t system_page_log2,
    uint32_t guest_page_log2, uint32_t buffer_size,
    IsGuestPageWritableFn&& is_guest_page_writable,
    std::vector<WatchAdmissionRun>& out_runs) {
  if (!length || start >= buffer_size) {
    return 0;
  }
  length = std::min(length, buffer_size - start);
  uint32_t system_page_size = uint32_t(1) << system_page_log2;
  uint32_t system_page_first = start >> system_page_log2;
  uint32_t system_page_last = (start + length - 1) >> system_page_log2;

  uint32_t admitted_bytes = 0;
  uint32_t run_first = UINT32_MAX;
  for (uint32_t page = system_page_first; page <= system_page_last; ++page) {
    // Every guest page covered by this system page must be writable.
    uint32_t page_start = page << system_page_log2;
    uint32_t guest_page_first = page_start >> guest_page_log2;
    uint32_t guest_page_last =
        (page_start + system_page_size - 1) >> guest_page_log2;
    bool admissible = true;
    for (uint32_t guest_page = guest_page_first; guest_page <= guest_page_last;
         ++guest_page) {
      if (!is_guest_page_writable(guest_page)) {
        admissible = false;
        break;
      }
    }
    if (admissible) {
      if (run_first == UINT32_MAX) {
        run_first = page;
      }
    } else if (run_first != UINT32_MAX) {
      uint32_t run_bytes = (page - run_first) << system_page_log2;
      out_runs.push_back(
          WatchAdmissionRun{run_first << system_page_log2, run_bytes});
      admitted_bytes += run_bytes;
      run_first = UINT32_MAX;
    }
  }
  if (run_first != UINT32_MAX) {
    uint32_t run_bytes = (system_page_last + 1 - run_first) << system_page_log2;
    out_runs.push_back(
        WatchAdmissionRun{run_first << system_page_log2, run_bytes});
    admitted_bytes += run_bytes;
  }
  return admitted_bytes;
}

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_WATCH_ADMISSION_H_
