/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_LAZY_PUBLICATION_H_
#define XENIA_GPU_METAL_METAL_LAZY_PUBLICATION_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "xenia/gpu/metal/metal_tiled_publish_run.h"
#include "xenia/gpu/texture_address.h"

namespace xe {
namespace gpu {
namespace metal {

// Stage 2 of optimize-metal-gpu-resident-resolve-coherence: demand-driven RAM
// publication of GPU-resident resolve output.
//
// A GPU-current PendingWriteback owes guest RAM its bytes. Instead of copying
// them at every swap / fence packet, the command processor ARMS the record
// here: every 4 KiB guest page of the record's tiled extent is mapped to the
// record (page -> owners) and the physical heap protects the pages no-access
// with a read watch. The first CPU touch of a page faults, the physical heap
// calls the read callback, and this table copies the OWNING records' bytes for
// the touched pages from the shared buffer (CPU-visible, already at their
// guest tiled addresses) into guest RAM, on the faulting thread, after those
// records' submissions have completed. Pages the CPU never touches are never
// copied.
//
// A page can have several owners (two resolves sharing a page, an export
// overlapping a resolve). The shared buffer is the single source of the bytes,
// so an owner only contributes COVERAGE (which bytes are GPU-written and may
// be copied); the fault copies the union of the owners' coverage after waiting
// for the newest owning submission. Bytes outside every owner's coverage are
// never copied: the shared buffer may hold stale content there.
//
// Thread model: the GPU thread arms and retires; any guest thread may fault.
// All table state is behind `mutex_`. The copy itself reads the shared
// buffer contents (stable pointer) and writes RAM through the physical alias,
// which the watch does not protect.
class MetalLazyPublication {
 public:
  static constexpr uint32_t kPageShift = 12;
  static constexpr uint32_t kPageSize = 1u << kPageShift;

  // Geometry of a GPU-current record, copied so a fault never touches the
  // command processor's pending list.
  struct Owner {
    uint64_t authority_sequence = 0;
    uint64_t submission = 0;  // 0 = untagged, never fault-publishable.
    uint32_t guest_addr = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bpp = 0;
    uint32_t dest_pitch_aligned = 0;  // texels; 0 = align32(width)
    uint32_t dest_offset_x = 0;
    uint32_t dest_offset_y = 0;
    uint32_t dest_extent_bytes = 0;
    uint32_t range_bytes = 0;
    // Memexport output: a plain linear range in the shared buffer.
    bool linear = false;
  };

  struct Stats {
    uint64_t armed_records = 0;
    uint64_t armed_pages = 0;
    uint64_t faults = 0;
    uint64_t fault_pages_copied = 0;
    uint64_t fault_bytes_copied = 0;
    uint64_t stale_reads = 0;  // owner not yet committed at fault time
    uint64_t retired_records = 0;
  };

  // Arms `owner`: records page ownership for [guest_addr, guest_addr +
  // range_bytes) and returns the page range so the caller can enable the read
  // watch. A linear owner that covers a whole page replaces that page's older
  // owners (their coverage is fully superseded); otherwise owners accumulate.
  void Arm(const Owner& owner, uint32_t* out_first_page,
           uint32_t* out_page_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t first = owner.guest_addr >> kPageShift;
    uint32_t last = (owner.guest_addr + owner.range_bytes - 1) >> kPageShift;
    std::vector<uint32_t>& owned = owner_pages_[owner.authority_sequence];
    owned.reserve(owned.size() + (last - first + 1));
    for (uint32_t page = first; page <= last; ++page) {
      std::vector<Owner>& owners = pages_[page];
      if (owner.linear && owner.guest_addr <= (page << kPageShift) &&
          owner.guest_addr + owner.range_bytes >= ((page + 1) << kPageShift)) {
        for (const Owner& old : owners) {
          ForgetPage(old.authority_sequence, page);
        }
        owners.clear();
      } else {
        // Re-arming the same record (watch re-enable) must not duplicate it.
        bool present = false;
        for (const Owner& old : owners) {
          if (old.authority_sequence == owner.authority_sequence) {
            present = true;
            break;
          }
        }
        if (present) {
          continue;
        }
      }
      owners.push_back(owner);
      owned.push_back(page);
    }
    ++stats_.armed_records;
    stats_.armed_pages += last - first + 1;
    if (out_first_page) {
      *out_first_page = first;
    }
    if (out_page_count) {
      *out_page_count = last - first + 1;
    }
  }

  // Drops every page still owned by `authority_sequence` (the record was
  // published eagerly or superseded). Returns the number of pages dropped.
  uint32_t Retire(uint64_t authority_sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = owner_pages_.find(authority_sequence);
    if (it == owner_pages_.end()) {
      return 0;
    }
    uint32_t dropped = 0;
    for (uint32_t page : it->second) {
      dropped += RemoveOwnerFromPage(page, authority_sequence) ? 1u : 0u;
    }
    owner_pages_.erase(it);
    if (dropped) {
      ++stats_.retired_records;
    }
    return dropped;
  }

  // Drops ownership of every page inside [phys_start, phys_start + length):
  // the contents are irrelevant now (allocation, release, host overwrite).
  uint32_t DropRange(uint32_t phys_start, uint32_t length) {
    if (!length) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t first = phys_start >> kPageShift;
    uint32_t last = (phys_start + length - 1) >> kPageShift;
    uint32_t dropped = 0;
    for (uint32_t page = first; page <= last; ++page) {
      auto it = pages_.find(page);
      if (it == pages_.end()) {
        continue;
      }
      for (const Owner& old : it->second) {
        ForgetPage(old.authority_sequence, page);
      }
      pages_.erase(it);
      ++dropped;
    }
    return dropped;
  }

  bool Owns(uint64_t authority_sequence) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = owner_pages_.find(authority_sequence);
    return it != owner_pages_.end() && !it->second.empty();
  }

  // Appends the page-granular sub-ranges of [phys_start, phys_start + length)
  // that some armed owner still holds (GPU output not yet published to RAM),
  // coalesced and clipped to the query. Pages the table does not own have
  // been published (RAM holds the GPU bytes) or were never GPU-written.
  void OwnedRanges(uint32_t phys_start, uint32_t length,
                   std::vector<std::pair<uint32_t, uint32_t>>* out) const {
    if (!length) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t end = uint64_t(phys_start) + length;
    uint32_t first = phys_start >> kPageShift;
    uint32_t last = uint32_t((end - 1) >> kPageShift);
    uint64_t run_start = 0, run_end = 0;
    for (uint32_t page = first; page <= last; ++page) {
      auto it = pages_.find(page);
      if (it == pages_.end() || it->second.empty()) {
        continue;
      }
      uint64_t p0 = std::max<uint64_t>(uint64_t(page) << kPageShift,
                                       phys_start);
      uint64_t p1 = std::min<uint64_t>(uint64_t(page + 1) << kPageShift, end);
      if (run_end && p0 == run_end) {
        run_end = p1;
      } else {
        if (run_end) {
          out->emplace_back(uint32_t(run_start), uint32_t(run_end - run_start));
        }
        run_start = p0;
        run_end = p1;
      }
    }
    if (run_end) {
      out->emplace_back(uint32_t(run_start), uint32_t(run_end - run_start));
    }
  }

  // Returns the oldest armed owner registered before `sequence` whose byte
  // range overlaps [phys_start, phys_start + length), or 0. Such an owner's
  // first-access copy would land over a newer record's bytes published to RAM
  // in that range (memexport/resolve ordering census).
  uint64_t OldestOwnerBefore(uint32_t phys_start, uint32_t length,
                             uint64_t sequence) const {
    if (!length) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t oldest = 0;
    uint64_t end = uint64_t(phys_start) + length;
    uint32_t first = phys_start >> kPageShift;
    uint32_t last = uint32_t((end - 1) >> kPageShift);
    for (uint32_t page = first; page <= last; ++page) {
      auto it = pages_.find(page);
      if (it == pages_.end()) {
        continue;
      }
      for (const Owner& owner : it->second) {
        if (owner.authority_sequence >= sequence ||
            (oldest && owner.authority_sequence >= oldest)) {
          continue;
        }
        uint64_t owner_end = uint64_t(owner.guest_addr) + owner.range_bytes;
        if (owner.guest_addr < end && phys_start < owner_end) {
          oldest = owner.authority_sequence;
        }
      }
    }
    return oldest;
  }

  // Fault path. `wait_for_submission(submission)` returns true once the
  // owner's submission has completed on the GPU (false if it cannot be
  // waited for, e.g. not yet committed). `shared_contents` is the shared
  // buffer's CPU pointer (guest physical offset == buffer offset);
  // `translate_physical(addr)` returns the RAM pointer for a guest physical
  // address. Copies only the touched pages, and within them only the owners'
  // coverage. A page whose owners cannot all be waited for stays armed (the
  // caller re-arms the watch) and is counted as a stale read.
  //
  // Returns the number of pages that remain unpublished in the range.
  uint32_t OnAccess(uint32_t phys_start, uint32_t length,
                    const std::function<bool(uint64_t)>& wait_for_submission,
                    const uint8_t* shared_contents,
                    const std::function<uint8_t*(uint32_t)>& translate_physical) {
    if (!length) {
      return 0;
    }
    uint32_t first = phys_start >> kPageShift;
    uint32_t last = (phys_start + length - 1) >> kPageShift;
    // Snapshot the owners under the lock; copy without it.
    std::vector<std::pair<uint32_t, std::vector<Owner>>> touched;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++stats_.faults;
      for (uint32_t page = first; page <= last; ++page) {
        auto it = pages_.find(page);
        if (it != pages_.end() && !it->second.empty()) {
          touched.emplace_back(page, it->second);
        }
      }
    }
    if (touched.empty()) {
      return 0;
    }
    uint32_t remaining = 0;
    for (const auto& [page, owners] : touched) {
      // Every owner's submission must be complete before any coverage is
      // copied: a newer owner may have overwritten an older one's bytes.
      bool waitable = true;
      for (const Owner& owner : owners) {
        if (!owner.submission || !wait_for_submission(owner.submission)) {
          waitable = false;
          break;
        }
      }
      if (!waitable) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.stale_reads;
        ++remaining;
        continue;
      }
      uint64_t bytes = 0;
      for (const Owner& owner : owners) {
        bytes += CopyPage(owner, page, shared_contents, translate_physical);
      }
      std::lock_guard<std::mutex> lock(mutex_);
      for (const Owner& owner : owners) {
        RemoveOwnerFromPage(page, owner.authority_sequence);
        ForgetPage(owner.authority_sequence, page);
      }
      ++stats_.fault_pages_copied;
      stats_.fault_bytes_copied += bytes;
    }
    return remaining;
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  size_t armed_page_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pages_.size();
  }

 private:
  // Removes `authority_sequence` from `page`'s owner list; erases the page
  // when no owner remains. Returns true when the owner was present.
  bool RemoveOwnerFromPage(uint32_t page, uint64_t authority_sequence) {
    auto it = pages_.find(page);
    if (it == pages_.end()) {
      return false;
    }
    std::vector<Owner>& owners = it->second;
    bool removed = false;
    for (size_t i = 0; i < owners.size();) {
      if (owners[i].authority_sequence == authority_sequence) {
        owners.erase(owners.begin() + i);
        removed = true;
      } else {
        ++i;
      }
    }
    if (owners.empty()) {
      pages_.erase(it);
    }
    return removed;
  }

  // Drops `page` from the owner's page index (the owner no longer covers it).
  void ForgetPage(uint64_t authority_sequence, uint32_t page) {
    auto it = owner_pages_.find(authority_sequence);
    if (it == owner_pages_.end()) {
      return;
    }
    std::vector<uint32_t>& pages = it->second;
    for (size_t i = 0; i < pages.size(); ++i) {
      if (pages[i] == page) {
        pages[i] = pages.back();
        pages.pop_back();
        break;
      }
    }
    if (pages.empty()) {
      owner_pages_.erase(it);
    }
  }

  // Copies the owner's tiled runs that fall inside `page` from the shared
  // buffer into RAM. Same walk as the eager publication: rows of the resolve
  // rectangle, runs of contiguous tiled texels, clipped to dest_extent_bytes.
  // Only the 32-row tile bands whose bytes can reach the page are walked: a
  // band's tiled addresses lie within its pitch*32*bpp span, plus the 4 KiB
  // micro-tile shuffle on either side, so the others are skipped whole.
  static uint64_t CopyPage(
      const Owner& o, uint32_t page, const uint8_t* shared_contents,
      const std::function<uint8_t*(uint32_t)>& translate_physical) {
    const uint32_t page_start = page << kPageShift;
    const uint32_t page_end = page_start + kPageSize;
    uint8_t* ram = translate_physical(o.guest_addr);
    if (!ram || !shared_contents) {
      return 0;
    }
    if (o.linear) {
      uint32_t c0 = std::max(o.guest_addr, page_start);
      uint32_t c1 = std::min(o.guest_addr + o.range_bytes, page_end);
      if (c0 >= c1) {
        return 0;
      }
      std::memcpy(ram + (c0 - o.guest_addr),
                  shared_contents + c0, c1 - c0);
      return c1 - c0;
    }
    if (!o.bpp) {
      return 0;
    }
    const uint8_t* src_base = shared_contents + o.guest_addr;
    const uint32_t pitch_aligned =
        o.dest_pitch_aligned ? o.dest_pitch_aligned : ((o.width + 31u) & ~31u);
    uint32_t bl2 = 0;
    while ((1u << bl2) < o.bpp) {
      ++bl2;
    }
    // Same run geometry as the eager publication walk.
    const uint32_t run_texels = TiledPublishRunTexels(o.bpp);
    const uint64_t band_bytes = uint64_t(pitch_aligned) * 32u * o.bpp;
    const uint64_t page_lo = uint64_t(page_start) + 8192u;  // slack applied
    const uint64_t page_hi = uint64_t(page_end) + 8192u;    // symmetrically
    uint64_t copied = 0;
    for (uint32_t y = 0; y < o.height;) {
      const uint32_t tiled_y = o.dest_offset_y + y;
      const uint32_t band = tiled_y >> 5;
      const uint64_t band_start = uint64_t(o.guest_addr) + band * band_bytes;
      const uint64_t band_end = band_start + band_bytes;
      // Band span [band_start - 4 KiB, band_end + 4 KiB) vs the page, both
      // shifted by 8 KiB so the comparison never underflows.
      if (band_end + 4096u + 8192u <= page_lo || band_start + 8192u - 4096u >= page_hi) {
        // Skip to the first row of the next band.
        const uint32_t next_tiled_y = (band + 1u) << 5;
        y = next_tiled_y - o.dest_offset_y;
        continue;
      }
      for (uint32_t x = 0; x < o.width;) {
        uint32_t tiled_x = o.dest_offset_x + x;
        uint32_t run =
            std::min(run_texels - (tiled_x & (run_texels - 1)), o.width - x);
        uint32_t off = uint32_t(texture_address::Tiled2D(
            int32_t(o.dest_offset_x + x), int32_t(tiled_y),
            pitch_aligned, int32_t(bl2)));
        uint32_t run_bytes = run * o.bpp;
        if (o.dest_extent_bytes && off + run_bytes > o.dest_extent_bytes) {
          uint32_t fit =
              o.dest_extent_bytes > off ? (o.dest_extent_bytes - off) / o.bpp
                                        : 0u;
          run_bytes = fit * o.bpp;
        }
        if (run_bytes) {
          uint32_t abs0 = o.guest_addr + off;
          uint32_t abs1 = abs0 + run_bytes;
          uint32_t c0 = std::max(abs0, page_start);
          uint32_t c1 = std::min(abs1, page_end);
          if (c0 < c1) {
            std::memcpy(ram + (c0 - o.guest_addr),
                        src_base + (c0 - o.guest_addr), c1 - c0);
            copied += c1 - c0;
          }
        }
        x += run;
      }
      ++y;
    }
    return copied;
  }

  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, std::vector<Owner>> pages_;
  // authority_sequence -> pages it still owns (O(1) retire).
  std::unordered_map<uint64_t, std::vector<uint32_t>> owner_pages_;
  Stats stats_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_LAZY_PUBLICATION_H_
