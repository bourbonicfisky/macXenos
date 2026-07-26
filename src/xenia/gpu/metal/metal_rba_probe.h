/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_RBA_PROBE_H_
#define XENIA_GPU_METAL_METAL_RBA_PROBE_H_

// TEMPORARY resolve/bridge-attribution observer.
//
// OpenSpec change: correct-metal-resolve-bridge-authority, task group 2.
// REVERTED BY TASK 2.8. This file, and every call site that references it,
// are deleted before the change closes; the closure proves removal by grep and
// by byte-identity of the production source against base 2e1dee1c0.
//
// WHY IT EXISTS. The inherited signedness census attributed texture traffic to
// this seam by testing membership in MetalTextureCache::resolved_targets_.
// That map tracks SNAPSHOT-REGISTRATION LIFETIME, not guest resolve
// provenance, so a retired bridge reads "not resolve-backed" by construction.
// Design decision D2 requires attribution to name the BASE and the WRITER
// directly instead. This observer is that corrected method.
//
// WHAT IT PROVES, AND HOW (the distinctions matter -- an earlier revision of
// this observer counted publications and compared an upload against the
// PREVIOUS upload, which establishes neither of the two facts group 2 needs):
//
//   * task 2.2, content oracle -- at each publication seam the observer hashes
//     and shadows the EXACT guest range immediately BEFORE and immediately
//     AFTER the writer's RAM write. The reported changed-byte count is an
//     exact byte-for-byte pre/post difference over that range, not a hash
//     inequality and not an upload-to-upload delta.
//   * task 2.4, served-byte identity -- each upload's guest read is compared
//     with std::memcmp against the retained POST-image of every publication
//     range contained in that upload's extent. Identity is therefore proven on
//     the actual bytes; it is never inferred from a size or range match. A
//     publication's monotonic sequence number is retained so "the most recent
//     overlapping publication" is a recorded fact rather than an assumption.
//   * task 2.1, attribution -- uploads are keyed by (base, format, dimension,
//     extent, signedness), so a byte total names the base that produced it.
//   * task 2.4, class history -- the ResolvedFetchService each measured
//     upload's fetch received, obtained by calling the ONE production
//     classifier. Registration-at-fetch is recorded as a SEPARATE fact
//     alongside the class, never as a proxy for provenance.
//
// The existing XE_METAL_LINEAGE ramHash cannot serve either purpose: it hashes
// pitch*height*4 bytes and therefore reads past the extent of any format
// narrower than 32bpp (a k_16 destination is hashed over twice its extent).
//
// BEHAVIOUR NEUTRALITY. Every entry point returns immediately unless the
// observer is explicitly engaged by environment, and no production decision
// reads anything this file computes. All state is private to the aggregator.
// Following the recorded C1 defect of the signedness census, NOTHING is
// constructed on the disengaged path: callers guard with Engaged() and no
// payload object exists unless the observer is on.
//
// SINGLE PROCESS-WIDE INSTANCE. Call sites live in two translation units
// (metal_texture_cache.mm records uploads; metal_command_processor.mm records
// publications and swaps), and a split instance would silently report each
// half against an empty other half. Instance() is an inline function-local
// static, which the ODR already makes unique per binary, and the class carries
// default visibility so -fvisibility=hidden cannot duplicate the guard symbol.
// The emitted snapshot is additionally self-checking: it reports how many
// distinct TUs have recorded into this instance, so a split would be visible
// as tus=1 rather than tus=2 instead of being inferred from output looking
// plausible.
//
// ENGAGEMENT.
//   XE_METAL_RBA=<n>          n <= 0 or absent disengages. n > 0 engages the
//                             census and emits a snapshot every n swaps.
//   XE_METAL_RBA_CONTENT=<n>  0/absent disables the exact-range publication
//                             pre/post oracle and the served-byte identity
//                             check; 1 engages them from the first frame; n>1
//                             engages them once the run reaches swap n, so a
//                             title's boot path is not charged ~90 ms/frame
//                             and the sample describes a settled scene.
//                             Shadow images are bounded (kShadowBudgetBytes).
//
// Counters are CUMULATIVE and are never reset. A settled-window rate is
// obtained by differencing two snapshots, which keeps the observer free of
// hidden state and makes the measured window explicit in the evidence.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {

class __attribute__((visibility("default"))) MetalRbaCensus {
 public:
  // Which publication seam wrote a range. Recorded per event, unsampled --
  // the durable production logs sample (first 8, then 3 every 512), which is
  // enough to see a pattern but not to attribute a per-frame byte total.
  enum class Writer : uint32_t {
    kMemexportDrain = 0,
    kResolveWriteback = 1,
  };

  // Which translation unit recorded, for the split-instance self-check.
  enum class Tu : uint32_t {
    kTextureCache = 0,
    kCommandProcessor = 1,
  };

  // Mirrors ResolvedFetchService without including it, so this temporary
  // header adds no coupling to the production classifier's header. The caller
  // maps the production enum; it is never re-derived here.
  enum class Service : uint32_t {
    kNotResolveBacked = 0,
    kBridgeSnapshot = 1,
    kExactCoverSnapshot = 2,
    kRamContent = 3,
  };

  static bool Engaged() { return Period() > 0; }
  static uint32_t Period() {
    static const uint32_t p = [] {
      const char* v = std::getenv("XE_METAL_RBA");
      if (!v) {
        return 0u;
      }
      long n = std::strtol(v, nullptr, 10);
      return n > 0 ? uint32_t(n) : 0u;
    }();
    return p;
  }
  // 0 disables the content oracle; 1 engages it from the first frame; N > 1
  // engages it once the run has reached swap N. Deferred engagement exists
  // because the oracle costs ~90 ms/frame: paying that through a title's boot
  // and menu path would desync the scripted profile's wall-clock waits before
  // gameplay is ever reached, and would also mix boot traffic into a sample
  // that is supposed to describe a settled scene.
  static uint32_t ContentOracleStartSwap() {
    static const uint32_t s = [] {
      const char* v = std::getenv("XE_METAL_RBA_CONTENT");
      if (!v) {
        return 0u;
      }
      long n = std::strtol(v, nullptr, 10);
      return n > 0 ? uint32_t(n) : 0u;
    }();
    return Engaged() ? s : 0u;
  }
  static bool ContentOracleEngaged() {
    return Engaged() && Instance().ContentActive();
  }

  // --- Publication seams -------------------------------------------------
  //
  // Called immediately BEFORE the writer's RAM write, with the guest pointer
  // for the exact range about to be written. Captures the pre-image.
  static void PublishBegin(Writer writer, uint32_t base, uint32_t length,
                           const uint8_t* guest) {
    if (!Engaged()) {
      return;
    }
    Instance().PublishBeginImpl(writer, base, length, guest);
  }

  // Called immediately AFTER the writer's RAM write completes, with the same
  // range. Captures the post-image, computes the exact pre/post changed-byte
  // count, and assigns this publication its monotonic sequence number.
  static void PublishEnd(Writer writer, uint32_t base, uint32_t length,
                         const uint8_t* guest) {
    if (!Engaged()) {
      return;
    }
    Instance().PublishEndImpl(writer, base, length, guest);
  }

  // --- Consumer ----------------------------------------------------------
  //
  // One CPU texture upload that actually read guest RAM. `service` is the
  // class the production classifier returned for this same fetch, and
  // `registered_at_fetch` is whether resolved_targets_ held the base at that
  // moment -- recorded side by side and never conflated. `guest` points at the
  // upload's own guest source, so served-byte identity is checked against the
  // bytes this upload actually read.
  static void RecordUpload(uint32_t base, uint32_t format, uint32_t dimension,
                           uint32_t width, uint32_t height, bool is_signed,
                           uint32_t extent_bytes, bool registered_at_fetch,
                           Service service, const uint8_t* guest) {
    if (!Engaged()) {
      return;
    }
    Instance().RecordUploadImpl(base, format, dimension, width, height,
                                is_signed, extent_bytes, registered_at_fetch,
                                service, guest);
  }

  // Called once per swap with the pre-increment swap index. Emits on the
  // configured period.
  static void OnSwap(uint32_t swap_index) {
    if (!Engaged()) {
      return;
    }
    Instance().OnSwapImpl(swap_index);
  }

  static void Emit(const char* reason) {
    if (!Engaged()) {
      return;
    }
    Instance().EmitImpl(reason);
  }

  static void NoteTu(Tu tu) {
    if (!Engaged()) {
      return;
    }
    Instance().NoteTuImpl(tu);
  }

 private:
  // Shadow images exist to answer "did these exact bytes change" and "are the
  // served bytes the published bytes". Bounded so a pathological workload
  // cannot exhaust host memory through the observer; a range that would
  // exceed the budget is reported as declined, never as unchanged or matched.
  static constexpr uint64_t kShadowBudgetBytes = 512ull * 1024ull * 1024ull;
  static constexpr size_t kReportBuckets = 48;
  static constexpr size_t kServedReportRows = 48;

  static uint64_t Fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
      h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
  }

  struct PubKey {
    uint32_t writer;
    uint32_t base;
    uint32_t length;

    bool operator<(const PubKey& o) const {
      if (base != o.base) return base < o.base;
      if (length != o.length) return length < o.length;
      return writer < o.writer;
    }
  };

  struct PubState {
    uint64_t events = 0;
    uint64_t bytes = 0;
    // Exact pre/post evidence for the most recent publication.
    uint64_t sequence = 0;
    uint64_t pre_hash = 0;
    uint64_t post_hash = 0;
    bool pre_valid = false;
    bool post_valid = false;
    bool declined = false;
    std::vector<uint8_t> pre_image;
    std::vector<uint8_t> post_image;
    // Accumulated exact pre->post difference across publications.
    uint64_t compares = 0;
    uint64_t changed_bytes = 0;
    uint64_t compared_bytes = 0;
    uint64_t unchanged_publications = 0;
  };

  struct UploadKey {
    uint32_t base;
    uint32_t format;
    uint32_t dimension;
    uint32_t width;
    uint32_t height;
    uint32_t is_signed;

    bool operator<(const UploadKey& o) const {
      if (base != o.base) return base < o.base;
      if (format != o.format) return format < o.format;
      if (dimension != o.dimension) return dimension < o.dimension;
      if (width != o.width) return width < o.width;
      if (height != o.height) return height < o.height;
      return is_signed < o.is_signed;
    }
  };

  struct UploadBucket {
    uint64_t uploads = 0;
    uint64_t bytes = 0;
    uint64_t registered = 0;
    uint64_t not_registered = 0;
    uint64_t service_counts[4] = {0, 0, 0, 0};
  };

  // Served-byte identity, per (upload key x publication range).
  struct ServedKey {
    UploadKey upload;
    uint32_t pub_base;
    uint32_t pub_length;

    bool operator<(const ServedKey& o) const {
      if (upload < o.upload) return true;
      if (o.upload < upload) return false;
      if (pub_base != o.pub_base) return pub_base < o.pub_base;
      return pub_length < o.pub_length;
    }
  };

  struct ServedState {
    uint64_t checks = 0;
    uint64_t identical = 0;
    uint64_t differing = 0;
    uint64_t differing_bytes = 0;
    uint64_t last_pub_sequence = 0;
    uint64_t last_check_sequence_gap = 0;
  };

  static MetalRbaCensus& Instance() {
    static MetalRbaCensus instance;
    return instance;
  }

  // Reads swaps_ without the lock. This is an observer-only comparison against
  // a monotonically increasing counter written once per swap on the GPU
  // thread; a torn read can only mis-time engagement by one frame, and the
  // snapshot records content_from_swap so the engaged window is never
  // inferred. Taking the lock here would recurse from the locked paths that
  // consult it.
  bool ContentActive() const {
    uint32_t start = ContentOracleStartSwap();
    return start && (swaps_.load() + 1) >= start;
  }

  void NoteTuImpl(Tu tu) {
    std::lock_guard<std::mutex> lock(mutex_);
    tu_mask_ |= (1u << uint32_t(tu));
  }

  void PublishBeginImpl(Writer writer, uint32_t base, uint32_t length,
                        const uint8_t* guest) {
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      PubState& p = pubs_[PubKey{uint32_t(writer), base, length}];
      p.pre_valid = false;
      // Hashing and shadowing are the CONTENT ORACLE, not the census. Plain
      // XE_METAL_RBA counts publications only, so it stays cheap enough to run
      // beside a performance observation; XE_METAL_RBA_CONTENT=1 buys the
      // exact-range evidence and is explicitly NOT a performance-valid mode.
      if (!ContentOracleEngaged() || !guest || !length) {
        AccrueLocked(t0);
        return;
      }
      p.pre_hash = Fnv1a64(guest, length);
      if (!p.declined) {
        if (p.pre_image.size() != size_t(length)) {
          if (!ReserveShadowLocked(p, length)) {
            AccrueLocked(t0);
            return;
          }
          p.pre_image.resize(length);
        }
        std::memcpy(p.pre_image.data(), guest, length);
      }
      p.pre_valid = true;
    }
    AccrueOverhead(t0);
    return;
  }

  void PublishEndImpl(Writer writer, uint32_t base, uint32_t length,
                      const uint8_t* guest) {
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      PubState& p = pubs_[PubKey{uint32_t(writer), base, length}];
      ++p.events;
      p.bytes += length;
      p.sequence = ++publication_sequence_;
      p.post_valid = false;
      if (!ContentOracleEngaged() || !guest || !length) {
        AccrueLocked(t0);
        return;
      }
      p.post_hash = Fnv1a64(guest, length);
      if (!p.declined) {
        // Exact pre/post changed-byte count over the published range.
        if (p.pre_valid && p.pre_image.size() == size_t(length)) {
          uint64_t changed = 0;
          const uint8_t* prev = p.pre_image.data();
          for (uint32_t i = 0; i < length; ++i) {
            changed += (prev[i] != guest[i]) ? 1u : 0u;
          }
          ++p.compares;
          p.compared_bytes += length;
          p.changed_bytes += changed;
          if (changed == 0) {
            ++p.unchanged_publications;
          }
        }
        // Retain the post-image so a later upload can be proven byte-identical
        // to what this publication actually put in RAM.
        if (p.post_image.size() != size_t(length)) {
          if (!ReserveShadowLocked(p, length)) {
            AccrueLocked(t0);
            return;
          }
          p.post_image.resize(length);
        }
        std::memcpy(p.post_image.data(), guest, length);
        p.post_valid = true;
      }
    }
    AccrueOverhead(t0);
  }

  // Caller holds mutex_. Charges `length` against the shared budget; marks the
  // state declined and returns false when it would be exceeded.
  bool ReserveShadowLocked(PubState& p, uint32_t length) {
    if (shadow_bytes_ + length > kShadowBudgetBytes) {
      p.declined = true;
      return false;
    }
    shadow_bytes_ += length;
    return true;
  }

  void RecordUploadImpl(uint32_t base, uint32_t format, uint32_t dimension,
                        uint32_t width, uint32_t height, bool is_signed,
                        uint32_t extent_bytes, bool registered_at_fetch,
                        Service service, const uint8_t* guest) {
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      UploadKey key{base, format, dimension, width, height,
                    uint32_t(is_signed)};
      UploadBucket& b = uploads_[key];
      ++b.uploads;
      b.bytes += extent_bytes;
      if (registered_at_fetch) {
        ++b.registered;
      } else {
        ++b.not_registered;
      }
      ++b.service_counts[uint32_t(service) & 3u];

      if (!ContentOracleEngaged() || !guest || !extent_bytes) {
        AccrueLocked(t0);
        return;
      }
      // Served-byte identity: every retained publication range that lies
      // wholly inside this upload's extent is compared byte-for-byte against
      // the bytes this upload actually read. No size or range correlation is
      // used to claim identity.
      const uint64_t upload_end = uint64_t(base) + extent_bytes;
      for (auto& kv : pubs_) {
        const PubKey& pk = kv.first;
        PubState& p = kv.second;
        if (!p.post_valid || p.post_image.size() != size_t(pk.length)) {
          continue;
        }
        const uint64_t pub_end = uint64_t(pk.base) + pk.length;
        if (pk.base < base || pub_end > upload_end) {
          continue;
        }
        const uint8_t* served = guest + (pk.base - base);
        ServedState& s =
            served_[ServedKey{key, pk.base, pk.length}];
        ++s.checks;
        if (std::memcmp(served, p.post_image.data(), pk.length) == 0) {
          ++s.identical;
        } else {
          ++s.differing;
          uint64_t diff = 0;
          const uint8_t* pub = p.post_image.data();
          for (uint32_t i = 0; i < pk.length; ++i) {
            diff += (served[i] != pub[i]) ? 1u : 0u;
          }
          s.differing_bytes += diff;
        }
        s.last_check_sequence_gap = publication_sequence_ - p.sequence;
        s.last_pub_sequence = p.sequence;
      }
    }
    AccrueOverhead(t0);
  }

  void OnSwapImpl(uint32_t swap_index) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      swaps_.store(swap_index + 1);
    }
    uint32_t period = Period();
    if (period && ((swap_index + 1) % period) == 0) {
      EmitImpl("swap");
    }
  }

  void EmitImpl(const char* reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t swaps = swaps_.load();
    double frames = double(swaps ? swaps : 1);
    // Totals are emitted before the ranked lists so a truncated report can
    // never read as full coverage: the per-key lines below are bounded, these
    // totals are not.
    uint64_t total_uploads = 0, total_bytes = 0;
    for (const auto& kv : uploads_) {
      total_uploads += kv.second.uploads;
      total_bytes += kv.second.bytes;
    }
    uint32_t tus = uint32_t(__builtin_popcount(tu_mask_));
    XELOGW(
        "MetalRBA v1 snapshot reason={} swaps={} tus={} content_oracle={} "
        "content_from_swap={} "
        "upload_keys={} pub_keys={} served_keys={} total_uploads={} "
        "total_bytes={} total_bytes_per_frame={:.1f} publications={} "
        "overhead_usec={:.1f} overhead_usec_per_frame={:.1f} shadow_bytes={}",
        reason, swaps, tus, ContentOracleEngaged() ? 1 : 0,
        ContentOracleStartSwap(), uploads_.size(),
        pubs_.size(), served_.size(), total_uploads, total_bytes,
        double(total_bytes) / frames, publication_sequence_, overhead_usec_,
        overhead_usec_ / frames, shadow_bytes_);
    if (tus != 2) {
      XELOGW(
          "MetalRBA v1 WARNING split-instance: tu_mask={:02X} -- upload and "
          "publication records are NOT in one instance; do not compare them",
          tu_mask_);
    }

    // Rank by measured bytes so the pool the census cared about is named
    // first, by base, rather than located by a size match.
    std::vector<std::pair<UploadKey, const UploadBucket*>> ranked;
    ranked.reserve(uploads_.size());
    for (const auto& kv : uploads_) {
      ranked.emplace_back(kv.first, &kv.second);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      return a.second->bytes > b.second->bytes;
    });
    size_t emitted = 0;
    for (const auto& entry : ranked) {
      if (emitted++ >= kReportBuckets) {
        XELOGW("MetalRBA v1 upload-truncated swaps={} remaining_keys={}",
               swaps, ranked.size() - kReportBuckets);
        break;
      }
      const UploadKey& k = entry.first;
      const UploadBucket& b = *entry.second;
      XELOGW(
          "MetalRBA v1 upload swaps={} base={:08X} fmt={} dim={} {}x{} sig={} "
          "uploads={} bytes={} bytes_per_frame={:.1f} uploads_per_frame={:.4f} "
          "registered={} not_registered={} svc_nrb={} svc_bridge={} "
          "svc_exact={} svc_ram={}",
          swaps, k.base, k.format, k.dimension, k.width, k.height, k.is_signed,
          b.uploads, b.bytes, double(b.bytes) / frames,
          double(b.uploads) / frames, b.registered, b.not_registered,
          b.service_counts[0], b.service_counts[1], b.service_counts[2],
          b.service_counts[3]);
    }

    // Publication evidence: exact pre/post identity over the written range.
    for (const auto& kv : pubs_) {
      const PubKey& k = kv.first;
      const PubState& p = kv.second;
      std::string content;
      if (p.declined) {
        content = " content=declined-budget";
      } else if (p.compares) {
        content = fmt::format(
            " pub_compares={} pub_compared_bytes={} pub_changed_bytes={} "
            "pub_changed_frac={:.6f} pub_unchanged={}",
            p.compares, p.compared_bytes, p.changed_bytes,
            double(p.changed_bytes) / double(p.compared_bytes),
            p.unchanged_publications);
      } else if (ContentOracleEngaged()) {
        content = " content=no-preimage";
      }
      XELOGW(
          "MetalRBA v1 pub swaps={} kind={} base={:08X} len={} events={} "
          "bytes={} events_per_frame={:.4f} bytes_per_frame={:.1f} "
          "last_seq={} pre_hash={:016X} post_hash={:016X} "
          "post_equals_pre={}{}",
          swaps, k.writer == uint32_t(Writer::kMemexportDrain)
                      ? "memexport_drain"
                      : "resolve_writeback",
          k.base, k.length, p.events, p.bytes, double(p.events) / frames,
          double(p.bytes) / frames, p.sequence, p.pre_hash, p.post_hash,
          (p.pre_valid && p.post_valid && p.pre_hash == p.post_hash) ? 1 : 0,
          content);
    }

    // Served-byte identity per (upload key x publication range).
    size_t served_emitted = 0;
    for (const auto& kv : served_) {
      if (served_emitted++ >= kServedReportRows) {
        XELOGW("MetalRBA v1 served-truncated swaps={} remaining_rows={}",
               swaps, served_.size() - kServedReportRows);
        break;
      }
      const ServedKey& k = kv.first;
      const ServedState& s = kv.second;
      XELOGW(
          "MetalRBA v1 served swaps={} upload_base={:08X} fmt={} {}x{} sig={} "
          "pub_base={:08X} pub_len={} checks={} identical={} differing={} "
          "differing_bytes={} last_pub_seq={} seq_gap_at_check={}",
          swaps, k.upload.base, k.upload.format, k.upload.width,
          k.upload.height, k.upload.is_signed, k.pub_base, k.pub_length,
          s.checks, s.identical, s.differing, s.differing_bytes,
          s.last_pub_sequence, s.last_check_sequence_gap);
    }
    XELOGW("MetalRBA v1 end reason={} swaps={}", reason, swaps);
  }

  // Caller holds mutex_.
  void AccrueLocked(std::chrono::steady_clock::time_point t0) {
    overhead_usec_ +=
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0)
            .count();
  }

  void AccrueOverhead(std::chrono::steady_clock::time_point t0) {
    double usec = std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
    std::lock_guard<std::mutex> lock(mutex_);
    overhead_usec_ += usec;
  }

  std::mutex mutex_;
  std::map<UploadKey, UploadBucket> uploads_;
  std::map<PubKey, PubState> pubs_;
  std::map<ServedKey, ServedState> served_;
  uint64_t publication_sequence_ = 0;
  uint64_t shadow_bytes_ = 0;
  std::atomic<uint32_t> swaps_{0};
  uint32_t tu_mask_ = 0;
  double overhead_usec_ = 0.0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RBA_PROBE_H_
