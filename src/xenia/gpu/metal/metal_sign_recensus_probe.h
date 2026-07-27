/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SIGN_RECENSUS_PROBE_H_
#define XENIA_GPU_METAL_METAL_SIGN_RECENSUS_PROBE_H_

// TEMPORARY MEASUREMENT PROBE — DELETE BEFORE DELIVERY.
//
// OpenSpec change `correct-metal-resolve-bridge-authority`, task group 7: the
// behaviour-neutral re-census of the deferred signedness residual, run under
// the CORRECTED attribution method decided at task 2.1. NON-DELIVERY: this
// file and every reference to it are reverted at task 7.3, and the closure
// proves removal by grep plus byte-identity of the production source.
//
// LINEAGE. This is the signedness census probe of `d9c4dda26` (change
// `align-metal-texture-binding-signedness`, tasks 4.1/4.2) — its pool
// definitions, successful-only accounting, and honest overhead accounting are
// preserved verbatim so the two measurements are comparable — with ONE
// deliberate correction and one addition, both required by design decision D2
// of this change:
//
//   * REMOVED: the `resolve_backed` axis computed as
//     `resolved_targets_.find(base) != end()`. That map tracks SNAPSHOT-
//     REGISTRATION LIFETIME, not guest resolve provenance, so a retired bridge
//     reads "not resolve-backed" by construction. Using it as a provenance
//     proxy is exactly the defect this change was opened to correct, and the
//     spec (`metal-resolve-bridge-authority`, "Attribution SHALL NOT use
//     registration lifetime as a provenance proxy") forbids it.
//   * ADDED: writer attribution by the ACTUAL LAST WRITER of the consumed
//     bytes. Both publication seams (memexport drain and resolve writeback)
//     record the exact guest range they wrote, in publication order, and each
//     upload's extent is decomposed into the bytes whose most recent
//     publication came from each writer. "Some writer once published an
//     overlapping range" is NOT used to classify: JC2's `0x11BE1000` is
//     written by a resolve writeback and then overwritten across the consumed
//     bytes by three later memexport drains in the same frame (task 2.4), so a
//     union test would relabel required memexport traffic as resolve/bridge
//     work. The union history is still reported, as a separate fact.
//   * ADDED: per-base rows. Every measured upload is keyed by
//     (base, format, dimension, extent, signedness), so a byte or time total
//     NAMES the guest base that produced it instead of matching a size.
//   * RETAINED AS A SEPARATE FACT: whether `resolved_targets_` held the base at
//     the moment of the fetch, and which `ResolvedFetchService` the ONE
//     production classifier returned. Both are reported beside the writer
//     attribution and are never used as a provenance proxy for it.
//
// Contract this probe must satisfy (spec.md "Census does not alter behaviour"):
//   * every `GetTextureSlot` call the unprobed build makes is still made
//     exactly once, and every returned slot / allocation-failure result is used
//     exactly as before;
//   * nothing here owns sign semantics, routing, allocation, ordering, or
//     return values — the classification is computed from the same shared
//     predicates the production gate would use and is then only counted;
//   * engagement is a single temporary environment variable parsed once,
//     outside per-draw hot work, and NO probe payload is constructed or
//     initialised on the disengaged path — `MetalSignRecensusCall` lives in a
//     `std::optional` the call site engages only when the switch is on;
//   * output is aggregate and machine-parseable; no per-call log, no cvar, no
//     HUD, no shipped counter.
//
// Engagement: XE_METAL_SIGN_RECENSUS=1 (any value >= 1 engages; a value > 1
// also sets the cumulative-snapshot period in swaps, default 300).
//
// Counters are CUMULATIVE and are never reset. A settled-window rate is the
// difference of two snapshots, which keeps the observer free of hidden state
// and makes the measured window explicit in the evidence.
//
// Threading: every entry point is called from the GPU thread only (the
// per-binding marshalling loop, `MetalTextureCache::GetTextureSlot`, the
// memexport-drain and resolve-writeback publication seams, IssueSwap, and
// ShutdownContext all run there). No locking, deliberately — a lock on the
// per-binding path would be probe-visible cost of the same order as the
// bookkeeping it guards. The assumption is not assumed silently: the first
// recording thread is latched and every later record from a different thread
// is counted and reported, so a violation is visible in the snapshot rather
// than inferred from output looking plausible.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Facts about ONE `GetTextureSlot` call that only the legacy texture cache can
// observe accurately. The cache fills this when the census is engaged; it never
// reads it back and never branches on it, so it cannot change cache behaviour.
struct MetalSignRecensusProbeResult {
  // The call reached the keyed cache lookup (i.e. it was not rejected by the
  // fetch-type / dimension / format / bounds early-outs and was not served by
  // the resolve-snapshot bridge above the lookup).
  bool keyed_lookup = false;
  bool cache_hit = false;
  // The miss reached a successful CPU untile + upload + finalize.
  bool upload_completed = false;
  // This exact cache key had already been uploaded at least once in this run
  // (i.e. the upload is an invalidation-driven re-upload, not first sight).
  bool recurring = false;
  // CORRECTED ATTRIBUTION (task 2.1 / design D2). The guest base this fetch
  // read, the extent it read, and the last-writer decomposition of that
  // extent. Computed only for a completed upload, which is the only work any
  // pool below counts.
  uint32_t base = 0;
  uint32_t extent_bytes = 0;
  // Bytes of this upload's extent whose LAST writer is each publisher AND
  // whose publication is newer than this cache key's previous upload; bytes
  // last written before that watermark (so the write cannot explain this
  // read); bytes never written; and the union history as a separate fact.
  uint64_t resolve_since_bytes = 0;
  uint64_t memexport_since_bytes = 0;
  uint64_t stale_written_bytes = 0;
  uint64_t unwritten_bytes = 0;
  uint64_t resolve_ever_bytes = 0;
  uint64_t memexport_ever_bytes = 0;
  uint64_t newest_pub_seq_gap = 0;
  // Shape of the guest texture this upload read, for naming the row.
  uint32_t guest_format = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  // SEPARATE CONTEMPORANEOUS FACTS, never a provenance proxy: whether
  // `resolved_targets_` held the base at the moment of this fetch, and the
  // class the ONE production classifier returned for it.
  bool registered_at_fetch = false;
  uint32_t service = 0;  // mirrors ResolvedFetchService; mapped by the caller.
  // Host bytes actually submitted to Metal across base slice, cube faces, and
  // mip levels.
  uint64_t host_bytes = 0;
  // CPU microseconds around the whole miss / untile / upload / finalize path.
  double miss_usec = 0.0;
};

// All per-binding probe state owned by the command-processor call site. It is
// held in a `std::optional` that is ENGAGED ONLY when the temporary switch is
// on, so a disengaged build performs no per-binding probe work beyond one
// engaged-flag store — the payload's member initializers never run.
struct MetalSignRecensusCall {
  bool reachable = false;
  bool admitted = false;
  bool legacy_called = false;
  MetalSignRecensusProbeResult result;
};

// Accounting contract, deliberately strict: `uploads` / `bytes` /
// `upload_usec` count SUCCESSFUL completed miss + untile + upload + finalize
// work ONLY, because that is the only work the production early-out could
// remove. A miss that did not complete an upload is counted in `miss_null` and
// its bytes/time go to the separately named `failed_*` fields, which no
// materiality pool ever includes. A failed miss must never inflate the PASS
// metric.
struct MetalSignRecensusBucket {
  uint64_t calls = 0;           // binding-loop calls classified into this key
  uint64_t route_served = 0;    // C1T-R shared route served it; no legacy call
  uint64_t legacy_calls = 0;    // GetTextureSlot invocations
  uint64_t bridge_served = 0;   // legacy call returned a slot above the lookup
  uint64_t early_null = 0;      // legacy call returned 0 above the key lookup
  uint64_t hits = 0;            // keyed cache hit
  uint64_t misses = 0;          // keyed cache miss (successful + failed)
  uint64_t uploads = 0;         // miss reached a successful untile/upload
  uint64_t miss_null = 0;       // miss returned 0 (no upload completed)
  uint64_t alloc_failures = 0;  // binding_allocation_failed() after the call
  uint64_t bytes = 0;           // host bytes submitted by SUCCESSFUL misses
  double upload_usec = 0.0;     // CPU usec of SUCCESSFUL miss/upload work
  uint64_t failed_bytes = 0;    // host bytes submitted by misses that failed
  double failed_usec = 0.0;     // CPU usec of misses that did NOT complete
};

// One measured upload key. The base is part of the key, so every byte and
// microsecond total below NAMES the guest base that produced it.
struct MetalSignRecensusBaseKey {
  uint32_t base;
  uint32_t format;
  uint32_t dimension;
  uint32_t width;
  uint32_t height;
  uint32_t is_signed;

  bool operator<(const MetalSignRecensusBaseKey& o) const {
    if (base != o.base) {
      return base < o.base;
    }
    if (format != o.format) {
      return format < o.format;
    }
    if (dimension != o.dimension) {
      return dimension < o.dimension;
    }
    if (width != o.width) {
      return width < o.width;
    }
    if (height != o.height) {
      return height < o.height;
    }
    return is_signed < o.is_signed;
  }
};

struct MetalSignRecensusBaseRow {
  uint64_t uploads = 0;
  uint64_t bytes = 0;
  double upload_usec = 0.0;
  uint64_t reachable_uploads = 0;
  uint64_t unreachable_admitted_uploads = 0;
  uint64_t unreachable_excluded_uploads = 0;
  uint64_t recurring_uploads = 0;
  // Writers that actually published an overlapping range, by observation.
  uint64_t writer_resolve_only = 0;
  uint64_t writer_memexport_only = 0;
  uint64_t writer_both = 0;
  uint64_t writer_none = 0;
  // Byte-exact last-writer coverage summed over the uploads in this row, plus
  // the union history as a separate fact and the publication-recency gap.
  uint64_t resolve_since_bytes = 0;
  uint64_t memexport_since_bytes = 0;
  uint64_t stale_written_bytes = 0;
  uint64_t unwritten_bytes = 0;
  uint64_t resolve_ever_bytes = 0;
  uint64_t memexport_ever_bytes = 0;
  uint64_t seq_gap_sum = 0;
  uint64_t extent_bytes_total = 0;
  // Separate contemporaneous facts.
  uint64_t registered_at_fetch = 0;
  uint64_t not_registered_at_fetch = 0;
  uint64_t service_counts[4] = {0, 0, 0, 0};
};

class MetalSignRecensus {
 public:
  // Publication seams. The two writers that can make a texture destination's
  // guest bytes new. Recorded per event with the EXACT range written.
  enum class Writer : uint32_t {
    kResolveWriteback = 0,
    kMemexportDrain = 1,
  };

  // The bucket table is allocated here, not lazily on the first record: the
  // singleton is constructed by BeginCall() outside every timed region, so its
  // one-time allocation cannot land inside an overhead sample.
  MetalSignRecensus() : buckets_(kBucketCount) {}

  static bool Enabled() { return Period() != 0; }

  static MetalSignRecensus& Get() {
    static MetalSignRecensus instance;
    return instance;
  }

  // Snapshot period in swaps; 0 when the census is disengaged. Parsed once,
  // out of any per-draw path.
  static uint32_t Period() {
    static const uint32_t s_period = [] {
      const char* v = std::getenv("XE_METAL_SIGN_RECENSUS");
      if (!v || !*v) {
        return 0u;
      }
      long n = std::strtol(v, nullptr, 10);
      if (n <= 0) {
        return 0u;
      }
      return n > 1 ? uint32_t(n) : 300u;
    }();
    return s_period;
  }

  // Observer bookkeeping cost measured by the call sites, so probe overhead is
  // reported separately from the pool it measures.
  //
  // Measured on a SAMPLE of calls: a clock read costs about as much as the
  // bookkeeping itself, so timing every call would report mostly the cost of
  // the measurement. One call in `kOverheadSampleStride` is timed across all
  // of its regions (call-site classification, call-site record, and the
  // cache-internal hook).
  //
  // What the ENABLED probe actually costs (the acceptance figure) is the sum
  // of two components, and only the second is sampled:
  //   1. bookkeeping, incurred on EVERY call — estimated as the sampled
  //      regions' instrument-corrected work scaled by the stride;
  //   2. the clock instrumentation itself, incurred ONLY on sampled calls —
  //      `kClockReadsPerRegion` clock reads per timed region, charged at the
  //      measured per-read cost, with NO stride scaling (unsampled calls read
  //      no clock at all).
  // The stride-scaled uncorrected figure is a COUNTERFACTUAL "if every call
  // were timed, uncorrected" bound, not the cost this build actually pays; it
  // is printed under a name that says so.
  static constexpr uint64_t kOverheadSampleStride = 64;
  // `MetalSignRecensusOverheadScope` reads `steady_clock::now()` twice per
  // timed region (construction and destruction).
  static constexpr double kClockReadsPerRegion = 2.0;

  // Opens one binding-loop call and decides whether it is a timed sample.
  bool BeginCall() {
    NoteThread();
    measuring_ = (call_seq_++ % kOverheadSampleStride) == 0;
    if (measuring_) {
      ++sampled_calls_;
    }
    return measuring_;
  }
  void EndCall() { measuring_ = false; }
  bool measuring() const { return measuring_; }
  void AddProbeNs(uint64_t ns) {
    probe_ns_ += ns;
    ++probe_regions_;
  }

  // --- Corrected attribution: actual writers -----------------------------
  //
  // Called immediately after a publication seam has written guest RAM, with
  // the exact range that writer wrote. Unsampled: the durable production logs
  // sample, which is enough to see a pattern but not to attribute an extent.
  void RecordPublication(Writer writer, uint32_t base, uint32_t length) {
    NoteThread();
    if (!length) {
      return;
    }
    PubRange& r = pubs_[PubKey{uint32_t(writer), base, length}];
    ++r.events;
    r.bytes += uint64_t(length);
    // LAST-WRITER PROVENANCE, byte-exact.
    //
    // Membership in "some writer once published an overlapping range" is not
    // provenance: JC2's `0x11BE1000` is written by a resolve writeback and
    // then overwritten across the consumed bytes by three later memexport
    // drains in the same frame (task 2.4), so a union test would relabel
    // required memexport traffic as resolve/bridge work. What a consumer
    // actually reads is the LAST writer of each byte.
    //
    // The record is a set of DISJOINT byte intervals, each carrying its
    // writer and the publication sequence that wrote it. Intervals are exact:
    // no page rounding, so a publication can never claim a byte of an
    // adjacent texture it did not write. A new publication splits or removes
    // whatever it overwrites.
    const uint64_t seq = ++publication_sequence_;
    MarkWritten(uint64_t(base), uint64_t(base) + length,
                (writer == Writer::kResolveWriteback) ? 1u : 2u, seq);
  }

  uint64_t publication_sequence() const { return publication_sequence_; }

  // Result of asking who last wrote each part of a consumed extent, and
  // whether that write can causally explain THIS read.
  struct WriterOverlap {
    // Bytes whose last writer is each publisher AND whose publication is
    // newer than this cache key's previous upload — i.e. bytes this writer
    // can causally explain. These are the ONLY provenance signal the pools
    // use.
    uint64_t resolve_since_bytes = 0;
    uint64_t memexport_since_bytes = 0;
    // Bytes whose last writer published at or before the previous upload of
    // this same cache key. Such a write cannot explain a later re-upload
    // (the previous upload already observed it), so it is NOT attributed to
    // the writer; the re-upload was driven by something else — an ordinary
    // CPU-side invalidation, an eviction, or a key change.
    uint64_t stale_written_bytes = 0;
    // Bytes no writer has ever published.
    uint64_t unwritten_bytes = 0;
    // SEPARATE FACT, reported but never used to classify: the union history —
    // bytes each writer has ever published into this extent, whether or not
    // it is still the last writer of them.
    uint64_t resolve_ever_bytes = 0;
    uint64_t memexport_ever_bytes = 0;
    // Publications between the newest governing write and this read.
    uint64_t newest_pub_seq_gap = 0;
    bool any_pub_seen = false;
  };

  // Who last wrote each byte of [base, base + extent), and did they write it
  // after `key_watermark` — the publication sequence as of the previous
  // completed upload of the SAME cache key (0 for a first touch)?
  //
  // The watermark is what makes the attribution causal rather than historical.
  // Without it, any base a writer once published to would be attributed to
  // that writer forever, so an ordinary CPU-invalidation-driven re-upload of
  // long-static bytes would be counted as resolve traffic. There is no age
  // threshold anywhere: the test is strictly "did a publication land in these
  // bytes since this key was last uploaded".
  WriterOverlap WriterOverlapFor(uint32_t base, uint32_t extent,
                                 uint64_t key_watermark) const {
    WriterOverlap out;
    if (!extent) {
      return out;
    }
    const uint64_t begin = uint64_t(base);
    const uint64_t end = begin + uint64_t(extent);
    // Union history, kept as a separate fact only.
    for (const auto& kv : pubs_) {
      const PubKey& k = kv.first;
      const uint64_t pub_begin = uint64_t(k.base);
      const uint64_t pub_end = pub_begin + uint64_t(k.length);
      if (pub_begin >= end || pub_end <= begin) {
        continue;
      }
      const uint64_t overlap =
          std::min(pub_end, end) - std::max(pub_begin, begin);
      if (k.writer == uint32_t(Writer::kResolveWriteback)) {
        out.resolve_ever_bytes += overlap;
      } else {
        out.memexport_ever_bytes += overlap;
      }
    }
    out.resolve_ever_bytes = std::min<uint64_t>(out.resolve_ever_bytes, extent);
    out.memexport_ever_bytes =
        std::min<uint64_t>(out.memexport_ever_bytes, extent);
    // Exact last-writer decomposition.
    uint64_t covered = 0;
    uint64_t newest_seq = 0;
    for (auto it = FirstOverlap(begin); it != written_.end() && it->first < end;
         ++it) {
      if (it->second.end <= begin) {
        continue;
      }
      const uint64_t overlap =
          std::min(it->second.end, end) - std::max(it->first, begin);
      covered += overlap;
      const bool since = it->second.seq > key_watermark;
      if (!since) {
        out.stale_written_bytes += overlap;
      } else if (it->second.writer == 1u) {
        out.resolve_since_bytes += overlap;
      } else {
        out.memexport_since_bytes += overlap;
      }
      newest_seq = std::max(newest_seq, it->second.seq);
      out.any_pub_seen = true;
    }
    out.unwritten_bytes = extent - std::min<uint64_t>(covered, extent);
    if (out.any_pub_seen) {
      out.newest_pub_seq_gap = publication_sequence_ - newest_seq;
    }
    return out;
  }

  // Runs the classifier against hand-checked cases and logs one PASS/FAIL
  // line per case plus a summary. Temporary, like everything else here; it
  // exists so the exactness and causality claims are smoke-tested rather than
  // asserted. Engaged by XE_METAL_SIGN_RECENSUS_SELFTEST=1.
  static void SelfTest() {
    MetalSignRecensus t;
    uint32_t pass = 0, fail = 0;
    auto check = [&](const char* name, bool ok, const char* detail) {
      if (ok) {
        ++pass;
      } else {
        ++fail;
      }
      XELOGW("MetalSignRecensus v2 selftest {} case={} {}",
             ok ? "PASS" : "FAIL", name, detail);
    };
    // 1. A publication must not claim a byte outside its exact range: a small
    //    texture adjacent to (but not inside) a resolve destination stays
    //    unattributed. Deliberately unaligned so page rounding would fail it.
    t.RecordPublication(Writer::kResolveWriteback, 0x10000000, 0x1000);
    {
      auto r = t.WriterOverlapFor(0x10001000, 0x100, 0);
      check("adjacent_small_texture_not_claimed",
            r.resolve_since_bytes == 0 && r.unwritten_bytes == 0x100,
            fmt::format("res={} unwritten={}", r.resolve_since_bytes,
                        r.unwritten_bytes)
                .c_str());
    }
    // 2. Partial overlap is counted in bytes, exactly.
    {
      auto r = t.WriterOverlapFor(0x10000F00, 0x200, 0);
      check("partial_overlap_is_byte_exact",
            r.resolve_since_bytes == 0x100 && r.unwritten_bytes == 0x100,
            fmt::format("res={} unwritten={}", r.resolve_since_bytes,
                        r.unwritten_bytes)
                .c_str());
    }
    // 3. A later writer overwriting a subrange takes ownership of exactly
    //    that subrange (the JC2 resolve-then-memexport case).
    t.RecordPublication(Writer::kResolveWriteback, 0x11BE1000, 983040);
    t.RecordPublication(Writer::kMemexportDrain, 0x11BE1000, 2097152);
    t.RecordPublication(Writer::kMemexportDrain, 0x11DE1000, 2097152);
    t.RecordPublication(Writer::kMemexportDrain, 0x11FE1000, 2097152);
    {
      auto r = t.WriterOverlapFor(0x11BE1000, 6291456, 0);
      check("later_memexport_owns_overwritten_bytes",
            r.memexport_since_bytes == 6291456 && r.resolve_since_bytes == 0 &&
                r.resolve_ever_bytes == 983040,
            fmt::format("mem={} res={} res_ever={}", r.memexport_since_bytes,
                        r.resolve_since_bytes, r.resolve_ever_bytes)
                .c_str());
    }
    // 4. An earlier writer keeps the bytes the later one did not touch.
    t.RecordPublication(Writer::kResolveWriteback, 0x12000000, 0x2000);
    t.RecordPublication(Writer::kMemexportDrain, 0x12001000, 0x1000);
    {
      auto r = t.WriterOverlapFor(0x12000000, 0x2000, 0);
      check(
          "split_extent_keeps_both_writers",
          r.resolve_since_bytes == 0x1000 && r.memexport_since_bytes == 0x1000,
          fmt::format("res={} mem={}", r.resolve_since_bytes,
                      r.memexport_since_bytes)
              .c_str());
    }
    // 5. Causality: with a watermark at the current sequence, an OLD write can
    //    no longer explain a re-upload; the bytes are stale, not attributed.
    {
      const uint64_t wm = t.publication_sequence();
      auto r = t.WriterOverlapFor(0x11BE1000, 6291456, wm);
      check("old_write_cannot_explain_later_reupload",
            r.memexport_since_bytes == 0 && r.resolve_since_bytes == 0 &&
                r.stale_written_bytes == 6291456,
            fmt::format("mem={} res={} stale={}", r.memexport_since_bytes,
                        r.resolve_since_bytes, r.stale_written_bytes)
                .c_str());
      // ... and a NEW publication after that watermark does explain it.
      t.RecordPublication(Writer::kMemexportDrain, 0x11BE1000, 2097152);
      auto r2 = t.WriterOverlapFor(0x11BE1000, 6291456, wm);
      check("publication_after_watermark_is_attributed",
            r2.memexport_since_bytes == 2097152 &&
                r2.stale_written_bytes == 6291456 - 2097152,
            fmt::format("mem={} stale={}", r2.memexport_since_bytes,
                        r2.stale_written_bytes)
                .c_str());
    }
    // 6. Never-written bytes are never attributed.
    {
      auto r = t.WriterOverlapFor(0x1D000000, 0x4000, 0);
      check("never_written_is_unattributed",
            r.unwritten_bytes == 0x4000 && r.resolve_since_bytes == 0 &&
                r.memexport_since_bytes == 0,
            fmt::format("unwritten={}", r.unwritten_bytes).c_str());
    }
    XELOGW("MetalSignRecensus v2 selftest summary pass={} fail={}", pass, fail);
  }

  // One classified binding-loop call. `reachable` is
  // IsMetalTextureSignInterpretationReachable(SwizzleSigns(fetch),
  // binding.is_signed); `admitted` is the task-3.3 admitted dimension set
  // (exactly binding.dimension == FetchOpDimension::k2D).
  void RecordBinding(bool is_signed, bool reachable, bool admitted,
                     xenos::FetchOpDimension dimension, uint32_t guest_format,
                     bool route_served, bool legacy_called, bool alloc_failed,
                     uint32_t returned_slot,
                     const MetalSignRecensusProbeResult& r) {
    // Writer attribution by the LAST WRITER of the consumed bytes, restricted
    // to writes that can causally explain THIS read (newer than the previous
    // upload of the same cache key). 4 = unknown (the call never reached the
    // keyed lookup or never completed an upload, so there is no measured
    // extent); 0 = no writer published into these bytes since this key was
    // last uploaded, i.e. ordinary streaming or an ordinary CPU-side
    // invalidation; 1 = a resolve writeback did; 2 = a memexport drain did;
    // 3 = both did, over different subranges — kept as its own class rather
    // than folded into either.
    uint32_t wa = 4u;
    if (r.keyed_lookup && r.upload_completed) {
      wa = ((r.resolve_since_bytes > 0) ? 1u : 0u) |
           ((r.memexport_since_bytes > 0) ? 2u : 0u);
    }
    uint32_t rec = r.keyed_lookup ? (r.recurring ? 1u : 0u) : 2u;
    // Direct-indexed: the packed key is 15 bits, so the hot path is an array
    // increment rather than a hash/tree lookup. Bucket order is the key order.
    MetalSignRecensusBucket& b =
        buckets_[PackKey(is_signed, reachable, admitted, uint32_t(dimension),
                         guest_format, wa, rec)];
    ++b.calls;
    if (route_served) {
      ++b.route_served;
    }
    if (legacy_called) {
      ++b.legacy_calls;
      if (!r.keyed_lookup) {
        if (returned_slot) {
          ++b.bridge_served;
        } else {
          ++b.early_null;
        }
      } else if (r.cache_hit) {
        ++b.hits;
      } else {
        ++b.misses;
        // Successful-only accounting: bytes and time are attributed to
        // `bytes`/`upload_usec` — the fields every materiality pool reads —
        // ONLY when the miss actually completed its untile/upload/finalize.
        // A miss that returned 0 submitted no usable texture, so its cost is
        // recorded under the honestly named failed fields and is excluded
        // from every pool.
        if (r.upload_completed) {
          ++b.uploads;
          b.bytes += r.host_bytes;
          b.upload_usec += r.miss_usec;
          RecordBaseRow(reachable, admitted, wa, is_signed, dimension, r);
        } else {
          ++b.miss_null;
          b.failed_bytes += r.host_bytes;
          b.failed_usec += r.miss_usec;
        }
      }
      if (alloc_failed) {
        ++b.alloc_failures;
      }
    }
  }

  void OnSwap(uint32_t swap_index) {
    swaps_ = swap_index + 1;
    uint32_t period = Period();
    if (period && (swap_index % period) == 0) {
      Dump("swap");
    }
  }

  void DumpFinal() {
    if (dumped_final_) {
      return;
    }
    dumped_final_ = true;
    Dump("shutdown");
  }

 private:
  // 1 + 1 + 1 + 2 + 6 + 3 + 2 packed key bits.
  static constexpr size_t kBucketCount = size_t(1) << 16;
  static constexpr size_t kBaseReportRows = 40;

  struct PubKey {
    uint32_t writer;
    uint32_t base;
    uint32_t length;

    bool operator<(const PubKey& o) const {
      if (base != o.base) {
        return base < o.base;
      }
      if (length != o.length) {
        return length < o.length;
      }
      return writer < o.writer;
    }
  };

  struct PubRange {
    uint64_t events = 0;
    uint64_t bytes = 0;
  };

  struct WrittenSpan {
    uint64_t end;
    uint8_t writer;  // 1 resolve writeback, 2 memexport drain
    uint64_t seq;
  };

  // First interval that could overlap `addr`.
  std::map<uint64_t, WrittenSpan>::const_iterator FirstOverlap(
      uint64_t addr) const {
    auto it = written_.upper_bound(addr);
    if (it != written_.begin()) {
      auto prev = std::prev(it);
      if (prev->second.end > addr) {
        return prev;
      }
    }
    return it;
  }

  // Records [begin, end) as last-written by `writer` at `seq`, splitting or
  // removing exactly the parts of existing intervals it overwrites.
  void MarkWritten(uint64_t begin, uint64_t end, uint8_t writer, uint64_t seq) {
    if (end <= begin) {
      return;
    }
    std::vector<std::pair<uint64_t, WrittenSpan>> reinsert;
    auto it = written_.upper_bound(begin);
    if (it != written_.begin()) {
      auto prev = std::prev(it);
      if (prev->second.end > begin) {
        it = prev;
      }
    }
    while (it != written_.end() && it->first < end) {
      if (it->second.end <= begin) {
        ++it;
        continue;
      }
      if (it->first < begin) {
        reinsert.emplace_back(
            it->first, WrittenSpan{begin, it->second.writer, it->second.seq});
      }
      if (it->second.end > end) {
        reinsert.emplace_back(
            end,
            WrittenSpan{it->second.end, it->second.writer, it->second.seq});
      }
      it = written_.erase(it);
    }
    for (auto& r : reinsert) {
      written_.emplace(r.first, r.second);
    }
    written_.emplace(begin, WrittenSpan{end, writer, seq});
  }

  static uint64_t PackKey(bool is_signed, bool reachable, bool admitted,
                          uint32_t dimension, uint32_t guest_format,
                          uint32_t wa, uint32_t rec) {
    return (uint64_t(is_signed ? 1 : 0) << 0) |
           (uint64_t(reachable ? 1 : 0) << 1) |
           (uint64_t(admitted ? 1 : 0) << 2) | (uint64_t(dimension & 3) << 3) |
           (uint64_t(guest_format & 0x3F) << 5) | (uint64_t(wa & 7) << 11) |
           (uint64_t(rec & 3) << 14);
  }

  // The single-thread assumption is checked, not assumed.
  void NoteThread() {
    std::thread::id self = std::this_thread::get_id();
    if (!thread_latched_) {
      thread_latched_ = true;
      recording_thread_ = self;
      return;
    }
    if (self != recording_thread_) {
      ++foreign_thread_records_;
    }
  }

  void RecordBaseRow(bool reachable, bool admitted, uint32_t wa, bool is_signed,
                     xenos::FetchOpDimension dimension,
                     const MetalSignRecensusProbeResult& r) {
    MetalSignRecensusBaseRow& row = base_rows_[MetalSignRecensusBaseKey{
        r.base, r.guest_format, uint32_t(dimension), r.width, r.height,
        uint32_t(is_signed ? 1 : 0)}];
    ++row.uploads;
    row.bytes += r.host_bytes;
    row.upload_usec += r.miss_usec;
    if (reachable) {
      ++row.reachable_uploads;
    } else if (admitted) {
      ++row.unreachable_admitted_uploads;
    } else {
      ++row.unreachable_excluded_uploads;
    }
    if (r.recurring) {
      ++row.recurring_uploads;
    }
    if (wa == 1u) {
      ++row.writer_resolve_only;
    } else if (wa == 2u) {
      ++row.writer_memexport_only;
    } else if (wa == 3u) {
      ++row.writer_both;
    } else {
      ++row.writer_none;
    }
    row.resolve_since_bytes += r.resolve_since_bytes;
    row.memexport_since_bytes += r.memexport_since_bytes;
    row.stale_written_bytes += r.stale_written_bytes;
    row.unwritten_bytes += r.unwritten_bytes;
    row.resolve_ever_bytes += r.resolve_ever_bytes;
    row.memexport_ever_bytes += r.memexport_ever_bytes;
    row.seq_gap_sum += r.newest_pub_seq_gap;
    row.extent_bytes_total += r.extent_bytes;
    if (r.registered_at_fetch) {
      ++row.registered_at_fetch;
    } else {
      ++row.not_registered_at_fetch;
    }
    ++row.service_counts[r.service & 3u];
  }

  // What an EMPTY timed region reports about itself — the instrument's own
  // floor, measured exactly the way a real region measures its work. A timed
  // span covers one clock read's latency, not two, so this (not the cost of a
  // clock pair) is what must be SUBTRACTED from a sampled region to recover
  // its bookkeeping work. It is also the per-clock-read cost used to ADD BACK
  // the clock instrumentation a sampled region really pays
  // (`kClockReadsPerRegion` of them). Measured once, at first dump.
  static double EmptyRegionNs() {
    static const double s_cost = [] {
      constexpr int kIters = 20000;
      int64_t total_ns = 0;
      for (int i = 0; i < kIters; ++i) {
        auto a = std::chrono::steady_clock::now();
        auto b = std::chrono::steady_clock::now();
        total_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
      }
      return double(total_ns) / double(kIters);
    }();
    return s_cost;
  }

  // Cumulative, machine-parseable. Every line is one record; a settled window
  // is the difference of two snapshots, so nothing is reset here.
  void Dump(const char* reason) {
    uint64_t total_calls = 0, total_uploads = 0, total_bytes = 0;
    double total_usec = 0.0;
    // Failed-miss telemetry, tracked in full but kept out of every pool below.
    uint64_t total_miss_null = 0, total_failed_bytes = 0;
    double total_failed_usec = 0.0;
    // The ONLY pool this change could remove: unreachable AND inside the
    // admitted dimension set, counted as SUCCESSFUL completed miss / untile /
    // upload / finalize work. Split by CORRECTED writer attribution.
    uint64_t rm_uploads = 0, rm_bytes = 0;
    double rm_usec = 0.0;
    // Index by the 5-valued writer class so no class is silently merged.
    uint64_t rm_wa_uploads[5] = {0, 0, 0, 0, 0};
    uint64_t rm_wa_bytes[5] = {0, 0, 0, 0, 0};
    double rm_wa_usec[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    // Reported for attribution only; never folded into the removable pool.
    uint64_t ex_uploads = 0, rc_uploads = 0;
    double ex_usec = 0.0, rc_usec = 0.0;
    uint64_t live_buckets = 0;
    for (size_t key = 0; key < buckets_.size(); ++key) {
      const MetalSignRecensusBucket& b = buckets_[key];
      if (!b.calls) {
        continue;
      }
      ++live_buckets;
      total_calls += b.calls;
      total_uploads += b.uploads;
      total_bytes += b.bytes;
      total_usec += b.upload_usec;
      total_miss_null += b.miss_null;
      total_failed_bytes += b.failed_bytes;
      total_failed_usec += b.failed_usec;
      bool reachable = (key >> 1) & 1;
      bool admitted = (key >> 2) & 1;
      uint32_t wa = uint32_t((key >> 11) & 7);
      if (reachable) {
        rc_uploads += b.uploads;
        rc_usec += b.upload_usec;
      } else if (admitted) {
        rm_uploads += b.uploads;
        rm_bytes += b.bytes;
        rm_usec += b.upload_usec;
        uint32_t slot = wa < 5u ? wa : 4u;
        rm_wa_uploads[slot] += b.uploads;
        rm_wa_bytes[slot] += b.bytes;
        rm_wa_usec[slot] += b.upload_usec;
      } else {
        ex_uploads += b.uploads;
        ex_usec += b.upload_usec;
      }
    }
    double frames = double(swaps_ ? swaps_ : 1);
    double empty_region_ns = EmptyRegionNs();
    double probe_usec_measured = double(probe_ns_) / 1000.0;
    // Bookkeeping work in the sampled regions, with the instrument's own floor
    // removed, scaled by the stride = the bookkeeping every call pays.
    double corrected_ns =
        double(probe_ns_) - double(probe_regions_) * empty_region_ns;
    if (corrected_ns < 0.0) {
      corrected_ns = 0.0;
    }
    double bookkeeping_ns = corrected_ns * double(kOverheadSampleStride);
    // Clock instrumentation actually executed: sampled regions only, NOT
    // stride-scaled — an unsampled call reads no clock.
    double clock_ns =
        double(probe_regions_) * kClockReadsPerRegion * empty_region_ns;
    // THE ACCEPTANCE FIGURE: what this enabled build actually costs.
    double probe_actual_usec = (bookkeeping_ns + clock_ns) / 1000.0;
    // COUNTERFACTUAL ONLY: what an uncorrected probe that timed EVERY call
    // would report. It is not this build's cost and must never be quoted as
    // one; it is printed so no reader mistakes the actual figure for the only
    // number available.
    double counterfactual_usec =
        probe_usec_measured * double(kOverheadSampleStride);
    XELOGW(
        "MetalSignRecensus v2 snapshot reason={} swaps={} buckets={} "
        "calls={} uploads={} bytes={} upload_usec={:.1f} "
        "miss_null={} failed_bytes={} failed_usec={:.1f} "
        "base_keys={} pub_keys={} foreign_thread_records={} "
        "probe_usec_measured_sampled={:.1f} probe_regions={} "
        "probe_sampled_calls={} empty_region_ns={:.1f} "
        "probe_sample_stride={} probe_bookkeeping_usec_all_calls={:.1f} "
        "probe_clock_usec_sampled_calls={:.1f} probe_actual_usec={:.1f} "
        "probe_actual_usec_per_frame={:.3f} "
        "counterfactual_alltimed_uncorrected_usec_per_frame={:.3f}",
        reason, swaps_, live_buckets, total_calls, total_uploads, total_bytes,
        total_usec, total_miss_null, total_failed_bytes, total_failed_usec,
        base_rows_.size(), pubs_.size(), foreign_thread_records_,
        probe_usec_measured, probe_regions_, sampled_calls_, empty_region_ns,
        kOverheadSampleStride, bookkeeping_ns / 1000.0, clock_ns / 1000.0,
        probe_actual_usec, probe_actual_usec / frames,
        counterfactual_usec / frames);
    if (foreign_thread_records_) {
      XELOGW(
          "MetalSignRecensus v2 WARNING records={} arrived from a thread other "
          "than the latched recording thread -- the lock-free assumption is "
          "violated and these counters may be torn",
          foreign_thread_records_);
    }
    for (size_t key = 0; key < buckets_.size(); ++key) {
      const MetalSignRecensusBucket& b = buckets_[key];
      if (!b.calls) {
        continue;
      }
      XELOGW(
          "MetalSignRecensus v2 rec swaps={} sig={} reach={} adm={} dim={} "
          "fmt={} wa={} rec={} calls={} route={} legacy={} bridge={} "
          "earlynull={} hit={} miss={} upl={} missnull={} allocfail={} "
          "bytes={} upl_usec={:.1f} failbytes={} fail_usec={:.1f}",
          swaps_, (key >> 0) & 1, (key >> 1) & 1, (key >> 2) & 1,
          (key >> 3) & 3, (key >> 5) & 0x3F, (key >> 11) & 3, (key >> 13) & 3,
          b.calls, b.route_served, b.legacy_calls, b.bridge_served,
          b.early_null, b.hits, b.misses, b.uploads, b.miss_null,
          b.alloc_failures, b.bytes, b.upload_usec, b.failed_bytes,
          b.failed_usec);
    }
    // Publication evidence: the exact ranges each writer actually wrote. This
    // is the provenance source for wa= above; nothing here is inferred from
    // `resolved_targets_` membership.
    for (const auto& kv : pubs_) {
      XELOGW(
          "MetalSignRecensus v2 pub swaps={} kind={} base={:08X} len={} "
          "events={} bytes={} events_per_frame={:.4f} bytes_per_frame={:.1f}",
          swaps_,
          kv.first.writer == uint32_t(Writer::kResolveWriteback)
              ? "resolve_writeback"
              : "memexport_drain",
          kv.first.base, kv.first.length, kv.second.events, kv.second.bytes,
          double(kv.second.events) / frames, double(kv.second.bytes) / frames);
    }
    // Per-base rows, ranked by measured CPU time so the pool is NAMED by base
    // rather than located by a size match. Totals above are unbounded; this
    // list is bounded and says so when it truncates.
    std::vector<
        std::pair<MetalSignRecensusBaseKey, const MetalSignRecensusBaseRow*>>
        ranked;
    ranked.reserve(base_rows_.size());
    for (const auto& kv : base_rows_) {
      ranked.emplace_back(kv.first, &kv.second);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      return a.second->upload_usec > b.second->upload_usec;
    });
    size_t emitted = 0;
    for (const auto& entry : ranked) {
      if (emitted++ >= kBaseReportRows) {
        XELOGW("MetalSignRecensus v2 base-truncated swaps={} remaining_keys={}",
               swaps_, ranked.size() - kBaseReportRows);
        break;
      }
      const MetalSignRecensusBaseKey& k = entry.first;
      const MetalSignRecensusBaseRow& row = *entry.second;
      XELOGW(
          "MetalSignRecensus v2 base swaps={} base={:08X} fmt={} dim={} "
          "{}x{} sig={} extent_note=host_bytes "
          "uploads={} bytes={} usec={:.1f} usec_per_frame={:.4f} "
          "reach_upl={} unreach_adm_upl={} unreach_excl_upl={} recurring={} "
          "writer_resolve_only={} writer_memexport_only={} writer_both={} "
          "writer_none={} resolve_since_bytes={} memexport_since_bytes={} "
          "stale_written_bytes={} unwritten_bytes={} resolve_ever_bytes={} "
          "memexport_ever_bytes={} "
          "mean_pub_seq_gap={:.1f} extent_bytes={} "
          "registered={} not_registered={} svc_nrb={} svc_bridge={} "
          "svc_exact={} svc_ram={}",
          swaps_, k.base, k.format, k.dimension, k.width, k.height, k.is_signed,
          row.uploads, row.bytes, row.upload_usec, row.upload_usec / frames,
          row.reachable_uploads, row.unreachable_admitted_uploads,
          row.unreachable_excluded_uploads, row.recurring_uploads,
          row.writer_resolve_only, row.writer_memexport_only, row.writer_both,
          row.writer_none, row.resolve_since_bytes, row.memexport_since_bytes,
          row.stale_written_bytes, row.unwritten_bytes, row.resolve_ever_bytes,
          row.memexport_ever_bytes,
          row.uploads ? double(row.seq_gap_sum) / double(row.uploads) : 0.0,
          row.extent_bytes_total, row.registered_at_fetch,
          row.not_registered_at_fetch, row.service_counts[0],
          row.service_counts[1], row.service_counts[2], row.service_counts[3]);
    }
    // Every pool line below is SUCCESSFUL completed miss/untile/upload/
    // finalize work only. Failed misses are reported once, separately, and
    // are in none of these pools.
    XELOGW(
        "MetalSignRecensus v2 pool swaps={} "
        "name=removable_unreachable_admitted "
        "basis=successful_uploads_only uploads={} bytes={} usec={:.1f} "
        "usec_per_frame={:.4f}",
        swaps_, rm_uploads, rm_bytes, rm_usec, rm_usec / frames);
    static const char* kWriterClassNames[5] = {
        "removable_no_writer_since_last_upload_ordinary_streaming",
        "removable_caused_by_resolve_writeback",
        "removable_caused_by_memexport_drain",
        "removable_caused_by_both_resolve_and_memexport",
        "removable_writer_unknown_no_measured_extent",
    };
    for (uint32_t i = 0; i < 5; ++i) {
      XELOGW(
          "MetalSignRecensus v2 pool swaps={} name={} "
          "basis=successful_uploads_only uploads={} bytes={} usec={:.1f} "
          "usec_per_frame={:.4f}",
          swaps_, kWriterClassNames[i], rm_wa_uploads[i], rm_wa_bytes[i],
          rm_wa_usec[i], rm_wa_usec[i] / frames);
    }
    XELOGW(
        "MetalSignRecensus v2 pool swaps={} "
        "name=unreachable_excluded_dimension "
        "basis=successful_uploads_only uploads={} usec={:.1f} "
        "usec_per_frame={:.4f}",
        swaps_, ex_uploads, ex_usec, ex_usec / frames);
    XELOGW(
        "MetalSignRecensus v2 pool swaps={} name=reachable_not_removable_here "
        "basis=successful_uploads_only uploads={} usec={:.1f} "
        "usec_per_frame={:.4f}",
        swaps_, rc_uploads, rc_usec, rc_usec / frames);
    XELOGW(
        "MetalSignRecensus v2 pool swaps={} "
        "name=failed_miss_excluded_from_pools "
        "basis=misses_that_completed_no_upload missnull={} bytes={} "
        "usec={:.1f} usec_per_frame={:.4f}",
        swaps_, total_miss_null, total_failed_bytes, total_failed_usec,
        total_failed_usec / frames);
    XELOGW("MetalSignRecensus v2 end reason={} swaps={}", reason, swaps_);
  }

  // Direct-indexed by the packed key, so snapshots print in a stable key
  // order and the hot path is an array increment. Allocated on first record.
  std::vector<MetalSignRecensusBucket> buckets_;
  // Corrected attribution state. `pubs_` is union history (a separate fact);
  // the page maps are the last-writer provenance the pools actually use.
  std::map<PubKey, PubRange> pubs_;
  // Disjoint byte intervals: start -> {end, writer, publication sequence}.
  // Exact; no page rounding. Bounded by the number of distinct destination
  // ranges the title writes, which is tens.
  std::map<uint64_t, WrittenSpan> written_;
  uint64_t publication_sequence_ = 0;
  std::map<MetalSignRecensusBaseKey, MetalSignRecensusBaseRow> base_rows_;
  uint64_t probe_ns_ = 0;
  uint64_t probe_regions_ = 0;
  uint64_t call_seq_ = 0;
  uint64_t sampled_calls_ = 0;
  uint64_t foreign_thread_records_ = 0;
  std::thread::id recording_thread_;
  bool thread_latched_ = false;
  uint32_t swaps_ = 0;
  bool measuring_ = false;
  bool dumped_final_ = false;
};

// Scoped observer-overhead accumulator for one bookkeeping region. Reads the
// clock only on sampled calls (MetalSignRecensus::BeginCall), so the instrument
// does not dominate what it measures.
class MetalSignRecensusOverheadScope {
 public:
  explicit MetalSignRecensusOverheadScope(MetalSignRecensus& census)
      : census_(census), active_(census.measuring()) {
    if (active_) {
      t0_ = std::chrono::steady_clock::now();
    }
  }
  MetalSignRecensusOverheadScope()
      : MetalSignRecensusOverheadScope(MetalSignRecensus::Get()) {}
  ~MetalSignRecensusOverheadScope() {
    if (!active_) {
      return;
    }
    census_.AddProbeNs(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0_)
                     .count()));
  }

 private:
  MetalSignRecensus& census_;
  bool active_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SIGN_RECENSUS_PROBE_H_
