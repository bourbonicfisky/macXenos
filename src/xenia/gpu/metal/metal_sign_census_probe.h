/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SIGN_CENSUS_PROBE_H_
#define XENIA_GPU_METAL_METAL_SIGN_CENSUS_PROBE_H_

// TEMPORARY MEASUREMENT PROBE — DELETE BEFORE DELIVERY.
//
// OpenSpec change `align-metal-texture-binding-signedness`, task 4.1: the
// behaviour-neutral live census that decides whether the unreachable
// sign-interpretation pool is material enough to justify the production gate
// (task 5.2). It exists ONLY on the census probe commit; task 4.3 reverts this
// file and every reference to it byte-identically.
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
//     initialised on the disengaged path — `MetalSignCensusCall` lives in a
//     `std::optional` the call site engages only when the switch is on;
//   * output is aggregate and machine-parseable; no per-call log, no cvar, no
//     HUD, no shipped counter.
//
// Engagement: XE_METAL_SIGN_CENSUS=1 (any value >= 1 engages; a value > 1 also
// sets the cumulative-snapshot period in swaps, default 300).
//
// Threading: every entry point is called from the GPU thread only (the
// per-binding marshalling loop, `MetalTextureCache::GetTextureSlot`, IssueSwap,
// and ShutdownContext all run there). No locking, deliberately — a lock would
// be probe-visible cost.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "xenia/base/logging.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Facts about ONE `GetTextureSlot` call that only the legacy texture cache can
// observe accurately. The cache fills this when the census is engaged; it never
// reads it back and never branches on it, so it cannot change cache behaviour.
struct MetalSignCensusProbeResult {
  // The call reached the keyed cache lookup (i.e. it was not rejected by the
  // fetch-type / dimension / format / bounds early-outs and was not served by
  // the resolve-snapshot bridge above the lookup).
  bool keyed_lookup = false;
  bool cache_hit = false;
  // The miss reached a successful CPU untile + upload + finalize.
  bool upload_completed = false;
  // The fetch base is a registered resolve-writeback destination.
  bool resolve_backed = false;
  // This exact cache key had already been uploaded at least once in this run
  // (i.e. the upload is an invalidation-driven re-upload, not first sight).
  bool recurring = false;
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
struct MetalSignCensusCall {
  bool reachable = false;
  bool admitted = false;
  bool legacy_called = false;
  MetalSignCensusProbeResult result;
};

// Accounting contract, deliberately strict: `uploads` / `bytes` /
// `upload_usec` count SUCCESSFUL completed miss + untile + upload + finalize
// work ONLY, because that is the only work the production early-out could
// remove. A miss that did not complete an upload is counted in `miss_null` and
// its bytes/time go to the separately named `failed_*` fields, which no
// materiality pool ever includes. A failed miss must never inflate the PASS
// metric.
struct MetalSignCensusBucket {
  uint64_t calls = 0;           // binding-loop calls classified into this key
  uint64_t route_served = 0;    // C1T-R shared route served it; no legacy call
  uint64_t legacy_calls = 0;    // GetTextureSlot invocations
  uint64_t bridge_served = 0;   // legacy call returned a slot above the key lookup
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

class MetalSignCensus {
 public:
  // The bucket table is allocated here, not lazily on the first record: the
  // singleton is constructed by BeginCall() outside every timed region, so its
  // one-time allocation cannot land inside an overhead sample.
  MetalSignCensus() : buckets_(kBucketCount) {}

  static bool Enabled() { return Period() != 0; }

  static MetalSignCensus& Get() {
    static MetalSignCensus instance;
    return instance;
  }

  // Snapshot period in swaps; 0 when the census is disengaged. Parsed once,
  // out of any per-draw path.
  static uint32_t Period() {
    static const uint32_t s_period = [] {
      const char* v = std::getenv("XE_METAL_SIGN_CENSUS");
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
  // `MetalSignCensusOverheadScope` reads `steady_clock::now()` twice per timed
  // region (construction and destruction).
  static constexpr double kClockReadsPerRegion = 2.0;

  // Opens one binding-loop call and decides whether it is a timed sample.
  bool BeginCall() {
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

  // One classified binding-loop call. `reachable` is
  // IsMetalTextureSignInterpretationReachable(SwizzleSigns(fetch),
  // binding.is_signed); `admitted` is the task-3.3 admitted dimension set
  // (exactly binding.dimension == FetchOpDimension::k2D).
  void RecordBinding(bool is_signed, bool reachable, bool admitted,
                     xenos::FetchOpDimension dimension, uint32_t guest_format,
                     bool route_served, bool legacy_called, bool alloc_failed,
                     uint32_t returned_slot,
                     const MetalSignCensusProbeResult& r) {
    // rb / rec are unknown (2) when the call never reached the keyed lookup.
    uint32_t rb = r.keyed_lookup ? (r.resolve_backed ? 1u : 0u) : 2u;
    uint32_t rec = r.keyed_lookup ? (r.recurring ? 1u : 0u) : 2u;
    // Direct-indexed: the packed key is 15 bits, so the hot path is an array
    // increment rather than a hash/tree lookup. Bucket order is the key order.
    MetalSignCensusBucket& b = buckets_[PackKey(
        is_signed, reachable, admitted, uint32_t(dimension), guest_format, rb,
        rec)];
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
  // 1 + 1 + 1 + 2 + 6 + 2 + 2 packed key bits.
  static constexpr size_t kBucketCount = size_t(1) << 15;

  static uint64_t PackKey(bool is_signed, bool reachable, bool admitted,
                          uint32_t dimension, uint32_t guest_format,
                          uint32_t rb, uint32_t rec) {
    return (uint64_t(is_signed ? 1 : 0) << 0) |
           (uint64_t(reachable ? 1 : 0) << 1) |
           (uint64_t(admitted ? 1 : 0) << 2) | (uint64_t(dimension & 3) << 3) |
           (uint64_t(guest_format & 0x3F) << 5) | (uint64_t(rb & 3) << 11) |
           (uint64_t(rec & 3) << 13);
  }

  // Cumulative, machine-parseable. Every line is one record; a settled window
  // is the difference of two snapshots, so nothing is reset here.
  void Dump(const char* reason) {
    uint64_t total_calls = 0, total_uploads = 0, total_bytes = 0;
    double total_usec = 0.0;
    // Failed-miss telemetry, tracked in full but kept out of every pool below.
    uint64_t total_miss_null = 0, total_failed_bytes = 0;
    double total_failed_usec = 0.0;
    // The ONLY pool this change can remove: unreachable AND inside the
    // admitted dimension set, counted as SUCCESSFUL completed miss / untile /
    // upload / finalize work.
    uint64_t rm_uploads = 0, rm_bytes = 0;
    double rm_usec = 0.0;
    // Reported for attribution only; never folded into the removable pool.
    uint64_t ex_uploads = 0, rc_uploads = 0;
    double ex_usec = 0.0, rc_usec = 0.0;
    uint64_t live_buckets = 0;
    for (size_t key = 0; key < buckets_.size(); ++key) {
      const MetalSignCensusBucket& b = buckets_[key];
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
      if (reachable) {
        rc_uploads += b.uploads;
        rc_usec += b.upload_usec;
      } else if (admitted) {
        rm_uploads += b.uploads;
        rm_bytes += b.bytes;
        rm_usec += b.upload_usec;
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
        "MetalSignCensus v1 snapshot reason={} swaps={} buckets={} "
        "calls={} uploads={} bytes={} upload_usec={:.1f} "
        "miss_null={} failed_bytes={} failed_usec={:.1f} "
        "probe_usec_measured_sampled={:.1f} probe_regions={} "
        "probe_sampled_calls={} empty_region_ns={:.1f} "
        "probe_sample_stride={} probe_bookkeeping_usec_all_calls={:.1f} "
        "probe_clock_usec_sampled_calls={:.1f} probe_actual_usec={:.1f} "
        "probe_actual_usec_per_frame={:.3f} "
        "counterfactual_alltimed_uncorrected_usec_per_frame={:.3f}",
        reason, swaps_, live_buckets, total_calls, total_uploads, total_bytes,
        total_usec, total_miss_null, total_failed_bytes, total_failed_usec,
        probe_usec_measured, probe_regions_, sampled_calls_, empty_region_ns,
        kOverheadSampleStride, bookkeeping_ns / 1000.0, clock_ns / 1000.0,
        probe_actual_usec, probe_actual_usec / frames,
        counterfactual_usec / frames);
    for (size_t key = 0; key < buckets_.size(); ++key) {
      const MetalSignCensusBucket& b = buckets_[key];
      if (!b.calls) {
        continue;
      }
      XELOGW(
          "MetalSignCensus v1 rec swaps={} sig={} reach={} adm={} dim={} "
          "fmt={} rb={} rec={} calls={} route={} legacy={} bridge={} "
          "earlynull={} hit={} miss={} upl={} missnull={} allocfail={} "
          "bytes={} upl_usec={:.1f} failbytes={} fail_usec={:.1f}",
          swaps_, (key >> 0) & 1, (key >> 1) & 1, (key >> 2) & 1,
          (key >> 3) & 3, (key >> 5) & 0x3F, (key >> 11) & 3, (key >> 13) & 3,
          b.calls, b.route_served, b.legacy_calls, b.bridge_served,
          b.early_null, b.hits, b.misses, b.uploads, b.miss_null,
          b.alloc_failures, b.bytes, b.upload_usec, b.failed_bytes,
          b.failed_usec);
    }
    // Every pool line below is SUCCESSFUL completed miss/untile/upload/
    // finalize work only. Failed misses are reported once, separately, and
    // are in none of these pools.
    XELOGW(
        "MetalSignCensus v1 pool swaps={} name=removable_unreachable_admitted "
        "basis=successful_uploads_only uploads={} bytes={} usec={:.1f} "
        "usec_per_frame={:.4f}",
        swaps_, rm_uploads, rm_bytes, rm_usec, rm_usec / frames);
    XELOGW(
        "MetalSignCensus v1 pool swaps={} name=unreachable_excluded_dimension "
        "basis=successful_uploads_only uploads={} usec={:.1f} "
        "usec_per_frame={:.4f}",
        swaps_, ex_uploads, ex_usec, ex_usec / frames);
    XELOGW(
        "MetalSignCensus v1 pool swaps={} name=reachable_not_removable_here "
        "basis=successful_uploads_only uploads={} usec={:.1f} "
        "usec_per_frame={:.4f}",
        swaps_, rc_uploads, rc_usec, rc_usec / frames);
    XELOGW(
        "MetalSignCensus v1 pool swaps={} name=failed_miss_excluded_from_pools "
        "basis=misses_that_completed_no_upload missnull={} bytes={} "
        "usec={:.1f} usec_per_frame={:.4f}",
        swaps_, total_miss_null, total_failed_bytes, total_failed_usec,
        total_failed_usec / frames);
    XELOGW("MetalSignCensus v1 end reason={} swaps={}", reason, swaps_);
  }

  // Direct-indexed by the packed key, so snapshots print in a stable key
  // order and the hot path is an array increment. Allocated on first record.
  std::vector<MetalSignCensusBucket> buckets_;
  uint64_t probe_ns_ = 0;
  uint64_t probe_regions_ = 0;
  uint64_t call_seq_ = 0;
  uint64_t sampled_calls_ = 0;
  uint32_t swaps_ = 0;
  bool measuring_ = false;
  bool dumped_final_ = false;
};

// Scoped observer-overhead accumulator for one bookkeeping region. Reads the
// clock only on sampled calls (MetalSignCensus::BeginCall), so the instrument
// does not dominate what it measures.
class MetalSignCensusOverheadScope {
 public:
  explicit MetalSignCensusOverheadScope(MetalSignCensus& census)
      : census_(census), active_(census.measuring()) {
    if (active_) {
      t0_ = std::chrono::steady_clock::now();
    }
  }
  MetalSignCensusOverheadScope()
      : MetalSignCensusOverheadScope(MetalSignCensus::Get()) {}
  ~MetalSignCensusOverheadScope() {
    if (!active_) {
      return;
    }
    census_.AddProbeNs(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0_)
                     .count()));
  }

 private:
  MetalSignCensus& census_;
  bool active_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SIGN_CENSUS_PROBE_H_
