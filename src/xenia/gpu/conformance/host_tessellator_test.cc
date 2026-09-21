/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-host-tessellation task 1.3 - device-free L0 oracle for the
// host tessellator (design D9.1):
//
//   1. integer partitioning produces the closed-form point/triangle counts;
//   2. every domain point lies in its domain and every triangle has one
//      consistent winding (the reference's PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CW);
//   3. factor derivation matches the canned hull shaders: explicit
//      min(max()) clamp with the live Viva Pinata [2,1] collapse, the
//      asymmetric adaptive edge remaps for both domains, inside = min;
//   4. the pattern cache returns the identical object for equal factor tuples
//      and evicts least-recently-used;
//   5. golden content hashes per (partitioning, domain, factor) pin the
//      vendored reference's arithmetic on this platform.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/host_tessellator.h"

namespace xe {
namespace gpu {
namespace {

using Domain = HostTessellator::Domain;

HostTessellationFactors Uniform(Domain domain, float f) {
  return HostTessellator::ComputeFactors(xenos::TessellationMode::kDiscrete,
                                         domain, f, f, nullptr);
}

// Signed doubled area of a triangle in (u, v); sign == winding.
double SignedArea(const HostTessellationPattern& p, size_t tri) {
  const auto& a = p.domain_points[p.indices[tri * 3 + 0]];
  const auto& b = p.domain_points[p.indices[tri * 3 + 1]];
  const auto& c = p.domain_points[p.indices[tri * 3 + 2]];
  return (double(b[0]) - a[0]) * (double(c[1]) - a[1]) -
         (double(b[1]) - a[1]) * (double(c[0]) - a[0]);
}

void CheckPatternWellFormed(const HostTessellationPattern& p, Domain domain) {
  REQUIRE(!p.domain_points.empty());
  REQUIRE(p.indices.size() % 3 == 0);
  REQUIRE(p.domain_points.size() <= HostTessellator::kMaxPointsPerPatch);
  REQUIRE(p.triangle_count() <= HostTessellator::kMaxTrianglesPerPatch);
  for (const auto& d : p.domain_points) {
    CHECK(d[0] >= 0.0f);
    CHECK(d[1] >= 0.0f);
    CHECK(d[0] <= 1.0f);
    CHECK(d[1] <= 1.0f);
    if (domain == Domain::kTriangle) {
      CHECK(d[0] + d[1] <= 1.0f + 1e-6f);
      CHECK(std::fabs((1.0f - d[0] - d[1]) - d[2]) <= 1e-6f);
    } else {
      CHECK(d[2] == 0.0f);
    }
  }
  int positive = 0, negative = 0, degenerate = 0;
  for (size_t t = 0; t < p.triangle_count(); ++t) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(p.indices[t * 3 + k] < p.domain_points.size());
    }
    double area = SignedArea(p, t);
    if (area > 0) {
      ++positive;
    } else if (area < 0) {
      ++negative;
    } else {
      ++degenerate;
    }
  }
  CHECK(degenerate == 0);
  // One consistent winding across the whole patch. Direct3D's "triangle_cw"
  // is clockwise with v pointing DOWN (screen convention), which in the
  // mathematical (u right, v up) frame used here is a POSITIVE signed area.
  // Measured on the vendored reference; every cell agrees.
  CHECK(negative == 0);
  CHECK(positive == int(p.triangle_count()));
}

TEST_CASE("Host tessellator integer partitioning counts match the closed form",
          "[l0][tessellation][host_tessellator]") {
  HostTessellator tess;
  for (uint32_t n = 1; n <= 8; ++n) {
    float f = float(n);
    {
      const auto& p = tess.Tessellate(xenos::TessellationMode::kDiscrete,
                                      Domain::kTriangle,
                                      Uniform(Domain::kTriangle, f));
      INFO("triangle n=" << n);
      // The Direct3D triangle domain is NOT the barycentric grid: it is
      // built from concentric rings around the inside factor, so there is no
      // (n+1)(n+2)/2 closed form (n=2 gives 7 points / 6 triangles, n=3
      // gives 12 / 13). What IS fixed: each edge is split into exactly n
      // segments (3n boundary points), and a triangulated disk obeys
      // T = 2V - B - 2.
      size_t boundary = 0;
      for (const auto& d : p.domain_points) {
        if (d[0] == 0.0f || d[1] == 0.0f || d[2] <= 0.0f) {
          ++boundary;
        }
      }
      CHECK(boundary == size_t(3 * n));
      CHECK(p.triangle_count() == 2 * p.domain_points.size() - boundary - 2);
      CheckPatternWellFormed(p, Domain::kTriangle);
    }
    {
      const auto& p = tess.Tessellate(xenos::TessellationMode::kDiscrete,
                                      Domain::kQuad, Uniform(Domain::kQuad, f));
      INFO("quad n=" << n);
      CHECK(p.domain_points.size() == size_t((n + 1) * (n + 1)));
      CHECK(p.triangle_count() == size_t(2 * n * n));
      CheckPatternWellFormed(p, Domain::kQuad);
    }
  }
  // Maximum factor: the reference's storage bound is exactly the design's.
  const auto& q = tess.Tessellate(xenos::TessellationMode::kDiscrete,
                                  Domain::kQuad, Uniform(Domain::kQuad, 64.0f));
  CHECK(q.domain_points.size() == HostTessellator::kMaxPointsPerPatch);
  CHECK(q.triangle_count() == HostTessellator::kMaxTrianglesPerPatch);
}

TEST_CASE("Host tessellator fractional_even patterns are well formed",
          "[l0][tessellation][host_tessellator]") {
  HostTessellator tess;
  const float factors[] = {1.0f, 2.0f, 2.5f, 3.0f, 5.2215f, 7.1951f, 8.0f};
  for (float f : factors) {
    for (Domain domain : {Domain::kTriangle, Domain::kQuad}) {
      const auto& p = tess.Tessellate(
          xenos::TessellationMode::kContinuous, domain,
          HostTessellator::ComputeFactors(xenos::TessellationMode::kContinuous,
                                          domain, 1.0f, f, nullptr));
      INFO("fractional_even f=" << f
                                << (domain == Domain::kTriangle ? " tri"
                                                                : " quad"));
      CheckPatternWellFormed(p, domain);
    }
  }
  // fractional_even rounds a factor of 1 up to 2 (the reference's minimum
  // even factor), so factor 1.0 and 2.0 produce the same geometry.
  const auto& one = tess.Tessellate(
      xenos::TessellationMode::kContinuous, Domain::kQuad,
      HostTessellator::ComputeFactors(xenos::TessellationMode::kContinuous,
                                      Domain::kQuad, 1.0f, 1.0f, nullptr));
  const auto& two = tess.Tessellate(
      xenos::TessellationMode::kContinuous, Domain::kQuad,
      HostTessellator::ComputeFactors(xenos::TessellationMode::kContinuous,
                                      Domain::kQuad, 1.0f, 2.0f, nullptr));
  CHECK(one.content_hash == two.content_hash);
}

TEST_CASE("Host tessellator clamp is HLSL min(max()) with the [2,1] collapse",
          "[l0][tessellation][host_tessellator][clamp]") {
  // Ordinary range.
  CHECK(HostTessellator::ClampFactor(0.5f, 1.0f, 8.0f) == 1.0f);
  CHECK(HostTessellator::ClampFactor(3.0f, 1.0f, 8.0f) == 3.0f);
  CHECK(HostTessellator::ClampFactor(9.0f, 1.0f, 8.0f) == 8.0f);
  // lo > hi collapses to hi, whatever x is (the live [2,1] range).
  CHECK(HostTessellator::ClampFactor(0.0f, 2.0f, 1.0f) == 1.0f);
  CHECK(HostTessellator::ClampFactor(1.5f, 2.0f, 1.0f) == 1.0f);
  CHECK(HostTessellator::ClampFactor(50.0f, 2.0f, 1.0f) == 1.0f);
  // Direct3D min/max return the non-NaN operand.
  float nan = std::numeric_limits<float>::quiet_NaN();
  CHECK(HostTessellator::ClampFactor(nan, 2.0f, 8.0f) == 2.0f);
  CHECK(HostTessellator::ClampFactor(nan, 2.0f, 1.0f) == 1.0f);

  // Through ComputeFactors: an adaptive stream under [2,1] renders exactly
  // like an honest factor-1.0 stream (the six-cell render oracle's arm, now
  // at the pattern level).
  HostTessellator tess;
  const float stream[4] = {0.0f, 3.0f, 7.5f, 1.0f};
  auto inverted = HostTessellator::ComputeFactors(
      xenos::TessellationMode::kAdaptive, Domain::kQuad, 2.0f, 1.0f, stream);
  for (float e : inverted.edges) CHECK(e == 1.0f);
  for (float i : inverted.inside) CHECK(i == 1.0f);
  const float ones[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // +1 -> 1.0 everywhere
  auto honest = HostTessellator::ComputeFactors(
      xenos::TessellationMode::kAdaptive, Domain::kQuad, 1.0f, 8.0f, ones);
  CHECK(tess.Tessellate(xenos::TessellationMode::kAdaptive, Domain::kQuad,
                        inverted)
            .content_hash ==
        tess.Tessellate(xenos::TessellationMode::kAdaptive, Domain::kQuad,
                        honest)
            .content_hash);
}

TEST_CASE("Host tessellator adaptive edge remaps match the hull shaders",
          "[l0][tessellation][host_tessellator][remap]") {
  // Guest factors (before +1) chosen distinct so any misplacement shows.
  const float tri_stream[3] = {2.0f, 4.0f, 6.0f};
  auto tri = HostTessellator::ComputeFactors(
      xenos::TessellationMode::kAdaptive, Domain::kTriangle, 1.0f, 64.0f,
      tri_stream);
  // adaptive_triangle.hs.hlsl: edges[i] = f[(i + 1) % 3] (after +1).
  CHECK(tri.edges[0] == 5.0f);
  CHECK(tri.edges[1] == 7.0f);
  CHECK(tri.edges[2] == 3.0f);
  CHECK(tri.inside[0] == 3.0f);
  CHECK(tri.edges[3] == 0.0f);
  CHECK(tri.inside[1] == 0.0f);

  const float quad_stream[4] = {2.0f, 4.0f, 6.0f, 8.0f};
  auto quad = HostTessellator::ComputeFactors(
      xenos::TessellationMode::kAdaptive, Domain::kQuad, 1.0f, 64.0f,
      quad_stream);
  // adaptive_quad.hs.hlsl: edges[i] = f[(i + 3) & 3] (after +1);
  // inside[0] = min(e1, e3); inside[1] = min(e0, e2).
  CHECK(quad.edges[0] == 9.0f);
  CHECK(quad.edges[1] == 3.0f);
  CHECK(quad.edges[2] == 5.0f);
  CHECK(quad.edges[3] == 7.0f);
  CHECK(quad.inside[0] == 3.0f);
  CHECK(quad.inside[1] == 5.0f);

  // Non-adaptive modes ignore the stream entirely.
  auto disc = HostTessellator::ComputeFactors(
      xenos::TessellationMode::kDiscrete, Domain::kTriangle, 1.0f, 4.0f,
      tri_stream);
  CHECK(disc.edges[0] == 4.0f);
  CHECK(disc.edges[1] == 4.0f);
  CHECK(disc.edges[2] == 4.0f);
  CHECK(disc.inside[0] == 4.0f);

  // [6,3] and [3,6] uniform ranges differ; asymmetric per-edge factors
  // differ from their rotation (the remap is observable in the pattern).
  HostTessellator tess;
  const auto& a = tess.Tessellate(xenos::TessellationMode::kAdaptive,
                                  Domain::kQuad, quad);
  const float rotated_stream[4] = {4.0f, 6.0f, 8.0f, 2.0f};
  auto rotated = HostTessellator::ComputeFactors(
      xenos::TessellationMode::kAdaptive, Domain::kQuad, 1.0f, 64.0f,
      rotated_stream);
  const auto& b = tess.Tessellate(xenos::TessellationMode::kAdaptive,
                                  Domain::kQuad, rotated);
  CHECK(a.content_hash != b.content_hash);
  CHECK(a.domain_points.size() == b.domain_points.size());
}

TEST_CASE("Host tessellator pattern cache is identity for equal tuples and LRU",
          "[l0][tessellation][host_tessellator][cache]") {
  HostTessellator tess(/*pattern_cache_capacity=*/2);
  auto f3 = Uniform(Domain::kQuad, 3.0f);
  auto f4 = Uniform(Domain::kQuad, 4.0f);
  auto f5 = Uniform(Domain::kQuad, 5.0f);
  const auto* p3 =
      &tess.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f3);
  CHECK(tess.pattern_cache_misses() == 1);
  const auto* p3_again =
      &tess.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f3);
  CHECK(p3 == p3_again);
  CHECK(tess.pattern_cache_misses() == 1);
  // Same factors, different partitioning or domain: distinct entries.
  tess.Tessellate(xenos::TessellationMode::kContinuous, Domain::kQuad, f3);
  CHECK(tess.pattern_cache_misses() == 2);
  CHECK(tess.pattern_cache_size() == 2);
  // Touch f3 (discrete) so the continuous entry is the LRU victim.
  tess.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f3);
  tess.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f4);
  CHECK(tess.pattern_cache_misses() == 3);
  CHECK(tess.pattern_cache_size() == 2);
  // f3 discrete survived (hit), continuous was evicted (miss on re-request).
  tess.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f3);
  CHECK(tess.pattern_cache_misses() == 3);
  tess.Tessellate(xenos::TessellationMode::kContinuous, Domain::kQuad, f3);
  CHECK(tess.pattern_cache_misses() == 4);
  // A pattern's content is independent of cache history.
  uint64_t h5 =
      tess.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f5)
          .content_hash;
  HostTessellator fresh;
  CHECK(fresh.Tessellate(xenos::TessellationMode::kDiscrete, Domain::kQuad, f5)
            .content_hash == h5);
}

TEST_CASE("Host tessellator golden pattern hashes",
          "[l0][tessellation][host_tessellator][golden]") {
  // Pinned from the first Release run on Apple M4 (2026-09-20); a change
  // here means the vendored reference's arithmetic changed, which is a
  // D3D12-parity regression until proven otherwise.
  struct Golden {
    const char* name;
    xenos::TessellationMode mode;
    Domain domain;
    float range_min;
    float range_max;
    const float* stream;
    uint64_t hash;
  };
  static const float kAdaptiveTri[3] = {1.5f, 3.25f, 5.0f};
  static const float kAdaptiveQuad[4] = {1.5f, 3.25f, 5.0f, 6.75f};
  const Golden goldens[] = {
      {"discrete/tri/4", xenos::TessellationMode::kDiscrete, Domain::kTriangle,
       4.0f, 4.0f, nullptr, 0xE9EAEF50E2D660C2ull},
      {"discrete/quad/4", xenos::TessellationMode::kDiscrete, Domain::kQuad,
       4.0f, 4.0f, nullptr, 0x7300DBC631A34BA5ull},
      {"continuous/tri/5.2215", xenos::TessellationMode::kContinuous,
       Domain::kTriangle, 2.0f, 5.2215f, nullptr, 0xE9BFF09F3A3C8CCFull},
      {"continuous/quad/7.1951", xenos::TessellationMode::kContinuous,
       Domain::kQuad, 2.0f, 7.1951f, nullptr, 0x7FC26B9B7FE31499ull},
      {"adaptive/tri/asym", xenos::TessellationMode::kAdaptive,
       Domain::kTriangle, 1.0f, 8.0f, kAdaptiveTri, 0x12BEFED80C1FE022ull},
      {"adaptive/quad/asym", xenos::TessellationMode::kAdaptive,
       Domain::kQuad, 1.0f, 8.0f, kAdaptiveQuad, 0xBFC49CCF1A083B4Dull},
  };
  HostTessellator tess;
  for (const Golden& g : goldens) {
    auto factors = HostTessellator::ComputeFactors(
        g.mode, g.domain, g.range_min, g.range_max, g.stream);
    const auto& p = tess.Tessellate(g.mode, g.domain, factors);
    std::printf("TESSGOLD|%s|points=%zu|tris=%zu|hash=%016llX\n", g.name,
                p.domain_points.size(), p.triangle_count(),
                (unsigned long long)p.content_hash);
    if (g.hash) {
      INFO(g.name);
      CHECK(p.content_hash == g.hash);
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xe
