/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-host-tessellation task 3.4 — device-free L0 oracle for the
// host route's PURE expansion helpers (metal_host_tessellation.h):
//
//   1. the shape table the shared ProcessingResult maps onto (record lanes,
//      stream words per patch, the adaptive/CP-indexed exclusion);
//   2. index resolution = the host vertex / hull shaders' clamp((I + offset)
//      & 0xFFFFFF, min, max) with HLSL min(max()) order (spec scenario:
//      offset 5, entries (0xFFFFFF, 1, 2), range [1,4] -> (4, 4, 4));
//   3. per-patch record lanes for every host VS type (CP indices from the
//      stream or sequential ids; patch ordinal for patch-indexed shapes);
//   4. Count/Write agreement, absolute index construction, and the budget
//      skip (over_budget, never a partial expansion);
//   5. the XeTessellationVertex record layout the translator's ld_raw ABI
//      assumes.

#include <cstdint>
#include <cstring>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_host_tessellation.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

using Domain = HostTessellator::Domain;
using Type = Shader::HostVertexShaderType;

uint32_t Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

TEST_CASE("Host tessellation record layout matches the translator ABI",
          "[l0][tessellation][host_route]") {
  STATIC_REQUIRE(sizeof(XeTessellationVertex) == 32);
  STATIC_REQUIRE(offsetof(XeTessellationVertex, domain_location) == 0);
  STATIC_REQUIRE(offsetof(XeTessellationVertex, control_point_index) == 16);
  STATIC_REQUIRE(kXeTessellationVertexSizeBytes == 32);
}

TEST_CASE("Host tessellation shape table", "[l0][tessellation][host_route]") {
  MetalHostTessellationShape s;
  REQUIRE(DescribeMetalHostTessellationShape(
      Type::kTriangleDomainCPIndexed, xenos::TessellationMode::kContinuous, &s));
  CHECK(s.domain == Domain::kTriangle);
  CHECK(!s.adaptive);
  CHECK(s.control_point_indexed);
  CHECK(s.control_points == 3);
  CHECK(s.stream_words_per_patch == 3);

  REQUIRE(DescribeMetalHostTessellationShape(
      Type::kQuadDomainCPIndexed, xenos::TessellationMode::kDiscrete, &s));
  CHECK(s.domain == Domain::kQuad);
  CHECK(s.control_point_indexed);
  CHECK(s.control_points == 4);
  CHECK(s.stream_words_per_patch == 4);

  REQUIRE(DescribeMetalHostTessellationShape(
      Type::kTriangleDomainPatchIndexed, xenos::TessellationMode::kAdaptive, &s));
  CHECK(s.adaptive);
  CHECK(!s.control_point_indexed);
  CHECK(s.control_points == 1);
  CHECK(s.stream_words_per_patch == 3);  // 3 edge factors per patch.

  REQUIRE(DescribeMetalHostTessellationShape(
      Type::kQuadDomainPatchIndexed, xenos::TessellationMode::kAdaptive, &s));
  CHECK(s.stream_words_per_patch == 4);

  // Non-adaptive patch lists carry one patch index per patch.
  REQUIRE(DescribeMetalHostTessellationShape(
      Type::kQuadDomainPatchIndexed, xenos::TessellationMode::kContinuous, &s));
  CHECK(!s.adaptive);
  CHECK(s.control_points == 1);
  CHECK(s.stream_words_per_patch == 1);

  // Exclusions: plain vertex, and adaptive CP-indexed (never produced by the
  // shared processor).
  CHECK(!DescribeMetalHostTessellationShape(
      Type::kVertex, xenos::TessellationMode::kDiscrete, &s));
  CHECK(!DescribeMetalHostTessellationShape(
      Type::kTriangleDomainCPIndexed, xenos::TessellationMode::kAdaptive, &s));
}

TEST_CASE("Host tessellation index resolution matches the hull shaders",
          "[l0][tessellation][host_route][index]") {
  // Spec scenario: offset 5, entries (0xFFFFFF, 1, 2), range [1, 4].
  // (0xFFFFFF + 5) & 0xFFFFFF = 4 -> clamp 4; 6 -> 4; 7 -> 4.
  CHECK(ResolveMetalTessellationIndex(0xFFFFFFu, 5, 1, 4) == 4u);
  CHECK(ResolveMetalTessellationIndex(1, 5, 1, 4) == 4u);
  CHECK(ResolveMetalTessellationIndex(2, 5, 1, 4) == 4u);
  // Inside the range: identity plus offset.
  CHECK(ResolveMetalTessellationIndex(2, 0, 0, 0xFFFFFFu) == 2u);
  CHECK(ResolveMetalTessellationIndex(2, 3, 0, 0xFFFFFFu) == 5u);
  // Below the minimum clamps up.
  CHECK(ResolveMetalTessellationIndex(0, 0, 10, 20) == 10u);
  // Only the low 24 bits of (index + offset) survive.
  CHECK(ResolveMetalTessellationIndex(0x01000003u, 0, 0, 0xFFFFFFu) == 3u);
  // min > max collapses to max (HLSL min(max()) order).
  CHECK(ResolveMetalTessellationIndex(7, 0, 9, 4) == 4u);
}

MetalHostTessellationDrawInput MakeInput(Type type,
                                         xenos::TessellationMode mode,
                                         const void* stream, bool stream_32,
                                         uint32_t words, float range_min,
                                         float range_max) {
  MetalHostTessellationDrawInput in{};
  REQUIRE(DescribeMetalHostTessellationShape(type, mode, &in.shape));
  in.mode = mode;
  in.factor_range_min = range_min;
  in.factor_range_max = range_max;
  in.index_offset = 0;
  in.min_index = 0;
  in.max_index = 0xFFFFFFu;
  in.stream = stream;
  in.stream_32bit = stream_32;
  in.stream_word_count = words;
  return in;
}

TEST_CASE("Host tessellation per-patch lanes per host VS type",
          "[l0][tessellation][host_route][lanes]") {
  HostTessellationFactors f;
  float lanes[4];

  SECTION("triangle CP-indexed from a 16-bit stream") {
    const uint16_t stream[6] = {10, 11, 12, 20, 21, 22};
    auto in = MakeInput(Type::kTriangleDomainCPIndexed,
                        xenos::TessellationMode::kDiscrete, stream, false, 6,
                        3.0f, 3.0f);
    in.index_offset = 100;
    CHECK(in.patch_count() == 2);
    ComputeMetalHostTessellationPatch(in, 1, &f, lanes);
    CHECK(lanes[0] == 120.0f);
    CHECK(lanes[1] == 121.0f);
    CHECK(lanes[2] == 122.0f);
    CHECK(lanes[3] == 0.0f);
    CHECK(f.edges[0] == 3.0f);
    CHECK(f.inside[0] == 3.0f);
  }
  SECTION("quad CP-indexed auto-indexed: sequential vertex ids") {
    auto in = MakeInput(Type::kQuadDomainCPIndexed,
                        xenos::TessellationMode::kContinuous, nullptr, false,
                        8, 2.0f, 4.0f);
    CHECK(in.patch_count() == 2);
    ComputeMetalHostTessellationPatch(in, 1, &f, lanes);
    CHECK(lanes[0] == 4.0f);
    CHECK(lanes[1] == 5.0f);
    CHECK(lanes[2] == 6.0f);
    CHECK(lanes[3] == 7.0f);
    for (uint32_t i = 0; i < 4; ++i) CHECK(f.edges[i] == 4.0f);
    CHECK(f.inside[0] == 4.0f);
    CHECK(f.inside[1] == 4.0f);
  }
  SECTION("adaptive quad patch: factors from the stream, patch ordinal index") {
    // Two patches; the second has asymmetric guest factors (1, 3, 5, 7).
    const uint32_t stream[8] = {Bits(0.0f), Bits(0.0f), Bits(0.0f), Bits(0.0f),
                                Bits(1.0f), Bits(3.0f), Bits(5.0f), Bits(7.0f)};
    auto in = MakeInput(Type::kQuadDomainPatchIndexed,
                        xenos::TessellationMode::kAdaptive, stream, true, 8,
                        1.0f, 64.0f);
    in.index_offset = 7;
    CHECK(in.patch_count() == 2);
    ComputeMetalHostTessellationPatch(in, 1, &f, lanes);
    CHECK(lanes[0] == 8.0f);  // patch 1 + offset 7
    CHECK(lanes[1] == 0.0f);
    // adaptive_quad.hs.hlsl: edges[i] = f[(i+3)&3] + 1.
    CHECK(f.edges[0] == 8.0f);
    CHECK(f.edges[1] == 2.0f);
    CHECK(f.edges[2] == 4.0f);
    CHECK(f.edges[3] == 6.0f);
    CHECK(f.inside[0] == 2.0f);
    CHECK(f.inside[1] == 4.0f);
  }
  SECTION("adaptive triangle patch: [2,1] range collapses every factor") {
    const uint32_t stream[3] = {Bits(7.0f), Bits(7.0f), Bits(7.0f)};
    auto in = MakeInput(Type::kTriangleDomainPatchIndexed,
                        xenos::TessellationMode::kAdaptive, stream, true, 3,
                        2.0f, 1.0f);
    ComputeMetalHostTessellationPatch(in, 0, &f, lanes);
    CHECK(lanes[0] == 0.0f);
    for (uint32_t i = 0; i < 3; ++i) CHECK(f.edges[i] == 1.0f);
    CHECK(f.inside[0] == 1.0f);
  }
  SECTION("non-adaptive patch list: one stream word per patch is the index") {
    const uint32_t stream[2] = {40, 41};
    auto in = MakeInput(Type::kTriangleDomainPatchIndexed,
                        xenos::TessellationMode::kContinuous, stream, true, 2,
                        2.0f, 2.0f);
    CHECK(in.patch_count() == 2);
    ComputeMetalHostTessellationPatch(in, 1, &f, lanes);
    CHECK(lanes[0] == 41.0f);
  }
  SECTION("incomplete trailing patch is dropped") {
    auto in = MakeInput(Type::kQuadDomainCPIndexed,
                        xenos::TessellationMode::kDiscrete, nullptr, false, 7,
                        1.0f, 1.0f);
    CHECK(in.patch_count() == 1);
  }
}

TEST_CASE("Host tessellation count/write agree and build absolute indices",
          "[l0][tessellation][host_route][expand]") {
  HostTessellator tessellator;
  // Two quad patches at integer factor 2: 9 points / 8 triangles each.
  auto in = MakeInput(Type::kQuadDomainCPIndexed,
                      xenos::TessellationMode::kDiscrete, nullptr, false, 8,
                      2.0f, 2.0f);
  auto counts = CountMetalHostTessellation(in, tessellator);
  CHECK(!counts.over_budget);
  CHECK(counts.patch_count == 2);
  CHECK(counts.vertex_count == 18);
  CHECK(counts.index_count == 48);
  std::vector<XeTessellationVertex> records(counts.vertex_count);
  std::vector<uint32_t> indices(counts.index_count);
  WriteMetalHostTessellation(in, tessellator, records.data(), indices.data());
  // Second patch's indices are offset by the first patch's vertex count and
  // every index addresses a record of its own patch.
  for (size_t i = 0; i < 24; ++i) CHECK(indices[i] < 9);
  for (size_t i = 24; i < 48; ++i) {
    CHECK(indices[i] >= 9);
    CHECK(indices[i] < 18);
  }
  // Records: domain w = 0 for quads, lanes = the patch's control points.
  for (size_t v = 0; v < 9; ++v) {
    CHECK(records[v].domain_location[2] == 0.0f);
    CHECK(records[v].domain_location[3] == 0.0f);
    CHECK(records[v].control_point_index[0] == 0.0f);
    CHECK(records[v].control_point_index[3] == 3.0f);
  }
  for (size_t v = 9; v < 18; ++v) {
    CHECK(records[v].control_point_index[0] == 4.0f);
    CHECK(records[v].control_point_index[3] == 7.0f);
  }
  // Domain corners present in the first patch.
  bool has_00 = false, has_11 = false;
  for (size_t v = 0; v < 9; ++v) {
    has_00 |= records[v].domain_location[0] == 0.0f &&
              records[v].domain_location[1] == 0.0f;
    has_11 |= records[v].domain_location[0] == 1.0f &&
              records[v].domain_location[1] == 1.0f;
  }
  CHECK(has_00);
  CHECK(has_11);
}

TEST_CASE("Host tessellation budget fails the whole draw",
          "[l0][tessellation][host_route][budget]") {
  HostTessellator tessellator;
  // 4 quad patches at factor 8: 81 points each = 324 vertices.
  auto in = MakeInput(Type::kQuadDomainCPIndexed,
                      xenos::TessellationMode::kDiscrete, nullptr, false, 16,
                      8.0f, 8.0f);
  auto ok = CountMetalHostTessellation(in, tessellator, 324);
  CHECK(!ok.over_budget);
  CHECK(ok.vertex_count == 324);
  auto over = CountMetalHostTessellation(in, tessellator, 323);
  CHECK(over.over_budget);
  // The production ceiling fits comfortably inside the 32 MiB upload ring.
  STATIC_REQUIRE(kMaxMetalHostTessellationVerticesPerDraw *
                     kXeTessellationVertexSizeBytes ==
                 8u * 1024u * 1024u);
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
