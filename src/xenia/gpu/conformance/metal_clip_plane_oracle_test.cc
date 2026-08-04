/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0: the independent Xenos user-clip-plane oracle
// (fix-metal-geometry-clip-distance-bridge task 3.2).
//
// These cases pin the ORACLE, not the backend. Every expectation below is
// derived by hand from the geometry -- a plane at x >= 0 halves a symmetric
// quad, so the covered pixel count must be half, and so on -- rather than from
// anything the oracle or the Metal implementation produces. That is what makes
// task 4's behavioral acceptance non-circular: if the oracle were tuned to
// agree with the renderer, agreeing with the renderer would prove nothing.

#include <cmath>
#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_clip_plane_oracle.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// A full-viewport quad in guest clip space (w = 1, so NDC == xy).
std::vector<ClipVec4> FullScreenQuad() {
  return {{-1.0f, -1.0f, 0.0f, 1.0f},
          {1.0f, -1.0f, 0.0f, 1.0f},
          {1.0f, 1.0f, 0.0f, 1.0f},
          {-1.0f, 1.0f, 0.0f, 1.0f}};
}

TEST_CASE("Clip oracle: distance is dot4 of position and plane",
          "[l0][metal][clip-oracle]") {
  // The Xenos semantic is a plain 4-component dot product. Hand-computed:
  // plane (1,0,0,0) reports the x coordinate; (0,0,0,1) reports w.
  CHECK(XenosClipDistance({0.5f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 0.0f}) ==
        Approx(0.5f));
  CHECK(XenosClipDistance({-0.25f, 0.0f, 0.0f, 1.0f},
                          {1.0f, 0.0f, 0.0f, 0.0f}) == Approx(-0.25f));
  // Offset plane x + 0.5 >= 0 keeps x >= -0.5.
  CHECK(XenosClipDistance({-0.5f, 0.0f, 0.0f, 1.0f},
                          {1.0f, 0.0f, 0.0f, 0.5f}) == Approx(0.0f));
  CHECK(XenosClipDistance({2.0f, 3.0f, 4.0f, 1.0f},
                          {1.0f, 1.0f, 1.0f, -9.0f}) == Approx(0.0f));
}

TEST_CASE("Clip oracle: classification covers inside, outside and crossing",
          "[l0][metal][clip-oracle]") {
  const std::vector<ClipVec4> quad = FullScreenQuad();
  // x >= 0 halves it -> crossing.
  CHECK(ClassifyAgainstPlane(quad, {1.0f, 0.0f, 0.0f, 0.0f}) ==
        ClipClassification::kCrossing);
  // x + 2 >= 0 contains it entirely -> inside (task 4.2's case).
  CHECK(ClassifyAgainstPlane(quad, {1.0f, 0.0f, 0.0f, 2.0f}) ==
        ClipClassification::kFullyInside);
  // x - 2 >= 0 excludes it entirely -> outside (task 4.3's case).
  CHECK(ClassifyAgainstPlane(quad, {1.0f, 0.0f, 0.0f, -2.0f}) ==
        ClipClassification::kFullyOutside);
  // BOUNDARY: distance exactly 0 is KEPT, so a plane through the edge is not
  // "outside". Sign conventions are the classic place this goes wrong.
  const std::vector<ClipVec4> edge = {{0.0f, 0.0f, 0.0f, 1.0f},
                                      {0.0f, 1.0f, 0.0f, 1.0f},
                                      {0.0f, -1.0f, 0.0f, 1.0f}};
  CHECK(ClassifyAgainstPlane(edge, {1.0f, 0.0f, 0.0f, 0.0f}) ==
        ClipClassification::kFullyInside);
}

TEST_CASE("Clip oracle: fully inside and fully outside coverage",
          "[l0][metal][clip-oracle]") {
  const uint32_t kW = 16, kH = 16;
  const std::vector<ClipVec4> quad = FullScreenQuad();
  // Fully inside -> every pixel covered (task 4.2: renders UNCLIPPED).
  auto inside = ExpectedCoverageUnderPlane(quad, {1.0f, 0.0f, 0.0f, 2.0f}, kW, kH);
  CHECK(CountCovered(inside) == kW * kH);
  // Fully outside -> nothing covered (task 4.3: fully clipped).
  auto outside =
      ExpectedCoverageUnderPlane(quad, {1.0f, 0.0f, 0.0f, -2.0f}, kW, kH);
  CHECK(CountCovered(outside) == 0u);
}

TEST_CASE("Clip oracle: a crossing plane produces exact partial coverage",
          "[l0][metal][clip-oracle]") {
  const uint32_t kW = 16, kH = 16;
  const std::vector<ClipVec4> quad = FullScreenQuad();
  // HAND-DERIVED: x >= 0 keeps the right half of a symmetric full-screen quad,
  // which is exactly half the pixels. Not read off the oracle.
  auto half = ExpectedCoverageUnderPlane(quad, {1.0f, 0.0f, 0.0f, 0.0f}, kW, kH);
  CHECK(CountCovered(half) == (kW * kH) / 2);
  // ...and it is the RIGHT half: NDC x >= 0 maps to the right of screen centre.
  for (uint32_t y = 0; y < kH; ++y) {
    CHECK(half[size_t(y) * kW + 0] == 0);          // leftmost column clipped
    CHECK(half[size_t(y) * kW + (kW - 1)] == 1);   // rightmost kept
  }

  // y >= 0 keeps the TOP half on screen, because screen Y is inverted from NDC.
  auto top = ExpectedCoverageUnderPlane(quad, {0.0f, 1.0f, 0.0f, 0.0f}, kW, kH);
  CHECK(CountCovered(top) == (kW * kH) / 2);
  for (uint32_t x = 0; x < kW; ++x) {
    CHECK(top[size_t(0) * kW + x] == 1);           // top row kept
    CHECK(top[size_t(kH - 1) * kW + x] == 0);      // bottom row clipped
  }

  // A quarter plane: x >= 0 AND y >= 0 applied in sequence leaves a quarter.
  auto poly = ClipPolygonToPlane(quad, {1.0f, 0.0f, 0.0f, 0.0f});
  poly = ClipPolygonToPlane(poly, {0.0f, 1.0f, 0.0f, 0.0f});
  CHECK(CountCovered(RasterizeConvexPolygon(poly, kW, kH)) == (kW * kH) / 4);
}

TEST_CASE("Clip oracle: an off-centre plane cuts at the arithmetic position",
          "[l0][metal][clip-oracle]") {
  const uint32_t kW = 16, kH = 16;
  const std::vector<ClipVec4> quad = FullScreenQuad();
  // x + 0.5 >= 0 keeps x >= -0.5, i.e. NDC [-0.5, 1] = 75% of the width.
  // HAND-DERIVED: 0.75 * 16 = 12 columns of 16 rows = 192 pixels.
  auto three_quarters =
      ExpectedCoverageUnderPlane(quad, {1.0f, 0.0f, 0.0f, 0.5f}, kW, kH);
  CHECK(CountCovered(three_quarters) == 12u * kH);
}

TEST_CASE("Clip oracle: clipping happens before the perspective divide",
          "[l0][metal][clip-oracle]") {
  // A triangle with one vertex behind the eye (w < 0). Dividing FIRST would
  // fold that vertex onto the wrong side and yield a different polygon; the
  // oracle must clip in homogeneous space. Plane w >= 0 removes exactly the
  // behind-the-eye region.
  const std::vector<ClipVec4> tri = {{-1.0f, -1.0f, 0.0f, 1.0f},
                                     {1.0f, -1.0f, 0.0f, 1.0f},
                                     {0.0f, 1.0f, 0.0f, -1.0f}};
  CHECK(ClassifyAgainstPlane(tri, {0.0f, 0.0f, 0.0f, 1.0f}) ==
        ClipClassification::kCrossing);
  auto clipped = ClipPolygonToPlane(tri, {0.0f, 0.0f, 0.0f, 1.0f});
  REQUIRE(clipped.size() >= 3);
  // Every surviving vertex is in front of the eye, so the result is projectable
  // -- which is the property that makes a coverage comparison meaningful.
  for (const ClipVec4& v : clipped) {
    CHECK(v.w >= 0.0f);
  }
}

TEST_CASE("Clip oracle: a plane that removes everything yields no polygon",
          "[l0][metal][clip-oracle]") {
  const std::vector<ClipVec4> quad = FullScreenQuad();
  auto gone = ClipPolygonToPlane(quad, {1.0f, 0.0f, 0.0f, -2.0f});
  CHECK(gone.empty());
  // And a fully-inside plane returns the polygon unchanged in size.
  auto kept = ClipPolygonToPlane(quad, {1.0f, 0.0f, 0.0f, 2.0f});
  CHECK(kept.size() == quad.size());
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
