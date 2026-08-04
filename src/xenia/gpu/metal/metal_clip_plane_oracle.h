/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_CLIP_PLANE_ORACLE_H_
#define XENIA_GPU_METAL_METAL_CLIP_PLANE_ORACLE_H_

// fix-metal-geometry-clip-distance-bridge task 3.2: a user-clip-plane oracle
// derived from XENOS PLANE SEMANTICS, evaluated INDEPENDENTLY of the Metal
// implementation, so behavioral acceptance in task 4 is not circular.
//
// WHY INDEPENDENCE MATTERS HERE. The remedy under test (XE_METAL_CLIP_RELABEL,
// default off) makes clip-1/clip-2 geometry pipelines MATERIALIZE. Materialization
// is not correctness: a pipeline that creates and then clips wrongly is worse
// than one that fails loudly, because the failure is currently a dropped-draw
// class that is at least visible in the skip accounting. Accepting the fix
// therefore needs an expected coverage computed WITHOUT reference to what the
// implementation produces.
//
// WHAT IS TAKEN FROM THE IMPLEMENTATION, AND WHY THAT IS NOT CIRCULAR.
// Exactly one fact: WHICH SPACE the plane operates in. The translator emits
//     clip_distance[i] = dot4(position, user_clip_planes[i])
// against `system_temp_position_` BEFORE the guest->host NDC scale/offset
// conversion (dxbc_shader_translator.cc, the UCP loop). That is an input to the
// oracle -- the coordinate space of the test -- not the clipping rule. The rule
// below (keep d >= 0, Sutherland-Hodgman in homogeneous space) is the standard
// Xenos/D3D half-space semantic and is derived independently. If the
// implementation later changes WHICH position it dots against, this oracle must
// be revisited deliberately; that is a real coupling and is recorded rather than
// hidden.
//
// SIGN CONVENTION: a vertex is KEPT where its clip distance is >= 0. A primitive
// with all distances >= 0 is unclipped; with all < 0 it is fully removed;
// otherwise it is partially clipped and the surviving region is the d >= 0 side.

#include <array>
#include <cstdint>
#include <vector>

namespace xe {
namespace gpu {
namespace metal {

struct ClipVec4 {
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

// dot4(position, plane) -- the guest clip distance for one user clip plane.
inline float XenosClipDistance(const ClipVec4& position, const ClipVec4& plane) {
  return position.x * plane.x + position.y * plane.y + position.z * plane.z +
         position.w * plane.w;
}

enum class ClipClassification {
  kFullyInside,   // every vertex has d >= 0: render unclipped
  kFullyOutside,  // every vertex has d < 0: primitive fully removed
  kCrossing,      // mixed: partial coverage on the d >= 0 side
};

inline ClipClassification ClassifyAgainstPlane(
    const std::vector<ClipVec4>& polygon, const ClipVec4& plane) {
  bool any_in = false, any_out = false;
  for (const ClipVec4& v : polygon) {
    if (XenosClipDistance(v, plane) >= 0.0f) {
      any_in = true;
    } else {
      any_out = true;
    }
  }
  if (any_in && any_out) {
    return ClipClassification::kCrossing;
  }
  return any_in ? ClipClassification::kFullyInside
                : ClipClassification::kFullyOutside;
}

// Sutherland-Hodgman against the half-space d >= 0, performed in HOMOGENEOUS
// clip space (before the perspective divide), which is where clipping must
// happen: dividing first would fold vertices behind the eye (w <= 0) onto the
// wrong side and silently produce a different polygon.
inline std::vector<ClipVec4> ClipPolygonToPlane(
    const std::vector<ClipVec4>& polygon, const ClipVec4& plane) {
  std::vector<ClipVec4> out;
  const size_t n = polygon.size();
  if (!n) {
    return out;
  }
  for (size_t i = 0; i < n; ++i) {
    const ClipVec4& cur = polygon[i];
    const ClipVec4& prev = polygon[(i + n - 1) % n];
    const float d_cur = XenosClipDistance(cur, plane);
    const float d_prev = XenosClipDistance(prev, plane);
    const bool in_cur = d_cur >= 0.0f;
    const bool in_prev = d_prev >= 0.0f;
    if (in_cur != in_prev) {
      // Crossing: the intersection is linear in homogeneous space, so a plain
      // lerp on all four components is exact.
      const float denom = d_prev - d_cur;
      const float t = denom != 0.0f ? d_prev / denom : 0.0f;
      ClipVec4 hit;
      hit.x = prev.x + (cur.x - prev.x) * t;
      hit.y = prev.y + (cur.y - prev.y) * t;
      hit.z = prev.z + (cur.z - prev.z) * t;
      hit.w = prev.w + (cur.w - prev.w) * t;
      out.push_back(hit);
    }
    if (in_cur) {
      out.push_back(cur);
    }
  }
  return out;
}

// Perspective divide + viewport transform to pixel centres. Vertices with
// w <= 0 are behind the eye; a caller that clipped first cannot produce them for
// a plane that excludes them, so this reports failure rather than inventing a
// projection.
inline bool ProjectToScreen(const ClipVec4& v, uint32_t width, uint32_t height,
                            float& out_x, float& out_y) {
  if (v.w <= 0.0f) {
    return false;
  }
  const float ndc_x = v.x / v.w;
  const float ndc_y = v.y / v.w;
  out_x = (ndc_x * 0.5f + 0.5f) * float(width);
  // Screen Y is inverted relative to NDC Y.
  out_y = (1.0f - (ndc_y * 0.5f + 0.5f)) * float(height);
  return true;
}

// Expected coverage mask (row-major, 1 = covered) for a convex polygon, by
// pixel-centre containment. Deliberately a simple even-odd/edge test rather than
// a rasterization model: the oracle answers "which pixels SHOULD be covered by
// this geometry", and must not reproduce any implementation's fill rule.
inline std::vector<uint8_t> RasterizeConvexPolygon(
    const std::vector<ClipVec4>& polygon, uint32_t width, uint32_t height) {
  std::vector<uint8_t> mask(size_t(width) * height, 0);
  if (polygon.size() < 3) {
    return mask;
  }
  std::vector<std::array<float, 2>> pts;
  pts.reserve(polygon.size());
  for (const ClipVec4& v : polygon) {
    float sx, sy;
    if (!ProjectToScreen(v, width, height, sx, sy)) {
      return mask;  // behind the eye: no defensible expected coverage
    }
    pts.push_back({sx, sy});
  }
  for (uint32_t py = 0; py < height; ++py) {
    for (uint32_t px = 0; px < width; ++px) {
      const float cx = float(px) + 0.5f;
      const float cy = float(py) + 0.5f;
      bool all_neg = true, all_pos = true;
      for (size_t i = 0; i < pts.size(); ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % pts.size()];
        const float cross =
            (b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0]);
        if (cross > 0.0f) all_neg = false;
        if (cross < 0.0f) all_pos = false;
      }
      if (all_neg || all_pos) {
        mask[size_t(py) * width + px] = 1;
      }
    }
  }
  return mask;
}

// Convenience: expected coverage of a primitive under one user clip plane.
inline std::vector<uint8_t> ExpectedCoverageUnderPlane(
    const std::vector<ClipVec4>& polygon, const ClipVec4& plane, uint32_t width,
    uint32_t height) {
  return RasterizeConvexPolygon(ClipPolygonToPlane(polygon, plane), width,
                                height);
}

inline uint32_t CountCovered(const std::vector<uint8_t>& mask) {
  uint32_t n = 0;
  for (uint8_t m : mask) {
    n += m ? 1u : 0u;
  }
  return n;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_CLIP_PLANE_ORACLE_H_
