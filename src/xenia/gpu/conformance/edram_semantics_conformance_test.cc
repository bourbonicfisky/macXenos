/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 2 of the Metal conformance ladder: a REFERENCE EDRAM state machine that
// encodes the semantic-transition contracts of the sample-accurate EDRAM / RT
// cache redesign as executable, value-level invariants -- the oracle that
// redesign is diffed against.
//
// L0 proved addresses are distinct; L1 proved Metal reproduces the format math.
// L2 composes multiple operations (store as 4x -> select a sample; store as
// color -> reinterpret as depth; disjoint ownership) and asserts the VALUES
// survive the transition, because "which contract did this op violate?" is the
// question every hard bug this era reduced to (see the SF4 white-out).
//
// A true PM4 replay needs the monolithic command processor, which would drag in
// the whole emulator and cost the lean, agent-runnable property that makes this
// ladder useful. So L2 here models EDRAM at the addressing-oracle level
// (XeEdramSnapshotOffsetBytes) -- the same tile/sample layout the backend and
// edram.xesli use. When a real synthetic-PM4 harness against the live Metal
// machine exists, it becomes L2.5/L3 and is checked against THIS model.

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/trace_analysis_decode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

using xenos::MsaaSamples;

namespace {

// A word-addressed model of the 10 MB EDRAM. Store/Load route through the SAME
// offset oracle the GPU uses, so this models the physical layout, not an
// idealized one -- the point is that the layout's quirks (40-col depth swap,
// sample interleave) are part of the contract, not bugs to paper over.
struct EdramRef {
  std::vector<uint32_t> words;
  EdramRef() : words(xenos::kEdramSizeBytes / 4u, 0u) {}

  uint32_t WordIndex(uint32_t x, uint32_t y, MsaaSamples msaa, uint32_t sample,
                     bool is_depth, uint32_t base_tiles,
                     uint32_t pitch_tiles) const {
    uint32_t off = XeEdramSnapshotOffsetBytes(x, y, base_tiles, pitch_tiles,
                                              uint32_t(msaa), is_depth,
                                              /*format_ints_log2=*/0u, sample);
    return off / 4u;  // 32bpp: one word per sample
  }
  void Store(uint32_t x, uint32_t y, MsaaSamples msaa, uint32_t sample,
             bool is_depth, uint32_t base_tiles, uint32_t pitch_tiles,
             uint32_t v) {
    words[WordIndex(x, y, msaa, sample, is_depth, base_tiles, pitch_tiles)] = v;
  }
  uint32_t Load(uint32_t x, uint32_t y, MsaaSamples msaa, uint32_t sample,
                bool is_depth, uint32_t base_tiles, uint32_t pitch_tiles) const {
    return words[WordIndex(x, y, msaa, sample, is_depth, base_tiles,
                           pitch_tiles)];
  }
};

// Deterministic per-(x,y,sample) value; the four samples of a pixel are always
// distinct (xor by an odd constant times sample).
static uint32_t Val(uint32_t x, uint32_t y, uint32_t s) {
  return (x * 2654435761u) ^ (y * 40503u) ^ (s * 0x9E3779B9u) ^ 0xA5A5A5A5u;
}

// 32bpp pitch: XeEdramPitchTiles is 32bpp (80-sample tile), matching fmt_log2=0.
static uint32_t Pitch(uint32_t w, MsaaSamples msaa) {
  return XeEdramPitchTiles(w, uint32_t(msaa));
}

}  // namespace

// Contract 1: a 4x surface holds four INDEPENDENT samples per pixel. Store four
// distinct values per pixel, read all four back -- every one must survive and
// stay distinct. This is the "collapsed sample" guard at the value level: the
// exact failure a naive 4x->1x ownership transfer causes.
TEST_CASE("EDRAM 4x holds four independent samples per pixel",
          "[l2][edram][msaa]") {
  const uint32_t w = 40, h = 20;
  const uint32_t pitch = Pitch(w, MsaaSamples::k4X);
  EdramRef e;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      for (uint32_t s = 0; s < 4; ++s)
        e.Store(x, y, MsaaSamples::k4X, s, false, 0, pitch, Val(x, y, s));
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint32_t got[4];
      for (uint32_t s = 0; s < 4; ++s) {
        got[s] = e.Load(x, y, MsaaSamples::k4X, s, false, 0, pitch);
        REQUIRE(got[s] == Val(x, y, s));
      }
      // The four recovered samples are mutually distinct (no aliasing collapse).
      REQUIRE(got[0] != got[1]);
      REQUIRE(got[0] != got[2]);
      REQUIRE(got[0] != got[3]);
      REQUIRE(got[1] != got[2]);
      REQUIRE(got[1] != got[3]);
      REQUIRE(got[2] != got[3]);
    }
  }
}

// Contract 2: a sample-select resolve reads EXACTLY the chosen sample. Model
// resolve(select S) as Load(...,S); it must equal what was stored at S and
// never any other sample's value.
TEST_CASE("EDRAM sample-select resolve picks the exact sample",
          "[l2][edram][resolve]") {
  const uint32_t w = 32, h = 16;
  const uint32_t pitch = Pitch(w, MsaaSamples::k4X);
  EdramRef e;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      for (uint32_t s = 0; s < 4; ++s)
        e.Store(x, y, MsaaSamples::k4X, s, false, 0, pitch, Val(x, y, s));
  for (uint32_t sel = 0; sel < 4; ++sel) {
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t resolved = e.Load(x, y, MsaaSamples::k4X, sel, false, 0, pitch);
        REQUIRE(resolved == Val(x, y, sel));
        for (uint32_t other = 0; other < 4; ++other) {
          if (other != sel) REQUIRE(resolved != Val(x, y, other));
        }
      }
    }
  }
}

// Contract 3: reinterpreting a color tile AS depth aliases through the exact
// 40-column half-tile swap -- everywhere, not just column 0 (the L0 spot
// check). Prove it at the ADDRESS level over a whole surface: the depth view of
// pixel x reads the byte the color view wrote for the swapped-within-tile x.
TEST_CASE("EDRAM color->depth reinterpret is the 40-col swap surface-wide",
          "[l2][edram][reinterpret]") {
  const uint32_t w = 240, h = 8;  // 3 tiles wide at 1x 32bpp (80 px/tile)
  const uint32_t pitch = Pitch(w, MsaaSamples::k1X);
  EdramRef e;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      e.Store(x, y, MsaaSamples::k1X, 0, /*is_depth=*/false, 0, pitch,
              Val(x, y, 0));
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      // Swap x within its 80-wide tile by +/-40.
      uint32_t tile = x / 80u, col = x % 80u;
      uint32_t col_sw = (col >= 40u) ? (col - 40u) : (col + 40u);
      uint32_t x_sw = tile * 80u + col_sw;
      uint32_t as_depth = e.Load(x, y, MsaaSamples::k1X, 0, /*is_depth=*/true, 0,
                                 pitch);
      REQUIRE(as_depth == Val(x_sw, y, 0));
    }
  }
}

// Contract 4: two surfaces owning DISJOINT tile ranges never leak into each
// other. Surface B is based past all of A's tiles; filling B must leave every
// A sample untouched. Guards the base-tile ownership boundary a resolve or a
// second render target relies on.
TEST_CASE("EDRAM disjoint base-tile ownership does not alias",
          "[l2][edram][ownership]") {
  const uint32_t w = 80, h = 16;  // exactly one tile at 1x 32bpp
  const uint32_t pitch = Pitch(w, MsaaSamples::k1X);
  const uint32_t tiles_a = pitch * ((h + 15u) / 16u);
  const uint32_t base_b = tiles_a + 4u;  // clearly past A
  EdramRef e;
  const uint32_t kA = 0xAAAAAAAAu, kB = 0xBBBBBBBBu;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      e.Store(x, y, MsaaSamples::k1X, 0, false, 0, pitch, kA);
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      e.Store(x, y, MsaaSamples::k1X, 0, false, base_b, pitch, kB);
  // A still reads pure A; B still reads pure B.
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      REQUIRE(e.Load(x, y, MsaaSamples::k1X, 0, false, 0, pitch) == kA);
      REQUIRE(e.Load(x, y, MsaaSamples::k1X, 0, false, base_b, pitch) == kB);
    }
  }
}

// Contract 5: MSAA level changes the physical footprint but a 1x store then a
// 1x load is stable regardless of what a later higher-MSAA surface does at a
// disjoint base -- i.e. resolution/sample count is a property of the surface,
// not global EDRAM state. Store 1x A, then 4x B disjoint, re-read A unchanged.
TEST_CASE("EDRAM 1x surface is stable across a later 4x surface",
          "[l2][edram][msaa][ownership]") {
  const uint32_t w = 80, h = 16;
  const uint32_t pitch1 = Pitch(w, MsaaSamples::k1X);
  const uint32_t tiles_a = pitch1 * ((h + 15u) / 16u);
  const uint32_t base_b = tiles_a + 8u;
  const uint32_t pitch4 = Pitch(w, MsaaSamples::k4X);
  EdramRef e;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      e.Store(x, y, MsaaSamples::k1X, 0, false, 0, pitch1, Val(x, y, 7));
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      for (uint32_t s = 0; s < 4; ++s)
        e.Store(x, y, MsaaSamples::k4X, s, false, base_b, pitch4, Val(x, y, s));
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      REQUIRE(e.Load(x, y, MsaaSamples::k1X, 0, false, 0, pitch1) ==
              Val(x, y, 7));
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
