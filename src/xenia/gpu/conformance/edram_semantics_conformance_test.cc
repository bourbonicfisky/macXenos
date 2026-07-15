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
#include "xenia/gpu/render_target_transfer_class.h"
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

// Contract 1b: cross-MSAA over the SAME tile range is a lossless reinterpret,
// NOT a resample. This is the manifest's biggest finding: in JC2, cross_msaa is
// ~50% of all ownership transitions (20210 of 40291 in one run), overwhelmingly
// full-screen 4x<->1x remaps of the same tiles. An EDRAM tile is 80x16 SAMPLES
// regardless of MSAA -- 4x packs 40x8 pixels x 4 samples, 1x packs 80x16 pixels
// x 1 sample, but BOTH are the same 1280 word-slots. So a 4x->1x->4x round-trip
// preserves every sample: only the pixel<->sample interpretation changes, the
// words never move. The COLLAPSED host cannot do this (one texel per pixel, so
// 4x->1x drops 3 of 4 samples and 1x->4x broadcasts) -- which is exactly why
// PerformCrossMsaaTransfer resamples lossily and why canonical per-sample EDRAM
// turns these 20210 transfers into free, lossless no-ops. This pins the target.
TEST_CASE("EDRAM cross-MSAA over the same tiles round-trips losslessly",
          "[l2][edram][msaa][cross_msaa]") {
  // One tile (base 0, pitch 1). The 4x view is 40x8 px x 4 samples; the 1x view
  // is 80x16 px x 1 sample; both bijectively cover the tile's 1280 word slots.
  EdramRef a;
  for (uint32_t py = 0; py < 8; ++py)
    for (uint32_t px = 0; px < 40; ++px)
      for (uint32_t s = 0; s < 4; ++s)
        a.Store(px, py, MsaaSamples::k4X, s, false, 0, 1, Val(px, py, s));
  // Reinterpret as a 1x view of the SAME tile (the cross_msaa 4x->1x leg) and
  // materialize it into a fresh EDRAM. No resample: each 1x pixel reads one of
  // the physical words the 4x view wrote.
  EdramRef b;
  for (uint32_t py = 0; py < 16; ++py)
    for (uint32_t px = 0; px < 80; ++px)
      b.Store(px, py, MsaaSamples::k1X, 0, false, 0, 1,
              a.Load(px, py, MsaaSamples::k1X, 0, false, 0, 1));
  // Read b back as a 4x view (the 1x->4x leg): every original sample survives.
  uint32_t distinct = 0;
  for (uint32_t py = 0; py < 8; ++py)
    for (uint32_t px = 0; px < 40; ++px)
      for (uint32_t s = 0; s < 4; ++s) {
        REQUIRE(b.Load(px, py, MsaaSamples::k4X, s, false, 0, 1) ==
                Val(px, py, s));
        ++distinct;
      }
  REQUIRE(distinct == 1280u);  // the full tile, every sample accounted for.
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

// Contract 3b: depth->color reinterpret is the SAME 40-column swap as c2d, in
// the other direction. The live PerformDepthToColorTransfer "real bits" leg
// reconstructs the guest depth word with the INVERSE swap (dtx<40 ? +40 : -40,
// metal_render_target_cache.mm:1330) -- symmetric with c2d. This is the crux
// that lets the d2c heuristic (skip / XE_METAL_D2C_REAL / same-frame ping-pong)
// be RETIRED: in canonical per-sample EDRAM the physical word is invariant to
// which key addresses it, so a color surface reclaiming depth-owned tiles reads
// exactly the depth bits at its own (unswapped) offset -- no game-specific
// branch, no lost content. The skip default exists only because the COLLAPSED
// host can't hold those bits; this pins the correct answer the canonical path
// must reproduce.
TEST_CASE("EDRAM depth->color reinterpret is the 40-col swap surface-wide",
          "[l2][edram][reinterpret][d2c]") {
  const uint32_t w = 240, h = 8;  // 3 tiles wide at 1x 32bpp
  const uint32_t pitch = Pitch(w, MsaaSamples::k1X);
  EdramRef e;
  // A depth surface owns these tiles: store guest depth words (d24<<8 | stencil
  // shape) at each pixel's DEPTH offset.
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      e.Store(x, y, MsaaSamples::k1X, 0, /*is_depth=*/true, 0, pitch,
              Val(x, y, 0));
  // A color surface now reclaims the same physical tiles: reading at its own
  // (unswapped) color offset yields the depth word from the swapped-within-tile
  // partner column -- the raw reinterpret, exact.
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint32_t tile = x / 80u, col = x % 80u;
      uint32_t col_sw = (col >= 40u) ? (col - 40u) : (col + 40u);
      uint32_t x_sw = tile * 80u + col_sw;
      uint32_t as_color = e.Load(x, y, MsaaSamples::k1X, 0, /*is_depth=*/false, 0,
                                 pitch);
      REQUIRE(as_color == Val(x_sw, y, 0));
    }
  }
}

// Contract 3c: c2d -> d2c is IDENTITY at the word level (the swap is its own
// inverse). This is why the same-frame ping-pong "real bit transport" is not a
// special case in canonical EDRAM but a free consequence: a color key that lent
// tiles to a depth key and takes them back reads its ORIGINAL words. The Sonic
// DoF packed-vector-map bug (zeroed words decode as max-magnitude vectors) is a
// COLLAPSED-model artifact; the canonical model round-trips losslessly.
TEST_CASE("EDRAM c2d->d2c round-trips a surface's words losslessly",
          "[l2][edram][reinterpret][d2c]") {
  const uint32_t w = 160, h = 16;  // 2 tiles wide at 1x
  const uint32_t pitch = Pitch(w, MsaaSamples::k1X);
  EdramRef a;
  // Original color content owned by a color key.
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      a.Store(x, y, MsaaSamples::k1X, 0, false, 0, pitch, Val(x, y, 3));
  // c2d: a depth key reclaims the tiles; its per-pixel view is the swapped read.
  // Materialize that depth view into a fresh EDRAM as depth-owned words.
  EdramRef b;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      b.Store(x, y, MsaaSamples::k1X, 0, /*is_depth=*/true, 0, pitch,
              a.Load(x, y, MsaaSamples::k1X, 0, /*is_depth=*/true, 0, pitch));
  // d2c: the color key takes the tiles back; reading b as color must reproduce
  // a's original words everywhere (double swap == identity).
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      REQUIRE(b.Load(x, y, MsaaSamples::k1X, 0, false, 0, pitch) ==
              Val(x, y, 3));
}

// Contract 3d: at 4x, d2c preserves ALL FOUR samples per pixel. This is the
// case the collapsed host physically CANNOT represent (one host texel per
// pixel), which is precisely why the live d2c leg falls back to heuristics and
// why the canonical per-sample model is the fix, not another branch.
TEST_CASE("EDRAM c2d->d2c at 4x round-trips every sample losslessly",
          "[l2][edram][reinterpret][d2c][msaa]") {
  const uint32_t w = 40, h = 16;
  const uint32_t pitch = Pitch(w, MsaaSamples::k4X);
  EdramRef a;
  // Original 4x color content: four DISTINCT words per pixel so any sample
  // collapse or cross-sample leak is caught.
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      for (uint32_t s = 0; s < 4; ++s)
        a.Store(x, y, MsaaSamples::k4X, s, false, 0, pitch, Val(x, y, s));
  // c2d: depth key reclaims, materialize its per-sample view (whatever the
  // sample-space swap is) into a fresh EDRAM as depth-owned words.
  EdramRef b;
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      for (uint32_t s = 0; s < 4; ++s)
        b.Store(x, y, MsaaSamples::k4X, s, /*is_depth=*/true, 0, pitch,
                a.Load(x, y, MsaaSamples::k4X, s, /*is_depth=*/true, 0, pitch));
  // d2c: color key takes them back; reading b as color reproduces a exactly,
  // all four samples -- the per-sample losslessness the collapsed host cannot
  // provide (one texel per pixel) and the canonical model delivers for free.
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      for (uint32_t s = 0; s < 4; ++s)
        REQUIRE(b.Load(x, y, MsaaSamples::k4X, s, false, 0, pitch) ==
                Val(x, y, s));
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

// ---------------------------------------------------------------------------
// L3b predictor: the ownership-transition CLASSIFIER.
//
// PerformTransfers (metal_render_target_cache.mm) picks one of seven branches
// per tile-range transfer from a cascade of predicates over the (source, dest)
// render-target pair. That cascade is the murkiest stage of frame assembly, and
// a single-frame trace can't prove it takes the RIGHT branch for a given key
// pair. ClassifyTransfer() is that cascade extracted as a pure function; the
// live code drives its branch from it, so these cases are an EXECUTABLE
// prediction of which transition every key pair produces -- the L2 oracle the
// recorded live manifest (RecordTransition) is diffed against.
namespace {

// Builds a color/depth surface descriptor for the classifier. `container`:
// 0 = raw-word 8888, 1 = R32F reinterpret, 2 = other (e.g. HDR / depth host).
struct Surf {
  bool is_depth = false;
  bool is_64bpp = false;
  MsaaSamples msaa = MsaaSamples::k1X;
  int container = 0;  // 0=8888, 1=r32f, 2=other
};

TransferClassInput MakeInput(const Surf& s, const Surf& d) {
  TransferClassInput in;
  in.src_is_depth = s.is_depth;
  in.dst_is_depth = d.is_depth;
  in.src_is_64bpp = s.is_64bpp;
  in.dst_is_64bpp = d.is_64bpp;
  in.src_msaa = uint32_t(s.msaa);
  in.dst_msaa = uint32_t(d.msaa);
  in.src_is_color8888 = !s.is_depth && s.container == 0;
  in.dst_is_color8888 = !d.is_depth && d.container == 0;
  in.src_is_r32f = !s.is_depth && s.container == 1;
  in.dst_is_r32f = !d.is_depth && d.container == 1;
  // Host formats are equal iff same depth-ness and same container tag (two
  // distinct depth host formats would differ, but the model uses one tag).
  in.host_format_equal =
      (s.is_depth == d.is_depth) && (s.container == d.container);
  return in;
}

}  // namespace

TEST_CASE("Transfer classify: same format + same MSAA is a blit",
          "[l3b][transfer][classify]") {
  Surf a{false, false, MsaaSamples::k4X, 0};
  REQUIRE(ClassifyTransfer(MakeInput(a, a)) == TransferKind::kBlit);
  Surf depth{true, false, MsaaSamples::k1X, 2};
  REQUIRE(ClassifyTransfer(MakeInput(depth, depth)) == TransferKind::kBlit);
}

TEST_CASE("Transfer classify: same format, different MSAA is cross-MSAA",
          "[l3b][transfer][classify]") {
  // 1x -> 4x: x AND y shift both differ.
  Surf lo{false, false, MsaaSamples::k1X, 0};
  Surf hi{false, false, MsaaSamples::k4X, 0};
  REQUIRE(ClassifyTransfer(MakeInput(lo, hi)) == TransferKind::kCrossMsaa);
  REQUIRE(ClassifyTransfer(MakeInput(hi, lo)) == TransferKind::kCrossMsaa);
  // 2x -> 4x differs only in the x shift, still not a blit.
  Surf two{false, false, MsaaSamples::k2X, 0};
  REQUIRE(ClassifyTransfer(MakeInput(two, hi)) == TransferKind::kCrossMsaa);
  // Depth cross-MSAA is also cross-MSAA, not a reinterpret.
  Surf dlo{true, false, MsaaSamples::k1X, 2};
  Surf dhi{true, false, MsaaSamples::k4X, 2};
  REQUIRE(ClassifyTransfer(MakeInput(dlo, dhi)) == TransferKind::kCrossMsaa);
}

TEST_CASE("Transfer classify: 1x vs 2x collapses to a blit (equal shifts)",
          "[l3b][transfer][classify]") {
  // GetMsaaShifts(1x) == GetMsaaShifts(2x)? No: 2x sets the y shift. But 2x vs
  // 4x share the y shift and differ in x. Pin the exact equal-shift boundary:
  // only same-enum pairs (and any pair with identical shift vector) are blits.
  Surf one{false, false, MsaaSamples::k1X, 0};
  Surf two{false, false, MsaaSamples::k2X, 0};
  // 1x (0,0) vs 2x (0,1) differ in y -> NOT a blit.
  REQUIRE(ClassifyTransfer(MakeInput(one, two)) == TransferKind::kCrossMsaa);
}

TEST_CASE("Transfer classify: color->depth reinterpret",
          "[l3b][transfer][classify]") {
  Surf color{false, false, MsaaSamples::k1X, 0};
  Surf depth{true, false, MsaaSamples::k1X, 2};
  REQUIRE(ClassifyTransfer(MakeInput(color, depth)) ==
          TransferKind::kColorToDepth);
  // Only from raw-word 8888: an R32F source into depth is not c2d.
  Surf r32f{false, false, MsaaSamples::k1X, 1};
  REQUIRE(ClassifyTransfer(MakeInput(r32f, depth)) != TransferKind::kColorToDepth);
}

TEST_CASE("Transfer classify: depth->color reinterpret (classified even when "
          "the live default skips the GPU work)",
          "[l3b][transfer][classify]") {
  Surf depth{true, false, MsaaSamples::k1X, 2};
  Surf color{false, false, MsaaSamples::k1X, 0};
  // Classification records the ownership handoff regardless of the runtime
  // XE_METAL_D2C_REAL / ping-pong skip policy.
  REQUIRE(ClassifyTransfer(MakeInput(depth, color)) ==
          TransferKind::kDepthToColor);
  // Depth -> R32F is not the d2c reinterpret path (dest is not raw-word 8888).
  Surf r32f{false, false, MsaaSamples::k1X, 1};
  REQUIRE(ClassifyTransfer(MakeInput(depth, r32f)) != TransferKind::kDepthToColor);
}

TEST_CASE("Transfer classify: 8888<->R32F is a pure reinterpret",
          "[l3b][transfer][classify]") {
  Surf c8888{false, false, MsaaSamples::k1X, 0};
  Surf r32f{false, false, MsaaSamples::k1X, 1};
  REQUIRE(ClassifyTransfer(MakeInput(c8888, r32f)) ==
          TransferKind::kColorReinterpret);
  REQUIRE(ClassifyTransfer(MakeInput(r32f, c8888)) ==
          TransferKind::kColorReinterpret);
}

TEST_CASE("Transfer classify: HDR/other color combos fall to convert",
          "[l3b][transfer][classify]") {
  // HDR (container 2, non-depth) <-> 8888: neither blit (formats differ) nor
  // the specific 8888<->R32F reinterpret -> general convert.
  Surf hdr{false, false, MsaaSamples::k1X, 2};
  Surf c8888{false, false, MsaaSamples::k1X, 0};
  REQUIRE(ClassifyTransfer(MakeInput(hdr, c8888)) ==
          TransferKind::kColorConvert);
  REQUIRE(ClassifyTransfer(MakeInput(c8888, hdr)) ==
          TransferKind::kColorConvert);
  // HDR <-> R32F is also convert, not reinterpret.
  Surf r32f{false, false, MsaaSamples::k1X, 1};
  REQUIRE(ClassifyTransfer(MakeInput(hdr, r32f)) == TransferKind::kColorConvert);
}

TEST_CASE("Transfer classify: any 64bpp owner is skipped",
          "[l3b][transfer][classify]") {
  // 64bpp keys have no color/depth reinterpret handler and are not blittable
  // against a 32bpp partner -> skip (key-local persistence). A 64bpp surface's
  // host format is NOT the raw-word 8888 container, so host_format_equal is
  // false against a 32bpp 8888 (container 2 models "some other host format").
  Surf wide{false, true, MsaaSamples::k1X, 2};
  Surf c8888{false, false, MsaaSamples::k1X, 0};
  REQUIRE(ClassifyTransfer(MakeInput(wide, c8888)) == TransferKind::kSkip);
  REQUIRE(ClassifyTransfer(MakeInput(c8888, wide)) == TransferKind::kSkip);
  // But two identical 64bpp surfaces ARE a blit (same host format + MSAA): the
  // blit branch precedes every 64bpp guard.
  REQUIRE(ClassifyTransfer(MakeInput(wide, wide)) == TransferKind::kBlit);
}

TEST_CASE("Transfer classify: precedence — blit wins over reinterpret",
          "[l3b][transfer][classify]") {
  // A same-format same-MSAA pair that ALSO looks like a color pair must resolve
  // to blit (the first branch), never fall through to convert. Guards against a
  // reordering regression in the live cascade.
  Surf c8888{false, false, MsaaSamples::k2X, 0};
  REQUIRE(ClassifyTransfer(MakeInput(c8888, c8888)) == TransferKind::kBlit);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
