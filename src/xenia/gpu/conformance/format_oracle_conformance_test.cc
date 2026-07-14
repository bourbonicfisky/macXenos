/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 of the Metal conformance ladder: CPU format/address helpers checked
// bit-exact against expected values. These are the oracles every higher level
// (Metal kernels, synthetic PM4, reduced traces) is validated against, so a
// break here must fail in milliseconds — no Metal device, no trace, no PNG.
//
// Deliberately links only the pure-math sources (xenos.cc + header-only
// addressers), NOT the monolithic xenia-gpu lib, so this suite stays a
// second-scale isolated contract check. See the metal-conformance-ladder plan.

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/texture_address.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

using xenos::Endian;

// --- Guest->host endian swap (the k8in16 / k8in32 / k16in32 contract) --------
// Every texture/vertex/index upload runs through GpuSwap; a wrong swap is the
// most common "garbled bytes" class. The contract: swap is an involution, and
// each mode permutes the exact documented byte lanes.

TEST_CASE("GpuSwap is an involution", "[l0][endian]") {
  const uint32_t samples[] = {0x00000000u, 0xFFFFFFFFu, 0x01020304u,
                              0xDEADBEEFu, 0x12345678u, 0x0000FF00u};
  for (Endian e : {Endian::k8in16, Endian::k8in32, Endian::k16in32}) {
    for (uint32_t v : samples) {
      // Swapping twice returns the original for every self-inverse mode.
      REQUIRE(xenos::GpuSwap(xenos::GpuSwap(v, e), e) == v);
    }
  }
}

TEST_CASE("GpuSwap permutes the documented byte lanes", "[l0][endian]") {
  // 0x01020304 in memory is bytes 01 02 03 04.
  const uint32_t v = 0x01020304u;
  REQUIRE(xenos::GpuSwap(v, Endian::kNone) == 0x01020304u);
  REQUIRE(xenos::GpuSwap(v, Endian::k8in32) == 0x04030201u);   // full reverse
  REQUIRE(xenos::GpuSwap(v, Endian::k16in32) == 0x03040102u);  // halfword swap
  // k8in16: swap the two bytes within each 16-bit half.
  REQUIRE(xenos::GpuSwap(v, Endian::k8in16) == 0x02010403u);
}

// --- 20e4 guest depth float (Float32To20e4 / Float20e4To32) -------------------
// The EDRAM float24 depth encode/decode. Round-trip of any value the guest can
// represent must be exact; endpoints and the round-to-even flag are pinned.

TEST_CASE("20e4 depth round-trips every representable code", "[l0][float24]") {
  // Sweep a spread of the 24-bit code space; decode->encode must be identity
  // (encode->decode isn't, since 32-bit floats have more precision than 20e4).
  for (uint32_t code = 0; code < (1u << 24); code += 1237u) {
    float f = xenos::Float20e4To32(code);
    uint32_t re = xenos::Float32To20e4(f, /*round_to_nearest_even=*/true);
    REQUIRE(re == code);
  }
}

TEST_CASE("20e4 depth endpoints", "[l0][float24]") {
  REQUIRE(xenos::Float20e4To32(0u) == 0.0f);
  // Encoding 0.0 is code 0 under both rounding modes.
  REQUIRE(xenos::Float32To20e4(0.0f, true) == 0u);
  REQUIRE(xenos::Float32To20e4(0.0f, false) == 0u);
  // Monotonic: larger code -> larger (or equal) decoded value.
  float prev = -1.0f;
  for (uint32_t code = 0; code < (1u << 24); code += 9973u) {
    float f = xenos::Float20e4To32(code);
    REQUIRE(f >= prev);
    prev = f;
  }
}

// --- unorm24 depth (UNorm24To32) ---------------------------------------------
// The non-float24 depth decode. The exact 2^24 divide matters: 0xC00000 (a very
// common 2_10_10_10 clear) must decode precisely, not via 1/16777215.

TEST_CASE("unorm24 endpoints are exact and 0xC00000 rounds up", "[l0][unorm24]") {
  // Endpoints must be EXACT: a depth clear to max must decode to precisely 1.0.
  REQUIRE(xenos::UNorm24To32(0u) == 0.0f);
  REQUIRE(xenos::UNorm24To32(0xFFFFFFu) == 1.0f);
  // The impl adds (n24>>23) before the /2^24 divide, so 0xC00000 decodes just
  // ABOVE 0.75 by exactly one code in 2^24 — NOT the naive 0.75 (that naive
  // divide is the bug the impl comment warns about). Pin the real value.
  REQUIRE(xenos::UNorm24To32(0xC00000u) ==
          float(0xC00000u + (0xC00000u >> 23)) * (1.0f / float(1u << 24)));
  REQUIRE(xenos::UNorm24To32(0xC00000u) > 0.75f);
  REQUIRE(xenos::UNorm24To32(0xC00000u) - 0.75f < 1.0e-6f);
}

// --- 7e3 guest color float (Float7e3To32) ------------------------------------
TEST_CASE("7e3 color endpoints and monotonicity", "[l0][float7e3]") {
  REQUIRE(xenos::Float7e3To32(0u) == 0.0f);
  float prev = -1.0f;
  for (uint32_t code = 0; code < (1u << 10); ++code) {
    float f = xenos::Float7e3To32(code);
    REQUIRE(f >= prev);
    prev = f;
  }
}

// --- Tiled 2D texture addressing (texture_address::Tiled2D) -------------------
// The single most load-bearing addresser: texture untile, EDRAM tile walk, and
// resolve all lean on it. The contract that matters for correctness is a
// BIJECTION over the storage extent — no two (x,y) blocks may alias one byte
// offset, or textures tear (the "noise block" glyph class). Verify injectivity
// and full coverage over one macro-tile-aligned region.

TEST_CASE("Tiled2D is a bijection over its storage extent", "[l0][address]") {
  // 32x32 blocks (one macro tile), 4 bytes/block (log2 = 2): a common 8888 case.
  const uint32_t pitch_aligned = 32u;
  const unsigned int bpb_log2 = 2u;
  const int32_t w = 32, h = 32;
  std::vector<uint8_t> hit(size_t(w) * h * (1u << bpb_log2), 0u);
  uint32_t covered = 0;
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      int32_t off = texture_address::Tiled2D(x, y, pitch_aligned, bpb_log2);
      REQUIRE(off >= 0);
      REQUIRE(size_t(off) + (1u << bpb_log2) <= hit.size());
      // No byte in this block may already be claimed by another (x,y).
      for (uint32_t b = 0; b < (1u << bpb_log2); ++b) {
        REQUIRE(hit[size_t(off) + b] == 0u);
        hit[size_t(off) + b] = 1u;
      }
      ++covered;
    }
  }
  REQUIRE(covered == uint32_t(w * h));
  // Every byte of the extent is covered exactly once (full surjection).
  for (uint8_t b : hit) {
    REQUIRE(b == 1u);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
