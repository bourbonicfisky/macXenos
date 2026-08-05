/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 for the cache-hit correctness invariant
// (adopt-xenios-shader-artifact-caching task 2.1 item 4):
//
//   a hit must equal a fresh production byte for byte, or it must miss.
//
// The predicate itself is trivial; what these cases pin is that it FAILS on
// every way an artifact can be wrong, including the two that a looser check
// would wave through: a one-bit difference in the middle of a large blob (the
// dxilconv-corruption shape — same length, same prefix, same suffix), and a
// hit that could not be verified at all because the fresh production failed.
// The second is the one worth stating out loud: "we could not check" must
// never resolve to "it checked out".
//
// The same predicate is exercised over REAL dxilconv artifacts in
// metal_dxil_determinism_test.mm; this file keeps the decision table itself in
// the fast device-free binary.

#include <cstdint>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_artifact_hit_verify.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::MetalArtifactHitStats;
using metal::MetalArtifactHitVerdict;
using metal::MetalArtifactHitIsServable;
using metal::VerifyMetalArtifactHit;

// A blob big enough that a single flipped byte is a genuinely small signal,
// with a non-repeating pattern so a truncation or a shift is not mistaken for
// equality.
std::vector<uint8_t> Blob(size_t size, uint8_t seed = 0) {
  std::vector<uint8_t> out(size);
  for (size_t i = 0; i < size; ++i) {
    out[i] = static_cast<uint8_t>((i * 31u + seed * 17u + 7u) & 0xFF);
  }
  return out;
}

}  // namespace

TEST_CASE("An identical hit serves", "[cache_identity][hit_verify]") {
  const std::vector<uint8_t> fresh = Blob(4096, 3);
  const std::vector<uint8_t> cached = fresh;
  const MetalArtifactHitVerdict verdict = VerifyMetalArtifactHit(cached, fresh);
  CHECK(verdict == MetalArtifactHitVerdict::kServe);
  CHECK(MetalArtifactHitIsServable(verdict));
}

TEST_CASE("A one-byte difference misses, wherever it is",
          "[cache_identity][hit_verify]") {
  // The corruption this invariant exists for did not change the artifact's
  // length. A check that compared sizes, or hashes of the first N bytes, would
  // have served it.
  const std::vector<uint8_t> fresh = Blob(4096, 3);
  for (size_t offset : {size_t(0), size_t(1), size_t(2047), size_t(4094),
                        size_t(4095)}) {
    INFO("flipped byte at offset " << offset);
    std::vector<uint8_t> cached = fresh;
    cached[offset] ^= 0x01;
    const MetalArtifactHitVerdict verdict =
        VerifyMetalArtifactHit(cached, fresh);
    CHECK(verdict == MetalArtifactHitVerdict::kMissByteMismatch);
    CHECK_FALSE(MetalArtifactHitIsServable(verdict));
    CHECK(cached.size() == fresh.size());  // the length alone proves nothing
  }
}

TEST_CASE("A length difference misses", "[cache_identity][hit_verify]") {
  const std::vector<uint8_t> fresh = Blob(4096, 3);

  SECTION("truncated cached entry") {
    std::vector<uint8_t> cached(fresh.begin(), fresh.end() - 1);
    CHECK(VerifyMetalArtifactHit(cached, fresh) ==
          MetalArtifactHitVerdict::kMissSizeMismatch);
  }
  SECTION("cached entry with trailing bytes") {
    std::vector<uint8_t> cached = fresh;
    cached.push_back(0x00);
    CHECK(VerifyMetalArtifactHit(cached, fresh) ==
          MetalArtifactHitVerdict::kMissSizeMismatch);
  }
  SECTION("a prefix of the fresh artifact is still a miss") {
    // Guards against any future "compare what we have" relaxation.
    std::vector<uint8_t> cached(fresh.begin(), fresh.begin() + 16);
    CHECK_FALSE(MetalArtifactHitIsServable(
        VerifyMetalArtifactHit(cached, fresh)));
  }
}

TEST_CASE("An empty cached entry is a cold miss, not a match",
          "[cache_identity][hit_verify]") {
  const std::vector<uint8_t> fresh = Blob(64, 1);
  const std::vector<uint8_t> empty;
  CHECK(VerifyMetalArtifactHit(empty, fresh) ==
        MetalArtifactHitVerdict::kMissEmpty);
  // Both empty must NOT be "equal, therefore serve".
  CHECK(VerifyMetalArtifactHit(empty, empty) ==
        MetalArtifactHitVerdict::kMissEmpty);
}

TEST_CASE("An unverifiable hit does not serve", "[cache_identity][hit_verify]") {
  // Fresh production failed (dxilconv unavailable, translation error). There
  // is nothing to compare against, so the cached bytes are unproven. Serving
  // them would make the verification pass silently in exactly the situation
  // where it is doing no work.
  const std::vector<uint8_t> cached = Blob(256, 5);
  const std::vector<uint8_t> no_fresh;
  const MetalArtifactHitVerdict verdict =
      VerifyMetalArtifactHit(cached, no_fresh);
  CHECK(verdict == MetalArtifactHitVerdict::kMissUnverifiable);
  CHECK_FALSE(MetalArtifactHitIsServable(verdict));
}

TEST_CASE("Only kServe serves", "[cache_identity][hit_verify]") {
  // Enumerated rather than derived, so a verdict added later without a
  // decision here fails this case instead of defaulting into service.
  CHECK(MetalArtifactHitIsServable(MetalArtifactHitVerdict::kServe));
  CHECK_FALSE(MetalArtifactHitIsServable(MetalArtifactHitVerdict::kMissEmpty));
  CHECK_FALSE(
      MetalArtifactHitIsServable(MetalArtifactHitVerdict::kMissUnverifiable));
  CHECK_FALSE(
      MetalArtifactHitIsServable(MetalArtifactHitVerdict::kMissSizeMismatch));
  CHECK_FALSE(
      MetalArtifactHitIsServable(MetalArtifactHitVerdict::kMissByteMismatch));
}

TEST_CASE("Verification telemetry counts mismatches separately",
          "[cache_identity][hit_verify]") {
  // 2.2/2.3 promote on these numbers. `mismatches` is a correctness incident
  // counter, not a hit-rate one: it must stay 0, so it cannot be folded into a
  // generic "miss" tally that a cold cache also increments.
  MetalArtifactHitStats stats;
  const std::vector<uint8_t> fresh = Blob(512, 9);
  std::vector<uint8_t> corrupt = fresh;
  corrupt[100] ^= 0x80;

  stats.Record(VerifyMetalArtifactHit(fresh, fresh));
  stats.Record(VerifyMetalArtifactHit(fresh, fresh));
  stats.Record(VerifyMetalArtifactHit(corrupt, fresh));

  CHECK(stats.verified.load() == 3);
  CHECK(stats.served.load() == 2);
  CHECK(stats.mismatches.load() == 1);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
