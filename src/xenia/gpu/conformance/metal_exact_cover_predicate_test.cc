/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the Stage-3 exact-cover reader safety invariant.
// The reader in MetalTextureCache::GetTextureSlot serves a resolve snapshot
// GPU-side instead of re-uploading its guest RAM ONLY when the snapshot is
// provably the same content the upload would produce. This suite pins each
// hardening guard: a single incompatible field must fall the fetch back to the
// exact-address RAM path. The reader builds ExactCoverInputs and calls the same
// predicate, so a guard removed in production fails here.

#include "xenia/gpu/metal/metal_exact_cover_predicate.h"

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// A fetch/snapshot pair that IS exact-cover eligible; each negative case below
// perturbs exactly one field so the guard under test is isolated.
ExactCoverInputs Eligible() {
  ExactCoverInputs in{};
  in.rt_is_depth = false;
  in.color_bridge = false;
  in.ram_backed = true;
  in.full_rect = true;
  in.is_signed = false;
  in.snapshot_is_rgba8unorm = true;
  in.fetch_format = xenos::TextureFormat::k_8_8_8_8;
  in.fetch_tiled = true;
  in.fetch_dimension = xenos::DataDimension::k2DOrStacked;
  in.fetch_stacked = false;
  in.fetch_mip_min_level = 0;
  in.fetch_mip_max_level = 0;
  in.fetch_width = 256;
  in.fetch_height = 256;
  in.fetch_pitch_texels = 256;
  in.rt_width = 256;
  in.rt_height = 256;
  in.fetch_endianness = 2;  // k8in32, JC2's resolve dests
  in.wb_endian = 2;
  return in;
}

TEST_CASE("Exact-cover positive baseline is eligible", "[l0][metal][reader]") {
  REQUIRE(ExactCoverSnapshotEligible(Eligible()));
}

TEST_CASE("Exact-cover rejects signed fetch", "[l0][metal][reader]") {
  auto in = Eligible();
  in.is_signed = true;  // SNORM view expected; RGBA8Unorm snapshot is wrong
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects incompatible snapshot format",
          "[l0][metal][reader]") {
  auto in = Eligible();
  in.snapshot_is_rgba8unorm = false;  // aliased/reinterpreted target
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects linear fetch", "[l0][metal][reader]") {
  auto in = Eligible();
  in.fetch_tiled = false;  // untile assumes a tiled base
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects stacked fetch", "[l0][metal][reader]") {
  auto in = Eligible();
  in.fetch_stacked = true;  // stacked addresses memory differently
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects non-2D dimension", "[l0][metal][reader]") {
  auto in = Eligible();
  in.fetch_dimension = xenos::DataDimension::k3D;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
  in.fetch_dimension = xenos::DataDimension::kCube;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects mipped fetch", "[l0][metal][reader]") {
  auto in = Eligible();
  in.fetch_mip_max_level = 3;  // single-level snapshot != mip chain
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
  in = Eligible();
  in.fetch_mip_min_level = 1;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects non-8888 format", "[l0][metal][reader]") {
  auto in = Eligible();
  in.fetch_format = xenos::TextureFormat::k_DXT1;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects partial / offset resolve state",
          "[l0][metal][reader]") {
  auto in = Eligible();
  in.full_rect = false;  // SF4 column resolves: not a full-extent writeback
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects extent, pitch and endian mismatch",
          "[l0][metal][reader]") {
  auto in = Eligible();
  in.fetch_width = 128;  // assembled fetch of a differently-sized snapshot
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
  in = Eligible();
  in.fetch_pitch_texels = 512;  // pitch != width
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
  in = Eligible();
  in.fetch_endianness = 1;  // fetch endian != writeback endian
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

TEST_CASE("Exact-cover rejects depth and bridge/non-RAM states",
          "[l0][metal][reader]") {
  auto in = Eligible();
  in.rt_is_depth = true;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
  in = Eligible();
  in.color_bridge = true;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
  in = Eligible();
  in.ram_backed = false;
  REQUIRE_FALSE(ExactCoverSnapshotEligible(in));
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
