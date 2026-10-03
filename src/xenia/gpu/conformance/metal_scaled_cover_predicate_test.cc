/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Scaled-cover serving (metal_scaled_cover_predicate.h) and its place in the
// one service classification (metal_resolved_fetch_service.h): it applies only
// in a scaled session, only to a fetch that would otherwise read RAM, and only
// when the fetch samples exactly the snapshot's written surface.

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_resolved_fetch_service.h"

namespace xe {
namespace gpu {
namespace test {

using metal::ClassifyResolvedFetch;
using metal::ResolvedFetchService;
using metal::ResolvedFetchServiceInputs;
using metal::ScaledCoverInputs;
using metal::ScaledCoverSnapshotEligible;

namespace {

// JC2's composite read of its 1280x720 scene: snapshot rows written by two
// vertical halves, RGBA8, same endian.
ScaledCoverInputs EligibleScaledCover() {
  ScaledCoverInputs in;
  in.scaled_session = true;
  in.rt_is_depth = false;
  in.ram_backed = true;
  in.is_signed = false;
  in.snapshot_format_matches_fetch = true;
  in.fetch_tiled = true;
  in.fetch_dimension = xenos::DataDimension::k2DOrStacked;
  in.fetch_stacked = false;
  in.fetch_width = 1280;
  in.fetch_height = 720;
  in.fetch_pitch_texels = 1280;
  in.snapshot_width = 1280;
  in.snapshot_height = 720;
  in.rows_written = 720;
  in.fetch_endianness = 2;
  in.wb_endian = 2;
  return in;
}

// A RAM-backed color base that is neither exact cover nor bridge.
ResolvedFetchServiceInputs RamContentBase() {
  ResolvedFetchServiceInputs in;
  in.has_resolved_target = true;
  in.exact.ram_backed = true;
  in.exact.fetch_format = xenos::TextureFormat::k_8_8_8_8;
  in.exact.rt_width = 1280;
  in.exact.rt_height = 384;  // the last half's registration
  in.exact.fetch_width = 1280;
  in.exact.fetch_height = 720;
  return in;
}

}  // namespace

TEST_CASE("Scaled cover: the composite read of a fully written snapshot",
          "[metal][draw_resolution_scale]") {
  CHECK(ScaledCoverSnapshotEligible(EligibleScaledCover()));
}

TEST_CASE("Scaled cover: each condition disqualifies on its own",
          "[metal][draw_resolution_scale]") {
  auto check_off = [](auto mutate) {
    ScaledCoverInputs in = EligibleScaledCover();
    mutate(in);
    CHECK_FALSE(ScaledCoverSnapshotEligible(in));
  };
  check_off([](ScaledCoverInputs& in) { in.scaled_session = false; });
  check_off([](ScaledCoverInputs& in) { in.rt_is_depth = true; });
  check_off([](ScaledCoverInputs& in) { in.ram_backed = false; });
  check_off([](ScaledCoverInputs& in) { in.is_signed = true; });
  check_off(
      [](ScaledCoverInputs& in) { in.snapshot_format_matches_fetch = false; });
  check_off([](ScaledCoverInputs& in) { in.fetch_tiled = false; });
  check_off([](ScaledCoverInputs& in) {
    in.fetch_dimension = xenos::DataDimension::k3D;
  });
  check_off([](ScaledCoverInputs& in) { in.fetch_stacked = true; });
  check_off([](ScaledCoverInputs& in) { in.fetch_mip_max_level = 1; });
  check_off([](ScaledCoverInputs& in) { in.fetch_width = 640; });
  check_off([](ScaledCoverInputs& in) { in.fetch_height = 360; });
  check_off([](ScaledCoverInputs& in) { in.fetch_pitch_texels = 2048; });
  // Only the first half's rows are in the snapshot yet.
  check_off([](ScaledCoverInputs& in) { in.rows_written = 384; });
  check_off([](ScaledCoverInputs& in) { in.fetch_endianness = 0; });
}

TEST_CASE("Scaled cover: classified only where RAM would otherwise serve",
          "[metal][draw_resolution_scale]") {
  // A RAM-content fetch becomes scaled cover in a scaled session.
  ResolvedFetchServiceInputs in = RamContentBase();
  in.scaled = EligibleScaledCover();
  CHECK(ClassifyResolvedFetch(in) ==
        ResolvedFetchService::kScaledCoverSnapshot);

  // Never at 1x: the same fetch keeps the RAM path.
  ResolvedFetchServiceInputs one_x = RamContentBase();
  CHECK(ClassifyResolvedFetch(one_x) == ResolvedFetchService::kRamContent);

  // The bridge keeps precedence (a depth snapshot, for example).
  ResolvedFetchServiceInputs depth = RamContentBase();
  depth.exact.rt_is_depth = true;
  depth.scaled = EligibleScaledCover();
  CHECK(ClassifyResolvedFetch(depth) == ResolvedFetchService::kBridgeSnapshot);

  // No registration: the ordinary guest-memory path, whatever the inputs.
  ResolvedFetchServiceInputs none;
  none.scaled = EligibleScaledCover();
  CHECK(ClassifyResolvedFetch(none) ==
        ResolvedFetchService::kNotResolveBacked);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
