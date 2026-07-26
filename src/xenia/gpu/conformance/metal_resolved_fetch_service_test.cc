/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Level 0 (CPU, device-free): the CURRENT resolved-fetch service classification,
// frozen as an executable contract.
//
// OpenSpec correct-metal-resolve-bridge-authority task 3.1. Group 2 selected
// outcome A (closed negative): no production authority edit was made. This
// suite therefore asserts EXISTING behaviour only and encodes NO proposed
// change. Its purpose is to make the group-2 structural argument executable
// instead of leaving it at code-path inspection (evidence level 4).
//
// The rows call the production classifier ClassifyResolvedFetch directly. No
// rule is re-modelled here: the expected column is a committed literal golden
// table captured once from the production classifier at this base, so a
// production semantic change fails a row rather than being tracked silently by
// a test-local formula.
//
// Enumerated cross-product (2^7 = 128 rows), one bit of the row index each:
//
//   bit 0  registration     no registered resolve target / registered snapshot
//   bit 1  depth            colour render target / depth render target
//   bit 2  ram_backed       no writeback has run / a writeback has run
//   bit 3  full_rect        newest writeback partial / full extent
//   bit 4  colour bridge    bridge off / on
//   bit 5  covered R32F     off / on
//   bit 6  exact shape      fetch shape matches the snapshot / mismatches
//
// Every OTHER ExactCoverInputs field is held at the exact-cover-eligible
// baseline (unsigned, RGBA8Unorm snapshot, tiled 2D single-mip k_8_8_8_8,
// matching endian), so each row isolates the seven enumerated factors. The
// remaining ExactCoverInputs guards are covered separately and exhaustively by
// metal_exact_cover_predicate_test.cc; this suite does not duplicate them.

#include <string>

#include "xenia/gpu/metal/metal_resolved_fetch_service.h"

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace metal {
namespace {

constexpr int kRowCount = 128;

// Row-index bit positions for the seven enumerated factors.
enum Factor : int {
  kFactorRegistration = 0,
  kFactorDepth = 1,
  kFactorRamBacked = 2,
  kFactorFullRect = 3,
  kFactorColorBridge = 4,
  kFactorCoveredR32F = 5,
  kFactorShapeMismatch = 6,
  kFactorCount = 7,
};

const char* const kFactorNames[kFactorCount] = {
    "registration", "depth",        "ram_backed",    "full_rect",
    "color_bridge", "covered_r32f", "shape_mismatch"};

bool FactorOf(int row, int factor) { return (row & (1 << factor)) != 0; }

// Builds one row of the cross-product. Only the seven enumerated factors vary;
// every other field stays at the eligible baseline.
ResolvedFetchServiceInputs Row(int row) {
  const bool shape_mismatch = FactorOf(row, kFactorShapeMismatch);

  ResolvedFetchServiceInputs in{};
  in.has_resolved_target = FactorOf(row, kFactorRegistration);
  in.covered_r32f = FactorOf(row, kFactorCoveredR32F);

  in.exact.rt_is_depth = FactorOf(row, kFactorDepth);
  in.exact.ram_backed = FactorOf(row, kFactorRamBacked);
  in.exact.full_rect = FactorOf(row, kFactorFullRect);
  in.exact.color_bridge = FactorOf(row, kFactorColorBridge);

  // Eligible baseline for every unrelated ExactCoverInputs field.
  in.exact.is_signed = false;
  in.exact.snapshot_is_rgba8unorm = true;
  in.exact.fetch_format = xenos::TextureFormat::k_8_8_8_8;
  in.exact.fetch_tiled = true;
  in.exact.fetch_dimension = xenos::DataDimension::k2DOrStacked;
  in.exact.fetch_stacked = false;
  in.exact.fetch_mip_min_level = 0;
  in.exact.fetch_mip_max_level = 0;
  in.exact.rt_width = 256;
  in.exact.rt_height = 256;
  // The shape factor is one knob: the fetch either covers the registered
  // snapshot exactly, or is a differently shaped fetch at the same base.
  in.exact.fetch_width = shape_mismatch ? 128u : 256u;
  in.exact.fetch_height = shape_mismatch ? 128u : 256u;
  in.exact.fetch_pitch_texels = shape_mismatch ? 128u : 256u;
  in.exact.fetch_endianness = 2;  // k8in32, JC2's resolve destinations
  in.exact.wb_endian = 2;
  return in;
}

// COMMITTED GOLDEN TABLE. Captured once from the production classifier at base
// 2e1dee1c0 (production source tree 797c00ffbad4937fba0e05e1bf9e846d51d1847a).
// This is recorded output, NOT a formula that re-derives the classifier.
constexpr ResolvedFetchService kGolden[kRowCount] = {
    /*   0 */ ResolvedFetchService::kNotResolveBacked,
    /*   1 */ ResolvedFetchService::kBridgeSnapshot,
    /*   2 */ ResolvedFetchService::kNotResolveBacked,
    /*   3 */ ResolvedFetchService::kBridgeSnapshot,
    /*   4 */ ResolvedFetchService::kNotResolveBacked,
    /*   5 */ ResolvedFetchService::kRamContent,
    /*   6 */ ResolvedFetchService::kNotResolveBacked,
    /*   7 */ ResolvedFetchService::kBridgeSnapshot,
    /*   8 */ ResolvedFetchService::kNotResolveBacked,
    /*   9 */ ResolvedFetchService::kBridgeSnapshot,
    /*  10 */ ResolvedFetchService::kNotResolveBacked,
    /*  11 */ ResolvedFetchService::kBridgeSnapshot,
    /*  12 */ ResolvedFetchService::kNotResolveBacked,
    /*  13 */ ResolvedFetchService::kExactCoverSnapshot,
    /*  14 */ ResolvedFetchService::kNotResolveBacked,
    /*  15 */ ResolvedFetchService::kBridgeSnapshot,
    /*  16 */ ResolvedFetchService::kNotResolveBacked,
    /*  17 */ ResolvedFetchService::kBridgeSnapshot,
    /*  18 */ ResolvedFetchService::kNotResolveBacked,
    /*  19 */ ResolvedFetchService::kBridgeSnapshot,
    /*  20 */ ResolvedFetchService::kNotResolveBacked,
    /*  21 */ ResolvedFetchService::kBridgeSnapshot,
    /*  22 */ ResolvedFetchService::kNotResolveBacked,
    /*  23 */ ResolvedFetchService::kBridgeSnapshot,
    /*  24 */ ResolvedFetchService::kNotResolveBacked,
    /*  25 */ ResolvedFetchService::kBridgeSnapshot,
    /*  26 */ ResolvedFetchService::kNotResolveBacked,
    /*  27 */ ResolvedFetchService::kBridgeSnapshot,
    /*  28 */ ResolvedFetchService::kNotResolveBacked,
    /*  29 */ ResolvedFetchService::kBridgeSnapshot,
    /*  30 */ ResolvedFetchService::kNotResolveBacked,
    /*  31 */ ResolvedFetchService::kBridgeSnapshot,
    /*  32 */ ResolvedFetchService::kNotResolveBacked,
    /*  33 */ ResolvedFetchService::kBridgeSnapshot,
    /*  34 */ ResolvedFetchService::kNotResolveBacked,
    /*  35 */ ResolvedFetchService::kBridgeSnapshot,
    /*  36 */ ResolvedFetchService::kNotResolveBacked,
    /*  37 */ ResolvedFetchService::kBridgeSnapshot,
    /*  38 */ ResolvedFetchService::kNotResolveBacked,
    /*  39 */ ResolvedFetchService::kBridgeSnapshot,
    /*  40 */ ResolvedFetchService::kNotResolveBacked,
    /*  41 */ ResolvedFetchService::kBridgeSnapshot,
    /*  42 */ ResolvedFetchService::kNotResolveBacked,
    /*  43 */ ResolvedFetchService::kBridgeSnapshot,
    /*  44 */ ResolvedFetchService::kNotResolveBacked,
    /*  45 */ ResolvedFetchService::kExactCoverSnapshot,
    /*  46 */ ResolvedFetchService::kNotResolveBacked,
    /*  47 */ ResolvedFetchService::kBridgeSnapshot,
    /*  48 */ ResolvedFetchService::kNotResolveBacked,
    /*  49 */ ResolvedFetchService::kBridgeSnapshot,
    /*  50 */ ResolvedFetchService::kNotResolveBacked,
    /*  51 */ ResolvedFetchService::kBridgeSnapshot,
    /*  52 */ ResolvedFetchService::kNotResolveBacked,
    /*  53 */ ResolvedFetchService::kBridgeSnapshot,
    /*  54 */ ResolvedFetchService::kNotResolveBacked,
    /*  55 */ ResolvedFetchService::kBridgeSnapshot,
    /*  56 */ ResolvedFetchService::kNotResolveBacked,
    /*  57 */ ResolvedFetchService::kBridgeSnapshot,
    /*  58 */ ResolvedFetchService::kNotResolveBacked,
    /*  59 */ ResolvedFetchService::kBridgeSnapshot,
    /*  60 */ ResolvedFetchService::kNotResolveBacked,
    /*  61 */ ResolvedFetchService::kBridgeSnapshot,
    /*  62 */ ResolvedFetchService::kNotResolveBacked,
    /*  63 */ ResolvedFetchService::kBridgeSnapshot,
    /*  64 */ ResolvedFetchService::kNotResolveBacked,
    /*  65 */ ResolvedFetchService::kBridgeSnapshot,
    /*  66 */ ResolvedFetchService::kNotResolveBacked,
    /*  67 */ ResolvedFetchService::kBridgeSnapshot,
    /*  68 */ ResolvedFetchService::kNotResolveBacked,
    /*  69 */ ResolvedFetchService::kRamContent,
    /*  70 */ ResolvedFetchService::kNotResolveBacked,
    /*  71 */ ResolvedFetchService::kBridgeSnapshot,
    /*  72 */ ResolvedFetchService::kNotResolveBacked,
    /*  73 */ ResolvedFetchService::kBridgeSnapshot,
    /*  74 */ ResolvedFetchService::kNotResolveBacked,
    /*  75 */ ResolvedFetchService::kBridgeSnapshot,
    /*  76 */ ResolvedFetchService::kNotResolveBacked,
    /*  77 */ ResolvedFetchService::kRamContent,
    /*  78 */ ResolvedFetchService::kNotResolveBacked,
    /*  79 */ ResolvedFetchService::kBridgeSnapshot,
    /*  80 */ ResolvedFetchService::kNotResolveBacked,
    /*  81 */ ResolvedFetchService::kBridgeSnapshot,
    /*  82 */ ResolvedFetchService::kNotResolveBacked,
    /*  83 */ ResolvedFetchService::kBridgeSnapshot,
    /*  84 */ ResolvedFetchService::kNotResolveBacked,
    /*  85 */ ResolvedFetchService::kBridgeSnapshot,
    /*  86 */ ResolvedFetchService::kNotResolveBacked,
    /*  87 */ ResolvedFetchService::kBridgeSnapshot,
    /*  88 */ ResolvedFetchService::kNotResolveBacked,
    /*  89 */ ResolvedFetchService::kBridgeSnapshot,
    /*  90 */ ResolvedFetchService::kNotResolveBacked,
    /*  91 */ ResolvedFetchService::kBridgeSnapshot,
    /*  92 */ ResolvedFetchService::kNotResolveBacked,
    /*  93 */ ResolvedFetchService::kBridgeSnapshot,
    /*  94 */ ResolvedFetchService::kNotResolveBacked,
    /*  95 */ ResolvedFetchService::kBridgeSnapshot,
    /*  96 */ ResolvedFetchService::kNotResolveBacked,
    /*  97 */ ResolvedFetchService::kBridgeSnapshot,
    /*  98 */ ResolvedFetchService::kNotResolveBacked,
    /*  99 */ ResolvedFetchService::kBridgeSnapshot,
    /* 100 */ ResolvedFetchService::kNotResolveBacked,
    /* 101 */ ResolvedFetchService::kBridgeSnapshot,
    /* 102 */ ResolvedFetchService::kNotResolveBacked,
    /* 103 */ ResolvedFetchService::kBridgeSnapshot,
    /* 104 */ ResolvedFetchService::kNotResolveBacked,
    /* 105 */ ResolvedFetchService::kBridgeSnapshot,
    /* 106 */ ResolvedFetchService::kNotResolveBacked,
    /* 107 */ ResolvedFetchService::kBridgeSnapshot,
    /* 108 */ ResolvedFetchService::kNotResolveBacked,
    /* 109 */ ResolvedFetchService::kBridgeSnapshot,
    /* 110 */ ResolvedFetchService::kNotResolveBacked,
    /* 111 */ ResolvedFetchService::kBridgeSnapshot,
    /* 112 */ ResolvedFetchService::kNotResolveBacked,
    /* 113 */ ResolvedFetchService::kBridgeSnapshot,
    /* 114 */ ResolvedFetchService::kNotResolveBacked,
    /* 115 */ ResolvedFetchService::kBridgeSnapshot,
    /* 116 */ ResolvedFetchService::kNotResolveBacked,
    /* 117 */ ResolvedFetchService::kBridgeSnapshot,
    /* 118 */ ResolvedFetchService::kNotResolveBacked,
    /* 119 */ ResolvedFetchService::kBridgeSnapshot,
    /* 120 */ ResolvedFetchService::kNotResolveBacked,
    /* 121 */ ResolvedFetchService::kBridgeSnapshot,
    /* 122 */ ResolvedFetchService::kNotResolveBacked,
    /* 123 */ ResolvedFetchService::kBridgeSnapshot,
    /* 124 */ ResolvedFetchService::kNotResolveBacked,
    /* 125 */ ResolvedFetchService::kBridgeSnapshot,
    /* 126 */ ResolvedFetchService::kNotResolveBacked,
    /* 127 */ ResolvedFetchService::kBridgeSnapshot,
};

static_assert(sizeof(kGolden) / sizeof(kGolden[0]) == kRowCount,
              "the golden table must hold exactly the 128 enumerated rows");

const char* ServiceName(ResolvedFetchService service) {
  switch (service) {
    case ResolvedFetchService::kNotResolveBacked:
      return "kNotResolveBacked";
    case ResolvedFetchService::kBridgeSnapshot:
      return "kBridgeSnapshot";
    case ResolvedFetchService::kExactCoverSnapshot:
      return "kExactCoverSnapshot";
    case ResolvedFetchService::kRamContent:
      return "kRamContent";
  }
  return "<unknown ResolvedFetchService>";
}

// Every diagnostic in this suite names the row index and all seven factors, so
// a failure identifies the exact enumerated combination without re-deriving it.
std::string DescribeRow(int row) {
  std::string text = "row " + std::to_string(row) + " [";
  for (int factor = 0; factor < kFactorCount; ++factor) {
    if (factor) {
      text += ' ';
    }
    text += kFactorNames[factor];
    text += '=';
    text += FactorOf(row, factor) ? '1' : '0';
  }
  text += ']';
  return text;
}

TEST_CASE("Resolved-fetch service: 128-row cross-product matches the golden table",
          "[l0][metal][reader][service]") {
  int rows_checked = 0;
  int counts[4] = {0, 0, 0, 0};

  for (int row = 0; row < kRowCount; ++row) {
    // The production classifier, called directly. No test-local re-model.
    const ResolvedFetchService actual = ClassifyResolvedFetch(Row(row));

    INFO(DescribeRow(row) << " expected=" << ServiceName(kGolden[row])
                          << " actual=" << ServiceName(actual));
    REQUIRE(actual == kGolden[row]);

    counts[static_cast<int>(actual)]++;
    ++rows_checked;
  }

  // Omission cannot pass: the row count and the literal class histogram are
  // both asserted, so a truncated or skipped enumeration fails here.
  REQUIRE(rows_checked == 128);
  INFO("nrb=" << counts[0] << " bridge=" << counts[1] << " exact=" << counts[2]
              << " ram=" << counts[3]);
  REQUIRE(counts[static_cast<int>(ResolvedFetchService::kNotResolveBacked)] ==
          64);
  REQUIRE(counts[static_cast<int>(ResolvedFetchService::kBridgeSnapshot)] == 59);
  REQUIRE(counts[static_cast<int>(ResolvedFetchService::kExactCoverSnapshot)] ==
          2);
  REQUIRE(counts[static_cast<int>(ResolvedFetchService::kRamContent)] == 3);
  REQUIRE(counts[0] + counts[1] + counts[2] + counts[3] == 128);
}

TEST_CASE("Resolved-fetch service: every enumerated factor is load-bearing",
          "[l0][metal][reader][service]") {
  // Perturbation check. For each factor, walk the 64 paired rows that differ in
  // that bit alone and count the pairs whose classification differs. A factor
  // that changed nothing anywhere would mean the enumeration is not exercising
  // it (or that production stopped reading it), so each factor must move at
  // least one pair. The per-factor pair counts are also asserted literally, so
  // a semantic drift that keeps a factor merely "non-zero" still fails.
  const int kExpectedDifferingPairs[kFactorCount] = {
      /* registration   */ 64,
      /* depth          */ 5,
      /* ram_backed     */ 5,
      /* full_rect      */ 2,
      /* color_bridge   */ 5,
      /* covered_r32f   */ 3,
      /* shape_mismatch */ 2,
  };

  for (int factor = 0; factor < kFactorCount; ++factor) {
    const int bit = 1 << factor;
    int pairs_compared = 0;
    int differing_pairs = 0;
    int first_differing_row = -1;

    for (int row = 0; row < kRowCount; ++row) {
      if (row & bit) {
        continue;  // each unordered pair is visited once, from its low member
      }
      const ResolvedFetchService low = ClassifyResolvedFetch(Row(row));
      const ResolvedFetchService high = ClassifyResolvedFetch(Row(row | bit));
      ++pairs_compared;
      if (low != high) {
        if (first_differing_row < 0) {
          first_differing_row = row;
        }
        ++differing_pairs;
      }
    }

    INFO("factor " << kFactorNames[factor] << " (row-index bit " << factor
                   << "): " << differing_pairs << " differing of "
                   << pairs_compared << " pairs");
    REQUIRE(pairs_compared == 64);
    REQUIRE(differing_pairs >= 1);
    REQUIRE(differing_pairs == kExpectedDifferingPairs[factor]);

    // Name the concrete paired rows that prove this factor independently
    // changes the classification.
    REQUIRE(first_differing_row >= 0);
    const ResolvedFetchService low = ClassifyResolvedFetch(Row(first_differing_row));
    const ResolvedFetchService high =
        ClassifyResolvedFetch(Row(first_differing_row | bit));
    INFO("witness: " << DescribeRow(first_differing_row) << " -> "
                     << ServiceName(low) << "  vs  "
                     << DescribeRow(first_differing_row | bit) << " -> "
                     << ServiceName(high));
    REQUIRE(low != high);
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
