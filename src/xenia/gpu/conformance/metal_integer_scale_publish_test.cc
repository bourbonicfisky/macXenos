/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Sparse-binding transport conformance for integer texture scaling
// (openspec/changes/implement-unsigned-integer-texture-scaling, task 6.4).
//
// The engaged Just Cause 2 consumers were measured at fetch constants tf6 and
// tf10 (and tf4 in one trace) — never at slot 0, and never contiguous. This
// suite pins the invariants a rendered frame cannot show: the value lands at
// the binding's own index, every unbound slot stays zero, and a stage with no
// integer fetches contributes nothing.
//
// It calls the same metal::PublishIntegerScaleBits the command processor uses,
// so it cannot pass against a re-implementation of the rule.

#include <cstdint>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_integer_scale_publish.h"
#include "xenia/gpu/texture_util.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::kMetalIntegerScaleSlotCount;
using metal::PublishIntegerScaleBits;
using metal::ResetIntegerScaleBits;

// A fetch constant shaped like the measured Just Cause 2 LUT binding:
// k_8_8_8_8, integer num_format, swizzle 0x60A, unsigned signs, point sampled
// (zero-initialized filters are point). point=false selects linear filters.
xenos::xe_gpu_texture_fetch_t MakeFetch(xenos::TextureFormat format,
                                        uint32_t num_format,
                                        uint32_t swizzle = 0x60A,
                                        bool point = true) {
  xenos::xe_gpu_texture_fetch_t fetch = {};
  fetch.format = format;
  fetch.num_format = num_format;
  fetch.swizzle = swizzle;
  if (!point) {
    fetch.mag_filter = xenos::TextureFilter::kLinear;
    fetch.min_filter = xenos::TextureFilter::kLinear;
    fetch.mip_filter = xenos::TextureFilter::kLinear;
  }
  return fetch;
}

// The value the backend publishes for that binding since fetch ABI 8: four
// unsigned 8-bit fields (7) in Canary's 6-bit layout plus the point bit 26.
constexpr uint32_t kJC2ScaleBits = 0x041C71C7u;

}  // namespace

TEST_CASE("Integer scale publishes at the binding's own sparse slot",
          "[integer_scale][transport]") {
  std::vector<uint32_t> slots(kMetalIntegerScaleSlotCount, 0xDEADBEEFu);
  ResetIntegerScaleBits(slots.data(), slots.size());
  for (uint32_t v : slots) {
    REQUIRE(v == 0);
  }

  SECTION("tf6 and tf10 only, every gap stays zero") {
    const xenos::xe_gpu_texture_fetch_t engaged =
        MakeFetch(xenos::TextureFormat::k_8_8_8_8, 1);
    CHECK(PublishIntegerScaleBits(6, engaged, 0, slots.data(), slots.size()) ==
          kJC2ScaleBits);
    CHECK(PublishIntegerScaleBits(10, engaged, 0, slots.data(), slots.size()) ==
          kJC2ScaleBits);

    for (uint32_t i = 0; i < kMetalIntegerScaleSlotCount; ++i) {
      INFO("slot " << i);
      if (i == 6 || i == 10) {
        CHECK(slots[i] == kJC2ScaleBits);
      } else {
        CHECK(slots[i] == 0);
      }
    }
  }

  SECTION("the tf4 site seen in 534307E7_4218 lands at 4, not 0") {
    const xenos::xe_gpu_texture_fetch_t engaged =
        MakeFetch(xenos::TextureFormat::k_8_8_8_8, 1);
    PublishIntegerScaleBits(4, engaged, 0, slots.data(), slots.size());
    CHECK(slots[4] == kJC2ScaleBits);
    CHECK(slots[0] == 0);
  }

  SECTION("the highest and lowest slots are both reachable") {
    const xenos::xe_gpu_texture_fetch_t engaged =
        MakeFetch(xenos::TextureFormat::k_8_8_8_8, 1);
    PublishIntegerScaleBits(0, engaged, 0, slots.data(), slots.size());
    PublishIntegerScaleBits(uint32_t(kMetalIntegerScaleSlotCount - 1), engaged,
                            0, slots.data(), slots.size());
    CHECK(slots[0] == kJC2ScaleBits);
    CHECK(slots[kMetalIntegerScaleSlotCount - 1] == kJC2ScaleBits);
    for (uint32_t i = 1; i + 1 < kMetalIntegerScaleSlotCount; ++i) {
      INFO("slot " << i);
      CHECK(slots[i] == 0);
    }
  }
}

TEST_CASE("Unaffected bindings publish zero into their own slots",
          "[integer_scale][transport]") {
  std::vector<uint32_t> slots(kMetalIntegerScaleSlotCount, 0);

  SECTION("filtered fractional bindings publish no integer scale") {
    // Mirrors the measured population: DXT, float and fractional-num_format
    // bindings dominate. Non-fixed formats publish nothing; fixed ones only
    // request Q16 rounding of unsigned components (bit 24), with no widths.
    const struct {
      uint32_t slot;
      xenos::TextureFormat format;
      uint32_t expected;
    } bindings[] = {
        {0, xenos::TextureFormat::k_DXT1, 0},
        {1, xenos::TextureFormat::k_DXT4_5, 0},
        {2, xenos::TextureFormat::k_8_8_8_8,
         texture_util::kIntegerScaleNormalizedBit},
        {11, xenos::TextureFormat::k_16,
         texture_util::kIntegerScaleNormalizedBit},
        {12, xenos::TextureFormat::k_4_4_4_4,
         texture_util::kIntegerScaleNormalizedBit},
    };
    for (const auto& b : bindings) {
      INFO("slot " << b.slot);
      CHECK(PublishIntegerScaleBits(b.slot,
                                    MakeFetch(b.format, 0, 0x60A, false), 0,
                                    slots.data(), slots.size()) == b.expected);
      CHECK(slots[b.slot] == b.expected);
    }
  }

  SECTION("integer num_format on a non-fixed format still publishes zero") {
    // These four appear with num_format=1 in the corpus and must not scale.
    for (xenos::TextureFormat format :
         {xenos::TextureFormat::k_24_8_FLOAT, xenos::TextureFormat::k_32_FLOAT,
          xenos::TextureFormat::k_16_16_16_16_FLOAT,
          xenos::TextureFormat::k_16_16_16_16_EXPAND}) {
      INFO("format " << uint32_t(format));
      CHECK(PublishIntegerScaleBits(7, MakeFetch(format, 1, 0x60A, false), 0,
                                    slots.data(), slots.size()) == 0);
      CHECK(slots[7] == 0);
    }
  }

  SECTION("an engaged slot rebound to an unaffected texture clears itself") {
    // Slot reuse across draws must not leave a stale scale behind.
    PublishIntegerScaleBits(6, MakeFetch(xenos::TextureFormat::k_8_8_8_8, 1), 0,
                            slots.data(), slots.size());
    REQUIRE(slots[6] == kJC2ScaleBits);
    PublishIntegerScaleBits(
        6, MakeFetch(xenos::TextureFormat::k_DXT1, 0, 0x60A, false), 0,
        slots.data(), slots.size());
    CHECK(slots[6] == 0);
  }
}

TEST_CASE("Out-of-range fetch constants are dropped, never clamped",
          "[integer_scale][transport]") {
  // Clamping would write a real scale onto an unrelated slot — a silent
  // corruption strictly worse than dropping the binding.
  std::vector<uint32_t> slots(kMetalIntegerScaleSlotCount, 0);
  const xenos::xe_gpu_texture_fetch_t engaged =
      MakeFetch(xenos::TextureFormat::k_8_8_8_8, 1);
  CHECK(PublishIntegerScaleBits(uint32_t(kMetalIntegerScaleSlotCount), engaged,
                                0, slots.data(), slots.size()) == 0);
  CHECK(PublishIntegerScaleBits(0xFFFFFFFFu, engaged, 0, slots.data(),
                                slots.size()) == 0);
  for (uint32_t v : slots) {
    CHECK(v == 0);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
