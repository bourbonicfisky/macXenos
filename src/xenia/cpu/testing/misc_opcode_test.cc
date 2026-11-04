/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Tests for remaining untested HIR opcodes: CAST, ROUND, TO_SINGLE,
// MUL_ADD, MUL_SUB, ADD_CARRY, RECIP, RSQRT, LVL/LVR/STVL/STVR,
// RESERVED_LOAD/RESERVED_STORE, VECTOR_AVERAGE.

#include "xenia/cpu/testing/util.h"

#include "xenia/base/cvar.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;

// ============================================================================
// CAST — bitcast I32<->F32, I64<->F64
// ============================================================================
TEST_CASE("CAST_I32_TO_F32", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    auto ival = b.Truncate(LoadGPR(b, 4), INT32_TYPE);
    auto fval = b.Cast(ival, FLOAT32_TYPE);
    // Store back as I32 via cast to verify round-trip.
    StoreGPR(b, 3, b.ZeroExtend(b.Cast(fval, INT32_TYPE), INT64_TYPE));
    b.Return();
  });
  // 1.0f = 0x3F800000
  test.Run([](PPCContext* ctx) { ctx->r[4] = 0x3F800000; },
           [](PPCContext* ctx) {
             REQUIRE(static_cast<uint32_t>(ctx->r[3]) == 0x3F800000);
           });
  // -0.0f = 0x80000000
  test.Run([](PPCContext* ctx) { ctx->r[4] = 0x80000000; },
           [](PPCContext* ctx) {
             REQUIRE(static_cast<uint32_t>(ctx->r[3]) == 0x80000000);
           });
  // NaN = 0x7FC00000
  test.Run([](PPCContext* ctx) { ctx->r[4] = 0x7FC00000; },
           [](PPCContext* ctx) {
             REQUIRE(static_cast<uint32_t>(ctx->r[3]) == 0x7FC00000);
           });
}

TEST_CASE("CAST_I64_TO_F64", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    auto ival = LoadGPR(b, 4);
    auto fval = b.Cast(ival, FLOAT64_TYPE);
    StoreGPR(b, 3, b.Cast(fval, INT64_TYPE));
    b.Return();
  });
  // 1.0 = 0x3FF0000000000000
  test.Run(
      [](PPCContext* ctx) { ctx->r[4] = 0x3FF0000000000000ULL; },
      [](PPCContext* ctx) { REQUIRE(ctx->r[3] == 0x3FF0000000000000ULL); });
  // -1.0 = 0xBFF0000000000000
  test.Run(
      [](PPCContext* ctx) { ctx->r[4] = 0xBFF0000000000000ULL; },
      [](PPCContext* ctx) { REQUIRE(ctx->r[3] == 0xBFF0000000000000ULL); });
}

// ============================================================================
// ROUND F64 (F32 is impossible on x64)
// ============================================================================
TEST_CASE("ROUND_F64_NEAREST", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.Round(LoadFPR(b, 4), ROUND_TO_NEAREST));
    b.Return();
  });
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.5; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 2.0); });
  test.Run(
      [](PPCContext* ctx) { ctx->f[4] = 2.5; },
      [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 2.0); });  // banker's rounding
  test.Run([](PPCContext* ctx) { ctx->f[4] = -1.5; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == -2.0); });
}

TEST_CASE("ROUND_F64_ZERO", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.Round(LoadFPR(b, 4), ROUND_TO_ZERO));
    b.Return();
  });
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.9; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 1.0); });
  test.Run([](PPCContext* ctx) { ctx->f[4] = -1.9; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == -1.0); });
}

TEST_CASE("ROUND_F64_CEIL", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.Round(LoadFPR(b, 4), ROUND_TO_POSITIVE_INFINITY));
    b.Return();
  });
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.1; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 2.0); });
  test.Run([](PPCContext* ctx) { ctx->f[4] = -1.1; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == -1.0); });
}

TEST_CASE("ROUND_F64_FLOOR", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.Round(LoadFPR(b, 4), ROUND_TO_MINUS_INFINITY));
    b.Return();
  });
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.9; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 1.0); });
  test.Run([](PPCContext* ctx) { ctx->f[4] = -1.1; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == -2.0); });
}

// ============================================================================
// TO_SINGLE — double -> single -> double rounding
// ============================================================================
TEST_CASE("TO_SINGLE", "[convert]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.ToSingle(LoadFPR(b, 4)));
    b.Return();
  });
  // Exact value survives.
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.0; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 1.0); });
  // Value requiring rounding: 1.0 + 2^-24 in double -> rounds in single.
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.0 + std::ldexp(1.0, -24); },
           [](PPCContext* ctx) {
             float f32 = static_cast<float>(ctx->f[3]);
             REQUIRE(f32 == 1.0f);  // rounded to nearest in single
           });
  // Negative value.
  test.Run([](PPCContext* ctx) { ctx->f[4] = -3.14; },
           [](PPCContext* ctx) {
             float f32 = static_cast<float>(ctx->f[3]);
             REQUIRE(f32 == static_cast<float>(-3.14));
           });
}

// ============================================================================
// MUL_ADD F64 — fused multiply-add: (1 * 2) + 3
// ============================================================================
TEST_CASE("MUL_ADD_F64", "[arithmetic]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.MulAdd(LoadFPR(b, 4), LoadFPR(b, 5), LoadFPR(b, 6)));
    b.Return();
  });
  // 2.0 * 3.0 + 4.0 = 10.0
  test.Run(
      [](PPCContext* ctx) {
        ctx->f[4] = 2.0;
        ctx->f[5] = 3.0;
        ctx->f[6] = 4.0;
      },
      [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 10.0); });
  // -1.0 * 2.0 + 3.0 = 1.0
  test.Run(
      [](PPCContext* ctx) {
        ctx->f[4] = -1.0;
        ctx->f[5] = 2.0;
        ctx->f[6] = 3.0;
      },
      [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 1.0); });
  // 0.0 * anything + 5.0 = 5.0
  test.Run(
      [](PPCContext* ctx) {
        ctx->f[4] = 0.0;
        ctx->f[5] = 999.0;
        ctx->f[6] = 5.0;
      },
      [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 5.0); });
}

// ============================================================================
// MUL_SUB F64 — fused multiply-subtract: (1 * 2) - 3
// ============================================================================
TEST_CASE("MUL_SUB_F64", "[arithmetic]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.MulSub(LoadFPR(b, 4), LoadFPR(b, 5), LoadFPR(b, 6)));
    b.Return();
  });
  // 2.0 * 3.0 - 4.0 = 2.0
  test.Run(
      [](PPCContext* ctx) {
        ctx->f[4] = 2.0;
        ctx->f[5] = 3.0;
        ctx->f[6] = 4.0;
      },
      [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 2.0); });
  // 5.0 * 5.0 - 25.0 = 0.0
  test.Run(
      [](PPCContext* ctx) {
        ctx->f[4] = 5.0;
        ctx->f[5] = 5.0;
        ctx->f[6] = 25.0;
      },
      [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 0.0); });
}

// ============================================================================
// ADD_CARRY I8/I32 — add with carry-in
// ============================================================================
TEST_CASE("ADD_CARRY_I8", "[arithmetic]") {
  TestFunction test([](HIRBuilder& b) {
    StoreGPR(b, 3,
             b.ZeroExtend(b.AddWithCarry(b.Truncate(LoadGPR(b, 4), INT8_TYPE),
                                         b.Truncate(LoadGPR(b, 5), INT8_TYPE),
                                         b.Truncate(LoadGPR(b, 6), INT8_TYPE)),
                          INT64_TYPE));
    b.Return();
  });
  // 0xFF + 0x00 + carry=1 = 0x00 (wraps)
  test.Run(
      [](PPCContext* ctx) {
        ctx->r[4] = 0xFF;
        ctx->r[5] = 0;
        ctx->r[6] = 1;
      },
      [](PPCContext* ctx) {
        REQUIRE(static_cast<uint8_t>(ctx->r[3]) == 0x00);
      });
  // 0x10 + 0x20 + carry=0 = 0x30
  test.Run(
      [](PPCContext* ctx) {
        ctx->r[4] = 0x10;
        ctx->r[5] = 0x20;
        ctx->r[6] = 0;
      },
      [](PPCContext* ctx) {
        REQUIRE(static_cast<uint8_t>(ctx->r[3]) == 0x30);
      });
  // 0x10 + 0x20 + carry=1 = 0x31
  test.Run(
      [](PPCContext* ctx) {
        ctx->r[4] = 0x10;
        ctx->r[5] = 0x20;
        ctx->r[6] = 1;
      },
      [](PPCContext* ctx) {
        REQUIRE(static_cast<uint8_t>(ctx->r[3]) == 0x31);
      });
}

// ============================================================================
// RECIP F32/F64 — exact reciprocal (1/x)
// ============================================================================
TEST_CASE("RECIP_F64", "[arithmetic]") {
  TestFunction test([](HIRBuilder& b) {
    StoreFPR(b, 3, b.Recip(LoadFPR(b, 4)));
    b.Return();
  });
  test.Run([](PPCContext* ctx) { ctx->f[4] = 2.0; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 0.5); });
  test.Run([](PPCContext* ctx) { ctx->f[4] = -4.0; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == -0.25); });
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.0; },
           [](PPCContext* ctx) { REQUIRE(ctx->f[3] == 1.0); });
}

// ============================================================================
// RSQRT F32 — 1/sqrt(x) (exact, not estimate)
// ============================================================================
TEST_CASE("RSQRT_F32", "[arithmetic]") {
  TestFunction test([](HIRBuilder& b) {
    auto val = b.Convert(LoadFPR(b, 4), FLOAT32_TYPE);
    auto rsqrt = b.RSqrt(val);
    StoreFPR(b, 3, b.Convert(rsqrt, FLOAT64_TYPE));
    b.Return();
  });
  // rsqrt(1.0) = 1.0
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.0; },
           [](PPCContext* ctx) {
             float r = static_cast<float>(ctx->f[3]);
             REQUIRE(r == 1.0f);
           });
  // rsqrt(4.0) = 0.5
  test.Run([](PPCContext* ctx) { ctx->f[4] = 4.0; },
           [](PPCContext* ctx) {
             float r = static_cast<float>(ctx->f[3]);
             REQUIRE(r == 0.5f);
           });
}

// ============================================================================
// RSQRT F64 — PPC frsqrte (lookup table estimate)
// ============================================================================
TEST_CASE("RSQRT_F64", "[arithmetic]") {
  TestFunction test([](HIRBuilder& b) {
    auto rsqrt = b.RSqrt(LoadFPR(b, 4));
    StoreFPR(b, 3, rsqrt);
    b.Return();
  });
  // rsqrt(1.0) — table gives an estimate near 1.0
  test.Run([](PPCContext* ctx) { ctx->f[4] = 1.0; },
           [](PPCContext* ctx) {
             // PPC frsqrte returns a table-based estimate, not exact 1.0.
             // Verify it's in the right ballpark (within ~3% of 1.0).
             REQUIRE(ctx->f[3] > 0.96);
             REQUIRE(ctx->f[3] < 1.04);
           });
  // rsqrt(4.0) — estimate near 0.5
  test.Run([](PPCContext* ctx) { ctx->f[4] = 4.0; },
           [](PPCContext* ctx) {
             REQUIRE(ctx->f[3] > 0.48);
             REQUIRE(ctx->f[3] < 0.52);
           });
  // rsqrt(+0.0) → +Inf
  test.Run([](PPCContext* ctx) { ctx->f[4] = 0.0; },
           [](PPCContext* ctx) {
             REQUIRE(std::isinf(ctx->f[3]));
             REQUIRE(ctx->f[3] > 0);
           });
  // rsqrt(negative) → QNaN
  test.Run([](PPCContext* ctx) { ctx->f[4] = -1.0; },
           [](PPCContext* ctx) { REQUIRE(std::isnan(ctx->f[3])); });
}

#if XE_ARCH_ARM64
DECLARE_bool(a64_inline_vrsqrtefp);

// The inline a64 vrsqrtefp must give exactly the helper's bits. Run every
// input through a function compiled with the helper calls, then through one
// compiled inline, and compare.
TEST_CASE("RSQRT_V128_INLINE_MATCHES_HELPER", "[vector]") {
  std::vector<uint32_t> inputs = {
      0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000,
      0x3F800000, 0x40800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
      0x7FC00000, 0xFFC00001, 0x7F800001, 0xBF800000, 0x3EFFFFFF,
      0x3F000000, 0x3F7FFFFF, 0x00FFFFFF, 0x01000000, 0x7E7FFFFF,
  };
  uint32_t seed = 0x2545F491u;
  while (inputs.size() < 120000) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    inputs.push_back(seed);
  }
  // Every table entry, both ends of the interpolation range.
  for (uint32_t index = 0; index < 32; ++index) {
    for (uint32_t interp : {0u, 1u, 511u, 1022u, 1023u}) {
      inputs.push_back(0x3F000000u + (index << 19) + (interp << 9));
    }
  }
  while (inputs.size() % 4) {
    inputs.push_back(0x3F800000);
  }
  auto run = [&](bool inline_sequence) {
    cvars::a64_inline_vrsqrtefp = inline_sequence;
    std::vector<uint32_t> out(inputs.size());
    TestFunction test([](HIRBuilder& b) {
      StoreVR(b, 3, b.RSqrt(LoadVR(b, 4)));
      b.Return();
    });
    for (size_t k = 0; k < inputs.size(); k += 4) {
      test.Run(
          [&](PPCContext* ctx) {
            ctx->v[4] = vec128i(inputs[k], inputs[k + 1], inputs[k + 2],
                                inputs[k + 3]);
          },
          [&](PPCContext* ctx) {
            for (int lane = 0; lane < 4; ++lane) {
              out[k + lane] = ctx->v[3].u32[lane];
            }
          });
    }
    return out;
  };
  const std::vector<uint32_t> helper = run(false);
  const std::vector<uint32_t> inlined = run(true);
  cvars::a64_inline_vrsqrtefp = true;
  size_t mismatches = 0;
  for (size_t k = 0; k < inputs.size(); ++k) {
    if (helper[k] != inlined[k]) {
      if (!mismatches) {
        UNSCOPED_INFO("first mismatch: input " << std::hex << inputs[k]
                                               << " helper " << helper[k]
                                               << " inline " << inlined[k]);
      }
      ++mismatches;
    }
  }
  REQUIRE(mismatches == 0);
}

DECLARE_bool(a64_skip_fma_input_denorm_flush);

// vmaddfp/vnmsubfp input flushes skipped under NJM (FPCR.FZ flushes inside
// the FMA) must give the software flush's bits, with NJM on and off.
TEST_CASE("VMX_FMA_INPUT_FLUSH_SKIP_MATCHES_SOFTWARE", "[vector]") {
  std::vector<uint32_t> inputs = {
      0x00000001, 0x007FFFFF, 0x80000001, 0x807FFFFF, 0x00400000,
      0x00000000, 0x80000000, 0x00800000, 0x80800000, 0x3F800000,
      0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001,
      0xFFC00001, 0x7F7FFFFF, 0x0DA24260, 0x8DA24260, 0x1F800000,
  };
  uint32_t seed = 0x9E3779B9u;
  while (inputs.size() < 3 * 4 * 2000) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    // Half the lanes denormal (sign kept), the rest any bits.
    inputs.push_back((seed & 1) ? (seed & 0x807FFFFFu) : seed);
  }
  auto run = [&](bool skip, bool njm, bool subtract) {
    cvars::a64_skip_fma_input_denorm_flush = skip;
    std::vector<uint32_t> out(inputs.size() / 3);
    TestFunction test([njm, subtract](HIRBuilder& b) {
      b.SetNJM(b.LoadConstantInt8(njm ? 1 : 0));
      auto a = b.VectorDenormFlush(LoadVR(b, 4));
      auto c = b.VectorDenormFlush(LoadVR(b, 5));
      auto d = b.VectorDenormFlush(LoadVR(b, 6));
      StoreVR(b, 3, subtract ? b.MulSub(a, c, d) : b.MulAdd(a, c, d));
      b.Return();
    });
    for (size_t k = 0; k + 12 <= inputs.size(); k += 12) {
      test.Run(
          [&](PPCContext* ctx) {
            ctx->v[4] =
                vec128i(inputs[k], inputs[k + 1], inputs[k + 2], inputs[k + 3]);
            ctx->v[5] = vec128i(inputs[k + 4], inputs[k + 5], inputs[k + 6],
                                inputs[k + 7]);
            ctx->v[6] = vec128i(inputs[k + 8], inputs[k + 9], inputs[k + 10],
                                inputs[k + 11]);
          },
          [&](PPCContext* ctx) {
            for (int lane = 0; lane < 4; ++lane) {
              out[k / 3 + lane] = ctx->v[3].u32[lane];
            }
          });
    }
    return out;
  };
  for (const bool njm : {true, false}) {
    for (const bool subtract : {false, true}) {
      const std::vector<uint32_t> software = run(false, njm, subtract);
      const std::vector<uint32_t> skipped = run(true, njm, subtract);
      size_t mismatches = 0;
      for (size_t k = 0; k < software.size(); ++k) {
        if (software[k] != skipped[k]) {
          if (!mismatches) {
            UNSCOPED_INFO("njm " << njm << " sub " << subtract
                                 << " first mismatch at " << k << std::hex
                                 << " software " << software[k] << " skipped "
                                 << skipped[k]);
          }
          ++mismatches;
        }
      }
      CHECK(mismatches == 0);
    }
  }
  cvars::a64_skip_fma_input_denorm_flush = true;
}

// The NJM-off flush sits in tail code: in a function far larger than a tbz's
// +-32 KiB reach the branch to it must still assemble and run.
TEST_CASE("VMX_FMA_INPUT_FLUSH_SKIP_LARGE_FUNCTION", "[vector]") {
  auto run = [](bool skip, bool njm) {
    cvars::a64_skip_fma_input_denorm_flush = skip;
    TestFunction test([njm](HIRBuilder& b) {
      b.SetNJM(b.LoadConstantInt8(njm ? 1 : 0));
      auto v = LoadVR(b, 4);
      for (int k = 0; k < 600; ++k) {
        v = b.MulAdd(b.VectorDenormFlush(v), b.VectorDenormFlush(LoadVR(b, 5)),
                     b.VectorDenormFlush(LoadVR(b, 6)));
      }
      StoreVR(b, 3, v);
      b.Return();
    });
    vec128_t out;
    test.Run(
        [](PPCContext* ctx) {
          ctx->v[4] = vec128i(0x00000001, 0x3F800000, 0x807FFFFF, 0x00400000);
          ctx->v[5] = vec128i(0x3F800000, 0x00000010, 0x3F800000, 0x3F800000);
          ctx->v[6] = vec128i(0x00000002, 0x80000003, 0x00000000, 0x00000001);
        },
        [&](PPCContext* ctx) { out = ctx->v[3]; });
    return out;
  };
  for (const bool njm : {true, false}) {
    const vec128_t software = run(false, njm);
    const vec128_t skipped = run(true, njm);
    INFO("njm " << njm);
    CHECK(software == skipped);
  }
  cvars::a64_skip_fma_input_denorm_flush = true;
}

DECLARE_bool(a64_fma_njm_check_per_op);
DECLARE_bool(a64_vmx_nan_fixup_out_of_line);

// Every combination of the vector-float switches against the original
// sequences (software flush per input, inline NaN fix-up, sources stashed on
// every FMA), bit for bit, with NJM on and off.
struct VmxFloatConfig {
  bool skip_flush;
  bool njm_check_per_op;
  bool nan_out_of_line;
};
constexpr VmxFloatConfig kVmxFloatReference = {false, false, false};
constexpr VmxFloatConfig kVmxFloatConfigs[] = {
    {true, false, false}, {true, true, false},  {true, false, true},
    {true, true, true},   {false, false, true}, {false, true, true},
};

static void SetVmxFloatConfig(const VmxFloatConfig& config) {
  cvars::a64_skip_fma_input_denorm_flush = config.skip_flush;
  cvars::a64_fma_njm_check_per_op = config.njm_check_per_op;
  cvars::a64_vmx_nan_fixup_out_of_line = config.nan_out_of_line;
}

// Lanes biased to the cases the sequences treat specially: denormals of both
// signs, quiet and signalling NaNs with payloads, infinities, zeros.
static std::vector<uint32_t> VmxFloatInputs(size_t count, uint32_t seed) {
  static constexpr uint32_t kSpecial[] = {
      0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00400000, 0x80000010,
      0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FC12345, 0xFFC00001,
      0x7F800001, 0xFF800001, 0x7FBFFFFF, 0xFFA00000, 0x3F800000, 0xBF800000,
      0x00800000, 0x80800000, 0x7F7FFFFF, 0x0DA24260, 0x8DA24260, 0x40490FDB,
  };
  std::vector<uint32_t> out;
  out.reserve(count);
  while (out.size() < count) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    switch (seed & 3) {
      case 0:
        out.push_back(kSpecial[(seed >> 8) % (sizeof(kSpecial) / 4)]);
        break;
      case 1:
        out.push_back(seed & 0x807FFFFFu);  // denormal or zero
        break;
      default:
        out.push_back(seed);
        break;
    }
  }
  return out;
}

// Runs build (which reads v4-v6 and writes v3 and v7) over the inputs and
// returns v3 and v7 per run.
static std::vector<uint32_t> RunVmxFloatCase(
    const VmxFloatConfig& config, bool njm,
    const std::function<void(HIRBuilder&)>& build,
    const std::vector<uint32_t>& inputs) {
  SetVmxFloatConfig(config);
  TestFunction test([njm, &build](HIRBuilder& b) {
    b.SetNJM(b.LoadConstantInt8(njm ? 1 : 0));
    build(b);
    b.Return();
  });
  std::vector<uint32_t> out;
  for (size_t k = 0; k + 12 <= inputs.size(); k += 12) {
    test.Run(
        [&](PPCContext* ctx) {
          ctx->v[4] =
              vec128i(inputs[k], inputs[k + 1], inputs[k + 2], inputs[k + 3]);
          ctx->v[5] = vec128i(inputs[k + 4], inputs[k + 5], inputs[k + 6],
                              inputs[k + 7]);
          ctx->v[6] = vec128i(inputs[k + 8], inputs[k + 9], inputs[k + 10],
                              inputs[k + 11]);
          ctx->v[3] = vec128i(0);
          ctx->v[7] = vec128i(0);
        },
        [&](PPCContext* ctx) {
          for (int lane = 0; lane < 4; ++lane) {
            out.push_back(ctx->v[3].u32[lane]);
            out.push_back(ctx->v[7].u32[lane]);
          }
        });
  }
  return out;
}

TEST_CASE("VMX_FLOAT_SWITCHES_MATCH_ORIGINAL_SEQUENCES", "[vector]") {
  const vec128_t kConstant =
      vec128i(0x3F800000u, 0x00000003u, 0x7FA00001u, 0xFF800000u);
  using Build = std::function<void(HIRBuilder&)>;
  const std::pair<const char*, Build> cases[] = {
      {"vmaddfp",
       [](HIRBuilder& b) {
         StoreVR(b, 3,
                 b.MulAdd(b.VectorDenormFlush(LoadVR(b, 4)),
                          b.VectorDenormFlush(LoadVR(b, 5)),
                          b.VectorDenormFlush(LoadVR(b, 6))));
       }},
      {"vnmsubfp",
       [](HIRBuilder& b) {
         StoreVR(b, 3,
                 b.Neg(b.MulSub(b.VectorDenormFlush(LoadVR(b, 4)),
                                b.VectorDenormFlush(LoadVR(b, 5)),
                                b.VectorDenormFlush(LoadVR(b, 6)))));
       }},
      // One flushed value feeding two FMAs in two operand positions, and an
      // FMA mixing flushed and unflushed inputs.
      {"shared flushed input",
       [](HIRBuilder& b) {
         auto a = b.VectorDenormFlush(LoadVR(b, 4));
         auto r = b.MulAdd(a, b.VectorDenormFlush(LoadVR(b, 5)), a);
         StoreVR(b, 7, r);
         StoreVR(b, 3, b.MulSub(a, a, r));
       }},
      {"unflushed inputs",
       [](HIRBuilder& b) {
         StoreVR(b, 3, b.MulAdd(LoadVR(b, 4), LoadVR(b, 5), LoadVR(b, 6)));
         StoreVR(b, 7, b.MulSub(LoadVR(b, 6), LoadVR(b, 4), LoadVR(b, 5)));
       }},
      {"constant sources",
       [kConstant](HIRBuilder& b) {
         StoreVR(b, 3,
                 b.MulAdd(b.VectorDenormFlush(LoadVR(b, 4)),
                          b.LoadConstantVec128(kConstant),
                          b.VectorDenormFlush(LoadVR(b, 6))));
         StoreVR(b, 7,
                 b.MulSub(b.LoadConstantVec128(kConstant),
                          b.VectorDenormFlush(LoadVR(b, 5)),
                          b.LoadConstantVec128(kConstant)));
       }},
      {"binary ops",
       [](HIRBuilder& b) {
         StoreVR(b, 3, b.Sub(b.Add(LoadVR(b, 4), LoadVR(b, 5)), LoadVR(b, 6)));
         StoreVR(b, 7, b.Div(b.Mul(LoadVR(b, 5), LoadVR(b, 6)), LoadVR(b, 4)));
       }},
  };
  const std::vector<uint32_t> inputs = VmxFloatInputs(12 * 600, 0x1234567u);
  for (const auto& [name, build] : cases) {
    for (const bool njm : {true, false}) {
      const std::vector<uint32_t> reference =
          RunVmxFloatCase(kVmxFloatReference, njm, build, inputs);
      for (const VmxFloatConfig& config : kVmxFloatConfigs) {
        const std::vector<uint32_t> out =
            RunVmxFloatCase(config, njm, build, inputs);
        size_t mismatches = 0;
        for (size_t k = 0; k < out.size(); ++k) {
          if (out[k] != reference[k]) {
            if (!mismatches) {
              UNSCOPED_INFO(name << " njm " << njm << " config "
                                 << config.skip_flush << config.njm_check_per_op
                                 << config.nan_out_of_line << " first at " << k
                                 << std::hex << " reference " << reference[k]
                                 << " got " << out[k]);
            }
            ++mismatches;
          }
        }
        CHECK(mismatches == 0);
      }
    }
  }
  SetVmxFloatConfig({true, true, true});
}

// Out-of-line NJM flushes and NaN fix-ups in a function far beyond a short
// branch's reach, with a NaN lane so every fix-up runs.
TEST_CASE("VMX_FLOAT_SWITCHES_LARGE_FUNCTION", "[vector]") {
  auto run = [](const VmxFloatConfig& config, bool njm) {
    SetVmxFloatConfig(config);
    TestFunction test([njm](HIRBuilder& b) {
      b.SetNJM(b.LoadConstantInt8(njm ? 1 : 0));
      auto v = LoadVR(b, 4);
      for (int k = 0; k < 400; ++k) {
        v = b.MulAdd(b.VectorDenormFlush(v), b.VectorDenormFlush(LoadVR(b, 5)),
                     b.VectorDenormFlush(LoadVR(b, 6)));
        v = b.Add(v, LoadVR(b, 6));
      }
      StoreVR(b, 3, v);
      b.Return();
    });
    vec128_t out;
    test.Run(
        [](PPCContext* ctx) {
          ctx->v[4] = vec128i(0x00000001, 0x3F800000, 0x807FFFFF, 0x00400000);
          ctx->v[5] = vec128i(0x3F800000, 0x00000010, 0x7FA00002, 0x3F800000);
          ctx->v[6] = vec128i(0x00000002, 0x80000003, 0x00000000, 0xFF900001);
        },
        [&](PPCContext* ctx) { out = ctx->v[3]; });
    return out;
  };
  for (const bool njm : {true, false}) {
    const vec128_t reference = run(kVmxFloatReference, njm);
    for (const VmxFloatConfig& config : kVmxFloatConfigs) {
      INFO("njm " << njm << " config " << config.skip_flush
                  << config.njm_check_per_op << config.nan_out_of_line);
      CHECK(run(config, njm) == reference);
    }
  }
  SetVmxFloatConfig({true, true, true});
}
#endif  // XE_ARCH_ARM64

// ============================================================================
// RSQRT V128 — PPC vrsqrtefp (per-lane lookup table estimate)
// ============================================================================
TEST_CASE("RSQRT_V128", "[vector]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3, b.RSqrt(LoadVR(b, 4)));
    b.Return();
  });
  // Normal positive values: estimates should be in the right ballpark.
  test.Run(
      [](PPCContext* ctx) { ctx->v[4] = vec128f(1.0f, 4.0f, 16.0f, 100.0f); },
      [](PPCContext* ctx) {
        auto r = ctx->v[3];
        float r0, r1, r2, r3;
        std::memcpy(&r0, &r.u32[0], 4);
        std::memcpy(&r1, &r.u32[1], 4);
        std::memcpy(&r2, &r.u32[2], 4);
        std::memcpy(&r3, &r.u32[3], 4);
        // rsqrt(1) ≈ 1.0, rsqrt(4) ≈ 0.5, rsqrt(16) ≈ 0.25, rsqrt(100) ≈ 0.1
        REQUIRE(r0 > 0.9f);
        REQUIRE(r0 < 1.1f);
        REQUIRE(r1 > 0.45f);
        REQUIRE(r1 < 0.55f);
        REQUIRE(r2 > 0.22f);
        REQUIRE(r2 < 0.28f);
        REQUIRE(r3 > 0.08f);
        REQUIRE(r3 < 0.12f);
      });
  // +0 → +Inf
  test.Run([](PPCContext* ctx) { ctx->v[4] = vec128f(0.0f, 0.0f, 0.0f, 0.0f); },
           [](PPCContext* ctx) {
             auto r = ctx->v[3];
             float r0;
             std::memcpy(&r0, &r.u32[0], 4);
             REQUIRE(std::isinf(r0));
             REQUIRE(r0 > 0);
           });
  // Negative → QNaN
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128f(-1.0f, -4.0f, -16.0f, -100.0f);
      },
      [](PPCContext* ctx) {
        auto r = ctx->v[3];
        float r0;
        std::memcpy(&r0, &r.u32[0], 4);
        REQUIRE(std::isnan(r0));
      });
}

// ============================================================================
// VECTOR_AVERAGE I8 — rounding halving add: (a+b+1)>>1
// ============================================================================
TEST_CASE("VECTOR_AVERAGE_UNSIGNED_I8", "[vector]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3,
            b.VectorAverage(LoadVR(b, 4), LoadVR(b, 5), INT8_TYPE,
                            ARITHMETIC_UNSIGNED));
    b.Return();
  });
  // Use uniform values so byte ordering doesn't matter.
  // avg(100, 200) unsigned = (100+200+1)/2 = 150
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128b(100);
        ctx->v[5] = vec128b(200);
      },
      [](PPCContext* ctx) { REQUIRE(ctx->v[3] == vec128b(150)); });
  // avg(255, 255) unsigned = (255+255+1)/2 = 255
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128b(255);
        ctx->v[5] = vec128b(255);
      },
      [](PPCContext* ctx) { REQUIRE(ctx->v[3] == vec128b(255)); });
  // avg(0, 1) unsigned = (0+1+1)/2 = 1
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128b(0);
        ctx->v[5] = vec128b(1);
      },
      [](PPCContext* ctx) { REQUIRE(ctx->v[3] == vec128b(1)); });
}

// ============================================================================
// RESERVED_LOAD + RESERVED_STORE — LL/SC emulation
// ============================================================================
TEST_CASE("RESERVED_LOAD_STORE_I32", "[atomic]") {
  TestFunction test([](HIRBuilder& b) {
    auto addr = LoadGPR(b, 4);
    // Load with reserve.
    auto loaded = b.LoadWithReserve(addr, INT32_TYPE);
    StoreGPR(b, 3, b.ZeroExtend(loaded, INT64_TYPE));
    // Store conditional with the loaded value + 1.
    auto new_val = b.Add(loaded, b.LoadConstantInt32(1));
    auto success = b.StoreWithReserve(addr, new_val, INT32_TYPE);
    StoreGPR(b, 5, b.ZeroExtend(success, INT64_TYPE));
    b.Return();
  });

  uint32_t guest_addr = test.memory->SystemHeapAlloc(4);
  REQUIRE(guest_addr != 0);
  auto* host_ptr =
      reinterpret_cast<uint32_t*>(test.memory->TranslateVirtual(guest_addr));

  test.Run(
      [&](PPCContext* ctx) {
        *host_ptr = 42;
        ctx->r[4] = guest_addr;
      },
      [&](PPCContext* ctx) {
        REQUIRE(static_cast<uint32_t>(ctx->r[3]) == 42);  // loaded value
        REQUIRE(ctx->r[5] == 1);                          // store succeeded
        REQUIRE(*host_ptr == 43);                         // incremented
      });

  test.memory->SystemHeapFree(guest_addr);
}

TEST_CASE("RESERVED_LOAD_STORE_I64", "[atomic]") {
  TestFunction test([](HIRBuilder& b) {
    auto addr = LoadGPR(b, 4);
    auto loaded = b.LoadWithReserve(addr, INT64_TYPE);
    StoreGPR(b, 3, loaded);
    auto new_val = b.Add(loaded, b.LoadConstantInt64(1));
    auto success = b.StoreWithReserve(addr, new_val, INT64_TYPE);
    StoreGPR(b, 5, b.ZeroExtend(success, INT64_TYPE));
    b.Return();
  });

  uint32_t guest_addr = test.memory->SystemHeapAlloc(8);
  REQUIRE(guest_addr != 0);
  auto* host_ptr =
      reinterpret_cast<uint64_t*>(test.memory->TranslateVirtual(guest_addr));

  test.Run(
      [&](PPCContext* ctx) {
        *host_ptr = 100;
        ctx->r[4] = guest_addr;
      },
      [&](PPCContext* ctx) {
        REQUIRE(ctx->r[3] == 100);  // loaded value
        REQUIRE(ctx->r[5] == 1);    // store succeeded
        REQUIRE(*host_ptr == 101);  // incremented
      });

  test.memory->SystemHeapFree(guest_addr);
}

TEST_CASE("RESERVED_STORE_I32_NO_RESERVATION", "[atomic]") {
  // StoreWithReserve without a preceding LoadWithReserve must fail (return 0).
  // Run with a timeout: the bug this catches produces an infinite loop.
  TestFunction test([](HIRBuilder& b) {
    auto addr = LoadGPR(b, 4);
    auto val = b.Truncate(LoadGPR(b, 5), INT32_TYPE);
    auto success = b.StoreWithReserve(addr, val, INT32_TYPE);
    StoreGPR(b, 3, b.ZeroExtend(success, INT64_TYPE));
    b.Return();
  });

  uint32_t guest_addr = test.memory->SystemHeapAlloc(4);
  REQUIRE(guest_addr != 0);
  auto* host_ptr =
      reinterpret_cast<uint32_t*>(test.memory->TranslateVirtual(guest_addr));

  std::atomic<bool> completed{false};
  std::thread worker([&]() {
    test.Run(
        [&](PPCContext* ctx) {
          *host_ptr = 42;
          ctx->r[4] = guest_addr;
          ctx->r[5] = 99;
        },
        [&](PPCContext* ctx) {
          CHECK(ctx->r[3] == 0);   // store must fail
          CHECK(*host_ptr == 42);  // memory unchanged
        });
    completed.store(true);
  });
  worker.detach();

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!completed.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!completed.load()) {
    // Detached thread is stuck in JIT'd code — can't unwind safely.
    FAIL("Timed out: no-reservation path likely has an infinite loop");
    std::_Exit(1);
  }

  test.memory->SystemHeapFree(guest_addr);
}

TEST_CASE("RESERVED_STORE_I64_NO_RESERVATION", "[atomic]") {
  // Run with a timeout: the bug this catches produces an infinite loop.
  TestFunction test([](HIRBuilder& b) {
    auto addr = LoadGPR(b, 4);
    auto val = LoadGPR(b, 5);
    auto success = b.StoreWithReserve(addr, val, INT64_TYPE);
    StoreGPR(b, 3, b.ZeroExtend(success, INT64_TYPE));
    b.Return();
  });

  uint32_t guest_addr = test.memory->SystemHeapAlloc(8);
  REQUIRE(guest_addr != 0);
  auto* host_ptr =
      reinterpret_cast<uint64_t*>(test.memory->TranslateVirtual(guest_addr));

  std::atomic<bool> completed{false};
  std::thread worker([&]() {
    test.Run(
        [&](PPCContext* ctx) {
          *host_ptr = 42;
          ctx->r[4] = guest_addr;
          ctx->r[5] = 99;
        },
        [&](PPCContext* ctx) {
          CHECK(ctx->r[3] == 0);   // store must fail
          CHECK(*host_ptr == 42);  // memory unchanged
        });
    completed.store(true);
  });
  worker.detach();

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!completed.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!completed.load()) {
    // Detached thread is stuck in JIT'd code — can't unwind safely.
    FAIL("Timed out: no-reservation path likely has an infinite loop");
    std::_Exit(1);
  }

  test.memory->SystemHeapFree(guest_addr);
}

// ============================================================================
// LVL / LVR — partial vector load left/right
// Detailed alignment semantics are tested by PPC instruction tests (lvlx etc.)
// These tests verify the HIR opcode executes without crashing.
// ============================================================================
TEST_CASE("LOAD_VECTOR_LEFT", "[memory]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3, b.LoadVectorLeft(LoadGPR(b, 4)));
    b.Return();
  });

  // Allocate 16-byte aligned memory with known pattern.
  uint32_t guest_addr = test.memory->SystemHeapAlloc(32);
  REQUIRE(guest_addr != 0);
  // Align to 16 bytes.
  uint32_t aligned_addr = (guest_addr + 15) & ~15u;
  auto* host_ptr =
      reinterpret_cast<uint8_t*>(test.memory->TranslateVirtual(aligned_addr));
  for (int i = 0; i < 16; ++i) {
    host_ptr[i] = static_cast<uint8_t>(0x10 + i);
  }

  // LVL at aligned address — should load data (not crash).
  test.Run([&](PPCContext* ctx) { ctx->r[4] = aligned_addr; },
           [&](PPCContext* ctx) {
             // Just verify it loaded something non-zero from the pattern.
             REQUIRE((ctx->v[3].u32[0] | ctx->v[3].u32[1] | ctx->v[3].u32[2] |
                      ctx->v[3].u32[3]) != 0);
           });

  test.memory->SystemHeapFree(guest_addr);
}

TEST_CASE("LOAD_VECTOR_RIGHT", "[memory]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3, b.LoadVectorRight(LoadGPR(b, 4)));
    b.Return();
  });

  uint32_t guest_addr = test.memory->SystemHeapAlloc(32);
  REQUIRE(guest_addr != 0);
  uint32_t aligned_addr = (guest_addr + 15) & ~15u;
  auto* host_ptr =
      reinterpret_cast<uint8_t*>(test.memory->TranslateVirtual(aligned_addr));
  for (int i = 0; i < 16; ++i) {
    host_ptr[i] = static_cast<uint8_t>(0x20 + i);
  }

  // LVR at aligned address returns zero (no bytes before alignment boundary).
  test.Run([&](PPCContext* ctx) { ctx->r[4] = aligned_addr; },
           [&](PPCContext* ctx) {
             REQUIRE(ctx->v[3].u32[0] == 0);
             REQUIRE(ctx->v[3].u32[1] == 0);
             REQUIRE(ctx->v[3].u32[2] == 0);
             REQUIRE(ctx->v[3].u32[3] == 0);
           });

  // LVR at aligned+4 loads 4 bytes into the high end of the vector.
  test.Run([&](PPCContext* ctx) { ctx->r[4] = aligned_addr + 4; },
           [&](PPCContext* ctx) {
             // Should have loaded some data (not all zero).
             REQUIRE((ctx->v[3].u32[0] | ctx->v[3].u32[1] | ctx->v[3].u32[2] |
                      ctx->v[3].u32[3]) != 0);
           });

  test.memory->SystemHeapFree(guest_addr);
}

// ============================================================================
// STVL / STVR - partial vector store left/right
// These are used by optimized memcpy implementations for unaligned heads/tails.
// ============================================================================
constexpr std::array<uint8_t, 16> kStoreVectorSource = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#if XE_ARCH_ARM64
DECLARE_bool(a64_chunked_partial_vector_stores);
#endif

// Every offset, with the 16 bytes on each side of the target block checked
// untouched. On a64 both the chunked stores and the byte loop run.
static void TestPartialVectorStore(bool left) {
#if XE_ARCH_ARM64
  const bool chunked_modes[] = {true, false};
#else
  const bool chunked_modes[] = {true};
#endif
  for (const bool chunked : chunked_modes) {
#if XE_ARCH_ARM64
    cvars::a64_chunked_partial_vector_stores = chunked;
#else
    (void)chunked;
#endif
    TestFunction test([left](HIRBuilder& b) {
      if (left) {
        b.StoreVectorLeft(LoadGPR(b, 4), LoadVR(b, 5));
      } else {
        b.StoreVectorRight(LoadGPR(b, 4), LoadVR(b, 5));
      }
      b.Return();
    });
    for (uint32_t offset = 0; offset < 16; ++offset) {
      uint32_t guest_addr = test.memory->SystemHeapAlloc(80, 16);
      REQUIRE(guest_addr != 0);
      const uint32_t window_addr = (guest_addr + 15) & ~15u;
      const uint32_t block_addr = window_addr + 16;
      auto* window = reinterpret_cast<uint8_t*>(
          test.memory->TranslateVirtual(window_addr));

      std::array<uint8_t, 48> expected;
      for (uint32_t i = 0; i < 48; ++i) {
        window[i] = static_cast<uint8_t>(0xA0 + i);
        expected[i] = window[i];
      }
      if (left) {
        for (uint32_t i = offset; i < 16; ++i) {
          expected[16 + i] = kStoreVectorSource[i - offset];
        }
      } else {
        for (uint32_t i = 0; i < offset; ++i) {
          expected[16 + i] = kStoreVectorSource[16 - offset + i];
        }
      }

      test.Run(
          [&](PPCContext* ctx) {
            ctx->r[4] = block_addr + offset;
            ctx->v[5] =
                vec128b(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
          },
          [&](PPCContext* ctx) {
            INFO("offset " << offset << " chunked " << chunked);
            REQUIRE(std::memcmp(window, expected.data(), expected.size()) == 0);
          });

      test.memory->SystemHeapFree(guest_addr);
    }
  }
#if XE_ARCH_ARM64
  cvars::a64_chunked_partial_vector_stores = true;
#endif
}

TEST_CASE("STORE_VECTOR_LEFT", "[memory]") { TestPartialVectorStore(true); }

TEST_CASE("STORE_VECTOR_RIGHT", "[memory]") { TestPartialVectorStore(false); }

TEST_CASE("STORE_VECTOR_LEFT_MEMCPY_HEAD", "[memory]") {
  TestFunction test([](HIRBuilder& b) {
    auto src_addr = LoadGPR(b, 4);
    auto dest_addr = LoadGPR(b, 5);
    auto aligned_src =
        b.And(src_addr, b.LoadConstantInt64(static_cast<int64_t>(~0xFULL)));
    auto next_src = b.And(b.Add(src_addr, b.LoadConstantInt64(15)),
                          b.LoadConstantInt64(static_cast<int64_t>(~0xFULL)));
    auto source_a = b.ByteSwap(b.Load(aligned_src, VEC128_TYPE));
    auto source_b = b.ByteSwap(b.Load(next_src, VEC128_TYPE));
    auto control = b.LoadVectorShl(
        b.Truncate(b.And(src_addr, b.LoadConstantInt64(0xF)), INT8_TYPE));
    auto head = b.Permute(control, source_a, source_b, INT8_TYPE);
    b.StoreVectorLeft(dest_addr, head);
    b.Return();
  });

  for (const uint32_t source_offset : {0u, 1u, 4u, 12u}) {
    uint32_t guest_addr = test.memory->SystemHeapAlloc(128, 16);
    REQUIRE(guest_addr != 0);
    const uint32_t aligned_addr = (guest_addr + 15) & ~15u;
    auto* host_ptr =
        reinterpret_cast<uint8_t*>(test.memory->TranslateVirtual(aligned_addr));
    auto* src_ptr = host_ptr;
    auto* dest_block = host_ptr + 64;
    const uint32_t src_addr = aligned_addr + source_offset;
    const uint32_t dest_addr = aligned_addr + 64 + 12;

    for (uint32_t i = 0; i < 64; ++i) {
      src_ptr[i] = static_cast<uint8_t>(i);
    }
    for (uint32_t i = 0; i < 16; ++i) {
      dest_block[i] = static_cast<uint8_t>(0xC0 + i);
    }

    std::array<uint8_t, 16> expected;
    std::memcpy(expected.data(), dest_block, expected.size());
    for (uint32_t i = 0; i < 4; ++i) {
      expected[12 + i] = src_ptr[source_offset + i];
    }

    test.Run(
        [&](PPCContext* ctx) {
          ctx->r[4] = src_addr;
          ctx->r[5] = dest_addr;
        },
        [&](PPCContext* ctx) {
          REQUIRE(std::memcmp(dest_block, expected.data(), expected.size()) ==
                  0);
        });

    test.memory->SystemHeapFree(guest_addr);
  }
}
