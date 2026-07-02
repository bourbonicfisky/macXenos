/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Regression tests pinning FPCR/MXCSR mode-discipline fixes found by the
// float divergence suite (OpenSpec change `fix-a64-float-vertex-precision`,
// defects D1/D3/D7 in FLOAT-OP-INVENTORY.md). Each case pins inputs that
// diverged on the a64 backend before the fix: the same logical computation
// must produce identical bits regardless of surrounding code context.

#include <cstring>

#include "xenia/cpu/testing/util.h"

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;

// D3: scalar CONVERT with dynamic rounding must honor guest FPSCR.RN even
// when a VMX vector op precedes it in the same HIR block (the leaked VMX
// FPCR forced round-to-nearest; 0.9999... converted to 1 instead of 0).
TEST_CASE("CONVERT_F64_I32_DYNAMIC_AFTER_VMX_OP", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    b.SetRoundingMode(b.LoadConstantInt32(1));  // FPSCR.RN=1: toward zero
    // VMX-mode prelude: put the backend's lazy FPCR tracking into Vmx.
    StoreVR(b, 10, b.VectorAdd(LoadVR(b, 11), LoadVR(b, 12), FLOAT32_TYPE));
    auto src = b.LoadContext(offsetof(PPCContext, f) + 4 * 8, FLOAT64_TYPE);
    StoreGPR(b, 3,
             b.ZeroExtend(b.Convert(src, INT32_TYPE, ROUND_DYNAMIC),
                          INT64_TYPE));
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[11] = vec128f(1.0f);
        ctx->v[12] = vec128f(1.0f);
        ctx->f[4] = 0.99999999999999989;  // 0x3FEFFFFFFFFFFFFF
      },
      [](PPCContext* ctx) {
        // Toward zero: must truncate to 0 (was 1 under leaked VMX mode).
        REQUIRE(ctx->r[3] == 0ull);
      });
}

// D3: ROUND_DYNAMIC (frin) must use the guest rounding mode after a VMX op.
TEST_CASE("ROUND_F64_DYNAMIC_AFTER_VMX_OP", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    b.SetRoundingMode(b.LoadConstantInt32(1));  // toward zero
    StoreVR(b, 10, b.VectorAdd(LoadVR(b, 11), LoadVR(b, 12), FLOAT32_TYPE));
    auto src = b.LoadContext(offsetof(PPCContext, f) + 4 * 8, FLOAT64_TYPE);
    b.StoreContext(offsetof(PPCContext, f) + 3 * 8,
                   b.Round(src, ROUND_DYNAMIC));
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[11] = vec128f(1.0f);
        ctx->v[12] = vec128f(1.0f);
        ctx->f[4] = 1.5707963267948966;  // pi/2
      },
      [](PPCContext* ctx) {
        // Toward zero: 1.0 (was 2.0 under leaked VMX round-to-nearest).
        REQUIRE(ctx->f[3] == 1.0);
      });
}

// D7: both-constant scalar MAX/MIN reach the backend (float constant
// folding is disabled by default) and previously computed against an
// uninitialized register.
TEST_CASE("MAX_MIN_F64_BOTH_CONSTANT", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    b.StoreContext(offsetof(PPCContext, f) + 3 * 8,
                   b.Max(b.LoadConstantFloat64(2.0),
                         b.LoadConstantFloat64(3.0)));
    b.StoreContext(offsetof(PPCContext, f) + 5 * 8,
                   b.Min(b.LoadConstantFloat64(-2.0),
                         b.LoadConstantFloat64(-3.0)));
    b.Return();
  });
  test.Run([](PPCContext* ctx) {},
           [](PPCContext* ctx) {
             REQUIRE(ctx->f[3] == 3.0);
             REQUIRE(ctx->f[5] == -3.0);
           });
}

// Cross-backend oracle findings (divergence suite, 2026-07-02):
// UNPACK SHORT_4 word order must match the x64 backend / vupkd3d: shorts
// h[5],h[4],h[7],h[6] of the source map to lanes 0..3 (the a64 backend
// previously swapped the 64-bit halves), and PPC fctiwz converts NaN to
// 0x80000000 (both backends previously got this wrong, differently).
TEST_CASE("UNPACK_SHORT_4_LANE_ORDER", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 3, b.Unpack(LoadVR(b, 4), PACK_TYPE_SHORT_4));
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128i(0x00000000, 0x80000000, 0x3F800000, 0xBF800000);
      },
      [](PPCContext* ctx) {
        REQUIRE(ctx->v[3] ==
                vec128i(0x40403F80, 0x40400000, 0x403FBF80, 0x40400000));
      });
}

#if XE_ARCH_ARM64
// arm64-gated: the x64 backend converts NaN to 0x7FFFFFFF via its
// saturation clamp (recorded as an upstream divergence from PPC fctiwz).
TEST_CASE("CONVERT_NAN_TO_INT_PPC", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    auto src = b.LoadContext(offsetof(PPCContext, f) + 4 * 8, FLOAT64_TYPE);
    StoreGPR(b, 3, b.ZeroExtend(b.Convert(src, INT32_TYPE, ROUND_TO_ZERO),
                                INT64_TYPE));
    StoreGPR(b, 5, b.Convert(src, INT64_TYPE, ROUND_TO_ZERO));
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        uint64_t snan = 0x7FF0000000012345ull;
        std::memcpy(&ctx->f[4], &snan, sizeof(snan));
      },
      [](PPCContext* ctx) {
        REQUIRE(ctx->r[3] == 0x80000000ull);
        REQUIRE(ctx->r[5] == 0x8000000000000000ull);
      });
}
#endif  // XE_ARCH_ARM64

// Lowering-path coherence: DOT_PRODUCT with a constant operand must equal
// the register form. Before the fix, constants materialized into scratch
// v0/v1 were clobbered by the low-half products before the high-half
// widening read them (a64 D-P sequences; found by the const-form suite:
// dp3((2^127 splat), {0,-0,1,-1}) returned 0 instead of 2^127).
TEST_CASE("DOT_PRODUCT_CONSTANT_OPERAND_FORM", "[instr]") {
  vec128_t k2_127 = vec128i(0x7F000000);  // 2^127 splat
  TestFunction test([&](HIRBuilder& b) {
    StoreVR(b, 3, b.DotProduct3(b.LoadConstantVec128(k2_127), LoadVR(b, 4)));
    StoreVR(b, 5, b.DotProduct4(LoadVR(b, 4), b.LoadConstantVec128(k2_127)));
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128i(0x00000000, 0x80000000, 0x3F800000, 0xBF800000);
      },
      [](PPCContext* ctx) {
        // dp3 = 2^127*0 + 2^127*(-0) + 2^127*1 = 2^127, splatted.
        REQUIRE(ctx->v[3] == vec128i(0x7F000000));
        // dp4 adds 2^127*(-1): exact sum 0.
        REQUIRE(ctx->v[5] == vec128i(0x00000000));
      });
}

#if XE_ARCH_ARM64
// D1: VMX float arithmetic must keep flushing denormals (NJM default on)
// after a host call (POW2 V128 lowers to CallNativeSafe) in the same block.
// Before the fix, the stale mode tracking skipped the FPCR switch and the
// second vector op ran without FZ. a64-only: the x64 backend has the same
// defect unfixed (bisect-neutral; tracked for upstream).
TEST_CASE("VECTOR_ADD_DENORM_FLUSH_AFTER_HOST_CALL", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    StoreVR(b, 10, b.VectorAdd(LoadVR(b, 11), LoadVR(b, 12), FLOAT32_TYPE));
    StoreVR(b, 9, b.Pow2(LoadVR(b, 13)));
    StoreVR(b, 3, b.VectorAdd(LoadVR(b, 4), LoadVR(b, 5), FLOAT32_TYPE));
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[11] = vec128f(1.0f);
        ctx->v[12] = vec128f(1.0f);
        ctx->v[13] = vec128f(1.0f);
        ctx->v[4] = vec128i(0x00000001);  // min denormal, all lanes
        ctx->v[5] = vec128i(0x00000001);
      },
      [](PPCContext* ctx) {
        // FZ: denormal inputs flush, sum is +0 (was 0x00000002 unflushed).
        REQUIRE(ctx->v[3] == vec128i(0x00000000));
      });
}

// D6: vrsqrtefp results must survive routing through the host-call thunk
// (the raw blr previously used could clobber live q registers). Pins the
// PowerISA table value for 4.0f and keeps an unrelated vector live across
// the op.
TEST_CASE("RSQRT_V128_TABLE_AND_LIVE_REGS", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    auto live = LoadVR(b, 6);
    StoreVR(b, 3, b.RSqrt(LoadVR(b, 4)));
    StoreVR(b, 5, live);  // keep an unrelated value live across the call
    b.Return();
  });
  test.Run(
      [](PPCContext* ctx) {
        ctx->v[4] = vec128f(4.0f);
        ctx->v[6] = vec128i(0x11223344, 0x55667788, 0x99AABBCC, 0xDDEEFF00);
      },
      [](PPCContext* ctx) {
        // vrsqrtefp(4.0) per the 32-entry PowerISA table.
        REQUIRE(ctx->v[3].u32[0] == ctx->v[3].u32[1]);
        REQUIRE(ctx->v[3].u32[0] == ctx->v[3].u32[2]);
        REQUIRE(ctx->v[3].u32[0] == ctx->v[3].u32[3]);
        // 1/sqrt(4) = 0.5 within vrsqrtefp precision (< 1/4096 rel. error).
        float r;
        std::memcpy(&r, &ctx->v[3].u32[0], sizeof(r));
        REQUIRE(r > 0.4998f);
        REQUIRE(r < 0.5002f);
        // The live vector must be untouched by the host call.
        REQUIRE(ctx->v[5] ==
                vec128i(0x11223344, 0x55667788, 0x99AABBCC, 0xDDEEFF00));
      });
}
#endif  // XE_ARCH_ARM64
