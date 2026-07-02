/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// float_divergence_suite.cc
// -------------------------
// Arch-neutral float-divergence harness for OpenSpec change
// `fix-a64-float-vertex-precision` (task 1.2). See
// openspec/changes/fix-a64-float-vertex-precision/FLOAT-OP-INVENTORY.md
// for the defect classes (D1-D8) this suite exists to expose.
//
// The suite compiles small HIR functions with the NATIVE backend of the host
// (a64 on Apple silicon, x64 on Intel) and prints result bit patterns in a
// stable, diff-able text format. There are NO assertions on float results:
// divergence is detected by diffing the output of two machines.
//
// How to run (INTERACTIVE terminal only — agent/sandbox environments SIGKILL
// Xenia binaries at exec; see project memory):
//   build/bin/macOS/Release/xenia-cpu-tests "[divergence]" \
//       > divergence-$(uname -m).txt
// on both the Apple-silicon machine (arm64) and the Intel Mac Pro (x86_64),
// then:
//   diff divergence-arm64.txt divergence-x86_64.txt
//
// The test cases are tagged hidden ("[.][divergence]") so normal test runs
// skip them; they only run when the [divergence] tag is requested explicitly.
// Expect a few minutes of runtime: every (op, context, rn, njm) combination
// compiles its own TestFunction (~215 JIT modules, ~3.4k output lines).
//
// Output format, one line per (op, input, context) combination:
//   DIV1|<op>|<variant>|<ctx>|<rn>|<njm>|<input-hex>|<output-hex>
//     op       HIR opcode name (e.g. CONVERT, ROUND, VECTOR_ADD)
//     variant  type/flag variant (F64, F64_I32_DYN, V128_RTZ, D3DCOLOR, ...)
//     ctx      ISO      = op alone in the block
//              VMX      = op emitted right after a float VECTOR_ADD in the
//                         same block (leaks VMX FPCR/MXCSR mode — defect D3)
//              HOSTCALL = VECTOR_ADD then POW2(V128) before the op; POW2 V128
//                         lowers to CallNativeSafe on BOTH backends (defect
//                         D1's stale-mode-tracking trigger)
//     rn       PPC FPSCR RN+NI value (0=nearest, 1=toward-zero, 4=nearest+NI)
//              set via a constant HIR SET_ROUNDING_MODE emitted first in the
//              block; NA when the op class ignores FPSCR
//     njm      0/1 via HIR SET_NJM (VMX non-Java mode), NA when not swept
//     hex      comma-separated words: 8 hex digits for f32/i32 (and i8,
//              zero-extended), 16 for f64/i64, v128 as 4x8-digit words
//              (u32[0]..u32[3], little-endian lane order)
// Exactly one header line is printed first: DIV0|schema=1
// Deterministic ordering: ops outer, contexts middle, inputs inner. No
// timestamps, pointers, or machine identifiers appear in the output.
//
// Inputs always flow through context registers (LoadContext of FPR/GPR/VR),
// never HIR constants, so nothing constant-folds.
//
// Ops skipped because a backend lacks a sequence (kept arch-neutral):
//   - ROUND F32: x64 ROUND_F32 is assert_impossible_sequence
//     (x64_sequences.cc:451). Covered via ROUND F64 + TO_SINGLE instead.
//   - CONVERT I32->F32: x64 CONVERT_F32_I32 (dest F32, src I32) is
//     assert_impossible_sequence (x64_sequences.cc:387).
//   - MUL_ADD/MUL_SUB F32: x64 asserts impossible (frontend emits F64 +
//     TO_SINGLE); a64 F32 handler is dead code (inventory D8 note).
//   - DIV V128: x64 DIV_V128 is assert_impossible_sequence
//     (x64_sequences.cc:1907). RECIP V128 covers vector float division.
//   - There is no VECTOR_MUL HIR opcode; vector float multiply is plain
//     OPCODE_MUL on V128 values (emitted here as op name MUL|V128).
// SET_NJM exists in HIRBuilder (hir_builder.h:305), so the NJM sweep is real.

#include "xenia/cpu/testing/util.h"

#include "xenia/base/math.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;

namespace {

// ---------------------------------------------------------------------------
// Output plumbing.
// ---------------------------------------------------------------------------

enum class Ctx { ISO, VMX, HOSTCALL };

const Ctx kIsoOnly[] = {Ctx::ISO};
const Ctx kAllCtx[] = {Ctx::ISO, Ctx::VMX, Ctx::HOSTCALL};

const char* CtxName(Ctx c) {
  switch (c) {
    case Ctx::ISO:
      return "ISO";
    case Ctx::VMX:
      return "VMX";
    default:
      return "HOSTCALL";
  }
}

void PrintHeaderOnce() {
  static bool printed = false;
  if (!printed) {
    printed = true;
    std::printf("DIV0|schema=1\n");
  }
}

void AppendHex32(std::string& s, uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%08" PRIx32, v);
  s += buf;
}

void AppendHex64(std::string& s, uint64_t v) {
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%016" PRIx64, v);
  s += buf;
}

void AppendVec(std::string& s, const vec128_t& v) {
  for (int i = 0; i < 4; ++i) {
    if (i) s += ',';
    AppendHex32(s, v.u32[i]);
  }
}

void EmitLine(const char* op, const char* variant, Ctx c, int rn, int njm,
              const std::string& in_hex, const std::string& out_hex) {
  PrintHeaderOnce();
  char rn_s[8];
  char njm_s[8];
  if (rn < 0) {
    std::snprintf(rn_s, sizeof(rn_s), "NA");
  } else {
    std::snprintf(rn_s, sizeof(rn_s), "%d", rn);
  }
  if (njm < 0) {
    std::snprintf(njm_s, sizeof(njm_s), "NA");
  } else {
    std::snprintf(njm_s, sizeof(njm_s), "%d", njm);
  }
  std::printf("DIV1|%s|%s|%s|%s|%s|%s|%s\n", op, variant, CtxName(c), rn_s,
              njm_s, in_hex.c_str(), out_hex.c_str());
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Context preludes (all emitted BEFORE the op under test, same HIR block).
// ---------------------------------------------------------------------------

// VMX: a genuine float vector op puts the backend's lazy FPCR/MXCSR mode
// tracking into Vmx state. HOSTCALL: additionally a POW2 V128, which lowers
// to CallNativeSafe on both backends — the D1 stale-tracking trigger.
void EmitCtxPrelude(HIRBuilder& b, Ctx c) {
  if (c == Ctx::ISO) {
    return;
  }
  StoreVR(b, 10, b.VectorAdd(LoadVR(b, 11), LoadVR(b, 12), FLOAT32_TYPE));
  if (c == Ctx::HOSTCALL) {
    StoreVR(b, 9, b.Pow2(LoadVR(b, 13)));
  }
}

// Benign normal values for the prelude operands.
void SetPreludeInputs(PPCContext* ctx) {
  ctx->v[11] = vec128f(1.0f);
  ctx->v[12] = vec128f(1.0f);
  ctx->v[13] = vec128f(1.0f);
}

// ---------------------------------------------------------------------------
// Input bit-pattern tables.
// ---------------------------------------------------------------------------

// ~21 f32 patterns: +-0, min/max denorm, FLT_MIN, 1.0 +- 1ulp, ~pi/2,
// 123.456..., 2^24 boundary, FLT_MAX, +-inf, NaN payloads (quiet/signaling),
// 1e-8, -100.
const uint64_t kF32Inputs[] = {
    0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x3F800000,
    0x3F800001, 0x3F7FFFFF, 0x3FC90FDB, 0x42F6E979, 0x4B7FFFFF, 0x4B800000,
    0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F812345,
    0x7FA00000, 0x322BCC77, 0xC2C80000,
};

// ~18 f64 patterns: +-0, min/max denorm, DBL_MIN, 1.0 +- 1ulp, pi/2, 2^53
// boundary, DBL_MAX, +-inf, NaN payloads (quiet/signaling), 1e-8, -100.
const uint64_t kF64Inputs[] = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull,
    0x800FFFFFFFFFFFFFull, 0x0010000000000000ull, 0x3FF0000000000000ull,
    0x3FF0000000000001ull, 0x3FEFFFFFFFFFFFFFull, 0x3FF921FB54442D18ull,
    0x4340000000000000ull, 0x433FFFFFFFFFFFFFull, 0x4340000000000001ull,
    0x7FEFFFFFFFFFFFFFull, 0x7FF0000000000000ull, 0xFFF0000000000000ull,
    0x7FF8000000000000ull, 0x7FF0000000012345ull, 0x3E45798EE2308C3Aull,
    0xC059000000000000ull,
};

// i64 patterns for CONVERT I64->F64: 0, +-1, 2^53-1/2^53/2^53+1 (rounding
// boundary), INT64_MAX/MIN.
const uint64_t kI64Inputs[] = {
    0x0000000000000000ull, 0x0000000000000001ull, 0xFFFFFFFFFFFFFFFFull,
    0x001FFFFFFFFFFFFFull, 0x0020000000000000ull, 0x0020000000000001ull,
    0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
};

// V128 inputs: splats of interesting f32s, mixed denorm/normal lanes,
// NaN-in-one-lane, near-overflow lanes (dot products), ulp-dense lanes, and
// integer-boundary patterns (serve VECTOR_CONVERT_I2F/F2I).
const vec128_t kV128Inputs[] = {
    vec128i(0x3F800000),  // all 1.0f
    vec128i(0x00000001),  // all min denorm
    vec128i(0x7F800000),  // all +inf
    vec128i(0x7FC00000),  // all QNaN
    vec128i(0x00000001, 0x3F800000, 0x80000001, 0x40000000),  // denorm mix
    vec128i(0x7FC00000, 0x3F800000, 0x40000000, 0x40400000),  // NaN one lane
    vec128i(0x7F000000),  // all 2^127: dot-product overflow
    vec128i(0x00000000, 0x80000000, 0x3F800000, 0xBF800000),  // signed zeros
    vec128i(0x3F800001, 0x3F7FFFFF, 0x3FC90FDB, 0x322BCC77),  // ulp-dense
    vec128i(0x7F7FFFFF, 0x00800000, 0xC2C80000, 0x4B800000),  // big/small mix
    vec128i(0x4F000000, 0xCF000000, 0x4EFFFFFF, 0xBF800000),  // 2^31 F2I edge
    vec128i(0x7FFFFFFF, 0x80000000, 0x00FFFFFF, 0x01000001),  // int patterns
};

// ---------------------------------------------------------------------------
// Scalar driver.
// ---------------------------------------------------------------------------

enum class SKind { kF32, kF64, kI32, kI64, kI8 };

bool KindIsFloatReg(SKind k) { return k == SKind::kF32 || k == SKind::kF64; }
bool KindIsWide(SKind k) { return k == SKind::kF64 || k == SKind::kI64; }

// Inputs come from f4..f6 / r4..r6; the result goes to f3 / r3. Inputs must
// flow through context registers, never constants, so nothing folds.
Value* LoadScalarInput(HIRBuilder& b, SKind k, int index) {
  size_t f_off = offsetof(PPCContext, f) + (4 + index) * 8;
  size_t r_off = offsetof(PPCContext, r) + (4 + index) * 8;
  switch (k) {
    case SKind::kF64:
      return b.LoadContext(f_off, FLOAT64_TYPE);
    case SKind::kF32:
      return b.LoadContext(f_off, FLOAT32_TYPE);
    case SKind::kI32:
      return b.LoadContext(r_off, INT32_TYPE);
    default:
      return b.LoadContext(r_off, INT64_TYPE);
  }
}

void StoreScalarResult(HIRBuilder& b, SKind k, Value* v) {
  switch (k) {
    case SKind::kF64:
    case SKind::kF32:
      b.StoreContext(offsetof(PPCContext, f) + 3 * 8, v);
      break;
    case SKind::kI64:
      StoreGPR(b, 3, v);
      break;
    default:  // kI32 / kI8: zero-extend (integer op, arch-neutral).
      StoreGPR(b, 3, b.ZeroExtend(v, INT64_TYPE));
      break;
  }
}

using ScalarGen = Value* (*)(HIRBuilder&, Value* const*);

struct ScalarOpDef {
  const char* op;
  const char* variant;
  int arity;        // 1..3, all inputs share in_kind
  SKind in_kind;
  SKind out_kind;
  bool rn_sweep;    // D3-sensitive: sweep rn {0,1,4}; else single pass, rn=NA
  bool ctx_sweep;   // D3-class: ISO+VMX+HOSTCALL; else ISO only
  ScalarGen gen;
};

void RunScalarOp(const ScalarOpDef& d, const uint64_t* inputs, size_t n) {
  static const int kRnSweep[] = {0, 1, 4};
  static const int kRnNA[] = {-1};
  const Ctx* ctxs = d.ctx_sweep ? kAllCtx : kIsoOnly;
  size_t ctx_count = d.ctx_sweep ? 3 : 1;
  const int* rns = d.rn_sweep ? kRnSweep : kRnNA;
  size_t rn_count = d.rn_sweep ? 3 : 1;

  for (size_t ci = 0; ci < ctx_count; ++ci) {
    Ctx c = ctxs[ci];
    for (size_t ri = 0; ri < rn_count; ++ri) {
      int rn = rns[ri];
      TestFunction test([&d, c, rn](HIRBuilder& b) {
        if (rn >= 0) {
          b.SetRoundingMode(b.LoadConstantInt32(rn));
        }
        EmitCtxPrelude(b, c);
        Value* src[3] = {nullptr, nullptr, nullptr};
        for (int a = 0; a < d.arity; ++a) {
          src[a] = LoadScalarInput(b, d.in_kind, a);
        }
        StoreScalarResult(b, d.out_kind, d.gen(b, src));
        b.Return();
      });
      for (size_t ii = 0; ii < n; ++ii) {
        // Deterministic operand tuples: each value appears once as src1,
        // paired with strided picks for src2/src3.
        uint64_t in[3];
        in[0] = inputs[ii];
        in[1] = inputs[(ii * 3 + 1) % n];
        in[2] = inputs[(ii * 5 + 2) % n];
        uint64_t out = 0;
        test.Run(
            [&](PPCContext* ctx) {
              SetPreludeInputs(ctx);
              ctx->r[3] = 0;
              std::memset(&ctx->f[3], 0, sizeof(double));
              for (int a = 0; a < d.arity; ++a) {
                if (KindIsFloatReg(d.in_kind)) {
                  std::memcpy(&ctx->f[4 + a], &in[a], sizeof(uint64_t));
                } else {
                  ctx->r[4 + a] = in[a];
                }
              }
            },
            [&](PPCContext* ctx) {
              if (KindIsFloatReg(d.out_kind)) {
                std::memcpy(&out, &ctx->f[3], sizeof(uint64_t));
              } else {
                out = ctx->r[3];
              }
            });
        std::string in_hex;
        for (int a = 0; a < d.arity; ++a) {
          if (a) in_hex += ',';
          if (KindIsWide(d.in_kind)) {
            AppendHex64(in_hex, in[a]);
          } else {
            AppendHex32(in_hex, uint32_t(in[a]));
          }
        }
        std::string out_hex;
        if (KindIsWide(d.out_kind)) {
          AppendHex64(out_hex, out);
        } else {
          AppendHex32(out_hex, uint32_t(out));
        }
        EmitLine(d.op, d.variant, c, rn, /*njm=*/-1, in_hex, out_hex);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Vector driver.
// ---------------------------------------------------------------------------

using VecGen = Value* (*)(HIRBuilder&, Value* const*);

struct VecOpDef {
  const char* op;
  const char* variant;
  int arity;       // 1..3; inputs in v4..v6, result in v3
  bool ctx_sweep;  // V128 float arithmetic: ISO+VMX+HOSTCALL
  bool njm_sweep;  // VMX float arithmetic: SET_NJM {0,1}; else NA
  VecGen gen;
};

void RunVecOp(const VecOpDef& d, const vec128_t* inputs, size_t n) {
  static const int kNjmSweep[] = {0, 1};
  static const int kNjmNA[] = {-1};
  const Ctx* ctxs = d.ctx_sweep ? kAllCtx : kIsoOnly;
  size_t ctx_count = d.ctx_sweep ? 3 : 1;
  const int* njms = d.njm_sweep ? kNjmSweep : kNjmNA;
  size_t njm_count = d.njm_sweep ? 2 : 1;

  for (size_t ci = 0; ci < ctx_count; ++ci) {
    Ctx c = ctxs[ci];
    for (size_t ni = 0; ni < njm_count; ++ni) {
      int njm = njms[ni];
      TestFunction test([&d, c, njm](HIRBuilder& b) {
        if (njm >= 0) {
          b.SetNJM(b.LoadConstantInt8(int8_t(njm)));
        }
        EmitCtxPrelude(b, c);
        Value* src[3] = {nullptr, nullptr, nullptr};
        for (int a = 0; a < d.arity; ++a) {
          src[a] = LoadVR(b, 4 + a);
        }
        StoreVR(b, 3, d.gen(b, src));
        b.Return();
      });
      for (size_t ii = 0; ii < n; ++ii) {
        vec128_t in[3];
        in[0] = inputs[ii];
        in[1] = inputs[(ii * 3 + 1) % n];
        in[2] = inputs[(ii * 5 + 2) % n];
        vec128_t out = vec128i(0);
        test.Run(
            [&](PPCContext* ctx) {
              SetPreludeInputs(ctx);
              ctx->v[3] = vec128i(0);
              for (int a = 0; a < d.arity; ++a) {
                ctx->v[4 + a] = in[a];
              }
            },
            [&](PPCContext* ctx) { out = ctx->v[3]; });
        std::string in_hex;
        for (int a = 0; a < d.arity; ++a) {
          if (a) in_hex += ',';
          AppendVec(in_hex, in[a]);
        }
        std::string out_hex;
        AppendVec(out_hex, out);
        EmitLine(d.op, d.variant, c, /*rn=*/-1, njm, in_hex, out_hex);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Op tables.
// ---------------------------------------------------------------------------

// Scalar F64 sweeps (inputs = kF64Inputs unless stated).
// rn_sweep only for D3-sensitive ops + ADD/MUL controls; ctx_sweep for the
// D3 class (CONVERT, ROUND, MAX, MIN, COMPARE).
const ScalarOpDef kScalarF64Ops[] = {
    {"ADD", "F64", 2, SKind::kF64, SKind::kF64, true, false,
     [](HIRBuilder& b, Value* const* s) { return b.Add(s[0], s[1]); }},
    {"SUB", "F64", 2, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Sub(s[0], s[1]); }},
    {"MUL", "F64", 2, SKind::kF64, SKind::kF64, true, false,
     [](HIRBuilder& b, Value* const* s) { return b.Mul(s[0], s[1]); }},
    {"DIV", "F64", 2, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Div(s[0], s[1]); }},
    {"MUL_ADD", "F64", 3, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.MulAdd(s[0], s[1], s[2]); }},
    {"MUL_SUB", "F64", 3, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.MulSub(s[0], s[1], s[2]); }},
    {"MAX", "F64", 2, SKind::kF64, SKind::kF64, false, true,
     [](HIRBuilder& b, Value* const* s) { return b.Max(s[0], s[1]); }},
    {"MIN", "F64", 2, SKind::kF64, SKind::kF64, false, true,
     [](HIRBuilder& b, Value* const* s) { return b.Min(s[0], s[1]); }},
    {"SQRT", "F64", 1, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Sqrt(s[0]); }},
    {"RECIP", "F64", 1, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Recip(s[0]); }},
    {"RSQRT", "F64", 1, SKind::kF64, SKind::kF64, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.RSqrt(s[0]); }},
    {"TO_SINGLE", "F64", 1, SKind::kF64, SKind::kF64, true, false,
     [](HIRBuilder& b, Value* const* s) { return b.ToSingle(s[0]); }},
    {"ROUND", "F64_RTZ", 1, SKind::kF64, SKind::kF64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_ZERO);
     }},
    {"ROUND", "F64_RTN", 1, SKind::kF64, SKind::kF64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_NEAREST);
     }},
    {"ROUND", "F64_RTMI", 1, SKind::kF64, SKind::kF64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_MINUS_INFINITY);
     }},
    {"ROUND", "F64_RTPI", 1, SKind::kF64, SKind::kF64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_POSITIVE_INFINITY);
     }},
    {"ROUND", "F64_DYN", 1, SKind::kF64, SKind::kF64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_DYNAMIC);
     }},
    {"COMPARE_EQ", "F64", 2, SKind::kF64, SKind::kI8, false, true,
     [](HIRBuilder& b, Value* const* s) { return b.CompareEQ(s[0], s[1]); }},
    {"COMPARE_SLT", "F64", 2, SKind::kF64, SKind::kI8, false, true,
     [](HIRBuilder& b, Value* const* s) { return b.CompareSLT(s[0], s[1]); }},
    {"CONVERT", "F64_I32_DYN", 1, SKind::kF64, SKind::kI32, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Convert(s[0], INT32_TYPE, ROUND_DYNAMIC);
     }},
    {"CONVERT", "F64_I32_RTZ", 1, SKind::kF64, SKind::kI32, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Convert(s[0], INT32_TYPE, ROUND_TO_ZERO);
     }},
    {"CONVERT", "F64_I64_DYN", 1, SKind::kF64, SKind::kI64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Convert(s[0], INT64_TYPE, ROUND_DYNAMIC);
     }},
    {"CONVERT", "F64_I64_RTZ", 1, SKind::kF64, SKind::kI64, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Convert(s[0], INT64_TYPE, ROUND_TO_ZERO);
     }},
};

// CONVERT from integer sources (separate input table).
const ScalarOpDef kConvertI64F64 = {
    "CONVERT", "I64_F64", 1, SKind::kI64, SKind::kF64, true, true,
    [](HIRBuilder& b, Value* const* s) {
      return b.Convert(s[0], FLOAT64_TYPE);
    }};

// Scalar F32 (FLOAT32_TYPE context loads from FPR slots).
// Skipped, with reasons (see file header): ROUND F32 and CONVERT I32->F32
// are assert_impossible on the x64 oracle; MUL_ADD/MUL_SUB F32 likewise.
const ScalarOpDef kScalarF32Ops[] = {
    {"ADD", "F32", 2, SKind::kF32, SKind::kF32, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Add(s[0], s[1]); }},
    {"MUL", "F32", 2, SKind::kF32, SKind::kF32, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Mul(s[0], s[1]); }},
    {"RECIP", "F32", 1, SKind::kF32, SKind::kF32, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Recip(s[0]); }},
    // Expected uniform diff: a64 = exact fsqrt+fdiv, x64 = vrsqrt14ss
    // estimate on AVX512 hosts (inventory section 5.1).
    {"RSQRT", "F32", 1, SKind::kF32, SKind::kF32, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.RSqrt(s[0]); }},
    {"CONVERT", "F32_I32_DYN", 1, SKind::kF32, SKind::kI32, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Convert(s[0], INT32_TYPE, ROUND_DYNAMIC);
     }},
    {"CONVERT", "F32_I32_RTZ", 1, SKind::kF32, SKind::kI32, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.Convert(s[0], INT32_TYPE, ROUND_TO_ZERO);
     }},
};

// V128 float arithmetic: every op gets ISO; all get VMX+HOSTCALL; the
// PPC-NJM-relevant subset also sweeps NJM {0,1}.
const VecOpDef kVecArithOps[] = {
    {"VECTOR_ADD", "V128F32", 2, true, true,
     [](HIRBuilder& b, Value* const* s) {
       return b.VectorAdd(s[0], s[1], FLOAT32_TYPE);
     }},
    {"VECTOR_SUB", "V128F32", 2, true, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.VectorSub(s[0], s[1], FLOAT32_TYPE);
     }},
    // No VECTOR_MUL HIR op exists; vector float multiply is OPCODE_MUL V128.
    {"MUL", "V128", 2, true, false,
     [](HIRBuilder& b, Value* const* s) { return b.Mul(s[0], s[1]); }},
    {"MUL_ADD", "V128", 3, true, true,
     [](HIRBuilder& b, Value* const* s) { return b.MulAdd(s[0], s[1], s[2]); }},
    {"MUL_SUB", "V128", 3, true, false,
     [](HIRBuilder& b, Value* const* s) { return b.MulSub(s[0], s[1], s[2]); }},
    {"VECTOR_MAX", "V128F32", 2, true, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.VectorMax(s[0], s[1], FLOAT32_TYPE);
     }},
    {"VECTOR_MIN", "V128F32", 2, true, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.VectorMin(s[0], s[1], FLOAT32_TYPE);
     }},
    {"RECIP", "V128", 1, true, true,
     [](HIRBuilder& b, Value* const* s) { return b.Recip(s[0]); }},
    {"RSQRT", "V128", 1, true, true,
     [](HIRBuilder& b, Value* const* s) { return b.RSqrt(s[0]); }},
    {"SQRT", "V128", 1, true, false,
     [](HIRBuilder& b, Value* const* s) { return b.Sqrt(s[0]); }},
    {"DOT_PRODUCT_3", "V128", 2, true, true,
     [](HIRBuilder& b, Value* const* s) { return b.DotProduct3(s[0], s[1]); }},
    {"DOT_PRODUCT_4", "V128", 2, true, true,
     [](HIRBuilder& b, Value* const* s) { return b.DotProduct4(s[0], s[1]); }},
};

// V128 converts / rounds / transcendentals / pack-unpack: ISO only (both
// backends declare VMX mode for these, or they are FPCR-immune / are
// themselves the host call).
const VecOpDef kVecMiscOps[] = {
    {"VECTOR_CONVERT_I2F", "V128_S", 1, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.VectorConvertI2F(s[0]); }},
    {"VECTOR_CONVERT_I2F", "V128_U", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.VectorConvertI2F(s[0], ARITHMETIC_UNSIGNED);
     }},
    {"VECTOR_CONVERT_F2I", "V128_S", 1, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.VectorConvertF2I(s[0]); }},
    {"VECTOR_CONVERT_F2I", "V128_U", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.VectorConvertF2I(s[0], ARITHMETIC_UNSIGNED);
     }},
    {"ROUND", "V128_RTZ", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_ZERO);
     }},
    {"ROUND", "V128_RTN", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_NEAREST);
     }},
    {"ROUND", "V128_RTMI", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_MINUS_INFINITY);
     }},
    {"ROUND", "V128_RTPI", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_TO_POSITIVE_INFINITY);
     }},
    {"ROUND", "V128_DYN", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Round(s[0], ROUND_DYNAMIC);
     }},
    // POW2/LOG2 V128 lower to CallNativeSafe(host libm) on both backends
    // (D1 carriers; expect uniform libm ulp diffs — inventory section 5.3).
    {"POW2", "V128", 1, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Pow2(s[0]); }},
    {"LOG2", "V128", 1, false, false,
     [](HIRBuilder& b, Value* const* s) { return b.Log2(s[0]); }},
    // PACK float-relevant variants (D4 class) + matching UNPACKs.
    {"PACK", "D3DCOLOR", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Pack(s[0], PACK_TYPE_D3DCOLOR);
     }},
    {"PACK", "SHORT_2", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Pack(s[0], PACK_TYPE_SHORT_2);
     }},
    {"PACK", "SHORT_4", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Pack(s[0], PACK_TYPE_SHORT_4);
     }},
    {"PACK", "FLOAT16_2", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Pack(s[0], PACK_TYPE_FLOAT16_2);
     }},
    {"PACK", "FLOAT16_4", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Pack(s[0], PACK_TYPE_FLOAT16_4);
     }},
    {"UNPACK", "D3DCOLOR", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Unpack(s[0], PACK_TYPE_D3DCOLOR);
     }},
    {"UNPACK", "SHORT_2", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Unpack(s[0], PACK_TYPE_SHORT_2);
     }},
    {"UNPACK", "SHORT_4", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Unpack(s[0], PACK_TYPE_SHORT_4);
     }},
    {"UNPACK", "FLOAT16_2", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Unpack(s[0], PACK_TYPE_FLOAT16_2);
     }},
    {"UNPACK", "FLOAT16_4", 1, false, false,
     [](HIRBuilder& b, Value* const* s) {
       return b.Unpack(s[0], PACK_TYPE_FLOAT16_4);
     }},
};

}  // namespace

// ---------------------------------------------------------------------------
// Test cases (hidden: only run via `xenia-cpu-tests "[divergence]"`).
// ---------------------------------------------------------------------------

TEST_CASE("DIVERGENCE_SCALAR_F64", "[.][divergence]") {
  for (const auto& d : kScalarF64Ops) {
    RunScalarOp(d, kF64Inputs, xe::countof(kF64Inputs));
  }
  RunScalarOp(kConvertI64F64, kI64Inputs, xe::countof(kI64Inputs));
}

TEST_CASE("DIVERGENCE_SCALAR_F32", "[.][divergence]") {
  for (const auto& d : kScalarF32Ops) {
    RunScalarOp(d, kF32Inputs, xe::countof(kF32Inputs));
  }
}

TEST_CASE("DIVERGENCE_V128_ARITH", "[.][divergence]") {
  for (const auto& d : kVecArithOps) {
    RunVecOp(d, kV128Inputs, xe::countof(kV128Inputs));
  }
}

TEST_CASE("DIVERGENCE_V128_MISC", "[.][divergence]") {
  for (const auto& d : kVecMiscOps) {
    RunVecOp(d, kV128Inputs, xe::countof(kV128Inputs));
  }
}
