/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Positive control for the XE_GPU_DELTA_CENSUS scalar-swizzle probe
// (ParsedAluInstruction::scalar_b_component_delta_92ada8ebc).
//
// Canary 92ada8ebc changes the second component of a two-component
// single-operand scalar ALU operation from X to Z whenever the co-issued
// vector operation uses source 3. The 2026-09-18 census measured ZERO such
// instructions whose X and Z selections differ across the 68-trace corpus,
// and the `=selftest` mode (which fires on every eligible co-issue) also
// measured zero on the Halo 3 gameplay trace: the co-issue shape itself does
// not occur there. A zero from a probe that cannot fire proves nothing, so this
// case assembles the exact shape and checks the parser sets the flag when the
// selections differ and clears it when they coincide.

#include <cstdint>
#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

// Assembles one ALU instruction: vector `mad r0.xyzw, r1, r2, r3` (three
// sources, so source 3 is used) co-issued with scalar `muls r4.x, r3.??`, a
// single-operand two-component scalar operation on source 3.
//
// Layout (ucode::AluInstruction::Data):
//   dword 0: vector_dest:6 | vector_dest_rel:1 | abs_constants:1 |
//            scalar_dest:6 | scalar_dest_rel:1 | export_data:1 |
//            vector_write_mask:4 | scalar_write_mask:4 | vector_clamp:1 |
//            scalar_clamp:1 | scalar_opc:6
//   dword 1: src3_swiz:8 | src2_swiz:8 | src1_swiz:8 | negates/pred/rel bits
//   dword 2: src3_reg:8 | src2_reg:8 | src1_reg:8 | vector_opc:5 |
//            src3_sel:1 | src2_sel:1 | src1_sel:1   (sel 1 = temp register)
ucode::AluInstruction MakeMadMuls(uint32_t src3_swizzle) {
  uint32_t d0 = (0u & 0x3F) |                                   // vector_dest r0
                (4u << 8) |                                     // scalar_dest r4
                (0xFu << 16) |                                  // vector mask
                (0x1u << 20) |                                  // scalar mask x
                ((uint32_t(ucode::AluScalarOpcode::kMuls) & 0x3F) << 26);
  uint32_t d1 = (src3_swizzle & 0xFF) | (0u << 8) | (0u << 16);
  uint32_t d2 = (3u) | (2u << 8) | (1u << 16) |
                ((uint32_t(ucode::AluVectorOpcode::kMad) & 0x1F) << 24) |
                (1u << 29) | (1u << 30) | (1u << 31);
  uint32_t words[3] = {d0, d1, d2};
  ucode::AluInstruction op;
  static_assert(sizeof(op) == sizeof(words), "AluInstruction is 3 dwords");
  std::memcpy(&op, words, sizeof(words));
  return op;
}

}  // namespace

TEST_CASE("The scalar-swizzle census fires when X and Z selections differ",
          "[delta_census][alu]") {
  // src3 swizzle bits 0:1 select b's X-relative component, bits 4:5 the
  // Z-relative one. 0b00 at 0:1 -> X; 0b01 at 4:5 -> (2 + 1) & 3 = W. Differ.
  ucode::AluInstruction op = MakeMadMuls(/*src3_swizzle=*/0b0001'0000);
  ParsedAluInstruction instr;
  ParseAluInstruction(op, xenos::ShaderType::kVertex, instr);
  REQUIRE(instr.vector_opcode == ucode::AluVectorOpcode::kMad);
  REQUIRE(instr.scalar_opcode == ucode::AluScalarOpcode::kMuls);
  REQUIRE(instr.vector_operand_count == 3);
  REQUIRE(instr.scalar_operand_count == 1);
  REQUIRE(instr.scalar_operands[0].component_count == 2);
  CHECK(instr.scalar_b_component_delta_92ada8ebc);
}

TEST_CASE("The scalar-swizzle census stays quiet when X and Z coincide",
          "[delta_census][alu]") {
  // 0b00 at 0:1 -> X; 0b10 at 4:5 -> (2 + 2) & 3 = X. Same component, so
  // Canary's rule would not change the arithmetic.
  ucode::AluInstruction op = MakeMadMuls(/*src3_swizzle=*/0b0010'0000);
  ParsedAluInstruction instr;
  ParseAluInstruction(op, xenos::ShaderType::kVertex, instr);
  REQUIRE(instr.vector_operand_count == 3);
  REQUIRE(instr.scalar_operand_count == 1);
  CHECK_FALSE(instr.scalar_b_component_delta_92ada8ebc);
}

TEST_CASE("The scalar-swizzle census ignores two-source vector co-issues",
          "[delta_census][alu]") {
  // Same swizzle as the positive case, but the vector op is `mul` (two
  // sources): source 3 is free for the scalar op, b stays X, no delta.
  ucode::AluInstruction op = MakeMadMuls(/*src3_swizzle=*/0b0001'0000);
  uint32_t words[3];
  std::memcpy(words, &op, sizeof(words));
  words[2] = (words[2] & ~(0x1Fu << 24)) |
             ((uint32_t(ucode::AluVectorOpcode::kMul) & 0x1F) << 24);
  std::memcpy(&op, words, sizeof(words));
  ParsedAluInstruction instr;
  ParseAluInstruction(op, xenos::ShaderType::kVertex, instr);
  REQUIRE(instr.vector_opcode == ucode::AluVectorOpcode::kMul);
  REQUIRE(instr.vector_operand_count == 2);
  CHECK_FALSE(instr.scalar_b_component_delta_92ada8ebc);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
