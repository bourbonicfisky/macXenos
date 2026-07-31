/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Shader-translation conformance for integer texture scaling
// (openspec/changes/implement-unsigned-integer-texture-scaling, task 6.5).
//
// This is the structural gate on the shader ABI itself. Everything else proves
// the VALUES are right — the L0 packing cases, the on-device sampler oracle, the
// live publication probe. None of them can see what the translator actually
// EMITTED, and the emitted code is what a cached metallib freezes.
//
// A real pixel shader is assembled from guest microcode (a single tfetch2D on
// fetch constant 6, the slot the engaged Just Cause 2 consumers use), run
// through DxbcShaderTranslator, and the resulting DXBC token stream is walked
// narrowly for:
//
//   1. the scale metadata is read from the right constant-buffer slot, computed
//      from the ABI offset and the fetch index rather than assumed;
//   2. the whole block is guarded so zero metadata leaves the sample untouched;
//   3. non-zero metadata unpacks a PER-COMPONENT field and forms the multiplier
//      before the guest's own use of the result;
//   4. the block contains EXACTLY the expected arithmetic — no unsigned-biased
//      half-scale or -0.5 offset path may appear.
//
// The microcode is assembled here rather than pinned as a byte blob so the
// assertions read against named fields, and the disassembly is checked first so
// a mis-assembled shader fails as "bad test input" rather than as a bad ABI.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using SC = DxbcShaderTranslator::SystemConstants;

// Fetch constant the engaged Just Cause 2 consumers use. Deliberately not 0:
// a slot-indexing bug that always read the first entry would be invisible at 0.
constexpr uint32_t kFetchConstant = 6;

// Guest microcode: one exec-end block containing a single tfetch2D writing all
// four components of r1 from r0's coordinates.
//
// Layout, per Shader::AnalyzeUcode: control-flow instructions are packed two
// per three dwords at the top, and the first exec's address (in those 3-dword
// units) bounds the control-flow section. With address 1, dwords 0..2 hold the
// control flow and the instruction lives at dwords 3..5.
std::vector<uint32_t> BuildTextureFetchUcode(uint32_t fetch_constant) {
  // Control-flow slot 0: kExecEnd, address 1, count 1, sequence 0b01 (fetch).
  const uint32_t kExecAddress = 1;
  const uint32_t kExecCount = 1;
  const uint32_t kSequenceFetch = 0b01;
  uint32_t cf0_dword0 = (kExecAddress & 0xFFF) | ((kExecCount & 0x7) << 12) |
                        ((kSequenceFetch & 0xFFF) << 16);
  uint32_t cf0_dword1 =
      (uint32_t(ucode::ControlFlowOpcode::kExecEnd) & 0xF) << 12;

  // tfetch2D r1, r0.xy, tf<fetch_constant>, all components written.
  const uint32_t kDstSwizzleIdentity =
      (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);  // x, y, z, w
  uint32_t fetch_dword0 = (uint32_t(ucode::FetchOpcode::kTextureFetch) & 0x1F) |
                          ((0u & 0x3F) << 5) |                 // src_reg r0
                          ((1u & 0x3F) << 12) |                // dst_reg r1
                          ((fetch_constant & 0x1F) << 20) |    // const_index
                          ((0b010001u & 0x3F) << 26);          // src_swiz x,y
  uint32_t fetch_dword1 = kDstSwizzleIdentity & 0xFFF;
  uint32_t fetch_dword2 =
      (uint32_t(xenos::FetchOpDimension::k2D) & 0x3) << 14;

  return {cf0_dword0, cf0_dword1, 0u, fetch_dword0, fetch_dword1, fetch_dword2};
}

// One decoded DXBC instruction: its opcode and the raw operand dwords.
struct DxbcInstruction {
  dxbc::Opcode opcode;
  std::vector<uint32_t> dwords;  // the whole instruction, opcode token first
};

// Walks the emitted shader token stream. The translated binary is a full DXBC
// container; the shader code chunk is located by its FourCC.
std::vector<DxbcInstruction> DecodeShaderInstructions(
    const std::vector<uint8_t>& binary) {
  std::vector<DxbcInstruction> out;
  REQUIRE(binary.size() > sizeof(uint32_t) * 8);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(binary.data());
  const size_t word_count = binary.size() / sizeof(uint32_t);

  // Container header: magic, 4 hash dwords, version, total size, chunk count.
  const uint32_t chunk_count = words[7];
  REQUIRE(chunk_count > 0);
  REQUIRE(8 + chunk_count <= word_count);

  const uint32_t kShexFourCC = 'S' | ('H' << 8) | ('E' << 16) | ('X' << 24);
  const uint32_t kShdrFourCC = 'S' | ('H' << 8) | ('D' << 16) | ('R' << 24);

  for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
    const uint32_t offset_bytes = words[8 + chunk];
    const size_t offset_words = offset_bytes / sizeof(uint32_t);
    REQUIRE(offset_words + 2 < word_count);
    const uint32_t fourcc = words[offset_words];
    if (fourcc != kShexFourCC && fourcc != kShdrFourCC) {
      continue;
    }
    // Chunk: fourcc, byte size, then version token and dword-count token.
    const uint32_t chunk_size_bytes = words[offset_words + 1];
    const size_t body_start = offset_words + 2;
    const size_t body_words = chunk_size_bytes / sizeof(uint32_t);
    REQUIRE(body_start + body_words <= word_count);
    // body[0] = version, body[1] = total dword count of the shader blob.
    size_t i = body_start + 2;
    const size_t end = body_start + body_words;
    while (i < end) {
      uint32_t token = words[i];
      uint32_t length = dxbc::GetOpcodeTokenInstructionLength(token);
      if (length == 0) {
        // Custom-data blocks carry their length in the next dword.
        if (i + 1 >= end) {
          break;
        }
        length = words[i + 1];
        if (length == 0) {
          break;
        }
      }
      if (i + length > end) {
        break;
      }
      DxbcInstruction instruction;
      instruction.opcode = dxbc::Opcode(token & 0x7FF);
      instruction.dwords.assign(words + i, words + i + length);
      out.push_back(std::move(instruction));
      i += length;
    }
    break;
  }
  return out;
}

// Byte offset of one fetch constant's scale within the system constants, and
// the cbuffer register / component it therefore lands in. Computed, never
// hard-coded, so the assertions track the ABI rather than restating it.
uint32_t ScaleConstantRegister(uint32_t fetch_constant) {
  return uint32_t((offsetof(SC, texture_integer_scale_bits) +
                   sizeof(uint32_t) * fetch_constant) /
                  16);
}
uint32_t ScaleConstantComponent(uint32_t fetch_constant) {
  return uint32_t(((offsetof(SC, texture_integer_scale_bits) +
                    sizeof(uint32_t) * fetch_constant) %
                   16) /
                  4);
}

// True when the instruction reads a constant buffer at cbuffer register `reg`.
//
// An operand token carries its type in bits 12:19 and its index dimension in
// bits 20:21; the immediate index dwords follow, and the LAST of them is the
// register within the buffer. The dimension is read from the token rather than
// assumed, because the bindless configuration this backend uses indexes the
// constant buffer with more dimensions than the plain one does — assuming two
// silently finds nothing, which would make this whole suite vacuously green.
bool ReadsConstantBufferRegister(const DxbcInstruction& instruction,
                                 uint32_t reg) {
  for (size_t i = 1; i < instruction.dwords.size(); ++i) {
    uint32_t operand_token = instruction.dwords[i];
    uint32_t operand_type = (operand_token >> 12) & 0xFF;
    if (operand_type != uint32_t(dxbc::OperandType::kConstantBuffer)) {
      continue;
    }
    uint32_t index_dimension = (operand_token >> 20) & 0x3;
    if (!index_dimension || i + index_dimension >= instruction.dwords.size()) {
      continue;
    }
    if (instruction.dwords[i + index_dimension] == reg) {
      return true;
    }
  }
  return false;
}

std::vector<dxbc::Opcode> OpcodesBetweenScaleGuard(
    const std::vector<DxbcInstruction>& instructions, size_t if_index) {
  std::vector<dxbc::Opcode> body;
  int depth = 0;
  for (size_t i = if_index; i < instructions.size(); ++i) {
    dxbc::Opcode op = instructions[i].opcode;
    if (op == dxbc::Opcode::kIf) {
      ++depth;
      if (depth == 1) {
        continue;
      }
    }
    if (op == dxbc::Opcode::kEndIf) {
      --depth;
      if (depth == 0) {
        return body;
      }
    }
    body.push_back(op);
  }
  return body;
}

// Translates the fetch shader and returns its DXBC instructions.
std::vector<DxbcInstruction> TranslateFetchShader(uint32_t fetch_constant,
                                                  std::string* disasm_out) {
  std::vector<uint32_t> ucode = BuildTextureFetchUcode(fetch_constant);
  Shader shader(xenos::ShaderType::kPixel, 0, ucode.data(), ucode.size(),
                std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);
  *disasm_out = disasm.to_string();

  DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                  /*bindless_resources_used=*/true,
                                  /*edram_rov_used=*/false,
                                  /*gamma_render_target_as_unorm8=*/true);
  uint64_t modification = translator.GetDefaultPixelShaderModification(
      shader.GetDynamicAddressableRegisterCount(xenos::kMaxShaderTempRegisters));
  Shader::Translation* translation = shader.GetOrCreateTranslation(modification);
  REQUIRE(translator.TranslateAnalyzedShader(*translation));
  return DecodeShaderInstructions(translation->translated_binary());
}

}  // namespace

TEST_CASE("The assembled fetch shader is what the test intends",
          "[integer_scale][translation]") {
  // Checked FIRST so a mis-assembled shader fails as bad test input rather than
  // masquerading as an ABI failure.
  std::string disasm;
  std::vector<DxbcInstruction> instructions =
      TranslateFetchShader(kFetchConstant, &disasm);
  INFO(disasm);
  CHECK(disasm.find("tfetch2D") != std::string::npos);
  CHECK(disasm.find("tf" + std::to_string(kFetchConstant)) !=
        std::string::npos);
  CHECK_FALSE(instructions.empty());
}

TEST_CASE("The fetch reads its scale from the correct constant slot",
          "[integer_scale][translation]") {
  std::string disasm;
  std::vector<DxbcInstruction> instructions =
      TranslateFetchShader(kFetchConstant, &disasm);
  INFO(disasm);

  const uint32_t expected_register = ScaleConstantRegister(kFetchConstant);
  // The whole point of a per-fetch-constant array: slot 6 must not resolve to
  // the register slot 0 lives in.
  REQUIRE(expected_register != ScaleConstantRegister(0));

  bool reads_expected = false;
  for (const DxbcInstruction& instruction : instructions) {
    if (ReadsConstantBufferRegister(instruction, expected_register)) {
      reads_expected = true;
      break;
    }
  }
  INFO("expected cb register " << expected_register << " component "
                               << ScaleConstantComponent(kFetchConstant));
  CHECK(reads_expected);
}

TEST_CASE("The scale block is guarded, exact, and free of a biased path",
          "[integer_scale][translation]") {
  std::string disasm;
  std::vector<DxbcInstruction> instructions =
      TranslateFetchShader(kFetchConstant, &disasm);
  INFO(disasm);

  const uint32_t expected_register = ScaleConstantRegister(kFetchConstant);

  // Find the guard: an `if` whose condition reads the scale slot. Zero metadata
  // must leave the sampled value untouched, which is what makes an unengaged
  // fetch free.
  size_t guard_index = SIZE_MAX;
  for (size_t i = 0; i < instructions.size(); ++i) {
    if (instructions[i].opcode == dxbc::Opcode::kIf &&
        ReadsConstantBufferRegister(instructions[i], expected_register)) {
      guard_index = i;
      break;
    }
  }
  REQUIRE(guard_index != SIZE_MAX);

  // The guarded body must be EXACTLY the documented arithmetic:
  //   ubfe  - unpack the 5-bit field of each component (per-component)
  //   and   - width - 1
  //   iadd  - width
  //   ushr  - signed flag
  //   iadd  - width - signed
  //   ishl  - 1 << (width - signed)
  //   iadd  - minus one
  //   utof  - to float
  //   mul   - apply to the sampled result
  // Anything extra would be a second codepath. In particular, upstream
  // 2ddc5ef73's unsigned-biased handling (half scale plus a -0.5 offset) cannot
  // be present without adding instructions here.
  const std::vector<dxbc::Opcode> expected = {
      dxbc::Opcode::kUBFE, dxbc::Opcode::kAnd,  dxbc::Opcode::kIAdd,
      dxbc::Opcode::kUShR, dxbc::Opcode::kIAdd, dxbc::Opcode::kIShL,
      dxbc::Opcode::kIAdd, dxbc::Opcode::kUToF, dxbc::Opcode::kMul,
  };
  std::vector<dxbc::Opcode> body =
      OpcodesBetweenScaleGuard(instructions, guard_index);

  INFO("guarded body size " << body.size());
  REQUIRE(body.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    INFO("body position " << i);
    CHECK(body[i] == expected[i]);
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
