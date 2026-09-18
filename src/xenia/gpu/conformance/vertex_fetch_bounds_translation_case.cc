/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Shader-translation conformance for vertex-fetch bounds masking (upstream
// Canary 9e9d3cdd3, adopted 2026-09-18).
//
// Xenos returns 0 for any fetched word at or past the size in the vertex fetch
// constant. The shared-memory binding covers all of guest physical memory, so
// without a mask the translated shader reads whatever follows the buffer. The
// 2026-09-18 upstream-delta census (XE_GPU_DELTA_CENSUS) measured this class
// on every corpus title: the dominant shape is a vfetch whose word offset
// reaches past its own stride, so the LAST vertex of every such draw reads the
// element after the buffer end.
//
// This is the structural gate on the emitted DXBC — the code a cached
// metallib freezes. A vertex shader with a single vfetch_full is assembled from
// guest microcode, run through DxbcShaderTranslator, and the token stream is
// walked for:
//
//   1. the buffer size is extracted from the fetch constant with ubfe(24, 2)
//      BEFORE any raw load;
//   2. a per-word unsigned compare (ult) against the exclusive end exists
//      between that extraction and the first raw load;
//   3. the loaded words are masked (and) AFTER the last raw load and BEFORE
//      the endian-swap switch, so every word the fetch produces passes
//      through the mask.
//
// Pre-fix translation emits none of 1-3 (the block between the load `endif`
// and the endian `switch` was empty), so each check is discriminating on its
// own.

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

// Fetch constant 95 = const_index 31, sel 2: the slot the Halo 3 sprite class
// in the census uses, and the LAST one, so an index-mapping bug that only
// works for slot 0 is visible.
constexpr uint32_t kFetchConstant = 95;

// Guest microcode: one exec-end block with a single vfetch_full writing all
// four components of r1 from index r0.x, format 32_32_32_32, stride 32 words,
// word offset 32 (past the stride — the census's dominant shape).
std::vector<uint32_t> BuildVertexFetchUcode(uint32_t fetch_constant,
                                            uint32_t stride_words,
                                            int32_t offset_words) {
  const uint32_t kExecAddress = 1;
  const uint32_t kExecCount = 1;
  const uint32_t kSequenceFetch = 0b01;
  uint32_t cf0_dword0 = (kExecAddress & 0xFFF) | ((kExecCount & 0x7) << 12) |
                        ((kSequenceFetch & 0xFFF) << 16);
  uint32_t cf0_dword1 =
      (uint32_t(ucode::ControlFlowOpcode::kExecEnd) & 0xF) << 12;

  const uint32_t const_index = fetch_constant / 3;
  const uint32_t const_index_sel = fetch_constant % 3;
  uint32_t fetch_dword0 =
      (uint32_t(ucode::FetchOpcode::kVertexFetch) & 0x1F) |
      ((0u & 0x3F) << 5) |                // src_reg r0
      ((1u & 0x3F) << 12) |               // dst_reg r1
      (1u << 19) |                        // must_be_one
      ((const_index & 0x1F) << 20) |      // const_index
      ((const_index_sel & 0x3) << 25) |   // const_index_sel
      ((0u & 0x7) << 27) |                // prefetch_count - 1
      ((0u & 0x3) << 30);                 // src_swiz x
  const uint32_t kDstSwizzleIdentity =
      (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);  // x, y, z, w
  uint32_t fetch_dword1 =
      (kDstSwizzleIdentity & 0xFFF) |
      (1u << 13) |  // num_format_all = 1: not normalized (integer/raw)
      ((uint32_t(xenos::VertexFormat::k_32_32_32_32) & 0x3F) << 16);
  uint32_t fetch_dword2 =
      (stride_words & 0xFF) | ((uint32_t(offset_words) & 0x7FFFFF) << 8);
  return {cf0_dword0, cf0_dword1, 0u, fetch_dword0, fetch_dword1, fetch_dword2};
}

struct DxbcInstruction {
  dxbc::Opcode opcode;
  std::vector<uint32_t> dwords;
};

std::vector<DxbcInstruction> DecodeShaderInstructions(
    const std::vector<uint8_t>& binary) {
  std::vector<DxbcInstruction> out;
  REQUIRE(binary.size() > sizeof(uint32_t) * 8);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(binary.data());
  const size_t word_count = binary.size() / sizeof(uint32_t);
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
    const uint32_t chunk_size_bytes = words[offset_words + 1];
    const size_t body_start = offset_words + 2;
    const size_t body_words = chunk_size_bytes / sizeof(uint32_t);
    REQUIRE(body_start + body_words <= word_count);
    size_t i = body_start + 2;
    const size_t end = body_start + body_words;
    while (i < end) {
      uint32_t token = words[i];
      uint32_t length = dxbc::GetOpcodeTokenInstructionLength(token);
      if (length == 0) {
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

// True when one of the instruction's operands is the 32-bit immediate `value`
// (single-component immediate: operand token followed by one dword).
bool HasImmediate32(const DxbcInstruction& instruction, uint32_t value) {
  for (size_t i = 1; i + 1 < instruction.dwords.size(); ++i) {
    uint32_t operand_token = instruction.dwords[i];
    uint32_t operand_type = (operand_token >> 12) & 0xFF;
    if (operand_type != uint32_t(dxbc::OperandType::kImmediate32)) {
      continue;
    }
    uint32_t num_components = operand_token & 0x3;  // 1 = one component
    if (num_components == 1 && instruction.dwords[i + 1] == value) {
      return true;
    }
  }
  return false;
}

std::vector<DxbcInstruction> TranslateVertexFetchShader(
    uint32_t fetch_constant, uint32_t stride_words, int32_t offset_words,
    std::string* disasm_out) {
  std::vector<uint32_t> ucode =
      BuildVertexFetchUcode(fetch_constant, stride_words, offset_words);
  Shader shader(xenos::ShaderType::kVertex, 0, ucode.data(), ucode.size(),
                std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);
  *disasm_out = disasm.to_string();

  DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                  /*bindless_resources_used=*/true,
                                  /*edram_rov_used=*/false,
                                  /*gamma_render_target_as_unorm8=*/true);
  uint64_t modification = translator.GetDefaultVertexShaderModification(
      shader.GetDynamicAddressableRegisterCount(xenos::kMaxShaderTempRegisters));
  Shader::Translation* translation = shader.GetOrCreateTranslation(modification);
  REQUIRE(translator.TranslateAnalyzedShader(*translation));
  return DecodeShaderInstructions(translation->translated_binary());
}

size_t FirstIndexOf(const std::vector<DxbcInstruction>& instructions,
                    dxbc::Opcode opcode, size_t from = 0) {
  for (size_t i = from; i < instructions.size(); ++i) {
    if (instructions[i].opcode == opcode) {
      return i;
    }
  }
  return SIZE_MAX;
}

size_t LastIndexOf(const std::vector<DxbcInstruction>& instructions,
                   dxbc::Opcode opcode) {
  size_t found = SIZE_MAX;
  for (size_t i = 0; i < instructions.size(); ++i) {
    if (instructions[i].opcode == opcode) {
      found = i;
    }
  }
  return found;
}

}  // namespace

TEST_CASE("The assembled vertex fetch shader is what the test intends",
          "[vertex_fetch_bounds][translation]") {
  std::string disasm;
  std::vector<DxbcInstruction> instructions = TranslateVertexFetchShader(
      kFetchConstant, /*stride_words=*/32, /*offset_words=*/32, &disasm);
  INFO(disasm);
  CHECK(disasm.find("vfetch_full") != std::string::npos);
  // The disassembler prints the constant as vf{95 - index}, so slot 95 reads
  // as vf0; assert on the fields that name the shape instead.
  CHECK(disasm.find("Offset=32") != std::string::npos);
  CHECK(disasm.find("Stride=32") != std::string::npos);
  // The shader really loads words: at least one raw load and the endian swap
  // switch are present.
  CHECK(FirstIndexOf(instructions, dxbc::Opcode::kLdRaw) != SIZE_MAX);
  CHECK(FirstIndexOf(instructions, dxbc::Opcode::kSwitch) != SIZE_MAX);
}

TEST_CASE("Vertex fetch extracts the buffer size before loading",
          "[vertex_fetch_bounds][translation]") {
  std::string disasm;
  std::vector<DxbcInstruction> instructions = TranslateVertexFetchShader(
      kFetchConstant, 32, 32, &disasm);
  INFO(disasm);
  size_t first_load = FirstIndexOf(instructions, dxbc::Opcode::kLdRaw);
  REQUIRE(first_load != SIZE_MAX);
  // ubfe(width 24, offset 2) of the second fetch constant word = size in
  // words. The literals are the contract: any other extraction is not the
  // hardware's size field.
  bool size_extracted = false;
  for (size_t i = 0; i < first_load; ++i) {
    if (instructions[i].opcode == dxbc::Opcode::kUBFE &&
        HasImmediate32(instructions[i], 24) &&
        HasImmediate32(instructions[i], 2)) {
      size_extracted = true;
      break;
    }
  }
  CHECK(size_extracted);
  // ...and turned into an exclusive byte end with umad(size, 4, base).
  bool end_computed = false;
  for (size_t i = 0; i < first_load; ++i) {
    if (instructions[i].opcode == dxbc::Opcode::kUMAd &&
        HasImmediate32(instructions[i], 4)) {
      end_computed = true;
      break;
    }
  }
  CHECK(end_computed);
}

TEST_CASE("Vertex fetch compares every word address against the end",
          "[vertex_fetch_bounds][translation]") {
  std::string disasm;
  std::vector<DxbcInstruction> instructions = TranslateVertexFetchShader(
      kFetchConstant, 32, 32, &disasm);
  INFO(disasm);
  size_t first_load = FirstIndexOf(instructions, dxbc::Opcode::kLdRaw);
  REQUIRE(first_load != SIZE_MAX);
  size_t ult = FirstIndexOf(instructions, dxbc::Opcode::kULT);
  REQUIRE(ult != SIZE_MAX);
  // The compare precedes the loads (the mask is built from the addresses, not
  // from the loaded values).
  CHECK(ult < first_load);
  // Its address operand is formed by iadd of the element address with the
  // per-word byte offsets (a 4-component immediate), immediately before.
  bool word_addresses_formed = false;
  for (size_t i = 0; i < ult; ++i) {
    if (instructions[i].opcode != dxbc::Opcode::kIAdd) {
      continue;
    }
    for (size_t k = 1; k + 4 < instructions[i].dwords.size(); ++k) {
      uint32_t token = instructions[i].dwords[k];
      if (((token >> 12) & 0xFF) ==
              uint32_t(dxbc::OperandType::kImmediate32) &&
          (token & 0x3) == 2 /* four components */) {
        // Word k's byte offset relative to the first needed word: with all
        // four words needed, the vector is 0, 4, 8, 12.
        if (instructions[i].dwords[k + 1] == 0 &&
            instructions[i].dwords[k + 2] == 4 &&
            instructions[i].dwords[k + 3] == 8 &&
            instructions[i].dwords[k + 4] == 12) {
          word_addresses_formed = true;
        }
      }
    }
  }
  CHECK(word_addresses_formed);
}

TEST_CASE("Vertex fetch masks the loaded words before the endian swap",
          "[vertex_fetch_bounds][translation]") {
  std::string disasm;
  std::vector<DxbcInstruction> instructions = TranslateVertexFetchShader(
      kFetchConstant, 32, 32, &disasm);
  INFO(disasm);
  size_t last_load = LastIndexOf(instructions, dxbc::Opcode::kLdRaw);
  REQUIRE(last_load != SIZE_MAX);
  // The vertex-index endian swap in the VS prologue is also a `switch`; the
  // fetch's own endian switch is the first one AFTER its loads.
  size_t first_switch =
      FirstIndexOf(instructions, dxbc::Opcode::kSwitch, last_load);
  REQUIRE(first_switch != SIZE_MAX);
  // Pre-fix, nothing but the load `endif` sits between the last raw load and
  // the endian switch. The mask is an `and` of the result register with the
  // per-word in-bounds mask.
  size_t mask_and = FirstIndexOf(instructions, dxbc::Opcode::kAnd, last_load);
  REQUIRE(mask_and != SIZE_MAX);
  CHECK(mask_and < first_switch);
  // And it is a register-with-register and (no immediate operand): the
  // in-bounds mask is per-word data, not a constant.
  CHECK_FALSE(HasImmediate32(instructions[mask_and], 0x00FF00FF));
}

TEST_CASE("A partial-word fetch still masks only the words it loads",
          "[vertex_fetch_bounds][translation]") {
  // Word offset 0 within a 2-word stride: the 32_32_32_32 fetch reads four
  // words whose last two lie in the NEXT element. The structure must be
  // identical; this guards against a first_word_index arithmetic regression
  // when the needed-word set is at a nonzero offset.
  std::string disasm;
  std::vector<DxbcInstruction> instructions = TranslateVertexFetchShader(
      kFetchConstant, /*stride_words=*/2, /*offset_words=*/0, &disasm);
  INFO(disasm);
  size_t first_load = FirstIndexOf(instructions, dxbc::Opcode::kLdRaw);
  size_t ult = FirstIndexOf(instructions, dxbc::Opcode::kULT);
  REQUIRE(first_load != SIZE_MAX);
  REQUIRE(ult != SIZE_MAX);
  CHECK(ult < first_load);
  size_t last_load = LastIndexOf(instructions, dxbc::Opcode::kLdRaw);
  size_t first_switch =
      FirstIndexOf(instructions, dxbc::Opcode::kSwitch, last_load);
  REQUIRE(first_switch != SIZE_MAX);
  size_t mask_and = FirstIndexOf(instructions, dxbc::Opcode::kAnd, last_load);
  REQUIRE(mask_and != SIZE_MAX);
  CHECK(mask_and < first_switch);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
