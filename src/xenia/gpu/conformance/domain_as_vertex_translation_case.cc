/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-host-tessellation task 2.4 — emitted-DXBC structure oracle
// for the `domain_as_vertex` vertex-shader modification (design D4, D9.2).
//
//   1. With the bit CLEAR, every domain host vertex shader type still emits
//      the DOMAIN shader it emitted before this change: the whole DXBC
//      container hashes are pinned from a build predating the translator
//      edit (captured 2026-09-20 on the tip before task 2.1 landed), so the
//      D3D12 path and the retained mesh route are provably untouched.
//   2. With the bit SET, the same guest program becomes a `vs_5_1`: the code
//      chunk's version token says vertex shader, the input signature is
//      SV_VertexID only, no patch-constant signature (PCSG) chunk exists, one
//      raw SRV (`xe_tessellation_vertices`, t1 / space 0) is declared and
//      read twice with ld_raw, and no vDomain / vICP input is declared.
//
// The checks parse the DXBC container directly (chunk table, RDEF/ISGN
// strings, SHEX opcode stream) — there is no disassembler on this platform.

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Same guest program as the tessellation render oracle's domain payload
// (metal_tessellation_render_conformance_test.mm): two temps, r0 read as the
// domain inputs, oPos.xy01 exported.
std::vector<uint32_t> BuildDomainPayloadUcode() {
  std::vector<uint32_t> u(12, 0);
  const uint32_t exec_d0 = 2u | (1u << 12);
  const uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExec) << 12;
  const uint32_t alloc_d0 = 0;
  const uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kVsPosition) << 9) |
                            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  const uint32_t exece_d0 = 3u | (1u << 12);
  const uint32_t exece_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;
  u[0] = exec_d0;
  u[1] = (exec_d1 & 0xFFFF) | ((alloc_d0 & 0xFFFF) << 16);
  u[2] = ((alloc_d0 >> 16) & 0xFFFF) | ((alloc_d1 & 0xFFFF) << 16);
  u[3] = exece_d0;
  u[4] = (exece_d1 & 0xFFFF);
  u[5] = 0;
  const uint32_t retain_prev =
      uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26;
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t src12_temp = (1u << 30) | (1u << 31);
  u[6] = 1u | (0b0011u << 16) | retain_prev;
  u[7] = 0;
  u[8] = kmax | src12_temp;
  u[9] = 62u | (1u << 14) | (1u << 15) | (0b1011u << 16) | (0b1000u << 20) |
         retain_prev;
  u[10] = 0;
  u[11] = kmax | src12_temp | (1u << 16) | (1u << 8);
  return u;
}

uint64_t Fnv1a(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 0xCBF29CE484222325ull;
  for (uint8_t b : bytes) {
    hash ^= b;
    hash *= 0x100000001B3ull;
  }
  return hash;
}

constexpr uint32_t FourCC(const char (&s)[5]) {
  return uint32_t(uint8_t(s[0])) | (uint32_t(uint8_t(s[1])) << 8) |
         (uint32_t(uint8_t(s[2])) << 16) | (uint32_t(uint8_t(s[3])) << 24);
}

struct DxbcInstruction {
  dxbc::Opcode opcode;
  std::vector<uint32_t> dwords;  // Including the opcode token.
};

struct ParsedDxbc {
  std::vector<uint8_t> binary;
  bool has_chunk(uint32_t fourcc) const {
    for (uint32_t c : chunks) {
      if (c == fourcc) return true;
    }
    return false;
  }
  bool contains_string(const char* needle) const {
    size_t n = std::strlen(needle);
    if (binary.size() < n) return false;
    for (size_t i = 0; i + n <= binary.size(); ++i) {
      if (std::memcmp(binary.data() + i, needle, n) == 0) return true;
    }
    return false;
  }
  size_t count_opcode(dxbc::Opcode opcode) const {
    size_t n = 0;
    for (const DxbcInstruction& i : instructions) {
      if (i.opcode == opcode) ++n;
    }
    return n;
  }
  // Operand type of the first operand of every instruction with `opcode`.
  std::vector<dxbc::OperandType> first_operand_types(
      dxbc::Opcode opcode) const {
    std::vector<dxbc::OperandType> out;
    for (const DxbcInstruction& i : instructions) {
      if (i.opcode == opcode && i.dwords.size() > 1) {
        out.push_back(dxbc::OperandType((i.dwords[1] >> 12) & 0xFF));
      }
    }
    return out;
  }
  std::vector<uint32_t> chunks;
  uint32_t program_type = UINT32_MAX;
  std::vector<DxbcInstruction> instructions;
};

ParsedDxbc Parse(const std::vector<uint8_t>& binary) {
  ParsedDxbc p;
  p.binary = binary;
  REQUIRE(binary.size() > sizeof(uint32_t) * 8);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(binary.data());
  const size_t word_count = binary.size() / sizeof(uint32_t);
  const uint32_t chunk_count = words[7];
  REQUIRE(8 + chunk_count <= word_count);
  for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t offset_words = words[8 + chunk] / sizeof(uint32_t);
    REQUIRE(offset_words + 2 < word_count);
    const uint32_t fourcc = words[offset_words];
    p.chunks.push_back(fourcc);
    if (fourcc != FourCC("SHEX") && fourcc != FourCC("SHDR")) {
      continue;
    }
    const size_t body_start = offset_words + 2;
    const size_t body_words = words[offset_words + 1] / sizeof(uint32_t);
    REQUIRE(body_start + body_words <= word_count);
    // Version token: program type in the high 16 bits.
    p.program_type = words[body_start] >> 16;
    size_t i = body_start + 2;
    const size_t end = body_start + body_words;
    while (i < end) {
      uint32_t token = words[i];
      uint32_t length = dxbc::GetOpcodeTokenInstructionLength(token);
      if (length == 0) {
        REQUIRE(i + 1 < end);
        length = words[i + 1];
        REQUIRE(length != 0);
      }
      REQUIRE(i + length <= end);
      DxbcInstruction instruction;
      instruction.opcode = dxbc::Opcode(token & 0x7FF);
      instruction.dwords.assign(words + i, words + i + length);
      p.instructions.push_back(std::move(instruction));
      i += length;
    }
  }
  REQUIRE(p.program_type != UINT32_MAX);
  return p;
}

ParsedDxbc Translate(Shader::HostVertexShaderType type,
                     bool domain_as_vertex) {
  std::vector<uint32_t> ucode = BuildDomainPayloadUcode();
  Shader shader(xenos::ShaderType::kVertex, 0, ucode.data(), ucode.size(),
                std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);
  DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                  /*bindless_resources_used=*/true,
                                  /*edram_rov_used=*/false,
                                  /*gamma_render_target_as_unorm8=*/true);
  DxbcShaderTranslator::Modification modification(
      translator.GetDefaultVertexShaderModification(
          shader.GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters),
          type));
  modification.vertex.domain_as_vertex = domain_as_vertex ? 1 : 0;
  Shader::Translation* translation =
      shader.GetOrCreateTranslation(modification.value);
  REQUIRE(translator.TranslateAnalyzedShader(*translation));
  return Parse(translation->translated_binary());
}

struct DomainType {
  const char* name;
  Shader::HostVertexShaderType type;
  uint64_t pre_change_hash;
};

// Pinned from the tip BEFORE task 2.1 (translator untouched, 2026-09-20). A
// change here means the domain-shader emission moved, which this change must
// not do.
const DomainType kDomainTypes[] = {
    {"TriangleDomainCPIndexed",
     Shader::HostVertexShaderType::kTriangleDomainCPIndexed,
     0x80CA894E232BFB54ull},
    {"TriangleDomainPatchIndexed",
     Shader::HostVertexShaderType::kTriangleDomainPatchIndexed,
     0x586233187F040311ull},
    {"QuadDomainCPIndexed", Shader::HostVertexShaderType::kQuadDomainCPIndexed,
     0xD1F68EF09DF612F2ull},
    {"QuadDomainPatchIndexed",
     Shader::HostVertexShaderType::kQuadDomainPatchIndexed,
     0x6EC14EA43DF6E6BFull},
};

constexpr uint32_t kProgramTypeVertex =
    uint32_t(dxbc::ProgramType::kVertexShader);
constexpr uint32_t kProgramTypeDomain =
    uint32_t(dxbc::ProgramType::kDomainShader);

TEST_CASE("Domain types without domain_as_vertex emit the pre-change domain shader",
          "[domain_as_vertex][translation][golden]") {
  for (const DomainType& dt : kDomainTypes) {
    ParsedDxbc t = Translate(dt.type, /*domain_as_vertex=*/false);
    uint64_t hash = Fnv1a(t.binary);
    std::printf("DAVGOLD|%s|bytes=%zu|hash=%016llX\n", dt.name,
                t.binary.size(), (unsigned long long)hash);
    INFO(dt.name);
    CHECK(hash == dt.pre_change_hash);
    // Structural, so a hash drift is also explained.
    CHECK(t.program_type == kProgramTypeDomain);
    CHECK(t.has_chunk(FourCC("PCSG")));
    CHECK(t.count_opcode(dxbc::Opcode::kDclInputControlPointCount) == 1);
    CHECK(!t.contains_string("xe_tessellation_vertices"));
    CHECK(!t.contains_string("SV_VertexID"));
    CHECK(t.count_opcode(dxbc::Opcode::kDclResourceRaw) == 0);
    CHECK(t.count_opcode(dxbc::Opcode::kLdRaw) == 0);
    bool declares_domain_point = false;
    for (dxbc::OperandType ot :
         t.first_operand_types(dxbc::Opcode::kDclInput)) {
      if (ot == dxbc::OperandType::kInputDomainPoint) {
        declares_domain_point = true;
      }
    }
    CHECK(declares_domain_point);
  }
}

TEST_CASE("Domain types with domain_as_vertex emit a vertex shader over t1",
          "[domain_as_vertex][translation]") {
  for (const DomainType& dt : kDomainTypes) {
    ParsedDxbc t = Translate(dt.type, /*domain_as_vertex=*/true);
    INFO(dt.name);
    CHECK(t.program_type == kProgramTypeVertex);
    // Inputs: SV_VertexID only; no domain location, no control points, no
    // patch constants.
    CHECK(t.contains_string("SV_VertexID"));
    CHECK(!t.has_chunk(FourCC("PCSG")));
    CHECK(!t.contains_string("SV_TessFactor"));
    CHECK(t.count_opcode(dxbc::Opcode::kDclInputControlPointCount) == 0);
    for (dxbc::OperandType ot :
         t.first_operand_types(dxbc::Opcode::kDclInput)) {
      CHECK(ot != dxbc::OperandType::kInputDomainPoint);
      CHECK(ot != dxbc::OperandType::kInputControlPoint);
    }
    // The record SRV: named in the RDEF, declared raw at t1 / space 0 (SM5.1
    // declaration operand: id, lower, upper, then the space dword), and read
    // twice (domain location, then the index floats).
    CHECK(t.contains_string("xe_tessellation_vertices"));
    // This payload performs no vertex fetch, so the shared-memory SRV is
    // absent and the record SRV is the ONLY raw resource.
    CHECK(!t.contains_string("xe_shared_memory_srv"));
    REQUIRE(t.count_opcode(dxbc::Opcode::kDclResourceRaw) == 1);
    for (const DxbcInstruction& i : t.instructions) {
      if (i.opcode != dxbc::Opcode::kDclResourceRaw) continue;
      REQUIRE(i.dwords.size() >= 6);
      CHECK(dxbc::OperandType((i.dwords[1] >> 12) & 0xFF) ==
            dxbc::OperandType::kResource);
      CHECK(i.dwords[3] ==
            uint32_t(DxbcShaderTranslator::SRVMainRegister::kTessellationVertices));
      CHECK(i.dwords[4] ==
            uint32_t(DxbcShaderTranslator::SRVMainRegister::kTessellationVertices));
      CHECK(i.dwords[3] == 1u);
      CHECK(i.dwords[5] == uint32_t(DxbcShaderTranslator::SRVSpace::kMain));
    }
    CHECK(t.count_opcode(dxbc::Opcode::kLdRaw) == 2);
    // Different from the domain-shader output for the same program.
    ParsedDxbc d = Translate(dt.type, /*domain_as_vertex=*/false);
    CHECK(Fnv1a(t.binary) != Fnv1a(d.binary));
  }
}

TEST_CASE("Plain vertex shaders ignore domain_as_vertex",
          "[domain_as_vertex][translation]") {
  // The bit only means something with a domain host vertex shader type;
  // kVertex emission is identical with it set or clear.
  ParsedDxbc clear = Translate(Shader::HostVertexShaderType::kVertex, false);
  ParsedDxbc set = Translate(Shader::HostVertexShaderType::kVertex, true);
  CHECK(Fnv1a(clear.binary) == Fnv1a(set.binary));
  CHECK(clear.program_type == kProgramTypeVertex);
  CHECK(!clear.contains_string("xe_tessellation_vertices"));
}

}  // namespace
}  // namespace test
}  // namespace gpu
}  // namespace xe
