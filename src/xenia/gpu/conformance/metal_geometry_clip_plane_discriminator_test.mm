/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Minimal on-device discriminator for the geometry-emulation `materializeAll`
// failures (docs/metal_geometry_pipeline_materialize_failures.md).
//
// Live attribution narrowed the failure to a single field: every one of 43
// successful geometry pipelines in a Red Dead run decoded to
// user_clip_plane_count == 0, while both failures decoded to 1 and 2. Depth
// format was falsified as the discriminator — the same attachment tuple both
// succeeds and fails.
//
// This isolates that field. It drives the PRODUCTION path end to end — guest
// microcode through DxbcShaderTranslator, DXBC->DXIL, MetalShaderConverter, and
// MetalPipelineCache::GetOrCreateGeometryPipelineState — rather than a
// hand-written Metal shader, because the question is whether the converter
// rejects what our translator emits. A hand-written mesh shader would answer a
// different question.
//
// One rect-list shader identity, identical attachments, sample count, blend
// state and write mask. The ONLY difference between the two arms is
// Modification::vertex::user_clip_plane_count.

#import <Metal/Metal.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/metal_dxbc_clip_relabel.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// Minimal fetch-free guest vertex program: receives the translated vertex index
// in r0.x and exports (r0.x, 0, 0, 1) as position. Shape taken from an
// authoritative trace shader. Real microcode keeps DxbcShaderTranslator in the
// loop instead of pinning a generated artifact.
alignas(uint32_t) constexpr uint8_t kVertexUcode[] = {
    0x03, 0x10, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC4, 0x04, 0x10, 0x00, 0x00,
    0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC2, 0x05, 0x10, 0x00, 0x00, 0x00, 0x12, 0x06, 0x10,
    0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2,
    0x00, 0xC0, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2, 0x3E, 0xC0, 0x89, 0xC8,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE2, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xE2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Minimal guest PIXEL program, assembled here rather than reused from the
// vertex fixture: analyzing vertex microcode as a pixel shader traps in the
// ucode analyzer (its exports are positions/interpolators, which a pixel
// shader cannot carry). Verified — that shortcut was tried first and aborted
// inside AnalyzeUcode.
//
// Shape: control flow occupies dwords 0..2 (two slots packed per three dwords,
// bounded by the first exec's address), the ALU instruction lives at dwords
// 3..5.
//   slot 0: alloc colors
//   slot 1: exece, address 1, count 1, sequence 0 (ALU)
//   instr : export to oC0, all four components taken from the vector op
std::vector<uint32_t> MakePixelUcode() {
  // --- control-flow slot 0: alloc, AllocType::kPsColors, size 0.
  uint32_t alloc_d0 = 0;  // size_ = 0
  uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kPsColors) << 9) |
                      (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);

  // --- control-flow slot 1: exece at instruction 1, one ALU instruction.
  uint32_t exec_d0 = (1u & 0xFFF) | ((1u & 0x7) << 12);  // address 1, count 1
  uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;

  // Two control-flow instructions packed into three dwords.
  std::vector<uint32_t> u(6, 0);
  u[0] = alloc_d0;
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = (exec_d0 >> 16) | ((exec_d1 & 0xFFFF) << 16);

  // --- ALU: vector op writing all four components of oC0 as an export.
  // vector_dest 0 (colour 0), export_data 1, vector_write_mask 0xF, and the
  // vector opcode kMax with both sources the same constant acts as a move.
  uint32_t alu0 = (0u & 0x3F)          // vector_dest = oC0
                  | (0u << 6)          // vector_dest_rel
                  | (0u << 7)          // abs_constants
                  | (0u << 8)          // scalar_dest
                  | (0u << 14)         // scalar_dest_rel
                  | (1u << 15)         // export_data
                  | (0xFu << 16)       // vector_write_mask
                  | (0u << 20)         // scalar_write_mask
                  | (0u << 24)         // vector_clamp
                  | (0u << 25)         // scalar_clamp
                  | (uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26);
  // Swizzles: identity for both vector sources.
  uint32_t alu1 = (0u << 0)            // src3_swiz
                  | (0u << 8)          // src2_swiz
                  | (0u << 16);        // src1_swiz
  // src1/src2 read constant 0; sel bits 0 select the constant file.
  uint32_t alu2 = (0u << 0)            // src3_reg
                  | (0u << 8)          // src2_reg
                  | (0u << 16)         // src1_reg
                  | (uint32_t(ucode::AluVectorOpcode::kMax) << 24);
  u[3] = alu0;
  u[4] = alu1;
  u[5] = alu2;
  return u;
}

// Attachment/raster state held IDENTICAL across both arms, matching the shape
// the live failure carried (single RGBA-class colour target, D32F_S8 depth,
// 1x raster).
constexpr uint64_t kColorFormat = uint64_t(MTLPixelFormatRGBA8Unorm);
constexpr uint64_t kDepthFormat = uint64_t(MTLPixelFormatDepth32Float_Stencil8);
constexpr uint32_t kWriteMask = 0x7;
constexpr uint32_t kSampleCount = 1;

struct Harness {
  id<MTLDevice> device = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  std::unique_ptr<MetalShader> vertex_shader;
  std::vector<uint32_t> pixel_ucode;
  std::unique_ptr<MetalShader> pixel_shader;
  bool ok = false;

  Harness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) return;
    cache = std::make_unique<MetalPipelineCache>((__bridge void*)device);
    if (!cache->Initialize()) return;
    vertex_shader = std::make_unique<MetalShader>(
        xenos::ShaderType::kVertex, 0x1111111111111111ull,
        reinterpret_cast<const uint32_t*>(kVertexUcode),
        sizeof(kVertexUcode) / sizeof(uint32_t), std::endian::little);
    pixel_ucode = MakePixelUcode();
    pixel_shader = std::make_unique<MetalShader>(
        xenos::ShaderType::kPixel, 0x2222222222222222ull, pixel_ucode.data(),
        pixel_ucode.size(), std::endian::little);
    StringBuffer disasm;
    vertex_shader->AnalyzeUcode(disasm);
    pixel_shader->AnalyzeUcode(disasm);
    ok = true;
  }
};

Harness& H() {
  static Harness harness;
  return harness;
}

struct Arm {
  MetalShader::MetalTranslation* vertex = nullptr;
  MetalShader::MetalTranslation* pixel = nullptr;
  DxbcShaderTranslator::Modification vertex_modification{0};
  DxbcShaderTranslator::Modification pixel_modification{0};
  void* pipeline = nullptr;
  uint32_t gs_vertex_size = 0;
  uint32_t gs_max_input_primitives = 0;
  GeometryShaderKey gs_key;
};

// Builds one arm. `clip_planes` is the ONLY input that differs between arms.
bool BuildArm(uint32_t clip_planes, Arm* out,
              MetalPipelineCache* cache_override = nullptr) {
  auto& h = H();
  MetalPipelineCache* cache = cache_override ? cache_override : h.cache.get();
  auto* translator = h.cache->shader_translator();

  DxbcShaderTranslator::Modification vm(
      translator->GetDefaultVertexShaderModification(
          h.vertex_shader->GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters),
          Shader::HostVertexShaderType::kVertex));
  DxbcShaderTranslator::Modification pm(
      translator->GetDefaultPixelShaderModification(
          h.pixel_shader->GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters)));
  // Interpolator masks must agree (GetGeometryShaderKey asserts it).
  vm.vertex.interpolator_mask = 0;
  pm.pixel.interpolator_mask = 0;
  vm.vertex.user_clip_plane_count = clip_planes;
  vm.vertex.user_clip_plane_cull = 0;

  out->vertex_modification = vm;
  out->pixel_modification = pm;
  out->vertex = static_cast<MetalShader::MetalTranslation*>(
      h.vertex_shader->GetOrCreateTranslation(vm.value));
  out->pixel = static_cast<MetalShader::MetalTranslation*>(
      h.pixel_shader->GetOrCreateTranslation(pm.value));
  if (!cache->TranslateShader(*out->vertex)) return false;
  if (!cache->TranslateShader(*out->pixel)) return false;

  if (!GetGeometryShaderKey(PipelineGeometryShader::kRectangleList, vm, pm,
                            out->gs_key)) {
    return false;
  }

  uint64_t colors[xenos::kMaxColorRenderTargets] = {kColorFormat, 0, 0, 0};
  uint32_t blends[xenos::kMaxColorRenderTargets] = {0x00010001, 0x00010001,
                                                    0x00010001, 0x00010001};
  MetalPipelineRasterKey raster_key{kSampleCount};
  out->pipeline = cache->GetOrCreateGeometryPipelineState(
      out->vertex, out->pixel, PipelineGeometryShader::kRectangleList, vm, pm,
      colors, kDepthFormat, raster_key, blends, kWriteMask,
      &out->gs_vertex_size, &out->gs_max_input_primitives);
  return true;
}

// Counts DXIL/DXBC signature elements naming a semantic, so the two arms'
// stage I/O can be compared without a full container parser. The translated
// binary is a DXBC container whose signature chunks store semantic names as
// plain strings.
uint32_t CountSemantic(const std::vector<uint8_t>& binary, const char* name) {
  if (binary.empty()) return 0;
  const size_t n = std::strlen(name);
  uint32_t count = 0;
  for (size_t i = 0; i + n <= binary.size(); ++i) {
    if (std::memcmp(binary.data() + i, name, n) == 0) ++count;
  }
  return count;
}

}  // namespace

// ---------------------------------------------------------------------------
// Task 1.1 audit (XE_CLIP_AUDIT=<dir>): preserve the exact artifacts for the
// minimal pair so the divergence point can be located rather than inferred.
// Writes per-arm files; inert unless the variable is set.
namespace {

void AuditWrite(const char* dir, const std::string& name,
                const std::string& text) {
  std::string path = std::string(dir) + "/" + name;
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return;
  fwrite(text.data(), 1, text.size(), f);
  fclose(f);
}

void AuditWriteBinary(const char* dir, const std::string& name,
                      const std::vector<uint8_t>& bytes) {
  std::string path = std::string(dir) + "/" + name;
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return;
  if (!bytes.empty()) fwrite(bytes.data(), 1, bytes.size(), f);
  fclose(f);
}

// Signature semantics carried by a DXBC/DXIL container, in order, with the
// index byte that follows each name in the signature element records.
std::string AuditSemantics(const std::vector<uint8_t>& binary) {
  static const char* kNames[] = {
      "SV_Position",   "SV_ClipDistance", "SV_CullDistance", "SV_PointSize",
      "SV_RenderTargetArrayIndex", "SV_Target", "TEXCOORD", "COLOR",
      "XEVERTEXID", "XEPSINVOCATION", "NORMAL", "POSITION"};
  std::string out;
  for (const char* n : kNames) {
    uint32_t c = CountSemantic(binary, n);
    if (c) {
      out += fmt::format("  {} x{}\n", n, c);
    }
  }
  return out.empty() ? std::string("  (none matched)\n") : out;
}

}  // namespace


TEST_CASE("Geometry clip-plane discriminator: count 0 materializes, count 1 does not",
          "[l1][metal][geometry][clip]") {
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  Arm zero, one;
  REQUIRE(BuildArm(0, &zero));
  REQUIRE(BuildArm(1, &one));

  // Premise: the arms differ in the clip-plane field and nothing else that
  // reaches the geometry shader key.
  INFO("gs_key zero=" << std::hex << zero.gs_key.key
                      << " one=" << one.gs_key.key);
  CHECK(zero.gs_key.user_clip_plane_count == 0);
  CHECK(one.gs_key.user_clip_plane_count == 1);
  CHECK(zero.gs_key.type == one.gs_key.type);
  CHECK(zero.gs_key.interpolator_count == one.gs_key.interpolator_count);
  CHECK(zero.gs_key.has_point_size == one.gs_key.has_point_size);
  CHECK(zero.gs_key.has_vertex_kill_and == one.gs_key.has_vertex_kill_and);
  CHECK(zero.gs_key.key != one.gs_key.key);

  SECTION("clip-plane count 0 materializes") {
    CHECK(zero.pipeline != nullptr);
  }

  SECTION("clip-plane count 1 reproduces the failure") {
    // The live classification: creation returns null and the entry is cached
    // as a negative, suppressing every later draw of the class.
    CHECK(one.pipeline == nullptr);
  }

  SECTION("SV_ClipDistance is the stage-I/O difference") {
    // If the two translated vertex binaries differ only by the clip-distance
    // output, the converter's rejection is about that signature element rather
    // than some other emitted change.
    const auto& zero_binary = zero.vertex->translated_binary();
    const auto& one_binary = one.vertex->translated_binary();
    uint32_t zero_clip = CountSemantic(zero_binary, "SV_ClipDistance");
    uint32_t one_clip = CountSemantic(one_binary, "SV_ClipDistance");
    INFO("SV_ClipDistance occurrences: zero=" << zero_clip
                                              << " one=" << one_clip);
    CHECK(zero_clip == 0);
    CHECK(one_clip > 0);
  }
}

// ---------------------------------------------------------------------------
// Task 4.1 — REGRESSION CHECKPOINT. Pipeline creation proves AVAILABILITY ONLY.
// It is not semantic acceptance: coverage against the independent Xenos-plane
// oracle (tasks 4.2-4.5) is the load-bearing evidence. A pipeline that creates
// and then clips wrongly is WORSE than one that fails loudly, because today's
// failure is at least visible in the skip accounting as a dropped-draw class.
//
// A FRESH CACHE PER GATE STATE IS REQUIRED, not tidiness: a clip-1 failure is
// NEGATIVE-CACHED, so reusing one cache across the gate flip would return the
// cached nullptr and the relabel-on arm would fail for the wrong reason.
TEST_CASE("Geometry clip-plane 4.1: relabel OFF still fails, ON creates 0/1/2",
          "[l1][metal][geometry][clip][clip41]") {
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  struct Outcome { void* pipe; };
  auto build = [&](uint32_t planes, bool relabel_on) -> void* {
    if (relabel_on) {
      setenv("XE_METAL_CLIP_RELABEL", "1", 1);
    } else {
      unsetenv("XE_METAL_CLIP_RELABEL");
    }
    auto cache = std::make_unique<MetalPipelineCache>((__bridge void*)h.device);
    void* pipe = nullptr;
    if (cache->Initialize()) {
      Arm arm;
      if (BuildArm(planes, &arm, cache.get())) {
        pipe = arm.pipeline;
      }
    }
    unsetenv("XE_METAL_CLIP_RELABEL");
    return pipe;
  };

  SECTION("relabel OFF — the default path is UNCHANGED (the preserved failure)") {
    // The pre-fix expectation, deliberately NOT inverted: default behaviour must
    // not drift while the remedy is gated off.
    CHECK(build(0, false) != nullptr);
    CHECK(build(1, false) == nullptr);
    CHECK(build(2, false) == nullptr);
  }

  SECTION("relabel ON — clip counts 0, 1 and 2 all create") {
    void* p0 = build(0, true);
    void* p1 = build(1, true);
    void* p2 = build(2, true);
    INFO("pipelines: 0=" << p0 << " 1=" << p1 << " 2=" << p2);
    CHECK(p0 != nullptr);
    CHECK(p1 != nullptr);
    CHECK(p2 != nullptr);
  }
}

TEST_CASE("Geometry clip-plane control: the same clip output WITHOUT geometry emulation",
          "[l1][metal][geometry][clip][control]") {
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  // Distinguishes "MetalShaderConverter cannot handle clip distances at all"
  // from "the mesh/object-stage geometry bridge cannot". Same clip-plane-1
  // vertex translation, but built as an ORDINARY render pipeline with no
  // geometry emulation. If this succeeds, the incompatibility is specific to
  // the geometry-emulation bridge.
  Arm one;
  REQUIRE(BuildArm(1, &one));
  REQUIRE(one.vertex != nullptr);
  REQUIRE(one.vertex->has_metal_function());
  REQUIRE(one.pixel->has_metal_function());

  MTLRenderPipelineDescriptor* desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  desc.vertexFunction = (__bridge id<MTLFunction>)one.vertex->metal_function();
  desc.fragmentFunction = (__bridge id<MTLFunction>)one.pixel->metal_function();
  desc.colorAttachments[0].pixelFormat = (MTLPixelFormat)kColorFormat;
  desc.depthAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
  desc.stencilAttachmentPixelFormat = (MTLPixelFormat)kDepthFormat;
  desc.rasterSampleCount = kSampleCount;

  NSError* error = nil;
  id<MTLRenderPipelineState> pipeline =
      [h.device newRenderPipelineStateWithDescriptor:desc error:&error];
  std::string message =
      error ? [[error localizedDescription] UTF8String] : "";
  INFO("plain render pipeline with clip-plane-1 VS: " << message);
  // Recorded either way — this is a discriminator, not a pass/fail requirement.
  if (pipeline != nil) {
    SUCCEED("clip distances materialize OUTSIDE geometry emulation — the "
            "incompatibility is specific to the mesh/object-stage bridge");
  } else {
    SUCCEED("clip distances fail outside geometry emulation too — the "
            "incompatibility is general to the converter, not the bridge");
  }
}

TEST_CASE("Geometry clip-plane AUDIT: preserve minimal-pair artifacts",
          "[l1][metal][geometry][clip][audit]") {
  const char* dir = std::getenv("XE_CLIP_AUDIT");
  if (!dir) {
    SUCCEED("XE_CLIP_AUDIT not set; audit artifact capture skipped");
    return;
  }
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  for (uint32_t clip : {0u, 1u, 2u}) {
    Arm arm;
    if (!BuildArm(clip, &arm)) {
      AuditWrite(dir, fmt::format("clip{}_BUILD_FAILED.txt", clip), "\n");
      continue;
    }
    const auto& vs_binary = arm.vertex->translated_binary();
    const auto& ps_binary = arm.pixel->translated_binary();

    // The reflected object-stage size is the load-bearing number: if the added
    // clip-distance semantic does not change it, the bridge never reserves or
    // describes transport for the semantic it is being handed.
    AuditWrite(
        dir, fmt::format("clip{}_summary.txt", clip),
        fmt::format(
            "clip_plane_count={}\n"
            "gs_key=0x{:08X} (type={} interp={} clip={} cull={} killand={} "
            "pointsize={} pointcoord={})\n"
            "vertex_modification=0x{:016X}\n"
            "pixel_modification=0x{:016X}\n"
            "REFLECTED vertex_output_size={} bytes\n"
            "gs_max_input_primitives={}\n"
            "pipeline_created={}\n"
            "vs_translated_binary_bytes={}\n"
            "ps_translated_binary_bytes={}\n"
            "\nVS signature semantics:\n{}"
            "\nPS signature semantics:\n{}",
            clip, arm.gs_key.key, uint32_t(arm.gs_key.type),
            uint32_t(arm.gs_key.interpolator_count),
            uint32_t(arm.gs_key.user_clip_plane_count),
            uint32_t(arm.gs_key.user_clip_plane_cull),
            uint32_t(arm.gs_key.has_vertex_kill_and),
            uint32_t(arm.gs_key.has_point_size),
            uint32_t(arm.gs_key.has_point_coordinates),
            arm.vertex_modification.value, arm.pixel_modification.value,
            arm.gs_vertex_size, arm.gs_max_input_primitives,
            arm.pipeline ? "YES" : "NO", vs_binary.size(), ps_binary.size(),
            AuditSemantics(vs_binary), AuditSemantics(ps_binary)));

    AuditWriteBinary(dir, fmt::format("clip{}_vs.dxbc", clip), vs_binary);
    AuditWriteBinary(dir, fmt::format("clip{}_ps.dxbc", clip), ps_binary);

    // The GENERATED geometry stage — what the bridge is actually handed. Taken
    // from the shared producer so the artifact is the production one.
    std::vector<uint32_t> gs_dwords;
    CreateDxbcGeometryShader(arm.gs_key, gs_dwords);
    std::vector<uint8_t> gs_bytes(gs_dwords.size() * sizeof(uint32_t));
    if (!gs_dwords.empty()) {
      std::memcpy(gs_bytes.data(), gs_dwords.data(), gs_bytes.size());
    }
    AuditWriteBinary(dir, fmt::format("clip{}_gs.dxbc", clip), gs_bytes);
    AuditWrite(dir, fmt::format("clip{}_gs_semantics.txt", clip),
               fmt::format("generated GS dwords={} bytes={}\n"
                           "GS signature semantics:\n{}",
                           gs_dwords.size(), gs_bytes.size(),
                           AuditSemantics(gs_bytes)));
  }
  SUCCEED("audit artifacts written");
}


// ---------------------------------------------------------------------------
// Task 2 candidate 1: byte/value transport oracle for the boundary relabel.
//
// The relabel must preserve the payload EXACTLY and change only the label at
// the object->mesh hop. Merely stripping the intermediate semantic is not an
// acceptable outcome, so this asserts what survives, not just what changes.

namespace {

// Describes one signature element well enough to prove the payload layout did
// not move: register, mask, semantic index, component type, system value.
struct SigElement {
  std::string name;
  uint32_t semantic_index = 0;
  uint32_t system_value = 0;
  uint32_t component_type = 0;
  uint32_t register_index = 0;
  uint32_t mask = 0;
};

std::vector<SigElement> ReadSignature(const std::vector<uint8_t>& c,
                                      const char* fourcc, bool gs_stream) {
  using namespace xe::gpu::metal::dxbc_relabel_detail;
  std::vector<SigElement> out;
  if (c.size() < sizeof(uint32_t) * 8) return out;
  uint32_t chunk_count = ReadU32(c, sizeof(uint32_t) * 7);
  uint32_t want = FourCC(fourcc);
  for (uint32_t i = 0; i < chunk_count; ++i) {
    uint32_t off = ReadU32(c, sizeof(uint32_t) * 8 + i * sizeof(uint32_t));
    if (size_t(off) + 8 > c.size() || ReadU32(c, off) != want) continue;
    size_t data = size_t(off) + 8;
    uint32_t n = ReadU32(c, data);
    uint32_t ptr = ReadU32(c, data + 4);
    size_t stride = gs_stream ? sizeof(uint32_t) * 7 : sizeof(uint32_t) * 6;
    for (uint32_t p = 0; p < n; ++p) {
      size_t e = data + ptr + size_t(p) * stride;
      if (e + stride > c.size()) break;
      size_t base = gs_stream ? e + sizeof(uint32_t) : e;
      SigElement el;
      uint32_t name_ptr = ReadU32(c, base);
      const char* nm = reinterpret_cast<const char*>(c.data() + data + name_ptr);
      el.name = std::string(nm);
      el.semantic_index = ReadU32(c, base + 4);
      el.system_value = ReadU32(c, base + 8);
      el.component_type = ReadU32(c, base + 12);
      el.register_index = ReadU32(c, base + 16);
      el.mask = c[base + 20];
      out.push_back(el);
    }
  }
  return out;
}

}  // namespace

TEST_CASE("Clip relabel transport oracle: payload preserved, only the hop is renamed",
          "[l1][metal][geometry][clip][relabel]") {
  auto& h = H();
  if (!h.ok) {
    SUCCEED("no Metal device");
    return;
  }

  for (uint32_t clip : {0u, 1u, 2u}) {
    INFO("clip count " << clip);
    Arm arm;
    REQUIRE(BuildArm(clip, &arm));

    std::vector<uint8_t> vs = arm.vertex->translated_binary();
    std::vector<uint32_t> gs_dwords;
    CreateDxbcGeometryShader(arm.gs_key, gs_dwords);
    std::vector<uint8_t> gs(gs_dwords.size() * sizeof(uint32_t));
    if (!gs_dwords.empty()) {
      std::memcpy(gs.data(), gs_dwords.data(), gs.size());
    }
    REQUIRE_FALSE(vs.empty());
    REQUIRE_FALSE(gs.empty());

    const size_t vs_size_before = vs.size();
    const size_t gs_size_before = gs.size();
    auto vs_out_before = ReadSignature(vs, "OSGN", false);
    auto gs_in_before = ReadSignature(gs, "ISGN", false);
    auto gs_out_before = ReadSignature(gs, "OSG5", true);

    uint32_t vs_relabeled =
        RelabelDxbcClipDistance(vs, DxbcSignatureKind::kOutput);
    uint32_t gs_relabeled =
        RelabelDxbcClipDistance(gs, DxbcSignatureKind::kInput);

    auto vs_out_after = ReadSignature(vs, "OSGN", false);
    auto gs_in_after = ReadSignature(gs, "ISGN", false);
    auto gs_out_after = ReadSignature(gs, "OSG5", true);

    // Container size is untouched — the rewrite is strictly in place.
    CHECK(vs.size() == vs_size_before);
    CHECK(gs.size() == gs_size_before);

    if (clip == 0) {
      // Nothing to relabel, and nothing may change.
      CHECK(vs_relabeled == 0);
      CHECK(gs_relabeled == 0);
    } else {
      CHECK(vs_relabeled >= 1);
      CHECK(gs_relabeled >= 1);
    }

    // Element COUNTS are preserved — this is a relabel, not a strip.
    REQUIRE(vs_out_after.size() == vs_out_before.size());
    REQUIRE(gs_in_after.size() == gs_in_before.size());
    REQUIRE(gs_out_after.size() == gs_out_before.size());

    // Every element keeps its register, mask, semantic index and component
    // type. Only the clip elements' name and system value may differ, and only
    // on the boundary signatures.
    auto compare = [&](const std::vector<SigElement>& before,
                       const std::vector<SigElement>& after,
                       bool boundary) {
      for (size_t i = 0; i < before.size(); ++i) {
        INFO("element " << i << " '" << before[i].name << "'");
        CHECK(after[i].register_index == before[i].register_index);
        CHECK(after[i].mask == before[i].mask);
        CHECK(after[i].semantic_index == before[i].semantic_index);
        CHECK(after[i].component_type == before[i].component_type);
        bool was_clip =
            before[i].system_value == uint32_t(dxbc::Name::kClipDistance);
        if (boundary && was_clip) {
          CHECK(after[i].name == std::string(kMetalClipPayloadSemantic));
          CHECK(after[i].system_value == uint32_t(dxbc::Name::kUndefined));
        } else {
          CHECK(after[i].name == before[i].name);
          CHECK(after[i].system_value == before[i].system_value);
        }
      }
    };
    compare(vs_out_before, vs_out_after, /*boundary=*/true);
    compare(gs_in_before, gs_in_after, /*boundary=*/true);

    // THE POINT OF THE MECHANISM: the mesh stage's rasterizer-facing OUTPUT
    // must still carry SV_ClipDistance. This is also the string-pooling check —
    // if the container shared one name string between input and output, the
    // input relabel would have renamed this too.
    compare(gs_out_before, gs_out_after, /*boundary=*/false);
    if (clip != 0) {
      bool output_has_clip = false;
      for (const auto& el : gs_out_after) {
        if (el.system_value == uint32_t(dxbc::Name::kClipDistance) &&
            el.name == "SV_ClipDistance") {
          output_has_clip = true;
        }
      }
      CHECK(output_has_clip);
    }

    // No collision with guest interpolators or any other element.
    for (const auto& el : vs_out_after) {
      if (el.name == std::string(kMetalClipPayloadSemantic)) {
        CHECK(el.system_value == uint32_t(dxbc::Name::kUndefined));
      }
    }
  }
}

TEST_CASE("Clip fixture: nondegenerate guest positions disassemble as intended",
          "[l1][metal][geometry][clip][fixture]") {
  // Gate A needs positions that actually cover pixels. The original fixture
  // exports (vertex_index, 0, 0, 1) — degenerate in Y, fine for a creation
  // discriminator and useless for a coverage one. This one computes
  // y = x * x so the three rect-list vertices are not collinear.
  static const uint32_t kUcode[] = {
      0x00001002, 0x00001000, 0xC2000000, 0x00001003,
      0x00002000, 0x00000000, 0xC8020000, 0x006C6C00,
      0xC1000000, 0xC88F803E, 0x00000000, 0xC2000000};
  MetalShader shader(xenos::ShaderType::kVertex, 0x3333333333333333ull, kUcode,
                     sizeof(kUcode) / sizeof(uint32_t), std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);
  std::string text = disasm.to_string();
  INFO(text);
  // Assert the shape rather than the exact text: a multiply feeding a position
  // export. If the hand assembly is wrong this fails as bad test input.
  // Sources must be TEMP registers (r0), not constants: src_sel=1 selects the
  // temp file. The first assembly used sel=0 and decoded as c0, which would
  // have made the fixture read uninitialised constants instead of the guest
  // vertex index.
  CHECK(text.find("mul r0._y__, r0.xxxx, r0.xxxx") != std::string::npos);
  CHECK(text.find("max oPos.xyz1, r0, r0") != std::string::npos);
  CHECK(text.find("exece") != std::string::npos);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
