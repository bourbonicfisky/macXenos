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
#include <cstring>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/dxbc_shader_translator.h"
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
bool BuildArm(uint32_t clip_planes, Arm* out) {
  auto& h = H();
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
  if (!h.cache->TranslateShader(*out->vertex)) return false;
  if (!h.cache->TranslateShader(*out->pixel)) return false;

  if (!GetGeometryShaderKey(PipelineGeometryShader::kRectangleList, vm, pm,
                            out->gs_key)) {
    return false;
  }

  uint64_t colors[xenos::kMaxColorRenderTargets] = {kColorFormat, 0, 0, 0};
  uint32_t blends[xenos::kMaxColorRenderTargets] = {0x00010001, 0x00010001,
                                                    0x00010001, 0x00010001};
  MetalPipelineRasterKey raster_key{kSampleCount};
  out->pipeline = h.cache->GetOrCreateGeometryPipelineState(
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

}  // namespace metal
}  // namespace gpu
}  // namespace xe
