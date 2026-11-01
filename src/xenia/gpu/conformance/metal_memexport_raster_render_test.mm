/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// extend-metal-rasterizing-memexport-producers task 3.1 — the L1/L2
// POISON-THEN-PROVE render family.
//
// A guest vertex program that BOTH rasterizes (exports oPos) and exports to
// memory (eA + eM0) is driven through the production shader chain
// (ucode -> DxbcShaderTranslator -> dxilconv -> MetalShaderConverter) and the
// production argument-buffer builder, once as a PLAIN vertex+fragment draw and
// once as a GEOMETRY-EMULATED object+mesh+fragment draw. The stand-in shared
// memory is poisoned before the draw, and afterwards:
//
//   1. every exported lane byte-matches the CPU-computed oracle,
//   2. lanes inside the declared stream that the draw never indexes still hold
//      the poison (no partial/over-wide writes),
//   3. every byte outside the stream's declared extent still holds the poison,
//   4. the RASTER output (color + depth, byte-compared) is IDENTICAL to the
//      gate-off arm -- routing the exports changes memory, never pixels.
//
// The gate-off arm models production exactly: u0/space0 points at the NULL UAV
// descriptor and the system-constants kSysFlag_SharedMemoryIsUAV bit is clear.
// Both halves matter and this suite pins both -- binding the writable UAV
// WITHOUT the flag produces a silently export-free draw (measured live on the
// Viva Pinata frames: five rasterizing producers published GPU-written ranges
// whose bytes never changed), and setting the flag without the writable
// binding would store into the 256-byte null descriptor.
//
// Probe-test style, matching metal_tessellation_render_conformance_test.mm:
// every arm is CHECKed (not REQUIREd) and prints an "MXRASTER|" matrix line so
// one failure still shows the whole picture.

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/byte_order.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/metal_binding_layout.h"
#include "xenia/gpu/metal/metal_memexport_raster_flags.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"

#include "metal_irconverter_runtime.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

using metal::BuildMetalTopLevelArgumentBuffer;
using metal::kMetalBindingSlotCount;
using metal::MetalBindingAddresses;
using metal::MetalPipelineCache;
using metal::MetalPipelineRasterKey;
using metal::MetalShader;
using HostVertexShaderType = Shader::HostVertexShaderType;

constexpr uint32_t kRT = 128;

// --- Guest microcode fixture -------------------------------------------------
//
// CF slots:
//   0: exec addr=4 count=1     -> @4: mul r0.y, r0.xxxx, r0.xxxx  (y = idx^2)
//   1: alloc export position
//   2: exec addr=5 count=1     -> @5: mad oPos.xyz1, r0, c2, c3
//   3: alloc export memory
//   4: execEnd addr=6 count=2  -> @6: mad eA, r0.xxxx, c0, c1
//                                @7: max eM0, r0, r0
//
// r0.x is the vertex index (StartVertexShader_LoadVertexIndex), r0.y is set by
// @4, r0.zw stay 0 from the translator's register-zeroing prologue. So the
// exported element for vertex i is EXACTLY (float(i), float(i*i), 0, 0) and
// its element index is i -- an oracle with no shader-side rounding at all
// (k_32_32_32_32_FLOAT stores the eM# bits verbatim, and Endian128::kNone
// leaves them unswapped).
//
// @6 must satisfy ParsedAluInstruction::GetMemExportStreamConstant: a kMad
// into eA, all four components from the vector operation, unclamped, whose
// THIRD operand is an absolute, identity-swizzled, unnegated float constant --
// that constant IS the stream descriptor (xe_gpu_memexport_stream_t).

constexpr uint32_t kAluRetainPrev =
    uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26;
// Component-relative swizzle byte selecting .x for all four components.
constexpr uint32_t kSwizzleXXXX = 0x6C;

void PackControlFlow(std::vector<uint32_t>& u, uint32_t slot, uint32_t d0,
                     uint32_t d1) {
  const uint32_t base = (slot >> 1) * 3;
  if ((slot & 1) == 0) {
    u[base + 0] = d0;
    u[base + 1] = (u[base + 1] & 0xFFFF0000u) | (d1 & 0xFFFFu);
  } else {
    u[base + 1] = (u[base + 1] & 0x0000FFFFu) | ((d0 & 0xFFFFu) << 16);
    u[base + 2] = ((d0 >> 16) & 0xFFFFu) | ((d1 & 0xFFFFu) << 16);
  }
}

std::vector<uint32_t> BuildRasterizingExporterUcode() {
  std::vector<uint32_t> u(24, 0);
  auto exec = [](uint32_t address, uint32_t count, bool end) {
    return std::make_pair(
        address | (count << 12),
        uint32_t(end ? ucode::ControlFlowOpcode::kExecEnd
                     : ucode::ControlFlowOpcode::kExec)
            << 12);
  };
  auto alloc = [](ucode::AllocType type) {
    return std::make_pair(
        uint32_t(0),
        (uint32_t(type) << 9) |
            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12));
  };
  auto cf0 = exec(4, 1, false);
  PackControlFlow(u, 0, cf0.first, cf0.second);
  auto cf1 = alloc(ucode::AllocType::kVsPosition);
  PackControlFlow(u, 1, cf1.first, cf1.second);
  auto cf2 = exec(5, 1, false);
  PackControlFlow(u, 2, cf2.first, cf2.second);
  auto cf3 = alloc(ucode::AllocType::kMemory);
  PackControlFlow(u, 3, cf3.first, cf3.second);
  auto cf4 = exec(6, 2, true);
  PackControlFlow(u, 4, cf4.first, cf4.second);

  const uint32_t kmul = uint32_t(ucode::AluVectorOpcode::kMul) << 24;
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t kmad = uint32_t(ucode::AluVectorOpcode::kMad) << 24;
  const uint32_t src1_temp = 1u << 31;
  const uint32_t src2_temp = 1u << 30;

  // @4: mul r0.y, r0.xxxx, r0.xxxx
  u[12] = 0u /*vector_dest=r0*/ | (0b0010u << 16) /*write mask .y*/ |
          kAluRetainPrev;
  u[13] = (kSwizzleXXXX << 8) | (kSwizzleXXXX << 16);
  u[14] = kmul | src1_temp | src2_temp;
  // @5: mad oPos.xyz1, r0, c2, c3  (w: vector+scalar mask both set = const 1)
  u[15] = 62u /*oPos*/ | (1u << 15) /*export*/ | (0b1111u << 16) |
          (0b1000u << 20) | kAluRetainPrev;
  u[16] = 0;  // identity swizzles for r0 / c2 / c3.
  u[17] = 3u /*src3=c3*/ | (2u << 8) /*src2=c2*/ | (0u << 16) /*src1=r0*/ |
          kmad | src1_temp;
  // @6: mad eA, r0.xxxx, c0, c1
  u[18] = uint32_t(ucode::ExportRegister::kExportAddress) | (1u << 15) |
          (0b1111u << 16) | kAluRetainPrev;
  u[19] = 0u /*src3 identity*/ | (0u << 8) /*src2 identity*/ |
          (kSwizzleXXXX << 16);
  u[20] = 1u /*src3=c1 (stream constant)*/ | (0u << 8) /*src2=c0*/ |
          (0u << 16) /*src1=r0*/ | kmad | src1_temp;
  // @7: max eM0, r0, r0
  u[21] = uint32_t(ucode::ExportRegister::kExportData0) | (1u << 15) |
          (0b1111u << 16) | kAluRetainPrev;
  u[22] = 0;
  u[23] = kmax | src1_temp | src2_temp;
  return u;
}

// Guest pixel program: oC0 = (c0.x, c0.y, 1, 1). Same fixture shape as the
// tessellation render suite (its provenance note applies verbatim).
std::vector<uint32_t> BuildPixelUcode() {
  const uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kPsColors) << 9) |
                            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  const uint32_t exec_d0 = 1u | (1u << 12);
  const uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;
  std::vector<uint32_t> u(6, 0);
  u[0] = 0;
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = ((exec_d0 >> 16) & 0xFFFF) | ((exec_d1 & 0xFFFF) << 16);
  u[3] = 0u /*oC0*/ | (1u << 15) | (0xFu << 16) | (0b1100u << 20) |
         kAluRetainPrev;
  u[4] = 0;
  u[5] = uint32_t(ucode::AluVectorOpcode::kMax) << 24;  // src sel = const c0.
  return u;
}

// Guest pixel program that rasterizes AND exports (2026-10-26, pixel-stage
// memexport):
//   0: alloc colors
//   1: exec addr=4 count=1     -> @4: max oC0, c0, c0
//   2: alloc export memory
//   3: execEnd addr=5 count=2  -> @5: mad eA, r0.xxxx, c0, c1
//                                @6: max eM0, c2, c2
// No interpolators are written, so r0.x is 0 in every pixel: every covered
// pixel stores the same element (index 0 of the c1 stream) with the same
// value c2, which makes the oracle race-free.
std::vector<uint32_t> BuildPixelExporterUcode() {
  std::vector<uint32_t> u(21, 0);
  auto exec = [](uint32_t address, uint32_t count, bool end) {
    return std::make_pair(
        address | (count << 12),
        uint32_t(end ? ucode::ControlFlowOpcode::kExecEnd
                     : ucode::ControlFlowOpcode::kExec)
            << 12);
  };
  auto alloc = [](ucode::AllocType type) {
    return std::make_pair(
        uint32_t(0),
        (uint32_t(type) << 9) |
            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12));
  };
  auto cf0 = alloc(ucode::AllocType::kPsColors);
  PackControlFlow(u, 0, cf0.first, cf0.second);
  auto cf1 = exec(4, 1, false);
  PackControlFlow(u, 1, cf1.first, cf1.second);
  auto cf2 = alloc(ucode::AllocType::kMemory);
  PackControlFlow(u, 2, cf2.first, cf2.second);
  auto cf3 = exec(5, 2, true);
  PackControlFlow(u, 3, cf3.first, cf3.second);
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t kmad = uint32_t(ucode::AluVectorOpcode::kMad) << 24;
  const uint32_t src1_temp = 1u << 31;
  // @4: max oC0, c0, c0 (the same color write as BuildPixelUcode).
  u[12] = 0u /*oC0*/ | (1u << 15) | (0xFu << 16) | (0b1100u << 20) |
          kAluRetainPrev;
  u[13] = 0;
  u[14] = kmax;
  // @5: mad eA, r0.xxxx, c0, c1
  u[15] = uint32_t(ucode::ExportRegister::kExportAddress) | (1u << 15) |
          (0b1111u << 16) | kAluRetainPrev;
  u[16] = 0u | (0u << 8) | (kSwizzleXXXX << 16);
  u[17] = 1u /*src3=c1 (stream constant)*/ | (0u << 8) /*src2=c0*/ |
          (0u << 16) /*src1=r0*/ | kmad | src1_temp;
  // @6: max eM0, c2, c2
  u[18] = uint32_t(ucode::ExportRegister::kExportData0) | (1u << 15) |
          (0b1111u << 16) | kAluRetainPrev;
  u[19] = 0;
  u[20] = kmax | (2u << 8) /*src2=c2*/ | (2u << 16) /*src1=c2*/;
  return u;
}

// LoadShader takes GUEST ucode (big-endian dwords).
std::vector<uint32_t> ToGuestEndian(const std::vector<uint32_t>& host_words) {
  std::vector<uint32_t> guest(host_words.size());
  for (size_t i = 0; i < host_words.size(); ++i) {
    guest[i] = xe::byte_swap(host_words[i]);
  }
  return guest;
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// --- Stream descriptor -------------------------------------------------------

constexpr uint32_t kSharedBufferSize = 64 * 1024;
constexpr uint32_t kStreamBaseBytes = 0x1000;      // Inside the stand-in buffer.
constexpr uint32_t kStreamIndexCount = 8;          // Declared elements.
constexpr uint32_t kElementBytes = 16;             // k_32_32_32_32_FLOAT.
constexpr uint32_t kStreamBytes = kStreamIndexCount * kElementBytes;
constexpr uint32_t kDrawVertexCount = 6;           // Indexes elements 0..5.
constexpr uint8_t kPoisonByte = 0xA5;

// xe_gpu_memexport_stream_t as the four float-constant dwords the guest VS
// adds into eA. Every dword's upper field is a hardware-validated constant
// that the TRANSLATED shader re-checks (const_0x1 in eA.x bits 31:30, and
// 0x96 in the top 9 bits of eA.y/eA.z/eA.w) -- a wrong constant here makes the
// shader skip every store, which is exactly the failure this fixture must be
// able to distinguish from a binding failure.
std::array<uint32_t, 4> BuildStreamConstant() {
  std::array<uint32_t, 4> stream = {};
  stream[0] = (1u << 30) | (kStreamBaseBytes >> 2);   // const_0x1 | base>>2
  stream[1] = 0x4B000000u;                             // const_0x4b000000
  stream[2] = (uint32_t(xenos::Endian128::kNone)) |
              (uint32_t(xenos::ColorFormat::k_32_32_32_32_FLOAT) << 8) |
              (uint32_t(xenos::SurfaceNumberFormat::kFloat) << 16) |
              (0x4B0u << 20);                          // const_0x4b0
  stream[3] = kStreamIndexCount | (0x96u << 23);       // const_0x96
  return stream;
}

// The pixel exporter's stream: element 0 at kPixelStreamBaseBytes holds c2.
constexpr uint32_t kPixelStreamBaseBytes = 0x2000;
constexpr uint32_t kPixelStreamIndexCount = 4;
constexpr uint32_t kPixelStreamBytes = kPixelStreamIndexCount * kElementBytes;
const std::array<float, 4> kPixelExportValue = {3.5f, -2.0f, 7.0f, 1.0f};

std::array<uint32_t, 4> BuildPixelStreamConstant() {
  std::array<uint32_t, 4> stream = BuildStreamConstant();
  stream[0] = (1u << 30) | (kPixelStreamBaseBytes >> 2);
  stream[3] = kPixelStreamIndexCount | (0x96u << 23);
  return stream;
}

// The CPU oracle: element i holds (float(i), float(i*i), 0, 0), verbatim.
std::array<uint32_t, 4> OracleElement(uint32_t index) {
  return {FloatBits(float(index)), FloatBits(float(index * index)),
          FloatBits(0.0f), FloatBits(0.0f)};
}

// --- Harness -----------------------------------------------------------------

struct ExportArmResult {
  bool ok = false;
  std::string error;
  std::vector<uint8_t> shared;  // Full stand-in shared-memory readback.
  std::vector<uint8_t> rgba;
  std::vector<float> depth;
  uint32_t covered = 0;
};

struct ExportHarness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  bool ok = false;
  std::string error;

  ExportHarness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      error = "no Metal device";
      return;
    }
    queue = [device newCommandQueue];
    cache = std::make_unique<MetalPipelineCache>((__bridge void*)device);
    if (!queue || !cache->Initialize()) {
      error = "queue/MetalPipelineCache::Initialize failed";
      return;
    }
    ok = true;
  }

  id<MTLBuffer> NewBuffer(NSUInteger length, uint8_t fill, NSString* label) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:length
                            options:MTLResourceStorageModeShared];
    if (buffer) {
      std::memset([buffer contents], fill, length);
      buffer.label = label;
    }
    return buffer;
  }

  MetalShader::MetalTranslation* MakePixelTranslation(
      bool pixel_exporter = false) {
    static const std::vector<uint32_t> plain_ucode =
        ToGuestEndian(BuildPixelUcode());
    static const std::vector<uint32_t> exporter_ucode =
        ToGuestEndian(BuildPixelExporterUcode());
    const std::vector<uint32_t>& ucode =
        pixel_exporter ? exporter_ucode : plain_ucode;
    MetalShader* shader = cache->LoadShader(xenos::ShaderType::kPixel,
                                            ucode.data(),
                                            uint32_t(ucode.size()));
    if (!shader->is_ucode_analyzed()) {
      StringBuffer disasm;
      shader->AnalyzeUcode(disasm);
    }
    DxbcShaderTranslator::Modification modification(
        cache->shader_translator()->GetDefaultPixelShaderModification(
            shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters)));
    modification.pixel.interpolator_mask = 0;
    modification.pixel.rt0_blend_rgb_factor_for_premult =
        xenos::BlendFactor::kOne;
    modification.pixel.rt0_blend_a_factor_for_premult =
        xenos::BlendFactor::kOne;
    auto* translation = static_cast<MetalShader::MetalTranslation*>(
        shader->GetOrCreateTranslation(modification.value));
    if (!translation->has_metal_function() &&
        !cache->TranslateShader(*translation)) {
      return nullptr;
    }
    return translation;
  }

  MetalShader* LoadExporterShader() {
    static const std::vector<uint32_t> ucode =
        ToGuestEndian(BuildRasterizingExporterUcode());
    MetalShader* shader = cache->LoadShader(xenos::ShaderType::kVertex,
                                            ucode.data(),
                                            uint32_t(ucode.size()));
    if (!shader->is_ucode_analyzed()) {
      StringBuffer disasm;
      shader->AnalyzeUcode(disasm);
    }
    return shader;
  }

  // `uav_bound` models BOTH halves of the production gate: u0/space0 points at
  // the writable shared-memory descriptor AND kSysFlag_SharedMemoryIsUAV is
  // published. Gate-off is the exact complement.
  ExportArmResult Run(bool geometry_emulated, bool uav_bound,
                      bool pixel_exporter = false) {
    ExportArmResult result;

    MetalShader* vertex_shader = LoadExporterShader();
    if (!vertex_shader || !vertex_shader->memexport_eM_written()) {
      result.error = "fixture VS does not write eM#";
      return result;
    }
    MetalShader::MetalTranslation* pixel = MakePixelTranslation(pixel_exporter);
    if (!pixel) {
      result.error = "guest PS translation failed";
      return result;
    }
    DxbcShaderTranslator::Modification vm(
        cache->shader_translator()->GetDefaultVertexShaderModification(
            vertex_shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters),
            HostVertexShaderType::kVertex));
    vm.vertex.interpolator_mask = 0;
    DxbcShaderTranslator::Modification pm(
        cache->shader_translator()->GetDefaultPixelShaderModification(0));
    pm.pixel.interpolator_mask = 0;
    pm.pixel.rt0_blend_rgb_factor_for_premult = xenos::BlendFactor::kOne;
    pm.pixel.rt0_blend_a_factor_for_premult = xenos::BlendFactor::kOne;
    auto* vertex_translation = static_cast<MetalShader::MetalTranslation*>(
        vertex_shader->GetOrCreateTranslation(vm.value));
    if (!vertex_translation->has_metal_function() &&
        !cache->TranslateShader(*vertex_translation)) {
      result.error = "exporter VS translation failed";
      return result;
    }

    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint32_t blends[xenos::kMaxColorRenderTargets];
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      colors[i] = uint64_t(MTLPixelFormatInvalid);
      blends[i] = 0x00010001;
    }
    colors[0] = uint64_t(MTLPixelFormatRGBA8Unorm);
    const uint64_t depth_format = uint64_t(MTLPixelFormatDepth32Float);

    void* geometry_pipeline = nullptr;
    id<MTLRenderPipelineState> plain_pipeline = nil;
    uint32_t gs_vertex_size = 0;
    uint32_t gs_max_input = 0;
    if (geometry_emulated) {
      geometry_pipeline = cache->GetOrCreateGeometryPipelineState(
          vertex_translation, pixel, PipelineGeometryShader::kRectangleList,
          vm, pm, colors, depth_format, MetalPipelineRasterKey{1}, blends,
          /*color_write_mask=*/0xF, &gs_vertex_size, &gs_max_input);
      if (!geometry_pipeline) {
        result.error = "GetOrCreateGeometryPipelineState returned null";
        return result;
      }
    } else {
      MTLRenderPipelineDescriptor* desc =
          [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction =
          (__bridge id<MTLFunction>)vertex_translation->metal_function();
      desc.fragmentFunction = (__bridge id<MTLFunction>)pixel->metal_function();
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      desc.depthAttachmentPixelFormat = (MTLPixelFormat)depth_format;
      desc.rasterSampleCount = 1;
      NSError* pipeline_error = nil;
      plain_pipeline =
          [device newRenderPipelineStateWithDescriptor:desc
                                                 error:&pipeline_error];
      if (!plain_pipeline) {
        result.error = pipeline_error
                           ? [[pipeline_error localizedDescription] UTF8String]
                           : "plain pipeline creation failed";
        return result;
      }
    }

    // --- Render targets + readback. ---
    MTLTextureDescriptor* cd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:kRT
                                    height:kRT
                                 mipmapped:NO];
    cd.usage = MTLTextureUsageRenderTarget;
    cd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> color_rt = [device newTextureWithDescriptor:cd];
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:(MTLPixelFormat)depth_format
                                     width:kRT
                                    height:kRT
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth_rt = [device newTextureWithDescriptor:dd];
    id<MTLBuffer> readback = NewBuffer(kRT * kRT * 4, 0, @"Readback");
    id<MTLBuffer> depth_readback = NewBuffer(kRT * kRT * 4, 0, @"Depth");
    if (!color_rt || !depth_rt || !readback || !depth_readback) {
      result.error = "RT/readback alloc failed";
      return result;
    }

    // --- System constants. ---
    DxbcShaderTranslator::SystemConstants system_constants;
    std::memset(&system_constants, 0, sizeof(system_constants));
    system_constants.flags =
        DxbcShaderTranslator::kSysFlag_WNotReciprocal |
        DxbcShaderTranslator::kSysFlag_AlphaPassIfLess |
        DxbcShaderTranslator::kSysFlag_AlphaPassIfEqual |
        DxbcShaderTranslator::kSysFlag_AlphaPassIfGreater;
    if (uav_bound) {
      system_constants.flags |=
          DxbcShaderTranslator::kSysFlag_SharedMemoryIsUAV;
    }
    system_constants.vertex_index_endian = xenos::Endian::kNone;
    system_constants.vertex_index_offset = 0;
    system_constants.vertex_index_min = 0;
    system_constants.vertex_index_max = 0xFFFFFFu;
    system_constants.ndc_scale[0] = 1.0f;
    system_constants.ndc_scale[1] = 1.0f;
    system_constants.ndc_scale[2] = 1.0f;
    for (uint32_t i = 0; i < 4; ++i) {
      system_constants.color_exp_bias[i] = 1.0f;
    }
    id<MTLBuffer> system_buffer =
        NewBuffer(sizeof(system_constants), 0, @"System Constants");
    if (!system_buffer) {
      result.error = "system constants alloc failed";
      return result;
    }
    std::memcpy([system_buffer contents], &system_constants,
                sizeof(system_constants));

    // --- Constant banks. The translated shader consumes a PACKED vertex float
    // bank (metal_binding_layout.h PackMetalFloatConstants), and the fixture
    // uses c0..c3 contiguously, so packed slot k IS c[k]. ---
    id<MTLBuffer> vertex_floats = NewBuffer(4096, 0, @"Vertex floats");
    id<MTLBuffer> pixel_floats = NewBuffer(4096, 0, @"Pixel floats");
    id<MTLBuffer> cbv_scratch = NewBuffer(4096, 0, @"CBV scratch");
    id<MTLBuffer> null_buffer = NewBuffer(256, 0, @"Null");
    id<MTLBuffer> shared_buffer =
        NewBuffer(kSharedBufferSize, kPoisonByte, @"Shared memory (poisoned)");
    if (!vertex_floats || !pixel_floats || !cbv_scratch || !null_buffer ||
        !shared_buffer) {
      result.error = "constant/shared alloc failed";
      return result;
    }
    {
      auto* words = static_cast<uint32_t*>([vertex_floats contents]);
      // c0: the eA index multiplier -- (0, 1, 0, 0) so eA = idx*c0 + c1 puts
      // the vertex index in the eA.y mantissa and leaves the other three
      // descriptor dwords bit-exact.
      words[0] = FloatBits(0.0f);
      words[1] = FloatBits(1.0f);
      words[2] = FloatBits(0.0f);
      words[3] = FloatBits(0.0f);
      // c1: the stream descriptor.
      std::array<uint32_t, 4> stream = BuildStreamConstant();
      std::memcpy(words + 4, stream.data(), sizeof(stream));
      // c2/c3: position scale/bias -> a large on-screen triangle.
      words[8] = FloatBits(2.0f);
      words[9] = FloatBits(1.5f);
      words[10] = FloatBits(0.0f);
      words[11] = FloatBits(0.0f);
      words[12] = FloatBits(-1.0f);
      words[13] = FloatBits(-1.0f);
      words[14] = FloatBits(0.0f);
      words[15] = FloatBits(0.0f);
    }
    {
      float* c0 = static_cast<float*>([pixel_floats contents]);
      c0[0] = 1.0f;
      c0[1] = 1.0f;
      c0[2] = 0.0f;
      c0[3] = 1.0f;
      // The pixel exporter's c1 (stream descriptor) and c2 (export data);
      // the plain pixel program reads only c0.
      std::array<uint32_t, 4> stream = BuildPixelStreamConstant();
      std::memcpy(c0 + 4, stream.data(), sizeof(stream));
      std::memcpy(c0 + 8, kPixelExportValue.data(), sizeof(kPixelExportValue));
    }

    // --- Descriptor heaps. ---
    MTLTextureDescriptor* ntd = [[MTLTextureDescriptor alloc] init];
    ntd.textureType = MTLTextureType2DArray;
    ntd.pixelFormat = MTLPixelFormatRGBA8Unorm;
    ntd.width = 1;
    ntd.height = 1;
    ntd.arrayLength = 1;
    ntd.usage = MTLTextureUsageShaderRead;
    ntd.storageMode = MTLStorageModeShared;
    id<MTLTexture> null_texture = [device newTextureWithDescriptor:ntd];
    id<MTLSamplerState> null_sampler = [device
        newSamplerStateWithDescriptor:[[MTLSamplerDescriptor alloc] init]];
    constexpr uint32_t kHeapSlots = 8;
    id<MTLBuffer> view_heap =
        NewBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), 0, @"Views");
    id<MTLBuffer> sampler_heap =
        NewBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), 0, @"Samplers");
    id<MTLBuffer> system_views =
        NewBuffer(3 * sizeof(IRDescriptorTableEntry), 0, @"System views");
    if (!null_texture || !null_sampler || !view_heap || !sampler_heap ||
        !system_views) {
      result.error = "heap alloc failed";
      return result;
    }
    auto* view_entries =
        static_cast<IRDescriptorTableEntry*>([view_heap contents]);
    auto* sampler_entries =
        static_cast<IRDescriptorTableEntry*>([sampler_heap contents]);
    for (uint32_t i = 0; i < kHeapSlots; ++i) {
      IRDescriptorTableSetTexture(&view_entries[i], null_texture, 0.0f, 0);
      IRDescriptorTableSetSampler(&sampler_entries[i], null_sampler, 0.0f);
    }
    auto* system_view_entries =
        static_cast<IRDescriptorTableEntry*>([system_views contents]);
    // 0 = shared-memory SRV, 1 = null UAV, 2 = shared-memory UAV -- the exact
    // three-entry system view table the backend builds.
    IRDescriptorTableSetBuffer(&system_view_entries[0],
                               [shared_buffer gpuAddress], kSharedBufferSize);
    IRDescriptorTableSetBuffer(&system_view_entries[1],
                               [null_buffer gpuAddress], 256);
    IRDescriptorTableSetBuffer(&system_view_entries[2],
                               [shared_buffer gpuAddress], kSharedBufferSize);

    // --- Top-level AB from the PURE production builder. ---
    MetalBindingAddresses addresses;
    addresses.null_buffer = [null_buffer gpuAddress];
    addresses.view_heap = [view_heap gpuAddress];
    addresses.sampler_heap = [sampler_heap gpuAddress];
    addresses.shared_memory_srv = [system_views gpuAddress];
    addresses.null_uav =
        [system_views gpuAddress] + sizeof(IRDescriptorTableEntry);
    addresses.shared_memory_uav =
        [system_views gpuAddress] + 2 * sizeof(IRDescriptorTableEntry);
    addresses.system_constants = [system_buffer gpuAddress];
    addresses.float_vertex = [vertex_floats gpuAddress];
    addresses.bool_loop = [cbv_scratch gpuAddress];
    addresses.fetch = [cbv_scratch gpuAddress];
    addresses.descriptor_indices_vertex = [cbv_scratch gpuAddress];
    addresses.float_pixel = [pixel_floats gpuAddress];
    addresses.descriptor_indices_pixel = [cbv_scratch gpuAddress];
    std::array<uint64_t, kMetalBindingSlotCount> argument_buffer =
        BuildMetalTopLevelArgumentBuffer(addresses,
                                         /*pixel_shader_present=*/true,
                                         /*memexport_draw=*/uav_bound);
    id<MTLBuffer> ab_buffer =
        NewBuffer(sizeof(argument_buffer), 0, @"Top-level AB");
    if (!ab_buffer) {
      result.error = "AB alloc failed";
      return result;
    }
    std::memcpy([ab_buffer contents], argument_buffer.data(),
                sizeof(argument_buffer));

    // --- Encode. ---
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color_rt;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    pass.depthAttachment.texture = depth_rt;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 1.0;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    if (!encoder) {
      result.error = "no encoder";
      return result;
    }
    [encoder setRenderPipelineState:
                 geometry_emulated
                     ? (__bridge id<MTLRenderPipelineState>)geometry_pipeline
                     : plain_pipeline];
    MTLDepthStencilDescriptor* dssd = [[MTLDepthStencilDescriptor alloc] init];
    dssd.depthCompareFunction = MTLCompareFunctionAlways;
    dssd.depthWriteEnabled = YES;
    [encoder setDepthStencilState:
                 [device newDepthStencilStateWithDescriptor:dssd]];
    auto bind_all_stages = [&](id<MTLBuffer> buffer, NSUInteger index) {
      if (geometry_emulated) {
        [encoder setObjectBuffer:buffer offset:0 atIndex:index];
        [encoder setMeshBuffer:buffer offset:0 atIndex:index];
      } else {
        [encoder setVertexBuffer:buffer offset:0 atIndex:index];
      }
      [encoder setFragmentBuffer:buffer offset:0 atIndex:index];
    };
    bind_all_stages(ab_buffer, kIRArgumentBufferBindPoint);
    bind_all_stages(view_heap, kIRDescriptorHeapBindPoint);
    bind_all_stages(sampler_heap, kIRSamplerHeapBindPoint);
    if (geometry_emulated) {
      IRRuntimeVertexBuffers vertex_buffers = {};
      [encoder setObjectBytes:vertex_buffers
                       length:sizeof(vertex_buffers)
                      atIndex:kIRVertexBufferBindPoint];
    }
    // The production stage mask (metal_binding_layout.h) -- the SAME mask the
    // command processor derives, so this arm cannot pass with a binding the
    // backend would not produce.
    const uint32_t production_stage_mask =
        metal::GetMetalBindingStageMask(geometry_emulated);
    MTLRenderStages stages = 0;
    if (production_stage_mask & metal::kMetalBindingStageVertex) {
      stages |= MTLRenderStageVertex;
    }
    if (production_stage_mask & metal::kMetalBindingStageObject) {
      stages |= MTLRenderStageObject;
    }
    if (production_stage_mask & metal::kMetalBindingStageMesh) {
      stages |= MTLRenderStageMesh;
    }
    if (production_stage_mask & metal::kMetalBindingStageFragment) {
      stages |= MTLRenderStageFragment;
    }
    for (id<MTLBuffer> resource : {ab_buffer, view_heap, sampler_heap,
                                   system_views, system_buffer, vertex_floats,
                                   pixel_floats, cbv_scratch, null_buffer}) {
      [encoder useResource:resource usage:MTLResourceUsageRead stages:stages];
    }
    [encoder useResource:shared_buffer
                   usage:uav_bound
                             ? (MTLResourceUsageRead | MTLResourceUsageWrite)
                             : MTLResourceUsageRead
                  stages:stages];
    [encoder useResource:null_texture
                   usage:MTLResourceUsageRead
                  stages:stages];

    if (geometry_emulated) {
      IRRuntimeGeometryPipelineConfig geometry_config = {};
      geometry_config.gsVertexSizeInBytes = gs_vertex_size;
      geometry_config.gsMaxInputPrimitivesPerMeshThreadgroup = gs_max_input;
      IRRuntimeDrawPrimitivesGeometryEmulation(
          encoder, IRRuntimePrimitiveTypeTriangle, geometry_config,
          /*instanceCount=*/1, /*vertexCountPerInstance=*/kDrawVertexCount,
          /*baseVertex=*/0, /*baseInstance=*/0);
    } else {
      IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle,
                              /*vertexStart=*/(uint64_t)0,
                              /*vertexCount=*/(uint64_t)kDrawVertexCount,
                              /*instanceCount=*/(uint64_t)1,
                              /*baseInstance=*/(uint64_t)0);
    }
    [encoder endEncoding];

    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    [blit copyFromTexture:color_rt
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(kRT, kRT, 1)
                    toBuffer:readback
           destinationOffset:0
      destinationBytesPerRow:kRT * 4
    destinationBytesPerImage:kRT * kRT * 4];
    [blit copyFromTexture:depth_rt
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(kRT, kRT, 1)
                    toBuffer:depth_readback
           destinationOffset:0
      destinationBytesPerRow:kRT * 4
    destinationBytesPerImage:kRT * kRT * 4];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted ||
        command_buffer.error) {
      result.error = command_buffer.error
                         ? [[command_buffer.error localizedDescription]
                               UTF8String]
                         : "command buffer not completed";
      return result;
    }

    result.shared.resize(kSharedBufferSize);
    std::memcpy(result.shared.data(), [shared_buffer contents],
                kSharedBufferSize);
    result.rgba.resize(kRT * kRT * 4);
    std::memcpy(result.rgba.data(), [readback contents], result.rgba.size());
    result.depth.resize(kRT * kRT);
    std::memcpy(result.depth.data(), [depth_readback contents],
                result.depth.size() * sizeof(float));
    for (uint32_t i = 0; i < kRT * kRT; ++i) {
      const uint8_t* p = result.rgba.data() + i * 4;
      if (p[0] | p[1] | p[2] | p[3]) {
        ++result.covered;
      }
    }
    result.ok = true;
    return result;
  }
};

ExportHarness& H() {
  static ExportHarness harness;
  return harness;
}

// Counts bytes that still hold the poison outside [begin, end).
uint32_t PoisonIntactOutside(const std::vector<uint8_t>& bytes, uint32_t begin,
                             uint32_t end, uint32_t* out_total) {
  uint32_t intact = 0, total = 0;
  for (uint32_t i = 0; i < bytes.size(); ++i) {
    if (i >= begin && i < end) {
      continue;
    }
    ++total;
    if (bytes[i] == kPoisonByte) {
      ++intact;
    }
  }
  *out_total = total;
  return intact;
}

void RunPoisonThenProve(bool geometry_emulated) {
  ExportHarness& h = H();
  INFO(h.error);
  REQUIRE(h.ok);
  const char* name = geometry_emulated ? "geometry-emulated" : "plain";

  ExportArmResult off = h.Run(geometry_emulated, /*uav_bound=*/false);
  ExportArmResult on = h.Run(geometry_emulated, /*uav_bound=*/true);
  if (!off.ok || !on.ok) {
    std::printf("MXRASTER|%s|ERROR|off=%s on=%s\n", name,
                off.ok ? "-" : off.error.c_str(),
                on.ok ? "-" : on.error.c_str());
    std::fflush(stdout);
    CHECK(off.ok);
    CHECK(on.ok);
    return;
  }

  // 1. Gate-off writes NOTHING: the whole stand-in shared memory is still
  //    poison. This is the exact-rollback property at the byte level.
  uint32_t off_total = 0;
  uint32_t off_intact =
      PoisonIntactOutside(off.shared, /*begin=*/0, /*end=*/0, &off_total);
  const bool off_untouched = off_intact == off_total;

  // 2. Gate-on: every INDEXED lane matches the oracle exactly.
  uint32_t lanes_matching = 0;
  for (uint32_t i = 0; i < kDrawVertexCount; ++i) {
    std::array<uint32_t, 4> oracle = OracleElement(i);
    std::array<uint32_t, 4> actual = {};
    std::memcpy(actual.data(), on.shared.data() + kStreamBaseBytes +
                                   i * kElementBytes,
                sizeof(actual));
    if (actual == oracle) {
      ++lanes_matching;
    } else {
      INFO("lane " << i << " expected " << oracle[0] << "," << oracle[1]
                   << " got " << actual[0] << "," << actual[1]);
      CHECK(actual == oracle);
    }
  }

  // 3. Declared-but-unindexed lanes (6, 7) still hold the poison -- no
  //    over-wide store, no whole-range clear.
  uint32_t unindexed_intact = 0;
  const uint32_t unindexed_bytes =
      (kStreamIndexCount - kDrawVertexCount) * kElementBytes;
  for (uint32_t i = 0; i < unindexed_bytes; ++i) {
    if (on.shared[kStreamBaseBytes + kDrawVertexCount * kElementBytes + i] ==
        kPoisonByte) {
      ++unindexed_intact;
    }
  }

  // 4. Everything outside the declared stream extent is untouched.
  uint32_t outside_total = 0;
  uint32_t outside_intact =
      PoisonIntactOutside(on.shared, kStreamBaseBytes,
                          kStreamBaseBytes + kStreamBytes, &outside_total);

  // 5. RASTER output is byte-identical between the arms: routing the exports
  //    must not perturb a single pixel or depth sample (design D1 -- the
  //    ORIGINAL draw, no shadow draw, no state change).
  const bool color_identical = on.rgba == off.rgba;
  const bool depth_identical = on.depth == off.depth;

  std::printf(
      "MXRASTER|%s|%s|off_poison=%u/%u lanes=%u/%u unindexed_poison=%u/%u "
      "outside_poison=%u/%u covered_off=%u covered_on=%u color_same=%d "
      "depth_same=%d\n",
      name,
      (off_untouched && lanes_matching == kDrawVertexCount &&
       unindexed_intact == unindexed_bytes && outside_intact == outside_total &&
       color_identical && depth_identical)
          ? "PASS"
          : "FAIL",
      off_intact, off_total, lanes_matching, kDrawVertexCount,
      unindexed_intact, unindexed_bytes, outside_intact, outside_total,
      off.covered, on.covered, color_identical ? 1 : 0,
      depth_identical ? 1 : 0);
  std::fflush(stdout);

  CHECK(off_untouched);
  CHECK(lanes_matching == kDrawVertexCount);
  CHECK(unindexed_intact == unindexed_bytes);
  CHECK(outside_intact == outside_total);
  CHECK(color_identical);
  CHECK(depth_identical);
  // The draw must actually RASTERIZE -- an empty draw would satisfy every
  // "unchanged" check above vacuously.
  CHECK(on.covered > 0);
}

// Pixel-stage producer: the fragment stage's eM0 store lands through the same
// top-level argument buffer slot (u0/space0) and residency the vertex route
// uses, and the gate-off arm writes nothing.
void RunPixelPoisonThenProve(bool geometry_emulated) {
  ExportHarness& h = H();
  INFO(h.error);
  REQUIRE(h.ok);
  const char* name = geometry_emulated ? "pixel-geometry-emulated" : "pixel";
  ExportArmResult off =
      h.Run(geometry_emulated, /*uav_bound=*/false, /*pixel_exporter=*/true);
  ExportArmResult on =
      h.Run(geometry_emulated, /*uav_bound=*/true, /*pixel_exporter=*/true);
  if (!off.ok || !on.ok) {
    std::printf("MXRASTER|%s|ERROR|off=%s on=%s\n", name,
                off.ok ? "-" : off.error.c_str(),
                on.ok ? "-" : on.error.c_str());
    std::fflush(stdout);
    CHECK(off.ok);
    CHECK(on.ok);
    return;
  }
  uint32_t off_total = 0;
  const bool off_untouched =
      PoisonIntactOutside(off.shared, 0, 0, &off_total) == off_total;
  std::array<uint32_t, 4> oracle;
  std::memcpy(oracle.data(), kPixelExportValue.data(), sizeof(oracle));
  std::array<uint32_t, 4> actual = {};
  std::memcpy(actual.data(), on.shared.data() + kPixelStreamBaseBytes,
              sizeof(actual));
  // Elements 1..3 of the pixel stream are never indexed.
  uint32_t unindexed_intact = 0;
  const uint32_t unindexed_bytes = kPixelStreamBytes - kElementBytes;
  for (uint32_t i = 0; i < unindexed_bytes; ++i) {
    if (on.shared[kPixelStreamBaseBytes + kElementBytes + i] == kPoisonByte) {
      ++unindexed_intact;
    }
  }
  // The vertex exporter's stream holds its oracle too (both stages route
  // together), and nothing outside the two streams is touched.
  uint32_t vertex_lanes = 0;
  for (uint32_t i = 0; i < kDrawVertexCount; ++i) {
    std::array<uint32_t, 4> v = {};
    std::memcpy(v.data(),
                on.shared.data() + kStreamBaseBytes + i * kElementBytes,
                sizeof(v));
    vertex_lanes += v == OracleElement(i) ? 1 : 0;
  }
  uint32_t outside_total = 0, outside_intact = 0;
  for (uint32_t i = 0; i < on.shared.size(); ++i) {
    if ((i >= kStreamBaseBytes && i < kStreamBaseBytes + kStreamBytes) ||
        (i >= kPixelStreamBaseBytes &&
         i < kPixelStreamBaseBytes + kPixelStreamBytes)) {
      continue;
    }
    ++outside_total;
    outside_intact += on.shared[i] == kPoisonByte ? 1 : 0;
  }
  const bool color_identical = on.rgba == off.rgba;
  std::printf(
      "MXRASTER|%s|%s|off_poison=%u/%u element0=%d unindexed_poison=%u/%u "
      "vertex_lanes=%u/%u outside_poison=%u/%u covered=%u color_same=%d\n",
      name,
      (off_untouched && actual == oracle &&
       unindexed_intact == unindexed_bytes &&
       vertex_lanes == kDrawVertexCount && outside_intact == outside_total &&
       color_identical && on.covered > 0)
          ? "PASS"
          : "FAIL",
      PoisonIntactOutside(off.shared, 0, 0, &off_total), off_total,
      actual == oracle ? 1 : 0, unindexed_intact, unindexed_bytes,
      vertex_lanes, kDrawVertexCount, outside_intact, outside_total,
      on.covered, color_identical ? 1 : 0);
  std::fflush(stdout);
  CHECK(off_untouched);
  INFO("element0 got " << actual[0] << "," << actual[1] << "," << actual[2]
                       << "," << actual[3]);
  CHECK(actual == oracle);
  CHECK(unindexed_intact == unindexed_bytes);
  CHECK(vertex_lanes == kDrawVertexCount);
  CHECK(outside_intact == outside_total);
  CHECK(color_identical);
  CHECK(on.covered > 0);
}

}  // namespace

TEST_CASE("pixel exporter fixture writes eA and eM0",
          "[memexport][raster][pixel][fixture]") {
  std::vector<uint32_t> host = BuildPixelExporterUcode();
  MetalShader shader(xenos::ShaderType::kPixel, 0x4D58504958454C31ull,
                     host.data(), host.size(), std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);
  std::string text = disasm.to_string();
  INFO(text);
  CHECK(shader.memexport_eM_written() == 0b1);
  REQUIRE(shader.memexport_stream_constants().size() == 1);
  CHECK(*shader.memexport_stream_constants().begin() == 1u);
  CHECK(text.find("oC0") != std::string::npos);
}

TEST_CASE("poison-then-prove: pixel-stage producer",
          "[memexport][raster][pixel][render][device]") {
  RunPixelPoisonThenProve(/*geometry_emulated=*/false);
}

TEST_CASE("poison-then-prove: pixel-stage producer, geometry-emulated",
          "[memexport][raster][pixel][render][device]") {
  RunPixelPoisonThenProve(/*geometry_emulated=*/true);
}

TEST_CASE("rasterizing exporter fixture writes eA and eM0",
          "[memexport][raster][fixture]") {
  // The producer-class predicate the command processor keys on, plus the
  // stream-constant extraction that gives the CPU the destination range.
  // Direct MetalShader construction takes HOST-order words when the source
  // endianness is declared little (the guest-endian swap is LoadShader's job,
  // and the render arms above go through LoadShader).
  std::vector<uint32_t> host = BuildRasterizingExporterUcode();
  MetalShader shader(xenos::ShaderType::kVertex, 0x4D58524153544552ull,
                     host.data(), host.size(), std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);
  std::string text = disasm.to_string();
  INFO(text);
  CHECK(shader.memexport_eM_written() == 0b1);
  CHECK(text.find("eA") != std::string::npos);
  CHECK(text.find("eM0") != std::string::npos);
  // The eA mad must be recognized so the CPU can derive the destination range;
  // an unrecognized stream constant is exactly the "no ranges" unrouted reason.
  REQUIRE(shader.memexport_stream_constants().size() == 1);
  CHECK(*shader.memexport_stream_constants().begin() == 1u);
  // And it still rasterizes: the position export is present.
  CHECK(text.find("oPos") != std::string::npos);
}

TEST_CASE("poison-then-prove: plain rasterizing producer",
          "[memexport][raster][render][device]") {
  RunPoisonThenProve(/*geometry_emulated=*/false);
}

TEST_CASE("poison-then-prove: geometry-emulated rasterizing producer",
          "[memexport][raster][render][device]") {
  RunPoisonThenProve(/*geometry_emulated=*/true);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
