/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 on-device execution probe for the Halo 3 additive light-volume pixel
// shader (FC3CCF4CD1C76640, trace t00015990 draw 991).
//
// Live observation (docs/metal_live_compatibility_matrix.md, Halo 3): on
// Metal the draw lights the character's face while Vulkan does not, and
// forcing the shadow-test boolean b141 OFF renders the identical frame -- the
// 16-tap point PCF block under `cexec b141` is inert. Every input the render
// path feeds it was verified equal to Vulkan (shadow snapshot, constants,
// bools, fetch constants, sampler, interpolator mask), so what remains is the
// translated shader itself: ucode -> DxbcShaderTranslator -> dxilconv ->
// MetalShaderConverter, executed on this GPU.
//
// This probe runs the REAL pixel microcode with the REAL captured constants
// (metal_halo3_light_fixture.h) under a hand-built vertex program that exports
// controlled interpolants, with synthetic textures: a 512x512 R32Float shadow
// map whose value is the arm's occluder depth and a fixed receiver depth of
// 0.5 in the shadow projection interpolant. Three arms:
//
//   lit      : occluder 1.0 >= receiver 0.5 -> every tap passes  -> factor 1
//   shadowed : occluder 0.0 <  receiver 0.5 -> every tap fails   -> factor 0
//   b141-off : the block is skipped                              -> factor 1
//
// A correct chain gives lit == b141-off and shadowed darker than lit. If the
// Metal chain gives shadowed == lit, the live inertness is reproduced in
// isolation and the ucode can be bisected instruction by instruction here.
//
// Harness lineage: metal_memexport_raster_render_test.mm (production
// MetalPipelineCache translation, PackMetalFloatConstants banks, the
// IRDescriptorTableEntry heaps and BuildMetalTopLevelArgumentBuffer).

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/byte_order.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/conformance/metal_halo3_light_fixture.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/metal_binding_layout.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
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
using metal::MetalShader;
using HostVertexShaderType = Shader::HostVertexShaderType;

constexpr uint32_t kRT = 64;
constexpr uint32_t kShadowSize = 512;
constexpr float kReceiverDepth = 0.5f;
// Bool constant b141 = word 4 bit 13 of the bool bank.
constexpr uint32_t kShadowBoolWord = 4;
constexpr uint32_t kShadowBoolBit = 13;

constexpr uint32_t kAluRetainPrev =
    uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26;
// Component-relative swizzle bytes: 2 bits per component, (src - dst) & 3.
constexpr uint32_t kSwizzleYXZW = 0x0D;

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

// Vertex program: a full-screen triangle whose six interpolators are the
// float constants c5..c10 verbatim.
//
//   CF 0: exec    @4 x1  -> mul r0.y, r0.xxxx, r0.xxxx      (y = i*i)
//   CF 1: alloc position
//   CF 2: exec    @5 x3  -> mul r2, r0, c2                  (i * c2)
//                          mad r1, r0.yxzw, c3, c4          (i*i * c3 + c4)
//                          add oPos.xyz1, r1, r2
//   CF 3: alloc interpolators
//   CF 4: exec    @8 x3  -> max o0, c5, c5 ... max o2, c7, c7
//   CF 5: execEnd @11 x3 -> max o3, c8, c8 ... max o5, c10, c10
//
// oPos.x = c2.x*i + c3.x*i*i + c4.x and oPos.y = c2.y*i*i + c3.y*i + c4.y, so
// c2 = (8, 2, 0, 0), c3 = (-4, -2, 0, 0), c4 = (-1, -1, 0, 0) put the three
// vertices at (-1,-1), (3,-1), (-1,3): the whole viewport.
std::vector<uint32_t> BuildProbeVertexUcode() {
  std::vector<uint32_t> u(14 * 3, 0);
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
  auto cf2 = exec(5, 3, false);
  PackControlFlow(u, 2, cf2.first, cf2.second);
  auto cf3 = alloc(ucode::AllocType::kVsInterpolators);
  PackControlFlow(u, 3, cf3.first, cf3.second);
  auto cf4 = exec(8, 3, false);
  PackControlFlow(u, 4, cf4.first, cf4.second);
  auto cf5 = exec(11, 3, true);
  PackControlFlow(u, 5, cf5.first, cf5.second);

  const uint32_t kmul = uint32_t(ucode::AluVectorOpcode::kMul) << 24;
  const uint32_t kmad = uint32_t(ucode::AluVectorOpcode::kMad) << 24;
  const uint32_t kadd = uint32_t(ucode::AluVectorOpcode::kAdd) << 24;
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t src1_temp = 1u << 31;
  const uint32_t src2_temp = 1u << 30;
  constexpr uint32_t kSwizzleXXXX = 0x6C;

  uint32_t* a = u.data() + 4 * 3;
  // @4: mul r0.y, r0.xxxx, r0.xxxx
  a[0] = 0u | (0b0010u << 16) | kAluRetainPrev;
  a[1] = (kSwizzleXXXX << 8) | (kSwizzleXXXX << 16);
  a[2] = kmul | src1_temp | src2_temp;
  // @5: mul r2, r0, c2
  a[3] = 2u | (0b1111u << 16) | kAluRetainPrev;
  a[4] = 0;
  a[5] = (2u << 8) /*src2=c2*/ | (0u << 16) /*src1=r0*/ | kmul | src1_temp;
  // @6: mad r1, r0.yxzw, c3, c4
  a[6] = 1u | (0b1111u << 16) | kAluRetainPrev;
  a[7] = kSwizzleYXZW << 16;
  a[8] = 4u /*src3=c4*/ | (3u << 8) /*src2=c3*/ | (0u << 16) /*src1=r0*/ |
         kmad | src1_temp;
  // @7: add oPos.xyz1, r1, r2 (w: vector + scalar mask both set = const 1)
  a[9] = 62u | (1u << 15) | (0b1111u << 16) | (0b1000u << 20) |
         kAluRetainPrev;
  a[10] = 0;
  a[11] = (2u << 8) /*src2=r2*/ | (1u << 16) /*src1=r1*/ | kadd | src1_temp |
          src2_temp;
  // @8..@13: max oN, c(5+N), c(5+N)
  for (uint32_t n = 0; n < 6; ++n) {
    uint32_t* e = a + 12 + n * 3;
    const uint32_t c = 5 + n;
    e[0] = n | (1u << 15) | (0b1111u << 16) | kAluRetainPrev;
    e[1] = 0;
    e[2] = (c << 8) | (c << 16) | kmax;
  }
  return u;
}

// Diagnostic stand-in PS: oC0 = (c0.x, c0.y, 1, 1) (the memexport fixture's).
std::vector<uint32_t> BuildConstantPixelUcode() {
  const uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kPsColors) << 9) |
                            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  const uint32_t exec_d0 = 1u | (1u << 12);
  const uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;
  std::vector<uint32_t> u(6, 0);
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = ((exec_d0 >> 16) & 0xFFFF) | ((exec_d1 & 0xFFFF) << 16);
  u[3] = 0u | (1u << 15) | (0xFu << 16) | (0b1100u << 20) | kAluRetainPrev;
  u[4] = 0;
  u[5] = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  return u;
}

// Diagnostic VS in the memexport fixture's exact shape (no interpolators):
// oPos = (i, i*i) * c2 + c3.
std::vector<uint32_t> BuildFixtureStyleVertexUcode() {
  std::vector<uint32_t> u(18, 0);
  auto exec = [](uint32_t address, uint32_t count, bool end) {
    return std::make_pair(
        address | (count << 12),
        uint32_t(end ? ucode::ControlFlowOpcode::kExecEnd
                     : ucode::ControlFlowOpcode::kExec)
            << 12);
  };
  auto cf0 = exec(4, 1, false);
  PackControlFlow(u, 0, cf0.first, cf0.second);
  PackControlFlow(u, 1, 0,
                  (uint32_t(ucode::AllocType::kVsPosition) << 9) |
                      (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12));
  auto cf2 = exec(5, 1, true);
  PackControlFlow(u, 2, cf2.first, cf2.second);
  const uint32_t kmul = uint32_t(ucode::AluVectorOpcode::kMul) << 24;
  const uint32_t kmad = uint32_t(ucode::AluVectorOpcode::kMad) << 24;
  u[12] = 0u | (0b0010u << 16) | kAluRetainPrev;
  u[13] = (0x6Cu << 8) | (0x6Cu << 16);
  u[14] = kmul | (1u << 31) | (1u << 30);
  u[15] = 62u | (1u << 15) | (0b1111u << 16) | (0b1000u << 20) |
          kAluRetainPrev;
  u[16] = 0;
  u[17] = 3u | (2u << 8) | (0u << 16) | kmad | (1u << 31);
  return u;
}

std::vector<uint32_t> ToGuestEndian(const uint32_t* host_words, size_t count) {
  std::vector<uint32_t> guest(count);
  for (size_t i = 0; i < count; ++i) {
    guest[i] = xe::byte_swap(host_words[i]);
  }
  return guest;
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct Float4 {
  float x, y, z, w;
};

struct ProbeInputs {
  // Interpolators o0..o5 -> pixel r0..r5 (r6 is the param-gen pixel position).
  std::array<Float4, 6> interpolators;
  float normal[3] = {0.0f, 0.0f, 1.0f};  // tf11 texel, encoded n*0.5+0.5.
  float shadow_depth = 1.0f;             // tf3 texel (R32Float).
  bool shadow_block = true;              // b141.
};

struct ProbeResult {
  bool ok = false;
  std::string error;
  std::vector<float> c0;  // kRT*kRT*4 floats, oC0.
  std::vector<float> c1;  // oC1.
  float mean0[4] = {};
  float center0[4] = {};
  uint32_t covered = 0;
};

struct ProbeHarness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  bool ok = false;
  std::string error;

  // Built once and reused by every arm.
  MetalShader* vertex_shader = nullptr;
  MetalShader* pixel_shader = nullptr;
  MetalShader::MetalTranslation* vertex_translation = nullptr;
  MetalShader::MetalTranslation* pixel_translation = nullptr;
  id<MTLRenderPipelineState> pipeline = nil;
  id<MTLTexture> white_texture = nil;
  id<MTLTexture> scene_texture = nil;
  id<MTLSamplerState> point_sampler = nil;
  id<MTLSamplerState> null_sampler = nil;
  id<MTLTexture> null_texture = nil;
  bool fixture_vs = false;
  bool mask_zero = false;
  bool one_rt = false;
  bool rgba8 = false;
  bool depth_write = false;
  bool manual_vs_bank = false;

  ProbeHarness() {
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
    ok = Build();
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

  // The translated 2D fetch samples a Texture2DArray -- every bound texture
  // is a one-layer array, like the production texture cache's views.
  id<MTLTexture> NewTexture2DArray(uint32_t width, uint32_t height,
                                   MTLPixelFormat format, const void* texels,
                                   uint32_t bytes_per_texel) {
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DArray;
    td.pixelFormat = format;
    td.width = width;
    td.height = height;
    td.arrayLength = 1;
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:td];
    if (texture && texels) {
      [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                 mipmapLevel:0
                       slice:0
                   withBytes:texels
                 bytesPerRow:width * bytes_per_texel
               bytesPerImage:width * height * bytes_per_texel];
    }
    return texture;
  }

  bool Build() {
    // --- Shaders: the probe VS and the REAL Halo 3 light PS. ---
    static const std::vector<uint32_t> vs_ucode = [] {
      std::vector<uint32_t> host = std::getenv("XE_H3PROBE_FIXTURE_VS")
                                       ? BuildFixtureStyleVertexUcode()
                                       : BuildProbeVertexUcode();
      return ToGuestEndian(host.data(), host.size());
    }();
    fixture_vs = std::getenv("XE_H3PROBE_FIXTURE_VS") != nullptr;
    mask_zero = std::getenv("XE_H3PROBE_MASK0") != nullptr;
    one_rt = std::getenv("XE_H3PROBE_ONE_RT") != nullptr;
    rgba8 = std::getenv("XE_H3PROBE_RGBA8") != nullptr;
    depth_write = std::getenv("XE_H3PROBE_DEPTHWRITE") != nullptr;
    manual_vs_bank = std::getenv("XE_H3PROBE_MANUAL_VS_BANK") != nullptr;
    static const std::vector<uint32_t> ps_ucode = [] {
      if (std::getenv("XE_H3PROBE_CONST_PS")) {
        std::vector<uint32_t> host = BuildConstantPixelUcode();
        return ToGuestEndian(host.data(), host.size());
      }
      return ToGuestEndian(halo3_light::kPixelUcodeHost,
                           sizeof(halo3_light::kPixelUcodeHost) /
                               sizeof(halo3_light::kPixelUcodeHost[0]));
    }();
    const bool const_ps = std::getenv("XE_H3PROBE_CONST_PS") != nullptr;
    vertex_shader = cache->LoadShader(xenos::ShaderType::kVertex,
                                      vs_ucode.data(),
                                      uint32_t(vs_ucode.size()));
    pixel_shader = cache->LoadShader(xenos::ShaderType::kPixel,
                                     ps_ucode.data(),
                                     uint32_t(ps_ucode.size()));
    if (!vertex_shader || !pixel_shader) {
      error = "LoadShader failed";
      return false;
    }
    for (MetalShader* shader : {vertex_shader, pixel_shader}) {
      if (!shader->is_ucode_analyzed()) {
        StringBuffer disasm;
        shader->AnalyzeUcode(disasm);
        if (shader == vertex_shader && std::getenv("XE_H3PROBE_DISASM")) {
          std::printf("H3LIGHT|vs-ucode|\n%s\n", disasm.to_string().c_str());
        }
      }
    }
    if (!const_ps &&
        pixel_shader->ucode_data_hash() != halo3_light::kPixelShaderHash) {
      error = "pixel ucode hash mismatch -- fixture corrupted";
      return false;
    }
    if (!fixture_vs && vertex_shader->writes_interpolators() != 0x3Fu) {
      error = "probe VS does not export o0..o5";
      return false;
    }

    const uint32_t kInterpolatorMask = (fixture_vs || mask_zero) ? 0 : 0x3F;
    DxbcShaderTranslator::Modification vm(
        cache->shader_translator()->GetDefaultVertexShaderModification(
            vertex_shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters),
            HostVertexShaderType::kVertex));
    vm.vertex.interpolator_mask = kInterpolatorMask;
    DxbcShaderTranslator::Modification pm(
        cache->shader_translator()->GetDefaultPixelShaderModification(
            pixel_shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters)));
    pm.pixel.interpolator_mask = kInterpolatorMask;
    // The shader reads |r6.xy| as the pixel position (add r0.zw, r_abs[6].xxyy,
    // c108) -- SQ_CONTEXT_MISC.param_gen_pos = 6 in the live draw.
    pm.pixel.param_gen_enable = 1;
    pm.pixel.param_gen_interpolator = 6;
    pm.pixel.param_gen_point = 0;
    pm.pixel.rt0_blend_rgb_factor_for_premult = xenos::BlendFactor::kOne;
    pm.pixel.rt0_blend_a_factor_for_premult = xenos::BlendFactor::kOne;
    vertex_translation = static_cast<MetalShader::MetalTranslation*>(
        vertex_shader->GetOrCreateTranslation(vm.value));
    pixel_translation = static_cast<MetalShader::MetalTranslation*>(
        pixel_shader->GetOrCreateTranslation(pm.value));
    if (!vertex_translation->has_metal_function() &&
        !cache->TranslateShader(*vertex_translation)) {
      error = "probe VS translation failed";
      return false;
    }
    if (!pixel_translation->has_metal_function() &&
        !cache->TranslateShader(*pixel_translation)) {
      error = "Halo 3 light PS translation failed";
      return false;
    }

    // --- Pipeline: two float color targets (the PS writes oC0 and oC1). ---
    MTLRenderPipelineDescriptor* desc =
        [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction =
        (__bridge id<MTLFunction>)vertex_translation->metal_function();
    desc.fragmentFunction =
        (__bridge id<MTLFunction>)pixel_translation->metal_function();
    const MTLPixelFormat color_format =
        rgba8 ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatRGBA32Float;
    desc.colorAttachments[0].pixelFormat = color_format;
    if (!one_rt) {
      desc.colorAttachments[1].pixelFormat = color_format;
    }
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    desc.rasterSampleCount = 1;
    NSError* pipeline_error = nil;
    pipeline = [device newRenderPipelineStateWithDescriptor:desc
                                                      error:&pipeline_error];
    if (!pipeline) {
      error = pipeline_error
                  ? [[pipeline_error localizedDescription] UTF8String]
                  : "pipeline creation failed";
      return false;
    }

    // --- Static textures + samplers. ---
    const uint8_t white[4] = {255, 255, 255, 255};
    white_texture = NewTexture2DArray(1, 1, MTLPixelFormatRGBA8Unorm, white, 4);
    std::vector<uint8_t> grey(kRT * kRT * 4, 128);
    for (uint32_t i = 3; i < grey.size(); i += 4) {
      grey[i] = 255;
    }
    scene_texture =
        NewTexture2DArray(kRT, kRT, MTLPixelFormatRGBA8Unorm, grey.data(), 4);
    null_texture = NewTexture2DArray(1, 1, MTLPixelFormatRGBA8Unorm, nullptr, 4);
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterNearest;
    sd.magFilter = MTLSamplerMinMagFilterNearest;
    sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.normalizedCoordinates = YES;
    sd.supportArgumentBuffers = YES;
    point_sampler = [device newSamplerStateWithDescriptor:sd];
    MTLSamplerDescriptor* nsd = [[MTLSamplerDescriptor alloc] init];
    nsd.supportArgumentBuffers = YES;
    null_sampler = [device newSamplerStateWithDescriptor:nsd];
    if (!white_texture || !scene_texture || !null_texture || !point_sampler ||
        !null_sampler) {
      error = "static texture/sampler alloc failed";
      return false;
    }
    return true;
  }

  static void FillFetchConstant(uint32_t* fetch_words, uint32_t fetch_constant,
                                uint32_t width, uint32_t height,
                                xenos::TextureFormat format) {
    xenos::xe_gpu_texture_fetch_t fetch;
    std::memset(&fetch, 0, sizeof(fetch));
    fetch.type = xenos::FetchConstantType::kTexture;
    fetch.clamp_x = xenos::ClampMode::kClampToEdge;
    fetch.clamp_y = xenos::ClampMode::kClampToEdge;
    fetch.clamp_z = xenos::ClampMode::kClampToEdge;
    fetch.pitch = (width + 31) >> 5;
    fetch.format = format;
    fetch.endianness = xenos::Endian::kNone;
    fetch.base_address = (0x04000000u + fetch_constant * 0x100000u) >> 12;
    fetch.size_2d.width = width - 1;
    fetch.size_2d.height = height - 1;
    fetch.swizzle = 0 | (1 << 3) | (2 << 6) | (3 << 9);
    fetch.mag_filter = xenos::TextureFilter::kPoint;
    fetch.min_filter = xenos::TextureFilter::kPoint;
    fetch.mip_filter = xenos::TextureFilter::kPoint;
    fetch.dimension = xenos::DataDimension::k2DOrStacked;
    std::memcpy(fetch_words + fetch_constant * 6, &fetch, sizeof(fetch));
  }

  ProbeResult Run(const ProbeInputs& in) {
    ProbeResult result;
    if (!ok) {
      result.error = error;
      return result;
    }

    // --- Per-arm textures: the shadow map and the normal buffer. ---
    std::vector<float> shadow(kShadowSize * kShadowSize, in.shadow_depth);
    id<MTLTexture> shadow_texture = NewTexture2DArray(
        kShadowSize, kShadowSize, MTLPixelFormatR32Float, shadow.data(), 4);
    std::vector<uint8_t> normal_texels(kRT * kRT * 4);
    for (uint32_t i = 0; i < kRT * kRT; ++i) {
      for (uint32_t c = 0; c < 3; ++c) {
        float v = in.normal[c] * 0.5f + 0.5f;
        normal_texels[i * 4 + c] =
            uint8_t(std::lround(std::min(std::max(v, 0.0f), 1.0f) * 255.0f));
      }
      normal_texels[i * 4 + 3] = 255;
    }
    id<MTLTexture> normal_texture = NewTexture2DArray(
        kRT, kRT, MTLPixelFormatRGBA8Unorm, normal_texels.data(), 4);
    if (!shadow_texture || !normal_texture) {
      result.error = "per-arm texture alloc failed";
      return result;
    }

    // --- Render targets + readback. ---
    auto new_rt = [&](MTLPixelFormat format) {
      MTLTextureDescriptor* d =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                             width:kRT
                                                            height:kRT
                                                         mipmapped:NO];
      d.usage = MTLTextureUsageRenderTarget;
      d.storageMode = MTLStorageModePrivate;
      return [device newTextureWithDescriptor:d];
    };
    const MTLPixelFormat color_format =
        rgba8 ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatRGBA32Float;
    const uint32_t texel_bytes = rgba8 ? 4 : 16;
    id<MTLTexture> color0 = new_rt(color_format);
    id<MTLTexture> color1 = new_rt(color_format);
    id<MTLTexture> depth_rt = new_rt(MTLPixelFormatDepth32Float);
    id<MTLBuffer> readback0 = NewBuffer(kRT * kRT * 16, 0, @"Readback0");
    id<MTLBuffer> readback1 = NewBuffer(kRT * kRT * 16, 0, @"Readback1");
    if (!color0 || !color1 || !depth_rt || !readback0 || !readback1) {
      result.error = "RT/readback alloc failed";
      return result;
    }

    // --- System constants. ---
    DxbcShaderTranslator::SystemConstants system_constants;
    std::memset(&system_constants, 0, sizeof(system_constants));
    system_constants.flags = DxbcShaderTranslator::kSysFlag_WNotReciprocal |
                             DxbcShaderTranslator::kSysFlag_AlphaPassIfLess |
                             DxbcShaderTranslator::kSysFlag_AlphaPassIfEqual |
                             DxbcShaderTranslator::kSysFlag_AlphaPassIfGreater;
    system_constants.vertex_index_endian = xenos::Endian::kNone;
    system_constants.vertex_index_max = 0xFFFFFFu;
    system_constants.ndc_scale[0] = 1.0f;
    system_constants.ndc_scale[1] = 1.0f;
    system_constants.ndc_scale[2] = 1.0f;
    for (uint32_t i = 0; i < 4; ++i) {
      system_constants.color_exp_bias[i] = 1.0f;
    }
    id<MTLBuffer> system_buffer =
        NewBuffer(sizeof(system_constants), 0, @"System Constants");
    std::memcpy([system_buffer contents], &system_constants,
                sizeof(system_constants));

    // --- Float banks: the captured draw-991 register file, with the probe
    // VS's c2..c10 overriding the (unused) vertex bank. Packed by the
    // production helper per each shader's constant map. ---
    RegisterFile regs;
    std::memset(regs.values, 0, sizeof(regs.values));
    for (uint32_t i = 0; i < 2048; ++i) {
      regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + i] =
          halo3_light::kFloatConstants[i];
    }
    auto set_vs_const = [&](uint32_t index, Float4 v) {
      uint32_t* w = &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + index * 4];
      w[0] = FloatBits(v.x);
      w[1] = FloatBits(v.y);
      w[2] = FloatBits(v.z);
      w[3] = FloatBits(v.w);
    };
    if (fixture_vs) {
      set_vs_const(2, {2.0f, 1.5f, 0.0f, 0.0f});
      set_vs_const(3, {-1.0f, -1.0f, 0.0f, 0.0f});
    } else {
      set_vs_const(2, {8.0f, 2.0f, 0.0f, 0.0f});
      set_vs_const(3, {-4.0f, -2.0f, 0.0f, 0.0f});
      set_vs_const(4, {-1.0f, -1.0f, 0.0f, 0.0f});
    }
    for (uint32_t n = 0; n < 6; ++n) {
      set_vs_const(5 + n, in.interpolators[n]);
    }
    id<MTLBuffer> vertex_floats = NewBuffer(4096, 0, @"Vertex floats");
    id<MTLBuffer> pixel_floats = NewBuffer(4096, 0, @"Pixel floats");
    if (manual_vs_bank) {
      // The fixture's layout: packed slot k IS c[2 + k] (c2, c3, c4, ...).
      std::memcpy([vertex_floats contents],
                  &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + 2 * 4],
                  9 * 16);
      std::printf("H3LIGHT|vsbank|");
      const float* f = static_cast<const float*>([vertex_floats contents]);
      for (uint32_t i = 0; i < 8; ++i) {
        std::printf("%g ", f[i]);
      }
      std::printf("\n");
    } else if (!metal::PackMetalFloatConstants(
            regs, vertex_shader->constant_register_map(),
            XE_GPU_REG_SHADER_CONSTANT_000_X, [vertex_floats contents], 4096)) {
      result.error = "PackMetalFloatConstants (vertex) failed";
      return result;
    }
    if (
        !metal::PackMetalFloatConstants(
            regs, pixel_shader->constant_register_map(),
            XE_GPU_REG_SHADER_CONSTANT_256_X, [pixel_floats contents], 4096)) {
      result.error = "PackMetalFloatConstants (pixel) failed";
      return result;
    }

    // --- Bool/loop bank: the captured words, b141 per arm. ---
    id<MTLBuffer> bool_loop = NewBuffer(256, 0, @"Bool/loop");
    {
      auto* w = static_cast<uint32_t*>([bool_loop contents]);
      std::memcpy(w, halo3_light::kBoolConstants, 8 * sizeof(uint32_t));
      std::memcpy(w + 8, halo3_light::kLoopConstants, 32 * sizeof(uint32_t));
      if (in.shadow_block) {
        w[kShadowBoolWord] |= 1u << kShadowBoolBit;
      } else {
        w[kShadowBoolWord] &= ~(1u << kShadowBoolBit);
      }
    }

    // --- Fetch constants for tf0..tf4, tf10, tf11 (formats as live). ---
    id<MTLBuffer> fetch_buffer = NewBuffer(48 * 16, 0, @"Fetch constants");
    {
      auto* w = static_cast<uint32_t*>([fetch_buffer contents]);
      for (uint32_t fc : {0u, 1u, 2u, 4u}) {
        FillFetchConstant(w, fc, 1, 1, xenos::TextureFormat::k_8_8_8_8);
      }
      FillFetchConstant(w, 3, kShadowSize, kShadowSize,
                        xenos::TextureFormat::k_24_8);
      FillFetchConstant(w, 10, kRT, kRT,
                        xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16);
      FillFetchConstant(w, 11, kRT, kRT, xenos::TextureFormat::k_2_10_10_10);
    }

    // --- Descriptor heaps: slot 0 null, 1 white, 2 scene, 3 normals, 4
    // shadow; sampler slot 1 = point/clamp. ---
    constexpr uint32_t kHeapSlots = 8;
    id<MTLBuffer> view_heap =
        NewBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), 0, @"Views");
    id<MTLBuffer> sampler_heap =
        NewBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), 0, @"Samplers");
    id<MTLBuffer> system_views =
        NewBuffer(3 * sizeof(IRDescriptorTableEntry), 0, @"System views");
    id<MTLBuffer> null_buffer = NewBuffer(256, 0, @"Null");
    id<MTLBuffer> shared_buffer = NewBuffer(64 * 1024, 0, @"Shared memory");
    auto* view_entries =
        static_cast<IRDescriptorTableEntry*>([view_heap contents]);
    auto* sampler_entries =
        static_cast<IRDescriptorTableEntry*>([sampler_heap contents]);
    for (uint32_t i = 0; i < kHeapSlots; ++i) {
      IRDescriptorTableSetTexture(&view_entries[i], null_texture, 0.0f, 0);
      IRDescriptorTableSetSampler(&sampler_entries[i], null_sampler, 0.0f);
    }
    IRDescriptorTableSetTexture(&view_entries[1], white_texture, 0.0f, 0);
    IRDescriptorTableSetTexture(&view_entries[2], scene_texture, 0.0f, 0);
    IRDescriptorTableSetTexture(&view_entries[3], normal_texture, 0.0f, 0);
    IRDescriptorTableSetTexture(&view_entries[4], shadow_texture, 0.0f, 0);
    IRDescriptorTableSetSampler(&sampler_entries[1], point_sampler, 0.0f);
    auto* system_view_entries =
        static_cast<IRDescriptorTableEntry*>([system_views contents]);
    IRDescriptorTableSetBuffer(&system_view_entries[0],
                               [shared_buffer gpuAddress], 64 * 1024);
    IRDescriptorTableSetBuffer(&system_view_entries[1],
                               [null_buffer gpuAddress], 256);
    IRDescriptorTableSetBuffer(&system_view_entries[2],
                               [shared_buffer gpuAddress], 64 * 1024);

    // --- Descriptor indices (b4): flat bindless index -> heap slot, from the
    // translator's own binding lists, exactly as the command processor. ---
    auto slot_for_fetch = [](uint32_t fc) -> uint32_t {
      switch (fc) {
        case 3:
          return 4;
        case 10:
          return 2;
        case 11:
          return 3;
        default:
          return 1;
      }
    };
    id<MTLBuffer> indices_vs = NewBuffer(4096, 0, @"Desc indices VS");
    id<MTLBuffer> indices_ps = NewBuffer(4096, 0, @"Desc indices PS");
    {
      auto* w = static_cast<uint32_t*>([indices_ps contents]);
      for (const auto& b : pixel_shader->GetTextureBindingsAfterTranslation()) {
        w[b.bindless_descriptor_index] = slot_for_fetch(b.fetch_constant);
      }
      for (const auto& s : pixel_shader->GetSamplerBindingsAfterTranslation()) {
        w[s.bindless_descriptor_index] = 1;
      }
    }

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
    addresses.bool_loop = [bool_loop gpuAddress];
    addresses.fetch = [fetch_buffer gpuAddress];
    addresses.descriptor_indices_vertex = [indices_vs gpuAddress];
    addresses.float_pixel = [pixel_floats gpuAddress];
    addresses.descriptor_indices_pixel = [indices_ps gpuAddress];
    std::array<uint64_t, kMetalBindingSlotCount> argument_buffer =
        BuildMetalTopLevelArgumentBuffer(addresses,
                                         /*pixel_shader_present=*/true,
                                         /*memexport_draw=*/false);
    id<MTLBuffer> ab_buffer =
        NewBuffer(sizeof(argument_buffer), 0, @"Top-level AB");
    std::memcpy([ab_buffer contents], argument_buffer.data(),
                sizeof(argument_buffer));

    // --- Encode. ---
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    for (uint32_t i = 0; i < (one_rt ? 1u : 2u); ++i) {
      pass.colorAttachments[i].texture = i ? color1 : color0;
      pass.colorAttachments[i].loadAction = MTLLoadActionClear;
      pass.colorAttachments[i].storeAction = MTLStoreActionStore;
      pass.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 0);
    }
    pass.depthAttachment.texture = depth_rt;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 1.0;
    pass.depthAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    MTLDepthStencilDescriptor* dssd = [[MTLDepthStencilDescriptor alloc] init];
    dssd.depthCompareFunction = MTLCompareFunctionAlways;
    dssd.depthWriteEnabled = depth_write ? YES : NO;
    [encoder setDepthStencilState:
                 [device newDepthStencilStateWithDescriptor:dssd]];
    auto bind = [&](id<MTLBuffer> buffer, NSUInteger index) {
      [encoder setVertexBuffer:buffer offset:0 atIndex:index];
      [encoder setFragmentBuffer:buffer offset:0 atIndex:index];
    };
    bind(ab_buffer, kIRArgumentBufferBindPoint);
    bind(view_heap, kIRDescriptorHeapBindPoint);
    bind(sampler_heap, kIRSamplerHeapBindPoint);
    const MTLRenderStages stages = MTLRenderStageVertex | MTLRenderStageFragment;
    for (id<MTLBuffer> resource :
         {ab_buffer, view_heap, sampler_heap, system_views, system_buffer,
          vertex_floats, pixel_floats, bool_loop, fetch_buffer, indices_vs,
          indices_ps, null_buffer, shared_buffer}) {
      [encoder useResource:resource usage:MTLResourceUsageRead stages:stages];
    }
    for (id<MTLTexture> texture : {null_texture, white_texture, scene_texture,
                                   normal_texture, shadow_texture}) {
      [encoder useResource:texture usage:MTLResourceUsageRead stages:stages];
    }
    IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle,
                            /*vertexStart=*/(uint64_t)0,
                            /*vertexCount=*/(uint64_t)(fixture_vs ? 6 : 3),
                            /*instanceCount=*/(uint64_t)1,
                            /*baseInstance=*/(uint64_t)0);
    [encoder endEncoding];

    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    for (uint32_t i = 0; i < (one_rt ? 1u : 2u); ++i) {
      [blit copyFromTexture:i ? color1 : color0
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(kRT, kRT, 1)
                      toBuffer:i ? readback1 : readback0
             destinationOffset:0
        destinationBytesPerRow:kRT * texel_bytes
      destinationBytesPerImage:kRT * kRT * texel_bytes];
    }
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

    result.c0.resize(kRT * kRT * 4);
    result.c1.resize(kRT * kRT * 4);
    if (rgba8) {
      const uint8_t* b0 = static_cast<const uint8_t*>([readback0 contents]);
      const uint8_t* b1 = static_cast<const uint8_t*>([readback1 contents]);
      for (uint32_t i = 0; i < kRT * kRT * 4; ++i) {
        result.c0[i] = b0[i] / 255.0f;
        result.c1[i] = b1[i] / 255.0f;
      }
    } else {
      std::memcpy(result.c0.data(), [readback0 contents], kRT * kRT * 16);
      std::memcpy(result.c1.data(), [readback1 contents], kRT * kRT * 16);
    }
    double sum[4] = {};
    for (uint32_t i = 0; i < kRT * kRT; ++i) {
      const float* p = result.c0.data() + i * 4;
      bool any = false;
      for (uint32_t c = 0; c < 4; ++c) {
        sum[c] += p[c];
        any |= p[c] != 0.0f;
      }
      if (any) {
        ++result.covered;
      }
    }
    for (uint32_t c = 0; c < 4; ++c) {
      result.mean0[c] = float(sum[c] / (kRT * kRT));
      result.center0[c] = result.c0[((kRT / 2) * kRT + kRT / 2) * 4 + c];
    }
    result.ok = true;
    return result;
  }
};

ProbeHarness& H() {
  static ProbeHarness harness;
  return harness;
}

float Luma(const float* rgb) {
  return rgb[0] + rgb[1] + rgb[2];
}

// Pixel-bank constant c[i] from the captured register file.
Float4 PsConst(uint32_t index) {
  const uint32_t* w = halo3_light::kFloatConstants + 1024 + index * 4;
  Float4 v;
  std::memcpy(&v, w, sizeof(v));
  return v;
}

// Light geometry in the microcode (r4 = interpolator o4 = position):
//   r3 = c16 - r4;  r7 = c18 - r3 = r4 + (c18 - c16);  spot = dot(norm(r7), c19)
//   cone factor = spot * c21.y + c21.w  (needs spot > 0.966 with the capture)
//   attenuation = c21.x / (c18.w + |r7|^2) + c21.z
//   diffuse     = sat(dot(norm(r7), n)) * c20.rgb * cookie * scene * ...
// So a position r4 = t * c19 - (c18 - c16) with t > 0 sits on the spot axis at
// distance t, and a normal along +c19 faces the light vector.
Float4 AxisPosition(float t) {
  const Float4 c16 = PsConst(16), c18 = PsConst(18), c19 = PsConst(19);
  return {t * c19.x - (c18.x - c16.x), t * c19.y - (c18.y - c16.y),
          t * c19.z - (c18.z - c16.z), 0.0f};
}

ProbeInputs BaseInputs() {
  ProbeInputs in;
  in.interpolators[0] = {0.5f, 0.5f, 0.0f, 0.0f};    // uv
  in.interpolators[1] = {0.0f, 0.0f, 1.0f, 0.0f};    // tangent frame rows
  in.interpolators[2] = {0.0f, 1.0f, 0.0f, 0.0f};
  in.interpolators[3] = {1.0f, 0.0f, 0.0f, 0.0f};
  in.interpolators[4] = AxisPosition(1.0f);          // position (swept)
  // Shadow projection: r9 = o5.xyz / o5.w -> texel (0.5, 0.5), receiver 0.5.
  in.interpolators[5] = {0.5f, 0.5f, kReceiverDepth, 1.0f};
  return in;
}

// Sweep the position and normal until the light term is non-zero, so the
// shadow arms compare something other than 0 == 0.
bool FindLitConfiguration(ProbeInputs* out, float* out_luma) {
  const Float4 axis = PsConst(19);
  const float normals[][3] = {{axis.x, axis.y, axis.z},
                              {-axis.x, -axis.y, -axis.z},
                              {0, 0, 1},
                              {0, 1, 0}};
  const float distances[] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, -1.0f};
  float best = 0.0f;
  bool found = false;
  for (float d : distances) {
    for (const auto& n : normals) {
      ProbeInputs in = BaseInputs();
      in.interpolators[4] = AxisPosition(d);
      std::memcpy(in.normal, n, sizeof(in.normal));
      ProbeResult r = H().Run(in);
      if (!r.ok) {
        std::printf("H3LIGHT|sweep|ERROR|%s\n", r.error.c_str());
        return false;
      }
      const float luma = Luma(r.center0);
      std::printf("H3LIGHT|sweep|d=%g n=(%g,%g,%g)|center=(%g,%g,%g,%g) "
                  "covered=%u\n",
                  d, n[0], n[1], n[2], r.center0[0], r.center0[1],
                  r.center0[2], r.center0[3], r.covered);
      if (luma > best) {
        best = luma;
        *out = in;
        found = true;
      }
    }
  }
  std::fflush(stdout);
  *out_luma = best;
  return found;
}

}  // namespace

TEST_CASE("Halo 3 light PS: probe VS and fixture translate",
          "[halo3][light][device]") {
  ProbeHarness& h = H();
  INFO(h.error);
  REQUIRE(h.ok);
  CHECK(h.pixel_shader->GetTextureBindingsAfterTranslation().size() >= 7);
  std::printf("H3LIGHT|bindings|textures=%zu samplers=%zu regs=%u\n",
              h.pixel_shader->GetTextureBindingsAfterTranslation().size(),
              h.pixel_shader->GetSamplerBindingsAfterTranslation().size(),
              h.pixel_shader->register_static_address_bound());
  std::fflush(stdout);
}

TEST_CASE("Halo 3 light PS: shadow block responds to the shadow map",
          "[halo3][light][device][shadow]") {
  ProbeHarness& h = H();
  INFO(h.error);
  REQUIRE(h.ok);

  ProbeInputs lit_inputs;
  float lit_luma = 0.0f;
  const bool found = FindLitConfiguration(&lit_inputs, &lit_luma);
  std::printf("H3LIGHT|config|found=%d luma=%g\n", found ? 1 : 0, lit_luma);
  CHECK(found);
  if (!found) {
    return;
  }

  ProbeInputs shadowed_inputs = lit_inputs;
  shadowed_inputs.shadow_depth = 0.0f;
  ProbeInputs off_inputs = lit_inputs;
  off_inputs.shadow_block = false;

  ProbeResult lit = h.Run(lit_inputs);
  ProbeResult shadowed = h.Run(shadowed_inputs);
  ProbeResult off = h.Run(off_inputs);
  REQUIRE(lit.ok);
  REQUIRE(shadowed.ok);
  REQUIRE(off.ok);

  const float lit_l = Luma(lit.center0);
  const float shadowed_l = Luma(shadowed.center0);
  const float off_l = Luma(off.center0);
  const bool lit_equals_off = lit.c0 == off.c0;
  const bool shadowed_differs = lit.c0 != shadowed.c0;
  std::printf(
      "H3LIGHT|arms|%s|lit=(%g,%g,%g) shadowed=(%g,%g,%g) off=(%g,%g,%g) "
      "lit==off:%d shadowed!=lit:%d\n",
      (lit_equals_off && shadowed_differs && shadowed_l < lit_l) ? "PASS"
                                                                 : "FAIL",
      lit.center0[0], lit.center0[1], lit.center0[2], shadowed.center0[0],
      shadowed.center0[1], shadowed.center0[2], off.center0[0],
      off.center0[1], off.center0[2], lit_equals_off ? 1 : 0,
      shadowed_differs ? 1 : 0);
  std::fflush(stdout);

  // A far occluder passes every tap: identical to skipping the block.
  CHECK(lit_equals_off);
  // A near occluder fails every tap: the light must be attenuated.
  CHECK(shadowed_differs);
  CHECK(shadowed_l < lit_l);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
