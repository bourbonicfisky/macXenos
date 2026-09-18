/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L1 on-device VALUE oracle for vertex-fetch bounds masking (upstream Canary
// 9e9d3cdd3, adopted 2026-09-18). Analytic, not comparative: the expected
// values come from the hardware contract (words at or past the size in the
// vertex fetch constant read as 0), never from another backend's render.
//
// A controlled vertex shader is assembled from guest microcode and run through
// the PRODUCTION chain (ucode -> DxbcShaderTranslator -> dxilconv -> Metal
// Shader Converter -> metallib -> MTLRenderPipelineState). Its vfetch_full
// takes its index from a float constant (c11.x), so all three vertices of a
// full-screen triangle fetch the SAME element and the fetched words arrive as
// a flat interpolator; a one-instruction pixel shader writes that interpolator
// to an RGBA32Float target and the centre pixel is read back.
//
// The shared-memory buffer holds float k+1 at word k from the fetch base on,
// and the fetch constant declares a buffer of exactly two 4-word elements.
//   index 0        -> (1, 2, 3, 4)            in bounds
//   index 1        -> (5, 6, 7, 8)            last element, in bounds
//   index 2        -> (0, 0, 0, 0)            wholly past the end (pre-port:
//                                             9, 10, 11, 12 -- the words that
//                                             follow in guest memory)
//   index 1, +2 w  -> (7, 8, 0, 0)            straddles: per-WORD masking
//                                             (pre-port: 7, 8, 9, 10)
// The pre-port values are the discriminator: they are what the census found
// live titles reading today.

#import <Metal/Metal.h>

#include <array>
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

constexpr uint32_t kRT = 16;
constexpr uint32_t kSharedBytes = 64 * 1024;
// Guest byte address of the fetched buffer inside the shared-memory buffer
// (the SRV covers the buffer from guest address 0).
constexpr uint32_t kBufferGuestBytes = 4096;
constexpr uint32_t kElementWords = 4;
constexpr uint32_t kElementCount = 2;
constexpr uint32_t kBufferSizeWords = kElementWords * kElementCount;
// The vfetch index lives in c11.x; c2..c4 drive the full-screen triangle.
constexpr uint32_t kIndexConstant = 11;

constexpr uint32_t kAluRetainPrev =
    uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26;
constexpr uint32_t kSwizzleYXZW = 0x0D;
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

// Vertex program:
//   CF 0: exec    @4  x1        mul r0.y, r0.xxxx, r0.xxxx        (y = i*i)
//   CF 1: alloc position
//   CF 2: exec    @5  x3        mul r2, r0, c2
//                               mad r1, r0.yxzw, c3, c4
//                               add oPos.xyz1, r1, r2
//   CF 3: alloc interpolators
//   CF 4: exec    @8  x3 seq=fetch@1
//                               max r3, c11, c11                 (index)
//                               vfetch_full r4, r3.x, vf0, stride/offset
//                               max o0, r4, r4
//   CF 5: execEnd @11 x5        max o1..o5, r4, r4
// c2 = (8, 2, 0, 0), c3 = (-4, -2, 0, 0), c4 = (-1, -1, 0, 0) put the three
// vertices at (-1,-1), (3,-1), (-1,3): the whole viewport (probe VS shape).
std::vector<uint32_t> BuildVertexUcode(uint32_t stride_words,
                                       int32_t offset_words) {
  std::vector<uint32_t> u(16 * 3, 0);
  auto exec = [](uint32_t address, uint32_t count, uint32_t sequence,
                 bool end) {
    return std::make_pair(
        address | (count << 12) | (sequence << 16),
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
  auto cf0 = exec(4, 1, 0, false);
  PackControlFlow(u, 0, cf0.first, cf0.second);
  auto cf1 = alloc(ucode::AllocType::kVsPosition);
  PackControlFlow(u, 1, cf1.first, cf1.second);
  auto cf2 = exec(5, 3, 0, false);
  PackControlFlow(u, 2, cf2.first, cf2.second);
  auto cf3 = alloc(ucode::AllocType::kVsInterpolators);
  PackControlFlow(u, 3, cf3.first, cf3.second);
  // Sequence: two bits per instruction, low bit = fetch. Instruction 1 (the
  // vfetch) is the fetch: 0b00'01'00.
  auto cf4 = exec(8, 3, 0b000100, false);
  PackControlFlow(u, 4, cf4.first, cf4.second);
  auto cf5 = exec(11, 5, 0, true);
  PackControlFlow(u, 5, cf5.first, cf5.second);

  const uint32_t kmul = uint32_t(ucode::AluVectorOpcode::kMul) << 24;
  const uint32_t kmad = uint32_t(ucode::AluVectorOpcode::kMad) << 24;
  const uint32_t kadd = uint32_t(ucode::AluVectorOpcode::kAdd) << 24;
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t src1_temp = 1u << 31;
  const uint32_t src2_temp = 1u << 30;

  uint32_t* a = u.data() + 4 * 3;
  // @4: mul r0.y, r0.xxxx, r0.xxxx
  a[0] = 0u | (0b0010u << 16) | kAluRetainPrev;
  a[1] = (kSwizzleXXXX << 8) | (kSwizzleXXXX << 16);
  a[2] = kmul | src1_temp | src2_temp;
  // @5: mul r2, r0, c2
  a[3] = 2u | (0b1111u << 16) | kAluRetainPrev;
  a[4] = 0;
  a[5] = (2u << 8) | (0u << 16) | kmul | src1_temp;
  // @6: mad r1, r0.yxzw, c3, c4
  a[6] = 1u | (0b1111u << 16) | kAluRetainPrev;
  a[7] = kSwizzleYXZW << 16;
  a[8] = 4u | (3u << 8) | (0u << 16) | kmad | src1_temp;
  // @7: add oPos.xyz1, r1, r2
  a[9] = 62u | (1u << 15) | (0b1111u << 16) | (0b1000u << 20) |
         kAluRetainPrev;
  a[10] = 0;
  a[11] = (2u << 8) | (1u << 16) | kadd | src1_temp | src2_temp;
  // @8: max r3, c11, c11
  a[12] = 3u | (0b1111u << 16) | kAluRetainPrev;
  a[13] = 0;
  a[14] = (kIndexConstant << 8) | (kIndexConstant << 16) | kmax;
  // @9: vfetch_full r4, r3.x, vf0 (FMT_32_32_32_32_FLOAT, not normalized)
  {
    const uint32_t kDstSwizzleIdentity =
        (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);
    a[15] = (uint32_t(ucode::FetchOpcode::kVertexFetch) & 0x1F) |
            ((3u & 0x3F) << 5) |   // src_reg r3
            ((4u & 0x3F) << 12) |  // dst_reg r4
            (1u << 19) |           // must_be_one
            (0u << 20) |           // const_index 0
            (0u << 25) |           // const_index_sel 0 -> vf0
            (0u << 30);            // src_swiz x
    a[16] = (kDstSwizzleIdentity & 0xFFF) | (1u << 13) |
            ((uint32_t(xenos::VertexFormat::k_32_32_32_32_FLOAT) & 0x3F)
             << 16);
    a[17] = (stride_words & 0xFF) | ((uint32_t(offset_words) & 0x7FFFFF) << 8);
  }
  // @10: max o0, r4, r4
  a[18] = 0u | (1u << 15) | (0b1111u << 16) | kAluRetainPrev;
  a[19] = 0;
  a[20] = (4u << 8) | (4u << 16) | kmax | src1_temp | src2_temp;
  // @11..@15: max o1..o5, r4, r4
  for (uint32_t n = 1; n < 6; ++n) {
    uint32_t* e = a + 21 + (n - 1) * 3;
    e[0] = n | (1u << 15) | (0b1111u << 16) | kAluRetainPrev;
    e[1] = 0;
    e[2] = (4u << 8) | (4u << 16) | kmax | src1_temp | src2_temp;
  }
  return u;
}

// Pixel program: oC0 = r0 (interpolator o0).
std::vector<uint32_t> BuildPixelUcode() {
  const uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kPsColors) << 9) |
                            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  const uint32_t exec_d0 = 1u | (1u << 12);
  const uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;
  std::vector<uint32_t> u(6, 0);
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = ((exec_d0 >> 16) & 0xFFFF) | ((exec_d1 & 0xFFFF) << 16);
  u[3] = 0u | (1u << 15) | (0xFu << 16) | kAluRetainPrev;
  u[4] = 0;
  u[5] = (0u << 8) | (0u << 16) |
         (uint32_t(ucode::AluVectorOpcode::kMax) << 24) | (1u << 31) |
         (1u << 30);
  return u;
}

std::vector<uint32_t> ToGuestEndian(const std::vector<uint32_t>& host) {
  std::vector<uint32_t> guest(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    guest[i] = xe::byte_swap(host[i]);
  }
  return guest;
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct FetchCase {
  uint32_t stride_words;
  int32_t offset_words;
  float index;
  uint32_t size_words;  // declared in the fetch constant
  std::array<float, 4> expected;
  std::array<float, 4> pre_port;  // what an unmasked fetch reads
  const char* name;
};

struct Harness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  std::string error;

  Harness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      error = "no Metal device";
      return;
    }
    queue = [device newCommandQueue];
    cache = std::make_unique<MetalPipelineCache>((__bridge void*)device);
    if (!queue || !cache->Initialize()) {
      error = "queue/MetalPipelineCache::Initialize failed";
    }
  }

  id<MTLBuffer> NewBuffer(NSUInteger length, NSString* label) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:length
                            options:MTLResourceStorageModeShared];
    if (buffer) {
      std::memset([buffer contents], 0, length);
      buffer.label = label;
    }
    return buffer;
  }

  // Translates a VS with the given fetch shape and the PS, builds the
  // pipeline, draws once with c11.x = index, returns the centre pixel.
  bool Run(const FetchCase& fc, std::array<float, 4>* out_center,
           std::string* out_error) {
    if (!error.empty()) {
      *out_error = error;
      return false;
    }
    std::vector<uint32_t> vs_ucode =
        ToGuestEndian(BuildVertexUcode(fc.stride_words, fc.offset_words));
    std::vector<uint32_t> ps_ucode = ToGuestEndian(BuildPixelUcode());
    MetalShader* vertex_shader = cache->LoadShader(
        xenos::ShaderType::kVertex, vs_ucode.data(), uint32_t(vs_ucode.size()));
    MetalShader* pixel_shader = cache->LoadShader(
        xenos::ShaderType::kPixel, ps_ucode.data(), uint32_t(ps_ucode.size()));
    if (!vertex_shader || !pixel_shader) {
      *out_error = "LoadShader failed";
      return false;
    }
    for (MetalShader* shader : {vertex_shader, pixel_shader}) {
      if (!shader->is_ucode_analyzed()) {
        StringBuffer disasm;
        shader->AnalyzeUcode(disasm);
        if (std::getenv("XE_VFB_DISASM")) {
          std::printf("VFB|%s|\n%s\n",
                      shader == vertex_shader ? "vs" : "ps",
                      disasm.to_string().c_str());
        }
      }
    }
    if (vertex_shader->writes_interpolators() != 0x3Fu) {
      *out_error = "VS does not export o0..o5";
      return false;
    }
    if (vertex_shader->vertex_bindings().size() != 1) {
      *out_error = "VS does not have exactly one vertex binding";
      return false;
    }

    DxbcShaderTranslator::Modification vm(
        cache->shader_translator()->GetDefaultVertexShaderModification(
            vertex_shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters),
            HostVertexShaderType::kVertex));
    vm.vertex.interpolator_mask = 0x3F;
    DxbcShaderTranslator::Modification pm(
        cache->shader_translator()->GetDefaultPixelShaderModification(
            pixel_shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters)));
    pm.pixel.interpolator_mask = 0x3F;
    pm.pixel.rt0_blend_rgb_factor_for_premult = xenos::BlendFactor::kOne;
    pm.pixel.rt0_blend_a_factor_for_premult = xenos::BlendFactor::kOne;
    auto* vertex_translation = static_cast<MetalShader::MetalTranslation*>(
        vertex_shader->GetOrCreateTranslation(vm.value));
    auto* pixel_translation = static_cast<MetalShader::MetalTranslation*>(
        pixel_shader->GetOrCreateTranslation(pm.value));
    if (!vertex_translation->has_metal_function() &&
        !cache->TranslateShader(*vertex_translation)) {
      *out_error = "VS translation failed";
      return false;
    }
    if (!pixel_translation->has_metal_function() &&
        !cache->TranslateShader(*pixel_translation)) {
      *out_error = "PS translation failed";
      return false;
    }

    MTLRenderPipelineDescriptor* desc =
        [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction =
        (__bridge id<MTLFunction>)vertex_translation->metal_function();
    desc.fragmentFunction =
        (__bridge id<MTLFunction>)pixel_translation->metal_function();
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    desc.rasterSampleCount = 1;
    NSError* pipeline_error = nil;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:desc
                                               error:&pipeline_error];
    if (!pipeline) {
      *out_error = pipeline_error
                       ? [[pipeline_error localizedDescription] UTF8String]
                       : "pipeline creation failed";
      return false;
    }

    // --- Render target + readback. ---
    MTLTextureDescriptor* cd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:kRT
                                    height:kRT
                                 mipmapped:NO];
    cd.usage = MTLTextureUsageRenderTarget;
    cd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> color0 = [device newTextureWithDescriptor:cd];
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:kRT
                                    height:kRT
                                 mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> depth_rt = [device newTextureWithDescriptor:dd];
    id<MTLBuffer> readback = NewBuffer(kRT * kRT * 16, @"Readback");
    if (!color0 || !depth_rt || !readback) {
      *out_error = "RT/readback alloc failed";
      return false;
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
        NewBuffer(sizeof(system_constants), @"System Constants");
    std::memcpy([system_buffer contents], &system_constants,
                sizeof(system_constants));

    // --- Float banks: c2..c4 (triangle), c11 (index). ---
    RegisterFile regs;
    std::memset(regs.values, 0, sizeof(regs.values));
    auto set_vs_const = [&](uint32_t index, float x, float y, float z,
                            float w) {
      uint32_t* v = &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + index * 4];
      v[0] = FloatBits(x);
      v[1] = FloatBits(y);
      v[2] = FloatBits(z);
      v[3] = FloatBits(w);
    };
    set_vs_const(2, 8.0f, 2.0f, 0.0f, 0.0f);
    set_vs_const(3, -4.0f, -2.0f, 0.0f, 0.0f);
    set_vs_const(4, -1.0f, -1.0f, 0.0f, 0.0f);
    set_vs_const(kIndexConstant, fc.index, 0.0f, 0.0f, 0.0f);
    id<MTLBuffer> vertex_floats = NewBuffer(4096, @"Vertex floats");
    id<MTLBuffer> pixel_floats = NewBuffer(4096, @"Pixel floats");
    if (!metal::PackMetalFloatConstants(
            regs, vertex_shader->constant_register_map(),
            XE_GPU_REG_SHADER_CONSTANT_000_X, [vertex_floats contents], 4096)) {
      *out_error = "PackMetalFloatConstants (vertex) failed";
      return false;
    }
    if (!metal::PackMetalFloatConstants(
            regs, pixel_shader->constant_register_map(),
            XE_GPU_REG_SHADER_CONSTANT_256_X, [pixel_floats contents], 4096)) {
      *out_error = "PackMetalFloatConstants (pixel) failed";
      return false;
    }
    id<MTLBuffer> bool_loop = NewBuffer(256, @"Bool/loop");

    // --- Fetch constants: vf0 = kElementCount elements of kElementWords at
    // kBufferGuestBytes. Vertex fetch constant k occupies words 2k, 2k+1. ---
    id<MTLBuffer> fetch_buffer = NewBuffer(48 * 16, @"Fetch constants");
    {
      xenos::xe_gpu_vertex_fetch_t vf;
      std::memset(&vf, 0, sizeof(vf));
      vf.type = xenos::FetchConstantType::kVertex;
      vf.address = kBufferGuestBytes >> 2;
      vf.endian = xenos::Endian::kNone;
      vf.size = fc.size_words;
      std::memcpy([fetch_buffer contents], &vf, sizeof(vf));
    }

    // --- Shared memory: float(k + 1) at word k from the buffer base, for
    // 64 words (well past the declared 8), so an unmasked read is visible. ---
    id<MTLBuffer> shared_buffer = NewBuffer(kSharedBytes, @"Shared memory");
    {
      float* words = reinterpret_cast<float*>(
          static_cast<uint8_t*>([shared_buffer contents]) + kBufferGuestBytes);
      for (uint32_t k = 0; k < 64; ++k) {
        words[k] = float(k + 1);
      }
    }

    // --- Descriptor heaps (unused but must exist) + system views. ---
    constexpr uint32_t kHeapSlots = 4;
    id<MTLBuffer> view_heap =
        NewBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), @"Views");
    id<MTLBuffer> sampler_heap =
        NewBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), @"Samplers");
    id<MTLBuffer> system_views =
        NewBuffer(3 * sizeof(IRDescriptorTableEntry), @"System views");
    id<MTLBuffer> null_buffer = NewBuffer(256, @"Null");
    auto* system_view_entries =
        static_cast<IRDescriptorTableEntry*>([system_views contents]);
    IRDescriptorTableSetBuffer(&system_view_entries[0],
                               [shared_buffer gpuAddress], kSharedBytes);
    IRDescriptorTableSetBuffer(&system_view_entries[1],
                               [null_buffer gpuAddress], 256);
    IRDescriptorTableSetBuffer(&system_view_entries[2],
                               [shared_buffer gpuAddress], kSharedBytes);
    id<MTLBuffer> indices_vs = NewBuffer(4096, @"Desc indices VS");
    id<MTLBuffer> indices_ps = NewBuffer(4096, @"Desc indices PS");

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
                                         /*memexport_draw=*/false,
                                         /*tessellation_draw=*/false);
    id<MTLBuffer> ab_buffer = NewBuffer(sizeof(argument_buffer), @"AB");
    std::memcpy([ab_buffer contents], argument_buffer.data(),
                sizeof(argument_buffer));

    // --- Encode: one full-screen triangle. ---
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color0;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
    pass.depthAttachment.texture = depth_rt;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 1.0;
    pass.depthAttachment.storeAction = MTLStoreActionDontCare;
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    MTLDepthStencilDescriptor* dssd = [[MTLDepthStencilDescriptor alloc] init];
    dssd.depthCompareFunction = MTLCompareFunctionAlways;
    dssd.depthWriteEnabled = NO;
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
    IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle,
                            /*vertexStart=*/(uint64_t)0,
                            /*vertexCount=*/(uint64_t)3,
                            /*instanceCount=*/(uint64_t)1,
                            /*baseInstance=*/(uint64_t)0);
    [encoder endEncoding];
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    [blit copyFromTexture:color0
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(kRT, kRT, 1)
                    toBuffer:readback
           destinationOffset:0
      destinationBytesPerRow:kRT * 16
    destinationBytesPerImage:kRT * kRT * 16];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted ||
        command_buffer.error) {
      *out_error = command_buffer.error
                       ? [[command_buffer.error localizedDescription]
                             UTF8String]
                       : "command buffer not completed";
      return false;
    }
    const float* pixels = static_cast<const float*>([readback contents]);
    const uint32_t center = ((kRT / 2) * kRT + kRT / 2) * 4;
    for (uint32_t c = 0; c < 4; ++c) {
      (*out_center)[c] = pixels[center + c];
    }
    // The triangle must have covered the target (clear is -1).
    if (pixels[0] == -1.0f) {
      *out_error = "full-screen triangle did not cover the target";
      return false;
    }
    return true;
  }
};

Harness& H() {
  static Harness harness;
  return harness;
}

const FetchCase kCases[] = {
    {4, 0, 0.0f, kBufferSizeWords, {1, 2, 3, 4}, {1, 2, 3, 4},
     "index 0: first element"},
    {4, 0, 1.0f, kBufferSizeWords, {5, 6, 7, 8}, {5, 6, 7, 8},
     "index 1: last element"},
    {4, 0, 2.0f, kBufferSizeWords, {0, 0, 0, 0}, {9, 10, 11, 12},
     "index 2: wholly past the end"},
    {4, 2, 1.0f, kBufferSizeWords, {7, 8, 0, 0}, {7, 8, 9, 10},
     "index 1 + 2 words: straddles the end (per-word mask)"},
    // Control: the SAME index 2 with the fetch constant declaring 64 words
    // reads the words that follow. This pins the zeros above to the mask,
    // not to an addressing accident or an empty buffer.
    {4, 0, 2.0f, 64, {9, 10, 11, 12}, {9, 10, 11, 12},
     "index 2 with a 64-word fetch constant: in bounds, reads 9..12"},
};

}  // namespace

TEST_CASE("Vertex fetch device oracle: in-bounds elements read their data",
          "[vertex_fetch_bounds][device]") {
  for (const FetchCase& fc : {kCases[0], kCases[1], kCases[4]}) {
    std::array<float, 4> center{};
    std::string error;
    INFO(fc.name);
    REQUIRE(H().Run(fc, &center, &error));
    INFO(error);
    for (uint32_t c = 0; c < 4; ++c) {
      INFO("component " << c);
      CHECK(center[c] == fc.expected[c]);
    }
  }
}

TEST_CASE("Vertex fetch device oracle: words past the fetch size read zero",
          "[vertex_fetch_bounds][device]") {
  for (const FetchCase& fc : {kCases[2], kCases[3]}) {
    std::array<float, 4> center{};
    std::string error;
    INFO(fc.name);
    REQUIRE(H().Run(fc, &center, &error));
    INFO(error);
    std::printf("VFB|%s|center=(%g,%g,%g,%g) expected=(%g,%g,%g,%g) "
                "pre_port=(%g,%g,%g,%g)\n",
                fc.name, center[0], center[1], center[2], center[3],
                fc.expected[0], fc.expected[1], fc.expected[2], fc.expected[3],
                fc.pre_port[0], fc.pre_port[1], fc.pre_port[2], fc.pre_port[3]);
    // The discriminator: an unmasked fetch returns the words that follow.
    bool read_pre_port = true;
    for (uint32_t c = 0; c < 4; ++c) {
      read_pre_port &= center[c] == fc.pre_port[c];
    }
    CHECK_FALSE(read_pre_port);
    for (uint32_t c = 0; c < 4; ++c) {
      INFO("component " << c);
      CHECK(center[c] == fc.expected[c]);
    }
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
