/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// port-xenios-tessellation task 5.1 — the L1 synthetic RENDER family.
//
// Where the encode smoke (task 4.5) proved "dispatch completes", these cases
// prove PIXELS: every draw goes through the PRODUCTION
// MetalPipelineCache::GetOrCreateTessellationPipelineState pipeline and the
// exact group-4 binding assembly, renders a small patch set to an offscreen
// target, and the readback is asserted.
//
//   1. All SIX mode/domain cells (discrete/continuous/adaptive x tri/quad)
//      render non-trivial coverage, with cheap extent + symmetry checks. The
//      guest domain payload exports oPos.xy01 = the two free domain
//      coordinates, so a quad patch fills the NDC [0,1]^2 quadrant and a
//      triangle patch fills its u+v<=1 half — coverage geometry is exact.
//      (The payload deliberately uses TWO temp registers: fixtures whose
//      register_static_address_bound is 1 trip
//      `assert_true(register_count() >= 2)` at dxbc_shader_translator.cc:489
//      in Checked builds — pre-existing, avoided here, pinned by the fixture
//      case.)
//   2. Factor-VALUE arms, rasterized in line fill mode so the tessellation
//      factor is pixel-observable (more segments -> more line pixels):
//      discrete/continuous read the published range max over the indexed
//      path; adaptive consumes a guest-style 32-bit factor stream, including
//      the [2,1] min>max range MEASURED LIVE in Viva Pinata (tasks.md 1.3) —
//      the arm pins what that range actually renders (HLSL
//      clamp(x,lo,hi)=min(max(x,lo),hi) collapses to hi when lo>hi, so [2,1]
//      must render EXACTLY like an honest factor-1.0 stream), plus a [6,3]
//      pair whose hi/lo outcomes are visually distinct, plus an
//      asymmetric-byte endian arm where a double swap visibly changes the
//      factor (design D5 oracle, now at the pixel level).
//   3. Two-draws-one-submission factor isolation (spec "Tess-factor buffers
//      are safe across multiple tessellated draws"): two indexed adaptive
//      draws with different factor streams in ONE submission each render
//      with their own factors (compared against single-draw references).
//   4. Fail-closed family: the depth-only/no-PS render (fallback PS engages,
//      depth output present, color never required), and the
//      tessellation+memexport rejection predicate input
//      (Shader::memexport_eM_written — exactly what the IssueDraw route keys
//      its "tess-memexport" skip on; the route itself is CP-embedded and is
//      exercised by the live gate 5.3). The no-mesh-device skip
//      ("tess-no-mesh-shaders") cannot be synthesized on hardware that HAS
//      mesh shaders — these cases REQUIRE mesh support as a precondition and
//      the skip stays a live/route concern.
//
// Probe-test style: every cell is CHECKed (not REQUIREd) and prints a
// "TESSRENDER|" matrix line so one failure still shows the whole picture.

#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/byte_order.h"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/host_tessellator.h"
#include "xenia/gpu/metal/metal_binding_layout.h"
#include "xenia/gpu/metal/metal_host_tessellation.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

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

constexpr uint32_t kRT = 128;              // Render target dimension.
constexpr uint32_t kHalf = kRT / 2;        // Quadrant dimension.
constexpr uint32_t kQuarterArea = kHalf * kHalf;

// ---------------------------------------------------------------------------
// Guest microcode fixtures (hand-assembled, verified by the disassembly case).
//
// Domain payload: oPos.xy01 = the two free domain coordinates.
//   instr@2: max r1.xy__, r0.<swiz>, r0.<swiz>   (copy via max a,a)
//   instr@3: max oPos.xy01, r1, r1               (z=const0, w=const1)
// Register use: r0 (domain inputs written by the translator prologue) and r1
// -> register_static_address_bound == 2, satisfying the translator's domain
// prologue assert (dxbc_shader_translator.cc:489).
//
// Domain input register layout (DxbcShaderTranslator::StartVertexOrDomainShader):
//   kTriangleDomain*  : r0.xyz = domain location (ZYX swizzle) — r0.xy are two
//                       independent barycentrics; identity source swizzle.
//   kQuadDomainCP     : r0.xy = UV; identity source swizzle.
//   kQuadDomainPatch  : r0.x = patch index, r0.yz = UV; source swizzle .yzzw
//                       so UV lands in position xy.

// Component-relative swizzle bytes (2 bits per component, value s so that
// component i reads ((s + i) & 3)).
constexpr uint32_t kSrcSwizzleIdentity = 0x00;  // xyzw
constexpr uint32_t kSrcSwizzleYZ = 0x05;        // yzzw (s0=1, s1=1)

std::vector<uint32_t> BuildDomainPayloadUcode(uint32_t source_swizzle) {
  std::vector<uint32_t> u(12, 0);
  // CF slot 0: exec addr=2 count=1; slot 1: alloc position; slot 2: exece
  // addr=3 count=1. 48-bit ops packed two per three dwords.
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
  // ALU dword layout: see ucode.h AluInstruction::Data.
  const uint32_t retain_prev =
      uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26;
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t src12_temp = (1u << 30) | (1u << 31);  // src1/src2 sel = temp
  // instr@2: max r1.xy__, r0.<swiz>, r0.<swiz>
  u[6] = 1u /*vector_dest=r1*/ | (0b0011u << 16) /*write mask xy*/ |
         retain_prev;
  u[7] = (source_swizzle << 8) | (source_swizzle << 16);
  u[8] = kmax | src12_temp;  // src regs 0.
  // instr@3: max oPos.xy01, r1, r1 — export; vector mask {x,y,w}, scalar mask
  // {w} (both set -> w=const1), scalar_dest_rel=1 -> unwritten z=const0.
  u[9] = 62u /*oPos*/ | (1u << 14) /*scalar_dest_rel*/ |
         (1u << 15) /*export*/ | (0b1011u << 16) | (0b1000u << 20) |
         retain_prev;
  u[10] = 0;  // identity swizzles.
  u[11] = kmax | src12_temp | (1u << 16) /*src1=r1*/ | (1u << 8) /*src2=r1*/;
  return u;
}

// Guest pixel program: oC0 = (c0.x, c0.y, 1, 1) — xy read the pixel float
// CBV (bank plumbing probe), zw are export-mask constant-1 writes (the color
// path proof that cannot be zeroed by a missing constant bank). Shape from
// metal_geometry_clip_plane_discriminator_test.mm.
std::vector<uint32_t> BuildPixelUcode() {
  const uint32_t alloc_d0 = 0;
  const uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kPsColors) << 9) |
                            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  const uint32_t exec_d0 = (1u & 0xFFF) | ((1u & 0x7) << 12);
  const uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;
  std::vector<uint32_t> u(6, 0);
  u[0] = alloc_d0;
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = ((exec_d0 >> 16) & 0xFFFF) | ((exec_d1 & 0xFFFF) << 16);
  // vector mask {x,y,z,w}, scalar mask {z,w}: z,w = const 1, x,y = vector op.
  u[3] = 0u /*oC0*/ | (1u << 15) | (0xFu << 16) | (0b1100u << 20) |
         (uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26);
  u[4] = 0;
  u[5] = (uint32_t(ucode::AluVectorOpcode::kMax) << 24);  // src sel = const c0.
  return u;
}

// Guest vertex program with a memory export (alloc kMemory + eA + eM0 writes):
// the input the IssueDraw route's "tess-memexport" fail-closed skip keys on
// (metal_command_processor.mm, memexport_eM_written()).
std::vector<uint32_t> BuildMemexportVertexUcode() {
  std::vector<uint32_t> u(12, 0);
  const uint32_t alloc_d0 = 0;
  const uint32_t alloc_d1 = (uint32_t(ucode::AllocType::kMemory) << 9) |
                            (uint32_t(ucode::ControlFlowOpcode::kAlloc) << 12);
  const uint32_t exec_d0 = 2u | (2u << 12);  // addr 2, count 2.
  const uint32_t exec_d1 = uint32_t(ucode::ControlFlowOpcode::kExecEnd) << 12;
  u[0] = alloc_d0;
  u[1] = (alloc_d1 & 0xFFFF) | ((exec_d0 & 0xFFFF) << 16);
  u[2] = ((exec_d0 >> 16) & 0xFFFF) | ((exec_d1 & 0xFFFF) << 16);
  const uint32_t retain_prev =
      uint32_t(ucode::AluScalarOpcode::kRetainPrev) << 26;
  const uint32_t kmax = uint32_t(ucode::AluVectorOpcode::kMax) << 24;
  const uint32_t src12_temp = (1u << 30) | (1u << 31);
  // instr@2: max eA, r0, r0
  u[6] = uint32_t(ucode::ExportRegister::kExportAddress) | (1u << 15) |
         (0xFu << 16) | retain_prev;
  u[7] = 0;
  u[8] = kmax | src12_temp;
  // instr@3: max eM0, r0, r0
  u[9] = uint32_t(ucode::ExportRegister::kExportData0) | (1u << 15) |
         (0xFu << 16) | retain_prev;
  u[10] = 0;
  u[11] = kmax | src12_temp;
  return u;
}

uint32_t FloatBits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// MetalPipelineCache::LoadShader takes GUEST ucode (big-endian dwords —
// MetalShader's default input contract, see the line-loop test's provenance
// note). The hand-assembled fixtures here are host-order, so they must be
// presented in guest byte order or LoadShader's swap turns them into garbage
// microcode that still analyzes/translates/builds but renders nothing.
std::vector<uint32_t> ToGuestEndian(const std::vector<uint32_t>& host_words) {
  std::vector<uint32_t> guest(host_words.size());
  for (size_t i = 0; i < host_words.size(); ++i) {
    guest[i] = xe::byte_swap(host_words[i]);
  }
  return guest;
}

// The proven nondegenerate plain-vertex fixture (clip-discriminator test):
// y = x * x from the translated vertex index, oPos.xyz1. Used ONLY by the
// geometry-emulation CONTROL arm that validates the harness environment
// through the live-proven mesh path.
const uint32_t kParabolaVertexUcode[] = {
    0x00001002, 0x00001000, 0xC2000000, 0x00001003,
    0x00002000, 0x00000000, 0xC8020000, 0x006C6C00,
    0xC1000000, 0xC88F803E, 0x00000000, 0xC2000000};

const std::vector<uint32_t>& GuestParabolaUcode() {
  static const std::vector<uint32_t> guest = [] {
    std::vector<uint32_t> host(
        kParabolaVertexUcode,
        kParabolaVertexUcode + sizeof(kParabolaVertexUcode) / sizeof(uint32_t));
    std::vector<uint32_t> out(host.size());
    for (size_t i = 0; i < host.size(); ++i) {
      out[i] = xe::byte_swap(host[i]);
    }
    return out;
  }();
  return guest;
}

// ---------------------------------------------------------------------------
// Render harness.

struct DrawSpec {
  bool indexed = false;
  bool indices_16bit = false;
  std::vector<uint32_t> words32;  // Pre-swapped (host-order) 32-bit stream.
  std::vector<uint16_t> words16;
  uint32_t count = 0;  // vertexCountPerInstance / indexCountPerInstance.
  bool has_viewport = false;
  double vp_x = 0, vp_y = 0, vp_w = 0, vp_h = 0;
};

struct RenderSpec {
  const char* label = "";
  xenos::TessellationMode mode = xenos::TessellationMode::kDiscrete;
  HostVertexShaderType host_vs_type = HostVertexShaderType::kVertex;
  xenos::PrimitiveType host_prim = xenos::PrimitiveType::kTriangleList;
  bool use_guest_ps = true;
  bool depth_only = false;  // No color RT; D32F depth, write-always DSS.
  bool wireframe = false;
  // PUBLISHED factor range values (i.e. VGT_HOS_{MIN,MAX}_TESS_LEVEL + 1, the
  // exact values the production route publishes into system constants).
  float range_min = 1.0f;
  float range_max = 8.0f;
  xenos::Endian published_endian = xenos::Endian::kNone;
  // Harness CONTROL: render through the live-proven geometry-emulation mesh
  // path instead of tessellation (same binding environment, same readback).
  bool geometry_control = false;
  // clip-bridge tasks 4.2-4.5: user clip planes on the geometry-emulation leg.
  // count 0 leaves the pre-existing behaviour untouched.
  uint32_t user_clip_plane_count = 0;
  float clip_plane0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // Harness CONTROL 2: plain vertex+fragment pipeline (no mesh stages at all)
  // over the same AB/constants — splits "mesh dispatch" from "content".
  bool plain_control = false;
  // --- Nondeterminism-campaign dimensions (defaults preserve the original
  // behavior byte-for-byte; see the [nondeterm] diagnostic case). ---
  uint32_t sample_count = 1;   // rasterSampleCount; >1 = MSAA targets +
                               // per-sample compute readback.
  bool depth_stencil8 = false;  // production Depth32Float_Stencil8 instead of
                                // the harness default Depth32Float.
  bool depth_compare_lequal = false;  // LessEqual instead of Always.
  // Dispatch-parameter experiments (adaptive INDEXED draws only). Any nonzero
  // value routes the draw through a LOCAL replica of
  // IRRuntimeDrawIndexedPatchesTessellationEmulation (the vendored header's
  // body is IR_INLINE and its helpers are public) so single dispatch knobs
  // can vary without touching the vendored code.
  uint32_t tg_mem_length = 0;       // 0 = runtime default (15360 hardcoded).
  uint32_t hs_patches_override = 0; // 0 = pipeline-reflected value.
  uint32_t index_pad_words = 0;     // prepend N filler words, startIndex = N.
  std::vector<DrawSpec> draws;
};

struct RenderResult {
  bool ok = false;
  std::string error;
  std::vector<uint8_t> rgba;   // kRT*kRT*4 (color arms).
  std::vector<float> depth;    // kRT*kRT (always read back).
  uint32_t covered = 0;        // Color coverage (depth coverage if depth-only).
  uint32_t depth_covered = 0;  // Rasterization proof independent of the PS.
  // The pipeline mirror the draw consumed (diagnostic identity trace).
  uint32_t cfg_vs_output = 0;
  uint32_t cfg_gs_max_input = 0;
  uint32_t cfg_hs_patches = 0;
  uint32_t cfg_hs_cp = 0;
  uint32_t cfg_hs_threads = 0;
  float cfg_max_factor = 0.0f;

  bool CoveredAt(uint32_t x, uint32_t y) const {
    if (!rgba.empty()) {
      const uint8_t* p = rgba.data() + (y * kRT + x) * 4;
      return p[0] | p[1] | p[2] | p[3];
    }
    if (!depth.empty()) {
      return depth[y * kRT + x] < 0.999f;
    }
    return false;
  }
};

// LOCAL replica of IRRuntimeDrawIndexedPatchesTessellationEmulation (the
// vendored header's body is IR_INLINE and every helper it calls is public in
// the same header) with the one hardcoded dispatch parameter exposed:
// objectThreadgroupMemoryLength (vendored: 15360). TEST-ONLY per the
// nondeterminism-campaign rules — the vendored header is not modified, and
// production keeps calling the vendored helper.
void LocalDrawIndexedPatchesTessellationEmulation(
    id<MTLRenderCommandEncoder> enc, IRRuntimePrimitiveType primitive_topology,
    MTLIndexType index_type, id<MTLBuffer> index_buffer,
    IRRuntimeTessellationPipelineConfig cfg, uint32_t instance_count,
    uint32_t index_count_per_instance, uint32_t base_instance,
    int32_t base_vertex, uint32_t start_index, uint32_t threadgroup_mem) {
  // The vendored Calculate* helpers are `static` under
  // IR_PRIVATE_IMPLEMENTATION (visible only to ir_runtime_impl.mm), so their
  // pure arithmetic is mirrored here verbatim from the header bodies
  // (IRTessellatorThreadgroupVertexOverlap :1072,
  // IRRuntimeCalculateDrawInfoForGSTSEmulation :1084,
  // IRRuntimeCalculateObjectTgCountForTessellationAndGeometryEmulation :905,
  // IRRuntimeCalculateThreadgroupSizeForTessellationAndGeometry :935 —
  // patchlists are not strips, so the strip-overlap branch never engages).
  uint16_t vertex_overlap =
      cfg.outputPrimitiveType == IRRuntimeTessellatorOutputPoint
          ? 0
          : (cfg.outputPrimitiveType == IRRuntimeTessellatorOutputLine ? 1
                                                                       : 2);
  IRRuntimeDrawInfo draw_info = {};
  draw_info.meshThreadgroupPrimitiveStride =
      (uint16_t)cfg.gsMaxInputPrimitivesPerMeshThreadgroup - vertex_overlap;
  draw_info.objectThreadgroupVertexStride =
      (uint16_t)(cfg.hsMaxPatchesPerObjectThreadgroup *
                 cfg.hsInputControlPointCount);
  draw_info.patchesPerObjectThreadgroup =
      (uint16_t)cfg.hsMaxPatchesPerObjectThreadgroup;
  draw_info.threadsPerPatch = (uint8_t)cfg.hsMaxObjectThreadsPerThreadgroup;
  draw_info.inputControlPointsPerPatch =
      (uint16_t)cfg.hsInputControlPointCount;
  draw_info.gsInstanceCount = (uint16_t)cfg.gsInstanceCount;
  draw_info.maxInputPrimitivesPerMeshThreadgroup =
      (uint16_t)cfg.gsMaxInputPrimitivesPerMeshThreadgroup;
  draw_info.indexType = (uint8_t)index_type + 1;
  draw_info.primitiveTopology = (uint8_t)primitive_topology;
  MTLSize object_threadgroup_count = MTLSizeMake(
      (index_count_per_instance + draw_info.objectThreadgroupVertexStride -
       1) /
          draw_info.objectThreadgroupVertexStride,
      instance_count, 1);
  uint32_t object_threadgroup_size = cfg.hsMaxPatchesPerObjectThreadgroup *
                                     cfg.hsMaxObjectThreadsPerThreadgroup;
  uint32_t mesh_threadgroup_size = cfg.gsMaxInputPrimitivesPerMeshThreadgroup;
  IRRuntimeDrawParams draw_params;
  draw_params.drawIndexed = (IRRuntimeDrawIndexedArgument){
      .indexCountPerInstance = index_count_per_instance,
      .instanceCount = instance_count,
      .startIndexLocation = start_index,
      .baseVertexLocation = base_vertex,
      .startInstanceLocation = base_instance};
  draw_info.indexBuffer = index_buffer.gpuAddress;
  [enc setObjectBytes:&draw_info
               length:sizeof(IRRuntimeDrawInfo)
              atIndex:kIRArgumentBufferUniformsBindPoint];
  [enc setMeshBytes:&draw_info
             length:sizeof(IRRuntimeDrawInfo)
            atIndex:kIRArgumentBufferUniformsBindPoint];
  [enc setObjectBytes:&draw_params
               length:sizeof(IRRuntimeDrawParams)
              atIndex:kIRArgumentBufferDrawArgumentsBindPoint];
  [enc setMeshBytes:&draw_params
             length:sizeof(IRRuntimeDrawParams)
            atIndex:kIRArgumentBufferDrawArgumentsBindPoint];
  [enc setObjectThreadgroupMemoryLength:threadgroup_mem atIndex:0];
  [enc drawMeshThreadgroups:object_threadgroup_count
      threadsPerObjectThreadgroup:MTLSizeMake(object_threadgroup_size, 1, 1)
        threadsPerMeshThreadgroup:MTLSizeMake(mesh_threadgroup_size, 1, 1)];
}

// Per-sample readback kernels (MSAA targets cannot be blitted to buffers;
// combined depth-stencil depth planes read uniformly through depth2d).
constexpr const char* kExpandKernelsSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void expand_depth_1x(depth2d<float, access::read> t [[texture(0)]],
                            device float* out [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
  uint w = t.get_width();
  if (gid.x >= w || gid.y >= t.get_height()) return;
  out[gid.y * w + gid.x] = t.read(gid);
}
kernel void expand_depth_ms(depth2d_ms<float, access::read> t [[texture(0)]],
                            device float* out [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
  uint w = t.get_width();
  uint h = t.get_height();
  if (gid.x >= w || gid.y >= h) return;
  uint n = t.get_num_samples();
  for (uint s = 0; s < n; ++s) {
    out[(s * h + gid.y) * w + gid.x] = t.read(gid, s);
  }
}
kernel void expand_color_ms(texture2d_ms<float, access::read> t [[texture(0)]],
                            device uchar* out [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
  uint w = t.get_width();
  uint h = t.get_height();
  if (gid.x >= w || gid.y >= h) return;
  uint n = t.get_num_samples();
  for (uint s = 0; s < n; ++s) {
    float4 c = saturate(t.read(gid, s));
    uint base = ((s * h + gid.y) * w + gid.x) * 4;
    out[base + 0] = uchar(c.r * 255.0f + 0.5f);
    out[base + 1] = uchar(c.g * 255.0f + 0.5f);
    out[base + 2] = uchar(c.b * 255.0f + 0.5f);
    out[base + 3] = uchar(c.a * 255.0f + 0.5f);
  }
}
)";

struct RenderHarness {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  bool ok = false;
  std::string error;
  // implement-metal-host-tessellation task 4.1: render through the HOST
  // tessellation route (reference tessellator -> XeTessellationVertex
  // records + triangle indices -> the production plain vertex+fragment
  // pipeline under the domain_as_vertex translation, records exposed through
  // the per-draw two-entry SRV space-0 table) instead of the MSC
  // object/mesh emulation. Every assertion stays the same on both routes.
  bool host_route = false;
  // Per-sample readback pipelines (lazy; nondeterminism campaign only).
  id<MTLComputePipelineState> expand_depth_1x = nil;
  id<MTLComputePipelineState> expand_depth_ms = nil;
  id<MTLComputePipelineState> expand_color_ms = nil;

  bool EnsureExpandPipelines(std::string* out_error) {
    if (expand_depth_1x) {
      return true;
    }
    NSError* ns_error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:
                                           kExpandKernelsSource]
                     options:nil
                       error:&ns_error];
    if (!library) {
      *out_error = ns_error ? ns_error.localizedDescription.UTF8String
                            : "expand library failed";
      return false;
    }
    auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> fn = [library newFunctionWithName:name];
      if (!fn) return nil;
      NSError* pe = nil;
      return [device newComputePipelineStateWithFunction:fn error:&pe];
    };
    expand_depth_1x = make(@"expand_depth_1x");
    expand_depth_ms = make(@"expand_depth_ms");
    expand_color_ms = make(@"expand_color_ms");
    if (!expand_depth_1x || !expand_depth_ms || !expand_color_ms) {
      *out_error = "expand pipeline creation failed";
      return false;
    }
    return true;
  }

  RenderHarness() {
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

  // Domain-payload guest VS translation with the DOMAIN host-VS-type
  // modification (never TranslateShader'd — the tessellation pipeline path
  // compiles Stage::kDomain itself).
  MetalShader::MetalTranslation* MakeDomainTranslation(
      HostVertexShaderType host_vertex_shader_type,
      bool domain_as_vertex = false) {
    const bool quad_patch =
        host_vertex_shader_type == HostVertexShaderType::kQuadDomainPatchIndexed;
    static const std::vector<uint32_t> identity_ucode =
        ToGuestEndian(BuildDomainPayloadUcode(kSrcSwizzleIdentity));
    static const std::vector<uint32_t> yz_ucode =
        ToGuestEndian(BuildDomainPayloadUcode(kSrcSwizzleYZ));
    const std::vector<uint32_t>& ucode = quad_patch ? yz_ucode : identity_ucode;
    MetalShader* shader = cache->LoadShader(xenos::ShaderType::kVertex,
                                            ucode.data(),
                                            uint32_t(ucode.size()));
    if (!shader->is_ucode_analyzed()) {
      StringBuffer disasm;
      shader->AnalyzeUcode(disasm);
    }
    DxbcShaderTranslator::Modification modification(
        cache->shader_translator()->GetDefaultVertexShaderModification(
            shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters),
            host_vertex_shader_type));
    modification.vertex.domain_as_vertex = domain_as_vertex ? 1 : 0;
    return static_cast<MetalShader::MetalTranslation*>(
        shader->GetOrCreateTranslation(modification.value));
  }

  MetalShader::MetalTranslation* MakeGuestPixelShader() {
    static const std::vector<uint32_t> ucode = ToGuestEndian(BuildPixelUcode());
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
    // No interpolators in these fixtures, and CRITICALLY the RT0 premult
    // factors must be kOne: the zero-initialized default is kZero, and the
    // translator's RTV epilogue PRE-MULTIPLIES the color by them — a
    // modification left at the default renders (0,0,0,0) from any PS
    // (production sets them from real blend state,
    // metal_command_processor.mm:3336).
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

  id<MTLBuffer> NewZeroBuffer(NSUInteger length, NSString* label) {
    id<MTLBuffer> buffer = [device newBufferWithLength:length
                                               options:MTLResourceStorageModeShared];
    if (buffer) {
      std::memset([buffer contents], 0, length);
      buffer.label = label;
    }
    return buffer;
  }

  RenderResult Render(const RenderSpec& spec) {
    RenderResult result;
    if (spec.draws.empty()) {
      result.error = "no draws";
      return result;
    }

    // --- Pipeline via the production cache. ---
    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint32_t blends[xenos::kMaxColorRenderTargets];
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      colors[i] = uint64_t(MTLPixelFormatInvalid);
      blends[i] = 0x00010001;  // src*One + dst*Zero, Add — no blending.
    }
    uint64_t depth_format;
    // D32F (no stencil) everywhere so depth can be blitted back on every arm
    // — depth coverage is the rasterization proof independent of the fragment
    // stage's color output. The nondeterminism campaign can select the
    // production D32F_S8 instead (depth then reads back via the expand
    // kernel).
    depth_format = uint64_t(spec.depth_stencil8
                                ? MTLPixelFormatDepth32Float_Stencil8
                                : MTLPixelFormatDepth32Float);
    // Per-sample compute readback whenever blit cannot express the readback.
    const bool kernel_readback = spec.sample_count > 1 || spec.depth_stencil8;
    if (kernel_readback && !EnsureExpandPipelines(&result.error)) {
      return result;
    }
    if (!spec.depth_only) {
      colors[0] = uint64_t(MTLPixelFormatRGBA8Unorm);
    }
    MetalShader::MetalTranslation* pixel = nullptr;
    if (spec.use_guest_ps) {
      pixel = MakeGuestPixelShader();
      if (!pixel) {
        result.error = "guest PS translation failed";
        return result;
      }
    }
    const MetalPipelineCache::TessellationPipeline* tess = nullptr;
    void* control_pipeline = nullptr;
    void* host_pipeline = nullptr;
    id<MTLRenderPipelineState> plain_pipeline = nil;
    uint32_t control_gs_vertex_size = 0;
    uint32_t control_gs_max_input = 0;
    if (spec.plain_control) {
      MetalShader* vertex_shader = cache->LoadShader(
          xenos::ShaderType::kVertex, GuestParabolaUcode().data(),
          uint32_t(GuestParabolaUcode().size()));
      if (!vertex_shader->is_ucode_analyzed()) {
        StringBuffer disasm;
        vertex_shader->AnalyzeUcode(disasm);
      }
      DxbcShaderTranslator::Modification vm(
          cache->shader_translator()->GetDefaultVertexShaderModification(
              vertex_shader->GetDynamicAddressableRegisterCount(
                  xenos::kMaxShaderTempRegisters),
              HostVertexShaderType::kVertex));
      vm.vertex.interpolator_mask = 0;
      // 4.8 boundary control: the ORDINARY pipeline takes the same clip planes,
      // so the relabel gate can be shown not to touch this path.
      vm.vertex.user_clip_plane_count = spec.user_clip_plane_count;
      vm.vertex.user_clip_plane_cull = 0;
      auto* vertex_translation = static_cast<MetalShader::MetalTranslation*>(
          vertex_shader->GetOrCreateTranslation(vm.value));
      if (!vertex_translation->has_metal_function() &&
          !cache->TranslateShader(*vertex_translation)) {
        result.error = "plain control VS translation failed";
        return result;
      }
      MTLRenderPipelineDescriptor* desc =
          [[MTLRenderPipelineDescriptor alloc] init];
      desc.vertexFunction =
          (__bridge id<MTLFunction>)vertex_translation->metal_function();
      desc.fragmentFunction =
          (__bridge id<MTLFunction>)pixel->metal_function();
      desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
      desc.depthAttachmentPixelFormat = (MTLPixelFormat)depth_format;
      desc.rasterSampleCount = spec.sample_count;
      NSError* plain_error = nil;
      plain_pipeline =
          [device newRenderPipelineStateWithDescriptor:desc
                                                 error:&plain_error];
      if (!plain_pipeline) {
        result.error =
            plain_error
                ? [[plain_error localizedDescription] UTF8String]
                : "plain pipeline creation failed";
        return result;
      }
    } else if (spec.geometry_control) {
      // Live-proven geometry-emulation leg over the identical environment.
      MetalShader* vertex_shader = cache->LoadShader(
          xenos::ShaderType::kVertex, GuestParabolaUcode().data(),
          uint32_t(GuestParabolaUcode().size()));
      if (!vertex_shader->is_ucode_analyzed()) {
        StringBuffer disasm;
        vertex_shader->AnalyzeUcode(disasm);
      }
      DxbcShaderTranslator::Modification vm(
          cache->shader_translator()->GetDefaultVertexShaderModification(
              vertex_shader->GetDynamicAddressableRegisterCount(
                  xenos::kMaxShaderTempRegisters),
              HostVertexShaderType::kVertex));
      DxbcShaderTranslator::Modification pm(
          cache->shader_translator()->GetDefaultPixelShaderModification(0));
      vm.vertex.interpolator_mask = 0;
      vm.vertex.user_clip_plane_count = spec.user_clip_plane_count;
      vm.vertex.user_clip_plane_cull = 0;
      pm.pixel.interpolator_mask = 0;
      // Must match MakeGuestPixelShader's translation modification.
      pm.pixel.rt0_blend_rgb_factor_for_premult = xenos::BlendFactor::kOne;
      pm.pixel.rt0_blend_a_factor_for_premult = xenos::BlendFactor::kOne;
      auto* vertex_translation = static_cast<MetalShader::MetalTranslation*>(
          vertex_shader->GetOrCreateTranslation(vm.value));
      if (!vertex_translation->has_metal_function() &&
          !cache->TranslateShader(*vertex_translation)) {
        result.error = "control VS translation failed";
        return result;
      }
      control_pipeline = cache->GetOrCreateGeometryPipelineState(
          vertex_translation, pixel, PipelineGeometryShader::kRectangleList,
          vm, pm, colors, depth_format,
          MetalPipelineRasterKey{spec.sample_count}, blends,
          /*color_write_mask=*/0xF, &control_gs_vertex_size,
          &control_gs_max_input);
      if (!control_pipeline) {
        result.error = "GetOrCreateGeometryPipelineState returned null";
        return result;
      }
    } else if (host_route) {
      // Host route: the guest domain payload as a VERTEX shader through the
      // production vertex translation path and the production plain
      // vertex+fragment pipeline family (no tessellation pipeline kind).
      MetalShader::MetalTranslation* domain_vs =
          MakeDomainTranslation(spec.host_vs_type, /*domain_as_vertex=*/true);
      if (!domain_vs) {
        result.error = "domain_as_vertex translation failed";
        return result;
      }
      if (!domain_vs->has_metal_function() &&
          !cache->TranslateShader(*domain_vs)) {
        result.error = "domain_as_vertex TranslateShader failed";
        return result;
      }
      host_pipeline = cache->GetOrCreateRenderPipelineState(
          domain_vs, pixel, colors, depth_format,
          MetalPipelineRasterKey{spec.sample_count}, blends,
          /*color_write_mask=*/0xF);
      if (!host_pipeline) {
        result.error = "GetOrCreateRenderPipelineState (host route) null";
        return result;
      }
    } else {
      MetalShader::MetalTranslation* domain =
          MakeDomainTranslation(spec.host_vs_type);
      if (!domain) {
        result.error = "domain translation failed";
        return result;
      }
      tess = cache->GetOrCreateTessellationPipelineState(
          domain, pixel, spec.host_vs_type, spec.mode, spec.host_prim, colors,
          depth_format, MetalPipelineRasterKey{spec.sample_count}, blends,
          /*color_write_mask=*/0xF);
      if (!tess || !tess->pipeline) {
        result.error = "GetOrCreateTessellationPipelineState returned null";
        return result;
      }
      if (tess->hs_input_control_point_count != 3 &&
          tess->hs_input_control_point_count != 4) {
        result.error = "unexpected hs_input_control_point_count";
        return result;
      }
      result.cfg_vs_output = tess->vs_output_size_in_bytes;
      result.cfg_gs_max_input =
          tess->gs_max_input_primitives_per_mesh_threadgroup;
      result.cfg_hs_patches = tess->hs_max_patches_per_object_threadgroup;
      result.cfg_hs_cp = tess->hs_input_control_point_count;
      result.cfg_hs_threads = tess->hs_max_object_threads_per_threadgroup;
      result.cfg_max_factor = tess->hs_max_tessellation_factor;
    }

    // --- Tessellator tables (the CP's allocate-once + load). ---
    id<MTLBuffer> tessellator_tables =
        [device newBufferWithLength:IRRuntimeTessellatorTablesSize()
                            options:MTLResourceStorageModeShared];
    if (!tessellator_tables) {
      result.error = "tables alloc failed";
      return result;
    }
    IRRuntimeLoadTessellatorTables(tessellator_tables);

    // --- Render targets + readback buffers. ---
    id<MTLTexture> color_rt = nil;
    id<MTLTexture> depth_rt = nil;
    if (!spec.depth_only) {
      MTLTextureDescriptor* cd = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                       width:kRT
                                      height:kRT
                                   mipmapped:NO];
      cd.usage = MTLTextureUsageRenderTarget;
      cd.storageMode = MTLStorageModePrivate;
      if (spec.sample_count > 1) {
        cd.textureType = MTLTextureType2DMultisample;
        cd.sampleCount = spec.sample_count;
        cd.usage |= MTLTextureUsageShaderRead;
      }
      color_rt = [device newTextureWithDescriptor:cd];
    }
    {
      MTLTextureDescriptor* dd = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:(MTLPixelFormat)depth_format
                                       width:kRT
                                      height:kRT
                                   mipmapped:NO];
      dd.usage = MTLTextureUsageRenderTarget;
      dd.storageMode = MTLStorageModePrivate;
      if (spec.sample_count > 1) {
        dd.textureType = MTLTextureType2DMultisample;
        dd.sampleCount = spec.sample_count;
      }
      if (kernel_readback) {
        dd.usage |= MTLTextureUsageShaderRead;
      }
      depth_rt = [device newTextureWithDescriptor:dd];
    }
    const uint32_t sample_planes = spec.sample_count;
    id<MTLBuffer> readback =
        NewZeroBuffer(kRT * kRT * 4 * sample_planes, @"Readback");
    id<MTLBuffer> depth_readback =
        NewZeroBuffer(kRT * kRT * 4 * sample_planes, @"Depth readback");
    if ((!spec.depth_only && !color_rt) || !depth_rt || !readback ||
        !depth_readback) {
      result.error = "RT/readback alloc failed";
      return result;
    }

    // --- System constants: exactly what the production route publishes for a
    // tessellated draw (factor range = VGT_HOS+1 values, endian per the D5
    // decision), plus the identity NDC transform and unit exponent bias so
    // the payload's positions/colors survive the translator epilogues. ---
    DxbcShaderTranslator::SystemConstants system_constants;
    std::memset(&system_constants, 0, sizeof(system_constants));
    // WNotReciprocal: the payload exports real W. The three alpha-test pass
    // bits = "alpha test disabled" (all comparisons pass) — with them zeroed
    // the translated guest PS discards EVERY fragment (pass-function never).
    system_constants.flags =
        DxbcShaderTranslator::kSysFlag_WNotReciprocal |
        DxbcShaderTranslator::kSysFlag_AlphaPassIfLess |
        DxbcShaderTranslator::kSysFlag_AlphaPassIfEqual |
        DxbcShaderTranslator::kSysFlag_AlphaPassIfGreater;
    system_constants.tessellation_factor_range_min = spec.range_min;
    system_constants.tessellation_factor_range_max = spec.range_max;
    system_constants.vertex_index_endian = spec.published_endian;
    system_constants.vertex_index_offset = 0;
    system_constants.vertex_index_min = 0;
    system_constants.vertex_index_max = 0xFFFFFFu;
    // Real production layout: user_clip_planes[6][4], pinned at offset 32.
    for (uint32_t i = 0; i < 4; ++i) {
      system_constants.user_clip_planes[0][i] = spec.clip_plane0[i];
    }
    system_constants.ndc_scale[0] = 1.0f;
    system_constants.ndc_scale[1] = 1.0f;
    system_constants.ndc_scale[2] = 1.0f;
    for (uint32_t i = 0; i < 4; ++i) {
      system_constants.color_exp_bias[i] = 1.0f;
    }
    id<MTLBuffer> system_buffer =
        NewZeroBuffer(sizeof(system_constants), @"System Constants");
    if (!system_buffer) {
      result.error = "system constants alloc failed";
      return result;
    }
    std::memcpy([system_buffer contents], &system_constants,
                sizeof(system_constants));

    // CBV banks. The domain payload reads no float constants; the guest PS
    // reads c0 -> pixel float bank slot 0 = the fixture color (1,1,0,1).
    id<MTLBuffer> cbv_scratch = NewZeroBuffer(4096, @"CBV scratch");
    id<MTLBuffer> pixel_floats = NewZeroBuffer(4096, @"Pixel floats");
    id<MTLBuffer> null_buffer = NewZeroBuffer(256, @"Null");
    if (!cbv_scratch || !pixel_floats || !null_buffer) {
      result.error = "CBV alloc failed";
      return result;
    }
    {
      float* c0 = static_cast<float*>([pixel_floats contents]);
      c0[0] = 1.0f;
      c0[1] = 1.0f;
      c0[2] = 0.0f;
      c0[3] = 1.0f;
    }

    // Descriptor heaps: every slot dereferenceable (null texture/sampler).
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
        NewZeroBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), @"Views");
    id<MTLBuffer> sampler_heap =
        NewZeroBuffer(kHeapSlots * sizeof(IRDescriptorTableEntry), @"Samplers");
    id<MTLBuffer> system_views =
        NewZeroBuffer(3 * sizeof(IRDescriptorTableEntry), @"System views");
    id<MTLBuffer> guest_scratch = NewZeroBuffer(64 * 1024, @"Guest scratch");
    if (!null_texture || !null_sampler || !view_heap || !sampler_heap ||
        !system_views || !guest_scratch) {
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
    IRDescriptorTableSetBuffer(&system_view_entries[0],
                               [guest_scratch gpuAddress], 64 * 1024);
    IRDescriptorTableSetBuffer(&system_view_entries[1],
                               [null_buffer gpuAddress], 256);
    IRDescriptorTableSetBuffer(&system_view_entries[2],
                               [guest_scratch gpuAddress], 64 * 1024);

    // --- Host route: expand every draw with the PURE production helpers
    // (metal_host_tessellation.h), exactly as IssueDraw does, into one
    // records buffer + one absolute-index buffer, and a two-entry SRV space-0
    // table ([0] = the "shared memory" scratch, [1] = the records). ---
    id<MTLBuffer> host_records = nil;
    id<MTLBuffer> host_indices = nil;
    id<MTLBuffer> host_table = nil;
    struct HostDrawRange {
      uint32_t index_offset_bytes;
      uint32_t index_count;
    };
    std::vector<HostDrawRange> host_ranges;
    if (host_route) {
      metal::MetalHostTessellationShape shape;
      if (!metal::DescribeMetalHostTessellationShape(spec.host_vs_type,
                                                     spec.mode, &shape)) {
        result.error = "host route: unsupported shape";
        return result;
      }
      HostTessellator tessellator;
      std::vector<metal::XeTessellationVertex> records;
      std::vector<uint32_t> indices;
      for (const DrawSpec& draw : spec.draws) {
        metal::MetalHostTessellationDrawInput in{};
        in.shape = shape;
        in.mode = spec.mode;
        in.factor_range_min = spec.range_min;
        in.factor_range_max = spec.range_max;
        in.index_offset = 0;
        in.min_index = 0;
        in.max_index = 0xFFFFFFu;
        in.stream = draw.indexed
                        ? (draw.indices_16bit
                               ? static_cast<const void*>(draw.words16.data())
                               : static_cast<const void*>(draw.words32.data()))
                        : nullptr;
        in.stream_32bit = !draw.indices_16bit;
        in.stream_word_count = draw.count;
        metal::MetalHostTessellationCounts counts =
            metal::CountMetalHostTessellation(in, tessellator);
        if (counts.over_budget || !counts.vertex_count) {
          result.error = "host route: expansion empty or over budget";
          return result;
        }
        size_t record_base = records.size();
        size_t index_base = indices.size();
        records.resize(record_base + counts.vertex_count);
        indices.resize(index_base + counts.index_count);
        metal::WriteMetalHostTessellation(in, tessellator,
                                          records.data() + record_base,
                                          indices.data() + index_base);
        for (size_t i = index_base; i < indices.size(); ++i) {
          indices[i] += uint32_t(record_base);
        }
        host_ranges.push_back({uint32_t(index_base * sizeof(uint32_t)),
                               counts.index_count});
      }
      host_records = NewZeroBuffer(
          records.size() * sizeof(metal::XeTessellationVertex),
          @"Tess records");
      host_indices =
          NewZeroBuffer(indices.size() * sizeof(uint32_t), @"Tess indices");
      host_table =
          NewZeroBuffer(2 * sizeof(IRDescriptorTableEntry), @"Tess SRV table");
      if (!host_records || !host_indices || !host_table) {
        result.error = "host route: buffer alloc failed";
        return result;
      }
      std::memcpy([host_records contents], records.data(),
                  records.size() * sizeof(metal::XeTessellationVertex));
      std::memcpy([host_indices contents], indices.data(),
                  indices.size() * sizeof(uint32_t));
      auto* table_entries =
          static_cast<IRDescriptorTableEntry*>([host_table contents]);
      IRDescriptorTableSetBuffer(&table_entries[0],
                                 [guest_scratch gpuAddress], 64 * 1024);
      IRDescriptorTableSetBuffer(
          &table_entries[1], [host_records gpuAddress],
          records.size() * sizeof(metal::XeTessellationVertex));
    }

    // --- Top-level AB from the PURE production builder. ---
    MetalBindingAddresses addresses;
    addresses.null_buffer = [null_buffer gpuAddress];
    addresses.view_heap = [view_heap gpuAddress];
    addresses.sampler_heap = [sampler_heap gpuAddress];
    addresses.shared_memory_srv =
        host_route ? [host_table gpuAddress] : [system_views gpuAddress];
    addresses.null_uav =
        [system_views gpuAddress] + sizeof(IRDescriptorTableEntry);
    addresses.shared_memory_uav =
        [system_views gpuAddress] + 2 * sizeof(IRDescriptorTableEntry);
    addresses.system_constants = [system_buffer gpuAddress];
    addresses.float_vertex = [cbv_scratch gpuAddress];
    addresses.bool_loop = [cbv_scratch gpuAddress];
    addresses.fetch = [cbv_scratch gpuAddress];
    addresses.descriptor_indices_vertex = [cbv_scratch gpuAddress];
    addresses.float_pixel = [pixel_floats gpuAddress];
    addresses.descriptor_indices_pixel = [cbv_scratch gpuAddress];
    std::array<uint64_t, kMetalBindingSlotCount> argument_buffer =
        BuildMetalTopLevelArgumentBuffer(addresses,
                                         /*pixel_shader_present=*/
                                         spec.use_guest_ps,
                                         /*memexport_draw=*/false,
                                         /*tessellation_draw=*/
                                         !spec.geometry_control && !host_route);
    id<MTLBuffer> ab_buffer =
        NewZeroBuffer(sizeof(argument_buffer), @"Top-level AB");
    if (!ab_buffer) {
      result.error = "AB alloc failed";
      return result;
    }
    std::memcpy([ab_buffer contents], argument_buffer.data(),
                sizeof(argument_buffer));

    // --- Per-draw index/factor buffers (pre-swapped host order, kNone
    // convention unless the spec publishes a different endian to model the
    // double-swap failure). ---
    std::vector<id<MTLBuffer>> index_buffers;
    for (const DrawSpec& draw : spec.draws) {
      id<MTLBuffer> buffer = nil;
      if (draw.indexed) {
        if (draw.indices_16bit) {
          buffer = NewZeroBuffer(draw.words16.size() * sizeof(uint16_t),
                                 @"Indices16");
          if (buffer) {
            std::memcpy([buffer contents], draw.words16.data(),
                        draw.words16.size() * sizeof(uint16_t));
          }
        } else {
          uint32_t pad = spec.index_pad_words;
          buffer = NewZeroBuffer(
              (pad + draw.words32.size()) * sizeof(uint32_t), @"Indices32");
          if (buffer) {
            uint32_t* words = static_cast<uint32_t*>([buffer contents]);
            for (uint32_t p = 0; p < pad; ++p) {
              words[p] = 0x7F7F7F7Fu;  // Large-finite filler before the data.
            }
            std::memcpy(words + pad, draw.words32.data(),
                        draw.words32.size() * sizeof(uint32_t));
          }
        }
        if (!buffer) {
          result.error = "index buffer alloc failed";
          return result;
        }
      }
      index_buffers.push_back(buffer);
    }

    // --- Encode. ---
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (color_rt) {
      pass.colorAttachments[0].texture = color_rt;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    }
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
    if (spec.plain_control) {
      [encoder setRenderPipelineState:plain_pipeline];
    } else if (host_route) {
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                          host_pipeline];
    } else {
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)(
                                          spec.geometry_control
                                              ? control_pipeline
                                              : tess->pipeline)];
    }
    if (spec.wireframe) {
      [encoder setTriangleFillMode:MTLTriangleFillModeLines];
    }
    // Depth always written (compare Always): depth coverage doubles as the
    // rasterization oracle on every arm.
    MTLDepthStencilDescriptor* dssd = [[MTLDepthStencilDescriptor alloc] init];
    dssd.depthCompareFunction = spec.depth_compare_lequal
                                    ? MTLCompareFunctionLessEqual
                                    : MTLCompareFunctionAlways;
    dssd.depthWriteEnabled = YES;
    id<MTLDepthStencilState> dss =
        [device newDepthStencilStateWithDescriptor:dssd];
    [encoder setDepthStencilState:dss];
    const bool vertex_path = spec.plain_control || host_route;
    auto bind_all_stages = [&](id<MTLBuffer> buffer, NSUInteger index) {
      if (vertex_path) {
        [encoder setVertexBuffer:buffer offset:0 atIndex:index];
      } else {
        [encoder setObjectBuffer:buffer offset:0 atIndex:index];
        [encoder setMeshBuffer:buffer offset:0 atIndex:index];
      }
      [encoder setFragmentBuffer:buffer offset:0 atIndex:index];
    };
    bind_all_stages(ab_buffer, kIRArgumentBufferBindPoint);
    bind_all_stages(view_heap, kIRDescriptorHeapBindPoint);
    bind_all_stages(sampler_heap, kIRSamplerHeapBindPoint);
    if (!vertex_path) {
      [encoder setObjectBuffer:ab_buffer
                        offset:0
                       atIndex:kIRArgumentBufferHullDomainBindPoint];
      [encoder setMeshBuffer:ab_buffer
                      offset:0
                     atIndex:kIRArgumentBufferHullDomainBindPoint];
      [encoder setObjectBuffer:tessellator_tables
                        offset:0
                       atIndex:kIRRuntimeTessellatorTablesBindPoint];
      [encoder setMeshBuffer:tessellator_tables
                      offset:0
                     atIndex:kIRRuntimeTessellatorTablesBindPoint];
      IRRuntimeVertexBuffers vertex_buffers = {};
      [encoder setObjectBytes:vertex_buffers
                       length:sizeof(vertex_buffers)
                      atIndex:kIRVertexBufferBindPoint];
    }
    MTLRenderStages stages =
        vertex_path
            ? (MTLRenderStageVertex | MTLRenderStageFragment)
            : (MTLRenderStageObject | MTLRenderStageMesh |
               MTLRenderStageFragment);
    for (id<MTLBuffer> resource :
         {tessellator_tables, ab_buffer, view_heap, sampler_heap, system_views,
          system_buffer, cbv_scratch, pixel_floats, null_buffer,
          guest_scratch}) {
      [encoder useResource:resource usage:MTLResourceUsageRead stages:stages];
    }
    for (id<MTLBuffer> buffer : index_buffers) {
      if (buffer) {
        [encoder useResource:buffer usage:MTLResourceUsageRead stages:stages];
      }
    }
    for (id<MTLBuffer> buffer : {host_records, host_indices, host_table}) {
      if (buffer) {
        [encoder useResource:buffer usage:MTLResourceUsageRead stages:stages];
      }
    }
    [encoder useResource:null_texture
                   usage:MTLResourceUsageRead
                  stages:stages];

    if (spec.plain_control) {
      for (const DrawSpec& draw : spec.draws) {
        IRRuntimeDrawPrimitives(encoder, MTLPrimitiveTypeTriangle,
                                /*vertexStart=*/(uint64_t)0,
                                /*vertexCount=*/(uint64_t)draw.count,
                                /*instanceCount=*/(uint64_t)1,
                                /*baseInstance=*/(uint64_t)0);
      }
    } else if (host_route) {
      for (size_t i = 0; i < spec.draws.size(); ++i) {
        const DrawSpec& draw = spec.draws[i];
        if (draw.has_viewport) {
          MTLViewport viewport = {draw.vp_x, draw.vp_y, draw.vp_w,
                                  draw.vp_h, 0.0,     1.0};
          [encoder setViewport:viewport];
        }
        IRRuntimeDrawIndexedPrimitives(
            encoder, MTLPrimitiveTypeTriangle, host_ranges[i].index_count,
            MTLIndexTypeUInt32, host_indices,
            host_ranges[i].index_offset_bytes, /*instanceCount=*/1,
            /*baseVertex=*/0, /*baseInstance=*/0);
      }
    } else if (spec.geometry_control) {
      IRRuntimeGeometryPipelineConfig geometry_config = {};
      geometry_config.gsVertexSizeInBytes = control_gs_vertex_size;
      geometry_config.gsMaxInputPrimitivesPerMeshThreadgroup =
          control_gs_max_input;
      for (const DrawSpec& draw : spec.draws) {
        IRRuntimeDrawPrimitivesGeometryEmulation(
            encoder, IRRuntimePrimitiveTypeTriangle, geometry_config,
            /*instanceCount=*/1, /*vertexCountPerInstance=*/draw.count,
            /*baseVertex=*/0, /*baseInstance=*/0);
      }
    } else {
      // Config rebuilt VERBATIM from the pipeline mirror (group-4 contract).
      IRRuntimeTessellationPipelineConfig config = {};
      config.outputPrimitiveType =
          IRRuntimeTessellatorOutputPrimitive(tess->output_primitive_type);
      config.vsOutputSizeInBytes = tess->vs_output_size_in_bytes;
      config.gsMaxInputPrimitivesPerMeshThreadgroup =
          tess->gs_max_input_primitives_per_mesh_threadgroup;
      config.hsMaxPatchesPerObjectThreadgroup =
          tess->hs_max_patches_per_object_threadgroup;
      config.hsInputControlPointCount = tess->hs_input_control_point_count;
      config.hsMaxObjectThreadsPerThreadgroup =
          tess->hs_max_object_threads_per_threadgroup;
      config.hsMaxTessellationFactor = tess->hs_max_tessellation_factor;
      config.gsInstanceCount = tess->gs_instance_count;
      IRRuntimePrimitiveType patch_primitive =
          tess->hs_input_control_point_count == 3
              ? IRRuntimePrimitiveType3ControlPointPatchlist
              : IRRuntimePrimitiveType4ControlPointPatchlist;

      for (size_t i = 0; i < spec.draws.size(); ++i) {
        const DrawSpec& draw = spec.draws[i];
        if (draw.has_viewport) {
          MTLViewport viewport = {draw.vp_x, draw.vp_y, draw.vp_w,
                                  draw.vp_h, 0.0,     1.0};
          [encoder setViewport:viewport];
        }
        const bool local_dispatch = spec.tg_mem_length != 0 ||
                                    spec.hs_patches_override != 0 ||
                                    spec.index_pad_words != 0;
        if (draw.indexed && local_dispatch) {
          IRRuntimeTessellationPipelineConfig local_config = config;
          if (spec.hs_patches_override) {
            local_config.hsMaxPatchesPerObjectThreadgroup =
                spec.hs_patches_override;
          }
          LocalDrawIndexedPatchesTessellationEmulation(
              encoder, patch_primitive,
              draw.indices_16bit ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32,
              index_buffers[i], local_config, /*instance_count=*/1,
              /*index_count_per_instance=*/draw.count, /*base_instance=*/0,
              /*base_vertex=*/0, /*start_index=*/spec.index_pad_words,
              spec.tg_mem_length ? spec.tg_mem_length : 15360u);
        } else if (draw.indexed) {
          IRRuntimeDrawIndexedPatchesTessellationEmulation(
              encoder, patch_primitive,
              draw.indices_16bit ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32,
              index_buffers[i], config, /*instanceCount=*/1,
              /*indexCountPerInstance=*/draw.count, /*baseInstance=*/0,
              /*baseVertex=*/0, /*startIndex=*/0);
        } else {
          IRRuntimeDrawPatchesTessellationEmulation(
              encoder, patch_primitive, config, /*instanceCount=*/1,
              /*vertexCountPerInstance=*/draw.count, /*baseInstance=*/0,
              /*baseVertex=*/0);
        }
      }
    }
    [encoder endEncoding];

    if (kernel_readback) {
      id<MTLComputeCommandEncoder> compute =
          [command_buffer computeCommandEncoder];
      [compute setComputePipelineState:spec.sample_count > 1
                                           ? expand_depth_ms
                                           : expand_depth_1x];
      [compute setTexture:depth_rt atIndex:0];
      [compute setBuffer:depth_readback offset:0 atIndex:0];
      [compute dispatchThreads:MTLSizeMake(kRT, kRT, 1)
          threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      if (color_rt && spec.sample_count > 1) {
        [compute setComputePipelineState:expand_color_ms];
        [compute setTexture:color_rt atIndex:0];
        [compute setBuffer:readback offset:0 atIndex:0];
        [compute dispatchThreads:MTLSizeMake(kRT, kRT, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      }
      [compute endEncoding];
    }
    if (!kernel_readback || (color_rt && spec.sample_count == 1)) {
      id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
      if (color_rt && spec.sample_count == 1) {
        [blit copyFromTexture:color_rt
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kRT, kRT, 1)
                        toBuffer:readback
               destinationOffset:0
          destinationBytesPerRow:kRT * 4
        destinationBytesPerImage:kRT * kRT * 4];
      }
      if (!kernel_readback) {
        [blit copyFromTexture:depth_rt
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kRT, kRT, 1)
                        toBuffer:depth_readback
               destinationOffset:0
          destinationBytesPerRow:kRT * 4
        destinationBytesPerImage:kRT * kRT * 4];
      }
      [blit endEncoding];
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted ||
        command_buffer.error) {
      result.error =
          command_buffer.error
              ? [[command_buffer.error localizedDescription] UTF8String]
              : "command buffer not completed";
      return result;
    }

    // --- Read back + coverage count (sample-plane-major; plane 0 first so
    // the single-sample coverage/oracle indexing is unchanged). ---
    if (!spec.depth_only) {
      result.rgba.resize(kRT * kRT * 4 * sample_planes);
      std::memcpy(result.rgba.data(), [readback contents],
                  result.rgba.size());
    }
    result.depth.resize(kRT * kRT * sample_planes);
    std::memcpy(result.depth.data(), [depth_readback contents],
                result.depth.size() * sizeof(float));
    result.ok = true;
    for (uint32_t y = 0; y < kRT; ++y) {
      for (uint32_t x = 0; x < kRT; ++x) {
        if (result.CoveredAt(x, y)) {
          ++result.covered;
        }
        if (result.depth[y * kRT + x] < 0.999f) {
          ++result.depth_covered;
        }
      }
    }
    return result;
  }
};

RenderHarness& H() {
  static RenderHarness harness;
  return harness;
}

// The payload maps NDC x in [0,1] -> pixels [kHalf, kRT), NDC y in [0,1] ->
// pixels [0, kHalf) (+Y up). "Outside the quadrant" leaves 1px of rasterizer
// slack at the shared edges.
uint32_t CountOutsideQuadrant(const RenderResult& r) {
  uint32_t outside = 0;
  for (uint32_t y = 0; y < kRT; ++y) {
    for (uint32_t x = 0; x < kRT; ++x) {
      bool inside_with_slack = (x + 1 >= kHalf) && (y <= kHalf);
      if (!inside_with_slack && r.CoveredAt(x, y)) {
        ++outside;
      }
    }
  }
  return outside;
}

// Triangle-domain symmetry: the covered set in the quadrant should be
// symmetric across the u=v diagonal (u = x-kHalf, v = kHalf-1-y).
uint32_t CountDiagonalAsymmetry(const RenderResult& r) {
  uint32_t mismatched = 0;
  for (uint32_t j = 0; j < kHalf; ++j) {
    for (uint32_t i = 0; i < kHalf; ++i) {
      bool a = r.CoveredAt(kHalf + i, kHalf - 1 - j);
      bool b = r.CoveredAt(kHalf + j, kHalf - 1 - i);
      if (a != b) {
        ++mismatched;
      }
    }
  }
  return mismatched;
}

DrawSpec NonIndexedDraw(uint32_t count) {
  DrawSpec draw;
  draw.count = count;
  return draw;
}

DrawSpec FactorStreamDraw(uint32_t control_points, float factor_value) {
  DrawSpec draw;
  draw.indexed = true;
  draw.words32.assign(control_points, FloatBits(factor_value));
  draw.count = control_points;
  return draw;
}

DrawSpec Indexed16Draw(std::vector<uint16_t> indices) {
  DrawSpec draw;
  draw.indexed = true;
  draw.indices_16bit = true;
  draw.count = uint32_t(indices.size());
  draw.words16 = std::move(indices);
  return draw;
}

}  // namespace

// ---------------------------------------------------------------------------
// 0. Fixture integrity: the hand-assembled microcode decodes as intended and
// carries the properties the other cases rely on.

TEST_CASE("tessellation render fixtures disassemble as intended",
          "[tessellation][render][fixture]") {
  // Identity-swizzle domain payload (triangle domains + quad CP-indexed).
  {
    std::vector<uint32_t> ucode = BuildDomainPayloadUcode(kSrcSwizzleIdentity);
    MetalShader shader(xenos::ShaderType::kVertex, 0x5511551155115511ull,
                       ucode.data(), ucode.size(), std::endian::little);
    StringBuffer disasm;
    shader.AnalyzeUcode(disasm);
    std::string text = disasm.to_string();
    INFO(text);
    CHECK(text.find("max r1.xy__, r0, r0") != std::string::npos);
    CHECK(text.find("max oPos.xy01, r1, r1") != std::string::npos);
    CHECK(text.find("exece") != std::string::npos);
    // TWO temp registers: register_count() >= 2 is asserted by the domain
    // prologue in Checked builds (dxbc_shader_translator.cc:489).
    CHECK(shader.register_static_address_bound() == 2);
    CHECK(shader.memexport_eM_written() == 0);
  }
  // .yz source swizzle (quad patch-indexed: r0.x is the patch index).
  {
    std::vector<uint32_t> ucode = BuildDomainPayloadUcode(kSrcSwizzleYZ);
    MetalShader shader(xenos::ShaderType::kVertex, 0x5522552255225522ull,
                       ucode.data(), ucode.size(), std::endian::little);
    StringBuffer disasm;
    shader.AnalyzeUcode(disasm);
    std::string text = disasm.to_string();
    INFO(text);
    CHECK(text.find("max r1.xy__, r0.yzzw, r0.yzzw") != std::string::npos);
    CHECK(shader.register_static_address_bound() == 2);
  }
  // The guest PS: oC0 = c0.
  {
    std::vector<uint32_t> ucode = BuildPixelUcode();
    MetalShader shader(xenos::ShaderType::kPixel, 0x5533553355335533ull,
                       ucode.data(), ucode.size(), std::endian::little);
    StringBuffer disasm;
    shader.AnalyzeUcode(disasm);
    std::string text = disasm.to_string();
    INFO(text);
    CHECK(text.find("oC0") != std::string::npos);
    CHECK(shader.writes_color_target(0));
  }
}

// ---------------------------------------------------------------------------
// 0b. Harness environment control: the SAME binding assembly, system
// constants, guest PS, and readback, but through the live-proven
// geometry-emulation mesh path. If this renders and the tessellation cells do
// not, the failure is tessellation-leg-specific by construction.

// ---------------------------------------------------------------------------
// fix-metal-geometry-clip-distance-bridge tasks 4.2-4.5: USER CLIP PLANE
// SEMANTICS on the geometry-emulation leg. This is the load-bearing evidence;
// 4.1's pipeline creation proved AVAILABILITY only.
//
// ORDERING IS A DEPENDENCY, NOT A PREFERENCE:
//   INSIDE   proves the production draw and the constants binding work at all.
//   OUTSIDE  proves clipping REMOVES that known-working geometry. An empty mask
//            is meaningless on its own -- a broken draw or binding produces the
//            same zero -- so outside is only read AFTER inside passes, on the
//            SAME pipeline, geometry and attachments, varying ONLY the plane.
//   CROSSING proves partial coverage against the independent oracle.
//
// WHAT THE ORACLE PREDICTS, stated precisely. The guest VS produces its own
// shape, and geometry is NOT under test -- clipping is. So the INSIDE render is
// taken as ground-truth geometry, and the oracle predicts the CLIPPED result
// from it using the Xenos half-space rule projected to screen space. For the
// plane x >= 0 on a 16-wide target that predicate is exactly "column >= 8",
// which falls between columns 7 and 8 and so avoids pixel-centre ambiguity.
TEST_CASE("clip-bridge 4.2-4.5: user clip plane coverage on geometry emulation",
          "[tessellation][render][device][clip42]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  // The relabel is UNCONDITIONAL now; nothing to enable.

  auto run = [&](const char* label, uint32_t count, float a, float b, float c,
                 float d) {
    RenderSpec spec;
    spec.label = label;
    spec.geometry_control = true;
    spec.user_clip_plane_count = count;
    spec.clip_plane0[0] = a;
    spec.clip_plane0[1] = b;
    spec.clip_plane0[2] = c;
    spec.clip_plane0[3] = d;
    spec.draws.push_back(NonIndexedDraw(3));
    return h.Render(spec);
  };

  // --- 4.2 INSIDE: plane x + 2 >= 0 contains everything. Proves the draw and
  // the constants binding work; every later reading depends on this. ---
  RenderResult inside = run("clip-inside", 1, 1.0f, 0.0f, 0.0f, 2.0f);
  INFO("inside: " << inside.error);
  std::printf("CLIP42|inside|%s|covered=%u\n",
              inside.ok && inside.covered > 0 ? "PASS" : "FAIL", inside.covered);
  REQUIRE(inside.ok);
  REQUIRE(inside.covered > 0);  // GATES everything below

  // --- 4.3 OUTSIDE: plane x - 2 >= 0 excludes everything. Same pipeline,
  // geometry and attachments; ONLY the plane constants differ. ---
  RenderResult outside = run("clip-outside", 1, 1.0f, 0.0f, 0.0f, -2.0f);
  INFO("outside: " << outside.error);
  std::printf("CLIP42|outside|%s|covered=%u (inside was %u)\n",
              outside.ok && outside.covered == 0 ? "PASS" : "FAIL",
              outside.covered, inside.covered);
  CHECK(outside.ok);
  CHECK(outside.covered == 0);

  // --- 4.4/4.5 CROSSING: plane x >= 0 keeps the right half. Compared against
  // the oracle's prediction derived from the INSIDE mask. ---
  // The geometry occupies screen x=[64,126], i.e. it begins exactly at NDC
  // x=0 -- so a plane at x >= 0 would keep ALL of it and prove nothing (the
  // guard below catches precisely that). Cut through its middle instead:
  // x - 0.5 >= 0 keeps NDC x >= 0.5, which is screen column 96 on a 128-wide
  // target. Pixel centres lie at 95.5 and 96.5, so the boundary falls strictly
  // BETWEEN centres and no pixel is ambiguous.
  constexpr float kCutNdcX = 0.5f;
  RenderResult crossing = run("clip-crossing", 1, 1.0f, 0.0f, 0.0f, -kCutNdcX);
  INFO("crossing: " << crossing.error);
  REQUIRE(crossing.ok);

  // Build both masks from the readback, then predict the crossing mask from the
  // inside mask by the oracle's screen-space half-space rule.
  auto mask_of = [](const RenderResult& r) {
    std::vector<uint8_t> m(size_t(kRT) * kRT, 0);
    for (size_t i = 0; i < m.size(); ++i) {
      const uint8_t* px = r.rgba.data() + i * 4;
      m[i] = (px[0] || px[1] || px[2] || px[3]) ? 1 : 0;
    }
    return m;
  };
  const std::vector<uint8_t> inside_mask = mask_of(inside);
  {
    // Locate the geometry before choosing a cutting plane: a plane that misses
    // it entirely produces predicted == inside and proves nothing.
    uint32_t min_x = kRT, max_x = 0, min_y = kRT, max_y = 0;
    for (uint32_t y = 0; y < kRT; ++y) {
      for (uint32_t x = 0; x < kRT; ++x) {
        if (inside_mask[size_t(y) * kRT + x]) {
          min_x = std::min(min_x, x); max_x = std::max(max_x, x);
          min_y = std::min(min_y, y); max_y = std::max(max_y, y);
        }
      }
    }
    std::printf("CLIP42|geometry_extent|x=[%u,%u] y=[%u,%u] of %u\n",
                min_x, max_x, min_y, max_y, kRT);
  }
  const std::vector<uint8_t> crossing_mask = mask_of(crossing);
  std::vector<uint8_t> predicted(inside_mask.size(), 0);
  uint32_t predicted_covered = 0;
  for (uint32_t y = 0; y < kRT; ++y) {
    for (uint32_t x = 0; x < kRT; ++x) {
      // Oracle prediction, derived from the plane rather than hardcoded:
      // NDC x >= kCutNdcX maps to screen column >= (kCutNdcX*0.5+0.5)*kRT.
      const uint32_t cut_column =
          uint32_t((kCutNdcX * 0.5f + 0.5f) * float(kRT));
      const bool keep = x >= cut_column;
      const size_t i = size_t(y) * kRT + x;
      predicted[i] = (inside_mask[i] && keep) ? 1 : 0;
      predicted_covered += predicted[i];
    }
  }
  uint32_t mismatches = 0;
  for (size_t i = 0; i < predicted.size(); ++i) {
    if (predicted[i] != crossing_mask[i]) ++mismatches;
  }
  std::printf(
      "CLIP42|crossing|%s|covered=%u predicted=%u mismatches=%u (inside %u)\n",
      mismatches == 0 ? "PASS" : "FAIL", crossing.covered, predicted_covered,
      mismatches, inside.covered);
  // The crossing plane must actually cut: a prediction equal to the inside mask
  // would mean the plane did nothing and the comparison proves nothing.
  CHECK(predicted_covered > 0);
  CHECK(predicted_covered < inside.covered);
  CHECK(mismatches == 0);
}

// ---------------------------------------------------------------------------
// Tasks 4.6/4.7 (sample-count stability) and 4.8 (the ordinary non-geometry
// path as the UNCHANGED boundary control).
TEST_CASE("clip-bridge 4.6-4.8: sample counts and the non-geometry boundary",
          "[tessellation][render][device][clip468]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  constexpr float kCutNdcX = 0.5f;
  const uint32_t cut_column = uint32_t((kCutNdcX * 0.5f + 0.5f) * float(kRT));

  auto mask_of = [](const RenderResult& r) {
    std::vector<uint8_t> m(size_t(kRT) * kRT, 0);
    for (size_t i = 0; i < m.size(); ++i) {
      const uint8_t* px = r.rgba.data() + i * 4;
      m[i] = (px[0] || px[1] || px[2] || px[3]) ? 1 : 0;
    }
    return m;
  };
  auto run = [&](const char* label, bool geometry, uint32_t samples,
                 uint32_t count, float d) {
    RenderSpec spec;
    spec.label = label;
    spec.geometry_control = geometry;
    spec.plain_control = !geometry;
    spec.sample_count = samples;
    spec.user_clip_plane_count = count;
    spec.clip_plane0[0] = 1.0f;
    spec.clip_plane0[3] = d;
    spec.draws.push_back(NonIndexedDraw(3));
    return h.Render(spec);
  };

  SECTION("4.6/4.7 sample-count stability at 1x and 4x") {
    for (uint32_t samples : {1u, 4u}) {
      // Expected geometry is derived from THIS sample count's own inside arm,
      // never carried across sample counts: 4x rasterization legitimately
      // covers different pixels, so borrowing the 1x mask would manufacture a
      // mismatch that says nothing about clipping.
      RenderResult inside = run("clip-inside", true, samples, 1, 2.0f);
      INFO("inside(" << samples << "x): " << inside.error);
      REQUIRE(inside.ok);
      REQUIRE(inside.covered > 0);

      RenderResult crossing = run("clip-crossing", true, samples, 1, -kCutNdcX);
      REQUIRE(crossing.ok);

      const auto in_mask = mask_of(inside);
      const auto cr_mask = mask_of(crossing);
      uint32_t predicted_covered = 0, mismatches = 0;
      for (uint32_t y = 0; y < kRT; ++y) {
        for (uint32_t x = 0; x < kRT; ++x) {
          const size_t i = size_t(y) * kRT + x;
          const uint8_t predicted = (in_mask[i] && x >= cut_column) ? 1 : 0;
          predicted_covered += predicted;
          if (predicted != cr_mask[i]) ++mismatches;
        }
      }
      std::printf(
          "CLIP468|%ux|%s|inside=%u crossing=%u predicted=%u mismatches=%u\n",
          samples, mismatches == 0 ? "PASS" : "FAIL", inside.covered,
          crossing.covered, predicted_covered, mismatches);
      CHECK(predicted_covered > 0);
      CHECK(predicted_covered < inside.covered);  // the plane must really cut
      CHECK(mismatches == 0);
    }
  }

  SECTION("4.8 the ordinary non-geometry path is UNCHANGED by the gate") {
    // Rendered, not merely materialized. With the relabel unconditional this
    // renders the ordinary path TWICE and requires byte identity: it is now a
    // determinism + non-interference witness rather than a gate A/B, and it
    // still fails loudly if the relabel ever leaked into ordinary pipelines.
    RenderResult off = run("plain-clip-a", false, 1, 1, -kCutNdcX);
    RenderResult on = run("plain-clip-b", false, 1, 1, -kCutNdcX);
    INFO("plain off: " << off.error << " on: " << on.error);
    REQUIRE(off.ok);
    REQUIRE(on.ok);
    REQUIRE(off.covered > 0);  // the control must actually render
    const bool identical = off.rgba == on.rgba;
    std::printf("CLIP468|plain-boundary|%s|off=%u on=%u byte_identical=%d\n",
                identical && off.covered == on.covered ? "PASS" : "FAIL",
                off.covered, on.covered, identical ? 1 : 0);
    CHECK(off.covered == on.covered);
    CHECK(identical);
  }
}

TEST_CASE("harness control: geometry-emulation path renders in this environment",
          "[tessellation][render][control][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  // Plain vertex+fragment first: no mesh stages, same AB/constants/PS.
  RenderSpec plain;
  plain.label = "plain-control";
  plain.plain_control = true;
  plain.draws.push_back(NonIndexedDraw(3));
  RenderResult pr = h.Render(plain);
  INFO("plain: " << pr.error);
  std::printf(
      "TESSRENDER|CONTROL|plain-vertex|%s|covered=%u depth_covered=%u\n",
      pr.ok && pr.covered > 0 ? "PASS" : "FAIL", pr.covered,
      pr.depth_covered);
  CHECK(pr.ok);
  CHECK(pr.depth_covered > 0);
  CHECK(pr.covered > 0);

  RenderSpec spec;
  spec.label = "geometry-control";
  spec.geometry_control = true;
  spec.draws.push_back(NonIndexedDraw(3));  // One rect-list primitive.
  RenderResult r = h.Render(spec);
  INFO(r.error);
  std::printf(
      "TESSRENDER|CONTROL|geometry-emulation|%s|covered=%u depth_covered=%u\n",
      r.ok && r.covered > 0 ? "PASS" : "FAIL", r.covered, r.depth_covered);
  CHECK(r.ok);
  // depth_covered proves rasterization even if the color path were broken.
  CHECK(r.depth_covered > 0);
  CHECK(r.covered > 0);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 0c. Plain-pipeline variant ladder (diagnostic): starts at the proven
// line-loop-test shape (native white fragment, no depth, null AB except
// system constants) and flips one delta per rung toward the tessellation
// harness environment. The rung where coverage dies names the killing delta.

namespace {

struct PlainVariant {
  const char* name;
  bool guest_ps;      // native white fragment otherwise.
  bool with_depth;    // depth attachment + always-write DSS.
  bool triangles;     // line strip otherwise.
  bool my_constants;  // the tessellation harness constants otherwise
                      // line-loop's exact block.
  bool real_ab;       // full builder AB + heaps otherwise null AB.
};

struct PlainResult {
  bool ok = false;
  std::string error;
  uint32_t covered = 0;
};

PlainResult RunPlainVariant(const PlainVariant& v) {
  PlainResult out;
  auto& h = H();
  id<MTLDevice> device = h.device;

  // Guest VS: the parabola fixture as a plain kVertex translation.
  MetalShader* vertex_shader = h.cache->LoadShader(
      xenos::ShaderType::kVertex, GuestParabolaUcode().data(),
      uint32_t(GuestParabolaUcode().size()));
  if (!vertex_shader->is_ucode_analyzed()) {
    StringBuffer disasm;
    vertex_shader->AnalyzeUcode(disasm);
  }
  DxbcShaderTranslator::Modification vm(
      h.cache->shader_translator()->GetDefaultVertexShaderModification(
          vertex_shader->GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters),
          HostVertexShaderType::kVertex));
  vm.vertex.interpolator_mask = 0;
  auto* vertex_translation = static_cast<MetalShader::MetalTranslation*>(
      vertex_shader->GetOrCreateTranslation(vm.value));
  if (!vertex_translation->has_metal_function() &&
      !h.cache->TranslateShader(*vertex_translation)) {
    out.error = "VS translation failed";
    return out;
  }

  id<MTLFunction> fragment_function = nil;
  id<MTLLibrary> white_library = nil;
  if (v.guest_ps) {
    MetalShader::MetalTranslation* pixel = h.MakeGuestPixelShader();
    if (!pixel) {
      out.error = "guest PS failed";
      return out;
    }
    fragment_function = (__bridge id<MTLFunction>)pixel->metal_function();
  } else {
    static NSString* source =
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "fragment float4 tess_white() { return float4(1.0); }\n";
    NSError* ns_error = nil;
    white_library = [device newLibraryWithSource:source
                                         options:nil
                                           error:&ns_error];
    fragment_function = [white_library newFunctionWithName:@"tess_white"];
    if (!fragment_function) {
      out.error = "white fragment failed";
      return out;
    }
  }

  MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
  desc.vertexFunction =
      (__bridge id<MTLFunction>)vertex_translation->metal_function();
  desc.fragmentFunction = fragment_function;
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  if (v.with_depth) {
    desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  }
  desc.rasterSampleCount = 1;
  NSError* ns_error = nil;
  id<MTLRenderPipelineState> pipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&ns_error];
  if (!pipeline) {
    out.error = ns_error ? ns_error.localizedDescription.UTF8String
                         : "pipeline failed";
    return out;
  }

  MTLTextureDescriptor* td = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:kRT
                                  height:kRT
                               mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> color_rt = [device newTextureWithDescriptor:td];
  id<MTLTexture> depth_rt = nil;
  if (v.with_depth) {
    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                     width:kRT
                                    height:kRT
                                 mipmapped:NO];
    dd.storageMode = MTLStorageModePrivate;
    dd.usage = MTLTextureUsageRenderTarget;
    depth_rt = [device newTextureWithDescriptor:dd];
  }

  id<MTLBuffer> system_buffer = h.NewZeroBuffer(
      sizeof(DxbcShaderTranslator::SystemConstants), @"SysConst (variant)");
  auto* constants = static_cast<DxbcShaderTranslator::SystemConstants*>(
      [system_buffer contents]);
  if (v.my_constants) {
    constants->flags = DxbcShaderTranslator::kSysFlag_WNotReciprocal |
                       DxbcShaderTranslator::kSysFlag_AlphaPassIfLess |
                       DxbcShaderTranslator::kSysFlag_AlphaPassIfEqual |
                       DxbcShaderTranslator::kSysFlag_AlphaPassIfGreater;
    constants->vertex_index_endian = xenos::Endian::kNone;
    constants->vertex_index_max = 0xFFFFFFu;
    constants->ndc_scale[0] = 1.0f;
    constants->ndc_scale[1] = 1.0f;
    constants->ndc_scale[2] = 1.0f;
    for (uint32_t i = 0; i < 4; ++i) {
      constants->color_exp_bias[i] = 1.0f;
    }
  } else {
    constants->flags = DxbcShaderTranslator::kSysFlag_WNotReciprocal;
    constants->vertex_index_endian = xenos::Endian::kNone;
    constants->vertex_index_min = 0;
    constants->vertex_index_max = 0xFFFFFFu;
    constants->ndc_scale[0] = 0.5f;
    constants->ndc_scale[1] = 1.0f;
    constants->ndc_scale[2] = 1.0f;
    constants->ndc_offset[0] = -0.5f;
  }

  id<MTLBuffer> null_buffer = h.NewZeroBuffer(4096, @"Null (variant)");
  id<MTLBuffer> ab_buffer =
      h.NewZeroBuffer(kMetalBindingSlotCount * sizeof(uint64_t), @"AB (variant)");
  id<MTLBuffer> pixel_floats = h.NewZeroBuffer(4096, @"PS floats (variant)");
  {
    float* c0 = static_cast<float*>([pixel_floats contents]);
    c0[0] = 1.0f;
    c0[1] = 1.0f;
    c0[2] = 0.0f;
    c0[3] = 1.0f;
  }
  auto* ab_words = static_cast<uint64_t*>([ab_buffer contents]);
  if (v.real_ab) {
    MetalBindingAddresses addresses;
    addresses.null_buffer = [null_buffer gpuAddress];
    addresses.view_heap = [null_buffer gpuAddress];
    addresses.sampler_heap = [null_buffer gpuAddress];
    addresses.shared_memory_srv = [null_buffer gpuAddress];
    addresses.null_uav = [null_buffer gpuAddress];
    addresses.shared_memory_uav = [null_buffer gpuAddress];
    addresses.system_constants = [system_buffer gpuAddress];
    addresses.float_vertex = [null_buffer gpuAddress];
    addresses.bool_loop = [null_buffer gpuAddress];
    addresses.fetch = [null_buffer gpuAddress];
    addresses.descriptor_indices_vertex = [null_buffer gpuAddress];
    addresses.float_pixel = [pixel_floats gpuAddress];
    addresses.descriptor_indices_pixel = [null_buffer gpuAddress];
    std::array<uint64_t, kMetalBindingSlotCount> argument_buffer =
        BuildMetalTopLevelArgumentBuffer(addresses, v.guest_ps,
                                         /*memexport_draw=*/false,
                                         /*tessellation_draw=*/false);
    std::memcpy(ab_words, argument_buffer.data(),
                kMetalBindingSlotCount * sizeof(uint64_t));
  } else {
    for (uint32_t i = 0; i < kMetalBindingSlotCount; ++i) {
      ab_words[i] = [null_buffer gpuAddress];
    }
    ab_words[uint32_t(metal::MetalBindingSlot::kCbvSystem)] =
        [system_buffer gpuAddress];
    if (v.guest_ps) {
      ab_words[uint32_t(metal::MetalBindingSlot::kCbvFloatPixel)] =
          [pixel_floats gpuAddress];
    }
  }

  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = color_rt;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  if (v.with_depth) {
    pass.depthAttachment.texture = depth_rt;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 1.0;
    pass.depthAttachment.storeAction = MTLStoreActionDontCare;
  }
  id<MTLCommandBuffer> command_buffer = [h.queue commandBuffer];
  id<MTLRenderCommandEncoder> encoder =
      [command_buffer renderCommandEncoderWithDescriptor:pass];
  [encoder setRenderPipelineState:pipeline];
  [encoder setViewport:MTLViewport{0.0, 0.0, double(kRT), double(kRT), 0.0,
                                   1.0}];
  if (v.with_depth) {
    MTLDepthStencilDescriptor* dssd = [[MTLDepthStencilDescriptor alloc] init];
    dssd.depthCompareFunction = MTLCompareFunctionAlways;
    dssd.depthWriteEnabled = YES;
    [encoder setDepthStencilState:[h.device
                                      newDepthStencilStateWithDescriptor:dssd]];
  }
  [encoder setVertexBuffer:ab_buffer offset:0 atIndex:kIRArgumentBufferBindPoint];
  [encoder setVertexBuffer:null_buffer offset:0 atIndex:kIRDescriptorHeapBindPoint];
  [encoder setVertexBuffer:null_buffer offset:0 atIndex:kIRSamplerHeapBindPoint];
  if (v.guest_ps) {
    [encoder setFragmentBuffer:ab_buffer
                        offset:0
                       atIndex:kIRArgumentBufferBindPoint];
    [encoder setFragmentBuffer:null_buffer
                        offset:0
                       atIndex:kIRDescriptorHeapBindPoint];
    [encoder setFragmentBuffer:null_buffer
                        offset:0
                       atIndex:kIRSamplerHeapBindPoint];
  }
  MTLRenderStages stages = MTLRenderStageVertex | MTLRenderStageFragment;
  for (id<MTLBuffer> b : {ab_buffer, system_buffer, null_buffer, pixel_floats}) {
    [encoder useResource:b usage:MTLResourceUsageRead stages:stages];
  }
  IRRuntimeDrawPrimitives(
      encoder, v.triangles ? MTLPrimitiveTypeTriangle : MTLPrimitiveTypeLineStrip,
      (uint64_t)0, (uint64_t)3, (uint64_t)1, (uint64_t)0);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (command_buffer.status != MTLCommandBufferStatusCompleted) {
    out.error = command_buffer.error
                    ? [[command_buffer.error localizedDescription] UTF8String]
                    : "CB failed";
    return out;
  }
  std::vector<uint8_t> rgba(kRT * kRT * 4);
  [color_rt getBytes:rgba.data()
         bytesPerRow:kRT * 4
          fromRegion:MTLRegionMake2D(0, 0, kRT, kRT)
         mipmapLevel:0];
  out.ok = true;
  for (uint32_t i = 0; i < kRT * kRT; ++i) {
    if (rgba[i * 4] | rgba[i * 4 + 1] | rgba[i * 4 + 2] | rgba[i * 4 + 3]) {
      ++out.covered;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("plain-pipeline variant ladder locates the environment delta",
          "[tessellation][render][control2][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  const PlainVariant variants[] = {
      {"V0 lineloop-replica", false, false, false, false, false},
      {"V1 +triangles", false, false, true, false, false},
      {"V2 +my-constants", false, false, true, true, false},
      {"V3 +real-ab", false, false, true, true, true},
      {"V4 +depth", false, true, true, true, true},
      {"V5 +guest-ps", true, true, true, true, true},
  };
  for (const PlainVariant& v : variants) {
    PlainResult r = RunPlainVariant(v);
    std::printf("TESSRENDER|LADDER|%s|%s|covered=%u%s%s\n", v.name,
                r.ok && r.covered > 0 ? "PASS" : "FAIL", r.covered,
                r.error.empty() ? "" : " err=", r.error.c_str());
  }
  SUCCEED("ladder recorded");
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 1. The six mode/domain render cells.


// implement-metal-host-tessellation task 4.1: runs a TEST_CASE body on BOTH
// routes (Catch GENERATE re-enters the case per value) and restores the
// shared harness's route on every exit, including assertion failures.
struct RouteScope {
  RenderHarness& h;
  RouteScope(RenderHarness& harness, bool host_route) : h(harness) {
    h.host_route = host_route;
  }
  ~RouteScope() { h.host_route = false; }
};
#define XE_TESS_BOTH_ROUTES(h)                                       \
  const bool host_route = GENERATE(false, true);                      \
  RouteScope route_scope(h, host_route);                              \
  INFO("route=" << (host_route ? "host" : "mesh"))

TEST_CASE("all six tessellation mode/domain cells render coverage",
          "[tessellation][render][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  XE_TESS_BOTH_ROUTES(h);

  struct Cell {
    const char* name;
    xenos::TessellationMode mode;
    HostVertexShaderType type;
    xenos::PrimitiveType prim;
    bool triangle_domain;
    DrawSpec draw;
  };
  const Cell cells[] = {
      {"discrete/triangle", xenos::TessellationMode::kDiscrete,
       HostVertexShaderType::kTriangleDomainCPIndexed,
       xenos::PrimitiveType::kTriangleList, true, NonIndexedDraw(3)},
      {"discrete/quad", xenos::TessellationMode::kDiscrete,
       HostVertexShaderType::kQuadDomainCPIndexed,
       xenos::PrimitiveType::kQuadList, false, NonIndexedDraw(4)},
      {"continuous/triangle", xenos::TessellationMode::kContinuous,
       HostVertexShaderType::kTriangleDomainCPIndexed,
       xenos::PrimitiveType::kTriangleList, true, NonIndexedDraw(3)},
      {"continuous/quad", xenos::TessellationMode::kContinuous,
       HostVertexShaderType::kQuadDomainCPIndexed,
       xenos::PrimitiveType::kQuadList, false, NonIndexedDraw(4)},
      {"adaptive/triangle", xenos::TessellationMode::kAdaptive,
       HostVertexShaderType::kTriangleDomainPatchIndexed,
       xenos::PrimitiveType::kTrianglePatch, true,
       FactorStreamDraw(3, 3.0f)},
      {"adaptive/quad", xenos::TessellationMode::kAdaptive,
       HostVertexShaderType::kQuadDomainPatchIndexed,
       xenos::PrimitiveType::kQuadPatch, false, FactorStreamDraw(4, 3.0f)},
  };

  for (const Cell& cell : cells) {
    RenderSpec spec;
    spec.label = cell.name;
    spec.mode = cell.mode;
    spec.host_vs_type = cell.type;
    spec.host_prim = cell.prim;
    spec.range_min = 1.0f;
    spec.range_max = 8.0f;
    spec.draws.push_back(cell.draw);
    RenderResult r = h.Render(spec);

    INFO("cell " << cell.name << ": " << r.error);
    if (!r.ok) {
      std::printf("TESSRENDER|CELL|%s|FAIL|%s\n", cell.name, r.error.c_str());
      CHECK(r.ok);
      continue;
    }
    uint32_t outside = CountOutsideQuadrant(r);
    uint32_t asymmetry = cell.triangle_domain ? CountDiagonalAsymmetry(r) : 0;
    // Coverage: a filled quad patch covers ~the whole quadrant; a filled
    // triangle patch covers ~half of it. Loose bounds — this asserts "the
    // patch rendered where the domain says", not rasterizer exactness.
    uint32_t lo = cell.triangle_domain ? kQuarterArea / 4 : (kQuarterArea * 7) / 10;
    uint32_t hi = cell.triangle_domain ? (kQuarterArea * 3) / 4 : kQuarterArea + kRT;
    bool coverage_ok = r.covered >= lo && r.covered <= hi;
    bool outside_ok = outside == 0;
    bool symmetry_ok = asymmetry <= kQuarterArea / 10;
    std::printf(
        "TESSRENDER|CELL|%s|%s|covered=%u expected=[%u,%u] outside=%u "
        "asym=%u\n",
        cell.name,
        (coverage_ok && outside_ok && symmetry_ok) ? "PASS" : "FAIL",
        r.covered, lo, hi, outside, asymmetry);
    CHECK(r.covered > 0);
    CHECK(coverage_ok);
    CHECK(outside_ok);
    if (cell.triangle_domain) {
      CHECK(symmetry_ok);
    }
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 2a. Discrete/continuous factor values over the indexed path (line fill mode
// makes the factor pixel-observable: more segments -> more line pixels).

TEST_CASE("discrete/continuous factor range renders more segments at higher max",
          "[tessellation][render][factors][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  XE_TESS_BOTH_ROUTES(h);

  struct Arm {
    const char* name;
    xenos::TessellationMode mode;
    float range_max;
  };
  const Arm arms[] = {
      {"discrete/quad max=2", xenos::TessellationMode::kDiscrete, 2.0f},
      {"discrete/quad max=8", xenos::TessellationMode::kDiscrete, 8.0f},
      {"continuous/quad max=2", xenos::TessellationMode::kContinuous, 2.0f},
      {"continuous/quad max=8", xenos::TessellationMode::kContinuous, 8.0f},
  };
  uint32_t counts[4] = {};
  bool arm_ok[4] = {};
  for (int i = 0; i < 4; ++i) {
    RenderSpec spec;
    spec.label = arms[i].name;
    spec.mode = arms[i].mode;
    spec.host_vs_type = HostVertexShaderType::kQuadDomainCPIndexed;
    spec.host_prim = xenos::PrimitiveType::kQuadList;
    spec.wireframe = true;
    spec.range_min = 1.0f;
    spec.range_max = arms[i].range_max;
    // The indexed path: a real 16-bit index stream (pre-swapped, kNone).
    spec.draws.push_back(Indexed16Draw({0, 1, 2, 3}));
    RenderResult r = h.Render(spec);
    INFO(arms[i].name << ": " << r.error);
    arm_ok[i] = r.ok && r.covered > 0;
    counts[i] = r.covered;
    std::printf("TESSRENDER|FACTOR|%s|%s|wire_covered=%u\n", arms[i].name,
                arm_ok[i] ? "PASS" : "FAIL", r.covered);
    CHECK(r.ok);
    CHECK(r.covered > 0);
  }
  // Higher max factor -> strictly more line pixels, both partitionings.
  if (arm_ok[0] && arm_ok[1]) {
    CHECK(counts[1] > counts[0]);
  }
  if (arm_ok[2] && arm_ok[3]) {
    CHECK(counts[3] > counts[2]);
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 2b. Adaptive factor stream values: monotonicity, the LIVE [2,1] min>max
// range, the hi/lo clamp discriminator, and the endian double-swap arm.

namespace {

RenderResult RenderAdaptiveQuadWire(RenderHarness& h, float range_min,
                                    float range_max, uint32_t factor_bits,
                                    xenos::Endian published_endian) {
  RenderSpec spec;
  spec.mode = xenos::TessellationMode::kAdaptive;
  spec.host_vs_type = HostVertexShaderType::kQuadDomainPatchIndexed;
  spec.host_prim = xenos::PrimitiveType::kQuadPatch;
  spec.wireframe = true;
  spec.range_min = range_min;
  spec.range_max = range_max;
  spec.published_endian = published_endian;
  DrawSpec draw;
  draw.indexed = true;
  draw.words32.assign(4, factor_bits);
  draw.count = 4;
  spec.draws.push_back(draw);
  return h.Render(spec);
}

bool SameImage(const RenderResult& a, const RenderResult& b) {
  return a.ok && b.ok && a.rgba.size() == b.rgba.size() &&
         std::memcmp(a.rgba.data(), b.rgba.data(), a.rgba.size()) == 0;
}

}  // namespace

TEST_CASE("adaptive factor stream values render distinctly; [2,1] clamps to hi",
          "[tessellation][render][factors][clamp][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  XE_TESS_BOTH_ROUTES(h);

  // The prebuilt adaptive VS computes clamp(asfloat(word) + 1, range).
  // Reference ladder, all with valid ranges (effective factor in the name):
  struct Arm {
    const char* name;
    float range_min, range_max;
    float stream_value;  // asfloat of the stream word.
    xenos::Endian endian;
  };
  const Arm arms[] = {
      {"f1 ([1,8], word 0.0)", 1.0f, 8.0f, 0.0f, xenos::Endian::kNone},
      {"f2 ([2,8], word 1.0)", 2.0f, 8.0f, 1.0f, xenos::Endian::kNone},
      {"f3 ([2,8], word 2.0)", 2.0f, 8.0f, 2.0f, xenos::Endian::kNone},
      {"f6 ([2,8], word 5.0)", 2.0f, 8.0f, 5.0f, xenos::Endian::kNone},
      {"f8 ([2,8], word 7.0)", 2.0f, 8.0f, 7.0f, xenos::Endian::kNone},
      // THE LIVE VIVA PINATA RANGE (tasks.md 1.3): published [min,max]=[2,1],
      // min>max. HLSL clamp(x,lo,hi)=min(max(x,lo),hi) collapses to hi=1, so
      // this must render exactly like the honest factor-1.0 arm above and
      // NOTHING like the unclamped factor the stream requests (8).
      {"vp21 ([2,1], word 7.0)", 2.0f, 1.0f, 7.0f, xenos::Endian::kNone},
      // hi/lo discriminator: [6,3] with an above-range stream word renders 3
      // under HLSL order (min(max(8,6),3)=3) and 6 if a lo-wins clamp crept in
      // — visually distinct, unlike [2,1] where both collapse the same way.
      {"sixthree ([6,3], word 7.0)", 6.0f, 3.0f, 7.0f, xenos::Endian::kNone},
      // Endian pair: identical asymmetric-byte stream (0x40A00000 = 5.0f);
      // kNone reads factor 6, publishing k8in32 over the SAME pre-swapped
      // stream double-swaps to a denormal -> +1 -> clamped to min 2.
      {"endianA (kNone, bytes of 5.0)", 2.0f, 8.0f, 5.0f, xenos::Endian::kNone},
      {"endianB (k8in32 over pre-swapped)", 2.0f, 8.0f, 5.0f,
       xenos::Endian::k8in32},
  };
  constexpr int kF1 = 0, kF2 = 1, kF3 = 2, kF6 = 3, kF8 = 4, kVP21 = 5,
                kSixThree = 6, kEndianA = 7, kEndianB = 8;

  RenderResult results[9];
  for (int i = 0; i < 9; ++i) {
    results[i] = RenderAdaptiveQuadWire(h, arms[i].range_min,
                                        arms[i].range_max,
                                        FloatBits(arms[i].stream_value),
                                        arms[i].endian);
    std::printf("TESSRENDER|ADAPTIVE|%s|%s|wire_covered=%u%s%s\n",
                arms[i].name, results[i].ok ? "OK" : "FAIL",
                results[i].covered, results[i].error.empty() ? "" : " err=",
                results[i].error.c_str());
    INFO(arms[i].name << ": " << results[i].error);
    CHECK(results[i].ok);
    CHECK(results[i].covered > 0);
  }

  // Value ladder is strictly monotone in line pixels.
  CHECK(results[kF2].covered < results[kF3].covered);
  CHECK(results[kF3].covered < results[kF6].covered);
  CHECK(results[kF6].covered < results[kF8].covered);

  // THE [2,1] CLAMP SEMANTICS PIN: identical to the honest factor-1 arm,
  // decidedly not the unclamped stream value.
  std::printf(
      "TESSRENDER|CLAMP|vp21=%u f1=%u f2=%u f8=%u same_image(vp21,f1)=%d\n",
      results[kVP21].covered, results[kF1].covered, results[kF2].covered,
      results[kF8].covered, int(SameImage(results[kVP21], results[kF1])));
  CHECK(results[kVP21].covered == results[kF1].covered);
  CHECK(SameImage(results[kVP21], results[kF1]));
  CHECK(results[kVP21].covered != results[kF8].covered);

  // hi/lo order discriminator: [6,3] == factor 3 (hi), != factor 6 (lo).
  std::printf("TESSRENDER|CLAMP|sixthree=%u f3=%u f6=%u\n",
              results[kSixThree].covered, results[kF3].covered,
              results[kF6].covered);
  CHECK(results[kSixThree].covered == results[kF3].covered);
  CHECK(SameImage(results[kSixThree], results[kF3]));
  CHECK(results[kSixThree].covered != results[kF6].covered);

  // Endian: kNone over the pre-swapped stream is exact (== factor 6); a
  // double swap visibly changes the factor (collapses to the range min 2).
  std::printf("TESSRENDER|ENDIAN|A=%u (f6=%u) B=%u (f2=%u)\n",
              results[kEndianA].covered, results[kF6].covered,
              results[kEndianB].covered, results[kF2].covered);
  CHECK(results[kEndianA].covered == results[kF6].covered);
  CHECK(SameImage(results[kEndianA], results[kF6]));
  if (host_route) {
    // The host route never swaps in-shader (vertex_index_endian is not
    // consulted by the domain_as_vertex translation), so the published
    // endian cannot change the factor: B renders exactly like A.
    CHECK(SameImage(results[kEndianB], results[kEndianA]));
  } else {
    CHECK(results[kEndianB].covered != results[kEndianA].covered);
    CHECK(results[kEndianB].covered == results[kF2].covered);
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 2b2. Translator purity probe (device-free): the quad-patch domain payload
// translated by a FRESH DxbcShaderTranslator vs by one that has ALREADY
// translated the triangle-patch payload. Any byte difference = shared
// translator state leaking across domain translations (the render-side
// creation-order divergence would then start here).

namespace {

std::vector<uint8_t> TranslateDomainPayload(
    DxbcShaderTranslator& translator, MetalShader& shader,
    HostVertexShaderType type) {
  DxbcShaderTranslator::Modification modification(
      translator.GetDefaultVertexShaderModification(
          shader.GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters),
          type));
  Shader::Translation* translation =
      shader.GetOrCreateTranslation(modification.value);
  if (!translation->translated_binary().empty()) {
    return translation->translated_binary();
  }
  if (!translator.TranslateAnalyzedShader(*translation)) {
    return {};
  }
  return translation->translated_binary();
}

}  // namespace

TEST_CASE("domain translation is independent of prior translator use",
          "[tessellation][render][translatorpurity]") {
  auto make_shader = [](uint32_t swizzle, uint64_t hash) {
    auto ucode = BuildDomainPayloadUcode(swizzle);
    auto shader = std::make_unique<MetalShader>(
        xenos::ShaderType::kVertex, hash, ucode.data(), ucode.size(),
        std::endian::little);
    StringBuffer disasm;
    shader->AnalyzeUcode(disasm);
    return shader;
  };

  // Fresh translator: quad-patch payload only.
  std::vector<uint8_t> fresh;
  {
    DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                    /*bindless_resources_used=*/true,
                                    /*edram_rov_used=*/false,
                                    /*gamma_render_target_as_unorm8=*/true);
    auto quad = make_shader(kSrcSwizzleYZ, 0x6001600160016001ull);
    fresh = TranslateDomainPayload(
        translator, *quad, HostVertexShaderType::kQuadDomainPatchIndexed);
  }
  // Used translator: triangle-patch payload first, then quad-patch.
  std::vector<uint8_t> after_triangle;
  {
    DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                    /*bindless_resources_used=*/true,
                                    /*edram_rov_used=*/false,
                                    /*gamma_render_target_as_unorm8=*/true);
    auto tri = make_shader(kSrcSwizzleIdentity, 0x6002600260026002ull);
    std::vector<uint8_t> tri_binary = TranslateDomainPayload(
        translator, *tri, HostVertexShaderType::kTriangleDomainPatchIndexed);
    CHECK_FALSE(tri_binary.empty());
    auto quad = make_shader(kSrcSwizzleYZ, 0x6003600360036003ull);
    after_triangle = TranslateDomainPayload(
        translator, *quad, HostVertexShaderType::kQuadDomainPatchIndexed);
  }
  REQUIRE_FALSE(fresh.empty());
  REQUIRE_FALSE(after_triangle.empty());
  bool identical = fresh.size() == after_triangle.size() &&
                   std::memcmp(fresh.data(), after_triangle.data(),
                               fresh.size()) == 0;
  std::printf(
      "TESSRENDER|TRANSLATOR|fresh=%zuB after_tri=%zuB identical=%d\n",
      fresh.size(), after_triangle.size(), int(identical));
  CHECK(identical);
  std::fflush(stdout);
}

// Same purity question one artifact level down: the shared production
// DxbcToDxilConverter and MetalShaderConverter instances, quad artifacts
// converted fresh vs after the corresponding triangle artifacts.

// RESOLVED 2026-08-02 (was KNOWN-ISSUE [!mayfail]): the order dependence was
// dxilconv per-instance call-history corruption — a reused IDxbcConverter
// carries state from the previous conversion into the next DXIL module.
// Fixed by creating a fresh converter per Convert
// (dxbc_to_dxil_converter.cc); conversions are now byte-identical regardless
// of order, in default and rand suite orderings.
TEST_CASE("dxilconv and MSC conversions are order-independent",
          "[tessellation][render][converterpurity]") {
  // Translate both domain payloads once (pure per the previous case).
  DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                  true, false, true);
  auto tri_ucode = BuildDomainPayloadUcode(kSrcSwizzleIdentity);
  MetalShader tri_shader(xenos::ShaderType::kVertex, 0x6101610161016101ull,
                         tri_ucode.data(), tri_ucode.size(),
                         std::endian::little);
  auto quad_ucode = BuildDomainPayloadUcode(kSrcSwizzleYZ);
  MetalShader quad_shader(xenos::ShaderType::kVertex, 0x6102610261026102ull,
                          quad_ucode.data(), quad_ucode.size(),
                          std::endian::little);
  StringBuffer disasm;
  tri_shader.AnalyzeUcode(disasm);
  quad_shader.AnalyzeUcode(disasm);
  std::vector<uint8_t> tri_dxbc = TranslateDomainPayload(
      translator, tri_shader, HostVertexShaderType::kTriangleDomainPatchIndexed);
  std::vector<uint8_t> quad_dxbc = TranslateDomainPayload(
      translator, quad_shader, HostVertexShaderType::kQuadDomainPatchIndexed);
  REQUIRE_FALSE(tri_dxbc.empty());
  REQUIRE_FALSE(quad_dxbc.empty());

  // DXIL level: shared DxbcToDxilConverter, quad-only vs tri-then-quad.
  std::string error;
  std::vector<uint8_t> quad_dxil_fresh;
  {
    metal::DxbcToDxilConverter converter;
    REQUIRE(converter.Initialize(&error));
    REQUIRE(converter.Convert(quad_dxbc, quad_dxil_fresh, &error));
  }
  std::vector<uint8_t> tri_dxil;
  std::vector<uint8_t> quad_dxil_after;
  {
    metal::DxbcToDxilConverter converter;
    REQUIRE(converter.Initialize(&error));
    REQUIRE(converter.Convert(tri_dxbc, tri_dxil, &error));
    REQUIRE(converter.Convert(quad_dxbc, quad_dxil_after, &error));
  }
  // Discriminator: second-call artifact after ANOTHER QUAD vs after the TRI.
  // If second-after-quad == second-after-tri, the fresh/second difference is
  // call-index noise (embedded UUID/hash); if they differ, the conversion is
  // CONTENT-dependent on the previous input — a real leak.
  std::vector<uint8_t> quad_dxil_after_quad;
  {
    metal::DxbcToDxilConverter converter;
    REQUIRE(converter.Initialize(&error));
    std::vector<uint8_t> first;
    REQUIRE(converter.Convert(quad_dxbc, first, &error));
    REQUIRE(converter.Convert(quad_dxbc, quad_dxil_after_quad, &error));
  }
  auto diff_stats = [](const std::vector<uint8_t>& a,
                       const std::vector<uint8_t>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t first_diff = n;
    size_t diff_bytes = 0;
    for (size_t i = 0; i < n; ++i) {
      if (a[i] != b[i]) {
        if (first_diff == n) first_diff = i;
        ++diff_bytes;
      }
    }
    return std::pair<size_t, size_t>(first_diff, diff_bytes);
  };
  bool dxil_identical =
      quad_dxil_fresh.size() == quad_dxil_after.size() &&
      std::memcmp(quad_dxil_fresh.data(), quad_dxil_after.data(),
                  quad_dxil_fresh.size()) == 0;
  bool second_calls_identical =
      quad_dxil_after_quad.size() == quad_dxil_after.size() &&
      std::memcmp(quad_dxil_after_quad.data(), quad_dxil_after.data(),
                  quad_dxil_after_quad.size()) == 0;
  auto stats = diff_stats(quad_dxil_fresh, quad_dxil_after);
  std::printf(
      "TESSRENDER|CONVERTER|dxil fresh=%zuB after_tri=%zuB identical=%d "
      "first_diff@%zu diff_bytes=%zu second_after_quad_vs_after_tri_same=%d\n",
      quad_dxil_fresh.size(), quad_dxil_after.size(), int(dxil_identical),
      stats.first, stats.second, int(second_calls_identical));
  // Thread-isolation probe (historical mechanism confirmation): pre-fix, the
  // reused object was a per-thread IDxbcConverter shared across
  // DxbcToDxilConverter instances, so a fresh thread converting quad alone
  // reproduced the fresh artifact while tri-then-quad reproduced the
  // contaminated one. Post-fix (fresh converter per Convert) both threads
  // must produce the identical artifact; the probe stays as a regression
  // tripwire for any reintroduction of converter reuse.
  std::vector<uint8_t> quad_dxil_fresh_thread;
  std::vector<uint8_t> quad_dxil_after_tri_thread;
  {
    std::thread t([&] {
      metal::DxbcToDxilConverter converter;
      std::string thread_error;
      if (converter.Initialize(&thread_error)) {
        converter.Convert(quad_dxbc, quad_dxil_fresh_thread, &thread_error);
      }
    });
    t.join();
    std::thread t2([&] {
      metal::DxbcToDxilConverter converter;
      std::string thread_error;
      std::vector<uint8_t> tri_scratch;
      if (converter.Initialize(&thread_error)) {
        converter.Convert(tri_dxbc, tri_scratch, &thread_error);
        converter.Convert(quad_dxbc, quad_dxil_after_tri_thread,
                          &thread_error);
      }
    });
    t2.join();
  }
  bool fresh_thread_matches_fresh =
      quad_dxil_fresh_thread == quad_dxil_fresh;
  bool tri_thread_matches_after = quad_dxil_after_tri_thread == quad_dxil_after;
  std::printf(
      "TESSRENDER|CONVERTER|thread-isolation fresh_thread==fresh=%d "
      "tri_thread==after_tri=%d (per-thread call HISTORY is the variable)\n",
      int(fresh_thread_matches_fresh), int(tri_thread_matches_after));
  CHECK(fresh_thread_matches_fresh);

  // PRODUCTION FINDING (resolved 2026-08-02): a reused dxilconv
  // IDxbcConverter carried state from the PREVIOUS conversion into the next
  // DXIL module (measured bitcode delta here). Render-level consequence: an
  // adaptive-QUAD tessellation pipeline whose domain-stage DXIL was produced
  // after an adaptive-TRIANGLE conversion on the same instance rendered
  // NOTHING (or a DIFFERENT tessellation pattern than the clean artifact).
  // DxbcToDxilConverter::Convert now creates a fresh converter per call, so
  // repeated conversions must be byte-identical.
  CHECK(second_calls_identical);
  if (const char* dump_dir = std::getenv("XE_TESS_DXIL_DUMP")) {
    auto dump = [&](const char* name, const std::vector<uint8_t>& bytes) {
      std::string path = std::string(dump_dir) + "/" + name;
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f) {
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
      }
    };
    dump("quad_fresh.dxil", quad_dxil_fresh);
    dump("quad_after_tri.dxil", quad_dxil_after);
    dump("quad_after_quad.dxil", quad_dxil_after_quad);
    dump("quad.dxbc", quad_dxbc);
  }

  // metallib level: shared MetalShaderConverter (kDomain geometry-emulation
  // conversion), quad-only vs tri-then-quad.
  using Stage = metal::MetalShaderConverter::Stage;
  using InputTopology = metal::MetalShaderConverter::InputTopology;
  metal::MetalShaderConverter::GeometryEmulationResult quad_fresh_r;
  {
    metal::MetalShaderConverter msc;
    REQUIRE(msc.Initialize(/*bindless_resources_used=*/true, &error));
    REQUIRE(msc.ConvertForGeometryEmulation(Stage::kDomain, quad_dxil_fresh,
                                            InputTopology::kUndefined,
                                            quad_fresh_r));
  }
  metal::MetalShaderConverter::GeometryEmulationResult quad_after_r;
  {
    metal::MetalShaderConverter msc;
    REQUIRE(msc.Initialize(/*bindless_resources_used=*/true, &error));
    metal::MetalShaderConverter::GeometryEmulationResult tri_r;
    REQUIRE(msc.ConvertForGeometryEmulation(Stage::kDomain, tri_dxil,
                                            InputTopology::kUndefined, tri_r));
    REQUIRE(msc.ConvertForGeometryEmulation(Stage::kDomain, quad_dxil_after,
                                            InputTopology::kUndefined,
                                            quad_after_r));
  }
  bool metallib_identical =
      quad_fresh_r.metallib.size() == quad_after_r.metallib.size() &&
      std::memcmp(quad_fresh_r.metallib.data(), quad_after_r.metallib.data(),
                  quad_fresh_r.metallib.size()) == 0;
  std::printf(
      "TESSRENDER|CONVERTER|metallib fresh=%zuB after_tri=%zuB identical=%d "
      "ds_domain=%u/%u gs_max=%u/%u\n",
      quad_fresh_r.metallib.size(), quad_after_r.metallib.size(),
      int(metallib_identical), quad_fresh_r.ds_tessellator_domain,
      quad_after_r.ds_tessellator_domain,
      quad_fresh_r.ds_max_input_prims_per_mesh_threadgroup,
      quad_after_r.ds_max_input_prims_per_mesh_threadgroup);
  CHECK(metallib_identical);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 2c. Same-spec repeatability: the identical adaptive wire render three times
// in a row. The suite's equality oracles (clamp/endian arms) compare renders
// within one process ordering; this pins that the comparison basis itself is
// stable.

TEST_CASE("identical tessellated wire renders are repeatable",
          "[tessellation][render][determinism][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  XE_TESS_BOTH_ROUTES(h);
  RenderResult runs[3];
  for (int i = 0; i < 3; ++i) {
    runs[i] = RenderAdaptiveQuadWire(h, 2.0f, 8.0f, FloatBits(7.0f),
                                     xenos::Endian::kNone);
    INFO("run " << i << ": " << runs[i].error);
    CHECK(runs[i].ok);
  }
  std::printf(
      "TESSRENDER|DETERMINISM|counts=%u,%u,%u same01=%d same12=%d "
      "cfg={vs_out=%u gs_max=%u hs_patches=%u hs_cp=%u hs_threads=%u "
      "factor=%.1f}\n",
      runs[0].covered, runs[1].covered, runs[2].covered,
      int(SameImage(runs[0], runs[1])), int(SameImage(runs[1], runs[2])),
      runs[0].cfg_vs_output, runs[0].cfg_gs_max_input, runs[0].cfg_hs_patches,
      runs[0].cfg_hs_cp, runs[0].cfg_hs_threads, runs[0].cfg_max_factor);
  CHECK(runs[0].covered == runs[1].covered);
  CHECK(runs[1].covered == runs[2].covered);
  CHECK(SameImage(runs[0], runs[1]));
  CHECK(SameImage(runs[1], runs[2]));
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 2d. Stage-memoization context probe: building the adaptive TRIANGLE
// pipeline first (as any real title would) and THEN rendering the adaptive
// quad wire. Compared against the determinism case's fresh-quad-first count,
// this isolates whether the shared kAdaptive stage memo makes the quad
// pipeline's OUTPUT depend on which cell was built first.

TEST_CASE("adaptive quad wire after adaptive triangle build (stage context)",
          "[tessellation][render][stagectx][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  // Sequence driven by XE_TESS_SEQ (comma list of trifill/trwire/quadfill/
  // quadwire); default = the original tri-fill-then-quad-wire probe. Each
  // fresh process = one sequence, so cross-arm contamination is observable.
  const char* seq_env = std::getenv("XE_TESS_SEQ");
  std::string seq = seq_env ? seq_env : "trifill,quadwire";
  auto run_step = [&](const std::string& step) -> RenderResult {
    if (step == "trifill" || step == "triwire") {
      RenderSpec tri;
      tri.mode = xenos::TessellationMode::kAdaptive;
      tri.host_vs_type = HostVertexShaderType::kTriangleDomainPatchIndexed;
      tri.host_prim = xenos::PrimitiveType::kTrianglePatch;
      tri.wireframe = step == "triwire";
      tri.range_min = 1.0f;
      tri.range_max = 8.0f;
      tri.draws.push_back(FactorStreamDraw(3, 3.0f));
      return h.Render(tri);
    }
    if (step == "quadfill") {
      RenderSpec quad;
      quad.mode = xenos::TessellationMode::kAdaptive;
      quad.host_vs_type = HostVertexShaderType::kQuadDomainPatchIndexed;
      quad.host_prim = xenos::PrimitiveType::kQuadPatch;
      quad.range_min = 1.0f;
      quad.range_max = 8.0f;
      quad.draws.push_back(FactorStreamDraw(4, 3.0f));
      return h.Render(quad);
    }
    // quadwire.
    return RenderAdaptiveQuadWire(h, 2.0f, 8.0f, FloatBits(7.0f),
                                  xenos::Endian::kNone);
  };
  size_t start = 0;
  while (start <= seq.size()) {
    size_t comma = seq.find(',', start);
    std::string step = seq.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!step.empty()) {
      RenderResult r = run_step(step);
      std::printf("TESSRENDER|STAGECTX|%s|%s|covered=%u depth_covered=%u%s%s\n",
                  step.c_str(), r.ok ? "OK" : "FAIL", r.covered,
                  r.depth_covered, r.error.empty() ? "" : " err=",
                  r.error.c_str());
      CHECK(r.ok);
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  SUCCEED("sequence recorded");
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 3. Two tessellated draws in one submission keep their own factors (spec:
// "Tess-factor buffers are safe across multiple tessellated draws").

// RESOLVED 2026-08-02 (was KNOWN-ISSUE [!mayfail]): the earlier "cross-case
// backend state" attribution was wrong — the full-suite disturbance was
// converter CONVERSION ORDER: dxilconv per-instance call-history corruption
// polluted the DXIL of pipelines built after other cases' conversions.
// Fixed by fresh-per-convert (dxbc_to_dxil_converter.cc); passes in default
// and rand suite orderings.
TEST_CASE("two tessellated draws in one submission render their own factors",
          "[tessellation][render][isolation][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  XE_TESS_BOTH_ROUTES(h);

  auto make_draw = [](float factor_value, double vp_x) {
    DrawSpec draw;
    draw.indexed = true;
    draw.words32.assign(4, FloatBits(factor_value));
    draw.count = 4;
    draw.has_viewport = true;
    draw.vp_x = vp_x;
    draw.vp_y = 0;
    draw.vp_w = kHalf;
    draw.vp_h = kRT;
    return draw;
  };
  auto base_spec = [] {
    RenderSpec spec;
    spec.mode = xenos::TessellationMode::kAdaptive;
    spec.host_vs_type = HostVertexShaderType::kQuadDomainPatchIndexed;
    spec.host_prim = xenos::PrimitiveType::kQuadPatch;
    spec.wireframe = true;
    spec.range_min = 2.0f;
    spec.range_max = 8.0f;
    return spec;
  };

  // One submission, two draws: factor 2 in the left viewport, factor 8 in the
  // right one.
  RenderSpec both = base_spec();
  both.draws.push_back(make_draw(1.0f, 0));
  both.draws.push_back(make_draw(7.0f, kHalf));
  RenderResult combined = h.Render(both);

  // Single-draw references in the same viewports.
  RenderSpec left_only = base_spec();
  left_only.draws.push_back(make_draw(1.0f, 0));
  RenderResult ref_left = h.Render(left_only);
  RenderSpec right_only = base_spec();
  right_only.draws.push_back(make_draw(7.0f, kHalf));
  RenderResult ref_right = h.Render(right_only);

  INFO("combined: " << combined.error << " refL: " << ref_left.error
                    << " refR: " << ref_right.error);
  REQUIRE(combined.ok);
  REQUIRE(ref_left.ok);
  REQUIRE(ref_right.ok);

  auto half_count = [](const RenderResult& r, bool right) {
    uint32_t count = 0;
    for (uint32_t y = 0; y < kRT; ++y) {
      for (uint32_t x = right ? kHalf : 0; x < (right ? kRT : kHalf); ++x) {
        if (r.CoveredAt(x, y)) ++count;
      }
    }
    return count;
  };
  auto half_equal = [](const RenderResult& a, const RenderResult& b,
                       bool right) {
    for (uint32_t y = 0; y < kRT; ++y) {
      const uint8_t* pa = a.rgba.data() + (y * kRT + (right ? kHalf : 0)) * 4;
      const uint8_t* pb = b.rgba.data() + (y * kRT + (right ? kHalf : 0)) * 4;
      if (std::memcmp(pa, pb, kHalf * 4) != 0) return false;
    }
    return true;
  };

  // Diagnostic arm: the same two draws with NO viewport changes (both cover
  // the same full-RT quadrant) — separates "the second tessellated dispatch
  // in an encoder is dropped" from "viewport state interaction".
  {
    RenderSpec overlap = base_spec();
    DrawSpec d1;
    d1.indexed = true;
    d1.words32.assign(4, FloatBits(1.0f));
    d1.count = 4;
    DrawSpec d2 = d1;
    d2.words32.assign(4, FloatBits(7.0f));
    overlap.draws.push_back(d1);
    overlap.draws.push_back(d2);
    RenderResult ro = h.Render(overlap);
    std::printf(
        "TESSRENDER|ISOLATION|overlap_two_draws covered=%u (f2 alone=558, f8 "
        "alone=1212, union>=1212 expected)\n",
        ro.covered);
  }

  uint32_t left = half_count(combined, false);
  uint32_t right = half_count(combined, true);
  std::printf(
      "TESSRENDER|ISOLATION|left=%u right=%u refL=%u refR=%u "
      "leftmatch=%d rightmatch=%d\n",
      left, right, half_count(ref_left, false), half_count(ref_right, true),
      int(half_equal(combined, ref_left, false)),
      int(half_equal(combined, ref_right, true)));
  CHECK(left > 0);
  CHECK(right > 0);
  // Different factors produce visibly different halves...
  CHECK(left != right);
  // ...and each half is EXACTLY the single-draw render: neither draw saw the
  // other's factor stream.
  CHECK(half_equal(combined, ref_left, false));
  CHECK(half_equal(combined, ref_right, true));
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 4. Fail-closed family.

TEST_CASE("depth-only tessellated draw: fallback PS engages and depth lands",
          "[tessellation][render][depthonly][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  XE_TESS_BOTH_ROUTES(h);

  RenderSpec spec;
  spec.mode = xenos::TessellationMode::kDiscrete;
  spec.host_vs_type = HostVertexShaderType::kQuadDomainCPIndexed;
  spec.host_prim = xenos::PrimitiveType::kQuadList;
  spec.use_guest_ps = false;  // NULL guest PS -> production depth-only fallback.
  spec.depth_only = true;     // No color attachments at all.
  spec.range_min = 1.0f;
  spec.range_max = 4.0f;
  spec.draws.push_back(NonIndexedDraw(4));
  RenderResult r = h.Render(spec);
  INFO(r.error);
  REQUIRE(r.ok);  // Pipeline non-null proves the fallback PS engaged.

  uint32_t outside = CountOutsideQuadrant(r);
  float min_depth = 1.0f;
  for (float d : r.depth) {
    min_depth = std::min(min_depth, d);
  }
  bool coverage_ok = r.covered >= (kQuarterArea * 7) / 10;
  std::printf(
      "TESSRENDER|DEPTHONLY|%s|covered=%u expected>=%u outside=%u "
      "min_depth=%f\n",
      (coverage_ok && outside == 0) ? "PASS" : "FAIL", r.covered,
      (kQuarterArea * 7) / 10, outside, min_depth);
  CHECK(r.covered > 0);
  CHECK(coverage_ok);
  CHECK(outside == 0);
  // The payload exports z=0 (const): written depth must be ~0 vs clear 1.
  CHECK(min_depth < 0.5f);
  std::fflush(stdout);
}

TEST_CASE("memexport predicate input: eM-writing guest VS is detected",
          "[tessellation][render][memexport]") {
  // The IssueDraw tessellation route fail-closed skips ("tess-memexport",
  // MetalDrawSkipReason::kUnsupportedCapability) on
  // vertex_shader->memexport_eM_written() BEFORE any pipeline work — the
  // route itself is CP-embedded (live gate 5.3 exercises it); this pins its
  // exact predicate input on both sides.
  //
  // (The sibling "tess-no-mesh-shaders" skip keys on the device capability;
  // it cannot be synthesized on hardware that HAS mesh shaders — every
  // render case above REQUIREs mesh support implicitly by building mesh
  // pipelines.)
  {
    std::vector<uint32_t> ucode = BuildMemexportVertexUcode();
    MetalShader shader(xenos::ShaderType::kVertex, 0x5544554455445544ull,
                       ucode.data(), ucode.size(), std::endian::little);
    StringBuffer disasm;
    shader.AnalyzeUcode(disasm);
    std::string text = disasm.to_string();
    INFO(text);
    CHECK(text.find("eM0") != std::string::npos);
    CHECK(shader.memexport_eM_written() != 0);
  }
  // The render payload (every cell above) must NOT trip the same predicate.
  {
    std::vector<uint32_t> ucode = BuildDomainPayloadUcode(kSrcSwizzleIdentity);
    MetalShader shader(xenos::ShaderType::kVertex, 0x5555555555555555ull,
                       ucode.data(), ucode.size(), std::endian::little);
    StringBuffer disasm;
    shader.AnalyzeUcode(disasm);
    CHECK(shader.memexport_eM_written() == 0);
  }
}

// ---------------------------------------------------------------------------
// 5. Adaptive-dispatch nondeterminism campaign (hidden diagnostics — run
// explicitly with the [nondeterm]/[nondeterm2] tags; they never gate the
// suite). Sweeps the config delta between the KNOWN-STABLE repeatability case
// (2c: adaptive quad WIRE, color RGBA8 1x, D32F, 1 patch, factor word 7.0,
// range [2,8], one draw per encoder) and the VP frame-3454 first-diverging
// draw shape measured live by MetalTessDispatchCfg (draw 1096: 2xMSAA,
// D32F_S8, fill, 256 patches = 8 FULL 32-patch object threadgroups, factor
// words 0x00000000 clamping to range min 2, guest PS + color mask F, 13
// adaptive quad draws in the encoder).

namespace {

uint64_t Fnv1a64(const uint8_t* data, size_t size) {
  uint64_t hash = 14695981039346656037ull;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ data[i]) * 1099511628211ull;
  }
  return hash;
}

DrawSpec PatchDraw(uint32_t patches, float factor_value) {
  DrawSpec draw;
  draw.indexed = true;
  draw.words32.assign(patches * 4, FloatBits(factor_value));
  draw.count = patches * 4;
  return draw;
}

RenderSpec NondetBaseSpec() {
  RenderSpec spec;
  spec.mode = xenos::TessellationMode::kAdaptive;
  spec.host_vs_type = HostVertexShaderType::kQuadDomainPatchIndexed;
  spec.host_prim = xenos::PrimitiveType::kQuadPatch;
  spec.wireframe = true;
  spec.range_min = 2.0f;
  spec.range_max = 8.0f;
  spec.draws.push_back(PatchDraw(1, 7.0f));
  return spec;
}

// The measured VP draw-1096 shape (minus the guest content, which the input
// hash already proved byte-stable): 13 draws x 256 zero-word patches, fill,
// 2xMSAA, production D32F_S8, guest PS + color.
RenderSpec NondetVpShapeSpec() {
  RenderSpec spec = NondetBaseSpec();
  spec.wireframe = false;
  spec.sample_count = 2;
  spec.depth_stencil8 = true;
  spec.draws.clear();
  for (int i = 0; i < 13; ++i) {
    spec.draws.push_back(PatchDraw(256, 0.0f));
  }
  return spec;
}

int NondetIters() {
  const char* value = std::getenv("XE_TESS_NONDET_ITERS");
  int iters = value ? std::atoi(value) : 5;
  return iters > 1 ? iters : 5;
}

// Renders the spec `iters` times and byte-compares depth + color across runs.
void RunNondetArm(RenderHarness& h, const char* name, const RenderSpec& spec,
                  int iters) {
  std::vector<RenderResult> runs;
  runs.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    runs.push_back(h.Render(spec));
    if (!runs.back().ok) {
      std::printf("TESSNONDET|%s|ERROR|run=%d err=%s\n", name, i,
                  runs.back().error.c_str());
      CHECK(runs.back().ok);
      return;
    }
  }
  // Distinct-image census.
  std::vector<uint64_t> hashes;
  for (const RenderResult& r : runs) {
    uint64_t hash = Fnv1a64(
        reinterpret_cast<const uint8_t*>(r.depth.data()),
        r.depth.size() * sizeof(float));
    if (!r.rgba.empty()) {
      hash ^= Fnv1a64(r.rgba.data(), r.rgba.size()) * 1099511628211ull;
    }
    hashes.push_back(hash);
  }
  int distinct = 0;
  for (size_t i = 0; i < hashes.size(); ++i) {
    bool seen = false;
    for (size_t j = 0; j < i; ++j) {
      if (hashes[j] == hashes[i]) {
        seen = true;
        break;
      }
    }
    if (!seen) ++distinct;
  }
  bool stable = distinct == 1;
  // Divergence anatomy vs run 0 (first diverging run): per-sample-plane depth
  // diff counts + color byte diffs.
  uint32_t depth_diffs_plane[4] = {};
  uint32_t color_diffs = 0;
  int first_diverging = -1;
  if (!stable) {
    for (int i = 1; i < iters; ++i) {
      if (hashes[i] != hashes[0]) {
        first_diverging = i;
        const RenderResult& a = runs[0];
        const RenderResult& b = runs[i];
        uint32_t planes = uint32_t(a.depth.size() / (kRT * kRT));
        for (uint32_t p = 0; p < planes && p < 4; ++p) {
          for (uint32_t k = 0; k < kRT * kRT; ++k) {
            if (a.depth[p * kRT * kRT + k] != b.depth[p * kRT * kRT + k]) {
              ++depth_diffs_plane[p];
            }
          }
        }
        for (size_t k = 0; k < a.rgba.size(); ++k) {
          if (a.rgba[k] != b.rgba[k]) ++color_diffs;
        }
        break;
      }
    }
  }
  std::printf(
      "TESSNONDET|%s|%s|iters=%d distinct=%d covered0=%u depth_covered0=%u "
      "first_div=%d depth_diffs=[%u,%u,%u,%u] color_diffs=%u "
      "cfg(vs_out=%u gs_max=%u hs_patches=%u hs_cp=%u hs_threads=%u "
      "factor=%.0f)\n",
      name, stable ? "STABLE" : "DIVERGES", iters, distinct, runs[0].covered,
      runs[0].depth_covered, first_diverging, depth_diffs_plane[0],
      depth_diffs_plane[1], depth_diffs_plane[2], depth_diffs_plane[3],
      color_diffs, runs[0].cfg_vs_output, runs[0].cfg_gs_max_input,
      runs[0].cfg_hs_patches, runs[0].cfg_hs_cp, runs[0].cfg_hs_threads,
      runs[0].cfg_max_factor);
  std::fflush(stdout);
}

}  // namespace

TEST_CASE("adaptive dispatch nondeterminism config sweep",
          "[.][nondeterm][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  int iters = NondetIters();

  struct Arm {
    const char* name;
    RenderSpec spec;
  };
  std::vector<Arm> arms;
  arms.push_back({"control", NondetBaseSpec()});
  {
    RenderSpec s = NondetBaseSpec();
    s.wireframe = false;
    arms.push_back({"fill", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.draws[0] = PatchDraw(1, 0.0f);  // words 0x0 -> clamp to range min 2.
    arms.push_back({"words0", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.draws[0] = PatchDraw(256, 7.0f);  // 8 FULL 32-patch threadgroups.
    arms.push_back({"patches256", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.draws[0] = PatchDraw(60, 7.0f);  // 1 full + 28-of-32 tail threadgroup.
    arms.push_back({"tail28of32", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.sample_count = 2;
    arms.push_back({"msaa2", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.depth_stencil8 = true;
    arms.push_back({"ds8", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.depth_compare_lequal = true;
    arms.push_back({"lequal", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    s.depth_only = true;
    s.use_guest_ps = false;
    arms.push_back({"depthonly", s});
  }
  {
    RenderSpec s = NondetBaseSpec();
    for (int i = 1; i < 13; ++i) {
      s.draws.push_back(s.draws[0]);
    }
    arms.push_back({"multidraw13", s});
  }
  arms.push_back({"vp-shape", NondetVpShapeSpec()});
  {
    RenderSpec s = NondetVpShapeSpec();
    s.depth_only = true;
    s.use_guest_ps = false;
    arms.push_back({"vp-depthonly", s});
  }

  for (const Arm& arm : arms) {
    RunNondetArm(h, arm.name, arm.spec, iters);
  }
  SUCCEED("sweep recorded");
}

// Phase 2: dispatch-parameter bisect on a reproducing config
// (XE_TESS_NONDET2_BASE=vp-shape|<sweep arm name>; default vp-shape). Local
// dispatch replica varies the knobs the vendored helper hardcodes.
TEST_CASE("adaptive dispatch nondeterminism knob bisect",
          "[.][nondeterm2][device]") {
  auto& h = H();
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  int iters = NondetIters();
  const char* base_name = std::getenv("XE_TESS_NONDET2_BASE");
  std::string base = base_name ? base_name : "vp-shape";
  RenderSpec spec = base == "vp-shape" ? NondetVpShapeSpec() : NondetBaseSpec();
  if (base == "msaa2") spec.sample_count = 2;
  if (base == "patches256") spec.draws[0] = PatchDraw(256, 7.0f);
  if (base == "multidraw13") {
    for (int i = 1; i < 13; ++i) spec.draws.push_back(spec.draws[0]);
  }

  struct Knob {
    const char* name;
    uint32_t tg_mem;
    uint32_t hs_patches;
    uint32_t pad_words;
  };
  const Knob knobs[] = {
      {"vendored", 0, 0, 0},        // control: the vendored dispatch helper.
      {"local15360", 15360, 0, 0},  // local replica parity control.
      {"tg64", 64, 0, 0},   // DELIBERATE undersize: the shader-validation
                            // engagement probe (MTL_SHADER_VALIDATION=1 must
                            // flag threadgroup-memory OOB if it instruments
                            // object/mesh pipelines at all).
      {"tg16384", 16384, 0, 0},     // larger object threadgroup memory.
      {"tg23040", 23040, 0, 0},     // 1.5x.
      {"hs16", 15360, 16, 0},       // fewer patches per object threadgroup.
      {"hs8", 15360, 8, 0},
      {"hs1", 15360, 1, 0},         // one patch per threadgroup.
      {"pad32", 15360, 0, 32},      // nonzero startIndex (arena-offset model).
  };
  for (const Knob& knob : knobs) {
    RenderSpec s = spec;
    s.tg_mem_length = knob.tg_mem;
    s.hs_patches_override = knob.hs_patches;
    s.index_pad_words = knob.pad_words;
    std::string label = base + "+" + knob.name;
    RunNondetArm(h, label.c_str(), s, iters);
  }
  SUCCEED("bisect recorded");
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
