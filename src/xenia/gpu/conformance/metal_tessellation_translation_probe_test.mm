/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// port-xenios-tessellation task 2.2: the standalone MSC tessellation probe.
//
// Proves, on-device and BEFORE any command-processor wiring, that the whole
// tessellation shader family survives our production translation chain
// (DXBC -> dxilconv DXIL -> Metal Shader Converter -> metallib -> MTLLibrary):
//
//   1. All 10 prebuilt SM5.1 hull-shader blobs
//      (src/xenia/gpu/metal/d3d12_5_1_bytecode/*_hs.h) compile as Stage::kHull
//      with complete hull reflection (domain per blob name, control-point
//      count, max tessellation factor) and their runtime-consumed functions
//      ("irconverter_hull_shader" + "irconverter_tessellator") instantiate on
//      the device with the function constants the IR runtime sets.
//   2. Both prebuilt tessellation VS blobs compile as Stage::kVertex with the
//      zero-element stage-in synthesized (an empty stage-in is a hard pipeline
//      failure — port notes section 2/gotcha 9) and their object-stage
//      function instantiates with tessellationEnabled=true.
//   3. A synthetic guest vertex program (the clip-discriminator test's
//      nondegenerate-position fixture) translates through DxbcShaderTranslator
//      as each of the four supported domain-shader types at user clip plane
//      counts 0/1/2 (12 cells) and compiles as Stage::kDomain with domain
//      reflection matching the host type.
//   4. Every supported (mode, domain, cp-count) HS pairs with its matching
//      guest-DS payload under IRRuntimeValidateTessellationPipeline (argument
//      order per port notes section 3 step 5).
//   5. STRETCH (recorded, never asserted): one full
//      IRRuntimeNewGeometryTessellationEmulationPipeline build for a
//      triangle-domain cell and a quad-domain cell, config per port notes
//      section 3 step 7 (gsMaxInputPrimitivesPerMeshThreadgroup from the
//      DOMAIN reflection — gotcha 8).
//
// KILL-GATE semantics (reported by the TESSPROBE summary lines, decided by the
// change owner, not this test): any mode/domain unloadable at clip 0 is the
// scope-reduction trigger; failures only at clip>0 make behavioral closure of
// fix-metal-geometry-clip-distance-bridge a hard precondition.
//
// Every cell is CHECKed (not REQUIREd) so a single failure still yields the
// complete blob x result / cell x result matrix on stdout ("TESSPROBE|"
// lines).

#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

// IRRuntimeValidateTessellationPipeline / tessellation pipeline descriptor +
// runtime constants. The implementations live in the single
// IR_PRIVATE_IMPLEMENTATION TU (gpu/metal/ir_runtime_impl.mm, ObjC id<MTL...>
// types — IR_RUNTIME_METALCPP intentionally undefined there and here).
#include "metal_irconverter_runtime.h"

// Prebuilt SM5.1 tessellation bytecode (imported by task 1.2; see the README
// in d3d12_5_1_bytecode/ for provenance + regeneration recipe).
using BYTE = uint8_t;
#include "xenia/gpu/metal/d3d12_5_1_bytecode/adaptive_quad_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/adaptive_triangle_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_quad_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_quad_4cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_triangle_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_triangle_3cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_quad_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_quad_4cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_3cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/tessellation_adaptive_vs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/tessellation_indexed_vs.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

using HostVertexShaderType = Shader::HostVertexShaderType;
using GeometryEmulationResult =
    metal::MetalShaderConverter::GeometryEmulationResult;
using Stage = metal::MetalShaderConverter::Stage;
using InputTopology = metal::MetalShaderConverter::InputTopology;

// IRTessellatorDomain values (third_party/metal-shader-converter/include/
// metal_irconverter/ir_tessellator_domain.h) as stored in the uint32_t
// reflection fields.
constexpr uint32_t kIRDomainTri = 2;   // IRTessellatorDomainTri
constexpr uint32_t kIRDomainQuad = 3;  // IRTessellatorDomainQuad

// ---------------------------------------------------------------------------
// Fixtures.

struct HsBlobCase {
  const char* name;
  const uint8_t* bytes;
  size_t size;
  uint32_t expected_domain;  // kIRDomainTri / kIRDomainQuad.
  bool adaptive;             // pairs with tessellation_adaptive_vs.
};

const HsBlobCase kHsBlobs[] = {
    {"discrete_triangle_3cp_hs", ::discrete_triangle_3cp_hs,
     sizeof(::discrete_triangle_3cp_hs), kIRDomainTri, false},
    {"discrete_triangle_1cp_hs", ::discrete_triangle_1cp_hs,
     sizeof(::discrete_triangle_1cp_hs), kIRDomainTri, false},
    {"discrete_quad_4cp_hs", ::discrete_quad_4cp_hs,
     sizeof(::discrete_quad_4cp_hs), kIRDomainQuad, false},
    {"discrete_quad_1cp_hs", ::discrete_quad_1cp_hs,
     sizeof(::discrete_quad_1cp_hs), kIRDomainQuad, false},
    {"continuous_triangle_3cp_hs", ::continuous_triangle_3cp_hs,
     sizeof(::continuous_triangle_3cp_hs), kIRDomainTri, false},
    {"continuous_triangle_1cp_hs", ::continuous_triangle_1cp_hs,
     sizeof(::continuous_triangle_1cp_hs), kIRDomainTri, false},
    {"continuous_quad_4cp_hs", ::continuous_quad_4cp_hs,
     sizeof(::continuous_quad_4cp_hs), kIRDomainQuad, false},
    {"continuous_quad_1cp_hs", ::continuous_quad_1cp_hs,
     sizeof(::continuous_quad_1cp_hs), kIRDomainQuad, false},
    {"adaptive_triangle_hs", ::adaptive_triangle_hs,
     sizeof(::adaptive_triangle_hs), kIRDomainTri, true},
    {"adaptive_quad_hs", ::adaptive_quad_hs, sizeof(::adaptive_quad_hs),
     kIRDomainQuad, true},
};

struct TessVsBlobCase {
  const char* name;
  const uint8_t* bytes;
  size_t size;
};

const TessVsBlobCase kTessVsBlobs[] = {
    {"tessellation_indexed_vs", ::tessellation_indexed_vs,
     sizeof(::tessellation_indexed_vs)},
    {"tessellation_adaptive_vs", ::tessellation_adaptive_vs,
     sizeof(::tessellation_adaptive_vs)},
};

const HostVertexShaderType kDomainTypes[] = {
    HostVertexShaderType::kTriangleDomainCPIndexed,
    HostVertexShaderType::kTriangleDomainPatchIndexed,
    HostVertexShaderType::kQuadDomainCPIndexed,
    HostVertexShaderType::kQuadDomainPatchIndexed,
};

const char* DomainTypeName(HostVertexShaderType type) {
  switch (type) {
    case HostVertexShaderType::kTriangleDomainCPIndexed:
      return "kTriangleDomainCPIndexed";
    case HostVertexShaderType::kTriangleDomainPatchIndexed:
      return "kTriangleDomainPatchIndexed";
    case HostVertexShaderType::kQuadDomainCPIndexed:
      return "kQuadDomainCPIndexed";
    case HostVertexShaderType::kQuadDomainPatchIndexed:
      return "kQuadDomainPatchIndexed";
    default:
      return "?";
  }
}

uint32_t ExpectedDomainFor(HostVertexShaderType type) {
  switch (type) {
    case HostVertexShaderType::kTriangleDomainCPIndexed:
    case HostVertexShaderType::kTriangleDomainPatchIndexed:
      return kIRDomainTri;
    default:
      return kIRDomainQuad;
  }
}

// The 10 supported (mode, host-VS-type) HS selections (port notes section 1
// table; XeniOS metal_pipeline_cache.cc:856-930). hs_name indexes kHsBlobs.
struct PairCase {
  const char* mode;
  const char* hs_name;
  HostVertexShaderType ds_type;
};

const PairCase kPairs[] = {
    {"discrete", "discrete_triangle_3cp_hs",
     HostVertexShaderType::kTriangleDomainCPIndexed},
    {"discrete", "discrete_triangle_1cp_hs",
     HostVertexShaderType::kTriangleDomainPatchIndexed},
    {"discrete", "discrete_quad_4cp_hs",
     HostVertexShaderType::kQuadDomainCPIndexed},
    {"discrete", "discrete_quad_1cp_hs",
     HostVertexShaderType::kQuadDomainPatchIndexed},
    {"continuous", "continuous_triangle_3cp_hs",
     HostVertexShaderType::kTriangleDomainCPIndexed},
    {"continuous", "continuous_triangle_1cp_hs",
     HostVertexShaderType::kTriangleDomainPatchIndexed},
    {"continuous", "continuous_quad_4cp_hs",
     HostVertexShaderType::kQuadDomainCPIndexed},
    {"continuous", "continuous_quad_1cp_hs",
     HostVertexShaderType::kQuadDomainPatchIndexed},
    {"adaptive", "adaptive_triangle_hs",
     HostVertexShaderType::kTriangleDomainPatchIndexed},
    {"adaptive", "adaptive_quad_hs",
     HostVertexShaderType::kQuadDomainPatchIndexed},
};

// Nondegenerate guest vertex program taken from the clip-discriminator test's
// coverage fixture (metal_geometry_clip_plane_discriminator_test.mm): computes
// y = x * x from the translated index input in r0.x and exports oPos.xyz1, so
// consecutive vertices are not collinear. Real microcode keeps
// DxbcShaderTranslator in the loop instead of pinning a generated artifact.
const uint32_t kGuestVertexUcode[] = {
    0x00001002, 0x00001000, 0xC2000000, 0x00001003,
    0x00002000, 0x00000000, 0xC8020000, 0x006C6C00,
    0xC1000000, 0xC88F803E, 0x00000000, 0xC2000000};

// Proven guest-PS fixture (metal_float24_translation_probe_test.mm), used only
// as the fragment stage of the stretch pipeline build.
std::vector<uint32_t> BuildFixturePixelUcode() {
  const uint32_t kExecAddress = 1;
  const uint32_t kExecCount = 1;
  const uint32_t kSequenceFetch = 0b01;
  uint32_t cf0_dword0 = (kExecAddress & 0xFFF) | ((kExecCount & 0x7) << 12) |
                        ((kSequenceFetch & 0xFFF) << 16);
  uint32_t cf0_dword1 =
      (uint32_t(ucode::ControlFlowOpcode::kExecEnd) & 0xF) << 12;
  const uint32_t kDstSwizzleIdentity =
      (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);
  uint32_t fetch_dword0 = (uint32_t(ucode::FetchOpcode::kTextureFetch) & 0x1F) |
                          ((0u & 0x3F) << 5) | ((1u & 0x3F) << 12) |
                          ((0u & 0x1F) << 20) | ((0b010001u & 0x3F) << 26);
  uint32_t fetch_dword1 = kDstSwizzleIdentity & 0xFFF;
  uint32_t fetch_dword2 = (uint32_t(xenos::FetchOpDimension::k2D) & 0x3) << 14;
  return {cf0_dword0, cf0_dword1, 0u, fetch_dword0, fetch_dword1, fetch_dword2};
}

// ---------------------------------------------------------------------------
// Harness: device + converters + memoized per-stage conversion results so the
// pairing/stretch cases reuse what the per-family cases already compiled.

struct StageResult {
  bool ok = false;
  std::string error;
  GeometryEmulationResult r;
};

struct Harness {
  id<MTLDevice> device = nil;
  metal::DxbcToDxilConverter dxil_converter;
  metal::MetalShaderConverter msc;
  bool ok = false;
  std::string init_error;

  std::map<std::string, StageResult> hull_results;
  std::map<std::string, StageResult> vs_results;
  // Key: "<domain-type-name>/clip<N>".
  std::map<std::string, StageResult> domain_results;

  Harness() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      init_error = "no Metal device";
      return;
    }
    if (!dxil_converter.Initialize(&init_error)) {
      return;
    }
    if (!msc.Initialize(/*bindless_resources_used=*/true, &init_error)) {
      return;
    }
    ok = true;
  }

  bool DxbcToDxil(const uint8_t* bytes, size_t size,
                  std::vector<uint8_t>& dxil_out, std::string& error) {
    std::vector<uint8_t> dxbc(bytes, bytes + size);
    return dxil_converter.Convert(dxbc, dxil_out, &error);
  }

  // Prebuilt HS blob -> DXIL -> MSC Stage::kHull. Topology stays undefined,
  // matching XeniOS's hull compile (metal_pipeline_cache.cc:940-943).
  const StageResult& GetHull(const HsBlobCase& blob) {
    auto it = hull_results.find(blob.name);
    if (it != hull_results.end()) {
      return it->second;
    }
    StageResult& out = hull_results[blob.name];
    std::vector<uint8_t> dxil;
    if (!DxbcToDxil(blob.bytes, blob.size, dxil, out.error)) {
      out.error = "dxilconv: " + out.error;
      return out;
    }
    if (!msc.ConvertForGeometryEmulation(Stage::kHull, dxil,
                                         InputTopology::kUndefined, out.r)) {
      out.error = "MSC: " + out.r.error;
      return out;
    }
    out.ok = true;
    return out;
  }

  // Prebuilt tessellation VS blob -> DXIL -> MSC Stage::kVertex with the patch
  // input topology (the tessellation object stage consumes patches). The
  // zero-element stage-in is synthesized inside ConvertForGeometryEmulation.
  const StageResult& GetTessVs(const TessVsBlobCase& blob) {
    auto it = vs_results.find(blob.name);
    if (it != vs_results.end()) {
      return it->second;
    }
    StageResult& out = vs_results[blob.name];
    std::vector<uint8_t> dxil;
    if (!DxbcToDxil(blob.bytes, blob.size, dxil, out.error)) {
      out.error = "dxilconv: " + out.error;
      return out;
    }
    if (!msc.ConvertForGeometryEmulation(Stage::kVertex, dxil,
                                         InputTopology::kPatch, out.r)) {
      out.error = "MSC: " + out.r.error;
      return out;
    }
    out.ok = true;
    return out;
  }

  // Guest VS translated as a domain shader (ds_5_1 profile is selected by the
  // translator from host_vertex_shader_type) -> DXIL -> MSC Stage::kDomain.
  const StageResult& GetDomain(HostVertexShaderType type, uint32_t clip_count) {
    std::string key =
        std::string(DomainTypeName(type)) + "/clip" + std::to_string(clip_count);
    auto it = domain_results.find(key);
    if (it != domain_results.end()) {
      return it->second;
    }
    StageResult& out = domain_results[key];

    Shader shader(xenos::ShaderType::kVertex, 0x7E55000000000001ull,
                  kGuestVertexUcode,
                  sizeof(kGuestVertexUcode) / sizeof(uint32_t),
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
    modification.vertex.user_clip_plane_count = clip_count;
    modification.vertex.user_clip_plane_cull = 0;
    Shader::Translation* translation =
        shader.GetOrCreateTranslation(modification.value);
    if (!translator.TranslateAnalyzedShader(*translation)) {
      out.error = "DxbcShaderTranslator failed";
      return out;
    }
    std::vector<uint8_t> dxbc = translation->translated_binary();

    std::vector<uint8_t> dxil;
    if (!DxbcToDxil(dxbc.data(), dxbc.size(), dxil, out.error)) {
      out.error = "dxilconv: " + out.error;
      return out;
    }
    if (!msc.ConvertForGeometryEmulation(Stage::kDomain, dxil,
                                         InputTopology::kUndefined, out.r)) {
      out.error = "MSC: " + out.r.error;
      return out;
    }
    out.ok = true;
    return out;
  }

  const HsBlobCase* FindHsBlob(const char* name) {
    for (const HsBlobCase& blob : kHsBlobs) {
      if (!std::strcmp(blob.name, name)) {
        return &blob;
      }
    }
    return nullptr;
  }
};

Harness& H() {
  static Harness harness;
  return harness;
}

// ---------------------------------------------------------------------------
// Device-side loading helpers, mirroring the exact function shapes
// IRRuntimeNewGeometryTessellationEmulationPipeline creates (runtime header,
// IR_PRIVATE_IMPLEMENTATION block).

id<MTLLibrary> NewLibrary(const std::vector<uint8_t>& metallib,
                          std::string& error) {
  if (metallib.size() < 4 || std::memcmp(metallib.data(), "MTLB", 4) != 0) {
    error = "not an MTLB container";
    return nil;
  }
  dispatch_data_t data =
      dispatch_data_create(metallib.data(), metallib.size(), nullptr,
                           DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NSError* ns_error = nil;
  id<MTLLibrary> library = [H().device newLibraryWithData:data error:&ns_error];
  if (!library) {
    error = ns_error ? ns_error.localizedDescription.UTF8String
                     : "newLibraryWithData failed";
  }
  return library;
}

id<MTLFunction> NewPlainFunction(id<MTLLibrary> library, NSString* name,
                                 std::string& error) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) {
    error = std::string("function '") + name.UTF8String + "' not loadable";
  }
  return function;
}

id<MTLFunction> NewFunctionWithConstants(id<MTLLibrary> library, NSString* name,
                                         MTLFunctionConstantValues* values,
                                         std::string& error) {
  NSError* ns_error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name
                                            constantValues:values
                                                     error:&ns_error];
  if (!function) {
    error = std::string("function '") + name.UTF8String + "': " +
            (ns_error ? ns_error.localizedDescription.UTF8String : "nil");
  }
  return function;
}

// Instantiates the two hull-library functions the IR runtime creates, with the
// function constants it sets (vertex_shader_output_size_fc from the tess VS
// reflection, max_tessellation_factor_fc from the HS reflection).
bool LoadHullFunctions(id<MTLLibrary> library, int vs_output_size,
                       float max_tessellation_factor, std::string& error) {
  MTLFunctionConstantValues* values = [[MTLFunctionConstantValues alloc] init];
  [values setConstantValue:&vs_output_size
                      type:MTLDataTypeInt
                  withName:@"vertex_shader_output_size_fc"];
  [values setConstantValue:&max_tessellation_factor
                      type:MTLDataTypeFloat
                  withName:@"max_tessellation_factor_fc"];
  if (!NewFunctionWithConstants(library, @"irconverter_hull_shader", values,
                                error)) {
    return false;
  }
  if (!NewFunctionWithConstants(library, @"irconverter_tessellator", values,
                                error)) {
    return false;
  }
  return true;
}

// Instantiates the tessellation object-stage (vertex) function the way the
// tessellation-emulation pipeline builder does: "<name>.
// dxil_irconverter_object_shader" with tessellationEnabled=true.
bool LoadTessObjectFunction(id<MTLLibrary> library,
                            const std::string& function_name,
                            std::string& error) {
  MTLFunctionConstantValues* values = [[MTLFunctionConstantValues alloc] init];
  bool tessellation_enabled = true;
  [values setConstantValue:&tessellation_enabled
                      type:MTLDataTypeBool
                  withName:@"tessellationEnabled"];
  NSString* name = [NSString
      stringWithUTF8String:(function_name + ".dxil_irconverter_object_shader")
                               .c_str()];
  return NewFunctionWithConstants(library, name, values, error) != nil;
}

// Maps the HS reflection's tessellator output primitive to the runtime's
// validation/geometry-primitive pair (XeniOS metal_pipeline_cache.cc:
// 2820-2859). Returns false on an unsupported output primitive.
bool MapOutputPrimitive(uint32_t hs_tessellator_output_primitive,
                        IRRuntimeTessellatorOutputPrimitive& output_primitive,
                        IRRuntimePrimitiveType& geometry_primitive,
                        const char*& geometry_function) {
  switch (hs_tessellator_output_primitive) {
    case IRRuntimeTessellatorOutputPoint:
      output_primitive = IRRuntimeTessellatorOutputPoint;
      geometry_primitive = IRRuntimePrimitiveTypePoint;
      geometry_function = kIRPointPassthroughGeometryShader;
      return true;
    case IRRuntimeTessellatorOutputLine:
      output_primitive = IRRuntimeTessellatorOutputLine;
      geometry_primitive = IRRuntimePrimitiveTypeLine;
      geometry_function = kIRLinePassthroughGeometryShader;
      return true;
    case IRRuntimeTessellatorOutputTriangleCW:
      output_primitive = IRRuntimeTessellatorOutputTriangleCW;
      geometry_primitive = IRRuntimePrimitiveTypeTriangle;
      geometry_function = kIRTrianglePassthroughGeometryShader;
      return true;
    case IRRuntimeTessellatorOutputTriangleCCW:
      output_primitive = IRRuntimeTessellatorOutputTriangleCCW;
      geometry_primitive = IRRuntimePrimitiveTypeTriangle;
      geometry_function = kIRTrianglePassthroughGeometryShader;
      return true;
    default:
      return false;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. All 10 prebuilt HS blobs.

TEST_CASE("tessellation HS blobs convert, reflect and load",
          "[tessellation][hull][device]") {
  auto& h = H();
  {
    INFO(h.init_error);
    REQUIRE(h.ok);
  }

  for (const HsBlobCase& blob : kHsBlobs) {
    INFO("HS blob " << blob.name);
    const StageResult& hs = h.GetHull(blob);
    if (!hs.ok) {
      std::printf("TESSPROBE|HS|%s|FAIL|%s\n", blob.name, hs.error.c_str());
      CHECK(hs.ok);
      continue;
    }
    const GeometryEmulationResult& r = hs.r;
    bool fields_ok = r.has_hull_info &&
                     r.hs_tessellator_domain == blob.expected_domain &&
                     r.hs_output_control_point_count > 0 &&
                     r.hs_max_tessellation_factor > 0.0f;
    CHECK(r.has_hull_info);
    CHECK(r.hs_tessellator_domain == blob.expected_domain);
    CHECK(r.hs_output_control_point_count > 0);
    CHECK(r.hs_max_tessellation_factor > 0.0f);

    std::string load_error;
    id<MTLLibrary> library = NewLibrary(r.metallib, load_error);
    bool functions_ok = false;
    if (library) {
      // The vertex-output-size constant's real value comes from the paired
      // tess VS reflection; use it so this probe exercises the exact function
      // specialization the pipeline builder will request.
      const StageResult& vs =
          h.GetTessVs(blob.adaptive ? kTessVsBlobs[1] : kTessVsBlobs[0]);
      int vs_output_size =
          vs.ok ? int(vs.r.vertex_output_size_in_bytes) : 64;
      functions_ok = LoadHullFunctions(library, vs_output_size,
                                       r.hs_max_tessellation_factor,
                                       load_error);
    }
    {
      INFO("hull load: " << load_error);
      CHECK(library != nil);
      CHECK(functions_ok);
    }

    std::printf(
        "TESSPROBE|HS|%s|%s|domain=%u part=%u outprim=%u in_cp=%u out_cp=%u "
        "out_cp_size=%u patch_const=%u max_patches=%u obj_threads=%u "
        "max_factor=%.1f half=%d lib=%s fns=%s%s%s\n",
        blob.name, (fields_ok && library && functions_ok) ? "PASS" : "FAIL",
        r.hs_tessellator_domain, r.hs_tessellator_partitioning,
        r.hs_tessellator_output_primitive, r.hs_input_control_point_count,
        r.hs_output_control_point_count, r.hs_output_control_point_size,
        r.hs_patch_constants_size, r.hs_max_patches_per_object_threadgroup,
        r.hs_max_object_threads_per_patch, r.hs_max_tessellation_factor,
        int(r.hs_tessellation_type_half), library ? "ok" : "FAIL",
        functions_ok ? "ok" : "FAIL",
        load_error.empty() ? "" : " err=", load_error.c_str());
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 2. Both tessellation VS blobs (object stage, zero-element stage-in).

TEST_CASE("tessellation VS blobs convert with synthesized stage-in",
          "[tessellation][vertex][device]") {
  auto& h = H();
  {
    INFO(h.init_error);
    REQUIRE(h.ok);
  }

  for (const TessVsBlobCase& blob : kTessVsBlobs) {
    INFO("tess VS blob " << blob.name);
    const StageResult& vs = h.GetTessVs(blob);
    if (!vs.ok) {
      std::printf("TESSPROBE|VS|%s|FAIL|%s\n", blob.name, vs.error.c_str());
      CHECK(vs.ok);
      continue;
    }
    const GeometryEmulationResult& r = vs.r;
    // The zero-element stage-in must still be synthesized — an empty stage-in
    // metallib is a hard pipeline failure (port notes gotcha 9).
    CHECK_FALSE(r.stage_in_metallib.empty());
    CHECK(r.vertex_output_size_in_bytes > 0);

    std::string load_error;
    id<MTLLibrary> vertex_library = NewLibrary(r.metallib, load_error);
    bool object_fn_ok = false;
    if (vertex_library) {
      object_fn_ok =
          LoadTessObjectFunction(vertex_library, r.function_name, load_error);
    }
    std::string stage_in_error;
    id<MTLLibrary> stage_in_library =
        NewLibrary(r.stage_in_metallib, stage_in_error);
    bool stage_in_fn_ok = false;
    if (stage_in_library) {
      stage_in_fn_ok = NewPlainFunction(stage_in_library,
                                        @"irconverter_stage_in_shader",
                                        stage_in_error) != nil;
    }
    {
      INFO("vertex load: " << load_error << " stage-in: " << stage_in_error);
      CHECK(vertex_library != nil);
      CHECK(object_fn_ok);
      CHECK(stage_in_library != nil);
      CHECK(stage_in_fn_ok);
    }

    std::printf(
        "TESSPROBE|VS|%s|%s|vs_output=%u stage_in_bytes=%zu fn=%s lib=%s "
        "obj_fn=%s stage_in_fn=%s%s%s%s%s\n",
        blob.name,
        (vertex_library && object_fn_ok && stage_in_library && stage_in_fn_ok &&
         !r.stage_in_metallib.empty() && r.vertex_output_size_in_bytes > 0)
            ? "PASS"
            : "FAIL",
        r.vertex_output_size_in_bytes, r.stage_in_metallib.size(),
        r.function_name.c_str(), vertex_library ? "ok" : "FAIL",
        object_fn_ok ? "ok" : "FAIL", stage_in_fn_ok ? "ok" : "FAIL",
        load_error.empty() ? "" : " err=", load_error.c_str(),
        stage_in_error.empty() ? "" : " stage_in_err=",
        stage_in_error.c_str());
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 3. Guest VS as the domain stage: 4 host types x clip counts 0/1/2.

TEST_CASE("guest VS translates and loads as the domain stage across clip counts",
          "[tessellation][domain][device]") {
  auto& h = H();
  {
    INFO(h.init_error);
    REQUIRE(h.ok);
  }

  std::vector<std::string> clip0_failures;
  std::vector<std::string> clip_gt0_failures;

  for (HostVertexShaderType type : kDomainTypes) {
    for (uint32_t clip : {0u, 1u, 2u}) {
      std::string cell =
          std::string(DomainTypeName(type)) + "/clip" + std::to_string(clip);
      INFO("domain cell " << cell);
      const StageResult& ds = h.GetDomain(type, clip);
      bool cell_ok = false;
      std::string detail;
      if (!ds.ok) {
        detail = ds.error;
      } else {
        const GeometryEmulationResult& r = ds.r;
        std::string load_error;
        id<MTLLibrary> library = NewLibrary(r.metallib, load_error);
        bool fn_ok = false;
        if (library) {
          fn_ok = NewPlainFunction(library, @"irconverter_dxil_domain_shader",
                                   load_error) != nil;
        }
        cell_ok = r.has_domain_info &&
                  r.ds_tessellator_domain == ExpectedDomainFor(type) &&
                  library != nil && fn_ok;
        CHECK(r.has_domain_info);
        CHECK(r.ds_tessellator_domain == ExpectedDomainFor(type));
        {
          INFO("domain load: " << load_error);
          CHECK(library != nil);
          CHECK(fn_ok);
        }
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
                      "ds_domain=%u in_cp=%u in_cp_size=%u patch_const=%u "
                      "gs_max_input=%u half=%d",
                      r.ds_tessellator_domain, r.ds_input_control_point_count,
                      r.ds_input_control_point_size, r.ds_patch_constants_size,
                      r.ds_max_input_prims_per_mesh_threadgroup,
                      int(r.ds_tessellation_type_half));
        detail = buffer;
        if (!load_error.empty()) {
          detail += " err=" + load_error;
        }
      }
      CHECK(cell_ok);
      if (!cell_ok) {
        (clip == 0 ? clip0_failures : clip_gt0_failures).push_back(cell);
      }
      std::printf("TESSPROBE|DS|%s|%s|%s\n", cell.c_str(),
                  cell_ok ? "PASS" : "FAIL", detail.c_str());
    }
  }

  // KILL-GATE report (decision belongs to the change owner, not this test):
  auto join = [](const std::vector<std::string>& v) {
    std::string out;
    for (const std::string& s : v) {
      if (!out.empty()) out += ",";
      out += s;
    }
    return out.empty() ? std::string("none") : out;
  };
  std::printf(
      "TESSPROBE|KILLGATE|clip0_unloadable=%s (non-empty -> scope-reduction "
      "trigger)\n",
      join(clip0_failures).c_str());
  std::printf(
      "TESSPROBE|KILLGATE|clip_gt0_only_failures=%s (non-empty with empty "
      "clip0 -> clip-bridge hard-precondition trigger)\n",
      join(clip_gt0_failures).c_str());
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 4. HS/DS pairing validation at clip 0.

TEST_CASE("HS/DS pairs validate under IRRuntimeValidateTessellationPipeline",
          "[tessellation][pairing][device]") {
  auto& h = H();
  {
    INFO(h.init_error);
    REQUIRE(h.ok);
  }

  for (const PairCase& pair : kPairs) {
    std::string cell = std::string(pair.mode) + "/" + pair.hs_name + " x " +
                       DomainTypeName(pair.ds_type);
    INFO("pair " << cell);
    const HsBlobCase* blob = h.FindHsBlob(pair.hs_name);
    REQUIRE(blob != nullptr);
    const StageResult& hs = h.GetHull(*blob);
    const StageResult& ds = h.GetDomain(pair.ds_type, /*clip_count=*/0);
    if (!hs.ok || !ds.ok) {
      std::printf("TESSPROBE|PAIR|%s|SKIP|hs=%s ds=%s\n", cell.c_str(),
                  hs.ok ? "ok" : hs.error.c_str(),
                  ds.ok ? "ok" : ds.error.c_str());
      CHECK(hs.ok);
      CHECK(ds.ok);
      continue;
    }

    IRRuntimeTessellatorOutputPrimitive output_primitive;
    IRRuntimePrimitiveType geometry_primitive;
    const char* geometry_function = nullptr;
    bool primitive_ok =
        MapOutputPrimitive(hs.r.hs_tessellator_output_primitive,
                           output_primitive, geometry_primitive,
                           geometry_function);
    CHECK(primitive_ok);
    if (!primitive_ok) {
      std::printf("TESSPROBE|PAIR|%s|FAIL|unsupported hs output primitive %u\n",
                  cell.c_str(), hs.r.hs_tessellator_output_primitive);
      continue;
    }

    // Argument order per port notes section 3 step 5.
    bool valid = IRRuntimeValidateTessellationPipeline(
        output_primitive, geometry_primitive,
        hs.r.hs_output_control_point_size, ds.r.ds_input_control_point_size,
        hs.r.hs_patch_constants_size, ds.r.ds_patch_constants_size,
        hs.r.hs_output_control_point_count,
        ds.r.ds_input_control_point_count);
    CHECK(valid);
    std::printf(
        "TESSPROBE|PAIR|%s|%s|hs_out_cp=%u/%uB ds_in_cp=%u/%uB "
        "hs_patch_const=%u ds_patch_const=%u outprim=%u\n",
        cell.c_str(), valid ? "PASS" : "FAIL",
        hs.r.hs_output_control_point_count, hs.r.hs_output_control_point_size,
        ds.r.ds_input_control_point_count, ds.r.ds_input_control_point_size,
        hs.r.hs_patch_constants_size, ds.r.ds_patch_constants_size,
        uint32_t(output_primitive));
  }
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// 5. STRETCH: one full tessellation-emulation pipeline per domain. Wiring
// failures that belong to task 3.1 are recorded as findings, never asserted.

namespace {

void AttemptStretchPipeline(const char* base_label, const char* hs_name,
                            HostVertexShaderType ds_type,
                            uint32_t clip_count) {
  std::string label_storage =
      std::string(base_label) + "/clip" + std::to_string(clip_count);
  const char* label = label_storage.c_str();
  auto& h = H();
  const HsBlobCase* blob = h.FindHsBlob(hs_name);
  if (!blob) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|unknown blob %s\n", label, hs_name);
    return;
  }
  const StageResult& hs = h.GetHull(*blob);
  const StageResult& ds = h.GetDomain(ds_type, clip_count);
  const StageResult& vs =
      h.GetTessVs(blob->adaptive ? kTessVsBlobs[1] : kTessVsBlobs[0]);
  if (!hs.ok || !ds.ok || !vs.ok) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|hs=%s ds=%s vs=%s\n", label,
                hs.ok ? "ok" : hs.error.c_str(),
                ds.ok ? "ok" : ds.error.c_str(),
                vs.ok ? "ok" : vs.error.c_str());
    return;
  }

  // Fragment: the fixture guest PS through the plain production back half.
  std::vector<uint32_t> ps_ucode = BuildFixturePixelUcode();
  Shader ps_shader(xenos::ShaderType::kPixel, 0x7E55000000000002ull,
                   ps_ucode.data(), ps_ucode.size(), std::endian::little);
  StringBuffer disasm;
  ps_shader.AnalyzeUcode(disasm);
  DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                  /*bindless_resources_used=*/true,
                                  /*edram_rov_used=*/false,
                                  /*gamma_render_target_as_unorm8=*/true);
  DxbcShaderTranslator::Modification ps_modification(
      translator.GetDefaultPixelShaderModification(
          ps_shader.GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters)));
  Shader::Translation* ps_translation =
      ps_shader.GetOrCreateTranslation(ps_modification.value);
  if (!translator.TranslateAnalyzedShader(*ps_translation)) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|fixture PS translation failed\n",
                label);
    return;
  }
  std::string error;
  std::vector<uint8_t> ps_dxil;
  if (!h.DxbcToDxil(ps_translation->translated_binary().data(),
                    ps_translation->translated_binary().size(), ps_dxil,
                    error)) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|fixture PS dxilconv: %s\n", label,
                error.c_str());
    return;
  }
  std::vector<uint8_t> ps_metallib;
  if (!h.msc.Convert(Stage::kFragment, ps_dxil, ps_metallib, &error)) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|fixture PS MSC: %s\n", label,
                error.c_str());
    return;
  }

  std::string lib_error;
  id<MTLLibrary> vertex_library = NewLibrary(vs.r.metallib, lib_error);
  id<MTLLibrary> stage_in_library =
      vertex_library ? NewLibrary(vs.r.stage_in_metallib, lib_error) : nil;
  id<MTLLibrary> hull_library =
      stage_in_library ? NewLibrary(hs.r.metallib, lib_error) : nil;
  id<MTLLibrary> domain_library =
      hull_library ? NewLibrary(ds.r.metallib, lib_error) : nil;
  id<MTLLibrary> fragment_library =
      domain_library ? NewLibrary(ps_metallib, lib_error) : nil;
  if (!fragment_library) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|library load: %s\n", label,
                lib_error.c_str());
    return;
  }
  NSString* fragment_name = [fragment_library functionNames].firstObject;
  if (!fragment_name) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|fragment library has no functions\n",
                label);
    return;
  }

  IRRuntimeTessellatorOutputPrimitive output_primitive;
  IRRuntimePrimitiveType geometry_primitive;
  const char* geometry_function = nullptr;
  if (!MapOutputPrimitive(hs.r.hs_tessellator_output_primitive,
                          output_primitive, geometry_primitive,
                          geometry_function)) {
    std::printf("TESSPROBE|STRETCH|%s|SKIP|unsupported hs output primitive\n",
                label);
    return;
  }

  MTLMeshRenderPipelineDescriptor* desc =
      [[MTLMeshRenderPipelineDescriptor alloc] init];
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  desc.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  desc.rasterSampleCount = 1;
  desc.alphaToCoverageEnabled = NO;

  IRGeometryTessellationEmulationPipelineDescriptor ir_desc = {};
  ir_desc.stageInLibrary = stage_in_library;
  ir_desc.vertexLibrary = vertex_library;
  ir_desc.vertexFunctionName = vs.r.function_name.c_str();
  ir_desc.hullLibrary = hull_library;
  ir_desc.hullFunctionName = hs.r.function_name.c_str();
  ir_desc.domainLibrary = domain_library;
  ir_desc.domainFunctionName = ds.r.function_name.c_str();
  // Runtime-provided passthrough geometry stage (lives in the domain library).
  ir_desc.geometryLibrary = nil;
  ir_desc.geometryFunctionName = geometry_function;
  ir_desc.fragmentLibrary = fragment_library;
  ir_desc.fragmentFunctionName = fragment_name.UTF8String;
  ir_desc.basePipelineDescriptor = desc;
  ir_desc.pipelineConfig.outputPrimitiveType = output_primitive;
  ir_desc.pipelineConfig.vsOutputSizeInBytes =
      vs.r.vertex_output_size_in_bytes;
  // Port notes gotcha 8: from the DOMAIN reflection, not the hull's.
  ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
      ds.r.ds_max_input_prims_per_mesh_threadgroup;
  ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup =
      hs.r.hs_max_patches_per_object_threadgroup;
  ir_desc.pipelineConfig.hsInputControlPointCount =
      hs.r.hs_input_control_point_count;
  ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup =
      hs.r.hs_max_object_threads_per_patch;
  ir_desc.pipelineConfig.hsMaxTessellationFactor =
      hs.r.hs_max_tessellation_factor;
  ir_desc.pipelineConfig.gsInstanceCount = 1;

  if (!ir_desc.pipelineConfig.vsOutputSizeInBytes ||
      !ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup ||
      !ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup ||
      !ir_desc.pipelineConfig.hsInputControlPointCount ||
      !ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup) {
    std::printf(
        "TESSPROBE|STRETCH|%s|FAIL|zero config field (vs_output=%u "
        "gs_max_input=%u hs_patches=%u hs_cp=%u hs_threads=%u)\n",
        label, ir_desc.pipelineConfig.vsOutputSizeInBytes,
        ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup,
        ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup,
        ir_desc.pipelineConfig.hsInputControlPointCount,
        ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup);
    return;
  }

  NSError* ns_error = nil;
  id<MTLRenderPipelineState> pipeline =
      IRRuntimeNewGeometryTessellationEmulationPipeline(h.device, &ir_desc,
                                                        &ns_error);
  if (pipeline) {
    std::printf(
        "TESSPROBE|STRETCH|%s|PASS|pipeline built (vs_output=%u "
        "gs_max_input=%u hs_patches=%u hs_cp=%u hs_threads=%u factor=%.1f)\n",
        label, ir_desc.pipelineConfig.vsOutputSizeInBytes,
        ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup,
        ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup,
        ir_desc.pipelineConfig.hsInputControlPointCount,
        ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup,
        ir_desc.pipelineConfig.hsMaxTessellationFactor);
  } else {
    std::printf(
        "TESSPROBE|STRETCH|%s|FAIL|%s\n", label,
        ns_error ? ns_error.localizedDescription.UTF8String : "nil error");
  }
}

}  // namespace

TEST_CASE("stretch: full tessellation emulation pipelines (recorded only)",
          "[tessellation][stretch][device]") {
  auto& h = H();
  {
    INFO(h.init_error);
    REQUIRE(h.ok);
  }
  // Clip 0 is the task-2.2 stretch deliverable; clips 1/2 are extra recorded
  // evidence because the known clip-distance bridge defect
  // (fix-metal-geometry-clip-distance-bridge) manifests at PIPELINE creation,
  // not at library load — a clip>0 STRETCH failure here with green clip-0 is
  // exactly the clip-bridge hard-precondition signal.
  for (uint32_t clip : {0u, 1u, 2u}) {
    AttemptStretchPipeline("triangle/discrete_triangle_3cp",
                           "discrete_triangle_3cp_hs",
                           HostVertexShaderType::kTriangleDomainCPIndexed,
                           clip);
    AttemptStretchPipeline("quad/discrete_quad_4cp", "discrete_quad_4cp_hs",
                           HostVertexShaderType::kQuadDomainCPIndexed, clip);
  }
  std::fflush(stdout);
  // The stretch build is task 3.1's deliverable; this case only records the
  // outcome so wiring failures never fail the 2.2 probe.
  SUCCEED("stretch outcomes recorded on TESSPROBE|STRETCH lines");
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
