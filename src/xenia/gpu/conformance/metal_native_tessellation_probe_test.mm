/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// PRE-GATE PROBE (pre-gate-native-metal-drawpatches task 2.1). REMOVED AT
// CLOSEOUT along with the converter's `enable_emulation` parameter.
//
// THE ONE QUESTION: with geometry/tessellation emulation DISABLED, does MSC emit
// a function that Metal accepts as a NATIVE POST-TESSELLATION VERTEX FUNCTION?
//
// Why a probe and not a header audit: the MSC headers expose no *named*
// post-tessellation API, but absence of a named API is not proof of
// incapability. `drawPatches` is a METAL RENDER API — MSC may only need to emit
// a compatible function. MSC recognises IRShaderStageDomain, its generic
// compiler returns stage-specific metallibs, and
// IRCompilerEnableGeometryAndTessellationEmulation explicitly accepts false — a
// case production never exercises because our converter hardcodes true. This
// probe decides whether an entire architecture is discarded, so it is measured
// rather than inferred.
//
// PRE-COMMITTED VERDICT (recorded before the probe was run):
//   compilation rejected, OR no domain/vertex output, OR unloadable function,
//   OR native tessellation pipeline rejected  -> MEASURED NO-GO
//   native pipeline materialises              -> 2.1 PASSES, and 2.2's
//                                                VICP/value oracle must still
//                                                run before ANY architectural
//                                                conclusion is drawn.

#include <cstdint>
#include <string>
#include <vector>

#import <Metal/Metal.h>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

using HostVertexShaderType = Shader::HostVertexShaderType;

// Same minimal guest VS fixture the tessellation translation probe uses.
const uint32_t kGuestVertexUcode[] = {
    0x00001002, 0x00001000, 0xC2000000, 0x00001003,
    0x00002000, 0x00000000, 0xC8020000, 0x006C6C00,
    0xC1000000, 0xC88F803E, 0x00000000, 0xC2000000};

// Guest VS translated as a domain shader (ds_5_1 selected from
// host_vertex_shader_type) -> DXBC -> dxilconv -> DXIL.
bool BuildGuestDomainDxil(std::vector<uint8_t>& dxil_out, std::string& error) {
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
          HostVertexShaderType::kTriangleDomainCPIndexed));
  modification.vertex.user_clip_plane_count = 0;
  modification.vertex.user_clip_plane_cull = 0;
  Shader::Translation* translation =
      shader.GetOrCreateTranslation(modification.value);
  if (!translator.TranslateAnalyzedShader(*translation)) {
    error = "DxbcShaderTranslator failed";
    return false;
  }
  std::vector<uint8_t> dxbc = translation->translated_binary();

  DxbcToDxilConverter converter;
  std::string init_error;
  if (!converter.Initialize(&init_error)) {
    error = "dxilconv init: " + init_error;
    return false;
  }
  if (!converter.Convert(dxbc, dxil_out, &error)) {
    error = "dxilconv: " + error;
    return false;
  }
  return true;
}

// Attempts a MINIMAL NATIVE tessellation render pipeline using `fn` as the
// post-tessellation vertex function. Returns the Metal error text on failure.
std::string TryPipeline(id<MTLDevice> device, id<MTLFunction> fn,
                        bool with_tessellation) {
  // A VALID descriptor is required or Metal rejects with "Empty request."
  // before it ever inspects the vertex function -- a probe artifact that must
  // not be mistaken for a real rejection. Give it a real fragment stage and a
  // colour attachment so the only thing left to reject is the function itself.
  NSError* frag_err = nil;
  id<MTLLibrary> frag_lib = [device
      newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
                            "fragment float4 probe_fs() { return float4(1); }"
                   options:nil
                     error:&frag_err];
  MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
  desc.vertexFunction = fn;
  desc.fragmentFunction =
      frag_lib ? [frag_lib newFunctionWithName:@"probe_fs"] : nil;
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  if (frag_lib == nil) {
    return std::string("probe fragment library failed: ") +
           (frag_err ? frag_err.localizedDescription.UTF8String : "(none)");
  }
  if (with_tessellation) {
    // The native tessellation properties. If `fn` is not a [[patch]] function,
    // Metal rejects the pipeline here -- which IS the measurement.
    desc.maxTessellationFactor = 16;
    desc.tessellationFactorScaleEnabled = NO;
    desc.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
    desc.tessellationControlPointIndexType =
        MTLTessellationControlPointIndexTypeNone;
    desc.tessellationFactorStepFunction =
        MTLTessellationFactorStepFunctionConstant;
    desc.tessellationOutputWindingOrder = MTLWindingClockwise;
    desc.tessellationPartitionMode = MTLTessellationPartitionModeFractionalOdd;
  }
  NSError* err = nil;
  id<MTLRenderPipelineState> pso =
      [device newRenderPipelineStateWithDescriptor:desc error:&err];
  if (pso) {
    return "";
  }
  return err ? std::string(err.localizedDescription.UTF8String)
             : std::string("pipeline creation failed with no error object");
}

struct ConvertOutcome {
  bool ok = false;
  std::string error;
  std::string function_name;
  size_t metallib_bytes = 0;
};

ConvertOutcome Convert(MetalShaderConverter& conv,
                       MetalShaderConverter::Stage stage,
                       const std::vector<uint8_t>& dxil, bool emulation) {
  MetalShaderConverter::GeometryEmulationResult r;
  ConvertOutcome out;
  out.ok = conv.ConvertForGeometryEmulation(
      stage, dxil, MetalShaderConverter::InputTopology::kUndefined, r,
      emulation);
  out.error = r.error;
  out.function_name = r.function_name;
  out.metallib_bytes = r.metallib.size();
  return out;
}

TEST_CASE("MSC native post-tessellation probe (emulation disabled)",
          "[l1][device][metal][native-tess-probe]") {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    WARN("no Metal device; skipping");
    return;
  }

  std::vector<uint8_t> dxil;
  std::string error;
  REQUIRE(BuildGuestDomainDxil(dxil, error));
  REQUIRE(!dxil.empty());

  MetalShaderConverter conv;
  REQUIRE(conv.Initialize(&error));

  // CONTROL: emulation ENABLED must succeed, proving the fixture and the whole
  // dxilconv->MSC chain are good. Without this a failure below would be
  // uninterpretable.
  ConvertOutcome control =
      Convert(conv, MetalShaderConverter::Stage::kDomain, dxil, true);
  INFO("control(emulation=on, kDomain): ok=" << control.ok << " err='"
                                             << control.error << "' fn='"
                                             << control.function_name << "' "
                                             << control.metallib_bytes << "B");
  CHECK(control.ok);

  // PROBE: emulation DISABLED. Try BOTH stages -- our production extraction for
  // kDomain only asks for IRShaderStageDomain, so if MSC emitted a native
  // post-tessellation function as a VERTEX stage we would otherwise miss it.
  ConvertOutcome as_domain =
      Convert(conv, MetalShaderConverter::Stage::kDomain, dxil, false);
  ConvertOutcome as_vertex =
      Convert(conv, MetalShaderConverter::Stage::kVertex, dxil, false);

  printf("NATIVETESS|emulation=off|kDomain|ok=%d|err=%s|fn=%s|bytes=%zu\n",
         as_domain.ok ? 1 : 0, as_domain.error.c_str(),
         as_domain.function_name.c_str(), as_domain.metallib_bytes);
  printf("NATIVETESS|emulation=off|kVertex|ok=%d|err=%s|fn=%s|bytes=%zu\n",
         as_vertex.ok ? 1 : 0, as_vertex.error.c_str(),
         as_vertex.function_name.c_str(), as_vertex.metallib_bytes);

  // If either produced a metallib, try to materialise a NATIVE tessellation
  // pipeline from it. That, not compilation, is the real question.
  for (const auto& pair :
       {std::make_pair("kDomain", &as_domain), std::make_pair("kVertex", &as_vertex)}) {
    const ConvertOutcome& oc = *pair.second;
    if (!oc.ok || !oc.metallib_bytes) {
      continue;
    }
    MetalShaderConverter::GeometryEmulationResult r;
    conv.ConvertForGeometryEmulation(
        std::string(pair.first) == "kDomain"
            ? MetalShaderConverter::Stage::kDomain
            : MetalShaderConverter::Stage::kVertex,
        dxil, MetalShaderConverter::InputTopology::kUndefined, r, false);
    dispatch_data_t data = dispatch_data_create(
        r.metallib.data(), r.metallib.size(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* lib_err = nil;
    id<MTLLibrary> lib = [device newLibraryWithData:data error:&lib_err];
    if (!lib) {
      printf("NATIVETESS|%s|library=UNLOADABLE|err=%s\n", pair.first,
             lib_err ? lib_err.localizedDescription.UTF8String : "(none)");
      continue;
    }
    printf("NATIVETESS|%s|library=ok|functions=%s\n", pair.first,
           [[lib.functionNames componentsJoinedByString:@","] UTF8String]);
    // DECISIVE: what KIND of function did MSC emit? A post-tessellation vertex
    // function must be MTLFunctionTypeVertex(1). Anything else (visible/
    // callable = 5, mesh = 7, object = 6) cannot be a vertex stage at all, and
    // explains a pipeline rejection that has nothing to do with tessellation.
    for (NSString* name in lib.functionNames) {
      id<MTLFunction> f = [lib newFunctionWithName:name];
      printf("NATIVETESS|%s|fn=%s|functionType=%lu\n", pair.first,
             name.UTF8String, f ? (unsigned long)f.functionType : 999UL);
    }
    id<MTLFunction> fn =
        [lib newFunctionWithName:[NSString
                                     stringWithUTF8String:r.function_name.c_str()]];
    if (!fn && lib.functionNames.count) {
      fn = [lib newFunctionWithName:lib.functionNames[0]];
    }
    if (!fn) {
      printf("NATIVETESS|%s|function=MISSING\n", pair.first);
      continue;
    }
    // CONTROL FIRST: the same descriptor WITHOUT tessellation properties.
    // If this also fails, the failure is about the descriptor or the function,
    // not about tessellation -- and reading it as a tessellation rejection
    // would be exactly the probe artifact this staging exists to catch.
    std::string plain_err = TryPipeline(device, fn, /*with_tessellation=*/false);
    printf("NATIVETESS|%s|plain_pipeline=%s|err=%s\n", pair.first,
           plain_err.empty() ? "MATERIALIZED" : "REJECTED", plain_err.c_str());
    std::string pipe_err = TryPipeline(device, fn, /*with_tessellation=*/true);
    printf("NATIVETESS|%s|native_pipeline=%s|err=%s\n", pair.first,
           pipe_err.empty() ? "MATERIALIZED" : "REJECTED", pipe_err.c_str());
  }
}

}  // namespace
}  // namespace metal
}  // namespace gpu
}  // namespace xe
