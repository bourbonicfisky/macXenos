/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_shader_converter.h"

#include "xenia/base/logging.h"

#include "metal_irconverter.h"

namespace xe {
namespace gpu {
namespace metal {

namespace {

// Metal Shader Converter's reserved function-constant register space, matching
// the value the D3D12 bindless path uses. Must be declared in the root signature
// when function constants are enabled.
constexpr uint32_t kFunctionConstantRegisterSpace = 2147420894u;

IRShaderStage ToIRStage(MetalShaderConverter::Stage stage) {
  switch (stage) {
    case MetalShaderConverter::Stage::kVertex:
      return IRShaderStageVertex;
    case MetalShaderConverter::Stage::kFragment:
      return IRShaderStageFragment;
    case MetalShaderConverter::Stage::kGeometry:
      return IRShaderStageGeometry;
  }
  return IRShaderStageInvalid;
}

IRInputTopology ToIRTopology(MetalShaderConverter::InputTopology topology) {
  switch (topology) {
    case MetalShaderConverter::InputTopology::kPoint:
      return IRInputTopologyPoint;
    case MetalShaderConverter::InputTopology::kLine:
      return IRInputTopologyLine;
    case MetalShaderConverter::InputTopology::kTriangle:
      return IRInputTopologyTriangle;
    case MetalShaderConverter::InputTopology::kPatch:
      return IRInputTopologyPatch;
    case MetalShaderConverter::InputTopology::kUndefined:
      break;
  }
  return IRInputTopologyUndefined;
}

// Copies the AIR bytecode for one stage out of a compiled IRObject. Returns false
// if the stage isn't present or is empty.
bool ExtractMetalLib(IRObject* metal_object, IRShaderStage stage,
                     std::vector<uint8_t>& out) {
  IRMetalLibBinary* metallib = IRMetalLibBinaryCreate();
  if (!metallib) {
    return false;
  }
  bool ok = false;
  if (IRObjectGetMetalLibBinary(metal_object, stage, metallib)) {
    size_t size = IRMetalLibGetBytecodeSize(metallib);
    if (size != 0) {
      out.resize(size);
      IRMetalLibGetBytecode(metallib, out.data());
      ok = true;
    }
  }
  IRMetalLibBinaryDestroy(metallib);
  return ok;
}

}  // namespace

MetalShaderConverter::~MetalShaderConverter() {
  if (root_signature_) {
    IRRootSignatureDestroy(root_signature_);
    root_signature_ = nullptr;
  }
}

IRRootSignature* MetalShaderConverter::CreateXbox360RootSignature(
    bool bindless_resources_used, std::string* error_out) {
  // Descriptor tables hold the heap-backed texture/UAV/sampler resources; shader
  // constants use root CBVs, mirroring the D3D12 bindless root-signature shape.
  IRDescriptorRange1 ranges[32] = {};
  IRRootDescriptorTable1 tables[32] = {};
  IRRootParameter1 params[32] = {};
  int range_count = 0;
  int table_count = 0;
  int param_count = 0;

  auto append_table = [&](IRDescriptorRangeType type, uint32_t descriptor_count,
                          uint32_t register_space, IRShaderVisibility vis) {
    IRDescriptorRange1& range = ranges[range_count++];
    range.RangeType = type;
    range.NumDescriptors = descriptor_count;
    range.BaseShaderRegister = 0;
    range.RegisterSpace = register_space;
    range.Flags = IRDescriptorRangeFlagNone;
    range.OffsetInDescriptorsFromTableStart = 0;

    IRRootDescriptorTable1& table = tables[table_count++];
    table.NumDescriptorRanges = 1;
    table.pDescriptorRanges = &range;

    IRRootParameter1& param = params[param_count++];
    param.ParameterType = IRRootParameterTypeDescriptorTable;
    param.DescriptorTable = table;
    param.ShaderVisibility = vis;
  };
  auto append_cbv = [&](uint32_t shader_register, IRShaderVisibility vis) {
    IRRootParameter1& param = params[param_count++];
    param.ParameterType = IRRootParameterTypeCBV;
    param.Descriptor.ShaderRegister = shader_register;
    param.Descriptor.RegisterSpace = 0;
    param.Descriptor.Flags = IRRootDescriptorFlagNone;
    param.ShaderVisibility = vis;
  };

  const uint32_t resource_descriptor_count =
      bindless_resources_used ? UINT32_MAX : 1025;
  const uint32_t sampler_descriptor_count =
      bindless_resources_used ? UINT32_MAX : 257;

  // SRVs in spaces 0-3, plus space 10 (hull), UAVs in spaces 0-3, samplers in
  // space 0 — the D3D12 bindless heap layout.
  for (uint32_t space = 0; space < 4; ++space) {
    append_table(IRDescriptorRangeTypeSRV, resource_descriptor_count, space,
                 IRShaderVisibilityAll);
  }
  append_table(IRDescriptorRangeTypeSRV, resource_descriptor_count, 10,
               IRShaderVisibilityAll);
  for (uint32_t space = 0; space < 4; ++space) {
    append_table(IRDescriptorRangeTypeUAV, resource_descriptor_count, space,
                 IRShaderVisibilityAll);
  }
  append_table(IRDescriptorRangeTypeSampler, sampler_descriptor_count, 0,
               IRShaderVisibilityAll);

  // Shader-constant CBVs, matching DxbcShaderTranslator::CbufferRegister:
  //   b0 = system constants   (all stages)
  //   b1 = float constants    (per-stage-local)
  //   b2 = bool/loop constants (all stages)
  //   b3 = fetch constants    (per-stage-local)
  //   b4 = descriptor indices (per-stage-local)
  // b0/b2 are common; the volatile stage-local CBVs are duplicated with disjoint
  // stage visibility so the runtime builds one graphics root argument block while
  // the DXIL-visible register numbers stay unchanged.
  append_cbv(0, IRShaderVisibilityAll);
  append_cbv(1, IRShaderVisibilityVertex);
  append_cbv(2, IRShaderVisibilityAll);
  append_cbv(3, IRShaderVisibilityVertex);
  append_cbv(4, IRShaderVisibilityVertex);
  const IRShaderVisibility stage_local[] = {IRShaderVisibilityHull,
                                            IRShaderVisibilityDomain,
                                            IRShaderVisibilityPixel};
  for (IRShaderVisibility vis : stage_local) {
    append_cbv(1, vis);
    append_cbv(3, vis);
    append_cbv(4, vis);
  }

  // MSC requires the function-constant CBV space to be declared when function
  // constants are enabled, though it is skipped in the top-level argument buffer.
  append_table(IRDescriptorRangeTypeCBV, 1, kFunctionConstantRegisterSpace,
               IRShaderVisibilityAll);

  IRRootSignatureDescriptor1 desc = {};
  desc.NumParameters = param_count;
  desc.pParameters = params;
  desc.NumStaticSamplers = 0;
  desc.pStaticSamplers = nullptr;
  desc.Flags = IRRootSignatureFlagNone;

  IRVersionedRootSignatureDescriptor versioned_desc = {};
  versioned_desc.version = IRRootSignatureVersion_1_1;
  versioned_desc.desc_1_1 = desc;

  IRError* error = nullptr;
  IRRootSignature* root_signature =
      IRRootSignatureCreateFromDescriptor(&versioned_desc, &error);
  if (error) {
    if (error_out) {
      const char* payload =
          reinterpret_cast<const char*>(IRErrorGetPayload(error));
      *error_out =
          std::string("root signature: ") + (payload ? payload : "unknown");
    }
    IRErrorDestroy(error);
    return nullptr;
  }
  return root_signature;
}

bool MetalShaderConverter::Initialize(bool bindless_resources_used,
                                      std::string* error_out) {
  if (root_signature_) {
    return true;
  }
  root_signature_ = CreateXbox360RootSignature(bindless_resources_used, error_out);
  if (!root_signature_) {
    XELOGE("MetalShaderConverter: failed to build Xbox-360 root signature");
    return false;
  }
  return true;
}

bool MetalShaderConverter::Convert(Stage stage,
                                   const std::vector<uint8_t>& dxil,
                                   std::vector<uint8_t>& metallib_out,
                                   std::string* error_out) {
  if (!root_signature_) {
    if (error_out) {
      *error_out = "MetalShaderConverter not initialized";
    }
    return false;
  }
  if (dxil.empty()) {
    if (error_out) {
      *error_out = "empty DXIL input";
    }
    return false;
  }

  IRObject* dxil_object = IRObjectCreateFromDXIL(dxil.data(), dxil.size(),
                                                 IRBytecodeOwnershipNone);
  if (!dxil_object) {
    if (error_out) {
      *error_out = "IRObjectCreateFromDXIL failed";
    }
    return false;
  }

  IRCompiler* compiler = IRCompilerCreate();
  if (!compiler) {
    IRObjectDestroy(dxil_object);
    if (error_out) {
      *error_out = "IRCompilerCreate failed";
    }
    return false;
  }

  // ForceTextureArray is mandatory: our DXBC translator emits texture2d_array and
  // our Metal textures are MTLTextureType2DArray. Bounds-check and Inf->NaN match
  // the D3D12 path's defensive codegen.
  IRCompilerSetCompatibilityFlags(
      compiler, static_cast<IRCompatibilityFlags>(
                    IRCompatibilityFlagForceTextureArray |
                    IRCompatibilityFlagBoundsCheck |
                    IRCompatibilityFlagVertexPositionInfToNan));
  // We supply our own root signature; ignore any embedded in the DXIL.
  IRCompilerIgnoreRootSignature(compiler, true);
  IRCompilerSetFunctionConstantResourceSpace(compiler,
                                             kFunctionConstantRegisterSpace);
  IRCompilerSetGlobalRootSignature(compiler, root_signature_);

  IRError* error = nullptr;
  IRObject* metal_object =
      IRCompilerAllocCompileAndLink(compiler, nullptr, dxil_object, &error);
  if (error) {
    if (error_out) {
      const char* payload =
          reinterpret_cast<const char*>(IRErrorGetPayload(error));
      *error_out =
          std::string("MSC compile failed: ") + (payload ? payload : "unknown");
    }
    IRErrorDestroy(error);
    IRCompilerDestroy(compiler);
    IRObjectDestroy(dxil_object);
    return false;
  }
  if (!metal_object) {
    if (error_out) {
      *error_out = "MSC returned null object without error";
    }
    IRCompilerDestroy(compiler);
    IRObjectDestroy(dxil_object);
    return false;
  }

  bool ok = false;
  IRMetalLibBinary* metallib = IRMetalLibBinaryCreate();
  if (metallib) {
    if (IRObjectGetMetalLibBinary(metal_object, ToIRStage(stage), metallib)) {
      size_t size = IRMetalLibGetBytecodeSize(metallib);
      if (size != 0) {
        metallib_out.resize(size);
        IRMetalLibGetBytecode(metallib, metallib_out.data());
        ok = true;
      }
    }
    IRMetalLibBinaryDestroy(metallib);
  }
  if (!ok && error_out) {
    *error_out = "no metallib produced for the requested stage";
  }

  IRObjectDestroy(metal_object);
  IRCompilerDestroy(compiler);
  IRObjectDestroy(dxil_object);
  return ok;
}

bool MetalShaderConverter::ConvertForGeometryEmulation(
    Stage stage, const std::vector<uint8_t>& dxil, InputTopology input_topology,
    GeometryEmulationResult& result) {
  result = GeometryEmulationResult{};
  if (stage != Stage::kVertex && stage != Stage::kGeometry) {
    result.error = "geometry emulation only converts the vertex or geometry stage";
    return false;
  }
  if (!root_signature_) {
    result.error = "MetalShaderConverter not initialized";
    return false;
  }
  if (dxil.empty()) {
    result.error = "empty DXIL input";
    return false;
  }

  IRObject* dxil_object =
      IRObjectCreateFromDXIL(dxil.data(), dxil.size(), IRBytecodeOwnershipNone);
  if (!dxil_object) {
    result.error = "IRObjectCreateFromDXIL failed";
    return false;
  }
  IRCompiler* compiler = IRCompilerCreate();
  if (!compiler) {
    IRObjectDestroy(dxil_object);
    result.error = "IRCompilerCreate failed";
    return false;
  }

  IRCompilerSetCompatibilityFlags(
      compiler, static_cast<IRCompatibilityFlags>(
                    IRCompatibilityFlagForceTextureArray |
                    IRCompatibilityFlagBoundsCheck |
                    IRCompatibilityFlagVertexPositionInfToNan));
  IRCompilerIgnoreRootSignature(compiler, true);
  IRCompilerSetFunctionConstantResourceSpace(compiler,
                                             kFunctionConstantRegisterSpace);
  IRCompilerSetGlobalRootSignature(compiler, root_signature_);
  // The primitive-expansion path runs the guest VS as an object stage feeding a
  // generated GS as a mesh stage; both must be compiled with emulation enabled and
  // the guest input topology set so MSC sizes the mesh threadgroups correctly.
  IRCompilerEnableGeometryAndTessellationEmulation(compiler, true);
  IRInputTopology topology = ToIRTopology(input_topology);
  if (topology != IRInputTopologyUndefined) {
    IRCompilerSetInputTopology(compiler, topology);
  }

  IRError* error = nullptr;
  IRObject* metal_object =
      IRCompilerAllocCompileAndLink(compiler, nullptr, dxil_object, &error);
  if (error) {
    const char* payload = reinterpret_cast<const char*>(IRErrorGetPayload(error));
    result.error =
        std::string("MSC geometry compile failed: ") + (payload ? payload : "?");
    IRErrorDestroy(error);
    IRCompilerDestroy(compiler);
    IRObjectDestroy(dxil_object);
    return false;
  }
  if (!metal_object) {
    result.error = "MSC returned null object without error";
    IRCompilerDestroy(compiler);
    IRObjectDestroy(dxil_object);
    return false;
  }

  // Pick the stage whose AIR + reflection we keep. The guest VS emits a vertex
  // (object) stage; the generated GS emits a mesh stage (or a geometry stage on
  // older toolchains).
  IRShaderStage reflect_stage = IRShaderStageInvalid;
  if (stage == Stage::kGeometry) {
    result.has_mesh_stage =
        ExtractMetalLib(metal_object, IRShaderStageMesh, result.metallib);
    if (result.has_mesh_stage) {
      reflect_stage = IRShaderStageMesh;
    } else if (ExtractMetalLib(metal_object, IRShaderStageGeometry,
                               result.metallib)) {
      result.has_geometry_stage = true;
      reflect_stage = IRShaderStageGeometry;
    }
  } else {
    if (ExtractMetalLib(metal_object, IRShaderStageVertex, result.metallib)) {
      reflect_stage = IRShaderStageVertex;
    }
  }
  if (reflect_stage == IRShaderStageInvalid || result.metallib.empty()) {
    result.error = "geometry emulation produced no metallib for the stage";
    IRObjectDestroy(metal_object);
    IRCompilerDestroy(compiler);
    IRObjectDestroy(dxil_object);
    return false;
  }

  IRShaderReflection* reflection = IRShaderReflectionCreate();
  if (reflection && IRObjectGetReflection(metal_object, reflect_stage, reflection)) {
    const char* entry = IRShaderReflectionGetEntryPointFunctionName(reflection);
    if (entry) {
      result.function_name = entry;
    }
    if (reflect_stage == IRShaderStageVertex) {
      IRVersionedVSInfo vs_info = {};
      vs_info.version = IRReflectionVersion_1_0;
      if (IRShaderReflectionCopyVertexInfo(reflection, IRReflectionVersion_1_0,
                                           &vs_info)) {
        result.vertex_output_size_in_bytes =
            vs_info.info_1_0.vertex_output_size_in_bytes;
        IRShaderReflectionReleaseVertexInfo(&vs_info);
      }
      // NOTE: XeniOS also synthesizes a stage-in function here
      // (IRMetalLibSynthesizeStageInFunction) because its VS fetches vertices via
      // stage-in inputs. Our VS uses IN-SHADER vfetch from shared memory (zero
      // vertex inputs), so whether a (possibly empty) stage-in library is required
      // by the geometry-emulation pipeline is the key on-device validation point;
      // left empty here and revisited when the pipeline is wired (next commit).
    } else {
      IRVersionedGSInfo gs_info = {};
      gs_info.version = IRReflectionVersion_1_0;
      if (IRShaderReflectionCopyGeometryInfo(reflection, IRReflectionVersion_1_0,
                                             &gs_info)) {
        result.gs_max_input_primitives_per_mesh_threadgroup =
            gs_info.info_1_0.max_input_primitives_per_mesh_threadgroup;
        IRShaderReflectionReleaseGeometryInfo(&gs_info);
      }
    }
  }
  if (reflection) {
    IRShaderReflectionDestroy(reflection);
  }
  if (result.function_name.empty()) {
    result.function_name = (stage == Stage::kVertex) ? "vertexMain" : "main";
  }

  result.success = true;
  IRObjectDestroy(metal_object);
  IRCompilerDestroy(compiler);
  IRObjectDestroy(dxil_object);
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
