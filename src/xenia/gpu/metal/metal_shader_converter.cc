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
#include "xenia/gpu/metal/metal_binding_layout.h"

#include "metal_irconverter.h"

namespace xe {
namespace gpu {
namespace metal {

namespace {

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
  // Descriptor tables hold the heap-backed texture/UAV/sampler resources;
  // shader constants use root CBVs, mirroring the D3D12 bindless root-signature
  // shape. kMetalBindingLayout is also consumed by IssueDraw's top-level
  // argument buffer, so parameter order has one authority.
  IRDescriptorRange1 ranges[kMetalBindingSlotCount] = {};
  IRRootDescriptorTable1 tables[kMetalBindingSlotCount] = {};
  IRRootParameter1 params[kMetalBindingSlotCount] = {};
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
  auto append_cbv = [&](uint32_t shader_register, uint32_t register_space,
                        IRShaderVisibility vis) {
    IRRootParameter1& param = params[param_count++];
    param.ParameterType = IRRootParameterTypeCBV;
    param.Descriptor.ShaderRegister = shader_register;
    param.Descriptor.RegisterSpace = register_space;
    param.Descriptor.Flags = IRRootDescriptorFlagNone;
    param.ShaderVisibility = vis;
  };

  auto get_visibility = [](MetalBindingVisibility visibility) {
    switch (visibility) {
      case MetalBindingVisibility::kVertex:
        return IRShaderVisibilityVertex;
      case MetalBindingVisibility::kHull:
        return IRShaderVisibilityHull;
      case MetalBindingVisibility::kDomain:
        return IRShaderVisibilityDomain;
      case MetalBindingVisibility::kPixel:
        return IRShaderVisibilityPixel;
      case MetalBindingVisibility::kAll:
      default:
        return IRShaderVisibilityAll;
    }
  };
  auto get_range_type = [](MetalBindingKind kind) {
    switch (kind) {
      case MetalBindingKind::kSrvTable:
        return IRDescriptorRangeTypeSRV;
      case MetalBindingKind::kUavTable:
        return IRDescriptorRangeTypeUAV;
      case MetalBindingKind::kSamplerTable:
        return IRDescriptorRangeTypeSampler;
      case MetalBindingKind::kFunctionConstantTable:
        return IRDescriptorRangeTypeCBV;
      case MetalBindingKind::kConstantBuffer:
      default:
        return IRDescriptorRangeTypeCBV;
    }
  };
  for (const MetalBindingParameter& binding : kMetalBindingLayout) {
    IRShaderVisibility visibility = get_visibility(binding.visibility);
    if (binding.kind == MetalBindingKind::kConstantBuffer) {
      append_cbv(binding.shader_register, binding.register_space, visibility);
    } else {
      append_table(
          get_range_type(binding.kind),
          GetMetalBindingDescriptorCount(binding, bindless_resources_used),
          binding.register_space, visibility);
    }
  }
  assert_true(param_count == int(kMetalBindingSlotCount));

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
  IRCompilerSetFunctionConstantResourceSpace(
      compiler, kMetalFunctionConstantRegisterSpace);
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
  IRCompilerSetFunctionConstantResourceSpace(
      compiler, kMetalFunctionConstantRegisterSpace);
  IRCompilerSetGlobalRootSignature(compiler, root_signature_);
  // The primitive-expansion path runs the guest VS as an object stage feeding a
  // generated GS as a mesh stage; both must be compiled with emulation enabled and
  // the guest input topology set so MSC sizes the mesh threadgroups correctly.
  IRCompilerEnableGeometryAndTessellationEmulation(compiler, true);
  // The guest VS becomes the Metal OBJECT stage, which the geometry-emulation
  // runtime links against a SEPARATE stage-in function (synthesized below). Put
  // the compiler in separate-stage-in mode before the compile so the object
  // shader is generated to consume a linked stage-in rather than Metal vertex
  // fetch. (No effect on the GS/mesh stage.)
  if (stage == Stage::kVertex) {
    IRCompilerSetStageInGenerationMode(
        compiler, IRStageInCodeGenerationModeUseSeparateStageInFunction);
  }
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
      // Synthesize the stage-in function the geometry-emulation runtime REQUIRES.
      // IRRuntimeNewGeometryEmulationPipeline dereferences
      // descriptor->stageInLibrary unconditionally (grabs function 0), so a null
      // stage-in library crashes — it is not an optional field. Our guest VS
      // fetches vertices in-shader (zero vertex inputs), so we hand MSC a
      // zero-element input layout; MSC still emits a valid (empty-input) stage-in
      // object that links to the VS object stage. Mirrors XeniOS's zero-input case
      // (metal_pipeline_cache.cc BuildStageInInputLayout -> numElements=0). The
      // compiler was put in separate-stage-in mode before the compile above.
      IRVersionedInputLayoutDescriptor input_layout = {};
      input_layout.version = IRInputLayoutDescriptorVersion_1;
      input_layout.desc_1_0.numElements = 0;
      IRMetalLibBinary* stage_in_lib = IRMetalLibBinaryCreate();
      if (stage_in_lib) {
        if (IRMetalLibSynthesizeStageInFunction(compiler, reflection,
                                                &input_layout, stage_in_lib)) {
          size_t stage_in_size = IRMetalLibGetBytecodeSize(stage_in_lib);
          if (stage_in_size) {
            result.stage_in_metallib.resize(stage_in_size);
            IRMetalLibGetBytecode(stage_in_lib, result.stage_in_metallib.data());
          }
        }
        IRMetalLibBinaryDestroy(stage_in_lib);
      }
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

  // A geometry-emulation vertex (object) stage without a stage-in library cannot
  // build a pipeline (the runtime would crash on the null library), so treat an
  // empty synthesis as a hard failure rather than deferring it to pipeline build.
  if (stage == Stage::kVertex && result.stage_in_metallib.empty()) {
    result.error = "failed to synthesize the geometry-emulation stage-in function";
    IRObjectDestroy(metal_object);
    IRCompilerDestroy(compiler);
    IRObjectDestroy(dxil_object);
    return false;
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
