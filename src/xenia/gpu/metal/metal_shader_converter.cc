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
  }
  return IRShaderStageInvalid;
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

}  // namespace metal
}  // namespace gpu
}  // namespace xe
