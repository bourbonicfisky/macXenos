/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHADER_CONVERTER_H_
#define XENIA_GPU_METAL_METAL_SHADER_CONVERTER_H_

#include <cstdint>
#include <string>
#include <vector>

// Opaque IR types; the implementation includes the Metal Shader Converter
// headers so callers don't have to.
struct IRRootSignature;

namespace xe {
namespace gpu {
namespace metal {

// Stage 3 of the microcode -> DXBC -> DXIL -> metallib pipeline: compiles a DXIL
// container (from DxbcToDxilConverter) into a Metal library (AIR) using Apple's
// Metal Shader Converter (libmetalirconverter). Produces the bytes a
// MTLLibrary/newLibraryWithData would load; it does not touch a Metal device, so
// it is usable offline.
class MetalShaderConverter {
 public:
  enum class Stage { kVertex, kFragment };

  MetalShaderConverter() = default;
  ~MetalShaderConverter();

  // Builds the shared Xbox-360 root signature. Must succeed before Convert().
  bool Initialize(bool bindless_resources_used, std::string* error_out = nullptr);

  bool is_available() const { return root_signature_ != nullptr; }

  // Compiles a DXIL container to a metallib for the given stage. Returns false
  // and fills error_out on failure.
  bool Convert(Stage stage, const std::vector<uint8_t>& dxil,
               std::vector<uint8_t>& metallib_out,
               std::string* error_out = nullptr);

 private:
  // Constructs the root signature mirroring DxbcShaderTranslator::CbufferRegister
  // (b0..b4, space 0) and the D3D12 bindless descriptor-table shape. Static and
  // shader-independent, so one instance is shared across every shader.
  IRRootSignature* CreateXbox360RootSignature(bool bindless_resources_used,
                                              std::string* error_out);

  IRRootSignature* root_signature_ = nullptr;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHADER_CONVERTER_H_
