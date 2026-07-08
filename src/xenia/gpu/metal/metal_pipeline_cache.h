/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
#define XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Owns guest-shader loading and the DXBC -> DXIL -> metallib -> MTLFunction
// translation for the Metal backend. This is the first slice of the render
// path (2c): it produces host shader functions but does not build full
// MTLRenderPipelineStates yet (that arrives with the render-target/draw wiring).
class MetalPipelineCache {
 public:
  // device is an id<MTLDevice>, owned by the MetalProvider.
  explicit MetalPipelineCache(void* device);
  ~MetalPipelineCache();

  bool Initialize();
  void Shutdown();

  // Creates (or returns the cached) analyzed shader for the given ucode.
  MetalShader* LoadShader(xenos::ShaderType shader_type,
                          const uint32_t* host_address, uint32_t dword_count);

  // Translates a specific modification of a shader all the way to an
  // MTLFunction (ucode -> DXBC -> DXIL -> metallib -> function). Returns true if
  // the translation produced a usable function. Idempotent per translation.
  bool TranslateShader(MetalShader::MetalTranslation& translation);

  // Builds (or returns the cached) id<MTLRenderPipelineState> for a translated
  // vertex + pixel function pair and the given attachment formats. The format
  // values are MTLPixelFormat. Returns null on failure. The pixel translation
  // may be null (depth-only). Both translations must already have functions.
  void* GetOrCreateRenderPipelineState(MetalShader::MetalTranslation* vertex,
                                       MetalShader::MetalTranslation* pixel,
                                       uint64_t color_pixel_format,
                                       uint64_t depth_pixel_format);

  DxbcShaderTranslator* shader_translator() const {
    return shader_translator_.get();
  }

 private:
  void* device_;  // id<MTLDevice>, not owned.

  std::unique_ptr<DxbcShaderTranslator> shader_translator_;
  DxbcToDxilConverter dxil_converter_;
  MetalShaderConverter metal_converter_;

  // Owns the loaded shaders, keyed by ucode hash.
  std::unordered_map<uint64_t, MetalShader*> shaders_;

  // Owns id<MTLRenderPipelineState> (retained), keyed by function+format hash.
  // A null value caches a known-failed combination.
  std::unordered_map<uint64_t, void*> render_pipeline_states_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
