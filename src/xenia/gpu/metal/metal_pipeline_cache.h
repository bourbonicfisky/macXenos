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
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_fixed_function_state.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"
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
  // Vertex-only, rasterization-disabled pipeline for memexport draws (VS
  // eA/eM exports, nothing rasterized, zero attachments). Returns null on
  // failure.
  void* GetOrCreateMemexportPipelineState(MetalShader::MetalTranslation* vertex);

  void* GetOrCreateRenderPipelineState(MetalShader::MetalTranslation* vertex,
                                       MetalShader::MetalTranslation* pixel,
                                       const uint64_t* color_pixel_formats,
                                       uint64_t depth_pixel_format,
                                       MetalPipelineRasterKey raster_key,
                                       const uint32_t* blend_controls,
                                       uint32_t color_write_mask);

  // Builds (or returns the cached) MESH id<MTLRenderPipelineState> that expands a
  // point/rectangle/quad primitive via MSC geometry emulation: the guest VS runs
  // as the object stage (with a synthesized stage-in), a generated DXBC geometry
  // shader runs as the mesh stage, and `pixel` supplies the fragment stage.
  // `vertex` must already be DXBC-translated (its translated_binary() is the guest
  // VS DXBC, reused as the object stage). On success, returns the pipeline and
  // fills the two out-params with the geometry-pipeline config the draw call needs
  // (VS output size + GS max input primitives per mesh threadgroup). Returns null
  // on failure. Requires mesh-shader support (Apple Silicon); the caller gates on
  // that. The pixel translation may be null (depth-only path unsupported here yet).
  void* GetOrCreateGeometryPipelineState(
      MetalShader::MetalTranslation* vertex,
      MetalShader::MetalTranslation* pixel,
      PipelineGeometryShader geometry_type,
      DxbcShaderTranslator::Modification vertex_modification,
      DxbcShaderTranslator::Modification pixel_modification,
      const uint64_t* color_pixel_formats, uint64_t depth_pixel_format,
      MetalPipelineRasterKey raster_key, const uint32_t* blend_controls,
      uint32_t color_write_mask,
      uint32_t* gs_vertex_size_out, uint32_t* gs_max_input_primitives_out);

  DxbcShaderTranslator* shader_translator() const {
    return shader_translator_.get();
  }

  // Opens the durable pipeline storage for a title (task group 3a). Called from
  // MetalCommandProcessor::InitializeShaderStorage, which the emulator invokes
  // on launch. A no-op unless a durable mechanism is enabled by cvar; failure
  // to open is never fatal — the backend runs exactly as it does today.
  void InitializeDurableStorage(const std::filesystem::path& cache_root,
                                uint32_t title_id);
  void ShutdownDurableStorage();

  const MetalPipelineStorage& durable_storage() const { return storage_; }

 private:
  // A guest VS compiled as an MSC geometry-emulation OBJECT stage, plus its
  // synthesized stage-in. Metal objects held as retained void*.
  struct GeometryVertexStage {
    void* library = nullptr;           // id<MTLLibrary> (object stage).
    void* stage_in_library = nullptr;  // id<MTLLibrary> (stage-in).
    std::string function_name;         // base name; MSC appends the object suffix.
    uint32_t vertex_output_size = 0;   // -> IRRuntimeGeometryPipelineConfig.
    bool valid = false;
  };
  // A generated DXBC geometry shader compiled as an MSC MESH stage.
  struct GeometryShaderStage {
    void* library = nullptr;  // id<MTLLibrary> (mesh stage).
    std::string function_name;
    uint32_t max_input_primitives = 0;  // -> IRRuntimeGeometryPipelineConfig.
    bool valid = false;
  };
  // An assembled geometry-emulation mesh pipeline + the config the draw needs.
  struct GeometryPipeline {
    void* pipeline = nullptr;  // id<MTLRenderPipelineState> (mesh), retained.
    uint32_t gs_vertex_size = 0;
    uint32_t gs_max_input_primitives = 0;
  };

  // Object-stage VS for geometry emulation, cached by the vertex translation.
  const GeometryVertexStage* GetOrCreateGeometryVertexStage(
      MetalShader::MetalTranslation* vertex,
      PipelineGeometryShader geometry_type);
  // Mesh-stage generated GS, cached by the geometry-shader key.
  const GeometryShaderStage* GetOrCreateGeometryShaderStage(GeometryShaderKey key);
  // Wraps a metallib blob in an id<MTLLibrary> (retained void*); null on failure.
  void* NewLibraryFromBytes(const std::vector<uint8_t>& bytes, const char* label);

  void* device_;  // id<MTLDevice>, not owned.

  std::unique_ptr<DxbcShaderTranslator> shader_translator_;
  DxbcToDxilConverter dxil_converter_;
  MetalShaderConverter metal_converter_;

  // Owns the loaded shaders, keyed by ucode hash.
  std::unordered_map<uint64_t, MetalShader*> shaders_;

  // Owns id<MTLRenderPipelineState> (retained), keyed by function+format hash.
  // A null value caches a known-failed combination.
  std::unordered_map<uint64_t, void*> render_pipeline_states_;

  // Geometry-emulation caches: object-stage VS by vertex translation, mesh-stage
  // GS by GeometryShaderKey, assembled pipeline by the full pipeline key. The
  // libraries are owned (retained) here and released in Shutdown.
  std::unordered_map<void*, GeometryVertexStage> geometry_vertex_stages_;
  std::unordered_map<uint32_t, GeometryShaderStage> geometry_shader_stages_;
  std::unordered_map<uint64_t, GeometryPipeline> geometry_pipelines_;

  // Durable on-disk storage for M1/M2. Closed means "no durable cache", which
  // is today's behavior and the default.
  MetalPipelineStorage storage_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
