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
#include <functional>
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

  // An assembled MSC tessellation-emulation pipeline (port-xenios-tessellation
  // task 3.1) plus everything the draw call must re-supply to
  // IRRuntimeDraw[Indexed]PatchesTessellationEmulation. The config fields are a
  // field-for-field IR-free mirror of IRRuntimeTessellationPipelineConfig (this
  // header stays free of the IR runtime types, like the converter header);
  // group 4 rebuilds the IR struct verbatim:
  //   outputPrimitiveType                    <- output_primitive_type
  //   vsOutputSizeInBytes                    <- vs_output_size_in_bytes
  //   gsMaxInputPrimitivesPerMeshThreadgroup <- gs_max_input_primitives_...
  //   hsMaxPatchesPerObjectThreadgroup       <- hs_max_patches_per_object_...
  //   hsInputControlPointCount               <- hs_input_control_point_count
  //   hsMaxObjectThreadsPerThreadgroup       <- hs_max_object_threads_...
  //   hsMaxTessellationFactor                <- hs_max_tessellation_factor
  //   gsInstanceCount                        <- gs_instance_count
  // and passes `geometry_primitive` (IRRuntimePrimitiveType) as the runtime
  // draw's primitive.
  struct TessellationPipeline {
    void* pipeline = nullptr;  // id<MTLRenderPipelineState> (mesh), retained.
    uint32_t output_primitive_type = 0;  // IRRuntimeTessellatorOutputPrimitive
    uint32_t vs_output_size_in_bytes = 0;
    uint32_t gs_max_input_primitives_per_mesh_threadgroup = 0;
    uint32_t hs_max_patches_per_object_threadgroup = 0;
    uint32_t hs_input_control_point_count = 0;
    uint32_t hs_max_object_threads_per_threadgroup = 0;
    float hs_max_tessellation_factor = 0.0f;
    uint32_t gs_instance_count = 1;
    uint32_t geometry_primitive = 0;  // IRRuntimePrimitiveType
    // Draws suppressed by a cached FAILED creation (pipeline == nullptr) —
    // same impact accounting as GeometryPipeline.
    uint32_t negative_hits = 0;
  };

  // Builds (or returns the cached) tessellation-emulation pipeline: prebuilt
  // tess VS (object stage, synthesized zero-element stage-in) + prebuilt HS +
  // the GUEST vertex shader compiled as the domain stage + guest PS (or the
  // depth-only fallback PS when `pixel` is null/functionless — tessellated
  // draws never require a guest PS).
  //
  // `domain_translation` must be the guest VS translation created with the
  // DOMAIN host_vertex_shader_type modification the shared PrimitiveProcessor
  // chose (ProcessingResult::host_vertex_shader_type — group 4 must NOT pass
  // the plain kVertex translation, and must NOT run it through
  // TranslateShader, which compiles Stage::kVertex; this path performs the
  // ucode->DXBC step itself if needed and compiles Stage::kDomain).
  // host_vertex_shader_type / tessellation_mode / host_primitive_type come
  // from the ProcessingResult + VGT_HOS_CNTL, never re-derived here.
  //
  // HS/DS pairing is checked with IRRuntimeValidateTessellationPipeline at
  // construction — a validation failure IS a construction failure (nullptr,
  // negative-cached). Returns null on any failure; never a partial pipeline.
  //
  // NOT called from the draw path yet (that is task group 4): dead code behind
  // this API, zero behavior change to existing paths.
  const TessellationPipeline* GetOrCreateTessellationPipelineState(
      MetalShader::MetalTranslation* domain_translation,
      MetalShader::MetalTranslation* pixel,
      Shader::HostVertexShaderType host_vertex_shader_type,
      xenos::TessellationMode tessellation_mode,
      xenos::PrimitiveType host_primitive_type,
      const uint64_t* color_pixel_formats, uint64_t depth_pixel_format,
      MetalPipelineRasterKey raster_key, const uint32_t* blend_controls,
      uint32_t color_write_mask);

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
    // Draws suppressed by a cached FAILED creation (pipeline == nullptr). Two
    // failed creations can suppress many later draws; this is the impact
    // number, counted separately from creation attempts.
    uint32_t negative_hits = 0;
  };

  // Object-stage VS for geometry emulation, cached by the vertex translation.
  const GeometryVertexStage* GetOrCreateGeometryVertexStage(
      MetalShader::MetalTranslation* vertex,
      PipelineGeometryShader geometry_type);
  // Mesh-stage generated GS, cached by the geometry-shader key.
  const GeometryShaderStage* GetOrCreateGeometryShaderStage(GeometryShaderKey key);
  // Wraps a metallib blob in an id<MTLLibrary> (retained void*); null on failure.
  void* NewLibraryFromBytes(const std::vector<uint8_t>& bytes, const char* label);

  // -- Tessellation static-stage caches (task 3.1). The blobs retain their
  // metallib bytes so a successful pipeline can publish the durable
  // all-or-nothing TessellationRecord without recompiling anything.

  // Prebuilt tessellation VS (object stage) compiled once per BLOB (adaptive vs
  // indexed — discrete and continuous share the indexed blob, so the key is
  // blob identity, not mode).
  struct TessellationVertexStage {
    void* library = nullptr;           // id<MTLLibrary> (object stage).
    void* stage_in_library = nullptr;  // id<MTLLibrary> (synthesized stage-in).
    std::string function_name;
    uint32_t vertex_output_size = 0;   // -> config.vsOutputSizeInBytes.
    std::vector<uint8_t> metallib;
    std::vector<uint8_t> stage_in_metallib;
    bool valid = false;
  };
  // Prebuilt HS compiled once per blob (mode x host-VS-type selects the blob).
  struct TessellationHullStage {
    void* library = nullptr;  // id<MTLLibrary>.
    std::string function_name;
    // Complete HS reflection set (design D3).
    uint32_t max_patches_per_object_threadgroup = 0;
    uint32_t max_object_threads_per_patch = 0;
    uint32_t patch_constants_size = 0;
    uint32_t input_control_point_count = 0;
    uint32_t output_control_point_count = 0;
    uint32_t output_control_point_size = 0;
    uint32_t tessellator_domain = 0;
    uint32_t tessellator_partitioning = 0;
    uint32_t tessellator_output_primitive = 0;
    float max_tessellation_factor = 0.0f;
    std::vector<uint8_t> metallib;
    bool valid = false;
  };
  // Guest VS compiled as the DOMAIN stage, cached by content-derived shader
  // identity (ucode hash + modification + type), never a translation pointer.
  struct TessellationDomainStage {
    void* library = nullptr;  // id<MTLLibrary>.
    std::string function_name;
    uint32_t max_input_prims_per_mesh_threadgroup = 0;  // -> config (NOT an HS field).
    uint32_t input_control_point_count = 0;
    uint32_t input_control_point_size = 0;
    uint32_t patch_constants_size = 0;
    uint32_t tessellator_domain = 0;
    std::vector<uint8_t> metallib;
    bool valid = false;
  };

  const TessellationVertexStage* GetOrCreateTessellationVertexStage(
      xenos::TessellationMode tessellation_mode);
  const TessellationHullStage* GetOrCreateTessellationHullStage(
      xenos::TessellationMode tessellation_mode,
      Shader::HostVertexShaderType host_vertex_shader_type);
  const TessellationDomainStage* GetOrCreateTessellationDomainStage(
      MetalShader::MetalTranslation* domain_translation);
  // Depth-only fallback PS for tessellated draws without a guest PS (ports
  // XeniOS's EnsureDepthOnlyPixelShader; the geometry-emulation descriptor
  // requires a fragment library, unlike the plain render-pipeline path).
  bool EnsureDepthOnlyPixelShader();
  // Populates any missing tessellation stage-cache entries from a durable
  // record (warm start). Returns false if any component failed to load onto
  // the device — the caller falls back to cold compilation.
  bool AdoptTessellationRecord(
      const MetalPipelineStorage::TessellationRecord& record,
      xenos::TessellationMode tessellation_mode,
      Shader::HostVertexShaderType host_vertex_shader_type,
      uint64_t domain_stage_key);

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
  std::unordered_map<MetalGeometryVertexStageKey, GeometryVertexStage,
                     MetalGeometryVertexStageKey::Hasher>
      geometry_vertex_stages_;
  std::unordered_map<uint32_t, GeometryShaderStage> geometry_shader_stages_;
  std::unordered_map<uint64_t, GeometryPipeline> geometry_pipelines_;

  // Tessellation caches (task 3.1). Stage keys: VS by blob identity
  // (0=indexed, 1=adaptive), HS by (mode << 32) | host_vs_type (1:1 with the
  // blob table), DS by content-derived shader identity. Pipelines by the full
  // XXH3 pipeline key. Libraries/pipelines retained here, released in Shutdown.
  std::unordered_map<uint32_t, TessellationVertexStage>
      tessellation_vertex_stages_;
  std::unordered_map<uint64_t, TessellationHullStage> tessellation_hull_stages_;
  std::unordered_map<uint64_t, TessellationDomainStage>
      tessellation_domain_stages_;
  std::unordered_map<uint64_t, TessellationPipeline> tessellation_pipelines_;
  // Depth-only fallback PS (built on first tessellated no-PS pipeline).
  void* depth_only_pixel_library_ = nullptr;  // id<MTLLibrary>, retained.
  std::string depth_only_pixel_function_name_;

  // Durable on-disk storage for M1/M2. Closed means "no durable cache", which
  // is today's behavior and the default.
  MetalPipelineStorage storage_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
