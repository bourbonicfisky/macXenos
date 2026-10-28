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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_fixed_function_state.h"
#include "xenia/gpu/metal/metal_dxil_store.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/shader_storage.h"
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
  // device is an id<MTLDevice>, owned by the MetalProvider. The draw
  // resolution scale is baked into translated shaders.
  explicit MetalPipelineCache(void* device,
                              uint32_t draw_resolution_scale_x = 1,
                              uint32_t draw_resolution_scale_y = 1);
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

  // The single cache-aware DXBC -> DXIL entry point. EVERY production
  // conversion site must use it; calling dxil_converter_raw_.Convert directly
  // silently opts that site out of both the DXIL store and the build census.
  // The key is the exact byte buffer passed in, so any pre-Convert transform
  // (e.g. RelabelDxbcClipDistance) is covered without being enumerated.
  // `site` is a diagnostic label and does not participate in the key.
  bool ConvertDxbcToDxilCached(const std::vector<uint8_t>& dxbc,
                               std::vector<uint8_t>& dxil_out,
                               std::string* error_out, const char* site);

  // Builds (or returns the cached) id<MTLRenderPipelineState> for a translated
  // vertex + pixel function pair and the given attachment formats. The format
  // values are MTLPixelFormat. Returns null on failure. The pixel translation
  // may be null (depth-only). Both translations must already have functions.
  // Vertex-only, rasterization-disabled pipeline for memexport draws (VS
  // eA/eM exports, nothing rasterized, zero attachments). Returns null on
  // failure.
  void* GetOrCreateMemexportPipelineState(MetalShader::MetalTranslation* vertex);

  // Flag bit OR-ed into a `depth_pixel_format` argument: the pass binds the
  // (combined Depth32Float_Stencil8) texture's depth plane only, so the
  // pipeline declares no stencil attachment (reduce-metal-draw-pass-traffic
  // D2: a pass whose draws never test or write stencil leaves the plane
  // unbound and skips its tile load/store). Part of the pipeline key.
  static constexpr uint64_t kDepthWithoutStencilBit = 1ull << 40;

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

  // M2 — writes the binary archive if this session added pipelines to it. The
  // emulator quits through std::quick_exit, so nothing runs at teardown: the
  // archive is saved from MetalCommandProcessor::SaveShaderStorageToDisk (GPU
  // paused, just before the exit) and periodically from the swap, which is
  // what keeps a crash or a SIGKILL from losing a whole session of pipelines.
  // `wait` blocks until every queued segment is on disk.
  void SaveBinaryArchive(const char* reason, bool wait);
  void MaybeSaveBinaryArchiveOnSwap();

  const MetalPipelineStorage& durable_storage() const { return storage_; }

  // Shader and pipeline storage: the guest shaders (.xsh, the format D3D12 and
  // Vulkan share) and render pipeline descriptions (.msc.metal.xpso) a title has
  // used, recorded as they are first built and rebuilt at the next launch.
  // The metallib cache and binary archive hold only translator output, so a
  // build that changes the translator discards them; this record is what
  // lets the next launch compile before gameplay instead of inside it.
  // XE_METAL_SHADER_STORAGE=0 disables it. GPU thread only.
  void InitializeShaderStorage(const std::filesystem::path& cache_root,
                               uint32_t title_id);
  void ShutdownShaderStorage();
  // Hands this frame's new records to the writer thread.
  void FlushShaderStorage();

 private:
  XEPACKEDSTRUCT(StoredPipelineDescription, {
    uint64_t vertex_shader_hash;
    uint64_t vertex_shader_modification;
    // 0 for a depth-only pipeline.
    uint64_t pixel_shader_hash;
    // With fetch_plain_pattern cleared: the pattern index is session-local,
    // so the pattern itself is stored below and re-registered on load.
    uint64_t pixel_shader_modification;
    uint64_t color_formats[xenos::kMaxColorRenderTargets];
    uint64_t depth_format;
    uint32_t sample_count;
    uint32_t blend_controls[xenos::kMaxColorRenderTargets];
    uint32_t color_write_mask;
    MetalShader::FetchPattern pixel_fetch_pattern;  // fetch_mask 0: none
    static constexpr uint32_t kVersion = 0x20261027;
  });
  XEPACKEDSTRUCT(PipelineStoredDescription, {
    uint64_t description_hash;
    StoredPipelineDescription description;
  });
  void StorePipeline(MetalShader::MetalTranslation* vertex,
                     MetalShader::MetalTranslation* pixel,
                     const uint64_t* color_pixel_formats,
                     uint64_t depth_pixel_format, uint32_t sample_count,
                     const uint32_t* blend_controls,
                     uint32_t color_write_mask);
  ShaderStorageWriter<PipelineStoredDescription> shader_storage_;
  bool shader_storage_flush_shaders_ = false;
  bool shader_storage_flush_pipelines_ = false;
  // Set while stored pipelines are rebuilt, so they are not written again.
  bool shader_storage_replaying_ = false;

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

  void OpenBinaryArchive();
  void CloseBinaryArchive();
  void RebuildBinaryArchiveLookupList();
  void FinishBinaryArchiveCompaction();

  void* device_;  // id<MTLDevice>, not owned.
  uint32_t draw_resolution_scale_x_;
  uint32_t draw_resolution_scale_y_;

  std::unique_ptr<DxbcShaderTranslator> shader_translator_;
  // NAMED to make a bypass visible. Only ConvertDxbcToDxilCached may call this
  // directly; every other site must go through it or it silently opts out of
  // both the DXIL store and the build census — which is exactly what six
  // production sites had done. `grep -c "dxil_converter_raw_\.Convert"` must
  // stay at 2 (the cold conversion and the verification re-convert, both inside
  // the helper).
  DxbcToDxilConverter dxil_converter_raw_;
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

  // Durable on-disk storage for M1/M2. Closed means "no durable cache", which
  // is today's behavior and the default.
  MetalPipelineStorage storage_;

  // M2 — the binary archive, null unless metal_pipeline_binary_archive is on
  // and storage_ opened. Only the thread that creates render pipelines touches
  // it, plus SaveShaderStorageToDisk with that thread paused. Segments
  // (id<MTLBinaryArchive>, retained) are loaded or already serialized and are
  // only read; `binary_archive_` is the fresh archive new pipelines go to, and
  // `binary_archive_lookup_list_` (NSArray, retained) is the LOADED segments
  // for MTLRenderPipelineDescriptor.binaryArchives (archives this session
  // made only hold pipelines already in render_pipeline_states_). `archived_pipeline_keys_` are
  // the keys in any of them, so a pipeline is added once ever and a lookup can
  // fail fast on a miss; `binary_archive_pending_keys_` are the ones only in
  // the fresh archive.
  std::vector<void*> binary_archive_segments_;
  void* binary_archive_ = nullptr;
  void* binary_archive_lookup_list_ = nullptr;
  std::unordered_set<uint64_t> archived_pipeline_keys_;
  std::vector<uint64_t> binary_archive_pending_keys_;
  uint32_t binary_archive_next_segment_ = 0;
  // Serial dispatch queue (retained) that serializes finished archives.
  void* binary_archive_queue_ = nullptr;
  std::chrono::steady_clock::time_point binary_archive_last_save_;
  // Compaction. A session that opens with more than kCompactArchiveSegments
  // segments adds every pipeline it obtains, archive hits included, to the
  // fresh archive again (`binary_archive_session_keys_`). At quit it deletes
  // every old segment if 80% of their pipelines were saved again, otherwise
  // only the segments whose pipelines all were.
  struct OldArchiveSegment {
    std::filesystem::path path;
    std::vector<uint64_t> keys;
  };
  std::vector<OldArchiveSegment> binary_archive_compacting_;
  std::unordered_set<uint64_t> binary_archive_session_keys_;
  uint64_t binary_archive_failed_at_open_ = 0;

  // Durable DXIL-intermediate store (task 2.2). SEPARATE from storage_ and
  // deliberately so: its identity omits the MSC hash, the device and the OS
  // build, none of which a DXIL artifact depends on. That is what lets it
  // survive an OS or MSC change that legitimately discards every metallib.
  // Default off (cvar metal_dxil_cache) — task 1.1 measured that it saves
  // nothing on an ordinary warm launch.
  MetalDxilStore dxil_store_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_CACHE_H_
