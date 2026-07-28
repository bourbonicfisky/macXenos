/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_pipeline_cache.h"

#import <Metal/Metal.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/metal/metal_pipeline_census.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

// Metal Shader Converter runtime (declarations only; implementation TU is
// ir_runtime_impl.mm). Provides IRGeometryEmulationPipelineDescriptor and
// IRRuntimeNewGeometryEmulationPipeline for the geometry-emulation mesh pipeline.
#include "metal_irconverter_runtime.h"

namespace xe {
namespace gpu {
namespace metal {

MetalPipelineCache::MetalPipelineCache(void* device) : device_(device) {}

MetalPipelineCache::~MetalPipelineCache() { Shutdown(); }

bool MetalPipelineCache::Initialize() {
  // Match the shader-spike configuration: Apple vendor, bindless resources, no
  // pixel-shader-interlock EDRAM (the copy-based RT model), which is what the
  // Xbox-360 root signature in MetalShaderConverter expects.
  // gamma_render_target_as_unorm8=true: k_8_8_8_8_GAMMA render targets are
  // plain unorm8 on our side, so the PS must PWL-gamma-encode its linear output
  // when kSysFlag_ConvertColorNToGamma is set. Without this ctor flag the
  // translator never emits that codepath and the flag is silently ignored —
  // JC2's scene renders ~0.02-linear sunlight that only becomes visible
  // (0x5C-range) after gamma encoding + exposure (the "black gameplay" bug).
  shader_translator_ = std::make_unique<DxbcShaderTranslator>(
      ui::GraphicsProvider::GpuVendorID::kApple,
      /*bindless_resources_used=*/true, /*edram_rov_used=*/false,
      /*gamma_render_target_as_unorm8=*/true);

  std::string error;
  if (!dxil_converter_.Initialize(&error)) {
    XELOGE("MetalPipelineCache: DXBC->DXIL converter init failed: {}", error);
    return false;
  }
  if (!metal_converter_.Initialize(/*bindless_resources_used=*/true, &error)) {
    XELOGE("MetalPipelineCache: Metal Shader Converter init failed: {}", error);
    return false;
  }
  return true;
}

void MetalPipelineCache::Shutdown() {
  // CENSUS (temporary; see metal_pipeline_census.h). A live title is usually
  // killed rather than shut down cleanly, so this full report is a bonus — the
  // periodic swap line is the evidence that always survives.
  pipeline_census::Report("pipeline cache shutdown");
  for (auto& pair : render_pipeline_states_) {
    if (pair.second) {
      id<MTLRenderPipelineState> state =
          (__bridge_transfer id<MTLRenderPipelineState>)pair.second;
      (void)state;
    }
  }
  render_pipeline_states_.clear();
  // Release geometry-emulation pipelines + stage libraries (all retained void*).
  for (auto& pair : geometry_pipelines_) {
    if (pair.second.pipeline) {
      id<MTLRenderPipelineState> state =
          (__bridge_transfer id<MTLRenderPipelineState>)pair.second.pipeline;
      (void)state;
    }
  }
  geometry_pipelines_.clear();
  for (auto& pair : geometry_vertex_stages_) {
    if (pair.second.library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)pair.second.library;
      (void)lib;
    }
    if (pair.second.stage_in_library) {
      id<MTLLibrary> lib =
          (__bridge_transfer id<MTLLibrary>)pair.second.stage_in_library;
      (void)lib;
    }
  }
  geometry_vertex_stages_.clear();
  for (auto& pair : geometry_shader_stages_) {
    if (pair.second.library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)pair.second.library;
      (void)lib;
    }
  }
  geometry_shader_stages_.clear();
  for (auto& pair : shaders_) {
    delete pair.second;
  }
  shaders_.clear();
  shader_translator_.reset();
}

MetalShader* MetalPipelineCache::LoadShader(xenos::ShaderType shader_type,
                                            const uint32_t* host_address,
                                            uint32_t dword_count) {
  uint64_t data_hash =
      XXH3_64bits(host_address, dword_count * sizeof(uint32_t));
  auto it = shaders_.find(data_hash);
  if (it != shaders_.end()) {
    return it->second;
  }
  // Create and stash the shader even if it later fails translation, so we don't
  // retry a known-bad shader.
  MetalShader* shader =
      new MetalShader(shader_type, data_hash, host_address, dword_count);
  shaders_.emplace(data_hash, shader);
  return shader;
}

bool MetalPipelineCache::TranslateShader(
    MetalShader::MetalTranslation& translation) {
  if (translation.is_translated()) {
    return translation.is_valid() && translation.has_metal_function();
  }

  MetalShader& shader = static_cast<MetalShader&>(translation.shader());

  // CENSUS (temporary; see metal_pipeline_census.h). Placed after the
  // is_translated() early-out so a cache hit is not counted as a creation.
  bool census_ok = false;
  pipeline_census::Scope census_scope(pipeline_census::Event::kShaderTranslate,
                                      &census_ok);

  // Stage 1: microcode -> DXBC (reused DxbcShaderTranslator).
  if (!shader_translator_->TranslateAnalyzedShader(translation)) {
    XELOGE("MetalPipelineCache: DXBC translation failed for shader {:016X}",
           shader.ucode_data_hash());
    return false;
  }
  const std::vector<uint8_t>& dxbc = translation.translated_binary();
  if (dxbc.empty()) {
    XELOGE("MetalPipelineCache: empty DXBC for shader {:016X}",
           shader.ucode_data_hash());
    return false;
  }

  // Stage 2: DXBC -> DXIL.
  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: DXBC->DXIL failed for shader {:016X}: {}",
           shader.ucode_data_hash(), error);
    return false;
  }

  // Shader-chain forensics (XE_METAL_DUMP_XLAT=<hex ucode hash>): dump the
  // translated DXBC + DXIL + metallib for that shader to /tmp for offline
  // disassembly (the light-VS frc-unpack outputs zero; the M4 GPU primitive
  // is proven exact, so the divergence is inside this chain).
  static const uint64_t s_dump_xlat = [] {
    const char* v = std::getenv("XE_METAL_DUMP_XLAT");
    return v ? strtoull(v, nullptr, 16) : 0ull;
  }();
  if (s_dump_xlat && translation.shader().ucode_data_hash() == s_dump_xlat) {
    auto dump = [&](const char* ext, const std::vector<uint8_t>& data) {
      std::string path =
          fmt::format("/tmp/xe_xlat_{:016X}_{:016X}.{}", s_dump_xlat, translation.modification(), ext);
      FILE* f = fopen(path.c_str(), "wb");
      if (f) {
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        XELOGW("MetalPipelineCache: dumped {} ({} bytes)", path, data.size());
      }
    };
    dump("dxbc", dxbc);
  }
  // Stage 3: DXIL -> metallib.
  MetalShaderConverter::Stage stage =
      shader.type() == xenos::ShaderType::kVertex
          ? MetalShaderConverter::Stage::kVertex
          : MetalShaderConverter::Stage::kFragment;
  if (s_dump_xlat && translation.shader().ucode_data_hash() == s_dump_xlat) {
    std::string path = fmt::format("/tmp/xe_xlat_{:016X}_{:016X}.dxil", s_dump_xlat, translation.modification());
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
      fwrite(dxil.data(), 1, dxil.size(), f);
      fclose(f);
      XELOGW("MetalPipelineCache: dumped {} ({} bytes)", path, dxil.size());
    }
  }
  std::vector<uint8_t> metallib;
  if (!metal_converter_.Convert(stage, dxil, metallib, &error)) {
    XELOGE("MetalPipelineCache: DXIL->metallib failed for shader {:016X}: {}",
           shader.ucode_data_hash(), error);
    return false;
  }

  // Load onto the device as an MTLFunction.
  if (s_dump_xlat && translation.shader().ucode_data_hash() == s_dump_xlat) {
    std::string path =
        fmt::format("/tmp/xe_xlat_{:016X}_{:016X}.metallib", s_dump_xlat, translation.modification());
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
      fwrite(metallib.data(), 1, metallib.size(), f);
      fclose(f);
      XELOGW("MetalPipelineCache: dumped {} ({} bytes)", path, metallib.size());
    }
  }
  if (!translation.CreateMetalFunction(device_, metallib)) {
    XELOGE("MetalPipelineCache: MTLFunction creation failed for shader {:016X}",
           shader.ucode_data_hash());
    return false;
  }
  census_ok = true;
  return true;
}

namespace {
MTLBlendFactor ToMetalBlendFactor(MetalBlendFactor factor) {
  switch (factor) {
    case MetalBlendFactor::kZero:
      return MTLBlendFactorZero;
    case MetalBlendFactor::kOne:
      return MTLBlendFactorOne;
    case MetalBlendFactor::kSourceColor:
      return MTLBlendFactorSourceColor;
    case MetalBlendFactor::kOneMinusSourceColor:
      return MTLBlendFactorOneMinusSourceColor;
    case MetalBlendFactor::kSourceAlpha:
      return MTLBlendFactorSourceAlpha;
    case MetalBlendFactor::kOneMinusSourceAlpha:
      return MTLBlendFactorOneMinusSourceAlpha;
    case MetalBlendFactor::kDestinationColor:
      return MTLBlendFactorDestinationColor;
    case MetalBlendFactor::kOneMinusDestinationColor:
      return MTLBlendFactorOneMinusDestinationColor;
    case MetalBlendFactor::kDestinationAlpha:
      return MTLBlendFactorDestinationAlpha;
    case MetalBlendFactor::kOneMinusDestinationAlpha:
      return MTLBlendFactorOneMinusDestinationAlpha;
    case MetalBlendFactor::kBlendColor:
      return MTLBlendFactorBlendColor;
    case MetalBlendFactor::kOneMinusBlendColor:
      return MTLBlendFactorOneMinusBlendColor;
    case MetalBlendFactor::kBlendAlpha:
      return MTLBlendFactorBlendAlpha;
    case MetalBlendFactor::kOneMinusBlendAlpha:
      return MTLBlendFactorOneMinusBlendAlpha;
    case MetalBlendFactor::kSourceAlphaSaturated:
      return MTLBlendFactorSourceAlphaSaturated;
    default:
      return MTLBlendFactorZero;
  }
}

MTLBlendOperation ToMetalBlendOperation(MetalBlendOperation operation) {
  switch (operation) {
    case MetalBlendOperation::kAdd:
      return MTLBlendOperationAdd;
    case MetalBlendOperation::kSubtract:
      return MTLBlendOperationSubtract;
    case MetalBlendOperation::kMin:
      return MTLBlendOperationMin;
    case MetalBlendOperation::kMax:
      return MTLBlendOperationMax;
    case MetalBlendOperation::kReverseSubtract:
      return MTLBlendOperationReverseSubtract;
    default:
      return MTLBlendOperationAdd;
  }
}

// Applies one normalized RB_BLENDCONTROL + RB_COLOR_MASK nibble to a color-
// attachment descriptor. Shared by the standard render pipeline and the
// geometry-emulation mesh pipeline (both use
// MTLRenderPipelineColorAttachmentDescriptor).
void ApplyRtBlendState(MTLRenderPipelineColorAttachmentDescriptor* rt,
                       uint32_t blend_control, uint32_t color_write_mask) {
  MetalColorAttachmentBlendState state =
      GetMetalColorAttachmentBlendState(blend_control, color_write_mask);
  MTLColorWriteMask write_mask = MTLColorWriteMaskNone;
  if (state.write_mask & 0b0001) {
    write_mask |= MTLColorWriteMaskRed;
  }
  if (state.write_mask & 0b0010) {
    write_mask |= MTLColorWriteMaskGreen;
  }
  if (state.write_mask & 0b0100) {
    write_mask |= MTLColorWriteMaskBlue;
  }
  if (state.write_mask & 0b1000) {
    write_mask |= MTLColorWriteMaskAlpha;
  }
  rt.writeMask = write_mask;
  rt.blendingEnabled = state.blending_required;
  if (!state.blending_required) {
    return;
  }
  rt.sourceRGBBlendFactor = ToMetalBlendFactor(state.rgb.source_factor);
  rt.destinationRGBBlendFactor = ToMetalBlendFactor(state.rgb.destination_factor);
  rt.rgbBlendOperation = ToMetalBlendOperation(state.rgb.operation);
  rt.sourceAlphaBlendFactor = ToMetalBlendFactor(state.alpha.source_factor);
  rt.destinationAlphaBlendFactor = ToMetalBlendFactor(state.alpha.destination_factor);
  rt.alphaBlendOperation = ToMetalBlendOperation(state.alpha.operation);
}

// Maps a geometry-shader type to the MSC input topology the guest primitive
// feeds the emulation with (rect/quad expand from triangles, points from points),
// mirroring XeniOS's GetGeometryStageInputTopology.
MetalShaderConverter::InputTopology GeometryInputTopology(
    PipelineGeometryShader type) {
  switch (type) {
    case PipelineGeometryShader::kPointList:
      return MetalShaderConverter::InputTopology::kPoint;
    case PipelineGeometryShader::kRectangleList:
      return MetalShaderConverter::InputTopology::kTriangle;
    case PipelineGeometryShader::kQuadList:
      // The generated DXBC GS declares line-with-adjacency. MSC's input
      // topology describes the base primitive family; the adjacency arity
      // comes from the shader declaration itself.
      return MetalShaderConverter::InputTopology::kLine;
    default:
      return MetalShaderConverter::InputTopology::kUndefined;
  }
}
}  // namespace

void* MetalPipelineCache::GetOrCreateMemexportPipelineState(
    MetalShader::MetalTranslation* vertex) {
  // Memexport-only draws (VS eA/eM exports): Metal requires a void-returning
  // vertex function for rasterizationEnabled=NO, but MSC-translated shaders
  // return a stage-out struct — so rasterization stays ON into a throwaway
  // scratch depth attachment (DontCare load/store; the command processor's
  // memexport pass provides it). That makes this exactly the existing
  // depth-only pipeline variant: no fragment function, no color attachment.
  uint64_t color_pixel_formats[xenos::kMaxColorRenderTargets] = {};
  uint32_t blend_controls[xenos::kMaxColorRenderTargets] = {};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    color_pixel_formats[i] = uint64_t(MTLPixelFormatInvalid);
  }
  return GetOrCreateRenderPipelineState(
      vertex, /*pixel=*/nullptr, color_pixel_formats,
      MetalRenderTargetCache::kDepthFormat,
      MetalPipelineRasterKey{/*raster_sample_count=*/1},
      blend_controls, /*color_write_mask=*/0);
}

void* MetalPipelineCache::GetOrCreateRenderPipelineState(
    MetalShader::MetalTranslation* vertex,
    MetalShader::MetalTranslation* pixel, const uint64_t* color_pixel_formats,
    uint64_t depth_pixel_format, MetalPipelineRasterKey raster_key,
    const uint32_t* blend_controls,
    uint32_t color_write_mask) {
  if (!vertex || !vertex->has_metal_function()) {
    return nullptr;
  }

  // Key on the identity of the host functions plus the attachment formats. The
  // functions are stable for a given translation, so their pointers are a valid
  // identity within one run.
  struct KeyBytes {
    void* vs;
    void* ps;
    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint64_t depth;
    uint32_t sample_count;
    uint32_t blends[xenos::kMaxColorRenderTargets];
    uint32_t write_mask;
    uint32_t pad;
  } key_bytes = {};
  key_bytes.vs = vertex->metal_function();
  key_bytes.ps = pixel ? pixel->metal_function() : nullptr;
  key_bytes.depth = depth_pixel_format;
  key_bytes.sample_count = raster_key.raster_sample_count;
  key_bytes.write_mask = color_write_mask;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    key_bytes.colors[i] = color_pixel_formats[i];
    key_bytes.blends[i] = blend_controls[i];
  }
  uint64_t key = XXH3_64bits(&key_bytes, sizeof(key_bytes));
  auto it = render_pipeline_states_.find(key);
  if (it != render_pipeline_states_.end()) {
    return it->second;
  }

  // CENSUS (temporary; see metal_pipeline_census.h). Cache miss only: this is
  // the wall time the frame that first wants this pipeline actually pays, which
  // is descriptor assembly plus the synchronous host creation call.
  bool census_ok = false;
  pipeline_census::Scope census_scope(pipeline_census::Event::kRenderPipeline,
                                      &census_ok);

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLRenderPipelineDescriptor* desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  desc.rasterSampleCount = raster_key.raster_sample_count;
  // Xenos alpha-to-mask is emitted as exact dynamic SV_Coverage by the reused
  // DXBC translator. Enabling Metal alpha-to-coverage here would apply a
  // second, host-defined coverage filter and would incorrectly require alpha
  // state in the pipeline key.
  desc.alphaToCoverageEnabled = NO;
  desc.vertexFunction = (__bridge id<MTLFunction>)vertex->metal_function();
  if (pixel && pixel->has_metal_function()) {
    desc.fragmentFunction = (__bridge id<MTLFunction>)pixel->metal_function();
  }
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (color_pixel_formats[i] == MTLPixelFormatInvalid) {
      continue;
    }
    desc.colorAttachments[i].pixelFormat =
        (MTLPixelFormat)color_pixel_formats[i];
    // Per-target blending and write mask from RB_BLENDCONTROL[0-3] and the
    // corresponding RB_COLOR_MASK nibble.
    ApplyRtBlendState(desc.colorAttachments[i], blend_controls[i],
                      (color_write_mask >> (i * 4u)) & 0xFu);
  }
  if (depth_pixel_format != MTLPixelFormatInvalid) {
    desc.depthAttachmentPixelFormat = (MTLPixelFormat)depth_pixel_format;
    if ((MTLPixelFormat)depth_pixel_format ==
        MTLPixelFormatDepth32Float_Stencil8) {
      desc.stencilAttachmentPixelFormat = (MTLPixelFormat)depth_pixel_format;
    }
  }

  NSError* error = nil;
  id<MTLRenderPipelineState> state =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
  if (!state) {
    XELOGE("MetalPipelineCache: newRenderPipelineState failed: {}",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    render_pipeline_states_.emplace(key, nullptr);
    return nullptr;
  }
  void* state_ptr = (__bridge_retained void*)state;
  render_pipeline_states_.emplace(key, state_ptr);
  census_ok = true;
  return state_ptr;
}

void* MetalPipelineCache::NewLibraryFromBytes(const std::vector<uint8_t>& bytes,
                                              const char* label) {
  if (!device_ || bytes.empty()) {
    return nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  dispatch_data_t data = dispatch_data_create(
      bytes.data(), bytes.size(), dispatch_get_main_queue(),
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
#if !__has_feature(objc_arc)
  dispatch_release(data);
#endif
  if (!library) {
    XELOGE("MetalPipelineCache: {} newLibraryWithData failed: {}",
           label ? label : "library",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    return nullptr;
  }
  return (__bridge_retained void*)library;
}

const MetalPipelineCache::GeometryVertexStage*
MetalPipelineCache::GetOrCreateGeometryVertexStage(
    MetalShader::MetalTranslation* vertex, PipelineGeometryShader geometry_type) {
  auto it = geometry_vertex_stages_.find(vertex);
  if (it != geometry_vertex_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  // Cache the result even on failure so a bad shader isn't retried every draw.
  GeometryVertexStage stage;

  // CENSUS (temporary; see metal_pipeline_census.h). Cache miss only.
  bool census_ok = false;
  pipeline_census::Scope census_scope(
      pipeline_census::Event::kGeometryVertexStage, &census_ok);

  // Reuse the DXBC the normal VS translation already produced (the guest VS runs
  // as the object stage with host_vertex_shader_type kVertex — geometry emulation
  // does the primitive expansion in the GS/mesh stage, so the VS DXBC is
  // identical to the normal path; see XeniOS host-VS-type remap).
  const std::vector<uint8_t>& dxbc = vertex->translated_binary();
  if (dxbc.empty()) {
    XELOGE("MetalPipelineCache: geometry VS has no DXBC (translate VS first)");
    geometry_vertex_stages_.emplace(vertex, std::move(stage));
    return nullptr;
  }
  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: geometry VS DXBC->DXIL failed: {}", error);
    geometry_vertex_stages_.emplace(vertex, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kVertex, dxil,
          GeometryInputTopology(geometry_type), result)) {
    XELOGE("MetalPipelineCache: geometry VS emulation convert failed: {}",
           result.error.empty() ? "unknown" : result.error);
    geometry_vertex_stages_.emplace(vertex, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "geometry VS");
  void* stage_in =
      NewLibraryFromBytes(result.stage_in_metallib, "geometry VS stage-in");
  if (!library || !stage_in) {
    if (library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
      (void)lib;
    }
    if (stage_in) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)stage_in;
      (void)lib;
    }
    geometry_vertex_stages_.emplace(vertex, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.stage_in_library = stage_in;
  stage.function_name = result.function_name;
  stage.vertex_output_size = result.vertex_output_size_in_bytes;
  stage.valid = true;
  auto inserted = geometry_vertex_stages_.emplace(vertex, std::move(stage));
  census_ok = true;
  return &inserted.first->second;
}

const MetalPipelineCache::GeometryShaderStage*
MetalPipelineCache::GetOrCreateGeometryShaderStage(GeometryShaderKey key) {
  auto it = geometry_shader_stages_.find(key.key);
  if (it != geometry_shader_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  GeometryShaderStage stage;

  // CENSUS (temporary; see metal_pipeline_census.h). Cache miss only.
  bool census_ok = false;
  pipeline_census::Scope census_scope(
      pipeline_census::Event::kGeometryShaderStage, &census_ok);

  // Generate the DXBC geometry shader (shared point/rect/quad generator), then
  // run it through the same DXBC->DXIL->MSC chain as a MESH stage.
  std::vector<uint32_t> dxbc_dwords;
  CreateDxbcGeometryShader(key, dxbc_dwords);
  if (dxbc_dwords.empty()) {
    XELOGE("MetalPipelineCache: geometry shader generator produced no DXBC");
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  std::vector<uint8_t> dxbc_bytes(dxbc_dwords.size() * sizeof(uint32_t));
  std::memcpy(dxbc_bytes.data(), dxbc_dwords.data(), dxbc_bytes.size());

  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc_bytes, dxil, &error)) {
    XELOGE("MetalPipelineCache: geometry GS DXBC->DXIL failed: {}", error);
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kGeometry, dxil,
          GeometryInputTopology(static_cast<PipelineGeometryShader>(key.type)),
          result) ||
      (!result.has_mesh_stage && !result.has_geometry_stage)) {
    XELOGE("MetalPipelineCache: geometry GS emulation convert failed: {}",
           result.error.empty() ? "no mesh/geometry stage" : result.error);
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "geometry GS");
  if (!library) {
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.function_name = result.function_name;
  stage.max_input_primitives = result.gs_max_input_primitives_per_mesh_threadgroup;
  stage.valid = true;
  auto inserted = geometry_shader_stages_.emplace(key.key, std::move(stage));
  census_ok = true;
  return &inserted.first->second;
}

void* MetalPipelineCache::GetOrCreateGeometryPipelineState(
    MetalShader::MetalTranslation* vertex, MetalShader::MetalTranslation* pixel,
    PipelineGeometryShader geometry_type,
    DxbcShaderTranslator::Modification vertex_modification,
    DxbcShaderTranslator::Modification pixel_modification,
    const uint64_t* color_pixel_formats, uint64_t depth_pixel_format,
    MetalPipelineRasterKey raster_key, const uint32_t* blend_controls,
    uint32_t color_write_mask,
    uint32_t* gs_vertex_size_out, uint32_t* gs_max_input_primitives_out) {
  if (gs_vertex_size_out) {
    *gs_vertex_size_out = 0;
  }
  if (gs_max_input_primitives_out) {
    *gs_max_input_primitives_out = 0;
  }
  // The geometry-emulation pipeline requires a fragment stage (the descriptor
  // dereferences fragmentLibrary); a depth-only geometry path isn't wired yet.
  if (!vertex || !pixel || !pixel->has_metal_function() ||
      geometry_type == PipelineGeometryShader::kNone) {
    return nullptr;
  }

  // The generated GS is keyed by the interpolator/clip layout derived from the
  // VS/PS modifications; returns false if this type needs no geometry shader.
  GeometryShaderKey gs_key;
  if (!GetGeometryShaderKey(geometry_type, vertex_modification,
                            pixel_modification, gs_key)) {
    return nullptr;
  }

  struct KeyBytes {
    uint32_t gs_key;
    void* vs;  // vertex translation identity (object-stage source).
    void* ps;  // fragment function identity.
    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint64_t depth;
    uint32_t sample_count;
    uint32_t blends[xenos::kMaxColorRenderTargets];
    uint32_t write_mask;
    uint32_t pad;
  } key_bytes = {};
  key_bytes.gs_key = gs_key.key;
  key_bytes.vs = static_cast<void*>(vertex);
  key_bytes.ps = pixel->metal_function();
  key_bytes.depth = depth_pixel_format;
  key_bytes.sample_count = raster_key.raster_sample_count;
  key_bytes.write_mask = color_write_mask;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    key_bytes.colors[i] = color_pixel_formats[i];
    key_bytes.blends[i] = blend_controls[i];
  }
  uint64_t key = XXH3_64bits(&key_bytes, sizeof(key_bytes));
  auto cache_it = geometry_pipelines_.find(key);
  if (cache_it != geometry_pipelines_.end()) {
    if (!cache_it->second.pipeline) {
      return nullptr;
    }
    if (gs_vertex_size_out) {
      *gs_vertex_size_out = cache_it->second.gs_vertex_size;
    }
    if (gs_max_input_primitives_out) {
      *gs_max_input_primitives_out = cache_it->second.gs_max_input_primitives;
    }
    return cache_it->second.pipeline;
  }

  const GeometryVertexStage* vstage =
      GetOrCreateGeometryVertexStage(vertex, geometry_type);
  const GeometryShaderStage* gstage = GetOrCreateGeometryShaderStage(gs_key);
  if (!vstage || !gstage || !vstage->vertex_output_size ||
      !gstage->max_input_primitives) {
    XELOGE(
        "MetalPipelineCache: geometry pipeline missing stage/reflection "
        "(vs_output={}, gs_max_input={})",
        vstage ? vstage->vertex_output_size : 0,
        gstage ? gstage->max_input_primitives : 0);
    geometry_pipelines_.emplace(key, GeometryPipeline{});
    return nullptr;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLMeshRenderPipelineDescriptor* desc =
      [[MTLMeshRenderPipelineDescriptor alloc] init];
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (color_pixel_formats[i] == MTLPixelFormatInvalid) {
      continue;
    }
    desc.colorAttachments[i].pixelFormat =
        (MTLPixelFormat)color_pixel_formats[i];
    ApplyRtBlendState(desc.colorAttachments[i], blend_controls[i],
                      (color_write_mask >> (i * 4u)) & 0xFu);
  }
  if (depth_pixel_format != MTLPixelFormatInvalid) {
    desc.depthAttachmentPixelFormat = (MTLPixelFormat)depth_pixel_format;
    if ((MTLPixelFormat)depth_pixel_format ==
        MTLPixelFormatDepth32Float_Stencil8) {
      desc.stencilAttachmentPixelFormat = (MTLPixelFormat)depth_pixel_format;
    }
  }
  desc.rasterSampleCount = raster_key.raster_sample_count;
  // The fragment function writes the complete Xenos coverage mask dynamically.
  // Keep the host coverage filter disabled for the same reason as the regular
  // render-pipeline path above.
  desc.alphaToCoverageEnabled = NO;

  id<MTLFunction> ps_fn = (__bridge id<MTLFunction>)pixel->metal_function();
  std::string pixel_function_name = ps_fn.name ? [ps_fn.name UTF8String] : "";

  IRGeometryEmulationPipelineDescriptor ir_desc = {};
  ir_desc.stageInLibrary = (__bridge id<MTLLibrary>)vstage->stage_in_library;
  ir_desc.vertexLibrary = (__bridge id<MTLLibrary>)vstage->library;
  ir_desc.vertexFunctionName = vstage->function_name.c_str();
  ir_desc.geometryLibrary = (__bridge id<MTLLibrary>)gstage->library;
  ir_desc.geometryFunctionName = gstage->function_name.c_str();
  ir_desc.fragmentLibrary = (__bridge id<MTLLibrary>)pixel->metal_library();
  ir_desc.fragmentFunctionName = pixel_function_name.c_str();
  ir_desc.basePipelineDescriptor = desc;
  ir_desc.pipelineConfig.gsVertexSizeInBytes = vstage->vertex_output_size;
  ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
      gstage->max_input_primitives;

  NSError* error = nil;
  id<MTLRenderPipelineState> pipeline = nil;
  {
    // CENSUS (temporary; see metal_pipeline_census.h). Deliberately narrow: the
    // object/mesh stage builds above have their own scopes, so a wider scope
    // here would double-count their time in the run total.
    bool census_ok = false;
    pipeline_census::Scope census_scope(
        pipeline_census::Event::kGeometryPipeline, &census_ok);
    pipeline = IRRuntimeNewGeometryEmulationPipeline(device, &ir_desc, &error);
    census_ok = pipeline != nil;
  }
  if (!pipeline) {
    XELOGE("MetalPipelineCache: geometry pipeline creation failed: {}",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    geometry_pipelines_.emplace(key, GeometryPipeline{});
    return nullptr;
  }

  XELOGI(
      "MetalPipelineCache: built geometry pipeline (type={}, gs_vertex_size={}, "
      "gs_max_input_primitives={}, vs_fn='{}', gs_fn='{}')",
      uint32_t(geometry_type), vstage->vertex_output_size,
      gstage->max_input_primitives, vstage->function_name,
      gstage->function_name);

  GeometryPipeline state;
  state.pipeline = (__bridge_retained void*)pipeline;
  state.gs_vertex_size = vstage->vertex_output_size;
  state.gs_max_input_primitives = gstage->max_input_primitives;
  if (gs_vertex_size_out) {
    *gs_vertex_size_out = state.gs_vertex_size;
  }
  if (gs_max_input_primitives_out) {
    *gs_max_input_primitives_out = state.gs_max_input_primitives;
  }
  auto inserted = geometry_pipelines_.emplace(key, std::move(state));
  return inserted.first->second.pipeline;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
