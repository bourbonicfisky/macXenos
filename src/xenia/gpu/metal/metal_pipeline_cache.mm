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
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/metal/metal_dxbc_clip_relabel.h"
#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

// Metal Shader Converter runtime (declarations only; implementation TU is
// ir_runtime_impl.mm). Provides IRGeometryEmulationPipelineDescriptor and
// IRRuntimeNewGeometryEmulationPipeline for the geometry-emulation mesh pipeline,
// plus IRGeometryTessellationEmulationPipelineDescriptor /
// IRRuntimeValidateTessellationPipeline /
// IRRuntimeNewGeometryTessellationEmulationPipeline for the tessellation leg.
#include "metal_irconverter_runtime.h"

// Prebuilt SM5.1 tessellation bytecode (port-xenios-tessellation task 1.2
// import — tracked location; see d3d12_5_1_bytecode/README.md for provenance
// and the regeneration recipe): 10 hull shaders (mode x domain x control-point
// count) + 2 tessellation vertex shaders (indexed / adaptive).
using BYTE = uint8_t;
#include "xenia/gpu/metal/d3d12_5_1_bytecode/adaptive_quad_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/adaptive_triangle_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_quad_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_quad_4cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_triangle_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/continuous_triangle_3cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_quad_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_quad_4cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_1cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/discrete_triangle_3cp_hs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/tessellation_adaptive_vs.h"
#include "xenia/gpu/metal/d3d12_5_1_bytecode/tessellation_indexed_vs.h"

// Defined in graphics_system.cc. Gates GraphicsSystem::InitializeShaderStorage
// before any backend is consulted, so it silently disables both durable
// mechanisms; reported in the effective-state line below.
DECLARE_bool(store_shaders);

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
  // Configuration comes from MetalTranslatorConfig so the durable cache's
  // schema fingerprint (3.3b) cannot drift from what is actually built here.
  shader_translator_ = std::make_unique<DxbcShaderTranslator>(
      ui::GraphicsProvider::GpuVendorID::kApple,
      MetalTranslatorConfig::kBindlessResourcesUsed,
      MetalTranslatorConfig::kEdramRovUsed,
      MetalTranslatorConfig::kGammaRenderTargetAsUnorm8);

  std::string error;
  if (!dxil_converter_.Initialize(&error)) {
    XELOGE("MetalPipelineCache: DXBC->DXIL converter init failed: {}", error);
    return false;
  }
  if (!metal_converter_.Initialize(
          MetalTranslatorConfig::kMscBindlessResourcesUsed, &error)) {
    XELOGE("MetalPipelineCache: Metal Shader Converter init failed: {}", error);
    return false;
  }

  // EFFECTIVE durable-cache state, logged unconditionally so a gate can assert
  // on it instead of trusting that a flag was applied.
  //
  // Two ways a requested mode silently fails to take effect, both hit in
  // practice: `xenia_canary` reads a persisted config that OVERRIDES
  // command-line cvars (while the trace-dump tool has no config, so there the
  // command line wins), and `store_shaders=false` makes
  // GraphicsSystem::InitializeShaderStorage return before any backend is
  // consulted, disabling BOTH mechanisms no matter what their own switches say.
  // Either one produces a run that looks like the mechanism is on and measures
  // as though it is off.
  XELOGI(
      "MetalPipelineCache: durable cache EFFECTIVE state — M1 metallib={}, "
      "M2 archive={}, store_shaders={}{}",
      cvars::metal_pipeline_metallib_cache ? "on" : "off",
      cvars::metal_pipeline_binary_archive ? "on" : "off",
      cvars::store_shaders ? "on" : "off",
      cvars::store_shaders
          ? ""
          : "  <- store_shaders=off DISABLES BOTH regardless of the switches");
  return true;
}

void MetalPipelineCache::InitializeDurableStorage(
    const std::filesystem::path& cache_root, uint32_t title_id) {
  ShutdownDurableStorage();
  // 3.1 — exact rollback switches, default OFF. Neither mechanism consumes the
  // storage yet (that is 3b/3c); opening it here is what makes the identity,
  // bounds, and locking path live rather than dormant.
  if (!cvars::metal_pipeline_metallib_cache &&
      !cvars::metal_pipeline_binary_archive) {
    return;
  }
  MetalPipelineStorageIdentity identity =
      ComputeMetalPipelineStorageIdentity(device_);
  if (!storage_.Open(cache_root, title_id, identity)) {
    XELOGW(
        "MetalPipelineCache: durable storage unavailable for title {:08X}; "
        "continuing with in-memory caches only",
        title_id);
  }
}

void MetalPipelineCache::ShutdownDurableStorage() { storage_.Close(); }

void MetalPipelineCache::Shutdown() {
  ShutdownDurableStorage();
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
  // Release tessellation pipelines + stage libraries + the fallback PS.
  for (auto& pair : tessellation_pipelines_) {
    if (pair.second.pipeline) {
      id<MTLRenderPipelineState> state =
          (__bridge_transfer id<MTLRenderPipelineState>)pair.second.pipeline;
      (void)state;
    }
  }
  tessellation_pipelines_.clear();
  for (auto& pair : tessellation_vertex_stages_) {
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
  tessellation_vertex_stages_.clear();
  for (auto& pair : tessellation_hull_stages_) {
    if (pair.second.library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)pair.second.library;
      (void)lib;
    }
  }
  tessellation_hull_stages_.clear();
  for (auto& pair : tessellation_domain_stages_) {
    if (pair.second.library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)pair.second.library;
      (void)lib;
    }
  }
  tessellation_domain_stages_.clear();
  if (depth_only_pixel_library_) {
    id<MTLLibrary> lib =
        (__bridge_transfer id<MTLLibrary>)depth_only_pixel_library_;
    (void)lib;
    depth_only_pixel_library_ = nullptr;
  }
  depth_only_pixel_function_name_.clear();
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

namespace {
// M1 entry key. Same content-derived identity the pipeline key uses, hashed to
// a filename. The storage-wide provenance (device, OS, translator artifacts and
// schema) was validated when the directory was opened, so it is not repeated
// per entry.
uint64_t MetallibCacheKey(const MetalShader::MetalTranslation& translation) {
  struct KeyBytes {
    uint64_t ucode_hash;
    uint64_t modification;
    uint32_t shader_type;
    uint32_t pad;
  } key_bytes = {};
  key_bytes.ucode_hash = translation.shader().ucode_data_hash();
  key_bytes.modification = translation.modification();
  key_bytes.shader_type = uint32_t(translation.shader().type());
  return XXH3_64bits(&key_bytes, sizeof(key_bytes));
}

}  // namespace

bool MetalPipelineCache::TranslateShader(
    MetalShader::MetalTranslation& translation) {
  if (translation.is_translated()) {
    return translation.is_valid() && translation.has_metal_function();
  }

  MetalShader& shader = static_cast<MetalShader&>(translation.shader());


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

  // M1 (3.7/3.8) — durable metallib store. Stage 1 above always runs even on a
  // hit: it costs 0.8-0.9% of translation time (~0.06 ms mean, measured in
  // task 3.9) and it keeps translated_binary() populated, which the
  // geometry-emulation stages consume. Skipping it to save that would break
  // geometry emulation for a saving smaller than the measurement noise.
  //
  // What a hit DOES skip is stages 2 and 3 — DXBC->DXIL and DXIL->metallib —
  // which are 99.7-99.8% of translation cost.
  const uint64_t m1_key = MetallibCacheKey(translation);
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    std::vector<uint8_t> cached_metallib;
    if (storage_.LoadMetallib(m1_key, &cached_metallib)) {
      if (translation.CreateMetalFunction(device_, cached_metallib)) {
        return true;
      }
      // A stored blob the device will not load is not trustworthy — fall
      // through to a full cold translation rather than failing the draw.
      XELOGW(
          "MetalPipelineCache: cached metallib {:016X} failed to load; "
          "retranslating",
          m1_key);
    }
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

  // Publish only after the device has accepted it, so a blob that cannot load
  // is never written for a later run to trip over.
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    storage_.StoreMetallib(m1_key, metallib);
  }
  return true;
}

namespace {

// 3.2 — content-derived shader identity for the pipeline key.
//
// The key previously used MTLFunction / translation POINTERS. Those are stable
// within one run and meaningless across runs, so no durable entry could be
// written against them. (ucode hash, modification, type) is the equivalent
// identity in content terms: MetalShader owns exactly one translation per
// modification and TranslateShader is idempotent per translation, so the
// pointer and this triple are 1:1 within a run — the swap cannot merge two
// distinct pipelines or split one.
//
// `present` keeps a null pixel shader distinct from any fragment-carrying
// pipeline, which the spec requires and which the old key got from a null
// pointer. A real shader can never collide with the absent case because
// present=0 zeroes the whole triple.
struct MetalShaderKeyIdentity {
  uint64_t ucode_hash;
  uint64_t modification;
  uint32_t shader_type;
  uint32_t present;
};

MetalShaderKeyIdentity MakeShaderKeyIdentity(
    const MetalShader::MetalTranslation* translation) {
  MetalShaderKeyIdentity identity = {};
  if (!translation) {
    return identity;
  }
  identity.ucode_hash = translation->shader().ucode_data_hash();
  identity.modification = translation->modification();
  identity.shader_type = uint32_t(translation->shader().type());
  identity.present = 1;
  return identity;
}

// The COMPLETE pipeline-creation identity for a geometry-emulation pipeline —
// the exact bytes hashed into its cache key. Everything MTLRenderPipelineState
// creation consumes is here: shader identities, every MRT format, blend
// controls, write mask, depth (and thereby stencil: Depth32Float_Stencil8
// implies the stencil attachment) format, and sample count.
//
// Depth/stencil ENABLE/WRITE state is deliberately absent: on Metal that lives
// in MTLDepthStencilState, set per-encoder, and does not participate in
// pipeline creation — logging it here would imply a discriminator that cannot
// exist at this boundary.
struct GeometryPipelineKeyBytes {
  uint32_t gs_key;
  uint32_t pad0;
  MetalShaderKeyIdentity vs;  // object-stage source translation.
  MetalShaderKeyIdentity ps;  // fragment stage.
  uint64_t colors[xenos::kMaxColorRenderTargets];
  uint64_t depth;
  uint32_t sample_count;
  uint32_t blends[xenos::kMaxColorRenderTargets];
  uint32_t write_mask;
  uint32_t pad;
};

// One formatter for BOTH the success and failure logs, so they cannot drift
// apart: the first attribution round found the same geometry triple both
// succeeding and failing, and the depth-format correlation could not be tested
// precisely because successes logged fewer fields than failures.
std::string DescribeGeometryPipelineKey(uint64_t key_digest,
                                        const GeometryPipelineKeyBytes& kb) {
  return fmt::format(
      "key={:016X} gs_key={:08X} vs={:016X}/{:08X} ps={:016X}/{:08X} "
      "colors=[{:X},{:X},{:X},{:X}] blends=[{:08X},{:08X},{:08X},{:08X}] "
      "write_mask={:X} depth_fmt={} samples={}",
      key_digest, kb.gs_key, kb.vs.ucode_hash, uint32_t(kb.vs.modification),
      kb.ps.ucode_hash, uint32_t(kb.ps.modification), kb.colors[0],
      kb.colors[1], kb.colors[2], kb.colors[3], kb.blends[0], kb.blends[1],
      kb.blends[2], kb.blends[3], kb.write_mask, kb.depth, kb.sample_count);
}

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

  // 3.2 — full pipeline identity, content-derived so it can back a durable
  // entry. Audit of every input that reaches the descriptor below:
  //   shader identities + modifications  -> vs / ps (MakeShaderKeyIdentity)
  //   null-vs-present pixel shader       -> ps.present
  //   render-target formats              -> colors
  //   depth format                       -> depth (the stencil format is
  //                                         derived from it, never independent)
  //   blend state                        -> blends
  //   write mask                         -> write_mask
  //   sample state                       -> sample_count
  //   vertex layout                      -> NOT APPLICABLE: this backend never
  //                                         sets a vertexDescriptor; MSC-
  //                                         translated shaders fetch vertices
  //                                         themselves, so no layout reaches
  //                                         the pipeline
  //   alphaToCoverageEnabled             -> constant NO (Xenos alpha-to-mask is
  //                                         exact SV_Coverage in the shader),
  //                                         so it is not an identity input
  struct KeyBytes {
    MetalShaderKeyIdentity vs;
    MetalShaderKeyIdentity ps;
    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint64_t depth;
    uint32_t sample_count;
    uint32_t blends[xenos::kMaxColorRenderTargets];
    uint32_t write_mask;
    uint32_t pad;
  } key_bytes = {};
  key_bytes.vs = MakeShaderKeyIdentity(vertex);
  // A pixel translation without a function produces the same depth-only
  // pipeline as no pixel translation at all (see the descriptor below), so it
  // must key the same. The old pointer key got this for free because
  // metal_function() was null in both cases.
  key_bytes.ps = MakeShaderKeyIdentity(
      pixel && pixel->has_metal_function() ? pixel : nullptr);
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
  // The object stage is compiled against the MSC input topology that
  // `geometry_type` selects (point -> kPoint, rect -> kTriangle, quad ->
  // kLine), so two geometry types produce DIFFERENT object stages for the same
  // guest VS. This cache used to key on `vertex` alone and would hand back
  // whichever was built first; the topology is now part of the key. Found while
  // building the durable key for 3.10, where persisting the same mistake would
  // have made it survive across runs instead of only within one.
  const MetalGeometryVertexStageKey stage_key{
      static_cast<const void*>(vertex),
      uint32_t(GeometryInputTopology(geometry_type))};
  auto it = geometry_vertex_stages_.find(stage_key);
  if (it != geometry_vertex_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  // Cache the result even on failure so a bad shader isn't retried every draw.
  GeometryVertexStage stage;

  // 3.10 — durable geometry record. Keyed on every input that affects object-
  // stage generation: the guest shader identity, its translation modification,
  // and the MSC input topology.
  const uint64_t record_key =
      MetalGeometryRecordKey(kMetalGeometryRecordKindVertexStage,
                        MetallibCacheKey(*vertex),
                        uint32_t(GeometryInputTopology(geometry_type)));
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    MetalPipelineStorage::GeometryRecord record;
    if (storage_.LoadGeometryRecord(record_key, &record) &&
        !record.aux_blob.empty()) {
      void* library = NewLibraryFromBytes(record.blob, "geometry VS (cached)");
      void* stage_in =
          NewLibraryFromBytes(record.aux_blob, "geometry VS stage-in (cached)");
      if (library && stage_in) {
        stage.library = library;
        stage.stage_in_library = stage_in;
        stage.function_name = record.function_name;
        stage.vertex_output_size = record.reflection;
        stage.valid = true;
        auto inserted = geometry_vertex_stages_.emplace(stage_key, std::move(stage));
        return &inserted.first->second;
      }
      // Either library failing makes the record unusable as a whole — release
      // whichever loaded and fall through to a full cold build.
      if (library) {
        id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
        (void)lib;
      }
      if (stage_in) {
        id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)stage_in;
        (void)lib;
      }
      XELOGW(
          "MetalPipelineCache: cached geometry VS record {:016X} failed to "
          "load; rebuilding",
          record_key);
    }
  }

  // Reuse the DXBC the normal VS translation already produced (the guest VS runs
  // as the object stage with host_vertex_shader_type kVertex — geometry emulation
  // does the primitive expansion in the GS/mesh stage, so the VS DXBC is
  // identical to the normal path; see XeniOS host-VS-type remap).
  const std::vector<uint8_t>& dxbc_source = vertex->translated_binary();
  if (dxbc_source.empty()) {
    XELOGE("MetalPipelineCache: geometry VS has no DXBC (translate VS first)");
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  // Clip-distance semantic legalization, UNCONDITIONAL. Relabels the
  // clip-distance semantic on the OBJECT STAGE'S OUTPUT so it crosses the
  // object->mesh hop as a private user parameter; without it a geometry-
  // emulation pipeline whose VS writes SV_ClipDistance fails to create and the
  // whole draw class is silently dropped.
  // Inert for clip-count-0 shaders: proven BYTE-IDENTICAL in and out by the
  // transport oracle, which is what makes running it on every geometry stage
  // safe rather than merely convenient.
  std::vector<uint8_t> dxbc = dxbc_source;
  {
    uint32_t n = RelabelDxbcClipDistance(dxbc, DxbcSignatureKind::kOutput);
    if (n) {
      XELOGI("MetalPipelineCache: clip relabel — object-stage output, {} element(s)", n);
    }
  }
  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: geometry VS DXBC->DXIL failed: {}", error);
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kVertex, dxil,
          GeometryInputTopology(geometry_type), result)) {
    XELOGE("MetalPipelineCache: geometry VS emulation convert failed: {}",
           result.error.empty() ? "unknown" : result.error);
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
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
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.stage_in_library = stage_in;
  stage.function_name = result.function_name;
  stage.vertex_output_size = result.vertex_output_size_in_bytes;
  stage.valid = true;
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open() &&
      stage.vertex_output_size) {
    MetalPipelineStorage::GeometryRecord record;
    record.function_name = stage.function_name;
    record.reflection = stage.vertex_output_size;
    record.blob = result.metallib;
    record.aux_blob = result.stage_in_metallib;
    storage_.StoreGeometryRecord(record_key, record);
  }
  auto inserted = geometry_vertex_stages_.emplace(stage_key, std::move(stage));
  return &inserted.first->second;
}

const MetalPipelineCache::GeometryShaderStage*
MetalPipelineCache::GetOrCreateGeometryShaderStage(GeometryShaderKey key) {
  auto it = geometry_shader_stages_.find(key.key);
  if (it != geometry_shader_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  GeometryShaderStage stage;

  // 3.10 — durable geometry record. The generated GS is fully determined by
  // GeometryShaderKey (topology type plus the interpolator/clip layout derived
  // from the VS/PS modifications), so that key IS the identity.
  const uint64_t record_key =
      MetalGeometryRecordKey(kMetalGeometryRecordKindShaderStage, key.key, 0);
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    MetalPipelineStorage::GeometryRecord record;
    if (storage_.LoadGeometryRecord(record_key, &record)) {
      void* library = NewLibraryFromBytes(record.blob, "geometry GS (cached)");
      if (library) {
        stage.library = library;
        stage.function_name = record.function_name;
        stage.max_input_primitives = record.reflection;
        stage.valid = true;
        auto inserted =
            geometry_shader_stages_.emplace(key.key, std::move(stage));
        return &inserted.first->second;
      }
      XELOGW(
          "MetalPipelineCache: cached geometry GS record {:016X} failed to "
          "load; rebuilding",
          record_key);
    }
  }

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
  // Matching relabel on the MESH STAGE'S INPUT. Its rasterizer-facing OUTPUT
  // deliberately keeps SV_ClipDistance, which is what makes the hardware clip.
  // Unconditional; inert byte-for-byte when the shader has no clip distances.
  {
    {
      uint32_t n =
          RelabelDxbcClipDistance(dxbc_bytes, DxbcSignatureKind::kInput);
      if (n) {
        XELOGI("MetalPipelineCache: clip relabel — mesh-stage input, {} element(s)", n);
      }
    }
  }

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
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open() &&
      stage.max_input_primitives) {
    MetalPipelineStorage::GeometryRecord record;
    record.function_name = stage.function_name;
    record.reflection = stage.max_input_primitives;
    record.blob = result.metallib;  // No stage-in for a mesh stage.
    storage_.StoreGeometryRecord(record_key, record);
  }
  auto inserted = geometry_shader_stages_.emplace(key.key, std::move(stage));
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

  // 3.2 — same content-derived identity as the render-pipeline key. `gs_key` is
  // already content-derived (topology type plus the interpolator/clip layout
  // taken from the VS/PS modifications), so it needs no change.
  GeometryPipelineKeyBytes key_bytes = {};
  key_bytes.gs_key = gs_key.key;
  key_bytes.vs = MakeShaderKeyIdentity(vertex);
  key_bytes.ps = MakeShaderKeyIdentity(pixel);
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
      // A cached negative is not "one failure": every hit here is a DRAW the
      // failed creation is still suppressing. Count them separately from
      // creation attempts so impact is visible; log the first suppression and
      // then powers of two, joined to the failure line by the key digest.
      uint32_t count = ++cache_it->second.negative_hits;
      if ((count & (count - 1)) == 0) {
        XELOGW(
            "MetalPipelineCache: geometry pipeline negative-cache hit #{} "
            "(draws suppressed by one failed creation) key={:016X}",
            count, key);
      }
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
  id<MTLRenderPipelineState> pipeline =
      IRRuntimeNewGeometryEmulationPipeline(device, &ir_desc, &error);
  if (!pipeline) {
    // Report the SAME identifying fields the success path logs. Without them a
    // failure is an unattributable string: it names neither which geometry type
    // failed, nor which guest shaders, nor the two pipeline-config values that
    // are the usual cause of a converter rejection. A negative cached below is
    // permanent for the run, so a draw class is silently dropped for the rest
    // of the session — that must be diagnosable from one log.
    XELOGE(
        "MetalPipelineCache: geometry pipeline creation failed: {} "
        "(type={}, gs_vertex_size={}, gs_max_input_primitives={}, "
        "vs_fn='{}', gs_fn='{}', ps_fn='{}') {}",
        error ? [[error localizedDescription] UTF8String] : "unknown",
        uint32_t(geometry_type), vstage->vertex_output_size,
        gstage->max_input_primitives, vstage->function_name,
        gstage->function_name, pixel_function_name,
        DescribeGeometryPipelineKey(key, key_bytes));
    geometry_pipelines_.emplace(key, GeometryPipeline{});
    return nullptr;
  }

  XELOGI(
      "MetalPipelineCache: built geometry pipeline (type={}, gs_vertex_size={}, "
      "gs_max_input_primitives={}, vs_fn='{}', gs_fn='{}') {}",
      uint32_t(geometry_type), vstage->vertex_output_size,
      gstage->max_input_primitives, vstage->function_name,
      gstage->function_name, DescribeGeometryPipelineKey(key, key_bytes));

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

// ---------------------------------------------------------------------------
// Tessellation pipeline family (port-xenios-tessellation task 3.1).
//
// TOPOLOGY DECISION: the tess VS, HS and DS are all compiled with
// InputTopology::kUndefined (the IRInputTopologyUndefined equivalent — our
// converter then never calls IRCompilerSetInputTopology). This is XeniOS's
// field-proven shape (its metal_pipeline_cache.cc:794 for the VS, :943 for the
// HS, :988 for the DS). The task-2.2 probe proved kPatch ALSO loads for the
// tess VS, but we match the reference to keep parity with its on-device
// history.
// ---------------------------------------------------------------------------

namespace {

struct TessellationBlob {
  const uint8_t* bytes = nullptr;
  size_t size = 0;
  const char* name = "";
};

// VS selection (XeniOS metal_pipeline_cache.cc:773-781): kAdaptive ->
// tessellation_adaptive_vs (factor stream in the index buffer), everything
// else -> tessellation_indexed_vs. Blob identity for the compile-once cache:
// 0 = indexed (shared by discrete and continuous), 1 = adaptive.
constexpr uint32_t kTessellationVertexBlobIndexed = 0;
constexpr uint32_t kTessellationVertexBlobAdaptive = 1;

uint32_t TessellationVertexBlobId(xenos::TessellationMode mode) {
  return mode == xenos::TessellationMode::kAdaptive
             ? kTessellationVertexBlobAdaptive
             : kTessellationVertexBlobIndexed;
}

TessellationBlob GetTessellationVertexBlob(xenos::TessellationMode mode) {
  if (mode == xenos::TessellationMode::kAdaptive) {
    return {::tessellation_adaptive_vs, sizeof(::tessellation_adaptive_vs),
            "tessellation_adaptive_vs"};
  }
  return {::tessellation_indexed_vs, sizeof(::tessellation_indexed_vs),
          "tessellation_indexed_vs"};
}

// HS selection: mode x ProcessingResult::host_vertex_shader_type (port notes
// section 1 table; XeniOS metal_pipeline_cache.cc:856-930). Unsupported combos
// (adaptive CP-indexed, non-domain host VS types) return an empty blob.
TessellationBlob GetTessellationHullBlob(
    xenos::TessellationMode mode,
    Shader::HostVertexShaderType host_vertex_shader_type) {
  using HostType = Shader::HostVertexShaderType;
  switch (mode) {
    case xenos::TessellationMode::kDiscrete:
      switch (host_vertex_shader_type) {
        case HostType::kTriangleDomainCPIndexed:
          return {::discrete_triangle_3cp_hs, sizeof(::discrete_triangle_3cp_hs),
                  "discrete_triangle_3cp_hs"};
        case HostType::kTriangleDomainPatchIndexed:
          return {::discrete_triangle_1cp_hs, sizeof(::discrete_triangle_1cp_hs),
                  "discrete_triangle_1cp_hs"};
        case HostType::kQuadDomainCPIndexed:
          return {::discrete_quad_4cp_hs, sizeof(::discrete_quad_4cp_hs),
                  "discrete_quad_4cp_hs"};
        case HostType::kQuadDomainPatchIndexed:
          return {::discrete_quad_1cp_hs, sizeof(::discrete_quad_1cp_hs),
                  "discrete_quad_1cp_hs"};
        default:
          break;
      }
      break;
    case xenos::TessellationMode::kContinuous:
      switch (host_vertex_shader_type) {
        case HostType::kTriangleDomainCPIndexed:
          return {::continuous_triangle_3cp_hs,
                  sizeof(::continuous_triangle_3cp_hs),
                  "continuous_triangle_3cp_hs"};
        case HostType::kTriangleDomainPatchIndexed:
          return {::continuous_triangle_1cp_hs,
                  sizeof(::continuous_triangle_1cp_hs),
                  "continuous_triangle_1cp_hs"};
        case HostType::kQuadDomainCPIndexed:
          return {::continuous_quad_4cp_hs, sizeof(::continuous_quad_4cp_hs),
                  "continuous_quad_4cp_hs"};
        case HostType::kQuadDomainPatchIndexed:
          return {::continuous_quad_1cp_hs, sizeof(::continuous_quad_1cp_hs),
                  "continuous_quad_1cp_hs"};
        default:
          break;
      }
      break;
    case xenos::TessellationMode::kAdaptive:
      switch (host_vertex_shader_type) {
        case HostType::kTriangleDomainPatchIndexed:
          return {::adaptive_triangle_hs, sizeof(::adaptive_triangle_hs),
                  "adaptive_triangle_hs"};
        case HostType::kQuadDomainPatchIndexed:
          return {::adaptive_quad_hs, sizeof(::adaptive_quad_hs),
                  "adaptive_quad_hs"};
        default:
          break;
      }
      break;
    default:
      break;
  }
  return {};
}

uint64_t TessellationHullStageKey(
    xenos::TessellationMode mode,
    Shader::HostVertexShaderType host_vertex_shader_type) {
  return (uint64_t(mode) << 32) | uint64_t(uint32_t(host_vertex_shader_type));
}

// Maps the HS reflection's tessellator output primitive to the runtime's
// validation/geometry-primitive/passthrough-GS triple (XeniOS
// metal_pipeline_cache.cc:2820-2859). The passthrough geometry function lives
// in the IR runtime itself (geometryLibrary stays null). Returns false on an
// unsupported output primitive.
bool MapTessellatorOutputPrimitive(
    uint32_t hs_tessellator_output_primitive,
    IRRuntimeTessellatorOutputPrimitive& output_primitive,
    IRRuntimePrimitiveType& geometry_primitive,
    const char*& geometry_function) {
  switch (hs_tessellator_output_primitive) {
    case IRRuntimeTessellatorOutputPoint:
      output_primitive = IRRuntimeTessellatorOutputPoint;
      geometry_primitive = IRRuntimePrimitiveTypePoint;
      geometry_function = kIRPointPassthroughGeometryShader;
      return true;
    case IRRuntimeTessellatorOutputLine:
      output_primitive = IRRuntimeTessellatorOutputLine;
      geometry_primitive = IRRuntimePrimitiveTypeLine;
      geometry_function = kIRLinePassthroughGeometryShader;
      return true;
    case IRRuntimeTessellatorOutputTriangleCW:
      output_primitive = IRRuntimeTessellatorOutputTriangleCW;
      geometry_primitive = IRRuntimePrimitiveTypeTriangle;
      geometry_function = kIRTrianglePassthroughGeometryShader;
      return true;
    case IRRuntimeTessellatorOutputTriangleCCW:
      output_primitive = IRRuntimeTessellatorOutputTriangleCCW;
      geometry_primitive = IRRuntimePrimitiveTypeTriangle;
      geometry_function = kIRTrianglePassthroughGeometryShader;
      return true;
    default:
      return false;
  }
}

}  // namespace

bool MetalPipelineCache::EnsureDepthOnlyPixelShader() {
  if (depth_only_pixel_library_) {
    return true;
  }
  if (!shader_translator_) {
    XELOGE("MetalPipelineCache: depth-only PS requested before Initialize");
    return false;
  }
  // Ports XeniOS's EnsureDepthOnlyPixelShader (metal_pipeline_cache.cc:
  // 1800-1843): the shared translator's generated depth-only DXBC through the
  // ordinary DXBC -> DXIL -> metallib back half. The entry point is recovered
  // from the library's own function list, exactly as CreateMetalFunction does.
  std::vector<uint8_t> dxbc = shader_translator_->CreateDepthOnlyPixelShader();
  if (dxbc.empty()) {
    XELOGE("MetalPipelineCache: depth-only PS DXBC generation failed");
    return false;
  }
  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: depth-only PS DXBC->DXIL failed: {}", error);
    return false;
  }
  std::vector<uint8_t> metallib;
  if (!metal_converter_.Convert(MetalShaderConverter::Stage::kFragment, dxil,
                                metallib, &error)) {
    XELOGE("MetalPipelineCache: depth-only PS DXIL->metallib failed: {}", error);
    return false;
  }
  void* library = NewLibraryFromBytes(metallib, "depth-only PS");
  if (!library) {
    return false;
  }
  id<MTLLibrary> mtl_library = (__bridge id<MTLLibrary>)library;
  NSString* first_function = mtl_library.functionNames.firstObject;
  if (!first_function) {
    XELOGE("MetalPipelineCache: depth-only PS library has no functions");
    id<MTLLibrary> released = (__bridge_transfer id<MTLLibrary>)library;
    (void)released;
    return false;
  }
  depth_only_pixel_library_ = library;
  depth_only_pixel_function_name_ = [first_function UTF8String];
  return true;
}

const MetalPipelineCache::TessellationVertexStage*
MetalPipelineCache::GetOrCreateTessellationVertexStage(
    xenos::TessellationMode tessellation_mode) {
  const uint32_t blob_id = TessellationVertexBlobId(tessellation_mode);
  auto it = tessellation_vertex_stages_.find(blob_id);
  if (it != tessellation_vertex_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  // Cache failures too, so a bad toolchain state isn't retried every draw.
  TessellationVertexStage stage;
  const TessellationBlob blob = GetTessellationVertexBlob(tessellation_mode);

  std::vector<uint8_t> dxbc(blob.bytes, blob.bytes + blob.size);
  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: tessellation VS {} DXBC->DXIL failed: {}",
           blob.name, error);
    tessellation_vertex_stages_.emplace(blob_id, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  // Topology decision (see the section comment): kUndefined, matching XeniOS.
  // The zero-element stage-in is synthesized inside ConvertForGeometryEmulation
  // and its absence is already a converter-level hard failure (gotcha 9), but
  // it is re-checked here because an empty stage-in metallib is a guaranteed
  // pipeline failure (XeniOS metal_pipeline_cache.cc:810-813).
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kVertex, dxil,
          MetalShaderConverter::InputTopology::kUndefined, result) ||
      result.stage_in_metallib.empty()) {
    XELOGE("MetalPipelineCache: tessellation VS {} conversion failed: {}",
           blob.name,
           !result.error.empty() ? result.error : "empty stage-in metallib");
    tessellation_vertex_stages_.emplace(blob_id, std::move(stage));
    return nullptr;
  }
  if (!result.vertex_output_size_in_bytes) {
    // Zero vsOutputSizeInBytes fails the pipeline-config zero-check later; a
    // stage that can never build a pipeline is a stage failure now.
    XELOGE(
        "MetalPipelineCache: tessellation VS {} reflection returned zero "
        "output size",
        blob.name);
    tessellation_vertex_stages_.emplace(blob_id, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "tessellation VS");
  void* stage_in =
      NewLibraryFromBytes(result.stage_in_metallib, "tessellation VS stage-in");
  if (!library || !stage_in) {
    if (library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
      (void)lib;
    }
    if (stage_in) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)stage_in;
      (void)lib;
    }
    tessellation_vertex_stages_.emplace(blob_id, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.stage_in_library = stage_in;
  stage.function_name = result.function_name;
  stage.vertex_output_size = result.vertex_output_size_in_bytes;
  stage.metallib = std::move(result.metallib);
  stage.stage_in_metallib = std::move(result.stage_in_metallib);
  stage.valid = true;
  auto inserted = tessellation_vertex_stages_.emplace(blob_id, std::move(stage));
  return &inserted.first->second;
}

const MetalPipelineCache::TessellationHullStage*
MetalPipelineCache::GetOrCreateTessellationHullStage(
    xenos::TessellationMode tessellation_mode,
    Shader::HostVertexShaderType host_vertex_shader_type) {
  const uint64_t key =
      TessellationHullStageKey(tessellation_mode, host_vertex_shader_type);
  auto it = tessellation_hull_stages_.find(key);
  if (it != tessellation_hull_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  TessellationHullStage stage;
  const TessellationBlob blob =
      GetTessellationHullBlob(tessellation_mode, host_vertex_shader_type);
  if (!blob.bytes) {
    XELOGE(
        "MetalPipelineCache: unsupported tessellation mode {} / host VS type "
        "{} combination",
        uint32_t(tessellation_mode), uint32_t(host_vertex_shader_type));
    tessellation_hull_stages_.emplace(key, std::move(stage));
    return nullptr;
  }

  std::vector<uint8_t> dxbc(blob.bytes, blob.bytes + blob.size);
  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: tessellation HS {} DXBC->DXIL failed: {}",
           blob.name, error);
    tessellation_hull_stages_.emplace(key, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  // Topology decision (see the section comment): kUndefined, matching XeniOS.
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kHull, dxil,
          MetalShaderConverter::InputTopology::kUndefined, result) ||
      !result.has_hull_info) {
    XELOGE("MetalPipelineCache: tessellation HS {} conversion failed: {}",
           blob.name,
           !result.error.empty() ? result.error : "missing hull reflection");
    tessellation_hull_stages_.emplace(key, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "tessellation HS");
  if (!library) {
    tessellation_hull_stages_.emplace(key, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.function_name = result.function_name;
  stage.max_patches_per_object_threadgroup =
      result.hs_max_patches_per_object_threadgroup;
  stage.max_object_threads_per_patch = result.hs_max_object_threads_per_patch;
  stage.patch_constants_size = result.hs_patch_constants_size;
  stage.input_control_point_count = result.hs_input_control_point_count;
  stage.output_control_point_count = result.hs_output_control_point_count;
  stage.output_control_point_size = result.hs_output_control_point_size;
  stage.tessellator_domain = result.hs_tessellator_domain;
  stage.tessellator_partitioning = result.hs_tessellator_partitioning;
  stage.tessellator_output_primitive = result.hs_tessellator_output_primitive;
  stage.max_tessellation_factor = result.hs_max_tessellation_factor;
  stage.metallib = std::move(result.metallib);
  stage.valid = true;
  auto inserted = tessellation_hull_stages_.emplace(key, std::move(stage));
  return &inserted.first->second;
}

const MetalPipelineCache::TessellationDomainStage*
MetalPipelineCache::GetOrCreateTessellationDomainStage(
    MetalShader::MetalTranslation* domain_translation) {
  // Content-derived identity (ucode hash + modification + type) — the
  // modification embeds the domain host_vertex_shader_type, so two domain
  // types of the same guest shader are distinct entries.
  const uint64_t key = MetallibCacheKey(*domain_translation);
  auto it = tessellation_domain_stages_.find(key);
  if (it != tessellation_domain_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  TessellationDomainStage stage;

  // Stage 1 (ucode -> DXBC) if the caller has not run it. The domain
  // translation must NOT be run through TranslateShader — that path compiles
  // Stage::kVertex, and this DXBC is a ds_5_1 program.
  if (!domain_translation->is_translated()) {
    if (!shader_translator_->TranslateAnalyzedShader(*domain_translation)) {
      XELOGE(
          "MetalPipelineCache: tessellation DS DXBC translation failed for "
          "shader {:016X}",
          domain_translation->shader().ucode_data_hash());
      tessellation_domain_stages_.emplace(key, std::move(stage));
      return nullptr;
    }
  }
  const std::vector<uint8_t>& dxbc = domain_translation->translated_binary();
  if (dxbc.empty()) {
    XELOGE("MetalPipelineCache: tessellation DS has no DXBC for shader {:016X}",
           domain_translation->shader().ucode_data_hash());
    tessellation_domain_stages_.emplace(key, std::move(stage));
    return nullptr;
  }

  std::string error;
  std::vector<uint8_t> dxil;
  if (!dxil_converter_.Convert(dxbc, dxil, &error)) {
    XELOGE("MetalPipelineCache: tessellation DS DXBC->DXIL failed: {}", error);
    tessellation_domain_stages_.emplace(key, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  // Topology decision (see the section comment): kUndefined, matching XeniOS.
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kDomain, dxil,
          MetalShaderConverter::InputTopology::kUndefined, result) ||
      !result.has_domain_info) {
    XELOGE("MetalPipelineCache: tessellation DS conversion failed: {}",
           !result.error.empty() ? result.error : "missing domain reflection");
    tessellation_domain_stages_.emplace(key, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "tessellation DS");
  if (!library) {
    tessellation_domain_stages_.emplace(key, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.function_name = result.function_name;
  stage.max_input_prims_per_mesh_threadgroup =
      result.ds_max_input_prims_per_mesh_threadgroup;
  stage.input_control_point_count = result.ds_input_control_point_count;
  stage.input_control_point_size = result.ds_input_control_point_size;
  stage.patch_constants_size = result.ds_patch_constants_size;
  stage.tessellator_domain = result.ds_tessellator_domain;
  stage.metallib = std::move(result.metallib);
  stage.valid = true;
  auto inserted = tessellation_domain_stages_.emplace(key, std::move(stage));
  return &inserted.first->second;
}

bool MetalPipelineCache::AdoptTessellationRecord(
    const MetalPipelineStorage::TessellationRecord& record,
    xenos::TessellationMode tessellation_mode,
    Shader::HostVertexShaderType host_vertex_shader_type,
    uint64_t domain_stage_key) {
  bool all_adopted = true;
  // Storage identity was validated at Open, so record components and freshly
  // compiled stages are interchangeable artifacts of the same translators; a
  // partial adoption (device refused one library) simply leaves the rest to
  // cold compilation.
  const uint32_t vs_blob_id = TessellationVertexBlobId(tessellation_mode);
  if (!tessellation_vertex_stages_.count(vs_blob_id)) {
    void* library =
        NewLibraryFromBytes(record.vs_metallib, "tessellation VS (cached)");
    void* stage_in = NewLibraryFromBytes(record.vs_stage_in_metallib,
                                         "tessellation VS stage-in (cached)");
    if (library && stage_in) {
      TessellationVertexStage stage;
      stage.library = library;
      stage.stage_in_library = stage_in;
      stage.function_name = record.vs_function_name;
      stage.vertex_output_size = record.vs_output_size;
      stage.metallib = record.vs_metallib;
      stage.stage_in_metallib = record.vs_stage_in_metallib;
      stage.valid = true;
      tessellation_vertex_stages_.emplace(vs_blob_id, std::move(stage));
    } else {
      if (library) {
        id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
        (void)lib;
      }
      if (stage_in) {
        id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)stage_in;
        (void)lib;
      }
      all_adopted = false;
    }
  }
  const uint64_t hull_key =
      TessellationHullStageKey(tessellation_mode, host_vertex_shader_type);
  if (!tessellation_hull_stages_.count(hull_key)) {
    void* library =
        NewLibraryFromBytes(record.hs_metallib, "tessellation HS (cached)");
    if (library) {
      TessellationHullStage stage;
      stage.library = library;
      stage.function_name = record.hs_function_name;
      stage.max_patches_per_object_threadgroup =
          record.hs_max_patches_per_object_threadgroup;
      stage.max_object_threads_per_patch =
          record.hs_max_object_threads_per_patch;
      stage.patch_constants_size = record.hs_patch_constants_size;
      stage.input_control_point_count = record.hs_input_control_point_count;
      stage.output_control_point_count = record.hs_output_control_point_count;
      stage.output_control_point_size = record.hs_output_control_point_size;
      stage.tessellator_domain = record.hs_tessellator_domain;
      stage.tessellator_partitioning = record.hs_tessellator_partitioning;
      stage.tessellator_output_primitive =
          record.hs_tessellator_output_primitive;
      stage.max_tessellation_factor = record.hs_max_tessellation_factor;
      stage.metallib = record.hs_metallib;
      stage.valid = true;
      tessellation_hull_stages_.emplace(hull_key, std::move(stage));
    } else {
      all_adopted = false;
    }
  }
  if (!tessellation_domain_stages_.count(domain_stage_key)) {
    void* library =
        NewLibraryFromBytes(record.ds_metallib, "tessellation DS (cached)");
    if (library) {
      TessellationDomainStage stage;
      stage.library = library;
      stage.function_name = record.ds_function_name;
      stage.max_input_prims_per_mesh_threadgroup =
          record.ds_max_input_prims_per_mesh_threadgroup;
      stage.input_control_point_count = record.ds_input_control_point_count;
      stage.input_control_point_size = record.ds_input_control_point_size;
      stage.patch_constants_size = record.ds_patch_constants_size;
      stage.tessellator_domain = record.ds_tessellator_domain;
      stage.metallib = record.ds_metallib;
      stage.valid = true;
      tessellation_domain_stages_.emplace(domain_stage_key, std::move(stage));
    } else {
      all_adopted = false;
    }
  }
  if (!all_adopted) {
    XELOGW(
        "MetalPipelineCache: cached tessellation record adopted partially; "
        "missing stages will be recompiled");
  }
  return all_adopted;
}

const MetalPipelineCache::TessellationPipeline*
MetalPipelineCache::GetOrCreateTessellationPipelineState(
    MetalShader::MetalTranslation* domain_translation,
    MetalShader::MetalTranslation* pixel,
    Shader::HostVertexShaderType host_vertex_shader_type,
    xenos::TessellationMode tessellation_mode,
    xenos::PrimitiveType host_primitive_type,
    const uint64_t* color_pixel_formats, uint64_t depth_pixel_format,
    MetalPipelineRasterKey raster_key, const uint32_t* blend_controls,
    uint32_t color_write_mask) {
  if (!domain_translation) {
    return nullptr;
  }

  // Pipeline key — the port notes section 3 step 2 field set, expressed in
  // this backend's conventions:
  //   * content-derived shader identities instead of translation POINTERS
  //     (the 3.2 rule; a pointer key could never back a durable entry and
  //     XeniOS's pointer keying is an explicit non-port — gotcha 7);
  //   * no separate stencil format: as in every other pipeline key here it is
  //     derived from the depth format, never independent;
  //   * no alpha_to_mask field: this backend never enables Metal
  //     alpha-to-coverage (Xenos alpha-to-mask is exact SV_Coverage in the
  //     shader), so it is not an identity input — same audit as
  //     GetOrCreateRenderPipelineState.
  // A null/functionless pixel shader keys as ps.present=0, which is exactly
  // the depth-only-fallback pipeline.
  struct TessellationPipelineKeyBytes {
    MetalShaderKeyIdentity ds;
    MetalShaderKeyIdentity ps;
    uint32_t host_vs_type;
    uint32_t tessellation_mode;
    uint32_t host_prim;
    uint32_t sample_count;
    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint64_t depth;
    uint32_t blends[xenos::kMaxColorRenderTargets];
    uint32_t write_mask;
    uint32_t pad;
  } key_bytes = {};
  const bool has_guest_pixel_shader = pixel && pixel->has_metal_function();
  key_bytes.ds = MakeShaderKeyIdentity(domain_translation);
  key_bytes.ps =
      MakeShaderKeyIdentity(has_guest_pixel_shader ? pixel : nullptr);
  key_bytes.host_vs_type = uint32_t(host_vertex_shader_type);
  key_bytes.tessellation_mode = uint32_t(tessellation_mode);
  key_bytes.host_prim = uint32_t(host_primitive_type);
  key_bytes.sample_count = raster_key.raster_sample_count;
  key_bytes.depth = depth_pixel_format;
  key_bytes.write_mask = color_write_mask;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    key_bytes.colors[i] = color_pixel_formats[i];
    key_bytes.blends[i] = blend_controls[i];
  }
  const uint64_t key = XXH3_64bits(&key_bytes, sizeof(key_bytes));
  auto cache_it = tessellation_pipelines_.find(key);
  if (cache_it != tessellation_pipelines_.end()) {
    if (!cache_it->second.pipeline) {
      uint32_t count = ++cache_it->second.negative_hits;
      if ((count & (count - 1)) == 0) {
        XELOGW(
            "MetalPipelineCache: tessellation pipeline negative-cache hit #{} "
            "(draws suppressed by one failed creation) key={:016X}",
            count, key);
      }
      return nullptr;
    }
    return &cache_it->second;
  }

  // Durable warm start (task 3.2): adopt the record's stages into the stage
  // caches BEFORE the getters run, so a warm run skips dxilconv + MSC for all
  // three stages. Our storage replays lazily — this lookup at first use IS the
  // replay; there is no prewarm enumeration pass (the record's aux triple
  // exists so one can be added without a format change).
  const uint64_t domain_stage_key = MetallibCacheKey(*domain_translation);
  const uint64_t record_key = MetalTessellationRecordKey(
      domain_stage_key, uint32_t(host_vertex_shader_type),
      uint32_t(tessellation_mode));
  bool record_loaded = false;
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    MetalPipelineStorage::TessellationRecord record;
    if (storage_.LoadTessellationRecord(record_key, &record)) {
      record_loaded =
          AdoptTessellationRecord(record, tessellation_mode,
                                  host_vertex_shader_type, domain_stage_key);
    }
  }

  const TessellationVertexStage* vertex_stage =
      GetOrCreateTessellationVertexStage(tessellation_mode);
  const TessellationHullStage* hull_stage = GetOrCreateTessellationHullStage(
      tessellation_mode, host_vertex_shader_type);
  const TessellationDomainStage* domain_stage =
      GetOrCreateTessellationDomainStage(domain_translation);
  if (!vertex_stage || !hull_stage || !domain_stage) {
    XELOGE(
        "MetalPipelineCache: tessellation pipeline missing stage (vs={}, "
        "hs={}, ds={}) key={:016X}",
        vertex_stage != nullptr, hull_stage != nullptr, domain_stage != nullptr,
        key);
    tessellation_pipelines_.emplace(key, TessellationPipeline{});
    return nullptr;
  }

  // Fragment stage: the guest PS, or the depth-only fallback — tessellated
  // draws never REQUIRE a guest PS (port notes section 3 step 1 / section 6).
  void* fragment_library = nullptr;
  std::string fragment_function_name;
  if (has_guest_pixel_shader) {
    fragment_library = pixel->metal_library();
    id<MTLFunction> ps_fn = (__bridge id<MTLFunction>)pixel->metal_function();
    fragment_function_name = ps_fn.name ? [ps_fn.name UTF8String] : "";
  } else {
    if (!EnsureDepthOnlyPixelShader()) {
      tessellation_pipelines_.emplace(key, TessellationPipeline{});
      return nullptr;
    }
    fragment_library = depth_only_pixel_library_;
    fragment_function_name = depth_only_pixel_function_name_;
  }
  if (!fragment_library || fragment_function_name.empty()) {
    XELOGE("MetalPipelineCache: tessellation pipeline has no fragment stage");
    tessellation_pipelines_.emplace(key, TessellationPipeline{});
    return nullptr;
  }

  // Hull-output-primitive -> passthrough geometry function selection.
  IRRuntimeTessellatorOutputPrimitive output_primitive;
  IRRuntimePrimitiveType geometry_primitive;
  const char* geometry_function = nullptr;
  if (!MapTessellatorOutputPrimitive(hull_stage->tessellator_output_primitive,
                                     output_primitive, geometry_primitive,
                                     geometry_function)) {
    XELOGE(
        "MetalPipelineCache: unsupported tessellator output primitive {} "
        "key={:016X}",
        hull_stage->tessellator_output_primitive, key);
    tessellation_pipelines_.emplace(key, TessellationPipeline{});
    return nullptr;
  }

  // HS/DS pairing validation — failure IS construction failure, negative-
  // cached like every other creation failure on this path (the first failure
  // logs; repeats are suppressed through the negative cache). Argument order
  // per port notes section 3 step 5.
  if (!IRRuntimeValidateTessellationPipeline(
          output_primitive, geometry_primitive,
          hull_stage->output_control_point_size,
          domain_stage->input_control_point_size,
          hull_stage->patch_constants_size, domain_stage->patch_constants_size,
          hull_stage->output_control_point_count,
          domain_stage->input_control_point_count)) {
    XELOGE(
        "MetalPipelineCache: tessellation HS/DS validation failed "
        "(hs_out_cp={}/{}B ds_in_cp={}/{}B hs_patch_const={} ds_patch_const={}) "
        "key={:016X}",
        hull_stage->output_control_point_count,
        hull_stage->output_control_point_size,
        domain_stage->input_control_point_count,
        domain_stage->input_control_point_size,
        hull_stage->patch_constants_size, domain_stage->patch_constants_size,
        key);
    tessellation_pipelines_.emplace(key, TessellationPipeline{});
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
  // Alpha-to-coverage stays OFF — same reasoning as every other pipeline here.
  desc.alphaToCoverageEnabled = NO;

  IRGeometryTessellationEmulationPipelineDescriptor ir_desc = {};
  ir_desc.stageInLibrary = (__bridge id<MTLLibrary>)vertex_stage->stage_in_library;
  ir_desc.vertexLibrary = (__bridge id<MTLLibrary>)vertex_stage->library;
  ir_desc.vertexFunctionName = vertex_stage->function_name.c_str();
  ir_desc.hullLibrary = (__bridge id<MTLLibrary>)hull_stage->library;
  ir_desc.hullFunctionName = hull_stage->function_name.c_str();
  ir_desc.domainLibrary = (__bridge id<MTLLibrary>)domain_stage->library;
  ir_desc.domainFunctionName = domain_stage->function_name.c_str();
  // Runtime-provided passthrough geometry stage (lives in the domain library).
  ir_desc.geometryLibrary = nil;
  ir_desc.geometryFunctionName = geometry_function;
  ir_desc.fragmentLibrary = (__bridge id<MTLLibrary>)fragment_library;
  ir_desc.fragmentFunctionName = fragment_function_name.c_str();
  ir_desc.basePipelineDescriptor = desc;
  ir_desc.pipelineConfig.outputPrimitiveType = output_primitive;
  ir_desc.pipelineConfig.vsOutputSizeInBytes = vertex_stage->vertex_output_size;
  // Port notes gotcha 8: from the DOMAIN reflection, not the hull's.
  ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
      domain_stage->max_input_prims_per_mesh_threadgroup;
  ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup =
      hull_stage->max_patches_per_object_threadgroup;
  ir_desc.pipelineConfig.hsInputControlPointCount =
      hull_stage->input_control_point_count;
  ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup =
      hull_stage->max_object_threads_per_patch;
  ir_desc.pipelineConfig.hsMaxTessellationFactor =
      hull_stage->max_tessellation_factor;
  ir_desc.pipelineConfig.gsInstanceCount = 1;

  // All five count fields zero-checked (XeniOS metal_pipeline_cache.cc:
  // 2914-2929): a zero would produce a pipeline that dispatches nothing or
  // divides by zero inside the IR runtime's patch math.
  if (!ir_desc.pipelineConfig.vsOutputSizeInBytes ||
      !ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup ||
      !ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup ||
      !ir_desc.pipelineConfig.hsInputControlPointCount ||
      !ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup) {
    XELOGE(
        "MetalPipelineCache: tessellation pipeline invalid reflection values "
        "(vs_output={}, gs_max_input={}, hs_patches={}, hs_cp_count={}, "
        "hs_threads={}) key={:016X}",
        ir_desc.pipelineConfig.vsOutputSizeInBytes,
        ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup,
        ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup,
        ir_desc.pipelineConfig.hsInputControlPointCount,
        ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup, key);
    tessellation_pipelines_.emplace(key, TessellationPipeline{});
    return nullptr;
  }

  NSError* error = nil;
  id<MTLRenderPipelineState> pipeline =
      IRRuntimeNewGeometryTessellationEmulationPipeline(device, &ir_desc,
                                                        &error);
  if (!pipeline) {
    XELOGE(
        "MetalPipelineCache: tessellation pipeline creation failed: {} "
        "(mode={}, host_vs_type={}, vs_fn='{}', hs_fn='{}', ds_fn='{}', "
        "ps_fn='{}') key={:016X}",
        error ? [[error localizedDescription] UTF8String] : "unknown",
        uint32_t(tessellation_mode), uint32_t(host_vertex_shader_type),
        vertex_stage->function_name, hull_stage->function_name,
        domain_stage->function_name, fragment_function_name, key);
    tessellation_pipelines_.emplace(key, TessellationPipeline{});
    return nullptr;
  }

  XELOGI(
      "MetalPipelineCache: built tessellation pipeline (mode={}, "
      "host_vs_type={}, outprim={}, vs_output={}, gs_max_input={}, "
      "hs_patches={}, hs_cp={}, hs_threads={}, factor={}) key={:016X}",
      uint32_t(tessellation_mode), uint32_t(host_vertex_shader_type),
      uint32_t(output_primitive), ir_desc.pipelineConfig.vsOutputSizeInBytes,
      ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup,
      ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup,
      ir_desc.pipelineConfig.hsInputControlPointCount,
      ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup,
      ir_desc.pipelineConfig.hsMaxTessellationFactor, key);

  TessellationPipeline state;
  state.pipeline = (__bridge_retained void*)pipeline;
  state.output_primitive_type = uint32_t(output_primitive);
  state.vs_output_size_in_bytes = ir_desc.pipelineConfig.vsOutputSizeInBytes;
  state.gs_max_input_primitives_per_mesh_threadgroup =
      ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup;
  state.hs_max_patches_per_object_threadgroup =
      ir_desc.pipelineConfig.hsMaxPatchesPerObjectThreadgroup;
  state.hs_input_control_point_count =
      ir_desc.pipelineConfig.hsInputControlPointCount;
  state.hs_max_object_threads_per_threadgroup =
      ir_desc.pipelineConfig.hsMaxObjectThreadsPerThreadgroup;
  state.hs_max_tessellation_factor =
      ir_desc.pipelineConfig.hsMaxTessellationFactor;
  state.gs_instance_count = ir_desc.pipelineConfig.gsInstanceCount;
  state.geometry_primitive = uint32_t(geometry_primitive);
  auto inserted = tessellation_pipelines_.emplace(key, std::move(state));

  // Publish the durable record only after the device accepted the pipeline,
  // and only if this key was not already served from storage.
  if (!record_loaded && cvars::metal_pipeline_metallib_cache &&
      storage_.is_open()) {
    MetalPipelineStorage::TessellationRecord record;
    record.host_vertex_shader_type = uint32_t(host_vertex_shader_type);
    record.tessellation_mode = uint32_t(tessellation_mode);
    record.host_primitive_type = uint32_t(host_primitive_type);
    record.vs_function_name = vertex_stage->function_name;
    record.hs_function_name = hull_stage->function_name;
    record.ds_function_name = domain_stage->function_name;
    record.vs_output_size = vertex_stage->vertex_output_size;
    record.hs_max_patches_per_object_threadgroup =
        hull_stage->max_patches_per_object_threadgroup;
    record.hs_max_object_threads_per_patch =
        hull_stage->max_object_threads_per_patch;
    record.hs_patch_constants_size = hull_stage->patch_constants_size;
    record.hs_input_control_point_count =
        hull_stage->input_control_point_count;
    record.hs_output_control_point_count =
        hull_stage->output_control_point_count;
    record.hs_output_control_point_size =
        hull_stage->output_control_point_size;
    record.hs_tessellator_domain = hull_stage->tessellator_domain;
    record.hs_tessellator_partitioning = hull_stage->tessellator_partitioning;
    record.hs_tessellator_output_primitive =
        hull_stage->tessellator_output_primitive;
    record.hs_max_tessellation_factor = hull_stage->max_tessellation_factor;
    record.ds_max_input_prims_per_mesh_threadgroup =
        domain_stage->max_input_prims_per_mesh_threadgroup;
    record.ds_input_control_point_count =
        domain_stage->input_control_point_count;
    record.ds_input_control_point_size =
        domain_stage->input_control_point_size;
    record.ds_patch_constants_size = domain_stage->patch_constants_size;
    record.ds_tessellator_domain = domain_stage->tessellator_domain;
    record.vs_metallib = vertex_stage->metallib;
    record.vs_stage_in_metallib = vertex_stage->stage_in_metallib;
    record.hs_metallib = hull_stage->metallib;
    record.ds_metallib = domain_stage->metallib;
    storage_.StoreTessellationRecord(record_key, record);
  }

  return &inserted.first->second;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
