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

#include <string>
#include <utility>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/logging.h"
#include "xenia/ui/graphics_provider.h"

namespace xe {
namespace gpu {
namespace metal {

MetalPipelineCache::MetalPipelineCache(void* device) : device_(device) {}

MetalPipelineCache::~MetalPipelineCache() { Shutdown(); }

bool MetalPipelineCache::Initialize() {
  // Match the shader-spike configuration: Apple vendor, bindless resources, no
  // pixel-shader-interlock EDRAM (the copy-based RT model), which is what the
  // Xbox-360 root signature in MetalShaderConverter expects.
  shader_translator_ = std::make_unique<DxbcShaderTranslator>(
      ui::GraphicsProvider::GpuVendorID::kApple,
      /*bindless_resources_used=*/true, /*edram_rov_used=*/false);

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
  for (auto& pair : render_pipeline_states_) {
    if (pair.second) {
      id<MTLRenderPipelineState> state =
          (__bridge_transfer id<MTLRenderPipelineState>)pair.second;
      (void)state;
    }
  }
  render_pipeline_states_.clear();
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

  // Stage 3: DXIL -> metallib.
  MetalShaderConverter::Stage stage =
      shader.type() == xenos::ShaderType::kVertex
          ? MetalShaderConverter::Stage::kVertex
          : MetalShaderConverter::Stage::kFragment;
  std::vector<uint8_t> metallib;
  if (!metal_converter_.Convert(stage, dxil, metallib, &error)) {
    XELOGE("MetalPipelineCache: DXIL->metallib failed for shader {:016X}: {}",
           shader.ucode_data_hash(), error);
    return false;
  }

  // Load onto the device as an MTLFunction.
  if (!translation.CreateMetalFunction(device_, metallib)) {
    XELOGE("MetalPipelineCache: MTLFunction creation failed for shader {:016X}",
           shader.ucode_data_hash());
    return false;
  }
  return true;
}

void* MetalPipelineCache::GetOrCreateRenderPipelineState(
    MetalShader::MetalTranslation* vertex,
    MetalShader::MetalTranslation* pixel, uint64_t color_pixel_format,
    uint64_t depth_pixel_format) {
  if (!vertex || !vertex->has_metal_function()) {
    return nullptr;
  }

  // Key on the identity of the host functions plus the attachment formats. The
  // functions are stable for a given translation, so their pointers are a valid
  // identity within one run.
  struct KeyBytes {
    void* vs;
    void* ps;
    uint64_t color;
    uint64_t depth;
  } key_bytes{vertex->metal_function(),
              pixel ? pixel->metal_function() : nullptr, color_pixel_format,
              depth_pixel_format};
  uint64_t key = XXH3_64bits(&key_bytes, sizeof(key_bytes));
  auto it = render_pipeline_states_.find(key);
  if (it != render_pipeline_states_.end()) {
    return it->second;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLRenderPipelineDescriptor* desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  desc.vertexFunction = (__bridge id<MTLFunction>)vertex->metal_function();
  if (pixel && pixel->has_metal_function()) {
    desc.fragmentFunction = (__bridge id<MTLFunction>)pixel->metal_function();
  }
  desc.colorAttachments[0].pixelFormat = (MTLPixelFormat)color_pixel_format;
  if (depth_pixel_format != MTLPixelFormatInvalid) {
    desc.depthAttachmentPixelFormat = (MTLPixelFormat)depth_pixel_format;
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

}  // namespace metal
}  // namespace gpu
}  // namespace xe
