/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>

#include "xenia/ui/metal/metal_presenter.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/ui/surface_mac.h"

#include "xenia/ui/shaders/bytecode/metal/guest_output_bilinear_dither_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_bilinear_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_cas_resample_dither_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_cas_resample_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_cas_sharpen_dither_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_cas_sharpen_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_fsr_easu_half_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_fsr_easu_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_fsr_rcas_dither_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_fsr_rcas_half_dither_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_fsr_rcas_half_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_ffx_fsr_rcas_ps.h"
#include "xenia/ui/shaders/bytecode/metal/guest_output_triangle_strip_rect_vs.h"

DEFINE_bool(
    metal_fsr_half_precision, true,
    "Run AMD FidelityFX Super Resolution (FSR) in half precision (16 bits) on "
    "the Metal renderer: 11-12% less GPU time at 4K and 5K output on an M4, "
    "with no visible difference. Takes effect on the next launch.",
    "Display");

namespace xe {
namespace ui {
namespace metal {

namespace {

// The CAMetalLayer drawable format.
constexpr MTLPixelFormat kDrawablePixelFormat = MTLPixelFormatBGRA8Unorm;
// Outputs of effects followed by another effect (FSR EASU passes, or CAS / FSR
// RCAS before the final bilinear stretch). 10bpc, like the Vulkan presenter's
// intermediate images.
constexpr MTLPixelFormat kIntermediatePixelFormat = MTLPixelFormatRGB10A2Unorm;
// The largest 2D texture dimension on all Apple-silicon Macs.
constexpr uint32_t kMaxRenderTargetSize = 16384;

id<MTLFunction> LoadEntryFunction(id<MTLDevice> device, const uint8_t* metallib,
                                  size_t size, const char* name) {
  dispatch_data_t data = dispatch_data_create(metallib, size, nullptr, ^{
  });
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  if (!library) {
    XELOGE("MetalPresenter: failed to load the {} shader: {}", name,
           error ? [[error localizedDescription] UTF8String] : "unknown");
    return nil;
  }
  // The XeSL MSL entry point is always entry_xe.
  id<MTLFunction> function = [library newFunctionWithName:@"entry_xe"];
  if (!function) {
    XELOGE("MetalPresenter: no entry_xe in the {} shader", name);
  }
  return function;
}

}  // namespace

MetalPresenter::~MetalPresenter() {
  // device_/command_queue_ are owned by the provider; metal_layer_ by the
  // window. Release the objects we retained: mailbox and intermediate textures,
  // shaders, pipelines and the sampler.
  auto release = [](void* object) {
    if (object) {
      CFRelease(object);
    }
  };
  for (void* texture : guest_output_textures_) {
    release(texture);
  }
  for (void* texture : guest_output_intermediate_textures_) {
    release(texture);
  }
  for (const std::array<void*, 2>& effect_pipelines :
       guest_output_paint_pipelines_) {
    for (void* pipeline : effect_pipelines) {
      release(pipeline);
    }
  }
  for (void* ps : guest_output_paint_ps_) {
    release(ps);
  }
  release(guest_output_paint_vs_);
  release(guest_output_paint_sampler_);
  release(metalfx_scaler_);
  release(metalfx_output_texture_);
}

std::unique_ptr<MetalPresenter> MetalPresenter::Create(
    HostGpuLossCallback host_gpu_loss_callback, void* device,
    void* command_queue) {
  if (!device || !command_queue) {
    return nullptr;
  }
  std::unique_ptr<MetalPresenter> presenter(
      new MetalPresenter(host_gpu_loss_callback, device, command_queue));
  if (!presenter->InitializeCommonSurfaceIndependent()) {
    return nullptr;
  }
  return presenter;
}

Surface::TypeFlags MetalPresenter::GetSupportedSurfaceTypes() const {
  return Surface::kTypeFlag_MetalLayer;
}

bool MetalPresenter::SupportsGuestOutputPaintEffect(
    GuestOutputPaintConfig::Effect effect) const {
  if (effect == GuestOutputPaintConfig::Effect::kMetalFxSpatial) {
    // The device doesn't change, so ask once.
    static const bool metalfx_spatial_supported = [MTLFXSpatialScalerDescriptor
        supportsDevice:(__bridge id<MTLDevice>)device_];
    return metalfx_spatial_supported;
  }
  return true;
}

bool MetalPresenter::CaptureGuestOutput(RawImage& image_out) {
  // Read back the most recently refreshed guest output mailbox texture (used by
  // the headless trace dump to dump the rendered frame). Consume the mailbox to
  // pin the ready image, blit it (Private RGBA8) into a shared staging buffer,
  // then copy into the RawImage as R8 G8 B8 X8 (bytes already RGBA).
  void* tex_ptr = nullptr;
  uint32_t width = 0, height = 0;
  {
    uint32_t mailbox_index = UINT32_MAX;
    std::unique_lock<std::mutex> consumer_lock(
        ConsumeGuestOutput(mailbox_index, nullptr, nullptr));
    if (mailbox_index != UINT32_MAX &&
        mailbox_index < guest_output_textures_.size()) {
      tex_ptr = guest_output_textures_[mailbox_index];
      width = guest_output_texture_widths_[mailbox_index];
      height = guest_output_texture_heights_[mailbox_index];
    }
  }
  if (!tex_ptr || !width || !height) {
    return false;
  }
  id<MTLTexture> texture = (__bridge id<MTLTexture>)tex_ptr;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
  size_t row_bytes = size_t(width) * 4;
  id<MTLBuffer> staging =
      [device newBufferWithLength:row_bytes * height
                          options:MTLResourceStorageModeShared];
  if (!staging) {
    return false;
  }
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:texture
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(width, height, 1)
                toBuffer:staging
       destinationOffset:0
  destinationBytesPerRow:row_bytes
destinationBytesPerImage:row_bytes * height];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  image_out.width = width;
  image_out.height = height;
  image_out.stride = row_bytes;
  image_out.data.resize(row_bytes * height);
  const uint8_t* src = static_cast<const uint8_t*>(staging.contents);
  for (size_t i = 0; i < size_t(width) * height; ++i) {
    image_out.data[i * 4 + 0] = src[i * 4 + 0];  // R
    image_out.data[i * 4 + 1] = src[i * 4 + 1];  // G
    image_out.data[i * 4 + 2] = src[i * 4 + 2];  // B
    image_out.data[i * 4 + 3] = 255;
  }
  return true;
}

Presenter::SurfacePaintConnectResult
MetalPresenter::ConnectOrReconnectPaintingToSurfaceFromUIThread(
    Surface& new_surface, uint32_t new_surface_width,
    uint32_t new_surface_height, bool was_paintable,
    bool& is_vsync_implicit_out) {
  if (new_surface.GetType() != Surface::kTypeIndex_MetalLayer) {
    return SurfacePaintConnectResult::kFailureSurfaceUnusable;
  }
  auto& metal_surface = static_cast<MetalLayerSurface&>(new_surface);
  CAMetalLayer* layer = (__bridge CAMetalLayer*)metal_surface.metal_layer();
  if (!layer) {
    return SurfacePaintConnectResult::kFailureSurfaceUnusable;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  layer.device = device;
  layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  // We only render into the drawable and present it (no readback/sampling).
  layer.framebufferOnly = YES;
  layer.drawableSize =
      CGSizeMake(CGFloat(new_surface_width), CGFloat(new_surface_height));

  {
    std::lock_guard<std::mutex> lock(surface_mutex_);
    metal_layer_ = metal_surface.metal_layer();
    surface_width_ = new_surface_width;
    surface_height_ = new_surface_height;
  }

  // CAMetalLayer paces presentation to the display refresh; treat vsync as
  // implicitly on so the base class doesn't also try to block for it.
  is_vsync_implicit_out = true;
  return SurfacePaintConnectResult::kSuccess;
}

void MetalPresenter::DisconnectPaintingFromSurfaceFromUIThreadImpl() {
  std::lock_guard<std::mutex> lock(surface_mutex_);
  metal_layer_ = nullptr;
  surface_width_ = 0;
  surface_height_ = 0;
}

void* MetalPresenter::EnsureGuestOutputTexture(uint32_t index, uint32_t width,
                                               uint32_t height) {
  if (index >= guest_output_textures_.size()) {
    return nullptr;
  }
  if (guest_output_textures_[index] &&
      guest_output_texture_widths_[index] == width &&
      guest_output_texture_heights_[index] == height) {
    return guest_output_textures_[index];
  }
  // Release the previous texture for this slot, if any.
  if (guest_output_textures_[index]) {
    id<MTLTexture> old =
        (__bridge_transfer id<MTLTexture>)guest_output_textures_[index];
    (void)old;
    guest_output_textures_[index] = nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  // RGBA8 matches the backend's canonical color render-target storage (bytes
  // == guest EDRAM bytes) so IssueSwap can blit it in directly; the present
  // pass SAMPLES it into the BGRA8 drawable, so the format difference there is
  // fine. Written by the producer's refresher and sampled by the present blit.
  desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalPresenter: failed to create {}x{} guest output texture", width,
           height);
    return nullptr;
  }
  guest_output_textures_[index] = (__bridge_retained void*)texture;
  guest_output_texture_widths_[index] = width;
  guest_output_texture_heights_[index] = height;
  return guest_output_textures_[index];
}

const char* MetalPresenter::GetGuestOutputPaintEffectName(
    GuestOutputPaintEffect effect) {
  using Effect = GuestOutputPaintEffect;
  switch (effect) {
    case Effect::kBilinear:
      return "bilinear";
    case Effect::kBilinearDither:
      return "bilinear+dither";
    case Effect::kCasSharpen:
      return "CAS sharpen";
    case Effect::kCasSharpenDither:
      return "CAS sharpen+dither";
    case Effect::kCasResample:
      return "CAS resample";
    case Effect::kCasResampleDither:
      return "CAS resample+dither";
    case Effect::kFsrEasu:
      return "FSR EASU";
    case Effect::kFsrRcas:
      return "FSR RCAS";
    case Effect::kFsrRcasDither:
      return "FSR RCAS+dither";
    default:
      return "unknown";
  }
}

bool MetalPresenter::EnsureGuestOutputPaintResources() {
  if (guest_output_paint_resources_loaded_) {
    return true;
  }
  if (guest_output_paint_resources_failed_) {
    return false;
  }
  // Fail once, loudly, rather than retrying every frame.
  guest_output_paint_resources_failed_ = true;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

  id<MTLFunction> vs = LoadEntryFunction(
      device, guest_output_triangle_strip_rect_vs_metallib,
      sizeof(guest_output_triangle_strip_rect_vs_metallib),
      "guest_output_triangle_strip_rect_vs");
  if (!vs) {
    return false;
  }
  struct PixelShader {
    const uint8_t* metallib;
    size_t size;
  };
  static const PixelShader kPixelShaders[] = {
      {guest_output_bilinear_ps_metallib,
       sizeof(guest_output_bilinear_ps_metallib)},
      {guest_output_bilinear_dither_ps_metallib,
       sizeof(guest_output_bilinear_dither_ps_metallib)},
      {guest_output_ffx_cas_sharpen_ps_metallib,
       sizeof(guest_output_ffx_cas_sharpen_ps_metallib)},
      {guest_output_ffx_cas_sharpen_dither_ps_metallib,
       sizeof(guest_output_ffx_cas_sharpen_dither_ps_metallib)},
      {guest_output_ffx_cas_resample_ps_metallib,
       sizeof(guest_output_ffx_cas_resample_ps_metallib)},
      {guest_output_ffx_cas_resample_dither_ps_metallib,
       sizeof(guest_output_ffx_cas_resample_dither_ps_metallib)},
      {guest_output_ffx_fsr_easu_ps_metallib,
       sizeof(guest_output_ffx_fsr_easu_ps_metallib)},
      {guest_output_ffx_fsr_rcas_ps_metallib,
       sizeof(guest_output_ffx_fsr_rcas_ps_metallib)},
      {guest_output_ffx_fsr_rcas_dither_ps_metallib,
       sizeof(guest_output_ffx_fsr_rcas_dither_ps_metallib)},
  };
  static_assert(xe::countof(kPixelShaders) ==
                    size_t(GuestOutputPaintEffect::kCount),
                "A pixel shader is needed for every guest output effect, in "
                "GuestOutputPaintEffect order");
  // The FSR passes in half precision, replacing their 32-bit versions.
  static const std::pair<GuestOutputPaintEffect, PixelShader>
      kHalfPixelShaders[] = {
          {GuestOutputPaintEffect::kFsrEasu,
           {guest_output_ffx_fsr_easu_half_ps_metallib,
            sizeof(guest_output_ffx_fsr_easu_half_ps_metallib)}},
          {GuestOutputPaintEffect::kFsrRcas,
           {guest_output_ffx_fsr_rcas_half_ps_metallib,
            sizeof(guest_output_ffx_fsr_rcas_half_ps_metallib)}},
          {GuestOutputPaintEffect::kFsrRcasDither,
           {guest_output_ffx_fsr_rcas_half_dither_ps_metallib,
            sizeof(guest_output_ffx_fsr_rcas_half_dither_ps_metallib)}},
      };
  std::array<PixelShader, size_t(GuestOutputPaintEffect::kCount)>
      pixel_shaders;
  std::copy(std::begin(kPixelShaders), std::end(kPixelShaders),
            pixel_shaders.begin());
  if (cvars::metal_fsr_half_precision) {
    for (const auto& half_pixel_shader : kHalfPixelShaders) {
      pixel_shaders[size_t(half_pixel_shader.first)] = half_pixel_shader.second;
    }
  }
  std::array<id<MTLFunction>, size_t(GuestOutputPaintEffect::kCount)> ps;
  for (size_t i = 0; i < ps.size(); ++i) {
    ps[i] = LoadEntryFunction(
        device, pixel_shaders[i].metallib, pixel_shaders[i].size,
        GetGuestOutputPaintEffectName(GuestOutputPaintEffect(i)));
    if (!ps[i]) {
      return false;
    }
  }

  MTLSamplerDescriptor* sampler_desc = [[MTLSamplerDescriptor alloc] init];
  sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
  sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
  sampler_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
  sampler_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
  id<MTLSamplerState> sampler =
      [device newSamplerStateWithDescriptor:sampler_desc];
  if (!sampler) {
    XELOGE("MetalPresenter: failed to create the guest output sampler");
    return false;
  }

  guest_output_paint_vs_ = (__bridge_retained void*)vs;
  for (size_t i = 0; i < ps.size(); ++i) {
    guest_output_paint_ps_[i] = (__bridge_retained void*)ps[i];
  }
  guest_output_paint_sampler_ = (__bridge_retained void*)sampler;
  guest_output_paint_resources_failed_ = false;
  guest_output_paint_resources_loaded_ = true;
  return true;
}

void* MetalPresenter::GetGuestOutputPaintPipeline(
    GuestOutputPaintEffect effect, bool to_drawable) {
  void*& pipeline_ref =
      guest_output_paint_pipelines_[size_t(effect)][to_drawable ? 1 : 0];
  if (pipeline_ref) {
    return pipeline_ref;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLRenderPipelineDescriptor* desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  desc.vertexFunction = (__bridge id<MTLFunction>)guest_output_paint_vs_;
  desc.fragmentFunction =
      (__bridge id<MTLFunction>)guest_output_paint_ps_[size_t(effect)];
  desc.colorAttachments[0].pixelFormat =
      to_drawable ? kDrawablePixelFormat : kIntermediatePixelFormat;
  NSError* error = nil;
  id<MTLRenderPipelineState> pipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
  if (!pipeline) {
    XELOGE("MetalPresenter: failed to create the {} pipeline: {}",
           GetGuestOutputPaintEffectName(effect),
           error ? [[error localizedDescription] UTF8String] : "unknown");
    return nullptr;
  }
  pipeline_ref = (__bridge_retained void*)pipeline;
  return pipeline_ref;
}

void* MetalPresenter::EnsureGuestOutputIntermediateTexture(size_t index,
                                                           uint32_t width,
                                                           uint32_t height) {
  void*& texture_ref = guest_output_intermediate_textures_[index];
  std::pair<uint32_t, uint32_t>& size_ref =
      guest_output_intermediate_texture_sizes_[index];
  if (texture_ref && size_ref.first == width && size_ref.second == height) {
    return texture_ref;
  }
  // A texture still in use by a submitted command buffer stays alive until it
  // completes, as command buffers retain their resources.
  if (texture_ref) {
    CFRelease(texture_ref);
    texture_ref = nullptr;
  }
  size_ref = std::make_pair(uint32_t(0), uint32_t(0));
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLTextureDescriptor* desc = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:kIntermediatePixelFormat
                                   width:width
                                  height:height
                               mipmapped:NO];
  desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModePrivate;
  id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
  if (!texture) {
    XELOGE("MetalPresenter: failed to create a {}x{} guest output "
           "intermediate texture",
           width, height);
    return nullptr;
  }
  texture_ref = (__bridge_retained void*)texture;
  size_ref = std::make_pair(width, height);
  return texture_ref;
}

void* MetalPresenter::EnsureMetalFxScaler(uint32_t input_width,
                                          uint32_t input_height,
                                          uint32_t output_width,
                                          uint32_t output_height,
                                          void*& output_texture_out) {
  if (metalfx_scaler_ && metalfx_input_width_ == input_width &&
      metalfx_input_height_ == input_height &&
      metalfx_output_width_ == output_width &&
      metalfx_output_height_ == output_height) {
    output_texture_out = metalfx_output_texture_;
    return metalfx_scaler_;
  }
  const uint32_t sizes[4] = {input_width, input_height, output_width,
                             output_height};
  if (std::equal(std::begin(sizes), std::end(sizes),
                 std::begin(metalfx_failed_sizes_))) {
    return nullptr;
  }
  if (metalfx_scaler_) {
    CFRelease(metalfx_scaler_);
    metalfx_scaler_ = nullptr;
  }
  if (metalfx_output_texture_) {
    CFRelease(metalfx_output_texture_);
    metalfx_output_texture_ = nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  // 10bpc output like the other effects' intermediates, 8bpc if MetalFX can't.
  id<MTLFXSpatialScaler> scaler = nil;
  for (MTLPixelFormat output_format :
       {kIntermediatePixelFormat, MTLPixelFormatRGBA8Unorm}) {
    MTLFXSpatialScalerDescriptor* desc =
        [[MTLFXSpatialScalerDescriptor alloc] init];
    desc.inputWidth = input_width;
    desc.inputHeight = input_height;
    desc.outputWidth = output_width;
    desc.outputHeight = output_height;
    // The guest output mailbox texture.
    desc.colorTextureFormat = MTLPixelFormatRGBA8Unorm;
    desc.outputTextureFormat = output_format;
    // The guest output is gamma-encoded.
    desc.colorProcessingMode =
        MTLFXSpatialScalerColorProcessingModePerceptual;
    scaler = [desc newSpatialScalerWithDevice:device];
    if (scaler) {
      MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:output_format
                                       width:output_width
                                      height:output_height
                                   mipmapped:NO];
      texture_desc.usage = scaler.outputTextureUsage | MTLTextureUsageShaderRead;
      texture_desc.storageMode = MTLStorageModePrivate;
      id<MTLTexture> output_texture =
          [device newTextureWithDescriptor:texture_desc];
      if (!output_texture) {
        scaler = nil;
        break;
      }
      metalfx_output_texture_ = (__bridge_retained void*)output_texture;
      break;
    }
  }
  if (!scaler) {
    XELOGW("MetalPresenter: MetalFX can't scale {}x{} to {}x{}; painting "
           "bilinear instead",
           input_width, input_height, output_width, output_height);
    std::copy(std::begin(sizes), std::end(sizes),
              std::begin(metalfx_failed_sizes_));
    return nullptr;
  }
  XELOGI("MetalPresenter: MetalFX spatial scaling {}x{} to {}x{}",
         input_width, input_height, output_width, output_height);
  metalfx_scaler_ = (__bridge_retained void*)scaler;
  metalfx_input_width_ = input_width;
  metalfx_input_height_ = input_height;
  metalfx_output_width_ = output_width;
  metalfx_output_height_ = output_height;
  output_texture_out = metalfx_output_texture_;
  return metalfx_scaler_;
}

bool MetalPresenter::EncodeGuestOutputPaintEffect(
    void* render_encoder, const GuestOutputPaintFlow& flow,
    size_t effect_index, const GuestOutputPaintConfig& config,
    void* source_texture, bool to_drawable, uint32_t target_width,
    uint32_t target_height) {
  GuestOutputPaintEffect effect = flow.effects[effect_index];
  void* pipeline = GetGuestOutputPaintPipeline(effect, to_drawable);
  if (!pipeline) {
    return false;
  }
  id<MTLRenderCommandEncoder> encoder =
      (__bridge id<MTLRenderCommandEncoder>)render_encoder;
  [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                      pipeline];

  // The rectangle in normalized device coordinates. Unlike Vulkan's, Metal's
  // Y axis points up, so the rectangle grows downwards from its top edge.
  int32_t output_x, output_y;
  flow.GetEffectOutputOffset(effect_index, output_x, output_y);
  const std::pair<uint32_t, uint32_t>& output_size =
      flow.effect_output_sizes[effect_index];
  float x_to_ndc = 2.0f / float(target_width);
  float y_to_ndc = 2.0f / float(target_height);
  float rect[4] = {
      -1.0f + float(output_x) * x_to_ndc,
      1.0f - float(output_y) * y_to_ndc,
      float(output_size.first) * x_to_ndc,
      -float(output_size.second) * y_to_ndc,
  };
  [encoder setVertexBytes:rect length:sizeof(rect) atIndex:0];

  // MSL lays the effect constants out from offset 0 of their own buffer, and
  // may pad the structure past the C++ size, so pass a zero-padded block.
  union {
    BilinearConstants bilinear;
    CasSharpenConstants cas_sharpen;
    CasResampleConstants cas_resample;
    FsrEasuConstants fsr_easu;
    FsrRcasConstants fsr_rcas;
    uint8_t padded[32];
  } constants = {};
  switch (effect) {
    case GuestOutputPaintEffect::kBilinear:
    case GuestOutputPaintEffect::kBilinearDither:
      constants.bilinear.Initialize(flow, effect_index);
      break;
    case GuestOutputPaintEffect::kCasSharpen:
    case GuestOutputPaintEffect::kCasSharpenDither:
      constants.cas_sharpen.Initialize(flow, effect_index, config);
      break;
    case GuestOutputPaintEffect::kCasResample:
    case GuestOutputPaintEffect::kCasResampleDither:
      constants.cas_resample.Initialize(flow, effect_index, config);
      break;
    case GuestOutputPaintEffect::kFsrEasu:
      constants.fsr_easu.Initialize(flow, effect_index);
      break;
    case GuestOutputPaintEffect::kFsrRcas:
    case GuestOutputPaintEffect::kFsrRcasDither:
      constants.fsr_rcas.Initialize(flow, effect_index, config);
      break;
    default:
      break;
  }
  [encoder setFragmentBytes:&constants length:sizeof(constants) atIndex:0];
  [encoder setFragmentTexture:(__bridge id<MTLTexture>)source_texture
                      atIndex:0];
  [encoder setFragmentSamplerState:(__bridge id<MTLSamplerState>)
                                       guest_output_paint_sampler_
                           atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
              vertexStart:0
              vertexCount:4];
  return true;
}

bool MetalPresenter::RefreshGuestOutputImpl(
    uint32_t mailbox_index, uint32_t frontbuffer_width,
    uint32_t frontbuffer_height,
    std::function<bool(GuestOutputRefreshContext& context)> refresher,
    bool& is_8bpc_out_ref) {
  void* texture =
      EnsureGuestOutputTexture(mailbox_index, frontbuffer_width,
                               frontbuffer_height);
  if (!texture) {
    return false;
  }
  // Hand the mailbox texture to the producer (IssueSwap) to fill with the
  // current frontbuffer; it records + submits its own Metal work.
  MetalGuestOutputRefreshContext context(is_8bpc_out_ref, texture);
  // The mailbox texture is RGBA8, so whatever the guest format, the image has
  // no more than 8 bits per channel and needs no dithering at 1:1.
  context.SetIs8bpc(true);
  return refresher(context);
}

void MetalPresenter::LogGuestOutputPaintFlowIfChanged(
    const GuestOutputPaintFlow& flow) {
  bool changed = flow.effect_count != last_logged_effect_count_;
  for (size_t i = 0; !changed && i < flow.effect_count; ++i) {
    changed = flow.effects[i] != last_logged_effects_[i] ||
              flow.effect_output_sizes[i] != last_logged_effect_output_sizes_[i];
  }
  if (!changed) {
    return;
  }
  last_logged_effect_count_ = flow.effect_count;
  std::string chain;
  for (size_t i = 0; i < flow.effect_count; ++i) {
    last_logged_effects_[i] = flow.effects[i];
    last_logged_effect_output_sizes_[i] = flow.effect_output_sizes[i];
    chain += fmt::format("{}{} -> {}x{}", i ? ", " : "",
                         GetGuestOutputPaintEffectName(flow.effects[i]),
                         flow.effect_output_sizes[i].first,
                         flow.effect_output_sizes[i].second);
  }
  if (flow.effect_count) {
    XELOGI("MetalPresenter: guest output {}x{} painted with: {}",
           flow.properties.frontbuffer_width,
           flow.properties.frontbuffer_height, chain);
  }
}

Presenter::PaintResult MetalPresenter::PaintAndPresentImpl(
    bool execute_ui_drawers) {
  void* layer_ptr;
  {
    std::lock_guard<std::mutex> lock(surface_mutex_);
    layer_ptr = metal_layer_;
  }
  if (!layer_ptr) {
    return PaintResult::kNotPresented;
  }

  @autoreleasepool {
    CAMetalLayer* layer = (__bridge CAMetalLayer*)layer_ptr;
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (!drawable) {
      return PaintResult::kNotPresented;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
    // Per-encoder execution status so a present-path GPU fault is attributed.
    MTLCommandBufferDescriptor* cb_desc =
        [[MTLCommandBufferDescriptor alloc] init];
    cb_desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> command_buffer =
        [queue commandBufferWithDescriptor:cb_desc];
    command_buffer.label = @"presenter:paint";

    // Take the ready guest-output image (if any). The consumer lock keeps the
    // acquired mailbox slot from being recycled while we sample it, so hold it
    // through commit.
    uint32_t mailbox_index = UINT32_MAX;
    GuestOutputProperties guest_output_properties;
    GuestOutputPaintConfig guest_output_paint_config;
    std::unique_lock<std::mutex> consumer_lock = ConsumeGuestOutput(
        mailbox_index, &guest_output_properties, &guest_output_paint_config);
    id<MTLTexture> guest_output_texture = nil;
    if (mailbox_index != UINT32_MAX &&
        mailbox_index < guest_output_textures_.size() &&
        guest_output_textures_[mailbox_index]) {
      guest_output_texture =
          (__bridge id<MTLTexture>)guest_output_textures_[mailbox_index];
    }

    uint32_t drawable_width = uint32_t(drawable.texture.width);
    uint32_t drawable_height = uint32_t(drawable.texture.height);
    GuestOutputPaintFlow flow = {};
    if (guest_output_texture && EnsureGuestOutputPaintResources()) {
      flow = GetGuestOutputPaintFlow(
          guest_output_properties, drawable_width, drawable_height,
          kMaxRenderTargetSize, kMaxRenderTargetSize,
          guest_output_paint_config);
    }
    LogGuestOutputPaintFlowIfChanged(flow);

    // Effects followed by another effect render into intermediate textures in
    // their own passes, before the drawable pass.
    void* effect_source = (__bridge void*)guest_output_texture;
    for (size_t i = 0; i + 1 < flow.effect_count; ++i) {
      const std::pair<uint32_t, uint32_t>& intermediate_size =
          flow.effect_output_sizes[i];
      void* intermediate = EnsureGuestOutputIntermediateTexture(
          i, intermediate_size.first, intermediate_size.second);
      if (!intermediate) {
        flow.effect_count = 0;
        break;
      }
      MTLRenderPassDescriptor* intermediate_pass =
          [MTLRenderPassDescriptor renderPassDescriptor];
      intermediate_pass.colorAttachments[0].texture =
          (__bridge id<MTLTexture>)intermediate;
      // The effect covers the whole intermediate texture.
      intermediate_pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
      intermediate_pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> intermediate_encoder =
          [command_buffer renderCommandEncoderWithDescriptor:intermediate_pass];
      bool encoded = EncodeGuestOutputPaintEffect(
          (__bridge void*)intermediate_encoder, flow, i,
          guest_output_paint_config, effect_source, false,
          intermediate_size.first, intermediate_size.second);
      [intermediate_encoder endEncoding];
      if (!encoded) {
        flow.effect_count = 0;
        break;
      }
      effect_source = intermediate;
    }

    // MetalFX spatial scaling: the flow painted for it is a bilinear stretch
    // (with dithering) to the output size. MetalFX upscales to that size
    // instead, and the bilinear pass then copies its output 1:1.
    if (flow.effect_count == 1 &&
        guest_output_paint_config.GetEffect() ==
            GuestOutputPaintConfig::Effect::kMetalFxSpatial) {
      uint32_t input_width = guest_output_properties.frontbuffer_width;
      uint32_t input_height = guest_output_properties.frontbuffer_height;
      const std::pair<uint32_t, uint32_t>& output_size =
          flow.effect_output_sizes[0];
      // Upscaling only.
      if (output_size.first >= input_width &&
          output_size.second >= input_height &&
          (output_size.first > input_width ||
           output_size.second > input_height) &&
          SupportsGuestOutputPaintEffect(
              GuestOutputPaintConfig::Effect::kMetalFxSpatial)) {
        void* metalfx_output = nullptr;
        void* scaler_ptr =
            EnsureMetalFxScaler(input_width, input_height, output_size.first,
                                output_size.second, metalfx_output);
        if (scaler_ptr) {
          id<MTLFXSpatialScaler> scaler =
              (__bridge id<MTLFXSpatialScaler>)scaler_ptr;
          scaler.colorTexture = guest_output_texture;
          scaler.outputTexture = (__bridge id<MTLTexture>)metalfx_output;
          scaler.inputContentWidth = input_width;
          scaler.inputContentHeight = input_height;
          [scaler encodeToCommandBuffer:command_buffer];
          effect_source = metalfx_output;
        }
      }
    }

    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    // Black letterbox/pillarbox bars around the aspect-corrected guest image.
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    if (flow.effect_count) {
      EncodeGuestOutputPaintEffect(
          (__bridge void*)encoder, flow, flow.effect_count - 1,
          guest_output_paint_config, effect_source, true, drawable_width,
          drawable_height);
    }

    // UI overlays (ImGui: emulator dialogs AND guest system dialogs) draw into
    // this same pass, on top of the guest image, before the encoder ends. They
    // composite into the drawable only -- no guest state is reachable from
    // here, which is what makes overlay drawing incapable of perturbing guest
    // output.
    //
    // Not drawing these is not cosmetic: xeXamDispatchDialog blocks its guest
    // thread on fence.Wait() until an ImGui-drawn modal is dismissed, so a
    // title that raises a system dialog hangs forever if nothing renders it.
    if (execute_ui_drawers) {
      MetalUIDrawContext ui_draw_context(
          *this, uint32_t(drawable.texture.width),
          uint32_t(drawable.texture.height), (__bridge void*)command_buffer,
          (__bridge void*)encoder);
      ExecuteUIDrawersFromUIThread(ui_draw_context);
    }

    [encoder endEncoding];

    // A silent GPU fault in the present blit leaves the window on the last good
    // frame while the game keeps running — reads as a "GPU freeze". Log it.
    // Also report the paint's GPU time every 600 paints, which is what the
    // upscaling effects cost on top of the guest's own rendering.
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      static std::atomic<uint64_t> s_paint_gpu_ns{0};
      static std::atomic<uint32_t> s_paint_count{0};
      if (cb.status == MTLCommandBufferStatusCompleted &&
          cb.GPUEndTime > cb.GPUStartTime) {
        uint64_t paint_gpu_ns =
            uint64_t((cb.GPUEndTime - cb.GPUStartTime) * 1e9);
        uint64_t gpu_ns =
            s_paint_gpu_ns.fetch_add(paint_gpu_ns) + paint_gpu_ns;
        uint32_t paint_count = s_paint_count.fetch_add(1) + 1;
        if (paint_count >= 600) {
          XELOGI("MetalPresenter: paint GPU time {:.3f} ms average over {} "
                 "paints",
                 double(gpu_ns) / double(paint_count) * 1e-6, paint_count);
          s_paint_gpu_ns.store(0);
          s_paint_count.store(0);
        }
      }
      if (cb.status == MTLCommandBufferStatusError) {
        static uint32_t s_present_errors = 0;
        if (s_present_errors++ < 32) {
          NSError* err = cb.error;
          XELOGE("Metal GPU FAULT (present): {} (code {})",
                 err.localizedDescription ? err.localizedDescription.UTF8String
                                          : "(no description)",
                 err ? long(err.code) : -1L);
        }
      }
    }];
    [command_buffer presentDrawable:drawable];
    [command_buffer commit];
    consumer_lock.unlock();
  }

  return PaintResult::kPresented;
}

}  // namespace metal
}  // namespace ui
}  // namespace xe
