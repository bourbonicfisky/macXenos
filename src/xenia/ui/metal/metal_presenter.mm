/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/metal/metal_presenter.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "xenia/base/logging.h"
#include "xenia/ui/surface_mac.h"

namespace xe {
namespace ui {
namespace metal {

MetalPresenter::~MetalPresenter() {
  // device_/command_queue_ are owned by the provider; metal_layer_ by the
  // window. Release the objects we retained: mailbox textures + blit pipeline.
  for (void* tex : guest_output_textures_) {
    if (tex) {
      id<MTLTexture> released = (__bridge_transfer id<MTLTexture>)tex;
      (void)released;
    }
  }
  if (blit_pipeline_state_) {
    id<MTLRenderPipelineState> released =
        (__bridge_transfer id<MTLRenderPipelineState>)blit_pipeline_state_;
    (void)released;
  }
  if (blit_sampler_) {
    id<MTLSamplerState> released =
        (__bridge_transfer id<MTLSamplerState>)blit_sampler_;
    (void)released;
  }
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

bool MetalPresenter::CaptureGuestOutput(RawImage& image_out) {
  // Debug-only front-buffer capture; not implemented for the clear+present
  // skeleton (no guest output yet).
  return false;
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
      texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                   width:width
                                  height:height
                               mipmapped:NO];
  // BGRA8 matches the backend's color render target so IssueSwap can copy it in
  // directly, and the drawable so the present blit is format-clean.
  // Written by the producer's refresher (render target) and sampled by the
  // present blit (shader read).
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

bool MetalPresenter::EnsureBlitPipeline() {
  if (blit_pipeline_state_) {
    return true;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  // Fullscreen triangle that samples the guest output into the drawable. The
  // texcoord flips Y so the guest image is upright on screen.
  NSString* source = @R"METAL(
    #include <metal_stdlib>
    using namespace metal;
    struct VOut { float4 pos [[position]]; float2 uv; };
    vertex VOut blit_vs(uint vid [[vertex_id]]) {
      float2 t = float2((vid << 1) & 2, vid & 2);  // (0,0),(2,0),(0,2)
      VOut o;
      o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);
      // Fullscreen TRIANGLE: the vertices sit at t=(0,0),(2,0),(0,2) so the
      // visible [-1,1]^2 square must sample uv across the full 0..1 range. Use
      // the raw t (0..2) for u and 1-t for the Y-flip; multiplying by 0.5 (as an
      // earlier version did) compresses uv to 0..0.5 -> only the bottom-left
      // quarter of the guest image shows, magnified 2x.
      o.uv = float2(t.x, 1.0 - t.y);               // flip Y, full 0..1 coverage
      return o;
    }
    fragment float4 blit_fs(VOut in [[stage_in]],
                            texture2d<float> tex [[texture(0)]],
                            sampler s [[sampler(0)]]) {
      return tex.sample(s, in.uv);
    }
  )METAL";
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source
                                                options:nil
                                                  error:&error];
  if (!library) {
    XELOGE("MetalPresenter: blit library compile failed: {}",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    return false;
  }
  id<MTLFunction> vs = [library newFunctionWithName:@"blit_vs"];
  id<MTLFunction> fs = [library newFunctionWithName:@"blit_fs"];
  if (!vs || !fs) {
    XELOGE("MetalPresenter: blit functions missing");
    return false;
  }
  MTLRenderPipelineDescriptor* desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  desc.vertexFunction = vs;
  desc.fragmentFunction = fs;
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  id<MTLRenderPipelineState> pipeline =
      [device newRenderPipelineStateWithDescriptor:desc error:&error];
  if (!pipeline) {
    XELOGE("MetalPresenter: blit pipeline creation failed: {}",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    return false;
  }
  MTLSamplerDescriptor* sampler_desc = [[MTLSamplerDescriptor alloc] init];
  sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
  sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
  id<MTLSamplerState> sampler =
      [device newSamplerStateWithDescriptor:sampler_desc];
  blit_pipeline_state_ = (__bridge_retained void*)pipeline;
  blit_sampler_ = (__bridge_retained void*)sampler;
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
  return refresher(context);
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
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];

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

    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    // Distinct dark blue so a working Metal present is visually unmistakable
    // from the black of the null/stub path (also the letterbox fill when the
    // guest output doesn't cover the whole drawable).
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.15, 1.0);

    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    if (guest_output_texture && EnsureBlitPipeline()) {
      // Stretch the guest output over the whole drawable for now (aspect-ratio
      // letterboxing via GetGuestOutputPaintFlow comes with the full paint path).
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                          blit_pipeline_state_];
      [encoder setFragmentTexture:guest_output_texture atIndex:0];
      [encoder setFragmentSamplerState:(__bridge id<MTLSamplerState>)
                                           blit_sampler_
                              atIndex:0];
      [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                  vertexStart:0
                  vertexCount:3];
    }
    [encoder endEncoding];

    [command_buffer presentDrawable:drawable];
    [command_buffer commit];
    consumer_lock.unlock();
  }

  return PaintResult::kPresented;
}

}  // namespace metal
}  // namespace ui
}  // namespace xe
