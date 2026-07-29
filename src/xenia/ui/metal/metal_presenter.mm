/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstdlib>

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
    // 4-vertex triangle-strip QUAD (not an oversized fullscreen triangle): a
    // triangle would overhang the aspect-ratio bars and sample out-of-range uv
    // there instead of leaving them as the clear color. Corners c = (0,0),(1,0),
    // (0,1),(1,1). `scale` (<=1 on one axis) shrinks the quad to preserve the
    // guest display aspect ratio; the vacated area shows the clear color as
    // letterbox/pillarbox bars.
    vertex VOut blit_vs(uint vid [[vertex_id]],
                        constant float2& scale [[buffer(0)]]) {
      float2 c = float2(vid & 1, (vid >> 1) & 1);
      VOut o;
      o.pos = float4((c * 2.0 - 1.0) * scale, 0.0, 1.0);
      o.uv = float2(c.x, 1.0 - c.y);               // flip Y
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

    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    // Black letterbox/pillarbox bars around the aspect-corrected guest image.
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    if (guest_output_texture && EnsureBlitPipeline()) {
      // Fit the guest image into the drawable preserving its display aspect
      // ratio: shrink the blit quad on the axis the drawable is "too long" on,
      // leaving the clear color as bars. Compare drawable aspect (actual pixels)
      // to the guest display aspect.
      float scale[2] = {1.0f, 1.0f};
      uint32_t out_w = uint32_t(drawable.texture.width);
      uint32_t out_h = uint32_t(drawable.texture.height);
      uint32_t dar_x = guest_output_properties.display_aspect_ratio_x;
      uint32_t dar_y = guest_output_properties.display_aspect_ratio_y;
      if (out_w && out_h && dar_x && dar_y) {
        double image_aspect = double(dar_x) / double(dar_y);
        double surface_aspect = double(out_w) / double(out_h);
        if (surface_aspect > image_aspect) {
          scale[0] = float(image_aspect / surface_aspect);  // pillarbox
        } else {
          scale[1] = float(surface_aspect / image_aspect);  // letterbox
        }
      }
      [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)
                                          blit_pipeline_state_];
      [encoder setVertexBytes:scale length:sizeof(scale) atIndex:0];
      [encoder setFragmentTexture:guest_output_texture atIndex:0];
      [encoder setFragmentSamplerState:(__bridge id<MTLSamplerState>)
                                           blit_sampler_
                              atIndex:0];
      [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                  vertexStart:0
                  vertexCount:4];
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

    // Contract 2 discovery: how the drawable is actually composed. Records
    // whether a guest image existed, whether UI drawers ran over it, and
    // whether the drawable size ever changed -- the last being what makes
    // resize a real question rather than a hypothetical one. Default-off and
    // observational.
    static const bool s_present_census =
        std::getenv("XE_METAL_PRESENT_CENSUS") != nullptr;
    if (s_present_census) {
      static uint64_t s_paints = 0, s_with_guest = 0, s_with_ui = 0,
                      s_letterboxed = 0;
      static uint32_t s_last_w = 0, s_last_h = 0, s_size_changes = 0;
      const uint32_t dw = uint32_t(drawable.texture.width);
      const uint32_t dh = uint32_t(drawable.texture.height);
      ++s_paints;
      if (guest_output_texture) {
        ++s_with_guest;
        // Any axis not filling the drawable means bars are visible, i.e. the
        // presenter is already doing aspect-preserving scaling.
        if (uint32_t(guest_output_texture.width) != dw ||
            uint32_t(guest_output_texture.height) != dh) {
          ++s_letterboxed;
        }
      }
      if (execute_ui_drawers) {
        ++s_with_ui;
      }
      if (s_last_w && (dw != s_last_w || dh != s_last_h)) {
        ++s_size_changes;
      }
      s_last_w = dw;
      s_last_h = dh;
      if ((s_paints % 256) == 0) {
        XELOGI(
            "MetalPresentCompose paints={} with_guest={} with_ui={} "
            "scaled_or_letterboxed={} drawable={}x{} drawable_size_changes={}",
            s_paints, s_with_guest, s_with_ui, s_letterboxed, dw, dh,
            s_size_changes);
      }
    }

    [encoder endEncoding];

    // A silent GPU fault in the present blit leaves the window on the last good
    // frame while the game keeps running — reads as a "GPU freeze". Log it.
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
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
