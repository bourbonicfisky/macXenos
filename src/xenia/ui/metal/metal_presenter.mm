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
  // window. Nothing to release here (non-owning __bridge references).
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

bool MetalPresenter::RefreshGuestOutputImpl(
    uint32_t mailbox_index, uint32_t frontbuffer_width,
    uint32_t frontbuffer_height,
    std::function<bool(GuestOutputRefreshContext& context)> refresher,
    bool& is_8bpc_out_ref) {
  // No guest-output image path yet: the game framebuffer is uploaded and blitted
  // by the render path (later phase). Reporting failure keeps the guest output
  // inactive so PaintAndPresentImpl just presents the clear color.
  return false;
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

    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    // Distinct dark blue so a working Metal present is visually unmistakable
    // from the black of the null/stub path.
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.05, 0.15, 1.0);

    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    // Nothing to draw yet; the clear happens on encoder begin.
    [encoder endEncoding];

    [command_buffer presentDrawable:drawable];
    [command_buffer commit];
  }

  return PaintResult::kPresented;
}

}  // namespace metal
}  // namespace ui
}  // namespace xe
