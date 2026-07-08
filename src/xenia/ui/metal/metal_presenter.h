/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_METAL_METAL_PRESENTER_H_
#define XENIA_UI_METAL_METAL_PRESENTER_H_

#include <cstdint>
#include <memory>
#include <mutex>

#include "xenia/ui/presenter.h"
#include "xenia/ui/surface.h"

namespace xe {
namespace ui {
namespace metal {

// Phase 2b.3 Metal presenter. Owns a CAMetalLayer connection and presents a
// cleared drawable each frame. Guest-output blitting (showing the actual game
// framebuffer, with the FSR/CAS paint effects) is deferred to the render path;
// RefreshGuestOutputImpl currently reports no guest output, so the frame is a
// solid clear color plus (later) the UI overlay. Metal/Objective-C objects are
// held as opaque pointers so this header stays usable from plain C++.
class MetalPresenter : public Presenter {
 public:
  ~MetalPresenter() override;

  // device is an id<MTLDevice>, command_queue an id<MTLCommandQueue>, both
  // owned by the MetalProvider (retained there for the presenter's lifetime).
  static std::unique_ptr<MetalPresenter> Create(
      HostGpuLossCallback host_gpu_loss_callback, void* device,
      void* command_queue);

  Surface::TypeFlags GetSupportedSurfaceTypes() const override;

  bool CaptureGuestOutput(RawImage& image_out) override;

 protected:
  SurfacePaintConnectResult ConnectOrReconnectPaintingToSurfaceFromUIThread(
      Surface& new_surface, uint32_t new_surface_width,
      uint32_t new_surface_height, bool was_paintable,
      bool& is_vsync_implicit_out) override;
  void DisconnectPaintingFromSurfaceFromUIThreadImpl() override;

  bool RefreshGuestOutputImpl(
      uint32_t mailbox_index, uint32_t frontbuffer_width,
      uint32_t frontbuffer_height,
      std::function<bool(GuestOutputRefreshContext& context)> refresher,
      bool& is_8bpc_out_ref) override;

  PaintResult PaintAndPresentImpl(bool execute_ui_drawers) override;

 private:
  MetalPresenter(HostGpuLossCallback host_gpu_loss_callback, void* device,
                 void* command_queue)
      : Presenter(host_gpu_loss_callback),
        device_(device),
        command_queue_(command_queue) {}

  // id<MTLDevice> / id<MTLCommandQueue>, owned by the provider.
  void* device_ = nullptr;
  void* command_queue_ = nullptr;

  // CAMetalLayer* for the currently connected surface (not retained beyond the
  // connection; owned by the window). Guarded because painting may occur off the
  // UI thread for guest output.
  std::mutex surface_mutex_;
  void* metal_layer_ = nullptr;
  uint32_t surface_width_ = 0;
  uint32_t surface_height_ = 0;
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_METAL_PRESENTER_H_
