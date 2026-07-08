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

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

#include "xenia/ui/presenter.h"
#include "xenia/ui/surface.h"

namespace xe {
namespace ui {
namespace metal {

// Refresh context handed to the guest-output refresher (the command processor's
// IssueSwap). Exposes the mailbox MTLTexture the producer must fill with the
// current frontbuffer. Derives from the base context so the refresher can
// static_cast to reach target_texture(). Metal objects stay opaque (void*).
class MetalGuestOutputRefreshContext : public Presenter::GuestOutputRefreshContext {
 public:
  MetalGuestOutputRefreshContext(bool& is_8bpc_out_ref, void* target_texture)
      : GuestOutputRefreshContext(is_8bpc_out_ref),
        target_texture_(target_texture) {}

  // id<MTLTexture> to write the current frontbuffer into. Not owned.
  void* target_texture() const { return target_texture_; }

 private:
  void* target_texture_;
};

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

  // Ensures the mailbox texture at `index` is an `width`x`height` RGBA8 render
  // target usable as a shader input. Recreated on size change. Returns the
  // id<MTLTexture> (as void*), or nullptr on failure.
  void* EnsureGuestOutputTexture(uint32_t index, uint32_t width,
                                 uint32_t height);
  // Lazily builds the fullscreen-blit pipeline (samples the guest output into
  // the drawable). Returns false on compile failure.
  bool EnsureBlitPipeline();

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

  // Guest-output mailbox images (id<MTLTexture>, retained), one per mailbox
  // slot, plus their current sizes so we can detect a resolution change.
  std::array<void*, kGuestOutputMailboxSize> guest_output_textures_{};
  std::array<uint32_t, kGuestOutputMailboxSize> guest_output_texture_widths_{};
  std::array<uint32_t, kGuestOutputMailboxSize> guest_output_texture_heights_{};

  // Fullscreen textured-blit pipeline + sampler (id<MTLRenderPipelineState> /
  // id<MTLSamplerState>, retained), built on first guest-output paint.
  void* blit_pipeline_state_ = nullptr;
  void* blit_sampler_ = nullptr;
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_METAL_PRESENTER_H_
