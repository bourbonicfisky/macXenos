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
#include <utility>

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

// Draw context handed to the UI immediate drawers during a paint. Carries the
// in-flight command buffer and the render encoder already open on the
// CAMetalLayer drawable, so overlay draws land in the SAME pass as the
// guest-output blit, after it and before endEncoding. That is what keeps
// overlay drawing off guest state entirely: it composites into the drawable and
// never touches EDRAM, guest render targets, or guest memory.
//
// Metal objects stay opaque (void*) so this header remains plain C++, matching
// MetalGuestOutputRefreshContext above. Neither pointer is owned: both are
// valid only for the duration of the ExecuteUIDrawersFromUIThread call.
class MetalUIDrawContext final : public UIDrawContext {
 public:
  MetalUIDrawContext(Presenter& presenter, uint32_t render_target_width,
                     uint32_t render_target_height, void* command_buffer,
                     void* render_encoder)
      : UIDrawContext(presenter, render_target_width, render_target_height),
        command_buffer_(command_buffer),
        render_encoder_(render_encoder) {}

  // id<MTLCommandBuffer>. Not owned.
  void* command_buffer() const { return command_buffer_; }
  // id<MTLRenderCommandEncoder>, already open on the drawable. Not owned.
  void* render_encoder() const { return render_encoder_; }

 private:
  void* command_buffer_;
  void* render_encoder_;
};

// Metal presenter. Owns a CAMetalLayer connection and paints the guest output
// into its drawable through the shared guest output paint flow (bilinear
// stretching, AMD FidelityFX CAS sharpening / resampling, FSR upscaling, and
// dithering, with letterboxing), then the UI overlays on top.
// Metal/Objective-C objects are held as opaque pointers so this header stays
// usable from plain C++.
class MetalPresenter : public Presenter {
 public:
  ~MetalPresenter() override;

  // device is an id<MTLDevice>, command_queue an id<MTLCommandQueue>, both
  // owned by the MetalProvider (retained there for the presenter's lifetime).
  static std::unique_ptr<MetalPresenter> Create(
      HostGpuLossCallback host_gpu_loss_callback, void* device,
      void* command_queue);

  Surface::TypeFlags GetSupportedSurfaceTypes() const override;

  // GPU-fault recovery: points the presenter at the provider's recreated
  // command queue after the driver revoked the old one (whose submissions are
  // ignored forever). The old queue object stays retained by the provider, so
  // a paint racing this setter uses a dead-but-valid queue for one frame.
  void SetCommandQueue(void* command_queue) { command_queue_ = command_queue; }

  bool CaptureGuestOutput(RawImage& image_out) override;

  bool SupportsGuestOutputPaintEffect(
      GuestOutputPaintConfig::Effect effect) const override;

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
  static const char* GetGuestOutputPaintEffectName(
      GuestOutputPaintEffect effect);
  // Lazily loads the guest output paint shaders and creates the sampler.
  // Returns false (and keeps returning false) if they can't be loaded.
  bool EnsureGuestOutputPaintResources();
  // The id<MTLRenderPipelineState> for an effect drawing either into an
  // intermediate texture or into the drawable, created on first use. Returns
  // nullptr on failure.
  void* GetGuestOutputPaintPipeline(GuestOutputPaintEffect effect,
                                    bool to_drawable);
  // Ensures the intermediate texture `index` (the output of effect `index`
  // when another effect follows) is `width`x`height`. Returns the
  // id<MTLTexture>, or nullptr on failure.
  void* EnsureGuestOutputIntermediateTexture(size_t index, uint32_t width,
                                             uint32_t height);
  // Encodes effect `effect_index` of the flow into `render_encoder`, reading
  // `source_texture` and writing a `target_width`x`target_height` target.
  // Returns false if the pipeline is unavailable.
  bool EncodeGuestOutputPaintEffect(void* render_encoder,
                                    const GuestOutputPaintFlow& flow,
                                    size_t effect_index,
                                    const GuestOutputPaintConfig& config,
                                    void* source_texture, bool to_drawable,
                                    uint32_t target_width,
                                    uint32_t target_height);
  // The MetalFX spatial scaler (id<MTLFXSpatialScaler>) from the guest output
  // size to the output size, and its output texture, recreated when a size
  // changes. Returns nullptr if MetalFX can't scale between these sizes; the
  // output texture is written to `output_texture_out`.
  void* EnsureMetalFxScaler(uint32_t input_width, uint32_t input_height,
                            uint32_t output_width, uint32_t output_height,
                            void*& output_texture_out);
  // Logs the effect chain when it differs from the previous paint's.
  void LogGuestOutputPaintFlowIfChanged(const GuestOutputPaintFlow& flow);

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

  // Guest output paint shaders (id<MTLFunction>, retained): the shared
  // rectangle vertex shader and one pixel shader per effect. Loaded on the
  // first paint with guest output.
  bool guest_output_paint_resources_loaded_ = false;
  bool guest_output_paint_resources_failed_ = false;
  void* guest_output_paint_vs_ = nullptr;
  std::array<void*, size_t(GuestOutputPaintEffect::kCount)>
      guest_output_paint_ps_{};
  // Linear clamp-to-edge sampler (id<MTLSamplerState>, retained).
  void* guest_output_paint_sampler_ = nullptr;
  // id<MTLRenderPipelineState> (retained) per effect, [0] for intermediate
  // textures, [1] for the drawable.
  std::array<std::array<void*, 2>, size_t(GuestOutputPaintEffect::kCount)>
      guest_output_paint_pipelines_{};

  // Intermediate textures between effects (id<MTLTexture>, retained).
  std::array<void*, kMaxGuestOutputPaintEffects - 1>
      guest_output_intermediate_textures_{};
  std::array<std::pair<uint32_t, uint32_t>, kMaxGuestOutputPaintEffects - 1>
      guest_output_intermediate_texture_sizes_{};

  // MetalFX spatial scaling (id<MTLFXSpatialScaler> and id<MTLTexture>,
  // retained), for the sizes below.
  void* metalfx_scaler_ = nullptr;
  void* metalfx_output_texture_ = nullptr;
  uint32_t metalfx_input_width_ = 0;
  uint32_t metalfx_input_height_ = 0;
  uint32_t metalfx_output_width_ = 0;
  uint32_t metalfx_output_height_ = 0;
  // Sizes MetalFX failed for, so creation isn't retried every paint.
  uint32_t metalfx_failed_sizes_[4] = {};

  // The effect chain of the last paint, for logging changes to it.
  size_t last_logged_effect_count_ = SIZE_MAX;
  std::array<GuestOutputPaintEffect, kMaxGuestOutputPaintEffects>
      last_logged_effects_{};
  std::array<std::pair<uint32_t, uint32_t>, kMaxGuestOutputPaintEffects>
      last_logged_effect_output_sizes_{};
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_METAL_PRESENTER_H_
