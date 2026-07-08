/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_command_processor.h"

#import <Metal/Metal.h>

// Headers required by the PM4 execute-loop template included at the bottom of
// this file (mirrors the generic set vulkan_command_processor.cc pulls in).
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/xenos.h"
#include "xenia/gpu/xenos_zpd_report.h"
#include "xenia/kernel/user_module.h"
#include "xenia/ui/metal/metal_presenter.h"
#include "xenia/ui/metal/metal_provider.h"
#include "xenia/ui/presenter.h"

namespace xe {
namespace gpu {
namespace metal {

MetalCommandProcessor::MetalCommandProcessor(
    MetalGraphicsSystem* graphics_system, kernel::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state) {}
MetalCommandProcessor::~MetalCommandProcessor() = default;

void MetalCommandProcessor::TracePlaybackWroteMemory(uint32_t base_ptr,
                                                     uint32_t length) {}

void MetalCommandProcessor::RestoreEdramSnapshot(const void* snapshot) {}

bool MetalCommandProcessor::SetupContext() {
  if (!CommandProcessor::SetupContext()) {
    return false;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  void* device = provider ? provider->device() : nullptr;
  if (!device) {
    XELOGE("MetalCommandProcessor: no Metal device from provider");
    return false;
  }
  pipeline_cache_ = std::make_unique<MetalPipelineCache>(device);
  if (!pipeline_cache_->Initialize()) {
    XELOGE("MetalCommandProcessor: pipeline cache init failed");
    return false;
  }
  render_target_cache_ = std::make_unique<MetalRenderTargetCache>(device);
  return true;
}

void MetalCommandProcessor::ShutdownContext() {
  render_target_cache_.reset();
  pipeline_cache_.reset();
  CommandProcessor::ShutdownContext();
}

void MetalCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr,
                                      uint32_t frontbuffer_width,
                                      uint32_t frontbuffer_height) {
  // Bring-up slice: prove the guest -> swap -> present path end-to-end before
  // any real render target exists. We don't yet read the guest frontbuffer; we
  // just fill the mailbox texture with a per-frame test color so the window
  // visibly cycles (blue -> cycling color) when frames flow.
  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    return;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue()) {
    return;
  }
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)provider->command_queue();

  uint32_t count = swap_count_++;

  // Present the IssueDraw-produced render target if one exists this frame;
  // otherwise fall back to the cycling test pattern (so no-draw frames still
  // show liveness). Copy the source into the presenter's mailbox texture.
  id<MTLTexture> source_rt = nil;
  uint32_t present_width = 1280;
  uint32_t present_height = 720;
  if (frame_has_render_ && render_target_cache_ &&
      render_target_cache_->color_target()) {
    source_rt = (__bridge id<MTLTexture>)render_target_cache_->color_target();
    present_width = render_target_cache_->width();
    present_height = render_target_cache_->height();
  }
  if (count == 0 || (count % 60) == 0) {
    XELOGI("MetalCommandProcessor: IssueSwap #{} ({}x{}, ptr {:08X}, {})", count,
           present_width, present_height, frontbuffer_ptr,
           source_rt ? "render target" : "test pattern");
  }

  // Test-pattern color (used only when there's no render target this frame).
  double t = double(count % 120) / 120.0;
  double pattern_r = t;
  double pattern_g = 1.0 - t;
  double pattern_b = 0.5;

  auto refresher =
      [&](ui::Presenter::GuestOutputRefreshContext& context) -> bool {
    auto& metal_context =
        static_cast<xe::ui::metal::MetalGuestOutputRefreshContext&>(context);
    id<MTLTexture> mailbox =
        (__bridge id<MTLTexture>)metal_context.target_texture();
    if (!mailbox) {
      return false;
    }
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    if (source_rt) {
      // Copy the rendered target into the mailbox (same size + format).
      id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
      [blit copyFromTexture:source_rt
                sourceSlice:0
                sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(present_width, present_height, 1)
                  toTexture:mailbox
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
    } else {
      MTLRenderPassDescriptor* pass =
          [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = mailbox;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      pass.colorAttachments[0].clearColor =
          MTLClearColorMake(pattern_r, pattern_g, pattern_b, 1.0);
      id<MTLRenderCommandEncoder> encoder =
          [command_buffer renderCommandEncoderWithDescriptor:pass];
      [encoder endEncoding];
    }
    [command_buffer commit];
    // The presenter samples this texture from the UI thread later; make sure
    // the write is done before we report the frame ready.
    [command_buffer waitUntilCompleted];
    return true;
  };
  presenter->RefreshGuestOutput(present_width, present_height, /*aspect_x=*/16,
                                /*aspect_y=*/9, refresher);
  frame_has_render_ = false;
}

Shader* MetalCommandProcessor::LoadShader(xenos::ShaderType shader_type,
                                          uint32_t guest_address,
                                          const uint32_t* host_address,
                                          uint32_t dword_count) {
  if (!pipeline_cache_) {
    return nullptr;
  }
  return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
}

bool MetalCommandProcessor::IssueDraw(xenos::PrimitiveType prim_type,
                                      uint32_t index_count,
                                      IndexBufferInfo* index_buffer_info,
                                      bool major_mode_explicit) {
  // Bring-up slice: no geometry yet. On the first draw of a frame, establish a
  // real color render target and clear it to a distinctly synthetic color, so
  // IssueSwap can present an IssueDraw-produced RT (proving the RT lifecycle +
  // RT -> present copy) rather than a fabricated swap-time color. Real pipeline
  // binding + geometry is the next slice.
  if (!render_target_cache_) {
    return true;
  }
  auto* provider =
      static_cast<xe::ui::metal::MetalProvider*>(graphics_system_->provider());
  if (!provider || !provider->command_queue()) {
    return true;
  }
  void* rt = render_target_cache_->EnsureColorTarget(1280, 720);
  if (!rt) {
    return true;
  }
  if (!frame_has_render_) {
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)provider->command_queue();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)rt;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    // Synthetic dark green: unmistakably "our render target", not the game.
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.05, 0.20, 0.08, 1.0);
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    [encoder endEncoding];
    [command_buffer commit];
    frame_has_render_ = true;
  }
  return true;
}

bool MetalCommandProcessor::IssueCopy() { return true; }

void MetalCommandProcessor::InitializeTrace() {}

// Compile this backend's own copy of the PM4 execute loop, statically bound to
// MetalCommandProcessor so its Issue*/LoadShader overrides are the ones called
// (mirrors the vulkan/d3d12 backends; included inside the namespace).
#define COMMAND_PROCESSOR MetalCommandProcessor
#include "../pm4_command_processor_implement.h"

}  // namespace metal
}  // namespace gpu
}  // namespace xe
