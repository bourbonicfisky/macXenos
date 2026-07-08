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
  return true;
}

void MetalCommandProcessor::ShutdownContext() {
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

  uint32_t width = frontbuffer_width ? frontbuffer_width : 1280;
  uint32_t height = frontbuffer_height ? frontbuffer_height : 720;
  uint32_t count = swap_count_++;
  if (count == 0 || (count % 60) == 0) {
    XELOGI("MetalCommandProcessor: IssueSwap #{} ({}x{}, ptr {:08X})", count,
           width, height, frontbuffer_ptr);
  }
  // Cycle a color so motion is obvious on-device.
  double t = double(count % 120) / 120.0;
  double r = t;
  double g = 1.0 - t;
  double b = 0.5;

  auto refresher =
      [&](ui::Presenter::GuestOutputRefreshContext& context) -> bool {
    auto& metal_context =
        static_cast<xe::ui::metal::MetalGuestOutputRefreshContext&>(context);
    id<MTLTexture> texture =
        (__bridge id<MTLTexture>)metal_context.target_texture();
    if (!texture) {
      return false;
    }
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    MTLRenderPassDescriptor* pass =
        [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(r, g, b, 1.0);
    id<MTLRenderCommandEncoder> encoder =
        [command_buffer renderCommandEncoderWithDescriptor:pass];
    [encoder endEncoding];
    [command_buffer commit];
    // The presenter samples this texture from the UI thread later; make sure
    // the write is done before we report the frame ready.
    [command_buffer waitUntilCompleted];
    return true;
  };
  presenter->RefreshGuestOutput(width, height, /*aspect_x=*/16,
                                /*aspect_y=*/9, refresher);
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
