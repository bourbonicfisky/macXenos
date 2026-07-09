/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_graphics_system.h"

#include "xenia/base/logging.h"
#include "xenia/gpu/metal/metal_command_processor.h"
#include "xenia/ui/metal/metal_provider.h"
#include "xenia/xbox.h"

namespace xe {
namespace gpu {
namespace metal {

MetalGraphicsSystem::MetalGraphicsSystem() {}

MetalGraphicsSystem::~MetalGraphicsSystem() {}

bool MetalGraphicsSystem::IsAvailable() {
  // TODO(phase-2b): probe MTLCreateSystemDefaultDevice() and require Argument
  // Buffers Tier 2 + macOS 14 (the Metal Shader Converter runtime baseline).
  // Until the render path exists the backend is opt-in via --gpu metal on a
  // build configured with XENIA_GPU_METAL, so report available on Apple.
  return true;
}

X_STATUS MetalGraphicsSystem::Setup(cpu::Processor* processor,
                                    kernel::KernelState* kernel_state,
                                    ui::WindowedAppContext* app_context,
                                    bool with_presentation) {
  // The provider owns the Metal device + command queue and is needed for
  // rendering even WITHOUT presentation (e.g. the headless trace dump, which
  // renders offscreen and reads back via the presenter's CaptureGuestOutput).
  // The device needs no window; only the on-screen presenter depends on
  // with_presentation, and the base GraphicsSystem::Setup handles that (creating
  // an offscreen presenter when there's no app context). Mirrors the Vulkan
  // backend, which always creates its provider.
  provider_ = xe::ui::metal::MetalProvider::Create();
  if (!provider_) {
    XELOGE("MetalGraphicsSystem: failed to create Metal provider");
    return X_STATUS_UNSUCCESSFUL;
  }
  return GraphicsSystem::Setup(processor, kernel_state, app_context,
                               with_presentation);
}

std::unique_ptr<CommandProcessor>
MetalGraphicsSystem::CreateCommandProcessor() {
  return std::unique_ptr<CommandProcessor>(
      new MetalCommandProcessor(this, kernel_state_));
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
