/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_GRAPHICS_SYSTEM_H_
#define XENIA_GPU_METAL_METAL_GRAPHICS_SYSTEM_H_

#include <memory>
#include <string>

#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"

namespace xe {
namespace gpu {
namespace metal {

// Experimental native-Metal graphics system. Phase 2b skeleton: it registers as
// the "metal" backend and drives a MetalCommandProcessor, but does not present
// yet (mirrors the null backend's headless setup on macOS). The render path is
// filled in by later phases; the shader translation back half already exists in
// this directory (dxbc_to_dxil_converter / metal_shader_converter).
class MetalGraphicsSystem : public GraphicsSystem {
 public:
  MetalGraphicsSystem();
  ~MetalGraphicsSystem() override;

  // Whether this host can run the Metal backend. macOS on Apple Silicon always
  // provides a Metal 3 device; a stricter check (Argument Buffers Tier 2 +
  // macOS 14, required by Metal Shader Converter) is added with the render path.
  static bool IsAvailable();

  std::string name() const override { return "Metal"; }

  X_STATUS Setup(cpu::Processor* processor, kernel::KernelState* kernel_state,
                 ui::WindowedAppContext* app_context,
                 bool with_presentation) override;

 private:
  std::unique_ptr<CommandProcessor> CreateCommandProcessor() override;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_GRAPHICS_SYSTEM_H_
