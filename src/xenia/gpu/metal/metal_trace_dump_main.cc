/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/console_app_main.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/metal/metal_graphics_system.h"
#include "xenia/gpu/trace_dump.h"

namespace xe {
namespace gpu {
namespace metal {

// Headless trace replay through the native Metal backend, so a captured guest
// frame can be re-rendered + dumped off-device (no window, no CPU JIT of the
// title) for fast iteration on the Metal render path. Mirrors the vulkan/d3d12
// trace-dump mains; RenderDoc host capture is a no-op (Metal uses Xcode GPU
// capture instead, which isn't driven here).
class MetalTraceDump : public TraceDump {
 public:
  std::unique_ptr<gpu::GraphicsSystem> CreateGraphicsSystem() override {
    return std::unique_ptr<gpu::GraphicsSystem>(new MetalGraphicsSystem());
  }

  void BeginHostCapture() override {}
  void EndHostCapture() override {}
};

int trace_dump_main(const std::vector<std::string>& args) {
  MetalTraceDump trace_dump;
  return trace_dump.Main(args);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

XE_DEFINE_CONSOLE_APP("xenia-gpu-metal-trace-dump",
                      xe::gpu::metal::trace_dump_main, "some.trace",
                      "target_trace_file");
