/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_METAL_METAL_PROVIDER_H_
#define XENIA_UI_METAL_METAL_PROVIDER_H_

#include <memory>

#include "xenia/ui/graphics_provider.h"

namespace xe {
namespace ui {
namespace metal {

// Phase 2b.3 Metal graphics provider. Owns the MTLDevice and a command queue and
// vends the Metal presenter / immediate drawer. Metal objects are held as opaque
// pointers (retained via __bridge_retained) so this header stays plain C++.
class MetalProvider : public GraphicsProvider {
 public:
  ~MetalProvider() override;

  // Creates the provider using the system default Metal device. Returns null if
  // no Metal device is available.
  static std::unique_ptr<MetalProvider> Create();

  // id<MTLDevice> / id<MTLCommandQueue>.
  void* device() const { return device_; }
  void* command_queue() const { return command_queue_; }

  std::unique_ptr<Presenter> CreatePresenter(
      Presenter::HostGpuLossCallback host_gpu_loss_callback =
          Presenter::FatalErrorHostGpuLossCallback) override;

  std::unique_ptr<ImmediateDrawer> CreateImmediateDrawer() override;

 private:
  MetalProvider() = default;

  bool Initialize();

  void* device_ = nullptr;         // id<MTLDevice> (retained)
  void* command_queue_ = nullptr;  // id<MTLCommandQueue> (retained)
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_METAL_PROVIDER_H_
