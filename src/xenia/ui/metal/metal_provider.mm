/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/metal/metal_provider.h"

#import <Metal/Metal.h>

#include "xenia/base/logging.h"
#include "xenia/ui/metal/metal_immediate_drawer.h"
#include "xenia/ui/metal/metal_presenter.h"

namespace xe {
namespace ui {
namespace metal {

MetalProvider::~MetalProvider() {
  // Balance the __bridge_retained transfers from Initialize(): move ownership
  // back into ARC so the objects are released.
  if (command_queue_) {
    id<MTLCommandQueue> queue = (__bridge_transfer id<MTLCommandQueue>)command_queue_;
    (void)queue;
    command_queue_ = nullptr;
  }
  if (device_) {
    id<MTLDevice> device = (__bridge_transfer id<MTLDevice>)device_;
    (void)device;
    device_ = nullptr;
  }
}

std::unique_ptr<MetalProvider> MetalProvider::Create() {
  std::unique_ptr<MetalProvider> provider(new MetalProvider());
  if (!provider->Initialize()) {
    return nullptr;
  }
  return provider;
}

bool MetalProvider::Initialize() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    XELOGE("MetalProvider: no system default Metal device");
    return false;
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) {
    XELOGE("MetalProvider: failed to create command queue");
    return false;
  }
  // Transfer ownership out of ARC into the retained opaque pointers.
  device_ = (__bridge_retained void*)device;
  command_queue_ = (__bridge_retained void*)queue;
  XELOGI("MetalProvider: initialized on {}",
         [[device name] UTF8String] ? [[device name] UTF8String] : "unknown");
  return true;
}

std::unique_ptr<Presenter> MetalProvider::CreatePresenter(
    Presenter::HostGpuLossCallback host_gpu_loss_callback) {
  return MetalPresenter::Create(host_gpu_loss_callback, device_, command_queue_);
}

std::unique_ptr<ImmediateDrawer> MetalProvider::CreateImmediateDrawer() {
  return std::make_unique<MetalImmediateDrawer>();
}

}  // namespace metal
}  // namespace ui
}  // namespace xe
