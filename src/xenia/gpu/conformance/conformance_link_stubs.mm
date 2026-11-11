/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Definitions the lean conformance closure needs from translation units it
// does not link (command_processor.cc and metal_provider.mm each pull the
// monolithic GPU and UI closures).

#import <Metal/Metal.h>

#include "xenia/gpu/command_processor.h"
#include "xenia/ui/metal/metal_provider.h"

namespace xe {
namespace gpu {

// The guest RAM write watch is a live diagnostic; no test arms it.
bool GuestRamWatchOverlaps(uint32_t base, uint32_t length) { return false; }

}  // namespace gpu

namespace ui {
namespace metal {

// The tests run on the system default device (no metal_device selection).
void* MetalProvider::SelectedDevice() {
  static id<MTLDevice> s_device = MTLCreateSystemDefaultDevice();
  return (__bridge void*)s_device;
}

}  // namespace metal
}  // namespace ui
}  // namespace xe
