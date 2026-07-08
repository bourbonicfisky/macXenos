/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_shared_memory.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>

#include "xenia/base/logging.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace metal {

MetalSharedMemory::MetalSharedMemory(void* device, Memory* memory)
    : device_(device), memory_(memory) {}

MetalSharedMemory::~MetalSharedMemory() {
  if (buffer_) {
    id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)buffer_;
    (void)released;
    buffer_ = nullptr;
  }
}

bool MetalSharedMemory::Initialize() {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  // Shared storage so we can memcpy guest RAM into it directly (unified memory
  // on Apple Silicon makes this a plain host write, no staging/blit).
  id<MTLBuffer> buffer =
      [device newBufferWithLength:kBufferSize
                          options:MTLResourceStorageModeShared |
                                  MTLResourceCPUCacheModeWriteCombined];
  if (!buffer) {
    XELOGE("MetalSharedMemory: failed to allocate {} MiB shared buffer",
           kBufferSize / (1024 * 1024));
    return false;
  }
  buffer_ = (__bridge_retained void*)buffer;
  buffer_contents_ = static_cast<uint8_t*>([buffer contents]);
  return true;
}

void MetalSharedMemory::UploadRange(uint32_t guest_address, uint32_t length) {
  if (!buffer_contents_ || !length) {
    return;
  }
  uint64_t end = uint64_t(guest_address) + length;
  if (guest_address >= kBufferSize) {
    return;
  }
  if (end > kBufferSize) {
    length = uint32_t(kBufferSize - guest_address);
  }
  const void* guest = memory_->TranslatePhysical(guest_address);
  if (!guest) {
    return;
  }
  std::memcpy(buffer_contents_ + guest_address, guest, length);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
