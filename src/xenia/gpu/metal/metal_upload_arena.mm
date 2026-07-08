/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_upload_arena.h"

#import <Metal/Metal.h>

#include "xenia/base/logging.h"
#include "xenia/base/math.h"

namespace xe {
namespace gpu {
namespace metal {

MetalUploadArena::MetalUploadArena(void* device, uint32_t size)
    : device_(device), size_(size) {}

MetalUploadArena::~MetalUploadArena() {
  if (buffer_) {
    id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)buffer_;
    (void)released;
    buffer_ = nullptr;
  }
}

bool MetalUploadArena::Initialize() {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  id<MTLBuffer> buffer =
      [device newBufferWithLength:size_
                          options:MTLResourceStorageModeShared |
                                  MTLResourceCPUCacheModeWriteCombined];
  if (!buffer) {
    XELOGE("MetalUploadArena: failed to allocate {} byte arena", size_);
    return false;
  }
  buffer_ = (__bridge_retained void*)buffer;
  contents_ = static_cast<uint8_t*>([buffer contents]);
  gpu_base_ = [buffer gpuAddress];
  return true;
}

void MetalUploadArena::Reset() { offset_ = 0; }

bool MetalUploadArena::Allocate(uint32_t size, uint32_t alignment,
                                MetalUploadAllocation* out) {
  if (!contents_ || !size) {
    return false;
  }
  uint32_t aligned = alignment ? xe::align(offset_, alignment) : offset_;
  if (uint64_t(aligned) + size > size_) {
    XELOGE("MetalUploadArena: exhausted ({} + {} > {})", aligned, size, size_);
    return false;
  }
  out->buffer = buffer_;
  out->offset = aligned;
  out->gpu_address = gpu_base_ + aligned;
  out->cpu_ptr = contents_ + aligned;
  offset_ = aligned + size;
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
