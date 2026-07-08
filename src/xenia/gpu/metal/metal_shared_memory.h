/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_
#define XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_

#include <cstdint>

namespace xe {
class Memory;
namespace gpu {
namespace metal {

// Bring-up minimal shared memory: a single Metal buffer mirroring guest physical
// memory, where buffer offset == guest physical byte address, so the translated
// vertex shader's vfetch (which addresses guest physical memory) reads straight
// out of it. Uses shared (CPU+GPU) storage and uploads on demand by memcpy from
// guest RAM for the specific ranges a draw touches.
//
// This is deliberately NOT the proper xe::gpu::SharedMemory subclass (no
// dirty-page watches, no sparse residency, re-uploads the requested range every
// time). It exists to prove the MSC argument-buffer binding / vfetch path
// executes; once that's confirmed it should be replaced with a
// MetalSharedMemory : xe::gpu::SharedMemory that the stable Vulkan backend's
// base class already gets right.
class MetalSharedMemory {
 public:
  // 512 MiB, matching xe::gpu::SharedMemory::kBufferSize (guest physical space).
  static constexpr uint64_t kBufferSize = 512ull * 1024 * 1024;

  MetalSharedMemory(void* device, Memory* memory);
  ~MetalSharedMemory();

  bool Initialize();

  // Copies [guest_address, guest_address + length) from guest physical memory
  // into the shared buffer at the same offset. Clamped to the buffer size.
  void UploadRange(uint32_t guest_address, uint32_t length);

  // id<MTLBuffer> (shared storage), not owned by the caller.
  void* GetBuffer() const { return buffer_; }

 private:
  void* device_;        // id<MTLDevice>, not owned.
  Memory* memory_;      // guest memory, not owned.
  void* buffer_ = nullptr;  // id<MTLBuffer>, retained.
  uint8_t* buffer_contents_ = nullptr;  // CPU-visible pointer into buffer_.
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_
