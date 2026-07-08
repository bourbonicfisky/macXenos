/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_UPLOAD_ARENA_H_
#define XENIA_GPU_METAL_METAL_UPLOAD_ARENA_H_

#include <cstdint>

namespace xe {
namespace gpu {
namespace metal {

// A slice handed out by MetalUploadArena: where to write CPU-side, and the
// buffer/offset/gpuAddress to reference from the encoder / argument buffer.
struct MetalUploadAllocation {
  void* buffer = nullptr;     // id<MTLBuffer>
  uint32_t offset = 0;        // byte offset within buffer
  uint64_t gpu_address = 0;   // buffer.gpuAddress() + offset
  void* cpu_ptr = nullptr;    // buffer.contents() + offset
};

// Bring-up transient allocator: one shared MTLBuffer, bump-allocated within a
// frame and reset (offset back to 0) once that frame's GPU work has completed.
// Holds the top-level argument buffers and the b0-b4 constant slices for a
// frame's draws. Single-buffered for simplicity (the command processor waits on
// the frame command buffer before Reset), so no in-flight aliasing.
class MetalUploadArena {
 public:
  MetalUploadArena(void* device, uint32_t size);
  ~MetalUploadArena();

  bool Initialize();

  // Returns the bump offset to 0. Only safe once the GPU is done reading the
  // previously handed-out slices.
  void Reset();

  // Allocates `size` bytes aligned to `alignment`. Returns false (and leaves
  // `out` untouched) if the arena is exhausted.
  bool Allocate(uint32_t size, uint32_t alignment, MetalUploadAllocation* out);

 private:
  void* device_;   // id<MTLDevice>, not owned.
  uint32_t size_;
  uint32_t offset_ = 0;
  void* buffer_ = nullptr;  // id<MTLBuffer>, retained.
  uint8_t* contents_ = nullptr;
  uint64_t gpu_base_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_UPLOAD_ARENA_H_
