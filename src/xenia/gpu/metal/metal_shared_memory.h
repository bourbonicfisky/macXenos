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
#include <functional>
#include <utility>
#include <vector>

#include "xenia/gpu/shared_memory.h"

namespace xe {
class Memory;
namespace gpu {
class TraceWriter;
namespace metal {

// Real xe::gpu::SharedMemory subclass over a single 512 MiB shared-storage
// MTLBuffer where buffer offset == guest physical byte address, so the
// translated vertex shader's vfetch reads straight out of it. The base class
// provides page-granular validity tracking with CPU write-watch invalidation:
// uploads happen only for pages actually written since their last use. (The
// bring-up shim this replaces memcmp'd entire fetch ranges per draw — the
// terrain mega-streams are tens of MB, which became tens of GB of comparison
// per frame once real gameplay rendered.) The buffer is CPU-visible, so
// UploadRanges is a plain memcpy — no staging, no GPU commands.
class MetalSharedMemory : public SharedMemory {
 public:
  MetalSharedMemory(Memory& memory, void* device);
  ~MetalSharedMemory() override;

  bool Initialize();
  void Shutdown();

  // Frame-trace recording: every upload's source range is written into the
  // open trace (uploads read guest RAM, which is exactly the data a replay
  // must restore). WriteMemoryRead self-guards when no trace is open.
  void SetTraceWriter(TraceWriter* trace_writer) {
    trace_writer_ = trace_writer;
  }
  // Frame-trace init: invalidates every non-GPU-written page (so ranges the
  // traced frame consumes re-upload and get recorded) and writes collected
  // GPU-written ranges straight from guest RAM (CPU-authoritative here).
  void WriteTraceInitialState(TraceWriter& trace_writer);

  // id<MTLBuffer> (shared storage), not owned by the caller.
  void* GetBuffer() const { return buffer_; }

  // Frame-batching correctness: every draw in the open frame command buffer
  // reads the buffer's state at COMMIT, not at encode. When a re-upload is
  // about to land in pages a draw encoded this frame has already consumed
  // (the intra-frame dynamic rewrite: UI glyph rings, particles), the flush
  // callback (commit + wait, set by the command processor) runs FIRST so the
  // encoded draws read the old data.
  void SetHazardFlushCallback(std::function<void()> callback) {
    hazard_flush_callback_ = std::move(callback);
  }

  // RequestRange plus recording [start, start+length) as consumed by the
  // current frame for the hazard detection above.
  bool RequestConsumedRange(uint32_t start, uint32_t length);

  // Forgets the consumed-range set (call after the frame command buffer has
  // been committed and waited on).
  void ResetFrameRanges() { consumed_ranges_.clear(); }

 protected:
  bool UploadRanges(const std::pair<uint32_t, uint32_t>* upload_page_ranges,
                    uint32_t num_upload_ranges) override;

 private:
  TraceWriter* trace_writer_ = nullptr;
  void* device_;                        // id<MTLDevice>, not owned.
  void* buffer_ = nullptr;              // id<MTLBuffer>, retained.
  uint8_t* buffer_contents_ = nullptr;  // CPU-visible pointer into buffer_.

  // Byte ranges consumed by draws encoded in the open frame command buffer.
  std::vector<std::pair<uint32_t, uint32_t>> consumed_ranges_;
  std::function<void()> hazard_flush_callback_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHARED_MEMORY_H_
