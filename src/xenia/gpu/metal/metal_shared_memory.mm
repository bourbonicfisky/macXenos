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

#include <cstring>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/memory.h"

DECLARE_bool(gpu_allow_invalid_upload_range);

namespace xe {
namespace gpu {
namespace metal {

MetalSharedMemory::MetalSharedMemory(Memory& memory, void* device)
    : SharedMemory(memory), device_(device) {}

MetalSharedMemory::~MetalSharedMemory() { Shutdown(); }

bool MetalSharedMemory::Initialize() {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  // Shared storage so uploads are plain host memcpys (unified memory on Apple
  // Silicon); write-combined is fine because nothing reads it from the CPU.
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
  InitializeCommon();
  return true;
}

void MetalSharedMemory::Shutdown() {
  ShutdownCommon();
  consumed_ranges_.clear();
  if (buffer_) {
    id<MTLBuffer> released = (__bridge_transfer id<MTLBuffer>)buffer_;
    (void)released;
    buffer_ = nullptr;
    buffer_contents_ = nullptr;
  }
}

bool MetalSharedMemory::RequestConsumedRange(uint32_t start, uint32_t length) {
  if (!length) {
    return true;
  }
  // RequestRange uploads only the pages invalidated since their last use; a
  // hazard flush (see UploadRanges) may run inside.
  bool result = RequestRange(start, length);
  uint32_t end = start + length;
  for (const auto& range : consumed_ranges_) {
    if (start >= range.first && end <= range.second) {
      return result;  // already covered.
    }
  }
  consumed_ranges_.emplace_back(start, end);
  return result;
}

bool MetalSharedMemory::UploadRanges(
    const std::pair<uint32_t, uint32_t>* upload_page_ranges,
    uint32_t num_upload_ranges) {
  if (!num_upload_ranges) {
    return true;
  }
  if (!buffer_contents_) {
    return false;
  }
  // Only invalidated pages reach here, so any overlap with a range a draw in
  // the open frame command buffer consumed is a genuine intra-frame rewrite:
  // flush (commit + wait) BEFORE the new bytes land so those draws read the
  // old data. The callback also resets consumed_ranges_.
  bool hazard = false;
  // Experiment (XE_METAL_FLUSH_ALL_UPLOADS): flush before ANY mid-frame
  // invalidating upload once a draw has consumed shared memory this frame,
  // ignoring the overlap test. Over-approximates per-draw serialization for the
  // shared-memory path (no arena coupling) — an in-agent A/B for whether the
  // stipple is shared-memory upload staleness at all.
  static const bool s_flush_all =
      std::getenv("XE_METAL_FLUSH_ALL_UPLOADS") != nullptr;
  if (s_flush_all && !consumed_ranges_.empty()) {
    hazard = true;
  }
  for (uint32_t i = 0; i < num_upload_ranges && !hazard; ++i) {
    uint32_t range_start = upload_page_ranges[i].first << page_size_log2();
    uint32_t range_end = range_start
                         + (upload_page_ranges[i].second << page_size_log2());
    for (const auto& consumed : consumed_ranges_) {
      if (range_start < consumed.second && range_end > consumed.first) {
        hazard = true;
        break;
      }
    }
  }
  if (hazard && hazard_flush_callback_) {
    static uint32_t s_hazard_flushes = 0;
    if (s_hazard_flushes++ < 8) {
      XELOGI("MetalSharedMemory: intra-frame rewrite hazard flush #{}",
             s_hazard_flushes);
    }
    hazard_flush_callback_();
  }
  for (uint32_t i = 0; i < num_upload_ranges; ++i) {
    uint32_t upload_start = upload_page_ranges[i].first << page_size_log2();
    uint32_t upload_length = upload_page_ranges[i].second << page_size_log2();
    if (!upload_length) {
      continue;
    }
    if (!cvars::gpu_allow_invalid_upload_range) {
      uint32_t last_page_start =
          upload_start + upload_length - (1u << page_size_log2());
      if (memory().GetPhysicalHeap()->QueryRangeAccess(
              upload_start, upload_start) == xe::memory::PageAccess::kNoAccess ||
          memory().GetPhysicalHeap()->QueryRangeAccess(
              last_page_start, last_page_start) ==
              xe::memory::PageAccess::kNoAccess) {
        XELOGE("MetalSharedMemory: invalid upload range {:08X}+{:X}",
               upload_start, upload_length);
        return false;
      }
    }
    // Validate BEFORE the memcpy so a CPU write racing the copy re-invalidates
    // the pages rather than being lost (base-class contract).
    MakeRangeValid(upload_start, upload_length, false);
    std::memcpy(buffer_contents_ + upload_start,
                memory().TranslatePhysical(upload_start), upload_length);
  }
  return true;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
