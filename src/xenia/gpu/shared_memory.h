/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_SHARED_MEMORY_H_
#define XENIA_GPU_SHARED_MEMORY_H_

#include <cstdlib>
#include <utility>
#include <vector>

#include "xenia/gpu/watch_admission.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {

// Manages memory for unconverted textures, resolve targets, vertex and index
// buffers that can be accessed from shaders with Xenon physical addresses, with
// system page size granularity.
class SharedMemory {
 public:
  static constexpr uint32_t kBufferSizeLog2 = 29;
  static constexpr uint32_t kBufferSize = 1 << kBufferSizeLog2;

  virtual ~SharedMemory();
  // Call in the implementation-specific ClearCache.
  virtual void ClearCache();
  virtual void SetSystemPageBlocksValidWithGpuDataWritten();

  typedef void (*GlobalWatchCallback)(
      const global_unique_lock_type& global_lock, void* context,
      uint32_t address_first, uint32_t address_last, bool invalidated_by_gpu);
  typedef void* GlobalWatchHandle;
  // Registers a callback invoked when something is invalidated in the GPU
  // memory copy by the CPU or (if triggered explicitly - such as by a resolve)
  // by the GPU. It will be fired for writes to pages previously requested, but
  // may also be fired regardless of whether it was used by GPU emulation - for
  // example, if the game changes protection level of a memory range containing
  // the watched range.
  //
  // The callback is called within the global critical region.
  GlobalWatchHandle RegisterGlobalWatch(GlobalWatchCallback callback,
                                        void* callback_context);
  void UnregisterGlobalWatch(GlobalWatchHandle handle);
  typedef void (*WatchCallback)(const global_unique_lock_type& global_lock,
                                void* context, void* data, uint64_t argument,
                                bool invalidated_by_gpu);
  typedef void* WatchHandle;
  // Registers a callback invoked when the specified memory range is invalidated
  // in the GPU memory copy by the CPU or (if triggered explicitly - such as by
  // a resolve) by the GPU. It will be fired for writes to pages previously
  // requested, but may also be fired regardless of whether it was used by GPU
  // emulation - for example, if the game changes protection level of a memory
  // range containing the watched range.
  //
  // Generally the context is the subsystem pointer (for example, the texture
  // cache), the data is the object (such as a texture), and the argument is
  // additional subsystem/object-specific data (such as whether the range
  // belongs to the base mip level or to the rest of the mips).
  //
  // Called with the global critical region locked. Do NOT watch or unwatch
  // ranges from within it! The watch for the callback is cancelled after the
  // callback - the handle becomes invalid.
  WatchHandle WatchMemoryRange(uint32_t start, uint32_t length,
                               WatchCallback callback, void* callback_context,
                               void* callback_data, uint64_t callback_argument);
  // Unregisters previously registered watched memory range.
  void UnwatchMemoryRange(WatchHandle handle);

  // Checks if the range has been updated, uploads new data if needed and
  // ensures the host GPU memory backing the range are resident. Returns true if
  // the range has been fully updated and is usable.
  bool RequestRange(uint32_t start, uint32_t length);

  // WATCH-ONLY admission for a consumer that reads guest RAM DIRECTLY instead
  // of through the shared buffer (the Metal legacy texture untile). Enables
  // CPU-write invalidation notifications for the admissible mapped pages of
  // [start, length) so an in-place guest rewrite fires the ordinary
  // MemoryInvalidationCallback -> FireWatches path, and returns the admitted
  // byte count (0 = nothing admissible, which is NOT an error).
  //
  // Deliberately NOT RequestRange: it uploads nothing and marks NO page valid,
  // because the shared buffer does not hold this content and must not claim to
  // — a valid bit here would make a later genuine RequestRange skip a needed
  // upload. Unmapped holes anywhere in the extent are skipped, never fatal, so
  // a caller can always fall back to serving the bytes it already read. Global
  // invalidation ownership stays here: this only widens which pages notify,
  // and reuses the single physical-memory callback registered in the
  // constructor. See xenia/gpu/watch_admission.h for the admission rule.
  uint32_t RequestWatchOnlyRange(uint32_t start, uint32_t length);

  void TryFindUploadRange(const uint32_t& block_first,
                          const uint32_t& block_last,
                          const uint32_t& page_first, const uint32_t& page_last,
                          uint32_t& range_start,
                          unsigned int& current_upload_range,
                          std::pair<uint32_t, uint32_t>* uploads);

  void TryGetNextUploadRange(uint32_t& range_start, uint64_t& block_valid,
                             const uint32_t& i,
                             unsigned int& current_upload_range,
                             std::pair<uint32_t, uint32_t>* uploads);

  // Marks the range and, if not exact_range, potentially its surroundings
  // (to up to the first GPU-written page, as an access violation exception
  // count optimization) as modified by the CPU, also invalidating GPU-written
  // pages directly in the range. Backend publications that already notify a
  // global cache directly may set fire_global_watches=false to avoid duplicate
  // global invalidation while still firing every per-range watch.
  // Trace-playback restore variant: invalidates the CPU->GPU copy of the
  // range EXCEPT pages the GPU itself wrote and that are still valid (resolve
  // and memexport output produced earlier in this replay). A kMemoryRead
  // restore carries what the GPU read at capture time; for pages this replay's
  // GPU already produced, the replay's own result is at least as authoritative
  // and re-uploading the captured bytes over it clobbers in-frame resolves
  // (Halo 3's light passes sampled a stale normal G-buffer on Vulkan).
  void InvalidateRangeKeepingGpuWritten(uint32_t physical_address_start,
                                        uint32_t length);
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallback(
      uint32_t physical_address_start, uint32_t length, bool exact_range,
      bool fire_global_watches = true);

  // Marks the range as containing GPU-generated data (such as resolves),
  // triggering modification callbacks, making it valid (so pages are not
  // copied from the main memory until they're modified by the CPU) and
  // protecting it. Before writing anything from the GPU side, RequestRange must
  // be called, to make sure, if the GPU writes don't overwrite *everything* in
  // the pages they touch, the CPU data is properly loaded to the unmodified
  // regions in those pages.
  void RangeWrittenByGpu(uint32_t start, uint32_t length);

  // Diagnostic-only page-state peek (reopen-metal-gpu-visible-resolve-
  // authority 3.1 clobber hunt); not synchronized — callers hold the CP
  // thread like every other validity user.
  bool IsPageValid(uint32_t page) const {
    return (system_page_flags_valid_[page >> 6] >> (page & 63)) & 1;
  }
  bool IsPageGpuWritten(uint32_t page) const {
    return (system_page_flags_valid_and_gpu_written_[page >> 6] >>
            (page & 63)) &
           1;
  }
  // Every page of [start, end) is uploaded (so watched for guest writes) and
  // holds guest-written rather than GPU-written data. Same threading as above.
  bool IsRangeValidAndNotGpuWritten(uint32_t start, uint32_t end) const {
    for (uint32_t page = start >> page_size_log2_,
                  page_last = (end - 1) >> page_size_log2_;
         page <= page_last; ++page) {
      if (!IsPageValid(page) || IsPageGpuWritten(page)) {
        return false;
      }
    }
    return true;
  }
  // Clobber hunt window (XE_METAL_UPLOAD_WATCH=<hex base>): its length,
  // XE_METAL_UPLOAD_WATCH_LEN=<hex>, default 256 KB.
  static uint64_t UploadWatchLength() {
    static const uint64_t length = [] {
      const char* v = std::getenv("XE_METAL_UPLOAD_WATCH_LEN");
      return v ? strtoull(v, nullptr, 16) : uint64_t(0x40000);
    }();
    return length;
  }
  // Diagnostic (XE_METAL_STALE_SCAN): the uploaded pages (valid, not
  // GPU-written, not watch-only) whose bytes in `mirror` (the host GPU copy,
  // offset = guest physical address) differ from guest RAM, coalesced into
  // [start, end) ranges. Only meaningful with the GPU idle and no upload in
  // flight. Returns the number of pages compared.
  uint32_t FindStaleMirrorRanges(
      const uint8_t* mirror,
      std::vector<std::pair<uint32_t, uint32_t>>* ranges);
  // Every page of [start, end) is valid (uploaded or GPU-written).
  bool IsRangeValid(uint32_t start, uint32_t end) const {
    for (uint32_t page = start >> page_size_log2_,
                  page_last = (end - 1) >> page_size_log2_;
         page <= page_last; ++page) {
      if (!IsPageValid(page)) {
        return false;
      }
    }
    return true;
  }

 protected:
  SharedMemory(Memory& memory);

  // Registered directly as the one physical-memory invalidation callback.
  // Protected so conformance subclasses can exercise the exact production
  // dispatch boundary without installing a second watch.
  static std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length,
      bool exact_range);

  // The PHYSICAL guest-CPU write observation point, invoked by the thunk above
  // before the invalidation itself runs. Default no-op.
  //
  // It hangs off the THUNK and not off MemoryInvalidationCallback deliberately.
  // The thunk is the only entry the guest's access-violation path reaches, so
  // this sees exactly the writes the guest CPU performed. Backends also call
  // MemoryInvalidationCallback DIRECTLY to publish their own GPU writes
  // (resolves, memexports, trace restore); those are the emulator's work
  // wearing the invalidation call's name, and counting them as guest CPU writes
  // would make any CPU/GPU overlap evidence describe the backend observing
  // itself.
  //
  // Runs on the FAULTING GUEST THREAD with the global critical region ALREADY
  // HELD by the access-violation path (PhysicalHeap::TriggerCallbacks walks the
  // callback list under the lock it was handed). So this sits on the critical
  // path of every guest fault: an override may do bounded, lock-free work only
  // -- no waiting, no logging, no allocation, no new lock, and no call into
  // command-processor state beyond an atomic read. That the lock is held is
  // also what publishes an override's setup: an owner that installs state
  // before registering the callback, and releases it after unregistering,
  // synchronizes with this thread through the same region.
  virtual void ObservePhysicalCpuWrite(uint32_t physical_address_start,
                                       uint32_t length) {}

 public:
  Memory& guest_memory() const { return memory_; }

 protected:
  // Call in implementation-specific initialization.
  bool InitializeCommon();
  void InitializeSparseHostGpuMemory(uint32_t granularity_log2);
  // Call last in implementation-specific shutdown, also callable from the
  // destructor.
  void ShutdownCommon();

  // Sparse allocations are 4 MB, so not too many of them are allocated, but
  // also not to waste too much memory for padding (with 16 MB there's too
  // much).
  static constexpr uint32_t kHostGpuMemoryOptimalSparseAllocationLog2 = 22;
  static_assert(kHostGpuMemoryOptimalSparseAllocationLog2 <= kBufferSizeLog2);

  Memory& memory() const { return memory_; }

  uint32_t page_size_log2() const { return page_size_log2_; }

  uint32_t host_gpu_memory_sparse_granularity_log2() const {
    return host_gpu_memory_sparse_granularity_log2_;
  }

  // Allocations in the host buffer are aligned the same way as in the guest
  // physical memory (for instance, if an allocation is 64 KB, it can represent
  // 0-64 KB, 64-128 KB, 128-192 KB in the guest memory, and so on, but not
  // something like 16-80 KB. This is assumed by the rules for texture data
  // access in the texture cache.
  virtual bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                uint32_t length_allocations);

  // Mark the memory range as updated and protect it.
  void MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu);

  // Fires ONLY the per-range watches for [start, start+length) — texture
  // entries watching these pages go outdated and reload — without the global
  // watch fan-out (whose default-scope invalidation retires resolve bridges)
  // and without touching page validity. For backends publishing GPU-written
  // ranges whose consumers are served under GPU ordering
  // (reopen-metal-gpu-visible-resolve-authority): the content-change signal
  // must reach the watching textures even though no drain runs before the
  // consuming draw.
  // Byte-precise: a texture sharing a host page with the range but none of
  // its bytes (16 KB pages on Apple hosts) is not invalidated.
  void FireRangeWatchesOnly(uint32_t start, uint32_t length,
                            bool invalidated_by_gpu);

  // Uploads a range of host pages - only called if host GPU sparse memory
  // allocation succeeded if needed. While uploading, MakeRangeValid must be
  // called for each successfully uploaded range as early as possible, before
  // the memcpy, to make sure invalidation that happened during the CPU -> GPU
  // memcpy isn't missed (upload_page_ranges is in pages because of this -
  // MakeRangeValid has page granularity). upload_page_ranges are sorted in
  // ascending address order, so front and back can be used to determine the
  // overall bounds of pages to be uploaded.
  virtual bool UploadRanges(
      const std::pair<uint32_t, uint32_t>* upload_page_ranges,
      uint32_t num_upload_ranges) = 0;

  const std::vector<std::pair<uint32_t, uint32_t>>& trace_download_ranges() {
    return trace_download_ranges_;
  }
  uint32_t trace_download_page_count() const {
    return trace_download_page_count_;
  }
  // Fills trace_download_ranges() and trace_download_page_count() with
  // GPU-written ranges that need to be downloaded, and also invalidates
  // non-GPU-written ranges so only the needed data - not the all the collected
  // data - will be written in the trace. trace_download_page_count() will be 0
  // if nothing to download.
  void PrepareForTraceDownload();
  // Release memory used for trace download ranges, to be called after
  // downloading or in cases when download is dropped.
  void ReleaseTraceDownloadRanges();

 private:
  Memory& memory_;

  // Log2 of invalidation granularity (the system page size, but the dependency
  // on it is not hard - the access callback takes a range as an argument, and
  // touched pages of the buffer of this size will be invalidated).
  uint32_t page_size_log2_;

  bool EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length);
  uint32_t host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;
  std::vector<uint64_t> host_gpu_memory_sparse_allocated_;
  uint32_t host_gpu_memory_sparse_allocations_ = 0;
  uint32_t host_gpu_memory_sparse_used_bytes_ = 0;

  void* memory_invalidation_callback_handle_ = nullptr;
  void* memory_data_provider_handle_ = nullptr;
  static constexpr unsigned int MAX_UPLOAD_RANGES = 65536;

  // Ranges that need to be uploaded, generated by GetRangesToUpload (a
  // persistently allocated vector).
  // std::vector<std::pair<uint32_t, uint32_t>> upload_ranges_;
  FixedVMemVector<MAX_UPLOAD_RANGES * sizeof(std::pair<uint32_t, uint32_t>)>
      upload_ranges_;
  // Scratch for RequestWatchOnlyRange's admission scan (GPU thread only, reused
  // to keep the per-upload watch registration allocation-free in steady state).
  std::vector<WatchAdmissionRun> watch_admission_runs_;

  // Mutex between the guest memory subsystem and the command processor, to be
  // locked when checking or updating validity of pages/ranges and when firing
  // watches.
  static constexpr xe::global_critical_region global_critical_region_{};

  // ***************************************************************************
  // Things below should be fully protected by global_critical_region.
  // ***************************************************************************

  struct SystemPageFlagsBlock {
    // Whether each page is up to date in the GPU buffer.
    uint64_t valid;
    // Subset of valid pages - whether each page in the GPU buffer contains data
    // that was written on the GPU, thus should not be invalidated spuriously.
    uint64_t valid_and_gpu_written;
  };

  // chrispy: todo, systempageflagsblock should be 3 different arrays
  //  Flags for each 64 system pages, interleaved as blocks, so bit scan can be
  //  used to quickly extract ranges.
  // std::vector<SystemPageFlagsBlock> system_page_flags_;

  uint64_t *system_page_flags_valid_ = nullptr,
           *system_page_flags_valid_and_gpu_written_ = nullptr;
  unsigned num_system_page_flags_ = 0;
  // Pages admitted by RequestWatchOnlyRange (a cached texture reads them
  // directly from RAM) and not invalidated since. The non-exact widening in
  // MemoryInvalidationCallback stops at them like at GPU-written pages: with
  // 16 KB host pages the widening reaches a 1 MB block, so one CPU write to a
  // dynamic buffer unwatched -- and evicted -- every texture in the block
  // (Red Dead gameplay: ~75 writes, ~440 evictions per frame, all but ~3 of
  // them byte-identical on revalidation). A hint only: a stale bit narrows
  // the widening, which can add a fault, never miss one. Needs the texture
  // cache's capacity eviction (XE_METAL_TEX_SLOT_LRU): the churn was what
  // kept its bindless heap from filling.
  // XE_GPU_WATCH_ONLY_WIDEN_BARRIER=0 restores the full widening.
  std::vector<uint64_t> system_page_flags_watch_only_;
  static bool WatchOnlyWidenBarrierEnabled();
  static bool RequestRangeFastPathEnabled();
  struct GlobalWatch {
    GlobalWatchCallback callback;
    void* callback_context;
  };
  std::vector<GlobalWatch*> global_watches_;
  struct WatchNode;
  // Watched range placed by other GPU subsystems.
  struct WatchRange {
    union {
      struct {
        WatchCallback callback;
        void* callback_context;
        void* callback_data;
        uint64_t callback_argument;
        WatchNode* node_first;
        uint32_t page_first;
        uint32_t page_last;
        // The watched bytes, inclusive, for FireRangeWatchesOnly.
        uint32_t address_first;
        uint32_t address_last;
      };
      WatchRange* next_free;
    };
  };
  // Node for faster checking of watches when pages have been written to - all
  // 512 MB are split into smaller equally sized buckets, and then ranges are
  // linearly checked.
  struct WatchNode {
    union {
      struct {
        WatchRange* range;
        // Link to another node of this watched range in the next bucket.
        WatchNode* range_node_next;
        // Links to nodes belonging to other watched ranges in the bucket.
        WatchNode* bucket_node_previous;
        WatchNode* bucket_node_next;
      };
      WatchNode* next_free;
    };
  };
  static constexpr uint32_t kWatchBucketSizeLog2 = 22;
  static constexpr uint32_t kWatchBucketCount =
      1 << (kBufferSizeLog2 - kWatchBucketSizeLog2);
  WatchNode* watch_buckets_[kWatchBucketCount] = {};
  // Allocation from pools - taking new WatchRanges and WatchNodes from the free
  // list, and if there are none, creating a pool if the current one is fully
  // used, and linearly allocating from the current pool.
  static constexpr uint32_t kWatchRangePoolSize = 8192;
  static constexpr uint32_t kWatchNodePoolSize = 8192;
  std::vector<WatchRange*> watch_range_pools_;
  std::vector<WatchNode*> watch_node_pools_;
  uint32_t watch_range_current_pool_allocated_ = 0;
  uint32_t watch_node_current_pool_allocated_ = 0;
  WatchRange* watch_range_first_free_ = nullptr;
  WatchNode* watch_node_first_free_ = nullptr;

  // GPU-written memory downloading for traces. <Start address, length>.
  std::vector<std::pair<uint32_t, uint32_t>> trace_download_ranges_;
  uint32_t trace_download_page_count_ = 0;

  // Triggers the watches (global and per-range), removing triggered range
  // watches.
  // Range watches are fired when they share a page with [page_first,
  // page_last] and, given exact bytes, a byte of [byte_first, byte_last].
  void FireWatches(uint32_t page_first, uint32_t page_last,
                   bool invalidated_by_gpu, bool fire_global_watches = true,
                   uint32_t byte_first = 0, uint32_t byte_last = UINT32_MAX);
  // Unlinks and frees the range and its nodes. Call this in the global critical
  // region.
  void UnlinkWatchRange(WatchRange* range);
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_SHARED_MEMORY_H_
