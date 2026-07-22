/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_COMMAND_PROCESSOR_H_
#define XENIA_GPU_COMMAND_PROCESSOR_H_

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "xenia/base/math.h"
#include "xenia/base/ring_buffer.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/xthread.h"
#include "xenia/memory.h"
#include "xenia/ui/presenter.h"

DECLARE_int32(replay_draw_first);
DECLARE_int32(replay_draw_last);
DECLARE_int32(replay_draw_include);
DECLARE_int32(replay_draw_skip_first);
DECLARE_int32(replay_draw_skip_last);
DECLARE_bool(replay_skip_depth_only);
DECLARE_int32(replay_zfunc_always_draw);
DECLARE_int32(replay_solid_draw);
DECLARE_int32(replay_duplicate_draw_stencil);
DECLARE_bool(replay_duplicate_draw_depth_only);
DECLARE_bool(replay_draw_log);
DECLARE_string(replay_inject_memory);
DECLARE_string(replay_index_window);
DECLARE_string(capture_dir);
DECLARE_string(capture_draw_range);
DECLARE_string(capture_edram_at);
DECLARE_bool(capture_edram_hashes);

namespace xe {

class ByteStream;

namespace gpu {

enum class GPUSetting { ClearMemoryPageState, ReadbackMemexport };

enum class ReadbackResolveMode {
  kDisabled,  // No readback (none)
  kFast,      // Delayed sync, 1 frame behind (fast)
  kFull       // Immediate sync with GPU stall (full)
};

// Occlusion queries - ZPD report mode.
enum class ZPDMode {
  kFake,     // Fake sample counts, no real GPU queries (fake)
  kFast,     // Real queries with speculative cached writes (fast)
  kFastAlt,  // Fast queries, but preserves cached zeroes (fast-alt)
  kStrict,   // Real queries, waits before writeback (strict)
};

void SaveGPUSetting(GPUSetting setting, uint64_t value);
bool GetGPUSetting(GPUSetting setting);
ReadbackResolveMode GetReadbackResolveMode();
void SetReadbackResolveMode(const std::string& mode);
ZPDMode GetZPDMode();
void SetZPDMode(const std::string& mode);

// Shared pool capacity for D3D12 and Vulkan.
constexpr uint32_t kZPDQueryPoolCapacity = 8192;

// Contiguous range of query indices for batched resolve/copy operations.
struct ResolveRange {
  uint32_t start;
  uint32_t count;
};

// Backstop for strict mode. Abandon any pending retires after this many polls
// so EVENT_WRITE_ZPD doesn't keep spinning on an unresolved report.
constexpr uint32_t kStrictZPDRetireMaxStalls = 16;
// Clock backstop used for strict retire if guest polling is sparse.
constexpr uint64_t kStrictZPDRetireDeadlineMs = 2;

// Cap for the fast-mode cached delta map.  Games reuse a small set of report
// addresses so this should never be hit, but prevents unbounded growth if a
// title cycles through unique addresses.  Clearing the cache has no
// correctness impact - it only removes speculative writeback hints.
constexpr size_t kFastZPDCacheMaxEntries = 1024;

class GraphicsSystem;
class Shader;

struct SwapState {
  // Lock must be held when changing data in this structure.
  std::mutex mutex;
  // Dimensions of the framebuffer textures. Should match window size.
  uint32_t width = 0;
  uint32_t height = 0;
  // Current front buffer, being drawn to the screen.
  uintptr_t front_buffer_texture = 0;
  // Current back buffer, being updated by the CP.
  uintptr_t back_buffer_texture = 0;
  // Backend data
  void* backend_data = nullptr;
  // Whether the back buffer is dirty and a swap is pending.
  bool pending = false;
};

enum class SwapMode {
  kNormal,
  kIgnored,
};

enum class GammaRampType {
  kUnknown = 0,
  kTable,
  kPWL,
};

class CommandProcessor {
 protected:
  RingBuffer
      reader_;  // chrispy: instead of having ringbuffer on stack, have it near
                // the start of the class so we can access it via rel8. This
                // also reduces the number of params we need to pass
 public:
  enum class SwapPostEffect {
    kNone,
    kFxaa,
    kFxaaExtreme,
  };

  CommandProcessor(GraphicsSystem* graphics_system,
                   kernel::KernelState* kernel_state);
  virtual ~CommandProcessor();
  uint32_t counter() const { return counter_; }
  void increment_counter() { counter_++; }

  // Resets the replay_draw_first/last bisection counter; called by
  // TracePlayer at the start of each playback so draw indices are stable.
  void ResetTraceDrawIndex() { trace_draw_index_ = 0; }

  // Counts a guest swap and logs the frame rate every 5 seconds when
  // --log_fps is set; called from the shared PM4 XE_SWAP handler.
  void LogFpsTick();

  Shader* active_vertex_shader() const { return active_vertex_shader_; }
  Shader* active_pixel_shader() const { return active_pixel_shader_; }

  virtual bool Initialize();
  virtual void Shutdown();

  // Persists any host shader/pipeline cache to disk without a full shutdown, so
  // it survives the app's "cheap-skate" exit (which skips destructors). The
  // caller must quiesce this command processor first (GraphicsSystem::Pause).
  // Default no-op; backends that keep an on-disk host cache override this.
  virtual void SaveShaderStorageToDisk() {}

  void CallInThread(std::function<void()> fn);

  virtual void ClearCaches();

  // "Desired" is for the external thread managing the post-processing effect.
  SwapPostEffect GetDesiredSwapPostEffect() const {
    return swap_post_effect_desired_;
  }
  void SetDesiredSwapPostEffect(SwapPostEffect swap_post_effect);
  // Implementations must not make assumptions that the front buffer will
  // necessarily be a resolve destination - it may be a texture generated by any
  // means like written to by the CPU or loaded from a file (the disclaimer
  // screen right in the beginning of 4D530AA4 is not a resolved render target,
  // for instance).
  virtual void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                         uint32_t frontbuffer_height) {}

  // May be called not only from the command processor thread when the command
  // processor is paused, and the termination of this function may be explicitly
  // awaited.
  virtual void InitializeShaderStorage(
      const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
      std::function<void()> completion_callback = nullptr);

  virtual void RequestFrameTrace(const std::filesystem::path& root_path);
  virtual void BeginTracing(const std::filesystem::path& root_path);
  virtual void EndTracing();

  virtual void TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) = 0;

  // Makes deferred backend writes to an exact guest-physical RAM range
  // observable before a host-side consumer reads it. Backends without
  // GPU-newer-than-RAM producer records have nothing to do. This is public so
  // audited consumers outside PM4 dispatch (save-state and trace export) can
  // use the same demand boundary. Call on the command-processor thread, or
  // while the command processor is paused and its state is exclusively owned.
  virtual bool PrepareGuestMemoryForHostRead(uint32_t guest_address,
                                             uint32_t length) {
    return true;
  }

  // Makes every deferred guest-memory producer visible before the command
  // processor exposes a completion signal to the guest CPU. Unlike a demand
  // read, a completion interrupt or event write has no range identifying what
  // the CPU may read next, so backends with deferred producers must publish
  // their complete pending domain. Backends without such producers are no-op.
  virtual bool PrepareGuestMemoryForCompletionExposure() { return true; }

  void RestoreRegisters(uint32_t first_register,
                        const uint32_t* register_values,
                        uint32_t register_count, bool execute_callbacks);
  void RestoreGammaRamp(
      const reg::DC_LUT_30_COLOR* new_gamma_ramp_256_entry_table,
      const reg::DC_LUT_PWL_DATA* new_gamma_ramp_pwl_rgb,
      uint32_t new_gamma_ramp_rw_component);
  virtual void RestoreEdramSnapshot(const void* snapshot) = 0;
  // Downloads the current host EDRAM representation as the raw 10 MB
  // tile-major EDRAM image (diagnostics, e.g. per-sample MSAA inspection in
  // trace replays). Returns false if unsupported. NOT passive: forces a
  // submission + queue drain.
  virtual bool DumpEdramToBuffer(std::vector<uint8_t>& edram_out) {
    return false;
  }
  // DumpEdramToBuffer, written to a file.
  bool DumpEdramToFile(const std::filesystem::path& path);

  void InitializeRingBuffer(uint32_t ptr, uint32_t size_log2);
  void EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2);

  void UpdateWritePointer(uint32_t value);

  void LogRegisterSet(uint32_t register_index, uint32_t value);
  void LogRegisterSets(uint32_t base_register_index, const uint32_t* values,
                       uint32_t n_values);

  bool is_paused() const { return paused_; }
  void Pause();
  void Resume();

  bool Save(ByteStream* stream);
  bool Restore(ByteStream* stream);

 protected:
  struct IndexBufferInfo {
    xenos::IndexFormat format = xenos::IndexFormat::kInt16;
    xenos::Endian endianness = xenos::Endian::kNone;
    uint32_t count = 0;
    uint32_t guest_base = 0;
    size_t length = 0;
  };

  static constexpr uint32_t kReadbackBufferSizeIncrement = 16 * 1024 * 1024;

  // Eviction policy constants for readback buffer cache
  static constexpr size_t kMaxReadbackBuffers = 64;
  static constexpr uint64_t kReadbackBufferEvictionAgeFrames = 60;

  // Progressive alignment for readback buffers to avoid wasting memory
  static inline uint32_t AlignReadbackBufferSize(uint32_t size) {
    if (size < 1 * 1024 * 1024) {
      return xe::align(size, 256u * 1024u);  // 256KB for < 1MB
    } else if (size < 4 * 1024 * 1024) {
      return xe::align(size, 1u * 1024u * 1024u);  // 1MB for < 4MB
    } else {
      return xe::align(size, kReadbackBufferSizeIncrement);  // 16MB for >= 4MB
    }
  }

  // Generate a cache key for a specific resolve operation
  static inline uint64_t MakeReadbackResolveKey(uint32_t address,
                                                uint32_t length) {
    return (uint64_t(address) << 32) | uint64_t(length);
  }

  void WorkerThreadMain();
  virtual bool SetupContext() = 0;
  virtual void ShutdownContext() = 0;
  // rarely needed, most register writes have no special logic here
  XE_NOINLINE
  void HandleSpecialRegisterWrite(uint32_t index, uint32_t value);

  virtual void WriteRegister(uint32_t index, uint32_t value);

  // mem has big-endian register values
  XE_FORCEINLINE
  virtual void WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                     uint32_t num_registers);

  XE_FORCEINLINE
  virtual void WriteRegisterRangeFromRing(xe::RingBuffer* ring, uint32_t base,
                                          uint32_t num_registers);

  XE_NOINLINE
  void WriteOneRegisterFromRing(
      uint32_t base,
      uint32_t
          num_times);  // repeatedly write a value to one register, presumably a
                       // register with special handling for writes

  void WriteALURangeFromRing(xe::RingBuffer* ring, uint32_t base,
                             uint32_t num_times);

  void WriteFetchRangeFromRing(xe::RingBuffer* ring, uint32_t base,
                               uint32_t num_times);

  void WriteBoolRangeFromRing(xe::RingBuffer* ring, uint32_t base,
                              uint32_t num_times);

  void WriteLoopRangeFromRing(xe::RingBuffer* ring, uint32_t base,
                              uint32_t num_times);

  void WriteREGISTERSRangeFromRing(xe::RingBuffer* ring, uint32_t base,
                                   uint32_t num_times);

  void WriteALURangeFromMem(uint32_t start_index, uint32_t* base,
                            uint32_t num_registers);

  void WriteFetchRangeFromMem(uint32_t start_index, uint32_t* base,
                              uint32_t num_registers);

  void WriteBoolRangeFromMem(uint32_t start_index, uint32_t* base,
                             uint32_t num_registers);

  void WriteLoopRangeFromMem(uint32_t start_index, uint32_t* base,
                             uint32_t num_registers);

  void WriteREGISTERSRangeFromMem(uint32_t start_index, uint32_t* base,
                                  uint32_t num_registers);

  const reg::DC_LUT_30_COLOR* gamma_ramp_256_entry_table() const {
    return gamma_ramp_256_entry_table_;
  }
  const reg::DC_LUT_PWL_DATA* gamma_ramp_pwl_rgb() const {
    return gamma_ramp_pwl_rgb_[0];
  }
  virtual void OnGammaRamp256EntryTableValueWritten() {}
  virtual void OnGammaRampPWLValueWritten() {}

  virtual void MakeCoherent();
  virtual void PrepareForWait();
  virtual void ReturnFromWait();
  // Called once before a WAIT_REG_MEM starts polling a MEMORY location:
  // backends with deferred GPU->guest-memory writebacks must make any
  // pending write covering the polled address observable, or a guest
  // waiting on GPU-written data (fence-after-copy) deadlocks.
  virtual bool FlushPendingWritebacksForWait(uint32_t guest_address) {
    return true;
  }

  virtual void PollCompletedSubmission() {}

  // Used by strict ZPD to distinguish normal in flight latency from a
  // genuinely stuck report.
  virtual uint64_t GetCompletedSubmission() const { return 0; }

  virtual void OnPrimaryBufferEnd() {}

  // TODO(boma): Add tracking for EVENT_WRITE_EXT reports.
  using ReportHandle = uint64_t;
  static constexpr ReportHandle kInvalidReportHandle = 0;

  enum class QueryOpenResult {
    kOpened,
    kDeferred,
    kPoolExhausted,
    kFailed,
  };

  // One active guest report slot. May span multiple host query segments split
  // across submissions or render passes, final value is the normalized sum.
  struct ZPDReport {
    // Raw host count across all segments, normalized at retirement.
    uint64_t accumulated_samples = 0;
    // Submission of the first closed segment.
    uint64_t first_segment_end_submission = 0;
    // Submission containing the most recently closed segment's resolve.
    uint64_t last_segment_end_submission = 0;
    uint64_t slot_sequence_id = 0;
    uint32_t slot_base = 0;
    uint32_t begin_record = 0;
    uint32_t end_record = 0;
    // Snapshotted at BEGIN from zpd_slot_values_.
    uint32_t begin_value = 0;
    uint32_t pending_segments = 0;
    // Last known delta. Carried forward on forced close so slot doesn't
    // briefly look fully occluded. 0 is a valid delta for alternate fast path.
    uint32_t cached_delta = 0;
    bool has_cached_delta = false;
    bool ended = false;
  };

  // Currently open guest lifetime. Retired reports are tracked separately
  // by handle until their query segments resolve. This intentionally models
  // only one logical report at a time. That's enough for conventional ZPD
  // reports, but QueryBatch can have multiple slots in flight, so it doesn't
  // fit this layout. Eventually this probably wants to become something more
  // like a map of active reports keyed by slot and sequence instead.
  struct ActiveZPDSegment {
    ReportHandle report_handle = kInvalidReportHandle;
    uint32_t slot_base = 0;
    uint32_t begin_record = 0;
    uint32_t end_record = 0;
    bool segment_active = false;
    bool segment_pending_begin = false;
    bool logical_active = false;
  };

  struct PendingZPDSlot {
    ReportHandle report_handle = kInvalidReportHandle;
    uint32_t cached_delta = 0;
    bool has_cached_delta = false;
  };

  virtual void EnsureZPDQueryResources() {}
  virtual void ShutdownZPDQueryResources() {}

  virtual bool IsZPDQueryPoolReady() const { return false; }
  virtual bool CanOpenZPDQuery() const { return true; }

  // Backend acquires a pool slot, records BeginQuery, tracks it internally.
  virtual QueryOpenResult OpenZPDQuery(ReportHandle report_handle,
                                       bool can_close_submission) {
    return QueryOpenResult::kFailed;
  }
  // Backend records EndQuery, queues a resolve for the active slot.
  virtual bool CloseZPDQuery(ReportHandle report_handle,
                             uint64_t& out_submission) {
    return false;
  }
  // Backend discards the active query without resolving.
  virtual bool DiscardZPDQuery() { return false; }

  // Backend drains completed resolves and calls OnZPDQueryResolved for each.
  virtual void PumpQueryResolves() {}
  // Backend waits for all pending segments of report_handle to resolve.
  virtual bool AwaitQueryResolve(ReportHandle report_handle,
                                 uint64_t wait_for_submission) {
    return false;
  }

  bool BeginZPDReport(uint32_t report_address);
  bool EndZPDReport(uint32_t report_address, bool guest_forced_end);
  // Opens a new host query segment when CanOpenZPDQuery is true.
  void OpenQuerySegment(bool can_close_submission);
  // Closes the current segment at a submission or render pass boundary.
  // The logical report stays open and a new segment will open at the next
  // opportunity.
  void CloseQuerySegment();

  // Called by backends when a host query resolve completes.  Accumulates
  // the raw sample count, and if all segments are done, commits the report
  // to guest memory.
  void OnZPDQueryResolved(ReportHandle report_handle, uint64_t raw_samples);

  // Writes guest report with begin_value read from guest memory.
  // Orphan END path only when no controller snapshot is available.
  void WriteZPDReport(uint32_t begin_record, uint32_t end_record,
                      uint32_t begin_value, uint32_t delta_value,
                      bool write_begin_record);

  // Called from PrepareForWait so strict mode can retire before guest loops
  // again. Gives up after kStrictZPDRetireMaxStalls.
  void PumpPendingRetire();

  // Divides host count by draw resolution scale.
  uint32_t NormalizeSampleCount(uint64_t samples) const;

  // Writes the final report to guest memory and advances the slot running
  // total.  Called when a report fully resolves or is abandoned.
  void CommitZPDReport(ZPDReport& report, uint32_t delta_value);
  // Checks that the report's slot sequence is still current (not reused).
  bool IsZPDReportCurrent(const ZPDReport& report) const;
  PendingZPDSlot GetPendingZPDSlot(uint32_t slot_base,
                                   uint32_t end_record) const;

  void ResetZPDState() {
    zpd_active_segment_ = {};
    zpd_next_report_handle_ = 1;
    zpd_slot_sequences_.clear();
    zpd_slot_values_.clear();
    logical_zpd_reports_.clear();
    fast_zpd_report_cached_values_.clear();
    fake_zpd_sample_count_ = 0;
    querybatch_zpd_sample_count_ = UINT32_MAX;
    zpd_pending_retire_handle_ = kInvalidReportHandle;
    zpd_pending_retire_stalls_ = 0;
    zpd_pending_retire_start_ms_ = 0;
    zpd_force_fake_fallback_ = false;
  }

#include "pm4_command_processor_declare.h"

  virtual Shader* LoadShader(xenos::ShaderType shader_type,
                             uint32_t guest_address,
                             const uint32_t* host_address,
                             uint32_t dword_count) {
    return nullptr;
  }

  virtual bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                         IndexBufferInfo* index_buffer_info,
                         bool major_mode_explicit) {
    return false;
  }
  virtual bool IssueCopy() { return false; }

  // "Actual" is for the command processor thread, to be read by the
  // implementations.
  SwapPostEffect GetActualSwapPostEffect() const {
    return swap_post_effect_actual_;
  }

  virtual void InitializeTrace();

  Memory* memory_ = nullptr;
  kernel::KernelState* kernel_state_ = nullptr;
  GraphicsSystem* graphics_system_ = nullptr;
  RegisterFile* XE_RESTRICT register_file_ = nullptr;

  ReportHandle zpd_next_report_handle_ = 1;
  std::unordered_map<uint32_t, uint64_t> zpd_slot_sequences_;
  std::unordered_map<uint32_t, uint32_t> zpd_slot_values_;
  std::unordered_map<ReportHandle, ZPDReport> logical_zpd_reports_;
  ActiveZPDSegment zpd_active_segment_{};

  // Cached delta per END.
  // Fast mode uses this for speculative writeback and orphaned END replay.
  std::unordered_map<uint32_t, uint32_t> fast_zpd_report_cached_values_;

  uint32_t querybatch_zpd_sample_count_ = UINT32_MAX;
  bool zpd_force_fake_fallback_ = false;

  // Strict mode defers guest completion until the queued END has retired.
  ReportHandle zpd_pending_retire_handle_ = kInvalidReportHandle;
  uint32_t zpd_pending_retire_stalls_ = 0;
  // Uptime in ms when zpd_pending_retire_handle_ was first set.
  uint64_t zpd_pending_retire_start_ms_ = 0;

  // Set by the backend when resolution scale changes.
  uint32_t zpd_draw_resolution_scale_x_ = 1;
  uint32_t zpd_draw_resolution_scale_y_ = 1;

  uint32_t zpd_draw_resolution_scale_x() const {
    return zpd_draw_resolution_scale_x_;
  }
  uint32_t zpd_draw_resolution_scale_y() const {
    return zpd_draw_resolution_scale_y_;
  }

  uint32_t fake_zpd_sample_count_ = 0;

  TraceWriter trace_writer_;
  enum class TraceState {
    kDisabled,
    kStreaming,
    kSingleFrame,
  };
  TraceState trace_state_ = TraceState::kDisabled;
  std::filesystem::path trace_stream_path_;
  std::filesystem::path trace_frame_path_;

  // Actual dimensions from the most recent PM4_XE_SWAP. These describe the
  // captured guest frontbuffer more precisely than the configured display
  // mode when a title uses a different scanout size.
  uint32_t last_swap_width_ = 0;
  uint32_t last_swap_height_ = 0;

  // A trace is finalized separately from TraceWriter::Close so the portable
  // v2 bundle and adjacent provenance sidecar can be emitted after the legacy
  // v1 binary is complete.
  std::filesystem::path active_trace_path_;
  uint32_t active_trace_title_id_ = 0;
  uint32_t active_trace_frame_index_ = 0;
  uint32_t active_trace_width_ = 0;
  uint32_t active_trace_height_ = 0;
  bool active_trace_streaming_ = false;
  std::string active_trace_started_utc_;

  std::string BuildTraceFileName(uint32_t title_id, uint32_t frame_index,
                                 bool streaming) const;
  bool OpenTrace(const std::filesystem::path& path, uint32_t title_id,
                 uint32_t frame_index, bool streaming);
  void CloseTrace();
  void AbortTrace(std::string_view reason);
  std::string BuildTraceMetadataJson(uintmax_t trace_size,
                                     bool trace_hash_valid,
                                     uint64_t trace_hash) const;
  bool WriteTraceMetadata(std::string_view metadata) const;
  bool WriteTraceBundle(std::string_view metadata, uintmax_t trace_size,
                        uint64_t trace_hash) const;

  std::atomic<bool> worker_running_;
  kernel::object_ref<kernel::XHostThread> worker_thread_;

  std::queue<std::function<void()>> pending_fns_;

  // MicroEngine binary from PM4_ME_INIT
  std::vector<uint32_t> me_bin_;

  uint32_t counter_ = 0;

  // --log_fps window state (guest swaps since the window start).
  uint64_t fps_log_start_ticks_ = 0;
  uint32_t fps_log_frame_count_ = 0;

  // Per-playback draw counter for the replay_draw_first/last bisection gate
  // (reset by TracePlayer at the start of each playback).
  uint32_t trace_draw_index_ = 0;

  // Headless trace-analysis capture (--capture_dir). manifest.jsonl is
  // lazily opened on the first captured draw; capture_meta.json (run/build/
  // device provenance) is written once at the same point. Both are closed
  // when the command processor shuts down.
  FILE* capture_manifest_file_ = nullptr;
  bool capture_meta_written_ = false;
  // <capture_dir>/edram_hashes.bin (--capture_edram_hashes), lazily opened
  // on the first hashed draw.
  FILE* capture_hashes_file_ = nullptr;
  // Appends one byte-stable JSON row to <capture_dir>/manifest.jsonl,
  // opening the file and emitting capture_meta.json on first use. No-op when
  // --capture_dir is empty.
  void WriteCaptureManifestRow(const std::string& json_row);
  // Emits <capture_dir>/capture_meta.json (commit/build/device provenance
  // kept OUT of the byte-stable per-draw stream, per design D4).
  void WriteCaptureMeta(const std::filesystem::path& capture_dir);
  // Backend device/driver provenance for capture_meta.json (a JSON object, or
  // "" for none). Overridden by backends that know their device (e.g. Vulkan
  // emits device name + apiVersion + driverVersion / MoltenVK version).
  virtual std::string GetCaptureDeviceInfoJson() const { return std::string(); }
  // True if this draw index is selected for an EDRAM snapshot by
  // --capture_draw_range and/or --capture_edram_at. No-op-fast when both are
  // empty.
  bool IsCaptureEdramDraw(uint32_t draw_index) const;
  // Snapshots raw EDRAM to <capture_dir>/draw_NNNNNN.edram.bin if this draw
  // is selected, and/or appends its per-tile/sample-plane hash record when
  // --capture_edram_hashes (one EDRAM download serves both). May perturb
  // (design D3): forces a submission + queue drain.
  void MaybeCaptureEdramSnapshot(uint32_t draw_index);
  // Hazard-bisection probe: if --serialize_draw_range selects this draw,
  // calls FlushForDrawSerialization() — a submission + wait with NO capture
  // output. Same call site as MaybeCaptureEdramSnapshot (the placement the
  // capture path proved safe; an earlier in-IssueDraw hook broke the Metal
  // upload arena).
  void MaybeSerializeDraw(uint32_t draw_index);
  // Backend hook for MaybeSerializeDraw: commit all encoded GPU work and wait
  // for completion, leaving state ready for the next draw. Default no-op.
  virtual void FlushForDrawSerialization() {}
  // Appends one record (u32 draw index + kEdramTileCount*4 u64 XXH3-64
  // tile/sample-plane hashes) to <capture_dir>/edram_hashes.bin, writing the
  // 16-byte header ('XEDH', version, tile_count, tile_bytes) on first use.
  void AppendCaptureEdramHashes(uint32_t draw_index,
                                const std::vector<uint8_t>& edram);

  uint32_t primary_buffer_ptr_ = 0;
  uint32_t primary_buffer_size_ = 0;

  uint32_t read_ptr_index_ = 0;
  uint32_t read_ptr_update_freq_ = 0;
  uint32_t read_ptr_writeback_ptr_ = 0;

  std::unique_ptr<xe::threading::Event> write_ptr_index_event_;
  std::atomic<uint32_t> write_ptr_index_;

  uint64_t bin_select_ = 0xFFFFFFFFull;
  uint64_t bin_mask_ = 0xFFFFFFFFull;

  Shader* active_vertex_shader_ = nullptr;
  Shader* active_pixel_shader_ = nullptr;

  bool paused_ = false;
  // Written on the command-processor thread immediately before Pause suspends
  // it, then read by Save after the pause fence has completed.
  bool pause_guest_memory_ready_ = false;

  // By default (such as for tools), post-processing is disabled.
  // "Desired" is for the external thread managing the post-processing effect.
  SwapPostEffect swap_post_effect_desired_ = SwapPostEffect::kNone;
  SwapPostEffect swap_post_effect_actual_ = SwapPostEffect::kNone;

 private:
  reg::DC_LUT_30_COLOR gamma_ramp_256_entry_table_[256] = {};
  reg::DC_LUT_PWL_DATA gamma_ramp_pwl_rgb_[128][3] = {};
  uint32_t gamma_ramp_rw_component_ = 0;

  XE_NOINLINE XE_COLD void LogKickoffInitator(uint32_t value);
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_COMMAND_PROCESSOR_H_
