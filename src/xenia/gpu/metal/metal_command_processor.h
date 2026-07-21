/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
#define XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/metal/metal_graphics_system.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/kernel_state.h"

namespace xe {
namespace gpu {
namespace metal {

class MetalPipelineCache;
class MetalPrimitiveProcessor;
class MetalRenderTargetCache;
class MetalSharedMemory;
class MetalSharedTextureCache;
class MetalTextureCache;
class MetalUploadArena;
struct MetalUploadAllocation;

// Phase 2b skeleton command processor. Overrides the same pure-virtual surface
// as the other backends; the draw/copy/shader paths are stubs for now (behaves
// like the null backend) and are implemented in later phases.
class MetalCommandProcessor : public CommandProcessor {
 protected:
  // Give this backend its own statically-bound PM4 packet chain so the execute
  // loop calls MetalCommandProcessor's Issue*/LoadShader overrides instead of
  // the base CommandProcessor stubs (whose IssueDraw returns false and logs
  // "Failed in backend" for every draw). Mirrors the vulkan/d3d12 backends.
#define OVERRIDING_BASE_CMDPROCESSOR
#include "../pm4_command_processor_declare.h"
#undef OVERRIDING_BASE_CMDPROCESSOR

 public:
  MetalCommandProcessor(MetalGraphicsSystem* graphics_system,
                        kernel::KernelState* kernel_state);
  ~MetalCommandProcessor();

  void TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) override;

  void RestoreEdramSnapshot(const void* snapshot) override;

  // Frame-dissection tooling (--capture_dir per-draw captures + --decode):
  // reconstructs the 10MB tile-major per-sample EDRAM image from the render
  // targets owning each tile range, so the existing capture/decode/diff
  // pipeline works identically for Metal and the Vulkan/AMD references.
  // Mid-frame safe: commits + waits the frame command buffer; the next draw
  // reopens it (attachments always LOAD, so content persists).
  bool DumpEdramToBuffer(std::vector<uint8_t>& edram_out) override;
  // --serialize_draw_range probe: commit + wait the frame command buffer
  // (same flush DumpEdramToBuffer performs, minus the readback).
  void FlushForDrawSerialization() override { EndFrameEncoder(); }

  // For MetalRenderTargetCache: closes any open render encoder (a transfer is
  // a pass boundary) and returns the frame id<MTLCommandBuffer> (creating it if
  // needed) so ownership-transfer blits / snapshot uploads can be encoded.
  void* PrepareTransferCommandBuffer();

  // Diagnostics: the current trace draw index (for correlating RT-cache
  // transfer logs with manifest/capture draw indices).
  uint32_t trace_draw_index() const { return trace_draw_index_; }
  // Frame counter (increments at IssueSwap) — the RT cache uses it to scope
  // same-frame c2d->d2c ping-pong detection.
  uint32_t swap_count() const { return swap_count_; }
  // The submission id the currently-open frame command buffer will get when
  // committed (commits are strictly ordered on the GPU thread). Retire lists
  // (textures, bindless slots, arena regions) are tagged with it so
  // reclamation frees exactly what completed submissions can no longer read.
  uint64_t CurrentEncodingSubmission() const { return submission_counter_ + 1; }
  // XE_METAL_S7_OFF=1 live-bisect kill switch (pre-S7 lifetime semantics).
  static bool S7LifetimeOff();
  static bool S7ArenaResetOff();  // XE_METAL_S7_OFF_ARENA
  static bool S7TagsOff();        // XE_METAL_S7_OFF_TAGS

 private:
  // Debug probes in the RT cache flush the frame CB directly.
  friend class MetalRenderTargetCache;
  bool SetupContext() override;
  void ShutdownContext() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                 uint32_t frontbuffer_height) override;

  Shader* LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                     const uint32_t* host_address,
                     uint32_t dword_count) override;

  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info,
                 bool major_mode_explicit) override;
  bool IssueCopy() override;

  void InitializeTrace() override;
  std::string GetCaptureDeviceInfoJson() const override;

  // Flush + CPU-copy pending resolve writebacks into guest RAM. Runs at
  // IssueSwap, and early via FlushPendingWritebacksForWait when the guest
  // WAIT_REG_MEMs on an address a pending writeback covers (deferring to
  // swap alone deadlocks such waits; draining everything at every wait fed
  // the auto-exposure loop mid-frame values — black flicker).
  // Drains completed writebacks/memexports to guest RAM. With a filter range,
  // only entries overlapping [filter_base, filter_base+filter_size) drain —
  // the same-frame vfetch barrier must NOT drain unrelated writebacks (the
  // exposure ping-pong reads mid-frame luminance otherwise: black flicker).
  void DrainPendingWritebacks(uint32_t filter_base = 0,
                              uint32_t filter_size = 0xFFFFFFFFu);
  void FlushPendingWritebacksForWait(uint32_t guest_address) override;

  // --- Draw path helpers (metal_command_processor.mm) ---

  // Creates the bindless heaps, the fixed system view table (shared-memory SRV),
  // the null buffer/texture/sampler, and the per-frame upload arena.
  bool InitializeDrawResources(void* device);
  void ShutdownDrawResources();

  // Returns a cached id<MTLDepthStencilState> for the given RB_DEPTHCONTROL
  // value + RB_STENCILREFMASK(_BF) masks (the reference values are encoder
  // state, not descriptor state), creating it on first use.
  void* GetOrCreateDepthStencilState(uint32_t depth_control,
                                     uint32_t stencil_ref_mask,
                                     uint32_t stencil_ref_mask_bf);

  // Opens the frame command buffer (lazily, once per frame) and a render encoder
  // into the color RT if one isn't already open. The render pass clears on the
  // first open of a frame and after each resolve (a pass boundary, see
  // reopen_clear_); otherwise it loads so multiple draws accumulate. Returns the
  // encoder (id<MTL...>) or nullptr on failure.
  void* BeginFrameEncoder();
  // Ends the current render encoder only (keeps the frame command buffer open so
  // a resolve blit / the next pass can be encoded on it).
  void EndRenderEncoder();
  // Ends any open encoder, then commits + waits on the frame command buffer.
  void EndFrameEncoder();

  // Depth-resolve bridge: runs a pass that samples the Depth32Float depth
  // texture (as depth2d) and writes its value into the R32Float snapshot
  // texture, so deferred lighting can sample the depth as texture2d_array<float>.
  // dest_y_offset places a multi-strip resolve's rows inside a full-height
  // snapshot (source rows align by frame-absolute pixel position). Builds the
  // conversion pipeline on first use. Returns false on failure.
  bool ConvertDepthToSnapshot(void* depth_texture, void* dest_snapshot,
                              uint32_t width, uint32_t height,
                              uint32_t dest_y_offset = 0);

  std::unique_ptr<MetalPipelineCache> pipeline_cache_;
  std::unique_ptr<MetalRenderTargetCache> render_target_cache_;
  std::unique_ptr<MetalSharedMemory> shared_memory_;
  std::unique_ptr<MetalTextureCache> texture_cache_;
  // C1T-R: the real shared xe::gpu::TextureCache subclass serving strict
  // tiled RGBA8 kRamContent misses (canonical for that class); null only if
  // its pipeline initialization failed, in which case the legacy cache
  // serves everything. Destroyed before texture_cache_ (its slot/retirement
  // adapter).
  std::unique_ptr<MetalSharedTextureCache> shared_texture_cache_;
  // Per-draw eligible fetch-constant mask computed by the pre-encoder
  // prepass; descriptor marshalling serves exactly these indices from the
  // shared subclass (seam audit M2 — prep and serve share one decision).
  uint32_t draw_shared_route_mask_ = 0;
  std::unique_ptr<MetalUploadArena> upload_arena_;
  // PP1: shared xe::gpu::PrimitiveProcessor subclass, the single normalization
  // authority for routed native non-expanded restart-disabled line/triangle
  // list/strip draws. Null only if its initialization failed, in which case
  // every draw stays on the inline path.
  std::unique_ptr<MetalPrimitiveProcessor> primitive_processor_;

  // Bindless / fixed-resource infrastructure for the MSC argument buffer. All
  // are id<...> held as retained void*.
  void* view_bindless_heap_ = nullptr;     // id<MTLBuffer> of IRDescriptorTableEntry
  void* sampler_bindless_heap_ = nullptr;  // id<MTLBuffer> of IRDescriptorTableEntry
  void* system_view_tables_ = nullptr;     // id<MTLBuffer> of IRDescriptorTableEntry
  void* null_buffer_ = nullptr;            // id<MTLBuffer> stand-in for inactive CBVs
  void* null_texture_ = nullptr;           // id<MTLTexture> 1x1, for unbound samples
  void* null_sampler_ = nullptr;           // id<MTLSamplerState>

  // Real guest samplers: filters/clamps/aniso packed into a key ->
  // sampler-heap slot, created on first use (states retained until shutdown).
  bool GetOrCreateSamplerSlot(uint32_t key, uint32_t* out_slot);
  std::unordered_map<uint32_t, uint32_t> sampler_slot_cache_;
  uint32_t next_sampler_slot_ = 1;     // 0 == null sampler.
  std::vector<void*> sampler_states_;  // id<MTLSamplerState>, retained.

  // Cache of id<MTLDepthStencilState> keyed by RB_DEPTHCONTROL's depth+stencil
  // bits combined with the read/write masks of RB_STENCILREFMASK(_BF).
  std::unordered_map<uint64_t, void*> depth_stencil_states_;

  // Whether the device supports mesh shaders (Apple7 / Mac2 family), a
  // prerequisite for the MSC geometry-emulation path that expands
  // rectangle/point/quad lists. Set in InitializeDrawResources.
  bool mesh_shader_supported_ = false;

  // id<MTLRenderPipelineState> for the depth->R32Float conversion pass, built on
  // first depth resolve (see ConvertDepthToSnapshot).
  void* depth_convert_pipeline_ = nullptr;

  // Frame command buffer / render encoder (id<...>). The command buffer lives
  // for the whole frame (committed at IssueSwap); the render encoder is opened
  // and closed multiple times as resolves and render-target changes split the
  // frame into passes.
  void* frame_command_buffer_ = nullptr;
  void* frame_render_encoder_ = nullptr;
  bool frame_open_ = false;  // a render encoder is currently open.

  // The attachment set (from MetalRenderTargetCache::current_attachment_key)
  // the open encoder was created with; a draw whose accumulated render targets
  // differ closes the encoder and opens a new pass.
  uint64_t encoder_attachment_key_ = 0;
  // Guest PS exports depth (oDepth) but the draw binds no depth target: Metal
  // rejects such pipelines ("shader writes to depth" with no attachment), and
  // the draw used to be NULL-PIPELINE skipped — Sonic's fullscreen bloom
  // composite (draw 1088) fell to this, leaving stale mip debris in the big
  // RT (the top-band scramble). Attach a throwaway scratch depth target
  // instead (DontCare/DontCare, default depth state = test Always, write off
  // — hardware semantics: z disabled ignores oDepth). Keyed by (w,h);
  // id<MTLTexture>, retained.
  bool dummy_depth_pass_ = false;
  // Set while encoding a memexport-only draw: BeginFrameEncoder builds an
  // attachment-less pass (renderTargetWidth/Height = 1) instead of requiring
  // bound render targets.
  bool memexport_pass_ = false;
  std::unordered_map<uint64_t, void*> dummy_depth_textures_;

  // RAM-frontbuffer present fallback (the mainline/XeniOS RequestSwapTexture
  // pattern, FMV slice): when a swap has neither an fb snapshot nor a
  // rendered target, the frontbuffer content lives only in guest RAM
  // (CPU-decoded video). Untile+upload it as the present source; content
  // hash skips the re-upload on unchanged frames.
  void* GetRamFrontbufferTexture(uint32_t base, uint32_t width,
                                 uint32_t height);
  void* fb_ram_texture_ = nullptr;  // id<MTLTexture>, shared storage, retained.
  uint64_t fb_ram_content_hash_ = 0;
  uint32_t fb_ram_width_ = 0;
  uint32_t fb_ram_height_ = 0;
  // Textures (id<MTLTexture> pointer identity) initialized once EVER: a render
  // target clears only on its first-lifetime use (uninitialized Metal memory),
  // and LOADS afterwards — in the ownership model, content comes from
  // transfers/persistence and clears come from guest RESOLVE CLEARS; a
  // per-frame clear would wipe what the ownership transfer just placed.
  std::unordered_set<uint64_t> frame_cleared_targets_;

  // Bring-up diagnostic: pair the real guest vertex function with a hardcoded
  // solid-color fragment function to isolate vertex-side (positions/constants)
  // from pixel-side (guest PS / textures). debug_solid_fragment_ is the shared
  // id<MTLFunction>; pipelines are cached by guest vertex-function pointer.
  void* debug_solid_fragment_ = nullptr;
  std::unordered_map<void*, void*> debug_solid_pipelines_;

  // Set when at least one draw rendered into the RT this frame, so IssueSwap
  // presents the RT rather than the bring-up test pattern. Reset each swap.
  bool frame_has_render_ = false;

  // Number of swaps issued; drives the bring-up test pattern color so on-device
  // it's visually obvious that frames are flowing through the guest-output path.
  uint32_t swap_count_ = 0;
  // Draws successfully encoded (across the whole run) — logged periodically so
  // on-device we can tell real geometry submission from the clear-only path.
  uint32_t draw_count_ = 0;
  uint32_t draw_skip_count_ = 0;
  // Frame command buffers committed (each = commit + waitUntilCompleted);
  // per-frame average logged by MetalFrameStats at IssueSwap.
  uint32_t frame_commit_count_ = 0;

  // S7 — submission-gated lifetime. Every frame-command-buffer commit gets a
  // monotonically increasing submission id; a completion handler publishes
  // the highest completed id (also published inline after synchronous waits,
  // which don't order against the async handler). Reclamation work enqueues
  // as (submission, action) and runs from PumpCompletedActions once its
  // submission has completed. While EndFrameEncoder still waits
  // synchronously this is behavior-neutral; it is the prerequisite for S8
  // (removing the hot-path waits) — arenas/slots/textures must not be
  // reclaimed while a submission that references them is in flight.
  uint64_t submission_counter_ = 0;
  std::atomic<uint64_t> completed_submission_{0};
  std::deque<std::pair<uint64_t, std::function<void()>>> completion_actions_;
  // Submissions whose command buffer FAULTED (GPU error status): their GPU
  // work never executed, yet the submission still publishes as completed
  // for reclamation. Recorded (GPU-thread only; bounded) as the seed
  // evidence for the command-buffer fault-attribution task, which
  // attributes the first failing submission by encoder label and resource
  // class before any lifetime fix.
  std::unordered_set<uint64_t> faulted_submissions_;
  void PublishCompletedSubmission(uint64_t submission);
  void PumpCompletedActions();

  // S8 — async submission. EndFrameEncoder commits WITHOUT waiting (unless
  // XE_METAL_SYNC_WAIT=1 restores the old synchronous behavior for A/B);
  // sites that read results on the CPU right after the commit use
  // EndFrameEncoderAndWait. Faults are recorded by the completion handler
  // and drained (logged + auto-recovered) on the GPU thread from
  // PumpCompletedActions.
  void EndFrameEncoderAndWait();
  // Blocks until `submission` completes (condition variable signaled by the
  // completion handler). Returns false on timeout — a wedged GPU.
  bool WaitForSubmission(uint64_t submission, uint32_t timeout_ms = 4000);
  // Arena allocation with ring-full handling: on failure, waits for the
  // oldest in-flight submission, reclaims, and retries once.
  bool ArenaAllocate(uint32_t size, uint32_t alignment,
                     MetalUploadAllocation* out);
  void DrainFaultedCommandBuffers();
  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  std::mutex fault_mutex_;
  std::vector<void*> faulted_command_buffers_;  // id<MTLCommandBuffer>, retained.
  std::vector<uint64_t> faulted_submission_ids_;  // Same lock as above.
  // Geometry-emulation (rect-list) diagnostics: draws dispatched via the mesh
  // path, and times the geometry pipeline failed to build (fell through to skip).
  uint32_t geo_draw_count_ = 0;
  uint32_t geo_pipeline_fail_count_ = 0;
  // How many indexed triangle-list draws we've dumped diagnostics for.
  uint32_t tri_log_count_ = 0;
  // Most recent color-resolve (IssueCopy) destination range in guest memory, for
  // the RT-as-texture bridge: a texture fetch whose base lands here is asking for
  // the scene we rendered into the color RT.
  uint32_t last_color_resolve_base_ = 0;
  uint32_t last_color_resolve_start_ = 0;
  uint32_t last_color_resolve_length_ = 0;

  // Stage-3 slice: SMALL resolves also write back into REAL guest memory at
  // frame end (staging blit + CPU memcpy). JC2's auto-exposure loop runs
  // through the CPU: the luminance pyramid resolves to RAM, game code READS
  // the 1x1 result and computes the next frame's exposure constant — the
  // snapshot bridge alone leaves that RAM unwritten (black/stale exposure).
  struct PendingWriteback {
    void* staging;  // id<MTLBuffer>, retained.
    uint32_t guest_addr;
    uint32_t width;
    uint32_t height;
    uint32_t bpp;     // destination bytes per texel (1, 2, 4, 8, or 16).
    // Scatter operation endian. Zero when a canonical packed kernel already
    // produced guest-endian bytes; otherwise the Endian128 value to apply.
    uint32_t endian;
    // Guest layout provenance retained even when staging is already endian-
    // transformed (used by presentation, logs, and sampling metadata).
    uint32_t layout_endian;
    // Tiled-dest layout (stage 3): resolve destinations are TILED guest
    // textures — the drain scatters rows through the tiled addresser (a
    // linear memcpy made every tiled fetch of the dest read block garbage).
    uint32_t dest_pitch_aligned;  // texels
    uint32_t dest_offset_x;
    uint32_t dest_offset_y;
    uint32_t dest_extent_bytes;   // write bound relative to guest_addr
    uint32_t dest_height_aligned;
    uint32_t dest_slice;
    uint32_t dest_is_3d;
    // RB_COPY_DEST_INFO.copy_dest_swap: hardware packs resolved pixels .bgra
    // (artist-texture channel order) — the scatter must too, or every fetch
    // of the dest through the game's swizzle reads R/B swapped.
    uint32_t swap_rb;
    // Guest layout provenance, independent of whether the snapshot shader has
    // already applied the swap before staging (and CPU scatter therefore must
    // not apply it again).
    uint32_t layout_swap_rb;
    // Staging already contains final guest-format, guest-endian texel bytes.
    uint32_t packed_bytes;
    // Submission whose command buffer fills `staging` — the drain may only
    // read the staging contents once this submission has completed.
    uint64_t submission;
    // Resolve provenance for the backend-neutral ResolveWB record
    // (XE_GPU_RESOLVE_WB_LOG): RB_COPY_DEST_INFO.copy_dest_format, the draw
    // index of the issuing resolve, and the swap it was enqueued on (drain
    // swap - enqueue swap = data age in frames).
    uint32_t dest_format;
    uint32_t draw_index;
    uint32_t enqueue_swap;
    // DEPTH resolve writeback: staging holds R32Float depth from the snapshot
    // conversion; the drain packs each float to the guest D24S8 word
    // (depth24 << 8, stencil 0 in v1) before the endian scatter.
    // 0 = color (no conversion), 1 = unorm24 (k_24_8), 2 = float24 (20e4).
    uint32_t depth_encode;

    // Guest-memory extent this writeback touches, for overlap / wait / drain-
    // filter tests. Tiled pitch, offsets, and row padding make the real extent
    // larger than a packed width*height*bpp span, so prefer the addresser-
    // computed dest_extent_bytes when it was carried; fall back to the packed
    // span only when it is zero (pre-tiled-layout enqueue paths).
    uint32_t range_bytes() const {
      return dest_extent_bytes ? dest_extent_bytes : width * height * bpp;
    }
  };
  std::vector<PendingWriteback> pending_writebacks_;

  // Memexport draws executed this frame whose exports are not yet coherent in
  // guest RAM. The shared-memory buffer is MTLStorageModeShared, so the drain
  // is a plain memcpy from its contents once `submission` has completed
  // (exports are linear and already guest-endian — no scatter, no swap).
  struct PendingMemexport {
    uint32_t base_address;  // guest physical
    uint32_t size_bytes;
    uint64_t submission;
  };
  std::vector<PendingMemexport> pending_memexports_;
  // Same-frame consumer visibility (v1) is handled by the stage-3 lazy flush
  // in IssueDraw: a draw whose texture fetches overlap a pending memexport
  // commits + waits + drains before its uploads read guest RAM.

  // XE_METAL_TIP_SNAP_VIEW: per-base derived 1x1 R32F copies of snapshot
  // texel (0,0) (retained id<MTLTexture>; experiment-scoped, freed with the
  // process).
  std::unordered_map<uint32_t, void*> tip_view_textures_;

  // Byte packing used by GPU resolve writebacks at each guest base. RAM
  // frontbuffer presentation reverses this packing back to logical RGBA.
  // Bases with no entry are CPU-produced scanout buffers and retain the
  // legacy ARGB scanout conversion.
  struct RamColorLayout {
    uint32_t endian;
    uint32_t swap_rb;
    // Snapshot generation this layout was recorded for (see
    // resolve_snapshot_generation_). ram_color_layouts_ entries are never
    // erased, so a base once resolved as canonical color keeps a layout entry
    // forever; presenting canonical RAM must only trust it when it still
    // matches the base's current snapshot generation. A later depth or
    // non-encodable resolve at the same base advances the generation without
    // rewriting the layout, so the stale layout is correctly rejected.
    uint32_t generation = 0;
  };
  std::unordered_map<uint32_t, RamColorLayout> ram_color_layouts_;
  // Monotonic per-base resolve generation: bumped after every successful
  // color or depth resolve, including resolves into an existing/coalesced
  // snapshot. A color writeback stamps its RamColorLayout with the value
  // bumped by the same resolve, so layout.generation == current generation
  // iff the newest resolved content at the base produced that layout.
  std::unordered_map<uint32_t, uint32_t> resolve_snapshot_generation_;

  // Gamma-parity diagnostic state (XE_METAL_GAMMA_PARITY=1).
  std::unordered_map<uint32_t, bool> gamma_parity_resolves_;
  std::unordered_set<uint32_t> gamma_parity_reported_;

  // Draws encoded into the current frame command buffer since its last
  // commit — gates the commit-before-resolve submission boundary.
  uint32_t draws_since_commit_ = 0;

  // Scratch for the per-encoder bindless residency sweep (BeginFrameEncoder).
  std::vector<void*> resident_sweep_scratch_;

  // Exposure-loop telemetry (XE_METAL_EXPO_TRACE=<tonemap ps ucode hash hex>):
  // per-swap snapshot of the auto-exposure FEEDBACK LOOP's three stages —
  // (1) the smallest luminance writeback value delivered to guest RAM this
  // frame, (2) the tonemap pixel shader's first float constants (the exposure
  // the game computed FROM (1) and fed back in), (3) the presented luma.
  // Correlating the three time series localizes an oscillation to producer,
  // game controller, or present. Written by DrainPendingWritebacks/IssueDraw,
  // emitted as one "MetalExpo" line per swap.
  float expo_last_wb_value_ = 0.0f;
  uint32_t expo_last_wb_words_ = 0;
  float expo_tonemap_c0_[4] = {};
  bool expo_tonemap_seen_ = false;

  // Per-frame autorelease pool for the GPU Commands thread (cycled at every
  // IssueSwap). The thread has no run-loop pool, so autoreleased Metal objects
  // created while encoding (command buffers, encoders, pass descriptors,
  // label strings) otherwise accumulate for the WHOLE session — a multi-hour
  // play session leaked gigabytes and stalled the machine (same class as the
  // capture path's per-owner pool fix). Opaque objc pool token.
  void* frame_autorelease_pool_ = nullptr;

  // Auto-exposure death-spiral guard: JC2's exposure feedback loop reads the
  // small luminance-pyramid writebacks from RAM. Our resolve occasionally
  // delivers an ALL-ZERO tile (failed owner lookup / stale snapshot / clear
  // timing); the game reads 0 -> exposure collapses to 0 -> the scene renders
  // black -> real luminance is 0 -> latched black forever. Real hardware never
  // hands the loop a pure-zero luminance tile, so when a small writeback would
  // write exactly-zero content we instead re-write the last non-zero content
  // seen at that guest address (hold-last-value), which lets exposure ride the
  // gap and recover. Keyed by guest_addr; stores the final guest-endian words.
  std::unordered_map<uint32_t, std::vector<uint32_t>> held_writeback_content_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
