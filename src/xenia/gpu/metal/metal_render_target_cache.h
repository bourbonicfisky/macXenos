/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_
#define XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "xenia/gpu/metal/metal_d2c_classifier.h"
#include "xenia/gpu/metal/metal_depth_semantics_flags.h"
#include "xenia/gpu/metal/metal_edram_authority.h"
#include "xenia/gpu/metal/metal_host_depth_reconcile.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"
#include "xenia/gpu/render_target_cache.h"
#include "xenia/gpu/render_target_transfer_class.h"
#include "xenia/gpu/xenos.h"

namespace xe {
class Memory;
namespace gpu {
class TraceWriter;
namespace metal {

class MetalCommandProcessor;

// Metal implementation of the shared copy-based EDRAM ownership model. The
// base class owns the EDRAM 80x16-tile ownership ranges and computes the
// transfers each draw's render targets need; this subclass creates the
// MTLTextures and executes the transfers. Current scope (2026-08-01,
// implement-metal-depth-semantics closeout): canonical per-sample EDRAM +
// real 2x/4x MSAA default-on; host-depth ownership with two-source
// pointwise reconciliation PROMOTED default-on (XE_METAL_HOST_DEPTH=0 exact
// rollback); same-frame d2c classification via the key-identity classifier;
// transfer tail verified word-level with bounded residual reporting (64bpp
// classes unengaged/unclaimed, residual-report tripwire); resolve does
// snapshot registration AND guest-RAM writeback. 1x resolution scale;
// gamma uses separate linear RGBA16Unorm host storage when the exact rollout
// gate is enabled; the legacy collapsed RGBA8 path remains the rollback.
// Metal objects held as opaque void* (retained) so the header stays C++.
class MetalRenderTargetCache final : public RenderTargetCache {
 public:
  // MTLPixelFormatRGBA8Unorm — THE CANONICAL COLOR STORAGE. Storage bytes
  // equal guest EDRAM bytes for drawn 8888 content (shader .r = guest
  // byte0 = guest red), so every raw crossing (EDRAM dump, snapshot
  // restore, guest-RAM writeback) is a straight byte copy, and the
  // reinterpret passes (c2d/d2c/c2f/f2c, resolve clears) all use word =
  // r | g<<8 | b<<16 | a<<24. Present samples logically, so the BGRA8
  // drawable is unaffected. (Was BGRA8: drawn content then disagreed
  // with the raw crossings by an R<->B swap.)
  static const uint64_t kColorFormat;
  // MTLPixelFormatDepth32Float_Stencil8 — full depth + stencil planes.
  static const uint64_t kDepthFormat;

  MetalRenderTargetCache(const RegisterFile& register_file,
                         const Memory& memory, TraceWriter* trace_writer,
                         MetalCommandProcessor& command_processor,
                         void* device);
  ~MetalRenderTargetCache();

  bool Initialize();
  void Shutdown();

  Path GetPath() const override {
    return Path::kHostRenderTargets;
  }

  // Failed-publication hook (command processor): lineage recorded before a
  // failed publication may describe content that never reached its
  // destination — nothing before this point may prove a d2c return.
  void InvalidateD2cLineage() { d2c_lineage_.InvalidateAll(); }

  bool Update(bool is_rasterization_done,
              reg::RB_DEPTHCONTROL normalized_depth_control,
              uint32_t normalized_color_mask,
              const Shader& vertex_shader) override;

  // --- Metal accessors for the command processor (valid after Update) ---
  // id<MTLTexture> of the bound depth / color-0 target, or nullptr.
  void* current_depth_texture() const;
  void* current_color_texture(uint32_t index, uint64_t* pixel_format_out) const;
  // Raster-only working views. Under the staged 4.1 gate, multisampled keys
  // return their real 2x/4x companions while compatibility consumers keep
  // using current_*_texture() until resolves migrate in task 4.3.
  void* current_depth_raster_texture() const;
  void* current_color_raster_texture(uint32_t index,
                                     uint64_t* pixel_format_out) const;
  uint32_t current_raster_sample_count() const;
  // Makes the current draw's exact attachment tile spans coherent with the
  // canonical per-sample EDRAM before the render pass opens. Returns false if
  // a required scheduled conversion couldn't be encoded.
  bool PrepareCurrentRasterViews();
  // Host depth a canonical depth->depth transfer leaves in the destination
  // view when the source view held `host_depth` (pack with the truncating
  // 24-bit encode, materialize decode, Stage B lane reconcile for float24).
  // The MSAA fast-clear fold pins a folded depth clear to this value so it
  // matches the transfer path it replaces.
  float CanonicalDepthTransferResult(
      float host_depth, xenos::DepthRenderTargetFormat format) const;
  // Packs every dirty attachment interval needed by canonical diagnostics.
  // The caller must submit/wait the returned command-buffer work before
  // reading canonical storage.
  bool PrepareCanonicalEdramForReadback();
  bool PrepareCanonicalEdramForResolve(uint32_t span_base,
                                       uint32_t row_length_used,
                                       uint32_t rows, uint32_t span_pitch);
  // Records successful raster writes after a draw has been emitted. Only the
  // exact spans and slots actually used by that draw become authoritative.
  void MarkCurrentRasterTargetsWritten(uint32_t color_write_mask,
                                       bool depth_or_stencil_written);
  // A stable identity for "did the attachment set change" pass management.
  uint64_t current_attachment_key() const;
  // Human-readable composition behind current_attachment_key(). That key is a
  // one-way rolling hash, so an rt= hash recurring in GPU fault reports cannot
  // otherwise be resolved back to concrete render targets. Lives here because
  // the accumulated-target accessor is protected on the base cache.
  std::string DescribeCurrentAttachments() const;

  // The most recent BGRA8-format color target drawn into (pre-resolve present
  // fallback for splash/menus).
  void* presentable_color_target() const { return presentable_color_target_; }
  uint32_t width() const { return 1280; }
  uint32_t height() const { return 720; }

  // --- Resolve source (ownership truth) ---
  // Returns the id<MTLTexture> of the render target owning the resolve source
  // tiles (base class GetResolveCopyRectanglesToDump), or nullptr. is_depth
  // selects the plane; pixel_format_out receives the texture's MTLPixelFormat.
  void* GetResolveSourceTexture(uint32_t edram_base, uint32_t row_length_used,
                                uint32_t rows, uint32_t pitch_tiles,
                                bool is_depth, uint64_t* pixel_format_out);

  // Depth resolve into the R32Float snapshot at rows
  // [dest_y_offset, dest_y_offset+height): iterates the base class's
  // per-owner dump rectangles (a span can be owned PIECEWISE by several RTs)
  // and converts each region from its owner — depth RTs read directly, color
  // RTs (raw EDRAM bits, e.g. the restored trace snapshot) decode the guest
  // 24-bit depth (unorm24 / 20e4 float24). Returns true if anything resolved.
  bool ResolveDepthToSnapshot(const void* command_buffer, uint32_t span_base,
                              uint32_t row_length_used, uint32_t rows,
                              uint32_t span_pitch, uint32_t span_msaa_samples,
                              bool is_float24, void* dest_snapshot,
                              uint32_t width, uint32_t height,
                              uint32_t dest_y_offset);

  // Color resolve into the snapshot: per-owner, per-tile blits with FULL tile
  // remapping between the owner's layout (its own pitch — e.g. a 4x-MSAA key
  // is twice as wide) and the resolve span's linear layout. The naive
  // whole-rect blit assumed owner pixel (x,y) == resolve pixel (x,y), which
  // scrambles any owner whose pitch differs from the resolve's.
  bool ResolveColorToSnapshot(const void* command_buffer, uint32_t span_base,
                              uint32_t row_length_used, uint32_t rows,
                              uint32_t span_pitch, uint32_t span_msaa_samples,
                              void* dest_snapshot, uint64_t dest_pixel_format,
                              uint32_t width, uint32_t height,
                              uint32_t dest_y_offset, float exp_bias_scale);
  bool ResolveCanonicalToSnapshot(
      const void* command_buffer, bool is_depth, uint32_t base_tiles,
      uint32_t pitch_tiles, uint32_t msaa_samples, uint32_t format,
      bool is_64bpp, uint32_t source_x, uint32_t source_y,
      uint32_t sample_select, bool dest_swap, bool raw_copy,
      void* dest_snapshot,
      uint64_t dest_pixel_format, uint32_t width, uint32_t height,
      uint32_t dest_y_offset, float exp_bias_scale);
  struct CanonicalScatterDest;
  bool ResolveCanonicalDepthToBuffer(
      const void* command_buffer, uint32_t base_tiles, uint32_t pitch_tiles,
      uint32_t msaa_samples, uint32_t format, uint32_t source_x,
      uint32_t source_y, uint32_t sample_select, void* destination_buffer,
      uint32_t width, uint32_t height, uint32_t dest_endian = 0,
      const CanonicalScatterDest* scatter = nullptr);
  // Stage B (implement-metal-gpu-visible-resolve-authority): when supplied,
  // the packed resolve dispatch writes GUEST-TILED into `destination_buffer`
  // (which must then be the shared-memory buffer, whose offset is the guest
  // physical address) instead of linearly into a staging buffer, removing both
  // the staging allocation and the CPU tiled scatter. 2D destinations only.
  // Omitting it is byte-identical to the pre-Stage-B dispatch.
  // Canonical per-sample EDRAM buffer (id<MTLBuffer>), for dispatchers that
  // bind it as the resolve SOURCE. Non-owning; may be null before the
  // canonical path initializes.
  void* canonical_edram_buffer() const { return canonical_edram_; }

  struct CanonicalScatterDest {
    uint32_t dest_base;
    uint32_t dest_pitch_aligned;
    uint32_t dest_offset_x;
    uint32_t dest_offset_y;
    uint32_t dest_extent_bytes;
  };
  bool ResolveCanonicalColorToBuffer(
      const void* command_buffer, uint32_t base_tiles, uint32_t pitch_tiles,
      uint32_t msaa_samples, uint32_t format, bool is_64bpp,
      uint32_t source_x, uint32_t source_y, uint32_t sample_select,
      bool dest_swap, float exp_bias_scale, uint32_t dest_format,
      uint32_t dest_bpp_log2, uint32_t dest_endian, bool raw_copy,
      void* destination_buffer, uint32_t width, uint32_t height,
      const CanonicalScatterDest* scatter = nullptr);
  bool PerformCanonicalResolveClear(
      const void* command_buffer, uint32_t span_base,
      uint32_t row_length_used, uint32_t rows, uint32_t span_pitch,
      uint32_t base_tiles, uint32_t surface_pitch_tiles,
      uint32_t msaa_samples, bool is_depth, bool is_64bpp,
      uint32_t source_x, uint32_t source_y, uint32_t width, uint32_t height,
      uint32_t clear_lo, uint32_t clear_hi);

  // Resolve CLEAR: writes the guest clear value into every render target
  // owning the span's tiles (fullscreen draw, in-shader tile-range discard).
  // Guests set render-target base values this way — e.g. JC2's quarter-res
  // scene multiplies by DESTINATION (blend dst-color), so an unset base value
  // of 0 blackens the whole chain. clear_value is the raw EDRAM word
  // (color: 8888 bits; depth: (depth24 << 8)).
  void PerformResolveClear(const void* command_buffer, uint32_t span_base,
                           uint32_t row_length_used, uint32_t rows,
                           uint32_t span_pitch, bool is_depth,
                           uint32_t clear_value);

  // --- RT-as-texture snapshot bridge (kept from bring-up; stage 3 replaces
  // it with direct guest-memory resolves) ---
  void* EnsureResolveSnapshot(uint32_t dest_base, uint32_t width,
                              uint32_t height, uint64_t pixel_format);
  void* GetResolveSnapshot(uint32_t dest_base) const;
  void* FindContainingSnapshot(uint32_t dest_base, uint32_t* y_offset_out) const;

  // Frees textures retired mid-frame (a snapshot that had to grow). They cannot
  // be released at retire time because draws already encoded into the open frame
  // command buffer still reference them via bindless gpuAddresses — freeing
  // early GPU-page-faults when those draws execute. The command processor calls
  // this only after the frame command buffer has completed (commit + wait).
  void DrainDeferredReleases(uint64_t completed_submission);

  // Cache-clear release (implement-metal-cache-clearing step 5).
  //
  // Overridden because this cache keeps derived state keyed by render-target
  // IDENTITY, and the shared clear destroys every render target that does not
  // currently own EDRAM data. A recycled address would then false-hit that
  // state -- for the c2d/d2c ping-pong record that means silently changing the
  // bit-transport decision, i.e. a rendering change caused by a cache clear.
  // Vulkan's subclass destroys framebuffers and render passes before
  // delegating for exactly this reason.
  //
  // RETAINED, deliberately: canonical EDRAM and its pipelines (the promoted
  // EDRAM authority, not derived content), every render target still owning
  // EDRAM data (the shared clear's own rule), and the resolve snapshots (see
  // MetalTextureCache::ClearCache for why a non-RAM-backed snapshot is the
  // only copy of its content).
  void ClearCache() override;

  // Cache-clear step: release the COLOR resolve snapshots that the bridges
  // MetalTextureCache::ReleaseDerivedResolveBridges just classified as derived
  // (RAM-backed, non-depth) actually named, and only those.
  //
  // Each entry is (destination base, the texture that bridge pointed at). The
  // texture is VERIFIED against the snapshot currently at that base before
  // anything is released: EnsureResolveSnapshot replaces the texture at a key
  // when a later resolve needs a larger one, so a bridge can name an older
  // object than the snapshot now living there. Releasing on base alone would
  // then destroy a newer snapshot that the present path or another consumer
  // still uses. A mismatch is skipped, not released.
  //
  // Reports the bases whose texture MATCHED and was released through
  // `released_bases_out`. A caller owning per-base state must key it off that
  // list, never off the requested list: a mismatch means a NEWER generation
  // currently lives at that base, and its tip texture, RAM layout, snapshot
  // generation, and held writeback content all belong to that newer generation
  // and must survive intact.
  //
  // The classification lives with the bridge because the writeback state does;
  // this side only releases what that decision names. Snapshots at other bases
  // -- including every depth/R32F snapshot and any snapshot no bridge refers
  // to -- are retained.
  uint32_t ReleaseResolveSnapshotsForBases(
      const std::vector<std::pair<uint32_t, void*>>& bases,
      std::vector<uint32_t>* released_bases_out);
  uint32_t resolve_snapshot_count() const {
    return uint32_t(resolve_snapshots_.size());
  }

  // Replay-boundary retirement (fix-metal-trace-replay-state-reset, carrier
  // C1): immediate release of EVERY resolve snapshot texture -- color AND
  // depth/R32F, bridge-served or not -- and erasure of the map. No live
  // retention rules: at the replay restore boundary the trace is about to
  // re-deliver authoritative RAM + EDRAM, so cross-loop snapshot texels are
  // exactly the state that must die. The next loop's resolves re-create and
  // creation-clear, reproducing loop-0 (fresh-playback) semantics. Immediate
  // release is legal ONLY at the quiesced boundary the caller asserts
  // (MetalCommandProcessor::ResetDerivedReplayState). Returns the number of
  // textures released.
  uint32_t ReleaseAllResolveSnapshots();

  // Replay-boundary pool retirement (carrier C5) -- deliberately STRONGER
  // than ClearCache: the shared clear keeps every EDRAM-owning render target
  // because its content may be the only copy of guest data, but the imminent
  // RestoreEdramSnapshot re-seeds ownership and content from the
  // authoritative snapshot, so pool texels (including MSAA companions, which
  // the restore never rewrites) are pure stale lineage here. Destroys ALL
  // pool render targets (their textures + MSAA companions retire to the
  // deferred list), invalidates d2c lineage, drops the borrowed presentable
  // target, invalidates accumulated attachment state (its pointers name
  // destroyed targets), and drains the deferred-release list outright.
  // Canonical EDRAM and its pipelines are untouched (authority, not derived
  // content -- the restore rewrites the buffer itself). Call ONLY at the
  // quiesced replay-reset boundary.
  void ResetForReplayRestore();

  // Replay-determinism oracle (XE_METAL_SNAPHASH=1, default off): logs one
  // "MetalSnapHash" line per registered resolve snapshot — FNV-1a 64 over the
  // full texture contents, deterministic (key-sorted) order. Called by the
  // command processor at the quiesced swap boundary so per-replay-loop
  // snapshot content can be compared across loops (the
  // fix-metal-trace-replay-state-reset content-hash sweep; it covers ALL
  // registered snapshots, not just bases a bridge currently serves). Reads
  // through its own command buffer on `queue` (blit + wait); costs nothing
  // unless invoked.
  void LogResolveSnapshotHashes(void* queue, uint32_t swap_index);

  // Frame-dissection tooling: enumerates the render targets owning each EDRAM
  // tile range (full-EDRAM span through the base class's dump-rectangle
  // machinery) so the command processor can reconstruct the 10MB tile-major
  // EDRAM image for --capture_dir / DumpEdramToFile — making the existing
  // per-draw capture + offline-decode + diff pipeline backend-agnostic.
  struct EdramOwnerDump {
    void* texture;  // id<MTLTexture>, borrowed.
    uint32_t base_tiles;
    uint32_t pitch_tiles;
    uint32_t msaa_samples;
    uint32_t is_depth;
    uint32_t is_float24_depth;  // kD24FS8 (vs kD24S8) when is_depth.
    uint64_t pixel_format;      // MTLPixelFormat.
    uint32_t is_64bpp;
    uint32_t row_first;  // In full-EDRAM 16-tile rows.
    uint32_t rows;
  };
  void GetEdramOwnersForDump(std::vector<EdramOwnerDump>& out);

  // Restores the trace's EDRAM snapshot: uploads the tile-major 10MB contents
  // into a full-EDRAM 1280-pitch render target and gives it ownership of all
  // tiles (base-class PrepareFullEdram1280xRenderTargetForSnapshotRestoration).
  void RestoreEdramSnapshot(const void* snapshot);

  // Canonical sample-accurate diagnostics. The restore is a direct raw copy to
  // the private 10 MiB physical-layout buffer; dump reads that exact buffer via
  // shared staging and is used by capture/hashing while the gate is enabled.
  // `command_queue` is id<MTLCommandQueue> passed opaquely.
  bool sample_edram_enabled() const { return sample_edram_enabled_; }
  bool sample_edram_raster_enabled() const {
    return sample_edram_raster_enabled_;
  }
  bool DumpCanonicalEdram(void* command_queue,
                          std::vector<uint8_t>& edram_out);

 protected:
  uint32_t GetMaxRenderTargetWidth() const override { return 8192; }
  uint32_t GetMaxRenderTargetHeight() const override { return 8192; }

  // XE_GPU_TILE_TRACE correlation: ownership-claim events log the trace draw
  // index they happened at.
  uint32_t GetTraceDrawIndexForOwnershipTrace() const override;

  RenderTarget* CreateRenderTarget(RenderTargetKey key) override;

  // Host depth is D32F vs guest unorm24/float24, but tracking the separate
  // Stage B (implement-metal-depth-semantics): format-accurate host-depth
  // ownership behind XE_METAL_HOST_DEPTH (default off = the historic
  // accept-precision-loss behavior). The decision matrix is the pure
  // MetalHostDepthEncodingDifferent (canary semantics: kD24FS8 only, and
  // never while the Stage A float24 PS conversion is engaged), pinned by
  // metal_depth_semantics_matrix_test.cc.
  bool IsHostDepthEncodingDifferent(
      xenos::DepthRenderTargetFormat format) const override {
    return MetalHostDepthEncodingDifferent(host_depth_gate_enabled_, format,
                                           f24_ps_engaged_);
  }

  bool IsGammaFormatHostStorageSeparate() const override {
    return MetalGammaRenderTargetLinearEnabled();
  }

  bool host_depth_enabled() const { return host_depth_gate_enabled_; }

 private:
  // Stage B gate state, parsed once at InitializeCommon time. f24_ps_engaged_
  // mirrors the Stage A env so the predicate can fail closed when both gates
  // are set (the misconfigured-arm guard reviewers required).
  bool host_depth_gate_enabled_ = false;
  bool host_depth_d2d_enabled_ = true;
  bool f24_ps_engaged_ = false;
  // Host-precision depth lane parallel to the canonical words (10 MiB of
  // float, private) — allocated lazily only when the gate is on. The dummy
  // (16 B) satisfies the changed kernel signatures when the gate is off;
  // mode bits 0 guard every access. HostLaneMode: bit0 pack-write, bit1
  // materialize-reconcile, bit2 round-to-nearest-even.
  void* host_depth_lane_ = nullptr;   // id<MTLBuffer>, retained
  void* host_lane_dummy_ = nullptr;   // id<MTLBuffer>, retained
  uint32_t HostLaneMode(bool materialize) const;
  void* EnsureHostLaneBuffer(void* device);
  // Two-source records of the CURRENT transfer batch — the provenance-
  // complete join {destination key, selected host source, exact range}.
  // Cleared at PerformTransfers entry; populated by RecordTransition;
  // consumed by the canonical depth materialize encoders via
  // MetalSelectTwoSourceIntersections (dest-keyed, intersection-clipped).
  std::vector<MetalTwoSourceRecord> two_source_records_;
  // Encodes a lane refresh from the SELECTED host source over the exact
  // intersection, so reconciliation reads the plan's provenance rather than
  // whatever pack last touched the lane. Returns false on setup failure
  // (the caller then materializes WITHOUT reconcile — fail closed).
  bool EncodeLaneRefresh(const void* command_buffer,
                         const MetalTwoSourceRecord& record);
  class MetalRenderTarget final : public RenderTarget {
   public:
    MetalRenderTarget(RenderTargetKey key, MetalRenderTargetCache& cache,
                      void* texture, uint64_t pixel_format)
        : RenderTarget(key),
          cache_(cache),
          texture_(texture),
          pixel_format_(pixel_format) {}
    // Destruction retires the texture to the cache's deferred-release list:
    // the base class can destroy render targets mid-frame while the open
    // frame command buffer still references the texture (attachment of an
    // earlier encoder, transfer/snapshot-resolve source) — an immediate
    // release GPU-page-faults those encoders when they execute (the
    // attributed 'rtcache:resolve-color-snapshot' fault).
    ~MetalRenderTarget();
    void* texture() const { return texture_; }  // id<MTLTexture>, retained.
    uint64_t pixel_format() const { return pixel_format_; }
    // Optional real MSAA attachment (sample-accurate path, gated). When set,
    // draws render per-sample into this and resolve to texture_ (the 1x
    // companion) so the collapsed consumers stay unchanged; the samples feed
    // the canonical EDRAM. Metal requires both attachments to have identical
    // dimensions. nullptr / sample_count 1 on the default path.
    void* msaa_texture() const { return msaa_texture_; }
    uint32_t sample_count() const { return sample_count_; }
    uint32_t msaa_texture_height() const { return msaa_texture_height_; }
    void* ReplaceMsaaTexture(void* texture, uint32_t sample_count,
                             uint32_t height) {
      void* old_texture = msaa_texture_;
      msaa_texture_ = texture;
      sample_count_ = sample_count;
      msaa_texture_height_ = height;
      return old_texture;
    }

   private:
    MetalRenderTargetCache& cache_;
    void* texture_;
    uint64_t pixel_format_;
    void* msaa_texture_ = nullptr;  // id<MTLTexture>, retained when set.
    uint32_t sample_count_ = 1;
    uint32_t msaa_texture_height_ = 0;
  };

  // Clears a freshly created texture to zero (color) / the far value (depth)
  // so uninitialized private memory (magenta/noise garbage) never reaches a
  // transfer, resolve, or the present. Deterministic-black beats garbage for
  // debugging.
  void ClearTextureAtCreation(void* texture, bool is_depth);

  // Allocates real-MSAA working views at the full single-sample companion
  // dimensions required by Metal resolve attachments. Replacement first makes
  // canonical EDRAM authoritative, then retires old views by submission.
  bool EnsureCurrentMsaaCompanions();

  // Executes the base class's requested ownership transfers for the current
  // accumulated render targets: same-host-format spans as per-tile blits,
  // color->depth reinterprets via the depth-writing pass; the rest is logged
  // and skipped for now.
  void PerformTransfers(RenderTarget* const* render_targets,
                        const std::vector<Transfer>* transfer_lists);
  // Same host format but different MSAA sample density (the scene depth /
  // color tiles ping-pong between 1x and 2x/4x keys): a blit can't remap the
  // differing per-tile pixel block sizes, so this samples the source through
  // the same tile arithmetic as the reinterpret transfers (host->host: no
  // raw-word 40-column swap). 64bpp keys not supported (still log-skipped).
  void PerformCrossMsaaTransfer(const void* command_buffer,
                                RenderTarget* source_rt, RenderTarget* dest_rt,
                                const Transfer& transfer);
  // Raw-EDRAM-bits color (BGRA8, e.g. the restored snapshot) -> D32F depth:
  // in-shader tile remap + guest 24-bit depth decode (unorm24 or 20e4 float).
  void PerformColorToDepthTransfer(const void* command_buffer,
                                   RenderTarget* source_rt,
                                   RenderTarget* dest_rt,
                                   const Transfer& transfer);
  // The mirror: D32F depth encoded back to guest 24-bit raw bits as BGRA8.
  // real_bits: encode the actual depth bits (same-frame c2d->d2c ping-pong
  // bit transport — Sonic's packed vector map). Without it the shader
  // writes ZERO words (the JC2 key-local-persistence default); the env
  // XE_METAL_D2C_REAL probe also forces it globally.
  void PerformDepthToColorTransfer(const void* command_buffer,
                                   RenderTarget* source_rt,
                                   RenderTarget* dest_rt,
                                   const Transfer& transfer, bool real_bits);
  // Color<->color 32bpp reinterpret (BGRA8 raw bytes <-> f32 bits of the same
  // EDRAM word): k_8_8_8_8-keyed and k_32_FLOAT-keyed RTs aliasing the same
  // tiles (JC2's auto-exposure luminance pyramid).
  void PerformColorReinterpretTransfer(const void* command_buffer,
                                       RenderTarget* source_rt,
                                       RenderTarget* dest_rt,
                                       const Transfer& transfer);

  // --- Ownership-transition manifest (L3b conformance rung) ---------------
  // A compact, ordered record of every EDRAM ownership transition performed in
  // the frame — the observable "transition stream" that makes the murkiest
  // frame-assembly stage (PerformTransfers) auditable against the L2 model.
  // Metadata only (no GPU sync): the classified kind comes from the shared
  // ClassifyTransfer() so a manifest entry is, by construction, the branch the
  // cascade took. GPU before/after hashes are a later rung and deliberately
  // omitted here (they need a command-stream reduction, not a commit+wait).
  struct TransitionRecord {
    uint32_t transition_id;  // Monotonic within the process (never reset).
    uint32_t draw_index;     // trace_draw_index() at the transition.
    uint64_t src_key;        // RenderTargetKey.key (source owner).
    uint64_t dst_key;        // RenderTargetKey.key (reclaiming owner).
    TransferKind kind;
    uint32_t start_tiles;
    uint32_t end_tiles;
  };
  // Appends one classified transition to the manifest and, when
  // XE_METAL_TRANSITION_MANIFEST is set, emits an ordered log line. Pure
  // instrumentation — does not touch the GPU or alter transfer behavior.
  void RecordTransition(TransferKind kind, const RenderTargetKey& src_key,
                        const RenderTargetKey& dst_key,
                        const Transfer& transfer);

  // --- Stage 2 of the canonical cross-MSAA slice: additive SHADOW ----------
  // For a depth cross-MSAA transition, pack the source depth attachment into a
  // canonical buffer via the SHARED substrate (metal_edram_msl.h),
  // materialize a scratch depth attachment from it, pack that back, and compare
  // the two canonicals. A match proves the shared pack/materialize round-trips
  // real JC2 depth data on-device -- the on-M4 proof before Stage 3 activates
  // the canonical path. Gated behind XE_METAL_CROSSMSAA_SHADOW; touches NO real
  // render target, so rendering is byte-identical.
  // The shadow work is encoded on the FRAME command buffer (correct ordering
  // after the source's draws; hazard-tracked) and read back in a completion
  // handler with fresh per-transition buffers -- so NO commit/wait and NO race
  // with the frame queue writing the source. `command_buffer` is the frame cb.
  void ShadowCrossMsaaDepth(const void* command_buffer, RenderTarget* source,
                            const Transfer& transfer);
  // Per-sample-count shadow pipelines, built lazily from kEdramMsl.
  struct ShadowDepthPipelines {
    void* materialize_depth = nullptr;  // id<MTLRenderPipelineState>
    void* materialize_sten = nullptr;
    void* pack = nullptr;  // id<MTLComputePipelineState>
    void* pack_cov = nullptr;  // diagnostic legacy adapter: pack + coverage mark
    void* pack_range = nullptr;      // canonical authority: pack, tile-range guard
    void* mat_depth_range = nullptr; // canonical authority: materialize depth, ranged
    void* mat_sten_range = nullptr;  // canonical authority: materialize stencil, ranged
    void* pack_ms_range = nullptr;   // real-MSAA canonical authority, ranged
    void* mat_depth_ms_range = nullptr;
    // Stage B: host-lane provenance refresh from a SELECTED source
    // (lane_store_range for 1x sources, lane_store_ms_range for 2x/4x).
    void* lane_store = nullptr;
    void* mat_sten_ms_range = nullptr;
    // Single-pass depth + exported-stencil materialize, ranged (the 1x or
    // 2x/4x variant matching sample_count); null falls back to the bit planes.
    void* mat_depth_sten_range = nullptr;
    void* depth_state = nullptr;    // depth write + stencil REPLACE-to-0
    void* bit_states[8] = {};       // one stencil bit-plane REPLACE state each
    uint32_t sample_count = 0;
  };
  const ShadowDepthPipelines* EnsureShadowDepthPipelines(uint32_t sample_count);
  // Full-view per-sample depth/stencil conversions for real 2x/4x working
  // attachments. Task 3.5 decides when their view or canonical words are
  // authoritative; task 3.4 owns the lossless encoder contract.
  bool EncodeCanonicalDepthMsaaMaterialize(const void* command_buffer,
                                           RenderTarget* render_target,
                                           uint32_t start_tiles,
                                           uint32_t end_tiles);
  bool EncodeCanonicalDepthMsaaPack(const void* command_buffer,
                                    RenderTarget* render_target,
                                    uint32_t start_tiles,
                                    uint32_t end_tiles);
  bool EncodeCanonicalDepthMaterialize(const void* command_buffer,
                                       RenderTarget* render_target,
                                       uint32_t start_tiles,
                                       uint32_t end_tiles);
  bool EncodeCanonicalDepthPack(const void* command_buffer,
                                RenderTarget* render_target,
                                uint32_t start_tiles, uint32_t end_tiles);
  void* shadow_depth_library_ = nullptr;  // id<MTLLibrary> from kEdramMsl.
  std::unordered_map<uint32_t, ShadowDepthPipelines> shadow_pipelines_;
  uint32_t shadow_count_ = 0;  // Cap the sampled transitions per run.

  MetalCommandProcessor& command_processor_;
  void* device_;  // id<MTLDevice>, not owned.

  // --- Canonical per-sample EDRAM (openspec implement-sample-accurate-metal-edram).
  // A raw 10 MB private buffer in Xenos physical sample order is the intended
  // source of truth for every color/depth/stencil sample across 1x/2x/4x, so
  // ownership transfers/resolves can reconstruct samples the collapsed host
  // targets discard. This is the promoted default; exact
  // XE_METAL_SAMPLE_EDRAM=0 retains the collapsed rollback through the next
  // milestone.
  static constexpr uint32_t kCanonicalEdramSize = 10u * 1024 * 1024;
  bool sample_edram_enabled_ = false;
  bool sample_edram_raster_enabled_ = false;
  // Snapshot restore validation is deliberately separate from canonical
  // authority. It performs a full 10 MiB dump plus scratch materialize/pack
  // comparison and is therefore opt-in for diagnostics, not normal replay.
  bool canonical_restore_self_check_enabled_ = false;
  // Diagnostic ordering oracle: when explicitly enabled with the canonical
  // gate, commit and wait after every ownership-transfer batch. This is an
  // intentionally expensive upper bound for finding missing producer/consumer
  // synchronization as attachment pack/materialize operations are migrated.
  bool canonical_full_sync_enabled_ = false;
  uint32_t canonical_full_sync_count_ = 0;
  void* canonical_edram_ = nullptr;  // id<MTLBuffer>, private, retained.
  // Tile-granular submission-order scheduler. Task 3.5 establishes the state
  // machine; tasks 4.1+ consume its operations while migrating ownership,
  // resolve, clear, and writeback paths to canonical EDRAM.
  MetalEdramAuthorityTracker canonical_authority_;
  uint64_t canonical_authority_order_ = 0;
  // Per raster slot: the attachment/tile range last synchronized and the
  // tracker version right after it. An unchanged version proves the repeat
  // SynchronizeAttachment a no-op (see MetalEdramAuthorityTracker::version),
  // so PrepareCurrentRasterViews skips it. XE_METAL_RASTER_SYNC_SKIP=0 off.
  struct RasterSyncMemo {
    uint64_t attachment = 0;
    uint32_t start_tiles = 0;
    uint32_t end_tiles = 0;
    uint64_t version = 0;
    bool valid = false;
  };
  std::array<RasterSyncMemo, xenos::kMaxColorRenderTargets + 1>
      raster_sync_memo_{};
  // The same memo for MarkCurrentRasterTargetsWritten: when an attachment
  // re-marks the range it marked last with the tracker version unchanged, the
  // mark would only re-stamp tiles it already holds as authority and clean
  // cache, which no tracker decision can observe (every order is drawn from
  // one strictly increasing counter). XE_METAL_RASTER_MARK_SKIP=0 off.
  std::array<RasterSyncMemo, xenos::kMaxColorRenderTargets + 1>
      raster_mark_memo_{};
  static MetalEdramAuthorityTracker::AttachmentId GetCanonicalAttachmentId(
      const RenderTarget* render_target);
  RenderTarget* FindCanonicalAttachment(
      MetalEdramAuthorityTracker::AttachmentId attachment) const;
  bool EncodeCanonicalAuthorityOperation(
      const void* command_buffer,
      const MetalEdramAuthorityTracker::SyncOperation& operation);

  // --- Canonical DEPTH authority (Option 1: durable per-sample EDRAM as the
  // authority for depth ownership flips, single-sample rendering). Gated behind
  // XE_METAL_CANONICAL_DEPTH. For a depth cross-MSAA transition it REPLACES the
  // lossy legacy spatial resample with: pack(source depth+stencil) -> canonical
  // EDRAM, then materialize(dest) <- canonical EDRAM, both range-guarded to the
  // transfer's tiles, using the conformance-proven kEdramMsl helpers. The
  // canonical buffer persists across frames/RT instances, so content survives the
  // ownership churn that the transient-RT companion-resolve persistence loses --
  // that old persistence model is NOT reused here. Changes visible output (the
  // dest depth), so it is off by default and validated with --edram_determinism
  // --replay_loops=3 before trusting.
  bool canonical_depth_enabled_ = false;
  void CanonicalDepthTransfer(const void* command_buffer, RenderTarget* source,
                              RenderTarget* dest, const Transfer& transfer);
  // Materialize pipeline (canonical words -> host RGBA8Unorm attachment). Built
  // lazily; retained. See metal_kernel_conformance_test.mm for the proven kernel.
  void* canonical_materialize_library_ = nullptr;   // id<MTLLibrary>.
  void* canonical_materialize_pipeline_ = nullptr;  // Generic corpus color.
  void* canonical_pack_pipeline_ = nullptr;         // Generic corpus color.
  void* canonical_materialize64_pipeline_ = nullptr;
  void* canonical_pack64_pipeline_ = nullptr;
  void* canonical_materialize_raw_pipeline_ = nullptr;
  void* canonical_pack_raw_pipeline_ = nullptr;
  void* canonical_materialize64_raw_pipeline_ = nullptr;
  void* canonical_pack64_raw_pipeline_ = nullptr;
  // Real 2x/4x color conversion pipelines. Materialize PSOs are keyed by host
  // pixel format and sample count; the texture2d_ms pack kernel is generic.
  std::unordered_map<uint64_t, void*>
      canonical_color_msaa_materialize_pipelines_;
  std::unordered_map<uint64_t, void*> canonical_resolve_pipelines_;
  void* canonical_depth_raw_resolve_pipeline_ = nullptr;
  void* canonical_color_packed_resolve_pipeline_ = nullptr;
  void* canonical_resolve_clear_pipeline_ = nullptr;
  void* canonical_color_msaa_pack_pipeline_ = nullptr;
  void* canonical_color64_msaa_pack_pipeline_ = nullptr;
  void* canonical_color_raw_msaa_pack_pipeline_ = nullptr;
  void* canonical_color64_raw_msaa_pack_pipeline_ = nullptr;
  bool EnsureCanonicalMaterializePipeline();
  void* EnsureCanonicalColorMsaaMaterializePipeline(
      MetalRenderTarget* render_target);
  // Full-view single-sample conversion primitives. Task 3.5 supplies dirty
  // range authority and decides when to schedule them; task 3.1 owns their
  // lossless format conversion and encoder contract.
  bool EncodeCanonicalColorMaterialize(const void* command_buffer,
                                       RenderTarget* render_target,
                                       uint32_t start_tiles,
                                       uint32_t end_tiles);
  bool EncodeCanonicalColorPack(const void* command_buffer,
                                RenderTarget* render_target,
                                uint32_t start_tiles, uint32_t end_tiles);
  bool EncodeCanonicalColorMsaaMaterialize(const void* command_buffer,
                                           RenderTarget* render_target,
                                           uint32_t start_tiles,
                                           uint32_t end_tiles);
  bool EncodeCanonicalColorMsaaPack(const void* command_buffer,
                                    RenderTarget* render_target,
                                    uint32_t start_tiles,
                                    uint32_t end_tiles);
  // Gated snapshot operations. RestoreCanonicalEdram always copies the
  // already-physical-order snapshot to the private buffer when canonical EDRAM
  // is active. The separately gated diagnostic self-check then reads it back
  // through DumpCanonicalEdram, materializes it into a scratch texture, and
  // verifies byte-for-byte against the raw snapshot and legacy CPU reorder. It
  // has no effect on the restored render target.
  bool RestoreCanonicalEdram(const void* snapshot);
  void RunCanonicalRestoreSelfCheck(const void* snapshot,
                                    const void* reordered_reference,
                                    uint32_t width, uint32_t height,
                                    uint32_t row_bytes);

  void* presentable_color_target_ = nullptr;  // borrowed from a render target.

  // Reinterpret transfer pipelines (lazily built; retained).
  void* transfer_c2d_pipeline_ = nullptr;     // id<MTLRenderPipelineState>.
  void* transfer_c2d_depth_state_ = nullptr;  // id<MTLDepthStencilState>.
  // Stencil bit-plane rebuild (one masked REPLACE pass per bit; Metal has no
  // shader stencil export). States shared by the c2d and x2x paths.
  void EnsureStencilBitStates();
  void* transfer_c2d_sten_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* transfer_x2x_sten_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* stencil_bit_states_[8] = {};            // id<MTLDepthStencilState>.
  // General color->color transfer: encode source texel to the raw EDRAM word
  // (per source host format), decode to the dest format, with the x2x tile
  // remap — covers format reinterprets the specific handlers don't (7e3).
  void PerformColorConvertTransfer(const void* command_buffer,
                                   RenderTarget* source_rt,
                                   RenderTarget* dest_rt,
                                   const Transfer& transfer);
  void* transfer_conv_library_ = nullptr;  // id<MTLLibrary>.
  std::unordered_map<uint64_t, void*> transfer_conv_pipelines_;
  // Cross-MSAA same-format transfer pipelines: depth (one) and color (keyed
  // by dest MTLPixelFormat).
  void* transfer_x2x_library_ = nullptr;        // id<MTLLibrary>.
  void* transfer_x2x_depth_pipeline_ = nullptr; // id<MTLRenderPipelineState>.
  void* transfer_x2x_depth_state_ = nullptr;    // id<MTLDepthStencilState>.
  std::unordered_map<uint64_t, void*> transfer_x2x_color_pipelines_;
  void* transfer_d2c_pipeline_ = nullptr;     // id<MTLRenderPipelineState>.
  // Color reinterpret transfers (BGRA8 <-> R32Float containers of one word).
  void* transfer_reint_c2f_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* transfer_reint_f2c_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* resolve_c2f_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
  void* resolve_d2f_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
  void* clear_color_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
  void* clear_gamma_color_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* clear_depth_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
  void* clear_depth_state_ = nullptr;         // id<MTLDepthStencilState>.

  // Resolve-copy sampling pass (replaces per-tile blits: hundreds of
  // copyFromTexture calls created a driver texture view EACH — the #2 cost
  // center in gameplay profiles). Pipelines cached per dest pixel format.
  void* resolve_copy_library_ = nullptr;  // id<MTLLibrary>.
  std::unordered_map<uint64_t, void*> resolve_copy_pipelines_;

  struct Snapshot {
    void* texture;  // id<MTLTexture>, retained.
    uint32_t width;
    uint32_t height;
    uint64_t pixel_format;
  };
  // Keyed by (dest_base << 1) | is_r32f — color and depth snapshots at the
  // same guest base coexist (JC2 resolves both to one base).
  std::unordered_map<uint64_t, Snapshot> resolve_snapshots_;

  // Same-frame c2d->d2c ping-pong detection (Sonic's DoF bit-transport: a
  // color key lends tiles to a depth key, one depth pass writes a packed
  // vector map, the SAME color key takes the tiles back and the shader
  // decodes the raw bits). Keyed by the depth RenderTarget; records who lent
  // the tiles and when. The d2c transfer runs the REAL bit transport only
  // for this pattern — everywhere else it keeps the key-local-persistence
  // SKIP (JC2 casino dst-blend, SF4: real bits regress both).
  // Stage C: key-identity d2c lineage (replaces the pointer-keyed
  // C2dPingPong tracker — RT addresses recycle; keys do not). Maintained
  // centrally in RecordTransition (c2d records; other kinds supersede their
  // overlap; the d2c branch itself consumes-then-supersedes so its own
  // manifest record cannot self-destroy the lineage it classifies against).
  MetalD2cLineage d2c_lineage_;

  // Transitions are classified and, under XE_METAL_TRANSITION_MANIFEST, logged
  // in order (RecordTransition); nothing retains them. A retained manifest
  // vector grew ~1.5 KB per frame for the life of the process (2026-09-26).
  uint32_t transition_counter_ = 0;

  // Snapshot textures retired mid-frame (grown), held retained until the frame
  // GPU work finishes so in-flight draws don't dereference freed memory. Drained
  // by DrainDeferredReleases() once the submissions encoded against them
  // complete (tagged with the last submission that may reference them).
  std::vector<std::pair<uint64_t, void*>> deferred_texture_releases_;

  // Scratch for GetResolveCopyRectanglesToDump.
  std::vector<ResolveCopyDumpRectangle> resolve_dump_rectangles_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_
