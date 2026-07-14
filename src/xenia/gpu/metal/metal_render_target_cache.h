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

#include "xenia/gpu/render_target_cache.h"
#include "xenia/gpu/xenos.h"

namespace xe {
class Memory;
namespace gpu {
class TraceWriter;
namespace metal {

class MetalCommandProcessor;

// Metal implementation of the shared copy-based EDRAM ownership model
// (stage 2 of the production-shaped vertical slice). The base class owns the
// EDRAM 80x16-tile ownership ranges and computes the transfers each draw's
// render targets need; this subclass creates the MTLTextures and executes the
// transfers. Bring-up scope (see openspec reduce-apple-gpu-frametime):
// 1x resolution scale, 1x MSAA textures, gamma collapsed to 8888 storage,
// IsHostDepthEncodingDifferent()=false (no two-source host-depth modes),
// transfers executed as standalone blits for the same-format cases and
// log-skipped otherwise, resolve still through the snapshot bridge.
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
  // MTLPixelFormatDepth32Float — depth-only (no stencil yet).
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

  bool Update(bool is_rasterization_done,
              reg::RB_DEPTHCONTROL normalized_depth_control,
              uint32_t normalized_color_mask,
              const Shader& vertex_shader) override;

  // --- Metal accessors for the command processor (valid after Update) ---
  // id<MTLTexture> of the bound depth / color-0 target, or nullptr.
  void* current_depth_texture() const;
  void* current_color_texture(uint32_t index, uint64_t* pixel_format_out) const;
  // A stable identity for "did the attachment set change" pass management.
  uint64_t current_attachment_key() const;

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
                              uint32_t dest_y_offset);

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

 protected:
  uint32_t GetMaxRenderTargetWidth() const override { return 8192; }
  uint32_t GetMaxRenderTargetHeight() const override { return 8192; }

  // XE_GPU_TILE_TRACE correlation: ownership-claim events log the trace draw
  // index they happened at.
  uint32_t GetTraceDrawIndexForOwnershipTrace() const override;

  RenderTarget* CreateRenderTarget(RenderTargetKey key) override;

  // Host depth is D32F vs guest unorm24/float24, but tracking the separate
  // host-depth ownership (two-source transfers + host depth store) is out of
  // bring-up scope; accept the precision loss on aliasing.
  bool IsHostDepthEncodingDifferent(
      xenos::DepthRenderTargetFormat format) const override {
    return false;
  }

  bool IsGammaFormatHostStorageSeparate() const override { return false; }

 private:
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

   private:
    MetalRenderTargetCache& cache_;
    void* texture_;
    uint64_t pixel_format_;
  };

  // Clears a freshly created texture to zero (color) / the far value (depth)
  // so uninitialized private memory (magenta/noise garbage) never reaches a
  // transfer, resolve, or the present. Deterministic-black beats garbage for
  // debugging.
  void ClearTextureAtCreation(void* texture, bool is_depth);

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
  // Raw-EDRAM-bits color (RGBA8 bytes or RGBA16F-hosted 7e3) -> D32F depth:
  // in-shader tile remap + guest 24-bit depth decode (unorm24 or 20e4 float).
  void PerformColorToDepthTransfer(const void* command_buffer,
                                   RenderTarget* source_rt,
                                   RenderTarget* dest_rt,
                                   const Transfer& transfer);
  // The mirror: D32F depth encoded back to guest 24-bit raw bits as RGBA8
  // bytes or decoded into the RGBA16F host container used for Xenos 7e3.
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

  MetalCommandProcessor& command_processor_;
  void* device_;  // id<MTLDevice>, not owned.

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
  void* transfer_d2c_pipeline_ =
      nullptr;  // RGBA8 id<MTLRenderPipelineState>.
  void* transfer_d2c_7e3_pipeline_ =
      nullptr;  // RGBA16F id<MTLRenderPipelineState>.
  // Color reinterpret transfers (BGRA8 <-> R32Float containers of one word).
  void* transfer_reint_c2f_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* transfer_reint_f2c_pipeline_ = nullptr;  // id<MTLRenderPipelineState>.
  void* resolve_c2f_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
  void* resolve_d2f_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
  void* clear_color_pipeline_ = nullptr;      // id<MTLRenderPipelineState>.
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
  struct C2dPingPong {
    const void* src_color_rt;
    uint32_t start_tiles;
    uint32_t end_tiles;
    uint32_t swap;
  };
  std::unordered_map<const void*, C2dPingPong> c2d_pingpong_;

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
