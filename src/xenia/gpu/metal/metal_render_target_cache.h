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
  // MTLPixelFormatBGRA8Unorm — matches the drawable and the present blit.
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

  // --- RT-as-texture snapshot bridge (kept from bring-up; stage 3 replaces
  // it with direct guest-memory resolves) ---
  void* EnsureResolveSnapshot(uint32_t dest_base, uint32_t width,
                              uint32_t height, uint64_t pixel_format);
  void* GetResolveSnapshot(uint32_t dest_base) const;
  void* FindContainingSnapshot(uint32_t dest_base, uint32_t* y_offset_out) const;

  // Restores the trace's EDRAM snapshot: uploads the tile-major 10MB contents
  // into a full-EDRAM 1280-pitch render target and gives it ownership of all
  // tiles (base-class PrepareFullEdram1280xRenderTargetForSnapshotRestoration).
  void RestoreEdramSnapshot(const void* snapshot);

 protected:
  uint32_t GetMaxRenderTargetWidth() const override { return 8192; }
  uint32_t GetMaxRenderTargetHeight() const override { return 8192; }

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
    MetalRenderTarget(RenderTargetKey key, void* texture,
                      uint64_t pixel_format)
        : RenderTarget(key), texture_(texture), pixel_format_(pixel_format) {}
    ~MetalRenderTarget();
    void* texture() const { return texture_; }  // id<MTLTexture>, retained.
    uint64_t pixel_format() const { return pixel_format_; }

   private:
    void* texture_;
    uint64_t pixel_format_;
  };

  // Executes the base class's requested ownership transfers for the current
  // accumulated render targets: same-host-format spans as per-tile blits,
  // color->depth reinterprets via the depth-writing pass; the rest is logged
  // and skipped for now.
  void PerformTransfers(RenderTarget* const* render_targets,
                        const std::vector<Transfer>* transfer_lists);
  // Raw-EDRAM-bits color (BGRA8, e.g. the restored snapshot) -> D32F depth:
  // in-shader tile remap + guest 24-bit depth decode (unorm24 or 20e4 float).
  void PerformColorToDepthTransfer(const void* command_buffer,
                                   RenderTarget* source_rt,
                                   RenderTarget* dest_rt,
                                   const Transfer& transfer);
  // The mirror: D32F depth encoded back to guest 24-bit raw bits as BGRA8.
  void PerformDepthToColorTransfer(const void* command_buffer,
                                   RenderTarget* source_rt,
                                   RenderTarget* dest_rt,
                                   const Transfer& transfer);

  MetalCommandProcessor& command_processor_;
  void* device_;  // id<MTLDevice>, not owned.

  void* presentable_color_target_ = nullptr;  // borrowed from a render target.

  // Reinterpret transfer pipelines (lazily built; retained).
  void* transfer_c2d_pipeline_ = nullptr;     // id<MTLRenderPipelineState>.
  void* transfer_c2d_depth_state_ = nullptr;  // id<MTLDepthStencilState>.
  void* transfer_d2c_pipeline_ = nullptr;     // id<MTLRenderPipelineState>.

  struct Snapshot {
    void* texture;  // id<MTLTexture>, retained.
    uint32_t width;
    uint32_t height;
    uint64_t pixel_format;
  };
  std::unordered_map<uint32_t, Snapshot> resolve_snapshots_;

  // Scratch for GetResolveCopyRectanglesToDump.
  std::vector<ResolveCopyDumpRectangle> resolve_dump_rectangles_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_
