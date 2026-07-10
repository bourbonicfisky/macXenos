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

namespace xe {
namespace gpu {
namespace metal {

// Render-target cache for the Metal backend bring-up. Guest passes bind at most
// ONE color + one depth target at a time (proven for JC2 — no simultaneous MRT),
// but sequentially reinterpret a few EDRAM tile bases under different formats
// (k_8_8_8_8 / _GAMMA / k_32_FLOAT color; several depth buffers incl. the
// shadow map). Targets are therefore keyed by (EDRAM tile base, guest format):
// distinct logical targets get distinct Metal textures with a format-appropriate
// pixel format, which stops float lighting data being quantized into BGRA8 and
// stops the shadow depth pass corrupting the scene depth. No EDRAM aliasing /
// ownership transfer between keys yet (each pass fully re-renders its target).
// Metal objects are held as opaque void* (retained) so the header stays C++.
class MetalRenderTargetCache {
 public:
  // MTLPixelFormatBGRA8Unorm — matches the drawable and the present blit so a
  // straight copy into the presenter's mailbox works.
  static const uint64_t kColorFormat;
  // MTLPixelFormatDepth32Float — depth-only (no stencil yet) for 3D ordering.
  static const uint64_t kDepthFormat;

  // Fixed bring-up target extents. Wide enough for the guest scene pitch (720p
  // titles), tall enough for the 1024x1024 shadow map — and IDENTICAL for color
  // and depth: Metal requires all attachments of a render pass to share the
  // same dimensions (mismatched 720-tall color with 1024-tall depth silently
  // broke every pass with Release validation off).
  static constexpr uint32_t kColorWidth = 1280;
  static constexpr uint32_t kColorHeight = 1024;
  static constexpr uint32_t kDepthWidth = 1280;
  static constexpr uint32_t kDepthHeight = 1024;

  explicit MetalRenderTargetCache(void* device);
  ~MetalRenderTargetCache();

  // Identity of the color target (EDRAM base, guest format) resolves to: keyed
  // by the HOST pixel format, so guest formats sharing EDRAM bits (8888 vs
  // 8888_GAMMA) share one texture. Used by the command processor to detect
  // same-target passes (encoder reuse + per-frame clear tracking).
  static uint32_t ColorTargetKey(uint32_t edram_base, uint32_t guest_format);

  // Returns (creating on first use) the color target for the given EDRAM tile
  // base + guest xenos::ColorRenderTargetFormat, as id<MTLTexture> void*.
  // pixel_format_out receives the target's MTLPixelFormat. Null on failure.
  void* GetColorTarget(uint32_t edram_base, uint32_t guest_format,
                       uint64_t* pixel_format_out);
  // Returns the color target for the key only if it already exists.
  void* FindColorTarget(uint32_t edram_base, uint32_t guest_format,
                        uint64_t* pixel_format_out) const;
  // Returns (creating on first use) the Depth32Float target for the EDRAM base.
  void* GetDepthTarget(uint32_t edram_base);
  void* FindDepthTarget(uint32_t edram_base) const;

  // The most recent BGRA8 color target drawn into — the only presentable kind
  // (the present blit requires the mailbox's BGRA8). For the pre-resolve
  // fallback present path (splash/menus before any frontbuffer snapshot).
  void* presentable_color_target() const { return presentable_color_target_; }
  uint32_t width() const { return kColorWidth; }
  uint32_t height() const { return kColorHeight; }

  // RT-as-texture bridge. A resolve (IssueCopy) blits the color RT into a
  // per-destination "snapshot" texture keyed by the guest resolve destination
  // base; the texture cache then samples that snapshot when a later deferred /
  // composite pass fetches the same base (whose guest RAM we never populate).
  // The snapshot is a BGRA8 2D-array texture (matches the color RT for the blit
  // and the texture cache's ForceTextureArray reads). Persists across frames;
  // its contents are overwritten in place by each resolve. Returns nullptr on
  // allocation failure.
  // pixel_format is an MTLPixelFormat: kColorFormat for color resolves (filled by
  // a blit) or MTLPixelFormatR32Float for the depth-conversion snapshot (filled
  // by the depth->color conversion pass). The texture is created 2D-array +
  // ShaderRead|RenderTarget so it can be both a conversion-pass target and a
  // sampled texture.
  void* EnsureResolveSnapshot(uint32_t dest_base, uint32_t width,
                              uint32_t height, uint64_t pixel_format);
  // Returns the snapshot texture registered for dest_base, or nullptr.
  void* GetResolveSnapshot(uint32_t dest_base) const;
  // Multi-strip resolves: the guest resolves one buffer as several vertically
  // adjacent strips to contiguous RAM (e.g. 1280x384 @ B then 1280x336 @
  // B+0x1E0000), and shaders sample the FIRST base as the full-height texture.
  // If dest_base lands row-aligned inside an existing snapshot's guest extent,
  // returns that snapshot and its row offset so the strip writes into the
  // parent instead of a disjoint texture.
  void* FindContainingSnapshot(uint32_t dest_base, uint32_t* y_offset_out) const;

 private:
  struct Target {
    void* texture = nullptr;  // id<MTLTexture>, retained.
    uint64_t pixel_format = 0;
  };

  void* device_;  // id<MTLDevice>, not owned.

  // Color targets keyed by (edram_base << 4) | guest_format; depth by base.
  std::unordered_map<uint32_t, Target> color_targets_;
  std::unordered_map<uint32_t, Target> depth_targets_;
  void* presentable_color_target_ = nullptr;  // borrowed from color_targets_.

  struct Snapshot {
    void* texture;  // id<MTLTexture>, retained.
    uint32_t width;
    uint32_t height;
    uint64_t pixel_format;
  };
  std::unordered_map<uint32_t, Snapshot> resolve_snapshots_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_RENDER_TARGET_CACHE_H_
