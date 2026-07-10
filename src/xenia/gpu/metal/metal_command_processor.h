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

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/metal/metal_graphics_system.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/kernel_state.h"

namespace xe {
namespace gpu {
namespace metal {

class MetalPipelineCache;
class MetalRenderTargetCache;
class MetalSharedMemory;
class MetalTextureCache;
class MetalUploadArena;

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

 private:
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

  // --- Draw path helpers (metal_command_processor.mm) ---

  // Creates the bindless heaps, the fixed system view table (shared-memory SRV),
  // the null buffer/texture/sampler, and the per-frame upload arena.
  bool InitializeDrawResources(void* device);
  void ShutdownDrawResources();

  // Returns a cached id<MTLDepthStencilState> for the given RB_DEPTHCONTROL
  // value (depth-only; stencil not yet implemented), creating it on first use.
  void* GetOrCreateDepthStencilState(uint32_t depth_control);

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
  std::unique_ptr<MetalUploadArena> upload_arena_;

  // Bindless / fixed-resource infrastructure for the MSC argument buffer. All
  // are id<...> held as retained void*.
  void* view_bindless_heap_ = nullptr;     // id<MTLBuffer> of IRDescriptorTableEntry
  void* sampler_bindless_heap_ = nullptr;  // id<MTLBuffer> of IRDescriptorTableEntry
  void* system_view_tables_ = nullptr;     // id<MTLBuffer> of IRDescriptorTableEntry
  void* null_buffer_ = nullptr;            // id<MTLBuffer> stand-in for inactive CBVs
  void* null_texture_ = nullptr;           // id<MTLTexture> 1x1, for unbound samples
  void* null_sampler_ = nullptr;           // id<MTLSamplerState>

  // Cache of id<MTLDepthStencilState> keyed by the depth-relevant bits of
  // RB_DEPTHCONTROL.
  std::unordered_map<uint32_t, void*> depth_stencil_states_;

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

  // Render-target binding for the NEXT encoder open, set by IssueDraw from the
  // guest registers (RB_COLOR_INFO RT0 / RB_DEPTH_INFO) before calling
  // BeginFrameEncoder. has_color==false is a depth-only (e.g. shadow) pass.
  bool pending_has_color_ = false;
  uint32_t pending_color_base_ = 0;    // EDRAM tile base.
  uint32_t pending_color_format_ = 0;  // xenos::ColorRenderTargetFormat.
  bool pending_has_depth_ = false;
  uint32_t pending_depth_base_ = 0;  // EDRAM tile base.
  // What the currently-open encoder is bound to (valid while frame_open_); a
  // draw whose pending config differs closes the encoder and opens a new pass.
  bool encoder_has_color_ = false;
  uint32_t encoder_color_key_ = 0;  // (base << 4) | format.
  bool encoder_has_depth_ = false;
  uint32_t encoder_depth_base_ = 0;
  // Targets already cleared this frame: each (color-key / depth-base) target
  // clears on its FIRST use in a frame and loads afterwards, so multi-pass
  // accumulation into the same logical target (HUD over composite) is kept
  // while distinct targets stop bleeding into each other.
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
  // The depth EDRAM base the most recent scene-scale, depth-writing draw bound
  // (viewport >= ~400 rows). VALIDATION PROBE for the deferred-lighting chain:
  // JC2's scene-depth resolves declare EDRAM tile base 0 (the previous frame's
  // layout / a reused tile range), which our identity-by-draw-base target map
  // cannot represent — the production fix is the mainline RenderTargetCache
  // ownership-transfer model (openspec stage 2); until then, wide depth
  // resolves read this frame's actual scene depth target.
  uint32_t last_scene_depth_base_ = UINT32_MAX;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
