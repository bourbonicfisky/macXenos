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

  // Opens the frame command buffer + render encoder (render pass into the color
  // RT, cleared once), if not already open. Returns the encoder (id<MTL...>) or
  // nullptr on failure.
  void* BeginFrameEncoder();
  // Ends the encoder and commits + waits on the frame command buffer.
  void EndFrameEncoder();

  std::unique_ptr<MetalPipelineCache> pipeline_cache_;
  std::unique_ptr<MetalRenderTargetCache> render_target_cache_;
  std::unique_ptr<MetalSharedMemory> shared_memory_;
  std::unique_ptr<MetalUploadArena> upload_arena_;

  // Bindless / fixed-resource infrastructure for the MSC argument buffer. All
  // are id<...> held as retained void*.
  void* view_bindless_heap_ = nullptr;     // id<MTLBuffer> of IRDescriptorTableEntry
  void* sampler_bindless_heap_ = nullptr;  // id<MTLBuffer> of IRDescriptorTableEntry
  void* system_view_tables_ = nullptr;     // id<MTLBuffer> of IRDescriptorTableEntry
  void* null_buffer_ = nullptr;            // id<MTLBuffer> stand-in for inactive CBVs
  void* null_texture_ = nullptr;           // id<MTLTexture> 1x1, for unbound samples
  void* null_sampler_ = nullptr;           // id<MTLSamplerState>

  // Frame command buffer / render encoder (id<...>), open across a frame's draws.
  void* frame_command_buffer_ = nullptr;
  void* frame_render_encoder_ = nullptr;
  bool frame_open_ = false;

  // Bring-up diagnostic: a hardcoded magenta triangle drawn into the RT each
  // frame with a trivial pipeline (no argument buffers). If it shows but guest
  // geometry doesn't, the render-pass-into-RT path works and the problem is the
  // guest shader binding / positions; if even it doesn't show, the encoder/RT
  // draw path itself is broken.
  void* debug_triangle_pipeline_ = nullptr;

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
  // How many indexed triangle-list draws we've dumped diagnostics for.
  uint32_t tri_log_count_ = 0;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
