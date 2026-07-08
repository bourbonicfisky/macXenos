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

  std::unique_ptr<MetalPipelineCache> pipeline_cache_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_COMMAND_PROCESSOR_H_
