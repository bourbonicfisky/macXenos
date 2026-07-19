/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_primitive_processor.h"

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"

namespace xe {
namespace gpu {
namespace metal {

bool MetalPrimitiveProcessor::Initialize() {
  // PP1 configuration (see the class comment):
  //   full_32bit_vertex_indices_supported = true
  //     Metal supports full 32-bit indices; keeps 32-bit guest DMA indices as
  //     kGuestDMA with host_shader_index_endian == the guest endian (no
  //     indirection / pre-masking), matching the command processor's existing
  //     CPU byte-swap-into-arena mechanism exactly.
  //   triangle_fans/line_loops/quad_lists supported = true,
  //   point_sprites/rectangle_lists native (no VS expansion) = true
  //     None of the routed classes (line/triangle list/strip) depend on these
  //     flags, and none of the non-routed classes reach this processor (the
  //     command processor's routing predicate keeps them wholly inline / on the
  //     MSC mesh-shader path). Declaring them all "supported" provisions NO
  //     conversion, built-in, or two-triangle-strip index machinery
  //     (builtin_index_buffer_size == 0), so InitializeBuiltinIndexBuffer and
  //     RequestHostConvertedIndexBufferForCurrentFrame are never invoked and no
  //     allocator is created. Later slices flip the relevant flag and add the
  //     backing buffer under their own gate.
  //   host_primitive_reset_always_enabled = false
  //     The 16->32 restart-index widening and the moltenvk_draw_diagnostics
  //     classifier are gated on this being true; keeping it false makes both
  //     statically unreachable, so the routed set is byte-neutral with the
  //     current inline path (which does no widening).
  return InitializeCommon(
      /*full_32bit_vertex_indices_supported=*/true,
      /*triangle_fans_supported=*/true,
      /*line_loops_supported=*/true,
      /*quad_lists_supported=*/true,
      /*point_sprites_supported_without_vs_expansion=*/true,
      /*rectangle_lists_supported_without_vs_expansion=*/true,
      /*host_primitive_reset_always_enabled=*/false);
}

bool MetalPrimitiveProcessor::InitializeBuiltinIndexBuffer(
    size_t size_bytes, std::function<void(void*)> fill_callback) {
  (void)size_bytes;
  (void)fill_callback;
  // Invariant guard: the PP1 configuration provisions zero built-in index
  // buffers (builtin_index_buffer_size == 0), so InitializeCommon never calls
  // this. Reaching here means the configuration changed without adding an owner
  // for the buffer. PP1 adds no allocator, so fail loudly.
  XELOGE(
      "MetalPrimitiveProcessor: InitializeBuiltinIndexBuffer called, but "
      "PP1 provisions no built-in index buffer. Configuration bug.");
  assert_always();
  return false;
}

void* MetalPrimitiveProcessor::RequestHostConvertedIndexBufferForCurrentFrame(
    xenos::IndexFormat format, uint32_t index_count, bool coalign_for_simd,
    uint32_t coalignment_original_address, size_t& backend_handle_out) {
  (void)format;
  (void)index_count;
  (void)coalign_for_simd;
  (void)coalignment_original_address;
  (void)backend_handle_out;
  // Invariant guard: PP1 routes only reset-disabled line/triangle list/strip,
  // for which Process returns kNone or kGuestDMA and never requests a
  // host-converted buffer. Reaching here means an out-of-scope draw was routed
  // (a routing-predicate bug) or a later capability slice landed without adding
  // its converted-buffer owner. Fail loudly rather than silently mis-render.
  XELOGE(
      "MetalPrimitiveProcessor: RequestHostConvertedIndexBufferForCurrentFrame "
      "called, but PP1 routes no converting class. Routing predicate bug.");
  assert_always();
  return nullptr;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
