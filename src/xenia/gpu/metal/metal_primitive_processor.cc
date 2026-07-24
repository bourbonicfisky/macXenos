/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_primitive_processor.h"

#include <cstring>

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
  //   line_loops_supported = false
  //     PP3-auto activates the shared auto line-loop -> line-strip result. Only
  //     auto-indexed reset-disabled loops are routed by the command processor,
  //     so the DMA kHostConverted path remains unreachable.
  //   quad_lists supported = true,
  //   point_sprites/rectangle_lists native (no VS expansion) = true
  //     None of the routed classes depend on these flags, and the routing
  //     predicates keep expanded classes on the MSC path. PP2-auto's fan table
  //     is the only built-in allocation; line-loop conversion adds none.
  //   host_primitive_reset_always_enabled = false
  //     The 16->32 restart-index widening and the moltenvk_draw_diagnostics
  //     classifier are gated on this being true; keeping it false makes both
  //     statically unreachable, so the routed set is byte-neutral with the
  //     current inline path (which does no widening).
  // PP2-auto: triangle_fans_supported = false enables shared fan-to-list
  // conversion. Only auto-indexed fans are routed into Process by the command
  // processor (kHostBuiltinForAuto); DMA/reset fans stay on the inline path.
  // convert_triangle_fans_first_provoking = true makes both the built-in table
  // and TriangleFanToList emit the center-first (v0,v1,v2),(v0,v2,v3) order,
  // preserving Metal's historical inline fan order and provoking behavior.
  return InitializeCommon(
      /*full_32bit_vertex_indices_supported=*/true,
      /*triangle_fans_supported=*/false,
      /*line_loops_supported=*/false,
      /*quad_lists_supported=*/true,
      /*point_sprites_supported_without_vs_expansion=*/true,
      /*rectangle_lists_supported_without_vs_expansion=*/true,
      /*host_primitive_reset_always_enabled=*/false,
      /*convert_triangle_fans_first_provoking=*/true);
}

bool MetalPrimitiveProcessor::InitializeBuiltinIndexBuffer(
    size_t size_bytes, std::function<void(void*)> fill_callback) {
  // PP2-auto: own the built-in auto-index fan table on the CPU. The command
  // processor materializes each draw's slice into its transient upload arena,
  // so no persistent MTL resource is created here.
  if (!size_bytes) {
    return false;
  }
  builtin_index_data_.resize(size_bytes);
  fill_callback(builtin_index_data_.data());
  return true;
}

bool MetalPrimitiveProcessor::CopyBuiltinIndexData(size_t handle,
                                                   size_t byte_count,
                                                   void* dst) const {
  if (handle > builtin_index_data_.size() ||
      byte_count > builtin_index_data_.size() - handle) {
    return false;
  }
  std::memcpy(dst, builtin_index_data_.data() + handle, byte_count);
  return true;
}

void* MetalPrimitiveProcessor::RequestHostConvertedIndexBufferForCurrentFrame(
    xenos::IndexFormat format, uint32_t index_count, bool coalign_for_simd,
    uint32_t coalignment_original_address, size_t& backend_handle_out) {
  (void)format;
  (void)index_count;
  (void)coalign_for_simd;
  (void)coalignment_original_address;
  (void)backend_handle_out;
  // Invariant guard: PP1 routes only pass-through classes, PP2 routes only auto
  // fans, and PP3 routes only auto line loops. None requests a host-converted
  // buffer. Reaching here means an out-of-scope draw was routed (a predicate
  // bug) or a later capability slice landed without adding its converted-buffer
  // owner. Fail loudly rather than silently mis-render.
  XELOGE(
      "MetalPrimitiveProcessor: RequestHostConvertedIndexBufferForCurrentFrame "
      "called, but the active Metal routes own no host-converted class. "
      "Routing predicate bug.");
  assert_always();
  return nullptr;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
