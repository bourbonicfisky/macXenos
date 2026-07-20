/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_PRIMITIVE_PROCESSOR_H_
#define XENIA_GPU_METAL_METAL_PRIMITIVE_PROCESSOR_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "xenia/base/math.h"
#include "xenia/gpu/primitive_processor.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Byte-swaps guest indices into a host (little-endian) copy for Metal. The
// production index-materialization primitive: the command processor calls this
// to fill the upload arena for both the inline path and the routed PP1 path
// (keyed on the PrimitiveProcessor-normalized format/endian), and the device-
// free conformance test byte-compares against it. Kept here (not a test
// replica) so there is exactly one definition of the contract.
inline void SwapIndicesInto(void* dst, const void* src, uint32_t count,
                            xenos::IndexFormat format, xenos::Endian endian) {
  if (format == xenos::IndexFormat::kInt16) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    uint16_t* d = static_cast<uint16_t*>(dst);
    bool swap =
        endian == xenos::Endian::k8in16 || endian == xenos::Endian::k8in32;
    for (uint32_t i = 0; i < count; ++i) {
      d[i] = swap ? xe::byte_swap(s[i]) : s[i];
    }
  } else {
    const uint32_t* s = static_cast<const uint32_t*>(src);
    uint32_t* d = static_cast<uint32_t*>(dst);
    for (uint32_t i = 0; i < count; ++i) {
      d[i] = endian == xenos::Endian::kNone ? s[i] : xe::byte_swap(s[i]);
    }
  }
}

// PP1 routing predicate. Returns true for exactly the native, non-expanded,
// non-tessellated, restart-disabled line/triangle list/strip draws — from an
// auto-indexed or guest-DMA index source — that the shared PrimitiveProcessor
// normalizes; every other class or source stays on the inline / MSC path. The
// register reads mirror the ones PrimitiveProcessor::Process itself consumes,
// so the partition matches the processor's own view of the draw. Defined here
// (the production predicate) so the command processor and the device-free
// conformance test share one contract.
inline bool IsPp1Routable(const RegisterFile& regs,
                          xenos::PrimitiveType prim_type) {
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  // The local IssueDraw prim_type must match the register Process reads;
  // otherwise the routed emission (which maps the register-derived
  // host_primitive_type) would disagree with the class this predicate checked.
  if (prim_type != vgt_draw_initiator.prim_type) {
    return false;
  }
  // Only the two index sources the routed bind/draw handles. Immediate-indexed
  // (and any reserved) source draws are outside the partition and stay inline;
  // routing them would enter Process only to fail back.
  if (vgt_draw_initiator.source_select != xenos::SourceSelect::kDMA &&
      vgt_draw_initiator.source_select != xenos::SourceSelect::kAutoIndex) {
    return false;
  }
  switch (prim_type) {
    case xenos::PrimitiveType::kLineList:
    case xenos::PrimitiveType::kTriangleList:
      // Lists are inherently non-restart (Process filters lists out of reset),
      // so eligible whenever not tessellated.
      break;
    case xenos::PrimitiveType::kLineStrip:
    case xenos::PrimitiveType::kTriangleStrip:
      // A restart-enabled strip would trigger primitive-reset normalization on
      // the shared path (index replacement / widening) — not behavior-neutral.
      // Those stay inline; primitive reset is a later capability slice.
      if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().multi_prim_ib_ena) {
        return false;
      }
      break;
    default:
      return false;
  }
  // Tessellated draws (explicit major mode + tessellation output path) make the
  // shared processor emit a domain host vertex-shader type; the inline Metal
  // path draws them as plain primitives. Exclude to stay behavior-neutral
  // (tessellation is a later capability slice).
  if (xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, prim_type) &&
      regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select ==
          xenos::VGTOutputPath::kTessellationEnable) {
    return false;
  }
  return true;
}

// PP1 view-consistency guard. PrimitiveProcessor::Process reads the guest draw
// from the VGT registers, but IssueDraw's local inputs (index_count and the
// IndexBufferInfo) can legitimately diverge from those registers:
//   - --replay_index_window rewrites the local index base/count for trace
//     bisection WITHOUT touching the registers, so Process would act on a
//     different sub-range than the inline path;
//   - the inline path treats a DMA draw whose IndexBufferInfo->guest_base == 0
//     as non-indexed, whereas Process would bind address 0.
// Routing either case changes existing behavior. This returns true only when
// the local IssueDraw view matches the register view Process will act on, so
// those cases stay inline. It is a general view-consistency check (not keyed on
// any cvar name). The IndexBufferInfo fields are passed in so this stays a pure
// header helper shared with the device-free test.
inline bool IsPp1DrawViewConsistent(const RegisterFile& regs,
                                    uint32_t index_count, bool has_index_buffer,
                                    uint32_t ib_guest_base,
                                    xenos::IndexFormat ib_format,
                                    xenos::Endian ib_endianness,
                                    uint32_t ib_count) {
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  // The draw count Process reads (VGT num_indices) must match the local count.
  if (index_count != vgt_draw_initiator.num_indices) {
    return false;
  }
  if (vgt_draw_initiator.source_select == xenos::SourceSelect::kAutoIndex) {
    // Auto-indexed: the inline path draws non-indexed, so any bound guest index
    // buffer is an inconsistent view.
    return !has_index_buffer;
  }
  // DMA: require a present, register-matching, nonzero aligned guest index
  // buffer. (IsPp1Routable already restricted source_select to
  // kDMA/kAutoIndex.)
  if (!has_index_buffer) {
    return false;
  }
  // Process clamps guest_draw_vertex_count to VGT_DMA_SIZE.num_words when the
  // draw asks for more indices than the DMA buffer holds; the inline Metal path
  // copies/draws the full index_count. Reject a short DMA buffer so the clamp
  // cannot silently change behavior — those draws stay inline.
  if (regs.Get<reg::VGT_DMA_SIZE>().num_words <
      vgt_draw_initiator.num_indices) {
    return false;
  }
  const uint32_t index_size_log2 =
      vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16 ? 1u : 2u;
  const uint32_t register_guest_base =
      regs[XE_GPU_REG_VGT_DMA_BASE] & ~uint32_t((1u << index_size_log2) - 1u);
  // Nonzero (base 0 is the inline non-indexed fallback) and equal to the base
  // Process will compute (which also proves the local base is aligned).
  if (register_guest_base == 0 || ib_guest_base != register_guest_base) {
    return false;
  }
  if (ib_count != vgt_draw_initiator.num_indices ||
      ib_format != vgt_draw_initiator.index_size ||
      ib_endianness != regs.Get<reg::VGT_DMA_SIZE>().swap_mode) {
    return false;
  }
  return true;
}

// PP2-auto routing predicate. Returns true for exactly the auto-indexed
// triangle-fan draws that the shared PrimitiveProcessor normalizes to a
// kHostBuiltinForAuto triangle list. DMA/index-buffer fans (including
// reset-enabled fans) return false and stay on the existing inline converter;
// tessellated fans stay inline too. Auto fans carry no guest index memory and
// have no primitive reset, so no reset/DMA checks are needed. Kept pure (no
// Metal types) so the command processor and the device-free test share one
// contract, mirroring IsPp1Routable without weakening it.
inline bool IsAutoFanRoutable(const RegisterFile& regs,
                              xenos::PrimitiveType prim_type) {
  if (prim_type != xenos::PrimitiveType::kTriangleFan) {
    return false;
  }
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  // The local prim_type must match the register Process reads.
  if (prim_type != vgt_draw_initiator.prim_type) {
    return false;
  }
  // Only auto-indexed fans: DMA (and reset) fans stay on the inline converter.
  if (vgt_draw_initiator.source_select != xenos::SourceSelect::kAutoIndex) {
    return false;
  }
  // Tessellated fans make Process emit a domain host vertex-shader type; the
  // inline path draws them as plain primitives. Exclude to stay behavior-neutral.
  if (xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, prim_type) &&
      regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select ==
          xenos::VGTOutputPath::kTessellationEnable) {
    return false;
  }
  return true;
}

// PP1 + PP2-auto slice: a real shared-`PrimitiveProcessor` subclass that is the
// single guest-semantic NORMALIZATION authority for the routed native,
// non-expanded, non-tessellated, restart-disabled topologies (line list/strip,
// triangle list/strip) AND for auto-indexed triangle fans.
//
// The Metal backend keeps its existing index MECHANISM: the command processor
// CPU-swaps the guest indices into MetalUploadArena and binds that transient as
// the index source; it never binds the shared-memory buffer as an index source.
// Which draws reach this processor is decided externally by IsPp1Routable /
// IsAutoFanRoutable, not by the configuration. This subclass therefore:
//   - configures InitializeCommon with triangle_fans_supported = false (so the
//     shared code provisions the auto-index fan built-in table) and
//     convert_triangle_fans_first_provoking = true (so both the built-in table
//     and TriangleFanToList emit Metal's historical center-first fan order).
//     host_primitive_reset_always_enabled = false keeps the 16->32
//     primitive-reset widening path statically unreachable (a later capability
//     slice). No host-CONVERTED buffer is ever provisioned for the routed set,
//     because only AUTO fans are routed (DMA/reset fans stay inline).
//   - implements InitializeBuiltinIndexBuffer to own the immutable fan table on
//     the CPU (no persistent MTL resource); the command processor materializes
//     each auto-fan draw's slice into the same transient upload arena via
//     CopyBuiltinIndexData. RequestHostConvertedIndexBufferForCurrentFrame stays
//     an unreachable invariant guard (no DMA fan is routed).
//   - overrides RequestGuestIndexSharedMemoryRange as a true no-op so Process
//     performs no shared-memory RequestRange for the routed set. That avoids
//     the unwanted shared-memory-buffer upload and its intra-frame hazard flush
//     (EndFrameEncoderAndWait / commit+wait). Separately: a routed Process emits
//     no trace record of its own for the routed set (auto fans read no guest
//     index memory), and the command processor's existing explicit
//     WriteMemoryRead of any DMA index range preserves capture. Distinct facts.
// Point/rectangle/quad expansion (MSC mesh shaders), DMA/reset triangle fans,
// line loops, primitive reset, and tessellation are NOT routed here and stay
// wholly on the existing inline path.
//
// Pure C++ (no Metal types): the built-in fan table is a CPU byte vector and the
// routed draws allocate only from the command processor's arena, so this class
// holds no device or MTLBuffer, which also lets the device-free conformance test
// instantiate the real production class.
class MetalPrimitiveProcessor final : public PrimitiveProcessor {
 public:
  MetalPrimitiveProcessor(const RegisterFile& register_file, Memory& memory,
                          TraceWriter& trace_writer,
                          SharedMemory& shared_memory)
      : PrimitiveProcessor(register_file, memory, trace_writer, shared_memory) {
  }
  // No explicit destructor: ~PrimitiveProcessor already calls ShutdownCommon().

  bool Initialize();

  // Copies `byte_count` bytes of the immutable built-in index data starting at
  // `handle` (a shared byte offset, i.e. the ProcessingResult
  // host_index_buffer_handle) into `dst`. Returns false if the range is out of
  // the owned buffer's bounds (the caller must then skip the draw rather than
  // read past the table). The data never escapes this processor by reference.
  bool CopyBuiltinIndexData(size_t handle, size_t byte_count, void* dst) const;

 protected:
  // PP2-auto: provisions the auto-index triangle-fan built-in index table. The
  // shared InitializeCommon calls this once with the full table size; the
  // processor owns the bytes (a CPU vector) and runs the shared fill callback.
  // The Metal command processor copies the needed slice into its transient
  // upload arena per draw (CopyBuiltinIndexData) — no persistent MTL resource.
  bool InitializeBuiltinIndexBuffer(
      size_t size_bytes, std::function<void(void*)> fill_callback) override;

  // Unreachable in PP1/PP2-auto: the routed set (reset-disabled line/triangle
  // list/strip, and AUTO triangle fans) returns kNone / kGuestDMA /
  // kHostBuiltinForAuto and never requests a host-converted buffer. Host
  // conversion only happens for DMA/reset fans, which stay inline. Fail loudly
  // as an invariant guard.
  void* RequestHostConvertedIndexBufferForCurrentFrame(
      xenos::IndexFormat format, uint32_t index_count, bool coalign_for_simd,
      uint32_t coalignment_original_address,
      size_t& backend_handle_out) override;

  // PP1: Metal materializes routed guest indices in the command processor's
  // upload arena, so no shared-memory residency is needed here. A true no-op:
  // no RequestRange -> no UploadRanges -> the intra-frame hazard flush
  // (EndFrameEncoderAndWait / commit+wait) is unreachable from Process.
  bool RequestGuestIndexSharedMemoryRange(
      uint32_t guest_index_base, uint32_t guest_index_buffer_needed_bytes,
      ProcessedIndexBufferType index_buffer_type) override {
    (void)guest_index_base;
    (void)guest_index_buffer_needed_bytes;
    (void)index_buffer_type;
    return true;
  }

 private:
  // Immutable built-in index table (auto-index fan expansion), filled once in
  // InitializeBuiltinIndexBuffer. Lifetime is the processor's; never handed out
  // by reference.
  std::vector<uint8_t> builtin_index_data_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PRIMITIVE_PROCESSOR_H_
