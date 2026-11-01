/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) conformance for the host-converted primitive classes
// (PP5, 2026-10-26): the Metal command processor now consumes every shared
// PrimitiveProcessor result type, so the classes that normalize through a
// kHostConverted index buffer are routed instead of drawn inline or skipped:
//
//   - guest-DMA triangle fans and line loops, with and without primitive
//     reset (per-segment conversion on reset);
//   - guest-DMA quad lists on devices converting quads to triangle lists;
//   - reset-enabled strips with a custom 16-bit reset index (replaced with
//     0xFFFF, or widened to 32-bit when 0xFFFF is also a real vertex) and
//     with 32-bit reset indices (pre-masked to 24 bits, 0xFFFFFFFF sentinel);
//   - reset-disabled 16-bit buffers using 0xFFFF as a real index (widened,
//     see the PP1 Gate 4 case).
//
// Three contracts, all against production code:
//   A. The routing partition (metal::IsConvertedClassRoutable).
//   B. The REAL metal::MetalPrimitiveProcessor over crafted guest index memory:
//      result fields, the per-frame converted pool (handle, bounds, EndFrame),
//      and the converted bytes after metal::SwapConvertedIndicesInto against
//      analytic oracles computed here from the guest indices.
//   C. SwapConvertedIndicesInto itself against xenos::GpuSwap per element.
//
// Known parity note (shared code, not Metal's): reset-separated line loops
// convert to ONE host line strip with no restart between the closed loops
// (GetMultiPrimitiveHostIndexCountAndRanges concatenates ranges and the
// converted result reports host_primitive_reset_enabled == false), so a
// joining segment is drawn between consecutive loops on every backend. The
// oracle below pins the shared behaviour as it is.

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_primitive_processor.h"
#include "xenia/gpu/metal/metal_sampler_contract.h"
#include "xenia/gpu/primitive_processor.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace {

class NullSharedMemory final : public SharedMemory {
 public:
  explicit NullSharedMemory(Memory& memory) : SharedMemory(memory) {}
  bool Init() { return InitializeCommon(); }
  bool UploadRanges(const std::pair<uint32_t, uint32_t>*, uint32_t) override {
    return true;
  }
};

void SetDraw(RegisterFile& regs, xenos::PrimitiveType prim_type,
             xenos::SourceSelect source_select, xenos::IndexFormat index_format,
             xenos::Endian endian, uint32_t count, uint32_t base,
             bool multi_prim_ib, bool tess, uint32_t reset_index = 0) {
  std::memset(regs.values, 0, sizeof(regs.values));
  reg::VGT_DRAW_INITIATOR di;
  di.value = 0;
  di.prim_type = prim_type;
  di.source_select = source_select;
  di.index_size = index_format;
  di.num_indices = count;
  di.major_mode =
      tess ? xenos::MajorMode::kExplicit : xenos::MajorMode::kImplicit;
  regs.values[reg::VGT_DRAW_INITIATOR::register_index] = di.value;
  reg::VGT_DMA_SIZE ds;
  ds.value = 0;
  ds.num_words = count;
  ds.swap_mode = endian;
  regs.values[reg::VGT_DMA_SIZE::register_index] = ds.value;
  regs.values[XE_GPU_REG_VGT_DMA_BASE] = base;
  reg::PA_SU_SC_MODE_CNTL pa;
  pa.value = 0;
  pa.multi_prim_ib_ena = multi_prim_ib ? 1 : 0;
  regs.values[reg::PA_SU_SC_MODE_CNTL::register_index] = pa.value;
  reg::VGT_MULTI_PRIM_IB_RESET_INDX reset;
  reset.value = 0;
  reset.reset_indx = reset_index;
  regs.values[reg::VGT_MULTI_PRIM_IB_RESET_INDX::register_index] = reset.value;
  reg::VGT_OUTPUT_PATH_CNTL op;
  op.value = 0;
  op.path_select = tess ? xenos::VGTOutputPath::kTessellationEnable
                        : xenos::VGTOutputPath::kVertexReuse;
  regs.values[reg::VGT_OUTPUT_PATH_CNTL::register_index] = op.value;
}

struct Harness {
  Memory* memory = nullptr;
  RegisterFile regs;
  TraceWriter trace{nullptr};
  std::unique_ptr<NullSharedMemory> sm;
  std::unique_ptr<metal::MetalPrimitiveProcessor> pp;
  bool ok = false;
  explicit Harness(bool convert_quads = true) {
    std::memset(regs.values, 0, sizeof(regs.values));
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    sm = std::make_unique<NullSharedMemory>(*memory);
    if (!sm->Init()) {
      return;
    }
    pp = std::make_unique<metal::MetalPrimitiveProcessor>(regs, *memory, trace,
                                                          *sm);
    if (!pp->Initialize(convert_quads)) {
      return;
    }
    ok = true;
  }
  ~Harness() {
    if (pp) {
      pp->EndFrame();
    }
  }
  // Writes guest-endian index words to physical memory.
  void Write16(uint32_t base, const std::vector<uint16_t>& host_values,
               xenos::Endian endian) {
    uint16_t* dst = memory->TranslatePhysical<uint16_t*>(base);
    REQUIRE(dst != nullptr);
    for (size_t i = 0; i < host_values.size(); ++i) {
      dst[i] = xenos::GpuSwap(host_values[i], endian);
    }
  }
  void Write32(uint32_t base, const std::vector<uint32_t>& host_values,
               xenos::Endian endian) {
    uint32_t* dst = memory->TranslatePhysical<uint32_t*>(base);
    REQUIRE(dst != nullptr);
    for (size_t i = 0; i < host_values.size(); ++i) {
      dst[i] = xenos::GpuSwap(host_values[i], endian);
    }
  }
  // The bytes the command processor binds: the converted pool slice swapped
  // by SwapConvertedIndicesInto with the result's format/endian.
  template <typename T>
  std::vector<T> HostIndices(const PrimitiveProcessor::ProcessingResult& r) {
    std::vector<T> converted(r.host_draw_vertex_count);
    REQUIRE(pp->CopyConvertedIndexData(r.host_index_buffer_handle,
                                       converted.size() * sizeof(T),
                                       converted.data()));
    std::vector<T> host(r.host_draw_vertex_count);
    metal::SwapConvertedIndicesInto(host.data(), converted.data(),
                                    r.host_draw_vertex_count,
                                    r.host_index_format,
                                    r.host_shader_index_endian);
    return host;
  }
};

// Analytic oracles over host-order guest indices.
template <typename T>
std::vector<T> FanToListOracle(const std::vector<T>& fan) {
  std::vector<T> out;
  for (size_t i = 2; i < fan.size(); ++i) {
    out.push_back(fan[0]);
    out.push_back(fan[i - 1]);
    out.push_back(fan[i]);
  }
  return out;
}
template <typename T>
std::vector<T> LoopToStripOracle(const std::vector<T>& loop) {
  std::vector<T> out;
  if (loop.size() < 2) {
    return out;
  }
  out = loop;
  out.push_back(loop[0]);
  return out;
}
template <typename T>
std::vector<T> QuadsToListOracle(const std::vector<T>& quads) {
  std::vector<T> out;
  for (size_t q = 0; q + 3 < quads.size(); q += 4) {
    out.insert(out.end(), {quads[q], quads[q + 1], quads[q + 2], quads[q],
                           quads[q + 2], quads[q + 3]});
  }
  return out;
}
// Splits at the reset value and converts each segment (shared behaviour:
// segments concatenate with no restart between them).
template <typename T, typename Convert>
std::vector<T> SplitOracle(const std::vector<T>& guest, T reset,
                           Convert convert) {
  std::vector<T> out;
  std::vector<T> segment;
  auto flush = [&] {
    std::vector<T> c = convert(segment);
    out.insert(out.end(), c.begin(), c.end());
    segment.clear();
  };
  for (T v : guest) {
    if (v == reset) {
      flush();
    } else {
      segment.push_back(v);
    }
  }
  flush();
  return out;
}

const uint32_t kBase = 0x5000;

// ---- Part A: routing partition ----------------------------------------------

TEST_CASE("PP5 routing: DMA fans, loops and quads route (any reset)",
          "[pp5][routing]") {
  RegisterFile regs;
  for (bool reset : {false, true}) {
    SetDraw(regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt16, xenos::Endian::kNone, 9, kBase, reset,
            false, 0xFFFF);
    CHECK(metal::IsConvertedClassRoutable(regs,
                                          xenos::PrimitiveType::kTriangleFan,
                                          false));
    SetDraw(regs, xenos::PrimitiveType::kLineLoop, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt32, xenos::Endian::k8in32, 9, kBase, reset,
            false, 0xFFFFFF);
    CHECK(metal::IsConvertedClassRoutable(regs, xenos::PrimitiveType::kLineLoop,
                                          false));
    SetDraw(regs, xenos::PrimitiveType::kQuadList, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt16, xenos::Endian::kNone, 8, kBase, reset,
            false, 0xFFFF);
    CHECK(metal::IsConvertedClassRoutable(regs, xenos::PrimitiveType::kQuadList,
                                          /*convert_quad_lists=*/true));
    // Quads only route on converting devices (else MSC geometry emulation
    // owns the guest quad list and Process would keep kQuadList).
    CHECK_FALSE(metal::IsConvertedClassRoutable(
        regs, xenos::PrimitiveType::kQuadList, /*convert_quad_lists=*/false));
  }
}

TEST_CASE("PP5 routing: reset-enabled strips route unless PP4 owns them",
          "[pp5][routing]") {
  RegisterFile regs;
  for (xenos::PrimitiveType strip : {xenos::PrimitiveType::kLineStrip,
                                     xenos::PrimitiveType::kTriangleStrip}) {
    // Custom 16-bit reset index.
    SetDraw(regs, strip, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::kNone, 9, kBase, /*multi_prim_ib=*/true, false,
            0x1234);
    CHECK(metal::IsConvertedClassRoutable(regs, strip, false));
    CHECK_FALSE(metal::IsFixedResetStripRoutable(regs, strip));
    // Fixed 0xFFFF 16-bit reset: PP4's, not routed here.
    SetDraw(regs, strip, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::kNone, 9, kBase, /*multi_prim_ib=*/true, false,
            0xFFFF);
    CHECK_FALSE(metal::IsConvertedClassRoutable(regs, strip, false));
    CHECK(metal::IsFixedResetStripRoutable(regs, strip));
    // 32-bit reset.
    SetDraw(regs, strip, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt32,
            xenos::Endian::k8in32, 9, kBase, /*multi_prim_ib=*/true, false,
            0xFFFFFF);
    CHECK(metal::IsConvertedClassRoutable(regs, strip, false));
    // Reset disabled: PP1's.
    SetDraw(regs, strip, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::kNone, 9, kBase, /*multi_prim_ib=*/false, false);
    CHECK_FALSE(metal::IsConvertedClassRoutable(regs, strip, false));
    CHECK(metal::IsPp1Routable(regs, strip));
  }
}

TEST_CASE("PP5 routing: auto source, tessellation, lists and mismatches reject",
          "[pp5][routing]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, xenos::IndexFormat::kInt16,
          xenos::Endian::kNone, 9, 0, false, false);
  CHECK_FALSE(metal::IsConvertedClassRoutable(
      regs, xenos::PrimitiveType::kTriangleFan, true));
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::kNone, 9, kBase, false,
          /*tess=*/true);
  CHECK_FALSE(metal::IsConvertedClassRoutable(
      regs, xenos::PrimitiveType::kTriangleFan, true));
  for (xenos::PrimitiveType pt :
       {xenos::PrimitiveType::kTriangleList, xenos::PrimitiveType::kLineList,
        xenos::PrimitiveType::kPointList, xenos::PrimitiveType::kRectangleList,
        xenos::PrimitiveType::kQuadStrip, xenos::PrimitiveType::kTrianglePatch}) {
    SetDraw(regs, pt, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::kNone, 9, kBase, /*multi_prim_ib=*/true, false,
            0x1234);
    INFO("prim " << uint32_t(pt));
    CHECK_FALSE(metal::IsConvertedClassRoutable(regs, pt, true));
  }
  // Register says fan, local says list.
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::kNone, 9, kBase, false,
          false);
  CHECK_FALSE(metal::IsConvertedClassRoutable(
      regs, xenos::PrimitiveType::kTriangleList, true));
  // The earlier predicates are unweakened.
  CHECK_FALSE(metal::IsPp1Routable(regs, xenos::PrimitiveType::kTriangleFan));
  CHECK_FALSE(
      metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
}

// ---- Part B: Process + converted pool ---------------------------------------

TEST_CASE("PP5 normalize: DMA triangle fan -> host-converted list, 16-bit "
          "both endians, with and without reset",
          "[pp5][normalize]") {
  Harness h;
  REQUIRE(h.ok);
  for (xenos::Endian endian : {xenos::Endian::kNone, xenos::Endian::k8in16}) {
    // No reset: one fan of 6.
    std::vector<uint16_t> fan = {10, 20, 30, 40, 50, 60};
    h.Write16(kBase, fan, endian);
    SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
            xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16, endian,
            uint32_t(fan.size()), kBase, false, false);
    REQUIRE(metal::IsConvertedClassRoutable(
        h.regs, xenos::PrimitiveType::kTriangleFan, false));
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("endian " << uint32_t(endian));
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(r.host_primitive_type == xenos::PrimitiveType::kTriangleList);
    CHECK(r.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex);
    CHECK(r.host_index_format == xenos::IndexFormat::kInt16);
    CHECK(r.host_primitive_reset_enabled == false);
    CHECK(r.host_draw_vertex_count ==
          PrimitiveProcessor::GetTriangleFanListIndexCount(
              uint32_t(fan.size())));
    CHECK(h.HostIndices<uint16_t>(r) == FanToListOracle(fan));
    h.pp->EndFrame();

    // Reset 0x0FFF splits two fans; a 0xFFFF real index would widen, so the
    // custom reset keeps this a 16-bit stream.
    const uint16_t reset = 0x0FFF;
    std::vector<uint16_t> fans = {1, 2, 3, 4, reset, 5, 6, 7, reset, 8, 9};
    h.Write16(kBase, fans, endian);
    SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
            xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16, endian,
            uint32_t(fans.size()), kBase, /*multi_prim_ib=*/true, false,
            reset);
    PrimitiveProcessor::ProcessingResult rr{};
    REQUIRE(h.pp->Process(rr));
    CHECK(rr.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(rr.host_primitive_type == xenos::PrimitiveType::kTriangleList);
    CHECK(rr.host_primitive_reset_enabled == false);
    std::vector<uint16_t> expected = SplitOracle<uint16_t>(
        fans, reset, [](const std::vector<uint16_t>& s) {
          return FanToListOracle(s);
        });
    // (1,2,3),(1,3,4),(5,6,7); the two-index tail draws nothing.
    CHECK(expected.size() == 9);
    CHECK(rr.host_draw_vertex_count == expected.size());
    CHECK(h.HostIndices<uint16_t>(rr) == expected);
    h.pp->EndFrame();
  }
}

TEST_CASE("PP5 normalize: DMA line loop -> host-converted strip, 32-bit, "
          "every endian, with and without reset",
          "[pp5][normalize]") {
  Harness h;
  REQUIRE(h.ok);
  for (xenos::Endian endian :
       {xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
        xenos::Endian::k16in32}) {
    std::vector<uint32_t> loop = {0x000101, 0x000202, 0x000303, 0x000404};
    h.Write32(kBase, loop, endian);
    SetDraw(h.regs, xenos::PrimitiveType::kLineLoop, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt32, endian, uint32_t(loop.size()), kBase,
            false, false);
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("endian " << uint32_t(endian));
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(r.host_primitive_type == xenos::PrimitiveType::kLineStrip);
    CHECK(r.host_index_format == xenos::IndexFormat::kInt32);
    CHECK(r.host_draw_vertex_count == loop.size() + 1);
    CHECK(r.line_loop_closing_index == 0);
    CHECK(h.HostIndices<uint32_t>(r) == LoopToStripOracle(loop));
    h.pp->EndFrame();

    // Reset 0xFFFFFF (24-bit): two loops. Shared behaviour: one strip, no
    // restart between the closed loops (see the file comment).
    const uint32_t reset = 0xFFFFFF;
    std::vector<uint32_t> loops = {1, 2, 3, reset, 4, 5, reset, 6};
    h.Write32(kBase, loops, endian);
    SetDraw(h.regs, xenos::PrimitiveType::kLineLoop, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt32, endian, uint32_t(loops.size()), kBase,
            /*multi_prim_ib=*/true, false, reset);
    PrimitiveProcessor::ProcessingResult rr{};
    REQUIRE(h.pp->Process(rr));
    CHECK(rr.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(rr.host_primitive_type == xenos::PrimitiveType::kLineStrip);
    std::vector<uint32_t> expected = SplitOracle<uint32_t>(
        loops, reset, [](const std::vector<uint32_t>& s) {
          return LoopToStripOracle(s);
        });
    CHECK(expected == std::vector<uint32_t>({1, 2, 3, 1, 4, 5, 4}));
    CHECK(rr.host_draw_vertex_count == expected.size());
    CHECK(h.HostIndices<uint32_t>(rr) == expected);
    h.pp->EndFrame();
  }
}

TEST_CASE("PP5 normalize: DMA quad list -> host-converted triangle list in "
          "the built-in table's order",
          "[pp5][normalize]") {
  Harness h(/*convert_quads=*/true);
  REQUIRE(h.ok);
  std::vector<uint16_t> quads = {0, 1, 2, 3, 10, 11, 12, 13, 7};
  h.Write16(kBase, quads, xenos::Endian::k8in16);
  SetDraw(h.regs, xenos::PrimitiveType::kQuadList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16,
          uint32_t(quads.size()), kBase, false, false);
  PrimitiveProcessor::ProcessingResult r{};
  REQUIRE(h.pp->Process(r));
  CHECK(r.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
  CHECK(r.host_primitive_type == xenos::PrimitiveType::kTriangleList);
  CHECK(r.host_draw_vertex_count == 12);
  CHECK(h.HostIndices<uint16_t>(r) == QuadsToListOracle(quads));
  h.pp->EndFrame();
  // Non-converting device: the guest quad list is not a converted class.
  Harness g(/*convert_quads=*/false);
  REQUIRE(g.ok);
  g.Write16(kBase, quads, xenos::Endian::kNone);
  SetDraw(g.regs, xenos::PrimitiveType::kQuadList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::kNone,
          uint32_t(quads.size()), kBase, false, false);
  CHECK_FALSE(metal::IsConvertedClassRoutable(
      g.regs, xenos::PrimitiveType::kQuadList, false));
}

TEST_CASE("PP5 normalize: custom 16-bit reset index on strips",
          "[pp5][normalize][reset]") {
  Harness h;
  REQUIRE(h.ok);
  const uint16_t reset = 0x1234;
  for (xenos::PrimitiveType strip : {xenos::PrimitiveType::kLineStrip,
                                     xenos::PrimitiveType::kTriangleStrip}) {
    for (xenos::Endian endian : {xenos::Endian::kNone, xenos::Endian::k8in16}) {
      // Reset used, 0xFFFF not a real index: replaced in place, 16-bit.
      std::vector<uint16_t> strip_indices = {1, 2, 3, reset, 4, 5, 6};
      h.Write16(kBase, strip_indices, endian);
      SetDraw(h.regs, strip, xenos::SourceSelect::kDMA,
              xenos::IndexFormat::kInt16, endian,
              uint32_t(strip_indices.size()), kBase, /*multi_prim_ib=*/true,
              false, reset);
      REQUIRE(metal::IsConvertedClassRoutable(h.regs, strip, false));
      PrimitiveProcessor::ProcessingResult r{};
      REQUIRE(h.pp->Process(r));
      INFO("strip " << uint32_t(strip) << " endian " << uint32_t(endian));
      CHECK(r.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
      CHECK(r.host_primitive_type == strip);
      CHECK(r.host_index_format == xenos::IndexFormat::kInt16);
      CHECK(r.host_primitive_reset_enabled == true);
      CHECK(r.host_draw_vertex_count == strip_indices.size());
      CHECK(h.HostIndices<uint16_t>(r) ==
            std::vector<uint16_t>({1, 2, 3, 0xFFFF, 4, 5, 6}));
      h.pp->EndFrame();

      // Reset used AND 0xFFFF a real vertex: widened to 32-bit with the
      // 0xFFFFFFFF sentinel and 0x0000FFFF kept as a vertex.
      std::vector<uint16_t> mixed = {1, 0xFFFF, 3, reset, 4, 0xFFFF};
      h.Write16(kBase, mixed, endian);
      SetDraw(h.regs, strip, xenos::SourceSelect::kDMA,
              xenos::IndexFormat::kInt16, endian, uint32_t(mixed.size()), kBase,
              /*multi_prim_ib=*/true, false, reset);
      PrimitiveProcessor::ProcessingResult rw{};
      REQUIRE(h.pp->Process(rw));
      CHECK(rw.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
      CHECK(rw.host_index_format == xenos::IndexFormat::kInt32);
      CHECK(rw.host_primitive_reset_enabled == true);
      CHECK(h.HostIndices<uint32_t>(rw) ==
            std::vector<uint32_t>({1, 0xFFFF, 3, 0xFFFFFFFFu, 4, 0xFFFF}));
      h.pp->EndFrame();

      // Reset enabled but never used: stays a guest-DMA stream with host
      // reset semantics (the emission binds it like PP4's fixed case).
      std::vector<uint16_t> unused = {1, 2, 3, 4};
      h.Write16(kBase, unused, endian);
      SetDraw(h.regs, strip, xenos::SourceSelect::kDMA,
              xenos::IndexFormat::kInt16, endian, uint32_t(unused.size()),
              kBase, /*multi_prim_ib=*/true, false, reset);
      PrimitiveProcessor::ProcessingResult ru{};
      REQUIRE(h.pp->Process(ru));
      CHECK(ru.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA);
      CHECK(ru.host_index_format == xenos::IndexFormat::kInt16);
      CHECK(ru.host_primitive_reset_enabled == true);
      h.pp->EndFrame();
    }
  }
}

TEST_CASE("PP5 normalize: 32-bit reset index on strips is pre-masked to 24 "
          "bits with the 0xFFFFFFFF sentinel, every endian",
          "[pp5][normalize][reset]") {
  Harness h;
  REQUIRE(h.ok);
  const uint32_t reset = 0x00ABCDEF;
  for (xenos::Endian endian :
       {xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
        xenos::Endian::k16in32}) {
    // Guest 32-bit indices carry an ignored top byte; the reset compare is on
    // the low 24 bits (0xFF000000 | reset matches the reset too).
    std::vector<uint32_t> guest = {0x01000001, 0x00000002, reset,
                                   0x00000003, 0xFF000000u | reset, 0x7F000004};
    h.Write32(kBase, guest, endian);
    SetDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
            xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt32, endian,
            uint32_t(guest.size()), kBase, /*multi_prim_ib=*/true, false,
            reset);
    REQUIRE(metal::IsConvertedClassRoutable(
        h.regs, xenos::PrimitiveType::kTriangleStrip, false));
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("endian " << uint32_t(endian));
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(r.host_index_format == xenos::IndexFormat::kInt32);
    CHECK(r.host_primitive_reset_enabled == true);
    CHECK(r.host_shader_index_endian == endian);
    CHECK(h.HostIndices<uint32_t>(r) ==
          std::vector<uint32_t>({1, 2, 0xFFFFFFFFu, 3, 0xFFFFFFFFu, 4}));
    h.pp->EndFrame();
  }
}

TEST_CASE("PP5 pool: cached results share a handle within a frame; EndFrame "
          "releases the pool; out-of-bounds reads refuse",
          "[pp5][pool]") {
  Harness h;
  REQUIRE(h.ok);
  // The shared per-frame cache only stores conversions of at least
  // primitive_processor_cache_min_indices (4096) guest indices.
  std::vector<uint16_t> fan(5000);
  for (size_t i = 0; i < fan.size(); ++i) {
    fan[i] = uint16_t(i * 3 + 1);
  }
  h.Write16(kBase, fan, xenos::Endian::kNone);
  SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::kNone,
          uint32_t(fan.size()), kBase, false, false);
  PrimitiveProcessor::ProcessingResult a{};
  REQUIRE(h.pp->Process(a));
  const size_t after_first = h.pp->converted_pool_bytes();
  CHECK(after_first >= a.host_draw_vertex_count * sizeof(uint16_t));
  PrimitiveProcessor::ProcessingResult b{};
  REQUIRE(h.pp->Process(b));
  // Same guest range, same frame: the shared per-frame cache returns the
  // first conversion (same handle), and the pool did not grow.
  CHECK(b.host_index_buffer_handle == a.host_index_buffer_handle);
  CHECK(h.pp->converted_pool_bytes() == after_first);
  CHECK(h.HostIndices<uint16_t>(b) == FanToListOracle(fan));
  // Bounds: one byte past the pool refuses; the exact slice succeeds.
  const size_t pool = h.pp->converted_pool_bytes();
  std::vector<uint8_t> scratch(pool + 16);
  CHECK_FALSE(h.pp->CopyConvertedIndexData(
      a.host_index_buffer_handle, pool + 1 - a.host_index_buffer_handle,
      scratch.data()));
  CHECK(h.pp->CopyConvertedIndexData(a.host_index_buffer_handle,
                                     pool - a.host_index_buffer_handle,
                                     scratch.data()));
  CHECK(h.pp->GetConvertedIndexData(pool, 1) == nullptr);
  CHECK(h.pp->GetConvertedIndexData(pool, 0) != nullptr);
  h.pp->EndFrame();
  CHECK(h.pp->converted_pool_bytes() == 0);
  CHECK(h.pp->GetConvertedIndexData(a.host_index_buffer_handle, 2) ==
        nullptr);
  // The next frame converts again (fresh handle at the pool start).
  PrimitiveProcessor::ProcessingResult c{};
  REQUIRE(h.pp->Process(c));
  CHECK(c.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
  CHECK(h.pp->converted_pool_bytes() == after_first);
  h.pp->EndFrame();
}

TEST_CASE("PP5 normalize: degenerate converted draws report zero host count",
          "[pp5][normalize]") {
  Harness h;
  REQUIRE(h.ok);
  std::vector<uint16_t> two = {1, 2};
  h.Write16(kBase, two, xenos::Endian::kNone);
  SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::kNone, 2, kBase, false,
          false);
  PrimitiveProcessor::ProcessingResult r{};
  REQUIRE(h.pp->Process(r));
  CHECK(r.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
  CHECK(r.host_draw_vertex_count == 0);
  h.pp->EndFrame();
  SetDraw(h.regs, xenos::PrimitiveType::kLineLoop, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::kNone, 1, kBase, false,
          false);
  PrimitiveProcessor::ProcessingResult l{};
  REQUIRE(h.pp->Process(l));
  CHECK(l.host_draw_vertex_count == 0);
  h.pp->EndFrame();
}

// ---- Part C: the converted-stream swap --------------------------------------

TEST_CASE("PP5 swap: SwapConvertedIndicesInto matches GpuSwap per element",
          "[pp5][swap]") {
  const uint32_t src32[] = {0x00010203u, 0x0000FFFFu, 0xFFFFFFFFu, 0x0000ABCDu};
  const uint16_t src16[] = {0x0102, 0xFFFF, 0x0304, 0xABCD};
  for (xenos::Endian endian :
       {xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
        xenos::Endian::k16in32}) {
    uint32_t out32[4];
    metal::SwapConvertedIndicesInto(out32, src32, 4, xenos::IndexFormat::kInt32,
                                    endian);
    for (uint32_t i = 0; i < 4; ++i) {
      INFO("endian " << uint32_t(endian) << " i " << i);
      CHECK(out32[i] == xenos::GpuSwap(src32[i], endian));
    }
    // The sentinel is endian-invariant.
    CHECK(out32[2] == 0xFFFFFFFFu);
    // A widened 16-bit index (zero upper half) under a 16-bit endian swaps
    // the bytes of its low half and stays a 16-bit-range value; a full dword
    // reversal (what SwapIndicesInto does for every non-kNone 32-bit endian)
    // would have produced 0xCDAB0000.
    if (endian == xenos::Endian::k8in16) {
      CHECK(out32[1] == 0x0000FFFFu);
      CHECK(out32[3] == 0x0000CDABu);
    }
  }
  // 16-bit streams keep the production SwapIndicesInto semantics.
  for (xenos::Endian endian : {xenos::Endian::kNone, xenos::Endian::k8in16}) {
    uint16_t out16[4], ref16[4];
    metal::SwapConvertedIndicesInto(out16, src16, 4, xenos::IndexFormat::kInt16,
                                    endian);
    metal::SwapIndicesInto(ref16, src16, 4, xenos::IndexFormat::kInt16, endian);
    CHECK(std::memcmp(out16, ref16, sizeof(out16)) == 0);
  }
}

// ---- Sampler residuals: border colour contract ------------------------------

TEST_CASE("Sampler border: ABGR colours and force_bc_w_to_max are exact; "
          "YCbCr blacks decay to black luma with the contract's alpha",
          "[sampler][border]") {
  using metal::DecideMetalSamplerBorder;
  using metal::MetalSamplerBorder;
  auto d = DecideMetalSamplerBorder(xenos::BorderColor::k_ABGR_Black, false);
  CHECK(d.border == MetalSamplerBorder::kTransparentBlack);
  CHECK_FALSE(d.approximated);
  d = DecideMetalSamplerBorder(xenos::BorderColor::k_ABGR_Black, true);
  CHECK(d.border == MetalSamplerBorder::kOpaqueBlack);
  CHECK_FALSE(d.approximated);
  for (bool force : {false, true}) {
    d = DecideMetalSamplerBorder(xenos::BorderColor::k_ABGR_White, force);
    CHECK(d.border == MetalSamplerBorder::kOpaqueWhite);
    CHECK_FALSE(d.approximated);
    for (xenos::BorderColor ycbcr : {xenos::BorderColor::k_ACBYCR_Black,
                                     xenos::BorderColor::k_ACBCRY_Black}) {
      d = DecideMetalSamplerBorder(ycbcr, force);
      CHECK(d.border == (force ? MetalSamplerBorder::kOpaqueBlack
                               : MetalSamplerBorder::kTransparentBlack));
      CHECK(d.approximated);
    }
  }
  // Key layout: the two border bits and the force bit never overlap the
  // fields packed around them (filters 0-5, clamps 6-14, aniso 15-17,
  // mip min 20-23, base map 24, mip max 26-29).
  CHECK(metal::kMetalSamplerKeyBorderShift == 18);
  CHECK(metal::kMetalSamplerKeyForceBcWShift == 30);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
