/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) conformance for the PP1 primitive-processor slice. Two
// contracts, both against production code (no replicas):
//
//   A. The routing partition (metal::IsPp1Routable): every routing positive and
//      every one-condition-at-a-time negative — point/rectangle/quad/fan/line-
//      loop/patch excluded, restart-enabled strips excluded, tessellated draws
//      excluded — for both auto and DMA sources.
//
//   B. The REAL production metal::MetalPrimitiveProcessor (instantiated
//      directly — pure C++, no Metal device) normalizing the routed set over
//      crafted register/index inputs. Asserts the ProcessingResult fields, and
//      for DMA draws byte-compares the host index bytes produced by the
//      production metal::SwapIndicesInto keyed on the NORMALIZED fields against
//      the same function keyed on the inline (raw guest) fields — the
//      byte-for-byte equivalence the slice must preserve. Also asserts the
//      RequestGuestIndexSharedMemoryRange no-op override performs ZERO
//      shared-memory uploads (a call counter on the test SharedMemory).
//      Includes the 0xFFFF / reset-disabled 16-bit edge (Gate 4): it must stay
//      a 16-bit kGuestDMA passthrough (no widening), matching the current
//      inline path.
//
// No Metal device is required: the routed set never uploads, converts, reads
// guest index memory, or touches a builtin/host-converted buffer.

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/metal/metal_primitive_processor.h"
#include "xenia/gpu/primitive_processor.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace {

// Device-free SharedMemory: the routed set never uploads. `upload_calls` lets
// the tests assert the production MetalPrimitiveProcessor::
// RequestGuestIndexSharedMemoryRange no-op override truly performs zero
// shared-memory uploads across routed DMA Process calls.
class NullSharedMemory final : public SharedMemory {
 public:
  explicit NullSharedMemory(Memory& memory) : SharedMemory(memory) {}
  // No explicit destructor: ~SharedMemory already calls ShutdownCommon().
  bool Init() { return InitializeCommon(); }
  int upload_calls = 0;
  bool UploadRanges(const std::pair<uint32_t, uint32_t>*, uint32_t) override {
    ++upload_calls;
    return true;
  }
};

// Builds a draw in the register file. tess = explicit-major-mode tessellation
// draw. multi_prim_ib = primitive restart enabled.
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

struct Pp1Harness {
  Memory* memory = nullptr;
  RegisterFile regs;
  TraceWriter trace{nullptr};
  std::unique_ptr<NullSharedMemory> sm;
  // The REAL production subclass (pure C++, no Metal device), exercised
  // directly — its configuration and no-op override are what runs, not a
  // mirror.
  std::unique_ptr<metal::MetalPrimitiveProcessor> pp;
  bool ok = false;
  Pp1Harness() {
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
    if (!pp->Initialize()) {
      return;
    }
    ok = true;
  }
};

constexpr xenos::PrimitiveType kRouted[] = {
    xenos::PrimitiveType::kLineList, xenos::PrimitiveType::kLineStrip,
    xenos::PrimitiveType::kTriangleList, xenos::PrimitiveType::kTriangleStrip};

constexpr xenos::PrimitiveType kNonRouted[] = {
    xenos::PrimitiveType::kPointList,     xenos::PrimitiveType::kRectangleList,
    xenos::PrimitiveType::kQuadList,      xenos::PrimitiveType::kTriangleFan,
    xenos::PrimitiveType::kLineLoop,      xenos::PrimitiveType::kQuadStrip,
    xenos::PrimitiveType::kTrianglePatch, xenos::PrimitiveType::kQuadPatch};

// ---- Part A: routing partition ----------------------------------------------

TEST_CASE("PP1 routing: the four native classes route (auto and DMA)",
          "[pp1][routing]") {
  RegisterFile regs;
  for (xenos::PrimitiveType pt : kRouted) {
    for (xenos::SourceSelect src :
         {xenos::SourceSelect::kAutoIndex, xenos::SourceSelect::kDMA}) {
      SetDraw(regs, pt, src, xenos::IndexFormat::kInt16, xenos::Endian::k8in16,
              6, 0x1000, /*multi_prim_ib=*/false, /*tess=*/false);
      INFO("prim " << uint32_t(pt) << " src " << uint32_t(src));
      CHECK(metal::IsPp1Routable(regs, pt));
    }
  }
}

TEST_CASE("PP1 routing: expanded/fan/loop/patch classes never route",
          "[pp1][routing]") {
  RegisterFile regs;
  for (xenos::PrimitiveType pt : kNonRouted) {
    SetDraw(regs, pt, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::k8in16, 6, 0x1000, false, false);
    INFO("non-routed prim " << uint32_t(pt));
    CHECK_FALSE(metal::IsPp1Routable(regs, pt));
  }
}

TEST_CASE("PP1 routing: restart-enabled strips are excluded, lists are not",
          "[pp1][routing]") {
  RegisterFile regs;
  // The single differing condition is multi_prim_ib_ena.
  for (xenos::PrimitiveType strip : {xenos::PrimitiveType::kLineStrip,
                                     xenos::PrimitiveType::kTriangleStrip}) {
    SetDraw(regs, strip, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::k8in16, 6, 0x1000, /*multi_prim_ib=*/true, false);
    INFO("restart strip " << uint32_t(strip));
    CHECK_FALSE(metal::IsPp1Routable(regs, strip));
  }
  // Lists are inherently non-restart: multi_prim_ib_ena set does not exclude
  // them (Process filters lists out of reset anyway).
  for (xenos::PrimitiveType list :
       {xenos::PrimitiveType::kLineList, xenos::PrimitiveType::kTriangleList}) {
    SetDraw(regs, list, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::k8in16, 6, 0x1000, /*multi_prim_ib=*/true, false);
    INFO("list with multi_prim_ib set " << uint32_t(list));
    CHECK(metal::IsPp1Routable(regs, list));
  }
}

TEST_CASE("PP1 routing: tessellated draws are excluded", "[pp1][routing]") {
  RegisterFile regs;
  // The single differing condition is explicit-major-mode tessellation.
  for (xenos::PrimitiveType pt : {xenos::PrimitiveType::kTriangleList,
                                  xenos::PrimitiveType::kTriangleStrip}) {
    SetDraw(regs, pt, xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16,
            xenos::Endian::k8in16, 6, 0x1000, false, /*tess=*/true);
    INFO("tessellated prim " << uint32_t(pt));
    CHECK_FALSE(metal::IsPp1Routable(regs, pt));
  }
}

TEST_CASE("PP1 routing: unsupported index source (immediate) is excluded",
          "[pp1][routing]") {
  RegisterFile regs;
  // The single differing condition is the index source: kImmediate is neither
  // auto nor guest-DMA, so it stays inline rather than entering Process.
  for (xenos::PrimitiveType pt : kRouted) {
    SetDraw(regs, pt, xenos::SourceSelect::kImmediate,
            xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 6, 0x1000, false,
            false);
    INFO("immediate-source prim " << uint32_t(pt));
    CHECK_FALSE(metal::IsPp1Routable(regs, pt));
  }
}

TEST_CASE("PP1 routing: local prim_type differing from VGT is excluded",
          "[pp1][routing]") {
  RegisterFile regs;
  // The single differing condition: the register holds a triangle list but the
  // local IssueDraw prim_type is a triangle strip. Routing would emit the
  // register-derived class, disagreeing with the checked one, so stay inline.
  SetDraw(regs, xenos::PrimitiveType::kTriangleList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 6, 0x1000, false,
          false);
  CHECK_FALSE(metal::IsPp1Routable(regs, xenos::PrimitiveType::kTriangleStrip));
  // Sanity: the matching prim_type still routes.
  CHECK(metal::IsPp1Routable(regs, xenos::PrimitiveType::kTriangleList));
}

// ---- Part A2: view-consistency guard ----------------------------------------
// Registers describe the draw; the local IssueDraw view (index_count + optional
// IndexBufferInfo fields) must match before routing. Positives, then one
// perturbed condition at a time. Registers are set with SetDraw; the local view
// is passed explicitly so a single field can diverge.

TEST_CASE("PP1 view: consistent auto and DMA views pass", "[pp1][view]") {
  RegisterFile regs;
  // Auto: no index buffer, count matches.
  SetDraw(regs, xenos::PrimitiveType::kTriangleList,
          xenos::SourceSelect::kAutoIndex, xenos::IndexFormat::kInt16,
          xenos::Endian::kNone, 12, 0, false, false);
  CHECK(metal::IsPp1DrawViewConsistent(regs, 12, /*has_index_buffer=*/false, 0,
                                       xenos::IndexFormat::kInt16,
                                       xenos::Endian::kNone, 0));
  // DMA: present, nonzero aligned base and all fields matching the registers.
  SetDraw(regs, xenos::PrimitiveType::kTriangleStrip, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt32, xenos::Endian::k8in32, 12, 0x2000, false,
          false);
  CHECK(metal::IsPp1DrawViewConsistent(regs, 12, /*has_index_buffer=*/true,
                                       0x2000, xenos::IndexFormat::kInt32,
                                       xenos::Endian::k8in32, 12));
}

TEST_CASE(
    "PP1 view: index_count differing from VGT num_indices fails "
    "(replay_index_window class)",
    "[pp1][view]") {
  RegisterFile regs;
  // num_indices = 12, but the local index_count was rewritten to 6.
  SetDraw(regs, xenos::PrimitiveType::kTriangleList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 12, 0x2000, false,
          false);
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, /*index_count=*/6, true,
                                             0x2000, xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 12));
}

TEST_CASE("PP1 view: auto source with a bound index buffer fails",
          "[pp1][view]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kLineList,
          xenos::SourceSelect::kAutoIndex, xenos::IndexFormat::kInt16,
          xenos::Endian::kNone, 8, 0, false, false);
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 8, /*has_index_buffer=*/true,
                                             0x2000, xenos::IndexFormat::kInt16,
                                             xenos::Endian::kNone, 8));
}

TEST_CASE("PP1 view: DMA base zero stays inline", "[pp1][view]") {
  RegisterFile regs;
  // VGT_DMA_BASE = 0 (the inline non-indexed fallback).
  SetDraw(regs, xenos::PrimitiveType::kTriangleList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 6, 0, false,
          false);
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 6, /*has_index_buffer=*/true,
                                             0, xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 6));
}

TEST_CASE("PP1 view: DMA local fields differing from registers fail",
          "[pp1][view]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kTriangleList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 6, 0x2000, false,
          false);
  // Missing index buffer.
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(
      regs, 6, /*has_index_buffer=*/false, 0x2000, xenos::IndexFormat::kInt16,
      xenos::Endian::k8in16, 6));
  // Base differs from the register base.
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 6, true, 0x4000,
                                             xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 6));
  // Format differs.
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 6, true, 0x2000,
                                             xenos::IndexFormat::kInt32,
                                             xenos::Endian::k8in16, 6));
  // Endian differs.
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 6, true, 0x2000,
                                             xenos::IndexFormat::kInt16,
                                             xenos::Endian::kNone, 6));
  // IndexBufferInfo count differs from VGT num_indices.
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 6, true, 0x2000,
                                             xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 5));
}

TEST_CASE("PP1 view: short DMA buffer (num_words < num_indices) stays inline",
          "[pp1][view]") {
  RegisterFile regs;
  // num_indices = 8 but the DMA buffer holds only 4 words. Process clamps the
  // draw to 4; the inline Metal path draws all 8. Routing must be refused so
  // that clamp cannot change behavior.
  SetDraw(regs, xenos::PrimitiveType::kTriangleList, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 8, 0x2000, false,
          false);
  reg::VGT_DMA_SIZE ds;
  ds.value = regs.values[reg::VGT_DMA_SIZE::register_index];
  ds.num_words = 4;  // short buffer (single differing condition)
  regs.values[reg::VGT_DMA_SIZE::register_index] = ds.value;
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 8, /*has_index_buffer=*/true,
                                             0x2000, xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 8));
  // Sanity: a sufficient buffer (num_words == num_indices) is consistent.
  ds.num_words = 8;
  regs.values[reg::VGT_DMA_SIZE::register_index] = ds.value;
  CHECK(metal::IsPp1DrawViewConsistent(regs, 8, true, 0x2000,
                                       xenos::IndexFormat::kInt16,
                                       xenos::Endian::k8in16, 8));
}

// ---- Part B: Process normalization + byte equivalence -----------------------

// Produces the host index bytes SwapIndicesInto would write for the given
// format/endian over `src`.
std::vector<uint8_t> SwapBytes(const std::vector<uint8_t>& src,
                               xenos::IndexFormat format, xenos::Endian endian,
                               uint32_t count) {
  std::vector<uint8_t> out(src.size(), 0);
  metal::SwapIndicesInto(out.data(), src.data(), count, format, endian);
  return out;
}

TEST_CASE("PP1 normalize: routed DMA draws are field- and byte-equivalent",
          "[pp1][normalize]") {
  Pp1Harness h;
  REQUIRE(h.ok);

  struct Case {
    xenos::IndexFormat format;
    xenos::Endian endian;
  };
  const Case cases[] = {
      {xenos::IndexFormat::kInt16, xenos::Endian::kNone},
      {xenos::IndexFormat::kInt16, xenos::Endian::k8in16},
      {xenos::IndexFormat::kInt16, xenos::Endian::k8in32},  // normalized->8in16
      {xenos::IndexFormat::kInt16, xenos::Endian::k16in32},  // normalized->None
      {xenos::IndexFormat::kInt32, xenos::Endian::kNone},
      {xenos::IndexFormat::kInt32, xenos::Endian::k8in16},
      {xenos::IndexFormat::kInt32, xenos::Endian::k8in32},
      {xenos::IndexFormat::kInt32, xenos::Endian::k16in32},
  };

  const uint32_t count = 8;
  for (xenos::PrimitiveType pt : kRouted) {
    for (const Case& c : cases) {
      // Crafted guest index bytes (not the reset sentinel).
      std::vector<uint8_t> guest;
      if (c.format == xenos::IndexFormat::kInt16) {
        for (uint16_t v :
             {0x0102, 0x0304, 0x0506, 0x0708, 0x090A, 0x0B0C, 0x0D0E, 0x0F10}) {
          guest.push_back(uint8_t(v & 0xFF));
          guest.push_back(uint8_t(v >> 8));
        }
      } else {
        for (uint32_t v :
             {0x00010203u, 0x00040506u, 0x00070809u, 0x000A0B0Cu, 0x000D0E0Fu,
              0x00101112u, 0x00131415u, 0x00161718u}) {
          guest.push_back(uint8_t(v & 0xFF));
          guest.push_back(uint8_t((v >> 8) & 0xFF));
          guest.push_back(uint8_t((v >> 16) & 0xFF));
          guest.push_back(uint8_t((v >> 24) & 0xFF));
        }
      }

      SetDraw(h.regs, pt, xenos::SourceSelect::kDMA, c.format, c.endian, count,
              0x2000, /*multi_prim_ib=*/false, /*tess=*/false);
      REQUIRE(metal::IsPp1Routable(h.regs, pt));

      PrimitiveProcessor::ProcessingResult r{};
      REQUIRE(h.pp->Process(r));

      INFO("prim " << uint32_t(pt) << " fmt " << uint32_t(c.format)
                   << " endian " << uint32_t(c.endian));
      // Fields: no conversion, no reset, DMA passthrough.
      CHECK(r.host_primitive_type == pt);
      CHECK(r.host_draw_vertex_count == count);
      CHECK(r.host_primitive_reset_enabled == false);
      CHECK(r.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA);
      CHECK(r.host_index_format == c.format);
      CHECK(r.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex);

      // Byte equivalence: normalized fields vs the raw inline fields produce
      // identical host index bytes.
      std::vector<uint8_t> pp_bytes = SwapBytes(
          guest, r.host_index_format, r.host_shader_index_endian, count);
      std::vector<uint8_t> inline_bytes =
          SwapBytes(guest, c.format, c.endian, count);
      CHECK(pp_bytes == inline_bytes);
    }
  }
  // The production RequestGuestIndexSharedMemoryRange no-op override means the
  // routed DMA path performs zero shared-memory uploads (hence no intra-frame
  // hazard flush / commit-wait). Trace equivalence is preserved separately by
  // the command processor's explicit WriteMemoryRead before the route, not by
  // this counter.
  CHECK(h.sm->upload_calls == 0);
}

TEST_CASE("PP1 normalize: routed auto draws are field-equivalent (no bytes)",
          "[pp1][normalize]") {
  Pp1Harness h;
  REQUIRE(h.ok);
  const uint32_t count = 9;
  for (xenos::PrimitiveType pt : kRouted) {
    SetDraw(h.regs, pt, xenos::SourceSelect::kAutoIndex,
            xenos::IndexFormat::kInt16, xenos::Endian::kNone, count, 0,
            /*multi_prim_ib=*/false, /*tess=*/false);
    REQUIRE(metal::IsPp1Routable(h.regs, pt));
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("auto prim " << uint32_t(pt));
    CHECK(r.host_primitive_type == pt);
    CHECK(r.host_draw_vertex_count == count);
    CHECK(r.host_primitive_reset_enabled == false);
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kNone);
    CHECK(r.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex);
  }
}

TEST_CASE(
    "PP1 normalize: 0xFFFF in a reset-disabled 16-bit strip is NOT widened "
    "(Gate 4)",
    "[pp1][normalize][gate4]") {
  Pp1Harness h;
  REQUIRE(h.ok);
  // A real 0xFFFF vertex index in a reset-disabled 16-bit strip. The widening
  // safeguard is gated off (host_primitive_reset_always_enabled == false), so
  // Process must keep this a 16-bit kGuestDMA passthrough — byte-identical to
  // the current inline path (which passes 0xFFFF straight to Metal). If this
  // ever returns kHostConverted / kInt32, PP1 stopped being behavior-neutral.
  const uint32_t count = 6;
  std::vector<uint8_t> guest;
  for (uint16_t v : {0x0001, 0xFFFF, 0x0002, 0x0003, 0xFFFF, 0x0004}) {
    guest.push_back(uint8_t(v & 0xFF));
    guest.push_back(uint8_t(v >> 8));
  }
  for (xenos::PrimitiveType strip : {xenos::PrimitiveType::kLineStrip,
                                     xenos::PrimitiveType::kTriangleStrip}) {
    SetDraw(h.regs, strip, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt16, xenos::Endian::kNone, count, 0x3000,
            /*multi_prim_ib=*/false,
            /*tess=*/false);
    REQUIRE(metal::IsPp1Routable(h.regs, strip));
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("reset-disabled strip with 0xFFFF, prim " << uint32_t(strip));
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA);
    CHECK(r.host_index_format == xenos::IndexFormat::kInt16);
    CHECK(r.host_primitive_reset_enabled == false);
    // Bytes pass through unchanged (endian kNone, 16-bit).
    std::vector<uint8_t> pp_bytes = SwapBytes(
        guest, r.host_index_format, r.host_shader_index_endian, count);
    CHECK(pp_bytes == guest);
  }
}

// ---- PP4 fixed-sentinel DMA strips -----------------------------------------

TEST_CASE("PP4 fixed reset: exact routing partition",
          "[pp4][primitive-reset][fixed][routing]") {
  RegisterFile regs;
  constexpr xenos::Endian kEndians[] = {
      xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
      xenos::Endian::k16in32};
  for (xenos::PrimitiveType strip : {xenos::PrimitiveType::kLineStrip,
                                     xenos::PrimitiveType::kTriangleStrip}) {
    for (xenos::Endian endian : kEndians) {
      SetDraw(regs, strip, xenos::SourceSelect::kDMA,
              xenos::IndexFormat::kInt16, endian, 8, 0x4000,
              /*multi_prim_ib=*/true, /*tess=*/false, UINT16_MAX);
      INFO("strip " << uint32_t(strip) << " endian " << uint32_t(endian));
      CHECK(metal::IsFixedResetStripRoutable(regs, strip));
      CHECK_FALSE(metal::IsPp1Routable(regs, strip));
      CHECK(metal::IsPp1DrawViewConsistent(
          regs, 8, true, 0x4000, xenos::IndexFormat::kInt16, endian, 8));
    }
  }

  auto set_base = [&](xenos::PrimitiveType primitive_type) {
    SetDraw(regs, primitive_type, xenos::SourceSelect::kDMA,
            xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 8, 0x4000,
            /*multi_prim_ib=*/true, /*tess=*/false, UINT16_MAX);
  };
  set_base(xenos::PrimitiveType::kTriangleStrip);
  CHECK_FALSE(
      metal::IsFixedResetStripRoutable(regs, xenos::PrimitiveType::kLineStrip));

  set_base(xenos::PrimitiveType::kTriangleList);
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleList));

  set_base(xenos::PrimitiveType::kTriangleStrip);
  reg::VGT_DRAW_INITIATOR draw = regs.Get<reg::VGT_DRAW_INITIATOR>();
  draw.source_select = xenos::SourceSelect::kAutoIndex;
  regs.values[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));

  set_base(xenos::PrimitiveType::kTriangleStrip);
  draw = regs.Get<reg::VGT_DRAW_INITIATOR>();
  draw.source_select = xenos::SourceSelect::kImmediate;
  regs.values[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));

  set_base(xenos::PrimitiveType::kTriangleStrip);
  draw = regs.Get<reg::VGT_DRAW_INITIATOR>();
  draw.index_size = xenos::IndexFormat::kInt32;
  regs.values[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));

  SetDraw(regs, xenos::PrimitiveType::kTriangleStrip, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 8, 0x4000,
          /*multi_prim_ib=*/false,
          /*tess=*/false, UINT16_MAX);
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));

  SetDraw(regs, xenos::PrimitiveType::kTriangleStrip, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 8, 0x4000,
          /*multi_prim_ib=*/true,
          /*tess=*/false, 0x1234);
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));

  SetDraw(regs, xenos::PrimitiveType::kTriangleStrip, xenos::SourceSelect::kDMA,
          xenos::IndexFormat::kInt16, xenos::Endian::k8in16, 8, 0x4000,
          /*multi_prim_ib=*/true,
          /*tess=*/true, UINT16_MAX);
  CHECK_FALSE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));

  // The register-only predicate is positive, but the command processor's
  // required local/register view guard rejects one-field-at-a-time divergence.
  set_base(xenos::PrimitiveType::kTriangleStrip);
  REQUIRE(metal::IsFixedResetStripRoutable(
      regs, xenos::PrimitiveType::kTriangleStrip));
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 7, true, 0x4000,
                                             xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 8));
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 8, false, 0x4000,
                                             xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 8));
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 8, true, 0x5000,
                                             xenos::IndexFormat::kInt16,
                                             xenos::Endian::k8in16, 8));
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 8, true, 0x4000,
                                             xenos::IndexFormat::kInt32,
                                             xenos::Endian::k8in16, 8));
}

TEST_CASE("PP4 fixed reset: production Process preserves DMA strip shape",
          "[pp4][primitive-reset][fixed][normalize]") {
  Pp1Harness h;
  REQUIRE(h.ok);
  constexpr xenos::Endian kEndians[] = {
      xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
      xenos::Endian::k16in32};
  const std::vector<uint8_t> guest = {0x01, 0x00, 0x02, 0x00, 0xFF, 0xFF,
                                      0x03, 0x00, 0x04, 0x00, 0xFF, 0xFF,
                                      0x05, 0x00, 0x06, 0x00};
  for (xenos::PrimitiveType strip : {xenos::PrimitiveType::kLineStrip,
                                     xenos::PrimitiveType::kTriangleStrip}) {
    for (xenos::Endian endian : kEndians) {
      SetDraw(h.regs, strip, xenos::SourceSelect::kDMA,
              xenos::IndexFormat::kInt16, endian, 8, 0x4000,
              /*multi_prim_ib=*/true, /*tess=*/false, UINT16_MAX);
      REQUIRE(metal::IsFixedResetStripRoutable(h.regs, strip));
      PrimitiveProcessor::ProcessingResult result{};
      REQUIRE(h.pp->Process(result));
      INFO("strip " << uint32_t(strip) << " endian " << uint32_t(endian));
      CHECK(result.guest_primitive_type == strip);
      CHECK(result.host_primitive_type == strip);
      CHECK(result.host_vertex_shader_type ==
            Shader::HostVertexShaderType::kVertex);
      CHECK(result.host_draw_vertex_count == 8);
      CHECK(result.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA);
      CHECK(result.guest_index_base == 0x4000);
      CHECK(result.host_index_format == xenos::IndexFormat::kInt16);
      const xenos::Endian normalized_endian =
          endian == xenos::Endian::k8in32
              ? xenos::Endian::k8in16
              : (endian == xenos::Endian::k16in32 ? xenos::Endian::kNone
                                                  : endian);
      CHECK(result.host_shader_index_endian == normalized_endian);
      CHECK(result.host_primitive_reset_enabled);
      CHECK(SwapBytes(guest, result.host_index_format,
                      result.host_shader_index_endian, 8) ==
            SwapBytes(guest, xenos::IndexFormat::kInt16, endian, 8));
    }
  }
  CHECK(h.sm->upload_calls == 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xe
