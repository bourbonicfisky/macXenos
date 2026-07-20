/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) conformance for the PP2-auto slice: auto-indexed triangle
// fans routed through the shared PrimitiveProcessor kHostBuiltinForAuto path.
// All contracts run against production code (no replicas):
//
//   A. Fan emit order (production PrimitiveProcessor::TriangleFanToList): the
//      default (D3D12/Vulkan) order is (v1,v2,v0),(v2,v3,v0),... and the Metal
//      center-first order is (v0,v1,v2),(v0,v2,v3),... Using a provoking-
//      sensitive index sequence (distinct values) pins both first- and last-
//      vertex behavior, proving the two orders are observably different and
//      that the default (D3D/Vulkan) bytes are unchanged.
//
//   B. Routing partition (metal::IsAutoFanRoutable): auto fans route; DMA/reset
//      fans, tessellated fans, non-fan classes, and prim-type mismatches do
//      not.
//
//   C. The REAL production metal::MetalPrimitiveProcessor (pure C++, no Metal
//      device) normalizing an auto fan to kHostBuiltinForAuto, plus its
//      CopyBuiltinIndexData built-in table read: the 16-bit expansion is
//      byte-identical to the retired inline auto-fan expansion (0, k, k+1),
//      across representative counts up to the UINT16_MAX maximum, with count<3
//      a no-draw success (host_draw_vertex_count == 0) and out-of-bounds reads
//      refused.
//
// No Metal device is required: the built-in fan table is a CPU byte vector.

#include <cstdint>
#include <cstring>
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

// Device-free SharedMemory (auto fans never upload).
class NullSharedMemory final : public SharedMemory {
 public:
  explicit NullSharedMemory(Memory& memory) : SharedMemory(memory) {}
  bool Init() { return InitializeCommon(); }
  bool UploadRanges(const std::pair<uint32_t, uint32_t>*, uint32_t) override {
    return true;
  }
};

void SetDraw(RegisterFile& regs, xenos::PrimitiveType prim_type,
             xenos::SourceSelect source_select, uint32_t count, uint32_t base,
             bool multi_prim_ib, bool tess) {
  std::memset(regs.values, 0, sizeof(regs.values));

  reg::VGT_DRAW_INITIATOR di;
  di.value = 0;
  di.prim_type = prim_type;
  di.source_select = source_select;
  di.index_size = xenos::IndexFormat::kInt16;
  di.num_indices = count;
  di.major_mode =
      tess ? xenos::MajorMode::kExplicit : xenos::MajorMode::kImplicit;
  regs.values[reg::VGT_DRAW_INITIATOR::register_index] = di.value;

  reg::VGT_DMA_SIZE ds;
  ds.value = 0;
  ds.num_words = count;
  ds.swap_mode = xenos::Endian::kNone;
  regs.values[reg::VGT_DMA_SIZE::register_index] = ds.value;

  regs.values[XE_GPU_REG_VGT_DMA_BASE] = base;

  reg::PA_SU_SC_MODE_CNTL pa;
  pa.value = 0;
  pa.multi_prim_ib_ena = multi_prim_ib ? 1 : 0;
  regs.values[reg::PA_SU_SC_MODE_CNTL::register_index] = pa.value;

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
  Harness() {
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

// The retired inline auto-fan expansion, center-first: (0, k, k+1) 16-bit.
std::vector<uint16_t> ExpectedCenterFirstAutoFan(uint32_t count) {
  std::vector<uint16_t> out;
  for (uint32_t k = 1; k + 1 < count; ++k) {
    out.push_back(0);
    out.push_back(uint16_t(k));
    out.push_back(uint16_t(k + 1));
  }
  return out;
}

// ---- Part A: fan emit order (production static) -----------------------------

TEST_CASE(
    "PP2-auto order: TriangleFanToList default is D3D, Metal is center-first",
    "[pp2auto][order]") {
  // Distinct guest index values so first- and last-vertex (provoking) behavior
  // is observable, not just winding.
  const uint16_t src[] = {10, 20, 30, 40, 50};
  const uint32_t n = 5;
  struct Identity {
    uint16_t operator()(uint16_t v) const { return v; }
  } passthrough;

  // Default order (D3D12/Vulkan): (v1,v2,v0),(v2,v3,v0),(v3,v4,v0).
  std::vector<uint16_t> d3d(
      PrimitiveProcessor::GetTriangleFanListIndexCount(n));
  PrimitiveProcessor::TriangleFanToList(d3d.data(), src, n, passthrough,
                                        /*first_provoking=*/false);
  const uint16_t d3d_expected[] = {20, 30, 10, 30, 40, 10, 40, 50, 10};
  CHECK(std::vector<uint16_t>(std::begin(d3d_expected),
                              std::end(d3d_expected)) == d3d);

  // Metal center-first: (v0,v1,v2),(v0,v2,v3),(v0,v3,v4).
  std::vector<uint16_t> metal(
      PrimitiveProcessor::GetTriangleFanListIndexCount(n));
  PrimitiveProcessor::TriangleFanToList(metal.data(), src, n, passthrough,
                                        /*first_provoking=*/true);
  const uint16_t metal_expected[] = {10, 20, 30, 10, 30, 40, 10, 40, 50};
  CHECK(std::vector<uint16_t>(std::begin(metal_expected),
                              std::end(metal_expected)) == metal);

  // The two orders must actually differ (provoking-vertex sensitivity).
  CHECK(d3d != metal);
}

// ---- Part B: routing partition ----------------------------------------------

TEST_CASE("PP2-auto routing: auto fans route", "[pp2auto][routing]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, 9, 0, false, false);
  CHECK(metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
}

TEST_CASE("PP2-auto routing: DMA/reset fans stay inline",
          "[pp2auto][routing]") {
  RegisterFile regs;
  // DMA fan: not routed (stays on the inline converter).
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
          9, 0x1000, /*multi_prim_ib=*/false, false);
  CHECK_FALSE(
      metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
  // Reset-enabled DMA fan: still not routed (reset is a later slice).
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan, xenos::SourceSelect::kDMA,
          9, 0x1000, /*multi_prim_ib=*/true, false);
  CHECK_FALSE(
      metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
}

TEST_CASE("PP2-auto routing: tessellated auto fans stay inline",
          "[pp2auto][routing]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, 9, 0, false, /*tess=*/true);
  CHECK_FALSE(
      metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
}

TEST_CASE("PP2-auto routing: non-fan classes and prim mismatches never route",
          "[pp2auto][routing]") {
  RegisterFile regs;
  for (xenos::PrimitiveType pt :
       {xenos::PrimitiveType::kTriangleList,
        xenos::PrimitiveType::kTriangleStrip, xenos::PrimitiveType::kLineList,
        xenos::PrimitiveType::kPointList, xenos::PrimitiveType::kRectangleList,
        xenos::PrimitiveType::kQuadList, xenos::PrimitiveType::kLineLoop}) {
    SetDraw(regs, pt, xenos::SourceSelect::kAutoIndex, 9, 0, false, false);
    INFO("non-fan prim " << uint32_t(pt));
    CHECK_FALSE(metal::IsAutoFanRoutable(regs, pt));
  }
  // Register holds a fan but the local prim_type is a triangle list: the local
  // view disagrees with the register the emission would map, so stay inline.
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, 9, 0, false, false);
  CHECK_FALSE(
      metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleList));
  // The matching fan prim_type still routes.
  CHECK(metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
}

TEST_CASE("PP2-auto routing: IsPp1Routable still rejects fans (unweakened)",
          "[pp2auto][routing]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, 9, 0, false, false);
  CHECK_FALSE(metal::IsPp1Routable(regs, xenos::PrimitiveType::kTriangleFan));
}

// ---- Part C: Process + built-in table ---------------------------------------

TEST_CASE("PP2-auto normalize: auto fan -> kHostBuiltinForAuto center-first",
          "[pp2auto][normalize]") {
  Harness h;
  REQUIRE(h.ok);
  for (uint32_t count : {3u, 4u, 5u, 9u, 100u, 1000u}) {
    SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
            xenos::SourceSelect::kAutoIndex, count, 0, false, false);
    REQUIRE(
        metal::IsAutoFanRoutable(h.regs, xenos::PrimitiveType::kTriangleFan));

    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("auto fan count " << count);
    CHECK(r.host_primitive_type == xenos::PrimitiveType::kTriangleList);
    CHECK(r.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex);
    CHECK(r.host_primitive_reset_enabled == false);
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto);
    CHECK(r.host_draw_vertex_count ==
          PrimitiveProcessor::GetTriangleFanListIndexCount(count));

    // The built-in table slice must be byte-identical to the retired inline
    // auto-fan expansion (0, k, k+1).
    std::vector<uint16_t> expected = ExpectedCenterFirstAutoFan(count);
    REQUIRE(expected.size() == r.host_draw_vertex_count);
    std::vector<uint16_t> got(r.host_draw_vertex_count, 0xEEEE);
    REQUIRE(h.pp->CopyBuiltinIndexData(
        r.host_index_buffer_handle,
        size_t(r.host_draw_vertex_count) * sizeof(uint16_t), got.data()));
    CHECK(got == expected);
  }
}

TEST_CASE("PP2-auto normalize: count<3 is a no-draw success",
          "[pp2auto][normalize]") {
  Harness h;
  REQUIRE(h.ok);
  for (uint32_t count : {0u, 1u, 2u}) {
    SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
            xenos::SourceSelect::kAutoIndex, count, 0, false, false);
    REQUIRE(
        metal::IsAutoFanRoutable(h.regs, xenos::PrimitiveType::kTriangleFan));
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    INFO("auto fan count " << count);
    // Zero expanded indices; the command processor treats
    // host_draw_vertex_count
    // == 0 as a skip (matching the inline count<3 skip).
    CHECK(r.host_draw_vertex_count == 0);
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto);
  }
}

TEST_CASE("PP2-auto normalize: maximum auto fan count is covered",
          "[pp2auto][normalize]") {
  Harness h;
  REQUIRE(h.ok);
  const uint32_t count = UINT16_MAX;  // 16-bit VGT num_indices maximum.
  SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, count, 0, false, false);
  PrimitiveProcessor::ProcessingResult r{};
  REQUIRE(h.pp->Process(r));
  CHECK(r.host_draw_vertex_count ==
        PrimitiveProcessor::GetTriangleFanListIndexCount(count));
  // The whole slice is in bounds of the owned table.
  std::vector<uint16_t> got(r.host_draw_vertex_count, 0);
  CHECK(h.pp->CopyBuiltinIndexData(
      r.host_index_buffer_handle,
      size_t(r.host_draw_vertex_count) * sizeof(uint16_t), got.data()));
  // Spot-check the first and last triangle.
  REQUIRE(got.size() >= 6);
  CHECK(got[0] == 0);
  CHECK(got[1] == 1);
  CHECK(got[2] == 2);
  CHECK(got[got.size() - 3] == 0);
  CHECK(got[got.size() - 2] == uint16_t(count - 2));
  CHECK(got[got.size() - 1] == uint16_t(count - 1));
}

TEST_CASE("PP2-auto: CopyBuiltinIndexData refuses out-of-bounds reads",
          "[pp2auto][bounds]") {
  Harness h;
  REQUIRE(h.ok);
  // Establish the table via a normal Process.
  SetDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, 9, 0, false, false);
  PrimitiveProcessor::ProcessingResult r{};
  REQUIRE(h.pp->Process(r));
  const size_t table_bytes =
      size_t(PrimitiveProcessor::GetTriangleFanListIndexCount(UINT16_MAX)) *
      sizeof(uint16_t);
  std::vector<uint8_t> dst(64, 0);
  // A byte_count one past the table end is refused.
  CHECK_FALSE(h.pp->CopyBuiltinIndexData(0, table_bytes + 1, dst.data()));
  // A handle past the table end is refused.
  CHECK_FALSE(h.pp->CopyBuiltinIndexData(table_bytes + 2, sizeof(uint16_t),
                                         dst.data()));
  // An exact in-bounds read at the tail succeeds.
  CHECK(h.pp->CopyBuiltinIndexData(table_bytes - sizeof(uint16_t),
                                   sizeof(uint16_t), dst.data()));
}

}  // namespace
}  // namespace gpu
}  // namespace xe
