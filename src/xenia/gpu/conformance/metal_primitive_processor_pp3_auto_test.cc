/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) conformance for PP3-auto line-loop closure. All assertions
// drive production code: the real IsAutoLineLoopRoutable predicate and the real
// MetalPrimitiveProcessor::Process path. No transform is replicated here.

#include <cstdint>
#include <cstring>
#include <memory>

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

class NullSharedMemory final : public SharedMemory {
 public:
  explicit NullSharedMemory(Memory& memory) : SharedMemory(memory) {}
  bool Init() { return InitializeCommon(); }
  int upload_calls = 0;
  bool UploadRanges(const std::pair<uint32_t, uint32_t>*, uint32_t) override {
    ++upload_calls;
    return true;
  }
};

void SetDraw(RegisterFile& regs, xenos::PrimitiveType prim_type,
             xenos::SourceSelect source_select, uint32_t count,
             bool multi_prim_ib, bool tess) {
  std::memset(regs.values, 0, sizeof(regs.values));

  reg::VGT_DRAW_INITIATOR draw;
  draw.value = 0;
  draw.prim_type = prim_type;
  draw.source_select = source_select;
  draw.index_size = xenos::IndexFormat::kInt16;
  draw.num_indices = count;
  draw.major_mode =
      tess ? xenos::MajorMode::kExplicit : xenos::MajorMode::kImplicit;
  regs.values[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;

  reg::VGT_DMA_SIZE dma_size;
  dma_size.value = 0;
  dma_size.num_words = count;
  dma_size.swap_mode = xenos::Endian::kNone;
  regs.values[reg::VGT_DMA_SIZE::register_index] = dma_size.value;

  reg::PA_SU_SC_MODE_CNTL mode;
  mode.value = 0;
  mode.multi_prim_ib_ena = multi_prim_ib ? 1 : 0;
  regs.values[reg::PA_SU_SC_MODE_CNTL::register_index] = mode.value;

  reg::VGT_OUTPUT_PATH_CNTL output_path;
  output_path.value = 0;
  output_path.path_select = tess ? xenos::VGTOutputPath::kTessellationEnable
                                 : xenos::VGTOutputPath::kVertexReuse;
  regs.values[reg::VGT_OUTPUT_PATH_CNTL::register_index] = output_path.value;
}

struct Harness {
  Memory* memory = nullptr;
  RegisterFile regs;
  TraceWriter trace{nullptr};
  std::unique_ptr<NullSharedMemory> shared_memory;
  std::unique_ptr<metal::MetalPrimitiveProcessor> processor;
  bool ok = false;

  Harness() {
    std::memset(regs.values, 0, sizeof(regs.values));
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    shared_memory = std::make_unique<NullSharedMemory>(*memory);
    if (!shared_memory->Init()) {
      return;
    }
    processor = std::make_unique<metal::MetalPrimitiveProcessor>(
        regs, *memory, trace, *shared_memory);
    ok = processor->Initialize();
  }
};

TEST_CASE("PP3-auto routing: eligible line loops route", "[pp3auto][routing]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kLineLoop,
          xenos::SourceSelect::kAutoIndex, 8, false, false);
  CHECK(metal::IsAutoLineLoopRoutable(regs, xenos::PrimitiveType::kLineLoop));
}

TEST_CASE("PP3-auto routing: DMA, immediate, reset, and tessellation reject",
          "[pp3auto][routing]") {
  RegisterFile regs;
  for (xenos::SourceSelect source :
       {xenos::SourceSelect::kDMA, xenos::SourceSelect::kImmediate}) {
    SetDraw(regs, xenos::PrimitiveType::kLineLoop, source, 8, false, false);
    INFO("source " << uint32_t(source));
    CHECK_FALSE(
        metal::IsAutoLineLoopRoutable(regs, xenos::PrimitiveType::kLineLoop));
  }

  SetDraw(regs, xenos::PrimitiveType::kLineLoop,
          xenos::SourceSelect::kAutoIndex, 8, true, false);
  CHECK_FALSE(
      metal::IsAutoLineLoopRoutable(regs, xenos::PrimitiveType::kLineLoop));

  SetDraw(regs, xenos::PrimitiveType::kLineLoop,
          xenos::SourceSelect::kAutoIndex, 8, false, true);
  CHECK_FALSE(
      metal::IsAutoLineLoopRoutable(regs, xenos::PrimitiveType::kLineLoop));
}

TEST_CASE("PP3-auto routing: non-loops and mismatched local type reject",
          "[pp3auto][routing]") {
  RegisterFile regs;
  for (xenos::PrimitiveType primitive :
       {xenos::PrimitiveType::kLineList, xenos::PrimitiveType::kLineStrip,
        xenos::PrimitiveType::kTriangleList,
        xenos::PrimitiveType::kTriangleStrip,
        xenos::PrimitiveType::kTriangleFan, xenos::PrimitiveType::kPointList,
        xenos::PrimitiveType::kRectangleList,
        xenos::PrimitiveType::kQuadList}) {
    SetDraw(regs, primitive, xenos::SourceSelect::kAutoIndex, 8, false, false);
    INFO("primitive " << uint32_t(primitive));
    CHECK_FALSE(metal::IsAutoLineLoopRoutable(regs, primitive));
  }

  SetDraw(regs, xenos::PrimitiveType::kLineLoop,
          xenos::SourceSelect::kAutoIndex, 8, false, false);
  CHECK_FALSE(
      metal::IsAutoLineLoopRoutable(regs, xenos::PrimitiveType::kLineStrip));
}

TEST_CASE("PP3-auto routing: existing PP1 and PP2 predicates are unweakened",
          "[pp3auto][routing]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kLineLoop,
          xenos::SourceSelect::kAutoIndex, 8, false, false);
  CHECK_FALSE(metal::IsPp1Routable(regs, xenos::PrimitiveType::kLineLoop));
  CHECK_FALSE(metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kLineLoop));

  SetDraw(regs, xenos::PrimitiveType::kTriangleFan,
          xenos::SourceSelect::kAutoIndex, 8, false, false);
  CHECK(metal::IsAutoFanRoutable(regs, xenos::PrimitiveType::kTriangleFan));
  CHECK_FALSE(
      metal::IsAutoLineLoopRoutable(regs, xenos::PrimitiveType::kTriangleFan));
}

TEST_CASE("PP3-auto routing: local auto draw view must match registers",
          "[pp3auto][routing][view]") {
  RegisterFile regs;
  SetDraw(regs, xenos::PrimitiveType::kLineLoop,
          xenos::SourceSelect::kAutoIndex, 8, false, false);
  CHECK(metal::IsPp1DrawViewConsistent(regs, 8, /*has_index_buffer=*/false, 0,
                                       xenos::IndexFormat::kInt16,
                                       xenos::Endian::kNone, 0));
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(
      regs, 7, /*has_index_buffer=*/false, 0, xenos::IndexFormat::kInt16,
      xenos::Endian::kNone, 0));
  CHECK_FALSE(metal::IsPp1DrawViewConsistent(regs, 8, /*has_index_buffer=*/true,
                                             0x2000, xenos::IndexFormat::kInt16,
                                             xenos::Endian::kNone, 8));
}

TEST_CASE("PP3-auto normalize: real Process closes line loops",
          "[pp3auto][normalize]") {
  Harness harness;
  REQUIRE(harness.ok);
  for (uint32_t count : {0u, 1u, 2u, 3u, 4u, 17u, uint32_t(UINT16_MAX)}) {
    SetDraw(harness.regs, xenos::PrimitiveType::kLineLoop,
            xenos::SourceSelect::kAutoIndex, count, false, false);
    REQUIRE(metal::IsAutoLineLoopRoutable(harness.regs,
                                          xenos::PrimitiveType::kLineLoop));

    PrimitiveProcessor::ProcessingResult result{};
    REQUIRE(harness.processor->Process(result));
    uint32_t expected_count =
        PrimitiveProcessor::GetLineLoopStripIndexCount(count);
    INFO("count " << count);
    CHECK(result.guest_primitive_type == xenos::PrimitiveType::kLineLoop);
    CHECK(result.host_primitive_type == xenos::PrimitiveType::kLineStrip);
    CHECK(result.host_vertex_shader_type ==
          Shader::HostVertexShaderType::kVertex);
    CHECK(result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kNone);
    CHECK_FALSE(result.host_primitive_reset_enabled);
    CHECK(result.host_draw_vertex_count == expected_count);
    CHECK(result.line_loop_closing_index ==
          (expected_count ? expected_count - 1 : 0));
    CHECK(result.host_index_buffer_handle == SIZE_MAX);
    CHECK(harness.shared_memory->upload_calls == 0);
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xe
