/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// L0 (device-free) conformance for tessellated triangle strips and fans
// (Canary 3eab2b8b3, 2026-10-26). 4D53085B (Halo: Reach) draws tessellated
// triangle strips; the shared PrimitiveProcessor used to reject them, so the
// Metal host tessellation route skipped 12-18 draws a frame
// (MetalDrawSkip[tessellation-unsupported-shape] tess-process). The processor
// now normalizes strips and fans to triangle-list patch streams:
//
//   - auto-indexed strips -> a runtime 16-bit kHostConverted list;
//   - auto-indexed fans   -> the built-in fan table when the backend
//     provisions it (Metal does), else a runtime kHostConverted list;
//   - guest-DMA strips    -> kHostConverted through the shared range
//     conversion, per reset-separated strip, any index width / endian.
//
// The cases run the REAL metal::MetalPrimitiveProcessor over crafted guest
// index memory and compare the host control-point stream the Metal
// tessellation route stages (metal::SwapConvertedIndicesInto of the pool
// slice, or CopyBuiltinIndexData of the fan table) against analytic oracles.
// Adaptive tessellation of strips stays rejected (its index buffer holds edge
// factors, not indices).

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
#include "xenia/gpu/shader.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace {

class TessNullSharedMemory final : public SharedMemory {
 public:
  explicit TessNullSharedMemory(Memory& memory) : SharedMemory(memory) {}
  bool Init() { return InitializeCommon(); }
  bool UploadRanges(const std::pair<uint32_t, uint32_t>*, uint32_t) override {
    return true;
  }
};

void SetTessDraw(RegisterFile& regs, xenos::PrimitiveType prim_type,
                 xenos::SourceSelect source_select,
                 xenos::IndexFormat index_format, xenos::Endian endian,
                 uint32_t count, uint32_t base, xenos::TessellationMode mode,
                 bool multi_prim_ib = false, uint32_t reset_index = 0) {
  std::memset(regs.values, 0, sizeof(regs.values));
  reg::VGT_DRAW_INITIATOR di;
  di.value = 0;
  di.prim_type = prim_type;
  di.source_select = source_select;
  di.index_size = index_format;
  di.num_indices = count;
  di.major_mode = xenos::MajorMode::kExplicit;
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
  op.path_select = xenos::VGTOutputPath::kTessellationEnable;
  regs.values[reg::VGT_OUTPUT_PATH_CNTL::register_index] = op.value;
  reg::VGT_HOS_CNTL hos;
  hos.value = 0;
  hos.tess_mode = mode;
  regs.values[reg::VGT_HOS_CNTL::register_index] = hos.value;
}

struct TessHarness {
  Memory* memory = nullptr;
  RegisterFile regs;
  TraceWriter trace{nullptr};
  std::unique_ptr<TessNullSharedMemory> sm;
  std::unique_ptr<metal::MetalPrimitiveProcessor> pp;
  bool ok = false;
  TessHarness() {
    std::memset(regs.values, 0, sizeof(regs.values));
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    sm = std::make_unique<TessNullSharedMemory>(*memory);
    if (!sm->Init()) {
      return;
    }
    pp = std::make_unique<metal::MetalPrimitiveProcessor>(regs, *memory, trace,
                                                          *sm);
    if (!pp->Initialize(true)) {
      return;
    }
    ok = true;
  }
  ~TessHarness() {
    if (pp) {
      pp->EndFrame();
    }
  }
  template <typename T>
  void Write(uint32_t base, const std::vector<T>& host_values,
             xenos::Endian endian) {
    T* dst = memory->TranslatePhysical<T*>(base);
    REQUIRE(dst != nullptr);
    for (size_t i = 0; i < host_values.size(); ++i) {
      dst[i] = xenos::GpuSwap(host_values[i], endian);
    }
  }
  // The host control-point stream the Metal tessellation route stages.
  template <typename T>
  std::vector<T> Stream(const PrimitiveProcessor::ProcessingResult& r) {
    std::vector<T> host(r.host_draw_vertex_count);
    const size_t bytes = host.size() * sizeof(T);
    if (r.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto) {
      REQUIRE(pp->CopyBuiltinIndexData(r.host_index_buffer_handle, bytes,
                                       host.data()));
      return host;
    }
    REQUIRE(r.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    const void* converted =
        pp->GetConvertedIndexData(r.host_index_buffer_handle, bytes);
    REQUIRE(converted != nullptr);
    metal::SwapConvertedIndicesInto(host.data(), converted,
                                    r.host_draw_vertex_count,
                                    r.host_index_format,
                                    r.host_shader_index_endian);
    return host;
  }
};

// Direct3D strip order: every odd triangle swaps its first two vertices so all
// triangles keep the winding of the first.
template <typename T>
std::vector<T> StripToListOracle(const std::vector<T>& strip) {
  std::vector<T> out;
  for (size_t i = 2; i < strip.size(); ++i) {
    if (i & 1) {
      out.insert(out.end(), {strip[i - 1], strip[i - 2], strip[i]});
    } else {
      out.insert(out.end(), {strip[i - 2], strip[i - 1], strip[i]});
    }
  }
  return out;
}
template <typename T>
std::vector<T> CenterFirstFanOracle(const std::vector<T>& fan) {
  std::vector<T> out;
  for (size_t i = 2; i < fan.size(); ++i) {
    out.insert(out.end(), {fan[0], fan[i - 1], fan[i]});
  }
  return out;
}
template <typename T>
std::vector<T> SplitStripOracle(const std::vector<T>& guest, T reset) {
  std::vector<T> out, segment;
  auto flush = [&] {
    std::vector<T> c = StripToListOracle(segment);
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
template <typename T>
std::vector<T> Iota(uint32_t n) {
  std::vector<T> v(n);
  for (uint32_t i = 0; i < n; ++i) {
    v[i] = T(i);
  }
  return v;
}

const uint32_t kTessBase = 0x9000;

TEST_CASE("Tessellated strip oracle pins the Direct3D winding",
          "[tess_strip][oracle]") {
  std::vector<uint16_t> strip = {10, 11, 12, 13, 14};
  std::vector<uint16_t> expected = {10, 11, 12, 12, 11, 13, 12, 13, 14};
  CHECK(StripToListOracle(strip) == expected);
  CHECK(PrimitiveProcessor::GetTriangleStripListIndexCount(0) == 0);
  CHECK(PrimitiveProcessor::GetTriangleStripListIndexCount(2) == 0);
  CHECK(PrimitiveProcessor::GetTriangleStripListIndexCount(3) == 3);
  CHECK(PrimitiveProcessor::GetTriangleStripListIndexCount(5) == 9);
  // The shared template and the auto-index writer agree with the oracle.
  std::vector<uint16_t> templ(9);
  PrimitiveProcessor::TriangleStripToList(
      templ.data(), strip.data(), uint32_t(strip.size()),
      [](uint16_t index) { return index; });
  CHECK(templ == expected);
  std::vector<uint16_t> auto_list(PrimitiveProcessor::GetTriangleStripListIndexCount(7));
  PrimitiveProcessor::AutoTriangleStripToList(auto_list.data(), 7);
  CHECK(auto_list == StripToListOracle(Iota<uint16_t>(7)));
}

TEST_CASE("Tessellated auto-indexed strip -> host-converted triangle-list "
          "patches, discrete and continuous",
          "[tess_strip][normalize]") {
  TessHarness h;
  REQUIRE(h.ok);
  for (xenos::TessellationMode mode : {xenos::TessellationMode::kDiscrete,
                                       xenos::TessellationMode::kContinuous}) {
    for (uint32_t n : {3u, 4u, 9u, 4909u}) {
      SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
                  xenos::SourceSelect::kAutoIndex, xenos::IndexFormat::kInt16,
                  xenos::Endian::kNone, n, 0, mode);
      PrimitiveProcessor::ProcessingResult r{};
      INFO("mode " << uint32_t(mode) << " n " << n);
      REQUIRE(h.pp->Process(r));
      CHECK(r.IsTessellated());
      CHECK(r.host_vertex_shader_type ==
            Shader::HostVertexShaderType::kTriangleDomainCPIndexed);
      CHECK(r.host_primitive_type == xenos::PrimitiveType::kTriangleList);
      CHECK(r.index_buffer_type ==
            PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
      CHECK(r.host_index_format == xenos::IndexFormat::kInt16);
      CHECK(r.host_shader_index_endian == xenos::Endian::kNone);
      CHECK(r.host_draw_vertex_count ==
            PrimitiveProcessor::GetTriangleStripListIndexCount(n));
      CHECK(h.Stream<uint16_t>(r) == StripToListOracle(Iota<uint16_t>(n)));
      h.pp->EndFrame();
    }
  }
}

TEST_CASE("Tessellated degenerate auto strip draws nothing",
          "[tess_strip][normalize]") {
  TessHarness h;
  REQUIRE(h.ok);
  SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
              xenos::SourceSelect::kAutoIndex, xenos::IndexFormat::kInt16,
              xenos::Endian::kNone, 2, 0, xenos::TessellationMode::kDiscrete);
  PrimitiveProcessor::ProcessingResult r{};
  REQUIRE(h.pp->Process(r));
  CHECK(r.host_draw_vertex_count == 0);
}

TEST_CASE("Tessellated auto-indexed fan -> built-in fan table (Metal "
          "provisions it)",
          "[tess_strip][normalize]") {
  TessHarness h;
  REQUIRE(h.ok);
  const uint32_t n = 8;
  SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleFan,
              xenos::SourceSelect::kAutoIndex, xenos::IndexFormat::kInt16,
              xenos::Endian::kNone, n, 0, xenos::TessellationMode::kContinuous);
  PrimitiveProcessor::ProcessingResult r{};
  REQUIRE(h.pp->Process(r));
  CHECK(r.IsTessellated());
  CHECK(r.host_primitive_type == xenos::PrimitiveType::kTriangleList);
  CHECK(r.index_buffer_type ==
        PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto);
  CHECK(r.host_draw_vertex_count ==
        PrimitiveProcessor::GetTriangleFanListIndexCount(n));
  // Metal's fan table is center-first (the PP2-auto contract).
  CHECK(h.Stream<uint16_t>(r) == CenterFirstFanOracle(Iota<uint16_t>(n)));
}

TEST_CASE("Tessellated DMA strip, 16-bit both endians, with and without "
          "primitive reset",
          "[tess_strip][normalize]") {
  TessHarness h;
  REQUIRE(h.ok);
  for (xenos::Endian endian : {xenos::Endian::kNone, xenos::Endian::k8in16}) {
    INFO("endian " << uint32_t(endian));
    std::vector<uint16_t> strip = {40, 41, 42, 43, 44, 45, 46};
    h.Write(kTessBase, strip, endian);
    SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
                xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16, endian,
                uint32_t(strip.size()), kTessBase,
                xenos::TessellationMode::kDiscrete);
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    CHECK(r.IsTessellated());
    CHECK(r.host_primitive_type == xenos::PrimitiveType::kTriangleList);
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(r.host_index_format == xenos::IndexFormat::kInt16);
    CHECK(r.host_primitive_reset_enabled == false);
    CHECK(r.host_draw_vertex_count ==
          PrimitiveProcessor::GetTriangleStripListIndexCount(
              uint32_t(strip.size())));
    CHECK(h.Stream<uint16_t>(r) == StripToListOracle(strip));
    h.pp->EndFrame();

    // Primitive reset separates strips; each restarts its winding parity.
    const uint16_t reset = 0xFFFF;
    std::vector<uint16_t> strips = {1, 2, 3, 4, reset, 7, 8, 9,
                                    10, 11, reset, 20, 21};
    h.Write(kTessBase, strips, endian);
    SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
                xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt16, endian,
                uint32_t(strips.size()), kTessBase,
                xenos::TessellationMode::kContinuous, /*multi_prim_ib=*/true,
                reset);
    PrimitiveProcessor::ProcessingResult rr{};
    REQUIRE(h.pp->Process(rr));
    std::vector<uint16_t> expected = SplitStripOracle<uint16_t>(strips, reset);
    // (1,2,3),(3,2,4),(7,8,9),(9,8,10),(9,10,11); the 2-index tail is empty.
    CHECK(expected.size() == 15);
    CHECK(rr.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(rr.host_draw_vertex_count == expected.size());
    CHECK(h.Stream<uint16_t>(rr) == expected);
    h.pp->EndFrame();
  }
}

TEST_CASE("Tessellated DMA strip, 32-bit every endian", "[tess_strip][normalize]") {
  TessHarness h;
  REQUIRE(h.ok);
  for (xenos::Endian endian :
       {xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
        xenos::Endian::k16in32}) {
    INFO("endian " << uint32_t(endian));
    std::vector<uint32_t> strip = {0x10000, 0x10001, 0x20002, 5, 0x30004, 6};
    h.Write(kTessBase, strip, endian);
    SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
                xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt32, endian,
                uint32_t(strip.size()), kTessBase,
                xenos::TessellationMode::kDiscrete);
    PrimitiveProcessor::ProcessingResult r{};
    REQUIRE(h.pp->Process(r));
    CHECK(r.IsTessellated());
    CHECK(r.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted);
    CHECK(r.host_index_format == xenos::IndexFormat::kInt32);
    CHECK(h.Stream<uint32_t>(r) == StripToListOracle(strip));
    h.pp->EndFrame();
  }
}

TEST_CASE("Tessellated adaptive strips stay rejected", "[tess_strip][reject]") {
  TessHarness h;
  REQUIRE(h.ok);
  std::vector<uint32_t> factors = {0x3F800000u, 0x40000000u, 0x40400000u};
  h.Write(kTessBase, factors, xenos::Endian::kNone);
  SetTessDraw(h.regs, xenos::PrimitiveType::kTriangleStrip,
              xenos::SourceSelect::kDMA, xenos::IndexFormat::kInt32,
              xenos::Endian::kNone, uint32_t(factors.size()), kTessBase,
              xenos::TessellationMode::kAdaptive);
  PrimitiveProcessor::ProcessingResult r{};
  CHECK_FALSE(h.pp->Process(r));
}

}  // namespace
}  // namespace gpu
}  // namespace xe
