/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// PP4 Phase-0 L0 oracle for the REAL shared PrimitiveProcessor configured as
// an always-on-fixed-restart host. This is deliberately observational: unsafe
// current kGuestDMA outcomes are asserted as such so later PP4 corrections are
// explicit changes rather than accidental baseline drift.

#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/conformance/test_shared_physical_memory.h"
#include "xenia/gpu/primitive_processor.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace {

class PrimitiveResetOracleSharedMemory final : public SharedMemory {
 public:
  explicit PrimitiveResetOracleSharedMemory(Memory& memory)
      : SharedMemory(memory) {}
  bool Initialize() { return InitializeCommon(); }
  bool UploadRanges(const std::pair<uint32_t, uint32_t>*, uint32_t) override {
    return true;
  }
};

class PrimitiveResetOracleProcessor final : public PrimitiveProcessor {
 public:
  PrimitiveResetOracleProcessor(const RegisterFile& register_file,
                                Memory& memory, TraceWriter& trace_writer,
                                SharedMemory& shared_memory)
      : PrimitiveProcessor(register_file, memory, trace_writer, shared_memory) {
  }

  bool Initialize() {
    return InitializeCommon(
        /*full_32bit_vertex_indices_supported=*/true,
        /*triangle_fans_supported=*/true,
        /*line_loops_supported=*/true,
        /*quad_lists_supported=*/true,
        /*point_sprites_supported_without_vs_expansion=*/true,
        /*rectangle_lists_supported_without_vs_expansion=*/true,
        /*host_primitive_reset_always_enabled=*/true);
  }

  std::vector<uint8_t> CopyAllocation(size_t handle, size_t byte_count) const {
    if (handle >= allocations_.size() ||
        byte_count > allocations_[handle].data_size) {
      return {};
    }
    const Allocation& allocation = allocations_[handle];
    const uint8_t* data = allocation.bytes.data() + allocation.data_offset;
    return std::vector<uint8_t>(data, data + byte_count);
  }

 protected:
  bool InitializeBuiltinIndexBuffer(size_t,
                                    std::function<void(void*)>) override {
    // Every primitive class used here is declared native, so no builtin table
    // may be requested.
    return false;
  }

  void* RequestHostConvertedIndexBufferForCurrentFrame(
      xenos::IndexFormat format, uint32_t index_count, bool coalign_for_simd,
      uint32_t coalignment_original_address,
      size_t& backend_handle_out) override {
    const size_t index_size = format == xenos::IndexFormat::kInt16
                                  ? sizeof(uint16_t)
                                  : sizeof(uint32_t);
    Allocation allocation;
    allocation.data_size = index_size * index_count;
    allocation.bytes.resize(allocation.data_size +
                            XE_GPU_PRIMITIVE_PROCESSOR_SIMD_SIZE);
    if (coalign_for_simd) {
      allocation.data_offset = size_t(GetSimdCoalignmentOffset(
          allocation.bytes.data(), coalignment_original_address));
    }
    allocations_.push_back(std::move(allocation));
    backend_handle_out = allocations_.size() - 1;
    Allocation& stored = allocations_.back();
    return stored.bytes.data() + stored.data_offset;
  }

  bool RequestGuestIndexSharedMemoryRange(uint32_t, uint32_t,
                                          ProcessedIndexBufferType) override {
    // The oracle examines Process normalization only. Like production Metal,
    // it does not bind the shared-memory buffer directly.
    return true;
  }

 private:
  struct Allocation {
    std::vector<uint8_t> bytes;
    size_t data_offset = 0;
    size_t data_size = 0;
  };
  std::deque<Allocation> allocations_;
};

struct PrimitiveResetOracleHarness {
  Memory* memory = nullptr;
  RegisterFile registers;
  TraceWriter trace_writer{nullptr};
  std::unique_ptr<PrimitiveResetOracleSharedMemory> shared_memory;
  std::unique_ptr<PrimitiveResetOracleProcessor> processor;
  bool initialized = false;

  PrimitiveResetOracleHarness() {
    std::memset(registers.values, 0, sizeof(registers.values));
    memory = conformance::TestPhysicalMemory();
    if (!memory) {
      return;
    }
    shared_memory = std::make_unique<PrimitiveResetOracleSharedMemory>(*memory);
    if (!shared_memory->Initialize()) {
      return;
    }
    processor = std::make_unique<PrimitiveResetOracleProcessor>(
        registers, *memory, trace_writer, *shared_memory);
    initialized = processor->Initialize();
  }
};

xenos::Endian NormalizeEndian(xenos::IndexFormat format, xenos::Endian endian) {
  if (format != xenos::IndexFormat::kInt16) {
    return endian;
  }
  if (endian == xenos::Endian::k8in32) {
    return xenos::Endian::k8in16;
  }
  if (endian == xenos::Endian::k16in32) {
    return xenos::Endian::kNone;
  }
  return endian;
}

bool IsStrip(xenos::PrimitiveType primitive_type) {
  return primitive_type == xenos::PrimitiveType::kLineStrip ||
         primitive_type == xenos::PrimitiveType::kTriangleStrip;
}

void SetPrimitiveResetDraw(RegisterFile& registers,
                           xenos::PrimitiveType primitive_type,
                           xenos::IndexFormat format, xenos::Endian endian,
                           uint32_t count, uint32_t base, bool reset_enabled,
                           uint32_t reset_index) {
  std::memset(registers.values, 0, sizeof(registers.values));

  reg::VGT_DRAW_INITIATOR draw;
  draw.value = 0;
  draw.prim_type = primitive_type;
  draw.source_select = xenos::SourceSelect::kDMA;
  draw.index_size = format;
  draw.num_indices = count;
  draw.major_mode = xenos::MajorMode::kImplicit;
  registers.values[reg::VGT_DRAW_INITIATOR::register_index] = draw.value;

  reg::VGT_DMA_SIZE dma_size;
  dma_size.value = 0;
  dma_size.num_words = count;
  dma_size.swap_mode = endian;
  registers.values[reg::VGT_DMA_SIZE::register_index] = dma_size.value;
  registers.values[XE_GPU_REG_VGT_DMA_BASE] = base;

  reg::PA_SU_SC_MODE_CNTL mode;
  mode.value = 0;
  mode.multi_prim_ib_ena = reset_enabled ? 1 : 0;
  registers.values[reg::PA_SU_SC_MODE_CNTL::register_index] = mode.value;

  reg::VGT_MULTI_PRIM_IB_RESET_INDX reset;
  reset.value = 0;
  reset.reset_indx = reset_index;
  registers.values[reg::VGT_MULTI_PRIM_IB_RESET_INDX::register_index] =
      reset.value;

  reg::VGT_OUTPUT_PATH_CNTL output_path;
  output_path.value = 0;
  output_path.path_select = xenos::VGTOutputPath::kVertexReuse;
  registers.values[reg::VGT_OUTPUT_PATH_CNTL::register_index] =
      output_path.value;
}

template <typename T>
void WriteGuestWords(Memory& memory, uint32_t base,
                     const std::vector<T>& words) {
  std::memcpy(memory.TranslatePhysical(base), words.data(),
              words.size() * sizeof(T));
}

TEST_CASE("PP4 reset: shared Process current-result cross-product",
          "[pp4][primitive-reset][phase0]") {
  PrimitiveResetOracleHarness harness;
  REQUIRE(harness.initialized);

  constexpr std::array<xenos::PrimitiveType, 4> kPrimitiveTypes = {
      xenos::PrimitiveType::kLineList, xenos::PrimitiveType::kLineStrip,
      xenos::PrimitiveType::kTriangleList,
      xenos::PrimitiveType::kTriangleStrip};
  constexpr std::array<xenos::IndexFormat, 2> kFormats = {
      xenos::IndexFormat::kInt16, xenos::IndexFormat::kInt32};
  constexpr std::array<xenos::Endian, 4> kEndians = {
      xenos::Endian::kNone, xenos::Endian::k8in16, xenos::Endian::k8in32,
      xenos::Endian::k16in32};

  uint32_t case_index = 0;
  uint32_t protected_real_maximum_cases = 0;
  uint32_t unprotected_real_maximum_cases = 0;
  for (xenos::PrimitiveType primitive_type : kPrimitiveTypes) {
    for (xenos::IndexFormat format : kFormats) {
      for (xenos::Endian requested_endian : kEndians) {
        for (bool reset_enabled : {false, true}) {
          for (bool fixed_reset : {false, true}) {
            for (bool configured_reset_present : {false, true}) {
              for (bool host_maximum_present : {false, true}) {
                const xenos::Endian normalized_endian =
                    NormalizeEndian(format, requested_endian);
                const uint32_t reset_index =
                    format == xenos::IndexFormat::kInt16
                        ? (fixed_reset ? UINT16_MAX : 0x1234u)
                        : (fixed_reset ? xenos::kVertexIndexMask : 0x123456u);
                const uint32_t base = 0x01000000u + case_index * 0x100u;
                ++case_index;
                constexpr uint32_t kCount = 8;

                bool reset_value_is_used = false;
                bool raw_host_maximum_is_present = false;
                bool real_host_maximum_is_present = false;
                if (format == xenos::IndexFormat::kInt16) {
                  std::vector<uint16_t> words = {
                      xenos::GpuSwap(uint16_t(1), normalized_endian),
                      xenos::GpuSwap(uint16_t(2), normalized_endian),
                      xenos::GpuSwap(uint16_t(3), normalized_endian),
                      xenos::GpuSwap(uint16_t(4), normalized_endian),
                      xenos::GpuSwap(uint16_t(5), normalized_endian),
                      xenos::GpuSwap(uint16_t(6), normalized_endian),
                      xenos::GpuSwap(uint16_t(7), normalized_endian),
                      xenos::GpuSwap(uint16_t(8), normalized_endian)};
                  const uint16_t reset_word =
                      xenos::GpuSwap(uint16_t(reset_index), normalized_endian);
                  if (configured_reset_present) {
                    words[2] = reset_word;
                  }
                  if (host_maximum_present) {
                    words[5] = UINT16_MAX;
                  }
                  for (uint16_t word : words) {
                    reset_value_is_used |= word == reset_word;
                    raw_host_maximum_is_present |= word == UINT16_MAX;
                  }
                  real_host_maximum_is_present =
                      raw_host_maximum_is_present &&
                      !(IsStrip(primitive_type) && reset_enabled &&
                        reset_word == UINT16_MAX);
                  WriteGuestWords(*harness.memory, base, words);
                } else {
                  std::vector<uint32_t> words = {
                      xenos::GpuSwap(1u, normalized_endian),
                      xenos::GpuSwap(2u, normalized_endian),
                      xenos::GpuSwap(3u, normalized_endian),
                      xenos::GpuSwap(4u, normalized_endian),
                      xenos::GpuSwap(5u, normalized_endian),
                      xenos::GpuSwap(6u, normalized_endian),
                      xenos::GpuSwap(7u, normalized_endian),
                      xenos::GpuSwap(8u, normalized_endian)};
                  const uint32_t reset_word =
                      xenos::GpuSwap(reset_index, normalized_endian);
                  const uint32_t reset_mask = xenos::GpuSwap(
                      xenos::kVertexIndexMask, normalized_endian);
                  if (configured_reset_present) {
                    words[2] = reset_word;
                  }
                  if (host_maximum_present) {
                    words[5] = UINT32_MAX;
                  }
                  for (uint32_t word : words) {
                    reset_value_is_used |= (word & reset_mask) == reset_word;
                    raw_host_maximum_is_present |= word == UINT32_MAX;
                  }
                  real_host_maximum_is_present =
                      raw_host_maximum_is_present &&
                      !(IsStrip(primitive_type) && reset_enabled &&
                        ((UINT32_MAX & reset_mask) == reset_word));
                  WriteGuestWords(*harness.memory, base, words);
                }

                SetPrimitiveResetDraw(harness.registers, primitive_type, format,
                                      requested_endian, kCount, base,
                                      reset_enabled, reset_index);
                PrimitiveProcessor::ProcessingResult result{};
                REQUIRE(harness.processor->Process(result));

                const bool guest_reset_is_effective =
                    IsStrip(primitive_type) && reset_enabled;
                PrimitiveProcessor::ProcessedIndexBufferType
                    expected_buffer_type =
                        PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA;
                xenos::IndexFormat expected_host_format = format;
                if (guest_reset_is_effective) {
                  if (format == xenos::IndexFormat::kInt16) {
                    const uint16_t reset_word = xenos::GpuSwap(
                        uint16_t(reset_index), normalized_endian);
                    if (reset_word != UINT16_MAX && reset_value_is_used) {
                      expected_buffer_type = PrimitiveProcessor::
                          ProcessedIndexBufferType::kHostConverted;
                      if (raw_host_maximum_is_present) {
                        expected_host_format = xenos::IndexFormat::kInt32;
                      }
                    }
                  } else if (reset_value_is_used) {
                    expected_buffer_type = PrimitiveProcessor::
                        ProcessedIndexBufferType::kHostConverted;
                  }
                } else if (format == xenos::IndexFormat::kInt16 &&
                           raw_host_maximum_is_present) {
                  expected_buffer_type = PrimitiveProcessor::
                      ProcessedIndexBufferType::kHostConverted;
                  expected_host_format = xenos::IndexFormat::kInt32;
                }

                INFO("case " << case_index << " prim "
                             << uint32_t(primitive_type) << " fmt "
                             << uint32_t(format) << " endian "
                             << uint32_t(requested_endian) << " reset-enabled "
                             << reset_enabled << " fixed " << fixed_reset
                             << " reset-present " << configured_reset_present
                             << " max-present " << host_maximum_present);
                CHECK(result.guest_primitive_type == primitive_type);
                CHECK(result.host_primitive_type == primitive_type);
                CHECK(result.host_draw_vertex_count == kCount);
                CHECK(result.index_buffer_type == expected_buffer_type);
                CHECK(result.host_index_format == expected_host_format);
                CHECK(result.host_shader_index_endian == normalized_endian);
                CHECK(result.host_primitive_reset_enabled ==
                      guest_reset_is_effective);

                if (real_host_maximum_is_present) {
                  const bool current_result_protects_real_maximum =
                      result.index_buffer_type ==
                          PrimitiveProcessor::ProcessedIndexBufferType::
                              kHostConverted &&
                      (format == xenos::IndexFormat::kInt32 ||
                       result.host_index_format == xenos::IndexFormat::kInt32);
                  if (current_result_protects_real_maximum) {
                    ++protected_real_maximum_cases;
                  } else {
                    ++unprotected_real_maximum_cases;
                    CHECK(result.index_buffer_type ==
                          PrimitiveProcessor::ProcessedIndexBufferType::
                              kGuestDMA);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // The matrix must continue to exercise both the existing safeguards and the
  // known PP4 gaps (reset-disabled/custom-reset-unused raw 32-bit maxima, plus
  // custom-reset-unused 16-bit maxima while reset is enabled).
  CHECK(protected_real_maximum_cases > 0);
  CHECK(unprotected_real_maximum_cases > 0);
}

TEST_CASE("PP4 reset: shared converted bytes expose the corner semantics",
          "[pp4][primitive-reset][phase0][bytes]") {
  PrimitiveResetOracleHarness harness;
  REQUIRE(harness.initialized);

  struct Case {
    uint32_t base;
    bool reset_enabled;
    uint32_t reset_index;
    std::vector<uint16_t> guest;
    PrimitiveProcessor::ProcessedIndexBufferType buffer_type;
    xenos::IndexFormat host_format;
    std::vector<uint32_t> expected_32;
  };
  const std::array<Case, 4> cases = {{
      {0x02000000u,
       true,
       0x1234u,
       {1, 0x1234, 2, 3},
       PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted,
       xenos::IndexFormat::kInt16,
       {}},
      {0x02000100u,
       true,
       0x1234u,
       {1, 0x1234, UINT16_MAX, 3},
       PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted,
       xenos::IndexFormat::kInt32,
       {1, UINT32_MAX, UINT16_MAX, 3}},
      {0x02000200u,
       false,
       0x1234u,
       {1, UINT16_MAX, 2, 3},
       PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted,
       xenos::IndexFormat::kInt32,
       {1, UINT16_MAX, 2, 3}},
      // Known gap: configured custom reset is absent, so the real maximum is
      // left in a raw 16-bit kGuestDMA buffer even though Metal consumes it.
      {0x02000300u,
       true,
       0x1234u,
       {1, UINT16_MAX, 2, 3},
       PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA,
       xenos::IndexFormat::kInt16,
       {}},
  }};

  for (const Case& test_case : cases) {
    WriteGuestWords(*harness.memory, test_case.base, test_case.guest);
    SetPrimitiveResetDraw(harness.registers,
                          xenos::PrimitiveType::kTriangleStrip,
                          xenos::IndexFormat::kInt16, xenos::Endian::kNone,
                          uint32_t(test_case.guest.size()), test_case.base,
                          test_case.reset_enabled, test_case.reset_index);
    PrimitiveProcessor::ProcessingResult result{};
    REQUIRE(harness.processor->Process(result));
    INFO("base 0x" << std::hex << test_case.base);
    CHECK(result.index_buffer_type == test_case.buffer_type);
    CHECK(result.host_index_format == test_case.host_format);
    if (result.index_buffer_type !=
        PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted) {
      continue;
    }
    const size_t byte_count =
        test_case.guest.size() *
        (result.host_index_format == xenos::IndexFormat::kInt16
             ? sizeof(uint16_t)
             : sizeof(uint32_t));
    const std::vector<uint8_t> converted = harness.processor->CopyAllocation(
        result.host_index_buffer_handle, byte_count);
    REQUIRE(converted.size() == byte_count);
    if (result.host_index_format == xenos::IndexFormat::kInt16) {
      const std::array<uint16_t, 4> expected = {1, UINT16_MAX, 2, 3};
      CHECK(std::memcmp(converted.data(), expected.data(), byte_count) == 0);
    } else {
      REQUIRE(test_case.expected_32.size() == test_case.guest.size());
      CHECK(std::memcmp(converted.data(), test_case.expected_32.data(),
                        byte_count) == 0);
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xe
