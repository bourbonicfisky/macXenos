/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// implement-metal-depth-semantics task 1.1: the decisive probe for design
// decision 3 — do the float24 depth-stencil modification modes survive OUR
// shader chain (ucode -> DxbcShaderTranslator -> dxilconv -> Metal Shader
// Converter), and does the truncating mode's SV_DepthLessEqual output reach
// the metallib as a conservative-depth qualifier?
//
// Structure mirrors integer_scale_translation_case.cc: a real guest pixel
// shader is assembled from microcode (same proven tfetch fixture — the shader
// BODY is irrelevant here; the float24 conversion is a translator epilogue
// keyed off the modification alone), translated under each DepthStencilMode,
// and the emitted DXBC's output signature is checked for the depth semantic
// each mode contracts to emit:
//   kNoModifiers        -> no depth output at all (shader writes no oDepth)
//   kFloat24Truncating  -> SV_DepthLessEqual (early-Z-preserving form)
//   kFloat24Rounding    -> SV_Depth
// Then, on the device, both modes run the production DXBC->DXIL->metallib
// back half and must yield a loadable MTLFunction. The AIR depth-qualifier
// survival scan is recorded as INFO, not asserted: whether MSC downgrades
// depth(less_equal) to depth(any) decides the truncating mode's early-Z value
// (stage A measurement), not this probe's pass/fail.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#import <Metal/Metal.h>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/dxbc_to_dxil_converter.h"
#include "xenia/gpu/metal/metal_shader_converter.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

using DepthStencilMode = DxbcShaderTranslator::Modification::DepthStencilMode;

// Proven guest-PS fixture from integer_scale_translation_case.cc (see that
// file for the ucode layout walkthrough). Fetch constant 0 — slot identity is
// irrelevant to the depth epilogue under test.
std::vector<uint32_t> BuildFixturePixelUcode() {
  const uint32_t kExecAddress = 1;
  const uint32_t kExecCount = 1;
  const uint32_t kSequenceFetch = 0b01;
  uint32_t cf0_dword0 = (kExecAddress & 0xFFF) | ((kExecCount & 0x7) << 12) |
                        ((kSequenceFetch & 0xFFF) << 16);
  uint32_t cf0_dword1 =
      (uint32_t(ucode::ControlFlowOpcode::kExecEnd) & 0xF) << 12;
  const uint32_t kDstSwizzleIdentity =
      (0u << 0) | (1u << 3) | (2u << 6) | (3u << 9);
  uint32_t fetch_dword0 = (uint32_t(ucode::FetchOpcode::kTextureFetch) & 0x1F) |
                          ((0u & 0x3F) << 5) | ((1u & 0x3F) << 12) |
                          ((0u & 0x1F) << 20) | ((0b010001u & 0x3F) << 26);
  uint32_t fetch_dword1 = kDstSwizzleIdentity & 0xFFF;
  uint32_t fetch_dword2 = (uint32_t(xenos::FetchOpDimension::k2D) & 0x3) << 14;
  return {cf0_dword0, cf0_dword1, 0u, fetch_dword0, fetch_dword1, fetch_dword2};
}

// Translates the fixture PS under the given depth-stencil mode and returns the
// full DXBC container bytes.
std::vector<uint8_t> TranslateWithDepthMode(DepthStencilMode mode) {
  std::vector<uint32_t> ucode = BuildFixturePixelUcode();
  Shader shader(xenos::ShaderType::kPixel, 0, ucode.data(), ucode.size(),
                std::endian::little);
  StringBuffer disasm;
  shader.AnalyzeUcode(disasm);

  DxbcShaderTranslator translator(ui::GraphicsProvider::GpuVendorID::kApple,
                                  /*bindless_resources_used=*/true,
                                  /*edram_rov_used=*/false,
                                  /*gamma_render_target_as_unorm8=*/true);
  DxbcShaderTranslator::Modification modification(
      translator.GetDefaultPixelShaderModification(
          shader.GetDynamicAddressableRegisterCount(
              xenos::kMaxShaderTempRegisters)));
  modification.pixel.depth_stencil_mode = mode;
  Shader::Translation* translation =
      shader.GetOrCreateTranslation(modification.value);
  REQUIRE(translator.TranslateAnalyzedShader(*translation));
  return translation->translated_binary();
}

bool BytesContain(const std::vector<uint8_t>& haystack, const char* needle) {
  size_t needle_length = std::strlen(needle);
  if (haystack.size() < needle_length) {
    return false;
  }
  for (size_t i = 0; i + needle_length <= haystack.size(); ++i) {
    if (std::memcmp(haystack.data() + i, needle, needle_length) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

TEST_CASE("float24 modes emit the contracted depth output semantic",
          "[float24][translation]") {
  // Control: without a mode, this shader has no depth output at all — the
  // fixture writes no oDepth. Signature semantic names are ASCII in the
  // container, so a byte scan is a faithful signature check.
  std::vector<uint8_t> plain = TranslateWithDepthMode(
      DepthStencilMode::kNoModifiers);
  CHECK_FALSE(BytesContain(plain, "SV_Depth"));

  std::vector<uint8_t> truncating =
      TranslateWithDepthMode(DepthStencilMode::kFloat24Truncating);
  CHECK(BytesContain(truncating, "SV_DepthLessEqual"));

  std::vector<uint8_t> rounding =
      TranslateWithDepthMode(DepthStencilMode::kFloat24Rounding);
  CHECK(BytesContain(rounding, "SV_Depth"));
  CHECK_FALSE(BytesContain(rounding, "SV_DepthLessEqual"));
}

TEST_CASE("float24 modes survive dxilconv and Metal Shader Converter",
          "[float24][device]") {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  REQUIRE(device != nil);

  for (DepthStencilMode mode : {DepthStencilMode::kFloat24Truncating,
                                DepthStencilMode::kFloat24Rounding}) {
    const bool is_truncating = mode == DepthStencilMode::kFloat24Truncating;
    INFO((is_truncating ? "kFloat24Truncating" : "kFloat24Rounding"));

    std::vector<uint8_t> dxbc = TranslateWithDepthMode(mode);

    std::string error;
    metal::DxbcToDxilConverter dxil_converter;
    REQUIRE(dxil_converter.Initialize(&error));
    std::vector<uint8_t> dxil;
    {
      INFO(error);
      REQUIRE(dxil_converter.Convert(dxbc, dxil, &error));
    }

    metal::MetalShaderConverter metal_converter;
    REQUIRE(metal_converter.Initialize(/*bindless_resources_used=*/true,
                                       &error));
    std::vector<uint8_t> metallib;
    {
      INFO(error);
      REQUIRE(metal_converter.Convert(
          metal::MetalShaderConverter::Stage::kFragment, dxil, metallib,
          &error));
    }
    REQUIRE(metallib.size() >= 4);
    REQUIRE(std::memcmp(metallib.data(), "MTLB", 4) == 0);

    dispatch_data_t data = dispatch_data_create(
        metallib.data(), metallib.size(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* ns_error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data
                                                  error:&ns_error];
    {
      std::string library_error =
          ns_error ? ns_error.localizedDescription.UTF8String : "(no error)";
      INFO(library_error);
      REQUIRE(library != nil);
    }
    NSArray<NSString*>* names = [library functionNames];
    REQUIRE(names.count > 0);
    id<MTLFunction> function =
        [library newFunctionWithName:[names firstObject]];
    REQUIRE(function != nil);
    CHECK(function.functionType == MTLFunctionTypeFragment);

    // Best-effort qualifier-survival evidence for the capability ledger: AIR
    // metadata strings are not guaranteed to survive bitcode encoding, so
    // absence proves nothing — only presence is signal. Never asserted.
    if (is_truncating) {
      INFO("truncating metallib 'less_equal' byte scan: "
           << (BytesContain(metallib, "less_equal") ? "FOUND" : "not found"));
      CHECK(true);
    }
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
