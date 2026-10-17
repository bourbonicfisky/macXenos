/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_SHADER_H_
#define XENIA_GPU_METAL_METAL_SHADER_H_

#include <atomic>
#include <cstdint>
#include <vector>

#include "xenia/gpu/dxbc_shader.h"
#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// A guest shader on the Metal backend. Reuses the DXBC front half (the Xbox-360
// microcode -> DXBC translator lives in DxbcShader), then each translation adds
// the DXIL -> metallib -> MTLFunction back half via the Metal Shader Converter
// chain. Mirrors D3D12Shader/DxbcShader; the ObjC Metal objects are opaque
// pointers so this header stays plain C++.
class MetalShader : public DxbcShader {
 public:
  class MetalTranslation : public DxbcTranslation {
   public:
    MetalTranslation(MetalShader& shader, uint64_t modification)
        : DxbcTranslation(shader, modification) {}
    ~MetalTranslation();

    // Background-thread translation claim (mirrors D3D12Translation).
    bool TryClaimTranslation() {
      return !translation_claimed_.test_and_set(std::memory_order_acq_rel);
    }

    // id<MTLLibrary> / id<MTLFunction>, retained; set once the metallib has been
    // produced and loaded onto the device.
    void* metal_library() const { return metal_library_; }
    void* metal_function() const { return metal_function_; }
    bool has_metal_function() const { return metal_function_ != nullptr; }

    // Builds the MTLLibrary + entry MTLFunction from a metallib blob on the given
    // id<MTLDevice>. Returns false on failure. Takes ownership of nothing; the
    // caller supplies the metallib bytes.
    bool CreateMetalFunction(void* device, const std::vector<uint8_t>& metallib);

   private:
    std::atomic_flag translation_claimed_ = ATOMIC_FLAG_INIT;
    void* metal_library_ = nullptr;   // id<MTLLibrary> (retained)
    void* metal_function_ = nullptr;  // id<MTLFunction> (retained)
  };

  MetalShader(xenos::ShaderType shader_type, uint64_t ucode_data_hash,
              const uint32_t* ucode_dwords, size_t ucode_dword_count,
              std::endian ucode_source_endian = std::endian::big);

 public:
  // Fetch-plain patterns (see Shader::GetFetchPlainPattern). Returns the
  // 1-based index of `mask`, adding it while the table has room (15), else
  // 0 (full decode). Patterns are stable per material, so a shader sees few.
  uint32_t FindOrAddFetchPlainPattern(uint32_t mask) {
    for (size_t i = 0; i < fetch_plain_patterns_.size(); ++i) {
      if (fetch_plain_patterns_[i] == mask) {
        return uint32_t(i + 1);
      }
    }
    if (fetch_plain_patterns_.size() >= 15) {
      return 0;
    }
    fetch_plain_patterns_.push_back(mask);
    return uint32_t(fetch_plain_patterns_.size());
  }
  uint32_t GetFetchPlainPattern(uint32_t pattern_index) const override {
    return pattern_index && pattern_index <= fetch_plain_patterns_.size()
               ? fetch_plain_patterns_[pattern_index - 1]
               : 0u;
  }

 protected:
  Translation* CreateTranslationInstance(uint64_t modification) override;

 private:
  std::vector<uint32_t> fetch_plain_patterns_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_SHADER_H_
