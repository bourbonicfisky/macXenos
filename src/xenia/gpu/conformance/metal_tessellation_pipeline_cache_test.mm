/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// port-xenios-tessellation tasks 3.1/3.2 — the cache-level L1 case.
//
// Where the task-2.2 probe exercised every INGREDIENT standalone, this drives
// the production MetalPipelineCache::GetOrCreateTessellationPipelineState end
// to end on the device:
//
//   1. triangle cell (discrete / kTriangleDomainCPIndexed, real guest PS) and
//      quad cell (discrete / kQuadDomainCPIndexed, NULL PS -> the depth-only
//      fallback) both build; the returned config mirrors the reflection
//      contract (all five zero-checked counts nonzero, triangle geometry
//      primitive);
//   2. the same key is a memo hit (same pointer, no rebuild);
//   3. a mismatched HS/DS pairing (quad HS selected against a triangle-domain
//      guest DS) fails IRRuntimeValidateTessellationPipeline -> construction
//      returns null WITHOUT crashing, and repeats hit the negative cache;
//   4. the durable-storage round-trip: build with storage open (record
//      published), reopen the store standalone and load the record, then
//      rebuild through a FRESH cache over the same directory and prove it was
//      served from the record (nothing newly admitted).

#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/string_buffer.h"
#include "xenia/gpu/dxbc_shader_translator.h"
#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"
#include "xenia/gpu/metal/metal_shader.h"
#include "xenia/gpu/ucode.h"
#include "xenia/gpu/xenos.h"

// IRRuntimePrimitiveType / IRRuntimeTessellatorOutputPrimitive numeric checks.
#include "metal_irconverter_runtime.h"

namespace xe {
namespace gpu {
namespace test {
namespace {

using metal::MetalPipelineCache;
using metal::MetalPipelineRasterKey;
using metal::MetalPipelineStorage;
using metal::MetalShader;
using HostVertexShaderType = Shader::HostVertexShaderType;

// Nondegenerate guest vertex program — the same fixture the translation probe
// and the clip-discriminator test use (y = x * x from the translated index).
const uint32_t kGuestVertexUcode[] = {
    0x00001002, 0x00001000, 0xC2000000, 0x00001003,
    0x00002000, 0x00000000, 0xC8020000, 0x006C6C00,
    0xC1000000, 0xC88F803E, 0x00000000, 0xC2000000};

// Proven guest-PS fixture (metal_float24_translation_probe_test.mm shape).
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

struct AttachmentFixture {
  uint64_t colors[xenos::kMaxColorRenderTargets];
  uint64_t depth;
  uint32_t blends[xenos::kMaxColorRenderTargets];
  uint32_t write_mask;
  MetalPipelineRasterKey raster;
};

AttachmentFixture MakeAttachments() {
  AttachmentFixture fixture = {};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    fixture.colors[i] = uint64_t(MTLPixelFormatInvalid);
    fixture.blends[i] = 0;
  }
  fixture.colors[0] = uint64_t(MTLPixelFormatRGBA8Unorm);
  fixture.depth = uint64_t(MTLPixelFormatDepth32Float_Stencil8);
  fixture.write_mask = 0xF;
  fixture.raster = MetalPipelineRasterKey{/*raster_sample_count=*/1};
  return fixture;
}

// One cache + device + fixture shaders. NOT a global singleton: the storage
// round-trip case needs multiple independent cache lifetimes.
struct CacheHarness {
  id<MTLDevice> device = nil;
  std::unique_ptr<MetalPipelineCache> cache;
  bool ok = false;
  std::string error;

  explicit CacheHarness(id<MTLDevice> existing_device = nil) {
    device = existing_device ? existing_device : MTLCreateSystemDefaultDevice();
    if (!device) {
      error = "no Metal device";
      return;
    }
    cache = std::make_unique<MetalPipelineCache>((__bridge void*)device);
    if (!cache->Initialize()) {
      error = "MetalPipelineCache::Initialize failed";
      return;
    }
    ok = true;
  }

  // Guest VS translation created with the DOMAIN host-VS-type modification —
  // the exact shape group 4 must produce from the ProcessingResult. NOT run
  // through TranslateShader (that compiles Stage::kVertex).
  MetalShader::MetalTranslation* MakeDomainTranslation(
      HostVertexShaderType host_vertex_shader_type) {
    MetalShader* shader = cache->LoadShader(
        xenos::ShaderType::kVertex, kGuestVertexUcode,
        uint32_t(sizeof(kGuestVertexUcode) / sizeof(uint32_t)));
    if (!shader->is_ucode_analyzed()) {
      StringBuffer disasm;
      shader->AnalyzeUcode(disasm);
    }
    DxbcShaderTranslator::Modification modification(
        cache->shader_translator()->GetDefaultVertexShaderModification(
            shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters),
            host_vertex_shader_type));
    return static_cast<MetalShader::MetalTranslation*>(
        shader->GetOrCreateTranslation(modification.value));
  }

  // Fixture guest PS through the full production TranslateShader chain (needs
  // a real MTLFunction/MTLLibrary for the fragment stage).
  MetalShader::MetalTranslation* MakeGuestPixelShader() {
    static const std::vector<uint32_t> ucode = BuildFixturePixelUcode();
    MetalShader* shader = cache->LoadShader(xenos::ShaderType::kPixel,
                                            ucode.data(),
                                            uint32_t(ucode.size()));
    if (!shader->is_ucode_analyzed()) {
      StringBuffer disasm;
      shader->AnalyzeUcode(disasm);
    }
    DxbcShaderTranslator::Modification modification(
        cache->shader_translator()->GetDefaultPixelShaderModification(
            shader->GetDynamicAddressableRegisterCount(
                xenos::kMaxShaderTempRegisters)));
    auto* translation = static_cast<MetalShader::MetalTranslation*>(
        shader->GetOrCreateTranslation(modification.value));
    if (!cache->TranslateShader(*translation)) {
      return nullptr;
    }
    return translation;
  }
};

std::filesystem::path MakeScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "xe_tess_cache" / name;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

constexpr uint32_t kTitleId = 0x4D5307F2;  // Viva Pinata, the capture title.

void CheckPipelineShape(const MetalPipelineCache::TessellationPipeline* p) {
  REQUIRE(p != nullptr);
  CHECK(p->pipeline != nullptr);
  // The five zero-checked config counts (construction enforces them; this
  // pins that the returned mirror carries them for the group-4 draw call).
  CHECK(p->vs_output_size_in_bytes > 0);
  CHECK(p->gs_max_input_primitives_per_mesh_threadgroup > 0);
  CHECK(p->hs_max_patches_per_object_threadgroup > 0);
  CHECK(p->hs_input_control_point_count > 0);
  CHECK(p->hs_max_object_threads_per_threadgroup > 0);
  CHECK(p->hs_max_tessellation_factor > 0.0f);
  CHECK(p->gs_instance_count == 1);
  // All ten HS blobs declare triangle output.
  CHECK((p->output_primitive_type ==
             uint32_t(IRRuntimeTessellatorOutputTriangleCW) ||
         p->output_primitive_type ==
             uint32_t(IRRuntimeTessellatorOutputTriangleCCW)));
  CHECK(p->geometry_primitive == uint32_t(IRRuntimePrimitiveTypeTriangle));
}

}  // namespace

TEST_CASE(
    "tessellation pipeline cache builds triangle and quad cells and memoizes",
    "[tessellation][cache][device]") {
  CacheHarness h;
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  const AttachmentFixture att = MakeAttachments();

  // Triangle cell with a REAL guest PS.
  MetalShader::MetalTranslation* tri_domain =
      h.MakeDomainTranslation(HostVertexShaderType::kTriangleDomainCPIndexed);
  REQUIRE(tri_domain != nullptr);
  MetalShader::MetalTranslation* guest_ps = h.MakeGuestPixelShader();
  REQUIRE(guest_ps != nullptr);
  const MetalPipelineCache::TessellationPipeline* tri =
      h.cache->GetOrCreateTessellationPipelineState(
          tri_domain, guest_ps, HostVertexShaderType::kTriangleDomainCPIndexed,
          xenos::TessellationMode::kDiscrete,
          xenos::PrimitiveType::kTriangleList, att.colors, att.depth,
          att.raster, att.blends, att.write_mask);
  CheckPipelineShape(tri);
  // 3-control-point patches for the CP-indexed triangle domain.
  CHECK(tri->hs_input_control_point_count == 3);

  // Same key -> memo hit: the SAME entry, not an equivalent rebuild.
  const MetalPipelineCache::TessellationPipeline* tri_again =
      h.cache->GetOrCreateTessellationPipelineState(
          tri_domain, guest_ps, HostVertexShaderType::kTriangleDomainCPIndexed,
          xenos::TessellationMode::kDiscrete,
          xenos::PrimitiveType::kTriangleList, att.colors, att.depth,
          att.raster, att.blends, att.write_mask);
  CHECK(tri_again == tri);

  // Quad cell with NO guest PS -> the depth-only fallback PS (tessellated
  // draws never require a guest PS).
  MetalShader::MetalTranslation* quad_domain =
      h.MakeDomainTranslation(HostVertexShaderType::kQuadDomainCPIndexed);
  REQUIRE(quad_domain != nullptr);
  const MetalPipelineCache::TessellationPipeline* quad =
      h.cache->GetOrCreateTessellationPipelineState(
          quad_domain, /*pixel=*/nullptr,
          HostVertexShaderType::kQuadDomainCPIndexed,
          xenos::TessellationMode::kDiscrete, xenos::PrimitiveType::kQuadList,
          att.colors, att.depth, att.raster, att.blends, att.write_mask);
  CheckPipelineShape(quad);
  CHECK(quad->hs_input_control_point_count == 4);
  CHECK(quad != tri);
}

TEST_CASE(
    "tessellation HS/DS validation failure returns null without crashing",
    "[tessellation][cache][device]") {
  CacheHarness h;
  {
    INFO(h.error);
    REQUIRE(h.ok);
  }
  const AttachmentFixture att = MakeAttachments();

  // A quad-domain HS (4 output control points) deliberately paired with a
  // guest DS translated for the TRIANGLE domain (3 input control points):
  // IRRuntimeValidateTessellationPipeline must reject the pairing, and the
  // rejection must be a null return — a construction failure, never a crash
  // or a silently different pipeline.
  MetalShader::MetalTranslation* tri_domain =
      h.MakeDomainTranslation(HostVertexShaderType::kTriangleDomainCPIndexed);
  REQUIRE(tri_domain != nullptr);
  const MetalPipelineCache::TessellationPipeline* mismatched =
      h.cache->GetOrCreateTessellationPipelineState(
          tri_domain, /*pixel=*/nullptr,
          HostVertexShaderType::kQuadDomainCPIndexed,
          xenos::TessellationMode::kDiscrete, xenos::PrimitiveType::kQuadList,
          att.colors, att.depth, att.raster, att.blends, att.write_mask);
  CHECK(mismatched == nullptr);
  // Repeat: the failure is negative-cached, still null, still no crash.
  const MetalPipelineCache::TessellationPipeline* mismatched_again =
      h.cache->GetOrCreateTessellationPipelineState(
          tri_domain, /*pixel=*/nullptr,
          HostVertexShaderType::kQuadDomainCPIndexed,
          xenos::TessellationMode::kDiscrete, xenos::PrimitiveType::kQuadList,
          att.colors, att.depth, att.raster, att.blends, att.write_mask);
  CHECK(mismatched_again == nullptr);

  // The failure must not poison the MATCHED pairing for the same guest shader.
  const MetalPipelineCache::TessellationPipeline* matched =
      h.cache->GetOrCreateTessellationPipelineState(
          tri_domain, /*pixel=*/nullptr,
          HostVertexShaderType::kTriangleDomainCPIndexed,
          xenos::TessellationMode::kDiscrete,
          xenos::PrimitiveType::kTriangleList, att.colors, att.depth,
          att.raster, att.blends, att.write_mask);
  CheckPipelineShape(matched);
}

TEST_CASE("tessellation pipeline durable storage round-trip",
          "[tessellation][cache][storage][device]") {
  std::filesystem::path root = MakeScratchRoot("roundtrip");
  const AttachmentFixture att = MakeAttachments();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  REQUIRE(device != nil);

  // Phase A: cold cache with storage open — builds the quad/no-PS cell and
  // publishes the tessellation record.
  {
    CacheHarness h(device);
    {
      INFO(h.error);
      REQUIRE(h.ok);
    }
    h.cache->InitializeDurableStorage(root, kTitleId);
    {
      INFO("durable storage did not open (cvar off or identity failure?)");
      REQUIRE(h.cache->durable_storage().is_open());
    }
    MetalShader::MetalTranslation* quad_domain =
        h.MakeDomainTranslation(HostVertexShaderType::kQuadDomainCPIndexed);
    REQUIRE(quad_domain != nullptr);
    const MetalPipelineCache::TessellationPipeline* quad =
        h.cache->GetOrCreateTessellationPipelineState(
            quad_domain, /*pixel=*/nullptr,
            HostVertexShaderType::kQuadDomainCPIndexed,
            xenos::TessellationMode::kDiscrete, xenos::PrimitiveType::kQuadList,
            att.colors, att.depth, att.raster, att.blends, att.write_mask);
    CheckPipelineShape(quad);
    // The cold build must have admitted the record.
    CHECK(h.cache->durable_storage().admitted_entry_count() >= 1);
    h.cache->Shutdown();
  }

  // Phase B: exactly one .tess record exists; recover its key from the
  // storage's own naming scheme.
  uint64_t record_key = 0;
  {
    const std::filesystem::path metallib_dir =
        root / "metal" /
        ([] {
          char name[16];
          std::snprintf(name, sizeof(name), "%08X", kTitleId);
          return std::string(name);
        })() /
        "metallib";
    size_t tess_files = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(metallib_dir)) {
      if (entry.path().extension() == ".tess") {
        ++tess_files;
        record_key =
            std::strtoull(entry.path().stem().string().c_str(), nullptr, 16);
      }
    }
    CHECK(tess_files == 1);
    REQUIRE(record_key != 0);
  }

  // Phase C: standalone reopen + load — the record round-trips whole.
  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId,
                         metal::ComputeMetalPipelineStorageIdentity(
                             (__bridge void*)device)));
    MetalPipelineStorage::TessellationRecord record;
    REQUIRE(storage.LoadTessellationRecord(record_key, &record));
    CHECK(record.host_vertex_shader_type ==
          uint32_t(HostVertexShaderType::kQuadDomainCPIndexed));
    CHECK(record.tessellation_mode ==
          uint32_t(xenos::TessellationMode::kDiscrete));
    CHECK(record.host_primitive_type ==
          uint32_t(xenos::PrimitiveType::kQuadList));
    CHECK_FALSE(record.vs_function_name.empty());
    CHECK_FALSE(record.hs_function_name.empty());
    CHECK_FALSE(record.ds_function_name.empty());
    CHECK_FALSE(record.vs_metallib.empty());
    CHECK_FALSE(record.vs_stage_in_metallib.empty());
    CHECK_FALSE(record.hs_metallib.empty());
    CHECK_FALSE(record.ds_metallib.empty());
    CHECK(record.vs_output_size > 0);
    CHECK(record.hs_input_control_point_count == 4);
    CHECK(record.ds_input_control_point_count == 4);
    storage.Close();
  }

  // Phase D: a FRESH cache over the same directory rebuilds the pipeline FROM
  // the record: the pipeline exists and nothing new was admitted (a cold
  // rebuild would have re-published the record).
  {
    CacheHarness h(device);
    {
      INFO(h.error);
      REQUIRE(h.ok);
    }
    h.cache->InitializeDurableStorage(root, kTitleId);
    REQUIRE(h.cache->durable_storage().is_open());
    MetalShader::MetalTranslation* quad_domain =
        h.MakeDomainTranslation(HostVertexShaderType::kQuadDomainCPIndexed);
    REQUIRE(quad_domain != nullptr);
    const MetalPipelineCache::TessellationPipeline* quad =
        h.cache->GetOrCreateTessellationPipelineState(
            quad_domain, /*pixel=*/nullptr,
            HostVertexShaderType::kQuadDomainCPIndexed,
            xenos::TessellationMode::kDiscrete, xenos::PrimitiveType::kQuadList,
            att.colors, att.depth, att.raster, att.blends, att.write_mask);
    CheckPipelineShape(quad);
    CHECK(h.cache->durable_storage().admitted_entry_count() == 0);
    h.cache->Shutdown();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
