/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// port-xenios-tessellation task 3.2 — the design-D4 storage-identity evidence.
//
// D4's rule: decide invalidation from the IMPLEMENTED artifact/key change; a
// fingerprint (or fetch-ABI) bump happens ONLY if an existing artifact kind's
// meaning changed. The tessellation record is a NEW kind under a NEW file
// extension (.tess) with its own key derivation (kMetalTessellationRecordKind
// baked into MetalTessellationRecordKey); neither the .metallib nor the .geom
// readers/writers were touched. These cases prove the consequence end to end
// through the real MetalPipelineStorage:
//
//   1. a store shaped exactly like a PRE-change store (only metallibs and
//      geometry records — the untouched writers ARE the pre-change shape)
//      still hits for the existing kinds after reopening, while tessellation
//      lookups miss cleanly and disturb nothing;
//   2. the three kinds coexist at a NUMERICALLY IDENTICAL key without
//      cross-loading;
//   3. a corrupt/truncated tessellation record is a clean miss that leaves
//      the other kinds intact;
//   4. the key derivation separates identities, types and modes.
//
// VERDICT ENCODED HERE: natural disjoint miss — NO fingerprint bump. The
// companion pin is metal_fetch_abi_cache_identity_test.cc's
// "kFetchAbiRevision == 4" case, which this change must NOT move.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_pipeline_storage.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::MetalPipelineStorage;
using metal::MetalPipelineStorageIdentity;

MetalPipelineStorageIdentity FixedIdentity() {
  MetalPipelineStorageIdentity identity;
  identity.translator_schema = 0x00000155u;
  identity.dxc_artifact_hash = 0x1111111111111111ull;
  identity.msc_artifact_hash = 0x2222222222222222ull;
  identity.device_hash = 0x3333333333333333ull;
  identity.os_build_hash = 0x4444444444444444ull;
  return identity;
}

std::filesystem::path MakeScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "xe_tess_storage" / name;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

constexpr uint32_t kTitleId = 0x4D5307F2;  // Viva Pinata, the capture title.

const std::vector<uint8_t>& SampleMetallib() {
  static const std::vector<uint8_t> bytes = {0xDE, 0xAD, 0xBE, 0xEF,
                                             0x01, 0x02, 0x03, 0x04};
  return bytes;
}

MetalPipelineStorage::GeometryRecord SampleGeometryRecord() {
  MetalPipelineStorage::GeometryRecord record;
  record.function_name = "geometry_main";
  record.reflection = 48;
  record.blob = {0x10, 0x20, 0x30, 0x40};
  record.aux_blob = {0x50, 0x60};
  return record;
}

// A record that passes every loader plausibility check (the required-nonzero
// reflection set mirrors the five zero-checked pipeline-config counts plus the
// validation-consumed control-point fields).
MetalPipelineStorage::TessellationRecord SampleTessellationRecord() {
  MetalPipelineStorage::TessellationRecord record;
  record.host_vertex_shader_type = 5;  // kQuadDomainCPIndexed
  record.tessellation_mode = 0;        // kDiscrete
  record.host_primitive_type = 0x0D;   // kQuadList
  record.vs_function_name = "tess_vs";
  record.hs_function_name = "tess_hs";
  record.ds_function_name = "tess_ds";
  record.vs_output_size = 64;
  record.hs_max_patches_per_object_threadgroup = 16;
  record.hs_max_object_threads_per_patch = 32;
  record.hs_patch_constants_size = 32;
  record.hs_input_control_point_count = 4;
  record.hs_output_control_point_count = 4;
  record.hs_output_control_point_size = 16;
  record.hs_tessellator_domain = 3;
  record.hs_tessellator_partitioning = 1;
  record.hs_tessellator_output_primitive = 3;
  record.hs_max_tessellation_factor = 64.0f;
  record.ds_max_input_prims_per_mesh_threadgroup = 64;
  record.ds_input_control_point_count = 4;
  record.ds_input_control_point_size = 16;
  record.ds_patch_constants_size = 32;
  record.ds_tessellator_domain = 3;
  record.vs_metallib = {0xA0, 0xA1, 0xA2, 0xA3};
  record.vs_stage_in_metallib = {0xB0, 0xB1};
  record.hs_metallib = {0xC0, 0xC1, 0xC2};
  record.ds_metallib = {0xD0, 0xD1, 0xD2, 0xD3, 0xD4};
  return record;
}

bool TessellationRecordsEqual(
    const MetalPipelineStorage::TessellationRecord& a,
    const MetalPipelineStorage::TessellationRecord& b) {
  return a.host_vertex_shader_type == b.host_vertex_shader_type &&
         a.tessellation_mode == b.tessellation_mode &&
         a.host_primitive_type == b.host_primitive_type &&
         a.vs_function_name == b.vs_function_name &&
         a.hs_function_name == b.hs_function_name &&
         a.ds_function_name == b.ds_function_name &&
         a.vs_output_size == b.vs_output_size &&
         a.hs_max_patches_per_object_threadgroup ==
             b.hs_max_patches_per_object_threadgroup &&
         a.hs_max_object_threads_per_patch ==
             b.hs_max_object_threads_per_patch &&
         a.hs_patch_constants_size == b.hs_patch_constants_size &&
         a.hs_input_control_point_count == b.hs_input_control_point_count &&
         a.hs_output_control_point_count == b.hs_output_control_point_count &&
         a.hs_output_control_point_size == b.hs_output_control_point_size &&
         a.hs_tessellator_domain == b.hs_tessellator_domain &&
         a.hs_tessellator_partitioning == b.hs_tessellator_partitioning &&
         a.hs_tessellator_output_primitive ==
             b.hs_tessellator_output_primitive &&
         a.hs_max_tessellation_factor == b.hs_max_tessellation_factor &&
         a.ds_max_input_prims_per_mesh_threadgroup ==
             b.ds_max_input_prims_per_mesh_threadgroup &&
         a.ds_input_control_point_count == b.ds_input_control_point_count &&
         a.ds_input_control_point_size == b.ds_input_control_point_size &&
         a.ds_patch_constants_size == b.ds_patch_constants_size &&
         a.ds_tessellator_domain == b.ds_tessellator_domain &&
         a.vs_metallib == b.vs_metallib &&
         a.vs_stage_in_metallib == b.vs_stage_in_metallib &&
         a.hs_metallib == b.hs_metallib && a.ds_metallib == b.ds_metallib;
}

}  // namespace

TEST_CASE(
    "A pre-change-shaped store keeps hitting for existing kinds while "
    "tessellation lookups miss cleanly",
    "[tessellation][storage][d4]") {
  std::filesystem::path root = MakeScratchRoot("pre_change_disjoint");
  const MetalPipelineStorageIdentity identity = FixedIdentity();
  constexpr uint64_t kMetallibKey = 0xABCDEF0123456789ull;
  constexpr uint64_t kGeometryKey = 0x0F0F0F0F0F0F0F0Full;
  const MetalPipelineStorage::GeometryRecord geometry = SampleGeometryRecord();

  // Populate a store using ONLY the untouched pre-change writers.
  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId, identity));
    REQUIRE(storage.StoreMetallib(kMetallibKey, SampleMetallib()));
    REQUIRE(storage.StoreGeometryRecord(kGeometryKey, geometry));
    storage.Close();
  }

  // Reopen (same identity — no discard) and interleave existing-kind hits with
  // tessellation misses. The misses must not disturb the existing entries.
  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId, identity));

    MetalPipelineStorage::TessellationRecord tess;
    CHECK_FALSE(storage.LoadTessellationRecord(kMetallibKey, &tess));
    CHECK_FALSE(storage.LoadTessellationRecord(kGeometryKey, &tess));
    CHECK_FALSE(storage.LoadTessellationRecord(
        metal::MetalTessellationRecordKey(0x123456789ABCDEF0ull, 2, 0),
        &tess));

    std::vector<uint8_t> metallib;
    REQUIRE(storage.LoadMetallib(kMetallibKey, &metallib));
    CHECK(metallib == SampleMetallib());
    MetalPipelineStorage::GeometryRecord served_geometry;
    REQUIRE(storage.LoadGeometryRecord(kGeometryKey, &served_geometry));
    CHECK(served_geometry.blob == geometry.blob);
    CHECK(served_geometry.aux_blob == geometry.aux_blob);
    CHECK(served_geometry.reflection == geometry.reflection);
    CHECK(served_geometry.function_name == geometry.function_name);

    // And again after the misses, proving a tessellation miss is pure.
    std::vector<uint8_t> metallib_again;
    REQUIRE(storage.LoadMetallib(kMetallibKey, &metallib_again));
    CHECK(metallib_again == SampleMetallib());
    storage.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "All three artifact kinds coexist at a numerically identical key without "
    "cross-loading",
    "[tessellation][storage][d4]") {
  std::filesystem::path root = MakeScratchRoot("same_numeric_key");
  const MetalPipelineStorageIdentity identity = FixedIdentity();
  // The SAME 64-bit key for all three kinds. Kind separation must come from
  // the key derivation AND the file namespace, never from luck.
  constexpr uint64_t kSharedKey = 0x5155414C49464945ull;
  const MetalPipelineStorage::GeometryRecord geometry = SampleGeometryRecord();
  const MetalPipelineStorage::TessellationRecord tessellation =
      SampleTessellationRecord();

  MetalPipelineStorage storage;
  REQUIRE(storage.Open(root, kTitleId, identity));
  REQUIRE(storage.StoreMetallib(kSharedKey, SampleMetallib()));
  REQUIRE(storage.StoreGeometryRecord(kSharedKey, geometry));
  REQUIRE(storage.StoreTessellationRecord(kSharedKey, tessellation));

  std::vector<uint8_t> metallib;
  REQUIRE(storage.LoadMetallib(kSharedKey, &metallib));
  CHECK(metallib == SampleMetallib());
  MetalPipelineStorage::GeometryRecord served_geometry;
  REQUIRE(storage.LoadGeometryRecord(kSharedKey, &served_geometry));
  CHECK(served_geometry.blob == geometry.blob);
  MetalPipelineStorage::TessellationRecord served_tessellation;
  REQUIRE(storage.LoadTessellationRecord(kSharedKey, &served_tessellation));
  CHECK(TessellationRecordsEqual(served_tessellation, tessellation));

  // Round-trip across a close/reopen too (same identity — nothing discarded).
  storage.Close();
  MetalPipelineStorage reopened;
  REQUIRE(reopened.Open(root, kTitleId, identity));
  MetalPipelineStorage::TessellationRecord warm;
  REQUIRE(reopened.LoadTessellationRecord(kSharedKey, &warm));
  CHECK(TessellationRecordsEqual(warm, tessellation));
  std::vector<uint8_t> warm_metallib;
  REQUIRE(reopened.LoadMetallib(kSharedKey, &warm_metallib));
  CHECK(warm_metallib == SampleMetallib());
  reopened.Close();

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "A corrupt tessellation record is a clean miss that leaves other kinds "
    "intact",
    "[tessellation][storage][d4]") {
  std::filesystem::path root = MakeScratchRoot("corrupt_tess");
  const MetalPipelineStorageIdentity identity = FixedIdentity();
  constexpr uint64_t kKey = 0x00C0FFEE00C0FFEEull;

  MetalPipelineStorage storage;
  REQUIRE(storage.Open(root, kTitleId, identity));
  REQUIRE(storage.StoreMetallib(kKey, SampleMetallib()));
  REQUIRE(storage.StoreTessellationRecord(kKey, SampleTessellationRecord()));

  // Truncate the .tess file mid-record: the header promises more bytes than
  // exist, so every component check after the short read must fail as a unit.
  const std::filesystem::path tess_path =
      storage.title_root() / "metallib" /
      // Same naming scheme the storage uses.
      ([] {
        char name[32];
        std::snprintf(name, sizeof(name), "%016llX.tess",
                      (unsigned long long)kKey);
        return std::string(name);
      })();
  REQUIRE(std::filesystem::exists(tess_path));
  std::error_code resize_ec;
  std::filesystem::resize_file(
      tess_path, std::filesystem::file_size(tess_path) / 2, resize_ec);
  REQUIRE_FALSE(resize_ec);

  MetalPipelineStorage::TessellationRecord served;
  CHECK_FALSE(storage.LoadTessellationRecord(kKey, &served));
  // The miss removed the corrupt entry; a second lookup is an ordinary miss.
  CHECK_FALSE(storage.LoadTessellationRecord(kKey, &served));
  // Other kinds are untouched by the discard.
  std::vector<uint8_t> metallib;
  REQUIRE(storage.LoadMetallib(kKey, &metallib));
  CHECK(metallib == SampleMetallib());
  storage.Close();

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("The tessellation record key separates identities, types and modes",
          "[tessellation][storage][d4]") {
  const uint64_t base =
      metal::MetalTessellationRecordKey(0x1234567890ABCDEFull, 2, 0);
  // Different domain-shader identity.
  CHECK(base !=
        metal::MetalTessellationRecordKey(0x1234567890ABCDF0ull, 2, 0));
  // Different host vertex shader type.
  CHECK(base != metal::MetalTessellationRecordKey(0x1234567890ABCDEFull, 3, 0));
  // Different tessellation mode.
  CHECK(base != metal::MetalTessellationRecordKey(0x1234567890ABCDEFull, 2, 1));
  // Never equal to a geometry key over the same identity inputs — the kind
  // participates in the derivation.
  CHECK(base != metal::MetalGeometryRecordKey(
                    metal::kMetalGeometryRecordKindVertexStage,
                    0x1234567890ABCDEFull, 2));
  CHECK(base != metal::MetalGeometryRecordKey(
                    metal::kMetalGeometryRecordKindShaderStage,
                    0x1234567890ABCDEFull, 0));
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
