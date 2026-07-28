/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Direct regression for the geometry object-stage IDENTITY defect
// (openspec/changes/implement-metal-shader-pipeline-lifetime, task 3.10).
//
// THE DEFECT: `geometry_vertex_stages_` keyed on the vertex translation POINTER
// alone, while the geometry type selects the MSC input topology the object
// stage is compiled against (point -> kPoint, rect -> kTriangle, quad ->
// kLine). The same guest VS used with two geometry types therefore received
// whichever object stage was built FIRST, and the second silently got a stage
// compiled for the wrong topology.
//
// WHY THIS TEST EXISTS AND BYTE-EQUIVALENCE DOES NOT REPLACE IT: the four-trace
// corpus never uses one VS with two geometry types, so every capture is
// byte-identical with the defect present OR fixed. Corpus equivalence cannot
// protect this fix. Without a direct test, a refactor that dropped the topology
// from either key would pass every existing gate.
//
// Covered here: distinct durable keys and distinct stored records for the same
// shader under two topologies, in BOTH creation orders, with each record's
// reflection and payload surviving a cold load and a warm reopen.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#import <Metal/Metal.h>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_pipeline_storage.h"

namespace xe {
namespace gpu {
namespace metal {
namespace {

// The two MSC input topologies a shared guest VS can legitimately be compiled
// against. Values mirror MetalShaderConverter::InputTopology; the test only
// needs them to be distinct and stable.
constexpr uint32_t kTopologyPoint = 1;
constexpr uint32_t kTopologyLine = 2;
constexpr uint32_t kTopologyTriangle = 3;

constexpr uint64_t kSharedShaderIdentity = 0x0123456789ABCDEFull;

std::filesystem::path MakeScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("xe_geom_identity_" + std::string(name));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

MetalPipelineStorageIdentity TestIdentity() {
  // Fixed, device-independent: this test is about record identity, not about
  // the provenance gate (covered separately by the 4.2 harness).
  MetalPipelineStorageIdentity identity;
  identity.translator_schema = 0x1234;
  identity.dxc_artifact_hash = 0x1111111111111111ull;
  identity.msc_artifact_hash = 0x2222222222222222ull;
  identity.device_hash = 0x3333333333333333ull;
  identity.os_build_hash = 0x4444444444444444ull;
  return identity;
}

MetalPipelineStorage::GeometryRecord MakeRecord(const char* name,
                                                uint32_t reflection,
                                                uint8_t fill) {
  MetalPipelineStorage::GeometryRecord record;
  record.function_name = name;
  record.reflection = reflection;
  record.blob.assign(256, fill);
  record.aux_blob.assign(128, uint8_t(fill ^ 0xFF));
  return record;
}

}  // namespace

TEST_CASE("Geometry record key separates input topologies",
          "[metal][geometry][identity]") {
  const uint64_t point = MetalGeometryRecordKey(
      kMetalGeometryRecordKindVertexStage, kSharedShaderIdentity,
      kTopologyPoint);
  const uint64_t line = MetalGeometryRecordKey(
      kMetalGeometryRecordKindVertexStage, kSharedShaderIdentity, kTopologyLine);
  const uint64_t triangle = MetalGeometryRecordKey(
      kMetalGeometryRecordKindVertexStage, kSharedShaderIdentity,
      kTopologyTriangle);

  // The defect in one assertion: identical shader, different topology, and the
  // identity must NOT collapse.
  REQUIRE(point != line);
  REQUIRE(point != triangle);
  REQUIRE(line != triangle);

  // Same inputs must still be stable, or nothing would ever hit.
  REQUIRE(point == MetalGeometryRecordKey(kMetalGeometryRecordKindVertexStage,
                                          kSharedShaderIdentity,
                                          kTopologyPoint));
}

TEST_CASE("Geometry record key separates record kinds",
          "[metal][geometry][identity]") {
  // Both kinds share one directory; colliding would load a mesh-stage record as
  // an object-stage record.
  REQUIRE(MetalGeometryRecordKey(kMetalGeometryRecordKindVertexStage,
                                 kSharedShaderIdentity, kTopologyPoint) !=
          MetalGeometryRecordKey(kMetalGeometryRecordKindShaderStage,
                                 kSharedShaderIdentity, kTopologyPoint));
}

TEST_CASE("One shader under two topologies stores two distinct records",
          "[metal][geometry][identity]") {
  // Both creation orders, because the defect was order-dependent: whichever was
  // built first won. An order-insensitive result is the property under test.
  for (bool point_first : {true, false}) {
    const char* order_name = point_first ? "point_first" : "line_first";
    const std::filesystem::path root = MakeScratchRoot(order_name);

    const uint64_t key_point = MetalGeometryRecordKey(
        kMetalGeometryRecordKindVertexStage, kSharedShaderIdentity,
        kTopologyPoint);
    const uint64_t key_line = MetalGeometryRecordKey(
        kMetalGeometryRecordKindVertexStage, kSharedShaderIdentity,
        kTopologyLine);

    const auto record_point = MakeRecord("vs_point", 32u, 0xA0);
    const auto record_line = MakeRecord("vs_line", 64u, 0x0B);

    {
      MetalPipelineStorage storage;
      REQUIRE(storage.Open(root, /*title_id=*/0x4C4107D1u, TestIdentity()));

      if (point_first) {
        REQUIRE(storage.StoreGeometryRecord(key_point, record_point));
        REQUIRE(storage.StoreGeometryRecord(key_line, record_line));
      } else {
        REQUIRE(storage.StoreGeometryRecord(key_line, record_line));
        REQUIRE(storage.StoreGeometryRecord(key_point, record_point));
      }

      // TWO artifacts admitted, not one. Under the defect the second would
      // have been dropped as a duplicate of the first.
      REQUIRE(storage.admitted_entry_count() == 2);

      // Cold read-back within the same session: each key returns ITS OWN
      // record, never the first-written one.
      MetalPipelineStorage::GeometryRecord loaded_point;
      MetalPipelineStorage::GeometryRecord loaded_line;
      REQUIRE(storage.LoadGeometryRecord(key_point, &loaded_point));
      REQUIRE(storage.LoadGeometryRecord(key_line, &loaded_line));

      REQUIRE(loaded_point.function_name == "vs_point");
      REQUIRE(loaded_line.function_name == "vs_line");
      REQUIRE(loaded_point.reflection == 32u);
      REQUIRE(loaded_line.reflection == 64u);
      REQUIRE(loaded_point.blob == record_point.blob);
      REQUIRE(loaded_line.blob == record_line.blob);
      REQUIRE(loaded_point.aux_blob == record_point.aux_blob);
      REQUIRE(loaded_line.aux_blob == record_line.aux_blob);
    }

    {
      // WARM reopen: a fresh storage object over the same directory must still
      // resolve each topology to its own record. This is the case that
      // distinguishes a durable defect from a merely in-memory one — a
      // mis-keyed record would survive the process.
      MetalPipelineStorage storage;
      REQUIRE(storage.Open(root, /*title_id=*/0x4C4107D1u, TestIdentity()));

      MetalPipelineStorage::GeometryRecord warm_point;
      MetalPipelineStorage::GeometryRecord warm_line;
      REQUIRE(storage.LoadGeometryRecord(key_point, &warm_point));
      REQUIRE(storage.LoadGeometryRecord(key_line, &warm_line));

      REQUIRE(warm_point.reflection == 32u);
      REQUIRE(warm_line.reflection == 64u);
      REQUIRE(warm_point.function_name == "vs_point");
      REQUIRE(warm_line.function_name == "vs_line");
      REQUIRE(warm_point.blob == record_point.blob);
      REQUIRE(warm_line.blob == record_line.blob);
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
}

TEST_CASE("In-memory object-stage key separates input topologies",
          "[metal][geometry][identity]") {
  // The production defect lived HERE, in the run-local map key, not in the
  // durable one. `translation` is an opaque identity, so this needs no MSC
  // fixture and no access to MetalPipelineCache internals.
  int translation_a = 0;  // Stand-ins for two distinct translation objects.
  int translation_b = 0;
  const void* vs_a = &translation_a;
  const void* vs_b = &translation_b;

  const MetalGeometryVertexStageKey point{vs_a, kTopologyPoint};
  const MetalGeometryVertexStageKey line{vs_a, kTopologyLine};
  const MetalGeometryVertexStageKey other_shader{vs_b, kTopologyPoint};

  // Same translated VS, different topology => different stages.
  REQUIRE(point != line);
  REQUIRE(point != other_shader);
  REQUIRE(point == MetalGeometryVertexStageKey{vs_a, kTopologyPoint});

  // Topology must reach the HASH too, not only equality. Equality alone would
  // still be correct but would pile every topology into one bucket, and a
  // hasher that ignores a field is a latent step toward dropping it.
  const MetalGeometryVertexStageKey::Hasher hasher;
  REQUIRE(hasher(point) != hasher(line));
  REQUIRE(hasher(point) == hasher(MetalGeometryVertexStageKey{vs_a,
                                                              kTopologyPoint}));
}

TEST_CASE("Object-stage map keeps both topologies in either insertion order",
          "[metal][geometry][identity]") {
  // The defect was order-dependent: whichever stage was built first won. Both
  // orders must therefore end with TWO entries, each resolving to its own value.
  int translation = 0;
  const void* vs = &translation;
  const MetalGeometryVertexStageKey point{vs, kTopologyPoint};
  const MetalGeometryVertexStageKey line{vs, kTopologyLine};

  for (bool point_first : {true, false}) {
    std::unordered_map<MetalGeometryVertexStageKey, std::string,
                       MetalGeometryVertexStageKey::Hasher>
        stages;
    if (point_first) {
      stages.emplace(point, "object_stage_point");
      stages.emplace(line, "object_stage_line");
    } else {
      stages.emplace(line, "object_stage_line");
      stages.emplace(point, "object_stage_point");
    }

    // Two entries, not one deduplicated.
    REQUIRE(stages.size() == 2);

    // And each lookup returns ITS OWN stage, never the first-inserted one.
    auto point_it = stages.find(point);
    auto line_it = stages.find(line);
    REQUIRE(point_it != stages.end());
    REQUIRE(line_it != stages.end());
    REQUIRE(point_it->second == "object_stage_point");
    REQUIRE(line_it->second == "object_stage_line");
  }
}

TEST_CASE("A geometry record missing a component is a whole-record miss",
          "[metal][geometry][identity]") {
  // The single-record design exists so a fresh component is never paired with a
  // stale one. Truncating away the stage-in must reject the WHOLE record rather
  // than returning an object stage with no stage-in.
  const std::filesystem::path root = MakeScratchRoot("partial");
  const uint64_t key = MetalGeometryRecordKey(
      kMetalGeometryRecordKindVertexStage, kSharedShaderIdentity,
      kTopologyPoint);
  const auto record = MakeRecord("vs_point", 32u, 0xA0);

  std::filesystem::path entry_path;
  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, /*title_id=*/0x4C4107D1u, TestIdentity()));
    REQUIRE(storage.StoreGeometryRecord(key, record));
    entry_path = storage.metallib_dir() / fmt::format("{:016X}.geom", key);
  }

  // Drop exactly the aux blob; the header still claims it is present.
  {
    std::error_code ec;
    const auto size = std::filesystem::file_size(entry_path, ec);
    REQUIRE(!ec);
    std::filesystem::resize_file(entry_path, size - record.aux_blob.size(), ec);
    REQUIRE(!ec);
  }

  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, /*title_id=*/0x4C4107D1u, TestIdentity()));
    MetalPipelineStorage::GeometryRecord loaded;
    REQUIRE_FALSE(storage.LoadGeometryRecord(key, &loaded));
    // And nothing partial leaks out of the failed load.
    REQUIRE(loaded.blob.empty());
    REQUIRE(loaded.aux_blob.empty());
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
