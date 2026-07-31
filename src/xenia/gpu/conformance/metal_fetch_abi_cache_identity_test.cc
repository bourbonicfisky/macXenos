/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Cache-identity conformance for the shader fetch ABI
// (openspec/changes/implement-unsigned-integer-texture-scaling, tasks 5.2/5.3
// and 6.6).
//
// The durable metallib cache keys on MetalPipelineStorageIdentity. Before this
// change its translator_schema encoded only the five MetalTranslatorConfig
// booleans, which describe the translator's CONFIGURATION. A change to what the
// translator EMITS for byte-identical guest ucode — a new system constant and a
// new fetch codepath — moves none of them, so a cache written by the previous
// build would have been served shaders compiled against the previous cbuffer
// layout. kFetchAbiRevision closes that.
//
// These cases go end to end through the real MetalPipelineStorage: write a
// metallib under one ABI revision, reopen under another, and require that the
// stale artifact is not served. A pure operator== check would not catch a
// storage layer that ignored the identity.

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
using metal::MetalTranslatorConfig;

// The identity the running build computes, minus the device/OS hashes (which
// need a device). Everything under test lives in translator_schema.
MetalPipelineStorageIdentity IdentityWithFetchAbi(uint32_t fetch_abi_revision) {
  // Mirrors ComputeMetalPipelineStorageIdentity's packing. Kept textually close
  // so a change to the layout there without a change here trips the equality
  // assertions below rather than passing silently.
  struct SchemaBits {
    uint32_t vendor_apple : 1;
    uint32_t bindless_resources_used : 1;
    uint32_t edram_rov_used : 1;
    uint32_t gamma_render_target_as_unorm8 : 1;
    uint32_t msc_bindless_resources_used : 1;
    uint32_t fetch_abi_revision : 8;
    uint32_t reserved : 19;
  } schema = {};
  schema.vendor_apple = 1;
  schema.bindless_resources_used = MetalTranslatorConfig::kBindlessResourcesUsed;
  schema.edram_rov_used = MetalTranslatorConfig::kEdramRovUsed;
  schema.gamma_render_target_as_unorm8 =
      MetalTranslatorConfig::kGammaRenderTargetAsUnorm8;
  schema.msc_bindless_resources_used =
      MetalTranslatorConfig::kMscBindlessResourcesUsed;
  schema.fetch_abi_revision = fetch_abi_revision;

  MetalPipelineStorageIdentity identity;
  std::memcpy(&identity.translator_schema, &schema, sizeof(schema));
  identity.dxc_artifact_hash = 0x1111111111111111ull;
  identity.msc_artifact_hash = 0x2222222222222222ull;
  identity.device_hash = 0x3333333333333333ull;
  identity.os_build_hash = 0x4444444444444444ull;
  return identity;
}

// Unique scratch root per case; removed on the way in and out so a previous
// failed run cannot make a later one pass.
std::filesystem::path MakeScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "xe_fetch_abi_cache" / name;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

constexpr uint32_t kTitleId = 0x534307E7;  // Just Cause 2, the engaged title.
constexpr uint64_t kMetallibKey = 0xABCDEF0123456789ull;

const std::vector<uint8_t>& SampleMetallib() {
  static const std::vector<uint8_t> bytes = {0xDE, 0xAD, 0xBE, 0xEF,
                                             0x01, 0x02, 0x03, 0x04};
  return bytes;
}

}  // namespace

TEST_CASE("The production fetch ABI revision is the current one",
          "[integer_scale][cache_identity]") {
  // If this fails, either the revision was bumped without updating the tests
  // below, or it was reverted and the stale-cache hazard is open again.
  // 4 = float24 depth invariance modes (implement-metal-depth-semantics
  // stage A); the 3->4 boundary is pinned by its own transition case below.
  CHECK(MetalTranslatorConfig::kFetchAbiRevision == 4);
}

TEST_CASE("The revision-3 to revision-4 boundary discards, not filters",
          "[cache_identity][float24][depth_semantics]") {
  // Stage A's stale-artifact boundary, pinned explicitly: a metallib written
  // by the last pre-float24 build (revision 3) must serve under revision 3,
  // must be rejected under the production revision 4 — and after that
  // rejection, MUTATING THE IDENTITY BACK to 3 must find nothing. The third
  // leg distinguishes a store that DISCARDED the stale directory (required:
  // rev-3 records were all written with the depth-stencil-mode bits at zero,
  // so they can never be trusted again) from one that merely filtered reads
  // by identity while leaving poisoned bytes to be resurrected by a
  // downgrade or an A/B rollback build.
  std::filesystem::path root = MakeScratchRoot("f24_transition");

  {
    MetalPipelineStorage rev3;
    REQUIRE(rev3.Open(root, kTitleId, IdentityWithFetchAbi(3)));
    REQUIRE(rev3.StoreMetallib(kMetallibKey, SampleMetallib()));
    rev3.Close();
    // Control: revision 3 serves its own artifact, or the rejections below
    // prove nothing.
    MetalPipelineStorage warm3;
    REQUIRE(warm3.Open(root, kTitleId, IdentityWithFetchAbi(3)));
    std::vector<uint8_t> served;
    REQUIRE(warm3.LoadMetallib(kMetallibKey, &served));
    CHECK(served == SampleMetallib());
    warm3.Close();
  }

  {
    // Production revision (4): the revision-3 artifact must not be served.
    MetalPipelineStorage rev4;
    REQUIRE(rev4.Open(root, kTitleId,
                      IdentityWithFetchAbi(
                          MetalTranslatorConfig::kFetchAbiRevision)));
    std::vector<uint8_t> served;
    CHECK_FALSE(rev4.LoadMetallib(kMetallibKey, &served));
    rev4.Close();
  }

  {
    // Downgrade back to 3 AFTER the revision-4 open: still nothing. The
    // rev-4 open must have discarded the stale directory, so there is no
    // revision-3 artifact left to resurrect.
    MetalPipelineStorage back3;
    REQUIRE(back3.Open(root, kTitleId, IdentityWithFetchAbi(3)));
    std::vector<uint8_t> served;
    CHECK_FALSE(back3.LoadMetallib(kMetallibKey, &served));
    back3.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("Fetch ABI revision participates in the cache identity",
          "[integer_scale][cache_identity]") {
  MetalPipelineStorageIdentity previous = IdentityWithFetchAbi(1);
  MetalPipelineStorageIdentity current =
      IdentityWithFetchAbi(MetalTranslatorConfig::kFetchAbiRevision);

  SECTION("a different revision is a different identity") {
    CHECK(previous != current);
    CHECK_FALSE(previous == current);
    CHECK(previous.translator_schema != current.translator_schema);
  }

  SECTION("the mismatch is named, not silent") {
    // A cache that rebuilds without saying why is indistinguishable from one
    // that never worked.
    std::string described = current.DescribeMismatch(previous);
    CHECK_FALSE(described.empty());
    CHECK(described != "none");
  }

  SECTION("the same revision is the same identity") {
    MetalPipelineStorageIdentity same =
        IdentityWithFetchAbi(MetalTranslatorConfig::kFetchAbiRevision);
    CHECK(current == same);
    CHECK(current.DescribeMismatch(same) == "none");
  }

  SECTION("the revision occupies bits above the configuration booleans") {
    // The five booleans must keep their meaning; the revision must not alias
    // them. Revision 0 vs the production revision may differ ONLY above bit 4.
    MetalPipelineStorageIdentity zero_revision = IdentityWithFetchAbi(0);
    uint32_t delta =
        zero_revision.translator_schema ^ current.translator_schema;
    CHECK((delta & 0x1Fu) == 0u);
    CHECK(delta != 0u);
  }
}

TEST_CASE("A cache written under the previous fetch ABI is not served",
          "[integer_scale][cache_identity]") {
  std::filesystem::path root = MakeScratchRoot("stale_rejected");

  // Cold open under the PREVIOUS ABI revision, and store an artifact.
  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId, IdentityWithFetchAbi(1)));
    REQUIRE(storage.is_open());
    REQUIRE(storage.StoreMetallib(kMetallibKey, SampleMetallib()));
    // Warm reopen under the SAME revision must serve it, otherwise the
    // rejection below would prove nothing — it could just be a store that never
    // worked.
    storage.Close();
    MetalPipelineStorage warm;
    REQUIRE(warm.Open(root, kTitleId, IdentityWithFetchAbi(1)));
    std::vector<uint8_t> served;
    REQUIRE(warm.LoadMetallib(kMetallibKey, &served));
    CHECK(served == SampleMetallib());
    warm.Close();
  }

  // Reopen under the CURRENT ABI revision: the directory must be discarded.
  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId,
                         IdentityWithFetchAbi(
                             MetalTranslatorConfig::kFetchAbiRevision)));
    std::vector<uint8_t> served;
    CHECK_FALSE(storage.LoadMetallib(kMetallibKey, &served));
    storage.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("An unchanged build keeps reusing its own cache",
          "[integer_scale][cache_identity]") {
  // The complement of the case above: invalidation must be caused by an actual
  // identity change, not by opening the cache twice. A store that rebuilt every
  // run would also "pass" a stale-rejection test.
  std::filesystem::path root = MakeScratchRoot("warm_reuse");
  const MetalPipelineStorageIdentity identity =
      IdentityWithFetchAbi(MetalTranslatorConfig::kFetchAbiRevision);

  {
    MetalPipelineStorage cold;
    REQUIRE(cold.Open(root, kTitleId, identity));
    REQUIRE(cold.StoreMetallib(kMetallibKey, SampleMetallib()));
    cold.Close();
  }

  for (int reopen = 0; reopen < 3; ++reopen) {
    INFO("reopen " << reopen);
    MetalPipelineStorage warm;
    REQUIRE(warm.Open(root, kTitleId, identity));
    std::vector<uint8_t> served;
    REQUIRE(warm.LoadMetallib(kMetallibKey, &served));
    CHECK(served == SampleMetallib());
    CHECK(warm.is_open());
    warm.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A geometry record written under a previous ABI is not served",
          "[integer_scale][cache_identity][geometry]") {
  // Geometry records share the title directory with metallibs, so the same
  // identity guards them. The clip-distance boundary relabel changes the
  // EMITTED object/mesh stages for byte-identical guest ucode — exactly the
  // case a configuration-only identity would miss.
  std::filesystem::path root = MakeScratchRoot("geometry_stale");
  constexpr uint64_t kGeometryKey = 0x0F0F0F0F0F0F0F0Full;
  MetalPipelineStorage::GeometryRecord record;
  record.function_name = "main";
  record.reflection = 32;
  record.blob = {0x01, 0x02, 0x03, 0x04};
  record.aux_blob = {0x05, 0x06};

  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId, IdentityWithFetchAbi(2)));
    REQUIRE(storage.StoreGeometryRecord(kGeometryKey, record));
    storage.Close();
    // Same revision must SERVE it, or the rejection below proves nothing.
    MetalPipelineStorage warm;
    REQUIRE(warm.Open(root, kTitleId, IdentityWithFetchAbi(2)));
    MetalPipelineStorage::GeometryRecord served;
    REQUIRE(warm.LoadGeometryRecord(kGeometryKey, &served));
    CHECK(served.blob == record.blob);
    CHECK(served.reflection == record.reflection);
    warm.Close();
  }

  {
    MetalPipelineStorage storage;
    REQUIRE(storage.Open(root, kTitleId,
                         IdentityWithFetchAbi(
                             MetalTranslatorConfig::kFetchAbiRevision)));
    MetalPipelineStorage::GeometryRecord served;
    CHECK_FALSE(storage.LoadGeometryRecord(kGeometryKey, &served));
    storage.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
