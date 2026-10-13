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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/metal/metal_converter_options.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::MetalPipelineStorage;
using metal::MetalPipelineStorageIdentity;
using metal::MetalGammaRenderTargetLinearSetting;
using metal::MetalTranslatorConfig;

// The identity the running build computes, minus the device/OS hashes (which
// need a device). Everything under test lives in translator_schema.
//
// MIRROR DRIFT, AND WHY THE GUARD BELOW EXISTS. This function used to claim
// that keeping it "textually close" to ComputeMetalPipelineStorageIdentity
// would make a layout change there trip the assertions here. It did not:
// dxilconv_generation was added to the production packing on 2026-08-02 and
// this mirror never noticed, because every case compares mirror values against
// other mirror values. A mirror can only catch drift by being compared to the
// thing it mirrors — see "The identity mirror matches the production packing".
MetalPipelineStorageIdentity IdentityWithSchema(uint32_t fetch_abi_revision,
                                                uint32_t dxilconv_generation,
                                                bool gamma_as_unorm8 =
                                                    MetalTranslatorConfig::
                                                        GammaRenderTargetAsUnorm8()) {
  struct SchemaBits {
    uint32_t vendor_apple : 1;
    uint32_t bindless_resources_used : 1;
    uint32_t edram_rov_used : 1;
    uint32_t gamma_render_target_as_unorm8 : 1;
    uint32_t msc_bindless_resources_used : 1;
    uint32_t fetch_abi_revision : 8;
    uint32_t dxilconv_generation : 4;
    uint32_t reserved : 15;
  } schema = {};
  schema.vendor_apple = 1;
  schema.bindless_resources_used = MetalTranslatorConfig::kBindlessResourcesUsed;
  schema.edram_rov_used = MetalTranslatorConfig::kEdramRovUsed;
  schema.gamma_render_target_as_unorm8 = gamma_as_unorm8;
  schema.msc_bindless_resources_used =
      MetalTranslatorConfig::kMscBindlessResourcesUsed;
  schema.fetch_abi_revision = fetch_abi_revision;
  schema.dxilconv_generation = dxilconv_generation;

  MetalPipelineStorageIdentity identity;
  std::memcpy(&identity.translator_schema, &schema, sizeof(schema));
  identity.dxc_artifact_hash = 0x1111111111111111ull;
  identity.msc_artifact_hash = 0x2222222222222222ull;
  identity.device_hash = 0x3333333333333333ull;
  identity.os_build_hash = 0x4444444444444444ull;
  identity.converter_options_hash = 0x5555555555555555ull;
  return identity;
}

// The generation is held at the production value wherever a case is varying
// the fetch ABI, so the two boundaries stay independently readable.
MetalPipelineStorageIdentity IdentityWithFetchAbi(uint32_t fetch_abi_revision) {
  return IdentityWithSchema(fetch_abi_revision,
                            MetalTranslatorConfig::kDxilconvGeneration);
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

TEST_CASE("The gamma render-target switch defaults on and rolls back closed",
          "[gamma_blend][cache_identity]") {
  using metal::ParseMetalGammaRenderTargetLinearSetting;

  CHECK(ParseMetalGammaRenderTargetLinearSetting(nullptr) ==
        MetalGammaRenderTargetLinearSetting::kEnabled);
  CHECK(ParseMetalGammaRenderTargetLinearSetting("") ==
        MetalGammaRenderTargetLinearSetting::kMalformed);
  CHECK(ParseMetalGammaRenderTargetLinearSetting("0") ==
        MetalGammaRenderTargetLinearSetting::kDisabled);
  CHECK(ParseMetalGammaRenderTargetLinearSetting("1") ==
        MetalGammaRenderTargetLinearSetting::kEnabled);
  for (const char* malformed : {"00", "01", "2", "true", " 1", "1 "}) {
    CHECK(ParseMetalGammaRenderTargetLinearSetting(malformed) ==
          MetalGammaRenderTargetLinearSetting::kMalformed);
  }

  CHECK(MetalTranslatorConfig::GammaRenderTargetAsUnorm8(
      MetalGammaRenderTargetLinearSetting::kDisabled));
  CHECK_FALSE(MetalTranslatorConfig::GammaRenderTargetAsUnorm8(
      MetalGammaRenderTargetLinearSetting::kEnabled));
  CHECK(MetalTranslatorConfig::GammaRenderTargetAsUnorm8(
      MetalGammaRenderTargetLinearSetting::kMalformed));
}

TEST_CASE("The gamma host-storage selection changes durable shader identity",
          "[gamma_blend][cache_identity]") {
  const MetalPipelineStorageIdentity legacy = IdentityWithSchema(
      MetalTranslatorConfig::kFetchAbiRevision,
      MetalTranslatorConfig::kDxilconvGeneration, true);
  const MetalPipelineStorageIdentity linear = IdentityWithSchema(
      MetalTranslatorConfig::kFetchAbiRevision,
      MetalTranslatorConfig::kDxilconvGeneration, false);

  CHECK(legacy.translator_schema != linear.translator_schema);
  CHECK((legacy.translator_schema ^ linear.translator_schema) == (1u << 3));
}

TEST_CASE("The production fetch ABI revision is the current one",
          "[integer_scale][cache_identity]") {
  // If this fails, either the revision was bumped without updating the tests
  // below, or it was reverted and the stale-cache hazard is open again.
  // 5 = the clip-distance relabel became UNCONDITIONAL.
  // 6 = vertex fetches mask words at or past the fetch constant's size to 0
  //     (upstream Canary 9e9d3cdd3): identical ucode now emits different DXBC,
  //     and the metallib/pipeline records key on ucode + this schema, so a
  //     revision-5 cache would serve unmasked shaders.
  // 7 = stacked-texture fetches clamp the layer index for Inf/NaN coordinates
  //     (upstream Canary 04085efaa), again different DXBC for the same ucode.
  // 8 = texture_integer_scale_bits takes Canary's packed fixed conversion
  //     (d11950528..c3cd8617b): new fetch DXBC and a new meaning for the
  //     same system-constant bytes.
  // Each boundary is pinned by its own transition case below.
  CHECK(MetalTranslatorConfig::kFetchAbiRevision == 8);
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
    // Revision 4: the revision-3 artifact must not be served. Pinned at the
    // LITERAL boundary, not at the production revision, so this case keeps
    // testing the 3->4 transition after later bumps.
    MetalPipelineStorage rev4;
    REQUIRE(rev4.Open(root, kTitleId, IdentityWithFetchAbi(4)));
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


TEST_CASE("The revision-4 to revision-5 boundary discards, not filters",
          "[cache_identity][clip][geometry]") {
  // The clip-relabel boundary. Revision 3 INTRODUCED the relabel, but it
  // shipped behind XE_METAL_CLIP_RELABEL defaulting OFF, so every record
  // written at revisions 3 and 4 was in fact produced WITHOUT it. Now that the
  // relabel is unconditional those records describe different emitted geometry
  // stages for the same guest ucode, and the geometry record keys CANNOT see
  // the difference — the object-stage key is (kind, MetallibCacheKey(vertex),
  // topology) and the mesh-stage key is (kind, GeometryShaderKey). The revision
  // is the only thing preventing a warm pre-fix cache from serving a silently
  // wrong pipeline.
  //
  // Three legs, and the third is the point: a store that merely FILTERED reads
  // by identity would leave the poisoned bytes on disk to be resurrected by a
  // downgrade or an A/B rollback build. It must DISCARD.
  std::filesystem::path root = MakeScratchRoot("clip_relabel_transition");

  {
    MetalPipelineStorage rev4;
    REQUIRE(rev4.Open(root, kTitleId, IdentityWithFetchAbi(4)));
    REQUIRE(rev4.StoreMetallib(kMetallibKey, SampleMetallib()));
    rev4.Close();
    // Control: revision 4 serves its own artifact, or the rejection below
    // proves nothing.
    MetalPipelineStorage warm4;
    REQUIRE(warm4.Open(root, kTitleId, IdentityWithFetchAbi(4)));
    std::vector<uint8_t> served;
    REQUIRE(warm4.LoadMetallib(kMetallibKey, &served));
    CHECK(served == SampleMetallib());
    warm4.Close();
  }

  {
    // Production revision (5): the pre-relabel artifact must not be served.
    MetalPipelineStorage rev5;
    REQUIRE(rev5.Open(root, kTitleId,
                      IdentityWithFetchAbi(
                          MetalTranslatorConfig::kFetchAbiRevision)));
    std::vector<uint8_t> served;
    CHECK_FALSE(rev5.LoadMetallib(kMetallibKey, &served));
    rev5.Close();
  }

  {
    // Downgrade back to 4 AFTER the revision-5 open: still nothing. There is no
    // pre-relabel artifact left to resurrect.
    MetalPipelineStorage back4;
    REQUIRE(back4.Open(root, kTitleId, IdentityWithFetchAbi(4)));
    std::vector<uint8_t> served;
    CHECK_FALSE(back4.LoadMetallib(kMetallibKey, &served));
    back4.Close();
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

// ---------------------------------------------------------------------------
// adopt-xenios-shader-artifact-caching task 2.1, items 2 and 3: the dxilconv
// converter generation and the dxilconv option string.
// ---------------------------------------------------------------------------

TEST_CASE("The identity mirror matches the production packing",
          "[cache_identity][mirror]") {
  // The guard the old "kept textually close" comment only promised. Without
  // it, every case in this file compares mirror values to mirror values, so a
  // field added to ComputeMetalPipelineStorageIdentity's packing — as
  // dxilconv_generation was — leaves the whole file green while testing a
  // layout production does not use.
  //
  // A null device is fine: device_hash is the only field it feeds, and this
  // compares translator_schema.
  const MetalPipelineStorageIdentity production =
      metal::ComputeMetalPipelineStorageIdentity(nullptr);
  const MetalPipelineStorageIdentity mirrored =
      IdentityWithSchema(MetalTranslatorConfig::kFetchAbiRevision,
                         MetalTranslatorConfig::kDxilconvGeneration);
  INFO("production=" << std::hex << production.translator_schema
                     << " mirror=" << mirrored.translator_schema);
  CHECK(production.translator_schema == mirrored.translator_schema);

  // And the production identity must actually carry an option fingerprint. 0
  // is what a manifest written before the field existed holds, so a computed 0
  // would silently match every stale directory.
  CHECK(production.converter_options_hash != 0ull);
}

TEST_CASE("The mulsc rounding mode participates in the cache identity",
          "[cache_identity]") {
  // --mulsc_round_toward_zero changes the DXBC emitted for mulsc, so a cache
  // written under one setting must not be served under the other.
  const bool saved = cvars::mulsc_round_toward_zero;
  cvars::mulsc_round_toward_zero = false;
  const MetalPipelineStorageIdentity nearest =
      metal::ComputeMetalPipelineStorageIdentity(nullptr);
  cvars::mulsc_round_toward_zero = true;
  const MetalPipelineStorageIdentity toward_zero =
      metal::ComputeMetalPipelineStorageIdentity(nullptr);
  cvars::mulsc_round_toward_zero = saved;
  CHECK(nearest.translator_schema != toward_zero.translator_schema);
  CHECK(nearest != toward_zero);
}

TEST_CASE("The production dxilconv generation is the current one",
          "[cache_identity][dxilconv]") {
  // 1 = fresh IDxbcConverter per Convert (2026-08-02). A revert to reused
  // instances without a generation bump would silently re-serve reuse-era
  // artifacts, which is the failure this constant exists to make impossible.
  CHECK(MetalTranslatorConfig::kDxilconvGeneration == 1);
}

TEST_CASE("A cache written under a previous dxilconv generation is discarded",
          "[cache_identity][dxilconv]") {
  // Item 2. Generation 0 is the pre-fix wrapper: reused IDxbcConverter
  // instances carried call history and emitted CORRUPT quad-domain DXIL. The
  // metallib key carries no DXIL bytes and dxc_artifact_hash fingerprints
  // libdxilconv, not our wrapper, so NOTHING except this field distinguishes a
  // metallib built from corrupt DXIL from a clean one.
  //
  // Three legs, and the third is the point: filtering reads by identity would
  // leave the corrupt bytes on disk for a downgrade or an A/B rollback build
  // to resurrect. It must DISCARD.
  std::filesystem::path root = MakeScratchRoot("dxilconv_generation");
  const MetalPipelineStorageIdentity pre_fix = IdentityWithSchema(
      MetalTranslatorConfig::kFetchAbiRevision, /*dxilconv_generation=*/0);
  const MetalPipelineStorageIdentity post_fix = IdentityWithSchema(
      MetalTranslatorConfig::kFetchAbiRevision,
      MetalTranslatorConfig::kDxilconvGeneration);
  REQUIRE(pre_fix != post_fix);

  {
    MetalPipelineStorage gen0;
    REQUIRE(gen0.Open(root, kTitleId, pre_fix));
    REQUIRE(gen0.StoreMetallib(kMetallibKey, SampleMetallib()));
    gen0.Close();
    // Control: generation 0 serves its own artifact, or the rejection below
    // proves nothing.
    MetalPipelineStorage warm0;
    REQUIRE(warm0.Open(root, kTitleId, pre_fix));
    std::vector<uint8_t> served;
    REQUIRE(warm0.LoadMetallib(kMetallibKey, &served));
    CHECK(served == SampleMetallib());
    warm0.Close();
  }

  {
    MetalPipelineStorage gen1;
    REQUIRE(gen1.Open(root, kTitleId, post_fix));
    std::vector<uint8_t> served;
    CHECK_FALSE(gen1.LoadMetallib(kMetallibKey, &served));
    gen1.Close();
  }

  {
    MetalPipelineStorage back0;
    REQUIRE(back0.Open(root, kTitleId, pre_fix));
    std::vector<uint8_t> served;
    CHECK_FALSE(back0.LoadMetallib(kMetallibKey, &served));
    back0.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("The dxilconv generation does not alias the fetch ABI revision",
          "[cache_identity][dxilconv]") {
  // The two fields are adjacent in the packing. If the generation overlapped
  // the revision, a generation bump would read as a revision bump and — worse
  // — some (revision, generation) pair would collide with another, which is a
  // stale-serve arriving through the guard itself.
  const uint32_t base =
      IdentityWithSchema(0, 0).translator_schema;
  const uint32_t revision_only =
      IdentityWithSchema(0xFF, 0).translator_schema;
  const uint32_t generation_only =
      IdentityWithSchema(0, 0xF).translator_schema;

  const uint32_t revision_bits = base ^ revision_only;
  const uint32_t generation_bits = base ^ generation_only;
  INFO("revision bits=" << std::hex << revision_bits
                        << " generation bits=" << generation_bits);
  CHECK(revision_bits != 0u);
  CHECK(generation_bits != 0u);
  CHECK((revision_bits & generation_bits) == 0u);
  // Neither may touch the five configuration booleans in the low bits.
  CHECK((revision_bits & 0x1Fu) == 0u);
  CHECK((generation_bits & 0x1Fu) == 0u);

  // Every (revision, generation) pair in range must be a distinct schema —
  // the property a bit overlap would break.
  std::vector<uint32_t> seen;
  for (uint32_t revision = 0; revision < 256; ++revision) {
    for (uint32_t generation = 0; generation < 16; ++generation) {
      seen.push_back(IdentityWithSchema(revision, generation).translator_schema);
    }
  }
  std::sort(seen.begin(), seen.end());
  CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
  CHECK(seen.size() == 256u * 16u);
}

TEST_CASE("The dxilconv option string participates in the cache identity",
          "[cache_identity][dxilconv][options]") {
  // Item 3. The option string changes the emitted DXIL for byte-identical DXBC
  // (measured in metal_dxil_determinism_test.mm: 3644 B under the default vs
  // 4436 B under an empty string), and before this field it reached no cache
  // key at all — a warm cache written under one setting was served under
  // another.
  const char* saved = std::getenv(metal::kMetalDxbcToDxilOptionsEnvVar);
  const std::string saved_value = saved ? std::string(saved) : std::string();
  const bool had_override = saved != nullptr;
  auto restore = [&]() {
    if (had_override) {
      setenv(metal::kMetalDxbcToDxilOptionsEnvVar, saved_value.c_str(), 1);
    } else {
      unsetenv(metal::kMetalDxbcToDxilOptionsEnvVar);
    }
  };

  unsetenv(metal::kMetalDxbcToDxilOptionsEnvVar);
  REQUIRE(metal::MetalDxbcToDxilEffectiveOptions() ==
          std::string(metal::kMetalDxbcToDxilDefaultOptions));
  const uint64_t default_hash =
      metal::ComputeMetalPipelineStorageIdentity(nullptr)
          .converter_options_hash;

  setenv(metal::kMetalDxbcToDxilOptionsEnvVar, "-skip-container-parts -Zi", 1);
  const uint64_t overridden_hash =
      metal::ComputeMetalPipelineStorageIdentity(nullptr)
          .converter_options_hash;

  setenv(metal::kMetalDxbcToDxilOptionsEnvVar, "", 1);
  const uint64_t empty_hash =
      metal::ComputeMetalPipelineStorageIdentity(nullptr)
          .converter_options_hash;

  restore();

  CHECK(default_hash != overridden_hash);
  CHECK(default_hash != empty_hash);
  CHECK(overridden_hash != empty_hash);
  // No option value may hash to 0: that is the value a manifest written before
  // this field existed carries, and matching it would serve entries built
  // under an option string nothing recorded.
  CHECK(default_hash != 0ull);
  CHECK(overridden_hash != 0ull);
  CHECK(empty_hash != 0ull);
}

TEST_CASE("A cache written under different dxilconv options is discarded",
          "[cache_identity][dxilconv][options]") {
  // The identity check above proves the field moves; this proves the STORE
  // acts on it. Same three legs as the generation case, for the same reason.
  std::filesystem::path root = MakeScratchRoot("dxilconv_options");
  MetalPipelineStorageIdentity option_a = IdentityWithFetchAbi(
      MetalTranslatorConfig::kFetchAbiRevision);
  option_a.converter_options_hash = 0xA1A1A1A1A1A1A1A1ull;
  MetalPipelineStorageIdentity option_b = option_a;
  option_b.converter_options_hash = 0xB2B2B2B2B2B2B2B2ull;
  REQUIRE(option_a != option_b);
  CHECK(option_a.DescribeMismatch(option_b) == "dxilconv options");

  {
    MetalPipelineStorage a;
    REQUIRE(a.Open(root, kTitleId, option_a));
    REQUIRE(a.StoreMetallib(kMetallibKey, SampleMetallib()));
    a.Close();
    MetalPipelineStorage warm_a;
    REQUIRE(warm_a.Open(root, kTitleId, option_a));
    std::vector<uint8_t> served;
    REQUIRE(warm_a.LoadMetallib(kMetallibKey, &served));
    CHECK(served == SampleMetallib());
    warm_a.Close();
  }

  {
    MetalPipelineStorage b;
    REQUIRE(b.Open(root, kTitleId, option_b));
    std::vector<uint8_t> served;
    CHECK_FALSE(b.LoadMetallib(kMetallibKey, &served));
    b.Close();
  }

  {
    MetalPipelineStorage back_a;
    REQUIRE(back_a.Open(root, kTitleId, option_a));
    std::vector<uint8_t> served;
    CHECK_FALSE(back_a.LoadMetallib(kMetallibKey, &served));
    back_a.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A manifest written before the option field existed is discarded",
          "[cache_identity][dxilconv][options]") {
  // The upgrade path. Those entries were produced under an option string
  // nothing recorded, so they cannot be trusted by a build that now records
  // one — the same reasoning as every other boundary in this file.
  std::filesystem::path root = MakeScratchRoot("dxilconv_options_upgrade");
  MetalPipelineStorageIdentity pre_field =
      IdentityWithFetchAbi(MetalTranslatorConfig::kFetchAbiRevision);
  pre_field.converter_options_hash = 0ull;  // what the old `reserved` slot held

  {
    MetalPipelineStorage old_build;
    REQUIRE(old_build.Open(root, kTitleId, pre_field));
    REQUIRE(old_build.StoreMetallib(kMetallibKey, SampleMetallib()));
    old_build.Close();
  }

  {
    // A build that computes a real option fingerprint must not be served it.
    MetalPipelineStorageIdentity current = pre_field;
    current.converter_options_hash =
        metal::ComputeMetalPipelineStorageIdentity(nullptr)
            .converter_options_hash;
    REQUIRE(current.converter_options_hash != 0ull);
    MetalPipelineStorage new_build;
    REQUIRE(new_build.Open(root, kTitleId, current));
    std::vector<uint8_t> served;
    CHECK_FALSE(new_build.LoadMetallib(kMetallibKey, &served));
    new_build.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
