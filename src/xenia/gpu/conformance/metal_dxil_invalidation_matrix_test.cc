/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// Dependency-specific invalidation matrix for the two durable stores
// (adopt-xenios-shader-artifact-caching task 2.4, rewritten 2026-08-05).
//
// The task's ORIGINAL rule — "flip a translator constant, all kinds miss" — is
// wrong now and would enshrine the bug this change exists to avoid. It was
// written when every artifact lived under one per-title manifest. The DXIL
// store deliberately does not, because DXIL and a metallib depend on different
// things:
//
//                              | metallib | DXIL
//   ---------------------------+----------+------
//   guest ucode / emitted DXBC |   yes    | yes  (via the DXBC bytes)
//   dxilconv lib/options/gen   |   yes    | yes
//   Metal Shader Converter     |   yes    | NO
//   Metal device / GPU family  |   yes    | NO
//   OS build                   |   yes    | NO
//   translator schema / ABI    |   yes    | NO   (except as it changes DXBC)
//
// "All kinds miss" would make the store worthless on precisely the launches it
// exists for. What must hold instead is that each store invalidates on ITS OWN
// dependencies and on nothing else — and, critically, that a metallib
// invalidation LEAVES VALID DXIL HITTING. These cases drive both real stores
// over one cache root, because that co-residency is the thing under test.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/metal/metal_dxil_store.h"
#include "xenia/gpu/metal/metal_pipeline_storage.h"

namespace xe {
namespace gpu {
namespace test {

namespace {

using metal::MetalDxilArtifactKey;
using metal::MetalDxilLoadResult;
using metal::MetalDxilStore;
using metal::MetalDxilStoreIdentity;
using metal::MetalPipelineStorage;
using metal::MetalPipelineStorageIdentity;
using metal::MetalTranslatorConfig;

constexpr uint32_t kTitleId = 0x53450848;  // Sonic Generations.
constexpr uint64_t kMetallibKey = 0x5150515051505150ull;

std::filesystem::path MakeScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "xe_dxil_matrix" / name;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

MetalPipelineStorageIdentity TitleIdentity() {
  MetalPipelineStorageIdentity identity;
  identity.translator_schema = 0xABCD0001u;
  identity.dxc_artifact_hash = 0xD1D1D1D1D1D1D1D1ull;
  identity.msc_artifact_hash = 0x3C3C3C3C3C3C3C3Cull;
  identity.device_hash = 0xDE0DE0DE0DE0DE0Dull;
  identity.os_build_hash = 0x0507B01D0507B01Dull;
  identity.converter_options_hash = 0xC0FFEE00C0FFEE00ull;
  return identity;
}

MetalDxilStoreIdentity DxilIdentity() {
  MetalDxilStoreIdentity identity;
  // The three shared facts are taken from the SAME values the title identity
  // uses, so a divergence between the two stores' views is a test failure
  // rather than a coincidence.
  identity.dxc_artifact_hash = TitleIdentity().dxc_artifact_hash;
  identity.converter_options_hash = TitleIdentity().converter_options_hash;
  identity.dxilconv_generation = MetalTranslatorConfig::kDxilconvGeneration;
  return identity;
}

const std::vector<uint8_t>& SampleMetallib() {
  static const std::vector<uint8_t> bytes = {0xFA, 0xCE, 0xB0, 0x0C, 0x11, 0x22};
  return bytes;
}

std::vector<uint8_t> Dxbc(uint8_t seed) {
  std::vector<uint8_t> out(256);
  out[0] = 'D'; out[1] = 'X'; out[2] = 'B'; out[3] = 'C';
  for (size_t i = 4; i < out.size(); ++i) {
    out[i] = static_cast<uint8_t>((i * 5u + seed * 37u) & 0xFF);
  }
  return out;
}

std::vector<uint8_t> Dxil(uint8_t seed) {
  std::vector<uint8_t> out(600);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<uint8_t>((i * 3u + seed * 41u + 9u) & 0xFF);
  }
  return out;
}

// Seeds both stores in one cache root and closes them.
void SeedBoth(const std::filesystem::path& root,
              const MetalDxilArtifactKey& dxil_key,
              const std::vector<uint8_t>& dxil) {
  MetalPipelineStorage title;
  REQUIRE(title.Open(root, kTitleId, TitleIdentity()));
  REQUIRE(title.StoreMetallib(kMetallibKey, SampleMetallib()));
  title.Close();

  MetalDxilStore dxil_store;
  REQUIRE(dxil_store.Open(root, DxilIdentity()));
  REQUIRE(dxil_store.Store(dxil_key, dxil));
  dxil_store.Close();
}

}  // namespace

TEST_CASE("Both stores serve their own artifacts after seeding",
          "[dxil_matrix]") {
  // The positive control for the whole matrix. Every "survives" claim below is
  // vacuous without it.
  std::filesystem::path root = MakeScratchRoot("control");
  const std::vector<uint8_t> dxbc = Dxbc(1);
  const std::vector<uint8_t> dxil = Dxil(1);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);
  SeedBoth(root, key, dxil);

  MetalPipelineStorage title;
  REQUIRE(title.Open(root, kTitleId, TitleIdentity()));
  std::vector<uint8_t> served_metallib;
  CHECK(title.LoadMetallib(kMetallibKey, &served_metallib));
  CHECK(served_metallib == SampleMetallib());
  title.Close();

  MetalDxilStore dxil_store;
  REQUIRE(dxil_store.Open(root, DxilIdentity()));
  std::vector<uint8_t> served_dxil;
  CHECK(dxil_store.Load(key, &served_dxil) == MetalDxilLoadResult::kHit);
  CHECK(served_dxil == dxil);
  dxil_store.Close();

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A metallib-only invalidation leaves DXIL hitting",
          "[dxil_matrix][survival]") {
  // THE case this store exists for, and the one the original 2.4 rule would
  // have forbidden. Four facts invalidate a metallib without touching what DXIL
  // depends on. After each, the title directory must be discarded AND the DXIL
  // artifact must still hit — if it did not, the store would be paying storage
  // and lookup cost for nothing on exactly the launches that justify it.
  struct Mutation {
    const char* name;
    MetalPipelineStorageIdentity identity;
  };
  std::vector<Mutation> mutations;
  {
    MetalPipelineStorageIdentity msc = TitleIdentity();
    msc.msc_artifact_hash ^= 0xFFull;
    mutations.push_back({"MSC upgrade", msc});
    MetalPipelineStorageIdentity device = TitleIdentity();
    device.device_hash ^= 0xFFull;
    mutations.push_back({"different GPU", device});
    MetalPipelineStorageIdentity os = TitleIdentity();
    os.os_build_hash ^= 0xFFull;
    mutations.push_back({"OS build update", os});
    MetalPipelineStorageIdentity schema = TitleIdentity();
    schema.translator_schema ^= 0xFFull;
    mutations.push_back({"translator schema / fetch ABI bump", schema});
  }

  const std::vector<uint8_t> dxbc = Dxbc(2);
  const std::vector<uint8_t> dxil = Dxil(2);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  for (const Mutation& mutation : mutations) {
    INFO("metallib invalidated by: " << mutation.name);
    std::filesystem::path root = MakeScratchRoot("survival");
    SeedBoth(root, key, dxil);
    REQUIRE(TitleIdentity() != mutation.identity);

    // The metallib must be gone...
    {
      MetalPipelineStorage title;
      REQUIRE(title.Open(root, kTitleId, mutation.identity));
      std::vector<uint8_t> served;
      CHECK_FALSE(title.LoadMetallib(kMetallibKey, &served));
      title.Close();
    }
    // ...and the DXIL must still be there. Opened AFTER the title store's
    // discard-and-rebuild, because that is the ordering a real launch has and
    // the ordering in which a wholesale remove_all could have taken the DXIL
    // directory with it.
    {
      MetalDxilStore dxil_store;
      REQUIRE(dxil_store.Open(root, DxilIdentity()));
      std::vector<uint8_t> served;
      CHECK(dxil_store.Load(key, &served) == MetalDxilLoadResult::kHit);
      CHECK(served == dxil);
      CHECK(dxil_store.hits() == 1);
      dxil_store.Close();
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
}

TEST_CASE("A DXIL-relevant change invalidates DXIL and the metallib together",
          "[dxil_matrix][invalidation]") {
  // The converse. dxilconv's library, options and generation are the only
  // facts BOTH stores depend on, so a change to any of them must take out
  // both — a metallib built from reuse-era corrupt DXIL is exactly as unusable
  // as the DXIL itself.
  struct Mutation {
    const char* name;
    MetalPipelineStorageIdentity title;
    MetalDxilStoreIdentity dxil;
  };
  std::vector<Mutation> mutations;
  {
    Mutation lib{"libdxilconv", TitleIdentity(), DxilIdentity()};
    lib.title.dxc_artifact_hash ^= 0xFFull;
    lib.dxil.dxc_artifact_hash ^= 0xFFull;
    mutations.push_back(lib);
    Mutation options{"dxilconv options", TitleIdentity(), DxilIdentity()};
    options.title.converter_options_hash ^= 0xFFull;
    options.dxil.converter_options_hash ^= 0xFFull;
    mutations.push_back(options);
    Mutation generation{"dxilconv generation", TitleIdentity(), DxilIdentity()};
    // The generation lives inside translator_schema for the title store and as
    // its own field for the DXIL store; both move together in production.
    generation.title.translator_schema ^= 0x2000u;
    generation.dxil.dxilconv_generation += 1;
    mutations.push_back(generation);
  }

  const std::vector<uint8_t> dxbc = Dxbc(3);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  for (const Mutation& mutation : mutations) {
    INFO("changed: " << mutation.name);
    std::filesystem::path root = MakeScratchRoot("both_invalidate");
    SeedBoth(root, key, Dxil(3));

    {
      MetalPipelineStorage title;
      REQUIRE(title.Open(root, kTitleId, mutation.title));
      std::vector<uint8_t> served;
      CHECK_FALSE(title.LoadMetallib(kMetallibKey, &served));
      title.Close();
    }
    {
      MetalDxilStore dxil_store;
      REQUIRE(dxil_store.Open(root, mutation.dxil));
      std::vector<uint8_t> served;
      CHECK(dxil_store.Load(key, &served) == MetalDxilLoadResult::kMissAbsent);
      dxil_store.Close();
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
}

TEST_CASE("Changed source DXBC is a per-artifact miss, not a store wipe",
          "[dxil_matrix][invalidation]") {
  // A translator change that alters emitted DXBC must miss for THAT artifact
  // and leave every other artifact intact. Wiping the store would throw away
  // reusable DXIL for every shader the change did not touch — which is what
  // "all kinds miss" would have mandated.
  std::filesystem::path root = MakeScratchRoot("dxbc_change");
  const std::vector<uint8_t> old_dxbc = Dxbc(4);
  const std::vector<uint8_t> new_dxbc = Dxbc(5);   // the changed shader
  const std::vector<uint8_t> other_dxbc = Dxbc(6); // an untouched shader
  const MetalDxilArtifactKey old_key = MetalDxilArtifactKey::ForDxbc(old_dxbc);
  const MetalDxilArtifactKey new_key = MetalDxilArtifactKey::ForDxbc(new_dxbc);
  const MetalDxilArtifactKey other_key =
      MetalDxilArtifactKey::ForDxbc(other_dxbc);

  MetalDxilStore store;
  REQUIRE(store.Open(root, DxilIdentity()));
  REQUIRE(store.Store(old_key, Dxil(4)));
  REQUIRE(store.Store(other_key, Dxil(6)));

  std::vector<uint8_t> served;
  // The changed shader misses...
  CHECK(store.Load(new_key, &served) == MetalDxilLoadResult::kMissAbsent);
  // ...the untouched one still hits...
  CHECK(store.Load(other_key, &served) == MetalDxilLoadResult::kHit);
  CHECK(served == Dxil(6));
  // ...and so does the pre-change artifact, which is correct: a build that
  // still emits that DXBC (a rollback, an A/B arm, another title) may reuse it.
  CHECK(store.Load(old_key, &served) == MetalDxilLoadResult::kHit);
  store.Close();

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("The DXIL store reclaims space instead of only refusing to grow",
          "[dxil_matrix][cap]") {
  // Bounded storage as a PROMOTION criterion is not satisfied by a ceiling
  // alone. Because this store survives the invalidations that discard the
  // per-title one, superseded artifacts accumulate forever; with only a cap it
  // would fill with entries no build will ask for again and then stop caching
  // while still occupying its full budget — bounded, and useless.
  //
  // Driven at a 1 MiB cap so the case runs in milliseconds.
  std::filesystem::path root = MakeScratchRoot("reclaim");
  const uint64_t saved_cap = cvars::metal_dxil_cache_cap_mib;
  cvars::metal_dxil_cache_cap_mib = 1;

  std::vector<uint8_t> big(64 * 1024, 0xAB);
  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, DxilIdentity()));
    for (int i = 0; i < 24; ++i) {
      std::vector<uint8_t> dxbc = Dxbc(uint8_t(100 + i));
      // Distinct payloads so nothing dedups.
      big[0] = uint8_t(i);
      store.Store(MetalDxilArtifactKey::ForDxbc(dxbc), big);
    }
    store.Close();
  }

  const auto count_entries = [&root]() {
    size_t n = 0;
    std::error_code ec;
    for (const auto& entry :
         std::filesystem::directory_iterator(root / "metal" / "dxil", ec)) {
      if (!ec && entry.path().extension() == ".dxil") {
        ++n;
      }
    }
    return n;
  };
  const size_t before = count_entries();
  INFO("entries before reopen: " << before);
  // Admission stopped at the ceiling, so the store is now sitting AT the cap
  // with no headroom. This is the state a ceiling-only policy is permanently
  // stuck in.
  REQUIRE(before > 0);

  {
    // Reopening runs reclamation.
    MetalDxilStore store;
    REQUIRE(store.Open(root, DxilIdentity()));
    const uint64_t cap = 1ull * 1024ull * 1024ull;
    CHECK(store.bytes_on_open() <= cap - cap / 4);
    // It must actually have evicted, not merely stopped admitting.
    CHECK(store.evicted_entries() > 0);
    // And it must still be able to admit — the property a ceiling-only policy
    // loses.
    CHECK(store.CanAdmit(64 * 1024));
    CHECK(count_entries() < before);  // reclaimed, not merely refused
    store.Close();
  }

  cvars::metal_dxil_cache_cap_mib = saved_cap;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
