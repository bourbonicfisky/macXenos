/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

// DXIL artifact store conformance (adopt-xenios-shader-artifact-caching 2.2).
//
// The store has FOUR outcomes that must never collapse into one another, and
// each case below drives the real MetalDxilStore over a real directory rather
// than testing the predicate in isolation — a store that ignored its own
// identity would pass a pure equality check:
//
//   store-manifest mismatch  the whole directory is discarded (not filtered).
//   artifact-key mismatch    an ordinary miss, including under a deliberate
//                            filename collision.
//   payload corruption       REJECTED and the entry deleted, counted apart
//                            from misses.
//   absent / unreadable      ordinary misses.
//
// The identity is the point of this store, so the cases also pin what it does
// NOT contain: an MSC, device or OS-build change must leave it intact, because
// surviving those is the only reason it exists (task 1.1 measured that it saves
// nothing on an ordinary warm launch).

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "third_party/fmt/include/fmt/format.h"
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
using metal::MetalTranslatorConfig;

std::filesystem::path MakeScratchRoot(const char* name) {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() / "xe_dxil_store" / name;
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  return root;
}

MetalDxilStoreIdentity FixedIdentity() {
  MetalDxilStoreIdentity identity;
  identity.dxc_artifact_hash = 0xD1D1D1D1D1D1D1D1ull;
  identity.converter_options_hash = 0xC0FFEE00C0FFEE00ull;
  identity.dxilconv_generation = MetalTranslatorConfig::kDxilconvGeneration;
  return identity;
}

std::vector<uint8_t> Dxbc(uint8_t seed, size_t size = 512) {
  std::vector<uint8_t> out(size);
  out[0] = 'D'; out[1] = 'X'; out[2] = 'B'; out[3] = 'C';
  for (size_t i = 4; i < size; ++i) {
    out[i] = static_cast<uint8_t>((i * 7u + seed * 13u) & 0xFF);
  }
  return out;
}

std::vector<uint8_t> Dxil(uint8_t seed, size_t size = 1024) {
  std::vector<uint8_t> out(size);
  for (size_t i = 0; i < size; ++i) {
    out[i] = static_cast<uint8_t>((i * 11u + seed * 29u + 3u) & 0xFF);
  }
  return out;
}

}  // namespace

TEST_CASE("A stored DXIL artifact round-trips", "[dxil_store]") {
  // The positive control. Every rejection case below is meaningless without it.
  std::filesystem::path root = MakeScratchRoot("roundtrip");
  const std::vector<uint8_t> dxbc = Dxbc(1);
  const std::vector<uint8_t> dxil = Dxil(1);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    REQUIRE(store.Store(key, dxil));
    store.Close();
  }
  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    std::vector<uint8_t> served;
    REQUIRE(store.Load(key, &served) == MetalDxilLoadResult::kHit);
    CHECK(served == dxil);
    CHECK(store.hits() == 1);
    CHECK(store.misses() == 0);
    CHECK(store.corrupt_rejects() == 0);
    store.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A different DXBC payload is an ordinary miss", "[dxil_store]") {
  std::filesystem::path root = MakeScratchRoot("different_payload");
  const std::vector<uint8_t> dxbc_a = Dxbc(1);
  const std::vector<uint8_t> dxbc_b = Dxbc(2);
  REQUIRE(dxbc_a != dxbc_b);

  MetalDxilStore store;
  REQUIRE(store.Open(root, FixedIdentity()));
  REQUIRE(store.Store(MetalDxilArtifactKey::ForDxbc(dxbc_a), Dxil(1)));

  std::vector<uint8_t> served;
  CHECK(store.Load(MetalDxilArtifactKey::ForDxbc(dxbc_b), &served) ==
        MetalDxilLoadResult::kMissAbsent);
  CHECK(store.corrupt_rejects() == 0);  // a miss is not an incident
  store.Close();

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A filename collision is a miss, not a wrong artifact",
          "[dxil_store][keyecho]") {
  // THE case the key echo exists for. The filename is a 64-bit hash, so two
  // distinct keys can in principle name the same file. Rather than wait for a
  // natural collision, this forges one: write payload A's record, then rename
  // it onto payload B's filename. Without the echo, B would be served a
  // perfectly well-formed artifact belonging to A — the exact shape of the
  // dxilconv corruption incident, where caches served valid-looking artifacts
  // that did not correspond to their key.
  std::filesystem::path root = MakeScratchRoot("collision");
  const std::vector<uint8_t> dxbc_a = Dxbc(1);
  const std::vector<uint8_t> dxbc_b = Dxbc(2);
  const MetalDxilArtifactKey key_a = MetalDxilArtifactKey::ForDxbc(dxbc_a);
  const MetalDxilArtifactKey key_b = MetalDxilArtifactKey::ForDxbc(dxbc_b);
  REQUIRE_FALSE(key_a == key_b);
  REQUIRE(key_a.FileKey() != key_b.FileKey());

  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    REQUIRE(store.Store(key_a, Dxil(1)));
    store.Close();
  }

  // Forge the collision on disk.
  const std::filesystem::path path_a =
      root / "metal" / "dxil" / fmt::format("{:016X}.dxil", key_a.FileKey());
  const std::filesystem::path path_b =
      root / "metal" / "dxil" / fmt::format("{:016X}.dxil", key_b.FileKey());
  std::error_code ec;
  REQUIRE(std::filesystem::exists(path_a));
  std::filesystem::rename(path_a, path_b, ec);
  REQUIRE_FALSE(ec);

  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    std::vector<uint8_t> served;
    // The record is intact and its payload hash is correct — only the KEY is
    // wrong. Nothing but the echo distinguishes this from a hit.
    CHECK(store.Load(key_b, &served) == MetalDxilLoadResult::kMissKeyMismatch);
    CHECK(served.empty());
    CHECK(store.corrupt_rejects() == 0);  // wrong key, not corrupt bytes
    store.Close();
  }

  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A corrupted payload is rejected and deleted, not served or missed",
          "[dxil_store][corruption]") {
  // The third outcome. A flipped byte in the payload leaves the key correct and
  // the header well-formed, so this is neither a miss nor a hit: it is a stored
  // artifact that is WRONG. It must be reported, deleted so the next lookup
  // rebuilds, and counted apart from misses — a cold cache increments misses
  // too, and burying an incident inside that number is how it stays unnoticed.
  std::filesystem::path root = MakeScratchRoot("corrupt");
  const std::vector<uint8_t> dxbc = Dxbc(3);
  const std::vector<uint8_t> dxil = Dxil(3);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    REQUIRE(store.Store(key, dxil));
    store.Close();
  }

  const std::filesystem::path path =
      root / "metal" / "dxil" / fmt::format("{:016X}.dxil", key.FileKey());
  REQUIRE(std::filesystem::exists(path));
  {
    // Flip one byte deep in the payload, past every header field.
    FILE* file = fopen(path.string().c_str(), "r+b");
    REQUIRE(file != nullptr);
    const long offset = long(std::filesystem::file_size(path)) - 16;
    REQUIRE(offset > 0);
    fseek(file, offset, SEEK_SET);
    int byte = fgetc(file);
    REQUIRE(byte != EOF);
    fseek(file, offset, SEEK_SET);
    fputc(byte ^ 0x01, file);
    fclose(file);
  }

  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    std::vector<uint8_t> served;
    CHECK(store.Load(key, &served) == MetalDxilLoadResult::kRejectedCorrupt);
    CHECK(served.empty());
    CHECK(store.corrupt_rejects() == 1);
    CHECK(store.hits() == 0);
    CHECK(store.misses() == 0);  // NOT counted as a miss
    // Deleted, so the next lookup rebuilds instead of repeating the rejection
    // every launch forever.
    CHECK_FALSE(std::filesystem::exists(path));
    CHECK(store.Load(key, &served) == MetalDxilLoadResult::kMissAbsent);
    store.Close();
  }

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A truncated record is an ordinary miss", "[dxil_store]") {
  std::filesystem::path root = MakeScratchRoot("truncated");
  const std::vector<uint8_t> dxbc = Dxbc(4);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);
  {
    MetalDxilStore store;
    REQUIRE(store.Open(root, FixedIdentity()));
    REQUIRE(store.Store(key, Dxil(4)));
    store.Close();
  }
  const std::filesystem::path path =
      root / "metal" / "dxil" / fmt::format("{:016X}.dxil", key.FileKey());
  std::error_code ec;
  std::filesystem::resize_file(path, 8, ec);
  REQUIRE_FALSE(ec);

  MetalDxilStore store;
  REQUIRE(store.Open(root, FixedIdentity()));
  std::vector<uint8_t> served;
  CHECK(store.Load(key, &served) == MetalDxilLoadResult::kMissUnreadable);
  store.Close();
  std::filesystem::remove_all(root, ec);
}

TEST_CASE("A converter-contract change discards the whole DXIL directory",
          "[dxil_store][identity]") {
  // The first outcome. Each of the three identity fields must discard, and the
  // third leg proves DISCARD rather than filter: a store that merely refused to
  // read would leave artifacts produced under the old contract on disk for a
  // rollback build to resurrect.
  const std::vector<uint8_t> dxbc = Dxbc(5);
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  struct Mutation {
    const char* name;
    MetalDxilStoreIdentity identity;
  };
  std::vector<Mutation> mutations;
  {
    MetalDxilStoreIdentity dylib = FixedIdentity();
    dylib.dxc_artifact_hash ^= 0xFFull;
    mutations.push_back({"libdxilconv", dylib});
    MetalDxilStoreIdentity options = FixedIdentity();
    options.converter_options_hash ^= 0xFFull;
    mutations.push_back({"dxilconv options", options});
    MetalDxilStoreIdentity generation = FixedIdentity();
    generation.dxilconv_generation += 1;
    mutations.push_back({"dxilconv generation", generation});
  }

  for (const Mutation& mutation : mutations) {
    INFO("mutated field: " << mutation.name);
    std::filesystem::path root = MakeScratchRoot("identity_discard");
    REQUIRE(FixedIdentity() != mutation.identity);
    CHECK(FixedIdentity().DescribeMismatch(mutation.identity) ==
          std::string(mutation.name));

    {
      MetalDxilStore store;
      REQUIRE(store.Open(root, FixedIdentity()));
      REQUIRE(store.Store(key, Dxil(5)));
      store.Close();
      // Control: the original identity serves it, or the rejection proves
      // nothing.
      MetalDxilStore warm;
      REQUIRE(warm.Open(root, FixedIdentity()));
      std::vector<uint8_t> served;
      REQUIRE(warm.Load(key, &served) == MetalDxilLoadResult::kHit);
      warm.Close();
    }
    {
      MetalDxilStore changed;
      REQUIRE(changed.Open(root, mutation.identity));
      std::vector<uint8_t> served;
      CHECK(changed.Load(key, &served) == MetalDxilLoadResult::kMissAbsent);
      changed.Close();
    }
    {
      // Back to the original: still nothing. The directory was DISCARDED.
      MetalDxilStore back;
      REQUIRE(back.Open(root, FixedIdentity()));
      std::vector<uint8_t> served;
      CHECK(back.Load(key, &served) == MetalDxilLoadResult::kMissAbsent);
      back.Close();
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
}

TEST_CASE("The DXIL identity omits what DXIL does not depend on",
          "[dxil_store][identity]") {
  // The reason this store exists at all. A metallib is compiled against a GPU
  // family and an OS build and is produced by the Metal Shader Converter; DXIL
  // is none of those things. If any of them reached this identity, the store
  // would be discarded on exactly the launches it was built to survive, and
  // task 1.1 measured that it saves nothing on the launches it would be left
  // with.
  //
  // Guarded two ways, because either alone has a hole. The size catches a
  // field added silently; the mismatch enumeration catches a field added AND
  // wired into equality. Both are meant to be updated deliberately — that is
  // the point. 32 = uint32 + pad + uint64 + uint64 + uint32 + pad on arm64;
  // it is NOT the sum of the member sizes, which is what a first attempt at
  // this assertion wrongly used.
  MetalDxilStoreIdentity identity = FixedIdentity();
  MetalDxilStoreIdentity same = FixedIdentity();
  CHECK(identity == same);
  CHECK(sizeof(MetalDxilStoreIdentity) == 32u);

  // Every field differing must be NAMED, and there must be exactly four of
  // them. A fifth would mean something joined this identity that DXIL does not
  // depend on — the failure mode this store exists to avoid.
  MetalDxilStoreIdentity all_different;
  all_different.format_version = identity.format_version + 1;
  all_different.dxc_artifact_hash = identity.dxc_artifact_hash ^ 0xFFull;
  all_different.converter_options_hash =
      identity.converter_options_hash ^ 0xFFull;
  all_different.dxilconv_generation = identity.dxilconv_generation + 1;
  const std::string described = identity.DescribeMismatch(all_different);
  INFO("described: " << described);
  size_t named = described.empty() ? 0 : 1;
  for (char c : described) {
    if (c == ',') ++named;
  }
  CHECK(named == 4u);
  CHECK(described.find("device") == std::string::npos);
  CHECK(described.find("OS build") == std::string::npos);
  CHECK(described.find("MSC") == std::string::npos);

  // And a full pipeline-storage identity carries strictly more: the DXIL store
  // must not be derivable from it by simple copy.
  CHECK(sizeof(metal::MetalPipelineStorageIdentity) >
        sizeof(MetalDxilStoreIdentity));
}

TEST_CASE("The artifact key covers the converter contract too",
          "[dxil_store][identity]") {
  // Manifest AND key, not either. The manifest is the coarse gate that discards
  // the directory; the key is the fine gate that misses one entry. XeniOS puts
  // these only in the key, which means a converter change there silently misses
  // every entry and leaves the stale bytes on disk forever.
  const std::vector<uint8_t> dxbc = Dxbc(6);
  MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  MetalDxilArtifactKey other_options = key;
  other_options.converter_options_hash ^= 0x1ull;
  CHECK_FALSE(key == other_options);
  CHECK(key.FileKey() != other_options.FileKey());

  MetalDxilArtifactKey other_generation = key;
  other_generation.dxilconv_generation += 1;
  CHECK_FALSE(key == other_generation);
  CHECK(key.FileKey() != other_generation.FileKey());

  // Size participates, so a truncated DXBC cannot alias a complete one even if
  // its content hash somehow matched.
  MetalDxilArtifactKey other_size = key;
  other_size.dxbc_size -= 1;
  CHECK_FALSE(key == other_size);
  CHECK(key.FileKey() != other_size.FileKey());

  // Distinct payloads produce distinct keys — the premise every miss case
  // above depends on.
  CHECK_FALSE(MetalDxilArtifactKey::ForDxbc(Dxbc(7)) ==
              MetalDxilArtifactKey::ForDxbc(Dxbc(8)));
}

TEST_CASE("The store bounds its own admission", "[dxil_store][cap]") {
  // It lives outside the per-title accounting, so it needs its own bound; and
  // reaching the bound must stop NEW admissions without disturbing what is
  // already there.
  std::filesystem::path root = MakeScratchRoot("cap");
  MetalDxilStore store;
  REQUIRE(store.Open(root, FixedIdentity()));
  CHECK(store.CanAdmit(1024));
  // Far beyond any configured cap.
  CHECK_FALSE(store.CanAdmit(uint64_t(1) << 60));
  const std::vector<uint8_t> dxbc = Dxbc(9);
  REQUIRE(store.Store(MetalDxilArtifactKey::ForDxbc(dxbc), Dxil(9)));
  CHECK(store.stores() == 1);
  CHECK(store.bytes_admitted() > 0);
  store.Close();
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
