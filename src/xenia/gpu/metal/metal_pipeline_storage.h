/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_PIPELINE_STORAGE_H_
#define XENIA_GPU_METAL_METAL_PIPELINE_STORAGE_H_

// Durable pipeline-storage foundation (task group 3a of
// openspec/changes/implement-metal-shader-pipeline-lifetime).
//
// This owns the parts BOTH durable mechanisms share and neither can be built
// without:
//
//   M1  a durable metallib store       (translation: 50-97% of measured cost)
//   M2  an MTLBinaryArchive            (host pipeline-state: 1.7-47.5%)
//
// It deliberately does not know what an entry contains. It answers: where does
// this title's storage live, is it trustworthy, may another artifact be
// admitted, and how does a directory get published or discarded safely.
//
// WHY IDENTITY IS BUILT FIRST (design D4): an entry written under an incomplete
// identity is indistinguishable from a correct one. Provenance is therefore not
// safely retrofittable, and neither reference meets the bar — Edge keys on
// ucode_hash + modification + stage alone, guarded by a hand-bumped constant.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"

// 3.1 — one exact rollback switch per mechanism, both default OFF.
DECLARE_bool(metal_pipeline_metallib_cache);
DECLARE_bool(metal_pipeline_binary_archive);

namespace xe {
namespace gpu {
namespace metal {

// The single source of truth for the translator configuration.
// MetalPipelineCache::Initialize constructs the translators from these, and the
// storage identity below fingerprints them, so the two cannot drift. Before
// this existed the values were duplicated, and a change to one without the
// other would have served a provably wrong shader under an unchanged cache key
// — flipping kGammaRenderTargetAsUnorm8 alone is the difference behind the JC2
// black-gameplay bug.
struct MetalTranslatorConfig {
  static constexpr bool kBindlessResourcesUsed = true;
  static constexpr bool kEdramRovUsed = false;
  static constexpr bool kGammaRenderTargetAsUnorm8 = true;
  static constexpr bool kMscBindlessResourcesUsed = true;
};

// Everything that can change generated output for byte-identical guest ucode.
// A mismatch in ANY field discards the whole title directory rather than
// partially trusting it.
struct MetalPipelineStorageIdentity {
  // 3.3e — bumped when the on-disk layout changes, so a format change
  // invalidates without relying on any other field being right.
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;

  // 3.3b — the translator SCHEMA, not merely its revision. MetalPipelineCache
  // builds DxbcShaderTranslator with a vendor plus three booleans and the MSC
  // converter with one more; each changes emitted code for identical ucode.
  // Flipping gamma_render_target_as_unorm8 alone is the difference behind the
  // JC2 black-gameplay bug, so a schema change under an unchanged key would
  // serve a provably wrong shader.
  uint32_t translator_schema = 0;

  // 3.3c — hashes of the ACTUAL translator binaries, not release version
  // strings. third_party/DirectXShaderCompiler carries an uncommitted local
  // patch, so a version string cannot distinguish patched from unpatched.
  uint64_t dxc_artifact_hash = 0;
  uint64_t msc_artifact_hash = 0;

  // 3.3d — a metallib is compiled against both a GPU family and an OS.
  uint64_t device_hash = 0;   // registry id + name + GPU family
  uint64_t os_build_hash = 0; // OS build string, not the marketing version

  bool operator==(const MetalPipelineStorageIdentity& other) const;
  bool operator!=(const MetalPipelineStorageIdentity& other) const {
    return !(*this == other);
  }

  // Human-readable, for the "why did the cache rebuild" log line. A silent
  // rebuild is indistinguishable from a cache that never worked.
  std::string DescribeMismatch(const MetalPipelineStorageIdentity& other) const;
};

// Computes the identity for the running process. `device` is an id<MTLDevice>.
// Never fails: an unavailable input hashes to 0, which still participates in
// equality, so an entry written when it was unavailable is not served once it
// becomes available.
MetalPipelineStorageIdentity ComputeMetalPipelineStorageIdentity(void* device);

// Owns one title's durable directory: trust, admission accounting, and the
// cross-session bounds. Fail-soft throughout — any error degrades to "no
// durable storage this session" and never faults (spec: a corrupt store is
// treated as absent, never as fatal).
class MetalPipelineStorage {
 public:
  MetalPipelineStorage() = default;
  ~MetalPipelineStorage();

  MetalPipelineStorage(const MetalPipelineStorage&) = delete;
  MetalPipelineStorage& operator=(const MetalPipelineStorage&) = delete;

  // Opens (or creates) <cache_root>/metal/<title-id>/ and validates its
  // manifest against `identity`. On any mismatch, corruption, or truncation the
  // directory is discarded and recreated empty. Returns false if durable
  // storage is unavailable for any reason, in which case the caller proceeds
  // exactly as it does today with no durable store.
  bool Open(const std::filesystem::path& cache_root, uint32_t title_id,
            const MetalPipelineStorageIdentity& identity);
  void Close();

  bool is_open() const { return is_open_; }

  // 3.6a — M1 libraries, the M2 archive, and the manifest live beneath the
  // title directory. Empty when not open.
  const std::filesystem::path& title_root() const { return title_root_; }
  std::filesystem::path metallib_dir() const;
  std::filesystem::path binary_archive_path() const;

  // 3.6b/3.6e — admission accounting. Returns false once this title has reached
  // its cap; the caller then stops admitting NEW artifacts for the session and
  // keeps using what is already there. It must not delete live entries or
  // rebuild, either of which would thrash exactly when the title is busiest.
  bool CanAdmit(uint64_t byte_count) const;
  void RecordAdmitted(uint64_t byte_count);

  uint64_t bytes_admitted() const { return bytes_admitted_; }
  uint64_t bytes_on_open() const { return bytes_on_open_; }
  bool admission_stopped() const { return admission_stopped_; }

  // 3.6f — reported before promotion, so the 512 MiB default is revised from
  // evidence rather than shipped close to a boundary nobody measured.
  void LogSizeReport(const char* reason) const;

  // M1 (3.7/3.8) — the durable metallib store. `key` is the content-derived
  // shader identity; the storage-wide provenance was already validated at Open,
  // so an entry does not repeat it. Only the metallib is stored: the function
  // name is recovered from the library's own function list, exactly as the cold
  // path does.
  //
  // Load returns false for a miss, a corrupt entry, or a truncated one — all
  // three are simply misses, and a miss costs a normal cold translation.
  bool LoadMetallib(uint64_t key, std::vector<uint8_t>* metallib_out) const;
  bool StoreMetallib(uint64_t key, const std::vector<uint8_t>& metallib);

  // Atomic publish: writes to `<final>.tmp` then renames. A crash mid-write
  // therefore cannot leave a torn file at the real path. Shape borrowed from
  // Edge's metallib cache, which is the one part of it worth copying verbatim.
  static bool WriteFileAtomic(const std::filesystem::path& path,
                              const void* data, size_t size);

 private:
  // 3.6c/3.6d — runs at Open, before this title is locked, so the active title
  // and any directory held by another process are never candidates.
  void EvictToGlobalCapLocked(const std::filesystem::path& metal_root);

  bool LoadAndValidateManifest(const MetalPipelineStorageIdentity& identity,
                               std::string* mismatch_out);
  bool WriteManifest(const MetalPipelineStorageIdentity& identity);
  bool AcquireLock();
  void ReleaseLock();

  bool is_open_ = false;
  std::filesystem::path title_root_;
  std::filesystem::path lock_path_;
  int lock_fd_ = -1;
  uint32_t title_id_ = 0;
  uint64_t bytes_on_open_ = 0;
  uint64_t bytes_admitted_ = 0;
  bool admission_stopped_ = false;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_STORAGE_H_
