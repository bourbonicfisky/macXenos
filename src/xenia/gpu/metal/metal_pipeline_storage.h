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
#include <functional>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"

// One exact rollback switch per mechanism. M1 (metallib cache) is promoted
// and default ON; M2 (binary archive) is default OFF pending its successor.
DECLARE_bool(metal_pipeline_metallib_cache);
DECLARE_bool(metal_pipeline_binary_archive);

namespace xe {
namespace gpu {
namespace metal {

// Exact process-level rollout switch for separate linear host storage of
// k_8_8_8_8_GAMMA render targets. Keep the parser pure so rollback semantics
// can be pinned without a Metal device or process-environment mutation.
enum class MetalGammaRenderTargetLinearSetting : uint8_t {
  kDisabled,
  kEnabled,
  kMalformed,
};

MetalGammaRenderTargetLinearSetting ParseMetalGammaRenderTargetLinearSetting(
    const char* value);
bool MetalGammaRenderTargetLinearEnabled();

// The single source of truth for the translator configuration.
// MetalPipelineCache::Initialize constructs the translators from these, and the
// storage identity below fingerprints them, so the two cannot drift. Before
// this existed the values were duplicated, and a change to one without the
// other would have served a provably wrong shader under an unchanged cache key
// — flipping GammaRenderTargetAsUnorm8 alone is the difference behind the JC2
// black-gameplay bug.
struct MetalTranslatorConfig {
  static constexpr bool kBindlessResourcesUsed = true;
  static constexpr bool kEdramRovUsed = false;
  static constexpr bool GammaRenderTargetAsUnorm8(
      MetalGammaRenderTargetLinearSetting setting) {
    // Disabled and malformed both select the proven legacy path. This is the
    // fail-closed half of the exact XE_METAL_GAMMA_RT_LINEAR contract.
    return setting != MetalGammaRenderTargetLinearSetting::kEnabled;
  }
  static bool GammaRenderTargetAsUnorm8() {
    return !MetalGammaRenderTargetLinearEnabled();
  }
  static constexpr bool kMscBindlessResourcesUsed = true;
  // Revision of the emitted shader ABI — the system-constant layout, the
  // fetch-path codegen, and the geometry-emulation stage signatures. Unlike the
  // booleans above this does not describe a configuration choice; it exists so
  // a change to what the translator or the bridge EMITS for byte-identical
  // guest ucode still invalidates cached metallibs AND cached geometry records.
  // Bump it whenever SystemConstants changes layout, the fetch path emits
  // different code, or the object/mesh stage signatures change.
  //   1 - baseline
  //   2 - texture_integer_scale_bits added; fetch applies the integer scale
  //       after signs/gamma and before exponent bias
  //   3 - geometry-emulation clip-distance boundary relabel: the object-stage
  //       output and mesh-stage input signatures carry the clip payload under a
  //       private semantic. Same guest ucode, DIFFERENT emitted geometry
  //       stages, so durable geometry records written before this are stale and
  //       must not be served.
  //   4 - float24 depth invariance modes (implement-metal-depth-semantics
  //       stage A): depth_stencil_mode joined the PS modification. The
  //       modification value already keys translations, but records cached
  //       before revision 4 were all written with the mode bits at zero under
  //       XE_METAL_F24_PS builds that did not exist yet — the revision pins
  //       the SCHEMA so a pre-stage-A cache can never satisfy a stage-A
  //       lookup (or vice versa) through an incidentally-equal key.
  //   5 - the clip-distance relabel became UNCONDITIONAL. Revision 3 introduced
  //       the mechanism, but it shipped behind XE_METAL_CLIP_RELABEL defaulting
  //       OFF, so every record written at revisions 3 and 4 was in fact produced
  //       WITHOUT the relabel. Now that it always runs, those records describe
  //       different emitted geometry stages for the same guest ucode. They must
  //       not be served, and the geometry record keys CANNOT distinguish them:
  //       the object-stage key is (kind, MetallibCacheKey(vertex), topology) and
  //       the mesh-stage key is (kind, GeometryShaderKey), neither of which can
  //       see the relabel. The revision is therefore the ONLY thing standing
  //       between a warm pre-fix cache and a silently wrong pipeline — the same
  //       failure shape as the dxilconv converter-reuse bug, where corrupt
  //       metallibs were served from caches keyed on hashes that could not see
  //       the fix.
  static constexpr uint32_t kFetchAbiRevision = 5;

  // Generation of the DXBC->DXIL conversion WRAPPER — how conversions are
  // produced, as distinct from which libdxilconv is loaded (that is
  // dxc_artifact_hash) and from what the translator emits (that is
  // kFetchAbiRevision).
  //   1 - fresh IDxbcConverter per Convert (2026-08-02). Reused converter
  //       instances carried call history and emitted CORRUPT quad-domain
  //       artifacts. The bug was in xenia's wrapper, so the dylib hash could
  //       not see it, and the metallib key carries no DXIL bytes — without
  //       this field a warm cache would serve reuse-era corruption forever.
  // Bump on ANY wrapper-side change to how conversions are produced. Lives
  // here, next to the revision it is a sibling of, so the storage identity and
  // the conformance mirror can both name it instead of repeating a literal.
  static constexpr uint32_t kDxilconvGeneration = 1;
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

  // adopt-xenios-shader-artifact-caching 2.1 item 3 — the dxilconv OPTION
  // string (metal_converter_options.h). The three fields above cover WHICH
  // translator binaries are loaded and WHAT our translator emits; none of them
  // covers HOW dxilconv is invoked. The option string changes the emitted DXIL
  // for byte-identical DXBC — measured 3644 B under the default
  // "-skip-container-parts" vs 4436 B under an empty string on the same
  // quad-domain payload — so before this field a warm cache written with one
  // XENIA_DXBC2DXIL_FLAGS setting was served under another. Pinned by
  // conformance/metal_fetch_abi_cache_identity_test.cc (identity + discard) and
  // conformance/metal_dxil_determinism_test.mm (the byte-level premise).
  uint64_t converter_options_hash = 0;

  bool operator==(const MetalPipelineStorageIdentity& other) const;
  bool operator!=(const MetalPipelineStorageIdentity& other) const {
    return !(*this == other);
  }

  // Human-readable, for the "why did the cache rebuild" log line. A silent
  // rebuild is indistinguishable from a cache that never worked.
  std::string DescribeMismatch(const MetalPipelineStorageIdentity& other) const;
};

// Geometry-record kinds. Both kinds share one directory, so the kind is part of
// the key; otherwise a vertex-stage record and a shader-stage record could
// collide and one would be loaded as the other.
inline constexpr uint32_t kMetalGeometryRecordKindVertexStage = 1;
inline constexpr uint32_t kMetalGeometryRecordKindShaderStage = 2;
// Tessellation pipeline record (port-xenios-tessellation task 3.2). A NEW kind
// with its own key derivation AND its own file extension (.tess), so it is
// doubly disjoint from the existing kinds: a pre-change store can never
// satisfy a tessellation lookup, and a tessellation record can never be loaded
// as a metallib or geometry record even under a numerically identical key.
// Design-D4 consequence: because no existing artifact kind changes meaning,
// no fingerprint / fetch-ABI bump accompanies this kind — proven by
// conformance/metal_tessellation_storage_disjointness_test.cc.
inline constexpr uint32_t kMetalTessellationRecordKind = 3;

// Durable identity of one geometry stage.
//
// `input_topology` is load-bearing and must never be dropped: it is the MSC
// input topology the OBJECT STAGE IS COMPILED AGAINST, selected by the
// geometry type (point -> kPoint, rect -> kTriangle, quad -> kLine). The same
// guest VS therefore yields DIFFERENT object stages for different geometry
// types, and an identity that omits it makes them collide — the first one
// built wins and the other silently gets the wrong stage. That defect existed
// in the in-memory cache and is covered by
// conformance/metal_geometry_stage_identity_test.mm.
uint64_t MetalGeometryRecordKey(uint32_t kind, uint64_t shader_identity,
                                uint32_t input_topology);

// Durable identity of one tessellation pipeline record
// (kMetalTessellationRecordKind baked in). `domain_shader_identity` is the
// content-derived identity of the guest domain-shader translation (ucode hash +
// modification + type — the same triple the M1 metallib key hashes), never a
// translation pointer (port notes gotcha 7). host_vertex_shader_type +
// tessellation_mode select the prebuilt HS/tess-VS blobs, so they are part of
// the record identity; host_primitive_type is NOT (it affects only dispatch,
// not any stored stage), it travels inside the record as replay aux data.
uint64_t MetalTessellationRecordKey(uint64_t domain_shader_identity,
                                    uint32_t host_vertex_shader_type,
                                    uint32_t tessellation_mode);

// In-memory identity of a geometry object stage, for the run-local cache.
//
// Same load-bearing rule as MetalGeometryRecordKey above: `input_topology` is
// what the object stage is COMPILED AGAINST, so it must participate in BOTH
// equality and hashing. Dropping it from either collapses two distinct stages
// onto one entry and the first one built wins — the production defect this
// type exists to prevent. `translation` is an opaque identity only; the key
// deliberately does not depend on the translation type, which keeps it pure
// and directly testable.
struct MetalGeometryVertexStageKey {
  const void* translation = nullptr;
  uint32_t input_topology = 0;

  bool operator==(const MetalGeometryVertexStageKey& other) const {
    return translation == other.translation &&
           input_topology == other.input_topology;
  }
  bool operator!=(const MetalGeometryVertexStageKey& other) const {
    return !(*this == other);
  }

  struct Hasher {
    size_t operator()(const MetalGeometryVertexStageKey& key) const {
      // Both fields must reach the hash, or two topologies land in the same
      // bucket and rely on equality alone to separate them.
      const size_t pointer_hash = std::hash<const void*>{}(key.translation);
      const size_t topology_hash =
          std::hash<uint32_t>{}(key.input_topology) * 0x9E3779B97F4A7C15ull;
      return pointer_hash ^ (topology_hash + 0x9E3779B9u + (pointer_hash << 6) +
                             (pointer_hash >> 2));
    }
  };
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

  // M1 for geometry emulation (3.10). A geometry stage is not one blob: the
  // object stage is a metallib PLUS a separate stage-in metallib PLUS
  // reflection the pipeline descriptor cannot work without (entry-point name,
  // and either the VS output size or the GS max input primitives).
  //
  // All of it lives in ONE versioned, atomically published record, because a
  // partially valid record is the dangerous case: combining a fresh component
  // with a stale one would build a pipeline out of mismatched stages. Any
  // missing, short, or inconsistent component makes the WHOLE record a miss.
  struct GeometryRecord {
    std::string function_name;
    uint32_t reflection = 0;      // vertex_output_size or max_input_primitives
    std::vector<uint8_t> blob;    // object/mesh stage metallib
    std::vector<uint8_t> aux_blob;  // stage-in metallib; empty for a GS stage
  };
  bool LoadGeometryRecord(uint64_t key, GeometryRecord* record_out) const;
  bool StoreGeometryRecord(uint64_t key, const GeometryRecord& record);

  // M1 for tessellation (port-xenios-tessellation task 3.2). One tessellation
  // pipeline family needs FOUR metallibs (prebuilt tess VS + its synthesized
  // stage-in, prebuilt HS, guest DS), their function names, and the reflection
  // fields IRRuntimeValidateTessellationPipeline and
  // IRRuntimeTessellationPipelineConfig consume — none of which the plain
  // metallib store or the single-reflection GeometryRecord can carry.
  //
  // Same all-or-nothing rule as GeometryRecord, for the same reason: a
  // pipeline assembled from a fresh HS and a stale DS would validate against
  // the wrong control-point layout. The whole record is published atomically
  // and any missing, short, or zero-where-nonzero-required component makes the
  // WHOLE record a miss.
  //
  // The aux triple (host_vertex_shader_type / tessellation_mode /
  // host_primitive_type) mirrors XeniOS's kTessellation description aux0/1/2:
  // it is what a warm-start prewarm pass would need to reconstruct a minimal
  // ProcessingResult. Our storage replays lazily (records are consulted at
  // first GetOrCreate, there is no prewarm enumeration), so today the triple
  // is written and verified but only mode/type are re-derived on load paths;
  // the record carries all three so prewarm wiring needs no format change.
  struct TessellationRecord {
    // Replay aux (see above).
    uint32_t host_vertex_shader_type = 0;
    uint32_t tessellation_mode = 0;
    uint32_t host_primitive_type = 0;
    // Stage entry points.
    std::string vs_function_name;
    std::string hs_function_name;
    std::string ds_function_name;
    // Reflection consumed by validation + the pipeline config. The config's
    // gsMaxInputPrimitivesPerMeshThreadgroup comes from the DOMAIN block
    // (ds_max_input_prims_per_mesh_threadgroup), never from an HS field.
    uint32_t vs_output_size = 0;
    uint32_t hs_max_patches_per_object_threadgroup = 0;
    uint32_t hs_max_object_threads_per_patch = 0;
    uint32_t hs_patch_constants_size = 0;
    uint32_t hs_input_control_point_count = 0;
    uint32_t hs_output_control_point_count = 0;
    uint32_t hs_output_control_point_size = 0;
    uint32_t hs_tessellator_domain = 0;
    uint32_t hs_tessellator_partitioning = 0;
    uint32_t hs_tessellator_output_primitive = 0;
    float hs_max_tessellation_factor = 0.0f;
    uint32_t ds_max_input_prims_per_mesh_threadgroup = 0;
    uint32_t ds_input_control_point_count = 0;
    uint32_t ds_input_control_point_size = 0;
    uint32_t ds_patch_constants_size = 0;
    uint32_t ds_tessellator_domain = 0;
    // Stage metallibs. All four are required (an empty stage-in is a hard
    // pipeline failure — port notes gotcha 9 — so it is required here too).
    std::vector<uint8_t> vs_metallib;
    std::vector<uint8_t> vs_stage_in_metallib;
    std::vector<uint8_t> hs_metallib;
    std::vector<uint8_t> ds_metallib;
  };
  bool LoadTessellationRecord(uint64_t key,
                              TessellationRecord* record_out) const;
  bool StoreTessellationRecord(uint64_t key, const TessellationRecord& record);

  // Test seam: how many entries this session admitted, so a test can prove a
  // second distinct artifact was actually written rather than deduplicated.
  uint64_t admitted_entry_count() const { return admitted_entry_count_; }

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
  uint64_t admitted_entry_count_ = 0;
  bool admission_stopped_ = false;
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_PIPELINE_STORAGE_H_
