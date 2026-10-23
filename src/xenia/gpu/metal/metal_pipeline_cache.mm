/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_pipeline_cache.h"
#include "xenia/gpu/metal/metal_draw_resolution_scale.h"

#import <Metal/Metal.h>

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/metal/metal_dxbc_clip_relabel.h"
#include "xenia/gpu/dxbc_geometry_shader.h"
#include "xenia/gpu/metal/metal_artifact_hit_verify.h"
#include "xenia/gpu/metal/metal_render_target_cache.h"
#include "xenia/gpu/metal/metal_shader_build_census.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/graphics_provider.h"

// Metal Shader Converter runtime (declarations only; implementation TU is
// ir_runtime_impl.mm). Provides IRGeometryEmulationPipelineDescriptor and
// IRRuntimeNewGeometryEmulationPipeline for the geometry-emulation mesh pipeline,
#include "metal_irconverter_runtime.h"


// Defined in graphics_system.cc. Gates GraphicsSystem::InitializeShaderStorage
// before any backend is consulted, so it silently disables both durable
// mechanisms; reported in the effective-state line below.
DECLARE_bool(store_shaders);

DEFINE_bool(metal_async_shader_translation, false,
            "Translate shaders the first time they are seen on background "
            "threads instead of inside the frame. Draws whose shader is still "
            "translating are skipped for the few frames it takes, so objects "
            "can appear a moment late the first time a shader is used; the "
            "alternative is a freeze of up to seconds (cold RDR on the Mac "
            "Pro: 7 s of in-frame translation). Cached shaders are unaffected. "
            "XE_METAL_ASYNC_SHADERS=0/1 overrides.",
            "GPU");

namespace xe {
namespace gpu {
namespace metal {


MetalPipelineCache::MetalPipelineCache(void* device,
                                       uint32_t draw_resolution_scale_x,
                                       uint32_t draw_resolution_scale_y)
    : device_(device),
      draw_resolution_scale_x_(draw_resolution_scale_x),
      draw_resolution_scale_y_(draw_resolution_scale_y) {}

MetalPipelineCache::~MetalPipelineCache() { Shutdown(); }

bool MetalPipelineCache::Initialize() {
  // Match the shader-spike configuration: Apple vendor, bindless resources, no
  // pixel-shader-interlock EDRAM (the copy-based RT model), which is what the
  // Xbox-360 root signature in MetalShaderConverter expects.
  // gamma_render_target_as_unorm8=true: k_8_8_8_8_GAMMA render targets are
  // plain unorm8 on our side, so the PS must PWL-gamma-encode its linear output
  // when kSysFlag_ConvertColorNToGamma is set. Without this ctor flag the
  // translator never emits that codepath and the flag is silently ignored —
  // JC2's scene renders ~0.02-linear sunlight that only becomes visible
  // (0x5C-range) after gamma encoding + exposure (the "black gameplay" bug).
  // Configuration comes from MetalTranslatorConfig so the durable cache's
  // schema fingerprint (3.3b) cannot drift from what is actually built here.
  shader_translator_ = std::make_unique<DxbcShaderTranslator>(
      ui::GraphicsProvider::GpuVendorID::kApple,
      MetalTranslatorConfig::kBindlessResourcesUsed,
      MetalTranslatorConfig::kEdramRovUsed,
      MetalTranslatorConfig::GammaRenderTargetAsUnorm8(),
      /*msaa_2x_supported=*/true, draw_resolution_scale_x_,
      draw_resolution_scale_y_);
  shader_translator_->SetExactScaledPixelPosition(
      MetalExactScaledPixelPositionEnabled());
  shader_translator_->SetDeferredKill(MetalDeferredKillEnabled());

  std::string error;
  if (!dxil_converter_raw_.Initialize(&error)) {
    XELOGE("MetalPipelineCache: DXBC->DXIL converter init failed: {}", error);
    return false;
  }
  if (!metal_converter_.Initialize(
          MetalTranslatorConfig::kMscBindlessResourcesUsed, &error)) {
    XELOGE("MetalPipelineCache: Metal Shader Converter init failed: {}", error);
    return false;
  }

  // EFFECTIVE durable-cache state, logged unconditionally so a gate can assert
  // on it instead of trusting that a flag was applied.
  //
  // Two ways a requested mode silently fails to take effect, both hit in
  // practice: `xenia_canary` reads a persisted config that OVERRIDES
  // command-line cvars (while the trace-dump tool has no config, so there the
  // command line wins), and `store_shaders=false` makes
  // GraphicsSystem::InitializeShaderStorage return before any backend is
  // consulted, disabling BOTH mechanisms no matter what their own switches say.
  // Either one produces a run that looks like the mechanism is on and measures
  // as though it is off.
  XELOGI(
      "MetalPipelineCache: durable cache EFFECTIVE state — M1 metallib={}, "
      "M2 archive={}, store_shaders={}{}",
      cvars::metal_pipeline_metallib_cache ? "on" : "off",
      cvars::metal_pipeline_binary_archive ? "on" : "off",
      cvars::store_shaders ? "on" : "off",
      cvars::store_shaders
          ? ""
          : "  <- store_shaders=off DISABLES BOTH regardless of the switches");
  return true;
}

void MetalPipelineCache::InitializeDurableStorage(
    const std::filesystem::path& cache_root, uint32_t title_id) {
  ShutdownDurableStorage();
  // 3.1 — exact rollback switches, default OFF. Neither mechanism consumes the
  // storage yet (that is 3b/3c); opening it here is what makes the identity,
  // bounds, and locking path live rather than dormant.
  if (!cvars::metal_pipeline_metallib_cache &&
      !cvars::metal_pipeline_binary_archive) {
    return;
  }
  MetalPipelineStorageIdentity identity = ComputeMetalPipelineStorageIdentity(
      device_, draw_resolution_scale_x_, draw_resolution_scale_y_);
  if (!storage_.Open(cache_root, title_id, identity)) {
    XELOGW(
        "MetalPipelineCache: durable storage unavailable for title {:08X}; "
        "continuing with in-memory caches only",
        title_id);
  }

  // Task 2.2 — the DXIL store, opened independently and with its own identity.
  // Note it is NOT under the title directory and its Open is not conditional on
  // the pipeline storage having opened: the whole point is that the two
  // invalidate on different facts.
  // EFFECTIVE-STATE LOG. Emitted in BOTH arms and greppable (MetalDxilGate), so
  // "this run was default-on" and "this run was the rollback" are provable from
  // the log rather than assumed from the command line — a persisted config
  // value silently overriding the code default is exactly how the promotion
  // measurement could have been wrong (the config had metal_dxil_cache=false
  // written into it while the flag was default-off).
  XELOGI(
      "MetalDxilGate: metal_dxil_cache={} (default-on since 2026-08-05; false "
      "is the exact rollback), verify_every_nth={}, cap={} MiB",
      cvars::metal_dxil_cache ? "TRUE" : "FALSE",
      cvars::metal_dxil_cache_verify_every_nth,
      cvars::metal_dxil_cache_cap_mib);
  if (cvars::metal_dxil_cache) {
    if (!dxil_store_.Open(cache_root, ComputeMetalDxilStoreIdentity())) {
      XELOGW(
          "MetalDxilGate: store unavailable; DXBC->DXIL runs cold this session");
    }
  } else {
    XELOGI(
        "MetalDxilGate: ROLLBACK ARM — no DXIL store opened, no durable reads "
        "or writes this session");
  }
}

void MetalPipelineCache::ShutdownDurableStorage() {
  storage_.Close();
  dxil_store_.Close();
}

void MetalPipelineCache::Shutdown() {
  StopAsyncTranslationWorkers();
  ShutdownDurableStorage();
  for (auto& pair : render_pipeline_states_) {
    if (pair.second) {
      id<MTLRenderPipelineState> state =
          (__bridge_transfer id<MTLRenderPipelineState>)pair.second;
      (void)state;
    }
  }
  render_pipeline_states_.clear();
  // Release geometry-emulation pipelines + stage libraries (all retained void*).
  for (auto& pair : geometry_pipelines_) {
    if (pair.second.pipeline) {
      id<MTLRenderPipelineState> state =
          (__bridge_transfer id<MTLRenderPipelineState>)pair.second.pipeline;
      (void)state;
    }
  }
  geometry_pipelines_.clear();
  for (auto& pair : geometry_vertex_stages_) {
    if (pair.second.library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)pair.second.library;
      (void)lib;
    }
    if (pair.second.stage_in_library) {
      id<MTLLibrary> lib =
          (__bridge_transfer id<MTLLibrary>)pair.second.stage_in_library;
      (void)lib;
    }
  }
  geometry_vertex_stages_.clear();
  for (auto& pair : geometry_shader_stages_) {
    if (pair.second.library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)pair.second.library;
      (void)lib;
    }
  }
  geometry_shader_stages_.clear();
  for (auto& pair : shaders_) {
    delete pair.second;
  }
  shaders_.clear();
  shader_translator_.reset();
}

MetalShader* MetalPipelineCache::LoadShader(xenos::ShaderType shader_type,
                                            const uint32_t* host_address,
                                            uint32_t dword_count) {
  uint64_t data_hash =
      XXH3_64bits(host_address, dword_count * sizeof(uint32_t));
  auto it = shaders_.find(data_hash);
  if (it != shaders_.end()) {
    return it->second;
  }
  // Create and stash the shader even if it later fails translation, so we don't
  // retry a known-bad shader.
  MetalShader* shader =
      new MetalShader(shader_type, data_hash, host_address, dword_count);
  shaders_.emplace(data_hash, shader);
  return shader;
}

namespace {
// M1 entry key. Same content-derived identity the pipeline key uses, hashed to
// a filename. The storage-wide provenance (device, OS, translator artifacts and
// schema) was validated when the directory was opened, so it is not repeated
// per entry.
uint64_t MetallibCacheKey(const MetalShader::MetalTranslation& translation) {
  struct KeyBytes {
    uint64_t ucode_hash;
    uint64_t modification;
    uint32_t shader_type;
    uint32_t pad;
  } key_bytes = {};
  key_bytes.ucode_hash = translation.shader().ucode_data_hash();
  key_bytes.modification = translation.modification();
  key_bytes.shader_type = uint32_t(translation.shader().type());
  return XXH3_64bits(&key_bytes, sizeof(key_bytes));
}

}  // namespace


// The ONE place DXBC becomes DXIL. Every production conversion site routes here
// (adopt-xenios-shader-artifact-caching, scope correction 2026-08-05): before
// this existed the cache covered only the ordinary vertex/pixel path while the
// geometry object stage, the generated geometry stage, the depth-only fragment
// stage and all three tessellation stages called dxilconv directly. Geometry
// emulation is DEFAULT production behavior, so the majority of a
// geometry-heavy title's conversions bypassed both the cache and the census —
// which is also why the task 1.1 dxilconv figures UNDERSTATE the real cost.
//
// THE KEY IS THE EXACT BYTE BUFFER HANDED TO dxilconv, never a shader
// identity. That is not a convenience: the geometry sites mutate the DXBC with
// RelabelDxbcClipDistance immediately before converting, and a shader-level key
// cannot see that mutation. Keying on the post-transform bytes makes every
// present and future pre-Convert transform automatically covered — the same
// class of blindness that made kFetchAbiRevision 4->5 necessary.
//
// `site` names the caller for diagnostics only; it deliberately does NOT
// participate in the key, because two sites converting identical bytes must
// share the artifact (the prebuilt tessellation blobs are identical across
// every title, and 8 of Sonic's 94 conversions were duplicate DXBC).
bool MetalPipelineCache::ConvertDxbcToDxilCached(
    const std::vector<uint8_t>& dxbc, std::vector<uint8_t>& dxil_out,
    std::string* error_out, const char* site) {
  MetalShaderBuildCensus& census = GetMetalShaderBuildCensus();
  const bool cache_enabled =
      cvars::metal_dxil_cache && dxil_store_.is_open();
  const MetalDxilArtifactKey key = MetalDxilArtifactKey::ForDxbc(dxbc);

  if (cache_enabled) {
    std::vector<uint8_t> cached;
    const auto load_start = std::chrono::steady_clock::now();
    const MetalDxilLoadResult result = dxil_store_.Load(key, &cached);
    census.dxil_load_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - load_start)
                     .count()),
        std::memory_order_relaxed);
    if (result == MetalDxilLoadResult::kRejectedCorrupt) {
      // Not a miss. The store already logged and deleted the entry; the cold
      // conversion below rebuilds it. Counted as a correctness incident.
      census.dxil_cache_corrupt.fetch_add(1, std::memory_order_relaxed);
    } else if (result == MetalDxilLoadResult::kHit) {
      const uint64_t hit_index =
          census.dxil_cache_hits.fetch_add(1, std::memory_order_relaxed) + 1;
      // SAMPLED HIT VERIFICATION (2.3, invariant from 2.1): a hit must equal a
      // fresh conversion byte for byte, or it must miss. Sampled rather than
      // universal because verifying every hit costs the very conversion the
      // cache exists to skip.
      const uint64_t every_nth = cvars::metal_dxil_cache_verify_every_nth;
      if (!every_nth || (hit_index % every_nth) != 0) {
        dxil_out = std::move(cached);
        return true;
      }
      std::vector<uint8_t> fresh;
      std::string verify_error;
      const bool converted = dxil_converter_raw_.Convert(dxbc, fresh, &verify_error);
      const MetalArtifactHitVerdict verdict =
          converted ? VerifyMetalArtifactHit(cached, fresh)
                    : MetalArtifactHitVerdict::kMissUnverifiable;
      census.dxil_verify.Record(verdict);
      if (MetalArtifactHitIsServable(verdict)) {
        dxil_out = std::move(cached);
        return true;
      }
      // A served artifact this build does not reproduce. Report it, drop the
      // entry so it rebuilds, and use the FRESH bytes — never the cached ones.
      XELOGE(
          "MetalDxilStore: VERIFICATION FAILURE at site '{}' (verdict={}, "
          "cached={} B, fresh={} B) — the served artifact is not what this "
          "build produces. Entry dropped and rebuilt.",
          site, MetalArtifactHitVerdictName(verdict), cached.size(),
          fresh.size());
      dxil_store_.Forget(key);
      if (!converted || fresh.empty()) {
        if (error_out) {
          *error_out = verify_error.empty()
                           ? std::string("verification re-convert failed")
                           : verify_error;
        }
        return false;
      }
      dxil_out = std::move(fresh);
      return true;
    }
  }

  // Cold conversion. Timed into the census at EVERY site, so the per-stage
  // totals finally cover geometry and tessellation too.
  //
  // The timer scope ends BEFORE the store write on purpose: census.dxil must
  // mean "dxilconv time" and nothing else, or enabling the cache appears to
  // make conversion slower. A first attempt left Store() inside the scope and
  // the cold arm's dxil total rose by ~30 ms with the gate on — cache
  // bookkeeping being reported as converter cost, which is exactly the
  // measurement error a cold-regression promotion gate would then act on.
  if (!ConvertDxbcToDxilCold(dxbc, dxil_out, error_out)) {
    return false;
  }
  if (cache_enabled) {
    const auto store_start = std::chrono::steady_clock::now();
    dxil_store_.Store(key, dxil_out);
    census.dxil_store_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - store_start)
                     .count()),
        std::memory_order_relaxed);
  }
  return true;
}

bool MetalPipelineCache::ConvertDxbcToDxilCold(
    const std::vector<uint8_t>& dxbc, std::vector<uint8_t>& dxil_out,
    std::string* error_out) {
  MetalShaderBuildTimer timer(GetMetalShaderBuildCensus().dxil);
  if (!dxil_converter_raw_.Convert(dxbc, dxil_out, error_out)) {
    timer.Dismiss();
    return false;
  }
  timer.set_bytes(dxil_out.size());
  return true;
}

struct MetalPipelineCache::AsyncTranslationJob {
  uint64_t ucode_hash = 0;
  MetalShaderConverter::Stage stage = MetalShaderConverter::Stage::kVertex;
  uint64_t metallib_key = 0;
  std::vector<uint8_t> dxbc;
  bool dxil_from_store = false;
  std::vector<uint8_t> dxil;  // preset when the DXIL store had it
  std::vector<uint8_t> metallib;
  std::string error;
  bool ok = false;
  std::atomic<bool> done{false};
};

bool MetalPipelineCache::AsyncTranslationEnabled() {
  static const bool s_enabled = [] {
    if (const char* v = std::getenv("XE_METAL_ASYNC_SHADERS")) {
      return v[0] == '1' && !v[1];
    }
    // The translation dump diagnostic needs the synchronous path.
    if (std::getenv("XE_METAL_DUMP_XLAT")) {
      return false;
    }
    return cvars::metal_async_shader_translation;
  }();
  return s_enabled;
}

void MetalPipelineCache::AsyncTranslationWorkerMain() {
  pthread_setname_np("Metal Shader Translation");
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  MetalShaderBuildCensus::SetBackgroundThread();
  for (;;) {
    std::shared_ptr<AsyncTranslationJob> job;
    {
      std::unique_lock<std::mutex> lock(async_mutex_);
      async_cv_.wait(lock, [&] { return async_stop_ || !async_queue_.empty(); });
      if (async_stop_) {
        return;
      }
      job = std::move(async_queue_.front());
      async_queue_.pop_front();
    }
    job->ok = job->dxil_from_store ||
              ConvertDxbcToDxilCold(job->dxbc, job->dxil, &job->error);
    if (job->ok) {
      MetalShaderBuildTimer timer(GetMetalShaderBuildCensus().msc);
      MetalStageRequestProbe probe(
          job->dxil, uint32_t(job->stage),
          uint32_t(MetalShaderConverter::InputTopology::kUndefined));
      job->ok = metal_converter_.Convert(job->stage, job->dxil, job->metallib,
                                         &job->error);
      if (job->ok) {
        timer.set_bytes(job->metallib.size());
      } else {
        timer.Dismiss();
      }
    }
    job->done.store(true, std::memory_order_release);
  }
}

void MetalPipelineCache::StopAsyncTranslationWorkers() {
  {
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_stop_ = true;
    async_queue_.clear();
  }
  async_cv_.notify_all();
  for (std::thread& worker : async_workers_) {
    worker.join();
  }
  async_workers_.clear();
  async_jobs_.clear();
}

bool MetalPipelineCache::FinishAsyncTranslation(
    MetalShader::MetalTranslation& translation, AsyncTranslationJob& job) {
  if (!job.ok) {
    XELOGE("MetalPipelineCache: background translation failed for shader "
           "{:016X}: {}",
           job.ucode_hash, job.error);
    return false;
  }
  MetalShaderBuildCensus& census = GetMetalShaderBuildCensus();
  if (!job.dxil_from_store && cvars::metal_dxil_cache && dxil_store_.is_open()) {
    const auto store_start = std::chrono::steady_clock::now();
    dxil_store_.Store(MetalDxilArtifactKey::ForDxbc(job.dxbc), job.dxil);
    census.dxil_store_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - store_start)
                     .count()),
        std::memory_order_relaxed);
  }
  {
    MetalShaderBuildTimer timer(census.function);
    if (!translation.CreateMetalFunction(device_, job.metallib)) {
      timer.Dismiss();
      XELOGE(
          "MetalPipelineCache: MTLFunction creation failed for shader {:016X}",
          job.ucode_hash);
      return false;
    }
    timer.set_bytes(job.metallib.size());
  }
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    const auto store_start = std::chrono::steady_clock::now();
    if (storage_.StoreMetallib(job.metallib_key, job.metallib)) {
      census.metallib_stores.fetch_add(1, std::memory_order_relaxed);
      census.metallib_store_bytes.fetch_add(job.metallib.size(),
                                            std::memory_order_relaxed);
    }
    census.metallib_store_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - store_start)
                     .count()),
        std::memory_order_relaxed);
  }
  return true;
}

bool MetalPipelineCache::TranslateShader(
    MetalShader::MetalTranslation& translation, bool* pending) {
  if (pending) {
    *pending = false;
  }
  if (!async_jobs_.empty()) {
    auto it = async_jobs_.find(&translation);
    if (it != async_jobs_.end()) {
      if (!it->second->done.load(std::memory_order_acquire)) {
        if (pending) {
          *pending = true;
        }
        return false;
      }
      std::shared_ptr<AsyncTranslationJob> job = std::move(it->second);
      async_jobs_.erase(it);
      return FinishAsyncTranslation(translation, *job);
    }
  }
  if (translation.is_translated()) {
    return translation.is_valid() && translation.has_metal_function();
  }

  MetalShader& shader = static_cast<MetalShader&>(translation.shader());


  // Stage 1: microcode -> DXBC (reused DxbcShaderTranslator).
  // Task 1.1 census: timed even though it is the cheap stage, because it is
  // the floor a metallib hit cannot go below (it runs on a hit too).
  MetalShaderBuildCensus& census = GetMetalShaderBuildCensus();
  {
    MetalShaderBuildTimer timer(census.dxbc);
    if (!shader_translator_->TranslateAnalyzedShader(translation)) {
      timer.Dismiss();
      XELOGE("MetalPipelineCache: DXBC translation failed for shader {:016X}",
             shader.ucode_data_hash());
      return false;
    }
    timer.set_bytes(translation.translated_binary().size());
  }
  const std::vector<uint8_t>& dxbc = translation.translated_binary();
  if (dxbc.empty()) {
    XELOGE("MetalPipelineCache: empty DXBC for shader {:016X}",
           shader.ucode_data_hash());
    return false;
  }

  // M1 (3.7/3.8) — durable metallib store. Stage 1 above always runs even on a
  // hit: it costs 0.8-0.9% of translation time (~0.06 ms mean, measured in
  // task 3.9) and it keeps translated_binary() populated, which the
  // geometry-emulation stages consume. Skipping it to save that would break
  // geometry emulation for a saving smaller than the measurement noise.
  //
  // What a hit DOES skip is stages 2 and 3 — DXBC->DXIL and DXIL->metallib —
  // which are 99.7-99.8% of translation cost.
  const uint64_t m1_key = MetallibCacheKey(translation);
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    std::vector<uint8_t> cached_metallib;
    const auto load_start = std::chrono::steady_clock::now();
    const bool loaded = storage_.LoadMetallib(m1_key, &cached_metallib);
    census.metallib_load_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - load_start)
                     .count()),
        std::memory_order_relaxed);
    if (loaded) {
      MetalShaderBuildTimer timer(census.function);
      if (translation.CreateMetalFunction(device_, cached_metallib)) {
        timer.set_bytes(cached_metallib.size());
        census.metallib_hits.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      timer.Dismiss();
      // A stored blob the device will not load is not trustworthy — fall
      // through to a full cold translation rather than failing the draw.
      // Counted as REJECTED, never as a miss: a miss is cold, this is a stored
      // artifact that is wrong, and one number covering both would hide the
      // second inside the first.
      census.metallib_rejected.fetch_add(1, std::memory_order_relaxed);
      XELOGW(
          "MetalPipelineCache: cached metallib {:016X} failed to load; "
          "retranslating",
          m1_key);
    } else {
      census.metallib_misses.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Background translation: stages 2 and 3 go to a worker; the draw that
  // asked is skipped and a later call publishes the function.
  if (pending && AsyncTranslationEnabled()) {
    auto job = std::make_shared<AsyncTranslationJob>();
    job->ucode_hash = shader.ucode_data_hash();
    job->stage = shader.type() == xenos::ShaderType::kVertex
                     ? MetalShaderConverter::Stage::kVertex
                     : MetalShaderConverter::Stage::kFragment;
    job->metallib_key = m1_key;
    job->dxbc = dxbc;
    if (cvars::metal_dxil_cache && dxil_store_.is_open()) {
      std::vector<uint8_t> cached;
      if (dxil_store_.Load(MetalDxilArtifactKey::ForDxbc(dxbc), &cached) ==
          MetalDxilLoadResult::kHit) {
        GetMetalShaderBuildCensus().dxil_cache_hits.fetch_add(
            1, std::memory_order_relaxed);
        job->dxil = std::move(cached);
        job->dxil_from_store = true;
      }
    }
    if (async_workers_.empty()) {
      const uint32_t cores = std::max(1u, std::thread::hardware_concurrency());
      const uint32_t count = std::min(4u, std::max(1u, cores / 4));
      for (uint32_t i = 0; i < count; ++i) {
        async_workers_.emplace_back([this] { AsyncTranslationWorkerMain(); });
      }
      XELOGI("MetalPipelineCache: {} background shader translation threads",
             count);
    }
    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      async_queue_.push_back(job);
    }
    async_cv_.notify_one();
    async_jobs_.emplace(&translation, std::move(job));
    *pending = true;
    return false;
  }

  // Stage 2: DXBC -> DXIL, through the ONE cache-aware helper every conversion
  // site uses (see ConvertDxbcToDxilCached).
  std::string error;
  std::vector<uint8_t> dxil;
  if (!ConvertDxbcToDxilCached(dxbc, dxil, &error, "vertex/pixel")) {
    XELOGE("MetalPipelineCache: DXBC->DXIL failed for shader {:016X}: {}",
           shader.ucode_data_hash(), error);
    return false;
  }

  // Shader-chain forensics (XE_METAL_DUMP_XLAT=<hex ucode hash>): dump the
  // translated DXBC + DXIL + metallib for that shader to /tmp for offline
  // disassembly (the light-VS frc-unpack outputs zero; the M4 GPU primitive
  // is proven exact, so the divergence is inside this chain).
  static const uint64_t s_dump_xlat = [] {
    const char* v = std::getenv("XE_METAL_DUMP_XLAT");
    return v ? strtoull(v, nullptr, 16) : 0ull;
  }();
  if (s_dump_xlat && translation.shader().ucode_data_hash() == s_dump_xlat) {
    auto dump = [&](const char* ext, const std::vector<uint8_t>& data) {
      std::string path =
          fmt::format("/tmp/xe_xlat_{:016X}_{:016X}.{}", s_dump_xlat, translation.modification(), ext);
      FILE* f = fopen(path.c_str(), "wb");
      if (f) {
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
        XELOGW("MetalPipelineCache: dumped {} ({} bytes)", path, data.size());
      }
    };
    dump("dxbc", dxbc);
  }
  // Stage 3: DXIL -> metallib.
  MetalShaderConverter::Stage stage =
      shader.type() == xenos::ShaderType::kVertex
          ? MetalShaderConverter::Stage::kVertex
          : MetalShaderConverter::Stage::kFragment;
  if (s_dump_xlat && translation.shader().ucode_data_hash() == s_dump_xlat) {
    std::string path = fmt::format("/tmp/xe_xlat_{:016X}_{:016X}.dxil", s_dump_xlat, translation.modification());
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
      fwrite(dxil.data(), 1, dxil.size(), f);
      fclose(f);
      XELOGW("MetalPipelineCache: dumped {} ({} bytes)", path, dxil.size());
    }
  }
  std::vector<uint8_t> metallib;
  {
    MetalShaderBuildTimer timer(census.msc);
    MetalStageRequestProbe probe(dxil, uint32_t(stage),
                                 uint32_t(MetalShaderConverter::InputTopology::
                                              kUndefined));
    if (!metal_converter_.Convert(stage, dxil, metallib, &error)) {
      timer.Dismiss();
      XELOGE("MetalPipelineCache: DXIL->metallib failed for shader {:016X}: {}",
             shader.ucode_data_hash(), error);
      return false;
    }
    timer.set_bytes(metallib.size());
  }

  // Load onto the device as an MTLFunction.
  if (s_dump_xlat && translation.shader().ucode_data_hash() == s_dump_xlat) {
    std::string path =
        fmt::format("/tmp/xe_xlat_{:016X}_{:016X}.metallib", s_dump_xlat, translation.modification());
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
      fwrite(metallib.data(), 1, metallib.size(), f);
      fclose(f);
      XELOGW("MetalPipelineCache: dumped {} ({} bytes)", path, metallib.size());
    }
  }
  {
    MetalShaderBuildTimer timer(census.function);
    if (!translation.CreateMetalFunction(device_, metallib)) {
      timer.Dismiss();
      XELOGE(
          "MetalPipelineCache: MTLFunction creation failed for shader {:016X}",
          shader.ucode_data_hash());
      return false;
    }
    timer.set_bytes(metallib.size());
  }

  // Publish only after the device has accepted it, so a blob that cannot load
  // is never written for a later run to trip over.
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    const auto store_start = std::chrono::steady_clock::now();
    if (storage_.StoreMetallib(m1_key, metallib)) {
      census.metallib_stores.fetch_add(1, std::memory_order_relaxed);
      census.metallib_store_bytes.fetch_add(metallib.size(),
                                            std::memory_order_relaxed);
    }
    census.metallib_store_ns.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - store_start)
                     .count()),
        std::memory_order_relaxed);
  }
  return true;
}

namespace {

// 3.2 — content-derived shader identity for the pipeline key.
//
// The key previously used MTLFunction / translation POINTERS. Those are stable
// within one run and meaningless across runs, so no durable entry could be
// written against them. (ucode hash, modification, type) is the equivalent
// identity in content terms: MetalShader owns exactly one translation per
// modification and TranslateShader is idempotent per translation, so the
// pointer and this triple are 1:1 within a run — the swap cannot merge two
// distinct pipelines or split one.
//
// `present` keeps a null pixel shader distinct from any fragment-carrying
// pipeline, which the spec requires and which the old key got from a null
// pointer. A real shader can never collide with the absent case because
// present=0 zeroes the whole triple.
struct MetalShaderKeyIdentity {
  uint64_t ucode_hash;
  uint64_t modification;
  uint32_t shader_type;
  uint32_t present;
};

MetalShaderKeyIdentity MakeShaderKeyIdentity(
    const MetalShader::MetalTranslation* translation) {
  MetalShaderKeyIdentity identity = {};
  if (!translation) {
    return identity;
  }
  identity.ucode_hash = translation->shader().ucode_data_hash();
  identity.modification = translation->modification();
  identity.shader_type = uint32_t(translation->shader().type());
  identity.present = 1;
  return identity;
}

// The COMPLETE pipeline-creation identity for a geometry-emulation pipeline —
// the exact bytes hashed into its cache key. Everything MTLRenderPipelineState
// creation consumes is here: shader identities, every MRT format, blend
// controls, write mask, depth (and thereby stencil: Depth32Float_Stencil8
// implies the stencil attachment) format, and sample count.
//
// Depth/stencil ENABLE/WRITE state is deliberately absent: on Metal that lives
// in MTLDepthStencilState, set per-encoder, and does not participate in
// pipeline creation — logging it here would imply a discriminator that cannot
// exist at this boundary.
struct GeometryPipelineKeyBytes {
  uint32_t gs_key;
  uint32_t pad0;
  MetalShaderKeyIdentity vs;  // object-stage source translation.
  MetalShaderKeyIdentity ps;  // fragment stage.
  uint64_t colors[xenos::kMaxColorRenderTargets];
  uint64_t depth;
  uint32_t sample_count;
  uint32_t blends[xenos::kMaxColorRenderTargets];
  uint32_t write_mask;
  uint32_t pad;
};

// One formatter for BOTH the success and failure logs, so they cannot drift
// apart: the first attribution round found the same geometry triple both
// succeeding and failing, and the depth-format correlation could not be tested
// precisely because successes logged fewer fields than failures.
std::string DescribeGeometryPipelineKey(uint64_t key_digest,
                                        const GeometryPipelineKeyBytes& kb) {
  return fmt::format(
      "key={:016X} gs_key={:08X} vs={:016X}/{:08X} ps={:016X}/{:08X} "
      "colors=[{:X},{:X},{:X},{:X}] blends=[{:08X},{:08X},{:08X},{:08X}] "
      "write_mask={:X} depth_fmt={} samples={}",
      key_digest, kb.gs_key, kb.vs.ucode_hash, uint32_t(kb.vs.modification),
      kb.ps.ucode_hash, uint32_t(kb.ps.modification), kb.colors[0],
      kb.colors[1], kb.colors[2], kb.colors[3], kb.blends[0], kb.blends[1],
      kb.blends[2], kb.blends[3], kb.write_mask, kb.depth, kb.sample_count);
}

MTLBlendFactor ToMetalBlendFactor(MetalBlendFactor factor) {
  switch (factor) {
    case MetalBlendFactor::kZero:
      return MTLBlendFactorZero;
    case MetalBlendFactor::kOne:
      return MTLBlendFactorOne;
    case MetalBlendFactor::kSourceColor:
      return MTLBlendFactorSourceColor;
    case MetalBlendFactor::kOneMinusSourceColor:
      return MTLBlendFactorOneMinusSourceColor;
    case MetalBlendFactor::kSourceAlpha:
      return MTLBlendFactorSourceAlpha;
    case MetalBlendFactor::kOneMinusSourceAlpha:
      return MTLBlendFactorOneMinusSourceAlpha;
    case MetalBlendFactor::kDestinationColor:
      return MTLBlendFactorDestinationColor;
    case MetalBlendFactor::kOneMinusDestinationColor:
      return MTLBlendFactorOneMinusDestinationColor;
    case MetalBlendFactor::kDestinationAlpha:
      return MTLBlendFactorDestinationAlpha;
    case MetalBlendFactor::kOneMinusDestinationAlpha:
      return MTLBlendFactorOneMinusDestinationAlpha;
    case MetalBlendFactor::kBlendColor:
      return MTLBlendFactorBlendColor;
    case MetalBlendFactor::kOneMinusBlendColor:
      return MTLBlendFactorOneMinusBlendColor;
    case MetalBlendFactor::kBlendAlpha:
      return MTLBlendFactorBlendAlpha;
    case MetalBlendFactor::kOneMinusBlendAlpha:
      return MTLBlendFactorOneMinusBlendAlpha;
    case MetalBlendFactor::kSourceAlphaSaturated:
      return MTLBlendFactorSourceAlphaSaturated;
    default:
      return MTLBlendFactorZero;
  }
}

MTLBlendOperation ToMetalBlendOperation(MetalBlendOperation operation) {
  switch (operation) {
    case MetalBlendOperation::kAdd:
      return MTLBlendOperationAdd;
    case MetalBlendOperation::kSubtract:
      return MTLBlendOperationSubtract;
    case MetalBlendOperation::kMin:
      return MTLBlendOperationMin;
    case MetalBlendOperation::kMax:
      return MTLBlendOperationMax;
    case MetalBlendOperation::kReverseSubtract:
      return MTLBlendOperationReverseSubtract;
    default:
      return MTLBlendOperationAdd;
  }
}

// Applies one normalized RB_BLENDCONTROL + RB_COLOR_MASK nibble to a color-
// attachment descriptor. Shared by the standard render pipeline and the
// geometry-emulation mesh pipeline (both use
// MTLRenderPipelineColorAttachmentDescriptor).
void ApplyRtBlendState(MTLRenderPipelineColorAttachmentDescriptor* rt,
                       uint32_t blend_control, uint32_t color_write_mask) {
  MetalColorAttachmentBlendState state =
      GetMetalColorAttachmentBlendState(blend_control, color_write_mask);
  MTLColorWriteMask write_mask = MTLColorWriteMaskNone;
  if (state.write_mask & 0b0001) {
    write_mask |= MTLColorWriteMaskRed;
  }
  if (state.write_mask & 0b0010) {
    write_mask |= MTLColorWriteMaskGreen;
  }
  if (state.write_mask & 0b0100) {
    write_mask |= MTLColorWriteMaskBlue;
  }
  if (state.write_mask & 0b1000) {
    write_mask |= MTLColorWriteMaskAlpha;
  }
  rt.writeMask = write_mask;
  rt.blendingEnabled = state.blending_required;
  if (!state.blending_required) {
    return;
  }
  rt.sourceRGBBlendFactor = ToMetalBlendFactor(state.rgb.source_factor);
  rt.destinationRGBBlendFactor = ToMetalBlendFactor(state.rgb.destination_factor);
  rt.rgbBlendOperation = ToMetalBlendOperation(state.rgb.operation);
  rt.sourceAlphaBlendFactor = ToMetalBlendFactor(state.alpha.source_factor);
  rt.destinationAlphaBlendFactor = ToMetalBlendFactor(state.alpha.destination_factor);
  rt.alphaBlendOperation = ToMetalBlendOperation(state.alpha.operation);
}

// Maps a geometry-shader type to the MSC input topology the guest primitive
// feeds the emulation with (rect/quad expand from triangles, points from points),
// mirroring XeniOS's GetGeometryStageInputTopology.
MetalShaderConverter::InputTopology GeometryInputTopology(
    PipelineGeometryShader type) {
  switch (type) {
    case PipelineGeometryShader::kPointList:
      return MetalShaderConverter::InputTopology::kPoint;
    case PipelineGeometryShader::kRectangleList:
      return MetalShaderConverter::InputTopology::kTriangle;
    case PipelineGeometryShader::kQuadList:
      // The generated DXBC GS declares line-with-adjacency. MSC's input
      // topology describes the base primitive family; the adjacency arity
      // comes from the shader declaration itself.
      return MetalShaderConverter::InputTopology::kLine;
    default:
      return MetalShaderConverter::InputTopology::kUndefined;
  }
}
}  // namespace

void* MetalPipelineCache::GetOrCreateMemexportPipelineState(
    MetalShader::MetalTranslation* vertex) {
  // Memexport-only draws (VS eA/eM exports): Metal requires a void-returning
  // vertex function for rasterizationEnabled=NO, but MSC-translated shaders
  // return a stage-out struct — so rasterization stays ON into a throwaway
  // scratch depth attachment (DontCare load/store; the command processor's
  // memexport pass provides it). That makes this exactly the existing
  // depth-only pipeline variant: no fragment function, no color attachment.
  uint64_t color_pixel_formats[xenos::kMaxColorRenderTargets] = {};
  uint32_t blend_controls[xenos::kMaxColorRenderTargets] = {};
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    color_pixel_formats[i] = uint64_t(MTLPixelFormatInvalid);
  }
  return GetOrCreateRenderPipelineState(
      vertex, /*pixel=*/nullptr, color_pixel_formats,
      MetalRenderTargetCache::kDepthFormat,
      MetalPipelineRasterKey{/*raster_sample_count=*/1},
      blend_controls, /*color_write_mask=*/0);
}

void* MetalPipelineCache::GetOrCreateRenderPipelineState(
    MetalShader::MetalTranslation* vertex,
    MetalShader::MetalTranslation* pixel, const uint64_t* color_pixel_formats,
    uint64_t depth_pixel_format, MetalPipelineRasterKey raster_key,
    const uint32_t* blend_controls,
    uint32_t color_write_mask) {
  if (!vertex || !vertex->has_metal_function()) {
    return nullptr;
  }

  // 3.2 — full pipeline identity, content-derived so it can back a durable
  // entry. Audit of every input that reaches the descriptor below:
  //   shader identities + modifications  -> vs / ps (MakeShaderKeyIdentity)
  //   null-vs-present pixel shader       -> ps.present
  //   render-target formats              -> colors
  //   depth format                       -> depth (the stencil format is
  //                                         derived from it, never independent)
  //   blend state                        -> blends
  //   write mask                         -> write_mask
  //   sample state                       -> sample_count
  //   vertex layout                      -> NOT APPLICABLE: this backend never
  //                                         sets a vertexDescriptor; MSC-
  //                                         translated shaders fetch vertices
  //                                         themselves, so no layout reaches
  //                                         the pipeline
  //   alphaToCoverageEnabled             -> constant NO (Xenos alpha-to-mask is
  //                                         exact SV_Coverage in the shader),
  //                                         so it is not an identity input
  struct KeyBytes {
    MetalShaderKeyIdentity vs;
    MetalShaderKeyIdentity ps;
    uint64_t colors[xenos::kMaxColorRenderTargets];
    uint64_t depth;
    uint32_t sample_count;
    uint32_t blends[xenos::kMaxColorRenderTargets];
    uint32_t write_mask;
    uint32_t pad;
  } key_bytes = {};
  key_bytes.vs = MakeShaderKeyIdentity(vertex);
  // A pixel translation without a function produces the same depth-only
  // pipeline as no pixel translation at all (see the descriptor below), so it
  // must key the same. The old pointer key got this for free because
  // metal_function() was null in both cases.
  key_bytes.ps = MakeShaderKeyIdentity(
      pixel && pixel->has_metal_function() ? pixel : nullptr);
  key_bytes.depth = depth_pixel_format;
  key_bytes.sample_count = raster_key.raster_sample_count;
  key_bytes.write_mask = color_write_mask;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    key_bytes.colors[i] = color_pixel_formats[i];
    key_bytes.blends[i] = blend_controls[i];
  }
  uint64_t key = XXH3_64bits(&key_bytes, sizeof(key_bytes));
  auto it = render_pipeline_states_.find(key);
  if (it != render_pipeline_states_.end()) {
    return it->second;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLRenderPipelineDescriptor* desc =
      [[MTLRenderPipelineDescriptor alloc] init];
  desc.rasterSampleCount = raster_key.raster_sample_count;
  // Xenos alpha-to-mask is emitted as exact dynamic SV_Coverage by the reused
  // DXBC translator. Enabling Metal alpha-to-coverage here would apply a
  // second, host-defined coverage filter and would incorrectly require alpha
  // state in the pipeline key.
  desc.alphaToCoverageEnabled = NO;
  // Guest shader identities in the pipeline label: Instruments' Shader
  // Timeline attributes GPU samples to the pipeline label (every translated
  // function is named "main"), which is the only per-shader GPU-time
  // attribution on Apple GPUs (no per-draw counter sampling).
  desc.label = [NSString
      stringWithFormat:@"xe vs=%016llX.%llX ps=%016llX.%llX s%u",
                       (unsigned long long)key_bytes.vs.ucode_hash,
                       (unsigned long long)key_bytes.vs.modification,
                       (unsigned long long)key_bytes.ps.ucode_hash,
                       (unsigned long long)key_bytes.ps.modification,
                       raster_key.raster_sample_count];
  desc.vertexFunction = (__bridge id<MTLFunction>)vertex->metal_function();
  if (pixel && pixel->has_metal_function()) {
    desc.fragmentFunction = (__bridge id<MTLFunction>)pixel->metal_function();
  }
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (color_pixel_formats[i] == MTLPixelFormatInvalid) {
      continue;
    }
    desc.colorAttachments[i].pixelFormat =
        (MTLPixelFormat)color_pixel_formats[i];
    // Per-target blending and write mask from RB_BLENDCONTROL[0-3] and the
    // corresponding RB_COLOR_MASK nibble.
    ApplyRtBlendState(desc.colorAttachments[i], blend_controls[i],
                      (color_write_mask >> (i * 4u)) & 0xFu);
  }
  {
    const MTLPixelFormat depth_format =
        (MTLPixelFormat)(depth_pixel_format & ~kDepthWithoutStencilBit);
    if (depth_format != MTLPixelFormatInvalid) {
      desc.depthAttachmentPixelFormat = depth_format;
      if (depth_format == MTLPixelFormatDepth32Float_Stencil8 &&
          !(depth_pixel_format & kDepthWithoutStencilBit)) {
        desc.stencilAttachmentPixelFormat = depth_format;
      }
    }
  }

  NSError* error = nil;
  id<MTLRenderPipelineState> state = nil;
  {
    // Task 1.1 census: host pipeline-state creation, the cost M2 (the binary
    // archive) targets and the one the async stage would move off the frame
    // path. Timed around the API call alone; descriptor assembly above is
    // ours and is not what either mechanism removes.
    MetalShaderBuildTimer timer(GetMetalShaderBuildCensus().pso);
    state = [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!state) {
      timer.Dismiss();
    }
  }
  if (!state) {
    XELOGE("MetalPipelineCache: newRenderPipelineState failed: {}",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    render_pipeline_states_.emplace(key, nullptr);
    return nullptr;
  }
  void* state_ptr = (__bridge_retained void*)state;
  render_pipeline_states_.emplace(key, state_ptr);
  return state_ptr;
}

void* MetalPipelineCache::NewLibraryFromBytes(const std::vector<uint8_t>& bytes,
                                              const char* label) {
  if (!device_ || bytes.empty()) {
    return nullptr;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  dispatch_data_t data = dispatch_data_create(
      bytes.data(), bytes.size(), dispatch_get_main_queue(),
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
#if !__has_feature(objc_arc)
  dispatch_release(data);
#endif
  if (!library) {
    XELOGE("MetalPipelineCache: {} newLibraryWithData failed: {}",
           label ? label : "library",
           error ? [[error localizedDescription] UTF8String] : "unknown");
    return nullptr;
  }
  return (__bridge_retained void*)library;
}

const MetalPipelineCache::GeometryVertexStage*
MetalPipelineCache::GetOrCreateGeometryVertexStage(
    MetalShader::MetalTranslation* vertex, PipelineGeometryShader geometry_type) {
  // The object stage is compiled against the MSC input topology that
  // `geometry_type` selects (point -> kPoint, rect -> kTriangle, quad ->
  // kLine), so two geometry types produce DIFFERENT object stages for the same
  // guest VS. This cache used to key on `vertex` alone and would hand back
  // whichever was built first; the topology is now part of the key. Found while
  // building the durable key for 3.10, where persisting the same mistake would
  // have made it survive across runs instead of only within one.
  const MetalGeometryVertexStageKey stage_key{
      static_cast<const void*>(vertex),
      uint32_t(GeometryInputTopology(geometry_type))};
  auto it = geometry_vertex_stages_.find(stage_key);
  if (it != geometry_vertex_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  // Cache the result even on failure so a bad shader isn't retried every draw.
  GeometryVertexStage stage;

  // 3.10 — durable geometry record. Keyed on every input that affects object-
  // stage generation: the guest shader identity, its translation modification,
  // and the MSC input topology.
  const uint64_t record_key =
      MetalGeometryRecordKey(kMetalGeometryRecordKindVertexStage,
                        MetallibCacheKey(*vertex),
                        uint32_t(GeometryInputTopology(geometry_type)));
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    MetalPipelineStorage::GeometryRecord record;
    if (storage_.LoadGeometryRecord(record_key, &record) &&
        !record.aux_blob.empty()) {
      void* library = NewLibraryFromBytes(record.blob, "geometry VS (cached)");
      void* stage_in =
          NewLibraryFromBytes(record.aux_blob, "geometry VS stage-in (cached)");
      if (library && stage_in) {
        stage.library = library;
        stage.stage_in_library = stage_in;
        stage.function_name = record.function_name;
        stage.vertex_output_size = record.reflection;
        stage.valid = true;
        auto inserted = geometry_vertex_stages_.emplace(stage_key, std::move(stage));
        return &inserted.first->second;
      }
      // Either library failing makes the record unusable as a whole — release
      // whichever loaded and fall through to a full cold build.
      if (library) {
        id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
        (void)lib;
      }
      if (stage_in) {
        id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)stage_in;
        (void)lib;
      }
      XELOGW(
          "MetalPipelineCache: cached geometry VS record {:016X} failed to "
          "load; rebuilding",
          record_key);
    }
  }

  // Reuse the DXBC the normal VS translation already produced (the guest VS runs
  // as the object stage with host_vertex_shader_type kVertex — geometry emulation
  // does the primitive expansion in the GS/mesh stage, so the VS DXBC is
  // identical to the normal path; see XeniOS host-VS-type remap).
  const std::vector<uint8_t>& dxbc_source = vertex->translated_binary();
  if (dxbc_source.empty()) {
    XELOGE("MetalPipelineCache: geometry VS has no DXBC (translate VS first)");
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  // Clip-distance semantic legalization, UNCONDITIONAL. Relabels the
  // clip-distance semantic on the OBJECT STAGE'S OUTPUT so it crosses the
  // object->mesh hop as a private user parameter; without it a geometry-
  // emulation pipeline whose VS writes SV_ClipDistance fails to create and the
  // whole draw class is silently dropped.
  // Inert for clip-count-0 shaders: proven BYTE-IDENTICAL in and out by the
  // transport oracle, which is what makes running it on every geometry stage
  // safe rather than merely convenient.
  std::vector<uint8_t> dxbc = dxbc_source;
  {
    uint32_t n = RelabelDxbcClipDistance(dxbc, DxbcSignatureKind::kOutput);
    if (n) {
      XELOGI("MetalPipelineCache: clip relabel — object-stage output, {} element(s)", n);
    }
  }
  std::string error;
  std::vector<uint8_t> dxil;
  // Cache-aware: the key is the POST-RELABEL buffer above, which no
  // shader-level key could see.
  if (!ConvertDxbcToDxilCached(dxbc, dxil, &error, "geometry-object")) {
    XELOGE("MetalPipelineCache: geometry VS DXBC->DXIL failed: {}", error);
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  MetalStageRequestProbe geom_vs_probe(
      dxil, uint32_t(MetalShaderConverter::Stage::kVertex),
      uint32_t(GeometryInputTopology(geometry_type)));
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kVertex, dxil,
          GeometryInputTopology(geometry_type), result)) {
    XELOGE("MetalPipelineCache: geometry VS emulation convert failed: {}",
           result.error.empty() ? "unknown" : result.error);
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "geometry VS");
  void* stage_in =
      NewLibraryFromBytes(result.stage_in_metallib, "geometry VS stage-in");
  if (!library || !stage_in) {
    if (library) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)library;
      (void)lib;
    }
    if (stage_in) {
      id<MTLLibrary> lib = (__bridge_transfer id<MTLLibrary>)stage_in;
      (void)lib;
    }
    geometry_vertex_stages_.emplace(stage_key, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.stage_in_library = stage_in;
  stage.function_name = result.function_name;
  stage.vertex_output_size = result.vertex_output_size_in_bytes;
  stage.valid = true;
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open() &&
      stage.vertex_output_size) {
    MetalPipelineStorage::GeometryRecord record;
    record.function_name = stage.function_name;
    record.reflection = stage.vertex_output_size;
    record.blob = result.metallib;
    record.aux_blob = result.stage_in_metallib;
    storage_.StoreGeometryRecord(record_key, record);
  }
  auto inserted = geometry_vertex_stages_.emplace(stage_key, std::move(stage));
  return &inserted.first->second;
}

const MetalPipelineCache::GeometryShaderStage*
MetalPipelineCache::GetOrCreateGeometryShaderStage(GeometryShaderKey key) {
  auto it = geometry_shader_stages_.find(key.key);
  if (it != geometry_shader_stages_.end()) {
    return it->second.valid ? &it->second : nullptr;
  }
  GeometryShaderStage stage;

  // 3.10 — durable geometry record. The generated GS is fully determined by
  // GeometryShaderKey (topology type plus the interpolator/clip layout derived
  // from the VS/PS modifications), so that key IS the identity.
  const uint64_t record_key =
      MetalGeometryRecordKey(kMetalGeometryRecordKindShaderStage, key.key, 0);
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open()) {
    MetalPipelineStorage::GeometryRecord record;
    if (storage_.LoadGeometryRecord(record_key, &record)) {
      void* library = NewLibraryFromBytes(record.blob, "geometry GS (cached)");
      if (library) {
        stage.library = library;
        stage.function_name = record.function_name;
        stage.max_input_primitives = record.reflection;
        stage.valid = true;
        auto inserted =
            geometry_shader_stages_.emplace(key.key, std::move(stage));
        return &inserted.first->second;
      }
      XELOGW(
          "MetalPipelineCache: cached geometry GS record {:016X} failed to "
          "load; rebuilding",
          record_key);
    }
  }

  // Generate the DXBC geometry shader (shared point/rect/quad generator), then
  // run it through the same DXBC->DXIL->MSC chain as a MESH stage.
  std::vector<uint32_t> dxbc_dwords;
  CreateDxbcGeometryShader(key, dxbc_dwords);
  if (dxbc_dwords.empty()) {
    XELOGE("MetalPipelineCache: geometry shader generator produced no DXBC");
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  std::vector<uint8_t> dxbc_bytes(dxbc_dwords.size() * sizeof(uint32_t));
  std::memcpy(dxbc_bytes.data(), dxbc_dwords.data(), dxbc_bytes.size());
  // Matching relabel on the MESH STAGE'S INPUT. Its rasterizer-facing OUTPUT
  // deliberately keeps SV_ClipDistance, which is what makes the hardware clip.
  // Unconditional; inert byte-for-byte when the shader has no clip distances.
  {
    {
      uint32_t n =
          RelabelDxbcClipDistance(dxbc_bytes, DxbcSignatureKind::kInput);
      if (n) {
        XELOGI("MetalPipelineCache: clip relabel — mesh-stage input, {} element(s)", n);
      }
    }
  }

  std::string error;
  std::vector<uint8_t> dxil;
  // Cache-aware; same post-relabel keying rule as the object stage.
  if (!ConvertDxbcToDxilCached(dxbc_bytes, dxil, &error, "geometry-mesh")) {
    XELOGE("MetalPipelineCache: geometry GS DXBC->DXIL failed: {}", error);
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  MetalShaderConverter::GeometryEmulationResult result;
  MetalStageRequestProbe geom_gs_probe(
      dxil, uint32_t(MetalShaderConverter::Stage::kGeometry),
      uint32_t(GeometryInputTopology(
          static_cast<PipelineGeometryShader>(key.type))));
  if (!metal_converter_.ConvertForGeometryEmulation(
          MetalShaderConverter::Stage::kGeometry, dxil,
          GeometryInputTopology(static_cast<PipelineGeometryShader>(key.type)),
          result) ||
      (!result.has_mesh_stage && !result.has_geometry_stage)) {
    XELOGE("MetalPipelineCache: geometry GS emulation convert failed: {}",
           result.error.empty() ? "no mesh/geometry stage" : result.error);
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  void* library = NewLibraryFromBytes(result.metallib, "geometry GS");
  if (!library) {
    geometry_shader_stages_.emplace(key.key, std::move(stage));
    return nullptr;
  }
  stage.library = library;
  stage.function_name = result.function_name;
  stage.max_input_primitives = result.gs_max_input_primitives_per_mesh_threadgroup;
  stage.valid = true;
  if (cvars::metal_pipeline_metallib_cache && storage_.is_open() &&
      stage.max_input_primitives) {
    MetalPipelineStorage::GeometryRecord record;
    record.function_name = stage.function_name;
    record.reflection = stage.max_input_primitives;
    record.blob = result.metallib;  // No stage-in for a mesh stage.
    storage_.StoreGeometryRecord(record_key, record);
  }
  auto inserted = geometry_shader_stages_.emplace(key.key, std::move(stage));
  return &inserted.first->second;
}

void* MetalPipelineCache::GetOrCreateGeometryPipelineState(
    MetalShader::MetalTranslation* vertex, MetalShader::MetalTranslation* pixel,
    PipelineGeometryShader geometry_type,
    DxbcShaderTranslator::Modification vertex_modification,
    DxbcShaderTranslator::Modification pixel_modification,
    const uint64_t* color_pixel_formats, uint64_t depth_pixel_format,
    MetalPipelineRasterKey raster_key, const uint32_t* blend_controls,
    uint32_t color_write_mask,
    uint32_t* gs_vertex_size_out, uint32_t* gs_max_input_primitives_out) {
  if (gs_vertex_size_out) {
    *gs_vertex_size_out = 0;
  }
  if (gs_max_input_primitives_out) {
    *gs_max_input_primitives_out = 0;
  }
  // The geometry-emulation pipeline requires a fragment stage (the descriptor
  // dereferences fragmentLibrary); a depth-only geometry path isn't wired yet.
  if (!vertex || !pixel || !pixel->has_metal_function() ||
      geometry_type == PipelineGeometryShader::kNone) {
    return nullptr;
  }

  // The generated GS is keyed by the interpolator/clip layout derived from the
  // VS/PS modifications; returns false if this type needs no geometry shader.
  GeometryShaderKey gs_key;
  if (!GetGeometryShaderKey(geometry_type, vertex_modification,
                            pixel_modification, gs_key)) {
    return nullptr;
  }

  // 3.2 — same content-derived identity as the render-pipeline key. `gs_key` is
  // already content-derived (topology type plus the interpolator/clip layout
  // taken from the VS/PS modifications), so it needs no change.
  GeometryPipelineKeyBytes key_bytes = {};
  key_bytes.gs_key = gs_key.key;
  key_bytes.vs = MakeShaderKeyIdentity(vertex);
  key_bytes.ps = MakeShaderKeyIdentity(pixel);
  key_bytes.depth = depth_pixel_format;
  key_bytes.sample_count = raster_key.raster_sample_count;
  key_bytes.write_mask = color_write_mask;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    key_bytes.colors[i] = color_pixel_formats[i];
    key_bytes.blends[i] = blend_controls[i];
  }
  uint64_t key = XXH3_64bits(&key_bytes, sizeof(key_bytes));
  auto cache_it = geometry_pipelines_.find(key);
  if (cache_it != geometry_pipelines_.end()) {
    if (!cache_it->second.pipeline) {
      // A cached negative is not "one failure": every hit here is a DRAW the
      // failed creation is still suppressing. Count them separately from
      // creation attempts so impact is visible; log the first suppression and
      // then powers of two, joined to the failure line by the key digest.
      uint32_t count = ++cache_it->second.negative_hits;
      if ((count & (count - 1)) == 0) {
        XELOGW(
            "MetalPipelineCache: geometry pipeline negative-cache hit #{} "
            "(draws suppressed by one failed creation) key={:016X}",
            count, key);
      }
      return nullptr;
    }
    if (gs_vertex_size_out) {
      *gs_vertex_size_out = cache_it->second.gs_vertex_size;
    }
    if (gs_max_input_primitives_out) {
      *gs_max_input_primitives_out = cache_it->second.gs_max_input_primitives;
    }
    return cache_it->second.pipeline;
  }

  const GeometryVertexStage* vstage =
      GetOrCreateGeometryVertexStage(vertex, geometry_type);
  const GeometryShaderStage* gstage = GetOrCreateGeometryShaderStage(gs_key);
  if (!vstage || !gstage || !vstage->vertex_output_size ||
      !gstage->max_input_primitives) {
    XELOGE(
        "MetalPipelineCache: geometry pipeline missing stage/reflection "
        "(vs_output={}, gs_max_input={})",
        vstage ? vstage->vertex_output_size : 0,
        gstage ? gstage->max_input_primitives : 0);
    geometry_pipelines_.emplace(key, GeometryPipeline{});
    return nullptr;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;
  MTLMeshRenderPipelineDescriptor* desc =
      [[MTLMeshRenderPipelineDescriptor alloc] init];
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (color_pixel_formats[i] == MTLPixelFormatInvalid) {
      continue;
    }
    desc.colorAttachments[i].pixelFormat =
        (MTLPixelFormat)color_pixel_formats[i];
    ApplyRtBlendState(desc.colorAttachments[i], blend_controls[i],
                      (color_write_mask >> (i * 4u)) & 0xFu);
  }
  {
    const MTLPixelFormat depth_format =
        (MTLPixelFormat)(depth_pixel_format & ~kDepthWithoutStencilBit);
    if (depth_format != MTLPixelFormatInvalid) {
      desc.depthAttachmentPixelFormat = depth_format;
      if (depth_format == MTLPixelFormatDepth32Float_Stencil8 &&
          !(depth_pixel_format & kDepthWithoutStencilBit)) {
        desc.stencilAttachmentPixelFormat = depth_format;
      }
    }
  }
  desc.rasterSampleCount = raster_key.raster_sample_count;
  // The fragment function writes the complete Xenos coverage mask dynamically.
  // Keep the host coverage filter disabled for the same reason as the regular
  // render-pipeline path above.
  desc.alphaToCoverageEnabled = NO;

  id<MTLFunction> ps_fn = (__bridge id<MTLFunction>)pixel->metal_function();
  std::string pixel_function_name = ps_fn.name ? [ps_fn.name UTF8String] : "";

  IRGeometryEmulationPipelineDescriptor ir_desc = {};
  ir_desc.stageInLibrary = (__bridge id<MTLLibrary>)vstage->stage_in_library;
  ir_desc.vertexLibrary = (__bridge id<MTLLibrary>)vstage->library;
  ir_desc.vertexFunctionName = vstage->function_name.c_str();
  ir_desc.geometryLibrary = (__bridge id<MTLLibrary>)gstage->library;
  ir_desc.geometryFunctionName = gstage->function_name.c_str();
  ir_desc.fragmentLibrary = (__bridge id<MTLLibrary>)pixel->metal_library();
  ir_desc.fragmentFunctionName = pixel_function_name.c_str();
  ir_desc.basePipelineDescriptor = desc;
  ir_desc.pipelineConfig.gsVertexSizeInBytes = vstage->vertex_output_size;
  ir_desc.pipelineConfig.gsMaxInputPrimitivesPerMeshThreadgroup =
      gstage->max_input_primitives;

  NSError* error = nil;
  id<MTLRenderPipelineState> pipeline = nil;
  {
    // Task 1.1 census: the geometry-emulation mesh pipeline is a separate
    // stage from the plain PSO — it compiles an object+mesh+fragment triple
    // through the IR runtime — so folding the two would make a title that uses
    // geometry emulation look like one with expensive ordinary pipelines.
    MetalShaderBuildTimer timer(GetMetalShaderBuildCensus().geometry_pso);
    pipeline = IRRuntimeNewGeometryEmulationPipeline(device, &ir_desc, &error);
    if (!pipeline) {
      timer.Dismiss();
    }
  }
  if (!pipeline) {
    // Report the SAME identifying fields the success path logs. Without them a
    // failure is an unattributable string: it names neither which geometry type
    // failed, nor which guest shaders, nor the two pipeline-config values that
    // are the usual cause of a converter rejection. A negative cached below is
    // permanent for the run, so a draw class is silently dropped for the rest
    // of the session — that must be diagnosable from one log.
    XELOGE(
        "MetalPipelineCache: geometry pipeline creation failed: {} "
        "(type={}, gs_vertex_size={}, gs_max_input_primitives={}, "
        "vs_fn='{}', gs_fn='{}', ps_fn='{}') {}",
        error ? [[error localizedDescription] UTF8String] : "unknown",
        uint32_t(geometry_type), vstage->vertex_output_size,
        gstage->max_input_primitives, vstage->function_name,
        gstage->function_name, pixel_function_name,
        DescribeGeometryPipelineKey(key, key_bytes));
    geometry_pipelines_.emplace(key, GeometryPipeline{});
    return nullptr;
  }

  XELOGI(
      "MetalPipelineCache: built geometry pipeline (type={}, gs_vertex_size={}, "
      "gs_max_input_primitives={}, vs_fn='{}', gs_fn='{}') {}",
      uint32_t(geometry_type), vstage->vertex_output_size,
      gstage->max_input_primitives, vstage->function_name,
      gstage->function_name, DescribeGeometryPipelineKey(key, key_bytes));

  GeometryPipeline state;
  state.pipeline = (__bridge_retained void*)pipeline;
  state.gs_vertex_size = vstage->vertex_output_size;
  state.gs_max_input_primitives = gstage->max_input_primitives;
  if (gs_vertex_size_out) {
    *gs_vertex_size_out = state.gs_vertex_size;
  }
  if (gs_max_input_primitives_out) {
    *gs_max_input_primitives_out = state.gs_max_input_primitives;
  }
  auto inserted = geometry_pipelines_.emplace(key, std::move(state));
  return inserted.first->second.pipeline;
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
