/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_pipeline_storage.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ctime>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "third_party/xxhash/xxhash.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"

// 3.6b — size-based bounds, configurable. Not age-based: a large title touched
// rarely is the thing worth evicting, and its age says nothing about its cost.
// 3.1 — one exact rollback switch per mechanism, both default OFF, so a
// regression bisects to one mechanism rather than to "the durable cache".
DEFINE_bool(metal_pipeline_metallib_cache, false,
            "Persist translated metallib across runs (M1). Addresses the "
            "translation cost the census measured at 50-97% of pipeline "
            "creation. Default off until gated.",
            "GPU");
DEFINE_bool(metal_pipeline_binary_archive, false,
            "Persist host pipeline state in an MTLBinaryArchive (M2). "
            "Addresses the 1.7-47.5% the census attributed to "
            "newRenderPipelineState. Default off until gated.",
            "GPU");

DEFINE_uint64(metal_pipeline_cache_title_cap_mib, 512,
              "Maximum durable Metal pipeline-cache size per title, in MiB. "
              "On reaching it, no NEW artifacts are admitted for the session; "
              "existing entries keep being used.",
              "GPU");
DEFINE_uint64(metal_pipeline_cache_global_cap_mib, 4096,
              "Maximum total durable Metal pipeline-cache size across all "
              "titles, in MiB. Enforced between sessions by evicting whole "
              "least-recently-used title directories.",
              "GPU");

namespace xe {
namespace gpu {
namespace metal {

namespace {

constexpr char kManifestName[] = "identity.manifest";
constexpr char kLockName[] = ".lock";
constexpr char kMetallibDirName[] = "metallib";
constexpr char kBinaryArchiveName[] = "pipelines.metalar";
constexpr uint32_t kManifestMagic = 0x584D5053u;  // 'XMPS'

// On-disk manifest. Fixed layout; every field participates in the trust
// decision. last_access_seconds is 3.6c's per-launch access record — filesystem
// atime is commonly disabled or relatime, so it cannot be relied on.
#pragma pack(push, 1)
struct ManifestFile {
  uint32_t magic;
  uint32_t format_version;
  uint32_t translator_schema;
  uint32_t title_id;
  uint64_t dxc_artifact_hash;
  uint64_t msc_artifact_hash;
  uint64_t device_hash;
  uint64_t os_build_hash;
  int64_t last_access_seconds;
  uint64_t reserved;
};
#pragma pack(pop)

uint64_t HashString(const std::string& value) {
  return value.empty() ? 0ull : XXH3_64bits(value.data(), value.size());
}

// 3.3c — hash the ACTUAL translator binary, not a version string it advertises.
// Both translators are dylibs (libdxilconv.dylib, libmetalirconverter.dylib),
// so the loaded image list names them directly. Size plus mtime distinguishes a
// locally patched build from a released one, which is the case that actually
// applies: third_party/DirectXShaderCompiler carries an uncommitted local
// patch, so no released version string could tell the two apart.
//
// Returns 0 when the image is not found, and 0 participates in equality — an
// entry written while a translator was unresolvable is not served once it
// becomes resolvable.
uint64_t HashLoadedDylib(const char* name_fragment) {
  const uint32_t image_count = _dyld_image_count();
  for (uint32_t i = 0; i < image_count; ++i) {
    const char* image_name = _dyld_get_image_name(i);
    if (!image_name || !std::strstr(image_name, name_fragment)) {
      continue;
    }
    struct stat st = {};
    if (stat(image_name, &st) != 0) {
      return 0;
    }
    struct {
      uint64_t size;
      int64_t mtime;
      int64_t mtime_ns;
    } data = {static_cast<uint64_t>(st.st_size), st.st_mtimespec.tv_sec,
              st.st_mtimespec.tv_nsec};
    return HashString(image_name) ^
           (XXH3_64bits(&data, sizeof(data)) * 0x9E3779B97F4A7C15ull);
  }
  return 0;
}

uint64_t DirectorySizeBytes(const std::filesystem::path& dir) {
  std::error_code ec;
  uint64_t total = 0;
  auto it = std::filesystem::recursive_directory_iterator(
      dir, std::filesystem::directory_options::skip_permission_denied, ec);
  if (ec) {
    return 0;
  }
  for (; it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) {
      break;
    }
    std::error_code file_ec;
    if (it->is_regular_file(file_ec) && !file_ec) {
      auto size = it->file_size(file_ec);
      if (!file_ec) {
        total += size;
      }
    }
  }
  return total;
}

int64_t NowSeconds() {
  return static_cast<int64_t>(time(nullptr));
}

}  // namespace

bool MetalPipelineStorageIdentity::operator==(
    const MetalPipelineStorageIdentity& other) const {
  return format_version == other.format_version &&
         translator_schema == other.translator_schema &&
         dxc_artifact_hash == other.dxc_artifact_hash &&
         msc_artifact_hash == other.msc_artifact_hash &&
         device_hash == other.device_hash &&
         os_build_hash == other.os_build_hash;
}

std::string MetalPipelineStorageIdentity::DescribeMismatch(
    const MetalPipelineStorageIdentity& other) const {
  std::string out;
  auto add = [&out](const char* name) {
    if (!out.empty()) {
      out += ", ";
    }
    out += name;
  };
  if (format_version != other.format_version) add("format version");
  if (translator_schema != other.translator_schema) add("translator schema");
  if (dxc_artifact_hash != other.dxc_artifact_hash) add("DXC artifact");
  if (msc_artifact_hash != other.msc_artifact_hash) add("MSC artifact");
  if (device_hash != other.device_hash) add("device");
  if (os_build_hash != other.os_build_hash) add("OS build");
  return out.empty() ? "none" : out;
}

MetalPipelineStorageIdentity ComputeMetalPipelineStorageIdentity(void* device) {
  MetalPipelineStorageIdentity identity;

  // 3.3b — read from MetalTranslatorConfig, the same constants
  // MetalPipelineCache::Initialize constructs the translators from, so the
  // schema cannot drift from the configuration it describes.
  struct SchemaBits {
    uint32_t vendor_apple : 1;
    uint32_t bindless_resources_used : 1;
    uint32_t edram_rov_used : 1;
    uint32_t gamma_render_target_as_unorm8 : 1;
    uint32_t msc_bindless_resources_used : 1;
    uint32_t reserved : 27;
  } schema = {};
  schema.vendor_apple = 1;
  schema.bindless_resources_used = MetalTranslatorConfig::kBindlessResourcesUsed;
  schema.edram_rov_used = MetalTranslatorConfig::kEdramRovUsed;
  schema.gamma_render_target_as_unorm8 =
      MetalTranslatorConfig::kGammaRenderTargetAsUnorm8;
  schema.msc_bindless_resources_used =
      MetalTranslatorConfig::kMscBindlessResourcesUsed;
  std::memcpy(&identity.translator_schema, &schema, sizeof(schema));

  identity.dxc_artifact_hash = HashLoadedDylib("libdxilconv");
  identity.msc_artifact_hash = HashLoadedDylib("libmetalirconverter");
  if (!identity.dxc_artifact_hash || !identity.msc_artifact_hash) {
    XELOGW(
        "MetalPipelineStorage: could not fingerprint a translator dylib "
        "(dxc={:016X}, msc={:016X}) — the durable cache will still key on it, "
        "so a later resolvable build discards rather than reuses",
        identity.dxc_artifact_hash, identity.msc_artifact_hash);
  }

  if (device) {
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
    std::string device_desc;
    if (mtl_device.name) {
      device_desc += [mtl_device.name UTF8String];
    }
    device_desc += fmt::format(
        "|rid={}|lp={}|uma={}", uint64_t(mtl_device.registryID),
        mtl_device.lowPower ? 1 : 0, mtl_device.hasUnifiedMemory ? 1 : 0);
    // GPU family participates because a metallib is compiled against it.
    for (NSInteger family = MTLGPUFamilyApple1; family <= MTLGPUFamilyApple9;
         ++family) {
      if ([mtl_device supportsFamily:(MTLGPUFamily)family]) {
        device_desc += fmt::format("|a{}", int(family - MTLGPUFamilyApple1 + 1));
      }
    }
    identity.device_hash = HashString(device_desc);
  }

  // The OS BUILD, not the marketing version: a build bump can change the
  // shader compiler without changing "26.5.2".
  @autoreleasepool {
    NSProcessInfo* info = [NSProcessInfo processInfo];
    std::string os_desc;
    if (info.operatingSystemVersionString) {
      os_desc = [info.operatingSystemVersionString UTF8String];
    }
    identity.os_build_hash = HashString(os_desc);
  }

  return identity;
}

MetalPipelineStorage::~MetalPipelineStorage() { Close(); }

std::filesystem::path MetalPipelineStorage::metallib_dir() const {
  return title_root_.empty() ? std::filesystem::path()
                             : title_root_ / kMetallibDirName;
}

std::filesystem::path MetalPipelineStorage::binary_archive_path() const {
  return title_root_.empty() ? std::filesystem::path()
                             : title_root_ / kBinaryArchiveName;
}

bool MetalPipelineStorage::Open(
    const std::filesystem::path& cache_root, uint32_t title_id,
    const MetalPipelineStorageIdentity& identity) {
  Close();
  if (cache_root.empty()) {
    return false;
  }

  // 3.6a — numeric title id only, never a game name.
  std::filesystem::path metal_root = cache_root / "metal";
  title_root_ = metal_root / fmt::format("{:08X}", title_id);
  title_id_ = title_id;

  std::error_code ec;
  std::filesystem::create_directories(title_root_, ec);
  if (ec) {
    XELOGW("MetalPipelineStorage: cannot create {}: {} — no durable storage",
           title_root_.string(), ec.message());
    title_root_.clear();
    return false;
  }

  // 3.6d — lock BEFORE eviction considers anything, so this title can never be
  // a candidate for its own eviction and a concurrent instance is visible.
  // Concurrent instances are not hypothetical: a stray second emulator
  // contaminated two census samples.
  if (!AcquireLock()) {
    XELOGW(
        "MetalPipelineStorage: {} is held by another process — no durable "
        "storage this session",
        title_root_.string());
    title_root_.clear();
    return false;
  }

  std::string mismatch;
  if (!LoadAndValidateManifest(identity, &mismatch)) {
    // 3.4 — discard and rebuild rather than partially trusting. Logged, because
    // a silent rebuild is indistinguishable from a cache that never worked.
    XELOGI(
        "MetalPipelineStorage: discarding {} and rebuilding (mismatch: {})",
        title_root_.string(), mismatch);
    ReleaseLock();
    std::filesystem::remove_all(title_root_, ec);
    if (ec) {
      XELOGW("MetalPipelineStorage: could not clear {}: {}",
             title_root_.string(), ec.message());
      title_root_.clear();
      return false;
    }
    std::filesystem::create_directories(title_root_, ec);
    if (ec || !AcquireLock()) {
      title_root_.clear();
      return false;
    }
    if (!WriteManifest(identity)) {
      ReleaseLock();
      title_root_.clear();
      return false;
    }
  }

  std::filesystem::create_directories(metallib_dir(), ec);

  bytes_on_open_ = DirectorySizeBytes(title_root_);
  bytes_admitted_ = 0;
  admission_stopped_ = false;
  is_open_ = true;

  // 3.6c — enforce the global cap between sessions, now that this title is
  // locked and therefore ineligible.
  EvictToGlobalCapLocked(metal_root);

  XELOGI(
      "MetalPipelineStorage: opened {} ({} MiB present, per-title cap {} MiB)",
      title_root_.string(), bytes_on_open_ / (1024 * 1024),
      cvars::metal_pipeline_cache_title_cap_mib);
  return true;
}

void MetalPipelineStorage::Close() {
  if (is_open_) {
    LogSizeReport("close");
  }
  ReleaseLock();
  is_open_ = false;
  title_root_.clear();
  bytes_on_open_ = 0;
  bytes_admitted_ = 0;
  admission_stopped_ = false;
}

bool MetalPipelineStorage::CanAdmit(uint64_t byte_count) const {
  if (!is_open_ || admission_stopped_) {
    return false;
  }
  const uint64_t cap = cvars::metal_pipeline_cache_title_cap_mib * 1024ull * 1024ull;
  return bytes_on_open_ + bytes_admitted_ + byte_count <= cap;
}

void MetalPipelineStorage::RecordAdmitted(uint64_t byte_count) {
  if (!is_open_) {
    return;
  }
  bytes_admitted_ += byte_count;
  const uint64_t cap = cvars::metal_pipeline_cache_title_cap_mib * 1024ull * 1024ull;
  if (!admission_stopped_ && bytes_on_open_ + bytes_admitted_ >= cap) {
    // 3.6e — stop admitting, keep serving. Deleting live entries or rebuilding
    // would thrash exactly when the title is busiest.
    admission_stopped_ = true;
    XELOGW(
        "MetalPipelineStorage: title {:08X} reached its {} MiB cap; no new "
        "artifacts will be admitted this session (existing entries still used)",
        title_id_, cvars::metal_pipeline_cache_title_cap_mib);
  }
}

void MetalPipelineStorage::LogSizeReport(const char* reason) const {
  if (!is_open_) {
    return;
  }
  const uint64_t cap_mib = cvars::metal_pipeline_cache_title_cap_mib;
  const uint64_t metallib_bytes = DirectorySizeBytes(metallib_dir());
  std::error_code ec;
  uint64_t archive_bytes = 0;
  if (std::filesystem::exists(binary_archive_path(), ec) && !ec) {
    archive_bytes = std::filesystem::file_size(binary_archive_path(), ec);
    if (ec) {
      archive_bytes = 0;
    }
  }
  const uint64_t total = metallib_bytes + archive_bytes;
  const double pct_of_cap =
      cap_mib ? 100.0 * double(total) / double(cap_mib * 1024ull * 1024ull) : 0.0;
  // 3.6f — the number that decides whether 512 MiB survives review.
  XELOGI(
      "MetalPipelineStorage SIZE ({}): title {:08X} M1(metallib)={} KiB, "
      "M2(archive)={} KiB, total={} KiB = {:.1f}% of the {} MiB per-title cap",
      reason ? reason : "", title_id_, metallib_bytes / 1024,
      archive_bytes / 1024, total / 1024, pct_of_cap, cap_mib);
  if (pct_of_cap > 50.0) {
    XELOGW(
        "MetalPipelineStorage: title {:08X} exceeds 50% of its per-title "
        "allowance — revise the cap default from this evidence rather than "
        "shipping close to the boundary (task 3.6f)",
        title_id_);
  }
}

namespace {
// Per-entry header for an M1 metallib. Deliberately small: the storage-wide
// provenance was validated at Open, so an entry only has to prove it is the
// entry that was asked for and that it is whole.
#pragma pack(push, 1)
struct MetallibFileHeader {
  uint32_t magic;
  uint32_t format_version;
  uint64_t key;
  uint64_t metallib_size;
};
#pragma pack(pop)
constexpr uint32_t kMetallibMagic = 0x314D4C4Du;  // 'MLM1'
// A metallib far larger than this is a corrupt length field, not a shader.
constexpr uint64_t kMaxMetallibBytes = 64ull * 1024ull * 1024ull;
}  // namespace

bool MetalPipelineStorage::LoadMetallib(uint64_t key,
                                        std::vector<uint8_t>* metallib_out) const {
  if (!is_open_ || !metallib_out) {
    return false;
  }
  const std::filesystem::path path =
      metallib_dir() / fmt::format("{:016X}.metallib", key);
  FILE* file = fopen(path.string().c_str(), "rb");
  if (!file) {
    return false;  // Ordinary miss.
  }
  MetallibFileHeader header = {};
  bool ok = fread(&header, 1, sizeof(header), file) == sizeof(header) &&
            header.magic == kMetallibMagic &&
            header.format_version == MetalPipelineStorageIdentity::kFormatVersion &&
            header.key == key && header.metallib_size &&
            header.metallib_size <= kMaxMetallibBytes;
  if (ok) {
    metallib_out->resize(header.metallib_size);
    ok = fread(metallib_out->data(), 1, header.metallib_size, file) ==
         header.metallib_size;
  }
  fclose(file);
  if (!ok) {
    // 3.5 — corrupt or truncated is a MISS, never a fault. Remove it so the
    // cold translation that follows can republish a whole one.
    metallib_out->clear();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    XELOGW("MetalPipelineStorage: discarded corrupt metallib entry {:016X}", key);
    return false;
  }
  return true;
}

bool MetalPipelineStorage::StoreMetallib(uint64_t key,
                                         const std::vector<uint8_t>& metallib) {
  if (!is_open_ || metallib.empty() || metallib.size() > kMaxMetallibBytes) {
    return false;
  }
  const uint64_t total = sizeof(MetallibFileHeader) + metallib.size();
  // 3.6e — at cap, stop admitting NEW artifacts; existing ones keep serving.
  if (!CanAdmit(total)) {
    RecordAdmitted(0);  // Latches admission_stopped_ and logs once.
    return false;
  }
  std::vector<uint8_t> buffer(total);
  MetallibFileHeader header = {};
  header.magic = kMetallibMagic;
  header.format_version = MetalPipelineStorageIdentity::kFormatVersion;
  header.key = key;
  header.metallib_size = metallib.size();
  std::memcpy(buffer.data(), &header, sizeof(header));
  std::memcpy(buffer.data() + sizeof(header), metallib.data(), metallib.size());
  if (!WriteFileAtomic(metallib_dir() / fmt::format("{:016X}.metallib", key),
                       buffer.data(), buffer.size())) {
    return false;
  }
  RecordAdmitted(total);
  return true;
}

bool MetalPipelineStorage::WriteFileAtomic(const std::filesystem::path& path,
                                           const void* data, size_t size) {
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  FILE* file = fopen(tmp.string().c_str(), "wb");
  if (!file) {
    return false;
  }
  bool ok = !size || fwrite(data, 1, size, file) == size;
  if (fflush(file) != 0) {
    ok = false;
  }
  fclose(file);
  std::error_code ec;
  if (!ok) {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  // Atomic publish — a crash mid-write cannot leave a torn file at `path`.
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

bool MetalPipelineStorage::LoadAndValidateManifest(
    const MetalPipelineStorageIdentity& identity, std::string* mismatch_out) {
  const std::filesystem::path manifest_path = title_root_ / kManifestName;
  FILE* file = fopen(manifest_path.string().c_str(), "rb");
  if (!file) {
    *mismatch_out = "no manifest (new or previously discarded)";
    return false;
  }
  ManifestFile stored = {};
  const size_t read = fread(&stored, 1, sizeof(stored), file);
  fclose(file);
  // 3.5 — truncated or garbage is treated as absent, never as fatal.
  if (read != sizeof(stored) || stored.magic != kManifestMagic) {
    *mismatch_out = "manifest truncated or corrupt";
    return false;
  }
  if (stored.title_id != title_id_) {
    *mismatch_out = "manifest belongs to a different title";
    return false;
  }
  MetalPipelineStorageIdentity on_disk;
  on_disk.format_version = stored.format_version;
  on_disk.translator_schema = stored.translator_schema;
  on_disk.dxc_artifact_hash = stored.dxc_artifact_hash;
  on_disk.msc_artifact_hash = stored.msc_artifact_hash;
  on_disk.device_hash = stored.device_hash;
  on_disk.os_build_hash = stored.os_build_hash;
  if (on_disk != identity) {
    *mismatch_out = identity.DescribeMismatch(on_disk);
    return false;
  }
  // 3.6c — record access once per successful launch, in the manifest. Nothing
  // here depends on filesystem atime.
  return WriteManifest(identity);
}

bool MetalPipelineStorage::WriteManifest(
    const MetalPipelineStorageIdentity& identity) {
  ManifestFile out = {};
  out.magic = kManifestMagic;
  out.format_version = identity.format_version;
  out.translator_schema = identity.translator_schema;
  out.title_id = title_id_;
  out.dxc_artifact_hash = identity.dxc_artifact_hash;
  out.msc_artifact_hash = identity.msc_artifact_hash;
  out.device_hash = identity.device_hash;
  out.os_build_hash = identity.os_build_hash;
  out.last_access_seconds = NowSeconds();
  return WriteFileAtomic(title_root_ / kManifestName, &out, sizeof(out));
}

void MetalPipelineStorage::EvictToGlobalCapLocked(
    const std::filesystem::path& metal_root) {
  const uint64_t global_cap =
      cvars::metal_pipeline_cache_global_cap_mib * 1024ull * 1024ull;
  if (!global_cap) {
    return;
  }

  struct Candidate {
    std::filesystem::path path;
    uint64_t bytes;
    int64_t last_access;
  };
  std::vector<Candidate> candidates;
  uint64_t total = 0;

  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(metal_root, ec)) {
    if (ec) {
      return;
    }
    std::error_code dir_ec;
    if (!entry.is_directory(dir_ec) || dir_ec) {
      continue;
    }
    const uint64_t bytes = DirectorySizeBytes(entry.path());
    total += bytes;
    // 3.6d — never the active title.
    if (entry.path() == title_root_) {
      continue;
    }
    // 3.6d — never a directory another process holds. flock is advisory, so
    // this only sees peers that took it, which is exactly our own instances.
    const std::filesystem::path peer_lock = entry.path() / kLockName;
    int fd = open(peer_lock.string().c_str(), O_RDONLY);
    if (fd >= 0) {
      const bool held = flock(fd, LOCK_EX | LOCK_NB) != 0;
      if (!held) {
        flock(fd, LOCK_UN);
      }
      close(fd);
      if (held) {
        continue;
      }
    }
    int64_t last_access = 0;
    FILE* manifest = fopen((entry.path() / kManifestName).string().c_str(), "rb");
    if (manifest) {
      ManifestFile stored = {};
      if (fread(&stored, 1, sizeof(stored), manifest) == sizeof(stored) &&
          stored.magic == kManifestMagic) {
        last_access = stored.last_access_seconds;
      }
      fclose(manifest);
    }
    candidates.push_back({entry.path(), bytes, last_access});
  }

  if (total <= global_cap) {
    return;
  }

  // Least-recently-used whole directories first. Entry-level LRU is
  // deliberately out of scope for the first implementation (3.6g): whole-
  // directory eviction is bounded and recoverable without it.
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.last_access < b.last_access;
            });

  for (const Candidate& candidate : candidates) {
    if (total <= global_cap) {
      break;
    }
    std::error_code remove_ec;
    std::filesystem::remove_all(candidate.path, remove_ec);
    if (remove_ec) {
      continue;
    }
    total -= std::min(total, candidate.bytes);
    XELOGI(
        "MetalPipelineStorage: evicted {} ({} MiB) to stay under the {} MiB "
        "global cap",
        candidate.path.filename().string(), candidate.bytes / (1024 * 1024),
        cvars::metal_pipeline_cache_global_cap_mib);
  }
}

bool MetalPipelineStorage::AcquireLock() {
  lock_path_ = title_root_ / kLockName;
  lock_fd_ = open(lock_path_.string().c_str(), O_CREAT | O_RDWR, 0644);
  if (lock_fd_ < 0) {
    return false;
  }
  if (flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
    close(lock_fd_);
    lock_fd_ = -1;
    return false;
  }
  return true;
}

void MetalPipelineStorage::ReleaseLock() {
  if (lock_fd_ >= 0) {
    flock(lock_fd_, LOCK_UN);
    close(lock_fd_);
    lock_fd_ = -1;
  }
  lock_path_.clear();
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe
