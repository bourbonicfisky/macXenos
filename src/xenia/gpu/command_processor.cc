/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/command_processor.h"

#include <cstdlib>

#include <algorithm>
#include <ctime>
#include <tuple>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/byte_stream.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"
#include "xenia/base/xxhash.h"
#if XE_PLATFORM_MAC
#include <sys/clonefile.h>
#endif
#include "xenia/emulator.h"
#include "xenia/gpu/edram_layout.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/packet_disassembler.h"
#include "xenia/gpu/sampler_info.h"
#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/texture_info.h"
#include "xenia/gpu/xenos_zpd_report.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#if __has_include("version.h")
#include "version.h"  // XE_BUILD_COMMIT / XE_BUILD_BRANCH (generated)
#endif
#if !defined(NDEBUG)

#define XE_ENABLE_GPU_REG_WRITE_LOGGING 1
#endif
DEFINE_bool(
    log_guest_driven_gpu_register_written_values, false,
    "Only does anything in debug builds, if set will log every write to a gpu "
    "register done by a guest. Does not log writes that are done by the CP on "
    "its own, just ones the guest makes or instructs it to make.",
    "GPU");

DEFINE_bool(disassemble_pm4, false,
            "Only does anything in debug builds, if set will disassemble and "
            "log all PM4 packets sent to the CP.",
            "GPU");

DEFINE_bool(log_fps, false,
            "Log the guest frame rate (swap rate) every 5 seconds.", "GPU");

DEFINE_bool(
    log_ringbuffer_kickoff_initiator_bts, false,
    "Only does anything in debug builds, if set will log the pseudo-stacktrace "
    "of the guest thread that wrote the new read position.",
    "GPU");

DEFINE_bool(clear_memory_page_state, false,
            "Refresh state of memory pages to enable gpu written data. (Use "
            "for 'Team Ninja' Games to fix missing character models)",
            "GPU");

DEFINE_int32(replay_draw_first, 0,
             "First draw index (inclusive, counted per trace playback) to "
             "actually issue to the backend. Draws outside [replay_draw_first"
             ", replay_draw_last] are skipped. For bisecting trace replays; "
             "no effect on live play at defaults.",
             "GPU");

DEFINE_int32(replay_draw_last, INT32_MAX,
             "Last draw index (inclusive) to actually issue to the backend "
             "during trace playback. See replay_draw_first.",
             "GPU");

DEFINE_int32(replay_draw_include, -1,
             "Draw index that is always issued during trace playback even "
             "when outside [replay_draw_first, replay_draw_last] (e.g. the "
             "frame's final EDRAM resolve, so scene-window bisection still "
             "produces displayable output).",
             "GPU");

DEFINE_int32(replay_draw_skip_first, -1,
             "First draw index (inclusive) of a range to SKIP during trace "
             "playback (in addition to the [first, last] window). -1 "
             "disables. For crack-creator bisection with intact exposure.",
             "GPU");

DEFINE_int32(replay_draw_skip_last, -1,
             "Last draw index (inclusive) of the skip range. See "
             "replay_draw_skip_first.",
             "GPU");

DEFINE_bool(replay_skip_depth_only, false,
            "Skip kDepthOnly (Z-prepass) draws during trace playback - "
            "diagnostic for prepass-vs-color-pass depth mismatches.",
            "GPU");

DEFINE_int32(replay_zfunc_always_draw, -1,
             "Draw index whose depth test function is forced to ALWAYS "
             "during trace playback (diagnostic: distinguishes depth-test "
             "failures from rasterization coverage gaps). -1 disables.",
             "GPU");

DEFINE_string(replay_index_window, "",
              "draw,start,count - restrict one draw during trace playback to "
              "a window of its index buffer (diagnostic: isolate strip "
              "segments/triangles; count=0 skips the draw). Empty disables.",
              "GPU");

DEFINE_int32(replay_solid_draw, -1,
             "Draw index whose RT0 blending is forced to REPLACE "
             "(src ONE, dst ZERO, ADD) and whose fixed-function alpha test / "
             "alpha-to-mask are disabled during trace playback (diagnostic: "
             "any pixel the draw rasterizes shows its raw fragment color, "
             "distinguishing absent fragments - cull/clip/coverage - from "
             "zero-alpha blend contributions). -1 disables.",
             "GPU");

DEFINE_int32(replay_duplicate_draw_stencil, -1,
             "Draw index that is issued a SECOND time during trace playback "
             "with zfunc=LESS, depth writes off, and stencil REPLACE (ref "
             "0xAB) on depth pass. Any sample whose re-rasterized depth is "
             "strictly below the just-written depth of the identical draw "
             "gets stencil-marked - a guest-visible detector of host "
             "fixed-function depth non-repeatability (diagnostic for the "
             "Apple wireframe-mesh hunt). -1 disables.",
             "GPU");

DEFINE_bool(replay_duplicate_draw_depth_only, false,
            "Make the second instance of replay_duplicate_draw_stencil run "
            "as a kDepthOnly (fragment-less, Z-prepass-like) draw, to test "
            "depth rasterization invariance BETWEEN pipeline variants of "
            "the same geometry rather than within one pipeline.",
            "GPU");

DEFINE_bool(replay_draw_log, false,
            "Log one line per draw (index, primitive, index count, EDRAM "
            "mode, shader ucode hashes) during trace playback bisection.",
            "GPU");

DEFINE_string(
    capture_dir, "",
    "Headless trace-analysis capture directory. When set during trace "
    "playback, writes one structured JSON row per draw to "
    "<dir>/manifest.jsonl (byte-stable per-draw guest/register state for "
    "cross-machine and cross-build divergence compare) plus run/build/device "
    "provenance to <dir>/capture_meta.json. Empty disables (default output "
    "unchanged).",
    "GPU");

DEFINE_string(
    capture_draw_range, "",
    "With --capture_dir: 'A..B' - snapshot raw EDRAM to "
    "<dir>/draw_NNNNNN.edram.bin AFTER each issued draw in the inclusive "
    "index range [A,B]. Selective and opt-in; the snapshot forces a "
    "submission + queue drain (NOT passive - may perturb, per design D3). "
    "Empty disables EDRAM snapshots.",
    "GPU");

DEFINE_string(
    capture_edram_at, "",
    "With --capture_dir: comma-separated explicit draw indices (e.g. "
    "'101,108,115') to snapshot raw EDRAM after, in addition to any "
    "--capture_draw_range. Empty disables.",
    "GPU");

DEFINE_bool(
    capture_edram_hashes, false,
    "With --capture_dir: after EVERY draw, append per-tile, per-physical-"
    "sample-plane XXH3-64 hashes of the raw EDRAM image to "
    "<dir>/edram_hashes.bin (16-byte header: 'XEDH', version, tile_count, "
    "tile_bytes; then records of uint32 draw index + tile_count*4 uint64 "
    "hashes, little-endian). Sample planes use the canonical 32bpp 4x "
    "physical layout. Full-frame first-divergence scanning at ~64 KB/draw "
    "instead of 10 MB snapshots. Forces a submission + queue drain per draw "
    "(NOT passive, design D3).",
    "GPU");

DEFINE_string(
    serialize_draw_range, "",
    "'A..B' (inclusive): force a GPU submission + wait after each draw whose "
    "index is in the range, WITHOUT any capture output. Hazard-bisection "
    "probe: draws in the range execute against fully-drained GPU state, so "
    "corruption that disappears when a range is serialized implicates a GPU "
    "ordering hazard among those draws (tools/bisect_stipple.sh drives a "
    "binary search on this). Same draw-index space as --capture_draw_range. "
    "Empty disables.",
    "GPU");

DEFINE_string(
    replay_inject_memory, "",
    "Trace-replay diagnostic: semicolon-separated list of "
    "hexaddr,afterdraw,filepath - before issuing draw index 'afterdraw', "
    "copy the file's contents into guest physical memory at hexaddr and "
    "invalidate caches (e.g. substitute another host's resolved shadow map "
    "for cross-GPU divergence attribution).",
    "GPU");

DEFINE_string(
    occlusion_query, "fast",
    "Controls hardware occlusion query behavior for EVENT_WRITE_ZPD.\n"
    "Used for effects like lens flares, object culling, and auto-exposure.\n"
    " fake: Write a fake result without asking the GPU. Safe for most games,\n"
    "       though some effects may look slightly wrong.\n"
    " fast: Ask the GPU but don't wait for the answer. Writes a cached\n"
    "       result immediately and updates it when the GPU catches up.\n"
    "       Cached results bias toward visible when guessing. (default)\n"
    " fast-alt: Variant of fast mode that keeps cached zero results for\n"
    "           unresolved reports. May improve effects relying on precise\n"
    "           visibility, but may be less stable for occlusion culling.\n"
    " strict: Ask the GPU and wait for the real result before continuing.\n"
    "         Most accurate, but may be somewhat less performant.",
    "GPU");

DEFINE_string(
    readback_resolve, "none",
    "Controls CPU readback of render-to-texture resolve results.\n"
    " fast: Read from previous frame (1 frame delay, no GPU stall, slight "
    "performance hit)\n"
    " full: Wait for GPU to finish (accurate but slow, GPU-CPU sync stall)\n"
    " none: Disable readback completely (some games render better without it)",
    "GPU");

UPDATE_from_string(readback_resolve, 2025, 12, 4, 21, "fast");

DEFINE_bool(
    readback_memexport, false,
    "Read data written by memory export in shaders on the CPU. "
    "This may be needed in some games (but many only access exported data on "
    "the GPU, and this flag isn't needed to handle such behavior), but causes "
    "mid-frame synchronization, so it has a huge performance impact.",
    "GPU");

namespace xe {
namespace gpu {

namespace {

std::string JsonEscape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  constexpr char kHex[] = "0123456789ABCDEF";
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (c < 0x20) {
          escaped += "\\u00";
          escaped.push_back(kHex[c >> 4]);
          escaped.push_back(kHex[c & 0xF]);
        } else {
          escaped.push_back(char(c));
        }
        break;
    }
  }
  return escaped;
}

std::string SanitizeTraceFileToken(std::string_view value,
                                   size_t max_length = 48) {
  std::string token;
  token.reserve(std::min(value.size(), max_length));
  bool pending_separator = false;
  for (unsigned char c : value) {
    bool ascii_alnum = (c >= 'A' && c <= 'Z') ||
                       (c >= 'a' && c <= 'z') ||
                       (c >= '0' && c <= '9');
    if (ascii_alnum) {
      if (pending_separator && !token.empty() && token.size() < max_length) {
        token.push_back('_');
      }
      pending_separator = false;
      if (token.size() < max_length) {
        token.push_back(char(c));
      }
    } else {
      pending_separator = true;
    }
    if (token.size() >= max_length) {
      break;
    }
  }
  while (!token.empty() && token.back() == '_') {
    token.pop_back();
  }
  return token;
}

const char* HostOperatingSystemName() {
#if XE_PLATFORM_MAC
  return "macOS";
#elif XE_PLATFORM_WIN32
  return "Windows";
#elif XE_PLATFORM_LINUX
  return "Linux";
#else
  return "unknown";
#endif
}

const char* HostArchitectureName() {
#if XE_ARCH_ARM64
  return "arm64";
#elif XE_ARCH_AMD64
  return "x86_64";
#else
  return "unknown";
#endif
}

const char* HostPlatformClass() {
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
  return "apple-silicon";
#elif XE_PLATFORM_MAC && XE_ARCH_AMD64
  return "intel-mac";
#elif XE_PLATFORM_WIN32 && XE_ARCH_ARM64
  return "windows-arm64";
#elif XE_PLATFORM_WIN32 && XE_ARCH_AMD64
  return "windows-x64";
#elif XE_PLATFORM_LINUX && XE_ARCH_ARM64
  return "linux-arm64";
#elif XE_PLATFORM_LINUX && XE_ARCH_AMD64
  return "linux-x64";
#else
  return "unknown-host";
#endif
}

std::string UtcTimestamp(bool compact) {
  std::time_t now = std::time(nullptr);
  std::tm utc = {};
#if XE_PLATFORM_WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  char buffer[32] = {};
  std::strftime(buffer, sizeof(buffer),
                compact ? "%Y%m%dT%H%M%SZ" : "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

std::string RendererFileToken(std::string_view backend_name) {
  std::string backend_lower;
  backend_lower.reserve(backend_name.size());
  for (unsigned char c : backend_name) {
    backend_lower.push_back(
        (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : char(c));
  }
  if (backend_lower.rfind("vulkan", 0) == 0) {
#if XE_PLATFORM_MAC
    return "vulkan-moltenvk";
#else
    return "vulkan";
#endif
  }
  if (backend_lower.rfind("metal", 0) == 0) {
    return "metal";
  }
  if (backend_lower.rfind("direct3d", 0) == 0 ||
      backend_lower.rfind("d3d12", 0) == 0) {
    return "d3d12";
  }
  std::string token = SanitizeTraceFileToken(backend_lower, 32);
  return token.empty() ? "unknown-renderer" : token;
}

std::string RendererApiName(std::string_view renderer_token) {
  if (renderer_token.rfind("vulkan", 0) == 0) {
    return "Vulkan";
  }
  if (renderer_token == "metal") {
    return "Metal";
  }
  if (renderer_token == "d3d12") {
    return "Direct3D 12";
  }
  return std::string(renderer_token);
}

bool HashFileXXH3(const std::filesystem::path& path, uint64_t* hash_out) {
  FILE* f = xe::filesystem::OpenFile(path, "rb");
  if (!f) {
    return false;
  }
  XXH3_state_t state = {};
  if (XXH3_64bits_reset(&state) == XXH_ERROR) {
    fclose(f);
    return false;
  }
  std::vector<uint8_t> buffer(256 * 1024);
  while (true) {
    size_t read = fread(buffer.data(), 1, buffer.size(), f);
    if (read && XXH3_64bits_update(&state, buffer.data(), read) == XXH_ERROR) {
      fclose(f);
      return false;
    }
    if (read != buffer.size()) {
      bool ok = !ferror(f);
      fclose(f);
      if (!ok) {
        return false;
      }
      *hash_out = XXH3_64bits_digest(&state);
      return true;
    }
  }
}

std::filesystem::path TraceBundlePath(const std::filesystem::path& trace_path) {
  std::filesystem::path path = trace_path;
  path.replace_extension(kTraceBundleExtension);
  return path;
}

std::filesystem::path TraceMetadataPath(
    const std::filesystem::path& trace_path) {
  std::filesystem::path path = trace_path;
  path += ".meta.json";
  return path;
}

bool CloneOrCopyFile(const std::filesystem::path& source,
                     const std::filesystem::path& destination,
                     bool* cloned_out) {
  *cloned_out = false;
#if XE_PLATFORM_MAC
  if (clonefile(source.c_str(), destination.c_str(), 0) == 0) {
    *cloned_out = true;
    return true;
  }
  // clonefile may leave a destination after a filesystem-specific failure.
  std::error_code remove_error;
  std::filesystem::remove(destination, remove_error);
#endif
  std::error_code copy_error;
  return std::filesystem::copy_file(
      source, destination, std::filesystem::copy_options::none, copy_error);
}

}  // namespace

// This should be written completely differently with support for different
// types.
void SaveGPUSetting(GPUSetting setting, uint64_t value) {
  switch (setting) {
    case GPUSetting::ClearMemoryPageState:
      OVERRIDE_bool(clear_memory_page_state, static_cast<bool>(value));
      break;
    case GPUSetting::ReadbackMemexport:
      OVERRIDE_bool(readback_memexport, static_cast<bool>(value));
      break;
  }
}

bool GetGPUSetting(GPUSetting setting) {
  switch (setting) {
    case GPUSetting::ClearMemoryPageState:
      return cvars::clear_memory_page_state;
    case GPUSetting::ReadbackMemexport:
      return cvars::readback_memexport;
  }
  return false;
}

ReadbackResolveMode GetReadbackResolveMode() {
  const std::string& mode = cvars::readback_resolve;
  if (mode == "full") {
    return ReadbackResolveMode::kFull;
  } else if (mode == "none") {
    return ReadbackResolveMode::kDisabled;
  } else {
    // Default to "fast" for any unrecognized value
    return ReadbackResolveMode::kFast;
  }
}

void SetReadbackResolveMode(const std::string& mode) {
  OVERRIDE_string(readback_resolve, mode);
}

ZPDMode GetZPDMode() {
  const std::string& mode = cvars::occlusion_query;
  if (mode == "fake") {
    return ZPDMode::kFake;
  } else if (mode == "strict") {
    return ZPDMode::kStrict;
  } else if (mode == "fast-alt") {
    return ZPDMode::kFastAlt;
  }
  return ZPDMode::kFast;
}

void SetZPDMode(const std::string& mode) {
  OVERRIDE_string(occlusion_query, mode);
}

using namespace xe::gpu::xenos;

CommandProcessor::CommandProcessor(GraphicsSystem* graphics_system,
                                   kernel::KernelState* kernel_state)
    : reader_(nullptr, 0),
      memory_(graphics_system->memory()),
      kernel_state_(kernel_state),
      graphics_system_(graphics_system),
      register_file_(graphics_system_->register_file()),
      trace_writer_(graphics_system->memory()->physical_membase()),
      worker_running_(true),
      write_ptr_index_event_(xe::threading::Event::CreateAutoResetEvent(false)),
      write_ptr_index_(0) {
  assert_not_null(write_ptr_index_event_);
}

CommandProcessor::~CommandProcessor() = default;

bool CommandProcessor::Initialize() {
  // Initialize the gamma ramps to their default (linear) values - taken from
  // what games set when starting with the sRGB (return value 1)
  // VdGetCurrentDisplayGamma.
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t value = i * 0x3FF / 0xFF;
    reg::DC_LUT_30_COLOR& gamma_ramp_entry = gamma_ramp_256_entry_table_[i];
    gamma_ramp_entry.color_10_blue = value;
    gamma_ramp_entry.color_10_green = value;
    gamma_ramp_entry.color_10_red = value;
  }
  for (uint32_t i = 0; i < 128; ++i) {
    reg::DC_LUT_PWL_DATA gamma_ramp_entry = {};
    gamma_ramp_entry.base = (i * 0xFFFF / 0x7F) & ~UINT32_C(0x3F);
    gamma_ramp_entry.delta = i < 0x7F ? 0x200 : 0;
    for (uint32_t j = 0; j < 3; ++j) {
      gamma_ramp_pwl_rgb_[i][j] = gamma_ramp_entry;
    }
  }

  worker_running_ = true;
  worker_thread_ =
      kernel::object_ref<kernel::XHostThread>(new kernel::XHostThread(
          kernel_state_, 128 * 1024, 0,
          [this]() {
            WorkerThreadMain();
            return 0;
          },
          kernel_state_->GetIdleProcess()));
  worker_thread_->set_name("GPU Commands");
  worker_thread_->Create();

  return true;
}

void CommandProcessor::Shutdown() {
  EndTracing();

  worker_running_ = false;
  write_ptr_index_event_->Set();
  worker_thread_->Wait(0, 0, 0, nullptr);
  worker_thread_.reset();
}

void CommandProcessor::InitializeShaderStorage(
    const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
    std::function<void()> completion_callback) {
  if (completion_callback) {
    completion_callback();
  }
}

void CommandProcessor::RequestFrameTrace(
    const std::filesystem::path& root_path) {
  if (trace_state_ == TraceState::kStreaming) {
    XELOGE("Streaming trace; cannot also trace frame.");
    return;
  }
  if (trace_state_ == TraceState::kSingleFrame) {
    XELOGE("Frame trace already pending; ignoring.");
    return;
  }
  trace_state_ = TraceState::kSingleFrame;
  trace_frame_path_ = root_path;
}

void CommandProcessor::BeginTracing(const std::filesystem::path& root_path) {
  if (trace_state_ == TraceState::kStreaming) {
    XELOGE("Streaming already active; ignoring request.");
    return;
  }
  if (trace_state_ == TraceState::kSingleFrame) {
    XELOGE("Frame trace pending; ignoring streaming request.");
    return;
  }
  // Streaming starts on the next primary buffer execute.
  trace_state_ = TraceState::kStreaming;
  trace_stream_path_ = root_path;
}

void CommandProcessor::EndTracing() {
  if (!trace_writer_.is_open()) {
    return;
  }
  assert_true(trace_state_ == TraceState::kStreaming);
  trace_state_ = TraceState::kDisabled;
  CloseTrace();
}

std::string CommandProcessor::BuildTraceFileName(uint32_t title_id,
                                                 uint32_t frame_index,
                                                 bool streaming) const {
  std::string title_name;
  if (kernel_state_ && kernel_state_->emulator()) {
    title_name = SanitizeTraceFileToken(
        kernel_state_->emulator()->title_name());
  }
  std::string title_component =
      title_name.empty() ? std::string() : fmt::format("_{}", title_name);

  uint32_t width = last_swap_width_;
  uint32_t height = last_swap_height_;
  if ((!width || !height) && graphics_system_) {
    std::tie(width, height) = graphics_system_->GetResolution();
  }
  std::string renderer = RendererFileToken(
      graphics_system_ ? graphics_system_->name() : "unknown");
  std::string position = streaming
                             ? "stream"
                             : fmt::format("frame_{:08d}", frame_index);
  return fmt::format(
      "{:08X}{}__{}__{}x{}__{}__{}__{}.{}", title_id, title_component,
      HostPlatformClass(), width, height, renderer, position,
      UtcTimestamp(true), kTraceExtension);
}

bool CommandProcessor::OpenTrace(const std::filesystem::path& path,
                                 uint32_t title_id, uint32_t frame_index,
                                 bool streaming) {
  std::filesystem::path output_path = path;
  for (uint32_t duplicate = 2;; ++duplicate) {
    std::error_code exists_error;
    bool artifact_exists = std::filesystem::exists(output_path, exists_error);
    if (!exists_error && !artifact_exists) {
      artifact_exists =
          std::filesystem::exists(TraceBundlePath(output_path), exists_error);
    }
    if (!exists_error && !artifact_exists) {
      artifact_exists =
          std::filesystem::exists(TraceMetadataPath(output_path), exists_error);
    }
    if (exists_error) {
      XELOGE("Failed to check GPU trace output path {}: {}",
             xe::path_to_utf8(output_path), exists_error.message());
      return false;
    }
    if (!artifact_exists) {
      break;
    }
    output_path = path.parent_path() /
                  fmt::format("{}_{}{}", path.stem().string(), duplicate,
                              path.extension().string());
  }
  if (!trace_writer_.Open(output_path, title_id)) {
    XELOGE("Failed to open GPU trace {}", xe::path_to_utf8(output_path));
    return false;
  }
  active_trace_path_ = std::filesystem::absolute(output_path);
  active_trace_title_id_ = title_id;
  active_trace_frame_index_ = frame_index;
  active_trace_width_ = last_swap_width_;
  active_trace_height_ = last_swap_height_;
  if ((!active_trace_width_ || !active_trace_height_) && graphics_system_) {
    std::tie(active_trace_width_, active_trace_height_) =
        graphics_system_->GetResolution();
  }
  active_trace_streaming_ = streaming;
  active_trace_started_utc_ = UtcTimestamp(false);
  XELOGI("Capturing GPU trace to {}", xe::path_to_utf8(active_trace_path_));
  return true;
}

void CommandProcessor::CloseTrace() {
  if (!trace_writer_.is_open()) {
    return;
  }
  trace_writer_.Close();

  std::error_code size_error;
  uintmax_t trace_size =
      std::filesystem::file_size(active_trace_path_, size_error);
  if (size_error) {
    trace_size = 0;
  }
  uint64_t trace_hash = 0;
  bool trace_hash_valid =
      !size_error && HashFileXXH3(active_trace_path_, &trace_hash);
  std::string metadata =
      BuildTraceMetadataJson(trace_size, trace_hash_valid, trace_hash);
  bool bundle_written =
      trace_hash_valid &&
      WriteTraceBundle(metadata, trace_size, trace_hash);
  bool metadata_written = WriteTraceMetadata(metadata);
  if (!trace_hash_valid) {
    XELOGE("Failed to hash completed GPU trace {}; v2 bundle not written",
           xe::path_to_utf8(active_trace_path_));
  }
  XELOGI("GPU trace complete: legacy={} portable={} metadata={}",
         xe::path_to_utf8(active_trace_path_),
         bundle_written
             ? xe::path_to_utf8(TraceBundlePath(active_trace_path_))
             : "ERROR",
         metadata_written
             ? xe::path_to_utf8(TraceMetadataPath(active_trace_path_))
             : "ERROR");
  active_trace_path_.clear();
  active_trace_title_id_ = 0;
  active_trace_frame_index_ = 0;
  active_trace_width_ = 0;
  active_trace_height_ = 0;
  active_trace_streaming_ = false;
  active_trace_started_utc_.clear();
}

void CommandProcessor::AbortTrace(std::string_view reason) {
  if (!trace_writer_.is_open()) {
    return;
  }
  std::filesystem::path aborted_path = active_trace_path_;
  trace_writer_.Close();
  trace_state_ = TraceState::kDisabled;
  active_trace_path_.clear();
  active_trace_title_id_ = 0;
  active_trace_frame_index_ = 0;
  active_trace_width_ = 0;
  active_trace_height_ = 0;
  active_trace_streaming_ = false;
  active_trace_started_utc_.clear();
  XELOGE("GPU trace aborted: path={} reason={}",
         xe::path_to_utf8(aborted_path), reason);
}

std::string CommandProcessor::BuildTraceMetadataJson(
    uintmax_t trace_size, bool trace_hash_valid, uint64_t trace_hash) const {
  if (active_trace_path_.empty()) {
    return std::string();
  }

#if defined(NDEBUG)
  const char* build_config = "Release";
#else
  const char* build_config = "Debug";
#endif
#if defined(XE_BUILD_COMMIT)
  const char* build_commit = XE_BUILD_COMMIT;
  const char* build_branch = XE_BUILD_BRANCH;
#else
  const char* build_commit = "unknown";
  const char* build_branch = "unknown";
#endif

  std::string title_name;
  std::string title_version;
  if (kernel_state_ && kernel_state_->emulator()) {
    title_name = kernel_state_->emulator()->title_name();
    title_version = kernel_state_->emulator()->title_version();
  }
  std::string backend_name =
      graphics_system_ ? graphics_system_->name() : "unknown";
  std::string renderer_token = RendererFileToken(backend_name);
  std::string renderer_api = RendererApiName(renderer_token);
  std::string translation_layer = "null";
  std::string host_api = renderer_api;
#if XE_PLATFORM_MAC
  if (renderer_token == "vulkan-moltenvk") {
    translation_layer = "\"MoltenVK\"";
    host_api = "Metal";
  }
#endif
  std::string device_json = GetCaptureDeviceInfoJson();
  if (device_json.empty()) {
    device_json = "null";
  }
  std::string trace_hash_json =
      trace_hash_valid ? fmt::format("\"{:016X}\"", trace_hash) : "null";
  std::filesystem::path bundle_path = TraceBundlePath(active_trace_path_);
  std::filesystem::path metadata_path = TraceMetadataPath(active_trace_path_);

  return fmt::format(
      "{{\n"
      "  \"schema\": 2,\n"
      "  \"complete\": true,\n"
      "  \"trace_format_version\": {},\n"
      "  \"bundle_format_version\": {},\n"
      "  \"trace_file\": \"{}\",\n"
      "  \"bundle_file\": \"{}\",\n"
      "  \"metadata_file\": \"{}\",\n"
      "  \"trace_size_bytes\": {},\n"
      "  \"trace_xxh3_64\": {},\n"
      "  \"capture_kind\": \"{}\",\n"
      "  \"frame_index\": {},\n"
      "  \"captured_utc\": \"{}\",\n"
      "  \"title\": {{\"id\": \"{:08X}\", \"name\": \"{}\", "
      "\"version\": \"{}\"}},\n"
      "  \"guest\": {{\"display_width\": {}, \"display_height\": {}}},\n"
      "  \"host\": {{\"platform\": \"{}\", \"os\": \"{}\", "
      "\"architecture\": \"{}\"}},\n"
      "  \"renderer\": {{\"backend\": \"{}\", \"api\": \"{}\", "
      "\"translation_layer\": {}, \"host_api\": \"{}\", "
      "\"device\": {}}},\n"
      "  \"build\": {{\"commit\": \"{}\", \"branch\": \"{}\", "
      "\"config\": \"{}\"}}\n"
      "}}\n",
      kTraceFormatVersion, kTraceBundleFormatVersion,
      JsonEscape(active_trace_path_.filename().string()),
      JsonEscape(bundle_path.filename().string()),
      JsonEscape(metadata_path.filename().string()), trace_size, trace_hash_json,
      active_trace_streaming_ ? "stream" : "single-frame",
      active_trace_frame_index_, JsonEscape(active_trace_started_utc_),
      active_trace_title_id_, JsonEscape(title_name), JsonEscape(title_version),
      active_trace_width_, active_trace_height_, HostPlatformClass(),
      HostOperatingSystemName(), HostArchitectureName(), JsonEscape(backend_name),
      JsonEscape(renderer_api), translation_layer, JsonEscape(host_api),
      device_json, JsonEscape(build_commit), JsonEscape(build_branch),
      build_config);
}

bool CommandProcessor::WriteTraceMetadata(std::string_view metadata) const {
  std::filesystem::path meta_path = TraceMetadataPath(active_trace_path_);
  FILE* f = xe::filesystem::OpenFile(meta_path, "wb");
  if (!f) {
    XELOGE("Failed to open GPU trace metadata {}",
           xe::path_to_utf8(meta_path));
    return false;
  }
  bool written = fwrite(metadata.data(), 1, metadata.size(), f) ==
                 metadata.size();
  fclose(f);
  if (!written) {
    XELOGE("Failed to write GPU trace metadata {}",
           xe::path_to_utf8(meta_path));
  }
  return written;
}

bool CommandProcessor::WriteTraceBundle(std::string_view metadata,
                                        uintmax_t trace_size,
                                        uint64_t trace_hash) const {
  std::filesystem::path bundle_path = TraceBundlePath(active_trace_path_);
  bool cloned = false;
  if (!CloneOrCopyFile(active_trace_path_, bundle_path, &cloned)) {
    XELOGE("Failed to create portable GPU trace bundle {}",
           xe::path_to_utf8(bundle_path));
    return false;
  }
  FILE* f = xe::filesystem::OpenFile(bundle_path, "ab");
  if (!f) {
    XELOGE("Failed to append portable GPU trace bundle {}",
           xe::path_to_utf8(bundle_path));
    std::error_code remove_error;
    std::filesystem::remove(bundle_path, remove_error);
    return false;
  }

  TraceBundleSection metadata_section = {};
  metadata_section.type = TraceBundleSectionType::kMetadataJson;
  metadata_section.offset = trace_size;
  metadata_section.length = metadata.size();
  metadata_section.xxh3_64 = XXH3_64bits(metadata.data(), metadata.size());
  TraceBundleFooter footer = {};
  footer.magic = kTraceBundleMagic;
  footer.version = kTraceBundleFormatVersion;
  footer.footer_size = sizeof(footer);
  footer.legacy_trace_size = trace_size;
  footer.legacy_trace_xxh3_64 = trace_hash;
  footer.section_table_offset = trace_size + metadata.size();
  footer.section_count = 1;

  bool written =
      fwrite(metadata.data(), 1, metadata.size(), f) == metadata.size() &&
      fwrite(&metadata_section, 1, sizeof(metadata_section), f) ==
          sizeof(metadata_section) &&
      fwrite(&footer, 1, sizeof(footer), f) == sizeof(footer);
  fclose(f);
  if (!written) {
    XELOGE("Failed to finalize portable GPU trace bundle {}",
           xe::path_to_utf8(bundle_path));
    std::error_code remove_error;
    std::filesystem::remove(bundle_path, remove_error);
    return false;
  }
  XELOGI("Portable GPU trace bundle created via {}: {}",
         cloned ? "APFS clone" : "file copy", xe::path_to_utf8(bundle_path));
  return true;
}

void CommandProcessor::RestoreRegisters(uint32_t first_register,
                                        const uint32_t* register_values,
                                        uint32_t register_count,
                                        bool execute_callbacks) {
  if (first_register > RegisterFile::kRegisterCount ||
      RegisterFile::kRegisterCount - first_register < register_count) {
    XELOGW(
        "CommandProcessor::RestoreRegisters out of bounds (0x{:X} registers "
        "starting with 0x{:X}, while a total of 0x{:X} registers are stored)",
        register_count, first_register, RegisterFile::kRegisterCount);
    if (first_register > RegisterFile::kRegisterCount) {
      return;
    }
    register_count =
        std::min(uint32_t(RegisterFile::kRegisterCount) - first_register,
                 register_count);
  }
  if (execute_callbacks) {
    for (uint32_t i = 0; i < register_count; ++i) {
      WriteRegister(first_register + i, register_values[i]);
    }
  } else {
    std::memcpy(register_file_->values + first_register, register_values,
                sizeof(uint32_t) * register_count);
  }
}

void CommandProcessor::RestoreGammaRamp(
    const reg::DC_LUT_30_COLOR* new_gamma_ramp_256_entry_table,
    const reg::DC_LUT_PWL_DATA* new_gamma_ramp_pwl_rgb,
    uint32_t new_gamma_ramp_rw_component) {
  std::memcpy(gamma_ramp_256_entry_table_, new_gamma_ramp_256_entry_table,
              sizeof(reg::DC_LUT_30_COLOR) * 256);
  std::memcpy(gamma_ramp_pwl_rgb_, new_gamma_ramp_pwl_rgb,
              sizeof(reg::DC_LUT_PWL_DATA) * 3 * 128);
  gamma_ramp_rw_component_ = new_gamma_ramp_rw_component;
  OnGammaRamp256EntryTableValueWritten();
  OnGammaRampPWLValueWritten();
}

void CommandProcessor::CallInThread(std::function<void()> fn) {
  if (pending_fns_.empty() &&
      kernel::XThread::IsInThread(worker_thread_.get())) {
    fn();
  } else {
    pending_fns_.push(std::move(fn));
  }
}

void CommandProcessor::LogFpsTick() {
  guest_swap_count_.fetch_add(1, std::memory_order_relaxed);
  if (!cvars::log_fps) {
    return;
  }
  ++fps_log_frame_count_;
  const uint64_t now = Clock::QueryHostTickCount();
  const uint64_t freq = Clock::QueryHostTickFrequency();
  if (!fps_log_start_ticks_) {
    fps_log_start_ticks_ = now;
    return;
  }
  const uint64_t delta = now - fps_log_start_ticks_;
  if (delta >= freq * 5) {
    const double seconds = double(delta) / double(freq);
    XELOGI("FPS: {:.1f} ({} frames / {:.1f} s)",
           double(fps_log_frame_count_) / seconds, fps_log_frame_count_,
           seconds);
    fps_log_frame_count_ = 0;
    fps_log_start_ticks_ = now;
  }
}

void CommandProcessor::ClearCaches() {}

void CommandProcessor::SetDesiredSwapPostEffect(
    SwapPostEffect swap_post_effect) {
  if (swap_post_effect_desired_ == swap_post_effect) {
    return;
  }
  swap_post_effect_desired_ = swap_post_effect;
  CallInThread([this, swap_post_effect]() {
    swap_post_effect_actual_ = swap_post_effect;
  });
}

void CommandProcessor::WorkerThreadMain() {
  if (!SetupContext()) {
    xe::FatalError("Unable to setup command processor internal state");
    return;
  }

  while (worker_running_) {
    while (!pending_fns_.empty()) {
      auto fn = std::move(pending_fns_.front());
      pending_fns_.pop();
      fn();
    }

    uint32_t write_ptr_index = write_ptr_index_.load();
    if (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index) {
      SCOPE_profile_cpu_i("gpu", "xe::gpu::CommandProcessor::Stall");
      // We've run out of commands to execute.
      // We spin here waiting for new ones, as the overhead of waiting on our
      // event is too high.
      PrepareForWait();
      const int64_t pacing_idle_start =
          FramePacingCensusEnabled() ? FramePacingNowNs() : 0;
      uint32_t loop_count = 0;
      do {
        // If we spin around too much, revert to a "low-power" state.
        if (loop_count > 500) {
          constexpr int wait_time_ms = 2;
          xe::threading::Wait(write_ptr_index_event_.get(), true,
                              std::chrono::milliseconds(wait_time_ms));
        } else {
          xe::threading::MaybeYield();
        }
        loop_count++;
        write_ptr_index = write_ptr_index_.load();
      } while (worker_running_ && pending_fns_.empty() &&
               (write_ptr_index == 0xBAADF00D ||
                read_ptr_index_ == write_ptr_index));
      if (pacing_idle_start) {
        GetFramePacingClock().ring_idle_ns.fetch_add(
            uint64_t(FramePacingNowNs() - pacing_idle_start),
            std::memory_order_relaxed);
      }
      ReturnFromWait();
      if (!worker_running_ || !pending_fns_.empty()) {
        continue;
      }
    }
    assert_true(read_ptr_index_ != write_ptr_index);

    // Execute. Note that we handle wraparound transparently.
    read_ptr_index_ = ExecutePrimaryBuffer(read_ptr_index_, write_ptr_index);

    // TODO(benvanik): use reader->Read_update_freq_ and only issue after moving
    //     that many indices.
    // Keep in mind that the gpu also updates the cpu-side copy if the write
    // pointer and read pointer would be equal
    if (read_ptr_writeback_ptr_) {
      xe::store_and_swap<uint32_t>(
          memory_->TranslatePhysical(read_ptr_writeback_ptr_), read_ptr_index_);
    }

    // FIXME: We're supposed to process the WAIT_UNTIL register at this point,
    // but no games seem to actually use it.
  }

  ShutdownContext();
}

void CommandProcessor::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;
  pause_guest_memory_ready_ = false;

  threading::Fence fence;
  CallInThread([this, &fence]() {
    // Transfer guest-RAM authority while the backend still owns its command
    // encoder on the command-processor thread. Save-state runs after this
    // thread is suspended; mutating an open Metal encoder from that caller
    // thread would violate the backend's thread ownership.
    pause_guest_memory_ready_ = PrepareGuestMemoryForHostRead(
        0, static_cast<uint32_t>(SharedMemory::kBufferSize));
    fence.Signal();
    threading::Thread::GetCurrentThread()->Suspend();
  });

  fence.Wait();
}

void CommandProcessor::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;
  pause_guest_memory_ready_ = false;

  worker_thread_->thread()->Resume();
}

bool CommandProcessor::Save(ByteStream* stream) {
  assert_true(paused_);

  // Pause performs the bulk RAM-consumer boundary on the command-processor
  // thread before suspending it. Save only verifies that transfer succeeded;
  // it must not touch a backend command encoder from this caller thread.
  if (!pause_guest_memory_ready_) {
    XELOGE("CommandProcessor::Save has no converged guest-RAM pause boundary");
    return false;
  }

  stream->Write<uint32_t>(primary_buffer_ptr_);
  stream->Write<uint32_t>(primary_buffer_size_);
  stream->Write<uint32_t>(read_ptr_index_);
  stream->Write<uint32_t>(read_ptr_update_freq_);
  stream->Write<uint32_t>(read_ptr_writeback_ptr_);
  stream->Write<uint32_t>(write_ptr_index_.load());

  return true;
}

bool CommandProcessor::Restore(ByteStream* stream) {
  assert_true(paused_);

  primary_buffer_ptr_ = stream->Read<uint32_t>();
  primary_buffer_size_ = stream->Read<uint32_t>();
  read_ptr_index_ = stream->Read<uint32_t>();
  read_ptr_update_freq_ = stream->Read<uint32_t>();
  read_ptr_writeback_ptr_ = stream->Read<uint32_t>();
  write_ptr_index_.store(stream->Read<uint32_t>());

  return true;
}

bool CommandProcessor::SetupContext() {
  ResetZPDState();
  return true;
}

void CommandProcessor::ShutdownContext() { ResetZPDState(); }

void CommandProcessor::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_ptr_index_ = 0;
  primary_buffer_ptr_ = ptr;
  primary_buffer_size_ = uint32_t(1) << (size_log2 + 3);

  std::memset(kernel_state_->memory()->TranslatePhysical(primary_buffer_ptr_),
              0, primary_buffer_size_);
}

void CommandProcessor::EnableReadPointerWriteBack(uint32_t ptr,
                                                  uint32_t block_size_log2) {
  // CP_RB_RPTR_ADDR Ring Buffer Read Pointer Address 0x70C
  // ptr = RB_RPTR_ADDR, pointer to write back the address to.
  read_ptr_writeback_ptr_ = ptr;
  // CP_RB_CNTL Ring Buffer Control 0x704
  // block_size = RB_BLKSZ, log2 of number of quadwords read between updates of
  //              the read pointer.
  read_ptr_update_freq_ = uint32_t(1) << block_size_log2 >> 2;
}

XE_NOINLINE XE_COLD void CommandProcessor::LogKickoffInitator(uint32_t value) {
  cpu::backend::GuestPseudoStackTrace st;

  if (logging::ShouldLog(LogLevel::Debug) &&
      kernel_state_->processor()->backend()->PopulatePseudoStacktrace(&st)) {
    logging::LoggerBatch<LogLevel::Debug> log_initiator{};

    log_initiator("Updating read ptr to {}, initiator stacktrace below\n",
                  value);

    for (uint32_t i = 0; i < st.count; ++i) {
      log_initiator("\t{:08X}\n", st.return_addrs[i]);
    }

    if (st.truncated_flag) {
      log_initiator("\t(Truncated stacktrace to {} entries)\n",
                    cpu::backend::MAX_GUEST_PSEUDO_STACKTRACE_ENTRIES);
    }
    log_initiator.submit('d');
  }
}

void CommandProcessor::UpdateWritePointer(uint32_t value) {
  XE_UNLIKELY_IF(cvars::log_ringbuffer_kickoff_initiator_bts) {
    LogKickoffInitator(value);
  }
  write_ptr_index_ = value;
  write_ptr_index_event_->SetBoostPriority();
}

void CommandProcessor::LogRegisterSet(uint32_t register_index, uint32_t value) {
#if XE_ENABLE_GPU_REG_WRITE_LOGGING == 1
  if (cvars::log_guest_driven_gpu_register_written_values &&
      logging::ShouldLog(LogLevel::Debug)) {
    const RegisterInfo* reginfo = RegisterFile::GetRegisterInfo(register_index);

    if (!reginfo) {
      XELOGD("Unknown_Reg{:04X} <- {:08X}\n", register_index, value);
    } else {
      XELOGD("{} <- {:08X}\n", reginfo->name, value);
    }
  }
#endif
}

void CommandProcessor::LogRegisterSets(uint32_t base_register_index,
                                       const uint32_t* values,
                                       uint32_t n_values) {
#if XE_ENABLE_GPU_REG_WRITE_LOGGING == 1
  if (cvars::log_guest_driven_gpu_register_written_values &&
      logging::ShouldLog(LogLevel::Debug)) {
    auto target = logging::internal::GetThreadBuffer();

    auto target_ptr = target.first;

    size_t total_size = 0;

    size_t rem_size = target.second;

    for (uint32_t i = 0; i < n_values; ++i) {
      uint32_t register_index = base_register_index + i;

      uint32_t value = xe::load_and_swap<uint32_t>(&values[i]);

      const RegisterInfo* reginfo =
          RegisterFile::GetRegisterInfo(register_index);

      if (!reginfo) {
        auto tmpres = fmt::format_to_n(target_ptr, rem_size,
                                       "Unknown_Reg{:04X} <- {:08X}\n",
                                       register_index, value);
        target_ptr = tmpres.out;
        rem_size -= tmpres.size;
        total_size += tmpres.size;

      } else {
        auto tmpres = fmt::format_to_n(target_ptr, rem_size, "{} <- {:08X}\n",
                                       reginfo->name, value);
        rem_size -= tmpres.size;
        target_ptr = tmpres.out;
        total_size += tmpres.size;
      }
    }
    logging::internal::AppendLogLine(LogLevel::Debug, 'd', total_size);
  }
#endif
}

void CommandProcessor::HandleSpecialRegisterWrite(uint32_t index,
                                                  uint32_t value) {
  RegisterFile& regs = *register_file_;
  // Scratch register writeback.
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    uint32_t scratch_reg = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1 << scratch_reg) & regs.values[XE_GPU_REG_SCRATCH_UMSK]) {
      // Enabled - write to address.
      uint32_t scratch_addr = regs.values[XE_GPU_REG_SCRATCH_ADDR];
      uint32_t mem_addr = scratch_addr + (scratch_reg * 4);
      xe::store_and_swap<uint32_t>(memory_->TranslatePhysical(mem_addr), value);
    }
  } else {
    switch (index) {
      // If this is a COHER register, set the dirty flag.
      // This will block the command processor the next time it WAIT_MEM_REGs
      // and allow us to synchronize the memory.
      case XE_GPU_REG_COHER_STATUS_HOST: {
        regs.values[index] |= UINT32_C(0x80000000);
      } break;

      case XE_GPU_REG_DC_LUT_RW_INDEX: {
        // Reset the sequential read / write component index (see the M56
        // DC_LUT_SEQ_COLOR documentation).
        gamma_ramp_rw_component_ = 0;
      } break;

      case XE_GPU_REG_DC_LUT_SEQ_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // DC_LUT_SEQ_COLOR is in the red, green, blue order, but the write
        // enable mask is blue, green, red.
        bool write_gamma_ramp_component =
            (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
             (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          // Bits 0:5 are hardwired to zero.
          uint32_t gamma_ramp_seq_color =
              regs.Get<reg::DC_LUT_SEQ_COLOR>().seq_color >> 6;
          switch (gamma_ramp_rw_component_) {
            case 0:
              gamma_ramp_entry.color_10_red = gamma_ramp_seq_color;
              break;
            case 1:
              gamma_ramp_entry.color_10_green = gamma_ramp_seq_color;
              break;
            case 2:
              gamma_ramp_entry.color_10_blue = gamma_ramp_seq_color;
              break;
          }
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          ++new_gamma_ramp_rw_index.rw_index;
          WriteRegister(
              XE_GPU_REG_DC_LUT_RW_INDEX,
              xe::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_PWL_DATA: {
        // Should be in the PWL writing mode.
        assert_not_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // Bit 7 of the index is ignored for PWL.
        uint32_t gamma_ramp_rw_index_pwl = gamma_ramp_rw_index.rw_index & 0x7F;
        // DC_LUT_PWL_DATA is likely in the red, green, blue order because
        // DC_LUT_SEQ_COLOR is, but the write enable mask is blue, green, red.
        bool write_gamma_ramp_component =
            (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
             (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_PWL_DATA& gamma_ramp_entry =
              gamma_ramp_pwl_rgb_[gamma_ramp_rw_index_pwl]
                                 [gamma_ramp_rw_component_];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_PWL_DATA>();
          // Bits 0:5 are hardwired to zero.
          gamma_ramp_entry.base = gamma_ramp_value.base & ~UINT32_C(0x3F);
          gamma_ramp_entry.delta = gamma_ramp_value.delta & ~UINT32_C(0x3F);
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          // TODO(Triang3l): Should this increase beyond 7 bits for PWL?
          // Direct3D 9 explicitly sets rw_index to 0x80 after writing the last
          // PWL entry. However, the DC_LUT_RW_INDEX documentation says that for
          // PWL, the bit 7 is ignored.
          new_gamma_ramp_rw_index.rw_index =
              (gamma_ramp_rw_index.rw_index & ~UINT32_C(0x7F)) |
              ((gamma_ramp_rw_index_pwl + 1) & 0x7F);
          WriteRegister(
              XE_GPU_REG_DC_LUT_RW_INDEX,
              xe::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRampPWLValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_30_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        uint32_t gamma_ramp_write_enable_mask =
            regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] & 0b111;
        if (gamma_ramp_write_enable_mask) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_30_COLOR>();
          if (gamma_ramp_write_enable_mask & 0b001) {
            gamma_ramp_entry.color_10_blue = gamma_ramp_value.color_10_blue;
          }
          if (gamma_ramp_write_enable_mask & 0b010) {
            gamma_ramp_entry.color_10_green = gamma_ramp_value.color_10_green;
          }
          if (gamma_ramp_write_enable_mask & 0b100) {
            gamma_ramp_entry.color_10_red = gamma_ramp_value.color_10_red;
          }
        }
        // TODO(Triang3l): Should this reset the component write index? If this
        // increase is assumed to behave like a full DC_LUT_RW_INDEX write, it
        // probably should. Currently this also calls WriteRegister for
        // DC_LUT_RW_INDEX, which resets gamma_ramp_rw_component_ as well.
        gamma_ramp_rw_component_ = 0;
        reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
        ++new_gamma_ramp_rw_index.rw_index;
        WriteRegister(
            XE_GPU_REG_DC_LUT_RW_INDEX,
            xe::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        if (gamma_ramp_write_enable_mask) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;
    }
  }
}
void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  // chrispy: rearrange check order, place set after checks

  if (XE_LIKELY(index < RegisterFile::kRegisterCount)) {
    register_file_->values[index] = value;
    // Backend-agnostic diagnostic (XE_GPU_FORCE_BOOL=<index>:<0|1>): pin one
    // guest bool constant at register-write time so a shader block gated by
    // it can be A/B'd identically on every backend (dual-backend attribution
    // of cexec/jump-gated code, e.g. the Halo 3 light PCF block b141).
    struct ForceBool {
      bool set = false;
      uint32_t reg = 0;
      uint32_t mask = 0;
      bool one = false;
    };
    static const ForceBool s_force_bool = [] {
      ForceBool f;
      const char* v = std::getenv("XE_GPU_FORCE_BOOL");
      if (v) {
        int idx = -1, val = 0;
        if (std::sscanf(v, "%d:%d", &idx, &val) == 2 && idx >= 0 &&
            idx < 256) {
          f.set = true;
          f.reg = XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 + (uint32_t(idx) >> 5);
          f.mask = 1u << (uint32_t(idx) & 31);
          f.one = val != 0;
        }
      }
      return f;
    }();
    if (XE_UNLIKELY(s_force_bool.set && index == s_force_bool.reg)) {
      uint32_t& word = register_file_->values[index];
      word = s_force_bool.one ? (word | s_force_bool.mask)
                              : (word & ~s_force_bool.mask);
    }

    // quick pre-test
    // todo: figure out just how unlikely this is. if very (it ought to be,
    // theres a ton of registers other than these) make this predicate
    // branchless and mark with unlikely, then make HandleSpecialRegisterWrite
    // noinline yep, its very unlikely. these ORS here are meant to be bitwise
    // ors, so that we do not do branching evaluation of the conditions (we will
    // almost always take all of the branches)

    unsigned expr = (index - XE_GPU_REG_SCRATCH_REG0 < 8) |
                    (index == XE_GPU_REG_COHER_STATUS_HOST) |
                    ((index - XE_GPU_REG_DC_LUT_RW_INDEX) <=
                     (XE_GPU_REG_DC_LUT_30_COLOR - XE_GPU_REG_DC_LUT_RW_INDEX));
    // chrispy: reordered for msvc branch probability (assumes if is taken and
    // else is not)
    if (XE_LIKELY(expr == 0)) {
      XE_MSVC_REORDER_BARRIER();

    } else {
      HandleSpecialRegisterWrite(index, value);
    }
  } else {
    XELOGW("CommandProcessor::WriteRegister index out of bounds: {}", index);
    return;
  }
}
void CommandProcessor::WriteRegistersFromMem(uint32_t start_index,
                                             uint32_t* base,
                                             uint32_t num_registers) {
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = xe::load_and_swap<uint32_t>(base + i);
    this->WriteRegister(start_index + i, data);
  }
}

void CommandProcessor::WriteRegisterRangeFromRing(xe::RingBuffer* ring,
                                                  uint32_t base,
                                                  uint32_t num_registers) {
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = ring->ReadAndSwap<uint32_t>();
    WriteRegister(base + i, data);
  }
}

void CommandProcessor::WriteALURangeFromRing(xe::RingBuffer* ring,
                                             uint32_t base,
                                             uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4000, num_times);
}

void CommandProcessor::WriteFetchRangeFromRing(xe::RingBuffer* ring,
                                               uint32_t base,
                                               uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4800, num_times);
}

void CommandProcessor::WriteBoolRangeFromRing(xe::RingBuffer* ring,
                                              uint32_t base,
                                              uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4900, num_times);
}

void CommandProcessor::WriteLoopRangeFromRing(xe::RingBuffer* ring,
                                              uint32_t base,
                                              uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x4908, num_times);
}

void CommandProcessor::WriteREGISTERSRangeFromRing(xe::RingBuffer* ring,
                                                   uint32_t base,
                                                   uint32_t num_times) {
  WriteRegisterRangeFromRing(ring, base + 0x2000, num_times);
}

void CommandProcessor::WriteALURangeFromMem(uint32_t start_index,
                                            uint32_t* base,
                                            uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4000, base, num_registers);
}

void CommandProcessor::WriteFetchRangeFromMem(uint32_t start_index,
                                              uint32_t* base,
                                              uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4800, base, num_registers);
}

void CommandProcessor::WriteBoolRangeFromMem(uint32_t start_index,
                                             uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4900, base, num_registers);
}

void CommandProcessor::WriteLoopRangeFromMem(uint32_t start_index,
                                             uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4908, base, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromMem(uint32_t start_index,
                                                  uint32_t* base,
                                                  uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x2000, base, num_registers);
}
XE_NOINLINE
void CommandProcessor::WriteOneRegisterFromRing(uint32_t base,
                                                uint32_t num_times) {
  for (uint32_t m = 0; m < num_times; m++) {
    uint32_t reg_data = reader_.ReadAndSwap<uint32_t>();
    uint32_t target_index = base;
    WriteRegister(target_index, reg_data);
  }
}
void CommandProcessor::MakeCoherent() {
  SCOPE_profile_cpu_f("gpu");

  // Status host often has 0x01000000 or 0x03000000.
  // This is likely toggling VC (vertex cache) or TC (texture cache).
  // Or, it also has a direction in here maybe - there is probably
  // some way to check for dest coherency (what all the COHER_DEST_BASE_*
  // registers are for).
  // Best docs I've found on this are here:
  // https://web.archive.org/web/20160711162346/https://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/10/R6xx_R7xx_3D.pdf
  // https://cgit.freedesktop.org/xorg/driver/xf86-video-radeonhd/tree/src/r6xx_accel.c?id=3f8b6eccd9dba116cc4801e7f80ce21a879c67d2#n454

  volatile uint32_t* regs_volatile = register_file_->values;
  auto status_host = xe::memory::Reinterpret<reg::COHER_STATUS_HOST>(
      uint32_t(regs_volatile[XE_GPU_REG_COHER_STATUS_HOST]));
  uint32_t base_host = regs_volatile[XE_GPU_REG_COHER_BASE_HOST];
  uint32_t size_host = regs_volatile[XE_GPU_REG_COHER_SIZE_HOST];

  if (!status_host.status) {
    return;
  }

  const char* action = "N/A";
  if (status_host.vc_action_ena && status_host.tc_action_ena) {
    action = "VC | TC";
  } else if (status_host.tc_action_ena) {
    action = "TC";
  } else if (status_host.vc_action_ena) {
    action = "VC";
  }

  // TODO(benvanik): notify resource cache of base->size and type.
  XELOGGPU("Make {:08X} -> {:08X} ({}b) coherent, action = {}", base_host,
           base_host + size_host, size_host, action);

  // Mark coherent.
  regs_volatile[XE_GPU_REG_COHER_STATUS_HOST] = 0;
}

void CommandProcessor::PrepareForWait() {
  trace_writer_.Flush();
  // Only refresh completion if there is a strict ZPD retire pending so
  // PumpPendingRetire sees the latest progress without adding extra overhead.
  if (zpd_pending_retire_handle_ != kInvalidReportHandle) {
    PollCompletedSubmission();
  }
  // Give strict ZPD a chance to retire a pending report before the guest's
  // loop polls again.
  PumpPendingRetire();
}

void CommandProcessor::ReturnFromWait() {}

void CommandProcessor::InitializeTrace() {
  // Write the initial register values, to be loaded directly into the
  // RegisterFile since all registers, including those that may have side
  // effects on setting, will be saved.
  trace_writer_.WriteRegisters(
      0, reinterpret_cast<const uint32_t*>(register_file_->values),
      RegisterFile::kRegisterCount, false);

  trace_writer_.WriteGammaRamp(gamma_ramp_256_entry_table(),
                               gamma_ramp_pwl_rgb(), gamma_ramp_rw_component_);
}

CommandProcessor::PendingZPDSlot CommandProcessor::GetPendingZPDSlot(
    uint32_t slot_base, uint32_t end_record) const {
  PendingZPDSlot pending_slot;

  for (const auto& report_pair : logical_zpd_reports_) {
    const ZPDReport& report = report_pair.second;
    if (!report.ended || report.pending_segments == 0 ||
        report.slot_base != slot_base) {
      continue;
    }

    // Wait on the oldest unresolved report for this slot first.
    if (pending_slot.report_handle == kInvalidReportHandle ||
        report_pair.first < pending_slot.report_handle) {
      pending_slot.report_handle = report_pair.first;
    }

    // Slot reuse needs to be handled carefully in fast mode. Keep the biggest
    // cached delta, not the newest one. A stale zero is a lot more dangerous
    // than a stale nonzero.
    if (report.has_cached_delta) {
      if (!pending_slot.has_cached_delta ||
          report.cached_delta > pending_slot.cached_delta) {
        pending_slot.cached_delta = report.cached_delta;
      }
      pending_slot.has_cached_delta = true;
    }

    if (report.end_record) {
      auto report_cache_it =
          fast_zpd_report_cached_values_.find(report.end_record);
      if (report_cache_it != fast_zpd_report_cached_values_.end()) {
        if (!pending_slot.has_cached_delta ||
            report_cache_it->second > pending_slot.cached_delta) {
          pending_slot.cached_delta = report_cache_it->second;
        }
        pending_slot.has_cached_delta = true;
      }
    }
  }

  auto end_record_cache_it = fast_zpd_report_cached_values_.find(end_record);
  if (end_record_cache_it != fast_zpd_report_cached_values_.end()) {
    if (!pending_slot.has_cached_delta ||
        end_record_cache_it->second > pending_slot.cached_delta) {
      pending_slot.cached_delta = end_record_cache_it->second;
    }
    pending_slot.has_cached_delta = true;
  }

  return pending_slot;
}

bool CommandProcessor::BeginZPDReport(uint32_t report_address) {
  if (GetZPDMode() == ZPDMode::kFake) {
    return false;
  }

  // Track any delta to carry forward if the same slot is immediately reused.
  uint32_t carried_cached_delta = 0;
  bool has_carried_cached_delta = false;
  uint32_t carried_from_slot_base = 0;

  if (zpd_active_segment_.logical_active) {
    // New BEGIN while a prior report is open. Hardware has one register for
    // the query address, so a new BEGIN implicitly ends the prior one.
    if (zpd_active_segment_.end_record) {
      EndZPDReport(zpd_active_segment_.end_record, true);
    } else {
      carried_from_slot_base = zpd_active_segment_.slot_base;

      auto dying_report =
          logical_zpd_reports_.find(zpd_active_segment_.report_handle);
      if (dying_report != logical_zpd_reports_.end() &&
          dying_report->second.has_cached_delta) {
        carried_cached_delta = dying_report->second.cached_delta;
        has_carried_cached_delta = true;
      }

      if (zpd_active_segment_.segment_active) {
        // Deactivate the segment before DiscardZPDQuery so that
        // EndSubmission -> CloseQuerySegment does not re-enter and
        // issue a second EndQuery on the same slot.
        zpd_active_segment_.segment_active = false;
        DiscardZPDQuery();
      }
      logical_zpd_reports_.erase(zpd_active_segment_.report_handle);
      zpd_active_segment_ = {};
    }
  }

  uint32_t slot_base = XenosZPDReport::GetSlotBase(report_address);
  uint32_t begin_record = XenosZPDReport::GetBeginRecordBase(slot_base);
  uint32_t end_record = XenosZPDReport::GetEndRecordBase(slot_base);
  if (!slot_base) {
    return false;
  }

  // Resolve same slot hazards before invalidating pending writes from the prior
  // lifetime. For finished strict queries with unpolled completion, refresh now
  // and drain, avoiding unnecessary AwaitQueryResolve blocking.
  if (GetZPDMode() == ZPDMode::kStrict) {
    PollCompletedSubmission();
  } else {
    PumpQueryResolves();
  }

  PendingZPDSlot pending_slot = GetPendingZPDSlot(slot_base, end_record);

  if (pending_slot.report_handle != kInvalidReportHandle) {
    if (GetZPDMode() == ZPDMode::kFast || GetZPDMode() == ZPDMode::kFastAlt) {
      if (pending_slot.has_cached_delta) {
        carried_cached_delta = pending_slot.cached_delta;
        has_carried_cached_delta = true;
        carried_from_slot_base = slot_base;
      }
    } else {
      while (pending_slot.report_handle != kInvalidReportHandle) {
        auto report_it = logical_zpd_reports_.find(pending_slot.report_handle);
        if (report_it == logical_zpd_reports_.end()) {
          break;
        }

        uint64_t wait_for_submission =
            report_it->second.last_segment_end_submission;

        bool wait_succeeded =
            AwaitQueryResolve(pending_slot.report_handle, wait_for_submission);

        if (!wait_succeeded) {
          if (pending_slot.cached_delta != 0) {
            carried_cached_delta = pending_slot.cached_delta;
            carried_from_slot_base = slot_base;
          }
          break;
        }

        PumpQueryResolves();

        pending_slot = GetPendingZPDSlot(slot_base, end_record);
      }
    }
  }

  // Bump slot sequence - invalidates pending writes from prior lifetime.
  uint64_t slot_sequence_id = ++zpd_slot_sequences_[slot_base];

  // By default, BEGIN drops the cached value so an orphaned END doesn't replay
  // something from a prior lifetime. The alternate fast path keeps it around
  // long enough for an async zero to help the next unresolved write.
  if (GetZPDMode() != ZPDMode::kFastAlt) {
    auto cache_it = fast_zpd_report_cached_values_.find(end_record);
    if (cache_it != fast_zpd_report_cached_values_.end() &&
        cache_it->second == 0) {
      fast_zpd_report_cached_values_.erase(cache_it);
    }
  }

  ReportHandle report_handle = zpd_next_report_handle_++;
  if (report_handle == kInvalidReportHandle) {
    report_handle = zpd_next_report_handle_++;
  }

  ZPDReport& logical = logical_zpd_reports_[report_handle];
  logical.slot_base = slot_base;
  logical.slot_sequence_id = slot_sequence_id;
  logical.begin_record = begin_record;
  logical.end_record = end_record;
  logical.begin_value = zpd_slot_values_[slot_base];
  logical.accumulated_samples = 0;
  logical.first_segment_end_submission = 0;
  logical.last_segment_end_submission = 0;
  logical.pending_segments = 0;
  logical.cached_delta = 0;
  logical.has_cached_delta = false;
  logical.ended = false;

  if (slot_base == carried_from_slot_base && has_carried_cached_delta) {
    logical.cached_delta = carried_cached_delta;
    logical.has_cached_delta = true;
  }

  zpd_active_segment_.report_handle = report_handle;
  zpd_active_segment_.slot_base = slot_base;
  zpd_active_segment_.begin_record = begin_record;
  zpd_active_segment_.end_record = end_record;
  zpd_active_segment_.segment_active = false;
  // Opens lazily. OpenQuerySegment will open it at the next valid opportunity.
  zpd_active_segment_.segment_pending_begin = true;
  zpd_active_segment_.logical_active = true;

  OpenQuerySegment(true);
  return true;
}

// Guest END closes the logical lifetime, but the final value may still depend
// on in flight query segments.
bool CommandProcessor::EndZPDReport(uint32_t report_address,
                                    bool guest_forced_end) {
  if (GetZPDMode() == ZPDMode::kFake) {
    return false;
  }

  CommandProcessor::ReportHandle report_handle =
      zpd_active_segment_.report_handle;
  uint32_t stored_end_record = zpd_active_segment_.end_record;
  uint32_t report_record_base = XenosZPDReport::GetRecordBase(report_address);
  if (!report_record_base) {
    report_record_base = stored_end_record;
  }

  if (zpd_active_segment_.segment_active) {
    CloseQuerySegment();
  }

  zpd_active_segment_.segment_pending_begin = false;

  if (!report_record_base) {
    logical_zpd_reports_.erase(report_handle);
    zpd_active_segment_ = {};
    return false;
  }

  bool resolved_immediately = false;
  uint32_t begin_record = 0;
  uint32_t begin_value = 0;
  uint32_t final_value = 0;
  uint32_t cached_delta = 0;
  bool has_cached_delta = false;

  auto it = logical_zpd_reports_.find(report_handle);
  if (it == logical_zpd_reports_.end()) {
    zpd_active_segment_ = {};
    return false;
  }

  ZPDReport& logical = it->second;
  logical.ended = true;
  logical.end_record = report_record_base;
  begin_record = logical.begin_record;
  begin_value = logical.begin_value;

  if (logical.pending_segments == 0) {
    resolved_immediately = true;
    final_value = NormalizeSampleCount(logical.accumulated_samples);

    cached_delta = final_value;
    has_cached_delta = true;
    logical.cached_delta = cached_delta;
    logical.has_cached_delta = true;
    if (fast_zpd_report_cached_values_.size() >= kFastZPDCacheMaxEntries &&
        !fast_zpd_report_cached_values_.count(report_record_base)) {
      fast_zpd_report_cached_values_.clear();
    }
    fast_zpd_report_cached_values_[report_record_base] = cached_delta;
    final_value = cached_delta;
  } else {
    if (logical.has_cached_delta) {
      cached_delta = logical.cached_delta;
      has_cached_delta = true;
    }
    auto cache_it = fast_zpd_report_cached_values_.find(report_record_base);
    if (cache_it != fast_zpd_report_cached_values_.end()) {
      cached_delta = cache_it->second;
      has_cached_delta = true;
    }
  }

  if (resolved_immediately) {
    CommitZPDReport(logical, final_value);
    logical_zpd_reports_.erase(it);
  }

  bool has_cross_slot_end =
      stored_end_record && stored_end_record != report_record_base;
  if (has_cross_slot_end) {
    // The guest ended a different ZPD record than the one opened by BEGIN.
    // Preserve the prior running slot value in the stored END record before
    // writing the actual END selected by the packet.
    WriteZPDReport(0, stored_end_record, 0, begin_value, false);
  }

  if (GetZPDMode() == ZPDMode::kFast || GetZPDMode() == ZPDMode::kFastAlt) {
    bool write_begin = begin_record && report_record_base &&
                       begin_record != report_record_base;
    // Unknown still means visible in fast mode. Reusing cached zeroes can help
    // flares stop shining through walls (545107FC, 454108D4, 4D5307D2), but it
    // also tends to break occlusion culling (4D5308AB, 4D530805), so only do it
    // in the alternate fast path.
    uint32_t speculative = cached_delta;
    if (!resolved_immediately) {
      speculative = 1;
      if (has_cached_delta &&
          (cached_delta != 0 || GetZPDMode() == ZPDMode::kFastAlt)) {
        speculative = cached_delta;
      }
    }
    WriteZPDReport(begin_record, report_record_base, begin_value, speculative,
                   write_begin);
  } else if (!resolved_immediately) {
    PumpQueryResolves();

    // Recheck after the drain. The report may have resolved synchronously if
    // all segments were already complete by the time we got here.
    if (!logical_zpd_reports_.count(report_handle)) {
      // OnZPDQueryResolved already committed and erased the report; nothing
      // left to defer.
    } else if (zpd_pending_retire_handle_ != report_handle) {
      zpd_pending_retire_handle_ = report_handle;
      zpd_pending_retire_stalls_ = 0;
      zpd_pending_retire_start_ms_ = Clock::QueryHostUptimeMillis();
    }
  }

  zpd_active_segment_ = {};
  return true;
}

void CommandProcessor::OpenQuerySegment(bool can_close_submission) {
  if (GetZPDMode() == ZPDMode::kFake || zpd_force_fake_fallback_ ||
      !zpd_active_segment_.logical_active ||
      !zpd_active_segment_.segment_pending_begin || !CanOpenZPDQuery()) {
    return;
  }

  EnsureZPDQueryResources();

  // Resource setup failed. Drop the logical report and fall back to fake mode.
  if (!IsZPDQueryPoolReady()) {
    zpd_force_fake_fallback_ = true;
    logical_zpd_reports_.erase(zpd_active_segment_.report_handle);
    zpd_active_segment_ = {};
    return;
  }

  // Frees any slots from completed submissions before asking for new ones.
  PumpQueryResolves();

  QueryOpenResult open_result =
      OpenZPDQuery(zpd_active_segment_.report_handle, can_close_submission);
  switch (open_result) {
    case QueryOpenResult::kOpened:
      break;
    case QueryOpenResult::kDeferred:
      return;
    case QueryOpenResult::kPoolExhausted: {
      if (GetZPDMode() == ZPDMode::kFast || GetZPDMode() == ZPDMode::kFastAlt) {
        // Fast mode favors forward progress over accuracy. Keep a minimal
        // accumulated value instead of waiting for a slot to become available.
        auto it = logical_zpd_reports_.find(zpd_active_segment_.report_handle);
        if (it != logical_zpd_reports_.end()) {
          it->second.accumulated_samples =
              std::max<uint64_t>(it->second.accumulated_samples, uint64_t{1});
        }
        zpd_active_segment_.segment_pending_begin = false;
        return;
      }
      return;
    }
    case QueryOpenResult::kFailed:
    default:
      return;
  }

  zpd_active_segment_.segment_active = true;
  zpd_active_segment_.segment_pending_begin = false;
}

// Closes the active host segment without ending the logical report.
// BeginQuery/EndQuery can't cross D3D12 command list or Vulkan render pass
// boundaries. The result accumulates across all pieces.
void CommandProcessor::CloseQuerySegment() {
  if (GetZPDMode() == ZPDMode::kFake || !zpd_active_segment_.segment_active) {
    return;
  }

  uint64_t submission = 0;
  if (!CloseZPDQuery(zpd_active_segment_.report_handle, submission)) {
    zpd_active_segment_.segment_active = false;
    zpd_active_segment_.segment_pending_begin =
        zpd_active_segment_.logical_active;
    return;
  }

  auto it = logical_zpd_reports_.find(zpd_active_segment_.report_handle);
  if (it != logical_zpd_reports_.end()) {
    // Lets PumpPendingRetire drain early segments without blocking on the
    // final segment's submission.
    if (it->second.pending_segments == 0) {
      it->second.first_segment_end_submission = submission;
    }
    it->second.pending_segments++;
    it->second.last_segment_end_submission = submission;
  }

  zpd_active_segment_.segment_active = false;

  zpd_active_segment_.segment_pending_begin =
      zpd_active_segment_.logical_active;
}

void CommandProcessor::OnZPDQueryResolved(ReportHandle report_handle,
                                          uint64_t raw_samples) {
  {
    static const bool s_zpd_trace = std::getenv("XE_GPU_ZPD_TRACE") != nullptr;
    if (s_zpd_trace) {
      XELOGW("ZPDResolved handle={} samples={}", report_handle, raw_samples);
    }
  }
  auto it = logical_zpd_reports_.find(report_handle);
  if (it == logical_zpd_reports_.end()) {
    return;
  }

  ZPDReport& logical = it->second;

  if (logical.pending_segments) {
    logical.pending_segments--;
  }

  logical.accumulated_samples += raw_samples;

  if (logical.ended && logical.pending_segments == 0) {
    uint32_t final_value = NormalizeSampleCount(logical.accumulated_samples);

    logical.cached_delta = final_value;
    logical.has_cached_delta = true;
    if (logical.end_record) {
      if (fast_zpd_report_cached_values_.size() >= kFastZPDCacheMaxEntries &&
          !fast_zpd_report_cached_values_.count(logical.end_record)) {
        fast_zpd_report_cached_values_.clear();
      }
      fast_zpd_report_cached_values_[logical.end_record] = final_value;
    }
    if (IsZPDReportCurrent(logical)) {
      CommitZPDReport(logical, final_value);
    }
    logical_zpd_reports_.erase(it);
  }
}

void CommandProcessor::PumpPendingRetire() {
  ReportHandle handle_to_await = zpd_pending_retire_handle_;
  if (handle_to_await == kInvalidReportHandle) {
    return;
  }

  auto logical_report = logical_zpd_reports_.find(handle_to_await);
  if (logical_report == logical_zpd_reports_.end()) {
    // If the report is already gone it retired through another path.
    // Clear so we don't spin on a handle that no longer exists.
    zpd_pending_retire_handle_ = kInvalidReportHandle;
    zpd_pending_retire_stalls_ = 0;
    return;
  }

  uint64_t wait_for_submission =
      logical_report->second.last_segment_end_submission;
  uint64_t first_submission =
      logical_report->second.first_segment_end_submission;

  // Early segments can be retired here and, in the best case, the report
  // fully resolves without any wait.
  if (first_submission != 0 && first_submission < wait_for_submission &&
      first_submission <= GetCompletedSubmission()) {
    PumpQueryResolves();
    logical_report = logical_zpd_reports_.find(handle_to_await);
    if (logical_report == logical_zpd_reports_.end()) {
      zpd_pending_retire_handle_ = kInvalidReportHandle;
      zpd_pending_retire_stalls_ = 0;
      return;
    }
    wait_for_submission = logical_report->second.last_segment_end_submission;
  }

  if (AwaitQueryResolve(handle_to_await, wait_for_submission)) {
    zpd_pending_retire_handle_ = kInvalidReportHandle;
    zpd_pending_retire_stalls_ = 0;
    return;
  }

  if (wait_for_submission == 0 ||
      GetCompletedSubmission() >= wait_for_submission) {
    ++zpd_pending_retire_stalls_;
  }

  // Abandon if the deadline has elapsed or the stall limit has been reached.
  // Both are checked to account for varied guest polling behavior.
  bool deadline_exceeded =
      (Clock::QueryHostUptimeMillis() - zpd_pending_retire_start_ms_ >=
       kStrictZPDRetireDeadlineMs);
  if (deadline_exceeded ||
      zpd_pending_retire_stalls_ >= kStrictZPDRetireMaxStalls) {
    // Write the cached delta to guest memory to avoid a sudden occlusion flash.
    if (IsZPDReportCurrent(logical_report->second)) {
      uint32_t fallback_delta = logical_report->second.cached_delta
                                    ? logical_report->second.cached_delta
                                    : 1;
      CommitZPDReport(logical_report->second, fallback_delta);
    }
    logical_zpd_reports_.erase(logical_report);
    zpd_pending_retire_handle_ = kInvalidReportHandle;
    zpd_pending_retire_stalls_ = 0;
  }
}

void CommandProcessor::WriteZPDReport(uint32_t begin_record,
                                      uint32_t end_record, uint32_t begin_value,
                                      uint32_t delta_value,
                                      bool write_begin_record) {
  xenos::xe_gpu_depth_sample_counts* begin =
      begin_record
          ? memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(
                begin_record)
          : nullptr;
  if (!end_record) {
    return;
  }
  xenos::xe_gpu_depth_sample_counts* end =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(
          end_record);

  // Pre-write journal notify (fix-replay-guest-ram-rollback 1.2) — before the
  // record bytes are overwritten.
  if (begin && write_begin_record) {
    NotifyGuestRamWrite(begin_record,
                        sizeof(xenos::xe_gpu_depth_sample_counts));
  }
  NotifyGuestRamWrite(end_record, sizeof(xenos::xe_gpu_depth_sample_counts));

  XenosZPDReport::WriteReportDelta(begin, end, begin_value, delta_value,
                                   write_begin_record);
  // Guest-RAM write census (fix-replay-guest-ram-rollback 1.1): ZPD report
  // records are GPU-derived sample counts written into guest RAM. This is the
  // funnel for both the resolved-commit and the orphaned-END cached-delta
  // paths. Diagnostic-only, default off.
  static const bool s_ram_write_census =
      std::getenv("XE_GPU_RAM_WRITE_CENSUS") != nullptr;
  if (s_ram_write_census) {
    XELOGI("RamWriteCensus writer=zpd_report begin={:08X} end={:08X} len={}",
           write_begin_record ? begin_record : 0u, end_record,
           uint32_t(sizeof(xenos::xe_gpu_depth_sample_counts)));
  }
}

void CommandProcessor::CommitZPDReport(ZPDReport& report,
                                       uint32_t delta_value) {
  uint32_t end_record =
      report.end_record ? report.end_record
                        : XenosZPDReport::GetEndRecordBase(report.slot_base);
  WriteZPDReport(report.begin_record, end_record, report.begin_value,
                 delta_value, report.begin_record != 0);

  // Advance running total so the next BeginReport on this slot picks up
  // the correct begin_value.
  uint32_t saturated_delta = XenosZPDReport::SaturateSampleCount(delta_value);
  uint32_t end_value = report.begin_value + saturated_delta;
  zpd_slot_values_[report.slot_base] = end_value;
}

void CommandProcessor::WriteCaptureMeta(
    const std::filesystem::path& capture_dir) {
  // Provenance that is EXPECTED to differ between runs (build, device,
  // driver) lives here, NOT in the byte-stable per-draw manifest (design D4),
  // so strict manifest comparison stays meaningful.
  std::filesystem::path meta_path = capture_dir / "capture_meta.json";
  FILE* f = xe::filesystem::OpenFile(meta_path, "wb");
  if (!f) {
    XELOGE("capture_dir: failed to open {}", meta_path.string());
    return;
  }
#if defined(NDEBUG)
  const char* build_config = "Release";
#else
  const char* build_config = "Debug";
#endif
#if defined(XE_BUILD_COMMIT)
  const char* build_commit = XE_BUILD_COMMIT;
  const char* build_branch = XE_BUILD_BRANCH;
#else
  const char* build_commit = "unknown";
  const char* build_branch = "unknown";
#endif
  std::string device_json = GetCaptureDeviceInfoJson();
  if (device_json.empty()) {
    device_json = "null";
  }
  std::string backend_name =
      graphics_system_ ? graphics_system_->name() : "unknown";
  std::string renderer_token = RendererFileToken(backend_name);
  std::string renderer_api = RendererApiName(renderer_token);
  std::string translation_layer = "null";
  std::string host_api = renderer_api;
#if XE_PLATFORM_MAC
  if (renderer_token == "vulkan-moltenvk") {
    translation_layer = "\"MoltenVK\"";
    host_api = "Metal";
  }
#endif
  uint32_t width = 0;
  uint32_t height = 0;
  if (graphics_system_) {
    std::tie(width, height) = graphics_system_->GetResolution();
  }
  std::string meta = fmt::format(
      "{{\"schema\":2,\"build_commit\":\"{}\",\"build_branch\":\"{}\","
      "\"build_config\":\"{}\","
      "\"host\":{{\"platform\":\"{}\",\"os\":\"{}\","
      "\"architecture\":\"{}\"}},"
      "\"guest\":{{\"display_width\":{},\"display_height\":{}}},"
      "\"renderer\":{{\"backend\":\"{}\",\"api\":\"{}\","
      "\"translation_layer\":{},\"host_api\":\"{}\"}},"
      "\"device\":{}}}\n",
      JsonEscape(build_commit), JsonEscape(build_branch), build_config,
      HostPlatformClass(), HostOperatingSystemName(), HostArchitectureName(),
      width, height, JsonEscape(backend_name), JsonEscape(renderer_api),
      translation_layer, JsonEscape(host_api), device_json);
  fwrite(meta.data(), 1, meta.size(), f);
  fclose(f);
}

bool CommandProcessor::IsCaptureEdramDraw(uint32_t draw_index) const {
  if (cvars::capture_dir.empty()) {
    return false;
  }
  int32_t idx = int32_t(draw_index);
  // Inclusive range "A..B".
  if (!cvars::capture_draw_range.empty()) {
    int32_t a = 0, b = 0;
    if (std::sscanf(cvars::capture_draw_range.c_str(), "%d..%d", &a, &b) == 2 &&
        idx >= a && idx <= b) {
      return true;
    }
  }
  // Explicit comma-separated list "i,j,k".
  if (!cvars::capture_edram_at.empty()) {
    std::stringstream list(cvars::capture_edram_at);
    std::string item;
    while (std::getline(list, item, ',')) {
      if (!item.empty() && std::atoi(item.c_str()) == idx) {
        return true;
      }
    }
  }
  return false;
}

void CommandProcessor::MaybeSerializeDraw(uint32_t draw_index) {
  if (cvars::serialize_draw_range.empty()) {
    return;
  }
  int32_t a = 0, b = 0;
  if (std::sscanf(cvars::serialize_draw_range.c_str(), "%d..%d", &a, &b) != 2) {
    return;
  }
  int32_t idx = int32_t(draw_index);
  if (idx < a || idx > b) {
    return;
  }
  FlushForDrawSerialization();
}

void CommandProcessor::MaybeCaptureEdramSnapshot(uint32_t draw_index) {
  bool want_snapshot = IsCaptureEdramDraw(draw_index);
  bool want_hashes =
      cvars::capture_edram_hashes && !cvars::capture_dir.empty();
  if (!want_snapshot && !want_hashes) {
    return;
  }
  // Per-draw captures and sample-plane hashes describe the 1x EDRAM image. At
  // a draw resolution scale (which the ZPD normalization fields carry) the
  // EDRAM is scaled, so they are unsupported rather than mixing scales.
  if (zpd_draw_resolution_scale_x_ != 1 || zpd_draw_resolution_scale_y_ != 1) {
    static bool s_scaled_capture_logged = false;
    if (!s_scaled_capture_logged) {
      s_scaled_capture_logged = true;
      XELOGE(
          "capture_dir: per-draw EDRAM capture and hashing are unsupported at "
          "a draw resolution scale ({}x{})",
          zpd_draw_resolution_scale_x_, zpd_draw_resolution_scale_y_);
    }
    return;
  }
  // One EDRAM download serves both the full snapshot and the hash record.
  std::vector<uint8_t> edram;
  if (!DumpEdramToBuffer(edram) || edram.size() != xenos::kEdramSizeBytes) {
    XELOGE("capture_dir: EDRAM download unsupported/failed for draw {}",
           draw_index);
    return;
  }
  if (want_snapshot) {
    std::filesystem::path capture_dir =
        std::filesystem::path(cvars::capture_dir);
    std::error_code ec;
    std::filesystem::create_directories(capture_dir, ec);
    std::filesystem::path snapshot_path =
        capture_dir / fmt::format("draw_{:06d}.edram.bin", draw_index);
    FILE* file = xe::filesystem::OpenFile(snapshot_path, "wb");
    bool written = false;
    if (file) {
      written = fwrite(edram.data(), 1, edram.size(), file) == edram.size();
      fclose(file);
    }
    if (!written) {
      XELOGE("capture_dir: failed to write {}", snapshot_path.string());
    }
  }
  if (want_hashes) {
    AppendCaptureEdramHashes(draw_index, edram);
  }
}

bool CommandProcessor::DumpEdramToFile(const std::filesystem::path& path) {
  std::vector<uint8_t> edram;
  if (!DumpEdramToBuffer(edram)) {
    return false;
  }
  FILE* file = xe::filesystem::OpenFile(path, "wb");
  if (!file) {
    return false;
  }
  bool written = fwrite(edram.data(), 1, edram.size(), file) == edram.size();
  fclose(file);
  return written;
}

void CommandProcessor::AppendCaptureEdramHashes(
    uint32_t draw_index, const std::vector<uint8_t>& edram) {
  constexpr uint32_t kTileBytes = xenos::kEdramTileWidthSamples *
                                  xenos::kEdramTileHeightSamples * 4;
  constexpr uint32_t kTileCount = xenos::kEdramTileCount;
  constexpr uint32_t kSamplePlaneCount = 4;
  constexpr uint32_t kWordsPerSamplePlane =
      xenos::kEdramTileWidthSamples * xenos::kEdramTileHeightSamples /
      kSamplePlaneCount;
  static_assert(kTileBytes * kTileCount == xenos::kEdramSizeBytes);
  static_assert(kWordsPerSamplePlane * kSamplePlaneCount * sizeof(uint32_t) ==
                kTileBytes);
  if (edram.size() != xenos::kEdramSizeBytes) {
    XELOGE("capture_dir: expected {} EDRAM bytes for hashing, got {}",
           xenos::kEdramSizeBytes, edram.size());
    return;
  }
  if (!capture_hashes_file_) {
    std::filesystem::path capture_dir =
        std::filesystem::path(cvars::capture_dir);
    std::error_code ec;
    std::filesystem::create_directories(capture_dir, ec);
    std::filesystem::path hashes_path = capture_dir / "edram_hashes.bin";
    capture_hashes_file_ = xe::filesystem::OpenFile(hashes_path, "wb");
    if (!capture_hashes_file_) {
      XELOGE("capture_dir: failed to open {}", hashes_path.string());
      return;
    }
    const uint32_t header[4] = {0x48444558u /* 'XEDH' */, 2u, kTileCount,
                                kTileBytes};
    fwrite(header, sizeof(header), 1, capture_hashes_file_);
  }
  uint64_t hashes[kTileCount * kSamplePlaneCount];
  uint32_t sample_plane_words[kSamplePlaneCount][kWordsPerSamplePlane];
  for (uint32_t tile = 0; tile < kTileCount; ++tile) {
    uint32_t sample_plane_word_counts[kSamplePlaneCount] = {};
    const uint8_t* tile_bytes = edram.data() + size_t(tile) * kTileBytes;
    for (uint32_t y = 0; y < xenos::kEdramTileHeightSamples; ++y) {
      for (uint32_t x = 0; x < xenos::kEdramTileWidthSamples; ++x) {
        // XeEdramOffsetBytes places the 4x sample index as
        // ((sample >> 1) & 1, sample & 1). Classifying the raw 80x16 dword
        // tile by x/y parity therefore yields the canonical 32bpp physical
        // sample plane without needing render-target metadata. The 40-column
        // depth swap preserves parity.
        uint32_t tile_word = y * xenos::kEdramTileWidthSamples + x;
        uint32_t sample_plane = EdramSamplePlaneForTileWord(tile_word);
        uint32_t word;
        std::memcpy(&word,
                    tile_bytes + size_t(tile_word) * sizeof(uint32_t),
                    sizeof(word));
        uint32_t& sample_plane_word_count =
            sample_plane_word_counts[sample_plane];
        sample_plane_words[sample_plane][sample_plane_word_count++] = word;
      }
    }
    for (uint32_t sample_plane = 0; sample_plane < kSamplePlaneCount;
         ++sample_plane) {
      assert_true(sample_plane_word_counts[sample_plane] ==
                  kWordsPerSamplePlane);
      hashes[tile * kSamplePlaneCount + sample_plane] =
          XXH3_64bits(sample_plane_words[sample_plane],
                      sizeof(sample_plane_words[sample_plane]));
    }
  }
  fwrite(&draw_index, sizeof(draw_index), 1, capture_hashes_file_);
  fwrite(hashes, sizeof(uint64_t), kTileCount * kSamplePlaneCount,
         capture_hashes_file_);
  // Flush per record: the two-machine workflow may SIGALRM-kill the replay
  // before stdio buffers drain (same policy as the manifest).
  fflush(capture_hashes_file_);
}

void CommandProcessor::WriteCaptureManifestRow(const std::string& json_row) {
  if (cvars::capture_dir.empty()) {
    return;
  }
  if (!capture_meta_written_) {
    std::filesystem::path capture_dir =
        std::filesystem::path(cvars::capture_dir);
    std::error_code ec;
    std::filesystem::create_directories(capture_dir, ec);
    WriteCaptureMeta(capture_dir);
    std::filesystem::path manifest_path = capture_dir / "manifest.jsonl";
    capture_manifest_file_ = xe::filesystem::OpenFile(manifest_path, "wb");
    if (!capture_manifest_file_) {
      XELOGE("capture_dir: failed to open {}", manifest_path.string());
    }
    // Set regardless of success so we do not retry the open every draw.
    capture_meta_written_ = true;
  }
  if (capture_manifest_file_) {
    fwrite(json_row.data(), 1, json_row.size(), capture_manifest_file_);
    fputc('\n', capture_manifest_file_);
    // Flush per row: trace-dump in the two-machine workflow is wrapped in a
    // perl alarm and may be SIGALRM-killed before stdio buffers drain.
    fflush(capture_manifest_file_);
  }
}

bool CommandProcessor::IsZPDReportCurrent(const ZPDReport& report) const {
  auto it = zpd_slot_sequences_.find(report.slot_base);
  uint64_t current_sequence = it != zpd_slot_sequences_.end() ? it->second : 0;
  return current_sequence == report.slot_sequence_id;
}

uint32_t CommandProcessor::NormalizeSampleCount(uint64_t samples) const {
  if (samples == 0) {
    return 0;
  }

  uint64_t scale = zpd_draw_resolution_scale_x_ * zpd_draw_resolution_scale_y_;
  // Round, don't truncate. 1 guest sample at 2x = 4 host samples, need >= 1.
  uint64_t normalized = scale <= 1 ? samples : (samples + (scale >> 1)) / scale;

  return static_cast<uint32_t>(std::min<uint64_t>(normalized, UINT32_MAX));
}
#define COMMAND_PROCESSOR CommandProcessor
#include "pm4_command_processor_implement.h"
}  // namespace gpu
}  // namespace xe
