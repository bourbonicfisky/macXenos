/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_dump.h"

#include "third_party/stb/stb_image_write.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/base/string.h"
#include "xenia/base/threading.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/trace_analysis_decode.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/memory.h"
#include "xenia/ui/file_picker.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/window.h"
#include "xenia/xbox.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#undef _CRT_SECURE_NO_WARNINGS
#undef _CRT_NONSTDC_NO_DEPRECATE
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "third_party/stb/stb_image_write.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

DEFINE_path(target_trace_file, "", "Specifies the trace file to load.", "GPU");
DEFINE_path(trace_dump_path, "", "Output path for dumped files.", "GPU");
DEFINE_string(dump_guest_memory, "",
              "After playback, dump a guest physical memory range next to the "
              "output image as <output>.mem.bin. Format: hexaddr,hexlength "
              "(e.g. 1D06C000,26AC00). Combine with --readback_resolve=full "
              "to capture resolve destinations.",
              "GPU");
DEFINE_int32(replay_loops, 1,
             "Replay the traced frame N times before capturing (async-race "
             "stress; see TraceDump::Run).",
             "GPU");
DEFINE_bool(dump_edram, false,
            "After playback, dump the raw 10 MB EDRAM contents (per-sample, "
            "tile-major) next to the output image as <output>.edram.bin.",
            "GPU");
DEFINE_bool(edram_determinism, false,
            "Replay-reset determinism: replay the SAME frame --replay_loops "
            "times and require the EDRAM dump is byte-identical after every "
            "loop. Each replay restores the trace's captured EDRAM snapshot + "
            "memory reads, so given the same restored inputs, persistent backend "
            "caches must not change the result. A divergence is hidden backend "
            "state leaking across replays (stale RT persistence, uninitialized "
            "reuse) -- invisible to a single-frame dump. NOT a temporal/"
            "multi-frame correctness test (no genuine frame N->N+1 feedback), "
            "and only as complete as the dump oracle (Metal skips 64bpp owners; "
            "collapsed backends can't expose lost samples -- stable zeros pass). "
            "Needs --replay_loops>=2. Exits nonzero on failure.",
            "GPU");
DEFINE_string(decode, "",
              "Offline decode mode (no replay): decode every draw_*.edram.bin "
              "in the given capture directory into color_drawNNNNNN.png using "
              "params from its manifest.jsonl, then exit. Runs the EDRAM "
              "offset self-test first.",
              "GPU");
DEFINE_uint64(decode_width, 0,
              "--decode: color decode width in pixels (0 = manifest surface "
              "pitch).",
              "GPU");
DEFINE_uint64(decode_height, 0,
              "--decode: color decode height in pixels (0 = 720).", "GPU");
DEFINE_uint64(decode_sample, 0,
              "--decode: MSAA sample index to decode (0..samples-1).", "GPU");

namespace xe {
namespace gpu {

using namespace xe::gpu::xenos;

TraceDump::TraceDump() = default;

TraceDump::~TraceDump() = default;

int TraceDump::Main(const std::vector<std::string>& args) {
  // Offline decode mode: no emulator/replay -- just decode a capture directory.
  if (!cvars::decode.empty()) {
    int decoded = DecodeCaptureDirectory(
        std::filesystem::path(cvars::decode), uint32_t(cvars::decode_width),
        uint32_t(cvars::decode_height), uint32_t(cvars::decode_sample));
    return decoded >= 0 ? 0 : 1;
  }

  // Grab path from the flag or unnamed argument.
  std::filesystem::path path;
  std::filesystem::path output_path;
  if (!cvars::target_trace_file.empty()) {
    // Passed as a named argument.
    // TODO(benvanik): find something better than gflags that supports
    // unicode.
    path = cvars::target_trace_file;
  } else if (args.size() >= 2) {
    // Passed as an unnamed argument.
    path = xe::to_path(args[1]);

    if (args.size() >= 3) {
      output_path = xe::to_path(args[2]);
    }
  }

  if (path.empty()) {
    XELOGE("No trace file specified");
    return 5;
  }

  // Normalize the path and make absolute.
  auto abs_path = std::filesystem::absolute(path);
  XELOGI("Loading trace file {}...", abs_path);

  if (!Setup()) {
    XELOGE("Unable to setup trace dump tool");
    return 4;
  }
  if (!Load(std::move(abs_path))) {
    XELOGE("Unable to load trace file; not found?");
    return 5;
  }

  // Root file name for outputs.
  if (output_path.empty()) {
    base_output_path_ = cvars::trace_dump_path;
    auto output_name = path.filename().replace_extension();

    base_output_path_ = base_output_path_ / output_name;
  } else {
    base_output_path_ = output_path;
  }

  // Ensure output path exists.
  xe::filesystem::CreateParentFolder(base_output_path_);

  return Run();
}

bool TraceDump::Setup() {
  // Trace replay renders a single frame with pipelines that have never been
  // created before. With asynchronous pipeline creation, every draw would be
  // recorded with the transparent-black placeholder pixel shader (the real
  // pipelines are hot-swapped in only on later frames, which a single-frame
  // replay doesn't have), producing an all-black dump. Force synchronous
  // pipeline creation for deterministic replay output.
  cvars::async_shader_compilation = false;

  // Create the emulator but don't initialize so we can setup the window.
  emulator_ = std::make_unique<Emulator>("", "", "", "");
  X_STATUS result = emulator_->Setup(
      nullptr, nullptr, false, nullptr,
      [this]() { return CreateGraphicsSystem(); }, nullptr);
  if (XFAILED(result)) {
    XELOGE("Failed to setup emulator: {:08X}", result);
    return false;
  }
  graphics_system_ = emulator_->graphics_system();
  player_ = std::make_unique<TracePlayer>(graphics_system_);
  return true;
}

bool TraceDump::Load(const std::filesystem::path& trace_file_path) {
  trace_file_path_ = trace_file_path;

  if (!player_->Open(xe::path_to_utf8(trace_file_path_))) {
    XELOGE("Could not load trace file");
    return false;
  }

  return true;
}

int TraceDump::Run() {
  BeginHostCapture();
  XELOGI("TraceDump: {} frames parsed; frame 0 has {} commands",
         player_->frame_count(),
         player_->frame_count()
             ? player_->frame(0)->commands.size()
             : 0);
  if (!player_->frame_count() || player_->frame(0)->commands.empty()) {
    XELOGE(
        "TraceDump: trace has no draw commands (captured while the GPU was "
        "idle, e.g. a static menu that only redraws on change) -- capture "
        "during animation/transitions instead");
    return 1;
  }
  // Replay-loop stress (--replay_loops=N): re-plays the frame N times before
  // the capture. Under async submission this creates cross-iteration
  // concurrency (iteration k's GPU work overlaps k+1's encoding) — the
  // in-agent reproducer for lifetime races that only fire at speed.
  int32_t replay_loops = std::max(int32_t(1), cvars::replay_loops);
  player_->SeekFrame(0);
  player_->SeekCommand(
      static_cast<int>(player_->current_frame()->commands.size() - 1));
  player_->WaitOnPlayback();
  // Replay-reset determinism check: TracePlayer restores the captured EDRAM
  // snapshot + memory reads on every replay, so given identical restored
  // inputs the observable EDRAM dump must be byte-identical every loop.
  // Divergence => hidden backend state leaking across replays (the class the
  // MSAA persistence regression belonged to). This is NOT temporal multi-frame
  // correctness (no real N->N+1 feedback) and is only as complete as the dump
  // oracle. Baseline captured after loop 0.
  bool determinism = cvars::edram_determinism && replay_loops >= 2;
  std::vector<uint8_t> det_baseline;
  bool determinism_failed = false;
  if (determinism) {
    if (!graphics_system_->command_processor()->DumpEdramToBuffer(
            det_baseline)) {
      XELOGW("--edram_determinism: baseline EDRAM dump failed; skipping check");
      determinism = false;
    }
  }
  for (int32_t loop = 1; loop < replay_loops; ++loop) {
    player_->ReplayCurrentFrame();
    player_->WaitOnPlayback();
    if (determinism) {
      std::vector<uint8_t> cur;
      if (!graphics_system_->command_processor()->DumpEdramToBuffer(cur) ||
          cur.size() != det_baseline.size()) {
        XELOGE("--edram_determinism: loop {} EDRAM dump failed/size mismatch",
               loop);
        determinism_failed = true;
        continue;
      }
      size_t diffs = 0, first = cur.size();
      for (size_t i = 0; i < cur.size(); ++i) {
        if (cur[i] != det_baseline[i]) {
          if (!diffs) first = i;
          ++diffs;
        }
      }
      if (diffs) {
        XELOGE(
            "EDRAM DETERMINISM FAIL: loop {} differs from loop 0 -- {} / {} "
            "bytes, first at byte {} (word {}). Cross-frame-state bug.",
            loop, diffs, cur.size(), first, first / 4);
        determinism_failed = true;
      } else {
        XELOGI("EDRAM determinism: loop {} == loop 0", loop);
      }
    }
  }
  if (determinism) {
    if (determinism_failed) {
      XELOGE("EDRAM DETERMINISM: FAILED ({} loops)", replay_loops);
    } else {
      XELOGI("EDRAM DETERMINISM: PASS -- {} loops byte-identical", replay_loops);
    }
  }
  EndHostCapture();

  if (cvars::dump_edram) {
    auto edram_path = base_output_path_;
    edram_path.replace_extension(".edram.bin");
    if (graphics_system_->command_processor()->DumpEdramToFile(edram_path)) {
      XELOGI("Dumped EDRAM contents to {}", xe::path_to_utf8(edram_path));
    } else {
      XELOGE("Failed to dump EDRAM contents");
    }
  }

  if (!cvars::dump_guest_memory.empty()) {
    uint32_t dump_address = 0, dump_length = 0;
    if (std::sscanf(cvars::dump_guest_memory.c_str(), "%x,%x", &dump_address,
                    &dump_length) == 2 &&
        dump_length) {
      const uint8_t* dump_source =
          emulator_->memory()->TranslatePhysical(dump_address);
      auto mem_path = base_output_path_;
      mem_path.replace_extension(".mem.bin");
      auto mem_file = filesystem::OpenFile(mem_path, "wb");
      fwrite(dump_source, 1, dump_length, mem_file);
      fclose(mem_file);
      XELOGI("Dumped guest memory {:08X}+{:X} to {}", dump_address,
             dump_length, xe::path_to_utf8(mem_path));
    } else {
      XELOGE("dump_guest_memory: expected hexaddr,hexlength");
    }
  }

  // Capture. A determinism failure must make the process exit nonzero (exit
  // code 2) so a CI/shell gate actually fails; otherwise the check is
  // advisory-only and a broken backend passes silently.
  int result = determinism_failed ? 2 : 0;
  ui::Presenter* presenter = graphics_system_->presenter();
  ui::RawImage raw_image;
  if (presenter && presenter->CaptureGuestOutput(raw_image)) {
    // Save framebuffer png.
    auto png_path = base_output_path_.replace_extension(".png");
    auto handle = filesystem::OpenFile(png_path, "wb");
    auto callback = [](void* context, void* data, int size) {
      fwrite(data, 1, size, (FILE*)context);
    };
    stbi_write_png_to_func(callback, handle, static_cast<int>(raw_image.width),
                           static_cast<int>(raw_image.height), 4,
                           raw_image.data.data(),
                           static_cast<int>(raw_image.stride));
    fclose(handle);
  } else {
    result = 1;
  }

  player_.reset();
  emulator_.reset();
  return result;
}

}  //  namespace gpu
}  //  namespace xe
