/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/sdl/sdl_audio_driver.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/conversion.h"
#include "xenia/base/assert.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/helper/sdl/sdl_helper.h"

namespace xe {
namespace apu {
namespace sdl {

SDLAudioDriver::SDLAudioDriver(xe::threading::Semaphore* semaphore,
                               uint32_t frequency, uint32_t channels,
                               bool need_format_conversion)
    : semaphore_(semaphore),
      frame_frequency_(frequency),
      frame_channels_(channels),
      need_format_conversion_(need_format_conversion) {
  switch (frame_channels_) {
    case 6:
      channel_samples_ = 256;
      break;
    case 2:
      channel_samples_ = 768;
      break;
    default:
      assert_unhandled_case(frame_channels_);
  }
  frame_size_ = sizeof(float) * frame_channels_ * channel_samples_;
  assert_true(frame_size_ <= kFrameSizeMax);
  assert_true(!need_format_conversion_ || frame_channels_ == 6);
}

SDLAudioDriver::~SDLAudioDriver() {
  assert_true(frames_queued_.empty());
  assert_true(frames_unused_.empty());
};

bool SDLAudioDriver::Initialize() {
  if (!xe::helper::sdl::SDLHelper::Prepare()) {
    return false;
  }
  if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
    return false;
  }
  sdl_initialized_ = true;

  // What the default output device is: stereo gets the driver's own 5.1 to
  // stereo fold, anything else is handed 5.1 (SDL converts a 4.0 or 7.1
  // layout itself, as before). A stereo guest stream stays stereo.
  SDL_AudioSpec device_spec = {};
  int device_frames = 0;
  const bool have_device_format = SDL_GetAudioDeviceFormat(
      SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &device_spec, &device_frames);
  sdl_device_channels_ =
      (frame_channels_ == 2 || (have_device_format && device_spec.channels == 2))
          ? 2
          : 6;

  // SDL2 took the callback size in the device spec; SDL3 reads it from this
  // hint when it opens the device. One guest frame, as before, keeps the
  // output latency where it was.
  SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES,
              std::to_string(channel_samples_).c_str());

  SDL_AudioSpec spec = {};
  spec.format = SDL_AUDIO_F32;
  spec.channels = sdl_device_channels_;
  spec.freq = int(frame_frequency_);
  frame_out_.assign(size_t(channel_samples_) * sdl_device_channels_, 0.0f);
  sdl_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
                                          &spec, SDLCallback, this);
  if (!sdl_stream_) {
    XELOGE("SDL_OpenAudioDeviceStream() failed: {}", SDL_GetError());
    return false;
  }
  sdl_device_id_ = uint32_t(SDL_GetAudioStreamDevice(sdl_stream_));

  stats_enabled_ = std::getenv("XE_APU_STATS") != nullptr;
  if (stats_enabled_) {
    // The device's own format now that it is open.
    SDL_AudioSpec opened = {};
    int opened_frames = 0;
    SDL_GetAudioDeviceFormat(SDL_AudioDeviceID(sdl_device_id_), &opened,
                             &opened_frames);
    XELOGI(
        "ApuStats device {}: stream {} Hz x {} channels ({} guest channels) "
        "into a {} Hz x {} channel device, {} sample frames/callback (needs "
        "{:.1f} guest frames/s)",
        sdl_device_id_, spec.freq, sdl_device_channels_, frame_channels_,
        opened.freq, opened.channels, opened_frames,
        double(frame_frequency_) / double(channel_samples_));
  }

  if (const char* dump_env = std::getenv("XE_APU_DUMP")) {
    // A title can register more than one audio client: the first device
    // writes <path>, later ones <path>.1, <path>.2 and so on.
    static std::atomic<uint32_t> dump_count{0};
    const uint32_t dump_index = dump_count.fetch_add(1);
    const std::string dump_path =
        dump_index ? fmt::format("{}.{}", dump_env, dump_index) : dump_env;
    dump_file_ = std::fopen(dump_path.c_str(), "wb");
    dump_flags_file_ = std::fopen((dump_path + ".flags").c_str(), "wb");
    XELOGI("ApuDump: {} Hz x {} channels float32 to {}", frame_frequency_,
           sdl_device_channels_, dump_path);
  }

  // The stream's device starts paused.
  SDL_ResumeAudioStreamDevice(sdl_stream_);

  return true;
}

void SDLAudioDriver::SubmitFrame(float* frame) {
  float* output_frame;
  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    if (frames_unused_.empty()) {
      output_frame = new float[frame_channels_ * channel_samples_];
    } else {
      output_frame = frames_unused_.top();
      frames_unused_.pop();
    }
  }

  std::memcpy(output_frame, frame, frame_size_);

  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    frames_queued_.push(output_frame);
    ++stats_submitted_;
  }
}

void SDLAudioDriver::LogStatsLocked() {
  const uint64_t now = SDL_GetTicks();
  if (!stats_window_start_ticks_) {
    stats_window_start_ticks_ = now;
    return;
  }
  const uint64_t elapsed_ms = now - stats_window_start_ticks_;
  if (elapsed_ms < 5000) {
    return;
  }
  const double seconds = double(elapsed_ms) / 1000.0;
  XELOGI(
      "ApuStats device {} {:.1f}s: callbacks {} ({:.1f}/s), guest frames "
      "submitted {} ({:.1f}/s), underruns {} in {} runs ({:.1f}% silent), "
      "queue depth {}-{}",
      sdl_device_id_, seconds, stats_callbacks_, stats_callbacks_ / seconds,
      stats_submitted_, stats_submitted_ / seconds, stats_underruns_,
      stats_underrun_runs_,
      stats_callbacks_ ? 100.0 * stats_underruns_ / stats_callbacks_ : 0.0,
      stats_depth_min_ == SIZE_MAX ? 0 : stats_depth_min_, stats_depth_max_);
  stats_window_start_ticks_ = now;
  stats_callbacks_ = 0;
  stats_submitted_ = 0;
  stats_underruns_ = 0;
  stats_underrun_runs_ = 0;
  stats_depth_min_ = SIZE_MAX;
  stats_depth_max_ = 0;
}

void SDLAudioDriver::Pause() {
  if (sdl_stream_) {
    SDL_PauseAudioStreamDevice(sdl_stream_);
  }
}

void SDLAudioDriver::Resume() {
  if (sdl_stream_) {
    SDL_ResumeAudioStreamDevice(sdl_stream_);
  }
}

void SDLAudioDriver::Shutdown() {
  if (sdl_stream_) {
    // Destroying the stream closes the device it was opened with, and
    // returns only once the callback is no longer running.
    SDL_DestroyAudioStream(sdl_stream_);
    sdl_stream_ = nullptr;
    sdl_device_id_ = 0;
  }
  if (dump_file_) {
    std::fclose(dump_file_);
    dump_file_ = nullptr;
  }
  if (dump_flags_file_) {
    std::fclose(dump_flags_file_);
    dump_flags_file_ = nullptr;
  }
  if (sdl_initialized_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    sdl_initialized_ = false;
  }
  std::unique_lock<std::mutex> guard(frames_mutex_);
  while (!frames_unused_.empty()) {
    delete[] frames_unused_.top();
    frames_unused_.pop();
  };
  while (!frames_queued_.empty()) {
    delete[] frames_queued_.front();
    frames_queued_.pop();
  };
}

void SDLAudioDriver::FillFrameLocked() {
  const size_t count = frame_out_.size();
  if (stats_enabled_) {
    const size_t depth = frames_queued_.size();
    ++stats_callbacks_;
    stats_depth_min_ = std::min(stats_depth_min_, depth);
    stats_depth_max_ = std::max(stats_depth_max_, depth);
    if (!depth) {
      ++stats_underruns_;
      if (!stats_in_underrun_) {
        ++stats_underrun_runs_;
      }
    }
    stats_in_underrun_ = !depth;
    LogStatsLocked();
  }
  if (frames_queued_.empty()) {
    std::fill(frame_out_.begin(), frame_out_.end(), 0.0f);
    if (dump_file_) {
      std::fwrite(frame_out_.data(), sizeof(float), count, dump_file_);
      if (dump_flags_file_) {
        std::fputc(1, dump_flags_file_);
      }
    }
    return;
  }
  auto buffer = frames_queued_.front();
  frames_queued_.pop();
  if (cvars::mute && !dump_file_) {
    std::fill(frame_out_.begin(), frame_out_.end(), 0.0f);
  } else if (need_format_conversion_) {
    switch (sdl_device_channels_) {
      case 2:
        conversion::sequential_6_BE_to_interleaved_2_LE(
            frame_out_.data(), buffer, channel_samples_);
        break;
      case 6:
        conversion::sequential_6_BE_to_interleaved_6_LE(
            frame_out_.data(), buffer, channel_samples_);
        break;
      default:
        assert_unhandled_case(sdl_device_channels_);
        break;
    }
  } else {
    assert_true(sdl_device_channels_ == frame_channels_);
    // (SDL2's SDL_MixAudioFormat into a zeroed buffer: a plain scale.)
    for (size_t i = 0; i < count; ++i) {
      frame_out_[i] = buffer[i] * volume_;
    }
  }
  if (dump_file_) {
    std::fwrite(frame_out_.data(), sizeof(float), count, dump_file_);
    if (dump_flags_file_) {
      std::fputc(0, dump_flags_file_);
    }
    if (cvars::mute) {
      std::fill(frame_out_.begin(), frame_out_.end(), 0.0f);
    }
  }
  // The volume setting, on a squared curve so each step sounds about as
  // big as the last (half the slider is a quarter of the amplitude).
  const int volume = std::clamp(int(cvars::volume), 0, 100);
  if (!cvars::mute && volume < 100) {
    const float gain = float(volume * volume) / (100.0f * 100.0f);
    for (size_t i = 0; i < count; ++i) {
      frame_out_[i] *= gain;
    }
  }
  frames_unused_.push(buffer);

  auto ret = semaphore_->Release(1, nullptr);
  assert_true(ret);
}

void SDLCALL SDLAudioDriver::SDLCallback(void* userdata,
                                         SDL_AudioStream* stream,
                                         int additional_amount,
                                         int total_amount) {
  SCOPE_profile_cpu_f("apu");
  if (!userdata || !stream) {
    XELOGE("SDLAudioDriver::sdl_callback called with nullptr.");
    return;
  }
  const auto driver = static_cast<SDLAudioDriver*>(userdata);
  const int frame_bytes = int(driver->frame_out_.size() * sizeof(float));
  // SDL2 called back once per guest frame; SDL3 asks for a number of bytes,
  // which is met in whole guest frames so each one still paces the guest
  // through its semaphore.
  std::unique_lock<std::mutex> guard(driver->frames_mutex_);
  while (additional_amount > 0) {
    driver->FillFrameLocked();
    SDL_PutAudioStreamData(stream, driver->frame_out_.data(), frame_bytes);
    additional_amount -= frame_bytes;
  }
};
}  // namespace sdl
}  // namespace apu
}  // namespace xe
