/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_SDL_SDL_AUDIO_DRIVER_H_
#define XENIA_APU_SDL_SDL_AUDIO_DRIVER_H_

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <queue>
#include <stack>
#include <vector>

#include <SDL3/SDL.h>
#include "xenia/apu/audio_driver.h"
#include "xenia/base/threading.h"

namespace xe {
namespace apu {
namespace sdl {

class SDLAudioDriver : public AudioDriver {
 public:
  SDLAudioDriver(xe::threading::Semaphore* semaphore,
                 uint32_t frequency = kFrameFrequencyDefault,
                 uint32_t channels = kFrameChannelsDefault,
                 bool need_format_conversion = true);
  ~SDLAudioDriver() override;

  bool Initialize() override;
  void SubmitFrame(float* frame) override;
  void Pause() override;
  void Resume() override;
  void SetVolume(float volume) override { volume_ = volume; };
  void Shutdown() override;

 protected:
  // SDL3: the device asks for at least `additional_amount` more bytes; they
  // are supplied one guest frame at a time (FillFrameLocked).
  static void SDLCALL SDLCallback(void* userdata, SDL_AudioStream* stream,
                                  int additional_amount, int total_amount);
  // Writes one frame for the device into frame_out_ from the queue (or
  // silence when it is empty) and releases the guest's semaphore for a frame
  // it took. Called with frames_mutex_ held.
  void FillFrameLocked();

  xe::threading::Semaphore* semaphore_ = nullptr;

  // The playback stream bound to the default device (SDL3's
  // SDL_OpenAudioDeviceStream); its device id labels the stats lines.
  SDL_AudioStream* sdl_stream_ = nullptr;
  uint32_t sdl_device_id_ = 0;
  // One frame in the device's layout (channel_samples_ x sdl_device_channels_
  // floats), reused by the callback.
  std::vector<float> frame_out_;
  bool sdl_initialized_ = false;
  uint8_t sdl_device_channels_ = 0;

  float volume_ = 1.0f;

  uint32_t frame_frequency_;
  uint32_t frame_channels_;
  uint32_t channel_samples_;
  uint32_t frame_size_;
  bool need_format_conversion_;
  std::queue<float*> frames_queued_ = {};
  std::stack<float*> frames_unused_ = {};
  std::mutex frames_mutex_ = {};

  // XE_APU_STATS=1: device-side pacing, logged every 5 s from the SDL
  // callback (guarded by frames_mutex_). An underrun is a device callback
  // that found no guest frame queued and played silence.
  void LogStatsLocked();
  bool stats_enabled_ = false;
  uint64_t stats_window_start_ticks_ = 0;
  uint32_t stats_callbacks_ = 0;
  uint32_t stats_submitted_ = 0;
  uint32_t stats_underruns_ = 0;
  uint32_t stats_underrun_runs_ = 0;
  bool stats_in_underrun_ = false;
  size_t stats_depth_min_ = SIZE_MAX;
  size_t stats_depth_max_ = 0;

  // XE_APU_DUMP=<path>: every frame handed to the device (underrun silence
  // included, before mute and volume) appended to <path> (<path>.N for a
  // later device) as raw interleaved little-endian float32 at
  // frame_frequency_ x sdl_device_channels_, and one byte per frame to
  // <path>.flags (1 = underrun). Diagnostic only.
  FILE* dump_file_ = nullptr;
  FILE* dump_flags_file_ = nullptr;
};

}  // namespace sdl
}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_SDL_SDL_AUDIO_DRIVER_H_
