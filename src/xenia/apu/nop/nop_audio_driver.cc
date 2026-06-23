/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/nop/nop_audio_driver.h"

#include <chrono>

namespace xe {
namespace apu {
namespace nop {

NopAudioDriver::NopAudioDriver(xe::threading::Semaphore* semaphore)
    : semaphore_(semaphore) {}

NopAudioDriver::~NopAudioDriver() { Shutdown(); }

bool NopAudioDriver::Initialize() {
  if (running_.exchange(true)) {
    return true;
  }
  pacing_thread_ = std::thread([this]() { PacingThreadMain(); });
  return true;
}

void NopAudioDriver::Shutdown() {
  if (!running_.exchange(false)) {
    return;
  }
  if (pacing_thread_.joinable()) {
    pacing_thread_.join();
  }
}

void NopAudioDriver::SubmitFrame(float* samples) {
  // Discard the audio; just account for the frame so the pacing thread releases
  // exactly one semaphore permit per submitted frame.
  queued_frames_.fetch_add(1, std::memory_order_relaxed);
}

void NopAudioDriver::Pause() { paused_.store(true, std::memory_order_relaxed); }

void NopAudioDriver::Resume() {
  paused_.store(false, std::memory_order_relaxed);
}

void NopAudioDriver::SetVolume(float volume) {}

void NopAudioDriver::PacingThreadMain() {
  xe::threading::set_name("xe::apu::nop Pacing");
  // One frame is kChannelSamplesDefault samples per channel at
  // kFrameFrequencyDefault Hz, i.e. ~5.33 ms for 256 samples at 48 kHz.
  const auto frame_period = std::chrono::microseconds(
      (UINT64_C(1000000) * AudioDriver::kChannelSamplesDefault) /
      AudioDriver::kFrameFrequencyDefault);
  while (running_.load(std::memory_order_relaxed)) {
    if (paused_.load(std::memory_order_relaxed) ||
        queued_frames_.load(std::memory_order_relaxed) == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    std::this_thread::sleep_for(frame_period);
    queued_frames_.fetch_sub(1, std::memory_order_relaxed);
    // Signal the worker that a frame "finished playing". If the semaphore is at
    // its max count (worker fell behind) the release is dropped harmlessly.
    semaphore_->Release(1, nullptr);
  }
}

}  // namespace nop
}  // namespace apu
}  // namespace xe
