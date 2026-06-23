/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_NOP_NOP_AUDIO_DRIVER_H_
#define XENIA_APU_NOP_NOP_AUDIO_DRIVER_H_

#include <atomic>
#include <thread>

#include "xenia/apu/audio_driver.h"
#include "xenia/base/threading.h"

namespace xe {
namespace apu {
namespace nop {

// A silent audio driver. It accepts and discards submitted frames but still
// paces the audio worker by releasing the client semaphore at the real-time
// frame rate, exactly like a hardware backend's buffer-completion callback
// (compare SDLAudioDriver::SDLCallback). This lets a title that registers an
// audio render driver run normally without an actual audio backend instead of
// receiving a failed registration and dereferencing a null driver handle
// (the cause of the macOS Banjo-Kazooie / Rick360 crashes when SDL audio is
// unavailable). It deliberately produces no sound.
class NopAudioDriver : public AudioDriver {
 public:
  explicit NopAudioDriver(xe::threading::Semaphore* semaphore);
  ~NopAudioDriver() override;

  bool Initialize() override;
  void Shutdown() override;

  void SubmitFrame(float* samples) override;
  void Pause() override;
  void Resume() override;
  void SetVolume(float volume) override;

 private:
  void PacingThreadMain();

  xe::threading::Semaphore* semaphore_ = nullptr;
  std::thread pacing_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> paused_{false};
  // Number of submitted-but-not-yet-"played" frames. The pacing thread releases
  // one semaphore permit per submitted frame so it never releases more than
  // were submitted (keeping the semaphore within its max count).
  std::atomic<uint32_t> queued_frames_{0};
};

}  // namespace nop
}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_NOP_NOP_AUDIO_DRIVER_H_
