/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/nop/nop_audio_system.h"

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/nop/nop_audio_driver.h"
#include "xenia/base/assert.h"

namespace xe {
namespace apu {
namespace nop {

std::unique_ptr<AudioSystem> NopAudioSystem::Create(cpu::Processor* processor) {
  return std::make_unique<NopAudioSystem>(processor);
}

NopAudioSystem::NopAudioSystem(cpu::Processor* processor)
    : AudioSystem(processor) {}

NopAudioSystem::~NopAudioSystem() = default;

X_STATUS NopAudioSystem::CreateDriver(size_t index,
                                      xe::threading::Semaphore* semaphore,
                                      AudioDriver** out_driver) {
  assert_not_null(out_driver);
  // Provide a silent driver that still paces the audio worker, so titles that
  // register an audio render driver get a valid handle instead of a failed
  // registration they may dereference as null.
  auto driver = std::make_unique<NopAudioDriver>(semaphore);
  if (!driver->Initialize()) {
    driver->Shutdown();
    return X_STATUS_UNSUCCESSFUL;
  }
  *out_driver = driver.release();
  return X_STATUS_SUCCESS;
}

AudioDriver* NopAudioSystem::CreateDriver(xe::threading::Semaphore* semaphore,
                                          uint32_t frequency, uint32_t channels,
                                          bool need_format_conversion) {
  return nullptr;
}

void NopAudioSystem::DestroyDriver(AudioDriver* driver) {
  if (!driver) {
    return;
  }
  driver->Shutdown();
  delete driver;
}

}  // namespace nop
}  // namespace apu
}  // namespace xe
