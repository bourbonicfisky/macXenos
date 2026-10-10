/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/sdl/sdl_audio_system.h"

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/nop/nop_audio_driver.h"
#include "xenia/apu/sdl/sdl_audio_driver.h"
#include "xenia/base/logging.h"

namespace xe {
namespace apu {
namespace sdl {

std::unique_ptr<AudioSystem> SDLAudioSystem::Create(cpu::Processor* processor) {
  return std::make_unique<SDLAudioSystem>(processor);
}

SDLAudioSystem::SDLAudioSystem(cpu::Processor* processor)
    : AudioSystem(processor) {}

SDLAudioSystem::~SDLAudioSystem() {}

void SDLAudioSystem::Initialize() { AudioSystem::Initialize(); }

X_STATUS SDLAudioSystem::CreateDriver(size_t index,
                                      xe::threading::Semaphore* semaphore,
                                      AudioDriver** out_driver) {
  assert_not_null(out_driver);
  auto driver = std::make_unique<SDLAudioDriver>(semaphore);
  if (driver->Initialize()) {
    *out_driver = driver.release();
    return X_STATUS_SUCCESS;
  }
  driver->Shutdown();
  // No host audio device (SDL_OpenAudioDevice failed): register a silent,
  // paced driver instead. A failed registration crashed Viva Pinata, which
  // dereferences the client it never got (Mac Pro, 2026-10-10).
  auto silent = std::make_unique<nop::NopAudioDriver>(semaphore);
  if (!silent->Initialize()) {
    silent->Shutdown();
    return X_STATUS_UNSUCCESSFUL;
  }
  XELOGW("SDLAudioSystem: no audio device for client {}; running silent",
         index);
  *out_driver = silent.release();
  return X_STATUS_SUCCESS;
}

AudioDriver* SDLAudioSystem::CreateDriver(xe::threading::Semaphore* semaphore,
                                          uint32_t frequency, uint32_t channels,
                                          bool need_format_conversion) {
  return new SDLAudioDriver(semaphore, frequency, channels,
                            need_format_conversion);
}

void SDLAudioSystem::DestroyDriver(AudioDriver* driver) {
  assert_not_null(driver);
  // The silent fallback (CreateDriver) is an AudioDriver of another type.
  driver->Shutdown();
  delete driver;
}

}  // namespace sdl
}  // namespace apu
}  // namespace xe
