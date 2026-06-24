/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/helper/sdl/sdl_helper.h"

#include "xenia/base/platform.h"

// On linux we likely build on an "outdated" system but still want to control
// these features when available on a newer system.
#if !SDL_VERSION_ATLEAST(2, 0, 14)
#define SDL_HINT_AUDIO_DEVICE_APP_NAME "SDL_AUDIO_DEVICE_APP_NAME"
#endif

// HIDAPI hint names are stable strings; define fallbacks so building against an
// older SDL header still compiles (setting an unknown hint at runtime is a
// harmless no-op). Mirrors the SDL_HINT_AUDIO_DEVICE_APP_NAME fallback above.
#ifndef SDL_HINT_JOYSTICK_HIDAPI
#define SDL_HINT_JOYSTICK_HIDAPI "SDL_JOYSTICK_HIDAPI"
#endif
#ifndef SDL_HINT_JOYSTICK_HIDAPI_PS4
#define SDL_HINT_JOYSTICK_HIDAPI_PS4 "SDL_JOYSTICK_HIDAPI_PS4"
#endif
#ifndef SDL_HINT_JOYSTICK_HIDAPI_PS5
#define SDL_HINT_JOYSTICK_HIDAPI_PS5 "SDL_JOYSTICK_HIDAPI_PS5"
#endif

#include "xenia/base/assert.h"
#include "xenia/base/logging.h"

namespace xe {
namespace helper {
namespace sdl {
bool SDLHelper::is_prepared_ = false;

bool SDLHelper::Prepare() {
  if (is_prepared_) {
    return true;
  }
  is_prepared_ = true;

  is_prepared_ &= SetHints();
  is_prepared_ &= RedirectLog();

  SDL_version ver = {};
  SDL_GetVersion(&ver);
  XELOGI("SDL Version {}.{}.{} initialized.", ver.major, ver.minor, ver.patch);

  return is_prepared_;
}

bool SDLHelper::SetHints() {
  bool suc = true;

  const auto setHint = [](const char* name, const char* value,
                          bool override_ = false) -> bool {
    // Setting hints with normal priority fails when the hint is set via env
    // vars or a hint with override priority is set, which does not conclude a
    // failure.
    if (SDL_FALSE ==
        SDL_SetHintWithPriority(
            name, value, override_ ? SDL_HINT_OVERRIDE : SDL_HINT_NORMAL)) {
      const char* msg_fmt =
          "SDLHelper: Unable to set hint \"{}\" to value \"{}\".";
      if (override_) {
        XELOGE(msg_fmt, name, value);
        return false;
      } else {
        XELOGI(msg_fmt, name, value);
      }
    }
    return true;
  };

  // SDL calls timeBeginPeriod(1) but xenia sets this to a lower value before
  // using NtSetTimerResolution(). Having that value overwritten causes overall
  // fps drops. Use override priority as timer resolution should always be
  // managed by xenia. https://bugzilla.libsdl.org/show_bug.cgi?id=5104
  suc &= setHint(SDL_HINT_TIMER_RESOLUTION, "0", true);

  suc &= setHint(SDL_HINT_AUDIO_CATEGORY, "playback");

  suc &= setHint(SDL_HINT_AUDIO_DEVICE_APP_NAME, "xenia emulator");

#if XE_PLATFORM_MAC
  // macOS only: make Bluetooth/USB PlayStation pad handling deterministic
  // rather than dependent on SDL's per-build defaults. SDL still owns the
  // physical->XInput-layout mapping; these only select the HIDAPI backend so
  // DualShock 4 / DualSense expose the full button/rumble set. Left at normal
  // priority so an explicit user env override still wins.
  suc &= setHint(SDL_HINT_JOYSTICK_HIDAPI, "1");
  suc &= setHint(SDL_HINT_JOYSTICK_HIDAPI_PS4, "1");
  suc &= setHint(SDL_HINT_JOYSTICK_HIDAPI_PS5, "1");
#endif  // XE_PLATFORM_MAC

  return suc;
}

bool SDLHelper::RedirectLog() {
  // Redirect SDL_Log* output (internal library stuff) to our log system.
  SDL_LogSetOutputFunction(
      [](void* userdata, int category, SDL_LogPriority priority,
         const char* message) {
        const char* msg_fmt = "SDL: {}";
        switch (priority) {
          case SDL_LOG_PRIORITY_VERBOSE:
          case SDL_LOG_PRIORITY_DEBUG:
            XELOGD(msg_fmt, message);
            break;
          case SDL_LOG_PRIORITY_INFO:
            XELOGI(msg_fmt, message);
            break;
          case SDL_LOG_PRIORITY_WARN:
            XELOGW(msg_fmt, message);
            break;
          case SDL_LOG_PRIORITY_ERROR:
          case SDL_LOG_PRIORITY_CRITICAL:
            XELOGE(msg_fmt, message);
            break;
          default:
            XELOGI(msg_fmt, message);
            assert_always("SDL: Unknown log priority");
            break;
        }
      },
      nullptr);
  // SDL itself isn't that talkative. Additionally to this settings there are
  // hints that can be switched on to output additional internal logging
  // information.
  SDL_LogSetAllPriority(SDL_LogPriority::SDL_LOG_PRIORITY_VERBOSE);
  return true;
}
}  // namespace sdl
}  // namespace helper
}  // namespace xe
