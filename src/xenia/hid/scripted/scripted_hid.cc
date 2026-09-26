/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_hid.h"

#include <utility>

#include "xenia/base/cvar.h"
#include "xenia/hid/scripted/scripted_input_driver.h"

DEFINE_transient_bool(
    scripted_input_transition_telemetry, false,
    "Diagnostic only. Log one record when each expanded active or neutral "
    "scripted transition is first returned to a guest input poll, plus one "
    "completion summary. Used only with --hid=scripted.",
    "HID");

namespace xe::hid::scripted {

std::unique_ptr<InputDriver> Create(xe::ui::Window* window,
                                    size_t window_z_order,
                                    std::shared_ptr<const InputScript> script,
                                    std::filesystem::path live_file) {
  if (!script) {
    return nullptr;
  }
  return std::make_unique<ScriptedInputDriver>(
      window, window_z_order, std::move(script),
      ScriptedInputTimeline::NowFunction{},
      ScriptedInputTimeline::EventCallback{},
      cvars::scripted_input_transition_telemetry, std::move(live_file));
}

}  // namespace xe::hid::scripted
