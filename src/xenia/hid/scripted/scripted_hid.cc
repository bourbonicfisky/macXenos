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

#include "xenia/hid/scripted/scripted_input_driver.h"

namespace xe::hid::scripted {

std::unique_ptr<InputDriver> Create(xe::ui::Window* window,
                                    size_t window_z_order,
                                    std::shared_ptr<const InputScript> script) {
  if (!script) {
    return nullptr;
  }
  return std::make_unique<ScriptedInputDriver>(window, window_z_order,
                                               std::move(script));
}

}  // namespace xe::hid::scripted
