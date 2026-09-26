/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_SCRIPTED_SCRIPTED_HID_H_
#define XENIA_HID_SCRIPTED_SCRIPTED_HID_H_

#include <filesystem>
#include <memory>

#include "xenia/hid/input_driver.h"
#include "xenia/hid/scripted/scripted_input_script.h"

namespace xe::hid::scripted {

// live_file (discovery only): see ParseLiveInputLine in
// scripted_input_script.h. Empty keeps the immutable-timeline behavior.
std::unique_ptr<InputDriver> Create(xe::ui::Window* window,
                                    size_t window_z_order,
                                    std::shared_ptr<const InputScript> script,
                                    std::filesystem::path live_file = {});

}  // namespace xe::hid::scripted

#endif  // XENIA_HID_SCRIPTED_SCRIPTED_HID_H_
