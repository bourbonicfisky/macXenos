/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_driver.h"

#include <limits>
#include <utility>

#include "xenia/base/logging.h"

namespace xe::hid::scripted {

ScriptedInputDriver::ScriptedInputDriver(
    xe::ui::Window* window, size_t window_z_order,
    std::shared_ptr<const InputScript> script,
    ScriptedInputTimeline::NowFunction now_function,
    ScriptedInputTimeline::EventCallback event_callback)
    : InputDriver(window, window_z_order), script_(std::move(script)) {
  if (!now_function) {
    now_function = &ScriptedInputDriver::SteadyNow;
  }
  if (!event_callback) {
    event_callback = [this](const TimelineEvent& event) {
      HandleTimelineEvent(event);
    };
  }
  timeline_ = std::make_unique<ScriptedInputTimeline>(
      script_, std::move(now_function), std::move(event_callback));
  CacheState({});
}

ScriptedInputDriver::~ScriptedInputDriver() = default;

ScriptedInputTimeline::Clock::time_point ScriptedInputDriver::SteadyNow() {
  return ScriptedInputTimeline::Clock::now();
}

X_STATUS ScriptedInputDriver::Setup() {
  return script_ ? X_STATUS_SUCCESS : X_STATUS_INVALID_PARAMETER;
}

X_RESULT ScriptedInputDriver::GetCapabilities(uint32_t user_index,
                                              uint32_t flags,
                                              X_INPUT_CAPABILITIES* out_caps) {
  if (user_index != script_->player()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  if (!out_caps) {
    return X_ERROR_INVALID_PARAMETER;
  }
  *out_caps = {};
  out_caps->type = XINPUT_DEVTYPE_GAMEPAD;
  out_caps->sub_type = XINPUT_DEVSUBTYPE_GAMEPAD;
  out_caps->flags = X_INPUT_CAPS_FFB_SUPPORTED;
  out_caps->gamepad.buttons =
      X_INPUT_GAMEPAD_DPAD_UP | X_INPUT_GAMEPAD_DPAD_DOWN |
      X_INPUT_GAMEPAD_DPAD_LEFT | X_INPUT_GAMEPAD_DPAD_RIGHT |
      X_INPUT_GAMEPAD_START | X_INPUT_GAMEPAD_BACK |
      X_INPUT_GAMEPAD_LEFT_THUMB | X_INPUT_GAMEPAD_RIGHT_THUMB |
      X_INPUT_GAMEPAD_LEFT_SHOULDER | X_INPUT_GAMEPAD_RIGHT_SHOULDER |
      X_INPUT_GAMEPAD_GUIDE | X_INPUT_GAMEPAD_A | X_INPUT_GAMEPAD_B |
      X_INPUT_GAMEPAD_X | X_INPUT_GAMEPAD_Y;
  out_caps->gamepad.left_trigger = std::numeric_limits<uint8_t>::max();
  out_caps->gamepad.right_trigger = std::numeric_limits<uint8_t>::max();
  out_caps->gamepad.thumb_lx = std::numeric_limits<int16_t>::max();
  out_caps->gamepad.thumb_ly = std::numeric_limits<int16_t>::max();
  out_caps->gamepad.thumb_rx = std::numeric_limits<int16_t>::max();
  out_caps->gamepad.thumb_ry = std::numeric_limits<int16_t>::max();
  out_caps->vibration.left_motor_speed = std::numeric_limits<uint16_t>::max();
  out_caps->vibration.right_motor_speed = std::numeric_limits<uint16_t>::max();
  return X_ERROR_SUCCESS;
}

void ScriptedInputDriver::CacheState(const GamepadState& state) {
  if (state != gamepad_state_) {
    gamepad_state_ = state;
    ++packet_number_;
  }
  cached_state_.packet_number = packet_number_;
  cached_state_.gamepad.buttons = state.buttons;
  cached_state_.gamepad.left_trigger = state.left_trigger;
  cached_state_.gamepad.right_trigger = state.right_trigger;
  cached_state_.gamepad.thumb_lx = state.left_stick_x;
  cached_state_.gamepad.thumb_ly = state.left_stick_y;
  cached_state_.gamepad.thumb_rx = state.right_stick_x;
  cached_state_.gamepad.thumb_ry = state.right_stick_y;
}

X_RESULT ScriptedInputDriver::GetState(uint32_t user_index,
                                       X_INPUT_STATE* out_state) {
  if (user_index != script_->player()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  if (!out_state) {
    return X_ERROR_INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!timeline_->completed()) {
    CacheState(timeline_->Poll());
  }
  *out_state = cached_state_;
  return X_ERROR_SUCCESS;
}

X_RESULT ScriptedInputDriver::GetStateSnapshot(uint32_t user_index,
                                               X_INPUT_STATE* out_state) {
  if (user_index != script_->player()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  if (!out_state) {
    return X_ERROR_INVALID_PARAMETER;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  *out_state = {};
  return X_ERROR_SUCCESS;
}

X_RESULT ScriptedInputDriver::SetState(uint32_t user_index,
                                       X_INPUT_VIBRATION* vibration) {
  return user_index == script_->player() ? X_ERROR_SUCCESS
                                         : X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT ScriptedInputDriver::GetKeystroke(uint32_t user_index, uint32_t flags,
                                           X_INPUT_KEYSTROKE* out_keystroke) {
  if (user_index != script_->player() && user_index != XUserIndexAny) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  if (out_keystroke) {
    *out_keystroke = {};
    out_keystroke->user_index = static_cast<uint8_t>(script_->player());
  }
  return X_ERROR_EMPTY;
}

InputType ScriptedInputDriver::GetInputType() const {
  return InputType::Controller;
}

bool ScriptedInputDriver::GetControllerIdentity(
    uint32_t user_index, ControllerIdentity* out_identity) {
  if (user_index != script_->player() || !out_identity) {
    return false;
  }
  *out_identity = {true, ControllerKind::kXbox, "Scripted Xbox 360 Controller"};
  return true;
}

bool ScriptedInputDriver::completed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return timeline_->completed();
}

void ScriptedInputDriver::HandleTimelineEvent(const TimelineEvent& event) {
  if (event.type == TimelineEventType::kMarker) {
    XELOGI("Scripted input marker '{}' at {} ms", event.marker,
           event.elapsed.count());
  } else {
    XELOGI("Scripted input completed at {} ms; controller remains neutral",
           event.elapsed.count());
  }
}

}  // namespace xe::hid::scripted
