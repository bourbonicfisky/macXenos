/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_driver.h"

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <utility>
#include <variant>

#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/ui/virtual_key.h"

namespace xe::hid::scripted {

namespace {

// Keystroke bit i reports kKeystrokeVirtualKeys[i]. Bits 0-15 are the XInput
// button field (bit 11 is unused and never set); 16-19 are the left stick's
// four directions and 20-21 the triggers, as the SDL driver derives them.
constexpr ui::VirtualKey kKeystrokeVirtualKeys[] = {
    ui::VirtualKey::kXInputPadDpadUp,      ui::VirtualKey::kXInputPadDpadDown,
    ui::VirtualKey::kXInputPadDpadLeft,    ui::VirtualKey::kXInputPadDpadRight,
    ui::VirtualKey::kXInputPadStart,       ui::VirtualKey::kXInputPadBack,
    ui::VirtualKey::kXInputPadLThumbPress, ui::VirtualKey::kXInputPadRThumbPress,
    ui::VirtualKey::kXInputPadLShoulder,   ui::VirtualKey::kXInputPadRShoulder,
    ui::VirtualKey::kXInputPadGuide,       ui::VirtualKey::kNone,
    ui::VirtualKey::kXInputPadA,           ui::VirtualKey::kXInputPadB,
    ui::VirtualKey::kXInputPadX,           ui::VirtualKey::kXInputPadY,
    ui::VirtualKey::kXInputPadLThumbUp,    ui::VirtualKey::kXInputPadLThumbDown,
    ui::VirtualKey::kXInputPadLThumbRight, ui::VirtualKey::kXInputPadLThumbLeft,
    ui::VirtualKey::kXInputPadLTrigger,    ui::VirtualKey::kXInputPadRTrigger,
};

uint32_t KeystrokeBits(const GamepadState& state) {
  // XInput's left-stick dead zone and trigger threshold.
  constexpr int32_t kStick = 7849;
  constexpr uint8_t kTrigger = 30;
  uint32_t bits = state.buttons & ~uint32_t(0x0800);
  bits |= state.left_stick_y > kStick ? uint32_t(1) << 16 : 0;
  bits |= state.left_stick_y < -kStick ? uint32_t(1) << 17 : 0;
  bits |= state.left_stick_x > kStick ? uint32_t(1) << 18 : 0;
  bits |= state.left_stick_x < -kStick ? uint32_t(1) << 19 : 0;
  bits |= state.left_trigger > kTrigger ? uint32_t(1) << 20 : 0;
  bits |= state.right_trigger > kTrigger ? uint32_t(1) << 21 : 0;
  return bits;
}

}  // namespace

namespace {

// Expanded active plus neutral transitions the script will produce. Waits,
// markers, and completion are not transitions and are excluded.
uint32_t CountExpandedTransitions(const InputScript& script) {
  uint64_t count = 0;
  for (const ScriptStep& step : script.steps()) {
    if (const auto* input = std::get_if<InputStep>(&step)) {
      count += static_cast<uint64_t>(input->repeat) * 2;
    }
  }
  return static_cast<uint32_t>(
      std::min<uint64_t>(count, kMaximumExpandedTransitions));
}

const char* TransitionKindName(TransitionKind kind) {
  switch (kind) {
    case TransitionKind::kActive:
      return "active";
    case TransitionKind::kNeutral:
      return "neutral";
    default:
      return "none";
  }
}

}  // namespace

ScriptedInputDriver::ScriptedInputDriver(
    xe::ui::Window* window, size_t window_z_order,
    std::shared_ptr<const InputScript> script,
    ScriptedInputTimeline::NowFunction now_function,
    ScriptedInputTimeline::EventCallback event_callback,
    bool transition_telemetry, std::filesystem::path live_file)
    : InputDriver(window, window_z_order),
      script_(std::move(script)),
      transition_telemetry_(transition_telemetry),
      live_file_(std::move(live_file)) {
  if (transition_telemetry_ && script_) {
    expected_transitions_ = CountExpandedTransitions(*script_);
  }
  if (!now_function) {
    now_function = &ScriptedInputDriver::SteadyNow;
  }
  now_function_ = now_function;
  if (!live_file_.empty()) {
    XELOGI(
        "scripted_input_live enabled file='{}' (discovery only; lines are "
        "consumed after the scripted timeline completes)",
        xe::path_to_utf8(live_file_));
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
  if (transition_telemetry_) {
    ++poll_ordinal_;
  }
  if (!timeline_->completed()) {
    CacheState(timeline_->Poll());
  } else if (!live_file_.empty()) {
    CacheState(PollLive(now_function_()));
  }
  if (transition_telemetry_) {
    LogTransitionObservation();
    if (timeline_->completed()) {
      LogTransitionSummary();
    }
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
  std::lock_guard<std::mutex> lock(mutex_);
  // Keystrokes never start the timeline (the first GetState does), but once
  // it runs they follow it: some menus read only keystrokes
  // (XamInputGetKeystrokeEx; Banjo-Kazooie: Nuts & Bolts).
  if (timeline_->started()) {
    if (!timeline_->completed()) {
      CacheState(timeline_->Poll());
    } else if (!live_file_.empty()) {
      CacheState(PollLive(now_function_()));
    }
    if (transition_telemetry_) {
      LogTransitionObservation();
    }
  }
  if (!out_keystroke) {
    return X_ERROR_EMPTY;
  }
  // One event per call: releases before presses, lowest bit first (the SDL
  // driver's order). No auto-repeat, so a timeline replays identically.
  const uint32_t current = KeystrokeBits(gamepad_state_);
  const uint32_t changed = current ^ keystroke_bits_;
  if (!changed) {
    return X_ERROR_EMPTY;
  }
  const uint32_t released = changed & ~current;
  const uint32_t pick = released ? released : changed;
  const uint32_t index = static_cast<uint32_t>(std::countr_zero(pick));
  const uint32_t bit = uint32_t(1) << index;
  keystroke_bits_ ^= bit;
  out_keystroke->virtual_key = uint16_t(kKeystrokeVirtualKeys[index]);
  out_keystroke->flags = (current & bit) ? X_INPUT_KEYSTROKE_KEYDOWN
                                         : X_INPUT_KEYSTROKE_KEYUP;
  return X_ERROR_SUCCESS;
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

uint32_t ScriptedInputDriver::transition_records_emitted() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return logged_transitions_;
}

uint32_t ScriptedInputDriver::completion_records_emitted() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return logged_summaries_;
}

void ScriptedInputDriver::LogTransitionObservation() {
  const uint32_t ordinal = timeline_->transition_ordinal();
  if (ordinal == last_transition_ordinal_) {
    return;
  }
  last_transition_ordinal_ = ordinal;
  ++observed_transitions_;
  // Fail closed: never emit more records than the script can produce.
  if (observed_transitions_ > expected_transitions_) {
    return;
  }
  ++logged_transitions_;
  XELOGI(
      "scripted_input_transition script_digest={:016X} ordinal={} "
      "expected_state_transitions={} "
      "kind={} state=B{:04X}:LT{:02X}:RT{:02X}:LX{}:LY{}:RX{}:RY{} poll={} "
      "packet={} elapsed_ms={}",
      script_->content_digest(), ordinal, expected_transitions_,
      TransitionKindName(timeline_->transition_kind()), gamepad_state_.buttons,
      gamepad_state_.left_trigger, gamepad_state_.right_trigger,
      gamepad_state_.left_stick_x, gamepad_state_.left_stick_y,
      gamepad_state_.right_stick_x, gamepad_state_.right_stick_y, poll_ordinal_,
      packet_number_, timeline_->elapsed().count());
}

void ScriptedInputDriver::LogTransitionSummary() {
  if (logged_summaries_ != 0) {
    return;
  }
  ++logged_summaries_;
  XELOGI(
      "scripted_input_transition_summary script_digest={:016X} "
      "expected_state_transitions={} observed_state_transitions={} poll={} "
      "packet={} elapsed_ms={}",
      script_->content_digest(), expected_transitions_, observed_transitions_,
      poll_ordinal_, packet_number_, timeline_->elapsed().count());
}

uint32_t ScriptedInputDriver::live_lines_applied() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return live_lines_applied_;
}

GamepadState ScriptedInputDriver::PollLive(
    ScriptedInputTimeline::Clock::time_point now) {
  if (live_timeline_ && !live_timeline_->completed()) {
    return live_timeline_->Poll();
  }
  // Between batches, look for appended lines at most every 100 ms.
  if (live_last_check_ != ScriptedInputTimeline::Clock::time_point{} &&
      now - live_last_check_ < std::chrono::milliseconds(100)) {
    return {};
  }
  live_last_check_ = now;
  IngestLiveLines(now);
  if (live_timeline_ && !live_timeline_->completed()) {
    return live_timeline_->Poll();
  }
  return {};
}

void ScriptedInputDriver::IngestLiveLines(
    ScriptedInputTimeline::Clock::time_point now) {
  std::error_code error_code;
  const auto size = std::filesystem::file_size(live_file_, error_code);
  if (error_code || size <= live_offset_) {
    return;
  }
  std::ifstream stream(live_file_, std::ios::binary);
  if (!stream) {
    return;
  }
  stream.seekg(static_cast<std::streamoff>(live_offset_));
  std::string appended(static_cast<size_t>(size - live_offset_), '\0');
  stream.read(appended.data(), static_cast<std::streamsize>(appended.size()));
  appended.resize(static_cast<size_t>(stream.gcount()));
  live_offset_ += appended.size();
  std::string buffer = live_partial_line_ + appended;
  live_partial_line_.clear();
  std::vector<ScriptStep> steps;
  uint64_t expanded = 0;
  uint64_t nominal_ms = 0;
  size_t start = 0;
  for (;;) {
    size_t newline = buffer.find('\n', start);
    if (newline == std::string::npos) {
      live_partial_line_ = buffer.substr(start);
      break;
    }
    std::string line = buffer.substr(start, newline - start);
    start = newline + 1;
    ++live_lines_seen_;
    LiveLineParseResult parsed = ParseLiveInputLine(line);
    if (!parsed.error.empty()) {
      XELOGE("scripted_input_live line={} rejected ({}): '{}'",
             live_lines_seen_, parsed.error, line);
      continue;
    }
    if (!parsed.step) {
      continue;
    }
    if (const auto* input = std::get_if<InputStep>(&*parsed.step)) {
      expanded += static_cast<uint64_t>(input->repeat) * 2;
      nominal_ms += static_cast<uint64_t>(
          (input->hold_duration + input->release_duration).count() *
          input->repeat);
    } else if (const auto* wait = std::get_if<WaitStep>(&*parsed.step)) {
      nominal_ms += static_cast<uint64_t>(wait->duration.count());
    }
    ++live_lines_applied_;
    XELOGI("scripted_input_live line={} applied: '{}'", live_lines_seen_,
           line);
    steps.push_back(std::move(*parsed.step));
  }
  if (steps.empty()) {
    return;
  }
  ++live_batches_;
  auto script = std::make_shared<const InputScript>(
      live_file_, static_cast<uint64_t>(live_batches_),
      "live batch " + std::to_string(live_batches_),
      script_->expected_title_id(), script_->player(), std::move(steps),
      static_cast<uint32_t>(
          std::min<uint64_t>(expanded, kMaximumExpandedTransitions)),
      std::chrono::milliseconds(nominal_ms));
  live_timeline_ = std::make_unique<ScriptedInputTimeline>(
      script, now_function_, [this](const TimelineEvent& event) {
        if (event.type == TimelineEventType::kMarker) {
          XELOGI("scripted_input_live marker '{}'", event.marker);
        }
      });
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
