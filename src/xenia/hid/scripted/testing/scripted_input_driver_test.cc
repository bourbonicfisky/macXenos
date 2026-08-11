/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_driver.h"

#include <chrono>
#include <memory>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/hid/input_system.h"

namespace xe::hid::scripted::test {
namespace {

using namespace std::chrono_literals;

class FakeClock {
 public:
  using Clock = ScriptedInputTimeline::Clock;

  Clock::time_point Now() const { return now_; }
  void Advance(std::chrono::milliseconds amount) { now_ += amount; }

 private:
  Clock::time_point now_ = Clock::time_point{};
};

std::shared_ptr<const InputScript> ParseDriverScript(uint32_t player = 1) {
  const std::string document =
      "version = 1\n"
      "name = \"driver integration\"\n"
      "expected_title_id = \"1234ABCD\"\n"
      "player = " +
      std::to_string(player) + R"(
[[step]]
buttons = ["start", "a"]
left_trigger = 17
right_trigger = 239
left_stick_x = 111
left_stick_y = -222
right_stick_x = 333
right_stick_y = -444
hold_ms = 200
release_ms = 250
)";
  ScriptLoadResult result = ParseInputScript(document);
  INFO(result.error);
  REQUIRE(result);
  return result.script;
}

// Marker, wait, a repeated button step, and a step whose active state is
// itself neutral, so active/neutral transitions cannot be inferred from the
// returned state or packet number alone.
std::shared_ptr<const InputScript> ParseTelemetryScript() {
  const std::string document = R"(version = 1
name = "transition telemetry"
expected_title_id = "1234ABCD"
player = 1

[[step]]
marker = "begin"

[[step]]
wait_ms = 40

[[step]]
buttons = ["a"]
hold_ms = 30
release_ms = 20
repeat = 2

[[step]]
hold_ms = 10
release_ms = 10

[[step]]
marker = "gameplay_expected"
)";
  ScriptLoadResult result = ParseInputScript(document);
  INFO(result.error);
  REQUIRE(result);
  return result.script;
}

constexpr uint32_t kTelemetryScriptTransitions = 6;

struct PollRecord {
  uint32_t packet_number;
  uint16_t buttons;
  uint8_t left_trigger;
  uint8_t right_trigger;
  int16_t thumb_lx;
  int16_t thumb_ly;
  int16_t thumb_rx;
  int16_t thumb_ry;

  bool operator==(const PollRecord&) const = default;
};

PollRecord Record(const X_INPUT_STATE& state) {
  return {state.packet_number,        state.gamepad.buttons,
          state.gamepad.left_trigger, state.gamepad.right_trigger,
          state.gamepad.thumb_lx,     state.gamepad.thumb_ly,
          state.gamepad.thumb_rx,     state.gamepad.thumb_ry};
}

struct TelemetryRun {
  std::vector<PollRecord> polls;
  uint32_t transition_records = 0;
  uint32_t completion_records = 0;
  bool completed = false;
};

// Polls the telemetry script on a fixed fake-clock schedule that is long
// enough to run it to completion and then keep polling afterwards.
TelemetryRun RunTelemetryScript(bool transition_telemetry,
                                size_t poll_count = 40) {
  FakeClock clock;
  ScriptedInputDriver driver(
      nullptr, 0, ParseTelemetryScript(), [&clock] { return clock.Now(); },
      ScriptedInputTimeline::EventCallback{}, transition_telemetry);
  TelemetryRun run;
  run.polls.reserve(poll_count);
  for (size_t i = 0; i < poll_count; ++i) {
    X_INPUT_STATE state = {};
    REQUIRE(driver.GetState(1, &state) == X_ERROR_SUCCESS);
    run.polls.push_back(Record(state));
    clock.Advance(5ms);
  }
  run.transition_records = driver.transition_records_emitted();
  run.completion_records = driver.completion_records_emitted();
  run.completed = driver.completed();
  return run;
}

class PhysicalStateProbe final : public InputDriver {
 public:
  PhysicalStateProbe() : InputDriver(nullptr, 0) {}

  X_STATUS Setup() override { return X_STATUS_SUCCESS; }

  X_RESULT GetCapabilities(uint32_t, uint32_t,
                           X_INPUT_CAPABILITIES* out_caps) override {
    ++capability_calls;
    *out_caps = {};
    out_caps->type = XINPUT_DEVTYPE_GAMEPAD;
    out_caps->sub_type = XINPUT_DEVSUBTYPE_GAMEPAD;
    return X_ERROR_SUCCESS;
  }

  X_RESULT GetState(uint32_t, X_INPUT_STATE* out_state) override {
    ++state_calls;
    *out_state = {};
    out_state->gamepad.buttons = X_INPUT_GAMEPAD_Y;
    out_state->gamepad.left_trigger = 255;
    return X_ERROR_SUCCESS;
  }

  X_RESULT SetState(uint32_t, X_INPUT_VIBRATION*) override {
    ++vibration_calls;
    return X_ERROR_SUCCESS;
  }

  X_RESULT GetKeystroke(uint32_t, uint32_t, X_INPUT_KEYSTROKE*) override {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  InputType GetInputType() const override { return InputType::Controller; }

  size_t capability_calls = 0;
  size_t state_calls = 0;
  size_t vibration_calls = 0;
};

}  // namespace

TEST_CASE("scripted driver exposes the standard connected gamepad contract",
          "[hid][scripted][driver]") {
  FakeClock clock;
  ScriptedInputDriver driver(nullptr, 0, ParseDriverScript(),
                             [&clock] { return clock.Now(); });

  REQUIRE(driver.Setup() == X_STATUS_SUCCESS);
  REQUIRE(driver.GetInputType() == InputType::Controller);
  REQUIRE_FALSE(driver.AppliesInputDeadzone());

  X_INPUT_CAPABILITIES capabilities = {};
  REQUIRE(driver.GetCapabilities(1, 0, &capabilities) == X_ERROR_SUCCESS);
  REQUIRE(capabilities.type == XINPUT_DEVTYPE_GAMEPAD);
  REQUIRE(capabilities.sub_type == XINPUT_DEVSUBTYPE_GAMEPAD);
  REQUIRE((capabilities.flags & X_INPUT_CAPS_FFB_SUPPORTED) != 0);
  REQUIRE(capabilities.gamepad.buttons == 0xF7FF);
  REQUIRE(capabilities.gamepad.left_trigger == 255);
  REQUIRE(capabilities.gamepad.right_trigger == 255);
  REQUIRE(capabilities.gamepad.thumb_lx == 32767);
  REQUIRE(capabilities.gamepad.thumb_ly == 32767);
  REQUIRE(capabilities.gamepad.thumb_rx == 32767);
  REQUIRE(capabilities.gamepad.thumb_ry == 32767);
  REQUIRE(capabilities.vibration.left_motor_speed == 65535);
  REQUIRE(capabilities.vibration.right_motor_speed == 65535);

  REQUIRE(driver.GetCapabilities(0, 0, &capabilities) ==
          X_ERROR_DEVICE_NOT_CONNECTED);
  REQUIRE(driver.GetCapabilities(1, 0, nullptr) == X_ERROR_INVALID_PARAMETER);

  X_INPUT_STATE state = {};
  REQUIRE(driver.GetState(0, &state) == X_ERROR_DEVICE_NOT_CONNECTED);
  REQUIRE(driver.GetState(1, nullptr) == X_ERROR_INVALID_PARAMETER);

  X_INPUT_VIBRATION vibration = {};
  REQUIRE(driver.SetState(1, &vibration) == X_ERROR_SUCCESS);
  REQUIRE(driver.SetState(0, &vibration) == X_ERROR_DEVICE_NOT_CONNECTED);

  X_INPUT_KEYSTROKE keystroke = {};
  REQUIRE(driver.GetKeystroke(1, 0, &keystroke) == X_ERROR_EMPTY);
  REQUIRE(keystroke.user_index == 1);
  REQUIRE(driver.GetKeystroke(XUserIndexAny, 0, &keystroke) == X_ERROR_EMPTY);
  REQUIRE(driver.GetKeystroke(0, 0, &keystroke) ==
          X_ERROR_DEVICE_NOT_CONNECTED);

  ControllerIdentity identity;
  REQUIRE(driver.GetControllerIdentity(1, &identity));
  REQUIRE(identity.present);
  REQUIRE(identity.kind == ControllerKind::kXbox);
  REQUIRE(identity.name == "Scripted Xbox 360 Controller");
  REQUIRE_FALSE(driver.GetControllerIdentity(0, &identity));
  REQUIRE_FALSE(driver.GetControllerIdentity(1, nullptr));
}

TEST_CASE("real InputSystem preserves exact scripted states and packets",
          "[hid][scripted][driver][integration]") {
  FakeClock clock;
  auto driver = std::make_unique<ScriptedInputDriver>(
      nullptr, 0, ParseDriverScript(), [&clock] { return clock.Now(); });
  ScriptedInputDriver* scripted_driver = driver.get();

  InputSystem input_system(nullptr);
  input_system.AddDriver(std::move(driver));

  auto physical_probe = std::make_unique<PhysicalStateProbe>();
  PhysicalStateProbe* physical_probe_ptr = physical_probe.get();
  input_system.AddDriver(std::move(physical_probe));

  X_INPUT_CAPABILITIES capabilities = {};
  REQUIRE(input_system.GetCapabilities(1, InputType::Controller,
                                       &capabilities) == X_ERROR_SUCCESS);
  REQUIRE(capabilities.type == XINPUT_DEVTYPE_GAMEPAD);
  REQUIRE(physical_probe_ptr->capability_calls == 0);

  X_INPUT_STATE state = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &state) ==
          X_ERROR_SUCCESS);
  REQUIRE(state.packet_number == 1);
  REQUIRE(state.gamepad.buttons == (X_INPUT_GAMEPAD_START | X_INPUT_GAMEPAD_A));
  REQUIRE(state.gamepad.left_trigger == 17);
  REQUIRE(state.gamepad.right_trigger == 239);
  REQUIRE(state.gamepad.thumb_lx == 111);
  REQUIRE(state.gamepad.thumb_ly == -222);
  REQUIRE(state.gamepad.thumb_rx == 333);
  REQUIRE(state.gamepad.thumb_ry == -444);
  REQUIRE(input_system.GetConnectedSlots().test(1));
  REQUIRE(input_system.GetLastUsedSlot() == 1);

  X_INPUT_STATE unchanged = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &unchanged) ==
          X_ERROR_SUCCESS);
  REQUIRE(unchanged.packet_number == 1);
  REQUIRE(unchanged.gamepad.buttons == state.gamepad.buttons);

  clock.Advance(200ms);
  X_INPUT_STATE released = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &released) ==
          X_ERROR_SUCCESS);
  REQUIRE(released.packet_number == 2);
  REQUIRE(released.gamepad.buttons == 0);
  REQUIRE(released.gamepad.left_trigger == 0);
  REQUIRE(released.gamepad.right_trigger == 0);
  REQUIRE(released.gamepad.thumb_lx == 0);
  REQUIRE(released.gamepad.thumb_ly == 0);
  REQUIRE(released.gamepad.thumb_rx == 0);
  REQUIRE(released.gamepad.thumb_ry == 0);

  X_INPUT_STATE same_release = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &same_release) ==
          X_ERROR_SUCCESS);
  REQUIRE(same_release.packet_number == 2);

  clock.Advance(250ms);
  X_INPUT_STATE completed = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &completed) ==
          X_ERROR_SUCCESS);
  REQUIRE(completed.packet_number == 2);
  REQUIRE(completed.gamepad.buttons == 0);
  REQUIRE(scripted_driver->completed());

  clock.Advance(24h);
  X_INPUT_STATE permanent_neutral = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &permanent_neutral) ==
          X_ERROR_SUCCESS);
  REQUIRE(permanent_neutral.packet_number == 2);
  REQUIRE(permanent_neutral.gamepad.buttons == 0);

  // The later physical source is never polled or merged for the configured
  // scripted slot, including after permanent-neutral completion.
  REQUIRE(physical_probe_ptr->state_calls == 0);

  X_INPUT_VIBRATION vibration = {};
  REQUIRE(input_system.SetState(1, &vibration) == X_ERROR_SUCCESS);
  REQUIRE(physical_probe_ptr->vibration_calls == 0);

  ControllerIdentity identity = input_system.GetControllerIdentity(1);
  REQUIRE(identity.present);
  REQUIRE(identity.kind == ControllerKind::kXbox);
  REQUIRE(identity.name == "Scripted Xbox 360 Controller");
}

TEST_CASE("host snapshots never anchor or advance a scripted timeline",
          "[hid][scripted][driver][integration]") {
  FakeClock clock;
  auto driver = std::make_unique<ScriptedInputDriver>(
      nullptr, 0, ParseDriverScript(), [&clock] { return clock.Now(); });
  ScriptedInputDriver* scripted_driver = driver.get();

  InputSystem input_system(nullptr);
  input_system.AddDriver(std::move(driver));

  X_INPUT_STATE snapshot = {};
  REQUIRE(input_system.GetStateSnapshot(1, InputType::Controller, &snapshot) ==
          X_ERROR_SUCCESS);
  REQUIRE(snapshot.packet_number == 0);
  REQUIRE(snapshot.gamepad.buttons == 0);

  clock.Advance(24h);
  REQUIRE(input_system.GetStateSnapshot(1, InputType::Controller, &snapshot) ==
          X_ERROR_SUCCESS);
  REQUIRE(snapshot.packet_number == 0);
  REQUIRE(snapshot.gamepad.buttons == 0);
  REQUIRE_FALSE(scripted_driver->completed());

  // The first guest-originated state poll anchors the timeline and exposes the
  // active state despite arbitrary prior host time and snapshot polling.
  X_INPUT_STATE active = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &active) ==
          X_ERROR_SUCCESS);
  REQUIRE(active.packet_number == 1);
  REQUIRE(active.gamepad.buttons ==
          (X_INPUT_GAMEPAD_START | X_INPUT_GAMEPAD_A));

  // Host inspection cannot satisfy the active hold or expose its release.
  clock.Advance(24h);
  REQUIRE(input_system.GetStateSnapshot(1, InputType::Controller, &snapshot) ==
          X_ERROR_SUCCESS);
  REQUIRE(snapshot.packet_number == 0);
  REQUIRE(snapshot.gamepad.buttons == 0);
  REQUIRE_FALSE(scripted_driver->completed());

  X_INPUT_STATE released = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &released) ==
          X_ERROR_SUCCESS);
  REQUIRE(released.packet_number == 2);
  REQUIRE(released.gamepad.buttons == 0);

  // Host inspection likewise cannot satisfy the neutral-release interval.
  clock.Advance(24h);
  REQUIRE(input_system.GetStateSnapshot(1, InputType::Controller, &snapshot) ==
          X_ERROR_SUCCESS);
  REQUIRE(snapshot.packet_number == 0);
  REQUIRE(snapshot.gamepad.buttons == 0);
  REQUIRE_FALSE(scripted_driver->completed());

  X_INPUT_STATE completed = {};
  REQUIRE(input_system.GetState(1, InputType::Controller, &completed) ==
          X_ERROR_SUCCESS);
  REQUIRE(completed.packet_number == 2);
  REQUIRE(completed.gamepad.buttons == 0);
  REQUIRE(scripted_driver->completed());
}

TEST_CASE("real InputSystem reports non-scripted slots disconnected",
          "[hid][scripted][driver][integration]") {
  FakeClock clock;
  InputSystem input_system(nullptr);
  input_system.AddDriver(std::make_unique<ScriptedInputDriver>(
      nullptr, 0, ParseDriverScript(2), [&clock] { return clock.Now(); }));

  for (uint32_t slot : {0u, 1u, 3u}) {
    X_INPUT_CAPABILITIES capabilities = {};
    X_INPUT_STATE state = {};
    X_INPUT_VIBRATION vibration = {};
    X_INPUT_KEYSTROKE keystroke = {};
    REQUIRE(input_system.GetCapabilities(slot, InputType::Controller,
                                         &capabilities) ==
            X_ERROR_DEVICE_NOT_CONNECTED);
    REQUIRE(input_system.GetState(slot, InputType::Controller, &state) ==
            X_ERROR_DEVICE_NOT_CONNECTED);
    REQUIRE(input_system.SetState(slot, &vibration) ==
            X_ERROR_DEVICE_NOT_CONNECTED);
    REQUIRE(
        input_system.GetKeystroke(slot, InputType::Controller, &keystroke) ==
        X_ERROR_DEVICE_NOT_CONNECTED);
    REQUIRE_FALSE(input_system.GetConnectedSlots().test(slot));
  }

  // Ineligible slot queries and arbitrary host time do not start or consume
  // the script. The first configured-slot state poll still enters active.
  clock.Advance(24h);
  X_INPUT_STATE first_eligible_state = {};
  REQUIRE(input_system.GetState(2, InputType::Controller,
                                &first_eligible_state) == X_ERROR_SUCCESS);
  REQUIRE(first_eligible_state.packet_number == 1);
  REQUIRE(first_eligible_state.gamepad.buttons ==
          (X_INPUT_GAMEPAD_START | X_INPUT_GAMEPAD_A));

  X_INPUT_KEYSTROKE keystroke = {};
  REQUIRE(input_system.GetKeystroke(2, InputType::Controller, &keystroke) ==
          X_ERROR_EMPTY);
  REQUIRE(keystroke.user_index == 2);
}

TEST_CASE("scripted transition telemetry is silent unless it is selected",
          "[hid][scripted][driver][telemetry]") {
  const TelemetryRun off = RunTelemetryScript(false);
  REQUIRE(off.completed);
  REQUIRE(off.transition_records == 0);
  REQUIRE(off.completion_records == 0);
}

TEST_CASE("scripted transition telemetry preserves states and packets",
          "[hid][scripted][driver][telemetry]") {
  const TelemetryRun off = RunTelemetryScript(false);
  const TelemetryRun on = RunTelemetryScript(true);

  REQUIRE(on.polls.size() == off.polls.size());
  REQUIRE(on.polls == off.polls);
  REQUIRE(on.completed);
  REQUIRE(off.completed);

  // The script really did drive an active state and a packet sequence.
  REQUIRE(on.polls.front().packet_number == 0);
  REQUIRE(on.polls.back().packet_number == 4);
  bool saw_active = false;
  for (const PollRecord& poll : on.polls) {
    saw_active = saw_active || poll.buttons == X_INPUT_GAMEPAD_A;
  }
  REQUIRE(saw_active);

  // Every expanded active and neutral transition is reported exactly once,
  // including the pair whose active state is indistinguishable from neutral,
  // and the completion summary is reported exactly once.
  REQUIRE(on.transition_records == kTelemetryScriptTransitions);
  REQUIRE(on.completion_records == 1);
}

TEST_CASE("scripted transition telemetry stays bounded after completion",
          "[hid][scripted][driver][telemetry]") {
  const TelemetryRun on = RunTelemetryScript(true, 400);
  REQUIRE(on.completed);
  REQUIRE(on.transition_records == kTelemetryScriptTransitions);
  REQUIRE(on.completion_records == 1);
}

TEST_CASE("scripted transition telemetry reports partial observation",
          "[hid][scripted][driver][telemetry]") {
  // A guest that stops polling before the script ends observes fewer
  // transitions and never reaches the completion summary.
  const TelemetryRun on = RunTelemetryScript(true, 16);
  REQUIRE_FALSE(on.completed);
  REQUIRE(on.transition_records < kTelemetryScriptTransitions);
  REQUIRE(on.completion_records == 0);
}

}  // namespace xe::hid::scripted::test
