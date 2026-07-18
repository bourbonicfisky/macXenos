/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_timeline.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/hid/input.h"

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

std::shared_ptr<const InputScript> MakeScript(std::vector<ScriptStep> steps) {
  return std::make_shared<const InputScript>(
      "test.toml", 0x1234, "timeline test", "1234ABCD", 0, std::move(steps), 0,
      std::chrono::milliseconds::zero());
}

GamepadState ButtonState(uint16_t buttons) {
  GamepadState state;
  state.buttons = buttons;
  return state;
}

bool IsNeutral(const GamepadState& state) { return state == GamepadState{}; }

struct RunTrace {
  std::vector<GamepadState> states;
  std::vector<TimelineEvent> events;

  bool operator==(const RunTrace& other) const {
    if (states != other.states || events.size() != other.events.size()) {
      return false;
    }
    for (size_t i = 0; i < events.size(); ++i) {
      if (events[i].type != other.events[i].type ||
          events[i].marker != other.events[i].marker ||
          events[i].elapsed != other.events[i].elapsed) {
        return false;
      }
    }
    return true;
  }
};

RunTrace RunDeterministicTrace() {
  FakeClock clock;
  RunTrace trace;
  ScriptedInputTimeline timeline(
      MakeScript({MarkerStep{"begin"}, WaitStep{7ms},
                  InputStep{ButtonState(X_INPUT_GAMEPAD_A), 5ms, 3ms, 2},
                  MarkerStep{"end"}}),
      [&clock] { return clock.Now(); },
      [&trace](const TimelineEvent& event) { trace.events.push_back(event); });

  const std::vector<std::chrono::milliseconds> advances = {
      0ms, 6ms, 1ms, 4ms, 1ms, 2ms, 1ms, 4ms, 1ms, 2ms, 1ms, 1000ms};
  for (std::chrono::milliseconds advance : advances) {
    clock.Advance(advance);
    trace.states.push_back(timeline.Poll());
  }
  REQUIRE(timeline.completed());
  return trace;
}

}  // namespace

TEST_CASE("timeline anchors waits to the delayed first eligible poll",
          "[hid][scripted][timeline]") {
  FakeClock clock;
  clock.Advance(6h);
  ScriptedInputTimeline timeline(
      MakeScript({WaitStep{100ms}, InputStep{ButtonState(X_INPUT_GAMEPAD_START),
                                             20ms, 30ms, 1}}),
      [&clock] { return clock.Now(); });

  REQUIRE_FALSE(timeline.started());
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE(timeline.started());
  REQUIRE(timeline.elapsed() == 0ms);

  clock.Advance(99ms);
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE(timeline.elapsed() == 99ms);

  clock.Advance(1ms);
  REQUIRE(timeline.Poll() == ButtonState(X_INPUT_GAMEPAD_START));
  REQUIRE(timeline.elapsed() == 100ms);
}

TEST_CASE("timeline observes active and release states across long host pauses",
          "[hid][scripted][timeline]") {
  FakeClock clock;
  std::vector<TimelineEvent> events;
  ScriptedInputTimeline timeline(
      MakeScript({InputStep{ButtonState(X_INPUT_GAMEPAD_A), 100ms, 50ms, 1}}),
      [&clock] { return clock.Now(); },
      [&events](const TimelineEvent& event) { events.push_back(event); });

  // Entry always returns the active state and starts its hold at this poll.
  REQUIRE(timeline.Poll() == ButtonState(X_INPUT_GAMEPAD_A));

  // Even a gap far longer than the hold can only advance to an observable
  // neutral release. Its release timer starts at this poll.
  clock.Advance(1h);
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE_FALSE(timeline.completed());
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE_FALSE(timeline.completed());

  clock.Advance(49ms);
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE_FALSE(timeline.completed());

  clock.Advance(1ms);
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE(timeline.completed());
  REQUIRE(events.size() == 1);
  REQUIRE(events.front().type == TimelineEventType::kCompletion);
  REQUIRE(events.front().elapsed == 1h + 50ms);
}

TEST_CASE("timeline executes waits repetitions and markers in order",
          "[hid][scripted][timeline]") {
  FakeClock clock;
  std::vector<TimelineEvent> events;
  ScriptedInputTimeline timeline(
      MakeScript({MarkerStep{"before_wait"}, WaitStep{10ms},
                  InputStep{ButtonState(X_INPUT_GAMEPAD_B), 5ms, 7ms, 2},
                  MarkerStep{"after_input"}}),
      [&clock] { return clock.Now(); },
      [&events](const TimelineEvent& event) { events.push_back(event); });

  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE(events.size() == 1);
  REQUIRE(events[0].type == TimelineEventType::kMarker);
  REQUIRE(events[0].marker == "before_wait");
  REQUIRE(events[0].elapsed == 0ms);

  clock.Advance(10ms);
  REQUIRE(timeline.Poll() == ButtonState(X_INPUT_GAMEPAD_B));
  REQUIRE(timeline.Poll() == ButtonState(X_INPUT_GAMEPAD_B));

  clock.Advance(5ms);
  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE(IsNeutral(timeline.Poll()));

  clock.Advance(7ms);
  REQUIRE(timeline.Poll() == ButtonState(X_INPUT_GAMEPAD_B));
  clock.Advance(5ms);
  REQUIRE(IsNeutral(timeline.Poll()));
  clock.Advance(7ms);
  REQUIRE(IsNeutral(timeline.Poll()));

  REQUIRE(timeline.completed());
  REQUIRE(events.size() == 3);
  REQUIRE(events[1].type == TimelineEventType::kMarker);
  REQUIRE(events[1].marker == "after_input");
  REQUIRE(events[1].elapsed == 34ms);
  REQUIRE(events[2].type == TimelineEventType::kCompletion);
  REQUIRE(events[2].marker.empty());
  REQUIRE(events[2].elapsed == 34ms);
}

TEST_CASE("timeline completion remains permanently neutral and emits once",
          "[hid][scripted][timeline]") {
  FakeClock clock;
  std::vector<TimelineEvent> events;
  ScriptedInputTimeline timeline(
      MakeScript({MarkerStep{"last"}}), [&clock] { return clock.Now(); },
      [&events](const TimelineEvent& event) { events.push_back(event); });

  REQUIRE(IsNeutral(timeline.Poll()));
  REQUIRE(timeline.completed());
  REQUIRE(events.size() == 2);

  for (int i = 0; i < 10; ++i) {
    clock.Advance(24h);
    REQUIRE(IsNeutral(timeline.Poll()));
    REQUIRE(timeline.completed());
  }
  REQUIRE(events.size() == 2);
}

TEST_CASE("timeline produces identical traces on repeated fake-clock runs",
          "[hid][scripted][timeline]") {
  const RunTrace first = RunDeterministicTrace();
  const RunTrace second = RunDeterministicTrace();

  REQUIRE(first == second);
  REQUIRE_FALSE(first.states.empty());
  REQUIRE(first.events.size() == 3);
}

}  // namespace xe::hid::scripted::test
