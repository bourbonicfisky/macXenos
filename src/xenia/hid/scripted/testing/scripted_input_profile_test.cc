/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_script.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <string_view>
#include <variant>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/hid/scripted/scripted_input_timeline.h"

namespace xe::hid::scripted::test {
namespace {

using namespace std::chrono_literals;

struct ExpectedProfile {
  std::string_view filename;
  std::string_view title_id;
  uint32_t source_steps;
  uint32_t transitions;
  std::chrono::milliseconds duration;
};

constexpr std::array<ExpectedProfile, 4> kExpectedProfiles = {{
    {"53450848-sonic-generations.toml", "53450848", 5, 7, 21100ms},
    {"434307FC-ultra-street-fighter-iv.toml", "434307FC", 17, 29, 48000ms},
    {"534307E7-just-cause-2.toml", "534307E7", 3, 8, 16000ms},
    {"4C4107D1-lego-star-wars-ii.toml", "4C4107D1", 3, 4, 12800ms},
}};

}  // namespace

TEST_CASE("every committed scripted-input profile parses and ends neutral",
          "[hid][scripted][profiles]") {
  const std::filesystem::path profile_directory =
      std::filesystem::path(XENIA_SOURCE_DIR) / "tools" / "live_tests" /
      "input_profiles";

  for (const ExpectedProfile& expected : kExpectedProfiles) {
    DYNAMIC_SECTION(expected.filename) {
      const std::filesystem::path path = profile_directory / expected.filename;
      ScriptLoadResult result = LoadInputScript(path);

      INFO(result.error);
      REQUIRE(result);
      REQUIRE(result.script->canonical_path() ==
              std::filesystem::canonical(path));
      REQUIRE(result.script->expected_title_id() == expected.title_id);
      REQUIRE(result.script->player() == 0);
      REQUIRE(result.script->source_step_count() == expected.source_steps);
      REQUIRE(result.script->expanded_transition_count() ==
              expected.transitions);
      REQUIRE(result.script->expanded_transition_count() <=
              kMaximumExpandedTransitions);
      REQUIRE(result.script->nominal_duration() == expected.duration);
      REQUIRE(result.script->nominal_duration() <= kMaximumNominalDuration);

      const auto* final_marker =
          std::get_if<MarkerStep>(&result.script->steps().back());
      REQUIRE(final_marker != nullptr);
      REQUIRE(final_marker->name == "gameplay_expected");

      ScriptedInputTimeline::Clock::time_point now = {};
      std::vector<TimelineEvent> events;
      ScriptedInputTimeline timeline(
          result.script, [&now] { return now; },
          [&events](const TimelineEvent& event) { events.push_back(event); });

      GamepadState final_state;
      const size_t maximum_polls = result.script->expanded_transition_count() +
                                   result.script->source_step_count() * 2 + 4;
      for (size_t poll = 0; poll < maximum_polls && !timeline.completed();
           ++poll) {
        final_state = timeline.Poll();
        now += kMaximumNominalDuration + 1ms;
      }

      REQUIRE(timeline.completed());
      REQUIRE(final_state == GamepadState{});
      REQUIRE(timeline.Poll() == GamepadState{});
      REQUIRE(events.size() == 2);
      REQUIRE(events[0].type == TimelineEventType::kMarker);
      REQUIRE(events[0].marker == "gameplay_expected");
      REQUIRE(events[1].type == TimelineEventType::kCompletion);
    }
  }
}

}  // namespace xe::hid::scripted::test
