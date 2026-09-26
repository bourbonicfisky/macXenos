/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_script.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>
#include <variant>

#include "third_party/catch/include/catch.hpp"
#include "xenia/hid/input.h"

namespace xe::hid::scripted::test {
namespace {

using namespace std::chrono_literals;

std::string MakeDocument(std::string_view steps, std::string_view header = R"(
version = 1
name = "test script"
expected_title_id = "1234ABCD"
player = 0
)") {
  return std::string(header) + std::string(steps);
}

void RequireFailureContaining(std::string_view document,
                              std::string_view expected_error) {
  ScriptLoadResult result = ParseInputScript(document);
  INFO(result.error);
  REQUIRE_FALSE(result);
  REQUIRE(result.script == nullptr);
  REQUIRE(result.error.find(expected_error) != std::string::npos);
}

}  // namespace

TEST_CASE("script parser accepts every step and gamepad control form",
          "[hid][scripted][parser]") {
  const std::filesystem::path source_path = "profiles/complete.toml";
  const std::string document = MakeDocument(R"(
[[step]]
wait_ms = 10

[[step]]
buttons = ["dpad_up", "dpad_down", "dpad_left", "dpad_right", "start",
           "back", "left_thumb", "right_thumb", "left_shoulder",
           "right_shoulder", "guide", "a", "b", "x", "y"]
left_trigger = 255
right_trigger = 1
left_stick_x = -32768
left_stick_y = 32767
right_stick_x = -12345
right_stick_y = 23456
hold_ms = 20
release_ms = 30
repeat = 2

[[step]]
marker = "gameplay_expected"
)");

  ScriptLoadResult result = ParseInputScript(document, source_path);

  INFO(result.error);
  REQUIRE(result);
  REQUIRE(result.error.empty());
  REQUIRE(result.script->version() == 1);
  REQUIRE(result.script->canonical_path() == source_path);
  REQUIRE(result.script->content_digest() != 0);
  REQUIRE(result.script->name() == "test script");
  REQUIRE(result.script->expected_title_id() == "1234ABCD");
  REQUIRE(result.script->player() == 0);
  REQUIRE(result.script->source_step_count() == 3);
  REQUIRE(result.script->expanded_transition_count() == 6);
  REQUIRE(result.script->nominal_duration() == 110ms);

  const auto* wait = std::get_if<WaitStep>(&result.script->steps()[0]);
  REQUIRE(wait != nullptr);
  REQUIRE(wait->duration == 10ms);

  const auto* input = std::get_if<InputStep>(&result.script->steps()[1]);
  REQUIRE(input != nullptr);
  REQUIRE(input->state.buttons ==
          (X_INPUT_GAMEPAD_DPAD_UP | X_INPUT_GAMEPAD_DPAD_DOWN |
           X_INPUT_GAMEPAD_DPAD_LEFT | X_INPUT_GAMEPAD_DPAD_RIGHT |
           X_INPUT_GAMEPAD_START | X_INPUT_GAMEPAD_BACK |
           X_INPUT_GAMEPAD_LEFT_THUMB | X_INPUT_GAMEPAD_RIGHT_THUMB |
           X_INPUT_GAMEPAD_LEFT_SHOULDER | X_INPUT_GAMEPAD_RIGHT_SHOULDER |
           X_INPUT_GAMEPAD_GUIDE | X_INPUT_GAMEPAD_A | X_INPUT_GAMEPAD_B |
           X_INPUT_GAMEPAD_X | X_INPUT_GAMEPAD_Y));
  REQUIRE(input->state.left_trigger == 255);
  REQUIRE(input->state.right_trigger == 1);
  REQUIRE(input->state.left_stick_x == -32768);
  REQUIRE(input->state.left_stick_y == 32767);
  REQUIRE(input->state.right_stick_x == -12345);
  REQUIRE(input->state.right_stick_y == 23456);
  REQUIRE(input->hold_duration == 20ms);
  REQUIRE(input->release_duration == 30ms);
  REQUIRE(input->repeat == 2);

  const auto* marker = std::get_if<MarkerStep>(&result.script->steps()[2]);
  REQUIRE(marker != nullptr);
  REQUIRE(marker->name == "gameplay_expected");
}

TEST_CASE("script parser supplies neutral omitted controls and repeat default",
          "[hid][scripted][parser]") {
  SECTION("analog fields are omitted") {
    ScriptLoadResult result = ParseInputScript(MakeDocument(R"(
[[step]]
buttons = ["a"]
hold_ms = 200
release_ms = 250
)"));

    INFO(result.error);
    REQUIRE(result);
    const auto& input = std::get<InputStep>(result.script->steps().front());
    REQUIRE(input.state.buttons == X_INPUT_GAMEPAD_A);
    REQUIRE(input.state.left_trigger == 0);
    REQUIRE(input.state.right_trigger == 0);
    REQUIRE(input.state.left_stick_x == 0);
    REQUIRE(input.state.left_stick_y == 0);
    REQUIRE(input.state.right_stick_x == 0);
    REQUIRE(input.state.right_stick_y == 0);
    REQUIRE(input.repeat == 1);
    REQUIRE(result.script->expanded_transition_count() == 2);
    REQUIRE(result.script->nominal_duration() == 450ms);
  }

  SECTION("every gamepad control is omitted") {
    ScriptLoadResult result = ParseInputScript(MakeDocument(R"(
[[step]]
hold_ms = 1
release_ms = 1
)"));
    INFO(result.error);
    REQUIRE(result);
    const auto& input = std::get<InputStep>(result.script->steps().front());
    REQUIRE(input.state == GamepadState{});
  }

  SECTION("highest player slot is valid") {
    ScriptLoadResult result =
        ParseInputScript(MakeDocument("[[step]]\nmarker = \"done\"\n", R"(
version = 1
name = "test script"
expected_title_id = "1234ABCD"
player = 3
)"));
    INFO(result.error);
    REQUIRE(result);
    REQUIRE(result.script->player() == 3);
  }
}

TEST_CASE("script loader rejects absent paths",
          "[hid][scripted][parser][negative]") {
  ScriptLoadResult empty = LoadInputScript({});
  REQUIRE_FALSE(empty);
  REQUIRE(empty.error.find("requires --input_script") != std::string::npos);

  const std::filesystem::path missing =
      std::filesystem::path(XENIA_SOURCE_DIR) / "tools" / "live_tests" /
      "input_profiles" / "missing-scripted-input-profile.toml";
  REQUIRE_FALSE(std::filesystem::exists(missing));
  ScriptLoadResult absent = LoadInputScript(missing);
  REQUIRE_FALSE(absent);
  REQUIRE(absent.error.find("cannot resolve input script") !=
          std::string::npos);
}

TEST_CASE("script digest is content-stable and content-sensitive",
          "[hid][scripted][parser]") {
  const std::string document = MakeDocument("[[step]]\nmarker = \"done\"\n");
  ScriptLoadResult first = ParseInputScript(document, "first.toml");
  ScriptLoadResult second = ParseInputScript(document, "second.toml");
  ScriptLoadResult changed =
      ParseInputScript(document + "# distinct content\n", "first.toml");

  REQUIRE(first);
  REQUIRE(second);
  REQUIRE(changed);
  REQUIRE(first.script->content_digest() == second.script->content_digest());
  REQUIRE(first.script->content_digest() != changed.script->content_digest());
}

TEST_CASE("script parser accepts exact resource limits",
          "[hid][scripted][parser]") {
  SECTION("maximum file size") {
    std::string document =
        MakeDocument("[[step]]\nmarker = \"size_boundary\"\n");
    document.resize(kMaximumInputFileSize, ' ');
    REQUIRE(document.size() == kMaximumInputFileSize);
    ScriptLoadResult result = ParseInputScript(document);
    INFO(result.error);
    REQUIRE(result);
  }
  SECTION("maximum nominal duration") {
    ScriptLoadResult result = ParseInputScript(MakeDocument(R"(
[[step]]
wait_ms = 1800000
)"));
    INFO(result.error);
    REQUIRE(result);
    REQUIRE(result.script->nominal_duration() == kMaximumNominalDuration);
  }

  SECTION("maximum expanded transitions") {
    ScriptLoadResult result = ParseInputScript(MakeDocument(R"(
[[step]]
buttons = ["a"]
hold_ms = 1
release_ms = 1
repeat = 2048
)"));
    INFO(result.error);
    REQUIRE(result);
    REQUIRE(result.script->expanded_transition_count() ==
            kMaximumExpandedTransitions);
  }
}

TEST_CASE("script parser rejects malformed and unsupported documents",
          "[hid][scripted][parser][negative]") {
  SECTION("malformed TOML") {
    RequireFailureContaining("version = [", "malformed TOML");
  }
  SECTION("unsupported version") {
    RequireFailureContaining(MakeDocument("[[step]]\nmarker = \"done\"\n", R"(
version = 2
name = "test script"
expected_title_id = "1234ABCD"
player = 0
)"),
                             "unsupported input script version 2");
  }
  SECTION("unknown top-level field") {
    RequireFailureContaining(
        MakeDocument("unknown = true\n[[step]]\nmarker = \"done\"\n"),
        "unknown field 'unknown'");
  }
  SECTION("unknown step field") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nmarker = \"done\"\nunknown = true\n"),
        "unknown field 'unknown'");
  }
  SECTION("missing step array") {
    RequireFailureContaining(MakeDocument(""), "nonempty array of tables");
  }
  SECTION("step is not a table") {
    RequireFailureContaining(MakeDocument("step = [1]\n"),
                             "step[0] must be a table");
  }
}

TEST_CASE("script parser rejects invalid metadata",
          "[hid][scripted][parser][negative]") {
  SECTION("missing required name") {
    RequireFailureContaining(R"(
version = 1
expected_title_id = "1234ABCD"
player = 0
[[step]]
marker = "done"
)",
                             "missing required field 'name'");
  }
  SECTION("empty name") {
    RequireFailureContaining(MakeDocument("[[step]]\nmarker = \"done\"\n", R"(
version = 1
name = ""
expected_title_id = "1234ABCD"
player = 0
)"),
                             "must contain 1 to 256 bytes");
  }
  SECTION("invalid title ID") {
    RequireFailureContaining(MakeDocument("[[step]]\nmarker = \"done\"\n", R"(
version = 1
name = "test"
expected_title_id = "not-a-title"
player = 0
)"),
                             "8 hexadecimal digits");
  }
  SECTION("player below range") {
    RequireFailureContaining(MakeDocument("[[step]]\nmarker = \"done\"\n", R"(
version = 1
name = "test"
expected_title_id = "1234ABCD"
player = -1
)"),
                             "player' must be in [0, 3]");
  }
  SECTION("player above range") {
    RequireFailureContaining(MakeDocument("[[step]]\nmarker = \"done\"\n", R"(
version = 1
name = "test"
expected_title_id = "1234ABCD"
player = 4
)"),
                             "player' must be in [0, 3]");
  }
}

TEST_CASE("script parser rejects ambiguous and invalid step kinds",
          "[hid][scripted][parser][negative]") {
  SECTION("step has no kind") {
    RequireFailureContaining(MakeDocument("[[step]]\n"),
                             "exactly one of wait, input, or marker");
  }
  SECTION("wait conflicts with marker") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nwait_ms = 1\nmarker = \"both\"\n"),
        "exactly one of wait, input, or marker");
  }
  SECTION("wait conflicts with input") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nwait_ms = 1\nhold_ms = 1\nrelease_ms = 1\n"),
        "exactly one of wait, input, or marker");
  }
  SECTION("zero wait") {
    RequireFailureContaining(MakeDocument("[[step]]\nwait_ms = 0\n"),
                             "wait_ms' must be in [1, 1800000]");
  }
  SECTION("input missing release") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nbuttons = [\"a\"]\nhold_ms = 1\n"),
        "missing required field 'release_ms'");
  }
  SECTION("nonpositive input timing") {
    RequireFailureContaining(
        MakeDocument(
            "[[step]]\nbuttons = [\"a\"]\nhold_ms = -1\nrelease_ms = 1\n"),
        "hold_ms' and 'release_ms' must each be in [1, 1800000]");
  }
  SECTION("wait above per-step timing bound") {
    RequireFailureContaining(MakeDocument("[[step]]\nwait_ms = 1800001\n"),
                             "wait_ms' must be in [1, 1800000]");
  }
  SECTION("hold above per-step timing bound") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nbuttons = [\"a\"]\nhold_ms = "
                     "1800001\nrelease_ms = 1\n"),
        "hold_ms' and 'release_ms' must each be in [1, 1800000]");
  }
  SECTION("empty marker") {
    RequireFailureContaining(MakeDocument("[[step]]\nmarker = \"\"\n"),
                             "must contain 1 to 256 bytes");
  }
}

TEST_CASE("script parser rejects invalid buttons and repetition",
          "[hid][scripted][parser][negative]") {
  SECTION("buttons is not an array") {
    RequireFailureContaining(
        MakeDocument(
            "[[step]]\nbuttons = \"a\"\nhold_ms = 1\nrelease_ms = 1\n"),
        "buttons' must be an array of strings");
  }
  SECTION("button is not a string") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nbuttons = [1]\nhold_ms = 1\nrelease_ms = 1\n"),
        "buttons' must contain only strings");
  }
  SECTION("unknown button") {
    RequireFailureContaining(
        MakeDocument(
            "[[step]]\nbuttons = [\"turbo\"]\nhold_ms = 1\nrelease_ms = 1\n"),
        "unknown button 'turbo'");
  }
  SECTION("duplicate button") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nbuttons = [\"a\", \"a\"]\nhold_ms = "
                     "1\nrelease_ms = 1\n"),
        "duplicate button 'a'");
  }
  SECTION("zero repeat") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nbuttons = [\"a\"]\nhold_ms = 1\nrelease_ms = "
                     "1\nrepeat = 0\n"),
        "repeat' must be in [1, 2048]");
  }
  SECTION("repeat above per-step bound") {
    RequireFailureContaining(
        MakeDocument("[[step]]\nbuttons = [\"a\"]\nhold_ms = 1\nrelease_ms = "
                     "1\nrepeat = 2049\n"),
        "repeat' must be in [1, 2048]");
  }
}

TEST_CASE("script parser rejects every numeric control outside its range",
          "[hid][scripted][parser][negative]") {
  const auto invalid_control = [](std::string_view assignment,
                                  std::string_view control) {
    const std::string step = "[[step]]\n" + std::string(assignment) +
                             "\nhold_ms = 1\nrelease_ms = 1\n";
    RequireFailureContaining(MakeDocument(step), control);
  };

  SECTION("left trigger below range") {
    invalid_control("left_trigger = -1", "left_trigger");
  }
  SECTION("left trigger above range") {
    invalid_control("left_trigger = 256", "left_trigger");
  }
  SECTION("right trigger below range") {
    invalid_control("right_trigger = -1", "right_trigger");
  }
  SECTION("right trigger above range") {
    invalid_control("right_trigger = 256", "right_trigger");
  }
  SECTION("left X below range") {
    invalid_control("left_stick_x = -32769", "left_stick_x");
  }
  SECTION("left X above range") {
    invalid_control("left_stick_x = 32768", "left_stick_x");
  }
  SECTION("left Y below range") {
    invalid_control("left_stick_y = -32769", "left_stick_y");
  }
  SECTION("left Y above range") {
    invalid_control("left_stick_y = 32768", "left_stick_y");
  }
  SECTION("right X below range") {
    invalid_control("right_stick_x = -32769", "right_stick_x");
  }
  SECTION("right X above range") {
    invalid_control("right_stick_x = 32768", "right_stick_x");
  }
  SECTION("right Y below range") {
    invalid_control("right_stick_y = -32769", "right_stick_y");
  }
  SECTION("right Y above range") {
    invalid_control("right_stick_y = 32768", "right_stick_y");
  }
}

TEST_CASE("script parser rejects inputs above global resource limits",
          "[hid][scripted][parser][negative]") {
  SECTION("file size") {
    RequireFailureContaining(std::string(kMaximumInputFileSize + 1, ' '),
                             "1 MiB file-size limit");
  }
  SECTION("nominal duration") {
    RequireFailureContaining(MakeDocument("[[step]]\nwait_ms = 1000000\n"
                                          "[[step]]\nwait_ms = 800001\n"),
                             "30 minute nominal-duration limit");
  }
  SECTION("expanded transitions across steps") {
    RequireFailureContaining(MakeDocument(R"(
[[step]]
buttons = ["a"]
hold_ms = 1
release_ms = 1
repeat = 2048
[[step]]
buttons = ["b"]
hold_ms = 1
release_ms = 1
)"),
                             "4096 expanded-transition limit");
  }
}

TEST_CASE("live input lines parse into single steps", "[hid][scripted][live]") {
  auto press = ParseLiveInputLine("press a");
  REQUIRE(press.error.empty());
  REQUIRE(press.step);
  const auto* input = std::get_if<InputStep>(&*press.step);
  REQUIRE(input);
  REQUIRE(input->state.buttons == X_INPUT_GAMEPAD_A);
  REQUIRE(input->hold_duration == 500ms);
  REQUIRE(input->release_duration == 500ms);
  REQUIRE(input->repeat == 1);

  auto combo = ParseLiveInputLine("  press start+b 400 9600 6\r");
  REQUIRE(combo.error.empty());
  input = std::get_if<InputStep>(&*combo.step);
  REQUIRE(input);
  REQUIRE(input->state.buttons ==
          (X_INPUT_GAMEPAD_START | X_INPUT_GAMEPAD_B));
  REQUIRE(input->hold_duration == 400ms);
  REQUIRE(input->release_duration == 9600ms);
  REQUIRE(input->repeat == 6);

  auto stick = ParseLiveInputLine("stick ly -32768 300");
  REQUIRE(stick.error.empty());
  input = std::get_if<InputStep>(&*stick.step);
  REQUIRE(input);
  REQUIRE(input->state.left_stick_y == -32768);
  REQUIRE(input->hold_duration == 300ms);

  auto diagonal = ParseLiveInputLine("stick lx -23000 ly -23000 250 750");
  REQUIRE(diagonal.error.empty());
  input = std::get_if<InputStep>(&*diagonal.step);
  REQUIRE(input);
  REQUIRE(input->state.left_stick_x == -23000);
  REQUIRE(input->state.left_stick_y == -23000);
  REQUIRE(input->hold_duration == 250ms);
  REQUIRE(input->release_duration == 750ms);

  auto trigger = ParseLiveInputLine("trigger rt 255");
  REQUIRE(trigger.error.empty());
  input = std::get_if<InputStep>(&*trigger.step);
  REQUIRE(input);
  REQUIRE(input->state.right_trigger == 255);

  auto wait = ParseLiveInputLine("wait 2500");
  REQUIRE(wait.error.empty());
  const auto* wait_step = std::get_if<WaitStep>(&*wait.step);
  REQUIRE(wait_step);
  REQUIRE(wait_step->duration == 2500ms);

  auto marker = ParseLiveInputLine("marker pause_menu");
  REQUIRE(marker.error.empty());
  const auto* marker_step = std::get_if<MarkerStep>(&*marker.step);
  REQUIRE(marker_step);
  REQUIRE(marker_step->name == "pause_menu");

  auto blank = ParseLiveInputLine("   ");
  REQUIRE(blank.error.empty());
  REQUIRE_FALSE(blank.step);
  auto comment = ParseLiveInputLine("# note");
  REQUIRE(comment.error.empty());
  REQUIRE_FALSE(comment.step);
}

TEST_CASE("live input lines reject malformed steps", "[hid][scripted][live]") {
  for (std::string_view line :
       {"press", "press nope", "press a+a", "press a 0", "press a 500 0",
        "press a 500 500 0", "press a 500 500 1 extra", "stick lz 1",
        "stick ly 40000", "stick lx 1 ly", "stick lx 1 lt 2", "trigger lt 256", "wait", "wait 0", "marker",
        "jump a"}) {
    auto result = ParseLiveInputLine(line);
    INFO(line);
    REQUIRE_FALSE(result.error.empty());
    REQUIRE_FALSE(result.step);
  }
}

}  // namespace xe::hid::scripted::test
