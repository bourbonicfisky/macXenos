/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_script.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

#include "third_party/tomlplusplus/include/toml++/toml.hpp"
#include "xenia/base/filesystem.h"
#include "xenia/base/xxhash.h"
#include "xenia/hid/input.h"

namespace xe::hid::scripted {
namespace {

constexpr uint32_t kMaximumPlayer = 3;
constexpr uint32_t kMaximumStringLength = 256;

struct NamedButton {
  std::string_view name;
  uint16_t value;
};

constexpr std::array<NamedButton, 15> kNamedButtons = {{
    {"dpad_up", X_INPUT_GAMEPAD_DPAD_UP},
    {"dpad_down", X_INPUT_GAMEPAD_DPAD_DOWN},
    {"dpad_left", X_INPUT_GAMEPAD_DPAD_LEFT},
    {"dpad_right", X_INPUT_GAMEPAD_DPAD_RIGHT},
    {"start", X_INPUT_GAMEPAD_START},
    {"back", X_INPUT_GAMEPAD_BACK},
    {"left_thumb", X_INPUT_GAMEPAD_LEFT_THUMB},
    {"right_thumb", X_INPUT_GAMEPAD_RIGHT_THUMB},
    {"left_shoulder", X_INPUT_GAMEPAD_LEFT_SHOULDER},
    {"right_shoulder", X_INPUT_GAMEPAD_RIGHT_SHOULDER},
    {"guide", X_INPUT_GAMEPAD_GUIDE},
    {"a", X_INPUT_GAMEPAD_A},
    {"b", X_INPUT_GAMEPAD_B},
    {"x", X_INPUT_GAMEPAD_X},
    {"y", X_INPUT_GAMEPAD_Y},
}};

ScriptLoadResult Failure(std::string error) {
  return {.script = nullptr, .error = std::move(error)};
}

bool IsAllowedKey(std::string_view key,
                  std::initializer_list<std::string_view> allowed) {
  return std::find(allowed.begin(), allowed.end(), key) != allowed.end();
}

bool ValidateKeys(const toml::table& table,
                  std::initializer_list<std::string_view> allowed,
                  std::string_view context, std::string* error) {
  for (const auto& [key, value] : table) {
    if (!IsAllowedKey(key.str(), allowed)) {
      *error = std::string(context) + " contains unknown field '" +
               std::string(key.str()) + "'";
      return false;
    }
  }
  return true;
}

std::optional<int64_t> RequiredInteger(const toml::table& table,
                                       std::string_view key,
                                       std::string_view context,
                                       std::string* error) {
  const toml::node* node = table.get(key);
  if (!node) {
    *error = std::string(context) + " is missing required field '" +
             std::string(key) + "'";
    return std::nullopt;
  }
  auto value = node->value<int64_t>();
  if (!value) {
    *error = std::string(context) + " field '" + std::string(key) +
             "' must be an integer";
  }
  return value;
}

std::optional<std::string> RequiredString(const toml::table& table,
                                          std::string_view key,
                                          std::string_view context,
                                          std::string* error) {
  const toml::node* node = table.get(key);
  if (!node) {
    *error = std::string(context) + " is missing required field '" +
             std::string(key) + "'";
    return std::nullopt;
  }
  auto value = node->value<std::string>();
  if (!value) {
    *error = std::string(context) + " field '" + std::string(key) +
             "' must be a string";
    return std::nullopt;
  }
  if (value->empty() || value->size() > kMaximumStringLength) {
    *error = std::string(context) + " field '" + std::string(key) +
             "' must contain 1 to " + std::to_string(kMaximumStringLength) +
             " bytes";
    return std::nullopt;
  }
  return value;
}

bool IsExpectedTitleId(std::string_view value) {
  return value.size() == 8 &&
         std::all_of(value.begin(), value.end(),
                     [](unsigned char c) { return std::isxdigit(c) != 0; });
}

std::optional<uint16_t> ButtonValue(std::string_view name) {
  auto it = std::find_if(
      kNamedButtons.begin(), kNamedButtons.end(),
      [name](const NamedButton& button) { return button.name == name; });
  if (it == kNamedButtons.end()) {
    return std::nullopt;
  }
  return it->value;
}

bool ReadOptionalControl(const toml::table& table, std::string_view key,
                         int64_t minimum, int64_t maximum, int64_t* output,
                         std::string_view context, std::string* error) {
  const toml::node* node = table.get(key);
  if (!node) {
    return true;
  }
  auto value = node->value<int64_t>();
  if (!value || *value < minimum || *value > maximum) {
    *error = std::string(context) + " field '" + std::string(key) +
             "' must be an integer in [" + std::to_string(minimum) + ", " +
             std::to_string(maximum) + "]";
    return false;
  }
  *output = *value;
  return true;
}

}  // namespace

InputScript::InputScript(std::filesystem::path canonical_path,
                         uint64_t content_digest, std::string name,
                         std::string expected_title_id, uint32_t player,
                         std::vector<ScriptStep> steps,
                         uint32_t expanded_transition_count,
                         std::chrono::milliseconds nominal_duration)
    : canonical_path_(std::move(canonical_path)),
      content_digest_(content_digest),
      name_(std::move(name)),
      expected_title_id_(std::move(expected_title_id)),
      player_(player),
      steps_(std::move(steps)),
      expanded_transition_count_(expanded_transition_count),
      nominal_duration_(nominal_duration) {}

ScriptLoadResult ParseInputScript(std::string_view content,
                                  std::filesystem::path source_path) {
  if (content.size() > kMaximumInputFileSize) {
    return Failure("input script exceeds the 1 MiB file-size limit");
  }

  toml::table document;
  try {
    document = toml::parse(content, xe::path_to_utf8(source_path));
  } catch (const toml::parse_error& error) {
    return Failure("malformed TOML input script: " +
                   std::string(error.description()));
  }

  std::string error;
  if (!ValidateKeys(document,
                    {"version", "name", "expected_title_id", "player", "step"},
                    "input script", &error)) {
    return Failure(std::move(error));
  }

  auto version = RequiredInteger(document, "version", "input script", &error);
  if (!version) {
    return Failure(std::move(error));
  }
  if (*version != kScriptVersion) {
    return Failure("unsupported input script version " +
                   std::to_string(*version) + "; expected version 1");
  }
  auto name = RequiredString(document, "name", "input script", &error);
  if (!name) {
    return Failure(std::move(error));
  }
  auto expected_title_id =
      RequiredString(document, "expected_title_id", "input script", &error);
  if (!expected_title_id) {
    return Failure(std::move(error));
  }
  if (!IsExpectedTitleId(*expected_title_id)) {
    return Failure(
        "input script field 'expected_title_id' must be 8 hexadecimal digits");
  }
  auto player = RequiredInteger(document, "player", "input script", &error);
  if (!player) {
    return Failure(std::move(error));
  }
  if (*player < 0 || *player > kMaximumPlayer) {
    return Failure("input script field 'player' must be in [0, 3]");
  }

  const toml::array* step_array = document["step"].as_array();
  if (!step_array || step_array->empty()) {
    return Failure(
        "input script field 'step' must be a nonempty array of tables");
  }

  std::vector<ScriptStep> steps;
  steps.reserve(step_array->size());
  uint64_t expanded_transitions = 0;
  uint64_t nominal_duration_ms = 0;

  for (size_t index = 0; index < step_array->size(); ++index) {
    const toml::table* table = (*step_array)[index].as_table();
    const std::string context = "step[" + std::to_string(index) + "]";
    if (!table) {
      return Failure(context + " must be a table");
    }
    if (!ValidateKeys(
            *table,
            {"wait_ms", "marker", "buttons", "left_trigger", "right_trigger",
             "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y",
             "hold_ms", "release_ms", "repeat"},
            context, &error)) {
      return Failure(std::move(error));
    }

    const bool is_wait = table->contains("wait_ms");
    const bool is_marker = table->contains("marker");
    const bool is_input =
        table->contains("buttons") || table->contains("left_trigger") ||
        table->contains("right_trigger") || table->contains("left_stick_x") ||
        table->contains("left_stick_y") || table->contains("right_stick_x") ||
        table->contains("right_stick_y") || table->contains("hold_ms") ||
        table->contains("release_ms") || table->contains("repeat");
    if (static_cast<unsigned>(is_wait) + static_cast<unsigned>(is_marker) +
            static_cast<unsigned>(is_input) !=
        1) {
      return Failure(
          context +
          " must contain exactly one of wait, input, or marker fields");
    }

    if (is_wait) {
      auto wait_ms = RequiredInteger(*table, "wait_ms", context, &error);
      if (!wait_ms) {
        return Failure(std::move(error));
      }
      if (*wait_ms <= 0 || *wait_ms > kMaximumNominalDuration.count()) {
        return Failure(context + " field 'wait_ms' must be in [1, 1800000]");
      }
      ++expanded_transitions;
      nominal_duration_ms += static_cast<uint64_t>(*wait_ms);
      steps.emplace_back(WaitStep{std::chrono::milliseconds(*wait_ms)});
    } else if (is_marker) {
      auto marker = RequiredString(*table, "marker", context, &error);
      if (!marker) {
        return Failure(std::move(error));
      }
      ++expanded_transitions;
      steps.emplace_back(MarkerStep{std::move(*marker)});
    } else {
      auto hold_ms = RequiredInteger(*table, "hold_ms", context, &error);
      auto release_ms = RequiredInteger(*table, "release_ms", context, &error);
      if (!hold_ms || !release_ms) {
        return Failure(std::move(error));
      }
      if (*hold_ms <= 0 || *release_ms <= 0 ||
          *hold_ms > kMaximumNominalDuration.count() ||
          *release_ms > kMaximumNominalDuration.count()) {
        return Failure(context +
                       " fields 'hold_ms' and 'release_ms' must each be in "
                       "[1, 1800000]");
      }
      int64_t repeat = 1;
      if (table->contains("repeat")) {
        auto parsed_repeat = RequiredInteger(*table, "repeat", context, &error);
        if (!parsed_repeat) {
          return Failure(std::move(error));
        }
        repeat = *parsed_repeat;
      }
      if (repeat <= 0 || repeat > kMaximumExpandedTransitions / 2) {
        return Failure(context + " field 'repeat' must be in [1, 2048]");
      }

      GamepadState state;
      if (const toml::node* buttons_node = table->get("buttons")) {
        const toml::array* buttons = buttons_node->as_array();
        if (!buttons) {
          return Failure(context +
                         " field 'buttons' must be an array of strings");
        }
        for (size_t button_index = 0; button_index < buttons->size();
             ++button_index) {
          auto button_name = (*buttons)[button_index].value<std::string>();
          if (!button_name) {
            return Failure(context +
                           " field 'buttons' must contain only strings");
          }
          auto value = ButtonValue(*button_name);
          if (!value) {
            return Failure(context + " contains unknown button '" +
                           *button_name + "'");
          }
          if ((state.buttons & *value) != 0) {
            return Failure(context + " contains duplicate button '" +
                           *button_name + "'");
          }
          state.buttons |= *value;
        }
      }
      int64_t left_trigger = 0;
      int64_t right_trigger = 0;
      int64_t left_stick_x = 0;
      int64_t left_stick_y = 0;
      int64_t right_stick_x = 0;
      int64_t right_stick_y = 0;
      if (!ReadOptionalControl(*table, "left_trigger", 0, 255, &left_trigger,
                               context, &error) ||
          !ReadOptionalControl(*table, "right_trigger", 0, 255, &right_trigger,
                               context, &error) ||
          !ReadOptionalControl(*table, "left_stick_x", -32768, 32767,
                               &left_stick_x, context, &error) ||
          !ReadOptionalControl(*table, "left_stick_y", -32768, 32767,
                               &left_stick_y, context, &error) ||
          !ReadOptionalControl(*table, "right_stick_x", -32768, 32767,
                               &right_stick_x, context, &error) ||
          !ReadOptionalControl(*table, "right_stick_y", -32768, 32767,
                               &right_stick_y, context, &error)) {
        return Failure(std::move(error));
      }
      state.left_trigger = static_cast<uint8_t>(left_trigger);
      state.right_trigger = static_cast<uint8_t>(right_trigger);
      state.left_stick_x = static_cast<int16_t>(left_stick_x);
      state.left_stick_y = static_cast<int16_t>(left_stick_y);
      state.right_stick_x = static_cast<int16_t>(right_stick_x);
      state.right_stick_y = static_cast<int16_t>(right_stick_y);

      expanded_transitions += static_cast<uint64_t>(repeat) * 2;
      nominal_duration_ms +=
          static_cast<uint64_t>(*hold_ms + *release_ms) * repeat;
      steps.emplace_back(InputStep{state, std::chrono::milliseconds(*hold_ms),
                                   std::chrono::milliseconds(*release_ms),
                                   static_cast<uint32_t>(repeat)});
    }

    if (expanded_transitions > kMaximumExpandedTransitions) {
      return Failure("input script exceeds the 4096 expanded-transition limit");
    }
    if (nominal_duration_ms >
        static_cast<uint64_t>(kMaximumNominalDuration.count())) {
      return Failure(
          "input script exceeds the 30 minute nominal-duration limit");
    }
  }

  const uint64_t digest = XXH3_64bits(content.data(), content.size());
  auto script = std::make_shared<const InputScript>(
      std::move(source_path), digest, std::move(*name),
      std::move(*expected_title_id), static_cast<uint32_t>(*player),
      std::move(steps), static_cast<uint32_t>(expanded_transitions),
      std::chrono::milliseconds(nominal_duration_ms));
  return {.script = std::move(script), .error = {}};
}

ScriptLoadResult LoadInputScript(const std::filesystem::path& path) {
  if (path.empty()) {
    return Failure("--hid=scripted requires --input_script=<path>");
  }
  std::error_code error_code;
  std::filesystem::path canonical_path =
      std::filesystem::canonical(path, error_code);
  if (error_code) {
    return Failure("cannot resolve input script '" + xe::path_to_utf8(path) +
                   "': " + error_code.message());
  }
  if (!std::filesystem::is_regular_file(canonical_path, error_code) ||
      error_code) {
    return Failure("input script is not a readable regular file: " +
                   xe::path_to_utf8(canonical_path));
  }
  uintmax_t file_size = std::filesystem::file_size(canonical_path, error_code);
  if (error_code) {
    return Failure("cannot read input script size: " + error_code.message());
  }
  if (file_size > kMaximumInputFileSize) {
    return Failure("input script exceeds the 1 MiB file-size limit");
  }

  std::ifstream stream(canonical_path, std::ios::binary);
  if (!stream) {
    return Failure("cannot open input script for reading: " +
                   xe::path_to_utf8(canonical_path));
  }
  std::string content(static_cast<size_t>(file_size), '\0');
  stream.read(content.data(), static_cast<std::streamsize>(content.size()));
  if (!stream ||
      stream.gcount() != static_cast<std::streamsize>(content.size())) {
    return Failure("failed while reading input script: " +
                   xe::path_to_utf8(canonical_path));
  }
  return ParseInputScript(content, std::move(canonical_path));
}

}  // namespace xe::hid::scripted
