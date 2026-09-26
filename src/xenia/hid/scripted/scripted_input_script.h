/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_SCRIPTED_SCRIPTED_INPUT_SCRIPT_H_
#define XENIA_HID_SCRIPTED_SCRIPTED_INPUT_SCRIPT_H_

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace xe::hid::scripted {

constexpr uint32_t kScriptVersion = 1;
constexpr uint64_t kMaximumInputFileSize = 1024 * 1024;
constexpr uint32_t kMaximumExpandedTransitions = 4096;
constexpr std::chrono::milliseconds kMaximumNominalDuration =
    std::chrono::minutes(30);

struct GamepadState {
  uint16_t buttons = 0;
  uint8_t left_trigger = 0;
  uint8_t right_trigger = 0;
  int16_t left_stick_x = 0;
  int16_t left_stick_y = 0;
  int16_t right_stick_x = 0;
  int16_t right_stick_y = 0;

  bool operator==(const GamepadState&) const = default;
};

struct WaitStep {
  std::chrono::milliseconds duration;
};

struct InputStep {
  GamepadState state;
  std::chrono::milliseconds hold_duration;
  std::chrono::milliseconds release_duration;
  uint32_t repeat = 1;
};

struct MarkerStep {
  std::string name;
};

using ScriptStep = std::variant<WaitStep, InputStep, MarkerStep>;

class InputScript final {
 public:
  InputScript(std::filesystem::path canonical_path, uint64_t content_digest,
              std::string name, std::string expected_title_id, uint32_t player,
              std::vector<ScriptStep> steps, uint32_t expanded_transition_count,
              std::chrono::milliseconds nominal_duration);

  uint32_t version() const { return kScriptVersion; }
  const std::filesystem::path& canonical_path() const {
    return canonical_path_;
  }
  uint64_t content_digest() const { return content_digest_; }
  const std::string& name() const { return name_; }
  const std::string& expected_title_id() const { return expected_title_id_; }
  uint32_t player() const { return player_; }
  const std::vector<ScriptStep>& steps() const { return steps_; }
  uint32_t source_step_count() const {
    return static_cast<uint32_t>(steps_.size());
  }
  uint32_t expanded_transition_count() const {
    return expanded_transition_count_;
  }
  std::chrono::milliseconds nominal_duration() const {
    return nominal_duration_;
  }

 private:
  const std::filesystem::path canonical_path_;
  const uint64_t content_digest_;
  const std::string name_;
  const std::string expected_title_id_;
  const uint32_t player_;
  const std::vector<ScriptStep> steps_;
  const uint32_t expanded_transition_count_;
  const std::chrono::milliseconds nominal_duration_;
};

struct ScriptLoadResult {
  std::shared_ptr<const InputScript> script;
  std::string error;

  explicit operator bool() const { return script != nullptr; }
};

ScriptLoadResult ParseInputScript(std::string_view content,
                                  std::filesystem::path source_path = {});
ScriptLoadResult LoadInputScript(const std::filesystem::path& path);

// Discovery-only live input lines (--input_live_file). Lines appended to the
// live file are consumed after the scripted timeline completes; each line is
// one step, in the same units as the TOML profile:
//   press <button>[+<button>...] [hold_ms] [release_ms] [repeat]
//   stick <lx|ly|rx|ry> <value> [<axis> <value>...] [hold_ms] [release_ms]
//   trigger <lt|rt> <value> [<axis> <value>...] [hold_ms] [release_ms]
//   wait <ms>
//   marker <name>
// Blank lines and lines starting with '#' are ignored (no step, no error).
struct LiveLineParseResult {
  std::optional<ScriptStep> step;
  std::string error;
};
LiveLineParseResult ParseLiveInputLine(std::string_view line);

}  // namespace xe::hid::scripted

#endif  // XENIA_HID_SCRIPTED_SCRIPTED_INPUT_SCRIPT_H_
