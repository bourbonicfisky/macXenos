/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_SCRIPTED_SCRIPTED_INPUT_DRIVER_H_
#define XENIA_HID_SCRIPTED_SCRIPTED_INPUT_DRIVER_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "xenia/hid/input_driver.h"
#include "xenia/hid/scripted/scripted_input_timeline.h"

namespace xe::hid::scripted {

class ScriptedInputDriver final : public InputDriver {
 public:
  ScriptedInputDriver(xe::ui::Window* window, size_t window_z_order,
                      std::shared_ptr<const InputScript> script,
                      ScriptedInputTimeline::NowFunction now_function = {},
                      ScriptedInputTimeline::EventCallback event_callback = {},
                      bool transition_telemetry = false,
                      std::filesystem::path live_file = {});
  ~ScriptedInputDriver() override;

  X_STATUS Setup() override;
  X_RESULT GetCapabilities(uint32_t user_index, uint32_t flags,
                           X_INPUT_CAPABILITIES* out_caps) override;
  X_RESULT GetState(uint32_t user_index, X_INPUT_STATE* out_state) override;
  X_RESULT GetStateSnapshot(uint32_t user_index,
                            X_INPUT_STATE* out_state) override;
  X_RESULT SetState(uint32_t user_index, X_INPUT_VIBRATION* vibration) override;
  X_RESULT GetKeystroke(uint32_t user_index, uint32_t flags,
                        X_INPUT_KEYSTROKE* out_keystroke) override;
  InputType GetInputType() const override;
  bool AppliesInputDeadzone() const override { return false; }
  bool GetControllerIdentity(uint32_t user_index,
                             ControllerIdentity* out_identity) override;

  bool completed() const;

  // Diagnostic-only accounting of emitted transition-telemetry records. Always
  // zero while the telemetry option is off.
  uint32_t transition_records_emitted() const;
  uint32_t completion_records_emitted() const;
  // Discovery only: lines applied from the live file so far.
  uint32_t live_lines_applied() const;

 private:
  static ScriptedInputTimeline::Clock::time_point SteadyNow();
  void HandleTimelineEvent(const TimelineEvent& event);
  void CacheState(const GamepadState& state);
  // Both require mutex_ to be held by the caller.
  void LogTransitionObservation();
  void LogTransitionSummary();
  // Live file (discovery only): consumed after the scripted timeline has
  // completed. Requires mutex_. Returns the state to present this poll.
  GamepadState PollLive(ScriptedInputTimeline::Clock::time_point now);
  void IngestLiveLines(ScriptedInputTimeline::Clock::time_point now);

  const std::shared_ptr<const InputScript> script_;
  const std::filesystem::path live_file_;
  ScriptedInputTimeline::NowFunction now_function_;
  std::unique_ptr<ScriptedInputTimeline> live_timeline_;
  std::uintmax_t live_offset_ = 0;
  std::string live_partial_line_;
  ScriptedInputTimeline::Clock::time_point live_last_check_ = {};
  uint32_t live_lines_seen_ = 0;
  uint32_t live_lines_applied_ = 0;
  uint32_t live_batches_ = 0;
  const bool transition_telemetry_;
  std::unique_ptr<ScriptedInputTimeline> timeline_;
  mutable std::mutex mutex_;
  GamepadState gamepad_state_ = {};
  X_INPUT_STATE cached_state_ = {};
  uint32_t packet_number_ = 0;
  uint64_t poll_ordinal_ = 0;
  uint32_t expected_transitions_ = 0;
  uint32_t observed_transitions_ = 0;
  uint32_t logged_transitions_ = 0;
  uint32_t logged_summaries_ = 0;
  uint32_t last_transition_ordinal_ = 0;
};

}  // namespace xe::hid::scripted

#endif  // XENIA_HID_SCRIPTED_SCRIPTED_INPUT_DRIVER_H_
