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

#include <memory>
#include <mutex>

#include "xenia/hid/input_driver.h"
#include "xenia/hid/scripted/scripted_input_timeline.h"

namespace xe::hid::scripted {

class ScriptedInputDriver final : public InputDriver {
 public:
  ScriptedInputDriver(xe::ui::Window* window, size_t window_z_order,
                      std::shared_ptr<const InputScript> script,
                      ScriptedInputTimeline::NowFunction now_function = {},
                      ScriptedInputTimeline::EventCallback event_callback = {});
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

 private:
  static ScriptedInputTimeline::Clock::time_point SteadyNow();
  void HandleTimelineEvent(const TimelineEvent& event);
  void CacheState(const GamepadState& state);

  const std::shared_ptr<const InputScript> script_;
  std::unique_ptr<ScriptedInputTimeline> timeline_;
  mutable std::mutex mutex_;
  GamepadState gamepad_state_ = {};
  X_INPUT_STATE cached_state_ = {};
  uint32_t packet_number_ = 0;
};

}  // namespace xe::hid::scripted

#endif  // XENIA_HID_SCRIPTED_SCRIPTED_INPUT_DRIVER_H_
