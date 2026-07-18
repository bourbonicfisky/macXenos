/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_SCRIPTED_SCRIPTED_INPUT_TIMELINE_H_
#define XENIA_HID_SCRIPTED_SCRIPTED_INPUT_TIMELINE_H_

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "xenia/hid/scripted/scripted_input_script.h"

namespace xe::hid::scripted {

enum class TimelineEventType { kMarker, kCompletion };

struct TimelineEvent {
  TimelineEventType type;
  std::string marker;
  std::chrono::milliseconds elapsed;
};

class ScriptedInputTimeline final {
 public:
  using Clock = std::chrono::steady_clock;
  using NowFunction = std::function<Clock::time_point()>;
  using EventCallback = std::function<void(const TimelineEvent&)>;

  ScriptedInputTimeline(std::shared_ptr<const InputScript> script,
                        NowFunction now_function,
                        EventCallback event_callback = {});

  GamepadState Poll();
  bool started() const { return started_; }
  bool completed() const { return completed_; }
  std::chrono::milliseconds elapsed() const;

 private:
  enum class Phase { kNone, kWait, kActive, kRelease };

  void EmitEvent(TimelineEventType type, std::string marker,
                 Clock::time_point now);

  const std::shared_ptr<const InputScript> script_;
  const NowFunction now_function_;
  const EventCallback event_callback_;

  bool started_ = false;
  bool completed_ = false;
  Clock::time_point start_time_ = {};
  Clock::time_point last_poll_time_ = {};
  Clock::time_point deadline_ = {};
  size_t step_index_ = 0;
  Phase phase_ = Phase::kNone;
  uint32_t repetitions_remaining_ = 0;
};

}  // namespace xe::hid::scripted

#endif  // XENIA_HID_SCRIPTED_SCRIPTED_INPUT_TIMELINE_H_
