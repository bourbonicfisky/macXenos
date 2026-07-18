/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/hid/scripted/scripted_input_timeline.h"

#include <cassert>
#include <utility>

namespace xe::hid::scripted {

ScriptedInputTimeline::ScriptedInputTimeline(
    std::shared_ptr<const InputScript> script, NowFunction now_function,
    EventCallback event_callback)
    : script_(std::move(script)),
      now_function_(std::move(now_function)),
      event_callback_(std::move(event_callback)) {
  assert(script_ != nullptr);
  assert(now_function_ != nullptr);
}

std::chrono::milliseconds ScriptedInputTimeline::elapsed() const {
  if (!started_) {
    return std::chrono::milliseconds::zero();
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(last_poll_time_ -
                                                               start_time_);
}

void ScriptedInputTimeline::EmitEvent(TimelineEventType type,
                                      std::string marker,
                                      Clock::time_point now) {
  if (!event_callback_) {
    return;
  }
  event_callback_({type, std::move(marker),
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       now - start_time_)});
}

GamepadState ScriptedInputTimeline::Poll() {
  if (completed_) {
    return {};
  }

  const Clock::time_point now = now_function_();
  if (!started_) {
    started_ = true;
    start_time_ = now;
  }
  last_poll_time_ = now;

  for (;;) {
    if (step_index_ >= script_->steps().size()) {
      completed_ = true;
      phase_ = Phase::kNone;
      EmitEvent(TimelineEventType::kCompletion, {}, now);
      return {};
    }

    const ScriptStep& step = script_->steps()[step_index_];
    if (const auto* marker = std::get_if<MarkerStep>(&step)) {
      EmitEvent(TimelineEventType::kMarker, marker->name, now);
      ++step_index_;
      phase_ = Phase::kNone;
      continue;
    }

    if (const auto* wait = std::get_if<WaitStep>(&step)) {
      if (phase_ != Phase::kWait) {
        phase_ = Phase::kWait;
        deadline_ = now + wait->duration;
        return {};
      }
      if (now < deadline_) {
        return {};
      }
      ++step_index_;
      phase_ = Phase::kNone;
      continue;
    }

    const auto& input = std::get<InputStep>(step);
    if (phase_ == Phase::kNone) {
      repetitions_remaining_ = input.repeat;
      phase_ = Phase::kActive;
      deadline_ = now + input.hold_duration;
      return input.state;
    }
    if (phase_ == Phase::kActive) {
      if (now < deadline_) {
        return input.state;
      }
      phase_ = Phase::kRelease;
      deadline_ = now + input.release_duration;
      return {};
    }
    if (phase_ == Phase::kRelease) {
      if (now < deadline_) {
        return {};
      }
      assert(repetitions_remaining_ != 0);
      --repetitions_remaining_;
      if (repetitions_remaining_ != 0) {
        phase_ = Phase::kActive;
        deadline_ = now + input.hold_duration;
        return input.state;
      }
      ++step_index_;
      phase_ = Phase::kNone;
      continue;
    }
  }
}

}  // namespace xe::hid::scripted
