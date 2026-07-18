/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_HID_INPUT_DRIVER_H_
#define XENIA_HID_INPUT_DRIVER_H_

#include <cstddef>
#include <functional>
#include <string>

#include "xenia/hid/input.h"
#include "xenia/ui/window.h"
#include "xenia/xbox.h"

namespace xe {
namespace ui {
class Window;
}  // namespace ui
}  // namespace xe

namespace xe {
namespace hid {

class InputSystem;

enum InputType { None, Controller = 1, Keyboard = 2, Other = 4 };

// Brand family of a connected controller, decoupled from any backend's native
// enum so UI code (glyphs/labels) doesn't depend on SDL. Drivers map their own
// type info into this.
enum class ControllerKind {
  kUnknown,
  kXbox,
  kPlayStation,
  kNintendo,
  kOther,
};

// Identity of the controller occupying a player slot, surfaced for UI (the
// guest always sees an Xbox 360 pad regardless of this).
struct ControllerIdentity {
  bool present = false;
  ControllerKind kind = ControllerKind::kUnknown;
  std::string name;  // backend-reported display name, e.g. "PS4 Controller"
};

class InputDriver {
 public:
  virtual ~InputDriver() = default;

  virtual X_STATUS Setup() = 0;

  virtual X_RESULT GetCapabilities(uint32_t user_index, uint32_t flags,
                                   X_INPUT_CAPABILITIES* out_caps) = 0;
  virtual X_RESULT GetState(uint32_t user_index, X_INPUT_STATE* out_state) = 0;
  // Host UI and hotkey polling must be able to inspect a controller without
  // advancing deterministic or event-driven input. Physical drivers keep
  // their existing behavior by default.
  virtual X_RESULT GetStateSnapshot(uint32_t user_index,
                                    X_INPUT_STATE* out_state) {
    return GetState(user_index, out_state);
  }
  virtual X_RESULT SetState(uint32_t user_index,
                            X_INPUT_VIBRATION* vibration) = 0;
  virtual X_RESULT GetKeystroke(uint32_t user_index, uint32_t flags,
                                X_INPUT_KEYSTROKE* out_keystroke) = 0;

  virtual InputType GetInputType() const = 0;

  // Physical controller drivers use the host-configured stick deadzone by
  // default. Deterministic or already-normalized drivers may opt out.
  virtual bool AppliesInputDeadzone() const { return true; }

  // Reports the identity of the controller in the given slot for UI display.
  // Returns false (and leaves out_identity untouched) for drivers that have no
  // notion of controller identity (keyboard, nop) or an empty slot.
  virtual bool GetControllerIdentity(uint32_t user_index,
                                     ControllerIdentity* out_identity) {
    return false;
  }

  // Reassigns the controller occupying from_slot to to_slot for UI-driven slot
  // management. If to_slot is occupied the two controllers swap. Returns true
  // if this driver owns from_slot and performed the move. Default: unsupported.
  virtual bool ReassignControllerSlot(uint32_t from_slot, uint32_t to_slot) {
    return false;
  }

  // Device-mapping rebind, for UI that lets the user remap a controller's
  // physical buttons. SDL-style binding tokens are used (e.g. "b3" for a
  // button). All default to unsupported.
  //
  // BeginInputCapture: start listening for the next physical input on the
  //   controller in user_index. Returns true if capture started.
  // PollCapturedInput: returns true once an input has been captured, writing
  //   its SDL binding token to out_binding.
  // GetOutputBinding: current binding for an SDL output name (e.g. "a").
  // RebindOutput: point an SDL output (e.g. "a") at a captured binding and
  //   apply + persist it for this controller's GUID.
  // ResetMapping: restore the controller's mapping to its as-loaded default.
  virtual bool BeginInputCapture(uint32_t user_index) { return false; }
  virtual void CancelInputCapture() {}
  virtual bool PollCapturedInput(std::string* out_binding) { return false; }
  virtual std::string GetOutputBinding(uint32_t user_index,
                                       const std::string& sdl_output) {
    return "";
  }
  virtual bool RebindOutput(uint32_t user_index, const std::string& sdl_output,
                            const std::string& binding) {
    return false;
  }
  virtual bool ResetMapping(uint32_t user_index) { return false; }

 protected:
  explicit InputDriver(xe::ui::Window* window, size_t window_z_order)
      : window_(window), window_z_order_(window_z_order) {}

  xe::ui::Window* window() const { return window_; }
  size_t window_z_order() const { return window_z_order_; }

 private:
  xe::ui::Window* window_;
  size_t window_z_order_;
};

}  // namespace hid
}  // namespace xe

#endif  // XENIA_HID_INPUT_DRIVER_H_
