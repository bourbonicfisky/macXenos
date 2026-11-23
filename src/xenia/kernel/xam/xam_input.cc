/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/emulator.h"
#include "xenia/hid/input.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xam/xam_private.h"
#include "xenia/xbox.h"

DECLARE_bool(allow_mic_initialization);

DEFINE_bool(log_input_census, false,
            "Log each distinct input/notification API call tuple (api, "
            "arguments, result, guest caller) once, and call counts every 5 "
            "seconds. Diagnostics for titles that ignore input.",
            "HID");

namespace xe {
namespace kernel {
namespace xam {

void InputCensusNote(const char* api, uint32_t a, uint32_t b, uint32_t c,
                     uint32_t result) {
  if (!cvars::log_input_census) {
    return;
  }
  using Key =
      std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;
  static std::mutex mutex;
  static std::map<Key, uint64_t> counts;
  static auto last_dump = std::chrono::steady_clock::now();
  uint32_t lr = 0;
  if (auto* thread_state = cpu::ThreadState::Get()) {
    lr = uint32_t(thread_state->context()->lr);
  }
  static std::map<Key, bool> seen;
  std::lock_guard<std::mutex> lock(mutex);
  Key key{api, a, b, c, result, lr};
  if (counts.size() < 4096 && seen.size() < 65536) {
    ++counts[key];
    if (seen.emplace(key, true).second) {
      XELOGI(
          "InputCensus new {} a={:08X} b={:08X} c={:08X} result={:08X} "
          "lr={:08X}",
          api, a, b, c, result, lr);
    }
  }
  auto now = std::chrono::steady_clock::now();
  if (now - last_dump >= std::chrono::seconds(5)) {
    last_dump = now;
    for (const auto& [k, n] : counts) {
      XELOGI(
          "InputCensus count {} a={:08X} b={:08X} c={:08X} result={:08X} "
          "lr={:08X} n={}",
          std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k),
          std::get<4>(k), std::get<5>(k), n);
    }
    counts.clear();
  }
}

using xe::hid::X_INPUT_CAPABILITIES;
using xe::hid::X_INPUT_FLAG;
using xe::hid::X_INPUT_KEYSTROKE;
using xe::hid::X_INPUT_STATE;
using xe::hid::X_INPUT_VIBRATION;

dword_result_t XAutomationpUnbindController_entry(dword_t user_index) {
  if (user_index >= XUserMaxUserCount) {
    return 0;
  }

  return 1;
}
DECLARE_XAM_EXPORT1(XAutomationpUnbindController, kInput, kStub);

void XamResetInactivity_entry() {
  // Do we need to do anything?
}
DECLARE_XAM_EXPORT1(XamResetInactivity, kInput, kStub);

dword_result_t XamEnableInactivityProcessing_entry(dword_t inactivity_index,
                                                   dword_t enable) {
  // Enables/disables screen saver and auto shutoff
  return X_ERROR_SUCCESS;
}
DECLARE_XAM_EXPORT1(XamEnableInactivityProcessing, kInput, kStub);

static dword_result_t XamInputGetCapabilitiesEx_impl(
    dword_t unk, dword_t user_index, dword_t flags,
    pointer_t<X_INPUT_CAPABILITIES> caps) {
  if (unk > 1) {
    return X_ERROR_NOT_SUPPORTED;
  }

  // Fail-safe check
  if (!caps) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  caps.Zero();

  if ((flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER) != 0) {
    // should trap
  }

  if ((flags & 4) != 0) {
    // should trap
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  uint32_t actual_flags = flags;
  if (!flags) {
    actual_flags = X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD |
                   X_INPUT_FLAG::X_INPUT_FLAG_KEYBOARD;
  }

  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  return input_system->GetCapabilities(actual_user_index, actual_flags, caps);
}
dword_result_t XamInputGetCapabilitiesEx_entry(
    dword_t unk, dword_t user_index, dword_t flags,
    pointer_t<X_INPUT_CAPABILITIES> caps) {
  uint32_t result =
      XamInputGetCapabilitiesEx_impl(unk, user_index, flags, caps);
  InputCensusNote("XamInputGetCapabilitiesEx", unk, user_index, flags, result);
  return result;
}
DECLARE_XAM_EXPORT1(XamInputGetCapabilitiesEx, kInput, kSketchy);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetcapabilities(v=vs.85).aspx
dword_result_t XamInputGetCapabilities_entry(
    dword_t user_index, dword_t flags, pointer_t<X_INPUT_CAPABILITIES> caps) {
  // chrispy: actually, it appears that caps is never checked for null, it is
  // memset at the start regardless
  return XamInputGetCapabilitiesEx_entry(1, user_index, flags, caps);
}
DECLARE_XAM_EXPORT1(XamInputGetCapabilities, kInput, kSketchy);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetstate(v=vs.85).aspx
static dword_result_t XamInputGetState_impl(
    dword_t user_index, dword_t flags, pointer_t<X_INPUT_STATE> input_state) {
  if (input_state) {
    memset((void*)input_state.host_address(), 0, sizeof(X_INPUT_STATE));
  }
  if (user_index >= XUserMaxUserCount) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  if (kernel_state()->xam_state()->IsUIActive()) {
    // 58410A24: report the connection state, not the input. A snapshot,
    // because a real read advances the scripted driver (its timeline starts
    // on the first read, so a read during boot-time UI shifted every route).
    X_INPUT_STATE discarded_state = {};
    auto input_system = kernel_state()->emulator()->input_system();
    auto lock = input_system->lock();
    return input_system->GetStateSnapshot(
        user_index, !flags ? X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD : flags,
        &discarded_state);
  }

  // Games call this with a NULL state ptr, probably as a query.

  uint32_t actual_user_index = user_index;
  // chrispy: change this, logic is not right
  if ((actual_user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  X_RESULT result;
  auto input_system = kernel_state()->emulator()->input_system();
  {
    auto lock = input_system->lock();
    result = input_system->GetState(
        user_index, !flags ? X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD : flags,
        input_state);
  }

  if (input_state && result == X_ERROR_SUCCESS) {
    if (auto patch = kernel_state()->xmp_volume_patch()) {
      patch->OnInputPoll(input_state->packet_number);
    }
  }

  return result;
}
dword_result_t XamInputGetState_entry(dword_t user_index, dword_t flags,
                                      pointer_t<X_INPUT_STATE> input_state) {
  uint32_t result = XamInputGetState_impl(user_index, flags, input_state);
  if (cvars::log_input_census && input_state && user_index < 4) {
    static uint64_t last[4] = {};
    uint64_t now_state =
        uint64_t(uint16_t(input_state->gamepad.buttons)) |
        (uint64_t(uint16_t(input_state->gamepad.thumb_lx)) << 16) |
        (uint64_t(uint16_t(input_state->gamepad.thumb_ly)) << 32) |
        (uint64_t(uint16_t(input_state->gamepad.thumb_rx)) << 48);
    if (now_state != last[user_index]) {
      last[user_index] = now_state;
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
      XELOGI(
          "InputCensus change user={} buttons={:04X} lx={} ly={} rx={} "
          "packet={} ui={} t={}",
          uint32_t(user_index), uint16_t(input_state->gamepad.buttons),
          int16_t(input_state->gamepad.thumb_lx),
          int16_t(input_state->gamepad.thumb_ly),
          int16_t(input_state->gamepad.thumb_rx),
          uint32_t(input_state->packet_number),
          kernel_state()->xam_state()->IsUIActive(), ms);
    }
  }
  InputCensusNote(
      "XamInputGetState", user_index, flags,
      (input_state ? uint32_t(input_state->gamepad.buttons) : 0xFFFFu) |
          (kernel_state()->xam_state()->IsUIActive() ? 0x80000000u : 0u),
      result);
  return result;
}
DECLARE_XAM_EXPORT2(XamInputGetState, kInput, kImplemented, kHighFrequency);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputsetstate(v=vs.85).aspx
dword_result_t XamInputSetState_entry(
    dword_t user_index,
    dword_t flags, /* flags, as far as i can see, is not used*/
    pointer_t<X_INPUT_VIBRATION> vibration) {
  if (user_index >= XUserMaxUserCount) {
    return X_E_DEVICE_NOT_CONNECTED;
  }
  if (!vibration) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  return input_system->SetState(user_index, vibration);
}
DECLARE_XAM_EXPORT1(XamInputSetState, kInput, kImplemented);

// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.reference.xinputgetkeystroke(v=vs.85).aspx
static dword_result_t XamInputGetKeystroke_impl(
    dword_t user_index, dword_t flags, pointer_t<X_INPUT_KEYSTROKE> keystroke) {
  // https://github.com/CodeAsm/ffplay360/blob/master/Common/AtgXime.cpp
  // user index = index or XUSER_INDEX_ANY
  // flags = XINPUT_FLAG_GAMEPAD (| _ANYUSER | _ANYDEVICE)

  if (!keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  uint32_t actual_user_index = user_index;
  if ((actual_user_index & XUserIndexAny) == XUserIndexAny ||
      (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER)) {
    // Always pin user to 0.
    actual_user_index = 0;
  }

  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  return input_system->GetKeystroke(user_index, flags, keystroke);
}
dword_result_t XamInputGetKeystroke_entry(
    dword_t user_index, dword_t flags, pointer_t<X_INPUT_KEYSTROKE> keystroke) {
  uint32_t result = XamInputGetKeystroke_impl(user_index, flags, keystroke);
  InputCensusNote("XamInputGetKeystroke", user_index, flags,
                  keystroke ? (uint32_t(keystroke->virtual_key) |
                               (uint32_t(keystroke->flags) << 16))
                            : 0xFFFFFFFFu,
                  result);
  return result;
}
DECLARE_XAM_EXPORT1(XamInputGetKeystroke, kInput, kImplemented);

// Same as non-ex, just takes a pointer to user index.
static dword_result_t XamInputGetKeystrokeEx_impl(
    lpdword_t user_index_ptr, dword_t flags,
    pointer_t<X_INPUT_KEYSTROKE> keystroke) {
  if (!keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }

  keystroke.Zero();

  if (kernel_state()->xam_state()->IsUIActive()) {
    return X_ERROR_SUCCESS;
  }

  uint32_t user_index = *user_index_ptr;
  auto input_system = kernel_state()->emulator()->input_system();
  auto lock = input_system->lock();
  if ((user_index & XUserIndexAny) == XUserIndexAny) {
    // Always pin user to 0.
    user_index = 0;
  }

  if (flags & X_INPUT_FLAG::X_INPUT_FLAG_ANY_USER) {
    // That flag means we should iterate over every connected controller and
    // check which one have pending request.
    auto result = X_ERROR_DEVICE_NOT_CONNECTED;
    for (uint32_t i = 0; i < XUserMaxUserCount; i++) {
      auto result = input_system->GetKeystroke(i, flags, keystroke);

      // Return result from first user that have pending request
      if (result == X_ERROR_SUCCESS) {
        *user_index_ptr = keystroke->user_index;
        return result;
      }
    }
    return result;
  }

  auto result = input_system->GetKeystroke(user_index, flags, keystroke);

  if (XSUCCEEDED(result)) {
    *user_index_ptr = keystroke->user_index;
  }
  return result;
}
dword_result_t XamInputGetKeystrokeEx_entry(
    lpdword_t user_index_ptr, dword_t flags,
    pointer_t<X_INPUT_KEYSTROKE> keystroke) {
  uint32_t result =
      XamInputGetKeystrokeEx_impl(user_index_ptr, flags, keystroke);
  InputCensusNote("XamInputGetKeystrokeEx",
                  user_index_ptr ? uint32_t(*user_index_ptr) : 0xFFFFFFFFu,
                  flags,
                  keystroke ? (uint32_t(keystroke->virtual_key) |
                               (uint32_t(keystroke->flags) << 16))
                            : 0xFFFFFFFFu,
                  result);
  return result;
}
DECLARE_XAM_EXPORT1(XamInputGetKeystrokeEx, kInput, kImplemented);

X_HRESULT_result_t XamUserGetDeviceContext_entry(dword_t user_index,
                                                 dword_t device_type,
                                                 lpdword_t out_ptr) {
  // Games check the result - usually with some masking.
  // If this function fails they assume zero, so let's fail AND
  // set zero just to be safe.
  *out_ptr = 0;
  if (kernel_state()->xam_state()->IsUserSignedIn(user_index) ||
      (user_index & XUserIndexAny) == XUserIndexAny) {
    if (device_type == 4 && cvars::allow_mic_initialization) {  // Microphone
      *out_ptr = 6 << 28;
    } else {
      *out_ptr = (uint32_t)user_index;
    }
    InputCensusNote("XamUserGetDeviceContext", user_index, device_type,
                    *out_ptr, X_E_SUCCESS);
    return X_E_SUCCESS;
  } else {
    InputCensusNote("XamUserGetDeviceContext", user_index, device_type, 0,
                    X_E_DEVICE_NOT_CONNECTED);
    return X_E_DEVICE_NOT_CONNECTED;
  }
}
DECLARE_XAM_EXPORT1(XamUserGetDeviceContext, kInput, kStub);

X_HRESULT_result_t XamInputNonControllerGetRawEx_entry(
    dword_t device_id, lpdword_t buffer_ptr, lpdword_t buffer_length_ptr,
    lpword_t state_ptr) {
  if (device_id != 5 && device_id != 6) {
    return X_ERROR_INVALID_PARAMETER;
  }
  if (!state_ptr || !buffer_length_ptr || !buffer_ptr) {
    return X_ERROR_INVALID_PARAMETER;
  }

  if (*buffer_length_ptr == 0 || *buffer_length_ptr > hid::kPortalBufferSize) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto portal = kernel_state()->emulator()->input_system()->GetPortal();
  if (!portal) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  uint32_t bytes_read = *buffer_length_ptr;
  uint16_t state = 0;

  const auto result = portal->Read(
      {kernel_memory()->TranslateVirtual(buffer_ptr.guest_address()),
       *buffer_length_ptr},
      bytes_read, state);

  if (XSUCCEEDED(result)) {
    *buffer_length_ptr = bytes_read;
    *state_ptr = state;
  }
  return result;
}
DECLARE_XAM_EXPORT1(XamInputNonControllerGetRawEx, kInput, kSketchy);

X_HRESULT_result_t XamInputNonControllerSetRawEx_entry(dword_t device_id,
                                                       lpdword_t buffer_ptr,
                                                       dword_t buffer_length) {
  if (device_id != 5 && device_id != 6) {
    return X_ERROR_INVALID_PARAMETER;
  }
  if (!buffer_ptr || !buffer_length || buffer_length > hid::kPortalBufferSize) {
    return X_ERROR_INVALID_PARAMETER;
  }

  auto portal = kernel_state()->emulator()->input_system()->GetPortal();
  if (!portal) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  return portal->Write(
      {kernel_memory()->TranslateVirtual(buffer_ptr.guest_address()),
       buffer_length});
}
DECLARE_XAM_EXPORT1(XamInputNonControllerSetRawEx, kInput, kSketchy);

X_HRESULT_result_t XamInputNonControllerGetRaw_entry(
    lpword_t state_ptr, lpdword_t buffer_length_ptr, lpdword_t buffer_ptr) {
  return XamInputNonControllerGetRawEx_entry(5, buffer_ptr, buffer_length_ptr,
                                             state_ptr);
}
DECLARE_XAM_EXPORT1(XamInputNonControllerGetRaw, kInput, kSketchy);

X_HRESULT_result_t XamInputNonControllerSetRaw_entry(dword_t buffer_length,
                                                     lpdword_t buffer_ptr) {
  // Normally there are handled separatelly with different first param, but
  // whatever.
  return XamInputNonControllerSetRawEx_entry(5, buffer_ptr, buffer_length);
}
DECLARE_XAM_EXPORT1(XamInputNonControllerSetRaw, kInput, kSketchy);

// Input from a remote control client (LRC), of which none is ever connected.
// xam reports an empty queue as ERROR_EMPTY and writes neither output.
dword_result_t XamLrcGetInputWithSeek_entry(
    pointer_t<X_INPUT_KEYSTROKE> keystroke, lpqword_t seek_ptr) {
  return X_ERROR_EMPTY;
}
DECLARE_XAM_EXPORT1(XamLrcGetInputWithSeek, kInput, kStub);

}  // namespace xam
}  // namespace kernel
}  // namespace xe

DECLARE_XAM_EMPTY_REGISTER_EXPORTS(Input);
