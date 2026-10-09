/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_CONTROLLER_HOME_BUTTON_MAC_H_
#define XENIA_APP_CONTROLLER_HOME_BUTTON_MAC_H_

#include <functional>

namespace xe {
namespace app {

// Whether macOS or the app gets the controller's Home button (Xbox Guide, PS
// button). By default macOS treats it as a system gesture (the Games overlay,
// Launchpad). Claimed, the press reaches the app instead, where a tap opens
// the dashboard. Applies to controllers already connected and any that
// connect later. SDL only does this for controllers it opens through Apple's
// own driver; Xbox and PlayStation pads go through its HID driver instead.
void SetControllerHomeButtonClaimed(bool claimed);

// Called (on the main thread) when a controller's Home button is pressed
// while it is claimed, as macOS's own framework reports it. SDL does not see
// the Home button of every pad (an 8BitDo through Apple's driver never
// reports it), so the dashboard opens from here too.
void SetControllerHomeButtonHandler(std::function<void()> on_press);

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_CONTROLLER_HOME_BUTTON_MAC_H_
