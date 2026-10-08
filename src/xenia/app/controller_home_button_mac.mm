/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/controller_home_button_mac.h"

#import <Foundation/Foundation.h>
#import <GameController/GameController.h>

#include "xenia/base/logging.h"

namespace {

bool g_claimed = false;
id g_connect_observer = nil;

void Apply(GCController* controller) {
  if (@available(macOS 11.0, *)) {
    for (NSString* key in controller.physicalInputProfile.buttons) {
      GCControllerButtonInput* button =
          controller.physicalInputProfile.buttons[key];
      if (button.boundToSystemGesture) {
        button.preferredSystemGestureState =
            g_claimed ? GCSystemGestureStateDisabled
                      : GCSystemGestureStateEnabled;
      }
    }
  }
}

}  // namespace

namespace xe {
namespace app {

void SetControllerHomeButtonClaimed(bool claimed) {
  @autoreleasepool {
    g_claimed = claimed;
    if (!g_connect_observer) {
      // The framework only lists controllers once something has asked; a
      // connection later gets the same treatment.
      g_connect_observer = [[NSNotificationCenter defaultCenter]
          addObserverForName:GCControllerDidConnectNotification
                      object:nil
                       queue:[NSOperationQueue mainQueue]
                  usingBlock:^(NSNotification* note) {
                    GCController* controller = note.object;
                    if ([controller isKindOfClass:[GCController class]]) {
                      Apply(controller);
                    }
                  }];
    }
    for (GCController* controller in GCController.controllers) {
      Apply(controller);
    }
    XELOGI("Controller Home button: {} ({} controllers connected)",
           claimed ? "claimed for the dashboard" : "left to macOS",
           GCController.controllers.count);
  }
}

}  // namespace app
}  // namespace xe
