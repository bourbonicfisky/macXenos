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

#include <string>

#include "xenia/base/logging.h"

namespace {

bool g_claimed = false;
id g_connect_observer = nil;
std::function<void()> g_on_press;

void Apply(GCController* controller) {
  if (@available(macOS 11.0, *)) {
    std::string bound;
    for (NSString* key in controller.physicalInputProfile.buttons) {
      GCControllerButtonInput* button =
          controller.physicalInputProfile.buttons[key];
      if (button.boundToSystemGesture) {
        button.preferredSystemGestureState =
            g_claimed ? GCSystemGestureStateDisabled
                      : GCSystemGestureStateEnabled;
        bound += std::string(bound.empty() ? "" : ", ") + key.UTF8String +
                 (button.preferredSystemGestureState ==
                          GCSystemGestureStateDisabled
                      ? " (claimed)"
                      : " (macOS)");
      }
    }
    // Hear the press ourselves: SDL may never report it.
    GCControllerButtonInput* home =
        controller.physicalInputProfile.buttons[GCInputButtonHome];
    home.pressedChangedHandler =
        ^(GCControllerButtonInput* button, float value, BOOL pressed) {
          if (!pressed) return;
          XELOGI("Controller Home button: pressed ({})",
                 g_claimed ? "opening or closing the dashboard"
                           : "left to macOS");
          if (g_claimed && g_on_press) g_on_press();
        };
    XELOGI("Controller Home button: {}: {}",
           controller.vendorName ? controller.vendorName.UTF8String : "?",
           bound.empty() ? "no button bound to a system gesture" : bound);
  }
}

}  // namespace

namespace xe {
namespace app {

void SetControllerHomeButtonHandler(std::function<void()> on_press) {
  g_on_press = std::move(on_press);
}

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
