/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/windowed_app_context_mac.h"

#import <AppKit/AppKit.h>
#import <dispatch/dispatch.h>

namespace xe {
namespace ui {

MacWindowedAppContext::MacWindowedAppContext() {
  // Ensure the shared application exists and behaves like a regular GUI app
  // (dock icon, menu bar, focusable windows) even when launched from a plain
  // executable rather than a bundle.
  [NSApplication sharedApplication];
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
}

MacWindowedAppContext::~MacWindowedAppContext() {
  // Execute anything still queued without invoking virtual methods.
  ExecutePendingFunctionsFromUIThread();
}

void MacWindowedAppContext::NotifyUILoopOfPendingFunctions() {
  // Can be called from any thread. Dispatching to the main queue both wakes the
  // run loop and runs the work on the UI thread.
  dispatch_async(dispatch_get_main_queue(), ^{
    if (!HasQuitFromUIThread()) {
      ExecutePendingFunctionsFromUIThread();
    }
  });
}

void MacWindowedAppContext::PlatformQuitFromUIThread() {
  [NSApp stop:nil];
  // [NSApp stop:] only takes effect after the next event is processed, so post
  // a dummy event to unblock the run loop immediately.
  @autoreleasepool {
    NSEvent* event = [NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                        location:NSMakePoint(0, 0)
                                   modifierFlags:0
                                       timestamp:0
                                    windowNumber:0
                                         context:nil
                                         subtype:0
                                           data1:0
                                           data2:0];
    [NSApp postEvent:event atStart:YES];
  }
}

void MacWindowedAppContext::RunMainMacLoop() {
  if (HasQuitFromUIThread()) {
    return;
  }
  [NSApp finishLaunching];
  [NSApp activateIgnoringOtherApps:YES];
  [NSApp run];
  // If the loop exited by some other means, keep the context consistent.
  QuitFromUIThread();
}

}  // namespace ui
}  // namespace xe
