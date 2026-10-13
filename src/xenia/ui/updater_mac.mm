/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/updater_mac.h"

#import <Cocoa/Cocoa.h>

#include <string>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"

DEFINE_string(check_for_updates, "launch",
              "When to look on macxenos.com for a newer macXenos (Sparkle): "
              "launch (each time the app starts), weekly, monthly or never. "
              "It offers the update and installs nothing without asking. "
              "Checking by hand from the app menu works whatever this is.",
              "UI");
DEFINE_bool(check_for_updates_asked, false,
            "Whether the first-launch question about update checks has been "
            "answered. Until it is, the app does not look for updates on its "
            "own.",
            "UI");

// The parts of Sparkle's API used here. The framework is loaded at run time
// (no link-time dependency), so its classes are reached through these
// declarations rather than its headers.
@protocol XeSparkleUpdater <NSObject>
@property(nonatomic) BOOL automaticallyChecksForUpdates;
@property(nonatomic) NSTimeInterval updateCheckInterval;
@property(nonatomic, readonly) BOOL canCheckForUpdates;
- (void)checkForUpdatesInBackground;
@end

@protocol XeSparkleUpdaterController <NSObject>
- (instancetype)initWithStartingUpdater:(BOOL)startUpdater
                        updaterDelegate:(id)updaterDelegate
                     userDriverDelegate:(id)userDriverDelegate;
@property(nonatomic, readonly) id<XeSparkleUpdater> updater;
- (void)startUpdater;
- (void)checkForUpdates:(id)sender;
@end

namespace xe {
namespace ui {

namespace {
id<XeSparkleUpdaterController> updater_controller = nil;
bool start_attempted = false;
}  // namespace

bool StartMacUpdater() {
  if (start_attempted) {
    return updater_controller != nil;
  }
  start_attempted = true;
  NSBundle* main_bundle = [NSBundle mainBundle];
  NSString* feed = [main_bundle objectForInfoDictionaryKey:@"SUFeedURL"];
  NSURL* frameworks = main_bundle.privateFrameworksURL;
  if (feed.length == 0 || !frameworks) {
    return false;
  }
  NSBundle* sparkle = [NSBundle
      bundleWithURL:[frameworks URLByAppendingPathComponent:@"Sparkle.framework"]];
  if (!sparkle) {
    return false;
  }
  NSError* error = nil;
  if (![sparkle loadAndReturnError:&error]) {
    XELOGE("Updater: could not load Sparkle: {}",
           error ? error.localizedDescription.UTF8String : "unknown error");
    return false;
  }
  Class controller_class = NSClassFromString(@"SPUStandardUpdaterController");
  if (!controller_class) {
    XELOGE("Updater: Sparkle has no SPUStandardUpdaterController");
    return false;
  }
  // Created stopped so the user's choice is set before the updater starts and
  // schedules its first check.
  updater_controller = [(id<XeSparkleUpdaterController>)[controller_class alloc]
      initWithStartingUpdater:NO
              updaterDelegate:nil
           userDriverDelegate:nil];
  if (!updater_controller) {
    return false;
  }
  ApplyMacUpdaterSettings();
  [updater_controller startUpdater];
  // "launch" is not a Sparkle schedule: a quiet check now, which shows
  // nothing unless an update is found.
  if (cvars::check_for_updates_asked && cvars::check_for_updates == "launch") {
    [updater_controller.updater checkForUpdatesInBackground];
  }
  XELOGI("Updater: Sparkle {} started, feed {}, checks {}",
         [[sparkle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]
             UTF8String] ?: "?",
         feed.UTF8String,
         cvars::check_for_updates_asked ? cvars::check_for_updates
                                        : std::string("not asked yet"));
  return true;
}

bool IsMacUpdaterAvailable() { return updater_controller != nil; }

void CheckForMacUpdates() {
  if (!StartMacUpdater()) {
    return;
  }
  [updater_controller checkForUpdates:nil];
}

void CheckForMacUpdatesInBackground() {
  if (updater_controller) {
    [updater_controller.updater checkForUpdatesInBackground];
  }
}

void ApplyMacUpdaterSettings() {
  if (!updater_controller) {
    return;
  }
  // weekly / monthly are Sparkle's own schedule; launch and never turn it off
  // (launch checks from StartMacUpdater instead), as does a first-launch
  // question not answered yet. Anything else reads as launch.
  const std::string& when = cvars::check_for_updates;
  id<XeSparkleUpdater> updater = updater_controller.updater;
  if (cvars::check_for_updates_asked &&
      (when == "weekly" || when == "monthly")) {
    updater.updateCheckInterval =
        when == "weekly" ? 7.0 * 24 * 3600 : 30.0 * 24 * 3600;
    updater.automaticallyChecksForUpdates = YES;
  } else {
    updater.automaticallyChecksForUpdates = NO;
  }
}

}  // namespace ui
}  // namespace xe
