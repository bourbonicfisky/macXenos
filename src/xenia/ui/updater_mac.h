/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_UPDATER_MAC_H_
#define XENIA_UI_UPDATER_MAC_H_

namespace xe {
namespace ui {

// App updates through Sparkle (macOS). Sparkle.framework is loaded at run time
// from the bundle's Frameworks folder, so a bare development binary, or a
// bundle packaged without it, has no updater and nothing here does anything.
// Main thread only.

// Loads Sparkle and starts the updater if the bundle carries it and names a
// feed (SUFeedURL). Safe to call more than once. True when it is running.
bool StartMacUpdater();

// Whether StartMacUpdater found and started Sparkle.
bool IsMacUpdaterAvailable();

// The user asked to check now: Sparkle shows its own progress and result.
void CheckForMacUpdates();

// A quiet check: shows nothing unless an update is found.
void CheckForMacUpdatesInBackground();

// Applies check_for_updates (launch, weekly, monthly, never) to Sparkle's
// scheduled checks, after a change.
void ApplyMacUpdaterSettings();

}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_UPDATER_MAC_H_
