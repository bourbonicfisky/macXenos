/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/file_picker.h"

#import <AppKit/AppKit.h>

#include <string>
#include <vector>

namespace xe {
namespace ui {

class MacFilePicker : public FilePicker {
 public:
  MacFilePicker() = default;
  ~MacFilePicker() override = default;

  bool Show(Window* parent_window) override;

 private:
  bool ShowOnMainThread(Window* parent_window);
};

std::unique_ptr<FilePicker> FilePicker::Create() {
  return std::make_unique<MacFilePicker>();
}

bool MacFilePicker::Show(Window* parent_window) {
  // AppKit panels may only be created and run on the main thread. Guest
  // threads reach here through XamSwapDisc (Emulator::GetNewDiscPath), and
  // creating the panel off-main raises an NSException that std::terminate
  // turns into a SIGABRT (Castlevania: Lords of Shadow disc 2, 2026-09-26).
  // The main thread services its dispatch queue from the AppKit run loop, so
  // a synchronous hop is safe and keeps the modal semantics the caller wants.
  if (![NSThread isMainThread]) {
    __block bool result = false;
    dispatch_sync(dispatch_get_main_queue(), ^{
      result = ShowOnMainThread(parent_window);
    });
    return result;
  }
  return ShowOnMainThread(parent_window);
}

bool MacFilePicker::ShowOnMainThread(Window* parent_window) {
  @autoreleasepool {
    std::vector<std::filesystem::path> results;

    if (mode() == Mode::kSave) {
      NSSavePanel* panel = [NSSavePanel savePanel];
      [panel setTitle:[NSString stringWithUTF8String:title().c_str()]];
      if ([panel runModal] == NSModalResponseOK) {
        NSURL* url = [panel URL];
        if (url) {
          results.emplace_back([[url path] UTF8String]);
        }
      }
    } else {
      NSOpenPanel* panel = [NSOpenPanel openPanel];
      [panel setTitle:[NSString stringWithUTF8String:title().c_str()]];
      bool directories = type() == Type::kDirectory;
      [panel setCanChooseFiles:directories ? NO : YES];
      [panel setCanChooseDirectories:directories ? YES : NO];
      [panel setAllowsMultipleSelection:multi_selection() ? YES : NO];
      if ([panel runModal] == NSModalResponseOK) {
        for (NSURL* url in [panel URLs]) {
          results.emplace_back([[url path] UTF8String]);
        }
      }
    }

    if (results.empty()) {
      return false;
    }
    set_selected_files(std::move(results));
    return true;
  }
}

}  // namespace ui
}  // namespace xe
