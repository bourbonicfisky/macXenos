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
};

std::unique_ptr<FilePicker> FilePicker::Create() {
  return std::make_unique<MacFilePicker>();
}

bool MacFilePicker::Show(Window* parent_window) {
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
