/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/system.h"

#import <AppKit/AppKit.h>

#include <string>

namespace xe {

void LaunchWebBrowser(const std::string_view url) {
  @autoreleasepool {
    NSString* url_string = [NSString stringWithUTF8String:std::string(url)
                                                              .c_str()];
    NSURL* ns_url = [NSURL URLWithString:url_string];
    if (ns_url) {
      [[NSWorkspace sharedWorkspace] openURL:ns_url];
    }
  }
}

void LaunchFileExplorer(const std::filesystem::path& path) {
  @autoreleasepool {
    NSString* path_string =
        [NSString stringWithUTF8String:path.string().c_str()];
    [[NSWorkspace sharedWorkspace] selectFile:path_string
                     inFileViewerRootedAtPath:@""];
  }
}

void ShowSimpleMessageBox(SimpleMessageBoxType type, std::string_view message) {
  @autoreleasepool {
    NSString* title;
    NSAlertStyle style;
    switch (type) {
      case SimpleMessageBoxType::Help:
        title = @"Xenia Help";
        style = NSAlertStyleInformational;
        break;
      case SimpleMessageBoxType::Warning:
        title = @"Xenia Warning";
        style = NSAlertStyleWarning;
        break;
      case SimpleMessageBoxType::Error:
      default:
        title = @"Xenia Error";
        style = NSAlertStyleCritical;
        break;
    }
    NSAlert* alert = [[NSAlert alloc] init];
    [alert setAlertStyle:style];
    [alert setMessageText:title];
    [alert setInformativeText:[NSString stringWithUTF8String:std::string(message)
                                                                 .c_str()]];
    [alert runModal];
  }
}

// Canary-era additions not present in the reference system_mac.mm. macOS has no
// equivalent of Windows priority classes here, and the Game Bar / Nexus overlay
// is Windows-only; mirror the gnulinux stubs.
bool SetProcessPriorityClass(const uint32_t priority_class) { return true; }

bool IsUseNexusForGameBarEnabled() { return false; }

}  // namespace xe
