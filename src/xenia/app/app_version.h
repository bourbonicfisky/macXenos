/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_APP_VERSION_H_
#define XENIA_APP_APP_VERSION_H_

#include <string>

#include "xenia/base/platform.h"

#include "version.h"  // XE_BUILD_VERSION (generated)

#if XE_PLATFORM_MAC
#include <CoreFoundation/CoreFoundation.h>
#endif  // XE_PLATFORM_MAC

namespace xe {
namespace app {

// A string from the app bundle's Info.plist, empty for a bare development
// binary (no bundle) or a missing key.
inline std::string BundleInfoString(const char* key) {
#if XE_PLATFORM_MAC
  CFBundleRef bundle = CFBundleGetMainBundle();
  if (!bundle) {
    return {};
  }
  CFStringRef cf_key =
      CFStringCreateWithCString(nullptr, key, kCFStringEncodingUTF8);
  CFTypeRef value = CFBundleGetValueForInfoDictionaryKey(bundle, cf_key);
  CFRelease(cf_key);
  if (!value || CFGetTypeID(value) != CFStringGetTypeID()) {
    return {};
  }
  char buffer[128];
  if (!CFStringGetCString(static_cast<CFStringRef>(value), buffer,
                          sizeof(buffer), kCFStringEncodingUTF8)) {
    return {};
  }
  return buffer;
#else
  return {};
#endif  // XE_PLATFORM_MAC
}

// The version this copy is. A packaged app reports the
// CFBundleShortVersionString tools/build/package_mac.sh stamped; a bare
// binary the version its commit was built as (version.h, from the release
// tags: "1.0" on the macxenos-1.0 tag, "1.0+3" three commits later).
inline std::string Version() {
  std::string version = BundleInfoString("CFBundleShortVersionString");
  return version.empty() ? std::string(XE_BUILD_VERSION) : version;
}

// Whether Version() is a release rather than a development build (a commit
// past the last release tag, "+N", or an old "-dev" bundle).
inline bool IsReleaseVersion() {
  const std::string version = Version();
  return version.find('+') == std::string::npos &&
         version.find("-dev") == std::string::npos && version != "dev";
}

// The release's version, empty for a development build.
inline std::string ReleaseVersion() {
  return IsReleaseVersion() ? Version() : std::string();
}

// The release's build number (CFBundleVersion, the commit count Sparkle
// compares), empty outside a bundle.
inline std::string BundleBuildNumber() {
  return BundleInfoString("CFBundleVersion");
}

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_APP_VERSION_H_
