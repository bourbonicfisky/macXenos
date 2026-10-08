/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/dashboard_sounds_mac.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>

#include <system_error>

#include "xenia/base/logging.h"
#include "xenia/base/filesystem.h"

namespace xe {
namespace app {
namespace dashboard_sounds {

const char* const kPackName = "Xbox 360 Blades";
const char* const kPackAuthor = "squabbled";
const char* const kPackSite = "DeckThemes";

namespace {

// The pack's v1.0 zip (theme c88e382d-a2cc-4f38-b963-403c0fac4b47). A changed
// file is refused rather than trusted: the mapping below is by ear.
constexpr const char* kPackUrl =
    "https://api.deckthemes.com/blobs/02d9cba0-64ce-4b16-81cd-01bdb300b5e1";
constexpr const char* kPackSha256 =
    "8c095eec48e51f68789d2b450d00e84910598432a94f2a2be27a75eab08947c1";
constexpr const char* kPackFolder = "Xbox 360 Blades";

// The pack holds eight distinct sounds under Steam Deck names; these are the
// ones the dashboard uses, renamed for what they do here (sound.ts FILES).
struct Mapping {
  const char* pack_file;
  const char* name;
};
constexpr Mapping kMappings[] = {
    {"deck_ui_navigation.wav", "focus.wav"},
    {"deck_ui_default_activation.wav", "select.wav"},
    {"deck_ui_bumper_end_02.wav", "back.wav"},
    {"deck_ui_tab_transition_01.wav", "blade.wav"},
    {"deck_ui_toast.wav", "notify.wav"},
    {"deck_ui_side_menu_fly_in.wav", "toggle.wav"},
};

std::string Sha256Hex(NSData* data) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.bytes, CC_LONG(data.length), digest);
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  for (unsigned char byte : digest) {
    out.push_back(kHex[byte >> 4]);
    out.push_back(kHex[byte & 15]);
  }
  return out;
}

NSString* ToNS(const std::filesystem::path& path) {
  return [NSString stringWithUTF8String:xe::path_to_utf8(path).c_str()];
}

// Unpacks `zip` and installs the mapped sounds into `folder`, replacing what
// was there only once every file is in place.
std::string Install(NSData* zip, const std::filesystem::path& folder) {
  std::error_code ec;
  const std::filesystem::path parent = folder.parent_path();
  const std::filesystem::path work =
      parent / (folder.filename().string() + ".download");
  std::filesystem::remove_all(work, ec);
  std::filesystem::create_directories(work / "unpacked", ec);
  if (ec) {
    return "could not create " + xe::path_to_utf8(work);
  }
  const std::filesystem::path zip_path = work / "pack.zip";
  if (![zip writeToFile:ToNS(zip_path) atomically:YES]) {
    return "could not save the download";
  }

  NSTask* task = [[NSTask alloc] init];
  task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/ditto"];
  task.arguments = @[ @"-x", @"-k", ToNS(zip_path), ToNS(work / "unpacked") ];
  NSError* error = nil;
  if (![task launchAndReturnError:&error]) {
    return "could not unpack the download";
  }
  [task waitUntilExit];
  if (task.terminationStatus != 0) {
    return "the download is not a readable zip";
  }

  const std::filesystem::path staged = work / "sounds";
  std::filesystem::create_directories(staged, ec);
  for (const Mapping& mapping : kMappings) {
    std::filesystem::copy_file(
        work / "unpacked" / kPackFolder / mapping.pack_file,
        staged / mapping.name,
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      return std::string("the pack has no ") + mapping.pack_file;
    }
  }

  std::filesystem::remove_all(folder, ec);
  std::filesystem::rename(staged, folder, ec);
  if (ec) {
    return "could not install into " + xe::path_to_utf8(folder);
  }
  std::filesystem::remove_all(work, ec);
  return std::string();
}

}  // namespace

bool Present(const std::filesystem::path& folder) {
  std::error_code ec;
  for (const Mapping& mapping : kMappings) {
    if (!std::filesystem::is_regular_file(folder / mapping.name, ec)) {
      return false;
    }
  }
  return true;
}

void Download(const std::filesystem::path& folder,
              std::function<void(std::string error)> done) {
  NSURL* url = [NSURL URLWithString:@(kPackUrl)];
  XELOGI("Dashboard sounds: downloading {}", kPackUrl);
  auto callback =
      std::make_shared<std::function<void(std::string)>>(std::move(done));
  // A block captures a C++ reference as a reference; take a copy for it.
  const std::filesystem::path target = folder;
  NSURLSessionDataTask* task = [NSURLSession.sharedSession
        dataTaskWithURL:url
      completionHandler:^(NSData* data, NSURLResponse* response,
                          NSError* error) {
        std::string failure;
        NSInteger status = [response isKindOfClass:NSHTTPURLResponse.class]
                               ? ((NSHTTPURLResponse*)response).statusCode
                               : 0;
        if (error) {
          failure = error.localizedDescription.UTF8String;
        } else if (status != 200 || !data) {
          failure = "the server answered " + std::to_string(status);
        } else if (Sha256Hex(data) != kPackSha256) {
          failure = "the download is not the expected pack";
        } else {
          failure = Install(data, target);
        }
        if (failure.empty()) {
          XELOGI("Dashboard sounds: installed in {}", xe::path_to_utf8(target));
        } else {
          XELOGW("Dashboard sounds: download failed: {}", failure);
        }
        (*callback)(failure);
      }];
  [task resume];
}

}  // namespace dashboard_sounds
}  // namespace app
}  // namespace xe
