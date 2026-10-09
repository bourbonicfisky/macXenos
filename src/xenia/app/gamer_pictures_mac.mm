/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/gamer_pictures_mac.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <system_error>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "third_party/tomlplusplus/toml.hpp"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/vfs/devices/xcontent_container_device.h"
#include "xenia/vfs/entry.h"
#include "xenia/vfs/file.h"

namespace xe {
namespace app {
namespace gamer_pictures {

const char* const kCollectionName = "Xbox 360 Profile Picture Collection";
const char* const kSite = "the Internet Archive";
const int64_t kDownloadBytes = 116904804;

namespace {

constexpr const char* kUrl =
    "https://archive.org/download/xbox-360-profile-picture-collection/"
    "Xbox%20360%20Profile%20Picture%20Collection.zip";
// As the archive publishes it (archive.org/metadata/...); a different file
// is refused.
constexpr const char* kSha1 = "729dea5880720fd533a23279e91a8e34aca6b1aa";
constexpr const char* kIndexFile = "gamer_pictures.toml";

std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return text;
}

std::string Sha1Hex(const std::filesystem::path& path) {
  CC_SHA1_CTX context;
  CC_SHA1_Init(&context);
  std::ifstream file(path, std::ios::binary);
  std::vector<char> chunk(1 << 20);
  while (file) {
    file.read(chunk.data(), std::streamsize(chunk.size()));
    if (file.gcount() > 0) {
      CC_SHA1_Update(&context, chunk.data(), CC_LONG(file.gcount()));
    }
  }
  unsigned char digest[CC_SHA1_DIGEST_LENGTH];
  CC_SHA1_Final(digest, &context);
  std::string out;
  for (unsigned char byte : digest) {
    out += fmt::format("{:02x}", byte);
  }
  return out;
}

// A PNG's width and height from its header, or 0x0 if it isn't one.
std::pair<uint32_t, uint32_t> PngSize(const std::vector<uint8_t>& data) {
  static constexpr uint8_t kSignature[] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (data.size() < 24 || std::memcmp(data.data(), kSignature, 8) != 0) {
    return {0, 0};
  }
  auto be32 = [&](size_t at) {
    return uint32_t(data[at]) << 24 | uint32_t(data[at + 1]) << 16 |
           uint32_t(data[at + 2]) << 8 | uint32_t(data[at + 3]);
  };
  return {be32(16), be32(20)};
}

NSString* ToNS(const std::filesystem::path& path) {
  return [NSString stringWithUTF8String:xe::path_to_utf8(path).c_str()];
}

// Unpacks the zip, then each gamer-picture package in it, into XAM's
// gamer-picture folders, and writes the index.
std::string Install(const std::filesystem::path& zip,
                    const std::filesystem::path& work,
                    const std::filesystem::path& storage_root,
                    const std::filesystem::path& content_root,
                    Summary* summary) {
  if (Sha1Hex(zip) != kSha1) {
    return "the download is not the expected collection";
  }
  std::error_code ec;
  const std::filesystem::path unpacked = work / "unpacked";
  std::filesystem::remove_all(unpacked, ec);
  std::filesystem::create_directories(unpacked, ec);
  NSTask* task = [[NSTask alloc] init];
  task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/ditto"];
  task.arguments = @[ @"-x", @"-k", ToNS(zip), ToNS(unpacked) ];
  if (![task launchAndReturnError:nil]) {
    return "could not unpack the download";
  }
  [task waitUntilExit];
  if (task.terminationStatus != 0) {
    return "the download is not a readable zip";
  }

  // XAM's folder for gamer pictures (xam_user.cc).
  const std::filesystem::path pictures_root =
      content_root / "0000000000000000" / "FFFE07D1" / "00020000";
  // 64_ or 32_, then title ID, big tile ID and small tile ID.
  const std::regex picture_name("^(64|32)_([0-9A-Fa-f]{8})([0-9A-Fa-f]{16})\\.png$");

  struct Title {
    std::string name;
    std::set<std::string> keys;
  };
  std::map<std::string, Title> titles;
  uint32_t packages = 0;
  uint32_t skipped = 0;

  for (auto it = std::filesystem::recursive_directory_iterator(unpacked, ec);
       it != std::filesystem::recursive_directory_iterator();
       it.increment(ec)) {
    if (ec) break;
    if (!it->is_regular_file(ec) || it->path().filename() != "Complete") {
      continue;
    }
    auto device = vfs::XContentContainerDevice::CreateContentDevice(
        "\\Device\\GamerPictures", it->path());
    if (!device || !device->Initialize()) {
      XELOGW("Gamer pictures: cannot open {}", xe::path_to_utf8(it->path()));
      continue;
    }
    // Device's ResolvePath is public; the container's override is not.
    vfs::Entry* root = static_cast<vfs::Device*>(device.get())->ResolvePath("");
    if (!root) continue;
    ++packages;
    // The game is the package's folder; the packages themselves all carry
    // the same display name.
    const std::string folder_name = xe::path_to_utf8(it->path().parent_path().filename());
    // Each picture is a pair: 64_<key> at 64x64 and 32_<key> at 32x32, as
    // the console stored a profile's picture (tile_64 / tile_32). Only
    // complete pairs of the right sizes are kept; the collection has a few
    // large ones that are really 32x32, and small ones with no partner.
    struct Pair {
      std::vector<uint8_t> big, small;
    };
    std::map<std::string, Pair> pairs;
    for (const auto& child : root->children()) {
      std::smatch match;
      const std::string& name = child->name();
      if (!std::regex_match(name, match, picture_name)) continue;
      vfs::File* file = nullptr;
      if (child->Open(xe::filesystem::FileAccess::kGenericRead, &file) != 0 ||
          !file) {
        continue;
      }
      std::vector<uint8_t> data(child->size());
      size_t read = 0;
      file->ReadSync(data, 0, &read);
      file->Destroy();
      if (read != data.size()) continue;
      const bool big = match[1] == "64";
      const uint32_t want = big ? 64 : 32;
      if (PngSize(data) != std::pair<uint32_t, uint32_t>(want, want)) {
        ++skipped;
        continue;
      }
      Pair& pair = pairs[Lower(match[2].str() + match[3].str())];
      (big ? pair.big : pair.small) = std::move(data);
    }
    for (auto& [key, pair] : pairs) {
      if (pair.big.empty() || pair.small.empty()) {
        ++skipped;
        continue;
      }
      const std::string title_id = key.substr(0, 8);
      const std::filesystem::path folder = pictures_root / title_id;
      std::filesystem::create_directories(folder, ec);
      const std::pair<const char*, const std::vector<uint8_t>*> sizes[] = {
          {"64", &pair.big}, {"32", &pair.small}};
      for (const auto& [prefix, data] : sizes) {
        std::ofstream out(folder / fmt::format("{}_{}.png", prefix, key),
                          std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data->data()),
                  std::streamsize(data->size()));
      }
      Title& title = titles[title_id];
      if (title.name.empty()) title.name = folder_name;
      title.keys.insert(key);
    }
  }
  std::filesystem::remove_all(unpacked, ec);
  if (titles.empty()) {
    return "no gamer pictures were found in the download";
  }

  toml::array list;
  for (const auto& [id, title] : titles) {
    toml::array keys;
    for (const auto& key : title.keys) keys.push_back(key);
    toml::table entry;
    entry.insert("id", id);
    entry.insert("name", title.name);
    entry.insert("pictures", std::move(keys));
    list.push_back(std::move(entry));
    summary->pictures += uint32_t(title.keys.size());
  }
  summary->titles = uint32_t(titles.size());
  toml::table index;
  index.insert("source", kUrl);
  index.insert("title", std::move(list));
  std::ofstream(storage_root / kIndexFile, std::ios::trunc) << index;
  XELOGI(
      "Gamer pictures: installed {} pictures from {} titles ({} packages; {} "
      "odd-sized or unpaired files left out)",
      summary->pictures, summary->titles, packages, skipped);
  return std::string();
}

}  // namespace

std::optional<Summary> Installed(const std::filesystem::path& storage_root) {
  std::ifstream file(storage_root / kIndexFile);
  if (!file.is_open()) return std::nullopt;
  try {
    toml::table index = toml::parse(file);
    Summary summary;
    if (auto* titles = index["title"].as_array()) {
      for (const auto& item : *titles) {
        const toml::table* title = item.as_table();
        if (!title) continue;
        ++summary.titles;
        if (auto* pictures = (*title)["pictures"].as_array()) {
          summary.pictures += uint32_t(pictures->size());
        }
      }
    }
    return summary.titles ? std::optional<Summary>(summary) : std::nullopt;
  } catch (const toml::parse_error&) {
    return std::nullopt;
  }
}

}  // namespace gamer_pictures
}  // namespace app
}  // namespace xe

// The download's delegate: reports progress, keeps the file NSURLSession
// hands over, and installs it once the transfer ends.
@interface XeniaGamerPicturesDownload : NSObject <NSURLSessionDownloadDelegate> {
 @public
  std::filesystem::path storage_root;
  std::filesystem::path content_root;
  std::filesystem::path work;
  std::shared_ptr<xe::app::gamer_pictures::Progress> progress;
  std::function<void(std::string, xe::app::gamer_pictures::Summary)> done;
  std::string failure;
  bool kept;
}
@end

@implementation XeniaGamerPicturesDownload
- (void)URLSession:(NSURLSession*)session
                 downloadTask:(NSURLSessionDownloadTask*)task
                 didWriteData:(int64_t)written
            totalBytesWritten:(int64_t)total_written
    totalBytesExpectedToWrite:(int64_t)expected {
  progress->received = total_written;
  progress->total = expected > 0 ? expected : xe::app::gamer_pictures::kDownloadBytes;
}
- (void)URLSession:(NSURLSession*)session
                 downloadTask:(NSURLSessionDownloadTask*)task
    didFinishDownloadingToURL:(NSURL*)location {
  // The file is gone once this returns: move it now.
  NSInteger status = [task.response isKindOfClass:NSHTTPURLResponse.class]
                         ? ((NSHTTPURLResponse*)task.response).statusCode
                         : 0;
  if (status != 200) {
    failure = "the server answered " + std::to_string(status);
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(work, ec);
  const std::filesystem::path zip = work / "collection.zip";
  std::filesystem::remove(zip, ec);
  NSError* error = nil;
  if ([NSFileManager.defaultManager moveItemAtURL:location
                                            toURL:[NSURL fileURLWithPath:xe::app::gamer_pictures::ToNS(zip)]
                                            error:&error]) {
    kept = true;
  } else {
    failure = "could not keep the download";
  }
}
- (void)URLSession:(NSURLSession*)session
                    task:(NSURLSessionTask*)task
    didCompleteWithError:(NSError*)error {
  if (error) {
    failure = error.localizedDescription.UTF8String;
  }
  XeniaGamerPicturesDownload* me = self;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
    xe::app::gamer_pictures::Summary summary;
    std::string result = me->failure;
    if (result.empty() && !me->kept) result = "the download did not arrive";
    if (result.empty()) {
      me->progress->installing = true;
      result = xe::app::gamer_pictures::Install(
          me->work / "collection.zip", me->work, me->storage_root,
          me->content_root, &summary);
    }
    std::error_code ec;
    std::filesystem::remove_all(me->work, ec);
    me->progress->installing = false;
    if (!result.empty()) XELOGW("Gamer pictures: {}", result);
    me->done(result, summary);
  });
}
@end

namespace xe {
namespace app {
namespace gamer_pictures {

void Download(const std::filesystem::path& storage_root,
              const std::filesystem::path& content_root,
              std::shared_ptr<Progress> progress,
              std::function<void(std::string error, Summary summary)> done) {
  progress->received = 0;
  progress->total = kDownloadBytes;
  progress->installing = false;
  XeniaGamerPicturesDownload* delegate = [[XeniaGamerPicturesDownload alloc] init];
  delegate->storage_root = storage_root;
  delegate->content_root = content_root;
  delegate->work = storage_root / "gamer_pictures.download";
  delegate->progress = progress;
  delegate->done = std::move(done);
  delegate->kept = false;
  NSURLSession* session =
      [NSURLSession sessionWithConfiguration:NSURLSessionConfiguration.defaultSessionConfiguration
                                    delegate:delegate
                               delegateQueue:nil];
  XELOGI("Gamer pictures: downloading {}", kUrl);
  [[session downloadTaskWithURL:[NSURL URLWithString:@(kUrl)]] resume];
  // The session keeps the delegate until the task ends.
  [session finishTasksAndInvalidate];
}

}  // namespace gamer_pictures
}  // namespace app
}  // namespace xe
