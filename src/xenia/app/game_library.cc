/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/game_library.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>

#include "third_party/fmt/include/fmt/format.h"
#include "third_party/tomlplusplus/toml.hpp"
#include "xenia/base/byte_order.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/string.h"
#include "xenia/kernel/util/xex2_info.h"
#include "xenia/vfs/devices/disc_image_device.h"
#include "xenia/vfs/devices/stfs_xbox.h"
#include "xenia/vfs/entry.h"
#include "xenia/vfs/file.h"

namespace xe {
namespace app {

namespace {

constexpr std::string_view kFoldersFile = "game_folders.toml";
constexpr std::string_view kIndexFile = "game_index.toml";
// Deep enough for Library/Publisher/Game/default.xex and for packages in
// Content/0000000000000000/TITLEID/000D0000/file; not so deep that pointing
// at a whole drive walks all of it.
constexpr int kMaxDepth = 5;

std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return text;
}

// "Disc 2" -> 2, else 0.
int DiscTag(std::string_view tag) {
  if (tag.size() < 6 || Lower(std::string(tag.substr(0, 5))) != "disc ") {
    return 0;
  }
  int disc = 0;
  for (char c : tag.substr(5)) {
    if (c < '0' || c > '9' || disc > 99) return 0;
    disc = disc * 10 + (c - '0');
  }
  return disc;
}

// "Just Cause 2 (USA, Europe) (En,Fr,De,Es,It)" -> "Just Cause 2": the
// region, language and revision tags disc image names carry. A "(Disc 2)"
// tag goes too, into `disc`, so every disc of a game has the same name.
std::string CleanName(std::string name, int* disc = nullptr) {
  for (;;) {
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) {
      name.pop_back();
    }
    if (name.empty() || (name.back() != ')' && name.back() != ']')) {
      break;
    }
    const char open = name.back() == ')' ? '(' : '[';
    const size_t start = name.rfind(open);
    if (start == std::string::npos || start == 0) {
      break;
    }
    const int tag_disc =
        DiscTag(std::string_view(name).substr(start + 1, name.size() - start - 2));
    if (disc && tag_disc) {
      *disc = tag_disc;
    }
    name.erase(start);
  }
  return name;
}

// The title ID from the start of an XEX2 file: its optional headers are
// plain (only the image behind them is encrypted).
uint32_t XexTitleId(const std::vector<uint8_t>& data) {
  auto be32 = [&](size_t offset) -> uint32_t {
    if (offset + 4 > data.size()) return 0;
    return xe::load_and_swap<uint32_t>(data.data() + offset);
  };
  if (data.size() < 0x18 || std::memcmp(data.data(), "XEX2", 4) != 0) {
    return 0;
  }
  const uint32_t count = std::min<uint32_t>(be32(0x14), 0x100);
  for (uint32_t i = 0; i < count; ++i) {
    const size_t entry = 0x18 + size_t(i) * 8;
    if (be32(entry) == xex2_header_keys::XEX_HEADER_EXECUTION_INFO) {
      const size_t info = be32(entry + 4);
      return be32(info + offsetof(xex2_opt_execution_info, title_id));
    }
  }
  return 0;
}

uint32_t DiscTitleId(const std::filesystem::path& path) {
  auto device = std::make_unique<vfs::DiscImageDevice>("\\Device\\Probe", path);
  if (!device->Initialize()) {
    return 0;
  }
  vfs::Entry* entry = device->ResolvePath("default.xex");
  if (!entry) {
    return 0;
  }
  vfs::File* file = nullptr;
  if (entry->Open(xe::filesystem::FileAccess::kGenericRead, &file) != 0 ||
      !file) {
    return 0;
  }
  std::vector<uint8_t> data(std::min<size_t>(entry->size(), 0x10000));
  size_t read = 0;
  file->ReadSync(data, 0, &read);
  file->Destroy();
  data.resize(read);
  return XexTitleId(data);
}

uint32_t XexFileTitleId(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<uint8_t> data(0x10000);
  file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size()));
  data.resize(size_t(file.gcount()));
  return XexTitleId(data);
}

bool IsPackageMagic(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  uint8_t magic[4] = {};
  if (!file.read(reinterpret_cast<char*>(magic), 4)) {
    return false;
  }
  const uint32_t value = xe::load_and_swap<uint32_t>(magic);
  return value == uint32_t(vfs::XContentPackageType::kCon) ||
         value == uint32_t(vfs::XContentPackageType::kLive) ||
         value == uint32_t(vfs::XContentPackageType::kPirs);
}

// A downloaded game's package header: whether it is a game at all, and its
// name, title ID and icon (saved beside the others in title_icons).
bool ReadPackage(const std::filesystem::path& path,
                 const std::filesystem::path& icon_folder, uint32_t* title_id,
                 std::string* name) {
  auto header = std::make_unique<vfs::XContentContainerHeader>();
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(header.get()), sizeof(*header))) {
    return false;
  }
  const auto& meta = header->content_metadata;
  switch (static_cast<XContentType>(meta.content_type)) {
    case XContentType::kArcadeTitle:
    case XContentType::kXbox360Title:  // Games on Demand
    case XContentType::kGameDemo:
    case XContentType::kGameTitle:
    case XContentType::kCommunityGame:
      break;
    default:
      return false;
  }
  *title_id = meta.execution_info.title_id;
  *name = xe::to_utf8(meta.display_name(XLanguage::kEnglish));
  if (name->empty()) {
    *name = xe::to_utf8(load_and_swap<std::u16string>(meta.title_name_raw.uint));
  }
  // The game's icon (title thumbnail), else the package's own.
  const uint8_t* png = meta.title_thumbnail;
  uint32_t png_size = meta.title_thumbnail_size;
  if (!png_size) {
    png = meta.thumbnail;
    png_size = meta.thumbnail_size;
  }
  png_size = std::min<uint32_t>(png_size, vfs::XContentMetadata::kThumbLengthV2);
  if (*title_id && png_size > 8 && png[0] == 0x89 && png[1] == 'P') {
    const std::filesystem::path icon =
        icon_folder / fmt::format("{:08X}.png", *title_id);
    std::error_code ec;
    if (!std::filesystem::exists(icon, ec)) {
      std::filesystem::create_directories(icon_folder, ec);
      std::ofstream out(icon, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char*>(png), png_size);
    }
  }
  return true;
}

int64_t ModifiedTime(const std::filesystem::path& path) {
  std::error_code ec;
  auto time = std::filesystem::last_write_time(path, ec);
  return ec ? 0 : int64_t(time.time_since_epoch().count());
}

}  // namespace

GameLibrary::GameLibrary(std::filesystem::path storage_root,
                         std::function<void()> on_changed)
    : storage_root_(std::move(storage_root)),
      on_changed_(std::move(on_changed)) {
  LoadFolders();
}

GameLibrary::~GameLibrary() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::vector<std::filesystem::path> GameLibrary::folders() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return folders_;
}

GameLibrary::ScanProgress GameLibrary::scan_progress() const {
  ScanProgress progress;
  progress.found = scan_found_;
  progress.checked = scan_checked_;
  progress.total = scan_total_;
  return progress;
}

void GameLibrary::ReportProgress(bool force) {
  const auto now = std::chrono::steady_clock::now();
  if (!force && now - last_report_ < std::chrono::milliseconds(250)) {
    return;
  }
  last_report_ = now;
  if (!stop_ && on_changed_) {
    on_changed_();
  }
}

std::vector<GameLibrary::Game> GameLibrary::games() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return games_;
}

void GameLibrary::AddFolder(const std::filesystem::path& folder) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(folders_.begin(), folders_.end(), folder) != folders_.end()) {
      return;
    }
    folders_.push_back(folder);
  }
  SaveFolders();
  Rescan();
}

void GameLibrary::RemoveFolder(const std::filesystem::path& folder) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    folders_.erase(std::remove(folders_.begin(), folders_.end(), folder),
                   folders_.end());
  }
  SaveFolders();
  Rescan();
}

void GameLibrary::LoadFolders() {
  std::ifstream file(storage_root_ / kFoldersFile);
  if (!file.is_open()) {
    return;
  }
  try {
    toml::table table = toml::parse(file);
    if (auto* list = table["folders"].as_array()) {
      for (const auto& item : *list) {
        if (auto value = item.value<std::string>()) {
          folders_.push_back(xe::to_path(*value));
        }
      }
    }
  } catch (const toml::parse_error& error) {
    XELOGE("Game folders: cannot read {}: {}", kFoldersFile, error.what());
  }
}

void GameLibrary::SaveFolders() const {
  toml::array list;
  for (const auto& folder : folders()) {
    list.push_back(xe::path_to_utf8(folder));
  }
  toml::table table;
  table.insert("folders", std::move(list));
  std::ofstream file(storage_root_ / kFoldersFile, std::ofstream::trunc);
  file << table;
}

void GameLibrary::Rescan() {
  if (scanning_.exchange(true)) {
    rescan_requested_ = true;
    return;
  }
  if (thread_.joinable()) {
    thread_.join();
  }
  thread_ = std::thread([this]() {
    do {
      rescan_requested_ = false;
      Scan(folders());
    } while (rescan_requested_ && !stop_);
    scanning_ = false;
    ReportProgress(true);
  });
}

void GameLibrary::Scan(std::vector<std::filesystem::path> folders) {
  const auto started = std::chrono::steady_clock::now();
  const std::filesystem::path icon_folder = storage_root_ / "title_icons";

  // What earlier scans learned, by path.
  std::map<std::string, IndexEntry> index;
  {
    std::ifstream file(storage_root_ / kIndexFile);
    if (file.is_open()) {
      try {
        toml::table table = toml::parse(file);
        if (auto* list = table["game"].as_array()) {
          for (const auto& item : *list) {
            const toml::table* entry = item.as_table();
            if (!entry) continue;
            auto path = (*entry)["path"].value<std::string>();
            if (!path) continue;
            IndexEntry value;
            value.size = uint64_t((*entry)["size"].value_or<int64_t>(0));
            value.mtime = (*entry)["mtime"].value_or<int64_t>(0);
            value.title_id = uint32_t((*entry)["title_id"].value_or<int64_t>(0));
            value.name = (*entry)["name"].value_or<std::string>("");
            index[*path] = std::move(value);
          }
        }
      } catch (const toml::parse_error&) {
        // Rebuilt below.
      }
    }
  }

  std::vector<Game> found;
  std::map<std::string, IndexEntry> next_index;
  size_t probed = 0;

  scan_found_ = 0;
  scan_checked_ = 0;
  scan_total_ = 0;
  ReportProgress(true);

  // First every game in the folders, then what each one is: a progress bar
  // needs the count, and reading a new disc image (over the network, often)
  // is the slow part.
  struct Candidate {
    std::filesystem::path path;
    std::string fallback;
    int disc = 0;
    int kind = 0;
  };
  std::vector<Candidate> candidates;
  auto listed = [&](std::filesystem::path path, const std::string& name,
                    int kind) {
    Candidate candidate;
    candidate.path = std::move(path);
    candidate.fallback = CleanName(name, &candidate.disc);
    candidate.kind = kind;
    candidates.push_back(std::move(candidate));
    scan_found_ = candidates.size();
    ReportProgress(false);
  };

  auto add = [&](const std::filesystem::path& path, const std::string& fallback,
                 int disc, int kind) {
    std::error_code ec;
    const std::string key = xe::path_to_utf8(path);
    IndexEntry entry;
    entry.size = std::filesystem::file_size(path, ec);
    entry.mtime = ModifiedTime(path);
    auto cached = index.find(key);
    if (cached != index.end() && cached->second.size == entry.size &&
        cached->second.mtime == entry.mtime) {
      entry = cached->second;
    } else {
      ++probed;
      if (kind == 0) {
        entry.title_id = DiscTitleId(path);
      } else if (kind == 1) {
        entry.title_id = XexFileTitleId(path);
      } else if (!ReadPackage(path, icon_folder, &entry.title_id, &entry.name)) {
        return;
      }
    }
    if (kind == 2 && entry.name.empty() && !entry.title_id) {
      return;
    }
    Game game;
    game.path = path;
    game.title_id = entry.title_id;
    game.name = entry.name.empty() ? fallback : entry.name;
    game.disc = disc;
    found.push_back(std::move(game));
    next_index[key] = std::move(entry);
  };

  for (const auto& folder : folders) {
    std::error_code ec;
    auto it = std::filesystem::recursive_directory_iterator(
        folder, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec) {
      XELOGW("Game folders: cannot read {}: {}", xe::path_to_utf8(folder),
             ec.message());
      continue;
    }
    for (; it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
      if (stop_ || ec) {
        break;
      }
      const std::filesystem::path& path = it->path();
      const std::string file_name = xe::path_to_utf8(path.filename());
      if (file_name.empty() || file_name[0] == '.') {
        if (it->is_directory(ec)) it.disable_recursion_pending();
        continue;
      }
      if (it->is_directory(ec)) {
        // An unpacked game: the folder is the game.
        const std::filesystem::path xex = path / "default.xex";
        if (std::filesystem::is_regular_file(xex, ec)) {
          listed(xex, file_name, 1);
          it.disable_recursion_pending();
        } else if (it.depth() >= kMaxDepth) {
          it.disable_recursion_pending();
        }
        continue;
      }
      if (!it->is_regular_file(ec)) {
        continue;
      }
      const std::string ext = Lower(xe::path_to_utf8(path.extension()));
      if (ext == ".iso") {
        listed(path, xe::path_to_utf8(path.stem()), 0);
      } else if (ext == ".zar") {
        listed(path, xe::path_to_utf8(path.stem()), 3);
      } else if (ext.empty() &&
                 it->file_size(ec) >= sizeof(vfs::XContentContainerHeader) &&
                 IsPackageMagic(path)) {
        listed(path, file_name, 2);
      }
    }
  }

  scan_total_ = candidates.size();
  ReportProgress(true);
  for (const Candidate& candidate : candidates) {
    if (stop_) {
      break;
    }
    add(candidate.path, candidate.fallback, candidate.disc, candidate.kind);
    ++scan_checked_;
    ReportProgress(false);
  }

  // Keep what was learned, for the next scan.
  toml::array list;
  for (const auto& [path, entry] : next_index) {
    toml::table item;
    item.insert("path", path);
    item.insert("size", int64_t(entry.size));
    item.insert("mtime", entry.mtime);
    item.insert("title_id", int64_t(entry.title_id));
    item.insert("name", entry.name);
    list.push_back(std::move(item));
  }
  toml::table table;
  table.insert("game", std::move(list));
  {
    std::ofstream file(storage_root_ / kIndexFile, std::ofstream::trunc);
    file << table;
  }

  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - started)
                      .count();
  XELOGI("Game folders: {} games in {} folders ({} read, {} ms)", found.size(),
         folders.size(), probed, ms);
  std::lock_guard<std::mutex> lock(mutex_);
  games_ = std::move(found);
}

}  // namespace app
}  // namespace xe
