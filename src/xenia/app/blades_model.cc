/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/blades_model.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <map>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "third_party/tomlplusplus/toml.hpp"
#include "xenia/app/emulator_window.h"
#include "xenia/app/game_library.h"
#if XE_PLATFORM_MAC
#include "xenia/app/controller_home_button_mac.h"
#include "xenia/app/blades_overlay_mac.h"
#include "xenia/app/dashboard_sounds_mac.h"
#include "xenia/app/gamer_pictures_mac.h"
#endif
#include "xenia/apu/audio_media_player.h"
#include "xenia/base/string_util.h"
#include "xenia/base/system.h"
#include "xenia/kernel/xam/profile_manager.h"
#include "xenia/kernel/xam/user_tracker.h"
#include "xenia/kernel/xam/xam_state.h"
#include "xenia/base/utf8.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/config.h"
#include "xenia/emulator.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xconfig.h"
#include "xenia/ui/presenter.h"

#include "version.h"

DECLARE_bool(enable_xmp);
DECLARE_bool(dashboard_sounds_asked);
DECLARE_string(preferences_button);
DECLARE_string(gpu);

namespace xe {
namespace app {

namespace {

// Shown in Preferences (Dashboard > About). Raise with each release.
constexpr const char* kReleaseName = "1.0 beta 3";

using Writer = rapidjson::Writer<rapidjson::StringBuffer>;

}  // namespace

// The state's JSON writer, named so blades_model.h can forward-declare it.
class StateWriter : public Writer {
 public:
  using Writer::Writer;
};

namespace {

// --- Config settings ---------------------------------------------------------

// Writes one setting's type, value and default. False when the setting is not
// of type T (the caller tries each type in turn).
template <typename T>
bool WriteTyped(cvar::IConfigVar* var, Writer& w, const char* type) {
  auto* typed = dynamic_cast<cvar::ConfigVar<T>*>(var);
  if (!typed) {
    return false;
  }
  auto write = [&w](const T& value) {
    if constexpr (std::is_same_v<T, bool>) {
      w.Bool(value);
    } else if constexpr (std::is_same_v<T, double>) {
      w.Double(std::isfinite(value) ? value : 0.0);
    } else if constexpr (std::is_same_v<T, int32_t> ||
                         std::is_same_v<T, int64_t>) {
      w.Int64(int64_t(value));
    } else if constexpr (std::is_same_v<T, uint32_t> ||
                         std::is_same_v<T, uint64_t>) {
      w.Uint64(uint64_t(value));
    } else if constexpr (std::is_same_v<T, std::string>) {
      w.String(value.c_str(), rapidjson::SizeType(value.size()));
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
      std::string utf8 = xe::path_to_utf8(value);
      w.String(utf8.c_str(), rapidjson::SizeType(utf8.size()));
    }
  };
  w.Key("type");
  w.String(type);
  w.Key("value");
  write(*typed->current_value());
  w.Key("defaultValue");
  write(typed->default_value());
  return true;
}

void WriteSetting(cvar::IConfigVar* var, Writer& w) {
  if (WriteTyped<bool>(var, w, "bool") ||
      WriteTyped<int32_t>(var, w, "int") ||
      WriteTyped<int64_t>(var, w, "int") ||
      WriteTyped<uint32_t>(var, w, "uint") ||
      WriteTyped<uint64_t>(var, w, "uint") ||
      WriteTyped<double>(var, w, "float") ||
      WriteTyped<std::string>(var, w, "string") ||
      WriteTyped<std::filesystem::path>(var, w, "path")) {
    return;
  }
  w.Key("type");
  w.String("string");
  w.Key("value");
  std::string value = var->config_value();
  w.String(value.c_str());
  w.Key("defaultValue");
  w.String(value.c_str());
}

// Sets one setting from the page's JSON value. False when the value does not
// fit the setting's type.
template <typename T>
bool SetTyped(cvar::IConfigVar* var, const rapidjson::Value& v, bool* handled) {
  auto* typed = dynamic_cast<cvar::ConfigVar<T>*>(var);
  if (!typed) {
    return false;
  }
  *handled = true;
  if constexpr (std::is_same_v<T, bool>) {
    if (!v.IsBool()) return false;
    typed->OverrideConfigValue(v.GetBool());
  } else if constexpr (std::is_same_v<T, double>) {
    if (!v.IsNumber()) return false;
    typed->OverrideConfigValue(v.GetDouble());
  } else if constexpr (std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, int64_t>) {
    if (!v.IsNumber()) return false;
    typed->OverrideConfigValue(T(v.IsInt64() ? v.GetInt64()
                                             : int64_t(v.GetDouble())));
  } else if constexpr (std::is_same_v<T, uint32_t> ||
                       std::is_same_v<T, uint64_t>) {
    if (!v.IsNumber()) return false;
    double d = v.GetDouble();
    if (d < 0) return false;
    typed->OverrideConfigValue(T(v.IsUint64() ? v.GetUint64() : uint64_t(d)));
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (!v.IsString()) return false;
    typed->OverrideConfigValue(std::string(v.GetString(), v.GetStringLength()));
  } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
    if (!v.IsString()) return false;
    typed->OverrideConfigValue(
        xe::to_path(std::string_view(v.GetString(), v.GetStringLength())));
  }
  return true;
}

bool SetSetting(cvar::IConfigVar* var, const rapidjson::Value& v) {
  bool handled = false;
  bool ok = SetTyped<bool>(var, v, &handled);
  if (!handled) ok = SetTyped<int32_t>(var, v, &handled);
  if (!handled) ok = SetTyped<int64_t>(var, v, &handled);
  if (!handled) ok = SetTyped<uint32_t>(var, v, &handled);
  if (!handled) ok = SetTyped<uint64_t>(var, v, &handled);
  if (!handled) ok = SetTyped<double>(var, v, &handled);
  if (!handled) ok = SetTyped<std::string>(var, v, &handled);
  if (!handled) ok = SetTyped<std::filesystem::path>(var, v, &handled);
  return handled && ok;
}

cvar::IConfigVar* FindSetting(const std::string& name) {
  if (!cvar::ConfigVars) {
    return nullptr;
  }
  auto it = cvar::ConfigVars->find(name);
  if (it == cvar::ConfigVars->end() || it->second->is_transient()) {
    return nullptr;
  }
  return it->second;
}

// Kept by "Reset to defaults" (Advanced): who signs in at boot, where saves
// and caches live, and the config file's own version stamp. Resetting those
// would sign people out or lose their data folders, not restore behaviour.
bool KeptByResetAll(const cvar::IConfigVar* var) {
  const std::string& category = var->category();
  const std::string& name = var->name();
  return category == "Profiles" || category == "Config" ||
         name == "storage_root" || name == "content_root" ||
         name == "cache_root";
}

// Settings the presenter reads once and must be handed again after a change.
bool IsDisplaySetting(const std::string& name) {
  return name.rfind("postprocess_", 0) == 0 || name.rfind("present_", 0) == 0;
}

// --- The console's own settings (XConfig) ------------------------------------

const std::map<uint32_t, const char*> kLanguages = {
    {1, "English"},    {2, "Japanese"},  {3, "German"},
    {4, "French"},     {5, "Spanish"},   {6, "Italian"},
    {7, "Korean"},     {8, "Traditional Chinese"},
    {9, "Portuguese"}, {11, "Polish"},   {12, "Russian"},
    {13, "Swedish"},   {14, "Turkish"},  {15, "Norwegian"},
    {16, "Dutch"},     {17, "Simplified Chinese"}};

const std::map<uint32_t, const char*> kCountries = {
    {1, "AE"},   {2, "AL"},   {3, "AM"},   {4, "AR"},   {5, "AT"},
    {6, "AU"},   {7, "AZ"},   {8, "BE"},   {9, "BG"},   {10, "BH"},
    {11, "BN"},  {12, "BO"},  {13, "BR"},  {14, "BY"},  {15, "BZ"},
    {16, "CA"},  {18, "CH"},  {19, "CL"},  {20, "CN"},  {21, "CO"},
    {22, "CR"},  {23, "CZ"},  {24, "DE"},  {25, "DK"},  {26, "DO"},
    {27, "DZ"},  {28, "EC"},  {29, "EE"},  {30, "EG"},  {31, "ES"},
    {32, "FI"},  {33, "FO"},  {34, "FR"},  {35, "GB"},  {36, "GE"},
    {37, "GR"},  {38, "GT"},  {39, "HK"},  {40, "HN"},  {41, "HR"},
    {42, "HU"},  {43, "ID"},  {44, "IE"},  {45, "IL"},  {46, "IN"},
    {47, "IQ"},  {48, "IR"},  {49, "IS"},  {50, "IT"},  {51, "JM"},
    {52, "JO"},  {53, "JP"},  {54, "KE"},  {55, "KG"},  {56, "KR"},
    {57, "KW"},  {58, "KZ"},  {59, "LB"},  {60, "LI"},  {61, "LT"},
    {62, "LU"},  {63, "LV"},  {64, "LY"},  {65, "MA"},  {66, "MC"},
    {67, "MK"},  {68, "MN"},  {69, "MO"},  {70, "MV"},  {71, "MX"},
    {72, "MY"},  {73, "NI"},  {74, "NL"},  {75, "NO"},  {76, "NZ"},
    {77, "OM"},  {78, "PA"},  {79, "PE"},  {80, "PH"},  {81, "PK"},
    {82, "PL"},  {83, "PR"},  {84, "PT"},  {85, "PY"},  {86, "QA"},
    {87, "RO"},  {88, "RU"},  {89, "SA"},  {90, "SE"},  {91, "SG"},
    {92, "SI"},  {93, "SK"},  {95, "SV"},  {96, "SY"},  {97, "TH"},
    {98, "TN"},  {99, "TR"},  {100, "TT"}, {101, "TW"}, {102, "UA"},
    {103, "US"}, {104, "UY"}, {105, "UZ"}, {106, "VE"}, {107, "VN"},
    {108, "YE"}, {109, "ZA"},
};

const std::map<uint32_t, const char*> kAvRegions = {{0x00400100, "NTSC"},
                                                    {0x00400200, "NTSC-J"},
                                                    {0x00400400, "PAL"},
                                                    {0x00800300, "PAL 50Hz"}};

// A flag in one of the XConfig flag words, shown as a checkbox.
struct ConsoleFlag {
  const char* id;
  const char* label;
  enum class Word { kRetail, kAudio, kVideo, kParental } word;
  uint32_t mask;
};

const std::array<ConsoleFlag, 12> kConsoleFlags = {{
    {"dst_off", "Turn off daylight saving time", ConsoleFlag::Word::kRetail,
     uint32_t(kernel::X_RETAIL_FLAGS::DSTOff)},
    {"clock_24h", "24-hour clock", ConsoleFlag::Word::kRetail,
     uint32_t(kernel::X_RETAIL_FLAGS::TwentyFourHourClock)},
    {"dashboard_initialized", "Dashboard set up", ConsoleFlag::Word::kRetail,
     uint32_t(kernel::X_RETAIL_FLAGS::DashboardInitialized)},
    {"iptv_initialized", "IPTV set up", ConsoleFlag::Word::kRetail,
     uint32_t(kernel::X_RETAIL_FLAGS::IPTVEnabled)},
    {"dvr_initialized", "DVR set up", ConsoleFlag::Word::kRetail,
     uint32_t(kernel::X_RETAIL_FLAGS::IPTVDVREnabled)},
    {"kinect_initialized", "Kinect set up", ConsoleFlag::Word::kRetail,
     uint32_t(kernel::X_RETAIL_FLAGS::KinectInitialized)},
    {"widescreen", "Widescreen", ConsoleFlag::Word::kVideo,
     uint32_t(kernel::X_VIDEO_FLAGS::Widescreen)},
    {"audio_mono", "Mono", ConsoleFlag::Word::kAudio,
     uint32_t(kernel::X_AUDIO_FLAGS::AnalogMono)},
    {"dolby_pro_logic", "Dolby Pro Logic", ConsoleFlag::Word::kAudio,
     uint32_t(kernel::X_AUDIO_FLAGS::DolbyProLogic)},
    {"dolby_digital", "Dolby Digital", ConsoleFlag::Word::kAudio,
     uint32_t(kernel::X_AUDIO_FLAGS::DolbyDigital)},
    {"dolby_digital_wma_pro", "Dolby Digital with WMA Pro",
     ConsoleFlag::Word::kAudio,
     uint32_t(kernel::X_AUDIO_FLAGS::DolbyDigitalWithWMAPRO)},
    {"parental_control", "Parental controls", ConsoleFlag::Word::kParental,
     uint32_t(kernel::X_PC_FLAGS::PCEnabled)},
}};

uint32_t FlagWord(const kernel::XConfigData& data, ConsoleFlag::Word word) {
  switch (word) {
    case ConsoleFlag::Word::kRetail:
      return data.user.retail_flags;
    case ConsoleFlag::Word::kAudio:
      return data.user.audio_flags;
    case ConsoleFlag::Word::kVideo:
      return data.user.video_flags;
    case ConsoleFlag::Word::kParental:
      return data.user.parental_control_flags;
  }
  return 0;
}

void SetFlag(kernel::XConfigData& data, const ConsoleFlag& flag, bool on) {
  auto apply = [&](uint32_t word) { return on ? word | flag.mask : word & ~flag.mask; };
  switch (flag.word) {
    case ConsoleFlag::Word::kRetail:
      data.user.retail_flags = apply(data.user.retail_flags);
      break;
    case ConsoleFlag::Word::kAudio:
      data.user.audio_flags = apply(data.user.audio_flags);
      break;
    case ConsoleFlag::Word::kVideo:
      data.user.video_flags = apply(data.user.video_flags);
      break;
    case ConsoleFlag::Word::kParental:
      data.user.parental_control_flags =
          uint8_t(apply(data.user.parental_control_flags));
      break;
  }
}

// The time-zone table index for the console's current zone; GMT if unknown.
size_t TimeZoneIndex(const kernel::XConfigData& data) {
  const kernel::TimeZone current = {
      data.user.time_zone_bias, data.user.tz_std_name, data.user.tz_dlt_name,
      data.user.tz_std_date,    data.user.tz_dlt_date, data.user.tz_std_bias,
      data.user.tz_dlt_bias};
  auto it = std::find(kernel::kTimezones.cbegin(), kernel::kTimezones.cend(),
                      current);
  return it != kernel::kTimezones.cend()
             ? size_t(std::distance(kernel::kTimezones.cbegin(), it))
             : size_t(0x19);
}

std::string HexBytes(const uint8_t* data, size_t size, char separator) {
  std::string out;
  for (size_t i = 0; i < size; ++i) {
    if (i && separator) out.push_back(separator);
    out += fmt::format("{:02X}", data[i]);
  }
  return out;
}

void WriteChoice(Writer& w, const char* id, const char* label, uint32_t value,
                 const std::map<uint32_t, const char*>& options) {
  w.StartObject();
  w.Key("id");
  w.String(id);
  w.Key("label");
  w.String(label);
  w.Key("type");
  w.String("choice");
  w.Key("value");
  w.Uint(value);
  w.Key("options");
  w.StartArray();
  for (const auto& [option, name] : options) {
    w.StartObject();
    w.Key("value");
    w.Uint(option);
    w.Key("label");
    w.String(name);
    w.EndObject();
  }
  w.EndArray();
  w.EndObject();
}

// --- Controllers --------------------------------------------------------------

const char* ControllerKindName(hid::ControllerKind kind) {
  switch (kind) {
    case hid::ControllerKind::kXbox:
      return "Xbox";
    case hid::ControllerKind::kPlayStation:
      return "PlayStation";
    case hid::ControllerKind::kNintendo:
      return "Nintendo";
    case hid::ControllerKind::kOther:
      return "Other";
    default:
      return "Unknown";
  }
}

struct Rebindable {
  const char* output;
  const char* label;
};
// Every button of the pad, in the order the page lists them (SDL's names).
constexpr Rebindable kRebindable[] = {
    {"a", "A / Cross"},
    {"b", "B / Circle"},
    {"x", "X / Square"},
    {"y", "Y / Triangle"},
    {"leftshoulder", "Left bumper (LB / L1)"},
    {"rightshoulder", "Right bumper (RB / R1)"},
    {"lefttrigger", "Left trigger (LT / L2)"},
    {"righttrigger", "Right trigger (RT / R2)"},
    {"back", "Back / Share"},
    {"start", "Start / Options"},
    {"guide", "Guide / Home"},
    {"leftstick", "Left stick click (LS / L3)"},
    {"rightstick", "Right stick click (RS / R3)"},
    {"dpup", "D-pad up"},
    {"dpdown", "D-pad down"},
    {"dpleft", "D-pad left"},
    {"dpright", "D-pad right"},
};

int FirstPresentSlot(hid::InputSystem* input) {
  for (uint32_t slot = 0; slot < XUserMaxUserCount; ++slot) {
    if (input->GetControllerIdentity(slot).present) {
      return int(slot);
    }
  }
  return -1;
}

}  // namespace

BladesModel::BladesModel(EmulatorWindow& window) : window_(window) {}

// --- The state, section by section -------------------------------------------
// BuildStateJson writes one object; each of these writes its own keys of it,
// in the order native.ts declares NativeState.

void BladesModel::WriteRenderer(StateWriter& w) {
  Emulator* emulator = window_.emulator();
  // Renderer: the one running, and the one `gpu` picks for the next launch.
  const uint32_t active = window_.ActiveRendererMask();
  w.Key("renderer");
  w.String(active == EmulatorWindow::kRendererMetal    ? "metal"
           : active == EmulatorWindow::kRendererVulkan ? "vulkan"
                                                       : "none");
  w.Key("selectedRenderer");
#if XE_GPU_METAL
  w.String(cvars::gpu == "vulkan" ? "vulkan" : "metal");
#else
  w.String(cvars::gpu == "vulkan" || cvars::gpu == "any" ? "vulkan" : "none");
#endif

  const bool title_open = emulator && emulator->is_title_open();
  w.Key("titleOpen");
  w.Bool(title_open);
  w.Key("titleName");
  w.String(title_open ? emulator->title_name().c_str() : "");

  w.Key("upscalers");
  w.StartArray();
  w.String("bilinear");
  w.String("cas");
  w.String("fsr");
  gpu::GraphicsSystem* graphics =
      emulator ? emulator->graphics_system() : nullptr;
  ui::Presenter* presenter = graphics ? graphics->presenter() : nullptr;
  if (presenter &&
      presenter->SupportsGuestOutputPaintEffect(
          ui::Presenter::GuestOutputPaintConfig::Effect::kMetalFxSpatial)) {
    w.String("metalfx");
  }
  w.EndArray();
}

void BladesModel::WriteSettings(StateWriter& w) {
  // Every config-file setting. The first state taken is the baseline the
  // page's "changed" (and so "applies next launch") is measured against.
  const bool first = baseline_.empty();
  w.Key("cvars");
  w.StartObject();
  if (cvar::ConfigVars) {
    for (const auto& [name, var] : *cvar::ConfigVars) {
      if (var->is_transient()) {
        continue;
      }
      std::string saved = var->config_value();
      if (first) {
        baseline_[name] = saved;
      }
      w.Key(name.c_str());
      w.StartObject();
      w.Key("name");
      w.String(name.c_str());
      w.Key("category");
      w.String(var->category().c_str());
      w.Key("description");
      w.String(var->description().c_str());
      WriteSetting(var, w);
      w.Key("changed");
      auto it = baseline_.find(name);
      w.Bool(it != baseline_.end() && it->second != saved);
      w.EndObject();
    }
  }
  w.EndObject();
}

void BladesModel::WriteControllers(StateWriter& w) {
  Emulator* emulator = window_.emulator();
  // Controllers, their remapping, and the controller hotkeys.
  hid::InputSystem* input = emulator ? emulator->input_system() : nullptr;
  w.Key("controllers");
  w.StartArray();
  int rebind_slot = -1;
  std::string rebind_name;
  if (input) {
    auto lock = input->lock();
    for (uint32_t slot = 0; slot < XUserMaxUserCount; ++slot) {
      hid::ControllerIdentity identity = input->GetControllerIdentity(slot);
      w.StartObject();
      w.Key("slot");
      w.Uint(slot);
      w.Key("present");
      w.Bool(identity.present);
      w.Key("kind");
      w.String(ControllerKindName(identity.kind));
      w.Key("name");
      w.String(identity.name.c_str());
      w.EndObject();
      if (identity.present && rebind_slot < 0) {
        rebind_slot = int(slot);
        rebind_name = identity.name.empty() ? ControllerKindName(identity.kind)
                                            : identity.name;
      }
    }
  }
  w.EndArray();

  w.Key("rebind");
  w.StartObject();
  w.Key("slot");
  w.Int(rebind_slot);
  w.Key("name");
  w.String(rebind_name.c_str());
  w.Key("capturing");
  w.String(capturing_output_.c_str());
  w.Key("outputs");
  w.StartArray();
  if (input && rebind_slot >= 0) {
    auto lock = input->lock();
    for (const Rebindable& r : kRebindable) {
      std::string binding =
          input->GetControllerOutputBinding(uint32_t(rebind_slot), r.output);
      w.StartObject();
      w.Key("output");
      w.String(r.output);
      w.Key("label");
      w.String(r.label);
      w.Key("binding");
      w.String(binding.c_str());
      w.EndObject();
    }
  }
  w.EndArray();
  w.EndObject();

  w.Key("hotkeys");
  w.StartArray();
  for (const auto& hotkey : window_.GetControllerHotkeys()) {
    w.StartObject();
    w.Key("id");
    w.Int(hotkey.function);
    w.Key("label");
    w.String(hotkey.description.c_str());
    w.Key("enabled");
    w.Bool(hotkey.enabled);
    w.Key("inGame");
    w.Bool(hotkey.in_game);
    w.EndObject();
  }
  w.EndArray();
}

void BladesModel::WriteConsole(StateWriter& w) {
  Emulator* emulator = window_.emulator();
  const bool title_open = emulator && emulator->is_title_open();
  kernel::KernelState* kernel = emulator ? emulator->kernel_state() : nullptr;
  // The console's own settings.
  w.Key("console");
  w.StartObject();
  w.Key("editable");
  w.Bool(!title_open);
  w.Key("fields");
  w.StartArray();
  if (kernel && kernel->xconfig()) {
    const kernel::XConfigData& data = *kernel->xconfig()->GetXConfig();
    WriteChoice(w, "language", "Language", data.user.language, kLanguages);
    WriteChoice(w, "country", "Country", data.user.country, kCountries);
    {
      std::map<uint32_t, const char*> zones;
      for (size_t i = 0; i < kernel::kTimezones.size(); ++i) {
        zones[uint32_t(i)] = kernel::kTimezones[i].name.c_str();
      }
      WriteChoice(w, "time_zone", "Time zone", uint32_t(TimeZoneIndex(data)),
                  zones);
    }
    WriteChoice(w, "av_region", "Region", data.secured.av_region, kAvRegions);
    {
      std::map<uint32_t, const char*> resolutions;
      for (const auto& r : kernel::XVGAResolution) {
        resolutions[r.to_host()] = r.name_.c_str();
      }
      WriteChoice(w, "resolution", "Resolution",
                  uint32_t(int32_t(data.user.av_pack_hdmi_sz)), resolutions);
    }
    for (const ConsoleFlag& flag : kConsoleFlags) {
      w.StartObject();
      w.Key("id");
      w.String(flag.id);
      w.Key("label");
      w.String(flag.label);
      w.Key("type");
      w.String("bool");
      w.Key("value");
      w.Bool((FlagWord(data, flag.word) & flag.mask) != 0);
      w.EndObject();
    }
    w.StartObject();
    w.Key("id");
    w.String("music_volume");
    w.Key("label");
    w.String("Music player volume");
    w.Key("type");
    w.String("float");
    w.Key("value");
    w.Double(double(float(data.user.music_volume)));
    w.EndObject();
    for (const auto& [id, label, text] :
         {std::tuple<const char*, const char*, std::string>{
              "mac_address", "MAC address",
              HexBytes(data.secured.mac_address.data(),
                       data.secured.mac_address.size(), ':')},
          std::tuple<const char*, const char*, std::string>{
              "network_id", "Network ID",
              HexBytes(data.secured.online_network_id.data(),
                       data.secured.online_network_id.size(), 0)}}) {
      w.StartObject();
      w.Key("id");
      w.String(id);
      w.Key("label");
      w.String(label);
      w.Key("type");
      w.String("text");
      w.Key("value");
      w.String(text.c_str());
      w.Key("readOnly");
      w.Bool(true);
      w.EndObject();
    }
  }
  w.EndArray();
  w.EndObject();
}

BladesModel::PlayedTitles BladesModel::CollectPlayedTitles() {
  Emulator* emulator = window_.emulator();
  kernel::KernelState* kernel = emulator ? emulator->kernel_state() : nullptr;
  // Games the signed-in profiles have played carry their icons; keep any we
  // don't have yet, and learn their title IDs by name for recent entries
  // recorded before IDs were kept.
  PlayedTitles played;
  if (kernel) {
    auto* profiles = kernel->xam_state()->profile_manager();
    auto* tracker = kernel->xam_state()->user_tracker();
    for (uint8_t slot = 0; profiles && tracker && slot < XUserMaxUserCount;
         ++slot) {
      auto* profile = profiles->GetProfile(slot);
      if (!profile) {
        continue;
      }
      for (const auto& title : tracker->GetPlayedTitles(profile->xuid())) {
        played.ids_by_name[xe::utf8::lower_ascii(
            xe::to_utf8(title.title_name))] = title.id;
        played.progress.try_emplace(
            title.id, PlayedTitles::Progress{title.unlocked_achievements_count,
                               title.achievements_count,
                               title.title_earned_gamerscore,
                               title.gamerscore_amount});
        window_.SaveTitleIcon(title.id, title.icon);
      }
    }
  }
  return played;
}

void BladesModel::WriteTitles(StateWriter& w, const PlayedTitles& played) {
  // A game's icon, once it has one in title_icons.
  auto icon_url = [this](uint32_t title_id) {
    std::error_code ec;
    if (title_id &&
        std::filesystem::exists(window_.TitleIconPath(title_id), ec)) {
      return fmt::format("/title-icons/{:08X}.png", title_id);
    }
    return std::string();
  };
  // A game's achievements and gamerscore, when the first player has played it.
  auto write_progress = [&](uint32_t title_id) {
    auto progress = played.progress.find(title_id);
    if (!title_id || progress == played.progress.end() ||
        !progress->second.total) {
      return;
    }
    w.Key("achievements");
    w.StartObject();
    w.Key("unlocked");
    w.Uint(progress->second.unlocked);
    w.Key("total");
    w.Uint(progress->second.total);
    w.Key("score");
    w.Uint(progress->second.score);
    w.Key("maxScore");
    w.Uint(progress->second.max_score);
    w.EndObject();
  };

  // Recently played games, most recent first, for the Home blade.
  w.Key("recentTitles");
  w.StartArray();
  for (const RecentTitleEntry& entry : window_.recently_launched_titles_) {
    std::string path = xe::path_to_utf8(entry.path_to_file);
    uint32_t title_id = entry.title_id;
    if (!title_id) {
      auto found =
          played.ids_by_name.find(xe::utf8::lower_ascii(entry.title_name));
      if (found != played.ids_by_name.end()) {
        title_id = found->second;
      }
    }
    const std::string icon = icon_url(title_id);
    w.StartObject();
    w.Key("name");
    w.String(entry.title_name.c_str());
    w.Key("path");
    w.String(path.c_str());
    w.Key("lastPlayed");
    w.Int64(int64_t(entry.last_run_time));
    w.Key("icon");
    w.String(icon.c_str());
    write_progress(title_id);
    w.EndObject();
  }
  w.EndArray();

  // All Games: every game in the game folders and every game played here,
  // each once. A played game found in a folder takes the name it gave itself
  // and the time it was last played; played games outside the folders stay.
  {
    struct Row {
      std::string name;
      std::string path;
      int64_t last_played = 0;
      uint32_t title_id = 0;
    };
    auto key = [](const std::string& name) {
      std::string out;
      for (unsigned char c : name) {
        if (std::isalnum(c)) out.push_back(char(std::tolower(c)));
      }
      return out;
    };
    std::vector<Row> rows;
    std::map<std::string, size_t> by_path, by_name;
    std::map<uint32_t, size_t> by_id;
    GameLibrary& library = window_.game_library();
    for (const GameLibrary::Game& game : library.games()) {
      Row row{game.name, xe::path_to_utf8(game.path), 0, game.title_id};
      const size_t index = rows.size();
      by_path.emplace(row.path, index);
      if (row.title_id) by_id.emplace(row.title_id, index);
      by_name.emplace(key(row.name), index);
      rows.push_back(std::move(row));
    }
    for (const RecentTitleEntry& entry : window_.recently_launched_titles_) {
      const std::string path = xe::path_to_utf8(entry.path_to_file);
      uint32_t title_id = entry.title_id;
      if (!title_id) {
        auto found = played.ids_by_name.find(xe::utf8::lower_ascii(entry.title_name));
        if (found != played.ids_by_name.end()) title_id = found->second;
      }
      auto match = by_path.find(path);
      size_t index = match != by_path.end() ? match->second : SIZE_MAX;
      if (index == SIZE_MAX && title_id) {
        auto id = by_id.find(title_id);
        if (id != by_id.end()) index = id->second;
      }
      if (index == SIZE_MAX && !entry.title_name.empty()) {
        auto name = by_name.find(key(entry.title_name));
        if (name != by_name.end()) index = name->second;
      }
      if (index == SIZE_MAX) {
        rows.push_back({entry.title_name.empty()
                            ? xe::path_to_utf8(entry.path_to_file.stem())
                            : entry.title_name,
                        path, int64_t(entry.last_run_time), title_id});
        continue;
      }
      Row& row = rows[index];
      if (int64_t(entry.last_run_time) > row.last_played) {
        row.last_played = int64_t(entry.last_run_time);
        if (!entry.title_name.empty()) row.name = entry.title_name;
      }
      if (!row.title_id) row.title_id = title_id;
    }

    w.Key("allGames");
    w.StartArray();
    for (const Row& row : rows) {
      const std::string icon = icon_url(row.title_id);
      w.StartObject();
      w.Key("name");
      w.String(row.name.c_str());
      w.Key("path");
      w.String(row.path.c_str());
      w.Key("lastPlayed");
      w.Int64(row.last_played);
      w.Key("icon");
      w.String(icon.c_str());
      write_progress(row.title_id);
      w.EndObject();
    }
    w.EndArray();

    w.Key("gameFolders");
    w.StartObject();
    w.Key("folders");
    w.StartArray();
    for (const auto& folder : library.folders()) {
      w.String(xe::path_to_utf8(folder).c_str());
    }
    w.EndArray();
    w.Key("scanning");
    w.Bool(library.scanning());
    w.EndObject();
  }
}

void BladesModel::WriteMusicPlayer(StateWriter& w) {
  Emulator* emulator = window_.emulator();
  // The console's music player (games hand it their playlists).
  w.Key("musicPlayer");
  w.StartObject();
  apu::AudioMediaPlayer* player =
      emulator ? emulator->audio_media_player() : nullptr;
  w.Key("available");
  w.Bool(player != nullptr && cvars::enable_xmp);
  w.Key("state");
  w.String(!player                ? "idle"
           : player->IsPlaying()  ? "playing"
           : player->IsPaused()   ? "paused"
                                  : "idle");
  w.Key("volume");
  w.Double(player ? double(player->GetVolume()->load()) : 1.0);
  w.EndObject();
}

void BladesModel::WriteProfiles(StateWriter& w) {
  Emulator* emulator = window_.emulator();
  kernel::KernelState* kernel = emulator ? emulator->kernel_state() : nullptr;
  // Profiles, and who is signed in where.
  w.Key("profiles");
  w.StartArray();
  kernel::xam::ProfileManager* profiles =
      kernel ? kernel->xam_state()->profile_manager() : nullptr;
  if (profiles) {
    for (const auto& [xuid, account] : *profiles->GetAccounts()) {
      const uint8_t slot = profiles->GetUserIndexAssignedToProfile(xuid);
      std::string gamertag = account.GetGamertagString();
      std::string id = fmt::format("{:016X}", xuid);
      w.StartObject();
      w.Key("xuid");
      w.String(id.c_str());
      w.Key("gamertag");
      w.String(gamertag.c_str());
      w.Key("slot");
      w.Int(slot < XUserMaxUserCount ? int(slot) : -1);
      // Its gamer picture (tile_64.png in the profile's folder), as the page
      // reaches it, versioned so a new one shows at once.
      {
        const std::filesystem::path tile = ProfileFolder(xuid) / "tile_64.png";
        std::error_code ec;
        const auto time = std::filesystem::last_write_time(tile, ec);
        w.Key("picture");
        w.String(ec ? ""
                    : fmt::format("/content/{0}/FFFE07D1/00010000/{0}/"
                                  "tile_64.png?v={1}",
                                  id, time.time_since_epoch().count())
                          .c_str());
      }
      if (const ProfileStats* stats = StatsFor(xuid)) {
        w.Key("stats");
        w.StartObject();
        w.Key("games");
        w.Uint(stats->games);
        w.Key("gamerscore");
        w.Uint(stats->gamerscore);
        w.Key("achievements");
        w.Uint(stats->achievements);
        w.EndObject();
      }
      w.EndObject();
    }
  }
  w.EndArray();
}

// --- Gamer pictures ------------------------------------------------------------

std::filesystem::path BladesModel::ProfileFolder(uint64_t xuid) {
  // Where ProfileManager keeps a profile (GetProfilePath): its package,
  // unpacked, under the content root.
  const std::string id = fmt::format("{:016X}", xuid);
  return window_.emulator()->content_root() / id / "FFFE07D1" / "00010000" / id;
}

void BladesModel::SetProfilePicture(uint64_t xuid, const std::string& picture) {
  Emulator* emulator = window_.emulator();
  if (!emulator || !xuid) return;
  // The picture as the page names it: one of the app's own
  // (/gamer-pictures/<name>.png, in the page) or a downloaded one
  // (/content/..., under the content root). Nothing outside either.
  std::filesystem::path base, relative;
  if (picture.rfind("/gamer-pictures/", 0) == 0) {
#if XE_PLATFORM_MAC
    base = BladesOverlay::FindPageRoot();
#endif
    relative = xe::to_path(picture.substr(1));
  } else if (picture.rfind("/content/", 0) == 0) {
    base = emulator->content_root();
    relative = xe::to_path(picture.substr(std::strlen("/content/")));
  }
  if (base.empty() || relative.extension() != ".png") {
    XELOGW("Blades: not a gamer picture: {}", picture);
    return;
  }
  std::error_code ec;
  const std::filesystem::path file =
      std::filesystem::weakly_canonical(base / relative, ec);
  const std::string base_text =
      xe::path_to_utf8(std::filesystem::weakly_canonical(base, ec)) + "/";
  if (xe::path_to_utf8(file).rfind(base_text, 0) != 0) {
    XELOGW("Blades: not a gamer picture: {}", picture);
    return;
  }
  std::ifstream in(file, std::ios::binary);
  std::vector<uint8_t> big((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
#if XE_PLATFORM_MAC
  // A profile keeps its picture in two sizes; the small one is made here.
  std::vector<uint8_t> small = gamer_pictures::SmallPicture(big);
#else
  std::vector<uint8_t> small;
#endif
  if (small.empty()) {
    XELOGW("Blades: {} is not a 64x64 picture", picture);
    return;
  }
  kernel::KernelState* kernel = emulator->kernel_state();
  if (kernel && kernel->xam_state()->GetUserProfile(xuid)) {
    // Signed in: through the profile, which also keeps it for games asking.
    auto* tracker = kernel->xam_state()->user_tracker();
    tracker->UpdateUserIcon(xuid, big);
    tracker->UpdateUserIcon(xuid, small);
  } else {
    const std::filesystem::path folder = ProfileFolder(xuid);
    if (!std::filesystem::is_directory(folder, ec)) return;
    std::ofstream(folder / "tile_64.png", std::ios::binary | std::ios::trunc)
        .write(reinterpret_cast<const char*>(big.data()), std::streamsize(big.size()));
    std::ofstream(folder / "tile_32.png", std::ios::binary | std::ios::trunc)
        .write(reinterpret_cast<const char*>(small.data()),
               std::streamsize(small.size()));
  }
  XELOGI("Blades: gamer picture of {:016X} is now {}", xuid, picture);
}

// --- Gamercard numbers ----------------------------------------------------------

namespace {
std::filesystem::path ProfileStatsPath(Emulator* emulator) {
  return emulator->storage_root() / "profile_stats.toml";
}
}  // namespace

const BladesModel::ProfileStats* BladesModel::StatsFor(uint64_t xuid) {
  LoadProfileStats();
  Emulator* emulator = window_.emulator();
  kernel::KernelState* kernel = emulator ? emulator->kernel_state() : nullptr;
  // Signed in: count it now, the way the console's gamercard did, and keep
  // the result for when it isn't.
  if (kernel && kernel->xam_state()->GetUserProfile(xuid)) {
    ProfileStats stats;
    for (const auto& title :
         kernel->xam_state()->user_tracker()->GetPlayedTitles(xuid)) {
      ++stats.games;
      stats.gamerscore += title.title_earned_gamerscore;
      stats.achievements += title.unlocked_achievements_count;
    }
    auto [it, inserted] = profile_stats_.try_emplace(xuid, stats);
    if (inserted || !(it->second == stats)) {
      it->second = stats;
      SaveProfileStats();
    }
    return &it->second;
  }
  auto it = profile_stats_.find(xuid);
  return it == profile_stats_.end() ? nullptr : &it->second;
}

void BladesModel::LoadProfileStats() {
  if (profile_stats_loaded_ || !window_.emulator()) {
    return;
  }
  profile_stats_loaded_ = true;
  std::ifstream file(ProfileStatsPath(window_.emulator()));
  if (!file.is_open()) {
    return;
  }
  try {
    toml::table table = toml::parse(file);
    for (const auto& [key, value] : table) {
      const toml::table* entry = value.as_table();
      if (!entry) continue;
      ProfileStats stats;
      stats.games = uint32_t((*entry)["games"].value_or<int64_t>(0));
      stats.gamerscore = uint32_t((*entry)["gamerscore"].value_or<int64_t>(0));
      stats.achievements =
          uint32_t((*entry)["achievements"].value_or<int64_t>(0));
      profile_stats_[std::stoull(std::string(key.str()), nullptr, 16)] = stats;
    }
  } catch (const std::exception& error) {
    XELOGW("Blades: cannot read profile_stats.toml: {}", error.what());
  }
}

void BladesModel::SaveProfileStats() {
  if (!window_.emulator()) {
    return;
  }
  toml::table table;
  for (const auto& [xuid, stats] : profile_stats_) {
    toml::table entry;
    entry.insert("games", int64_t(stats.games));
    entry.insert("gamerscore", int64_t(stats.gamerscore));
    entry.insert("achievements", int64_t(stats.achievements));
    table.insert(fmt::format("{:016X}", xuid), std::move(entry));
  }
  std::ofstream file(ProfileStatsPath(window_.emulator()), std::ios::trunc);
  file << table;
}

void BladesModel::WriteApp(StateWriter& w) {
  w.Key("timeScalar");
  w.Double(Clock::guest_time_scalar());

  // The release this build belongs to, as Preferences names it; the build
  // itself (branch, commit, date) goes beside it.
  // The dashboard's sounds are downloaded on request, never shipped.
  w.Key("dashSounds");
  w.StartObject();
  w.Key("status");
  w.String(window_.DashboardSoundsStatus());
  w.Key("error");
  w.String(window_.dashboard_sounds_error_.c_str());
  w.Key("asked");
  w.Bool(cvars::dashboard_sounds_asked);
#if XE_PLATFORM_MAC
  w.Key("pack");
  w.String(dashboard_sounds::kPackName);
  w.Key("author");
  w.String(dashboard_sounds::kPackAuthor);
  w.Key("site");
  w.String(dashboard_sounds::kPackSite);
#endif
  w.EndObject();

  // The gamer-picture collection: offered, coming in, or installed.
  w.Key("gamerPictures");
  w.StartObject();
  w.Key("status");
  w.String(window_.GamerPicturesStatus());
  w.Key("error");
  w.String(window_.gamer_pictures_error_.c_str());
  w.Key("pictures");
  w.Uint(window_.gamer_pictures_count_);
  w.Key("titles");
  w.Uint(window_.gamer_pictures_titles_);
  w.Key("bytes");
  w.Uint64(window_.gamer_pictures_bytes_);
  int64_t received = 0, total = 0;
#if XE_PLATFORM_MAC
  total = gamer_pictures::kDownloadBytes;
  if (auto progress = std::static_pointer_cast<gamer_pictures::Progress>(
          window_.gamer_pictures_progress_)) {
    received = progress->received;
    total = progress->total ? progress->total.load() : total;
  }
  w.Key("collection");
  w.String(gamer_pictures::kCollectionName);
  w.Key("site");
  w.String(gamer_pictures::kSite);
#endif
  w.Key("received");
  w.Int64(received);
  w.Key("total");
  w.Int64(total);
  w.EndObject();
  gamer_pictures_progress_sent_ =
      total ? int(received * 1000 / total) : -1;

  // The Mac's languages, for the dashboard's "Same as my Mac" (i18n/).
  w.Key("systemLanguages");
  w.StartArray();
#if XE_PLATFORM_MAC
  for (const std::string& tag : PreferredLanguages()) {
    w.String(tag.c_str());
  }
#endif
  w.EndArray();

  w.Key("appVersion");
  w.String(kReleaseName);
  w.Key("appBuild");
  w.String(XE_BUILD_BRANCH "@" XE_BUILD_COMMIT_SHORT " (" XE_BUILD_DATE ")");
}

std::string BladesModel::BuildStateJson() {
  rapidjson::StringBuffer buffer;
  StateWriter w(buffer);
  w.StartObject();
  WriteRenderer(w);
  WriteSettings(w);
  WriteControllers(w);
  WriteConsole(w);
  WriteTitles(w, CollectPlayedTitles());
  WriteMusicPlayer(w);
  WriteProfiles(w);
  WriteApp(w);
  w.EndObject();
  last_controllers_ = ControllersFingerprint();
  std::string json = buffer.GetString();
  // XE_BLADES_STATE_DUMP=<file>: keep the latest state the page was given,
  // to compare two builds' output or to capture the real app's state as a
  // fixture for the page's tests.
  if (const char* dump = std::getenv("XE_BLADES_STATE_DUMP")) {
    std::ofstream(dump, std::ios::trunc) << json;
  }
  return json;
}

BladesModel::Result BladesModel::HandleMessage(const std::string& json) {
  rapidjson::Document message;
  message.Parse(json.c_str(), json.size());
  if (message.HasParseError() || !message.IsObject() ||
      !message.HasMember("op") || !message["op"].IsString()) {
    XELOGW("Blades: ignoring malformed message {}", json);
    return Result::kNone;
  }
  const std::string op = message["op"].GetString();
  Emulator* emulator = window_.emulator();
  hid::InputSystem* input = emulator ? emulator->input_system() : nullptr;
  auto string_member = [&](const char* key) -> std::string {
    return message.HasMember(key) && message[key].IsString()
               ? message[key].GetString()
               : std::string();
  };
  auto int_member = [&](const char* key, int fallback) -> int {
    return message.HasMember(key) && message[key].IsNumber()
               ? int(message[key].GetDouble())
               : fallback;
  };

  if (op == "ready") {
    return Result::kPush;
  }
  if (op == "close") {
    return Result::kClose;
  }

  if (op == "set" || op == "reset") {
    const std::string name = string_member("name");
    cvar::IConfigVar* var = FindSetting(name);
    if (!var) {
      XELOGW("Blades: no setting named {}", name);
      return Result::kPush;
    }
    if (op == "reset") {
      var->ResetConfigValueToDefault();
    } else if (!message.HasMember("value") ||
               !SetSetting(var, message["value"])) {
      XELOGW("Blades: value does not fit setting {}", name);
      return Result::kPush;
    }
    if (IsDisplaySetting(name)) {
      window_.ApplyDisplayConfigForCvars();
    }
    if (name == "show_fps") {
      window_.UpdateFpsOverlay();
    }
#if XE_PLATFORM_MAC
    if (name == "preferences_button") {
      SetControllerHomeButtonClaimed(cvars::preferences_button == "guide");
    }
#endif
    config::SaveConfig();
    return Result::kPush;
  }

  if (op == "resetAll") {
    size_t reset = 0;
    if (cvar::ConfigVars) {
      for (const auto& [name, var] : *cvar::ConfigVars) {
        if (var->is_transient() || KeptByResetAll(var)) {
          continue;
        }
        const std::string before = var->config_value();
        var->ResetConfigValueToDefault();
        if (var->config_value() != before) {
          ++reset;
        }
      }
    }
    XELOGI("Blades: reset {} settings to their defaults", reset);
    window_.ApplyDisplayConfigForCvars();
    config::SaveConfig();
    return Result::kPush;
  }

  if (op == "console" || op == "consoleReset") {
    kernel::KernelState* kernel =
        emulator ? emulator->kernel_state() : nullptr;
    if (!kernel || !kernel->xconfig() || emulator->is_title_open()) {
      // A running title holds the console settings; the page says so.
      return Result::kPush;
    }
    if (op == "consoleReset") {
      kernel->xconfig()->SetDefaults();
      return Result::kPush;
    }
    kernel::XConfigData data = *kernel->xconfig()->GetXConfig();
    const std::string id = string_member("id");
    const rapidjson::Value* value =
        message.HasMember("value") ? &message["value"] : nullptr;
    if (!value) {
      return Result::kPush;
    }
    bool changed = false;
    if (value->IsBool()) {
      for (const ConsoleFlag& flag : kConsoleFlags) {
        if (id == flag.id) {
          SetFlag(data, flag, value->GetBool());
          changed = true;
        }
      }
    } else if (value->IsNumber()) {
      const double number = value->GetDouble();
      if (id == "language" && kLanguages.count(uint32_t(number))) {
        data.user.language = uint32_t(number);
        changed = true;
      } else if (id == "country" && kCountries.count(uint32_t(number))) {
        data.user.country = uint8_t(number);
        changed = true;
      } else if (id == "av_region" && kAvRegions.count(uint32_t(number))) {
        data.secured.av_region = uint32_t(number);
        changed = true;
      } else if (id == "time_zone" && number >= 0 &&
                 size_t(number) < kernel::kTimezones.size()) {
        const kernel::TimeZone& tz = kernel::kTimezones[size_t(number)];
        data.user.time_zone_bias = tz.timezone_bias;
        std::memcpy(data.user.tz_std_name.data(), tz.tz_std_name.data(), 4);
        std::memcpy(data.user.tz_dlt_name.data(), tz.tz_dlt_name.data(), 4);
        std::memcpy(data.user.tz_std_date.data(), tz.tz_std_date.data(), 4);
        std::memcpy(data.user.tz_dlt_date.data(), tz.tz_dlt_date.data(), 4);
        data.user.tz_std_bias = tz.tz_std_bias;
        data.user.tz_dlt_bias = tz.tz_dlt_bias;
        changed = true;
      } else if (id == "resolution") {
        for (const auto& r : kernel::XVGAResolution) {
          if (r.to_host() == uint32_t(number)) {
            data.user.av_pack_hdmi_sz = int32_t(r.to_host());
            ConsoleFlag widescreen = kConsoleFlags[6];
            SetFlag(data, widescreen, r.is_widescreen());
            changed = true;
          }
        }
      } else if (id == "music_volume") {
        data.user.music_volume = float(std::clamp(number, 0.0, 1.0));
        changed = true;
      }
    }
    if (changed) {
      kernel->xconfig()->WriteXConfig(&data);
    }
    return Result::kPush;
  }

  if (op == "hotkey") {
    window_.SetControllerHotkeyEnabled(
        int_member("id", -1),
        message.HasMember("enabled") && message["enabled"].IsBool() &&
            message["enabled"].GetBool());
    return Result::kPush;
  }

  if (op == "assign" && input) {
    int from = int_member("from", -1), to = int_member("to", -1);
    if (from >= 0 && to >= 0 && from < int(XUserMaxUserCount) &&
        to < int(XUserMaxUserCount)) {
      auto lock = input->lock();
      input->ReassignControllerSlot(uint32_t(from), uint32_t(to));
    }
    return Result::kPush;
  }

  if (op == "rebind" && input) {
    auto lock = input->lock();
    if (!capturing_output_.empty()) {
      input->CancelControllerInputCapture();
      capturing_output_.clear();
    }
    int slot = FirstPresentSlot(input);
    std::string output = string_member("output");
    if (slot >= 0 && !output.empty() &&
        input->BeginControllerInputCapture(uint32_t(slot))) {
      capturing_output_ = output;
      capturing_slot_ = slot;
    }
    return Result::kPush;
  }
  if (op == "rebindCancel") {
    CancelCapture();
    return Result::kPush;
  }
  if (op == "rebindReset" && input) {
    CancelCapture();
    auto lock = input->lock();
    int slot = FirstPresentSlot(input);
    if (slot >= 0) {
      input->ResetControllerMapping(uint32_t(slot));
    }
    return Result::kPush;
  }

  // Home: games, and the lifecycle of the one running.
  if (op == "launch") {
    const std::string path = string_member("path");
    if (path.empty()) {
      return Result::kNone;
    }
    EmulatorWindow* window = &window_;
    deferred_ = [window, path]() { window->RunTitle(xe::to_path(path)); };
    return Result::kCloseThen;
  }
  if (op == "openGame" || op == "installContent") {
    EmulatorWindow* window = &window_;
    const bool open = op == "openGame";
    deferred_ = [window, open]() {
      if (open) {
        window->FileOpen();
      } else {
        window->InstallContent();
      }
    };
    return Result::kCloseThen;
  }
  if (op == "stopGame") {
    if (emulator && emulator->is_title_open()) {
      window_.FileClose();
    }
    return Result::kPush;
  }

  // The music player.
  if (op == "music") {
    apu::AudioMediaPlayer* player =
        emulator ? emulator->audio_media_player() : nullptr;
    if (player) {
      const std::string id = string_member("id");
      if (id == "pause" && player->IsPlaying()) {
        player->Pause();
      } else if (id == "resume" && player->IsPaused()) {
        player->Continue();
      } else if (id == "next") {
        player->Next();
      } else if (id == "previous") {
        player->Previous();
      } else if (id == "volume" && message.HasMember("value") &&
                 message["value"].IsNumber()) {
        player->SetVolume(
            float(std::clamp(message["value"].GetDouble(), 0.0, 1.0)));
      }
    }
    return Result::kPush;
  }

  // Profiles.
  // Before the other profile ops, which take every op named profile*.
  if (op == "profilePicture") {
    SetProfilePicture(std::stoull(string_member("xuid"), nullptr, 16),
                      string_member("picture"));
    return Result::kPush;
  }

  if (op.rfind("profile", 0) == 0) {
    kernel::KernelState* kernel =
        emulator ? emulator->kernel_state() : nullptr;
    kernel::xam::ProfileManager* profiles =
        kernel ? kernel->xam_state()->profile_manager() : nullptr;
    if (!profiles) {
      return Result::kPush;
    }
    const std::string xuid_text = string_member("xuid");
    uint64_t xuid = 0;
    if (!xuid_text.empty()) {
      xuid = std::strtoull(xuid_text.c_str(), nullptr, 16);
    }
    const auto* account = xuid ? profiles->GetAccount(xuid) : nullptr;
    if (op == "profileCreate") {
      const std::string gamertag = string_member("gamertag");
      if (kernel::xam::ProfileManager::IsGamertagValid(gamertag)) {
        profiles->CreateProfile(gamertag, false);
      }
    } else if (!account) {
      XELOGW("Blades: no profile {}", xuid_text);
    } else if (op == "profileSignIn") {
      const int slot = int_member("slot", -1);
      if (slot >= 0 && slot < int(XUserMaxUserCount)) {
        const uint8_t current = profiles->GetUserIndexAssignedToProfile(xuid);
        if (current != uint8_t(slot)) {
          if (current < XUserMaxUserCount) {
            profiles->Logout(current);
          }
          if (profiles->GetProfile(uint8_t(slot))) {
            profiles->Logout(uint8_t(slot));
          }
          profiles->Login(xuid, uint8_t(slot));
        }
      }
    } else if (op == "profileSignOut") {
      const uint8_t current = profiles->GetUserIndexAssignedToProfile(xuid);
      if (current < XUserMaxUserCount) {
        profiles->Logout(current);
      }
    } else if (op == "profileRename") {
      const std::string gamertag = string_member("gamertag");
      if (kernel::xam::ProfileManager::IsGamertagValid(gamertag)) {
        kernel::xam::X_XAMACCOUNTINFO updated = *account;
        string_util::copy_and_swap_truncating(updated.gamertag,
                                              xe::to_utf16(gamertag),
                                              std::size(updated.gamertag));
        const bool signed_in = profiles->GetProfile(xuid) != nullptr;
        if (!signed_in) {
          profiles->MountProfile(xuid);
        }
        profiles->UpdateAccount(xuid, &updated);
        if (!signed_in) {
          profiles->DismountProfile(xuid);
        }
      }
    } else if (op == "profileDelete") {
      // A running game may hold the profile's files.
      if (!emulator->is_title_open()) {
        const uint8_t current = profiles->GetUserIndexAssignedToProfile(xuid);
        if (current < XUserMaxUserCount) {
          profiles->Logout(current);
        }
        profiles->DeleteProfile(xuid);
      }
    } else if (op == "profileFolder") {
      std::filesystem::path path = profiles->GetProfileContentPath(xuid);
      std::error_code ec;
      std::filesystem::create_directories(path, ec);
      LaunchFileExplorer(path);
    }
    return Result::kPush;
  }

  if (op == "gamerPicturesDownload") {
    window_.DownloadGamerPictures();
    return Result::kPush;
  }
  if (op == "gamerPicturesReveal") {
    window_.RevealGamerPictures();
    return Result::kNone;
  }
  if (op == "gamerPicturesDelete") {
    window_.DeleteGamerPictures();
    return Result::kPush;
  }

  if (op == "gameFolderAdd") {
    // After this message returns: the folder picker runs its own loop.
    EmulatorWindow* window = &window_;
    window_.app_context().CallInUIThreadDeferred([window]() {
      window->AddGameFolders();
#if XE_PLATFORM_MAC
      if (window->blades_overlay_ && window->blades_model_) {
        window->blades_overlay_->PushState(
            window->blades_model_->BuildStateJson());
      }
#endif
    });
    return Result::kNone;
  }
  if (op == "gameFolderRemove") {
    window_.game_library().RemoveFolder(xe::to_path(string_member("path")));
    return Result::kPush;
  }
  if (op == "gameFoldersRescan") {
    window_.game_library().Rescan();
    return Result::kPush;
  }

  if (op == "dashSounds") {
    // Either answer to the first-run offer is final: it isn't asked again.
    if (cvar::IConfigVar* var = FindSetting("dashboard_sounds_asked")) {
      SetSetting(var, rapidjson::Value(true));
      config::SaveConfig();
    }
    if (string_member("action") == "download") {
      window_.DownloadDashboardSounds();
    }
    return Result::kPush;
  }

  if (op == "action") {
    const std::string id = string_member("id");
    if (id == "timeScalar" && message.HasMember("value") &&
        message["value"].IsNumber()) {
      double scalar = message["value"].GetDouble();
      if (scalar > 0.0 && scalar <= 8.0) {
        Clock::set_guest_time_scalar(scalar);
        window_.UpdateTitle();
      }
      return Result::kPush;
    }
    if (id == "relaunch") {
      config::SaveConfig();
      return Result::kRelaunch;
    }
  }

  XELOGW("Blades: unknown message {}", json);
  return Result::kNone;
}

std::string BladesModel::ControllersFingerprint() {
  Emulator* emulator = window_.emulator();
  hid::InputSystem* input = emulator ? emulator->input_system() : nullptr;
  if (!input) {
    return std::string();
  }
  std::string out;
  auto lock = input->lock();
  for (uint32_t slot = 0; slot < XUserMaxUserCount; ++slot) {
    hid::ControllerIdentity identity = input->GetControllerIdentity(slot);
    out += identity.present ? identity.name : std::string("-");
    out.push_back('|');
  }
  return out;
}

bool BladesModel::Poll() {
  Emulator* emulator = window_.emulator();
  hid::InputSystem* input = emulator ? emulator->input_system() : nullptr;
  bool push = false;
  if (input && !capturing_output_.empty()) {
    auto lock = input->lock();
    std::string binding;
    if (input->PollControllerInputCapture(&binding)) {
      input->RebindControllerOutput(uint32_t(capturing_slot_),
                                    capturing_output_, binding);
      capturing_output_.clear();
      capturing_slot_ = -1;
      push = true;
    }
  }
  if (ControllersFingerprint() != last_controllers_) {
    push = true;
  }
#if XE_PLATFORM_MAC
  // A gamer-picture download moving on (or turning to installing).
  if (window_.gamer_pictures_downloading_) {
    auto progress = std::static_pointer_cast<gamer_pictures::Progress>(
        window_.gamer_pictures_progress_);
    if (progress) {
      const int64_t total = progress->total ? progress->total.load() : 1;
      const int now = int(progress->received * 1000 / total);
      if (now / 5 != gamer_pictures_progress_sent_ / 5 || progress->installing) {
        push = true;
      }
    }
  }
#endif
  return push;
}

void BladesModel::CancelCapture() {
  if (capturing_output_.empty()) {
    return;
  }
  Emulator* emulator = window_.emulator();
  hid::InputSystem* input = emulator ? emulator->input_system() : nullptr;
  if (input) {
    auto lock = input->lock();
    input->CancelControllerInputCapture();
  }
  capturing_output_.clear();
  capturing_slot_ = -1;
}

}  // namespace app
}  // namespace xe
