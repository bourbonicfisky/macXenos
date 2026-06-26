/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/host_settings_dialog.h"

#include <string>
#include <vector>

#include "xenia/app/emulator_window.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/config.h"
#include "xenia/emulator.h"

namespace xe {
namespace app {

namespace {

#if XE_PLATFORM_MAC
constexpr bool kIsMac = true;
#else
constexpr bool kIsMac = false;
#endif

enum class ControlKind { kBool, kIntSlider, kIntInput, kFloatSlider, kCombo };
enum class ApplyTiming { kLive, kNextLaunch };

struct Entry {
  const char* cvar_name;
  const char* label;
  ControlKind kind;
  ApplyTiming timing;
  // For kIntSlider / kFloatSlider: inclusive range. For kCombo: the option set.
  double min_value = 0.0;
  double max_value = 0.0;
  std::vector<const char*> combo_options = {};
};

struct Group {
  const char* title;
  bool macos_only;
  std::vector<Entry> entries;
};

// CURATED allowlist. This is deliberately hand-picked - the cvar registry has
// no "user-facing" flag, so iterating it would expose dev/debug internals as
// footguns. Add an entry here to surface a setting; the help text comes from
// the cvar's own description() (see DrawEntry), so it is never duplicated.
const std::vector<Group>& GetGroups() {
  static const std::vector<Group> kGroups = {
      {"Rendering",
       false,
       {
           {"vsync", "VSync", ControlKind::kBool, ApplyTiming::kLive},
           {"draw_resolution_scale_x", "Resolution scale X",
            ControlKind::kIntSlider, ApplyTiming::kNextLaunch, 1, 7},
           {"draw_resolution_scale_y", "Resolution scale Y",
            ControlKind::kIntSlider, ApplyTiming::kNextLaunch, 1, 7},
           {"anisotropic_override", "Anisotropic filtering override",
            ControlKind::kIntInput, ApplyTiming::kNextLaunch},
       }},
      {"MoltenVK (macOS)",
       true,
       {
           {"mvk_config_defaults", "Apply curated MoltenVK defaults",
            ControlKind::kBool, ApplyTiming::kNextLaunch},
       }},
      {"Input",
       false,
       {
           {"vibration", "Controller vibration", ControlKind::kBool,
            ApplyTiming::kLive},
           {"left_stick_deadzone_percentage", "Left stick deadzone",
            ControlKind::kFloatSlider, ApplyTiming::kLive, 0.0, 1.0},
           {"right_stick_deadzone_percentage", "Right stick deadzone",
            ControlKind::kFloatSlider, ApplyTiming::kLive, 0.0, 1.0},
       }},
      {"Window / Output",
       false,
       {
           {"mute", "Mute audio", ControlKind::kBool, ApplyTiming::kLive},
           {"window_guest_resolution_mode", "Window sizing from guest res",
            ControlKind::kCombo, ApplyTiming::kNextLaunch, 0, 0,
            {"scaled", "actual", "off"}},
       }},
  };
  return kGroups;
}

cvar::IConfigVar* FindConfigVar(const char* name) {
  if (!cvar::ConfigVars) {
    return nullptr;
  }
  auto it = cvar::ConfigVars->find(name);
  return it == cvar::ConfigVars->end() ? nullptr : it->second;
}

template <typename T>
cvar::ConfigVar<T>* FindTypedVar(const char* name) {
  return dynamic_cast<cvar::ConfigVar<T>*>(FindConfigVar(name));
}

// Renders the cvar description() as a hoverable "(?)" with wrapped text, so the
// explainer is always exactly the cvar's own documentation.
void DrawHelpMarker(const std::string& description) {
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
    ImGui::TextUnformatted(description.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

// Returns true if the entry rendered (its cvar exists). Missing cvars are
// skipped (e.g. a setting from a not-yet-merged change) rather than crashing.
bool DrawEntry(const Entry& entry) {
  cvar::IConfigVar* base = FindConfigVar(entry.cvar_name);
  if (!base) {
    return false;
  }
  ImGui::PushID(entry.cvar_name);
  switch (entry.kind) {
    case ControlKind::kBool: {
      if (auto* v = FindTypedVar<bool>(entry.cvar_name)) {
        bool value = *v->current_value();
        if (ImGui::Checkbox(entry.label, &value)) {
          v->OverrideConfigValue(value);
        }
      }
    } break;
    case ControlKind::kIntSlider: {
      if (auto* v = FindTypedVar<int32_t>(entry.cvar_name)) {
        int value = *v->current_value();
        if (ImGui::SliderInt(entry.label, &value,
                             static_cast<int>(entry.min_value),
                             static_cast<int>(entry.max_value))) {
          v->OverrideConfigValue(value);
        }
      }
    } break;
    case ControlKind::kIntInput: {
      if (auto* v = FindTypedVar<int32_t>(entry.cvar_name)) {
        int value = *v->current_value();
        if (ImGui::InputInt(entry.label, &value)) {
          v->OverrideConfigValue(value);
        }
      }
    } break;
    case ControlKind::kFloatSlider: {
      if (auto* v = FindTypedVar<double>(entry.cvar_name)) {
        float value = static_cast<float>(*v->current_value());
        if (ImGui::SliderFloat(entry.label, &value,
                               static_cast<float>(entry.min_value),
                               static_cast<float>(entry.max_value), "%.2f")) {
          v->OverrideConfigValue(static_cast<double>(value));
        }
      }
    } break;
    case ControlKind::kCombo: {
      if (auto* v = FindTypedVar<std::string>(entry.cvar_name)) {
        const std::string current = *v->current_value();
        if (ImGui::BeginCombo(entry.label, current.c_str())) {
          for (const char* option : entry.combo_options) {
            bool selected = current == option;
            if (ImGui::Selectable(option, selected)) {
              v->OverrideConfigValue(std::string(option));
            }
            if (selected) {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }
      }
    } break;
  }
  if (entry.timing == ApplyTiming::kNextLaunch) {
    ImGui::SameLine();
    ImGui::TextDisabled("(next launch)");
  }
  DrawHelpMarker(base->description());
  ImGui::PopID();
  return true;
}

}  // namespace

void HostSettingsDialog::OnClose() {
  // ImGuiDialog::Draw() runs `delete this` immediately after this call when the
  // dialog is closing (e.g. the window's X button). EmulatorWindow still holds
  // an owning unique_ptr to us; release it here so that pointer does not dangle.
  // Otherwise reopening the dialog would read or delete freed memory (a heap
  // churned by a running title makes the latent crash reliable). release() drops
  // ownership without deleting - the base class deletes itself.
  emulator_window_.host_settings_dialog_.release();
}

void HostSettingsDialog::OnDraw(ImGuiIO& io) {
  ImGui::SetNextWindowPos(ImVec2(40, 40), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.90f);
  bool dialog_open = true;
  if (!ImGui::Begin("Host / Advanced Settings", &dialog_open,
                    ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::End();
    return;
  }

  ImGui::TextWrapped(
      "Host and renderer settings. Hover (?) for details. Some settings take "
      "effect only on the next launch.");
  ImGui::Separator();

  for (const Group& group : GetGroups()) {
    if (group.macos_only && !kIsMac) {
      continue;
    }
    // Only show a group header if at least one of its cvars exists.
    bool any_present = false;
    for (const Entry& entry : group.entries) {
      if (FindConfigVar(entry.cvar_name)) {
        any_present = true;
        break;
      }
    }
    if (!any_present) {
      continue;
    }
    ImGui::SeparatorText(group.title);
    for (const Entry& entry : group.entries) {
      DrawEntry(entry);
    }
  }

  ImGui::Separator();

  // Save always available. Apply & Relaunch only on macOS with a title loaded,
  // because it re-execs the current title to apply startup-only settings.
  if (ImGui::Button("Save")) {
    config::SaveConfig();
    save_confirmation_disappearance_ = ImGui::GetTime() + 3.0;
  }

  const bool title_loaded =
      emulator_window_.emulator() &&
      emulator_window_.emulator()->is_title_open();
#if XE_PLATFORM_MAC
  if (title_loaded) {
    ImGui::SameLine();
    if (ImGui::Button("Apply & Relaunch")) {
      config::SaveConfig();
      // Warm reboot: persists shader/pipeline caches, re-execs the title.
      emulator_window_.RelaunchLastTitle(true);
    }
  } else {
    ImGui::SameLine();
    ImGui::TextDisabled("(next-launch settings apply when you next load a title)");
  }
#else
  if (!title_loaded) {
    ImGui::SameLine();
    ImGui::TextDisabled("(next-launch settings apply on next launch)");
  }
#endif

  if (save_confirmation_disappearance_ > ImGui::GetTime()) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Saved.");
  }

  ImGui::End();

  if (!dialog_open) {
    Close();
  }
}

}  // namespace app
}  // namespace xe
