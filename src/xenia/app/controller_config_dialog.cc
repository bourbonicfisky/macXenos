/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/controller_config_dialog.h"

#include <array>

#include "xenia/app/emulator_window.h"
#include "xenia/hid/input_system.h"

#include "third_party/fmt/include/fmt/format.h"

namespace xe {
namespace app {

const char* ControllerKindLabel(hid::ControllerKind kind) {
  switch (kind) {
    case hid::ControllerKind::kXbox:
      return "Xbox";
    case hid::ControllerKind::kPlayStation:
      return "PlayStation";
    case hid::ControllerKind::kNintendo:
      return "Nintendo";
    case hid::ControllerKind::kOther:
      return "Other";
    case hid::ControllerKind::kUnknown:
    default:
      return "Unknown";
  }
}

void ControllerConfigDialog::DrawConnectedControllers() {
  auto* input_sys = emulator_window_.emulator()->input_system();
  if (!input_sys) {
    ImGui::TextUnformatted("Input system is not available.");
    return;
  }

  auto lock = input_sys->lock();

  // Snapshot identities first so a reassignment triggered below doesn't mutate
  // the list while we're still iterating it.
  std::array<hid::ControllerIdentity, XUserMaxUserCount> identities;
  for (uint32_t slot = 0; slot < XUserMaxUserCount; ++slot) {
    identities[slot] = input_sys->GetControllerIdentity(slot);
  }

  int reassign_from = -1;
  int reassign_to = -1;

  if (ImGui::BeginTable("##controllers", 4,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Slot");
    ImGui::TableSetupColumn("Type");
    ImGui::TableSetupColumn("Controller");
    ImGui::TableSetupColumn("Assign to");
    ImGui::TableHeadersRow();

    const char* slot_items[] = {"Player 1", "Player 2", "Player 3", "Player 4"};

    for (uint32_t slot = 0; slot < XUserMaxUserCount; ++slot) {
      const auto& identity = identities[slot];
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("%u", slot + 1);
      ImGui::TableNextColumn();
      if (identity.present) {
        ImGui::TextUnformatted(ControllerKindLabel(identity.kind));
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(identity.name.empty() ? "(unnamed)"
                                                     : identity.name.c_str());
        ImGui::TableNextColumn();
        int current = static_cast<int>(slot);
        ImGui::PushID(static_cast<int>(slot));
        ImGui::SetNextItemWidth(110.0f);
        if (ImGui::Combo("##assign", &current, slot_items,
                         IM_ARRAYSIZE(slot_items)) &&
            current != static_cast<int>(slot)) {
          reassign_from = static_cast<int>(slot);
          reassign_to = current;
        }
        ImGui::PopID();
      } else {
        ImGui::TextDisabled("-");
        ImGui::TableNextColumn();
        ImGui::TextDisabled("(empty)");
        ImGui::TableNextColumn();
        ImGui::TextDisabled("-");
      }
    }
    ImGui::EndTable();
  }

  // Apply any reassignment after rendering the table (swaps with the target
  // slot if it is occupied).
  if (reassign_from >= 0 && reassign_to >= 0) {
    input_sys->ReassignControllerSlot(static_cast<uint32_t>(reassign_from),
                                      static_cast<uint32_t>(reassign_to));
  }

  ImGui::TextDisabled(
      "Note: the game always sees an Xbox 360 controller. Type is shown for "
      "Xenia's UI only.");
}

namespace {
// Brand-appropriate face-button legend for Xenia's UI. The guest still receives
// the Xbox A/B/X/Y layout; this only helps the user map Xenia's hotkey text
// (written in Xbox terms) to the buttons printed on their controller.
const char* FaceButtonLegend(hid::ControllerKind kind) {
  switch (kind) {
    case hid::ControllerKind::kPlayStation:
      return "On your PlayStation pad: A = Cross, B = Circle, X = Square, "
             "Y = Triangle.";
    case hid::ControllerKind::kNintendo:
      return "On your Nintendo pad the A/B and X/Y positions are swapped vs "
             "Xbox; hotkeys below use the Xbox names.";
    default:
      return nullptr;
  }
}
}  // namespace

void ControllerConfigDialog::DrawHotkeys() {
  // Brand legend for the first connected, non-Xbox controller (so PlayStation
  // users can map the Xbox-worded hotkeys to their buttons).
  auto* input_sys = emulator_window_.emulator()->input_system();
  if (input_sys) {
    auto lock = input_sys->lock();
    for (uint32_t slot = 0; slot < XUserMaxUserCount; ++slot) {
      hid::ControllerIdentity identity = input_sys->GetControllerIdentity(slot);
      if (identity.present) {
        const char* legend = FaceButtonLegend(identity.kind);
        if (legend) {
          ImGui::TextWrapped("%s", legend);
          ImGui::Separator();
        }
        break;
      }
    }
  }

  ImGui::TextDisabled(
      "Controller hotkeys (hold Guide + the button). Uncheck to disable:");
  for (const auto& hotkey : emulator_window_.GetControllerHotkeys()) {
    bool enabled = hotkey.enabled;
    ImGui::PushID(hotkey.function);
    if (ImGui::Checkbox(hotkey.description.c_str(), &enabled)) {
      emulator_window_.SetControllerHotkeyEnabled(hotkey.function, enabled);
    }
    ImGui::PopID();
  }
}

void ControllerConfigDialog::DrawRebind() {
  auto* input_sys = emulator_window_.emulator()->input_system();
  if (!input_sys) {
    ImGui::TextUnformatted("Input system is not available.");
    return;
  }
  auto lock = input_sys->lock();

  // Rebind the first connected controller (per-GUID, so it applies to every pad
  // of that model).
  int slot = -1;
  hid::ControllerIdentity identity;
  for (uint32_t s = 0; s < XUserMaxUserCount; ++s) {
    identity = input_sys->GetControllerIdentity(s);
    if (identity.present) {
      slot = static_cast<int>(s);
      break;
    }
  }
  if (slot < 0) {
    ImGui::TextUnformatted("Connect a controller to remap its buttons.");
    capturing_output_.clear();
    return;
  }

  ImGui::Text("Remapping: Player %d - %s", slot + 1,
              identity.name.empty() ? ControllerKindLabel(identity.kind)
                                     : identity.name.c_str());
  ImGui::TextDisabled(
      "Click Rebind, then press the physical button you want for that action.");
  ImGui::Separator();

  // Apply a captured input to the rebind in progress.
  if (!capturing_output_.empty()) {
    std::string binding;
    if (input_sys->PollControllerInputCapture(&binding)) {
      input_sys->RebindControllerOutput(static_cast<uint32_t>(slot),
                                        capturing_output_, binding);
      capturing_output_.clear();
    }
  }

  struct Rebindable {
    const char* output;
    const char* label;
  };
  static const Rebindable kRebindable[] = {
      {"a", "A / Cross"},
      {"b", "B / Circle"},
      {"x", "X / Square"},
      {"y", "Y / Triangle"},
      {"leftshoulder", "Left Shoulder (LB / L1)"},
      {"rightshoulder", "Right Shoulder (RB / R1)"},
      {"back", "Back / Share"},
      {"start", "Start / Options"},
  };

  if (ImGui::BeginTable("##rebind", 3,
                        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Button");
    ImGui::TableSetupColumn("Current");
    ImGui::TableSetupColumn("");
    ImGui::TableHeadersRow();
    for (const auto& rb : kRebindable) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(rb.label);
      ImGui::TableNextColumn();
      std::string binding = input_sys->GetControllerOutputBinding(
          static_cast<uint32_t>(slot), rb.output);
      ImGui::TextUnformatted(binding.empty() ? "(unmapped)" : binding.c_str());
      ImGui::TableNextColumn();
      ImGui::PushID(rb.output);
      if (capturing_output_ == rb.output) {
        ImGui::TextUnformatted("press a button...");
        ImGui::SameLine();
        if (ImGui::SmallButton("Cancel")) {
          input_sys->CancelControllerInputCapture();
          capturing_output_.clear();
        }
      } else {
        ImGui::BeginDisabled(!capturing_output_.empty());
        if (ImGui::SmallButton("Rebind") &&
            input_sys->BeginControllerInputCapture(
                static_cast<uint32_t>(slot))) {
          capturing_output_ = rb.output;
        }
        ImGui::EndDisabled();
      }
      ImGui::PopID();
    }
    ImGui::EndTable();
  }

  ImGui::Separator();
  ImGui::BeginDisabled(!capturing_output_.empty());
  if (ImGui::Button("Reset to default")) {
    input_sys->ResetControllerMapping(static_cast<uint32_t>(slot));
  }
  ImGui::EndDisabled();
  ImGui::TextDisabled(
      "Binding values are SDL physical inputs (e.g. b3). Saved to the user "
      "gamecontrollerdb.txt next to the app.");
}

void ControllerConfigDialog::OnDraw(ImGuiIO& io) {
  ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.90f);
  bool dialog_open = true;
  if (!ImGui::Begin("Controllers", &dialog_open,
                    ImGuiWindowFlags_NoCollapse |
                        ImGuiWindowFlags_AlwaysAutoResize |
                        ImGuiWindowFlags_NoMove)) {
    ImGui::End();
    return;
  }

  if (!dialog_open) {
    Close();
    ImGui::End();
    emulator_window_.ToggleControllerConfigDialog();
    return;
  }

  if (ImGui::BeginTabBar("##ControllerCategories")) {
    if (ImGui::BeginTabItem("Connected")) {
      ImGui::Dummy(ImVec2(2.f, 2.f));
      DrawConnectedControllers();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Rebind")) {
      ImGui::Dummy(ImVec2(2.f, 2.f));
      DrawRebind();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Hotkeys")) {
      ImGui::Dummy(ImVec2(2.f, 2.f));
      DrawHotkeys();
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }

  ImGui::End();
}

}  // namespace app
}  // namespace xe
