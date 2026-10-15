/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/app/emulator_window.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <span>

#include "third_party/imgui/imgui.h"
#include "third_party/stb/stb_image_write.h"
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
#endif
#include "third_party/tomlplusplus/toml.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "xenia/app/blades_model.h"
#include "xenia/app/game_library.h"
#include "xenia/base/assert.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/string_util.h"
#include "xenia/base/utf8.h"
#include "xenia/config.h"
#include "xenia/base/debugging.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/base/profiling.h"
#include "xenia/base/system.h"
#include "xenia/base/threading.h"
#include "xenia/apu/apu_flags.h"
#include "xenia/apu/audio_system.h"
#include "xenia/cpu/processor.h"
#include "xenia/emulator.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/vulkan/vulkan_graphics_system.h"
#if XE_GPU_METAL
#include "xenia/gpu/metal/metal_graphics_system.h"
#endif  // XE_GPU_METAL
#include "xenia/hid/input_system.h"
#include "xenia/kernel/xam/profile_manager.h"
#include "xenia/kernel/xam/xam_module.h"
#include "xenia/kernel/xam/xam_state.h"
#include "xenia/kernel/xconfig.h"
#include "xenia/kernel/util/game_info_database.h"
#include "xenia/ui/file_picker.h"
#include "xenia/ui/graphics_provider.h"
#if XE_PLATFORM_MAC
#include "xenia/ui/process_relaunch_mac.h"
#include "xenia/app/blades_overlay_mac.h"
#include "xenia/app/controller_home_button_mac.h"
#include "xenia/app/dashboard_sounds_mac.h"
#include "xenia/app/gamer_pictures_mac.h"
#include "xenia/ui/window_mac.h"
#endif  // XE_PLATFORM_MAC
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/imgui_host_notification.h"
#include "xenia/ui/immediate_drawer.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/ui_event.h"
#include "xenia/ui/virtual_key.h"

#include "version.h"

DECLARE_bool(debug);

DECLARE_string(hid);

DECLARE_bool(guide_button);

DECLARE_bool(clear_memory_page_state);

DECLARE_string(readback_resolve);

DECLARE_bool(readback_memexport);

DEFINE_bool(fullscreen, false, "Whether to launch the emulator in fullscreen.",
            "Display");

DEFINE_bool(controller_hotkeys, false, "Hotkeys for Xbox and PS controllers.",
            "General");

DEFINE_string(disabled_controller_hotkeys, "",
              "Comma-separated controller-hotkey function ids the user has "
              "disabled (managed by the Controllers UI).",
              "HID");

DEFINE_bool(dashboard_sounds_asked, false,
            "Whether the dashboard has offered to download its sounds. They "
            "are not included with the app.",
            "UI");

DEFINE_bool(dashboard_freezes_game, true,
            "Freeze the running game while the dashboard is open, so nothing "
            "happens in it until you go back.",
            "UI");
DEFINE_bool(show_fps, false,
            "Show the frame rate in a corner of the window while a game runs.",
            "Display");
DEFINE_string(preferences_key, "F6",
              "Keyboard key that opens and closes Preferences: F6, F7, F8, F10 "
              "or none. Command-comma always works.",
              "UI");
DEFINE_string(preferences_button, "guide",
              "Controller button that opens Preferences: guide (a tap of "
              "Guide on its own), back_start (Back and Start together) or "
              "none.",
              "UI");
DEFINE_string(
    postprocess_scaling_and_sharpening, "",
    "Post-processing effect to use for resampling and/or sharpening of the "
    "final display output.\n"
    "Use: [bilinear, cas, fsr, metalfx]\n"
    " bilinear (or any value not listed here):\n"
    "  Original image at 1:1, simple bilinear stretching for resampling.\n"
    " cas:\n"
    "  Use AMD FidelityFX Contrast Adaptive Sharpening (CAS) for sharpening "
    "at scaling factors of up to 2x2, with additional bilinear stretching for "
    "larger factors.\n"
    " fsr:\n"
    "  Use AMD FidelityFX Super Resolution 1.0 (FSR) for highest-quality "
    "upscaling, or AMD FidelityFX Contrast Adaptive Sharpening for sharpening "
    "while not scaling or downsampling.\n"
    "  For scaling by factors of more than 2x2, multiple FSR passes are done.\n"
    " metalfx:\n"
    "  Use Apple MetalFX spatial upscaling, in one pass at any factor (Metal "
    "renderer only; bilinear elsewhere).",
    "Display");
DEFINE_double(
    postprocess_ffx_cas_additional_sharpness,
    xe::ui::Presenter::GuestOutputPaintConfig::kCasAdditionalSharpnessDefault,
    "Additional sharpness for AMD FidelityFX Contrast Adaptive Sharpening "
    "(CAS), from 0 to 1.\n"
    "Higher is sharper.",
    "Display");
DEFINE_uint32(
    postprocess_ffx_fsr_max_upsampling_passes,
    xe::ui::Presenter::GuestOutputPaintConfig::kFsrMaxUpscalingPassesMax,
    "Maximum number of upsampling passes performed in AMD FidelityFX Super "
    "Resolution 1.0 (FSR) before falling back to bilinear stretching after the "
    "final pass.\n"
    "Each pass upscales only to up to 2x2 the previous size. If the game "
    "outputs a 1280x720 image, 1 pass will upscale it to up to 2560x1440 "
    "(below 4K), after 2 passes it will be upscaled to a maximum of 5120x2880 "
    "(including 3840x2160 for 4K), and so on.\n"
    "This variable has no effect if the display resolution isn't very high, "
    "but may be reduced on resolutions like 4K or 8K in case the performance "
    "impact of multiple FSR upsampling passes is too high, or if softer edges "
    "are desired.\n"
    "The default value is the maximum internally supported by Xenia.",
    "Display");
DEFINE_double(
    postprocess_ffx_fsr_sharpness_reduction,
    xe::ui::Presenter::GuestOutputPaintConfig::kFsrSharpnessReductionDefault,
    "Sharpness reduction for AMD FidelityFX Super Resolution 1.0 (FSR), in "
    "stops.\n"
    "Lower is sharper.",
    "Display");
// Dithering to 8bpc is enabled by default since the effect is minor, only
// effects what can't be shown normally by host displays, and nothing is changed
// by it for 8bpc source without resampling.
DEFINE_bool(
    postprocess_dither, true,
    "Dither the final image output from the internal precision to 8 bits per "
    "channel so gradients are smoother.\n"
    "On a 10bpc display, the lower 2 bits will still be kept, but noise will "
    "be added to them - disabling may be recommended for 10bpc, but it "
    "depends on the 10bpc displaying capabilities of the actual display used.",
    "Display");

DEFINE_int32(recent_titles_entry_amount, 10,
             "How many recently played titles the File menu lists. Every "
             "title played is kept for the dashboard's All Games list; 0 "
             "keeps none.",
             "General");
DEFINE_path(screenshot_path, "",
            "Folder for screenshots (F12 and screenshot_interval). Empty: a "
            "screenshots folder beside the executable.",
            "General");
DEFINE_int32(screenshot_interval, 0,
             "Seconds between automatic screenshots of the guest output, for "
             "unattended test runs (no screen-recording permission needed). "
             "0 = off.",
             "General");
DEFINE_int32(screenshot_interval_ms, 0,
             "Milliseconds between automatic screenshots, overriding "
             "screenshot_interval when > 0 (burst capture for flicker hunts). "
             "Burst filenames carry a sequence number and the guest swap "
             "count. 0 = off.",
             "General");
DEFINE_int32(log_fps_interval, 0,
             "Seconds between GuestFps log lines (guest swaps per second, any "
             "GPU backend), for unattended test runs. 0 = off.",
             "General");
DEFINE_bool(disable_doubleclick_fullscreen, false,
            "Allows the user to disable the behavior where a fast double-click "
            "causes Xenia to enter fullscreen mode.",
            "General");

namespace xe {
namespace app {

using xe::ui::FileDropEvent;
using xe::ui::KeyEvent;
using xe::ui::MenuItem;
using xe::ui::UIEvent;

using namespace xe::hid;
using namespace xe::gpu;

constexpr std::string_view kRecentlyPlayedTitlesFilename = "recent.toml";
// The most titles recent.toml keeps: the dashboard's All Games list.
constexpr size_t kMaxPlayedTitles = 1000;
constexpr std::string_view kBaseTitle = "macXenos";

EmulatorWindow::EmulatorWindow(Emulator* emulator,
                               ui::WindowedAppContext& app_context,
                               uint32_t width, uint32_t height)
    : emulator_(emulator),
      app_context_(app_context),
      window_listener_(*this),
      window_(ui::Window::Create(app_context, kBaseTitle, width, height)),
      imgui_drawer_(
          std::make_unique<ui::ImGuiDrawer>(window_.get(), kZOrderImGui)),
      display_config_game_config_load_callback_(
          new DisplayConfigGameConfigLoadCallback(*emulator, *this)) {
  base_title_ = std::string(kBaseTitle) +
#ifdef DEBUG
#if _NO_DEBUG_HEAP == 1
                " DEBUG"
#else
                " CHECKED"
#endif
#endif
                " ("
#ifdef XE_BUILD_IS_PR
                "PR#" XE_BUILD_PR_NUMBER " - "
#endif
                XE_BUILD_BRANCH "@" XE_BUILD_COMMIT_SHORT " on " XE_BUILD_DATE
                ")";

  LoadRecentlyLaunchedTitles();
}

std::unique_ptr<EmulatorWindow> EmulatorWindow::Create(
    Emulator* emulator, ui::WindowedAppContext& app_context, uint32_t width,
    uint32_t height) {
  assert_true(app_context.IsInUIThread());
  std::unique_ptr<EmulatorWindow> emulator_window(
      new EmulatorWindow(emulator, app_context, width, height));
  if (!emulator_window->Initialize()) {
    return nullptr;
  }
  return emulator_window;
}

EmulatorWindow::~EmulatorWindow() {
  if (preferences_button_listener_) {
    preferences_button_stop_ = true;
    xe::threading::Wait(preferences_button_listener_.get(), false);
  }
  if (auto_screenshot_thread_) {
    auto_screenshot_stop_ = true;
    xe::threading::Wait(auto_screenshot_thread_.get(), false);
  }
  // Notify the ImGui drawer that the immediate drawer is being destroyed.
  ShutdownGraphicsSystemPresenterPainting();
}

ui::Presenter* EmulatorWindow::GetGraphicsSystemPresenter() const {
  gpu::GraphicsSystem* graphics_system = emulator_->graphics_system();
  return graphics_system ? graphics_system->presenter() : nullptr;
}

void EmulatorWindow::SetupGraphicsSystemPresenterPainting() {
  ShutdownGraphicsSystemPresenterPainting();

  if (!window_) {
    return;
  }

  ui::Presenter* presenter = GetGraphicsSystemPresenter();
  if (!presenter) {
    return;
  }

  ApplyDisplayConfigForCvars();

  window_->SetPresenter(presenter);

  immediate_drawer_ =
      emulator_->graphics_system()->provider()->CreateImmediateDrawer();
  if (immediate_drawer_) {
    immediate_drawer_->SetPresenter(presenter);
    imgui_drawer_->SetPresenterAndImmediateDrawer(presenter,
                                                  immediate_drawer_.get());
    Profiler::SetUserIO(kZOrderProfiler, window_.get(), presenter,
                        immediate_drawer_.get());
  }
}

void EmulatorWindow::ShutdownGraphicsSystemPresenterPainting() {
  Profiler::SetUserIO(kZOrderProfiler, window_.get(), nullptr, nullptr);
  imgui_drawer_->SetPresenterAndImmediateDrawer(nullptr, nullptr);
  immediate_drawer_.reset();
  if (window_) {
    window_->SetPresenter(nullptr);
  }
}

void EmulatorWindow::OnEmulatorInitialized() {
  if (!emulator_->kernel_state()
           ->xam_state()
           ->profile_manager()
           ->GetAccountCount()) {
#if XE_PLATFORM_MAC
    // No profile yet: the dashboard opens on Profiles, where one is made
    // (Create a profile), rather than the old message box.
    app_context().CallInUIThread([this]() { ShowPreferences("profiles"); });
#else
    new NoProfileDialog(imgui_drawer_.get(), this);
    disable_hotkeys_ = true;
#endif
  }

  emulator_initialized_ = true;
  window_->SetMainMenuEnabled(true);
  UpdateFpsOverlay();
#if XE_PLATFORM_MAC
  if (!preferences_button_listener_) {
    preferences_button_listener_ = threading::Thread::Create(
        {}, [this] { PreferencesButtonListener(); });
    preferences_button_listener_->set_name("Preferences Button Listener");
  }
#endif
  if ((cvars::screenshot_interval > 0 || cvars::screenshot_interval_ms > 0 ||
       cvars::log_fps_interval > 0) &&
      !auto_screenshot_thread_) {
    auto_screenshot_thread_ =
        threading::Thread::Create({}, [this] { AutoScreenshotLoop(); });
    auto_screenshot_thread_->set_name("Auto Screenshot");
  }
  // Set the initial lifecycle item enable-state now that the menu is live.
  UpdateLifecycleMenuState();
  // When the user can see that the emulator isn't initializing anymore (the
  // menu isn't disabled), enter fullscreen if requested.
  // Metal-backend bring-up: auto-fullscreen-on-launch is disabled because the
  // config persists fullscreen=true on exit, making it impossible to escape while
  // iterating. Toggle fullscreen manually instead. TODO: revert before upstream.
  if (false && cvars::fullscreen) {
    SetFullscreen(true);
  }

  if (IsUseNexusForGameBarEnabled()) {
    XELOGE(
        "Xbox Gamebar Enabled, using BACK button instead of GUIDE for "
        "controller hotkeys!!!");
  }

  // Create a thread to listen for controller hotkeys.
  if (cvars::controller_hotkeys) {
    Gamepad_HotKeys_Listener =
        threading::Thread::Create({}, [&] { GamepadHotKeys(); });
    Gamepad_HotKeys_Listener->set_name("Gamepad HotKeys Listener");
  }
}

void EmulatorWindow::EmulatorWindowListener::OnClosing(ui::UIEvent& e) {
  emulator_window_.FreezeGame(false);
  emulator_window_.app_context_.QuitFromUIThread();
}

void EmulatorWindow::EmulatorWindowListener::OnFileDrop(ui::FileDropEvent& e) {
  emulator_window_.FileDrop(e.filename());
}

void EmulatorWindow::EmulatorWindowListener::OnKeyDown(ui::KeyEvent& e) {
  emulator_window_.OnKeyDown(e);
}

void EmulatorWindow::EmulatorWindowListener::OnMouseDown(ui::MouseEvent& e) {
  emulator_window_.OnMouseDown(e);
}

void EmulatorWindow::EmulatorWindowListener::OnMouseUp(ui::MouseEvent& e) {
  emulator_window_.OnMouseUp(e);
}

void EmulatorWindow::EmulatorWindowListener::OnUsbDeviceChanged(
    bool is_arrival) {
  if (!emulator_window_.emulator()) {
    return;
  }

  if (!emulator_window_.emulator()->input_system()) {
    return;
  }

  auto* portal = emulator_window_.emulator()->input_system()->GetPortal();
  if (!portal) {
    return;
  }

  if (is_arrival) {
    portal->OnDeviceArrival();
  } else {
    portal->OnDeviceRemoval();
  }
}

void EmulatorWindow::DisplayConfigGameConfigLoadCallback::PostGameConfigLoad() {
  emulator_window_.ApplyDisplayConfigForCvars();
}

void EmulatorWindow::ContentInstallDialog::OnDraw(ImGuiIO& io) {
  ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(20, 20), ImGuiCond_FirstUseEver);

  bool dialog_open = true;
  if (!ImGui::Begin(
          fmt::format("Installation Progress###{}", window_id_).c_str(),
          &dialog_open,
          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
              ImGuiWindowFlags_HorizontalScrollbar)) {
    Close();
    ImGui::End();
    return;
  }

  bool is_everything_installed = true;
  for (const auto& entry : *installation_entries_) {
    ImGui::BeginTable(fmt::format("table_{}", entry.name_).c_str(), 2);
    ImGui::TableNextRow(0);
    ImGui::TableSetColumnIndex(0);
    if (entry.icon_) {
      ImGui::Image(reinterpret_cast<ImTextureID>(entry.icon_.get()),
                   ui::default_image_icon_size);
    } else {
      ImGui::Dummy(ui::default_image_icon_size);
    }
    ImGui::TableNextColumn();

    ImGui::Text("Name: %s", entry.name_.c_str());
    ImGui::Text("Installation Path:");
    ImGui::SameLine();
    if (ImGui::TextLink(
            xe::path_to_utf8(entry.data_installation_path_).c_str())) {
      LaunchFileExplorer(emulator_window_.emulator_->kernel_state()
                             ->xam_state()
                             ->content_manager()
                             ->RootPathFor(entry.content_type_) /
                         entry.data_installation_path_);
    }

    if (entry.content_type_ != xe::XContentType::kInvalid) {
      ImGui::Text("Content Type: %s",
                  XContentTypeMap.at(entry.content_type_).c_str());
    }

    std::string result = fmt::format(
        "Status: {}", xe::Emulator::installStateStringName[static_cast<uint8_t>(
                          entry.installation_state_)]);

    if (entry.installation_state_ == xe::Emulator::InstallState::failed) {
      result += fmt::format(" - {} ({:08X})",
                            entry.installation_error_message_.c_str(),
                            entry.installation_result_);
    }

    ImGui::Text("%s", result.c_str());
    ImGui::EndTable();

    if (entry.content_size_ > 0) {
      ImGui::ProgressBar(static_cast<float>(entry.currently_installed_size_) /
                         entry.content_size_);

      if (entry.currently_installed_size_ != entry.content_size_ &&
          entry.installation_result_ == X_ERROR_SUCCESS) {
        is_everything_installed = false;
      }
    } else {
      ImGui::ProgressBar(0.0f);
    }

    if (installation_entries_->size() > 1) {
      ImGui::Separator();
    }
  }
  ImGui::Spacing();

  ImGui::BeginDisabled(!is_everything_installed);
  if (ImGui::Button("Close")) {
    ImGui::EndDisabled();
    Close();
    ImGui::End();
    return;
  }
  ImGui::EndDisabled();

  if (!dialog_open && is_everything_installed) {
    Close();
    ImGui::End();
    return;
  }
  ImGui::End();
}

void EmulatorWindow::FpsOverlay::OnDraw(ImGuiIO& io) {
  Emulator* emulator = emulator_window_.emulator_;
  gpu::GraphicsSystem* graphics =
      emulator ? emulator->graphics_system() : nullptr;
  gpu::CommandProcessor* command_processor =
      graphics ? graphics->command_processor() : nullptr;
  if (!cvars::show_fps || !command_processor || !emulator->is_title_open()) {
    last_time_ = 0.0;
    return;
  }
  // Guest swaps per second, over half-second windows.
  const double now = ImGui::GetTime();
  const uint64_t count = command_processor->guest_swap_count();
  if (last_time_ == 0.0) {
    last_time_ = now;
    last_count_ = count;
  } else if (now - last_time_ >= 0.5) {
    fps_ = double(count - last_count_) / (now - last_time_);
    last_time_ = now;
    last_count_ = count;
  }
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, 12.0f),
                          ImGuiCond_Always, ImVec2(1.0f, 0.0f));
  ImGui::SetNextWindowBgAlpha(0.55f);
  if (ImGui::Begin("##fps", nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                       ImGuiWindowFlags_AlwaysAutoResize |
                       ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing |
                       ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove)) {
    ImGui::Text("%.0f FPS", fps_);
  }
  ImGui::End();
}

void EmulatorWindow::UpdateFpsOverlay() {
  if (cvars::show_fps && !fps_overlay_) {
    fps_overlay_ = std::make_unique<FpsOverlay>(imgui_drawer_.get(), *this);
  } else if (!cvars::show_fps && fps_overlay_) {
    fps_overlay_.reset();
  }
}

void EmulatorWindow::ShowStartScreen() {
#if XE_PLATFORM_MAC
  auto* profiles = emulator_->kernel_state()
                       ? emulator_->kernel_state()->xam_state()->profile_manager()
                       : nullptr;
  ShowPreferences(profiles && profiles->GetAccountCount() == 0 ? "profiles"
                                                               : "home");
#endif
}

// Opens Preferences from a controller (preferences_button): a tap of Guide
// (released without another button, so Guide + A and the other hotkeys don't
// open it), or Back and Start together. While Preferences is open the page
// takes the buttons itself (PumpPreferencesController), and Guide or Start
// closes it.
void EmulatorWindow::PreferencesButtonListener() {
  bool guide_down = false;
  bool guide_chord = false;
  bool combo_down = false;
  while (!preferences_button_stop_.load()) {
    xe::threading::Sleep(std::chrono::milliseconds(30));
    const std::string mode = cvars::preferences_button;
    auto* input = emulator_->input_system();
    if (mode == "none" || !input) {
      continue;
    }
    uint16_t buttons = 0;
    {
      auto lock = input->lock();
      for (uint32_t user = 0; user < XUserMaxUserCount; ++user) {
        X_INPUT_STATE state;
        if (input->GetStateSnapshot(user, X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD,
                                    &state) == X_ERROR_SUCCESS) {
          buttons |= state.gamepad.buttons;
        }
      }
    }
    bool open = false;
    if (mode == "back_start") {
      const uint16_t both = X_INPUT_GAMEPAD_BACK | X_INPUT_GAMEPAD_START;
      const bool down = (buttons & both) == both;
      open = down && !combo_down;
      combo_down = down;
    } else {
      const bool guide = (buttons & X_INPUT_GAMEPAD_GUIDE) != 0;
      if (guide) {
        if (!guide_down) {
          guide_down = true;
          XELOGI("Controller: Guide pressed");
          // A press while the dashboard is open closes it (the page does);
          // the release must not open it again.
          guide_chord = blades_visible_.load();
        }
        if (buttons & ~uint16_t(X_INPUT_GAMEPAD_GUIDE)) {
          guide_chord = true;
        }
      } else if (guide_down) {
        guide_down = false;
        open = !guide_chord;
        XELOGI("Controller: Guide released ({})",
               open ? "opening the dashboard" : "not a tap on its own");
      }
    }
    if (open && !blades_visible_.load()) {
      app_context().CallInUIThread([this]() { ShowPreferences(""); });
    }
  }
}

bool EmulatorWindow::Initialize() {
  window_->AddListener(&window_listener_);
  window_->AddInputListener(&window_listener_, kZOrderEmulatorWindowInput);

#if XE_PLATFORM_MAC
  // The application menu's Settings… (⌘,) opens the Preferences blades.
  static_cast<ui::MacWindow*>(window_.get())
      ->SetSettingsHandler([this]() { TogglePreferences(""); });
  // The controller's Home button opens the dashboard rather than macOS's
  // Games overlay, while the dashboard is set to use it.
  SetControllerHomeButtonHandler([this]() { TogglePreferences(""); });
  SetControllerHomeButtonClaimed(cvars::preferences_button == "guide");

  // A title asking for another title (XamLoaderLaunchTitle) is served the way
  // the lifecycle Reboot is: flush config + the shader cache, then spawn a
  // fresh process on the new target with the loader data on its command line.
  // Runs on a host thread spawned by the XAM shim; the requesting guest thread
  // is parked and dies with the title.
  emulator_->set_on_launch_new_title(
      [this](const std::string& host_path, const std::string& launch_module,
             uint32_t launch_flags, const std::vector<uint8_t>& launch_data) {
        // A title that re-requests itself every boot would spawn forever;
        // after a few hops fall back to the "restart Xenia" message.
        constexpr int kMaxTitleRelaunchHops = 4;
        const char* hops_str = std::getenv("XENIA_TITLE_RELAUNCH_HOPS");
        int hops = hops_str ? std::atoi(hops_str) : 0;
        if (hops >= kMaxTitleRelaunchHops) {
          XELOGE(
              "Title relaunch: {} hops already; not spawning again for '{}'",
              hops, host_path);
          app_context().CallInUIThreadSynchronous([this]() {
            ui::ImGuiDialog::ShowMessageBox(
                imgui_drawer_.get(), "Title was restarted",
                "The title keeps asking to be relaunched. Please restart "
                "macXenos.");
          });
          return;
        }
        setenv("XENIA_TITLE_RELAUNCH_HOPS", std::to_string(hops + 1).c_str(),
               1);

        std::vector<std::string> extra_args;
        extra_args.push_back("--log_append=true");
        if (!launch_module.empty()) {
          extra_args.push_back("--relaunch_module=" + launch_module);
        }
        if (launch_flags) {
          extra_args.push_back(fmt::format("--launch_flags={}", launch_flags));
        }
        if (!launch_data.empty()) {
          std::string hex;
          hex.reserve(launch_data.size() * 2);
          for (uint8_t b : launch_data) {
            hex += fmt::format("{:02X}", b);
          }
          extra_args.push_back("--launch_data=" + hex);
        }
        if (window_->IsFullscreen()) {
          extra_args.push_back("--fullscreen=true");
        }

        FlushBeforeRelaunch();
        xe::ui::RelaunchWithTarget(host_path, extra_args);
        // RelaunchWithTarget only returns on failure.
        XELOGE("Title relaunch to '{}' failed; the title has been stopped.",
               host_path);
        app_context().CallInUIThreadSynchronous([this, host_path]() {
          ui::ImGuiDialog::ShowMessageBox(
              imgui_drawer_.get(), "Title was restarted",
              fmt::format("The title asked to launch '{}' but the new process "
                          "could not be started. Please restart macXenos.",
                          host_path)
                  .c_str());
        });
      });
#endif  // XE_PLATFORM_MAC

  // Main menu.
  // FIXME: This code is really messy.
  auto main_menu = MenuItem::Create(MenuItem::Type::kNormal);
  auto file_menu = MenuItem::Create(MenuItem::Type::kPopup, "&File");
  auto recent_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Open Recent");
  FillRecentlyLaunchedTitlesMenu(recent_menu.get());
  {
    file_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Open...", "Ctrl+O",
                         std::bind(&EmulatorWindow::FileOpen, this)));
    file_menu->AddChild(std::move(recent_menu));
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    file_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Install Content...",
                         std::bind(&EmulatorWindow::InstallContent, this)));
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    {
      auto stop_item =
          MenuItem::Create(MenuItem::Type::kString, "Stop Emulation",
                           std::bind(&EmulatorWindow::FileClose, this));
      stop_menu_item_ = stop_item.get();
      file_menu->AddChild(std::move(stop_item));

      auto reboot_item = MenuItem::Create(
          MenuItem::Type::kString, "Reboot Title",
          std::bind(&EmulatorWindow::RelaunchLastTitle, this, /*warm=*/true));
      reboot_menu_item_ = reboot_item.get();
      file_menu->AddChild(std::move(reboot_item));

      auto full_reboot_item = MenuItem::Create(
          MenuItem::Type::kString, "Full Reboot",
          std::bind(&EmulatorWindow::RelaunchLastTitle, this, /*warm=*/false));
      full_reboot_menu_item_ = full_reboot_item.get();
      file_menu->AddChild(std::move(full_reboot_item));
    }
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    file_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Show content directory...",
        std::bind(&EmulatorWindow::ShowContentDirectory, this)));
    file_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    file_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "E&xit", "Alt+F4",
                         [this]() { window_->RequestClose(); }));
  }
  main_menu->AddChild(std::move(file_menu));

  // Profile Menu
  auto profile_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Profile");
  {
    profile_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Profiles...", "",
        std::bind(&EmulatorWindow::ToggleProfilesConfigDialog, this)));
  }
  main_menu->AddChild(std::move(profile_menu));

  // Display menu.
  auto display_menu = MenuItem::Create(MenuItem::Type::kPopup, "&View");
  {
    display_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Graphics settings...", "F6",
        [this]() { TogglePreferences("graphics"); }));
  }
  display_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    display_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Fullscreen", "F11",
                         std::bind(&EmulatorWindow::ToggleFullscreen, this)));
    display_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Take Screenshot", "F12",
                         std::bind(&EmulatorWindow::TakeScreenshot, this)));
  }
  main_menu->AddChild(std::move(display_menu));

  // HID menu.
  auto hid_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Controller");
  {
    hid_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Toggle controller vibration", "",
        std::bind(&EmulatorWindow::ToggleControllerVibration, this)));
    // The hotkey chart on the Controller blade (the old message box is
    // gone from the menus).
    hid_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Display controller hotkeys", "",
        [this]() { ShowPreferences("controller/hotkeys"); }));
    hid_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Controller settings...", "",
        [this]() { ShowPreferences("controller"); }));
  }
  main_menu->AddChild(std::move(hid_menu));

  // Console menu
  auto console_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Console");
  {
    // The dashboard over the game, as Guide, F6 or ⌘, bring it up.
    console_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Show or Hide &Dashboard", "",
        [this]() { TogglePreferences(""); }));
    console_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Console settings...", "",
        [this]() { ShowPreferences("system"); }));
  }
  main_menu->AddChild(std::move(console_menu));

  // Debug menu: tools for diagnosing the emulator rather than playing. The
  // key shortcuts (F3-F5, the numpad time scalar) work whether or not the
  // menu is open.
  auto debug_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Debug");
  {
    debug_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Trace Frame", "F4",
                         std::bind(&EmulatorWindow::GpuTraceFrame, this)));
    debug_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "&Clear Runtime Caches", "F5",
                         std::bind(&EmulatorWindow::GpuClearCaches, this)));
  }
  debug_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    debug_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Reset Time Scalar", "Numpad *",
        std::bind(&EmulatorWindow::CpuTimeScalarReset, this)));
    debug_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Time Scalar /= 2", "Numpad -",
        std::bind(&EmulatorWindow::CpuTimeScalarSetHalf, this)));
    debug_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Time Scalar *= 2", "Numpad +",
        std::bind(&EmulatorWindow::CpuTimeScalarSetDouble, this)));
  }
#if XE_OPTION_PROFILING
  debug_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    debug_menu->AddChild(MenuItem::Create(MenuItem::Type::kString,
                                          "Toggle Profiler &Display", "F3",
                                          []() { Profiler::ToggleDisplay(); }));
    debug_menu->AddChild(MenuItem::Create(MenuItem::Type::kString,
                                          "&Pause/Resume Profiler", "`",
                                          []() { Profiler::TogglePause(); }));
  }
#endif
  debug_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    debug_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Break and Show Guest Debugger",
        "Pause/Break", std::bind(&EmulatorWindow::CpuBreakIntoDebugger, this)));
    debug_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&Break into Host Debugger",
        "Ctrl+Pause/Break",
        std::bind(&EmulatorWindow::CpuBreakIntoHostDebugger, this)));
  }
  debug_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
  {
    auto zar_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Zar Package");
    zar_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Create",
                         std::bind(&EmulatorWindow::CreateZarchive, this)));
    zar_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Extract",
                         std::bind(&EmulatorWindow::ExtractZarchive, this)));
    debug_menu->AddChild(std::move(zar_menu));
  }
  main_menu->AddChild(std::move(debug_menu));

  // Help menu.
  auto help_menu = MenuItem::Create(MenuItem::Type::kPopup, "&Help");
  {
    help_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "FA&Q...", "F1",
                         std::bind(&EmulatorWindow::ShowFAQ, this)));
    help_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    help_menu->AddChild(
        MenuItem::Create(MenuItem::Type::kString, "Game &compatibility...",
                         std::bind(&EmulatorWindow::ShowCompatibility, this)));
    help_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    help_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Build commit on GitHub...", "F2",
        std::bind(&EmulatorWindow::ShowBuildCommit, this)));
    help_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "Recent changes on GitHub...", []() {
          LaunchWebBrowser(
              "https://github.com/xenia-canary/xenia-canary/"
              "compare/" XE_BUILD_COMMIT "..." XE_BUILD_BRANCH);
        }));
    help_menu->AddChild(MenuItem::Create(MenuItem::Type::kSeparator));
    help_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, "&About...",
        [this]() { window_->ShowAboutDialog(); }));
  }
  main_menu->AddChild(std::move(help_menu));

  window_->SetMainMenu(std::move(main_menu));

  window_->SetMainMenuEnabled(false);

  UpdateTitle();

  if (!window_->Open()) {
    XELOGE("Failed to open the platform window");
    return false;
  }

  Profiler::SetUserIO(kZOrderProfiler, window_.get(), nullptr, nullptr);

  return true;
}

const char* EmulatorWindow::GetCvarValueForGuestOutputPaintEffect(
    ui::Presenter::GuestOutputPaintConfig::Effect effect) {
  switch (effect) {
    case ui::Presenter::GuestOutputPaintConfig::Effect::kCas:
      return "cas";
    case ui::Presenter::GuestOutputPaintConfig::Effect::kFsr:
      return "fsr";
    case ui::Presenter::GuestOutputPaintConfig::Effect::kMetalFxSpatial:
      return "metalfx";
    default:
      return "";
  }
}

ui::Presenter::GuestOutputPaintConfig::Effect
EmulatorWindow::GetGuestOutputPaintEffectForCvarValue(
    const std::string& cvar_value) {
  if (cvar_value == GetCvarValueForGuestOutputPaintEffect(
                        ui::Presenter::GuestOutputPaintConfig::Effect::kCas)) {
    return ui::Presenter::GuestOutputPaintConfig::Effect::kCas;
  }
  if (cvar_value == GetCvarValueForGuestOutputPaintEffect(
                        ui::Presenter::GuestOutputPaintConfig::Effect::kFsr)) {
    return ui::Presenter::GuestOutputPaintConfig::Effect::kFsr;
  }
  if (cvar_value ==
      GetCvarValueForGuestOutputPaintEffect(
          ui::Presenter::GuestOutputPaintConfig::Effect::kMetalFxSpatial)) {
    return ui::Presenter::GuestOutputPaintConfig::Effect::kMetalFxSpatial;
  }
  return ui::Presenter::GuestOutputPaintConfig::Effect::kBilinear;
}

ui::Presenter::GuestOutputPaintConfig
EmulatorWindow::GetGuestOutputPaintConfigForCvars() {
  ui::Presenter::GuestOutputPaintConfig paint_config;
  paint_config.SetAllowOverscanCutoff(true);
  paint_config.SetEffect(GetGuestOutputPaintEffectForCvarValue(
      cvars::postprocess_scaling_and_sharpening));
  paint_config.SetCasAdditionalSharpness(
      float(cvars::postprocess_ffx_cas_additional_sharpness));
  paint_config.SetFsrMaxUpsamplingPasses(
      cvars::postprocess_ffx_fsr_max_upsampling_passes);
  paint_config.SetFsrSharpnessReduction(
      float(cvars::postprocess_ffx_fsr_sharpness_reduction));
  paint_config.SetDither(cvars::postprocess_dither);
  return paint_config;
}

uint32_t EmulatorWindow::ActiveRendererMask() const {
  gpu::GraphicsSystem* graphics_system =
      emulator_ ? emulator_->graphics_system() : nullptr;
  if (!graphics_system) {
    return 0;
  }
  if (dynamic_cast<gpu::vulkan::VulkanGraphicsSystem*>(graphics_system)) {
    return kRendererVulkan;
  }
#if XE_GPU_METAL
  if (dynamic_cast<gpu::metal::MetalGraphicsSystem*>(graphics_system)) {
    return kRendererMetal;
  }
#endif  // XE_GPU_METAL
  return 0;
}

void EmulatorWindow::ApplyDisplayConfigForCvars() {
  gpu::GraphicsSystem* graphics_system = emulator_->graphics_system();
  if (!graphics_system) {
    return;
  }

  ui::Presenter* presenter = graphics_system->presenter();
  if (presenter) {
    presenter->SetGuestOutputPaintConfigFromUIThread(
        GetGuestOutputPaintConfigForCvars());
  }
}

void EmulatorWindow::OnKeyDown(ui::KeyEvent& e) {
  if (!emulator_initialized_) {
    return;
  }

  // The Preferences key (preferences_key), checked before the fixed keys so
  // it can take any of the free function keys.
  {
    const std::string& key = cvars::preferences_key;
    const ui::VirtualKey vk = e.virtual_key();
    if ((key == "F6" && vk == ui::VirtualKey::kF6) ||
        (key == "F7" && vk == ui::VirtualKey::kF7) ||
        (key == "F8" && vk == ui::VirtualKey::kF8) ||
        (key == "F10" && vk == ui::VirtualKey::kF10)) {
      TogglePreferences("");
      e.set_handled(true);
      return;
    }
  }

  switch (e.virtual_key()) {
    case ui::VirtualKey::kO: {
      if (!e.is_ctrl_pressed()) {
        return;
      }
      FileOpen();
    } break;
    case ui::VirtualKey::kMultiply: {
      CpuTimeScalarReset();
    } break;
    case ui::VirtualKey::kSubtract: {
      CpuTimeScalarSetHalf();
    } break;
    case ui::VirtualKey::kAdd: {
      CpuTimeScalarSetDouble();
    } break;

    case ui::VirtualKey::kF3: {
      Profiler::ToggleDisplay();
    } break;

    case ui::VirtualKey::kF4: {
      GpuTraceFrame();
    } break;
    case ui::VirtualKey::kF5: {
      GpuClearCaches();
    } break;

    case ui::VirtualKey::kF11: {
      ToggleFullscreen();
    } break;
    case ui::VirtualKey::kF12: {
      TakeScreenshot();
    } break;

    case ui::VirtualKey::kEscape: {
      // Allow users to escape fullscreen (but not enter it).
      if (!window_->IsFullscreen()) {
        return;
      }
      SetFullscreen(false);
    } break;

#ifdef DEBUG
    case ui::VirtualKey::kF7: {
      // Save to file
      // TODO: Choose path based on user input, or from options
      // TODO: Spawn a new thread to do this.
      emulator()->SaveToFile("test.sav");
    } break;
    case ui::VirtualKey::kF8: {
      // Restore from file
      // TODO: Choose path from user
      // TODO: Spawn a new thread to do this.
      emulator()->RestoreFromFile("test.sav");
    } break;
#endif  // #ifdef DEBUG

    case ui::VirtualKey::kPause: {
      CpuBreakIntoDebugger();
    } break;
    case ui::VirtualKey::kCancel: {
      CpuBreakIntoHostDebugger();
    } break;

    case ui::VirtualKey::kF1: {
      ShowFAQ();
    } break;

    case ui::VirtualKey::kF2: {
      ShowBuildCommit();
    } break;

    case ui::VirtualKey::kF9: {
      RunPreviouslyPlayedTitle();
    } break;

    default:
      return;
  }

  e.set_handled(true);
}

void EmulatorWindow::OnMouseDown(const ui::MouseEvent& e) {
  if (imgui_drawer_->IsAnyDialogOpen()) {
    return;
  }

  if (e.button() == ui::MouseEvent::Button::kLeft) {
    ToggleFullscreenOnDoubleClick();
  }
}

void EmulatorWindow::OnMouseUp(const ui::MouseEvent& e) {
  last_mouse_up = steady_clock::now();
}

void EmulatorWindow::TakeScreenshot() {
  xe::ui::RawImage image;

  imgui_drawer_->EnableNotifications(false);

  if (!GetGraphicsSystemPresenter()->CaptureGuestOutput(image) ||
      GetGraphicsSystemPresenter() == nullptr) {
    XELOGE("Failed to capture guest output for screenshot");
    return;
  }

  imgui_drawer_->EnableNotifications(true);
  ExportScreenshot(image);
}

// screenshot_interval: captures the guest output on the UI thread every
// interval, silently (no notification), once a title is running.
// log_fps_interval: logs the guest swap rate every interval.
void EmulatorWindow::AutoScreenshotLoop() {
  using clock = std::chrono::steady_clock;
  const bool burst = cvars::screenshot_interval_ms > 0;
  const auto shot_interval =
      burst ? std::chrono::milliseconds(cvars::screenshot_interval_ms)
            : std::chrono::milliseconds(
                  std::max(cvars::screenshot_interval, 0) * 1000);
  const auto poll_interval =
      std::min(std::chrono::milliseconds(100),
               burst ? shot_interval : std::chrono::milliseconds(100));
  const auto fps_interval = std::chrono::seconds(cvars::log_fps_interval);
  auto next_shot = clock::now() + shot_interval;
  auto fps_time = clock::now();
  uint64_t fps_count = 0;
  bool fps_started = false;
  uint64_t burst_sequence = 0;
  while (!auto_screenshot_stop_.load()) {
    xe::threading::Sleep(poll_interval);
    const auto now = clock::now();
    if (cvars::log_fps_interval > 0 && emulator_->title_id()) {
      gpu::GraphicsSystem* graphics = emulator_->graphics_system();
      gpu::CommandProcessor* command_processor =
          graphics ? graphics->command_processor() : nullptr;
      if (command_processor) {
        const uint64_t count = command_processor->guest_swap_count();
        if (!fps_started) {
          fps_started = true;
          fps_time = now;
          fps_count = count;
        } else if (now - fps_time >= fps_interval) {
          const double seconds =
              std::chrono::duration<double>(now - fps_time).count();
          XELOGI("GuestFps: {:.1f} fps", double(count - fps_count) / seconds);
          fps_time = now;
          fps_count = count;
        }
      }
    }
    if (shot_interval.count() <= 0 || now < next_shot) {
      continue;
    }
    next_shot += shot_interval;
    if (next_shot < now) {
      // A slow capture must not queue a backlog of shots.
      next_shot = now + shot_interval;
    }
    if (!emulator_->title_id()) {
      continue;
    }
    std::string suffix;
    if (burst) {
      gpu::GraphicsSystem* graphics = emulator_->graphics_system();
      gpu::CommandProcessor* command_processor =
          graphics ? graphics->command_processor() : nullptr;
      suffix = fmt::format(
          "{:06}-swap{}", burst_sequence++,
          command_processor ? command_processor->guest_swap_count() : 0);
    }
    app_context().CallInUIThread([this, suffix]() {
      ui::Presenter* presenter = GetGraphicsSystemPresenter();
      xe::ui::RawImage image;
      if (presenter && presenter->CaptureGuestOutput(image)) {
        ExportScreenshot(image, /*notify=*/false, suffix);
      }
    });
  }
}

void EmulatorWindow::ExportScreenshot(const xe::ui::RawImage& image,
                                      bool notify,
                                      const std::string& suffix) {
  auto t = std::time(nullptr);

  // The format is: Year-Month-DayTHours-Minutes-Seconds based off ISO 8601
  std::string datetime =
      fmt::format("{:%Y-%m-%dT%H-%M-%S}", *std::localtime(&t));

  // Get the title id of the game because some titles contain characters that
  // cannot be used as a directory
  std::string title_id;
  if (emulator()->title_id()) {
    title_id = fmt::format("{:08X}", emulator()->title_id());
  } else {
    XELOGE("Failed to get the current title id");
    return;
  }

  // Find where xenia.exe or xenia_canary.exe is located and create a
  // screenshots folder
  auto screenshot_path =
      (cvars::screenshot_path.empty()
           ? xe::filesystem::GetExecutableFolder() / "screenshots"
           : cvars::screenshot_path) /
      title_id;

  if (!std::filesystem::exists(screenshot_path)) {
    std::filesystem::create_directories(screenshot_path);
  }

  std::string filename =
      suffix.empty()
          ? fmt::format("{} - {}.png", title_id, datetime)
          : fmt::format("{} - {} - {}.png", title_id, datetime, suffix);
  SaveImage(screenshot_path / filename, image);
  if (!notify) {
    XELOGI("Screenshot saved: {}",
           xe::path_to_utf8(screenshot_path / filename));
    return;
  }

  const std::string notification_text =
      fmt::format("Screenshot saved: {}", filename);

  app_context_.CallInUIThread([&, notification_text]() {
    new xe::ui::HostNotificationWindow(imgui_drawer(), "Screenshot Created!",
                                       notification_text, 0);
  });
}

// Converts a RawImage into a PNG file
void EmulatorWindow::SaveImage(const std::filesystem::path& filepath,
                               const xe::ui::RawImage& image) {
  auto file = std::ofstream(filepath, std::ios::binary);
  if (!file.is_open()) {
    XELOGE("Failed to open file for writing: {}", filepath);
    return;
  }

  auto result = stbi_write_png_to_func(
      [](void* context, void* data, int size) {
        auto file = reinterpret_cast<std::ofstream*>(context);
        file->write(reinterpret_cast<const char*>(data), size);
      },
      &file, image.width, image.height, 4, image.data.data(),
      (int)image.stride);
  if (result == 0) {
    XELOGE("Failed to write PNG to file: {}", filepath);
    return;
  }
}

void EmulatorWindow::ToggleFullscreenOnDoubleClick() {
  if (cvars::disable_doubleclick_fullscreen) {
    return;
  }

  // this function tests if user has double clicked.
  // if double click was achieved the fullscreen gets toggled
  const auto now = steady_clock::now();  // current mouse event time
  constexpr int16_t mouse_down_max_threshold = 250;
  constexpr int16_t mouse_up_max_threshold = 250;
  constexpr int16_t mouse_up_down_max_delta = 100;
  // max delta to prevent 'chaining' of double clicks with next mouse events

  const auto last_mouse_down_delta = diff_in_ms(now, last_mouse_down);
  if (last_mouse_down_delta >= mouse_down_max_threshold) {
    last_mouse_down = now;
    return;
  }

  const auto last_mouse_up_delta = diff_in_ms(now, last_mouse_up);
  const auto mouse_event_deltas = diff_in_ms(last_mouse_up, last_mouse_down);
  if (last_mouse_up_delta >= mouse_up_max_threshold) {
    return;
  }

  if (mouse_event_deltas < mouse_up_down_max_delta) {
    ToggleFullscreen();
  }
}

void EmulatorWindow::FileDrop(const std::filesystem::path& path) {
  if (!emulator_initialized_) {
    return;
  }

  RunTitle(path);
}

void EmulatorWindow::FileOpen() {
  std::filesystem::path path;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kFile);
  file_picker->set_multi_selection(false);
  file_picker->set_title("Select Content Package");
  file_picker->set_extensions({
      {"Supported Files", "*.iso;*.xex;*.zar;*.*"},
      {"Disc Image (*.iso)", "*.iso"},
      {"Disc Archive (*.zar)", "*.zar"},
      {"Xbox Executable (*.xex)", "*.xex"},
      //{"Content Package (*.xcp)", "*.xcp" },
      {"All Files (*.*)", "*.*"},
  });
  if (file_picker->Show(window_.get())) {
    auto selected_files = file_picker->selected_files();
    if (!selected_files.empty()) {
      path = selected_files[0];
    }
    // Only run the title if a file is selected
    RunTitle(path);
  }
}

void EmulatorWindow::FileClose() {
  FreezeGame(false);
  emulator_->TerminateTitle();
  // Back to the dashboard, as the console went when a game was quit.
  ShowStartScreen();
  // The guest has stopped producing frames, but the presenter still holds the
  // last one in its mailbox, so the window would look frozen/paused. Push an
  // inactive (0x0) guest-output frame to clear the screen to black - this is
  // the presenter's own active->inactive blanking path, just driven by us.
  if (ui::Presenter* presenter = GetGraphicsSystemPresenter()) {
    presenter->RefreshGuestOutput(
        0, 0, 0, 0,
        [](ui::Presenter::GuestOutputRefreshContext&) { return true; });
  }
}

void EmulatorWindow::RelaunchLastTitle(bool warm) {
  FreezeGame(false);
#if XE_PLATFORM_MAC
  if (last_media_path_.empty()) {
    XELOGW("RelaunchLastTitle: no media loaded yet; ignoring.");
    return;
  }
  if (warm) {
    // Reboot: terminate + finalize shader/pipeline storage so the relaunch
    // boots warm (no shader re-translation).
    FlushBeforeRelaunch();
  } else {
    // Full Reboot: skip all GPU work, just persist config and relaunch from a
    // fresh process. Boots cold (re-translates shaders) for a clean slate, and
    // is bulletproof - no terminate, no GPU touch, so nothing to deadlock or
    // race. Also the safe fallback if the warm path ever misbehaves.
    config::SaveConfig();
  }
  // A user-driven relaunch starts a fresh title-relaunch chain.
  unsetenv("XENIA_TITLE_RELAUNCH_HOPS");
  xe::ui::RelaunchWithTarget(xe::path_to_utf8(last_media_path_));
  // RelaunchWithTarget only returns on failure.
  XELOGE("Relaunch failed; staying in the current state.");
#else
  XELOGW("RelaunchLastTitle: re-exec relaunch is macOS-only.");
#endif  // XE_PLATFORM_MAC
}

void EmulatorWindow::UpdateLifecycleMenuState() {
  const bool running = emulator_->is_title_open();
  const bool have_media = !last_media_path_.empty();
  // Stop only applies while a title is running; Reboot/Full Reboot need media
  // to have been loaded (they re-exec the remembered path, from running OR
  // stopped). Load/Open stays always-available.
  if (stop_menu_item_) {
    stop_menu_item_->SetEnabled(running);
  }
  if (reboot_menu_item_) {
    reboot_menu_item_->SetEnabled(have_media);
  }
  if (full_reboot_menu_item_) {
    full_reboot_menu_item_->SetEnabled(have_media);
  }
  if (window_) {
    window_->CompleteMainMenuItemsUpdate();
  }
}

void EmulatorWindow::FlushBeforeRelaunch() {
  // Quiesce the guest FIRST. graphics_system->Pause() below waits on the
  // command-processor worker from the UI thread (CallInThread + fence.Wait);
  // while a title is actively rendering the worker is mid-ExecutePrimaryBuffer
  // (presenting, needing this very thread), so pausing then would deadlock ->
  // beachball. Terminating the title stops guest GPU submission so the worker
  // drains and goes idle, making the pause safe. This is the same crash-safe
  // terminate Stop uses.
  if (emulator_->is_title_open()) {
    emulator_->TerminateTitle();
  }

  // Persist cvar overrides so they survive the re-exec (a re-exec replaces the
  // process image with no destructors, so OnDestroy's SaveConfig never runs).
  config::SaveConfig();

  // Finalize the host shader/pipeline storage so the relaunched title boots warm
  // instead of re-translating every shader from cold. SaveShaderStorageToDisk
  // drains the storage-writer thread and closes the storage files (see
  // VulkanCommandProcessor::SaveShaderStorageToDisk).
  //
  // We deliberately do NOT call graphics_system->Pause() first (OnDestroy does,
  // but it runs after the emulator thread is already shut down). Pause() blocks
  // the UI thread on the command-processor worker via fence.Wait(); if that
  // worker is still draining a present it needs this very thread -> deadlock
  // (beachball), which terminate-first does not fully close. The storage
  // finalize runs on the independent storage-writer thread (never blocked on the
  // UI thread, so it joins cleanly) and the guest is already terminated, so no
  // new shaders are being enqueued - the pause buys nothing here but the hang.
  if (auto* graphics_system = emulator_->graphics_system()) {
    if (auto* command_processor = graphics_system->command_processor()) {
      command_processor->SaveShaderStorageToDisk();
    }
  }
}

void EmulatorWindow::InstallContent() {
  std::vector<std::filesystem::path> paths;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kFile);
  file_picker->set_multi_selection(true);
  file_picker->set_title("Select Content Package");
  file_picker->set_extensions({
      {"All Files (*.*)", "*.*"},
  });
  if (file_picker->Show(window_.get())) {
    paths = file_picker->selected_files();
  }

  if (paths.empty()) {
    return;
  }

  std::shared_ptr<std::vector<Emulator::ContentInstallEntry>>
      content_installation_status =
          std::make_shared<std::vector<Emulator::ContentInstallEntry>>();

  for (const auto& path : paths) {
    content_installation_status->push_back({path});
  }

  for (auto& entry : *content_installation_status) {
    emulator_->ProcessContentPackageHeader(entry.path_, entry);
  }

  auto installationThread = std::thread([this, content_installation_status] {
    for (auto& entry : *content_installation_status) {
      emulator_->InstallContentPackage(entry.path_, entry);
    }
  });
  installationThread.detach();

  new ContentInstallDialog(imgui_drawer_.get(), *this,
                           content_installation_status);
}

void EmulatorWindow::ExtractZarchive() {
  std::vector<std::filesystem::path> zarchive_files;
  std::filesystem::path extract_dir;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kFile);
  file_picker->set_multi_selection(true);
  file_picker->set_title("Select Zar Package");
  file_picker->set_extensions({
      {"Zarchive Files (*.zar)", "*.zar"},
  });

  if (file_picker->Show(window_.get())) {
    zarchive_files = file_picker->selected_files();
  }

  if (zarchive_files.empty()) {
    return;
  }

  file_picker->set_type(ui::FilePicker::Type::kDirectory);
  file_picker->set_title("Select Directory to Extract");

  if (file_picker->Show(window_.get())) {
    extract_dir = file_picker->selected_files().front();
  }

  if (extract_dir.empty()) {
    return;
  }

  std::string extract_overview = "";

  for (auto& zarchive_file_path : zarchive_files) {
    extract_overview += "\n" + path_to_utf8(zarchive_file_path);
  }

  app_context_.CallInUIThread([&]() {
    new xe::ui::HostNotificationWindow(imgui_drawer(), "Extracting...",
                                       string_util::trim(extract_overview), 0);
  });

  auto run = [this, extract_dir, zarchive_files]() -> void {
    std::string summary = "";

    for (auto& zarchive_file_path : zarchive_files) {
      // Normalize the path and make absolute.
      auto abs_path = std::filesystem::absolute(zarchive_file_path);
      std::filesystem::path abs_extract_dir;

      if (zarchive_files.size() > 1) {
        abs_extract_dir =
            std::filesystem::absolute((extract_dir / abs_path.stem()));
      } else {
        abs_extract_dir = std::filesystem::absolute(extract_dir);
      }

      XELOGI("Extracting zar package: {}\n",
             zarchive_file_path.filename().string());

      auto result =
          emulator_->ExtractZarchivePackage(abs_path, abs_extract_dir);

      if (result != X_STATUS_SUCCESS) {
        std::error_code ec;

        if (!std::filesystem::is_empty(abs_extract_dir)) {
          std::filesystem::remove(abs_extract_dir, ec);
        }

        summary += fmt::format("\nFailed: {}", zarchive_file_path);

        XELOGE("Failed to extract Zarchive package.", result);
      } else {
        summary += fmt::format("\nSuccess: {}", abs_extract_dir);
      }
    }

    new xe::ui::HostNotificationWindow(imgui_drawer(), "Zar Extraction Summary",
                                       string_util::trim(summary), 0);
  };

  auto zarThread = std::thread(run);
  zarThread.detach();
}

void EmulatorWindow::CreateZarchive() {
  std::vector<std::filesystem::path> content_dirs;
  std::filesystem::path zarchive_dir;

  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kDirectory);
  file_picker->set_multi_selection(true);
  file_picker->set_title("Select Contents");

  if (file_picker->Show(window_.get())) {
    content_dirs = file_picker->selected_files();
  }

  if (content_dirs.empty()) {
    return;
  }

  if (content_dirs.size() == 1) {
    file_picker->set_mode(ui::FilePicker::Mode::kSave);
    file_picker->set_type(ui::FilePicker::Type::kFile);
    file_picker->set_multi_selection(false);
    file_picker->set_file_name(content_dirs.front().filename().string());
    file_picker->set_default_extension("zar");
    file_picker->set_title("Zarchive File");
    file_picker->set_extensions({
        {"Zarchive File (*.zar)", "*.zar"},
    });
  } else {
    file_picker->set_title("Output Directory");
  }

  if (file_picker->Show(window_.get())) {
    zarchive_dir = file_picker->selected_files().front();
  }

  if (zarchive_dir.empty()) {
    return;
  }

  std::string create_overview = "";

  std::map<std::filesystem::path, std::filesystem::path> zarchive_files{};

  for (auto& content_path : content_dirs) {
    // Normalize the path and make absolute.
    auto abs_content_dir = std::filesystem::absolute(content_path);
    std::filesystem::path abs_zarchive_file;

    if (content_dirs.size() > 1) {
      abs_zarchive_file = std::filesystem::absolute(
          (zarchive_dir / abs_content_dir.filename().concat(".zar")));
    } else {
      abs_zarchive_file = std::filesystem::absolute(zarchive_dir);
    }

    zarchive_files[content_path] = abs_zarchive_file;

    create_overview += "\n" + path_to_utf8(abs_zarchive_file);
  }

  app_context_.CallInUIThread([&]() {
    new xe::ui::HostNotificationWindow(imgui_drawer(), "Creating...",
                                       string_util::trim(create_overview), 0);
  });

  auto run = [this, zarchive_files]() -> void {
    std::string summary = "";

    for (auto const& [content_path, zarchive_file] : zarchive_files) {
      // Normalize the path and make absolute.
      auto abs_content_dir = std::filesystem::absolute(content_path);

      XELOGI("Creating zar package: {}\n", zarchive_file.filename().string());

      auto result =
          emulator_->CreateZarchivePackage(abs_content_dir, zarchive_file);

      if (result != X_ERROR_SUCCESS) {
        std::error_code ec;

        // delete incomplete output file
        std::filesystem::remove(zarchive_file, ec);

        summary += fmt::format("\nFailed: {}", abs_content_dir);

        XELOGE("Failed to create Zarchive package.", result);
      } else {
        summary += fmt::format("\nSuccess: {}", zarchive_file);
      }
    }

    new xe::ui::HostNotificationWindow(imgui_drawer(), "Zar Creation Summary",
                                       string_util::trim(summary), 0);
  };

  auto zarThread = std::thread(run);
  zarThread.detach();
}

void EmulatorWindow::ShowContentDirectory() {
  auto content_root = emulator_->content_root();

  if (!std::filesystem::exists(content_root)) {
    std::filesystem::create_directories(content_root);
  }

  LaunchFileExplorer(content_root);
}

void EmulatorWindow::CpuTimeScalarReset() {
  Clock::set_guest_time_scalar(1.0);
  UpdateTitle();
}

void EmulatorWindow::CpuTimeScalarSetHalf() {
  Clock::set_guest_time_scalar(Clock::guest_time_scalar() / 2.0);
  UpdateTitle();
}

void EmulatorWindow::CpuTimeScalarSetDouble() {
  Clock::set_guest_time_scalar(Clock::guest_time_scalar() * 2.0);
  UpdateTitle();
}

void EmulatorWindow::CpuBreakIntoDebugger() {
  if (!cvars::debug) {
    xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(), "Xenia Debugger",
                                        "Xenia must be launched with the "
                                        "--debug flag in order to enable "
                                        "debugging.");
    return;
  }
  auto processor = emulator()->processor();
  if (processor->execution_state() == cpu::ExecutionState::kRunning) {
    // Currently running, so interrupt (and show the debugger).
    processor->Pause();
  } else {
    // Not running, so just bring the debugger into focus.
    processor->ShowDebugger();
  }
}

void EmulatorWindow::CpuBreakIntoHostDebugger() { xe::debugging::Break(); }

void EmulatorWindow::GpuTraceFrame() {
  emulator()->graphics_system()->RequestFrameTrace();
}

void EmulatorWindow::GpuClearCaches() {
  emulator()->graphics_system()->ClearCaches();
}

void EmulatorWindow::SetFullscreen(bool fullscreen_) {
  if (window_->IsFullscreen() == fullscreen_) {
    return;
  }

  OVERRIDE_bool(fullscreen, fullscreen_);

  window_->SetFullscreen(fullscreen_);
  window_->SetCursorVisibility(fullscreen_
                                   ? ui::Window::CursorVisibility::kAutoHidden
                                   : ui::Window::CursorVisibility::kVisible);
}

void EmulatorWindow::ToggleFullscreen() {
  SetFullscreen(!window_->IsFullscreen());
}

// --- Preferences (the blades) ------------------------------------------------

bool EmulatorWindow::IsPreferencesVisible() const {
#if XE_PLATFORM_MAC
  return blades_overlay_ && blades_overlay_->visible();
#else
  return false;
#endif
}

void EmulatorWindow::ShowPreferences(const std::string& blade) {
#if XE_PLATFORM_MAC
  if (!blades_overlay_) {
    auto* mac_window = static_cast<ui::MacWindow*>(window_.get());
    if (!mac_window->native_container_view()) {
      return;
    }
    blades_model_ = std::make_unique<BladesModel>(*this);
    BladesOverlay::Callbacks callbacks;
    callbacks.on_message = [this](const std::string& json) {
      HandlePreferencesMessage(json);
    };
    callbacks.on_tick = [this]() {
      if (blades_model_ && blades_model_->Poll()) {
        blades_overlay_->PushState(blades_model_->BuildStateJson());
      }
      PumpPreferencesController();
    };
    blades_overlay_ = std::make_unique<BladesOverlay>(
        mac_window->native_container_view(), mac_window->native_game_view(),
        TitleIconPath(0).parent_path(), DashboardSoundsPath(),
        emulator_->content_root(), emulator_->storage_root(),
        std::move(callbacks));
  }
  if (!blades_overlay_->visible()) {
    // Opening Preferences is opening the console's system UI: the running
    // game is told (most pause, as under the 360 guide) and sees no input.
    auto* xam_state = emulator_->kernel_state()
                          ? emulator_->kernel_state()->xam_state()
                          : nullptr;
    if (xam_state && emulator_->is_title_open() && !xam_state->IsUIActive()) {
      emulator_->kernel_state()->BroadcastNotification(kXNotificationSystemUI,
                                                       true);
      xam_state->is_xam_dialog_present_.store(true);
      blades_raised_system_ui_ = true;
    }
    if (cvars::dashboard_freezes_game && emulator_->is_title_open()) {
      FreezeGame(true);
    }
    // A Guide still held from the press that opened it isn't a new press
    // (the page would close again).
    blades_pad_buttons_ = X_INPUT_GAMEPAD_GUIDE;
  }
  blades_overlay_->Show(blade);
  blades_visible_ = true;
  // The page asks for state when it first loads; after that it is refreshed
  // on every open (a title may have started or a controller arrived since).
  blades_overlay_->PushState(blades_model_->BuildStateJson());
#endif
}

void EmulatorWindow::HidePreferences() {
#if XE_PLATFORM_MAC
  if (!blades_overlay_ || !blades_overlay_->visible()) {
    return;
  }
  blades_model_->CancelCapture();
  blades_overlay_->Hide();
  blades_visible_ = false;
  FreezeGame(false);
  if (blades_raised_system_ui_) {
    blades_raised_system_ui_ = false;
    if (emulator_->kernel_state()) {
      emulator_->kernel_state()->xam_state()->is_xam_dialog_present_.store(
          false);
      emulator_->kernel_state()->BroadcastNotification(kXNotificationSystemUI,
                                                       false);
    }
  }
#endif
}

void EmulatorWindow::FreezeGame(bool freeze) {
  if (freeze == game_frozen_) {
    return;
  }
  cpu::Processor* processor = emulator_->processor();
  if (freeze) {
    if (!processor || !emulator_->is_title_open()) {
      return;
    }
    {
      // Suspended while holding a host lock, a guest thread would keep it
      // held; the dashboard reads controllers through the input system's,
      // so it is held here while the threads stop and no guest thread can
      // be inside it. (AudioSystem::Pause is not used: its handshake waits on
      // a worker that, under steady audio, never sees the request.)
      hid::InputSystem* input = emulator_->input_system();
      std::unique_lock<xe_unlikely_mutex> input_lock;
      if (input) {
        input_lock = input->lock();
      }
      processor->SuspendGuestThreads();
    }
    // Some games mix on the host-side audio worker, which keeps running;
    // silence the output (in memory only, never saved) until they resume.
    mute_before_freeze_ = cvars::mute;
    cvars::mute = true;
    game_frozen_ = true;
    XELOGI("Dashboard: game frozen");
  } else {
    if (processor) {
      processor->ResumeGuestThreads();
    }
    cvars::mute = mute_before_freeze_;
    game_frozen_ = false;
    XELOGI("Dashboard: game resumed");
  }
}

void EmulatorWindow::TogglePreferences(const std::string& blade) {
  if (IsPreferencesVisible()) {
    HidePreferences();
  } else {
    ShowPreferences(blade);
  }
}

void EmulatorWindow::HandlePreferencesMessage(const std::string& json) {
#if XE_PLATFORM_MAC
  if (!blades_model_) {
    return;
  }
  XELOGD("Blades: message {}", json);
  switch (blades_model_->HandleMessage(json)) {
    case BladesModel::Result::kNone:
      break;
    case BladesModel::Result::kPush:
      blades_overlay_->PushState(blades_model_->BuildStateJson());
      break;
    case BladesModel::Result::kClose:
      HidePreferences();
      break;
    case BladesModel::Result::kCloseThen: {
      std::function<void()> then = blades_model_->TakeDeferred();
      HidePreferences();
      if (then) {
        // Not from inside the page's message: a picker or a launch can take
        // a while, and a launch tears down state the page is using.
        app_context().CallInUIThreadDeferred(std::move(then));
      }
    } break;
    case BladesModel::Result::kRelaunch:
      HidePreferences();
      // Not from inside the page's message: relaunching tears the window down.
      app_context().CallInUIThreadDeferred(
          [this]() { RelaunchLastTitle(true); });
      break;
  }
#endif
}

void EmulatorWindow::PumpPreferencesController() {
#if XE_PLATFORM_MAC
  // Any connected controller drives the blades. The page also reads the
  // Gamepad API; it merges the two, so a press seen by both counts once.
  auto* input = emulator_->input_system();
  if (!input || !blades_overlay_) {
    return;
  }
  uint32_t buttons = 0;
  {
    auto lock = input->lock();
    for (uint32_t user = 0; user < XUserMaxUserCount; ++user) {
      X_INPUT_STATE state;
      if (input->GetStateSnapshot(user, X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD,
                                  &state) != X_ERROR_SUCCESS) {
        continue;
      }
      buttons |= state.gamepad.buttons;
      // The left stick counts as the d-pad.
      constexpr int16_t kStick = 18000;
      const int16_t x = state.gamepad.thumb_lx, y = state.gamepad.thumb_ly;
      if (y > kStick) buttons |= X_INPUT_GAMEPAD_DPAD_UP;
      if (y < -kStick) buttons |= X_INPUT_GAMEPAD_DPAD_DOWN;
      if (x < -kStick) buttons |= X_INPUT_GAMEPAD_DPAD_LEFT;
      if (x > kStick) buttons |= X_INPUT_GAMEPAD_DPAD_RIGHT;
    }
  }
  static constexpr std::pair<uint32_t, const char*> kButtons[] = {
      {X_INPUT_GAMEPAD_DPAD_UP, "up"},
      {X_INPUT_GAMEPAD_DPAD_DOWN, "down"},
      {X_INPUT_GAMEPAD_DPAD_LEFT, "left"},
      {X_INPUT_GAMEPAD_DPAD_RIGHT, "right"},
      {X_INPUT_GAMEPAD_A, "a"},
      {X_INPUT_GAMEPAD_B, "b"},
      {X_INPUT_GAMEPAD_Y, "y"},
      {X_INPUT_GAMEPAD_LEFT_SHOULDER, "lb"},
      {X_INPUT_GAMEPAD_RIGHT_SHOULDER, "rb"},
      {X_INPUT_GAMEPAD_X, "x"},
      {X_INPUT_GAMEPAD_GUIDE, "guide"},
  };
  const uint32_t changed = buttons ^ blades_pad_buttons_;
  blades_pad_buttons_ = buttons;
  for (const auto& [mask, name] : kButtons) {
    if (changed & mask) {
      blades_overlay_->PadButton(name, (buttons & mask) != 0);
    }
  }
#endif
}

void EmulatorWindow::ToggleProfilesConfigDialog() {
#if XE_PLATFORM_MAC
  // Profiles live in the dashboard (Profiles blade).
  ShowPreferences("profiles");
  return;
#endif
  if (!profile_config_dialog_) {
    disable_hotkeys_ = true;

    if (emulator_->kernel_state()->xam_state()->IsUIActive()) {
      return;
    }

    emulator_->kernel_state()->BroadcastNotification(kXNotificationSystemUI,
                                                     true);
    emulator_->kernel_state()->xam_state()->is_xam_dialog_present_.store(true);

    profile_config_dialog_ =
        std::make_unique<ProfileConfigDialog>(imgui_drawer_.get(), this);
  } else {
    disable_hotkeys_ = false;
    emulator_->kernel_state()->BroadcastNotification(kXNotificationSystemUI,
                                                     false);
    if (profile_config_dialog_->IsClosing()) {
      profile_config_dialog_.release();
    } else {
      profile_config_dialog_.reset();
    }
    emulator_->kernel_state()->xam_state()->is_xam_dialog_present_.store(false);
  }
}

void EmulatorWindow::ToggleControllerVibration() {
  auto input_sys = emulator()->input_system();
  if (input_sys) {
    auto input_lock = input_sys->lock();

    input_sys->ToggleVibration();

    if (emulator_->kernel_state()) {
      emulator_->kernel_state()->BroadcastNotification(
          kXNotificationSystemProfileSettingChanged,
          static_cast<uint32_t>(input_sys->GetConnectedSlots().count()));
    }
  }
}

void EmulatorWindow::ShowCompatibility() {
  const std::string_view base_url =
      "https://github.com/xenia-canary/game-compatibility/issues";
  std::string url;
  // Avoid searching for a title ID of "00000000".
  uint32_t title_id = emulator_->title_id();
  if (!title_id) {
    url = base_url;
  } else {
    url = fmt::format("{}?q=is%3Aissue+is%3Aopen+{:08X}", base_url, title_id);
  }
  LaunchWebBrowser(url);
}

void EmulatorWindow::ShowFAQ() {
  LaunchWebBrowser("https://github.com/xenia-canary/xenia-canary/wiki/FAQ");
}

void EmulatorWindow::ShowBuildCommit() {
#ifdef XE_BUILD_IS_PR
  LaunchWebBrowser(
      "https://github.com/xenia-canary/xenia-canary/pull/" XE_BUILD_PR_NUMBER);
#else
  LaunchWebBrowser(
      "https://github.com/xenia-canary/xenia-canary/commit/" XE_BUILD_COMMIT);
#endif
}

void EmulatorWindow::UpdateTitle() {
  xe::StringBuffer sb;
  sb.Append(base_title_);

  // Title information, if available
  if (emulator()->is_title_open()) {
    sb.AppendFormat(" | [{:08X}", emulator()->title_id());
    auto title_version = emulator()->title_version();
    if (!title_version.empty()) {
      sb.Append(" v");
      sb.Append(title_version);
    }
    sb.Append("]");

    auto title_name = emulator()->title_name();
    if (!title_name.empty()) {
      sb.Append(" ");
      sb.Append(title_name);
    }
  }

  // Graphics system name, if available
  auto graphics_system = emulator()->graphics_system();
  if (graphics_system) {
    auto graphics_name = graphics_system->name();
    if (!graphics_name.empty()) {
      sb.Append(" <");
      sb.Append(graphics_name);
      sb.Append(">");
    }
  }

  if (Clock::guest_time_scalar() != 1.0) {
    sb.AppendFormat(" (@{:.2f}x)", Clock::guest_time_scalar());
  }

  if (initializing_shader_storage_) {
    sb.Append(" (Preloading shaders\u2026)");
  }

  patcher::Patcher* patcher = emulator()->patcher();
  if (patcher && patcher->IsAnyPatchApplied()) {
    sb.Append(" [Patches Applied]");
  }

  patcher::PluginLoader* pluginloader = emulator()->plugin_loader();
  if (pluginloader && pluginloader->IsAnyPluginLoaded()) {
    sb.Append(" [Plugins Loaded]");
  }

  window_->SetTitle(sb.to_string_view());
}

void EmulatorWindow::SetInitializingShaderStorage(bool initializing) {
  if (initializing_shader_storage_ == initializing) {
    return;
  }
  initializing_shader_storage_ = initializing;
  UpdateTitle();
}

// Notes:
// SDL and XInput both support the guide button
//
// Assumes titles do not use the guide button.
// For titles that do such as dashboards these titles could be excluded based on
// their title ID.
//
// Xbox Gamebar:
// If the Xbox Gamebar overlay is enabled Windows will consume the guide
// button's input, this can be seen using hid-demo.
//
// Workaround: Detect if the Xbox Gamebar overlay is enabled then use the BACK
// button instead of the GUIDE button. Therefore BACK and GUIDE are reserved
// buttons for hotkeys.
//
// This is not an issue with DualShock controllers because Windows will not
// open the gamebar overlay using the PlayStation menu button.
//
// Xbox One S Controller:
// The guide button on this controller is very buggy no idea why.
// Using xinput usually registers after a double tap.
// Doesn't work at all using SDL.
// Needs more testing.
//
// Steam:
// If guide button focus is enabled steam will open.
// Steam uses BACK + GUIDE to open an On-Screen keyboard, however this is not a
// problem since both these buttons are reserved.
const std::map<int, EmulatorWindow::ControllerHotKey> controller_hotkey_map = {
    // Must use the Guide Button for all pass through hotkeys
    {X_INPUT_GAMEPAD_A | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ReadbackResolve,
         "A + Guide = Toggle Readback Resolve (Vulkan only)", true)},
    {X_INPUT_GAMEPAD_B | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleLogging,
         "B + Guide = Toggle between loglevel set in config and the 'Disabled' "
         "loglevel.",
         true, true)},
    {X_INPUT_GAMEPAD_Y | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleFullscreen,
         "Y + Guide = Toggle Fullscreen", true)},
    {X_INPUT_GAMEPAD_X | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ClearMemoryPageState,
         "X + Guide = Toggle Clear Memory Page State (Vulkan only)", true)},

    {X_INPUT_GAMEPAD_RIGHT_SHOULDER | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ClearGPUCache,
         "Right Shoulder + Guide = Clear GPU Cache", true)},
    {X_INPUT_GAMEPAD_LEFT_SHOULDER | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleControllerVibration,
         "Left Shoulder + Guide = Toggle Controller Vibration", true)},

    // CPU Time Scalar with no rumble feedback
    {X_INPUT_GAMEPAD_DPAD_DOWN | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::CpuTimeScalarSetHalf,
         "D-PAD Down + Guide = Half CPU Scalar")},
    {X_INPUT_GAMEPAD_DPAD_UP | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::CpuTimeScalarSetDouble,
         "D-PAD Up + Guide = Double CPU Scalar")},
    {X_INPUT_GAMEPAD_DPAD_RIGHT | X_INPUT_GAMEPAD_GUIDE,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::CpuTimeScalarReset,
         "D-PAD Right + Guide = Reset CPU Scalar")},

    // non-pass through hotkeys
    {X_INPUT_GAMEPAD_Y, EmulatorWindow::ControllerHotKey(
                            EmulatorWindow::ButtonFunctions::ToggleFullscreen,
                            "Y = Toggle Fullscreen", true, false)},
    {X_INPUT_GAMEPAD_START, EmulatorWindow::ControllerHotKey(
                                EmulatorWindow::ButtonFunctions::RunTitle,
                                "Start = Run Selected Title", false, false)},
    {X_INPUT_GAMEPAD_BACK | X_INPUT_GAMEPAD_START,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::ToggleLogging,
         "Back + Start = Toggle between loglevel set in config and the "
         "'Disabled' loglevel.",
         false, false)},
    {X_INPUT_GAMEPAD_DPAD_DOWN,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::IncTitleSelect,
         "D-PAD Down = Title Selection +1", true, false)},
    {X_INPUT_GAMEPAD_DPAD_UP,
     EmulatorWindow::ControllerHotKey(
         EmulatorWindow::ButtonFunctions::DecTitleSelect,
         "D-PAD Up = Title Selection -1", true, false)}};

namespace {
// The set of ButtonFunctions ids the user has disabled, parsed from the
// comma-separated cvar.
std::set<int> ParseDisabledHotkeys() {
  std::set<int> disabled;
  for (const std::string_view token :
       utf8::split(cvars::disabled_controller_hotkeys, ",", true)) {
    if (token.empty()) {
      continue;
    }
    disabled.insert(string_util::from_string<int>(token, false));
  }
  return disabled;
}

std::string SerializeDisabledHotkeys(const std::set<int>& disabled) {
  std::string out;
  for (int function : disabled) {
    if (!out.empty()) {
      out += ",";
    }
    out += std::to_string(function);
  }
  return out;
}
}  // namespace

std::vector<EmulatorWindow::ControllerHotkeyInfo>
EmulatorWindow::GetControllerHotkeys() const {
  const std::set<int> disabled = ParseDisabledHotkeys();
  std::vector<ControllerHotkeyInfo> hotkeys;
  std::set<int> seen_functions;
  for (const auto& [buttons, hotkey] : controller_hotkey_map) {
    const int function = static_cast<int>(hotkey.function);
    if (hotkey.pretty.empty() || !seen_functions.insert(function).second) {
      continue;
    }
    hotkeys.push_back({function, hotkey.pretty,
                       disabled.find(function) == disabled.end(),
                       hotkey.title_passthru});
  }
  return hotkeys;
}

void EmulatorWindow::SetControllerHotkeyEnabled(int function, bool enabled) {
  std::set<int> disabled = ParseDisabledHotkeys();
  if (enabled) {
    disabled.erase(function);
  } else {
    disabled.insert(function);
  }
  OVERRIDE_string(disabled_controller_hotkeys, SerializeDisabledHotkeys(disabled));
  config::SaveConfig();
}

EmulatorWindow::ControllerHotKey EmulatorWindow::ProcessControllerHotkey(
    int buttons) {
  // Default return value
  EmulatorWindow::ControllerHotKey Unknown_hotkey = {};

  if (buttons == 0) {
    return Unknown_hotkey;
  }

  if (disable_hotkeys_.load()) {
    return Unknown_hotkey;
  }

  // Hotkey cool-down to prevent toggling too fast
  constexpr std::chrono::milliseconds delay(75);

  // If the Xbox Gamebar is enabled or the Guide button is disabled then
  // replace the Guide button with the Back button without redeclaring the key
  // mappings
  if (IsUseNexusForGameBarEnabled() || !cvars::guide_button) {
    if ((buttons & X_INPUT_GAMEPAD_BACK) == X_INPUT_GAMEPAD_BACK) {
      buttons &= ~X_INPUT_GAMEPAD_BACK;
      buttons |= X_INPUT_GAMEPAD_GUIDE;
    }
  }

  auto it = controller_hotkey_map.find(buttons);
  if (it == controller_hotkey_map.end()) {
    return Unknown_hotkey;
  }

  // Skip hotkeys the user has disabled in the Controllers UI.
  const std::set<int> disabled_hotkeys = ParseDisabledHotkeys();
  if (disabled_hotkeys.find(static_cast<int>(it->second.function)) !=
      disabled_hotkeys.end()) {
    return Unknown_hotkey;
  }

  // Do not activate hotkeys that are not intended for activation during
  // gameplay
  if (emulator_->is_title_open()) {
    // If non-pass through (menu hoykeys) or hotkeys disabled then return
    if (!it->second.title_passthru || !cvars::controller_hotkeys) {
      return Unknown_hotkey;
    }
  }

  std::string notificationTitle = "";
  std::string notificationDesc = "";

  EmulatorWindow::ControllerHotKey button_combination = it->second;

  switch (button_combination.function) {
    case ButtonFunctions::ToggleFullscreen:
      app_context().CallInUIThread([this]() { ToggleFullscreen(); });

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::RunTitle: {
      if (selected_title_index == -1) {
        selected_title_index++;
      }

      if (selected_title_index < recently_launched_titles_.size()) {
        app_context().CallInUIThread([this]() {
          RunTitle(
              recently_launched_titles_[selected_title_index].path_to_file);
        });
      }
    } break;
    case ButtonFunctions::ClearMemoryPageState:
      // Only the Vulkan renderer reads this; elsewhere say so instead of
      // silently flipping (and saving) a setting with no effect.
      if (ActiveRendererMask() != kRendererVulkan) {
        notificationTitle = "Toggle Clear Memory Page State";
        notificationDesc = "Vulkan renderer only";
        break;
      }
      ToggleGPUSetting(GPUSetting::ClearMemoryPageState);

      // Assume the user wants ClearCaches as well
      if (cvars::clear_memory_page_state) {
        GpuClearCaches();
      }

      notificationTitle = "Toggle Clear Memory Page State";
      notificationDesc =
          cvars::clear_memory_page_state ? "Enabled" : "Disabled";

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::ReadbackResolve:
      // Only the Vulkan renderer reads this (see ClearMemoryPageState).
      if (ActiveRendererMask() != kRendererVulkan) {
        notificationTitle = "Readback Resolve Mode";
        notificationDesc = "Vulkan renderer only";
        break;
      }
      CycleReadbackResolve();

      notificationTitle = "Readback Resolve Mode";
      notificationDesc = cvars::readback_resolve;

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::CpuTimeScalarSetHalf:
      CpuTimeScalarSetHalf();

      notificationTitle = "Time Scalar";
      notificationDesc =
          fmt::format("Decreased to {}", Clock::guest_time_scalar());
      break;
    case ButtonFunctions::CpuTimeScalarSetDouble:
      CpuTimeScalarSetDouble();

      notificationTitle = "Time Scalar";
      notificationDesc =
          fmt::format("Increased to {}", Clock::guest_time_scalar());
      break;
    case ButtonFunctions::CpuTimeScalarReset:
      CpuTimeScalarReset();

      notificationTitle = "Time Scalar";
      notificationDesc = fmt::format("Reset to {}", Clock::guest_time_scalar());
      break;
    case ButtonFunctions::ClearGPUCache:
      GpuClearCaches();

      notificationTitle = "Clear GPU Cache";
      notificationDesc = "Complete";

      // Extra Sleep
      xe::threading::Sleep(delay);
      break;
    case ButtonFunctions::ToggleControllerVibration: {
      ToggleControllerVibration();

      bool vibration = false;

      auto input_sys = emulator()->input_system();
      if (input_sys) {
        vibration = input_sys->GetVibrationCvar();
      }

      notificationTitle = "Toggle Controller Vibration";
      notificationDesc = vibration ? "Enabled" : "Disabled";

      // Extra Sleep
      xe::threading::Sleep(delay);
    } break;
    case ButtonFunctions::IncTitleSelect:
      selected_title_index++;
      break;
    case ButtonFunctions::DecTitleSelect:
      selected_title_index--;
      break;
    case ButtonFunctions::ToggleLogging: {
      logging::ToggleLogLevel();

      notificationTitle = "Toggle Logging";

      LogLevel level = static_cast<LogLevel>(logging::internal::GetLogLevel());
      notificationDesc = level == LogLevel::Disabled ? "Disabled" : "Enabled";
    } break;
    case ButtonFunctions::Unknown:
    default:
      break;
  }

  if ((button_combination.function == ButtonFunctions::IncTitleSelect ||
       button_combination.function == ButtonFunctions::DecTitleSelect) &&
      recently_launched_titles_.size() > 0) {
    selected_title_index =
        std::clamp(selected_title_index, 0,
                   static_cast<int32_t>(recently_launched_titles_.size() - 1));

    // Must clear dialogs to prevent stacking
    ClearDialogs();

    // Titles may contain Unicode characters such as At World’s End
    // Must use ImGUI font that can render these Unicode characters
    std::string title_name;

    // Use filename if title name is empty
    if (recently_launched_titles_[selected_title_index].title_name.empty()) {
      title_name = recently_launched_titles_[selected_title_index]
                       .path_to_file.filename()
                       .string();
    } else {
      title_name = recently_launched_titles_[selected_title_index].title_name;
    }

    std::string title = fmt::format(
        "{}: {}\n\n{}", selected_title_index + 1, title_name,
        controller_hotkey_map.find(X_INPUT_GAMEPAD_START)->second.pretty);

    xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(), "Title Selection",
                                        title);
  }

  if (!notificationTitle.empty()) {
    app_context_.CallInUIThread(
        [imgui_drawer = imgui_drawer(), notificationTitle, notificationDesc]() {
          new xe::ui::HostNotificationWindow(imgui_drawer, notificationTitle,
                                             notificationDesc, 0);
        });
  }

  xe::threading::Sleep(delay);

  return it->second;
}

void EmulatorWindow::VibrateController(xe::hid::InputSystem* input_sys,
                                       uint32_t user_index,
                                       bool toggle_rumble) {
  constexpr std::chrono::milliseconds rumble_duration(100);

  // Hold lock while sleeping this thread for the duration of the rumble,
  // otherwise the rumble may fail.
  auto input_lock = input_sys->lock();

  X_INPUT_VIBRATION vibration = {};

  vibration.left_motor_speed = toggle_rumble ? UINT16_MAX : 0;
  vibration.right_motor_speed = toggle_rumble ? UINT16_MAX : 0;

  input_sys->SetState(user_index, &vibration);

  // Vibration duration
  if (toggle_rumble) {
    xe::threading::Sleep(rumble_duration);
  }
}

void EmulatorWindow::GamepadHotKeys() {
  X_INPUT_STATE state;

  constexpr std::chrono::milliseconds thread_delay(75);

  auto input_sys = emulator_->input_system();

  if (input_sys) {
    while (true) {
      // Collect controller states while holding the lock
      std::array<std::pair<bool, X_INPUT_STATE>, XUserMaxUserCount>
          controller_states;
      {
        auto input_lock = input_sys->lock();
        for (uint32_t user_index = 0; user_index < XUserMaxUserCount;
             ++user_index) {
          X_RESULT result = input_sys->GetStateSnapshot(
              user_index, X_INPUT_FLAG::X_INPUT_FLAG_GAMEPAD, &state);
          controller_states[user_index] = {result == X_ERROR_SUCCESS, state};
        }
      }  // Lock is released here when input_lock goes out of scope

      // Process hotkeys without holding the lock
      for (uint32_t user_index = 0; user_index < XUserMaxUserCount;
           ++user_index) {
        if (controller_states[user_index].first) {
          if (ProcessControllerHotkey(
                  controller_states[user_index].second.gamepad.buttons)
                  .rumble) {
            // Enable Vibration
            VibrateController(input_sys, user_index, true);

            // Disable Vibration
            VibrateController(input_sys, user_index, false);
          }
        }
      }

      xe::threading::Sleep(thread_delay);
    }
  }
}

void EmulatorWindow::ToggleGPUSetting(gpu::GPUSetting setting) {
  switch (setting) {
    case GPUSetting::ClearMemoryPageState:
      SaveGPUSetting(GPUSetting::ClearMemoryPageState,
                     !cvars::clear_memory_page_state);
      break;
    case GPUSetting::ReadbackMemexport:
      SaveGPUSetting(GPUSetting::ReadbackMemexport, !cvars::readback_memexport);
      break;
  }
}

void EmulatorWindow::CycleReadbackResolve() {
  const std::string& current = cvars::readback_resolve;
  if (current == "fast") {
    gpu::SetReadbackResolveMode("full");
  } else if (current == "full") {
    gpu::SetReadbackResolveMode("none");
  } else {
    gpu::SetReadbackResolveMode("fast");
  }
}

void EmulatorWindow::DisplayHotKeysConfig() {
  std::string msg = "";
  std::string msg_passthru = "";

  bool guide_enabled = !IsUseNexusForGameBarEnabled() && cvars::guide_button;

  for (auto const& [key, val] : controller_hotkey_map) {
    std::string pretty_text = val.pretty;

    if (!guide_enabled) {
      pretty_text = std::regex_replace(
          pretty_text, std::regex("Guide", std::regex_constants::icase),
          "Back");
    }

    if (emulator_->is_title_open() && !val.title_passthru) {
      pretty_text += " (Disabled)";
    }

    if (val.title_passthru && !cvars::controller_hotkeys) {
      pretty_text += " (Disabled)";
    }

    if (val.title_passthru) {
      msg += pretty_text + "\n";
    } else {
      msg_passthru += pretty_text + "\n";
    }
  }

  // Add Title
  msg.insert(0, "Gameplay Hotkeys\n");

  // Prepend non-passthru hotkeys
  msg_passthru += "\n";
  msg.insert(0, msg_passthru);
  msg += "\n";

  msg += "Readback Resolve: " + cvars::readback_resolve;
  msg += "\n";

  msg += "Clear Memory Page State: " +
         xe::string_util::BoolToString(cvars::clear_memory_page_state);
  msg += "\n";

  msg += "Controller Hotkeys: " +
         xe::string_util::BoolToString(cvars::controller_hotkeys);

  ClearDialogs();
  xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(), "Controller Hotkeys",
                                      msg);
}

std::string EmulatorWindow::CanonicalizeFileExtension(
    const std::filesystem::path& path) {
  return xe::utf8::lower_ascii(xe::path_to_utf8(path.extension()));
}

xe::X_STATUS EmulatorWindow::RunTitle(
    const std::filesystem::path& path_to_file) {
  // However a game is started (Home, File > Open, the recent menu, F9, a
  // dropped file), the dashboard steps aside for it.
  HidePreferences();
  std::error_code ec = {};
  bool titleExists = std::filesystem::exists(path_to_file, ec);

  if (path_to_file.empty() || !titleExists) {
    std::string log_msg =
        fmt::format("Failed to launch title path is {}.",
                    path_to_file.empty() ? "empty" : "invalid");

    if (!path_to_file.empty() && !titleExists) {
      log_msg.append(fmt::format("\nProvided Path: {}", path_to_file));
    }

    if (ec) {
      log_msg.append(fmt::format("\nExtended message info: {} ({:08X})",
                                 ec.message(), ec.value()));
    }

    XELOGE("{}", log_msg);

    ClearDialogs();

    xe::ui::ImGuiDialog::ShowMessageBox(imgui_drawer_.get(),
                                        "Title Launch Failed!", log_msg);

    return X_STATUS_NO_SUCH_FILE;
  }

  if (emulator_->is_title_open()) {
#if XE_PLATFORM_MAC
    // Switching titles in a process that already booted one corrupts guest
    // memory in-process (kernel_state_ persists across titles; Memory::Reset()
    // can't safely scrub it - see the add-emulation-lifecycle-controls design).
    // Re-exec into a pristine process with the new target instead. Flush config
    // + the host shader cache first so the relaunch boots warm.
    FlushBeforeRelaunch();
    unsetenv("XENIA_TITLE_RELAUNCH_HOPS");
    xe::ui::RelaunchWithTarget(
        xe::path_to_utf8(std::filesystem::absolute(path_to_file)));
    // RelaunchWithTarget only returns on failure.
    XELOGE("Relaunch failed; staying in the current title.");
    return X_STATUS_UNSUCCESSFUL;
#else
    // Terminate the current title and start a new title.
    // if (emulator_->TerminateTitle() == X_STATUS_SUCCESS) {
    //   return RunTitle(path);
    // }

    return X_STATUS_UNSUCCESSFUL;
#endif  // XE_PLATFORM_MAC
  }

  // Prevent crashing the emulator by not loading a game if a game is already
  // loaded.
  auto abs_path = std::filesystem::absolute(path_to_file);

  auto extension = CanonicalizeFileExtension(abs_path);

  if (extension == ".7z" || extension == ".zip" || extension == ".rar" ||
      extension == ".tar" || extension == ".gz") {
    xe::ShowSimpleMessageBox(
        xe::SimpleMessageBoxType::Error,
        fmt::format(
            "Unsupported format!\n"
            "Xenia does not support running software in an archived format."));

    return X_STATUS_UNSUCCESSFUL;
  }

  auto result = emulator_->LaunchPath(abs_path);

  disable_hotkeys_ = false;

  if (profile_config_dialog_) {
    profile_config_dialog_.reset();
    emulator_->kernel_state()->xam_state()->is_xam_dialog_present_.store(false);
  }

  ClearDialogs();

  if (result) {
    XELOGE("Failed to launch target: {:08X}", result);

    xe::ui::ImGuiDialog::ShowMessageBox(
        imgui_drawer_.get(), "Title Launch Failed!",
        "Failed to launch title.\n\nCheck xenia.log for technical details.");

    emulator_->file_system()->Clear();
  } else {
    // Remember the media so Start/Reboot can re-boot it after a Stop.
    last_media_path_ = abs_path;
    // Title is now running and media is remembered - refresh menu enable state
    // (on_launch fires before this point, before last_media_path_ is set).
    UpdateLifecycleMenuState();
    AddRecentlyLaunchedTitle(path_to_file, emulator_->title_name());

    auto xam =
        emulator_->kernel_state()->GetKernelModule<kernel::xam::XamModule>(
            "xam.xex");

    xam->loader_data().host_path = xe::path_to_utf8(abs_path);
  }

  return result;
}

void EmulatorWindow::RunPreviouslyPlayedTitle() {
  if (recently_launched_titles_.size() >= 1) {
    RunTitle(recently_launched_titles_[0].path_to_file);
  }
}

void EmulatorWindow::FillRecentlyLaunchedTitlesMenu(
    xe::ui::MenuItem* recent_menu) {
  const size_t count =
      std::min(recently_launched_titles_.size(),
               size_t(std::max(cvars::recent_titles_entry_amount, 0)));
  for (size_t i = 0; i < count; ++i) {
    std::string hotkey = (i == 0) ? "F9" : "";

    const RecentTitleEntry& entry = recently_launched_titles_[i];
    const std::string item_text = entry.title_name.empty()
                                      ? entry.path_to_file.string()
                                      : entry.title_name;

    recent_menu->AddChild(MenuItem::Create(
        MenuItem::Type::kString, item_text, hotkey,
        std::bind(&EmulatorWindow::RunTitle, this, entry.path_to_file)));
  }
}

void EmulatorWindow::LoadRecentlyLaunchedTitles() {
  std::ifstream file(emulator()->storage_root() /
                     kRecentlyPlayedTitlesFilename);
  if (!file.is_open()) {
    return;
  }

  toml::parse_result parsed_file;
  try {
    parsed_file = toml::parse(file);
  } catch (toml::parse_error& exception) {
    XELOGE("Cannot parse file: recent.toml. Error: {}", exception.what());
    return;
  }

  if (parsed_file.is_table()) {
    for (const auto& [index, entry] : *parsed_file.as_table()) {
      if (!entry.is_table()) {
        continue;
      }

      const toml::table* entry_table = entry.as_table();

      std::string title_name =
          entry_table->get_as<std::string>("title_name")->get();
      std::string path = entry_table->get_as<std::string>("path")->get();
      std::time_t last_run_time =
          entry_table->get_as<int64_t>("last_run_time")->get();

      std::error_code ec = {};
      if (path.empty() || !std::filesystem::exists(path, ec)) {
        continue;
      }

      uint32_t title_id = 0;
      if (auto id = entry_table->get_as<int64_t>("title_id")) {
        title_id = uint32_t(id->get());
      }
      recently_launched_titles_.push_back(
          {title_name, path, last_run_time, title_id});
    }
  }
}

void EmulatorWindow::AddRecentlyLaunchedTitle(
    std::filesystem::path path_to_file, std::string title_name) {
  if (cvars::recent_titles_entry_amount <= 0) {
    return;
  }

  // Check if game is already on list and pop it to front
  auto entry_index = std::find_if(recently_launched_titles_.cbegin(),
                                  recently_launched_titles_.cend(),
                                  [&title_name](const RecentTitleEntry& entry) {
                                    return entry.title_name == title_name;
                                  });
  if (entry_index != recently_launched_titles_.cend()) {
    recently_launched_titles_.erase(entry_index);
  }

  const uint32_t title_id = emulator_->title_id();
  recently_launched_titles_.insert(
      recently_launched_titles_.cbegin(),
      {title_name, path_to_file, time(nullptr), title_id});
  // Keep the game's icon for the dashboard.
  if (title_id && emulator_->game_info_database()) {
    const std::vector<uint8_t> icon =
        emulator_->game_info_database()->GetIcon();
    SaveTitleIcon(title_id, icon);
  }
  // Serialize to toml
  auto toml_table = toml::table();

  size_t index = 0;
  for (const RecentTitleEntry& entry : recently_launched_titles_) {
    auto entry_table = toml::table();

    // Fill entry under specific index.
    std::string str_path = xe::path_to_utf8(entry.path_to_file);
    entry_table.insert("title_name", entry.title_name);
    entry_table.insert("path", str_path);
    entry_table.insert("last_run_time", entry.last_run_time);
    if (entry.title_id) {
      entry_table.insert("title_id", int64_t(entry.title_id));
    }

    toml_table.insert(std::to_string(index++), entry_table);

    if (index >= kMaxPlayedTitles) {
      break;
    }
  }
  // Open and write serialized data.
  std::ofstream file(emulator()->storage_root() / kRecentlyPlayedTitlesFilename,
                     std::ofstream::trunc);
  file << toml_table;
  file.close();
}

GameLibrary& EmulatorWindow::game_library() {
  if (!game_library_) {
    game_library_ = std::make_unique<GameLibrary>(
        emulator_->storage_root(), [this]() {
          // A scan finished: the dashboard shows what it found.
          app_context().CallInUIThread([this]() {
#if XE_PLATFORM_MAC
            if (blades_overlay_ && blades_model_) {
              blades_overlay_->PushState(blades_model_->BuildStateJson());
            }
#endif
          });
        });
    game_library_->Rescan();
  }
  return *game_library_;
}

void EmulatorWindow::AddGameFolders() {
  auto file_picker = xe::ui::FilePicker::Create();
  file_picker->set_mode(ui::FilePicker::Mode::kOpen);
  file_picker->set_type(ui::FilePicker::Type::kDirectory);
  file_picker->set_multi_selection(true);
  file_picker->set_title("Add Game Folders");
  if (file_picker->Show(window_.get())) {
    for (const auto& folder : file_picker->selected_files()) {
      game_library().AddFolder(folder);
    }
  }
}

const char* EmulatorWindow::GamerPicturesStatus() {
#if XE_PLATFORM_MAC
  if (gamer_pictures_downloading_) {
    auto progress = std::static_pointer_cast<gamer_pictures::Progress>(
        gamer_pictures_progress_);
    return progress && progress->installing ? "installing" : "downloading";
  }
  if (!gamer_pictures_checked_) {
    gamer_pictures_checked_ = true;
    if (auto summary = gamer_pictures::Installed(emulator_->storage_root(),
                                                 emulator_->content_root())) {
      gamer_pictures_count_ = summary->pictures;
      gamer_pictures_titles_ = summary->titles;
      gamer_pictures_bytes_ = summary->bytes;
    }
  }
  if (gamer_pictures_count_) {
    return "ready";
  }
  return gamer_pictures_error_.empty() ? "missing" : "failed";
#else
  return "unavailable";
#endif
}

void EmulatorWindow::DownloadGamerPictures() {
#if XE_PLATFORM_MAC
  if (gamer_pictures_downloading_) {
    return;
  }
  gamer_pictures_downloading_ = true;
  gamer_pictures_error_.clear();
  auto progress = std::make_shared<gamer_pictures::Progress>();
  gamer_pictures_progress_ = progress;
  gamer_pictures::Download(
      emulator_->storage_root(), emulator_->content_root(), progress,
      [this](std::string error, gamer_pictures::Summary summary) {
        app_context().CallInUIThread([this, error, summary]() {
          gamer_pictures_downloading_ = false;
          gamer_pictures_error_ = error;
          if (error.empty()) {
            gamer_pictures_count_ = summary.pictures;
            gamer_pictures_titles_ = summary.titles;
            gamer_pictures_bytes_ = summary.bytes;
          }
          if (blades_overlay_ && blades_model_) {
            blades_overlay_->PushState(blades_model_->BuildStateJson());
          }
        });
      });
#endif
}

void EmulatorWindow::RevealGamerPictures() {
#if XE_PLATFORM_MAC
  LaunchFileExplorer(gamer_pictures::Folder(emulator_->content_root()));
#endif
}

void EmulatorWindow::DeleteGamerPictures() {
#if XE_PLATFORM_MAC
  if (gamer_pictures_downloading_) return;
  gamer_pictures_error_ = gamer_pictures::Remove(emulator_->storage_root(),
                                                 emulator_->content_root());
  if (gamer_pictures_error_.empty()) {
    gamer_pictures_count_ = 0;
    gamer_pictures_titles_ = 0;
    gamer_pictures_bytes_ = 0;
  }
#endif
}

std::filesystem::path EmulatorWindow::DashboardSoundsPath() const {
  return emulator_->storage_root() / "dashboard_sounds";
}

const char* EmulatorWindow::DashboardSoundsStatus() const {
#if XE_PLATFORM_MAC
  if (dashboard_sounds_downloading_) {
    return "downloading";
  }
  if (dashboard_sounds::Present(DashboardSoundsPath())) {
    return "ready";
  }
  return dashboard_sounds_error_.empty() ? "missing" : "failed";
#else
  return "unavailable";
#endif
}

void EmulatorWindow::DownloadDashboardSounds() {
#if XE_PLATFORM_MAC
  if (dashboard_sounds_downloading_) {
    return;
  }
  dashboard_sounds_downloading_ = true;
  dashboard_sounds_error_.clear();
  dashboard_sounds::Download(
      DashboardSoundsPath(), [this](std::string error) {
        app_context().CallInUIThread([this, error]() {
          dashboard_sounds_downloading_ = false;
          dashboard_sounds_error_ = error;
          if (blades_overlay_ && blades_model_) {
            blades_overlay_->PushState(blades_model_->BuildStateJson());
          }
        });
      });
#endif
}

std::filesystem::path EmulatorWindow::TitleIconPath(uint32_t title_id) const {
  return emulator_->storage_root() / "title_icons" /
         fmt::format("{:08X}.png", title_id);
}

void EmulatorWindow::SaveTitleIcon(uint32_t title_id,
                                   std::span<const uint8_t> png) const {
  // Only a PNG (the console's title icons are), and only once.
  static constexpr uint8_t kPngSignature[] = {0x89, 'P', 'N', 'G'};
  if (!title_id || png.size() < sizeof(kPngSignature) ||
      std::memcmp(png.data(), kPngSignature, sizeof(kPngSignature)) != 0) {
    return;
  }
  const std::filesystem::path path = TitleIconPath(title_id);
  std::error_code ec;
  if (std::filesystem::exists(path, ec)) {
    return;
  }
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream file(path, std::ofstream::binary | std::ofstream::trunc);
  file.write(reinterpret_cast<const char*>(png.data()),
             std::streamsize(png.size()));
}

void EmulatorWindow::ClearDialogs() {
  if (profile_config_dialog_) {
    profile_config_dialog_.reset();
  }

  imgui_drawer_.get()->ClearDialogs();
  emulator_->kernel_state()->xam_state()->is_xam_dialog_present_.store(false);
  // The frame rate counter is a dialog too; put it back if it is on.
  fps_overlay_.reset();
  UpdateFpsOverlay();
}

}  // namespace app
}  // namespace xe
