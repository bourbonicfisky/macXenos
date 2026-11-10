/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APP_EMULATOR_WINDOW_H_
#define XENIA_APP_EMULATOR_WINDOW_H_

#include <memory>
#include <string>

#include "xenia/app/profile_dialogs.h"
#include "xenia/emulator.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/ui/imgui_dialog.h"
#include "xenia/ui/imgui_drawer.h"
#include "xenia/ui/immediate_drawer.h"
#include "xenia/ui/menu_item.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/window.h"
#include "xenia/ui/window_listener.h"
#include "xenia/ui/windowed_app_context.h"
#include "xenia/xbox.h"

namespace xe {
namespace app {

class BladesModel;
class BladesOverlay;
class GameLibrary;

struct RecentTitleEntry {
  std::string title_name;
  std::filesystem::path path_to_file;
  std::time_t last_run_time;
  // 0 for entries recorded before title IDs were kept.
  uint32_t title_id = 0;
};

class EmulatorWindow {
  // The Preferences blades read and change emulator state through the same
  // private members the dialogs use.
  friend class BladesModel;

 public:
  using steady_clock = std::chrono::steady_clock;  // stdlib steady clock

  enum : size_t {
    // The UI is on top of the game and is open in special cases, so
    // lowest-priority.
    kZOrderHidInput,
    kZOrderImGui,
    kZOrderProfiler,
    // Emulator window controls are expected to be always accessible by the
    // user, so highest-priority.
    kZOrderEmulatorWindowInput,
  };

  virtual ~EmulatorWindow();

  static std::unique_ptr<EmulatorWindow> Create(
      Emulator* emulator, ui::WindowedAppContext& app_context, uint32_t width,
      uint32_t height);

  std::unique_ptr<xe::threading::Thread> Gamepad_HotKeys_Listener;

  int32_t selected_title_index = -1;

  static constexpr int64_t diff_in_ms(
      const steady_clock::time_point t1,
      const steady_clock::time_point t2) noexcept {
    using ms = std::chrono::milliseconds;
    return std::chrono::duration_cast<ms>(t1 - t2).count();
  }

  steady_clock::time_point last_mouse_up = steady_clock::now();
  steady_clock::time_point last_mouse_down = steady_clock::now();

  Emulator* emulator() const { return emulator_; }
  ui::WindowedAppContext& app_context() const { return app_context_; }
  ui::Window* window() const { return window_.get(); }
  ui::ImGuiDrawer* imgui_drawer() const { return imgui_drawer_.get(); }

  ui::Presenter* GetGraphicsSystemPresenter() const;
  void SetupGraphicsSystemPresenterPainting();
  void ShutdownGraphicsSystemPresenterPainting();

  void OnEmulatorInitialized();

  xe::X_STATUS RunTitle(const std::filesystem::path& path_to_file);
  // With no game on the command line: Preferences on the Home blade (or
  // Profiles, when there is no profile yet), as the dashboard was.
  void ShowStartScreen();
  // Where a game's icon is kept once seen (<storage>/title_icons/ID.png),
  // for the dashboard's Home blade.
  std::filesystem::path TitleIconPath(uint32_t title_id) const;
  // The downloaded dashboard sounds (dashboard_sounds_mac.h).
  std::filesystem::path DashboardSoundsPath() const;
  // "missing", "downloading", "ready" or "failed" ("unavailable" off macOS).
  const char* DashboardSoundsStatus() const;
  // Starts the download in the background; the page hears when it ends.
  void DownloadDashboardSounds();
  // The gamer-picture collection (gamer_pictures_mac.h): "missing",
  // "downloading", "installing", "ready" or "failed" ("unavailable" off
  // macOS), and starting its one-time download.
  const char* GamerPicturesStatus();
  void DownloadGamerPictures();
  void RevealGamerPictures();
  void DeleteGamerPictures();
  // The user's game folders and what is in them, created (and first scanned)
  // on first use.
  GameLibrary& game_library();
  // Asks for folders to add to the game folders.
  void AddGameFolders();
  void SaveTitleIcon(uint32_t title_id, std::span<const uint8_t> png) const;
  // Creates or removes the frame rate counter to match show_fps.
  void UpdateFpsOverlay();
  // Grey the lifecycle menu items per current state (Stop when running, Reboot/
  // Full Reboot when media has been loaded). Must be called on the UI thread.
  void UpdateLifecycleMenuState();
  void UpdateTitle();
  void SetFullscreen(bool fullscreen);
  void ToggleFullscreen();
  void SetInitializingShaderStorage(bool initializing);

  void TakeScreenshot();
  void ExportScreenshot(const xe::ui::RawImage& image, bool notify = true,
                        const std::string& suffix = {});
  void AutoScreenshotLoop();
  void SaveImage(const std::filesystem::path& path,
                 const xe::ui::RawImage& image);

  void ToggleProfilesConfigDialog();

  // A controller hotkey for display/toggling in the controller UI.
  struct ControllerHotkeyInfo {
    int function;  // ButtonFunctions value
    std::string description;
    bool enabled;
    // Works while a game runs (and needs controller_hotkeys); otherwise it is
    // a home-screen hotkey.
    bool in_game;
  };
  // Controller hotkeys (deduplicated by function) for the controller UI.
  std::vector<ControllerHotkeyInfo> GetControllerHotkeys() const;
  // Enables/disables a hotkey by ButtonFunctions value and persists the choice.
  void SetControllerHotkeyEnabled(int function, bool enabled);

  void SetHotkeysState(bool enabled) { disable_hotkeys_ = !enabled; }

  // Types of button functions for hotkeys.
  enum class ButtonFunctions {
    ToggleFullscreen,
    RunTitle,
    CpuTimeScalarSetHalf,
    CpuTimeScalarSetDouble,
    CpuTimeScalarReset,
    ClearGPUCache,
    ToggleControllerVibration,
    ClearMemoryPageState,
    ReadbackResolve,
    ToggleLogging,
    IncTitleSelect,
    DecTitleSelect,
    Unknown
  };

  class ControllerHotKey {
   public:
    // If true the hotkey can be activated while a title is running, otherwise
    // false.
    bool title_passthru;

    // If true vibrate the controller after activating the hotkey, otherwise
    // false.
    bool rumble;
    std::string pretty;
    ButtonFunctions function;

    ControllerHotKey(ButtonFunctions fn = ButtonFunctions::Unknown,
                     std::string pretty = "", bool rumble = false,
                     bool active = true) {
      function = fn;
      this->pretty = pretty;
      title_passthru = active;
      this->rumble = rumble;
    }
  };

 private:
  class EmulatorWindowListener final : public ui::WindowListener,
                                       public ui::WindowInputListener {
   public:
    explicit EmulatorWindowListener(EmulatorWindow& emulator_window)
        : emulator_window_(emulator_window) {}

    void OnClosing(ui::UIEvent& e) override;
    void OnFileDrop(ui::FileDropEvent& e) override;

    void OnKeyDown(ui::KeyEvent& e) override;

    void OnMouseDown(ui::MouseEvent& e) override;
    void OnMouseUp(ui::MouseEvent& e) override;

    void OnUsbDeviceChanged(bool is_arrival) override;

   private:
    EmulatorWindow& emulator_window_;
  };

  class DisplayConfigGameConfigLoadCallback
      : public Emulator::GameConfigLoadCallback {
   public:
    DisplayConfigGameConfigLoadCallback(Emulator& emulator,
                                        EmulatorWindow& emulator_window)
        : Emulator::GameConfigLoadCallback(emulator),
          emulator_window_(emulator_window) {}

    void PostGameConfigLoad() override;

   private:
    EmulatorWindow& emulator_window_;
  };

  class ContentInstallDialog final : public ui::ImGuiDialog {
   public:
    ContentInstallDialog(
        ui::ImGuiDrawer* imgui_drawer, EmulatorWindow& emulator_window,
        std::shared_ptr<std::vector<Emulator::ContentInstallEntry>> entries)
        : ui::ImGuiDialog(imgui_drawer),
          emulator_window_(emulator_window),
          installation_entries_(entries) {
      window_id_ = GetWindowId();
    }

    ~ContentInstallDialog() {
      for (auto& entry : *installation_entries_) {
        entry.icon_.release();
      }
    }

   protected:
    void OnDraw(ImGuiIO& io) override;

   private:
    uint64_t window_id_;

    EmulatorWindow& emulator_window_;
    std::shared_ptr<std::vector<Emulator::ContentInstallEntry>>
        installation_entries_;
  };

  // The frame rate in a corner of the window while a game runs (show_fps).
  // Exists only while the setting is on.
  class FpsOverlay final : public ui::ImGuiDialog {
   public:
    FpsOverlay(ui::ImGuiDrawer* imgui_drawer, EmulatorWindow& emulator_window)
        : ui::ImGuiDialog(imgui_drawer), emulator_window_(emulator_window) {}

   protected:
    void OnDraw(ImGuiIO& io) override;

   private:
    EmulatorWindow& emulator_window_;
    uint64_t last_count_ = 0;
    double last_time_ = 0.0;
    double fps_ = 0.0;
  };

  explicit EmulatorWindow(Emulator* emulator,
                          ui::WindowedAppContext& app_context, uint32_t width,
                          uint32_t height);

  bool Initialize();

  // For comparisons, use GetGuestOutputPaintEffectForCvarValue instead as the
  // default fallback may be used for multiple values.
  static const char* GetCvarValueForGuestOutputPaintEffect(
      ui::Presenter::GuestOutputPaintConfig::Effect effect);
  static ui::Presenter::GuestOutputPaintConfig::Effect
  GetGuestOutputPaintEffectForCvarValue(const std::string& cvar_value);
  static ui::Presenter::GuestOutputPaintConfig
  GetGuestOutputPaintConfigForCvars();
  void ApplyDisplayConfigForCvars();
  // Renderers as bits, for settings only one of them reads.
  enum RendererMask : uint32_t {
    kRendererMetal = 1u << 0,
    kRendererVulkan = 1u << 1,
  };
  // The running renderer as a RendererMask bit, or 0 when no graphics system
  // is up.
  uint32_t ActiveRendererMask() const;

  void OnKeyDown(ui::KeyEvent& e);
  void OnMouseDown(const ui::MouseEvent& e);
  void ToggleFullscreenOnDoubleClick();
  void FileDrop(const std::filesystem::path& filename);
  void OnMouseUp(const ui::MouseEvent& e);
  // Both return false when the picker was cancelled.
  bool FileOpen();
  void FileClose();
  // Persist config + host shader cache before a re-exec relaunch (macOS
  // lifecycle controls), mirroring the OnDestroy pre-quick_exit flush.
  void FlushBeforeRelaunch();
  // Re-boot the last-loaded media in a pristine process (re-exec). Used by the
  // Reboot/Start lifecycle controls; no-op if no media has been loaded. When
  // warm, finalizes the shader/pipeline storage first so the relaunch boots
  // warm (Reboot); when not warm, just persists config and relaunches
  // immediately (Full Reboot) - bulletproof but boots cold.
  void RelaunchLastTitle(bool warm);
  bool InstallContent();
  void ExtractZarchive();
  void CreateZarchive();
  void ShowContentDirectory();
  void CpuTimeScalarReset();
  void CpuTimeScalarSetHalf();
  void CpuTimeScalarSetDouble();
  void CpuBreakIntoDebugger();
  void CpuBreakIntoHostDebugger();
  void GpuTraceFrame();
  void GpuClearCaches();
  // The Preferences blades (blades_overlay_mac.h). `blade` picks the blade to
  // open on ("graphics", "controller", "audio", "cpu", "system", "dashboard",
  // "advanced"); empty keeps the last one.
  void ShowPreferences(const std::string& blade);
  void HidePreferences();
  void TogglePreferences(const std::string& blade);
  bool IsPreferencesVisible() const;
  void HandlePreferencesMessage(const std::string& json);
  void PumpPreferencesController();
  // Suspends the running game's threads and audio while the dashboard is
  // open (dashboard_freezes_game), and resumes them.
  void FreezeGame(bool freeze);
  void PreferencesButtonListener();
  void ToggleControllerVibration();
  void ShowWebsite();

  EmulatorWindow::ControllerHotKey ProcessControllerHotkey(int buttons);
  void VibrateController(xe::hid::InputSystem* input_sys, uint32_t user_index,
                         bool vibrate = true);
  void GamepadHotKeys();
  void ToggleGPUSetting(gpu::GPUSetting setting);
  void CycleReadbackResolve();
  void DisplayHotKeysConfig();

  static std::string CanonicalizeFileExtension(
      const std::filesystem::path& path);

  void RunPreviouslyPlayedTitle();
  void FillRecentlyLaunchedTitlesMenu(xe::ui::MenuItem* recent_menu);
  void LoadRecentlyLaunchedTitles();
  void AddRecentlyLaunchedTitle(std::filesystem::path path_to_file,
                                std::string title_name);

  void ClearDialogs();

  Emulator* emulator_;
  // Absolute path of the last media that successfully launched; persists across
  // a Stop so Start/Reboot can re-boot it. Empty until something is loaded.
  std::filesystem::path last_media_path_;
  // Non-owning pointers to the lifecycle menu items (owned by the menu tree),
  // kept so UpdateLifecycleMenuState can grey them per emulation state.
  ui::MenuItem* stop_menu_item_ = nullptr;
  ui::MenuItem* reboot_menu_item_ = nullptr;
  ui::MenuItem* full_reboot_menu_item_ = nullptr;
  ui::WindowedAppContext& app_context_;
  EmulatorWindowListener window_listener_;
  std::unique_ptr<ui::Window> window_;
  std::unique_ptr<ui::ImGuiDrawer> imgui_drawer_;
  std::unique_ptr<DisplayConfigGameConfigLoadCallback>
      display_config_game_config_load_callback_;
  // Creation may fail, in this case immediate drawer UI must not be drawn.
  std::unique_ptr<ui::ImmediateDrawer> immediate_drawer_;

  bool emulator_initialized_ = false;
  std::atomic<bool> disable_hotkeys_ = false;

  std::string base_title_;
  bool initializing_shader_storage_ = false;


  // Storing pointers and toggling dialog state is useful for broadcasting
  // messages back to guest.
  std::unique_ptr<ProfileConfigDialog> profile_config_dialog_;

  std::unique_ptr<FpsOverlay> fps_overlay_;
  // The controller button that opens Preferences (preferences_button), polled
  // on its own thread so it works whether or not hotkeys are on.
  std::unique_ptr<xe::threading::Thread> preferences_button_listener_;
  std::atomic<bool> preferences_button_stop_{false};
  // screenshot_interval(_ms) and log_fps_interval.
  std::unique_ptr<xe::threading::Thread> auto_screenshot_thread_;
  std::atomic<bool> auto_screenshot_stop_{false};
  std::atomic<bool> blades_visible_{false};
  bool game_frozen_ = false;
  bool dashboard_sounds_downloading_ = false;
  // The gamer-picture download, while one runs, and what the last left.
  std::shared_ptr<void> gamer_pictures_progress_;
  bool gamer_pictures_downloading_ = false;
  std::string gamer_pictures_error_;
  uint32_t gamer_pictures_count_ = 0;
  uint32_t gamer_pictures_titles_ = 0;
  uint64_t gamer_pictures_bytes_ = 0;
  bool gamer_pictures_checked_ = false;
  std::unique_ptr<GameLibrary> game_library_;
  // Why the last download failed; empty when it hasn't.
  std::string dashboard_sounds_error_;
  bool mute_before_freeze_ = false;

  // Declared after window_, so they go before the window they draw in.
  std::unique_ptr<BladesModel> blades_model_;
  std::unique_ptr<BladesOverlay> blades_overlay_;
  // Whether opening Preferences raised the console's system-UI state (which
  // pauses most games and hides input from them), so closing lowers it.
  bool blades_raised_system_ui_ = false;
  // Controller buttons last forwarded to the page.
  uint32_t blades_pad_buttons_ = 0;

  std::vector<RecentTitleEntry> recently_launched_titles_;
};

}  // namespace app
}  // namespace xe

#endif  // XENIA_APP_EMULATOR_WINDOW_H_
