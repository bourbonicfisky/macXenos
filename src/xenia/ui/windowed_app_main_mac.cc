/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <crt_externs.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <unistd.h>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/ui/process_relaunch_mac.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context_mac.h"

namespace {
// Captured at process entry so a lifecycle relaunch can rebuild the command
// line. argv lives for the life of the process, so storing the pointers is
// safe.
int g_saved_argc = 0;
char** g_saved_argv = nullptr;
}  // namespace

namespace xe {
namespace ui {

bool RelaunchWithTarget(const std::string& target_path,
                        const std::vector<std::string>& extra_args) {
  if (!g_saved_argv || g_saved_argc < 1) {
    XELOGE("RelaunchWithTarget: no saved argv; cannot relaunch");
    return false;
  }

  // Rebuild argv: keep argv[0] and all original flags, drop any prior
  // positional path or --target (we supply a fresh one) and the per-launch
  // loader-data flags (a title-to-title chain must not inherit the previous
  // hop's launch data), then append the new --target and the extra args last
  // so they win.
  static const char* const kPerLaunchFlags[] = {
      "--target", "--relaunch_module", "--launch_flags", "--launch_data",
      "--log_append",
  };
  std::vector<char*> new_argv;
  new_argv.push_back(g_saved_argv[0]);
  for (int i = 1; i < g_saved_argc; ++i) {
    char* a = g_saved_argv[i];
    if (!a || a[0] != '-') {
      continue;  // positional (the old target) - drop
    }
    bool per_launch = false;
    for (const char* flag : kPerLaunchFlags) {
      size_t n = std::strlen(flag);
      if (std::strncmp(a, flag, n) == 0 && (a[n] == '\0' || a[n] == '=')) {
        per_launch = true;
        break;
      }
    }
    if (per_launch) {
      continue;
    }
    new_argv.push_back(a);
  }
  std::string target_arg = "--target=" + target_path;
  new_argv.push_back(const_cast<char*>(target_arg.c_str()));
  for (const std::string& extra : extra_args) {
    new_argv.push_back(const_cast<char*>(extra.c_str()));
  }
  new_argv.push_back(nullptr);

  // Give the relaunch a fresh ASLR-retry budget (we may be a relaunch already).
  unsetenv("XENIA_ASLR_RETRY");

  // Resolve the real executable path: posix_spawn does not search PATH, and our
  // argv[0] may be relative to the original working directory.
  char exe_path[PATH_MAX];
  uint32_t exe_path_size = sizeof(exe_path);
  const char* spawn_path = new_argv[0];
  if (_NSGetExecutablePath(exe_path, &exe_path_size) == 0) {
    spawn_path = exe_path;
  }

  XELOGI("Relaunching for new target: {}", target_path);
  xe::FlushLog();

  // Spawn a brand-new process rather than execv-ing in place. An in-place execv
  // keeps the same PID and the kernel-side WindowServer/NSApplication state;
  // on a re-exec-of-a-re-exec that stale registration leaves the new Cocoa app
  // unable to take foreground - the window launches but stays unfocused and
  // unrendered until the user clicks it. A fresh PID activates cleanly.
  // Only stdin/stdout/stderr cross over: by default the child inherits every
  // descriptor, and the guest's bound sockets then stay open in it (unused),
  // so its own bind of the same port fails with EADDRINUSE. Modern Warfare 2
  // relaunches itself at boot and looped until the hop cap on "Failed to
  // bind socket".
  posix_spawnattr_t spawn_attr;
  posix_spawnattr_init(&spawn_attr);
  posix_spawnattr_setflags(&spawn_attr, POSIX_SPAWN_CLOEXEC_DEFAULT);
  posix_spawn_file_actions_t spawn_actions;
  posix_spawn_file_actions_init(&spawn_actions);
  for (int fd = 0; fd <= 2; ++fd) {
    posix_spawn_file_actions_addinherit_np(&spawn_actions, fd);
  }
  pid_t child_pid = 0;
  int spawn_result =
      posix_spawn(&child_pid, spawn_path, &spawn_actions, &spawn_attr,
                  new_argv.data(), *_NSGetEnviron());
  posix_spawn_file_actions_destroy(&spawn_actions);
  posix_spawnattr_destroy(&spawn_attr);
  if (spawn_result != 0) {
    XELOGE("RelaunchWithTarget: posix_spawn failed: {}",
           std::strerror(spawn_result));
    return false;
  }

  // Hand off: quit this process immediately so only the fresh instance remains.
  // Config was already persisted by the caller; nothing else needs teardown.
  xe::FlushLog();
  _exit(EXIT_SUCCESS);
}

}  // namespace ui
}  // namespace xe

extern "C" int main(int argc, char** argv) {
  int result;

  g_saved_argc = argc;
  g_saved_argv = argv;

  // Reserve the fixed sub-4 GB JIT ranges before anything (NSApplication, dyld
  // runtime dylib loads such as MoltenVK, malloc zones) can land in the low
  // address space that the shrunk __PAGEZERO opens up - otherwise the code cache
  // intermittently collides with an ASLR-placed mapping at startup. On the rare
  // launch where an image already squats in the range, re-exec the process for a
  // fresh ASLR layout (capped, so a persistent occupant can't loop forever).
  if (!xe::memory::ReserveFixedRangesEarly()) {
    constexpr int kMaxAslrRetries = 8;
    const char* attempt_str = getenv("XENIA_ASLR_RETRY");
    int attempt = attempt_str ? atoi(attempt_str) : 0;
    if (attempt < kMaxAslrRetries) {
      char next[16];
      std::snprintf(next, sizeof(next), "%d", attempt + 1);
      setenv("XENIA_ASLR_RETRY", next, 1);
      execv(argv[0], argv);
      // If execv returns it failed; fall through and try to run anyway.
    }
    // Out of retries (or execv failed): proceed; init will fail loudly and the
    // detection fixes keep it from corrupting rather than just crashing.
  }

  {
    xe::ui::MacWindowedAppContext app_context;

    std::unique_ptr<xe::ui::WindowedApp> app =
        xe::ui::GetWindowedAppCreator()(app_context);

    cvar::ParseLaunchArguments(argc, argv, app->GetPositionalOptionsUsage(),
                               app->GetPositionalOptions());

    // Initialize logging. Needs parsed cvars.
    xe::InitializeLogging(app->GetName());

    if (app->OnInitialize()) {
      app_context.RunMainMacLoop();
      result = EXIT_SUCCESS;
    } else {
      result = EXIT_FAILURE;
    }

    app->InvokeOnDestroy();
  }

  // Logging may still be needed in the destructors.
  xe::ShutdownLogging();

  return result;
}
