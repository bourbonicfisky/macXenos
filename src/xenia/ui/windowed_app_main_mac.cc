/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/ui/windowed_app.h"
#include "xenia/ui/windowed_app_context_mac.h"

extern "C" int main(int argc, char** argv) {
  int result;

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
