/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/apu_flags.h"

DEFINE_bool(mute, false, "Mutes all audio output.", "APU")
DEFINE_bool(media_diag, false,
            "Log audio/XMA media-engine diagnostics (final-mix peak, XMA "
            "context drive/decode) for x64-vs-arm64 comparison.",
            "APU")

namespace xe {
namespace apu {
MediaDiagStats g_media_diag_stats;
}  // namespace apu
}  // namespace xe
