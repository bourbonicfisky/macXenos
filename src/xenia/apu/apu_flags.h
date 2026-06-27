/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_APU_APU_FLAGS_H_
#define XENIA_APU_APU_FLAGS_H_

#include "xenia/base/cvar.h"
DECLARE_bool(mute)
DECLARE_bool(media_diag)

#include <atomic>
#include <cstdint>

namespace xe {
namespace apu {

// Shared, lock-free counters for the --media_diag pipeline heartbeat. Each
// audio stage bumps its counter; the SDL driver flushes per-window deltas so a
// stall is localized to the first stage whose nonzero-rate drops to zero.
// All counters are cumulative; deltas are computed at flush time.
struct MediaDiagStats {
  std::atomic<uint64_t> work_enabled{0};    // XMA contexts seen enabled in Work
  std::atomic<uint64_t> decode_attempt{0};  // DecodePacket() called
  std::atomic<uint64_t> decode_ok{0};       // DecodePacket() returned a frame
  std::atomic<uint64_t> decode_nonzero{0};  // decoded frame had nonzero PCM
  std::atomic<uint64_t> submit_frames{0};   // final-mix frames submitted to SDL
  std::atomic<uint64_t> submit_nonzero{0};  // submitted frame had nonzero peak
};
extern MediaDiagStats g_media_diag_stats;

}  // namespace apu
}  // namespace xe

#endif  // XENIA_APU_APU_FLAGS_H_
