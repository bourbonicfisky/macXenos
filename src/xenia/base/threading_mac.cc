/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/threading.h"

#include <mach/mach_time.h>

namespace xe {
namespace threading {

// Only ticks() lives here. The other thread primitives are provided by the
// shared threading.cc (current_thread_id) and threading_posix.cc, which carries
// in-file XE_PLATFORM_MAC branches for the spots where macOS diverges from Linux
// (realtime signals, SYS_gettid, pthread_setname_np arity, affinity, sigqueue).
uint64_t ticks() { return mach_absolute_time(); }

}  // namespace threading
}  // namespace xe
