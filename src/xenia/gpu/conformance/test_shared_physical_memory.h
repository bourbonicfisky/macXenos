/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_CONFORMANCE_TEST_SHARED_PHYSICAL_MEMORY_H_
#define XENIA_GPU_CONFORMANCE_TEST_SHARED_PHYSICAL_MEMORY_H_

#include <memory>

#include "xenia/memory.h"

namespace xe {
namespace gpu {
namespace conformance {

// One process-wide Memory for every shared-memory test harness: Memory maps
// named file views and installs access-violation handling, so two instances
// in one process conflict. Returns nullptr if initialization fails.
inline Memory* TestPhysicalMemory() {
  static std::unique_ptr<Memory> memory = [] {
    auto m = std::make_unique<Memory>();
    if (!m->Initialize()) {
      m.reset();
    }
    return m;
  }();
  return memory.get();
}

}  // namespace conformance
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_CONFORMANCE_TEST_SHARED_PHYSICAL_MEMORY_H_
