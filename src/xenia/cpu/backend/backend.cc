/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/backend.h"

#include <cstring>

#include "xenia/base/byte_order.h"
#include "xenia/base/platform.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/ppc/ppc_context.h"

namespace xe {
namespace cpu {
namespace backend {

Backend::Backend() { std::memset(&machine_info_, 0, sizeof(machine_info_)); }
Backend::~Backend() = default;

bool Backend::Initialize(Processor* processor) {
  processor_ = processor;
  return true;
}

void* Backend::AllocThreadData() { return nullptr; }

void Backend::FreeThreadData(void* thread_data) {}

namespace {
// Hint to the core that it is spinning on a contended location.
inline void SpinPause() {
#if XE_ARCH_AMD64
  _mm_pause();
#elif XE_ARCH_ARM64
  // yield retires as a nop on cores without SMT, isb is what Arm's porting
  // guidance substitutes for pause.
  __asm__ volatile("isb");
#endif
}
}  // namespace

uint32_t AwaitReserveGranule(const std::atomic<uint32_t>& granule) {
  // Bounded so a claim that never ends leaves the guest loop at its safepoint.
  constexpr uint32_t kSpinsPerYield = 64;
  constexpr uint32_t kYields = 4;
  uint32_t generation = granule.load(std::memory_order_acquire);
  for (uint32_t yields = 0; (generation & 1) && yields < kYields; ++yields) {
    for (uint32_t spins = 0; (generation & 1) && spins < kSpinsPerYield;
         ++spins) {
      SpinPause();
      generation = granule.load(std::memory_order_acquire);
    }
    if (generation & 1) {
      threading::MaybeYield();
      generation = granule.load(std::memory_order_acquire);
    }
  }
  return generation;
}

uint32_t Backend::ReservedLoad32(ppc::PPCContext* context, uint32_t address) {
  return xe::byte_swap(*context->TranslateVirtual<uint32_t*>(address));
}

uint64_t Backend::ReservedLoad64(ppc::PPCContext* context, uint32_t address) {
  return xe::byte_swap(*context->TranslateVirtual<uint64_t*>(address));
}

bool Backend::ReservedStore32(ppc::PPCContext* context, uint32_t address,
                              uint32_t value) {
  *context->TranslateVirtual<uint32_t*>(address) = xe::byte_swap(value);
  return true;
}

bool Backend::ReservedStore64(ppc::PPCContext* context, uint32_t address,
                              uint64_t value) {
  *context->TranslateVirtual<uint64_t*>(address) = xe::byte_swap(value);
  return true;
}

}  // namespace backend
}  // namespace cpu
}  // namespace xe
