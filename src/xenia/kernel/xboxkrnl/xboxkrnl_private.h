/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_XBOXKRNL_PRIVATE_H_
#define XENIA_KERNEL_XBOXKRNL_XBOXKRNL_PRIVATE_H_

#include "xenia/cpu/export_resolver.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_ordinals.h"

namespace xe {
namespace kernel {
namespace xboxkrnl {

xe::cpu::Export* RegisterExport_xboxkrnl(xe::cpu::Export* export_entry);

// --io_ledger_diag (default off): async-I/O completion ledger for the DOA5
// load-stall hunt. Implemented in xboxkrnl_io.cc; called from the APC delivery
// loop (xeProcessUserApcs) and the spin detector (DelayPollDiagProbe). Revert
// before any fix.
void IoLedgerOnApcDelivered(uint32_t normal_routine, uint32_t normal_context);
void IoLedgerDumpOutstanding(const char* reason);
// Loader inter-read wait probe: called from the kernel wait entries with the
// measured wait duration. Logs (and resolves the object type) only for an active
// reader thread (the loader) blocking >100ms — names the inter-read dependency.
void IoLedgerOnLoaderWait(const char* api, uint32_t handle_or_ptr, bool by_handle,
                          uint32_t count, uint64_t timeout, uint32_t alertable,
                          uint64_t dur_ms, uint32_t result, uint32_t lr,
                          uint32_t thid);
// Gate-signal tracer: called from the event Set paths with the event's guest
// object address. If it matches the loader's gate event, logs the signaler
// (the consumer) thid/LR + interval, and marks it so its waits get tapped too.
void IoLedgerOnEventSet(uint32_t event_guest_addr, const char* api,
                        uint32_t thid, uint32_t lr);

// Registration functions, one per file.
#define XE_MODULE_EXPORT_GROUP(m, n)                                  \
  void Register##n##Exports(xe::cpu::ExportResolver* export_resolver, \
                            KernelState* kernel_state);
#include "xboxkrnl_module_export_groups.inc"
#undef XE_MODULE_EXPORT_GROUP

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XBOXKRNL_XBOXKRNL_PRIVATE_H_
