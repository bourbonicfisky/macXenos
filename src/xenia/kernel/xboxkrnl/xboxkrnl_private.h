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

// Delivers a guest call to an address with no code to the guest's frame-based
// exception handlers, as the console's kernel does for the instruction fetch
// fault the call raises there. Returns the address a handler continued at, with
// the context loaded from its CONTEXT record, and halts the thread as crashed
// if none did.
uint32_t DispatchCodeFetchFault(xe::cpu::ppc::PPCContext* context,
                                uint32_t address);

// Delivers a guest load or store at |pc| that faulted on |address| to the
// guest's frame-based exception handlers, as the console's kernel dispatches
// the access violation of a storage fault. Returns the address a handler
// continued at, with the context loaded from its CONTEXT record, or 0 if none
// did.
uint32_t DispatchDataFault(xe::cpu::ppc::PPCContext* context, uint32_t pc,
                           bool is_store, uint32_t address);

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
// Gate-signal tracer + per-object signal summary: called from the event Set/
// Pulse paths with the event's guest object address.
void IoLedgerOnEventSet(uint32_t event_guest_addr, const char* api,
                        uint32_t thid, uint32_t lr);
// Event Reset (KeResetEvent / NtClearEvent / auto-reset path) → signal summary.
void IoLedgerOnEventReset(uint32_t event_guest_addr, uint32_t thid, uint32_t lr);

// Passive wait registry (the "terminal unresolved wait" hunt). WaitEnter records
// an in-progress wait before it blocks; WaitExit clears it on return. On the
// spin/freeze dump, waits outstanding > a threshold are printed with each waited
// object's signal summary. Names the object the AsyncScheduler is stuck on.
void IoLedgerWaitEnter(const char* api, uint32_t wait_type, uint64_t timeout,
                       uint32_t alertable, uint32_t count, bool by_handle,
                       const uint32_t* handles);
void IoLedgerWaitExit();

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
