/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_backend.h"

#include <cstddef>
#include <cstring>

#include "xenia/base/clock.h"
#include "xenia/base/exception_handler.h"
#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif
#if XE_ARCH_ARM64 && XE_COMPILER_MSVC
#include <intrin.h>
#endif
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
#include <pthread.h>
#endif
#include "xenia/cpu/backend/a64/a64_assembler.h"
#include "xenia/cpu/backend/a64/a64_code_cache.h"
#include "xenia/cpu/backend/a64/a64_emitter.h"
#include "xenia/cpu/backend/a64/a64_function.h"
#include "xenia/cpu/backend/a64/a64_sequences.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/breakpoint.h"
#include "xenia/cpu/mmio_handler.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/stack_walker.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/cpu/xex_module.h"

DEFINE_int64(a64_max_stackpoints, 65536,
             "Max number of host->guest stack mappings we can record.", "a64");

DEFINE_bool(a64_enable_host_guest_stack_synchronization, true,
            "Records entries for guest/host stack mappings at function starts "
            "and checks for reentry at return sites. Has slight performance "
            "impact, but fixes crashes in games that use setjmp/longjmp.",
            "a64");

DEFINE_bool(a64_longjmp_reentry, true,
            "Resume a guest longjmp inside the live frame of the function "
            "that called setjmp, restoring the host stack and stackpoints. "
            "Off translates the landing site as a new function, which leaves "
            "the skipped frames on the host stack (Fable III and Black Ops "
            "III then overflow the stackpoints). Needs "
            "a64_enable_host_guest_stack_synchronization.",
            "a64");

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

// Resolve a guest function at runtime. Called by the resolve thunk when
// a guest address has not yet been compiled.
uint64_t ResolveFunction(void* raw_context, uint64_t target_address);

// The stackpoint depth a branch at guest stack pointer |guest_sp| unwinds to
// (the innermost live frame entered above it, plus one), or 0 when the guest
// still runs in the innermost live frame. A function allocates its guest frame
// below the r1 it was entered with and only restores that r1 to return or tail
// call, which pop its stackpoint before the branch resolves; so a live frame
// entered at or below r1 is one the guest has already unwound.
//
// x64 (and Xenia Edge's a64) count only frames entered strictly below r1 and
// require two of them. The frame the setjmp caller called was entered at
// exactly the r1 longjmp restores and is never counted, so that misses a
// longjmp raised one or two calls below the setjmp caller. Fable III's Lua
// error path (luaD_throw under the protected call) takes that branch on
// script errors, and each miss translated the landing site as a new function
// that left the skipped frames on the host stack.
static uint32_t FindStackpointSyncDepth(const A64BackendStackpoint* stackpoints,
                                        uint32_t current_depth,
                                        uint32_t guest_sp) {
  if (!stackpoints || current_depth == 0) {
    return 0;
  }
  uint32_t idx = current_depth - 1;
  uint32_t frames_skipped = 0;
  while (idx != 0xFFFFFFFFu && guest_sp >= stackpoints[idx].guest_stack_) {
    --idx;
    ++frames_skipped;
  }
  if (idx == 0xFFFFFFFFu || frames_skipped == 0) {
    return 0;
  }
  return idx + 1;
}

// ==========================================================================
// A64HelperEmitter — generates thunks using xbyak_aarch64.
// ==========================================================================
class A64HelperEmitter : public A64Emitter {
 public:
  A64HelperEmitter(A64Backend* backend, XbyakA64Allocator* allocator);

  HostToGuestThunk EmitHostToGuestThunk();
  GuestToHostThunk EmitGuestToHostThunk();
  ResolveFunctionThunk EmitResolveFunctionThunk();
  void* EmitGuestAndHostSynchronizeStackHelper();
  void* EmitReservedStoreHelper(bool bit64, ReservedStoreFaultSite& fault_site);
  ReservedStoreThunk EmitReservedStoreThunk(void* helper);
};

A64HelperEmitter::A64HelperEmitter(A64Backend* backend,
                                   XbyakA64Allocator* allocator)
    : A64Emitter(backend, allocator) {}

// --------------------------------------------------------------------------
// HostToGuestThunk
// --------------------------------------------------------------------------
// Called from host C++ code to enter JIT'd guest code.
//   x0 = target machine code address
//   x1 = PPCContext* (arg0)
//   x2 = return address value (arg1)
//
// ARM64 AAPCS64 calling convention:
//   Caller-saved: x0-x17, v0-v7, v16-v31 (x18 is the platform register,
//                 reserved by Apple/Windows; never used by the emitter)
//   Callee-saved: x19-x28, x29(FP), x30(LR), d8-d15
//
// We save all callee-saved regs, set up context (x20) and membase (x21),
// then call the target. On return, restore and return to host.
HostToGuestThunk A64HelperEmitter::EmitHostToGuestThunk() {
  struct {
    size_t prolog;
    size_t prolog_stack_alloc;
    size_t body;
    size_t epilog;
    size_t tail;
  } code_offsets = {};

  code_offsets.prolog = getSize();

  // Allocate thunk stack frame.
  // Save x29(FP) and x30(LR) first, then callee-saved GPRs and NEON regs.
  const size_t thunk_stack = StackLayout::THUNK_STACK_SIZE;

  // sub sp, sp, #thunk_stack
  sub(sp, sp, static_cast<uint32_t>(thunk_stack));
  code_offsets.prolog_stack_alloc = getSize();

  // Save callee-saved GPRs: x19-x28, x29, x30
  stp(x19, x20, ptr(sp, 0x00));
  stp(x21, x22, ptr(sp, 0x10));
  stp(x23, x24, ptr(sp, 0x20));
  stp(x25, x26, ptr(sp, 0x30));
  stp(x27, x28, ptr(sp, 0x40));
  stp(x29, x30, ptr(sp, 0x50));

  // Save callee-saved NEON regs: full q8-q15 (JIT uses all 128 bits).
  stp(Xbyak_aarch64::QReg(8), Xbyak_aarch64::QReg(9), ptr(sp, 0x60));
  stp(Xbyak_aarch64::QReg(10), Xbyak_aarch64::QReg(11), ptr(sp, 0x80));
  stp(Xbyak_aarch64::QReg(12), Xbyak_aarch64::QReg(13), ptr(sp, 0xA0));
  stp(Xbyak_aarch64::QReg(14), Xbyak_aarch64::QReg(15), ptr(sp, 0xC0));

  code_offsets.body = getSize();

  // Set up guest execution state.
  // x20 = context (PPCContext*)
  mov(x20, x1);
  // x19 = backend context (immediately before PPCContext in memory)
  sub(x19, x20, static_cast<uint32_t>(sizeof(A64BackendContext)));
  // x21 = virtual_membase (loaded from context)
  ldr(x21, ptr(x20, static_cast<int32_t>(
                        offsetof(ppc::PPCContext, virtual_membase))));
  // Restore the guest scalar FPCR on every host->guest entry so host-side
  // work done before the call can't leak a stale rounding / non-IEEE mode.
  ldr(w11,
      ptr(x19, static_cast<uint32_t>(offsetof(A64BackendContext, fpcr_fpu))));
  msr(3, 3, 4, 4, 0, x11);
  // x0 still holds target, x2 holds return address.
  // The guest function's prolog stores x0 to GUEST_RET_ADDR on its stack
  // frame. Move the target to a scratch reg and put the guest return
  // address into x0.
  mov(x9, x0);  // x9 = target (scratch reg)
  // Pass guest return address in x0 (convention for guest function entry).
  mov(x0, x2);  // x0 = guest return address

  // Call the guest function.
  blr(x9);

  code_offsets.epilog = getSize();

  // Restore callee-saved NEON regs (full q8-q15).
  ldp(Xbyak_aarch64::QReg(14), Xbyak_aarch64::QReg(15), ptr(sp, 0xC0));
  ldp(Xbyak_aarch64::QReg(12), Xbyak_aarch64::QReg(13), ptr(sp, 0xA0));
  ldp(Xbyak_aarch64::QReg(10), Xbyak_aarch64::QReg(11), ptr(sp, 0x80));
  ldp(Xbyak_aarch64::QReg(8), Xbyak_aarch64::QReg(9), ptr(sp, 0x60));

  // Restore callee-saved GPRs.
  ldp(x29, x30, ptr(sp, 0x50));
  ldp(x27, x28, ptr(sp, 0x40));
  ldp(x25, x26, ptr(sp, 0x30));
  ldp(x23, x24, ptr(sp, 0x20));
  ldp(x21, x22, ptr(sp, 0x10));
  ldp(x19, x20, ptr(sp, 0x00));

  // Deallocate stack.
  add(sp, sp, static_cast<uint32_t>(thunk_stack));
  ret();

  code_offsets.tail = getSize();

  EmitFunctionInfo func_info = {};
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;
  func_info.stack_size = thunk_stack;
  func_info.lr_save_offset = 0x058;  // stp x29, x30, [sp, #0x50]
  func_info.frame_kind = EmitFunctionInfo::FrameKind::kHostToGuestThunk;

  void* fn = Emplace(func_info);
  return reinterpret_cast<HostToGuestThunk>(fn);
}

// --------------------------------------------------------------------------
// GuestToHostThunk
// --------------------------------------------------------------------------
// Called from guest JIT code to transition into a host (C++) function.
//   x0 = target host function
//   x1 = arg0
//   x2 = arg1
//
// We save volatile guest registers that we need to preserve across the
// host call, then call the host function with context as the first arg.
GuestToHostThunk A64HelperEmitter::EmitGuestToHostThunk() {
  struct {
    size_t prolog;
    size_t prolog_stack_alloc;
    size_t body;
    size_t epilog;
    size_t tail;
  } code_offsets = {};

  code_offsets.prolog = getSize();

  // The guest JIT uses v4-v15, v16-v31 as allocatable VEC regs.
  // v0-v7, v16-v31 are caller-saved in AAPCS64 (fully clobbered by C).
  // v8-v15 lower 64 bits are callee-saved, but upper 64 bits are not.
  // We must save all guest-allocated VEC regs (full 128-bit Q regs).
  // GPRs x19-x28 are callee-saved in AAPCS64, so the C function preserves them.
  //
  // Stack layout:
  //   q4, q5       sp + 0x000  (32 bytes)
  //   q6, q7       sp + 0x020
  //   q8, q9       sp + 0x040
  //   q10, q11     sp + 0x060
  //   q12, q13     sp + 0x080
  //   q14, q15     sp + 0x0A0
  //   q16, q17     sp + 0x0C0
  //   q18, q19     sp + 0x0E0
  //   q20, q21     sp + 0x100
  //   q22, q23     sp + 0x120
  //   q24, q25     sp + 0x140
  //   q26, q27     sp + 0x160
  //   q28, q29     sp + 0x180
  //   q30, q31     sp + 0x1A0
  //   x29, x30     sp + 0x1C0
  //   Total: 0x1D0 = 464 bytes (16-byte aligned)
  const size_t g2h_stack = 464;
  sub(sp, sp, static_cast<uint32_t>(g2h_stack));
  code_offsets.prolog_stack_alloc = getSize();

  // Save guest-allocated VEC regs (full Q = 128-bit).
  stp(Xbyak_aarch64::QReg(4), Xbyak_aarch64::QReg(5), ptr(sp, 0x000));
  stp(Xbyak_aarch64::QReg(6), Xbyak_aarch64::QReg(7), ptr(sp, 0x020));
  stp(Xbyak_aarch64::QReg(8), Xbyak_aarch64::QReg(9), ptr(sp, 0x040));
  stp(Xbyak_aarch64::QReg(10), Xbyak_aarch64::QReg(11), ptr(sp, 0x060));
  stp(Xbyak_aarch64::QReg(12), Xbyak_aarch64::QReg(13), ptr(sp, 0x080));
  stp(Xbyak_aarch64::QReg(14), Xbyak_aarch64::QReg(15), ptr(sp, 0x0A0));
  stp(Xbyak_aarch64::QReg(16), Xbyak_aarch64::QReg(17), ptr(sp, 0x0C0));
  stp(Xbyak_aarch64::QReg(18), Xbyak_aarch64::QReg(19), ptr(sp, 0x0E0));
  stp(Xbyak_aarch64::QReg(20), Xbyak_aarch64::QReg(21), ptr(sp, 0x100));
  stp(Xbyak_aarch64::QReg(22), Xbyak_aarch64::QReg(23), ptr(sp, 0x120));
  stp(Xbyak_aarch64::QReg(24), Xbyak_aarch64::QReg(25), ptr(sp, 0x140));
  stp(Xbyak_aarch64::QReg(26), Xbyak_aarch64::QReg(27), ptr(sp, 0x160));
  stp(Xbyak_aarch64::QReg(28), Xbyak_aarch64::QReg(29), ptr(sp, 0x180));
  stp(Xbyak_aarch64::QReg(30), Xbyak_aarch64::QReg(31), ptr(sp, 0x1A0));
  // Save x29/x30 (FP/LR).
  stp(x29, x30, ptr(sp, 0x1C0));

  code_offsets.body = getSize();

  // Host C code runs with the AAPCS64 default FPCR (round to nearest, no
  // flush-to-zero), not the guest's scalar mode: mtfsf can put rounding
  // controls or FPSCR.NI (FZ) in fpcr_fpu, which host code must not inherit.
  Xbyak_aarch64::Label host_fpcr_ready;
  mrs(x11, 3, 3, 4, 4, 0);  // mrs x11, FPCR
  cbz(x11, host_fpcr_ready);
  msr(3, 3, 4, 4, 0, xzr);
  L(host_fpcr_ready);

  // Call host function.
  // AAPCS64: x0=first arg. We set x0=context (from x20).
  mov(x9, x0);   // x9 = target function (scratch)
  mov(x0, x20);  // x0 = PPCContext* (our context reg)
  // x1, x2, x3 already hold args from the caller.
  blr(x9);

  // Host callbacks may change FPCR. Restore the guest scalar FPCR before
  // resuming the JIT so later guest ops observe the cached PPC mode.
  // x19 (backend context) is callee-saved, so it survives the host call.
  ldr(w11,
      ptr(x19, static_cast<uint32_t>(offsetof(A64BackendContext, fpcr_fpu))));
  msr(3, 3, 4, 4, 0, x11);

  code_offsets.epilog = getSize();

  // Restore.
  ldp(x29, x30, ptr(sp, 0x1C0));
  ldp(Xbyak_aarch64::QReg(30), Xbyak_aarch64::QReg(31), ptr(sp, 0x1A0));
  ldp(Xbyak_aarch64::QReg(28), Xbyak_aarch64::QReg(29), ptr(sp, 0x180));
  ldp(Xbyak_aarch64::QReg(26), Xbyak_aarch64::QReg(27), ptr(sp, 0x160));
  ldp(Xbyak_aarch64::QReg(24), Xbyak_aarch64::QReg(25), ptr(sp, 0x140));
  ldp(Xbyak_aarch64::QReg(22), Xbyak_aarch64::QReg(23), ptr(sp, 0x120));
  ldp(Xbyak_aarch64::QReg(20), Xbyak_aarch64::QReg(21), ptr(sp, 0x100));
  ldp(Xbyak_aarch64::QReg(18), Xbyak_aarch64::QReg(19), ptr(sp, 0x0E0));
  ldp(Xbyak_aarch64::QReg(16), Xbyak_aarch64::QReg(17), ptr(sp, 0x0C0));
  ldp(Xbyak_aarch64::QReg(14), Xbyak_aarch64::QReg(15), ptr(sp, 0x0A0));
  ldp(Xbyak_aarch64::QReg(12), Xbyak_aarch64::QReg(13), ptr(sp, 0x080));
  ldp(Xbyak_aarch64::QReg(10), Xbyak_aarch64::QReg(11), ptr(sp, 0x060));
  ldp(Xbyak_aarch64::QReg(8), Xbyak_aarch64::QReg(9), ptr(sp, 0x040));
  ldp(Xbyak_aarch64::QReg(6), Xbyak_aarch64::QReg(7), ptr(sp, 0x020));
  ldp(Xbyak_aarch64::QReg(4), Xbyak_aarch64::QReg(5), ptr(sp, 0x000));

  add(sp, sp, static_cast<uint32_t>(g2h_stack));
  ret();

  code_offsets.tail = getSize();

  EmitFunctionInfo func_info = {};
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;
  func_info.stack_size = g2h_stack;
  func_info.lr_save_offset = 0x1C8;  // stp x29, x30, [sp, #0x1C0]
  func_info.frame_kind = EmitFunctionInfo::FrameKind::kGuestToHostThunk;

  void* fn = Emplace(func_info);
  return reinterpret_cast<GuestToHostThunk>(fn);
}

// --------------------------------------------------------------------------
// ResolveFunctionThunk
// --------------------------------------------------------------------------
// Called when guest code calls an unresolved function address.
// The indirection table initially points all entries here.
// We call ResolveFunction to compile/lookup the target, then jump to it.
//
// On entry from the indirection table:
//   w16 = guest PPC address (loaded by the call sequence)
//   x20 = context
//   x30 = return address (from the BLR that got us here)
ResolveFunctionThunk A64HelperEmitter::EmitResolveFunctionThunk() {
  struct {
    size_t prolog;
    size_t prolog_stack_alloc;
    size_t body;
    size_t epilog;
    size_t tail;
  } code_offsets = {};

  code_offsets.prolog = getSize();

  const size_t thunk_stack = StackLayout::THUNK_STACK_SIZE;
  sub(sp, sp, static_cast<uint32_t>(thunk_stack));
  code_offsets.prolog_stack_alloc = getSize();

  // Save x29/x30 and x0 (guest return address, needed by the resolved
  // function's prolog). x19 is callee-saved so it survives the C call.
  stp(x29, x30, ptr(sp, 0x50));
  stp(x0, x19, ptr(sp, 0x00));  // save x0 (guest ret addr) and x19

  code_offsets.body = getSize();

  // Call ResolveFunction(context, target_address).
  mov(x0, x20);  // x0 = PPCContext*
  mov(x1, x16);  // x1 = guest address (32-bit in w16)
  // Load address of ResolveFunction.
  mov(x9, reinterpret_cast<uint64_t>(&ResolveFunction));
  blr(x9);
  // x0 now holds the resolved host machine code address.
  mov(x9, x0);

  code_offsets.epilog = getSize();

  // Restore x0 (guest return address) and saved regs.
  ldp(x0, x19, ptr(sp, 0x00));
  ldp(x29, x30, ptr(sp, 0x50));
  add(sp, sp, static_cast<uint32_t>(thunk_stack));

  cbz(x9, 8);   // skip br x9 if null, fall through to brk
  br(x9);       // Jump to the resolved function (tail call — preserves LR).
  brk(0xF000);  // Resolution failed — trap for debugging.

  code_offsets.tail = getSize();

  EmitFunctionInfo func_info = {};
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;
  func_info.stack_size = thunk_stack;
  func_info.lr_save_offset = 0x058;  // stp x29, x30, [sp, #0x50]
  func_info.frame_kind = EmitFunctionInfo::FrameKind::kResolveThunk;

  void* fn = Emplace(func_info);
  return reinterpret_cast<ResolveFunctionThunk>(fn);
}

// --------------------------------------------------------------------------
// GuestAndHostSynchronizeStackHelper
// --------------------------------------------------------------------------
// Branched to by the stack synchronization check after a guest call when
// ResolveFunction landed a guest longjmp in a live frame (ResolveLongjmp).
// Restores that frame's host SP and makes it the innermost stackpoint, which
// drops the frames the longjmp skipped, then resumes in the frame.
//
// On entry:
//   x8  = where to resume (the check's continuation in the landing frame)
//   x19 = A64BackendContext*
void* A64HelperEmitter::EmitGuestAndHostSynchronizeStackHelper() {
  using namespace Xbyak_aarch64;
  struct {
    size_t prolog;
    size_t prolog_stack_alloc;
    size_t body;
    size_t epilog;
    size_t tail;
  } code_offsets = {};

  code_offsets.prolog = getSize();
  code_offsets.prolog_stack_alloc = getSize();
  code_offsets.body = getSize();

  auto& underflow = NewCachedLabel();

  // x10 = stackpoints array, w11 = current depth, w13 = target depth.
  ldr(x10, ptr(x19, static_cast<uint32_t>(
                        offsetof(A64BackendContext, stackpoints))));
  ldr(w11, ptr(x19, static_cast<uint32_t>(offsetof(A64BackendContext,
                                                   current_stackpoint_depth))));
  ldr(w13, ptr(x19, static_cast<uint32_t>(offsetof(
                        A64BackendContext, pending_stackpoint_sync_depth))));
  cbz(x10, underflow);
  cbz(w13, underflow);
  // The landing frame must still be live.
  cmp(w13, w11);
  b(HI, underflow);

  // x14 = &stackpoints[target_depth - 1]
  sub(w14, w13, 1);
  mov(w15, static_cast<uint32_t>(sizeof(A64BackendStackpoint)));
  umull(x14, w14, w15);
  add(x14, x10, x14);

  // a64 records a stackpoint after the prolog allocates the frame, so its
  // host_stack_ is already the SP the landing frame's body runs at.
  ldr(x16, ptr(x14, static_cast<uint32_t>(
                        offsetof(A64BackendStackpoint, host_stack_))));
  mov(sp, x16);

  // The landing frame is the innermost live one now; consume the marker.
  str(w13, ptr(x19, static_cast<uint32_t>(offsetof(A64BackendContext,
                                                   current_stackpoint_depth))));
  str(wzr, ptr(x19, static_cast<uint32_t>(offsetof(
                        A64BackendContext, pending_stackpoint_sync_depth))));

  br(x8);

  L(underflow);
  // Should be impossible — stackpoint array underflowed.
  brk(0xF001);  // assertion failure

  code_offsets.epilog = getSize();
  code_offsets.tail = getSize();

  EmitFunctionInfo func_info = {};
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;
  func_info.stack_size = 0;

  return Emplace(func_info);
}

// ==========================================================================
// ResolveFunction — runtime function resolution.
// ==========================================================================
// Whether translated code at |host_address| starts with the stack
// synchronization check A64Emitter::EnsureSynchronizedGuestAndHostStack emits
// after a guest call: ldr w16, [x19, #pending_stackpoint_sync_depth].
static bool IsStackSyncCheck(uintptr_t host_address) {
  constexpr uint32_t offset = static_cast<uint32_t>(
      offsetof(A64BackendContext, pending_stackpoint_sync_depth));
  static_assert(offset % 4 == 0 && offset / 4 < 4096,
                "the check loads the marker with a scaled 12-bit offset");
  constexpr uint32_t kLdrW16FromX19 =
      0xB9400000u | ((offset / 4) << 10) | (19u << 5) | 16u;
  return *reinterpret_cast<const uint32_t*>(host_address) == kLdrW16FromX19;
}

// The function running in the live frame stackpoints[index], from the host
// address the next frame returns to, or nullptr when host code entered the
// next frame (a guest callback) or there is no next frame.
static A64Function* LookupFrameFunction(A64Backend* backend,
                                        const A64BackendStackpoint* stackpoints,
                                        uint32_t index, uint32_t depth) {
  if (index + 1 >= depth) {
    return nullptr;
  }
  // The next frame's prolog saved x30, which a tail call passes on unchanged.
  const uint64_t host_return = *reinterpret_cast<const uint64_t*>(
      stackpoints[index + 1].host_stack_ + StackLayout::HOST_RET_ADDR);
  return static_cast<A64Function*>(
      backend->code_cache()->LookupFunction(host_return));
}

// Where a branch to |target_address|, the instruction after a call, continues
// when the guest is unwinding to it past live frames (longjmp), or 0.
// The branch lands in the translated code of the live frame that made the
// call, and the stack synchronization check there restores that frame. Mirrors
// x64_emitter.cc's ResolveFunction and Xenia Edge's a64 ResolveLongjmp.
static uint64_t ResolveLongjmp(ppc::PPCContext* guest_context,
                               uint32_t target_address) {
  auto* backend = static_cast<A64Backend*>(guest_context->processor->backend());
  auto* backend_context = backend->BackendContextForGuestContext(guest_context);
  const A64BackendStackpoint* stackpoints = backend_context->stackpoints;
  const uint32_t depth = backend_context->current_stackpoint_depth;
  const uint32_t stack_pointer = static_cast<uint32_t>(guest_context->r[1]);
  const uint32_t sync_depth =
      FindStackpointSyncDepth(stackpoints, depth, stack_pointer);
  if (!sync_depth) {
    return 0;
  }
  // r1 alone can name the wrong frame (recursion, or a frame that made a
  // call at the same r1), so land in the innermost frame entered above r1
  // whose own function holds the target.
  for (uint32_t i = sync_depth; i-- > 0;) {
    // The frame the branch lands in runs below the r1 it was entered with.
    if (stackpoints[i].guest_stack_ <= stack_pointer) {
      continue;
    }
    auto* function = LookupFrameFunction(backend, stackpoints, i, depth);
    if (!function) {
      // Host code entered the next frame (a guest callback). Never unwind
      // past host frames.
      break;
    }
    // Landing on a function's first instruction would skip its prolog.
    if (function->address() == target_address) {
      continue;
    }
    // A function's range can hold unreachable instructions it has no code
    // for.
    const uintptr_t host_address =
        function->MapGuestAddressToMachineCode(target_address);
    if (!host_address) {
      continue;
    }
    // Only a return site after a guest call starts with the check that
    // restores the frame (one after a kernel call does not).
    if (!IsStackSyncCheck(host_address)) {
      return 0;
    }
    backend_context->pending_stackpoint_sync_depth = i + 1;
    return host_address;
  }
  return 0;
}

// stwcx./stdcx. on the reservation the last lwarx/ldarx took. One copy per
// width, called from JIT code and (through EmitReservedStoreThunk) from host
// code, so the fault handler has two sites to recognise.
// w12 = guest address, x1 = host address, x2 = value
// w0 = 1 if the store landed. Clobbers x0-x3, x8-x17 and the flags.
void* A64HelperEmitter::EmitReservedStoreHelper(
    bool bit64, ReservedStoreFaultSite& fault_site) {
  using namespace Xbyak_aarch64;
  struct {
    size_t prolog;
    size_t prolog_stack_alloc;
    size_t body;
    size_t epilog;
    size_t tail;
  } code_offsets = {};
  code_offsets.prolog = getSize();
  code_offsets.prolog_stack_alloc = getSize();
  code_offsets.body = getSize();

  auto bctx = [this](size_t offset) {
    return ptr(x19, static_cast<uint32_t>(offset));
  };
  constexpr uint32_t kReserveFlagBit = 1u << kA64BackendHasReserveBit;
  const bool lse = IsFeatureEnabled(kA64EmitLSE);
  Label fail, release_fail, claim_mismatch, value_mismatch;

  // stwcx. always clears the reservation, stored or not
  ldr(w8, bctx(offsetof(A64BackendContext, flags)));
  and_(w9, w8, static_cast<uint64_t>(~kReserveFlagBit));
  str(w9, bctx(offsetof(A64BackendContext, flags)));
  tst(w8, static_cast<uint64_t>(kReserveFlagBit));
  b(EQ, fail);

  // the reservation must be for the address we're storing to
  ldr(w10, bctx(offsetof(A64BackendContext, reserve_address)));
  cmp(w10, w12);
  b(NE, fail);

  ldr(x11, bctx(offsetof(A64BackendContext, reserve_helper_)));
  ubfx(w13, w12, A64_RESERVE_GRANULE_SHIFT, A64_RESERVE_ENTRY_BITS);
  add(x11, x11, x13, LSL, 2);
  // claim the generation (even to odd) so no other store can land until we
  // publish the next even value, odd snapshots always fail
  ldr(w15, bctx(offsetof(A64BackendContext, reserve_generation)));
  tbnz(w15, 0, fail);
  ldr(x16, bctx(offsetof(A64BackendContext, cached_reserve_value_)));

  // A pair that puts back the value it read has nothing to write: it holds if
  // memory still has the value and no reserved store reached the granule since
  // the lwarx, in that order. Job systems poll a queue head this way (USF4
  // exchanges it with zero from five threads); claiming for each poll bounced
  // the generation between them and every claim failed the others' pairs.
  Label changes_memory;
  if (bit64) {
    cmp(x2, x16);
    b(NE, changes_memory);
    ldar(x14, ptr(x1));
    cmp(x14, x16);
  } else {
    cmp(w2, w16);
    b(NE, changes_memory);
    ldar(w14, ptr(x1));
    cmp(w14, w16);
  }
  b(NE, fail);
  ldar(w14, ptr(x11));
  cmp(w14, w15);
  b(NE, fail);
  mov(w0, 1);
  ret();

  L(changes_memory);
  add(w17, w15, 1);

  // a fault here leaves the claim in x11 and w17 for the handler
  size_t first_access;
  size_t last_access;
  if (lse) {
    mov(w14, w15);
    casal(w14, w17, ptr(x11));
    cmp(w14, w15);
    b(NE, fail);

    // casal returns the old value in the compare register, so keep a copy
    mov(x3, x16);
    first_access = getSize();
    last_access = first_access;
    if (bit64) {
      casal(x16, x2, ptr(x1));
      cmp(x16, x3);
    } else {
      casal(w16, w2, ptr(x1));
      cmp(w16, w3);
    }
    b(NE, release_fail);
  } else {
    Label claim_retry, cas_retry;
    L(claim_retry);
    ldaxr(w14, ptr(x11));
    cmp(w14, w15);
    b(NE, claim_mismatch);
    stlxr(w13, w17, ptr(x11));
    cbnz(w13, claim_retry);

    L(cas_retry);
    first_access = getSize();
    if (bit64) {
      ldaxr(x14, ptr(x1));
      cmp(x14, x16);
      b(NE, value_mismatch);
      last_access = getSize();
      stlxr(w15, x2, ptr(x1));
    } else {
      ldaxr(w14, ptr(x1));
      cmp(w14, w16);
      b(NE, value_mismatch);
      last_access = getSize();
      stlxr(w15, w2, ptr(x1));
    }
    cbnz(w15, cas_retry);
  }

  add(w17, w17, 1);
  stlr(w17, ptr(x11));
  mov(w0, 1);
  ret();

  if (!lse) {
    // ldaxr compare failed, drop the exclusive monitor
    L(claim_mismatch);
    clrex(15);
    b(fail);
    L(value_mismatch);
    clrex(15);
  }
  // claimed, so publish even on failure
  L(release_fail);
  add(w17, w17, 1);
  stlr(w17, ptr(x11));
  const size_t fail_offset = getSize();
  L(fail);
  mov(w0, 0);
  ret();

  code_offsets.epilog = getSize();
  code_offsets.tail = getSize();

  EmitFunctionInfo func_info = {};
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;
  func_info.stack_size = 0;

  auto helper = reinterpret_cast<uint8_t*>(Emplace(func_info));
  fault_site.first_access = reinterpret_cast<uint64_t>(helper + first_access);
  fault_site.last_access = reinterpret_cast<uint64_t>(helper + last_access);
  fault_site.fail = reinterpret_cast<uint64_t>(helper + fail_offset);
  return helper;
}

// Puts host arguments where the stwcx. helper takes them, so a fault in a host
// store is released the same way as one in guest code.
// w0 = guest address, x1 = context, x2 = value, x3 = host address
ReservedStoreThunk A64HelperEmitter::EmitReservedStoreThunk(void* helper) {
  using namespace Xbyak_aarch64;
  struct {
    size_t prolog;
    size_t prolog_stack_alloc;
    size_t body;
    size_t epilog;
    size_t tail;
  } code_offsets = {};
  code_offsets.prolog = getSize();
  sub(sp, sp, static_cast<uint32_t>(16));
  code_offsets.prolog_stack_alloc = getSize();
  stp(x19, x30, ptr(sp, 0x00));
  code_offsets.body = getSize();

  sub(x19, x1, static_cast<uint32_t>(sizeof(A64BackendContext)));
  mov(w12, w0);
  mov(x1, x3);
  mov(x9, reinterpret_cast<uint64_t>(helper));
  blr(x9);

  code_offsets.epilog = getSize();
  ldp(x19, x30, ptr(sp, 0x00));
  add(sp, sp, static_cast<uint32_t>(16));
  ret();
  code_offsets.tail = getSize();

  EmitFunctionInfo func_info = {};
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;
  func_info.stack_size = 16;

  return reinterpret_cast<ReservedStoreThunk>(Emplace(func_info));
}

// ==========================================================================
// Reservation helpers. A global per-granule generation counter so
// cross-thread stores invalidate other threads' reservations. A CAS on the
// value alone would be ABA-vulnerable.
// ==========================================================================
namespace {

A64BackendContext* BackendContextFromRawContext(void* raw_context) {
  return reinterpret_cast<A64BackendContext*>(
      reinterpret_cast<uint8_t*>(raw_context) - sizeof(A64BackendContext));
}

std::atomic<uint32_t>& ReserveGranule(ReserveHelper* reserve_helper,
                                      uint32_t guest_address) {
  const uint32_t granule = guest_address >> A64_RESERVE_GRANULE_SHIFT;
  return reserve_helper->generations[granule & A64_RESERVE_ENTRY_MASK];
}

void TryAcquireReservation(void* raw_context, uint32_t guest_address) {
  auto* bctx = BackendContextFromRawContext(raw_context);
  // snapshot the generation first, the acquire pins the value read after
  bctx->reserve_generation = AwaitReserveGranule(
      ReserveGranule(bctx->reserve_helper_, guest_address));
  bctx->reserve_address = guest_address;
  // lwarx replaces any reservation this thread already held
  bctx->flags |= 1u << kA64BackendHasReserveBit;
}

template <typename T>
T ReservedLoadImpl(ppc::PPCContext* context, uint32_t address) {
  T* host_address = context->TranslateVirtual<T*>(address);
  TryAcquireReservation(context, address);
  auto* bctx = BackendContextFromRawContext(context);
  const T raw = *host_address;
  bctx->cached_reserve_value_ = static_cast<uint64_t>(raw);
  return xe::byte_swap(raw);
}

}  // namespace

uint64_t A64Backend::AwaitReservation(void* raw_context,
                                      uint64_t guest_address) {
  auto* bctx = BackendContextFromRawContext(raw_context);
  bctx->reserve_generation = AwaitReserveGranule(
      ReserveGranule(bctx->reserve_helper_, uint32_t(guest_address)));
  return 0;
}

uint64_t A64Backend::ReleaseFaultingReservedStore(void* data, Exception* ex) {
  auto* backend = reinterpret_cast<A64Backend*>(data);
  for (const auto& site : backend->reserved_store_fault_sites_) {
    if (ex->pc() < site.first_access || ex->pc() > site.last_access) {
      continue;
    }
    // x11 = the claimed granule, w17 = its odd generation
    const HostThreadContext& context = *ex->thread_context();
    reinterpret_cast<std::atomic<uint32_t>*>(context.x[11])
        ->store(uint32_t(context.x[17]) + 1, std::memory_order_release);
    return site.fail;
  }
  return 0;
}

uint32_t A64Backend::ReservedLoad32(ppc::PPCContext* context,
                                    uint32_t address) {
  return ReservedLoadImpl<uint32_t>(context, address);
}

uint64_t A64Backend::ReservedLoad64(ppc::PPCContext* context,
                                    uint32_t address) {
  return ReservedLoadImpl<uint64_t>(context, address);
}

bool A64Backend::ReservedStore32(ppc::PPCContext* context, uint32_t address,
                                 uint32_t value) {
  return reserved_store_32_thunk_(
      address, context, xe::byte_swap(value),
      context->TranslateVirtual<uint32_t*>(address));
}

bool A64Backend::ReservedStore64(ppc::PPCContext* context, uint32_t address,
                                 uint64_t value) {
  return reserved_store_64_thunk_(
      address, context, xe::byte_swap(value),
      context->TranslateVirtual<uint64_t*>(address));
}


uint64_t ResolveFunction(void* raw_context, uint64_t target_address) {
  auto guest_context = reinterpret_cast<ppc::PPCContext*>(raw_context);
  auto thread_state = guest_context->thread_state;
  assert_not_zero(target_address);

  // A guest longjmp branches (blr) to the instruction after its setjmp call.
  // Resume there in the live frame instead of translating the site as a new
  // function: that would leave every skipped frame on the host stack and in
  // the stackpoints, and every later return in the unwound chain would miss
  // its frame's return address too. Landing creates no function, so the
  // site's indirection slot keeps the resolve thunk and each longjmp comes
  // back here.
  if (cvars::a64_enable_host_guest_stack_synchronization &&
      cvars::a64_longjmp_reentry && target_address <= 0xFFFFFFFFu) {
    auto* processor = thread_state->processor();
    auto* xexmod = dynamic_cast<XexModule*>(
        processor->LookupModule(static_cast<uint32_t>(target_address)));
    if (xexmod) {
      InfoCacheFlags* flags = xexmod->GetInstructionAddressFlags(
          static_cast<uint32_t>(target_address));
      if (flags && flags->is_return_site) {
        if (uint64_t host_address = ResolveLongjmp(
                guest_context, static_cast<uint32_t>(target_address))) {
          return host_address;
        }
      }
    }
  }

  // Where there is no code the call faults on the instruction fetch, which
  // the guest's exception handlers can continue elsewhere.
  uint32_t address = static_cast<uint32_t>(target_address);
  auto fn =
      thread_state->processor()->ResolveCallTarget(guest_context, &address);
  if (!fn) {
    // Unresolvable — return 0, and the thunk traps (brk 0xF000). Name the
    // address and thread first: the crash reporter cannot walk JIT frames, so
    // this line is the only attribution a release build leaves behind. Flush
    // it: the trap kills the process before the log thread writes it out.
    XELOGE(
        "A64 ResolveFunction: guest address {:08X} does not resolve to a "
        "function (thread {}, lr {:08X}); trapping",
        uint32_t(target_address), thread_state->thread_id(),
        uint32_t(guest_context->lr));
    xe::FlushLog();
    return 0;
  }

  auto guest_fn = static_cast<GuestFunction*>(fn);
  auto code = guest_fn->machine_code();
  if (!code) {
    XELOGE(
        "A64 ResolveFunction: guest address {:08X} resolved but has no "
        "machine code (thread {}); trapping",
        uint32_t(target_address), thread_state->thread_id());
    xe::FlushLog();
    return 0;
  }
  return reinterpret_cast<uint64_t>(code);
}

// ==========================================================================
// A64Backend
// ==========================================================================

// ARM64 guest trampoline template.
// Loads proc, userdata1, userdata2 into x0-x2, then jumps to guest_to_host
// thunk via x9.  Each 64-bit immediate uses movz + 3x movk (16 bytes).
// Total: 4 registers × 16 bytes + 4 bytes (br x9) = 68 bytes.
//
// Template layout (offsets where 64-bit immediates are patched):
//   +0x00: movz x0, #imm16; movk x0, ..., lsl 16/32/48  -> proc
//   +0x10: movz x1, #imm16; movk x1, ..., lsl 16/32/48  -> userdata1
//   +0x20: movz x2, #imm16; movk x2, ..., lsl 16/32/48  -> userdata2
//   +0x30: movz x9, #imm16; movk x9, ..., lsl 16/32/48  -> g2h thunk
//   +0x40: br x9
//
// ARM64 encoding helpers:
//   movz xN, #imm16          = 0xD2800000 | (imm16 << 5) | N
//   movk xN, #imm16, lsl #S  = 0xF2800000 | (hw << 21) | (imm16 << 5) | N
//     where hw = S/16 (0,1,2,3)
static void EncodeMovImm64(uint32_t* out, uint32_t reg, uint64_t imm) {
  out[0] = 0xD2800000 | (static_cast<uint32_t>(imm & 0xFFFF) << 5) | reg;
  out[1] =
      0xF2A00000 | (static_cast<uint32_t>((imm >> 16) & 0xFFFF) << 5) | reg;
  out[2] =
      0xF2C00000 | (static_cast<uint32_t>((imm >> 32) & 0xFFFF) << 5) | reg;
  out[3] =
      0xF2E00000 | (static_cast<uint32_t>((imm >> 48) & 0xFFFF) << 5) | reg;
}

static constexpr size_t kGuestTrampolineSize = 68;  // 17 instructions × 4
static constexpr uint32_t kTrampolineOffsetProc = 0x00;
static constexpr uint32_t kTrampolineOffsetArg1 = 0x10;
static constexpr uint32_t kTrampolineOffsetArg2 = 0x20;
static constexpr uint32_t kTrampolineOffsetThunk = 0x30;

static void BuildGuestTrampoline(uint8_t* buf, void* proc, void* userdata1,
                                 void* userdata2, void* g2h_thunk) {
  auto* code = reinterpret_cast<uint32_t*>(buf);
  // x0 = proc (target function for guest-to-host thunk)
  EncodeMovImm64(&code[0], 0, reinterpret_cast<uint64_t>(proc));
  // x1 = userdata1
  EncodeMovImm64(&code[4], 1, reinterpret_cast<uint64_t>(userdata1));
  // x2 = userdata2
  EncodeMovImm64(&code[8], 2, reinterpret_cast<uint64_t>(userdata2));
  // x9 = guest_to_host_thunk
  EncodeMovImm64(&code[12], 9, reinterpret_cast<uint64_t>(g2h_thunk));
  // br x9
  code[16] = 0xD61F0120;  // br x9
}

A64Backend::A64Backend() {
  code_cache_ = A64CodeCache::Create();

  // Allocate executable memory for guest trampolines.
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
  // Apple Silicon forbids sub-4 GB mappings (default __PAGEZERO), so this buffer
  // cannot live near 0x10000. Let the kernel place it high (nullptr base); its
  // host address is stored full-width in the 64-bit indirection table
  // (see CreateGuestTrampoline's AddIndirection), so the location is free.
  void* buf = memory::AllocFixed(nullptr,
                                 kGuestTrampolineSize * MAX_GUEST_TRAMPOLINES,
                                 xe::memory::AllocationType::kReserveCommit,
                                 xe::memory::PageAccess::kExecuteReadWrite);
#else
  uint32_t base_address = 0x10000;
  void* buf = nullptr;
  while (base_address < 0x80000000) {
    buf = memory::AllocFixed(
        reinterpret_cast<void*>(static_cast<uintptr_t>(base_address)),
        kGuestTrampolineSize * MAX_GUEST_TRAMPOLINES,
        xe::memory::AllocationType::kReserveCommit,
        xe::memory::PageAccess::kExecuteReadWrite);
    if (!buf) {
      base_address += 65536;
    } else {
      break;
    }
  }
#endif
  xenia_assert(buf);
  guest_trampoline_memory_ = reinterpret_cast<uint8_t*>(buf);
  guest_trampoline_address_bitmap_.Resize(MAX_GUEST_TRAMPOLINES);
}

A64Backend::~A64Backend() {
  ExceptionHandler::Uninstall(&ExceptionCallbackThunk, this);
  // gone if Memory was torn down first
  if (auto mmio_handler = MMIOHandler::global_handler()) {
    mmio_handler->SetReservedStoreFaultCallback(nullptr, nullptr);
  }
  if (guest_trampoline_memory_) {
    memory::DeallocFixed(guest_trampoline_memory_,
                         kGuestTrampolineSize * MAX_GUEST_TRAMPOLINES,
                         memory::DeallocationType::kRelease);
    guest_trampoline_memory_ = nullptr;
  }
}

bool A64Backend::Initialize(Processor* processor) {
  if (!Backend::Initialize(processor)) {
    return false;
  }

  // Initialize the code cache.
  if (!code_cache_->Initialize()) {
    XELOGE("A64Backend: Failed to initialize code cache");
    return false;
  }

  // Expose the code cache to the base Backend class.
  Backend::code_cache_ = code_cache_.get();

  // Set up machine info for the register allocator.
  machine_info_.supports_extended_load_store = true;
  // GPR set: x22-x28 (7 registers; x19=backend ctx, x20=context, x21=membase)
  auto& gpr_set = machine_info_.register_sets[0];
  gpr_set.id = 0;
  std::strcpy(gpr_set.name, "gpr");
  gpr_set.types = MachineInfo::RegisterSet::INT_TYPES;
  gpr_set.count = A64Emitter::GPR_COUNT;
  // VEC set: v4-v15, v16-v31 (28 registers, v0-v3 scratch)
  auto& vec_set = machine_info_.register_sets[1];
  vec_set.id = 1;
  std::strcpy(vec_set.name, "vec");
  vec_set.types = MachineInfo::RegisterSet::FLOAT_TYPES |
                  MachineInfo::RegisterSet::VEC_TYPES;
  vec_set.count = A64Emitter::VEC_COUNT;

  // Generate thunks using ARM64 assembler.
  XbyakA64Allocator allocator;
  A64HelperEmitter thunk_emitter(this, &allocator);

  host_to_guest_thunk_ = thunk_emitter.EmitHostToGuestThunk();
  guest_to_host_thunk_ = thunk_emitter.EmitGuestToHostThunk();
  resolve_function_thunk_ = thunk_emitter.EmitResolveFunctionThunk();

  if (!host_to_guest_thunk_ || !guest_to_host_thunk_ ||
      !resolve_function_thunk_) {
    XELOGE("A64Backend: Failed to generate thunks");
    return false;
  }
  XELOGI("A64Backend: thunks host_to_guest={} guest_to_host={} resolve={}",
         reinterpret_cast<void*>(host_to_guest_thunk_),
         reinterpret_cast<void*>(guest_to_host_thunk_),
         reinterpret_cast<void*>(resolve_function_thunk_));

  if (cvars::a64_enable_host_guest_stack_synchronization) {
    synchronize_guest_and_host_stack_helper_ =
        thunk_emitter.EmitGuestAndHostSynchronizeStackHelper();
  }
  reserved_store_32_helper_ = thunk_emitter.EmitReservedStoreHelper(
      false, reserved_store_fault_sites_[0]);
  reserved_store_64_helper_ = thunk_emitter.EmitReservedStoreHelper(
      true, reserved_store_fault_sites_[1]);
  reserved_store_32_thunk_ =
      thunk_emitter.EmitReservedStoreThunk(reserved_store_32_helper_);
  reserved_store_64_thunk_ =
      thunk_emitter.EmitReservedStoreThunk(reserved_store_64_helper_);

  // Set the indirection table default to point at the resolve thunk.
  // Pass the full host address; the code cache truncates to the slot width where
  // it stays sub-4 GB (non-macOS arm64) and keeps all 64 bits on macOS arm64.
  code_cache_->set_indirection_default(
      reinterpret_cast<uintptr_t>(resolve_function_thunk_));

  // Commit the indirection table range used by guest trampolines so that
  // CreateGuestTrampoline can call AddIndirection without faulting.
  code_cache_->CommitExecutableRange(GUEST_TRAMPOLINE_BASE,
                                     GUEST_TRAMPOLINE_END);

  // Commit special indirection ranges (force return address, etc.).
  code_cache_->CommitExecutableRange(0x9FFF0000, 0x9FFFFFFF);

  // Register exception handler for MMIO access from JIT code.
  ExceptionHandler::Install(ExceptionCallbackThunk, this);
  if (auto mmio_handler = MMIOHandler::global_handler()) {
    mmio_handler->SetReservedStoreFaultCallback(ReleaseFaultingReservedStore,
                                                this);
  }

  return true;
}

void A64Backend::CommitExecutableRange(uint32_t guest_low,
                                       uint32_t guest_high) {
  code_cache_->CommitExecutableRange(guest_low, guest_high);
}

std::unique_ptr<Assembler> A64Backend::CreateAssembler() {
  return std::make_unique<A64Assembler>(this);
}

std::unique_ptr<GuestFunction> A64Backend::CreateGuestFunction(
    Module* module, uint32_t address) {
  return std::make_unique<A64Function>(module, address);
}

uint64_t A64Backend::CalculateNextHostInstruction(ThreadDebugInfo* thread_info,
                                                  uint64_t current_pc) {
  // ARM64 instructions are fixed 4 bytes.
  return current_pc + 4;
}

// ARM64 BRK #0 encoding (4 bytes, fixed-width instruction).
static constexpr uint32_t kArm64Brk0 = 0xD4200000;

void A64Backend::InstallBreakpoint(Breakpoint* breakpoint) {
  breakpoint->ForEachHostAddress([breakpoint](uint64_t host_address) {
    auto ptr = reinterpret_cast<void*>(host_address);
    auto original_bytes = xe::load<uint32_t>(ptr);
    assert_true(original_bytes != kArm64Brk0);
    xe::store<uint32_t>(ptr, kArm64Brk0);
    breakpoint->backend_data().emplace_back(host_address, original_bytes);
  });
}

void A64Backend::InstallBreakpoint(Breakpoint* breakpoint, Function* fn) {
  assert_true(breakpoint->address_type() == Breakpoint::AddressType::kGuest);
  assert_true(fn->is_guest());
  auto guest_function = reinterpret_cast<cpu::GuestFunction*>(fn);
  auto host_address =
      guest_function->MapGuestAddressToMachineCode(breakpoint->guest_address());
  if (!host_address) {
    assert_always();
    return;
  }

  auto ptr = reinterpret_cast<void*>(host_address);
  auto original_bytes = xe::load<uint32_t>(ptr);
  assert_true(original_bytes != kArm64Brk0);
  xe::store<uint32_t>(ptr, kArm64Brk0);
  breakpoint->backend_data().emplace_back(host_address, original_bytes);
}

void A64Backend::UninstallBreakpoint(Breakpoint* breakpoint) {
  for (auto& pair : breakpoint->backend_data()) {
    auto ptr = reinterpret_cast<uint8_t*>(pair.first);
    auto instruction_bytes = xe::load<uint32_t>(ptr);
    assert_true(instruction_bytes == kArm64Brk0);
    xe::store<uint32_t>(ptr, static_cast<uint32_t>(pair.second));
  }
  breakpoint->backend_data().clear();
}

bool A64FpcrUnified() {
  static const bool s_unified = [] {
    const char* v = std::getenv("XE_A64_FPCR_UNIFIED");
    return v && v[0] == '1' && !v[1];
  }();
  return s_unified;
}

void A64Backend::InitializeBackendContext(void* ctx) {
  auto* a64_ctx = BackendContextForGuestContext(ctx);
  std::memset(a64_ctx, 0, sizeof(A64BackendContext));
  a64_ctx->reserve_helper_ = &reserve_helper_;
  a64_ctx->Ox1000 = 0x1000;
  a64_ctx->fpcr_fpu =
      DEFAULT_FPU_FPCR | (A64FpcrUnified() ? DEFAULT_VMX_FPCR : 0u);
  a64_ctx->fpcr_vmx = DEFAULT_VMX_FPCR;
  a64_ctx->flags = (1U << kA64BackendNJMOn);  // NJM on by default
  a64_ctx->guest_tick_count = Clock::GetGuestTickCountPointer();

  // Allocate stackpoints for longjmp detection.
  if (cvars::a64_enable_host_guest_stack_synchronization) {
    uint64_t max_stackpoints = cvars::a64_max_stackpoints;
    if (max_stackpoints > 0) {
      a64_ctx->stackpoints = new A64BackendStackpoint[max_stackpoints]();
    }
  }

  // Reset the live host FPCR for a fresh PPC context so one test's rounding
  // state does not leak into the next on the shared PPC test runner thread.
  SetGuestRoundingMode(ctx, 0);
}

void A64Backend::DeinitializeBackendContext(void* ctx) {
  auto* a64_ctx = BackendContextForGuestContext(ctx);
  if (a64_ctx->stackpoints) {
    delete[] a64_ctx->stackpoints;
    a64_ctx->stackpoints = nullptr;
  }
}

void A64Backend::PrepareForReentry(void* ctx) {
  auto* a64_ctx = BackendContextForGuestContext(ctx);
  a64_ctx->current_stackpoint_depth = 0;
}

uint32_t A64Backend::CreateGuestTrampoline(GuestTrampolineProc proc,
                                           void* userdata1, void* userdata2,
                                           bool long_term) {
  size_t new_index;
  if (long_term) {
    new_index = guest_trampoline_address_bitmap_.AcquireFromBack();
  } else {
    new_index = guest_trampoline_address_bitmap_.Acquire();
  }
  xenia_assert(new_index != static_cast<size_t>(-1));

  uint8_t* write_pos =
      &guest_trampoline_memory_[kGuestTrampolineSize * new_index];

#if XE_PLATFORM_MAC && XE_ARCH_ARM64
  // guest_trampoline_memory_ is a MAP_JIT region (AllocFixed adds MAP_JIT for
  // executable anonymous mappings). On Apple Silicon it is writable XOR
  // executable per thread, so disable write-protection before emitting the
  // trampoline and re-enable it before the code can be branched to.
  pthread_jit_write_protect_np(0);
#endif
  BuildGuestTrampoline(write_pos, reinterpret_cast<void*>(proc), userdata1,
                       userdata2,
                       reinterpret_cast<void*>(guest_to_host_thunk_));
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
  pthread_jit_write_protect_np(1);
#endif

  // Flush instruction cache for the new trampoline code.
#if XE_PLATFORM_WIN32
  FlushInstructionCache(GetCurrentProcess(), write_pos, kGuestTrampolineSize);
#else
  __builtin___clear_cache(
      reinterpret_cast<char*>(write_pos),
      reinterpret_cast<char*>(write_pos + kGuestTrampolineSize));
#endif

  uint32_t indirection_guest_addr =
      GUEST_TRAMPOLINE_BASE +
      (static_cast<uint32_t>(new_index) * GUEST_TRAMPOLINE_MIN_LEN);

  code_cache()->AddIndirection(indirection_guest_addr,
                               reinterpret_cast<uintptr_t>(write_pos));

  return indirection_guest_addr;
}

void A64Backend::FreeGuestTrampoline(uint32_t trampoline_addr) {
  xenia_assert(trampoline_addr >= GUEST_TRAMPOLINE_BASE &&
               trampoline_addr < GUEST_TRAMPOLINE_END);
  size_t index =
      (trampoline_addr - GUEST_TRAMPOLINE_BASE) / GUEST_TRAMPOLINE_MIN_LEN;
  guest_trampoline_address_bitmap_.Release(index);
}

// PPC rounding mode (3-bit) to ARM64 FPCR value.
// Same table as in a64_sequences.cc SET_ROUNDING_MODE.
static constexpr uint32_t fpcr_table[8] = {
    (0b00 << 22),              // PPC 0: nearest, IEEE
    (0b11 << 22),              // PPC 1: toward zero, IEEE
    (0b01 << 22),              // PPC 2: toward +inf, IEEE
    (0b10 << 22),              // PPC 3: toward -inf, IEEE
    (0b00 << 22) | (1 << 24),  // PPC 4: nearest, flush-to-zero
    (0b11 << 22) | (1 << 24),  // PPC 5: toward zero, flush-to-zero
    (0b01 << 22) | (1 << 24),  // PPC 6: toward +inf, flush-to-zero
    (0b10 << 22) | (1 << 24),  // PPC 7: toward -inf, flush-to-zero
};

void A64Backend::SetGuestRoundingMode(void* ctx, unsigned int mode) {
  A64BackendContext* bctx = BackendContextForGuestContext(ctx);
  uint32_t control = mode & 7;
  uint32_t fpcr_val = fpcr_table[control];
  if (A64FpcrUnified()) {
    fpcr_val |= DEFAULT_VMX_FPCR;
  }
#if XE_COMPILER_MSVC
  // MSVC ARM64 intrinsic: ARM64_FPCR = register ID 0x5A20.
  _WriteStatusReg(0x5A20, static_cast<uint64_t>(fpcr_val));
#else
  __asm__ volatile("msr fpcr, %0" : : "r"(static_cast<uint64_t>(fpcr_val)));
#endif
  bctx->fpcr_fpu = fpcr_val;
  if (control & 0b100) {
    bctx->flags |= (1u << kA64BackendNonIEEEMode);
  } else {
    bctx->flags &= ~(1u << kA64BackendNonIEEEMode);
  }
  auto ppc_context = reinterpret_cast<ppc::PPCContext*>(ctx);
  ppc_context->fpscr.bits.rn = control;
  ppc_context->fpscr.bits.ni = control >> 2;
}

bool A64Backend::PopulatePseudoStacktrace(GuestPseudoStackTrace* st) {
  ThreadState* thrd_state = ThreadState::Get();
  if (!thrd_state) {
    return false;
  }
  ppc::PPCContext* ctx = thrd_state->context();
  A64BackendContext* backend_ctx = BackendContextForGuestContext(ctx);

  uint32_t depth = backend_ctx->current_stackpoint_depth - 1;
  if (static_cast<int32_t>(depth) < 1) {
    return false;
  }
  uint32_t num_entries_to_populate =
      std::min(MAX_GUEST_PSEUDO_STACKTRACE_ENTRIES, depth);

  st->count = num_entries_to_populate;
  st->truncated_flag = num_entries_to_populate < depth ? 1 : 0;

  A64BackendStackpoint* current_stackpoint =
      &backend_ctx->stackpoints[backend_ctx->current_stackpoint_depth - 1];

  for (uint32_t stp_index = 0; stp_index < num_entries_to_populate;
       ++stp_index) {
    st->return_addrs[stp_index] = current_stackpoint->guest_return_address_;
    current_stackpoint--;
  }
  return true;
}

void A64Backend::RecordMMIOExceptionForGuestInstruction(void* host_address) {
  uint64_t host_addr_u64 = reinterpret_cast<uint64_t>(host_address);
  auto fnfor = code_cache()->LookupFunction(host_addr_u64);
  if (fnfor) {
    uint32_t guestaddr = fnfor->MapMachineCodeToGuestAddress(host_addr_u64);
    Module* guest_module = fnfor->module();
    if (guest_module) {
      XexModule* xex_guest_module = dynamic_cast<XexModule*>(guest_module);
      if (xex_guest_module) {
        cpu::InfoCacheFlags* icf =
            xex_guest_module->GetInstructionAddressFlags(guestaddr);
        if (icf) {
          icf->accessed_mmio = true;
        }
      }
    }
  }
}

bool A64Backend::ExceptionCallbackThunk(Exception* ex, void* data) {
  auto* backend = reinterpret_cast<A64Backend*>(data);
  return backend->ExceptionCallback(ex);
}

bool A64Backend::ExceptionCallback(Exception* ex) {
  if (ex->code() != Exception::Code::kIllegalInstruction) {
    return false;
  }

  // Verify it's our BRK #0 instruction.
  auto instruction_bytes =
      xe::load<uint32_t>(reinterpret_cast<void*>(ex->pc()));
  if (instruction_bytes != kArm64Brk0) {
    return false;
  }

  return processor()->OnThreadBreakpointHit(ex);
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
