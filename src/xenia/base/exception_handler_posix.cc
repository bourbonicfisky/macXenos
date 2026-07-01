/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/exception_handler.h"

#include <signal.h>
#include "xenia/base/platform.h"
#if XE_PLATFORM_MAC
// On macOS the machine-context types live here and are exposed without the
// _XOPEN_SOURCE dance that <ucontext.h> requires.
#include <sys/ucontext.h>
#endif
#include <cstdint>

#include "xenia/base/assert.h"
#include "xenia/base/host_thread_context.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"

namespace xe {

bool signal_handlers_installed_ = false;
struct sigaction original_sigill_handler_;
struct sigaction original_sigsegv_handler_;
#if XE_PLATFORM_MAC
struct sigaction original_sigbus_handler_;
#endif  // XE_PLATFORM_MAC

// This can be as large as needed, but isn't often needed.
// As we will be sometimes firing many exceptions we want to avoid having to
// scan the table too much or invoke many custom handlers.
constexpr size_t kMaxHandlerCount = 8;

// All custom handlers, left-aligned and null terminated.
// Executed in order.
std::pair<ExceptionHandler::Handler, void*> handlers_[kMaxHandlerCount];

static void ExceptionHandlerCallback(int signal_number, siginfo_t* signal_info,
                                     void* signal_context) {
#if XE_PLATFORM_MAC
  // On macOS, uc_mcontext is a pointer to the machine context, and register
  // state lives in __ss/__es/__fs rather than in a gregs[] array.
  _STRUCT_MCONTEXT& mcontext =
      *reinterpret_cast<ucontext_t*>(signal_context)->uc_mcontext;
#else
  mcontext_t& mcontext =
      reinterpret_cast<ucontext_t*>(signal_context)->uc_mcontext;
#endif  // XE_PLATFORM_MAC

  HostThreadContext thread_context;

#if XE_ARCH_AMD64
#if XE_PLATFORM_MAC
  const _STRUCT_X86_THREAD_STATE64& ss = mcontext.__ss;
  thread_context.rip = uint64_t(ss.__rip);
  thread_context.eflags = uint32_t(ss.__rflags);
  thread_context.rax = uint64_t(ss.__rax);
  thread_context.rcx = uint64_t(ss.__rcx);
  thread_context.rdx = uint64_t(ss.__rdx);
  thread_context.rbx = uint64_t(ss.__rbx);
  thread_context.rsp = uint64_t(ss.__rsp);
  thread_context.rbp = uint64_t(ss.__rbp);
  thread_context.rsi = uint64_t(ss.__rsi);
  thread_context.rdi = uint64_t(ss.__rdi);
  thread_context.r8 = uint64_t(ss.__r8);
  thread_context.r9 = uint64_t(ss.__r9);
  thread_context.r10 = uint64_t(ss.__r10);
  thread_context.r11 = uint64_t(ss.__r11);
  thread_context.r12 = uint64_t(ss.__r12);
  thread_context.r13 = uint64_t(ss.__r13);
  thread_context.r14 = uint64_t(ss.__r14);
  thread_context.r15 = uint64_t(ss.__r15);
  // __fpu_xmm0..__fpu_xmm15 are declared consecutively, so they can be copied
  // as a single 16-register block.
  std::memcpy(thread_context.xmm_registers, &mcontext.__fs.__fpu_xmm0,
              sizeof(thread_context.xmm_registers));
#else
  thread_context.rip = uint64_t(mcontext.gregs[REG_RIP]);
  thread_context.eflags = uint32_t(mcontext.gregs[REG_EFL]);
  // The REG_ order may be different than the register indices in the
  // instruction encoding.
  thread_context.rax = uint64_t(mcontext.gregs[REG_RAX]);
  thread_context.rcx = uint64_t(mcontext.gregs[REG_RCX]);
  thread_context.rdx = uint64_t(mcontext.gregs[REG_RDX]);
  thread_context.rbx = uint64_t(mcontext.gregs[REG_RBX]);
  thread_context.rsp = uint64_t(mcontext.gregs[REG_RSP]);
  thread_context.rbp = uint64_t(mcontext.gregs[REG_RBP]);
  thread_context.rsi = uint64_t(mcontext.gregs[REG_RSI]);
  thread_context.rdi = uint64_t(mcontext.gregs[REG_RDI]);
  thread_context.r8 = uint64_t(mcontext.gregs[REG_R8]);
  thread_context.r9 = uint64_t(mcontext.gregs[REG_R9]);
  thread_context.r10 = uint64_t(mcontext.gregs[REG_R10]);
  thread_context.r11 = uint64_t(mcontext.gregs[REG_R11]);
  thread_context.r12 = uint64_t(mcontext.gregs[REG_R12]);
  thread_context.r13 = uint64_t(mcontext.gregs[REG_R13]);
  thread_context.r14 = uint64_t(mcontext.gregs[REG_R14]);
  thread_context.r15 = uint64_t(mcontext.gregs[REG_R15]);
  std::memcpy(thread_context.xmm_registers, mcontext.fpregs->_xmm,
              sizeof(thread_context.xmm_registers));
#endif  // XE_PLATFORM_MAC
#elif XE_ARCH_ARM64
#if XE_PLATFORM_MAC
  // Darwin arm64: registers live in __ss (thread state) and __ns (NEON), not in
  // the Linux sigcontext layout. __ss stores x0-x28 contiguously in __x[], with
  // x29 (fp), x30 (lr), sp and pc as separate fields. On arm64e (ptrauth) those
  // pointer fields are opaque and must be read via the accessor macros; on plain
  // arm64 (__DARWIN_OPAQUE_ARM_THREAD_STATE64 == 0) they are direct fields.
  const _STRUCT_ARM_THREAD_STATE64& ss = mcontext.__ss;
  std::memcpy(thread_context.x, ss.__x, sizeof(ss.__x));  // x0-x28
#if __DARWIN_OPAQUE_ARM_THREAD_STATE64
  thread_context.x[29] = uint64_t(__darwin_arm_thread_state64_get_fp(ss));
  thread_context.x[30] = uint64_t(__darwin_arm_thread_state64_get_lr(ss));
  thread_context.sp = uint64_t(__darwin_arm_thread_state64_get_sp(ss));
  thread_context.pc = uint64_t(__darwin_arm_thread_state64_get_pc(ss));
#else
  thread_context.x[29] = ss.__fp;
  thread_context.x[30] = ss.__lr;
  thread_context.sp = ss.__sp;
  thread_context.pc = ss.__pc;
#endif  // __DARWIN_OPAQUE_ARM_THREAD_STATE64
  thread_context.pstate = ss.__cpsr;
  const _STRUCT_ARM_NEON_STATE64& ns = mcontext.__ns;
  thread_context.fpsr = ns.__fpsr;
  thread_context.fpcr = ns.__fpcr;
  std::memcpy(thread_context.v, ns.__v, sizeof(thread_context.v));
#else
  std::memcpy(thread_context.x, mcontext.regs, sizeof(thread_context.x));
  thread_context.sp = mcontext.sp;
  thread_context.pc = mcontext.pc;
  thread_context.pstate = mcontext.pstate;
  struct fpsimd_context* mcontext_fpsimd = nullptr;
  struct esr_context* mcontext_esr = nullptr;
  for (struct _aarch64_ctx* mcontext_extension =
           reinterpret_cast<struct _aarch64_ctx*>(mcontext.__reserved);
       mcontext_extension->magic;
       mcontext_extension = reinterpret_cast<struct _aarch64_ctx*>(
           reinterpret_cast<uint8_t*>(mcontext_extension) +
           mcontext_extension->size)) {
    switch (mcontext_extension->magic) {
      case FPSIMD_MAGIC:
        mcontext_fpsimd =
            reinterpret_cast<struct fpsimd_context*>(mcontext_extension);
        break;
      case ESR_MAGIC:
        mcontext_esr =
            reinterpret_cast<struct esr_context*>(mcontext_extension);
        break;
      default:
        break;
    }
  }
  assert_not_null(mcontext_fpsimd);
  if (mcontext_fpsimd) {
    thread_context.fpsr = mcontext_fpsimd->fpsr;
    thread_context.fpcr = mcontext_fpsimd->fpcr;
    std::memcpy(thread_context.v, mcontext_fpsimd->vregs,
                sizeof(thread_context.v));
  }
#endif  // XE_PLATFORM_MAC
#endif  // XE_ARCH

  Exception ex;
  switch (signal_number) {
    case SIGILL:
      ex.InitializeIllegalInstruction(&thread_context);
      break;
#if XE_PLATFORM_MAC
    // macOS delivers memory protection violations (e.g. guest write-watch pages
    // mprotected read-only) as SIGBUS rather than SIGSEGV. Both carry the same
    // x86 page-fault error code in the exception state, so handle them
    // identically as access violations.
    case SIGBUS:
#endif  // XE_PLATFORM_MAC
    case SIGSEGV: {
      Exception::AccessViolationOperation access_violation_operation;
#if XE_PLATFORM_MAC
      // Fault address for macOS, refined below for x86_64 page faults. Defaults
      // to the (less reliable) BSD si_addr, which the ARM64 path still uses.
      uint64_t mac_fault_address =
          reinterpret_cast<uint64_t>(signal_info->si_addr);
#endif  // XE_PLATFORM_MAC
#if XE_ARCH_AMD64
      // x86_pf_error_code::X86_PF_WRITE
      constexpr uint64_t kX86PageFaultErrorCodeWrite = UINT64_C(1) << 1;
#if XE_PLATFORM_MAC
      // macOS x86_64 records the trap number, page-fault error code, and the
      // faulting linear address (CR2) in the exception state. The BSD signal's
      // si_addr is unreliable for Mach-translated SIGSEGV/SIGBUS (often 0), and
      // __err is a page-fault error code only when the trap is actually a page
      // fault (#PF == 14). For other traps (e.g. #GP == 13 from a non-canonical
      // address) the CPU records no fault address and __err is not a page-fault
      // code, so reporting a page-fault "read at 0x0" is wrong. Decode by trap:
      // for #PF use __faultvaddr + the error-code write bit; otherwise report an
      // unknown access rather than fabricating a read at a bogus address.
      constexpr uint16_t kX86TrapPageFault = 14;
      if (mcontext.__es.__trapno == kX86TrapPageFault) {
        mac_fault_address = uint64_t(mcontext.__es.__faultvaddr);
        access_violation_operation =
            (uint64_t(mcontext.__es.__err) & kX86PageFaultErrorCodeWrite)
                ? Exception::AccessViolationOperation::kWrite
                : Exception::AccessViolationOperation::kRead;
      } else {
        access_violation_operation =
            Exception::AccessViolationOperation::kUnknown;
      }
#else
      access_violation_operation =
          (uint64_t(mcontext.gregs[REG_ERR]) & kX86PageFaultErrorCodeWrite)
              ? Exception::AccessViolationOperation::kWrite
              : Exception::AccessViolationOperation::kRead;
#endif  // XE_PLATFORM_MAC
#elif XE_ARCH_ARM64
      // For a Data Abort (EC - ESR_EL1 bits 31:26 - 0b100100 from a lower
      // Exception Level, 0b100101 without a change in the Exception Level),
      // bit 6 is 0 for reading from a memory location, 1 for writing to a
      // memory location.
#if XE_PLATFORM_MAC
      // Contrary to a common belief, macOS *does* surface the ESR and the
      // faulting virtual address via the ARM exception state (__es). Use them:
      // si_addr is unreliable for Mach-translated SIGSEGV/SIGBUS (often 0), and
      // a 0 fault address makes the write-watch handler decline a recoverable
      // fault and crash via the restore-default path. Decode direction from the
      // ESR WnR bit, mirroring the Linux path, and only fall back to decoding
      // the faulting instruction if the syndrome is not a Data Abort or
      // Instruction Abort.
      const _STRUCT_ARM_EXCEPTION_STATE64& es = mcontext.__es;
      mac_fault_address = uint64_t(es.__far);
      const uint32_t mac_esr = es.__esr;
      const uint32_t mac_esr_ec = (mac_esr >> 26) & 0b111111;
      const bool is_data_abort = (mac_esr_ec & 0b111110) == 0b100100;
      const bool is_instruction_abort =
          (mac_esr_ec & 0b111110) == 0b100000;
      if (is_data_abort) {
        access_violation_operation =
            (mac_esr & (UINT32_C(1) << 6))
                ? Exception::AccessViolationOperation::kWrite
                : Exception::AccessViolationOperation::kRead;
      } else if (is_instruction_abort || !thread_context.pc) {
        access_violation_operation =
            Exception::AccessViolationOperation::kUnknown;
      } else {
        bool instruction_is_store;
        uint32_t instruction = 0;
        size_t pc_region_length = sizeof(instruction);
        xe::memory::PageAccess pc_access = xe::memory::PageAccess::kNoAccess;
        const bool can_read_pc =
            xe::memory::QueryProtect(reinterpret_cast<void*>(thread_context.pc),
                                     pc_region_length, pc_access) &&
            pc_region_length >= sizeof(instruction) &&
            pc_access != xe::memory::PageAccess::kNoAccess;
        if (can_read_pc) {
          instruction =
              *reinterpret_cast<const uint32_t*>(thread_context.pc);
        }
        if (can_read_pc &&
            IsArm64LoadPrefetchStore(instruction, instruction_is_store)) {
          access_violation_operation =
              instruction_is_store ? Exception::AccessViolationOperation::kWrite
                                   : Exception::AccessViolationOperation::kRead;
        } else {
          access_violation_operation =
              Exception::AccessViolationOperation::kUnknown;
        }
      }
#else
      if (mcontext_esr && ((mcontext_esr->esr >> 26) & 0b111110) == 0b100100) {
        access_violation_operation =
            (mcontext_esr->esr & (UINT64_C(1) << 6))
                ? Exception::AccessViolationOperation::kWrite
                : Exception::AccessViolationOperation::kRead;
      } else {
        // Determine the memory access direction based on which instruction has
        // requested it.
        // esr_context may be unavailable on certain hosts (for instance, on
        // Android, it was added only in NDK r16 - which is the first NDK
        // version to support the Android API level 27, while NDK r15 doesn't
        // have esr_context in its API 26 sigcontext.h).
        // On AArch64 (unlike on AArch32), the program counter is the address of
        // the currently executing instruction.
        bool instruction_is_store;
        if (IsArm64LoadPrefetchStore(
                *reinterpret_cast<const uint32_t*>(mcontext.pc),
                instruction_is_store)) {
          access_violation_operation =
              instruction_is_store ? Exception::AccessViolationOperation::kWrite
                                   : Exception::AccessViolationOperation::kRead;
        } else {
          assert_always(
              "No ESR in the exception thread context, or it's not a Data "
              "Abort, and the faulting instruction is not a known load, "
              "prefetch or store instruction");
          access_violation_operation =
              Exception::AccessViolationOperation::kUnknown;
        }
      }
#endif  // XE_PLATFORM_MAC
#else
      access_violation_operation =
          Exception::AccessViolationOperation::kUnknown;
#endif  // XE_ARCH
      ex.InitializeAccessViolation(
          &thread_context,
#if XE_PLATFORM_MAC
          mac_fault_address,
#else
          reinterpret_cast<uint64_t>(signal_info->si_addr),
#endif  // XE_PLATFORM_MAC
          access_violation_operation);
    } break;
    default:
      assert_unhandled_case(signal_number);
  }

  for (size_t i = 0; i < xe::countof(handlers_) && handlers_[i].first; ++i) {
    if (handlers_[i].first(&ex, handlers_[i].second)) {
      // Exception handled.
#if XE_ARCH_AMD64
#if XE_PLATFORM_MAC
      _STRUCT_X86_THREAD_STATE64& ss_out = mcontext.__ss;
      ss_out.__rip = thread_context.rip;
      ss_out.__rflags = thread_context.eflags;
      uint32_t modified_register_index;
      // Pointers to the macOS thread-state integer registers, in X64Register
      // order (must match the order in X64Register).
      uint64_t* const kIntRegisterMap[] = {
          &ss_out.__rax, &ss_out.__rcx, &ss_out.__rdx, &ss_out.__rbx,
          &ss_out.__rsp, &ss_out.__rbp, &ss_out.__rsi, &ss_out.__rdi,
          &ss_out.__r8,  &ss_out.__r9,  &ss_out.__r10, &ss_out.__r11,
          &ss_out.__r12, &ss_out.__r13, &ss_out.__r14, &ss_out.__r15,
      };
      uint16_t modified_int_registers_remaining = ex.modified_int_registers();
      while (xe::bit_scan_forward(modified_int_registers_remaining,
                                  &modified_register_index)) {
        modified_int_registers_remaining &=
            ~(UINT16_C(1) << modified_register_index);
        *kIntRegisterMap[modified_register_index] =
            thread_context.int_registers[modified_register_index];
      }
      uint16_t modified_xmm_registers_remaining = ex.modified_xmm_registers();
      while (xe::bit_scan_forward(modified_xmm_registers_remaining,
                                  &modified_register_index)) {
        modified_xmm_registers_remaining &=
            ~(UINT16_C(1) << modified_register_index);
        // __fpu_xmm0..__fpu_xmm15 are consecutive 16-byte registers.
        std::memcpy((&mcontext.__fs.__fpu_xmm0) + modified_register_index,
                    &thread_context.xmm_registers[modified_register_index],
                    sizeof(vec128_t));
      }
#else
      mcontext.gregs[REG_RIP] = greg_t(thread_context.rip);
      mcontext.gregs[REG_EFL] = greg_t(thread_context.eflags);
      uint32_t modified_register_index;
      // The order must match the order in X64Register.
      static constexpr size_t kIntRegisterMap[] = {
          REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP,
          REG_RSI, REG_RDI, REG_R8,  REG_R9,  REG_R10, REG_R11,
          REG_R12, REG_R13, REG_R14, REG_R15,
      };
      uint16_t modified_int_registers_remaining = ex.modified_int_registers();
      while (xe::bit_scan_forward(modified_int_registers_remaining,
                                  &modified_register_index)) {
        modified_int_registers_remaining &=
            ~(UINT16_C(1) << modified_register_index);
        mcontext.gregs[kIntRegisterMap[modified_register_index]] =
            thread_context.int_registers[modified_register_index];
      }
      uint16_t modified_xmm_registers_remaining = ex.modified_xmm_registers();
      while (xe::bit_scan_forward(modified_xmm_registers_remaining,
                                  &modified_register_index)) {
        modified_xmm_registers_remaining &=
            ~(UINT16_C(1) << modified_register_index);
        std::memcpy(&mcontext.fpregs->_xmm[modified_register_index],
                    &thread_context.xmm_registers[modified_register_index],
                    sizeof(vec128_t));
      }
#endif  // XE_PLATFORM_MAC
#elif XE_ARCH_ARM64
#if XE_PLATFORM_MAC
      _STRUCT_ARM_THREAD_STATE64& ss_out = mcontext.__ss;
      uint32_t modified_register_index;
      uint32_t modified_x_registers_remaining = ex.modified_x_registers();
      while (xe::bit_scan_forward(modified_x_registers_remaining,
                                  &modified_register_index)) {
        modified_x_registers_remaining &=
            ~(UINT32_C(1) << modified_register_index);
        // x0-x28 are the contiguous __x[] array; x29/x30 are the separate
        // fp/lr fields, restored unconditionally below.
        if (modified_register_index < 29) {
          ss_out.__x[modified_register_index] =
              thread_context.x[modified_register_index];
        }
      }
#if __DARWIN_OPAQUE_ARM_THREAD_STATE64
      __darwin_arm_thread_state64_set_fp(ss_out, thread_context.x[29]);
      __darwin_arm_thread_state64_set_lr_fptr(ss_out, thread_context.x[30]);
      __darwin_arm_thread_state64_set_sp(ss_out, thread_context.sp);
      __darwin_arm_thread_state64_set_pc_fptr(ss_out, thread_context.pc);
#else
      ss_out.__fp = thread_context.x[29];
      ss_out.__lr = thread_context.x[30];
      ss_out.__sp = thread_context.sp;
      ss_out.__pc = thread_context.pc;
#endif  // __DARWIN_OPAQUE_ARM_THREAD_STATE64
      ss_out.__cpsr = uint32_t(thread_context.pstate);
      _STRUCT_ARM_NEON_STATE64& ns_out = mcontext.__ns;
      ns_out.__fpsr = thread_context.fpsr;
      ns_out.__fpcr = thread_context.fpcr;
      uint32_t modified_v_registers_remaining = ex.modified_v_registers();
      while (xe::bit_scan_forward(modified_v_registers_remaining,
                                  &modified_register_index)) {
        modified_v_registers_remaining &=
            ~(UINT32_C(1) << modified_register_index);
        std::memcpy(&ns_out.__v[modified_register_index],
                    &thread_context.v[modified_register_index],
                    sizeof(vec128_t));
      }
#else
      uint32_t modified_register_index;
      uint32_t modified_x_registers_remaining = ex.modified_x_registers();
      while (xe::bit_scan_forward(modified_x_registers_remaining,
                                  &modified_register_index)) {
        modified_x_registers_remaining &=
            ~(UINT32_C(1) << modified_register_index);
        mcontext.regs[modified_register_index] =
            thread_context.x[modified_register_index];
      }
      mcontext.sp = thread_context.sp;
      mcontext.pc = thread_context.pc;
      mcontext.pstate = thread_context.pstate;
      if (mcontext_fpsimd) {
        mcontext_fpsimd->fpsr = thread_context.fpsr;
        mcontext_fpsimd->fpcr = thread_context.fpcr;
        uint32_t modified_v_registers_remaining = ex.modified_v_registers();
        while (xe::bit_scan_forward(modified_v_registers_remaining,
                                    &modified_register_index)) {
          modified_v_registers_remaining &=
              ~(UINT32_C(1) << modified_register_index);
          std::memcpy(&mcontext_fpsimd->vregs[modified_register_index],
                      &thread_context.v[modified_register_index],
                      sizeof(vec128_t));
          mcontext.regs[modified_register_index] =
              thread_context.x[modified_register_index];
        }
      }
#endif  // XE_PLATFORM_MAC
#endif  // XE_ARCH
      return;
    }
  }

#if XE_PLATFORM_MAC
  // No registered handler resolved this exception. Just returning would
  // re-execute the faulting instruction and fault again forever (macOS surfaces
  // guest access violations as SIGBUS), turning a real bug into a silent
  // livelock. Restore the original disposition and let the process crash with a
  // proper report at the faulting instruction instead.
  XELOGE(
      "ExceptionHandler: unhandled signal {} at pc={:#018x} fault_address="
      "{:#018x}; restoring default handler to crash with diagnostics",
      signal_number,
#if XE_ARCH_AMD64
      uint64_t(thread_context.rip),
#else
      uint64_t(thread_context.pc),
#endif  // XE_ARCH
      uint64_t(reinterpret_cast<uintptr_t>(signal_info->si_addr)));
  switch (signal_number) {
    case SIGILL:
      sigaction(SIGILL, &original_sigill_handler_, nullptr);
      break;
    case SIGSEGV:
      sigaction(SIGSEGV, &original_sigsegv_handler_, nullptr);
      break;
    case SIGBUS:
      sigaction(SIGBUS, &original_sigbus_handler_, nullptr);
      break;
    default:
      break;
  }
#endif  // XE_PLATFORM_MAC
}

void ExceptionHandler::Install(Handler fn, void* data) {
  if (!signal_handlers_installed_) {
    struct sigaction signal_handler;

    std::memset(&signal_handler, 0, sizeof(signal_handler));
    signal_handler.sa_sigaction = ExceptionHandlerCallback;
    signal_handler.sa_flags = SA_SIGINFO;

    if (sigaction(SIGILL, &signal_handler, &original_sigill_handler_) != 0) {
      assert_always("Failed to install new SIGILL handler");
    }
    if (sigaction(SIGSEGV, &signal_handler, &original_sigsegv_handler_) != 0) {
      assert_always("Failed to install new SIGSEGV handler");
    }
#if XE_PLATFORM_MAC
    // macOS reports memory protection violations as SIGBUS, so it must be
    // intercepted too for guest access-violation / write-watch handling.
    if (sigaction(SIGBUS, &signal_handler, &original_sigbus_handler_) != 0) {
      assert_always("Failed to install new SIGBUS handler");
    }
#endif  // XE_PLATFORM_MAC
    signal_handlers_installed_ = true;
  }

  for (size_t i = 0; i < xe::countof(handlers_); ++i) {
    if (!handlers_[i].first) {
      handlers_[i].first = fn;
      handlers_[i].second = data;
      return;
    }
  }
  assert_always("Too many exception handlers installed");
}

void ExceptionHandler::Uninstall(Handler fn, void* data) {
  for (size_t i = 0; i < xe::countof(handlers_); ++i) {
    if (handlers_[i].first == fn && handlers_[i].second == data) {
      for (; i < xe::countof(handlers_) - 1; ++i) {
        handlers_[i] = handlers_[i + 1];
      }
      handlers_[i].first = nullptr;
      handlers_[i].second = nullptr;
      break;
    }
  }

  bool has_any = false;
  for (size_t i = 0; i < xe::countof(handlers_); ++i) {
    if (handlers_[i].first) {
      has_any = true;
      break;
    }
  }
  if (!has_any) {
    if (signal_handlers_installed_) {
      if (sigaction(SIGILL, &original_sigill_handler_, NULL) != 0) {
        assert_always("Failed to restore original SIGILL handler");
      }
      if (sigaction(SIGSEGV, &original_sigsegv_handler_, NULL) != 0) {
        assert_always("Failed to restore original SIGSEGV handler");
      }
#if XE_PLATFORM_MAC
      if (sigaction(SIGBUS, &original_sigbus_handler_, NULL) != 0) {
        assert_always("Failed to restore original SIGBUS handler");
      }
#endif  // XE_PLATFORM_MAC
      signal_handlers_installed_ = false;
    }
  }
}

}  // namespace xe
