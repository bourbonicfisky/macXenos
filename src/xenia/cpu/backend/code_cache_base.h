/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_CODE_CACHE_BASE_H_
#define XENIA_CPU_BACKEND_CODE_CACHE_BASE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/assert.h"
#include "xenia/base/clock.h"
#include "xenia/base/literals.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/base/mutex.h"
#include "xenia/base/platform.h"
#include "xenia/cpu/backend/code_cache.h"
#include "xenia/cpu/function.h"

#if XE_PLATFORM_MAC || XE_PLATFORM_LINUX
#include <unistd.h>
#endif

#if XE_PLATFORM_MAC && XE_ARCH_ARM64
#include <pthread.h>
#endif

namespace xe {
namespace cpu {
namespace backend {

#if XE_PLATFORM_MAC && XE_ARCH_ARM64
// Apple Silicon maps the JIT code cache as a single MAP_JIT region that is
// writable XOR executable per thread (never both simultaneously). Toggle
// write-protection off before writing generated code and back on before it is
// executed. Used only on this (platform x arch) cell; the Intel macOS path uses
// RWX MAP_JIT and other platforms use a dual-aliased mapping, so the brackets
// are compiled out everywhere else.
static inline void SetJitWritable() { pthread_jit_write_protect_np(0); }
static inline void SetJitExecutable() { pthread_jit_write_protect_np(1); }
#endif  // XE_PLATFORM_MAC && XE_ARCH_ARM64

struct EmitFunctionInfo {
  struct _code_size {
    size_t prolog;
    size_t body;
    size_t epilog;
    size_t tail;
    size_t total;
  } code_size;
  size_t prolog_stack_alloc_offset;
  size_t stack_size;
#if XE_ARCH_ARM64
  // What kind of frame this is.  The POSIX DWARF generator must describe each
  // frame from ground truth, NOT infer it from stack_size: THUNK_STACK_SIZE
  // (224) is also a perfectly ordinary guest-function frame size (80-byte base
  // + 144 bytes of locals), and a guest function saves LR at a different offset
  // than the HostToGuest thunk.  Misclassifying one as the other yields a wrong
  // return-address rule and breaks C++ exception unwinding (the fiber-reentry
  // crash).  Default kGuest; thunk emitters set their own kind.
  enum class FrameKind : uint8_t {
    kGuest = 0,         // ordinary JIT guest function (generic LR-only CFI)
    kHostToGuestThunk,  // saves all callee-saved regs (x19-x28, fp/lr, q8-q15)
    kGuestToHostThunk,
    kResolveThunk,
  };
  FrameKind frame_kind = FrameKind::kGuest;
  // Offset from SP where x30 (LR) is saved.  ARM64 callees save LR
  // explicitly at varying offsets; the unwind info generator needs this
  // to tell the unwinder where to find the return address.
  // Currently only used by the POSIX DWARF .eh_frame generator; the
  // Windows .xdata format encodes LR saves differently.  Adds 8 bytes
  // to the struct on ARM64 Windows builds where it is unused, to avoid
  // #if clutter in the backend/emitter code that sets it.
  size_t lr_save_offset;
#endif
};

// CRTP base class for JIT code caches. Contains all platform-independent
// logic for memory management, indirection tables, code placement, and
// function lookup. Derived classes provide architecture-specific hooks:
//
//   void FillCode(void* address, size_t size)
//     Fill unused code regions with trap instructions (0xCC / BRK).
//
//   void FlushCodeRange(void* address, size_t size)
//     Flush I-cache after writing code (no-op on x86, required on ARM64).
//
//   UnwindReservation RequestUnwindReservation(uint8_t* entry_address)
//     Reserve space for platform-specific unwind info.
//
//   void PlaceCode(uint32_t guest_address, void* machine_code,
//                  const EmitFunctionInfo& func_info,
//                  void* code_execute_address,
//                  UnwindReservation unwind_reservation)
//     Register unwind info and perform platform-specific post-placement.
//
//   void OnCodePlaced(uint32_t guest_address, GuestFunction* function_info,
//                     void* code_execute_address, size_t code_size)
//     Optional hook called after code is placed outside the critical section
//     (used for VTune integration on x64). Default is no-op.
template <typename Derived>
class CodeCacheBase : public CodeCache {
 public:
  ~CodeCacheBase() override {
    if (indirection_table_base_) {
      xe::memory::DeallocFixed(indirection_table_base_, kIndirectionTableSize,
                               xe::memory::DeallocationType::kRelease);
    }
#if XE_PLATFORM_MAC
    // macOS reserves the generated-code region as a private anonymous MAP_JIT
    // mapping via AllocFixed (see Initialize), not a shm file view, so mapping_
    // is never set here and the file-view teardown below is dead. Release it
    // with DeallocFixed. Without this the fixed kGeneratedCodeExecuteBase
    // window leaks: a second CodeCache in the same process (e.g. sequential
    // TestFunctions in xenia-cpu-tests) can't re-reserve it — AllocFixed has no
    // MAP_FIXED_NOREPLACE on macOS, lands elsewhere, and returns null, so the
    // backend is left uninitialized (EXC_BAD_ACCESS in TestModule at the
    // TestFunction #1->#2 boundary). write_base == execute_base on macOS.
    if (generated_code_execute_base_) {
      xe::memory::DeallocFixed(generated_code_execute_base_, kGeneratedCodeSize,
                               xe::memory::DeallocationType::kRelease);
      generated_code_execute_base_ = nullptr;
      generated_code_write_base_ = nullptr;
    }
#endif  // XE_PLATFORM_MAC
    if (mapping_ != xe::memory::kFileMappingHandleInvalid) {
      if (generated_code_write_base_ &&
          generated_code_write_base_ != generated_code_execute_base_) {
        xe::memory::UnmapFileView(mapping_, generated_code_write_base_,
                                  kGeneratedCodeSize);
      }
      if (generated_code_execute_base_) {
        xe::memory::UnmapFileView(mapping_, generated_code_execute_base_,
                                  kGeneratedCodeSize);
      }
      xe::memory::CloseFileMappingHandle(mapping_, file_name_);
      mapping_ = xe::memory::kFileMappingHandleInvalid;
    }
  }

  const std::filesystem::path& file_name() const override { return file_name_; }
  uintptr_t execute_base_address() const override {
    // The real base, which on macOS arm64 is relocated above 4 GB. Equals
    // kGeneratedCodeExecuteBase on platforms that keep the cache sub-4 GB.
    // LookupFunction depends on this matching where code actually lives.
    return reinterpret_cast<uintptr_t>(generated_code_execute_base_);
  }
  size_t total_size() const override { return kGeneratedCodeSize; }

  // Indirection-table slot: holds a host code address. On Apple Silicon the
  // code cache lives above 4 GB (macOS' default __PAGEZERO forbids sub-4 GB
  // mappings), so a slot must hold a full 64-bit host pointer. Everywhere else
  // the cache stays sub-4 GB and slots stay 32-bit: the table is mapped *at*
  // the guest-address range, so the guest address itself is the slot pointer
  // (see the Call/CallIndirect dispatch in the emitters). A universal 8-byte
  // slot is intentionally NOT used: on x64 it would push the table (still based
  // at kIndirectionTableBase) into the code region and break that aliasing.
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
  using IndirectionSlot = uint64_t;
#else
  using IndirectionSlot = uint32_t;
#endif

  // Host address of the slot for a guest address. Guest addresses are 4-byte
  // aligned (PPC), so the slot index is the guest word offset; this matches the
  // legacy uint32 layout exactly (byte offset == guest offset) and, for 8-byte
  // slots, doubles the span.
  IndirectionSlot* indirection_slot_for(uint32_t guest_address) {
    assert_true((guest_address & 0x3u) == 0u);
    return reinterpret_cast<IndirectionSlot*>(indirection_table_base_) +
           ((guest_address - kIndirectionTableBase) >> 2);
  }

  bool has_indirection_table() { return indirection_table_base_ != nullptr; }

  // Real host base of the indirection table (relocated high on macOS arm64).
  // The emitter bakes slot addresses from this at emit time — the table is
  // allocated once in Initialize() before any code is generated.
  uint8_t* indirection_table_base() const { return indirection_table_base_; }

  void set_indirection_default(uintptr_t default_value) {
    indirection_default_value_ = static_cast<IndirectionSlot>(default_value);
  }

  void AddIndirection(uint32_t guest_address, uintptr_t host_address) {
    if (!indirection_table_base_) {
      return;
    }
    *indirection_slot_for(guest_address) =
        static_cast<IndirectionSlot>(host_address);
  }

  void CommitExecutableRange(uint32_t guest_low, uint32_t guest_high) {
    if (!indirection_table_base_) {
      return;
    }
    uint8_t* slot_low = reinterpret_cast<uint8_t*>(indirection_slot_for(guest_low));
    uint8_t* slot_high =
        reinterpret_cast<uint8_t*>(indirection_slot_for(guest_high));
    xe::memory::AllocFixed(
        slot_low, static_cast<size_t>(slot_high - slot_low),
        xe::memory::AllocationType::kCommit, xe::memory::PageAccess::kReadWrite);
    for (uint32_t address = guest_low; address < guest_high; address += 4) {
      *indirection_slot_for(address) = indirection_default_value_;
    }
  }

  void PlaceHostCode(uint32_t guest_address, void* machine_code,
                     const EmitFunctionInfo& func_info,
                     void*& code_execute_address_out,
                     void*& code_write_address_out) {
    PlaceGuestCode(guest_address, machine_code, func_info, nullptr,
                   code_execute_address_out, code_write_address_out);
  }

  void PlaceGuestCode(uint32_t guest_address, void* machine_code,
                      const EmitFunctionInfo& func_info,
                      GuestFunction* function_info,
                      void*& code_execute_address_out,
                      void*& code_write_address_out) {
    using namespace xe::literals;
    uint8_t* code_execute_address;
    {
      auto global_lock = global_critical_region_.Acquire();

      code_execute_address =
          generated_code_execute_base_ + generated_code_offset_;
      code_execute_address_out = code_execute_address;
      uint8_t* code_write_address =
          generated_code_write_base_ + generated_code_offset_;
      code_write_address_out = code_write_address;
      generated_code_offset_ += xe::round_up(func_info.code_size.total, 16);

      auto tail_write_address =
          generated_code_write_base_ + generated_code_offset_;

      auto unwind_reservation = self().RequestUnwindReservation(
          generated_code_write_base_ + generated_code_offset_);
      generated_code_offset_ += xe::round_up(unwind_reservation.data_size, 16);

      auto end_write_address =
          generated_code_write_base_ + generated_code_offset_;

      size_t high_mark = generated_code_offset_;
      CheckCapacity(high_mark);

      generated_code_map_.emplace_back(
          (uint64_t(code_execute_address - generated_code_execute_base_)
           << 32) |
              generated_code_offset_,
          function_info);

      // Commit memory if needed.
      EnsureCommitted(high_mark);

#if XE_PLATFORM_MAC && XE_ARCH_ARM64
      // Make the MAP_JIT region writable for this thread before any store into
      // it. Re-enabled after PlaceCode below, which can still write DWARF
      // unwind (.eh_frame) bytes into the reserved tail.
      SetJitWritable();
#endif

      // Copy code.
      std::memcpy(code_write_address, machine_code, func_info.code_size.total);

      // Fill unused tail/unwind gap with arch-specific trap instructions.
      self().FillCode(
          tail_write_address,
          static_cast<size_t>(end_write_address - tail_write_address));

      // Flush I-cache for code and fill regions. (Cache maintenance works in
      // either W^X state.)
      self().FlushCodeRange(code_write_address, func_info.code_size.total);
      if (tail_write_address < end_write_address) {
        self().FlushCodeRange(
            tail_write_address,
            static_cast<size_t>(end_write_address - tail_write_address));
      }

      // Platform-specific unwind registration (may write into the reserved tail).
      self().PlaceCode(guest_address, machine_code, func_info,
                       code_execute_address, unwind_reservation);

#if XE_PLATFORM_MAC && XE_ARCH_ARM64
      // All writes to the generated-code region are done; make it executable
      // again for this thread before it can be run.
      SetJitExecutable();
#endif
    }

    // Post-placement hook (e.g. VTune notification).
    self().OnCodePlaced(guest_address, function_info, code_execute_address,
                        func_info.code_size.total);

#if XE_PLATFORM_MAC || XE_PLATFORM_LINUX
    // XE_JIT_PERF_MAP=1 (diagnostic): append "<host start> <size> guest_<addr>"
    // to /tmp/perf-<pid>.map, the perf-map format, so host sample addresses in
    // generated code resolve to guest functions.
    static FILE* perf_map = [] {
      if (!std::getenv("XE_JIT_PERF_MAP")) {
        return static_cast<FILE*>(nullptr);
      }
      return std::fopen(fmt::format("/tmp/perf-{}.map", ::getpid()).c_str(),
                        "w");
    }();
    if (perf_map) {
      std::fprintf(perf_map, "%llx %zx guest_%08X\n",
                   static_cast<unsigned long long>(
                       reinterpret_cast<uintptr_t>(code_execute_address)),
                   size_t(func_info.code_size.total), guest_address);
      std::fflush(perf_map);
    }
#endif

    // Fix up indirection table.
    if (guest_address && indirection_table_base_) {
      *indirection_slot_for(guest_address) = static_cast<IndirectionSlot>(
          reinterpret_cast<uintptr_t>(code_execute_address));
    }
  }

  uint32_t PlaceData(const void* data, size_t length) {
    size_t high_mark;
    uint8_t* data_address = nullptr;
    {
      auto global_lock = global_critical_region_.Acquire();
      data_address = generated_code_write_base_ + generated_code_offset_;
      generated_code_offset_ += xe::round_up(length, 16);
      high_mark = generated_code_offset_;
    }
    CheckCapacity(high_mark);
    EnsureCommitted(high_mark);
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
    SetJitWritable();
    std::memcpy(data_address, data, length);
    SetJitExecutable();
#else
    std::memcpy(data_address, data, length);
#endif
    return uint32_t(uintptr_t(data_address));
  }

  GuestFunction* LookupFunction(uint64_t host_pc) override {
    if (generated_code_map_.empty()) {
      return nullptr;
    }
    const uint64_t code_base = execute_base_address();
    const uint64_t code_end = code_base + total_size();
    if (host_pc < code_base || host_pc >= code_end) {
      return nullptr;
    }
    uint32_t key = uint32_t(host_pc - code_base);
    void* fn_entry = std::bsearch(
        &key, generated_code_map_.data(), generated_code_map_.size(),
        sizeof(std::pair<uint32_t, Function*>),
        [](const void* key_ptr, const void* element_ptr) {
          auto key = *reinterpret_cast<const uint32_t*>(key_ptr);
          auto element =
              reinterpret_cast<const std::pair<uint64_t, GuestFunction*>*>(
                  element_ptr);
          if (key < (element->first >> 32)) {
            return -1;
          } else if (key > uint32_t(element->first)) {
            return 1;
          } else {
            return 0;
          }
        });
    if (fn_entry) {
      return reinterpret_cast<const std::pair<uint64_t, GuestFunction*>*>(
                 fn_entry)
          ->second;
    } else {
      return nullptr;
    }
  }

 protected:
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
  // 8-byte slots double the table span (committed sparsely, macOS arm64 only).
  static constexpr size_t kIndirectionTableSize = 0x3FFFFFFE;
#else
  static constexpr size_t kIndirectionTableSize = 0x1FFFFFFF;
#endif
  static constexpr uintptr_t kIndirectionTableBase = 0x80000000;
  // Encoded slots and x64 rel32 need offsets below 2GB, fast-path slots need
  // the cache below 4GB.
  static constexpr size_t kGeneratedCodeSize = 0x3FFFFFFF;
  static constexpr uintptr_t kGeneratedCodeExecuteBase = 0xA0000000;
  static_assert(kGeneratedCodeSize < 0x80000000);
  static_assert(kGeneratedCodeExecuteBase + kGeneratedCodeSize <
                0x100000000ull);
  static constexpr size_t kMaximumFunctionCount = 1000000;

  struct UnwindReservation {
    size_t data_size = 0;
    size_t table_slot = 0;
    uint8_t* entry_address = 0;
  };

  CodeCacheBase() = default;

  bool Initialize() {
#if XE_PLATFORM_MAC && XE_ARCH_ARM64
    // Apple Silicon forbids sub-4 GB mappings (default __PAGEZERO), so the table
    // cannot live at kIndirectionTableBase. Let the kernel place it high
    // (nullptr base) and store the real base; the emitter reads it at emit time,
    // and slots hold full 64-bit host pointers (see IndirectionSlot). kReserve
    // only; CommitExecutableRange commits the touched ranges.
    indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
        nullptr, kIndirectionTableSize, xe::memory::AllocationType::kReserve,
        xe::memory::PageAccess::kReadWrite));
#else
    indirection_table_base_ = reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
        reinterpret_cast<void*>(kIndirectionTableBase), kIndirectionTableSize,
        xe::memory::AllocationType::kReserve,
        xe::memory::PageAccess::kReadWrite));
#endif
    if (!indirection_table_base_) {
      XELOGE("Unable to allocate code cache indirection table");
      XELOGE(
          "This is likely because the {:X}-{:X} range is in use by some "
          "other system DLL",
          static_cast<uint64_t>(kIndirectionTableBase),
          kIndirectionTableBase + kIndirectionTableSize);
    }

#if XE_PLATFORM_MAC
    // macOS cannot execute from POSIX shared memory: an shm-backed MAP_SHARED
    // view caps at a maximum protection of rw- (verified via vmmap on the
    // 0xA0000000 execute view), even with the allow-unsigned-executable-memory
    // entitlement. So the dual-aliased-view (RX/RW) shm code cache used on
    // Linux/Windows can never be executed here — the guest faults with
    // EXC_BAD_ACCESS code=2 on the very first generated instruction. Instead
    // use a single private anonymous MAP_JIT region (AllocFixed adds MAP_JIT
    // for executable anon mappings on macOS) with write_base == execute_base.
    // This requires the app to be ad-hoc codesigned with
    // com.apple.security.cs.allow-jit / allow-unsigned-executable-memory (see
    // src/xenia/app/xenia.entitlements + the POST_BUILD codesign step).
#if XE_ARCH_ARM64
    // Apple Silicon: place the MAP_JIT region high (nullptr base) so it is not
    // sub-4 GB. Indirection slots are 64-bit (IndirectionSlot) so they can point
    // here; execute_base_address() returns this real base for LookupFunction.
    void* code_base = nullptr;
#else
    // Intel macOS keeps the fixed sub-4 GB window (still relies on the shrunk
    // __PAGEZERO; the x64 backend's 32-bit indirection slots require it).
    void* code_base = reinterpret_cast<void*>(kGeneratedCodeExecuteBase);
#endif
    generated_code_execute_base_ =
        reinterpret_cast<uint8_t*>(xe::memory::AllocFixed(
            code_base, kGeneratedCodeSize,
            xe::memory::AllocationType::kReserveCommit,
            xe::memory::PageAccess::kExecuteReadWrite));
    generated_code_write_base_ = generated_code_execute_base_;
    if (!generated_code_execute_base_) {
      XELOGE("Unable to allocate MAP_JIT code cache at {:X}",
             uint64_t(kGeneratedCodeExecuteBase));
      return false;
    }
#else
    file_name_ =
        fmt::format("xenia_code_cache_{}", Clock::QueryHostTickCount());
    mapping_ = xe::memory::CreateFileMappingHandle(
        file_name_, kGeneratedCodeSize,
        xe::memory::PageAccess::kExecuteReadWrite, false);
    if (mapping_ == xe::memory::kFileMappingHandleInvalid) {
      XELOGE("Unable to create code cache mmap");
      return false;
    }

    if (xe::memory::IsWritableExecutableMemoryPreferred()) {
      generated_code_execute_base_ =
          reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
              mapping_, reinterpret_cast<void*>(kGeneratedCodeExecuteBase),
              kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadWrite,
              0));
      generated_code_write_base_ = generated_code_execute_base_;
      if (!generated_code_execute_base_ || !generated_code_write_base_) {
        XELOGE("Unable to allocate code cache generated code storage");
        XELOGE(
            "This is likely because the {:X}-{:X} range is in use by some "
            "other system DLL",
            uint64_t(kGeneratedCodeExecuteBase),
            uint64_t(kGeneratedCodeExecuteBase + kGeneratedCodeSize));
        return false;
      }
    } else {
      generated_code_execute_base_ =
          reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
              mapping_, reinterpret_cast<void*>(kGeneratedCodeExecuteBase),
              kGeneratedCodeSize, xe::memory::PageAccess::kExecuteReadOnly, 0));
      // The write view goes wherever the OS puts it: a fixed address past
      // the execute view would collide with guest memory at this size.
      generated_code_write_base_ = reinterpret_cast<uint8_t*>(
          xe::memory::MapFileView(mapping_, nullptr, kGeneratedCodeSize,
                                  xe::memory::PageAccess::kReadWrite, 0));
      if (!generated_code_execute_base_ || !generated_code_write_base_) {
        XELOGE("Unable to allocate code cache generated code storage");
        XELOGE(
            "This is likely because the {:X}-{:X} range is in use by some "
            "other system DLL",
            uint64_t(kGeneratedCodeExecuteBase),
            uint64_t(kGeneratedCodeExecuteBase + kGeneratedCodeSize));
        return false;
      }
    }
#endif  // XE_PLATFORM_MAC

    generated_code_map_.reserve(kMaximumFunctionCount);
    return true;
  }

  // Default no-op for the OnCodePlaced hook.
  void OnCodePlaced(uint32_t guest_address, GuestFunction* function_info,
                    void* code_execute_address, size_t code_size) {}

  std::filesystem::path file_name_;
  xe::memory::FileMappingHandle mapping_ =
      xe::memory::kFileMappingHandleInvalid;
  xe::global_critical_region global_critical_region_;
  IndirectionSlot indirection_default_value_ = 0xFEEDF00D;
  uint8_t* indirection_table_base_ = nullptr;
  uint8_t* generated_code_execute_base_ = nullptr;
  uint8_t* generated_code_write_base_ = nullptr;
  size_t generated_code_offset_ = 0;
  std::atomic<size_t> generated_code_commit_mark_ = {0};
  std::vector<std::pair<uint64_t, GuestFunction*>> generated_code_map_;

 private:
  Derived& self() { return static_cast<Derived&>(*this); }

  static void CheckCapacity(size_t high_mark) {
    if (high_mark > kGeneratedCodeSize) {
      xe::FatalError(fmt::format(
          "JIT code cache is full ({} MiB). Please report this game to "
          "macXenos developers.",
          (kGeneratedCodeSize + 1) >> 20));
    }
  }

  void EnsureCommitted(size_t high_mark) {
    using namespace xe::literals;
    size_t old_commit_mark, new_commit_mark;
    do {
      old_commit_mark = generated_code_commit_mark_;
      if (high_mark <= old_commit_mark) {
        break;
      }
      new_commit_mark = old_commit_mark + 16_MiB;
      if (generated_code_execute_base_ == generated_code_write_base_) {
        xe::memory::AllocFixed(generated_code_execute_base_, new_commit_mark,
                               xe::memory::AllocationType::kCommit,
                               xe::memory::PageAccess::kExecuteReadWrite);
      } else {
        xe::memory::AllocFixed(generated_code_execute_base_, new_commit_mark,
                               xe::memory::AllocationType::kCommit,
                               xe::memory::PageAccess::kExecuteReadOnly);
        xe::memory::AllocFixed(generated_code_write_base_, new_commit_mark,
                               xe::memory::AllocationType::kCommit,
                               xe::memory::PageAccess::kReadWrite);
      }
    } while (generated_code_commit_mark_.compare_exchange_weak(
        old_commit_mark, new_commit_mark));
  }
};

}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_CODE_CACHE_BASE_H_
