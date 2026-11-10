/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/memory.h"

#include <cstring>
#include <random>

#include "xenia/base/platform.h"
#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <cerrno>
#endif

#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/assert.h"
#include "xenia/base/byte_stream.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/threading.h"

#include "xenia/cpu/mmio_handler.h"

// TODO(benvanik): move xbox.h out
#include "xenia/xbox.h"

DEFINE_bool(protect_zero, true, "Protect the zero page from reads and writes.",
            "Memory");
DEFINE_bool(emit_inline_mmio_checks, false,
            "Emit inline MMIO range checks for all I32 loads/stores instead "
            "of relying on exception-based MMIO detection.",
            "CPU");
DEFINE_bool(emit_mmio_aware_stores_for_recorded_exception_addresses, true,
            "Uses info gathered via record_mmio_access_exceptions to emit "
            "special stores that are faster than trapping the exception",
            "CPU");
DEFINE_bool(record_mmio_access_exceptions, true,
            "For guest addresses records whether we caught any mmio accesses "
            "for them. This info can then be used on a subsequent run to "
            "instruct the recompiler to emit checks",
            "CPU");
DEFINE_bool(protect_on_release, false,
            "Protect released memory to prevent accesses.", "Memory");
DEFINE_bool(scribble_heap, false,
            "Scribble specific or random value into all allocated heap memory.",
            "Memory");
DEFINE_int32(scribble_heap_value, 0,
             "Value used to fill all allocated heap memory. 0 - Random value. "
             "Valid range: [1-255]",
             "Memory");
DEFINE_bool(exact_write_fault_invalidation, true,
            "Invalidate only the faulting page on a guest write to watched "
            "memory instead of the surrounding 256 KB block, so per-frame "
            "dynamic writes stop invalidating neighbouring static textures "
            "and vertex data (GTA IV: ~230 needless texture revalidations "
            "per frame).",
            "Memory");

namespace xe {
uint32_t get_page_count(uint32_t value, uint32_t page_size) {
  return xe::round_up(value, page_size) / page_size;
}

/**
 * Memory map:
 * 0x00000000 - 0x3FFFFFFF (1024mb) - virtual 4k pages
 * 0x40000000 - 0x7FFFFFFF (1024mb) - virtual 64k pages
 * 0x80000000 - 0x8BFFFFFF ( 192mb) - xex 64k pages
 * 0x8C000000 - 0x8FFFFFFF (  64mb) - xex 64k pages (encrypted)
 * 0x90000000 - 0x9FFFFFFF ( 256mb) - xex 4k pages
 * 0xA0000000 - 0xBFFFFFFF ( 512mb) - physical 64k pages
 * 0xC0000000 - 0xDFFFFFFF          - physical 16mb pages
 * 0xE0000000 - 0xFFFFFFFF          - physical 4k pages
 *
 * We use the host OS to create an entire addressable range for this. That way
 * we don't have to emulate a TLB. It'd be really cool to pass through page
 * sizes or use madvice to let the OS know what to expect.
 *
 * We create our own heap of committed memory that lives at
 * memory_HEAP_LOW to memory_HEAP_HIGH - all normal user allocations
 * come from there. Since the Xbox has no paging, we know that the size of
 * this heap will never need to be larger than ~512MB (realistically, smaller
 * than that). We place it far away from the XEX data and keep the memory
 * around it uncommitted so that we have some warning if things go astray.
 *
 * For XEX/GPU/etc data we allow placement allocations (base_address != 0) and
 * commit the requested memory as needed. This bypasses the standard heap, but
 * XEXs should never be overwriting anything so that's fine. We can also query
 * for previous commits and assert that we really isn't committing twice.
 *
 * GPU memory is mapped onto the lower 512mb of the virtual 4k range (0).
 * So 0xA0000000 = 0x00000000. A more sophisticated allocator could handle
 * this.
 */

static Memory* active_memory_ = nullptr;

void CrashDump() {
  static std::atomic<int> in_crash_dump(0);
  if (in_crash_dump.fetch_add(1)) {
    xe::FatalError(
        "Hard crash: the memory system crashed while dumping a crash dump.");
    return;
  }
  active_memory_->DumpMap();
  --in_crash_dump;
}

static inline bool ShouldSkipHostCommit(const BaseHeap& heap) {
  // Host commit/protect uses mprotect, which fails with EINVAL when the range
  // is not host-page-aligned. On hosts with a >4 KB page (16 KB macOS ARM64,
  // 64 KB some ARM64 Linux), a heap whose guest page size is SMALLER than the
  // host page (the 4 KB heaps) can't be committed/protected at guest
  // granularity, so we skip the host op and rely on the shared RW mapping.
  //
  // Heaps whose guest page size is >= the host page (e.g. the 64 KB heaps) are
  // host-page-aligned and MUST perform the real commit: a freshly committed
  // page is host PROT_NONE until mprotect makes it RW, and skipping leaves it
  // inaccessible so the guest faults writing to it (page table says RW, host
  // says no-access).
  return xe::memory::page_size() > heap.page_size();
}

xe::memory::PageAccess ToPageAccess(uint32_t protect) {
  // Write-combine memory is CPU-writable (for GPU uploads)
  bool is_writable =
      (protect & kMemoryProtectWrite) || (protect & kMemoryProtectWriteCombine);

  if ((protect & kMemoryProtectRead) && !is_writable) {
    return xe::memory::PageAccess::kReadOnly;
  } else if ((protect & kMemoryProtectRead) && is_writable) {
    return xe::memory::PageAccess::kReadWrite;
  } else {
    return xe::memory::PageAccess::kNoAccess;
  }
}

void RandomizeMemory(void* range_start, uint32_t size) {
  if (!cvars::scribble_heap) {
    return;
  }

  if (!cvars::scribble_heap_value) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, std::numeric_limits<uint8_t>::max());

    std::generate(static_cast<char*>(range_start),
                  static_cast<char*>(range_start) + size,
                  [&]() { return dis(gen); });
  } else {
    std::memset(range_start, cvars::scribble_heap_value, size);
  }
}

Memory::Memory() {
  system_page_size_ = uint32_t(xe::memory::page_size());
  system_allocation_granularity_ =
      uint32_t(xe::memory::allocation_granularity());
  assert_zero(active_memory_);
  active_memory_ = this;
}

Memory::~Memory() {
  assert_true(active_memory_ == this);
  active_memory_ = nullptr;

  // Uninstall the MMIO handler, as we won't be able to service more
  // requests.
  mmio_handler_.reset();

  for (auto invalidation_callback : physical_memory_invalidation_callbacks_) {
    delete invalidation_callback;
  }

  heaps_.v00000000.Dispose();
  heaps_.v40000000.Dispose();
  heaps_.v80000000.Dispose();
  heaps_.v90000000.Dispose();
  heaps_.vA0000000.Dispose();
  heaps_.vC0000000.Dispose();
  heaps_.vE0000000.Dispose();
  heaps_.physical.Dispose();

  // Unmap all views and close mapping.
  if (mapping_ != xe::memory::kFileMappingHandleInvalid) {
    UnmapViews();
    xe::memory::CloseFileMappingHandle(mapping_, file_name_);
    mapping_base_ = nullptr;
    mapping_ = xe::memory::kFileMappingHandleInvalid;
  }

  virtual_membase_ = nullptr;
  physical_membase_ = nullptr;
}

bool Memory::Initialize() {
  file_name_ = fmt::format("xenia_memory_{}", Clock::QueryHostTickCount());

  // Create main page file-backed mapping. This is all reserved but
  // uncommitted (so it shouldn't expand page file).
#if XE_PLATFORM_MAC
  // entire 4gb space + 512mb physical, plus page-alignment slack: macOS rejects
  // MAP_SHARED views that run past the shm object, and the 4K-offset physical
  // view (offset 0x100001000, size 0x20000000) extends one page beyond
  // 0x11FFFFFFF, so size the object up to cover it (canary/macos).
  const size_t mapping_size = xe::round_up(
      0x120000000ull + system_allocation_granularity_,
      system_allocation_granularity_);
#else
  // entire 4gb space + 512mb physical:
  const size_t mapping_size = 0x11FFFFFFF;
#endif
  mapping_ = xe::memory::CreateFileMappingHandle(
      file_name_, mapping_size, xe::memory::PageAccess::kReadWrite, false);
  if (mapping_ == xe::memory::kFileMappingHandleInvalid) {
    XELOGE("Unable to reserve the 4gb guest address space.");
    assert_always();
    return false;
  }

#if XE_PLATFORM_MAC
  // macOS does not honor a fixed address hint for a fresh mmap, and rejects
  // MAP_SHARED views that run past the shm object. Reserve one OS-chosen
  // contiguous block, then MAP_FIXED each file-backed view into it (canary/macos
  // MapViewsMac), keeping the guest layout identical to other platforms without
  // needing a fixed low base.
  if (MapViewsMac()) {
    XELOGE("Unable to find a continuous block in the 64bit address space.");
    assert_always();
    return false;
  }
  mapping_base_ = views_.all_views[0];
#else
  // Attempt to create our views. This may fail at the first address
  // we pick, so try a few times.
  mapping_base_ = 0;
  for (size_t n = 32; n < 64; n++) {
    auto mapping_base = reinterpret_cast<uint8_t*>(1ull << n);
    if (!MapViews(mapping_base)) {
      mapping_base_ = mapping_base;
      break;
    }
  }
  if (!mapping_base_) {
    XELOGE("Unable to find a continuous block in the 64bit address space.");
    assert_always();
    return false;
  }
#endif
  virtual_membase_ = mapping_base_;
  physical_membase_ = mapping_base_ + 0x100000000ull;

  // Prepare virtual heaps.
  heaps_.v00000000.Initialize(this, virtual_membase_, HeapType::kGuestVirtual,
                              0x00000000, 0x40000000, 4096);
  heaps_.v40000000.Initialize(this, virtual_membase_, HeapType::kGuestVirtual,
                              0x40000000, 0x40000000 - 0x01000000, 64 * 1024);
  heaps_.v80000000.Initialize(this, virtual_membase_, HeapType::kGuestXex,
                              0x80000000, 0x10000000, 64 * 1024);
  heaps_.v90000000.Initialize(this, virtual_membase_, HeapType::kGuestXex,
                              0x90000000, 0x10000000, 4096);

  // Prepare physical heaps.
  heaps_.physical.Initialize(this, physical_membase_, HeapType::kGuestPhysical,
                             0x00000000, 0x20000000, 4096);
  heaps_.vA0000000.Initialize(this, virtual_membase_, HeapType::kGuestPhysical,
                              0xA0000000, 0x20000000, 64 * 1024,
                              &heaps_.physical);
  heaps_.vC0000000.Initialize(this, virtual_membase_, HeapType::kGuestPhysical,
                              0xC0000000, 0x20000000, 16 * 1024 * 1024,
                              &heaps_.physical);
  heaps_.vE0000000.Initialize(this, virtual_membase_, HeapType::kGuestPhysical,
                              0xE0000000, 0x1FD00000, 4096, &heaps_.physical);

  // Protect the first and last 64kb of memory.
  heaps_.v00000000.AllocFixed(
      0x00000000, 0x10000, 0x10000,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      !cvars::protect_zero ? kMemoryProtectRead | kMemoryProtectWrite
                           : kMemoryProtectNoAccess);
  heaps_.physical.AllocFixed(0x1FFF0000, 0x10000, 0x10000,
                             kMemoryAllocationReserve, kMemoryProtectNoAccess);

  // GPU writeback.
  // 0xC... is physical, 0x7F... is virtual. We may need to overlay these.
  heaps_.vC0000000.AllocFixed(
      0xC0000000, 0x01000000, 32,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite);

  // TODO(Gliniak): Seems like GPU has access to whole physical memory range
  // without any restriction. This however needs some form of validation.
  // That's why we're commiting whole physical memory range and deal with
  // allocations issues on custom page protection level.
  for (size_t i = 1; i <= 16; i++) {
    xe::memory::AllocFixed(heaps_.physical.TranslateRelative(i << 24),
                           heaps_.physical.page_size() * 0x10000,
                           xe::memory::AllocationType::kCommit,
                           xe::memory::PageAccess::kReadWrite);
  }

  // Add handlers for MMIO.
  mmio_handler_ = cpu::MMIOHandler::Install(
      virtual_membase_, physical_membase_, physical_membase_ + 0x1FFFFFFF,
      HostToGuestVirtualThunk, this, AccessViolationCallbackThunk, this,
      nullptr, nullptr);
  if (!mmio_handler_) {
    XELOGE("Unable to install MMIO handlers");
    assert_always();
    return false;
  }

  // ?
  uint32_t unk_phys_alloc;
  heaps_.vA0000000.Alloc(0x340000, 64 * 1024, kMemoryAllocationReserve,
                         kMemoryProtectNoAccess, true, &unk_phys_alloc);

  // Allocate memory hypervisor and kernel would use
  uint32_t hypervisor_range;
  heaps_.v80000000.Alloc(0x40000, 4 * 1024, kMemoryAllocationCommit,
                         kMemoryProtectRead | kMemoryProtectWrite, false,
                         &hypervisor_range);

  // Value taken from 544307D5. Title explicitly access this address and this is
  // a value underneath it (It's constant between multiple runs)
  uint32_t value_to_write = xe::byte_swap(0x2a6e3f38);
  memcpy(TranslateVirtual(0x80000000 + 0x1C), &value_to_write,
         sizeof(uint32_t));

  // Host membase bases (ASLR-probed, so they differ every launch): print them
  // once so a guest address can be turned into a host watch address for an LLDB
  // hardware read watchpoint -- host_va = virtual_membase + guest_ea. The PPC
  // JIT reads guest RAM through the VIRTUAL mirror, so watch that one, not the
  // physical mirror. Backend-agnostic (works under Metal and Vulkan alike).
  XELOGI("Memory: virtual_membase={} physical_membase={} (host_va = "
         "virtual_membase + guest_ea; LLDB watch target)",
         static_cast<void*>(virtual_membase_),
         static_cast<void*>(physical_membase_));

  return true;
}

void Memory::SetMMIOExceptionRecordingCallback(
    cpu::MmioAccessRecordCallback callback, void* context) {
  mmio_handler_->SetMMIOExceptionRecordingCallback(callback, context);
}

static const struct {
  uint64_t virtual_address_start;
  uint64_t virtual_address_end;
  uint64_t target_address;
} map_info[] = {
    // (1024mb) - virtual 4k pages
    {
        0x00000000,
        0x3FFFFFFF,
        0x0000000000000000ull,
    },
    // (1024mb) - virtual 64k pages (cont)
    {
        0x40000000,
        0x7EFFFFFF,
        0x0000000040000000ull,
    },
    //   (16mb) - GPU writeback + 15mb of XPS?
    {
        0x7F000000,
        0x7FFFFFFF,
        0x0000000100000000ull,
    },
    //  (256mb) - xex 64k pages
    {
        0x80000000,
        0x8FFFFFFF,
        0x0000000080000000ull,
    },
    //  (256mb) - xex 4k pages
    // Own backing: aliasing the 64k xex range lets a module in one heap
    // overwrite a module in the other.
    {
        0x90000000,
        0x9FFFFFFF,
        0x0000000090000000ull,
    },
    //  (512mb) - physical 64k pages
    {
        0xA0000000,
        0xBFFFFFFF,
        0x0000000100000000ull,
    },
    //          - physical 16mb pages
    {
        0xC0000000,
        0xDFFFFFFF,
        0x0000000100000000ull,
    },
    //          - physical 4k pages
    {
        0xE0000000,
        0xFFFFFFFF,
        0x0000000100001000ull,
    },
    //          - physical raw
    {
        0x100000000,
        0x11FFFFFFF,
        0x0000000100000000ull,
    },
};

#if XE_PLATFORM_MAC
int Memory::MapViewsMac() {
  assert_true(xe::countof(map_info) == xe::countof(views_.all_views));

  // macOS does not guarantee that a non-MAP_FIXED mmap will honor the requested
  // address. Reserve a contiguous address range first, then MAP_FIXED each view
  // within that reserved range to keep the guest layout identical to Windows.
  const size_t total_size =
      map_info[xe::countof(map_info) - 1].virtual_address_end -
      map_info[0].virtual_address_start + 1;

  // The guest block must NOT land in the sub-4 GB code-cache window
  // (0x80000000-0xC0000000), which the JIT later claims at fixed addresses. With
  // __PAGEZERO shrunk to 0x1000, mmap(nullptr) hands back a low base that
  // swallows that window, and macOS ignores address hints without MAP_FIXED. So
  // walk the address space above 4 GB with mach_vm_region to find a verified-free
  // hole, then MAP_FIXED the reservation there.
  mach_vm_address_t probe = 0x200000000ull;  // 8 GB, clear of the code cache
  constexpr mach_vm_address_t kProbeCeiling = 0x100000000000ull;  // 16 TB
  void* reserved_base = MAP_FAILED;
  while (probe < kProbeCeiling) {
    mach_vm_address_t region_addr = probe;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    kern_return_t kr = mach_vm_region(
        mach_task_self(), &region_addr, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &info_count, &object_name);
    if (kr != KERN_SUCCESS || region_addr >= probe + total_size) {
      // Either nothing is mapped at/above probe, or the next mapping starts far
      // enough away to leave a big-enough hole at probe.
      void* p = mmap(reinterpret_cast<void*>(probe), total_size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (p != MAP_FAILED) {
        reserved_base = p;
        break;
      }
      // MAP_FIXED failed despite the hole; step past and keep looking.
      probe += total_size;
      continue;
    }
    // A region occupies part of [probe, probe+total_size); jump past it.
    probe = region_addr + region_size;
    probe = (probe + system_allocation_granularity_ - 1) &
            ~uint64_t(system_allocation_granularity_ - 1);
  }
  if (reserved_base == MAP_FAILED) {
    XELOGE("MapViewsMac: reserve failed: {}", std::strerror(errno));
    return 1;
  }

  uint8_t* mapping_base = reinterpret_cast<uint8_t*>(reserved_base);
  uint64_t granularity_mask = ~uint64_t(system_allocation_granularity_ - 1);

  for (size_t n = 0; n < xe::countof(map_info); n++) {
    size_t view_size =
        map_info[n].virtual_address_end - map_info[n].virtual_address_start + 1;
    size_t file_offset = map_info[n].target_address & granularity_mask;
    void* target_address = mapping_base + map_info[n].virtual_address_start;
    void* result = mmap(target_address, view_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, mapping_, file_offset);
    if (result == MAP_FAILED || result != target_address) {
      int err = errno;
      XELOGE(
          "MapViewsMac: map failed view {} addr 0x{:016X} size 0x{:X} "
          "offset 0x{:X} err {} ({})",
          n, reinterpret_cast<uintptr_t>(target_address), view_size,
          file_offset, err, std::strerror(err));
      munmap(reserved_base, total_size);
      for (auto& view : views_.all_views) {
        view = nullptr;
      }
      return 1;
    }
    views_.all_views[n] = reinterpret_cast<uint8_t*>(result);
  }

  return 0;
}
#endif  // XE_PLATFORM_MAC

int Memory::MapViews(uint8_t* mapping_base) {
  assert_true(xe::countof(map_info) == xe::countof(views_.all_views));
  // 0xE0000000 4 KB offset is emulated via host_address_offset and on the CPU
  // side if system allocation granularity is bigger than 4 KB.
  uint64_t granularity_mask = ~uint64_t(system_allocation_granularity_ - 1);
  for (size_t n = 0; n < xe::countof(map_info); n++) {
    views_.all_views[n] = reinterpret_cast<uint8_t*>(xe::memory::MapFileView(
        mapping_, mapping_base + map_info[n].virtual_address_start,
        map_info[n].virtual_address_end - map_info[n].virtual_address_start + 1,
        xe::memory::PageAccess::kReadWrite,
        map_info[n].target_address & granularity_mask));
    if (!views_.all_views[n]) {
      // Failed, so bail and try again.
      UnmapViews();
      return 1;
    }
  }
  return 0;
}

void Memory::UnmapViews() {
  for (size_t n = 0; n < xe::countof(views_.all_views); n++) {
    if (views_.all_views[n]) {
      size_t length = map_info[n].virtual_address_end -
                      map_info[n].virtual_address_start + 1;
      xe::memory::UnmapFileView(mapping_, views_.all_views[n], length);
    }
  }
}

void Memory::Reset() {
  heaps_.v00000000.Reset();
  heaps_.v40000000.Reset();
  heaps_.v80000000.Reset();
  heaps_.v90000000.Reset();
  heaps_.physical.Reset();
}
// clang does not like non-standard layout offsetof
#if XE_COMPILER_MSVC == 1 && XE_COMPILER_CLANG_CL == 0
XE_NOALIAS
const BaseHeap* Memory::LookupHeap(uint32_t address) const {
#define HEAP_INDEX(name) \
  offsetof(Memory, heaps_.name) - offsetof(Memory, heaps_)

  const char* heap_select = (const char*)&this->heaps_;

  unsigned selected_heap_offset = 0;
  unsigned high_nibble = address >> 28;

  if (high_nibble < 0x4) {
    selected_heap_offset = HEAP_INDEX(v00000000);
  } else if (address < 0x7F000000) {
    selected_heap_offset = HEAP_INDEX(v40000000);
  } else if (high_nibble < 0x8) {
    heap_select = nullptr;
    // return nullptr;
  } else if (high_nibble < 0x9) {
    selected_heap_offset = HEAP_INDEX(v80000000);
    // return &heaps_.v80000000;
  } else if (high_nibble < 0xA) {
    // return &heaps_.v90000000;
    selected_heap_offset = HEAP_INDEX(v90000000);
  } else if (high_nibble < 0xC) {
    // return &heaps_.vA0000000;
    selected_heap_offset = HEAP_INDEX(vA0000000);
  } else if (high_nibble < 0xE) {
    // return &heaps_.vC0000000;
    selected_heap_offset = HEAP_INDEX(vC0000000);
  } else if (address < 0xFFD00000) {
    // return &heaps_.vE0000000;
    selected_heap_offset = HEAP_INDEX(vE0000000);
  } else {
    //  return nullptr;
    heap_select = nullptr;
  }
  return reinterpret_cast<const BaseHeap*>(selected_heap_offset + heap_select);
}
#else
XE_NOALIAS
const BaseHeap* Memory::LookupHeap(uint32_t address) const {
  if (address < 0x40000000) {
    return &heaps_.v00000000;
  } else if (address < 0x7F000000) {
    return &heaps_.v40000000;
  } else if (address < 0x80000000) {
    return nullptr;
  } else if (address < 0x90000000) {
    return &heaps_.v80000000;
  } else if (address < 0xA0000000) {
    return &heaps_.v90000000;
  } else if (address < 0xC0000000) {
    return &heaps_.vA0000000;
  } else if (address < 0xE0000000) {
    return &heaps_.vC0000000;
  } else if (address < 0xFFD00000) {
    return &heaps_.vE0000000;
  } else {
    return nullptr;
  }
}
#endif
BaseHeap* Memory::LookupHeapByType(bool physical, uint32_t page_size) {
  if (physical) {
    if (page_size <= 4096) {
      return &heaps_.vE0000000;
    } else if (page_size <= 64 * 1024) {
      return &heaps_.vA0000000;
    } else {
      return &heaps_.vC0000000;
    }
  } else {
    if (page_size <= 4096) {
      return &heaps_.v00000000;
    } else {
      return &heaps_.v40000000;
    }
  }
}

VirtualHeap* Memory::GetPhysicalHeap() { return &heaps_.physical; }

void Memory::GetHeapsPageStatsSummary(const BaseHeap* const* provided_heaps,
                                      size_t heaps_count,
                                      uint32_t& unreserved_pages,
                                      uint32_t& reserved_pages,
                                      uint32_t& used_pages,
                                      uint32_t& reserved_bytes) {
  auto lock = global_critical_region_.Acquire();
  for (size_t i = 0; i < heaps_count; i++) {
    const BaseHeap* heap = provided_heaps[i];
    uint32_t heap_unreserved_pages = heap->unreserved_page_count();
    uint32_t heap_reserved_pages = heap->reserved_page_count();

    unreserved_pages += heap_unreserved_pages;
    reserved_pages += heap_reserved_pages;
    used_pages += ((heap->total_page_count() - heap_unreserved_pages) *
                   heap->page_size()) /
                  4096;
    reserved_bytes += heap_reserved_pages * heap->page_size();
  }
}

uint32_t Memory::HostToGuestVirtual(const void* host_address) const {
  size_t virtual_address = reinterpret_cast<size_t>(host_address) -
                           reinterpret_cast<size_t>(virtual_membase_);
  uint32_t vE0000000_host_offset = heaps_.vE0000000.host_address_offset();
  size_t vE0000000_host_base =
      size_t(heaps_.vE0000000.heap_base()) + vE0000000_host_offset;
  if (virtual_address >= vE0000000_host_base &&
      virtual_address <=
          (vE0000000_host_base + (heaps_.vE0000000.heap_size() - 1))) {
    virtual_address -= vE0000000_host_offset;
  }
  return uint32_t(virtual_address);
}

uint32_t Memory::HostToGuestVirtualThunk(const void* context,
                                         const void* host_address) {
  return reinterpret_cast<const Memory*>(context)->HostToGuestVirtual(
      host_address);
}

uint32_t Memory::GetPhysicalAddress(uint32_t address) const {
  const BaseHeap* heap = LookupHeap(address);
  if (!heap) {
    return UINT32_MAX;
  }

  // Assumption that we already received physical address, so just return it.
  if (heap->heap_type() != HeapType::kGuestPhysical && address < 0x1FFFFFFF) {
    return address;
  }

  return static_cast<const PhysicalHeap*>(heap)->GetPhysicalAddress(address);
}

void Memory::Zero(uint32_t address, uint32_t size) {
  std::memset(TranslateVirtual(address), 0, size);
}

void Memory::Fill(uint32_t address, uint32_t size, uint8_t value) {
  std::memset(TranslateVirtual(address), value, size);
}

void Memory::Copy(uint32_t dest, uint32_t src, uint32_t size) {
  uint8_t* pdest = TranslateVirtual(dest);
  const uint8_t* psrc = TranslateVirtual(src);
  std::memcpy(pdest, psrc, size);
}

uint32_t Memory::SearchAligned(uint32_t start, uint32_t end,
                               const uint32_t* values, size_t value_count) {
  assert_true(start <= end);
  auto p = TranslateVirtual<const uint32_t*>(start);
  auto pe = TranslateVirtual<const uint32_t*>(end);
  while (p != pe) {
    if (*p == values[0]) {
      const uint32_t* pc = p + 1;
      size_t matched = 1;
      for (size_t n = 1; n < value_count; n++, pc++) {
        if (*pc != values[n]) {
          break;
        }
        matched++;
      }
      if (matched == value_count) {
        return HostToGuestVirtual(p);
      }
    }
    p++;
  }
  return 0;
}

bool Memory::AddVirtualMappedRange(uint32_t virtual_address, uint32_t mask,
                                   uint32_t size, void* context,
                                   cpu::MMIOReadCallback read_callback,
                                   cpu::MMIOWriteCallback write_callback) {
  if (!cvars::emit_inline_mmio_checks) {
    if (!xe::memory::AllocFixed(TranslateVirtual(virtual_address), size,
                                xe::memory::AllocationType::kCommit,
                                xe::memory::PageAccess::kNoAccess)) {
      XELOGE("Unable to map range; commit/protect failed");
      return false;
    }
  }
  return mmio_handler_->RegisterRange(virtual_address, mask, size, context,
                                      read_callback, write_callback);
}

cpu::MMIORange* Memory::LookupVirtualMappedRange(uint32_t virtual_address) {
  return mmio_handler_->LookupRange(virtual_address);
}

bool Memory::AccessViolationCallback(
    global_unique_lock_type global_lock_locked_once, void* host_address,
    bool is_write) {
  // Access via physical_membase_ is special, when need to bypass everything
  // (for instance, for a data provider to actually write the data) so only
  // triggering callbacks on virtual memory regions.
  if (reinterpret_cast<size_t>(host_address) <
          reinterpret_cast<size_t>(virtual_membase_) ||
      reinterpret_cast<size_t>(host_address) >=
          reinterpret_cast<size_t>(physical_membase_)) {
    return false;
  }
  uint32_t virtual_address = HostToGuestVirtual(host_address);
  BaseHeap* heap = LookupHeap(virtual_address);
  if (heap->heap_type() != HeapType::kGuestPhysical) {
    return false;
  }

  // Access violation callbacks from the guest are triggered when the global
  // critical region mutex is locked once.
  //
  // Will be rounded to physical page boundaries internally, so just pass 1 as
  // the length - guranteed not to cross page boundaries also.
  auto physical_heap = static_cast<PhysicalHeap*>(heap);
  physical_heap->ProvideReadWatchedPage(global_lock_locked_once,
                                        virtual_address, is_write);
  // The provide above waited for the page's GPU writers without the global
  // lock; the read callbacks this pass repeats under the lock must not wait
  // again, or every other global-lock user (the GPU thread's per-draw range
  // requests) stalls behind the GPU (GTA IV: ~3 ms/frame on the M4).
  return physical_heap->TriggerCallbacks(
      std::move(global_lock_locked_once), virtual_address, 1, is_write,
      cvars::exact_write_fault_invalidation, true, false, false, false);
}

bool Memory::AccessViolationCallbackThunk(
    global_unique_lock_type global_lock_locked_once, void* context,
    void* host_address, bool is_write) {
  return reinterpret_cast<Memory*>(context)->AccessViolationCallback(
      std::move(global_lock_locked_once), host_address, is_write);
}

bool Memory::TriggerPhysicalMemoryCallbacks(
    global_unique_lock_type global_lock_locked_once, uint32_t virtual_address,
    uint32_t length, bool is_write, bool unwatch_exact_range, bool unprotect) {
  BaseHeap* heap = LookupHeap(virtual_address);
  if (heap->heap_type() == HeapType::kGuestPhysical) {
    auto physical_heap = static_cast<PhysicalHeap*>(heap);
    return physical_heap->TriggerCallbacks(std::move(global_lock_locked_once),
                                           virtual_address, length, is_write,
                                           unwatch_exact_range, unprotect);
  }
  return false;
}

void* Memory::RegisterPhysicalMemoryInvalidationCallback(
    PhysicalMemoryInvalidationCallback callback, void* callback_context) {
  auto entry = new std::pair<PhysicalMemoryInvalidationCallback, void*>(
      callback, callback_context);
  auto lock = global_critical_region_.Acquire();
  physical_memory_invalidation_callbacks_.push_back(entry);
  return entry;
}

void Memory::UnregisterPhysicalMemoryInvalidationCallback(
    void* callback_handle) {
  auto entry =
      reinterpret_cast<std::pair<PhysicalMemoryInvalidationCallback, void*>*>(
          callback_handle);
  {
    auto lock = global_critical_region_.Acquire();
    auto it = std::find(physical_memory_invalidation_callbacks_.begin(),
                        physical_memory_invalidation_callbacks_.end(), entry);
    assert_true(it != physical_memory_invalidation_callbacks_.end());
    if (it != physical_memory_invalidation_callbacks_.end()) {
      physical_memory_invalidation_callbacks_.erase(it);
    }
  }
  delete entry;
}

void* Memory::RegisterPhysicalMemoryReadCallback(
    PhysicalMemoryReadCallback callback, void* callback_context) {
  auto entry = new std::pair<PhysicalMemoryReadCallback, void*>(
      callback, callback_context);
  auto lock = global_critical_region_.Acquire();
  physical_memory_read_callbacks_.push_back(entry);
  return entry;
}

void Memory::UnregisterPhysicalMemoryReadCallback(void* callback_handle) {
  auto entry = reinterpret_cast<std::pair<PhysicalMemoryReadCallback, void*>*>(
      callback_handle);
  {
    auto lock = global_critical_region_.Acquire();
    auto it = std::find(physical_memory_read_callbacks_.begin(),
                        physical_memory_read_callbacks_.end(), entry);
    assert_true(it != physical_memory_read_callbacks_.end());
    if (it != physical_memory_read_callbacks_.end()) {
      physical_memory_read_callbacks_.erase(it);
    }
  }
  // Wait out callbacks still running without the global lock.
  std::unique_lock<std::shared_mutex> calls_lock(
      physical_memory_read_callback_calls_mutex_);
  delete entry;
}

global_unique_lock_type Memory::LockPhysicalMemoryInvalidationCallbacks() {
  // The SAME region PhysicalHeap::TriggerCallbacks holds while it dispatches,
  // and the same one Register/Unregister above take: one lock, so a producer
  // that acquires it here is ordered against every callback, not merely against
  // the list.
  return global_critical_region_.Acquire();
}

std::string Memory::DescribePhysicalWatch(uint32_t physical_address) {
  auto global_lock = global_critical_region_.Acquire();
  return fmt::format("A0[{}] C0[{}] E0[{}]",
                     heaps_.vA0000000.DescribeAccessState(physical_address),
                     heaps_.vC0000000.DescribeAccessState(physical_address),
                     heaps_.vE0000000.DescribeAccessState(physical_address));
}

void Memory::EnablePhysicalMemoryAccessCallbacks(
    uint32_t physical_address, uint32_t length,
    bool enable_invalidation_notifications, bool enable_data_providers) {
  heaps_.vA0000000.EnableAccessCallbacks(physical_address, length,
                                         enable_invalidation_notifications,
                                         enable_data_providers);
  heaps_.vC0000000.EnableAccessCallbacks(physical_address, length,
                                         enable_invalidation_notifications,
                                         enable_data_providers);
  heaps_.vE0000000.EnableAccessCallbacks(physical_address, length,
                                         enable_invalidation_notifications,
                                         enable_data_providers);
}

uint32_t Memory::SystemHeapAlloc(uint32_t size, uint32_t alignment,
                                 uint32_t system_heap_flags) {
  // TODO(benvanik): lightweight pool.
  bool is_physical = !!(system_heap_flags & kSystemHeapPhysical);
  auto heap = LookupHeapByType(is_physical, 4096);
  uint32_t address;
  if (!heap->AllocSystemHeap(
          size, alignment, kMemoryAllocationReserve | kMemoryAllocationCommit,
          kMemoryProtectRead | kMemoryProtectWrite, false, &address)) {
    return 0;
  }
  Zero(address, size);
  return address;
}

void Memory::SystemHeapFree(uint32_t address, uint32_t* out_region_size) {
  if (!address) {
    return;
  }
  // TODO(benvanik): lightweight pool.
  auto heap = LookupHeap(address);
  heap->Release(address, out_region_size);
}

void Memory::DumpMap() {
  XELOGE("==================================================================");
  XELOGE("Memory Dump");
  XELOGE("==================================================================");
  XELOGE("               System Page Size: {0} ({0:08X})", system_page_size_);
  XELOGE("  System Allocation Granularity: {0} ({0:08X})",
         system_allocation_granularity_);
  XELOGE("                Virtual Membase: {}",
         static_cast<void*>(virtual_membase_));
  XELOGE("               Physical Membase: {}",
         static_cast<void*>(physical_membase_));
  XELOGE("");
  XELOGE("------------------------------------------------------------------");
  XELOGE("Virtual Heaps");
  XELOGE("------------------------------------------------------------------");
  XELOGE("");
  heaps_.v00000000.DumpMap();
  heaps_.v40000000.DumpMap();
  heaps_.v80000000.DumpMap();
  heaps_.v90000000.DumpMap();
  XELOGE("");
  XELOGE("------------------------------------------------------------------");
  XELOGE("Physical Heaps");
  XELOGE("------------------------------------------------------------------");
  XELOGE("");
  heaps_.physical.DumpMap();
  heaps_.vA0000000.DumpMap();
  heaps_.vC0000000.DumpMap();
  heaps_.vE0000000.DumpMap();
  XELOGE("");
}

bool Memory::Save(ByteStream* stream) {
  XELOGD("Serializing memory...");
  heaps_.v00000000.Save(stream);
  heaps_.v40000000.Save(stream);
  heaps_.v80000000.Save(stream);
  heaps_.v90000000.Save(stream);
  heaps_.physical.Save(stream);

  return true;
}

bool Memory::Restore(ByteStream* stream) {
  XELOGD("Restoring memory...");
  heaps_.v00000000.Restore(stream);
  heaps_.v40000000.Restore(stream);
  heaps_.v80000000.Restore(stream);
  heaps_.v90000000.Restore(stream);
  heaps_.physical.Restore(stream);

  return true;
}

uint32_t FromPageAccess(xe::memory::PageAccess protect) {
  switch (protect) {
    case memory::PageAccess::kNoAccess:
      return kMemoryProtectNoAccess;
    case memory::PageAccess::kReadOnly:
      return kMemoryProtectRead;
    case memory::PageAccess::kReadWrite:
      return kMemoryProtectRead | kMemoryProtectWrite;
    case memory::PageAccess::kExecuteReadOnly:
      // Guest memory cannot be executable - this should never happen :)
      assert_always();
      return kMemoryProtectRead;
    case memory::PageAccess::kExecuteReadWrite:
      // Guest memory cannot be executable - this should never happen :)
      assert_always();
      return kMemoryProtectRead | kMemoryProtectWrite;
  }

  return kMemoryProtectNoAccess;
}

BaseHeap::BaseHeap()
    : membase_(nullptr), heap_base_(0), heap_size_(0), page_size_(0) {}

BaseHeap::~BaseHeap() = default;

void BaseHeap::Initialize(Memory* memory, uint8_t* membase, HeapType heap_type,
                          uint32_t heap_base, uint32_t heap_size,
                          uint32_t page_size, uint32_t host_address_offset) {
  memory_ = memory;
  membase_ = membase;
  heap_type_ = heap_type;
  heap_base_ = heap_base;
  heap_size_ = heap_size;
  page_size_ = page_size;
  xenia_assert(xe::is_pow2(page_size_));
  page_size_shift_ = xe::log2_floor(page_size_);
  host_address_offset_ = host_address_offset;
  page_table_.resize(heap_size / page_size);
  unreserved_page_count_ = uint32_t(page_table_.size());

  // Initialize free block tracker with a single block covering the entire heap.
  free_blocks_.clear();
  free_blocks_[0] = uint32_t(page_table_.size());
}

void BaseHeap::Dispose() {
  // Walk table and release all regions.
  for (uint32_t page_number = 0; page_number < page_table_.size();
       ++page_number) {
    auto& page_entry = page_table_[page_number];
    if (page_entry.state) {
      xe::memory::DeallocFixed(TranslateRelative(page_number * page_size_), 0,
                               xe::memory::DeallocationType::kRelease);
      page_number += page_entry.region_page_count;
    }
  }
  free_blocks_.clear();
}

void BaseHeap::DumpMap() {
  auto global_lock = global_critical_region_.Acquire();
  XELOGE("------------------------------------------------------------------");
  XELOGE("Heap: {:08X}-{:08X}", heap_base_, heap_base_ + (heap_size_ - 1));
  XELOGE("------------------------------------------------------------------");
  XELOGE("            Heap Base: {:08X}", heap_base_);
  XELOGE("            Heap Size: {0} ({0:08X})", heap_size_);
  XELOGE("            Page Size: {0} ({0:08X})", page_size_);
  XELOGE("           Page Count: {}", page_table_.size());
  XELOGE("  Host Address Offset: {0} ({0:08X})", host_address_offset_);
  bool is_empty_span = false;
  uint32_t empty_span_start = 0;
  for (uint32_t i = 0; i < uint32_t(page_table_.size()); ++i) {
    auto& page = page_table_[i];
    if (!page.state) {
      if (!is_empty_span) {
        is_empty_span = true;
        empty_span_start = i;
      }
      continue;
    }
    if (is_empty_span) {
      XELOGE("  {:08X}-{:08X} {:6d}p {:10d}b unreserved",
             heap_base_ + empty_span_start * page_size_,
             heap_base_ + i * page_size_, i - empty_span_start,
             (i - empty_span_start) * page_size_);
      is_empty_span = false;
    }
    const char* state_name = "   ";
    if (page.state & kMemoryAllocationCommit) {
      state_name = "COM";
    } else if (page.state & kMemoryAllocationReserve) {
      state_name = "RES";
    }
    char access_r = (page.current_protect & kMemoryProtectRead) ? 'R' : ' ';
    char access_w = (page.current_protect & kMemoryProtectWrite) ? 'W' : ' ';
    XELOGE("  {:08X}-{:08X} {:6d}p {:10d}b {} {}{}",
           heap_base_ + i * page_size_,
           heap_base_ + (i + page.region_page_count) * page_size_,
           page.region_page_count, page.region_page_count * page_size_,
           state_name, access_r, access_w);
    i += page.region_page_count - 1;
  }
  if (is_empty_span) {
    XELOGE("  {:08X}-{:08X} - {} unreserved pages)",
           heap_base_ + empty_span_start * page_size_,
           heap_base_ + (heap_size_ - 1),
           page_table_.size() - empty_span_start);
  }
}

bool BaseHeap::Save(ByteStream* stream) {
  XELOGD("Heap {:08X}-{:08X}", heap_base_, heap_base_ + (heap_size_ - 1));

  for (size_t i = 0; i < page_table_.size(); i++) {
    auto& page = page_table_[i];
    stream->Write(page.qword);
    if (!page.state) {
      // Unallocated.
      continue;
    }

    // TODO(DrChat): write compressed with snappy.
    if (page.state & kMemoryAllocationCommit) {
      void* addr = TranslateRelative(i * page_size_);

      // See BaseHeap::Restore: on >4 KB host pages a 4 KB-granular mprotect
      // fails with EINVAL and the mapping is already RW, so skip the host
      // protect dance and read straight from the mapping.
      if (ShouldSkipHostCommit(*this)) {
        stream->Write(addr, page_size_);
      } else {
        memory::PageAccess old_access;
        memory::Protect(addr, page_size_, memory::PageAccess::kReadWrite,
                        &old_access);

        stream->Write(addr, page_size_);

        memory::Protect(addr, page_size_, old_access, nullptr);
      }
    }
  }

  return true;
}

bool BaseHeap::Restore(ByteStream* stream) {
  XELOGD("Heap {:08X}-{:08X}", heap_base_, heap_base_ + (heap_size_ - 1));

  for (size_t i = 0; i < page_table_.size(); i++) {
    auto& page = page_table_[i];
    page.qword = stream->Read<uint64_t>();
    if (!page.state) {
      // Unallocated.
      continue;
    }

    memory::PageAccess page_access = memory::PageAccess::kNoAccess;
    if ((page.current_protect & kMemoryProtectRead) &&
        (page.current_protect & kMemoryProtectWrite)) {
      page_access = memory::PageAccess::kReadWrite;
    } else if (page.current_protect & kMemoryProtectRead) {
      page_access = memory::PageAccess::kReadOnly;
    }

    // On hosts whose page size exceeds the 4 KB guest page (16 KB on macOS
    // ARM64, 64 KB on some ARM64 Linux), host commit/mprotect on a 4 KB
    // boundary fails with EINVAL, and the heap is already mapped RW via the
    // shared file mapping (the software page_table_ is authoritative). Skip the
    // host-level ops in that case, exactly as the normal commit/protect path
    // does (see ShouldSkipHostCommit), and read straight into the mapping.
    const bool skip_host_ops = ShouldSkipHostCommit(*this);

    // Commit the memory if it isn't already. We do not need to reserve any
    // memory, as the mapping has already taken care of that.
    if ((page.state & kMemoryAllocationCommit) && !skip_host_ops) {
      xe::memory::AllocFixed(TranslateRelative(i * page_size_), page_size_,
                             memory::AllocationType::kCommit,
                             memory::PageAccess::kReadWrite);
    }

    // Now read into memory. We'll set R/W protection first, then set the
    // protection back to its previous state.
    // TODO(DrChat): read compressed with snappy.
    if (page.state & kMemoryAllocationCommit) {
      void* addr = TranslateRelative(i * page_size_);
      if (!skip_host_ops) {
        xe::memory::Protect(addr, page_size_, memory::PageAccess::kReadWrite,
                            nullptr);
      }

      stream->Read(addr, page_size_);

      if (!skip_host_ops) {
        xe::memory::Protect(addr, page_size_, page_access, nullptr);
      }
    }
  }

  RebuildFreeBlocks();

  return true;
}

void BaseHeap::RebuildFreeBlocks() {
  free_blocks_.clear();
  uint32_t run_start = UINT32_MAX;
  for (uint32_t i = 0; i < uint32_t(page_table_.size()); ++i) {
    if (page_table_[i].state == 0) {
      if (run_start == UINT32_MAX) {
        run_start = i;
      }
    } else {
      if (run_start != UINT32_MAX) {
        free_blocks_[run_start] = i - run_start;
        run_start = UINT32_MAX;
      }
    }
  }
  if (run_start != UINT32_MAX) {
    free_blocks_[run_start] = uint32_t(page_table_.size()) - run_start;
  }
}

void BaseHeap::RemoveFreeBlock(uint32_t start_page, uint32_t page_count) {
  if (free_blocks_.empty()) {
    return;
  }

  // Find the free block that contains the allocated range.
  auto it = free_blocks_.upper_bound(start_page);
  if (it != free_blocks_.begin()) {
    --it;
  }

  // Verify the block actually contains our range.
  uint32_t block_start = it->first;
  uint32_t block_count = it->second;
  uint32_t block_end = block_start + block_count;
  assert_true(start_page >= block_start &&
              start_page + page_count <= block_end);

  free_blocks_.erase(it);

  // Insert remnant before the allocated range.
  if (block_start < start_page) {
    free_blocks_[block_start] = start_page - block_start;
  }

  // Insert remnant after the allocated range.
  uint32_t alloc_end = start_page + page_count;
  if (alloc_end < block_end) {
    free_blocks_[alloc_end] = block_end - alloc_end;
  }
}

void BaseHeap::InsertFreeBlock(uint32_t start_page, uint32_t page_count) {
  uint32_t new_start = start_page;
  uint32_t new_count = page_count;

  // Try to merge with block immediately after.
  auto it_after = free_blocks_.find(start_page + page_count);
  if (it_after != free_blocks_.end()) {
    new_count += it_after->second;
    free_blocks_.erase(it_after);
  }

  // Try to merge with block immediately before.
  auto it_at = free_blocks_.lower_bound(start_page);
  if (it_at != free_blocks_.begin()) {
    auto it_before = std::prev(it_at);
    if (it_before->first + it_before->second == start_page) {
      new_start = it_before->first;
      new_count += it_before->second;
      free_blocks_.erase(it_before);
    }
  }

  free_blocks_[new_start] = new_count;
}

void BaseHeap::Reset() {
  // TODO(DrChat): protect pages.
  std::memset(page_table_.data(), 0, sizeof(PageEntry) * page_table_.size());
  unreserved_page_count_ = uint32_t(page_table_.size());
  // TODO(Triang3l): Remove access callbacks from pages if this is a physical
  // memory heap.

  // Re-initialize free block tracker.
  free_blocks_.clear();
  free_blocks_[0] = uint32_t(page_table_.size());
}

bool BaseHeap::Alloc(uint32_t size, uint32_t alignment,
                     uint32_t allocation_type, uint32_t protect, bool top_down,
                     uint32_t* out_address) {
  *out_address = 0;
  size = xe::round_up(size, page_size_);
  alignment = xe::round_up(alignment, page_size_);

  // Exclude the top 240MB of the v40000000 heap (64KB guest pages) from
  // general allocation to protect the thread stack region
  // (0x70000000-0x7F000000)
  uint32_t heap_virtual_guest_offset = 0;
  if (heap_type_ == HeapType::kGuestVirtual && page_size_ == 0x10000) {
    heap_virtual_guest_offset = 0x0F000000;
  }

  uint32_t low_address = heap_base_;
  uint32_t high_address =
      heap_base_ + (heap_size_ - 1) - heap_virtual_guest_offset;
  return AllocRange(low_address, high_address, size, alignment, allocation_type,
                    protect, top_down, out_address);
}

bool BaseHeap::AllocFixed(uint32_t base_address, uint32_t size,
                          uint32_t alignment, uint32_t allocation_type,
                          uint32_t protect) {
  alignment = xe::round_up(alignment, page_size_);
  size = xe::align(size, alignment);
  assert_true((base_address + host_address_offset_) % alignment == 0);
  uint32_t page_count = get_page_count(size, page_size_);
  uint32_t start_page_number = (base_address - heap_base_) / page_size_;
  uint32_t end_page_number = start_page_number + page_count - 1;
  if (start_page_number >= page_table_.size() ||
      end_page_number > page_table_.size()) {
    XELOGE("BaseHeap::AllocFixed passed out of range address range");
    return false;
  }

  auto global_lock = global_critical_region_.Acquire();

  // - If we are reserving, the entire range must not be already reserved.
  // - If we are committing it's ok for pages within the range to already be
  //   committed.
  const bool is_pure_reserve = allocation_type == kMemoryAllocationReserve;
  for (uint32_t page_number = start_page_number; page_number <= end_page_number;
       ++page_number) {
    uint32_t state = page_table_[page_number].state;
    if ((allocation_type == kMemoryAllocationReserve) && state) {
      // Already reserved.
      XELOGE(
          "BaseHeap::AllocFixed attempting to reserve an already reserved "
          "range");
      return false;
    }
    if ((allocation_type == kMemoryAllocationCommit) &&
        !(state & kMemoryAllocationReserve)) {
      // Attempting a commit-only op on an unreserved page.
      // This may be OK.
      XELOGW("BaseHeap::AllocFixed attempting commit on unreserved page");
      allocation_type |= kMemoryAllocationReserve;
      break;
    }
  }

  // Allocate from host.
  if (allocation_type == kMemoryAllocationReserve) {
    // Reserve is not needed, as we are mapped already.
  } else {
    if (!ShouldSkipHostCommit(*this)) {
      auto alloc_type = (allocation_type & kMemoryAllocationCommit)
                            ? xe::memory::AllocationType::kCommit
                            : xe::memory::AllocationType::kReserve;
      void* result = xe::memory::AllocFixed(
          TranslateRelative(start_page_number * page_size_),
          page_count * page_size_, alloc_type, ToPageAccess(protect));
      if (!result) {
        XELOGE("BaseHeap::AllocFixed failed to alloc range from host");
        return false;
      }

      if (cvars::scribble_heap && protect & kMemoryProtectWrite) {
        RandomizeMemory(result, page_count * page_size_);
      }
    } else if (cvars::scribble_heap && protect & kMemoryProtectWrite) {
      RandomizeMemory(TranslateRelative(start_page_number * page_size_),
                      page_count * page_size_);
    }
  }

  // Set page state.
  bool had_free_pages = false;
  for (uint32_t page_number = start_page_number; page_number <= end_page_number;
       ++page_number) {
    auto& page_entry = page_table_[page_number];
    if (allocation_type & kMemoryAllocationReserve) {
      // Region is based on reservation.
      page_entry.base_address = start_page_number;
      page_entry.region_page_count = page_count;
    }
    page_entry.allocation_protect = protect;
    page_entry.current_protect = protect;
    if (!(page_entry.state & kMemoryAllocationReserve)) {
      had_free_pages = true;
      unreserved_page_count_--;
    }
    page_entry.state = kMemoryAllocationReserve | allocation_type;
  }

  // Update free block tracker if any pages transitioned from free.
  if (had_free_pages) {
    if (is_pure_reserve) {
      // Pure reserve: validation confirmed all pages were free, so the range
      // is within a single coalesced free block.
      RemoveFreeBlock(start_page_number, page_count);
    } else {
      // Mixed state (commit upgraded to reserve+commit): pages may span
      // multiple free blocks, rebuild from page_table_.
      RebuildFreeBlocks();
    }
  }

  return true;
}
template <typename T>
static inline T QuickMod(T value, uint32_t modv) {
  if (xe::is_pow2(modv)) {
    return value & (modv - 1);
  } else {
    return value % modv;
  }
}

bool BaseHeap::AllocRange(uint32_t low_address, uint32_t high_address,
                          uint32_t size, uint32_t alignment,
                          uint32_t allocation_type, uint32_t protect,
                          bool top_down, uint32_t* out_address) {
  *out_address = 0;

  alignment = xe::round_up(alignment, page_size_);
  uint32_t page_count = get_page_count(size, page_size_);
  low_address = std::max(heap_base_, xe::align(low_address, alignment));
  high_address = std::min(heap_base_ + (heap_size_ - 1),
                          xe::align(high_address, alignment));

  uint32_t low_page_number = (low_address - heap_base_) >> page_size_shift_;
  uint32_t high_page_number = (high_address - heap_base_) >> page_size_shift_;
  low_page_number = std::min(uint32_t(page_table_.size()) - 1, low_page_number);
  high_page_number =
      std::min(uint32_t(page_table_.size()) - 1, high_page_number);

  if (page_count > (high_page_number - low_page_number)) {
    XELOGE("BaseHeap::Alloc page count too big for requested range");
    return false;
  }

  auto global_lock = global_critical_region_.Acquire();

  // Find a free page range using the free block tracker.
  // The base page must match the requested alignment.
  uint32_t start_page_number = UINT_MAX;
  uint32_t end_page_number = UINT_MAX;
  uint32_t page_scan_stride = alignment >> page_size_shift_;

  if (top_down) {
    // Search free blocks from high addresses downward.
    // Find the first block that could overlap our range.
    auto it = free_blocks_.upper_bound(high_page_number);
    while (it != free_blocks_.begin()) {
      --it;
      uint32_t block_start = it->first;
      uint32_t block_count = it->second;
      uint32_t block_end = block_start + block_count;

      // Block is entirely below our search range — stop.
      if (block_end <= low_page_number) {
        break;
      }

      // Skip blocks too small to possibly fit.
      if (block_count < page_count) {
        continue;
      }

      // Compute the highest aligned start within this block and range.
      // high_page_number is exclusive and rounded down to the stride, so
      // the top stride of pages is never returned.
      uint32_t high_aligned =
          high_page_number - QuickMod(high_page_number, page_scan_stride);
      uint32_t usable_end = std::min(block_end, high_aligned);
      if (usable_end < page_count) {
        continue;
      }
      uint32_t latest_start = usable_end - page_count;
      // Align down to stride.
      latest_start -= QuickMod(latest_start, page_scan_stride);
      uint32_t usable_start = std::max(block_start, low_page_number);
      if (latest_start >= usable_start &&
          latest_start + page_count <= block_end) {
        start_page_number = latest_start;
        end_page_number = latest_start + page_count - 1;
        break;
      }
    }
  } else {
    // Search free blocks from low addresses upward.
    auto it = free_blocks_.lower_bound(low_page_number);
    // Check if the previous block extends into our range.
    if (it != free_blocks_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second > low_page_number) {
        it = prev;
      }
    }
    for (; it != free_blocks_.end(); ++it) {
      uint32_t block_start = it->first;
      uint32_t block_count = it->second;
      uint32_t block_end = block_start + block_count;

      // Block is entirely above our search range — stop.
      if (block_start > high_page_number) {
        break;
      }

      // Skip blocks too small to possibly fit.
      if (block_count < page_count) {
        continue;
      }

      // Compute the lowest aligned start within this block and range.
      // high_page_number is treated as exclusive — the page at
      // high_page_number itself is never returned.
      uint32_t earliest = std::max(block_start, low_page_number);
      uint32_t aligned_start = xe::round_up(earliest, page_scan_stride, false);
      if (aligned_start + page_count <= block_end &&
          aligned_start + page_count <= high_page_number) {
        start_page_number = aligned_start;
        end_page_number = aligned_start + page_count - 1;
        break;
      }
    }
  }

  if (start_page_number == UINT_MAX || end_page_number == UINT_MAX) {
    // Out of memory.
    XELOGE("BaseHeap::Alloc failed to find contiguous range");
    // assert_always("Heap exhausted!");
    return false;
  }

  // Update free block tracker.
  RemoveFreeBlock(start_page_number, page_count);

  // Allocate from host.
  if (allocation_type == kMemoryAllocationReserve) {
    // Reserve is not needed, as we are mapped already.
  } else {
    if (!ShouldSkipHostCommit(*this)) {
      auto alloc_type = (allocation_type & kMemoryAllocationCommit)
                            ? xe::memory::AllocationType::kCommit
                            : xe::memory::AllocationType::kReserve;
      void* result = xe::memory::AllocFixed(
          TranslateRelative(start_page_number << page_size_shift_),
          page_count << page_size_shift_, alloc_type, ToPageAccess(protect));
      if (!result) {
        XELOGE("BaseHeap::Alloc failed to alloc range from host");
        // Restore the free block since we failed.
        InsertFreeBlock(start_page_number, page_count);
        return false;
      }

      if (cvars::scribble_heap && (protect & kMemoryProtectWrite)) {
        RandomizeMemory(result, page_count << page_size_shift_);
      }
    } else if (cvars::scribble_heap && (protect & kMemoryProtectWrite)) {
      RandomizeMemory(TranslateRelative(start_page_number << page_size_shift_),
                      page_count << page_size_shift_);
    }
  }

  // Set page state.
  for (uint32_t page_number = start_page_number; page_number <= end_page_number;
       ++page_number) {
    auto& page_entry = page_table_[page_number];
    page_entry.base_address = start_page_number;
    page_entry.region_page_count = page_count;
    page_entry.allocation_protect = protect;
    page_entry.current_protect = protect;
    page_entry.state = kMemoryAllocationReserve | allocation_type;
    unreserved_page_count_--;
  }

  *out_address = heap_base_ + (start_page_number << page_size_shift_);
  return true;
}

bool BaseHeap::AllocSystemHeap(uint32_t size, uint32_t alignment,
                               uint32_t allocation_type, uint32_t protect,
                               bool top_down, uint32_t* out_address) {
  *out_address = 0;
  size = xe::round_up(size, page_size_);
  alignment = xe::round_up(alignment, page_size_);

  uint32_t low_address = heap_base_;
  if (heap_type_ == xe::HeapType::kGuestVirtual) {
    // Both virtual heaps are same size, so we can assume that we substract
    // constant value.
    low_address = heap_base_ + heap_size_ - 0x10000000;
  }
  uint32_t high_address = heap_base_ + (heap_size_ - 1);
  return AllocRange(low_address, high_address, size, alignment, allocation_type,
                    protect, top_down, out_address);
}

bool BaseHeap::Decommit(uint32_t address, uint32_t size) {
  uint32_t page_count = get_page_count(size, page_size_);
  uint32_t start_page_number = (address - heap_base_) / page_size_;
  uint32_t end_page_number = start_page_number + page_count - 1;
  start_page_number =
      std::min(uint32_t(page_table_.size()) - 1, start_page_number);
  end_page_number = std::min(uint32_t(page_table_.size()) - 1, end_page_number);

  auto global_lock = global_critical_region_.Acquire();

  // Release from host.
  // TODO(benvanik): find a way to actually decommit memory;
  //     mapped memory cannot be decommitted.
  /*BOOL result =
      VirtualFree(TranslateRelative(start_page_number * page_size_),
                  page_count * page_size_, MEM_DECOMMIT);
  if (!result) {
    PLOGW("BaseHeap::Decommit failed due to host VirtualFree failure");
    return false;
  }*/

  // Perform table change.
  for (uint32_t page_number = start_page_number; page_number <= end_page_number;
       ++page_number) {
    auto& page_entry = page_table_[page_number];
    page_entry.state &= ~kMemoryAllocationCommit;
  }

  return true;
}

bool BaseHeap::Release(uint32_t base_address, uint32_t* out_region_size) {
  auto global_lock = global_critical_region_.Acquire();

  // Given address must be a region base address.
  uint32_t base_page_number = (base_address - heap_base_) / page_size_;
  auto base_page_entry = page_table_[base_page_number];
  if (base_page_entry.base_address != base_page_number) {
    XELOGE("BaseHeap::Release failed because address is not a region start");
    return false;
  }

  if (heap_base_ == 0x00000000 && base_page_number == 0) {
    XELOGE("BaseHeap::Release: Attempt to free 0!");
    return false;
  }

  if (out_region_size) {
    *out_region_size = (base_page_entry.region_page_count * page_size_);
  }

  // Release from host not needed as mapping reserves the range for us.
  // TODO(benvanik): protect with NOACCESS?
  /*BOOL result = VirtualFree(
      TranslateRelative(base_page_number * page_size_), 0, MEM_RELEASE);
  if (!result) {
    PLOGE("BaseHeap::Release failed due to host VirtualFree failure");
    return false;
  }*/
  // Instead, we just protect it, if we can.
  if (page_size_ == xe::memory::page_size() ||
      ((base_page_entry.region_page_count * page_size_) %
               xe::memory::page_size() ==
           0 &&
       ((base_page_number * page_size_) % xe::memory::page_size() == 0))) {
    // TODO(benvanik): figure out why games are using memory after releasing
    // it. It's possible this is some virtual/physical stuff where the GPU
    // still can access it.
    if (cvars::protect_on_release) {
      if (!xe::memory::Protect(TranslateRelative(base_page_number * page_size_),
                               base_page_entry.region_page_count * page_size_,
                               xe::memory::PageAccess::kNoAccess, nullptr)) {
        XELOGW("BaseHeap::Release failed due to host VirtualProtect failure");
      }
    }
  }

  // Perform table change.
  uint32_t end_page_number =
      base_page_number + base_page_entry.region_page_count - 1;
  for (uint32_t page_number = base_page_number; page_number <= end_page_number;
       ++page_number) {
    auto& page_entry = page_table_[page_number];
    page_entry.qword = 0;
    unreserved_page_count_++;
  }

  // Insert freed block into tracker with coalescing.
  InsertFreeBlock(base_page_number, base_page_entry.region_page_count);

  return true;
}

bool BaseHeap::Protect(uint32_t address, uint32_t size, uint32_t protect,
                       uint32_t* old_protect) {
  if (!size) {
    XELOGE("BaseHeap::Protect failed due to zero size");
    return false;
  }

  // From the VirtualProtect MSDN page:
  //
  // "The region of affected pages includes all pages containing one or more
  //  bytes in the range from the lpAddress parameter to (lpAddress+dwSize).
  //  This means that a 2-byte range straddling a page boundary causes the
  //  protection attributes of both pages to be changed."
  //
  // "The access protection value can be set only on committed pages. If the
  //  state of any page in the specified region is not committed, the function
  //  fails and returns without modifying the access protection of any pages in
  //  the specified region."

  uint32_t start_page_number = (address - heap_base_) >> page_size_shift_;
  if (start_page_number >= page_table_.size()) {
    XELOGE("BaseHeap::Protect failed due to out-of-bounds base address {:08X}",
           address);
    return false;
  }
  uint32_t end_page_number =
      uint32_t((uint64_t(address) + size - 1 - heap_base_) >> page_size_shift_);
  if (end_page_number >= page_table_.size()) {
    XELOGE(
        "BaseHeap::Protect failed due to out-of-bounds range ({:08X} bytes "
        "from {:08x})",
        size, address);
    return false;
  }

  auto global_lock = global_critical_region_.Acquire();

  // Ensure all pages are in the same reserved region and all are committed.
  uint32_t first_base_address = UINT_MAX;
  for (uint32_t page_number = start_page_number; page_number <= end_page_number;
       ++page_number) {
    auto page_entry = page_table_[page_number];
    if (first_base_address == UINT_MAX) {
      first_base_address = page_entry.base_address;
    } else if (first_base_address != page_entry.base_address) {
      XELOGE("BaseHeap::Protect failed due to request spanning regions");
      return false;
    }
    if (!(page_entry.state & kMemoryAllocationCommit)) {
      XELOGE("BaseHeap::Protect failed due to uncommitted page");
      return false;
    }
  }
  uint32_t xe_page_size = static_cast<uint32_t>(xe::memory::page_size());

  uint32_t page_size_mask = xe_page_size - 1;

  // Attempt host change (hopefully won't fail).
  // We can only do this if our size matches system page granularity.
  uint32_t page_count = end_page_number - start_page_number + 1;
  bool host_offset_aligned = (host_address_offset_ & page_size_mask) == 0;
  if (page_size_ == xe_page_size ||
      (host_offset_aligned &&
       (((page_count << page_size_shift_) & page_size_mask) == 0) &&
       (((start_page_number << page_size_shift_) & page_size_mask) == 0))) {
    memory::PageAccess old_protect_access;
    if (!xe::memory::Protect(
            TranslateRelative(start_page_number << page_size_shift_),
            page_count << page_size_shift_, ToPageAccess(protect),
            old_protect ? &old_protect_access : nullptr)) {
      XELOGE("BaseHeap::Protect failed due to host VirtualProtect failure");
      return false;
    }

    if (old_protect) {
      *old_protect = FromPageAccess(old_protect_access);
    }
  } else {
    // If the host page size is larger than the guest page size, align
    // protection to host pages and use the most permissive access needed by
    // any guest page in each host page to avoid over-restricting smaller guest
    // pages within a host page.
    if (page_size_ < xe_page_size) {
      uint32_t start_offset =
          host_address_offset_ + (start_page_number << page_size_shift_);
      uint32_t end_offset = host_address_offset_ +
                            ((end_page_number + 1) << page_size_shift_) - 1;

      uint32_t aligned_start_offset = start_offset & ~page_size_mask;
      uint32_t aligned_end_offset = (end_offset | page_size_mask) + 1;

      for (uint32_t host_offset = aligned_start_offset;
           host_offset < aligned_end_offset; host_offset += xe_page_size) {
        uint32_t first_guest_page = 0;
        if (host_offset > host_address_offset_) {
          first_guest_page =
              (host_offset - host_address_offset_) >> page_size_shift_;
        }
        uint32_t host_page_end = host_offset + xe_page_size - 1;
        if (host_page_end < host_address_offset_) {
          continue;
        }
        uint32_t last_guest_page =
            (host_page_end - host_address_offset_) >> page_size_shift_;
        if (last_guest_page >= page_table_.size()) {
          last_guest_page = static_cast<uint32_t>(page_table_.size()) - 1;
        }

        xe::memory::PageAccess host_access = xe::memory::PageAccess::kNoAccess;
        for (uint32_t p = first_guest_page; p <= last_guest_page; ++p) {
          uint32_t page_prot = (p >= start_page_number && p <= end_page_number)
                                   ? protect
                                   : page_table_[p].current_protect;
          xe::memory::PageAccess page_access = ToPageAccess(page_prot);
          if (page_access == xe::memory::PageAccess::kReadWrite) {
            host_access = xe::memory::PageAccess::kReadWrite;
            break;
          }
          if (page_access == xe::memory::PageAccess::kReadOnly &&
              host_access == xe::memory::PageAccess::kNoAccess) {
            host_access = xe::memory::PageAccess::kReadOnly;
          }
        }

        xe::memory::Protect(
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(membase_) +
                                    heap_base_ + host_offset),
            xe_page_size, host_access, nullptr);
      }

      if (old_protect) {
        *old_protect = page_table_[start_page_number].current_protect;
      }
    } else {
      XELOGW(
          "BaseHeap::Protect: unaligned to host page size; skipping mprotect");
      if (old_protect) {
        *old_protect = page_table_[start_page_number].current_protect;
      }
      return false;
    }
  }

  // Perform table change.
  for (uint32_t page_number = start_page_number; page_number <= end_page_number;
       ++page_number) {
    auto& page_entry = page_table_[page_number];
    page_entry.current_protect = protect;
  }

  return true;
}

bool BaseHeap::QueryRegionInfo(uint32_t base_address,
                               HeapAllocationInfo* out_info,
                               uint32_t max_region_size) {
  uint32_t start_page_number = (base_address - heap_base_) >> page_size_shift_;
  if (start_page_number > page_table_.size()) {
    XELOGE("BaseHeap::QueryRegionInfo base page out of range");
    return false;
  }

  auto global_lock = global_critical_region_.Acquire();

  auto start_page_entry = page_table_[start_page_number];
  out_info->base_address = base_address;
  out_info->allocation_base = 0;
  out_info->allocation_protect = 0;
  out_info->region_size = 0;
  out_info->state = 0;
  out_info->protect = 0;
  if (start_page_entry.state) {
    // Committed/reserved region.
    out_info->allocation_base =
        heap_base_ + (start_page_entry.base_address << page_size_shift_);
    out_info->allocation_protect = start_page_entry.allocation_protect;
    out_info->allocation_size = start_page_entry.region_page_count
                                << page_size_shift_;
    out_info->state = start_page_entry.state;
    out_info->protect = start_page_entry.current_protect;

    // Scan forward and report the size of the region matching the initial
    // base address's attributes.
    for (uint32_t page_number = start_page_number;
         page_number <
         start_page_entry.base_address + start_page_entry.region_page_count;
         ++page_number) {
      auto page_entry = page_table_[page_number];
      if (page_entry.base_address != start_page_entry.base_address ||
          page_entry.state != start_page_entry.state ||
          page_entry.current_protect != start_page_entry.current_protect) {
        // Different region or different properties within the region; done.
        break;
      }
      out_info->region_size += page_size_;
      if (out_info->region_size >= max_region_size) {
        break;
      }
    }
  } else {
    // Free region.
    for (uint32_t page_number = start_page_number;
         page_number < page_table_.size(); ++page_number) {
      auto page_entry = page_table_[page_number];
      if (page_entry.state) {
        // First non-free page; done with region.
        break;
      }
      out_info->region_size += page_size_;
      if (out_info->region_size >= max_region_size) {
        break;
      }
    }
  }
  return true;
}

bool BaseHeap::QuerySize(uint32_t address, uint32_t* out_size) {
  uint32_t page_number = (address - heap_base_) >> page_size_shift_;
  if (page_number > page_table_.size()) {
    XELOGE("BaseHeap::QuerySize base page out of range");
    *out_size = 0;
    return false;
  }
  auto global_lock = global_critical_region_.Acquire();
  auto page_entry = page_table_[page_number];
  *out_size = (page_entry.region_page_count << page_size_shift_);
  return true;
}

bool BaseHeap::QueryBaseAndSize(uint32_t* in_out_address, uint32_t* out_size) {
  uint32_t page_number = (*in_out_address - heap_base_) >> page_size_shift_;
  if (page_number > page_table_.size()) {
    XELOGE("BaseHeap::QuerySize base page out of range");
    *out_size = 0;
    return false;
  }
  auto global_lock = global_critical_region_.Acquire();
  auto page_entry = page_table_[page_number];
  *in_out_address = (page_entry.base_address << page_size_shift_);
  *out_size = (page_entry.region_page_count << page_size_shift_);
  return true;
}

bool BaseHeap::QueryProtect(uint32_t address, uint32_t* out_protect) {
  uint32_t page_number = (address - heap_base_) >> page_size_shift_;
  if (page_number > page_table_.size()) {
    XELOGE("BaseHeap::QueryProtect base page out of range");
    *out_protect = 0;
    return false;
  }
  auto global_lock = global_critical_region_.Acquire();
  auto page_entry = page_table_[page_number];
  *out_protect = page_entry.current_protect;
  return true;
}

xe::memory::PageAccess BaseHeap::QueryRangeAccess(uint32_t low_address,
                                                  uint32_t high_address) {
  if (low_address > high_address || low_address < heap_base_ ||
      (high_address - heap_base_) >= heap_size_) {
    return xe::memory::PageAccess::kNoAccess;
  }
  uint32_t low_page_number = (low_address - heap_base_) >> page_size_shift_;
  uint32_t high_page_number = (high_address - heap_base_) >> page_size_shift_;
  bool all_readable = true;
  bool all_writable = true;
  {
    auto global_lock = global_critical_region_.Acquire();
    for (uint32_t i = low_page_number; i <= high_page_number; ++i) {
      uint32_t page_protect = page_table_[i].current_protect;
      if (!(page_protect & kMemoryProtectRead)) {
        all_readable = false;
      }
      // Check if page is writable in any form (Write or WriteCombine)
      if (!(page_protect & kMemoryProtectWrite) &&
          !(page_protect & kMemoryProtectWriteCombine)) {
        all_writable = false;
      }
    }
  }
  if (all_readable && all_writable) {
    return xe::memory::PageAccess::kReadWrite;
  } else if (all_readable) {
    return xe::memory::PageAccess::kReadOnly;
  } else {
    return xe::memory::PageAccess::kNoAccess;
  }
}

VirtualHeap::VirtualHeap() = default;

VirtualHeap::~VirtualHeap() = default;

void VirtualHeap::Initialize(Memory* memory, uint8_t* membase,
                             HeapType heap_type, uint32_t heap_base,
                             uint32_t heap_size, uint32_t page_size) {
  BaseHeap::Initialize(memory, membase, heap_type, heap_base, heap_size,
                       page_size);
}

PhysicalHeap::PhysicalHeap() : parent_heap_(nullptr) {}

PhysicalHeap::~PhysicalHeap() = default;

void PhysicalHeap::Initialize(Memory* memory, uint8_t* membase,
                              HeapType heap_type, uint32_t heap_base,
                              uint32_t heap_size, uint32_t page_size,
                              VirtualHeap* parent_heap) {
  uint32_t host_address_offset;
  if (heap_base >= 0xE0000000 &&
      xe::memory::allocation_granularity() > 0x1000) {
    host_address_offset = 0x1000;
  } else {
    host_address_offset = 0;
  }

  BaseHeap::Initialize(memory, membase, heap_type, heap_base, heap_size,
                       page_size, host_address_offset);
  parent_heap_ = parent_heap;

  // The physical base offset (host_address_offset) must be a multiple of
  // page_size. Otherwise, aligned parent allocations become misaligned after
  // translation back to virtual addresses (parent_address + heap_base_ -
  // GetPhysicalAddress(heap_base_) loses alignment).
  //
  // The allocation guards below check alignment against the guest physical
  // base offset (0x1000 for the E0000000 heap, as on the console), not
  // host_address_offset: a 64 KB-aligned physical block is at xxxxF000 in
  // the E0 range whatever the host. The two offsets agree where host
  // allocation granularity exceeds 4 KB (Windows, Apple Silicon); on a 4 KB
  // host (x86 macOS, Linux) host_address_offset is 0 and the old check
  // refused correct allocations (Burnout Paradise's first
  // MmAllocatePhysicalMemoryEx returned null on an Intel Mac).
  xenia_assert(host_address_offset % page_size == 0);

  system_page_size_ = uint32_t(xe::memory::page_size());
  xenia_assert(xe::is_pow2(system_page_size_));
  system_page_shift_ = xe::log2_floor(system_page_size_);

  system_page_count_ =
      (size_t(heap_size_) + host_address_offset + (system_page_size_ - 1)) /
      system_page_size_;
  system_page_flags_.resize((system_page_count_ + 63) / 64);
}

namespace {
// A physical range can be uploaded to the GPU and write-watched through one
// view (0xA0000000 / 0xC0000000 / 0xE0000000) while another view of the same
// memory has no watch: a page unallocated or read-only in a view when the
// watch was armed is skipped there (EnableAccessCallbacks). Allocating it in
// that view, or making it writable there, must then invalidate the range even
// though this view watches nothing, or the guest's writes through it are never
// seen. F.E.A.R. frees and reallocates a 4 KB-page (0xE0000000) block between
// frames while the GPU still reads it; its next texture landed in the block
// unnoticed and the GPU kept sampling the old one (the gun rendered red).
// XE_INVALIDATE_UNWATCHED=0 invalidates watched pages only, as before.
bool InvalidateUnwatchedEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("XE_INVALIDATE_UNWATCHED");
    return !(v && v[0] == '0' && !v[1]);
  }();
  return enabled;
}

// PhysicalHeap::Protect invalidates every guest page the protection change
// covers. BaseHeap::Protect applies the new protection to whole guest pages,
// 64 KB in the A0000000 heap and 16 MB in C0000000, so making a sub-page range
// writable also lifts the host write watch on the rest of its first and last
// pages. Invalidating only the requested bytes left those pages valid in the
// GPU caches while the guest rewrote them with no fault (Dragon Age: Origins
// unprotects a few KB of a 64 KB page, streams textures and vertices across
// it, then protects it again: stale vertices drew huge flat planes and
// missing bodies, stale textures a band over every face).
// XE_PROTECT_WHOLE_PAGES=0 invalidates only the requested range, as before.
bool ProtectWholePagesEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("XE_PROTECT_WHOLE_PAGES");
    return !(v && v[0] == '0' && !v[1]);
  }();
  return enabled;
}
}  // namespace

bool PhysicalHeap::Alloc(uint32_t size, uint32_t alignment,
                         uint32_t allocation_type, uint32_t protect,
                         bool top_down, uint32_t* out_address) {
  *out_address = 0;

  // Default top-down. Since parent heap is bottom-up this prevents
  // collisions.
  top_down = true;

  // Adjust alignment size our page size differs from the parent.
  size = xe::round_up(size, page_size_);
  alignment = xe::round_up(alignment, page_size_);

  auto global_lock = global_critical_region_.Acquire();

  // Allocate from parent heap (gets our physical address in 0-512mb).
  uint32_t parent_heap_start = GetPhysicalAddress(heap_base_);
  uint32_t parent_heap_end = GetPhysicalAddress(heap_base_ + (heap_size_ - 1));
  uint32_t parent_address;
  if (!parent_heap_->AllocRange(parent_heap_start, parent_heap_end, size,
                                alignment, allocation_type, protect, top_down,
                                &parent_address)) {
    XELOGE(
        "PhysicalHeap::Alloc unable to alloc physical memory in parent heap "
        "(requested {} bytes, parent free {}/{} pages)",
        size, parent_heap_->unreserved_page_count(),
        parent_heap_->total_page_count());
    return false;
  }

  // Given the address we've reserved in the parent heap, pin that here.
  // Shouldn't be possible for it to be allocated already.
  const uint32_t address = heap_base_ + parent_address - parent_heap_start;
  if ((address + GetPhysicalAddress(heap_base_)) % alignment != 0) {
    XELOGE(
        "PhysicalHeap::Alloc translated address {:08X} misaligned "
        "(alignment {:08X}, physical base offset {:08X})",
        address, alignment, parent_heap_start);
    parent_heap_->Release(parent_address);
    return false;
  }
  if (!BaseHeap::AllocFixed(address, size, alignment, allocation_type,
                            protect)) {
    XELOGE(
        "PhysicalHeap::Alloc unable to pin physical memory in physical heap");
    parent_heap_->Release(parent_address);
    return false;
  }
  // Fresh allocation: whatever a read watch still owed these pages is
  // irrelevant now (kDiscard); unwatched pages are untouched.
  TriggerCallbacks(std::move(global_lock), address, size, true, true, true,
                   InvalidateUnwatchedEnabled(), true);
  ReconcileHostProtection(address, size);
  *out_address = address;
  return true;
}

bool PhysicalHeap::AllocFixed(uint32_t base_address, uint32_t size,
                              uint32_t alignment, uint32_t allocation_type,
                              uint32_t protect) {
  // Adjust alignment size our page size differs from the parent.
  size = xe::round_up(size, page_size_);
  alignment = xe::round_up(alignment, page_size_);

  auto global_lock = global_critical_region_.Acquire();

  // Allocate from parent heap (gets our physical address in 0-512mb).
  // NOTE: this can potentially overwrite heap contents if there are already
  // committed pages in the requested physical range.
  // TODO(benvanik): flag for ensure-not-committed?
  uint32_t parent_base_address = GetPhysicalAddress(base_address);
  if (!parent_heap_->AllocFixed(parent_base_address, size, alignment,
                                allocation_type, protect)) {
    XELOGE(
        "PhysicalHeap::AllocFixed unable to alloc physical memory in parent "
        "heap");
    return false;
  }

  // Given the address we've reserved in the parent heap, pin that here.
  // Shouldn't be possible for it to be allocated already.
  const uint32_t address =
      heap_base_ + parent_base_address - GetPhysicalAddress(heap_base_);
  if ((address + GetPhysicalAddress(heap_base_)) % alignment != 0) {
    XELOGE(
        "PhysicalHeap::AllocFixed translated address {:08X} misaligned "
        "(alignment {:08X}, physical base offset {:08X})",
        address, alignment, GetPhysicalAddress(heap_base_));
    parent_heap_->Release(parent_base_address);
    return false;
  }
  if (!BaseHeap::AllocFixed(address, size, alignment, allocation_type,
                            protect)) {
    XELOGE(
        "PhysicalHeap::AllocFixed unable to pin physical memory in physical "
        "heap");
    parent_heap_->Release(parent_base_address);
    return false;
  }
  TriggerCallbacks(std::move(global_lock), address, size, true, true, true,
                   InvalidateUnwatchedEnabled(), true);
  ReconcileHostProtection(address, size);
  return true;
}

bool PhysicalHeap::AllocRange(uint32_t low_address, uint32_t high_address,
                              uint32_t size, uint32_t alignment,
                              uint32_t allocation_type, uint32_t protect,
                              bool top_down, uint32_t* out_address) {
  *out_address = 0;

  // Adjust alignment size our page size differs from the parent.
  size = xe::round_up(size, page_size_);
  alignment = xe::round_up(alignment, page_size_);

  auto global_lock = global_critical_region_.Acquire();

  // Allocate from parent heap (gets our physical address in 0-512mb).
  low_address = std::max(heap_base_, low_address);
  high_address = std::min(heap_base_ + (heap_size_ - 1), high_address);
  uint32_t parent_low_address = GetPhysicalAddress(low_address);
  uint32_t parent_high_address = GetPhysicalAddress(high_address);
  uint32_t parent_address;
  if (!parent_heap_->AllocRange(parent_low_address, parent_high_address, size,
                                alignment, allocation_type, protect, top_down,
                                &parent_address)) {
    XELOGE(
        "PhysicalHeap::AllocRange unable to alloc physical memory in parent "
        "heap (requested {} bytes, parent free {}/{} pages)",
        size, parent_heap_->unreserved_page_count(),
        parent_heap_->total_page_count());
    return false;
  }
  // Given the address we've reserved in the parent heap, pin that here.
  // Shouldn't be possible for it to be allocated already.
  const uint32_t address =
      heap_base_ + parent_address - GetPhysicalAddress(heap_base_);
  if ((address + GetPhysicalAddress(heap_base_)) % alignment != 0) {
    XELOGE(
        "PhysicalHeap::AllocRange translated address {:08X} misaligned "
        "(alignment {:08X}, physical base offset {:08X})",
        address, alignment, GetPhysicalAddress(heap_base_));
    parent_heap_->Release(parent_address);
    return false;
  }
  if (!BaseHeap::AllocFixed(address, size, alignment, allocation_type,
                            protect)) {
    XELOGE(
        "PhysicalHeap::AllocRange unable to pin physical memory in physical "
        "heap");
    parent_heap_->Release(parent_address);
    return false;
  }
  TriggerCallbacks(std::move(global_lock), address, size, true, true, true,
                   InvalidateUnwatchedEnabled(), true);
  ReconcileHostProtection(address, size);
  *out_address = address;
  return true;
}

bool PhysicalHeap::AllocSystemHeap(uint32_t size, uint32_t alignment,
                                   uint32_t allocation_type, uint32_t protect,
                                   bool top_down, uint32_t* out_address) {
  return Alloc(size, alignment, allocation_type, protect, top_down,
               out_address);
}

bool PhysicalHeap::Decommit(uint32_t address, uint32_t size) {
  auto global_lock = global_critical_region_.Acquire();

  uint32_t parent_address = GetPhysicalAddress(address);
  if (!parent_heap_->Decommit(parent_address, size)) {
    XELOGE("PhysicalHeap::Decommit failed due to parent heap failure");
    return false;
  }

  // Not caring about the contents anymore.
  TriggerCallbacks(std::move(global_lock), address, size, true, true, true,
                   false, true);

  return BaseHeap::Decommit(address, size);
}

bool PhysicalHeap::Release(uint32_t base_address, uint32_t* out_region_size) {
  auto global_lock = global_critical_region_.Acquire();

  uint32_t parent_base_address = GetPhysicalAddress(base_address);
  if (!parent_heap_->Release(parent_base_address, out_region_size)) {
    XELOGE("PhysicalHeap::Release failed due to parent heap failure");
    return false;
  }

  // Must invalidate here because the range being released may be reused in
  // another mapping of physical memory - but callback flags are set in each
  // heap separately (https://github.com/xenia-project/xenia/issues/1559 -
  // dynamic vertices in 4D5307F2 start screen and menu allocated in 0xA0000000
  // at addresses that overlap intro video textures in 0xE0000000, with the
  // state of the allocator as of February 24th, 2020). If memory is invalidated
  // in Alloc instead, Alloc won't be aware of callbacks enabled in other heaps,
  // thus callback handlers will keep considering this range valid forever.
  uint32_t region_size;
  if (QuerySize(base_address, &region_size)) {
    TriggerCallbacks(std::move(global_lock), base_address, region_size, true,
                     true, true, false, true);
  }

  return BaseHeap::Release(base_address, out_region_size);
}

bool PhysicalHeap::Protect(uint32_t address, uint32_t size, uint32_t protect,
                           uint32_t* old_protect) {
  auto global_lock = global_critical_region_.Acquire();

  // Only invalidate if making writable again, for simplicity - not when simply
  // marking some range as immutable, for instance. Read-only pages carry no
  // write watch in this view (EnableAccessCallbacks skips them), so the range
  // is invalidated even unwatched (see InvalidateUnwatchedEnabled).
  if (protect & kMemoryProtectWrite) {
    uint32_t invalidate_address = address;
    uint32_t invalidate_size = size;
    if (ProtectWholePagesEnabled() && size && address >= heap_base_) {
      // The guest pages BaseHeap::Protect changes (see
      // ProtectWholePagesEnabled).
      const uint64_t page_mask = page_size_ - 1;
      const uint64_t relative = address - heap_base_;
      const uint64_t first = relative & ~page_mask;
      const uint64_t end = std::min<uint64_t>(
          ((relative + size - 1) | page_mask) + 1, heap_size_);
      if (end > first) {
        invalidate_address = heap_base_ + uint32_t(first);
        invalidate_size = uint32_t(end - first);
      }
    }
    TriggerCallbacks(std::move(global_lock), invalidate_address,
                     invalidate_size, true, true, false,
                     InvalidateUnwatchedEnabled());
  }

  if (!parent_heap_->Protect(GetPhysicalAddress(address), size, protect,
                             old_protect)) {
    XELOGE("PhysicalHeap::Protect failed due to parent heap failure");
    return false;
  }

  if (!BaseHeap::Protect(address, size, protect)) {
    return false;
  }

  // BaseHeap::Protect changes host protection only when the heap's host
  // offset is aligned to the host page, which the E0000000 heap's 0x1000 is
  // not on a 16 KB-page host (Apple Silicon): there the guest's change never
  // reached the host, and the invalidation above has already dropped the
  // write watches (Banjo-Kazooie: Nuts & Bolts re-faulted forever on E0
  // pages from F2438000).
  ReconcileHostProtection(address, size);
  return true;
}

void PhysicalHeap::EnableAccessCallbacks(uint32_t physical_address,
                                         uint32_t length,
                                         bool enable_invalidation_notifications,
                                         bool enable_data_providers) {
  if (!enable_invalidation_notifications && !enable_data_providers) {
    return;
  }
  uint32_t physical_address_offset = GetPhysicalAddress(heap_base_);
  if (physical_address < physical_address_offset) {
    if (physical_address_offset - physical_address >= length) {
      return;
    }
    length -= physical_address_offset - physical_address;
    physical_address = physical_address_offset;
  }
  uint32_t heap_relative_address = physical_address - physical_address_offset;
  if (heap_relative_address >= heap_size_) {
    return;
  }
  length = std::min(length, heap_size_ - heap_relative_address);
  if (length == 0) {
    return;
  }

  uint32_t system_page_first =
      (heap_relative_address + host_address_offset()) >> system_page_shift_;
  swcache::PrefetchL1(&system_page_flags_[system_page_first >> 6]);
  uint32_t system_page_last =
      (heap_relative_address + length - 1 + host_address_offset()) >>
      system_page_shift_;
  system_page_last = std::min(system_page_last, system_page_count_ - 1);
  assert_true(system_page_first <= system_page_last);

  // Update callback flags for system pages and make their protection stricter
  // if needed.
  xe::memory::PageAccess protect_access =
      enable_data_providers ? xe::memory::PageAccess::kNoAccess
                            : xe::memory::PageAccess::kReadOnly;

  auto global_lock = global_critical_region_.Acquire();
  if (enable_invalidation_notifications) {
    if (enable_data_providers) {
      EnableAccessCallbacksInner<true, true>(system_page_first,
                                             system_page_last, protect_access);
    } else {
      EnableAccessCallbacksInner<true, false>(system_page_first,
                                              system_page_last, protect_access);
    }
  } else if (enable_data_providers) {
    EnableAccessCallbacksInner<false, true>(system_page_first, system_page_last,
                                            protect_access);
  } else {
    EnableAccessCallbacksInner<false, false>(system_page_first,
                                             system_page_last, protect_access);
  }
}

template <bool enable_invalidation_notifications, bool enable_data_providers>
XE_NOINLINE void PhysicalHeap::EnableAccessCallbacksInner(
    const uint32_t system_page_first, const uint32_t system_page_last,
    xe::memory::PageAccess protect_access) XE_RESTRICT {
  uint8_t* protect_base = membase_ + heap_base_;
  uint32_t protect_system_page_first = UINT32_MAX;

  SystemPageFlagsBlock* XE_RESTRICT sys_page_flags = system_page_flags_.data();
  PageEntry* XE_RESTRICT page_table_ptr = page_table_.data();
  (void)page_table_ptr;

  // chrispy: a lot of time is spent in this loop, and i think some of the work
  // may be avoidable and repetitive profiling shows quite a bit of time spent
  // in this loop, but very little spent actually calling Protect
  uint32_t i = system_page_first;

  uint32_t first_guest_page = SystemPagenumToGuestPagenum(system_page_first);
  uint32_t last_guest_page = SystemPagenumToGuestPagenum(system_page_last);

  uint32_t guest_one = SystemPagenumToGuestPagenum(1);

  uint32_t system_one = GuestPagenumToSystemPagenum(1);
  for (; i <= system_page_last; ++i) {
    {
      // Pages still armed, as most are when a resolve or memexport range is
      // re-armed every frame, need nothing, so skip a fully armed block.
      uint32_t block_last = std::min(i | 63, system_page_last);
      uint64_t block_mask = (~uint64_t(0) >> (63 - (block_last & 63))) &
                            (~uint64_t(0) << (i & 63));
      const SystemPageFlagsBlock& block = sys_page_flags[i >> 6];
      uint64_t armed = ~uint64_t(0);
      if constexpr (enable_invalidation_notifications) {
        armed &= block.notify_on_invalidation;
      }
      if constexpr (enable_data_providers) {
        armed &= block.notify_on_read;
      }
      if ((armed & block_mask) == block_mask) {
        if (protect_system_page_first != UINT32_MAX) {
          xe::memory::Protect(
              protect_base + (protect_system_page_first << system_page_shift_),
              (i - protect_system_page_first) << system_page_shift_,
              protect_access);
          protect_system_page_first = UINT32_MAX;
        }
        i = block_last;
        continue;
      }
    }
    // Check if need to enable callbacks for the page and raise its protection.
    //
    // If enabling invalidation notifications:
    // - Page writable and not watched for changes yet - protect and enable
    //   invalidation notifications.
    // - Page seen as writable by the guest, but only needs data providers -
    //   just set the bits to enable invalidation notifications (already has
    //   even stricter protection than needed).
    // - Page not writable as requested by the game - don't do anything (need
    //   real access violations here).
    // If enabling data providers:
    // - Page accessible (either read/write or read-only) and didn't need data
    //   providers initially - protect and enable data providers.
    // - Otherwise - do nothing.
    //
    // It's safe not to await data provider completion here before protecting as
    // this never makes protection lighter, so it can't interfere with page
    // faults that await data providers.
    //
    // Enabling data providers doesn't need to be deferred - providers will be
    // polled for the last time without releasing the lock.
    SystemPageFlagsBlock& page_flags_block = sys_page_flags[i >> 6];

#if XE_ARCH_AMD64 == 1
    // x86 modulus shift
    uint64_t page_flags_bit = uint64_t(1) << i;
#else
    uint64_t page_flags_bit = uint64_t(1) << (i & 63);
#endif

    xe::memory::PageAccess current_page_access = SystemPageGuestAccess(i);
    bool protect_system_page = false;
    // Don't do anything with inaccessible pages - don't protect, don't enable
    // callbacks - because real access violations are needed there. And don't
    // enable invalidation notifications for read-only pages for the same
    // reason.
    if (current_page_access != xe::memory::PageAccess::kNoAccess) {
      if constexpr (enable_invalidation_notifications) {
        if (current_page_access != xe::memory::PageAccess::kReadOnly &&
            (page_flags_block.notify_on_invalidation & page_flags_bit) == 0) {
          page_flags_block.notify_on_invalidation |= page_flags_bit;
          // A read-watched page is already protected no-access, stricter
          // than read-only, so don't loosen it here.
          if ((page_flags_block.notify_on_read & page_flags_bit) == 0) {
            protect_system_page = true;
          }
        }
      }
      if constexpr (enable_data_providers) {
        // Read watches protect the page no-access, so an accessible page not
        // yet read-watched needs protecting.
        if ((page_flags_block.notify_on_read & page_flags_bit) == 0) {
          protect_system_page = true;
          page_flags_block.notify_on_read |= page_flags_bit;
        }
      }
    }
    if (protect_system_page) {
      if (protect_system_page_first == UINT32_MAX) {
        protect_system_page_first = i;
      }
    } else {
      if (protect_system_page_first != UINT32_MAX) {
        xe::memory::Protect(
            protect_base + (protect_system_page_first << system_page_shift_),
            (i - protect_system_page_first) << system_page_shift_,
            protect_access);
        protect_system_page_first = UINT32_MAX;
      }
    }
  }

  if (protect_system_page_first != UINT32_MAX) {
    xe::memory::Protect(
        protect_base + (protect_system_page_first << system_page_shift_),
        (system_page_last + 1 - protect_system_page_first)
            << system_page_shift_,
        protect_access);
  }
}

std::string PhysicalHeap::DescribeAccessState(uint32_t physical_address) const {
  const uint32_t physical_address_offset = GetPhysicalAddress(heap_base_);
  if (physical_address < physical_address_offset ||
      physical_address - physical_address_offset >= heap_size_) {
    return "-";
  }
  const uint32_t relative = physical_address - physical_address_offset;
  const uint32_t system_page =
      (relative + host_address_offset()) >> system_page_shift_;
  if (system_page >= system_page_count_) {
    return "-";
  }
  const SystemPageFlagsBlock& flags = system_page_flags_[system_page >> 6];
  const uint64_t bit = uint64_t(1) << (system_page & 63);
  void* host =
      membase_ + heap_base_ + (size_t(system_page) << system_page_shift_);
  size_t length = system_page_size_;
  xe::memory::PageAccess host_access = xe::memory::PageAccess::kNoAccess;
  const bool queried = xe::memory::QueryProtect(host, length, host_access);
  const uint32_t guest_page = relative >> page_size_shift_;
  const PageEntry& entry = page_table_[guest_page];
  return fmt::format(
      "inv={} read={} host={} guest_state={:X} guest_protect={:X} "
      "system_access={}",
      (flags.notify_on_invalidation & bit) ? 1 : 0,
      (flags.notify_on_read & bit) ? 1 : 0,
      queried ? int(host_access) : -1, uint32_t(entry.state),
      uint32_t(entry.current_protect), int(SystemPageGuestAccess(system_page)));
}

xe::memory::PageAccess PhysicalHeap::SystemPageGuestAccess(
    uint32_t system_page_number) const {
  uint32_t offset = host_address_offset();
  uint32_t system_base = system_page_number << system_page_shift_;
  uint32_t system_last = system_base + (system_page_size_ - 1);
  if (system_last < offset) {
    return xe::memory::PageAccess::kNoAccess;
  }
  uint32_t guest_page_first =
      system_base > offset ? (system_base - offset) >> page_size_shift_ : 0;
  uint32_t guest_page_count = uint32_t(page_table_.size());
  if (guest_page_first >= guest_page_count) {
    return xe::memory::PageAccess::kNoAccess;
  }
  uint32_t guest_page_last = (system_last - offset) >> page_size_shift_;
  if (guest_page_last >= guest_page_count) {
    guest_page_last = guest_page_count - 1;
  }
  xe::memory::PageAccess access = xe::memory::PageAccess::kNoAccess;
  bool any_unallocated = false;
  for (uint32_t i = guest_page_first; i <= guest_page_last; ++i) {
    if (!page_table_[i].state) {
      any_unallocated = true;
      continue;
    }
    xe::memory::PageAccess page_access =
        ToPageAccess(page_table_[i].current_protect);
    if (page_access == xe::memory::PageAccess::kReadWrite) {
      return xe::memory::PageAccess::kReadWrite;
    }
    if (page_access == xe::memory::PageAccess::kReadOnly) {
      access = xe::memory::PageAccess::kReadOnly;
    }
  }
  if (any_unallocated && parent_heap_) {
    // Guest pages unallocated in the physical view may still be committed
    // through the 1:1 virtual parent heap.
    uint32_t relative_address = system_base > offset ? system_base - offset : 0;
    uint32_t physical_first = GetPhysicalAddress(heap_base_) + relative_address;
    uint32_t physical_last =
        physical_first + (system_last - offset + 1 - relative_address) - 1;
    // Most permissive parent page, as for this view's own pages above: every
    // window aliases the same physical memory, so a host page is reachable if
    // any physical page it covers is. The strictest page (QueryRangeAccess
    // over the whole host page) made a 16 KB host page in the 0xC0000000
    // window no-access when only one of its four 4 KB physical pages was
    // allocated (through 0xE0000000), and Gears of War: Judgment's read of
    // that page through 0xC0000000 crashed its game thread.
    const uint32_t parent_page_size = parent_heap_->page_size();
    for (uint32_t physical_page = physical_first & ~(parent_page_size - 1);
         physical_page <= physical_last; physical_page += parent_page_size) {
      xe::memory::PageAccess parent_access = parent_heap_->QueryRangeAccess(
          physical_page, physical_page + (parent_page_size - 1));
      if (parent_access == xe::memory::PageAccess::kReadWrite) {
        return xe::memory::PageAccess::kReadWrite;
      }
      if (parent_access == xe::memory::PageAccess::kReadOnly) {
        access = parent_access;
      }
      if (physical_page + parent_page_size < physical_page) {
        break;
      }
    }
  }
  return access;
}

void PhysicalHeap::ProvideReadWatchedPage(
    global_unique_lock_type& global_lock_locked_once, uint32_t virtual_address,
    bool is_write) {
  if (virtual_address < heap_base_ ||
      virtual_address - heap_base_ >= heap_size_) {
    return;
  }
  uint32_t heap_relative_address = virtual_address - heap_base_;
  uint32_t system_page =
      (heap_relative_address + host_address_offset()) >> system_page_shift_;
  if (system_page >= system_page_count_ ||
      !(system_page_flags_[system_page >> 6].notify_on_read &
        (uint64_t(1) << (system_page & 63)))) {
    return;
  }
  std::shared_lock<std::shared_mutex> calls_lock(
      memory_->physical_memory_read_callback_calls_mutex_);
  // Copied, as the list may change once the global lock is released.
  std::pair<Memory::PhysicalMemoryReadCallback, void*> callbacks[4];
  size_t callback_count = memory_->physical_memory_read_callbacks_.size();
  if (!callback_count || callback_count > xe::countof(callbacks)) {
    return;
  }
  for (size_t i = 0; i < callback_count; ++i) {
    callbacks[i] = *memory_->physical_memory_read_callbacks_[i];
  }
  uint32_t physical_address_offset = GetPhysicalAddress(heap_base_);
  uint32_t physical_address_start =
      xe::sat_sub(system_page << system_page_shift_, host_address_offset()) +
      physical_address_offset;
  uint32_t physical_length = std::min(
      xe::sat_sub((system_page << system_page_shift_) + system_page_size_,
                  host_address_offset()) +
          physical_address_offset - physical_address_start,
      heap_size_ - (physical_address_start - physical_address_offset));
  Memory::PhysicalAccess access =
      is_write ? Memory::PhysicalAccess::kWriteUnlocked
               : Memory::PhysicalAccess::kRead;
  global_lock_locked_once.unlock();
  for (size_t i = 0; i < callback_count; ++i) {
    callbacks[i].first(callbacks[i].second, physical_address_start,
                       physical_length, access);
  }
  calls_lock.unlock();
  global_lock_locked_once.lock();
}

void PhysicalHeap::ProvideReadWatchedEdgePages(uint32_t virtual_address,
                                               uint32_t length) {
  if (!length || virtual_address < heap_base_ ||
      virtual_address - heap_base_ >= heap_size_) {
    return;
  }
  length = std::min(length, heap_size_ - (virtual_address - heap_base_));
  uint32_t host_first = virtual_address - heap_base_ + host_address_offset();
  uint32_t host_last = host_first + length - 1;
  uint32_t page_mask = system_page_size_ - 1;
  bool first_partial = (host_first & page_mask) != 0;
  bool last_partial = (host_last & page_mask) != page_mask;
  bool one_page =
      (host_first >> system_page_shift_) == (host_last >> system_page_shift_);
  auto provide_page = [&](uint32_t address) {
    auto global_lock = global_critical_region_.Acquire();
    ProvideReadWatchedPage(global_lock, address, true);
  };
  if (first_partial) {
    provide_page(virtual_address);
  }
  if (last_partial && !(first_partial && one_page)) {
    provide_page(virtual_address + length - 1);
  }
}

void PhysicalHeap::ReconcileHostProtection(uint32_t address, uint32_t size) {
  // On a host whose page is larger than the guest page, BaseHeap skips host
  // commits and protection changes for this heap (ShouldSkipHostCommit, the
  // Protect alignment check), so a host page a watch made read-only keeps
  // that protection after its watch is dropped. Bring each host page in the
  // range to its guest protection, keeping what remaining watches need.
  if (!size || (page_size_ >= system_page_size_ &&
                 !(host_address_offset() & (system_page_size_ - 1)))) {
    // BaseHeap applies this heap's protection on the host itself.
    return;
  }
  auto global_lock = global_critical_region_.Acquire();
  uint32_t heap_relative = address - heap_base_;
  uint32_t heap_relative_end =
      std::min(heap_relative + (size - 1), heap_size_ - 1);
  uint32_t system_page_first =
      (heap_relative + host_address_offset()) >> system_page_shift_;
  uint32_t system_page_last = std::min(
      (heap_relative_end + host_address_offset()) >> system_page_shift_,
      system_page_count_ - 1);
  for (uint32_t i = system_page_first; i <= system_page_last; ++i) {
    RestoreUnwatchedSystemPage(i, false);
  }
}

void PhysicalHeap::RestoreUnwatchedSystemPage(uint32_t system_page,
                                              bool from_fault) {
  // Called with the global lock held, for a page without a read watch: the
  // host protection is the guest's, or read-only under a write watch.
  uint64_t bit = uint64_t(1) << (system_page & 63);
  const SystemPageFlagsBlock& flags = system_page_flags_[system_page >> 6];
  if (flags.notify_on_read & bit) {
    return;
  }
  xe::memory::PageAccess target = SystemPageGuestAccess(system_page);
  if ((flags.notify_on_invalidation & bit) &&
      target == xe::memory::PageAccess::kReadWrite) {
    target = xe::memory::PageAccess::kReadOnly;
  }
  uint8_t* host_page =
      membase_ + heap_base_ + (system_page << system_page_shift_);
  if (from_fault) {
    // Two threads faulting on one watched page is ordinary: the first
    // unprotects it and clears the watch, and the second then finds no watch
    // on a page that is already writable. Only a page still protected is a
    // repair (Banjo-Kazooie: Nuts & Bolts logged that race as a repair on
    // every run, which hid whether any real case remained).
    size_t query_length = system_page_size_;
    xe::memory::PageAccess current;
    if (xe::memory::QueryProtect(host_page, query_length, current) &&
        current == target) {
      return;
    }
    static std::atomic<uint32_t> restored{0};
    if (restored.fetch_add(1, std::memory_order_relaxed) < 8) {
      XELOGW("PhysicalHeap: restored host protection of unwatched system page "
             "{} (heap {:08X})",
             system_page, heap_base_);
    }
  }
  xe::memory::Protect(host_page, system_page_size_, target);
}

bool PhysicalHeap::TriggerCallbacks(
    global_unique_lock_type global_lock_locked_once, uint32_t virtual_address,
    uint32_t length, bool is_write, bool unwatch_exact_range, bool unprotect,
    bool invalidate_unwatched, bool contents_discarded,
    bool read_callbacks_may_wait) {

  if (virtual_address < heap_base_) {
    if (heap_base_ - virtual_address >= length) {
      return false;
    }
    length -= heap_base_ - virtual_address;
    virtual_address = heap_base_;
  }
  uint32_t heap_relative_address = virtual_address - heap_base_;
  if (heap_relative_address >= heap_size_) {
    return false;
  }
  length = std::min(length, heap_size_ - heap_relative_address);
  if (length == 0) {
    return false;
  }

  uint32_t system_page_first =
      (heap_relative_address + host_address_offset()) >> system_page_shift_;
  uint32_t system_page_last =
      (heap_relative_address + length - 1 + host_address_offset()) >>
      system_page_shift_;
  system_page_last = std::min(system_page_last, system_page_count_ - 1);
  assert_true(system_page_first <= system_page_last);
  uint32_t block_index_first = system_page_first >> 6;
  uint32_t block_index_last = system_page_last >> 6;

  if (!is_write) {
    // Read fault (or a host reader about to read through the virtual view):
    // publish read-watched pages and lift their no-access protection.
    bool any_read_watched = false;
    for (uint32_t i = block_index_first; i <= block_index_last; ++i) {
      uint64_t block = system_page_flags_[i].notify_on_read;
      if (i == block_index_first) {
        block &= ~((uint64_t(1) << (system_page_first & 63)) - 1);
      }
      if (i == block_index_last && (system_page_last & 63) != 63) {
        block &= (uint64_t(1) << ((system_page_last & 63) + 1)) - 1;
      }
      if (block) {
        any_read_watched = true;
        break;
      }
    }
    if (!any_read_watched) {
      // No read watch here. If the guest mapping is accessible this is a race
      // with another thread that cleared the watch, so retry. If it is
      // no-access it is a genuine access violation, so propagate.
      if (SystemPageGuestAccess(system_page_first) ==
          xe::memory::PageAccess::kNoAccess) {
        return false;
      }
      // A faulting read must also find the host page readable (see the
      // write path below).
      if (length == 1) {
        RestoreUnwatchedSystemPage(system_page_first, true);
      }
      return true;
    }
    uint32_t physical_address_offset = GetPhysicalAddress(heap_base_);
    uint32_t physical_address_start =
        xe::sat_sub(system_page_first << system_page_shift_,
                    host_address_offset()) +
        physical_address_offset;
    uint32_t physical_length = std::min(
        xe::sat_sub(
            (system_page_last << system_page_shift_) + system_page_size_,
            host_address_offset()) +
            physical_address_offset - physical_address_start,
        heap_size_ - (physical_address_start - physical_address_offset));
    for (auto read_callback : memory_->physical_memory_read_callbacks_) {
      read_callback->first(read_callback->second, physical_address_start,
                           physical_length,
                           read_callbacks_may_wait
                               ? Memory::PhysicalAccess::kRead
                               : Memory::PhysicalAccess::kReadNoWait);
    }
    // Downgrade each read-watched page so the access proceeds: read-only if
    // it also carries a write watch, otherwise the guest protection.
    if (unprotect) {
      uint8_t* protect_base = membase_ + heap_base_;
      for (uint32_t i = system_page_first; i <= system_page_last; ++i) {
        uint64_t bit = uint64_t(1) << (i & 63);
        SystemPageFlagsBlock& flags = system_page_flags_[i >> 6];
        if (!(flags.notify_on_read & bit)) {
          continue;
        }
        xe::memory::PageAccess guest_access = SystemPageGuestAccess(i);
        xe::memory::PageAccess target;
        if (guest_access == xe::memory::PageAccess::kNoAccess) {
          target = xe::memory::PageAccess::kNoAccess;
        } else if (flags.notify_on_invalidation & bit) {
          target = xe::memory::PageAccess::kReadOnly;
        } else {
          target = guest_access;
        }
        xe::memory::Protect(protect_base + (i << system_page_shift_),
                            system_page_size_, target);
      }
    }
    for (uint32_t i = block_index_first; i <= block_index_last; ++i) {
      uint64_t mask = 0;
      if (i == block_index_first) {
        mask |= (uint64_t(1) << (system_page_first & 63)) - 1;
      }
      if (i == block_index_last && (system_page_last & 63) != 63) {
        mask |= ~((uint64_t(1) << ((system_page_last & 63) + 1)) - 1);
      }
      system_page_flags_[i].notify_on_read &= mask;
    }
    return true;
  }

  // Check if watching any page, whether need to call the callback at all.
  bool any_watched = false;
  for (uint32_t i = block_index_first; i <= block_index_last; ++i) {
    uint64_t block = system_page_flags_[i].notify_on_invalidation |
                     system_page_flags_[i].notify_on_read;
    if (i == block_index_first) {
      block &= ~((uint64_t(1) << (system_page_first & 63)) - 1);
    }
    if (i == block_index_last && (system_page_last & 63) != 63) {
      block &= (uint64_t(1) << ((system_page_last & 63) + 1)) - 1;
    }
    if (block) {
      any_watched = true;
      break;
    }
  }
  if (!any_watched && !invalidate_unwatched) {
    // No watches on this page — another thread already cleared them (race
    // condition between the fault firing and acquiring the lock). Return true
    // so the faulting instruction retries; the page is now unprotected and the
    // access will succeed. This is the signal-safe equivalent of the
    // QueryProtect check in the non-Linux path of
    // MMIOHandler::ExceptionCallback. If the guest doesn't permit the write
    // either, retrying it faults forever, so report it unhandled and let the
    // violation surface.
    if (SystemPageGuestAccess(system_page_first) !=
        xe::memory::PageAccess::kReadWrite) {
      return false;
    }
    // With no watch left the host protection must be the guest's. A faulting
    // access (length 1) on a page still host-protected would retry forever:
    // Banjo-Kazooie: Nuts & Bolts spun on a write fault at F2438000 (flags
    // clear, guest read-write, host read-only) at its first level load.
    if (length == 1) {
      RestoreUnwatchedSystemPage(system_page_first, true);
    }
    return true;
  }

  // Trigger callbacks.
  if (!unprotect) {
    // If not doing anything with protection, no point in unwatching excess
    // pages.
    unwatch_exact_range = true;
  }
  uint32_t physical_address_offset = GetPhysicalAddress(heap_base_);
  uint32_t physical_address_start =
      xe::sat_sub(system_page_first << system_page_shift_,
                  host_address_offset()) +
      physical_address_offset;
  uint32_t physical_length = std::min(
      xe::sat_sub((system_page_last << system_page_shift_) + system_page_size_,
                  host_address_offset()) +
          physical_address_offset - physical_address_start,
      heap_size_ - (physical_address_start - physical_address_offset));
  uint32_t unwatch_first = 0;
  uint32_t unwatch_last = UINT32_MAX;
  for (auto invalidation_callback :
       memory_->physical_memory_invalidation_callbacks_) {
    std::pair<uint32_t, uint32_t> callback_unwatch_range =
        invalidation_callback->first(invalidation_callback->second,
                                     physical_address_start, physical_length,
                                     unwatch_exact_range);
    if (!unwatch_exact_range) {
      unwatch_first = std::max(unwatch_first, callback_unwatch_range.first);
      unwatch_last = std::min(
          unwatch_last,
          xe::sat_add(
              callback_unwatch_range.first,
              std::max(callback_unwatch_range.second, uint32_t(1)) - 1));
    }
  }
  if (!unwatch_exact_range) {
    // Always unwatch at least the requested pages.
    unwatch_first = std::min(unwatch_first, physical_address_start);
    unwatch_last =
        std::max(unwatch_last, physical_address_start + physical_length - 1);
    // Don't unprotect too much if not caring much about the region (limit to
    // 4 MB - somewhat random, but max 1024 iterations of the page loop).
    constexpr uint32_t kMaxUnwatchExcess = 4 * 1024 * 1024;
    unwatch_first = std::max(unwatch_first,
                             physical_address_start & ~(kMaxUnwatchExcess - 1));
    unwatch_last =
        std::min(unwatch_last, (physical_address_start + physical_length - 1) |
                                   (kMaxUnwatchExcess - 1));
    // Convert to heap-relative addresses.
    unwatch_first = xe::sat_sub(unwatch_first, physical_address_offset);
    unwatch_last = xe::sat_sub(unwatch_last, physical_address_offset);
    // Clamp to the heap upper bound.
    unwatch_first = std::min(unwatch_first, heap_size_ - 1);
    unwatch_last = std::min(unwatch_last, heap_size_ - 1);
    // Convert to system pages and update the range.
    unwatch_first += host_address_offset();
    unwatch_last += host_address_offset();
    assert_true(unwatch_first <= unwatch_last);
    system_page_first = unwatch_first >> system_page_shift_;
    system_page_last = unwatch_last >> system_page_shift_;
    block_index_first = system_page_first >> 6;
    block_index_last = system_page_last >> 6;
  }

  // The read watches this write drops, whose callbacks may have to publish
  // something that must land before the write does.
  {
    uint32_t dropped_first = UINT32_MAX, dropped_last = 0, dropped_count = 0;
    for (uint32_t i = system_page_first; i <= system_page_last; ++i) {
      if (system_page_flags_[i >> 6].notify_on_read &
          (uint64_t(1) << (i & 63))) {
        dropped_first = std::min(dropped_first, i);
        dropped_last = i;
        ++dropped_count;
      }
    }
    if (dropped_count) {
      // A discard covers only the pages wholly inside the range named. Parts
      // of pages outside it, and pages unwatched past it, keep their contents.
      uint32_t discard_first = UINT32_MAX, discard_last = 0;
      if (contents_discarded) {
        uint32_t host_first = heap_relative_address + host_address_offset();
        uint64_t host_end = uint64_t(host_first) + length;
        discard_first =
            (host_first + system_page_size_ - 1) >> system_page_shift_;
        if ((host_end >> system_page_shift_) > discard_first) {
          discard_last = uint32_t(host_end >> system_page_shift_) - 1;
        } else {
          discard_first = UINT32_MAX;
        }
      }
      auto notify = [&](uint32_t page_first, uint32_t page_last,
                        Memory::PhysicalAccess access) {
        if (page_first > page_last) {
          return;
        }
        uint32_t start = xe::sat_sub(page_first << system_page_shift_,
                                     host_address_offset()) +
                         physical_address_offset;
        uint32_t end = xe::sat_sub((page_last + 1) << system_page_shift_,
                                   host_address_offset()) +
                       physical_address_offset;
        for (auto read_callback : memory_->physical_memory_read_callbacks_) {
          read_callback->first(read_callback->second, start, end - start,
                               access);
        }
      };
      // The global lock is held here: a write fault's repeat pass must not
      // wait (its unlocked first pass already did).
      const Memory::PhysicalAccess write_access =
          read_callbacks_may_wait ? Memory::PhysicalAccess::kWrite
                                  : Memory::PhysicalAccess::kWriteNoWait;
      if (discard_first > discard_last || discard_last < dropped_first ||
          discard_first > dropped_last) {
        notify(dropped_first, dropped_last, write_access);
      } else {
        if (discard_first > dropped_first) {
          notify(dropped_first, discard_first - 1, write_access);
        }
        notify(std::max(dropped_first, discard_first),
               std::min(dropped_last, discard_last),
               Memory::PhysicalAccess::kDiscard);
        if (discard_last < dropped_last) {
          notify(discard_last + 1, dropped_last, write_access);
        }
      }
    }
  }

  // Unprotect ranges that need unprotection.
  if (unprotect) {
    uint8_t* protect_base = membase_ + heap_base_;
    uint32_t unprotect_system_page_first = UINT32_MAX;
    for (uint32_t i = system_page_first; i <= system_page_last; ++i) {
      // Check if need to allow writing to this page. Read-watched pages are
      // unprotected here too so a write to one doesn't re-fault.
      bool unprotect_page =
          ((system_page_flags_[i >> 6].notify_on_invalidation |
            system_page_flags_[i >> 6].notify_on_read) &
           (uint64_t(1) << (i & 63))) != 0;
      if (unprotect_page) {
        if (SystemPageGuestAccess(i) != xe::memory::PageAccess::kReadWrite) {
          unprotect_page = false;
        }
      }
      if (unprotect_page) {
        if (unprotect_system_page_first == UINT32_MAX) {
          unprotect_system_page_first = i;
        }
      } else {
        if (unprotect_system_page_first != UINT32_MAX) {
          xe::memory::Protect(
              protect_base +
                  (unprotect_system_page_first << system_page_shift_),
              (i - unprotect_system_page_first) << system_page_shift_,
              xe::memory::PageAccess::kReadWrite);
          unprotect_system_page_first = UINT32_MAX;
        }
      }
    }
    if (unprotect_system_page_first != UINT32_MAX) {
      xe::memory::Protect(
          protect_base + (unprotect_system_page_first << system_page_shift_),
          (system_page_last + 1 - unprotect_system_page_first)
              << system_page_shift_,
          xe::memory::PageAccess::kReadWrite);
    }
  }

  // Mark pages as not write-watched. Unprotected pages are readable and
  // writable now, so clear the read watch too.
  for (uint32_t i = block_index_first; i <= block_index_last; ++i) {
    uint64_t mask = 0;
    if (i == block_index_first) {
      mask |= (uint64_t(1) << (system_page_first & 63)) - 1;
    }
    if (i == block_index_last && (system_page_last & 63) != 63) {
      mask |= ~((uint64_t(1) << ((system_page_last & 63) + 1)) - 1);
    }
    system_page_flags_[i].notify_on_invalidation &= mask;
    system_page_flags_[i].notify_on_read &= mask;
  }

  return true;
}

uint32_t PhysicalHeap::GetPhysicalAddress(uint32_t address) const {
  assert_true(address >= heap_base_);
  address -= heap_base_;
  assert_true(address < heap_size_);
  if (heap_base_ >= 0xE0000000) {
    address += 0x1000;
  }
  return address;
}

}  // namespace xe
