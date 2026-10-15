/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/memory.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>

#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"

#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

#if XE_PLATFORM_ANDROID
#include <dlfcn.h>
#include <linux/ashmem.h>
#include <string.h>
#include <sys/ioctl.h>

#include "xenia/base/main_android.h"
#endif

namespace xe {
namespace memory {

#if XE_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* libandroid_;
// API 26+.
static int (*android_ASharedMemory_create_)(const char* name, size_t size);

void AndroidInitialize() {
  if (xe::GetAndroidApiLevel() >= 26) {
    libandroid_ = dlopen("libandroid.so", RTLD_NOW);
    assert_not_null(libandroid_);
    if (libandroid_) {
      android_ASharedMemory_create_ =
          reinterpret_cast<decltype(android_ASharedMemory_create_)>(
              dlsym(libandroid_, "ASharedMemory_create"));
      assert_not_null(android_ASharedMemory_create_);
    }
  }
}

void AndroidShutdown() {
  android_ASharedMemory_create_ = nullptr;
  if (libandroid_) {
    dlclose(libandroid_);
    libandroid_ = nullptr;
  }
}
#endif

size_t page_size() { return getpagesize(); }
size_t allocation_granularity() { return page_size(); }

uint32_t ToPosixProtectFlags(PageAccess access) {
  switch (access) {
    case PageAccess::kNoAccess:
      return PROT_NONE;
    case PageAccess::kReadOnly:
      return PROT_READ;
    case PageAccess::kReadWrite:
      return PROT_READ | PROT_WRITE;
    case PageAccess::kExecuteReadOnly:
      return PROT_READ | PROT_EXEC;
    case PageAccess::kExecuteReadWrite:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      assert_unhandled_case(access);
      return PROT_NONE;
  }
}

PageAccess ToXeniaProtectFlags(const char* protection) {
  if (protection[0] == 'r' && protection[1] == 'w' && protection[2] == 'x') {
    return PageAccess::kExecuteReadWrite;
  }
  if (protection[0] == 'r' && protection[1] == '-' && protection[2] == 'x') {
    return PageAccess::kExecuteReadOnly;
  }
  if (protection[0] == 'r' && protection[1] == 'w' && protection[2] == '-') {
    return PageAccess::kReadWrite;
  }
  if (protection[0] == 'r' && protection[1] == '-' && protection[2] == '-') {
    return PageAccess::kReadOnly;
  }
  return PageAccess::kNoAccess;
}

#if XE_PLATFORM_MAC
bool IsWritableExecutableMemorySupported() {
  // macOS enforces W^X: a single shared file mapping cannot be both writable and
  // executable. Report unsupported so the code cache uses separate RW (write)
  // and RX (execute) views of the same shm instead of one RWX mapping.
  return false;
}
#else
bool IsWritableExecutableMemorySupported() { return true; }
#endif  // XE_PLATFORM_MAC

bool ReserveFixedRangesEarly() {
  // No-op. This existed to pre-reserve the fixed sub-4 GB code-cache window
  // against ASLR squatters, but canary's guest memory now lands at an OS-chosen
  // base (Memory::MapViewsMac maps with mmap(nullptr,...) then MAP_FIXED), so
  // there is no fixed low guest block to protect. Worse, an early anonymous
  // PROT_NONE reservation here makes the code cache's later shm-backed
  // MAP_FIXED views (RX at 0xA0000000, RW at 0xB0000000) fail with ENOMEM on
  // macOS - the kernel will not overlay a sub-range of an anonymous reservation
  // with a different (file) backing. The code cache instead MAP_FIXEDs into
  // address space that is still free this early in startup. Kept as a no-op so
  // the entry point's contract is unchanged.
  return true;
}

struct MappedFileRange {
  uintptr_t region_begin;
  uintptr_t region_end;
};

std::vector<MappedFileRange> mapped_file_ranges;
std::mutex g_mapped_file_ranges_mutex;

void* AllocFixed(void* base_address, size_t length,
                 AllocationType allocation_type, PageAccess access) {
  // mmap does not support reserve / commit, so ignore allocation_type.
  uint32_t prot = ToPosixProtectFlags(access);
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;

#if XE_PLATFORM_MAC
  // macOS requires MAP_JIT for anonymous writable+executable allocations.
  if (access == PageAccess::kExecuteReadWrite ||
      access == PageAccess::kExecuteReadOnly) {
    flags |= MAP_JIT;
  }
#endif  // XE_PLATFORM_MAC

  if (base_address != nullptr) {
    if (allocation_type == AllocationType::kCommit) {
#if XE_PLATFORM_MAC
      // Protect() can't query old access on macOS; mprotect the page-aligned
      // span directly to commit.
      size_t host_page = page_size();
      uintptr_t aligned_addr =
          reinterpret_cast<uintptr_t>(base_address) & ~(host_page - 1);
      uintptr_t end_addr = reinterpret_cast<uintptr_t>(base_address) + length;
      end_addr = (end_addr + host_page - 1) & ~(host_page - 1);
      if (mprotect(reinterpret_cast<void*>(aligned_addr),
                   end_addr - aligned_addr, prot) == 0) {
        return base_address;
      }
      return nullptr;
#else
      if (Protect(base_address, length, access)) {
        return base_address;
      }
      return nullptr;
#endif  // XE_PLATFORM_MAC
    }
#ifdef MAP_FIXED_NOREPLACE
    flags |= MAP_FIXED_NOREPLACE;
#endif
  }

  void* result = mmap(base_address, length, prot, flags, -1, 0);

  if (result == MAP_FAILED) {
    return nullptr;
  }

  // macOS has no MAP_FIXED_NOREPLACE, so a fixed request is only a hint. If the
  // kernel placed it elsewhere, fail rather than silently squatting a different
  // range (e.g. the indirection table landing on the code-cache window and
  // making the later code-cache MAP_FIXED fail with ENOMEM).
  if (base_address != nullptr && result != base_address) {
    munmap(result, length);
    return nullptr;
  }

  return result;
}

bool DeallocFixed(void* base_address, size_t length,
                  DeallocationType deallocation_type) {
  const auto region_begin = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t region_end =
      reinterpret_cast<uintptr_t>(base_address) + length;

  std::lock_guard guard(g_mapped_file_ranges_mutex);
  for (const auto& mapped_range : mapped_file_ranges) {
    if (region_begin >= mapped_range.region_begin &&
        region_end <= mapped_range.region_end) {
      switch (deallocation_type) {
        case DeallocationType::kDecommit:
          return Protect(base_address, length, PageAccess::kNoAccess);
        case DeallocationType::kRelease:
          return false;
        default:
          assert_unhandled_case(deallocation_type);
      }
    }
  }

  switch (deallocation_type) {
    case DeallocationType::kDecommit:
      return Protect(base_address, length, PageAccess::kNoAccess);
    case DeallocationType::kRelease:
      return munmap(base_address, length) == 0;
    default:
      assert_unhandled_case(deallocation_type);
  }
}

bool Protect(void* base_address, size_t length, PageAccess access,
             PageAccess* out_old_access) {
  if (out_old_access) {
    size_t length_copy = length;
    QueryProtect(base_address, length_copy, *out_old_access);
  }

  uint32_t prot = ToPosixProtectFlags(access);
  return mprotect(base_address, length, prot) == 0;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
#if XE_PLATFORM_MAC
  // macOS has no /proc/self/maps; query the kernel's VM map via mach_vm_region.
  // (The Linux path below would silently fail on macOS and leave access_out
  // uninitialized.)
  mach_vm_address_t region_address =
      reinterpret_cast<mach_vm_address_t>(base_address);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  kern_return_t kr = mach_vm_region(
      mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
      reinterpret_cast<vm_region_info_t>(&info), &info_count, &object_name);
  // mach_vm_region returns the region containing the address, or the next region
  // above it if the address lies in an unmapped hole.
  if (kr != KERN_SUCCESS ||
      region_address > reinterpret_cast<mach_vm_address_t>(base_address)) {
    // base_address is not within any mapped region.
    access_out = PageAccess::kNoAccess;
    return false;
  }
  vm_prot_t protection = info.protection;
  if (!(protection & VM_PROT_READ)) {
    access_out = PageAccess::kNoAccess;
  } else if (protection & VM_PROT_EXECUTE) {
    access_out = (protection & VM_PROT_WRITE) ? PageAccess::kExecuteReadWrite
                                              : PageAccess::kExecuteReadOnly;
  } else {
    access_out = (protection & VM_PROT_WRITE) ? PageAccess::kReadWrite
                                              : PageAccess::kReadOnly;
  }
  length = static_cast<size_t>(
      (region_address + region_size) -
      reinterpret_cast<mach_vm_address_t>(base_address));
  return true;
#else
  // No generic POSIX solution exists. The Linux solution should work on all
  // Linux kernel based OS, including Android.
  std::ifstream memory_maps;
  memory_maps.open("/proc/self/maps", std::ios_base::in);
  std::string maps_entry_string;

  while (std::getline(memory_maps, maps_entry_string)) {
    std::stringstream entry_stream(maps_entry_string);
    uintptr_t map_region_begin, map_region_end;
    char separator;
    char protection[5];  // 4 chars (e.g., "r-xp") + null terminator

    entry_stream >> std::hex >> map_region_begin >> separator >>
        map_region_end >> protection;

    if (map_region_begin <= reinterpret_cast<uintptr_t>(base_address) &&
        map_region_end > reinterpret_cast<uintptr_t>(base_address)) {
      length = map_region_end - reinterpret_cast<uintptr_t>(base_address);

      access_out = ToXeniaProtectFlags(protection);

      // Look at the next consecutive mappings
      while (std::getline(memory_maps, maps_entry_string)) {
        std::stringstream next_entry_stream(maps_entry_string);
        uintptr_t next_map_region_begin, next_map_region_end;
        char next_protection[5];  // 4 chars (e.g., "r-xp") + null terminator

        next_entry_stream >> std::hex >> next_map_region_begin >> separator >>
            next_map_region_end >> next_protection;
        if (map_region_end == next_map_region_begin &&
            access_out == ToXeniaProtectFlags(next_protection)) {
          length =
              next_map_region_end - reinterpret_cast<uintptr_t>(base_address);
          continue;
        }
        break;
      }

      memory_maps.close();
      return true;
    }
  }

  memory_maps.close();
  return false;
#endif  // XE_PLATFORM_MAC
}

// POSIX shared-memory object name for a mapping path. macOS limits these names
// to ~31 chars (PSHMNAMLEN) - much shorter than Linux - so descriptive names
// like "xenia_memory_<host ticks>" overflow it and shm_open fails. On macOS
// derive a short, deterministic name from the path so create and unlink stay
// consistent for the same path.
#if !XE_PLATFORM_ANDROID
static std::string ShmName(const std::filesystem::path& path) {
#if XE_PLATFORM_MAC
  char buf[24];
  size_t hash = std::hash<std::string>{}(path.string());
  std::snprintf(buf, sizeof(buf), "/xe%016zx", hash);
  return std::string(buf);
#else
  return ("/" / path).string();
#endif
}
#endif  // !XE_PLATFORM_ANDROID

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path& path,
                                          size_t length, PageAccess access,
                                          bool commit) {
#if XE_PLATFORM_ANDROID
  // TODO(Triang3l): Check if memfd can be used instead on API 30+.
  if (android_ASharedMemory_create_) {
    int sharedmem_fd = android_ASharedMemory_create_(path.c_str(), length);
    return sharedmem_fd >= 0 ? sharedmem_fd : kFileMappingHandleInvalid;
  }

  // Use /dev/ashmem on API versions below 26, which added ASharedMemory.
  // /dev/ashmem was disabled on API 29 for apps targeting it.
  // https://chromium.googlesource.com/chromium/src/+/master/third_party/ashmem/ashmem-dev.c
  int ashmem_fd = open("/" ASHMEM_NAME_DEF, O_RDWR);
  if (ashmem_fd < 0) {
    return kFileMappingHandleInvalid;
  }
  char ashmem_name[ASHMEM_NAME_LEN];
  strlcpy(ashmem_name, path.c_str(), xe::countof(ashmem_name));
  if (ioctl(ashmem_fd, ASHMEM_SET_NAME, ashmem_name) < 0 ||
      ioctl(ashmem_fd, ASHMEM_SET_SIZE, length) < 0) {
    close(ashmem_fd);
    return kFileMappingHandleInvalid;
  }
  return ashmem_fd;
#else
  int oflag;
  switch (access) {
    case PageAccess::kNoAccess:
      oflag = 0;
      break;
    case PageAccess::kReadOnly:
    case PageAccess::kExecuteReadOnly:
      oflag = O_RDONLY;
      break;
    case PageAccess::kReadWrite:
    case PageAccess::kExecuteReadWrite:
      oflag = O_RDWR;
      break;
    default:
      assert_always();
      return kFileMappingHandleInvalid;
  }
  oflag |= O_CREAT;
  std::string shm_name = ShmName(path);
  int ret = shm_open(shm_name.c_str(), oflag, 0777);
  if (ret < 0) {
    return kFileMappingHandleInvalid;
  }
  // Unlink the name at once: the descriptor and the views keep the object
  // alive, and the kernel frees it when the last of them goes, however the
  // process ends. A name left linked until exit outlived every SIGKILL and
  // crash, holding the guest memory the run had touched (hundreds of MB to
  // GB, compressed or swapped) until the next reboot.
  shm_unlink(shm_name.c_str());
  if (ftruncate(ret, length) < 0) {
    close(ret);
    return kFileMappingHandleInvalid;
  }
  return ret;
#endif
}

void CloseFileMappingHandle(FileMappingHandle handle,
                            const std::filesystem::path& path) {
  // The name was unlinked when the mapping was created.
  close(handle);
}

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length,
                  PageAccess access, size_t file_offset) {
  uint32_t prot = ToPosixProtectFlags(access);

  int flags = MAP_SHARED;
  if (base_address != nullptr) {
#if XE_PLATFORM_MAC
    // macOS has no MAP_FIXED_NOREPLACE; without MAP_FIXED the address hint is
    // ignored and the aliased guest views land at system-chosen addresses,
    // breaking the fixed view layout. The caller reserved this range first
    // (BaseHeap / ReserveFixedRangesEarly), so overlay it with MAP_FIXED.
    flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
    flags |= MAP_FIXED_NOREPLACE;
#endif
  }

  void* result = mmap(base_address, length, prot, flags, handle, file_offset);

  if (result != MAP_FAILED) {
    std::lock_guard guard(g_mapped_file_ranges_mutex);
    mapped_file_ranges.push_back(
        {reinterpret_cast<uintptr_t>(result),
         reinterpret_cast<uintptr_t>(result) + length});
    return result;
  }

  return nullptr;
}

bool UnmapFileView(FileMappingHandle handle, void* base_address,
                   size_t length) {
  std::lock_guard guard(g_mapped_file_ranges_mutex);
#if XE_PLATFORM_MAC
  // MapViewsMac maps the guest views into one larger reserved block, so a single
  // tracked range can be unmapped in parts. Support partial unmapping by
  // splitting the tracked range around the unmapped span (canary/macos).
  uintptr_t unmap_begin = reinterpret_cast<uintptr_t>(base_address);
  uintptr_t unmap_end = unmap_begin + length;
  for (auto mapped_range = mapped_file_ranges.begin();
       mapped_range != mapped_file_ranges.end(); ++mapped_range) {
    if (unmap_begin >= mapped_range->region_begin &&
        unmap_end <= mapped_range->region_end) {
      uintptr_t orig_begin = mapped_range->region_begin;
      uintptr_t orig_end = mapped_range->region_end;
      mapped_file_ranges.erase(mapped_range);
      if (orig_begin < unmap_begin) {
        mapped_file_ranges.push_back({orig_begin, unmap_begin});
      }
      if (unmap_end < orig_end) {
        mapped_file_ranges.push_back({unmap_end, orig_end});
      }
      return munmap(base_address, length) == 0;
    }
  }
  return munmap(base_address, length) == 0;
#else
  for (auto mapped_range = mapped_file_ranges.begin();
       mapped_range != mapped_file_ranges.end();) {
    if (mapped_range->region_begin ==
            reinterpret_cast<uintptr_t>(base_address) &&
        mapped_range->region_end ==
            reinterpret_cast<uintptr_t>(base_address) + length) {
      mapped_file_ranges.erase(mapped_range);
      return munmap(base_address, length) == 0;
    }
    ++mapped_range;
  }
  // TODO: Implement partial file unmapping.
  assert_always("Error: Partial unmapping of files not yet supported.");
  return munmap(base_address, length) == 0;
#endif
}

}  // namespace memory
}  // namespace xe
