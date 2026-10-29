/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/exception_handler.h"

#include <atomic>
#include <chrono>
#include <thread>

#include "third_party/catch/include/catch.hpp"

#include "xenia/base/memory.h"
#include "xenia/base/platform.h"

namespace xe {
namespace base {
namespace test {

#if XE_PLATFORM_MAC

namespace {
// Claims faults on one page by making it writable, as the write-watch handler
// does for guest memory.
std::atomic<uint8_t*> claimed_page{nullptr};
bool ClaimPageHandler(Exception* ex, void* data) {
  uint8_t* page = claimed_page.load();
  if (ex->code() != Exception::Code::kAccessViolation || !page ||
      ex->fault_address() < uint64_t(page) ||
      ex->fault_address() >= uint64_t(page) + memory::page_size()) {
    return false;
  }
  memory::Protect(page, memory::page_size(), memory::PageAccess::kReadWrite);
  return true;
}
}  // namespace

TEST_CASE("unclaimed_transient_fault_keeps_handler", "[exception_handler]") {
  size_t page_size = memory::page_size();
  auto pages = static_cast<uint8_t*>(memory::AllocFixed(
      nullptr, page_size * 2, memory::AllocationType::kReserveCommit,
      memory::PageAccess::kNoAccess));
  REQUIRE(pages);
  ExceptionHandler::Install(ClaimPageHandler, nullptr);

  // No handler claims the first page; another thread makes it writable while
  // the faulting write is being retried, like a page-in that fails once.
  std::thread unprotect([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    memory::Protect(pages, page_size, memory::PageAccess::kReadWrite);
  });
  *reinterpret_cast<volatile uint8_t*>(pages) = 1;
  unprotect.join();
  REQUIRE(pages[0] == 1);

  // The process's handler must still be installed: a claimed fault after the
  // transient one is resolved instead of crashing.
  claimed_page = pages + page_size;
  *reinterpret_cast<volatile uint8_t*>(pages + page_size) = 2;
  REQUIRE(pages[page_size] == 2);

  claimed_page = nullptr;
  ExceptionHandler::Uninstall(ClaimPageHandler, nullptr);
  memory::DeallocFixed(pages, 0, memory::DeallocationType::kRelease);
}

#endif  // XE_PLATFORM_MAC

}  // namespace test
}  // namespace base
}  // namespace xe
