/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstring>

#include "third_party/catch/include/catch.hpp"
#include "xenia/kernel/kernel.h"

namespace xe {
namespace kernel {
namespace test {

TEST_CASE("XOverlappedCompleteFields clears InternalContext on completion",
          "[kernel][overlapped]") {
  XAM_OVERLAPPED overlapped = {};
  constexpr X_HANDLE kRequesterThread = 0xF8000004u;

  SECTION("Successful completion with no event/APC leaves structure all-zero") {
    XOverlappedSetResult(&overlapped, X_ERROR_IO_PENDING);
    XOverlappedSetContext(&overlapped, kRequesterThread);

    const X_HANDLE ctx =
        XOverlappedCompleteFields(&overlapped, X_ERROR_SUCCESS, 0, 0);
    CHECK(ctx == kRequesterThread);
    CHECK(XOverlappedGetResult(&overlapped) == X_ERROR_SUCCESS);
    CHECK(XOverlappedGetLength(&overlapped) == 0);
    CHECK(XOverlappedGetContext(&overlapped) == 0);

    // Titles like F.E.A.R. 3 memcmp all 28 bytes of XOVERLAPPED against zero
    // to detect whether an async profile operation is still in flight.
    XAM_OVERLAPPED zeroed = {};
    CHECK(std::memcmp(&overlapped, &zeroed, sizeof(overlapped)) == 0);
  }

  SECTION("Error/APC completion preserves caller fields and clears context") {
    overlapped.event = 0xF8000010u;
    overlapped.completion_routine = 0x82100000u;
    overlapped.completion_context = 0x44A6E088u;
    XOverlappedSetResult(&overlapped, X_ERROR_IO_PENDING);
    XOverlappedSetContext(&overlapped, kRequesterThread);

    const X_HANDLE ctx = XOverlappedCompleteFields(
        &overlapped, X_ERROR_FUNCTION_FAILED, X_E_FAIL, 0x1234u);
    CHECK(ctx == kRequesterThread);
    CHECK(XOverlappedGetResult(&overlapped) == X_ERROR_FUNCTION_FAILED);
    CHECK(XOverlappedGetLength(&overlapped) == 0x1234u);
    CHECK(XOverlappedGetContext(&overlapped) == 0);
    CHECK(XOverlappedGetEvent(&overlapped) == 0xF8000010u);
    CHECK(XOverlappedGetCompletionRoutine(&overlapped) == 0x82100000u);
    CHECK(XOverlappedGetCompletionContext(&overlapped) == 0x44A6E088u);
    CHECK(static_cast<uint32_t>(overlapped.extended_error) == X_E_FAIL);
  }
}

}  // namespace test
}  // namespace kernel
}  // namespace xe
