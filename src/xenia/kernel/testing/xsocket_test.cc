/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Xenia Canary. All rights reserved.                          *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xsocket.h"

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/byte_order.h"

namespace {
constexpr auto kGuestAfInet = xe::kernel::XSocket::AddressFamily::AF_INET;
constexpr auto kGuestSockDgram = xe::kernel::XSocket::Type::SOCK_DGRAM;
}  // namespace

#ifndef XE_PLATFORM_WIN32
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/time.h>
#endif

namespace xe {
namespace kernel {
namespace test {

namespace {

constexpr uint32_t kGuestSolSocket = 0xFFFF;
constexpr uint32_t kGuestSoReuseAddr = 0x0004;
constexpr uint32_t kGuestSoBroadcast = 0x0020;
constexpr uint32_t kGuestSoLinger = 0x0080;
constexpr uint32_t kGuestSoSndBuf = 0x1001;
constexpr uint32_t kGuestSoRcvBuf = 0x1002;
constexpr uint32_t kGuestSoSndTimeo = 0x1005;
constexpr uint32_t kGuestSoRcvTimeo = 0x1006;

}  // namespace

TEST_CASE("XSocket::SetOption translates guest option widths and endianness",
          "[kernel][xsocket]") {
  auto sock = make_object<XSocket>(nullptr);
  REQUIRE(sock->Initialize(kGuestAfInet, kGuestSockDgram,
                           XSocket::Protocol::XE_IPPROTO_UDP) ==
          X_STATUS_SUCCESS);

  SECTION("1-byte boolean SO_BROADCAST (F.E.A.R. 3)") {
    uint8_t on = 1;
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoBroadcast, &on,
                            sizeof(on)) == X_STATUS_SUCCESS);
#ifndef XE_PLATFORM_WIN32
    int host_val = 0;
    socklen_t host_len = sizeof(host_val);
    REQUIRE(getsockopt(static_cast<int>(sock->native_handle()), SOL_SOCKET,
                       SO_BROADCAST, &host_val, &host_len) == 0);
    CHECK(host_val != 0);
#endif
  }

  SECTION("1-byte boolean SO_REUSEADDR") {
    uint8_t on = 1;
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoReuseAddr, &on,
                            sizeof(on)) == X_STATUS_SUCCESS);
  }

  SECTION("4-byte big-endian SO_BROADCAST") {
    uint32_t on_be = xe::byte_swap<uint32_t>(1);
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoBroadcast, &on_be,
                            sizeof(on_be)) == X_STATUS_SUCCESS);
#ifndef XE_PLATFORM_WIN32
    int host_val = 0;
    socklen_t host_len = sizeof(host_val);
    REQUIRE(getsockopt(static_cast<int>(sock->native_handle()), SOL_SOCKET,
                       SO_BROADCAST, &host_val, &host_len) == 0);
    CHECK(host_val != 0);
#endif
  }

  SECTION("4-byte big-endian SO_RCVTIMEO and SO_SNDTIMEO (ms -> timeval)") {
    uint32_t ms_be = xe::byte_swap<uint32_t>(1250);
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoRcvTimeo, &ms_be,
                            sizeof(ms_be)) == X_STATUS_SUCCESS);
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoSndTimeo, &ms_be,
                            sizeof(ms_be)) == X_STATUS_SUCCESS);
#ifndef XE_PLATFORM_WIN32
    struct timeval tv = {};
    socklen_t tv_len = sizeof(tv);
    REQUIRE(getsockopt(static_cast<int>(sock->native_handle()), SOL_SOCKET,
                       SO_RCVTIMEO, &tv, &tv_len) == 0);
    CHECK(tv.tv_sec == 1);
    CHECK(tv.tv_usec == 250000);
#endif
  }

  SECTION("4-byte big-endian SO_SNDBUF and SO_RCVBUF") {
    uint32_t sz_be = xe::byte_swap<uint32_t>(32768);
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoSndBuf, &sz_be,
                            sizeof(sz_be)) == X_STATUS_SUCCESS);
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoRcvBuf, &sz_be,
                            sizeof(sz_be)) == X_STATUS_SUCCESS);
  }

  SECTION("4-byte big-endian SO_LINGER") {
    uint8_t linger_be[4] = {0x00, 0x01, 0x00, 0x05};  // onoff=1, linger=5s
    REQUIRE(sock->SetOption(kGuestSolSocket, kGuestSoLinger, linger_be,
                            sizeof(linger_be)) == X_STATUS_SUCCESS);
  }
}

TEST_CASE("XSocket::EventSelect switches socket to non-blocking mode",
          "[kernel][xsocket]") {
  auto sock = make_object<XSocket>(nullptr);
  REQUIRE(sock->Initialize(kGuestAfInet, kGuestSockDgram,
                           XSocket::Protocol::XE_IPPROTO_UDP) ==
          X_STATUS_SUCCESS);

  REQUIRE(sock->EventSelect(0xF8000128u, 0x00000001u /* FD_READ */) ==
          X_STATUS_SUCCESS);
#ifndef XE_PLATFORM_WIN32
  int flags = fcntl(static_cast<int>(sock->native_handle()), F_GETFL, 0);
  REQUIRE(flags >= 0);
  CHECK((flags & O_NONBLOCK) != 0);
#endif

  // Clearing events (mask == 0) must also succeed.
  REQUIRE(sock->EventSelect(X_INVALID_HANDLE_VALUE, 0) == X_STATUS_SUCCESS);
}

}  // namespace test
}  // namespace kernel
}  // namespace xe
