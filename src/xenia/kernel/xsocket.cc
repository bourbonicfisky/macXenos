/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "src/xenia/kernel/xsocket.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <set>

#include "xenia/base/logging.h"
#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xam/xam_module.h"

#ifdef XE_PLATFORM_WIN32
// clang-format off
#include "xenia/base/platform_win.h"
#include <WS2tcpip.h>
#include <WinSock2.h>
// clang-format on
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace xe {
namespace kernel {

namespace {

// Ports below 1024 need privileges on some hosts, so Bind moves them up by
// kPrivilegedPortShift. The guest must not see the move: a title that binds
// port 1000 and then sends to port 1000 on its own (or the broadcast) address
// expects that socket to receive it, and expects its own port back in
// recvfrom/getsockname/accept. These are the host ports Bind produced.
constexpr uint16_t kPrivilegedPortShift = 10000;
std::mutex remapped_ports_mutex;
std::set<uint16_t> remapped_ports;

bool IsRemappedHostPort(uint16_t host_port) {
  std::lock_guard<std::mutex> lock(remapped_ports_mutex);
  return remapped_ports.count(host_port) != 0;
}

// Guest destination port -> host port: a privileged port some local socket
// was moved from, or any privileged port on a loopback or broadcast address
// (another instance on this host made the same move).
uint16_t HostDestinationPort(uint16_t guest_port, uint32_t guest_addr) {
  if (!guest_port || guest_port >= 1024) {
    return guest_port;
  }
  const uint16_t shifted = uint16_t(guest_port + kPrivilegedPortShift);
  const bool loopback = (guest_addr >> 24) == 127;
  const bool broadcast = guest_addr == 0xFFFFFFFFu;
  if (loopback || broadcast || IsRemappedHostPort(shifted)) {
    return shifted;
  }
  return guest_port;
}

// Host port seen in a peer or local address -> the port the guest asked for.
uint16_t GuestPort(uint16_t host_port) {
  if (host_port > kPrivilegedPortShift &&
      host_port < kPrivilegedPortShift + 1024 &&
      IsRemappedHostPort(host_port)) {
    return uint16_t(host_port - kPrivilegedPortShift);
  }
  return host_port;
}

// N_XSOCKADDR_IN holds the port and address as big-endian values; build a
// host sockaddr_in from them (network order, BSD sin_len set).
void ToNativeAddress(const N_XSOCKADDR_IN& in, sockaddr_in* out) {
  std::memset(out, 0, sizeof(*out));
#if XE_PLATFORM_MAC
  out->sin_len = sizeof(sockaddr_in);
#endif
  out->sin_family = AF_INET;
  const uint32_t addr = uint32_t(in.sin_addr);
  out->sin_addr.s_addr = htonl(addr);
  out->sin_port = htons(HostDestinationPort(uint16_t(in.sin_port), addr));
}

void FromNativeAddress(const sockaddr_in& in, N_XSOCKADDR_IN* out) {
  out->sin_family = 2;  // AF_INET
  out->sin_addr = ntohl(in.sin_addr.s_addr);
  out->sin_port = GuestPort(ntohs(in.sin_port));
  std::memset(out->x_sin_zero, 0, sizeof(out->x_sin_zero));
}

}  // namespace

// Translate socket options to native
// Note:
// SO_DONTLINGER = ~SO_LINGER
// SO_EXCLUSIVEADDRUSE = ~SO_REUSEADDR
// TODO: Check SO_DONTLINGER and SO_EXCLUSIVEADDRUSE usage on linux
const std::map<uint32_t, uint32_t> supported_socket_options = {
    {0x0004, SO_REUSEADDR}, {0x0020, SO_BROADCAST}, {0x0080, SO_LINGER},
    {0x1001, SO_SNDBUF},    {0x1002, SO_RCVBUF},    {0x1005, SO_SNDTIMEO},
    {0x1006, SO_RCVTIMEO},  {~0x0080, ~SO_LINGER},  {~0x0004, ~SO_REUSEADDR}};

// Translate socket TCP options to native
const std::map<uint32_t, uint32_t> supported_tcp_options = {
    {0x0001, TCP_NODELAY}};

// Translate socket levels to native
const std::map<uint32_t, uint32_t> supported_levels = {{0xFFFF, SOL_SOCKET},
                                                       {0x6, IPPROTO_TCP}};

// Translate ioctl commands to native
const std::map<uint32_t, uint32_t> supported_controls = {
    {0x8004667E, FIONBIO}, {0x4004667F, FIONREAD}};

XSocket::XSocket(KernelState* kernel_state)
    : XObject(kernel_state, kObjectType) {}

XSocket::XSocket(KernelState* kernel_state, uint64_t native_handle)
    : XObject(kernel_state, kObjectType), native_handle_(native_handle) {}

XSocket::~XSocket() { Close(); }

X_STATUS XSocket::Initialize(AddressFamily af, Type type, Protocol proto) {
  af_ = af;
  type_ = type;
  proto_ = proto;

  if (proto == Protocol::XE_IPPROTO_VDP) {
    // VDP is a layer on top of UDP.
    proto = Protocol::XE_IPPROTO_UDP;
  }

  native_handle_ = socket(af, type, proto);
  if (native_handle_ == -1) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Close() {
#if XE_PLATFORM_WIN32
  int ret = closesocket(native_handle_);
#else
  int ret = close(native_handle_);
#endif

  if (ret != 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::GetOption(uint32_t level, uint32_t optname, void* optval_ptr,
                            uint32_t* optlen) {
  int ret =
      getsockopt(native_handle_, level, optname, static_cast<char*>(optval_ptr),
                 reinterpret_cast<socklen_t*>(optlen));
  if (ret < 0) {
    // TODO: WSAGetLastError()
    return X_STATUS_UNSUCCESSFUL;
  }
  return X_STATUS_SUCCESS;
}
X_STATUS XSocket::SetOption(uint32_t level, uint32_t optname, void* optval_ptr,
                            uint32_t optlen) {
  if (level == 0xFFFF && (optname == 0x5801 || optname == 0x5802)) {
    // Disable socket encryption
    secure_ = false;
    return X_STATUS_SUCCESS;
  }

  int native_level = level;

  assert_false(!supported_levels.contains(level));

  if (supported_levels.contains(level)) {
    native_level = supported_levels.at(level);
  }

  int native_optname = optname;

  if (level == 0xFFFF) {
    assert_false(!supported_socket_options.contains(optname));

    if (supported_socket_options.contains(optname)) {
      native_optname = supported_socket_options.at(optname);
    }
  }

  if (level == IPPROTO_TCP) {
    assert_false(!supported_tcp_options.contains(optname));

    if (supported_tcp_options.contains(optname)) {
      native_optname = supported_tcp_options.at(optname);
    }
  }

  int ret = setsockopt(native_handle_, native_level, native_optname,
                       static_cast<char*>(optval_ptr), optlen);
  if (ret < 0) {
    // TODO: WSAGetLastError()
    XELOGE("XSocket::SetOption: failed with error {:08X}", GetLastWSAError());
    return X_STATUS_UNSUCCESSFUL;
  }

  // SO_BROADCAST
  if (level == 0xFFFF && optname == 0x0020) {
    broadcast_socket_ = true;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::IOControl(uint32_t cmd, uint8_t* arg_ptr) {
#ifdef XE_PLATFORM_WIN32
  int ret = ioctlsocket(native_handle_, cmd, (u_long*)arg_ptr);
  if (ret < 0) {
    // TODO: Get last error
    return X_STATUS_UNSUCCESSFUL;
  }
  return X_STATUS_SUCCESS;
#else
  assert_false(!supported_controls.contains(cmd));

  // The request must stay an unsigned long: FIONBIO (0x8004667E) has the top
  // bit set, and narrowing it through `int` sign-extends to
  // 0xFFFFFFFF8004667E. Linux truncates the syscall argument to 32 bits so
  // that went unnoticed there, but XNU compares the full 64-bit value and
  // fails with ENXIO - Gears of War 2/3 then can't make their listen socket
  // non-blocking, UE3's Listen() fails and the campaign map reloads with
  // "?failed" (an empty persistent level: the black "void").
  unsigned long native_cmd = supported_controls.contains(cmd)
                                 ? supported_controls.at(cmd)
                                 : static_cast<unsigned long>(cmd);

  // The guest argument is a big-endian u_long; the host ioctl reads and
  // writes a native int, so swap on both sides (FIONBIO in, FIONREAD out).
  uint32_t value = xe::load_and_swap<uint32_t>(arg_ptr);
  int ret = ioctl(native_handle_, native_cmd, &value);

  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  xe::store_and_swap<uint32_t>(arg_ptr, value);
  return X_STATUS_SUCCESS;
#endif
}

X_STATUS XSocket::Connect(N_XSOCKADDR* name, int name_len) {
  int ret;
  if (name->address_family == 2 && name_len >= 8) {
    // AF_INET: sa_data holds the big-endian port then the address.
    N_XSOCKADDR_IN in;
    in.sin_family = 2;
    in.sin_port = uint16_t((uint8_t(name->sa_data[0]) << 8) |
                           uint8_t(name->sa_data[1]));
    in.sin_addr = (uint32_t(uint8_t(name->sa_data[2])) << 24) |
                  (uint32_t(uint8_t(name->sa_data[3])) << 16) |
                  (uint32_t(uint8_t(name->sa_data[4])) << 8) |
                  uint32_t(uint8_t(name->sa_data[5]));
    sockaddr_in native;
    ToNativeAddress(in, &native);
    ret = connect(native_handle_, (sockaddr*)&native, sizeof(native));
  } else {
    ret = connect(native_handle_, (sockaddr*)name, name_len);
  }
  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Bind(N_XSOCKADDR_IN* name, int name_len) {
  // On Linux and Windows (when running under Wine), ports < 1024 require root
  // privileges. Remap to port + 10000 to avoid privilege issues.
  // Note: sin_port is xe::be<uint16_t> which automatically handles endianness,
  // so we use it directly without ntohs/htons.
  const uint16_t original_port = uint16_t(name->sin_port);
  uint16_t host_port = original_port;
  if (original_port && original_port < 1024) {
    host_port = uint16_t(original_port + kPrivilegedPortShift);
    XELOGW("XSocket::Bind: port {} requires privileges, remapping to port {}",
           original_port, host_port);
  }

  sockaddr_in native = {};
#if XE_PLATFORM_MAC
  native.sin_len = sizeof(sockaddr_in);
#endif
  native.sin_family = AF_INET;
  native.sin_addr.s_addr = htonl(uint32_t(name->sin_addr));
  native.sin_port = htons(host_port);
  int ret = bind(native_handle_, (sockaddr*)&native, sizeof(native));

  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  if (host_port != original_port) {
    std::lock_guard<std::mutex> lock(remapped_ports_mutex);
    remapped_ports.insert(host_port);
  }
  bound_ = true;
  bound_port_ = host_port;

  return X_STATUS_SUCCESS;
}

X_STATUS XSocket::Listen(int backlog) {
  int ret = listen(native_handle_, backlog);
  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  return X_STATUS_SUCCESS;
}

object_ref<XSocket> XSocket::Accept(N_XSOCKADDR* name, int* name_len) {
  sockaddr n_sockaddr;
  socklen_t n_name_len = sizeof(sockaddr);
  uintptr_t ret = accept(native_handle_, &n_sockaddr, &n_name_len);
  if (ret == -1) {
    std::memset(name, 0, *name_len);
    *name_len = 0;
    return nullptr;
  }

  if (n_sockaddr.sa_family == AF_INET) {
    N_XSOCKADDR_IN in;
    FromNativeAddress(*reinterpret_cast<sockaddr_in*>(&n_sockaddr), &in);
    name->address_family = 2;
    // sa_data: big-endian port, then the big-endian address.
    const uint16_t port = uint16_t(in.sin_port);
    const uint32_t addr = uint32_t(in.sin_addr);
    std::memset(name->sa_data, 0, sizeof(name->sa_data));
    name->sa_data[0] = char(port >> 8);
    name->sa_data[1] = char(port);
    name->sa_data[2] = char(addr >> 24);
    name->sa_data[3] = char(addr >> 16);
    name->sa_data[4] = char(addr >> 8);
    name->sa_data[5] = char(addr);
    *name_len = sizeof(N_XSOCKADDR_IN);
  } else {
    std::memcpy(name, &n_sockaddr, n_name_len);
    *name_len = n_name_len;
  }

  // Create a kernel object to represent the new socket, and copy parameters
  // over.
  auto socket = object_ref<XSocket>(new XSocket(kernel_state_, ret));
  socket->af_ = af_;
  socket->type_ = type_;
  socket->proto_ = proto_;

  return socket;
}

int XSocket::Shutdown(int how) { return shutdown(native_handle_, how); }

int XSocket::Recv(uint8_t* buf, uint32_t buf_len, uint32_t flags) {
  return recv(native_handle_, reinterpret_cast<char*>(buf), buf_len, flags);
}

int XSocket::RecvFrom(uint8_t* buf, uint32_t buf_len, uint32_t flags,
                      N_XSOCKADDR_IN* from, uint32_t* from_len) {
  // Pop from secure packets first
  // TODO(DrChat): Enable when I commit XNet
  /*
  {
    std::lock_guard<std::mutex> lock(incoming_packet_mutex_);
    if (incoming_packets_.size()) {
      packet* pkt = (packet*)incoming_packets_.front();
      int data_len = pkt->data_len;
      std::memcpy(buf, pkt->data, std::min((uint32_t)pkt->data_len, buf_len));

      from->sin_family = 2;
      from->sin_addr = pkt->src_ip;
      from->sin_port = pkt->src_port;

      incoming_packets_.pop();
      uint8_t* pkt_ui8 = (uint8_t*)pkt;
      delete[] pkt_ui8;

      return data_len;
    }
  }
  */

  sockaddr_in nfrom = {};
  socklen_t nfromlen = sizeof(sockaddr_in);
  int ret = recvfrom(native_handle_, reinterpret_cast<char*>(buf), buf_len,
                     flags, (sockaddr*)&nfrom, &nfromlen);
  if (from && ret >= 0) {
    FromNativeAddress(nfrom, from);
  }

  if (from_len) {
    *from_len = nfromlen;
  }

  return ret;
}

int XSocket::Send(const uint8_t* buf, uint32_t buf_len, uint32_t flags) {
  return send(native_handle_, reinterpret_cast<const char*>(buf), buf_len,
              flags);
}

int XSocket::SendTo(uint8_t* buf, uint32_t buf_len, uint32_t flags,
                    N_XSOCKADDR_IN* to, uint32_t to_len) {
  // Send 2 copies of the packet: One to XNet (for network security) and an
  // unencrypted copy for other Xenia hosts.
  // TODO(DrChat): Enable when I commit XNet.
  /*
  auto xam = kernel_state()->GetKernelModule<xam::XamModule>("xam.xex");
  auto xnet = xam->xnet();
  if (xnet) {
    xnet->SendPacket(this, to, buf, buf_len);
  }
  */

  sockaddr_in nto;
  if (to) {
    ToNativeAddress(*to, &nto);
  }

  return sendto(native_handle_, reinterpret_cast<char*>(buf), buf_len, flags,
                to ? (sockaddr*)&nto : nullptr, to ? sizeof(nto) : 0);
}

bool XSocket::QueuePacket(uint32_t src_ip, uint16_t src_port,
                          const uint8_t* buf, size_t len) {
  packet* pkt = reinterpret_cast<packet*>(new uint8_t[sizeof(packet) + len]);
  pkt->src_ip = src_ip;
  pkt->src_port = src_port;

  pkt->data_len = (uint16_t)len;
  std::memcpy(pkt->data, buf, len);

  std::lock_guard<std::mutex> lock(incoming_packet_mutex_);
  incoming_packets_.push((uint8_t*)pkt);

  // TODO: Limit on number of incoming packets?
  return true;
}

X_STATUS XSocket::GetSockName(uint8_t* buf, int* buf_len) {
  sockaddr_storage ss = {};
  socklen_t len = sizeof(ss);
  int ret = getsockname(native_handle_, (sockaddr*)&ss, &len);
  if (ret < 0) {
    return X_STATUS_UNSUCCESSFUL;
  }

  if (ss.ss_family == AF_INET && *buf_len >= int(sizeof(XSOCKADDR_IN))) {
    // The guest layout: big-endian family, port and address.
    N_XSOCKADDR_IN in;
    FromNativeAddress(*reinterpret_cast<sockaddr_in*>(&ss), &in);
    XSOCKADDR_IN guest = {};
    guest.sin_family = in.sin_family;
    guest.sin_port = uint16_t(in.sin_port);
    guest.sin_addr = uint32_t(in.sin_addr);
    std::memcpy(buf, &guest, sizeof(guest));
    *buf_len = sizeof(guest);
    return X_STATUS_SUCCESS;
  }

  const int copy = std::min<int>(*buf_len, int(len));
  std::memcpy(buf, &ss, copy);
  *buf_len = copy;
  return X_STATUS_SUCCESS;
}

uint32_t XSocket::GetLastWSAError() const {
#ifdef XE_PLATFORM_WIN32
  return WSAGetLastError();
#else
  // Titles compare against the Winsock codes the console returns; the raw
  // POSIX errno means nothing to them. Modern Warfare 2 polls a non-blocking
  // socket, got 35 (EAGAIN on macOS) instead of WSAEWOULDBLOCK, reported
  // "Failed to recvfrom() socket, error UNKNOWN ERROR" and relaunched itself
  // until the hop cap.
  const int error = errno;
  switch (error) {
    case EINTR: return 10004;         // WSAEINTR
    case EBADF: return 10009;         // WSAEBADF
    case EACCES: return 10013;        // WSAEACCES
    case EFAULT: return 10014;        // WSAEFAULT
    case EINVAL: return 10022;        // WSAEINVAL
    case EMFILE: return 10024;        // WSAEMFILE
#if EAGAIN != EWOULDBLOCK
    case EAGAIN:
#endif
    case EWOULDBLOCK: return 10035;   // WSAEWOULDBLOCK
    case EINPROGRESS: return 10036;   // WSAEINPROGRESS
    case EALREADY: return 10037;      // WSAEALREADY
    case ENOTSOCK: return 10038;      // WSAENOTSOCK
    case EDESTADDRREQ: return 10039;  // WSAEDESTADDRREQ
    case EMSGSIZE: return 10040;      // WSAEMSGSIZE
    case EPROTOTYPE: return 10041;    // WSAEPROTOTYPE
    case ENOPROTOOPT: return 10042;   // WSAENOPROTOOPT
    case EPROTONOSUPPORT: return 10043;  // WSAEPROTONOSUPPORT
    case ESOCKTNOSUPPORT: return 10044;  // WSAESOCKTNOSUPPORT
    case EOPNOTSUPP: return 10045;    // WSAEOPNOTSUPP
    case EPFNOSUPPORT: return 10046;  // WSAEPFNOSUPPORT
    case EAFNOSUPPORT: return 10047;  // WSAEAFNOSUPPORT
    case EADDRINUSE: return 10048;    // WSAEADDRINUSE
    case EADDRNOTAVAIL: return 10049; // WSAEADDRNOTAVAIL
    case ENETDOWN: return 10050;      // WSAENETDOWN
    case ENETUNREACH: return 10051;   // WSAENETUNREACH
    case ENETRESET: return 10052;     // WSAENETRESET
    case ECONNABORTED: return 10053;  // WSAECONNABORTED
    case ECONNRESET: return 10054;    // WSAECONNRESET
    case ENOBUFS: return 10055;       // WSAENOBUFS
    case EISCONN: return 10056;       // WSAEISCONN
    case ENOTCONN: return 10057;      // WSAENOTCONN
    case ESHUTDOWN: return 10058;     // WSAESHUTDOWN
    case ETOOMANYREFS: return 10059;  // WSAETOOMANYREFS
    case ETIMEDOUT: return 10060;     // WSAETIMEDOUT
    case ECONNREFUSED: return 10061;  // WSAECONNREFUSED
    case ELOOP: return 10062;         // WSAELOOP
    case ENAMETOOLONG: return 10063;  // WSAENAMETOOLONG
    case EHOSTDOWN: return 10064;     // WSAEHOSTDOWN
    case EHOSTUNREACH: return 10065;  // WSAEHOSTUNREACH
    default: return uint32_t(error);
  }
#endif
}

}  // namespace kernel
}  // namespace xe
