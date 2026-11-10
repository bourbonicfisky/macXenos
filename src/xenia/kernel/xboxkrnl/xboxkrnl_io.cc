/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/kernel/info/file.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_ob.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/kernel/xevent.h"
#include "xenia/kernel/xfile.h"
#include "xenia/kernel/xiocompletion.h"
#include "xenia/kernel/xsymboliclink.h"
#include "xenia/kernel/xthread.h"
#include "xenia/vfs/device.h"
#include "xenia/xbox.h"

DEFINE_bool(
    io_ledger_diag, false,
    "Diagnostic (default off): async-I/O completion ledger for the DOA5 "
    "load-stall hunt. Logs NtCreateFile opens and NtReadFile issues (path, "
    "offset, length, sync, event, APC routine/context, io_status_block, "
    "thid/LR, bytes/status), tracks completion-APC queued-vs-delivered, and on "
    "the spin detector dumps outstanding (queued-but-undelivered) completion "
    "APCs older than a threshold. Goal: find an issued read whose completion is "
    "never delivered, or prove all I/O completed (then the queue waits on a "
    "non-I/O job). Revert before any fix.",
    "Kernel");

namespace xe {
namespace kernel {
namespace xboxkrnl {

// ── io_ledger_diag ──────────────────────────────────────────────────────────
namespace {
struct IoLedgerEntry {
  uint64_t seq;
  uint32_t apc_routine, apc_context, iosb, thid, lr;
  uint64_t offset;
  uint32_t length;
  uint64_t queued_tick;
  std::string path;
};
struct RecentRead {
  uint64_t seq;
  std::string path;
  uint64_t offset;
  uint32_t length, bytes, thid, lr;
  uint64_t tick;
};
struct IoLedger {
  std::mutex mu;
  uint64_t read_count = 0, open_count = 0;
  uint64_t apc_queued = 0, apc_delivered = 0;
  uint64_t last_read_tick = 0;
  uint64_t seq = 0;
  std::unordered_map<uint32_t, IoLedgerEntry> outstanding;  // key: apc_context
  // Recent-reads ring (progress-vs-loop discriminator during the stall).
  static constexpr uint32_t kRing = 32;
  std::vector<RecentRead> recent{kRing};
  uint32_t recent_head = 0;
  // Snapshot at previous dump, to report per-dump deltas.
  uint64_t reads_at_last_dump = 0, opens_at_last_dump = 0;
  // Reader threads (the loader) → their last read, for the inter-read wait probe.
  struct ReaderInfo {
    uint64_t last_seq;
    uint64_t last_off;
    uint64_t last_tick;
    std::string last_path;
  };
  std::unordered_map<uint32_t, ReaderInfo> readers;  // key: thid
  // Gate event (what the loader waits on) + its signaler (the consumer).
  uint32_t gate_obj_addr = 0;       // guest X_KEVENT addr of the loader's gate
  uint32_t gate_signaler_thid = 0;  // who KeSetEvent's the gate (the consumer)
  uint64_t gate_last_signal_tick = 0;
  // Passive wait registry: in-progress waits, keyed by thid (one wait/thread).
  struct WaitEntry {
    const char* api;
    uint32_t lr, wait_type, alertable, count;
    uint64_t timeout, enter_tick;
    bool by_handle;
    uint32_t handles[8];  // capped
  };
  std::unordered_map<uint32_t, WaitEntry> waits;  // key: thid
  // Per-object signal summary (cheap O(1) update; no full history ring).
  struct SignalSummary {
    uint64_t set_count = 0, last_set_tick = 0;
    uint32_t last_set_thid = 0, last_set_lr = 0;
    const char* last_set_op = "";
    uint64_t reset_count = 0, last_reset_tick = 0;
    uint32_t last_reset_thid = 0;
  };
  std::mutex sig_mu;  // separate lock: event Set is high-frequency
  std::unordered_map<uint32_t, SignalSummary> sigsum;  // key: obj guest addr
};
// Intentionally leaked to avoid static-destruction races at shutdown.
IoLedger* io_ledger() {
  static IoLedger* g = new IoLedger();
  return g;
}
void IoLedgerOnOpen(const std::string& path, uint32_t status) {
  if (!cvars::io_ledger_diag) return;
  auto* L = io_ledger();
  std::lock_guard<std::mutex> lk(L->mu);
  L->open_count++;
  if (L->open_count <= 48) {
    XELOGI("[io-ledger] OPEN #{} status={:08X} path={}", L->open_count, status,
           path);
  }
}
void IoLedgerOnRead(const std::string& path, uint64_t offset, uint32_t length,
                    bool sync, uint32_t ev_handle, uint32_t apc_routine,
                    uint32_t apc_context, uint32_t iosb, uint32_t status,
                    uint32_t bytes, bool apc_queued, bool ev_signaled,
                    uint32_t thid, uint32_t lr) {
  if (!cvars::io_ledger_diag) return;
  auto* L = io_ledger();
  std::lock_guard<std::mutex> lk(L->mu);
  L->read_count++;
  L->last_read_tick = Clock::QueryHostUptimeMillis();
  const uint64_t s = ++L->seq;
  L->recent[L->recent_head % IoLedger::kRing] =
      RecentRead{s, path, offset, length, bytes, thid, lr, L->last_read_tick};
  L->recent_head++;
  L->readers[thid] =
      IoLedger::ReaderInfo{s, offset, L->last_read_tick, path};
  if (apc_queued && apc_context) {
    L->apc_queued++;
    L->outstanding[apc_context] =
        IoLedgerEntry{s,      apc_routine, apc_context,        iosb,
                      thid,   lr,          offset,             length,
                      L->last_read_tick, path};
  }
  if (L->read_count <= 48 || (L->read_count % 4000) == 0) {
    XELOGI(
        "[io-ledger] READ #{} seq={} path={} off={:X} len={:X} sync={} "
        "ev={:08X} apc_r={:08X} apc_c={:08X} iosb={:08X} st={:08X} bytes={:X} "
        "apcq={} evsig={} thid={:08X} lr={:08X}",
        L->read_count, s, path, offset, length, sync, ev_handle, apc_routine,
        apc_context, iosb, status, bytes, apc_queued, ev_signaled, thid, lr);
  }
}
}  // namespace

void IoLedgerOnLoaderWait(const char* api, uint32_t handle_or_ptr, bool by_handle,
                          uint32_t count, uint64_t timeout, uint32_t alertable,
                          uint64_t dur_ms, uint32_t result, uint32_t lr,
                          uint32_t thid) {
  if (!cvars::io_ledger_diag || dur_ms <= 100) return;
  auto* L = io_ledger();
  // Resolve the waited object's type + guest address (done only for the rare
  // long wait, outside the ledger lock).
  uint32_t type = 0xFFFFFFFFu, obj_addr = 0;
  object_ref<XObject> obj;
  if (by_handle) {
    obj = kernel_state()->object_table()->LookupObject<XObject>(handle_or_ptr);
  } else if (handle_or_ptr) {
    auto* p = kernel_memory()->TranslateVirtual(handle_or_ptr);
    obj = XObject::GetNativeObject<XObject>(kernel_state(), p);
  }
  if (obj) {
    type = static_cast<uint32_t>(obj->type());
    obj_addr = obj->guest_object();
  }

  const char* role = nullptr;
  uint64_t last_seq = 0, last_off = 0;
  std::string last_path;
  {
    std::lock_guard<std::mutex> lk(L->mu);
    const uint64_t now = Clock::QueryHostUptimeMillis();
    auto it = L->readers.find(thid);
    if (it != L->readers.end() && now - it->second.last_tick <= 5000) {
      role = "loader";
      last_seq = it->second.last_seq;
      last_off = it->second.last_off;
      last_path = it->second.last_path;
      // The loader's single-object wait names the flow-control gate event.
      if (count == 1 && obj_addr) L->gate_obj_addr = obj_addr;
    } else if (thid == L->gate_signaler_thid && L->gate_signaler_thid) {
      role = "consumer";  // the thread that signals the loader's gate
    } else {
      return;
    }
  }
  XELOGW(
      "[loader-wait] role={} thid={:08X} {} handle={:08X} obj={:08X} type={:X} "
      "count={} timeout={:X} alertable={} dur_ms={} result={:08X} lr={:08X} "
      "last_read seq={} off={:X} path={}",
      role, thid, api, handle_or_ptr, obj_addr, type, count, timeout, alertable,
      dur_ms, result, lr, last_seq, last_off, last_path);
}

void IoLedgerOnEventSet(uint32_t event_guest_addr, const char* api,
                        uint32_t thid, uint32_t lr) {
  if (!cvars::io_ledger_diag || !event_guest_addr) return;
  auto* L = io_ledger();
  const uint64_t now = Clock::QueryHostUptimeMillis();
  // Per-object signal summary (its own lock; event Set is very high-frequency).
  {
    std::lock_guard<std::mutex> lk(L->sig_mu);
    auto& s = L->sigsum[event_guest_addr];
    s.set_count++;
    s.last_set_tick = now;
    s.last_set_thid = thid;
    s.last_set_lr = lr;
    s.last_set_op = api;
  }
  // Gate-signal logging + mark the consumer.
  std::lock_guard<std::mutex> lk(L->mu);
  if (!L->gate_obj_addr || event_guest_addr != L->gate_obj_addr) return;
  const uint64_t interval =
      L->gate_last_signal_tick ? now - L->gate_last_signal_tick : 0;
  L->gate_last_signal_tick = now;
  L->gate_signaler_thid = thid;  // mark the consumer so its waits get tapped
  XELOGW(
      "[gate-signal] event={:08X} {} signaler_thid={:08X} lr={:08X} "
      "interval_ms={}",
      event_guest_addr, api, thid, lr, interval);
}

void IoLedgerOnEventReset(uint32_t event_guest_addr, uint32_t thid,
                          uint32_t lr) {
  if (!cvars::io_ledger_diag || !event_guest_addr) return;
  auto* L = io_ledger();
  std::lock_guard<std::mutex> lk(L->sig_mu);
  auto& s = L->sigsum[event_guest_addr];
  s.reset_count++;
  s.last_reset_tick = Clock::QueryHostUptimeMillis();
  s.last_reset_thid = thid;
}

void IoLedgerWaitEnter(const char* api, uint32_t wait_type, uint64_t timeout,
                       uint32_t alertable, uint32_t count, bool by_handle,
                       const uint32_t* handles) {
  if (!cvars::io_ledger_diag) return;
  auto* cur = XThread::GetCurrentThread();
  if (!cur) return;
  auto* ts = cpu::ThreadState::Get();
  auto* L = io_ledger();
  IoLedger::WaitEntry e{};
  e.api = api;
  e.lr = ts ? static_cast<uint32_t>(ts->context()->lr) : 0u;
  e.wait_type = wait_type;
  e.alertable = alertable;
  e.count = count;
  e.timeout = timeout;
  e.enter_tick = Clock::QueryHostUptimeMillis();
  e.by_handle = by_handle;
  const uint32_t n = count < 8 ? count : 8;
  for (uint32_t i = 0; i < n; ++i) e.handles[i] = handles[i];
  std::lock_guard<std::mutex> lk(L->mu);
  L->waits[cur->thread_id()] = e;
}

void IoLedgerWaitExit() {
  if (!cvars::io_ledger_diag) return;
  auto* cur = XThread::GetCurrentThread();
  if (!cur) return;
  auto* L = io_ledger();
  std::lock_guard<std::mutex> lk(L->mu);
  L->waits.erase(cur->thread_id());
}

void IoLedgerOnApcDelivered(uint32_t normal_routine, uint32_t normal_context) {
  if (!cvars::io_ledger_diag) return;
  auto* L = io_ledger();
  std::lock_guard<std::mutex> lk(L->mu);
  auto it = L->outstanding.find(normal_context);
  if (it != L->outstanding.end()) {
    L->apc_delivered++;
    L->outstanding.erase(it);
  }
}

void IoLedgerDumpOutstanding(const char* reason) {
  if (!cvars::io_ledger_diag) return;
  auto* L = io_ledger();
  std::lock_guard<std::mutex> lk(L->mu);
  static uint64_t s_last_dump_tick = 0;
  const uint64_t now = Clock::QueryHostUptimeMillis();
  if (s_last_dump_tick && now - s_last_dump_tick < 2000) return;  // once / 2s
  s_last_dump_tick = now;
  const uint64_t read_age = L->last_read_tick ? now - L->last_read_tick : 0;
  const uint64_t d_reads = L->read_count - L->reads_at_last_dump;
  const uint64_t d_opens = L->open_count - L->opens_at_last_dump;
  L->reads_at_last_dump = L->read_count;
  L->opens_at_last_dump = L->open_count;
  XELOGW(
      "[io-ledger] DUMP ({}) reads={} (+{}) opens={} (+{}) apc_queued={} "
      "apc_delivered={} outstanding={} last_read_age_ms={}",
      reason, L->read_count, d_reads, L->open_count, d_opens, L->apc_queued,
      L->apc_delivered, L->outstanding.size(), read_age);
  uint32_t shown = 0;
  for (auto& kv : L->outstanding) {
    auto& e = kv.second;
    const uint64_t age = now - e.queued_tick;
    if (age < 1000) continue;  // only stale ones
    XELOGW(
        "[io-ledger]   OUTSTANDING seq={} age_ms={} path={} off={:X} len={:X} "
        "apc_r={:08X} apc_c={:08X} iosb={:08X} thid={:08X} lr={:08X}",
        e.seq, age, e.path, e.offset, e.length, e.apc_routine, e.apc_context,
        e.iosb, e.thid, e.lr);
    if (++shown >= 32) break;
  }
  // Recent-reads ring: the last kRing reads in chronological order. If they loop
  // on one file / fixed offsets ⇒ stuck retry; if path/offset advance ⇒ progress.
  const uint32_t count =
      L->recent_head < IoLedger::kRing ? L->recent_head : IoLedger::kRing;
  for (uint32_t k = 0; k < count; ++k) {
    const uint32_t idx =
        (L->recent_head - count + k) % IoLedger::kRing;
    auto& r = L->recent[idx];
    XELOGW(
        "[io-ledger]   RECENT seq={} age_ms={} path={} off={:X} len={:X} "
        "bytes={:X} thid={:08X} lr={:08X}",
        r.seq, now - r.tick, r.path, r.offset, r.length, r.bytes, r.thid, r.lr);
  }

  // Passive wait registry: terminal unresolved waits (outstanding > 2s) + the
  // signal summary of each waited object — names what a thread is stuck on.
  for (auto& kv : L->waits) {
    auto& w = kv.second;
    const uint64_t age = now - w.enter_tick;
    if (age < 2000) continue;
    XELOGW(
        "[wait-stuck] thid={:08X} {} wait_type={} count={} timeout={:X} "
        "alertable={} age_ms={} lr={:08X}",
        kv.first, w.api, w.wait_type, w.count, w.timeout, w.alertable, age,
        w.lr);
    const uint32_t n = w.count < 8 ? w.count : 8;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t obj_addr = 0, type = 0xFFFFFFFFu;
      object_ref<XObject> obj;
      if (w.by_handle) {
        obj =
            kernel_state()->object_table()->LookupObject<XObject>(w.handles[i]);
      } else if (w.handles[i]) {
        obj = XObject::GetNativeObject<XObject>(
            kernel_state(), kernel_memory()->TranslateVirtual(w.handles[i]));
      }
      if (obj) {
        obj_addr = obj->guest_object();
        type = static_cast<uint32_t>(obj->type());
      }
      uint64_t setc = 0, lset = 0, rstc = 0, lrst = 0;
      uint32_t sthid = 0, slr = 0;
      const char* sop = "";
      {
        std::lock_guard<std::mutex> sk(L->sig_mu);
        auto it = L->sigsum.find(obj_addr);
        if (it != L->sigsum.end()) {
          setc = it->second.set_count;
          lset = it->second.last_set_tick;
          sthid = it->second.last_set_thid;
          slr = it->second.last_set_lr;
          sop = it->second.last_set_op;
          rstc = it->second.reset_count;
          lrst = it->second.last_reset_tick;
        }
      }
      XELOGW(
          "[wait-stuck]   obj[{}/{}] handle={:08X} obj={:08X} type={:X} "
          "set_count={} last_set={}ms_ago by_thid={:08X} lr={:08X} op={} "
          "reset_count={} last_reset={}ms_ago",
          i, w.count, w.handles[i], obj_addr, type, setc, lset ? now - lset : 0,
          sthid, slr, sop, rstc, lrst ? now - lrst : 0);
    }
  }
}

struct CreateOptions {
  // https://processhacker.sourceforge.io/doc/ntioapi_8h.html
  static constexpr uint32_t FILE_DIRECTORY_FILE = 0x00000001;
  // Optimization - files access will be sequential, not random.
  static constexpr uint32_t FILE_SEQUENTIAL_ONLY = 0x00000004;
  static constexpr uint32_t FILE_SYNCHRONOUS_IO_ALERT = 0x00000010;
  static constexpr uint32_t FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020;
  static constexpr uint32_t FILE_NON_DIRECTORY_FILE = 0x00000040;
  // Optimization - file access will be random, not sequential.
  static constexpr uint32_t FILE_RANDOM_ACCESS = 0x00000800;
};

dword_result_t NtCreateFile_entry(lpdword_t handle_out, dword_t desired_access,
                                  pointer_t<X_OBJECT_ATTRIBUTES> object_attrs,
                                  pointer_t<X_IO_STATUS_BLOCK> io_status_block,
                                  lpqword_t allocation_size_ptr,
                                  dword_t file_attributes, dword_t share_access,
                                  dword_t creation_disposition,
                                  dword_t create_options) {
  uint64_t allocation_size = 0;  // is this correct???
  if (allocation_size_ptr) {
    allocation_size = *allocation_size_ptr;
  }

  if (!object_attrs) {
    // ..? Some games do this. This parameter is not optional.
    return X_STATUS_INVALID_PARAMETER;
  }
  assert_not_null(handle_out);

  // A null name opens what the root directory handle refers to. That's how a
  // file is reopened from a handle.
  auto object_name =
      util::TranslateAnsiStringPointer(kernel_memory(), object_attrs->name_ptr);

  vfs::Entry* root_entry = nullptr;

  const bool has_root_handle =
      object_attrs->root_directory != 0xFFFFFFFD &&  // ObDosDevices
      object_attrs->root_directory != 0;

  // Compute path, possibly attrs relative.
  auto target_path = util::TranslateAnsiPath(kernel_memory(), object_name);
  // A name relative to a directory handle is never under \??\.
  if (!has_root_handle) {
    xeObStripDosDevicesPrefix(target_path);
  }

  // Enforce that the path is ASCII.
  if (!IsValidPath(target_path, false)) {
    return X_STATUS_OBJECT_NAME_INVALID;
  }

  if (has_root_handle) {
    auto root_file = kernel_state()->object_table()->LookupObject<XFile>(
        object_attrs->root_directory);
    assert_not_null(root_file);
    assert_true(root_file->type() == XObject::Type::File);

    root_entry = root_file->entry();
  }

  // Attempt open (or create).
  vfs::File* vfs_file;
  vfs::FileAction file_action;
  X_STATUS result = kernel_state()->file_system()->OpenFile(
      root_entry, target_path,
      vfs::FileDisposition((uint32_t)creation_disposition), desired_access,
      (create_options & CreateOptions::FILE_DIRECTORY_FILE) != 0,
      (create_options & CreateOptions::FILE_NON_DIRECTORY_FILE) != 0, &vfs_file,
      &file_action);
  object_ref<XFile> file = nullptr;

  // A failed open leaves the handle and the information 0.
  X_HANDLE handle = 0;
  if (XSUCCEEDED(result)) {
    // If true, desired_access SYNCHRONIZE flag must be set.
    bool synchronous =
        (create_options & CreateOptions::FILE_SYNCHRONOUS_IO_ALERT) ||
        (create_options & CreateOptions::FILE_SYNCHRONOUS_IO_NONALERT);
    file = object_ref<XFile>(new XFile(kernel_state(), vfs_file, synchronous));

    // Handle ref is incremented, so return that.
    handle = file->handle();
  }

  if (io_status_block) {
    io_status_block->status = result;
    io_status_block->information =
        XSUCCEEDED(result) ? uint32_t(file_action) : 0;
  }

  *handle_out = handle;

  IoLedgerOnOpen(target_path, result);
  return result;
}
DECLARE_XBOXKRNL_EXPORT1(NtCreateFile, kFileSystem, kImplemented);

dword_result_t NtOpenFile_entry(
    lpdword_t handle_out, dword_t desired_access,
    pointer_t<X_OBJECT_ATTRIBUTES> object_attributes,
    pointer_t<X_IO_STATUS_BLOCK> io_status_block, dword_t share_access,
    dword_t open_options) {
  return NtCreateFile_entry(
      handle_out, desired_access, object_attributes, io_status_block, nullptr,
      0, share_access, static_cast<uint32_t>(xe::vfs::FileDisposition::kOpen),
      open_options);
}
DECLARE_XBOXKRNL_EXPORT1(NtOpenFile, kFileSystem, kImplemented);

// Status last, so a caller polling it for completion reads a valid count.
static void WriteIoStatus(X_IO_STATUS_BLOCK* status_block, X_STATUS status,
                          uint32_t information) {
  status_block->information = information;
  std::atomic_thread_fence(std::memory_order_release);
  status_block->status = status;
}

dword_result_t NtReadFile_entry(dword_t file_handle, dword_t event_handle,
                                lpvoid_t apc_routine_ptr, lpvoid_t apc_context,
                                pointer_t<X_IO_STATUS_BLOCK> io_status_block,
                                lpvoid_t buffer, dword_t buffer_length,
                                lpqword_t byte_offset_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  bool signal_event = false;
  uint32_t ledger_bytes = 0;       // io_ledger_diag capture
  bool ledger_apc_queued = false;  // io_ledger_diag capture
  auto ev = kernel_state()->object_table()->LookupObject<XEvent>(event_handle);
  if (event_handle && !ev) {
    result = X_STATUS_INVALID_HANDLE;
  }

  auto file = kernel_state()->object_table()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  } else if (XSUCCEEDED(result) && !file->is_synchronous() &&
             !byte_offset_ptr) {
    // Without a byte offset, only a synchronous handle uses its file position.
    result = X_STATUS_INVALID_PARAMETER;
  }

  if (XSUCCEEDED(result)) {
    if (true || file->is_synchronous()) {
      // Synchronous.
      uint32_t bytes_read = 0;
      result = file->Read(
          buffer.guest_address(), buffer_length,
          byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : -1,
          &bytes_read, apc_context);
      ledger_bytes = bytes_read;
      if (cvars::io_ledger_diag && bytes_read >= 16) {
        // First 16 bytes of what the guest received (content check).
        const uint8_t* hb = kernel_memory()->TranslateVirtual(buffer.guest_address());
        XELOGI("IoLedger DATA {} off={:X} len={:X} -> {:02X}{:02X}{:02X}{:02X} "
               "{:02X}{:02X}{:02X}{:02X} {:02X}{:02X}{:02X}{:02X} "
               "{:02X}{:02X}{:02X}{:02X}",
               file->entry() ? file->entry()->name() : "?",
               byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : ~0ull,
               uint32_t(buffer_length), hb[0], hb[1], hb[2], hb[3], hb[4], hb[5],
               hb[6], hb[7], hb[8], hb[9], hb[10], hb[11], hb[12], hb[13],
               hb[14], hb[15]);
      }
      if (io_status_block) {
        WriteIoStatus(io_status_block, result, bytes_read);
      }

      // Queue the APC callback. It must be delivered via the APC mechanism even
      // though were are completing immediately.
      // Low bit probably means do not queue to IO ports.
      // A caller told PENDING always gets its APC, as on NT, even on failure:
      // it has no other way to learn the read finished.
      bool pending = !file->is_synchronous() && result != X_STATUS_END_OF_FILE;
      if ((uint32_t)apc_routine_ptr & ~1) {
        if (apc_context && (pending || result == X_STATUS_SUCCESS)) {
          auto thread = XThread::GetCurrentThread();
          thread->EnqueueApc(static_cast<uint32_t>(apc_routine_ptr) & ~1u,
                             apc_context, io_status_block, 0);
          ledger_apc_queued = true;
        }
      }

      if (!file->is_synchronous() && result != X_STATUS_END_OF_FILE) {
        result = X_STATUS_PENDING;
      }

      // Mark that we should signal the event now. We do this after
      // we have written the info out.
      signal_event = true;

      if (XSUCCEEDED(result)) {
        if (auto patch = kernel_state()->xmp_volume_patch()) {
          auto host_buf =
              kernel_memory()->TranslateVirtual(buffer.guest_address());
          patch->OnFileRead(file->entry()->name(), host_buf, buffer_length,
                            buffer.guest_address());
        }
      }
    } else {
      // TODO(benvanik): async.

      // X_STATUS_PENDING if not returning immediately.
      // XFile is waitable and signalled after each async req completes.
      // reset the input event (->Reset())
      /*xeNtReadFileState* call_state = new xeNtReadFileState();
      XAsyncRequest* request = new XAsyncRequest(
      state, file,
      (XAsyncRequest::CompletionCallback)xeNtReadFileCompleted,
      call_state);*/
      // result = file->Read(buffer.guest_address(), buffer_length, byte_offset,
      //                     request);
      if (io_status_block) {
        io_status_block->status = X_STATUS_PENDING;
        io_status_block->information = 0;
      }

      result = X_STATUS_PENDING;
    }
  }

  if (XFAILED(result) && io_status_block) {
    io_status_block->status = result;
    io_status_block->information = 0;
  }

  if (ev && signal_event) {
    ev->Set(0, false);
  }

  if (cvars::io_ledger_diag) {
    auto* cur = XThread::GetCurrentThread();
    auto* ctx = cpu::ThreadState::Get() ? cpu::ThreadState::Get()->context()
                                        : nullptr;
    IoLedgerOnRead(
        file ? (file->entry() ? file->entry()->name() : std::string("?"))
             : std::string("<bad-handle>"),
        byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : ~0ull,
        static_cast<uint32_t>(buffer_length), file && file->is_synchronous(),
        static_cast<uint32_t>(event_handle),
        static_cast<uint32_t>(apc_routine_ptr) & ~1u,
        static_cast<uint32_t>(apc_context),
        io_status_block.guest_address(), result, ledger_bytes,
        ledger_apc_queued, (ev && signal_event),
        cur ? cur->thread_id() : 0,
        ctx ? static_cast<uint32_t>(ctx->lr) : 0);
  }

  return result;
}
DECLARE_XBOXKRNL_EXPORT2(NtReadFile, kFileSystem, kImplemented, kHighFrequency);

dword_result_t NtReadFileScatter_entry(
    dword_t file_handle, dword_t event_handle, lpvoid_t apc_routine_ptr,
    lpvoid_t apc_context, pointer_t<X_IO_STATUS_BLOCK> io_status_block,
    lpdword_t segment_array, dword_t length, lpqword_t byte_offset_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  bool signal_event = false;
  auto ev = kernel_state()->object_table()->LookupObject<XEvent>(event_handle);
  if (event_handle && !ev) {
    result = X_STATUS_INVALID_HANDLE;
  }

  auto file = kernel_state()->object_table()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  } else if (XSUCCEEDED(result) && !file->is_synchronous() &&
             !byte_offset_ptr) {
    // Without a byte offset, only a synchronous handle uses its file position.
    result = X_STATUS_INVALID_PARAMETER;
  }

  if (XSUCCEEDED(result)) {
    if (true || file->is_synchronous()) {
      // Synchronous.
      uint32_t bytes_read = 0;
      result = file->ReadScatter(
          segment_array.guest_address(), length,
          byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : -1,
          &bytes_read, apc_context);
      if (io_status_block) {
        WriteIoStatus(io_status_block, result, bytes_read);
      }

      // Queue the APC callback. It must be delivered via the APC mechanism even
      // though were are completing immediately.
      // Low bit probably means do not queue to IO ports.
      // An async handle is always told PENDING, and then always gets its APC.
      if ((uint32_t)apc_routine_ptr & ~1) {
        if (apc_context &&
            (!file->is_synchronous() || result == X_STATUS_SUCCESS)) {
          auto thread = XThread::GetCurrentThread();
          thread->EnqueueApc(static_cast<uint32_t>(apc_routine_ptr) & ~1u,
                             apc_context, io_status_block, 0);
        }
      }

      if (!file->is_synchronous()) {
        result = X_STATUS_PENDING;
      }

      // Mark that we should signal the event now. We do this after
      // we have written the info out.
      signal_event = true;
    } else {
      // TODO(benvanik): async.

      // TODO: On Windows it might be worth trying to use Win32 ReadFileScatter
      // here instead of handling it ourselves

      // X_STATUS_PENDING if not returning immediately.
      // XFile is waitable and signalled after each async req completes.
      // reset the input event (->Reset())
      /*xeNtReadFileState* call_state = new xeNtReadFileState();
      XAsyncRequest* request = new XAsyncRequest(
      state, file,
      (XAsyncRequest::CompletionCallback)xeNtReadFileCompleted,
      call_state);*/
      // result = file->Read(buffer.guest_address(), buffer_length, byte_offset,
      //                     request);
      if (io_status_block) {
        io_status_block->status = X_STATUS_PENDING;
        io_status_block->information = 0;
      }

      result = X_STATUS_PENDING;
    }
  }

  if (XFAILED(result) && io_status_block) {
    io_status_block->status = result;
    io_status_block->information = 0;
  }

  if (ev && signal_event) {
    ev->Set(0, false);
  }

  return result;
}
DECLARE_XBOXKRNL_EXPORT1(NtReadFileScatter, kFileSystem, kImplemented);

dword_result_t NtWriteFile_entry(dword_t file_handle, dword_t event_handle,
                                 function_t apc_routine, lpvoid_t apc_context,
                                 pointer_t<X_IO_STATUS_BLOCK> io_status_block,
                                 lpvoid_t buffer, dword_t buffer_length,
                                 lpqword_t byte_offset_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  // Grab event to signal.
  bool signal_event = false;
  auto ev = kernel_state()->object_table()->LookupObject<XEvent>(event_handle);
  if (event_handle && !ev) {
    result = X_STATUS_INVALID_HANDLE;
  }

  // Grab file.
  auto file = kernel_state()->object_table()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  } else if (XSUCCEEDED(result) && !file->is_synchronous() &&
             !byte_offset_ptr) {
    // Without a byte offset, only a synchronous handle uses its file position.
    result = X_STATUS_INVALID_PARAMETER;
  }

  // Execute write.
  if (XSUCCEEDED(result)) {
    // TODO(benvanik): async path.
    if (true || file->is_synchronous()) {
      // Synchronous request.
      uint32_t bytes_written = 0;
      result = file->Write(
          buffer.guest_address(), buffer_length,
          byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : -1,
          &bytes_written, apc_context);

      if (io_status_block) {
        io_status_block->status = result;
        io_status_block->information = static_cast<uint32_t>(bytes_written);
      }

      // Queue the APC callback. It must be delivered via the APC mechanism even
      // though were are completing immediately.
      // Low bit probably means do not queue to IO ports.
      if ((uint32_t)apc_routine & ~1) {
        if (apc_context) {
          auto thread = XThread::GetCurrentThread();
          thread->EnqueueApc(static_cast<uint32_t>(apc_routine) & ~1u,
                             apc_context, io_status_block, 0);
        }
      }

      if (!file->is_synchronous()) {
        result = X_STATUS_PENDING;
      }

      // Mark that we should signal the event now. We do this after
      // we have written the info out.
      signal_event = true;

      if (XSUCCEEDED(result)) {
        if (auto patch = kernel_state()->xmp_volume_patch()) {
          auto host_buf =
              kernel_memory()->TranslateVirtual(buffer.guest_address());
          patch->OnFileWrite(file->entry()->name(), host_buf, buffer_length,
                             buffer.guest_address());
        }
      }
    } else {
      // X_STATUS_PENDING if not returning immediately.
      result = X_STATUS_PENDING;

      if (io_status_block) {
        io_status_block->status = X_STATUS_PENDING;
        io_status_block->information = 0;
      }
    }
  }

  if (XFAILED(result) && io_status_block) {
    io_status_block->status = result;
    io_status_block->information = 0;
  }

  if (ev && signal_event) {
    ev->Set(0, false);
  }

  if (cvars::io_ledger_diag) {
    XELOGI("IoLedger WRITE {} off={:X} len={:X} -> {:08X}",
           file && file->entry() ? file->entry()->name() : "?",
           byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : ~0ull,
           uint32_t(buffer_length), uint32_t(result));
  }
  return result;
}
DECLARE_XBOXKRNL_EXPORT1(NtWriteFile, kFileSystem, kImplemented);

dword_result_t NtCreateIoCompletion_entry(lpdword_t out_handle,
                                          dword_t desired_access,
                                          lpvoid_t object_attribs,
                                          dword_t num_concurrent_threads) {
  auto completion =
      object_ref<XIOCompletion>(new XIOCompletion(kernel_state()));
  if (out_handle) {
    *out_handle = completion->handle();
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(NtCreateIoCompletion, kFileSystem, kImplemented);

dword_result_t NtSetIoCompletion_entry(dword_t handle, dword_t key_context,
                                       dword_t apc_context,
                                       dword_t completion_status,
                                       dword_t num_bytes) {
  auto port =
      kernel_state()->object_table()->LookupObject<XIOCompletion>(handle);
  if (!port) {
    return X_STATUS_INVALID_HANDLE;
  }

  XIOCompletion::IONotification notification;
  notification.key_context = key_context;
  notification.apc_context = apc_context;
  notification.num_bytes = num_bytes;
  notification.status = completion_status;

  port->QueueNotification(notification);
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT2(NtSetIoCompletion, kFileSystem, kImplemented,
                         kHighFrequency);

// Dequeues a packet from the completion port.
dword_result_t NtRemoveIoCompletion_entry(
    dword_t handle, lpdword_t key_context, lpdword_t apc_context,
    pointer_t<X_IO_STATUS_BLOCK> io_status_block, lpqword_t timeout) {
  X_STATUS status = X_STATUS_SUCCESS;
  uint32_t info = 0;

  auto port =
      kernel_state()->object_table()->LookupObject<XIOCompletion>(handle);
  if (!port) {
    status = X_STATUS_INVALID_HANDLE;
  }

  uint64_t timeout_ticks =
      timeout ? static_cast<uint32_t>(*timeout)
              : static_cast<uint64_t>(std::numeric_limits<int64_t>::min());
  XIOCompletion::IONotification notification;
  if (port->WaitForNotification(timeout_ticks, &notification)) {
    if (key_context) {
      *key_context = notification.key_context;
    }
    if (apc_context) {
      *apc_context = notification.apc_context;
    }

    if (io_status_block) {
      io_status_block->status = notification.status;
      io_status_block->information = notification.num_bytes;
    }
  } else {
    status = X_STATUS_TIMEOUT;
  }

  return status;
}
DECLARE_XBOXKRNL_EXPORT2(NtRemoveIoCompletion, kFileSystem, kImplemented,
                         kHighFrequency);

dword_result_t NtCancelIoFile_entry(dword_t handle) {
  auto file = kernel_state()->object_table()->LookupObject<XFile>(handle);
  if (!file) {
    return X_STATUS_INVALID_HANDLE;
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(NtCancelIoFile, kFileSystem, kStub);

dword_result_t NtQueryFullAttributesFile_entry(
    pointer_t<X_OBJECT_ATTRIBUTES> obj_attribs,
    pointer_t<X_FILE_NETWORK_OPEN_INFORMATION> file_info) {
  auto object_name =
      util::TranslateAnsiStringPointer(kernel_memory(), obj_attribs->name_ptr);

  object_ref<XFile> root_file;
  if (obj_attribs->root_directory != 0xFFFFFFFD &&  // ObDosDevices
      obj_attribs->root_directory != 0) {
    root_file = kernel_state()->object_table()->LookupObject<XFile>(
        obj_attribs->root_directory);
    assert_not_null(root_file);
    assert_true(root_file->type() == XObject::Type::File);
    assert_always();
  }

  auto target_path = util::TranslateAnsiPath(kernel_memory(), object_name);
  // A name relative to a directory handle is never under \??\.
  if (!root_file) {
    xeObStripDosDevicesPrefix(target_path);
  }

  // Enforce that the path is ASCII.
  if (!IsValidPath(target_path, false)) {
    return X_STATUS_OBJECT_NAME_INVALID;
  }

  // Resolve the file using the virtual file system.
  auto entry = kernel_state()->file_system()->ResolvePath(target_path);
  if (entry) {
    // Found.
    file_info->creation_time = entry->create_timestamp();
    file_info->last_access_time = entry->access_timestamp();
    file_info->last_write_time = entry->write_timestamp();
    file_info->change_time = entry->write_timestamp();
    file_info->allocation_size = entry->allocation_size();
    file_info->end_of_file = entry->size();
    file_info->attributes = entry->attributes();

    if (cvars::io_ledger_diag) {
      XELOGI("IoLedger ATTR '{}' -> 00000000", target_path);
    }
    return X_STATUS_SUCCESS;
  }

  // A missing directory or link before the last component is a missing path.
  X_STATUS missing_status = X_STATUS_OBJECT_NAME_NOT_FOUND;
  auto base_path = xe::utf8::find_base_guest_path(target_path);
  if (!base_path.empty() &&
      !kernel_state()->file_system()->ResolvePath(base_path)) {
    missing_status = X_STATUS_OBJECT_PATH_NOT_FOUND;
  }
  if (cvars::io_ledger_diag) {
    XELOGI("IoLedger ATTR '{}' -> {:08X}", target_path, missing_status);
  }
  return missing_status;
}
DECLARE_XBOXKRNL_EXPORT1(NtQueryFullAttributesFile, kFileSystem, kImplemented);

dword_result_t NtQueryDirectoryFile_entry(
    dword_t file_handle, dword_t event_handle, function_t apc_routine,
    lpvoid_t apc_context, pointer_t<X_IO_STATUS_BLOCK> io_status_block,
    pointer_t<X_FILE_DIRECTORY_INFORMATION> file_info_ptr, dword_t length,
    pointer_t<X_ANSI_STRING> file_name, dword_t restart_scan) {
  if (length < 72) {
    return X_STATUS_INFO_LENGTH_MISMATCH;
  }

  X_STATUS result = X_STATUS_UNSUCCESSFUL;
  uint32_t info = 0;

  auto file = kernel_state()->object_table()->LookupObject<XFile>(file_handle);
  auto name = util::TranslateAnsiPath(kernel_memory(), file_name);

  // Enforce that the path is ASCII.
  if (!IsValidPath(name, true)) {
    return X_STATUS_INVALID_PARAMETER;
  }

  if (file) {
    X_FILE_DIRECTORY_INFORMATION dir_info = {0};
    result =
        file->QueryDirectory(file_info_ptr, length, name, restart_scan != 0);
    if (XSUCCEEDED(result)) {
      info = length;
    }
  } else {
    result = X_STATUS_NO_SUCH_FILE;
  }
  if (cvars::io_ledger_diag) {
    XELOGI("IoLedger QDIR {} pattern='{}' restart={} -> {:08X}",
           file && file->entry() ? file->entry()->absolute_path() : "?", name,
           uint32_t(restart_scan), uint32_t(result));
  }

  if (XFAILED(result)) {
    info = 0;
  }

  if (io_status_block) {
    io_status_block->status = result;
    io_status_block->information = info;
  }

  return result;
}
DECLARE_XBOXKRNL_EXPORT1(NtQueryDirectoryFile, kFileSystem, kImplemented);

dword_result_t NtFlushBuffersFile_entry(
    dword_t file_handle, pointer_t<X_IO_STATUS_BLOCK> io_status_block_ptr) {
  auto result = X_STATUS_SUCCESS;

  if (io_status_block_ptr) {
    io_status_block_ptr->status = result;
    io_status_block_ptr->information = 0;
  }

  return result;
}
DECLARE_XBOXKRNL_EXPORT1(NtFlushBuffersFile, kFileSystem, kStub);

// https://docs.microsoft.com/en-us/windows/win32/devnotes/ntopensymboliclinkobject
dword_result_t NtOpenSymbolicLinkObject_entry(
    lpdword_t handle_out, pointer_t<X_OBJECT_ATTRIBUTES> object_attrs) {
  if (!object_attrs) {
    return X_STATUS_INVALID_PARAMETER;
  }
  assert_not_null(handle_out);

  assert_true(object_attrs->attributes == 64);  // case insensitive

  auto object_name =
      util::TranslateAnsiStringPointer(kernel_memory(), object_attrs->name_ptr);

  auto target_path = xeObSymbolicLinkName(
      util::TranslateAnsiPath(kernel_memory(), object_name));

  // Enforce that the path is ASCII.
  if (!IsValidPath(target_path, false)) {
    return X_STATUS_OBJECT_NAME_INVALID;
  }

  if (object_attrs->root_directory != 0) {
    assert_always();
  }

  std::string link_path;
  if (!kernel_state()->file_system()->FindSymbolicLink(target_path,
                                                       link_path)) {
    return X_STATUS_NO_SUCH_FILE;
  }

  object_ref<XSymbolicLink> symlink(new XSymbolicLink(kernel_state()));
  symlink->Initialize(target_path, link_path);

  *handle_out = symlink->handle();

  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(NtOpenSymbolicLinkObject, kFileSystem, kImplemented);

// https://docs.microsoft.com/en-us/windows/win32/devnotes/ntquerysymboliclinkobject
dword_result_t NtQuerySymbolicLinkObject_entry(
    dword_t handle, pointer_t<X_ANSI_STRING> target) {
  auto symlink =
      kernel_state()->object_table()->LookupObject<XSymbolicLink>(handle);
  if (!symlink) {
    return X_STATUS_NO_SUCH_FILE;
  }
  auto length = std::min(static_cast<size_t>(target->maximum_length),
                         symlink->target().size());
  if (length > 0) {
    auto target_buf = kernel_memory()->TranslateVirtual(target->pointer);
    std::memcpy(target_buf, symlink->target().c_str(), length);
  }
  target->length = static_cast<uint16_t>(length);
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(NtQuerySymbolicLinkObject, kFileSystem, kImplemented);

dword_result_t FscGetCacheElementCount_entry(dword_t r3) { return 0; }
DECLARE_XBOXKRNL_EXPORT1(FscGetCacheElementCount, kFileSystem, kStub);

dword_result_t FscSetCacheElementCount_entry(dword_t unk_0, dword_t unk_1) {
  // unk_0 = 0
  // unk_1 looks like a count? in what units? 256 is a common value
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(FscSetCacheElementCount, kFileSystem, kStub);
// todo: this should fill in the io status block and queue the apc
dword_result_t NtDeviceIoControlFile_entry(
    dword_t handle, dword_t event_handle, dword_t apc_routine,
    dword_t apc_context, pointer_t<X_IO_STATUS_BLOCK> io_status_block,
    dword_t io_control_code, lpvoid_t input_buffer, dword_t input_buffer_len,
    lpvoid_t output_buffer, dword_t output_buffer_len) {
  // Called by XMountUtilityDrive cache-mounting code
  // (checks if the returned values look valid, values below seem to pass the
  // checks)
  constexpr uint32_t cache_size = 0xFF000;

  if (io_control_code == X_IOCTL_DISK_GET_DRIVE_GEOMETRY) {
    if (output_buffer_len < 0x8) {
      assert_always();
      return X_STATUS_BUFFER_TOO_SMALL;
    }
    xe::store_and_swap<uint32_t>(output_buffer, cache_size / 512);
    xe::store_and_swap<uint32_t>(output_buffer + 4, 512);
  } else if (io_control_code == X_IOCTL_DISK_GET_PARTITION_INFO) {
    if (output_buffer_len < 0x10) {
      assert_always();
      return X_STATUS_BUFFER_TOO_SMALL;
    }
    xe::store_and_swap<uint64_t>(output_buffer, 0);
    xe::store_and_swap<uint64_t>(output_buffer + 8, cache_size);
  } else {
    // Warning level: a Release build filters Debug, and an unhandled IOCTL is
    // exactly the kind of failure a title turns into a "dirty disc" error.
    XELOGW("NtDeviceIoControlFile(0x{:X}) - unhandled IOCTL! (in={} out={})",
           uint32_t(io_control_code), uint32_t(input_buffer_len),
           uint32_t(output_buffer_len));
    assert_always();
    return X_STATUS_INVALID_PARAMETER;
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(NtDeviceIoControlFile, kFileSystem, kStub);
// device_extension_size = additional bytes of data (aligned up to 8 byte
// granularity) that will be allocated at the tail of the resulting device
// object. although it is allocated at the tail, it is accessed through a
// pointer at offset 0x18 so in theory a guest could be unaware that its a
// single allocation device_name is optional, extra_device_object_attributes
// gets assigned to the attributes field of the OBJECT_ATTRIBUTES used for
// ObCreateObject

// todo: need device guest object struct + host object for device
dword_result_t IoCreateDevice_entry(dword_t driver_object,
                                    dword_t device_extension_size,
                                    pointer_t<X_ANSI_STRING> device_name,
                                    dword_t device_type,
                                    dword_t extra_device_object_attributes,
                                    lpdword_t device_object,
                                    const ppc_context_t& ctx) {
  // Called from XMountUtilityDrive XAM-task code
  // That code tries writing things to a pointer at out_struct+0x18
  // We'll alloc some scratch space for it so it doesn't cause any exceptions

  // 0x24 is guessed size from accesses to out_struct - likely incorrect
  auto current_kernel = ctx->kernel_state;

  uint32_t required_size = 80 + xe::align<uint32_t>(device_extension_size, 8);

  auto kernel_mem = current_kernel->memory();

  auto out_guest = kernel_mem->SystemHeapAlloc(required_size);

  auto out = kernel_mem->TranslateVirtual<uint8_t*>(out_guest);

  memset(out, 0, required_size);

  xe::store<unsigned char>(out, 3);  // maybe device object's Ob type?

  // this stores the total object size, without alignment!

  xe::store_and_swap<uint16_t>(out + 2, device_extension_size + 80);

  // from 17559
  if (device_type == 7 || device_type == 58 || device_type == 62 ||
      device_type == 45 || device_type == 2 || device_type == 60 ||
      device_type == 61 || device_type == 36 || device_type == 64 ||
      device_type == 65 || device_type == 66 || device_type == 67 ||
      device_type == 68 || device_type == 69 || device_type == 70 ||
      device_type == 72 || device_type == 73) {
    xe::store_and_swap<uint32_t>(out + 0xC, 0);
  } else {
    // pointer to itself?
    xe::store_and_swap<uint32_t>(out + 0xC, out_guest);
  }
  xe::store<uint8_t>(out + 0x1C, static_cast<uint8_t>(device_type));

  uint32_t flags_field_value = 16;
  if (device_name) {
    flags_field_value |= 8;
  }
  xe::store<unsigned char>(out + 0x1e, 1);
  xe::store_and_swap<uint32_t>(out + 0x14, flags_field_value);
  if (device_extension_size != 0) {
    // pointer to device specific data
    //  XMountUtilityDrive writes some kind of header here
    xe::store_and_swap<uint32_t>(out + 0x18, out_guest + 80);
  }

  xe::store_and_swap<uint32_t>(out + 8, driver_object);

  *device_object = out_guest;
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(IoCreateDevice, kFileSystem, kStub);

// supposed to invoke a callback on the driver object! its some sort of
// destructor function intended to be called for all devices created from the
// driver
void IoDeleteDevice_entry(dword_t device_ptr, const ppc_context_t& ctx) {
  if (device_ptr) {
    auto kernel_mem = ctx->kernel_state->memory();
    kernel_mem->SystemHeapFree(device_ptr);
  }
}

DECLARE_XBOXKRNL_EXPORT1(IoDeleteDevice, kFileSystem, kStub);

// Volume dismount: we never cache mounted-volume state per handle, so
// acknowledging success is complete behavior. DOA5's WMV player calls this
// (previously an undefined extern) around its intro-video file handling.
dword_result_t IoDismountVolumeByFileHandle_entry(dword_t file_handle,
                                                  const ppc_context_t& ctx) {
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(IoDismountVolumeByFileHandle, kFileSystem, kStub);

// Same contract as the by-handle form: no per-volume mount state exists, so
// acknowledging success is complete behavior. Halo 3's cache-partition code
// calls this (previously an undefined extern) right before its disc check.
dword_result_t IoDismountVolume_entry(lpvoid_t device_object,
                                      const ppc_context_t& ctx) {
  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(IoDismountVolume, kFileSystem, kStub);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(Io);
