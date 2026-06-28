/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/apu/xma_decoder.h"

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/xma_context.h"
#include "xenia/apu/xma_context_fake.h"
#include "xenia/apu/xma_context_master.h"
#include "xenia/apu/xma_context_new.h"
#include "xenia/apu/xma_context_old.h"

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/profiling.h"
#include "xenia/base/ring_buffer.h"
#include "xenia/base/string_buffer.h"
#include <cstdio>

#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/ppc/ppc_opcode_info.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/thread_state.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/xthread.h"
extern "C" {
#include "third_party/FFmpeg/libavutil/log.h"
}  // extern "C"

// As with normal Microsoft, there are like twelve different ways to access
// the audio APIs. Early games use XMA*() methods almost exclusively to touch
// decoders. Later games use XAudio*() and direct memory writes to the XMA
// structures (as opposed to the XMA* calls), meaning that we have to support
// both.
//
// The XMA*() functions just manipulate the audio system in the guest context
// and let the normal XmaDecoder handling take it, to prevent duplicate
// implementations. They can be found in xboxkrnl_audio_xma.cc
//
// XMA details:
// https://devel.nuclex.org/external/svn/directx/trunk/include/xma2defs.h
// https://github.com/gdawg/fsbext/blob/master/src/xma_header.h
//
// XAudio2 uses XMA under the covers, and seems to map with the same
// restrictions of frame/subframe/etc:
// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.xaudio2.xaudio2_buffer(v=vs.85).aspx
//
// XMA contexts are 64b in size and tight bitfields. They are in physical
// memory not usually available to games. Games will use MmMapIoSpace to get
// the 64b pointer in user memory so they can party on it. If the game doesn't
// do this, it's likely they are either passing the context to XAudio or
// using the XMA* functions.

DEFINE_bool(ffmpeg_verbose, false, "Verbose FFmpeg output (debug and above)",
            "APU");

DEFINE_bool(use_dedicated_xma_thread, true,
            "Enables XMA decoding on separate thread. Disabled should produce "
            "better results, but decrease performance a bit.",
            "APU");

DEFINE_string(
    xma_decoder, "new",
    "Decoder version used to process XMA audio.\n"
    "Use: [fake, master, old, new]\n"
    " fake: \n  No audio will be decoded.\n"
    " master: \n  Version of decoder exactly like on base version of Xenia.\n"
    " old: \n  Decoder based on master version of decoder with few "
    "improvements.\n"
    " new: \n  New version of decoder. Provides highest stability, but isn't "
    "yet finished.\n",
    "APU");

namespace xe {
namespace apu {

XmaDecoder::XmaDecoder(cpu::Processor* processor)
    : memory_(processor->memory()), processor_(processor) {}

XmaDecoder::~XmaDecoder() = default;

void av_log_callback(void* avcl, int level, const char* fmt, va_list va) {
  if (!cvars::ffmpeg_verbose && level > AV_LOG_WARNING) {
    return;
  }

  char level_char = '?';
  LogLevel log_level;
  switch (level) {
    case AV_LOG_ERROR: {
      level_char = '!';
      log_level = xe::LogLevel::Error;
      break;
    }
    case AV_LOG_WARNING: {
      level_char = 'w';
      log_level = xe::LogLevel::Warning;
      break;
    }
    case AV_LOG_INFO: {
      level_char = 'i';
      log_level = xe::LogLevel::Info;
      break;
    }
    case AV_LOG_VERBOSE: {
      level_char = 'v';
      log_level = xe::LogLevel::Debug;
      break;
    }
    case AV_LOG_DEBUG: {
      level_char = 'd';
      log_level = xe::LogLevel::Debug;
      break;
    }
    default: {
      level_char = '?';
      log_level = xe::LogLevel::Debug;
      break;
    }
  }

  StringBuffer buff;
  buff.AppendVarargs(fmt, va);
  xe::logging::AppendLogLineFormat(LogSrc::Apu, log_level, level_char,
                                   "ffmpeg: {}", buff.to_string_view());
}

X_STATUS XmaDecoder::Setup(kernel::KernelState* kernel_state) {
  // Setup ffmpeg logging callback
  av_log_set_callback(av_log_callback);

  // Let the processor know we want register access callbacks.
  memory_->AddVirtualMappedRange(
      0x7FEA0000, 0xFFFF0000, 0x0000FFFF, this,
      reinterpret_cast<cpu::MMIOReadCallback>(MMIOReadRegisterThunk),
      reinterpret_cast<cpu::MMIOWriteCallback>(MMIOWriteRegisterThunk));

  // Setup XMA context data.
  // The Xbox 360 kernel allocates the contexts with X_PAGE_NOCACHE |
  // X_PAGE_READWRITE and writes MmGetPhysicalAddress for the address to the
  // register.
  context_data_first_ptr_ = memory()->SystemHeapAlloc(
      sizeof(XMA_CONTEXT_DATA) * kContextCount, 256, kSystemHeapPhysical);
  context_data_last_ptr_ =
      context_data_first_ptr_ + (sizeof(XMA_CONTEXT_DATA) * kContextCount - 1);
  register_file_[XmaRegister::ContextArrayAddress] =
      memory()->GetPhysicalAddress(context_data_first_ptr_);

  // Setup XMA contexts.
  for (int i = 0; i < kContextCount; ++i) {
    if (cvars::xma_decoder == "fake") {
      contexts_[i] = new XmaContextFake();
    } else if (cvars::xma_decoder == "master") {
      contexts_[i] = new XmaContextMaster();
    } else if (cvars::xma_decoder == "old") {
      contexts_[i] = new XmaContextOld();
    } else if (cvars::xma_decoder == "new") {
      contexts_[i] = new XmaContextNew();
    } else {
      contexts_[i] = new XmaContextNew();
    }

    uint32_t guest_ptr = context_data_first_ptr_ + i * sizeof(XMA_CONTEXT_DATA);
    if (contexts_[i]->Setup(i, memory(), guest_ptr)) {
      assert_always();
    }
  }
  register_file_[XmaRegister::NextContextIndex] = 1;
  context_bitmap_.Resize(kContextCount);

  worker_running_ = true;
  work_event_ = xe::threading::Event::CreateAutoResetEvent(false);
  assert_not_null(work_event_);
  worker_thread_ =
      kernel::object_ref<kernel::XHostThread>(new kernel::XHostThread(
          kernel_state, 128 * 1024, 0,
          [this]() {
            if (cvars::use_dedicated_xma_thread) {
              WorkerThreadMain();
            }
            return 0;
          },
          kernel_state
              ->GetIdleProcess()));  // this one doesnt need any process
                                     // actually. never calls any guest code
  worker_thread_->set_name("XMA Decoder");
  worker_thread_->set_can_debugger_suspend(true);
  worker_thread_->Create();

  return X_STATUS_SUCCESS;
}

void XmaDecoder::WorkerThreadMain() {
  while (worker_running_) {
    // Okay, let's loop through XMA contexts to find ones we need to decode!
    bool did_work = false;
    for (uint32_t n = 0; n < kContextCount; n++) {
      bool worked = contexts_[n]->Work();
      if (worked) {
        contexts_[n]->SignalWorkDone();
      }
      did_work = did_work || worked;
    }

    if (paused_) {
      pause_fence_.Signal();
      resume_fence_.Wait();
    }

    if (did_work) {
      continue;
    }
    xe::threading::Wait(work_event_.get(), false);
  }
}

void XmaDecoder::Shutdown() {
  worker_running_ = false;

  if (work_event_) {
    work_event_->Set();
  }

  if (paused_) {
    Resume();
  }

  if (worker_thread_) {
    // Wait for work thread.
    xe::threading::Wait(worker_thread_->thread(), false);
    worker_thread_.reset();
  }

  if (context_data_first_ptr_) {
    memory()->SystemHeapFree(context_data_first_ptr_);
  }

  context_data_first_ptr_ = 0;
  context_data_last_ptr_ = 0;
}

int XmaDecoder::GetContextId(uint32_t guest_ptr) {
  static_assert_size(XMA_CONTEXT_DATA, 64);
  if (guest_ptr < context_data_first_ptr_ ||
      guest_ptr > context_data_last_ptr_) {
    return -1;
  }
  assert_zero(guest_ptr & 0x3F);
  return (guest_ptr - context_data_first_ptr_) >> 6;
}

uint32_t XmaDecoder::AllocateContext() {
  size_t index = context_bitmap_.Acquire();
  if (index == -1) {
    // Out of contexts.
    return 0;
  }

  XmaContext& context = *contexts_[index];
  assert_false(context.is_allocated());
  context.set_is_allocated(true);
  // [media-diag] Correlate the poisoned kick ctx_id (0xD580) with the context
  // pointer handed to the guest. The guest must convert ptr->index to kick
  // ((ptr-base)/0x40); if the base-subtract is dropped/miscompiled on arm64 it
  // gets ptr/0x40 instead. If guest_ptr/0x40 == 0xD580, that's the bug class.
  if (cvars::media_diag) {
    const uint32_t gp = context.guest_ptr();
    XELOGW(
        "[media-diag] xma.AllocateContext index={} guest_ptr={:08X} base={:08X} "
        "gp/0x40={:08X} (gp-base)/0x40={} gp&FFFF={:04X}",
        static_cast<uint32_t>(index), gp, context_data_first_ptr_, gp / 0x40,
        (gp - context_data_first_ptr_) / 0x40, gp & 0xFFFF);
  }
  return context.guest_ptr();
}

void XmaDecoder::ReleaseContext(uint32_t guest_ptr) {
  auto context_id = GetContextId(guest_ptr);
  assert_true(context_id >= 0);

  XmaContext& context = *contexts_[context_id];
  assert_true(context.is_allocated());
  context.Release();
  context_bitmap_.Release(context_id);
}

bool XmaDecoder::BlockOnContext(uint32_t guest_ptr, bool poll) {
  auto context_id = GetContextId(guest_ptr);
  assert_true(context_id >= 0);

  XmaContext& context = *contexts_[context_id];
  return context.Block(poll);
}

uint32_t XmaDecoder::ReadRegister(uint32_t addr) {
  auto r = (addr & 0xFFFF) / 4;

  assert_true(r < XmaRegisterFile::kRegisterCount);

  if (cvars::media_diag) {
    // A spin-wait polls via reads. Capture which register the guest keeps
    // reading and the value we hand back (post byte_swap) so we can see exactly
    // what acknowledgement DOA5 is waiting on before it would issue a kick.
    static std::atomic<uint64_t> s_reads{0};
    uint64_t rd = s_reads.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((rd % 1000) == 0) {
      XELOGI("[media-diag] xma.reg_read reads={} r={:04X} ret={:08X}", rd, r,
             xe::byte_swap(register_file_[r]));
    }
  }

  switch (r) {
    case XmaRegister::ContextArrayAddress:
      break;
    case XmaRegister::CurrentContextIndex: {
      // 0606h (1818h) is rotating context processing # set to hardware ID of
      // context being processed.
      // If bit 200h is set, the locking code will possibly collide on hardware
      // IDs and error out, so we should never set it (I think?).
      uint32_t& current_context_index =
          register_file_[XmaRegister::CurrentContextIndex];
      uint32_t& next_context_index =
          register_file_[XmaRegister::NextContextIndex];
      // To prevent games from seeing a stuck XMA context, return a rotating
      // number.
      current_context_index = next_context_index;
      next_context_index = (next_context_index + 1) % kContextCount;
      break;
    }
    default:
      const auto register_info = register_file_.GetRegisterInfo(r);
      if (register_info) {
        XELOGW("XMA: Read from unhandled register ({:04X}, {})", r,
               register_info->name);
      } else {
        XELOGW("XMA: Read from unknown register ({:04X})", r);
      }
      break;
  }

  return xe::byte_swap(register_file_[r]);
}

void XmaDecoder::WriteRegister(uint32_t addr, uint32_t value) {
  SCOPE_profile_cpu_f("apu");

  uint32_t r = (addr & 0xFFFF) / 4;
  value = xe::byte_swap(value);

  assert_true(r < XmaRegisterFile::kRegisterCount);
  register_file_[r] = value;

  // [media-diag] 0xCFC-writer trace. DOA5 gameplay floods this XMA reg and never
  // kicks on arm64 (Intel kicks normally) → gameplay audio is final-mix zeros.
  // The MMIO write callback is passed a null ppc_context (all dispatch paths use
  // write(nullptr,...)), so read the FAULTING GUEST THREAD's context for the
  // writer LR. Identifies who floods 0xCFC: a misrouted XMA kick (a64 codegen) or
  // unrelated side traffic. Diagnostic only; does NOT handle/alias 0xCFC.
  if (cvars::media_diag && r == 0xCFC) {
    auto* th = kernel::XThread::GetCurrentThread();
    auto* ctx =
        (th && th->thread_state()) ? th->thread_state()->context() : nullptr;
    if (ctx) {
      static std::atomic<int> s_fires{0};
      static std::atomic<int> s_n_lrs{0};
      static uint32_t s_seen_lrs[8] = {0};
      const uint32_t lr = static_cast<uint32_t>(ctx->lr);
      const int fires = s_fires.fetch_add(1, std::memory_order_relaxed);
      bool known = false;
      const int n = s_n_lrs.load(std::memory_order_relaxed);
      for (int i = 0; i < n && i < 8; ++i) {
        if (s_seen_lrs[i] == lr) {
          known = true;
          break;
        }
      }
      if (!known && n < 8) {
        s_seen_lrs[n] = lr;
        s_n_lrs.store(n + 1, std::memory_order_relaxed);
        XELOGW(
            "[media-diag] xma.cfc-writer NEW-LR lr={:08X} r1={:08X} r3={:08X} "
            "r4={:08X} r5={:08X} r6={:08X} r9={:08X} r11={:08X} r12={:08X} "
            "val={:08X} (#{})",
            lr, static_cast<uint32_t>(ctx->r[1]),
            static_cast<uint32_t>(ctx->r[3]), static_cast<uint32_t>(ctx->r[4]),
            static_cast<uint32_t>(ctx->r[5]), static_cast<uint32_t>(ctx->r[6]),
            static_cast<uint32_t>(ctx->r[9]), static_cast<uint32_t>(ctx->r[11]),
            static_cast<uint32_t>(ctx->r[12]), value, fires);
        // Disassemble the actual writer function (0x8297E0E8, called at lr-4) and
        // the sibling called right after (0x829769F0 at lr+4 — a likely real
        // XMA/audio work/kick path). lr is only the caller return site; the store
        // + address math live in 0x8297E0E8. Dump each forward to blr / +0x180.
        // (Addresses from the prior run; DOA5 guest code is at fixed addresses.)
        StringBuffer sb;
        auto dump_fn = [&](uint32_t start, const char* tag) {
          for (uint32_t pc = start; pc < start + 0x180; pc += 4) {
            const uint8_t* p = memory()->TranslateVirtual<const uint8_t*>(pc);
            const uint32_t code = (uint32_t(p[0]) << 24) |
                                  (uint32_t(p[1]) << 16) |
                                  (uint32_t(p[2]) << 8) | uint32_t(p[3]);
            sb.Reset();
            cpu::ppc::DisasmPPC(pc, code, &sb);
            XELOGW("[media-diag] {} {:08X}: {:08X}  {}", tag, pc, code,
                   sb.to_string_view());
            if (code == 0x4E800020u) break;  // blr (bclr 20,0) = function end
          }
        };
        dump_fn(0x8297E0E8u, "cfc.fn");   // the 0xCFC writer
        dump_fn(0x829769F0u, "cfc.sib");  // sibling called right after
        std::string callers;
        char cb[24];
        const uint32_t sp = static_cast<uint32_t>(ctx->r[1]);
        for (uint32_t off = 0; off <= 0x140; off += 4) {
          const uint8_t* q =
              memory()->TranslateVirtual<const uint8_t*>(sp + off);
          const uint32_t w = (uint32_t(q[0]) << 24) | (uint32_t(q[1]) << 16) |
                             (uint32_t(q[2]) << 8) | uint32_t(q[3]);
          if (w >= 0x82000000u && w < 0x83000000u) {
            snprintf(cb, sizeof(cb), "+%X:%08X ", off, w);
            callers += cb;
          }
        }
        XELOGW("[media-diag] cfc.callers sp={:08X} [{}]", sp, callers);

        // Context-struct dump — find who poisoned ctx_id (entry[0x50]). r3 = the
        // audio manager (=0xE8077000); [r3+8] = context-entry array (0x60 stride);
        // entry[0x50] (BE halfword) = the context id that derives the kick reg
        // (valid 0..319 -> reg 0x650+id/32; 0xD580 -> reg 0xCFC = the misdirect).
        auto rd32 = [&](uint32_t a) -> uint32_t {
          const uint8_t* m = memory()->TranslateVirtual<const uint8_t*>(a);
          return (uint32_t(m[0]) << 24) | (uint32_t(m[1]) << 16) |
                 (uint32_t(m[2]) << 8) | uint32_t(m[3]);
        };
        auto rd16 = [&](uint32_t a) -> uint16_t {
          const uint8_t* m = memory()->TranslateVirtual<const uint8_t*>(a);
          return static_cast<uint16_t>((uint16_t(m[0]) << 8) | m[1]);
        };
        const uint32_t mgr = static_cast<uint32_t>(ctx->r[3]);
        const uint32_t base = rd32(mgr + 8);
        XELOGW(
            "[media-diag] cfc.mgr r3={:08X} [0]={:08X} [4]={:08X} [8]={:08X} "
            "[C]={:08X} [10]={:08X}",
            mgr, rd32(mgr + 0), rd32(mgr + 4), base, rd32(mgr + 0xC),
            rd32(mgr + 0x10));
        for (uint32_t i = 0; i < 8; ++i) {
          const uint32_t e = base + i * 0x60;
          const uint16_t id = rd16(e + 0x50);
          XELOGW(
              "[media-diag] cfc.entry[{}] @{:08X} +40={:08X} +44={:08X} "
              "+48={:08X} +4C={:08X} +50(id)={:04X} grp={} kickreg={:X}",
              i, e, rd32(e + 0x40), rd32(e + 0x44), rd32(e + 0x48),
              rd32(e + 0x4C), id, (id >> 5), 0x650 + (id >> 5));
        }
      } else if ((fires % 50000) == 0) {
        XELOGW("[media-diag] xma.cfc-writer count={} latest_lr={:08X}", fires,
               lr);
      }
    } else {
      static std::atomic<bool> s_warned{false};
      if (!s_warned.exchange(true)) {
        XELOGW("[media-diag] xma.cfc-writer: no current-thread context");
      }
    }
  }

  if (cvars::media_diag) {
    // Categorize every XMA register the guest touches so we can tell whether it
    // performs full XMA setup (context-array address + lock/clear) and merely
    // never kicks, or never really drives XMA at all. Counters are per-category;
    // a periodic dump shows the whole pattern, and any kick logs immediately.
    static std::atomic<uint64_t> s_total{0}, s_ctxarr{0}, s_curnext{0},
        s_kick{0}, s_lock{0}, s_clear{0}, s_other{0};
    const bool is_kick =
        (r >= XmaRegister::Context0Kick && r <= XmaRegister::Context9Kick);
    const bool is_lock =
        (r >= XmaRegister::Context0Lock && r <= XmaRegister::Context9Lock);
    const bool is_clear =
        (r >= XmaRegister::Context0Clear && r <= XmaRegister::Context9Clear);
    const bool is_ctxarr = (r == XmaRegister::ContextArrayAddress);
    const bool is_curnext = (r == XmaRegister::CurrentContextIndex ||
                             r == XmaRegister::NextContextIndex);
    if (is_kick) s_kick.fetch_add(1, std::memory_order_relaxed);
    else if (is_lock) s_lock.fetch_add(1, std::memory_order_relaxed);
    else if (is_clear) s_clear.fetch_add(1, std::memory_order_relaxed);
    else if (is_ctxarr) s_ctxarr.fetch_add(1, std::memory_order_relaxed);
    else if (is_curnext) s_curnext.fetch_add(1, std::memory_order_relaxed);
    else s_other.fetch_add(1, std::memory_order_relaxed);
    uint64_t t = s_total.fetch_add(1, std::memory_order_relaxed) + 1;
    if (is_kick || is_lock || is_clear || is_ctxarr || (t % 1000) == 0) {
      XELOGI(
          "[media-diag] xma.reg_write total={} ctxarr={} curnext={} kick={} "
          "lock={} clear={} other={} | this r={} val={:08X}{}",
          t, s_ctxarr.load(std::memory_order_relaxed),
          s_curnext.load(std::memory_order_relaxed),
          s_kick.load(std::memory_order_relaxed),
          s_lock.load(std::memory_order_relaxed),
          s_clear.load(std::memory_order_relaxed),
          s_other.load(std::memory_order_relaxed), r, value,
          is_kick ? " <KICK>" : (is_lock ? " <LOCK>" : (is_clear ? " <CLEAR>"
              : (is_ctxarr ? " <CTXARR>" : ""))));
    }
  }

  if (r >= XmaRegister::Context0Kick && r <= XmaRegister::Context9Kick) {
    // Context kick command.
    // This will kick off the given hardware contexts.
    // Basically, this kicks the SPU and says "hey, decode that audio!"
    // XMAEnableContext

    // The context ID is a bit in the range of the entire context array.
    const uint32_t base_context_id = (r - XmaRegister::Context0Kick) * 32;
    const uint32_t kicked_value = value;
    while (value) {
      const uint32_t context_id = base_context_id + std::countr_zero(value);
      auto& context = *contexts_[context_id];
      context.Enable();
      if (!cvars::use_dedicated_xma_thread) {
        context.Work();
      }
      value &= value - 1;
    }
    // Signal the decoder thread to start processing.
    work_event_->SetBoostPriority();
    if (cvars::use_dedicated_xma_thread) {
      // Block until the worker finishes, so the game sees updated context data.
      uint32_t remaining = kicked_value;
      while (remaining) {
        const uint32_t context_id =
            base_context_id + std::countr_zero(remaining);
        contexts_[context_id]->WaitForWorkDone();
        remaining &= remaining - 1;
      }
    }
  } else if (r >= XmaRegister::Context0Lock && r <= XmaRegister::Context9Lock) {
    // Context lock command.
    // This requests a lock by flagging the context.
    // XMADisableContext
    const uint32_t base_context_id = (r - XmaRegister::Context0Lock) * 32;
    while (value) {
      const uint32_t context_id = base_context_id + std::countr_zero(value);
      auto& context = *contexts_[context_id];
      context.Disable();
      // Ensure the worker isn't mid-processing this context.
      context.Block(false);
      value &= value - 1;
    }
  } else if (r >= XmaRegister::Context0Clear &&
             r <= XmaRegister::Context9Clear) {
    // Context clear command.
    // This will reset the given hardware contexts.
    const uint32_t base_context_id = (r - XmaRegister::Context0Clear) * 32;
    while (value) {
      const uint32_t context_id = base_context_id + std::countr_zero(value);
      auto& context = *contexts_[context_id];
      context.Clear();
      value &= value - 1;
    }
  } else {
    // 0601h (1804h) is written to with 0x02000000 and 0x03000000 around a lock
    // operation
    switch (r) {
      case 0x601:
        break;
      default: {
        const auto register_info = register_file_.GetRegisterInfo(r);
        if (register_info) {
          XELOGW("XMA: Write to unhandled register ({:04X}, {}): {:08X}", r,
                 register_info->name, value);
        } else {
          XELOGW("XMA: Write to unknown register ({:04X}): {:08X}", r, value);
        }
        break;
      }
#pragma warning(suppress : 4065)
    }
  }
}

void XmaDecoder::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  pause_fence_.Wait();
}

void XmaDecoder::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  resume_fence_.Signal();
}

}  // namespace apu
}  // namespace xe
