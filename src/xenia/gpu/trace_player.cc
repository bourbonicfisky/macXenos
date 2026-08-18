/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_player.h"

#include "xenia/base/logging.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/guest_ram_write_journal.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {

// Adapter from the command processor's pre-write notify surface to the
// journal's capture (fix-replay-guest-ram-rollback 1.2/2.1). The translate
// hook clamps to the 512 MiB guest physical space; an out-of-space range is
// skipped (and counted) rather than faulting.
struct TracePlayerRamJournalBinding final
    : public CommandProcessor::GuestRamWriteJournal {
  explicit TracePlayerRamJournalBinding(Memory* memory)
      : journal([memory](uint32_t base, uint32_t length) -> uint8_t* {
          if (uint64_t(base) + length > 0x20000000ull) {
            return nullptr;
          }
          return memory->TranslatePhysical(base);
        }) {}

  void OnBeforeGuestRamWrite(uint32_t base, uint32_t length) override {
    journal.CaptureBeforeWrite(base, length);
  }

  gpu::GuestRamWriteJournal journal;
};

TracePlayer::TracePlayer(GraphicsSystem* graphics_system)
    : graphics_system_(graphics_system),
      current_frame_index_(0),
      current_command_index_(-1) {
  // Need to allocate all of physical memory so that we can write to it during
  // playback. The 64 KB page heap is larger, covers the entire physical memory,
  // so it is used instead of the 4 KB page one.
  auto heap = graphics_system_->memory()->LookupHeapByType(true, 64 * 1024);
  heap->AllocFixed(heap->heap_base(), heap->heap_size(), heap->page_size(),
                   kMemoryAllocationReserve | kMemoryAllocationCommit,
                   kMemoryProtectRead | kMemoryProtectWrite);

  playback_event_ = xe::threading::Event::CreateAutoResetEvent(false);
  assert_not_null(playback_event_);
}

TracePlayer::~TracePlayer() {
  // Best-effort uninstall so a command processor outliving this player never
  // notifies a dangling journal. (Notifications only fire during playback,
  // which this player drives, so this is belt-and-suspenders.)
  if (ram_journal_ && graphics_system_ &&
      graphics_system_->command_processor()) {
    graphics_system_->command_processor()->SetGuestRamWriteJournal(nullptr);
  }
}

const TraceReader::Frame* TracePlayer::current_frame() const {
  if (current_frame_index_ >= frame_count()) {
    return nullptr;
  }
  return frame(current_frame_index_);
}

void TracePlayer::SeekFrame(int target_frame) {
  if (current_frame_index_ == target_frame) {
    return;
  }
  current_frame_index_ = target_frame;
  auto frame = current_frame();
  current_command_index_ = int(frame->commands.size()) - 1;

  assert_true(frame->start_ptr <= frame->end_ptr);
  // NOT frame_initial: stepping the viewer to another frame continues from
  // the state the preceding frames built (a frame's restore prologue only
  // re-delivers what the capture-time GPU read; cross-frame derived state is
  // the live-continuation semantic here, unlike a same-frame replay loop).
  PlayTrace(frame->start_ptr, frame->end_ptr - frame->start_ptr,
            TracePlaybackMode::kBreakOnSwap, false, /*frame_initial=*/false);
}

void TracePlayer::SeekCommand(int target_command) {
  if (current_command_index_ == target_command) {
    return;
  }
  int previous_command_index = current_command_index_;
  current_command_index_ = target_command;
  if (current_command_index_ == -1) {
    return;
  }
  auto frame = current_frame();
  const auto& command = frame->commands[target_command];
  assert_true(frame->start_ptr <= command.end_ptr);
  if (previous_command_index != -1 && target_command > previous_command_index) {
    // Seek forward: incremental continuation from live mid-frame state --
    // never frame_initial.
    const auto& previous_command = frame->commands[previous_command_index];
    PlayTrace(previous_command.end_ptr,
              command.end_ptr - previous_command.end_ptr,
              TracePlaybackMode::kBreakOnSwap, false,
              /*frame_initial=*/false);
  } else {
    // Full playback from frame start: the leading commands are the frame's
    // authoritative restores, so derived replay state retires first.
    PlayTrace(frame->start_ptr, command.end_ptr - frame->start_ptr,
              TracePlaybackMode::kBreakOnSwap, true, /*frame_initial=*/true);
  }
}

void TracePlayer::ReplayCurrentFrame(bool clear_caches) {
  auto frame = current_frame();
  if (!frame) {
    return;
  }
  current_command_index_ = int(frame->commands.size()) - 1;
  PlayTrace(frame->start_ptr, frame->end_ptr - frame->start_ptr,
            TracePlaybackMode::kBreakOnSwap, clear_caches,
            /*frame_initial=*/true);
}

void TracePlayer::WaitOnPlayback() {
  xe::threading::Wait(playback_event_.get(), true);
}

void TracePlayer::PlayTrace(const uint8_t* trace_data, size_t trace_size,
                            TracePlaybackMode playback_mode,
                            bool clear_caches, bool frame_initial) {
  playing_trace_ = true;
  graphics_system_->command_processor()->CallInThread([=, this]() {
    PlayTraceOnThread(trace_data, trace_size, playback_mode, clear_caches,
                      frame_initial);
  });
}

void TracePlayer::PlayTraceOnThread(const uint8_t* trace_data,
                                    size_t trace_size,
                                    TracePlaybackMode playback_mode,
                                    bool clear_caches, bool frame_initial) {
  auto memory = graphics_system_->memory();
  auto command_processor = graphics_system_->command_processor();

  // Declare replay context (on the command-processor thread, before any
  // playback command) so backends may assert their replay-only seams are
  // never reached from a live title path.
  command_processor->SetTracePlaybackContext();

  // Guest-RAM rollback journal (fix-replay-guest-ram-rollback D2): installed
  // before ANY playback command so the first playback's GPU-derived RAM
  // writes (resolve writeback, memexport convergence, ZPD reports, readback
  // resolve) capture their pre-frame bytes at first write. A backend that
  // reports no writes never notifies, so this is structurally inert there.
  if (!ram_journal_) {
    ram_journal_ = std::make_unique<TracePlayerRamJournalBinding>(memory);
    command_processor->SetGuestRamWriteJournal(ram_journal_.get());
  }
  if (ram_journal_frame_index_ != current_frame_index_) {
    // Stepping to a different frame is live continuation (see SeekFrame): the
    // captured pre-images belong to the old frame and must not be replayed
    // into the new one.
    ram_journal_->journal.Clear();
    ram_journal_frame_index_ = current_frame_index_;
  }

  if (frame_initial) {
    // Quiescent replay-boundary retirement, BEFORE the frame's restore
    // commands re-deliver authoritative RAM + EDRAM: nothing GPU-derived
    // (resolve snapshots, bridge registrations, RT-pool lineage) may survive
    // into this playback (fix-metal-trace-replay-state-reset design D2).
    // Unconditional -- unlike clear_caches, which is a scripted-gate request
    // for the DEFERRED live-semantics cache clear.
    command_processor->ResetDerivedReplayState();
    // Restore the guest-RAM ranges earlier playbacks of this frame wrote
    // (design D3): same replay boundary as the derived-state reset, before
    // the trace's own kMemoryRead/kEdramSnapshot restores re-deliver
    // authoritative content. No-op on the first frame-initial playback (the
    // journal is empty until the frame's first GPU-derived write).
    size_t restored_ranges = ram_journal_->journal.RestoreAll();
    if (restored_ranges) {
      XELOGI(
          "TracePlayer: guest-RAM rollback restored {} ranges / {} bytes "
          "(failed captures {}, failed restores {})",
          restored_ranges, ram_journal_->journal.captured_bytes(),
          ram_journal_->journal.failed_captures(),
          ram_journal_->journal.failed_restores());
    }
  }
  if (clear_caches) {
    command_processor->ClearCaches();
  }

  // Stable draw indices for the replay_draw_first/last bisection gate.
  command_processor->ResetTraceDrawIndex();

  playback_percent_ = 0;
  auto trace_end = trace_data + trace_size;

  playing_trace_ = true;
  auto trace_ptr = trace_data;
  bool pending_break = false;
  const PacketStartCommand* pending_packet = nullptr;
  while (trace_ptr < trace_data + trace_size) {
    playback_percent_ = uint32_t(
        (float(trace_ptr - trace_data) / float(trace_end - trace_data)) *
        10000);

    auto type = static_cast<TraceCommandType>(xe::load<uint32_t>(trace_ptr));
    switch (type) {
      case TraceCommandType::kPrimaryBufferStart: {
        auto cmd =
            reinterpret_cast<const PrimaryBufferStartCommand*>(trace_ptr);
        //
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        break;
      }
      case TraceCommandType::kPrimaryBufferEnd: {
        auto cmd = reinterpret_cast<const PrimaryBufferEndCommand*>(trace_ptr);
        //
        trace_ptr += sizeof(*cmd);
        break;
      }
      case TraceCommandType::kIndirectBufferStart: {
        auto cmd =
            reinterpret_cast<const IndirectBufferStartCommand*>(trace_ptr);
        //
        trace_ptr += sizeof(*cmd) + cmd->count * 4;
        break;
      }
      case TraceCommandType::kIndirectBufferEnd: {
        auto cmd = reinterpret_cast<const IndirectBufferEndCommand*>(trace_ptr);
        //
        trace_ptr += sizeof(*cmd);
        break;
      }
      case TraceCommandType::kPacketStart: {
        auto cmd = reinterpret_cast<const PacketStartCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        std::memcpy(memory->TranslatePhysical(cmd->base_ptr), trace_ptr,
                    cmd->count * 4);
        trace_ptr += cmd->count * 4;
        pending_packet = cmd;
        break;
      }
      case TraceCommandType::kPacketEnd: {
        auto cmd = reinterpret_cast<const PacketEndCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        if (pending_packet) {
          command_processor->ExecutePacket(pending_packet->base_ptr,
                                           pending_packet->count);
          pending_packet = nullptr;
        }
        if (pending_break) {
          playing_trace_ = false;
          return;
        }
        break;
      }
      case TraceCommandType::kMemoryRead: {
        auto cmd = reinterpret_cast<const MemoryCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        DecompressMemory(cmd->encoding_format, trace_ptr, cmd->encoded_length,
                         memory->TranslatePhysical(cmd->base_ptr),
                         cmd->decoded_length);
        trace_ptr += cmd->encoded_length;
        command_processor->TracePlaybackWroteMemory(cmd->base_ptr,
                                                    cmd->decoded_length);
        // Diagnostic: XE_TRACE_WATCH=<hex phys addr> logs each kMemoryRead
        // restore covering that address (what the hardware ACTUALLY read
        // there at capture time) with a data sample — separates "the trace
        // delivers these bytes" from "our replay corrupted them".
        static const uint64_t watch_addr = [] {
          const char* v = std::getenv("XE_TRACE_WATCH");
          return v ? strtoull(v, nullptr, 16) : 0ull;
        }();
        if (watch_addr && watch_addr >= cmd->base_ptr &&
            watch_addr < cmd->base_ptr + cmd->decoded_length) {
          const uint32_t* w = reinterpret_cast<const uint32_t*>(
              memory->TranslatePhysical(uint32_t(watch_addr)));
          XELOGW(
              "TraceWatch: kMemoryRead [{:08X}+{:X}] covers {:08X}: "
              "{:08X} {:08X} {:08X} {:08X}",
              cmd->base_ptr, cmd->decoded_length, watch_addr, w[0], w[1], w[2],
              w[3]);
        }
        break;
      }
      case TraceCommandType::kMemoryWrite: {
        auto cmd = reinterpret_cast<const MemoryCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        // ?
        // Assuming the command processor will do the same write.
        trace_ptr += cmd->encoded_length;
        break;
      }
      case TraceCommandType::kEdramSnapshot: {
        auto cmd = reinterpret_cast<const EdramSnapshotCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        std::unique_ptr<uint8_t[]> edram_snapshot(
            new uint8_t[xenos::kEdramSizeBytes]);
        DecompressMemory(cmd->encoding_format, trace_ptr, cmd->encoded_length,
                         edram_snapshot.get(), xenos::kEdramSizeBytes);
        trace_ptr += cmd->encoded_length;
        XELOGI("TracePlayer: Restoring EDRAM snapshot ({} encoded bytes)",
               cmd->encoded_length);
        command_processor->RestoreEdramSnapshot(edram_snapshot.get());
        break;
      }
      case TraceCommandType::kEvent: {
        auto cmd = reinterpret_cast<const EventCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        switch (cmd->event_type) {
          case EventCommand::Type::kSwap: {
            // XE_TRACE_WATCH companion: sample guest RAM at the watched
            // address at swap — with readback resolves enabled this shows
            // the REPLAYING backend's effective content (what a consumer
            // pass read), comparable across backends.
            static const uint64_t swap_watch_addr = [] {
              const char* v = std::getenv("XE_TRACE_WATCH");
              return v ? strtoull(v, nullptr, 16) : 0ull;
            }();
            if (swap_watch_addr) {
              uint32_t swap_watch_physical =
                  xenos::CpuToGpu(uint32_t(swap_watch_addr));
              if (!command_processor->PrepareGuestMemoryForHostRead(
                      swap_watch_physical, 4 * sizeof(uint32_t))) {
                XELOGE("TraceWatch: failed to converge swap sample {:08X}",
                       swap_watch_physical);
                playing_trace_ = false;
                playback_event_->Set();
                return;
              }
              const uint32_t* w = reinterpret_cast<const uint32_t*>(
                  memory->TranslatePhysical(swap_watch_physical));
              XELOGW("TraceWatch: at swap {:08X}: {:08X} {:08X} {:08X} {:08X}",
                     swap_watch_physical, w[0], w[1], w[2], w[3]);
            }
            if (playback_mode == TracePlaybackMode::kBreakOnSwap) {
              pending_break = true;
            }
            break;
          }
        }
        break;
      }
      case TraceCommandType::kRegisters: {
        auto cmd = reinterpret_cast<const RegistersCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        std::unique_ptr<uint32_t[]> register_values(
            new uint32_t[cmd->register_count]);
        DecompressMemory(cmd->encoding_format, trace_ptr, cmd->encoded_length,
                         register_values.get(),
                         sizeof(uint32_t) * cmd->register_count);
        trace_ptr += cmd->encoded_length;
        command_processor->RestoreRegisters(
            cmd->first_register, register_values.get(), cmd->register_count,
            cmd->execute_callbacks);
        break;
      }
      case TraceCommandType::kGammaRamp: {
        auto cmd = reinterpret_cast<const GammaRampCommand*>(trace_ptr);
        trace_ptr += sizeof(*cmd);
        std::unique_ptr<uint32_t[]> gamma_ramps(new uint32_t[256 + 3 * 128]);
        DecompressMemory(cmd->encoding_format, trace_ptr, cmd->encoded_length,
                         gamma_ramps.get(), sizeof(uint32_t) * (256 + 3 * 128));
        trace_ptr += cmd->encoded_length;
        command_processor->RestoreGammaRamp(
            reinterpret_cast<const reg::DC_LUT_30_COLOR*>(gamma_ramps.get()),
            reinterpret_cast<const reg::DC_LUT_PWL_DATA*>(gamma_ramps.get() +
                                                          256),
            cmd->rw_component);
        break;
      }
    }
  }

  // XE_TRACE_WATCH companion: sample guest RAM at the watched address after
  // the whole stream has played — with readback resolves enabled this is the
  // REPLAYING backend's effective final content, comparable across backends.
  {
    static const uint64_t end_watch_addr = [] {
      const char* v = std::getenv("XE_TRACE_WATCH");
      return v ? strtoull(v, nullptr, 16) : 0ull;
    }();
    if (end_watch_addr) {
      uint32_t end_watch_physical =
          xenos::CpuToGpu(uint32_t(end_watch_addr));
      if (!command_processor->PrepareGuestMemoryForHostRead(
              end_watch_physical, 4 * sizeof(uint32_t))) {
        XELOGE("TraceWatch: failed to converge end sample {:08X}",
               end_watch_physical);
        playing_trace_ = false;
        playback_event_->Set();
        return;
      }
      const uint32_t* w = reinterpret_cast<const uint32_t*>(
          memory->TranslatePhysical(end_watch_physical));
      XELOGW("TraceWatch: at end {:08X}: {:08X} {:08X} {:08X} {:08X}",
             end_watch_physical, w[0], w[1], w[2], w[3]);
    }
  }

  playing_trace_ = false;

  playback_event_->Set();
}

}  // namespace gpu
}  // namespace xe
