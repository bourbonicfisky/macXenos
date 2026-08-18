/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_TRACE_PLAYER_H_
#define XENIA_GPU_TRACE_PLAYER_H_

#include <atomic>
#include <memory>
#include <string>

#include "xenia/base/threading.h"
#include "xenia/gpu/trace_protocol.h"
#include "xenia/gpu/trace_reader.h"

namespace xe {
namespace gpu {

class GraphicsSystem;

// Binds the guest-RAM rollback journal to the command processor's pre-write
// notify surface; defined in trace_player.cc (fix-replay-guest-ram-rollback).
struct TracePlayerRamJournalBinding;

enum class TracePlaybackMode {
  kUntilEnd,
  kBreakOnSwap,
};

class TracePlayer : public TraceReader {
 public:
  TracePlayer(GraphicsSystem* graphics_system);
  ~TracePlayer();

  GraphicsSystem* graphics_system() const { return graphics_system_; }
  int current_frame_index() const { return current_frame_index_; }
  int current_command_index() const { return current_command_index_; }
  bool is_playing_trace() const { return playing_trace_; }
  const Frame* current_frame() const;

  // Only valid if playing_trace is true.
  // Scalar from 0-10000
  uint32_t playback_percent() const { return playback_percent_; }

  void SeekFrame(int target_frame);
  void SeekCommand(int target_command);
  // Re-plays the current frame from its start unconditionally (SeekFrame /
  // SeekCommand no-op when already positioned — a repeat caller would wedge
  // in WaitOnPlayback). Multi-pass replay builds cross-frame state (RT/key
  // persistence, temporal history) that single-pass replay cannot.
  //
  // clear_caches requests a backend cache clear before the replay, through the
  // same path a trace-viewer rewind uses (playback to an earlier point
  // invalidates previously cached content). It is the only scriptable way to
  // reach the ClearCaches contract, which otherwise needs an interactive F5.
  void ReplayCurrentFrame(bool clear_caches = false);

  void WaitOnPlayback();

 private:
  // frame_initial: this playback starts at the frame's start_ptr, whose
  // leading commands are the authoritative restores (kMemoryRead /
  // kEdramSnapshot). Such playbacks retire the backend's derived replay state
  // first (CommandProcessor::ResetDerivedReplayState) so restoration
  // re-materializes from authoritative content only. Incremental forward
  // steps (SeekCommand within a frame) deliberately continue from live state
  // and must NOT pass it.
  void PlayTrace(const uint8_t* trace_data, size_t trace_size,
                 TracePlaybackMode playback_mode, bool clear_caches,
                 bool frame_initial);
  void PlayTraceOnThread(const uint8_t* trace_data, size_t trace_size,
                         TracePlaybackMode playback_mode, bool clear_caches,
                         bool frame_initial);

  GraphicsSystem* graphics_system_;
  int current_frame_index_;
  int current_command_index_;
  bool playing_trace_ = false;
  std::atomic<uint32_t> playback_percent_ = {0};
  std::unique_ptr<xe::threading::Event> playback_event_;
  // Guest-RAM rollback for replay loops (fix-replay-guest-ram-rollback):
  // installed on the command processor at the first playback so loop 0's
  // GPU-derived RAM writes capture their pre-frame bytes; restored at every
  // frame-initial playback boundary. Created and used on the CP thread only.
  std::unique_ptr<TracePlayerRamJournalBinding> ram_journal_;
  int ram_journal_frame_index_ = -1;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_TRACE_PLAYER_H_
