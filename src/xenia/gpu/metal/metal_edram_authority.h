/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_METAL_METAL_EDRAM_AUTHORITY_H_
#define XENIA_GPU_METAL_METAL_EDRAM_AUTHORITY_H_

#include <array>
#include <cstdint>
#include <vector>

#include "xenia/gpu/xenos.h"

namespace xe {
namespace gpu {
namespace metal {

// Submission-ordered authority for the physical EDRAM tile space. This class
// deliberately contains no Metal objects: it decides which exact overlapping
// intervals need attachment->canonical packing or canonical->attachment
// materialization; MetalRenderTargetCache encodes those operations in order.
//
// Only one clean attachment cache is remembered per tile. Forgetting an older
// clean cache may cause an extra materialize, but can never return stale data.
class MetalEdramAuthorityTracker {
 public:
  using AttachmentId = uint64_t;

  enum class Location : uint8_t {
    kUnknown,
    kCanonical,
    kAttachment,
  };

  enum class SyncKind : uint8_t {
    kPackToCanonical,
    kMaterializeFromCanonical,
  };

  struct TileState {
    Location location = Location::kUnknown;
    AttachmentId authority_attachment = 0;
    AttachmentId clean_attachment = 0;
    uint64_t submission_order = 0;
  };

  struct SyncOperation {
    SyncKind kind;
    AttachmentId attachment;
    uint32_t start_tiles;
    uint32_t end_tiles;
    uint64_t producer_order;
    uint64_t consumer_order;
  };

  void ResetUnknown() { tiles_.fill(TileState{}); }

  void MarkCanonical(uint32_t start_tiles, uint32_t end_tiles,
                     uint64_t submission_order) {
    ClampRange(start_tiles, end_tiles);
    for (uint32_t tile = start_tiles; tile < end_tiles; ++tile) {
      TileState& state = tiles_[tile];
      if (submission_order < state.submission_order) {
        continue;
      }
      state.location = Location::kCanonical;
      state.authority_attachment = 0;
      // A canonical write invalidates every previously materialized view.
      state.clean_attachment = 0;
      state.submission_order = submission_order;
    }
  }

  void MarkAttachment(AttachmentId attachment, uint32_t start_tiles,
                      uint32_t end_tiles, uint64_t submission_order) {
    if (!attachment) {
      return;
    }
    ClampRange(start_tiles, end_tiles);
    for (uint32_t tile = start_tiles; tile < end_tiles; ++tile) {
      TileState& state = tiles_[tile];
      if (submission_order < state.submission_order) {
        continue;
      }
      state.location = Location::kAttachment;
      state.authority_attachment = attachment;
      state.clean_attachment = attachment;
      state.submission_order = submission_order;
    }
  }

  // Makes canonical current for the requested interval. Only tiles whose
  // newest value resides in an attachment generate pack operations; unknown
  // and already-canonical tiles are untouched.
  std::vector<SyncOperation> SynchronizeCanonical(
      uint32_t start_tiles, uint32_t end_tiles, uint64_t consumer_order) {
    return SynchronizeCanonicalImpl(start_tiles, end_tiles, consumer_order,
                                    /*compatible_attachment=*/0);
  }

  // Makes `attachment` a clean working view for the requested interval. Dirty
  // values in any other attachment are packed first, then only canonical tiles
  // not already cached by this attachment are materialized. Returned operation
  // order is therefore directly encodable in one Metal command stream.
  std::vector<SyncOperation> SynchronizeAttachment(
      AttachmentId attachment, uint32_t start_tiles, uint32_t end_tiles,
      uint64_t consumer_order) {
    if (!attachment) {
      return {};
    }
    ClampRange(start_tiles, end_tiles);
    std::vector<SyncOperation> operations = SynchronizeCanonicalImpl(
        start_tiles, end_tiles, consumer_order, attachment);
    for (uint32_t tile = start_tiles; tile < end_tiles;) {
      TileState& state = tiles_[tile];
      if (state.location != Location::kCanonical ||
          state.clean_attachment == attachment ||
          state.submission_order > consumer_order) {
        ++tile;
        continue;
      }
      uint64_t producer_order = state.submission_order;
      uint32_t operation_end = tile + 1;
      while (operation_end < end_tiles) {
        const TileState& next = tiles_[operation_end];
        if (next.location != Location::kCanonical ||
            next.clean_attachment == attachment ||
            next.submission_order > consumer_order) {
          break;
        }
        producer_order =
            producer_order > next.submission_order ? producer_order
                                                   : next.submission_order;
        ++operation_end;
      }
      operations.push_back({SyncKind::kMaterializeFromCanonical, attachment,
                            tile, operation_end, producer_order,
                            consumer_order});
      for (uint32_t materialized = tile; materialized < operation_end;
           ++materialized) {
        TileState& materialized_state = tiles_[materialized];
        materialized_state.clean_attachment = attachment;
        materialized_state.submission_order = consumer_order;
      }
      tile = operation_end;
    }
    return operations;
  }

  const TileState& tile(uint32_t index) const { return tiles_[index]; }

 private:
  std::vector<SyncOperation> SynchronizeCanonicalImpl(
      uint32_t start_tiles, uint32_t end_tiles, uint64_t consumer_order,
      AttachmentId compatible_attachment) {
    ClampRange(start_tiles, end_tiles);
    std::vector<SyncOperation> operations;
    for (uint32_t tile = start_tiles; tile < end_tiles;) {
      TileState& state = tiles_[tile];
      if (state.location != Location::kAttachment ||
          state.authority_attachment == compatible_attachment ||
          state.submission_order > consumer_order) {
        ++tile;
        continue;
      }
      const AttachmentId source = state.authority_attachment;
      uint64_t producer_order = state.submission_order;
      uint32_t operation_end = tile + 1;
      while (operation_end < end_tiles) {
        const TileState& next = tiles_[operation_end];
        if (next.location != Location::kAttachment ||
            next.authority_attachment != source ||
            next.submission_order > consumer_order) {
          break;
        }
        producer_order =
            producer_order > next.submission_order ? producer_order
                                                   : next.submission_order;
        ++operation_end;
      }
      operations.push_back({SyncKind::kPackToCanonical, source, tile,
                            operation_end, producer_order, consumer_order});
      for (uint32_t packed = tile; packed < operation_end; ++packed) {
        TileState& packed_state = tiles_[packed];
        packed_state.location = Location::kCanonical;
        packed_state.authority_attachment = 0;
        packed_state.clean_attachment = source;
        packed_state.submission_order = consumer_order;
      }
      tile = operation_end;
    }
    return operations;
  }
  static void ClampRange(uint32_t& start_tiles, uint32_t& end_tiles) {
    if (start_tiles > xenos::kEdramTileCount) {
      start_tiles = xenos::kEdramTileCount;
    }
    if (end_tiles > xenos::kEdramTileCount) {
      end_tiles = xenos::kEdramTileCount;
    }
    if (end_tiles < start_tiles) {
      end_tiles = start_tiles;
    }
  }

  std::array<TileState, xenos::kEdramTileCount> tiles_{};
};

}  // namespace metal
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_METAL_METAL_EDRAM_AUTHORITY_H_
